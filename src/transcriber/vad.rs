use parking_lot::Mutex;
use voice_activity_detector::{IteratorExt, LabeledAudio};

use crate::audio::pcm::PcmS16Convertible;
use crate::utils::errors::RibbleWhisperError;

/// A voice activity detector backend for use with [crate::transcriber::realtime_transcriber::RealtimeTranscriber]
/// or [crate::transcriber::offline_transcriber::OfflineTranscriber].
pub trait VAD<T>: Resettable {
    // For realtime VAD to determine pauses, ends of phrases, and to reduce the amount of whisper
    // processing.
    fn voice_detected(&mut self, samples: &[T]) -> bool;
    // For optimizing offline transcription by reducing the amount of audio that whisper needs to process
    fn extract_voiced_frames(&mut self, samples: &[T]) -> Box<[T]>;
}

/// For resetting the state of a voice activity detector backend so that it can be reused
/// for different audio samples.
pub trait Resettable {
    fn reset_session(&mut self);
}

/// Builder for [Silero] that adapts voice_activity_detector's builder
/// and also includes a starting detection probability.
/// The probability threshold can be swapped after building if needed.
///
/// This uses Silero VAD v5 which operates on fixed-sizes only. An 8kHz sample rate will only allow
/// 256-sample windows; A 16kHz sample rate will only allow 512-sample windows.
/// To avoid any truncation/padding which might cause issues for real-time applications,
/// these are restricted based on the sample rate.
///
/// NOTE: On Windows, this may include some telemetry as per: <https://docs.rs/ort/latest/ort/#strategies>
/// Self-hosted ONNX runtime binaries have not yet been implemented and may not be.
/// In the meantime, use [WebRtc] or [Earshot]
/// if telemetry is a concern.
#[derive(Copy, Clone)]
pub struct SileroBuilder {
    sample_rate: SileroSampleRate,
    /// Samples with probabilities higher than this threshold are considered to have voice activity.
    detection_probability_threshold: f32,
    /// Samples with a total voice proportion higher than this threshold are considered to be a
    /// voiced sample.
    voiced_proportion_threshold: f32,
}

// TODO: docstring
#[derive(Default, Copy, Clone)]
pub enum SileroSampleRate {
    R8kHz,
    #[default]
    R16kHz,
}

impl SileroSampleRate {
    fn vad_sample_rate(&self) -> i64 {
        (*self).into()
    }

    fn chunk_size(&self) -> usize {
        match self {
            SileroSampleRate::R8kHz => 256,
            SileroSampleRate::R16kHz => 512,
        }
    }
}

impl From<SileroSampleRate> for i64 {
    fn from(value: SileroSampleRate) -> Self {
        match value {
            SileroSampleRate::R8kHz => 8000,
            SileroSampleRate::R16kHz => 16000,
        }
    }
}

impl SileroBuilder {
    pub fn new() -> Self {
        Self {
            sample_rate: Default::default(),
            detection_probability_threshold: 0.0,
            voiced_proportion_threshold: 0.0,
        }
    }
    /// Set the sample rate.
    pub fn with_sample_rate(mut self, sample_rate: SileroSampleRate) -> Self {
        self.sample_rate = sample_rate;
        self
    }
    /// Set the detection probability threshold. Values that exceed this will be considered "voiced."
    pub fn with_detection_probability_threshold(mut self, probability: f32) -> Self {
        self.detection_probability_threshold = probability;
        self
    }

    /// Set the voiced proportion threshold. If the proportion of frames is greater than this value,
    /// the whole sample will be considered "voiced."
    pub fn with_voiced_proportion_threshold(mut self, voiced_proportion: f32) -> Self {
        self.voiced_proportion_threshold = voiced_proportion;
        self
    }

    /// Builds a Silero VAD backend.
    /// Returns Err when [voice_activity_detector::VoiceActivityDetector]'s builder fails to build.
    /// To ensure this doesn't happen, ensure the sample rate and chunk size are provided
    /// and that the sample rate is no larger than 31.25 times the chunk size.
    pub fn build(self) -> Result<Silero, RibbleWhisperError> {
        voice_activity_detector::VoiceActivityDetector::builder()
            .sample_rate(self.sample_rate.vad_sample_rate())
            .chunk_size(self.sample_rate.chunk_size())
            .build()
            .map(|vad| Silero {
                vad,
                detection_probability_threshold: self.detection_probability_threshold,
                voiced_proportion_threshold: self.voiced_proportion_threshold,
            })
            .map_err(|e| {
                RibbleWhisperError::ParameterError(format!(
                    "Failed to build Silero VAD. Error: {e}",
                ))
            })
    }
}

impl Default for SileroBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Silero VAD backend for use in realtime transcription.
/// Adapts [voice_activity_detector::VoiceActivityDetector] to predict voice activity using Silero.
/// NOTE: On Windows, this may include some telemetry as per: <https://docs.rs/ort/latest/ort/#strategies>
/// Self-hosted ONNX runtime binaries have not yet been implemented and may not be.
/// In the meantime, use [WebRtc] or [Earshot]
/// if telemetry is a concern.
pub struct Silero {
    vad: voice_activity_detector::VoiceActivityDetector,
    /// Samples with probabilities higher than this threshold are considered to have voice activity.
    detection_probability_threshold: f32,
    /// If the proportion of voiced frames exceed this threshold value, the sample is considered
    /// "voiced".
    voiced_proportion_threshold: f32,
}

impl Silero {
    const PADDING_CHUNKS: usize = 3;

    pub fn with_detection_probability_threshold(mut self, probability: f32) -> Self {
        self.detection_probability_threshold = probability;
        self
    }

    /// A "Default" whisper-ready Silero configuration for realtime transcription.
    pub fn try_new_whisper_realtime_default() -> Result<Self, RibbleWhisperError> {
        SileroBuilder::new()
            .with_sample_rate(SileroSampleRate::R16kHz)
            .with_detection_probability_threshold(REAL_TIME_VOICE_PROBABILITY_THRESHOLD)
            .with_voiced_proportion_threshold(DEFAULT_VOICE_PROPORTION_THRESHOLD)
            .build()
    }

    /// A "Default" whisper-ready Silero configuration for offline transcription.
    pub fn try_new_whisper_offline_default() -> Result<Self, RibbleWhisperError> {
        SileroBuilder::new()
            .with_sample_rate(SileroSampleRate::R16kHz)
            .with_detection_probability_threshold(OFFLINE_VOICE_PROBABILITY_THRESHOLD)
            .with_voiced_proportion_threshold(DEFAULT_VOICE_PROPORTION_THRESHOLD)
            .build()
    }
}

impl Resettable for Silero {
    /// Clears the state of the VAD backend. For VAD reuse.
    fn reset_session(&mut self) {
        // VoiceActivityDetector does not reset configurations to default settings when resetting the context
        // so this method is just a simple delegate.
        self.vad.reset()
    }
}

impl<T: voice_activity_detector::Sample> VAD<T> for Silero {
    /// Detects whether the given samples contain voiced audio.
    /// NOTE: This implementation assumes that the samples are at the same sample rate as the configured VAD.
    /// A mismatch is likely to produce incorrect results.
    fn voice_detected(&mut self, samples: &[T]) -> bool {
        // If a zero-length slice of samples are sent, there is obviously no voice
        if samples.is_empty() {
            return false;
        }
        // Create a LabelIterator to stream the prediction over the sample chunks at the given detection threshold.
        // LabelIterator allows for padding to compensate for sudden speech cutoffs/gaps in audio;
        // 5 frames is expected to be sufficient.
        let probabilites = samples.iter().copied().label(
            &mut self.vad,
            self.detection_probability_threshold,
            Self::PADDING_CHUNKS,
        );

        // Since LabelIterator/ProbabilityIterator do not have a non-consuming way to compare
        // the number of voiced frames versus the total number of frames, the accmumulation has to
        // be done explicitly.
        let mut total_frames = 0usize;
        let mut voiced_frames = 0usize;
        for label in probabilites {
            match label {
                LabeledAudio::Speech(_) => {
                    total_frames += 1;
                    voiced_frames += 1
                }
                LabeledAudio::NonSpeech(_) => total_frames += 1,
            }
        }

        assert_ne!(total_frames, 0);
        // If more than half the frames meet the given threshold, treat the sample as containing speech
        let voiced_proportion = voiced_frames as f32 / total_frames as f32;
        voiced_proportion >= self.voiced_proportion_threshold
    }
    fn extract_voiced_frames(&mut self, samples: &[T]) -> Box<[T]> {
        if samples.is_empty() {
            return vec![].into_boxed_slice();
        }

        samples
            .iter()
            .copied()
            .label(
                &mut self.vad,
                self.detection_probability_threshold,
                Self::PADDING_CHUNKS,
            )
            .filter(|frame| frame.is_speech())
            // Extract the chunks which contain speech
            .flat_map(|frame| frame.iter().copied().collect::<Vec<T>>())
            .collect()
    }
}

/// Encapsulates available sample rates available for [WebRtc] and [Earshot].
#[derive(Copy, Clone, Debug)]
pub enum WebRtcSampleRate {
    R8kHz,
    R16kHz,
    R32kHz,
    R48kHz,
}

impl WebRtcSampleRate {
    fn to_webrtc_sample_rate(self) -> webrtc_vad::SampleRate {
        match self {
            Self::R8kHz => webrtc_vad::SampleRate::Rate8kHz,
            Self::R16kHz => webrtc_vad::SampleRate::Rate16kHz,
            Self::R32kHz => webrtc_vad::SampleRate::Rate32kHz,
            Self::R48kHz => webrtc_vad::SampleRate::Rate48kHz,
        }
    }
    fn to_sample_rate_hz(self) -> usize {
        match self {
            Self::R8kHz => 8000usize,
            Self::R16kHz => 16000usize,
            Self::R32kHz => 32000usize,
            Self::R48kHz => 48000usize,
        }
    }
}

/// Encapsulates available aggressiveness parameters for [WebRtc] and [Earshot]
/// This parameter sets the "mode" from which predetermined speech threshold constants are selected for filtering out non-speech.
/// See: <https://chromium.googlesource.com/external/webrtc/+/refs/heads/master/common_audio/vad/vad_core.c#68>
/// Quality = low filtering, detects most speech and then some. May introduce some false positives
/// VeryAggressive = high filtering, only clear speech passes. Might introduce false negatives
///
/// For small samples (and clear enough audio), higher aggressiveness is likely to produce better results
#[derive(Copy, Clone, Debug)]
pub enum WebRtcFilterAggressiveness {
    Quality,
    LowBitrate,
    Aggressive,
    VeryAggressive,
}

impl WebRtcFilterAggressiveness {
    fn to_webrtc_vad_mode(self) -> webrtc_vad::VadMode {
        match self {
            Self::Quality => webrtc_vad::VadMode::Quality,
            Self::LowBitrate => webrtc_vad::VadMode::LowBitrate,
            Self::Aggressive => webrtc_vad::VadMode::Aggressive,
            Self::VeryAggressive => webrtc_vad::VadMode::VeryAggressive,
        }
    }
    fn to_earshot_vad_profile(self) -> earshot::VoiceActivityProfile {
        match self {
            Self::Quality => earshot::VoiceActivityProfile::QUALITY,
            Self::LowBitrate => earshot::VoiceActivityProfile::LBR,
            Self::Aggressive => earshot::VoiceActivityProfile::AGGRESSIVE,
            Self::VeryAggressive => earshot::VoiceActivityProfile::VERY_AGGRESSIVE,
        }
    }
}

/// Encapsulates available Frame lengths (in ms) for [WebRtc]
/// and [Earshot].
/// Since WebRTC expects frames of specific fixed length, this enumeration captures the only possible
/// valid lengths. It is not used directly in the implementation or the VAD backend; its purpose is
/// to provide information required to compute the necessary sample padding/truncation to fit the
/// frame size requirements.
/// This can be considered a less-flexible equivalent to the Silero chunk_size parameter.

#[derive(Copy, Clone, Debug)]
pub enum WebRtcFrameLengthMillis {
    MS10 = 10,
    MS20 = 20,
    MS30 = 30,
}

impl WebRtcFrameLengthMillis {
    fn to_ms(self) -> usize {
        match self {
            Self::MS10 => 10usize,
            Self::MS20 => 20usize,
            Self::MS30 => 30usize,
        }
    }
}

/// Builder for creating either a WebRtc or an Earshot backend for use in transcription.
/// Due to the way WebRTC is implemented, detection_probability_threshold should be treated as the
/// minimum proportion of frames that must be detected as having speech for the sample to contain
/// voice activity.
/// This is a non-inclusive lower-bound; samples with VAD frame proportions higher than this
/// threshold are thus considered to contain speech.
#[derive(Copy, Clone)]
pub struct WebRtcBuilder {
    sample_rate: WebRtcSampleRate,
    aggressiveness: WebRtcFilterAggressiveness,
    frame_length: WebRtcFrameLengthMillis,
    voiced_proportion_threshold: f32,
}

impl WebRtcBuilder {
    pub fn new() -> Self {
        Self {
            sample_rate: WebRtcSampleRate::R8kHz,
            aggressiveness: WebRtcFilterAggressiveness::Quality,
            frame_length: WebRtcFrameLengthMillis::MS10,
            voiced_proportion_threshold: 0.0,
        }
    }
    /// Sets the sample rate.
    pub fn with_sample_rate(mut self, sample_rate: WebRtcSampleRate) -> Self {
        self.sample_rate = sample_rate;
        self
    }
    /// Sets the filter aggressiveness.
    pub fn with_filter_aggressiveness(
        mut self,
        aggressiveness: WebRtcFilterAggressiveness,
    ) -> Self {
        self.aggressiveness = aggressiveness;
        self
    }
    /// Sets the detection probability threshold: the proportion of sample frames that must contain
    /// speech for the VAD to conclude a sample contains voice.
    pub fn with_voiced_proportion_threshold(mut self, probability_threshold: f32) -> Self {
        self.voiced_proportion_threshold = probability_threshold;
        self
    }
    /// Sets the frame length (in ms).
    pub fn with_frame_length_millis(mut self, frame_length: WebRtcFrameLengthMillis) -> Self {
        self.frame_length = frame_length;
        self
    }

    /// Builds a [WebRtc] VAD backend.
    /// Returns Err if there's an internal panic due to a memory allocation error.
    pub fn build_webrtc(self) -> Result<WebRtc, RibbleWhisperError> {
        std::panic::catch_unwind(|| {
            webrtc_vad::Vad::new_with_rate_and_mode(
                self.sample_rate.to_webrtc_sample_rate(),
                self.aggressiveness.to_webrtc_vad_mode(),
            )
        })
        .map(|vad| WebRtc {
            vad: Mutex::new(vad),
            // WebRtcSampleRate and WebRtcFilterAgressiveness both have Copy semantics
            sample_rate: self.sample_rate,
            aggressiveness: self.aggressiveness,
            frame_length_in_ms: self.frame_length.to_ms(),
            voiced_proportion_threshold: self.voiced_proportion_threshold,
        })
        .map_err(|_| {
            RibbleWhisperError::ParameterError(
                "Failed to build WebRTC due to memory allocation error.".to_string(),
            )
        })
    }

    /// Builds a [Earshot] VAD backend.
    pub fn build_earshot(self) -> Result<Earshot, RibbleWhisperError> {
        let vad = earshot::VoiceActivityDetector::new(self.aggressiveness.to_earshot_vad_profile());
        let predicate = match self.sample_rate {
            WebRtcSampleRate::R8kHz => earshot::VoiceActivityDetector::predict_8khz,
            WebRtcSampleRate::R16kHz => earshot::VoiceActivityDetector::predict_16khz,
            WebRtcSampleRate::R32kHz => earshot::VoiceActivityDetector::predict_32khz,
            WebRtcSampleRate::R48kHz => earshot::VoiceActivityDetector::predict_48khz,
        };
        Ok(Earshot {
            vad,
            sample_rate: self.sample_rate.to_sample_rate_hz(),
            frame_length_in_ms: self.frame_length.to_ms(),
            voiced_proportion_threshold: self.voiced_proportion_threshold,
            prediction_predicate: predicate,
        })
    }
}

impl Default for WebRtcBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// A thread-safe WebRtc VAD backend for use in transcription.
/// Adapts [webrtc_vad::Vad] to predict voice activity using WebRtc
pub struct WebRtc {
    vad: Mutex<webrtc_vad::Vad>,
    sample_rate: WebRtcSampleRate,
    aggressiveness: WebRtcFilterAggressiveness,
    // This value will always be restricted to either 10ms, 20ms, 30ms as per the Enumeration
    // used in the accompanying builder
    frame_length_in_ms: usize,
    /// The proportion threshold for voiced frames. Samples with proportions of voiced frames higher
    /// than this threshold are assumed to contain voice activity.
    voiced_proportion_threshold: f32,
}

impl WebRtc {
    /// Sets the realtime detection probability threshold.
    pub fn with_voiced_proportion_threshold(mut self, proportion: f32) -> Self {
        self.voiced_proportion_threshold = proportion;
        self
    }

    /// A "Default" whisper-ready WebRtc configuration for realtime transcription.
    pub fn try_new_whisper_realtime_default() -> Result<Self, RibbleWhisperError> {
        WebRtcBuilder::new()
            .with_sample_rate(WebRtcSampleRate::R16kHz)
            .with_filter_aggressiveness(WebRtcFilterAggressiveness::LowBitrate)
            .with_voiced_proportion_threshold(DEFAULT_VOICE_PROPORTION_THRESHOLD)
            .build_webrtc()
    }
    /// A "Default" whisper-ready WebRtc configuration for offline transcription.
    pub fn try_new_whisper_offline_default() -> Result<Self, RibbleWhisperError> {
        WebRtcBuilder::new()
            .with_sample_rate(WebRtcSampleRate::R16kHz)
            .with_filter_aggressiveness(WebRtcFilterAggressiveness::Aggressive)
            .with_voiced_proportion_threshold(DEFAULT_VOICE_PROPORTION_THRESHOLD)
            .build_webrtc()
    }
}

/// WebRtc is Mutex-protected to adhere to the following thread-safety guarantees made by WebRtc Vad:
/// <https://chromium.googlesource.com/external/webrtc/+/0332c2db39d6f5c780ce9e92b850bcb57e24e7f8/webrtc/modules/audio_processing/include/audio_processing.h#197>
unsafe impl Send for WebRtc {}
unsafe impl Sync for WebRtc {}

impl Resettable for WebRtc {
    /// Clears the state of the VAD backend. For VAD reuse.
    fn reset_session(&mut self) {
        // WebRtc_vad reverts to default settings when the context is cleared.
        // The vad must be reconfigured accordingly.
        let mut vad = self.vad.lock();
        vad.reset();
        vad.set_sample_rate(self.sample_rate.to_webrtc_sample_rate());
        vad.set_mode(self.aggressiveness.to_webrtc_vad_mode());
    }
}

impl<T: PcmS16Convertible + Copy> VAD<T> for WebRtc {
    /// Detects whether the given samples contain voiced audio.
    /// NOTE: This implementation assumes that the samples are at the same sample rate as the configured VAD
    /// A mismatch is likely to produce incorrect results.
    /// Samples of insufficient length are zero-padded/truncated to avoid internal panicking.
    fn voice_detected(&mut self, samples: &[T]) -> bool {
        // If a zero-length slice of samples are sent, there is obviously no voice
        if samples.is_empty() {
            return false;
        }
        let (int_audio, frame_size) = prepare_webrtc_frames(
            samples,
            self.frame_length_in_ms,
            self.sample_rate.to_sample_rate_hz(),
        );

        let frames = int_audio.chunks_exact(frame_size);

        let total_num_frames = frames.len();
        assert_ne!(total_num_frames, 0);
        let mut vad = self.vad.lock();

        let speech_frames = frames.filter(|&frame| {
            // This should never, ever panic unless my arithmetic is busted
            // Unwrap to force a panic to catch errors in the implementation.
            vad.is_voice_segment(frame)
                .expect("The frame size should be valid.")
        });

        // Since WebRtc doesn't allow users to set the "threshold" directly, treat the threshold
        // like a minimum proportion of frames that have to be detected to be considered speech
        let voiced_proportion = (speech_frames.count() as f32) / (total_num_frames as f32);
        voiced_proportion >= self.voiced_proportion_threshold
    }
    fn extract_voiced_frames(&mut self, samples: &[T]) -> Box<[T]> {
        if samples.is_empty() {
            return vec![].into_boxed_slice();
        }
        let (int_audio, frame_size) = prepare_webrtc_frames(
            samples,
            self.frame_length_in_ms,
            self.sample_rate.to_sample_rate_hz(),
        );
        let frames = int_audio.chunks_exact(frame_size);
        let mut vad = self.vad.lock();
        frames
            .filter(|&frame| {
                vad.is_voice_segment(frame)
                    .expect("The Frame size should be valid")
            })
            .flatten()
            .map(|&s| T::from_pcm_s16(s))
            .collect()
    }
}

/// Type alias for earshot function pointers, eg [earshot::VoiceActivityDetector::predict_8khz].
/// For statically dispatching the correct voice detection based on sample rate.
type EarshotPredictionFilterPredicate =
    fn(&mut earshot::VoiceActivityDetector, &[i16]) -> Result<bool, earshot::Error>;

/// Earshot VAD backend for use in transcription.
/// Adapts [earshot::VoiceActivityDetector] to predict voice activity using Earshot (WebRtc)
/// #[deprecated(since = "0.2.1", note = "There are significant issues with this implementation. Prefer using WebRtc."]
pub struct Earshot {
    vad: earshot::VoiceActivityDetector,
    /// Used to break the sample into frames of size frame_length_in_ms
    sample_rate: usize,
    /// Used to break the sample into sized chunks that can be run through the voice detector.
    frame_length_in_ms: usize,
    /// The proportion threshold for voiced frames. Samples with proportions of voiced frames higher
    /// than this threshold are assumed to contain voice activity.
    voiced_proportion_threshold: f32,
    /// Used to statically dispatch the correct method based on the sample rate.
    prediction_predicate: EarshotPredictionFilterPredicate,
}

impl Earshot {
    /// Sets the realtime detection probability threshold.
    pub fn with_realtime_detection_probability_threshold(
        mut self,
        voiced_proportion_threshold: f32,
    ) -> Self {
        self.voiced_proportion_threshold = voiced_proportion_threshold;
        self
    }

    /// A "Default" whisper-ready Earhshot configuration for realtime transcription.
    pub fn try_new_whisper_realtime_default() -> Result<Self, RibbleWhisperError> {
        WebRtcBuilder::new()
            .with_sample_rate(WebRtcSampleRate::R16kHz)
            .with_filter_aggressiveness(WebRtcFilterAggressiveness::LowBitrate)
            .with_voiced_proportion_threshold(DEFAULT_VOICE_PROPORTION_THRESHOLD)
            .build_earshot()
    }

    /// A "Default" whisper-ready Earhshot configuration for offline transcription.
    pub fn try_new_whisper_offline_default() -> Result<Self, RibbleWhisperError> {
        WebRtcBuilder::new()
            .with_sample_rate(WebRtcSampleRate::R16kHz)
            .with_filter_aggressiveness(WebRtcFilterAggressiveness::Aggressive)
            .with_voiced_proportion_threshold(DEFAULT_VOICE_PROPORTION_THRESHOLD)
            .build_earshot()
    }
}

impl Resettable for Earshot {
    /// Clears the state of the VAD backend. For VAD reuse.
    fn reset_session(&mut self) {
        // Earshot does not revert back to default settings when resetting the context,
        // so this method is just a delegate.
        self.vad.reset()
    }
}

impl<T: PcmS16Convertible + Copy> VAD<T> for Earshot {
    /// Detects whether the given samples contain voiced audio.
    /// NOTE: This implementation assumes that the samples are at the same sample rate as the configured VAD.
    /// A mismatch is likely to produce incorrect results.
    /// Samples of insufficient length are zero-padded/truncated to avoid internal panicking.
    fn voice_detected(&mut self, samples: &[T]) -> bool {
        // If a zero-length slice of samples are sent, there is obviously no voice
        if samples.is_empty() {
            return false;
        }

        let (int_audio, frame_size) =
            prepare_webrtc_frames(samples, self.frame_length_in_ms, self.sample_rate);
        let frames = int_audio.chunks_exact(frame_size);

        let total_num_frames = frames.len();
        assert_ne!(total_num_frames, 0);
        let vad = &mut self.vad;
        let speech_frames = frames.filter(|&frame| {
            (self.prediction_predicate)(vad, frame).expect("Frame size should be valid.")
        });

        // Like WebRtc, (this is a WebRtc implementation),
        // doesn't allow users to set the "threshold" directly, treat the threshold
        // like a minimum proportion of frames that have to be detected to be considered speech

        let voiced_proportion = (speech_frames.count() as f32) / (total_num_frames as f32);
        voiced_proportion >= self.voiced_proportion_threshold
    }

    fn extract_voiced_frames(&mut self, samples: &[T]) -> Box<[T]> {
        if samples.is_empty() {
            return vec![].into_boxed_slice();
        }

        let (int_audio, frame_size) =
            prepare_webrtc_frames(samples, self.frame_length_in_ms, self.sample_rate);
        let frames = int_audio.chunks_exact(frame_size);
        let vad = &mut self.vad;
        frames
            .filter(|&frame| {
                (self.prediction_predicate)(vad, frame).expect("Frame size should be valid.")
            })
            .flatten()
            .map(|&s| T::from_pcm_s16(s))
            .collect()
    }
}

// This small utility function handles the padding/truncation required by WebRTC implementations
// It returns the converted and padded audio, as well as the frame size so that methods consuming
// this function do not need to recompute the frame size.
fn prepare_webrtc_frames<T: PcmS16Convertible + Copy>(
    samples: &[T],
    frame_length_in_ms: usize,
    sample_rate: usize,
) -> (Vec<i16>, usize) {
    // Convert to integer audio
    let mut int_audio: Vec<i16> = samples.iter().map(|s| s.into_pcm_s16()).collect();
    // Because of implementation details in WebRTC, frames need to be either 10ms, 20ms, or 30ms in length
    // This means the length must be a multiple of (sample_rate * audio_length) / 1000
    // eg. 8kHz -> 80, 160, 240
    let frame_size = (sample_rate * frame_length_in_ms) / 1000;
    let zero_pad = frame_size - int_audio.len().rem_euclid(frame_size);
    int_audio.resize(int_audio.len() + zero_pad, 0);
    (int_audio, frame_size)
}

// THESE ARE (LOWER-BOUND) RECOMMENDATIONS
// Silero can be especially picky with certain words so err on the side of caution.
pub const REAL_TIME_VOICE_PROBABILITY_THRESHOLD: f32 = 0.3;
pub const DEFAULT_VOICE_PROPORTION_THRESHOLD: f32 = 0.5;
pub const OFFLINE_VOICE_PROBABILITY_THRESHOLD: f32 = 0.60;
