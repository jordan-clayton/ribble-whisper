use std::sync::Arc;

use crate::audio::audio_ring_buffer::AudioRingBuffer;
use crate::audio::recorder::{
    AudioInputAdapter, ClosedLoopRecorder, FanoutRecorder, RecorderSample, UseArc, UseVec,
};
use crate::utils::errors::RibbleWhisperError;
use crate::utils::{constants, Sender};
use sdl2::audio::{AudioDevice, AudioSpecDesired};
use sdl2::{AudioSubsystem, Sdl};

/// A Basic Audio Backend that uses SDL to gain access to the default audio input
/// At this time there is no support for other audio backends, but this may change in the future.
#[derive(Clone)]
pub struct AudioBackend {
    sdl_ctx: Arc<Sdl>,
    audio_subsystem: AudioSubsystem,
}

impl AudioBackend {
    /// Initializes the audio backend to access the microphone when running a realtime transcription
    /// It is not encouraged to call new more than once, but it is not an error to do so.
    /// Returns an error if the backend fails to initialize.
    pub fn new() -> Result<Self, RibbleWhisperError> {
        let ctx = sdl2::init().map_err(|e| {
            RibbleWhisperError::ParameterError(format!(
                "Failed to create SDL context, error: {}",
                e
            ))
        })?;

        let audio_subsystem = ctx.audio().map_err(|e| {
            RibbleWhisperError::ParameterError(format!(
                "Failed to initialize audio subsystem, error: {}",
                e
            ))
        })?;

        let sdl_ctx = Arc::new(ctx);
        Ok(Self {
            sdl_ctx,
            audio_subsystem,
        })
    }

    /// To access the inner [Sdl] context
    pub fn sdl_ctx(&self) -> Arc<Sdl> {
        self.sdl_ctx.clone()
    }
    /// To access the inner [AudioSubsystem]
    pub fn audio_subsystem(&self) -> AudioSubsystem {
        self.audio_subsystem.clone()
    }

    /// A convenience method that prepares [AudioDevice] for use
    /// with [crate::transcriber::realtime_transcriber::RealtimeTranscriber] to transcribe
    /// audio realtime. Use the fanout capture when doing other audio processing concurrently
    /// with transcription.
    ///
    /// # Arguments:
    /// * audio_sender: a message sender to forward audio from the input device
    /// # Returns:
    /// * Ok(AudioDevice) on success, Err(RibbleWhisperError) on failure to build.
    /// See: [MicCaptureBuilder] for error conditions.
    pub fn build_whisper_fanout_default<
        T: RecorderSample,
        AC: AudioInputAdapter<T> + Send + Clone,
    >(
        &self,
        audio_sender: Sender<AC::SenderOutput>,
    ) -> Result<FanoutMicCapture<T, AC>, RibbleWhisperError> {
        self.build_whisper_default().build_fanout(audio_sender)
    }

    /// A convenience method that prepares [AudioDevice] for use
    /// with [crate::transcriber::realtime_transcriber::RealtimeTranscriber] to transcribe
    /// audio realtime. Use the closed loop capture when only transcription processing is required.
    ///
    /// # Arguments:
    /// * buffer: a ringbuffer for storing audio from the input device.
    /// # Returns:
    /// * Ok(AudioDevice) on success, Err(RibbleWhisperError) on failure to build.
    /// See: [MicCaptureBuilder] for error conditions.
    pub fn build_whisper_closed_loop_default<T: RecorderSample>(
        &self,
        buffer: &AudioRingBuffer<T>,
    ) -> Result<ClosedLoopMicCapture<T>, RibbleWhisperError> {
        self.build_whisper_default().build_closed_loop(buffer)
    }

    fn build_whisper_default(&self) -> MicCaptureBuilder {
        self.build_microphone()
            .with_num_channels(Some(1))
            .with_sample_rate(Some(constants::WHISPER_SAMPLE_RATE as i32))
            .with_sample_size(Some(constants::AUDIO_BUFFER_SIZE))
    }

    /// Returns a builder to set up the audio capture parameters
    pub fn build_microphone(&self) -> MicCaptureBuilder {
        MicCaptureBuilder::new(&self.audio_subsystem)
    }
}

/// A builder for setting (SDL) audio input configurations
#[derive(Clone)]
pub struct MicCaptureBuilder<'a> {
    audio_subsystem: &'a AudioSubsystem,
    audio_spec_desired: AudioSpecDesired,
}

impl<'a> MicCaptureBuilder<'a> {
    pub fn new(audio_subsystem: &'a AudioSubsystem) -> Self {
        // AudioSpecDesired does not implement Default
        let audio_spec_desired = AudioSpecDesired {
            freq: None,
            channels: None,
            samples: None,
        };

        Self {
            audio_subsystem,
            audio_spec_desired,
        }
    }
    /// To change the [AudioSubsystem]
    pub fn with_audio_subsystem(mut self, audio_subsystem: &'a AudioSubsystem) -> Self {
        self.audio_subsystem = audio_subsystem;
        self
    }

    /// To change the desired sample rate
    pub fn with_sample_rate(mut self, sample_rate: Option<i32>) -> Self {
        self.audio_spec_desired.freq = sample_rate;
        self
    }

    /// To change the desired number of channels (eg. 1 = mono, 2 = stereo)
    /// NOTE: Realtime transcription requires audio to be in/converted to mono
    /// [crate::transcriber::realtime_transcriber::RealtimeTranscriber] does not handle conversion.
    pub fn with_num_channels(mut self, num_channels: Option<u8>) -> Self {
        self.audio_spec_desired.channels = num_channels;
        self
    }

    /// To set the input audio buffer size.
    /// NOTE: this must be a power of two.
    /// Providing an invalid size will result in falling back to default settings
    pub fn with_sample_size(mut self, samples: Option<u16>) -> Self {
        self.audio_spec_desired.samples = samples.filter(|s| s.is_power_of_two());
        self
    }

    /// To set the desired audio spec all at once.
    /// This will not be useful unless you are already managing SDL on your own.
    ///
    pub fn with_desired_audio_spec(mut self, spec: AudioSpecDesired) -> Self {
        self.audio_spec_desired = spec;
        self
    }

    /// Builds [AudioDevice] to open audio capture (eg. for use in realtime transcription).
    /// Fans out data via message passing for use when doing additional audio processing concurrently
    /// with transcription.
    /// # Arguments:
    /// * audio_sender: a message sender to forward audio from the input device
    /// # Returns:
    /// * Ok(AudioDevice) on success, Err(RibbleWhisperError) on an SDL failure.
    pub fn build_fanout<T: RecorderSample, AC: AudioInputAdapter<T> + Send + Clone>(
        self,
        sender: Sender<AC::SenderOutput>,
    ) -> Result<FanoutMicCapture<T, AC>, RibbleWhisperError> {
        let device = self
            .audio_subsystem
            .open_capture(None, &self.audio_spec_desired, |_| {
                FanoutRecorder::new(sender)
            })
            .map_err(|e| {
                RibbleWhisperError::ParameterError(format!("Failed to build audio stream: {}", e))
            })?;
        Ok(FanoutMicCapture { device })
    }

    /// Builds [AudioDevice] to open audio capture (eg. for use in realtime transcription).
    /// Writes directly into the ringbuffer, for when only transcription is required.
    /// Prefer the fanout implementation when doing additional processing during transcription
    /// to guarantee data coherence.
    ///
    /// # Arguments:
    /// * buffer: a ringbuffer for storing audio from the input device.
    /// # Returns:
    /// * Ok(AudioDevice) on success, Err(RibbleWhisperError) on an SDL failure.
    pub fn build_closed_loop<T: RecorderSample>(
        self,
        buffer: &AudioRingBuffer<T>,
    ) -> Result<ClosedLoopMicCapture<T>, RibbleWhisperError> {
        let device = self
            .audio_subsystem
            .open_capture(None, &self.audio_spec_desired, |_| {
                ClosedLoopRecorder::new(buffer.clone())
            })
            .map_err(|e| {
                RibbleWhisperError::ParameterError(format!("Failed to build audio stream: {}", e))
            })?;
        Ok(ClosedLoopMicCapture { device })
    }
}

/// Trait for starting/stopping audio capture.
pub trait MicCapture {
    fn play(&self);
    fn pause(&self);
}

/// Encapsulates [AudioDevice] and sends audio samples out via message channels.
/// Use when performing other audio processing concurrently with transcription
/// (see: examples/realtime_stream.rs).
/// Due to the use of SDL, this cannot be shared across threads.
pub struct FanoutMicCapture<T, AC>
where
    T: RecorderSample,
    AC: AudioInputAdapter<T> + Clone + Send,
{
    device: AudioDevice<FanoutRecorder<T, AC>>,
}

impl<T, AC> MicCapture for FanoutMicCapture<T, AC>
where
    T: RecorderSample,
    AC: AudioInputAdapter<T> + Clone + Send,
{
    fn play(&self) {
        self.device.resume()
    }
    fn pause(&self) {
        self.device.pause()
    }
}

/// Encapsulates [AudioDevice] and writes directly into [AudioRingBuffer].
/// Use when only transcription processing is required.
/// Due to the use of SDL, this cannot be shared across threads.
pub struct ClosedLoopMicCapture<T: RecorderSample> {
    device: AudioDevice<ClosedLoopRecorder<T>>,
}

impl<T: RecorderSample> MicCapture for ClosedLoopMicCapture<T> {
    fn play(&self) {
        self.device.resume()
    }

    fn pause(&self) {
        self.device.pause()
    }
}

// The following functions are exposed but their use is not encouraged unless required.

/// This is deprecated and will be removed at a later date.
/// Prefer [MicCaptureBuilder].
#[inline]
pub fn get_desired_audio_spec(
    freq: Option<i32>,
    channels: Option<u8>,
    samples: Option<u16>,
) -> AudioSpecDesired {
    AudioSpecDesired {
        freq,
        channels,
        samples,
    }
}

/// Visibility is currently exposed due to legacy implementations.
/// Do not rely on this function, as it will be marked private in the future.
/// Prefer [MicCaptureBuilder]
#[inline]
pub fn build_audio_stream<T: RecorderSample, AC: AudioInputAdapter<T> + Send>(
    audio_subsystem: &AudioSubsystem,
    desired_spec: &AudioSpecDesired,
    audio_sender: Sender<AC::SenderOutput>,
) -> Result<AudioDevice<FanoutRecorder<T, AC>>, RibbleWhisperError> {
    let audio_stream = audio_subsystem
        .open_capture(
            // Device - should be default, SDL should change if the user changes devices in their sysprefs.
            None,
            desired_spec,
            |_spec| FanoutRecorder::new(audio_sender),
        )
        .map_err(|e| {
            RibbleWhisperError::ParameterError(format!("Failed to build audio stream: {}", e))
        })?;

    Ok(audio_stream)
}

/// Visibility is currently exposed due to legacy implementations.
/// Do not rely on this function, as it will be marked private in the future.
/// Prefer [MicCaptureBuilder]
#[inline]
pub fn build_audio_stream_vec_sender<T: RecorderSample>(
    audio_subsystem: &AudioSubsystem,
    desired_spec: &AudioSpecDesired,
    audio_sender: Sender<Vec<T>>,
) -> Result<AudioDevice<FanoutRecorder<T, UseVec>>, RibbleWhisperError> {
    let audio_stream = audio_subsystem.open_capture(
        // Device - should be default, SDL should change if the user changes devices in their sysprefs.
        None,
        desired_spec,
        |_spec| FanoutRecorder::new_vec(audio_sender),
    );
    match audio_stream {
        Err(e) => Err(RibbleWhisperError::ParameterError(format!(
            "Failed to build audio stream: {}",
            e
        ))),
        Ok(audio) => Ok(audio),
    }
}

/// Visibility is currently exposed due to legacy implementations.
/// Do not rely on this function, as it will be marked private in the future.
/// Prefer [MicCaptureBuilder]
#[inline]
pub fn build_audio_stream_slice_sender<T: RecorderSample>(
    audio_subsystem: &AudioSubsystem,
    desired_spec: &AudioSpecDesired,
    audio_sender: Sender<Arc<[T]>>,
) -> Result<AudioDevice<FanoutRecorder<T, UseArc>>, RibbleWhisperError> {
    let audio_stream = audio_subsystem
        .open_capture(
            // Device - should be default, SDL should change if the user changes devices in their sysprefs.
            None,
            desired_spec,
            |_spec| FanoutRecorder::new_arc(audio_sender),
        )
        .map_err(|e| {
            RibbleWhisperError::ParameterError(format!("Failed to build audio stream: {}", e))
        })?;

    Ok(audio_stream)
}
