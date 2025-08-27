use parking_lot::Mutex;
use std::ffi::{CStr, c_int, c_void};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use whisper_rs::{WhisperNewSegmentCallback, WhisperProgressCallback};

use crate::audio::{AudioChannelConfiguration, WhisperAudioSample};
use crate::transcriber::vad::VAD;
use crate::transcriber::{
    OfflineWhisperNewSegmentCallback, OfflineWhisperProgressCallback, WhisperCallbacks,
    build_whisper_context,
};
use crate::utils::errors::RibbleWhisperError;
use crate::whisper::configs::WhisperConfigsV2;
use crate::whisper::model::ModelRetriever;

/// Builder for [OfflineTranscriber]
/// Silero: [crate::transcriber::vad::Silero] is recommended for accuracy.
pub struct OfflineTranscriberBuilder<V, M>
where
    V: VAD<f32>,
    M: ModelRetriever,
{
    configs: Option<Arc<WhisperConfigsV2>>,
    audio: Option<WhisperAudioSample>,
    channels: Option<AudioChannelConfiguration>,
    model_retriever: Option<Arc<M>>,
    /// (Optional) Used to extract voiced segments to reduce overall transcription time.
    voice_activity_detector: Option<Arc<Mutex<V>>>,
}

impl<V, M> OfflineTranscriberBuilder<V, M>
where
    V: VAD<f32>,
    M: ModelRetriever,
{
    pub fn new() -> Self {
        Self {
            configs: None,
            audio: None,
            channels: None,
            model_retriever: None,
            voice_activity_detector: None,
        }
    }
    /// Sets the whisper configurations
    pub fn with_configs(mut self, configs: WhisperConfigsV2) -> Self {
        self.configs = Some(Arc::new(configs));
        self
    }

    /// Sets the audio to be transcribed
    /// WhisperAudioSamples are cheap to clone (and share audio).
    pub fn with_audio(mut self, audio: WhisperAudioSample) -> Self {
        self.audio = Some(audio.clone());
        self
    }

    /// Sets the audio channel configurations.
    /// NOTE: This must match the audio source channel configurations or there will be transcription artefacts.
    pub fn with_channel_configurations(mut self, channels: AudioChannelConfiguration) -> Self {
        self.channels = Some(channels);
        self
    }

    /// Sets an optional voice activity detector to optimize transcription by pruning out unvoiced audio frames
    pub fn with_voice_activity_detector<V2: VAD<f32>>(
        self,
        vad: V2,
    ) -> OfflineTranscriberBuilder<V2, M> {
        let v = Arc::new(Mutex::new(vad));
        OfflineTranscriberBuilder {
            configs: self.configs,
            audio: self.audio,
            channels: self.channels,
            model_retriever: self.model_retriever,
            voice_activity_detector: Some(v),
        }
    }
    /// Sets an optional voice activity detector to optimize transcription by pruning out unvoiced audio frames.
    /// This method is for when using a shared (e.g. pre-allocated VAD)
    /// **NOTE: Trying to use this VAD in 2 places simultaneously will result in significant lock contention.**
    /// **NOTE: VADs must be reset before being used in a different context**
    pub fn with_shared_voice_activity_detector<V2: VAD<f32>>(
        self,
        vad: Arc<Mutex<V2>>,
    ) -> OfflineTranscriberBuilder<V2, M> {
        OfflineTranscriberBuilder {
            configs: self.configs,
            audio: self.audio,
            channels: self.channels,
            model_retriever: self.model_retriever,
            voice_activity_detector: Some(Arc::clone(&vad)),
        }
    }

    /// Sets the model retriever to allow OfflineTranscriber to access models to set up a Whisper state.
    /// See: [ModelRetriever]
    pub fn with_model_retriever<M2: ModelRetriever>(
        self,
        model_retriever: M2,
    ) -> OfflineTranscriberBuilder<V, M2> {
        OfflineTranscriberBuilder {
            configs: self.configs,
            audio: self.audio,
            channels: self.channels,
            model_retriever: Some(Arc::new(model_retriever)),
            voice_activity_detector: None,
        }
    }

    /// Sets a shared model retriever to allow OfflineTranscriber to access models to set up a Whisper state.
    /// See: [ModelRetriever]
    pub fn with_shared_model_retriever<M2: ModelRetriever>(
        self,
        model_retriever: Arc<M2>,
    ) -> OfflineTranscriberBuilder<V, M2> {
        OfflineTranscriberBuilder {
            configs: self.configs,
            audio: self.audio,
            channels: self.channels,
            model_retriever: Some(Arc::clone(&model_retriever)),
            voice_activity_detector: None,
        }
    }
    /// Builds an `OfflineTranscriber<V>` according to the given parameters
    /// # Returns:
    /// * Ok(`OfflineTranscriber<V>`) on successful build
    /// * Err(RibbleWhisperError) if one of the following are true:
    ///   ** missing whisper configurations,
    ///   ** missing channel configurations,
    ///   ** missing audio
    ///   ** Model ID is not set in configs.
    pub fn build(self) -> Result<OfflineTranscriber<V, M>, RibbleWhisperError> {
        let configs = self.configs.ok_or(RibbleWhisperError::ParameterError(
            "Configs missing in OfflineTranscriberBuilder..".to_string(),
        ))?;

        let _model_id = configs.model_id().ok_or(RibbleWhisperError::ParameterError(
            "Model ID missing from configs in OfflineTranscriberBuilder".to_string(),
        ));

        let audio = self.audio.filter(|audio| !audio.is_empty()).ok_or(
            RibbleWhisperError::ParameterError(
                "Audio missing in OfflineTranscriberBuilder.".to_string(),
            ),
        )?;
        let channels = self.channels.ok_or(RibbleWhisperError::ParameterError(
            "Channel configurations missing in OfflineTranscriberBuilder.".to_string(),
        ))?;
        let model_retriever = self
            .model_retriever
            .ok_or(RibbleWhisperError::ParameterError(
                "Model retriever missing in OfflineTranscriberBuilder.".to_string(),
            ))?;

        // Vad can be None; if there is no VAD provided, the full speech will be processed.
        let vad = self.voice_activity_detector;
        Ok(OfflineTranscriber {
            configs,
            audio,
            channels,
            voice_activity_detector: vad,
            model_retriever,
        })
    }
}

impl<V, M> Default for OfflineTranscriberBuilder<V, M>
where
    V: VAD<f32>,
    M: ModelRetriever,
{
    fn default() -> Self {
        Self::new()
    }
}

/// For running offline (non-realtime) transcription using whisper.
/// NOTE: timestamps have not yet been implemented.
pub struct OfflineTranscriber<V, M>
where
    V: VAD<f32>,
    M: ModelRetriever,
{
    /// Whisper configurations
    configs: Arc<WhisperConfigsV2>,
    /// The audio to transcribe.
    audio: WhisperAudioSample,
    /// Mono or Stereo. Stereo will be converted to mono before transcription
    channels: AudioChannelConfiguration,
    /// (Optional) Used to extract voiced segments to reduce overall transcription time.
    voice_activity_detector: Option<Arc<Mutex<V>>>,
    model_retriever: Arc<M>,
}

impl<V, M> OfflineTranscriber<V, M>
where
    V: VAD<f32>,
    M: ModelRetriever,
{
    fn run_transcription(
        &self,
        full_params: whisper_rs::FullParams,
        run_transcription: Arc<AtomicBool>,
    ) -> Result<String, RibbleWhisperError> {
        let whisper_context_params = self.configs.to_whisper_context_params();
        // Since it's not possible to build an OfflineTranscriber without the ID set, this can be
        // safely unwrapped.
        let model_id = self.configs.model_id().unwrap();

        let model_location = self.model_retriever.retrieve_model(model_id).ok_or(
            RibbleWhisperError::ParameterError(format!("Failed to find model: {model_id}")),
        )?;

        // Set up a whisper context
        let ctx = build_whisper_context(model_location, whisper_context_params)?;

        let mut whisper_state = ctx.create_state()?;

        // Prepare audio
        let mut audio_samples = match &self.audio {
            WhisperAudioSample::I16(audio) => {
                let len = audio.len();
                let mut float_samples = vec![0.0; len];
                whisper_rs::convert_integer_to_float_audio(audio, &mut float_samples)?;
                Arc::from(float_samples)
            }
            WhisperAudioSample::F32(audio) => Arc::clone(audio),
        };

        // Extract speech frames if there's a VAD
        if let Some(vad) = self.voice_activity_detector.as_ref() {
            audio_samples = Arc::from(vad.lock().extract_voiced_frames(&audio_samples))
        }

        let mono_audio = match self.channels {
            AudioChannelConfiguration::Mono => audio_samples.to_vec(),
            AudioChannelConfiguration::Stereo => {
                whisper_rs::convert_stereo_to_mono_audio(&audio_samples)?
            }
        };

        if let Err(e) = whisper_state.full(full_params, &mono_audio) {
            // Only escape early if the transcription is still supposed to be running;
            // Otherwise, the abort callback fired true, and run_transcription is false - indicating
            // the user has stopped the transcription.
            if run_transcription.load(Ordering::Acquire) {
                return Err(RibbleWhisperError::WhisperError(e));
            }
        }

        // Otherwise, expect the transcription to have been successful; subsequent errors
        // will bubble up.
        let num_segments = whisper_state.full_n_segments();
        let mut text = Vec::with_capacity(num_segments as usize);

        // Push the transcribed segments to the text buffer
        for segment in whisper_state.as_iter() {
            text.push(segment.to_string())
        }

        // Clean up the whisper context
        drop(whisper_state);
        drop(ctx);
        // Return the final transcription.
        Ok(text.join("").trim().to_string())
    }

    /// Loads a compatible whisper model, sets up the whisper state and runs the full model
    /// # Arguments
    /// * run_transcription: `Arc<AtomicBool>`, a shared flag used to indicate when to stop transcribing
    /// # Returns
    /// * Ok(String) on success, Err(RibbleWhisperError) on failure
    pub fn process_audio(
        &self,
        run_transcription: Arc<AtomicBool>,
    ) -> Result<String, RibbleWhisperError> {
        let confs = Arc::clone(&self.configs);
        let mut full_params = confs.to_whisper_full_params();
        // Abort callback
        let r_transcription = Arc::clone(&run_transcription);

        // Coerce to a void pointer
        let a_ptr = Arc::into_raw(r_transcription) as *mut c_void;
        unsafe {
            full_params.set_abort_callback_user_data(a_ptr);
            full_params.set_abort_callback(Some(abort_callback))
        }

        let res = self.run_transcription(full_params, Arc::clone(&run_transcription));

        // Since the Arc is peeked in the C callback, a_ptr needs to be consumed one last time
        // to prevent memory leaks.
        unsafe {
            let _ = Arc::from_raw(a_ptr as *const AtomicBool);
        }
        res
    }

    /// Handles running Whisper transcription, with support for optional callbacks
    /// These callbacks are called from whisper so their safety cannot be completely guaranteed.
    /// Loads a compatible whisper model, sets up the whisper state and runs the full model
    /// # Arguments
    /// * run_transcription: `Arc<AtomicBool>`, a shared flag used to indicate when to stop transcribing
    /// # Returns
    /// * Ok(String) on success, Err(RibbleWhisperError) on failure
    pub fn process_with_callbacks<P, S>(
        &self,
        run_transcription: Arc<AtomicBool>,
        callbacks: WhisperCallbacks<P, S>,
    ) -> Result<String, RibbleWhisperError>
    where
        P: OfflineWhisperProgressCallback,
        S: OfflineWhisperNewSegmentCallback,
    {
        // Decompose the callbacks struct
        let WhisperCallbacks {
            progress: maybe_progress_callback,
            new_segment: maybe_new_segment_callback,
        } = callbacks;

        let confs = Arc::clone(&self.configs);
        let mut full_params = confs.to_whisper_full_params();

        // Named stack binding for the progress callback
        let mut p_callback;
        // Named stack binding for the new_segment callback
        let mut s_callback;

        // Abort callback
        let r_transcription = Arc::clone(&run_transcription);
        // Coerce to a void pointer
        let a_ptr = Arc::into_raw(r_transcription) as *mut c_void;

        let (progress_callback, progress_user_data): (WhisperProgressCallback, *mut c_void) =
            match maybe_progress_callback {
                None => (None, std::ptr::null_mut::<c_void>()),
                Some(cb) => {
                    p_callback = cb;
                    (
                        Some(progress_callback::<P>),
                        &mut p_callback as *mut P as *mut c_void,
                    )
                }
            };

        let (new_segment_callback, new_segment_user_data): (
            WhisperNewSegmentCallback,
            *mut c_void,
        ) = match maybe_new_segment_callback {
            None => (None, std::ptr::null_mut::<c_void>()),
            Some(cb) => {
                s_callback = cb;
                (
                    Some(new_segment_callback::<S>),
                    &mut s_callback as *mut S as *mut c_void,
                )
            }
        };

        unsafe {
            full_params.set_progress_callback_user_data(progress_user_data);
            full_params.set_progress_callback(progress_callback);
            full_params.set_new_segment_callback_user_data(new_segment_user_data);
            full_params.set_new_segment_callback(new_segment_callback);
            full_params.set_abort_callback_user_data(a_ptr);
            full_params.set_abort_callback(Some(abort_callback))
        }
        let res = self.run_transcription(full_params, Arc::clone(&run_transcription));
        // Since the Arc is peeked in the C callback, a_ptr needs to be consumed one last time
        // to prevent memory leaks.
        unsafe {
            let _ = Arc::from_raw(a_ptr as *const AtomicBool);
        }

        res
    }
}

// C-Callbacks (until "safe" handles are working in whisper-rs)
// More callbacks will be implemented and exposed as necessary.
// NOTE: As of the most current version of this library, all callbacks have been tested and should
// be considered safe. To date, there has not been an issue with the code called across the FFI boundary.
// It is therefore assumed that the callback mechanisms are correct -- there should not be a need to
// heap-allocate the trait objects due to the execution flow of OfflineTranscriber::process_with_callbacks.

// Any panics/segfaults that are encountered imply a bug exists within the closure/function
// passed to a StaticRibbleWhisperCallback object rather than the trampolines.

// This function gets called at the beginning of each run of the decoder to determine whether to
// abort transcription. The function aborts on true.
// Since transcription is being controlled by a run_transcription boolean, the transcription
// is expected to stop when run_transcription is false.
// Internally, this peeks the Arc so that the refcount remains where it is.
// Thus, the calling scope must manually consume the arc after this callback is no longer called.
unsafe extern "C" fn abort_callback(user_data: *mut c_void) -> bool {
    // Consume the pointer
    let ptr = unsafe { Arc::from_raw(user_data as *const AtomicBool) };
    // let arc = Arc::clone(&ptr);
    let run_transcription = ptr.load(Ordering::Acquire);
    // Prevent the refcount from decrementing
    let _ = Arc::into_raw(ptr);
    !run_transcription
}

// This callback gets called in order to forward progress updates from Whisper to a UI.
// To guarantee the safety of the C library, whisper_context and whisper_states should
// not be mutated.
// This function may or may not be called from a multithreaded context.
unsafe extern "C" fn progress_callback<PC: OfflineWhisperProgressCallback>(
    _: *mut whisper_rs_sys::whisper_context,
    _: *mut whisper_rs_sys::whisper_state,
    progress: c_int,
    user_data: *mut c_void,
) {
    let callback = unsafe { &mut *(user_data as *mut PC) };
    callback.call(progress);
}

// This callback fires once new segments have been confirmed to push the last n segments
// joined together into a single string which can be pushed to a working buffer or similar.
unsafe extern "C" fn new_segment_callback<S: OfflineWhisperNewSegmentCallback>(
    _: *mut whisper_rs_sys::whisper_context,
    state: *mut whisper_rs_sys::whisper_state,
    n_new: c_int,
    user_data: *mut c_void,
) {
    let callback = unsafe { &mut *(user_data as *mut S) };

    // Collect into a snapshot and then call the callback.
    let n_segments = unsafe { whisper_rs_sys::whisper_full_n_segments_from_state(state) };
    let s0 = (n_segments - n_new).min(0);
    let mut segments = Vec::with_capacity(n_new as usize);

    for i in s0..n_segments {
        let text = unsafe { whisper_rs_sys::whisper_full_get_segment_text_from_state(state, i) };
        let segment = unsafe { CStr::from_ptr(text) };
        segments.push(segment.to_string_lossy())
    }
    let new_segments = segments.join(" ");
    callback.call(new_segments);
}
