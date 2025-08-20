use std::ops::Deref;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use crate::utils::callback::Callback;
use crate::utils::errors::RibbleWhisperError;
use crate::whisper::model::ModelLocation;
use strum::{Display, EnumString, IntoStaticStr};
use whisper_rs::WhisperSegment;

pub mod offline_transcriber;
pub mod realtime_transcriber;
pub mod vad;

// Trait alias, used until the feature reaches stable
pub trait OfflineWhisperProgressCallback: Callback<Argument = i32> + Send + Sync + 'static {}
impl<T: Callback<Argument = i32> + Send + Sync + 'static> OfflineWhisperProgressCallback for T {}

// This no longer needs to short circuit; the segment callback only fires once things are confirmed.
pub trait OfflineWhisperNewSegmentCallback:
    Callback<Argument = String> + Send + Sync + 'static
{
}
impl<T: Callback<Argument = String> + Send + Sync + 'static> OfflineWhisperNewSegmentCallback
    for T
{
}

#[inline]
pub fn redirect_whisper_logging_to_hooks() {
    whisper_rs::install_logging_hooks()
}

/// Handles running Whisper transcription
pub trait Transcriber {
    /// Loads a compatible whisper model, sets up the whisper state and runs the full model
    /// # Arguments
    /// * run_transcription: `Arc<AtomicBool>`, a shared flag used to indicate when to stop transcribing
    /// # Returns
    /// * Ok(String) on success, Err(RibbleWhisperError) on failure
    fn process_audio(
        &self,
        run_transcription: Arc<AtomicBool>,
    ) -> Result<String, RibbleWhisperError>;
}

/// Handles running Whisper transcription, with support for optional callbacks
/// These callbacks are called from whisper so their safety cannot be completely guaranteed.
/// However, since these callbacks do not touch the whisper state, they should work as expected.
pub trait CallbackTranscriber<P, S>: Transcriber
where
    P: OfflineWhisperProgressCallback,
    S: OfflineWhisperNewSegmentCallback,
{
    fn process_with_callbacks(
        &self,
        run_transcription: Arc<AtomicBool>,
        callbacks: WhisperCallbacks<P, S>,
    ) -> Result<String, RibbleWhisperError>;
}

/// Encapsulates various whisper callbacks which can be set before running transcription
/// Other callbacks will be added as needed.
pub struct WhisperCallbacks<P, S>
where
    P: OfflineWhisperProgressCallback,
    S: OfflineWhisperNewSegmentCallback,
{
    /// Optional progress callback
    pub progress: Option<P>,
    /// Optional new segment callback.
    /// NOTE: this operates at a snapshot level and produces a full representation of the
    /// transcription whenever the new_segment callback fires. This is very expensive and should
    /// not be called frequently:
    /// * Implement [crate::utils::callback::ShortCircuitCallback] or use
    /// [crate::utils::callback::ShortCircuitRibbleWhisperCallback] to provide a mechanism for controlling how often the
    /// snapshotting happens.
    pub new_segment: Option<S>,
}

/// Encapsulates a whisper segment with start and end timestamps
#[derive(Clone)]
pub struct RibbleWhisperSegment {
    /// Segment text
    pub text: Arc<str>,
    /// Timestamp start time, measured in centiseconds
    pub start_time: i64,
    /// Timestamp end time, measured in centiseconds
    pub end_time: i64,
}

impl RibbleWhisperSegment {
    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn replace_text(&mut self, new_text: Arc<str>) {
        self.text = new_text;
    }

    pub fn into_text(self) -> Arc<str> {
        self.text
    }
    pub fn start_timestamp(&self) -> i64 {
        self.start_time
    }

    pub fn end_timestamp(&self) -> i64 {
        self.end_time
    }
}

impl<'a> From<WhisperSegment<'a>> for RibbleWhisperSegment {
    fn from(value: WhisperSegment) -> Self {
        Self {
            text: Arc::from(
                value
                    .to_str_lossy()
                    .expect("This only fails on null pointer. This should not be null pointer")
                    .to_string(),
            ),
            start_time: value.start_timestamp(),
            end_time: value.end_timestamp(),
        }
    }
}

impl<'a> From<&WhisperSegment<'a>> for RibbleWhisperSegment {
    fn from(value: &WhisperSegment<'a>) -> Self {
        Self {
            text: Arc::from(
                value
                    .to_str_lossy()
                    .expect("This only fails on null pointer. This should not be null pointer")
                    .to_string(),
            ),
            start_time: value.start_timestamp(),
            end_time: value.end_timestamp(),
        }
    }
}

/// Encapsulates the state of whisper transcription (confirmed + working segments) at a given point in time
#[derive(Clone, Default)]
pub struct TranscriptionSnapshot {
    confirmed: Arc<str>,
    // This should probably be Arc<[Arc<str>]>
    // Otherwise this is going to involve a lot of string clones.
    string_segments: Arc<[Arc<str>]>,
}
impl TranscriptionSnapshot {
    pub fn new(confirmed: Arc<str>, string_segments: Arc<[Arc<str>]>) -> Self {
        Self {
            confirmed,
            string_segments,
        }
    }

    pub fn confirmed(&self) -> &str {
        &self.confirmed
    }
    pub fn string_segments(&self) -> &[Arc<str>] {
        &self.string_segments
    }

    pub fn into_parts(self) -> (Arc<str>, Arc<[Arc<str>]>) {
        (self.confirmed, self.string_segments)
    }
    pub fn into_string(self) -> String {
        let confirmed = self.confirmed.deref();
        let segment_string = self.string_segments.join(" ");
        format!("{confirmed} {segment_string}")
    }
}

impl std::fmt::Display for TranscriptionSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let confirmed = self.confirmed.deref();
        let segment_string = self.string_segments.join(" ");
        write!(f, "{confirmed} {segment_string}")
    }
}

/// Encapsulates possible types of output sent through a Transcriber channel
/// NOTE: Outputs with accompanying timestamps are not yet implemented.
#[derive(Clone)]
pub enum WhisperOutput {
    TranscriptionSnapshot(Arc<TranscriptionSnapshot>),
    /// For sending running state and control messages from the Transcriber
    ControlPhrase(WhisperControlPhrase),
}

impl WhisperOutput {
    // Consumes and extracts the inner contents of a WhisperOutput into a string
    pub fn into_inner(self) -> String {
        match self {
            WhisperOutput::TranscriptionSnapshot(snapshot) => snapshot.to_string(),
            WhisperOutput::ControlPhrase(control_phrase) => control_phrase.to_string(),
        }
    }
}

/// A set of control phrases to pass information from the transcriber to a UI
// These would benefit from some eventual localization
#[derive(Default, Clone, EnumString, IntoStaticStr, Display)]
pub enum WhisperControlPhrase {
    /// The default "ready" state
    #[default]
    #[strum(serialize = "[IDLE]")]
    Idle,
    /// Preparing whisper for transcription
    #[strum(serialize = "[GETTING_READY]")]
    GettingReady,
    /// Whisper is set up and the transcriber loop is running to decode audio
    #[strum(serialize = "[START SPEAKING]")]
    StartSpeaking,
    /// The transcription time has exceeded its user-specified timeout boundary
    #[strum(serialize = "[TRANSCRIPTION TIMEOUT]")]
    TranscriptionTimeout,
    /// The transcription has fully ended and the final string will be returned
    #[strum(serialize = "[END TRANSCRIPTION]")]
    EndTranscription,
    /// For passing debugging messages across the channel
    #[strum(serialize = "Debug: {0}")]
    Debug(String),
}

pub const WHISPER_SAMPLE_RATE: f64 = 16000f64;

// Quick and dirty utility function for both transcriber objects.
fn build_whisper_context(
    model_location: ModelLocation,
    params: whisper_rs::WhisperContextParameters,
) -> Result<whisper_rs::WhisperContext, RibbleWhisperError> {
    Ok(match model_location {
        ModelLocation::StaticFilePath(path) => {
            whisper_rs::WhisperContext::new_with_params(&path.to_string_lossy(), params)
        }
        ModelLocation::DynamicFilePath(path_buf) => {
            whisper_rs::WhisperContext::new_with_params(&path_buf.to_string_lossy(), params)
        }
        ModelLocation::StaticBuffer(buf) => {
            whisper_rs::WhisperContext::new_from_buffer_with_params(buf, params)
        }
        ModelLocation::DynamicBuffer(buf) => {
            whisper_rs::WhisperContext::new_from_buffer_with_params(&buf, params)
        }
    }?)
}
