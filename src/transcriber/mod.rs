use std::ops::Deref;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

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

// TODO: this does not need to be a trait.
// Move the description to the realtime_transcriber and add a flag for slow-stop.
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

/// Encapsulates various whisper callbacks which can be set before running transcription
/// Other callbacks will be added as needed.
#[derive(Default)]
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

impl<'a> TryFrom<WhisperSegment<'a>> for RibbleWhisperSegment {
    type Error = RibbleWhisperError;
    fn try_from(value: WhisperSegment) -> Result<Self, Self::Error> {
        let text = value.to_str_lossy()?;
        let start_time = value.start_timestamp();
        let end_time = value.end_timestamp();
        Ok(Self {
            text: text.into(),
            start_time,
            end_time,
        })
    }
}

impl<'a> TryFrom<&WhisperSegment<'a>> for RibbleWhisperSegment {
    type Error = RibbleWhisperError;

    fn try_from(value: &WhisperSegment<'a>) -> Result<Self, Self::Error> {
        let text = value.to_str_lossy()?;
        let start_time = value.start_timestamp();
        let end_time = value.end_timestamp();
        Ok(Self {
            text: text.into(),
            start_time,
            end_time,
        })
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
    #[strum(serialize = "[CLEANING UP]")]
    SlowStop,
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
