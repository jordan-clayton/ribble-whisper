use std::ops::Deref;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use strum::{Display, EnumString, IntoStaticStr};

use crate::utils::callback::{Callback, ShortCircuitCallback};
use crate::utils::errors::RibbleWhisperError;

pub mod offline_transcriber;
pub mod realtime_transcriber;
pub mod vad;

// Trait alias, used until the feature reaches stable
pub trait OfflineWhisperProgressCallback: Callback<Argument = i32> + Send + Sync + 'static {}
impl<T: Callback<Argument = i32> + Send + Sync + 'static> OfflineWhisperProgressCallback for T {}

// Trait alias, used until the feature reaches stable
pub trait OfflineWhisperNewSegmentCallback:
    ShortCircuitCallback<Argument = TranscriptionSnapshot> + Send + Sync + 'static
{
}
impl<T: ShortCircuitCallback<Argument = TranscriptionSnapshot> + Send + Sync + 'static>
    OfflineWhisperNewSegmentCallback for T
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
pub struct WhisperSegment {
    /// Segment text
    pub text: String,
    /// Timestamp start time, measured in 0.1 ms
    pub start_time: i64,
    /// Timestamp end time, measured in 0.1 ms
    pub end_time: i64,
}

/// Encapsulates the state of whisper transcription (confirmed + working segments) at a given point in time
#[derive(Clone, Default)]
pub struct TranscriptionSnapshot {
    confirmed: Arc<String>,
    string_segments: Arc<[String]>,
}
impl TranscriptionSnapshot {
    pub fn new(confirmed: Arc<String>, string_segments: Arc<[String]>) -> Self {
        Self {
            confirmed,
            string_segments,
        }
    }

    pub fn confirmed(&self) -> &str {
        &self.confirmed
    }
    pub fn string_segments(&self) -> &[String] {
        &self.string_segments
    }

    pub fn into_parts(self) -> (Arc<String>, Arc<[String]>) {
        (self.confirmed, self.string_segments)
    }
    pub fn into_string(self) -> String {
        let mut confirmed = self.confirmed.deref().clone();
        confirmed.extend(self.string_segments.iter().cloned());
        confirmed
    }
}

impl std::fmt::Display for TranscriptionSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut confirmed = self.confirmed.deref().clone();
        confirmed.extend(self.string_segments.iter().cloned());
        write!(f, "{confirmed}")
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
