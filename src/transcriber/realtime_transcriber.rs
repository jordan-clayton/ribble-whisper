use parking_lot::Mutex;
use std::collections::VecDeque;
use std::ops::Deref;
use std::sync::{atomic::AtomicBool, atomic::Ordering, Arc};
use std::thread::sleep;
use std::time::{Duration, Instant};
use strsim::jaro_winkler;

use crate::audio::audio_ring_buffer::AudioRingBuffer;
use crate::transcriber::vad::VAD;
use crate::transcriber::{
    Transcriber, TranscriptionSnapshot, WhisperControlPhrase, WhisperOutput, WhisperSegment,
    WHISPER_SAMPLE_RATE,
};
use crate::utils::errors::RibbleWhisperError;
use crate::utils::Sender;
use crate::whisper::configs::WhisperRealtimeConfigs;
use crate::whisper::model::ModelRetriever;
use std::error::Error;

/// Builder for [RealtimeTranscriber]
/// All fields are necessary and thus required to successfully build a RealtimeTranscriber.
/// Multiple VAD implementations have been provided, see: [crate::transcriber::vad]
/// Silero: [crate::transcriber::vad::Silero] is recommended for accuracy.
/// See: examples/realtime_transcriber.rs for example usage.
pub struct RealtimeTranscriberBuilder<V, M>
where
    V: VAD<f32>,
    M: ModelRetriever,
{
    configs: Option<Arc<WhisperRealtimeConfigs>>,
    audio_buffer: Option<AudioRingBuffer<f32>>,
    output_sender: Option<Sender<WhisperOutput>>,
    model_retriever: Option<Arc<M>>,
    voice_activity_detector: Option<Arc<Mutex<V>>>,
}

impl<V, M> RealtimeTranscriberBuilder<V, M>
where
    V: VAD<f32>,
    M: ModelRetriever,
{
    pub fn new() -> Self {
        Self {
            configs: None,
            audio_buffer: None,
            output_sender: None,
            model_retriever: None,
            voice_activity_detector: None,
        }
    }

    /// Set configurations.
    pub fn with_configs(mut self, configs: WhisperRealtimeConfigs) -> Self {
        self.configs = Some(Arc::new(configs));
        self
    }
    /// Set the (shared) AudioRingBuffer.
    pub fn with_audio_buffer(mut self, audio_buffer: &AudioRingBuffer<f32>) -> Self {
        self.audio_buffer = Some(audio_buffer.clone());
        self
    }

    /// Set the output sender.
    pub fn with_output_sender(mut self, sender: Sender<WhisperOutput>) -> Self {
        self.output_sender = Some(sender);
        self
    }

    // For setting the model retriever; for handling grabbing the model path
    // (e.g. from a shared bank)
    pub fn with_model_retriever<M2: ModelRetriever>(
        self,
        model_retriever: M2,
    ) -> RealtimeTranscriberBuilder<V, M2> {
        RealtimeTranscriberBuilder {
            configs: self.configs,
            audio_buffer: self.audio_buffer,
            output_sender: self.output_sender,
            model_retriever: Some(Arc::new(model_retriever)),
            voice_activity_detector: self.voice_activity_detector,
        }
    }

    // For setting a shared model retriever; for handling grabbing the model path
    // (e.g. from a shared bank)
    pub fn with_shared_model_retriever<M2: ModelRetriever>(
        self,
        model_retriever: Arc<M2>,
    ) -> RealtimeTranscriberBuilder<V, M2> {
        RealtimeTranscriberBuilder {
            configs: self.configs,
            audio_buffer: self.audio_buffer,
            output_sender: self.output_sender,
            model_retriever: Some(Arc::clone(&model_retriever)),
            voice_activity_detector: self.voice_activity_detector,
        }
    }

    /// Set the voice activity detector.
    pub fn with_voice_activity_detector<V2: VAD<f32> + Sync + Send>(
        self,
        vad: V2,
    ) -> RealtimeTranscriberBuilder<V2, M> {
        let voice_activity_detector = Some(Arc::new(Mutex::new(vad)));
        RealtimeTranscriberBuilder {
            configs: self.configs,
            audio_buffer: self.audio_buffer,
            output_sender: self.output_sender,
            model_retriever: self.model_retriever,
            voice_activity_detector,
        }
    }
    /// Set the voice activity detector to a shared VAD, (e.g. pre-allocated).
    /// **NOTE: Trying to use this VAD in 2 places simultaneously will result in significant lock contention.**
    /// **NOTE: VADs must be reset before being used in a different context**
    pub fn with_shared_voice_activity_detector<V2: VAD<f32> + Sync + Send>(
        self,
        vad: Arc<Mutex<V2>>,
    ) -> RealtimeTranscriberBuilder<V2, M> {
        RealtimeTranscriberBuilder {
            configs: self.configs,
            audio_buffer: self.audio_buffer,
            output_sender: self.output_sender,
            model_retriever: self.model_retriever,
            voice_activity_detector: Some(Arc::clone(&vad)),
        }
    }

    /// This returns a tuple struct containing both the transcriber object and a handle to check the
    /// transcriber's ready state from another location.
    /// Returns Err when a parameter is missing.
    pub fn build(
        self,
    ) -> Result<(RealtimeTranscriber<V, M>, RealtimeTranscriberHandle), RibbleWhisperError> {
        let configs = self.configs.ok_or(RibbleWhisperError::ParameterError(
            "Configs missing in RealtimeTranscriberBuilder.".to_string(),
        ))?;

        let model_retriever = self
            .model_retriever
            .ok_or(RibbleWhisperError::ParameterError(
                "Model retriever missing in RealtimeTranscriberBuilder.".to_string(),
            ))?;

        let _model_id = configs
            .model_id()
            .ok_or(RibbleWhisperError::ParameterError(
                "Configs are missing model ID in RealtimeTranscriberBuilder.".to_string(),
            ))?;

        let audio_feed = self.audio_buffer.ok_or(RibbleWhisperError::ParameterError(
            "Audio feed missing in RealtimeTranscriberBuilder".to_string(),
        ))?;
        let output_sender = self
            .output_sender
            .ok_or(RibbleWhisperError::ParameterError(
                "Output sender missing in RealtimeTranscriberBuilder".to_string(),
            ))?;
        let vad = self
            .voice_activity_detector
            .ok_or(RibbleWhisperError::ParameterError(
                "Voice activity detector missing in RealtimeTranscriberBuilder.".to_string(),
            ))?;
        let ready = Arc::new(AtomicBool::new(false));

        let handle = RealtimeTranscriberHandle {
            ready: Arc::clone(&ready),
        };
        let transcriber = RealtimeTranscriber {
            configs,
            audio_feed,
            output_sender,
            ready,
            model_retriever,
            vad,
        };
        Ok((transcriber, handle))
    }
}

/// A realtime whisper transcription runner. See: examples/realtime_stream.rs for suggested use
/// RealtimeTranscriber cannot be shared across threads because it has a singular ready state.
/// It is also infeasible to call [Transcriber::process_audio] in parallel due
/// to the cost of running whisper.
pub struct RealtimeTranscriber<V, M>
where
    V: VAD<f32>,
    M: ModelRetriever,
{
    configs: Arc<WhisperRealtimeConfigs>,
    /// The shared input buffer from which samples are pulled for transcription
    audio_feed: AudioRingBuffer<f32>,
    /// For sending output to a UI
    output_sender: Sender<WhisperOutput>,
    /// Ready flag.
    /// A RealtimeTranscriber is considered to be ready when all of its whisper initialization has completed,
    /// and it is about to enter its transcription loop.
    /// NOTE: This cannot be accessed directly, because RealtimeTranscriber is not Sync.
    /// Use a [RealtimeTranscriberHandle] to check the ready state.
    ready: Arc<AtomicBool>,
    /// For obtaining a model's file path based on an ID stored in [WhisperRealtimeConfigs].
    model_retriever: Arc<M>,
    /// For voice detection
    vad: Arc<Mutex<V>>,
}

impl<V, M> Default for RealtimeTranscriberBuilder<V, M>
where
    V: VAD<f32>,
    M: ModelRetriever,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<V, M> Transcriber for RealtimeTranscriber<V, M>
where
    V: VAD<f32>,
    M: ModelRetriever,
{
    // This streaming implementation uses a sliding window + VAD + diffing approach to approximate
    // a continuous audio file. This will only start transcribing segments when voice is detected.
    // Its accuracy isn't bulletproof (and highly depends on the model), but it's reasonably fast
    // on average hardware.
    // GPU processing is more or less a necessity for running realtime; this will not work well using CPU inference.
    //
    // This implementation is synchronous and can be run on a single thread--however, due to the
    // bounded channel, it is recommended to process in parallel/spawn a worker to drain the data channel
    //
    // Argument:
    // - run_transcription: an atomic state flag so that the transcriber can be terminated from another location
    // e.g. UI
    fn process_audio(
        &self,
        run_transcription: Arc<AtomicBool>,
    ) -> Result<String, RibbleWhisperError> {
        // Alert the UI
        if let Err(e) = self.output_sender.try_send(WhisperOutput::ControlPhrase(
            WhisperControlPhrase::GettingReady,
        )) {
            #[cfg(feature = "ribble-logging")]
            {
                log::warn!("Error sending getting-ready snapshot: {:#?}", e.source());
            }
            #[cfg(not(feature = "ribble-logging"))]
            {
                eprintln!("Error sending getting-ready snapshot: {:#?}", e.source())
            }
        }

        let mut t_last = Instant::now();

        // To collect audio from the ring buffer.
        let mut audio_samples: Vec<f32> = vec![0f32; N_SAMPLES_30S];

        // For timing the transcription (and timeout)
        let mut total_time = 0u128;

        // For collecting the transcribed segments to return a full transcription at the end
        let mut output_string = Arc::new(String::default());
        let mut working_set: VecDeque<WhisperSegment> = VecDeque::with_capacity(WORKING_SET_SIZE);

        // Set up whisper
        let full_params = self.configs.to_whisper_full_params();
        let whisper_context_params = self.configs.to_whisper_context_params();

        // Since it's not possible to build a realtime transcriber, there must be an ID; it's fine to unwrap.
        let model_id = self.configs.model_id().unwrap();

        let model_path = self.model_retriever.retrieve_model_path(model_id).ok_or(
            RibbleWhisperError::ParameterError(format!("Failed to find model: {model_id}")),
        )?;

        let ctx = whisper_rs::WhisperContext::new_with_params(
            &model_path.to_string_lossy(),
            whisper_context_params,
        )?;

        let mut whisper_state = ctx.create_state()?;
        self.ready.store(true, Ordering::Release);
        if let Err(e) = self.output_sender.send(WhisperOutput::ControlPhrase(
            WhisperControlPhrase::StartSpeaking,
        )) {
            #[cfg(feature = "ribble-logging")]
            {
                log::warn!("Error sending start-speaking snapshot: {:#?}", e.source())
            }
            #[cfg(not(feature = "ribble-logging"))]
            {
                eprintln!("Error sending start-speaking snapshot: {:#?}", e.source())
            }
        }
        while run_transcription.load(Ordering::Acquire) {
            let t_now = Instant::now();
            let diff = t_now - t_last;
            let millis = diff.as_millis();
            total_time += millis;

            // To prevent accidental audio clearing, hold off to ensure at least
            // vad_sample_len ms have passed before trying to detect voice.
            if millis < self.configs.vad_sample_len() as u128 {
                sleep(Duration::from_millis(PAUSE_DURATION));
                continue;
            }

            // read_into will return min(requested_len, audio_len)
            // It will also escape early if the buffer is length 0
            self.audio_feed
                .read_into(self.configs.vad_sample_len(), &mut audio_samples);

            let vad_size =
                (self.configs.vad_sample_len() as f64 / 1000f64 * WHISPER_SAMPLE_RATE) as usize;

            // If the buffer has recently been cleared/there's not enough data to send to the voice detector,
            // sleep for a little bit longer.
            if audio_samples.len() < vad_size {
                sleep(Duration::from_millis(PAUSE_DURATION));
                continue;
            }

            // Check for voice activity
            // In case the audio needs to be cleared, record the amount of time for VAD + lock
            // contention, so that audio isn't fully lost.
            let voice_detected = self.vad.lock().voice_detected(&audio_samples);
            if !voice_detected {
                // DEBUGGING.
                if let Err(e) = self.output_sender.try_send(WhisperOutput::ControlPhrase(
                    WhisperControlPhrase::Debug("PAUSE DETECTED".to_string()),
                )) {
                    #[cfg(feature = "ribble-logging")]
                    {
                        eprintln!("Error sending pause debug phrase: {:#?}", e.source())
                    }
                    #[cfg(not(feature = "ribble-logging"))]
                    {
                        eprintln!("Error sending debug phrase: {:#?}", e.source())
                    }
                };

                // Drain the dequeue and push to the confirmed output_string
                let next_output = working_set.drain(..).map(|output| output.text);
                let mut new_out = output_string.deref().clone();
                new_out.extend(next_output);
                output_string = Arc::new(new_out);

                // Clear the audio buffer to prevent data incoherence messing up the transcription.
                // Since VAD + clearing takes up a small amount of time, keep diff ms of audio in
                // case speech has resumed.
                self.audio_feed.clear();

                // Sleep for a little bit to give the buffer time to fill up
                sleep(Duration::from_millis(PAUSE_DURATION));
                // Jump to the next iteration.
                continue;
            }

            // DEBUGGING -> just ignore these in the print loop if undesired.
            if let Err(e) = self.output_sender.try_send(WhisperOutput::ControlPhrase(
                WhisperControlPhrase::Debug("RUNNING INFERENCE".to_string()),
            )) {
                #[cfg(feature = "ribble-logging")]
                {
                    log::warn!("Error sending inference debug phrase: {:#?}", e.source());
                }
                #[cfg(not(feature = "ribble-logging"))]
                {
                    eprintln!("Error sending inference debug phrase: {:#?}", e.source());
                }
            };

            // Update the time (for timeout)
            t_last = t_now;

            // Read the audio buffer in chunks of audio_sample_len
            self.audio_feed
                .read_into(self.configs.audio_sample_len_ms(), &mut audio_samples);

            let _ = whisper_state.full(full_params.clone(), &audio_samples)?;
            let num_segments = whisper_state.full_n_segments()?;
            if num_segments == 0 {
                continue;
            }

            // It's not a big deal if there are invalid utf-8 characters
            // Use lossy to just swap it with the replacement character
            let mut segments = (0..num_segments)
                .map(|i| {
                    let text = whisper_state.full_get_segment_text_lossy(i)?;
                    let start_time = whisper_state.full_get_segment_t0(i)? * 10;
                    let end_time = whisper_state.full_get_segment_t1(i)? * 10;
                    Ok(WhisperSegment {
                        text,
                        start_time,
                        end_time,
                    })
                })
                .collect::<Result<Vec<WhisperSegment>, whisper_rs::WhisperError>>()?;

            // If the working set is empty, push the segments into the working set.
            // i.e. This should only happen on first run.
            if working_set.is_empty() {
                working_set.extend(segments.drain(..));
            }
            // Otherwise, run the diffing algorithm
            // This is admittedly a little haphazard, but it seems good enough and can tolerate
            // long sentences reasonably well. It is likely that a phrase will finish and get detected
            // by the VAD well before issues are encountered.
            // There are small chances of false negatives (duplicated output), and false positives (clobbering)
            // These tend to happen with abnormal speech patterns (extra long sentence length), strange prosody, and the like.

            // In the worst cases, the entire working set can decay, but it is rare and very difficult to trigger
            // because of how whisper works.
            else {
                // Run a diff over the last N_SEGMENTS of the working set and the new segments and try
                // to resolve overlap.
                let old_segments = working_set.make_contiguous();
                // Get the tail N_SEGMENTS
                let old_len = old_segments.len();
                let tail = old_segments[old_len.saturating_sub(N_SEGMENTS_DIFF)..].iter_mut();

                let head = segments
                    .drain(..N_SEGMENTS_DIFF.min(segments.len()))
                    .collect::<Vec<_>>();

                // For collecting swaps
                let mut swap_indices = Vec::with_capacity(N_SEGMENTS_DIFF);

                // This is Amortized O(1), with an upper bound of constants::N_SEGMENTS_DIFF * constants::N_SEGMENTS_DIFF iterations
                // In practice this is very unlikely to hit that upper bound.
                for old_segment in tail {
                    // This might be a little conservative, but it's better to be safe.
                    // It is expected that if there is a good match, it's 1:1 on each of the timestamps
                    let mut best_score = 0.0;
                    let mut best_match = None::<&WhisperSegment>;
                    // This is out of bounds, but it should always be swapped if there's a best_match
                    let mut best_index = N_SEGMENTS_DIFF;
                    // Get the head N_SEGMENTS
                    for (index, new_segment) in head.iter().enumerate() {
                        // With the way that segments are being output, it seems to work a little better
                        // If when comparing timestamps, to match on starting alignment.
                        // The size of the working set is small enough such that the
                        let time_gap = (old_segment.start_time - new_segment.start_time).abs();
                        let old_lower = old_segment.text.to_lowercase();
                        let new_lower = new_segment.text.to_lowercase();
                        let similar = jaro_winkler(&old_lower, &new_lower);
                        // If it's within the same alignment, it's likely for the segments to be
                        // a match (i.e. the audio buffer has recently been cleared, and this is a new window)
                        // Compare based on similarity to confirm.

                        // If the timestamp is close enough such that it's lower than the epsilon: (10 ms)
                        // Consider it to be a 1:1 match.
                        let timestamp_close =
                            time_gap < TIMESTAMP_EPSILON && similar > DIFF_THRESHOLD_MIN;
                        let compare_score = if time_gap <= TIMESTAMP_GAP {
                            timestamp_close || {
                                if similar >= DIFF_THRESHOLD_MED {
                                    true
                                } else if similar >= DIFF_THRESHOLD_LOW {
                                    best_score < DIFF_THRESHOLD_MED || best_match.is_none()
                                } else if similar > DIFF_THRESHOLD_MIN {
                                    best_score < DIFF_THRESHOLD_LOW || best_match.is_none()
                                } else {
                                    false
                                }
                            }
                        } else {
                            // Otherwise, if it's outside the timestamp gap, only match on likely probability
                            // High matches indicate close segments (i.e. a word/phrase boundary)
                            if similar >= DIFF_THRESHOLD_HIGH {
                                true
                            } else if similar >= DIFF_THRESHOLD_MED {
                                best_score < DIFF_THRESHOLD_HIGH || best_match.is_none()
                            } else if similar >= DIFF_THRESHOLD_LOW {
                                best_score < DIFF_THRESHOLD_MED || best_match.is_none()
                            } else {
                                false
                            }
                        };

                        if compare_score && similar > best_score {
                            best_score = similar;
                            best_match = Some(new_segment);
                            best_index = index;
                        }
                    }
                    if let Some(new_seg) = best_match {
                        // Swap the longer of the two segments in hopes that false positives do not clobber the output
                        // And also that it is the most semantically correct output. Anticipate and expect a little crunchiness.
                        if new_seg.text.len() > old_segment.text.len() {
                            *old_segment = new_seg.clone();
                        }
                        assert_ne!(best_index, N_SEGMENTS_DIFF);
                        swap_indices.push(best_index)
                    }
                }

                // This is Amortized O(1), with an upper bound of constants::N_SEGMENTS_DIFF * constants::N_SEGMENTS_DIFF iterations
                // In practice this is very unlikely to hit that upper bound.
                for (index, new_seg) in head.into_iter().enumerate() {
                    if !swap_indices.contains(&index) {
                        // If the segment was never swapped in, treat it as a new segment and push it to the output buffer
                        // It is not impossible for this to break sequence, but it is highly unlikely.
                        working_set.push_back(new_seg)
                    }
                }
            }
            // If there are any remaining segments, drain them into the working set.
            working_set.extend(segments.drain(..));

            // Drain the working set when it exceeds its bounded size. It is most likely that the
            // n segments drained are actually part of the transcription.
            // It is highly, highly unlikely for this condition to ever trigger, given that VAD are
            // generally pretty good at detecting pauses.
            // It is most likely that the working set will get drained beforehand, but this is a
            // fallback to ensure the working_set is always WORKING_SET_SIZE
            if working_set.len() > WORKING_SET_SIZE {
                let next_text = working_set
                    .drain(0..working_set.len().saturating_sub(WORKING_SET_SIZE))
                    .map(|segment| segment.text);
                let mut new_out = output_string.deref().clone();
                new_out.extend(next_text);
                // Push to the output string.
                output_string = Arc::new(new_out);
            }

            // Send the current transcription as it exists, so that the UI can update
            if !output_string.is_empty() || !working_set.is_empty() {
                let snapshot = Arc::new(TranscriptionSnapshot::new(
                    Arc::clone(&output_string),
                    Arc::from(
                        working_set
                            .iter()
                            .map(|segment| segment.text.clone())
                            .collect::<Vec<_>>(),
                    ),
                ));
                if let Err(e) = self
                    .output_sender
                    .try_send(WhisperOutput::TranscriptionSnapshot(snapshot))
                {
                    #[cfg(feature = "ribble-logging")]
                    {
                        log::warn!(
                            "Error sending transcription-snapshot mid loop: {:#?}",
                            e.source()
                        )
                    }
                    #[cfg(not(feature = "ribble-logging"))]
                    {
                        eprintln!(
                            "Error sending transcription-snapshot mid loop: {:#?}",
                            e.source()
                        )
                    }
                }
            }

            // If the timeout is set to 0, this loop runs infinitely.
            if self.configs.realtime_timeout() == 0 {
                continue;
            }

            // Otherwise check for timeout.
            if total_time > self.configs.realtime_timeout() {
                if let Err(e) = self.output_sender.try_send(WhisperOutput::ControlPhrase(
                    WhisperControlPhrase::TranscriptionTimeout,
                )) {
                    #[cfg(feature = "ribble-logging")]
                    {
                        log::warn!("Error sending timeout control phrase: {:#?}", e.source())
                    }
                    #[cfg(not(feature = "ribble-logging"))]
                    {
                        eprintln!(
                            "Error sending end-of-transcription control phrase: {:#?}",
                            e.source()
                        )
                    }
                }

                run_transcription.store(false, Ordering::Release);
            }
        }

        if let Err(e) = self.output_sender.send(WhisperOutput::ControlPhrase(
            WhisperControlPhrase::EndTranscription,
        )) {
            #[cfg(feature = "ribble-logging")]
            {
                log::warn!(
                    "Error sending end-of-transcription control phrase: {:#?}",
                    e.source()
                )
            }
            #[cfg(not(feature = "ribble-logging"))]
            {
                eprintln!(
                    "Error sending end-of-transcription control phrase: {:#?}",
                    e.source()
                )
            }
        }

        // Clean up the whisper context
        drop(whisper_state);
        drop(ctx);

        // Drain the last of the working set.
        let next_text = working_set.drain(..).map(|segment| segment.text);
        let mut final_out = output_string.deref().clone();
        final_out.extend(next_text);

        // Set internal state to non-ready in case the transcriber is going to be reused
        self.ready.store(false, Ordering::Release);
        // Strip remaining whitespace and return
        Ok(final_out.trim().to_string())
    }
}

/// A simple handle that allows for checking the ready state of a RealtimeTranscriber from another
/// location (e.g. a different thread).
#[derive(Clone)]
pub struct RealtimeTranscriberHandle {
    ready: Arc<AtomicBool>,
}

impl RealtimeTranscriberHandle {
    pub fn ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }
}

// Conservatively at 90% match
pub const DIFF_THRESHOLD_HIGH: f64 = 0.9;
pub const DIFF_THRESHOLD_MED: f64 = 0.7;
pub const DIFF_THRESHOLD_LOW: f64 = 0.50;
pub const DIFF_THRESHOLD_MIN: f64 = 0.40;
pub const TIMESTAMP_GAP: i64 = 3000;
pub const TIMESTAMP_EPSILON: i64 = 10;
pub const N_SEGMENTS_DIFF: usize = 3;
pub const WORKING_SET_SIZE: usize = N_SEGMENTS_DIFF * 5;
pub const PAUSE_DURATION: u64 = 100;
pub const N_SAMPLES_30S: usize = ((1e-3 * 30000.0) * WHISPER_SAMPLE_RATE) as usize;
