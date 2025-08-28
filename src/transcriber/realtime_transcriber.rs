use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::{atomic::AtomicBool, atomic::Ordering, Arc};
use std::thread::sleep;
use std::time::{Duration, Instant};
use strsim::jaro_winkler;

use crate::audio::audio_ring_buffer::AudioRingBuffer;
use crate::transcriber::vad::VAD;
use crate::transcriber::{
    build_whisper_context, RibbleWhisperSegment, TranscriptionSnapshot, WhisperControlPhrase,
    WhisperOutput, WHISPER_SAMPLE_RATE,
};
use crate::utils::errors::RibbleWhisperError;
use crate::utils::Sender;
use crate::whisper::configs::WhisperRealtimeConfigs;
use crate::whisper::model::ModelRetriever;
use std::error::Error;

// Conservatively at 90% match
// TODO: possibly bump these to 0.9/0.95 and Med: 0.85
pub const DIFF_THRESHOLD_HIGH: f64 = 0.9;
pub const DIFF_THRESHOLD_MED: f64 = 0.85;
pub const DIFF_THRESHOLD_LOW: f64 = 0.75;
pub const TIMESTAMP_GAP: i64 = 1000;
pub const TIMESTAMP_EPSILON: i64 = 10;

// TEST THIS TO SEE IF THE DEDUP IS A BIT FASTER ON ONLY THE LAST SEGMENT
// IF THAT'S THE CASE, DON'T RUN IT OVER A WORKING SET SIZE OF 3
// FOR RESOLVING WORD BOUNDARIES, IT SHOULD PROBABLY BE THE LAST SEGMENT OR SO ANYWAY...?

// TODO: find a good "close-enough" happy medium window.
// Sometimes 5 is good, sometimes it's terrible. If someone is prone to repeating themselves, it will cause issues.
// But perhaps that is not quite something I can solve.

// This seems to be a reasonably good combination that balances word-boundaries.
// This is possibly too high.
// This needs to be tested a little bit more--it is more than likely, at most going to have 1-2
// duplicate words from the blending, so running over 5 risks clobbering the output a little too hard.
pub const N_TOKENS: usize = 5;

// 400ms seems to be the most accurate so far -- perhaps tweaking the tokens down is a little better.
// It's better to let ~1-2 words get duplicated (but similar enough), vs being cautious and risking a bad next-segment.
pub const RETAIN_MS: usize = 500;

// This is an artifact from the old implementation--but I believe whisper resolves across ~3
// segments before it "confirms", so this is just going to main here until it really needs to
// change.
pub const N_SEGMENTS_DIFF: usize = 3;
// This does not need to be that big.
pub const WORKING_SET_SIZE: usize = N_SEGMENTS_DIFF * 2;
pub const PAUSE_DURATION: u64 = 100;

pub const N_SAMPLES_30S: usize = ((1e-3 * 30000.0) * WHISPER_SAMPLE_RATE) as usize;
// This could probably be a little shorter
const VAD_TIMEOUT_MS: u128 = 1500;

// TODO: do some investigation -> try and locate a full-segment duplication to set a breakpoint:
// Try to find the moments where what I think are "hallucinations" are being hallucinated
// Turning off context seems to fully eliminate the problem, but it could be covering up the real problem
// The issue is very difficult to reproduce reliably, so I'm not quite sure what to look for just yet.

/// Builder for [RealtimeTranscriber]
/// All fields are necessary and thus required to successfully build a RealtimeTranscriber.
/// Multiple VAD implementations have been provided, see: [crate::transcriber::vad]
/// Silero: [crate::transcriber::vad::Silero] is recommended for accuracy, but is very sensitive to background noise.
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
    /// **NOTE: Trying to use this VAD in 2 places simultaneously will result in lock contention**
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

impl<V, M> RealtimeTranscriber<V, M>
where
    V: VAD<f32>,
    M: ModelRetriever,
{
    fn send_snapshot(&self, confirmed: Arc<str>, segments: &VecDeque<RibbleWhisperSegment>) {
        let string_segments = segments
            .iter()
            .map(|segment| segment.text.clone())
            .collect();
        let snapshot = Arc::new(TranscriptionSnapshot::new(confirmed, string_segments));

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

    fn send_control_phrase(&self, control_phrase: WhisperControlPhrase) {
        // Extract the control phrase type if there's an error/would-block.
        let control_phrase_type = match &control_phrase {
            WhisperControlPhrase::Debug(..) => "Debug",
            _ => control_phrase.clone().into(),
        };

        if let Err(e) = self
            .output_sender
            .try_send(WhisperOutput::ControlPhrase(control_phrase))
        {
            #[cfg(feature = "ribble-logging")]
            {
                log::warn!(
                    "Error sending control phrase: {control_phrase_type} \n\
                    Error: {}
                    Error source: {:#?}",
                    &e,
                    e.source()
                );
            }
            #[cfg(not(feature = "ribble-logging"))]
            {
                eprintln!(
                    "Error sending control phrase: {control_phrase_type} \n\
                    Error: {}
                    Error source: {:#?}",
                    &e,
                    e.source()
                )
            }
        }
    }

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

    pub fn run_stream(
        &self,
        run_transcription: Arc<AtomicBool>,
        slow_stop: Arc<AtomicBool>,
    ) -> Result<String, RibbleWhisperError> {
        // Alert the UI
        self.send_control_phrase(WhisperControlPhrase::GettingReady);

        // Set up whisper
        let full_params = self.configs.as_whisper_full_params();

        let whisper_context_params = self.configs.as_whisper_context_params();

        // Since it's not possible to build a realtime transcriber, there must be an ID; it's fine to unwrap.
        let model_id = self.configs.model_id().unwrap();

        let model_location = self.model_retriever.retrieve_model(model_id).ok_or(
            RibbleWhisperError::ParameterError(format!("Failed to find model: {model_id}")),
        )?;

        // Set up a whisper context
        let ctx = build_whisper_context(model_location, whisper_context_params)?;

        let mut whisper_state = ctx.create_state()?;
        self.ready.store(true, Ordering::Release);
        self.send_control_phrase(WhisperControlPhrase::StartSpeaking);

        // Set up remaining loop data.

        // This is a relic from the old implementation--the time check could and should be simplified.

        // Instant marker for timekeeping.
        let mut t_last = Instant::now();
        // For timing the transcription (and timeout)
        let mut total_time = 0u128;

        // To collect audio from the ring buffer.
        let mut audio_samples: Vec<f32> = vec![0f32; N_SAMPLES_30S];

        // For collecting the transcribed segments to return a full transcription at the end
        // NOTE: since this implementation is read-heavy, Arc<str> is used over a preallocated string
        // to reduce the cost of cloning.
        let mut output_string: Arc<str> = Default::default();
        let mut working_set: VecDeque<RibbleWhisperSegment> =
            VecDeque::with_capacity(WORKING_SET_SIZE);

        // If voice is detected early but there's not enough data to run whisper, this flag should
        // be set to guarantee inference happens after a pause.
        let mut skip_vad_run_inference = false;
        let mut run_segment_merge = false;

        let mut previous_pause_clear_buffer = false;

        // TODO: this is probably causing problems -> set to false and remove this variable.
        let mut use_context = false;

        // NOTE: so, instants don't seem to be the right way to test things.
        // It seems to be triggering before 1 second has passed.
        let mut vad_timeout_start_instant = None;

        let min_sample_len = self.configs.min_sample_len();

        while run_transcription.load(Ordering::Acquire) {
            let t_now = Instant::now();
            let diff = t_now - t_last;
            let millis = diff.as_millis();
            total_time += millis;

            // To prevent accidental audio clearing, hold off to ensure at least
            // vad_sample_len() ms have passed before trying to detect voice.
            // This gives the audio some time to collect in-between this loop and when the user is
            // alerted to start speaking.
            if millis < self.configs.vad_sample_len() as u128 {
                sleep(Duration::from_millis(PAUSE_DURATION));
                continue;
            }

            t_last = t_now;

            // read_into will return min(requested_len, audio_len)
            // It will also escape early if the buffer is length 0
            self.audio_feed
                .read_into(self.configs.vad_sample_len(), &mut audio_samples);

            let vad_size =
                (self.configs.vad_sample_len() as f64 / 1000f64 * WHISPER_SAMPLE_RATE) as usize;

            // If there's not enough samples yet to perform VAD, just skip the loop.
            // Sleeping may or may not be required/beneficial; this has not been tested
            // The spinlock might produce better results.
            if audio_samples.len() < vad_size {
                continue;
            }

            let pause_detected = if !skip_vad_run_inference {
                let voice_detected = self.vad.lock().voice_detected(&audio_samples);
                if !voice_detected {
                    let vad_t_now = Instant::now();

                    // Sometimes Silero can just fail...
                    // Also: fans/background noise can throw it off badly; whisper can usually get
                    // speech from a bad signal -> YMMV, WebRtc might work better.
                    if vad_timeout_start_instant.is_none() {
                        vad_timeout_start_instant = Some(vad_t_now);
                    }

                    let timeout_start_instant = vad_timeout_start_instant.unwrap();

                    if vad_t_now.duration_since(timeout_start_instant).as_millis() < VAD_TIMEOUT_MS
                    {
                        #[cfg(debug_assertions)]
                        self.send_control_phrase(WhisperControlPhrase::Debug(
                            "PAUSE TIMEOUT TICKING".to_string(),
                        ));
                        // Run the VAD check again to test for silence.
                        continue;
                    }

                    #[cfg(debug_assertions)]
                    self.send_control_phrase(WhisperControlPhrase::Debug(
                        "PAUSE DETECTED".to_string(),
                    ));

                    // This means inference has been run at least 1 last time and the dedup has run
                    // I think I might be baking this incorrectly.
                    if previous_pause_clear_buffer {
                        #[cfg(debug_assertions)]
                        self.send_control_phrase(WhisperControlPhrase::Debug(
                            "PAUSE TIMEOUT: CLEARING BUFFER".to_string(),
                        ));

                        self.audio_feed.clear();

                        #[cfg(debug_assertions)]
                        self.send_control_phrase(WhisperControlPhrase::Debug(
                            "RUNNING OUTPUT DEDUP".to_string(),
                        ));

                        output_string = confirm_transcription(output_string, &mut working_set);
                        self.send_snapshot(Arc::clone(&output_string), &working_set);

                        run_segment_merge = false;
                        // RESET the VAD timeout so it doesn't get stuck in a clearing loop.
                        vad_timeout_start_instant = None;
                        continue;
                    }
                    previous_pause_clear_buffer = true;
                    true
                } else {
                    previous_pause_clear_buffer = false;
                    false
                }
            } else {
                // If the inference needs to be run, avoid early-clearing the buffer.
                previous_pause_clear_buffer = false;
                false
            };

            if !pause_detected {
                vad_timeout_start_instant = None;
            }

            // Read the audio buffer in chunks of audio_sample_len
            self.audio_feed
                .read_into(self.configs.audio_sample_len_ms(), &mut audio_samples);

            // Depending on the buffering strategy, this will hold off on running the decode loop
            // excessively at the cost of some latency.
            if audio_samples.len() < min_sample_len {
                // Skip over the next VAD
                // This will also skip over the clearing.
                skip_vad_run_inference = true;
                continue;
            }

            #[cfg(debug_assertions)]
            {
                let inference_msg = if pause_detected {
                    "INFERENCE AFTER PAUSE"
                } else {
                    "RUNNING INFERENCE"
                };

                self.send_control_phrase(WhisperControlPhrase::Debug(inference_msg.to_string()));
            }

            let mut params = full_params.clone();
            params.set_no_context(!use_context);

            let _ = whisper_state.full(params, &audio_samples)?;
            let num_segments = whisper_state.full_n_segments();

            if num_segments == 0 {
                #[cfg(debug_assertions)]
                self.send_control_phrase(WhisperControlPhrase::Debug("NO SEGMENTS".to_string()));
                // TODO: test for excess cycle burning on low hardware -- sleeping might be beneficial.
                continue;
            }

            skip_vad_run_inference = false;

            // If there's a null pointer, just skip over the segment
            // Expect that to happen extremely rarely-to-never.
            let mut segments = whisper_state.as_iter().flat_map(|ws| ws.try_into());

            if !run_segment_merge {
                use_context = false;
                let audio_len = self.audio_feed.get_audio_length_ms();
                // TODO: this can be retained before the loop starts.
                // ONCE THE BUG IS FIXED, MOVE THIS HIGHER UP.
                // ALSO, JUST USE CAPACITY - LEN, NO NEED FOR MS -> it forces atomics and cpu time to do this comparison with ms.
                let capacity_len = self.audio_feed.get_capacity_in_ms();
                run_segment_merge = audio_len >= capacity_len;

                // If the "differ" should be run on the next pass, clear the audio, push the entire audio buffer to the working set,
                // And expect the differ to run on the next pass.
                if run_segment_merge {
                    // TODO: determine whether to actually keep ~300 ms -> in practice, this does sometimes chop off words..
                    // It might even be better to do 400-500 ms with the deduplication.
                    self.audio_feed.clear_from_back_retain_ms(RETAIN_MS);

                    working_set.clear();
                    working_set.extend(segments);
                    use_context = true;

                    #[cfg(debug_assertions)]
                    {
                        // TODO: remove this later.
                        // I'm not sure -where- the overwrite is happening, but I think the audio length is getting overwritten.
                        let check_audio_len = self.audio_feed.get_audio_length_ms();
                        debug_assert!(check_audio_len < capacity_len);
                    }
                } else {
                    working_set.clear();
                    working_set.extend(segments);
                }
            } else {
                #[cfg(debug_assertions)]
                self.send_control_phrase(WhisperControlPhrase::Debug(
                    "RUNNING SEGMENT BLEND".to_string(),
                ));

                let last_segment = working_set.iter_mut().last();
                let first_new_segment = segments.next();

                // If there's no old segment (somehow), then there's no need to diff.
                // If there's no new segment, then there's also no need to diff -> the next iteration is going to clobber the segments anyway.
                // if let Some(last_seg) = last_segment
                //     && let Some(new_seg) = first_new_segment
                // {
                //     blend_segments(last_seg, &new_seg);
                // }

                match (last_segment, first_new_segment) {
                    (Some(last_seg), Some(new_seg)) => {
                        #[cfg(debug_assertions)]
                        {
                            // TODO: remove this later.
                            // I'm not sure -where- the overwrite is happening, but I think the audio length is getting overwritten.
                            let check_audio_len = self.audio_feed.get_audio_length_ms();
                            let check_capacity_len = self.audio_feed.get_capacity_in_ms();
                            debug_assert!(
                                check_audio_len < check_capacity_len,
                                "BUFFER LIKELY OVERWRITTEN."
                            );
                        }

                        // TODO: REMOVE THIS AFTER DIAGNOSING THE PROBLEME.
                        // -- if it doesn't happen here, then look at the other marked spots.
                        // Maybe this needs to leverage the message queues.
                        #[cfg(debug_assertions)]
                        {
                            let test_jaro = jaro_winkler(last_seg.text(), new_seg.text());
                            // These will throw on a segment context hallucination.
                            // I think the problem might be here, and due to context.

                            if test_jaro >= DIFF_THRESHOLD_HIGH {
                                let out_str = format!(
                                    "PROBLEM! SCORE: {test_jaro}\nLAST: {}\nNEW{}",
                                    last_seg.text(),
                                    new_seg.text()
                                );
                                eprintln!("{out_str}");
                                panic!("HALLUCINATION MOST LIKELY: {out_str}");
                            }

                            if test_jaro >= DIFF_THRESHOLD_MED {
                                let out_str = format!(
                                    "PROBLEM! SCORE: {test_jaro}\nLAST: {}\nNEW{}",
                                    last_seg.text(),
                                    new_seg.text()
                                );
                                eprintln!("{out_str}");
                                panic!("HALLUCINATION MOST LIKELY: {out_str}");
                            }

                            if test_jaro >= DIFF_THRESHOLD_LOW {
                                let out_str = format!(
                                    "PROBLEM! SCORE: {test_jaro}\nLAST: {}\nNEW{}",
                                    last_seg.text(),
                                    new_seg.text()
                                );
                                eprintln!("{out_str}");
                                panic!("HALLUCINATION MOST LIKELY: {out_str}");
                            }
                        }

                        blend_segments(last_seg, &new_seg);
                    }

                    // If the working set has just been cleared (pauses, etc.)
                    // Push the data to the working set and skip onto the next iteration.
                    // In the case where this is being run as a last-pass before
                    (None, Some(new_seg)) => {
                        // I -THINK- this is necessary?
                        // It is possibly not and possibly the cause of the sporadic duplications.
                        // TODO: investigate this further.
                        working_set.push_back(new_seg);
                        working_set.extend(segments);
                        run_segment_merge = false;
                        use_context = false;
                        continue;
                    }

                    // The final 2 cases (Some, None) = (None, None) = just proceed with
                    // the rest of the confirmation.
                    (_, _) => {}
                }

                if !working_set.is_empty() {
                    #[cfg(debug_assertions)]
                    self.send_control_phrase(WhisperControlPhrase::Debug(
                        "RUNNING DEDUP AFTER BLEND".to_string(),
                    ));

                    output_string = confirm_transcription(output_string, &mut working_set);
                }

                run_segment_merge = false;

                // Once the "differ" has been run to blend the segments, don't use previous context
                // to inform the transcription to prevent any artifacts.
                use_context = false;
            }

            // Drain the working set when it exceeds its bounded size. It is most likely that the
            // n segments drained are actually part of the transcription.
            // It is highly, highly unlikely for this condition to ever trigger, given that
            // the VAD implementations are generally pretty good at detecting pauses.
            // It is most likely that the working set will get drained beforehand, but this is a
            // fallback to ensure the working_set bounded to WORKING_SET_SIZE
            if working_set.len() > WORKING_SET_SIZE {
                #[cfg(debug_assertions)]
                self.send_control_phrase(WhisperControlPhrase::Debug(
                    "BAKING_WORKING_SET".to_string(),
                ));
                let up_to = working_set.len().saturating_sub(WORKING_SET_SIZE);
                let mut confirm_from = working_set.drain(..up_to).collect();

                output_string = confirm_transcription(output_string, &mut confirm_from);
            }

            // Send the current transcription as it exists, so that the UI can update.
            // Since the working set is updated after every run of the inference/differ/buffer
            // clear, and there are earlier skips to avoid running inference, it can generally be
            // assumed that each inference = needs snapshot.
            let push_snapshot = !(output_string.trim().is_empty() && working_set.is_empty());

            if push_snapshot {
                self.send_snapshot(Arc::clone(&output_string), &working_set);
            }

            // If the timeout is set to 0, this loop runs infinitely.
            if self.configs.realtime_timeout() == 0 {
                continue;
            }

            // Otherwise check for timeout.
            if total_time > self.configs.realtime_timeout() {
                self.send_control_phrase(WhisperControlPhrase::TranscriptionTimeout);

                run_transcription.store(false, Ordering::Release);
            }
        }

        if slow_stop.load(Ordering::Acquire) {
            self.send_control_phrase(WhisperControlPhrase::SlowStop);
            // This can just consume full params
            if whisper_state.full(full_params, &audio_samples).is_ok() {
                let mut segments = whisper_state.as_iter().flat_map(|ws| ws.try_into());
                if run_segment_merge {
                    let last_segment = working_set.iter_mut().last();
                    let first_new_segment: Option<RibbleWhisperSegment> = segments.next();

                    match (last_segment, first_new_segment) {
                        (Some(l_seg), Some(mut r_seg)) => {
                            let (l_str, r_str) =
                                match deduplicate_strings(l_seg.text(), r_seg.text()) {
                                    None => (Arc::clone(&l_seg.text), Arc::clone(&r_seg.text)),
                                    Some((new_l_str, new_r_str)) => {
                                        (Arc::from(new_l_str.trim()), Arc::from(new_r_str.trim()))
                                    }
                                };

                            l_seg.replace_text(l_str);
                            r_seg.replace_text(r_str);

                            working_set.push_back(r_seg);
                            working_set.extend(segments);
                        }

                        // If the run_segment_merge happens after the working set has recently been cleared, somehow,
                        // then push any new segments and let the deduplication take care of resolving the last boundary.
                        (None, Some(r_seg)) => {
                            working_set.push_back(r_seg);
                            working_set.extend(segments);
                        }

                        // If both are none, then both sets are empty and this is a Nop.
                        // If last_segment.is_some(), and segments is empty, this is a Nop
                        (_, _) => working_set.extend(segments),
                    }
                } else {
                    working_set.clear();
                    working_set.extend(segments);
                }
            }
        }
        self.send_control_phrase(WhisperControlPhrase::EndTranscription);

        // Clean up the whisper context
        drop(whisper_state);
        drop(ctx);

        // Drain the last of the working set,
        // deduplicate any possible duplicate words from greedy segment
        // overlapping/transcription errors.
        #[cfg(debug_assertions)]
        self.send_control_phrase(WhisperControlPhrase::Debug(
            "RUNNING FINAL OUTPUT DEDUP".to_string(),
        ));

        output_string = confirm_transcription(output_string, &mut working_set);
        // Set internal state to non-ready in case the transcriber is going to be reused
        self.ready.store(false, Ordering::Release);

        // Strip remaining whitespace and return
        Ok(output_string.trim().to_string())
    }
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

fn find_closest_match(buf1: &[&str], buf2: &[&str]) -> Option<(usize, usize)> {
    let mut l_match = None;
    let mut r_match = None;
    for (idx, l_token) in buf1.iter().enumerate() {
        let mut max_score = 0.0;
        for (jdx, r_token) in buf2.iter().enumerate() {
            let similar = jaro_winkler(l_token, r_token);
            // Take greater-equal the greatest score in-case there's a lot of repeating going on in the actual speech.
            if similar >= DIFF_THRESHOLD_HIGH && similar >= max_score {
                l_match = Some(idx);
                r_match = Some(jdx);
                max_score = similar;
            }
        }
    }
    Some((l_match?, r_match?))
}

// This could be done with slices and just return the offsets, but it's easier to just write this
// imperatively.
fn run_stride(
    buf1: &[&str],
    buf1_start: usize,
    buf2: &[&str],
    buf2_start: usize,
) -> (usize, usize) {
    let mut l_start = buf1_start;
    let mut r_start = buf2_start;

    loop {
        let l_token = buf1.get(l_start);
        let r_token = buf2.get(r_start);
        if l_token.is_none() || r_token.is_none() {
            break;
        }
        let similar = jaro_winkler(l_token.unwrap(), r_token.unwrap());
        // PERHAPS this should be ~0.85-0.9, 0.8 is a little low I think.
        // TODO: possibly swap to high.
        if similar < DIFF_THRESHOLD_MED {
            break;
        }
        l_start += 1;
        r_start += 1;
    }
    (l_start, r_start)
}

// SO: this is working well for the most part, but it is triggering on some false positives.

// EDGE CASE: repeated word, closest match.
// If a closest match happens and the stride l_end - l_start (or r_end - r_start) = 0 and it's
// -not- at the end of the left half, then it's very unlikely to be an actual match.

// This runs right-side priority--since this is to catch words that are potentially duplicated, they're
// most likely going to have better punctuation. Sometimes whisper will insert punctuation on the
// left hand side when it doesn't have enough audio--this helps to mitigate that.
fn deduplicate_strings(str1: &str, str2: &str) -> Option<(String, String)> {
    let (mut l_buf, mut r_buf) = split_text(str1, str2);
    let l_start = if l_buf.len() == N_TOKENS + 1 { 1 } else { 0 };
    let r_end = N_TOKENS.min(r_buf.len());
    find_closest_match(&l_buf[l_start..], &r_buf[..r_end]).and_then(|(l_match, r_match)| {
        // If there are more than 5 tokens, the l_buf is compared from 1 instead of 0;
        // The index needs to be decremented by one.
        let l_match_start = if l_start == 1 {
            l_match.saturating_add(1).min(l_buf.len() - 1)
        } else {
            l_match
        };
        // For sanity's sake, double-check that this is correct.
        debug_assert!(l_buf.get(l_match_start).is_some());
        debug_assert!(r_buf.get(r_match).is_some());
        debug_assert!(jaro_winkler(l_buf[l_match_start], r_buf[r_match]) >= DIFF_THRESHOLD_HIGH);
        let (l_end, r_end) = run_stride(&l_buf, l_match_start, &r_buf, r_match);

        let num_words = l_end.saturating_sub(l_match_start);

        if num_words < 2 {
            // So, if this is catching only one word, make sure it's toward the -end- of the buffer.
            // Otherwise, it's more-than-likely a false positive.
            // TODO: strictly-end is too strict, some duplications get through.
            // HOWEVER, it might be the case where the match is also too far down the new string...
            // EITHER: Reduce the number of tokens compared (likely bad idea),
            // OR: Add a second check to make sure the r_end is toward the start of the string
            // PERHAPS, it is better to loosely match on the midpoint of both.
            if l_end < l_buf.len().saturating_sub(2) {
                // This is to test out the algorithm thus far to see that things are working as expected.
                eprintln!("EARLY MATCH");
                return None;
            } else {
                // TODO: remove this branch when testing done
                // May still have artifacts.
                eprintln!("MAYBE NOT AN EARLY MATCH?");
            }
        }

        // Confirm up to just before the end of the match on the left.
        l_buf.truncate(l_end);
        // Drop up to just before the end of the match on the right.
        let up_to = (r_end).min(r_buf.len());
        drop(r_buf.drain(..up_to));

        Some((l_buf.join(" "), r_buf.join(" ")))
    })
}

fn split_text<'a>(str1: &'a str, str2: &'a str) -> (Vec<&'a str>, Vec<&'a str>) {
    let mut l_buf = str1.rsplitn(N_TOKENS + 1, " ").collect::<Vec<_>>();
    l_buf.reverse();
    let r_buf = str2.splitn(N_TOKENS + 1, " ").collect::<Vec<_>>();
    (l_buf, r_buf)
}

// NOTE: this is doing left priority in-case words end up cut off.
fn blend_segments(l_segment: &mut RibbleWhisperSegment, r_segment: &RibbleWhisperSegment) {
    let (mut l_buf, mut r_buf) = split_text(l_segment.text.as_ref(), r_segment.text.as_ref());
    let l_start = if l_buf.len() == N_TOKENS + 1 { 1 } else { 0 };
    let r_end = N_TOKENS.min(r_buf.len());
    let last_is_word = r_buf.len() <= N_TOKENS;

    if let Some((l_match, r_match)) = find_closest_match(&l_buf[l_start..], &r_buf[..r_end]) {
        // If there are more than 5 tokens, the l_buf is compared from 1 instead of 0;
        // The index needs to be decremented by one.
        let l_match_start = if l_start == 1 {
            l_match.saturating_add(1).min(l_buf.len() - 1)
        } else {
            l_match
        };
        // For sanity's sake, double-check that this is correct.
        debug_assert!(l_buf.get(l_match_start).is_some());
        debug_assert!(r_buf.get(r_match).is_some());
        debug_assert!(jaro_winkler(l_buf[l_match_start], r_buf[r_match]) >= DIFF_THRESHOLD_HIGH);
        let (l_end, _r_end) = run_stride(&l_buf, l_match_start, &r_buf, r_match);

        let num_words = l_end.saturating_sub(l_match_start);

        if num_words < 2 {
            if l_end < l_buf.len().saturating_sub(2) {
                // This is to test out the algorithm thus far to see that things are working as expected.
                eprintln!("EARLY MATCH");
                return;
            } else {
                eprintln!("MAYBE NOT AN EARLY MATCH?");
            }
        }
        // Confirm up to the end of the match on the left.
        l_buf.truncate(l_end + 1);

        // Drop up to the end of the match on the right.
        let up_to = (r_end + 1).min(r_buf.len());
        drop(r_buf.drain(..up_to));

        // If the buffer is still full, it either has the rest of the segment, or it has a word
        // If it has a word, it's either the boundary word, or a duplicate that will get deduplicated.
        // If it has the rest of the segment,
        if !r_buf.is_empty() && last_is_word {
            // Since the last element is just the rest of the string
            // (and it's going to be dropped anyway), just swap-remove.
            l_buf.push(r_buf.swap_remove(0));
        }

        l_segment.replace_text(Arc::from(l_buf.join(" ").trim()))
    }
}

fn confirm_transcription(
    output_string: Arc<str>,
    working_set: &mut VecDeque<RibbleWhisperSegment>,
) -> Arc<str> {
    if output_string.trim().is_empty() {
        Arc::from(
            working_set
                .drain(..)
                .map(|seg| seg.into_text())
                .collect::<Vec<_>>()
                .join(" ")
                .trim(),
        )
    } else {
        match working_set.pop_front() {
            None => output_string,
            Some(segment) => match deduplicate_strings(output_string.as_ref(), segment.text()) {
                None => {
                    let mut deduped = format!("{output_string} {}", segment.text());
                    let remaining = working_set
                        .drain(..)
                        .map(|seg| seg.into_text())
                        .collect::<Vec<_>>()
                        .join(" ");
                    deduped.push(' ');
                    deduped.push_str(&remaining);
                    Arc::from(deduped.trim())
                }
                Some((mut deduped, rest)) => {
                    deduped.push(' ');
                    deduped.push_str(&rest);
                    let remaining = working_set
                        .drain(..)
                        .map(|seg| seg.into_text())
                        .collect::<Vec<_>>()
                        .join(" ");
                    deduped.push(' ');
                    deduped.push_str(&remaining);
                    Arc::from(deduped.trim())
                }
            },
        }
    }
}
