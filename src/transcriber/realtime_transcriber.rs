use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::{atomic::AtomicBool, atomic::Ordering, Arc};
use std::thread::sleep;
use std::time::{Duration, Instant};
use strsim::jaro_winkler;

use crate::audio::audio_ring_buffer::AudioRingBuffer;
use crate::transcriber::vad::VAD;
use crate::transcriber::{
    build_whisper_context, RibbleWhisperSegment, Transcriber, TranscriptionSnapshot,
    WhisperControlPhrase, WhisperOutput, WHISPER_SAMPLE_RATE,
};
use crate::utils::errors::RibbleWhisperError;
use crate::utils::Sender;
use crate::whisper::configs::WhisperRealtimeConfigs;
use crate::whisper::model::ModelRetriever;
use std::error::Error;

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

// TODO: clean up extraneous variables used for debugger watching (or possibly move to a separate branch)
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
        self.send_control_phrase(WhisperControlPhrase::GettingReady);

        let mut t_last = Instant::now();

        // NOTE: so, instants don't seem to be the right way to test things.
        // It seems to be triggering before 1 second has passed.
        let mut vad_timeout_start_instant = None;

        // To collect audio from the ring buffer.
        let mut audio_samples: Vec<f32> = vec![0f32; N_SAMPLES_30S];

        // For timing the transcription (and timeout)
        // This could just use duration objects instead of accumulating millis.
        let mut total_time = 0u128;

        // For collecting the transcribed segments to return a full transcription at the end
        let mut output_string: Arc<str> = Default::default();
        let mut working_set: VecDeque<RibbleWhisperSegment> =
            VecDeque::with_capacity(WORKING_SET_SIZE);

        // If voice is detected early but there's not enough data to run whisper, this flag should
        // be set to guarantee inference happens after a pause.
        let mut skip_vad_run_inference = false;
        let mut run_differ = false;

        // Set up whisper
        let full_params = self.configs.to_whisper_full_params();

        let whisper_context_params = self.configs.to_whisper_context_params();

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

        // TODO: clean these up and get rid of them before merge.
        #[cfg(debug_assertions)]
        let mut dedup_swaps = 0usize;

        #[cfg(debug_assertions)]
        let mut working_set_exceeded = 0usize;

        #[cfg(debug_assertions)]
        let mut max_num_segments = 0usize;

        #[cfg(debug_assertions)]
        let mut max_working_set_size = 0usize;

        let mut previous_pause_clear_buffer = false;

        #[cfg(debug_assertions)]
        let mut num_differ_runs = 0usize;

        // This actually seems to be helpful for resolving things across word boundaries, ONLY after a clear.
        let mut use_context = false;

        while run_transcription.load(Ordering::Acquire) {
            let t_now = Instant::now();
            let diff = t_now - t_last;
            let millis = diff.as_millis();
            total_time += millis;

            // To prevent accidental audio clearing, hold off to ensure at least
            // vad_sample_len() ms have passed before trying to detect voice.
            // In case the audio backend isn't quite up to speed with
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
                        {
                            self.send_control_phrase(WhisperControlPhrase::Debug(
                                "PAUSE TIMEOUT TICKING".to_string(),
                            ));
                        }
                        // Run the VAD check again to test for silence.
                        continue;
                    }

                    #[cfg(debug_assertions)]
                    {
                        self.send_control_phrase(WhisperControlPhrase::Debug(
                            "PAUSE DETECTED".to_string(),
                        ));
                    }

                    // This means inference has been run at least 1 last time and the dedup has run
                    // I think I might be baking this incorrectly.
                    if previous_pause_clear_buffer {
                        #[cfg(debug_assertions)]
                        {
                            self.send_control_phrase(WhisperControlPhrase::Debug(
                                "CLEARING BUFFER".to_string(),
                            ));
                        }

                        // This -should- bake the working set here.
                        // If the audio is cleared, the previous output might accidentally be
                        // similar enough to cause issues with the deduping

                        // This -could- alter the full_params to run with context to inform the next segments
                        // But first this really needs to be tinkered out.

                        let next_text = working_set.drain(..).map(|segment| segment.into_text());
                        let last_text = next_text.collect::<Vec<_>>().join(" ");
                        let next_out = if output_string.trim().is_empty() {
                            last_text
                        } else {
                            format!("{} {last_text}", output_string.trim())
                        };

                        output_string = Arc::from(next_out.trim());
                        self.audio_feed.clear();
                        self.send_snapshot(Arc::clone(&output_string), &working_set);

                        // Use context to inform the next transcription
                        use_context = true;
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

            // This probably shouldn't ever happen (with the new changes).
            if audio_samples.len() < AUDIO_MIN_LEN {
                // Skip over the next VAD
                // This will also skip over the clearing.
                skip_vad_run_inference = true;
                continue;
            }

            #[cfg(debug_assertions)]
            {
                // DEBUGGING -> just ignore these in the print loop if undesired.
                let inference_msg = if pause_detected {
                    "INFERENCE AFTER PAUSE"
                } else {
                    "RUNNING INFERENCE"
                };

                self.send_control_phrase(WhisperControlPhrase::Debug(inference_msg.to_string()));
            }

            // TESTING THIS OUT A MIN
            let mut params = full_params.clone();
            params.set_no_context(!use_context);

            let _ = whisper_state.full(params, &audio_samples)?;
            let num_segments = whisper_state.full_n_segments();

            #[cfg(debug_assertions)]
            {
                max_num_segments = max_num_segments.max(num_segments as usize);
            }

            if num_segments == 0 {
                #[cfg(debug_assertions)]
                {
                    self.send_control_phrase(WhisperControlPhrase::Debug(
                        "NO SEGMENTS".to_string(),
                    ));
                }

                // TODO: determine whether or not this should sleep -> This branch is unlikely to
                // be taken in most cases.
                // It is difficult to trigger in practice.
                sleep(Duration::from_millis(PAUSE_DURATION));
                continue;
            }

            skip_vad_run_inference = false;

            // If there's a null pointer, just skip over the segment
            // Expect that to happen extremely rarely-to-never.
            let mut segments = whisper_state.as_iter().flat_map(|ws| {
                let text = ws.to_str_lossy()?;
                let start_time = ws.start_timestamp();
                let end_time = ws.end_timestamp();
                // This needs explicit type and the turbofish flat-map would be way too gnarly.
                let res: Result<RibbleWhisperSegment, RibbleWhisperError> =
                    Ok(RibbleWhisperSegment {
                        text: text.into(),
                        start_time,
                        end_time,
                    });
                res
            });

            // POSSIBLY run a clear before the last diff.
            if !run_differ {
                use_context = false;
                run_differ =
                    self.audio_feed.get_audio_length_ms() >= self.audio_feed.get_capacity_in_ms();

                // If the differ should be run on the next pass, clear the audio, push the entire audio buffer to the working set,
                // And expect the differ to run on the next pass.
                if run_differ {
                    self.audio_feed.clear();
                    working_set.clear();
                    working_set.extend(segments);
                    use_context = true;
                } else {
                    working_set.clear();
                    working_set.extend(segments);
                }
            } else {
                // This is a best-effort de-dup that examines the last n<=5 tokens (words) of the
                // currently last segment, and the first m<=5 tokens (words) of the first new segment.
                // A quadratic search + jaro_winkler is used to fuzzy match the closest overlap
                // (~.90). If there is an overlap, take the next w<= 2 words, append it to the old
                // text and clear ms <= w * CLEAR_MS from the audio buffer to re-start the window.
                //
                // This seems to be generally enough to chop out the newly appended boundary words
                // so that whisper remains coherent.
                //
                // This seems generally "good enough" to resolve boundaries. It might be improved
                // with continuous timestamping, or with additional improvements to the algorithm thus far.
                #[cfg(debug_assertions)]
                {
                    num_differ_runs += 1;
                }

                #[cfg(debug_assertions)]
                {
                    self.send_control_phrase(WhisperControlPhrase::Debug(
                        "RUNNING DEDUP".to_string(),
                    ));
                }

                let last_segment = working_set.iter_mut().last();
                let first_new_segment = segments.next();

                // If there's no old segment (somehow), then there's no need to diff.
                // If there's no new segment, then there's also no need to diff -> the next iteration is going to clobber the segments anyway.
                if let Some(last_seg) = last_segment
                    && let Some(new_seg) = first_new_segment
                {
                    // Find the first common prefix by comparing the first N tokens.
                    let mut old_text = last_seg.text.split(" ").collect::<Vec<_>>();
                    let new_text = new_seg.text.split(" ").collect::<Vec<_>>();

                    let mut old_boundary = None;
                    let mut new_boundary = None;

                    let old_start = old_text.len().saturating_sub(N_TOKENS);
                    let new_end = N_TOKENS.min(new_text.len());

                    // PERHAPS, since this is using jaro-winkler, it should maybe find the common
                    // prefix by strings of size 2+
                    // It might also be wise to reconsider the indexing business if doing so.
                    // A prefix of 2 strings is most likely going to be more accurate.
                    //
                    //enumerate(): `(idx, <item>)`, `old_text.iter().enumerate().skip(old_start)`

                    for (idx, token) in old_text.iter().enumerate().skip(old_start) {
                        let mut max_score = 0.0;
                        for (jdx, new_token) in new_text.iter().enumerate().take(new_end) {
                            // I'm -not- quite sure whether this could really optimize for the "first"
                            // without false positives.
                            let similar = jaro_winkler(token, new_token);
                            // Perhaps levenstein is better... not sure.
                            if similar >= DIFF_THRESHOLD_HIGH && similar > max_score {
                                old_boundary = Some(idx);
                                new_boundary = Some(jdx);
                                max_score = similar;
                            }
                        }
                    }

                    if let Some(swap_idx) = old_boundary
                        && let Some(new_idx) = new_boundary
                        && swap_idx < old_text.len() - 1
                        && new_idx < new_text.len() - 1
                    {
                        #[cfg(debug_assertions)]
                        {
                            dedup_swaps += 1;
                        }
                        let new_start = new_idx + 1;
                        let new_end = (new_start + 2).min(new_text.len() - 1);

                        // So, this should -probably- be around 300ms * num_words,
                        // but I'm not quite sure just yet, or whether this is going to work out as planned.

                        // Since I don't have timestamp information, I'm going out on a limb here.
                        // The boundary
                        let num_words = new_end.saturating_sub(new_start);
                        old_text.truncate(swap_idx + 1);
                        old_text.extend_from_slice(&new_text[new_start..=new_end]);
                        last_seg.replace_text(Arc::from(old_text.join(" ").trim()));

                        self.audio_feed.clear_n_samples(CLEAR_MS * num_words);
                    }
                }

                // Bake the working set into the confirmed string.
                let confirmed_text = working_set
                    .drain(..)
                    .map(|seg| seg.into_text())
                    .collect::<Vec<_>>()
                    .join(" ");
                let new_text = if output_string.trim().is_empty() {
                    confirmed_text
                } else {
                    format!("{} {confirmed_text}", output_string.trim())
                };

                output_string = Arc::from(new_text.trim());

                // If there are any remaining segments, drain them into the working set.
                // If the dedup somehow happens to happen again, -or- if a pause is detected,
                // the remaining segments need to be in the working set.
                working_set.extend(segments);

                run_differ = false;

                // Once the differ has been run, don't use previous context to inform the transcription.
                // The sliding window is going to continue transcribing over a ~10-20 ms buffer and it's likely to
                // get a little spicy.
                use_context = false;
            }

            #[cfg(debug_assertions)]
            {
                max_working_set_size = max_working_set_size.max(working_set.len());
            }

            // Drain the working set when it exceeds its bounded size. It is most likely that the
            // n segments drained are actually part of the transcription.
            // It is highly, highly unlikely for this condition to ever trigger, given that VAD are
            // generally pretty good at detecting pauses.
            // It is most likely that the working set will get drained beforehand, but this is a
            // fallback to ensure the working_set is always WORKING_SET_SIZE
            if working_set.len() > WORKING_SET_SIZE {
                #[cfg(debug_assertions)]
                {
                    self.send_control_phrase(WhisperControlPhrase::Debug(
                        "BAKING_WORKING_SET".to_string(),
                    ));
                }

                #[cfg(debug_assertions)]
                {
                    working_set_exceeded += 1;
                }

                let next_text = working_set
                    .drain(0..working_set.len().saturating_sub(WORKING_SET_SIZE))
                    .map(|segment| segment.into_text());

                let next_text_string = next_text.collect::<Vec<_>>().join(" ");
                // Push to the output string.
                let next_out = if output_string.trim().is_empty() {
                    next_text_string
                } else {
                    format!("{output_string} {next_text_string}")
                };

                output_string = Arc::from(next_out.trim());
            }

            // Send the current transcription as it exists, so that the UI can update.
            // Since the working set is updated after every run of the inference/differ/buffer
            //
            // clear, and there are earlier skips to avoid running inference, it can generally be
            // assumed that each inference = needs snapshot
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
        self.send_control_phrase(WhisperControlPhrase::EndTranscription);

        // Clean up the whisper context
        drop(whisper_state);
        drop(ctx);

        // Drain the last of the working set.
        let next_text = working_set.drain(..).map(|segment| segment.into_text());
        let last_text = next_text.collect::<Vec<_>>().join(" ");

        let final_out = if output_string.trim().is_empty() {
            last_text
        } else {
            format!("{output_string} {last_text}")
        };
        // Set internal state to non-ready in case the transcriber is going to be reused
        self.ready.store(false, Ordering::Release);

        // TODO: remove this once the diffing has been sorted out

        #[cfg(debug_assertions)]
        {
            eprintln!("BREAKPOINT TO LOOK AT TALLY VALUES.");
        }

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
pub const DIFF_THRESHOLD_LOW: f64 = 0.65;
pub const TIMESTAMP_GAP: i64 = 1000;
pub const TIMESTAMP_EPSILON: i64 = 10;

// TEST THIS TO SEE IF THE DEDUP IS A BIT FASTER ON ONLY THE LAST SEGMENT
// IF THAT'S THE CASE, DON'T RUN IT OVER A WORKING SET SIZE OF 3
// FOR RESOLVING WORD BOUNDARIES, IT SHOULD PROBABLY BE THE LAST SEGMENT OR SO ANYWAY...?

pub const N_TOKENS: usize = 5;
pub const N_SEGMENTS_DIFF: usize = 3;
pub const WORKING_SET_SIZE: usize = N_SEGMENTS_DIFF * 5;
pub const PAUSE_DURATION: u64 = 100;

// This might be a little bit too high?
// The dedup is pretty rare with a large-enough buffer size that it seems unlikely
// Perhaps this is treated as "best effort" and hope that natural speech patterns are enough to
// get a "reasonably natural" stream.
const CLEAR_MS: usize = 100;
pub const N_SAMPLES_30S: usize = ((1e-3 * 30000.0) * WHISPER_SAMPLE_RATE) as usize;
const VAD_TIMEOUT_MS: u128 = 1500;
const AUDIO_MIN_LEN: usize = WHISPER_SAMPLE_RATE as usize;
