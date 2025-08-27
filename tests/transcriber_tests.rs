mod common;
#[cfg(test)]
mod transcriber_tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, LazyLock};
    use std::thread::scope;

    use crate::common::prep_model_bank;
    use ribble_whisper::audio::audio_ring_buffer::AudioRingBuffer;
    #[cfg(feature = "resampler")]
    use ribble_whisper::audio::loading::load_normalized_audio_file;
    use ribble_whisper::audio::{AudioChannelConfiguration, WhisperAudioSample, audio_backend};
    use ribble_whisper::transcriber::offline_transcriber::OfflineTranscriberBuilder;
    use ribble_whisper::transcriber::realtime_transcriber::RealtimeTranscriberBuilder;
    use ribble_whisper::transcriber::vad::Silero;
    use ribble_whisper::transcriber::{
        WHISPER_SAMPLE_RATE, WhisperCallbacks, WhisperOutput, redirect_whisper_logging_to_hooks,
    };
    use ribble_whisper::utils;
    use ribble_whisper::utils::callback::{Nop, RibbleWhisperCallback};
    use ribble_whisper::whisper::configs::{WhisperConfigsV2, WhisperRealtimeConfigs};
    use ribble_whisper::whisper::model::{DefaultModelBank, DefaultModelType};

    // Prepare an audio sample with a known output to try and make conditions as replicable as
    // possible
    static AUDIO_SAMPLE: LazyLock<Arc<[f32]>> = LazyLock::new(|| {
        let sample = load_normalized_audio_file(
            "tests/audio_files/128896__joshenanigans__sentence-recitation.wav",
            None::<fn(usize)>,
        )
        .expect("Test audio should load without issue.");
        match sample {
            WhisperAudioSample::I16(_) => unreachable!(),
            WhisperAudioSample::F32(audio) => audio,
        }
    });

    #[test]
    fn test_realtime_transcriber() {
        let model_type = DefaultModelType::Medium;
        let (model_bank, model_id) = prep_model_bank(model_type);

        let configs = WhisperRealtimeConfigs::default()
            .with_n_threads(8)
            .with_model_id(Some(model_id))
            // Also, optionally set flash attention.
            // Generally keep this on for a performance gain.
            .set_flash_attention(true);

        let (text_sender, text_receiver) = utils::get_channel(32);
        let vad = Silero::try_new_whisper_realtime_default()
            .expect("Silero VAD expected to build without issue");

        let audio_ring_buffer = AudioRingBuffer::default();
        let (transcriber, handle) = RealtimeTranscriberBuilder::<Silero, DefaultModelBank>::new()
            .with_configs(configs.clone())
            .with_audio_buffer(&audio_ring_buffer)
            .with_output_sender(text_sender)
            .with_voice_activity_detector(vad)
            .with_model_retriever(model_bank)
            .build()
            .expect("RealtimeTranscriber expected to build without issues.");

        // The returned string should resemble this one with fewer than roughly 2-3 edits.
        let expected_offline_transcription =
            "Mary has many dreams but can't touch Tennessee by way of flight";

        // Prevent logging to stderr
        redirect_whisper_logging_to_hooks();

        // The audio sample has to be offset by around 206 ms or so to cut out the initial silence
        let start_offset = ((206.0 / 1000.0) * WHISPER_SAMPLE_RATE) as usize;
        // The last half-second or so is basically silent, so chop it off.
        let last_idx_offset = (WHISPER_SAMPLE_RATE / 2.0) as usize;
        let end_audio = AUDIO_SAMPLE.len() - last_idx_offset;

        // Break the audio sample into chunks of size constants::AUDIO_CHUNK_SIZE to simulate default
        // audio input
        let chunks = AUDIO_SAMPLE[start_offset..end_audio].chunks(audio_backend::AUDIO_BUFFER_SIZE);
        let run_transcription = Arc::new(AtomicBool::new(true));

        let transcribed = scope(|s| {
            let t_thread_run_transcription = Arc::clone(&run_transcription);
            let d_thread_run_transcription = Arc::clone(&run_transcription);
            let _a_thread = s.spawn(move || {
                // Clear the audio_ring_buffer
                audio_ring_buffer.clear();

                // Wait until the transcriber is ready to simulate actual speech
                while !handle.ready() {}
                assert!(chunks.len() > 0);
                for chunk in chunks {
                    audio_ring_buffer.push_audio(chunk)
                }
            });
            let t_thread = s.spawn(move || {
                transcriber.run_stream(t_thread_run_transcription, Default::default())
            });
            // Simple thread to just drain the audio - this will sleep until it receives output from
            // the transcriber
            let _d_thread = s.spawn(move || {
                let offline_output_length = expected_offline_transcription.len();
                let epsilon = 5usize;

                // repeatedly drain the audio buffer to prevent a memory panic, and also set the
                // exit condition
                while let Ok(out) = text_receiver.recv() {
                    let message = match out {
                        WhisperOutput::TranscriptionSnapshot(message) => message.to_string(),
                        WhisperOutput::ControlPhrase(_) => "".to_string(),
                    };
                    let current_len = message.len();
                    if current_len > offline_output_length - epsilon {
                        d_thread_run_transcription.store(false, Ordering::Release);
                    }
                }
            });
            // Return the output of the transcriber thread.
            t_thread.join()
        });
        assert!(
            transcribed.is_ok(),
            "Transcription thread panicked: {:?}.",
            transcribed.unwrap_err()
        );

        let transcription = transcribed.unwrap();
        assert!(
            transcription.is_ok(),
            "Transcription returned an error: {}",
            transcription.unwrap_err()
        );

        let transcription = transcription.unwrap();

        // At most, there should only be around 2 edits (inserted punctuation)
        // More than that indicates an error with transcription.
        let max_edit_distance = 2usize;

        let edit_distance = strsim::levenshtein(&transcription, expected_offline_transcription);

        assert!(
            edit_distance <= max_edit_distance,
            "Failed to output reasonable match.\nEdit distance: {}, Max distance: {}\n, Output: {}, Expected: {}",
            edit_distance,
            max_edit_distance,
            transcription,
            expected_offline_transcription
        )
    }

    #[test]
    fn test_offline_segments_callback() {
        let model_type = DefaultModelType::Medium;
        let (model_bank, model_id) = prep_model_bank(model_type);

        let configs = WhisperConfigsV2::default()
            .with_n_threads(8)
            .with_model_id(Some(model_id))
            .set_flash_attention(true);

        // For receiving data in the print loop.
        let (text_sender, text_receiver) = utils::get_channel(32);
        // Since audio is pre-processed (silence removed), it needs to be cloned back into a
        // WhisperAudioSample.
        let audio = WhisperAudioSample::F32(Arc::clone(&AUDIO_SAMPLE));

        let transcriber = OfflineTranscriberBuilder::<Silero, DefaultModelBank>::new()
            .with_configs(configs.clone())
            .with_audio(audio)
            .with_channel_configurations(AudioChannelConfiguration::Mono)
            .with_model_retriever(model_bank)
            .build()
            .expect("Offline transcriber expected to build without issues.");

        // Since the snapshot is being sent to an owned context (1 spot), it's easiest to just move
        // the memory around using a message queue.
        let segment_closure = move |snapshot| {
            text_sender
                .send(snapshot)
                .expect("Receiver should not be deallocated.");
        };

        let segment_callback = RibbleWhisperCallback::new(segment_closure);

        let callbacks = WhisperCallbacks {
            progress: None::<Nop<i32>>,
            new_segment: Some(segment_callback),
        };

        let run_offline_transcription = Arc::new(AtomicBool::new(true));

        // The print string should resemble this one with fewer than roughly 2-3 edits.
        let expected_offline_transcription =
            "Mary has many dreams but can't touch Tennessee by way of flight";

        let t_thread = scope(|s| {
            redirect_whisper_logging_to_hooks();
            let transcription_thread = s.spawn(move || {
                transcriber.process_with_callbacks(run_offline_transcription, callbacks)
            });
            let print_thread = s.spawn(move || {
                let mut transcription = String::new();
                while let Ok(out) = text_receiver.recv() {
                    transcription.push_str(&out);
                    transcription.push_str(" ");
                }
                transcription.trim().to_string()
            });

            (transcription_thread.join(), print_thread.join())
        });

        let (transcription, printed) = t_thread;

        // Check the threads running to completion.
        assert!(
            transcription.is_ok(),
            "Transcription thread panicked: {:?}.",
            transcription.unwrap_err()
        );

        assert!(
            printed.is_ok(),
            "Print thread panicked: {:?}.",
            printed.unwrap_err()
        );

        let transcription = transcription.unwrap();
        assert!(
            transcription.is_ok(),
            "Transcription returned an error: {}",
            transcription.unwrap_err()
        );

        // Unwrap the data for comparison
        let transcription = transcription.unwrap();
        let printed = printed.unwrap();
        // At most, expect there to be around 2 edits (punctuation).
        // More than that indicates an error with transcription.
        let max_edit_distance = 2usize;
        let edit_distance = strsim::levenshtein(&transcription, &printed);
        assert!(
            edit_distance <= max_edit_distance,
            "Failed to output reasonable match.\nEdit distance: {}, Max distance: {}\n, Output: {}, Expected: {}",
            edit_distance,
            max_edit_distance,
            transcription,
            expected_offline_transcription
        )
    }
}
