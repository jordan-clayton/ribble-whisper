#[cfg(test)]
#[cfg(feature = "resampler")]
mod resampler_test {
    // NOTE: You will need to supply your own audio file and modify the tests accordingly.

    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    use hound::{SampleFormat, WavSpec, WavWriter};

    use ribble_whisper::audio::loading::load_normalized_audio_file;
    use ribble_whisper::audio::resampler::file_needs_normalizing;
    use ribble_whisper::audio::{AudioChannelConfiguration, WhisperAudioSample};
    use ribble_whisper::transcriber::offline_transcriber::OfflineTranscriberBuilder;
    use ribble_whisper::transcriber::vad::Silero;
    use ribble_whisper::transcriber::Transcriber;
    use ribble_whisper::utils::constants;
    use ribble_whisper::whisper::configs::WhisperConfigsV2;
    use ribble_whisper::whisper::model::DefaultModelType;

    // Tests the resampling from a file path, which will also implicitly using the track handle
    #[test]
    fn test_needs_resampling() {
        let needs_normalizing = file_needs_normalizing(
            "tests/audio_files/128896__joshenanigans__sentence-recitation.wav",
        )
        .unwrap();

        assert!(needs_normalizing);
    }

    #[test]
    fn test_needs_resampling_mp3() {
        let needs_normalizing = file_needs_normalizing("tests/audio_files/test_mp3.mp3").unwrap();

        assert!(needs_normalizing);
    }
    // Loads some audio at 44.1 khz, resamples it to 16kHz, then writes it to an output file.
    // The audio will need to be checked manually to ensure the integrity
    #[test]
    fn test_resample() {
        let audio = load_normalized_audio_file(
            "tests/audio_files/128896__joshenanigans__sentence-recitation.wav",
            None::<fn(usize)>,
        )
        .unwrap();
        let wav_spec = WavSpec {
            bits_per_sample: 32,
            channels: 1,
            sample_format: SampleFormat::Float,
            sample_rate: constants::WHISPER_SAMPLE_RATE as u32,
        };

        let mut writer = WavWriter::create("tests/audio_files/resampled.wav", wav_spec).unwrap();
        let WhisperAudioSample::F32(samples) = audio else {
            unreachable!()
        };
        for sample in samples.iter() {
            writer.write_sample(*sample).unwrap();
        }
        writer.finalize().unwrap();
    }

    // Loads some audio at 44.1 khz, resamples it to 16kHz, then sends the audio to whisper to
    // transcribe. The audio is simple enough such that the transcription should be 1:1
    // A successful transcription means the resampling is correct.
    #[test]
    fn test_resample_whisper() {
        // Punctuation (commas, periods) can sometimes be detected/not-detected based on settings
        // and the output of the model is nondeterministic.
        // It does do a good enough job of getting the words, so the transcription is stripped of
        // punctuation to ensure the outputs match exactly most of the time.
        let expected_transcription =
            "Mary has many dreams but can't touch Tennessee by way of flight";

        let audio = load_normalized_audio_file(
            "tests/audio_files/128896__joshenanigans__sentence-recitation.wav",
            None::<fn(usize)>,
        )
        .unwrap();

        // This presumes a model is already downloaded. Handle accordingly.
        let proj_dir = std::env::current_dir().unwrap().join("data").join("models");
        let model_type = DefaultModelType::MediumEn;
        let model = model_type.to_model_with_path_prefix(proj_dir.as_path());

        let configs = WhisperConfigsV2::default()
            .with_model(model)
            .with_n_threads(8)
            .set_flash_attention(true);

        // NOTE: this could throw off the test (lol); remove if silero is too aggressive.
        let vad = Silero::try_new_whisper_offline_default()
            .expect("Silero expected to build with whisper-defaults.");

        let mut offline_transcriber = OfflineTranscriberBuilder::<Silero>::new()
            .with_configs(configs)
            .with_audio(audio)
            .with_channel_configurations(AudioChannelConfiguration::Mono)
            .with_voice_activity_detector(vad)
            .build()
            .expect("Offline transcriber expected to build without issue.");

        let run_transcription = Arc::new(AtomicBool::new(true));

        // Transcribe the audio
        let transcription = offline_transcriber
            .process_audio(run_transcription)
            .expect("Transcription expected to run without issue.")
            .replace(",", "")
            .replace(".", "");

        assert_eq!(transcription, expected_transcription);
    }

    #[test]
    fn test_resample_whisper_from_mp3() {
        // Punctuation (commas, periods) can sometimes be detected/not-detected based on settings
        // and the output of the model is nondeterministic.
        // It does do a good enough job of getting the words, so the transcription is stripped of
        // punctuation to ensure the outputs match exactly most of the time.
        let expected_transcription =
            "Mary has many dreams but can't touch Tennessee by way of flight";

        let audio = load_normalized_audio_file("tests/audio_files/test_mp3.mp3", None::<fn(usize)>)
            .unwrap();
        // This presumes a model is already downloaded. Handle accordingly.
        let proj_dir = std::env::current_dir().unwrap().join("data").join("models");
        let model_type = DefaultModelType::MediumEn;
        let model = model_type.to_model_with_path_prefix(proj_dir.as_path());

        let configs = WhisperConfigsV2::default()
            .with_model(model)
            .with_n_threads(8)
            .set_flash_attention(true);

        // NOTE: this could throw off the test (lol); remove if silero is too aggressive.
        let vad = Silero::try_new_whisper_offline_default()
            .expect("Silero expected to build with whisper-defaults");

        let mut offline_transcriber = OfflineTranscriberBuilder::<Silero>::new()
            .with_configs(configs)
            .with_audio(audio)
            .with_channel_configurations(AudioChannelConfiguration::Mono)
            .with_voice_activity_detector(vad)
            .build()
            .expect("Offline transcriber expected to build without issue.");

        let run_transcription = Arc::new(AtomicBool::new(true));

        // Transcribe the audio
        let transcription = offline_transcriber
            .process_audio(run_transcription)
            .expect("Transcription expected to complete without issue.")
            .replace(",", "")
            .replace(".", "");

        assert_eq!(transcription, expected_transcription);
    }
}
