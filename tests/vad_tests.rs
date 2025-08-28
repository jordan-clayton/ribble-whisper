#[cfg(test)]
mod vad_tests {
    use std::sync::{Arc, LazyLock};

    use hound::SampleFormat;
    use ribble_whisper::audio::loading::load_normalized_audio_file;
    use ribble_whisper::audio::pcm::IntoPcmS16;
    use ribble_whisper::audio::resampler::{resample, ResampleableAudio};
    use ribble_whisper::audio::WhisperAudioSample;
    use ribble_whisper::transcriber::vad::{
        Earshot, Resettable, Silero,
        SileroBuilder, SileroSampleRate, WebRtc, WebRtcBuilder, WebRtcFilterAggressiveness,
        WebRtcFrameLengthMillis, WebRtcSampleRate, DEFAULT_VOICE_PROPORTION_THRESHOLD, OFFLINE_VOICE_PROBABILITY_THRESHOLD, REAL_TIME_VOICE_PROBABILITY_THRESHOLD,
        VAD,
    };
    use ribble_whisper::transcriber::WHISPER_SAMPLE_RATE;

    // This audio file contains a speaker who methodically reads out a series of random sentences.
    // The voice clip is not super clear, nor loud, and there are significant gaps between phrases,
    // making it a relatively good candidate for testing the accuracy of the voice detection.
    // Tests that probe this sample for speech are expected to determine there is, in fact, speech.

    // The sample rate for this file is 8kHz, and it should be in Mono.
    static AUDIO_SAMPLE: LazyLock<Vec<i16>> = LazyLock::new(|| {
        let mut reader =
            hound::WavReader::open("tests/audio_files/OSR_us_000_0060_8k.wav").unwrap();
        let spec = reader.spec();
        let sample_format = spec.sample_format;
        match sample_format {
            SampleFormat::Float => reader
                .samples::<f32>()
                .map(|s| s.expect("Audio expected to read properly.").into_pcm_s16())
                .collect(),
            SampleFormat::Int => reader
                .samples::<i16>()
                .map(|s| s.expect("Audio expected to read properly."))
                .collect(),
        }
    });

    static WHISPER_AUDIO_SAMPLE: LazyLock<Arc<[f32]>> = LazyLock::new(|| {
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

    // Build a 10-second silent audio clip at 16kHz to tease out false positives.
    static SILENCE: LazyLock<Vec<i16>> = LazyLock::new(|| {
        let secs = 10.;
        vec![0; (secs * WHISPER_SAMPLE_RATE) as usize]
    });

    // Silero is very good for detecting -actual- speech, but it can get tripped up
    // with audio that has a lot of pauses.
    #[test]
    fn test_silero_detection() {
        let mut vad = SileroBuilder::new()
            .with_sample_rate(SileroSampleRate::R8kHz)
            .with_detection_probability_threshold(OFFLINE_VOICE_PROBABILITY_THRESHOLD)
            .with_voiced_proportion_threshold(DEFAULT_VOICE_PROPORTION_THRESHOLD)
            .build()
            .expect("Silero VAD expected to build without issues.");

        // Prune out the detected speech frames to cut out pauses.
        let voiced_frames = vad.extract_voiced_frames(&AUDIO_SAMPLE);
        let voice_detected = vad.voice_detected(&voiced_frames);

        assert!(
            voice_detected,
            "Silero failed to detect voice in audio samples @ 60% probability, 50% threshold",
        );

        let mut whisper_vad = Silero::try_new_whisper_realtime_default()
            .expect("Whisper-ready Silero VAD expected to build without issues");
        let voice_detected = whisper_vad.voice_detected(&SILENCE);
        assert!(
            !voice_detected,
            "Silero detected voice in a silent clip with whisper parameters."
        )
    }
    #[test]
    fn test_silero_detection_audio_source_2() {
        let mut vad = Silero::try_new_whisper_realtime_default()
            .expect("Whisper-read Silero VAD expected to build without issues.");
        let voice_detected = vad.voice_detected(&WHISPER_AUDIO_SAMPLE);
        assert!(
            voice_detected,
            "Silero failed to detect voice in audio samples @ 30% probability threshold, 50% voiced proportion threshold"
        );

        // Run a test over a 300 MS chunk that's ~ 1 second into the audio
        let vad_ms_len = ((300.0 / 1000.0) * WHISPER_SAMPLE_RATE) as usize;
        let audio_ms_len = WHISPER_SAMPLE_RATE as usize;
        let start_idx = audio_ms_len - vad_ms_len;
        let audio_sample = &WHISPER_AUDIO_SAMPLE[start_idx..start_idx + vad_ms_len];

        let sample_detected_primed = vad.voice_detected(audio_sample);

        assert!(
            sample_detected_primed,
            "Primed Silero failed on small voice sample"
        );
        vad.reset_session();

        let sample_detected_fresh = vad.voice_detected(audio_sample);
        assert!(
            sample_detected_fresh,
            "Resetted Silero failed on small voice sample"
        );

        // Test a fresh-vad to rule out silero init weirdness
        let mut new_vad = Silero::try_new_whisper_realtime_default()
            .expect("Whisper-ready Silero model expected to build without issues.");
        let new_voice_detected = new_vad.voice_detected(audio_sample);

        assert!(
            new_voice_detected,
            "New VAD failed to detect small-sample voice"
        );

        // Test from the very first 4800 samples.
        let first_vad = &WHISPER_AUDIO_SAMPLE[..vad_ms_len];
        vad.reset_session();
        let first_vad_voice_detected = vad.voice_detected(first_vad);

        assert!(
            !first_vad_voice_detected,
            "Silero is going funky; the audio doesn't pick up until ~200 ms in"
        );

        vad.reset_session();

        // This is the minimum which passes
        // This more-or-less chops out the silence.
        let offset = ((206.0 / 1000.0) * WHISPER_SAMPLE_RATE) as usize;
        let offset_first_vad = &WHISPER_AUDIO_SAMPLE[offset..offset + vad_ms_len];
        let offset_vad_voice_detected = vad.voice_detected(offset_first_vad);
        assert!(
            offset_vad_voice_detected,
            "Failed to detect voice even after pruning."
        );
    }

    #[test]
    fn test_webrtc_detection() {
        let mut vad = WebRtcBuilder::new()
            .with_sample_rate(WebRtcSampleRate::R8kHz)
            .with_filter_aggressiveness(WebRtcFilterAggressiveness::LowBitrate)
            .with_frame_length_millis(WebRtcFrameLengthMillis::MS10)
            .with_voiced_proportion_threshold(0.65)
            .build_webrtc()
            .expect("WebRtc expected to build without issues.");
        let voice_detected = vad.voice_detected(&AUDIO_SAMPLE);
        assert!(
            voice_detected,
            "WebRtc failed to detect voice in audio samples @ 65% threshold with LowBitrate aggressiveness."
        );

        let mut whisper_vad = WebRtc::try_new_whisper_realtime_default()
            .expect("Whisper-ready WebRtc VAD expected to build without issues");
        let voice_detected = whisper_vad.voice_detected(&SILENCE);
        assert!(
            !voice_detected,
            "WebRtc detected voice in a silent clip with whisper parameters."
        )
    }

    #[test]
    fn test_earshot_detection() {
        let mut vad = WebRtcBuilder::new()
            .with_sample_rate(WebRtcSampleRate::R8kHz)
            .with_filter_aggressiveness(WebRtcFilterAggressiveness::LowBitrate)
            .with_frame_length_millis(WebRtcFrameLengthMillis::MS10)
            .with_voiced_proportion_threshold(0.65)
            .build_earshot()
            .expect("Earshot expected to build without issues.");
        let voice_detected = vad.voice_detected(&AUDIO_SAMPLE);
        assert!(
            voice_detected,
            "Earshot failed to detect voice in audio samples @ 65% threshold with LowBitrate aggressiveness."
        );

        let mut whisper_vad = Earshot::try_new_whisper_realtime_default()
            .expect("Whisper-ready WebRtc VAD expected to build without issues");
        let voice_detected = whisper_vad.voice_detected(&SILENCE);
        assert!(
            !voice_detected,
            "Earshot detected voice in a silent clip with whisper parameters."
        )
    }

    // Due to limitations with a dependency this test cannot control for/rule the filter_aggressiveness
    // being maintained across resets.
    // This is likely the best test that I could write given the limitations.
    #[test]
    fn test_webrtc_reset() {
        // The audio is known to contain speech, but the audio quality is poor enough that it should be possible to
        // overtune the configurations intentionally produce a false negative.
        let mut vad = WebRtcBuilder::new()
            .with_sample_rate(WebRtcSampleRate::R16kHz)
            .with_filter_aggressiveness(WebRtcFilterAggressiveness::Quality)
            .with_frame_length_millis(WebRtcFrameLengthMillis::MS10)
            // The entire audio clip is around ~50% speech with lots of pauses
            // A high-enough threshold will produce a false-negative.
            .with_voiced_proportion_threshold(0.75)
            .build_webrtc()
            .expect("WebRtc expected to build without issues.");

        // Resample the audio track to 16 kHz to match the VAD
        let upsampled_audio = resample(&ResampleableAudio::I16(&AUDIO_SAMPLE), 16000., 8000., 1)
            .expect("Resampling audio should pass");
        let upsampled_audio = match upsampled_audio {
            WhisperAudioSample::I16(_) => unreachable!(),
            WhisperAudioSample::F32(audio) => audio,
        };

        // Run on the sample to produce a false negative.
        let voice_detected = vad.voice_detected(&upsampled_audio);
        assert!(!voice_detected, "Not able to produce a false negative");

        // Reset the vad
        vad.reset_session();

        // If the settings are not properly maintained, this VAD will then have a sample rate of
        // 8kHz at Quality aggressiveness. If this is true, running the VAD on the 8kHz sample is
        // also expected to produce a false negative.

        // If the VAD's sampling rate is still at 16kHz, it should overestimate the speech, so test for a
        // false (but not really false, actually true) positive to conclude the sample rate is maintained.
        let voice_detected = vad.voice_detected(&AUDIO_SAMPLE);
        assert!(voice_detected, "Still produced a false negative.");
    }

    #[test]
    fn silero_vad_extraction_loose() {
        let mut vad = SileroBuilder::new()
            .with_sample_rate(SileroSampleRate::R8kHz)
            .with_detection_probability_threshold(REAL_TIME_VOICE_PROBABILITY_THRESHOLD)
            .build()
            .expect("Silero expected to build without issues");
        let voiced_frames = vad.extract_voiced_frames(&AUDIO_SAMPLE);

        assert!(
            !voiced_frames.is_empty(),
            "Failed to extract any voiced frames"
        );

        assert!(
            voiced_frames.len() < AUDIO_SAMPLE.len(),
            "Failed to exclude silent frames. Voiced: {}, Sample: {}",
            voiced_frames.len(),
            AUDIO_SAMPLE.len()
        );

        // Run the voice-detection over the extracted frames with the offline threshold to ensure
        // that most frames are speech
        vad = vad.with_detection_probability_threshold(OFFLINE_VOICE_PROBABILITY_THRESHOLD);

        let voice_detected = vad.voice_detected(&voiced_frames);
        assert!(voice_detected, "Too many non-voiced samples.");
    }

    // Perhaps this needs to prune harder.
    // I'm not quite sure.
    #[test]
    fn silero_vad_extraction_strict() {
        let mut vad = SileroBuilder::new()
            .with_sample_rate(SileroSampleRate::R8kHz)
            .with_detection_probability_threshold(OFFLINE_VOICE_PROBABILITY_THRESHOLD)
            .with_voiced_proportion_threshold(DEFAULT_VOICE_PROPORTION_THRESHOLD)
            .build()
            .expect("Silero expected to build without issues");
        let voiced_frames = vad.extract_voiced_frames(&AUDIO_SAMPLE);

        assert!(
            !voiced_frames.is_empty(),
            "Failed to extract any voiced frames; vad might be too strict."
        );

        assert!(
            voiced_frames.len() < AUDIO_SAMPLE.len(),
            "Failed to exclude silent frames. Voiced: {}, Sample: {}",
            voiced_frames.len(),
            AUDIO_SAMPLE.len()
        );

        let voice_detected = vad.voice_detected(&voiced_frames);
        assert!(
            voice_detected,
            "Too many or too few non-voiced samples, {}",
            voiced_frames.len()
        )
    }

    #[test]
    fn webrtc_vad_extraction_loose() {
        let builder = WebRtcBuilder::new()
            .with_sample_rate(WebRtcSampleRate::R8kHz)
            .with_filter_aggressiveness(WebRtcFilterAggressiveness::LowBitrate)
            .with_voiced_proportion_threshold(REAL_TIME_VOICE_PROBABILITY_THRESHOLD);

        let mut vad = builder
            .build_webrtc()
            .expect("WebRtc expected to build without issue.");

        let voiced_frames = vad.extract_voiced_frames(&AUDIO_SAMPLE);

        assert!(
            !voiced_frames.is_empty(),
            "Failed to extract any voiced frames"
        );

        assert!(
            voiced_frames.len() < AUDIO_SAMPLE.len(),
            "Failed to exclude silent frames. Voiced: {}, Sample: {}",
            voiced_frames.len(),
            AUDIO_SAMPLE.len()
        );

        // Run the voice-detection over the extracted frames with the offline threshold to ensure
        // that most frames are speech
        vad = builder
            .with_filter_aggressiveness(WebRtcFilterAggressiveness::Aggressive)
            .with_voiced_proportion_threshold(OFFLINE_VOICE_PROBABILITY_THRESHOLD)
            .build_webrtc()
            .expect("WebRtc expected to build without issue");

        let voice_detected = vad.voice_detected(&voiced_frames);
        assert!(voice_detected, "Too many non-voiced samples.");
    }
    #[test]
    fn webrtc_vad_extraction_strict() {
        // Start the builder with the "Offline" aggressiveness settings
        let builder = WebRtcBuilder::new()
            .with_sample_rate(WebRtcSampleRate::R8kHz)
            .with_filter_aggressiveness(WebRtcFilterAggressiveness::Aggressive)
            .with_voiced_proportion_threshold(OFFLINE_VOICE_PROBABILITY_THRESHOLD);

        let mut vad = builder
            .build_webrtc()
            .expect("WebRtc expected to build without issue.");

        let voiced_frames = vad.extract_voiced_frames(&AUDIO_SAMPLE);

        assert!(
            !voiced_frames.is_empty(),
            "Failed to extract any voiced frames"
        );

        assert!(
            voiced_frames.len() < AUDIO_SAMPLE.len(),
            "Failed to exclude silent frames. Voiced: {}, Sample: {}",
            voiced_frames.len(),
            AUDIO_SAMPLE.len()
        );

        // Run the voice-detection over the extracted frames with a much stricter threshold to ensure
        // that most frames are speech
        // VeryAggressive prunes out a significant portion of frames and might actually be missing on some overlaps
        // Aggressive detects around .9, VeryAggressive detects just over .75
        vad = builder
            .with_filter_aggressiveness(WebRtcFilterAggressiveness::VeryAggressive)
            .build_webrtc()
            .expect("WebRtc expected to build without issue");

        let voice_detected = vad.voice_detected(&voiced_frames);
        assert!(voice_detected, "Too many non-voiced samples.");
    }

    #[test]
    fn earshot_vad_extraction_loose() {
        let builder = WebRtcBuilder::new()
            .with_sample_rate(WebRtcSampleRate::R8kHz)
            .with_filter_aggressiveness(WebRtcFilterAggressiveness::LowBitrate)
            .with_voiced_proportion_threshold(REAL_TIME_VOICE_PROBABILITY_THRESHOLD);

        let mut vad = builder
            .build_earshot()
            .expect("Earshot expected to build without issue.");

        let voiced_frames = vad.extract_voiced_frames(&AUDIO_SAMPLE);

        assert!(
            !voiced_frames.is_empty(),
            "Failed to extract any voiced frames"
        );

        assert!(
            voiced_frames.len() < AUDIO_SAMPLE.len(),
            "Failed to exclude silent frames. Voiced: {}, Sample: {}",
            voiced_frames.len(),
            AUDIO_SAMPLE.len()
        );

        // Run the voice-detection over the extracted frames with the offline threshold to ensure
        // that most frames are speech
        vad = builder
            .with_filter_aggressiveness(WebRtcFilterAggressiveness::Aggressive)
            .with_voiced_proportion_threshold(OFFLINE_VOICE_PROBABILITY_THRESHOLD)
            .build_earshot()
            .expect("Earshot expected to build without issue");

        let voice_detected = vad.voice_detected(&voiced_frames);
        assert!(voice_detected, "Too many non-voiced samples.");
    }
    #[test]
    fn earshot_vad_extraction_strict() {
        // Start the builder with the "Offline" aggressiveness settings
        let builder = WebRtcBuilder::new()
            .with_sample_rate(WebRtcSampleRate::R8kHz)
            .with_filter_aggressiveness(WebRtcFilterAggressiveness::Aggressive)
            .with_voiced_proportion_threshold(OFFLINE_VOICE_PROBABILITY_THRESHOLD);

        let mut vad = builder
            .build_earshot()
            .expect("Earshot expected to build without issue.");

        let voiced_frames = vad.extract_voiced_frames(&AUDIO_SAMPLE);

        assert!(
            !voiced_frames.is_empty(),
            "Failed to extract any voiced frames"
        );

        assert!(
            voiced_frames.len() < AUDIO_SAMPLE.len(),
            "Failed to exclude silent frames. Voiced: {}, Sample: {}",
            voiced_frames.len(),
            AUDIO_SAMPLE.len()
        );

        // Run the voice-detection over the extracted frames with a much stricter threshold to ensure
        // that most frames are speech.
        // Earshot is much less accurate than WebRtc, and so even with VeryAggressive, this will
        // detect .9 of frames containing speech
        vad = builder
            .with_filter_aggressiveness(WebRtcFilterAggressiveness::VeryAggressive)
            .with_voiced_proportion_threshold(0.8)
            .build_earshot()
            .expect("Earshot expected to build without issue");

        let voice_detected = vad.voice_detected(&voiced_frames);
        assert!(voice_detected, "Too many non-voiced samples.");
    }

    #[test]
    fn silero_vad_extraction_silent() {
        let mut vad = SileroBuilder::new()
            .with_sample_rate(SileroSampleRate::R8kHz)
            .with_detection_probability_threshold(REAL_TIME_VOICE_PROBABILITY_THRESHOLD)
            .build()
            .expect("Silero expected to build without issues");
        let voiced_frames = vad.extract_voiced_frames(&SILENCE);

        assert!(
            voiced_frames.is_empty(),
            "Erroneously extracted voice frames from silence."
        );
    }

    #[test]
    fn webrtc_vad_extraction_silent() {
        let mut vad = WebRtcBuilder::new()
            .with_sample_rate(WebRtcSampleRate::R8kHz)
            .with_filter_aggressiveness(WebRtcFilterAggressiveness::LowBitrate)
            .with_voiced_proportion_threshold(DEFAULT_VOICE_PROPORTION_THRESHOLD)
            .build_webrtc()
            .expect("Webrtc expected to build without issues");
        let voiced_frames = vad.extract_voiced_frames(&SILENCE);

        assert!(
            voiced_frames.is_empty(),
            "Erroneously extracted voice frames from silence."
        );
    }

    #[test]
    fn earshot_vad_extraction_silent() {
        let mut vad = WebRtcBuilder::new()
            .with_sample_rate(WebRtcSampleRate::R8kHz)
            .with_filter_aggressiveness(WebRtcFilterAggressiveness::LowBitrate)
            .with_voiced_proportion_threshold(DEFAULT_VOICE_PROPORTION_THRESHOLD)
            .build_earshot()
            .expect("Earshot expected to build without issues");
        let voiced_frames = vad.extract_voiced_frames(&SILENCE);

        assert!(
            voiced_frames.is_empty(),
            "Erroneously extracted voice frames from silence."
        );
    }
}
