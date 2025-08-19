#[cfg(test)]
mod vad_tests {
    use std::sync::LazyLock;

    use hound::SampleFormat;

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

    // Build a 10-second silent audio clip at 16kHz to tease out false positives.
    static SILENCE: LazyLock<Vec<i16>> = LazyLock::new(|| {
        let secs = 10.;
        vec![0; (secs * WHISPER_SAMPLE_RATE) as usize]
    });

    #[test]
    fn test_silero_detection() {
        let mut vad = SileroBuilder::new()
            .with_sample_rate(SileroSampleRate::R8kHz)
            .with_detection_probability_threshold(OFFLINE_VOICE_PROBABILITY_THRESHOLD)
            .with_voiced_proportion_threshold(DEFAULT_VOICE_PROPORTION_THRESHOLD)
            .build()
            .expect("Silero VAD expected to build without issues.");

        let voice_detected = vad.voice_detected(&AUDIO_SAMPLE);
        assert!(
            voice_detected,
            "Silero failed to detect voice in audio samples @ 65% threshold"
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

        // Run the voice-detection over the extracted frames with a much stricter threshold to
        // confirm that most frames are speech
        vad = vad.with_detection_probability_threshold(0.8);
        let voice_detected = vad.voice_detected(&voiced_frames);
        assert!(voice_detected, "Too many non-voiced samples.")
    }

    #[test]
    fn webrtc_vad_extraction_loose() {
        let builder = WebRtcBuilder::new()
            .with_sample_rate(WebRtcSampleRate::R8kHz)
            .with_filter_aggressiveness(WebRtcFilterAggressiveness::LowBitrate)
            .with_voiced_proportion_threshold(REAL_TIME_VOICE_PROBABILITY_THRESHOLD);

        let mut vad = builder
            .clone()
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
            .clone()
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
            .clone()
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
            .clone()
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
            .clone()
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
            .clone()
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
            .clone()
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
            .clone()
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
