use criterion::{criterion_group, criterion_main, Criterion};
use hound::SampleFormat;

use ribble_whisper::audio::pcm::IntoPcmS16;
use ribble_whisper::transcriber::vad::WebRtcSampleRate::R8kHz;
use ribble_whisper::transcriber::vad::{
    Earshot, Silero, SileroBuilder, SileroSampleRate,
    WebRtc, WebRtcBuilder, WebRtcFilterAggressiveness, DEFAULT_VOICE_PROPORTION_THRESHOLD, REAL_TIME_VOICE_PROBABILITY_THRESHOLD, VAD,
};
pub fn vad_benchmark(c: &mut Criterion) {
    // Load the audio sample to pass to each benchmark function
    let audio: Vec<i16> = {
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
    };

    let mut silero = SileroBuilder::new()
        .with_sample_rate(SileroSampleRate::R8kHz)
        .with_detection_probability_threshold(REAL_TIME_VOICE_PROBABILITY_THRESHOLD)
        .with_voiced_proportion_threshold(DEFAULT_VOICE_PROPORTION_THRESHOLD)
        .build()
        .expect("Silero VAD should build without problems");

    let mut webrtc = WebRtcBuilder::new()
        .with_sample_rate(R8kHz)
        .with_filter_aggressiveness(WebRtcFilterAggressiveness::LowBitrate)
        .with_voiced_proportion_threshold(DEFAULT_VOICE_PROPORTION_THRESHOLD)
        .build_webrtc()
        .expect("Silero VAD should build without problems");
    let mut earshot = WebRtcBuilder::new()
        .with_sample_rate(R8kHz)
        .with_filter_aggressiveness(WebRtcFilterAggressiveness::LowBitrate)
        .with_voiced_proportion_threshold(DEFAULT_VOICE_PROPORTION_THRESHOLD)
        .build_earshot()
        .expect("Silero VAD should build without problems");

    c.bench_function("Earshot:", |b| {
        b.iter(|| bench_vad::<i16, Earshot>(&mut earshot, &audio))
    });
    c.bench_function("WebRtc:", |b| {
        b.iter(|| bench_vad::<i16, WebRtc>(&mut webrtc, &audio))
    });
    c.bench_function("Silero:", |b| {
        b.iter(|| bench_vad::<i16, Silero>(&mut silero, &audio))
    });
}

fn bench_vad<S: IntoPcmS16, T: VAD<S>>(vad: &mut impl VAD<S>, samples: &[S]) {
    let _ = vad.voice_detected(samples);
}

criterion_group!(benches, vad_benchmark);
criterion_main!(benches);
