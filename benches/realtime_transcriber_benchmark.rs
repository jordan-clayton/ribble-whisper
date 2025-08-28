use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::scope;

use criterion::{criterion_group, criterion_main, Criterion};

use ribble_whisper::audio::audio_ring_buffer::AudioRingBuffer;
use ribble_whisper::audio::loading::load_normalized_audio_file;
use ribble_whisper::audio::{audio_backend, WhisperAudioSample};
use ribble_whisper::transcriber::realtime_transcriber::{
    RealtimeTranscriber, RealtimeTranscriberBuilder, RealtimeTranscriberHandle,
};
use ribble_whisper::transcriber::vad::{
    Silero, SileroBuilder, SileroSampleRate,
    WebRtcBuilder, DEFAULT_VOICE_PROPORTION_THRESHOLD, REAL_TIME_VOICE_PROBABILITY_THRESHOLD, VAD,
};
use ribble_whisper::transcriber::WhisperOutput;
use ribble_whisper::utils;
use ribble_whisper::utils::errors::RibbleWhisperError;
use ribble_whisper::utils::Receiver;
use ribble_whisper::whisper::configs;
use ribble_whisper::whisper::configs::WhisperRealtimeConfigs;
use ribble_whisper::whisper::model::{DefaultModelBank, DefaultModelType, ModelBank};

// Bear in mind, this benchmark is a little fragile given the test structure and the difficulty of
// simulating the realtime loop. It is highly unlikely to fail at a point where the bench will
// get stuck in the loop, but not impossible due to the nondeterminism involved.

// Early benching suggests that the choice of VAD, from what is currently implemented, is irrelevant for realtime.
// The bottleneck will always be Whisper.

pub fn realtime_vad_benchmark(c: &mut Criterion) {
    // To prevent excess memory allocations from clouding the benchmark, pre-allocate as many
    // resources as feasible. Pass and share where appropriate.
    let (configs, model_bank) = prep_configs();
    let model_bank = Arc::new(model_bank);
    let audio_sample = prep_audio();
    let audio_ring_buffer = AudioRingBuffer::default();
    // Pre-fill the audio buffer to warm up the VAD - it can sometimes fail on first read.
    // In practice, this is fine because it will pick up next read and the audio will not get lost.
    // In these test conditions, there is a risk the audio buffer might get cleared prematurely

    let chunks = audio_sample.chunks(audio_backend::AUDIO_BUFFER_SIZE);
    for chunk in chunks {
        audio_ring_buffer.push_audio(chunk);
    }

    let silero_build_method = || {
        SileroBuilder::new()
            .with_sample_rate(SileroSampleRate::R16kHz)
            .with_detection_probability_threshold(REAL_TIME_VOICE_PROBABILITY_THRESHOLD)
            .with_voiced_proportion_threshold(DEFAULT_VOICE_PROPORTION_THRESHOLD)
            .build()
    };

    let webrtc_build_method = || {
        WebRtcBuilder::new()
            .with_sample_rate(ribble_whisper::transcriber::vad::WebRtcSampleRate::R16kHz)
            .with_frame_length_millis(
                ribble_whisper::transcriber::vad::WebRtcFrameLengthMillis::MS10,
            )
            .with_filter_aggressiveness(
                ribble_whisper::transcriber::vad::WebRtcFilterAggressiveness::LowBitrate,
            )
            .with_voiced_proportion_threshold(0.5)
            .build_webrtc()
    };

    let earshot_build_method = || {
        WebRtcBuilder::new()
            .with_sample_rate(ribble_whisper::transcriber::vad::WebRtcSampleRate::R16kHz)
            .with_frame_length_millis(
                ribble_whisper::transcriber::vad::WebRtcFrameLengthMillis::MS10,
            )
            .with_filter_aggressiveness(
                ribble_whisper::transcriber::vad::WebRtcFilterAggressiveness::LowBitrate,
            )
            .with_voiced_proportion_threshold(0.5)
            .build_earshot()
    };

    // Prep each transcriber. Each has its own sender/receiver channel to avoid the need to drain in-between tests.
    // Silero
    let (mut s_transcriber, s_handle, s_channel) = build_transcriber(
        &configs,
        &audio_ring_buffer,
        Arc::clone(&model_bank),
        silero_build_method,
    );
    eprintln!("SILERO BUILT");
    let (mut w_transcriber, w_handle, w_channel) = build_transcriber(
        &configs,
        &audio_ring_buffer,
        Arc::clone(&model_bank),
        webrtc_build_method,
    );

    eprintln!("WEBRTC BUILT");
    let (mut e_transcriber, e_handle, e_channel) = build_transcriber(
        &configs,
        &audio_ring_buffer,
        Arc::clone(&model_bank),
        earshot_build_method,
    );
    eprintln!("EARSHOT BUILT");

    let s_channel = Arc::new(s_channel);
    let w_channel = Arc::new(w_channel);
    let e_channel = Arc::new(e_channel);

    c.bench_function("Silero realtime", |b| {
        b.iter(|| {
            realtime_bencher(
                &mut s_transcriber,
                s_handle.clone(),
                Arc::clone(&s_channel),
                &audio_ring_buffer,
                &audio_sample,
            )
        });
    });

    c.bench_function("Webrtc_realtime", |b| {
        b.iter(|| {
            realtime_bencher(
                &mut w_transcriber,
                w_handle.clone(),
                Arc::clone(&w_channel),
                &audio_ring_buffer,
                &audio_sample,
            )
        });
    });

    c.bench_function("Earshot", |b| {
        b.iter(|| {
            realtime_bencher(
                &mut e_transcriber,
                e_handle.clone(),
                Arc::clone(&e_channel),
                &audio_ring_buffer,
                &audio_sample,
            )
        });
    });
}

pub fn realtime_bencher<V: VAD<f32> + Send + Sync>(
    transcriber: &mut RealtimeTranscriber<V, DefaultModelBank>,
    handle: RealtimeTranscriberHandle,
    receiver: Arc<Receiver<WhisperOutput>>,
    audio_ring_buffer: &AudioRingBuffer<f32>,
    audio_sample: &[f32],
) {
    // Prevent logging whisper to stderr
    whisper_rs::install_logging_hooks();
    // Break the audio sample into chunks of size constants::AUDIO_CHUNK_SIZE to simulate default
    // audio input
    let chunks = audio_sample.chunks(audio_backend::AUDIO_BUFFER_SIZE);
    let run_transcription = Arc::new(AtomicBool::new(true));
    scope(|s| {
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
        let _t_thread = s.spawn(move || {
            transcriber
                .run_stream(t_thread_run_transcription, Default::default())
                .expect("Transcription should not not fail under these conditions.");
        });
        // Simple thread to just drain the audio - this will sleep until it receives output from
        // the transcriber
        let _d_thread = s.spawn(move || {
            let offline_transcription =
                "Mary has many dreams but can't touch Tennessee by way of flight";
            let offline_output_length = offline_transcription.len();
            let epsilon = 5usize;

            // repeatedly drain the audio buffer to prevent a memory panic, and also set the
            // exit condition
            while d_thread_run_transcription.load(Ordering::Acquire) {
                match receiver.recv() {
                    Ok(out) => {
                        let message = match out {
                            WhisperOutput::TranscriptionSnapshot(snapshot) => snapshot.to_string(),
                            WhisperOutput::ControlPhrase(_) => "".to_string(),
                        };
                        let current_len = message.len();
                        if current_len > offline_output_length - epsilon {
                            d_thread_run_transcription.store(false, Ordering::Release);
                        }
                    }
                    Err(_) => d_thread_run_transcription.store(false, Ordering::Release),
                }
            }
            // Drain any excess messages.
            while receiver.try_recv().is_ok() {}
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = realtime_vad_benchmark
}

criterion_main!(benches);

// Some quick-n-dirty functions to automate building the transcriber objects to avoid reallocations
// during the testing.
fn prep_audio() -> Box<[f32]> {
    let audio = load_normalized_audio_file(
        "tests/audio_files/128896__joshenanigans__sentence-recitation.wav",
        None::<fn(usize)>,
    )
    .expect("Resampling should not cause issues.");
    let audio = match audio {
        WhisperAudioSample::I16(_) => unreachable!(),
        WhisperAudioSample::F32(audio) => audio,
    };

    // Extract only the voiced frames to try and prevent accidental early silence.
    // Use Silero for accuracy
    let mut vad =
        Silero::try_new_whisper_offline_default().expect("Silero should build without issues");
    let out = vad.extract_voiced_frames(&audio);
    assert!(!out.is_empty());
    out
}

fn prep_configs() -> (WhisperRealtimeConfigs, DefaultModelBank) {
    let model_bank = DefaultModelBank::new();
    let model_type = DefaultModelType::Medium;
    let model_id = model_bank.get_model_id(model_type);
    let exists = model_bank.model_exists_in_storage(model_id);
    assert!(exists.is_ok(), "Error checking model file.");
    assert!(exists.unwrap(), "Whisper medium has not been downloaded.");

    (
        WhisperRealtimeConfigs::default()
            .with_n_threads(8)
            .with_model_id(Some(model_id))
            // Also, optionally set flash attention.
            // Generally keep this on for a performance gain.
            .with_use_flash_attention(true),
        model_bank,
    )
}

fn build_transcriber<V: VAD<f32> + Send + Sync>(
    configs: &WhisperRealtimeConfigs,
    audio_buffer: &AudioRingBuffer<f32>,
    model_bank: Arc<DefaultModelBank>,
    build_method: fn() -> Result<V, RibbleWhisperError>,
) -> (
    RealtimeTranscriber<V, DefaultModelBank>,
    RealtimeTranscriberHandle,
    Receiver<WhisperOutput>,
) {
    let (text_sender, text_receiver) = utils::get_channel(30);

    let mut vad = build_method().expect("realtime VAD expected to build without issue");

    // Prime the VAD to prevent false negatives.
    let sample = audio_buffer.read(configs::VAD_SAMPLE_MS);
    let detected_audio = vad.voice_detected(&sample);
    if !detected_audio {
        eprintln!("FAILED TO DETECT FIRST RUN");
        // Try again
        let try_detect_again = vad.voice_detected(&sample);
        assert!(
            try_detect_again,
            "Failed to detect audio again, might be audio and not vad."
        );
    }

    // Transcriber
    let (transcriber, transcriber_handle) =
        RealtimeTranscriberBuilder::<V, DefaultModelBank>::new()
            .with_configs(*configs)
            .with_audio_buffer(audio_buffer)
            .with_output_sender(text_sender)
            .with_shared_model_retriever(model_bank)
            .with_voice_activity_detector(vad)
            .build()
            .expect("RealtimeTranscriber expected to build without issues.");
    (transcriber, transcriber_handle, text_receiver)
}
