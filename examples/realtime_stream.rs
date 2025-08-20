use std::io::{stdout, Write};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::scope;

use indicatif::{ProgressBar, ProgressStyle};
use parking_lot::Mutex;

#[cfg(feature = "sdl2")]
use ribble_whisper::audio::audio_backend::default_backend;
use ribble_whisper::audio::audio_backend::AudioBackend;
use ribble_whisper::audio::audio_backend::CaptureSpec;
use ribble_whisper::audio::audio_ring_buffer::AudioRingBuffer;
use ribble_whisper::audio::microphone::MicCapture;
use ribble_whisper::audio::recorder::ArcChannelSink;
use ribble_whisper::audio::{AudioChannelConfiguration, WhisperAudioSample};
#[cfg(feature = "downloader")]
use ribble_whisper::downloader::downloaders::sync_download_request;
#[cfg(feature = "downloader")]
use ribble_whisper::downloader::SyncDownload;
use ribble_whisper::transcriber::offline_transcriber::OfflineTranscriberBuilder;
use ribble_whisper::transcriber::realtime_transcriber::RealtimeTranscriberBuilder;
use ribble_whisper::transcriber::vad::Silero;
use ribble_whisper::transcriber::{
    redirect_whisper_logging_to_hooks, Transcriber, TranscriptionSnapshot,
};
use ribble_whisper::transcriber::{
    CallbackTranscriber, WhisperCallbacks, WhisperControlPhrase, WhisperOutput,
};
use ribble_whisper::utils;
use ribble_whisper::utils::callback::{Nop, RibbleWhisperCallback, StaticRibbleWhisperCallback};
use ribble_whisper::whisper::configs::WhisperRealtimeConfigs;
use ribble_whisper::whisper::model;
use ribble_whisper::whisper::model::{DefaultModelBank, ModelBank, ModelId};

fn main() {
    let (model_bank, model_id) = prepare_model_bank();
    let model_bank = Arc::new(model_bank);
    // Set the number of threads according to your hardware.
    // If you can allocate around 7-8, do so as this tends to be more performant.
    let configs = WhisperRealtimeConfigs::default()
        .with_n_threads(8)
        .with_model_id(Some(model_id))
        // Also, optionally set flash attention.
        // (Generally keep this on for a performance gain with gpu processing).
        .set_flash_attention(true);

    let audio_ring_buffer = AudioRingBuffer::<f32>::default();
    // These -might- need to update a little bit more quickly.
    let (audio_sender, audio_receiver) = utils::get_channel::<Arc<[f32]>>(64);
    let (text_sender, text_receiver) = utils::get_channel(64);

    // Note: Any VAD<T> + Send can be used.
    let vad = Silero::try_new_whisper_realtime_default()
        .expect("Silero realtime VAD expected to build without issue when configured properly.");

    // Transcriber
    let (transcriber, transcriber_handle) =
        RealtimeTranscriberBuilder::<Silero, DefaultModelBank>::new()
            .with_configs(configs.clone())
            .with_audio_buffer(&audio_ring_buffer)
            .with_output_sender(text_sender)
            .with_voice_activity_detector(vad)
            .with_shared_model_retriever(Arc::clone(&model_bank))
            .build()
            .expect(
                "RealtimeTranscriber expected to build without issues when configured properly.",
            );

    // Optional store + re-transcription.
    // Get input for stdin.
    print!("Would you like to store audio and re-transcribe once realtime has finished? y/n: ");
    stdout().flush().unwrap();

    let mut confirm_re_transcribe = String::new();

    let offline_audio_buffer: Arc<Mutex<Option<Vec<f32>>>> = Arc::new(Mutex::new(None));
    if let Ok(_) = std::io::stdin().read_line(&mut confirm_re_transcribe) {
        let confirm = confirm_re_transcribe.trim().to_lowercase();
        if "y" == confirm {
            println!("Confirmed. Recording audio.\n");
            let mut guard = offline_audio_buffer.lock();
            *guard = Some(vec![])
        }
    }

    let t_offline_audio_buffer = Arc::clone(&offline_audio_buffer);

    let run_transcription = Arc::new(AtomicBool::new(true));
    let c_handler_run_transcription = Arc::clone(&run_transcription);

    // Use CTRL-C to stop the transcription.
    ctrlc::set_handler(move || {
        println!("Interrupt received");
        c_handler_run_transcription.store(false, Ordering::SeqCst);
    })
    .expect("failed to set SIGINT handler");

    // Set up the Audio Backend.
    let spec = CaptureSpec::default();
    let sink = ArcChannelSink::new(audio_sender);
    let (_ctx, backend) =
        default_backend().expect("Audio backend expected to build without issue.");

    // For all intents and purposes, the backend should be able to handle most if not all devices,
    // Expect this to always work until it doesn't

    let mic = backend
        .open_capture(spec, sink)
        .expect("Audio capture expected to open without issue");

    let transcriber_thread = scope(|s| {
        let a_thread_run_transcription = Arc::clone(&run_transcription);
        let t_thread_run_transcription = Arc::clone(&run_transcription);
        let p_thread_run_transcription = Arc::clone(&run_transcription);

        // Block Whisper.cpp from logging to stdout/stderr and instead redirect to an optional logging hook.
        redirect_whisper_logging_to_hooks();
        mic.play();
        // Read data from the AudioBackend and write to the ringbuffer + (optional) static audio
        let _audio_thread = s.spawn(move || {
            // Just hog the mutex because there's only one writer.
            let mut offline_buffer = t_offline_audio_buffer.lock();
            while a_thread_run_transcription.load(Ordering::Acquire) {
                match audio_receiver.recv() {
                    Ok(audio_data) => {
                        // If the transcriber is not yet loaded, just consume the audio
                        if !transcriber_handle.ready() {
                            continue;
                        }

                        // Otherwise, fan out to the ring-buffer (for transcrption) and the optional
                        // offline buffer
                        audio_ring_buffer.push_audio(&audio_data);
                        if let Some(buffer) = offline_buffer.as_mut() {
                            buffer.extend_from_slice(&audio_data);
                        }
                    }
                    Err(_) => {
                        eprintln!("AUDIO CHANNEL CLOSED");
                        a_thread_run_transcription.store(false, Ordering::Release);
                    }
                }
            }

            println!("Audio fanout thread completed.");
            // Drop the guard to release the mutex.
            drop(offline_buffer);
        });

        // Move the transcriber off to a thread to handle processing audio
        let transcription_thread =
            s.spawn(move || transcriber.process_audio(t_thread_run_transcription));

        // Update the UI with the newly transcribed data
        let print_thread = s.spawn(move || {
            let mut latest_control_message = WhisperControlPhrase::GettingReady;
            let mut latest_snapshot = Arc::new(TranscriptionSnapshot::default());
            while p_thread_run_transcription.load(Ordering::Acquire) {
                match text_receiver.recv() {
                    Ok(output) => match output {
                        // This is the most up-to-date full string transcription
                        WhisperOutput::TranscriptionSnapshot(snapshot) => {
                            latest_snapshot = Arc::clone(&snapshot);
                        }

                        WhisperOutput::ControlPhrase(message) => {
                            latest_control_message = message;
                        }
                    },
                    Err(_) => {
                        eprintln!("PRINT CHANNEL CLOSED");
                        p_thread_run_transcription.store(false, Ordering::Release);
                    }
                }

                clear_stdout();
                println!("Latest Control Message: {}\n", latest_control_message);
                println!("Transcription:\n");
                // Print the latest confirmed transcription.
                print!("{}", latest_snapshot.confirmed());
                // Print the remaining current working set of segments.
                for segment in latest_snapshot.string_segments() {
                    print!("{}", segment);
                }
                stdout().flush().expect("Stdout should clear normally.");
            }

            println!("\nPrint thread completed.");
            // Take the last received snapshot and join it into a string.
            // This is just to demonstrate that the sending mechanism terminates to the same
            // state as the final transcription string.
            latest_snapshot.to_string()
        });

        // Send both outputs for comparison in stdout.
        (transcription_thread.join(), print_thread.join())
    });

    mic.pause();

    let transcription = transcriber_thread
        .0
        .expect("Transcription thread should not panic.")
        .expect("Transcription should run properly.");
    let rt_transcription = transcriber_thread
        .1
        .expect("Print thread should not panic.");

    // Comparison
    // clear_stdout();
    println!("\nFinal Transcription (print thread):");
    println!("{}", &rt_transcription);

    println!();

    println!("Final Transcription (returned):");
    println!("{}", &transcription);

    // Offline audio (re) transcription:
    if let Some(buffer) = offline_audio_buffer.lock().take() {
        // Use silero to prune out the silence
        let vad = Silero::try_new_whisper_offline_default()
            .expect("Silero expected to build with whisper defaults");
        // Consume the configs into whisper v2 (or reuse)
        let s_configs = configs.into_whisper_v2_configs();
        let offline_transcriber = OfflineTranscriberBuilder::<Silero, DefaultModelBank>::new()
            .with_configs(s_configs)
            .with_audio(WhisperAudioSample::F32(Arc::from(buffer)))
            .with_channel_configurations(AudioChannelConfiguration::Mono)
            .with_voice_activity_detector(vad)
            .with_shared_model_retriever(Arc::clone(&model_bank))
            .build()
            .expect("OfflineTranscriber expected to build with no issues.");

        // Initiate a progress bar
        let pb = ProgressBar::new(100);
        pb.set_style(ProgressStyle::default_bar()
            .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {percent} ({eta})").unwrap()
            .progress_chars("#>-")
        );

        pb.set_message("Transcription Progress: ");
        let pb_c = pb.clone();

        // Reset run_transcription
        let run_offline_transcription = Arc::clone(&run_transcription);
        run_offline_transcription.store(true, Ordering::Release);
        let progress_closure = move |p| {
            pb_c.set_position(p as u64);
        };
        let static_progress_callback = StaticRibbleWhisperCallback::new(progress_closure);

        let callbacks = WhisperCallbacks {
            progress: Some(static_progress_callback),
            // If you want to run a similar UI RPL like in the realtime example, the new segment callback
            // will let you access a snapshot to send via a message queue or similar.
            new_segment: None::<Nop<String>>,
        };

        let transcription =
            offline_transcriber.process_with_callbacks(run_offline_transcription, callbacks);
        pb.finish_and_clear();
        println!("\nOffline re-transcription:");
        if let Err(e) = &transcription {
            eprintln!("{}", e);
            return;
        }
        println!("{}", &transcription.unwrap());
    };
}

// Downloads the model if it doesn't exist within CWD/data/models.
// If the path does not exist already, this will create the full path upon downloading the model.
fn prepare_model_bank() -> (DefaultModelBank, ModelId) {
    let bank = DefaultModelBank::new();

    // GPU acceleration is currently required to run larger models in realtime.
    let model_type = if cfg!(feature = "_gpu") {
        model::DefaultModelType::Medium
    } else {
        model::DefaultModelType::Small
    };

    let model_id = bank.get_model_id(model_type);
    let exists_in_storage = bank
        .model_exists_in_storage(model_id)
        .expect("Model should be retrieved by key without issue.");

    if !exists_in_storage {
        println!("Downloading model:");
        stdout().flush().unwrap();

        // Downloading.
        let url = model_type.url();

        let sync_downloader = sync_download_request(url.as_str(), model_type.to_file_name());
        if let Err(e) = sync_downloader.as_ref() {
            eprintln!("{}", e);
        }

        let sync_downloader = sync_downloader.unwrap();

        // Initiate a progress bar.
        let pb = ProgressBar::new(sync_downloader.total_size() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})").unwrap()
            .progress_chars("#>-")
        );
        pb.set_message(format!("Downloading {}", url));

        let pb_c = pb.clone();

        // Set the progress callback
        let progress_callback_closure = move |n: usize| {
            pb_c.set_position(n as u64);
        };

        let progress_callback = RibbleWhisperCallback::new(progress_callback_closure);
        let mut sync_downloader = sync_downloader.with_progress_callback(progress_callback);

        let _model = bank
            .get_model(model_id)
            .expect("Model is expected to exist in default storage.");

        let download = sync_downloader.download(bank.model_directory());
        assert!(download.is_ok());
        let model_in_directory = bank.model_exists_in_storage(model_id);
        assert!(
            model_in_directory.is_ok(),
            "Failed to probe directory for model"
        );
        assert!(model_in_directory.unwrap(), "Model failed to download");
    }
    (bank, model_id)
}

fn clear_stdout() {
    Command::new("cls")
        .status()
        .or_else(|_| Command::new("clear").status())
        .expect("Failed to clear stdout");
}
