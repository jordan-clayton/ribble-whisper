# Ribble-Whisper

A high-level adapter for real-time (streaming) and offline transcription using
OpenAI's Whisper. This crate wraps core [whisper-rs](https://github.com/tazz4843/whisper-rs) functionality,
voice activity detection (VAD), file loading, and optional features like resampling, making Whisper easier to integrate
into your own projects.

Real-time transcription is fairly difficult to run well without GPU acceleration.
It is recommended to build this crate using one of the exposed GPU backends (CUDA, Vulkan, etc.) to achieve
better performance (see: [Features](#features)). Depending on your hardware, it might be feasible to run real-time on
CPU using a smaller quantized model (e.g. base.en-q5, tiny.en-q5). These are likely to be less accurate, but
your results may vary.

***
NOTE: This library is primarily intended for personal and research use, but is actively used
in [Ribble](https://github.com/jordan-clayton/ribble) and has proven stable in that context.
While it has not been tested in large-scale deployments, it is likely suitable for general use--though this is not
guaranteed.
***

## External Dependencies

- If you are building with an optional GPU backend (e.g. CUDA), you will require the appropriate SDK
- CMake for building whisper-rs

## Audio Backends

By default this crate uses SDL2 to handle audio, but other backends can be used by implementing `AudioBackend`. Since
SDL2 and related subsystems are not thread-safe (but devices are), it is up to the user to handle ownership and uphold
SDL2 invariants in the application. For an example of how to do this,
see: [Ribble](https://github.com/jordan-clayton/ribble).

## Building

```bash
git clone --recursive https://github.com/jordan-clayton/ribble-whisper.git
cd ribble_whisper
cargo build --release 
# To build with a GPU backend or one of the optional features, run:
# cargo build --release --features <Desired Features>
```

Building is expected to work out-of-the-box for Windows/macOS/Linux.
If you encounter issues with building due to whisper-rs,
check [here](https://github.com/tazz4843/whisper-rs/blob/master/BUILDING.md).

***NOTE: This library currently targets Windows, macOS, and Linux. Other platforms are not officially supported or
tested.***

## Quickstart: Real-time Transcription

```bash
git clone --recursive https://github.com/jordan-clayton/ribble-whisper.git
cd ribble_whisper
# Hardware acceleration is more-or-less required for real-time transcription.
# Swap out CUDA with your preferred backend (see: Features).
cargo run --example realtime_stream --features "cuda downloader"
```

## Example Usage (Real-time)

Here is a minimal but complete example. For a more detailed version,
see [examples/realtime_stream](https://github.com/jordan-clayton/whisper-realtime/blob/main/examples/realtime_stream.rs).

```rust
// Imports are omitted here for brevity; refer to examples/realtime_stream.
// ... 
// Handle this how you see fit and pass a model to the configs builder.
// See: realtime_stream::prepare_model_bank() for an example of how to use the downloading API to retrieve a model from huggingface.
let (model_bank, model_id) = prepare_model_bank();
let model_bank = Arc::new(model_bank);
// Set the number of threads according to your hardware.
// If you can allocate around 7-8, do so as this tends to be more performant.
let configs = WhisperRealtimeConfigs::default ()
.with_n_threads(8)
.with_model_id(Some(model_id))
// Also, optionally set flash attention.
// (Generally keep this on for a performance gain with gpu processing).
.set_flash_attention(true);

// Prepare a ring buffer for writing into; the RealtimeTranscriber reads from this buffer as part of
// its transcription loop.
let audio_ring_buffer = AudioRingBuffer::<f32>::default ();

// Message channels for grabbing audio from SDL, and sending transcribed segments to a UI.
// Since this crate offers optional crossbeam support, utils::get_channel<T> will return the appropriate channel
// based on both configurations and type.
let (audio_sender, audio_receiver) =
utils::get_channel::<Arc<[f32] > > (constants::INPUT_BUFFER_CAPACITY);
let (text_sender, text_receiver) = utils::get_channel(constants::INPUT_BUFFER_CAPACITY);

// Note: Any VAD<T> + Send can be used in a RealtimeTranscriber for voice detection.
let vad = Silero::try_new_whisper_realtime_default()
.expect("Silero realtime VAD expected to build without issue when configured properly.");

// Set up the RealtimeTranscriber + Ready handle.
let (transcriber, transcriber_handle) =
RealtimeTranscriberBuilder::<Silero, DefaultModelBank>::new()
.with_configs(configs.clone())
.with_audio_buffer( & audio_ring_buffer)
.with_output_sender(text_sender)
.with_voice_activity_detector(vad)
.with_shared_model_retriever(Arc::clone( & model_bank))
.build()
.expect(
"RealtimeTranscriber expected to build without issues when configured properly.",
);

// Atomic flag for starting/stopping the transcription loop.
// This is a "UI control" to allow the user to stop the transcription without an explicit timeout.
let run_transcription = Arc::new(AtomicBool::new(true));

// Set up the Audio Backend.
let spec = CaptureSpec::default ();
let sink = ArcChannelSink::new(audio_sender);
let (_ctx, backend) =
default_backend().expect("Audio backend expected to build without issue.");

// For all intents and purposes, the backend should be able to handle most if not all devices,
// Expect this to always work until it doesn't

let mic = backend
.open_capture(spec, sink)
.expect("Audio capture expected to open without issue");

// Ribble-Whisper is designed to be as flexible as possible but real-time transcriptions need to be run using 
// asynchronous/concurrent programming.
// It is recommended to do this using threads.
let transcriber_thread = scope( | s| {
let a_thread_run_transcription = Arc::clone( & run_transcription);
let t_thread_run_transcription = Arc::clone( & run_transcription);
let p_thread_run_transcription = Arc::clone( &run_transcription);

// Block Whisper.cpp from logging to stdout/stderr and instead redirect to an optional logging hook.
redirect_whisper_logging_to_hooks();
// Start audio capture.
mic.play();
// Read data from the AudioBackend and write to the ringbuffer + (optional) static audio
let _audio_thread = s.spawn(move | | {
while a_thread_run_transcription.load(Ordering::Acquire) {
match audio_receiver.recv() {
Ok(audio_data) => {
// If the transcriber is not yet loaded, just consume the audio
if ! transcriber_handle.ready() {
continue;
}
// Write the sample into the ringbuffer
audio_ring_buffer.push_audio( & audio_data);

// ... Fan out data for other audio processing.
}
Err(_) => {
a_thread_run_transcription.store(false, Ordering::Release);
}
}
}
});

// Move the transcriber off to a thread to handle processing audio
let transcription_thread =
s.spawn( move | | transcriber.process_audio(t_thread_run_transcription));

// Update the UI with the newly transcribed data
let print_thread = s.spawn( move | | {
let mut latest_control_message = WhisperControlPhrase::GettingReady;
let mut latest_snapshot = Arc::new(TranscriptionSnapshot::default());
while p_thread_run_transcription.load(Ordering::Acquire) {
match text_receiver.recv() {
Ok(output) => match output {
// This is the most up-to-date full string transcription
WhisperOutput::TranscriptionSnapshot(snapshot) => {
latest_snapshot = Arc::clone( & snapshot);
}

WhisperOutput::ControlPhrase(message) => {
latest_control_message = message;
}
},
Err(_) => p_thread_run_transcription.store(false, Ordering::Release),
}

clear_stdout();
println ! ("Latest Control Message: {}\n", latest_control_message);
println ! ("Transcription:\n");
// Print the latest confirmed transcription.
print !("{}", latest_snapshot.confirmed());

// Print the remaining current working set of segments.
for segment in latest_segments.iter() {
print ! ("{}", segment);
}

stdout().flush().expect("Stdout should clear normally.");
}

// Take the last received snapshot and join it into a string.
// This is done in examples/realtime_stream to demonstrate that the sending mechanism terminates to
// the same state as the final transcription string.
latest_snapshot.to_string()
});

// Return the full transcription once the loop has either timed out or been stopped by the user.
transcription_thread.join()
});

// Stop audio capture.
mic.pause();

// Obtain the final transcription string:
// The outer result is for thread panics, the inner result is for transcriber logic.
let transcription = transcriber_thread
.expect("Transcription thread should not panic.")
.expect("Transcription should return without error.");

println!("Final Transcription: {}", &transcription);

//...
```

## Features

### Logging

Most errors in Ribble-Whisper are fatal and will return a result where appropriate. To get information about non fatal
errors that occur during real-time transcription, the following flag can be enabled:

- ribble-logging: Enable Ribble-Whisper logging. This is pulled in implicitly when enabling the whisper-rs tracing and
  logging backends

### Audio Backends

The default audio backend uses SDL2. If you want to implement AudioBackend yourself, disable default features:

```toml
ribble-whisper = { version = "...", default features = false }
```

- sdl2-static: Statically link with sdl2 when building Ribble-Whisper (recommended). Implicitly enables the `bundled`
  flag to build SDL2 from source for linking.

### Whisper Hardware Acceleration

This crate follows
whisper-rs [conventions](https://github.com/tazz4843/whisper-rs/tree/master?tab=readme-ov-file#feature-flags).
All backends are disabled by default and considered opt-in features.
It is recommended to enable at least one of these if supported by your system to ensure
that real-time performance is acceptable.

- cuda: enable CUDA support (Windows and Linux)
- hipblas: enable ROCm/hipBLAS support (Linux only)
- openblas: enable OpenBLAS support (Windows only)
- metal: enable Metal support (Apple only)
- coreml: enable CoreML support. Implicitly enables metal support (Apple only)
- vulkan: enable Vulkan support (Windows and Linux)

### Additional Whisper-rs Flags

- log\_backend: enable whisper-rs log\_backend for hooking into whisper.cpp's log output
- tracing\_backend: enable whisper-rs tracing\_backend for hooking into whisper.cpp's log output

### Symphonia Codecs

See: [here](https://github.com/pdeljanov/Symphonia?tab=readme-ov-file#codecs-decoders) for more information.
These flags implicitly enable the required format container flags required to support the codec.

- symphonia-all: enable support for all symphonia supported codecs
- symphonia-mpeg: enable support for all MPEG audio codecs
- symphonia-aac: enable support for AAC audio
- symphonia-alac: enable support for ALAC audio

### Additional Symphonia Flags

- symphonia-simd: enables all Symphonia SIMD optimizations

### Additional Features

- all: enable all optional additional features
- resampler: enable support for resampling audio and normalizing audio for transcribing with Whisper (highly
  recommended)
- crossbeam: enable Crossbeam support for message channels
- serde: enable Serde support for Configs serialization
- downloader: enable the synchronous (blocking) download API
- downloader-async: enables both the asynchronous and synchronous downloading APIs
- integrity: enable utilities for verifying ggml model integrity

## License

This project is licensed under MIT. See [LICENSE](./LICENSE).

## Third-Party Licenses

This project includes third-party software components:

- [`symphonia`](https://github.com/pdeljanov/symphonia) (Mozilla Public License 2.0)
    - Used for audio decoding and file loading
    - MPL-2.0 is a file-level copyleft license. This does **not** affect your usage of this library unless
      you modify and redistribute Symphonia source files
    - See [`LICENSE-MPL-2.0`](./LICENSE-MPL-2.0) for the full license
- [`earshot`](https://github.com/pykeio/earshot) (BSD-3-Clause)
    - Used for (fallback) voice activity detection
    - See [`LICENSE-BSD-3`](./LICENSE-BSD-3) for the full license

## A Note About Tests

Many of the tests and benchmarks in this project rely on audio files not included as part of the project.
You will need to provide your own audio files if you wish to run the included tests and benchmarks. I do not claim any
rights to these files
nor will they ever be distributed; they are for testing purposes only.

### Telemetry Concerns

This crate on its own does not collect telemetry. However, one of the included Voice Activity Detection implementations
may, as
noted [here](https://github.com/jordan-clayton/whisper-realtime/blob/dec51c350d80a5442f515f08737c5c14b5b07868/src/transcriber/vad.rs#L36).
This only affects Windows platforms, but if this is of concern, prefer using one of the other provided implementations
(WebRtc, etc.) or implement VAD with your preferred solution. At some point in the distant future, I will look into
self-hosting
the ONNX runtime binaries to remove any potential telemetry.
