use crate::audio::audio_ring_buffer::AudioRingBuffer;
use crate::utils::Sender;
use sdl2::audio::{AudioCallback, AudioFormatNum};
use std::error::Error;
use std::sync::Arc;
use std::thread::sleep;

const SLEEP_MILLIS: u64 = 100;

/// Trait alias to unify Audio formats to meet the bounds of Audio Backends.
pub trait RecorderSample:
    Default + Copy + AudioFormatNum + voice_activity_detector::Sample + Send + Sync + 'static
{
}
impl<T: Default + Copy + AudioFormatNum + voice_activity_detector::Sample + Send + Sync + 'static>
    RecorderSample for T
{
}

/// A trait responsible for pushing audio out from the backend capture.
pub trait SampleSink: Send + 'static {
    type Sample: RecorderSample;
    fn push(&mut self, data: &[Self::Sample]);
}

/// A backend-agnostic recorder struct used in audio callbacks to push audio out for consumption.
pub struct Recorder<S: SampleSink> {
    sink: S,
}

impl<S: SampleSink> Recorder<S> {
    pub fn new(sink: S) -> Self {
        Self { sink }
    }
}

impl<S: SampleSink> AudioCallback for Recorder<S> {
    type Channel = S::Sample;

    fn callback(&mut self, input: &mut [Self::Channel]) {
        self.sink.push(input)
    }
}

/// Pushes audio out by writing directly into a ring-buffer that can be used by
/// [crate::transcriber::realtime_transcriber::RealtimeTranscriber]
pub struct RingBufSink<T: RecorderSample>(AudioRingBuffer<T>);
impl<T: RecorderSample> RingBufSink<T> {
    pub fn new(buffer: AudioRingBuffer<T>) -> Self {
        Self(buffer)
    }
}
/// Pushes audio out using a message queue to fan out data as `Arc<[T]>`
/// Significantly faster for audio fanout than `Vec<T>`; prefer when possible.
/// NOTE: Due to synchronization difficulties, pushing can result in logging false positives if the sink is still
/// in scope and has not yet been paused. This is most likely to occur when transcription finishes.
pub struct ArcChannelSink<T> {
    channel: Sender<Arc<[T]>>,
    logged_disconnect: bool,
}
impl<T: RecorderSample> ArcChannelSink<T> {
    pub fn new(sender: Sender<Arc<[T]>>) -> Self {
        Self {
            channel: sender,
            logged_disconnect: false,
        }
    }

    pub fn is_disconnected(&self) -> bool {
        self.logged_disconnect
    }
}
/// Pushes audio out using a message queue to fan out data as `Vec<T>`
/// Use only if vectors are required in further processing, otherwise prefer the ArcChannelSink.
/// NOTE: Due to synchronization difficulties, pushing can result in logging false positives if the sink is still
/// in scope and has not yet been paused. This is most likely to occur when transcription finishes.
pub struct VecChannelSink<T> {
    channel: Sender<Vec<T>>,
    logged_disconnect: bool,
}
impl<T: RecorderSample> VecChannelSink<T> {
    pub fn new(sender: Sender<Vec<T>>) -> Self {
        Self {
            channel: sender,
            logged_disconnect: false,
        }
    }

    pub fn is_disconnected(&self) -> bool {
        self.logged_disconnect
    }
}

impl<T: RecorderSample> SampleSink for RingBufSink<T> {
    type Sample = T;
    fn push(&mut self, data: &[Self::Sample]) {
        self.0.push_audio(data);
    }
}

impl<T: RecorderSample> SampleSink for ArcChannelSink<T> {
    type Sample = T;
    /// NOTE: Due to synchronization difficulties, this can log false positives if the sink is still
    /// in scope and has not yet been paused. This is most likely to occur when transcription finishes.
    fn push(&mut self, data: &[Self::Sample]) {
        if let Err(e) = self.channel.try_send(Arc::from(data)) {
            #[cfg(feature = "crossbeam")]
            let disconnected = e.is_disconnected();
            #[cfg(not(feature = "crossbeam"))]
            let disconnected = matches!(e, std::sync::mpsc::TrySendError::Disconnected(_));

            if disconnected {
                if !self.logged_disconnect {
                    self.logged_disconnect = true;
                    #[cfg(feature = "ribble-logging")]
                    {
                        log::warn!("Arc Recorder channel disconnected!");
                    }
                    #[cfg(not(feature = "ribble-logging"))]
                    {
                        eprintln!("Arc Recorder channel disconnected!");
                    }
                }
                sleep(std::time::Duration::from_millis(SLEEP_MILLIS));
                return;
            }
            #[cfg(feature = "ribble-logging")]
            {
                log::warn!(
                    "Failed to send audio data over recorder channel.\n\
                    Error: {}\n\
                    Error source:{:#?}",
                    &e,
                    e.source()
                );
            }
            #[cfg(not(feature = "ribble-logging"))]
            {
                eprintln!(
                    "Failed to send audio data over recorder channel.\n\
                    Error: {}\n\
                    Error source:{:#?}",
                    &e,
                    e.source()
                );
            }
        }
    }
}

impl<T: RecorderSample> SampleSink for VecChannelSink<T> {
    type Sample = T;
    fn push(&mut self, data: &[Self::Sample]) {
        if let Err(e) = self.channel.try_send(data.to_vec()) {
            #[cfg(feature = "crossbeam")]
            let disconnected = e.is_disconnected();
            #[cfg(not(feature = "crossbeam"))]
            let disconnected = matches!(e, std::sync::mpsc::TrySendError::Disconnected(_));

            if disconnected {
                if !self.logged_disconnect {
                    self.logged_disconnect = true;
                    #[cfg(feature = "ribble-logging")]
                    {
                        log::warn!("Vec Recorder channel disconnected!");
                    }
                    #[cfg(not(feature = "ribble-logging"))]
                    {
                        eprintln!("Vec Recorder channel disconnected!");
                    }
                }
                sleep(std::time::Duration::from_millis(SLEEP_MILLIS));
                return;
            }
            #[cfg(feature = "ribble-logging")]
            {
                log::warn!(
                    "Failed to send audio data over recorder channel.\n\
                    Error: {}\n\
                    Error source:{:#?}",
                    &e,
                    e.source()
                );
            }
            #[cfg(not(feature = "ribble-logging"))]
            {
                eprintln!(
                    "Failed to send audio data over recorder channel.\n\
                    Error: {}\n\
                    Error source:{:#?}",
                    &e,
                    e.source()
                );
            }
        };
    }
}
