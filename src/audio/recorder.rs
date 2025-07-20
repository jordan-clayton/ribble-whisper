use std::error::Error;
use std::sync::Arc;

use crate::audio::audio_ring_buffer::AudioRingBuffer;
use crate::utils::Sender;
use sdl2::audio::{AudioCallback, AudioFormatNum};

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
pub struct ArcChannelSink<T>(Sender<Arc<[T]>>);
impl<T: RecorderSample> ArcChannelSink<T> {
    pub fn new(sender: Sender<Arc<[T]>>) -> Self {
        Self(sender)
    }
}
/// Pushes audio out using a message queue to fan out data as `Vec<T>`
/// Use only if vectors are required in further processing, otherwise prefer the ArcChannelSink.
pub struct VecChannelSink<T>(Sender<Vec<T>>);
impl<T: RecorderSample> VecChannelSink<T> {
    pub fn new(sender: Sender<Vec<T>>) -> Self {
        Self(sender)
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

    fn push(&mut self, data: &[Self::Sample]) {
        if let Err(e) = self.0.try_send(Arc::from(data)) {
            #[cfg(feature = "ribble-logging")]
            {
                log::warn!(
                    "Failed to send audio data over recorder channel, {:#?}",
                    e.source()
                );
            }
            #[cfg(not(feature = "ribble-logging"))]
            {
                eprintln!(
                    "Failed to send audio data over recorder channel, {:#?}",
                    e.source()
                );
            }
        }
    }
}

impl<T: RecorderSample> SampleSink for VecChannelSink<T> {
    type Sample = T;
    fn push(&mut self, data: &[Self::Sample]) {
        // The only way for this to fail is if there's no receiver to receive the audio.
        // This means that transcription has stopped for whatever reason, so this device is due to
        // be dropped.
        if let Err(e) = self.0.try_send(data.to_vec()) {
            #[cfg(feature = "ribble-logging")]
            {
                log::warn!(
                    "Failed to send audio data over recorder channel, {:#?}",
                    e.source()
                );
            }
            #[cfg(not(feature = "ribble-logging"))]
            {
                eprintln!(
                    "Failed to send audio data over recorder channel, {:#?}",
                    e.source()
                );
            }
        };
    }
}
