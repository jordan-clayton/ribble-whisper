use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::transcriber::WHISPER_SAMPLE_RATE;
use crate::utils::errors::RibbleWhisperError;
use parking_lot::Mutex;

struct InnerAudioRingBuffer<T: Copy + Clone + Default> {
    // Insertion pointer
    head: AtomicUsize,
    // The amount of audio within the buffer, in units of sizeof(T)
    audio_len: AtomicUsize,
    capacity_ms: AtomicUsize,
    buffer_capacity: AtomicUsize,
    sample_rate: AtomicUsize,
    // If at some point in the future it becomes imperative to support a reader/writer paradigm
    // this will change to an RW lock.
    buffer: Mutex<Vec<T>>,
}

/// A thread-safe mpmc ring-buffer designed for use with a [transcriber::realtime_transcriber::RealtimeTranscriber].
/// Due to thread-safety requirements it cannot be lock-free, but it should have minimal overhead
/// in most use-cases.
#[derive(Clone)]
pub struct AudioRingBuffer<T: Copy + Clone + Default> {
    inner: Arc<InnerAudioRingBuffer<T>>,
}

/// Builder to set the parameters of an AudioRingBuffer
#[derive(Clone)]
pub struct AudioRingBufferBuilder {
    /// Size of the buffer in milliseconds
    capacity_ms: Option<usize>,
    /// Sample rate of the audio in the buffer
    sample_rate: Option<usize>,
}

impl AudioRingBufferBuilder {
    pub fn new() -> Self {
        Self {
            capacity_ms: None,
            sample_rate: None,
        }
    }

    /// Sets the requested capacity measured in milliseconds
    pub fn with_capacity_ms(mut self, capacity_ms: usize) -> Self {
        self.capacity_ms = Some(capacity_ms);
        self
    }
    /// Sets the requested sample rate measured in Hz
    pub fn with_sample_rate(mut self, sample_rate: usize) -> Self {
        self.sample_rate = Some(sample_rate);
        self
    }

    /// Build an AudioRingBuffer with the desired parameters.
    /// This will return Err if the length/sample rate are missing or zero.
    pub fn build<T: Copy + Clone + Default>(
        self,
    ) -> Result<AudioRingBuffer<T>, RibbleWhisperError> {
        let c_ms =
            self.capacity_ms
                .filter(|&ms| ms > 0)
                .ok_or(RibbleWhisperError::ParameterError(
                    "AudioRingBufferBuilder has zero-length buffer.".to_string(),
                ))?;
        let s_rate =
            self.sample_rate
                .filter(|&sr| sr > 0)
                .ok_or(RibbleWhisperError::ParameterError(
                    " AudioRingBufferBuilder has zero-size sample rate.".to_string(),
                ))?;

        let capacity_ms = AtomicUsize::new(c_ms);
        let buffer_size = ((c_ms as f64 / 1000.) * (s_rate as f64)) as usize;
        let buffer_len = AtomicUsize::new(buffer_size);
        let audio_len = AtomicUsize::new(0);
        let head = AtomicUsize::new(0);
        let sample_rate = AtomicUsize::new(s_rate);
        let buffer = Mutex::new(vec![T::default(); buffer_size]);
        let inner = Arc::new(InnerAudioRingBuffer {
            head,
            audio_len,
            capacity_ms,
            buffer_capacity: buffer_len,
            sample_rate,
            buffer,
        });

        Ok(AudioRingBuffer { inner })
    }
}

impl Default for AudioRingBufferBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy + Clone + Default> AudioRingBuffer<T> {
    /// Returns the currently stored audio length measured in units of size_of(T)
    pub fn get_audio_length(&self) -> usize {
        self.inner.audio_len.load(Ordering::Acquire)
    }

    /// Returns the currently stored audio length measured in ms
    pub fn get_audio_length_ms(&self) -> usize {
        // UH, this is totally incorrect.
        let audio_len = self.inner.audio_len.load(Ordering::Acquire) as f64;
        let sample_rate = self.inner.sample_rate.load(Ordering::Acquire) as f64;
        ((audio_len * 1000f64) / sample_rate) as usize
    }
    /// Returns the ringbuffer capacity measured in milliseconds
    pub fn get_capacity_in_ms(&self) -> usize {
        self.inner.capacity_ms.load(Ordering::Acquire)
    }
    /// Returns the ringbuffer capacity measured in size_of(T)
    pub fn get_capacity(&self) -> usize {
        self.inner.buffer_capacity.load(Ordering::Acquire)
    }
    /// returns the current position of the write head
    pub fn get_head_position(&self) -> usize {
        self.inner.head.load(Ordering::Acquire)
    }

    /// Writes the input samples to the buffer.
    /// NOTE: if the input length exceeds the buffer capacity, only the last n samples are written
    /// to the buffer, where n = buffer capacity
    pub fn push_audio(&self, input: &[T]) {
        let mut n_samples = input.len();
        let mut stream = input.to_vec();

        let buffer_len = self.inner.buffer_capacity.load(Ordering::Acquire);
        if n_samples > buffer_len {
            let len = n_samples;
            n_samples = buffer_len;
            let new_start = len - n_samples;
            stream = stream[new_start..].to_vec();
        }

        // Grab the buffer to hold the state before grabbing the head position
        let mut buffer = self.inner.buffer.lock();
        let head_pos = self.inner.head.load(Ordering::Acquire);
        if head_pos + n_samples > buffer_len {
            let offset = buffer_len - head_pos;
            // memcpy stuff
            // First copy.
            let copy_buffer = &mut buffer[head_pos..head_pos + offset];
            let stream_buffer = &stream[0..offset];

            copy_buffer.copy_from_slice(stream_buffer);

            // Second copy
            let diff_offset = n_samples - offset;
            let copy_buffer = &mut buffer[0..diff_offset];
            let stream_buffer = &stream[offset..offset + diff_offset];

            copy_buffer.copy_from_slice(stream_buffer);

            let new_head_pos = (head_pos + n_samples) % buffer_len;
            self.inner.head.store(new_head_pos, Ordering::Release);

            let old_audio_len = self.inner.audio_len.load(Ordering::Acquire);
            let new_audio_len = (old_audio_len + n_samples).min(buffer_len);
            self.inner.audio_len.store(new_audio_len, Ordering::Release);
        } else {
            let copy_buffer = &mut buffer[head_pos..head_pos + n_samples];
            let stream_buffer = &stream[0..n_samples];

            copy_buffer.copy_from_slice(stream_buffer);

            let new_head_pos = (head_pos + n_samples) % buffer_len;
            self.inner.head.store(new_head_pos, Ordering::Release);

            let old_audio_len = self.inner.audio_len.load(Ordering::Acquire);
            let new_audio_len = (old_audio_len + n_samples).min(buffer_len);

            self.inner.audio_len.store(new_audio_len, Ordering::Release);
        }
    }

    /// Reads min(len_ms, audio length) ms from the buffer and returns the output as `Vec<T>`
    /// NOTE: set len_ms to 0 to read the full buffer.
    pub fn read(&self, len_ms: usize) -> Vec<T> {
        let mut buf = vec![];
        self.read_into(len_ms, &mut buf);
        buf
    }

    /// Reads min(len_ms, audio length) ms from the buffer and writes to the provided result vector.
    /// NOTE: set len_ms to 0 to read the full buffer.
    pub fn read_into(&self, len_ms: usize, result: &mut Vec<T>) {
        let mut ms = len_ms;

        if ms == 0 {
            ms = self.inner.capacity_ms.load(Ordering::Acquire);
        }

        result.clear();
        let sample_rate = self.inner.sample_rate.load(Ordering::Acquire);
        let mut n_samples = (ms as f64 * sample_rate as f64 / 1000f64) as usize;

        // Grab the buffer to hold the state before checking the audio length.
        let buffer = self.inner.buffer.lock();

        let audio_len = self.inner.audio_len.load(Ordering::Acquire);
        if n_samples > audio_len {
            n_samples = audio_len;
        }
        result.resize(n_samples, T::default());
        // If n_samples == 0 (ie. the audio buffer has just been cleared).
        if result.is_empty() {
            return;
        }

        let head_pos = self.inner.head.load(Ordering::Acquire);
        let buffer_len = self.inner.buffer_capacity.load(Ordering::Acquire);

        let mut start_pos: i64 = head_pos as i64 - n_samples as i64;

        if start_pos < 0 {
            start_pos += buffer_len as i64;
        }

        let start_pos = start_pos as usize;

        if start_pos + n_samples > buffer_len {
            let to_endpoint = buffer_len - start_pos;

            // First copy: starting pos (head - n_samples) up to the end of the buffer
            let copy_buffer = &mut result[0..to_endpoint];
            let stream = &buffer[start_pos..start_pos + to_endpoint];

            copy_buffer.copy_from_slice(stream);

            // second copy: start of the buffer up to the end of the remaining samples.
            let remaining_samples = n_samples - to_endpoint;
            let copy_buffer = &mut result[to_endpoint..to_endpoint + remaining_samples];
            let stream = &buffer[0..remaining_samples];

            copy_buffer.copy_from_slice(stream);
        } else {
            let copy_buffer = &mut result[0..n_samples];
            let stream = &buffer[start_pos..start_pos + n_samples];

            copy_buffer.copy_from_slice(stream);
        }
    }

    /// Clears the AudioRingBuffer completely
    pub fn clear(&self) {
        // Guard state by hogging the mutex to prevent data inconsistencies
        let _buffer = self.inner.buffer.lock();
        self.inner.head.store(0, Ordering::SeqCst);
        self.inner.audio_len.store(0, Ordering::SeqCst);
    }

    /// Clears the AudioRingBuffer retains at most len_ms amount of data.
    pub fn clear_from_back_retain_ms(&self, len_ms: usize) {
        if len_ms == 0 {
            self.clear();
            return;
        }
        // Guard state by hogging the mutex to prevent data inconsistencies
        let _buffer = self.inner.buffer.lock();
        let sample_rate = self.inner.sample_rate.load(Ordering::Acquire);
        let audio_len = self.inner.audio_len.load(Ordering::Acquire);
        let n_samples = ((len_ms as f64 * sample_rate as f64 / 1000f64) as usize).min(audio_len);
        self.inner.audio_len.store(n_samples, Ordering::Release);
    }

    /// Clears the requested amount of audio data from the **back** of the buffer.
    /// If you want to retain a small amount of data, you will need to manually calculate the
    /// correct length in milliseconds to clear.
    pub fn clear_n_ms_from_back(&self, len_ms: usize) {
        if len_ms == 0 {
            return;
        }
        // Guard state by hogging the mutex to prevent data inconsistencies
        let _buffer = self.inner.buffer.lock();

        let sample_rate = self.inner.sample_rate.load(Ordering::Acquire);
        let mut n_samples = (len_ms as f64 * sample_rate as f64 / 1000f64) as usize;

        let audio_len = self.inner.audio_len.load(Ordering::Acquire);
        if n_samples > audio_len {
            n_samples = audio_len;
        }

        let new_len = audio_len - n_samples;
        self.inner.audio_len.store(new_len, Ordering::Release);
    }
}

impl<T: Copy + Clone + Default> Default for AudioRingBuffer<T> {
    /// Returns a Whisper-ready AudioRingBuffer, ready for use in a [transcriber::realtime_transcriber::RealtimeTranscriber].
    fn default() -> Self {
        AudioRingBufferBuilder::new()
            .with_capacity_ms(AUDIO_BUFFER_CAPACITY)
            .with_sample_rate(WHISPER_SAMPLE_RATE as usize)
            .build()
            .expect("Default AudioRingbuffer should build without problems.")
    }
}

// Total length is 10s
pub const AUDIO_BUFFER_CAPACITY: usize = 10000;
const KEEP_MS: f64 = 300f64;
// To help with resolving word boundaries.
pub const N_SAMPLES_KEEP: usize = ((1e-3 * KEEP_MS) * WHISPER_SAMPLE_RATE) as usize;
