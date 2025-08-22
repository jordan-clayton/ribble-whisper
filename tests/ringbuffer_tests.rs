// Basic tests to ensure the ring_buffer runs properly and wraparound logic is correct
#[cfg(test)]
mod ringbuffer_tests {
    use ribble_whisper::audio::audio_ring_buffer::AudioRingBuffer;
    use ribble_whisper::transcriber;
    #[test]
    fn test_get_audio_length_ms() {
        let expected_ms = 4000;
        let ring_buffer = AudioRingBuffer::default();
        let sample_len = (4f64 * transcriber::WHISPER_SAMPLE_RATE) as usize;
        let samples = vec![0.5f32; sample_len];
        ring_buffer.push_audio(&samples);
        // Ensure the push was okay
        assert_eq!(ring_buffer.get_audio_length(), sample_len);
        // Now, get the ms
        assert_eq!(ring_buffer.get_audio_length_ms(), expected_ms);
    }

    #[test]
    fn test_copy_buffer_lengths() {
        // Full length
        let ring_buffer: AudioRingBuffer<f32> = AudioRingBuffer::default();
        let sample_len = (crate::SAMPLE_DURATION / 1000) as f64 * transcriber::WHISPER_SAMPLE_RATE;
        let sample_len = sample_len as usize;

        let samples = vec![0.5f32; sample_len];
        // This is expected to not panic.
        ring_buffer.push_audio(&samples);
        assert_eq!(ring_buffer.get_audio_length(), sample_len);

        // Half-length
        let ring_buffer: AudioRingBuffer<f32> = AudioRingBuffer::default();
        let sample_len =
            (crate::SAMPLE_DURATION / 1000) as f64 * transcriber::WHISPER_SAMPLE_RATE / 2f64;
        let sample_len = sample_len as usize;

        let mut samples = vec![0.5f32; sample_len];
        // This is expected to not panic.
        ring_buffer.push_audio(&mut samples);
        assert_eq!(ring_buffer.get_audio_length(), sample_len);
        assert_eq!(ring_buffer.get_capacity(), sample_len * 2);
    }

    #[test]
    fn test_insert_and_get() {
        let ring_buffer: AudioRingBuffer<f32> = AudioRingBuffer::default();

        // Insert four seconds of audio data.
        let sample_len = 4f64 * transcriber::WHISPER_SAMPLE_RATE;
        let sample_len = sample_len as usize;

        let samples = vec![0.5f32; sample_len];
        // This is expected to not panic.
        ring_buffer.push_audio(&samples);

        // Ensure that 4s of audio data are in the buffer.
        assert_eq!(ring_buffer.get_audio_length(), sample_len);

        // Get two seconds of audio data.
        let mut result: Vec<f32> = vec![];
        // This is expected to not panic
        ring_buffer.read_into(2000, &mut result);
        // Test length
        assert_eq!(result.len(), sample_len / 2);

        // Ensure all 0.5
        let right_data = &result.iter().all(|x| *x >= 0.49f32 && *x <= 0.51f32);
        assert!(right_data);
    }

    #[test]
    fn test_insert_and_overflow() {
        // Half-length
        let ring_buffer: AudioRingBuffer<f32> = AudioRingBuffer::default();
        let sample_len =
            (crate::SAMPLE_DURATION / 1000) as f64 * transcriber::WHISPER_SAMPLE_RATE / 2f64;
        let sample_len = sample_len as usize;

        let samples = vec![0.5f32; sample_len];

        // This is expected to not panic.
        for _i in 0..3 {
            ring_buffer.push_audio(&samples);
        }

        // Expect a full buffer
        assert_eq!(ring_buffer.get_audio_length(), ring_buffer.get_capacity());
        // Expect head to be at the midpoint.
        assert_eq!(
            ring_buffer.get_head_position(),
            ring_buffer.get_audio_length() / 2
        );
    }
    #[test]
    fn test_wraparound_audio() {
        // Half-length
        let ring_buffer: AudioRingBuffer<f32> = AudioRingBuffer::default();
        // These are 5-second samples
        let sample_len =
            ((crate::SAMPLE_DURATION / 1000) as f64 * transcriber::WHISPER_SAMPLE_RATE) / 2f64;
        let sample_len = sample_len as usize;

        // Fill the first half with 0.1
        let first_half_samples = vec![0.1f32; sample_len];
        // Fill the second half with 0.2
        let second_half_samples = vec![0.2f32; sample_len];

        // This is expected to not panic.
        ring_buffer.push_audio(&first_half_samples);
        ring_buffer.push_audio(&second_half_samples);
        // Move the head to position 1/4
        ring_buffer.push_audio(&first_half_samples[0..sample_len / 2]);

        // Expect a full buffer
        assert_eq!(ring_buffer.get_audio_length(), ring_buffer.get_capacity());
        // Expect head to be a quarter way in.
        assert_eq!(
            ring_buffer.get_head_position(),
            ring_buffer.get_audio_length() / 4
        );

        // Read 5 seconds of audio
        let mut read_buffer = vec![];
        ring_buffer.read_into(crate::SAMPLE_DURATION / 2, &mut read_buffer);
        // Bisect at the midpoint and compare the audio output
        let (first_half, second_half) = read_buffer.split_at(sample_len / 2);
        assert_eq!(first_half.len(), second_half.len());

        let first_wrong = first_half.iter().position(|s| *s != 0.2);
        let second_wrong = second_half.iter().position(|s| *s != 0.1);

        assert!(
            first_wrong.is_none(),
            "Found 0.1 in first half at index: {}",
            first_wrong.unwrap()
        );
        assert!(
            second_wrong.is_none(),
            "Found 0.2 in second half at index: {}",
            second_wrong.unwrap()
        );
    }

    #[test]
    fn test_get_with_overflow() {
        // Half-length
        let ring_buffer: AudioRingBuffer<f32> = AudioRingBuffer::default();
        let sample_len =
            ((crate::SAMPLE_DURATION / 1000) as f64 * transcriber::WHISPER_SAMPLE_RATE) / 2f64;
        let sample_len = sample_len as usize;

        let expect_length = (2f64 * transcriber::WHISPER_SAMPLE_RATE) as usize;

        let mut results_vec: [Vec<f32>; 3] = [vec![], vec![], vec![]];
        // This is expected to not panic.
        for i in 0..3 {
            let mut samples = vec![i as f32; sample_len];
            // Insert 5s of samples.
            ring_buffer.push_audio(&mut samples);
            // Get 2s of samples.
            ring_buffer.read_into(2000, &mut results_vec[i]);
        }

        // Length check.
        for vec in results_vec {
            assert_eq!(vec.len(), expect_length);
            let nd = non_decreasing(&vec);
            assert!(nd);
        }
    }
    #[test]
    fn test_clear_n_samples() {
        let ring_buffer: AudioRingBuffer<f32> = AudioRingBuffer::default();

        // Insert four seconds of audio data.
        let sample_len = (4f64 * transcriber::WHISPER_SAMPLE_RATE) as usize;

        let samples = vec![0.5f32; sample_len];
        // This is expected to not panic.
        ring_buffer.push_audio(&samples);
        // Get the current head length; this should be unchanged.
        let head_len = ring_buffer.get_head_position();

        // Now, clear 4 seconds of audio; expect the buffer to be empty.
        // The old implementation kept roughly N_SAMPLES, but this is not necessary with the new implementation.
        ring_buffer.clear_n_samples(4000);
        assert_eq!(ring_buffer.get_audio_length(), 0);
        // Ensure the head is unchanged (only the audio length should be reduced to nearly 0).
        assert_eq!(ring_buffer.get_head_position(), head_len);
    }
    #[test]
    fn test_clear_n_samples_with_wraparound() {
        let ring_buffer: AudioRingBuffer<f32> = AudioRingBuffer::default();

        // Insert 15 seconds of audio data.
        let sample_len = (15f64 * transcriber::WHISPER_SAMPLE_RATE) as usize;

        let samples = vec![0.5f32; sample_len];
        let sample_chunks =
            samples.chunks_exact((5f64 * transcriber::WHISPER_SAMPLE_RATE) as usize);
        // This is expected to not panic, the head should be halfway through the buffer.
        // This has to be added in chunks to get the full sample, otherwise only the last buffer_len
        // samples get pushed to the ringbuffer.
        for sample in sample_chunks {
            ring_buffer.push_audio(&sample);
        }

        // Ensure that the head is at the middle of the audio buffer
        let head_pos = ring_buffer.get_head_position();
        assert_eq!(head_pos, ring_buffer.get_audio_length() / 2);
        let expected_audio_len = (10f64 * transcriber::WHISPER_SAMPLE_RATE) as usize;

        assert_eq!(ring_buffer.get_audio_length(), expected_audio_len);

        // Now, clear 5 seconds of audio; expect the buffer to be of length 5s + constants::N_SAMPLES_KEEP
        ring_buffer.clear_n_samples(5000);
        let anticipated_len = (5f64 * transcriber::WHISPER_SAMPLE_RATE) as usize;

        assert_eq!(anticipated_len, ring_buffer.get_audio_length());
        // Expect the head to be at position constants::N_SAMPLES_KEEP
        assert_eq!(ring_buffer.get_head_position(), head_pos);
    }

    fn non_decreasing(v: &Vec<f32>) -> bool {
        for i in 0..v.len() - 1 {
            let j = i + 1;
            if v[j] < v[i] {
                return false;
            }
        }
        true
    }
}

pub const SAMPLE_DURATION: usize = 10000;
