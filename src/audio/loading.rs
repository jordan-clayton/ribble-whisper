use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::Decoder;
use symphonia::core::errors::Error;
use symphonia::core::formats::FormatReader;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::probe::{Hint, ProbeResult};

#[cfg(feature = "resampler")]
use crate::audio::resampler::{needs_normalizing, normalize_audio, ResampleableAudio};
use crate::audio::WhisperAudioSample;
use crate::utils::callback::{Callback, Nop, RibbleWhisperCallback};
use crate::utils::errors::RibbleWhisperError;

fn get_audio_probe<P: AsRef<Path> + Sized>(path: P) -> Result<ProbeResult, RibbleWhisperError> {
    let file = Box::new(File::open(path)?);
    let mss = MediaSourceStream::new(file, Default::default());
    let hint = Hint::new();
    let format_opts = Default::default();
    let metadata_opts = Default::default();
    let probe = symphonia::default::get_probe().format(&hint, mss, &format_opts, &metadata_opts)?;
    Ok(probe)
}

/// Probes an audio file to try and get the total number of audio frames.
/// NOTE: When using this to get a total size for measuring progress in percent, do not try to
/// manually perform the channel arithmetic in the progress_callback.
/// Both [load_audio_file] and [load_normalized_audio_file]
/// already handle Stereo/Mono and will return the correct number of frames decoded **measured in frames.**
pub fn audio_file_num_frames<P: AsRef<Path> + Sized>(path: P) -> Result<u64, RibbleWhisperError> {
    let probe = get_audio_probe(path)?;
    let format = probe.format;
    let track = format
        .default_track()
        .ok_or(RibbleWhisperError::ParameterError(
            "Failed to get default audio track".to_string(),
        ))?;
    let codec_params = &track.codec_params;
    codec_params
        .n_frames
        .ok_or(RibbleWhisperError::ParameterError(
            "Failed to get the number of frames".to_string(),
        ))
}

/// Loads a RibbleWhisper-compatible (i.e. Stereo/mono, can be converted into whisper-compatible) audio file
/// for transcription.
/// To receive the number of frames copied per each decode iteration, use the optional progress_callback.
/// NOTE: this expects the audio to be sampled at 16kHz. Either resample the audio beforehand, or use: [load_normalized_audio_file]
pub fn load_audio_file<P: AsRef<Path>>(
    path: P,
    progress_callback: Option<impl FnMut(usize)>,
) -> Result<WhisperAudioSample, RibbleWhisperError> {
    let decoder_opts = Default::default();
    let probed = get_audio_probe(path)?;

    let format = probed.format;
    let track = format
        .default_track()
        .ok_or(RibbleWhisperError::ParameterError(
            "Failed to get default audio track".to_string(),
        ))?;

    let decoder = symphonia::default::get_codecs().make(&track.codec_params, &decoder_opts)?;
    // Decode loop
    let samples = match progress_callback {
        Some(p) => decode_loop(track.id, decoder, format, RibbleWhisperCallback::new(p)),
        None => decode_loop(track.id, decoder, format, Nop::new()),
    };
    Ok(WhisperAudioSample::F32(Arc::from(samples?)))
}

/// Loads a WhisperRealtime-compatible (i.e. Can be converted into whisper-compatible) audio file,
/// and resamples to 16 kHz as necessary.
/// To receive the number of frames copied per each decode iteration, use the optional progress_callback.
/// NOTE: requires the resampler feature flag to be set
#[cfg(feature = "resampler")]
pub fn load_normalized_audio_file<P: AsRef<Path> + Sized>(
    path: P,
    progress_callback: Option<impl FnMut(usize)>,
) -> Result<WhisperAudioSample, RibbleWhisperError> {
    let decoder_opts = Default::default();
    let probed = get_audio_probe(path)?;
    let format = probed.format;
    let track = format
        .default_track()
        .ok_or(RibbleWhisperError::ParameterError(
            "Failed to get default track".to_string(),
        ))?;

    // Get the codec parameters before passing ownership to the decode loop.
    let codec_params = &track.codec_params;
    let sample_rate = codec_params
        .sample_rate
        .ok_or(RibbleWhisperError::ParameterError(
            "Failed to grab sample rate".to_string(),
        ))? as f64;

    let num_channels = codec_params
        .channels
        .ok_or(RibbleWhisperError::ParameterError(
            "Failed to grab number of channels".to_string(),
        ))?
        .count();

    let needs_normalizing = needs_normalizing(track);
    let decoder = symphonia::default::get_codecs().make(&track.codec_params, &decoder_opts)?;
    // Decode loop

    let samples = match progress_callback {
        Some(p) => decode_loop(track.id, decoder, format, RibbleWhisperCallback::new(p)),
        None => decode_loop(track.id, decoder, format, Nop::new()),
    }?;

    // Normalize
    if needs_normalizing? {
        let audio = ResampleableAudio::F32(&samples);
        normalize_audio(&audio, sample_rate, num_channels)
    } else {
        Ok(WhisperAudioSample::F32(Arc::from(samples)))
    }
}

// Note: the progress_callback returns the total number of frames decoded per iteration in the
// decode loop.
fn decode_loop(
    track_id: u32,
    mut decoder: Box<dyn Decoder>,
    mut reader: Box<dyn FormatReader>,
    mut progress_callback: impl Callback<Argument = usize>,
) -> Result<Vec<f32>, RibbleWhisperError> {
    let mut samples = vec![];
    let mut sample_buf = None;

    loop {
        let next_packet = reader.next_packet();
        // This is the only recoverable error - and only applies to chained OGG
        // For all other containers, afaik, this can be "The end" of the stream
        // if let Err(Error::ResetRequired) = next_packet.as_ref() {
        //     break;
        // }
        // Otherwise, an unrecoverable error has occured; break the decoding loop
        if next_packet.as_ref().is_err() {
            break;
        }
        // Otherwise, unpack the packet and let the error bubble up
        let packet = next_packet?;

        // Consume metadata; not really sure what to do with this.
        while !reader.metadata().is_latest() {
            reader.metadata().pop();
        }

        // Skip over irrelevant tracks
        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(audio_buffer) => {
                if sample_buf.is_none() {
                    let spec = *(audio_buffer.spec());
                    let duration = audio_buffer.capacity() as u64;
                    sample_buf = Some(SampleBuffer::<f32>::new(duration, spec));
                }
                let channels = audio_buffer.spec().channels.iter().count();
                if channels > 2 {
                    return Err(RibbleWhisperError::ParameterError(format!(
                        "Only Stereo/Mono audio supported. Number of channels: {channels}"
                    )));
                }

                let in_mono = channels == 1;

                if let Some(buf) = sample_buf.as_mut() {
                    if in_mono {
                        buf.copy_planar_ref(audio_buffer);
                    } else {
                        buf.copy_interleaved_ref(audio_buffer);
                    }

                    samples.extend_from_slice(buf.samples());
                    progress_callback.call(buf.samples().len() / channels)
                }
            }
            // Skip malformed data
            Err(Error::DecodeError(_)) => (),
            // On an IO/Seek error, just break the loop.
            Err(_) => break,
        };
    }
    // Return all decoded samples
    Ok(samples)
}
