use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use symphonia::core::formats::Track;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::probe::Hint;

use crate::audio::WhisperAudioSample;
use crate::transcriber;
use crate::utils::errors::RibbleWhisperError;

/// Encapsulates a reference to a slice of (supported-format) audio to be resampled.
pub enum ResampleableAudio<'a> {
    I16(&'a [i16]),
    F32(&'a [f32]),
    F64(&'a [f64]),
}

/// Resamples decoded audio to the desired sample rate.
/// Audio will be converted to f32 because it is the most convenient applications using this library
/// # Arguments:
/// * samples: The audio to resample,
/// * out_sample_rate: The new sample rate
/// * in_sample_rate: The original audio's sample rate
/// * num_channels: The channel configurations (number of channels)
/// # Returns:
/// * Ok(WhisperAudioSample) on success, Err(RibbleWhisperError) on failure to resample
pub fn resample(
    samples: &ResampleableAudio,
    out_sample_rate: f64,
    in_sample_rate: f64,
    num_channels: usize,
) -> Result<WhisperAudioSample, RibbleWhisperError> {
    if num_channels == 0 {
        return Err(RibbleWhisperError::ParameterError(
            "Zero channels.".to_owned(),
        ));
    }
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let samples_to_process = match samples {
        ResampleableAudio::I16(audio_in) => {
            let mut output = vec![0.0f32; audio_in.len()];
            whisper_rs::convert_integer_to_float_audio(audio_in, &mut output)?;
            output
        }
        ResampleableAudio::F32(audio_in) => audio_in.to_vec(),
        ResampleableAudio::F64(audio_in) => audio_in.iter().map(|s| *s as f32).collect(),
    };

    let resampler = SincFixedIn::new(
        out_sample_rate / in_sample_rate,
        2.0,
        params,
        samples_to_process.len() / num_channels,
        num_channels,
    )?;

    if num_channels == 2 {
        resample_stereo(samples_to_process, resampler)
    } else {
        resample_mono(samples_to_process, resampler)
    }
}

// These two functions have to take ownership of the resources for resampling
// The original slice provided is untouched; these work on a copy of the samples.
#[inline]
fn resample_mono(
    samples: Vec<f32>,
    mut resampler: SincFixedIn<f32>,
) -> Result<WhisperAudioSample, RibbleWhisperError> {
    let waves_in = vec![samples];
    let waves_out = resampler.process(&waves_in, None)?;
    Ok(WhisperAudioSample::F32(Arc::from(waves_out[0].clone())))
}

#[inline]
fn resample_stereo(
    samples: Vec<f32>,
    mut resampler: SincFixedIn<f32>,
) -> Result<WhisperAudioSample, RibbleWhisperError> {
    let waves_in: Vec<Vec<f32>> = vec![
        samples.iter().step_by(2).copied().collect(),
        samples[1..].iter().step_by(2).copied().collect(),
    ];

    let waves_out = resampler.process(&waves_in, None)?;

    let left = &waves_out[0];
    let right = &waves_out[1];
    let interleaved: Vec<f32> = left.iter().zip(right).flat_map(|(x, y)| [*x, *y]).collect();

    Ok(WhisperAudioSample::F32(Arc::from(interleaved)))
}

/// Normalizes audio to sample at 16kHz. For use with whisper.
/// # Arguments:
/// * samples: the audio to resample
/// * in_sample_rate: the original sampling rate
/// * num_channels: The channel configurations (number of channels)
/// # Returns:
/// * Ok(WhisperAudioSample) on success, Err(RibbleWhisperError) on failure to resample
#[inline]
pub fn normalize_audio(
    samples: &ResampleableAudio,
    in_sample_rate: f64,
    num_channels: usize,
) -> Result<WhisperAudioSample, RibbleWhisperError> {
    let resampled = resample(samples, 16000., in_sample_rate, num_channels)?;
    if let WhisperAudioSample::F32(stereo) = resampled {
        let mono = whisper_rs::convert_stereo_to_mono_audio(&stereo)?;
        Ok(WhisperAudioSample::F32(Arc::from(mono)))
    } else {
        // This should never, ever happen
        Err(RibbleWhisperError::ParameterError(
            "Resampling returned invalid audio format".to_owned(),
        ))
    }
}

/// Opens an audio file to check whether it needs to be resampled to 16kHz for use with whisper.
pub fn file_needs_normalizing<P: AsRef<Path>>(path: P) -> Result<bool, RibbleWhisperError> {
    let file = Box::new(File::open(path)?);
    let mss = MediaSourceStream::new(file, Default::default());
    let hint = Hint::new();
    let format_opts = Default::default();
    let metadata_opts = Default::default();
    let probed =
        symphonia::default::get_probe().format(&hint, mss, &format_opts, &metadata_opts)?;

    let format = probed.format;
    let track = format
        .default_track()
        .ok_or(RibbleWhisperError::ParameterError(
            "Failed to get default track".to_owned(),
        ))?;
    needs_normalizing(track)
}

/// Uses a [Track] to determine if audio needs to be resampled to 16kHz
/// for use with whisper.
#[inline]
pub fn needs_normalizing(track: &Track) -> Result<bool, RibbleWhisperError> {
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or(RibbleWhisperError::ParameterError(
            "Failed to get sample rate".to_owned(),
        ))? as f64;
    Ok(sample_rate != transcriber::WHISPER_SAMPLE_RATE)
}
