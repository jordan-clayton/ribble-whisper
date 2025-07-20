use thiserror::Error;

#[derive(Debug, Error)]
pub enum RibbleWhisperError {
    /// Called on a data-channel failure
    /// Contains the output message, so it can be recovered.
    #[error("{0}")]
    TranscriptionSenderError(String),
    /// User either supplied an incorrect parameter, or failed to supply a parameter at all
    #[error("Parameter Error {0}")]
    ParameterError(String),
    /// [std::io::Error]
    #[error("I/O error: {0}")]
    IOError(#[from] std::io::Error),
    /// [symphonia::core::errors::Error]
    #[error("Decode error: {0}")]
    DecodeError(#[from] symphonia::core::errors::Error),
    /// Otherwise unknown error
    #[error("Unknown Error {0}")]
    Unknown(String),
    /// [std::string::ParseError]
    #[error("Parse Error {0}")]
    ParseError(#[from] std::string::ParseError),
    /// [url::ParseError]
    #[error("UrlParse Error {0}")]
    UrlParseError(#[from] url::ParseError),
    /// [whisper_rs::WhisperError]
    #[error("Whisper Error {0}")]
    WhisperError(#[from] whisper_rs::WhisperError),
    #[error("Device Error {0}")]
    DeviceError(String),
    /// [rubato::ResampleError]
    #[cfg(feature = "resampler")]
    #[error("ResampleError: {0}")]
    ResampleError(#[from] rubato::ResampleError),
    /// [rubato::ResamplerConstructionError]
    #[cfg(feature = "resampler")]
    #[error("ResamplerConstructionError: {0}")]
    ResamplerConstructionError(#[from] rubato::ResamplerConstructionError),
    /// [reqwest::Error]
    #[cfg(feature = "downloader")]
    #[error("Reqwest Error {0}")]
    ReqwestError(#[from] reqwest::Error),
    /// Failure to download a model
    #[error("Download Error {0}")]
    #[cfg(feature = "downloader")]
    DownloadError(String),
    /// Indicates a user has cancelled the download.
    /// Contains the file_name of the cancelled download.
    #[cfg(feature = "downloader")]
    #[error("Download Aborted: {0}")]
    DownloadAborted(String),
    /// [serde_json::Error]
    #[error("JSON Parse Error {0}")]
    #[cfg(feature = "integrity")]
    JsonParseError(#[from] serde_json::Error),
    #[error("ModelError")]
    ModelError(String),
}
