#[cfg(feature = "downloader-async")]
use std::io::Write;
use std::io::{copy, Read};
use std::path::{Path, PathBuf};

#[cfg(feature = "downloader-async")]
use bytes::Bytes;
#[cfg(feature = "downloader-async")]
use futures::StreamExt;
#[cfg(feature = "downloader-async")]
use futures_core::stream::Stream;
use url::Url;

#[cfg(feature = "downloader-async")]
use crate::downloader::AsyncDownload;
use crate::downloader::{SyncDownload, Writable};
use crate::utils::callback::{AbortCallback, Callback, Nop};
use crate::utils::errors::RibbleWhisperError;

const TEMP_FILE_EXTENSION: &'static str = ".tmp";

/// Streams in bytes (asynchronously) to download data.
/// Current progress can be obtained by supplying a Callback.
/// It is recommended to use [async_download_request] to construct this object and use the builder to set the
/// callbacks over manual creation.
/// Note: At this time, the content_name is fixed at creation time.
#[cfg(feature = "downloader-async")]
pub struct StreamDownloader<S, CB, A>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
    CB: Callback<Argument = usize>,
    A: AbortCallback,
{
    // Bytestream
    file_stream: S,
    // In most cases this will be a file_name.
    content_name: String,
    // Total progress thus far
    progress: usize,
    // Total download size
    /// This is retrieved from the content-length field from an HTTP header. If this value is not set,
    /// the body has no size, or gzip compression is being used, this will be 1.
    /// Treat this as "indeterminate", as there is no way to measure the body without downloading it.
    total_size: usize,
    // Optional progress callback. Default is a Nop
    progress_callback: CB,
    abort_callback: A,
}
#[cfg(feature = "downloader-async")]
impl<S> StreamDownloader<S, Nop<usize>, Nop<()>>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
{
    /// Returns a StreamDownloader with the default (NOP) callback
    pub fn new_with_parameters(file_stream: S, content_name: String, total_size: usize) -> Self {
        Self {
            file_stream,
            content_name,
            progress: 0,
            total_size,
            progress_callback: Nop::new(),
            abort_callback: Nop::new(),
        }
    }
}

#[cfg(feature = "downloader-async")]
impl<S, CB, A> StreamDownloader<S, CB, A>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
    CB: Callback<Argument = usize>,
    A: AbortCallback,
{
    /// Returns a StreamDownloader with both a ProgressCallback and an AbortCallback
    pub fn new_full(
        file_stream: S,
        content_name: String,
        total_size: usize,
        progress_callback: CB,
        abort_callback: A,
    ) -> Self {
        Self {
            file_stream,
            content_name,
            progress: 0,
            total_size,
            progress_callback,
            abort_callback,
        }
    }

    /// Returns a StreamDownloader with a set ProgressCallback
    pub fn new_with_parameters_and_progress_callback(
        file_stream: S,
        content_name: String,
        total_size: usize,
        progress_callback: CB,
    ) -> StreamDownloader<S, CB, Nop<()>> {
        StreamDownloader {
            file_stream,
            content_name,
            progress: 0,
            total_size,
            progress_callback,
            abort_callback: Nop::new(),
        }
    }

    pub fn new_with_parameters_and_abort_callback(
        file_stream: S,
        content_name: String,
        total_size: usize,
        abort_callback: A,
    ) -> StreamDownloader<S, Nop<usize>, A> {
        StreamDownloader {
            file_stream,
            content_name,
            progress: 0,
            total_size,
            progress_callback: Nop::new(),
            abort_callback,
        }
    }

    /// Sets the file stream to be downloaded
    pub fn with_file_stream<S2>(self, file_stream: S2) -> StreamDownloader<S2, CB, A>
    where
        S2: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
    {
        StreamDownloader::new_full(
            file_stream,
            self.content_name,
            self.total_size,
            self.progress_callback,
            self.abort_callback,
        )
    }

    /// Sets an optional progress callback.
    /// To un-set a callback, supply [Nop]
    pub fn with_progress_callback<C>(self, progress_callback: C) -> StreamDownloader<S, C, A>
    where
        C: Callback<Argument = usize>,
    {
        StreamDownloader::new_full(
            self.file_stream,
            self.content_name,
            self.total_size,
            progress_callback,
            self.abort_callback,
        )
    }

    pub fn with_abort_callback<A2>(self, abort_callback: A2) -> StreamDownloader<S, CB, A2>
    where
        A2: AbortCallback,
    {
        StreamDownloader::new_full(
            self.file_stream,
            self.content_name,
            self.total_size,
            self.progress_callback,
            abort_callback,
        )
    }

    /// Gets the download's total size
    pub fn total_size(&self) -> usize {
        self.total_size
    }
    /// Gets the download's content_name
    pub fn content_name(&self) -> &str {
        self.content_name.as_str()
    }
}

#[cfg(feature = "downloader-async")]
impl<S, CB, A> Writable for StreamDownloader<S, CB, A>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
    CB: Callback<Argument = usize>,
    A: AbortCallback,
{
}

#[cfg(feature = "downloader-async")]
impl<S, CB, A> AsyncDownload for StreamDownloader<S, CB, A>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
    CB: Callback<Argument = usize>,
    A: AbortCallback,
{
    /// Downloads a given file asynchronously to the desired location. Returns Err on I/O failure.
    /// This function must be awaited and should not be called on a UI thread.
    async fn download(&mut self, file_directory: &Path) -> Result<PathBuf, RibbleWhisperError> {
        Self::prepare_file_path(file_directory)?;

        let file_path = file_directory.join(&self.content_name);
        let tmp_path = file_directory.join([&self.content_name, TEMP_FILE_EXTENSION].concat());

        let mut dest = Self::open_write_file(tmp_path.as_path())?;

        let stream = &mut self.file_stream;

        let cleanup = || {
            // It's not particularly necessary to know that the file has been successfully removed.
            // If this fails, it most likely didn't begin to the first place and cleanup isn't
            // required.
            if tmp_path.is_file() {
                let _ = std::fs::remove_file(&tmp_path.as_path());
            } else {
                let _ = std::fs::remove_dir(&tmp_path.as_path());
            }
        };

        while let Some(next) = stream.next().await {
            // Call the abort callback and escape if the download is cancelled
            if self.abort_callback.abort() {
                cleanup();
                return Err(RibbleWhisperError::DownloadAborted(
                    self.content_name.clone(),
                ));
            }

            let buf = next.or_else(|e| {
                cleanup();
                Err(e)
            })?;
            dest.write_all(&buf).or_else(|e| {
                cleanup();
                Err(e)
            })?;

            let mut cur_progress = self.progress;

            cur_progress = std::cmp::min(cur_progress + (buf.len()), self.total_size);
            self.progress = cur_progress;

            // Update the UI with the current progress
            self.progress_callback.call(self.progress);
        }

        std::fs::rename(tmp_path.as_path(), file_path.as_path())?;
        Ok(file_path)
    }
}

/// Downloads a file synchronously (blocking).
/// Current progress can be obtained by supplying a Callback.
/// It is recommended to use [sync_download_request] to construct this object and use the builder to set the
/// callbacks over manual creation.
/// Note: At this time, the content_name is fixed at creation time.
pub struct SyncDownloader<R, CB, A>
where
    R: Read,
    CB: Callback<Argument = usize>,
    A: AbortCallback,
{
    file_stream: R,
    content_name: String,
    progress: usize,
    /// This is retrieved from the content-length field from an HTTP header. If this value is not set,
    /// the body has no size, or gzip compression is being used, this will be 1.
    /// Treat this as "indeterminate", as there is no way to measure the body without downloading it.
    total_size: usize,
    progress_callback: CB,
    abort_callback: A,
}

impl<R: Read> SyncDownloader<R, Nop<usize>, Nop<()>> {
    /// Returns a SyncDownloader with the default (NOP) callback
    pub fn new_with_parameters(file_stream: R, content_name: String, total_size: usize) -> Self {
        Self {
            file_stream,
            content_name,
            progress: 0,
            total_size,
            progress_callback: Nop::new(),
            abort_callback: Nop::new(),
        }
    }
}

impl<R: Read, CB, A> SyncDownloader<R, CB, A>
where
    R: Read,
    CB: Callback<Argument = usize>,
    A: AbortCallback,
{
    /// Returns a SyncDownloader with both callbacks set.
    pub fn new_full(
        file_stream: R,
        content_name: String,
        total_size: usize,
        progress_callback: CB,
        abort_callback: A,
    ) -> Self {
        Self {
            file_stream,
            content_name,
            progress: 0,
            total_size,
            progress_callback,
            abort_callback,
        }
    }

    /// Returns a SyncDownloader with the provided optional progress callback.
    pub fn new_with_parameters_and_progress_callback(
        file_stream: R,
        content_name: String,
        total_size: usize,
        progress_callback: CB,
    ) -> SyncDownloader<R, CB, Nop<()>> {
        SyncDownloader {
            file_stream,
            content_name,
            progress: 0,
            total_size,
            progress_callback,
            abort_callback: Nop::new(),
        }
    }

    /// Sets the file stream.
    pub fn with_file_stream<R2: Read>(self, file_stream: R2) -> SyncDownloader<R2, CB, A> {
        SyncDownloader::new_full(
            file_stream,
            self.content_name,
            self.total_size,
            self.progress_callback,
            self.abort_callback,
        )
    }
    /// Sets the (optional) progress callback.
    /// To un-set the callback, supply a [Nop]
    pub fn with_progress_callback<C>(self, progress_callback: C) -> SyncDownloader<R, C, A>
    where
        C: Callback<Argument = usize>,
    {
        SyncDownloader::new_full(
            self.file_stream,
            self.content_name,
            self.total_size,
            progress_callback,
            self.abort_callback,
        )
    }

    pub fn with_abort_callback<A2>(self, abort_callback: A2) -> SyncDownloader<R, CB, A2>
    where
        A2: AbortCallback,
    {
        SyncDownloader::new_full(
            self.file_stream,
            self.content_name,
            self.total_size,
            self.progress_callback,
            abort_callback,
        )
    }

    /// Gets the download's total size
    pub fn total_size(&self) -> usize {
        self.total_size
    }

    /// Gets the download's content_name
    pub fn content_name(&self) -> &str {
        self.content_name.as_str()
    }
}

impl<R, CB, A> Read for SyncDownloader<R, CB, A>
where
    R: Read,
    CB: Callback<Argument = usize>,
    A: AbortCallback,
{
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.abort_callback.abort() {
            return Err(std::io::Error::from(std::io::ErrorKind::ConnectionAborted));
        }

        let byte_read = self.file_stream.read(buf);
        if let Ok(num_bytes) = byte_read {
            self.progress += num_bytes;

            // Update the UI with the current progress
            self.progress_callback.call(self.progress);
        };
        byte_read
    }
}

impl<R, CB, A> Writable for SyncDownloader<R, CB, A>
where
    R: Read,
    CB: Callback<Argument = usize>,
    A: AbortCallback,
{
}

impl<R, CB, A> SyncDownload for SyncDownloader<R, CB, A>
where
    R: Read,
    CB: Callback<Argument = usize>,
    A: AbortCallback,
{
    /// Downloads a file synchronously to the desired location. Returns Err on I/O failure.
    /// This will block the calling thread.
    fn download(&mut self, file_directory: &Path) -> Result<PathBuf, RibbleWhisperError> {
        Self::prepare_file_path(file_directory)?;

        let file_path = file_directory.join(&self.content_name);
        let tmp_path = file_directory.join([&self.content_name, TEMP_FILE_EXTENSION].concat());

        let mut dest = Self::open_write_file(tmp_path.as_path())?;

        let downloaded = copy(self, &mut dest).map_err(|e| {
            if e.kind() == std::io::ErrorKind::ConnectionAborted {
                RibbleWhisperError::DownloadAborted(self.content_name.clone())
            } else {
                e.into()
            }
        });

        if downloaded.is_err() {
            // It's not particularly necessary to know that the file has been successfully removed.
            // If this fails, it most likely didn't begin to the first place and cleanup isn't
            // required.
            if tmp_path.is_file() {
                let _ = std::fs::remove_file(tmp_path.as_path());
            } else {
                let _ = std::fs::remove_dir(tmp_path.as_path());
            }

            return Err(downloaded.err().unwrap().into());
        } else {
            // Otherwise, rename the temporary file to the file_path
            // Expect that this will never fail, but in case it does, the error will be returned.
            std::fs::rename(tmp_path.as_path(), file_path.as_path())?;
        }

        Ok(file_path)
    }
}

/// Returns a StreamDownloader that encapsulates the request bytestream, progress,
/// and total response size.
/// Call [StreamDownloader::with_progress_callback] to set an optional callback to receive
/// updates on the number of bytes downloaded.
///
/// NOTE: This function must be awaited and should not be called on a UI thread.
/// # Arguments:
/// * url: the download url
/// * fallback_file_name: a fallback name to use in-case response parsing fails
/// # Returns:
/// Ok(SyncDownloader) on success, Err on a failure to either send the request or get the content length
#[cfg(feature = "downloader-async")]
pub async fn async_download_request(
    url: &str,
    fallback_file_name: &str,
) -> Result<
    StreamDownloader<impl Stream<Item = Result<Bytes, reqwest::Error>>, Nop<usize>, Nop<()>>,
    RibbleWhisperError,
> {
    let m_url = Url::parse(url)?;
    let client = reqwest::Client::new();

    let res = client.get(m_url).send().await?;

    if !res.status().is_success() {
        return Err(RibbleWhisperError::DownloadError(format!(
            "Failed to download, status code: {}",
            res.status()
        )));
    }

    // If this returns size 1-byte, assume this either "no-body" or gzipped.
    // If that's the case, treat the download as "indeterminate"
    let total_size = res.content_length().unwrap_or(1) as usize;

    // Type-wrapper for covariant request structs
    let borrowed_resp = BorrowedDownloadResponse::Async(&res);

    let content_name = get_content_name(borrowed_resp).unwrap_or(fallback_file_name.to_string());
    let stream = res.bytes_stream();

    // Try to get the content_name
    // Return the appropriate streamdownloader.
    // Uh.
    Ok(StreamDownloader::new_with_parameters(
        stream,
        content_name,
        total_size,
    ))
}

/// Creates a SyncDownloader that encapsulates the request bytestream, progress,
/// and total response size.
/// Call [SyncDownloader::with_progress_callback] to set an optional callback to receive
/// updates on the number of bytes downloaded.
///
/// NOTE: SyncDownloaders are blocking and thus will block the calling thread.
/// # Arguments:
/// * url: the download url
/// * fallback_file_name: a fallback name to use in-case response parsing fails
/// # Returns:
/// Ok(SyncDownloader) on success, Err on a failure to either send the request or get the content length

pub fn sync_download_request(
    url: &str,
    fallback_file_name: &str,
) -> Result<SyncDownloader<impl Read, Nop<usize>, Nop<()>>, RibbleWhisperError> {
    let m_url = Url::parse(url)?;
    let client = reqwest::blocking::Client::new();

    let res = client.get(m_url).send()?;

    if !res.status().is_success() {
        return Err(RibbleWhisperError::DownloadError(format!(
            "Failed to download, status code: {}",
            res.status()
        )));
    }

    // If this returns size 1-byte, assume this either "no-body" or gzipped.
    // If that's the case, treat the download as "indeterminate"
    let total_size = res.content_length().unwrap_or(1) as usize;

    // Type-wrapper for covariant request structs
    let borrowed_resp = BorrowedDownloadResponse::Blocking(&res);

    // Try to get the content_name
    let content_name = get_content_name(borrowed_resp).unwrap_or(fallback_file_name.to_string());

    Ok(SyncDownloader::new_with_parameters(
        res,
        content_name,
        total_size,
    ))
}

enum BorrowedDownloadResponse<'a> {
    Async(&'a reqwest::Response),
    Blocking(&'a reqwest::blocking::Response),
}

impl BorrowedDownloadResponse<'_> {
    fn url(&self) -> &Url {
        match self {
            BorrowedDownloadResponse::Async(resp) => resp.url(),
            BorrowedDownloadResponse::Blocking(resp) => resp.url(),
        }
    }
    fn response_headers_get_all(
        &'_ self,
        header: reqwest::header::HeaderName,
    ) -> reqwest::header::GetAll<'_, reqwest::header::HeaderValue> {
        match self {
            BorrowedDownloadResponse::Async(resp) => resp.headers().get_all(header),
            BorrowedDownloadResponse::Blocking(resp) => resp.headers().get_all(header),
        }
    }
}

fn get_content_name(response: BorrowedDownloadResponse) -> Option<String> {
    let content_disp = response.response_headers_get_all(reqwest::header::CONTENT_DISPOSITION);
    let try_content_name = content_disp
        .iter()
        .find(|&val| {
            val.to_str()
                .ok()
                .is_some_and(|field| field.contains("filename"))
        })
        .and_then(|file_field| file_field.to_str().ok())
        .and_then(|file_string| file_string.split("filename=").last())
        .and_then(|filename| Some(filename.trim_end_matches(";")));

    if let Some(content_name) = try_content_name {
        return Some(sanitize_filename::sanitize(content_name));
    }

    // If the Content Disposition fails, try to grab the end of the url
    // This is most likely to be the filename
    response
        .url()
        .path_segments()
        .and_then(|segments| segments.last())
        .and_then(|name| {
            if name.is_empty() {
                None
            } else {
                Some(sanitize_filename::sanitize(name))
            }
        })
}
