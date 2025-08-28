use std::fs::{self, File};
use std::path::{Path, PathBuf};

use crate::utils::errors::RibbleWhisperError;

pub mod downloaders;

/// For downloading an object synchronously (blocking)
pub trait SyncDownload: Writable {
    /// Downloads from the URL and returns the sanitized file stem
    fn download(&mut self, file_directory: &Path) -> Result<PathBuf, RibbleWhisperError>;
}

/// For downloading an object asynchronously (non-blocking, requires async runtime)
#[cfg(feature = "downloader-async")]
pub trait AsyncDownload: Writable {
    /// Downloads from the URL and returns the file destination.
    fn download(
        &mut self,
        file_directory: &Path,
    ) -> impl Future<Output = Result<PathBuf, RibbleWhisperError>>;
}

/// To handle basic IO operations when downlading files
pub trait Writable {
    // If a file path does not already exist, it will be created.
    fn prepare_file_path(file_directory: &Path) -> Result<(), RibbleWhisperError> {
        if !file_directory.exists() {
            fs::create_dir_all(file_directory)?;
        }

        Ok(())
    }

    fn open_write_file(file_path: &Path) -> Result<File, RibbleWhisperError> {
        let dest = fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(file_path)?;

        Ok(dest)
    }
}
