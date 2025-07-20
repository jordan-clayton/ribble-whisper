#![doc = include_str!("../README.md")]
pub mod audio;
#[cfg(feature = "downloader")]
pub mod downloader;
pub mod transcriber;
pub mod utils;
pub mod whisper;
// Export sdl2 when using SDL as the audio backend.
#[cfg(feature = "sdl2")]
pub use sdl2;
