//! Frontend-free core of checkpoint-explorer.
//!
//! Holds the serializable checkpoint model, the readers (local + SSH/S3) that
//! fill it, the derived views (tensor tree, file tree, byte layout), and the
//! reports (stats, health check, diff) — with **no** terminal / TUI / CLI
//! dependency. Frontends (the interactive terminal, and future web-server / MCP
//! bins) drive it and render its serializable outputs.

pub mod check;
pub mod codec;
pub mod config;
pub mod diff;
pub mod filetree;
pub mod filter;
pub mod gguf;
pub mod health;
pub mod model;
pub mod npy;
pub mod progress;
pub mod remote;
pub mod rename;
pub mod s3;
pub mod safelayout;
pub mod sample;
pub mod sftp;
pub mod stats;
pub mod stheader;
pub mod tree;
pub mod utils;

#[cfg(feature = "hdf5")]
pub mod convert;
#[cfg(feature = "hdf5")]
pub mod hdf5;
#[cfg(feature = "hdf5")]
pub mod hdf5_lz4;
#[cfg(feature = "hdf5")]
pub mod hdf5_zstd;
