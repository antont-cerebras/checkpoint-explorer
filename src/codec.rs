//! Compression codecs offered by the repack / `convert` command. Kept in its
//! own (ungated) module so the CLI argument can use it even in a build without
//! the `hdf5` feature; the actual repack lives in [`crate::convert`].

/// A codec for re-compressing HDF5 datasets.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, clap::ValueEnum)]
pub enum Codec {
    /// gzip / DEFLATE — built into libhdf5, entropy-coded (~3.5× on 4-bit
    /// weights), but slower.
    #[default]
    Gzip,
    /// Zstandard — registered in-process; best ratio with fast decode.
    Zstd,
    /// LZ4 — the format these checkpoints ship with: fast, but only ~2× on
    /// 4-bit weights (no entropy coding).
    Lz4,
    /// No compression.
    #[value(name = "none", aliases = ["store", "uncompressed"])]
    Uncompressed,
}

// Most accessors are only exercised by the `hdf5`-gated repack paths; in a
// default build only `label` is used (by the TUI codec menu).
#[cfg_attr(not(feature = "hdf5"), allow(dead_code))]
impl Codec {
    /// Short display label.
    pub fn label(self) -> &'static str {
        match self {
            Codec::Gzip => "gzip",
            Codec::Zstd => "zstd",
            Codec::Lz4 => "lz4",
            Codec::Uncompressed => "none",
        }
    }

    /// Whether a compression level applies (gzip/zstd) or is ignored (lz4/none).
    pub fn uses_level(self) -> bool {
        matches!(self, Codec::Gzip | Codec::Zstd)
    }

    /// The default level when the user doesn't specify one.
    pub fn default_level(self) -> u8 {
        match self {
            Codec::Gzip => 6, // 0–9
            Codec::Zstd => 3, // 1–22; 3 is fast, raise for more compression
            _ => 0,
        }
    }

    /// Clamp a level into the codec's valid range.
    pub fn clamp_level(self, level: u8) -> u8 {
        match self {
            Codec::Gzip => level.min(9),
            Codec::Zstd => level.clamp(1, 22),
            _ => 0,
        }
    }
}
