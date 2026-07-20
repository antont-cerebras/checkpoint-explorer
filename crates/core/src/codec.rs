//! Compression codecs offered by the repack / `convert` command. Kept in its
//! own (ungated) module so the CLI argument can use it even in a build without
//! the `hdf5` feature; the actual repack lives in [`crate::convert`].

/// A codec for re-compressing HDF5 datasets. Frontend-free: the CLI parses it via
/// [`std::str::FromStr`] (see the bin), so no `clap` dependency leaks into core.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
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
    Uncompressed,
}

impl std::fmt::Display for Codec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

impl std::str::FromStr for Codec {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "gzip" => Ok(Codec::Gzip),
            "zstd" => Ok(Codec::Zstd),
            "lz4" => Ok(Codec::Lz4),
            "none" | "store" | "uncompressed" => Ok(Codec::Uncompressed),
            other => Err(format!(
                "unknown codec '{other}' (expected gzip, zstd, lz4, or none)"
            )),
        }
    }
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
