//! The central, serializable checkpoint model — **the one datatype** all
//! primary metadata is read into, and everything else is derived from.
//!
//! A [`Checkpoint`] holds the filesystem structure of a checkpoint (every file's
//! size, on-disk allocation, kind, symlink target, permissions/mtime) *and* each
//! safetensors file's parsed header (tensors + `__metadata__`), plus the parsed
//! `config.json`, index health inputs, and — for an `s3://` source — the S3 object
//! metadata. Readers ([`crate::readers`], Stage 3) fill it in **one pass**; the
//! tensor tree, file tree, byte-layout map, and every report are then pure
//! functions of it with **no further disk access**. It round-trips through JSON
//! (and any other serde format), which is the on-the-wire contract for the future
//! web-server / MCP frontends.

use crate::config::ModelConfig;
use crate::filetree::FileKind;
use crate::remote::S3Meta;
use crate::stats::DiskUsage;
use crate::tree::{MetadataInfo, TensorInfo};

/// Where a checkpoint was read from — determines how paths are interpreted and
/// which reader produced the model.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Source {
    /// A local directory / file on this machine.
    Local,
    /// A remote safetensors directory read over SFTP (`--ssh-read host /path`).
    Sftp { host: String, root: String },
    /// An `s3://…` cstorch checkpoint read via the remote host (`--ssh-read`).
    S3 { uri: String },
}

/// One entry in the checkpoint's directory tree — the unified filesystem metadata
/// that used to be scattered across `filetree::FileNode`, `stats::ShardDisk`,
/// `sftp::RemoteStat`, and `remote::S3Object`. Sizes are **symlink-followed**
/// (the single-source-of-truth invariant): `apparent` is `st_size` of the target,
/// `allocated` its on-disk block allocation (0 when unknown, e.g. over SFTP
/// without a `stat -L`, or for an s3 object).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FileEntry {
    /// Path relative to the checkpoint root (POSIX `/`-separated).
    pub rel_path: String,
    /// The final path component (the display name).
    pub name: String,
    /// Depth below the root (0 = a top-level entry).
    pub depth: usize,
    /// Apparent size in bytes (`st_size`, symlink target). 0 for a directory.
    pub apparent: u64,
    /// On-disk allocation in bytes (`st_blocks × block-size`), or 0 when unknown.
    pub allocated: u64,
    pub is_dir: bool,
    pub kind: FileKind,
    /// The link target when this entry is a symlink (kept for display), else None.
    pub symlink_target: Option<String>,
    /// Unix mode bits, when known (local reads).
    pub mode: Option<u32>,
    /// Modification time (seconds since the epoch), when known.
    pub mtime: Option<i64>,
}

/// One safetensors file's parsed header — the tensors it stores and its
/// `__metadata__`, plus the byte sizes needed for the layout map. Non-safetensors
/// checkpoint files (gguf/npy/hdf5) also land here, one `ShardHeader` per file.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ShardHeader {
    /// The `source_path` these tensors carry (a local path, or a remote marker).
    pub path: String,
    /// Whole-file size in bytes (for the layout map's trailing gap); 0 if unknown.
    pub total_len: u64,
    /// Size of the header region (`8 + N` for safetensors), or 0 for other formats.
    pub header_len: u64,
    pub tensors: Vec<TensorInfo>,
    pub metadata: Vec<MetadataInfo>,
}

/// A checkpoint's `model.safetensors.index.json` (the pieces the health check
/// needs), in a serde-friendly form (no `PathBuf`).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct IndexEntry {
    /// The index file's path (display form).
    pub path: String,
    /// tensor name → shard file basename.
    pub weight_map: std::collections::BTreeMap<String, String>,
}

/// The one serializable checkpoint model. Read once; everything derives from it.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Checkpoint {
    pub source: Source,
    /// The checkpoint's root directory / prefix (display form) — what `f` on the
    /// tree root copies, and the base `rel_path`s are relative to.
    pub root: String,
    /// Every file in the checkpoint directory (recursively), for the file browser
    /// and the on-disk stats — no further `readdir`/`stat` needed after this.
    pub files: Vec<FileEntry>,
    /// Per-file parsed headers, for the tensor tree, layout map, and reports.
    pub shards: Vec<ShardHeader>,
    /// The parsed `config.json`, when present.
    pub config: Option<ModelConfig>,
    /// Parsed index(es), for the health check.
    pub index: Vec<IndexEntry>,
    /// S3 object metadata — `Some` only for an `s3://` source.
    pub s3: Option<S3Meta>,
}

impl Checkpoint {
    /// Every tensor across all shards, in shard order (the flattened primary
    /// tensor list the tree / stats / diff consume).
    pub fn tensors(&self) -> impl Iterator<Item = &TensorInfo> {
        self.shards.iter().flat_map(|s| s.tensors.iter())
    }

    /// Every `__metadata__` entry across all shards, in shard order.
    pub fn metadata(&self) -> impl Iterator<Item = &MetadataInfo> {
        self.shards.iter().flat_map(|s| s.metadata.iter())
    }

    /// Owned copies of the flattened tensors — a bridge for the (still
    /// `Vec<TensorInfo>`-based) views/reports until they take `&Checkpoint`.
    pub fn tensors_vec(&self) -> Vec<TensorInfo> {
        self.tensors().cloned().collect()
    }

    /// Owned copies of the flattened metadata (same bridging role).
    pub fn metadata_vec(&self) -> Vec<MetadataInfo> {
        self.metadata().cloned().collect()
    }

    /// The on-disk footprint rolled up from every **checkpoint file** the walk
    /// found (all `.safetensors`/`.gguf`/… in the directory, not just the loaded
    /// shards) — the `DiskUsage` the stats "on disk" section shows, now derived
    /// from the cached model (symlink-followed sizes) instead of a live `stat`.
    pub fn disk_usage(&self) -> Option<DiskUsage> {
        use crate::stats::ShardDisk;
        let shards: Vec<ShardDisk> = self
            .files
            .iter()
            .filter(|f| !f.is_dir && f.kind == FileKind::Checkpoint)
            .map(|f| ShardDisk {
                name: f.name.clone(),
                apparent: f.apparent,
                allocated: f.allocated,
            })
            .collect();
        DiskUsage::from_shards(shards)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::{Layout, Storage};

    fn sample() -> Checkpoint {
        Checkpoint {
            source: Source::Sftp {
                host: "net004".into(),
                root: "/opt/ckpt".into(),
            },
            root: "/opt/ckpt".into(),
            files: vec![FileEntry {
                rel_path: "model-00001-of-00002.safetensors".into(),
                name: "model-00001-of-00002.safetensors".into(),
                depth: 0,
                apparent: 4_000_000_000,
                allocated: 4_000_000_000,
                is_dir: false,
                kind: FileKind::Checkpoint,
                symlink_target: Some("/blobs/abc".into()),
                mode: Some(0o644),
                mtime: Some(1_700_000_000),
            }],
            shards: vec![ShardHeader {
                path: "net004:/opt/ckpt/model-00001-of-00002.safetensors".into(),
                total_len: 4_000_000_000,
                header_len: 8 + 512,
                tensors: vec![TensorInfo {
                    name: "model.embed_tokens.weight".into(),
                    dtype: "BF16".into(),
                    shape: vec![152064, 4096],
                    size_bytes: 152064 * 4096 * 2,
                    num_elements: 152064 * 4096,
                    storage: Storage::Unknown,
                    source_path: "net004:/opt/ckpt/model-00001-of-00002.safetensors".into(),
                    layout: Layout::ByteRange {
                        start: 0,
                        end: 1245708288,
                    },
                }],
                metadata: vec![MetadataInfo {
                    name: "format".into(),
                    value: "pt".into(),
                    value_type: "string".into(),
                }],
            }],
            config: Some(ModelConfig {
                model_type: Some("qwen3_moe".into()),
                num_hidden_layers: Some(48),
                ..Default::default()
            }),
            index: vec![IndexEntry {
                path: "model.safetensors.index.json".into(),
                weight_map: [(
                    "model.embed_tokens.weight".to_string(),
                    "model-00001-of-00002.safetensors".to_string(),
                )]
                .into_iter()
                .collect(),
            }],
            s3: None,
        }
    }

    #[test]
    fn checkpoint_round_trips_through_json() {
        let cp = sample();
        let json = serde_json::to_string(&cp).unwrap();
        let back: Checkpoint = serde_json::from_str(&json).unwrap();
        assert_eq!(back.root, "/opt/ckpt");
        assert_eq!(back.tensors().count(), 1);
        assert_eq!(back.metadata().count(), 1);
        assert_eq!(back.source, cp.source);
        // Disk usage is rolled up from the file entries (symlink-followed sizes).
        let disk = back.disk_usage().unwrap();
        assert_eq!(disk.total_apparent, 4_000_000_000);
        assert_eq!(disk.shards.len(), 1);
        // config + index + symlink target survive the round-trip.
        assert_eq!(back.config.unwrap().num_hidden_layers, Some(48));
        assert_eq!(back.index[0].weight_map.len(), 1);
        assert_eq!(back.files[0].symlink_target.as_deref(), Some("/blobs/abc"));
    }
}
