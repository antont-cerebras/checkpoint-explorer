//! Readers: the **only** code that touches disk / SSH. Each fills the central
//! [`crate::model::Checkpoint`] in one pass — the filesystem walk, every
//! safetensors (and gguf/npy/hdf5) header, `config.json`, and the index — so that
//! afterwards the tensor tree, file browser, byte-layout map, and reports are all
//! pure functions of the cached model with no further disk access.
//!
//! This module owns the local reader; the remote (SFTP / s3-cstorch) readers stay
//! in [`crate::remote`] / [`crate::sftp`] and are adapted to produce a
//! `Checkpoint` in a later step.

use std::io::{Read, Seek};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::filetree::FileKind;
use crate::model::{Checkpoint, FileEntry, FsNode, IndexEntry, ShardHeader, Source};
use crate::tree::{Layout, MetadataInfo, Storage, TensorInfo};

/// Read a local checkpoint (a directory, a single file, or several files) fully
/// into a [`Checkpoint`]: the recursive filesystem walk (sizes symlink-followed,
/// with on-disk allocation / mode / mtime), every checkpoint file's header, the
/// sidecar `config.json`, and any `model.safetensors.index.json`.
pub fn read_local(files: &[PathBuf]) -> Result<Checkpoint> {
    let root = common_root(files);
    let root_str = root.to_string_lossy().into_owned();

    // The whole directory tree — one walk, reused by the file browser and the
    // on-disk stats (no later `readdir`/`stat`).
    let mut entries = Vec::new();
    walk(&root, &root, 0, &mut entries);
    entries.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));

    // Per checkpoint file: its parsed header (header-only; never the tensor data).
    let mut shards = Vec::new();
    for file_path in files {
        if let Some(shard) = read_shard_header(file_path)? {
            shards.push(shard);
        }
    }

    let config = crate::config::load_local(files);
    let index = read_indexes(&root);

    Ok(Checkpoint {
        source: Source::Local,
        root: root_str,
        files: entries,
        shards,
        config,
        index,
        s3: None,
    })
}

/// The directory a set of paths shares — a single file's parent, or the common
/// parent of several; `.` when there's nothing to anchor to.
fn common_root(files: &[PathBuf]) -> PathBuf {
    match files {
        [] => PathBuf::from("."),
        [one] => one
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf),
        many => {
            // Longest shared directory prefix by component.
            let first = many[0].parent().unwrap_or(Path::new("."));
            let mut common = first.to_path_buf();
            for f in &many[1..] {
                let p = f.parent().unwrap_or(Path::new("."));
                while !p.starts_with(&common) {
                    if !common.pop() {
                        return PathBuf::from(".");
                    }
                }
            }
            if common.as_os_str().is_empty() {
                PathBuf::from(".")
            } else {
                common
            }
        }
    }
}

/// Recursively collect [`FileEntry`]s under `dir`. Symlinks are followed for size
/// (matching the file browser / layout invariant) but a symlinked directory is a
/// leaf (not descended) so the walk can't cycle. Dotfiles are skipped.
fn walk(root: &Path, dir: &Path, depth: usize, out: &mut Vec<FileEntry>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in rd.flatten() {
        let name = match entry.file_name().to_str() {
            Some(n) if !n.starts_with('.') => n.to_string(),
            _ => continue,
        };
        let path = entry.path();
        let is_symlink = entry.file_type().ok().is_some_and(|t| t.is_symlink());
        // Followed metadata (target), with the link's own as a broken-link fallback.
        let meta = std::fs::metadata(&path).or_else(|_| entry.metadata());
        let (is_dir, apparent, allocated, mode, mtime, links, inode) = match &meta {
            Ok(m) => (
                m.is_dir(),
                if m.is_dir() { 0 } else { m.len() },
                block_bytes(m),
                unix_mode(m),
                mtime_secs(m),
                nlink(m),
                inode_of(m),
            ),
            Err(_) => (false, 0, 0, None, None, 1, None),
        };
        // A real directory descends; a symlinked directory stays a leaf.
        let descendable = is_dir && !is_symlink;
        let rel_path = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .into_owned();
        // Classify into the tagged fs-node: a symlink (with its followed sizes),
        // a real directory, or a regular file.
        let node = if is_symlink {
            let target = std::fs::read_link(&path)
                .ok()
                .map(|t| t.to_string_lossy().into_owned())
                .unwrap_or_default();
            FsNode::Symlink {
                target,
                apparent,
                allocated,
                kind: FileKind::of(&name),
                links,
            }
        } else if descendable {
            FsNode::Directory
        } else {
            FsNode::File {
                apparent,
                allocated,
                kind: FileKind::of(&name),
                links,
            }
        };
        out.push(FileEntry {
            rel_path,
            name: name.clone(),
            depth,
            mode,
            mtime,
            inode,
            node,
        });
        if descendable {
            walk(root, &path, depth + 1, out);
        }
    }
}

#[cfg(unix)]
fn block_bytes(m: &std::fs::Metadata) -> u64 {
    use std::os::unix::fs::MetadataExt;
    if m.is_dir() { 0 } else { m.blocks() * 512 }
}
#[cfg(not(unix))]
fn block_bytes(_m: &std::fs::Metadata) -> u64 {
    0
}

#[cfg(unix)]
fn unix_mode(m: &std::fs::Metadata) -> Option<u32> {
    use std::os::unix::fs::MetadataExt;
    Some(m.mode())
}
#[cfg(not(unix))]
fn unix_mode(_m: &std::fs::Metadata) -> Option<u32> {
    None
}

/// Hard-link count (`st_nlink`) of the (followed) target; `1` when unknown.
#[cfg(unix)]
fn nlink(m: &std::fs::Metadata) -> u64 {
    use std::os::unix::fs::MetadataExt;
    m.nlink()
}
#[cfg(not(unix))]
fn nlink(_m: &std::fs::Metadata) -> u64 {
    1
}

/// The (followed) inode number (`st_ino`), for the on-disk dedup; `None` off-Unix.
#[cfg(unix)]
fn inode_of(m: &std::fs::Metadata) -> Option<u64> {
    use std::os::unix::fs::MetadataExt;
    Some(m.ino())
}
#[cfg(not(unix))]
fn inode_of(_m: &std::fs::Metadata) -> Option<u64> {
    None
}

fn mtime_secs(m: &std::fs::Metadata) -> Option<i64> {
    m.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
}

/// Read one checkpoint file's header into a [`ShardHeader`], dispatching by
/// extension. Non-checkpoint files (and unsupported formats) yield `None`.
fn read_shard_header(file_path: &Path) -> Result<Option<ShardHeader>> {
    let source_path = absolute_path(file_path);
    match file_path.extension().and_then(|s| s.to_str()) {
        Some("safetensors") => {
            let mut file = std::fs::File::open(file_path)
                .with_context(|| format!("Failed to open file: {}", file_path.display()))?;
            let total_len = file.metadata().map(|m| m.len()).unwrap_or(0);
            let mut len_buf = [0u8; 8];
            file.read_exact(&mut len_buf).with_context(|| {
                format!("Failed to read header length: {}", file_path.display())
            })?;
            let n = crate::stheader::header_len(u64::from_le_bytes(len_buf), &source_path)?;
            let mut header_buf = vec![0u8; n];
            file.read_exact(&mut header_buf)
                .with_context(|| format!("Failed to read header: {}", file_path.display()))?;
            let (tensors, metadata) = crate::stheader::parse_header(&header_buf, &source_path)?;
            Ok(Some(ShardHeader {
                path: source_path,
                total_len,
                header_len: 8 + n as u64,
                tensors,
                metadata,
            }))
        }
        Some("gguf") => {
            let (tensors, metadata) = read_gguf(file_path, &source_path)?;
            Ok(Some(shard(source_path, file_path, tensors, metadata)))
        }
        Some("npy") => {
            let (tensors, metadata) = read_numpy(file_path, &source_path)?;
            Ok(Some(shard(source_path, file_path, tensors, metadata)))
        }
        Some("npz") => {
            let (tensors, metadata) = read_npz(file_path, &source_path)?;
            Ok(Some(shard(source_path, file_path, tensors, metadata)))
        }
        Some("h5") | Some("hdf5") => {
            #[cfg(feature = "hdf5")]
            {
                let (tensors, metadata) = crate::hdf5::read(file_path)?;
                Ok(Some(shard(source_path, file_path, tensors, metadata)))
            }
            #[cfg(not(feature = "hdf5"))]
            {
                Ok(None)
            }
        }
        _ => Ok(None),
    }
}

/// A [`ShardHeader`] for a non-safetensors format: `total_len` = the file size,
/// `header_len` = 0 (no safetensors-style header region).
fn shard(
    source_path: String,
    file_path: &Path,
    tensors: Vec<TensorInfo>,
    metadata: Vec<MetadataInfo>,
) -> ShardHeader {
    let total_len = std::fs::metadata(file_path).map(|m| m.len()).unwrap_or(0);
    ShardHeader {
        path: source_path,
        total_len,
        header_len: 0,
        tensors,
        metadata,
    }
}

fn read_gguf(file_path: &Path, source_path: &str) -> Result<(Vec<TensorInfo>, Vec<MetadataInfo>)> {
    use crate::gguf::{GGUFFile, GGUFValue};
    let mut file = std::fs::File::open(file_path)
        .with_context(|| format!("Failed to open file: {}", file_path.display()))?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .with_context(|| format!("Failed to read file: {}", file_path.display()))?;
    let gguf = GGUFFile::read(&buffer)
        .with_context(|| format!("Failed to parse GGUF file: {}", file_path.display()))?;
    let mut metadata = Vec::new();
    for (key, value) in &gguf.metadata {
        let value_type = match value {
            GGUFValue::U8(_) => "u8",
            GGUFValue::I8(_) => "i8",
            GGUFValue::U16(_) => "u16",
            GGUFValue::I16(_) => "i16",
            GGUFValue::U32(_) => "u32",
            GGUFValue::I32(_) => "i32",
            GGUFValue::F32(_) => "f32",
            GGUFValue::U64(_) => "u64",
            GGUFValue::I64(_) => "i64",
            GGUFValue::F64(_) => "f64",
            GGUFValue::Bool(_) => "bool",
            GGUFValue::String(_) => "string",
            GGUFValue::Array(_) => "array",
        };
        metadata.push(MetadataInfo {
            name: key.clone(),
            value: value.to_string(),
            value_type: value_type.to_string(),
        });
    }
    let mut tensors = Vec::new();
    for tensor in &gguf.tensors {
        let shape: Vec<usize> = tensor.dimensions.iter().map(|&d| d as usize).collect();
        let num_elements = shape.iter().product::<usize>();
        let size_bytes = (num_elements as f32 * tensor.tensor_type.element_size_bytes()) as usize;
        tensors.push(TensorInfo {
            name: tensor.name.clone(),
            dtype: tensor.tensor_type.to_string(),
            shape,
            size_bytes,
            num_elements,
            storage: Storage::Unknown,
            source_path: source_path.to_string(),
            layout: Layout::Offset(tensor.offset),
        });
    }
    Ok((tensors, metadata))
}

fn read_numpy(file_path: &Path, source_path: &str) -> Result<(Vec<TensorInfo>, Vec<MetadataInfo>)> {
    let mut file = std::fs::File::open(file_path)
        .with_context(|| format!("Failed to open file: {}", file_path.display()))?;
    let total_len = file.metadata().map(|m| m.len()).unwrap_or(0);
    let name = file_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("array")
        .to_string();
    let header =
        crate::npy::parse_header(&mut file).map_err(|e| anyhow::anyhow!("{source_path}: {e}"))?;
    let num_elements = header.shape.iter().product::<usize>();
    let tensor = TensorInfo {
        name,
        dtype: header.dtype,
        shape: header.shape,
        size_bytes: (total_len as usize).saturating_sub(header.data_offset),
        num_elements,
        storage: Storage::Unknown,
        source_path: source_path.to_string(),
        layout: Layout::ByteRange {
            start: header.data_offset as u64,
            end: total_len,
        },
    };
    Ok((vec![tensor], Vec::new()))
}

fn read_npz(file_path: &Path, source_path: &str) -> Result<(Vec<TensorInfo>, Vec<MetadataInfo>)> {
    let file = std::fs::File::open(file_path)
        .with_context(|| format!("Failed to open file: {}", file_path.display()))?;
    read_npz_reader(file, source_path)
}

fn read_npz_reader<R: Read + Seek>(
    reader: R,
    source_path: &str,
) -> Result<(Vec<TensorInfo>, Vec<MetadataInfo>)> {
    let mut tensors = Vec::new();
    let mut zip = zip::ZipArchive::new(reader)
        .with_context(|| format!("Failed to read .npz archive: {source_path}"))?;
    let entries: Vec<String> = zip.file_names().map(String::from).collect();
    for entry_name in entries {
        let Some(name) = entry_name.strip_suffix(".npy") else {
            continue;
        };
        let mut entry = zip
            .by_name(&entry_name)
            .with_context(|| format!("Failed to read {entry_name} in {source_path}"))?;
        let stored_bytes = entry.compressed_size() as usize;
        let uncompressed = entry.size() as usize;
        let compressed = entry.compression() != zip::CompressionMethod::Stored;
        let header = crate::npy::parse_header(&mut entry)
            .map_err(|e| anyhow::anyhow!("{source_path}: {entry_name}: {e}"))?;
        let num_elements = header.shape.iter().product::<usize>();
        let storage = if compressed {
            Storage::Compressed {
                codec: "deflate".to_string(),
                stored_bytes,
            }
        } else {
            Storage::Raw
        };
        tensors.push(TensorInfo {
            name: name.to_string(),
            dtype: header.dtype,
            shape: header.shape,
            size_bytes: uncompressed.saturating_sub(header.data_offset),
            num_elements,
            storage,
            source_path: source_path.to_string(),
            layout: Layout::None,
        });
    }
    Ok((tensors, Vec::new()))
}

/// Read every `model.safetensors.index.json` under `root` into serde-friendly
/// [`IndexEntry`]s (for the health check).
fn read_indexes(root: &Path) -> Vec<IndexEntry> {
    let mut out = Vec::new();
    let index_path = root.join("model.safetensors.index.json");
    if let Ok(bytes) = std::fs::read(&index_path)
        && let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes)
        && let Some(wm) = v.get("weight_map").and_then(|w| w.as_object())
    {
        let weight_map = wm
            .iter()
            .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
            .collect();
        out.push(IndexEntry {
            path: index_path.to_string_lossy().into_owned(),
            weight_map,
        });
    }
    out
}

/// Absolute path of `p` (best-effort: canonicalization-free, just prefixes the
/// current dir when `p` is relative) — the `source_path` tensors carry.
fn absolute_path(p: &Path) -> String {
    if p.is_absolute() {
        p.to_string_lossy().into_owned()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(p))
            .unwrap_or_else(|_| p.to_path_buf())
            .to_string_lossy()
            .into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a minimal safetensors file: one f32 tensor `w` of shape [2,2].
    fn write_st(path: &Path) {
        let header = r#"{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"__metadata__":{"format":"pt"}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0u8; 16]);
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn read_local_fills_the_model_in_one_pass() {
        let dir = std::env::temp_dir().join("ce_readers_local_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        write_st(&dir.join("model.safetensors"));
        std::fs::write(
            dir.join("config.json"),
            br#"{"model_type":"llama","num_hidden_layers":2}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            br#"{"weight_map":{"w":"model.safetensors"}}"#,
        )
        .unwrap();

        let cp = read_local(&[dir.join("model.safetensors")]).unwrap();
        // Header parsed into the shard.
        assert_eq!(cp.shards.len(), 1);
        assert_eq!(cp.tensors().count(), 1);
        assert_eq!(cp.tensors().next().unwrap().name, "w");
        assert_eq!(cp.metadata().count(), 1);
        assert!(cp.shards[0].total_len > cp.shards[0].header_len);
        // Filesystem walk captured the files (with sizes), so the browser + on-disk
        // stats need no further disk access.
        assert!(
            cp.files
                .iter()
                .any(|f| f.name == "model.safetensors" && f.apparent() > 0)
        );
        assert!(cp.files.iter().any(|f| f.name == "config.json"));
        // Sidecar config + index parsed in the same pass.
        assert_eq!(
            cp.config.as_ref().unwrap().model_type.as_deref(),
            Some("llama")
        );
        assert_eq!(cp.index.len(), 1);
        assert_eq!(
            cp.index[0].weight_map.get("w").map(String::as_str),
            Some("model.safetensors")
        );
        // The whole model serializes.
        let json = serde_json::to_string(&cp).unwrap();
        assert!(json.contains("\"model_type\":\"llama\""));

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The walk classifies each entry into the tagged [`FsNode`]: a regular file,
    /// a real directory, and a symlink (carrying its raw target + followed size).
    #[cfg(unix)]
    #[test]
    fn walk_tags_files_dirs_and_symlinks() {
        use crate::model::FsNode;
        let dir = std::env::temp_dir().join("ce_readers_fsnode_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        write_st(&dir.join("model.safetensors"));
        // An HF-cache-style symlink to the real shard.
        std::os::unix::fs::symlink(dir.join("model.safetensors"), dir.join("link.safetensors"))
            .unwrap();

        let cp = read_local(&[dir.join("model.safetensors")]).unwrap();
        let node = |name: &str| {
            cp.files
                .iter()
                .find(|f| f.name == name)
                .map(|f| f.node.clone())
        };

        // Regular shard → File, with a nonzero size and the Checkpoint content kind.
        assert!(matches!(
            node("model.safetensors"),
            Some(FsNode::File { kind: FileKind::Checkpoint, apparent, .. }) if apparent > 0
        ));
        // Subdirectory → Directory (no size fields at all).
        assert!(matches!(node("sub"), Some(FsNode::Directory)));
        // Symlink → Symlink, carrying its raw target and the *followed* size/kind.
        match node("link.safetensors") {
            Some(FsNode::Symlink {
                target,
                apparent,
                kind,
                ..
            }) => {
                assert!(target.ends_with("model.safetensors"), "target: {target}");
                assert!(apparent > 0, "followed size");
                assert_eq!(kind, FileKind::Checkpoint);
            }
            other => panic!("expected a symlink fs-node, got {other:?}"),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Two hard links to one shard share an inode, so the on-disk rollup counts
    /// its bytes once (not twice) — and the walk records `links > 1`.
    #[cfg(unix)]
    #[test]
    fn disk_usage_dedups_hardlinked_shards() {
        use crate::model::FsNode;
        let dir = std::env::temp_dir().join("ce_readers_hardlink_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        write_st(&dir.join("model.safetensors"));
        // A second name for the same inode (a hard link, not a symlink).
        std::fs::hard_link(
            dir.join("model.safetensors"),
            dir.join("model-copy.safetensors"),
        )
        .unwrap();

        let cp = read_local(&[dir.join("model.safetensors")]).unwrap();
        // Both names are regular files reporting links == 2 (the inode has 2 names).
        for name in ["model.safetensors", "model-copy.safetensors"] {
            let f = cp.files.iter().find(|f| f.name == name).unwrap();
            assert!(
                matches!(f.node, FsNode::File { links: 2, .. }),
                "{name} should report 2 hard links, got {:?}",
                f.node
            );
        }
        // Two shard files on disk, but one physical inode → counted once.
        let disk = cp.disk_usage().unwrap();
        assert_eq!(disk.shards.len(), 1, "shared inode counted once");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
