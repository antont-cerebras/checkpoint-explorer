//! Reader for Cerebras-style HDF5 checkpoints.
//!
//! These are plain HDF5 files where every top-level dataset is one tensor and
//! its link name is the URL-quoted tensor name (so `/` in a name is escaped as
//! `%2F`; `.` is left as-is, matching PyTorch state-dict names). The dataset's
//! own dataspace and datatype give the shape and dtype — we never read the
//! (possibly compressed, possibly huge) data itself. Datasets are often chunked
//! and gzip-compressed, so we also report the on-disk (compressed) size.

use anyhow::{Context, Result};
use hdf5_metno::filters::Filter;
use hdf5_metno::types::TypeDescriptor;

use crate::tree::{Layout, Storage, TensorInfo};

/// Read tensor metadata (name, dtype, shape, logical + on-disk size) from an
/// HDF5 checkpoint.
pub fn read_tensors(path: &std::path::Path) -> Result<Vec<TensorInfo>> {
    let file = hdf5_metno::File::open(path)
        .with_context(|| format!("Failed to open HDF5 file: {}", path.display()))?;

    let members = file
        .member_names()
        .with_context(|| format!("Failed to list members of: {}", path.display()))?;

    // Every tensor in this file shares the same source path.
    let source_path = std::path::absolute(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .into_owned();

    let mut tensors = Vec::with_capacity(members.len());
    for key in members {
        // Each top-level member is a tensor dataset; skip anything that is not a
        // dataset (e.g. a stray group) rather than failing the whole file.
        let Ok(dataset) = file.dataset(&key) else {
            continue;
        };

        let shape = dataset.shape();
        let num_elements: usize = shape.iter().product();

        let (dtype, item_size) = match dataset.dtype() {
            Ok(dt) => {
                let item = dt.size();
                let name = dt
                    .to_descriptor()
                    .ok()
                    .map(|d| dtype_name(&d))
                    .unwrap_or_else(|| "?".to_string());
                (name, item)
            }
            Err(_) => ("?".to_string(), 0),
        };

        // Logical (uncompressed) size, and the on-disk size when a compression
        // filter is in the pipeline.
        let size_bytes = num_elements * item_size;
        let storage = match dataset.filters().iter().find_map(compression_codec) {
            Some(codec) => {
                let stored = dataset.storage_size() as usize;
                if stored > 0 {
                    Storage::Compressed {
                        codec,
                        stored_bytes: stored,
                    }
                } else {
                    Storage::Raw
                }
            }
            None => Storage::Raw,
        };

        // HDF5 data is chunked rather than a flat slice; report the chunk shape
        // and count when present.
        let layout = match (dataset.chunk(), dataset.num_chunks()) {
            (Some(chunk), Some(num_chunks)) => Layout::Chunked { chunk, num_chunks },
            _ => Layout::None,
        };

        tensors.push(TensorInfo {
            name: percent_decode(&key),
            dtype,
            shape,
            size_bytes,
            num_elements,
            storage,
            source_path: source_path.clone(),
            layout,
        });
    }

    Ok(tensors)
}

/// Name the size-reducing compressor in an HDF5 filter, if any. Shuffle and
/// Fletcher32 reorder/checksum but do not compress, so they map to `None`.
fn compression_codec(filter: &Filter) -> Option<String> {
    match filter {
        Filter::Deflate(_) => Some("gzip".to_string()),
        Filter::SZip(..) => Some("szip".to_string()),
        Filter::ScaleOffset(_) => Some("scaleoffset".to_string()),
        Filter::NBit => Some("nbit".to_string()),
        // Third-party filters (e.g. LZ4, Zstd) appear as `User` when their
        // plugin isn't compiled in; name the well-known registered IDs.
        Filter::User(id, _) => user_filter_codec(*id),
        _ => None,
    }
}

/// Friendly name for a registered third-party HDF5 filter id, or `None` for
/// filters that only reorder bytes (e.g. bitshuffle) rather than compress.
/// IDs are from The HDF Group's filter registry.
fn user_filter_codec(id: i32) -> Option<String> {
    let name = match id {
        305 => "lzo",
        307 => "bzip2",
        32000 => "lzf",
        32001 => "blosc",
        32004 => "lz4",
        32013 => "zfp",
        32015 => "zstd",
        // Bitshuffle reorders bytes (usually paired with a real compressor),
        // so skip it and let the next filter name the compression.
        32008 => return None,
        _ => return Some(format!("filter#{id}")),
    };
    Some(name.to_string())
}

/// Map an HDF5 type descriptor to the short dtype label used elsewhere in the
/// UI (e.g. `F32`, `I64`, `U8`).
fn dtype_name(desc: &TypeDescriptor) -> String {
    use hdf5_metno::types::{FloatSize, IntSize};
    match desc {
        TypeDescriptor::Integer(IntSize::U1) => "I8".to_string(),
        TypeDescriptor::Integer(IntSize::U2) => "I16".to_string(),
        TypeDescriptor::Integer(IntSize::U4) => "I32".to_string(),
        TypeDescriptor::Integer(IntSize::U8) => "I64".to_string(),
        TypeDescriptor::Unsigned(IntSize::U1) => "U8".to_string(),
        TypeDescriptor::Unsigned(IntSize::U2) => "U16".to_string(),
        TypeDescriptor::Unsigned(IntSize::U4) => "U32".to_string(),
        TypeDescriptor::Unsigned(IntSize::U8) => "U64".to_string(),
        TypeDescriptor::Float(FloatSize::U2) => "F16".to_string(),
        TypeDescriptor::Float(FloatSize::U4) => "F32".to_string(),
        TypeDescriptor::Float(FloatSize::U8) => "F64".to_string(),
        TypeDescriptor::Boolean => "BOOL".to_string(),
        other => format!("{other:?}"),
    }
}

/// Decode `%XX` percent-escapes (the inverse of Python's `urllib.parse.quote`,
/// which Cerebras uses to make a tensor name a valid flat HDF5 link).
pub(crate) fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let (Some(hi), Some(lo)) = (hex_val(bytes[i + 1]), hex_val(bytes[i + 2]))
        {
            out.push((hi << 4) | lo);
            i += 3;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_percent_escapes() {
        assert_eq!(
            percent_decode("model.layers.0.weight"),
            "model.layers.0.weight"
        );
        assert_eq!(percent_decode("a%2Fb"), "a/b");
        assert_eq!(percent_decode("x%2Fy%2Fz"), "x/y/z");
        // A stray, malformed escape is left untouched.
        assert_eq!(percent_decode("100%done"), "100%done");
    }

    #[test]
    fn names_only_size_reducing_filters() {
        assert_eq!(
            compression_codec(&Filter::deflate(6)),
            Some("gzip".to_string())
        );
        // Registered third-party filters are named by id.
        assert_eq!(
            compression_codec(&Filter::user(32004, &[])),
            Some("lz4".to_string())
        );
        assert_eq!(
            compression_codec(&Filter::user(32015, &[])),
            Some("zstd".to_string())
        );
        // Shuffle, Fletcher32 and bitshuffle (32008) are not compressors.
        assert_eq!(compression_codec(&Filter::shuffle()), None);
        assert_eq!(compression_codec(&Filter::fletcher32()), None);
        assert_eq!(compression_codec(&Filter::user(32008, &[])), None);
    }

    #[test]
    fn reads_metadata_and_compression_from_a_cerebras_style_file() {
        // Build a fixture mimicking the checkpoint layout: top-level datasets
        // keyed by the (quoted) tensor name, one of them gzip-compressed.
        let dir = std::env::temp_dir().join("safetensors_explorer_hdf5_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("fixture.h5");
        let _ = std::fs::remove_file(&path);

        {
            let file = hdf5_metno::File::create(&path).unwrap();
            file.new_dataset::<f32>()
                .shape([2, 3])
                .create("model.layers.0.weight")
                .unwrap();
            file.new_dataset::<i64>()
                .shape([4])
                .create("model.embed")
                .unwrap();
            // A name containing '/', stored URL-quoted as Cerebras does.
            file.new_dataset::<f32>()
                .shape([5])
                .create("a%2Fb")
                .unwrap();
            // A chunked + gzip-compressed dataset of highly compressible zeros.
            let ds = file
                .new_dataset::<f32>()
                .shape([64, 64])
                .chunk([16, 16])
                .deflate(6)
                .create("model.compressed")
                .unwrap();
            ds.write_raw(&vec![0f32; 64 * 64]).unwrap();
        }

        let tensors = read_tensors(&path).unwrap();
        let by_name = |n: &str| tensors.iter().find(|t| t.name == n).unwrap();

        let weight = by_name("model.layers.0.weight");
        assert_eq!(weight.dtype, "F32");
        assert_eq!(weight.shape, vec![2, 3]);
        assert_eq!(weight.size_bytes, 6 * 4);
        assert!(matches!(weight.storage, Storage::Raw));

        let embed = by_name("model.embed");
        assert_eq!(embed.dtype, "I64");
        assert_eq!(embed.size_bytes, 4 * 8);

        // The quoted '/' name round-trips back to a slash.
        assert!(tensors.iter().any(|t| t.name == "a/b"));

        // The compressed dataset reports gzip and a smaller on-disk size.
        let comp = by_name("model.compressed");
        assert_eq!(comp.size_bytes, 64 * 64 * 4);
        match &comp.storage {
            Storage::Compressed {
                codec,
                stored_bytes,
            } => {
                assert_eq!(codec, "gzip");
                assert!(*stored_bytes < comp.size_bytes);
            }
            other => panic!("expected compressed storage, got {other:?}"),
        }
        // The compressed dataset is chunked, so its layout is reported.
        assert!(matches!(comp.layout, crate::tree::Layout::Chunked { .. }));

        let _ = std::fs::remove_file(&path);
    }
}
