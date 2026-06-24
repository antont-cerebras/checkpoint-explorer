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

use crate::tree::{Layout, MetadataInfo, Storage, TensorInfo};

/// Read tensor metadata (name, dtype, shape, logical + on-disk size) from an
/// HDF5 checkpoint.
pub fn read_tensors(path: &std::path::Path) -> Result<Vec<TensorInfo>> {
    let file = hdf5_metno::File::open(path)
        .with_context(|| format!("Failed to open HDF5 file: {}", path.display()))?;

    // libhdf5 is now initialised; teach it the LZ4 filter so compressed datasets
    // (stats/preview) are readable later in the session.
    crate::hdf5_lz4::register();
    crate::hdf5_zstd::register();

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

/// Read root-level HDF5 attributes as checkpoint metadata. Cerebras checkpoints
/// store free-form metadata in root attributes — scalars like `__version__` /
/// `__SUCCESS__`, the layout `__spec__`, and one `<object>.__metadata__` JSON
/// attribute per tensor and per config object (e.g. `inference_version`,
/// `codebook_packing_schema`). Each `__metadata__` attribute wraps the real
/// payload (torch-serialization plumbing: `__spec__` / `__objects__` / the
/// payload under the attribute's own name), so we unwrap it to the useful part.
pub fn read_metadata(path: &std::path::Path) -> Result<Vec<MetadataInfo>> {
    let file = hdf5_metno::File::open(path)
        .with_context(|| format!("Failed to open HDF5 file: {}", path.display()))?;
    let names = file.attr_names().unwrap_or_default();
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        let Some((raw, value_type)) = read_attr_value(&file, &name) else {
            continue;
        };
        // Unwrap the `<object>.__metadata__` serialization wrapper to the payload
        // it carries; everything else (scalars, `__spec__`) is shown verbatim.
        let (value, value_type) = unwrap_metadata_value(&name, &raw).unwrap_or((raw, value_type));
        out.push(MetadataInfo {
            name,
            value,
            value_type,
        });
    }
    Ok(out)
}

/// Read a single attribute as a `(value, type-label)` string pair, trying the
/// scalar types Cerebras checkpoints use (a variable-length string, else a
/// boolean / float / integer). `None` for an attribute we can't read as any.
fn read_attr_value(loc: &hdf5_metno::Location, name: &str) -> Option<(String, String)> {
    use hdf5_metno::types::{VarLenAscii, VarLenUnicode};
    let attr = loc.attr(name).ok()?;
    if let Ok(s) = attr.read_scalar::<VarLenUnicode>() {
        return Some((s.as_str().to_string(), "string".to_string()));
    }
    if let Ok(s) = attr.read_scalar::<VarLenAscii>() {
        return Some((s.as_str().to_string(), "string".to_string()));
    }
    if let Ok(b) = attr.read_scalar::<bool>() {
        return Some((b.to_string(), "bool".to_string()));
    }
    if let Ok(v) = attr.read_scalar::<f64>() {
        return Some((v.to_string(), "float".to_string()));
    }
    if let Ok(v) = attr.read_scalar::<i64>() {
        return Some((v.to_string(), "int".to_string()));
    }
    None
}

/// Unwrap a `<object>.__metadata__` attribute's JSON to the payload it carries.
/// The payload lives under a key equal to the attribute name; a `string_value`
/// (the `StringSerializer` case, e.g. `inference_version` → `"1.5"`, or a config
/// object stored as a JSON string) is returned as that string (pretty-printed if
/// it is itself JSON), otherwise the payload object is pretty-printed. Returns
/// `None` for non-`__metadata__` attributes or anything that doesn't parse, so
/// the caller falls back to the raw value.
fn unwrap_metadata_value(attr_name: &str, raw: &str) -> Option<(String, String)> {
    if !attr_name.ends_with(".__metadata__") {
        return None;
    }
    let json: serde_json::Value = serde_json::from_str(raw).ok()?;
    let payload = json.get(attr_name)?;
    if let Some(sv) = payload.get("string_value").and_then(|v| v.as_str()) {
        let value = serde_json::from_str::<serde_json::Value>(sv)
            .ok()
            .and_then(|inner| serde_json::to_string_pretty(&inner).ok())
            .unwrap_or_else(|| sv.to_string());
        return Some((value, "string".to_string()));
    }
    let value = serde_json::to_string_pretty(payload).unwrap_or_else(|_| raw.to_string());
    Some((value, "json".to_string()))
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

    #[test]
    fn unwraps_string_serializer_metadata() {
        // `inference_version`: a plain string payload under `string_value`.
        let raw = r#"{"__spec__":[{"inference_version":"*"}],"__objects__":["inference_version"],
            "inference_version.__metadata__":{"string_value":"1.5","__TYPE__":"StringSerializer"}}"#;
        let (val, ty) = unwrap_metadata_value("inference_version.__metadata__", raw).unwrap();
        assert_eq!(val, "1.5");
        assert_eq!(ty, "string");
    }

    #[test]
    fn unwraps_json_string_value_pretty() {
        // `codebook_packing_schema`: `string_value` is itself JSON → pretty-print.
        let raw = r#"{"__objects__":["codebook_packing_schema"],
            "codebook_packing_schema.__metadata__":{"string_value":"{\"down_proj\": {\"quant_mode\": \"3bit\"}}","__TYPE__":"StringSerializer"}}"#;
        let (val, ty) = unwrap_metadata_value("codebook_packing_schema.__metadata__", raw).unwrap();
        assert_eq!(ty, "string");
        assert!(val.contains("\"down_proj\""));
        assert!(val.contains("\"quant_mode\": \"3bit\""));
        assert!(val.contains('\n'), "should be pretty-printed: {val}");
    }

    #[test]
    fn unwraps_non_string_payload_as_pretty_json() {
        // A torch-tensor payload has no `string_value`; show the dict itself.
        let raw =
            r#"{"lm_head.weight.__metadata__":{"__TORCH__":true,"dtypes":["torch.float16"]}}"#;
        let (val, ty) = unwrap_metadata_value("lm_head.weight.__metadata__", raw).unwrap();
        assert_eq!(ty, "json");
        assert!(val.contains("__TORCH__"));
        assert!(val.contains("torch.float16"));
    }

    #[test]
    fn leaves_non_metadata_and_malformed_attrs_to_the_caller() {
        // Non-`__metadata__` names are shown verbatim by the caller.
        assert!(unwrap_metadata_value("__spec__", "[1,2,3]").is_none());
        assert!(unwrap_metadata_value("__version__", "0.5").is_none());
        // Unparseable JSON or a missing payload key falls back to the raw value.
        assert!(unwrap_metadata_value("x.__metadata__", "not json").is_none());
        assert!(unwrap_metadata_value("x.__metadata__", r#"{"other":1}"#).is_none());
    }

    #[test]
    fn reads_root_attributes_as_metadata() {
        use hdf5_metno::types::VarLenUnicode;
        use std::str::FromStr;

        let dir = std::env::temp_dir().join("safetensors_explorer_hdf5_attrs");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("attrs.h5");
        let _ = std::fs::remove_file(&path);
        {
            let file = hdf5_metno::File::create(&path).unwrap();
            file.new_dataset::<f32>().shape([2]).create("w").unwrap();
            // A Cerebras-style string `__metadata__` attribute.
            let json = r#"{"__objects__":["inference_version"],"inference_version.__metadata__":{"string_value":"1.5","__TYPE__":"StringSerializer"}}"#;
            let v = VarLenUnicode::from_str(json).unwrap();
            file.new_attr::<VarLenUnicode>()
                .create("inference_version.__metadata__")
                .unwrap()
                .write_scalar(&v)
                .unwrap();
            // Scalar attributes (a bool and a float).
            file.new_attr::<bool>()
                .create("__SUCCESS__")
                .unwrap()
                .write_scalar(&true)
                .unwrap();
            file.new_attr::<f64>()
                .create("__version__")
                .unwrap()
                .write_scalar(&0.5f64)
                .unwrap();
        }

        let meta = read_metadata(&path).unwrap();
        let find = |n: &str| meta.iter().find(|m| m.name == n).unwrap();
        let iv = find("inference_version.__metadata__");
        assert_eq!(iv.value, "1.5");
        assert_eq!(iv.value_type, "string");
        assert_eq!(find("__SUCCESS__").value, "true");
        assert_eq!(find("__SUCCESS__").value_type, "bool");
        assert_eq!(find("__version__").value_type, "float");

        let _ = std::fs::remove_file(&path);
    }
}
