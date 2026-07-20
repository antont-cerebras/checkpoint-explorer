//! Pure parsing of a safetensors **header** (the JSON blob after the 8-byte
//! length) into [`TensorInfo`]/[`MetadataInfo`]. Header-only — never touches the
//! tensor data. Shared by the local file reader, the remote SFTP reader
//! (`crate::sftp`), and the `--ssh-read` path, so the one parse lives in the
//! core crate with no dependency on the TUI/`Explorer`.

use anyhow::{Context, Result};

use crate::tree::{Layout, MetadataInfo, Storage, TensorInfo};

/// Validate a safetensors header length against a sane ceiling (guards a
/// corrupt / non-safetensors file claiming a huge header).
pub fn header_len(raw: u64, source: &str) -> Result<usize> {
    const MAX_HEADER_SIZE: u64 = 100_000_000;
    if raw > MAX_HEADER_SIZE {
        anyhow::bail!("SafeTensors header too large ({raw} bytes): {source}");
    }
    Ok(raw as usize)
}

/// Parse a safetensors header (the JSON blob after the 8-byte length) into
/// tensors + metadata. `source` is the tensors' `source_path` (a local path or a
/// remote marker). Every non-`__metadata__` entry describes a tensor.
pub fn parse_header(
    header_buf: &[u8],
    source: &str,
) -> Result<(Vec<TensorInfo>, Vec<MetadataInfo>)> {
    let mut tensors: Vec<TensorInfo> = Vec::new();
    let mut metadata: Vec<MetadataInfo> = Vec::new();
    let source_path = source.to_string();

    let header: serde_json::Value = serde_json::from_slice(header_buf)
        .with_context(|| format!("Failed to parse SafeTensors header: {source}"))?;

    let obj = header
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("Invalid SafeTensors header: {source}"))?;

    for (key, value) in obj {
        // The `__metadata__` entry holds free-form string key/value pairs.
        if key == "__metadata__" {
            if let Some(meta_obj) = value.as_object() {
                for (meta_key, meta_value) in meta_obj {
                    metadata.push(MetadataInfo {
                        name: meta_key.clone(),
                        value: match meta_value.as_str() {
                            Some(s) => s.to_string(),
                            None => meta_value.to_string(),
                        },
                        value_type: "string".to_string(),
                    });
                }
            }
            continue;
        }

        // Every other entry describes a tensor.
        let dtype = value
            .get("dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("?")
            .to_string();
        let shape: Vec<usize> = value
            .get("shape")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_u64().map(|n| n as usize))
                    .collect()
            })
            .unwrap_or_default();
        let data_offsets = value
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .filter(|offsets| offsets.len() == 2)
            .and_then(|offsets| Some((offsets[0].as_u64()?, offsets[1].as_u64()?)));
        let size_bytes = data_offsets
            .map(|(start, end)| end.saturating_sub(start) as usize)
            .unwrap_or(0);
        let layout = match data_offsets {
            Some((start, end)) => Layout::ByteRange { start, end },
            None => Layout::None,
        };
        let num_elements = shape.iter().product::<usize>();

        tensors.push(TensorInfo {
            name: key.clone(),
            dtype,
            shape,
            size_bytes,
            num_elements,
            storage: Storage::Unknown,
            source_path: source_path.clone(),
            layout,
        });
    }

    Ok((tensors, metadata))
}
