//! The byte-level layout of a single `.safetensors` file — the data behind the
//! file browser's layout map (a "separate mode" that shows how a checkpoint's
//! tensors are physically stored). Parses only the header, so it's cheap for any
//! shard regardless of size.
//!
//! safetensors on disk is: an 8-byte little-endian `u64` header length `N`, then
//! `N` bytes of JSON (`name → {dtype, shape, data_offsets:[begin,end]}`, plus an
//! optional `__metadata__` entry), then the tensor data — tensor `i` occupying
//! `[8 + N + begin_i, 8 + N + end_i)`. See <https://github.com/huggingface/safetensors>.

use std::io::Read;
use std::path::Path;

/// One contiguous region of the file: the header, a tensor, or a gap (padding /
/// an unaccounted span between tensors). Offsets are absolute within the file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    pub name: String,
    /// `None` for the header / gap rows; a tensor's dtype otherwise.
    pub dtype: Option<String>,
    pub shape: Vec<usize>,
    pub start: u64,
    pub end: u64,
    pub kind: SegmentKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentKind {
    /// The 8-byte length prefix plus the JSON metadata header.
    Header,
    /// A tensor's data.
    Tensor,
    /// An unaccounted gap between segments (rare — alignment padding).
    Gap,
}

impl Segment {
    pub fn len(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }
}

/// The parsed layout of a safetensors file: its total size, the header size, and
/// every segment in file order (header first, then tensors by offset, with any
/// gaps filled in).
#[derive(Debug, Clone)]
pub struct LayoutMap {
    /// The file's leaf name, for the title.
    pub name: String,
    pub total_len: u64,
    /// Size of the header region (`8 + N`).
    pub header_len: u64,
    pub tensor_count: usize,
    /// The `__metadata__` entries (key → value), sorted by key — shown as a
    /// tree-like list under the header band. Empty when the file has none.
    pub metadata: Vec<(String, String)>,
    pub segments: Vec<Segment>,
}

impl LayoutMap {
    /// Number of `__metadata__` entries (for the header summary line).
    pub fn metadata_entries(&self) -> usize {
        self.metadata.len()
    }
}

/// Cap on the JSON header we'll read — safetensors headers are kilobytes to a few
/// megabytes; a wildly larger length prefix means a corrupt / non-safetensors
/// file, so we refuse rather than allocate it.
const MAX_HEADER: u64 = 100 << 20; // 100 MiB

/// Parse the layout of the safetensors file at `path` (header only — the tensor
/// data is never read). Returns a readable error string on any malformation.
pub fn parse(path: &Path) -> Result<LayoutMap, String> {
    let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let total_len = file.metadata().map_err(|e| e.to_string())?.len();

    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)
        .map_err(|_| "not a safetensors file (too short for a header length)".to_string())?;
    let n = u64::from_le_bytes(len_buf);
    if n == 0 {
        return Err("empty safetensors header".to_string());
    }
    if n > MAX_HEADER || 8 + n > total_len {
        return Err("header length is out of range (not a safetensors file?)".to_string());
    }

    let mut json = vec![0u8; n as usize];
    file.read_exact(&mut json)
        .map_err(|_| "truncated safetensors header".to_string())?;
    let name = path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string_lossy().into_owned());
    parse_from(&name, total_len, &json)
}

/// Parse a safetensors layout from its already-read header — shared by the local
/// [`parse`] and the remote SFTP read. `header_json` is the `N` bytes after the
/// 8-byte length prefix; `total_len` is the whole file's size (for the trailing
/// data gap); `name` is the display label.
pub fn parse_from(name: &str, total_len: u64, header_json: &[u8]) -> Result<LayoutMap, String> {
    let n = header_json.len() as u64;
    if n == 0 {
        return Err("empty safetensors header".to_string());
    }
    let header_len = 8 + n;
    if header_len > total_len {
        return Err("header length is out of range (not a safetensors file?)".to_string());
    }
    let value: serde_json::Value =
        serde_json::from_slice(header_json).map_err(|e| format!("invalid header JSON: {e}"))?;
    let obj = value
        .as_object()
        .ok_or_else(|| "header is not a JSON object".to_string())?;

    let data_start = header_len;

    // Collect tensor entries (everything but `__metadata__`), resolving offsets to
    // absolute file positions.
    let mut tensors: Vec<Segment> = Vec::new();
    let mut metadata: Vec<(String, String)> = Vec::new();
    for (key, entry) in obj {
        if key == "__metadata__" {
            if let Some(m) = entry.as_object() {
                metadata = m
                    .iter()
                    .map(|(k, v)| {
                        // safetensors metadata is string→string; keep a string
                        // for anything else (its compact JSON form).
                        let val = v
                            .as_str()
                            .map(str::to_string)
                            .unwrap_or_else(|| v.to_string());
                        (k.clone(), val)
                    })
                    .collect();
                metadata.sort_by(|a, b| a.0.cmp(&b.0));
            }
            continue;
        }
        let e = match entry.as_object() {
            Some(e) => e,
            None => continue,
        };
        let dtype = e
            .get("dtype")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("?")
            .to_string();
        let shape = e
            .get("shape")
            .and_then(serde_json::Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64().map(|u| u as usize))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let offsets = e.get("data_offsets").and_then(serde_json::Value::as_array);
        let (begin, end) = match offsets {
            Some(a) if a.len() == 2 => (a[0].as_u64().unwrap_or(0), a[1].as_u64().unwrap_or(0)),
            _ => continue,
        };
        tensors.push(Segment {
            name: key.clone(),
            dtype: Some(dtype),
            shape,
            start: data_start + begin,
            end: data_start + end,
            kind: SegmentKind::Tensor,
        });
    }
    tensors.sort_by_key(|s| (s.start, s.end));
    let tensor_count = tensors.len();

    // Build the ordered segment list: header, then tensors, inserting a `Gap`
    // wherever there's an unaccounted span (alignment padding, or a hole).
    let mut segments: Vec<Segment> = Vec::with_capacity(tensor_count + 2);
    segments.push(Segment {
        name: "header (8 B length + JSON metadata)".to_string(),
        dtype: None,
        shape: Vec::new(),
        start: 0,
        end: header_len,
        kind: SegmentKind::Header,
    });
    let mut cursor = header_len;
    for t in tensors {
        if t.start > cursor {
            segments.push(gap(cursor, t.start));
        }
        cursor = t.end.max(cursor);
        segments.push(t);
    }
    if total_len > cursor {
        segments.push(gap(cursor, total_len));
    }

    Ok(LayoutMap {
        name: name.to_string(),
        total_len,
        header_len,
        tensor_count,
        metadata,
        segments,
    })
}

/// Read the raw JSON header of the safetensors file at `path` (the `N` bytes
/// after the 8-byte length), capped at `cap` bytes. Returns `(json, truncated)`.
/// For previewing the header's contents (the `Enter`-on-header action).
pub fn read_header_json(path: &Path, cap: u64) -> Result<(String, bool), String> {
    let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)
        .map_err(|_| "not a safetensors file".to_string())?;
    let n = u64::from_le_bytes(len_buf);
    if n == 0 || n > MAX_HEADER {
        return Err("header length is out of range".to_string());
    }
    let take = n.min(cap);
    let mut buf = vec![0u8; take as usize];
    file.read_exact(&mut buf)
        .map_err(|_| "truncated safetensors header".to_string())?;
    let json = String::from_utf8(buf).map_err(|_| "header is not valid UTF-8".to_string())?;
    Ok((json, n > cap))
}

/// Read the *full* JSON header of the safetensors file at `path`: the header
/// length `N` and the exact `N` bytes of JSON that follow the 8-byte prefix.
/// Unlike [`read_header_json`] this never truncates — the in-place rename
/// (`convert --map`) has to rewrite the whole header — so a header beyond
/// [`MAX_HEADER`] is refused rather than partially read. Returns `(N, json_bytes)`.
pub fn read_header_full(path: &Path) -> Result<(u64, Vec<u8>), String> {
    let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)
        .map_err(|_| "not a safetensors file (too short for a header length)".to_string())?;
    let n = u64::from_le_bytes(len_buf);
    if n == 0 {
        return Err("empty safetensors header".to_string());
    }
    if n > MAX_HEADER {
        return Err("header length is out of range (not a safetensors file?)".to_string());
    }
    let mut json = vec![0u8; n as usize];
    file.read_exact(&mut json)
        .map_err(|_| "truncated safetensors header".to_string())?;
    Ok((n, json))
}

fn gap(start: u64, end: u64) -> Segment {
    Segment {
        name: "(padding)".to_string(),
        dtype: None,
        shape: Vec::new(),
        start,
        end,
        kind: SegmentKind::Gap,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a minimal safetensors file: two f32 tensors laid out back-to-back.
    fn write_fixture(path: &Path) {
        // w1: [2,2] f32 → 16 bytes at [0,16); w2: [2] f32 → 8 bytes at [16,24).
        let header = r#"{"w1":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"w2":{"dtype":"F32","shape":[2],"data_offsets":[16,24]},"__metadata__":{"format":"pt"}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0u8; 24]); // tensor data
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn parses_header_and_tensor_segments_in_offset_order() {
        let path = std::env::temp_dir().join("ce_safelayout_test.safetensors");
        write_fixture(&path);
        let map = parse(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(map.tensor_count, 2);
        assert_eq!(map.metadata_entries(), 1);
        assert_eq!(map.metadata, vec![("format".to_string(), "pt".to_string())]);
        // Header first, then the two tensors in offset order.
        assert_eq!(map.segments[0].kind, SegmentKind::Header);
        assert_eq!(map.segments[0].start, 0);
        let hlen = map.header_len;
        assert_eq!(map.segments[1].name, "w1");
        assert_eq!(map.segments[1].start, hlen);
        assert_eq!(map.segments[1].end, hlen + 16);
        assert_eq!(map.segments[1].dtype.as_deref(), Some("F32"));
        assert_eq!(map.segments[1].shape, vec![2, 2]);
        assert_eq!(map.segments[2].name, "w2");
        assert_eq!(map.segments[2].start, hlen + 16);
        assert_eq!(map.segments[2].end, hlen + 24);
        // Data is contiguous, so no trailing gap.
        assert!(map.segments.iter().all(|s| s.kind != SegmentKind::Gap));
        assert_eq!(map.total_len, hlen + 24);
    }

    #[test]
    fn parse_from_matches_parse_and_finds_trailing_gap() {
        // The same header bytes the fixture writes, fed straight to the pure core.
        let header = r#"{"w1":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"w2":{"dtype":"F32","shape":[2],"data_offsets":[16,24]},"__metadata__":{"format":"pt"}}"#;
        let header_len = 8 + header.len() as u64;
        let total_len = header_len + 24;

        // Matches `parse` on the equivalent temp file, segment for segment.
        let path = std::env::temp_dir().join("ce_safelayout_parsefrom.safetensors");
        write_fixture(&path);
        let via_file = parse(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let via_core = parse_from(
            "ce_safelayout_parsefrom.safetensors",
            total_len,
            header.as_bytes(),
        )
        .unwrap();
        assert_eq!(via_core.name, via_file.name);
        assert_eq!(via_core.header_len, via_file.header_len);
        assert_eq!(via_core.tensor_count, via_file.tensor_count);
        assert_eq!(via_core.metadata, via_file.metadata);
        assert_eq!(via_core.segments.len(), via_file.segments.len());

        // A larger `total_len` than the tensors occupy yields a trailing Gap.
        let padded = parse_from("x.safetensors", total_len + 512, header.as_bytes()).unwrap();
        let last = padded.segments.last().unwrap();
        assert_eq!(last.kind, SegmentKind::Gap);
        assert_eq!(last.end, total_len + 512);

        // Guard: a header claiming more than the file holds is rejected.
        assert!(parse_from("x", 4, header.as_bytes()).is_err());
    }

    #[test]
    fn rejects_non_safetensors() {
        let path = std::env::temp_dir().join("ce_safelayout_bad.bin");
        std::fs::write(&path, b"not a safetensors file at all").unwrap();
        let got = parse(&path);
        let _ = std::fs::remove_file(&path);
        assert!(got.is_err(), "garbage should not parse: {got:?}");
    }

    #[test]
    fn reads_the_raw_header_json() {
        let path = std::env::temp_dir().join("ce_safelayout_hdr.safetensors");
        write_fixture(&path);
        let (json, truncated) = read_header_json(&path, 1 << 20).unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(!truncated);
        assert!(json.contains("\"w1\""), "raw header JSON: {json}");
        assert!(json.contains("data_offsets"), "raw header JSON: {json}");
    }
}
