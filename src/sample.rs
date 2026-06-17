//! On-demand sampling of tensor data for the heatmap and numeric views.
//!
//! Tensors can be many GB, so we never read a whole one: we pick a small grid
//! of element indices that fit the screen (including the edges) and read just
//! those. safetensors are read by seeking to the sampled rows; HDF5 datasets
//! are read via libhdf5 (which converts any numeric dtype to `f64` and handles
//! decompression) with a size cap.

use std::io::{Read, Seek, SeekFrom};

use crate::tree::{Layout, TensorInfo};

/// A downsampled grid of tensor values plus the original indices it came from.
pub struct Sample {
    /// Original row indices that were sampled (logical rows).
    pub rows: Vec<usize>,
    /// Original column indices that were sampled.
    pub cols: Vec<usize>,
    /// Sampled values, `values[i][j]` for `(rows[i], cols[j])`.
    pub values: Vec<Vec<f64>>,
    pub min: f64,
    pub max: f64,
    /// Logical dimensions of the sampled matrix (1D is treated as `1 x n`; for
    /// 3D this is the slice's `d1 x d2`).
    pub total_rows: usize,
    pub total_cols: usize,
    /// Number of leading-index slices (`d0` for 3D, else 1).
    pub slices: usize,
    /// The slice index this sample is from (0 for 1D/2D).
    pub slice: usize,
}

/// Sample a 1D/2D/3D tensor into at most `max_rows` x `max_cols` values. For a
/// 3D tensor `[d0, d1, d2]`, `slice` selects the leading index and the `d1 x d2`
/// matrix at that index is sampled (clamped to a valid slice).
pub fn sample_tensor(
    t: &TensorInfo,
    max_rows: usize,
    max_cols: usize,
    slice: usize,
) -> Result<Sample, String> {
    let (total_rows, total_cols, slices) = match t.shape.as_slice() {
        [n] => (1usize, *n, 1usize),
        [r, c] => (*r, *c, 1usize),
        [d0, d1, d2] => (*d1, *d2, *d0),
        _ => {
            return Err(format!(
                "data preview supports 1D, 2D and 3D tensors only (this one is {}D)",
                t.shape.len()
            ));
        }
    };
    if total_rows == 0 || total_cols == 0 || slices == 0 {
        return Err("tensor has no elements".to_string());
    }
    let slice = slice.min(slices - 1);
    // Elements to skip to reach the chosen slice (0 for 1D/2D).
    let base = slice * total_rows * total_cols;

    let rows = sample_indices(total_rows, max_rows.max(1));
    let cols = sample_indices(total_cols, max_cols.max(1));

    let ext = std::path::Path::new(&t.source_path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    let values = match ext {
        "safetensors" => read_safetensors(t, total_cols, base, &rows, &cols)?,
        "h5" | "hdf5" => read_hdf5(t, total_cols, base, &rows, &cols)?,
        _ => return Err("data preview is not supported for this format".to_string()),
    };

    let (mut min, mut max) = (f64::INFINITY, f64::NEG_INFINITY);
    for row in &values {
        for &v in row {
            if v.is_finite() {
                min = min.min(v);
                max = max.max(v);
            }
        }
    }
    if !min.is_finite() {
        min = 0.0;
        max = 0.0;
    }

    Ok(Sample {
        rows,
        cols,
        values,
        min,
        max,
        total_rows,
        total_cols,
        slices,
        slice,
    })
}

/// Up to `k` evenly-spaced indices in `0..n`, always including `0` and `n-1`.
fn sample_indices(n: usize, k: usize) -> Vec<usize> {
    if n <= k {
        return (0..n).collect();
    }
    if k == 1 {
        return vec![0];
    }
    let mut idx: Vec<usize> = (0..k)
        .map(|i| (i * (n - 1) + (k - 1) / 2) / (k - 1))
        .collect();
    idx.dedup();
    idx
}

fn item_size(dtype: &str) -> Option<usize> {
    Some(match dtype {
        "F64" | "I64" | "U64" => 8,
        "F32" | "I32" | "U32" => 4,
        "F16" | "BF16" | "I16" | "U16" => 2,
        "I8" | "U8" | "BOOL" => 1,
        _ => return None,
    })
}

/// Decode `item_size(dtype)` little-endian bytes into an `f64`.
fn decode(dtype: &str, b: &[u8]) -> f64 {
    match dtype {
        "F64" => f64::from_le_bytes(b.try_into().unwrap()),
        "F32" => f32::from_le_bytes(b.try_into().unwrap()) as f64,
        "F16" => f16_to_f64(u16::from_le_bytes(b.try_into().unwrap())),
        "BF16" => bf16_to_f64(u16::from_le_bytes(b.try_into().unwrap())),
        "I64" => i64::from_le_bytes(b.try_into().unwrap()) as f64,
        "I32" => i32::from_le_bytes(b.try_into().unwrap()) as f64,
        "I16" => i16::from_le_bytes(b.try_into().unwrap()) as f64,
        "I8" => (b[0] as i8) as f64,
        "U64" => u64::from_le_bytes(b.try_into().unwrap()) as f64,
        "U32" => u32::from_le_bytes(b.try_into().unwrap()) as f64,
        "U16" => u16::from_le_bytes(b.try_into().unwrap()) as f64,
        "U8" | "BOOL" => b[0] as f64,
        _ => f64::NAN,
    }
}

/// bf16 is just the high 16 bits of an f32.
fn bf16_to_f64(bits: u16) -> f64 {
    f32::from_bits((bits as u32) << 16) as f64
}

/// IEEE-754 half precision -> f64.
fn f16_to_f64(bits: u16) -> f64 {
    let sign = (bits >> 15) & 1;
    let exp = (bits >> 10) & 0x1f;
    let frac = bits & 0x3ff;
    let val: f32 = if exp == 0 {
        (frac as f32) * 2f32.powi(-24) // subnormal
    } else if exp == 0x1f {
        if frac == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        (1.0 + (frac as f32) / 1024.0) * 2f32.powi(exp as i32 - 15)
    };
    let signed = if sign == 1 { -val } else { val };
    signed as f64
}

/// Read sampled values from a safetensors file by seeking to each sampled row.
fn read_safetensors(
    t: &TensorInfo,
    total_cols: usize,
    base: usize,
    rows: &[usize],
    cols: &[usize],
) -> Result<Vec<Vec<f64>>, String> {
    let Layout::ByteRange { start, .. } = t.layout else {
        return Err("tensor data location is unknown".to_string());
    };
    let item = item_size(&t.dtype).ok_or_else(|| format!("unsupported dtype: {}", t.dtype))?;

    let mut file = std::fs::File::open(&t.source_path).map_err(|e| e.to_string())?;
    // The data blob begins after the 8-byte header length and the JSON header.
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf).map_err(|e| e.to_string())?;
    let data_start = 8 + u64::from_le_bytes(len_buf) + start;

    let first = *cols.first().unwrap();
    let last = *cols.last().unwrap();
    let span = last - first + 1;
    let span_bytes = span * item;

    let mut out = Vec::with_capacity(rows.len());
    // Read each sampled row's column span in one go when it's reasonably sized;
    // otherwise fall back to one read per sampled element.
    const MAX_SPAN: usize = 64 * 1024 * 1024;
    if span_bytes <= MAX_SPAN {
        let mut buf = vec![0u8; span_bytes];
        for &r in rows {
            let off = data_start
                + (base as u64 + (r as u64) * (total_cols as u64) + (first as u64)) * (item as u64);
            file.seek(SeekFrom::Start(off)).map_err(|e| e.to_string())?;
            file.read_exact(&mut buf).map_err(|e| e.to_string())?;
            let row = cols
                .iter()
                .map(|&c| {
                    decode(
                        &t.dtype,
                        &buf[(c - first) * item..(c - first) * item + item],
                    )
                })
                .collect();
            out.push(row);
        }
    } else {
        let mut buf = vec![0u8; item];
        for &r in rows {
            let mut row = Vec::with_capacity(cols.len());
            for &c in cols {
                let off = data_start
                    + (base as u64 + (r as u64) * (total_cols as u64) + (c as u64)) * (item as u64);
                file.seek(SeekFrom::Start(off)).map_err(|e| e.to_string())?;
                file.read_exact(&mut buf).map_err(|e| e.to_string())?;
                row.push(decode(&t.dtype, &buf));
            }
            out.push(row);
        }
    }
    Ok(out)
}

#[cfg(feature = "hdf5")]
fn read_hdf5(
    t: &TensorInfo,
    total_cols: usize,
    base: usize,
    rows: &[usize],
    cols: &[usize],
) -> Result<Vec<Vec<f64>>, String> {
    // Reading is whole-dataset (libhdf5 decompresses everything), so cap it.
    const MAX_ELEMS: usize = 8_000_000;
    if t.num_elements > MAX_ELEMS {
        return Err(format!(
            "tensor too large to preview ({} elements); sampling large HDF5 datasets is a planned follow-up",
            t.num_elements
        ));
    }

    let file = hdf5_metno::File::open(&t.source_path).map_err(|e| e.to_string())?;
    // The decoded tensor name maps to a URL-quoted dataset key.
    let key = file
        .member_names()
        .map_err(|e| e.to_string())?
        .into_iter()
        .find(|k| crate::hdf5::percent_decode(k) == t.name)
        .ok_or_else(|| "dataset not found in file".to_string())?;
    let dataset = file.dataset(&key).map_err(|e| e.to_string())?;
    // libhdf5 converts any numeric dtype to f64 on read.
    let flat = dataset.read_raw::<f64>().map_err(|e| e.to_string())?;

    let out = rows
        .iter()
        .map(|&r| {
            cols.iter()
                .map(|&c| {
                    flat.get(base + r * total_cols + c)
                        .copied()
                        .unwrap_or(f64::NAN)
                })
                .collect()
        })
        .collect();
    Ok(out)
}

#[cfg(not(feature = "hdf5"))]
fn read_hdf5(
    _t: &TensorInfo,
    _total_cols: usize,
    _base: usize,
    _rows: &[usize],
    _cols: &[usize],
) -> Result<Vec<Vec<f64>>, String> {
    Err("HDF5 support is not compiled in (rebuild with `--features hdf5`)".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_indices_includes_edges_and_is_bounded() {
        assert_eq!(sample_indices(3, 10), vec![0, 1, 2]); // n <= k: all
        let idx = sample_indices(100, 5);
        assert_eq!(idx.first(), Some(&0));
        assert_eq!(idx.last(), Some(&99));
        assert!(idx.len() <= 5);
        assert!(idx.windows(2).all(|w| w[0] < w[1])); // strictly increasing
    }

    #[test]
    fn decodes_float_dtypes() {
        assert_eq!(decode("F32", &1.5f32.to_le_bytes()), 1.5);
        assert_eq!(decode("F64", &(-2.25f64).to_le_bytes()), -2.25);
        // bf16 of 1.0 is 0x3F80; f16 of 1.0 is 0x3C00.
        assert_eq!(decode("BF16", &0x3F80u16.to_le_bytes()), 1.0);
        assert_eq!(decode("F16", &0x3C00u16.to_le_bytes()), 1.0);
        assert_eq!(decode("I16", &(-5i16).to_le_bytes()), -5.0);
    }

    fn fixture(
        path: &std::path::Path,
        name: &str,
        shape: &[usize],
        offsets: (u64, u64),
    ) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            dtype: "F32".to_string(),
            shape: shape.to_vec(),
            size_bytes: (offsets.1 - offsets.0) as usize,
            num_elements: shape.iter().product(),
            storage: crate::tree::Storage::Unknown,
            source_path: path.to_string_lossy().into_owned(),
            layout: Layout::ByteRange {
                start: offsets.0,
                end: offsets.1,
            },
        }
    }

    #[test]
    fn samples_a_safetensors_tensor_by_value() {
        use std::io::Write;
        let dir = std::env::temp_dir().join("checkpoint_explorer_sample_st");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("w.safetensors");
        // 4x5 f32, value[r][c] = r*5 + c
        let header = br#"{"w":{"dtype":"F32","shape":[4,5],"data_offsets":[0,80]}}"#;
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(header).unwrap();
        for i in 0..20u32 {
            f.write_all(&(i as f32).to_le_bytes()).unwrap();
        }
        drop(f);

        let t = fixture(&path, "w", &[4, 5], (0, 80));
        let s = sample_tensor(&t, 10, 10, 0).unwrap();
        assert_eq!((s.total_rows, s.total_cols), (4, 5));
        assert_eq!((s.slices, s.slice), (1, 0));
        assert_eq!(s.min, 0.0);
        assert_eq!(s.max, 19.0);
        for (i, &r) in s.rows.iter().enumerate() {
            for (j, &c) in s.cols.iter().enumerate() {
                assert_eq!(s.values[i][j], (r * 5 + c) as f64);
            }
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn samples_a_3d_safetensors_slice() {
        use std::io::Write;
        let dir = std::env::temp_dir().join("checkpoint_explorer_sample_3d");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("w.safetensors");
        // [2, 3, 4] f32, value[s][r][c] = s*12 + r*4 + c
        let header = br#"{"w":{"dtype":"F32","shape":[2,3,4],"data_offsets":[0,96]}}"#;
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(header).unwrap();
        for i in 0..24u32 {
            f.write_all(&(i as f32).to_le_bytes()).unwrap();
        }
        drop(f);

        let t = fixture(&path, "w", &[2, 3, 4], (0, 96));
        // Slice 1 is the matrix [[12..16],[16..20],[20..24]].
        let s = sample_tensor(&t, 10, 10, 1).unwrap();
        assert_eq!((s.total_rows, s.total_cols), (3, 4));
        assert_eq!((s.slices, s.slice), (2, 1));
        for (i, &r) in s.rows.iter().enumerate() {
            for (j, &c) in s.cols.iter().enumerate() {
                assert_eq!(s.values[i][j], (12 + r * 4 + c) as f64);
            }
        }
        // An out-of-range slice clamps to the last one.
        assert_eq!(sample_tensor(&t, 10, 10, 99).unwrap().slice, 1);
        let _ = std::fs::remove_file(&path);
    }
}

#[cfg(all(test, feature = "hdf5"))]
mod hdf5_tests {
    use super::*;
    use crate::tree::{Layout, Storage};

    #[test]
    fn samples_an_hdf5_dataset_by_value() {
        let dir = std::env::temp_dir().join("checkpoint_explorer_sample_h5");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("d.h5");
        let _ = std::fs::remove_file(&path);
        {
            let file = hdf5_metno::File::create(&path).unwrap();
            let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
            let ds = file.new_dataset::<f32>().shape([4, 5]).create("w").unwrap();
            ds.write_raw(&data).unwrap();
        }

        let t = TensorInfo {
            name: "w".to_string(),
            dtype: "F32".to_string(),
            shape: vec![4, 5],
            size_bytes: 80,
            num_elements: 20,
            storage: Storage::Unknown,
            source_path: path.to_string_lossy().into_owned(),
            layout: Layout::None,
        };
        // libhdf5 converts the stored f32 to f64 on read.
        let s = sample_tensor(&t, 10, 10, 0).unwrap();
        assert_eq!((s.total_rows, s.total_cols), (4, 5));
        for (i, &r) in s.rows.iter().enumerate() {
            for (j, &c) in s.cols.iter().enumerate() {
                assert_eq!(s.values[i][j], (r * 5 + c) as f64);
            }
        }
        let _ = std::fs::remove_file(&path);
    }
}
