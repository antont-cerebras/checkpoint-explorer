//! On-demand sampling of tensor data for the heatmap and numeric views.
//!
//! Tensors can be many GB, so we never read a whole one: we pick a small grid
//! of element indices that fit the screen (including the edges) and read just
//! those. safetensors are read by seeking to the sampled rows; HDF5 datasets
//! are read via libhdf5 (which converts any numeric dtype to `f64` and handles
//! decompression) with a size cap.

use std::io::{Read, Seek, SeekFrom};

use rayon::prelude::*;

use crate::tree::{Layout, TensorInfo};

/// A user override for how a tensor's bytes are decoded, for visualization.
///
/// The stored dtype can misrepresent the data, so the user can reinterpret the
/// raw bytes two ways:
///
/// * [`ViewDtype::As`] decodes each stored container as a *different same-width*
///   dtype — e.g. show a `BF16`-tagged tensor as `F16` (both 16-bit). No shape
///   change.
/// * The 4-bit views handle quantized weights stored inside a wider container
///   (e.g. gpt-oss MoE: 4-bit values in a `bf16`/`f16` slot). `*Lo`/`*Hi` take
///   a single value from the low / high nibble of each container (formats
///   differ on which nibble carries the data); `*Packed` unpacks every nibble
///   densely (a 16-bit slot yields four values, expanding the last dimension).
///
/// Overrides only apply where we read raw bytes (safetensors); HDF5 always uses
/// the stored dtype.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum ViewDtype {
    /// Decode using the tensor's real dtype.
    #[default]
    Stored,
    /// Reinterpret each container as this (same byte width) dtype, e.g. `"F16"`.
    As(&'static str),
    /// Unsigned 4-bit, low nibble of each stored container.
    U4Lo,
    /// Unsigned 4-bit, high nibble of each stored container.
    U4Hi,
    /// Unsigned 4-bit, all nibbles packed densely (last dim ×(bytes·2)).
    U4Packed,
    /// Signed 4-bit (two's complement), low nibble of each stored container.
    I4Lo,
    /// Signed 4-bit, high nibble of each stored container.
    I4Hi,
    /// Signed 4-bit, all nibbles packed densely.
    I4Packed,
}

impl ViewDtype {
    /// Short label for the active override, or `None` when using the stored dtype.
    pub fn label(self) -> Option<&'static str> {
        match self {
            ViewDtype::Stored => None,
            ViewDtype::As(dt) => Some(dt),
            ViewDtype::U4Lo => Some("u4 (low nibble)"),
            ViewDtype::U4Hi => Some("u4 (high nibble)"),
            ViewDtype::U4Packed => Some("u4 (packed)"),
            ViewDtype::I4Lo => Some("i4 (low nibble)"),
            ViewDtype::I4Hi => Some("i4 (high nibble)"),
            ViewDtype::I4Packed => Some("i4 (packed)"),
        }
    }

    /// A compact label for the selection menu (e.g. `stored`, `F16`, `u4·hi`).
    pub fn menu_label(self) -> &'static str {
        match self {
            ViewDtype::Stored => "stored",
            ViewDtype::As(dt) => dt,
            ViewDtype::U4Lo => "u4·lo",
            ViewDtype::U4Hi => "u4·hi",
            ViewDtype::U4Packed => "u4·packed",
            ViewDtype::I4Lo => "i4·lo",
            ViewDtype::I4Hi => "i4·hi",
            ViewDtype::I4Packed => "i4·packed",
        }
    }

    /// How many logical 4-bit values are unpacked from each stored container of
    /// `item_bytes` bytes. `1` for everything except the packed 4-bit views,
    /// which yield `item_bytes * 2` nibbles per container.
    fn packing(self, item_bytes: usize) -> usize {
        match self {
            ViewDtype::U4Packed | ViewDtype::I4Packed => item_bytes * 2,
            _ => 1,
        }
    }

    fn is_signed(self) -> bool {
        matches!(
            self,
            ViewDtype::I4Lo | ViewDtype::I4Hi | ViewDtype::I4Packed
        )
    }

    /// Whether the decoded values are integers (so they should be shown without
    /// a fractional part). True for the 4-bit views and for integer stored / `As`
    /// dtypes; false for floats.
    pub fn is_integer(self, stored: &str) -> bool {
        match self {
            ViewDtype::Stored => dtype_is_integer(stored),
            ViewDtype::As(dt) => dtype_is_integer(dt),
            _ => true, // all 4-bit views are integer-valued
        }
    }
}

impl ViewDtype {
    /// The logical shape under this view: the stored `shape` with its last
    /// dimension scaled by the packing factor. Unchanged unless this is a
    /// packed 4-bit view (which unpacks several values per stored container).
    pub fn logical_shape(self, shape: &[usize], stored_dtype: &str) -> Vec<usize> {
        let packing = item_size(stored_dtype)
            .map(|b| self.packing(b))
            .unwrap_or(1);
        let mut shape = shape.to_vec();
        if packing > 1
            && let Some(last) = shape.last_mut()
        {
            *last *= packing;
        }
        shape
    }
}

/// Whether a dtype label denotes an integer (or boolean) type.
fn dtype_is_integer(dtype: &str) -> bool {
    matches!(
        dtype,
        "I8" | "U8" | "I16" | "U16" | "I32" | "U32" | "I64" | "U64" | "BOOL"
    )
}

/// The ordered list of views to choose from for a tensor of the given stored
/// dtype: the stored dtype, then the other same-width dtypes, then the 4-bit
/// reinterpretations.
pub fn view_options(stored: &str) -> Vec<ViewDtype> {
    let mut opts = vec![ViewDtype::Stored];
    // Same-width float/int reinterpretations (excluding the stored dtype).
    let same_width: &[&str] = match item_size(stored) {
        Some(1) => &["I8", "U8"],
        Some(2) => &["F16", "BF16", "I16", "U16"],
        Some(4) => &["F32", "I32", "U32"],
        Some(8) => &["F64", "I64", "U64"],
        _ => &[],
    };
    opts.extend(
        same_width
            .iter()
            .copied()
            .filter(|&dt| dt != stored)
            .map(ViewDtype::As),
    );
    opts.extend([
        ViewDtype::U4Lo,
        ViewDtype::U4Hi,
        ViewDtype::U4Packed,
        ViewDtype::I4Lo,
        ViewDtype::I4Hi,
        ViewDtype::I4Packed,
    ]);
    opts
}

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
    /// The dtype reinterpretation this sample was decoded with.
    pub view: ViewDtype,
    /// Whether a dtype override is available for this tensor (safetensors only).
    pub overridable: bool,
}

/// Sample a 1D/2D/3D tensor into at most `max_rows` x `max_cols` values. For a
/// 3D tensor `[d0, d1, d2]`, `slice` selects the leading index and the `d1 x d2`
/// matrix at that index is sampled (clamped to a valid slice). `view` overrides
/// how bytes are decoded (e.g. as packed 4-bit), which for a packed view
/// expands the last dimension; it only applies to safetensors.
pub fn sample_tensor(
    t: &TensorInfo,
    max_rows: usize,
    max_cols: usize,
    slice: usize,
    view: ViewDtype,
) -> Result<Sample, String> {
    let ext = std::path::Path::new(&t.source_path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    // We can only reinterpret raw bytes for safetensors; elsewhere fall back to
    // the stored dtype so the header never mislabels what's shown.
    let overridable = ext == "safetensors";
    let view = if overridable { view } else { ViewDtype::Stored };

    // A packed override unpacks several 4-bit values from each stored element,
    // expanding the innermost (last) dimension by that factor.
    let packing = item_size(&t.dtype)
        .map(|bytes| view.packing(bytes))
        .unwrap_or(1);

    let (total_rows, stored_cols, slices) = match t.shape.as_slice() {
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
    let total_cols = stored_cols * packing;
    if total_rows == 0 || total_cols == 0 || slices == 0 {
        return Err("tensor has no elements".to_string());
    }
    let slice = slice.min(slices - 1);
    // Logical elements to skip to reach the chosen slice (0 for 1D/2D).
    let base = slice * total_rows * total_cols;

    let rows = sample_indices(total_rows, max_rows.max(1));
    let cols = sample_indices(total_cols, max_cols.max(1));

    let values = match ext {
        "safetensors" => read_safetensors(t, total_cols, base, &rows, &cols, view)?,
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
        view,
        overridable,
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

/// Decode sub-element `sub` of a stored container `bytes` under `view`. For
/// `Stored`/`As` this decodes the whole container; for the 4-bit views it
/// extracts one nibble of the little-endian container, sign-extending for the
/// signed views. The packed views read nibble `sub`; the low/high views always
/// read the least/most significant nibble (so `sub` is ignored).
fn decode_view(view: ViewDtype, dtype: &str, bytes: &[u8], sub: usize) -> f64 {
    match view {
        ViewDtype::Stored => return decode(dtype, bytes),
        // Same-width reinterpretation: decode the container as the chosen dtype.
        ViewDtype::As(dt) => return decode(dt, bytes),
        _ => {}
    }
    // Little-endian integer value of the container (up to 8 bytes).
    let mut container: u64 = 0;
    for (i, &b) in bytes.iter().take(8).enumerate() {
        container |= (b as u64) << (8 * i);
    }
    let nib_index = match view {
        ViewDtype::U4Packed | ViewDtype::I4Packed => sub,
        ViewDtype::U4Hi | ViewDtype::I4Hi => bytes.len() * 2 - 1,
        _ => 0, // low-nibble views
    };
    let nibble = ((container >> (nib_index * 4)) & 0xF) as i64;
    if view.is_signed() && nibble >= 8 {
        (nibble - 16) as f64
    } else {
        nibble as f64
    }
}

/// Exact whole-tensor statistics (under a given [`ViewDtype`]), computed by
/// scanning every element once.
#[derive(Clone, Copy, Debug)]
pub struct Stats {
    /// Total elements scanned.
    pub count: u64,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    /// Population standard deviation.
    pub std: f64,
    /// Number of exactly-zero elements (sparsity).
    pub zeros: u64,
    /// Number of non-finite elements (NaN / ±Inf).
    pub nonfinite: u64,
    /// How long the scan took (set by [`tensor_stats`]).
    pub elapsed: std::time::Duration,
}

impl Stats {
    /// Fraction of elements that are exactly zero, in `0.0..=1.0`.
    pub fn zero_fraction(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.zeros as f64 / self.count as f64
        }
    }
}

/// Mergeable accumulator for a parallel single-pass scan. Mean and variance use
/// Welford's algorithm (tracking the running `mean` and `m2`, the sum of squared
/// deviations) — numerically stable and free of the catastrophic cancellation
/// that `E[x²] − E[x]²` suffers when the mean dominates the spread. The merge
/// rule (Chan et al.) combines two partials associatively, so rayon can reduce.
#[derive(Clone, Copy)]
struct Acc {
    count: u64,
    /// Finite elements (the `n` for mean/variance).
    finite: u64,
    zeros: u64,
    nonfinite: u64,
    min: f64,
    max: f64,
    mean: f64,
    m2: f64,
}

impl Acc {
    const ID: Acc = Acc {
        count: 0,
        finite: 0,
        zeros: 0,
        nonfinite: 0,
        min: f64::INFINITY,
        max: f64::NEG_INFINITY,
        mean: 0.0,
        m2: 0.0,
    };

    #[inline]
    fn push(&mut self, v: f64) {
        self.count += 1;
        if !v.is_finite() {
            self.nonfinite += 1;
            return;
        }
        if v < self.min {
            self.min = v;
        }
        if v > self.max {
            self.max = v;
        }
        if v == 0.0 {
            self.zeros += 1;
        }
        // Welford online update.
        self.finite += 1;
        let delta = v - self.mean;
        self.mean += delta / self.finite as f64;
        self.m2 += delta * (v - self.mean);
    }

    fn merge(a: Acc, b: Acc) -> Acc {
        let finite = a.finite + b.finite;
        let (mean, m2) = if finite == 0 {
            (0.0, 0.0)
        } else {
            // Parallel-variance combine; the `nb/n` and `na*nb/n` factors make
            // an empty side contribute nothing (so this also handles ID).
            let (na, nb) = (a.finite as f64, b.finite as f64);
            let n = na + nb;
            let delta = b.mean - a.mean;
            (
                a.mean + delta * nb / n,
                a.m2 + b.m2 + delta * delta * na * nb / n,
            )
        };
        Acc {
            count: a.count + b.count,
            finite,
            zeros: a.zeros + b.zeros,
            nonfinite: a.nonfinite + b.nonfinite,
            min: a.min.min(b.min),
            max: a.max.max(b.max),
            mean,
            m2,
        }
    }

    fn finish(self) -> Stats {
        let (min, max, mean, std) = if self.finite > 0 {
            // Population variance is M2 / n.
            let std = (self.m2 / self.finite as f64).sqrt();
            (self.min, self.max, self.mean, std)
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };
        Stats {
            count: self.count,
            min,
            max,
            mean,
            std,
            zeros: self.zeros,
            nonfinite: self.nonfinite,
            elapsed: std::time::Duration::ZERO,
        }
    }
}

/// Containers processed per parallel task (keeps per-task overhead low).
const STATS_CHUNK: usize = 1 << 16;

/// Compute exact statistics over the whole tensor under `view`. Reads every
/// element once — memory-mapped and decoded in parallel for safetensors; for
/// HDF5 it reads the (decompressed) dataset, capped in size. Only safetensors
/// honours a non-`Stored` view (HDF5 always uses the stored dtype).
pub fn tensor_stats(t: &TensorInfo, view: ViewDtype) -> Result<Stats, String> {
    let ext = std::path::Path::new(&t.source_path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    let started = std::time::Instant::now();
    let mut stats = match ext {
        "safetensors" => stats_safetensors(t, view),
        "h5" | "hdf5" => stats_hdf5(t),
        _ => Err("statistics are not supported for this format".to_string()),
    }?;
    stats.elapsed = started.elapsed();
    Ok(stats)
}

fn stats_safetensors(t: &TensorInfo, view: ViewDtype) -> Result<Stats, String> {
    let Layout::ByteRange { start, end } = t.layout else {
        return Err("tensor data location is unknown".to_string());
    };
    let item = item_size(&t.dtype).ok_or_else(|| format!("unsupported dtype: {}", t.dtype))?;
    let packing = view.packing(item);

    let file = std::fs::File::open(&t.source_path).map_err(|e| e.to_string())?;
    // SAFETY: read-only inspection; we accept that a concurrent external write
    // could change the mapping (standard tradeoff for mmap-based readers).
    let mmap = unsafe { memmap2::Mmap::map(&file).map_err(|e| e.to_string())? };

    let header_len =
        u64::from_le_bytes(mmap.get(0..8).ok_or("file too small")?.try_into().unwrap());
    let data_start = (8 + header_len + start) as usize;
    let data_end = (8 + header_len + end) as usize;
    let bytes = mmap
        .get(data_start..data_end)
        .ok_or("tensor data range is out of bounds")?;

    let dtype = t.dtype.as_str();
    let acc = bytes
        .par_chunks(item * STATS_CHUNK)
        .map(|chunk| {
            let mut a = Acc::ID;
            for container in chunk.chunks_exact(item) {
                for sub in 0..packing {
                    a.push(decode_view(view, dtype, container, sub));
                }
            }
            a
        })
        .reduce(|| Acc::ID, Acc::merge);
    Ok(acc.finish())
}

#[cfg(feature = "hdf5")]
fn stats_hdf5(t: &TensorInfo) -> Result<Stats, String> {
    use hdf5_metno::{Hyperslab, SliceOrIndex};

    let file = hdf5_metno::File::open(&t.source_path).map_err(|e| e.to_string())?;
    // Ensure LZ4-compressed datasets are decodable (no-op after the first call).
    crate::hdf5_lz4::register();
    let key = file
        .member_names()
        .map_err(|e| e.to_string())?
        .into_iter()
        .find(|k| crate::hdf5::percent_decode(k) == t.name)
        .ok_or_else(|| "dataset not found in file".to_string())?;
    let dataset = file.dataset(&key).map_err(|e| e.to_string())?;
    let shape = dataset.shape();

    // ≤32-bit-exact sources are read as f32 to halve the per-block buffer (the
    // values round-trip exactly); wider/integer sources stay f64. Either way the
    // accumulators are double precision. `read` reads a block and folds it.
    let use_f32 = matches!(
        t.dtype.as_str(),
        "F16" | "BF16" | "F32" | "I8" | "U8" | "I16" | "U16"
    );
    let read = |hyper: Hyperslab| -> Result<Acc, String> {
        if use_f32 {
            let a = dataset
                .read_slice::<f32, _, ndarray::IxDyn>(hyper)
                .map_err(|e| e.to_string())?;
            Ok(fold_block(a.as_slice(), a.iter(), |v| v as f64))
        } else {
            let a = dataset
                .read_slice::<f64, _, ndarray::IxDyn>(hyper)
                .map_err(|e| e.to_string())?;
            Ok(fold_block(a.as_slice(), a.iter(), |v| v))
        }
    };

    // 0-D (scalar): a single block over the whole (degenerate) shape.
    if shape.is_empty() {
        return read(Hyperslab::from(Vec::new())).map(Acc::finish);
    }

    // Stream along the outer axis in row-blocks so memory stays bounded
    // regardless of tensor size (HDF5 decompresses the overlapping chunks).
    let outer = shape[0];
    let inner: usize = shape[1..].iter().product::<usize>().max(1);
    const BLOCK_ELEMS: usize = 16 << 20; // ≈16M elements (~64 MiB as f32) per read
    let block = (BLOCK_ELEMS / inner).max(1);

    let mut acc = Acc::ID;
    let mut i = 0;
    while i < outer {
        let hi = (i + block).min(outer);
        // Hyperslab: rows [i, hi) on axis 0, the full (bounded) extent elsewhere.
        let mut dims: Vec<SliceOrIndex> = Vec::with_capacity(shape.len());
        dims.push(SliceOrIndex::from(i..hi));
        for &d in &shape[1..] {
            dims.push(SliceOrIndex::from(0..d));
        }
        acc = Acc::merge(acc, read(Hyperslab::from(dims))?);
        i = hi;
    }
    Ok(acc.finish())
}

/// Fold one read block into an [`Acc`]: rayon-reduce the contiguous slice when
/// available (a fresh read is standard-layout), else iterate.
#[cfg(feature = "hdf5")]
fn fold_block<'a, T: Copy + Send + Sync + 'a>(
    contiguous: Option<&[T]>,
    iter: impl Iterator<Item = &'a T>,
    to_f64: impl Fn(T) -> f64 + Sync,
) -> Acc {
    if let Some(s) = contiguous {
        reduce_to_acc(s, to_f64)
    } else {
        let mut a = Acc::ID;
        for &v in iter {
            a.push(to_f64(v));
        }
        a
    }
}

/// Scan a safetensors tensor's bytes into an [`Acc`], either with the rayon
/// `par_chunks` reduce (as production does) or a plain sequential `chunks` fold
/// over the same chunking and decode. Used by the seq-vs-parallel benchmark to
/// compare timing and results fairly. Returns `(stats, scan_time)` (timing the
/// reduce only, not the mmap/header parse).
#[cfg(test)]
fn bench_scan(t: &TensorInfo, view: ViewDtype, parallel: bool) -> (Stats, std::time::Duration) {
    let Layout::ByteRange { start, end } = t.layout else {
        panic!("benchmark expects a safetensors ByteRange tensor");
    };
    let item = item_size(&t.dtype).expect("known dtype");
    let packing = view.packing(item);
    let file = std::fs::File::open(&t.source_path).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap());
    let bytes = &mmap[(8 + header_len + start) as usize..(8 + header_len + end) as usize];
    let dtype = t.dtype.as_str();

    let chunk_acc = |chunk: &[u8]| {
        let mut a = Acc::ID;
        for container in chunk.chunks_exact(item) {
            for sub in 0..packing {
                a.push(decode_view(view, dtype, container, sub));
            }
        }
        a
    };

    let started = std::time::Instant::now();
    let acc = if parallel {
        bytes
            .par_chunks(item * STATS_CHUNK)
            .map(chunk_acc)
            .reduce(|| Acc::ID, Acc::merge)
    } else {
        bytes
            .chunks(item * STATS_CHUNK)
            .map(chunk_acc)
            .fold(Acc::ID, Acc::merge)
    };
    (acc.finish(), started.elapsed())
}

/// Reduce a typed slice into an [`Acc`] in parallel, converting each element to
/// `f64` for the (double-precision) accumulators.
#[cfg(feature = "hdf5")]
fn reduce_to_acc<T: Copy + Send + Sync>(data: &[T], to_f64: impl Fn(T) -> f64 + Sync) -> Acc {
    data.par_chunks(STATS_CHUNK)
        .map(|chunk| {
            let mut a = Acc::ID;
            for &v in chunk {
                a.push(to_f64(v));
            }
            a
        })
        .reduce(|| Acc::ID, Acc::merge)
}

#[cfg(not(feature = "hdf5"))]
fn stats_hdf5(_t: &TensorInfo) -> Result<Stats, String> {
    Err("HDF5 support is not compiled in (rebuild with `--features hdf5`)".to_string())
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
///
/// Indices are logical: under a packed `view`, `total_cols` and `cols` count
/// 4-bit values, while the file stores `item`-byte containers each holding
/// `packing` of them. So a logical element `flat` lives in container `flat /
/// packing` at nibble `flat % packing`.
fn read_safetensors(
    t: &TensorInfo,
    total_cols: usize,
    base: usize,
    rows: &[usize],
    cols: &[usize],
    view: ViewDtype,
) -> Result<Vec<Vec<f64>>, String> {
    let Layout::ByteRange { start, .. } = t.layout else {
        return Err("tensor data location is unknown".to_string());
    };
    let item = item_size(&t.dtype).ok_or_else(|| format!("unsupported dtype: {}", t.dtype))?;
    let packing = view.packing(item);

    let mut file = std::fs::File::open(&t.source_path).map_err(|e| e.to_string())?;
    // The data blob begins after the 8-byte header length and the JSON header.
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf).map_err(|e| e.to_string())?;
    let data_start = 8 + u64::from_le_bytes(len_buf) + start;

    let decode_at = |buf: &[u8], local_container: usize, sub: usize| {
        let off = local_container * item;
        decode_view(view, &t.dtype, &buf[off..off + item], sub)
    };

    let mut out = Vec::with_capacity(rows.len());
    // Read each sampled row's container span in one go when it's reasonably
    // sized; otherwise fall back to one read per sampled element.
    const MAX_SPAN: usize = 64 * 1024 * 1024;
    for &r in rows {
        let row_base = base + r * total_cols;
        let first_container = (row_base + *cols.first().unwrap()) / packing;
        let last_container = (row_base + *cols.last().unwrap()) / packing;
        let span_bytes = (last_container - first_container + 1) * item;

        let row: Vec<f64> = if span_bytes <= MAX_SPAN {
            let mut buf = vec![0u8; span_bytes];
            let off = data_start + (first_container as u64) * (item as u64);
            file.seek(SeekFrom::Start(off)).map_err(|e| e.to_string())?;
            file.read_exact(&mut buf).map_err(|e| e.to_string())?;
            cols.iter()
                .map(|&c| {
                    let flat = row_base + c;
                    decode_at(&buf, flat / packing - first_container, flat % packing)
                })
                .collect()
        } else {
            // Per-element reads for very wide rows.
            let mut buf = vec![0u8; item];
            let mut row = Vec::with_capacity(cols.len());
            for &c in cols {
                let flat = row_base + c;
                let off = data_start + ((flat / packing) as u64) * (item as u64);
                file.seek(SeekFrom::Start(off)).map_err(|e| e.to_string())?;
                file.read_exact(&mut buf).map_err(|e| e.to_string())?;
                row.push(decode_view(view, &t.dtype, &buf, flat % packing));
            }
            row
        };
        out.push(row);
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
    // Ensure LZ4-compressed datasets are decodable (no-op after the first call).
    crate::hdf5_lz4::register();
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
        fixture_dtype(path, name, "F32", shape, offsets)
    }

    fn fixture_dtype(
        path: &std::path::Path,
        name: &str,
        dtype: &str,
        shape: &[usize],
        offsets: (u64, u64),
    ) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            dtype: dtype.to_string(),
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
        let s = sample_tensor(&t, 10, 10, 0, ViewDtype::Stored).unwrap();
        assert_eq!((s.total_rows, s.total_cols), (4, 5));
        assert_eq!((s.slices, s.slice), (1, 0));
        assert!(s.overridable && s.view == ViewDtype::Stored);
        assert_eq!(s.min, 0.0);
        assert_eq!(s.max, 19.0);
        for (i, &r) in s.rows.iter().enumerate() {
            for (j, &c) in s.cols.iter().enumerate() {
                assert_eq!(s.values[i][j], (r * 5 + c) as f64);
            }
        }
        let _ = std::fs::remove_file(&path);
    }

    /// Manual benchmark: `BENCH_FILE=... BENCH_TENSOR=... cargo test --release
    /// -- --ignored --nocapture seq_vs_parallel`. Compares the rayon reduce
    /// against a sequential fold (same decode + accumulator) on a real tensor.
    #[test]
    #[ignore = "manual benchmark; set BENCH_FILE and BENCH_TENSOR"]
    fn seq_vs_parallel_accumulation() {
        let path = std::env::var("BENCH_FILE").expect("set BENCH_FILE");
        let name = std::env::var("BENCH_TENSOR").expect("set BENCH_TENSOR");

        // Parse the safetensors header to build a TensorInfo for `name`.
        let mut f = std::fs::File::open(&path).unwrap();
        let mut len = [0u8; 8];
        f.read_exact(&mut len).unwrap();
        let mut hb = vec![0u8; u64::from_le_bytes(len) as usize];
        f.read_exact(&mut hb).unwrap();
        let hdr: serde_json::Value = serde_json::from_slice(&hb).unwrap();
        let info = &hdr[&name];
        let dtype = info["dtype"].as_str().unwrap().to_string();
        let shape: Vec<usize> = info["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let off = info["data_offsets"].as_array().unwrap();
        let (s, e) = (off[0].as_u64().unwrap(), off[1].as_u64().unwrap());
        let t = TensorInfo {
            name: name.clone(),
            dtype,
            shape: shape.clone(),
            size_bytes: (e - s) as usize,
            num_elements: shape.iter().product(),
            storage: crate::tree::Storage::Unknown,
            source_path: path.clone(),
            layout: Layout::ByteRange { start: s, end: e },
        };

        let view = ViewDtype::Stored;
        // First sequential run is cold (it faults in the whole tensor); the next
        // two run from the warmed page cache, isolating accumulation cost.
        let (_, t_cold) = bench_scan(&t, view, false);
        let (seq, t_seq) = bench_scan(&t, view, false);
        let (par, t_par) = bench_scan(&t, view, true);

        eprintln!("tensor {name} — {} elements, {}", t.num_elements, t.dtype);
        eprintln!("sequential (cold, incl I/O): {t_cold:?}");
        eprintln!("sequential (warm):           {t_seq:?}");
        eprintln!("parallel   (warm):           {t_par:?}");
        eprintln!(
            "speedup (seq/par, warm):     {:.2}x",
            t_seq.as_secs_f64() / t_par.as_secs_f64()
        );
        eprintln!("seq stats: {seq:?}");
        eprintln!("par stats: {par:?}");

        // min/max are order-independent (exact); mean/std differ only by
        // floating-point summation order, so allow a tiny relative slack.
        assert_eq!((seq.min, seq.max), (par.min, par.max));
        assert!((seq.mean - par.mean).abs() <= seq.mean.abs() * 1e-9 + 1e-12);
        assert!((seq.std - par.std).abs() <= seq.std.abs() * 1e-9 + 1e-12);
    }

    #[test]
    fn computes_exact_whole_tensor_stats() {
        use std::io::Write;
        let dir = std::env::temp_dir().join("checkpoint_explorer_stats");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("w.safetensors");
        // 4x5 f32, values 0..=19 (so one exact zero, mean 9.5).
        let header = br#"{"w":{"dtype":"F32","shape":[4,5],"data_offsets":[0,80]}}"#;
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(header).unwrap();
        for i in 0..20u32 {
            f.write_all(&(i as f32).to_le_bytes()).unwrap();
        }
        drop(f);

        let t = fixture(&path, "w", &[4, 5], (0, 80));
        let s = tensor_stats(&t, ViewDtype::Stored).unwrap();
        assert_eq!(s.count, 20);
        assert_eq!((s.min, s.max), (0.0, 19.0));
        assert!((s.mean - 9.5).abs() < 1e-9);
        // Population std of 0..=19 is sqrt(33.25) ≈ 5.76628.
        assert!((s.std - 5.766_281_3).abs() < 1e-5);
        assert_eq!(s.zeros, 1);
        assert_eq!(s.nonfinite, 0);
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
        let s = sample_tensor(&t, 10, 10, 1, ViewDtype::Stored).unwrap();
        assert_eq!((s.total_rows, s.total_cols), (3, 4));
        assert_eq!((s.slices, s.slice), (2, 1));
        for (i, &r) in s.rows.iter().enumerate() {
            for (j, &c) in s.cols.iter().enumerate() {
                assert_eq!(s.values[i][j], (12 + r * 4 + c) as f64);
            }
        }
        // An out-of-range slice clamps to the last one.
        assert_eq!(
            sample_tensor(&t, 10, 10, 99, ViewDtype::Stored)
                .unwrap()
                .slice,
            1
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn reinterprets_packed_4bit_views() {
        use std::io::Write;
        let dir = std::env::temp_dir().join("checkpoint_explorer_sample_u4");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("w.safetensors");
        // Shape [2] of F16 (2-byte containers): u16 values 0x1234 and 0x00AB.
        let header = br#"{"w":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}"#;
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(header).unwrap();
        f.write_all(&0x1234u16.to_le_bytes()).unwrap();
        f.write_all(&0x00ABu16.to_le_bytes()).unwrap();
        drop(f);

        let t = fixture_dtype(&path, "w", "F16", &[2], (0, 4));

        // Low nibble of each container -> [0x4, 0xB]. Shape unchanged.
        let s = sample_tensor(&t, 10, 10, 0, ViewDtype::U4Lo).unwrap();
        assert_eq!(s.total_cols, 2);
        assert_eq!(s.values[0], vec![4.0, 11.0]);

        // High nibble (bits 12-15) of each container -> 0x1234->0x1, 0x00AB->0x0.
        let s = sample_tensor(&t, 10, 10, 0, ViewDtype::U4Hi).unwrap();
        assert_eq!(s.total_cols, 2);
        assert_eq!(s.values[0], vec![1.0, 0.0]);

        // Packed: four nibbles per 16-bit container, last dim ×4 -> 8 values.
        // 0x1234 -> [4,3,2,1]; 0x00AB -> [11,10,0,0].
        let s = sample_tensor(&t, 10, 10, 0, ViewDtype::U4Packed).unwrap();
        assert_eq!(s.total_cols, 8);
        assert_eq!(s.values[0], vec![4.0, 3.0, 2.0, 1.0, 11.0, 10.0, 0.0, 0.0]);

        // Signed packed: nibbles >= 8 are negative (0xB->-5, 0xA->-6).
        let s = sample_tensor(&t, 10, 10, 0, ViewDtype::I4Packed).unwrap();
        assert_eq!(s.values[0], vec![4.0, 3.0, 2.0, 1.0, -5.0, -6.0, 0.0, 0.0]);

        let _ = std::fs::remove_file(&path);
    }
}

#[cfg(all(test, feature = "hdf5"))]
mod hdf5_tests {
    use super::*;
    use crate::tree::{Layout, Storage};

    /// Manual: `BENCH_FILE=<.hdf5> BENCH_TENSOR=<name> cargo test --release
    /// --features hdf5 -- --ignored --nocapture hdf5_stats_timing`.
    #[test]
    #[ignore = "manual; set BENCH_FILE and BENCH_TENSOR"]
    fn hdf5_stats_timing() {
        let path = std::env::var("BENCH_FILE").expect("set BENCH_FILE");
        let name = std::env::var("BENCH_TENSOR").expect("set BENCH_TENSOR");
        let tensors = crate::hdf5::read_tensors(std::path::Path::new(&path)).unwrap();
        let t = tensors.into_iter().find(|t| t.name == name).expect("tensor");
        eprintln!("tensor {} dtype={} shape={:?}", t.name, t.dtype, t.shape);
        let started = std::time::Instant::now();
        match tensor_stats(&t, ViewDtype::Stored) {
            Ok(s) => eprintln!("ok in {:?}: {s:?}", started.elapsed()),
            Err(e) => eprintln!("ERR in {:?}: {e}", started.elapsed()),
        }
    }

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
        let s = sample_tensor(&t, 10, 10, 0, ViewDtype::Stored).unwrap();
        assert_eq!((s.total_rows, s.total_cols), (4, 5));
        // HDF5 cannot be byte-reinterpreted, so it is not overridable.
        assert!(!s.overridable);
        for (i, &r) in s.rows.iter().enumerate() {
            for (j, &c) in s.cols.iter().enumerate() {
                assert_eq!(s.values[i][j], (r * 5 + c) as f64);
            }
        }

        // Exact stats over the whole dataset (exercises the f32 read path for a
        // ≤32-bit float source). Values 0..=19, so mean 9.5 and one zero.
        let st = tensor_stats(&t, ViewDtype::Stored).unwrap();
        assert_eq!(st.count, 20);
        assert_eq!((st.min, st.max), (0.0, 19.0));
        assert!((st.mean - 9.5).abs() < 1e-9);
        assert_eq!(st.zeros, 1);
        assert_eq!(st.nonfinite, 0);

        let _ = std::fs::remove_file(&path);
    }
}
