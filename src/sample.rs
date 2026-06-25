//! On-demand sampling of tensor data for the heatmap and numeric views.
//!
//! Tensors can be many GB, so we never read a whole one for the preview: we
//! pick a small grid of element indices that fit the screen (including the
//! edges) and read just those rows' column spans. Backing formats are reached
//! through one [`TensorReader`] abstraction — memory-mapped safetensors and
//! libhdf5 datasets today — so the preview and the statistics scan are written
//! once and work the same regardless of format (and a new one, e.g. remote/S3
//! shards, is just another implementation).

use std::ops::{ControlFlow, Range};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

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
/// * The 4-bit views ([`ViewDtype::U4`] / [`ViewDtype::I4`]) handle quantized
///   weights stored inside a wider container (e.g. gpt-oss MoE: 4-bit values in
///   a `bf16`/`f16` slot). They unpack every nibble densely (a 16-bit slot
///   yields four values, expanding the last dimension).
///
/// Overrides apply wherever we read the raw stored bytes — both safetensors and
/// HDF5.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum ViewDtype {
    /// Decode using the tensor's real dtype.
    #[default]
    Stored,
    /// Reinterpret each container as this (same byte width) dtype, e.g. `"F16"`.
    As(&'static str),
    /// Unsigned 4-bit: every nibble unpacked densely (last dim ×(bytes·2)).
    U4,
    /// Signed 4-bit (two's complement): every nibble unpacked densely.
    I4,
}

impl ViewDtype {
    /// Short label for the active override, or `None` when using the stored dtype.
    pub fn label(self) -> Option<&'static str> {
        match self {
            ViewDtype::Stored => None,
            ViewDtype::As(dt) => Some(dt),
            ViewDtype::U4 => Some("u4"),
            ViewDtype::I4 => Some("i4"),
        }
    }

    /// The `--dtype` CLI value that re-selects this view (e.g. `f16`, `u4`), or
    /// `None` for the stored dtype (no flag needed). Inverse of
    /// [`parse_view_dtype`].
    pub fn cli_value(self) -> Option<String> {
        Some(match self {
            ViewDtype::Stored => return None,
            ViewDtype::As(dt) => dt.to_ascii_lowercase(),
            ViewDtype::U4 => "u4".to_string(),
            ViewDtype::I4 => "i4".to_string(),
        })
    }

    /// A compact label for the selection menu (e.g. `stored`, `F16`, `u4`).
    pub fn menu_label(self) -> &'static str {
        match self {
            ViewDtype::Stored => "stored",
            ViewDtype::As(dt) => dt,
            ViewDtype::U4 => "u4",
            ViewDtype::I4 => "i4",
        }
    }

    /// How many logical 4-bit values are unpacked from each stored container of
    /// `item_bytes` bytes. `1` for everything except the 4-bit views, which
    /// unpack `item_bytes * 2` nibbles per container.
    fn packing(self, item_bytes: usize) -> usize {
        match self {
            ViewDtype::U4 | ViewDtype::I4 => item_bytes * 2,
            _ => 1,
        }
    }

    fn is_signed(self) -> bool {
        matches!(self, ViewDtype::I4)
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

    /// Bit width of one element under this view — 4 for the packed/nibble views,
    /// otherwise the byte width of the (possibly reinterpreted) dtype. Used to
    /// size and zero-pad hex/octal/binary cells, and matches [`RawBits::width`].
    pub fn bit_width(self, stored: &str) -> u32 {
        match self {
            ViewDtype::Stored => item_size(stored).unwrap_or(4) as u32 * 8,
            ViewDtype::As(dt) => item_size(dt).unwrap_or(4) as u32 * 8,
            _ => 4, // all 4-bit views
        }
    }

    /// Display width (chars, incl. a 1-col gap) for one value cell in the
    /// numeric grid. Floats use a fixed scientific-notation width. Integer
    /// views size to the *actual* values: given the exact whole-tensor `range`
    /// (min, max), a sparse 16-bit tensor of two-digit numbers packs as many
    /// columns as a 4-bit view, instead of always reserving room for `-32768`.
    /// Without a range (stats not computed yet) it falls back to the dtype's
    /// theoretical maximum width.
    pub fn cell_width(self, stored: &str, range: Option<(f64, f64)>) -> usize {
        // Floats render in scientific notation — a fixed width regardless of
        // magnitude (e.g. `-1.234e-05`).
        if !self.is_integer(stored) {
            return 11;
        }
        let digits = match range {
            Some((lo, hi)) => int_digits(lo).max(int_digits(hi)),
            None => self.int_max_digits(stored),
        };
        // +1 for a separating space; a small floor keeps tiny values readable.
        (digits + 1).max(3)
    }

    /// Widest decimal width (digits plus any minus sign) this integer view can
    /// produce, used to size cells before the exact value range is known.
    fn int_max_digits(self, stored: &str) -> usize {
        let dt = match self {
            ViewDtype::U4 => return 2, // 0..=15
            ViewDtype::I4 => return 2, // -8..=7
            ViewDtype::As(dt) => dt,
            ViewDtype::Stored => stored,
        };
        match dt {
            "I8" | "U8" | "BOOL" => 4, // -128
            "I16" | "U16" => 6,        // -32768
            "I32" | "U32" => 11,       // -2147483648
            "I64" | "U64" => 20,       // -9223372036854775808
            _ => 10,
        }
    }
}

/// Decimal width of an integer-valued `f64` (digit count plus a leading minus).
fn int_digits(v: f64) -> usize {
    (v as i64).to_string().len()
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
    opts.extend([ViewDtype::U4, ViewDtype::I4]);
    opts
}

/// Parse a CLI dtype-override string (e.g. `u4`, `i4`, `f16`, `stored`) into a
/// [`ViewDtype`]. Case-insensitive; `-` and `_` are interchangeable. Used by the
/// `--dtype` flag.
pub fn parse_view_dtype(s: &str) -> Result<ViewDtype, String> {
    let norm = s.trim().to_ascii_lowercase().replace('_', "-");
    Ok(match norm.as_str() {
        "stored" => ViewDtype::Stored,
        "u4" => ViewDtype::U4,
        "i4" => ViewDtype::I4,
        "f16" => ViewDtype::As("F16"),
        "bf16" => ViewDtype::As("BF16"),
        "f32" => ViewDtype::As("F32"),
        "f64" => ViewDtype::As("F64"),
        "i8" => ViewDtype::As("I8"),
        "u8" => ViewDtype::As("U8"),
        "i16" => ViewDtype::As("I16"),
        "u16" => ViewDtype::As("U16"),
        "i32" => ViewDtype::As("I32"),
        "u32" => ViewDtype::As("U32"),
        "i64" => ViewDtype::As("I64"),
        "u64" => ViewDtype::As("U64"),
        "bool" => ViewDtype::As("BOOL"),
        _ => {
            return Err(format!(
                "unknown dtype view '{s}'; expected one of: stored, u4, i4, f16, bf16, f32, f64, \
                 i8, u8, i16, u16, i32, u32, i64, u64, bool"
            ));
        }
    })
}

/// A downsampled grid of tensor values plus the original indices it came from.
/// The raw stored bit pattern of one sampled element, for non-decimal numeral
/// display. `bits` is the little-endian integer value of the element's bytes (a
/// single nibble for the 4-bit views); `width` is its bit count (e.g. 32 for an
/// `F32`, 8 for an `I8`, 4 for a packed-4-bit view). Decimal display ignores
/// this and uses the decoded [`f64`] in `values`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawBits {
    pub bits: u64,
    pub width: u8,
}

pub struct Sample {
    /// Original row indices that were sampled (logical rows).
    pub rows: Vec<usize>,
    /// Original column indices that were sampled.
    pub cols: Vec<usize>,
    /// Sampled values, `values[i][j]` for `(rows[i], cols[j])`.
    pub values: Vec<Vec<f64>>,
    /// Raw stored bits of each sampled element, parallel to `values`, for
    /// hex/octal/binary display.
    pub raw: Vec<Vec<RawBits>>,
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
    /// Whether a dtype override is available for this tensor (safetensors/HDF5).
    pub overridable: bool,
    /// Which sampling produced this grid (evenly-spaced vs. edges).
    pub mode: SampleMode,
    /// The shape the data is being viewed as — the effective (possibly
    /// overridden) shape with the dtype packing applied to its last dimension.
    /// The header shows it against the stored shape.
    pub display_shape: Vec<usize>,
}

/// How [`sample_tensor`] chooses which rows/columns to show.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub enum SampleMode {
    /// Evenly-spaced indices across the whole matrix (the default overview).
    #[default]
    Grid,
    /// The first and last rows and columns, contiguously — to inspect edge
    /// padding (e.g. is a tensor zero-padded, or padded with something else).
    /// `row_tail` / `col_tail` bias the fixed budget toward the tail: `0.0`
    /// shows only the first rows/cols, `1.0` only the last, `0.5` is balanced.
    Edges { row_tail: f32, col_tail: f32 },
    /// A contiguous window into the matrix, its top-left corner at
    /// `(row_off, col_off)` — pan it around to read the actual neighbouring
    /// values rather than a downsample. Offsets are clamped so the window never
    /// runs past the end.
    Window { row_off: usize, col_off: usize },
}

/// Sample a 1D/2D/3D tensor into at most `max_rows` x `max_cols` values. For a
/// 3D tensor `[d0, d1, d2]`, `slice` selects the leading index and the `d1 x d2`
/// matrix at that index is sampled (clamped to a valid slice). `view` overrides
/// how bytes are decoded (e.g. as packed 4-bit), which for a packed view
/// expands the last dimension; it applies to safetensors and HDF5. `mode`
/// selects an evenly-spaced grid or the first/last rows & columns (edges).
/// A tensor's shape with size-1 dimensions removed. A dimension of length 1
/// holds a single index, so it carries no data and contributes a no-op stride
/// to the row-major layout — the flat byte sequence is identical with or without
/// it. Dropping such dimensions lets an `(N, M, 1, K)` tensor preview as the 3D
/// `(N, M, K)` it effectively is. An all-ones (or already empty) shape squeezes
/// to a single element, returned as the empty slice.
pub fn squeezed_shape(shape: &[usize]) -> Vec<usize> {
    shape.iter().copied().filter(|&d| d != 1).collect()
}

/// Convenience wrapper: open a reader for `t` and sample in one call. Interactive
/// views use [`sample_tensor_with`] with a reader cached across redraws, so in the
/// binary this one-shot form is currently used only by tests.
#[allow(dead_code)]
pub fn sample_tensor(
    t: &TensorInfo,
    max_rows: usize,
    max_cols: usize,
    slice: usize,
    view: ViewDtype,
    mode: SampleMode,
) -> Result<Sample, String> {
    let reader = open_reader(t)?;
    sample_tensor_with(
        reader.as_ref(),
        t,
        &t.shape,
        max_rows,
        max_cols,
        slice,
        view,
        mode,
    )
}

/// Like [`sample_tensor`], but reads through a caller-supplied open reader so an
/// interactive view can reuse one across redraws (re-opening the file each frame
/// dominates the cost — see the `window_pan_open_cost` benchmark — and for HDF5
/// it also throws away libhdf5's chunk cache, forcing re-decompression). The
/// reader must be for `t` (same file and dtype/shape).
/// `shape` is the shape to fold the flat data into for display — normally the
/// tensor's stored shape, but a shape override (with the same element count)
/// lets the caller reinterpret the dimensions. Region reads still go through
/// `reader`'s real stored shape, so any reshape with a matching element count is
/// a valid row-major reinterpretation.
#[allow(clippy::too_many_arguments)] // distinct sampling parameters
pub(crate) fn sample_tensor_with(
    reader: &dyn TensorReader,
    t: &TensorInfo,
    shape: &[usize],
    max_rows: usize,
    max_cols: usize,
    slice: usize,
    view: ViewDtype,
    mode: SampleMode,
) -> Result<Sample, String> {
    let ext = std::path::Path::new(&t.source_path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    // Dtype overrides reinterpret raw stored bytes; supported for the formats we
    // read byte-for-byte (safetensors, NumPy, HDF5). For any other format fall
    // back to the stored dtype so the header never mislabels what's shown.
    let overridable = matches!(ext, "safetensors" | "h5" | "hdf5" | "npy" | "npz");
    let view = if overridable { view } else { ViewDtype::Stored };

    // A packed override unpacks several 4-bit values from each stored element,
    // expanding the innermost (last) dimension by that factor.
    let item = item_size(&t.dtype);
    let packing = item.map(|bytes| view.packing(bytes)).unwrap_or(1);

    // Size-1 dimensions don't change the flat layout, so we fold rows/cols/slices
    // from the squeezed shape (region reads below still use the real shape — flat
    // indices are layout-invariant under squeezing, and under any reshape with a
    // matching element count).
    let (total_rows, stored_cols, slices) = match squeezed_shape(shape).as_slice() {
        [] => (1usize, 1usize, 1usize),
        [n] => (1usize, *n, 1usize),
        [r, c] => (*r, *c, 1usize),
        [d0, d1, d2] => (*d1, *d2, *d0),
        rest => {
            return Err(format!(
                "data preview supports 1D, 2D and 3D tensors only (this one is {}D)",
                rest.len()
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

    let (rows, cols) = match mode {
        SampleMode::Grid => (
            sample_indices(total_rows, max_rows.max(1)),
            sample_indices(total_cols, max_cols.max(1)),
        ),
        SampleMode::Edges { row_tail, col_tail } => (
            edge_indices(total_rows, max_rows.max(1), row_tail),
            edge_indices(total_cols, max_cols.max(1), col_tail),
        ),
        SampleMode::Window { row_off, col_off } => (
            window_indices(total_rows, max_rows.max(1), row_off),
            window_indices(total_cols, max_cols.max(1), col_off),
        ),
    };

    let (values, raw) = read_sampled(reader, t, total_cols, base, &rows, &cols, view)?;

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
        raw,
        min,
        max,
        total_rows,
        total_cols,
        slices,
        slice,
        view,
        overridable,
        mode,
        display_shape: view.logical_shape(shape, &t.dtype),
    })
}

/// Up to `k` *contiguous* indices in `0..n`, starting at `off` but clamped so
/// the window never runs past the end (and pinned to `0` when the whole axis
/// fits). The first returned index is the clamped offset, which the caller
/// reads back to keep its stored pan position in bounds.
fn window_indices(n: usize, k: usize, off: usize) -> Vec<usize> {
    let k = k.min(n);
    if k == 0 {
        return Vec::new();
    }
    let start = off.min(n - k);
    (start..start + k).collect()
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

/// The first and last indices of `0..n` (so padding at either end is visible),
/// filling the available space. The total shown is `2 * ((max - 1) / 2)` (the
/// screen budget, leaving one slot for the "⋯" / "⋮" gap the UI draws between
/// the halves). `tail_frac` splits that budget between the head (first) and
/// tail (last): `0.0` is all-first, `1.0` is all-last, `0.5` is balanced.
/// Returns all of `0..n` when the budget already covers it (no gap).
/// The total number of indices the edges view shows for one axis with `max`
/// cells available (leaving one slot for the "⋯" / "⋮" gap). Exposed so the UI
/// can size an arrow-key step to exactly one index (`1 / edge_total`).
pub fn edge_total(max: usize) -> usize {
    2 * (max.saturating_sub(1) / 2).max(1)
}

fn edge_indices(n: usize, max: usize, tail_frac: f32) -> Vec<usize> {
    let total = edge_total(max);
    if n <= total {
        return (0..n).collect();
    }
    let tail = ((tail_frac.clamp(0.0, 1.0) * total as f32).round() as usize).min(total);
    let head = total - tail;
    // A window entirely at one end is contiguous (no gap); otherwise the head
    // and tail blocks are disjoint (`head + tail = total < n`) and the UI marks
    // the skipped middle with a gap.
    if head == 0 {
        return ((n - tail)..n).collect();
    }
    if tail == 0 {
        return (0..head).collect();
    }
    let mut idx: Vec<usize> = (0..head).collect();
    idx.extend((n - tail)..n);
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

/// A primitive numeric dtype, parsed once from its string label so the hot scan
/// loop dispatches on a cheap enum instead of matching the `&str` per element.
#[derive(Clone, Copy)]
enum Prim {
    F64,
    F32,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
}

/// Parse a dtype label into a [`Prim`], or `None` for unknown labels.
fn parse_prim(dtype: &str) -> Option<Prim> {
    Some(match dtype {
        "F64" => Prim::F64,
        "F32" => Prim::F32,
        "F16" => Prim::F16,
        "BF16" => Prim::BF16,
        "I64" => Prim::I64,
        "I32" => Prim::I32,
        "I16" => Prim::I16,
        "I8" => Prim::I8,
        "U64" => Prim::U64,
        "U32" => Prim::U32,
        "U16" => Prim::U16,
        "U8" | "BOOL" => Prim::U8,
        _ => return None,
    })
}

/// Decode `item_size` little-endian bytes as `p` into an `f64`.
fn decode_prim(p: Prim, b: &[u8]) -> f64 {
    match p {
        Prim::F64 => f64::from_le_bytes(b.try_into().unwrap()),
        Prim::F32 => f32::from_le_bytes(b.try_into().unwrap()) as f64,
        Prim::F16 => f16_to_f64(u16::from_le_bytes(b.try_into().unwrap())),
        Prim::BF16 => bf16_to_f64(u16::from_le_bytes(b.try_into().unwrap())),
        Prim::I64 => i64::from_le_bytes(b.try_into().unwrap()) as f64,
        Prim::I32 => i32::from_le_bytes(b.try_into().unwrap()) as f64,
        Prim::I16 => i16::from_le_bytes(b.try_into().unwrap()) as f64,
        Prim::I8 => (b[0] as i8) as f64,
        Prim::U64 => u64::from_le_bytes(b.try_into().unwrap()) as f64,
        Prim::U32 => u32::from_le_bytes(b.try_into().unwrap()) as f64,
        Prim::U16 => u16::from_le_bytes(b.try_into().unwrap()) as f64,
        Prim::U8 => b[0] as f64,
    }
}

/// Decode `item_size(dtype)` little-endian bytes into an `f64`.
fn decode(dtype: &str, b: &[u8]) -> f64 {
    parse_prim(dtype).map_or(f64::NAN, |p| decode_prim(p, b))
}

/// Decode sub-element `sub` of a stored container `bytes` under `view`. For
/// `Stored`/`As` this decodes the whole container; for the 4-bit views it
/// extracts nibble `sub` of the little-endian container, sign-extending for the
/// signed view.
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
    let nibble = ((container >> (sub * 4)) & 0xF) as i64;
    if view.is_signed() && nibble >= 8 {
        (nibble - 16) as f64
    } else {
        nibble as f64
    }
}

/// The raw stored bit pattern of sub-element `sub` of container `bytes` under
/// `view`, for hex/octal/binary display. For `Stored`/`As` it is the whole
/// little-endian container (the byte reinterpretation `As` applies doesn't
/// change the bits); for the 4-bit views it is nibble `sub` (the raw
/// two's-complement pattern, *not* sign-extended).
fn raw_bits(view: ViewDtype, bytes: &[u8], sub: usize) -> RawBits {
    let mut container: u64 = 0;
    for (i, &b) in bytes.iter().take(8).enumerate() {
        container |= (b as u64) << (8 * i);
    }
    match view {
        ViewDtype::Stored | ViewDtype::As(_) => RawBits {
            bits: container,
            width: (bytes.len().min(8) * 8) as u8,
        },
        _ => RawBits {
            bits: (container >> (sub * 4)) & 0xF,
            width: 4,
        },
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

/// Build a per-container decoder for `(view, dtype)`, resolving the `&str` dtype
/// dispatch once up front. The returned closure maps a container's bytes and a
/// nibble index to an `f64`; it runs once per logical value in the hot scan
/// loop (billions of times), so it must avoid string matching.
fn view_decoder(view: ViewDtype, dtype: &str) -> impl Fn(&[u8], usize) -> f64 {
    // For Stored / same-width `As`, decode the whole container as this primitive.
    let prim = match view {
        ViewDtype::As(dt) => parse_prim(dt),
        _ => parse_prim(dtype),
    };
    let signed = view.is_signed();
    move |bytes: &[u8], sub: usize| match view {
        ViewDtype::Stored | ViewDtype::As(_) => prim.map_or(f64::NAN, |p| decode_prim(p, bytes)),
        _ => {
            // 4-bit views: pull nibble `sub` from the little-endian container.
            let mut container: u64 = 0;
            for (i, &b) in bytes.iter().take(8).enumerate() {
                container |= (b as u64) << (8 * i);
            }
            let nibble = ((container >> (sub * 4)) & 0xF) as i64;
            if signed && nibble >= 8 {
                (nibble - 16) as f64
            } else {
                nibble as f64
            }
        }
    }
}

/// Reduce a flat little-endian byte buffer of `item`-byte containers into an
/// [`Acc`] under `view`, decoding every logical value (a packed view yields
/// several per container). Parallel over chunks; shared by the safetensors and
/// HDF5 scanners so a dtype reinterpretation means the same thing in both.
fn reduce_view_bytes(bytes: &[u8], item: usize, view: ViewDtype, dtype: &str) -> Acc {
    let packing = view.packing(item);
    let decode = view_decoder(view, dtype);
    bytes
        .par_chunks(item * STATS_CHUNK)
        .map(|chunk| {
            let mut a = Acc::ID;
            for container in chunk.chunks_exact(item) {
                for sub in 0..packing {
                    a.push(decode(container, sub));
                }
            }
            a
        })
        .reduce(|| Acc::ID, Acc::merge)
}

/// Largest block, in elements, read at once when scanning a contiguous tensor
/// (the default reader) for stats — keeps peak memory bounded regardless of size.
const STATS_BLOCK_ELEMS: usize = 16 << 20; // ≈16M elements

/// Target tile size, in elements, for the chunk-aligned HDF5 stats scan. Small
/// on purpose: each tile read holds libhdf5's lock only briefly, so a concurrent
/// foreground read stays responsive while a background scan runs. Each chunk is
/// still decompressed once, so the total scan work is unchanged.
#[cfg(feature = "hdf5")]
const STATS_TILE_ELEMS: usize = 2 << 20; // ≈2M elements

/// A chunk-aligned tile shape whose element count stays within `budget`, growing
/// inner (faster-varying) dimensions first for read contiguity. Each extent is a
/// multiple of the chunk extent (clamped to the shape), so tiles land on the
/// chunk grid and libhdf5 decompresses each chunk exactly once. Never smaller
/// than a single chunk (the smallest aligned read), even if that exceeds budget.
#[cfg(feature = "hdf5")]
fn stats_tile_shape(shape: &[usize], chunk: &[usize], budget: usize) -> Vec<usize> {
    let n = shape.len();
    // Start at one chunk per dimension (clamped to the shape).
    let mut tile: Vec<usize> = (0..n)
        .map(|d| chunk[d].max(1).min(shape[d].max(1)))
        .collect();
    // Grow inner dimensions first while staying within the element budget.
    for d in (0..n).rev() {
        if tile[d] >= shape[d] {
            continue; // already spans the whole dimension
        }
        let others: usize = tile
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != d)
            .map(|(_, &t)| t)
            .product::<usize>()
            .max(1);
        let c = chunk[d].max(1);
        // Largest chunk-multiple for this dimension that keeps `tile` ≤ budget.
        let grown = ((budget / others / c).max(1) * c).min(shape[d]);
        tile[d] = grown;
        if grown < shape[d] {
            break; // this dimension is the frontier; leave outer dims at one chunk
        }
    }
    tile
}

/// Format-agnostic access to one tensor's stored bytes. An implementation opens
/// its backing store once (a memory-mapped safetensors file, an HDF5 dataset, …)
/// and serves the raw stored containers — little-endian, row-major, exactly as
/// [`decode`] / [`decode_view`] expect. The sampling preview and the statistics
/// scan are written once against this trait, so supporting a new format (e.g.
/// remote/S3 shards) is just another implementation.
pub(crate) trait TensorReader {
    /// The stored shape.
    fn shape(&self) -> &[usize];

    /// Read the axis-aligned region selected by `ranges` (one half-open range
    /// per stored dimension, empty for a 0-D scalar) as a flat row-major
    /// little-endian buffer of stored containers.
    fn read_region(&self, ranges: &[Range<usize>]) -> Result<Vec<u8>, String>;

    /// Read several regions, one buffer each. The default reads them
    /// independently; a format may override to fetch one enclosing block when
    /// that is cheaper (e.g. to avoid re-decompressing shared HDF5 chunks).
    fn read_regions(&self, regions: &[Vec<Range<usize>>]) -> Result<Vec<Vec<u8>>, String> {
        regions.iter().map(|r| self.read_region(r)).collect()
    }

    /// Scan the whole tensor, handing each bounded block of stored bytes to `f`
    /// in order (used by the statistics pass). The default streams the dataset
    /// in row-blocks via [`read_region`]; formats override for a zero-copy scan.
    /// `f` returns [`ControlFlow::Break`] to stop early (e.g. on cancellation).
    fn fold_blocks(&self, f: &mut dyn FnMut(&[u8]) -> ControlFlow<()>) -> Result<(), String> {
        let shape = self.shape();
        if shape.is_empty() {
            let bytes = self.read_region(&[])?;
            let _ = f(&bytes);
            return Ok(());
        }
        let outer = shape[0];
        let inner: usize = shape[1..].iter().product::<usize>().max(1);
        let block = (STATS_BLOCK_ELEMS / inner).max(1);
        let mut i = 0;
        while i < outer {
            let hi = (i + block).min(outer);
            let mut ranges: Vec<Range<usize>> = Vec::with_capacity(shape.len());
            ranges.push(i..hi);
            ranges.extend(shape[1..].iter().map(|&d| 0..d));
            let bytes = self.read_region(&ranges)?;
            if f(&bytes).is_break() {
                break;
            }
            i = hi;
        }
        Ok(())
    }
}

/// Open the right [`TensorReader`] for a tensor, dispatching by file extension.
pub(crate) fn open_reader(t: &TensorInfo) -> Result<Box<dyn TensorReader>, String> {
    let ext = std::path::Path::new(&t.source_path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    match ext {
        "safetensors" => Ok(Box::new(BlobReader::open_safetensors(t)?)),
        "npy" => Ok(Box::new(BlobReader::open_npy(t)?)),
        "npz" => Ok(Box::new(BlobReader::open_npz(t)?)),
        #[cfg(feature = "hdf5")]
        "h5" | "hdf5" => Ok(Box::new(Hdf5Reader::open(t)?)),
        _ => Err("reading tensor data is not supported for this format".to_string()),
    }
}

/// Decompose a flat row-major container index into per-dimension indices.
fn unravel(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut idx = vec![0usize; shape.len()];
    for d in (0..shape.len()).rev() {
        idx[d] = flat % shape[d];
        flat /= shape[d];
    }
    idx
}

/// The axis-aligned region covering containers `first..=last`, which must lie
/// within a single innermost row (so every dimension but the last is a
/// singleton). Empty `shape` (0-D) yields an empty selection.
fn region_for_span(shape: &[usize], first: usize, last: usize) -> Vec<Range<usize>> {
    if shape.is_empty() {
        return Vec::new();
    }
    let lo = unravel(first, shape);
    let hi = unravel(last, shape);
    (0..shape.len())
        .map(|d| {
            if d + 1 == shape.len() {
                lo[d]..hi[d] + 1
            } else {
                lo[d]..lo[d] + 1
            }
        })
        .collect()
}

/// Copy the sub-region `want` out of `src`, a row-major buffer of `item`-byte
/// containers laid out over `src_ranges` (same rank, `want` ⊆ `src_ranges`;
/// both use absolute indices). Used to slice individual rows out of a larger
/// block read.
fn gather_region(
    src: &[u8],
    src_ranges: &[Range<usize>],
    want: &[Range<usize>],
    item: usize,
) -> Vec<u8> {
    let total: usize = want.iter().map(|r| r.len()).product();
    let n = src_ranges.len();
    if n == 0 {
        return src[..total * item].to_vec();
    }
    // Row-major container strides of the source layout.
    let mut stride = vec![1usize; n];
    for d in (0..n - 1).rev() {
        stride[d] = stride[d + 1] * src_ranges[d + 1].len();
    }
    let last = n - 1;
    let span = want[last].len();
    let mut out = Vec::with_capacity(total * item);
    // Odometer over `want`'s leading dimensions; copy the contiguous last-dim
    // span for each combination.
    let mut idx: Vec<usize> = want[..last].iter().map(|r| r.start).collect();
    loop {
        let mut off = (want[last].start - src_ranges[last].start) * stride[last];
        for d in 0..last {
            off += (idx[d] - src_ranges[d].start) * stride[d];
        }
        let bo = off * item;
        out.extend_from_slice(&src[bo..bo + span * item]);
        if last == 0 {
            break;
        }
        let mut d = last;
        loop {
            d -= 1;
            idx[d] += 1;
            if idx[d] < want[d].end {
                break;
            }
            idx[d] = want[d].start;
            if d == 0 {
                return out;
            }
        }
    }
    out
}

/// Compute exact statistics over the whole tensor under `view`, scanning every
/// element once and decoding in parallel. Works the same for any backing format
/// via [`TensorReader`]; a non-`Stored` view reinterprets the raw stored bytes.
pub fn tensor_stats(
    t: &TensorInfo,
    view: ViewDtype,
    cancel: &AtomicBool,
    pause: &AtomicBool,
    progress: Option<&AtomicUsize>,
) -> Result<Stats, String> {
    let item = item_size(&t.dtype).ok_or_else(|| format!("unsupported dtype: {}", t.dtype))?;
    let started = std::time::Instant::now();
    let reader = open_reader(t)?;
    let mut acc = Acc::ID;
    reader.fold_blocks(&mut |bytes| {
        acc = Acc::merge(acc, reduce_view_bytes(bytes, item, view, &t.dtype));
        // Report bytes scanned so the caller can show a progress bar. Summed over
        // every block this equals the tensor's stored byte size (`size_bytes`).
        if let Some(p) = progress {
            p.fetch_add(bytes.len(), Ordering::Relaxed);
        }
        // Wait here while paused — this is between block reads, so the backing
        // file's lock (libhdf5 serialises all access) is released, letting a
        // foreground read such as the dtype menu's live preview run without
        // contending; the scan resumes from this block when unpaused. Cancel
        // takes priority so a paused scan still tears down promptly.
        while pause.load(Ordering::Relaxed) && !cancel.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(8));
        }
        // Stop between blocks when the caller cancels (e.g. leaving the view).
        if cancel.load(Ordering::Relaxed) {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    })?;
    if cancel.load(Ordering::Relaxed) {
        return Err("cancelled".to_string());
    }
    let mut stats = acc.finish();
    stats.elapsed = started.elapsed();
    Ok(stats)
}

/// How a value histogram's bins are laid out.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HistBins {
    /// Integer bins: bin `i` covers the `step` integers starting at
    /// `start + i * step`. `step == 1` gives one bin per value; a larger step
    /// groups a wide integer range into whole-number buckets (never fractional).
    IntBins { start: i64, step: i64 },
    /// Equal-width bins spanning `[lo, hi]` (the last bin includes `hi`).
    Range { lo: f64, hi: f64 },
}

/// A whole-tensor value histogram (a snapshot of the running scan, or final).
#[derive(Clone, Debug)]
pub struct Histogram {
    pub bins: HistBins,
    /// One count per bin.
    pub counts: Vec<u64>,
    /// Finite elements binned so far.
    pub total: u64,
    /// Non-finite elements (NaN / ±Inf), not binned.
    pub nonfinite: u64,
}

/// Live bin counts shared between the scan worker and the UI, so the histogram
/// can be drawn filling in as it forms. The bin layout is fixed up front.
pub struct HistShared {
    pub counts: Vec<AtomicU64>,
    pub total: AtomicU64,
    pub nonfinite: AtomicU64,
}

impl HistShared {
    pub fn new(n: usize) -> Self {
        HistShared {
            counts: (0..n).map(|_| AtomicU64::new(0)).collect(),
            total: AtomicU64::new(0),
            nonfinite: AtomicU64::new(0),
        }
    }

    /// A consistent-enough snapshot of the running counts for one rendered frame.
    pub fn snapshot(&self, bins: HistBins) -> Histogram {
        Histogram {
            bins,
            counts: self
                .counts
                .iter()
                .map(|c| c.load(Ordering::Relaxed))
                .collect(),
            total: self.total.load(Ordering::Relaxed),
            nonfinite: self.nonfinite.load(Ordering::Relaxed),
        }
    }
}

impl ViewDtype {
    /// The full inclusive integer value range when it is small and intrinsic to
    /// the view (the 4-bit views), so the histogram shows every possible value —
    /// including ones absent from the data. `None` otherwise (use the data range).
    fn small_int_span(self) -> Option<(i64, i64)> {
        match self {
            ViewDtype::U4 => Some((0, 15)),
            ViewDtype::I4 => Some((-8, 7)),
            _ => None,
        }
    }
}

/// Max distinct integer values shown as one bin each; above this the histogram
/// uses equal-width range bins instead.
const MAX_VALUE_BINS: usize = 64;
/// Number of equal-width bins for a range histogram (floats / wide integers).
const RANGE_BINS: usize = 40;

/// Decide a histogram's bin layout for `(view, dtype)`. `range` is the tensor's
/// `(min, max)` from its stats — needed for range bins and data-range integer
/// bins, but not for the 4-bit views (whose range is intrinsic). `bins` requests
/// a specific bucket count (the `b` key / `--bins`); `None` uses the defaults.
/// Returns the layout and bin count, or `None` when a range is required but not
/// yet known.
pub fn histogram_bins(
    view: ViewDtype,
    dtype: &str,
    range: Option<(f64, f64)>,
    bins: Option<usize>,
) -> Option<(HistBins, usize)> {
    // A requested bucket count overrides the defaults: it sets the float bin
    // count and the wide-integer grouping target, and also caps the per-value
    // threshold (you can't show more buckets than there are distinct values).
    let target = bins.unwrap_or(RANGE_BINS).max(1) as u64;
    if view.is_integer(dtype) {
        // 4-bit views: every possible value, even absent ones — no range needed,
        // and the count is intrinsic, so a requested count doesn't apply.
        if let Some((lo, hi)) = view.small_int_span() {
            return Some((
                HistBins::IntBins { start: lo, step: 1 },
                (hi - lo + 1) as usize,
            ));
        }
        let (min, max) = range?;
        let (lo, hi) = (min.floor() as i64, max.ceil() as i64);
        if hi < lo {
            return Some((HistBins::IntBins { start: lo, step: 1 }, 1));
        }
        let distinct = (hi - lo + 1) as u64;
        // One bin per value when few enough — capped by any requested count.
        let per_value_cap = bins.unwrap_or(MAX_VALUE_BINS).max(1) as u64;
        if distinct <= per_value_cap {
            return Some((HistBins::IntBins { start: lo, step: 1 }, distinct as usize));
        }
        // Otherwise group into ~`target` whole-number-width buckets, so the
        // edges stay integers rather than the fractional ones a float range
        // would produce.
        let step = distinct.div_ceil(target) as i64;
        let n = distinct.div_ceil(step as u64) as usize;
        return Some((HistBins::IntBins { start: lo, step }, n));
    }
    match range? {
        (min, max) if min < max => Some((HistBins::Range { lo: min, hi: max }, target as usize)),
        (min, max) => Some((HistBins::Range { lo: min, hi: max }, 1)), // all-equal: one bin
    }
}

/// The bin a value falls in (clamped to the end bins for out-of-range values).
fn bin_index(bins: HistBins, n: usize, v: f64) -> usize {
    let last = n.saturating_sub(1) as i64;
    match bins {
        HistBins::IntBins { start, step } => {
            ((v.round() as i64 - start).div_euclid(step.max(1))).clamp(0, last) as usize
        }
        HistBins::Range { lo, hi } => {
            if hi <= lo {
                return 0;
            }
            let frac = (v - lo) / (hi - lo);
            ((frac * n as f64).floor() as i64).clamp(0, last) as usize
        }
    }
}

/// Reduce a flat byte buffer into per-bin counts (plus finite/non-finite totals)
/// under `view`, in parallel. Mirrors [`reduce_view_bytes`] but bins instead of
/// folding moments.
fn reduce_view_hist(
    bytes: &[u8],
    item: usize,
    view: ViewDtype,
    dtype: &str,
    bins: HistBins,
    n: usize,
) -> (Vec<u64>, u64, u64) {
    let packing = view.packing(item);
    let decode = view_decoder(view, dtype);
    bytes
        .par_chunks(item * STATS_CHUNK)
        .map(|chunk| {
            let mut counts = vec![0u64; n];
            let (mut total, mut nonfinite) = (0u64, 0u64);
            for container in chunk.chunks_exact(item) {
                for sub in 0..packing {
                    let v = decode(container, sub);
                    if v.is_finite() {
                        counts[bin_index(bins, n, v)] += 1;
                        total += 1;
                    } else {
                        nonfinite += 1;
                    }
                }
            }
            (counts, total, nonfinite)
        })
        .reduce(
            || (vec![0u64; n], 0, 0),
            |mut a, b| {
                for (x, y) in a.0.iter_mut().zip(&b.0) {
                    *x += *y;
                }
                (a.0, a.1 + b.1, a.2 + b.2)
            },
        )
}

/// Scan the whole tensor and accumulate its value histogram into `shared` (whose
/// bin count must match `n`), so the UI can draw the bins filling in as the scan
/// runs. `bins` is the fixed layout. Honours `pause`/`cancel` like [`tensor_stats`].
#[allow(clippy::too_many_arguments)] // a scan driver; the params are all distinct
pub fn tensor_histogram_into(
    t: &TensorInfo,
    view: ViewDtype,
    bins: HistBins,
    n: usize,
    shared: &HistShared,
    cancel: &AtomicBool,
    pause: &AtomicBool,
    progress: Option<&AtomicUsize>,
) -> Result<(), String> {
    let item = item_size(&t.dtype).ok_or_else(|| format!("unsupported dtype: {}", t.dtype))?;
    let reader = open_reader(t)?;
    reader.fold_blocks(&mut |bytes| {
        let (counts, total, nonfinite) = reduce_view_hist(bytes, item, view, &t.dtype, bins, n);
        for (slot, c) in shared.counts.iter().zip(&counts) {
            slot.fetch_add(*c, Ordering::Relaxed);
        }
        shared.total.fetch_add(total, Ordering::Relaxed);
        shared.nonfinite.fetch_add(nonfinite, Ordering::Relaxed);
        if let Some(p) = progress {
            p.fetch_add(bytes.len(), Ordering::Relaxed);
        }
        while pause.load(Ordering::Relaxed) && !cancel.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(8));
        }
        if cancel.load(Ordering::Relaxed) {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    })?;
    if cancel.load(Ordering::Relaxed) {
        return Err("cancelled".to_string());
    }
    Ok(())
}

/// Decoded values and the parallel raw stored bits for a sampled grid, each
/// indexed `[row][col]`.
type SampledGrid = (Vec<Vec<f64>>, Vec<Vec<RawBits>>);

/// Sample the grid of `rows × cols` logical values from `reader`, decoding under
/// `view`. Indices are logical: under a packed view a logical element `flat`
/// lives in container `flat / packing` at nibble `flat % packing`. Reads only
/// each sampled row's column span (never the whole tensor), so it scales to any
/// size and any format.
fn read_sampled(
    reader: &dyn TensorReader,
    t: &TensorInfo,
    total_cols: usize,
    base: usize,
    rows: &[usize],
    cols: &[usize],
    view: ViewDtype,
) -> Result<SampledGrid, String> {
    let dtype = t.dtype.as_str();
    let item = item_size(dtype).ok_or_else(|| format!("unsupported dtype: {dtype}"))?;
    let shape = reader.shape().to_vec();
    let packing = view.packing(item);
    // Elements per stored innermost row — used to detect when a sampled row's
    // span crosses stored rows (only possible under a shape override).
    let inner: usize = if shape.len() > 1 {
        shape[1..].iter().product::<usize>().max(1)
    } else {
        1
    };
    let first_col = *cols.first().unwrap();
    let last_col = *cols.last().unwrap();
    let container_for = |row_base: usize, col: usize| (row_base + col) / packing;

    // Per sampled row: the region to read, and the flat container index its
    // buffer starts at. A row's column span is normally within one stored
    // innermost row, so we read just that span; a shape override can make it
    // straddle stored rows, so then we read the full stored rows it covers (the
    // buffer starts at the first of them). The flat container range is the same
    // contiguous data either way — only the read selection differs.
    let mut regions: Vec<Vec<Range<usize>>> = Vec::with_capacity(rows.len());
    let mut starts: Vec<usize> = Vec::with_capacity(rows.len());
    for &r in rows {
        let row_base = base + r * total_cols;
        let first = container_for(row_base, first_col);
        let last = container_for(row_base, last_col);
        if shape.len() <= 1 || first / inner == last / inner {
            regions.push(region_for_span(&shape, first, last));
            starts.push(first);
        } else {
            let (r0, r1) = (first / inner, last / inner);
            let mut region = Vec::with_capacity(shape.len());
            region.push(r0..r1 + 1);
            region.extend(shape[1..].iter().map(|&d| 0..d));
            regions.push(region);
            starts.push(r0 * inner);
        }
    }
    let bufs = reader.read_regions(&regions)?;

    let width = view.bit_width(dtype) as u8;
    let mut values = Vec::with_capacity(rows.len());
    let mut raws = Vec::with_capacity(rows.len());
    for ((&r, &start), buf) in rows.iter().zip(&starts).zip(bufs) {
        let row_base = base + r * total_cols;
        let mut vrow = Vec::with_capacity(cols.len());
        let mut rrow = Vec::with_capacity(cols.len());
        for &c in cols {
            let flat = row_base + c;
            let off = (flat / packing - start) * item;
            match buf.get(off..off + item) {
                Some(cont) => {
                    vrow.push(decode_view(view, dtype, cont, flat % packing));
                    rrow.push(raw_bits(view, cont, flat % packing));
                }
                None => {
                    vrow.push(f64::NAN);
                    rrow.push(RawBits { bits: 0, width });
                }
            }
        }
        values.push(vrow);
        raws.push(rrow);
    }
    Ok((values, raws))
}

/// Memory-mapped reader for a safetensors tensor.
/// Backing bytes for a tensor stored as one contiguous little-endian row-major
/// blob: either a memory-mapped file (safetensors, `.npy`) or an owned buffer (a
/// decompressed `.npz` entry).
enum Backing {
    Mmap(memmap2::Mmap),
    Owned(Vec<u8>),
}

impl Backing {
    fn bytes(&self) -> &[u8] {
        match self {
            Backing::Mmap(m) => &m[..],
            Backing::Owned(v) => v,
        }
    }
}

/// Reader for a tensor that is one contiguous little-endian row-major blob.
/// Serves safetensors, NumPy `.npy`, and (decompressed) `.npz` entries; the
/// region/scan logic is identical once the backing bytes and data range are set.
struct BlobReader {
    backing: Backing,
    data_start: usize,
    data_end: usize,
    shape: Vec<usize>,
    item: usize,
}

impl BlobReader {
    /// safetensors: `ByteRange` is relative to the data blob, which starts after
    /// the 8-byte header length and the JSON header.
    fn open_safetensors(t: &TensorInfo) -> Result<Self, String> {
        let Layout::ByteRange { start, end } = t.layout else {
            return Err("tensor data location is unknown".to_string());
        };
        let item = item_size(&t.dtype).ok_or_else(|| format!("unsupported dtype: {}", t.dtype))?;
        let mmap = mmap_file(&t.source_path)?;
        let header_len =
            u64::from_le_bytes(mmap.get(0..8).ok_or("file too small")?.try_into().unwrap());
        let data_start = (8 + header_len + start) as usize;
        let data_end = (8 + header_len + end) as usize;
        Self::mmapped(mmap, data_start, data_end, t, item)
    }

    /// `.npy`: `ByteRange` holds absolute file offsets — the raw data directly
    /// follows the header, so it can be mapped in place.
    fn open_npy(t: &TensorInfo) -> Result<Self, String> {
        let Layout::ByteRange { start, end } = t.layout else {
            return Err("tensor data location is unknown".to_string());
        };
        let item = item_size(&t.dtype).ok_or_else(|| format!("unsupported dtype: {}", t.dtype))?;
        let mmap = mmap_file(&t.source_path)?;
        Self::mmapped(mmap, start as usize, end as usize, t, item)
    }

    /// `.npz`: decompress the entry named `<tensor>.npy` into memory and serve
    /// from it (compressed entries can't be mapped in place).
    fn open_npz(t: &TensorInfo) -> Result<Self, String> {
        use std::io::Read;
        let item = item_size(&t.dtype).ok_or_else(|| format!("unsupported dtype: {}", t.dtype))?;
        let file = std::fs::File::open(&t.source_path).map_err(|e| e.to_string())?;
        let mut zip = zip::ZipArchive::new(file).map_err(|e| e.to_string())?;
        let entry_name = format!("{}.npy", t.name);
        let mut entry = zip
            .by_name(&entry_name)
            .map_err(|e| format!("{entry_name}: {e}"))?;
        let mut bytes = Vec::with_capacity(entry.size() as usize);
        entry.read_to_end(&mut bytes).map_err(|e| e.to_string())?;
        let header = crate::npy::parse_header(&mut std::io::Cursor::new(&bytes))?;
        let data_end = bytes.len();
        Ok(Self {
            backing: Backing::Owned(bytes),
            data_start: header.data_offset,
            data_end,
            shape: t.shape.clone(),
            item,
        })
    }

    fn mmapped(
        mmap: memmap2::Mmap,
        data_start: usize,
        data_end: usize,
        t: &TensorInfo,
        item: usize,
    ) -> Result<Self, String> {
        if data_end > mmap.len() || data_start > data_end {
            return Err("tensor data range is out of bounds".to_string());
        }
        Ok(Self {
            backing: Backing::Mmap(mmap),
            data_start,
            data_end,
            shape: t.shape.clone(),
            item,
        })
    }

    fn blob(&self) -> &[u8] {
        &self.backing.bytes()[self.data_start..self.data_end]
    }
}

/// Memory-map a file read-only. SAFETY: read-only inspection; we accept that a
/// concurrent external write could change the mapping (standard mmap tradeoff).
fn mmap_file(path: &str) -> Result<memmap2::Mmap, String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    unsafe { memmap2::Mmap::map(&file).map_err(|e| e.to_string()) }
}

impl TensorReader for BlobReader {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn read_region(&self, ranges: &[Range<usize>]) -> Result<Vec<u8>, String> {
        let full: Vec<Range<usize>> = self.shape.iter().map(|&d| 0..d).collect();
        Ok(gather_region(self.blob(), &full, ranges, self.item))
    }

    /// Zero-copy full scan: the tensor's bytes are already contiguous, so hand
    /// them straight to `f` — but in element-aligned chunks rather than one giant
    /// call, so the caller's per-block work (progress bar, pause, cancel) runs
    /// incrementally instead of only after the whole tensor has been reduced.
    /// Each chunk is still a borrowed sub-slice (no copy); the reducer itself
    /// parallelises within a chunk, so chunking this coarsely costs nothing.
    fn fold_blocks(&self, f: &mut dyn FnMut(&[u8]) -> ControlFlow<()>) -> Result<(), String> {
        let blob = self.blob();
        let chunk = STATS_BLOCK_ELEMS * self.item; // element-aligned by construction
        let mut off = 0;
        while off < blob.len() {
            let hi = (off + chunk).min(blob.len());
            if f(&blob[off..hi]).is_break() {
                break;
            }
            off = hi;
        }
        Ok(())
    }
}

/// Read an HDF5 dataset selection and hand its bytes (little-endian, the order
/// `decode`/`decode_view` expect) to `f`. The memory type matches the stored
/// dtype so libhdf5 copies the bits through without a lossy numeric conversion
/// — this is what lets the dtype-override views (same-width reinterpretation,
/// 4-bit nibbles) work on HDF5 too. When the read is contiguous on a
/// little-endian host (the common case) the bytes are borrowed in place with no
/// copy; otherwise each element is serialised in row-major logical order.
#[cfg(feature = "hdf5")]
fn with_hdf5_block_bytes<R>(
    dataset: &hdf5_metno::Dataset,
    hyper: hdf5_metno::Hyperslab,
    dtype: &str,
    f: impl FnOnce(&[u8]) -> R,
) -> Result<R, String> {
    macro_rules! run {
        ($ty:ty) => {{
            let a = dataset
                .read_slice::<$ty, _, ndarray::IxDyn>(hyper)
                .map_err(|e| e.to_string())?;
            match a.as_slice() {
                // Contiguous + little-endian: the native bytes already match the
                // little-endian layout we decode, so reinterpret them in place.
                Some(s) if cfg!(target_endian = "little") => {
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            s.as_ptr() as *const u8,
                            std::mem::size_of_val(s),
                        )
                    };
                    f(bytes)
                }
                // Non-contiguous or big-endian: serialise to little-endian first.
                _ => {
                    let mut buf = Vec::with_capacity(a.len() * std::mem::size_of::<$ty>());
                    for v in a.iter() {
                        buf.extend_from_slice(&v.to_le_bytes());
                    }
                    f(&buf)
                }
            }
        }};
    }
    Ok(match dtype {
        "F64" => run!(f64),
        "F32" => run!(f32),
        "F16" => run!(half::f16),
        "I64" => run!(i64),
        "I32" => run!(i32),
        "I16" => run!(i16),
        "I8" => run!(i8),
        "U64" => run!(u64),
        "U32" => run!(u32),
        "U16" => run!(u16),
        "U8" | "BOOL" => run!(u8),
        other => return Err(format!("unsupported dtype: {other}")),
    })
}

/// Reader for an HDF5 dataset. Holds the open file/dataset so repeated region
/// reads (one per sampled row) don't reopen it.
#[cfg(feature = "hdf5")]
struct Hdf5Reader {
    // Kept only to hold the file open for the dataset's lifetime.
    _file: hdf5_metno::File,
    dataset: hdf5_metno::Dataset,
    shape: Vec<usize>,
    dtype: String,
    item: usize,
}

#[cfg(feature = "hdf5")]
impl Hdf5Reader {
    fn open(t: &TensorInfo) -> Result<Self, String> {
        let item = item_size(&t.dtype).ok_or_else(|| format!("unsupported dtype: {}", t.dtype))?;
        let file = hdf5_metno::File::open(&t.source_path).map_err(|e| e.to_string())?;
        // Ensure LZ4-compressed datasets are decodable (no-op after first call).
        crate::hdf5_lz4::register();
        crate::hdf5_zstd::register();
        let key = file
            .member_names()
            .map_err(|e| e.to_string())?
            .into_iter()
            .find(|k| crate::hdf5::percent_decode(k) == t.name)
            .ok_or_else(|| "dataset not found in file".to_string())?;
        let dataset = file.dataset(&key).map_err(|e| e.to_string())?;
        let shape = dataset.shape();
        Ok(Self {
            _file: file,
            dataset,
            shape,
            dtype: t.dtype.clone(),
            item,
        })
    }

    fn hyperslab(ranges: &[Range<usize>]) -> hdf5_metno::Hyperslab {
        use hdf5_metno::SliceOrIndex;
        hdf5_metno::Hyperslab::from(
            ranges
                .iter()
                .cloned()
                .map(SliceOrIndex::from)
                .collect::<Vec<_>>(),
        )
    }
}

#[cfg(feature = "hdf5")]
impl TensorReader for Hdf5Reader {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn read_region(&self, ranges: &[Range<usize>]) -> Result<Vec<u8>, String> {
        with_hdf5_block_bytes(
            &self.dataset,
            Self::hyperslab(ranges),
            &self.dtype,
            <[u8]>::to_vec,
        )
    }

    /// Fetch one enclosing block when it fits a memory budget: reading the
    /// bounding box of the sampled rows decompresses each overlapping chunk once
    /// rather than once per row. When the box is too large the tensor is huge,
    /// so the sampled rows are spread far apart and rarely share a chunk anyway,
    /// and per-row reads are fine.
    fn read_regions(&self, regions: &[Vec<Range<usize>>]) -> Result<Vec<Vec<u8>>, String> {
        const BUDGET_BYTES: usize = 256 << 20;
        if regions.len() < 2 {
            return regions.iter().map(|r| self.read_region(r)).collect();
        }
        let ndim = regions[0].len();
        let bbox: Vec<Range<usize>> = (0..ndim)
            .map(|d| {
                let lo = regions.iter().map(|r| r[d].start).min().unwrap_or(0);
                let hi = regions.iter().map(|r| r[d].end).max().unwrap_or(0);
                lo..hi
            })
            .collect();
        let bbox_elems: usize = bbox.iter().map(|r| r.len()).product();
        if bbox_elems.saturating_mul(self.item) > BUDGET_BYTES {
            return regions.iter().map(|r| self.read_region(r)).collect();
        }
        let buf = self.read_region(&bbox)?;
        Ok(regions
            .iter()
            .map(|r| gather_region(&buf, &bbox, r, self.item))
            .collect())
    }

    fn fold_blocks(&self, f: &mut dyn FnMut(&[u8]) -> ControlFlow<()>) -> Result<(), String> {
        let shape = &self.shape;
        if shape.is_empty() {
            return with_hdf5_block_bytes(&self.dataset, Self::hyperslab(&[]), &self.dtype, |b| {
                f(b)
            })
            .map(|_| ());
        }
        // Walk the tensor in chunk-aligned tiles bounded by a small element
        // budget. Aligning to the chunk grid means libhdf5 decompresses each
        // chunk exactly once (a tile that split a chunk would re-decompress it);
        // keeping the tiles small means the dataset lock — libhdf5 serialises all
        // access — is released between reads, so a concurrent foreground read
        // (e.g. re-sampling the data view while this scan runs in the background)
        // isn't stuck behind a big block. A contiguous dataset has no chunks, so
        // we tile on a unit grid (any split is free without compression).
        let ndim = shape.len();
        let chunk = self.dataset.chunk().unwrap_or_else(|| vec![1; ndim]);
        let tile = stats_tile_shape(shape, &chunk, STATS_TILE_ELEMS);
        let mut origin = vec![0usize; ndim];
        loop {
            let ranges: Vec<Range<usize>> = (0..ndim)
                .map(|d| origin[d]..(origin[d] + tile[d]).min(shape[d]))
                .collect();
            let flow =
                with_hdf5_block_bytes(&self.dataset, Self::hyperslab(&ranges), &self.dtype, |b| {
                    f(b)
                })?;
            if flow.is_break() {
                return Ok(());
            }
            // Advance the tile odometer, innermost dimension first.
            let mut d = ndim - 1;
            loop {
                origin[d] += tile[d];
                if origin[d] < shape[d] {
                    break;
                }
                origin[d] = 0;
                if d == 0 {
                    return Ok(());
                }
                d -= 1;
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Manual benchmark for the window-pan hot path: compares re-opening the
    /// reader on every pan (current `sample_tensor` behaviour) against reusing a
    /// single open reader. Run with:
    /// `cargo test --features hdf5 --release -- --ignored --nocapture window_pan_open_cost`
    #[cfg(feature = "hdf5")]
    #[test]
    #[ignore = "manual benchmark"]
    fn window_pan_open_cost() {
        use std::time::Instant;
        let dir = std::env::temp_dir().join("checkpoint_explorer_bench_pan");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("bench.h5");
        let _ = std::fs::remove_file(&path);
        let (r, c) = (1500usize, 1500usize);
        {
            let file = hdf5_metno::File::create(&path).unwrap();
            // Many small datasets so `member_names()` is realistically long, like
            // a real checkpoint with hundreds of tensors.
            for i in 0..400 {
                let _ = file
                    .new_dataset::<f32>()
                    .shape([4, 4])
                    .create(format!("layer.{i}.weight").as_str())
                    .unwrap();
            }
            // One big gzip-compressed, chunked tensor we pan over.
            let data: Vec<f32> = (0..r * c).map(|i| (i % 997) as f32).collect();
            let ds = file
                .new_dataset::<f32>()
                .shape([r, c])
                .chunk([128, 128])
                .deflate(4)
                .create("big.weight")
                .unwrap();
            ds.write_raw(&data).unwrap();
        }
        let t = TensorInfo {
            name: "big.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![r, c],
            size_bytes: r * c * 4,
            num_elements: r * c,
            storage: crate::tree::Storage::Unknown,
            source_path: path.to_string_lossy().into_owned(),
            layout: Layout::None,
        };

        const ITERS: usize = 60;
        // (A) open-only, to isolate File::open + member_names + dataset open.
        let t0 = Instant::now();
        for _ in 0..ITERS {
            let _r = open_reader(&t).unwrap();
        }
        let open_only = t0.elapsed();

        // (B) current behaviour: a full window sample per pan (reopens each time).
        let t0 = Instant::now();
        for i in 0..ITERS {
            let _ = sample_tensor(
                &t,
                40,
                40,
                0,
                ViewDtype::Stored,
                SampleMode::Window {
                    row_off: i,
                    col_off: 0,
                },
            )
            .unwrap();
        }
        let reopen_each = t0.elapsed();

        // (C) proposed behaviour: open once, read the window region each pan.
        let reader = open_reader(&t).unwrap();
        let shape = reader.shape().to_vec();
        let t0 = Instant::now();
        for i in 0..ITERS {
            let regions: Vec<Vec<Range<usize>>> =
                (i..i + 40).map(|row| vec![row..row + 1, 0..40]).collect();
            let _ = reader.read_regions(&regions).unwrap();
        }
        let reuse = t0.elapsed();
        let _ = shape;

        eprintln!("--- window pan, {ITERS} iters ---");
        eprintln!(
            "open-only (reopen each):   {open_only:?}  ({:?}/pan)",
            open_only / ITERS as u32
        );
        eprintln!(
            "sample_tensor (reopens):   {reopen_each:?}  ({:?}/pan)",
            reopen_each / ITERS as u32
        );
        eprintln!(
            "reused reader (read only): {reuse:?}  ({:?}/pan)",
            reuse / ITERS as u32
        );
        let _ = std::fs::remove_file(&path);
    }

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
    fn edge_indices_takes_first_and_last() {
        // Small enough to show whole (n <= total): no gap.
        assert_eq!(edge_indices(5, 100, 0.5), vec![0, 1, 2, 3, 4]);
        // Balanced (tail_frac = 0.5) fills the screen: total = 2*((max-1)/2),
        // split evenly. With max = 100 that's 49 first and 49 last of 1000.
        let e = edge_indices(1000, 100, 0.5);
        assert_eq!(e.len(), 2 * 49);
        assert_eq!(&e[..3], &[0, 1, 2]);
        assert_eq!(e[48], 48);
        assert_eq!(e[49], 951);
        assert_eq!(e.last(), Some(&999));
        // Tight budget: total = 2*((8-1)/2) = 6, balanced = 3 + 3.
        assert_eq!(edge_indices(100, 8, 0.5), vec![0, 1, 2, 97, 98, 99]);
        // All-tail (tail_frac = 1.0): only the last `total` indices, contiguous.
        assert_eq!(edge_indices(100, 8, 1.0), vec![94, 95, 96, 97, 98, 99]);
        // All-head (tail_frac = 0.0): only the first `total`, contiguous.
        assert_eq!(edge_indices(100, 8, 0.0), vec![0, 1, 2, 3, 4, 5]);
        // Biased toward the tail: fewer first, more last (still a gap).
        let b = edge_indices(100, 14, 0.75); // total = 12 -> 3 first, 9 last
        assert_eq!(&b[..3], &[0, 1, 2]);
        assert_eq!(b.len(), 12);
        assert_eq!(&b[3..], &[91, 92, 93, 94, 95, 96, 97, 98, 99]);
    }

    #[cfg(feature = "hdf5")]
    #[test]
    fn stats_tile_shape_is_chunk_aligned_and_bounded() {
        // Monolith case: fill the innermost dim fully, then one chunk of the next.
        assert_eq!(
            stats_tile_shape(&[128, 3088, 1, 2992], &[4, 97, 1, 187], 2 << 20),
            vec![4, 97, 1, 2992]
        );
        // Contiguous (unit chunks): inner dim fills, outer dim is the frontier.
        let c = stats_tile_shape(&[100, 50], &[1, 1], 1000);
        assert_eq!(c, vec![20, 50]);
        assert!(c.iter().product::<usize>() <= 1000);
        // A single chunk already over budget → tile is exactly one chunk.
        assert_eq!(stats_tile_shape(&[10, 10], &[10, 10], 4), vec![10, 10]);
    }

    #[test]
    fn window_indices_is_contiguous_and_clamped() {
        // Whole axis fits: the offset is ignored, starts at 0.
        assert_eq!(window_indices(5, 10, 3), vec![0, 1, 2, 3, 4]);
        // A k-wide window at an in-range offset is contiguous.
        assert_eq!(window_indices(100, 4, 10), vec![10, 11, 12, 13]);
        // Offset past the end clamps so the window still shows k indices ending
        // at n-1 (this is what keeps panning stuck-to-the-edge well-behaved).
        assert_eq!(window_indices(100, 4, 999), vec![96, 97, 98, 99]);
        // Exactly at the last valid offset.
        assert_eq!(window_indices(10, 3, 7), vec![7, 8, 9]);
    }

    #[test]
    fn samples_a_contiguous_window_at_an_offset() {
        use std::io::Write;
        let dir = std::env::temp_dir().join("checkpoint_explorer_sample_win");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("w.safetensors");
        // 6x6 f32, value[r][c] = r*6 + c.
        let header = br#"{"w":{"dtype":"F32","shape":[6,6],"data_offsets":[0,144]}}"#;
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(header).unwrap();
        for i in 0..36u32 {
            f.write_all(&(i as f32).to_le_bytes()).unwrap();
        }
        drop(f);

        let t = fixture(&path, "w", &[6, 6], (0, 144));
        // A 3x3 window with its corner at (2, 1) is the contiguous block
        // rows 2..5, cols 1..4 — no downsampling.
        let mode = SampleMode::Window {
            row_off: 2,
            col_off: 1,
        };
        let s = sample_tensor(&t, 3, 3, 0, ViewDtype::Stored, mode).unwrap();
        assert_eq!(s.rows, vec![2, 3, 4]);
        assert_eq!(s.cols, vec![1, 2, 3]);
        for (i, &r) in s.rows.iter().enumerate() {
            for (j, &c) in s.cols.iter().enumerate() {
                assert_eq!(s.values[i][j], (r * 6 + c) as f64);
            }
        }
        // An offset past the end clamps so the window stays in bounds (its first
        // index is the clamped corner the UI reads back).
        let clamped = sample_tensor(
            &t,
            3,
            3,
            0,
            ViewDtype::Stored,
            SampleMode::Window {
                row_off: 99,
                col_off: 99,
            },
        )
        .unwrap();
        assert_eq!(clamped.rows, vec![3, 4, 5]);
        assert_eq!(clamped.cols, vec![3, 4, 5]);
        let _ = std::fs::remove_file(&path);
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
        let s = sample_tensor(&t, 10, 10, 0, ViewDtype::Stored, SampleMode::Grid).unwrap();
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

    /// Build a `.npy` byte image (v1.0 header + row-major f32 data) and return
    /// it with the data offset, for the reader tests below.
    fn npy_image(shape_dict: &str, values: &[f32]) -> (Vec<u8>, usize) {
        let dict = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_dict}, }}");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"\x93NUMPY\x01\x00");
        bytes.extend_from_slice(&(dict.len() as u16).to_le_bytes());
        bytes.extend_from_slice(dict.as_bytes());
        let data_offset = bytes.len();
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        (bytes, data_offset)
    }

    #[test]
    fn samples_a_npy_array() {
        use std::io::Write;
        let dir = std::env::temp_dir().join("checkpoint_explorer_npy");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("w.npy");
        // 3x4 f32, value[r][c] = r*4 + c.
        let vals: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let (bytes, data_offset) = npy_image("(3, 4)", &vals);
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&bytes)
            .unwrap();

        let t = TensorInfo {
            name: "w".to_string(),
            dtype: "F32".to_string(),
            shape: vec![3, 4],
            size_bytes: 48,
            num_elements: 12,
            storage: crate::tree::Storage::Unknown,
            source_path: path.to_string_lossy().into_owned(),
            layout: Layout::ByteRange {
                start: data_offset as u64,
                end: bytes.len() as u64,
            },
        };
        let s = sample_tensor(&t, 10, 10, 0, ViewDtype::Stored, SampleMode::Grid).unwrap();
        assert_eq!((s.total_rows, s.total_cols), (3, 4));
        for (i, &r) in s.rows.iter().enumerate() {
            for (j, &c) in s.cols.iter().enumerate() {
                assert_eq!(s.values[i][j], (r * 4 + c) as f64);
            }
        }
        // Stats scan reads the same contiguous blob.
        let st = tensor_stats(
            &t,
            ViewDtype::Stored,
            &AtomicBool::new(false),
            &AtomicBool::new(false),
            None,
        )
        .unwrap();
        assert_eq!((st.count, st.min, st.max), (12, 0.0, 11.0));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn samples_a_deflated_npz_entry() {
        use std::io::Write;
        let dir = std::env::temp_dir().join("checkpoint_explorer_npz");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("a.npz");
        // A 2x3 f32 array stored as a DEFLATE-compressed `w.npy` entry, to
        // exercise the decompress-into-memory reader path.
        let vals: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let (npy, _) = npy_image("(2, 3)", &vals);
        let f = std::fs::File::create(&path).unwrap();
        let mut zw = zip::ZipWriter::new(f);
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        zw.start_file("w.npy", opts).unwrap();
        zw.write_all(&npy).unwrap();
        zw.finish().unwrap();

        let t = TensorInfo {
            name: "w".to_string(),
            dtype: "F32".to_string(),
            shape: vec![2, 3],
            size_bytes: 24,
            num_elements: 6,
            storage: crate::tree::Storage::Unknown,
            source_path: path.to_string_lossy().into_owned(),
            layout: Layout::None,
        };
        let s = sample_tensor(&t, 10, 10, 0, ViewDtype::Stored, SampleMode::Grid).unwrap();
        assert_eq!((s.total_rows, s.total_cols), (2, 3));
        for (i, &r) in s.rows.iter().enumerate() {
            for (j, &c) in s.cols.iter().enumerate() {
                assert_eq!(s.values[i][j], (r * 3 + c) as f64);
            }
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn reshape_across_stored_rows() {
        use std::io::Write;
        let dir = std::env::temp_dir().join("checkpoint_explorer_reshape");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("w.safetensors");
        // 4x5 f32 (value = flat index), reshaped to 5x4 — the override rows
        // straddle the stored (4,5) rows, which used to read back as NaN.
        let header = br#"{"w":{"dtype":"F32","shape":[4,5],"data_offsets":[0,80]}}"#;
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(header).unwrap();
        for i in 0..20u32 {
            f.write_all(&(i as f32).to_le_bytes()).unwrap();
        }
        drop(f);

        let t = fixture(&path, "w", &[4, 5], (0, 80));
        let reader = open_reader(&t).unwrap();
        let s = sample_tensor_with(
            reader.as_ref(),
            &t,
            &[5, 4], // shape override
            10,
            10,
            0,
            ViewDtype::Stored,
            SampleMode::Grid,
        )
        .unwrap();
        assert_eq!((s.total_rows, s.total_cols), (5, 4));
        assert_eq!(s.display_shape, vec![5, 4]);
        // Every cell is the flat index r*4 + c — no NaNs from row-straddling.
        for (i, &r) in s.rows.iter().enumerate() {
            for (j, &c) in s.cols.iter().enumerate() {
                assert_eq!(s.values[i][j], (r * 4 + c) as f64);
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
        use std::io::Read;
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
        let s = tensor_stats(
            &t,
            ViewDtype::Stored,
            &AtomicBool::new(false),
            &AtomicBool::new(false),
            None,
        )
        .unwrap();
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
        let s = sample_tensor(&t, 10, 10, 1, ViewDtype::Stored, SampleMode::Grid).unwrap();
        assert_eq!((s.total_rows, s.total_cols), (3, 4));
        assert_eq!((s.slices, s.slice), (2, 1));
        for (i, &r) in s.rows.iter().enumerate() {
            for (j, &c) in s.cols.iter().enumerate() {
                assert_eq!(s.values[i][j], (12 + r * 4 + c) as f64);
            }
        }
        // An out-of-range slice clamps to the last one.
        assert_eq!(
            sample_tensor(&t, 10, 10, 99, ViewDtype::Stored, SampleMode::Grid)
                .unwrap()
                .slice,
            1
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn squeezes_size_one_dimensions() {
        // Pure shape logic: size-1 dims drop out, anything left is unchanged.
        assert_eq!(squeezed_shape(&[128, 3088, 1, 748]), vec![128, 3088, 748]);
        assert_eq!(squeezed_shape(&[1, 5]), vec![5]);
        assert_eq!(squeezed_shape(&[128, 1, 748]), vec![128, 748]);
        assert_eq!(squeezed_shape(&[2, 3, 4]), vec![2, 3, 4]);
        assert_eq!(squeezed_shape(&[1, 1, 1]), Vec::<usize>::new());
    }

    #[test]
    fn previews_a_4d_tensor_with_a_unit_dim_as_3d() {
        use std::io::Write;
        let dir = std::env::temp_dir().join("checkpoint_explorer_sample_4d1");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("w.safetensors");
        // [2, 1, 3, 4] f32 — same 24 bytes as the [2, 3, 4] case, with an inert
        // size-1 dim wedged in the middle. value[s][r][c] = s*12 + r*4 + c.
        let header = br#"{"w":{"dtype":"F32","shape":[2,1,3,4],"data_offsets":[0,96]}}"#;
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(header).unwrap();
        for i in 0..24u32 {
            f.write_all(&(i as f32).to_le_bytes()).unwrap();
        }
        drop(f);

        // Squeezed to (2, 3, 4): 2 slices, each a 3×4 matrix. Slice 1 is
        // [[12..16],[16..20],[20..24]] — identical to the plain 3D case, proving
        // the reads are layout-correct (not just the row/col/slice counts).
        let t = fixture(&path, "w", &[2, 1, 3, 4], (0, 96));
        let s = sample_tensor(&t, 10, 10, 1, ViewDtype::Stored, SampleMode::Grid).unwrap();
        assert_eq!((s.total_rows, s.total_cols), (3, 4));
        assert_eq!((s.slices, s.slice), (2, 1));
        for (i, &r) in s.rows.iter().enumerate() {
            for (j, &c) in s.cols.iter().enumerate() {
                assert_eq!(s.values[i][j], (12 + r * 4 + c) as f64);
            }
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn previews_a_leading_unit_dim_as_lower_rank() {
        use std::io::Write;
        let dir = std::env::temp_dir().join("checkpoint_explorer_sample_1xn");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("w.safetensors");
        // [1, 5] f32 → squeezes to a single row of 5.
        let header = br#"{"w":{"dtype":"F32","shape":[1,5],"data_offsets":[0,20]}}"#;
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(header).unwrap();
        for i in 0..5u32 {
            f.write_all(&(i as f32).to_le_bytes()).unwrap();
        }
        drop(f);

        let t = fixture(&path, "w", &[1, 5], (0, 20));
        let s = sample_tensor(&t, 10, 10, 0, ViewDtype::Stored, SampleMode::Grid).unwrap();
        assert_eq!((s.total_rows, s.total_cols, s.slices), (1, 5, 1));
        assert_eq!(s.values[0], vec![0.0, 1.0, 2.0, 3.0, 4.0]);
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

        // u4: four nibbles per 16-bit container, last dim ×4 -> 8 values.
        // 0x1234 -> [4,3,2,1]; 0x00AB -> [11,10,0,0].
        let s = sample_tensor(&t, 10, 10, 0, ViewDtype::U4, SampleMode::Grid).unwrap();
        assert_eq!(s.total_cols, 8);
        assert_eq!(s.values[0], vec![4.0, 3.0, 2.0, 1.0, 11.0, 10.0, 0.0, 0.0]);

        // Signed: nibbles >= 8 are negative (0xB->-5, 0xA->-6).
        let s = sample_tensor(&t, 10, 10, 0, ViewDtype::I4, SampleMode::Grid).unwrap();
        assert_eq!(s.values[0], vec![4.0, 3.0, 2.0, 1.0, -5.0, -6.0, 0.0, 0.0]);

        // Raw bits are the individual nibbles, 4 bits wide.
        let s = sample_tensor(&t, 10, 10, 0, ViewDtype::U4, SampleMode::Grid).unwrap();
        let bits: Vec<u64> = s.raw[0].iter().map(|r| r.bits).collect();
        assert_eq!(bits, vec![0x4, 0x3, 0x2, 0x1, 0xB, 0xA, 0x0, 0x0]);
        assert!(s.raw[0].iter().all(|r| r.width == 4));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn raw_bits_capture_the_stored_pattern() {
        use std::io::Write;
        let dir = std::env::temp_dir().join("checkpoint_explorer_sample_rawbits");
        let _ = std::fs::create_dir_all(&dir);

        // F32 [1,3] = 1.0, 2.0, -1.0 → IEEE-754 bits 0x3f800000 / 0x40000000 /
        // 0xbf800000, each 32 bits wide.
        let path = dir.join("f32.safetensors");
        let header = br#"{"w":{"dtype":"F32","shape":[1,3],"data_offsets":[0,12]}}"#;
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(header).unwrap();
        for v in [1.0f32, 2.0, -1.0] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        drop(f);
        let t = fixture(&path, "w", &[1, 3], (0, 12));
        let s = sample_tensor(&t, 10, 10, 0, ViewDtype::Stored, SampleMode::Grid).unwrap();
        let bits: Vec<u64> = s.raw[0].iter().map(|r| r.bits).collect();
        assert_eq!(bits, vec![0x3f80_0000, 0x4000_0000, 0xbf80_0000]);
        assert!(s.raw[0].iter().all(|r| r.width == 32));
        let _ = std::fs::remove_file(&path);

        // I8 [1,3] = 1, -1, -128 → raw two's-complement bytes 0x01 / 0xFF / 0x80,
        // 8 bits wide, while the decoded values stay signed.
        let path = dir.join("i8.safetensors");
        let header = br#"{"w":{"dtype":"I8","shape":[1,3],"data_offsets":[0,3]}}"#;
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(header).unwrap();
        f.write_all(&[0x01, 0xFF, 0x80]).unwrap();
        drop(f);
        let t = fixture_dtype(&path, "w", "I8", &[1, 3], (0, 3));
        let s = sample_tensor(&t, 10, 10, 0, ViewDtype::Stored, SampleMode::Grid).unwrap();
        let bits: Vec<u64> = s.raw[0].iter().map(|r| r.bits).collect();
        assert_eq!(bits, vec![0x01, 0xFF, 0x80]);
        assert!(s.raw[0].iter().all(|r| r.width == 8));
        assert_eq!(s.values[0], vec![1.0, -1.0, -128.0]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn histogram_bins_per_value_for_4bit_views() {
        // The 4-bit views bin every possible value with no range needed.
        assert_eq!(
            histogram_bins(ViewDtype::U4, "I16", None, None),
            Some((HistBins::IntBins { start: 0, step: 1 }, 16))
        );
        assert_eq!(
            histogram_bins(ViewDtype::I4, "I16", None, None),
            Some((HistBins::IntBins { start: -8, step: 1 }, 16))
        );
    }

    #[test]
    fn histogram_bins_use_range_for_floats_and_wide_ints() {
        // Floats need the range; without it, undecidable.
        assert_eq!(histogram_bins(ViewDtype::Stored, "F32", None, None), None);
        let (bins, n) = histogram_bins(ViewDtype::Stored, "F32", Some((-1.0, 1.0)), None).unwrap();
        assert!(matches!(bins, HistBins::Range { .. }));
        assert_eq!(n, RANGE_BINS);
        // A small integer data span bins per value.
        assert_eq!(
            histogram_bins(ViewDtype::Stored, "I8", Some((0.0, 5.0)), None),
            Some((HistBins::IntBins { start: 0, step: 1 }, 6))
        );
    }

    #[test]
    fn histogram_bins_keep_integer_edges_for_wide_int_spans() {
        // A wide integer span must use whole-number-width bins, not a float
        // range — the bucket edges stay integers (start + i*step).
        let (bins, n) =
            histogram_bins(ViewDtype::Stored, "U16", Some((0.0, 65_535.0)), None).unwrap();
        let HistBins::IntBins { start, step } = bins else {
            panic!("wide integer span should produce integer bins, got {bins:?}");
        };
        assert_eq!(start, 0);
        assert!(step > 1, "wide span should group multiple integers per bin");
        assert!(n <= RANGE_BINS, "no more than {RANGE_BINS} bins");
        // The bins must cover the whole span.
        assert!(start + (n as i64) * step > 65_535);
    }

    #[test]
    fn histogram_bins_honour_a_requested_count() {
        // Floats: the requested count is the bin count exactly.
        let (_, n) = histogram_bins(ViewDtype::Stored, "F32", Some((-1.0, 1.0)), Some(12)).unwrap();
        assert_eq!(n, 12);
        // Wide integers: grouped into about the requested number of whole-number
        // buckets (never more, since the step is a whole number).
        let (bins, n) =
            histogram_bins(ViewDtype::Stored, "U16", Some((0.0, 65_535.0)), Some(16)).unwrap();
        assert!(matches!(bins, HistBins::IntBins { step, .. } if step > 1));
        assert!(n <= 16, "grouped to at most the requested count, got {n}");
        // A count larger than the distinct integer span still shows one bin per
        // value (you can't have more buckets than values).
        assert_eq!(
            histogram_bins(ViewDtype::Stored, "I8", Some((0.0, 5.0)), Some(40)),
            Some((HistBins::IntBins { start: 0, step: 1 }, 6))
        );
        // A small count groups an otherwise per-value integer span.
        let (bins, n) =
            histogram_bins(ViewDtype::Stored, "I16", Some((0.0, 19.0)), Some(5)).unwrap();
        assert!(matches!(bins, HistBins::IntBins { step, .. } if step > 1));
        assert!(n <= 5);
    }

    #[test]
    fn bin_index_maps_and_clamps() {
        let pv = HistBins::IntBins { start: -8, step: 1 };
        assert_eq!(bin_index(pv, 16, -8.0), 0);
        assert_eq!(bin_index(pv, 16, 7.0), 15);
        assert_eq!(bin_index(pv, 16, 99.0), 15); // clamped to the last bin
        // Stepped integer bins: each bin spans `step` values.
        let iv = HistBins::IntBins {
            start: 0,
            step: 100,
        };
        assert_eq!(bin_index(iv, 10, 0.0), 0);
        assert_eq!(bin_index(iv, 10, 99.0), 0);
        assert_eq!(bin_index(iv, 10, 100.0), 1);
        assert_eq!(bin_index(iv, 10, 250.0), 2);
        assert_eq!(bin_index(iv, 10, 9_999.0), 9); // clamped to the last bin
        let rg = HistBins::Range { lo: 0.0, hi: 10.0 };
        assert_eq!(bin_index(rg, 10, 0.0), 0);
        assert_eq!(bin_index(rg, 10, 9.999), 9);
        assert_eq!(bin_index(rg, 10, 10.0), 9); // the max lands in the last bin
        assert_eq!(bin_index(rg, 10, -5.0), 0); // clamped
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
        let (tensors, _metadata) = crate::hdf5::read(std::path::Path::new(&path)).unwrap();
        let t = tensors
            .into_iter()
            .find(|t| t.name == name)
            .expect("tensor");
        eprintln!("tensor {} dtype={} shape={:?}", t.name, t.dtype, t.shape);
        let started = std::time::Instant::now();
        match tensor_stats(
            &t,
            ViewDtype::Stored,
            &AtomicBool::new(false),
            &AtomicBool::new(false),
            None,
        ) {
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
        // Read the stored f32 bytes back, decoding under the stored view.
        let s = sample_tensor(&t, 10, 10, 0, ViewDtype::Stored, SampleMode::Grid).unwrap();
        assert_eq!((s.total_rows, s.total_cols), (4, 5));
        // HDF5 reads raw stored bytes now, so dtype overrides are available.
        assert!(s.overridable);
        for (i, &r) in s.rows.iter().enumerate() {
            for (j, &c) in s.cols.iter().enumerate() {
                assert_eq!(s.values[i][j], (r * 5 + c) as f64);
            }
        }

        // Exact stats over the whole dataset (raw byte read + streaming scan).
        // Values 0..=19, so mean 9.5 and one zero.
        let st = tensor_stats(
            &t,
            ViewDtype::Stored,
            &AtomicBool::new(false),
            &AtomicBool::new(false),
            None,
        )
        .unwrap();
        assert_eq!(st.count, 20);
        assert_eq!((st.min, st.max), (0.0, 19.0));
        assert!((st.mean - 9.5).abs() < 1e-9);
        assert_eq!(st.zeros, 1);
        assert_eq!(st.nonfinite, 0);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn reinterprets_hdf5_dtype_bytes() {
        // A small I16 dataset whose values pack two 4-bit nibbles each, so the
        // packed-u4 view should unpack them — proving HDF5 reads honour overrides
        // by reinterpreting the stored bytes (not libhdf5's converted values).
        let dir = std::env::temp_dir().join("checkpoint_explorer_reinterp_h5");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("q.h5");
        let _ = std::fs::remove_file(&path);
        // 0x21, 0x43 → low/high nibbles (1,2) and (3,4).
        let data: Vec<i16> = vec![0x21, 0x43];
        {
            let file = hdf5_metno::File::create(&path).unwrap();
            let ds = file.new_dataset::<i16>().shape([1, 2]).create("w").unwrap();
            ds.write_raw(&data).unwrap();
        }
        let t = TensorInfo {
            name: "w".to_string(),
            dtype: "I16".to_string(),
            shape: vec![1, 2],
            size_bytes: 4,
            num_elements: 2,
            storage: Storage::Unknown,
            source_path: path.to_string_lossy().into_owned(),
            layout: Layout::None,
        };

        // Stored view: the raw signed 16-bit values.
        let s = sample_tensor(&t, 10, 10, 0, ViewDtype::Stored, SampleMode::Grid).unwrap();
        assert_eq!(s.values[0], vec![0x21 as f64, 0x43 as f64]);

        // Packed u4: each 16-bit slot yields four nibbles, last dim ×4.
        let p = sample_tensor(&t, 10, 10, 0, ViewDtype::U4, SampleMode::Grid).unwrap();
        assert_eq!(p.total_cols, 8);
        assert_eq!(p.values[0], vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0]);

        // Stats under the packed view see all eight unpacked nibbles.
        let st = tensor_stats(
            &t,
            ViewDtype::U4,
            &AtomicBool::new(false),
            &AtomicBool::new(false),
            None,
        )
        .unwrap();
        assert_eq!(st.count, 8);
        assert_eq!((st.min, st.max), (0.0, 4.0));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn samples_a_3d_hdf5_slice() {
        // A 3D dataset [d0=2, d1=3, d2=4] of values v = d0*100 + d1*10 + d2, so
        // each element identifies its own (slice, row, col). Verifies the reader
        // maps a sampled (slice, row, col) to the right dataset element.
        let dir = std::env::temp_dir().join("checkpoint_explorer_3d_h5");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("v3.h5");
        let _ = std::fs::remove_file(&path);
        let data: Vec<f32> = (0..2)
            .flat_map(|d0| {
                (0..3).flat_map(move |d1| (0..4).map(move |d2| (d0 * 100 + d1 * 10 + d2) as f32))
            })
            .collect();
        {
            let file = hdf5_metno::File::create(&path).unwrap();
            let ds = file
                .new_dataset::<f32>()
                .shape([2, 3, 4])
                .create("w")
                .unwrap();
            ds.write_raw(&data).unwrap();
        }
        let t = TensorInfo {
            name: "w".to_string(),
            dtype: "F32".to_string(),
            shape: vec![2, 3, 4],
            size_bytes: 24 * 4,
            num_elements: 24,
            storage: Storage::Unknown,
            source_path: path.to_string_lossy().into_owned(),
            layout: Layout::None,
        };

        // Slice 1: every value should read as 100 + row*10 + col.
        let s = sample_tensor(&t, 10, 10, 1, ViewDtype::Stored, SampleMode::Grid).unwrap();
        assert_eq!((s.total_rows, s.total_cols, s.slices), (3, 4, 2));
        for (i, &r) in s.rows.iter().enumerate() {
            for (j, &c) in s.cols.iter().enumerate() {
                assert_eq!(s.values[i][j], (100 + r * 10 + c) as f64);
            }
        }

        let _ = std::fs::remove_file(&path);
    }
}
