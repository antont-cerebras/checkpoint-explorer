//! Repack an HDF5 checkpoint into a new file with a different compression codec.
//!
//! The Cerebras checkpoints compress their chunks with LZ4 (filter 32004), which
//! — being byte-oriented with no entropy coding — only reaches ~2× on the 4-bit
//! weights packed into 16-bit words. Re-compressing with an entropy-coding codec
//! (gzip, or the faster/denser zstd) recovers that win.
//!
//! We read each dataset's stored bytes (LZ4/zstd-decoded via the in-process
//! filters) and write a fresh dataset with the chosen codec, streaming along the
//! outer axis in a configurable buffer so peak memory stays bounded regardless
//! of tensor size.

use std::path::Path;

use anyhow::{Context, Result, bail};
use hdf5_metno::filters::Filter;
use hdf5_metno::types::{FloatSize, IntSize, TypeDescriptor};

use crate::codec::Codec;

/// Outcome of a repack, for the summary line.
pub struct Report {
    /// Number of datasets copied.
    pub tensors: usize,
    /// Datasets skipped (e.g. an unsupported dtype) — 0 on a clean run.
    pub skipped: usize,
    /// On-disk size of the source file (its existing compression).
    pub in_bytes: u64,
    /// On-disk size of the repacked file.
    pub out_bytes: u64,
    /// Total uncompressed (logical) size of all datasets.
    pub logical_bytes: u64,
    /// The codec the source used, if every compressed dataset shared one
    /// (`None` if uncompressed or mixed).
    pub source_codec: Option<Codec>,
}

impl Report {
    /// On-disk size ratio of source / repacked (>1 means we got smaller).
    pub fn ratio(&self) -> f64 {
        self.in_bytes as f64 / self.out_bytes.max(1) as f64
    }

    /// A human summary of how the repack changed the on-disk size, including the
    /// new codec's overall ratio against the uncompressed data.
    pub fn summary(&self, new: Codec) -> String {
        let pct = if self.in_bytes > 0 {
            (self.out_bytes as f64 / self.in_bytes as f64 - 1.0) * 100.0
        } else {
            0.0
        };
        // Relative to the source's existing on-disk size.
        let change = if self.out_bytes < self.in_bytes {
            format!("{:.0}% smaller ({:.2}×)", -pct, self.ratio())
        } else if self.out_bytes > self.in_bytes {
            format!("{pct:.0}% LARGER — the source compressed better")
        } else {
            "same size".to_string()
        };
        let from = match self.source_codec {
            Some(c) => c.label(),
            None => "uncompressed/mixed",
        };
        let vs_logical = self.logical_bytes as f64 / self.out_bytes.max(1) as f64;
        format!(
            "{} datasets{} · on disk {} ({from}) → {} ({}): {change} · {:.2}× vs uncompressed",
            self.tensors,
            if self.skipped > 0 {
                format!(", {} skipped", self.skipped)
            } else {
                String::new()
            },
            crate::utils::format_size(self.in_bytes as usize),
            crate::utils::format_size(self.out_bytes as usize),
            new.label(),
            vs_logical,
        )
    }
}

/// Map a dataset's filter pipeline to the codec it uses, or `None` if it stores
/// data uncompressed.
fn dataset_codec(ds: &hdf5_metno::Dataset) -> Option<Codec> {
    ds.filters().iter().find_map(|f| match f {
        Filter::Deflate(_) => Some(Codec::Gzip),
        Filter::User(crate::hdf5_lz4::LZ4_FILTER_ID, _) => Some(Codec::Lz4),
        Filter::User(crate::hdf5_zstd::ZSTD_FILTER_ID, _) => Some(Codec::Zstd),
        _ => None,
    })
}

/// How to repack: codec, its level (ignored for lz4/store), and the streaming
/// buffer size (bytes read/written per block).
pub struct Options {
    pub codec: Codec,
    pub level: u8,
    pub buffer_bytes: usize,
}

/// Repack the HDF5 file at `input` into `output` under `opts`. `progress(done,
/// total, name)` is called before each dataset is copied. Fails if `output`
/// already exists.
pub fn convert_hdf5(
    input: &Path,
    output: &Path,
    opts: &Options,
    mut progress: impl FnMut(usize, usize, &str),
) -> Result<Report> {
    // Never read and write the same file (a copy would obliterate the source).
    if same_file(input, output) {
        bail!("input and output are the same file: {}", input.display());
    }
    if output.exists() {
        bail!("output already exists: {}", output.display());
    }

    let src =
        hdf5_metno::File::open(input).with_context(|| format!("opening {}", input.display()))?;
    // Register the LZ4 + zstd filters (both directions): decode for reading the
    // source, encode for writing those codecs.
    crate::hdf5_lz4::register();
    crate::hdf5_zstd::register();
    let dst = hdf5_metno::File::create(output)
        .with_context(|| format!("creating {}", output.display()))?;

    let names = src.member_names().context("listing datasets")?;
    let total = names.len();
    let mut tensors = 0;
    let mut skipped = 0;
    let mut logical_bytes = 0u64;
    // The source's codec, if every compressed dataset shares one.
    let mut source_codec: Option<Codec> = None;
    let mut mixed = false;

    for (i, name) in names.iter().enumerate() {
        progress(i, total, name);
        let Ok(ds) = src.dataset(name) else {
            skipped += 1;
            continue;
        };
        if let Some(c) = dataset_codec(&ds) {
            match source_codec {
                None => source_codec = Some(c),
                Some(p) if p != c => mixed = true,
                _ => {}
            }
        }
        match copy_dataset(&dst, name, &ds, opts)
            .with_context(|| format!("copying dataset {name}"))?
        {
            Some(bytes) => {
                tensors += 1;
                logical_bytes += bytes;
            }
            None => skipped += 1,
        }
    }

    // Flush and close `dst` before measuring its on-disk size.
    drop(dst);
    drop(src);

    Ok(Report {
        tensors,
        skipped,
        in_bytes: std::fs::metadata(input).map(|m| m.len()).unwrap_or(0),
        out_bytes: std::fs::metadata(output).map(|m| m.len()).unwrap_or(0),
        logical_bytes,
        source_codec: if mixed { None } else { source_codec },
    })
}

/// Whether two paths refer to the same file (by absolute, lexically-normalised
/// path — `output` need not exist yet).
fn same_file(a: &Path, b: &Path) -> bool {
    match (std::path::absolute(a), std::path::absolute(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

/// Detect the codec a source file uses (for a same-codec warning), if every
/// compressed dataset shares one; `None` if uncompressed, mixed, or unreadable.
pub fn source_codec(input: &Path) -> Option<Codec> {
    let src = hdf5_metno::File::open(input).ok()?;
    crate::hdf5_lz4::register();
    crate::hdf5_zstd::register();
    let mut found: Option<Codec> = None;
    for name in src.member_names().ok()? {
        if let Ok(ds) = src.dataset(&name)
            && let Some(c) = dataset_codec(&ds)
        {
            match found {
                None => found = Some(c),
                Some(p) if p != c => return None,
                _ => {}
            }
        }
    }
    found
}

/// Copy one dataset into `dst` under `opts`. Returns the dataset's logical
/// (uncompressed) size in bytes when written, or `None` (skipped) for a dtype we
/// can't round-trip.
fn copy_dataset(
    dst: &hdf5_metno::File,
    name: &str,
    ds: &hdf5_metno::Dataset,
    opts: &Options,
) -> Result<Option<u64>> {
    let shape = ds.shape();
    let chunk = ds.chunk();
    let dtype = match ds.dtype() {
        Ok(dt) => dt,
        Err(_) => return Ok(None),
    };
    let descr = match dtype.to_descriptor() {
        Ok(d) => d,
        Err(_) => return Ok(None),
    };
    let logical = (shape.iter().product::<usize>() * dtype.size()) as u64;

    // Compression needs chunking, so stream only chunked, non-scalar datasets;
    // tiny contiguous ones (e.g. 1-D norms stored unchunked) are copied verbatim
    // (their size is negligible, so the codec wouldn't matter anyway).
    let stream = !shape.is_empty() && chunk.is_some();
    let level = opts.codec.clamp_level(opts.level);

    macro_rules! dispatch {
        ($T:ty) => {{
            if stream {
                let chunk = chunk.clone().unwrap();
                let b = dst
                    .new_dataset::<$T>()
                    .shape(shape.as_slice())
                    .chunk(chunk.as_slice());
                // Apply the chosen codec to the write pipeline.
                let out = match opts.codec {
                    Codec::Uncompressed => b.create(name)?,
                    Codec::Gzip => b.deflate(level).create(name)?,
                    Codec::Lz4 => b
                        .set_filters(&[Filter::user(crate::hdf5_lz4::LZ4_FILTER_ID, &[])])
                        .create(name)?,
                    Codec::Zstd => b
                        .set_filters(&[Filter::user(
                            crate::hdf5_zstd::ZSTD_FILTER_ID,
                            &[level as u32],
                        )])
                        .create(name)?,
                };
                stream_copy::<$T>(ds, &out, &shape, &chunk, opts.buffer_bytes)?;
            } else {
                let data = ds.read_raw::<$T>()?;
                dst.new_dataset::<$T>()
                    .shape(shape.as_slice())
                    .create(name)?
                    .write_raw(&data)?;
            }
        }};
    }

    match descr {
        TypeDescriptor::Float(FloatSize::U2) => dispatch!(half::f16),
        TypeDescriptor::Float(FloatSize::U4) => dispatch!(f32),
        TypeDescriptor::Float(FloatSize::U8) => dispatch!(f64),
        TypeDescriptor::Integer(IntSize::U1) => dispatch!(i8),
        TypeDescriptor::Integer(IntSize::U2) => dispatch!(i16),
        TypeDescriptor::Integer(IntSize::U4) => dispatch!(i32),
        TypeDescriptor::Integer(IntSize::U8) => dispatch!(i64),
        TypeDescriptor::Unsigned(IntSize::U1) => dispatch!(u8),
        TypeDescriptor::Unsigned(IntSize::U2) => dispatch!(u16),
        TypeDescriptor::Unsigned(IntSize::U4) => dispatch!(u32),
        TypeDescriptor::Unsigned(IntSize::U8) => dispatch!(u64),
        _ => return Ok(None),
    }
    Ok(Some(logical))
}

/// Copy `src` → `dst` (same shape/dtype) in row-blocks along the outer axis, so
/// only one block is resident at a time. Blocks are rounded to a multiple of the
/// chunk's outer extent so each write covers whole chunks.
fn stream_copy<T: hdf5_metno::H5Type + Clone + Default>(
    src: &hdf5_metno::Dataset,
    dst: &hdf5_metno::Dataset,
    shape: &[usize],
    chunk: &[usize],
    buffer_bytes: usize,
) -> Result<()> {
    use hdf5_metno::{Hyperslab, SliceOrIndex};

    let outer = shape[0];
    let inner: usize = shape[1..].iter().product::<usize>().max(1);
    let chunk0 = chunk.first().copied().unwrap_or(outer).max(1);
    // Aim for ~buffer_bytes per block, rounded to whole chunks along axis 0.
    let target_elems = (buffer_bytes / std::mem::size_of::<T>().max(1)).max(1);
    let rows = ((target_elems / inner).max(1) / chunk0).max(1) * chunk0;

    let mut i = 0;
    while i < outer {
        let hi = (i + rows).min(outer);
        let sel: Vec<SliceOrIndex> = std::iter::once(SliceOrIndex::from(i..hi))
            .chain(shape[1..].iter().map(|&d| SliceOrIndex::from(0..d)))
            .collect();
        let block = src.read_slice::<T, _, ndarray::IxDyn>(Hyperslab::from(sel.clone()))?;
        dst.write_slice(block.view(), Hyperslab::from(sel))?;
        i = hi;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hdf5_metno::filters::Filter;

    #[test]
    fn repacks_datasets_and_roundtrips() {
        let dir = std::env::temp_dir().join("checkpoint_explorer_convert_test");
        let _ = std::fs::create_dir_all(&dir);
        let src = dir.join("src.h5");
        let dst = dir.join("dst.h5");
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&dst);

        // Highly compressible data (few distinct values), so gzip clearly shrinks.
        let w: Vec<f32> = (0..64 * 64).map(|i| (i % 17) as f32).collect();
        let q: Vec<i16> = (0..2 * 8 * 8).map(|i| (i % 16) as i16).collect();
        {
            let f = hdf5_metno::File::create(&src).unwrap();
            f.new_dataset::<f32>()
                .shape([64, 64])
                .chunk([16, 16])
                .deflate(4)
                .create("w")
                .unwrap()
                .write_raw(&w)
                .unwrap();
            // 3D chunked (exercises the streaming hyperslab copy).
            f.new_dataset::<i16>()
                .shape([2, 8, 8])
                .chunk([1, 8, 8])
                .deflate(4)
                .create("q")
                .unwrap()
                .write_raw(&q)
                .unwrap();
            // Tiny unchunked 1-D (exercises the verbatim copy path).
            f.new_dataset::<f32>()
                .shape([5])
                .create("b")
                .unwrap()
                .write_raw(&[1.0f32, 2.0, 3.0, 4.0, 5.0])
                .unwrap();
        }

        let gzip = Options {
            codec: Codec::Gzip,
            level: 6,
            buffer_bytes: 256 << 20,
        };
        let report = convert_hdf5(&src, &dst, &gzip, |_, _, _| {}).unwrap();
        assert_eq!(report.tensors, 3);
        assert_eq!(report.skipped, 0);

        let f = hdf5_metno::File::open(&dst).unwrap();
        // Data round-trips exactly.
        assert_eq!(f.dataset("w").unwrap().read_raw::<f32>().unwrap(), w);
        assert_eq!(f.dataset("q").unwrap().read_raw::<i16>().unwrap(), q);
        assert_eq!(
            f.dataset("b").unwrap().read_raw::<f32>().unwrap(),
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0]
        );
        // The big dataset is gzip-compressed (stored smaller than logical).
        let w_ds = f.dataset("w").unwrap();
        assert!(w_ds.storage_size() < (64 * 64 * 4) as u64);
        assert!(
            w_ds.filters()
                .iter()
                .any(|fl| matches!(fl, Filter::Deflate(_)))
        );

        // Refuses to clobber an existing output.
        assert!(convert_hdf5(&src, &dst, &gzip, |_, _, _| {}).is_err());

        // The other codecs also round-trip exactly (zstd / lz4 / store).
        for codec in [Codec::Zstd, Codec::Lz4, Codec::Uncompressed] {
            let out = dir.join(format!("dst-{}.h5", codec.label()));
            let _ = std::fs::remove_file(&out);
            let opts = Options {
                codec,
                level: codec.default_level(),
                buffer_bytes: 4096, // tiny, to exercise multi-block streaming
            };
            convert_hdf5(&src, &out, &opts, |_, _, _| {}).unwrap();
            let g = hdf5_metno::File::open(&out).unwrap();
            assert_eq!(
                g.dataset("w").unwrap().read_raw::<f32>().unwrap(),
                w,
                "{codec:?}"
            );
            assert_eq!(
                g.dataset("q").unwrap().read_raw::<i16>().unwrap(),
                q,
                "{codec:?}"
            );
            let _ = std::fs::remove_file(&out);
        }

        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&dst);
    }
}
