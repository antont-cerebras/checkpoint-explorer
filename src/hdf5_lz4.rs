//! Runtime registration of the HDF5 LZ4 filter (id 32004).
//!
//! Cerebras checkpoints compress their dataset chunks with the LZ4 filter from
//! The HDF Group's filter registry. The libhdf5 we statically bundle only knows
//! its built-in filters (gzip/szip/shuffle/…), so reading an LZ4 dataset fails
//! with `can't open directory …/plugin` — libhdf5 goes hunting for an external
//! plugin `.so` that we don't ship. Rather than ship that plugin, we register a
//! *decompression-only* filter in-process: a Rust callback that undoes the
//! framing the HDF Group's `H5Zlz4` encoder writes and decompresses each block
//! with the pure-Rust `lz4_flex` (no extra C dependency).
//!
//! Framing produced by `H5Zlz4` (all integers big-endian):
//! ```text
//!   u64  total decompressed size
//!   u32  block size
//!   repeat until the whole output is filled:
//!     u32 compressed block length
//!     <that many> LZ4-block-compressed bytes
//! ```
//! A block whose stored length equals its uncompressed length was kept verbatim
//! (the encoder skips LZ4 when it would not shrink the block).

use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::sync::Once;

use hdf5_metno_sys::h5::{H5allocate_memory, H5free_memory};
use hdf5_metno_sys::h5z::{
    H5Z_CLASS_T_VERS, H5Z_FLAG_REVERSE, H5Z_class2_t, H5Z_filter_t, H5Zregister,
};

/// The HDF Group registered id for LZ4 (`H5Z_FILTER_LZ4`).
const LZ4_FILTER_ID: H5Z_filter_t = 32004;
/// Stored by reference inside libhdf5, so it must outlive the program: a string
/// literal does.
static FILTER_NAME: &[u8] = b"HDF5 lz4 filter (in-process)\0";

/// Register the LZ4 decompression filter with libhdf5, exactly once per process.
/// Cheap to call before every read; later calls are no-ops.
///
/// Call this only when libhdf5 has already been initialised (i.e. after a
/// `File::open`) and no other thread is inside an HDF5 call — libhdf5 is not
/// built thread-safe, and our reads are never concurrent with each other.
pub fn register() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let class = H5Z_class2_t {
            version: H5Z_CLASS_T_VERS as c_int,
            id: LZ4_FILTER_ID,
            encoder_present: 0, // we only ever decompress
            decoder_present: 1,
            name: FILTER_NAME.as_ptr() as *const c_char,
            can_apply: None,
            set_local: None,
            filter: Some(lz4_filter),
        };
        // H5Zregister copies the struct (but not the name string, hence the
        // 'static above). The cast matches its `*const c_void` signature.
        unsafe {
            H5Zregister(&class as *const H5Z_class2_t as *const c_void);
        }
    });
}

/// libhdf5 filter callback. We register a decoder only, so the forward
/// (compression) direction is a hard error. A `0` return tells libhdf5 the
/// filter failed, which surfaces as a normal read error upstream.
unsafe extern "C" fn lz4_filter(
    flags: c_uint,
    _cd_nelmts: usize,
    _cd_values: *const c_uint,
    nbytes: usize,
    buf_size: *mut usize,
    buf: *mut *mut c_void,
) -> usize {
    if flags & H5Z_FLAG_REVERSE == 0 {
        return 0; // no encoder registered
    }
    // Decompression works off untrusted file bytes; a malformed chunk must not
    // unwind into C. Pass pointers as integers so the closure stays UnwindSafe.
    let buf_addr = buf as usize;
    let bs_addr = buf_size as usize;
    std::panic::catch_unwind(|| unsafe {
        decompress(nbytes, buf_addr as *mut *mut c_void, bs_addr as *mut usize)
    })
    .unwrap_or(None)
    .unwrap_or(0)
}

/// Read a big-endian `u32` at `off`. The caller guarantees `off + 4 <= b.len()`.
#[inline]
fn be_u32(b: &[u8], off: usize) -> u32 {
    u32::from_be_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

/// The frame's declared total decompressed size (its `u64` big-endian header),
/// or `None` if `input` is too short to hold one.
fn frame_total_size(input: &[u8]) -> Option<usize> {
    let header = input.get(0..8)?;
    Some(u64::from_be_bytes(header.try_into().ok()?) as usize)
}

/// Decode an `H5Zlz4` frame in `input` fully into `out`, which must already be
/// sized to the frame's declared total (see [`frame_total_size`]). Returns
/// `false` on any malformed or inconsistent input. Every slice index is
/// bounds-checked first, so this never panics. Pure (no FFI), hence unit-tested.
fn decode_into(input: &[u8], out: &mut [u8]) -> bool {
    if input.len() < 12 || frame_total_size(input) != Some(out.len()) {
        return false;
    }
    let total = out.len();
    let mut block_size = be_u32(input, 8) as usize;
    if block_size == 0 {
        return false;
    }
    if block_size > total {
        block_size = total;
    }

    let mut rpos = 12usize; // past the 8-byte total + 4-byte block size
    let mut wpos = 0usize;
    while wpos < total {
        let this_block = block_size.min(total - wpos);
        if rpos + 4 > input.len() {
            return false;
        }
        let clen = be_u32(input, rpos) as usize;
        rpos += 4;
        if clen == 0 || rpos + clen > input.len() {
            return false;
        }
        let src = &input[rpos..rpos + clen];
        let dst = &mut out[wpos..wpos + this_block];
        let ok = if clen == this_block {
            // Stored uncompressed (LZ4 wouldn't have shrunk it).
            dst.copy_from_slice(src);
            true
        } else {
            matches!(lz4_flex::block::decompress_into(src, dst), Ok(n) if n == this_block)
        };
        if !ok {
            return false;
        }
        rpos += clen;
        wpos += this_block;
    }
    true
}

/// Undo the `H5Zlz4` framing in `*buf` (length `nbytes`), allocating the output
/// with libhdf5's own allocator and handing it back via `*buf`/`*buf_size`.
/// Returns the decompressed byte count, or `None` on any malformed input.
unsafe fn decompress(nbytes: usize, buf: *mut *mut c_void, buf_size: *mut usize) -> Option<usize> {
    unsafe {
        if buf.is_null() || (*buf).is_null() {
            return None;
        }
        let input = std::slice::from_raw_parts(*buf as *const u8, nbytes);
        let total = frame_total_size(input)?;

        // Allocate the result with HDF5's own allocator so the pipeline can free
        // it; decode straight into it (no extra copy).
        let out_ptr = H5allocate_memory(total, 0) as *mut u8;
        if out_ptr.is_null() {
            return None;
        }
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        if !decode_into(input, out) {
            H5free_memory(out_ptr as *mut c_void);
            return None;
        }

        // Swap our buffer in for the compressed one HDF5 handed us.
        H5free_memory(*buf);
        *buf = out_ptr as *mut c_void;
        *buf_size = total;
        Some(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Frame `data` the way The HDF Group's `H5Zlz4` encoder does: a big-endian
    /// `u64` total and `u32` block size, then each block as `u32` compressed
    /// length + payload (raw when LZ4 would not shrink it).
    fn encode_frame(data: &[u8], block_size: usize) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(data.len() as u64).to_be_bytes());
        out.extend_from_slice(&(block_size as u32).to_be_bytes());
        for block in data.chunks(block_size) {
            let compressed = lz4_flex::block::compress(block);
            // Match the encoder: store verbatim when compression doesn't help.
            let payload: &[u8] = if compressed.len() < block.len() {
                &compressed
            } else {
                block
            };
            out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
            out.extend_from_slice(payload);
        }
        out
    }

    fn roundtrip(data: &[u8], block_size: usize) {
        let frame = encode_frame(data, block_size);
        let mut out = vec![0u8; data.len()];
        assert!(decode_into(&frame, &mut out), "decode failed");
        assert_eq!(out, data);
    }

    #[test]
    fn decodes_compressible_single_and_multi_block() {
        // Highly compressible (LZ4 shrinks it), exercising the decompress path.
        let data = vec![7u8; 100_000];
        roundtrip(&data, 1 << 16); // single block
        roundtrip(&data, 4096); // many full blocks
        roundtrip(&data, 30_000); // a smaller final partial block
    }

    #[test]
    fn decodes_stored_uncompressed_blocks() {
        // Pseudo-random, incompressible data is stored verbatim (clen == block).
        let data: Vec<u8> = (0..5000u32)
            .map(|i| (i.wrapping_mul(2654435761) >> 13) as u8)
            .collect();
        roundtrip(&data, 1024);
    }

    #[test]
    fn rejects_malformed_frames() {
        let mut out = vec![0u8; 16];
        // Too short to even hold the header.
        assert!(!decode_into(&[0, 1, 2, 3], &mut out));
        // Header total disagrees with the output length.
        let mut frame = (32u64).to_be_bytes().to_vec();
        frame.extend_from_slice(&(8u32).to_be_bytes());
        assert!(!decode_into(&frame, &mut out));
        // Valid header (total 16) but the block table is truncated.
        let mut frame = (16u64).to_be_bytes().to_vec();
        frame.extend_from_slice(&(16u32).to_be_bytes());
        assert!(!decode_into(&frame, &mut out));
    }
}
