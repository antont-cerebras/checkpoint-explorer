//! In-process Zstandard HDF5 filter (registry id 32015), encode + decode.
//!
//! libhdf5 has no built-in zstd, so — as with the LZ4 decoder — we register a
//! Rust filter at runtime. Unlike LZ4 (decode-only, since these checkpoints
//! ship LZ4-compressed and we only read them), here we register *both*
//! directions so the repack command can write zstd and the explorer can read it
//! back. The framing is a plain zstd frame per chunk, matching the community
//! `H5Z_FILTER_ZSTD` filter (h5py + `hdf5plugin`), so files we write are
//! readable by other tools too.

use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::sync::Once;

use hdf5_metno_sys::h5::{H5allocate_memory, H5free_memory};
use hdf5_metno_sys::h5z::{
    H5Z_CLASS_T_VERS, H5Z_FLAG_REVERSE, H5Z_class2_t, H5Z_filter_t, H5Zregister,
};

/// The HDF Group registered id for Zstandard.
pub const ZSTD_FILTER_ID: H5Z_filter_t = 32015;
/// Default compression level when none is supplied via `cd_values`.
pub const DEFAULT_LEVEL: i32 = 3;

static FILTER_NAME: &[u8] = b"zstd (in-process)\0";

/// Register the zstd filter (encode + decode) with libhdf5, once per process.
/// Call before reading or writing a zstd-filtered dataset; later calls are
/// no-ops. Must be invoked while no other thread is inside an HDF5 call.
pub fn register() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let class = H5Z_class2_t {
            version: H5Z_CLASS_T_VERS as c_int,
            id: ZSTD_FILTER_ID,
            encoder_present: 1,
            decoder_present: 1,
            name: FILTER_NAME.as_ptr() as *const c_char,
            can_apply: None,
            set_local: None,
            filter: Some(zstd_filter),
        };
        unsafe {
            H5Zregister(&class as *const H5Z_class2_t as *const c_void);
        }
    });
}

/// libhdf5 filter callback: zstd-compress (forward) or decompress (reverse) the
/// chunk in `*buf`. Returns the new byte length, or 0 on failure (which libhdf5
/// surfaces as a read/write error).
unsafe extern "C" fn zstd_filter(
    flags: c_uint,
    cd_nelmts: usize,
    cd_values: *const c_uint,
    nbytes: usize,
    buf_size: *mut usize,
    buf: *mut *mut c_void,
) -> usize {
    // Read the compression level (forward only) before entering catch_unwind.
    let level = if !cd_values.is_null() && cd_nelmts >= 1 {
        unsafe { *cd_values as i32 }
    } else {
        DEFAULT_LEVEL
    };
    let reverse = flags & H5Z_FLAG_REVERSE != 0;
    let buf_addr = buf as usize;
    let bs_addr = buf_size as usize;
    std::panic::catch_unwind(|| unsafe {
        run(
            reverse,
            level,
            nbytes,
            buf_addr as *mut *mut c_void,
            bs_addr as *mut usize,
        )
    })
    .unwrap_or(0)
}

unsafe fn run(
    reverse: bool,
    level: i32,
    nbytes: usize,
    buf: *mut *mut c_void,
    buf_size: *mut usize,
) -> usize {
    unsafe {
        if buf.is_null() || (*buf).is_null() {
            return 0;
        }
        let input = std::slice::from_raw_parts(*buf as *const u8, nbytes);
        let out = if reverse {
            zstd::decode_all(input)
        } else {
            zstd::encode_all(input, level)
        };
        let Ok(out) = out else {
            return 0;
        };
        install_output(buf, buf_size, &out)
    }
}

/// Replace `*buf` with a fresh, HDF5-owned buffer holding `out`. Returns its
/// length, or 0 on allocation failure.
unsafe fn install_output(buf: *mut *mut c_void, buf_size: *mut usize, out: &[u8]) -> usize {
    unsafe {
        let p = H5allocate_memory(out.len().max(1), 0) as *mut u8;
        if p.is_null() {
            return 0;
        }
        std::ptr::copy_nonoverlapping(out.as_ptr(), p, out.len());
        H5free_memory(*buf);
        *buf = p as *mut c_void;
        *buf_size = out.len();
        out.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hdf5_metno::filters::Filter;

    #[test]
    fn writes_and_reads_zstd_compressed_dataset() {
        register();
        let dir = std::env::temp_dir().join("checkpoint_explorer_zstd_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("z.h5");
        let _ = std::fs::remove_file(&path);

        // Compressible i16 data (few distinct values).
        let data: Vec<i16> = (0..64 * 64).map(|i| (i % 16) as i16).collect();
        {
            let f = hdf5_metno::File::create(&path).unwrap();
            let ds = f
                .new_dataset::<i16>()
                .shape([64, 64])
                .chunk([16, 64])
                .set_filters(&[Filter::user(ZSTD_FILTER_ID, &[6])])
                .create("w")
                .unwrap();
            ds.write_raw(&data).unwrap();
        }

        let f = hdf5_metno::File::open(&path).unwrap();
        let ds = f.dataset("w").unwrap();
        // The data round-trips through the zstd encode + decode filter.
        assert_eq!(ds.read_raw::<i16>().unwrap(), data);
        // It really used the zstd filter, and it shrank the data.
        assert!(
            ds.filters()
                .iter()
                .any(|fl| matches!(fl, Filter::User(id, _) if *id == ZSTD_FILTER_ID))
        );
        assert!(ds.storage_size() < (64 * 64 * 2) as u64);

        let _ = std::fs::remove_file(&path);
    }
}
