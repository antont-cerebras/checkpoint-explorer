//! Generate the committed HDF5 end-to-end test fixture, `tests/fixtures/tiny.hdf5`.
//!
//! Pure Rust (hdf5-metno's write API). The integration tests can't generate it
//! themselves (hdf5-metno isn't a dev-dependency, and there's no lib target to
//! call into), so the tiny file is committed and regenerated with:
//!
//! ```text
//! cargo run --example gen_hdf5_fixture --features hdf5
//! ```
//!
//! It mirrors a fused-MoE Qwen3-coder-ish layout small enough to eyeball: a few
//! dtypes and shapes, gzip-compressed + chunked and uncompressed datasets, and
//! quantization schemas in both storage shapes (top-level `codebook_packing_schema`
//! keyed by projection, and a per-tensor `…__metadata__`) plus a non-uniform one.

use std::str::FromStr;

use half::f16;
use hdf5_metno::File;
use hdf5_metno::types::VarLenUnicode;

const PATH: &str = "tests/fixtures/tiny.hdf5";

fn main() {
    std::fs::create_dir_all("tests/fixtures").expect("create fixtures dir");
    let _ = std::fs::remove_file(PATH);
    let f = File::create(PATH).expect("create hdf5");

    // A few transformer layers of fused-MoE U16 codebook weights (3D experts),
    // gzip + multi-chunk, plus a layernorm and an attention projection each. The
    // expert schemas come from the top-level `codebook_packing_schema` below
    // (uniform u3×5 for down_proj, non-uniform [4,3,3,6] for gate_up_proj), so
    // every layer's projections pick one up — exercising the layer grouping
    // (`layers (☰ N, …)`) and multi-layer schema application.
    for layer in 0..3 {
        let moe = format!("model.layers.{layer}.block_sparse_moe.experts");
        write_ds::<u16>(
            &f,
            &format!("{moe}.down_proj.weight"),
            &[6, 3, 4],
            Some(&[2, 3, 4]),
            true,
            &ramp_u16(72),
        );
        write_ds::<f16>(
            &f,
            &format!("{moe}.down_proj.weight.codebook"),
            &[4, 4, 2],
            None,
            true,
            &ramp_f16(32),
        );
        write_ds::<f16>(
            &f,
            &format!("{moe}.down_proj.weight.qscale"),
            &[6, 4, 4],
            None,
            true,
            &ramp_f16(96),
        );
        write_ds::<u16>(
            &f,
            &format!("{moe}.gate_up_proj.weight"),
            &[6, 3, 4],
            Some(&[2, 3, 4]),
            true,
            &ramp_u16(72),
        );
        write_ds::<f32>(
            &f,
            &format!("model.layers.{layer}.input_layernorm.weight"),
            &[4],
            None,
            true,
            &ramp_f32(4),
        );
        write_ds::<i32>(
            &f,
            &format!("model.layers.{layer}.self_attn.q_proj.weight"),
            &[4, 4],
            None,
            true,
            &ramp_i32(16),
        );
    }

    // A U16 tensor whose schema is per-tensor (`…__metadata__`, uniform u4×4).
    write_ds::<u16>(
        &f,
        "model.layers.0.custom_proj.weight",
        &[4, 5],
        None,
        true,
        &ramp_u16(20),
    );

    // Assorted dtypes / shapes / storage outside the layer stack.
    write_ds::<f16>(&f, "lm_head.weight", &[10, 4], None, true, &ramp_f16(40));
    write_ds::<f16>(
        &f,
        "model.embed_tokens.weight",
        &[10, 4],
        None,
        true,
        &ramp_f16(40),
    );
    write_ds::<f32>(&f, "model.norm.weight", &[4], None, false, &ramp_f32(4)); // uncompressed
    write_ds::<i8>(&f, "model.scale.i8", &[8], None, true, &ramp_i8(8));

    // Metadata (root attributes).
    // Top-level schema keyed by projection segment; no `.__metadata__` wrapper.
    write_attr_str(
        &f,
        "codebook_packing_schema",
        r#"{"down_proj":{"bit_widths":[3,3,3,3,3]},"gate_up_proj":{"bit_widths":[4,3,3,6]}}"#,
    );
    // Per-tensor schema: a torch `.__metadata__` wrapper carrying a quantization_schema.
    write_attr_str(
        &f,
        "model.layers.0.custom_proj.weight.__metadata__",
        r#"{"model.layers.0.custom_proj.weight.__metadata__":{"quantization_schema":{"bit_widths":[4,4,4,4]}}}"#,
    );
    // A string-valued config metadata (the StringSerializer wrapper).
    write_attr_str(
        &f,
        "inference_version.__metadata__",
        r#"{"inference_version.__metadata__":{"string_value":"1.5","__TYPE__":"StringSerializer"}}"#,
    );
    f.new_attr::<f64>()
        .create("__version__")
        .unwrap()
        .write_scalar(&0.5f64)
        .unwrap();
    f.new_attr::<bool>()
        .create("__SUCCESS__")
        .unwrap()
        .write_scalar(&true)
        .unwrap();

    drop(f);
    println!("wrote {PATH}");
}

/// Write one dataset, optionally chunked and/or gzip-compressed.
fn write_ds<T: hdf5_metno::H5Type>(
    f: &File,
    name: &str,
    shape: &[usize],
    chunk: Option<&[usize]>,
    gzip: bool,
    data: &[T],
) {
    let ds = match (chunk, gzip) {
        (Some(c), true) => f
            .new_dataset::<T>()
            .shape(shape)
            .chunk(c)
            .deflate(4)
            .create(name),
        (Some(c), false) => f.new_dataset::<T>().shape(shape).chunk(c).create(name),
        (None, true) => f.new_dataset::<T>().shape(shape).deflate(4).create(name),
        (None, false) => f.new_dataset::<T>().shape(shape).create(name),
    }
    .unwrap_or_else(|e| panic!("create {name}: {e}"));
    ds.write_raw(data)
        .unwrap_or_else(|e| panic!("write {name}: {e}"));
}

fn write_attr_str(f: &File, name: &str, json: &str) {
    f.new_attr::<VarLenUnicode>()
        .create(name)
        .unwrap()
        .write_scalar(&VarLenUnicode::from_str(json).unwrap())
        .unwrap();
}

// Deterministic payloads (values aren't checked by the tree / detail screens).
fn ramp_u16(n: usize) -> Vec<u16> {
    (0..n).map(|i| (i % 32) as u16).collect()
}
fn ramp_f16(n: usize) -> Vec<f16> {
    (0..n)
        .map(|i| f16::from_f32((i % 8) as f32 * 0.5))
        .collect()
}
fn ramp_f32(n: usize) -> Vec<f32> {
    (0..n).map(|i| i as f32 * 0.5).collect()
}
fn ramp_i32(n: usize) -> Vec<i32> {
    (0..n).map(|i| i as i32 - 5).collect()
}
fn ramp_i8(n: usize) -> Vec<i8> {
    (0..n).map(|i| i as i8 - 4).collect()
}
