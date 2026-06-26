//! End-to-end "cram"-style tests: run the real binary in `--plain` mode against
//! a generated fixture checkpoint and snapshot its rendered screen. Because
//! almost every screen is reproducible from CLI flags, one case == one command
//! line, and `--plain` makes the output stable plain text.
//!
//! Golden snapshots live under `tests/snapshots/`. After an intentional change,
//! review and accept them with:
//!
//! ```text
//! cargo insta review          # or: INSTA_UPDATE=always cargo test --test cli
//! ```
//!
//! Fixtures: the safetensors one is generated fresh each run (pure Rust,
//! deterministic, git-ignored); the HDF5 one is committed (`tests/fixtures/tiny.hdf5`,
//! regenerated with `cargo run --example gen_hdf5_fixture --features hdf5`),
//! because hdf5-metno isn't a dev-dependency. The HDF5 cases are gated on the
//! `hdf5` feature.

use std::collections::HashMap;
use std::path::Path;
use std::process::Command;
use std::sync::Once;

use safetensors::tensor::{Dtype, TensorView};

const FIXTURE: &str = "tests/fixtures/tiny.safetensors";

/// Build a tiny safetensors checkpoint mirroring a Qwen3-coder-ish layout: a few
/// dtypes and shapes (1D/2D/3D) under dotted names so the tree has structure.
/// Values don't matter for the tree / detail screens (no statistics are scanned
/// in `--plain`), so each payload is just the right number of bytes.
fn write_fixture(path: &str) {
    // (name, dtype, shape) — payloads are a byte ramp of the right size.
    let specs: &[(&str, Dtype, Vec<usize>)] = &[
        ("lm_head.weight", Dtype::I32, vec![2, 4]),
        ("model.embed_tokens.weight", Dtype::F16, vec![6, 4]),
        (
            "model.layers.0.self_attn.q_proj.weight",
            Dtype::BF16,
            vec![4, 4],
        ),
        (
            "model.layers.0.mlp.down_proj.weight",
            Dtype::U16,
            vec![3, 4, 5],
        ),
        ("model.layers.0.input_layernorm.weight", Dtype::F32, vec![4]),
        ("model.norm.weight", Dtype::F32, vec![4]),
        ("model.scale.u8", Dtype::U8, vec![8]),
    ];

    // Own the byte buffers so the borrowing `TensorView`s stay valid until write.
    let buffers: Vec<Vec<u8>> = specs
        .iter()
        .map(|(_, dt, shape)| {
            let bytes = shape.iter().product::<usize>() * dtype_size(*dt);
            (0..bytes).map(|i| (i % 251) as u8).collect()
        })
        .collect();

    let data: HashMap<String, TensorView> = specs
        .iter()
        .zip(&buffers)
        .map(|((name, dt, shape), buf)| {
            (
                name.to_string(),
                TensorView::new(*dt, shape.clone(), buf).expect("valid tensor view"),
            )
        })
        .collect();

    let metadata = Some(HashMap::from([("format".to_string(), "pt".to_string())]));

    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent).expect("create fixtures dir");
    }
    safetensors::serialize_to_file(&data, &metadata, Path::new(path)).expect("write fixture");
}

fn dtype_size(dt: Dtype) -> usize {
    match dt {
        Dtype::U8 | Dtype::I8 | Dtype::BOOL => 1,
        Dtype::F16 | Dtype::BF16 | Dtype::I16 | Dtype::U16 => 2,
        Dtype::F32 | Dtype::I32 | Dtype::U32 => 4,
        Dtype::F64 | Dtype::I64 | Dtype::U64 => 8,
        _ => 4,
    }
}

/// Generate the fixture once, even with tests running in parallel.
fn ensure_fixture() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| write_fixture(FIXTURE));
}

/// Run the binary in `--plain` mode against `fixture` and return its screen text.
fn run_plain(fixture: &str, extra_args: &[&str]) -> String {
    let mut args = vec![fixture];
    args.extend_from_slice(extra_args);
    args.push("--plain");
    let out = Command::new(env!("CARGO_BIN_EXE_checkpoint-explorer"))
        .args(&args)
        .output()
        .expect("run checkpoint-explorer");
    assert!(
        out.status.success(),
        "non-zero exit; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// The generated safetensors fixture, in `--plain`.
fn plain(extra_args: &[&str]) -> String {
    ensure_fixture();
    run_plain(FIXTURE, extra_args)
}

/// Normalize the fixture's path (shown verbatim, absolute, or left-elided with
/// `…` depending on the screen) to a stable token, so snapshots don't depend on
/// the checkout location.
fn settings() -> insta::Settings {
    let mut s = insta::Settings::clone_current();
    s.add_filter(r"\S*tiny\.(?:safetensors|hdf5)", "[FIXTURE]");
    s
}

#[test]
fn plain_tree() {
    settings().bind(|| insta::assert_snapshot!(plain(&[])));
}

#[test]
fn plain_detail_u16() {
    settings().bind(|| {
        insta::assert_snapshot!(plain(&["--tensor", "model.layers.0.mlp.down_proj.weight"]))
    });
}

#[test]
fn plain_detail_f16() {
    settings().bind(|| insta::assert_snapshot!(plain(&["--tensor", "model.embed_tokens.weight"])));
}

/// HDF5 fixture (`tests/fixtures/tiny.hdf5`, committed; regenerate with
/// `cargo run --example gen_hdf5_fixture --features hdf5`). Gated on the `hdf5`
/// feature so it only runs when the binary can read HDF5. Pins the fused-MoE
/// quantization-schema display (top-level + per-tensor + non-uniform), the
/// compression codec / `(uncompressed)` tags, and chunk reporting.
#[cfg(feature = "hdf5")]
mod hdf5 {
    use super::{run_plain, settings};

    const H5: &str = "tests/fixtures/tiny.hdf5";
    const MOE: &str = "model.layers.0.block_sparse_moe.experts";

    fn plain(extra_args: &[&str]) -> String {
        run_plain(H5, extra_args)
    }

    #[test]
    fn tree() {
        settings().bind(|| insta::assert_snapshot!(plain(&[])));
    }

    #[test]
    fn detail_down_proj_uniform_schema() {
        let t = format!("{MOE}.down_proj.weight");
        settings().bind(|| insta::assert_snapshot!(plain(&["--tensor", &t])));
    }

    #[test]
    fn detail_gate_up_nonuniform_schema() {
        let t = format!("{MOE}.gate_up_proj.weight");
        settings().bind(|| insta::assert_snapshot!(plain(&["--tensor", &t])));
    }

    #[test]
    fn detail_per_tensor_schema() {
        settings().bind(|| {
            insta::assert_snapshot!(plain(&["--tensor", "model.layers.0.custom_proj.weight"]))
        });
    }

    #[test]
    fn detail_compressed_f16() {
        settings().bind(|| insta::assert_snapshot!(plain(&["--tensor", "lm_head.weight"])));
    }
}
