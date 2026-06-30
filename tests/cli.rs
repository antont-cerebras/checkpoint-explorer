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

/// Run the binary with exactly `args` and return its stdout.
fn run_bin(args: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_checkpoint-explorer"))
        .args(args)
        .output()
        .expect("run checkpoint-explorer");
    assert!(
        out.status.success(),
        "non-zero exit; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Run the binary in `--plain` mode against `fixture` and return its screen text.
fn run_plain(fixture: &str, extra_args: &[&str]) -> String {
    let mut args = vec![fixture];
    args.extend_from_slice(extra_args);
    args.push("--plain");
    run_bin(&args)
}

/// Verify the `y` round-trip for a screen: render it directly, take the CLI
/// command `y` would copy to reopen it (`--emit-command`), re-render from that,
/// and require the two screens to be identical. Catches any state a screen shows
/// but its reopen command fails to express.
fn assert_y_roundtrip(fixture: &str, extra_args: &[&str]) {
    let direct = run_plain(fixture, extra_args);

    let mut emit = vec![fixture];
    emit.extend_from_slice(extra_args);
    emit.push("--emit-command");
    let command = run_bin(&emit);

    // The command is `checkpoint-explorer <path> <flags…>`; drop the program name
    // and render what's left (the fixture's names/paths are shell-safe, so the
    // tokens never need de-quoting).
    let mut reopen: Vec<&str> = command.split_whitespace().skip(1).collect();
    reopen.push("--plain");
    let reopened = run_bin(&reopen);

    // The two renders are independent scans, so a statistics / histogram duration
    // (`(2ms)`) differs run to run — normalize it before comparing.
    assert_eq!(
        strip_scan_time(&direct),
        strip_scan_time(&reopened),
        "y round-trip diverged\n  opened with: {extra_args:?}\n  reopened by: {}",
        command.trim(),
    );
}

/// Replace the scan-duration suffix (`(2ms)`, `(1.0s)`) with a stable token, so
/// the round-trip compares everything except the inherently-varying timing.
fn strip_scan_time(s: &str) -> String {
    use std::sync::OnceLock;
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| regex::Regex::new(r"\(\d+(?:\.\d+)?m?s\)").unwrap())
        .replace_all(s, "(<time>)")
        .into_owned()
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
    // The statistics / histogram scan duration (e.g. `(2ms)`, `(1.0s)`) is timing.
    s.add_filter(r"\(\d+(?:\.\d+)?m?s\)", "(<time>)");
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

#[test]
fn plain_values_u16() {
    settings().bind(|| {
        insta::assert_snapshot!(plain(&[
            "--tensor",
            "model.layers.0.mlp.down_proj.weight",
            "--values"
        ]))
    });
}

#[test]
fn plain_histogram_u16() {
    settings().bind(|| {
        insta::assert_snapshot!(plain(&[
            "--tensor",
            "model.layers.0.mlp.down_proj.weight",
            "--histogram"
        ]))
    });
}

#[test]
fn plain_tree_expanded() {
    settings().bind(|| insta::assert_snapshot!(plain(&["--tree-state", "expanded"])));
}

#[test]
fn y_roundtrips() {
    ensure_fixture();
    let t = "model.layers.0.mlp.down_proj.weight";
    for extra in [
        vec![],                             // tree (default expansion)
        vec!["--tree-state", "expanded"],   // E
        vec!["--tree-state", "collapsed"],  // C
        vec!["--tensor", t],                // detail
        vec!["--tensor", t, "--histogram"], // detail + histogram
        vec!["--tensor", t, "--values", "--slice", "1"],
        vec!["--tensor", t, "--values", "--overview", "--base", "hex"],
        vec!["--tensor", t, "--heatmap"],
    ] {
        assert_y_roundtrip(FIXTURE, &extra);
    }
}

/// Run a failing `--plain` request: assert it exits non-zero (a snapshot can't
/// see the exit code) and return the command line + its stderr for snapshotting.
fn run_plain_err(extra_args: &[&str]) -> String {
    ensure_fixture();
    let mut args = vec![FIXTURE];
    args.extend_from_slice(extra_args);
    args.push("--plain");
    let out = Command::new(env!("CARGO_BIN_EXE_checkpoint-explorer"))
        .args(&args)
        .output()
        .expect("run checkpoint-explorer");
    assert!(
        !out.status.success(),
        "expected non-zero exit for {extra_args:?}, got success"
    );
    format!(
        "$ checkpoint-explorer {}\n{}",
        args.join(" "),
        String::from_utf8_lossy(&out.stderr)
    )
}

/// A request that can't be honored must exit non-zero with an explanation, not
/// silently fall back to an unrelated screen. `--plain` exercises the same
/// resolution path as the interactive `--exit` one-shot (both headless), so it
/// stands in for the `--exit` exit code (which needs a tty to reach). The
/// snapshot pins the exact wording — which names the specific problem rather
/// than a vague "invalid" — so any reword surfaces in `cargo insta review`.
#[test]
fn plain_request_errors() {
    let t = "model.layers.0.mlp.down_proj.weight";
    let report = [
        run_plain_err(&["--tensor", "no.such.tensor"]),
        run_plain_err(&["--metadata", "no.such.meta"]),
        run_plain_err(&["--tensor", t, "--shape", "abc"]),
        run_plain_err(&["--tensor", t, "--slice", "9999"]),
    ]
    .join("\n");
    settings().bind(|| insta::assert_snapshot!(report));
}

/// Opening an HDF5 file with a binary built *without* the `hdf5` feature must
/// fail loudly (non-zero exit + an explanation that names the rebuild flag),
/// rather than silently loading an empty checkpoint that reads "0 tensors". The
/// non-zero exit must hold in headless `--exit`/`--plain` modes too, so scripts
/// detect it. Only meaningful when the feature is off, so it's gated out of the
/// `hdf5` build.
#[cfg(not(feature = "hdf5"))]
#[test]
fn hdf5_without_feature_errors() {
    const H5: &str = "tests/fixtures/tiny.hdf5";
    for extra in [&[][..], &["--exit"][..], &["--plain"][..]] {
        let mut args = vec![H5];
        args.extend_from_slice(extra);
        let out = Command::new(env!("CARGO_BIN_EXE_checkpoint-explorer"))
            .args(&args)
            .output()
            .expect("run checkpoint-explorer");
        assert!(
            !out.status.success(),
            "expected non-zero exit for {args:?}, got success"
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("without HDF5 support") && stderr.contains("--features hdf5"),
            "expected an HDF5-support error naming the rebuild flag for {args:?}; stderr:\n{stderr}"
        );
    }
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

    // Synchronously-scanned screens: the histogram (intrinsic 0..7 span for the
    // unpacked codebook view), statistics, and the numeric / heatmap data views
    // in each layout. The scan time is filtered out by `settings`.

    #[test]
    fn detail_histogram() {
        let t = format!("{MOE}.down_proj.weight");
        settings().bind(|| insta::assert_snapshot!(plain(&["--tensor", &t, "--histogram"])));
    }

    #[test]
    fn detail_compute_stats() {
        let t = format!("{MOE}.down_proj.weight");
        settings().bind(|| insta::assert_snapshot!(plain(&["--tensor", &t, "--compute-stats"])));
    }

    #[test]
    fn values_edges() {
        let t = format!("{MOE}.down_proj.weight");
        settings().bind(|| insta::assert_snapshot!(plain(&["--tensor", &t, "--values"])));
    }

    #[test]
    fn values_overview() {
        let t = format!("{MOE}.down_proj.weight");
        settings()
            .bind(|| insta::assert_snapshot!(plain(&["--tensor", &t, "--values", "--overview"])));
    }

    #[test]
    fn heatmap() {
        let t = format!("{MOE}.down_proj.weight");
        settings().bind(|| insta::assert_snapshot!(plain(&["--tensor", &t, "--heatmap"])));
    }

    // Main-screen keyboard shortcuts, reached via their flags: bulk expand /
    // collapse (E / C), search (/), and the context-sensitive legend (l) over
    // the tree, a detail, and a data view.

    #[test]
    fn tree_expanded() {
        settings().bind(|| insta::assert_snapshot!(plain(&["--tree-state", "expanded"])));
    }

    #[test]
    fn tree_collapsed() {
        settings().bind(|| insta::assert_snapshot!(plain(&["--tree-state", "collapsed"])));
    }

    #[test]
    fn tree_search() {
        settings().bind(|| insta::assert_snapshot!(plain(&["--search", "down_proj"])));
    }

    #[test]
    fn legend_tree() {
        settings().bind(|| insta::assert_snapshot!(plain(&["--legend"])));
    }

    #[test]
    fn legend_detail() {
        let t = format!("{MOE}.down_proj.weight");
        settings().bind(|| insta::assert_snapshot!(plain(&["--tensor", &t, "--legend"])));
    }

    #[test]
    fn legend_values() {
        let t = format!("{MOE}.down_proj.weight");
        settings()
            .bind(|| insta::assert_snapshot!(plain(&["--tensor", &t, "--values", "--legend"])));
    }

    // The `y` round-trip meta-test: every state-bearing screen must reopen to
    // itself from the command `y` copies. Covers the bulk tree expansion, the
    // schema views, and the full matrix of data-view state (layout + position,
    // slice, zebra, base). (Search / legend are transient overlays you can't `y`
    // from, so they're cram-only above.)
    #[test]
    fn y_roundtrips() {
        let dp = format!("{MOE}.down_proj.weight");
        let cases: &[Vec<&str>] = &[
            vec![],                                     // tree (default expansion)
            vec!["--tree-state", "expanded"],           // E
            vec!["--tree-state", "collapsed"],          // C
            vec!["--tensor", &dp, "--tree"],            // tree with a tensor revealed
            vec!["--tensor", &dp],                      // unpacked detail
            vec!["--tensor", &dp, "--dtype", "stored"], // raw U16 over a schema
            vec!["--tensor", &dp, "--histogram"],
            vec!["--tensor", &dp, "--histogram", "--bins", "4"],
            vec!["--tensor", &dp, "--compute-stats"],
            vec!["--tensor", "model.layers.0.custom_proj.weight"], // per-tensor schema
            vec!["--tensor", &dp, "--values"],
            vec!["--tensor", &dp, "--values", "--overview"],
            vec!["--tensor", &dp, "--values", "--window=1,1"],
            vec!["--tensor", &dp, "--values", "--edge=0.25,0.75"],
            vec!["--tensor", &dp, "--values", "--zebra", "cols"],
            vec!["--tensor", &dp, "--values", "--base", "hex"],
            vec!["--tensor", &dp, "--values", "--slice", "2"],
            vec!["--tensor", &dp, "--heatmap"],
        ];
        for extra in cases {
            super::assert_y_roundtrip(H5, extra);
        }
    }

    // Pin the actual command `y` copies for each screen (documents the round-trip
    // verified above). The fixture path is filtered to `[FIXTURE]`.
    #[test]
    fn emit_commands() {
        let dp = format!("{MOE}.down_proj.weight");
        let cases: &[(&str, Vec<&str>)] = &[
            ("detail", vec!["--tensor", &dp]),
            ("dtype stored", vec!["--tensor", &dp, "--dtype", "stored"]),
            ("histogram", vec!["--tensor", &dp, "--histogram"]),
            (
                "histogram bins",
                vec!["--tensor", &dp, "--histogram", "--bins", "4"],
            ),
            (
                "values window",
                vec!["--tensor", &dp, "--values", "--window=1,1"],
            ),
            (
                "values hex",
                vec!["--tensor", &dp, "--values", "--base", "hex"],
            ),
            ("heatmap", vec!["--tensor", &dp, "--heatmap"]),
        ];
        let mut out = String::new();
        for (label, args) in cases {
            let mut a = vec![H5];
            a.extend_from_slice(args);
            a.push("--emit-command");
            out.push_str(&format!("{label}: {}\n", super::run_bin(&a).trim()));
        }
        settings().bind(|| insta::assert_snapshot!(out));
    }
}

// ---- `diff` subcommand ----

/// Write a safetensors file from (name, dtype, shape) specs + string metadata —
/// a parametric sibling of `write_fixture`, used to build the diff fixtures.
fn write_st(path: &str, specs: &[(&str, Dtype, Vec<usize>)], metadata: &[(&str, &str)]) {
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
    let meta = Some(
        metadata
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect::<HashMap<_, _>>(),
    );
    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent).expect("create fixtures dir");
    }
    safetensors::serialize_to_file(&data, &meta, Path::new(path)).expect("write fixture");
}

const DIFF_OLD: &str = "tests/fixtures/diff_old.safetensors";
const DIFF_NEW: &str = "tests/fixtures/diff_new.safetensors";

/// Two checkpoints differing by one removed, one added, and two changed tensors
/// (a dtype change and a shape change), plus one added and one changed metadata
/// entry — and one tensor + one metadata entry identical in both.
fn ensure_diff_fixtures() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        write_st(
            DIFF_OLD,
            &[
                ("lm_head.weight", Dtype::F16, vec![2, 2]),
                ("model.embed_tokens.weight", Dtype::F16, vec![6, 4]),
                ("model.norm.weight", Dtype::F32, vec![4]),
                ("model.layers.0.input_layernorm.weight", Dtype::F32, vec![4]),
            ],
            &[("format", "pt"), ("note", "original")],
        );
        write_st(
            DIFF_NEW,
            &[
                ("model.embed_tokens.weight", Dtype::BF16, vec![6, 4]),
                ("model.norm.weight", Dtype::F32, vec![8]),
                ("model.layers.0.input_layernorm.weight", Dtype::F32, vec![4]),
                ("model.rotary_emb.inv_freq", Dtype::F32, vec![16]),
            ],
            &[("format", "pt"), ("note", "edited"), ("extra", "x")],
        );
    });
}

/// Run `diff OLD NEW` (relative paths, so the header is checkout-independent) and
/// return its stdout plus exit code.
fn run_diff(old: &str, new: &str) -> (String, i32) {
    let out = Command::new(env!("CARGO_BIN_EXE_checkpoint-explorer"))
        .args(["diff", old, new])
        .output()
        .expect("run diff");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        out.status.code().unwrap_or(-1),
    )
}

#[test]
fn diff_lists_changes_and_exits_1() {
    ensure_diff_fixtures();
    let (out, code) = run_diff(DIFF_OLD, DIFF_NEW);
    assert_eq!(code, 1, "differences should exit 1; stdout:\n{out}");
    insta::assert_snapshot!(out);
}

#[test]
fn diff_identical_exits_0() {
    ensure_diff_fixtures();
    let (out, code) = run_diff(DIFF_OLD, DIFF_OLD);
    assert_eq!(code, 0, "identical should exit 0; stdout:\n{out}");
    assert!(out.contains("tensors: -0 +0 ~0"), "{out}");
    assert!(out.contains("metadata: -0 +0 ~0"), "{out}");
}

#[test]
fn diff_unreadable_path_exits_2() {
    ensure_diff_fixtures();
    let (_out, code) = run_diff(DIFF_OLD, "tests/fixtures/does_not_exist.safetensors");
    assert_eq!(code, 2, "an unreadable path should exit 2");
}
