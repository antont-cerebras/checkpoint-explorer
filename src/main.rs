mod codec;
#[cfg(feature = "hdf5")]
mod convert;
mod diff;
mod explorer;
mod gguf;
#[cfg(feature = "hdf5")]
mod hdf5;
#[cfg(feature = "hdf5")]
mod hdf5_lz4;
#[cfg(feature = "hdf5")]
mod hdf5_zstd;
mod health;
mod npy;
mod sample;
mod tree;
mod tui;
mod ui;
mod utils;

use anyhow::{Context, Result};
use clap::{Args as ClapArgs, Parser, Subcommand};
use std::fs;
use std::path::{Path, PathBuf};

use crate::explorer::{DataLayout, Explorer, OpenRequest, OpenView};

#[derive(Parser)]
#[command(name = "checkpoint-explorer")]
#[command(
    about = "Interactive explorer for model checkpoints (.safetensors, .gguf, .npy, .npz, .hdf5)"
)]
#[command(args_conflicts_with_subcommands = true)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Files/directories/globs to explore (the default action when no
    /// subcommand is given).
    #[command(flatten)]
    explore: ExploreArgs,
}

#[derive(ClapArgs)]
struct ExploreArgs {
    #[arg(
        help = "Checkpoint files, directories, or glob patterns to explore (e.g., *.safetensors, model-*.gguf, *.npy, *.npz, *.hdf5)"
    )]
    paths: Vec<PathBuf>,

    #[arg(
        short,
        long,
        help = "Recursively search directories for checkpoint files"
    )]
    recursive: bool,

    #[arg(
        long = "no-health-check",
        help = "Skip the checkpoint health check (index vs. files on disk)"
    )]
    no_health_check: bool,

    #[arg(
        long = "no-preload",
        help = "Don't compute a tensor's statistics in the background when its detail screen opens (the scan reads the tensor, warming the OS/disk cache to speed up the heatmap/values views especially over NFS; with this flag, statistics are computed only when you press s)"
    )]
    no_preload: bool,

    #[arg(
        long,
        value_name = "NAME",
        help = "Open a specific tensor on startup (exact name); optional when the checkpoint has only one tensor (e.g. a .npy). Combine with --dtype/--shape/--values/--heatmap/--edge"
    )]
    tensor: Option<String>,

    #[arg(
        long,
        value_name = "NAME",
        conflicts_with = "tensor",
        help = "Reveal a metadata entry on startup (exact name, e.g. model.norm.weight.__metadata__) — opens the tree with it selected"
    )]
    metadata: Option<String>,

    #[arg(
        long,
        value_name = "DTYPE",
        value_parser = sample::parse_view_dtype,
        help = "Reinterpret the opened tensor's dtype: u4, i4, unpacked (fused codebook, needs a packing schema), f16, bf16, i16, u16, f32, i32, u32, f64, i64, u64, i8, u8, stored"
    )]
    dtype: Option<sample::ViewDtype>,

    #[arg(
        long,
        conflicts_with = "heatmap",
        help = "Open the opened tensor's numeric values grid"
    )]
    values: bool,

    #[arg(long, help = "Open the opened tensor's heatmap")]
    heatmap: bool,

    #[arg(
        long,
        conflicts_with_all = ["values", "heatmap", "tree"],
        help = "Show the opened tensor's value histogram on its detail screen"
    )]
    histogram: bool,

    #[arg(
        long,
        value_name = "N",
        value_parser = parse_bins,
        conflicts_with_all = ["values", "heatmap", "tree"],
        help = "Histogram bucket count (1–512); implies --histogram"
    )]
    bins: Option<usize>,

    #[arg(
        long,
        conflicts_with_all = ["values", "heatmap"],
        help = "Reveal the tensor highlighted in the tree browser instead of opening a view"
    )]
    tree: bool,

    #[arg(
        long,
        visible_alias = "edges",
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "0.5,0.5",
        value_name = "RFRAC,CFRAC",
        conflicts_with_all = ["overview", "window"],
        help = "Show the first/last edges (padding) submode; optional ROW,COL head/tail split fractions 0..1 (0=first, 1=last, 0.5=balanced)"
    )]
    edge: Option<String>,

    #[arg(long, help = "Show the evenly-spaced overview submode")]
    overview: bool,

    #[arg(
        long,
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "0,0",
        value_name = "ROW,COL",
        conflicts_with_all = ["edge", "overview"],
        help = "Show the contiguous pannable window submode; optional ROW,COL top-left corner (default 0,0)"
    )]
    window: Option<String>,

    #[arg(
        long,
        value_name = "MODE",
        value_parser = ui::parse_stripe_mode,
        help = "Zebra-stripe the numeric grid by rows, cols, or off"
    )]
    zebra: Option<ui::StripeMode>,

    #[arg(
        long,
        value_name = "BASE",
        value_parser = ui::parse_num_base,
        help = "Numeral base for the numeric grid: dec, hex, oct, or bin (non-decimal shows raw stored bits)"
    )]
    base: Option<ui::NumBase>,

    #[arg(
        long,
        value_name = "INDEX",
        help = "Starting slice for a 3D tensor: an index (e.g. 12) or a percentage (e.g. 50%)"
    )]
    slice: Option<String>,

    #[arg(
        long,
        value_name = "DIMS",
        help = "Reinterpret the tensor's shape (same element count); dims like 10,100 or -1,768 (one dim may be -1/*/_ to infer)"
    )]
    shape: Option<String>,

    #[arg(
        long,
        help = "Start computing statistics immediately when opening the detail view (data views always compute them)"
    )]
    compute_stats: bool,

    #[arg(
        long,
        value_name = "STATE",
        value_parser = explorer::parse_tree_state,
        help = "Open the tree fully expanded or collapsed (the `E` / `C` keys): expanded or collapsed"
    )]
    tree_state: Option<explorer::TreeState>,

    #[arg(
        long,
        value_name = "QUERY",
        help = "Open the tree in search mode filtered to QUERY (the `/` key)"
    )]
    search: Option<String>,

    #[arg(
        long,
        help = "Overlay the requested screen's legend (the `l` key) — useful with --plain"
    )]
    legend: bool,

    #[arg(
        long,
        help = "Render the requested view once and exit, without entering interactive navigation"
    )]
    exit: bool,

    #[arg(
        long,
        help = "Render the requested view once as plain text (no colour, no cursor control) and exit — for piping, grep, and end-to-end tests"
    )]
    plain: bool,

    #[arg(
        long,
        help = "Print the CLI command that reopens the requested view (what `y` copies) and exit, instead of rendering"
    )]
    emit_command: bool,
}

#[derive(Subcommand)]
enum Command {
    /// Repack an HDF5 checkpoint into a new file, re-compressing every dataset
    /// with the chosen codec (e.g. gzip/zstd are ~2× smaller than the LZ4 these
    /// checkpoints ship with).
    Convert {
        /// Source `.h5`/`.hdf5` checkpoint.
        input: PathBuf,
        /// Destination file to create.
        output: PathBuf,
        /// Compression codec for the output.
        #[arg(short, long, value_enum, default_value_t = codec::Codec::default())]
        codec: codec::Codec,
        /// Compression level (gzip 0–9, zstd 1–22; ignored for lz4/none).
        /// Defaults to a sensible level for the codec.
        #[arg(short, long)]
        level: Option<u8>,
        /// Streaming buffer per dataset block, e.g. `256M`, `1G` (default 256M).
        #[arg(short, long, default_value = "256M")]
        buffer: String,
        /// Overwrite the output file if it already exists.
        #[arg(short, long)]
        force: bool,
    },

    /// Compare two checkpoints and summarize their structural differences:
    /// tensors (by name, dtype, shape) and metadata (by name, value) that were
    /// added, removed, or changed. Tensor data/values are not compared.
    ///
    /// Exit status follows `diff`: 0 = structurally identical, 1 = differences
    /// found, 2 = trouble (a path couldn't be read).
    Diff {
        /// The baseline ("old") checkpoint — a file, directory, or glob.
        old: PathBuf,
        /// The checkpoint to compare against the baseline ("new").
        new: PathBuf,
        /// Recursively search directories for checkpoint files.
        #[arg(short, long)]
        recursive: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Command::Convert {
            input,
            output,
            codec,
            level,
            buffer,
            force,
        }) => run_convert(&input, &output, codec, level, &buffer, force),
        Some(Command::Diff {
            old,
            new,
            recursive,
        }) => {
            // `diff`-style exit codes (0 same / 1 differ / 2 trouble) don't map to
            // the `Result` convention `main` uses elsewhere, so exit explicitly.
            std::process::exit(run_diff(&old, &new, recursive))
        }
        None => run_explore(cli.explore),
    }
}

/// Compare two checkpoints' structure and print the summary. Returns the process
/// exit code: `0` identical, `1` differences found, `2` trouble (unreadable path).
fn run_diff(old: &Path, new: &Path, recursive: bool) -> i32 {
    let load = |path: &Path| -> Result<diff::CheckpointSummary> {
        let (files, _health) =
            collect_safetensors_files(std::slice::from_ref(&path.to_path_buf()), recursive, true)?;
        if files.is_empty() {
            anyhow::bail!("no checkpoint files found at {}", path.display());
        }
        let (tensors, metadata) = Explorer::gather_checkpoint(&files)?;
        Ok(diff::CheckpointSummary::from_loaded(tensors, metadata))
    };

    let report = match load(old)
        .with_context(|| format!("reading {}", old.display()))
        .and_then(|o| {
            load(new)
                .with_context(|| format!("reading {}", new.display()))
                .map(|n| diff::compare(&o, &n))
        }) {
        Ok(report) => report,
        Err(e) => {
            eprintln!("checkpoint-explorer diff: {e:#}");
            return 2;
        }
    };

    print!(
        "{}",
        report.render(&old.display().to_string(), &new.display().to_string())
    );
    i32::from(report.has_differences())
}

/// Parse a `ROW,COL` pair of non-negative integers (the `--window` top-left).
/// Parse and bound the `--bins` histogram bucket count to `1..=512`.
fn parse_bins(s: &str) -> std::result::Result<usize, String> {
    match s.trim().parse::<usize>() {
        Ok(n) if (1..=512).contains(&n) => Ok(n),
        Ok(_) => Err("must be between 1 and 512".to_string()),
        Err(_) => Err(format!("expected a whole number, got '{s}'")),
    }
}

fn parse_offset_pair(s: &str) -> Result<(usize, usize)> {
    let (r, c) = s
        .split_once(',')
        .with_context(|| format!("expected ROW,COL (two integers), got '{s}'"))?;
    let row = r
        .trim()
        .parse()
        .with_context(|| format!("invalid row '{r}'"))?;
    let col = c
        .trim()
        .parse()
        .with_context(|| format!("invalid col '{c}'"))?;
    Ok((row, col))
}

/// Parse a `RFRAC,CFRAC` pair of fractions in `0..=1` (the `--edge` head/tail
/// split: 0 keeps only the first indices, 1 only the last, 0.5 is balanced).
fn parse_fraction_pair(s: &str) -> Result<(f32, f32)> {
    let (r, c) = s
        .split_once(',')
        .with_context(|| format!("expected RFRAC,CFRAC (two fractions 0..1), got '{s}'"))?;
    let row: f32 = r
        .trim()
        .parse()
        .with_context(|| format!("invalid row '{r}'"))?;
    let col: f32 = c
        .trim()
        .parse()
        .with_context(|| format!("invalid col '{c}'"))?;
    if !(0.0..=1.0).contains(&row) || !(0.0..=1.0).contains(&col) {
        anyhow::bail!("edge split fractions must be between 0 and 1, got '{s}'");
    }
    Ok((row, col))
}

fn run_explore(args: ExploreArgs) -> Result<()> {
    if args.paths.is_empty() {
        eprintln!("Error: Please specify one or more checkpoint files or directories to explore.");
        eprintln!(
            "Usage: checkpoint-explorer <file1.safetensors> [file2.gguf] [model.hdf5] [directory] [*.safetensors] ..."
        );
        eprintln!("       checkpoint-explorer convert <input.hdf5> <output.hdf5>");
        std::process::exit(1);
    }

    let (files, health_reports) =
        collect_safetensors_files(&args.paths, args.recursive, args.no_health_check)?;

    if files.is_empty() {
        eprintln!("Error: No checkpoint files found in the specified paths.");
        std::process::exit(1);
    }

    // Reading HDF5 needs the `hdf5` build feature; without it these files would
    // load as empty and the tree would misleadingly read "0 tensors, 0 params,
    // 0 B". Say so plainly instead. Directory scans already skip HDF5 when the
    // feature is off, so this only fires for files the user named explicitly.
    #[cfg(not(feature = "hdf5"))]
    {
        let hdf5: Vec<&PathBuf> = files
            .iter()
            .filter(|p| matches!(p.extension().and_then(|s| s.to_str()), Some("h5" | "hdf5")))
            .collect();
        if !hdf5.is_empty() {
            eprintln!(
                "Error: this build of checkpoint-explorer was compiled without HDF5 support, so it cannot read:"
            );
            for path in &hdf5 {
                eprintln!("  {}", path.display());
            }
            eprintln!();
            eprintln!("Rebuild and reinstall with the `hdf5` feature, e.g.:");
            eprintln!("  cargo install --path . --features hdf5");
            std::process::exit(1);
        }
    }

    // Flags that target the tree browser rather than a tensor view: with no
    // `--tensor` (and no data view), they make the tree the opened screen, so
    // e.g. `--expand-all` or `--legend` alone don't demand a tensor.
    let tree_oriented =
        args.tree || args.tree_state.is_some() || args.search.is_some() || args.legend;
    let view = if args.values {
        OpenView::Values
    } else if args.heatmap {
        OpenView::Heatmap
    } else if args.tree || (tree_oriented && args.tensor.is_none()) {
        OpenView::Tree
    } else {
        OpenView::Detail
    };
    let layout = if args.window.is_some() {
        Some(DataLayout::Window)
    } else if args.edge.is_some() {
        Some(DataLayout::Edges)
    } else if args.overview {
        Some(DataLayout::Overview)
    } else {
        None
    };
    // Position within the layout: the window's top-left corner, or the edges
    // head/tail split — parsed from the optional `--window`/`--edge` value.
    let window_at = args.window.as_deref().map(parse_offset_pair).transpose()?;
    let edge_split = args.edge.as_deref().map(parse_fraction_pair).transpose()?;
    // Seed an open request when a tensor is named *or* any view/override flag is
    // given — the latter targets the sole tensor when the checkpoint has one.
    let wants_open = args.tensor.is_some()
        || args.metadata.is_some()
        || args.values
        || args.heatmap
        || args.tree
        || args.dtype.is_some()
        || args.edge.is_some()
        || args.overview
        || args.window.is_some()
        || args.zebra.is_some()
        || args.base.is_some()
        || args.slice.is_some()
        || args.shape.is_some()
        || args.compute_stats
        || args.histogram
        || args.bins.is_some()
        || args.tree_state.is_some()
        || args.search.is_some()
        || args.legend
        || args.exit;
    let open = wants_open.then_some(OpenRequest {
        tensor: args.tensor,
        metadata: args.metadata,
        view,
        histogram: args.histogram,
        bins: args.bins,
        dtype: args.dtype,
        layout,
        window_at,
        edge_split,
        zebra: args.zebra,
        base: args.base,
        slice: args.slice,
        shape: args.shape,
        compute_stats: args.compute_stats,
        tree_state: args.tree_state,
        search: args.search,
        legend: args.legend,
        exit_after: args.exit,
    });

    let mut explorer = Explorer::new(files, health_reports, open, !args.no_preload);
    if args.emit_command {
        explorer.render_plain(true)
    } else if args.plain {
        explorer.render_plain(false)
    } else {
        explorer.run()
    }
}

#[cfg(feature = "hdf5")]
fn run_convert(
    input: &Path,
    output: &Path,
    codec: codec::Codec,
    level: Option<u8>,
    buffer: &str,
    force: bool,
) -> Result<()> {
    use anyhow::bail;
    use std::io::Write;

    let ext = input.extension().and_then(|e| e.to_str());
    if !matches!(ext, Some("h5" | "hdf5")) {
        bail!(
            "convert only supports HDF5 inputs (.h5/.hdf5), got: {}",
            input.display()
        );
    }
    // Refuse to read and write the same file (checked before --force removes the
    // output, so we never delete the input).
    if std::path::absolute(input).ok() == std::path::absolute(output).ok()
        && std::path::absolute(input).is_ok()
    {
        bail!("input and output are the same file: {}", input.display());
    }
    // Warn when the target codec is what the source already uses (a re-encode;
    // a plain file copy would be equivalent).
    if convert::source_codec(input) == Some(codec) {
        eprintln!(
            "warning: source is already {}; repacking just re-encodes it — a plain copy would be equivalent",
            codec.label()
        );
    }
    if force && output.exists() {
        fs::remove_file(output)
            .with_context(|| format!("removing existing {}", output.display()))?;
    }

    let level = codec.clamp_level(level.unwrap_or_else(|| codec.default_level()));
    let buffer_bytes = utils::parse_size(buffer).map_err(anyhow::Error::msg)?;
    let opts = convert::Options {
        codec,
        level,
        buffer_bytes,
    };
    let level_note = if codec.uses_level() {
        format!(" level {level}")
    } else {
        String::new()
    };
    eprintln!(
        "Repacking {} → {} ({}{level_note}, {} buffer)",
        input.display(),
        output.display(),
        codec.label(),
        utils::format_size(buffer_bytes),
    );

    let mut stderr = std::io::stderr();
    let report = convert::convert_hdf5(input, output, &opts, |done, total, name| {
        let bar = progress_bar(done, total, 28);
        let _ = write!(stderr, "\r{bar} [{done}/{total}] {name:.<48}\x1b[K");
        let _ = stderr.flush();
    })?;
    eprintln!("\rDone: {}\x1b[K", report.summary(codec));
    Ok(())
}

/// A `[####----]` progress bar of the given width.
#[cfg(feature = "hdf5")]
fn progress_bar(done: usize, total: usize, width: usize) -> String {
    let filled = (done * width).checked_div(total).unwrap_or(0);
    format!(
        "[{}{}]",
        "#".repeat(filled),
        "-".repeat(width.saturating_sub(filled))
    )
}

#[cfg(not(feature = "hdf5"))]
fn run_convert(
    _input: &Path,
    _output: &Path,
    _codec: codec::Codec,
    _level: Option<u8>,
    _buffer: &str,
    _force: bool,
) -> Result<()> {
    anyhow::bail!("`convert` requires building with `--features hdf5`")
}

fn collect_safetensors_files(
    paths: &[PathBuf],
    recursive: bool,
    no_health_check: bool,
) -> Result<(Vec<PathBuf>, Vec<health::HealthReport>)> {
    let mut files = Vec::new();
    let mut health_reports = Vec::new();

    for path in paths {
        // Try to expand as glob pattern
        let expanded_paths: Vec<PathBuf> = match glob::glob(&path.to_string_lossy()) {
            Ok(paths) => paths.filter_map(Result::ok).collect(),
            Err(_) => vec![path.clone()], // Not a valid glob, treat as literal path
        };

        // Process each expanded path
        for expanded_path in expanded_paths {
            if !expanded_path.exists() {
                eprintln!("Warning: Path does not exist: {}", expanded_path.display());
                continue;
            }

            if expanded_path.is_file() {
                let ext = expanded_path.extension().and_then(|s| s.to_str());
                if matches!(
                    ext,
                    Some("safetensors" | "gguf" | "h5" | "hdf5" | "npy" | "npz")
                ) {
                    files.push(expanded_path.clone());
                } else {
                    eprintln!(
                        "Warning: Skipping unsupported file: {}",
                        expanded_path.display()
                    );
                }
            } else if expanded_path.is_dir() {
                // Check for SafeTensors index file first
                let index_path = expanded_path.join("model.safetensors.index.json");
                let mut found_from_index = false;
                if index_path.exists() {
                    let index_files = parse_safetensors_index(&index_path)?;
                    let mut missing = Vec::new();
                    for file in index_files {
                        let full_path = expanded_path.join(&file);
                        if full_path.exists() {
                            files.push(full_path);
                            found_from_index = true;
                        } else {
                            missing.push(file);
                        }
                    }
                    if !missing.is_empty() {
                        eprintln!(
                            "Warning: {} file(s) listed in {} were not found on disk (e.g. {}).",
                            missing.len(),
                            index_path.display(),
                            missing[0],
                        );
                    }
                    if !found_from_index {
                        eprintln!(
                            "Warning: index file references no existing files (it may be stale); scanning {} directly instead.",
                            expanded_path.display()
                        );
                    }

                    // Health check: compare the index against the files on disk
                    // and record any mismatch to surface in the UI.
                    if !no_health_check
                        && let Ok(report) = health::check(&expanded_path, &index_path)
                        && report.has_issues()
                    {
                        health_reports.push(report);
                    }
                }

                // Always scan the directory as well, so files present on disk
                // but missing from the index — a partially-stale index, e.g.
                // extra `codebooks`/`qscales` shards — still show up, alongside
                // the no-index and fully-stale cases. Duplicates (a shard listed
                // in the index *and* found by the scan) are removed below.
                scan_directory(&expanded_path, recursive, &mut files)?;
            }
        }
    }

    // Sort for consistent ordering and drop duplicates — the same file can be
    // collected both from the index and the directory scan (identical paths).
    files.sort();
    files.dedup();
    Ok((files, health_reports))
}

fn scan_directory(dir: &Path, recursive: bool, files: &mut Vec<PathBuf>) -> Result<()> {
    // HDF5 is only scanned for when compiled in, to avoid surfacing files the
    // build cannot read.
    let exts: &[&str] = if cfg!(feature = "hdf5") {
        &["safetensors", "gguf", "h5", "hdf5", "npy", "npz"]
    } else {
        &["safetensors", "gguf", "npy", "npz"]
    };
    let glob_prefix = if recursive { "**/" } else { "" };
    let patterns: Vec<String> = exts
        .iter()
        .map(|ext| format!("{}/{glob_prefix}*.{ext}", dir.display()))
        .collect();

    for pattern in patterns {
        for entry in glob::glob(&pattern).context("Failed to read glob pattern")? {
            match entry {
                Ok(file_path) => files.push(file_path),
                Err(e) => eprintln!("Warning: Error reading file: {e}"),
            }
        }
    }

    Ok(())
}

fn parse_safetensors_index(index_path: &PathBuf) -> Result<Vec<String>> {
    let content = fs::read_to_string(index_path)
        .with_context(|| format!("Failed to read index file: {}", index_path.display()))?;

    let index: serde_json::Value = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse index file: {}", index_path.display()))?;

    let mut files = Vec::new();

    if let Some(weight_map) = index.get("weight_map").and_then(|v| v.as_object()) {
        for file_name in weight_map.values() {
            if let Some(file_str) = file_name.as_str()
                && !files.iter().any(|existing| existing == file_str)
            {
                files.push(file_str.to_string());
            }
        }
    }

    files.sort();
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// A directory whose index lists only some of the `.safetensors` on disk (a
    /// partially-stale index) must still surface the extra files — the bug where
    /// `codebooks`/`qscales` shards were silently dropped.
    #[test]
    fn collects_extra_files_absent_from_a_stale_index() {
        let dir = std::env::temp_dir().join("ckpt_explorer_stale_index_test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        // Three shards on disk; the index references only the first.
        fs::write(dir.join("model.safetensors"), b"x").unwrap();
        fs::write(dir.join("codebooks.safetensors"), b"x").unwrap();
        fs::write(dir.join("qscales.safetensors"), b"x").unwrap();
        fs::write(
            dir.join("model.safetensors.index.json"),
            br#"{"weight_map": {"w": "model.safetensors"}}"#,
        )
        .unwrap();

        let (files, _) =
            collect_safetensors_files(std::slice::from_ref(&dir), false, true).unwrap();
        let names: Vec<String> = files
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();

        for want in [
            "model.safetensors",
            "codebooks.safetensors",
            "qscales.safetensors",
        ] {
            assert!(
                names.iter().any(|n| n == want),
                "{want} should be collected; got {names:?}"
            );
        }
        // The shard listed in the index *and* found by the scan must appear once.
        let unique: std::collections::HashSet<_> = files.iter().collect();
        assert_eq!(files.len(), unique.len(), "duplicate paths: {names:?}");

        let _ = fs::remove_dir_all(&dir);
    }
}
