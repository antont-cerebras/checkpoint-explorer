mod codec;
#[cfg(feature = "hdf5")]
mod convert;
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
mod ui;
mod utils;

use anyhow::{Context, Result};
use clap::{Args as ClapArgs, Parser, Subcommand};
use std::fs;
use std::path::{Path, PathBuf};

use crate::explorer::{Explorer, OpenRequest, OpenView};

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
        value_name = "DTYPE",
        value_parser = sample::parse_view_dtype,
        help = "Reinterpret the opened tensor's dtype: u4-packed, u4-lo, u4-hi, i4-packed, i4-lo, i4-hi, f16, bf16, i16, u16, f32, i32, u32, f64, i64, u64, i8, u8, stored"
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
        visible_alias = "edges",
        conflicts_with = "overview",
        help = "Show the first/last edges (padding) submode"
    )]
    edge: bool,

    #[arg(long, help = "Show the evenly-spaced overview submode")]
    overview: bool,

    #[arg(
        long,
        value_name = "MODE",
        value_parser = ui::parse_stripe_mode,
        help = "Zebra-stripe the numeric grid by rows, cols, or off"
    )]
    zebra: Option<ui::StripeMode>,

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
        help = "Render the requested view once and exit, without entering interactive navigation"
    )]
    exit: bool,
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
        None => run_explore(cli.explore),
    }
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

    let view = if args.values {
        OpenView::Values
    } else if args.heatmap {
        OpenView::Heatmap
    } else {
        OpenView::Detail
    };
    let edges = if args.edge {
        Some(true)
    } else if args.overview {
        Some(false)
    } else {
        None
    };
    // Seed an open request when a tensor is named *or* any view/override flag is
    // given — the latter targets the sole tensor when the checkpoint has one.
    let wants_open = args.tensor.is_some()
        || args.values
        || args.heatmap
        || args.dtype.is_some()
        || args.edge
        || args.overview
        || args.zebra.is_some()
        || args.slice.is_some()
        || args.shape.is_some()
        || args.compute_stats
        || args.exit;
    let open = wants_open.then_some(OpenRequest {
        tensor: args.tensor,
        view,
        dtype: args.dtype,
        edges,
        zebra: args.zebra,
        slice: args.slice,
        shape: args.shape,
        compute_stats: args.compute_stats,
        exit_after: args.exit,
    });

    let mut explorer = Explorer::new(files, health_reports, open, !args.no_preload);
    explorer.run()
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

                // Scan the directory when there is no index, or when the index
                // is stale and pointed at files that no longer exist.
                if !found_from_index {
                    scan_directory(&expanded_path, recursive, &mut files)?;
                }
            }
        }
    }

    // Sort files for consistent ordering
    files.sort();
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
