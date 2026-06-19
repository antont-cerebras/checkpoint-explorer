mod explorer;
mod gguf;
#[cfg(feature = "hdf5")]
mod hdf5;
#[cfg(feature = "hdf5")]
mod hdf5_lz4;
mod health;
mod sample;
mod tree;
mod ui;
mod utils;

use anyhow::{Context, Result};
use clap::Parser;
use std::fs;
use std::path::{Path, PathBuf};

use crate::explorer::{Explorer, OpenRequest, OpenView};

#[derive(Parser)]
#[command(name = "checkpoint-explorer")]
#[command(about = "Interactive explorer for model checkpoints (.safetensors, .gguf, .hdf5)")]
struct Args {
    #[arg(
        help = "Checkpoint files, directories, or glob patterns to explore (e.g., *.safetensors, model-*.gguf, *.hdf5)"
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
        long,
        value_name = "NAME",
        help = "Open a specific tensor on startup (exact name); combine with --dtype/--values/--heatmap/--edge"
    )]
    tensor: Option<String>,

    #[arg(
        long,
        value_name = "DTYPE",
        requires = "tensor",
        value_parser = sample::parse_view_dtype,
        help = "Reinterpret the opened tensor's dtype: u4-packed, u4-lo, u4-hi, i4-packed, i4-lo, i4-hi, f16, bf16, i16, u16, f32, i32, u32, f64, i64, u64, i8, u8, stored"
    )]
    dtype: Option<sample::ViewDtype>,

    #[arg(
        long,
        requires = "tensor",
        conflicts_with = "heatmap",
        help = "Open the opened tensor's numeric values grid"
    )]
    values: bool,

    #[arg(long, requires = "tensor", help = "Open the opened tensor's heatmap")]
    heatmap: bool,

    #[arg(
        long,
        visible_alias = "edges",
        requires = "tensor",
        conflicts_with = "overview",
        help = "Show the first/last edges (padding) submode"
    )]
    edge: bool,

    #[arg(
        long,
        requires = "tensor",
        help = "Show the evenly-spaced overview submode"
    )]
    overview: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.paths.is_empty() {
        eprintln!("Error: Please specify one or more checkpoint files or directories to explore.");
        eprintln!(
            "Usage: checkpoint-explorer <file1.safetensors> [file2.gguf] [model.hdf5] [directory] [*.safetensors] ..."
        );
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
    let open = args.tensor.map(|tensor| OpenRequest {
        tensor,
        view,
        dtype: args.dtype,
        edges,
    });

    let mut explorer = Explorer::new(files, health_reports, open);
    explorer.run()
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
                if matches!(ext, Some("safetensors" | "gguf" | "h5" | "hdf5")) {
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
        &["safetensors", "gguf", "h5", "hdf5"]
    } else {
        &["safetensors", "gguf"]
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
