//! Serializable wire shapes for the web API — projections of internal types that
//! either leak the server's absolute paths (the file tree) or carry non-serde /
//! heavy fields (`Duration`, raw bits) we don't want on the JSON contract. The
//! tensor tree, layout, and reports serialize directly from core (no DTO needed).

use std::path::Path;

use serde::Serialize;

use crate::filetree::{FileKind, FileNode};
use crate::sample::{HistBins, Histogram, Sample, SampleMode, Stats, ViewDtype};

/// A file-tree node with every `path` relativized to the checkpoint root (never
/// leak the server's absolute paths) — mirrors [`crate::filetree::FileNode`]. The
/// client flattens/folds this the way `filetree::flatten` does.
#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WebFileNode {
    Dir {
        name: String,
        path: String,
        size: u64,
        files: usize,
        children: Vec<WebFileNode>,
    },
    File {
        name: String,
        path: String,
        size: u64,
        file_kind: FileKind,
    },
}

impl WebFileNode {
    /// Project a local `FileNode` tree into the web shape, making each `path`
    /// relative to `root`.
    pub fn from_node(node: &FileNode, root: &Path) -> Self {
        match node {
            FileNode::Dir {
                name,
                path,
                children,
                size,
                files,
                ..
            } => WebFileNode::Dir {
                name: name.clone(),
                path: rel(path, root),
                size: *size,
                files: *files,
                children: children.iter().map(|c| Self::from_node(c, root)).collect(),
            },
            FileNode::File {
                name,
                path,
                size,
                kind,
            } => WebFileNode::File {
                name: name.clone(),
                path: rel(path, root),
                size: *size,
                file_kind: *kind,
            },
        }
    }
}

fn rel(path: &Path, root: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned()
}

/// Whole-tensor statistics (`sample::Stats`) with `Duration` → `elapsed_ms`.
#[derive(Serialize, Clone)]
pub struct StatsDto {
    pub count: u64,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std: f64,
    pub zeros: u64,
    pub nonfinite: u64,
    pub zero_fraction: f64,
    pub elapsed_ms: f64,
}

impl From<&Stats> for StatsDto {
    fn from(s: &Stats) -> Self {
        StatsDto {
            count: s.count,
            min: s.min,
            max: s.max,
            mean: s.mean,
            std: s.std,
            zeros: s.zeros,
            nonfinite: s.nonfinite,
            zero_fraction: s.zero_fraction(),
            elapsed_ms: s.elapsed.as_secs_f64() * 1000.0,
        }
    }
}

/// A sampled value grid (`sample::Sample`) — the heatmap / slice payload. The raw
/// stored bits are dropped (heavy; the client renders from `values`).
#[derive(Serialize)]
pub struct SampleDto {
    pub rows: Vec<usize>,
    pub cols: Vec<usize>,
    pub values: Vec<Vec<f64>>,
    pub min: f64,
    pub max: f64,
    pub total_rows: usize,
    pub total_cols: usize,
    pub slices: usize,
    pub slice: usize,
    pub display_shape: Vec<usize>,
    pub view: String,
    pub mode: String,
    pub overridable: bool,
}

impl From<&Sample> for SampleDto {
    fn from(s: &Sample) -> Self {
        SampleDto {
            rows: s.rows.clone(),
            cols: s.cols.clone(),
            values: s.values.clone(),
            min: s.min,
            max: s.max,
            total_rows: s.total_rows,
            total_cols: s.total_cols,
            slices: s.slices,
            slice: s.slice,
            display_shape: s.display_shape.clone(),
            view: view_label(s.view),
            mode: mode_label(&s.mode),
            overridable: s.overridable,
        }
    }
}

/// A value histogram (`sample::Histogram`) with `Duration` → `elapsed_ms`.
#[derive(Serialize)]
pub struct HistogramDto {
    pub bins: HistBinsDto,
    pub counts: Vec<u64>,
    pub total: u64,
    pub nonfinite: u64,
    pub elapsed_ms: f64,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HistBinsDto {
    /// Integer bins: bin `i` covers `[start + i*step, start + (i+1)*step)`.
    Int { start: i64, step: i64 },
    /// Equal-width bins spanning `[lo, hi]`.
    Range { lo: f64, hi: f64 },
}

impl From<&Histogram> for HistogramDto {
    fn from(h: &Histogram) -> Self {
        HistogramDto {
            bins: match h.bins {
                HistBins::IntBins { start, step } => HistBinsDto::Int { start, step },
                HistBins::Range { lo, hi } => HistBinsDto::Range { lo, hi },
            },
            counts: h.counts.clone(),
            total: h.total,
            nonfinite: h.nonfinite,
            elapsed_ms: h.elapsed.as_secs_f64() * 1000.0,
        }
    }
}

/// The `?dtype=` value that re-selects a view (`stored` when using the real dtype).
pub fn view_label(v: ViewDtype) -> String {
    v.label().unwrap_or("stored").to_string()
}

fn mode_label(m: &SampleMode) -> String {
    match m {
        SampleMode::Grid => "grid",
        SampleMode::Edges { .. } => "edges",
        SampleMode::Window { .. } => "window",
    }
    .to_string()
}
