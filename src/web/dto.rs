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

/// A sampled value grid (`sample::Sample`) — the heatmap / slice payload. Raw
/// stored bits are included only when asked for (the hex/oct/bin value view): each
/// cell as a zero-padded hex string of `raw_width` bits, so the client can reformat
/// to any base via BigInt without u64 precision loss.
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_width: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<Vec<Vec<String>>>,
}

impl SampleDto {
    pub fn from_sample(s: &Sample, include_raw: bool) -> Self {
        let raw_width = s.raw.iter().flatten().next().map(|b| b.width);
        let raw = include_raw.then(|| {
            s.raw
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|b| {
                            let hex_digits = (b.width as usize).div_ceil(4).max(1);
                            format!("{:0hex_digits$x}", b.bits)
                        })
                        .collect()
                })
                .collect()
        });
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
            raw_width,
            raw,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::Duration;

    #[test]
    fn file_node_relativizes_paths() {
        let root = PathBuf::from("/abs/root");
        let node = FileNode::Dir {
            name: "root".into(),
            path: root.clone(),
            expanded: true,
            size: 10,
            files: 1,
            children: vec![FileNode::File {
                name: "a.safetensors".into(),
                path: root.join("sub/a.safetensors"),
                size: 10,
                kind: FileKind::Checkpoint,
            }],
        };
        let web = WebFileNode::from_node(&node, &root);
        let json = serde_json::to_value(&web).unwrap();
        // Root path is empty (relative to itself); child path is root-relative,
        // never the absolute server path.
        assert_eq!(json["path"], "");
        assert_eq!(json["children"][0]["path"], "sub/a.safetensors");
        assert_eq!(json["children"][0]["kind"], "file");
        assert!(!json.to_string().contains("/abs/root"));
    }

    #[test]
    fn stats_dto_converts_duration_and_zero_fraction() {
        let stats = Stats {
            count: 100,
            min: -1.0,
            max: 2.0,
            mean: 0.5,
            std: 1.0,
            zeros: 25,
            nonfinite: 0,
            elapsed: Duration::from_millis(12),
        };
        let dto = StatsDto::from(&stats);
        assert_eq!(dto.zero_fraction, 0.25);
        assert!((dto.elapsed_ms - 12.0).abs() < 1e-6);
    }

    #[test]
    fn histogram_dto_tags_bin_kind() {
        let hist = crate::sample::Histogram {
            bins: HistBins::Range { lo: 0.0, hi: 1.0 },
            counts: vec![3, 5],
            total: 8,
            nonfinite: 1,
            elapsed: Duration::from_millis(4),
        };
        let json = serde_json::to_value(HistogramDto::from(&hist)).unwrap();
        assert_eq!(json["bins"]["type"], "range");
        assert_eq!(json["counts"][1], 5);
        assert_eq!(json["nonfinite"], 1);
    }
}
