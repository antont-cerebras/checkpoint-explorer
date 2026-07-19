//! Overall checkpoint statistics — the `s` popup on the tree.
//!
//! A cheap, header-only aggregation over the already-loaded tensor metadata:
//! file/shard count, parameter and byte totals, the largest/smallest/typical
//! tensor, the dtype mix, and the repeated layer / MoE-expert structure of
//! transformer checkpoints. Nothing here reads tensor data — it's all derived
//! from the shapes and dtypes already in memory, so the popup is instant even on
//! multi-GB checkpoints.

use std::collections::{BTreeSet, HashMap};

use crate::check::{TensorRole, classify_role, expert_index, proj_category, split_layer_index};
use crate::tree::{Storage, TensorInfo};

/// Section glyphs, matching the tree view's (`▦` tensors, `≡` layers) so the
/// popup reads like the rest of the UI rather than a flat table.
pub(crate) const GLYPH_FILES: &str = "▤";
pub(crate) const GLYPH_TENSORS: &str = "▦";
pub(crate) const GLYPH_LAYERS: &str = "≡";
pub(crate) const GLYPH_EXPERTS: &str = "◆";

// ── Per-layer graph geometry + math ─────────────────────────────────────────
// Pure functions (no ratatui) so the plain report and the styled view agree and
// are unit-testable without a `Frame`.

/// Caps the number of chart columns. A pure function of the layer count (not the
/// terminal width), so headless `--stats` snapshots are stable everywhere; models
/// with ≤ `GRAPH_W` layers render one column per layer, larger stacks are bucketed.
pub(crate) const GRAPH_W: usize = 72;
/// Width of the single 100%-stacked composition bar.
pub(crate) const BAR_W: usize = 40;
/// The eight block-eighths for the scalar sparklines, low → high.
pub(crate) const SPARK_BLOCKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
/// Composition segment glyphs (attention / ffn-experts / other) — distinct
/// characters, not just colour, so the monochrome `--stats` / `r` report is legible.
pub(crate) const SHADES: [char; 3] = ['█', '▓', '░'];

/// Number of chart cells for a series of length `n`, capped at `width`.
fn cell_count(n: usize, width: usize) -> usize {
    n.min(width).max(1)
}

/// The half-open layer range `[start, end)` feeding chart cell `c`. Integer math;
/// with `c < cells ≤ n` the range is always non-empty.
fn bucket_bounds(n: usize, cells: usize, c: usize) -> (usize, usize) {
    (c * n / cells, (c + 1) * n / cells)
}

/// Bucket a series to `≤ width` cells, averaging each bucket.
pub(crate) fn bucket_means(values: &[usize], width: usize) -> Vec<f64> {
    let n = values.len();
    let cells = cell_count(n, width);
    (0..cells)
        .map(|c| {
            let (s, e) = bucket_bounds(n, cells, c);
            values[s..e].iter().sum::<usize>() as f64 / (e - s) as f64
        })
        .collect()
}

/// Sparkline glyph indices (`0..=7`) for `values` at `width`, plus the raw
/// min / max (the true per-layer extremes, for the range label). Min-anchored so
/// small variation is visible; all-equal → every cell at mid (index 3, no div-by-0).
pub(crate) fn spark_levels(values: &[usize], width: usize) -> (Vec<usize>, usize, usize) {
    let lo = values.iter().copied().min().unwrap_or(0);
    let hi = values.iter().copied().max().unwrap_or(0);
    let (flo, fhi) = (lo as f64, hi as f64);
    let levels = bucket_means(values, width)
        .iter()
        .map(|&x| {
            if fhi <= flo {
                3
            } else {
                ((x - flo) / (fhi - flo) * 7.0).round().clamp(0.0, 7.0) as usize
            }
        })
        .collect();
    (levels, lo, hi)
}

/// The sparkline string for `values` at `width`.
pub(crate) fn spark_string(values: &[usize], width: usize) -> String {
    spark_levels(values, width)
        .0
        .iter()
        .map(|&l| SPARK_BLOCKS[l])
        .collect()
}

/// Split `parts` (attn, ffn, other) into row counts summing to exactly `height`
/// (largest-remainder), or `[0, 0, 0]` for an all-zero column.
pub(crate) fn alloc_rows(parts: [usize; 3], height: usize) -> [usize; 3] {
    let total: usize = parts.iter().sum();
    if total == 0 {
        return [0, 0, 0];
    }
    let raw = parts.map(|p| p as f64 / total as f64 * height as f64);
    let mut out = raw.map(|r| r.floor() as usize);
    let mut leftover = height - out.iter().sum::<usize>();
    // Hand the leftover rows to the largest fractional parts (ties → lowest index).
    let mut order = [0usize, 1, 2];
    order.sort_by(|&i, &j| {
        let (fi, fj) = (raw[i] - raw[i].floor(), raw[j] - raw[j].floor());
        fj.partial_cmp(&fi)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(i.cmp(&j))
    });
    for &i in &order {
        if leftover == 0 {
            break;
        }
        out[i] += 1;
        leftover -= 1;
    }
    out
}

/// A 100%-stacked composition bar: `width` cells split `[attn, ffn, other]` by
/// share (largest-remainder), each cell the matching [`SHADES`] glyph.
pub(crate) fn composition_bar(totals: [usize; 3], width: usize) -> String {
    alloc_rows(totals, width)
        .iter()
        .zip(SHADES)
        .flat_map(|(&n, ch)| std::iter::repeat_n(ch, n))
        .collect()
}

/// One named tensor with its logical size — for the largest / smallest rows.
#[derive(Debug, Clone)]
pub struct NamedSize {
    pub name: String,
    pub bytes: usize,
}

/// The repeated transformer-layer stack (`…layers.<i>.…`), aggregated.
#[derive(Debug, Clone)]
pub struct LayerStats {
    /// Number of layers (highest layer index + 1).
    pub count: usize,
    /// Total parameters across all layers.
    pub params: usize,
    /// Total logical bytes across all layers.
    pub bytes: usize,
}

impl LayerStats {
    /// Average parameters in a single layer.
    pub fn params_each(&self) -> usize {
        self.params / self.count.max(1)
    }
    /// Average bytes in a single layer.
    pub fn bytes_each(&self) -> usize {
        self.bytes / self.count.max(1)
    }
}

/// One layer's aggregate, with a byte-composition split for the stacked chart.
/// (The layer index is the row's position in [`PerLayerStats::rows`].)
#[derive(Debug, Clone, Default)]
pub struct LayerRow {
    /// Tensor count in this layer.
    pub tensors: usize,
    /// Total parameters in this layer.
    pub params: usize,
    /// Total logical bytes in this layer.
    pub bytes: usize,
    pub attn_bytes: usize,
    /// MLP + expert bytes.
    pub ffn_bytes: usize,
    pub other_bytes: usize,
}

/// The per-layer series behind the stats graphs. Present only when a canonical
/// layer family exists (`None` for a dense checkpoint with no indexed stack).
#[derive(Debug, Clone)]
pub struct PerLayerStats {
    /// One row per index `0..count`, in order (so index ↔ position always align).
    pub rows: Vec<LayerRow>,
}

impl PerLayerStats {
    pub fn bytes_series(&self) -> Vec<usize> {
        self.rows.iter().map(|r| r.bytes).collect()
    }
    pub fn params_series(&self) -> Vec<usize> {
        self.rows.iter().map(|r| r.params).collect()
    }
    pub fn tensor_series(&self) -> Vec<usize> {
        self.rows.iter().map(|r| r.tensors).collect()
    }

    /// Total `[attention, ffn/experts, other]` bytes across every layer.
    pub fn composition_totals(&self) -> [usize; 3] {
        self.rows.iter().fold([0; 3], |[a, f, o], r| {
            [a + r.attn_bytes, f + r.ffn_bytes, o + r.other_bytes]
        })
    }

    /// Aggregate the per-layer series over the canonical layer family — the same
    /// family [`LayerStats`] uses, so the graphs line up with `Layers ×N`.
    fn compute(tensors: &[TensorInfo]) -> Option<PerLayerStats> {
        let (prefix, count) = canonical_family(tensors)?;
        let mut rows: Vec<LayerRow> = vec![LayerRow::default(); count];
        for t in tensors {
            if let Some((p, idx, suffix)) = split_layer_index(&t.name)
                && p == prefix
                && idx < count
            {
                let r = &mut rows[idx];
                r.tensors += 1;
                r.params += t.num_elements;
                r.bytes += t.size_bytes;
                match classify_role(&suffix) {
                    TensorRole::Attention => r.attn_bytes += t.size_bytes,
                    TensorRole::Ffn => r.ffn_bytes += t.size_bytes,
                    TensorRole::Other => r.other_bytes += t.size_bytes,
                }
            }
        }
        Some(PerLayerStats { rows })
    }
}

/// How MoE experts are laid out on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertStorage {
    /// Each expert is its own tensor: `…experts.<e>.down_proj.weight`.
    Unfused,
    /// Experts are stacked into one tensor per projection (a leading expert
    /// dimension), with no per-expert index in the name.
    Fused,
}

impl ExpertStorage {
    pub fn label(self) -> &'static str {
        match self {
            ExpertStorage::Unfused => "unfused (per-expert tensors)",
            ExpertStorage::Fused => "fused (stacked tensors)",
        }
    }
}

/// One expert projection category (`down_proj` / `gate_proj` / `up_proj` /
/// `gate_up_proj`) aggregated across every expert and layer.
#[derive(Debug, Clone)]
pub struct ExpertCategory {
    pub name: String,
    /// Total logical bytes in this projection across all experts.
    pub bytes: usize,
}

/// MoE expert structure — present only when the checkpoint has experts.
#[derive(Debug, Clone)]
pub struct ExpertStats {
    /// Experts per layer.
    pub per_layer: usize,
    pub storage: ExpertStorage,
    /// gate & up projections combined into one tensor (`gate_up_proj` /
    /// `gate_proj__up_proj`), a common MoE fusion.
    pub gate_up_fused: bool,
    /// Total parameters across all experts (every layer).
    pub params: usize,
    /// Total logical bytes across all experts.
    pub bytes: usize,
    /// Layers that carry experts — the divisor (with `per_layer`) for a single
    /// expert's average.
    pub layers: usize,
    /// Per-projection breakdown (down/gate/up/gate_up), in that canonical order.
    pub by_category: Vec<ExpertCategory>,
}

impl ExpertStats {
    fn divisor(&self) -> usize {
        (self.layers.max(1) * self.per_layer.max(1)).max(1)
    }
    /// Average parameters in a single expert.
    pub fn params_each(&self) -> usize {
        self.params / self.divisor()
    }
    /// Average bytes in a single expert.
    pub fn bytes_each(&self) -> usize {
        self.bytes / self.divisor()
    }
}

/// A dtype and how much of the checkpoint it accounts for.
#[derive(Debug, Clone)]
pub struct DtypeStat {
    pub dtype: String,
    pub count: usize,
    pub bytes: usize,
}

/// Per-file (per-shard) logical-size distribution — the tensor-size stats, but
/// over whole files. Sizes are logical (Σ of each file's tensor `size_bytes`).
#[derive(Debug, Clone)]
pub struct FileStats {
    /// Number of distinct files the tensors were read from; 1 for a single file.
    pub count: usize,
    /// Singular noun for a file — "safetensors file" vs. a plain "file".
    pub noun: &'static str,
    /// Largest / smallest file by logical size, named (the shard basename) like
    /// the per-tensor rows. `None` only when there are no files.
    pub largest: Option<NamedSize>,
    pub smallest: Option<NamedSize>,
    pub mean: usize,
    pub median: usize,
}

/// One shard file's on-disk footprint: its apparent size vs. the blocks the
/// filesystem actually allocated. `allocated < apparent` means the filesystem
/// (e.g. ZFS/btrfs transparent compression, or sparse-file holes) is squeezing
/// it — a saving invisible to the logical byte counts above.
#[derive(Debug, Clone)]
pub struct ShardDisk {
    /// The shard's basename, for display.
    pub name: String,
    /// Apparent size (`st_size`) — the nominal file length.
    pub apparent: u64,
    /// Bytes the filesystem actually allocated (`st_blocks × 512`).
    pub allocated: u64,
}

/// Filesystem allocation across the checkpoint's shard files — the true on-disk
/// footprint, gathered from the OS `stat` (`st_blocks`) rather than the logical
/// byte counts. `None` when it can't be measured (remote `s3://`, a failed stat,
/// or a non-Unix host).
#[derive(Debug, Clone)]
pub struct DiskUsage {
    pub shards: Vec<ShardDisk>,
    pub total_apparent: u64,
    pub total_allocated: u64,
}

impl DiskUsage {
    /// Build from per-shard rows, summing the totals. `None` if empty.
    pub fn from_shards(shards: Vec<ShardDisk>) -> Option<DiskUsage> {
        if shards.is_empty() {
            return None;
        }
        let total_apparent = shards.iter().map(|s| s.apparent).sum();
        let total_allocated = shards.iter().map(|s| s.allocated).sum();
        Some(DiskUsage {
            shards,
            total_apparent,
            total_allocated,
        })
    }

    /// Stat local files through the OS (`st_blocks × 512`). Paths that don't stat
    /// (e.g. a remote scp-form path, or one that's since vanished) are skipped.
    #[cfg(unix)]
    pub fn from_local(paths: &[&str]) -> Option<DiskUsage> {
        use std::os::unix::fs::MetadataExt;
        let shards = paths
            .iter()
            .filter_map(|p| {
                let md = std::fs::metadata(p).ok()?;
                Some(ShardDisk {
                    name: shard_name(p),
                    apparent: md.len(),
                    allocated: md.blocks() * 512,
                })
            })
            .collect();
        DiskUsage::from_shards(shards)
    }

    #[cfg(not(unix))]
    pub fn from_local(_paths: &[&str]) -> Option<DiskUsage> {
        None
    }
}

/// A path's final component — the shard's filename. Splits on `/` and `:` so an
/// scp-form remote path (`host:/dir/shard.safetensors`) also reduces to the name.
pub fn shard_name(path: &str) -> String {
    path.rsplit(['/', ':']).next().unwrap_or(path).to_string()
}

/// Everything the `s` popup shows, computed once when the popup opens.
#[derive(Debug, Clone)]
pub struct CheckpointStats {
    /// Per-file (shard) count and size distribution.
    pub files: FileStats,
    pub n_tensors: usize,
    pub params: usize,
    /// Logical (uncompressed) bytes: Σ `size_bytes`.
    pub logical_bytes: usize,
    /// On-disk bytes: Σ `on_disk_size()` (equal to logical unless compressed).
    pub disk_bytes: usize,
    /// True when any tensor is stored compressed, so `disk_bytes < logical_bytes`
    /// is a meaningful compression ratio (HDF5).
    pub compressed: bool,
    pub largest: Option<NamedSize>,
    pub smallest: Option<NamedSize>,
    pub mean_bytes: usize,
    pub median_bytes: usize,
    /// dtypes, largest share first.
    pub dtypes: Vec<DtypeStat>,
    pub layers: Option<LayerStats>,
    /// Per-layer series (tensor count / params / bytes / composition) for the
    /// stats graphs — `Some` whenever `layers` is.
    pub per_layer: Option<PerLayerStats>,
    pub experts: Option<ExpertStats>,
    /// `config.json`'s `model_type`, when a config was found.
    pub model_type: Option<String>,
    /// True on-disk footprint from the filesystem, when measurable.
    pub disk: Option<DiskUsage>,
}

impl CheckpointStats {
    pub fn compute(
        tensors: &[TensorInfo],
        config: Option<&crate::config::ModelConfig>,
        disk: Option<DiskUsage>,
    ) -> CheckpointStats {
        let n_tensors = tensors.len();
        let params: usize = tensors.iter().map(|t| t.num_elements).sum();
        let logical_bytes: usize = tensors.iter().map(|t| t.size_bytes).sum();
        let disk_bytes: usize = tensors.iter().map(TensorInfo::on_disk_size).sum();
        let compressed = tensors
            .iter()
            .any(|t| matches!(t.storage, Storage::Compressed { .. }));

        // Per-file (shard) logical size = Σ of that file's tensor bytes.
        let mut per_file: HashMap<&str, usize> = HashMap::new();
        for t in tensors {
            *per_file.entry(t.source_path.as_str()).or_default() += t.size_bytes;
        }
        let noun = if per_file
            .keys()
            .next()
            .and_then(|p| p.rsplit('.').next())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"))
        {
            "safetensors file"
        } else {
            "file"
        };
        // Sort by size, breaking ties by path so the named largest/smallest are
        // deterministic (the map's iteration order isn't).
        let mut sized: Vec<(&str, usize)> = per_file.into_iter().collect();
        sized.sort_unstable_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(b.0)));
        let named = |&(path, bytes): &(&str, usize)| NamedSize {
            name: shard_name(path),
            bytes,
        };
        let files = FileStats {
            count: sized.len(),
            noun,
            largest: sized.last().map(named),
            smallest: sized.first().map(named),
            mean: logical_bytes.checked_div(sized.len()).unwrap_or(0),
            median: sized.get(sized.len() / 2).map(|&(_, s)| s).unwrap_or(0),
        };

        let largest = tensors
            .iter()
            .max_by_key(|t| t.size_bytes)
            .map(|t| NamedSize {
                name: t.name.clone(),
                bytes: t.size_bytes,
            });
        let smallest = tensors
            .iter()
            .min_by_key(|t| t.size_bytes)
            .map(|t| NamedSize {
                name: t.name.clone(),
                bytes: t.size_bytes,
            });
        let mean_bytes = logical_bytes.checked_div(n_tensors).unwrap_or(0);
        let median_bytes = {
            let mut sizes: Vec<usize> = tensors.iter().map(|t| t.size_bytes).collect();
            sizes.sort_unstable();
            sizes.get(sizes.len() / 2).copied().unwrap_or(0)
        };

        // dtype breakdown, biggest byte-share first (ties broken by name).
        let mut dmap: HashMap<&str, (usize, usize)> = HashMap::new();
        for t in tensors {
            let e = dmap.entry(t.dtype.as_str()).or_insert((0, 0));
            e.0 += 1;
            e.1 += t.size_bytes;
        }
        let mut dtypes: Vec<DtypeStat> = dmap
            .into_iter()
            .map(|(d, (count, bytes))| DtypeStat {
                dtype: d.to_string(),
                count,
                bytes,
            })
            .collect();
        dtypes.sort_by(|a, b| b.bytes.cmp(&a.bytes).then_with(|| a.dtype.cmp(&b.dtype)));

        CheckpointStats {
            files,
            n_tensors,
            params,
            logical_bytes,
            disk_bytes,
            compressed,
            largest,
            smallest,
            mean_bytes,
            median_bytes,
            dtypes,
            layers: layer_stats(tensors),
            per_layer: PerLayerStats::compute(tensors),
            experts: expert_stats(tensors, config),
            model_type: config.and_then(|c| c.model_type.clone()),
            disk,
        }
    }

    /// A plain-text rendering of the stats — what the popup's `r` copies. Mirrors
    /// the on-screen sections so the copied text reads the same, minus styling.
    pub fn render(&self, shards_expanded: bool) -> String {
        use crate::utils::{format_parameters, format_size};
        const LW: usize = 12;
        // A guaranteed separator space follows the padded label, so a full-width
        // label (e.g. "Architecture") still has a gap before its value.
        let row = |label: &str, value: String| format!("  {label:<LW$} {value}");
        let each_total = |each: String, total: String| format!("{each} each · {total} total");

        let mut out: Vec<String> = vec!["Checkpoint stats".into()];

        // Overview.
        out.push(String::new());
        out.push("Overview".into());
        if let Some(mt) = &self.model_type {
            out.push(row("Architecture", mt.clone()));
        }
        out.push(row("Parameters", format_parameters(self.params)));
        out.push(row(
            "Size",
            if self.compressed && self.disk_bytes > 0 {
                format!(
                    "{} on disk · {} logical ({:.2}× smaller)",
                    format_size(self.disk_bytes),
                    format_size(self.logical_bytes),
                    self.logical_bytes as f64 / self.disk_bytes as f64
                )
            } else {
                format_size(self.logical_bytes)
            },
        ));

        // Files (per-shard logical size distribution).
        out.push(String::new());
        out.push(format!(
            "{GLYPH_FILES} Files  ×{} {}",
            self.files.count, self.files.noun
        ));
        if let Some(l) = &self.files.largest {
            out.push(row(
                "Largest",
                format!("{:<9} {}", format_size(l.bytes), l.name),
            ));
        }
        if let Some(sm) = &self.files.smallest {
            out.push(row(
                "Smallest",
                format!("{:<9} {}", format_size(sm.bytes), sm.name),
            ));
        }
        out.push(row("Average", format_size(self.files.mean)));
        out.push(row("Median", format_size(self.files.median)));

        // Tensors (count + size distribution).
        out.push(String::new());
        out.push(format!("{GLYPH_TENSORS} Tensors  ×{}", self.n_tensors));
        if let Some(l) = &self.largest {
            out.push(row(
                "Largest",
                format!("{:<9} {}", format_size(l.bytes), l.name),
            ));
        }
        if let Some(sm) = &self.smallest {
            out.push(row(
                "Smallest",
                format!("{:<9} {}", format_size(sm.bytes), sm.name),
            ));
        }
        out.push(row("Average", format_size(self.mean_bytes)));
        out.push(row("Median", format_size(self.median_bytes)));

        // Layers.
        if let Some(l) = &self.layers {
            out.push(String::new());
            out.push(format!("{GLYPH_LAYERS} Layers  ×{}", l.count));
            out.push(row(
                "Params",
                each_total(
                    format_parameters(l.params_each()),
                    format_parameters(l.params),
                ),
            ));
            out.push(row(
                "Size",
                each_total(format_size(l.bytes_each()), format_size(l.bytes)),
            ));
        }

        // Experts.
        if let Some(x) = &self.experts {
            out.push(String::new());
            let count = if x.per_layer > 0 {
                format!("  ×{} per layer", x.per_layer)
            } else {
                String::new()
            };
            out.push(format!("{GLYPH_EXPERTS} Experts{count}"));
            let mut storage = x.storage.label().to_string();
            if x.gate_up_fused {
                storage.push_str(" · gate+up fused");
            }
            out.push(row("Storage", storage));
            if x.per_layer > 0 || x.storage == ExpertStorage::Unfused {
                out.push(row(
                    "Params",
                    each_total(
                        format_parameters(x.params_each()),
                        format_parameters(x.params),
                    ),
                ));
                out.push(row(
                    "Size",
                    each_total(format_size(x.bytes_each()), format_size(x.bytes)),
                ));
            }
            // Per-projection split (down/gate/up), each with its per-layer footprint.
            for c in &x.by_category {
                let per_layer = c.bytes / x.layers.max(1);
                out.push(row(
                    &c.name,
                    each_total(format_size(per_layer), format_size(c.bytes)),
                ));
            }
        }

        // Per-layer profile: a per-metric sparkline when it varies across the
        // stack, else a plain "uniform" note (a flat sparkline says nothing); plus
        // a single composition bar for the whole stack.
        if let Some(pl) = &self.per_layer {
            const LBL: usize = 13;
            out.push(String::new());
            out.push("Per-layer profile".into());
            let metric = |label: &str, vals: &[usize], fmt: fn(usize) -> String| -> String {
                let (lo, hi) = (
                    vals.iter().copied().min().unwrap_or(0),
                    vals.iter().copied().max().unwrap_or(0),
                );
                if lo == hi {
                    format!("  {label:<LBL$}  uniform · {} each", fmt(lo))
                } else {
                    format!(
                        "  {label:<LBL$}  {}  {}–{}",
                        spark_string(vals, GRAPH_W),
                        fmt(lo),
                        fmt(hi)
                    )
                }
            };
            out.push(metric("Size/layer", &pl.bytes_series(), format_size));
            out.push(metric(
                "Params/layer",
                &pl.params_series(),
                format_parameters,
            ));
            out.push(metric("Tensors/layer", &pl.tensor_series(), |n| {
                n.to_string()
            }));
            // Composition: one 100%-stacked bar over the whole stack, with shares.
            let comp = pl.composition_totals();
            let total: usize = comp.iter().sum();
            if total > 0 {
                let pct = |x: usize| (x * 100 + total / 2) / total;
                out.push(format!(
                    "  {:<LBL$}  {}",
                    "Composition",
                    composition_bar(comp, BAR_W)
                ));
                out.push(format!(
                    "  {:<LBL$}  {} {}% attention · {} {}% ffn/experts · {} {}% other",
                    "",
                    SHADES[0],
                    pct(comp[0]),
                    SHADES[1],
                    pct(comp[1]),
                    SHADES[2],
                    pct(comp[2]),
                ));
            }
        }

        // By dtype.
        if !self.dtypes.is_empty() {
            out.push(String::new());
            out.push("By dtype".into());
            let dw = self.dtypes.iter().map(|d| d.dtype.len()).max().unwrap_or(0);
            for d in &self.dtypes {
                out.push(format!(
                    "  {:<dw$}  {:>8}  {} tensor{}",
                    d.dtype,
                    format_size(d.bytes),
                    d.count,
                    if d.count == 1 { "" } else { "s" }
                ));
            }
        }

        // On disk (filesystem allocation) — the true footprint, ZFS/sparse-aware.
        if let Some(d) = &self.disk {
            out.push(String::new());
            out.push("On disk (filesystem)".into());
            out.push(row(
                "Allocated",
                format!(
                    "{}  ({} apparent, {})",
                    format_size(d.total_allocated as usize),
                    format_size(d.total_apparent as usize),
                    ratio_phrase(d.total_apparent, d.total_allocated),
                ),
            ));
            // Per-shard breakdown, folded away by default (a many-shard model is
            // otherwise a wall of rows) and only for shards the filesystem shrank.
            if d.shards.len() > 1 {
                let savers: Vec<&ShardDisk> = d
                    .shards
                    .iter()
                    .filter(|s| has_saving(s.apparent, s.allocated))
                    .collect();
                if shards_expanded {
                    // Unfolding shows *every* shard (savers and not) — the folded
                    // summary already gave the "N of M smaller" headline, so the
                    // expanded view is the full breakdown, not a filtered one.
                    let nw = d.shards.iter().map(|s| s.name.len()).max().unwrap_or(0);
                    for s in &d.shards {
                        out.push(format!(
                            "    {:<nw$}  {:>9} → {:>9}  ({})",
                            s.name,
                            format_size(s.apparent as usize),
                            format_size(s.allocated as usize),
                            ratio_phrase(s.apparent, s.allocated),
                        ));
                    }
                } else {
                    out.push(format!(
                        "  ▸ per-shard breakdown ({} of {} smaller)",
                        savers.len(),
                        d.shards.len()
                    ));
                }
            }
        }

        out.join("\n")
    }
}

/// Whether the filesystem saved a *meaningful* amount on this file — at least
/// ~1%, so files the filesystem left untouched (and trivial block-rounding
/// differences) don't clutter the per-shard list; only real savings are worth a
/// row.
pub(crate) fn has_saving(apparent: u64, allocated: u64) -> bool {
    allocated < apparent && (apparent - allocated).saturating_mul(100) >= apparent
}

/// "N.N× smaller" when the filesystem shrank the file, else "no filesystem
/// saving" — describing `allocated` relative to `apparent`.
pub(crate) fn ratio_phrase(apparent: u64, allocated: u64) -> String {
    if allocated == 0 || allocated >= apparent {
        "no filesystem saving".to_string()
    } else {
        format!("{:.2}× smaller", apparent as f64 / allocated as f64)
    }
}

/// The canonical layer family: its prefix and layer count (highest index + 1).
/// Mirrors `check`'s selection — prefer the conventional `…layers` prefix, else
/// the largest indexed family. Shared by [`layer_stats`] and
/// [`PerLayerStats::compute`] so the summary and the per-layer series can't drift.
fn canonical_family(tensors: &[TensorInfo]) -> Option<(String, usize)> {
    let mut fam: HashMap<String, BTreeSet<usize>> = HashMap::new();
    for t in tensors {
        if let Some((prefix, idx, _)) = split_layer_index(&t.name) {
            fam.entry(prefix).or_default().insert(idx);
        }
    }
    let chosen = fam
        .iter()
        .find(|(p, _)| p.rsplit('.').next() == Some("layers"))
        .or_else(|| fam.iter().max_by_key(|(_, idxs)| idxs.len()))?;
    let count = chosen.1.iter().next_back().map(|&m| m + 1)?;
    Some((chosen.0.clone(), count))
}

/// Aggregate the repeated layer stack over the canonical family.
fn layer_stats(tensors: &[TensorInfo]) -> Option<LayerStats> {
    let (prefix, count) = canonical_family(tensors)?;
    let mut params = 0;
    let mut bytes = 0;
    for t in tensors {
        if let Some((p, _, _)) = split_layer_index(&t.name)
            && p == prefix
        {
            params += t.num_elements;
            bytes += t.size_bytes;
        }
    }
    Some(LayerStats {
        count,
        params,
        bytes,
    })
}

/// Whether a tensor name has an `experts` segment at all.
fn is_expert(name: &str) -> bool {
    name.split('.').any(|s| s == "experts")
}

/// Aggregate MoE expert structure, or `None` for a dense checkpoint.
fn expert_stats(
    tensors: &[TensorInfo],
    config: Option<&crate::config::ModelConfig>,
) -> Option<ExpertStats> {
    if !tensors.iter().any(|t| is_expert(&t.name)) {
        return None;
    }

    // A per-expert index anywhere means the experts are stored unfused; experts
    // that appear only as stacked tensors (no index) are fused.
    let max_expert_idx = tensors.iter().filter_map(|t| expert_index(&t.name)).max();
    let storage = if max_expert_idx.is_some() {
        ExpertStorage::Unfused
    } else {
        ExpertStorage::Fused
    };

    // Experts per layer: for unfused, the highest expert index + 1; for fused,
    // the declared config count, else the leading dimension of a stacked expert
    // tensor (its expert axis).
    let per_layer = match (storage, max_expert_idx) {
        (ExpertStorage::Unfused, Some(m)) => m + 1,
        _ => config
            .and_then(|c| c.num_experts)
            .map(|n| n as usize)
            .or_else(|| {
                tensors
                    .iter()
                    .find(|t| is_expert(&t.name) && !t.shape.is_empty())
                    .map(|t| t.shape[0])
            })
            .unwrap_or(0),
    };

    // Layers carrying experts, the expert totals, and the per-projection split.
    let mut layers = BTreeSet::new();
    let mut params = 0;
    let mut bytes = 0;
    // Logical bytes per projection, emitted below in a fixed canonical order
    // (independent of tensor iteration order).
    let mut cat_bytes: HashMap<&'static str, usize> = HashMap::new();
    for t in tensors {
        if is_expert(&t.name) {
            params += t.num_elements;
            bytes += t.size_bytes;
            if let Some((_, idx, _)) = split_layer_index(&t.name) {
                layers.insert(idx);
            }
            if let Some(cat) = proj_category(&t.name) {
                *cat_bytes.entry(cat).or_default() += t.size_bytes;
            }
        }
    }
    let by_category = ["down_proj", "gate_proj", "up_proj", "gate_up_proj"]
        .into_iter()
        .filter_map(|name| {
            cat_bytes.get(name).map(|&bytes| ExpertCategory {
                name: name.to_string(),
                bytes,
            })
        })
        .collect();

    let gate_up_fused = tensors
        .iter()
        .any(|t| t.name.contains("gate_up_proj") || t.name.contains("gate_proj__up_proj"));

    Some(ExpertStats {
        per_layer,
        storage,
        gate_up_fused,
        params,
        bytes,
        layers: layers.len().max(1),
        by_category,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::Layout;

    fn ti(name: &str, dtype: &str, shape: &[usize], dsize: usize) -> TensorInfo {
        let num_elements = shape.iter().product();
        TensorInfo {
            name: name.into(),
            dtype: dtype.into(),
            shape: shape.to_vec(),
            size_bytes: num_elements * dsize,
            num_elements,
            storage: Storage::Unknown,
            source_path: "mem.safetensors".into(),
            layout: Layout::None,
        }
    }

    #[test]
    fn overall_totals_and_extremes() {
        let tensors = vec![
            ti("embed", "F32", &[10, 10], 4), // 100 elems, 400 B
            ti("big", "F32", &[100, 100], 4), // 10_000 elems, 40_000 B
            ti("small", "F32", &[2], 4),      // 2 elems, 8 B
        ];
        let s = CheckpointStats::compute(&tensors, None, None);
        assert_eq!(s.n_tensors, 3);
        assert_eq!(s.params, 10_102);
        assert_eq!(s.logical_bytes, 40_408);
        assert_eq!(s.disk_bytes, 40_408); // no compression
        assert!(!s.compressed);
        assert_eq!(s.largest.unwrap().name, "big");
        assert_eq!(s.smallest.unwrap().name, "small");
        assert_eq!(s.mean_bytes, 40_408 / 3);
        assert_eq!(s.median_bytes, 400); // middle of {8, 400, 40000}
    }

    #[test]
    fn file_extremes_are_named() {
        // A tensor of `elems` F32 elements (4 B each) living in shard `path`.
        let at = |name: &str, elems: usize, path: &str| TensorInfo {
            name: name.into(),
            dtype: "F32".into(),
            shape: vec![elems],
            size_bytes: elems * 4,
            num_elements: elems,
            storage: Storage::Unknown,
            source_path: path.into(),
            layout: Layout::None,
        };
        // Two shards of different total size — the file stats name each (by
        // basename), like the per-tensor largest/smallest rows.
        let tensors = vec![
            at("a", 1000, "/m/model-00001-of-00002.safetensors"), // 4000 B
            at("b", 10, "/m/model-00002-of-00002.safetensors"),   // 40 B
        ];
        let s = CheckpointStats::compute(&tensors, None, None);
        assert_eq!(s.files.count, 2);
        let largest = s.files.largest.unwrap();
        assert_eq!(largest.name, "model-00001-of-00002.safetensors");
        assert_eq!(largest.bytes, 4000);
        let smallest = s.files.smallest.unwrap();
        assert_eq!(smallest.name, "model-00002-of-00002.safetensors");
        assert_eq!(smallest.bytes, 40);
    }

    #[test]
    fn dtype_breakdown_sorted_by_bytes() {
        let tensors = vec![
            ti("a", "BF16", &[1000], 2),    // 2000 B
            ti("b", "BF16", &[1000], 2),    // 2000 B
            ti("c", "F8_E4M3", &[1000], 1), // 1000 B
        ];
        let s = CheckpointStats::compute(&tensors, None, None);
        assert_eq!(s.dtypes.len(), 2);
        assert_eq!(s.dtypes[0].dtype, "BF16");
        assert_eq!(s.dtypes[0].count, 2);
        assert_eq!(s.dtypes[0].bytes, 4000);
        assert_eq!(s.dtypes[1].dtype, "F8_E4M3");
    }

    #[test]
    fn layer_stack_counted_and_aggregated() {
        let tensors = vec![
            ti("model.embed_tokens.weight", "F32", &[100], 4),
            ti("model.layers.0.mlp.weight", "F32", &[10], 4),
            ti("model.layers.1.mlp.weight", "F32", &[10], 4),
            ti("model.layers.2.mlp.weight", "F32", &[10], 4),
        ];
        let s = CheckpointStats::compute(&tensors, None, None);
        let l = s.layers.unwrap();
        assert_eq!(l.count, 3);
        assert_eq!(l.params, 30);
        assert_eq!(l.params_each(), 10);
    }

    #[test]
    fn unfused_experts_detected() {
        let mut tensors = vec![ti("model.embed_tokens.weight", "F32", &[100], 4)];
        // 2 layers × 4 experts, one tensor each.
        for layer in 0..2 {
            for e in 0..4 {
                tensors.push(ti(
                    &format!("model.layers.{layer}.mlp.experts.{e}.down_proj.weight"),
                    "F32",
                    &[10],
                    4,
                ));
            }
        }
        let s = CheckpointStats::compute(&tensors, None, None);
        let x = s.experts.unwrap();
        assert_eq!(x.storage, ExpertStorage::Unfused);
        assert_eq!(x.per_layer, 4);
        assert_eq!(x.layers, 2);
        assert_eq!(x.params, 80); // 8 experts × 10
        assert_eq!(x.params_each(), 10);
        assert!(!x.gate_up_fused);
    }

    #[test]
    fn experts_split_by_projection_category() {
        let mut tensors = vec![ti("model.embed_tokens.weight", "F32", &[100], 4)];
        // 2 layers × 3 experts, each with down/gate/up projections of distinct sizes.
        for layer in 0..2 {
            for e in 0..3 {
                for (proj, elems) in [("down_proj", 10), ("gate_proj", 20), ("up_proj", 20)] {
                    tensors.push(ti(
                        &format!("model.layers.{layer}.mlp.experts.{e}.{proj}.weight"),
                        "F32",
                        &[elems],
                        4,
                    ));
                }
            }
        }
        let x = CheckpointStats::compute(&tensors, None, None)
            .experts
            .unwrap();
        // Canonical order down/gate/up; gate_up absent here.
        let cats: Vec<&str> = x.by_category.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(cats, ["down_proj", "gate_proj", "up_proj"]);
        let down = &x.by_category[0];
        assert_eq!(down.bytes, 6 * 10 * 4); // 6 tensors (2 layers × 3 experts), 240 B total
        assert_eq!(down.bytes / x.layers.max(1), 120); // per layer (2 layers)
        let gate = &x.by_category[1];
        assert_eq!(gate.bytes, 6 * 20 * 4); // 480 B total
    }

    #[test]
    fn spark_levels_are_min_anchored() {
        // Distinct values map min → 0, max → 7 across the range.
        let (levels, lo, hi) = spark_levels(&[10, 20, 30], GRAPH_W);
        assert_eq!((lo, hi), (10, 30));
        assert_eq!(levels.first(), Some(&0));
        assert_eq!(levels.last(), Some(&7));
        // All-equal → every cell at mid (index 3), no divide-by-zero.
        let (flat, lo, hi) = spark_levels(&[5, 5, 5], GRAPH_W);
        assert_eq!((lo, hi), (5, 5));
        assert!(flat.iter().all(|&l| l == 3));
    }

    #[test]
    fn bucketing_fits_the_width_cap() {
        // Fewer layers than the cap → one cell per layer.
        assert_eq!(bucket_means(&[1, 2, 3, 4], GRAPH_W).len(), 4);
        // More layers than the width → bucketed to the width, averaging each bucket.
        let vals: Vec<usize> = (0..100).collect();
        let means = bucket_means(&vals, 10);
        assert_eq!(means.len(), 10);
        assert!((means[0] - 4.5).abs() < 1e-9); // mean of 0..=9
    }

    #[test]
    fn alloc_rows_sums_to_height_by_largest_remainder() {
        assert_eq!(alloc_rows([1, 1, 1], 6), [2, 2, 2]);
        // Leftover row goes to the largest fractional part (ties → lowest index):
        // 1:1:0 of height 3 → raw 1.5/1.5/0 → floors 1/1/0, +1 → index 0.
        assert_eq!(alloc_rows([1, 1, 0], 3), [2, 1, 0]);
        assert_eq!(alloc_rows([0, 0, 0], 6), [0, 0, 0]); // empty column
        for parts in [[3, 1, 0], [7, 2, 1], [1, 0, 5]] {
            assert_eq!(alloc_rows(parts, 6).iter().sum::<usize>(), 6);
        }
    }

    #[test]
    fn per_layer_series_aligns_with_layer_count() {
        let mut tensors = vec![ti("model.embed_tokens.weight", "F32", &[100], 4)];
        for layer in 0..4 {
            tensors.push(ti(
                &format!("model.layers.{layer}.self_attn.q_proj.weight"),
                "F32",
                &[10],
                4,
            ));
            tensors.push(ti(
                &format!("model.layers.{layer}.mlp.down_proj.weight"),
                "F32",
                &[20],
                4,
            ));
            tensors.push(ti(
                &format!("model.layers.{layer}.input_layernorm.weight"),
                "F32",
                &[2],
                4,
            ));
        }
        let s = CheckpointStats::compute(&tensors, None, None);
        let pl = s.per_layer.unwrap();
        assert_eq!(pl.rows.len(), s.layers.unwrap().count); // 4, aligned
        let row = &pl.rows[0];
        assert_eq!(row.tensors, 3);
        assert_eq!(row.attn_bytes, 10 * 4); // q_proj
        assert_eq!(row.ffn_bytes, 20 * 4); // down_proj
        assert_eq!(row.other_bytes, 2 * 4); // layernorm
    }

    #[test]
    fn dense_checkpoint_has_no_per_layer() {
        let tensors = vec![ti("lm_head.weight", "F32", &[10], 4)];
        assert!(
            CheckpointStats::compute(&tensors, None, None)
                .per_layer
                .is_none()
        );
    }

    #[test]
    fn fused_experts_use_config_or_shape() {
        let tensors = vec![
            ti(
                "model.layers.0.mlp.experts.gate_up_proj.weight",
                "F32",
                &[8, 10],
                4,
            ),
            ti(
                "model.layers.1.mlp.experts.gate_up_proj.weight",
                "F32",
                &[8, 10],
                4,
            ),
        ];
        let s = CheckpointStats::compute(&tensors, None, None);
        let x = s.experts.unwrap();
        assert_eq!(x.storage, ExpertStorage::Fused);
        assert_eq!(x.per_layer, 8); // leading dim of the stacked tensor
        assert_eq!(x.layers, 2);
        assert!(x.gate_up_fused);
    }

    #[test]
    fn dense_checkpoint_has_no_experts() {
        let tensors = vec![ti("model.layers.0.mlp.weight", "F32", &[10], 4)];
        assert!(
            CheckpointStats::compute(&tensors, None, None)
                .experts
                .is_none()
        );
    }

    #[test]
    fn report_has_median_row_and_architecture_in_overview() {
        let tensors = vec![
            ti("model.embed_tokens.weight", "F32", &[100], 4),
            ti("model.layers.0.mlp.weight", "F32", &[10], 4),
            ti("model.layers.1.mlp.weight", "F32", &[10], 4),
        ];
        let config = crate::config::ModelConfig {
            model_type: Some("qwen3_moe".into()),
            num_hidden_layers: None,
            num_experts: None,
            vocab_size: None,
            hidden_size: None,
            tie_word_embeddings: None,
            use_qk_norm: None,
        };
        let report = CheckpointStats::compute(&tensors, Some(&config), None).render(false);

        // Median is its own labelled row, not folded into Average's parens.
        assert!(report.contains("\n  Median"), "report:\n{report}");
        assert!(
            !report.contains("(median"),
            "median should not be parenthetical"
        );
        // A full-width label keeps a gap before its value (not "Architectureqwen3_moe").
        assert!(
            report.contains("Architecture qwen3_moe"),
            "label and value should be separated:\n{report}"
        );
        // Architecture sits under Overview — before the first glyphed section.
        let arch = report.find("Architecture").expect("architecture row");
        let files = report.find("Files").expect("files header");
        assert!(
            arch < files,
            "architecture should be in the Overview section"
        );
    }

    #[cfg(unix)]
    #[test]
    fn from_local_stats_real_files() {
        // Stat two files that certainly exist in the repo; the allocated size is
        // whatever the filesystem reports, but the totals must add up and a
        // present file's allocation is non-zero.
        let paths = ["Cargo.toml", "src/stats.rs"];
        let du = DiskUsage::from_local(&paths).expect("both files stat");
        assert_eq!(du.shards.len(), 2);
        assert_eq!(
            du.total_apparent,
            du.shards.iter().map(|s| s.apparent).sum::<u64>()
        );
        assert_eq!(
            du.total_allocated,
            du.shards.iter().map(|s| s.allocated).sum::<u64>()
        );
        assert!(du.shards.iter().all(|s| s.allocated > 0));
        // A path that doesn't stat is skipped, not fatal.
        assert!(DiskUsage::from_local(&["definitely/not/here.xyz"]).is_none());
    }

    #[test]
    fn report_on_disk_folds_to_a_summary_and_unfolds_to_every_shard() {
        let tensors = vec![ti("w", "F32", &[10], 4)];
        // One shard squeezed 4× (a real saving) among two the filesystem left
        // alone (allocated ≥ apparent) — deterministic, so the wording is pinned.
        let disk = DiskUsage::from_shards(vec![
            ShardDisk {
                name: "shard-saver.safetensors".into(),
                apparent: 4 * 1024 * 1024,
                allocated: 1024 * 1024,
            },
            ShardDisk {
                name: "shard-plain.safetensors".into(),
                apparent: 4 * 1024 * 1024,
                allocated: 4 * 1024 * 1024,
            },
            ShardDisk {
                name: "shard-bigger.safetensors".into(),
                apparent: 4 * 1024 * 1024,
                allocated: 4 * 1024 * 1024 + 4096, // block rounding — larger on disk
            },
        ]);
        let stats = CheckpointStats::compute(&tensors, None, disk);

        // Folded (default): a one-line summary with the saver count, no shard rows.
        let folded = stats.render(false);
        assert!(folded.contains("On disk (filesystem)"), "report:\n{folded}");
        assert!(
            folded.contains("per-shard breakdown (1 of 3 smaller)"),
            "report:\n{folded}"
        );
        assert!(!folded.contains("shard-saver"), "report:\n{folded}");

        // Unfolded: *every* shard is listed — savers and not — not just the savers.
        let expanded = stats.render(true);
        assert!(
            expanded.contains("shard-saver.safetensors"),
            "report:\n{expanded}"
        );
        assert!(expanded.contains("4.00× smaller"), "report:\n{expanded}");
        assert!(
            expanded.contains("shard-plain.safetensors"),
            "report:\n{expanded}"
        );
        assert!(
            expanded.contains("shard-bigger.safetensors"),
            "report:\n{expanded}"
        );
        // The old "N shards with no filesystem saving" collapse line is gone
        // (the per-shard rows still carry "(no filesystem saving)" individually).
        assert!(
            !expanded.contains("shards with no filesystem saving"),
            "report:\n{expanded}"
        );
    }

    #[test]
    fn ratio_and_saving_predicates() {
        // Allocated ≥ apparent (small file rounded up to a block) → no saving.
        assert_eq!(ratio_phrase(888, 4096), "no filesystem saving");
        assert_eq!(ratio_phrase(0, 0), "no filesystem saving");
        assert_eq!(ratio_phrase(1000, 500), "2.00× smaller");
        // A real (≥1%) saving counts; a larger-on-disk or trivial one doesn't.
        assert!(has_saving(1000, 500));
        assert!(!has_saving(1000, 1000));
        assert!(!has_saving(1000, 1200)); // allocated larger
        assert!(!has_saving(1000, 999)); // 0.1% — below the threshold
    }
}
