//! Overall checkpoint statistics — the `s` popup on the tree.
//!
//! A cheap, header-only aggregation over the already-loaded tensor metadata:
//! file/shard count, parameter and byte totals, the largest/smallest/typical
//! tensor, the dtype mix, and the repeated layer / MoE-expert structure of
//! transformer checkpoints. Nothing here reads tensor data — it's all derived
//! from the shapes and dtypes already in memory, so the popup is instant even on
//! multi-GB checkpoints.

use std::collections::{BTreeSet, HashMap};

use crate::check::{expert_index, split_layer_index};
use crate::tree::{Storage, TensorInfo};

/// Section glyphs, matching the tree view's (`▦` tensors, `≡` layers) so the
/// popup reads like the rest of the UI rather than a flat table.
pub(crate) const GLYPH_FILES: &str = "▤";
pub(crate) const GLYPH_TENSORS: &str = "▦";
pub(crate) const GLYPH_LAYERS: &str = "≡";
pub(crate) const GLYPH_EXPERTS: &str = "◆";

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
    pub largest: usize,
    pub smallest: usize,
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
        let mut file_sizes: Vec<usize> = per_file.into_values().collect();
        file_sizes.sort_unstable();
        let files = FileStats {
            count: file_sizes.len(),
            noun,
            largest: file_sizes.last().copied().unwrap_or(0),
            smallest: file_sizes.first().copied().unwrap_or(0),
            mean: logical_bytes.checked_div(file_sizes.len()).unwrap_or(0),
            median: file_sizes.get(file_sizes.len() / 2).copied().unwrap_or(0),
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
        out.push(row("Largest", format_size(self.files.largest)));
        out.push(row("Smallest", format_size(self.files.smallest)));
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

/// Aggregate the repeated layer stack. Mirrors `check`'s family selection:
/// prefer the conventional `…layers` prefix, else the largest indexed family.
fn layer_stats(tensors: &[TensorInfo]) -> Option<LayerStats> {
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
    let prefix = chosen.0.clone();
    let count = chosen.1.iter().next_back().map(|&m| m + 1)?;

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

    // Layers carrying experts, and the expert totals.
    let mut layers = BTreeSet::new();
    let mut params = 0;
    let mut bytes = 0;
    for t in tensors {
        if is_expert(&t.name) {
            params += t.num_elements;
            bytes += t.size_bytes;
            if let Some((_, idx, _)) = split_layer_index(&t.name) {
                layers.insert(idx);
            }
        }
    }

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
