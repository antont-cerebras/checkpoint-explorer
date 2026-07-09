//! The `diff` subcommand: compare two checkpoints' *structure* and summarize the
//! differences. "Structure" means the tensors (by name, dtype, and shape) and the
//! metadata (by name, value, and value type) — **not** the tensor data/values,
//! which a structural diff never reads (so it stays fast even on multi-GB files).
//!
//! The comparison ([`compare`]) is a pure function over two [`CheckpointSummary`]s
//! and produces a [`DiffReport`]; rendering ([`DiffReport::render`]) and the
//! `diff`-style exit code ([`DiffReport::has_differences`]) are separate so the
//! logic is testable without any I/O.

use std::collections::{BTreeMap, HashSet};
use std::fmt::Write;

use glob::{MatchOptions, Pattern};
use serde_json::Value;

use crate::sample::{HistBins, HistogramDiff, ValueDiff};
use crate::tree::{MetadataInfo, TensorInfo};
use crate::utils::{format_parameters, format_shape, format_size};

const RED: &str = "\x1b[31m";
const GREEN: &str = "\x1b[32m";
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

/// Rendering options for the diff output.
#[derive(Clone, Copy)]
pub struct DiffOpts {
    /// Colorize with ANSI escapes (removed in red, added in green; for a changed
    /// tensor only the dtype/shape token that differs).
    pub color: bool,
    /// Include the metadata section (off under `--only-tensors`).
    pub metadata: bool,
    /// Collapse entries sharing a name template + the same change into one line
    /// with a count and index range (off under `--full`).
    pub group: bool,
    /// Element values were compared (`--values`): show per-change value stats and
    /// note when a change's values weren't compared.
    pub values: bool,
    /// Value distributions were compared (`--histogram`): show a per-change
    /// total-variation-distance summary.
    pub histogram: bool,
    /// A [`TensorFilter`] scoped the diff to a subset of tensors — the metadata
    /// section's "not compared" note names this (rather than `--only-tensors`) so
    /// it's clear why the whole checkpoint wasn't diffed.
    pub filtered: bool,
}

/// A tensor's distribution shift for `diff --histogram`: total variation distance
/// (`0` = same shape, `1` = disjoint) and the bin count it was measured over.
#[derive(Clone, Copy)]
pub struct HistShift {
    pub tvd: f64,
    pub bins: usize,
}

/// Per-tensor element / distribution comparison attached to a change — filled by
/// `--values` / `--histogram`, empty for a pure structural diff.
#[derive(Default)]
pub struct TensorExtras {
    pub values: Option<ValueDiff>,
    pub histogram: Option<HistShift>,
}

impl TensorExtras {
    /// Whether the extras themselves indicate a difference (so a structurally
    /// identical tensor still counts as changed).
    fn differ(&self) -> bool {
        self.values.is_some_and(|v| v.differing > 0) || self.histogram.is_some_and(|h| h.tvd > 0.0)
    }
}

/// Wrap `text` in an ANSI colour `code` when `on`, else return it unchanged.
fn paint(text: &str, on: bool, code: &str) -> String {
    if on {
        format!("{code}{text}{RESET}")
    } else {
        text.to_string()
    }
}

/// A `label: old → new (±abs, ±rel%)` line summarizing an overall total's change
/// (checkpoint size or parameter count), formatting values with `fmt`. Shows
/// "(unchanged)" when equal, and omits the percentage when the old side is zero
/// (no baseline). Coloured like the tensor diff — the old value red, the new value
/// green — while the parenthetical delta is dimmed (a convenience; its sign
/// already shows the direction).
fn totals_line(
    label: &str,
    old: usize,
    new: usize,
    color: bool,
    fmt: fn(usize) -> String,
) -> String {
    if old == new {
        return format!("{label}: {} (unchanged)", fmt(new));
    }
    let delta = new as i128 - old as i128;
    let sign = if delta >= 0 { "+" } else { "-" };
    let mag = fmt(delta.unsigned_abs() as usize);
    let rel = if old == 0 {
        String::new()
    } else {
        format!(
            ", {sign}{:.1}%",
            delta.unsigned_abs() as f64 / old as f64 * 100.0
        )
    };
    let old_s = paint(&fmt(old), color, RED);
    let new_s = paint(&fmt(new), color, GREEN);
    let delta_s = paint(&format!("{sign}{mag}{rel}"), color, DIM);
    format!("{label}: {old_s} → {new_s} ({delta_s})")
}

/// Render a changed tensor's `old` and `new` signatures, colouring only what
/// actually differs — the dtype (if it changed) and the individual shape
/// dimensions that changed — old side red, new green, so the eye lands on the
/// change. When the ranks differ the whole shape is coloured (dims don't line up).
/// No colour when `color` is off.
fn render_change(old: &TensorSig, new: &TensorSig, color: bool) -> (String, String) {
    let dtype_changed = old.dtype != new.dtype;
    let same_rank = old.shape.len() == new.shape.len();
    let one = |sig: &TensorSig, other: &[usize], code: &str| {
        let dtype = paint(&sig.dtype, color && dtype_changed, code);
        let shape = if !color {
            format_shape(&sig.shape)
        } else if !same_rank {
            paint(&format_shape(&sig.shape), true, code)
        } else {
            // Colour each dimension only where it differs from the other side.
            let dims: Vec<String> = sig
                .shape
                .iter()
                .zip(other)
                .map(|(d, o)| paint(&d.to_string(), d != o, code))
                .collect();
            format!("({})", dims.join(", "))
        };
        format!("{dtype} {shape}")
    };
    (one(old, &new.shape, RED), one(new, &old.shape, GREEN))
}

/// Split a name into a template (each run of digits → a `{}` placeholder) and the
/// digit-run values, so entries differing only by an index — a layer number, an
/// expert id — share a template and can be collapsed.
fn templatize(name: &str) -> (String, Vec<String>) {
    let mut template = String::new();
    let mut indices = Vec::new();
    let mut digits = String::new();
    for ch in name.chars() {
        if ch.is_ascii_digit() {
            digits.push(ch);
        } else {
            if !digits.is_empty() {
                template.push_str("{}");
                indices.push(std::mem::take(&mut digits));
            }
            template.push(ch);
        }
    }
    if !digits.is_empty() {
        template.push_str("{}");
        indices.push(digits);
    }
    (template, indices)
}

/// One collapsed run of entries: the shared `template`, the index values seen at
/// each placeholder, the member `count`, and the (identical) change `key`.
struct Group<K> {
    template: String,
    indices: Vec<Vec<String>>,
    count: usize,
    key: K,
}

/// Group `(name, change-key)` entries by `(template, key)` in first-seen order, so
/// only entries with the same structure *and* the same change merge.
fn group_entries<K: Clone + Eq + std::hash::Hash>(items: &[(String, K)]) -> Vec<Group<K>> {
    use std::collections::HashMap;
    let mut index: HashMap<(String, K), usize> = HashMap::new();
    let mut groups: Vec<Group<K>> = Vec::new();
    for (name, key) in items {
        let (template, idx) = templatize(name);
        let gi = match index.get(&(template.clone(), key.clone())) {
            Some(&i) => i,
            None => {
                index.insert((template.clone(), key.clone()), groups.len());
                groups.push(Group {
                    template,
                    indices: vec![Vec::new(); idx.len()],
                    count: 0,
                    key: key.clone(),
                });
                groups.len() - 1
            }
        };
        let g = &mut groups[gi];
        g.count += 1;
        for (p, v) in idx.into_iter().enumerate() {
            g.indices[p].push(v);
        }
    }
    groups
}

/// Collapse tensor `names` into their index-templated schema: names sharing a
/// template (each run of digits — a layer number, an expert id — becomes a range
/// placeholder) merge into one `(display_name, count)`, e.g.
/// `model.layers.{0-47}.…experts.{0-3}.down_proj.weight` → count 192. Ordered by
/// first appearance (alphabetical when `names` is sorted). Used to summarize which
/// tensors a `diff` filter matched.
pub fn name_schema(names: &[&str]) -> Vec<(String, usize)> {
    let items: Vec<(String, ())> = names.iter().map(|n| ((*n).to_string(), ())).collect();
    group_entries(&items)
        .into_iter()
        .map(|g| (display_name(&g.template, &g.indices), g.count))
        .collect()
}

/// Render each entry as its own group (no collapsing) — for `--full`. Each
/// placeholder gets its single value back, so the displayed name is the original.
fn singletons<K: Clone>(items: &[(String, K)]) -> Vec<Group<K>> {
    items
        .iter()
        .map(|(name, key)| {
            let (template, idx) = templatize(name);
            Group {
                template,
                indices: idx.into_iter().map(|v| vec![v]).collect(),
                count: 1,
                key: key.clone(),
            }
        })
        .collect()
}

/// Reconstruct a group's display name: fill each `{}` with its index — the single
/// value when constant across the group, else `{lo-hi,…}` for the range.
fn display_name(template: &str, indices: &[Vec<String>]) -> String {
    let mut out = String::new();
    for (i, part) in template.split("{}").enumerate() {
        out.push_str(part);
        if let Some(vals) = indices.get(i) {
            out.push_str(&summarize_indices(vals));
        }
    }
    out
}

/// One placeholder's index values as a compact string: the lone value when they're
/// all equal, else `{0-47}` / `{0-3,5}` (integer ranges) or `{a,b}` (sorted list).
fn summarize_indices(values: &[String]) -> String {
    use std::collections::BTreeSet;
    let distinct: BTreeSet<&str> = values.iter().map(String::as_str).collect();
    if distinct.len() == 1 {
        return values[0].clone();
    }
    match distinct
        .iter()
        .map(|s| s.parse::<i64>().ok())
        .collect::<Option<Vec<i64>>>()
    {
        Some(mut nums) => {
            nums.sort_unstable();
            format!("{{{}}}", compact_int_ranges(&nums))
        }
        None => format!("{{{}}}", distinct.into_iter().collect::<Vec<_>>().join(",")),
    }
}

/// Collapse a sorted integer list into comma-separated runs: `[0,1,2,5]` → `0-2,5`.
fn compact_int_ranges(sorted: &[i64]) -> String {
    let mut out = String::new();
    let mut i = 0;
    while i < sorted.len() {
        let mut j = i;
        while j + 1 < sorted.len() && sorted[j + 1] == sorted[j] + 1 {
            j += 1;
        }
        if !out.is_empty() {
            out.push(',');
        }
        if j == i {
            let _ = write!(out, "{}", sorted[i]);
        } else {
            let _ = write!(out, "{}-{}", sorted[i], sorted[j]);
        }
        i = j + 1;
    }
    out
}

/// The `  (×N)` suffix for a collapsed group (empty for a single entry).
fn count_suffix(count: usize) -> String {
    if count > 1 {
        format!("  (×{count})")
    } else {
        String::new()
    }
}

/// A collapsed run of changed tensors sharing a template and the same dtype/shape
/// change, with their value comparisons aggregated across the run.
struct ChangedGroup {
    template: String,
    indices: Vec<Vec<String>>,
    count: usize,
    old: TensorSig,
    new: TensorSig,
    values: Option<ValueDiff>,
    /// Each member's histogram TVD (empty when `--histogram` wasn't run), plus the
    /// shared bin count — so the group can report max & mean shift.
    hist_tvds: Vec<f64>,
    hist_bins: usize,
}

/// Combine two value comparisons: counts sum, `max_abs` is the max, `mean_abs` is
/// the element-weighted mean — so a group's aggregate reads like one comparison.
fn merge_values(acc: Option<ValueDiff>, next: Option<ValueDiff>) -> Option<ValueDiff> {
    match (acc, next) {
        (None, x) | (x, None) => x,
        (Some(a), Some(b)) => {
            let elements = a.elements + b.elements;
            let mean_abs = if elements > 0 {
                (a.mean_abs * a.elements as f64 + b.mean_abs * b.elements as f64) / elements as f64
            } else {
                0.0
            };
            Some(ValueDiff {
                elements,
                differing: a.differing + b.differing,
                max_abs: a.max_abs.max(b.max_abs),
                mean_abs,
                nonfinite_mismatch: a.nonfinite_mismatch + b.nonfinite_mismatch,
            })
        }
    }
}

/// Group changed tensors by `(template, old_sig, new_sig)` in first-seen order
/// (aggregating their value comparisons), or one group per tensor when `!group`.
fn group_changed(items: &[TensorChange], group: bool) -> Vec<ChangedGroup> {
    use std::collections::HashMap;
    let mut index: HashMap<(String, TensorSig, TensorSig), usize> = HashMap::new();
    let mut groups: Vec<ChangedGroup> = Vec::new();
    for c in items {
        let (template, idx) = templatize(&c.name);
        // `!group` keeps every entry distinct: key on the unique name too.
        let bucket = if group {
            (template.clone(), c.old.clone(), c.new.clone())
        } else {
            (c.name.clone(), c.old.clone(), c.new.clone())
        };
        let gi = match index.get(&bucket) {
            Some(&i) => i,
            None => {
                index.insert(bucket, groups.len());
                groups.push(ChangedGroup {
                    template,
                    indices: vec![Vec::new(); idx.len()],
                    count: 0,
                    old: c.old.clone(),
                    new: c.new.clone(),
                    values: None,
                    hist_tvds: Vec::new(),
                    hist_bins: 0,
                });
                groups.len() - 1
            }
        };
        let g = &mut groups[gi];
        g.count += 1;
        for (p, v) in idx.into_iter().enumerate() {
            g.indices[p].push(v);
        }
        g.values = merge_values(g.values, c.values);
        if let Some(h) = c.histogram {
            g.hist_tvds.push(h.tvd);
            g.hist_bins = h.bins;
        }
    }
    groups
}

/// A tensor's compared identity: dtype + shape. Two tensors with the same name
/// are "changed" when these differ (data bytes are not part of the comparison).
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct TensorSig {
    pub dtype: String,
    pub shape: Vec<usize>,
}

impl TensorSig {
    /// The signature of a loaded tensor.
    pub fn of(t: &TensorInfo) -> Self {
        Self {
            dtype: t.dtype.clone(),
            shape: t.shape.clone(),
        }
    }

    fn render(&self) -> String {
        format!("{} {}", self.dtype, format_shape(&self.shape))
    }
}

/// The element-value comparison outcome for the focused (`--tensor`) diff, when
/// the tensor exists on both sides.
pub enum ValueCmp {
    /// All elements are equal (bit-equal, or NaN in the same slots).
    Identical,
    /// Some elements differ; carries the diff statistics.
    Differ(ValueDiff),
    /// Values weren't compared — the reason (e.g. "shapes differ", an unreadable
    /// dtype, or an I/O error).
    Skipped(String),
}

/// A metadata entry's compared value: its string value + declared type.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct MetaVal {
    pub value: String,
    pub value_type: String,
}

/// One checkpoint reduced to what the structural diff compares. Both maps are
/// keyed by name and ordered, so the diff output is deterministic and alphabetical.
pub struct CheckpointSummary {
    pub tensors: BTreeMap<String, TensorSig>,
    pub metadata: BTreeMap<String, MetaVal>,
    /// Total size in bytes and total parameter count, summed over the deduped
    /// tensors (so a sharded checkpoint isn't double-counted) — for the diff's
    /// overall size/params comparison.
    pub total_bytes: usize,
    pub total_params: usize,
}

impl CheckpointSummary {
    /// Reduce a freshly-loaded checkpoint to its comparable structure. A sharded
    /// checkpoint can list a name in more than one file; the last one wins (the
    /// same name+shape is expected across shards, so this only matters if they
    /// genuinely disagree, which a diff can't meaningfully represent anyway).
    pub fn from_loaded(tensors: &[TensorInfo], metadata: &[MetadataInfo]) -> Self {
        let mut t = BTreeMap::new();
        // Track size/params per name (last-wins, matching `t`) so totals are over
        // the deduped set rather than counting a shared name once per shard.
        let mut sizes: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        for ti in tensors {
            t.insert(ti.name.clone(), TensorSig::of(ti));
            sizes.insert(ti.name.clone(), (ti.size_bytes, ti.num_elements));
        }
        let total_bytes = sizes.values().map(|(b, _)| b).sum();
        let total_params = sizes.values().map(|(_, p)| p).sum();
        let mut m = BTreeMap::new();
        for mi in metadata {
            m.insert(
                mi.name.clone(),
                MetaVal {
                    value: mi.value.clone(),
                    value_type: mi.value_type.clone(),
                },
            );
        }
        Self {
            tensors: t,
            metadata: m,
            total_bytes,
            total_params,
        }
    }
}

/// A tensor's shape as a glob-matchable path, `dim/dim/…` (empty for a scalar) —
/// so a shape pattern can wildcard one dimension with `*` and any number with
/// `**`, matched with [`shape_match_opts`] (a literal `/` separates dims).
fn shape_key(shape: &[usize]) -> String {
    shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join("/")
}

/// Glob options for matching a [`shape_key`]: `/` is a real separator, so `*`
/// matches within one dimension and only `**` spans several — mirroring how a
/// filesystem glob treats path components.
fn shape_match_opts() -> MatchOptions {
    MatchOptions {
        require_literal_separator: true,
        ..MatchOptions::new()
    }
}

/// A CLI-driven selection of which tensors to diff (`--name` / `--names` /
/// `--names-from` / `--dtype-is` / `--shape-is`). The constraints compose with
/// **AND** — a tensor is kept only if it satisfies every constraint that was
/// given; an unset constraint always passes. Names, dtypes and shapes are matched
/// with the same [`glob`] engine, so `*`/`**`/`?`/`[…]` work everywhere (shapes
/// via [`shape_key`], dtypes case-insensitively).
#[derive(Default)]
pub struct TensorFilter {
    /// Name globs; a tensor passes if it matches **any** (empty = unconstrained).
    pub name_globs: Vec<Pattern>,
    /// Exact names (union of `--names` and `--names-from`); `None` = unconstrained.
    pub names_exact: Option<HashSet<String>>,
    /// A dtype glob, matched against the UPPERCASED dtype; `None` = unconstrained.
    pub dtype: Option<Pattern>,
    /// A shape glob, matched against the [`shape_key`]; `None` = unconstrained.
    pub shape: Option<Pattern>,
}

impl TensorFilter {
    /// Whether any constraint is set (so the diff is scoped to a subset).
    pub fn is_active(&self) -> bool {
        !self.name_globs.is_empty()
            || self.names_exact.is_some()
            || self.dtype.is_some()
            || self.shape.is_some()
    }

    /// Whether `name` — with its old and/or new signature (either may be absent
    /// when the tensor is only on one side) — passes every constraint. A dtype /
    /// shape constraint matches if **either** side matches, so a tensor whose
    /// dtype or shape changed is still selected.
    fn matches(&self, name: &str, old: Option<&TensorSig>, new: Option<&TensorSig>) -> bool {
        if !self.name_globs.is_empty() && !self.name_globs.iter().any(|p| p.matches(name)) {
            return false;
        }
        if self
            .names_exact
            .as_ref()
            .is_some_and(|set| !set.contains(name))
        {
            return false;
        }
        if let Some(pat) = &self.dtype {
            let hit = |s: &TensorSig| pat.matches(&s.dtype.to_uppercase());
            if !old.is_some_and(hit) && !new.is_some_and(hit) {
                return false;
            }
        }
        if let Some(pat) = &self.shape {
            let opts = shape_match_opts();
            let hit = |s: &TensorSig| pat.matches_with(&shape_key(&s.shape), opts);
            if !old.is_some_and(hit) && !new.is_some_and(hit) {
                return false;
            }
        }
        true
    }

    /// Restrict both summaries to the tensors that pass the filter. The union of
    /// names is tested, so a tensor present on only one side is kept iff it
    /// matches (and still shows as added/removed). No-op when inactive.
    pub fn apply(&self, old: &mut CheckpointSummary, new: &mut CheckpointSummary) {
        if !self.is_active() {
            return;
        }
        let keep: HashSet<String> = old
            .tensors
            .keys()
            .chain(new.tensors.keys())
            .filter(|n| self.matches(n, old.tensors.get(*n), new.tensors.get(*n)))
            .cloned()
            .collect();
        old.tensors.retain(|k, _| keep.contains(k));
        new.tensors.retain(|k, _| keep.contains(k));
    }

    /// A one-line, human-readable summary of the active constraints (for the
    /// "diff: …" context line), or `None` when inactive.
    pub fn describe(&self) -> Option<String> {
        if !self.is_active() {
            return None;
        }
        let mut parts = Vec::new();
        if !self.name_globs.is_empty() {
            let globs: Vec<&str> = self.name_globs.iter().map(Pattern::as_str).collect();
            parts.push(format!("name~{}", globs.join("|")));
        }
        if let Some(set) = &self.names_exact {
            parts.push(format!("names({})", set.len()));
        }
        if let Some(p) = &self.dtype {
            parts.push(format!("dtype~{}", p.as_str()));
        }
        if let Some(p) = &self.shape {
            // Show dims comma-separated, as the user wrote them.
            parts.push(format!("shape~{}", p.as_str().replace('/', ",")));
        }
        Some(parts.join(", "))
    }
}

/// A tensor present in both checkpoints that differs — by dtype/shape, or (with
/// `--values`) by element values even when the signature is unchanged.
pub struct TensorChange {
    pub name: String,
    pub old: TensorSig,
    pub new: TensorSig,
    /// The element-value comparison, when `--values` ran it (`None` otherwise, or
    /// when the shapes differ so an element-wise comparison isn't defined).
    pub values: Option<ValueDiff>,
    /// The distribution shift, when `--histogram` ran it.
    pub histogram: Option<HistShift>,
}

/// A metadata entry present in both checkpoints whose value and/or type differ.
pub struct MetaChange {
    pub name: String,
    pub old: MetaVal,
    pub new: MetaVal,
}

/// The structural difference between two checkpoints (old → new). "Removed" is in
/// the old but not the new; "added" is in the new but not the old; "changed" is in
/// both with a differing signature.
pub struct DiffReport {
    pub tensors_removed: Vec<(String, TensorSig)>,
    pub tensors_added: Vec<(String, TensorSig)>,
    pub tensors_changed: Vec<TensorChange>,
    pub tensors_unchanged: usize,
    pub meta_removed: Vec<(String, MetaVal)>,
    pub meta_added: Vec<(String, MetaVal)>,
    pub meta_changed: Vec<MetaChange>,
    pub meta_unchanged: usize,
    /// Overall size (bytes) and parameter count of each side, for the size/params
    /// comparison in the summary.
    pub old_bytes: usize,
    pub new_bytes: usize,
    pub old_params: usize,
    pub new_params: usize,
}

impl DiffReport {
    /// True when anything was added, removed, or changed — drives the exit code
    /// (`1` like `diff`, vs `0` when the two checkpoints are structurally identical).
    pub fn has_differences(&self) -> bool {
        !self.tensors_removed.is_empty()
            || !self.tensors_added.is_empty()
            || !self.tensors_changed.is_empty()
            || !self.meta_removed.is_empty()
            || !self.meta_added.is_empty()
            || !self.meta_changed.is_empty()
    }

    /// Render the report as plain text: a `---`/`+++` header naming the two sides,
    /// then a counts line and a `- removed / + added / ~ changed` list for tensors,
    /// then the same for metadata (unless `opts.metadata` is false). Entries are
    /// collapsed by name template + change when `opts.group`; colourised per
    /// `opts.color`. The counts lines always report raw entry totals.
    pub fn render(&self, old_label: &str, new_label: &str, opts: DiffOpts) -> String {
        // `--full` (no grouping) renders each entry as its own singleton group.
        let grouped = |items: &[(String, TensorSig)]| {
            if opts.group {
                group_entries(items)
            } else {
                singletons(items)
            }
        };

        let mut s = String::new();
        // Old side red, new side green — the same convention as the entries/totals.
        let _ = writeln!(s, "{}", paint(&format!("--- {old_label}"), opts.color, RED));
        let _ = writeln!(
            s,
            "{}",
            paint(&format!("+++ {new_label}"), opts.color, GREEN)
        );

        // Spell out what was (and wasn't) compared, and what the -/+/~ markers on
        // the summary and the entries below mean.
        let scope = if opts.values {
            "scope: tensor structure (name, dtype, shape) + element values"
        } else {
            "scope: tensor structure (name, dtype, shape) — element values not compared"
        };
        let _ = writeln!(s, "{}", paint(scope, opts.color, DIM));
        let _ = writeln!(
            s,
            "{}",
            paint("legend: - removed, + added, ~ changed", opts.color, DIM)
        );

        // Overall change: total on-disk size and parameter count (absolute +
        // relative %); the per-tensor breakdown follows.
        let _ = writeln!(s);
        let _ = writeln!(
            s,
            "{}",
            totals_line(
                "size",
                self.old_bytes,
                self.new_bytes,
                opts.color,
                format_size
            )
        );
        let _ = writeln!(
            s,
            "{}",
            totals_line(
                "params",
                self.old_params,
                self.new_params,
                opts.color,
                format_parameters
            )
        );

        let _ = writeln!(
            s,
            "\ntensors: -{} +{} ~{} ({} unchanged)",
            self.tensors_removed.len(),
            self.tensors_added.len(),
            self.tensors_changed.len(),
            self.tensors_unchanged,
        );
        for g in grouped(&self.tensors_removed) {
            let line = format!(
                "- {}  [{}]",
                display_name(&g.template, &g.indices),
                g.key.render()
            );
            let _ = writeln!(
                s,
                "  {}{}",
                paint(&line, opts.color, RED),
                count_suffix(g.count)
            );
        }
        for g in grouped(&self.tensors_added) {
            let line = format!(
                "+ {}  [{}]",
                display_name(&g.template, &g.indices),
                g.key.render()
            );
            let _ = writeln!(
                s,
                "  {}{}",
                paint(&line, opts.color, GREEN),
                count_suffix(g.count)
            );
        }
        for g in group_changed(&self.tensors_changed, opts.group) {
            let name = display_name(&g.template, &g.indices);
            let suffix = count_suffix(g.count);
            if g.old == g.new {
                // Same dtype & shape — only the values / distribution changed.
                let reason = if g.values.is_some_and(|v| v.differing > 0) {
                    "values differ"
                } else {
                    "distribution differs"
                };
                let _ = writeln!(s, "  ~ {name}  [{}]  ({reason}){suffix}", g.old.render());
            } else {
                let (old, new) = render_change(&g.old, &g.new, opts.color);
                let _ = writeln!(s, "  ~ {name}  [{old}] → [{new}]{suffix}");
            }
            if opts.values {
                match &g.values {
                    Some(vd) if vd.differing > 0 => {
                        let _ = writeln!(s, "{}", value_line(vd));
                    }
                    Some(_) => {
                        let _ = writeln!(s, "    values: identical");
                    }
                    // --values requested but a shape change made it undefined.
                    None => {
                        let _ = writeln!(s, "    values: not compared (shapes differ)");
                    }
                }
            }
            if opts.histogram {
                let _ = writeln!(s, "{}", histogram_line(&g.hist_tvds, g.hist_bins));
            }
        }

        if opts.metadata {
            let _ = writeln!(
                s,
                "\nmetadata: -{} +{} ~{} ({} unchanged)",
                self.meta_removed.len(),
                self.meta_added.len(),
                self.meta_changed.len(),
                self.meta_unchanged,
            );
            let meta_grouped = |items: &[(String, MetaVal)]| {
                if opts.group {
                    group_entries(items)
                } else {
                    singletons(items)
                }
            };
            for g in meta_grouped(&self.meta_removed) {
                let line = format!(
                    "- {} = {}",
                    display_name(&g.template, &g.indices),
                    quote_trunc(&g.key.value)
                );
                let _ = writeln!(
                    s,
                    "  {}{}",
                    paint(&line, opts.color, RED),
                    count_suffix(g.count)
                );
            }
            for g in meta_grouped(&self.meta_added) {
                let line = format!(
                    "+ {} = {}",
                    display_name(&g.template, &g.indices),
                    quote_trunc(&g.key.value)
                );
                let _ = writeln!(
                    s,
                    "  {}{}",
                    paint(&line, opts.color, GREEN),
                    count_suffix(g.count)
                );
            }
            let mchanged: Vec<(String, (MetaVal, MetaVal))> = self
                .meta_changed
                .iter()
                .map(|c| (c.name.clone(), (c.old.clone(), c.new.clone())))
                .collect();
            let mchanged_groups = if opts.group {
                group_entries(&mchanged)
            } else {
                singletons(&mchanged)
            };
            for g in &mchanged_groups {
                let (old, new) = (&g.key.0, &g.key.1);
                let name = display_name(&g.template, &g.indices);
                let suffix = count_suffix(g.count);
                if old.value != new.value {
                    // Prefer a git-style line diff for long values: JSON is
                    // pretty-printed first (so even a minified one-liner diffs
                    // line-by-line), else any already-multi-line value is diffed
                    // as-is. Short single-line values stay inline, windowed around
                    // where they first diverge.
                    let w = meta_line_width();
                    let line_pair = match (pretty_json(&old.value, w), pretty_json(&new.value, w)) {
                        // JSON on both sides: decide purely on the width-aware pretty
                        // form — line diff if it expanded, else inline (small JSON
                        // stays compact even if its raw form had newlines).
                        (Some(o), Some(n)) => {
                            (is_multiline(&o) || is_multiline(&n)).then_some((o, n))
                        }
                        // Non-JSON: line diff a raw multi-line value; else inline.
                        _ if is_multiline(&old.value) || is_multiline(&new.value) => {
                            Some((old.value.clone(), new.value.clone()))
                        }
                        _ => None,
                    };
                    if let Some((o, n)) = line_pair {
                        let _ = writeln!(s, "  ~ {name}:{suffix}");
                        write_meta_line_diff(&mut s, &o, &n, opts.color);
                    } else {
                        let (o, n) = quote_diff(&old.value, &new.value);
                        let _ = writeln!(
                            s,
                            "  ~ {name} = {} → {}{suffix}",
                            paint(&o, opts.color, RED),
                            paint(&n, opts.color, GREEN),
                        );
                    }
                } else {
                    // Same value, different declared type.
                    let _ = writeln!(
                        s,
                        "  ~ {name} (type {} → {}){suffix}",
                        paint(&old.value_type, opts.color, RED),
                        paint(&new.value_type, opts.color, GREEN),
                    );
                }
            }
        } else {
            // Make it obvious the metadata was deliberately left out, rather than
            // silently showing only the tensors section, and say why.
            let reason = if opts.filtered {
                "filtered subset"
            } else {
                "--only-tensors"
            };
            let _ = writeln!(s, "\nmetadata: not compared ({reason})");
        }
        s
    }
}

/// Structural comparison of two checkpoint summaries (old → new). Tensor values
/// are not read; see [`compare_with`].
pub fn compare(old: &CheckpointSummary, new: &CheckpointSummary) -> DiffReport {
    compare_with(old, new, |_| TensorExtras::default())
}

/// Like [`compare`] but also runs `extras_fn(name)` for each tensor present in
/// both checkpoints — its element-value (`--values`) and/or distribution
/// (`--histogram`) comparison. A tensor counts as changed when its dtype or shape
/// differs *or* its extras indicate a difference, so a values-only / distribution
/// change surfaces even when the signature is unchanged.
pub fn compare_with(
    old: &CheckpointSummary,
    new: &CheckpointSummary,
    extras_fn: impl Fn(&str) -> TensorExtras,
) -> DiffReport {
    let mut tensors_removed = Vec::new();
    let mut tensors_changed = Vec::new();
    let mut tensors_unchanged = 0usize;
    for (name, osig) in &old.tensors {
        let Some(nsig) = new.tensors.get(name) else {
            tensors_removed.push((name.clone(), osig.clone()));
            continue;
        };
        let extras = extras_fn(name);
        if nsig != osig || extras.differ() {
            tensors_changed.push(TensorChange {
                name: name.clone(),
                old: osig.clone(),
                new: nsig.clone(),
                values: extras.values,
                histogram: extras.histogram,
            });
        } else {
            tensors_unchanged += 1;
        }
    }
    let tensors_added: Vec<_> = new
        .tensors
        .iter()
        .filter(|(name, _)| !old.tensors.contains_key(*name))
        .map(|(name, sig)| (name.clone(), sig.clone()))
        .collect();

    let mut meta_removed = Vec::new();
    let mut meta_changed = Vec::new();
    let mut meta_unchanged = 0usize;
    for (name, oval) in &old.metadata {
        match new.metadata.get(name) {
            None => meta_removed.push((name.clone(), oval.clone())),
            Some(nval) if nval == oval => meta_unchanged += 1,
            Some(nval) => meta_changed.push(MetaChange {
                name: name.clone(),
                old: oval.clone(),
                new: nval.clone(),
            }),
        }
    }
    let meta_added: Vec<_> = new
        .metadata
        .iter()
        .filter(|(name, _)| !old.metadata.contains_key(*name))
        .map(|(name, v)| (name.clone(), v.clone()))
        .collect();

    DiffReport {
        tensors_removed,
        tensors_added,
        tensors_changed,
        tensors_unchanged,
        meta_removed,
        meta_added,
        meta_changed,
        meta_unchanged,
        old_bytes: old.total_bytes,
        new_bytes: new.total_bytes,
        old_params: old.total_params,
        new_params: new.total_params,
    }
}

/// Whether the focused (`--tensor`) diff counts as a difference — drives exit `1`
/// vs `0`. The tensor differs if it's present on only one side, its signature
/// changed, or (same signature) its values changed.
pub fn tensor_focus_differs(
    old: Option<&TensorSig>,
    new: Option<&TensorSig>,
    values: Option<&ValueCmp>,
) -> bool {
    match (old, new) {
        (Some(o), Some(n)) => o != n || matches!(values, Some(ValueCmp::Differ(_))),
        // Present on only one side (the both-absent case is handled as "not found"
        // by the caller, which exits 2 before reaching here).
        _ => true,
    }
}

/// Render the focused single-tensor diff: the `[old] → [new]` signature line (or
/// added/removed/identical), then an indented `values:` line from the element
/// comparison when both sides exist.
pub fn render_tensor_focus(
    old_label: &str,
    new_label: &str,
    name: &str,
    old: Option<&TensorSig>,
    new: Option<&TensorSig>,
    values: Option<&ValueCmp>,
    color: bool,
) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "{}", paint(&format!("--- {old_label}"), color, RED));
    let _ = writeln!(s, "{}", paint(&format!("+++ {new_label}"), color, GREEN));
    let _ = writeln!(s);
    match (old, new) {
        (Some(o), None) => {
            let line = format!("- {name}  [{}]  (only in old)", o.render());
            let _ = writeln!(s, "  {}", paint(&line, color, RED));
        }
        (None, Some(n)) => {
            let line = format!("+ {name}  [{}]  (only in new)", n.render());
            let _ = writeln!(s, "  {}", paint(&line, color, GREEN));
        }
        (Some(o), Some(n)) if o == n => {
            // Same dtype & shape: the only possible difference is in the values.
            match values {
                Some(ValueCmp::Differ(vd)) => {
                    let _ = writeln!(s, "  ~ {name}  [{}]  (values differ)", o.render());
                    let _ = writeln!(s, "{}", value_line(vd));
                }
                Some(ValueCmp::Skipped(why)) => {
                    let _ = writeln!(s, "  = {name}  [{}]", o.render());
                    let _ = writeln!(s, "    values: not compared ({why})");
                }
                _ => {
                    let _ = writeln!(s, "  = {name}  [{}]  (identical)", o.render());
                }
            }
        }
        (Some(o), Some(n)) => {
            // dtype and/or shape changed.
            let (orender, nrender) = render_change(o, n, color);
            let _ = writeln!(s, "  ~ {name}  [{orender}] → [{nrender}]");
            match values {
                Some(ValueCmp::Differ(vd)) => {
                    let _ = writeln!(s, "{}", value_line(vd));
                }
                Some(ValueCmp::Identical) => {
                    let _ = writeln!(s, "    values: identical");
                }
                Some(ValueCmp::Skipped(why)) => {
                    let _ = writeln!(s, "    values: not compared ({why})");
                }
                None => {}
            }
        }
        (None, None) => {}
    }
    s
}

/// The indented `histogram:` summary line for a group's distribution shift(s):
/// the total variation distance (max & mean across the group).
fn histogram_line(tvds: &[f64], bins: usize) -> String {
    if tvds.is_empty() {
        return "    histogram: not compared (shapes differ)".to_string();
    }
    let max = tvds.iter().copied().fold(0.0_f64, f64::max);
    if tvds.len() == 1 {
        format!("    histogram: TVD {} ({bins} bins)", fmt_delta(max))
    } else {
        let mean = tvds.iter().sum::<f64>() / tvds.len() as f64;
        format!(
            "    histogram: TVD max {} mean {} ({bins} bins)",
            fmt_delta(max),
            fmt_delta(mean)
        )
    }
}

/// The full per-tensor histogram comparison table for `diff --tensor --histogram`:
/// one row per shared bin with its label and the old / new counts and delta. Only
/// bins where at least one side is non-empty are shown.
pub fn render_histogram_table(name: &str, hd: &HistogramDiff, color: bool) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "  histogram of {name}  ({} bins, TVD {})",
        hd.n,
        fmt_delta(hd.tvd())
    );
    let _ = writeln!(
        s,
        "    {:>18}  {:>12}  {:>12}  {:>12}",
        "bin", "old", "new", "Δ"
    );
    for i in 0..hd.n {
        let (o, n) = (
            hd.old.get(i).copied().unwrap_or(0),
            hd.new.get(i).copied().unwrap_or(0),
        );
        if o == 0 && n == 0 {
            continue;
        }
        let delta = n as i64 - o as i64;
        let delta_s = match delta.cmp(&0) {
            std::cmp::Ordering::Greater => paint(&format!("+{delta}"), color, GREEN),
            std::cmp::Ordering::Less => paint(&format!("{delta}"), color, RED),
            std::cmp::Ordering::Equal => "0".to_string(),
        };
        let _ = writeln!(
            s,
            "    {:>18}  {o:>12}  {n:>12}  {delta_s:>12}",
            bin_label(hd.bins, i, hd.n)
        );
    }
    if hd.old_nonfinite > 0 || hd.new_nonfinite > 0 {
        let _ = writeln!(
            s,
            "    {:>18}  {:>12}  {:>12}",
            "non-finite", hd.old_nonfinite, hd.new_nonfinite
        );
    }
    s
}

/// A short label for histogram bin `i` of `n`: the integer (or integer range) for
/// `IntBins`, or the `[lo, hi)` interval for `Range`.
fn bin_label(bins: HistBins, i: usize, n: usize) -> String {
    match bins {
        HistBins::IntBins { start, step } => {
            let lo = start + i as i64 * step;
            if step == 1 {
                format!("{lo}")
            } else {
                format!("{lo}..{}", lo + step - 1)
            }
        }
        HistBins::Range { lo, hi } => {
            let w = if n > 0 { (hi - lo) / n as f64 } else { 0.0 };
            fmt_delta(lo + i as f64 * w)
        }
    }
}

/// The indented `values:` summary line for a value difference.
fn value_line(vd: &ValueDiff) -> String {
    let mut line = format!(
        "    values: {} of {} differ  (max |Δ| {}, mean |Δ| {})",
        vd.differing,
        vd.elements,
        fmt_delta(vd.max_abs),
        fmt_delta(vd.mean_abs),
    );
    if vd.nonfinite_mismatch > 0 {
        let _ = write!(line, "  [{} non-finite mismatch]", vd.nonfinite_mismatch);
    }
    line
}

/// Format a difference magnitude compactly: fixed-point with trailing zeros
/// trimmed for everyday magnitudes, scientific for very small/large ones.
fn fmt_delta(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    let a = x.abs();
    if (1e-3..1e6).contains(&a) {
        let fixed = format!("{x:.6}");
        let trimmed = fixed.trim_end_matches('0').trim_end_matches('.');
        trimmed.to_string()
    } else {
        format!("{x:.3e}")
    }
}

/// Quote a metadata value for one-line display: flatten newlines to spaces and
/// truncate to a readable length (multi-line JSON blobs are common).
fn quote_trunc(v: &str) -> String {
    const MAX: usize = 60;
    let flat = v.replace(['\n', '\r'], " ");
    if flat.chars().count() > MAX {
        let head: String = flat.chars().take(MAX).collect();
        format!("\"{head}…\"")
    } else {
        format!("\"{flat}\"")
    }
}

/// Quote a *changed* value pair for one-line display, each windowed around the
/// first character where they differ (with `…` where truncated) — so the actual
/// change is visible even in a long JSON blob, where head-truncation would print
/// the same shared prefix for both sides. Newlines are flattened to spaces. Short
/// values that fit are shown in full.
fn quote_diff(old: &str, new: &str) -> (String, String) {
    const WINDOW: usize = 60; // chars shown per side
    const CTX: usize = 12; // shared context kept before the first difference
    let o: Vec<char> = old.replace(['\n', '\r'], " ").chars().collect();
    let n: Vec<char> = new.replace(['\n', '\r'], " ").chars().collect();
    let prefix = o.iter().zip(&n).take_while(|(a, b)| a == b).count();
    let start = prefix.saturating_sub(CTX);
    let render = |chars: &[char]| -> String {
        let end = (start + WINDOW).min(chars.len());
        let mut s = String::new();
        if start > 0 {
            s.push('…');
        }
        s.extend(&chars[start..end]);
        if end < chars.len() {
            s.push('…');
        }
        format!("\"{s}\"")
    };
    (render(&o), render(&n))
}

/// Max diff lines shown for one changed metadata value before the rest is
/// summarised as a count — bounds a huge value (e.g. a big `weight_map`).
const MAX_META_DIFF_LINES: usize = 20;

/// Max columns for a single metadata diff line before it's clipped with `…` —
/// bounds a value with one enormous line (e.g. a nested, serialised tensor list
/// that pretty-JSON leaves as a single escaped string). Uses the terminal width
/// when attached, else a sane default; leaves room for the indent.
fn meta_line_width() -> usize {
    crossterm::terminal::size()
        .map(|(cols, _)| cols as usize)
        .unwrap_or(120)
        .saturating_sub(6)
        .max(40)
}

/// Window two long, differing lines each to `max` columns around where they first
/// diverge — `…<shared context><difference>…` — so a changed line whose values
/// share a long prefix still shows the actual change (rather than both clipping to
/// the same prefix). Keeps ~a quarter of the window as leading shared context.
fn window_pair(o: &str, n: &str, max: usize) -> (String, String) {
    let oc: Vec<char> = o.chars().collect();
    let nc: Vec<char> = n.chars().collect();
    let prefix = oc.iter().zip(&nc).take_while(|(a, b)| a == b).count();
    let start = prefix.saturating_sub(max / 4);
    let render = |chars: &[char]| -> String {
        let end = (start + max).min(chars.len());
        let mut s = String::new();
        if start > 0 {
            s.push('…');
        }
        s.extend(&chars[start..end]);
        if end < chars.len() {
            s.push('…');
        }
        s
    };
    (render(&oc), render(&nc))
}

/// Clip `line` to `max` columns, appending `…` when truncated.
fn clip_width(line: String, max: usize) -> String {
    if line.chars().count() <= max {
        line
    } else {
        line.chars()
            .take(max.saturating_sub(1))
            .chain(std::iter::once('…'))
            .collect()
    }
}

/// Whether a metadata value spans multiple lines — the cue to show a line diff
/// rather than a one-line `old → new` (typically a pretty-printed JSON blob).
fn is_multiline(v: &str) -> bool {
    v.contains('\n')
}

/// Pretty-print `v` if it parses as JSON, expanded to one field/element per line
/// so it diffs readably — but **width-aware**: any object/array whose one-line
/// form fits in `width` stays inline (a small `{"bit_widths": [3, 3, 3]}` isn't
/// blown up into eight lines). `None` when `v` isn't JSON.
fn pretty_json(v: &str, width: usize) -> Option<String> {
    let value: Value = serde_json::from_str(v.trim()).ok()?;
    let mut out = String::new();
    write_json(&mut out, &value, 0, 0, width);
    Some(out)
}

/// A JSON value on one line with `: `/`, ` separators (no newlines) — the inline
/// form the width test compares against.
fn compact_json(v: &Value) -> String {
    match v {
        Value::Object(m) => {
            let items: Vec<String> = m
                .iter()
                .map(|(k, val)| {
                    format!(
                        "{}: {}",
                        serde_json::to_string(k).unwrap_or_default(),
                        compact_json(val)
                    )
                })
                .collect();
            format!("{{{}}}", items.join(", "))
        }
        Value::Array(a) => {
            let items: Vec<String> = a.iter().map(compact_json).collect();
            format!("[{}]", items.join(", "))
        }
        other => other.to_string(),
    }
}

/// Write `v` starting at column `col`: inline (via [`compact_json`]) when it fits
/// in `width`, else expanded one child per line at `indent`, recursing so nested
/// values that fit stay inline.
fn write_json(out: &mut String, v: &Value, indent: usize, col: usize, width: usize) {
    let compact = compact_json(v);
    if col + compact.chars().count() <= width || !matches!(v, Value::Object(_) | Value::Array(_)) {
        out.push_str(&compact);
        return;
    }
    let (pad, cpad) = ("  ".repeat(indent), "  ".repeat(indent + 1));
    match v {
        Value::Object(m) => {
            out.push_str("{\n");
            for (i, (k, val)) in m.iter().enumerate() {
                let key = serde_json::to_string(k).unwrap_or_default();
                out.push_str(&cpad);
                out.push_str(&key);
                out.push_str(": ");
                write_json(
                    out,
                    val,
                    indent + 1,
                    cpad.len() + key.chars().count() + 2,
                    width,
                );
                out.push_str(if i + 1 < m.len() { ",\n" } else { "\n" });
            }
            out.push_str(&pad);
            out.push('}');
        }
        Value::Array(a) => {
            out.push_str("[\n");
            for (i, val) in a.iter().enumerate() {
                out.push_str(&cpad);
                write_json(out, val, indent + 1, cpad.len(), width);
                out.push_str(if i + 1 < a.len() { ",\n" } else { "\n" });
            }
            out.push_str(&pad);
            out.push(']');
        }
        _ => {}
    }
}

/// Write a git-style line diff of two metadata values, indented under the entry
/// name: removed lines red `-`, added lines green `+`, a few lines of context
/// (dim), with `⋮` between hunks. Uses [`similar`] for the line matching. Capped
/// at [`MAX_META_DIFF_LINES`] so one huge value (e.g. a big `weight_map`) can't
/// flood the output — the remainder is summarised as a count.
fn write_meta_line_diff(s: &mut String, old: &str, new: &str, color: bool) {
    use similar::{DiffOp, TextDiff};
    let width = meta_line_width();
    let diff = TextDiff::from_lines(old, new);
    let (ol, nl) = (diff.old_slices(), diff.new_slices());
    let strip = |line: &str| line.strip_suffix('\n').unwrap_or(line).to_string();
    // Render the diff lines (with `⋮` between hunks), tallying total changes.
    let mut lines: Vec<String> = Vec::new();
    let (mut removed, mut added) = (0usize, 0usize);
    // A removed/added/context line, clipped to the width from the left.
    let push = |lines: &mut Vec<String>, sign: char, code: &str, line: &str| {
        lines.push(paint(
            &clip_width(format!("{sign} {line}"), width),
            color,
            code,
        ));
    };
    for (hunk, group) in diff.grouped_ops(3).iter().enumerate() {
        if hunk > 0 {
            lines.push(paint("⋮", color, DIM));
        }
        for op in group {
            match *op {
                DiffOp::Equal { old_index, len, .. } => {
                    for l in &ol[old_index..old_index + len] {
                        push(&mut lines, ' ', DIM, &strip(l));
                    }
                }
                DiffOp::Delete {
                    old_index, old_len, ..
                } => {
                    removed += old_len;
                    for l in &ol[old_index..old_index + old_len] {
                        push(&mut lines, '-', RED, &strip(l));
                    }
                }
                DiffOp::Insert {
                    new_index, new_len, ..
                } => {
                    added += new_len;
                    for l in &nl[new_index..new_index + new_len] {
                        push(&mut lines, '+', GREEN, &strip(l));
                    }
                }
                DiffOp::Replace {
                    old_index,
                    old_len,
                    new_index,
                    new_len,
                } => {
                    removed += old_len;
                    added += new_len;
                    // Pair replaced lines old[k]↔new[k]. When a pair is too wide to
                    // show whole, window each around where they diverge (… diff …)
                    // rather than clipping both to the same shared prefix.
                    let pairs = old_len.min(new_len);
                    for k in 0..pairs {
                        let (o, n) = (strip(ol[old_index + k]), strip(nl[new_index + k]));
                        if o.chars().count() > width || n.chars().count() > width {
                            let (ow, nw) = window_pair(&o, &n, width.saturating_sub(2));
                            lines.push(paint(&format!("- {ow}"), color, RED));
                            lines.push(paint(&format!("+ {nw}"), color, GREEN));
                        } else {
                            lines.push(paint(&format!("- {o}"), color, RED));
                            lines.push(paint(&format!("+ {n}"), color, GREEN));
                        }
                    }
                    for l in &ol[old_index + pairs..old_index + old_len] {
                        push(&mut lines, '-', RED, &strip(l));
                    }
                    for l in &nl[new_index + pairs..new_index + new_len] {
                        push(&mut lines, '+', GREEN, &strip(l));
                    }
                }
            }
        }
    }
    for line in lines.iter().take(MAX_META_DIFF_LINES) {
        let _ = writeln!(s, "      {line}");
    }
    if lines.len() > MAX_META_DIFF_LINES {
        let note = format!(
            "… {} more diff line(s) — {removed} removed, {added} added in total",
            lines.len() - MAX_META_DIFF_LINES
        );
        let _ = writeln!(s, "      {}", paint(&note, color, DIM));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sig(dtype: &str, shape: &[usize]) -> TensorSig {
        TensorSig {
            dtype: dtype.to_string(),
            shape: shape.to_vec(),
        }
    }

    #[test]
    fn totals_line_shows_absolute_and_relative_change() {
        assert_eq!(
            totals_line("size", 100, 150, false, format_size),
            "size: 100 B → 150 B (+50 B, +50.0%)"
        );
        assert_eq!(
            totals_line("params", 56, 40, false, format_parameters),
            "params: 56 → 40 (-16, -28.6%)"
        );
        // equal → unchanged; zero baseline → no percentage
        assert_eq!(
            totals_line("size", 100, 100, false, format_size),
            "size: 100 B (unchanged)"
        );
        assert_eq!(
            totals_line("size", 0, 100, false, format_size),
            "size: 0 B → 100 B (+100 B)"
        );
        // Coloured like the tensor diff: old red, new green, delta dimmed.
        assert_eq!(
            totals_line("size", 100, 150, true, format_size),
            "size: \x1b[31m100 B\x1b[0m → \x1b[32m150 B\x1b[0m (\x1b[2m+50 B, +50.0%\x1b[0m)"
        );
    }
    fn mv(value: &str, ty: &str) -> MetaVal {
        MetaVal {
            value: value.to_string(),
            value_type: ty.to_string(),
        }
    }
    fn summary(tensors: &[(&str, TensorSig)], metadata: &[(&str, MetaVal)]) -> CheckpointSummary {
        CheckpointSummary {
            tensors: tensors
                .iter()
                .map(|(n, s)| (n.to_string(), s.clone()))
                .collect(),
            metadata: metadata
                .iter()
                .map(|(n, v)| (n.to_string(), v.clone()))
                .collect(),
            total_bytes: 0,
            total_params: 0,
        }
    }

    #[test]
    fn change_colours_only_differing_dtype_and_dims() {
        // dtype F16→U16 and only the first dim 256→64 differ; 3072/1540 are shared.
        let (o, n) = render_change(
            &sig("F16", &[256, 3072, 1540]),
            &sig("U16", &[64, 3072, 1540]),
            true,
        );
        assert!(o.contains(&format!("{RED}F16{RESET}"))); // dtype coloured
        assert!(n.contains(&format!("{GREEN}U16{RESET}")));
        assert!(o.contains(&format!("{RED}256{RESET}"))); // changed dim coloured
        assert!(n.contains(&format!("{GREEN}64{RESET}")));
        // Unchanged dims are plain (not wrapped in a colour code).
        assert!(o.contains(", 3072, 1540)") && n.contains(", 3072, 1540)"));
    }

    #[test]
    fn change_leaves_dtype_plain_when_only_a_dim_differs() {
        let (o, _n) = render_change(&sig("F16", &[4, 8]), &sig("F16", &[2, 8]), true);
        assert!(!o.contains(&format!("{RED}F16"))); // dtype unchanged → not coloured
        assert!(o.contains(&format!("({RED}4{RESET}, 8)"))); // only dim0 coloured
    }

    #[test]
    fn change_colours_whole_shape_when_ranks_differ() {
        let (o, _n) = render_change(&sig("F16", &[4, 8]), &sig("F16", &[32]), true);
        assert!(o.contains(&format!("{RED}(4, 8){RESET}")));
    }

    #[test]
    fn identical_checkpoints_have_no_differences() {
        let a = summary(&[("w", sig("F16", &[2, 2]))], &[("k", mv("v", "string"))]);
        let b = summary(&[("w", sig("F16", &[2, 2]))], &[("k", mv("v", "string"))]);
        let r = compare(&a, &b);
        assert!(!r.has_differences());
        assert_eq!(r.tensors_unchanged, 1);
        assert_eq!(r.meta_unchanged, 1);
    }

    #[test]
    fn classifies_added_removed_changed_tensors() {
        let old = summary(
            &[
                ("keep", sig("F16", &[2, 2])),
                ("gone", sig("F32", &[8, 8])),
                ("retyped", sig("F32", &[4, 4])),
                ("reshaped", sig("F16", &[10, 4])),
            ],
            &[],
        );
        let new = summary(
            &[
                ("keep", sig("F16", &[2, 2])),
                ("fresh", sig("BF16", &[1, 1])),
                ("retyped", sig("BF16", &[4, 4])),
                ("reshaped", sig("F16", &[20, 2])),
            ],
            &[],
        );
        let r = compare(&old, &new);
        assert!(r.has_differences());
        assert_eq!(
            r.tensors_removed
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
            ["gone"]
        );
        assert_eq!(
            r.tensors_added
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
            ["fresh"]
        );
        let changed: Vec<_> = r.tensors_changed.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(changed, ["reshaped", "retyped"]); // BTreeMap order
        assert_eq!(r.tensors_unchanged, 1);
    }

    #[test]
    fn classifies_metadata_changes_including_type_only() {
        let old = summary(
            &[],
            &[
                ("same", mv("1", "int")),
                ("v", mv("0.4", "string")),
                ("typed", mv("1", "int")),
            ],
        );
        let new = summary(
            &[],
            &[
                ("same", mv("1", "int")),
                ("v", mv("0.5", "string")),
                ("typed", mv("1", "float")),
                ("extra", mv("x", "string")),
            ],
        );
        let r = compare(&old, &new);
        assert_eq!(
            r.meta_added
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
            ["extra"]
        );
        assert!(r.meta_removed.is_empty());
        let changed: Vec<_> = r.meta_changed.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(changed, ["typed", "v"]);
        assert_eq!(r.meta_unchanged, 1);
        // The type-only change renders as a "(type … → …)" note, not a value diff.
        let out = r.render("old", "new", PLAIN);
        assert!(out.contains("~ typed (type int → float)"), "{out}");
        assert!(out.contains("~ v = \"0.4\" → \"0.5\""), "{out}");
    }

    #[test]
    fn render_notes_when_metadata_excluded() {
        let old = summary(&[("w", sig("F16", &[2, 2]))], &[("k", mv("a", "string"))]);
        let new = summary(&[("w", sig("F16", &[2, 2]))], &[("k", mv("b", "string"))]);
        let r = compare(&old, &new);
        // Default: the metadata change is shown.
        assert!(r.render("o", "n", PLAIN).contains("metadata: -0 +0 ~1"));
        // --only-tensors: a clear note instead, and no per-entry metadata lines.
        let without = r.render(
            "o",
            "n",
            DiffOpts {
                metadata: false,
                ..PLAIN
            },
        );
        assert!(
            without.contains("metadata: not compared (--only-tensors)"),
            "{without}"
        );
        assert!(!without.contains("  ~ k"), "{without}");
    }

    #[test]
    fn full_value_diff_promotes_values_only_change() {
        // Same dtype & shape on both sides; a value comparison says they differ.
        let old = summary(&[("model.layers.0.w", sig("U8", &[4]))], &[]);
        let new = summary(&[("model.layers.0.w", sig("U8", &[4]))], &[]);
        let r = compare_with(&old, &new, |_| TensorExtras {
            values: Some(ValueDiff {
                elements: 4,
                differing: 2,
                max_abs: 7.0,
                mean_abs: 3.5,
                nonfinite_mismatch: 0,
            }),
            histogram: None,
        });
        // The structurally-identical tensor is now a change.
        assert_eq!(r.tensors_changed.len(), 1);
        assert_eq!(r.tensors_unchanged, 0);
        let out = r.render(
            "o",
            "n",
            DiffOpts {
                values: true,
                ..PLAIN
            },
        );
        assert!(
            out.contains("~ model.layers.0.w  [U8 (4)]  (values differ)"),
            "{out}"
        );
        assert!(
            out.contains("values: 2 of 4 differ  (max |Δ| 7, mean |Δ| 3.5)"),
            "{out}"
        );
    }

    #[test]
    fn full_value_diff_aggregates_within_a_group() {
        // Two layers, each a values-only change → collapse, stats aggregated.
        let names = ["model.layers.0.w", "model.layers.1.w"];
        let mk = || CheckpointSummary {
            tensors: names
                .iter()
                .map(|n| (n.to_string(), sig("U8", &[4])))
                .collect(),
            metadata: Default::default(),
            total_bytes: 0,
            total_params: 0,
        };
        let per = ValueDiff {
            elements: 4,
            differing: 1,
            max_abs: 2.0,
            mean_abs: 0.5,
            nonfinite_mismatch: 0,
        };
        let r = compare_with(&mk(), &mk(), |_| TensorExtras {
            values: Some(per),
            histogram: None,
        });
        let out = r.render(
            "o",
            "n",
            DiffOpts {
                values: true,
                ..PLAIN
            },
        );
        // One collapsed line with the aggregate (8 elements, 2 differing, max 2).
        assert!(
            out.contains("~ model.layers.{0-1}.w  [U8 (4)]  (values differ)  (×2)"),
            "{out}"
        );
        assert!(
            out.contains("values: 2 of 8 differ  (max |Δ| 2, mean |Δ| 0.5)"),
            "{out}"
        );
    }

    #[test]
    fn color_highlights_only_the_changed_token() {
        // dtype changed, shape same → colour the dtype, not the shape.
        let old = summary(&[("w", sig("F16", &[2, 2]))], &[]);
        let new = summary(&[("w", sig("BF16", &[2, 2]))], &[]);
        let out = compare(&old, &new).render(
            "o",
            "n",
            DiffOpts {
                color: true,
                ..PLAIN
            },
        );
        assert!(out.contains(&format!("{RED}F16{RESET}")), "{out:?}");
        assert!(out.contains(&format!("{GREEN}BF16{RESET}")), "{out:?}");
        // The unchanged shape isn't wrapped in a colour code.
        assert!(!out.contains(&format!("{RED}(2, 2){RESET}")), "{out:?}");
    }

    #[test]
    fn groups_repeated_per_index_changes() {
        // The same dtype change across layers 0..=3 collapses to one line.
        let mk = |dt: &str| {
            (0..4)
                .map(|n| (format!("model.layers.{n}.mlp.weight"), sig(dt, &[8])))
                .collect::<Vec<_>>()
        };
        let old = CheckpointSummary {
            tensors: mk("F16").into_iter().collect(),
            metadata: Default::default(),
            total_bytes: 0,
            total_params: 0,
        };
        let new = CheckpointSummary {
            tensors: mk("BF16").into_iter().collect(),
            metadata: Default::default(),
            total_bytes: 0,
            total_params: 0,
        };
        let r = compare(&old, &new);
        // Grouped (default): one collapsed line with the range and ×count.
        let grouped = r.render("o", "n", PLAIN);
        assert!(
            grouped.contains("~ model.layers.{0-3}.mlp.weight  [F16 (8)] → [BF16 (8)]  (×4)"),
            "{grouped}"
        );
        assert_eq!(
            grouped.matches(".mlp.weight").count(),
            1,
            "should be one line:\n{grouped}"
        );
        // The counts line still reports the true total (4 changed).
        assert!(grouped.contains("tensors: -0 +0 ~4"), "{grouped}");

        // `--full` (group off): every layer listed, no count suffix.
        let full = r.render(
            "o",
            "n",
            DiffOpts {
                group: false,
                ..PLAIN
            },
        );
        assert_eq!(
            full.matches(".mlp.weight").count(),
            4,
            "should list all four:\n{full}"
        );
        assert!(full.contains("~ model.layers.0.mlp.weight"), "{full}");
        assert!(!full.contains("(×"), "no count suffix when full:\n{full}");
    }

    #[test]
    fn compact_int_ranges_merges_runs() {
        assert_eq!(compact_int_ranges(&[0, 1, 2, 3]), "0-3");
        assert_eq!(compact_int_ranges(&[0, 1, 2, 5, 7, 8]), "0-2,5,7-8");
        assert_eq!(compact_int_ranges(&[4]), "4");
    }

    #[test]
    fn templatize_replaces_digit_runs() {
        let (t, idx) = templatize("model.layers.12.experts.3.weight");
        assert_eq!(t, "model.layers.{}.experts.{}.weight");
        assert_eq!(idx, ["12", "3"]);
    }

    #[test]
    fn pretty_json_is_width_aware() {
        // Small object fits on one line — not blown up.
        assert!(!pretty_json(r#"{"a":1,"b":2}"#, 80).unwrap().contains('\n'));
        // Too wide → expanded, one field per line.
        let big = format!(
            r#"{{"items":[{}]}}"#,
            (0..40)
                .map(|i| format!(r#""x{i}""#))
                .collect::<Vec<_>>()
                .join(",")
        );
        assert!(pretty_json(&big, 80).unwrap().contains('\n'));
        // A nested small object stays inline even inside an expanded parent.
        let nested = format!(r#"{{"pad":"{}","q":{{"bits":[3,3,3]}}}}"#, "z".repeat(90));
        assert!(
            pretty_json(&nested, 80)
                .unwrap()
                .contains(r#""q": {"bits": [3, 3, 3]}"#),
            "{:?}",
            pretty_json(&nested, 80)
        );
        assert!(pretty_json("not json", 80).is_none());
        assert!(pretty_json("d5f887bb41", 80).is_none());
    }

    #[test]
    fn changed_large_json_metadata_renders_as_a_line_diff() {
        let mv = |v: &str| MetaVal {
            value: v.to_string(),
            value_type: "string".to_string(),
        };
        // A JSON object large enough to expand, with one field changed.
        let obj = |val: &str| {
            let mut fields: Vec<String> = (0..20).map(|i| format!(r#""k{i}":"x""#)).collect();
            fields.push(format!(r#""v":"{val}""#));
            format!("{{{}}}", fields.join(","))
        };
        let old = summary(&[], &[("spec", mv(&obj("old")))]);
        let new = summary(&[], &[("spec", mv(&obj("new")))]);
        let out = compare(&old, &new).render("o", "n", PLAIN);
        assert!(out.contains("~ spec:"), "{out}"); // line-diff header, not `= … → …`
        let line = |sign: &str, needle: &str| {
            out.lines()
                .any(|l| l.trim_start().starts_with(sign) && l.contains(needle))
        };
        assert!(line("- ", r#""v": "old""#), "{out}");
        assert!(line("+ ", r#""v": "new""#), "{out}");
    }

    #[test]
    fn long_changed_line_is_windowed_around_its_difference() {
        // A single changed line whose two versions share a long prefix: each is
        // windowed around the divergence (AAA/BBB visible), not clipped to the
        // shared prefix.
        let old = format!("{{\n  \"x\": \"{}AAA\"\n}}", "z".repeat(300));
        let new = format!("{{\n  \"x\": \"{}BBB\"\n}}", "z".repeat(300));
        let mut s = String::new();
        write_meta_line_diff(&mut s, &old, &new, false);
        assert!(
            s.contains("AAA") && s.contains("BBB") && s.contains('…'),
            "{s}"
        );
    }

    #[test]
    fn large_metadata_line_diff_is_capped() {
        let old = (0..100)
            .map(|i| format!("line {i} old"))
            .collect::<Vec<_>>()
            .join("\n");
        let new = (0..100)
            .map(|i| format!("line {i} new"))
            .collect::<Vec<_>>()
            .join("\n");
        let mut s = String::new();
        write_meta_line_diff(&mut s, &old, &new, false);
        let n = s.lines().count();
        assert!(n <= MAX_META_DIFF_LINES + 1, "capped, got {n} lines");
        assert!(s.contains("more diff line"), "{s}");
    }

    #[test]
    fn multiline_metadata_shows_a_line_diff() {
        let old = "{\n  \"k\": 1,\n  \"v\": \"aaa\"\n}";
        let new = "{\n  \"k\": 1,\n  \"v\": \"bbb\"\n}";
        let mut s = String::new();
        write_meta_line_diff(&mut s, old, new, false);
        // The changed line shows as -/+, the unchanged line as context.
        assert!(s.contains("- ") && s.contains("aaa"), "{s}");
        assert!(s.contains("+ ") && s.contains("bbb"), "{s}");
        assert!(s.contains("\"k\": 1"), "{s}");
    }

    #[test]
    fn quote_diff_windows_around_the_first_difference() {
        // Long values sharing a prefix: both are windowed so the diverging token
        // shows (a plain head-truncation would print the same prefix for both).
        let old = format!("{}ALPHA-tail", "x".repeat(100));
        let new = format!("{}BETA-tail", "x".repeat(100));
        let (o, n) = quote_diff(&old, &new);
        assert!(o.starts_with("\"…") && o.contains("ALPHA"), "{o}");
        assert!(n.starts_with("\"…") && n.contains("BETA"), "{n}");
        // Short values are shown in full, no ellipsis.
        assert_eq!(
            quote_diff("d5f887bb41", "46c41d7cf4"),
            ("\"d5f887bb41\"".to_string(), "\"46c41d7cf4\"".to_string())
        );
    }

    #[test]
    fn quote_trunc_flattens_and_truncates() {
        assert_eq!(quote_trunc("a\nb"), "\"a b\"");
        let long = "x".repeat(100);
        let q = quote_trunc(&long);
        assert!(q.starts_with('"') && q.ends_with("…\""));
        assert_eq!(q.chars().count(), 60 + 3); // 60 chars + ellipsis + 2 quotes
    }

    #[test]
    fn fmt_delta_trims_and_switches_to_scientific() {
        assert_eq!(fmt_delta(0.0), "0");
        assert_eq!(fmt_delta(7.0), "7");
        assert_eq!(fmt_delta(0.5), "0.5");
        assert_eq!(fmt_delta(0.001953125), "0.001953");
        assert_eq!(fmt_delta(1e-8), "1.000e-8");
    }

    const PLAIN: DiffOpts = DiffOpts {
        color: false,
        metadata: true,
        group: true,
        values: false,
        histogram: false,
        filtered: false,
    };

    const COLOUR: DiffOpts = DiffOpts {
        color: true,
        metadata: true,
        group: true,
        values: false,
        histogram: false,
        filtered: false,
    };

    #[test]
    fn header_colours_old_red_new_green() {
        let s = summary(&[("a", sig("F16", &[2]))], &[]);
        let out = compare(&s, &s).render("OLD", "NEW", COLOUR);
        assert!(out.contains("\x1b[31m--- OLD\x1b[0m"), "{out}");
        assert!(out.contains("\x1b[32m+++ NEW\x1b[0m"), "{out}");
    }

    fn vd(differing: u64, elements: u64, max_abs: f64, mean_abs: f64) -> ValueDiff {
        ValueDiff {
            elements,
            differing,
            max_abs,
            mean_abs,
            nonfinite_mismatch: 0,
        }
    }

    #[test]
    fn focus_differs_predicate() {
        let a = sig("F16", &[2, 2]);
        let b = sig("BF16", &[2, 2]);
        // same sig, identical values → not a difference
        assert!(!tensor_focus_differs(
            Some(&a),
            Some(&a),
            Some(&ValueCmp::Identical)
        ));
        // same sig, values differ → a difference
        assert!(tensor_focus_differs(
            Some(&a),
            Some(&a),
            Some(&ValueCmp::Differ(vd(1, 4, 0.5, 0.1)))
        ));
        // differing sig → a difference regardless of values
        assert!(tensor_focus_differs(
            Some(&a),
            Some(&b),
            Some(&ValueCmp::Identical)
        ));
        // present on one side only → a difference
        assert!(tensor_focus_differs(Some(&a), None, None));
    }

    #[test]
    fn focus_render_same_sig_values_differ() {
        let a = sig("U8", &[4]);
        let out = render_tensor_focus(
            "old",
            "new",
            "w",
            Some(&a),
            Some(&a),
            Some(&ValueCmp::Differ(vd(4, 4, 7.0, 7.0))),
            false,
        );
        assert!(out.contains("~ w  [U8 (4)]  (values differ)"), "{out}");
        assert!(
            out.contains("values: 4 of 4 differ  (max |Δ| 7, mean |Δ| 7)"),
            "{out}"
        );
    }

    #[test]
    fn focus_render_identical_and_added_and_shape_skip() {
        let a = sig("F32", &[4]);
        let ident = render_tensor_focus(
            "o",
            "n",
            "w",
            Some(&a),
            Some(&a),
            Some(&ValueCmp::Identical),
            false,
        );
        assert!(ident.contains("= w  [F32 (4)]  (identical)"), "{ident}");

        let added = render_tensor_focus("o", "n", "w", None, Some(&a), None, false);
        assert!(added.contains("+ w  [F32 (4)]  (only in new)"), "{added}");

        let b = sig("F32", &[8]);
        let reshape = render_tensor_focus(
            "o",
            "n",
            "w",
            Some(&a),
            Some(&b),
            Some(&ValueCmp::Skipped("shapes differ".to_string())),
            false,
        );
        assert!(reshape.contains("~ w  [F32 (4)] → [F32 (8)]"), "{reshape}");
        assert!(
            reshape.contains("values: not compared (shapes differ)"),
            "{reshape}"
        );
    }

    // ---- TensorFilter ----

    fn glob(p: &str) -> Pattern {
        Pattern::new(p).unwrap()
    }

    #[test]
    fn filter_name_glob_matches_any() {
        let f = TensorFilter {
            name_globs: vec![glob("*.mlp.*.weight"), glob("*.norm.weight")],
            ..Default::default()
        };
        assert!(f.is_active());
        let s = sig("F16", &[4, 4]);
        assert!(f.matches("model.layers.0.mlp.down_proj.weight", Some(&s), Some(&s)));
        assert!(f.matches("model.norm.weight", Some(&s), None));
        assert!(!f.matches("model.embed_tokens.weight", Some(&s), Some(&s)));
    }

    #[test]
    fn filter_names_exact() {
        let f = TensorFilter {
            names_exact: Some(["a.w", "b.w"].iter().map(|s| s.to_string()).collect()),
            ..Default::default()
        };
        let s = sig("F16", &[2]);
        assert!(f.matches("a.w", Some(&s), Some(&s)));
        assert!(!f.matches("c.w", Some(&s), Some(&s)));
    }

    #[test]
    fn filter_dtype_glob_is_case_insensitive_and_either_side() {
        let f = TensorFilter {
            dtype: Some(glob("F*")),
            ..Default::default()
        };
        assert!(f.matches("w", Some(&sig("F16", &[2])), Some(&sig("F16", &[2]))));
        assert!(f.matches("w", Some(&sig("f32", &[2])), None)); // lowercase stored dtype
        assert!(!f.matches("w", Some(&sig("BF16", &[2])), Some(&sig("I8", &[2]))));
        // dtype changed F16 → BF16 still matches: the OLD side is F16.
        assert!(f.matches("w", Some(&sig("F16", &[2])), Some(&sig("BF16", &[2]))));
    }

    #[test]
    fn filter_shape_glob_star_one_dim_starstar_any() {
        // `*` matches exactly one dimension (of any size).
        let one = TensorFilter {
            shape: Some(glob("768/*")),
            ..Default::default()
        };
        assert!(one.matches("w", Some(&sig("F16", &[768, 2048])), None));
        assert!(!one.matches("w", Some(&sig("F16", &[768, 2048, 4])), None)); // rank 3
        assert!(!one.matches("w", Some(&sig("F16", &[768])), None)); // rank 1

        // `**` matches any number of dimensions.
        let any = TensorFilter {
            shape: Some(glob("768/**")),
            ..Default::default()
        };
        assert!(any.matches("w", Some(&sig("F16", &[768, 2048])), None));
        assert!(any.matches("w", Some(&sig("F16", &[768, 2048, 4])), None));

        // Trailing dimension at any rank.
        let tail = TensorFilter {
            shape: Some(glob("**/2048")),
            ..Default::default()
        };
        assert!(tail.matches("w", Some(&sig("F16", &[768, 2048])), None));
        assert!(tail.matches("w", Some(&sig("F16", &[6, 3, 2048])), None));
        assert!(!tail.matches("w", Some(&sig("F16", &[2048, 6])), None));
    }

    #[test]
    fn filter_constraints_compose_with_and() {
        let f = TensorFilter {
            name_globs: vec![glob("*.down_proj.weight")],
            dtype: Some(glob("BF16")),
            ..Default::default()
        };
        let bf = sig("BF16", &[2048, 768]);
        let f16 = sig("F16", &[2048, 768]);
        assert!(f.matches("model.layers.0.mlp.down_proj.weight", Some(&bf), Some(&bf)));
        assert!(!f.matches(
            "model.layers.0.mlp.down_proj.weight",
            Some(&f16),
            Some(&f16)
        )); // dtype fails
        assert!(!f.matches("model.layers.0.mlp.gate_proj.weight", Some(&bf), Some(&bf))); // name fails
    }

    #[test]
    fn filter_apply_restricts_both_sides_and_keeps_add_remove() {
        let mut old = summary(
            &[
                ("keep.down_proj.weight", sig("BF16", &[8, 4])),
                ("skip.gate_proj.weight", sig("BF16", &[8, 4])),
                ("only_old.down_proj.weight", sig("BF16", &[8, 4])),
            ],
            &[],
        );
        let mut new = summary(
            &[
                ("keep.down_proj.weight", sig("BF16", &[8, 4])),
                ("skip.gate_proj.weight", sig("BF16", &[8, 4])),
                ("only_new.down_proj.weight", sig("BF16", &[8, 4])),
            ],
            &[],
        );
        let f = TensorFilter {
            name_globs: vec![glob("*.down_proj.weight")],
            ..Default::default()
        };
        f.apply(&mut old, &mut new);
        assert_eq!(
            old.tensors.keys().cloned().collect::<Vec<_>>(),
            vec!["keep.down_proj.weight", "only_old.down_proj.weight"]
        );
        assert_eq!(
            new.tensors.keys().cloned().collect::<Vec<_>>(),
            vec!["keep.down_proj.weight", "only_new.down_proj.weight"]
        );
        // The diff over the filtered subset: one unchanged, one removed, one added.
        let r = compare(&old, &new);
        assert_eq!(r.tensors_unchanged, 1);
        assert_eq!(r.tensors_removed.len(), 1);
        assert_eq!(r.tensors_added.len(), 1);
    }

    #[test]
    fn name_schema_collapses_layers_and_experts() {
        let mut names = Vec::new();
        for l in 0..3 {
            for e in 0..2 {
                names.push(format!("model.layers.{l}.experts.{e}.down_proj.weight"));
                names.push(format!("model.layers.{l}.experts.{e}.gate_proj.weight"));
            }
        }
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let schema = name_schema(&refs);
        // Two templates (down / gate), each covering 3 layers × 2 experts = 6.
        assert_eq!(schema.len(), 2);
        assert!(
            schema.contains(&(
                "model.layers.{0-2}.experts.{0-1}.down_proj.weight".to_string(),
                6
            )),
            "{schema:?}"
        );
        assert!(
            schema.contains(&(
                "model.layers.{0-2}.experts.{0-1}.gate_proj.weight".to_string(),
                6
            )),
            "{schema:?}"
        );
    }

    #[test]
    fn filter_inactive_is_noop() {
        let f = TensorFilter::default();
        assert!(!f.is_active());
        assert_eq!(f.describe(), None);
        let mut a = summary(&[("w", sig("F16", &[2]))], &[]);
        let mut b = summary(&[("w", sig("F16", &[2]))], &[]);
        f.apply(&mut a, &mut b);
        assert_eq!(a.tensors.len(), 1); // untouched
    }
}
