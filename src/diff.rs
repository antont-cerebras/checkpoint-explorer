//! The `diff` subcommand: compare two checkpoints' *structure* and summarize the
//! differences. "Structure" means the tensors (by name, dtype, and shape) and the
//! metadata (by name, value, and value type) — **not** the tensor data/values,
//! which a structural diff never reads (so it stays fast even on multi-GB files).
//!
//! The comparison ([`compare`]) is a pure function over two [`CheckpointSummary`]s
//! and produces a [`DiffReport`]; rendering ([`DiffReport::render`]) and the
//! `diff`-style exit code ([`DiffReport::has_differences`]) are separate so the
//! logic is testable without any I/O.

use std::collections::BTreeMap;
use std::fmt::Write;

use crate::sample::ValueDiff;
use crate::tree::{MetadataInfo, TensorInfo};
use crate::utils::format_shape;

const RED: &str = "\x1b[31m";
const GREEN: &str = "\x1b[32m";
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
}

/// Wrap `text` in an ANSI colour `code` when `on`, else return it unchanged.
fn paint(text: &str, on: bool, code: &str) -> String {
    if on {
        format!("{code}{text}{RESET}")
    } else {
        text.to_string()
    }
}

/// Render a changed tensor's `old` and `new` signatures, colouring only the token
/// (dtype and/or shape) that actually differs — the old side red, the new green —
/// so the eye lands on what changed. No colour when `color` is off.
fn render_change(old: &TensorSig, new: &TensorSig, color: bool) -> (String, String) {
    let dtype_changed = old.dtype != new.dtype;
    let shape_changed = old.shape != new.shape;
    let one = |sig: &TensorSig, code: &str| {
        let dtype = paint(&sig.dtype, color && dtype_changed, code);
        let shape = paint(&format_shape(&sig.shape), color && shape_changed, code);
        format!("{dtype} {shape}")
    };
    (one(old, RED), one(new, GREEN))
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
}

impl CheckpointSummary {
    /// Reduce a freshly-loaded checkpoint to its comparable structure. A sharded
    /// checkpoint can list a name in more than one file; the last one wins (the
    /// same name+shape is expected across shards, so this only matters if they
    /// genuinely disagree, which a diff can't meaningfully represent anyway).
    pub fn from_loaded(tensors: &[TensorInfo], metadata: &[MetadataInfo]) -> Self {
        let mut t = BTreeMap::new();
        for ti in tensors {
            t.insert(ti.name.clone(), TensorSig::of(ti));
        }
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
        }
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
        let _ = writeln!(s, "--- {old_label}");
        let _ = writeln!(s, "+++ {new_label}");

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
                // Same dtype & shape — a values-only change (only seen with --values).
                let _ = writeln!(
                    s,
                    "  ~ {name}  [{}]  (values differ){suffix}",
                    g.old.render()
                );
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
                    let _ = writeln!(
                        s,
                        "  ~ {name} = {} → {}{suffix}",
                        paint(&quote_trunc(&old.value), opts.color, RED),
                        paint(&quote_trunc(&new.value), opts.color, GREEN),
                    );
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
            // silently showing only the tensors section.
            let _ = writeln!(s, "\nmetadata: not compared (--only-tensors)");
        }
        s
    }
}

/// Structural comparison of two checkpoint summaries (old → new). Tensor values
/// are not read; see [`compare_with_values`].
pub fn compare(old: &CheckpointSummary, new: &CheckpointSummary) -> DiffReport {
    compare_inner(old, new, |_| None)
}

/// Like [`compare`] but also compares element values: `value_fn(name)` returns the
/// value comparison for a tensor present in both checkpoints (`None` when not
/// comparable, e.g. mismatched shapes). A tensor counts as changed when its dtype
/// or shape differs *or* its values differ, so a values-only change surfaces.
pub fn compare_with_values(
    old: &CheckpointSummary,
    new: &CheckpointSummary,
    value_fn: impl Fn(&str) -> Option<ValueDiff>,
) -> DiffReport {
    compare_inner(old, new, value_fn)
}

fn compare_inner(
    old: &CheckpointSummary,
    new: &CheckpointSummary,
    value_fn: impl Fn(&str) -> Option<ValueDiff>,
) -> DiffReport {
    let mut tensors_removed = Vec::new();
    let mut tensors_changed = Vec::new();
    let mut tensors_unchanged = 0usize;
    for (name, osig) in &old.tensors {
        let Some(nsig) = new.tensors.get(name) else {
            tensors_removed.push((name.clone(), osig.clone()));
            continue;
        };
        let values = value_fn(name);
        let values_differ = values.is_some_and(|v| v.differing > 0);
        if nsig != osig || values_differ {
            tensors_changed.push(TensorChange {
                name: name.clone(),
                old: osig.clone(),
                new: nsig.clone(),
                values,
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
    let _ = writeln!(s, "--- {old_label}");
    let _ = writeln!(s, "+++ {new_label}");
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

#[cfg(test)]
mod tests {
    use super::*;

    fn sig(dtype: &str, shape: &[usize]) -> TensorSig {
        TensorSig {
            dtype: dtype.to_string(),
            shape: shape.to_vec(),
        }
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
        }
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
        let r = compare_with_values(&old, &new, |_| {
            Some(ValueDiff {
                elements: 4,
                differing: 2,
                max_abs: 7.0,
                mean_abs: 3.5,
                nonfinite_mismatch: 0,
            })
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
        };
        let per = ValueDiff {
            elements: 4,
            differing: 1,
            max_abs: 2.0,
            mean_abs: 0.5,
            nonfinite_mismatch: 0,
        };
        let r = compare_with_values(&mk(), &mk(), |_| Some(per));
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
        };
        let new = CheckpointSummary {
            tensors: mk("BF16").into_iter().collect(),
            metadata: Default::default(),
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
    };

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
}
