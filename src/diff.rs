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

/// A tensor's compared identity: dtype + shape. Two tensors with the same name
/// are "changed" when these differ (data bytes are not part of the comparison).
#[derive(Clone, PartialEq, Eq)]
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
#[derive(Clone, PartialEq, Eq)]
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
    pub fn from_loaded(tensors: Vec<TensorInfo>, metadata: Vec<MetadataInfo>) -> Self {
        let mut t = BTreeMap::new();
        for ti in tensors {
            t.insert(
                ti.name,
                TensorSig {
                    dtype: ti.dtype,
                    shape: ti.shape,
                },
            );
        }
        let mut m = BTreeMap::new();
        for mi in metadata {
            m.insert(
                mi.name,
                MetaVal {
                    value: mi.value,
                    value_type: mi.value_type,
                },
            );
        }
        Self {
            tensors: t,
            metadata: m,
        }
    }
}

/// A tensor present in both checkpoints whose dtype and/or shape differ.
pub struct TensorChange {
    pub name: String,
    pub old: TensorSig,
    pub new: TensorSig,
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
    /// then the same for metadata. Pipe-friendly (no colour, sorted by name).
    pub fn render(&self, old_label: &str, new_label: &str) -> String {
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
        for (name, sig) in &self.tensors_removed {
            let _ = writeln!(s, "  - {name}  [{}]", sig.render());
        }
        for (name, sig) in &self.tensors_added {
            let _ = writeln!(s, "  + {name}  [{}]", sig.render());
        }
        for c in &self.tensors_changed {
            let _ = writeln!(
                s,
                "  ~ {}  [{}] → [{}]",
                c.name,
                c.old.render(),
                c.new.render()
            );
        }

        let _ = writeln!(
            s,
            "\nmetadata: -{} +{} ~{} ({} unchanged)",
            self.meta_removed.len(),
            self.meta_added.len(),
            self.meta_changed.len(),
            self.meta_unchanged,
        );
        for (name, v) in &self.meta_removed {
            let _ = writeln!(s, "  - {name} = {}", quote_trunc(&v.value));
        }
        for (name, v) in &self.meta_added {
            let _ = writeln!(s, "  + {name} = {}", quote_trunc(&v.value));
        }
        for c in &self.meta_changed {
            if c.old.value != c.new.value {
                let _ = writeln!(
                    s,
                    "  ~ {} = {} → {}",
                    c.name,
                    quote_trunc(&c.old.value),
                    quote_trunc(&c.new.value)
                );
            } else {
                // Same value, different declared type.
                let _ = writeln!(
                    s,
                    "  ~ {} (type {} → {})",
                    c.name, c.old.value_type, c.new.value_type
                );
            }
        }
        s
    }
}

/// Compare two checkpoint summaries (old → new) into a [`DiffReport`].
pub fn compare(old: &CheckpointSummary, new: &CheckpointSummary) -> DiffReport {
    let mut tensors_removed = Vec::new();
    let mut tensors_changed = Vec::new();
    let mut tensors_unchanged = 0usize;
    for (name, osig) in &old.tensors {
        match new.tensors.get(name) {
            None => tensors_removed.push((name.clone(), osig.clone())),
            Some(nsig) if nsig == osig => tensors_unchanged += 1,
            Some(nsig) => tensors_changed.push(TensorChange {
                name: name.clone(),
                old: osig.clone(),
                new: nsig.clone(),
            }),
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
) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "--- {old_label}");
    let _ = writeln!(s, "+++ {new_label}");
    let _ = writeln!(s);
    match (old, new) {
        (Some(o), None) => {
            let _ = writeln!(s, "  - {name}  [{}]  (only in old)", o.render());
        }
        (None, Some(n)) => {
            let _ = writeln!(s, "  + {name}  [{}]  (only in new)", n.render());
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
            let _ = writeln!(s, "  ~ {name}  [{}] → [{}]", o.render(), n.render());
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
        let out = r.render("old", "new");
        assert!(out.contains("~ typed (type int → float)"), "{out}");
        assert!(out.contains("~ v = \"0.4\" → \"0.5\""), "{out}");
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
        );
        assert!(ident.contains("= w  [F32 (4)]  (identical)"), "{ident}");

        let added = render_tensor_focus("o", "n", "w", None, Some(&a), None);
        assert!(added.contains("+ w  [F32 (4)]  (only in new)"), "{added}");

        let b = sig("F32", &[8]);
        let reshape = render_tensor_focus(
            "o",
            "n",
            "w",
            Some(&a),
            Some(&b),
            Some(&ValueCmp::Skipped("shapes differ".to_string())),
        );
        assert!(reshape.contains("~ w  [F32 (4)] → [F32 (8)]"), "{reshape}");
        assert!(
            reshape.contains("values: not compared (shapes differ)"),
            "{reshape}"
        );
    }
}
