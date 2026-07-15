//! `check` subcommand: run health checks on a checkpoint and report findings.
//!
//! Two tiers, mirroring the metadata-only / local split the rest of the tool
//! lives by:
//!   * **structural** checks read only headers/names, so they are cheap and work
//!     over the metadata-only remote path (`--ssh-read` / `s3://`);
//!   * **value** checks (`--values`) scan tensor data and so need the bytes locally,
//!     exactly like the heatmap / stats views.
//!
//! Exit codes mirror `diff`: `0` clean, `1` findings, `2` couldn't run (the
//! caller maps a load failure to `2`). Warnings only fail under `--strict`.

use crate::filter::NameFilter;
use crate::health::HealthReport;
use crate::progress::LoadProgress;
use crate::sample::{PackingSchema, ViewDtype, parse_packing_schemas, tensor_stats};
use crate::tree::{Layout, MetadataInfo, TensorInfo};
use crate::utils::{format_parameters, format_shape, format_size};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};

/// `check --format`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum Format {
    /// Human-readable report (default).
    #[default]
    Text,
    /// A structured JSON report: per-check status, findings, and the overall
    /// exit code — for scripts / agents / CI.
    Json,
    /// SARIF 2.1.0 — for GitHub code scanning / static-analysis tooling.
    Sarif,
}

/// How serious a finding is. `Error` always fails the run; `Warning` fails only
/// under `--strict`. Ordered so `Error > Warning` for sorting/severity.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Warning,
    Error,
}

/// A single problem a check turned up.
pub struct Finding {
    pub severity: Severity,
    /// The tensor / file the finding concerns, when it's about one thing.
    pub subject: Option<String>,
    pub message: String,
}

impl Finding {
    fn error(subject: Option<String>, message: String) -> Self {
        Finding {
            severity: Severity::Error,
            subject,
            message,
        }
    }
    fn warning(subject: Option<String>, message: String) -> Self {
        Finding {
            severity: Severity::Warning,
            subject,
            message,
        }
    }
}

/// The outcome of one named check.
pub struct CheckResult {
    /// Stable machine id, e.g. `byte_ranges` — surfaced by the upcoming
    /// `--format json` output.
    #[allow(dead_code)]
    pub id: &'static str,
    /// Human title for the text report.
    pub title: &'static str,
    /// A one-line "what passing means" note, shown when the check passes.
    pub note: &'static str,
    /// `false` when the check doesn't apply to this checkpoint (e.g. byte-range
    /// integrity on a GGUF file) — rendered as `n/a`, never as a pass.
    pub applicable: bool,
    pub findings: Vec<Finding>,
    /// Wall-clock time the check took, shown beside it — set only for the value
    /// scan (the one check slow enough to be worth timing).
    pub elapsed: Option<std::time::Duration>,
    /// A dynamic one-line summary shown *in place of* `note` when the check
    /// passes — lets a check report what it actually verified (e.g. the config
    /// check's "48 layers · 128 experts/layer"). `None` falls back to `note`.
    pub summary: Option<String>,
}

impl CheckResult {
    fn na(id: &'static str, title: &'static str, note: &'static str) -> Self {
        CheckResult {
            id,
            title,
            note,
            applicable: false,
            findings: Vec::new(),
            elapsed: None,
            summary: None,
        }
    }
    fn done(
        id: &'static str,
        title: &'static str,
        note: &'static str,
        findings: Vec<Finding>,
    ) -> Self {
        CheckResult {
            id,
            title,
            note,
            applicable: true,
            findings,
            elapsed: None,
            summary: None,
        }
    }
    pub fn errors(&self) -> usize {
        self.findings
            .iter()
            .filter(|f| f.severity == Severity::Error)
            .count()
    }
    pub fn warnings(&self) -> usize {
        self.findings
            .iter()
            .filter(|f| f.severity == Severity::Warning)
            .count()
    }
}

/// The full report for one checkpoint.
pub struct CheckReport {
    pub label: String,
    pub n_files: usize,
    pub n_tensors: usize,
    pub params: usize,
    /// Whether the value tier ran (drives the "value scan skipped" note).
    pub values: bool,
    pub results: Vec<CheckResult>,
}

impl CheckReport {
    pub fn errors(&self) -> usize {
        self.results.iter().map(CheckResult::errors).sum()
    }
    pub fn warnings(&self) -> usize {
        self.results.iter().map(CheckResult::warnings).sum()
    }
    /// `diff`-style: `1` when the checkpoint is unhealthy, else `0`. Errors
    /// always count; warnings only when `strict`.
    pub fn exit_code(&self, strict: bool) -> i32 {
        if self.errors() > 0 || (strict && self.warnings() > 0) {
            1
        } else {
            0
        }
    }
}

/// Whether a check passed, warned, failed, or didn't apply.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Pass,
    Warn,
    Fail,
    Na,
}

impl Status {
    fn as_str(self) -> &'static str {
        match self {
            Status::Pass => "pass",
            Status::Warn => "warn",
            Status::Fail => "fail",
            Status::Na => "na",
        }
    }
}

impl CheckResult {
    pub fn status(&self) -> Status {
        if !self.applicable {
            Status::Na
        } else if self.errors() > 0 {
            Status::Fail
        } else if self.warnings() > 0 {
            Status::Warn
        } else {
            Status::Pass
        }
    }
}

// ---- text report ----------------------------------------------------------

const GREEN: &str = "\x1b[32m";
const RED: &str = "\x1b[31m";
const YELLOW: &str = "\x1b[33m";
const DIM: &str = "\x1b[2m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

fn paint(s: &str, color: bool, code: &str) -> String {
    if color {
        format!("{code}{s}{RESET}")
    } else {
        s.to_string()
    }
}

impl CheckReport {
    /// Render the human-readable report. `color` gates ANSI styling.
    pub fn render(&self, color: bool) -> String {
        use std::fmt::Write;
        let mut s = String::new();

        let _ = writeln!(
            s,
            "{} {}",
            paint("checkpoint-explorer check:", color, BOLD),
            self.label
        );
        let _ = writeln!(
            s,
            "  {}",
            paint(
                &format!(
                    "{} file(s) · {} tensors · {} params",
                    self.n_files,
                    self.n_tensors,
                    format_parameters(self.params)
                ),
                color,
                DIM,
            ),
        );

        let width = self
            .results
            .iter()
            .map(|r| r.title.len())
            .max()
            .unwrap_or(0);
        for r in &self.results {
            let (mark, mcolor) = match r.status() {
                Status::Pass => ("✓", GREEN),
                Status::Warn => ("⚠", YELLOW),
                Status::Fail => ("✗", RED),
                Status::Na => ("⊘", DIM),
            };
            let title = format!("{:<width$}", r.title);
            let trailer = match r.status() {
                Status::Pass => {
                    let text = r.summary.as_deref().unwrap_or(r.note);
                    paint(&format!("— {text}"), color, DIM)
                }
                Status::Na => paint("— n/a for this checkpoint", color, DIM),
                _ => paint(
                    &format!("({})", count_phrase(r.errors(), r.warnings())),
                    color,
                    DIM,
                ),
            };
            // The value scan carries its wall-clock time; show it dim at the end.
            let elapsed = r
                .elapsed
                .map(|d| paint(&format!("  ({})", fmt_elapsed(d)), color, DIM))
                .unwrap_or_default();
            let _ = writeln!(
                s,
                "  {} {}  {}{}",
                paint(mark, color, mcolor),
                title,
                trailer,
                elapsed
            );

            for f in &r.findings {
                let (fmark, fcolor) = match f.severity {
                    Severity::Error => ("✗", RED),
                    Severity::Warning => ("⚠", YELLOW),
                };
                let subject = match &f.subject {
                    Some(subj) => format!("{}  ", paint(subj, color, BOLD)),
                    None => String::new(),
                };
                let _ = writeln!(
                    s,
                    "      {} {}{}",
                    paint(fmark, color, fcolor),
                    subject,
                    f.message
                );
            }
        }

        if !self.values {
            let _ = writeln!(
                s,
                "  {} {:<width$}  {}",
                paint("·", color, DIM),
                "Value scan",
                paint("— skipped (pass --values to scan tensor data)", color, DIM),
            );
        }

        let (errors, warnings) = (self.errors(), self.warnings());
        let summary = if errors > 0 {
            paint(
                &format!("FAIL — {}", count_phrase(errors, warnings)),
                color,
                RED,
            )
        } else if warnings > 0 {
            paint(
                &format!("OK with warnings — {}", count_phrase(0, warnings)),
                color,
                YELLOW,
            )
        } else if self.values {
            paint("OK — no issues found", color, GREEN)
        } else {
            // The value tier didn't run, so only the metadata/structural checks
            // passed — don't imply the tensor data was verified.
            paint("OK — no metadata issues found", color, GREEN)
        };
        let _ = writeln!(s, "  {summary}");
        s
    }
}

impl CheckReport {
    /// Structured report for `--format json`. `strict` decides the top-level
    /// `healthy` / `exit_code` (whether warnings count).
    pub fn to_json(&self, strict: bool) -> serde_json::Value {
        use serde_json::json;
        let checks: Vec<serde_json::Value> = self
            .results
            .iter()
            .map(|r| {
                json!({
                    "id": r.id,
                    "title": r.title,
                    "status": r.status().as_str(),
                    "findings": r.findings.iter().map(|f| json!({
                        "severity": match f.severity {
                            Severity::Error => "error",
                            Severity::Warning => "warning",
                        },
                        "subject": f.subject,
                        "message": f.message,
                    })).collect::<Vec<_>>(),
                })
            })
            .collect();
        json!({
            "checkpoint": self.label,
            "summary": {
                "files": self.n_files,
                "tensors": self.n_tensors,
                "params": self.params,
                "errors": self.errors(),
                "warnings": self.warnings(),
            },
            "values": self.values,
            "checks": checks,
            "healthy": self.exit_code(strict) == 0,
            "exit_code": self.exit_code(strict),
        })
    }

    /// A SARIF 2.1.0 log for `--format sarif`: one `run` whose driver lists the
    /// checks as `rules` and whose `results` are the findings (errors/warnings),
    /// each located at its file (safetensors/HDF5 path) or, for a tensor, a
    /// logical location. Consumable by GitHub code scanning and other SARIF tools.
    pub fn to_sarif(&self) -> serde_json::Value {
        use serde_json::{Value, json};

        let rules: Vec<Value> = self
            .results
            .iter()
            .map(|r| {
                json!({
                    "id": r.id,
                    "name": r.title,
                    "shortDescription": { "text": r.note },
                })
            })
            .collect();

        let mut results: Vec<Value> = Vec::new();
        for r in &self.results {
            for f in &r.findings {
                let level = match f.severity {
                    Severity::Error => "error",
                    Severity::Warning => "warning",
                };
                let mut obj = serde_json::Map::new();
                obj.insert("ruleId".into(), json!(r.id));
                obj.insert("level".into(), json!(level));
                let text = match &f.subject {
                    Some(s) => format!("{s}: {}", f.message),
                    None => f.message.clone(),
                };
                obj.insert("message".into(), json!({ "text": text }));
                if let Some(subject) = &f.subject {
                    obj.insert("locations".into(), json!([sarif_location(subject)]));
                }
                results.push(Value::Object(obj));
            }
        }

        json!({
            "version": "2.1.0",
            "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
            "runs": [{
                "tool": { "driver": {
                    "name": "checkpoint-explorer",
                    "version": env!("CARGO_PKG_VERSION"),
                    "informationUri": "https://github.com/antont-cerebras/checkpoint-explorer",
                    "rules": rules,
                }},
                "results": results,
            }],
        })
    }
}

/// A SARIF location for a finding's subject: a `physicalLocation` when it names a
/// checkpoint file, else a `logicalLocation` (a tensor name).
fn sarif_location(subject: &str) -> serde_json::Value {
    use serde_json::json;
    const FILE_EXTS: [&str; 6] = [".safetensors", ".hdf5", ".h5", ".gguf", ".npy", ".npz"];
    if FILE_EXTS.iter().any(|e| subject.ends_with(e)) {
        json!({ "physicalLocation": { "artifactLocation": { "uri": subject } } })
    } else {
        json!({ "logicalLocations": [{ "fullyQualifiedName": subject, "kind": "member" }] })
    }
}

/// A scan duration for the report, in the progress bar's `12.3s` style.
pub(crate) fn fmt_elapsed(d: std::time::Duration) -> String {
    format!("{:.1}s", d.as_secs_f64())
}

/// "1 error, 2 warnings" (omitting a zero side, keeping at least one).
pub(crate) fn count_phrase(errors: usize, warnings: usize) -> String {
    let plural = |n: usize, word: &str| format!("{n} {word}{}", if n == 1 { "" } else { "s" });
    match (errors, warnings) {
        (0, w) => plural(w, "warning"),
        (e, 0) => plural(e, "error"),
        (e, w) => format!("{}, {}", plural(e, "error"), plural(w, "warning")),
    }
}

// ---- the checks -----------------------------------------------------------

/// Run every applicable check against an already-loaded checkpoint. Structural
/// checks always run; the value tier runs only when `values`.
#[allow(clippy::too_many_arguments)]
pub fn run(
    label: String,
    tensors: &[TensorInfo],
    metadata: &[MetadataInfo],
    files: &[std::path::PathBuf],
    health: &[HealthReport],
    config: Option<&crate::config::ModelConfig>,
    filter: &NameFilter,
    values: bool,
    jobs: usize,
) -> CheckReport {
    let params = tensors.iter().map(|t| t.num_elements).sum();
    let mut results = vec![
        check_byte_ranges(tensors),
        check_hdf5(tensors),
        check_layers(tensors),
        check_shapes_dtypes(tensors),
        check_config(tensors, config),
        check_files(tensors, files, health),
    ];
    if values {
        results.push(check_values(tensors, metadata, filter, jobs));
    }
    CheckReport {
        label,
        n_files: files.len(),
        n_tensors: tensors.len(),
        params,
        values,
        results,
    }
}

/// safetensors byte-layout integrity: every tensor's byte span matches its
/// declared dtype×shape, spans are contiguous from 0 with no gaps/overlaps, and
/// the file is neither truncated nor padded with trailing data. Only applies to
/// safetensors (GGUF/HDF5/NumPy locate data differently).
fn check_byte_ranges(tensors: &[TensorInfo]) -> CheckResult {
    const ID: &str = "byte_ranges";
    const TITLE: &str = "Byte-range integrity";
    const NOTE: &str = "safetensors spans contiguous, sized correctly, no truncation";

    // file -> its tensors (name, start, end, stored_size), safetensors only.
    let mut by_file: BTreeMap<&str, Vec<(&str, u64, u64, u64)>> = BTreeMap::new();
    for t in tensors {
        if !t.source_path.ends_with(".safetensors") {
            continue;
        }
        if let Layout::ByteRange { start, end } = t.layout {
            by_file.entry(&t.source_path).or_default().push((
                &t.name,
                start,
                end,
                t.size_bytes as u64,
            ));
        }
    }
    if by_file.is_empty() {
        return CheckResult::na(ID, TITLE, NOTE);
    }

    let mut findings = Vec::new();
    for (file, mut spans) in by_file {
        let short = file.rsplit('/').next().unwrap_or(file);
        spans.sort_by_key(|&(_, start, _, _)| start);

        // Per-tensor: the span must match the declared (stored) size.
        for &(name, start, end, size) in &spans {
            if end < start {
                findings.push(Finding::error(
                    Some(name.into()),
                    format!("{short}: inverted byte range [{start}, {end})"),
                ));
            } else if end - start != size {
                findings.push(Finding::error(
                    Some(name.into()),
                    format!(
                        "{short}: byte span {} ≠ declared size {} (header corrupt or dtype/shape wrong)",
                        format_size((end - start) as usize),
                        format_size(size as usize),
                    ),
                ));
            }
        }

        // Contiguity: safetensors packs tensors tightly from offset 0.
        let mut cursor = 0u64;
        for &(name, start, end, _) in &spans {
            if start > cursor {
                findings.push(Finding::warning(
                    Some(name.into()),
                    format!(
                        "{short}: {} unused gap before this tensor",
                        format_size((start - cursor) as usize)
                    ),
                ));
            } else if start < cursor {
                findings.push(Finding::error(
                    Some(name.into()),
                    format!("{short}: byte range overlaps the previous tensor"),
                ));
            }
            cursor = cursor.max(end);
        }

        // Truncation: the file must be exactly header + blob. `cursor` is the end
        // of the last tensor, relative to the data blob (starts after the 8-byte
        // length prefix + JSON header). Only checkable when the file is local —
        // over `--ssh-read` the span/contiguity checks above still run (they're
        // header-only), but there's no local file to stat.
        if !crate::remote::is_remote_source(file) {
            match safetensors_blob(file) {
                Ok((blob_start, file_size)) => {
                    let blob_len = file_size.saturating_sub(blob_start);
                    if cursor > blob_len {
                        findings.push(Finding::error(
                            Some(short.into()),
                            format!(
                                "file truncated: header declares {} of tensor data but only {} present (short by {})",
                                format_size(cursor as usize),
                                format_size(blob_len as usize),
                                format_size((cursor - blob_len) as usize),
                            ),
                        ));
                    } else if blob_len > cursor {
                        findings.push(Finding::warning(
                            Some(short.into()),
                            format!(
                                "{} of trailing data after the last tensor",
                                format_size((blob_len - cursor) as usize)
                            ),
                        ));
                    }
                }
                Err(e) => findings.push(Finding::error(
                    Some(short.into()),
                    format!("could not read header: {e}"),
                )),
            }
        }
    }
    findings.sort_by(sort_key);
    CheckResult::done(ID, TITLE, NOTE, findings)
}

/// Read a safetensors file's `(data_blob_start, file_size)`: 8-byte little-endian
/// header length, then the JSON header, then the data blob.
fn safetensors_blob(path: &str) -> Result<(u64, u64), String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let file_size = f.metadata().map_err(|e| e.to_string())?.len();
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok((8 + u64::from_le_bytes(buf), file_size))
}

/// HDF5 chunk/dtype integrity — the storage check for `.hdf5` (safetensors uses
/// `byte_ranges` instead). HDF5 stores each tensor as a chunked, filter-pipelined
/// dataset, so there are no flat byte offsets to validate; instead check that
/// each dataset's chunk shape is consistent with its tensor shape, the stored
/// chunk count doesn't exceed the chunk grid, and its datatype was recognized.
/// (The filter/decompression pipeline itself is exercised by `--values`, which
/// reads the data and surfaces any unreadable dataset as a scan error.)
fn check_hdf5(tensors: &[TensorInfo]) -> CheckResult {
    const ID: &str = "hdf5";
    const TITLE: &str = "HDF5 integrity";
    const NOTE: &str = "chunk shapes consistent with tensor shapes, datatypes recognized";

    let hdf5: Vec<&TensorInfo> = tensors
        .iter()
        .filter(|t| t.source_path.ends_with(".hdf5") || t.source_path.ends_with(".h5"))
        .collect();
    if hdf5.is_empty() {
        return CheckResult::na(ID, TITLE, NOTE);
    }

    let mut findings = Vec::new();
    for t in hdf5 {
        if t.dtype == "?" {
            findings.push(Finding::warning(
                Some(t.name.clone()),
                "unrecognized HDF5 datatype (its type descriptor could not be parsed)".into(),
            ));
        }
        if let Layout::Chunked { chunk, num_chunks } = &t.layout {
            if chunk.contains(&0) {
                findings.push(Finding::error(
                    Some(t.name.clone()),
                    format!("zero-size chunk dimension {}", format_shape(chunk)),
                ));
            } else if !t.shape.is_empty() && chunk.len() != t.shape.len() {
                findings.push(Finding::error(
                    Some(t.name.clone()),
                    format!("chunk rank {} ≠ tensor rank {}", chunk.len(), t.shape.len()),
                ));
            } else if chunk.len() == t.shape.len() {
                // A chunked dataset holds at most one chunk per grid cell
                // (∏ ⌈dim/chunk⌉); more stored chunks than that means a corrupt
                // chunk index. Fewer is fine (unwritten cells read as fill).
                let grid: usize = t
                    .shape
                    .iter()
                    .zip(chunk)
                    .map(|(&d, &c)| d.div_ceil(c))
                    .product();
                if *num_chunks > grid {
                    findings.push(Finding::error(
                        Some(t.name.clone()),
                        format!(
                            "chunk index inconsistent: {num_chunks} stored chunks but the grid holds only {grid}"
                        ),
                    ));
                }
            }
        }
    }
    findings.sort_by(sort_key);
    CheckResult::done(ID, TITLE, NOTE, findings)
}

/// Split a tensor name at its first all-digit, dot-delimited segment — the
/// conventional layer index in `model.layers.<i>.mlp.…`. Returns
/// `(prefix, index, suffix)`, or `None` when there's no such segment.
pub(crate) fn split_layer_index(name: &str) -> Option<(String, usize, String)> {
    let parts: Vec<&str> = name.split('.').collect();
    let pos = parts
        .iter()
        .position(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))?;
    let idx: usize = parts[pos].parse().ok()?;
    Some((parts[..pos].join("."), idx, parts[pos + 1..].join(".")))
}

/// Layer completeness: for each repeated `<prefix>.<i>.<suffix>` stack, the
/// indices should be contiguous `0..=max`, and every layer should carry the same
/// set of sub-tensors. Catches dropped shards and partial checkpoints from names
/// alone.
fn check_layers(tensors: &[TensorInfo]) -> CheckResult {
    const ID: &str = "layers";
    const TITLE: &str = "Layer completeness";
    const NOTE: &str = "layer indices contiguous, tensor set uniform across layers";

    // prefix -> (index -> set of per-layer sub-tensor *roles*). The suffix has its
    // own numeric indices blanked (`role_key`) — chiefly the MoE expert index — so
    // a layer is compared at the role level: a dense layer among MoE layers reads
    // as missing one role (`mlp.experts.*.down_proj.weight`), not one finding per
    // expert (which buried the report under thousands of near-identical lines).
    let mut fam: BTreeMap<String, BTreeMap<usize, BTreeSet<String>>> = BTreeMap::new();
    for t in tensors {
        if let Some((prefix, idx, suffix)) = split_layer_index(&t.name) {
            fam.entry(prefix)
                .or_default()
                .entry(idx)
                .or_default()
                .insert(role_key(&suffix));
        }
    }

    let mut findings = Vec::new();
    let mut checked_any = false;
    for (prefix, idxmap) in &fam {
        // Two indices is the smallest thing that's a "stack" — enough to check
        // contiguity and cross-layer uniformity (so a 2-layer / truncated / debug
        // checkpoint is validated, not skipped). A lone index can't be compared.
        if idxmap.len() < 2 {
            continue;
        }
        checked_any = true;
        let max = *idxmap.keys().next_back().unwrap();

        // Index gaps.
        let missing: Vec<usize> = (0..=max).filter(|i| !idxmap.contains_key(i)).collect();
        if !missing.is_empty() {
            // The subject already names the stack — the message needn't repeat it.
            let subject = Some(format!("{prefix}.*"));
            let gaps = fmt_indices(&missing);
            // Show the indices that *are* present (run-collapsed), not the `0–max`
            // extent: `0–2` would read as if 1 were present too, contradicting the
            // gap it's reporting.
            let present: Vec<usize> = idxmap.keys().copied().collect();
            let have = fmt_indices(&present);
            // A gap in the transformer block stack (`…layers.<i>`) is a dropped
            // layer — a real error. A gap in some *other* indexed stack (e.g. an
            // `nn.Sequential` projector) is usually just a paramless module, which
            // stores no tensors — a soft warning, not a failure.
            if prefix.rsplit('.').next() == Some("layers") {
                findings.push(Finding::error(
                    subject,
                    format!("missing layer {gaps} (have {have})"),
                ));
            } else {
                findings.push(Finding::warning(
                    subject,
                    format!("index {gaps} absent (have {have}) — may be a paramless module"),
                ));
            }
        }

        // Per-layer uniformity, measured against the most common sub-tensor set.
        let mut set_counts: HashMap<&BTreeSet<String>, usize> = HashMap::new();
        for set in idxmap.values() {
            *set_counts.entry(set).or_default() += 1;
        }
        let Some((modal, _)) = set_counts.into_iter().max_by_key(|&(_, c)| c) else {
            continue;
        };
        // For each sub-tensor most layers have, list the layers missing it.
        for suffix in modal {
            let absent: Vec<usize> = idxmap
                .iter()
                .filter(|(_, set)| !set.contains(suffix))
                .map(|(&i, _)| i)
                .collect();
            if !absent.is_empty() {
                // The subject names the role; the message needn't repeat it (that
                // doubled the line length and truncated the report).
                findings.push(Finding::warning(
                    Some(format!("{prefix}.*.{suffix}")),
                    format!("missing from layer {}", fmt_indices(&absent)),
                ));
            }
        }
    }

    if !checked_any {
        return CheckResult::na(ID, TITLE, NOTE);
    }
    findings.sort_by(sort_key);
    CheckResult::done(ID, TITLE, NOTE, findings)
}

/// Shape/dtype sanity: no duplicate tensor names, no zero-element tensors, and a
/// heads-up when a handful of tensors use a different dtype than the rest.
fn check_shapes_dtypes(tensors: &[TensorInfo]) -> CheckResult {
    const ID: &str = "shapes_dtypes";
    const TITLE: &str = "Shape / dtype sanity";
    const NOTE: &str = "no duplicate names, no empty tensors, dtype uniform within each role";

    if tensors.is_empty() {
        return CheckResult::na(ID, TITLE, NOTE);
    }
    let mut findings = Vec::new();

    // Duplicate names (a correctly-sharded checkpoint names each tensor once).
    let mut name_counts: BTreeMap<&str, usize> = BTreeMap::new();
    for t in tensors {
        *name_counts.entry(&t.name).or_default() += 1;
    }
    for (name, count) in &name_counts {
        if *count > 1 {
            findings.push(Finding::error(
                Some((*name).into()),
                format!("duplicate tensor name ({count}×)"),
            ));
        }
    }

    // Zero-element tensors.
    for t in tensors {
        if t.num_elements == 0 || t.shape.contains(&0) {
            findings.push(Finding::warning(
                Some(t.name.clone()),
                format!("zero-element tensor, shape {}", format_shape(&t.shape)),
            ));
        }
    }

    // Dtype consistency is per *role*, not global: a checkpoint legitimately
    // mixes dtypes by role (weights BF16, quant scales F16, codebooks F32, …), so
    // a global "dominant dtype" flags all of those as anomalies. Instead, group
    // tensors by a role key — the name with numeric path segments (layer / expert
    // indices) blanked to `*` — and only flag a dtype that's inconsistent *within*
    // one role: a stray F32 among a role's F16s is the signature of a partial cast.
    //
    // role -> dtype -> (count, an example tensor of that dtype)
    let mut roles: BTreeMap<String, BTreeMap<&str, (usize, &str)>> = BTreeMap::new();
    for t in tensors {
        let entry = roles
            .entry(role_key(&t.name))
            .or_default()
            .entry(&t.dtype)
            .or_insert((0, t.name.as_str()));
        entry.0 += 1;
    }
    for (role, dtypes) in &roles {
        if dtypes.len() < 2 {
            continue; // one dtype (or a lone tensor) for this role — nothing to compare
        }
        let (&dom, &(dom_n, _)) = dtypes.iter().max_by_key(|&(_, &(c, _))| c).unwrap();
        for (&dt, &(n, example)) in dtypes {
            if dt == dom {
                continue;
            }
            let msg = if n == 1 {
                format!("stored as {dt}, but the other {dom_n} `{role}` tensors are {dom}")
            } else {
                format!("one of {n} `{role}` tensors stored as {dt}; the other {dom_n} are {dom}")
            };
            findings.push(Finding::warning(Some(example.into()), msg));
        }
    }

    findings.sort_by(sort_key);
    CheckResult::done(ID, TITLE, NOTE, findings)
}

/// A tensor's "role": its name with numeric path segments (layer / expert
/// indices) blanked to `*`, so every tensor that plays the same structural role
/// across layers/experts shares one key (e.g.
/// `model.layers.*.mlp.experts.*.down_proj.weight.qscale`).
fn role_key(name: &str) -> String {
    name.split('.')
        .map(|seg| {
            if !seg.is_empty() && seg.bytes().all(|b| b.is_ascii_digit()) {
                "*"
            } else {
                seg
            }
        })
        .collect::<Vec<_>>()
        .join(".")
}

/// The number of transformer blocks in the tensor tree: the size of the
/// `…layers.<i>` index stack (preferring a family whose prefix ends in `layers`,
/// else the largest index family). `None` when there's no such stack.
fn detected_layer_count(tensors: &[TensorInfo]) -> Option<usize> {
    let mut fam: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();
    for t in tensors {
        if let Some((prefix, idx, _)) = split_layer_index(&t.name) {
            fam.entry(prefix).or_default().insert(idx);
        }
    }
    let chosen = fam
        .iter()
        .find(|(prefix, _)| prefix.rsplit('.').next() == Some("layers"))
        .or_else(|| fam.iter().max_by_key(|(_, idxs)| idxs.len()))?;
    chosen.1.iter().next_back().map(|&m| m + 1)
}

/// The expert index in a MoE tensor name — the segment right after `experts`, as
/// in `…mlp.experts.<e>.down_proj.weight`. `None` when the name has no expert.
pub(crate) fn expert_index(name: &str) -> Option<usize> {
    let parts: Vec<&str> = name.split('.').collect();
    let pos = parts.iter().position(|&p| p == "experts")?;
    parts.get(pos + 1)?.parse().ok()
}

/// The number of experts (max expert index + 1) seen across the checkpoint.
fn detected_expert_count(tensors: &[TensorInfo]) -> Option<usize> {
    tensors
        .iter()
        .filter_map(|t| expert_index(&t.name))
        .max()
        .map(|m| m + 1)
}

/// Config ↔ tensor-tree consistency: cross-check `config.json` against the tensor
/// names/shapes — layer & expert counts, the tied/untied LM head, the embedding
/// shape, and QK-norm — catching a config that doesn't match the weights (or a
/// checkpoint assembled against the wrong config). Name-based, so it holds up on
/// quantized/packed weights; the shape check is a soft warning for the same reason.
fn check_config(
    tensors: &[TensorInfo],
    config: Option<&crate::config::ModelConfig>,
) -> CheckResult {
    const ID: &str = "config";
    const TITLE: &str = "Config consistency";
    const NOTE: &str = "tensor tree matches config.json";

    let Some(cfg) = config else {
        return CheckResult::na(ID, TITLE, NOTE);
    };
    let mut findings = Vec::new();
    // The facts we positively verified, joined into the pass-line summary.
    let mut facts: Vec<String> = Vec::new();

    // Layer count.
    if let Some(want) = cfg.num_hidden_layers {
        match detected_layer_count(tensors) {
            Some(got) if got as u64 == want => facts.push(format!("{want} layers")),
            Some(got) => findings.push(Finding::error(
                None,
                format!("config says {want} layers, but the tensor tree has {got}"),
            )),
            None => {}
        }
    }

    // Expert count (MoE).
    if let Some(want) = cfg.num_experts {
        match detected_expert_count(tensors) {
            Some(got) if got as u64 == want => facts.push(format!("{want} experts/layer")),
            Some(got) => findings.push(Finding::error(
                None,
                format!("config says {want} experts, but the tensor tree has {got}"),
            )),
            // Config declares experts but none are named `…experts.<n>…`.
            None if want > 0 => findings.push(Finding::error(
                None,
                format!("config says {want} experts, but no expert tensors were found"),
            )),
            None => {}
        }
    }

    // Tied vs untied LM head.
    if let Some(tied) = cfg.tie_word_embeddings {
        let has_head = tensors
            .iter()
            .any(|t| t.name == "lm_head.weight" || t.name.ends_with(".lm_head.weight"));
        match (tied, has_head) {
            (false, true) => facts.push("untied lm_head".into()),
            (true, false) => facts.push("tied lm_head".into()),
            (false, false) => findings.push(Finding::warning(
                Some("lm_head".into()),
                "config sets tie_word_embeddings=false, but no lm_head weight was found".into(),
            )),
            (true, true) => findings.push(Finding::warning(
                Some("lm_head".into()),
                "config sets tie_word_embeddings=true, but a separate lm_head weight is present"
                    .into(),
            )),
        }
    }

    // Embedding shape vs vocab_size × hidden_size (soft: quantized embeddings pack
    // to a different stored shape).
    if let (Some(vs), Some(hs)) = (cfg.vocab_size, cfg.hidden_size)
        && let Some(embed) = tensors.iter().find(|t| t.name.contains("embed_tokens"))
    {
        let dims: Vec<u64> = embed.shape.iter().map(|&d| d as u64).collect();
        if dims.contains(&vs) && dims.contains(&hs) {
            facts.push(format!("vocab {vs}"));
        } else {
            findings.push(Finding::warning(
                Some(embed.name.clone()),
                format!(
                    "shape {} doesn't match config vocab_size={vs} × hidden_size={hs} (or the embedding is packed)",
                    format_shape(&embed.shape)
                ),
            ));
        }
    }

    // QK-norm tensors present iff config enables it.
    if let Some(want) = cfg.use_qk_norm {
        let has = tensors
            .iter()
            .any(|t| t.name.contains("q_norm") || t.name.contains("k_norm"));
        match (want, has) {
            (true, true) => facts.push("qk-norm".into()),
            (true, false) => findings.push(Finding::warning(
                None,
                "config sets use_qk_norm=true, but no q_norm/k_norm tensors were found".into(),
            )),
            (false, true) => findings.push(Finding::warning(
                None,
                "q_norm/k_norm tensors are present, but config sets use_qk_norm=false".into(),
            )),
            (false, false) => {}
        }
    }

    // Nothing in the config was checkable against these tensors.
    if facts.is_empty() && findings.is_empty() {
        return CheckResult::na(ID, TITLE, NOTE);
    }
    findings.sort_by(sort_key);
    let mut result = CheckResult::done(ID, TITLE, NOTE, findings);
    if !facts.is_empty() {
        let arch = cfg
            .model_type
            .as_deref()
            .map(|m| format!("{m}: "))
            .unwrap_or_default();
        result.summary = Some(format!("{arch}{}", facts.join(" · ")));
    }
    result
}

/// File/index correspondence: fold in the `model.safetensors.index.json` health
/// report(s), and verify `model-XXXXX-of-NNNNN` shard numbering is complete. The
/// numbering check reads the shard filenames from both the on-disk file list and
/// the tensors' `source_path`s, so it works for a **remote** checkpoint too (a
/// missing shard shows up as a gap in the present shards' shared `-of-<N>`).
fn check_files(
    tensors: &[TensorInfo],
    files: &[std::path::PathBuf],
    health: &[HealthReport],
) -> CheckResult {
    const ID: &str = "files";
    const TITLE: &str = "Files & sharding";
    const NOTE: &str = "index matches files on disk, shard numbering complete";

    let mut findings = Vec::new();
    let mut applicable = false;

    // Index correspondence (only present when an index.json was found).
    for report in health {
        applicable = true;
        for f in &report.missing_files {
            findings.push(Finding::error(
                Some(f.clone()),
                "referenced by the index but missing on disk".into(),
            ));
        }
        for f in &report.extra_files {
            findings.push(Finding::warning(
                Some(f.clone()),
                "on disk but not referenced by the index".into(),
            ));
        }
        for t in &report.missing_tensors {
            findings.push(Finding::error(
                Some(t.clone()),
                "in the index but not in its file".into(),
            ));
        }
        for t in &report.extra_tensors {
            findings.push(Finding::warning(
                Some(t.clone()),
                "in a file but not in the index".into(),
            ));
        }
    }

    // Shard numbering: parse `…-<idx>-of-<total>.safetensors` from every shard
    // filename we know — the on-disk file list (local) and the tensors'
    // `source_path`s (which name the shards for a remote read too).
    let mut totals: BTreeSet<usize> = BTreeSet::new();
    let mut present: BTreeSet<usize> = BTreeSet::new();
    let names = files
        .iter()
        .filter_map(|f| f.file_name().and_then(|n| n.to_str()))
        .chain(
            tensors
                .iter()
                .filter_map(|t| t.source_path.rsplit(['/', ':']).next()),
        );
    for name in names {
        if let Some((idx, total)) = parse_shard_name(name) {
            present.insert(idx);
            totals.insert(total);
        }
    }
    if !present.is_empty() {
        applicable = true;
        if totals.len() > 1 {
            findings.push(Finding::error(
                None,
                format!("shard files disagree on the total count ({totals:?})"),
            ));
        } else if let Some(&total) = totals.iter().next() {
            let missing: Vec<usize> = (1..=total).filter(|i| !present.contains(i)).collect();
            if !missing.is_empty() {
                findings.push(Finding::error(
                    None,
                    format!(
                        "{} of {total} shards present; missing shard {}",
                        present.len(),
                        fmt_indices(&missing)
                    ),
                ));
            }
        }
    }

    if !applicable {
        return CheckResult::na(ID, TITLE, NOTE);
    }
    findings.sort_by(sort_key);
    CheckResult::done(ID, TITLE, NOTE, findings)
}

/// Parse a `…-<idx>-of-<total>.safetensors` shard filename.
fn parse_shard_name(name: &str) -> Option<(usize, usize)> {
    let stem = name.strip_suffix(".safetensors")?;
    let (rest, total) = stem.rsplit_once("-of-")?;
    let (_, idx) = rest.rsplit_once('-')?;
    Some((idx.parse().ok()?, total.parse().ok()?))
}

/// Value tier (`--values`), CLI entry: run [`scan_values`] behind a thin
/// determinate progress bar on stderr — the same style as the SSH-load bar, and
/// animated only on a terminal, so a piped or `--format json` run stays clean.
fn check_values(
    tensors: &[TensorInfo],
    metadata: &[MetadataInfo],
    filter: &NameFilter,
    jobs: usize,
) -> CheckResult {
    let bars = crate::progress::Bars::start(vec!["scanning tensor data".to_string()]);
    let fallback = LoadProgress::new();
    let progress = bars.progress(0);
    // Never cancelled from the CLI — cancellation is a TUI affordance.
    let cancel = AtomicBool::new(false);
    let result = scan_values(
        tensors,
        metadata,
        filter,
        jobs,
        progress.as_deref().unwrap_or(&fallback),
        &cancel,
    );
    bars.finish(0, true);
    bars.join();
    result
}

/// The value-tier scan itself, UI-agnostic: scan every (filtered) tensor's data
/// across `jobs` worker threads — flagging non-finite (NaN/±Inf) elements,
/// all-zero tensors, and constant tensors — reading bytes locally via the same
/// streaming/mmap path as the stats view. Reports completed-tensor count into
/// `progress`, and honours `cancel` (checked between tensors and, via
/// [`tensor_stats`], mid-tensor). Driven by the CLI ([`check_values`], with a
/// stderr bar) and the TUI (with a ratatui bar).
pub fn scan_values(
    tensors: &[TensorInfo],
    metadata: &[MetadataInfo],
    filter: &NameFilter,
    jobs: usize,
    progress: &LoadProgress,
    cancel: &AtomicBool,
) -> CheckResult {
    const ID: &str = "values";
    const TITLE: &str = "Value scan";
    const NOTE: &str = "no NaN/±Inf, no all-zero or constant tensors";

    let schemas = parse_packing_schemas(tensors, metadata);
    let targets: Vec<&TensorInfo> = tensors.iter().filter(|t| filter.matches(&t.name)).collect();
    if targets.is_empty() {
        return CheckResult::na(ID, TITLE, NOTE);
    }
    let n = targets.len();
    progress.set_total(n);
    let started = std::time::Instant::now();
    let pause = AtomicBool::new(false);

    // Run the whole scan inside one rayon pool of `jobs` threads, so the
    // across-tensor parallelism and the intra-tensor `par_chunks` inside
    // `tensor_stats` share it and compose. (Raw worker threads *around* rayon
    // oversubscribe the CPU — each blocks in its own fork/join over the global
    // pool — which is pathological with tens of thousands of small tensors.)
    let mut findings = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs.max(1))
        .build()
        .map(|pool| pool.install(|| scan_par(&targets, &schemas, &pause, cancel, progress)))
        .unwrap_or_else(|_| scan_par(&targets, &schemas, &pause, cancel, progress));
    findings.sort_by(sort_key);
    let mut result = CheckResult::done(ID, TITLE, NOTE, findings);
    result.elapsed = Some(started.elapsed());
    result
}

/// Scan the filtered `targets` in parallel (within the caller's rayon pool),
/// classifying each tensor's stats into findings. `progress` is bumped per tensor
/// and `cancel` short-circuits remaining tensors (and aborts mid-tensor via
/// [`tensor_stats`]).
fn scan_par(
    targets: &[&TensorInfo],
    schemas: &HashMap<String, PackingSchema>,
    pause: &AtomicBool,
    cancel: &AtomicBool,
    progress: &LoadProgress,
) -> Vec<Finding> {
    use rayon::prelude::*;
    targets
        .par_iter()
        .flat_map(|&t| {
            if cancel.load(Ordering::Relaxed) {
                return Vec::new();
            }
            let out = match tensor_stats(
                t,
                ViewDtype::Stored,
                schemas.get(&t.name),
                cancel,
                pause,
                None,
            ) {
                Ok(st) if st.nonfinite > 0 => vec![Finding::error(
                    Some(t.name.clone()),
                    format!("{} non-finite element(s) (NaN/±Inf)", st.nonfinite),
                )],
                Ok(st) if st.count > 0 && st.zeros == st.count => {
                    vec![Finding::warning(
                        Some(t.name.clone()),
                        "all elements are zero".into(),
                    )]
                }
                Ok(st) if st.count > 0 && st.min == st.max => vec![Finding::warning(
                    Some(t.name.clone()),
                    format!("constant value {}", st.min),
                )],
                Ok(_) => Vec::new(),
                // A scan aborted by `cancel` also lands here; the caller discards a
                // cancelled run's result, so it's harmless.
                Err(e) => vec![Finding::warning(
                    Some(t.name.clone()),
                    format!("could not scan: {e}"),
                )],
            };
            progress.advance();
            out
        })
        .collect()
}

// ---- helpers --------------------------------------------------------------

/// Deterministic finding order: errors first, then by subject, then message.
fn sort_key(a: &Finding, b: &Finding) -> std::cmp::Ordering {
    b.severity
        .cmp(&a.severity)
        .then_with(|| a.subject.cmp(&b.subject))
        .then_with(|| a.message.cmp(&b.message))
}

/// Format a sorted index list compactly: `[3, 4, 5, 9]` -> `3–5, 9`.
fn fmt_indices(idx: &[usize]) -> String {
    let mut out = String::new();
    let mut i = 0;
    while i < idx.len() {
        let start = idx[i];
        let mut end = start;
        while i + 1 < idx.len() && idx[i + 1] == end + 1 {
            i += 1;
            end = idx[i];
        }
        if !out.is_empty() {
            out.push_str(", ");
        }
        if start == end {
            out.push_str(&start.to_string());
        } else {
            out.push_str(&format!("{start}–{end}"));
        }
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::Storage;

    /// A minimal tensor with no data location — enough for the name/shape/dtype
    /// checks, which don't touch bytes.
    fn ti(name: &str, dtype: &str, shape: &[usize]) -> TensorInfo {
        let num_elements = shape.iter().product();
        TensorInfo {
            name: name.into(),
            dtype: dtype.into(),
            shape: shape.to_vec(),
            size_bytes: num_elements * 4,
            num_elements,
            storage: Storage::Unknown,
            source_path: "mem.safetensors".into(),
            layout: Layout::None,
        }
    }

    /// A minimal chunked HDF5 tensor.
    fn hdf5_ti(name: &str, shape: &[usize], chunk: &[usize], num_chunks: usize) -> TensorInfo {
        TensorInfo {
            layout: Layout::Chunked {
                chunk: chunk.to_vec(),
                num_chunks,
            },
            source_path: "model.hdf5".into(),
            ..ti(name, "F32", shape)
        }
    }

    #[test]
    fn fmt_indices_collapses_runs() {
        assert_eq!(fmt_indices(&[3, 4, 5, 9]), "3–5, 9");
        assert_eq!(fmt_indices(&[2]), "2");
        assert_eq!(fmt_indices(&[1, 3, 5]), "1, 3, 5");
    }

    #[test]
    fn split_layer_index_finds_the_first_int_segment() {
        assert_eq!(
            split_layer_index("model.layers.5.mlp.down_proj.weight"),
            Some(("model.layers".into(), 5, "mlp.down_proj.weight".into()))
        );
        assert_eq!(split_layer_index("lm_head.weight"), None);
    }

    #[test]
    fn parse_shard_name_reads_index_and_total() {
        assert_eq!(
            parse_shard_name("model-00002-of-00005.safetensors"),
            Some((2, 5))
        );
        assert_eq!(parse_shard_name("model.safetensors"), None);
    }

    #[test]
    fn layers_flags_gaps_and_nonuniform_sets() {
        // Layers 0,1,3 (2 missing), all with the same two sub-tensors.
        let mut tensors = Vec::new();
        for i in [0, 1, 3] {
            tensors.push(ti(&format!("model.layers.{i}.mlp.weight"), "F32", &[4]));
            tensors.push(ti(&format!("model.layers.{i}.attn.weight"), "F32", &[4]));
        }
        let r = check_byte_ranges(&tensors); // n/a (Layout::None) — sanity that it doesn't panic
        assert!(!r.applicable);

        let r = check_layers(&tensors);
        assert!(r.applicable);
        assert_eq!(r.errors(), 1); // the missing layer 2
        assert!(
            r.findings
                .iter()
                .any(|f| f.message.contains("missing layer 2"))
        );
    }

    #[test]
    fn layers_flags_a_layer_missing_a_subtensor() {
        // Layers 0,1,2; layer 2 is missing `attn.weight` that the others have.
        let mut tensors = Vec::new();
        for i in [0, 1, 2] {
            tensors.push(ti(&format!("model.layers.{i}.mlp.weight"), "F32", &[4]));
            if i != 2 {
                tensors.push(ti(&format!("model.layers.{i}.attn.weight"), "F32", &[4]));
            }
        }
        let r = check_layers(&tensors);
        assert_eq!(r.errors(), 0);
        assert_eq!(r.warnings(), 1);
        assert!(r.findings.iter().any(|f| f.subject.as_deref()
            == Some("model.layers.*.attn.weight")
            && f.message.contains("missing from layer 2")));
    }

    #[test]
    fn shapes_dtypes_flags_dupes_zeros_and_role_dtype_outlier() {
        let mut tensors = vec![
            ti("a", "F16", &[2, 2]),
            ti("a", "F16", &[2, 2]),  // duplicate name
            ti("empty", "F16", &[0]), // zero-element
        ];
        // A per-layer role that's BF16 except one stray F32 (a partial-cast smell).
        tensors.push(ti("model.layers.0.mlp.weight", "BF16", &[2]));
        tensors.push(ti("model.layers.1.mlp.weight", "BF16", &[2]));
        tensors.push(ti("model.layers.2.mlp.weight", "F32", &[2]));
        // A different role, uniformly F16 across layers — must NOT warn, even
        // though its dtype differs from the mlp.weight role's.
        tensors.push(ti("model.layers.0.attn.qscale", "F16", &[2]));
        tensors.push(ti("model.layers.1.attn.qscale", "F16", &[2]));

        let r = check_shapes_dtypes(&tensors);
        assert!(r.findings.iter().any(|f| f.message.contains("duplicate")));
        assert!(
            r.findings
                .iter()
                .any(|f| f.message.contains("zero-element"))
        );

        // Exactly one dtype finding: the stray F32, anchored on layer 2.
        let dtype: Vec<_> = r
            .findings
            .iter()
            .filter(|f| f.message.contains("stored as"))
            .collect();
        assert_eq!(dtype.len(), 1);
        assert!(dtype[0].message.contains("F32"));
        assert_eq!(
            dtype[0].subject.as_deref(),
            Some("model.layers.2.mlp.weight")
        );
        // The uniform qscale role is left alone.
        assert!(!r.findings.iter().any(|f| f.message.contains("qscale")));
    }

    #[test]
    fn role_key_blanks_numeric_segments() {
        assert_eq!(
            role_key("model.layers.0.mlp.experts.7.down_proj.weight.qscale"),
            "model.layers.*.mlp.experts.*.down_proj.weight.qscale"
        );
        assert_eq!(role_key("lm_head.weight"), "lm_head.weight");
    }

    #[test]
    fn shard_numbering_checked_from_remote_source_paths() {
        // A remote read has no local files or index health, but the tensors carry
        // their shards' scp-form paths — so a missing shard (here shard 2 of 3,
        // whose tensors never loaded) is still caught via the `-of-00003` naming.
        let mut tensors = Vec::new();
        for idx in [1usize, 3] {
            let mut t = ti(&format!("model.layers.{idx}.w"), "F32", &[4]);
            t.source_path = format!("host:/ckpt/model-{idx:05}-of-00003.safetensors");
            tensors.push(t);
        }
        let res = check_files(&tensors, &[], &[]);
        assert!(res.applicable, "shard numbering applies from source paths");
        assert!(res.status() == Status::Fail, "a missing shard should fail");
        let msgs: Vec<&str> = res.findings.iter().map(|f| f.message.as_str()).collect();
        assert!(
            msgs.iter().any(|m| m.contains("missing shard 2")),
            "expected a missing-shard finding, got {msgs:?}"
        );
    }

    #[test]
    fn index_mismatch_from_a_health_report_fails() {
        // A remote index/file mismatch arrives as a HealthReport (built by the
        // remote read); Files & sharding folds it in and fails, just as for local.
        let report = crate::health::HealthReport {
            index_path: "host:/ckpt/model.safetensors.index.json".into(),
            missing_files: vec!["model-00000-of-00014.safetensors".into()],
            extra_files: vec!["model-00001-of-00073.safetensors".into()],
            missing_tensors: Vec::new(),
            extra_tensors: Vec::new(),
        };
        let res = check_files(&[], &[], std::slice::from_ref(&report));
        assert!(
            res.status() == Status::Fail,
            "a missing referenced file fails"
        );
        let msgs: Vec<&str> = res.findings.iter().map(|f| f.message.as_str()).collect();
        assert!(
            msgs.iter().any(|m| m.contains("missing on disk")),
            "expected a missing-file finding, got {msgs:?}"
        );
        assert!(
            msgs.iter()
                .any(|m| m.contains("not referenced by the index")),
            "expected an extra-file finding, got {msgs:?}"
        );
    }

    #[test]
    fn layers_checks_a_two_layer_stack() {
        // A 2-layer checkpoint (below the old ≥3 cutoff) is now validated.
        let mut tensors = Vec::new();
        for i in [0, 1] {
            tensors.push(ti(&format!("model.layers.{i}.mlp.weight"), "F32", &[4]));
            tensors.push(ti(&format!("model.layers.{i}.attn.weight"), "F32", &[4]));
        }
        let r = check_layers(&tensors);
        assert!(r.applicable); // no longer n/a
        assert_eq!(r.findings.len(), 0); // both layers present and uniform
    }

    #[test]
    fn layers_collapse_expert_indices_in_moe() {
        // A dense layer 0 among MoE layers 1–2 (4 experts × 2 projs each). The
        // uniformity check collapses expert indices to a role, so layer 0 reads as
        // missing 2 roles (down_proj, gate_proj) — not 8 individual expert tensors,
        // which used to bury the report under thousands of near-identical lines.
        let mut tensors = Vec::new();
        for l in [0, 1, 2] {
            tensors.push(ti(
                &format!("model.layers.{l}.self_attn.q_proj.weight"),
                "F16",
                &[4, 4],
            ));
        }
        for l in [1, 2] {
            for e in 0..4 {
                for proj in ["down_proj", "gate_proj"] {
                    tensors.push(ti(
                        &format!("model.layers.{l}.mlp.experts.{e}.{proj}.weight"),
                        "F16",
                        &[4, 4],
                    ));
                }
            }
        }
        let r = check_layers(&tensors);
        assert!(r.applicable);
        assert_eq!(r.errors(), 0, "layer indices 0–2 are contiguous");
        assert_eq!(
            r.warnings(),
            2,
            "one warning per expert *role*, not per expert"
        );
        assert!(
            r.findings.iter().all(|f| f
                .subject
                .as_deref()
                .is_some_and(|s| s.contains("mlp.experts.*"))),
            "findings name the collapsed role in the subject"
        );
    }

    #[test]
    fn layers_gap_in_a_non_block_stack_is_a_soft_warning() {
        // A projector `nn.Sequential` with a paramless module at index 1 (stores
        // no tensors): only 0 and 2 are present. That gap is a warning, not the
        // dropped-layer error a gap in the `…layers.<i>` block stack would be.
        let tensors = vec![
            ti("mm_projector.proj.0.weight", "F16", &[4, 4]),
            ti("mm_projector.proj.2.weight", "F16", &[4, 4]),
        ];
        let r = check_layers(&tensors);
        assert!(r.applicable);
        assert_eq!(r.errors(), 0, "a non-'layers' stack gap isn't a hard error");
        assert_eq!(r.warnings(), 1);
        let msg = &r.findings[0].message;
        assert!(msg.contains("paramless"), "{msg}");
        // Names the indices actually present, not a `0–2` span that would read as
        // if 1 (the absent one) were present too.
        assert!(msg.contains("have 0, 2") && !msg.contains("0–2"), "{msg}");
    }

    #[test]
    fn hdf5_flags_bad_chunks_and_dtype() {
        let mut tensors = vec![
            hdf5_ti("ok", &[4, 4], &[2, 2], 4), // grid 4, 4 chunks — healthy
        ];
        tensors.push(hdf5_ti("bad_rank", &[8], &[2, 2], 4)); // 2D chunk on a 1D tensor
        tensors.push(hdf5_ti("too_many", &[4, 4], &[2, 2], 9)); // 9 chunks > grid of 4
        let mut unk = hdf5_ti("unk", &[2], &[2], 1);
        unk.dtype = "?".into();
        tensors.push(unk);

        let r = check_hdf5(&tensors);
        assert!(r.applicable);
        assert_eq!(r.errors(), 2, "bad_rank + too_many");
        assert_eq!(r.warnings(), 1, "the '?' dtype");
        assert!(
            r.findings
                .iter()
                .any(|f| f.message.contains("chunk index inconsistent"))
        );
    }

    #[test]
    fn hdf5_check_is_na_for_safetensors() {
        assert!(!check_hdf5(&[ti("w", "F32", &[2])]).applicable);
    }

    #[test]
    fn exit_code_honors_strict() {
        let warn_only = CheckReport {
            label: "x".into(),
            n_files: 1,
            n_tensors: 0,
            params: 0,
            values: false,
            results: vec![CheckResult::done(
                "t",
                "T",
                "n",
                vec![Finding::warning(None, "w".into())],
            )],
        };
        assert_eq!(warn_only.exit_code(false), 0);
        assert_eq!(warn_only.exit_code(true), 1);

        let with_error = CheckReport {
            results: vec![CheckResult::done(
                "t",
                "T",
                "n",
                vec![Finding::error(None, "e".into())],
            )],
            ..warn_only
        };
        assert_eq!(with_error.exit_code(false), 1);
        assert_eq!(with_error.exit_code(true), 1);
    }

    /// A 2-layer, 2-expert MoE stack with an untied head and matching embedding.
    fn moe_tensors() -> Vec<TensorInfo> {
        let mut tensors = Vec::new();
        for l in [0, 1] {
            for e in [0, 1] {
                tensors.push(ti(
                    &format!("model.layers.{l}.mlp.experts.{e}.down_proj.weight"),
                    "F16",
                    &[2, 2],
                ));
            }
        }
        tensors.push(ti("lm_head.weight", "F16", &[10, 4]));
        tensors.push(ti("model.embed_tokens.weight", "F16", &[10, 4]));
        tensors
    }

    #[test]
    fn config_passes_and_summarizes_a_match() {
        let cfg = crate::config::ModelConfig {
            model_type: Some("qwen3_moe".into()),
            num_hidden_layers: Some(2),
            num_experts: Some(2),
            vocab_size: Some(10),
            hidden_size: Some(4),
            tie_word_embeddings: Some(false),
            use_qk_norm: Some(false),
        };
        let r = check_config(&moe_tensors(), Some(&cfg));
        assert!(r.applicable);
        assert_eq!(r.errors(), 0);
        assert_eq!(r.warnings(), 0);
        let summary = r.summary.expect("a passing config check summarizes facts");
        assert!(summary.contains("2 layers"), "{summary}");
        assert!(summary.contains("2 experts/layer"), "{summary}");
        assert!(summary.contains("untied lm_head"), "{summary}");
        assert!(summary.contains("qwen3_moe"), "{summary}");
    }

    #[test]
    fn config_flags_layer_and_expert_mismatch() {
        // The tensors have 2 layers × 2 experts; the config claims 3 × 4.
        let cfg = crate::config::ModelConfig {
            num_hidden_layers: Some(3),
            num_experts: Some(4),
            ..Default::default()
        };
        let r = check_config(&moe_tensors(), Some(&cfg));
        assert_eq!(r.errors(), 2, "layer + expert count mismatch");
        assert!(r.findings.iter().any(|f| f.message.contains("3 layers")));
        assert!(r.findings.iter().any(|f| f.message.contains("4 experts")));
    }

    #[test]
    fn config_warns_on_tie_and_qk_norm_inconsistency() {
        // Config says weights are tied (no lm_head expected) and qk-norm is on, but
        // the tensors have a separate lm_head and no q_norm/k_norm.
        let cfg = crate::config::ModelConfig {
            tie_word_embeddings: Some(true),
            use_qk_norm: Some(true),
            ..Default::default()
        };
        let r = check_config(&moe_tensors(), Some(&cfg));
        assert_eq!(r.errors(), 0);
        assert_eq!(r.warnings(), 2, "tied-but-has-head + qk-norm-missing");
    }

    #[test]
    fn config_check_is_na_without_config() {
        assert!(!check_config(&[ti("w", "F16", &[2])], None).applicable);
    }
}
