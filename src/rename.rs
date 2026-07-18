//! In-place tensor renaming for **local** safetensors checkpoints — the engine
//! behind `convert --map` and the TUI's rename action.
//!
//! Renaming is done *in place*: only a shard's header region is rewritten (the
//! 8-byte length `N` plus the `N` JSON bytes after it), and the new JSON is padded
//! back to exactly `N` with trailing spaces — which the safetensors format allows
//! (it pads headers with spaces for alignment) — so the tensor data that follows
//! never moves. A rename whose new (compact) header would exceed `N` is rejected
//! up front, because honouring it would mean shifting every byte of tensor data.
//! `model.safetensors.index.json`'s `weight_map` keys are renamed too, so a
//! sharded checkpoint stays internally consistent.
//!
//! The whole plan is validated before a single byte is written — every source
//! exists, no two names collide onto one, every rewritten header fits in place —
//! and the caller confirms, so [`apply`] only ever runs a rename already known to
//! be safe.

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Map, Value};

use crate::diff::NameMap;
use crate::safelayout;

/// The conventional index filename for a sharded safetensors checkpoint.
const INDEX_NAME: &str = "model.safetensors.index.json";

/// The local safetensors files (and optional index) a rename would touch.
#[derive(Debug)]
pub struct Target {
    /// Directory shown in messages (the input dir, or a lone file's parent).
    pub root: PathBuf,
    /// Every `*.safetensors` shard, sorted by path.
    pub shards: Vec<PathBuf>,
    /// `model.safetensors.index.json`, when present.
    pub index: Option<PathBuf>,
}

/// One shard whose header the rename will rewrite.
#[derive(Debug)]
pub struct ShardPlan {
    pub path: PathBuf,
    /// Header JSON length `N` (the region is `8 + N`); the rewrite pads back to it.
    pub header_n: u64,
    /// The renamed, compact header JSON (guaranteed `len() as u64 <= header_n`).
    pub new_json: Vec<u8>,
    /// `(old, new)` pairs changed in this shard, sorted — for the report.
    pub renames: Vec<(String, String)>,
}

/// A validated, ready-to-apply rename. Built by [`plan`]; nothing is written to
/// disk until [`apply`].
#[derive(Debug)]
pub struct Plan {
    pub target: Target,
    /// Only the shards that actually change (untouched shards are skipped).
    pub shards: Vec<ShardPlan>,
    /// The rewritten `index.json` (path + full new text) when one is present and
    /// its `weight_map` keys change.
    pub index: Option<(PathBuf, String)>,
    /// Every `(old, new)` rename across the checkpoint, sorted & deduped.
    pub renames: Vec<(String, String)>,
    /// Non-fatal notes (a dead rule, an inconsistent index) surfaced to the user.
    pub warnings: Vec<String>,
}

impl Plan {
    /// The number of tensors this rename will change.
    pub fn rename_count(&self) -> usize {
        self.renames.len()
    }

    /// The number of shard files whose header will be rewritten.
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Human-readable summary lines for the CLI prompt and the TUI confirmation,
    /// capped at `cap` rename rows so a huge rename doesn't scroll off the screen.
    pub fn summary_lines(&self, cap: usize) -> Vec<String> {
        let mut out = Vec::new();
        out.push(format!(
            "Rename {} tensor(s) across {} shard file(s) in {}:",
            self.rename_count(),
            self.shard_count(),
            self.target.root.display(),
        ));
        for (old, new) in self.renames.iter().take(cap) {
            out.push(format!("  {old}  →  {new}"));
        }
        if self.renames.len() > cap {
            out.push(format!("  … and {} more", self.renames.len() - cap));
        }
        if self.index.is_some() {
            out.push(format!("Also updating {INDEX_NAME}."));
        }
        for w in &self.warnings {
            out.push(format!("warning: {w}"));
        }
        out.push(
            "Headers are rewritten in place — tensor data is NOT moved. This cannot be undone."
                .to_string(),
        );
        out
    }
}

/// Resolve `path` (a local directory, or a single `.safetensors` file) into the
/// set of shards + index a rename would touch. Errors if `path` doesn't exist
/// locally, isn't safetensors, or is a lone shard of a larger sharded set (in
/// which case the caller should point at the directory so the whole set — and its
/// index — is renamed consistently).
pub fn discover(path: &Path) -> Result<Target> {
    let meta = fs::metadata(path).with_context(|| {
        format!(
            "{} does not exist locally (rename is local-only)",
            path.display()
        )
    })?;

    if meta.is_dir() {
        let mut shards = Vec::new();
        for entry in fs::read_dir(path).with_context(|| format!("reading {}", path.display()))? {
            let p = entry?.path();
            if p.is_file()
                && p.extension()
                    .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"))
            {
                shards.push(p);
            }
        }
        shards.sort();
        if shards.is_empty() {
            bail!("no .safetensors files in {}", path.display());
        }
        let index_path = path.join(INDEX_NAME);
        let index = index_path.is_file().then_some(index_path);
        return Ok(Target {
            root: path.to_path_buf(),
            shards,
            index,
        });
    }

    // A single file: it must be safetensors, and must not be one shard of a
    // sharded checkpoint (renaming it alone would desync the shared index).
    if !path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"))
    {
        bail!("not a .safetensors file: {}", path.display());
    }
    let root = path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    let index_path = root.join(INDEX_NAME);
    if index_path.is_file() {
        let shard_files = index_shard_files(&index_path)?;
        let this = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        let others = shard_files.len() - usize::from(shard_files.contains(this));
        if others > 0 {
            bail!(
                "{} is one shard of a sharded checkpoint ({} references {} other shard(s)); \
                 point at the directory {} to rename the whole set consistently",
                path.display(),
                INDEX_NAME,
                others,
                root.display(),
            );
        }
        // The index references only this file: rename it too.
        return Ok(Target {
            root,
            shards: vec![path.to_path_buf()],
            index: Some(index_path),
        });
    }
    Ok(Target {
        root,
        shards: vec![path.to_path_buf()],
        index: None,
    })
}

/// A checkpoint's shard headers read once and held in memory, so a rename rule can
/// be previewed live (as the user types) without re-reading the files. Built by
/// [`load`]; drives both [`Loaded::preview`] (cheap, per-keystroke) and
/// [`Loaded::plan`] (the validated, ready-to-apply rename).
pub struct Loaded {
    target: Target,
    headers: Vec<ShardHeader>,
    /// Every tensor name across the checkpoint (excluding `__metadata__`), in file
    /// order — the source-field autocomplete list.
    all_names: Vec<String>,
    /// Names defined in more than one shard (a pre-existing inconsistency).
    duplicated: Vec<String>,
}

struct ShardHeader {
    path: PathBuf,
    /// Header JSON length `N` (the writable region is `8 + N`).
    n: u64,
    obj: Map<String, Value>,
}

/// A changed shard as rebuilt by [`Loaded::rebuild`]: its index into
/// `Loaded::headers`, the new compact header JSON, and the `(old, new)` renames.
type ShardRebuild = (usize, Vec<u8>, Vec<(String, String)>);

/// One tensor a rule would rename, with whether it can be done in place.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreviewRow {
    pub old: String,
    pub new: String,
    pub status: RenameStatus,
}

/// Why a single rename can or can't be applied in place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenameStatus {
    /// Applies cleanly in place.
    Ok,
    /// Two tensors map onto this same target name (would lose data).
    Collision,
    /// This shard's renamed header no longer fits its fixed-size region — the new
    /// names are longer, so applying would require moving tensor data.
    WontFit,
    /// The target is empty or the reserved `__metadata__`.
    Invalid,
}

/// How one changed shard's rewritten header sizes up against its fixed region:
/// `needed` (the new compact header) vs `current` (the writable `N` bytes). It
/// fits when `needed <= current`; otherwise it's `needed - current` bytes too big.
#[derive(Debug, Clone)]
pub struct ShardFit {
    /// The shard's leaf name (shown in the preview).
    pub file: String,
    /// The shard's full path (to open its layout view when clicked).
    pub path: String,
    /// The writable header length `N` (bytes available in place).
    pub current: u64,
    /// The rewritten compact header's length.
    pub needed: u64,
    /// How many tensors in this shard the rule renames.
    pub tensors: usize,
}

impl ShardFit {
    pub fn fits(&self) -> bool {
        self.needed <= self.current
    }
    /// Bytes the new header is over the region (0 when it fits).
    pub fn over(&self) -> u64 {
        self.needed.saturating_sub(self.current)
    }
    /// Bytes to spare (0 when it doesn't fit).
    pub fn spare(&self) -> u64 {
        self.current.saturating_sub(self.needed)
    }
}

/// A live, no-I/O preview of what a rule would do to a [`Loaded`] checkpoint: the
/// affected tensors (each marked with its [`RenameStatus`]) plus non-fatal notes.
#[derive(Debug, Default)]
pub struct RenamePreview {
    /// Affected tensors, sorted by old name.
    pub rows: Vec<PreviewRow>,
    /// Non-fatal notes (a dead rule, an inconsistent/updated index).
    pub warnings: Vec<String>,
    /// Whether the index.json will be updated too.
    pub has_index: bool,
}

impl RenamePreview {
    /// Whether every affected row applies cleanly, so the rename can go ahead.
    pub fn applicable(&self) -> bool {
        !self.rows.is_empty() && self.rows.iter().all(|r| r.status == RenameStatus::Ok)
    }
}

/// Read a local checkpoint's shard headers (no tensor data) into a [`Loaded`],
/// ready for live previewing and applying a rename. See [`discover`] for what
/// `path` may be.
pub fn load(path: &Path) -> Result<Loaded> {
    let target = discover(path)?;
    // Read each shard header once, keeping the parsed object (preserve_order keeps
    // its entry order so a rewrite doesn't reshuffle the file).
    let mut headers = Vec::with_capacity(target.shards.len());
    for shard in &target.shards {
        let (n, json) =
            safelayout::read_header_full(shard).map_err(|e| anyhow!("{}: {e}", shard.display()))?;
        let value: Value = serde_json::from_slice(&json)
            .with_context(|| format!("{}: invalid safetensors header JSON", shard.display()))?;
        let obj = value
            .as_object()
            .cloned()
            .ok_or_else(|| anyhow!("{}: header is not a JSON object", shard.display()))?;
        headers.push(ShardHeader {
            path: shard.clone(),
            n,
            obj,
        });
    }

    let mut all_names: Vec<String> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut duplicated: BTreeSet<String> = BTreeSet::new();
    for h in &headers {
        for k in h.obj.keys() {
            if k == "__metadata__" {
                continue;
            }
            if !seen.insert(k.clone()) {
                duplicated.insert(k.clone());
            }
            all_names.push(k.clone());
        }
    }

    Ok(Loaded {
        target,
        headers,
        all_names,
        duplicated: duplicated.into_iter().collect(),
    })
}

impl Loaded {
    /// The checkpoint's tensor names (for the source-field autocomplete).
    pub fn names(&self) -> &[String] {
        &self.all_names
    }

    /// The checkpoint root (for the mode's title).
    pub fn root(&self) -> &Path {
        &self.target.root
    }

    /// Rebuild each *changed* shard's header under `map` (renaming keys, keeping
    /// order and `__metadata__`). Pure — no I/O. Returns, per changed shard, its
    /// index into `self.headers`, the new compact JSON, and the `(old, new)` pairs.
    fn rebuild(&self, map: &NameMap) -> Vec<ShardRebuild> {
        let mut out = Vec::new();
        for (i, h) in self.headers.iter().enumerate() {
            let mut changed: Vec<(String, String)> = Vec::new();
            let mut new_obj = Map::new();
            for (k, v) in &h.obj {
                if k == "__metadata__" {
                    new_obj.insert(k.clone(), v.clone());
                    continue;
                }
                let nk = map.map(k).into_owned();
                if nk != *k {
                    changed.push((k.clone(), nk.clone()));
                }
                new_obj.insert(nk, v.clone());
            }
            if changed.is_empty() {
                continue;
            }
            let new_json = serde_json::to_vec(&Value::Object(new_obj)).unwrap_or_default();
            out.push((i, new_json, changed));
        }
        out
    }

    /// How each *changed* shard's rewritten header sizes up against its fixed region
    /// — the detail behind a `won't fit` verdict, so the user can see exactly which
    /// file overflows and by how much.
    pub fn shard_fits(&self, map: &NameMap) -> Vec<ShardFit> {
        self.rebuild(map)
            .into_iter()
            .map(|(i, new_json, changed)| ShardFit {
                file: self.headers[i]
                    .path
                    .file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                path: self.headers[i].path.to_string_lossy().into_owned(),
                current: self.headers[i].n,
                needed: new_json.len() as u64,
                tensors: changed.len(),
            })
            .collect()
    }

    /// A live preview of applying `map` — which tensors change and whether each can
    /// be done in place — computed entirely from the cached headers (no file reads),
    /// so it's cheap to recompute on every keystroke.
    pub fn preview(&self, map: &NameMap) -> RenamePreview {
        let rp = map.plan_renames(self.all_names.iter().map(String::as_str));
        let collisions: BTreeSet<&str> = rp.collisions.iter().map(String::as_str).collect();
        let rebuilt = self.rebuild(map);

        let mut rows: Vec<PreviewRow> = Vec::new();
        for (i, new_json, changed) in &rebuilt {
            let over = new_json.len() as u64 > self.headers[*i].n;
            for (old, new) in changed {
                let status = if new.is_empty() || new == "__metadata__" {
                    RenameStatus::Invalid
                } else if collisions.contains(new.as_str()) {
                    RenameStatus::Collision
                } else if over {
                    RenameStatus::WontFit
                } else {
                    RenameStatus::Ok
                };
                rows.push(PreviewRow {
                    old: old.clone(),
                    new: new.clone(),
                    status,
                });
            }
        }
        rows.sort_by(|a, b| a.old.cmp(&b.old));

        let mut warnings: Vec<String> = Vec::new();
        for idx in map.unmatched_rules(self.all_names.iter().map(String::as_str)) {
            warnings.push(format!("rule #{} matched no tensor names", idx + 1));
        }
        if !self.duplicated.is_empty() {
            warnings.push(format!(
                "the checkpoint already defines {} name(s) in more than one shard",
                self.duplicated.len()
            ));
        }
        RenamePreview {
            rows,
            warnings,
            has_index: self.target.index.is_some(),
        }
    }

    /// Build the fully-validated, ready-to-apply [`Plan`] for `map`. Returns a rich
    /// multi-line error listing *every* reason the rename is unsafe (collisions,
    /// invalid targets, headers that won't fit, an inconsistent index) so the user
    /// can fix them all at once.
    pub fn plan(&self, map: &NameMap) -> Result<Plan> {
        let mut problems: Vec<String> = Vec::new();
        let mut warnings: Vec<String> = Vec::new();

        if !self.duplicated.is_empty() {
            problems.push(format!(
                "the checkpoint already defines {} tensor name(s) in more than one shard: {}",
                self.duplicated.len(),
                preview_names(&self.duplicated),
            ));
        }

        let rp = map.plan_renames(self.all_names.iter().map(String::as_str));
        if rp.renames.is_empty() {
            problems
                .push("no tensor names matched the rename rules — nothing to rename".to_string());
        }
        if !rp.collisions.is_empty() {
            problems.push(format!(
                "{} rename target(s) would collide (two tensors mapped onto one name): {}",
                rp.collisions.len(),
                preview_names(&rp.collisions),
            ));
        }
        let bad_targets: Vec<String> = rp
            .renames
            .iter()
            .filter(|(_, to)| to.is_empty() || to == "__metadata__")
            .map(|(_, to)| to.clone())
            .collect();
        if !bad_targets.is_empty() {
            problems.push(format!(
                "{} rename target(s) are invalid (empty, or the reserved \"__metadata__\"): {}",
                bad_targets.len(),
                preview_names(&bad_targets),
            ));
        }
        for i in map.unmatched_rules(self.all_names.iter().map(String::as_str)) {
            warnings.push(format!("rename rule #{} matched no tensor names", i + 1));
        }

        // Only build the concrete headers (and thus the fit check + index update)
        // once the map is structurally sound — a collapsing insert would otherwise
        // report bogus sizes.
        let map_sound =
            !rp.renames.is_empty() && rp.collisions.is_empty() && bad_targets.is_empty();

        let mut shards: Vec<ShardPlan> = Vec::new();
        let mut index: Option<(PathBuf, String)> = None;
        if map_sound {
            for (i, new_json, mut changed) in self.rebuild(map) {
                let h = &self.headers[i];
                if new_json.len() as u64 > h.n {
                    problems.push(format!(
                        "{}: the renamed header is {} B but only {} B fit in place ({} B too big) — \
                         the new names are longer, which would require moving tensor data",
                        h.path.display(),
                        new_json.len(),
                        h.n,
                        new_json.len() as u64 - h.n,
                    ));
                    continue;
                }
                changed.sort();
                shards.push(ShardPlan {
                    path: h.path.clone(),
                    header_n: h.n,
                    new_json,
                    renames: changed,
                });
            }

            if let Some(index_path) = &self.target.index {
                match build_index_update(index_path, map, &self.all_names) {
                    Ok((new_text, mut w)) => {
                        warnings.append(&mut w);
                        if let Some(text) = new_text {
                            index = Some((index_path.clone(), text));
                        }
                    }
                    Err(e) => problems.push(format!("{e:#}")),
                }
            }
        }

        if !problems.is_empty() {
            bail!(
                "cannot rename tensors in {}:\n{}",
                self.target.root.display(),
                problems
                    .iter()
                    .map(|p| format!("  - {p}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
            );
        }

        // The applied rename set is exactly what the shard plans carry (every
        // renamed tensor lives in one shard); flatten them for the report.
        let mut renames: Vec<(String, String)> = shards
            .iter()
            .flat_map(|s| s.renames.iter().cloned())
            .collect();
        renames.sort();
        renames.dedup();
        Ok(Plan {
            target: Target {
                root: self.target.root.clone(),
                shards: self.target.shards.clone(),
                index: self.target.index.clone(),
            },
            shards,
            index,
            renames,
            warnings,
        })
    }
}

/// Build (and fully validate) a rename plan for the checkpoint at `path` under the
/// rules `map` — the one-shot path for the CLI (`load` then `plan`).
pub fn plan(path: &Path, map: &NameMap) -> Result<Plan> {
    load(path)?.plan(map)
}

/// Whether `path` can be opened for writing — the exact operation [`apply`]
/// performs, so it accounts for a read-only mount, the permission bits, ACLs, and
/// the immutable flag alike (a mode-bit check misses the mount; `access(2)` misses
/// the immutable flag). Non-destructive: opens `O_WRONLY` without truncating, so it
/// changes neither the contents nor the timestamps. Shared by the in-place-rename
/// pre-flight below and the `editable` badge's `checkpoint_writable` probe.
pub fn is_writable(path: &Path) -> bool {
    fs::OpenOptions::new().write(true).open(path).is_ok()
}

/// Apply a validated [`Plan`]: rewrite each shard's header region in place, then
/// update the index. Headers are written first so a rare mid-run I/O error can't
/// leave an index that points at renames the shards don't yet have.
pub fn apply(plan: &Plan) -> Result<()> {
    // Pre-flight: confirm every shard *and* the index can be opened for writing
    // before rewriting any of them, so a read-only file partway through can't leave
    // a partially-renamed (inconsistent) checkpoint. Cheap relative to the rewrite,
    // and turns a mid-run failure into a clean "nothing changed" one.
    for sp in &plan.shards {
        fs::OpenOptions::new()
            .write(true)
            .open(&sp.path)
            .with_context(|| {
                format!(
                    "{} is not writable — nothing was renamed",
                    sp.path.display()
                )
            })?;
    }
    if let Some((path, _)) = &plan.index {
        fs::OpenOptions::new()
            .write(true)
            .open(path)
            .with_context(|| format!("{} is not writable — nothing was renamed", path.display()))?;
    }

    for sp in &plan.shards {
        // The region after the 8-byte length is exactly `header_n` bytes: the new
        // JSON, space-padded back to that length so the data offsets stay valid.
        let mut buf = sp.new_json.clone();
        buf.resize(sp.header_n as usize, b' ');
        let mut f = fs::OpenOptions::new()
            .write(true)
            .open(&sp.path)
            .with_context(|| format!("opening {} for rename", sp.path.display()))?;
        f.seek(SeekFrom::Start(8))
            .with_context(|| format!("seeking in {}", sp.path.display()))?;
        f.write_all(&buf)
            .with_context(|| format!("rewriting {} header", sp.path.display()))?;
        f.flush()
            .with_context(|| format!("flushing {}", sp.path.display()))?;
    }
    if let Some((path, text)) = &plan.index {
        fs::write(path, text).with_context(|| format!("updating {}", path.display()))?;
    }
    Ok(())
}

/// Rewrite the `weight_map` keys of an index.json under `map`. Returns the full
/// new file text (when any key changed) plus non-fatal warnings (e.g. the index is
/// inconsistent with the shards). A parse error is fatal (bubbled up).
fn build_index_update(
    index_path: &Path,
    map: &NameMap,
    tensor_names: &[String],
) -> Result<(Option<String>, Vec<String>)> {
    let text = fs::read_to_string(index_path)
        .with_context(|| format!("reading {}", index_path.display()))?;
    let mut root: Value = serde_json::from_str(&text)
        .with_context(|| format!("{}: invalid JSON", index_path.display()))?;
    let wm = root
        .get_mut("weight_map")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| anyhow!("{}: no \"weight_map\" object", index_path.display()))?;

    let mut warnings = Vec::new();
    let wm_keys: BTreeSet<&str> = wm.keys().map(String::as_str).collect();
    let shard_keys: BTreeSet<&str> = tensor_names.iter().map(String::as_str).collect();
    if wm_keys != shard_keys {
        let only_index = wm_keys.difference(&shard_keys).count();
        let only_shards = shard_keys.difference(&wm_keys).count();
        warnings.push(format!(
            "{}: weight_map disagrees with the shards ({only_index} name(s) only in the index, \
             {only_shards} only in the shards) — renaming the names that match",
            index_path.display(),
        ));
    }

    let old_wm = wm.clone();
    let mut new_wm = Map::new();
    let mut changed = false;
    for (k, v) in old_wm {
        let nk = map.map(&k).into_owned();
        if nk != k {
            changed = true;
        }
        new_wm.insert(nk, v);
    }
    *wm = new_wm;

    if !changed {
        return Ok((None, warnings));
    }
    let mut new_text = serde_json::to_string_pretty(&root)
        .with_context(|| format!("{}: re-serialising", index_path.display()))?;
    new_text.push('\n'); // match the trailing newline these files ship with
    Ok((Some(new_text), warnings))
}

/// The set of shard filenames referenced by an index.json's `weight_map` values.
fn index_shard_files(index_path: &Path) -> Result<BTreeSet<String>> {
    let text = fs::read_to_string(index_path)
        .with_context(|| format!("reading {}", index_path.display()))?;
    let root: Value = serde_json::from_str(&text)
        .with_context(|| format!("{}: invalid JSON", index_path.display()))?;
    let wm = root
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow!("{}: no \"weight_map\" object", index_path.display()))?;
    Ok(wm
        .values()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect())
}

/// Turn a concrete tensor name into an editable **schema**: each run of digits is
/// replaced by a placeholder token — `{layer}` after a `layers`/`block`-style
/// segment, `{expert}` after an `experts` segment, else `{n0}`, `{n1}`, … — so
/// editing the schema and renaming applies to *every* layer / expert at once.
/// Returns the schema string plus the placeholder tokens in left-to-right order
/// (that order is the regex capture-group order [`rule_from_fields`] maps to).
///
/// e.g. `model.layers.3.mlp.experts.5.down_proj.weight`
///   →  `model.layers.{layer}.mlp.experts.{expert}.down_proj.weight`, `[layer, expert]`
pub fn generalize(name: &str) -> (String, Vec<String>) {
    let chars: Vec<char> = name.chars().collect();
    let mut schema = String::new();
    let mut tokens: Vec<String> = Vec::new();
    let mut generic = 0usize;
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_ascii_digit() {
            let start = i;
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            let prefix: String = chars[..start].iter().collect();
            let base = classify_number_segment(&prefix, &mut generic);
            let token = unique_token(&base, &tokens);
            schema.push('{');
            schema.push_str(&token);
            schema.push('}');
            tokens.push(token);
        } else {
            schema.push(chars[i]);
            i += 1;
        }
    }
    (schema, tokens)
}

/// Name a numeric field from the path segment right before it: a layer index after
/// a `layers`/`block`-style segment, an expert index after `experts`, else a
/// generic `n0`, `n1`, … (bumping `generic`).
fn classify_number_segment(prefix: &str, generic: &mut usize) -> String {
    let seg = prefix
        .trim_end_matches(['.', '_', '/', '-'])
        .rsplit(['.', '_', '/', '-'])
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    match seg.as_str() {
        "layers" | "layer" | "blocks" | "block" | "h" => "layer".to_string(),
        "experts" | "expert" => "expert".to_string(),
        _ => {
            let t = format!("n{generic}");
            *generic += 1;
            t
        }
    }
}

/// Disambiguate a placeholder base against those already used (`layer`, `layer2`…).
fn unique_token(base: &str, used: &[String]) -> String {
    if !used.iter().any(|t| t == base) {
        return base.to_string();
    }
    (2..)
        .map(|k| format!("{base}{k}"))
        .find(|c| !used.iter().any(|t| t == c))
        .unwrap()
}

/// Build a `(pattern, replacement)` rename rule from a source and a new name, both
/// as *schemas*: the `{token}` placeholders (from [`generalize`]) are the only
/// wildcards — each becomes an anchored `(\d+)` capture in the source and the
/// matching backreference in the new name, so one edit renames every layer /
/// expert. Everything else matches **exactly, literal digits included** — so a
/// concrete `…layers.0.…` source renames only layer 0. Errors if the new name uses
/// a token the source doesn't define. Shared by the TUI editor and the CLI `--map`.
pub fn rule_from_fields(
    source: &str,
    target: &str,
) -> std::result::Result<(String, String), String> {
    if source.trim().is_empty() {
        return Err("choose a source tensor".to_string());
    }
    if target.trim().is_empty() {
        return Err("enter the new name".to_string());
    }

    // Pattern from the source: `{token}` → an anchored `(\d+)` capture (recorded in
    // order); every other run — literal digits included — is matched verbatim.
    let mut pattern = String::from("^");
    let mut tokens: Vec<String> = Vec::new();
    let mut lit = String::new();
    let chars: Vec<char> = source.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '{' {
            let rel = chars[i + 1..]
                .iter()
                .position(|&c| c == '}')
                .ok_or_else(|| "unclosed `{` in the source".to_string())?;
            let tok: String = chars[i + 1..i + 1 + rel].iter().collect();
            if tok.trim().is_empty() {
                return Err("empty `{}` placeholder in the source".to_string());
            }
            if !lit.is_empty() {
                pattern.push_str(&regex::escape(&lit));
                lit.clear();
            }
            pattern.push_str(r"(\d+)");
            tokens.push(tok);
            i += 1 + rel + 1;
        } else {
            lit.push(chars[i]);
            i += 1;
        }
    }
    if !lit.is_empty() {
        pattern.push_str(&regex::escape(&lit));
    }
    pattern.push('$');

    // Replacement from the target schema: `{token}` → `${group}`, literal `$` → `$$`.
    let group_of: HashMap<&str, usize> = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t.as_str(), i + 1))
        .collect();
    let tchars: Vec<char> = target.chars().collect();
    let mut replacement = String::new();
    let mut i = 0;
    while i < tchars.len() {
        match tchars[i] {
            '{' => {
                let rel = tchars[i + 1..]
                    .iter()
                    .position(|&c| c == '}')
                    .ok_or_else(|| "unclosed `{` in the new name".to_string())?;
                let tok: String = tchars[i + 1..i + 1 + rel].iter().collect();
                let g = group_of.get(tok.as_str()).ok_or_else(|| {
                    let avail = tokens
                        .iter()
                        .map(|t| format!("{{{t}}}"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    if avail.is_empty() {
                        format!(
                            "the source has no {{…}} placeholders, so the placeholder {{{tok}}} \
                             can't be used (add one to the source to make it a wildcard)"
                        )
                    } else {
                        format!("unknown placeholder {{{tok}}} — the source has: {avail}")
                    }
                })?;
                replacement.push_str(&format!("${{{g}}}"));
                i += 1 + rel + 1;
            }
            '$' => {
                replacement.push_str("$$");
                i += 1;
            }
            c => {
                replacement.push(c);
                i += 1;
            }
        }
    }
    Ok((pattern, replacement))
}

/// A compact, comma-joined preview of up to six names (with a `+N more` tail).
fn preview_names(names: &[String]) -> String {
    const MAX: usize = 6;
    if names.len() <= MAX {
        names.join(", ")
    } else {
        format!(
            "{}, … (+{} more)",
            names[..MAX].join(", "),
            names.len() - MAX
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a safetensors file from `(name, dtype, nbytes)` tensors laid out
    /// back-to-back, with an optional trailing header padding (extra spaces baked
    /// into `N`). Returns the header length `N`.
    fn write_st(path: &Path, tensors: &[(&str, &str, u64)], pad: usize) -> u64 {
        let mut offset = 0u64;
        let entries: Vec<String> = tensors
            .iter()
            .map(|(name, dtype, nbytes)| {
                let e = format!(
                    "{name:?}:{{\"dtype\":\"{dtype}\",\"shape\":[{nbytes}],\"data_offsets\":[{},{}]}}",
                    offset,
                    offset + nbytes,
                );
                offset += nbytes;
                e
            })
            .collect();
        let mut json = format!("{{{}}}", entries.join(","));
        json.push_str(&" ".repeat(pad)); // spec-legal header padding
        let n = json.len() as u64;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&n.to_le_bytes());
        bytes.extend_from_slice(json.as_bytes());
        bytes.extend_from_slice(&vec![7u8; offset as usize]); // dummy tensor data
        fs::write(path, bytes).unwrap();
        n
    }

    fn map_from(rules: &[&str]) -> NameMap {
        let pairs = NameMap::parse_rules(rules.iter().copied()).unwrap();
        NameMap::from_pairs(pairs).unwrap()
    }

    #[test]
    fn same_length_rename_fits_and_round_trips() {
        let dir = std::env::temp_dir().join("ce_rename_fit");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("model.safetensors");
        // "aaa"/"bbb" → "xxx"/"yyy": same length, so it fits with no padding.
        write_st(&file, &[("aaa", "F32", 8), ("bbb", "F32", 8)], 0);

        let map = map_from(&["a=>x", "b=>y"]);
        let plan = plan(&file, &map).unwrap();
        assert_eq!(plan.rename_count(), 2);
        apply(&plan).unwrap();

        let layout = safelayout::parse(&file).unwrap();
        let names: BTreeSet<&str> = layout
            .segments
            .iter()
            .filter(|s| s.kind == safelayout::SegmentKind::Tensor)
            .map(|s| s.name.as_str())
            .collect();
        assert_eq!(names, BTreeSet::from(["xxx", "yyy"]));
        // Data untouched: the file still ends in the seven dummy data bytes.
        let raw = fs::read(&file).unwrap();
        assert!(raw.ends_with(&[7u8; 16]));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn apply_bails_without_writing_when_a_shard_is_read_only() {
        let dir = std::env::temp_dir().join("ce_rename_ro");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("model.safetensors");
        write_st(&file, &[("aaa", "F32", 8), ("bbb", "F32", 8)], 0);
        let map = map_from(&["a=>x", "b=>y"]);
        let plan = plan(&file, &map).unwrap();
        let before = fs::read(&file).unwrap();

        // Read-only shard → the pre-flight refuses before rewriting anything.
        let mut perms = fs::metadata(&file).unwrap().permissions();
        perms.set_readonly(true);
        fs::set_permissions(&file, perms).unwrap();

        let err = apply(&plan).unwrap_err();
        assert!(
            format!("{err:#}").contains("not writable"),
            "unexpected error: {err:#}"
        );
        // The file is still readable (0444), so compare without restoring write —
        // and removing it only needs a writable parent dir.
        assert_eq!(
            fs::read(&file).unwrap(),
            before,
            "a read-only shard must be left byte-for-byte untouched"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn longer_rename_that_overflows_is_rejected() {
        let dir = std::env::temp_dir().join("ce_rename_overflow");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("model.safetensors");
        // No padding, so lengthening a name can't fit in place.
        write_st(&file, &[("w", "F32", 8)], 0);
        let map = map_from(&["^w$=>a_much_longer_tensor_name"]);
        let err = plan(&file, &map).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("fit in place"), "{msg}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn padding_lets_a_longer_rename_fit() {
        let dir = std::env::temp_dir().join("ce_rename_pad");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("model.safetensors");
        // 40 bytes of header padding gives room to lengthen "w" → "weight".
        write_st(&file, &[("w", "F32", 8)], 40);
        let map = map_from(&["^w$=>weight"]);
        let plan = plan(&file, &map).unwrap();
        apply(&plan).unwrap();
        let layout = safelayout::parse(&file).unwrap();
        assert!(
            layout
                .segments
                .iter()
                .any(|s| s.name == "weight" && s.kind == safelayout::SegmentKind::Tensor)
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn collision_is_rejected() {
        let dir = std::env::temp_dir().join("ce_rename_collide");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("model.safetensors");
        write_st(&file, &[("a", "F32", 8), ("b", "F32", 8)], 0);
        // Both names map onto "same".
        let map = map_from(&["^[ab]$=>same"]);
        let err = plan(&file, &map).unwrap_err();
        assert!(format!("{err:#}").contains("collide"), "{err:#}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn no_match_is_rejected() {
        let dir = std::env::temp_dir().join("ce_rename_nomatch");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("model.safetensors");
        write_st(&file, &[("a", "F32", 8)], 0);
        let map = map_from(&["zzz=>qqq"]);
        let err = plan(&file, &map).unwrap_err();
        assert!(format!("{err:#}").contains("nothing to rename"), "{err:#}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn sharded_index_weight_map_is_updated() {
        let dir = std::env::temp_dir().join("ce_rename_index");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let f1 = dir.join("model-00001-of-00002.safetensors");
        let f2 = dir.join("model-00002-of-00002.safetensors");
        write_st(&f1, &[("enc.a", "F32", 8)], 8);
        write_st(&f2, &[("enc.b", "F32", 8)], 8);
        let index = dir.join(INDEX_NAME);
        fs::write(
            &index,
            serde_json::to_string_pretty(&serde_json::json!({
                "metadata": {"total_size": 16},
                "weight_map": {
                    "enc.a": "model-00001-of-00002.safetensors",
                    "enc.b": "model-00002-of-00002.safetensors",
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let map = map_from(&["^enc\\.=>encoder."]);
        let plan = plan(&dir, &map).unwrap();
        assert_eq!(plan.shard_count(), 2);
        assert!(plan.index.is_some());
        apply(&plan).unwrap();

        let new_index: Value = serde_json::from_str(&fs::read_to_string(&index).unwrap()).unwrap();
        let wm = new_index["weight_map"].as_object().unwrap();
        assert!(wm.contains_key("encoder.a"));
        assert!(wm.contains_key("encoder.b"));
        assert!(!wm.contains_key("enc.a"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn generalize_names_layer_and_expert_numbers() {
        let (schema, tokens) = generalize("model.layers.3.mlp.experts.5.down_proj.weight");
        assert_eq!(
            schema,
            "model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"
        );
        assert_eq!(tokens, vec!["layer".to_string(), "expert".to_string()]);

        // A single layer number.
        let (schema, tokens) = generalize("model.layers.0.self_attn.q_proj.weight");
        assert_eq!(schema, "model.layers.{layer}.self_attn.q_proj.weight");
        assert_eq!(tokens, vec!["layer".to_string()]);

        // No numbers → unchanged, no tokens.
        let (schema, tokens) = generalize("model.embed_tokens.weight");
        assert_eq!(schema, "model.embed_tokens.weight");
        assert!(tokens.is_empty());
    }

    #[test]
    fn rule_from_fields_builds_a_family_wide_rule_from_placeholders() {
        // A `{layer}` placeholder in BOTH fields renames the tensor across every
        // layer (the placeholder is the wildcard).
        let (pat, rep) = rule_from_fields(
            "model.layers.{layer}.self_attn.q_proj.weight",
            "model.layers.{layer}.attn.q_proj.weight",
        )
        .unwrap();
        let map = NameMap::from_pairs([(pat, rep)]).unwrap();
        assert_eq!(
            map.map("model.layers.0.self_attn.q_proj.weight")
                .into_owned(),
            "model.layers.0.attn.q_proj.weight"
        );
        assert_eq!(
            map.map("model.layers.11.self_attn.q_proj.weight")
                .into_owned(),
            "model.layers.11.attn.q_proj.weight"
        );
        // A different tensor (not this family) is untouched — the rule is anchored.
        assert_eq!(
            map.map("model.layers.0.mlp.up_proj.weight").into_owned(),
            "model.layers.0.mlp.up_proj.weight"
        );
    }

    #[test]
    fn rule_from_fields_keeps_a_concrete_number_literal() {
        // A concrete number in the source (no placeholder) matches only that layer,
        // not the whole family.
        let (pat, rep) = rule_from_fields(
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.attn.q_proj.weight",
        )
        .unwrap();
        let map = NameMap::from_pairs([(pat, rep)]).unwrap();
        assert_eq!(
            map.map("model.layers.0.self_attn.q_proj.weight")
                .into_owned(),
            "model.layers.0.attn.q_proj.weight"
        );
        // Layer 1 is left alone.
        assert_eq!(
            map.map("model.layers.1.self_attn.q_proj.weight")
                .into_owned(),
            "model.layers.1.self_attn.q_proj.weight"
        );
    }

    #[test]
    fn rule_from_fields_rejects_an_unknown_placeholder() {
        // {layer} exists in the source but {bogus} does not.
        let err = rule_from_fields(
            "model.layers.{layer}.self_attn.q_proj.weight",
            "model.layers.{bogus}.attn.q_proj.weight",
        )
        .unwrap_err();
        assert!(err.contains("unknown placeholder"), "{err}");
    }

    #[test]
    fn preview_marks_status_per_pair_without_io_repeats() {
        let dir = std::env::temp_dir().join("ce_rename_preview");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("model.safetensors");
        // No header padding, so any lengthening overflows in place.
        write_st(&file, &[("a", "F32", 8), ("b", "F32", 8)], 0);
        let loaded = load(&file).unwrap();

        // A same-length rename applies cleanly.
        let prev = loaded.preview(&map_from(&["^a$=>x"]));
        assert_eq!(prev.rows.len(), 1);
        assert_eq!(prev.rows[0].status, RenameStatus::Ok);
        assert!(prev.applicable());

        // A much longer target won't fit in place — the pair is marked, not a bail.
        let prev = loaded.preview(&map_from(&["^a$=>a_very_long_new_tensor_name"]));
        assert_eq!(prev.rows[0].status, RenameStatus::WontFit);
        assert!(!prev.applicable());

        // Two names onto one → both marked as collisions.
        let prev = loaded.preview(&map_from(&["^[ab]$=>same"]));
        assert!(
            prev.rows
                .iter()
                .all(|r| r.status == RenameStatus::Collision)
        );
        assert!(!prev.applicable());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn lone_shard_of_a_sharded_set_is_refused() {
        let dir = std::env::temp_dir().join("ce_rename_lone");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let f1 = dir.join("model-00001-of-00002.safetensors");
        write_st(&f1, &[("a", "F32", 8)], 0);
        fs::write(
            dir.join(INDEX_NAME),
            serde_json::json!({"weight_map": {
                "a": "model-00001-of-00002.safetensors",
                "b": "model-00002-of-00002.safetensors",
            }})
            .to_string(),
        )
        .unwrap();
        let err = discover(&f1).unwrap_err();
        assert!(
            format!("{err:#}").contains("point at the directory"),
            "{err:#}"
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
