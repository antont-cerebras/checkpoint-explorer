//! Checkpoint health check: compare a `model.safetensors.index.json` against
//! the `.safetensors` files actually present, at both the file and tensor
//! level, and report any mismatch.

use anyhow::{Context, Result};
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use crate::tree::TensorInfo;

/// The result of comparing an index against the files on disk.
pub struct HealthReport {
    /// The index file that was checked.
    pub index_path: String,
    /// Files referenced by the index but absent on disk.
    pub missing_files: Vec<String>,
    /// `.safetensors` files present on disk but not referenced by the index.
    pub extra_files: Vec<String>,
    /// Tensors the index assigns to a present file that the file does not
    /// contain (formatted with the expected file).
    pub missing_tensors: Vec<String>,
    /// Tensors found in a referenced, present file that the index does not
    /// assign there (formatted with the containing file).
    pub extra_tensors: Vec<String>,
}

impl HealthReport {
    pub fn has_issues(&self) -> bool {
        !self.missing_files.is_empty()
            || !self.extra_files.is_empty()
            || !self.missing_tensors.is_empty()
            || !self.extra_tensors.is_empty()
    }
}

/// A checkpoint directory's index parsed once, ready to health-check against the
/// tensors the loader parses — so the shard headers are read a single time (by the
/// loader) rather than again here. Built by [`parse_index_spec`]; consumed by
/// [`check_loaded`].
pub struct IndexSpec {
    /// The directory the index and its shards live in.
    pub dir: PathBuf,
    /// The `model.safetensors.index.json` path (for the report label).
    pub index_path: PathBuf,
    /// tensor name -> file the index claims it lives in.
    pub weight_map: HashMap<String, String>,
}

/// Read and parse a `model.safetensors.index.json` once into an [`IndexSpec`].
pub fn parse_index_spec(dir: &Path, index_path: &Path) -> Result<IndexSpec> {
    Ok(IndexSpec {
        dir: dir.to_path_buf(),
        index_path: index_path.to_path_buf(),
        weight_map: parse_weight_map(index_path)?,
    })
}

/// Compare an index against the checkpoint as actually loaded: the `.safetensors`
/// files on disk (a directory listing — no header reads) and the tensor names the
/// loader already parsed from each shard's header. `tensors` is the whole loaded
/// set; only those whose `source_path` is a file directly in `spec.dir` count, so
/// this is safe when several checkpoints are loaded together.
pub fn check_loaded(spec: &IndexSpec, tensors: &[TensorInfo]) -> HealthReport {
    let actual = list_safetensors(&spec.dir);

    // Tensor names present per file, taken from the already-parsed tensors (grouped
    // by the file name of their `source_path`) — keyed to `spec.dir` so a same-named
    // shard in another loaded directory can't leak in.
    let abs_dir = std::path::absolute(&spec.dir).unwrap_or_else(|_| spec.dir.clone());
    let mut present_by_file: HashMap<String, BTreeSet<String>> = HashMap::new();
    for t in tensors {
        let path = Path::new(&t.source_path);
        let in_dir = path
            .parent()
            .map(|p| p == abs_dir || p == spec.dir)
            .unwrap_or(false);
        if !in_dir {
            continue;
        }
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            present_by_file
                .entry(name.to_string())
                .or_default()
                .insert(t.name.clone());
        }
    }

    reconcile(
        &spec.index_path.display().to_string(),
        &spec.weight_map,
        &actual,
        &present_by_file,
    )
}

/// The pure index-vs-checkpoint comparison shared by the local
/// ([`check_loaded`]) and remote (`--ssh-read`) health checks: given the index's
/// `weight_map` (tensor -> file), the `.safetensors` files actually present, and
/// the tensor names present in each file (from the already-parsed headers), report
/// the file- and tensor-level mismatches. No I/O — both callers supply the pieces
/// from data they've already read, so a header is never read twice.
pub fn reconcile(
    index_path: &str,
    weight_map: &HashMap<String, String>,
    actual: &BTreeSet<String>,
    present_by_file: &HashMap<String, BTreeSet<String>>,
) -> HealthReport {
    let referenced: BTreeSet<String> = weight_map.values().cloned().collect();

    // File-level diff.
    let missing_files: Vec<String> = referenced.difference(actual).cloned().collect();
    let extra_files: Vec<String> = actual.difference(&referenced).cloned().collect();

    // Tensor-level diff, limited to files that are both referenced and present
    // (wholesale-missing / wholesale-extra files are already covered above).
    let mut claimed_by_file: HashMap<String, BTreeSet<String>> = HashMap::new();
    for (tensor, file) in weight_map {
        claimed_by_file
            .entry(file.clone())
            .or_default()
            .insert(tensor.clone());
    }

    let mut missing_tensors = Vec::new();
    let mut extra_tensors = Vec::new();
    for file in referenced.intersection(actual) {
        let present = present_by_file.get(file).cloned().unwrap_or_default();
        let claimed = claimed_by_file.get(file).cloned().unwrap_or_default();
        for tensor in claimed.difference(&present) {
            missing_tensors.push(format!("{tensor}  (expected in {file})"));
        }
        for tensor in present.difference(&claimed) {
            extra_tensors.push(format!("{tensor}  (in {file})"));
        }
    }
    missing_tensors.sort();
    extra_tensors.sort();

    HealthReport {
        index_path: index_path.to_string(),
        missing_files,
        extra_files,
        missing_tensors,
        extra_tensors,
    }
}

/// Parse the `weight_map` of an index into a tensor-name -> file-name map.
fn parse_weight_map(index_path: &Path) -> Result<HashMap<String, String>> {
    let content = std::fs::read_to_string(index_path)
        .with_context(|| format!("Failed to read index file: {}", index_path.display()))?;
    let index: serde_json::Value = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse index file: {}", index_path.display()))?;

    let mut map = HashMap::new();
    if let Some(weight_map) = index.get("weight_map").and_then(|v| v.as_object()) {
        for (tensor, file) in weight_map {
            if let Some(file) = file.as_str() {
                map.insert(tensor.clone(), file.to_string());
            }
        }
    }
    Ok(map)
}

/// The set of `.safetensors` file names directly inside `dir`.
fn list_safetensors(dir: &Path) -> BTreeSet<String> {
    let mut files = BTreeSet::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("safetensors")
                && let Some(name) = path.file_name().and_then(|n| n.to_str())
            {
                files.insert(name.to_string());
            }
        }
    }
    files
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::{Layout, Storage};

    /// A `TensorInfo` named `name` whose `source_path` is `dir/file` (absolute,
    /// matching what the loader records) — enough for the health check's grouping.
    fn ti(name: &str, dir: &Path, file: &str) -> TensorInfo {
        let src = std::path::absolute(dir.join(file))
            .unwrap()
            .to_string_lossy()
            .into_owned();
        TensorInfo {
            name: name.to_string(),
            dtype: "F32".into(),
            shape: vec![1],
            size_bytes: 4,
            num_elements: 1,
            storage: Storage::Unknown,
            source_path: src,
            layout: Layout::None,
        }
    }

    #[test]
    fn detects_file_and_tensor_mismatches() {
        let dir = std::env::temp_dir().join("checkpoint_explorer_health_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // Index references model-00001 (present) and model-00002 (missing).
        let index = serde_json::json!({
            "weight_map": {
                "a.weight": "model-00001.safetensors",
                "b.weight": "model-00002.safetensors"
            }
        });
        let index_path = dir.join("model.safetensors.index.json");
        std::fs::write(&index_path, serde_json::to_vec(&index).unwrap()).unwrap();

        // On disk: model-00001 (present) and model-00003 (never referenced); the
        // file-level check only lists directory entries, so empty files suffice.
        std::fs::write(dir.join("model-00001.safetensors"), b"").unwrap();
        std::fs::write(dir.join("model-00003.safetensors"), b"").unwrap();

        // The tensors the loader parsed: model-00001 holds the claimed a.weight
        // plus an unlisted extra; model-00003 holds c.weight.
        let tensors = vec![
            ti("a.weight", &dir, "model-00001.safetensors"),
            ti("extra.weight", &dir, "model-00001.safetensors"),
            ti("c.weight", &dir, "model-00003.safetensors"),
        ];

        let spec = parse_index_spec(&dir, &index_path).unwrap();
        let report = check_loaded(&spec, &tensors);
        assert!(report.has_issues());
        assert_eq!(report.missing_files, vec!["model-00002.safetensors"]);
        assert_eq!(report.extra_files, vec!["model-00003.safetensors"]);
        // `extra.weight` lives in a referenced+present file but the index does
        // not list it there.
        assert!(
            report
                .extra_tensors
                .iter()
                .any(|t| t.starts_with("extra.weight"))
        );
        // `a.weight` matches; `b.weight`'s file is missing (covered by the file
        // diff), so there are no tensor-level "missing" entries.
        assert!(report.missing_tensors.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
