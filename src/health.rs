//! Checkpoint health check: compare a `model.safetensors.index.json` against
//! the `.safetensors` files actually present, at both the file and tensor
//! level, and report any mismatch.

use anyhow::{Context, Result};
use std::collections::{BTreeSet, HashMap};
use std::path::Path;

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

/// Compare `index_path` (a `model.safetensors.index.json` inside `dir`) against
/// the `.safetensors` files present in `dir`.
pub fn check(dir: &Path, index_path: &Path) -> Result<HealthReport> {
    // tensor name -> file the index claims it lives in
    let weight_map = parse_weight_map(index_path)?;
    let referenced: BTreeSet<String> = weight_map.values().cloned().collect();
    let actual = list_safetensors(dir);

    // File-level diff.
    let missing_files: Vec<String> = referenced.difference(&actual).cloned().collect();
    let extra_files: Vec<String> = actual.difference(&referenced).cloned().collect();

    // Tensor-level diff, limited to files that are both referenced and present
    // (wholesale-missing / wholesale-extra files are already covered above).
    let mut claimed_by_file: HashMap<String, BTreeSet<String>> = HashMap::new();
    for (tensor, file) in &weight_map {
        claimed_by_file
            .entry(file.clone())
            .or_default()
            .insert(tensor.clone());
    }

    let mut missing_tensors = Vec::new();
    let mut extra_tensors = Vec::new();
    for file in referenced.intersection(&actual) {
        let present: BTreeSet<String> = match read_tensor_names(&dir.join(file)) {
            Ok(names) => names.into_iter().collect(),
            Err(_) => continue, // unreadable header; skip tensor-level for this file
        };
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

    Ok(HealthReport {
        index_path: index_path.display().to_string(),
        missing_files,
        extra_files,
        missing_tensors,
        extra_tensors,
    })
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

/// Read just the tensor names from a safetensors header (excludes the
/// `__metadata__` entry).
fn read_tensor_names(path: &Path) -> Result<Vec<String>> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open file: {}", path.display()))?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let header_len = u64::from_le_bytes(len_buf) as usize;

    const MAX_HEADER_SIZE: usize = 100_000_000;
    if header_len > MAX_HEADER_SIZE {
        anyhow::bail!("SafeTensors header too large: {}", path.display());
    }

    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf)?;
    let header: serde_json::Value = serde_json::from_slice(&header_buf)?;

    let names = header
        .as_object()
        .map(|obj| {
            obj.keys()
                .filter(|k| *k != "__metadata__")
                .cloned()
                .collect()
        })
        .unwrap_or_default();
    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Write a minimal safetensors file containing the given tensor names.
    fn write_safetensors(path: &Path, tensors: &[&str]) {
        let mut entries = serde_json::Map::new();
        for (i, name) in tensors.iter().enumerate() {
            entries.insert(
                name.to_string(),
                serde_json::json!({
                    "dtype": "F32", "shape": [1], "data_offsets": [i * 4, i * 4 + 4]
                }),
            );
        }
        let header = serde_json::to_vec(&serde_json::Value::Object(entries)).unwrap();
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&header).unwrap();
        f.write_all(&vec![0u8; tensors.len() * 4]).unwrap();
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

        // Present: model-00001 (the claimed a.weight plus an unlisted extra),
        // and model-00003 which the index never references.
        write_safetensors(
            &dir.join("model-00001.safetensors"),
            &["a.weight", "extra.weight"],
        );
        write_safetensors(&dir.join("model-00003.safetensors"), &["c.weight"]);

        let report = check(&dir, &index_path).unwrap();
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
