//! The file-browser tree: a checkpoint's directory shown as a hierarchy of
//! directories and files (the `Tab` file view). Kept separate from the tensor
//! [`crate::tree::TreeNode`] so the mature tensor paths stay untouched — this
//! models only what a file explorer needs (name, path, size, kind, expansion).

use std::path::{Path, PathBuf};

use crate::tree::natural_sort_key;

/// What a file is, for its glyph and what `Enter` does with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileKind {
    /// A checkpoint we can open in the tensor view.
    Checkpoint,
    /// JSON — previewed with syntax highlighting.
    Json,
    /// Other UTF-8 text (README, LICENSE, .txt, .py, …) — previewed plain.
    Text,
    /// Anything else (binary, unknown) — info only.
    Other,
}

impl FileKind {
    /// Classify by extension / name. `Text` is a best-effort guess refined when
    /// the file is actually read (a non-UTF-8 "text" file falls back to info).
    pub fn of(name: &str) -> FileKind {
        let lower = name.to_ascii_lowercase();
        let ext = lower.rsplit('.').next().unwrap_or("");
        match ext {
            "safetensors" | "gguf" | "npy" | "npz" | "h5" | "hdf5" => FileKind::Checkpoint,
            "json" => FileKind::Json,
            "txt" | "md" | "py" | "yaml" | "yml" | "toml" | "cfg" | "ini" | "csv" | "tsv"
            | "jsonl" | "text" | "log" | "sh" | "rs" => FileKind::Text,
            // Extensionless docs that are conventionally text.
            _ if matches!(
                lower.as_str(),
                "readme" | "license" | "licence" | "notice" | "authors" | "copying" | "changelog"
            ) =>
            {
                FileKind::Text
            }
            _ => FileKind::Other,
        }
    }
}

/// A node in the file-browser tree.
#[derive(Debug, Clone)]
pub enum FileNode {
    Dir {
        name: String,
        path: PathBuf,
        children: Vec<FileNode>,
        expanded: bool,
        /// Aggregate size (bytes) and file count of everything under here.
        size: u64,
        files: usize,
    },
    File {
        name: String,
        path: PathBuf,
        size: u64,
        kind: FileKind,
    },
}

impl FileNode {
    pub fn size(&self) -> u64 {
        match self {
            FileNode::Dir { size, .. } | FileNode::File { size, .. } => *size,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            FileNode::Dir { name, .. } | FileNode::File { name, .. } => name,
        }
    }
}

/// Build the file tree rooted at `root`, recursively (bounded by `max_depth` to
/// avoid pathological trees). Directories sort first, then files, each in natural
/// order; hidden entries (dotfiles) are skipped. The returned node is the root
/// directory itself, expanded. Symlinked directories are listed but not descended
/// (their size counts as the link's, i.e. skipped) to avoid cycles.
pub fn build(root: &Path, max_depth: usize) -> FileNode {
    let mut node = build_dir(root, root_name(root), max_depth);
    // The root is expanded so its contents show immediately; nested dirs start
    // collapsed (a checkpoint is usually flat, so this rarely matters).
    if let FileNode::Dir { expanded, .. } = &mut node {
        *expanded = true;
    }
    node
}

/// The label for the root directory node — its final component, or the whole
/// path when it has none (e.g. `/`).
fn root_name(root: &Path) -> String {
    root.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| root.to_string_lossy().into_owned())
}

fn build_dir(dir: &Path, name: String, depth_left: usize) -> FileNode {
    let mut dirs: Vec<FileNode> = Vec::new();
    let mut files: Vec<FileNode> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let fname = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) if !n.starts_with('.') => n.to_string(),
                _ => continue, // unreadable name or a dotfile
            };
            // `file_type` doesn't follow symlinks, so a symlinked dir reads as a
            // symlink and is treated as a leaf (never descended → no cycles).
            let ft = entry.file_type().ok();
            let is_dir = ft.is_some_and(|t| t.is_dir());
            if is_dir && depth_left > 0 {
                dirs.push(build_dir(&path, fname, depth_left - 1));
            } else if is_dir {
                // Depth limit reached: represent as an empty (unexpanded) dir.
                dirs.push(FileNode::Dir {
                    name: fname,
                    path,
                    children: Vec::new(),
                    expanded: false,
                    size: 0,
                    files: 0,
                });
            } else {
                let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                let kind = FileKind::of(&fname);
                files.push(FileNode::File {
                    name: fname,
                    path,
                    size,
                    kind,
                });
            }
        }
    }
    let by_name =
        |a: &FileNode, b: &FileNode| natural_sort_key(a.name()).cmp(&natural_sort_key(b.name()));
    dirs.sort_by(by_name);
    files.sort_by(by_name);
    let mut children = dirs;
    children.extend(files);

    let size = children.iter().map(FileNode::size).sum();
    let file_count = children
        .iter()
        .map(|c| match c {
            FileNode::Dir { files, .. } => *files,
            FileNode::File { .. } => 1,
        })
        .sum();

    FileNode::Dir {
        name,
        path: dir.to_path_buf(),
        children,
        expanded: false, // the root is expanded by `build`; nested dirs collapsed
        size,
        files: file_count,
    }
}

/// One visible row of the flattened file tree — the data a row needs to render
/// and to act on (`Enter`), so the browser never re-walks the tree per frame.
#[derive(Debug, Clone)]
pub struct FileRow {
    pub depth: usize,
    pub name: String,
    pub path: PathBuf,
    pub size: u64,
    pub is_dir: bool,
    pub expanded: bool,
    /// Number of files under a directory (0 for a file row).
    pub files: usize,
    /// A file's kind (`Checkpoint` for a dir row — unused there).
    pub kind: FileKind,
}

/// Flatten the tree into the visible rows (a collapsed directory hides its
/// subtree), root first, mirroring the tensor tree's flattening.
pub fn flatten(root: &FileNode) -> Vec<FileRow> {
    let mut out = Vec::new();
    flatten_node(root, 0, &mut out);
    out
}

fn flatten_node(node: &FileNode, depth: usize, out: &mut Vec<FileRow>) {
    match node {
        FileNode::Dir {
            name,
            path,
            children,
            expanded,
            size,
            files,
        } => {
            out.push(FileRow {
                depth,
                name: name.clone(),
                path: path.clone(),
                size: *size,
                is_dir: true,
                expanded: *expanded,
                files: *files,
                kind: FileKind::Other,
            });
            if *expanded {
                for child in children {
                    flatten_node(child, depth + 1, out);
                }
            }
        }
        FileNode::File {
            name,
            path,
            size,
            kind,
        } => out.push(FileRow {
            depth,
            name: name.clone(),
            path: path.clone(),
            size: *size,
            is_dir: false,
            expanded: false,
            files: 0,
            kind: *kind,
        }),
    }
}

/// Toggle the expanded state of the directory at flattened index `idx` (in the
/// same visit order as [`flatten`]). Returns whether a directory was toggled.
pub fn toggle_by_index(root: &mut FileNode, idx: usize) -> bool {
    let mut cur = 0usize;
    toggle_walk(root, idx, &mut cur)
}

fn toggle_walk(node: &mut FileNode, target: usize, cur: &mut usize) -> bool {
    let here = *cur;
    *cur += 1;
    if let FileNode::Dir {
        children, expanded, ..
    } = node
    {
        if here == target {
            *expanded = !*expanded;
            return true;
        }
        if *expanded {
            for child in children.iter_mut() {
                if toggle_walk(child, target, cur) {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_classifies_by_extension_and_known_names() {
        assert_eq!(
            FileKind::of("model-00001-of-2.safetensors"),
            FileKind::Checkpoint
        );
        assert_eq!(FileKind::of("weights.gguf"), FileKind::Checkpoint);
        assert_eq!(FileKind::of("config.json"), FileKind::Json);
        assert_eq!(FileKind::of("tokenizer_config.json"), FileKind::Json);
        assert_eq!(FileKind::of("README"), FileKind::Text);
        assert_eq!(FileKind::of("LICENSE"), FileKind::Text);
        assert_eq!(FileKind::of("notes.md"), FileKind::Text);
        assert_eq!(FileKind::of("tool_parser.py"), FileKind::Text);
        assert_eq!(FileKind::of("mystery.bin"), FileKind::Other);
    }

    #[test]
    fn flatten_hides_collapsed_subtrees_and_toggle_reveals_them() {
        let dir = std::env::temp_dir().join("ce_filetree_flatten_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("sub/a.json"), b"{}").unwrap();
        std::fs::write(dir.join("top.json"), b"{}").unwrap();

        let mut root = build(&dir, 8);
        // `sub` starts collapsed, so its child is hidden; root + sub + top.json.
        let rows = flatten(&root);
        let names: Vec<&str> = rows.iter().map(|r| r.name.as_str()).collect();
        assert!(names.contains(&"sub") && names.contains(&"top.json"));
        assert!(
            !names.contains(&"a.json"),
            "collapsed subtree hidden: {names:?}"
        );

        // Toggle `sub` (index 1: root is 0) → its child appears.
        assert!(toggle_by_index(&mut root, 1));
        assert!(
            flatten(&root).iter().any(|r| r.name == "a.json"),
            "expanded subtree shows its child"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn build_sorts_dirs_first_and_sums_sizes() {
        let dir = std::env::temp_dir().join("ce_filetree_build_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap(); // 2 bytes
        std::fs::write(dir.join("model.safetensors"), vec![0u8; 100]).unwrap();
        std::fs::write(dir.join("sub/extra.json"), vec![0u8; 8]).unwrap();
        std::fs::write(dir.join(".hidden"), b"x").unwrap(); // skipped

        let root = build(&dir, 8);
        let FileNode::Dir {
            children,
            size,
            files,
            ..
        } = &root
        else {
            panic!("root is a dir");
        };
        // Directory ("sub") sorts before the files.
        assert!(matches!(&children[0], FileNode::Dir { name, .. } if name == "sub"));
        // Files after, natural-sorted, dotfile skipped.
        let names: Vec<&str> = children.iter().map(FileNode::name).collect();
        assert_eq!(names, ["sub", "config.json", "model.safetensors"]);
        // Aggregate size = 2 + 100 + 8 (sub) = 110; 3 files counted.
        assert_eq!(*size, 110);
        assert_eq!(*files, 3);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
