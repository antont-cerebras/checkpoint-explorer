//! The file-browser tree: a checkpoint's directory shown as a hierarchy of
//! directories and files (the `Tab` file view). Kept separate from the tensor
//! [`crate::tree::TreeNode`] so the mature tensor paths stay untouched — this
//! models only what a file explorer needs (name, path, size, kind, expansion).

use std::path::{Path, PathBuf};

use crate::tree::natural_sort_key;

/// What a file is, for its glyph and what `Enter` does with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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

/// One directory entry from a listing backend (local `std::fs`, remote SFTP, …).
/// The backend is responsible for skipping dotfiles and for the two invariants the
/// browser relies on:
/// - `size` is the entry's **readable content size** — symlinks are *followed*, so
///   a linked shard reports its target's size (what opening it yields, and what
///   the layout map reads), never the link-path length. `0` for a directory.
/// - `is_dir` is whether it's a **descendable** directory. A symlink to a
///   directory must be `false` (a leaf) so the recursive walk can't cycle.
#[derive(Debug, Clone)]
pub struct DirEntry {
    pub name: String,
    pub size: u64,
    pub is_dir: bool,
}

/// Build the file tree rooted at `root` from the **local filesystem**, recursively
/// (bounded by `max_depth`). Directories sort first, then files, each in natural
/// order; hidden entries (dotfiles) are skipped. The returned node is the root
/// directory itself, expanded. Symlinks are followed for a file's size (so a linked
/// shard shows its target's size) but a symlinked directory is a leaf, not
/// descended, to avoid cycles — see [`DirEntry`].
pub fn build(root: &Path, max_depth: usize) -> FileNode {
    build_from(&local_list, root, max_depth)
}

/// Build the file tree from a pluggable directory-listing backend `list` — so the
/// same tree (and everything downstream: [`flatten`], [`toggle_by_index`],
/// `FileRow`, the renderer) works for a local walk or a remote SFTP `readdir`.
/// `list(dir)` returns that directory's entries (already dotfile-filtered); the
/// child path is `dir.join(name)`, so remote paths compose the same way.
pub fn build_from(
    list: &dyn Fn(&Path) -> Vec<DirEntry>,
    root: &Path,
    max_depth: usize,
) -> FileNode {
    let mut node = build_dir(list, root, root_name(root), max_depth);
    // The root is expanded so its contents show immediately; nested dirs start
    // collapsed (a checkpoint is usually flat, so this rarely matters).
    if let FileNode::Dir { expanded, .. } = &mut node {
        *expanded = true;
    }
    node
}

/// Build an **s3-native** browse tree from a flat object listing (prefix-relative
/// `(key, size)` pairs): each key is split on `/`, shared prefixes become
/// expandable directories and the leaves are the objects; sizes and file counts
/// aggregate bottom-up, exactly like the local/SFTP walk (it reuses
/// [`build_from`] over a synthetic in-memory listing). `root_label` is the root
/// node's display name; every other node's `path` is its **exact prefix-relative
/// key** (`a/b/c` for the object `a/b/c`), so the browser rebuilds the full
/// `s3://…` URI as `{uri}/{path}`. Browse-only — no per-object layout or preview.
pub fn build_from_keys(root_label: &str, objects: &[(String, u64)]) -> FileNode {
    use std::collections::{HashMap, HashSet};
    // Directory (a relative path; "" is the root) → its immediate entries.
    // Intermediate dirs are materialized once (`dir_seen`); files carry their
    // size, dirs 0 (aggregated by `build_from`). Rooting at "" makes each node's
    // composed path (`dir.join(name)`) equal its exact prefix-relative key.
    let mut listing: HashMap<PathBuf, Vec<DirEntry>> = HashMap::new();
    let mut dir_seen: HashSet<PathBuf> = HashSet::new();
    for (key, size) in objects {
        let comps: Vec<&str> = key.split('/').filter(|s| !s.is_empty()).collect();
        let Some((leaf, dirs)) = comps.split_last() else {
            continue;
        };
        let mut cur = PathBuf::new();
        for comp in dirs {
            let child = cur.join(comp);
            if dir_seen.insert(child.clone()) {
                listing.entry(cur.clone()).or_default().push(DirEntry {
                    name: (*comp).to_string(),
                    size: 0,
                    is_dir: true,
                });
            }
            cur = child;
        }
        listing.entry(cur).or_default().push(DirEntry {
            name: (*leaf).to_string(),
            size: *size,
            is_dir: false,
        });
    }
    let list = move |p: &Path| listing.get(p).cloned().unwrap_or_default();
    // Depth generous enough for any realistic object-key nesting.
    let mut node = build_from(&list, Path::new(""), 64);
    if let FileNode::Dir { name, .. } = &mut node {
        *name = root_label.to_string();
    }
    node
}

/// The local-filesystem listing backend for [`build`].
fn local_list(dir: &Path) -> Vec<DirEntry> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = match entry.file_name().to_str() {
                Some(n) if !n.starts_with('.') => n.to_string(),
                _ => continue, // unreadable name or a dotfile
            };
            // Follow symlinks (`fs::metadata`, not the link's own `entry.metadata`)
            // so a symlinked shard shows the *target's* real size — what opening it
            // reads — not the ~link-path length. A *real* directory descends; a
            // symlinked directory stays a leaf so the walk can't cycle. A broken
            // link falls back to its own (lstat) metadata.
            let is_symlink = entry.file_type().ok().is_some_and(|t| t.is_symlink());
            let meta = std::fs::metadata(entry.path()).or_else(|_| entry.metadata());
            let (is_dir, size) = match meta {
                Ok(m) if m.is_dir() => (!is_symlink, 0),
                Ok(m) => (false, m.len()),
                Err(_) => (false, 0),
            };
            out.push(DirEntry { name, size, is_dir });
        }
    }
    out
}

/// The label for the root directory node — its final component, or the whole
/// path when it has none (e.g. `/`).
fn root_name(root: &Path) -> String {
    root.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| root.to_string_lossy().into_owned())
}

fn build_dir(
    list: &dyn Fn(&Path) -> Vec<DirEntry>,
    dir: &Path,
    name: String,
    depth_left: usize,
) -> FileNode {
    let mut dirs: Vec<FileNode> = Vec::new();
    let mut files: Vec<FileNode> = Vec::new();
    for entry in list(dir) {
        let path = dir.join(&entry.name);
        if entry.is_dir && depth_left > 0 {
            dirs.push(build_dir(list, &path, entry.name, depth_left - 1));
        } else if entry.is_dir {
            // Depth limit reached: represent as an empty (unexpanded) dir.
            dirs.push(FileNode::Dir {
                name: entry.name,
                path,
                children: Vec::new(),
                expanded: false,
                size: 0,
                files: 0,
            });
        } else {
            let kind = FileKind::of(&entry.name);
            files.push(FileNode::File {
                name: entry.name,
                path,
                size: entry.size,
                kind,
            });
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

    #[test]
    fn build_from_walks_a_synthetic_listing() {
        // A remote-like listing backend keyed by directory path — no filesystem.
        let list = |dir: &Path| -> Vec<DirEntry> {
            match dir.to_str().unwrap() {
                "/ckpt" => vec![
                    DirEntry {
                        name: "model.safetensors".into(),
                        size: 100,
                        is_dir: false,
                    },
                    DirEntry {
                        name: "sub".into(),
                        size: 0,
                        is_dir: true,
                    },
                    DirEntry {
                        name: "config.json".into(),
                        size: 2,
                        is_dir: false,
                    },
                ],
                "/ckpt/sub" => vec![
                    DirEntry {
                        name: "extra.json".into(),
                        size: 8,
                        is_dir: false,
                    },
                    DirEntry {
                        name: "deep".into(),
                        size: 0,
                        is_dir: true,
                    },
                ],
                "/ckpt/sub/deep" => vec![DirEntry {
                    name: "leaf.bin".into(),
                    size: 4,
                    is_dir: false,
                }],
                _ => Vec::new(),
            }
        };

        let root = build_from(&list, Path::new("/ckpt"), 8);
        let FileNode::Dir {
            children,
            size,
            files,
            expanded,
            ..
        } = &root
        else {
            panic!("root is a dir");
        };
        assert!(*expanded, "root is expanded");
        // Dirs first, then files natural-sorted; child paths compose from the parent.
        let names: Vec<&str> = children.iter().map(FileNode::name).collect();
        assert_eq!(names, ["sub", "config.json", "model.safetensors"]);
        assert!(matches!(&children[0], FileNode::Dir { path, expanded, .. }
            if path == Path::new("/ckpt/sub") && !*expanded));
        // Bottom-up aggregation: 100 + 2 + (8 + 4) = 114; 4 files across all depths.
        assert_eq!(*size, 114);
        assert_eq!(*files, 4);

        // Depth cap: at depth 1, `sub` is entered but `deep` is a stubbed empty dir.
        let shallow = build_from(&list, Path::new("/ckpt"), 1);
        let FileNode::Dir { children, .. } = &shallow else {
            panic!("root is a dir");
        };
        let FileNode::Dir {
            children: sub_children,
            ..
        } = &children[0]
        else {
            panic!("sub is a dir");
        };
        let deep = sub_children
            .iter()
            .find(|c| c.name() == "deep")
            .expect("deep dir present");
        assert!(matches!(deep, FileNode::Dir { children, .. } if children.is_empty()));
    }

    #[cfg(unix)]
    #[test]
    fn build_follows_symlinked_files_to_their_real_size() {
        // A blob-dedup checkpoint: the shard in the checkpoint dir is a *symlink* to
        // the real file living elsewhere. The browser must show the target's size —
        // so it agrees with the layout map, which opens (follows) the file. A
        // directory listing's own (lstat) size would be the ~link-path length.
        use std::os::unix::fs::symlink;
        let base = std::env::temp_dir().join("ce_filetree_symlink_test");
        let _ = std::fs::remove_dir_all(&base);
        let ckpt = base.join("ckpt");
        let store = base.join("store");
        std::fs::create_dir_all(&ckpt).unwrap();
        std::fs::create_dir_all(&store).unwrap();
        std::fs::write(store.join("blob"), vec![0u8; 4096]).unwrap();
        symlink(store.join("blob"), ckpt.join("model-00000.safetensors")).unwrap();
        // A symlinked *directory* must stay a leaf (no descent → no cycle).
        symlink(&store, ckpt.join("linkdir")).unwrap();

        let root = build(&ckpt, 8);
        let FileNode::Dir { children, .. } = &root else {
            panic!("root is a dir");
        };
        let shard = children
            .iter()
            .find(|c| c.name() == "model-00000.safetensors")
            .expect("shard present");
        assert!(
            matches!(shard, FileNode::File { size, .. } if *size == 4096),
            "symlinked shard shows the target's real size, not the link length"
        );
        let linkdir = children.iter().find(|c| c.name() == "linkdir").unwrap();
        assert!(
            matches!(linkdir, FileNode::File { .. }),
            "a symlinked directory is a leaf (not descended)"
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn build_from_keys_makes_an_s3_native_tree() {
        // A flat object listing with a shared "layer_0/" prefix + a top-level file.
        let objects = vec![
            ("layer_0/weight".to_string(), 100u64),
            ("layer_0/bias".to_string(), 10u64),
            ("metadata.json".to_string(), 5u64),
        ];
        let root = build_from_keys("my-ckpt", &objects);
        let FileNode::Dir {
            name,
            children,
            size,
            files,
            expanded,
            ..
        } = &root
        else {
            panic!("root is a dir");
        };
        assert_eq!(name, "my-ckpt");
        assert!(*expanded);
        // Dir ("layer_0") sorts before the top-level file.
        let names: Vec<&str> = children.iter().map(FileNode::name).collect();
        assert_eq!(names, ["layer_0", "metadata.json"]);
        // Bottom-up aggregation: (100 + 10) + 5 = 115; 3 objects.
        assert_eq!(*size, 115);
        assert_eq!(*files, 3);

        // A nested object's `path` is its exact prefix-relative key (for URI rebuild).
        let FileNode::Dir {
            children: layer_children,
            ..
        } = &children[0]
        else {
            panic!("layer_0 is a dir");
        };
        let weight = layer_children
            .iter()
            .find(|c| c.name() == "weight")
            .expect("weight object present");
        assert!(
            matches!(weight, FileNode::File { path, .. } if path == Path::new("layer_0/weight"))
        );
    }
}
