//! The **kernel**: a frontend-agnostic session over a cached
//! [`crate::model::Checkpoint`]. It owns the browser state (which screen, the
//! selection, the search query), derives + caches the views (tensor tree, file
//! tree) and reports (stats) from the model, and exposes:
//!
//! - **command methods** — `select_next`/`select_prev`/`toggle`/`search`/… —
//!   that mutate the state, and
//! - a **query** — [`Session::view`] returning a serializable [`ViewModel`] of
//!   what's on screen (rows, selection, status).
//!
//! No terminal, no disk: it's driven by method calls and observed through the
//! `ViewModel`, so it's trivially unit-testable and the same session can back the
//! interactive terminal, a headless web server, or an MCP tool. The kernel reads
//! nothing from disk — everything comes from the model the readers already
//! cached.

use crate::model::Checkpoint;
use crate::stats::CheckpointStats;
use crate::tree::{TreeBuilder, TreeNode};

/// Which screen the session is showing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Screen {
    Tree,
    Files,
}

/// One visible row in the [`ViewModel`] — the frontend renders these; it doesn't
/// walk the tree itself.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Row {
    pub depth: usize,
    pub label: String,
    /// A group/directory (has children) vs a leaf (tensor / file).
    pub is_group: bool,
}

/// The serializable snapshot of what's on screen — the kernel's one output
/// contract, shared by every frontend (TUI renders it, a web server sends it as
/// JSON, an MCP tool returns it).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ViewModel {
    pub screen: Screen,
    /// The checkpoint root (the header line).
    pub root: String,
    pub rows: Vec<Row>,
    /// Index of the highlighted row in `rows`.
    pub selected: usize,
    /// The bottom status line (e.g. the selected row's path/summary).
    pub status: String,
    /// The active search query, if any.
    pub search: Option<String>,
}

/// The tensor-tree browser state — the tree itself, its flattened/filtered rows,
/// and the selection/scroll/search. Owned by the kernel (this is the state the
/// interactive tree screen drives and renders from); the operations currently
/// live on the frontend and mutate these fields directly, and will move onto this
/// type as the migration continues.
#[derive(Default)]
pub struct TreeState {
    /// The grouped tree (a single root node summarising the checkpoint).
    pub tree: Vec<TreeNode>,
    /// The tree flattened to visible rows `(node, depth)` (fold-aware).
    pub flattened: Vec<(TreeNode, usize)>,
    /// The search-filtered rows (used instead of `flattened` in search mode).
    pub filtered: Vec<(TreeNode, usize)>,
    /// Highlighted row index (into the visible tree).
    pub selected: usize,
    /// Viewport scroll offset.
    pub scroll: usize,
    /// The live search query.
    pub search_query: String,
    /// Caret position within `search_query` (character index).
    pub search_cursor: usize,
    /// Whether search input is active.
    pub search_mode: bool,
}

/// The file-browser state — the directory tree (built from the model / a remote
/// listing), its flattened visible rows, and the selection/scroll. Kernel-owned,
/// like [`TreeState`].
#[derive(Default)]
pub struct FileState {
    pub tree: Option<crate::filetree::FileNode>,
    pub rows: Vec<crate::filetree::FileRow>,
    pub selected: usize,
    pub scroll: usize,
}

/// A frontend-agnostic browsing session over a cached checkpoint.
pub struct Session {
    model: Checkpoint,
    /// The tensor tree flattened to visible rows, derived once from the model
    /// (no disk); search filters it on the fly.
    flat: Vec<(TreeNode, usize)>,
    screen: Screen,
    selected: usize,
    search: String,
    /// Cached stats report (computed on first request).
    stats: Option<CheckpointStats>,
}

impl Session {
    /// Open a session over an already-read checkpoint model.
    pub fn new(model: Checkpoint) -> Self {
        let tensors = model.tensors_vec();
        let metadata = model.metadata_vec();
        let tree = if metadata.is_empty() {
            TreeBuilder::build_tree(&tensors)
        } else {
            TreeBuilder::build_tree_mixed(&tensors, &metadata)
        };
        let flat = TreeBuilder::flatten_tree(&tree);
        Session {
            model,
            flat,
            screen: Screen::Tree,
            selected: 0,
            search: String::new(),
            stats: None,
        }
    }

    /// The underlying model (for serialization / reports / other queries).
    pub fn model(&self) -> &Checkpoint {
        &self.model
    }

    /// The checkpoint stats report — computed once from the model, then cached.
    pub fn stats(&mut self) -> &CheckpointStats {
        if self.stats.is_none() {
            let tensors = self.model.tensors_vec();
            self.stats = Some(CheckpointStats::compute(
                &tensors,
                self.model.config.as_ref(),
                self.model.disk_usage(),
            ));
        }
        self.stats.as_ref().expect("just set")
    }

    // ── Command methods (input) ──────────────────────────────────────────────

    /// Switch screen (tree ⇆ files); resets the selection.
    pub fn show(&mut self, screen: Screen) {
        self.screen = screen;
        self.selected = 0;
    }

    /// Move the selection by `delta` rows, clamped to the visible range.
    pub fn move_selection(&mut self, delta: isize) {
        let n = self.rows_len();
        if n == 0 {
            self.selected = 0;
            return;
        }
        let cur = self.selected as isize;
        self.selected = cur.saturating_add(delta).clamp(0, n as isize - 1) as usize;
    }

    pub fn select_next(&mut self) {
        self.move_selection(1);
    }
    pub fn select_prev(&mut self) {
        self.move_selection(-1);
    }

    /// Set the fuzzy search query (tree screen); empty clears it.
    pub fn search(&mut self, query: &str) {
        self.search = query.to_string();
        self.selected = 0;
    }

    // ── Query (output) ───────────────────────────────────────────────────────

    /// The number of currently visible rows (search-filtered on the tree screen).
    fn rows_len(&self) -> usize {
        match self.screen {
            Screen::Tree => self.tree_rows().count(),
            Screen::Files => self.model.files.len(),
        }
    }

    /// The tree rows matching the active search (all rows when the query is empty).
    fn tree_rows(&self) -> impl Iterator<Item = &(TreeNode, usize)> {
        let q = self.search.to_ascii_lowercase();
        self.flat
            .iter()
            .filter(move |(node, _)| q.is_empty() || node.name().to_ascii_lowercase().contains(&q))
    }

    /// A serializable snapshot of the current screen — the kernel's sole output.
    pub fn view(&self) -> ViewModel {
        let rows: Vec<Row> = match self.screen {
            Screen::Tree => self
                .tree_rows()
                .map(|(node, depth)| Row {
                    depth: *depth,
                    label: node.name().to_string(),
                    is_group: matches!(node, TreeNode::Group { .. }),
                })
                .collect(),
            Screen::Files => self
                .model
                .files
                .iter()
                .map(|f| Row {
                    depth: f.depth,
                    label: f.name.clone(),
                    is_group: f.is_dir,
                })
                .collect(),
        };
        let selected = self.selected.min(rows.len().saturating_sub(1));
        let status = rows
            .get(selected)
            .map(|r| r.label.clone())
            .unwrap_or_default();
        ViewModel {
            screen: self.screen,
            root: self.model.root.clone(),
            rows,
            selected,
            status,
            search: (!self.search.is_empty()).then(|| self.search.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{FileEntry, ShardHeader, Source};
    use crate::tree::{Layout, MetadataInfo, Storage, TensorInfo};

    fn model() -> Checkpoint {
        let ti = |name: &str| TensorInfo {
            name: name.into(),
            dtype: "F32".into(),
            shape: vec![2, 2],
            size_bytes: 16,
            num_elements: 4,
            storage: Storage::Unknown,
            source_path: "/ckpt/model.safetensors".into(),
            layout: Layout::ByteRange { start: 0, end: 16 },
        };
        Checkpoint {
            source: Source::Local,
            root: "/ckpt".into(),
            files: vec![
                FileEntry {
                    rel_path: "model.safetensors".into(),
                    name: "model.safetensors".into(),
                    depth: 0,
                    apparent: 100,
                    allocated: 512,
                    is_dir: false,
                    kind: crate::filetree::FileKind::Checkpoint,
                    symlink_target: None,
                    mode: None,
                    mtime: None,
                },
                FileEntry {
                    rel_path: "config.json".into(),
                    name: "config.json".into(),
                    depth: 0,
                    apparent: 20,
                    allocated: 512,
                    is_dir: false,
                    kind: crate::filetree::FileKind::Json,
                    symlink_target: None,
                    mode: None,
                    mtime: None,
                },
            ],
            shards: vec![ShardHeader {
                path: "/ckpt/model.safetensors".into(),
                total_len: 116,
                header_len: 100,
                tensors: vec![
                    ti("model.embed_tokens.weight"),
                    ti("model.layers.0.mlp.down_proj.weight"),
                ],
                metadata: vec![MetadataInfo {
                    name: "format".into(),
                    value: "pt".into(),
                    value_type: "string".into(),
                }],
            }],
            config: None,
            index: vec![],
            s3: None,
        }
    }

    #[test]
    fn drives_by_methods_and_yields_a_serializable_viewmodel() {
        let mut s = Session::new(model());
        // Tree screen: rows derived from the model, no disk.
        let v = s.view();
        assert_eq!(v.screen, Screen::Tree);
        assert_eq!(v.root, "/ckpt");
        assert!(!v.rows.is_empty());
        assert_eq!(v.selected, 0);

        // Navigation clamps.
        s.select_prev();
        assert_eq!(s.view().selected, 0);
        s.select_next();
        assert_eq!(s.view().selected, 1);

        // Search filters the rows.
        s.search("embed");
        let v = s.view();
        assert!(
            v.rows
                .iter()
                .all(|r| r.label.to_lowercase().contains("embed"))
        );
        assert_eq!(v.search.as_deref(), Some("embed"));
        s.search("");

        // Switch to the file screen (from the cached model's file list).
        s.show(Screen::Files);
        let v = s.view();
        assert_eq!(v.screen, Screen::Files);
        assert!(v.rows.iter().any(|r| r.label == "config.json"));

        // Stats derived + cached from the model.
        assert_eq!(s.stats().n_tensors, 2);

        // The whole ViewModel serializes (the frontends' output contract).
        let json = serde_json::to_string(&s.view()).unwrap();
        assert!(json.contains("\"screen\":\"files\""), "{json}");
        // And round-trips.
        let back: ViewModel = serde_json::from_str(&json).unwrap();
        assert_eq!(back.screen, Screen::Files);
    }
}
