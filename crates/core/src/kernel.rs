//! The **kernel**: the frontend-agnostic core over a cached
//! [`crate::model::Checkpoint`]. No terminal, no disk — everything comes from the
//! model the readers already cached, so it's trivially unit-testable and the same
//! kernel backs the interactive terminal, a headless web server, or an MCP tool.
//! It has three parts:
//!
//! - [`Session`] — the single owner of the checkpoint's **canonical data**
//!   (deduped + natural-sorted tensors, metadata, config, parameter count) and
//!   its cached reports (stats). A frontend loads the tree from it and asks it for
//!   reports.
//! - the **view-state + command surface** — [`TreeState`] / [`FileState`] /
//!   [`DataViewState`], the browser state a frontend owns and persists across
//!   loads, with the navigation/fold operations as methods (`move_selection`,
//!   `toggle_group_at`, `reveal`, …) the frontend drives.
//! - the **output contract** — [`ViewModel`], a serializable snapshot projected
//!   from the live view-state ([`ViewModel::from_tree`] / [`ViewModel::from_files`])
//!   that a TUI renders, a web server sends as JSON, or an MCP tool returns.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};

use crate::config::ModelConfig;
use crate::model::Checkpoint;
use crate::sample::ViewDtype;
use crate::stats::CheckpointStats;
use crate::tree::{MetadataInfo, TensorInfo, TreeBuilder, TreeNode, natural_sort_key};
use crate::viewstate::{DataLayout, NumBase, StripeMode};

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
/// and the selection/scroll/search. Kernel-owned, with its navigation + fold
/// operations as methods on this type (`move_selection`, `move_to_*`,
/// `toggle_group_at`, `set_all_expanded`, `reveal`, `reflatten`) — the tree
/// screen's command surface. A frontend drives those methods and renders the
/// resulting fields; the search-filter refresh stays on the frontend only because
/// it needs the tensor list (which the [`Session`] owns).
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

impl TreeState {
    /// The currently visible rows: the search results while searching, else the
    /// fold-aware flattened tree. The one selector every navigation op reads.
    pub fn visible(&self) -> &[(TreeNode, usize)] {
        if self.search_mode {
            &self.filtered
        } else {
            &self.flattened
        }
    }

    /// Rebuild the flattened rows from the (possibly re-folded) tree. The pure
    /// half of a re-flatten; the search-filtered rows refresh separately (they
    /// need the tensor list) only when the query changes.
    pub fn reflatten(&mut self) {
        self.flattened = TreeBuilder::flatten_tree(&self.tree);
    }

    /// Expand/collapse the group at flattened index `idx` in place and re-flatten.
    /// Toggles the tree directly rather than cloning it first — that full
    /// deep-clone made every expand/collapse lag on a big checkpoint.
    pub fn toggle_group_at(&mut self, idx: usize) {
        TreeBuilder::toggle_node_by_index(idx, &mut self.tree);
        self.reflatten();
    }

    /// Expand or collapse every group, then reset the cursor to the top since the
    /// visible rows change wholesale.
    pub fn set_all_expanded(&mut self, expanded: bool) {
        TreeBuilder::set_all_expanded(&mut self.tree, expanded);
        self.reflatten();
        self.selected = 0;
        self.scroll = 0;
    }

    /// Move the cursor by `delta` rows within the visible list, clamped.
    pub fn move_selection(&mut self, delta: i32) {
        let n = self.visible().len();
        if n == 0 {
            return;
        }
        self.selected = if delta < 0 {
            self.selected.saturating_sub((-delta) as usize)
        } else {
            (self.selected + delta as usize).min(n - 1)
        };
    }

    /// Move the cursor to the parent group of the selected row (the nearest
    /// preceding row at a shallower depth). No-op at the top level.
    pub fn move_to_parent(&mut self) {
        let Some(&(_, depth)) = self.visible().get(self.selected) else {
            return;
        };
        if depth == 0 {
            return;
        }
        let parent = (0..self.selected)
            .rev()
            .find(|&i| self.visible()[i].1 < depth);
        if let Some(p) = parent {
            self.selected = p;
        }
    }

    /// Move the cursor to the next/previous sibling: the nearest row at the same
    /// depth before a shallower row (i.e. without leaving the parent).
    pub fn move_to_sibling(&mut self, forward: bool) {
        let Some(&(_, depth)) = self.visible().get(self.selected) else {
            return;
        };
        let indices: Vec<usize> = if forward {
            (self.selected + 1..self.visible().len()).collect()
        } else {
            (0..self.selected).rev().collect()
        };
        for i in indices {
            let d = self.visible()[i].1;
            if d < depth {
                break; // left the parent: no sibling in this direction
            }
            if d == depth {
                self.selected = i;
                break;
            }
            // d > depth: a descendant, keep scanning
        }
    }

    /// Enter the selected group: expand it if collapsed, then move the cursor to
    /// its first child. No-op for leaf rows or empty groups (and in search mode,
    /// where the list is flat).
    pub fn move_to_first_child(&mut self) {
        if self.search_mode {
            return;
        }
        let (expanded, has_children, depth) = match self.flattened.get(self.selected) {
            Some((
                TreeNode::Group {
                    expanded, children, ..
                },
                depth,
            )) => (*expanded, !children.is_empty(), *depth),
            _ => return,
        };
        if !has_children {
            return;
        }
        if !expanded {
            self.toggle_group_at(self.selected);
        }
        // The first child is the next row, one level deeper.
        if let Some((_, child_depth)) = self.flattened.get(self.selected + 1)
            && *child_depth == depth + 1
        {
            self.selected += 1;
        }
    }

    /// Move the cursor onto the leaf named `name`, expanding any collapsed groups
    /// so it's visible. Fast path when the row is already on screen (returning to
    /// an expanded tree — the common case): just move the cursor, no rebuild. In
    /// search mode the list is flat, so an absent name leaves the cursor put.
    pub fn reveal(&mut self, name: &str) {
        if let Some(idx) = self
            .visible()
            .iter()
            .position(|(node, _)| node.name() == name)
        {
            self.selected = idx;
            return;
        }
        if !self.search_mode {
            TreeBuilder::expand_to_tensor(&mut self.tree, name);
            self.reflatten();
            if let Some(idx) = self.flattened.iter().position(|(n, _)| n.name() == name) {
                self.selected = idx;
            }
        }
    }
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

impl FileState {
    /// Re-flatten the directory tree into visible rows, clamping the selection.
    pub fn rebuild_rows(&mut self) {
        self.rows = self
            .tree
            .as_ref()
            .map(crate::filetree::flatten)
            .unwrap_or_default();
        let n = self.rows.len();
        self.selected = self.selected.min(n.saturating_sub(1));
    }

    /// Move the cursor by `delta` rows within the file list, clamped.
    pub fn move_selection(&mut self, delta: i32) {
        let len = self.rows.len();
        if len == 0 {
            return;
        }
        self.selected = if delta < 0 {
            self.selected.saturating_sub((-delta) as usize)
        } else {
            (self.selected + delta as usize).min(len - 1)
        };
    }

    /// Expand/collapse the directory at row `idx` and re-flatten.
    pub fn toggle_dir(&mut self, idx: usize) {
        if let Some(tree) = self.tree.as_mut() {
            crate::filetree::toggle_by_index(tree, idx);
        }
        self.rebuild_rows();
    }

    /// `←`: collapse the selected directory if it's open, else jump to its parent.
    pub fn collapse_or_parent(&mut self) {
        let Some((is_dir, expanded, depth)) = self
            .rows
            .get(self.selected)
            .map(|r| (r.is_dir(), r.expanded(), r.depth))
        else {
            return;
        };
        if is_dir && expanded {
            self.toggle_dir(self.selected);
            return;
        }
        if depth == 0 {
            return;
        }
        if let Some(parent) = (0..self.selected)
            .rev()
            .find(|&i| self.rows[i].depth < depth)
        {
            self.selected = parent;
        }
    }

    /// `→`: expand the selected directory if it's collapsed (a no-op otherwise).
    pub fn expand_or_child(&mut self) {
        let Some((is_dir, expanded)) = self
            .rows
            .get(self.selected)
            .map(|r| (r.is_dir(), r.expanded()))
        else {
            return;
        };
        if is_dir && !expanded {
            self.toggle_dir(self.selected);
        }
    }
}

/// The **data-view presentation state** — the session-remembered toggles that
/// control how the numeric grid / heatmap / histogram screens render a tensor:
/// the per-tensor dtype/shape reinterpretations, the histogram bucket count, the
/// layout (overview / edges / window) and its edge-split / window-offset knobs,
/// and the zebra-striping + numeral-base choices.
///
/// Kernel-owned, like [`TreeState`]/[`FileState`]. The fields keep interior
/// mutability (`Cell`/`RefCell`) because the TUI mutates them from within `&self`
/// renders; a `&mut self` command surface and serde (for `y` / JSON) come when
/// the data-view screens migrate onto the kernel. These toggles already
/// round-trip through the argv-based `--emit-command` path today.
pub struct DataViewState {
    /// Per-tensor dtype reinterpretation chosen in the data views, keyed by
    /// tensor name. Session-scoped: remembered until the app exits.
    pub dtype_overrides: RefCell<HashMap<String, ViewDtype>>,
    /// Per-tensor shape override (a reshape with the same element count) chosen
    /// in the data views with `r`, keyed by tensor name. Session-scoped.
    pub shape_overrides: RefCell<HashMap<String, Vec<usize>>>,
    /// Requested histogram bucket count (the `b` key / `--bins`); `None` lets the
    /// layout pick automatically. Session-wide, like the other view toggles.
    pub histogram_bins: Cell<Option<usize>>,
    /// Which layout the data views use (overview / edges / window). Session-
    /// scoped: remembered as you move between tensors and in/out of the preview.
    pub data_view_layout: Cell<DataLayout>,
    /// In the edges view, how the fixed row/column budget is split between the
    /// first (head) and last (tail) indices: `0.0` shows only the first, `1.0`
    /// only the last, `0.5` is balanced. Adjustable with the arrow keys.
    pub data_view_row_tail: Cell<f32>,
    pub data_view_col_tail: Cell<f32>,
    /// The last edges-view row/column budgets actually rendered, so an arrow
    /// press can move the divider by exactly one index (step = 1 / budget).
    pub edge_row_budget: Cell<usize>,
    pub edge_col_budget: Cell<usize>,
    /// The window view's top-left corner (row/column offset into the matrix).
    /// Clamped to a valid position on every draw (read back from the rendered
    /// sample), so panning behaves at the edges. Session-remembered.
    pub data_view_win_row: Cell<usize>,
    pub data_view_win_col: Cell<usize>,
    /// The last window's visible size (rows/cols actually shown), so a
    /// `Shift`+arrow press can stride by one screenful.
    pub win_page_rows: Cell<usize>,
    pub win_page_cols: Cell<usize>,
    /// The numeric grid's zebra striping (rows / columns / off). Session-
    /// remembered; cycled with `z`.
    pub data_view_stripe: Cell<StripeMode>,
    /// The numeric grid's numeral base (dec / hex / oct / bin). Session-
    /// remembered; cycled with `b`.
    pub data_view_base: Cell<NumBase>,
}

impl Default for DataViewState {
    fn default() -> Self {
        DataViewState {
            dtype_overrides: RefCell::new(HashMap::new()),
            shape_overrides: RefCell::new(HashMap::new()),
            histogram_bins: Cell::new(None),
            data_view_layout: Cell::new(DataLayout::default()),
            data_view_row_tail: Cell::new(0.5),
            data_view_col_tail: Cell::new(0.5),
            edge_row_budget: Cell::new(1),
            edge_col_budget: Cell::new(1),
            data_view_win_row: Cell::new(0),
            data_view_win_col: Cell::new(0),
            win_page_rows: Cell::new(1),
            win_page_cols: Cell::new(1),
            data_view_stripe: Cell::new(StripeMode::default()),
            data_view_base: Cell::new(NumBase::default()),
        }
    }
}

/// A frontend-agnostic browsing session over a cached checkpoint.
///
/// The session is the single owner of the checkpoint's **canonical** primary
/// data — the tensors deduplicated by name (first shard wins) and natural-sorted,
/// the metadata, and the config — so a frontend never keeps its own copy that can
/// drift. For a local checkpoint it's built from the serializable
/// [`Checkpoint`] model ([`Session::from_model`]); a remote (`--ssh-read`) read
/// that hasn't produced a model yet supplies the parts directly
/// ([`Session::from_parts`]).
pub struct Session {
    /// The serializable model — `Some` for a local read; `None` for a remote read
    /// whose model isn't assembled yet (its tensors/metadata still populate the
    /// canonical fields below).
    model: Option<Checkpoint>,
    /// Canonical tensor list: deduplicated by name (first occurrence in shard
    /// order wins) then natural-sorted. The one primary tensor list every view /
    /// report / status line reads.
    tensors: Vec<TensorInfo>,
    /// The metadata entries, in model / shard order.
    metadata: Vec<MetadataInfo>,
    /// The checkpoint's `config.json`, when present.
    config: Option<ModelConfig>,
    /// Total element count across the canonical tensors (parameter count).
    total_parameters: usize,
    /// Cached stats report (computed on first request).
    stats: Option<CheckpointStats>,
}

impl Session {
    /// Open a session over an already-read checkpoint model (local reads).
    pub fn from_model(model: Checkpoint) -> Self {
        let tensors = model.tensors_vec();
        let metadata = model.metadata_vec();
        let config = model.config.clone();
        Self::assemble(Some(model), tensors, metadata, config)
    }

    /// Open a session from raw parts — a remote read whose serializable model
    /// isn't assembled yet. Canonicalises the tensors exactly as [`from_model`].
    pub fn from_parts(
        tensors: Vec<TensorInfo>,
        metadata: Vec<MetadataInfo>,
        config: Option<ModelConfig>,
    ) -> Self {
        Self::assemble(None, tensors, metadata, config)
    }

    /// Shared construction: canonicalise the tensors (dedup by name, natural-sort),
    /// build the tree, and cache the parameter count.
    fn assemble(
        model: Option<Checkpoint>,
        tensors: Vec<TensorInfo>,
        metadata: Vec<MetadataInfo>,
        config: Option<ModelConfig>,
    ) -> Self {
        let tensors = Self::canonical_tensors(tensors);
        let total_parameters = tensors.iter().map(|t| t.num_elements).sum();
        Session {
            model,
            tensors,
            metadata,
            config,
            total_parameters,
            stats: None,
        }
    }

    /// Build the initial tensor tree (fold-aware) from the canonical data — the
    /// starting point a frontend loads into its [`TreeState`]. No disk.
    pub fn build_tree(&self) -> Vec<TreeNode> {
        if self.metadata.is_empty() {
            TreeBuilder::build_tree(&self.tensors)
        } else {
            TreeBuilder::build_tree_mixed(&self.tensors, &self.metadata)
        }
    }

    /// Deduplicate tensors by name (keeping the first occurrence in shard order,
    /// so a name in two shards is resolved to the first) and natural-sort them —
    /// the canonical order the tree / stats / diff all consume.
    fn canonical_tensors(mut tensors: Vec<TensorInfo>) -> Vec<TensorInfo> {
        let mut seen = HashSet::new();
        tensors.retain(|t| seen.insert(t.name.clone()));
        // `sort_by_cached_key`, not `sort_by_key`: the natural-sort key allocates a
        // `Vec`, and `sort_by_key` would recompute it O(n log n) times.
        tensors.sort_by_cached_key(|a| natural_sort_key(&a.name));
        tensors
    }

    /// Drop the tensors and metadata whose names don't pass `keep`, recompute the
    /// parameter count, and invalidate the cached stats — backs the
    /// `--print-tree`/`--print-tensors` name-filter subset. The frontend rebuilds
    /// its tree afterwards.
    pub fn retain_named<F: FnMut(&str) -> bool>(&mut self, mut keep: F) {
        self.tensors.retain(|t| keep(&t.name));
        self.metadata.retain(|m| keep(&m.name));
        self.total_parameters = self.tensors.iter().map(|t| t.num_elements).sum();
        self.stats = None;
    }

    /// The serializable model (local reads only), for serialization / reports.
    pub fn model(&self) -> Option<&Checkpoint> {
        self.model.as_ref()
    }

    /// The canonical tensor list (deduped + natural-sorted).
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// The metadata entries (model / shard order).
    pub fn metadata(&self) -> &[MetadataInfo] {
        &self.metadata
    }

    /// The checkpoint's config, when present.
    pub fn config(&self) -> Option<&ModelConfig> {
        self.config.as_ref()
    }

    /// Total element count across the canonical tensors.
    pub fn total_parameters(&self) -> usize {
        self.total_parameters
    }

    /// The checkpoint stats report — computed once from the canonical data, cached.
    /// `disk` is the on-disk footprint the caller resolves (from the model for a
    /// local read, or the captured remote usage for `--ssh-read`).
    pub fn stats_with_disk(&mut self, disk: Option<crate::stats::DiskUsage>) -> &CheckpointStats {
        if self.stats.is_none() {
            self.stats = Some(CheckpointStats::compute(
                &self.tensors,
                self.config.as_ref(),
                disk,
            ));
        }
        self.stats.as_ref().expect("just set")
    }

    /// The checkpoint stats report, using the model's own disk usage (local reads).
    pub fn stats(&mut self) -> &CheckpointStats {
        let disk = self.model.as_ref().and_then(|m| m.disk_usage());
        self.stats_with_disk(disk)
    }
}

impl ViewModel {
    /// Project the tensor-tree screen into a serializable snapshot: the visible
    /// (fold-aware, search-filtered) rows, the selection, and the search query,
    /// straight from the frontend's live [`TreeState`]. Pure — the kernel's output
    /// contract that a TUI renders, a web server sends as JSON, or an MCP tool
    /// returns.
    pub fn from_tree(root: &str, tree: &TreeState) -> Self {
        let rows: Vec<Row> = tree
            .visible()
            .iter()
            .map(|(node, depth)| Row {
                depth: *depth,
                label: node.name().to_string(),
                is_group: matches!(node, TreeNode::Group { .. }),
            })
            .collect();
        let selected = tree.selected.min(rows.len().saturating_sub(1));
        let status = rows
            .get(selected)
            .map(|r| r.label.clone())
            .unwrap_or_default();
        ViewModel {
            screen: Screen::Tree,
            root: root.to_string(),
            rows,
            selected,
            status,
            search: tree
                .search_mode
                .then(|| tree.search_query.clone())
                .filter(|q| !q.is_empty()),
        }
    }

    /// Project the file-browser screen into a serializable snapshot from the
    /// frontend's live [`FileState`].
    pub fn from_files(root: &str, files: &FileState) -> Self {
        let rows: Vec<Row> = files
            .rows
            .iter()
            .map(|r| Row {
                depth: r.depth,
                label: r.name.clone(),
                is_group: r.is_dir(),
            })
            .collect();
        let selected = files.selected.min(rows.len().saturating_sub(1));
        let status = rows
            .get(selected)
            .map(|r| r.label.clone())
            .unwrap_or_default();
        ViewModel {
            screen: Screen::Files,
            root: root.to_string(),
            rows,
            selected,
            status,
            search: None,
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
                    mode: None,
                    mtime: None,
                    node: crate::model::FsNode::File {
                        apparent: 100,
                        allocated: 512,
                        kind: crate::filetree::FileKind::Checkpoint,
                    },
                },
                FileEntry {
                    rel_path: "config.json".into(),
                    name: "config.json".into(),
                    depth: 0,
                    mode: None,
                    mtime: None,
                    node: crate::model::FsNode::File {
                        apparent: 20,
                        allocated: 512,
                        kind: crate::filetree::FileKind::Json,
                    },
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
    fn session_owns_data_and_viewmodel_projects_the_live_state() {
        let mut s = Session::from_model(model());
        // The session owns the canonical data + reports (no disk).
        assert_eq!(s.tensors().len(), 2);
        assert_eq!(s.stats().n_tensors, 2);

        // A frontend loads the session's tree into its own TreeState and drives it.
        let mut tree = TreeState {
            tree: s.build_tree(),
            ..Default::default()
        };
        tree.reflatten();
        tree.set_all_expanded(true);

        // The ViewModel is a projection of that live view-state.
        let vm = ViewModel::from_tree(s.model().unwrap().root.as_str(), &tree);
        assert_eq!(vm.screen, Screen::Tree);
        assert_eq!(vm.root, "/ckpt");
        assert!(!vm.rows.is_empty());
        assert_eq!(vm.selected, 0);
        assert!(vm.search.is_none());

        // Search state flows through the projection.
        tree.search_mode = true;
        tree.search_query = "q".into();
        tree.filtered = tree.flattened.iter().take(1).cloned().collect();
        let vm = ViewModel::from_tree("/ckpt", &tree);
        assert_eq!(vm.search.as_deref(), Some("q"));
        assert_eq!(vm.rows.len(), 1);
        // An empty query projects no search even in search mode.
        tree.search_query.clear();
        assert!(ViewModel::from_tree("/ckpt", &tree).search.is_none());

        // The file screen projects from a FileState the same way.
        let files = FileState {
            rows: vec![crate::filetree::FileRow {
                depth: 0,
                name: "config.json".into(),
                path: "/ckpt/config.json".into(),
                size: 20,
                kind: crate::filetree::FileRowKind::File {
                    kind: crate::filetree::FileKind::Json,
                },
            }],
            ..Default::default()
        };
        let fvm = ViewModel::from_files("/ckpt", &files);
        assert_eq!(fvm.screen, Screen::Files);
        assert_eq!(fvm.rows[0].label, "config.json");

        // The ViewModel serializes (the frontends' output contract) and round-trips.
        let json = serde_json::to_string(&fvm).unwrap();
        assert!(json.contains("\"screen\":\"files\""), "{json}");
        let back: ViewModel = serde_json::from_str(&json).unwrap();
        assert_eq!(back.screen, Screen::Files);
    }

    #[test]
    fn tree_state_ops_navigate_fold_and_reveal() {
        let ti = |name: &str| TensorInfo {
            name: name.into(),
            dtype: "F32".into(),
            shape: vec![2, 2],
            size_bytes: 16,
            num_elements: 4,
            storage: Storage::Unknown,
            source_path: "/x.safetensors".into(),
            layout: Layout::ByteRange { start: 0, end: 16 },
        };
        // Two nested groups (blk.0.{a,b}, blk.1.{a,b}).
        let tensors = vec![ti("blk.0.a"), ti("blk.0.b"), ti("blk.1.a"), ti("blk.1.b")];
        let mut ts = TreeState {
            tree: TreeBuilder::build_tree(&tensors),
            ..Default::default()
        };
        ts.reflatten();

        // Expand everything; selection clamps to the visible range.
        ts.set_all_expanded(true);
        let n = ts.visible().len();
        assert!(n > 0);
        ts.move_selection(1000);
        assert_eq!(ts.selected, n - 1);
        ts.move_selection(-1000);
        assert_eq!(ts.selected, 0);

        // Reveal an already-visible leaf: the cursor just moves onto it.
        ts.reveal("blk.1.b");
        assert_eq!(ts.visible()[ts.selected].0.name(), "blk.1.b");

        // Collapse-all resets the cursor and hides the leaves; reveal re-expands
        // to the target and grows the visible list.
        ts.set_all_expanded(false);
        assert_eq!(ts.selected, 0);
        let collapsed = ts.visible().len();
        ts.reveal("blk.1.b");
        assert!(ts.visible().len() > collapsed);
        assert_eq!(ts.visible()[ts.selected].0.name(), "blk.1.b");
    }
}
