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
            .map(|r| (r.is_dir, r.expanded, r.depth))
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
        let Some((is_dir, expanded)) = self.rows.get(self.selected).map(|r| (r.is_dir, r.expanded))
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
    /// The tensor tree flattened to visible rows, derived once from the tensors
    /// (no disk); search filters it on the fly.
    flat: Vec<(TreeNode, usize)>,
    screen: Screen,
    selected: usize,
    search: String,
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
        let flat = Self::flatten(&tensors, &metadata);
        Session {
            model,
            tensors,
            metadata,
            config,
            total_parameters,
            flat,
            screen: Screen::Tree,
            selected: 0,
            search: String::new(),
            stats: None,
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

    /// Build + flatten the tensor tree from the given tensors/metadata.
    fn flatten(tensors: &[TensorInfo], metadata: &[MetadataInfo]) -> Vec<(TreeNode, usize)> {
        let tree = if metadata.is_empty() {
            TreeBuilder::build_tree(tensors)
        } else {
            TreeBuilder::build_tree_mixed(tensors, metadata)
        };
        TreeBuilder::flatten_tree(&tree)
    }

    /// Drop the tensors and metadata whose names don't pass `keep`, recompute the
    /// parameter count, rebuild the flattened tree, and invalidate the cached
    /// stats — backs the `--print-tree`/`--print-tensors` name-filter subset.
    pub fn retain_named<F: FnMut(&str) -> bool>(&mut self, mut keep: F) {
        self.tensors.retain(|t| keep(&t.name));
        self.metadata.retain(|m| keep(&m.name));
        self.total_parameters = self.tensors.iter().map(|t| t.num_elements).sum();
        self.flat = Self::flatten(&self.tensors, &self.metadata);
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
            Screen::Files => self.model.as_ref().map_or(0, |m| m.files.len()),
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
                .as_ref()
                .map(|m| m.files.as_slice())
                .unwrap_or_default()
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
            root: self
                .model
                .as_ref()
                .map(|m| m.root.clone())
                .unwrap_or_default(),
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
        let mut s = Session::from_model(model());
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
