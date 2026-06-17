use anyhow::{Context, Result};
use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    execute,
    terminal::{self, ClearType},
};
use fuzzy_matcher::FuzzyMatcher;
use fuzzy_matcher::skim::SkimMatcherV2;
use std::{
    collections::{BTreeSet, HashSet},
    fs::File,
    io::{self, Read, Write},
    path::{Path, PathBuf},
};

use crate::gguf::GGUFFile;

use crate::tree::{
    Layout, MetadataInfo, Storage, TensorInfo, TreeBuilder, TreeNode, natural_sort_key,
};
use crate::ui::{DrawConfig, UI};
use crate::utils::base64_encode;

pub struct Explorer {
    files: Vec<PathBuf>,
    tensors: Vec<TensorInfo>,
    metadata: Vec<MetadataInfo>,
    tree: Vec<TreeNode>,
    selected_idx: usize,
    scroll_offset: usize,
    flattened_tree: Vec<(TreeNode, usize)>,
    total_parameters: usize,
    search_query: String,
    search_mode: bool,
    filtered_tree: Vec<(TreeNode, usize)>,
    /// Transient "✓ Copied …" message shown after pressing `c`; cleared on the
    /// next key press.
    copied_flash: Option<String>,
    /// Index/file mismatches detected at startup, shown as a warning panel.
    health_reports: Vec<crate::health::HealthReport>,
}

impl Explorer {
    pub fn new(files: Vec<PathBuf>, health_reports: Vec<crate::health::HealthReport>) -> Self {
        Self {
            files,
            tensors: Vec::new(),
            metadata: Vec::new(),
            tree: Vec::new(),
            selected_idx: 0,
            scroll_offset: 0,
            flattened_tree: Vec::new(),
            total_parameters: 0,
            search_query: String::new(),
            search_mode: false,
            filtered_tree: Vec::new(),
            copied_flash: None,
            health_reports,
        }
    }

    fn load_all_files(&mut self) -> Result<()> {
        self.tensors.clear();
        self.metadata.clear();

        let files = self.files.clone();
        for file_path in &files {
            let extension = file_path.extension().and_then(|s| s.to_str());

            match extension {
                Some("safetensors") => {
                    self.load_safetensors_file(file_path)?;
                }
                Some("gguf") => {
                    self.load_gguf_file(file_path)?;
                }
                Some("h5") | Some("hdf5") => {
                    #[cfg(feature = "hdf5")]
                    self.load_hdf5_file(file_path)?;
                    #[cfg(not(feature = "hdf5"))]
                    eprintln!(
                        "Warning: HDF5 support is not compiled in; rebuild with `--features hdf5` to read {}",
                        file_path.display()
                    );
                }
                _ => {
                    eprintln!("Warning: Unsupported file format: {}", file_path.display());
                }
            }
        }

        // Deduplicate tensors by name
        let mut seen_names = HashSet::new();
        self.tensors
            .retain(|tensor| seen_names.insert(tensor.name.clone()));

        self.tensors.sort_by_key(|a| natural_sort_key(&a.name));
        self.total_parameters = self.tensors.iter().map(|t| t.num_elements).sum::<usize>();
        self.build_tree();
        Ok(())
    }

    fn load_safetensors_file(&mut self, file_path: &PathBuf) -> Result<()> {
        let source_path = absolute_path(file_path);
        let mut file = File::open(file_path)
            .with_context(|| format!("Failed to open file: {}", file_path.display()))?;

        // A safetensors file begins with an 8-byte little-endian header length N
        // followed by N bytes of JSON describing every tensor (name, dtype, shape,
        // byte offsets) and an optional `__metadata__` map. The tensor data follows.
        // We only display that header, so read just it instead of the whole file
        // (which can be many GB per shard).
        let mut len_buf = [0u8; 8];
        file.read_exact(&mut len_buf)
            .with_context(|| format!("Failed to read header length: {}", file_path.display()))?;
        let header_len = u64::from_le_bytes(len_buf) as usize;

        // Guard against a corrupt or non-safetensors file claiming a huge header.
        const MAX_HEADER_SIZE: usize = 100_000_000;
        if header_len > MAX_HEADER_SIZE {
            anyhow::bail!(
                "SafeTensors header too large ({header_len} bytes): {}",
                file_path.display()
            );
        }

        let mut header_buf = vec![0u8; header_len];
        file.read_exact(&mut header_buf)
            .with_context(|| format!("Failed to read header: {}", file_path.display()))?;

        let header: serde_json::Value = serde_json::from_slice(&header_buf).with_context(|| {
            format!(
                "Failed to parse SafeTensors header: {}",
                file_path.display()
            )
        })?;

        let obj = header.as_object().ok_or_else(|| {
            anyhow::anyhow!("Invalid SafeTensors header: {}", file_path.display())
        })?;

        for (key, value) in obj {
            // The `__metadata__` entry holds free-form string key/value pairs.
            if key == "__metadata__" {
                if let Some(meta_obj) = value.as_object() {
                    for (meta_key, meta_value) in meta_obj {
                        self.metadata.push(MetadataInfo {
                            name: meta_key.clone(),
                            value: match meta_value.as_str() {
                                Some(s) => s.to_string(),
                                None => meta_value.to_string(),
                            },
                            value_type: "string".to_string(),
                        });
                    }
                }
                continue;
            }

            // Every other entry describes a tensor.
            let dtype = value
                .get("dtype")
                .and_then(|v| v.as_str())
                .unwrap_or("?")
                .to_string();
            let shape: Vec<usize> = value
                .get("shape")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|x| x.as_u64().map(|n| n as usize))
                        .collect()
                })
                .unwrap_or_default();
            let data_offsets = value
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .filter(|offsets| offsets.len() == 2)
                .and_then(|offsets| Some((offsets[0].as_u64()?, offsets[1].as_u64()?)));
            let size_bytes = data_offsets
                .map(|(start, end)| end.saturating_sub(start) as usize)
                .unwrap_or(0);
            let layout = match data_offsets {
                Some((start, end)) => Layout::ByteRange { start, end },
                None => Layout::None,
            };
            let num_elements = shape.iter().product::<usize>();

            self.tensors.push(TensorInfo {
                name: key.clone(),
                dtype,
                shape,
                size_bytes,
                num_elements,
                storage: Storage::Unknown,
                source_path: source_path.clone(),
                layout,
            });
        }

        Ok(())
    }

    fn load_gguf_file(&mut self, file_path: &PathBuf) -> Result<()> {
        let source_path = absolute_path(file_path);
        let mut file = File::open(file_path)
            .with_context(|| format!("Failed to open file: {}", file_path.display()))?;

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .with_context(|| format!("Failed to read file: {}", file_path.display()))?;

        let gguf = GGUFFile::read(&buffer)
            .with_context(|| format!("Failed to parse GGUF file: {}", file_path.display()))?;

        // Load metadata
        for (key, value) in &gguf.metadata {
            let value_type = match value {
                crate::gguf::GGUFValue::U8(_) => "u8",
                crate::gguf::GGUFValue::I8(_) => "i8",
                crate::gguf::GGUFValue::U16(_) => "u16",
                crate::gguf::GGUFValue::I16(_) => "i16",
                crate::gguf::GGUFValue::U32(_) => "u32",
                crate::gguf::GGUFValue::I32(_) => "i32",
                crate::gguf::GGUFValue::F32(_) => "f32",
                crate::gguf::GGUFValue::U64(_) => "u64",
                crate::gguf::GGUFValue::I64(_) => "i64",
                crate::gguf::GGUFValue::F64(_) => "f64",
                crate::gguf::GGUFValue::Bool(_) => "bool",
                crate::gguf::GGUFValue::String(_) => "string",
                crate::gguf::GGUFValue::Array(_) => "array",
            };

            self.metadata.push(MetadataInfo {
                name: key.clone(),
                value: value.to_string(),
                value_type: value_type.to_string(),
            });
        }

        // Load tensors
        for tensor in &gguf.tensors {
            let shape: Vec<usize> = tensor.dimensions.iter().map(|&d| d as usize).collect();
            let dtype = tensor.tensor_type.to_string();

            // Calculate size using the element size from our custom implementation
            let num_elements = shape.iter().product::<usize>();
            let size_bytes =
                (num_elements as f32 * tensor.tensor_type.element_size_bytes()) as usize;

            self.tensors.push(TensorInfo {
                name: tensor.name.clone(),
                dtype,
                shape,
                size_bytes,
                num_elements,
                storage: Storage::Unknown,
                source_path: source_path.clone(),
                layout: Layout::Offset(tensor.offset),
            });
        }

        Ok(())
    }

    #[cfg(feature = "hdf5")]
    fn load_hdf5_file(&mut self, file_path: &std::path::Path) -> Result<()> {
        let tensors = crate::hdf5::read_tensors(file_path)?;
        self.tensors.extend(tensors);
        Ok(())
    }

    fn build_tree(&mut self) {
        if self.metadata.is_empty() {
            self.tree = TreeBuilder::build_tree(&self.tensors);
        } else {
            self.tree = TreeBuilder::build_tree_mixed(&self.tensors, &self.metadata);
        }
        self.flatten_tree();
    }

    fn flatten_tree(&mut self) {
        self.flattened_tree = TreeBuilder::flatten_tree(&self.tree);
        self.update_filtered_tree();
    }

    /// Expand or collapse every group, then reset the cursor to the top since
    /// the visible rows change wholesale.
    fn set_all_expanded(&mut self, expanded: bool) {
        TreeBuilder::set_all_expanded(&mut self.tree, expanded);
        self.flatten_tree();
        self.selected_idx = 0;
        self.scroll_offset = 0;
    }

    fn update_filtered_tree(&mut self) {
        if self.search_query.is_empty() {
            self.filtered_tree = self.flattened_tree.clone();
        } else {
            let matcher = SkimMatcherV2::default();
            let mut scored_results: Vec<(TreeNode, i64)> = Vec::new();

            // Search through ALL tensors, not just the flattened tree
            for tensor in &self.tensors {
                if let Some(score) = matcher.fuzzy_match(&tensor.name, &self.search_query) {
                    scored_results.push((
                        TreeNode::Tensor {
                            info: tensor.clone(),
                        },
                        score,
                    ));
                }
            }

            // Also search through metadata if present
            for metadata in &self.metadata {
                if let Some(score) = matcher.fuzzy_match(&metadata.name, &self.search_query) {
                    scored_results.push((
                        TreeNode::Metadata {
                            info: metadata.clone(),
                        },
                        score,
                    ));
                }
            }

            // Sort by score (highest first)
            scored_results.sort_by_key(|b| std::cmp::Reverse(b.1));

            // Create a flat list with depth 0 for all results
            self.filtered_tree = scored_results
                .into_iter()
                .map(|(node, _)| (node, 0))
                .collect();
        }
    }

    pub fn run(&mut self) -> Result<()> {
        if self.files.is_empty() {
            return Ok(());
        }

        terminal::enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, terminal::Clear(ClearType::All), cursor::Hide)?;

        let result = self.interactive_loop();

        execute!(stdout, terminal::Clear(ClearType::All), cursor::Show)?;
        terminal::disable_raw_mode()?;

        result
    }

    fn interactive_loop(&mut self) -> Result<()> {
        self.load_all_files()?;

        loop {
            let title = if self.files.len() == 1 {
                self.files[0].to_string_lossy().to_string()
            } else {
                "Multiple files".to_string()
            };

            let tree_to_display = if self.search_mode {
                &self.filtered_tree
            } else {
                &self.flattened_tree
            };

            let status_bar = self.status_bar_text();

            let config = DrawConfig {
                tree: tree_to_display,
                current_file: &title,
                file_idx: 0,
                total_files: 1,
                total_parameters: self.total_parameters,
                selected_idx: self.selected_idx,
                scroll_offset: self.scroll_offset,
                search_mode: self.search_mode,
                search_query: &self.search_query,
                status_bar: &status_bar,
                health_warning: !self.health_reports.is_empty(),
            };
            self.scroll_offset = UI::draw_screen(&config)?;

            if let Event::Key(key_event) = event::read()? {
                // The copy confirmation only lasts until the next key press.
                self.copied_flash = None;
                match key_event {
                    KeyEvent {
                        code: KeyCode::Char('q'),
                        ..
                    } => {
                        if self.search_mode {
                            self.exit_search_mode();
                        } else {
                            break;
                        }
                    }
                    KeyEvent {
                        code: KeyCode::Char('c'),
                        modifiers: KeyModifiers::CONTROL,
                        ..
                    } => break,
                    // `c` (no modifier) copies the selected tensor's source path.
                    // In search mode it falls through to be typed into the query.
                    KeyEvent {
                        code: KeyCode::Char('c'),
                        ..
                    } if !self.search_mode => self.copy_selected_path(),
                    // `h` shows the checkpoint health report (when there is one).
                    KeyEvent {
                        code: KeyCode::Char('h'),
                        ..
                    } if !self.search_mode => self.show_health_report(),
                    // `E` / `C` expand / collapse every group at once.
                    KeyEvent {
                        code: KeyCode::Char('E'),
                        ..
                    } if !self.search_mode => self.set_all_expanded(true),
                    KeyEvent {
                        code: KeyCode::Char('C'),
                        ..
                    } if !self.search_mode => self.set_all_expanded(false),
                    KeyEvent {
                        code: KeyCode::Char('/'),
                        ..
                    } if !self.search_mode => self.enter_search_mode(),
                    // While searching, '/' is ignored rather than typed into the query.
                    KeyEvent {
                        code: KeyCode::Char('/'),
                        ..
                    } => {}
                    KeyEvent {
                        code: KeyCode::Esc, ..
                    } if self.search_mode => self.exit_search_mode(),
                    // Shift+↑/↓ jump to the previous/next sibling (same depth,
                    // same parent). These must precede the plain arrow arms,
                    // which match any modifiers.
                    KeyEvent {
                        code: KeyCode::Up,
                        modifiers: KeyModifiers::SHIFT,
                        ..
                    } => self.move_to_sibling(false),
                    KeyEvent {
                        code: KeyCode::Down,
                        modifiers: KeyModifiers::SHIFT,
                        ..
                    } => self.move_to_sibling(true),
                    KeyEvent {
                        code: KeyCode::Up, ..
                    } => self.move_selection(-1),
                    KeyEvent {
                        code: KeyCode::Down,
                        ..
                    } => self.move_selection(1),
                    // ← jumps to the parent group; → enters the group (its
                    // first child), expanding it first if collapsed.
                    KeyEvent {
                        code: KeyCode::Left,
                        ..
                    } => self.move_to_parent(),
                    KeyEvent {
                        code: KeyCode::Right,
                        ..
                    } => self.move_to_first_child(),
                    // Enter acts on the highlighted row in both modes: expand a
                    // group or open a tensor/metadata detail. In search mode it
                    // opens the result's detail (and stays in search); use Esc
                    // or `q` to leave search.
                    KeyEvent {
                        code: KeyCode::Enter,
                        ..
                    } => self.handle_selection(),
                    KeyEvent {
                        code: KeyCode::Char(' '),
                        ..
                    } if !self.search_mode => self.handle_selection(),
                    // While searching, space is ignored rather than typed into the query.
                    KeyEvent {
                        code: KeyCode::Char(' '),
                        ..
                    } => {}
                    KeyEvent {
                        code: KeyCode::Backspace,
                        ..
                    } if self.search_mode => {
                        self.search_query.pop();
                        self.update_filtered_tree();
                        self.selected_idx = 0;
                        self.scroll_offset = 0;
                    }
                    KeyEvent {
                        code: KeyCode::Char(c),
                        ..
                    } if self.search_mode => {
                        self.search_query.push(c);
                        self.update_filtered_tree();
                        self.selected_idx = 0;
                        self.scroll_offset = 0;
                    }
                    // Remove left/right file navigation since we're showing all files merged
                    _ => {}
                }
            }
        }

        Ok(())
    }

    /// Text for the bottom status bar: the source file(s) of the row under the
    /// cursor, or the transient copy confirmation.
    fn status_bar_text(&self) -> String {
        if let Some(flash) = &self.copied_flash {
            return format!("✓ Copied to clipboard: {flash}");
        }

        let tree = if self.search_mode {
            &self.filtered_tree
        } else {
            &self.flattened_tree
        };
        let Some((node, _)) = tree.get(self.selected_idx) else {
            return String::new();
        };

        match node {
            TreeNode::Tensor { info } => info.source_path.clone(),
            TreeNode::Group { .. } => {
                let mut files = BTreeSet::new();
                collect_source_paths(node, &mut files);
                match files.len() {
                    0 => String::new(),
                    1 => files.into_iter().next().unwrap(),
                    n => {
                        let first = file_name(files.iter().next().unwrap());
                        let last = file_name(files.iter().next_back().unwrap());
                        format!("stored across {n} files: {first} … {last}")
                    }
                }
            }
            TreeNode::Metadata { .. } => String::new(),
        }
    }

    /// Copy the source path of the selected tensor to the clipboard (OSC 52).
    fn copy_selected_path(&mut self) {
        if let Some((TreeNode::Tensor { info }, _)) = self.flattened_tree.get(self.selected_idx) {
            let path = info.source_path.clone();
            copy_to_clipboard(&path);
            self.copied_flash = Some(path);
        }
    }

    fn move_selection(&mut self, delta: i32) {
        let tree = if self.search_mode {
            &self.filtered_tree
        } else {
            &self.flattened_tree
        };

        if tree.is_empty() {
            return;
        }

        let new_idx = if delta < 0 {
            self.selected_idx.saturating_sub((-delta) as usize)
        } else {
            (self.selected_idx + delta as usize).min(tree.len() - 1)
        };

        self.selected_idx = new_idx;
    }

    /// Move the cursor to the parent group of the selected row (the nearest
    /// preceding row at a shallower depth). No-op at the top level.
    fn move_to_parent(&mut self) {
        let tree = if self.search_mode {
            &self.filtered_tree
        } else {
            &self.flattened_tree
        };
        let Some(&(_, depth)) = tree.get(self.selected_idx) else {
            return;
        };
        if depth == 0 {
            return;
        }
        if let Some(parent) = (0..self.selected_idx).rev().find(|&i| tree[i].1 < depth) {
            self.selected_idx = parent;
        }
    }

    /// Move the cursor to the next/previous sibling: the nearest row at the
    /// same depth before a shallower row (i.e. without leaving the parent).
    fn move_to_sibling(&mut self, forward: bool) {
        let tree = if self.search_mode {
            &self.filtered_tree
        } else {
            &self.flattened_tree
        };
        let Some(&(_, depth)) = tree.get(self.selected_idx) else {
            return;
        };

        let indices: Vec<usize> = if forward {
            (self.selected_idx + 1..tree.len()).collect()
        } else {
            (0..self.selected_idx).rev().collect()
        };
        for i in indices {
            let d = tree[i].1;
            if d < depth {
                break; // left the parent: no sibling in this direction
            }
            if d == depth {
                self.selected_idx = i;
                break;
            }
            // d > depth: a descendant, keep scanning
        }
    }

    /// Enter the selected group: expand it if collapsed, then move the cursor
    /// to its first child. No-op for leaf rows or empty groups (and in search
    /// mode, where the list is flat).
    fn move_to_first_child(&mut self) {
        if self.search_mode {
            return;
        }
        let (expanded, has_children, depth) = match self.flattened_tree.get(self.selected_idx) {
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
            let mut tree_clone = self.tree.clone();
            let _ = TreeBuilder::toggle_node_by_index(self.selected_idx, &mut tree_clone);
            self.tree = tree_clone;
            self.flatten_tree();
        }
        // The first child is the next row, one level deeper.
        if let Some((_, child_depth)) = self.flattened_tree.get(self.selected_idx + 1)
            && *child_depth == depth + 1
        {
            self.selected_idx += 1;
        }
    }

    fn enter_search_mode(&mut self) {
        self.search_mode = true;
        self.search_query.clear();
        self.update_filtered_tree();
        self.selected_idx = 0;
        self.scroll_offset = 0;
    }

    fn exit_search_mode(&mut self) {
        self.search_mode = false;
        self.search_query.clear();
        self.update_filtered_tree();
        self.selected_idx = 0;
        self.scroll_offset = 0;
    }

    fn handle_selection(&mut self) {
        let tree = if self.search_mode {
            &self.filtered_tree
        } else {
            &self.flattened_tree
        };

        if self.selected_idx < tree.len() {
            let (selected_node, _) = &tree[self.selected_idx];

            match selected_node {
                TreeNode::Group { .. } => {
                    // In search mode, groups shouldn't appear, but if they do, do nothing
                    if !self.search_mode {
                        let mut tree_clone = self.tree.clone();
                        let _ =
                            TreeBuilder::toggle_node_by_index(self.selected_idx, &mut tree_clone);
                        self.tree = tree_clone;
                        self.flatten_tree();
                    }
                }
                TreeNode::Tensor { info } => {
                    self.show_tensor_detail(info);
                }
                TreeNode::Metadata { info } => {
                    self.show_metadata_detail(info);
                }
            }
        }
    }

    fn show_tensor_detail(&self, tensor: &TensorInfo) {
        // Detail screen with sub-views: `m` heatmap, `v` numeric values, any
        // other key returns to the tree.
        loop {
            if UI::draw_tensor_detail(tensor).is_err() {
                return;
            }
            match event::read() {
                Ok(Event::Key(key)) if is_ctrl_c(&key) => quit_immediately(),
                Ok(Event::Key(KeyEvent {
                    code: KeyCode::Char('m'),
                    ..
                })) => self.show_tensor_data(tensor, true),
                Ok(Event::Key(KeyEvent {
                    code: KeyCode::Char('v'),
                    ..
                })) => self.show_tensor_data(tensor, false),
                Ok(Event::Key(_)) => return,
                Ok(_) => {} // resize etc.: just redraw the detail
                Err(_) => return,
            }
        }
    }

    /// Draw a heatmap (`heatmap = true`) or numeric grid for the tensor, sized
    /// to the terminal. `m`/`v` switch representation in place (no trip back to
    /// the detail screen). For 3D tensors this shows one 2D slice at a fixed
    /// first index (the 0th by default); `[`/`]` and the ← → arrows step
    /// through the slices, wrapping around at both ends. Any other key returns
    /// to the detail screen.
    fn show_tensor_data(&self, tensor: &TensorInfo, heatmap: bool) {
        let mut heatmap = heatmap;
        let mut slice = 0usize;
        loop {
            let (cols, rows) = terminal::size().unwrap_or((100, 40));
            let max_rows = (rows as usize).saturating_sub(8).max(1);
            let sampled = if heatmap {
                let max_cols = (cols as usize).saturating_sub(1).max(1);
                crate::sample::sample_tensor(tensor, max_rows, max_cols, slice)
            } else {
                // Numeric cells are ~11 wide plus a row-index column.
                let max_cols = ((cols as usize).saturating_sub(7) / 11).max(1);
                crate::sample::sample_tensor(tensor, max_rows, max_cols, slice)
            };

            // The number of slices to navigate between (1 unless this is 3D);
            // also reflects clamping of `slice` done inside the sampler.
            let slices = match &sampled {
                Ok(s) => {
                    slice = s.slice;
                    s.slices
                }
                Err(_) => 1,
            };

            let result = sampled.and_then(|s| {
                if heatmap {
                    UI::draw_heatmap(tensor, &s).map_err(|e| e.to_string())
                } else {
                    UI::draw_values(tensor, &s).map_err(|e| e.to_string())
                }
            });
            if let Err(msg) = result {
                let _ = UI::draw_message("Data preview unavailable", &msg);
                if let Ok(Event::Key(key)) = event::read()
                    && is_ctrl_c(&key)
                {
                    quit_immediately();
                }
                return;
            }

            match event::read() {
                Ok(Event::Key(key)) if is_ctrl_c(&key) => quit_immediately(),
                Ok(Event::Key(KeyEvent {
                    code, modifiers, ..
                })) => {
                    let shift = modifiers.contains(KeyModifiers::SHIFT);
                    match code {
                        // Switch representation in place, keeping the current slice.
                        KeyCode::Char('m') => heatmap = true,
                        KeyCode::Char('v') => heatmap = false,
                        // Jump straight to a slice by typing its index.
                        KeyCode::Char('/') if slices > 1 => {
                            if let Some(n) = self.prompt_slice(slices) {
                                slice = n;
                            }
                        }
                        // Shift + arrows jump ~5% of the slices at once (wrapping).
                        KeyCode::Right if slices > 1 && shift => {
                            slice = (slice + slice_step(slices)) % slices
                        }
                        KeyCode::Left if slices > 1 && shift => {
                            slice = (slice + slices - slice_step(slices)) % slices
                        }
                        // Plain arrows / brackets step one slice (wrapping).
                        KeyCode::Char(']') | KeyCode::Right if slices > 1 => {
                            slice = (slice + 1) % slices
                        }
                        KeyCode::Char('[') | KeyCode::Left if slices > 1 => {
                            slice = (slice + slices - 1) % slices
                        }
                        _ => return,
                    }
                }
                Ok(_) => {} // resize etc.: re-sample and redraw the same slice
                Err(_) => return,
            }
        }
    }

    /// Prompt for a slice to jump to — either an absolute index (`123`) or a
    /// percentage of the way through (`50%`, where 0% is the first slice and
    /// 100% the last). Returns the chosen slice, or `None` if cancelled / left
    /// empty. Out-of-range entries are reported in the prompt, not jumped to.
    /// Ctrl-C quits the app outright.
    fn prompt_slice(&self, slices: usize) -> Option<usize> {
        let mut input = String::new();
        let mut error: Option<String> = None;
        loop {
            if UI::draw_slice_prompt(slices, &input, error.as_deref()).is_err() {
                return None;
            }
            match event::read() {
                Ok(Event::Key(key)) if is_ctrl_c(&key) => quit_immediately(),
                Ok(Event::Key(KeyEvent { code, .. })) => match code {
                    KeyCode::Enter => match parse_slice_input(&input, slices) {
                        Ok(Some(n)) => return Some(n),
                        Ok(None) => return None, // empty + Enter cancels
                        Err(msg) => error = Some(msg),
                    },
                    KeyCode::Esc => return None,
                    KeyCode::Backspace => {
                        input.pop();
                        error = None;
                    }
                    // Accept digits, a decimal point and a trailing `%`.
                    KeyCode::Char(c) if c.is_ascii_digit() || c == '.' || c == '%' => {
                        input.push(c);
                        error = None;
                    }
                    _ => {}
                },
                Ok(_) => {}
                Err(_) => return None,
            }
        }
    }

    fn show_metadata_detail(&self, metadata: &MetadataInfo) {
        if UI::draw_metadata_detail(metadata).is_ok() {
            // Wait for any key press (Ctrl-C quits the app).
            if let Ok(Event::Key(key)) = event::read()
                && is_ctrl_c(&key)
            {
                quit_immediately();
            }
        }
    }

    fn show_health_report(&self) {
        if !self.health_reports.is_empty() && UI::draw_health_warning(&self.health_reports).is_ok()
        {
            // Wait for any key press (Ctrl-C quits the app).
            if let Ok(Event::Key(key)) = event::read()
                && is_ctrl_c(&key)
            {
                quit_immediately();
            }
        }
    }
}

/// How many slices one Shift+arrow jump moves: about 5% of the total, at
/// least 1 so it always advances.
fn slice_step(slices: usize) -> usize {
    (slices / 20).max(1)
}

/// Parse the slice-jump prompt input into a target slice for a tensor with
/// `slices` slices. Accepts either an absolute index (`"123"`) or a percentage
/// (`"50%"`, where 0% is the first slice and 100% the last). Returns `Ok(None)`
/// for empty input (cancel) and `Err(message)` for invalid / out-of-range input
/// (so the prompt can report it instead of jumping).
fn parse_slice_input(input: &str, slices: usize) -> Result<Option<usize>, String> {
    let s = input.trim();
    if s.is_empty() {
        return Ok(None);
    }
    if let Some(pct_str) = s.strip_suffix('%') {
        let pct: f64 = pct_str
            .trim()
            .parse()
            .map_err(|_| "invalid percentage — try e.g. 50%".to_string())?;
        if !(0.0..=100.0).contains(&pct) {
            return Err(format!("{pct}% is out of range — use 0% to 100%"));
        }
        // 0% -> first slice, 100% -> last slice; round to the nearest.
        let idx = ((pct / 100.0) * (slices - 1) as f64).round() as usize;
        Ok(Some(idx.min(slices - 1)))
    } else {
        let n: usize = s
            .parse()
            .map_err(|_| "enter a slice number or a percentage (e.g. 12 or 50%)".to_string())?;
        if n < slices {
            Ok(Some(n))
        } else {
            Err(format!(
                "index {n} is out of range — the last slice is {}",
                slices - 1
            ))
        }
    }
}

/// Whether a key event is Ctrl-C.
fn is_ctrl_c(key: &KeyEvent) -> bool {
    key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL)
}

/// Restore the terminal (leave raw mode, show the cursor, clear the screen) and
/// exit the process immediately. Used for Ctrl-C from any of the detail/data
/// sub-screens so it quits outright instead of stepping back one screen.
fn quit_immediately() -> ! {
    let mut stdout = io::stdout();
    let _ = execute!(stdout, terminal::Clear(ClearType::All), cursor::Show);
    let _ = terminal::disable_raw_mode();
    std::process::exit(0);
}

/// Resolve a path to an absolute string without requiring it to exist or
/// resolving symlinks; falls back to the original path on error.
fn absolute_path(path: &Path) -> String {
    std::path::absolute(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

/// The final path component (file name) of a path string.
fn file_name(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string())
}

/// Collect the distinct source files of every tensor under `node`.
fn collect_source_paths(node: &TreeNode, out: &mut BTreeSet<String>) {
    match node {
        TreeNode::Tensor { info } => {
            out.insert(info.source_path.clone());
        }
        TreeNode::Group { children, .. } => {
            for child in children {
                collect_source_paths(child, out);
            }
        }
        TreeNode::Metadata { .. } => {}
    }
}

/// Copy `text` to the terminal clipboard via the OSC 52 escape sequence. This
/// reaches the *local* clipboard even over SSH/tmux (when the terminal supports
/// OSC 52), unlike shelling out to xclip/pbcopy on the remote host.
fn copy_to_clipboard(text: &str) {
    let mut stdout = io::stdout();
    let _ = write!(stdout, "\x1b]52;c;{}\x07", base64_encode(text.as_bytes()));
    let _ = stdout.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_slice_input_handles_indices_percentages_and_errors() {
        // Empty input cancels.
        assert_eq!(parse_slice_input("", 10), Ok(None));
        assert_eq!(parse_slice_input("   ", 10), Ok(None));

        // Absolute indices.
        assert_eq!(parse_slice_input("0", 10), Ok(Some(0)));
        assert_eq!(parse_slice_input("9", 10), Ok(Some(9)));
        assert!(parse_slice_input("10", 10).is_err()); // out of range (max 9)

        // Percentages: 0% -> first, 100% -> last, rounded to nearest in between.
        assert_eq!(parse_slice_input("0%", 360), Ok(Some(0)));
        assert_eq!(parse_slice_input("100%", 360), Ok(Some(359)));
        assert_eq!(parse_slice_input("50%", 360), Ok(Some(180))); // 0.5 * 359 = 179.5 -> 180
        assert_eq!(parse_slice_input("50%", 11), Ok(Some(5))); // 0.5 * 10 = 5
        assert_eq!(parse_slice_input("33.3%", 100), Ok(Some(33)));

        // Out-of-range / malformed percentages and numbers are reported.
        assert!(parse_slice_input("101%", 360).is_err());
        assert!(parse_slice_input("-5%", 360).is_err());
        assert!(parse_slice_input("abc", 360).is_err());
        assert!(parse_slice_input("%", 360).is_err());
    }

    /// Build an explorer whose flattened tree has the given row depths (the
    /// node contents don't matter for coarse navigation, only the depths).
    fn explorer_with_depths(depths: &[usize]) -> Explorer {
        let mut e = Explorer::new(Vec::new(), Vec::new());
        e.flattened_tree = depths
            .iter()
            .map(|&d| {
                (
                    TreeNode::Group {
                        name: String::new(),
                        children: Vec::new(),
                        expanded: false,
                        tensor_count: 0,
                        total_size: 0,
                        stored_size: 0,
                    },
                    d,
                )
            })
            .collect();
        e
    }

    // Depths:  0:0  1:1  2:1  3:2  4:1  5:0
    #[test]
    fn move_to_parent_jumps_to_the_nearest_shallower_row() {
        let mut e = explorer_with_depths(&[0, 1, 1, 2, 1, 0]);

        e.selected_idx = 3;
        e.move_to_parent();
        assert_eq!(e.selected_idx, 2);

        e.selected_idx = 1;
        e.move_to_parent();
        assert_eq!(e.selected_idx, 0);

        // Top-level row has no parent.
        e.selected_idx = 0;
        e.move_to_parent();
        assert_eq!(e.selected_idx, 0);
    }

    #[test]
    fn move_to_sibling_skips_descendants_and_stops_at_the_parent_boundary() {
        let mut e = explorer_with_depths(&[0, 1, 1, 2, 1, 0]);

        // Forward from idx2 (depth 1) skips the descendant idx3 (depth 2).
        e.selected_idx = 2;
        e.move_to_sibling(true);
        assert_eq!(e.selected_idx, 4);

        // Forward from idx4: the next row (idx5) is shallower, so no sibling.
        e.selected_idx = 4;
        e.move_to_sibling(true);
        assert_eq!(e.selected_idx, 4);

        // Backward from idx4 lands on idx2, skipping idx3.
        e.selected_idx = 4;
        e.move_to_sibling(false);
        assert_eq!(e.selected_idx, 2);
    }

    fn group(depth: usize, expanded: bool, child: bool) -> (TreeNode, usize) {
        let children = if child {
            vec![TreeNode::Group {
                name: String::new(),
                children: Vec::new(),
                expanded: false,
                tensor_count: 0,
                total_size: 0,
                stored_size: 0,
            }]
        } else {
            Vec::new()
        };
        (
            TreeNode::Group {
                name: String::new(),
                children,
                expanded,
                tensor_count: 0,
                total_size: 0,
                stored_size: 0,
            },
            depth,
        )
    }

    #[test]
    fn move_to_first_child_enters_an_expanded_group() {
        let mut e = Explorer::new(Vec::new(), Vec::new());
        // idx0: expanded group with a child; idx1: that child (depth 1);
        // idx2: a childless group at depth 0.
        e.flattened_tree = vec![
            group(0, true, true),
            group(1, false, false),
            group(0, false, false),
        ];

        e.selected_idx = 0;
        e.move_to_first_child();
        assert_eq!(e.selected_idx, 1);

        // A group with no children does not move.
        e.selected_idx = 2;
        e.move_to_first_child();
        assert_eq!(e.selected_idx, 2);
    }
}
