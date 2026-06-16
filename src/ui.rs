use anyhow::Result;
use crossterm::{
    cursor, execute, queue,
    style::{Color, ResetColor, SetForegroundColor},
    terminal::{self, ClearType},
};
use std::io::{self, BufWriter, Write};

use crate::tree::{MetadataInfo, Storage, TensorInfo, TreeNode};
use crate::utils::{format_parameters, format_shape, format_size};

pub struct DrawConfig<'a> {
    pub tree: &'a [(TreeNode, usize)],
    pub current_file: &'a str,
    pub file_idx: usize,
    pub total_files: usize,
    pub total_parameters: usize,
    pub selected_idx: usize,
    pub scroll_offset: usize,
    pub search_mode: bool,
    pub search_query: &'a str,
}

pub struct UI;

impl UI {
    pub fn draw_screen(config: &DrawConfig) -> Result<usize> {
        // Render the whole frame into one buffered writer and flush it once.
        // `io::Stdout` is line-buffered, so writing directly would flush on
        // every newline and paint the frame progressively; a `BufWriter` makes
        // the update atomic. Combined with overwriting in place (clearing each
        // line rather than the whole screen up front), this removes the flicker
        // that a per-frame `Clear(All)` produced.
        let stdout = io::stdout();
        let mut out = BufWriter::new(stdout.lock());

        let (_, terminal_height) = terminal::size()?;
        let header_height = 3;
        let footer_height = 2;
        let available_height =
            (terminal_height as usize).saturating_sub(header_height + footer_height);

        queue!(out, cursor::MoveTo(0, 0))?;

        // Header
        queue!(out, terminal::Clear(ClearType::CurrentLine))?;
        write!(
            out,
            "SafeTensors Explorer - {} ({}/{})\r\n",
            config.current_file,
            config.file_idx + 1,
            config.total_files
        )?;
        queue!(out, terminal::Clear(ClearType::CurrentLine))?;
        if config.search_mode {
            write!(
                out,
                "SEARCH MODE: {} | Type to search, Enter/Esc to exit search\r\n",
                if config.search_query.is_empty() {
                    "_"
                } else {
                    config.search_query
                }
            )?;
        } else {
            write!(
                out,
                "Use ↑/↓ to navigate, Enter/Space to expand/collapse, / to search, q to quit\r\n"
            )?;
        }
        queue!(out, terminal::Clear(ClearType::CurrentLine))?;
        write!(out, "{}\r\n", "=".repeat(80))?;

        // Calculate scroll offset
        let new_scroll_offset = if config.selected_idx >= config.scroll_offset + available_height {
            config.selected_idx.saturating_sub(available_height - 1)
        } else if config.selected_idx < config.scroll_offset {
            config.selected_idx
        } else {
            config.scroll_offset
        };

        // Draw tree
        for (actual_index, (node, depth)) in config
            .tree
            .iter()
            .enumerate()
            .skip(new_scroll_offset)
            .take(available_height)
        {
            let is_selected = actual_index == config.selected_idx;

            queue!(out, terminal::Clear(ClearType::CurrentLine))?;
            if is_selected {
                queue!(
                    out,
                    SetForegroundColor(Color::Black),
                    crossterm::style::SetBackgroundColor(Color::White)
                )?;
            }

            Self::draw_node(node, *depth, &mut out)?;

            if is_selected {
                queue!(out, ResetColor)?;
            }
        }

        // Wipe any rows left over from a previous, taller frame, then pin the
        // footer to the bottom line (no trailing newline, to avoid scrolling).
        queue!(out, terminal::Clear(ClearType::FromCursorDown))?;
        queue!(out, cursor::MoveTo(0, terminal_height - 1))?;
        if config.search_mode && config.tree.is_empty() {
            write!(
                out,
                "No results found for \"{}\" | Press Esc to exit search\r",
                config.search_query
            )?;
        } else {
            write!(
                out,
                "Total Parameters: {} | Selected: {}/{} | Scroll: {} | Matches: {}\r",
                format_parameters(config.total_parameters),
                config.selected_idx + 1,
                config.tree.len(),
                new_scroll_offset,
                config.tree.len()
            )?;
        }

        out.flush()?;
        Ok(new_scroll_offset)
    }

    fn draw_node(node: &TreeNode, depth: usize, out: &mut impl Write) -> Result<()> {
        let indent = "  ".repeat(depth);

        match node {
            TreeNode::Group {
                name,
                children,
                expanded,
                tensor_count,
                total_size,
                stored_size,
            } => {
                let icon = if *expanded { "▼" } else { "▶" };
                // A repeated-block stack (e.g. a transformer's `layers` group)
                // has children that are all numbered subgroups; surface how many
                // there are so the depth is visible without expanding the tree.
                let layer_prefix = match layer_count(children) {
                    Some(1) => "1 layer, ".to_string(),
                    Some(n) => format!("{n} layers, "),
                    None => String::new(),
                };
                // When any descendant is compressed the on-disk total differs
                // from the logical total; show both, mirroring tensor rows.
                let size_field = if stored_size != total_size {
                    format!(
                        "{} → {}",
                        format_size(*total_size),
                        format_size(*stored_size)
                    )
                } else {
                    format_size(*total_size)
                };
                writeln!(
                    out,
                    "{}{} 📁 {} ({}{} tensors, {})\r",
                    indent, icon, name, layer_prefix, tensor_count, size_field
                )?;
            }
            TreeNode::Tensor { info } => {
                // In search mode (depth 0), show full name; otherwise show short name
                let display_name = if depth == 0 {
                    &info.name
                } else {
                    info.name.split('.').next_back().unwrap_or(&info.name)
                };
                // Size field: for compressed tensors show both the logical size
                // and the smaller on-disk size plus a codec marker; for formats
                // that track it, mark raw; otherwise just the size.
                let size_field = match &info.storage {
                    Storage::Unknown => format_size(info.size_bytes),
                    Storage::Raw => format!("{} (raw)", format_size(info.size_bytes)),
                    Storage::Compressed {
                        codec,
                        stored_bytes,
                    } => format!(
                        "{} → {} ({codec})",
                        format_size(info.size_bytes),
                        format_size(*stored_bytes)
                    ),
                };
                writeln!(
                    out,
                    "{}  📄 {} [{}, {}, {}]\r",
                    indent,
                    display_name,
                    info.dtype,
                    format_shape(&info.shape),
                    size_field
                )?;
            }
            TreeNode::Metadata { info } => {
                let truncated_value = if info.value.len() > 50 {
                    format!("{}...", &info.value[..47])
                } else {
                    info.value.clone()
                };
                writeln!(
                    out,
                    "{}  🏷️  {} [{}]: {}\r",
                    indent, info.name, info.value_type, truncated_value
                )?;
            }
        }
        Ok(())
    }

    pub fn draw_tensor_detail(tensor: &TensorInfo) -> Result<()> {
        let mut stdout = io::stdout();
        execute!(
            stdout,
            terminal::Clear(ClearType::All),
            cursor::MoveTo(0, 0)
        )?;

        writeln!(stdout, "Tensor Details\r")?;
        writeln!(stdout, "==============\r")?;
        writeln!(stdout, "Name: {}\r", tensor.name)?;
        writeln!(stdout, "Data Type: {}\r", tensor.dtype)?;
        writeln!(stdout, "Shape: {}\r", format_shape(&tensor.shape))?;
        writeln!(stdout, "Size: {}\r", format_size(tensor.size_bytes))?;
        writeln!(stdout, "\r")?;
        writeln!(stdout, "Press any key to return...\r")?;

        stdout.flush()?;
        Ok(())
    }

    pub fn draw_metadata_detail(metadata: &MetadataInfo) -> Result<()> {
        let mut stdout = io::stdout();
        execute!(
            stdout,
            terminal::Clear(ClearType::All),
            cursor::MoveTo(0, 0)
        )?;

        writeln!(stdout, "Metadata Details\r")?;
        writeln!(stdout, "================\r")?;
        writeln!(stdout, "Key: {}\r", metadata.name)?;
        writeln!(stdout, "Type: {}\r", metadata.value_type)?;
        writeln!(stdout, "Value:\r")?;

        // Handle multi-line values or long values
        let lines = metadata.value.lines();
        for line in lines.take(20) {
            // Limit to 20 lines
            writeln!(stdout, "  {line}\r")?;
        }

        writeln!(stdout, "\r")?;
        writeln!(stdout, "Press any key to return...\r")?;

        stdout.flush()?;
        Ok(())
    }
}

/// Returns the number of layers when `children` form a stack of numbered
/// subgroups (as in a transformer's `layers` group): there is at least one
/// subgroup and every subgroup has a purely numeric name. A single layer
/// counts too, so incomplete checkpoints still report their depth. Returns
/// `None` when the children are not such a stack.
fn layer_count(children: &[TreeNode]) -> Option<usize> {
    let mut numbered = 0;
    let mut groups = 0;
    for child in children {
        if let TreeNode::Group { name, .. } = child {
            groups += 1;
            if !name.is_empty() && name.chars().all(|c| c.is_ascii_digit()) {
                numbered += 1;
            }
        }
    }
    (groups > 0 && numbered == groups).then_some(numbered)
}
