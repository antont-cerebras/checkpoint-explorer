use anyhow::Result;
use crossterm::{
    cursor, execute, queue,
    style::{Attribute, Color, ResetColor, SetAttribute, SetBackgroundColor, SetForegroundColor},
    terminal::{self, BeginSynchronizedUpdate, ClearType, EndSynchronizedUpdate},
};
use std::io::{self, BufWriter, Write};

use crate::health::HealthReport;
use crate::sample::Sample;
use crate::tree::{Layout, MetadataInfo, Storage, TensorInfo, TreeNode};
use crate::utils::{format_parameters, format_shape, format_size};

/// The app's colour palette — the single source of truth for how each kind of
/// thing is styled, so the same role looks the same on every screen. Change a
/// colour here and it updates everywhere it's used.
mod palette {
    use crossterm::style::Color;

    /// Interactive keys in hint lines (rendered bold, via [`super::key_hint`]).
    pub const KEY: Color = Color::Cyan;
    /// Secondary / de-emphasised hint text (ranges, "to cancel", …).
    pub const DIM: Color = Color::DarkGrey;
    /// Selected tree row (foreground on background).
    pub const SELECT_FG: Color = Color::Black;
    pub const SELECT_BG: Color = Color::White;
    /// The slice-jump input box (foreground on background).
    pub const INPUT_FG: Color = Color::White;
    pub const INPUT_BG: Color = Color::DarkBlue;
    /// Something missing / wrong / out of range.
    pub const ERROR: Color = Color::Red;
    /// Something present but unexpected (a softer alert than [`ERROR`]).
    pub const WARN: Color = Color::Yellow;
}

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
    /// Bottom status line: source file(s) of the selected row, or a copy
    /// confirmation.
    pub status_bar: &'a str,
    /// Whether a checkpoint health issue was detected (shows a header hint to
    /// press `h` for the report).
    pub health_warning: bool,
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

        let (terminal_width, terminal_height) = terminal::size()?;
        let header_height = 3;
        let footer_height = 2;
        let available_height =
            (terminal_height as usize).saturating_sub(header_height + footer_height);

        queue!(out, BeginSynchronizedUpdate, cursor::MoveTo(0, 0))?;

        // Header
        queue!(out, terminal::Clear(ClearType::CurrentLine))?;
        write!(
            out,
            "Checkpoint Explorer - {} ({}/{})",
            config.current_file,
            config.file_idx + 1,
            config.total_files
        )?;
        if config.health_warning {
            queue!(out, SetForegroundColor(palette::ERROR))?;
            write!(out, "   ⚠ index/file mismatch — press ")?;
            key_hint(&mut out, "h")?;
            queue!(out, ResetColor)?;
        }
        write!(out, "\r\n")?;
        queue!(out, terminal::Clear(ClearType::CurrentLine))?;
        if config.search_mode {
            let query = if config.search_query.is_empty() {
                "_"
            } else {
                config.search_query
            };
            write!(out, "SEARCH MODE: {query} | Type to search, ")?;
            key_hint(&mut out, "Enter")?;
            write!(out, " to view, ")?;
            key_hint(&mut out, "Esc/q")?;
            write!(out, " to exit\r\n")?;
        } else {
            hint_line(
                &mut out,
                &[
                    ("↑/↓", "navigate"),
                    ("←/→", "parent/child"),
                    ("Shift+↑/↓", "sibling"),
                    ("Enter/Space", "expand"),
                    ("E/C", "all"),
                    ("/", "search"),
                    ("c", "copy"),
                    ("q", "quit"),
                ],
            )?;
            write!(out, "\r\n")?;
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
                    SetForegroundColor(palette::SELECT_FG),
                    SetBackgroundColor(palette::SELECT_BG)
                )?;
            }

            Self::draw_node(node, *depth, &mut out)?;

            if is_selected {
                queue!(out, ResetColor)?;
            }
        }

        // Wipe any rows left over from a previous, taller frame.
        queue!(out, terminal::Clear(ClearType::FromCursorDown))?;

        // Status bar (line above the footer): source file of the selected row.
        // Truncate keeping the tail so the file name stays visible.
        queue!(out, cursor::MoveTo(0, terminal_height.saturating_sub(2)))?;
        write!(
            out,
            "{}",
            truncate_keep_end(config.status_bar, terminal_width as usize)
        )?;

        // Footer pinned to the bottom line (no trailing newline, to avoid scrolling).
        queue!(out, cursor::MoveTo(0, terminal_height - 1))?;
        if config.search_mode && config.tree.is_empty() {
            write!(
                out,
                "No results found for \"{}\" | Press ",
                config.search_query
            )?;
            key_hint(&mut out, "Esc")?;
            write!(out, " to exit search\r")?;
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

        queue!(out, EndSynchronizedUpdate)?;
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
        writeln!(
            stdout,
            "Parameters: {} ({})\r",
            format_parameters(tensor.num_elements),
            with_thousands(tensor.num_elements)
        )?;
        writeln!(stdout, "Size: {}\r", format_size(tensor.size_bytes))?;
        // On-disk size + codec, for formats that track compression (HDF5).
        match &tensor.storage {
            Storage::Compressed {
                codec,
                stored_bytes,
            } => {
                writeln!(
                    stdout,
                    "On disk: {} ({codec})\r",
                    format_size(*stored_bytes)
                )?;
            }
            Storage::Raw => {
                writeln!(
                    stdout,
                    "On disk: {} (uncompressed)\r",
                    format_size(tensor.size_bytes)
                )?;
            }
            Storage::Unknown => {}
        }
        // Where the data lives within the file.
        match &tensor.layout {
            Layout::ByteRange { start, end } => {
                writeln!(
                    stdout,
                    "Data offsets: {} – {}  (within file data)\r",
                    with_thousands(*start as usize),
                    with_thousands(*end as usize)
                )?;
            }
            Layout::Offset(offset) => {
                writeln!(
                    stdout,
                    "Data offset: {}  (within tensor data)\r",
                    with_thousands(*offset as usize)
                )?;
            }
            Layout::Chunked { chunk, num_chunks } => {
                writeln!(
                    stdout,
                    "Chunks: {} × {}\r",
                    format_shape(chunk),
                    with_thousands(*num_chunks)
                )?;
            }
            Layout::None => {}
        }
        writeln!(stdout, "File: {}\r", tensor.source_path)?;
        writeln!(stdout, "\r")?;
        // Highlight the `m` / `v` keys so they stand out from the prose.
        write!(stdout, "Press ")?;
        key_hint(&mut stdout, "m")?;
        write!(stdout, " for a heatmap, ")?;
        key_hint(&mut stdout, "v")?;
        writeln!(stdout, " for numeric values, any other key to return...\r")?;

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

    /// Render a sampled tensor as an ASCII heatmap (one colored block per
    /// sampled element, colored by relative value).
    pub fn draw_heatmap(tensor: &TensorInfo, sample: &Sample) -> Result<()> {
        let stdout = io::stdout();
        let mut out = BufWriter::new(stdout.lock());
        // Present the whole frame atomically (the terminal buffers everything
        // between Begin/End and paints it in one go, so a redraw never shows a
        // half-updated screen — this is what eliminates the flicker). We also
        // overwrite in place: write each line's new content first, then clear
        // only the leftover tail (`line_end`), and never emit a trailing
        // newline (which could scroll the screen and flash).
        queue!(out, BeginSynchronizedUpdate, cursor::MoveTo(0, 0))?;

        write!(out, "Heatmap: {}", tensor.name)?;
        line_end(&mut out)?;
        write!(
            out,
            "{} {} → sampled {}×{}, value range [{:.4}, {:.4}]",
            tensor.dtype,
            format_shape(&tensor.shape),
            sample.rows.len(),
            sample.cols.len(),
            sample.min,
            sample.max
        )?;
        line_end(&mut out)?;
        if sample.slices > 1 {
            write_slice_header(&mut out, sample)?;
            line_end(&mut out)?;
        }
        line_end(&mut out)?;

        let range = sample.max - sample.min;
        for row in &sample.values {
            for &v in row {
                let t = if range > 0.0 {
                    (v - sample.min) / range
                } else {
                    0.5
                };
                queue!(out, SetForegroundColor(heat_color(t)))?;
                write!(out, "█")?;
            }
            queue!(out, ResetColor)?;
            line_end(&mut out)?;
        }

        line_end(&mut out)?;
        write!(out, "{:.4} low ", sample.min)?;
        for i in 0..24 {
            queue!(out, SetForegroundColor(heat_color(i as f64 / 23.0)))?;
            write!(out, "█")?;
        }
        queue!(out, ResetColor)?;
        write!(out, " high {:.4}", sample.max)?;
        line_end(&mut out)?;

        line_end(&mut out)?;
        write_view_footer(&mut out, sample, true)?;

        // Clear the footer's tail and everything below (no trailing newline),
        // then end the synchronized frame.
        queue!(
            out,
            terminal::Clear(ClearType::FromCursorDown),
            EndSynchronizedUpdate
        )?;
        out.flush()?;
        Ok(())
    }

    /// Render a sampled tensor as a grid of numeric values with row/column
    /// indices (edges included).
    pub fn draw_values(tensor: &TensorInfo, sample: &Sample) -> Result<()> {
        const W: usize = 11;
        let stdout = io::stdout();
        let mut out = BufWriter::new(stdout.lock());
        // Synchronized, in-place overwrite (see `draw_heatmap`) to avoid flicker.
        queue!(out, BeginSynchronizedUpdate, cursor::MoveTo(0, 0))?;

        write!(out, "Values: {}", tensor.name)?;
        line_end(&mut out)?;
        write!(
            out,
            "{} {} → sampled {} of {} rows × {} of {} cols (indices shown)",
            tensor.dtype,
            format_shape(&tensor.shape),
            sample.rows.len(),
            sample.total_rows,
            sample.cols.len(),
            sample.total_cols
        )?;
        line_end(&mut out)?;
        if sample.slices > 1 {
            write_slice_header(&mut out, sample)?;
            line_end(&mut out)?;
        }
        line_end(&mut out)?;

        // Column-index header.
        write!(out, "{:>6} ", "")?;
        for &c in &sample.cols {
            write!(out, "{c:>W$}")?;
        }
        line_end(&mut out)?;

        for (i, row) in sample.values.iter().enumerate() {
            write!(out, "{:>6} ", sample.rows[i])?;
            for &v in row {
                write!(out, "{v:>W$.3e}")?;
            }
            line_end(&mut out)?;
        }

        line_end(&mut out)?;
        write_view_footer(&mut out, sample, false)?;

        queue!(
            out,
            terminal::Clear(ClearType::FromCursorDown),
            EndSynchronizedUpdate
        )?;
        out.flush()?;
        Ok(())
    }

    /// A prompt pinned to the bottom of the screen for jumping to a slice by
    /// typing its index (overlaid on the current data view). The label and a
    /// fixed-width input box are colored; `error`, when set, is shown in red on
    /// the line below (e.g. for an out-of-range index).
    pub fn draw_slice_prompt(slices: usize, input: &str, error: Option<&str>) -> Result<()> {
        let stdout = io::stdout();
        let mut out = BufWriter::new(stdout.lock());
        let (_w, h) = terminal::size()?;

        // Prompt line.
        queue!(
            out,
            cursor::MoveTo(0, h.saturating_sub(2)),
            terminal::Clear(ClearType::CurrentLine),
            SetForegroundColor(palette::KEY)
        )?;
        write!(out, "Go to slice ")?;
        queue!(out, SetForegroundColor(palette::DIM))?;
        write!(out, "(0-{} or 0-100%)", slices.saturating_sub(1))?;
        queue!(out, ResetColor)?;
        write!(out, "  ")?;
        // The input box: a fixed-width highlighted field with a block cursor.
        queue!(
            out,
            SetBackgroundColor(palette::INPUT_BG),
            SetForegroundColor(palette::INPUT_FG)
        )?;
        write!(out, " {input}█ ")?;
        for _ in input.len()..5 {
            write!(out, " ")?;
        }
        queue!(out, ResetColor)?;
        write!(out, "  ")?;
        key_hint(&mut out, "Enter")?;
        queue!(out, SetForegroundColor(palette::DIM))?;
        write!(out, " to jump · ")?;
        queue!(out, ResetColor)?;
        key_hint(&mut out, "Esc")?;
        queue!(out, SetForegroundColor(palette::DIM))?;
        write!(out, " to cancel")?;
        queue!(out, ResetColor)?;

        // Feedback line below (out-of-range / invalid input).
        queue!(
            out,
            cursor::MoveTo(0, h.saturating_sub(1)),
            terminal::Clear(ClearType::CurrentLine)
        )?;
        if let Some(msg) = error {
            queue!(out, SetForegroundColor(palette::ERROR))?;
            write!(out, "{msg}")?;
            queue!(out, ResetColor)?;
        }

        out.flush()?;
        Ok(())
    }

    /// A simple full-screen message (e.g. when a data preview is unavailable).
    pub fn draw_message(title: &str, message: &str) -> Result<()> {
        let mut stdout = io::stdout();
        execute!(
            stdout,
            terminal::Clear(ClearType::All),
            cursor::MoveTo(0, 0)
        )?;
        writeln!(stdout, "{title}\r")?;
        writeln!(stdout, "{}\r", "=".repeat(title.len().max(10)))?;
        writeln!(stdout, "{message}\r")?;
        writeln!(stdout, "\r")?;
        writeln!(stdout, "Press any key to return...\r")?;
        stdout.flush()?;
        Ok(())
    }

    /// Draw a full-screen warning panel summarising checkpoint health issues,
    /// shown once at startup. Each category is capped so the panel stays small.
    pub fn draw_health_warning(reports: &[HealthReport]) -> Result<()> {
        let mut stdout = io::stdout();
        execute!(
            stdout,
            terminal::Clear(ClearType::All),
            cursor::MoveTo(0, 0)
        )?;

        execute!(stdout, SetForegroundColor(palette::WARN))?;
        writeln!(stdout, "⚠  Checkpoint health check\r")?;
        writeln!(stdout, "{}\r", "=".repeat(60))?;
        execute!(stdout, ResetColor)?;

        for report in reports {
            writeln!(stdout, "\r")?;
            execute!(stdout, SetForegroundColor(palette::WARN))?;
            writeln!(
                stdout,
                "{} does not match the .safetensors files on disk.\r",
                report.index_path
            )?;
            execute!(stdout, ResetColor)?;
            writeln!(stdout, "\r")?;
            // "missing" issues are red (something is gone), "extra" issues are
            // yellow (present but unexpected).
            health_section(
                &mut stdout,
                "Referenced by the index but MISSING",
                &report.missing_files,
                palette::ERROR,
            )?;
            health_section(
                &mut stdout,
                "Present on disk but NOT in the index",
                &report.extra_files,
                palette::WARN,
            )?;
            health_section(
                &mut stdout,
                "Expected by the index but absent from their file",
                &report.missing_tensors,
                palette::ERROR,
            )?;
            health_section(
                &mut stdout,
                "In files but not listed in the index",
                &report.extra_tensors,
                palette::WARN,
            )?;
        }

        execute!(stdout, SetForegroundColor(palette::DIM))?;
        writeln!(
            stdout,
            "The explorer scans the directory directly when the index is stale. Press any key to return.\r"
        )?;
        execute!(stdout, ResetColor)?;

        stdout.flush()?;
        Ok(())
    }
}

/// Write a key name highlighted (bold bright-cyan) so it stands out from the
/// surrounding prose in a hint line. Uses `queue!` so it composes inside a
/// buffered frame; the caller is responsible for flushing.
fn key_hint(out: &mut impl Write, key: &str) -> Result<()> {
    queue!(
        out,
        SetAttribute(Attribute::Bold),
        SetForegroundColor(palette::KEY)
    )?;
    write!(out, "{key}")?;
    queue!(out, ResetColor, SetAttribute(Attribute::Reset))?;
    Ok(())
}

/// Render a hint line as `key label · key label · …`, highlighting each key.
/// An item with an empty key is written as a plain segment (e.g. a trailing
/// "any other key to return"); an empty label writes just the key.
fn hint_line(out: &mut impl Write, items: &[(&str, &str)]) -> Result<()> {
    for (i, (key, label)) in items.iter().enumerate() {
        if i > 0 {
            write!(out, " · ")?;
        }
        if key.is_empty() {
            write!(out, "{label}")?;
        } else {
            key_hint(out, key)?;
            if !label.is_empty() {
                write!(out, " {label}")?;
            }
        }
    }
    Ok(())
}

/// Finish the current line: clear any leftover tail (so a shorter new line
/// doesn't leave stale characters), then move to the start of the next line.
/// Writing content *before* this clear is what keeps redraws flicker-free.
fn line_end(out: &mut impl Write) -> Result<()> {
    queue!(out, terminal::Clear(ClearType::UntilNewLine))?;
    write!(out, "\r\n")?;
    Ok(())
}

/// For a multi-slice (3D) tensor, write the line announcing which 2D slice is
/// shown and how to change it (keys highlighted, no trailing newline — the
/// caller ends the line). Only called when `sample.slices > 1`.
fn write_slice_header(out: &mut impl Write, sample: &Sample) -> Result<()> {
    write!(
        out,
        "slice {} of {} (fixed leading index) — ",
        sample.slice, sample.slices
    )?;
    hint_line(
        out,
        &[
            ("← →", "step"),
            ("Shift+← →", "jump 5% (both wrap)"),
            ("/", "index or %"),
        ],
    )
}

/// Footer for the data views: offers the other representation (`m`/`v` switch
/// in place, no trip back to the detail screen) and mentions slice navigation
/// only when there is more than one slice to move between. Keys highlighted.
fn write_view_footer(out: &mut impl Write, sample: &Sample, heatmap: bool) -> Result<()> {
    let switch = if heatmap {
        ("v", "numeric values")
    } else {
        ("m", "heatmap")
    };
    let mut items = vec![switch];
    if sample.slices > 1 {
        items.push(("← →", "step"));
        items.push(("Shift+← →", "jump 5%"));
        items.push(("/", "index or %"));
    }
    items.push(("", "any other key to return..."));
    hint_line(out, &items)
}

/// Write one capped section of the health panel: a titled list (in `color`) of
/// up to `CAP` items, then a dimmed "… and N more" when truncated.
fn health_section(
    stdout: &mut io::Stdout,
    title: &str,
    items: &[String],
    color: Color,
) -> Result<()> {
    if items.is_empty() {
        return Ok(());
    }
    const CAP: usize = 6;
    execute!(stdout, SetForegroundColor(color))?;
    writeln!(stdout, "{title} ({}):\r", items.len())?;
    execute!(stdout, ResetColor)?;
    for item in items.iter().take(CAP) {
        writeln!(stdout, "  {item}\r")?;
    }
    if items.len() > CAP {
        execute!(stdout, SetForegroundColor(palette::DIM))?;
        writeln!(stdout, "  … and {} more\r", items.len() - CAP)?;
        execute!(stdout, ResetColor)?;
    }
    writeln!(stdout, "\r")?;
    Ok(())
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

/// Truncate `s` to at most `width` characters, keeping the END (so a path's
/// file name stays visible) and prefixing `…` when truncated.
fn truncate_keep_end(s: &str, width: usize) -> String {
    let count = s.chars().count();
    if count <= width {
        return s.to_string();
    }
    if width == 0 {
        return String::new();
    }
    let tail: String = s.chars().skip(count - (width - 1)).collect();
    format!("…{tail}")
}

/// Map a normalized value in `[0, 1]` to a blue→green→red 256-color ramp
/// (the 6×6×6 ANSI color cube, indices 16..=231).
fn heat_color(t: f64) -> Color {
    let t = t.clamp(0.0, 1.0);
    let r = (t * 5.0).round() as u8;
    let b = ((1.0 - t) * 5.0).round() as u8;
    let g = ((1.0 - (t - 0.5).abs() * 2.0) * 5.0).round() as u8;
    Color::AnsiValue(16 + 36 * r + 6 * g + b)
}

/// Format an integer with thousands separators (e.g. 579133440 -> "579,133,440").
fn with_thousands(n: usize) -> String {
    let digits = n.to_string();
    let len = digits.len();
    let mut out = String::with_capacity(len + len / 3);
    for (i, ch) in digits.chars().enumerate() {
        if i > 0 && (len - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}
