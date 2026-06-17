use anyhow::Result;
use crossterm::{
    cursor, execute, queue,
    style::{Attribute, Color, ResetColor, SetAttribute, SetBackgroundColor, SetForegroundColor},
    terminal::{self, BeginSynchronizedUpdate, ClearType, EndSynchronizedUpdate},
};
use std::io::{self, BufWriter, Write};

use crate::health::HealthReport;
use crate::sample::{Sample, ViewDtype};
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
        // One bottom line for the status bar (the per-checkpoint totals now live
        // in the tree root instead of a footer).
        let footer_height = 1;
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
            queue!(out, SetForegroundColor(palette::DIM))?;
            write!(out, "Search ")?;
            queue!(out, ResetColor)?;
            input_box(&mut out, config.search_query, 16)?;
            write!(out, "  ")?;
            hint_line(&mut out, &[("Enter", "view"), ("Esc/q", "exit")])?;
            write!(out, "\r\n")?;
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

        // Status bar pinned to the bottom line (no trailing newline, to avoid
        // scrolling): the source file(s)/directory of the selected row, or the
        // empty-search message. Truncate keeping the tail so names stay visible.
        queue!(out, cursor::MoveTo(0, terminal_height.saturating_sub(1)))?;
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
                "{}",
                truncate_keep_end(config.status_bar, terminal_width as usize)
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
                params,
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
                if depth == 0 {
                    // The checkpoint root: summarise the whole model, including
                    // the total parameter count (which used to live in a footer).
                    writeln!(
                        out,
                        "{icon} 📦 {name} ({tensor_count} tensors, {} params, {size_field})\r",
                        format_parameters(*params)
                    )?;
                } else {
                    writeln!(
                        out,
                        "{}{} 📁 {} ({}{} tensors, {})\r",
                        indent, icon, name, layer_prefix, tensor_count, size_field
                    )?;
                }
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

    /// Draw the tensor detail screen. `view` is the active dtype reinterpretation
    /// (which changes the shown dtype, shape and parameter count); `overridable`
    /// gates the `d` hint. Rendered flicker-free so it can also serve as the
    /// live preview while choosing a dtype in the menu.
    pub fn draw_tensor_detail(
        tensor: &TensorInfo,
        view: ViewDtype,
        overridable: bool,
    ) -> Result<()> {
        let stdout = io::stdout();
        let mut out = BufWriter::new(stdout.lock());
        queue!(out, BeginSynchronizedUpdate, cursor::MoveTo(0, 0))?;

        write!(out, "Tensor Details")?;
        line_end(&mut out)?;
        write!(out, "==============")?;
        line_end(&mut out)?;
        write!(out, "Name: {}", tensor.name)?;
        line_end(&mut out)?;

        // Data type, with the active reinterpretation highlighted.
        write!(out, "Data Type: ")?;
        write_view_dtype(&mut out, &tensor.dtype, view)?;
        line_end(&mut out)?;

        // Shape and parameter count reflect the override (a packed view unpacks
        // several values per stored element, growing the last dimension).
        let shape = view.logical_shape(&tensor.shape, &tensor.dtype);
        let num_elements: usize = shape.iter().product();
        write!(out, "Shape: {}", format_shape(&shape))?;
        line_end(&mut out)?;
        write!(
            out,
            "Parameters: {} ({})",
            format_parameters(num_elements),
            with_thousands(num_elements)
        )?;
        line_end(&mut out)?;

        write!(out, "Size: {}", format_size(tensor.size_bytes))?;
        line_end(&mut out)?;
        // On-disk size + codec, for formats that track compression (HDF5).
        match &tensor.storage {
            Storage::Compressed {
                codec,
                stored_bytes,
            } => {
                write!(out, "On disk: {} ({codec})", format_size(*stored_bytes))?;
                line_end(&mut out)?;
            }
            Storage::Raw => {
                write!(
                    out,
                    "On disk: {} (uncompressed)",
                    format_size(tensor.size_bytes)
                )?;
                line_end(&mut out)?;
            }
            Storage::Unknown => {}
        }
        // Where the data lives within the file.
        match &tensor.layout {
            Layout::ByteRange { start, end } => {
                write!(
                    out,
                    "Data offsets: {} – {}  (within file data)",
                    with_thousands(*start as usize),
                    with_thousands(*end as usize)
                )?;
                line_end(&mut out)?;
            }
            Layout::Offset(offset) => {
                write!(
                    out,
                    "Data offset: {}  (within tensor data)",
                    with_thousands(*offset as usize)
                )?;
                line_end(&mut out)?;
            }
            Layout::Chunked { chunk, num_chunks } => {
                write!(
                    out,
                    "Chunks: {} × {}",
                    format_shape(chunk),
                    with_thousands(*num_chunks)
                )?;
                line_end(&mut out)?;
            }
            Layout::None => {}
        }
        write!(out, "File: {}", tensor.source_path)?;
        line_end(&mut out)?;
        line_end(&mut out)?;

        // Footer hints (keys highlighted).
        write!(out, "Press ")?;
        key_hint(&mut out, "m")?;
        write!(out, " for a heatmap, ")?;
        key_hint(&mut out, "v")?;
        write!(out, " for numeric values, ")?;
        if overridable {
            key_hint(&mut out, "d")?;
            write!(out, " to reinterpret the dtype, ")?;
        }
        write!(out, "any other key to return...")?;

        // No trailing newline (avoids scrolling); clear anything below.
        queue!(
            out,
            terminal::Clear(ClearType::FromCursorDown),
            EndSynchronizedUpdate
        )?;
        out.flush()?;
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

    /// Render a sampled tensor as a colored heatmap. Each character cell is an
    /// upper-half block `▀` whose foreground is the value above and background
    /// the value below, so one text row shows two data rows — doubling the
    /// vertical resolution (a terminal cell is ~twice as tall as it is wide).
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
        let integer = sample.view.is_integer(&tensor.dtype);
        let lo = fmt_value(sample.min, integer);
        let hi = fmt_value(sample.max, integer);
        write_view_dtype(&mut out, &tensor.dtype, sample.view)?;
        write!(
            out,
            " {} → sampled {}×{}, value range [{lo}, {hi}]",
            format_shape(&tensor.shape),
            sample.rows.len(),
            sample.cols.len(),
        )?;
        line_end(&mut out)?;
        if sample.slices > 1 {
            write_slice_header(&mut out, sample)?;
            line_end(&mut out)?;
        }
        line_end(&mut out)?;

        let range = sample.max - sample.min;
        let norm = |v: f64| {
            if range > 0.0 {
                (v - sample.min) / range
            } else {
                0.5
            }
        };
        // Two data rows per text line: foreground = the upper row's value,
        // background = the lower row's. A trailing odd row keeps the default
        // (dark) background for its empty lower half.
        let mut r = 0;
        while r < sample.values.len() {
            let top = &sample.values[r];
            let bottom = sample.values.get(r + 1);
            for (c, &tv) in top.iter().enumerate() {
                queue!(out, SetForegroundColor(heat_color(norm(tv))))?;
                match bottom {
                    Some(below) => queue!(out, SetBackgroundColor(heat_color(norm(below[c]))))?,
                    None => queue!(out, SetBackgroundColor(Color::Reset))?,
                }
                write!(out, "▀")?;
            }
            queue!(out, ResetColor)?;
            line_end(&mut out)?;
            r += 2;
        }

        line_end(&mut out)?;
        write!(out, "{lo} low ")?;
        for i in 0..24 {
            queue!(out, SetForegroundColor(heat_color(i as f64 / 23.0)))?;
            write!(out, "█")?;
        }
        queue!(out, ResetColor)?;
        write!(out, " high {hi}")?;
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
        write_view_dtype(&mut out, &tensor.dtype, sample.view)?;
        write!(
            out,
            " {} → sampled {} of {} rows × {} of {} cols (indices shown)",
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

        // Integer dtypes print as plain integers; floats use scientific notation.
        let integer = sample.view.is_integer(&tensor.dtype);
        for (i, row) in sample.values.iter().enumerate() {
            write!(out, "{:>6} ", sample.rows[i])?;
            for &v in row {
                if integer {
                    write!(out, "{:>W$}", v as i64)?;
                } else {
                    write!(out, "{v:>W$.3e}")?;
                }
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

    /// Overlay a dtype-selection menu on the bottom two lines of a data view: a
    /// strip of the available views with `current` highlighted, plus a hint
    /// line. The data view behind it is a live preview of the highlighted view.
    pub fn draw_dtype_menu(options: &[ViewDtype], current: usize) -> Result<()> {
        let stdout = io::stdout();
        let mut out = BufWriter::new(stdout.lock());
        let (_w, h) = terminal::size()?;

        queue!(
            out,
            cursor::MoveTo(0, h.saturating_sub(2)),
            terminal::Clear(ClearType::CurrentLine),
            SetForegroundColor(palette::DIM)
        )?;
        write!(out, "view as:")?;
        queue!(out, ResetColor)?;
        for (i, opt) in options.iter().enumerate() {
            if i == current {
                // The selected entry is a highlighted "button".
                queue!(
                    out,
                    SetAttribute(Attribute::Bold),
                    SetForegroundColor(palette::SELECT_FG),
                    SetBackgroundColor(palette::SELECT_BG)
                )?;
                write!(out, " {} ", opt.menu_label())?;
                queue!(out, ResetColor, SetAttribute(Attribute::Reset))?;
            } else {
                queue!(out, SetForegroundColor(palette::DIM))?;
                write!(out, " {} ", opt.menu_label())?;
                queue!(out, ResetColor)?;
            }
        }

        queue!(
            out,
            cursor::MoveTo(0, h.saturating_sub(1)),
            terminal::Clear(ClearType::CurrentLine)
        )?;
        hint_line(
            &mut out,
            &[
                ("← → or d/D", "move"),
                ("Enter", "apply"),
                ("Esc", "cancel"),
            ],
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
        input_box(&mut out, input, 5)?;
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
    if sample.overridable {
        items.push(("d", "dtype"));
    }
    items.push(("", "any other key to return..."));
    hint_line(out, &items)
}

/// Render a text-input field: the typed `text` plus a block cursor, on the
/// input palette colours, padded to at least `min_chars` characters wide. Used
/// by the search bar and the slice-jump prompt so every input box matches.
fn input_box(out: &mut impl Write, text: &str, min_chars: usize) -> Result<()> {
    queue!(
        out,
        SetBackgroundColor(palette::INPUT_BG),
        SetForegroundColor(palette::INPUT_FG)
    )?;
    write!(out, " {text}█")?;
    for _ in text.chars().count()..min_chars {
        write!(out, " ")?;
    }
    write!(out, " ")?;
    queue!(out, ResetColor)?;
    Ok(())
}

/// Format a heatmap legend / range value: integers without a fractional part,
/// floats with four decimals.
fn fmt_value(v: f64, integer: bool) -> String {
    if integer {
        format!("{v:.0}")
    } else {
        format!("{v:.4}")
    }
}

/// Write the dtype shown in a data-view header. With no override this is just
/// the stored dtype; when overridden it fades the original dtype and highlights
/// the active reinterpretation, e.g. a dimmed `BF16 as` then a bold `u4 (packed)`.
fn write_view_dtype(out: &mut impl Write, stored: &str, view: ViewDtype) -> Result<()> {
    match view.label() {
        Some(label) => {
            queue!(out, SetForegroundColor(palette::DIM))?;
            write!(out, "{stored} as ")?;
            queue!(
                out,
                ResetColor,
                SetAttribute(Attribute::Bold),
                SetForegroundColor(palette::KEY)
            )?;
            write!(out, "{label}")?;
            queue!(out, ResetColor, SetAttribute(Attribute::Reset))?;
        }
        None => write!(out, "{stored}")?,
    }
    Ok(())
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
