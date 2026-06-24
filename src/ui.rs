use anyhow::Result;
use crossterm::{
    cursor, execute, queue,
    style::{Attribute, Color, ResetColor, SetAttribute, SetBackgroundColor, SetForegroundColor},
    terminal::{self, BeginSynchronizedUpdate, ClearType, EndSynchronizedUpdate},
};
use std::collections::HashSet;
use std::io::{self, BufWriter, Write};
use std::time::Duration;

use crate::health::HealthReport;
use crate::sample::{Sample, SampleMode, Stats, ViewDtype};
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
    /// The bottom status bar (foreground on background).
    pub const STATUS_FG: Color = Color::White;
    pub const STATUS_BG: Color = Color::DarkGrey;
    /// A success accent (e.g. the "copied" status bar): dark text on green, as
    /// a light foreground is hard to read on the bright green.
    pub const OK_BG: Color = Color::Green;
    pub const OK_FG: Color = Color::Black;
    /// A success accent used as a *foreground* (e.g. the "✓ copied" confirmation).
    pub const SUCCESS: Color = Color::Green;
    /// Marks a tensor present on disk but missing from the index — a vivid red
    /// that stands out clearly against the tree's default and dimmed text.
    pub const UNINDEXED: Color = Color::AnsiValue(196);
    /// Group names and expand arrows in the tree — the primary accent (a bright
    /// sky-cyan), so the structure stands out from the leaf tensors.
    pub const ACCENT: Color = Color::AnsiValue(81);
    /// A tensor's data type (warm amber, so the type pops).
    pub const DTYPE: Color = Color::AnsiValue(215);
    /// Zebra striping for the numeric grid — two subtle dark backgrounds (one
    /// "dark", one "less dark") that alternate to guide the eye along the rows
    /// or columns, like a dim highlighter.
    pub const STRIPE_DARK: Color = Color::AnsiValue(234);
    pub const STRIPE_LITE: Color = Color::AnsiValue(237);
}

/// Marks a tensor that's on disk but not listed in the index (an "extra"),
/// shown in [`palette::UNINDEXED`] in the tree, detail screen and legends.
const UNINDEXED_MARK: &str = "✚";

pub struct DrawConfig<'a> {
    pub tree: &'a [(TreeNode, usize)],
    pub current_file: &'a str,
    pub file_idx: usize,
    pub total_files: usize,
    pub selected_idx: usize,
    pub scroll_offset: usize,
    pub search_mode: bool,
    pub search_query: &'a str,
    /// Leading glyph for the status bar (e.g. `▪`, `▸`, `✓`), and whether it
    /// is a success message (the copy confirmation) for accent colouring.
    pub status_icon: &'a str,
    pub status_ok: bool,
    /// Bottom status line: source file(s)/directory of the selected row, or a
    /// copy confirmation.
    pub status_bar: &'a str,
    /// Whether a checkpoint health issue was detected (shows a header hint to
    /// press `h` for the report).
    pub health_warning: bool,
    /// Whether the loaded checkpoint can be repacked (a single HDF5 file), which
    /// gates the `R` hint.
    pub can_repack: bool,
    /// `source_path`s of tensors present on disk but not listed in the index
    /// (a stale `model.safetensors.index.json`), flagged in the tree.
    pub unindexed: &'a HashSet<String>,
}

/// How a screen should render the statistics area: not computed yet, a scan in
/// progress (with a spinner + running timer), or the finished `Stats`.
#[derive(Clone, Copy)]
pub enum StatsView<'a> {
    Pending,
    Computing {
        spinner: char,
        elapsed: Duration,
        /// Fraction scanned so far (`0.0..=1.0`) for the progress bar, or `None`
        /// when unknown (then only the spinner + timer show).
        progress: Option<f64>,
    },
    Ready(&'a Stats),
}

impl StatsView<'_> {
    /// The exact whole-tensor value range, available only once the scan has
    /// finished. Used to size numeric cells to the data actually present.
    pub fn value_range(&self) -> Option<(f64, f64)> {
        match self {
            StatsView::Ready(s) => Some((s.min, s.max)),
            _ => None,
        }
    }
}

/// The numeric grid's zebra striping: a subtle alternating background down the
/// rows, down the columns, or none. Cycled with `z`.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum StripeMode {
    #[default]
    Rows,
    Cols,
    Off,
}

impl StripeMode {
    /// The next mode in the `z` cycle: rows → cols → off → rows.
    pub fn next(self) -> Self {
        match self {
            StripeMode::Rows => StripeMode::Cols,
            StripeMode::Cols => StripeMode::Off,
            StripeMode::Off => StripeMode::Rows,
        }
    }
}

/// Parse a CLI `--zebra` value (`rows`, `cols`, or `off`) into a [`StripeMode`].
pub fn parse_stripe_mode(s: &str) -> Result<StripeMode, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "rows" | "row" => Ok(StripeMode::Rows),
        "cols" | "col" | "columns" | "column" => Ok(StripeMode::Cols),
        "off" | "none" => Ok(StripeMode::Off),
        _ => Err(format!(
            "unknown zebra mode '{s}'; expected rows, cols, or off"
        )),
    }
}

/// Which screen a legend explains. The legend (`l`) is context-sensitive — it
/// lists only the glyphs and colour cues that appear on the screen it was opened
/// from.
#[derive(Clone, Copy)]
pub enum Legend {
    Tree,
    Detail,
    Heatmap,
    Values,
}

pub struct UI;

impl UI {
    /// Render the tree browser into `out` (a buffered stdout for the live
    /// screen, or an in-memory buffer when capturing the screen for copy).
    /// Writing the whole frame at once and flushing once — combined with
    /// overwriting in place (clearing each line rather than the whole screen up
    /// front) — removes the flicker a per-frame `Clear(All)` produced.
    pub fn draw_screen(out: &mut impl Write, config: &DrawConfig) -> Result<usize> {
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
            key_hint(&mut *out, "h")?;
            queue!(out, ResetColor)?;
        }
        write!(out, "\r\n")?;
        queue!(out, terminal::Clear(ClearType::CurrentLine))?;
        if config.search_mode {
            queue!(out, SetForegroundColor(palette::DIM))?;
            write!(out, "Search ")?;
            queue!(out, ResetColor)?;
            input_box(&mut *out, config.search_query, 16)?;
            write!(out, "  ")?;
            hint_line(&mut *out, &[("Enter", "view"), ("Esc/q", "exit")])?;
            write!(out, "\r\n")?;
        } else {
            let mut hints: Vec<(&str, &str)> = vec![
                ("↑/↓", "navigate"),
                ("←/→", "parent/child"),
                ("Shift+↑/↓", "sibling"),
                ("Enter/Space", "expand"),
                ("E/C", "all"),
                ("/", "search"),
                ("l", "legend"),
                ("c", "copy screen"),
                ("f", "copy file"),
                ("y", "copy command"),
                ("⌫/\\", "back/fwd"),
            ];
            if config.can_repack {
                hints.push(("R", "repack"));
            }
            hints.push(("q", "quit"));
            hint_line(&mut *out, &hints)?;
            write!(out, "\r\n")?;
        }
        queue!(
            out,
            terminal::Clear(ClearType::CurrentLine),
            SetForegroundColor(palette::DIM)
        )?;
        write!(out, "{}\r\n", "─".repeat(terminal_width as usize))?;
        queue!(out, ResetColor)?;

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

            Self::draw_node(node, *depth, is_selected, config.unindexed, &mut *out)?;

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
            key_hint(&mut *out, "Esc")?;
            write!(out, " to exit search\r")?;
        } else if !config.status_bar.is_empty() {
            // A colored chip: leading glyph + the path/text, truncated tail-first
            // so the file name stays visible. Dark-on-green for a copy success,
            // light-on-grey otherwise.
            let (bg, fg) = if config.status_ok {
                (palette::OK_BG, palette::OK_FG)
            } else {
                (palette::STATUS_BG, palette::STATUS_FG)
            };
            let text = truncate_keep_end(
                config.status_bar,
                (terminal_width as usize).saturating_sub(6),
            );
            queue!(out, SetBackgroundColor(bg), SetForegroundColor(fg))?;
            write!(out, " {} {text} ", config.status_icon)?;
            queue!(out, ResetColor)?;
        }

        queue!(out, EndSynchronizedUpdate)?;
        out.flush()?;
        Ok(new_scroll_offset)
    }

    /// Render one tree row. `selected` rows are drawn plain so the caller's
    /// highlight (inverse video) reads cleanly; other rows are colour-coded by
    /// kind — group names in the accent and dtypes amber, with the name, shape
    /// and size at full strength and only the leaf marker / storage tag dimmed.
    fn draw_node(
        node: &TreeNode,
        depth: usize,
        selected: bool,
        unindexed: &HashSet<String>,
        out: &mut impl Write,
    ) -> Result<()> {
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
                // ▾ open / ▸ collapsed — the arrow doubles as the folder marker.
                let arrow = if *expanded { "▾" } else { "▸" };
                // A repeated-block stack (e.g. a transformer's `layers` group)
                // has children that are all numbered subgroups; the `☰` glyph
                // counts them, `▦` the tensors — so depth is visible without
                // expanding. When any descendant is compressed the on-disk total
                // differs from the logical total; show both, mirroring tensors.
                let layer_prefix = match layer_count(children) {
                    Some(n) => format!("☰ {n}, "),
                    None => String::new(),
                };
                let size_field = if stored_size != total_size {
                    format!(
                        "{} → {}",
                        format_size(*total_size),
                        format_size(*stored_size)
                    )
                } else {
                    format_size(*total_size)
                };
                write!(out, "{indent}")?;
                paint(out, selected, palette::ACCENT, arrow)?;
                write!(out, " ")?;
                paint(out, selected, palette::ACCENT, name)?;
                let meta = if depth == 0 {
                    // The checkpoint root: summarise the whole model, including
                    // the total parameter count (which used to live in a footer).
                    format!(
                        " (▦ {tensor_count}, {} params, {size_field})",
                        format_parameters(*params)
                    )
                } else {
                    format!(" ({layer_prefix}▦ {tensor_count}, {size_field})")
                };
                write!(out, "{meta}\r\n")?;
            }
            TreeNode::Tensor { info } => {
                // In search mode (depth 0), show full name; otherwise short name.
                let display_name = if depth == 0 {
                    &info.name
                } else {
                    // The last path component, treating `.` and `__` as separators
                    // (so `…_down_proj_weight__variant` shows just `variant`).
                    let after = info.name.rsplit("__").next().unwrap_or(&info.name);
                    after.rsplit('.').next().unwrap_or(after)
                };
                // The name, shape and size read at full strength; only the leaf
                // marker and the storage tag (codec / "raw") are dimmed, and the
                // dtype is tinted. `⇩` marks a compressed tensor. A tensor on disk
                // but absent from the index gets a red `✚` (an "extra") instead of
                // the dot.
                write!(out, "{indent}  ")?;
                if unindexed.contains(&info.source_path) {
                    paint(out, selected, palette::UNINDEXED, UNINDEXED_MARK)?;
                } else {
                    paint(out, selected, palette::DIM, "·")?;
                }
                write!(out, " {display_name} [")?;
                paint(out, selected, palette::DTYPE, &info.dtype)?;
                write!(out, ", {}, ", format_shape(&info.shape))?;
                match &info.storage {
                    Storage::Compressed {
                        codec,
                        stored_bytes,
                    } => {
                        write!(
                            out,
                            "{} → {} ",
                            format_size(info.size_bytes),
                            format_size(*stored_bytes)
                        )?;
                        paint(out, selected, palette::DIM, &format!("(⇩ {codec})"))?;
                    }
                    Storage::Raw => {
                        write!(out, "{} ", format_size(info.size_bytes))?;
                        paint(out, selected, palette::DIM, "(raw)")?;
                    }
                    Storage::Unknown => write!(out, "{}", format_size(info.size_bytes))?,
                }
                write!(out, "]\r\n")?;
            }
            TreeNode::Metadata { info } => {
                let truncated_value = if info.value.len() > 50 {
                    format!("{}...", &info.value[..47])
                } else {
                    info.value.clone()
                };
                write!(out, "{indent}  ")?;
                paint(out, selected, palette::DIM, "≡")?;
                write!(out, " {}", info.name)?;
                paint(
                    out,
                    selected,
                    palette::DIM,
                    &format!(" [{}]: {truncated_value}", info.value_type),
                )?;
                write!(out, "\r\n")?;
            }
        }
        Ok(())
    }

    /// Draw the tensor detail screen. `view` is the active dtype reinterpretation
    /// (which changes the shown dtype, shape and parameter count); `overridable`
    /// gates the `d` hint. Rendered flicker-free so it can also serve as the
    /// live preview while choosing a dtype in the menu.
    pub fn draw_tensor_detail(
        out: &mut impl Write,
        tensor: &TensorInfo,
        shape: &[usize],
        view: ViewDtype,
        overridable: bool,
        unindexed: bool,
        stats: StatsView,
    ) -> Result<()> {
        queue!(out, BeginSynchronizedUpdate, cursor::MoveTo(0, 0))?;

        queue!(out, SetForegroundColor(palette::ACCENT))?;
        write!(out, "Tensor Details")?;
        queue!(out, ResetColor, SetForegroundColor(palette::DIM))?;
        line_end(&mut *out)?;
        write!(out, "{}", "─".repeat(14))?;
        queue!(out, ResetColor)?;
        line_end(&mut *out)?;
        paint(&mut *out, false, palette::DIM, "Name: ")?;
        write!(out, "{}", tensor.name)?;
        line_end(&mut *out)?;

        // Data type, with the active reinterpretation highlighted.
        paint(&mut *out, false, palette::DIM, "Data Type: ")?;
        write_view_dtype(&mut *out, &tensor.dtype, view)?;
        line_end(&mut *out)?;

        // Shape and parameter count reflect the overrides: `shape` is the
        // effective (possibly reshaped) shape, and a packed dtype view unpacks
        // several values per stored element, growing the last dimension. Show
        // `stored as reinterpreted` just like the dtype line above.
        let logical = view.logical_shape(shape, &tensor.dtype);
        let num_elements: usize = logical.iter().product();
        paint(&mut *out, false, palette::DIM, "Shape: ")?;
        write_view_shape(&mut *out, &tensor.shape, &logical)?;
        line_end(&mut *out)?;
        paint(&mut *out, false, palette::DIM, "Parameters: ")?;
        write!(out, "{} ", format_parameters(num_elements))?;
        paint(
            &mut *out,
            false,
            palette::DIM,
            &format!("({})", with_thousands(num_elements)),
        )?;
        line_end(&mut *out)?;

        paint(&mut *out, false, palette::DIM, "Size: ")?;
        write!(out, "{}", format_size(tensor.size_bytes))?;
        // On-disk size + codec on the same line, for formats that track
        // compression (HDF5); safetensors leaves it off entirely.
        match &tensor.storage {
            Storage::Compressed {
                codec,
                stored_bytes,
            } => {
                // Show the compression ratio so the on-disk vs logical gap is
                // explicit (e.g. 4-bit weights in 16-bit words only reach ~2×
                // under byte-oriented LZ4).
                let ratio = tensor.size_bytes as f64 / (*stored_bytes).max(1) as f64;
                write!(out, " · on disk: {} ", format_size(*stored_bytes))?;
                paint(
                    &mut *out,
                    false,
                    palette::DIM,
                    &format!("(⇩ {codec}, {ratio:.1}×)"),
                )?;
            }
            Storage::Raw => {
                write!(
                    out,
                    " · on disk: {} (uncompressed)",
                    format_size(tensor.size_bytes)
                )?;
            }
            Storage::Unknown => {}
        }
        line_end(&mut *out)?;
        // Where the data lives within the file.
        match &tensor.layout {
            Layout::ByteRange { start, end } => {
                paint(&mut *out, false, palette::DIM, "Data offsets: ")?;
                write!(
                    out,
                    "{} – {}  (within file data)",
                    with_thousands(*start as usize),
                    with_thousands(*end as usize)
                )?;
                line_end(&mut *out)?;
            }
            Layout::Offset(offset) => {
                paint(&mut *out, false, palette::DIM, "Data offset: ")?;
                write!(
                    out,
                    "{}  (within tensor data)",
                    with_thousands(*offset as usize)
                )?;
                line_end(&mut *out)?;
            }
            Layout::Chunked { chunk, num_chunks } => {
                paint(&mut *out, false, palette::DIM, "Chunks: ")?;
                write!(
                    out,
                    "{} × {}",
                    format_shape(chunk),
                    with_thousands(*num_chunks)
                )?;
                line_end(&mut *out)?;
            }
            Layout::None => {}
        }
        paint(&mut *out, false, palette::DIM, "File: ")?;
        write!(out, "{}", tensor.source_path)?;
        line_end(&mut *out)?;
        // Flag a tensor that's on disk but absent from the index.
        if unindexed {
            queue!(out, SetForegroundColor(palette::UNINDEXED))?;
            write!(
                out,
                "{UNINDEXED_MARK} on disk but not listed in model.safetensors.index.json"
            )?;
            queue!(out, ResetColor)?;
            line_end(&mut *out)?;
        }
        line_end(&mut *out)?;

        // Exact whole-tensor statistics: shown once computed, else a hint.
        match stats {
            StatsView::Ready(s) => {
                // min/max are exact integers for integer dtypes — show them as
                // such (no `.0000`).
                let integer = view.is_integer(&tensor.dtype);
                paint(&mut *out, false, palette::DIM, "Statistics: ")?;
                write!(
                    out,
                    "min {} · max {} · ",
                    fmt_value(s.min, integer),
                    fmt_value(s.max, integer)
                )?;
                write_stats_line(&mut *out, s)?;
            }
            StatsView::Computing {
                spinner,
                elapsed,
                progress,
            } => {
                paint(&mut *out, false, palette::DIM, "Statistics: ")?;
                write_computing(&mut *out, spinner, elapsed, progress)?;
            }
            StatsView::Pending => {
                queue!(out, SetForegroundColor(palette::DIM))?;
                write!(out, "Statistics: press ")?;
                queue!(out, ResetColor)?;
                key_hint(&mut *out, "s")?;
                queue!(out, SetForegroundColor(palette::DIM))?;
                write!(out, " to scan the full tensor")?;
                queue!(out, ResetColor)?;
            }
        }
        line_end(&mut *out)?;
        line_end(&mut *out)?;

        // Footer hints (keys highlighted).
        write!(out, "Press ")?;
        key_hint(&mut *out, "m")?;
        write!(out, " for a heatmap, ")?;
        key_hint(&mut *out, "v")?;
        write!(out, " for numeric values, ")?;
        if overridable {
            key_hint(&mut *out, "d")?;
            write!(out, " to reinterpret the dtype, ")?;
            key_hint(&mut *out, "r")?;
            write!(out, " to reshape, ")?;
        }
        key_hint(&mut *out, "c")?;
        write!(out, " to copy, ")?;
        key_hint(&mut *out, "y")?;
        write!(out, " to copy the command, ")?;
        key_hint(&mut *out, "l")?;
        write!(out, " for the legend, ")?;
        key_hint(&mut *out, "⌫")?;
        write!(out, " / ")?;
        key_hint(&mut *out, "\\")?;
        write!(out, " to step back / forward, any other key to return...")?;

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
    pub fn draw_heatmap(
        out: &mut impl Write,
        tensor: &TensorInfo,
        sample: &Sample,
        stats: StatsView,
    ) -> Result<()> {
        // Present the whole frame atomically (the terminal buffers everything
        // between Begin/End and paints it in one go, so a redraw never shows a
        // half-updated screen — this is what eliminates the flicker). We also
        // overwrite in place: write each line's new content first, then clear
        // only the leftover tail (`line_end`), and never emit a trailing
        // newline (which could scroll the screen and flash).
        queue!(out, BeginSynchronizedUpdate, cursor::MoveTo(0, 0))?;

        write!(out, "Heatmap: {}", tensor.name)?;
        line_end(&mut *out)?;
        let integer = sample.view.is_integer(&tensor.dtype);
        // Use the exact whole-tensor range (and color scale) once stats are
        // ready; otherwise fall back to the sampled range, flagged as such.
        let (rmin, rmax) = match stats {
            StatsView::Ready(s) => (s.min, s.max),
            _ => (sample.min, sample.max),
        };
        let lo = fmt_value(rmin, integer);
        let hi = fmt_value(rmax, integer);
        let range_note = if matches!(stats, StatsView::Ready(_)) {
            ""
        } else {
            " (sampled)"
        };
        write_view_dtype(&mut *out, &tensor.dtype, sample.view)?;
        write!(out, " ")?;
        write_view_shape(&mut *out, &tensor.shape, &sample.display_shape)?;
        let what = match sample.mode {
            SampleMode::Edges { .. } => "edges",
            SampleMode::Window { .. } => "window",
            SampleMode::Grid => "sampled",
        };
        write!(
            out,
            " → {what} {}×{}, value range [{lo}, {hi}]{range_note}",
            sample.rows.len(),
            sample.cols.len(),
        )?;
        line_end(&mut *out)?;
        write_stats_view(&mut *out, stats)?;
        if sample.slices > 1 {
            write_slice_header(&mut *out, sample)?;
            line_end(&mut *out)?;
        }
        line_end(&mut *out)?;

        let range = rmax - rmin;
        let norm = |v: f64| {
            if range > 0.0 { (v - rmin) / range } else { 0.5 }
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
            line_end(&mut *out)?;
            r += 2;
        }

        line_end(&mut *out)?;
        write!(out, "{lo} low ")?;
        for i in 0..24 {
            queue!(out, SetForegroundColor(heat_color(i as f64 / 23.0)))?;
            write!(out, "█")?;
        }
        queue!(out, ResetColor)?;
        write!(out, " high {hi}")?;
        line_end(&mut *out)?;

        line_end(&mut *out)?;
        write_view_footer(&mut *out, sample, true, StripeMode::Off)?;

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
    pub fn draw_values(
        out: &mut impl Write,
        tensor: &TensorInfo,
        sample: &Sample,
        stats: StatsView,
        stripe: StripeMode,
    ) -> Result<()> {
        // Cell width adapts to the data: floats need room for scientific
        // notation, while small integers (incl. sparse values in a wide dtype)
        // are 1-3 digits, so we pack many narrow columns onto the screen.
        let cw = sample.view.cell_width(&tensor.dtype, stats.value_range());
        // Synchronized, in-place overwrite (see `draw_heatmap`) to avoid flicker.
        queue!(out, BeginSynchronizedUpdate, cursor::MoveTo(0, 0))?;

        write!(out, "Values: {}", tensor.name)?;
        line_end(&mut *out)?;
        write_view_dtype(&mut *out, &tensor.dtype, sample.view)?;
        write!(out, " ")?;
        write_view_shape(&mut *out, &tensor.shape, &sample.display_shape)?;
        // Describe the layout: a contiguous window, the first/last edges (for
        // padding), or an evenly-spaced overview.
        let edges = matches!(sample.mode, SampleMode::Edges { .. });
        match sample.mode {
            SampleMode::Edges { .. } => write!(
                out,
                " → edges: {} of {} rows × {} of {} cols (indices shown)",
                edge_desc(&sample.rows, sample.total_rows),
                sample.total_rows,
                edge_desc(&sample.cols, sample.total_cols),
                sample.total_cols
            )?,
            SampleMode::Window { .. } => write!(
                out,
                " → window: rows {} of {} × cols {} of {} (contiguous)",
                span_desc(&sample.rows),
                sample.total_rows,
                span_desc(&sample.cols),
                sample.total_cols
            )?,
            SampleMode::Grid => write!(
                out,
                " → sampled {} of {} rows × {} of {} cols (indices shown)",
                sample.rows.len(),
                sample.total_rows,
                sample.cols.len(),
                sample.total_cols
            )?,
        }
        line_end(&mut *out)?;
        write_stats_view(&mut *out, stats)?;
        if sample.slices > 1 {
            write_slice_header(&mut *out, sample)?;
            line_end(&mut *out)?;
        }
        line_end(&mut *out)?;

        // In edges mode, the index after which rows/cols jump (the padding
        // boundary), so we can draw a dotted separator there. `None` in grid
        // mode, or when the matrix was small enough to show contiguously.
        let gap = |idx: &[usize]| -> Option<usize> {
            edges
                .then(|| idx.windows(2).position(|w| w[1] != w[0] + 1))
                .flatten()
        };
        let row_gap = gap(&sample.rows);
        let col_gap = gap(&sample.cols);
        // Width of the row-index column. Values are right-aligned in their own
        // cells (each with ≥1 leading space), so no extra separator is needed —
        // keeping the whole grid one column further left.
        let lw = 6usize;

        // Column-index header (with a "⋯" separator column at the gap). With
        // wide cells the index fits in a single row; with narrow cells (sub-byte
        // / small-int views) the index is as wide as a cell or wider, so we
        // stagger the labels across two rows ("leap-frog") to keep them legible.
        let idx_w = sample
            .cols
            .iter()
            .map(|&c| c.to_string().len())
            .max()
            .unwrap_or(1);
        if idx_w >= cw {
            // Stagger labels across two rows ("leap-frog") so each may be up to
            // two cells wide; and when even that is too tight (very wide indices
            // over very narrow cells) skip every `step`-th label so the ones we
            // do show don't collide. `step` is the smallest stride whose
            // two-row spacing (`2 * step * cw`) fits a label plus a space.
            let step = (idx_w + 1).div_ceil(2 * cw).max(1);
            // Column offset (within the line) of the right edge of cell `j`,
            // accounting for the row-label prefix and the extra "⋯" gap cell
            // that sits after `col_gap`.
            let right_edge = |j: usize| -> usize {
                let gap_cells = matches!(col_gap, Some(g) if j > g) as usize;
                lw + (j + 1 + gap_cells) * cw
            };
            let width = right_edge(sample.cols.len().saturating_sub(1)).max(lw);
            let mut top = vec![' '; width];
            let mut bot = vec![' '; width];
            let mut rank = 0usize; // position among the labels we actually show
            for (j, &c) in sample.cols.iter().enumerate() {
                if !j.is_multiple_of(step) {
                    continue;
                }
                let label = c.to_string();
                let end = right_edge(j);
                let start = end.saturating_sub(label.len());
                let buf = if rank.is_multiple_of(2) {
                    &mut top
                } else {
                    &mut bot
                };
                for (k, ch) in label.chars().enumerate() {
                    buf[start + k] = ch;
                }
                rank += 1;
            }
            // Mark the skipped-columns gap with a dotted separator on both rows,
            // but only where the cell is still blank — a label wider than a cell
            // can overflow into the gap, and we must not clobber its digits.
            if let Some(g) = col_gap {
                let pos = right_edge(g) + cw - 1;
                if pos < width {
                    for buf in [&mut top, &mut bot] {
                        if buf[pos] == ' ' {
                            buf[pos] = '⋯';
                        }
                    }
                }
            }
            let top: String = top.into_iter().collect();
            let bot: String = bot.into_iter().collect();
            // The index labels are dimmed so they recede behind the values.
            queue!(out, SetForegroundColor(palette::DIM))?;
            write!(out, "{}", top.trim_end())?;
            queue!(out, ResetColor)?;
            line_end(&mut *out)?;
            queue!(out, SetForegroundColor(palette::DIM))?;
            write!(out, "{}", bot.trim_end())?;
            queue!(out, ResetColor)?;
            line_end(&mut *out)?;
        } else {
            queue!(out, SetForegroundColor(palette::DIM))?;
            write!(out, "{:>lw$}", "")?;
            for (j, &c) in sample.cols.iter().enumerate() {
                write!(out, "{c:>cw$}")?;
                if Some(j) == col_gap {
                    write!(out, "{:>cw$}", "⋯")?;
                }
            }
            queue!(out, ResetColor)?;
            line_end(&mut *out)?;
        }

        // Integer dtypes print as plain integers; floats use scientific notation.
        let integer = sample.view.is_integer(&tensor.dtype);
        // Alternating "dim highlighter" backgrounds: a full band per visual row,
        // or — for columns — a background hugging just the digits of each visual
        // column (so it never colours the empty alignment padding).
        let band = |k: usize| {
            if k.is_multiple_of(2) {
                palette::STRIPE_DARK
            } else {
                palette::STRIPE_LITE
            }
        };
        for (i, row) in sample.values.iter().enumerate() {
            // Row striping paints the whole band (label + values) in one go.
            if stripe == StripeMode::Rows {
                queue!(out, SetBackgroundColor(band(i)))?;
            }
            // Dimmed row index (then back to the default fg, keeping any bg).
            queue!(out, SetForegroundColor(palette::DIM))?;
            write!(out, "{:>lw$}", sample.rows[i])?;
            queue!(out, SetForegroundColor(Color::Reset))?;
            let mut vcol = 0usize; // visual column ordinal (counts the gap cell)
            for (j, &v) in row.iter().enumerate() {
                let s = if integer {
                    format!("{:>cw$}", v as i64)
                } else {
                    format!("{v:>cw$.3e}")
                };
                let bg = (stripe == StripeMode::Cols).then(|| band(vcol));
                write_grid_cell(&mut *out, &s, bg, false)?;
                vcol += 1;
                if Some(j) == col_gap {
                    let bg = (stripe == StripeMode::Cols).then(|| band(vcol));
                    write_grid_cell(&mut *out, &format!("{:>cw$}", "⋯"), bg, true)?;
                    vcol += 1;
                }
            }
            queue!(out, ResetColor)?;
            line_end(&mut *out)?;
            // Dotted row after the gap to mark the rows that were skipped.
            if Some(i) == row_gap {
                queue!(out, SetForegroundColor(palette::DIM))?;
                write!(out, "{:>lw$}", "⋮")?;
                for j in 0..row.len() {
                    write!(out, "{:>cw$}", "⋮")?;
                    if Some(j) == col_gap {
                        write!(out, "{:>cw$}", "⋱")?;
                    }
                }
                queue!(out, ResetColor)?;
                line_end(&mut *out)?;
            }
        }

        line_end(&mut *out)?;
        write_view_footer(&mut *out, sample, false, stripe)?;

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

    /// The reshape prompt (`r`): shows the stored shape and the element count the
    /// entry must multiply to, the input box, and a feedback line for errors.
    pub fn draw_reshape_prompt(
        elements: usize,
        stored: &[usize],
        input: &str,
        error: Option<&str>,
    ) -> Result<()> {
        let stdout = io::stdout();
        let mut out = BufWriter::new(stdout.lock());
        let (_w, h) = terminal::size()?;

        queue!(
            out,
            cursor::MoveTo(0, h.saturating_sub(2)),
            terminal::Clear(ClearType::CurrentLine),
            SetForegroundColor(palette::KEY)
        )?;
        write!(out, "Reshape {} ", format_shape(stored))?;
        queue!(out, SetForegroundColor(palette::DIM))?;
        write!(
            out,
            "(dims multiplying to {elements}; `-1`/`*`/`_` infers one; empty clears)"
        )?;
        queue!(out, ResetColor)?;
        write!(out, "  ")?;
        input_box(&mut out, input, 16)?;
        write!(out, "  ")?;
        key_hint(&mut out, "Enter")?;
        queue!(out, SetForegroundColor(palette::DIM))?;
        write!(out, " to apply · ")?;
        queue!(out, ResetColor)?;
        key_hint(&mut out, "Esc")?;
        queue!(out, SetForegroundColor(palette::DIM))?;
        write!(out, " to cancel")?;
        queue!(out, ResetColor)?;

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

    /// A full-screen single-choice menu: a title and a strip of `options` with
    /// `current` highlighted. Used to pick the repack codec.
    pub fn draw_choice_menu(title: &str, options: &[&str], current: usize) -> Result<()> {
        let stdout = io::stdout();
        let mut out = BufWriter::new(stdout.lock());
        queue!(
            out,
            BeginSynchronizedUpdate,
            terminal::Clear(ClearType::All),
            cursor::MoveTo(0, 0)
        )?;
        write!(out, "{title}\r\n")?;
        write!(out, "{}\r\n\r\n", "=".repeat(title.len().max(10)))?;
        for (i, opt) in options.iter().enumerate() {
            if i == current {
                queue!(
                    out,
                    SetAttribute(Attribute::Bold),
                    SetForegroundColor(palette::SELECT_FG),
                    SetBackgroundColor(palette::SELECT_BG)
                )?;
                write!(out, " {opt} ")?;
                queue!(out, ResetColor, SetAttribute(Attribute::Reset))?;
            } else {
                queue!(out, SetForegroundColor(palette::DIM))?;
                write!(out, " {opt} ")?;
                queue!(out, ResetColor)?;
            }
            write!(out, " ")?;
        }
        write!(out, "\r\n\r\n")?;
        hint_line(
            &mut out,
            &[("← →", "move"), ("Enter", "select"), ("Esc", "cancel")],
        )?;
        write!(out, "\r\n")?;
        queue!(out, EndSynchronizedUpdate)?;
        out.flush()?;
        Ok(())
    }

    /// A free-text input prompt pinned to the bottom (label + editable box +
    /// optional error line). Used to ask for the repack output filename.
    pub fn draw_text_prompt(label: &str, input: &str, error: Option<&str>) -> Result<()> {
        let stdout = io::stdout();
        let mut out = BufWriter::new(stdout.lock());
        let (_w, h) = terminal::size()?;

        queue!(
            out,
            cursor::MoveTo(0, h.saturating_sub(2)),
            terminal::Clear(ClearType::CurrentLine),
            SetForegroundColor(palette::KEY)
        )?;
        write!(out, "{label} ")?;
        queue!(out, ResetColor)?;
        input_box(&mut out, input, 24)?;
        write!(out, "  ")?;
        key_hint(&mut out, "Enter")?;
        queue!(out, SetForegroundColor(palette::DIM))?;
        write!(out, " to confirm · ")?;
        queue!(out, ResetColor)?;
        key_hint(&mut out, "Esc")?;
        queue!(out, SetForegroundColor(palette::DIM))?;
        write!(out, " to cancel")?;
        queue!(out, ResetColor)?;

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

    /// A full-screen progress view with a bar, `done/total` count and a detail
    /// line (e.g. the dataset currently being written). Drawn in place.
    #[cfg(feature = "hdf5")]
    pub fn draw_progress(title: &str, done: usize, total: usize, detail: &str) -> Result<()> {
        let stdout = io::stdout();
        let mut out = BufWriter::new(stdout.lock());
        queue!(out, BeginSynchronizedUpdate, cursor::MoveTo(0, 0))?;
        write!(out, "{title}")?;
        line_end(&mut out)?;
        write!(out, "{}", "=".repeat(title.len().max(10)))?;
        line_end(&mut out)?;
        line_end(&mut out)?;

        const WIDTH: usize = 40;
        let frac = if total > 0 {
            done as f64 / total as f64
        } else {
            0.0
        };
        let filled = (frac * WIDTH as f64).round() as usize;
        write!(out, "[")?;
        queue!(out, SetForegroundColor(palette::KEY))?;
        write!(out, "{}", "█".repeat(filled))?;
        queue!(out, SetForegroundColor(palette::DIM))?;
        write!(out, "{}", "░".repeat(WIDTH.saturating_sub(filled)))?;
        queue!(out, ResetColor)?;
        write!(out, "] {done}/{total}")?;
        line_end(&mut out)?;
        queue!(out, SetForegroundColor(palette::DIM))?;
        write!(out, "{detail}")?;
        queue!(out, ResetColor)?;
        line_end(&mut out)?;

        queue!(
            out,
            terminal::Clear(ClearType::FromCursorDown),
            EndSynchronizedUpdate
        )?;
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

    /// Draw the copied CLI command as a borderless pop-up *over* the current
    /// screen (the surrounding view stays visible above and below the band; the
    /// caller redraws it on dismiss — the screen is not cleared). The command
    /// sits on its **own line at column 0**, bracketed by horizontal rules but
    /// with nothing before or after it on its row(s), so it can be selected
    /// cleanly with the mouse or a multiplexer's copy mode — important when the
    /// OSC-52 clipboard copy doesn't reach the terminal and it must be copied by
    /// hand. The terminal soft-wraps a long command, but it stays one logical
    /// line, so the selection still yields the whole command.
    pub fn draw_command(command: &str) -> Result<()> {
        let mut out = io::stdout();
        let (term_w, term_h) = terminal::size()?;
        let (term_w, term_h) = (term_w as usize, term_h as usize);
        let rule = "─".repeat(term_w);

        // How many rows the command occupies once soft-wrapped at full width;
        // used to place the closing rule/footer below it. Centre the band.
        let cmd_rows = command.chars().count().div_ceil(term_w.max(1)).max(1);
        // blank, header, rule, command, rule, footer, blank
        let band_h = cmd_rows + 6;
        let mut row = (term_h.saturating_sub(band_h) / 2) as u16;

        // Clear a band row so the underlying screen doesn't show through.
        let clear = ClearType::CurrentLine;
        queue!(out, BeginSynchronizedUpdate)?;

        // A cleared margin row above, so the band reads as a floating pop-up.
        queue!(out, cursor::MoveTo(0, row), terminal::Clear(clear))?;
        row += 1;

        // Header: title + copied confirmation.
        queue!(
            out,
            cursor::MoveTo(0, row),
            terminal::Clear(clear),
            SetForegroundColor(palette::KEY),
            SetAttribute(Attribute::Bold)
        )?;
        write!(out, "CLI command")?;
        queue!(
            out,
            SetAttribute(Attribute::Reset),
            SetForegroundColor(palette::SUCCESS)
        )?;
        write!(out, "   ✓ copied to the clipboard")?;
        queue!(out, ResetColor)?;
        row += 1;

        // Opening rule.
        queue!(
            out,
            cursor::MoveTo(0, row),
            terminal::Clear(clear),
            SetForegroundColor(palette::ACCENT)
        )?;
        write!(out, "{rule}")?;
        queue!(out, ResetColor)?;
        row += 1;

        // The command: blank its rows first, then write it at column 0 so it
        // soft-wraps cleanly with nothing flanking it.
        for r in 0..cmd_rows as u16 {
            queue!(out, cursor::MoveTo(0, row + r), terminal::Clear(clear))?;
        }
        queue!(out, cursor::MoveTo(0, row))?;
        write!(out, "{command}")?;
        row += cmd_rows as u16;

        // Closing rule.
        queue!(
            out,
            cursor::MoveTo(0, row),
            terminal::Clear(clear),
            SetForegroundColor(palette::ACCENT)
        )?;
        write!(out, "{rule}")?;
        queue!(out, ResetColor)?;
        row += 1;

        // Footer hint.
        queue!(
            out,
            cursor::MoveTo(0, row),
            terminal::Clear(clear),
            SetForegroundColor(palette::DIM)
        )?;
        write!(
            out,
            "select the command above to copy it by hand · any key to dismiss"
        )?;
        queue!(out, ResetColor)?;
        row += 1;

        // A cleared margin row below.
        queue!(out, cursor::MoveTo(0, row), terminal::Clear(clear))?;

        queue!(out, EndSynchronizedUpdate)?;
        out.flush()?;
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

    /// Draw a context-sensitive legend explaining the glyphs (and a few colour
    /// cues) on whichever screen the user opened it from (`l`). Full-screen and
    /// flicker-free, like the detail view; the caller waits for a key, then
    /// redraws its own screen over it.
    pub fn draw_legend(legend: Legend) -> Result<()> {
        let stdout = io::stdout();
        let mut out = BufWriter::new(stdout.lock());
        queue!(out, BeginSynchronizedUpdate, cursor::MoveTo(0, 0))?;

        let title = match legend {
            Legend::Tree => "Legend — checkpoint tree",
            Legend::Detail => "Legend — tensor details",
            Legend::Heatmap => "Legend — heatmap",
            Legend::Values => "Legend — numeric values",
        };
        queue!(out, SetForegroundColor(palette::ACCENT))?;
        write!(out, "{title}")?;
        queue!(out, ResetColor, SetForegroundColor(palette::DIM))?;
        line_end(&mut out)?;
        write!(out, "{}", "─".repeat(title.chars().count()))?;
        queue!(out, ResetColor)?;
        line_end(&mut out)?;
        line_end(&mut out)?;

        match legend {
            Legend::Tree => {
                let rows = [
                    (
                        Some(palette::ACCENT),
                        "▾ ▸",
                        "a group, expanded / collapsed (Enter or Space toggles it)",
                    ),
                    (Some(palette::DIM), "·", "a tensor (a stored array)"),
                    (
                        Some(palette::UNINDEXED),
                        UNINDEXED_MARK,
                        "an extra tensor on disk but not listed in the index (model.safetensors.index.json)",
                    ),
                    (Some(palette::DIM), "≡", "a metadata entry"),
                    (
                        None,
                        "☰ N",
                        "number of layers (numbered sub-groups) in the group",
                    ),
                    (None, "▦ N", "number of tensors in the group / checkpoint"),
                    (
                        None,
                        "A → B",
                        "logical size → on-disk size (shown only when they differ)",
                    ),
                    (
                        Some(palette::DIM),
                        "⇩ lz4",
                        "compressed on disk; the codec is named after the glyph",
                    ),
                    (Some(palette::DIM), "(raw)", "stored uncompressed on disk"),
                    (None, "", ""),
                    (
                        Some(palette::DTYPE),
                        "I16",
                        "the tensor's data type is tinted (warm amber)",
                    ),
                    (
                        None,
                        "▪ ▸",
                        "status bar: a single source file / a directory of shards",
                    ),
                ];
                let col = legend_desc_col(&rows, 0);
                for (color, sym, desc) in rows {
                    legend_line(&mut out, color, sym, desc, col)?;
                }
            }
            Legend::Detail => {
                let rows = [
                    (
                        Some(palette::DIM),
                        "⇩ lz4",
                        "on-disk compression codec; the N× beside it is the ratio (logical ÷ stored)",
                    ),
                    (
                        Some(palette::KEY),
                        "as",
                        "the active dtype reinterpretation (press d), e.g. 'BF16 as u4 (packed)'",
                    ),
                    (
                        None,
                        "A – B",
                        "a byte range within the file (the tensor's data offsets)",
                    ),
                    (Some(palette::DIM), "·", "separates fields on a line"),
                    (
                        Some(palette::UNINDEXED),
                        UNINDEXED_MARK,
                        "this tensor is an extra: on disk but not listed in the index (model.safetensors.index.json)",
                    ),
                    (
                        Some(palette::KEY),
                        "⠋",
                        "a statistics scan is running (press s to start; any key cancels)",
                    ),
                ];
                let col = legend_desc_col(&rows, 0);
                for (color, sym, desc) in rows {
                    legend_line(&mut out, color, sym, desc, col)?;
                }
                legend_line(&mut out, None, "", "", col)?;
                queue!(
                    out,
                    terminal::Clear(ClearType::CurrentLine),
                    SetForegroundColor(palette::DIM)
                )?;
                write!(
                    out,
                    "  Statistics:  zeros = fraction of exactly-zero values · non-finite = count of NaN/∞"
                )?;
                queue!(out, ResetColor)?;
                write!(out, "\r\n")?;
            }
            Legend::Heatmap => {
                let rows = [
                    (
                        None,
                        "▀",
                        "one cell packs two data rows: its top half is the upper row, its lower half the next",
                    ),
                    (
                        None,
                        "A → B",
                        "the stored dtype/shape → the sampled grid size and value range",
                    ),
                ];
                let col = legend_desc_col(&rows, 0);
                for (color, sym, desc) in rows {
                    legend_line(&mut out, color, sym, desc, col)?;
                }
                // The actual colour ramp, so the scale is unambiguous.
                queue!(out, terminal::Clear(ClearType::CurrentLine))?;
                write!(out, "  ")?;
                queue!(out, SetForegroundColor(palette::DIM))?;
                write!(out, "low ")?;
                queue!(out, ResetColor)?;
                for i in 0..24 {
                    queue!(out, SetForegroundColor(heat_color(i as f64 / 23.0)))?;
                    write!(out, "█")?;
                }
                queue!(out, ResetColor, SetForegroundColor(palette::DIM))?;
                write!(out, " high")?;
                queue!(out, ResetColor)?;
                write!(out, "   colour scale: cool = low value, warm = high value")?;
                write!(out, "\r\n")?;
            }
            Legend::Values => {
                let rows = [
                    (
                        Some(palette::DIM),
                        "12  34",
                        "row / column indices into the full tensor (dimmed), not data values",
                    ),
                    (
                        Some(palette::DIM),
                        "⋯",
                        "columns were skipped here (the gap between the first and last columns)",
                    ),
                    (Some(palette::DIM), "⋮", "rows were skipped here"),
                    (
                        Some(palette::DIM),
                        "⋱",
                        "both rows and columns were skipped (the corner)",
                    ),
                    (
                        None,
                        "1.2e-3",
                        "floats use scientific notation; integers print plain",
                    ),
                ];
                // Reserve room for the wider zebra swatch row drawn below.
                let col = legend_desc_col(&rows, 8);
                for (color, sym, desc) in rows {
                    legend_line(&mut out, color, sym, desc, col)?;
                }
                // A live zebra swatch, since it is a background cue, not a glyph.
                queue!(out, terminal::Clear(ClearType::CurrentLine))?;
                write!(out, "  ")?;
                queue!(out, SetBackgroundColor(palette::STRIPE_DARK))?;
                write!(out, " 12 ")?;
                queue!(out, SetBackgroundColor(palette::STRIPE_LITE))?;
                write!(out, " 34 ")?;
                queue!(out, SetBackgroundColor(Color::Reset))?;
                queue!(out, cursor::MoveToColumn(col))?;
                write!(
                    out,
                    "zebra striping traces a row or column (cycle rows/cols/off with z)"
                )?;
                write!(out, "\r\n")?;
            }
        }

        queue!(out, terminal::Clear(ClearType::CurrentLine))?;
        write!(out, "\r\n")?;
        queue!(out, SetForegroundColor(palette::DIM))?;
        write!(out, "Press any key to close.")?;
        queue!(out, ResetColor)?;

        queue!(
            out,
            terminal::Clear(ClearType::FromCursorDown),
            EndSynchronizedUpdate
        )?;
        out.flush()?;
        Ok(())
    }
}

/// Worst-case display width of a legend symbol: every non-ASCII glyph is counted
/// as two cells. The symbols are box-drawing / geometric glyphs whose rendered
/// width is terminal-dependent (one cell in many terminals, two in others), so
/// assuming the wider case keeps the description column from ever overlapping
/// the symbol — see [`legend_desc_col`].
fn legend_symbol_width(symbol: &str) -> usize {
    symbol
        .chars()
        .map(|c| if c.is_ascii() { 1 } else { 2 })
        .sum()
}

/// The column (0-based) at which every legend description should start: past a
/// two-space indent, the widest symbol, and a two-space gap. `reserve` is an
/// extra minimum width for a non-symbol row drawn separately (e.g. the zebra
/// swatch) so its description lines up too.
fn legend_desc_col(rows: &[(Option<Color>, &str, &str)], reserve: usize) -> u16 {
    let widest = rows
        .iter()
        .map(|(_, sym, _)| legend_symbol_width(sym))
        .max()
        .unwrap_or(0)
        .max(reserve);
    (2 + widest + 2) as u16
}

/// Write one legend row: the `symbol` (in `color`, or the default foreground
/// when `None`), then its description starting at the absolute column `desc_col`.
/// The description is positioned with a cursor move rather than space-padding, so
/// it lines up no matter how wide the terminal renders the symbol glyphs. The
/// whole line is cleared first so the skipped gap shows nothing from the screen
/// underneath. An all-empty row is just a blank separator line.
fn legend_line(
    out: &mut impl Write,
    color: Option<Color>,
    symbol: &str,
    desc: &str,
    desc_col: u16,
) -> Result<()> {
    queue!(out, terminal::Clear(ClearType::CurrentLine))?;
    if symbol.is_empty() && desc.is_empty() {
        write!(out, "\r\n")?;
        return Ok(());
    }
    write!(out, "  ")?;
    match color {
        Some(c) => {
            queue!(out, SetForegroundColor(c))?;
            write!(out, "{symbol}")?;
            queue!(out, SetForegroundColor(Color::Reset))?;
        }
        None => write!(out, "{symbol}")?,
    }
    queue!(out, cursor::MoveToColumn(desc_col))?;
    write!(out, "{desc}\r\n")?;
    Ok(())
}

/// Write `text` in `color`, unless `selected` — then write it plain so the
/// caller's row highlight (inverse video) shows through. Only the foreground is
/// touched (reset to default after), so any background the caller set persists.
fn paint(out: &mut impl Write, selected: bool, color: Color, text: &str) -> Result<()> {
    if selected {
        write!(out, "{text}")?;
    } else {
        queue!(out, SetForegroundColor(color))?;
        write!(out, "{text}")?;
        queue!(out, SetForegroundColor(Color::Reset))?;
    }
    Ok(())
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
    // The overview frees the arrows for slice stepping; the edges and window
    // layouts claim them (divider / pan), so slices move on `[` / `]` there.
    if matches!(sample.mode, SampleMode::Grid) {
        hint_line(
            out,
            &[
                ("← →", "step"),
                ("Shift+← →", "jump 5% (both wrap)"),
                ("/", "index or %"),
            ],
        )
    } else {
        hint_line(out, &[("[ ]", "step"), ("/", "index or %")])
    }
}

/// Footer for the data views: offers the other representation (`m`/`v` switch
/// in place, no trip back to the detail screen) and mentions slice navigation
/// only when there is more than one slice to move between. Keys highlighted.
fn write_view_footer(
    out: &mut impl Write,
    sample: &Sample,
    heatmap: bool,
    stripe: StripeMode,
) -> Result<()> {
    let switch = if heatmap {
        ("v", "numeric values")
    } else {
        ("m", "heatmap")
    };
    let mut items = vec![switch];
    let edges = matches!(sample.mode, SampleMode::Edges { .. });
    let window = matches!(sample.mode, SampleMode::Window { .. });
    // In the edges view the arrows rebalance first vs. last (Shift snaps to one
    // end); in the window view they pan the block (Shift a screenful, Ctrl to an
    // edge). Either way slice stepping moves to `[`/`]` so the arrows are free.
    if edges {
        items.push(("← →", "first/last cols"));
        items.push(("↑ ↓", "first/last rows"));
        items.push(("+Shift", "one end"));
    }
    if window {
        items.push(("←↑↓→", "pan"));
        items.push(("+Shift", "page"));
        items.push(("Home/End", "col edge"));
        items.push(("PgUp/Dn", "row edge"));
    }
    if sample.slices > 1 {
        if edges || window {
            items.push(("[ ]", "slice"));
        } else {
            items.push(("← →", "step"));
            items.push(("Shift+← →", "jump 5%"));
        }
        items.push(("/", "index or %"));
    }
    if sample.overridable {
        items.push(("d", "dtype"));
        items.push(("r", "reshape"));
    }
    // Cycle the layout overview → edges → window → overview; the label names the
    // layout `e` switches to next.
    items.push(match sample.mode {
        SampleMode::Grid => ("e", "edges"),
        SampleMode::Edges { .. } => ("e", "window"),
        SampleMode::Window { .. } => ("e", "overview"),
    });
    // Cycle the zebra striping rows → cols → off (numeric grid only); the label
    // shows the current mode.
    if !heatmap {
        items.push(match stripe {
            StripeMode::Rows => ("z", "zebra: rows"),
            StripeMode::Cols => ("z", "zebra: cols"),
            StripeMode::Off => ("z", "zebra: off"),
        });
    }
    // Copy the screen's text to the clipboard.
    items.push(("c", "copy"));
    // Show and copy the CLI command that reopens this view.
    items.push(("y", "copy cmd"));
    // Open the legend for this view's glyphs.
    items.push(("l", "legend"));
    // Step back / forward through the screen history.
    items.push(("⌫", "back"));
    items.push(("\\", "fwd"));
    items.push(("", "any other key to return..."));
    hint_line(out, &items)
}

/// Write one right-aligned numeric cell (already formatted to the cell width).
/// When `bg` is set (column striping), the background covers a *constant* band —
/// the whole cell except its first column — so every column's stripe is the
/// same width and a one-space gutter separates neighbouring stripes (values are
/// right-aligned and never fill that first column). `dim` dims the glyphs (used
/// for the "⋯" gap marker).
fn write_grid_cell(out: &mut impl Write, s: &str, bg: Option<Color>, dim: bool) -> Result<()> {
    if dim {
        queue!(out, SetForegroundColor(palette::DIM))?;
    }
    match bg {
        // Leave the first column an uncoloured gutter and band the rest, so the
        // stripe is the same width for every column.
        Some(c) => {
            let split = s.char_indices().nth(1).map_or(s.len(), |(i, _)| i);
            let (gutter, band) = s.split_at(split);
            write!(out, "{gutter}")?;
            queue!(out, SetBackgroundColor(c))?;
            write!(out, "{band}")?;
            queue!(out, SetBackgroundColor(Color::Reset))?;
        }
        None => write!(out, "{s}")?,
    }
    if dim {
        queue!(out, SetForegroundColor(Color::Reset))?;
    }
    Ok(())
}

/// Describe a contiguous window's extent along one axis — e.g. `120–179` for the
/// rows/cols currently shown (the header pairs it with the axis total).
fn span_desc(idx: &[usize]) -> String {
    match (idx.first(), idx.last()) {
        (Some(a), Some(b)) => format!("{a}–{b}"),
        _ => "—".to_string(),
    }
}

/// Describe an edges-view index slice for the header — e.g. `first 26 & last 25`,
/// `last 50`, `first 50`, or `all 50` when the whole axis fits — so the current
/// first/last split (and any bias the user dialed in) is visible at a glance.
fn edge_desc(idx: &[usize], total: usize) -> String {
    let n = idx.len();
    if n >= total {
        return format!("all {n}");
    }
    match idx.windows(2).position(|w| w[1] != w[0] + 1) {
        Some(g) => format!("first {} & last {}", g + 1, n - (g + 1)),
        None if idx.first() == Some(&0) => format!("first {n}"),
        None => format!("last {n}"),
    }
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

/// Write a one-line statistics summary (mean, std, sparsity, non-finite count),
/// with field labels dimmed; the non-finite count is highlighted when nonzero.
fn write_stats_line(out: &mut impl Write, s: &Stats) -> Result<()> {
    queue!(out, SetForegroundColor(palette::DIM))?;
    write!(out, "mean ")?;
    queue!(out, ResetColor)?;
    write!(out, "{:.4}", s.mean)?;
    queue!(out, SetForegroundColor(palette::DIM))?;
    write!(out, " · std ")?;
    queue!(out, ResetColor)?;
    write!(out, "{:.4}", s.std)?;
    queue!(out, SetForegroundColor(palette::DIM))?;
    write!(out, " · zeros ")?;
    queue!(out, ResetColor)?;
    // Distinguish "no zeros at all" from "some, but a tiny fraction": the latter
    // would otherwise round to a misleading `0.0%` (e.g. when min is exactly 0),
    // so show the small fraction in scientific notation to keep its magnitude.
    let pct = s.zero_fraction() * 100.0;
    if s.zeros == 0 {
        write!(out, "0%")?;
    } else if pct < 0.1 {
        write!(out, "{pct:.1e}%")?;
    } else {
        write!(out, "{pct:.1}%")?;
    }
    if s.nonfinite > 0 {
        queue!(out, SetForegroundColor(palette::WARN))?;
        write!(out, " · {} non-finite", s.nonfinite)?;
        queue!(out, ResetColor)?;
    }
    // How long the scan took, dimmed.
    queue!(out, SetForegroundColor(palette::DIM))?;
    write!(out, "  ({})", fmt_duration(s.elapsed))?;
    queue!(out, ResetColor)?;
    Ok(())
}

/// Render the stats line for a data view (heatmap/numeric): the stats once
/// `Ready`, a spinner while `Computing`, nothing while `Pending`. Ends the line.
fn write_stats_view(out: &mut impl Write, stats: StatsView) -> Result<()> {
    match stats {
        StatsView::Ready(s) => {
            write_stats_line(out, s)?;
            line_end(out)?;
        }
        StatsView::Computing {
            spinner,
            elapsed,
            progress,
        } => {
            write_computing(out, spinner, elapsed, progress)?;
            line_end(out)?;
        }
        StatsView::Pending => {}
    }
    Ok(())
}

/// Write the "scan in progress" stats segment: a spinner (accent colour), a
/// dimmed label, a progress bar with a percentage (when the fraction is known)
/// and the running elapsed time. Drawn in place of the stats.
fn write_computing(
    out: &mut impl Write,
    spinner: char,
    elapsed: Duration,
    progress: Option<f64>,
) -> Result<()> {
    queue!(out, SetForegroundColor(palette::KEY))?;
    write!(out, "{spinner} ")?;
    queue!(out, SetForegroundColor(palette::DIM))?;
    write!(out, "computing statistics… ")?;
    if let Some(frac) = progress {
        const WIDTH: usize = 16;
        let frac = frac.clamp(0.0, 1.0);
        let filled = (frac * WIDTH as f64).round() as usize;
        write!(out, "[")?;
        queue!(out, SetForegroundColor(palette::KEY))?;
        write!(out, "{}", "█".repeat(filled))?;
        queue!(out, SetForegroundColor(palette::DIM))?;
        write!(out, "{}", "░".repeat(WIDTH - filled))?;
        write!(out, "] {:>3.0}% · ", frac * 100.0)?;
    }
    write!(out, "{}", fmt_duration(elapsed))?;
    queue!(out, ResetColor)?;
    Ok(())
}

/// Human-readable scan duration: milliseconds under a second, else seconds.
fn fmt_duration(d: Duration) -> String {
    let ms = d.as_millis();
    if ms >= 1000 {
        format!("{:.1}s", d.as_secs_f64())
    } else {
        format!("{ms}ms")
    }
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

/// Write the shape shown in a detail / data-view header. When the active view
/// changes the logical shape — only a packed 4-bit view does, growing the last
/// dimension — fade the stored shape and highlight the reinterpreted one (e.g.
/// `(128, 2880) as (128, 11520)`), mirroring how [`write_view_dtype`] shows the
/// dtype. Otherwise just the (unchanged) shape.
fn write_view_shape(out: &mut impl Write, stored: &[usize], logical: &[usize]) -> Result<()> {
    if stored == logical {
        write!(out, "{}", format_shape(logical))?;
    } else {
        queue!(out, SetForegroundColor(palette::DIM))?;
        write!(out, "{} as ", format_shape(stored))?;
        queue!(
            out,
            ResetColor,
            SetAttribute(Attribute::Bold),
            SetForegroundColor(palette::KEY)
        )?;
        write!(out, "{}", format_shape(logical))?;
        queue!(out, ResetColor, SetAttribute(Attribute::Reset))?;
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
