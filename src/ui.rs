use anyhow::Result;
use crossterm::{
    cursor, execute, queue,
    style::{Attribute, Color, ResetColor, SetAttribute, SetBackgroundColor, SetForegroundColor},
    terminal::{self, BeginSynchronizedUpdate, ClearType, EndSynchronizedUpdate},
};
use std::collections::{HashMap, HashSet};
use std::io::{self, BufWriter, Write};
use std::time::Duration;

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget};

use crate::health::HealthReport;
use crate::sample::{HistBins, Histogram, PackingSchema, Sample, SampleMode, Stats, ViewDtype};
use crate::tree::{Layout, MetadataInfo, Storage, TensorInfo, TreeNode, metadata_short};
use crate::tui::to_ratatui;
use crate::utils::{format_parameters, format_shape, format_size};

/// A still-forming scan's progress indicator: a spinner glyph, the elapsed time,
/// and an optional completed fraction (`None` when the total isn't known).
pub type ScanProgress = (char, std::time::Duration, Option<f64>);

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
    /// Metadata entries (the `†` marker and the entry name) — a muted slate
    /// violet, distinct from the cyan groups and amber dtypes but quiet enough
    /// that metadata reads as a side note rather than competing with tensors.
    pub const META: Color = Color::AnsiValue(103);
    /// Zebra striping for the numeric grid — two subtle dark backgrounds (one
    /// "dark", one "less dark") that alternate to guide the eye along the rows
    /// or columns, like a dim highlighter.
    pub const STRIPE_DARK: Color = Color::AnsiValue(234);
    pub const STRIPE_LITE: Color = Color::AnsiValue(237);
    /// Background for floating pop-ups (legend, the `y` command panel, message
    /// screens) — a neutral dark grey a few shades above black, in the same
    /// family as the zebra greys above, so an overlay reads as a raised surface
    /// over the main screen while staying within the dark theme. Light/accent
    /// foregrounds keep their contrast; dim text stays legible.
    pub const PANEL_BG: Color = Color::AnsiValue(236);
}

/// Marks a tensor that's on disk but not listed in the index (an "extra"),
/// shown in [`palette::UNINDEXED`] in the tree, detail screen and legends.
const UNINDEXED_MARK: &str = "✚";

/// Storage tag for a tensor stored uncompressed on disk. Shared by the tree row,
/// the detail screen and the legend so the wording stays consistent.
const UNCOMPRESSED_TAG: &str = "(uncompressed)";

/// On-disk compression codec marker, e.g. `⇩ lz4`. Shared by the tree row, the
/// detail screen and the legend so the glyph stays consistent.
const COMPRESSED_MARK: &str = "⇩";

/// Separator between a tensor's logical size and its (smaller) on-disk size,
/// e.g. `593 MiB → 588 MiB`. Shared by the tree rows and the legend.
const SIZE_ARROW: &str = "→";

pub struct DrawConfig<'a> {
    pub tree: &'a [(TreeNode, usize)],
    pub current_file: &'a str,
    pub file_idx: usize,
    pub total_files: usize,
    pub selected_idx: usize,
    pub scroll_offset: usize,
    pub search_mode: bool,
    pub search_query: &'a str,
    /// Caret position within `search_query`, as a character index in `0..=len`.
    pub search_cursor: usize,
    /// Leading glyph for the status bar (e.g. `▪`, `▸`, `†`).
    pub status_icon: &'a str,
    /// Bottom status line: a tensor's full name, or a group's source
    /// file(s)/directory.
    pub status_bar: &'a str,
    /// Second status line, below `status_bar`: a tensor's source file (empty for
    /// groups).
    pub status_secondary: &'a str,
    /// Whether a checkpoint health issue was detected (shows a header hint to
    /// press `h` for the report).
    pub health_warning: bool,
    /// Whether the loaded checkpoint can be repacked (a single HDF5 file), which
    /// gates the `R` hint.
    pub can_repack: bool,
    /// `source_path`s of tensors present on disk but not listed in the index
    /// (a stale `model.safetensors.index.json`), flagged in the tree.
    pub unindexed: &'a HashSet<String>,
    /// Per-tensor fused-codebook packing schemas, keyed by tensor name. A tensor
    /// with one shows its logical (unmerged) dtype and shape beside the physical.
    pub packing_schemas: &'a HashMap<String, PackingSchema>,
    /// A transient "✓ Copied …" confirmation to flash on the bottom line (over
    /// the secondary status), set by the tree's copy shortcuts.
    pub copied_flash: Option<&'a str>,
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

/// The numeral base the numeric grid prints values in. `Decimal` is the normal
/// human-readable form (floats in scientific notation, integers as signed
/// decimals); the other bases show each element's raw stored bit pattern,
/// zero-padded to the dtype's width. Cycled with `b`.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub enum NumBase {
    #[default]
    Decimal,
    Hex,
    Octal,
    Binary,
}

impl NumBase {
    /// The next base in the `b` cycle: dec → hex → oct → bin → dec.
    pub fn next(self) -> Self {
        match self {
            NumBase::Decimal => NumBase::Hex,
            NumBase::Hex => NumBase::Octal,
            NumBase::Octal => NumBase::Binary,
            NumBase::Binary => NumBase::Decimal,
        }
    }

    /// Short label for the footer/command (`dec`, `hex`, `oct`, `bin`).
    pub fn label(self) -> &'static str {
        match self {
            NumBase::Decimal => "dec",
            NumBase::Hex => "hex",
            NumBase::Octal => "oct",
            NumBase::Binary => "bin",
        }
    }

    /// Number of digits needed to print `width` bits in this base (raw-bit
    /// bases only; `Decimal` returns 0 since it sizes cells differently).
    fn digits(self, width: u32) -> usize {
        match self {
            NumBase::Decimal => 0,
            NumBase::Hex => width.div_ceil(4) as usize,
            NumBase::Octal => width.div_ceil(3) as usize,
            NumBase::Binary => width as usize,
        }
    }

    /// Display width (chars, incl. a 1-col gap) of one numeric-grid cell under
    /// this base, for the given `view`/`dtype`. Decimal sizes to the actual
    /// value `range` (small ints pack tighter); the raw-bit bases use the
    /// dtype's fixed digit count. Both the sampler (how many columns to fetch)
    /// and the renderer call this, so they can't disagree on the width.
    pub fn cell_width(self, view: ViewDtype, dtype: &str, range: Option<(f64, f64)>) -> usize {
        match self {
            NumBase::Decimal => view.cell_width(dtype, range),
            _ => self.digits(view.bit_width(dtype)) + 1,
        }
    }
}

/// Parse a CLI `--base` value into a [`NumBase`].
pub fn parse_num_base(s: &str) -> Result<NumBase, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "dec" | "decimal" | "10" => Ok(NumBase::Decimal),
        "hex" | "hexadecimal" | "16" => Ok(NumBase::Hex),
        "oct" | "octal" | "8" => Ok(NumBase::Octal),
        "bin" | "binary" | "2" => Ok(NumBase::Binary),
        _ => Err(format!(
            "unknown base '{s}'; expected dec, hex, oct, or bin"
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

/// A floating pop-up the detail screen can show *over* its live frame — drawn as
/// the last layer of [`UI::draw_tensor_detail`] so the screen behind it keeps
/// redrawing (a running scan's progress animates) while it's up. Dismissed by
/// any key. Composited via [`write_legend_band`] / [`write_command_band`].
pub enum Overlay {
    /// The context-sensitive glyph legend (`l`).
    Legend(Legend),
    /// The copied CLI command box (`y`); holds the command to display.
    Command(String),
}

/// Rows of chrome above the tree list: the title, the search/hint line, and the
/// separator rule.
const TREE_HEADER_HEIGHT: usize = 3;
/// Rows of chrome below the tree list: the two-line status bar.
const TREE_FOOTER_HEIGHT: usize = 2;

pub struct UI;

impl UI {
    /// How many tree rows are visible at once (one screenful), used to size a
    /// PageUp/PageDown jump. `terminal_height` is the full terminal height.
    pub fn visible_tree_rows(terminal_height: u16) -> usize {
        (terminal_height as usize)
            .saturating_sub(TREE_HEADER_HEIGHT + TREE_FOOTER_HEIGHT)
            .max(1)
    }

    /// Render the tree browser into `out` (a buffered stdout for the live
    /// screen, or an in-memory buffer when capturing the screen for copy).
    /// Writing the whole frame at once and flushing once — combined with
    /// overwriting in place (clearing each line rather than the whole screen up
    /// front) — removes the flicker a per-frame `Clear(All)` produced.
    pub fn draw_screen(out: &mut impl Write, config: &DrawConfig) -> Result<usize> {
        let (terminal_width, terminal_height) = crate::plain::term_size();
        let header_height = TREE_HEADER_HEIGHT;
        // Two bottom lines for the status bar: the selected tensor's full name on
        // the first, its source file on the second (the per-checkpoint totals now
        // live in the tree root instead of a footer).
        let footer_height = TREE_FOOTER_HEIGHT;
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
            input_box(&mut *out, config.search_query, config.search_cursor, 16)?;
            // The running match count, between the query box and the hints. Only
            // shown once something is typed — with an empty query the list is the
            // whole tree, not a set of matches.
            if config.search_query.is_empty() {
                write!(out, "  ")?;
            } else {
                let n = config.tree.len();
                queue!(out, SetForegroundColor(palette::DIM))?;
                write!(out, "  {n} {}  ", if n == 1 { "match" } else { "matches" })?;
                queue!(out, ResetColor)?;
            }
            hint_line(
                &mut *out,
                &[("Enter", "view"), ("Tab", "in tree"), ("Esc", "exit")],
            )?;
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
                ("n", "copy name"),
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

            Self::draw_node(
                node,
                *depth,
                is_selected,
                config.unindexed,
                config.packing_schemas,
                &mut *out,
            )?;

            if is_selected {
                queue!(out, ResetColor)?;
            }
        }

        // Wipe any rows left over from a previous, taller frame.
        queue!(out, terminal::Clear(ClearType::FromCursorDown))?;

        // Two-line status bar pinned to the bottom (no trailing newline on the
        // last, to avoid scrolling). First line: the selected row's primary text
        // (tensor name / group files / copy confirmation) as a coloured chip;
        // second line: the tensor's source file, dimmed and aligned under the
        // chip text. Both truncate tail-first so the meaningful end stays visible.
        let max_text = (terminal_width as usize).saturating_sub(6);
        queue!(out, cursor::MoveTo(0, terminal_height.saturating_sub(2)))?;
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
            // so the file name stays visible.
            let text = truncate_keep_end(config.status_bar, max_text);
            queue!(
                out,
                SetBackgroundColor(palette::STATUS_BG),
                SetForegroundColor(palette::STATUS_FG)
            )?;
            write!(out, " {} {text} ", config.status_icon)?;
            queue!(out, ResetColor)?;
        }
        // Second line: the source file, dimmed, indented under the chip's text.
        queue!(
            out,
            cursor::MoveTo(0, terminal_height.saturating_sub(1)),
            terminal::Clear(ClearType::CurrentLine)
        )?;
        if !config.status_secondary.is_empty() {
            let text = truncate_keep_end(config.status_secondary, max_text);
            queue!(out, SetForegroundColor(palette::DIM))?;
            write!(out, "   {text}")?;
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
        packing_schemas: &HashMap<String, PackingSchema>,
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
                        "{} {SIZE_ARROW} {}",
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
            TreeNode::Tensor { info, label } => {
                // In search mode (depth 0), show the full name; otherwise the
                // compacted label if this leaf absorbed a single-child folder
                // chain (e.g. `self_attn.k_norm.weight`), else the last segment.
                let display_name = if depth == 0 {
                    info.name.as_str()
                } else if let Some(label) = label {
                    label.as_str()
                } else {
                    crate::tree::last_segment(&info.name)
                };
                // The name, shape and size read at full strength; only the leaf
                // marker and the storage tag (codec / "uncompressed") are dimmed, and the
                // dtype is tinted. `⇩` marks a compressed tensor. A tensor on disk
                // but absent from the index gets a red `✚` (an "extra") instead of
                // the dot.
                // Align the leaf marker with sibling group markers at this depth
                // (the depth already accounts for nesting — no extra indent).
                write!(out, "{indent}")?;
                if unindexed.contains(&info.source_path) {
                    paint(out, selected, palette::UNINDEXED, UNINDEXED_MARK)?;
                } else {
                    paint(out, selected, palette::DIM, "·")?;
                }
                write!(out, " {display_name} [")?;
                paint(out, selected, palette::DTYPE, &info.dtype)?;
                // A fused-codebook tensor carries a logical (unmerged) layout that
                // matters more than its sizes: show `physical as logical` for both
                // the dtype and the shape (e.g. `U16 as u3×5, (26,…) as (130,…)`).
                let schema = packing_schemas.get(&info.name);
                if let Some(s) = schema {
                    paint(out, selected, palette::DIM, " as ")?;
                    paint(out, selected, palette::DTYPE, &s.label())?;
                }
                write!(out, ", {}", format_shape(&info.shape))?;
                if let Some(s) = schema {
                    let logical =
                        ViewDtype::Unpacked.logical_shape_with(&info.shape, &info.dtype, Some(s));
                    paint(out, selected, palette::DIM, " as ")?;
                    write!(out, "{}", format_shape(&logical))?;
                }
                write!(out, ", ")?;
                match &info.storage {
                    Storage::Compressed {
                        codec,
                        stored_bytes,
                    } => {
                        write!(
                            out,
                            "{} {SIZE_ARROW} {} ",
                            format_size(info.size_bytes),
                            format_size(*stored_bytes)
                        )?;
                        paint(
                            out,
                            selected,
                            palette::DIM,
                            &format!("({COMPRESSED_MARK} {codec})"),
                        )?;
                    }
                    Storage::Raw => {
                        write!(out, "{} ", format_size(info.size_bytes))?;
                        paint(out, selected, palette::DIM, UNCOMPRESSED_TAG)?;
                    }
                    Storage::Unknown => write!(out, "{}", format_size(info.size_bytes))?,
                }
                write!(out, "]\r\n")?;
            }
            TreeNode::Metadata { info } => {
                // Collapse the value (which may be multi-line pretty-printed
                // JSON) into a compact one-line preview — otherwise its newlines
                // cascade across the tree. The full value shows in the detail
                // view. Truncate by chars so we never split a UTF-8 boundary.
                let flat = info.value.split_whitespace().collect::<Vec<_>>().join(" ");
                let truncated_value = if flat.chars().count() > 50 {
                    let head: String = flat.chars().take(47).collect();
                    format!("{head}...")
                } else {
                    flat
                };
                // Align the leaf marker with sibling group markers at this depth
                // (the depth already accounts for nesting — no extra indent).
                write!(out, "{indent}")?;
                // Muted symbol + name so the whole row reads as a side note.
                paint(out, selected, palette::META, "†")?;
                write!(out, " ")?;
                paint(out, selected, palette::META, &metadata_short(&info.name))?;
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

    /// Body rows visible in the tree at the given size — used to compute the
    /// scroll offset so it stays consistent with [`Self::render_tree`]'s layout
    /// (header = title + hint/search line(s) + rule; footer = the two status
    /// lines).
    pub fn tree_visible_rows(
        width: u16,
        height: u16,
        search_mode: bool,
        can_repack: bool,
    ) -> usize {
        let hint_rows = if search_mode {
            1
        } else {
            tree_hint_lines(can_repack, width).len()
        };
        let header = 1 + hint_rows + 1; // title + hint(s) + rule
        (height as usize)
            .saturating_sub(header + TREE_FOOTER_HEIGHT)
            .max(1)
    }

    /// Ratatui render of the tree browser: header (title, hint or search line,
    /// rule), the visible tree rows from `config.scroll_offset`, and the bottom
    /// two-line status bar. The Ratatui port of [`Self::draw_screen`]; shares the
    /// same `DrawConfig`.
    pub fn render_tree(frame: &mut Frame, config: &DrawConfig) {
        let area = frame.area();
        let (width, height) = (area.width, area.height);
        if height < (TREE_FOOTER_HEIGHT as u16 + 1) {
            return;
        }

        // --- header + tree rows (the region above the 2-line status bar) ---
        let mut lines: Vec<Line> = Vec::new();

        // Title (+ optional health warning).
        let mut title = vec![Span::raw(format!(
            "Checkpoint Explorer - {} ({}/{})",
            config.current_file,
            config.file_idx + 1,
            config.total_files
        ))];
        if config.health_warning {
            title.push(Span::styled(
                "   ⚠ index/file mismatch — press ",
                Style::default().fg(to_ratatui(palette::ERROR)),
            ));
            title.push(Span::styled(
                "h",
                Style::default()
                    .fg(to_ratatui(palette::KEY))
                    .add_modifier(Modifier::BOLD),
            ));
        }
        lines.push(Line::from(title));

        // Hint line(s), or the search bar when searching.
        if config.search_mode {
            lines.push(tree_search_line(config));
        } else {
            lines.extend(tree_hint_lines(config.can_repack, width));
        }

        // Separator rule.
        lines.push(Line::from(Span::styled(
            "─".repeat(width as usize),
            Style::default().fg(to_ratatui(palette::DIM)),
        )));

        let header_rows = lines.len();
        let body_rows = (height as usize).saturating_sub(header_rows + TREE_FOOTER_HEIGHT);

        // Visible tree rows from the (pre-computed) scroll offset.
        if !(config.search_mode && config.tree.is_empty()) {
            for (idx, (node, depth)) in config
                .tree
                .iter()
                .enumerate()
                .skip(config.scroll_offset)
                .take(body_rows)
            {
                let selected = idx == config.selected_idx;
                lines.push(tree_node_line(
                    node,
                    *depth,
                    selected,
                    config.unindexed,
                    config.packing_schemas,
                ));
            }
        }

        let top = Rect {
            x: 0,
            y: 0,
            width,
            height: height.saturating_sub(TREE_FOOTER_HEIGHT as u16),
        };
        Paragraph::new(lines).render(top, frame.buffer_mut());

        // --- bottom two-line status bar ---
        let max_text = (width as usize).saturating_sub(6);
        let row0 = if config.search_mode && config.tree.is_empty() {
            Line::from(vec![
                Span::raw(format!(
                    "No results found for \"{}\" | Press ",
                    config.search_query
                )),
                Span::styled(
                    "Esc",
                    Style::default()
                        .fg(to_ratatui(palette::KEY))
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(" to exit search"),
            ])
        } else if !config.status_bar.is_empty() {
            let text = truncate_keep_end(config.status_bar, max_text);
            Line::from(Span::styled(
                format!(" {} {text} ", config.status_icon),
                Style::default()
                    .bg(to_ratatui(palette::STATUS_BG))
                    .fg(to_ratatui(palette::STATUS_FG)),
            ))
        } else {
            Line::default()
        };
        Paragraph::new(row0).render(
            Rect {
                x: 0,
                y: height.saturating_sub(2),
                width,
                height: 1,
            },
            frame.buffer_mut(),
        );

        // Second line: a copy confirmation (green) overrides the dimmed source file.
        let row1 = if let Some(flash) = config.copied_flash {
            Line::from(Span::styled(
                format!("✓ Copied {flash} to the clipboard"),
                Style::default()
                    .fg(to_ratatui(palette::SUCCESS))
                    .add_modifier(Modifier::BOLD),
            ))
        } else if !config.status_secondary.is_empty() {
            let text = truncate_keep_end(config.status_secondary, max_text);
            Line::from(Span::styled(
                format!("   {text}"),
                Style::default().fg(to_ratatui(palette::DIM)),
            ))
        } else {
            Line::default()
        };
        Paragraph::new(row1).render(
            Rect {
                x: 0,
                y: height.saturating_sub(1),
                width,
                height: 1,
            },
            frame.buffer_mut(),
        );
    }

    /// Render the tensor detail screen — the Ratatui port of
    /// [`UI::draw_tensor_detail`]. `view` is the active dtype reinterpretation
    /// (which changes the shown dtype, shape and parameter count); `overridable`
    /// gates the `d`/`r` hints. `histogram` adds the value-histogram section below
    /// the header. A pop-up `overlay` (legend / copied command) composites last.
    ///
    /// Header fields are one [`Line`] each (clipped, not wrapped); when a
    /// histogram is present the header pins to the top, the histogram fills the
    /// middle (sized to `h - header - footer - 1`), one blank row separates it from
    /// the footer pinned to the bottom — filling the screen exactly with no scroll.
    /// Without a histogram the header is immediately followed by the footer,
    /// top-aligned.
    #[allow(clippy::too_many_arguments)] // a screen renderer; the params are all distinct
    pub fn render_detail(
        frame: &mut Frame,
        tensor: &TensorInfo,
        shape: &[usize],
        view: ViewDtype,
        overridable: bool,
        unindexed: bool,
        stats: StatsView,
        histogram: Option<&Histogram>,
        hist_scanning: Option<ScanProgress>,
        schema: Option<&PackingSchema>,
        overlay: Option<&Overlay>,
    ) {
        let area = frame.area();
        let (width, height) = (area.width, area.height);

        let header = detail_field_lines(tensor, shape, view, unindexed, stats, schema);
        let footer = detail_footer_lines(overridable, width);
        let header_len = header.len();
        let footer_len = footer.len();

        if let Some(hist) = histogram {
            // Header, then the histogram (capped to the rows the raw renderer's
            // `term_h - body_rows - footer_rows - 1` budget would allow), then a
            // blank spacer, then the footer flowed right below the bars — the raw
            // path wrote these sequentially, so a small histogram leaves the footer
            // near the top while a full one pushes it to the screen's bottom.
            Paragraph::new(header).render(
                Rect {
                    x: 0,
                    y: 0,
                    width,
                    height: header_len as u16,
                },
                frame.buffer_mut(),
            );
            let section = (height as usize)
                .saturating_sub(header_len + footer_len + 1)
                .max(2);
            let used = render_histogram(
                frame,
                Rect {
                    x: 0,
                    y: header_len as u16,
                    width,
                    height: section as u16,
                },
                hist,
                hist_scanning,
            );
            // Footer one blank row below the bars, clamped so it always fits.
            let footer_y =
                (header_len + used + 1).min((height as usize).saturating_sub(footer_len)) as u16;
            Paragraph::new(footer).render(
                Rect {
                    x: 0,
                    y: footer_y,
                    width,
                    height: footer_len as u16,
                },
                frame.buffer_mut(),
            );
        } else {
            // No histogram: header then footer, top-aligned, the rest left blank.
            let mut lines = header;
            lines.extend(footer);
            Paragraph::new(lines).render(area, frame.buffer_mut());
        }

        // A pop-up overlay composites last, over the live frame, so the detail
        // (including a running scan's progress) keeps animating behind it.
        match overlay {
            Some(Overlay::Legend(l)) => Self::render_legend_band(frame, *l),
            Some(Overlay::Command(c)) => Self::render_command_band(frame, c),
            None => {}
        }
    }

    /// Composite the context-sensitive glyph legend over the live detail frame as
    /// a floating, panel-backed band centred vertically — the Ratatui port of
    /// [`Self::write_legend_band`], drawn last so the screen behind keeps
    /// animating. (The raw band is still used by the tree's `show_legend`, the
    /// data views and `--plain` for the non-detail screens.)
    pub fn render_legend_band(frame: &mut Frame, legend: Legend) {
        let content = legend_band_lines(legend);
        // Band = a blank margin row, the content, a blank margin row; centred.
        let mut band: Vec<Line> = vec![Line::default()];
        band.extend(content);
        band.push(Line::default());
        render_panel_band(frame, band);
    }

    /// Composite the copied-CLI-command pop-up over the live detail frame — a
    /// centred, panel-backed band (blank, title, rule, the wrapped command, rule,
    /// footer, blank). The Ratatui port of [`Self::write_command_band`].
    pub fn render_command_band(frame: &mut Frame, command: &str) {
        let term_w = frame.area().width as usize;
        let rule_color = Style::default().fg(to_ratatui(palette::ACCENT));
        let rule = "─".repeat(term_w);

        // blank, header, rule, command(rows), rule, footer, blank.
        let mut band: Vec<Line> = vec![Line::default()];
        // Header: title + copied confirmation.
        band.push(Line::from(vec![
            Span::styled(
                "CLI command",
                Style::default()
                    .fg(to_ratatui(palette::KEY))
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "   ✓ copied to the clipboard",
                Style::default().fg(to_ratatui(palette::SUCCESS)),
            ),
        ]));
        band.push(Line::from(Span::styled(rule.clone(), rule_color)));
        // The command, soft-wrapped at full width onto its own line(s).
        let chars: Vec<char> = command.chars().collect();
        let cmd_rows = chars.len().div_ceil(term_w.max(1)).max(1);
        for r in 0..cmd_rows {
            let seg: String = chars.iter().skip(r * term_w).take(term_w).collect();
            band.push(Line::from(Span::raw(seg)));
        }
        band.push(Line::from(Span::styled(rule, rule_color)));
        band.push(Line::from(dim_span(
            "select the command above to copy it by hand · any key to dismiss",
        )));
        band.push(Line::default());
        render_panel_band(frame, band);
    }

    /// A loading screen shown while the checkpoint structure is read: the same
    /// title line + rule as the tree browser, a spinner where the rows will land,
    /// and a hint pinned to the bottom — so the header/footer are up immediately
    /// and the tree fills into the same frame once the read finishes.
    pub fn draw_loading(
        file: &str,
        total_files: usize,
        spinner: char,
        elapsed: std::time::Duration,
    ) -> Result<()> {
        let mut out = io::stdout();
        let (w, h) = terminal::size()?;
        let (w, h) = (w as usize, h as usize);
        queue!(out, BeginSynchronizedUpdate, cursor::MoveTo(0, 0))?;

        // Each element is positioned with an explicit `MoveTo` so the layout is
        // exact regardless of how the terminal handles the full-width rule's
        // wrap. Header: title line (row 0), full-width rule (row 1).
        queue!(out, terminal::Clear(ClearType::CurrentLine))?;
        write!(out, "Checkpoint Explorer - {file}")?;
        if total_files > 1 {
            queue!(out, SetForegroundColor(palette::DIM))?;
            write!(out, "  (+{} more)", total_files - 1)?;
            queue!(out, ResetColor)?;
        }
        queue!(
            out,
            cursor::MoveTo(0, 1),
            terminal::Clear(ClearType::CurrentLine),
            SetForegroundColor(palette::DIM)
        )?;
        write!(out, "{}", "─".repeat(w))?;
        queue!(out, ResetColor)?;

        // Wipe everything below the header, then the spinner just under it — on
        // the row where the tree's first node will land, so it reads as content
        // loading in place rather than floating mid-screen.
        let spinner_row = 3u16.min(h.saturating_sub(2) as u16);
        queue!(
            out,
            cursor::MoveTo(0, 2),
            terminal::Clear(ClearType::FromCursorDown),
            cursor::MoveTo(2, spinner_row),
            SetForegroundColor(palette::ACCENT)
        )?;
        write!(out, "{spinner} reading checkpoint structure")?;
        queue!(out, ResetColor, SetForegroundColor(palette::DIM))?;
        write!(out, "  ({:.1}s)", elapsed.as_secs_f64())?;
        queue!(out, ResetColor)?;

        // Footer hint pinned to the bottom (no trailing newline → no scroll).
        queue!(
            out,
            cursor::MoveTo(0, h.saturating_sub(1) as u16),
            terminal::Clear(ClearType::CurrentLine),
            SetForegroundColor(palette::DIM)
        )?;
        write!(out, "Press ")?;
        queue!(out, ResetColor)?;
        key_hint(&mut out, "q")?;
        queue!(out, SetForegroundColor(palette::DIM))?;
        write!(out, " to cancel")?;
        queue!(out, ResetColor)?;

        queue!(out, EndSynchronizedUpdate)?;
        out.flush()?;
        Ok(())
    }

    /// Draw the tensor detail screen. `view` is the active dtype reinterpretation
    /// (which changes the shown dtype, shape and parameter count); `overridable`
    /// gates the `d` hint. `histogram` adds the value-histogram section below the
    /// stats. Rendered flicker-free so it can also serve as the live preview
    /// while choosing a dtype in the menu.
    #[allow(clippy::too_many_arguments)] // a screen renderer; the params are all distinct
    pub fn draw_tensor_detail(
        out: &mut impl Write,
        tensor: &TensorInfo,
        shape: &[usize],
        view: ViewDtype,
        overridable: bool,
        unindexed: bool,
        stats: StatsView,
        histogram: Option<&Histogram>,
        hist_scanning: Option<ScanProgress>,
        schema: Option<&PackingSchema>,
        overlay: Option<&Overlay>,
    ) -> Result<()> {
        queue!(out, BeginSynchronizedUpdate, cursor::MoveTo(0, 0))?;

        // The header (title, fields, statistics) is rendered into a buffer so
        // its exact wrapped height can be measured; the histogram below is then
        // sized to the rows that remain, leaving no gap and never scrolling.
        let mut body: Vec<u8> = Vec::new();
        queue!(body, SetForegroundColor(palette::ACCENT))?;
        write!(body, "Tensor Details")?;
        queue!(body, ResetColor, SetForegroundColor(palette::DIM))?;
        line_end(&mut body)?;
        write!(body, "{}", "─".repeat(14))?;
        queue!(body, ResetColor)?;
        line_end(&mut body)?;
        paint(&mut body, false, palette::DIM, "Name: ")?;
        write!(body, "{}", tensor.name)?;
        line_end(&mut body)?;

        // Data type, with the active reinterpretation highlighted.
        let unpacked_label = schema.map(PackingSchema::label);
        paint(&mut body, false, palette::DIM, "Data Type: ")?;
        write_view_dtype(&mut body, &tensor.dtype, view, unpacked_label.as_deref())?;
        line_end(&mut body)?;

        // Shape and parameter count reflect the overrides: `shape` is the
        // effective (possibly reshaped) shape, and a packed dtype view unpacks
        // several values per stored element (the codebook unmerge grows the first
        // dimension, the 4-bit views the last). Show `stored as reinterpreted`.
        let logical = view.logical_shape_with(shape, &tensor.dtype, schema);
        let num_elements: usize = logical.iter().product();
        paint(&mut body, false, palette::DIM, "Shape: ")?;
        write_view_shape(&mut body, &tensor.shape, &logical)?;
        line_end(&mut body)?;
        paint(&mut body, false, palette::DIM, "Parameters: ")?;
        write!(body, "{} ", format_parameters(num_elements))?;
        paint(
            &mut body,
            false,
            palette::DIM,
            &format!("({})", with_thousands(num_elements)),
        )?;
        line_end(&mut body)?;

        // Codebook packing schema disclosure (only for tensors that carry one).
        if let Some(s) = schema {
            paint(&mut body, false, palette::DIM, "Packing: ")?;
            write!(body, "{} ", s.label())?;
            let widths = s
                .bit_widths()
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            let mode = s
                .quant_mode()
                .map(|m| format!(" · {m}"))
                .unwrap_or_default();
            let uniform = if s.uniform_width().is_some() {
                "uniform"
            } else {
                "non-uniform"
            };
            paint(
                &mut body,
                false,
                palette::DIM,
                &format!(
                    "(bit widths [{widths}] · {} experts/word · {uniform}{mode})",
                    s.len_p()
                ),
            )?;
            line_end(&mut body)?;
        }

        paint(&mut body, false, palette::DIM, "Size: ")?;
        write!(body, "{}", format_size(tensor.size_bytes))?;
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
                write!(body, " · on disk: {} ", format_size(*stored_bytes))?;
                paint(
                    &mut body,
                    false,
                    palette::DIM,
                    &format!("({COMPRESSED_MARK} {codec}, {ratio:.1}×)"),
                )?;
            }
            Storage::Raw => {
                write!(
                    body,
                    " · on disk: {} {UNCOMPRESSED_TAG}",
                    format_size(tensor.size_bytes)
                )?;
            }
            Storage::Unknown => {}
        }
        line_end(&mut body)?;
        // Where the data lives within the file.
        match &tensor.layout {
            Layout::ByteRange { start, end } => {
                paint(&mut body, false, palette::DIM, "Data offsets: ")?;
                write!(
                    body,
                    "{} – {}  (within file data)",
                    with_thousands(*start as usize),
                    with_thousands(*end as usize)
                )?;
                line_end(&mut body)?;
            }
            Layout::Offset(offset) => {
                paint(&mut body, false, palette::DIM, "Data offset: ")?;
                write!(
                    body,
                    "{}  (within tensor data)",
                    with_thousands(*offset as usize)
                )?;
                line_end(&mut body)?;
            }
            Layout::Chunked { chunk, num_chunks } => {
                paint(&mut body, false, palette::DIM, "Chunks: ")?;
                write!(
                    body,
                    "{} × {}",
                    format_shape(chunk),
                    with_thousands(*num_chunks)
                )?;
                line_end(&mut body)?;
            }
            Layout::None => {}
        }
        paint(&mut body, false, palette::DIM, "File: ")?;
        write!(body, "{}", tensor.source_path)?;
        line_end(&mut body)?;
        // Flag a tensor that's on disk but absent from the index.
        if unindexed {
            queue!(body, SetForegroundColor(palette::UNINDEXED))?;
            write!(
                body,
                "{UNINDEXED_MARK} on disk but not listed in model.safetensors.index.json"
            )?;
            queue!(body, ResetColor)?;
            line_end(&mut body)?;
        }
        line_end(&mut body)?;

        // Exact whole-tensor statistics: shown once computed, else a hint.
        match stats {
            StatsView::Ready(s) => {
                // min/max are exact integers for integer dtypes — show them as
                // such (no `.0000`).
                let integer = view.is_integer(&tensor.dtype);
                paint(&mut body, false, palette::DIM, "Statistics: ")?;
                write!(
                    body,
                    "min {} · max {} · ",
                    fmt_value(s.min, integer),
                    fmt_value(s.max, integer)
                )?;
                write_stats_line(&mut body, s)?;
            }
            StatsView::Computing {
                spinner,
                elapsed,
                progress,
            } => {
                paint(&mut body, false, palette::DIM, "Statistics: ")?;
                write_computing(&mut body, spinner, elapsed, progress)?;
            }
            StatsView::Pending => {
                queue!(body, SetForegroundColor(palette::DIM))?;
                write!(body, "Statistics: press ")?;
                queue!(body, ResetColor)?;
                key_hint(&mut body, "s")?;
                queue!(body, SetForegroundColor(palette::DIM))?;
                write!(body, " to scan the full tensor")?;
                queue!(body, ResetColor)?;
            }
        }
        line_end(&mut body)?;
        line_end(&mut body)?;

        // Footer hints (keys highlighted) — built into a buffer first so its
        // wrapped height can be measured and the histogram sized to fit exactly.
        let mut footer: Vec<u8> = Vec::new();
        write!(footer, "Press ")?;
        key_hint(&mut footer, "m")?;
        write!(footer, " for a heatmap, ")?;
        key_hint(&mut footer, "v")?;
        write!(footer, " for numeric values, ")?;
        key_hint(&mut footer, "h")?;
        write!(footer, " for a histogram, ")?;
        key_hint(&mut footer, "b")?;
        write!(footer, " to set its bin count, ")?;
        if overridable {
            key_hint(&mut footer, "d")?;
            write!(footer, " to reinterpret the dtype, ")?;
            key_hint(&mut footer, "r")?;
            write!(footer, " to reshape, ")?;
        }
        key_hint(&mut footer, "c")?;
        write!(footer, " to copy, ")?;
        key_hint(&mut footer, "y")?;
        write!(footer, " to copy the command, ")?;
        key_hint(&mut footer, "l")?;
        write!(footer, " for the legend, ")?;
        key_hint(&mut footer, "⌫")?;
        write!(footer, " / ")?;
        key_hint(&mut footer, "\\")?;
        write!(
            footer,
            " to step back / forward, any other key to return..."
        )?;

        out.write_all(&body)?;

        // The whole-tensor value histogram, below the statistics (when computed
        // or being computed), sized to the rows left between the header and the
        // footer — so it fills the screen and never scrolls. Both heights are
        // measured from the rendered bytes. A blank line is left below the bars
        // (it's reserved in the budget) so the footer's key hints don't crowd the
        // last bar — mirroring the blank line above the histogram.
        if let Some(hist) = histogram {
            let (term_w, term_h) = crate::plain::term_size();
            let (tw, th) = (term_w as usize, term_h as usize);
            let body_rows = count_physical_lines(&body, tw);
            let footer_rows = count_physical_lines(&footer, tw);
            let section = th.saturating_sub(body_rows + footer_rows + 1).max(2);
            write_histogram_section(&mut *out, hist, hist_scanning, tw, section)?;
            line_end(&mut *out)?; // blank spacer before the footer
        }

        out.write_all(&footer)?;

        // No trailing newline (avoids scrolling); clear anything below.
        queue!(out, terminal::Clear(ClearType::FromCursorDown))?;
        // A pop-up overlay composites last, over the live frame, so the detail
        // (including a running scan's progress) keeps animating behind it.
        match overlay {
            Some(Overlay::Legend(l)) => Self::write_legend_band(out, *l)?,
            Some(Overlay::Command(c)) => Self::write_command_band(out, c)?,
            None => {}
        }
        queue!(out, EndSynchronizedUpdate)?;
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

        // When the value is a JSON object/array, pretty-print it with syntax
        // highlighting (colors keys/strings/numbers/literals); otherwise show the
        // raw text lines.
        let highlighted = highlight_json(&metadata.value);
        let lines: Vec<String> = match highlighted {
            Some(colored) => colored,
            None => metadata.value.lines().map(str::to_string).collect(),
        };
        writeln!(stdout, "Value:\r")?;

        // Show as many value lines as fit the terminal (the lines above plus a
        // short footer below), noting how many were elided rather than cutting
        // silently — metadata values like a quant config run dozens of lines.
        let rows = terminal::size().map(|(_, h)| h as usize).unwrap_or(40);
        let budget = rows.saturating_sub(8).max(1);
        let shown = lines.len().min(budget);
        for line in &lines[..shown] {
            writeln!(stdout, "  {line}\r")?;
        }
        if lines.len() > shown {
            writeln!(stdout, "  … ({} more lines)\r", lines.len() - shown)?;
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

        write_data_view_title(&mut *out, "Heatmap", tensor)?;
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
        write_view_dtype(
            &mut *out,
            &tensor.dtype,
            sample.view,
            sample.schema_label.as_deref(),
        )?;
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
        write_view_footer(&mut *out, sample, true, StripeMode::Off, NumBase::Decimal)?;

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
        base: NumBase,
    ) -> Result<()> {
        // Cell width adapts to the data: floats need room for scientific
        // notation, while small integers (incl. sparse values in a wide dtype)
        // are 1-3 digits, so we pack many narrow columns onto the screen. The
        // raw-bit bases reserve a fixed width: the dtype's digit count + a gap.
        let cw = base.cell_width(sample.view, &tensor.dtype, stats.value_range());
        // Synchronized, in-place overwrite (see `draw_heatmap`) to avoid flicker.
        queue!(out, BeginSynchronizedUpdate, cursor::MoveTo(0, 0))?;

        write_data_view_title(&mut *out, "Values", tensor)?;
        write_view_dtype(
            &mut *out,
            &tensor.dtype,
            sample.view,
            sample.schema_label.as_deref(),
        )?;
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
                let s = match base {
                    NumBase::Decimal if integer => format!("{:>cw$}", v as i64),
                    NumBase::Decimal => format!("{v:>cw$.3e}"),
                    // The raw stored bits, zero-padded to the dtype's width.
                    _ => {
                        let rb = sample.raw[i][j];
                        let d = base.digits(rb.width as u32);
                        let body = match base {
                            NumBase::Hex => format!("{:0d$x}", rb.bits),
                            NumBase::Octal => format!("{:0d$o}", rb.bits),
                            NumBase::Binary => format!("{:0d$b}", rb.bits),
                            NumBase::Decimal => unreachable!(),
                        };
                        format!("{body:>cw$}")
                    }
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
        write_view_footer(&mut *out, sample, false, stripe, base)?;

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
        input_box(&mut out, input, input.chars().count(), 5)?;
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
        input_box(&mut out, input, input.chars().count(), 16)?;
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
        input_box(&mut out, input, input.chars().count(), 24)?;
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
        // Panel background first, so `Clear(All)` erases the whole screen to the
        // pop-up surface (the text is then written over it).
        execute!(
            stdout,
            SetBackgroundColor(palette::PANEL_BG),
            terminal::Clear(ClearType::All),
            cursor::MoveTo(0, 0)
        )?;
        writeln!(stdout, "{title}\r")?;
        writeln!(stdout, "{}\r", "=".repeat(title.len().max(10)))?;
        writeln!(stdout, "{message}\r")?;
        writeln!(stdout, "\r")?;
        writeln!(stdout, "Press any key to return...\r")?;
        execute!(stdout, ResetColor)?;
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
    /// Flash a "✓ Copied … to the clipboard" confirmation on the bottom line,
    /// over whatever the view drew there, until the next redraw clears it. Shared
    /// by every screen's copy shortcuts (tree, detail, data) so the confirmation
    /// never hides the content above it. `what` names what was copied.
    pub fn draw_copied_flash(what: &str) -> Result<()> {
        let mut out = io::stdout();
        let (term_w, term_h) = terminal::size()?;
        // Clamp to the width so a long message can't wrap off the last row and
        // scroll the frame (this overlays the terminal's final line). Keep the
        // head — the "✓ Copied …" — rather than the tail.
        let full = format!("✓ Copied {what} to the clipboard");
        let width = term_w as usize;
        let msg = if full.chars().count() > width {
            full.chars()
                .take(width.saturating_sub(1))
                .chain(std::iter::once('…'))
                .collect()
        } else {
            full
        };
        queue!(
            out,
            cursor::MoveTo(0, term_h.saturating_sub(1)),
            terminal::Clear(ClearType::CurrentLine),
            SetForegroundColor(palette::SUCCESS),
            SetAttribute(Attribute::Bold)
        )?;
        write!(out, "{msg}")?;
        queue!(out, SetAttribute(Attribute::Reset), ResetColor)?;
        out.flush()?;
        Ok(())
    }

    pub fn draw_command(command: &str) -> Result<()> {
        let mut out = io::stdout();
        queue!(out, BeginSynchronizedUpdate)?;
        Self::write_command_band(&mut out, command)?;
        queue!(out, EndSynchronizedUpdate)?;
        out.flush()?;
        Ok(())
    }

    /// Composite the copied-command box onto `out` (it does not bracket its own
    /// synchronized update), centred vertically as a floating pop-up. Shared by
    /// [`Self::draw_command`] and the overlay layer of
    /// [`Self::draw_tensor_detail`].
    fn write_command_band(mut out: &mut impl Write, command: &str) -> Result<()> {
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
        // Panel background: the band's cleared rows then erase to the pop-up
        // surface, lifting it off the view above and below.
        begin_panel(out)?;

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
            // `Attribute::Reset` clears the bold *and* the panel background, so
            // restore the surface before continuing the band.
            SetAttribute(Attribute::Reset),
            SetBackgroundColor(palette::PANEL_BG),
            SetForegroundColor(palette::SUCCESS)
        )?;
        write!(out, "   ✓ copied to the clipboard")?;
        reset_fg(&mut out)?;
        row += 1;

        // Opening rule.
        queue!(
            out,
            cursor::MoveTo(0, row),
            terminal::Clear(clear),
            SetForegroundColor(palette::ACCENT)
        )?;
        write!(out, "{rule}")?;
        reset_fg(&mut out)?;
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
        reset_fg(&mut out)?;
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
        reset_fg(&mut out)?;
        row += 1;

        // A cleared margin row below.
        queue!(out, cursor::MoveTo(0, row), terminal::Clear(clear))?;

        end_panel(&mut out)?;
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
    /// cues) on whichever screen the user opened it from (`l`). A flicker-free
    /// floating band centred over the current screen — not a full-screen
    /// takeover — so the surrounding view stays visible; the caller waits for a
    /// key, then redraws its own screen over it.
    pub fn draw_legend(legend: Legend) -> Result<()> {
        let stdout = io::stdout();
        let mut out = BufWriter::new(stdout.lock());
        queue!(out, BeginSynchronizedUpdate)?;
        Self::write_legend_band(&mut out, legend)?;
        queue!(out, EndSynchronizedUpdate)?;
        out.flush()?;
        Ok(())
    }

    /// Composite the legend band onto `dst` (it does not bracket its own
    /// synchronized update), centred vertically as a floating pop-up. Used both
    /// standalone (by [`Self::draw_legend`]) and as the overlay layer of
    /// [`Self::draw_tensor_detail`], drawn last over the live detail frame so the
    /// screen behind keeps animating; also appended after a `--plain` screen to
    /// render `--legend`.
    pub fn write_legend_band(dst: &mut impl Write, legend: Legend) -> Result<()> {
        // Render the legend body into a buffer first so it can be centred as a
        // floating band on replay. The body writes plain content — each line
        // ending in a newline — while the panel background and positioning are
        // applied to `dst` below.
        let mut out: Vec<u8> = Vec::new();

        let title = match legend {
            Legend::Tree => "Legend — checkpoint tree",
            Legend::Detail => "Legend — tensor details",
            Legend::Heatmap => "Legend — heatmap",
            Legend::Values => "Legend — numeric values",
        };
        queue!(out, SetForegroundColor(palette::ACCENT))?;
        write!(out, "{title}")?;
        queue!(out, SetForegroundColor(palette::DIM))?;
        line_end(&mut out)?;
        write!(out, "{}", "─".repeat(title.chars().count()))?;
        reset_fg(&mut out)?;
        line_end(&mut out)?;
        line_end(&mut out)?;

        match legend {
            Legend::Tree => {
                // Example symbols built from the shared glyph constants so the
                // legend matches what the tree actually renders.
                let size_example = format!("A {SIZE_ARROW} B");
                let codec_example = format!("{COMPRESSED_MARK} lz4");
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
                    (
                        Some(palette::META),
                        "†",
                        "a metadata entry (shown beside its tensor, or in the Metadata group)",
                    ),
                    (
                        None,
                        "☰ N",
                        "number of layers (numbered sub-groups) in the group",
                    ),
                    (None, "▦ N", "number of tensors in the group / checkpoint"),
                    (
                        None,
                        size_example.as_str(),
                        "logical size → on-disk size (shown only when they differ)",
                    ),
                    (
                        Some(palette::DIM),
                        codec_example.as_str(),
                        "compressed on disk; the codec is named after the glyph",
                    ),
                    (
                        Some(palette::DIM),
                        UNCOMPRESSED_TAG,
                        "stored uncompressed on disk",
                    ),
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
                let codec_example = format!("{COMPRESSED_MARK} lz4");
                let rows = [
                    (
                        Some(palette::DIM),
                        codec_example.as_str(),
                        "on-disk compression codec; the N× beside it is the ratio (logical ÷ stored)",
                    ),
                    (
                        Some(palette::KEY),
                        "as",
                        "the active dtype reinterpretation (press d), e.g. 'BF16 as u4'",
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
                reset_fg(&mut out)?;
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
                reset_fg(&mut out)?;
                for i in 0..24 {
                    queue!(out, SetForegroundColor(heat_color(i as f64 / 23.0)))?;
                    write!(out, "█")?;
                }
                queue!(out, SetForegroundColor(palette::DIM))?;
                write!(out, " high")?;
                reset_fg(&mut out)?;
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
                    (
                        None,
                        "3f800000",
                        "press b to cycle the base: dec / hex / oct / bin (raw stored bits)",
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
                // Back to the pop-up surface (not the terminal default) for the
                // description that follows.
                queue!(out, SetBackgroundColor(palette::PANEL_BG))?;
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
        reset_fg(&mut out)?;
        line_end(&mut out)?;

        // Replay the buffered legend as a floating band centred over the current
        // screen: a panel-filled row per content line, framed by a blank margin
        // row above and below, with the surrounding view left untouched. Every
        // line ends in a newline, so the newline count is the content height.
        let lines = out.iter().filter(|&&b| b == b'\n').count();
        let (_w, h) = crate::plain::term_size();
        let band_h = lines + 2;
        let start = ((h as usize).saturating_sub(band_h) / 2) as u16;

        begin_panel(dst)?;
        // Blank margin row above the content (cleared to the panel surface).
        queue!(
            dst,
            cursor::MoveTo(0, start),
            terminal::Clear(ClearType::CurrentLine)
        )?;
        // The content, then a blank margin row below it — the cursor lands there
        // after the footer's trailing newline.
        queue!(dst, cursor::MoveTo(0, start + 1))?;
        dst.write_all(&out)?;
        queue!(dst, terminal::Clear(ClearType::CurrentLine))?;
        end_panel(dst)?;
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

/// One legend row as a styled [`Line`]: a two-space indent, the `symbol` (in
/// `color`, else default), then the description starting at absolute column
/// `desc_col` — the Ratatui port of [`legend_line`]. The gap is filled with
/// spaces sized to the symbol's *rendered* display width (so the description
/// lines up like the raw `MoveToColumn`). An all-empty row is a blank separator.
fn legend_row_line(color: Option<Color>, symbol: &str, desc: &str, desc_col: u16) -> Line<'static> {
    use unicode_width::UnicodeWidthStr;
    if symbol.is_empty() && desc.is_empty() {
        return Line::default();
    }
    let mut spans: Vec<Span> = vec![Span::raw("  ")];
    match color {
        Some(c) => spans.push(Span::styled(
            symbol.to_string(),
            Style::default().fg(to_ratatui(c)),
        )),
        None => spans.push(Span::raw(symbol.to_string())),
    }
    // Pad from the current column (2 + rendered symbol width) to `desc_col`.
    let used = 2 + symbol.width();
    let pad = (desc_col as usize).saturating_sub(used).max(1);
    spans.push(Span::raw(" ".repeat(pad)));
    spans.push(Span::raw(desc.to_string()));
    Line::from(spans)
}

/// Composite `lines` as a floating, panel-backed band centred vertically over the
/// frame — shared by [`UI::render_legend_band`] and [`UI::render_command_band`].
/// Every band row is padded to the full width with panel-background spaces so it
/// fully overwrites the screen beneath (symbols *and* colour), reading as a raised
/// pop-up — the Ratatui equivalent of the raw bands' full-width line clears.
fn render_panel_band(frame: &mut Frame, lines: Vec<Line<'static>>) {
    use unicode_width::UnicodeWidthStr;
    let area = frame.area();
    let width = area.width as usize;
    let panel = Style::default().bg(to_ratatui(palette::PANEL_BG));
    let band_h = lines.len() as u16;
    let start = area.height.saturating_sub(band_h) / 2;

    // Pad each line to the full width with panel-styled spaces so the cells under
    // the band (the live frame's symbols) are overwritten, not just recoloured.
    let padded: Vec<Line> = lines
        .into_iter()
        .map(|mut line| {
            let used: usize = line.spans.iter().map(|s| s.content.width()).sum();
            if used < width {
                line.spans
                    .push(Span::styled(" ".repeat(width - used), panel));
            }
            line.style(panel)
        })
        .collect();
    Paragraph::new(padded).style(panel).render(
        Rect {
            x: 0,
            y: start,
            width: area.width,
            height: band_h.min(area.height.saturating_sub(start)),
        },
        frame.buffer_mut(),
    );
}

/// Build the context-sensitive legend body as styled [`Line`]s — the Ratatui port
/// of the body [`UI::write_legend_band`] composes. Title, rule, blank, the
/// per-screen rows, then the closing blank + "Press any key to close." footer.
fn legend_band_lines(legend: Legend) -> Vec<Line<'static>> {
    let title = match legend {
        Legend::Tree => "Legend — checkpoint tree",
        Legend::Detail => "Legend — tensor details",
        Legend::Heatmap => "Legend — heatmap",
        Legend::Values => "Legend — numeric values",
    };
    let mut lines: Vec<Line> = vec![
        Line::from(Span::styled(
            title.to_string(),
            Style::default().fg(to_ratatui(palette::ACCENT)),
        )),
        Line::from(dim_span("─".repeat(title.chars().count()))),
        Line::default(),
    ];

    match legend {
        Legend::Tree => {
            let size_example = format!("A {SIZE_ARROW} B");
            let codec_example = format!("{COMPRESSED_MARK} lz4");
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
                (
                    Some(palette::META),
                    "†",
                    "a metadata entry (shown beside its tensor, or in the Metadata group)",
                ),
                (
                    None,
                    "☰ N",
                    "number of layers (numbered sub-groups) in the group",
                ),
                (None, "▦ N", "number of tensors in the group / checkpoint"),
                (
                    None,
                    size_example.as_str(),
                    "logical size → on-disk size (shown only when they differ)",
                ),
                (
                    Some(palette::DIM),
                    codec_example.as_str(),
                    "compressed on disk; the codec is named after the glyph",
                ),
                (
                    Some(palette::DIM),
                    UNCOMPRESSED_TAG,
                    "stored uncompressed on disk",
                ),
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
                lines.push(legend_row_line(color, sym, desc, col));
            }
        }
        Legend::Detail => {
            let codec_example = format!("{COMPRESSED_MARK} lz4");
            let rows = [
                (
                    Some(palette::DIM),
                    codec_example.as_str(),
                    "on-disk compression codec; the N× beside it is the ratio (logical ÷ stored)",
                ),
                (
                    Some(palette::KEY),
                    "as",
                    "the active dtype reinterpretation (press d), e.g. 'BF16 as u4'",
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
                lines.push(legend_row_line(color, sym, desc, col));
            }
            lines.push(legend_row_line(None, "", "", col));
            lines.push(Line::from(dim_span(
                "  Statistics:  zeros = fraction of exactly-zero values · non-finite = count of NaN/∞",
            )));
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
                lines.push(legend_row_line(color, sym, desc, col));
            }
            // The actual colour ramp, so the scale is unambiguous.
            let mut ramp: Vec<Span> = vec![Span::raw("  "), dim_span("low ")];
            for i in 0..24 {
                ramp.push(Span::styled(
                    "█",
                    Style::default().fg(to_ratatui(heat_color(i as f64 / 23.0))),
                ));
            }
            ramp.push(dim_span(" high"));
            ramp.push(Span::raw(
                "   colour scale: cool = low value, warm = high value",
            ));
            lines.push(Line::from(ramp));
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
                (
                    None,
                    "3f800000",
                    "press b to cycle the base: dec / hex / oct / bin (raw stored bits)",
                ),
            ];
            // Reserve room for the wider zebra swatch row drawn below.
            let col = legend_desc_col(&rows, 8);
            for (color, sym, desc) in rows {
                lines.push(legend_row_line(color, sym, desc, col));
            }
            // A live zebra swatch, since it is a background cue, not a glyph.
            let mut swatch: Vec<Span> = vec![
                Span::raw("  "),
                Span::styled(
                    " 12 ",
                    Style::default().bg(to_ratatui(palette::STRIPE_DARK)),
                ),
                Span::styled(
                    " 34 ",
                    Style::default().bg(to_ratatui(palette::STRIPE_LITE)),
                ),
            ];
            // Pad to the description column (the swatch is 2 + 8 = 10 cells wide).
            let pad = (col as usize).saturating_sub(2 + 8).max(1);
            swatch.push(Span::raw(" ".repeat(pad)));
            swatch.push(Span::raw(
                "zebra striping traces a row or column (cycle rows/cols/off with z)",
            ));
            lines.push(Line::from(swatch));
        }
    }

    lines.push(Line::default());
    lines.push(Line::from(dim_span("Press any key to close.")));
    lines
}

/// Start a floating pop-up panel: set the distinct [`palette::PANEL_BG`] so the
/// overlay reads as a raised surface. While it is active, every `Clear` fills
/// with the panel background (background-colour erase) and written cells inherit
/// it, so the whole panel — content, gaps, and cleared margins — is one surface.
/// Reset foregrounds with [`reset_fg`] (not `ResetColor`) so the panel persists,
/// and finish with [`end_panel`].
fn begin_panel(out: &mut impl Write) -> Result<()> {
    queue!(out, SetBackgroundColor(palette::PANEL_BG))?;
    Ok(())
}

/// Reset only the foreground colour, keeping the pop-up panel background so a
/// following `Clear` still erases to the panel surface rather than to the
/// terminal default. Use inside a [`begin_panel`] region in place of `ResetColor`.
fn reset_fg(out: &mut impl Write) -> Result<()> {
    queue!(out, SetForegroundColor(Color::Reset))?;
    Ok(())
}

/// Finish a [`begin_panel`] region: clear colours so the panel background does
/// not leak into later writes. Cells already painted keep the panel surface
/// until the caller redraws its own screen over the dismissed pop-up.
fn end_panel(out: &mut impl Write) -> Result<()> {
    queue!(out, ResetColor)?;
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
    match sample.unpacked_field {
        // The codebook unmerge: each logical expert is a field unmerged from a
        // stored word, so spell out the mapping rather than "fixed leading index".
        Some(f) => write!(
            out,
            "expert {} of {} — stored word {}, field {}/{} ({}-bit) — ",
            sample.slice, sample.slices, f.stored_slice, f.field, f.len_p, f.field_bits,
        )?,
        None => write!(
            out,
            "slice {} of {} (fixed leading index) — ",
            sample.slice, sample.slices
        )?,
    }
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
/// The footer hint items for a data view — shared by the renderer and the
/// height calculation so the two can't drift. Depends only on values known
/// before sampling (layout mode, slice count, whether the dtype is overridable,
/// the representation, and the zebra/base toggles).
fn view_footer_items(
    mode: SampleMode,
    slices: usize,
    overridable: bool,
    heatmap: bool,
    stripe: StripeMode,
    base: NumBase,
) -> Vec<(&'static str, &'static str)> {
    let switch = if heatmap {
        ("v", "numeric values")
    } else {
        ("m", "heatmap")
    };
    let mut items = vec![switch];
    let edges = matches!(mode, SampleMode::Edges { .. });
    let window = matches!(mode, SampleMode::Window { .. });
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
    if slices > 1 {
        if edges || window {
            items.push(("[ ]", "slice"));
        } else {
            items.push(("← →", "step"));
            items.push(("Shift+← →", "jump 5%"));
        }
        items.push(("/", "index or %"));
    }
    if overridable {
        items.push(("d", "dtype"));
        items.push(("r", "reshape"));
    }
    // Cycle the layout overview → edges → window → overview; the label names the
    // layout `e` switches to next.
    items.push(match mode {
        SampleMode::Grid => ("e", "edges"),
        SampleMode::Edges { .. } => ("e", "window"),
        SampleMode::Window { .. } => ("e", "overview"),
    });
    // Cycle the zebra striping / numeral base (numeric grid only).
    if !heatmap {
        items.push(match stripe {
            StripeMode::Rows => ("z", "zebra: rows"),
            StripeMode::Cols => ("z", "zebra: cols"),
            StripeMode::Off => ("z", "zebra: off"),
        });
        items.push(match base {
            NumBase::Decimal => ("b", "base: dec"),
            NumBase::Hex => ("b", "base: hex"),
            NumBase::Octal => ("b", "base: oct"),
            NumBase::Binary => ("b", "base: bin"),
        });
    }
    items.push(("c", "copy"));
    items.push(("y", "copy cmd"));
    items.push(("l", "legend"));
    items.push(("⌫", "back"));
    items.push(("\\", "fwd"));
    items.push(("", "any other key to return..."));
    items
}

/// Physical lines the data view footer occupies at `width`: the blank spacer
/// line above it plus the hint line (one logical line the terminal auto-wraps).
/// Used to size the grid so the header (tensor name + file) never scrolls off.
pub fn data_view_footer_lines(
    mode: SampleMode,
    slices: usize,
    overridable: bool,
    heatmap: bool,
    stripe: StripeMode,
    base: NumBase,
    width: usize,
) -> usize {
    let items = view_footer_items(mode, slices, overridable, heatmap, stripe, base);
    // Visible width: each item is `key`, `label`, or `key label`, joined by " · ".
    let len: usize = items
        .iter()
        .enumerate()
        .map(|(i, (k, l))| {
            let sep = usize::from(i > 0) * 3;
            let body = if k.is_empty() {
                l.chars().count()
            } else if l.is_empty() {
                k.chars().count()
            } else {
                k.chars().count() + 1 + l.chars().count()
            };
            sep + body
        })
        .sum();
    1 + len.div_ceil(width.max(1)).max(1)
}

fn write_view_footer(
    out: &mut impl Write,
    sample: &Sample,
    heatmap: bool,
    stripe: StripeMode,
    base: NumBase,
) -> Result<()> {
    let items = view_footer_items(
        sample.mode,
        sample.slices,
        sample.overridable,
        heatmap,
        stripe,
        base,
    );
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
/// by the search bar and the slice-jump prompt so every input box matches. The
/// block cursor sits at character index `cursor` (in `0..=len`): inside the text
/// it inverts the character under it, at the end it is a trailing block.
fn input_box(out: &mut impl Write, text: &str, cursor: usize, min_chars: usize) -> Result<()> {
    queue!(
        out,
        SetBackgroundColor(palette::INPUT_BG),
        SetForegroundColor(palette::INPUT_FG)
    )?;
    let chars: Vec<char> = text.chars().collect();
    let cursor = cursor.min(chars.len());
    write!(out, " ")?;
    for (i, ch) in chars.iter().enumerate() {
        if i == cursor {
            // Invert the input colours to draw the caret over this character.
            queue!(
                out,
                SetBackgroundColor(palette::INPUT_FG),
                SetForegroundColor(palette::INPUT_BG)
            )?;
            write!(out, "{ch}")?;
            queue!(
                out,
                SetBackgroundColor(palette::INPUT_BG),
                SetForegroundColor(palette::INPUT_FG)
            )?;
        } else {
            write!(out, "{ch}")?;
        }
    }
    if cursor >= chars.len() {
        write!(out, "█")?;
    }
    for _ in chars.len()..min_chars {
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
/// the active reinterpretation, e.g. a dimmed `BF16 as` then a bold `u4`.
fn write_view_dtype(
    out: &mut impl Write,
    stored: &str,
    view: ViewDtype,
    unpacked_label: Option<&str>,
) -> Result<()> {
    // The codebook unmerge shows the schema-derived label (e.g. `u3×5`) instead
    // of the generic `unpacked`.
    let label: Option<String> = match (view, unpacked_label) {
        (ViewDtype::Unpacked, Some(l)) => Some(format!("{l} (unpacked)")),
        _ => view.label().map(str::to_string),
    };
    match label.as_deref() {
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

/// Translate a palette [`Color`] to the equivalent `yansi` color, so the JSON
/// highlighter can be styled from the same constants as the rest of the UI.
fn to_yansi(color: Color) -> yansi::Color {
    use yansi::Color as Y;
    match color {
        Color::AnsiValue(n) => Y::Fixed(n),
        Color::Rgb { r, g, b } => Y::Rgb(r, g, b),
        Color::Black => Y::Black,
        Color::DarkGrey => Y::BrightBlack,
        Color::Red | Color::DarkRed => Y::Red,
        Color::Green | Color::DarkGreen => Y::Green,
        Color::Yellow | Color::DarkYellow => Y::Yellow,
        Color::Blue | Color::DarkBlue => Y::Blue,
        Color::Magenta | Color::DarkMagenta => Y::Magenta,
        Color::Cyan | Color::DarkCyan => Y::Cyan,
        Color::White | Color::Grey => Y::White,
        _ => Y::Primary,
    }
}

/// JSON highlighting styled from the app palette, so a metadata config reads in
/// the same colors as the rest of the UI: keys in the structural cyan accent
/// (like tree groups), numbers in the amber dtype color, strings green, and the
/// brackets/colons dimmed so the structure recedes behind the values.
fn json_styler() -> colored_json::Styler {
    let dim = to_yansi(palette::DIM).foreground();
    colored_json::Styler {
        object_brackets: dim,
        object_colon: dim,
        array_brackets: dim,
        key: to_yansi(palette::ACCENT).bold(),
        string_value: to_yansi(palette::SUCCESS).foreground(),
        integer_value: to_yansi(palette::DTYPE).foreground(),
        float_value: to_yansi(palette::DTYPE).foreground(),
        bool_value: to_yansi(palette::WARN).foreground(),
        nil_value: dim,
        string_include_quotation: true,
    }
}

/// One styled span for a tree row: the kind's color normally, or the selection
/// highlight (black on white) when the row is selected (so the highlight reads
/// cleanly over the whole row, matching the old inverse-video selection).
fn tree_span(selected: bool, color: Color, text: impl Into<String>) -> Span<'static> {
    let style = if selected {
        Style::default()
            .fg(to_ratatui(palette::SELECT_FG))
            .bg(to_ratatui(palette::SELECT_BG))
    } else {
        Style::default().fg(to_ratatui(color))
    };
    Span::styled(text.into(), style)
}

/// The tree browser's key-hint line(s), word-wrapped to `width` on the
/// ` · `-separated `key label` chips (the long hint spills onto a second line).
fn tree_hint_lines(can_repack: bool, width: u16) -> Vec<Line<'static>> {
    let mut items: Vec<(&str, &str)> = vec![
        ("↑/↓", "navigate"),
        ("←/→", "parent/child"),
        ("Shift+↑/↓", "sibling"),
        ("Enter/Space", "expand"),
        ("E/C", "all"),
        ("/", "search"),
        ("l", "legend"),
        ("c", "copy screen"),
        ("f", "copy file"),
        ("n", "copy name"),
        ("y", "copy command"),
        ("⌫/\\", "back/fwd"),
    ];
    if can_repack {
        items.push(("R", "repack"));
    }
    items.push(("q", "quit"));

    let width = width as usize;
    let key_style = Style::default()
        .fg(to_ratatui(palette::KEY))
        .add_modifier(Modifier::BOLD);
    let sep_style = Style::default().fg(to_ratatui(palette::DIM));
    let mut lines: Vec<Line> = Vec::new();
    let mut spans: Vec<Span> = Vec::new();
    let mut col = 0usize;
    for (key, label) in items {
        let item_w = key.chars().count() + 1 + label.chars().count();
        let has_prev = !spans.is_empty();
        if has_prev && col + 3 + item_w > width {
            lines.push(Line::from(std::mem::take(&mut spans)));
            col = 0;
        }
        if !spans.is_empty() {
            spans.push(Span::styled(" · ", sep_style));
            col += 3;
        }
        spans.push(Span::styled(key.to_string(), key_style));
        spans.push(Span::raw(format!(" {label}")));
        col += item_w;
    }
    if !spans.is_empty() {
        lines.push(Line::from(spans));
    }
    lines
}

/// The search bar header line: `Search [query▒]  N matches  Enter view · …`.
fn tree_search_line(config: &DrawConfig) -> Line<'static> {
    let dim = Style::default().fg(to_ratatui(palette::DIM));
    let key_style = Style::default()
        .fg(to_ratatui(palette::KEY))
        .add_modifier(Modifier::BOLD);
    let mut spans: Vec<Span> = vec![Span::styled("Search ", dim)];

    // Input box: leading space, the query, a caret block when the cursor is at
    // the end, padded to a minimum width, then a trailing space.
    let q = config.search_query;
    let qlen = q.chars().count();
    let mut boxed = String::from(" ");
    boxed.push_str(q);
    if config.search_cursor >= qlen {
        boxed.push('█');
    }
    for _ in qlen..16 {
        boxed.push(' ');
    }
    boxed.push(' ');
    spans.push(Span::styled(
        boxed,
        Style::default()
            .bg(to_ratatui(palette::INPUT_BG))
            .fg(to_ratatui(palette::INPUT_FG)),
    ));

    if q.is_empty() {
        spans.push(Span::raw("  "));
    } else {
        let n = config.tree.len();
        spans.push(Span::styled(
            format!("  {n} {}  ", if n == 1 { "match" } else { "matches" }),
            dim,
        ));
    }
    for (i, (key, label)) in [("Enter", "view"), ("Tab", "in tree"), ("Esc", "exit")]
        .iter()
        .enumerate()
    {
        if i > 0 {
            spans.push(Span::styled(" · ", dim));
        }
        spans.push(Span::styled(key.to_string(), key_style));
        spans.push(Span::raw(format!(" {label}")));
    }
    Line::from(spans)
}

/// One tree row as a styled [`Line`] — the Ratatui port of [`UI::draw_node`].
fn tree_node_line(
    node: &TreeNode,
    depth: usize,
    selected: bool,
    unindexed: &HashSet<String>,
    packing_schemas: &HashMap<String, PackingSchema>,
) -> Line<'static> {
    let indent = "  ".repeat(depth);
    let plain = |t: String| tree_span(selected, Color::Reset, t);
    let mut s: Vec<Span> = vec![tree_span(selected, Color::Reset, indent)];

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
            let arrow = if *expanded { "▾" } else { "▸" };
            let layer_prefix = match layer_count(children) {
                Some(n) => format!("☰ {n}, "),
                None => String::new(),
            };
            let size_field = if stored_size != total_size {
                format!(
                    "{} {SIZE_ARROW} {}",
                    format_size(*total_size),
                    format_size(*stored_size)
                )
            } else {
                format_size(*total_size)
            };
            s.push(tree_span(selected, palette::ACCENT, arrow));
            s.push(tree_span(selected, Color::Reset, " "));
            s.push(tree_span(selected, palette::ACCENT, name.clone()));
            let meta = if depth == 0 {
                format!(
                    " (▦ {tensor_count}, {} params, {size_field})",
                    format_parameters(*params)
                )
            } else {
                format!(" ({layer_prefix}▦ {tensor_count}, {size_field})")
            };
            s.push(plain(meta));
        }
        TreeNode::Tensor { info, label } => {
            let display_name = if depth == 0 {
                info.name.as_str()
            } else if let Some(label) = label {
                label.as_str()
            } else {
                crate::tree::last_segment(&info.name)
            };
            if unindexed.contains(&info.source_path) {
                s.push(tree_span(selected, palette::UNINDEXED, UNINDEXED_MARK));
            } else {
                s.push(tree_span(selected, palette::DIM, "·"));
            }
            s.push(plain(format!(" {display_name} [")));
            s.push(tree_span(selected, palette::DTYPE, info.dtype.clone()));
            let schema = packing_schemas.get(&info.name);
            if let Some(sc) = schema {
                s.push(tree_span(selected, palette::DIM, " as "));
                s.push(tree_span(selected, palette::DTYPE, sc.label()));
            }
            s.push(plain(format!(", {}", format_shape(&info.shape))));
            if let Some(sc) = schema {
                let logical =
                    ViewDtype::Unpacked.logical_shape_with(&info.shape, &info.dtype, Some(sc));
                s.push(tree_span(selected, palette::DIM, " as "));
                s.push(plain(format_shape(&logical)));
            }
            s.push(plain(", ".to_string()));
            match &info.storage {
                Storage::Compressed {
                    codec,
                    stored_bytes,
                } => {
                    s.push(plain(format!(
                        "{} {SIZE_ARROW} {} ",
                        format_size(info.size_bytes),
                        format_size(*stored_bytes)
                    )));
                    s.push(tree_span(
                        selected,
                        palette::DIM,
                        format!("({COMPRESSED_MARK} {codec})"),
                    ));
                }
                Storage::Raw => {
                    s.push(plain(format!("{} ", format_size(info.size_bytes))));
                    s.push(tree_span(selected, palette::DIM, UNCOMPRESSED_TAG));
                }
                Storage::Unknown => s.push(plain(format_size(info.size_bytes))),
            }
            s.push(plain("]".to_string()));
        }
        TreeNode::Metadata { info } => {
            let flat = info.value.split_whitespace().collect::<Vec<_>>().join(" ");
            let truncated_value = if flat.chars().count() > 50 {
                let head: String = flat.chars().take(47).collect();
                format!("{head}...")
            } else {
                flat
            };
            s.push(tree_span(selected, palette::META, "†"));
            s.push(tree_span(selected, Color::Reset, " "));
            s.push(tree_span(
                selected,
                palette::META,
                metadata_short(&info.name),
            ));
            s.push(tree_span(
                selected,
                palette::DIM,
                format!(" [{}]: {truncated_value}", info.value_type),
            ));
        }
    }
    Line::from(s)
}

/// A dimmed span (field labels, chrome) for the detail screen.
fn dim_span(text: impl Into<String>) -> Span<'static> {
    Span::styled(text.into(), Style::default().fg(to_ratatui(palette::DIM)))
}

/// A bold bright-cyan key span (e.g. `s`, `d`) — the Ratatui equivalent of the
/// raw [`key_hint`].
fn key_span(key: impl Into<String>) -> Span<'static> {
    Span::styled(
        key.into(),
        Style::default()
            .fg(to_ratatui(palette::KEY))
            .add_modifier(Modifier::BOLD),
    )
}

/// Build the detail screen's dtype span(s): the stored dtype plain, or — when a
/// view reinterpretation is active — a dimmed `stored as` followed by the bold
/// reinterpretation label. The Ratatui port of [`write_view_dtype`].
fn detail_dtype_spans(
    stored: &str,
    view: ViewDtype,
    unpacked_label: Option<&str>,
) -> Vec<Span<'static>> {
    let label: Option<String> = match (view, unpacked_label) {
        (ViewDtype::Unpacked, Some(l)) => Some(format!("{l} (unpacked)")),
        _ => view.label().map(str::to_string),
    };
    match label {
        Some(label) => vec![dim_span(format!("{stored} as ")), key_span(label)],
        None => vec![Span::raw(stored.to_string())],
    }
}

/// Build the detail screen's shape span(s): the (unchanged) shape plain, or a
/// dimmed `stored as` followed by the bold reinterpreted shape. Port of
/// [`write_view_shape`].
fn detail_shape_spans(stored: &[usize], logical: &[usize]) -> Vec<Span<'static>> {
    if stored == logical {
        vec![Span::raw(format_shape(logical))]
    } else {
        vec![
            dim_span(format!("{} as ", format_shape(stored))),
            key_span(format_shape(logical)),
        ]
    }
}

/// The one-line statistics summary (mean, std, sparsity, non-finite count) as
/// styled spans — the Ratatui port of [`write_stats_line`]. Field labels dimmed;
/// the non-finite count highlighted (warn) when nonzero.
fn detail_stats_summary_spans(s: &Stats) -> Vec<Span<'static>> {
    let mut spans = vec![
        dim_span("mean "),
        Span::raw(format!("{:.4}", s.mean)),
        dim_span(" · std "),
        Span::raw(format!("{:.4}", s.std)),
        dim_span(" · zeros "),
    ];
    // Distinguish "no zeros" from "a tiny fraction" (which would round to a
    // misleading `0.0%`), matching the raw line.
    let pct = s.zero_fraction() * 100.0;
    let zeros = if s.zeros == 0 {
        "0%".to_string()
    } else if pct < 0.1 {
        format!("{pct:.1e}%")
    } else {
        format!("{pct:.1}%")
    };
    spans.push(Span::raw(zeros));
    if s.nonfinite > 0 {
        spans.push(Span::styled(
            format!(" · {} non-finite", s.nonfinite),
            Style::default().fg(to_ratatui(palette::WARN)),
        ));
    }
    spans.push(dim_span(format!("  ({})", fmt_duration(s.elapsed))));
    spans
}

/// The "scan in progress" stats segment as styled spans — Ratatui port of
/// [`write_computing`]: an accent spinner, a dimmed label, a progress bar with a
/// percentage (when the fraction is known), and the running elapsed time.
fn detail_computing_spans(
    spinner: char,
    elapsed: Duration,
    progress: Option<f64>,
) -> Vec<Span<'static>> {
    let mut spans = vec![
        key_span(format!("{spinner} ")),
        dim_span("computing statistics… "),
    ];
    if let Some(frac) = progress {
        const WIDTH: usize = 16;
        let frac = frac.clamp(0.0, 1.0);
        let filled = (frac * WIDTH as f64).round() as usize;
        spans.push(Span::raw("["));
        spans.push(key_span("█".repeat(filled)));
        spans.push(dim_span("░".repeat(WIDTH - filled)));
        spans.push(Span::raw(format!("] {:>3.0}% · ", frac * 100.0)));
    }
    spans.push(Span::raw(fmt_duration(elapsed)));
    spans
}

/// Build the detail screen's header field lines (title + rule, Name, Data Type,
/// Shape, Parameters, optional Packing, Size [+ on-disk/codec], offsets/Chunks,
/// File, optional unindexed flag, blank, Statistics, blank) — one [`Line`] each,
/// clipped (not wrapped) by the caller's `Paragraph`.
fn detail_field_lines(
    tensor: &TensorInfo,
    shape: &[usize],
    view: ViewDtype,
    unindexed: bool,
    stats: StatsView,
    schema: Option<&PackingSchema>,
) -> Vec<Line<'static>> {
    let mut lines: Vec<Line> = Vec::new();

    lines.push(Line::from(Span::styled(
        "Tensor Details",
        Style::default().fg(to_ratatui(palette::ACCENT)),
    )));
    lines.push(Line::from(dim_span("─".repeat(14))));
    lines.push(Line::from(vec![
        dim_span("Name: "),
        Span::raw(tensor.name.clone()),
    ]));

    // Data type, with the active reinterpretation highlighted.
    let unpacked_label = schema.map(PackingSchema::label);
    let mut dtype_line = vec![dim_span("Data Type: ")];
    dtype_line.extend(detail_dtype_spans(
        &tensor.dtype,
        view,
        unpacked_label.as_deref(),
    ));
    lines.push(Line::from(dtype_line));

    // Shape and parameter count reflect the overrides.
    let logical = view.logical_shape_with(shape, &tensor.dtype, schema);
    let num_elements: usize = logical.iter().product();
    let mut shape_line = vec![dim_span("Shape: ")];
    shape_line.extend(detail_shape_spans(&tensor.shape, &logical));
    lines.push(Line::from(shape_line));
    lines.push(Line::from(vec![
        dim_span("Parameters: "),
        Span::raw(format!("{} ", format_parameters(num_elements))),
        dim_span(format!("({})", with_thousands(num_elements))),
    ]));

    // Codebook packing schema disclosure (only for tensors that carry one).
    if let Some(s) = schema {
        let widths = s
            .bit_widths()
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let mode = s
            .quant_mode()
            .map(|m| format!(" · {m}"))
            .unwrap_or_default();
        let uniform = if s.uniform_width().is_some() {
            "uniform"
        } else {
            "non-uniform"
        };
        lines.push(Line::from(vec![
            dim_span("Packing: "),
            Span::raw(format!("{} ", s.label())),
            dim_span(format!(
                "(bit widths [{widths}] · {} experts/word · {uniform}{mode})",
                s.len_p()
            )),
        ]));
    }

    // Size, with on-disk size + codec for formats that track compression.
    let mut size_line = vec![
        dim_span("Size: "),
        Span::raw(format_size(tensor.size_bytes)),
    ];
    match &tensor.storage {
        Storage::Compressed {
            codec,
            stored_bytes,
        } => {
            let ratio = tensor.size_bytes as f64 / (*stored_bytes).max(1) as f64;
            size_line.push(Span::raw(format!(
                " · on disk: {} ",
                format_size(*stored_bytes)
            )));
            size_line.push(dim_span(format!(
                "({COMPRESSED_MARK} {codec}, {ratio:.1}×)"
            )));
        }
        Storage::Raw => {
            size_line.push(Span::raw(format!(
                " · on disk: {} {UNCOMPRESSED_TAG}",
                format_size(tensor.size_bytes)
            )));
        }
        Storage::Unknown => {}
    }
    lines.push(Line::from(size_line));

    // Where the data lives within the file.
    match &tensor.layout {
        Layout::ByteRange { start, end } => {
            lines.push(Line::from(vec![
                dim_span("Data offsets: "),
                Span::raw(format!(
                    "{} – {}  (within file data)",
                    with_thousands(*start as usize),
                    with_thousands(*end as usize)
                )),
            ]));
        }
        Layout::Offset(offset) => {
            lines.push(Line::from(vec![
                dim_span("Data offset: "),
                Span::raw(format!(
                    "{}  (within tensor data)",
                    with_thousands(*offset as usize)
                )),
            ]));
        }
        Layout::Chunked { chunk, num_chunks } => {
            lines.push(Line::from(vec![
                dim_span("Chunks: "),
                Span::raw(format!(
                    "{} × {}",
                    format_shape(chunk),
                    with_thousands(*num_chunks)
                )),
            ]));
        }
        Layout::None => {}
    }

    lines.push(Line::from(vec![
        dim_span("File: "),
        Span::raw(tensor.source_path.clone()),
    ]));
    // Flag a tensor that's on disk but absent from the index.
    if unindexed {
        lines.push(Line::from(Span::styled(
            format!("{UNINDEXED_MARK} on disk but not listed in model.safetensors.index.json"),
            Style::default().fg(to_ratatui(palette::UNINDEXED)),
        )));
    }
    lines.push(Line::default());

    // Exact whole-tensor statistics: shown once computed, else a hint.
    let stats_line: Vec<Span> = match stats {
        StatsView::Ready(s) => {
            let integer = view.is_integer(&tensor.dtype);
            let mut spans = vec![
                dim_span("Statistics: "),
                Span::raw(format!(
                    "min {} · max {} · ",
                    fmt_value(s.min, integer),
                    fmt_value(s.max, integer)
                )),
            ];
            spans.extend(detail_stats_summary_spans(s));
            spans
        }
        StatsView::Computing {
            spinner,
            elapsed,
            progress,
        } => {
            let mut spans = vec![dim_span("Statistics: ")];
            spans.extend(detail_computing_spans(spinner, elapsed, progress));
            spans
        }
        StatsView::Pending => vec![
            dim_span("Statistics: press "),
            key_span("s"),
            dim_span(" to scan the full tensor"),
        ],
    };
    lines.push(Line::from(stats_line));
    lines.push(Line::default());

    lines
}

/// The detail screen's footer hint as wrapped [`Line`]s — the Ratatui port of the
/// raw footer in [`UI::draw_tensor_detail`], wrapped on the ` · `-free `key
/// label,` chips the raw line builds (here split into wrappable chunks). Mirrors
/// the tree's [`tree_hint_lines`] wrapping at `width`.
fn detail_footer_lines(overridable: bool, width: u16) -> Vec<Line<'static>> {
    // Each chunk is a run of spans that should not be split across lines; the
    // trailing text of each (the comma + space) keeps the on-screen wording.
    let mut chunks: Vec<Vec<Span<'static>>> = vec![vec![Span::raw("Press ")]];
    let mut push = |chunk: Vec<Span<'static>>| chunks.push(chunk);
    push(vec![key_span("m"), Span::raw(" for a heatmap, ")]);
    push(vec![key_span("v"), Span::raw(" for numeric values, ")]);
    push(vec![key_span("h"), Span::raw(" for a histogram, ")]);
    push(vec![key_span("b"), Span::raw(" to set its bin count, ")]);
    if overridable {
        push(vec![
            key_span("d"),
            Span::raw(" to reinterpret the dtype, "),
        ]);
        push(vec![key_span("r"), Span::raw(" to reshape, ")]);
    }
    push(vec![key_span("c"), Span::raw(" to copy, ")]);
    push(vec![key_span("y"), Span::raw(" to copy the command, ")]);
    push(vec![key_span("l"), Span::raw(" for the legend, ")]);
    push(vec![
        key_span("⌫"),
        Span::raw(" / "),
        key_span("\\"),
        Span::raw(" to step back / forward, any other key to return..."),
    ]);

    // Greedily pack chunks onto lines, breaking before a chunk that would overflow
    // (each chunk already carries its own separator, so no extra is inserted).
    let width = (width as usize).max(1);
    let chunk_w = |c: &[Span]| -> usize { c.iter().map(|s| s.content.chars().count()).sum() };
    let mut lines: Vec<Line> = Vec::new();
    let mut spans: Vec<Span> = Vec::new();
    let mut col = 0usize;
    for chunk in chunks {
        let w = chunk_w(&chunk);
        if !spans.is_empty() && col + w > width {
            lines.push(Line::from(std::mem::take(&mut spans)));
            col = 0;
        }
        spans.extend(chunk);
        col += w;
    }
    if !spans.is_empty() {
        lines.push(Line::from(spans));
    }
    lines
}

/// Render the value histogram into `rect` — the Ratatui port of
/// [`write_histogram_section`]: a heading (value count, any non-finite, the scan
/// indicator), then one bar per bin (label │ bar count (pct)), then a clipped-bin
/// note when they don't all fit. The whole section stays within `rect.height`.
/// Returns the number of rows actually drawn, so the caller can flow the footer
/// right below it (the raw renderer wrote these sequentially, so a small histogram
/// leaves the footer near the top rather than at the screen's bottom).
fn render_histogram(
    frame: &mut Frame,
    rect: Rect,
    hist: &Histogram,
    scanning: Option<ScanProgress>,
) -> usize {
    let term_w = rect.width as usize;
    let max_rows = rect.height as usize;
    if max_rows == 0 {
        return 0;
    }
    let mut lines: Vec<Line> = Vec::new();

    // Heading.
    let mut head = vec![
        dim_span("Histogram: "),
        Span::raw(format!("{} values", with_thousands(hist.total as usize))),
    ];
    if hist.nonfinite > 0 {
        head.push(dim_span(format!(
            "  ·  {} non-finite",
            with_thousands(hist.nonfinite as usize)
        )));
    }
    if let Some((spinner, elapsed, progress)) = scanning {
        let mut s = format!("   {spinner} scanning");
        if let Some(p) = progress {
            s.push_str(&format!(" {:.0}%", p * 100.0));
        }
        s.push_str(&format!(" ({:.1}s)", elapsed.as_secs_f64()));
        head.push(Span::styled(
            s,
            Style::default().fg(to_ratatui(palette::ACCENT)),
        ));
    } else if !hist.elapsed.is_zero() {
        head.push(dim_span(format!("  ({})", fmt_duration(hist.elapsed))));
    }
    lines.push(Line::from(head));
    let heading_rows = 1usize;

    let n = hist.counts.len();
    let labels: Vec<String> = (0..n)
        .map(|i| match hist.bins {
            HistBins::IntBins { start, step } => (start + i as i64 * step).to_string(),
            HistBins::Range { lo, hi } => fmt_hist_edge(lo + (hi - lo) * i as f64 / n as f64),
        })
        .collect();
    let label_w = labels.iter().map(|l| l.chars().count()).max().unwrap_or(1);
    let counts: Vec<String> = hist
        .counts
        .iter()
        .map(|c| with_thousands(*c as usize))
        .collect();
    let count_w = counts.iter().map(|s| s.chars().count()).max().unwrap_or(1);
    let max_count = hist.counts.iter().copied().max().unwrap_or(0).max(1);
    let total = hist.total.max(1);
    let pcts: Vec<String> = hist
        .counts
        .iter()
        .map(|&c| {
            let pct = c as f64 / total as f64 * 100.0;
            if c == 0 {
                "0.0%".to_string()
            } else if pct < 0.1 {
                format!("{pct:.1e}%")
            } else {
                format!("{pct:.1}%")
            }
        })
        .collect();
    let pct_w = pcts.iter().map(|s| s.chars().count()).max().unwrap_or(4);

    // The bar gets whatever width is left after `label │ … count (pct)`.
    let fixed = label_w + 3 + 1 + count_w + pct_w + 3;
    let bar_w = term_w.saturating_sub(fixed).clamp(1, 100);
    let bar_rows = max_rows.saturating_sub(heading_rows).max(1);
    let shown = if n <= bar_rows {
        n
    } else {
        bar_rows.saturating_sub(1).max(1)
    };

    let accent = Style::default().fg(to_ratatui(palette::ACCENT));
    let bold = Style::default().add_modifier(Modifier::BOLD);
    for i in 0..shown {
        let frac = hist.counts[i] as f64 / max_count as f64;
        lines.push(Line::from(vec![
            Span::raw(format!("{:>label_w$} ", labels[i])),
            dim_span("│"),
            Span::styled(bar(frac, bar_w), accent),
            Span::styled(format!(" {:>count_w$} ", counts[i]), bold),
            dim_span("("),
            Span::raw(pcts[i].clone()),
            dim_span(")"),
        ]));
    }
    if n > shown {
        lines.push(Line::from(dim_span(format!(
            "… {} more bins (enlarge the terminal)",
            n - shown
        ))));
    }

    let used = lines.len().min(max_rows);
    Paragraph::new(lines).render(rect, frame.buffer_mut());
    used
}

/// If `raw` is a JSON object or array, pretty-print it with syntax highlighting
/// (via `colored_json`, styled from [`json_styler`]) and return one ANSI-colored
/// string per line; otherwise `None`, so the caller shows the raw text. Bare
/// scalars (a lone string/number) aren't worth reformatting, so they fall
/// through to the raw path too.
fn highlight_json(raw: &str) -> Option<Vec<String>> {
    let value: serde_json::Value = serde_json::from_str(raw.trim()).ok()?;
    if !value.is_object() && !value.is_array() {
        return None;
    }
    // `colored_json` paints via yansi, whose default condition drops the ANSI
    // codes when stdout isn't a detected TTY (which would also make the result
    // non-deterministic). We render into our own buffer and own the terminal, so
    // force coloring on.
    yansi::enable();
    let pretty = colored_json::ColoredFormatter::with_styler(
        colored_json::PrettyFormatter::new(),
        json_styler(),
    )
    .to_colored_json(&value, colored_json::ColorMode::On)
    .ok()?;
    Some(pretty.split('\n').map(str::to_string).collect())
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

/// Write a data view's title block — the tensor name and its source file — each
/// kept to a single line (truncated tail-first, so the distinguishing end stays)
/// so both remain on screen above a grid of any size. `kind` is the view label
/// (`Values` / `Heatmap`).
fn write_data_view_title(out: &mut impl Write, kind: &str, tensor: &TensorInfo) -> Result<()> {
    let width = crate::plain::term_size().0 as usize;
    write!(out, "{kind}: ")?;
    write!(
        out,
        "{}",
        truncate_keep_end(&tensor.name, width.saturating_sub(kind.len() + 2))
    )?;
    line_end(&mut *out)?;
    paint(&mut *out, false, palette::DIM, "File: ")?;
    write!(
        out,
        "{}",
        truncate_keep_end(&tensor.source_path, width.saturating_sub(6))
    )?;
    line_end(&mut *out)?;
    Ok(())
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

/// A horizontal bar `width` cells wide filled to `frac` of `[0, 1]`. Uses the
/// lower three-quarters block `▆` (rather than a full `█`) so its top sits below
/// the cell ceiling, leaving a thin gap between stacked bars; any non-zero bar
/// shows at least one cell so tiny bins stay visible.
fn bar(frac: f64, width: usize) -> String {
    let frac = frac.clamp(0.0, 1.0);
    let mut cells = (frac * width as f64).round() as usize;
    if frac > 0.0 {
        cells = cells.max(1);
    }
    let cells = cells.min(width);
    if cells == 0 {
        // An empty (zero-count) bin still occupies the one-cell baseline so its
        // count lines up with the smallest non-zero bars rather than shifting
        // a column to the left.
        " ".to_string()
    } else {
        "▆".repeat(cells)
    }
}

/// Render the value histogram as horizontal bars — one per bin, with its label,
/// absolute count, and percentage of the finite values — for embedding in the
/// detail screen below the statistics. The whole section (heading + bars + an
/// "N more" note when clipped) fits within `max_rows`, so it never pushes the
/// footer off a short screen. `scanning` (spinner, elapsed, fraction) marks a
/// still-forming scan so the bars animate as they fill in.
fn write_histogram_section(
    out: &mut impl Write,
    hist: &Histogram,
    scanning: Option<ScanProgress>,
    term_w: usize,
    max_rows: usize,
) -> Result<()> {
    // Heading: how many values, any non-finite, and the scan indicator. Built
    // into a buffer so its own wrapped height is known (the scan line can be
    // long), leaving the rest of the budget for bars.
    let mut head: Vec<u8> = Vec::new();
    paint(&mut head, false, palette::DIM, "Histogram: ")?;
    write!(head, "{} values", with_thousands(hist.total as usize))?;
    if hist.nonfinite > 0 {
        paint(
            &mut head,
            false,
            palette::DIM,
            &format!(
                "  ·  {} non-finite",
                with_thousands(hist.nonfinite as usize)
            ),
        )?;
    }
    if let Some((spinner, elapsed, progress)) = scanning {
        queue!(head, SetForegroundColor(palette::ACCENT))?;
        write!(head, "   {spinner} scanning")?;
        if let Some(p) = progress {
            write!(head, " {:.0}%", p * 100.0)?;
        }
        write!(head, " ({:.1}s)", elapsed.as_secs_f64())?;
        queue!(head, ResetColor)?;
    } else if !hist.elapsed.is_zero() {
        // Finished: keep the scan time on the heading, like the statistics line.
        paint(
            &mut head,
            false,
            palette::DIM,
            &format!("  ({})", fmt_duration(hist.elapsed)),
        )?;
    }
    line_end(&mut head)?;
    let heading_rows = count_physical_lines(&head, term_w);
    out.write_all(&head)?;

    let n = hist.counts.len();
    // Bin labels: the integer value, or the bin's lower edge for range bins.
    let labels: Vec<String> = (0..n)
        .map(|i| match hist.bins {
            HistBins::IntBins { start, step } => (start + i as i64 * step).to_string(),
            HistBins::Range { lo, hi } => fmt_hist_edge(lo + (hi - lo) * i as f64 / n as f64),
        })
        .collect();
    let label_w = labels.iter().map(|l| l.chars().count()).max().unwrap_or(1);
    let counts: Vec<String> = hist
        .counts
        .iter()
        .map(|c| with_thousands(*c as usize))
        .collect();
    let count_w = counts.iter().map(|s| s.chars().count()).max().unwrap_or(1);
    let max_count = hist.counts.iter().copied().max().unwrap_or(0).max(1);
    let total = hist.total.max(1);
    // Percentages: a non-empty bin with a tiny share would round to a misleading
    // `0.0%`, so show its magnitude in scientific notation instead (matching the
    // stats line's zero-fraction). Empty bins stay a plain `0.0%`.
    let pcts: Vec<String> = hist
        .counts
        .iter()
        .map(|&c| {
            let pct = c as f64 / total as f64 * 100.0;
            if c == 0 {
                "0.0%".to_string()
            } else if pct < 0.1 {
                format!("{pct:.1e}%")
            } else {
                format!("{pct:.1}%")
            }
        })
        .collect();
    let pct_w = pcts.iter().map(|s| s.chars().count()).max().unwrap_or(4);

    // The bar gets whatever width is left after `label │ … count (pct)`.
    let fixed = label_w + 3 + 1 + count_w + pct_w + 3;
    let bar_w = term_w.saturating_sub(fixed).clamp(1, 100);
    // `max_rows` bounds the whole section; the heading took `heading_rows`, so
    // the rest is for bars — and when clipping, one more is left for the note.
    let bar_rows = max_rows.saturating_sub(heading_rows).max(1);
    let shown = if n <= bar_rows {
        n
    } else {
        bar_rows.saturating_sub(1).max(1)
    };

    for i in 0..shown {
        let frac = hist.counts[i] as f64 / max_count as f64;
        // The bin value (left) and the count (right) are the data, so both read
        // at full strength; only the `│` separator and the percentage's
        // parentheses are dimmed as chrome.
        write!(out, "{:>label_w$} ", labels[i])?;
        paint(out, false, palette::DIM, "│")?;
        queue!(out, SetForegroundColor(palette::ACCENT))?;
        write!(out, "{}", bar(frac, bar_w))?;
        queue!(out, ResetColor)?;
        queue!(out, SetAttribute(Attribute::Bold))?;
        write!(out, " {:>count_w$} ", counts[i])?;
        queue!(out, SetAttribute(Attribute::Reset))?;
        paint(out, false, palette::DIM, "(")?;
        write!(out, "{}", pcts[i])?;
        paint(out, false, palette::DIM, ")")?;
        line_end(out)?;
    }
    if n > shown {
        paint(
            out,
            false,
            palette::DIM,
            &format!("… {} more bins (enlarge the terminal)", n - shown),
        )?;
        line_end(out)?;
    }
    Ok(())
}

/// Visible (printable) character count of a rendered byte buffer — ANSI escape
/// sequences and carriage returns excluded. Used to measure how wide a styled
/// line is so its wrapped height can be computed.
fn visible_len(buf: &[u8]) -> usize {
    let text = String::from_utf8_lossy(buf);
    let mut chars = text.chars().peekable();
    let mut len = 0;
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            // Skip a CSI sequence: `ESC [ … <final letter>`.
            if chars.peek() == Some(&'[') {
                chars.next();
                while let Some(&d) = chars.peek() {
                    chars.next();
                    if d.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
        } else if c != '\r' && c != '\n' {
            len += 1;
        }
    }
    len
}

/// Number of terminal rows a rendered buffer occupies at the given width:
/// every `\n`-terminated line wraps to `ceil(visible / width)` rows (at least
/// one), so the height accounts for both explicit line breaks and autowrap.
fn count_physical_lines(buf: &[u8], width: usize) -> usize {
    let text = String::from_utf8_lossy(buf);
    let mut lines: Vec<&str> = text.split('\n').collect();
    // A trailing newline leaves an empty final segment that isn't its own row.
    if lines.last() == Some(&"") {
        lines.pop();
    }
    lines
        .iter()
        .map(|line| visible_len(line.as_bytes()).div_ceil(width.max(1)).max(1))
        .sum()
}

/// Compact label for a range-histogram bin's lower edge.
fn fmt_hist_edge(x: f64) -> String {
    if x == 0.0 {
        "0".to_string()
    } else if x.abs() >= 1e5 || x.abs() < 1e-3 {
        format!("{x:.2e}")
    } else {
        format!("{x:.4}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drop CSI escape sequences (`\x1b[…<letter>`) so a colored string can be
    /// compared against its plain text.
    fn strip_ansi_codes(s: &str) -> String {
        let mut out = String::new();
        let mut chars = s.chars();
        while let Some(c) = chars.next() {
            if c == '\u{1b}' {
                for c in chars.by_ref() {
                    if c.is_ascii_alphabetic() {
                        break;
                    }
                }
            } else {
                out.push(c);
            }
        }
        out
    }

    #[test]
    fn highlight_json_colors_objects_and_arrays_only() {
        // Non-JSON text and bare scalars fall through to the raw path.
        assert!(highlight_json("just some text").is_none());
        assert!(highlight_json("\"a lone string\"").is_none());
        assert!(highlight_json("42").is_none());

        let raw = r#"{"b":[true,null,"x"],"a":1}"#;
        let lines = highlight_json(raw).expect("an object is highlighted");
        let joined = lines.join("\n");
        // Styled from the app palette: keys in the ACCENT color, numbers in the
        // DTYPE color (256-color SGR `38;5;<n>`), not colored_json's defaults.
        assert!(
            joined.contains("38;5;81"),
            "expected keys in the ACCENT color (81)"
        );
        assert!(
            joined.contains("38;5;215"),
            "expected numbers in the DTYPE color (215)"
        );
        // Stripping the color recovers exactly serde_json's pretty layout, so the
        // highlighter only adds color and never alters the text itself.
        let value: serde_json::Value = serde_json::from_str(raw).unwrap();
        assert_eq!(
            strip_ansi_codes(&joined),
            serde_json::to_string_pretty(&value).unwrap()
        );
    }

    #[test]
    fn num_base_parses_aliases() {
        assert_eq!(parse_num_base("dec"), Ok(NumBase::Decimal));
        assert_eq!(parse_num_base("DECIMAL"), Ok(NumBase::Decimal));
        assert_eq!(parse_num_base("hex"), Ok(NumBase::Hex));
        assert_eq!(parse_num_base("16"), Ok(NumBase::Hex));
        assert_eq!(parse_num_base(" Oct "), Ok(NumBase::Octal));
        assert_eq!(parse_num_base("bin"), Ok(NumBase::Binary));
        assert!(parse_num_base("base64").is_err());
    }

    #[test]
    fn num_base_cycles_and_round_trips_its_label() {
        // dec → hex → oct → bin → dec
        assert_eq!(NumBase::Decimal.next(), NumBase::Hex);
        assert_eq!(NumBase::Hex.next(), NumBase::Octal);
        assert_eq!(NumBase::Octal.next(), NumBase::Binary);
        assert_eq!(NumBase::Binary.next(), NumBase::Decimal);
        for b in [
            NumBase::Decimal,
            NumBase::Hex,
            NumBase::Octal,
            NumBase::Binary,
        ] {
            assert_eq!(parse_num_base(b.label()), Ok(b));
        }
    }

    #[test]
    fn num_base_digit_widths_match_bit_count() {
        // 32-bit element (e.g. F32/I32): 8 hex, 11 octal, 32 binary digits.
        assert_eq!(NumBase::Hex.digits(32), 8);
        assert_eq!(NumBase::Octal.digits(32), 11);
        assert_eq!(NumBase::Binary.digits(32), 32);
        // 8-bit and 4-bit elements.
        assert_eq!(NumBase::Hex.digits(8), 2);
        assert_eq!(NumBase::Hex.digits(4), 1);
        assert_eq!(NumBase::Octal.digits(8), 3);
    }
}
