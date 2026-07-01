use std::collections::{HashMap, HashSet};
use std::time::Duration;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Clear, LineGauge, Padding, Paragraph, Widget};

use crate::health::HealthReport;
use crate::sample::{HistBins, Histogram, PackingSchema, Sample, SampleMode, Stats, ViewDtype};
use crate::tree::{Layout, MetadataInfo, Storage, TensorInfo, TreeNode, metadata_short};
use crate::utils::{format_parameters, format_shape, format_size};

/// A clickable footer key-hint chip: where it sits within a hint block (line
/// index + column + width) and the key it stands for. The `render_*` functions
/// translate these to absolute screen [`Rect`]s and pair them with the key, so a
/// click can be turned into the equivalent keypress.
pub struct ChipHit {
    pub line: u16,
    pub col: u16,
    pub width: u16,
    pub key: KeyEvent,
}

/// A plain (no-modifier) key event — what clicking a single-letter hint stands for.
fn hint_key(c: char) -> KeyEvent {
    KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)
}

/// A piece of a footer chip's key text: either a clickable glyph paired with the
/// key it synthesizes, or a non-clickable separator (`/`, `Shift+`). The footer
/// builders emit one [`ChipHit`] per [`Seg::Key`] at its own sub-column, so each
/// half of a dual chip (`E/C`, `↑/↓`, `⌫/\`, …) is independently clickable.
enum Seg {
    Key(&'static str, KeyEvent),
    Sep(&'static str),
}

impl Seg {
    fn text(&self) -> &'static str {
        match self {
            Seg::Key(t, _) | Seg::Sep(t) => t,
        }
    }
}

/// Draw a `[×]` close control in the top-right corner and return its clickable
/// region paired with the key a click should synthesize (`q` to quit the tree,
/// `⌫` to step back from a sub-screen). No-op (empty region list) if too narrow.
fn close_button(frame: &mut Frame, key: KeyEvent) -> Vec<(Rect, KeyEvent)> {
    let area = frame.area();
    if area.width < 3 {
        return Vec::new();
    }
    let rect = Rect {
        x: area.width - 3,
        y: 0,
        width: 3,
        height: 1,
    };
    frame
        .buffer_mut()
        .set_string(rect.x, rect.y, "[×]", Style::default().fg(palette::ACCENT));
    vec![(rect, key)]
}

/// Translate a data view's footer [`ChipHit`]s (lines relative to `footer_top`)
/// into absolute screen regions and append the top-right `[×]` (→ step back).
/// Shared by the heatmap and numeric-grid renderers, which lay out identically.
fn data_view_regions(
    frame: &mut Frame,
    chips: &[ChipHit],
    footer_top: u16,
) -> Vec<(Rect, KeyEvent)> {
    let mut regions: Vec<(Rect, KeyEvent)> = chips
        .iter()
        .map(|c| {
            (
                Rect {
                    x: c.col,
                    y: footer_top + c.line,
                    width: c.width,
                    height: 1,
                },
                c.key,
            )
        })
        .collect();
    regions.extend(close_button(
        frame,
        KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE),
    ));
    regions
}

/// True when `(col, row)` falls inside a clickable region.
pub fn region_hit(regions: &[(Rect, KeyEvent)], col: u16, row: u16) -> Option<KeyEvent> {
    regions
        .iter()
        .find(|(r, _)| col >= r.x && col < r.x + r.width && row >= r.y && row < r.y + r.height)
        .map(|(_, k)| *k)
}

/// A still-forming scan's progress indicator: a spinner glyph, the elapsed time,
/// and an optional completed fraction (`None` when the total isn't known).
pub type ScanProgress = (char, std::time::Duration, Option<f64>);

/// The app's colour palette — the single source of truth for how each kind of
/// thing is styled, so the same role looks the same on every screen. Change a
/// colour here and it updates everywhere it's used.
mod palette {
    use ratatui::style::Color;

    /// Interactive keys in hint lines (rendered bold).
    pub const KEY: Color = Color::Indexed(14);
    /// Secondary / de-emphasised hint text (ranges, "to cancel", …).
    pub const DIM: Color = Color::Indexed(8);
    /// Selected tree row (foreground on background).
    pub const SELECT_FG: Color = Color::Indexed(0);
    pub const SELECT_BG: Color = Color::Indexed(15);
    /// The slice-jump input box (foreground on background).
    pub const INPUT_FG: Color = Color::Indexed(15);
    pub const INPUT_BG: Color = Color::Indexed(4);
    /// Something missing / wrong / out of range.
    pub const ERROR: Color = Color::Indexed(9);
    /// Something present but unexpected (a softer alert than [`ERROR`]).
    pub const WARN: Color = Color::Indexed(11);
    /// The bottom status bar (foreground on background).
    pub const STATUS_FG: Color = Color::Indexed(15);
    pub const STATUS_BG: Color = Color::Indexed(8);
    /// A success accent used as a *foreground* (e.g. the "✓ copied" confirmation).
    pub const SUCCESS: Color = Color::Indexed(10);
    /// Marks a tensor present on disk but missing from the index — a vivid red
    /// that stands out clearly against the tree's default and dimmed text.
    pub const UNINDEXED: Color = Color::Indexed(196);
    /// Group names and expand arrows in the tree — the primary accent (a bright
    /// sky-cyan), so the structure stands out from the leaf tensors.
    pub const ACCENT: Color = Color::Indexed(81);
    /// A tensor's data type (warm amber, so the type pops).
    pub const DTYPE: Color = Color::Indexed(215);
    /// Metadata entries (the `†` marker and the entry name) — a muted slate
    /// violet, distinct from the cyan groups and amber dtypes but quiet enough
    /// that metadata reads as a side note rather than competing with tensors.
    pub const META: Color = Color::Indexed(103);
    /// Zebra striping for the numeric grid — two subtle dark backgrounds (one
    /// "dark", one "less dark") that alternate to guide the eye along the rows
    /// or columns, like a dim highlighter.
    pub const STRIPE_DARK: Color = Color::Indexed(234);
    pub const STRIPE_LITE: Color = Color::Indexed(237);
    /// Background for floating pop-ups (legend, the `y` command panel, message
    /// screens) — a neutral dark grey a few shades above black, in the same
    /// family as the zebra greys above, so an overlay reads as a raised surface
    /// over the main screen while staying within the dark theme. Light/accent
    /// foregrounds keep their contrast; dim text stays legible.
    pub const PANEL_BG: Color = Color::Indexed(236);

    /// Backdrop behind a full-frame message screen ([`Backdrop::Fill`]): one shade
    /// darker than [`PANEL_BG`], so the box reads as a raised card over an even,
    /// dark field. (Floating pop-ups like the legend keep the live screen behind
    /// them and don't use this.)
    pub const SCRIM: Color = Color::Indexed(234);
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
/// the last layer of [`UI::render_detail`] so the screen behind it keeps
/// redrawing (a running scan's progress animates) while it's up. Dismissed by
/// any key. Composited via [`UI::render_legend_band`] / [`UI::render_command_band`].
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
        let header = Self::tree_header_rows(width, search_mode, can_repack);
        (height as usize)
            .saturating_sub(header + TREE_FOOTER_HEIGHT)
            .max(1)
    }

    /// The first terminal row of the tree body — the header height (title +
    /// hint/search line(s) + rule). Used for mouse hit-testing: a click at row
    /// `r >= tree_header_rows()` and above the 2-line status bar lands on tree
    /// row `scroll_offset + (r - tree_header_rows())`.
    pub fn tree_header_rows(width: u16, search_mode: bool, can_repack: bool) -> usize {
        let hint_rows = if search_mode {
            1
        } else {
            tree_hint_lines(can_repack, width).0.len()
        };
        1 + hint_rows + 1 // title + hint(s) + rule
    }

    /// Ratatui render of the tree browser: header (title, hint or search line,
    /// rule), the visible tree rows from `config.scroll_offset`, and the bottom
    /// two-line status bar, driven by the shared `DrawConfig`.
    pub fn render_tree(frame: &mut Frame, config: &DrawConfig) -> Vec<(Rect, KeyEvent)> {
        let area = frame.area();
        let (width, height) = (area.width, area.height);
        if height < (TREE_FOOTER_HEIGHT as u16 + 1) {
            return Vec::new();
        }
        // Clickable footer chips (built below) become absolute screen rects here.
        let mut chips: Vec<ChipHit> = Vec::new();

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
                Style::default().fg(palette::ERROR),
            ));
            title.push(Span::styled(
                "h",
                Style::default()
                    .fg(palette::KEY)
                    .add_modifier(Modifier::BOLD),
            ));
        }
        lines.push(Line::from(title));

        // Hint line(s), or the search bar when searching.
        if config.search_mode {
            lines.push(tree_search_line(config));
        } else {
            let (hint_lines, hint_chips) = tree_hint_lines(config.can_repack, width);
            lines.extend(hint_lines);
            chips = hint_chips;
        }

        // Separator rule.
        lines.push(Line::from(Span::styled(
            "─".repeat(width as usize),
            Style::default().fg(palette::DIM),
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
                        .fg(palette::KEY)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(" to exit search"),
            ])
        } else if !config.status_bar.is_empty() {
            let text = truncate_keep_end(config.status_bar, max_text);
            Line::from(Span::styled(
                format!(" {} {text} ", config.status_icon),
                Style::default()
                    .bg(palette::STATUS_BG)
                    .fg(palette::STATUS_FG),
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
                    .fg(palette::SUCCESS)
                    .add_modifier(Modifier::BOLD),
            ))
        } else if !config.status_secondary.is_empty() {
            let text = truncate_keep_end(config.status_secondary, max_text);
            Line::from(Span::styled(
                format!("   {text}"),
                Style::default().fg(palette::DIM),
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

        // Clickable regions: each footer chip (the hint block starts at row 1,
        // below the title) plus the top-right `[×]` (→ quit the tree).
        let mut regions: Vec<(Rect, KeyEvent)> = chips
            .iter()
            .map(|c| {
                (
                    Rect {
                        x: c.col,
                        y: 1 + c.line,
                        width: c.width,
                        height: 1,
                    },
                    c.key,
                )
            })
            .collect();
        regions.extend(close_button(frame, hint_key('q')));
        regions
    }

    /// Render the tensor detail screen. `view` is the active dtype reinterpretation
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
    ) -> Vec<(Rect, KeyEvent)> {
        let area = frame.area();
        let (width, height) = (area.width, area.height);

        let (header, stats_gauge_row) =
            detail_field_lines(tensor, shape, view, unindexed, stats, schema);
        let (footer, chips) = detail_footer_lines(overridable, width);
        let header_len = header.len();
        let footer_len = footer.len();
        // The screen row the footer block begins on (filled in per layout branch
        // below), so the relative chip lines can be made absolute for hit-testing.
        let footer_top;

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
            footer_top = footer_y;
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
            footer_top = header_len as u16;
            let mut lines = header;
            lines.extend(footer);
            Paragraph::new(lines).render(area, frame.buffer_mut());
        }

        // The header rows sit at `y = index` in both layouts, so overlay the stats
        // progress bar (native LineGauge) on its reserved row.
        if let (Some(row), Some((ratio, label))) = (stats_gauge_row, computing_gauge(stats)) {
            render_line_gauge(
                frame,
                Rect {
                    x: 0,
                    y: row as u16,
                    width,
                    height: 1,
                },
                label,
                ratio,
                Some(30),
            );
        }

        // Clickable regions: each footer chip (made absolute via the footer's
        // start row) plus the top-right `[×]` (→ step back, like `⌫`).
        let mut regions: Vec<(Rect, KeyEvent)> = chips
            .iter()
            .map(|c| {
                (
                    Rect {
                        x: c.col,
                        y: footer_top + c.line,
                        width: c.width,
                        height: 1,
                    },
                    c.key,
                )
            })
            .collect();
        regions.extend(close_button(
            frame,
            KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE),
        ));

        // A pop-up overlay composites last, over the live frame, so the detail
        // (including a running scan's progress) keeps animating behind it.
        match overlay {
            Some(Overlay::Legend(l)) => Self::render_legend_band(frame, *l),
            Some(Overlay::Command(c)) => Self::render_command_band(frame, c),
            None => {}
        }
        regions
    }

    /// Composite the context-sensitive glyph legend over the live frame as a
    /// centred, rounded [`Block`] pop-up (its context is the box title), drawn last
    /// so the screen behind keeps animating. Shared by every screen's `l` overlay
    /// and by `--plain --legend`.
    pub fn render_legend_band(frame: &mut Frame, legend: Legend) {
        render_popup_box(
            frame,
            legend_title(legend),
            legend_band_lines(legend),
            Backdrop::Float,
        );
    }

    /// Composite the copied-CLI-command pop-up over the live frame — a full-width
    /// [`render_titled_bar`] (label + copied confirmation ride the top border) with
    /// the wrapped command flush at column 0 so it stays cleanly selectable, then a
    /// dismiss hint.
    pub fn render_command_band(frame: &mut Frame, command: &str) {
        let term_w = frame.area().width as usize;
        let title = Line::from(vec![
            Span::styled(
                " CLI command ",
                Style::default()
                    .fg(palette::KEY)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "✓ copied to the clipboard ",
                Style::default().fg(palette::SUCCESS),
            ),
        ]);
        // The command, soft-wrapped at full width onto its own line(s), flush at
        // column 0 so it can still be selected cleanly by hand when the OSC-52
        // copy doesn't reach the terminal.
        let chars: Vec<char> = command.chars().collect();
        let cmd_rows = chars.len().div_ceil(term_w.max(1)).max(1);
        let mut content: Vec<Line> = (0..cmd_rows)
            .map(|r| {
                let seg: String = chars.iter().skip(r * term_w).take(term_w).collect();
                Line::from(Span::raw(seg))
            })
            .collect();
        content.push(Line::from(dim_span("click or press any key to dismiss")));
        render_titled_bar(frame, title, content);
    }

    /// The Ratatui port of [`Self::draw_loading`]: the tree browser's title + rule
    /// header, a spinner on the row where the tree's first node will land, and the
    /// cancel hint pinned to the bottom — so the chrome is up immediately and the
    /// tree fills into the same frame once the read finishes.
    pub fn render_loading(
        frame: &mut Frame,
        file: &str,
        total_files: usize,
        spinner: char,
        elapsed: std::time::Duration,
    ) {
        let area = frame.area();
        let width = area.width as usize;
        let height = area.height;

        // Title (row 0), with the same "+N more" note for a multi-file load.
        let mut title = vec![Span::raw(format!("Checkpoint Explorer - {file}"))];
        if total_files > 1 {
            title.push(dim_span(format!("  (+{} more)", total_files - 1)));
        }
        // Full-width rule (row 1).
        let mut lines: Vec<Line> = vec![
            Line::from(title),
            Line::from(dim_span("─".repeat(width))),
            Line::default(),
        ];
        // The spinner lands on the row where the tree's first node will (row 3,
        // clamped). Rows above it are blank spacers added above.
        let spinner_row = 3u16.min(height.saturating_sub(2));
        for _ in lines.len() as u16..spinner_row {
            lines.push(Line::default());
        }
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(
                format!("{spinner} reading checkpoint structure"),
                Style::default().fg(palette::ACCENT),
            ),
            dim_span(format!("  ({:.1}s)", elapsed.as_secs_f64())),
        ]));
        Paragraph::new(lines).render(area, frame.buffer_mut());

        // Footer hint pinned to the bottom row.
        Paragraph::new(Line::from(vec![
            dim_span("Press "),
            key_span("q"),
            dim_span(" to cancel"),
        ]))
        .render(
            Rect {
                x: 0,
                y: height.saturating_sub(1),
                width: area.width,
                height: 1,
            },
            frame.buffer_mut(),
        );
    }

    /// The Ratatui port of [`Self::draw_metadata_detail`]: the Key/Type/Value
    /// header, then the value — pretty, syntax-highlighted JSON converted from its
    /// ANSI form via `ansi-to-tui` (so the same `colored_json` palette shows
    /// through), or the raw text lines for a non-JSON value — with the same
    /// line-budget elision and footer.
    pub fn render_metadata_detail(frame: &mut Frame, metadata: &MetadataInfo) {
        use ansi_to_tui::IntoText;
        let area = frame.area();
        let rows = area.height as usize;

        let mut lines: Vec<Line> = vec![
            Line::from(Span::styled(
                "Metadata Details",
                Style::default().fg(palette::ACCENT),
            )),
            Line::from(dim_span("================")),
            Line::from(vec![dim_span("Key: "), Span::raw(metadata.name.clone())]),
            Line::from(vec![
                dim_span("Type: "),
                Span::raw(metadata.value_type.clone()),
            ]),
            Line::from(dim_span("Value:")),
        ];

        // A JSON object/array is highlighted (via `colored_json`'s ANSI, parsed
        // back into styled spans); everything else falls back to plain text lines.
        let value_lines: Vec<Line> = match highlight_json(&metadata.value) {
            Some(colored) => colored
                .join("\n")
                .into_text()
                .map(|t| t.lines)
                .unwrap_or_else(|_| {
                    metadata
                        .value
                        .lines()
                        .map(|l| Line::from(l.to_string()))
                        .collect()
                }),
            None => metadata
                .value
                .lines()
                .map(|l| Line::from(l.to_string()))
                .collect(),
        };

        // Show as many value lines as fit (header above + a short footer below),
        // noting how many were elided rather than cutting silently.
        let budget = rows.saturating_sub(8).max(1);
        let shown = value_lines.len().min(budget);
        for line in value_lines.iter().take(shown) {
            let mut indented = vec![Span::raw("  ")];
            indented.extend(line.spans.iter().cloned());
            lines.push(Line::from(indented));
        }
        if value_lines.len() > shown {
            lines.push(Line::from(dim_span(format!(
                "  … ({} more lines)",
                value_lines.len() - shown
            ))));
        }

        lines.push(Line::default());
        lines.push(Line::from(Span::raw("Click or press any key to return...")));
        Paragraph::new(lines).render(area, frame.buffer_mut());
    }

    /// Render a sampled tensor as a heatmap — the Ratatui port of
    /// [`UI::draw_heatmap`]. Each text row shows two data rows via the upper-half
    /// block `▀`: the cell's foreground is the upper data row's heat color, its
    /// background the lower row's. A trailing odd row keeps the default background
    /// for its empty lower half. The title / dtype-shape / slice / range chrome
    /// and the footer match the numeric grid; the layout is top-aligned so a small
    /// sample leaves the lower screen blank, exactly like the raw renderer (which
    /// wrote sequentially and cleared below).
    pub fn render_heatmap(
        frame: &mut Frame,
        tensor: &TensorInfo,
        sample: &Sample,
        stats: StatsView,
    ) -> Vec<(Rect, KeyEvent)> {
        let area = frame.area();
        let width = area.width as usize;
        let mut lines: Vec<Line> = data_view_title_lines("Heatmap", tensor, width);

        let integer = sample.view.is_integer(&tensor.dtype);
        // The exact whole-tensor range once stats are ready; else the sampled
        // range, flagged as such.
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
        let what = match sample.mode {
            SampleMode::Edges { .. } => "edges",
            SampleMode::Window { .. } => "window",
            SampleMode::Grid => "sampled",
        };
        let mut dtype_line =
            view_dtype_spans(&tensor.dtype, sample.view, sample.schema_label.as_deref());
        dtype_line.push(Span::raw(" "));
        dtype_line.extend(view_shape_spans(&tensor.shape, &sample.display_shape));
        dtype_line.push(Span::raw(format!(
            " → {what} {}×{}, value range [{lo}, {hi}]{range_note}",
            sample.rows.len(),
            sample.cols.len(),
        )));
        lines.push(Line::from(dtype_line));

        // A computing-with-fraction stats row is a native progress bar: reserve a
        // blank line and render a `LineGauge` over it after the paragraph.
        let stats_gauge_row = if computing_gauge(stats).is_some() {
            let row = lines.len();
            lines.push(Line::default());
            Some(row)
        } else {
            if let Some(stats_line) = data_stats_view_line(stats) {
                lines.push(stats_line);
            }
            None
        };
        if sample.slices > 1 {
            lines.push(slice_header_line(sample));
        }
        lines.push(Line::default());

        let range = rmax - rmin;
        let norm = |v: f64| {
            if range > 0.0 { (v - rmin) / range } else { 0.5 }
        };
        // Two data rows per text line: foreground = the upper row's value,
        // background = the lower row's; a trailing odd row keeps the default bg.
        let mut r = 0;
        while r < sample.values.len() {
            let top = &sample.values[r];
            let bottom = sample.values.get(r + 1);
            let mut spans: Vec<Span> = Vec::with_capacity(top.len());
            for (c, &tv) in top.iter().enumerate() {
                let mut style = Style::default().fg(heat_color(norm(tv)));
                if let Some(below) = bottom {
                    style = style.bg(heat_color(norm(below[c])));
                }
                spans.push(Span::styled("▀", style));
            }
            lines.push(Line::from(spans));
            r += 2;
        }

        lines.push(Line::default());
        let mut legend = vec![Span::raw(format!("{lo} low "))];
        for i in 0..24 {
            legend.push(Span::styled(
                "█",
                Style::default().fg(heat_color(i as f64 / 23.0)),
            ));
        }
        legend.push(Span::raw(format!(" high {hi}")));
        lines.push(Line::from(legend));

        lines.push(Line::default());
        let footer_top = lines.len() as u16;
        let (footer, chips) = data_view_footer_wrapped_lines(
            sample.mode,
            sample.slices,
            true,
            true,
            StripeMode::Off,
            NumBase::Decimal,
            width,
        );
        lines.extend(footer);

        Paragraph::new(lines).render(area, frame.buffer_mut());
        if let (Some(row), Some((ratio, label))) = (stats_gauge_row, computing_gauge(stats)) {
            render_line_gauge(
                frame,
                Rect {
                    x: 0,
                    y: row as u16,
                    width: area.width,
                    height: 1,
                },
                label,
                ratio,
                Some(30),
            );
        }
        data_view_regions(frame, &chips, footer_top)
    }

    /// Render a sampled tensor as a grid of numeric values with row/column
    /// indices — the Ratatui port of [`UI::draw_values`]. Same title / dtype-shape
    /// / slice / footer chrome as the heatmap; each value cell is a styled span
    /// (right-aligned, optional zebra-stripe background, dimmed gap markers) built
    /// the same way [`write_grid_cell`] writes one. Top-aligned, like the raw
    /// renderer.
    pub fn render_values(
        frame: &mut Frame,
        tensor: &TensorInfo,
        sample: &Sample,
        stats: StatsView,
        stripe: StripeMode,
        base: NumBase,
    ) -> Vec<(Rect, KeyEvent)> {
        let area = frame.area();
        let width = area.width as usize;
        // Cell width adapts to the data (same call the sampler uses, so the column
        // count agrees).
        let cw = base.cell_width(sample.view, &tensor.dtype, stats.value_range());

        let mut lines: Vec<Line> = data_view_title_lines("Values", tensor, width);

        let mut dtype_line =
            view_dtype_spans(&tensor.dtype, sample.view, sample.schema_label.as_deref());
        dtype_line.push(Span::raw(" "));
        dtype_line.extend(view_shape_spans(&tensor.shape, &sample.display_shape));
        let edges = matches!(sample.mode, SampleMode::Edges { .. });
        dtype_line.push(Span::raw(match sample.mode {
            SampleMode::Edges { .. } => format!(
                " → edges: {} of {} rows × {} of {} cols (indices shown)",
                edge_desc(&sample.rows, sample.total_rows),
                sample.total_rows,
                edge_desc(&sample.cols, sample.total_cols),
                sample.total_cols
            ),
            SampleMode::Window { .. } => format!(
                " → window: rows {} of {} × cols {} of {} (contiguous)",
                span_desc(&sample.rows),
                sample.total_rows,
                span_desc(&sample.cols),
                sample.total_cols
            ),
            SampleMode::Grid => format!(
                " → sampled {} of {} rows × {} of {} cols (indices shown)",
                sample.rows.len(),
                sample.total_rows,
                sample.cols.len(),
                sample.total_cols
            ),
        }));
        lines.push(Line::from(dtype_line));

        // A computing-with-fraction stats row is a native progress bar (see
        // `render_heatmap`).
        let stats_gauge_row = if computing_gauge(stats).is_some() {
            let row = lines.len();
            lines.push(Line::default());
            Some(row)
        } else {
            if let Some(stats_line) = data_stats_view_line(stats) {
                lines.push(stats_line);
            }
            None
        };
        if sample.slices > 1 {
            lines.push(slice_header_line(sample));
        }
        lines.push(Line::default());

        // The index after which rows/cols jump (the padding boundary in edges
        // mode), so the dotted separator can be drawn there.
        let gap = |idx: &[usize]| -> Option<usize> {
            edges
                .then(|| idx.windows(2).position(|w| w[1] != w[0] + 1))
                .flatten()
        };
        let row_gap = gap(&sample.rows);
        let col_gap = gap(&sample.cols);
        let lw = 6usize;
        let dim = Style::default().fg(palette::DIM);

        // Column-index header (with a "⋯" gap column). Wide cells fit the index
        // in a single row; narrow cells stagger labels across two rows.
        let idx_w = sample
            .cols
            .iter()
            .map(|&c| c.to_string().len())
            .max()
            .unwrap_or(1);
        if idx_w >= cw {
            let step = (idx_w + 1).div_ceil(2 * cw).max(1);
            let right_edge = |j: usize| -> usize {
                let gap_cells = matches!(col_gap, Some(g) if j > g) as usize;
                lw + (j + 1 + gap_cells) * cw
            };
            let hwidth = right_edge(sample.cols.len().saturating_sub(1)).max(lw);
            let mut top = vec![' '; hwidth];
            let mut bot = vec![' '; hwidth];
            let mut rank = 0usize;
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
            if let Some(g) = col_gap {
                let pos = right_edge(g) + cw - 1;
                if pos < hwidth {
                    for buf in [&mut top, &mut bot] {
                        if buf[pos] == ' ' {
                            buf[pos] = '⋯';
                        }
                    }
                }
            }
            let top: String = top.into_iter().collect();
            let bot: String = bot.into_iter().collect();
            lines.push(Line::from(Span::styled(top.trim_end().to_string(), dim)));
            lines.push(Line::from(Span::styled(bot.trim_end().to_string(), dim)));
        } else {
            let mut header = String::new();
            header.push_str(&format!("{:>lw$}", ""));
            for (j, &c) in sample.cols.iter().enumerate() {
                header.push_str(&format!("{c:>cw$}"));
                if Some(j) == col_gap {
                    header.push_str(&format!("{:>cw$}", "⋯"));
                }
            }
            lines.push(Line::from(Span::styled(header, dim)));
        }

        let integer = sample.view.is_integer(&tensor.dtype);
        let band = |k: usize| {
            if k.is_multiple_of(2) {
                palette::STRIPE_DARK
            } else {
                palette::STRIPE_LITE
            }
        };
        for (i, row) in sample.values.iter().enumerate() {
            // Row striping bands the whole line; carried as a per-span background
            // so the index label is included like the raw path's band start.
            let row_bg = (stripe == StripeMode::Rows).then(|| band(i));
            let bg_style = |base: Style| match row_bg {
                Some(c) => base.bg(c),
                None => base,
            };
            let mut spans: Vec<Span> = Vec::new();
            // Dimmed row index.
            spans.push(Span::styled(
                format!("{:>lw$}", sample.rows[i]),
                bg_style(dim),
            ));
            let mut vcol = 0usize;
            for (j, &v) in row.iter().enumerate() {
                let s = match base {
                    NumBase::Decimal if integer => format!("{:>cw$}", v as i64),
                    NumBase::Decimal => format!("{v:>cw$.3e}"),
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
                let col_bg = (stripe == StripeMode::Cols).then(|| band(vcol));
                spans.extend(grid_cell_spans(&s, col_bg, false, row_bg));
                vcol += 1;
                if Some(j) == col_gap {
                    let col_bg = (stripe == StripeMode::Cols).then(|| band(vcol));
                    spans.extend(grid_cell_spans(
                        &format!("{:>cw$}", "⋯"),
                        col_bg,
                        true,
                        row_bg,
                    ));
                    vcol += 1;
                }
            }
            lines.push(Line::from(spans));
            // Dotted row marking the rows skipped after the gap.
            if Some(i) == row_gap {
                let mut s = String::new();
                s.push_str(&format!("{:>lw$}", "⋮"));
                for j in 0..row.len() {
                    s.push_str(&format!("{:>cw$}", "⋮"));
                    if Some(j) == col_gap {
                        s.push_str(&format!("{:>cw$}", "⋱"));
                    }
                }
                lines.push(Line::from(Span::styled(s, dim)));
            }
        }

        lines.push(Line::default());
        let footer_top = lines.len() as u16;
        let (footer, chips) = data_view_footer_wrapped_lines(
            sample.mode,
            sample.slices,
            sample.overridable,
            false,
            stripe,
            base,
            width,
        );
        lines.extend(footer);

        Paragraph::new(lines).render(area, frame.buffer_mut());
        if let (Some(row), Some((ratio, label))) = (stats_gauge_row, computing_gauge(stats)) {
            render_line_gauge(
                frame,
                Rect {
                    x: 0,
                    y: row as u16,
                    width: area.width,
                    height: 1,
                },
                label,
                ratio,
                Some(30),
            );
        }
        data_view_regions(frame, &chips, footer_top)
    }

    /// The Ratatui port of [`Self::draw_dtype_menu`]: overlay a dtype-selection
    /// menu on the bottom two rows of the live preview frame — a `view as:` label
    /// followed by the available views as buttons (`current` highlighted), with a
    /// hint line below. Composited *after* the preview is drawn into the frame.
    pub fn render_dtype_menu(frame: &mut Frame, options: &[ViewDtype], current: usize) {
        let mut menu: Vec<Span> = vec![dim_span("view as:")];
        for (i, opt) in options.iter().enumerate() {
            let label = format!(" {} ", opt.menu_label());
            if i == current {
                menu.push(Span::styled(
                    label,
                    Style::default()
                        .fg(palette::SELECT_FG)
                        .bg(palette::SELECT_BG)
                        .add_modifier(Modifier::BOLD),
                ));
            } else {
                menu.push(dim_span(label));
            }
        }
        let hints = Line::from(hint_spans(&[
            ("← → or d/D", "move"),
            ("Enter", "apply"),
            ("Esc", "cancel"),
        ]));
        render_bottom_band(frame, Line::from(menu), hints);
    }

    /// The Ratatui port of [`Self::draw_slice_prompt`]: a bottom-pinned prompt to
    /// jump to a slice by index (over the live data view), with a fixed-width
    /// input box and a feedback line below for an out-of-range / invalid entry.
    pub fn render_slice_prompt(frame: &mut Frame, slices: usize, input: &str, error: Option<&str>) {
        let mut prompt: Vec<Span> = vec![
            Span::styled("Go to slice ", Style::default().fg(palette::KEY)),
            dim_span(format!("(0-{} or 0-100%)", slices.saturating_sub(1))),
            Span::raw("  "),
        ];
        prompt.extend(input_box_spans(input, input.chars().count(), 5));
        prompt.push(Span::raw("  "));
        prompt.push(key_span("Enter"));
        prompt.push(dim_span(" to jump · "));
        prompt.push(key_span("Esc"));
        prompt.push(dim_span(" to cancel"));
        render_bottom_band(frame, Line::from(prompt), error_line(error));
    }

    /// The Ratatui port of [`Self::draw_reshape_prompt`]: shows the stored shape
    /// and the element count the entry must multiply to, the input box, and a
    /// feedback line for errors.
    pub fn render_reshape_prompt(
        frame: &mut Frame,
        elements: usize,
        stored: &[usize],
        input: &str,
        error: Option<&str>,
    ) {
        let mut prompt: Vec<Span> = vec![
            Span::styled(
                format!("Reshape {} ", format_shape(stored)),
                Style::default().fg(palette::KEY),
            ),
            dim_span(format!(
                "(dims multiplying to {elements}; `-1`/`*`/`_` infers one; empty clears)"
            )),
            Span::raw("  "),
        ];
        prompt.extend(input_box_spans(input, input.chars().count(), 16));
        prompt.push(Span::raw("  "));
        prompt.push(key_span("Enter"));
        prompt.push(dim_span(" to apply · "));
        prompt.push(key_span("Esc"));
        prompt.push(dim_span(" to cancel"));
        render_bottom_band(frame, Line::from(prompt), error_line(error));
    }

    /// The Ratatui port of [`Self::draw_text_prompt`]: a bottom-pinned free-text
    /// input (label + editable box + optional error line). Used for the repack
    /// output filename, buffer size, and histogram bin count.
    pub fn render_text_prompt(frame: &mut Frame, label: &str, input: &str, error: Option<&str>) {
        let mut prompt: Vec<Span> = vec![Span::styled(
            format!("{label} "),
            Style::default().fg(palette::KEY),
        )];
        prompt.extend(input_box_spans(input, input.chars().count(), 24));
        prompt.push(Span::raw("  "));
        prompt.push(key_span("Enter"));
        prompt.push(dim_span(" to confirm · "));
        prompt.push(key_span("Esc"));
        prompt.push(dim_span(" to cancel"));
        render_bottom_band(frame, Line::from(prompt), error_line(error));
    }

    /// The Ratatui port of [`Self::draw_choice_menu`]: a full-screen single-choice
    /// menu — a title, an underline rule, and a strip of `options` with `current`
    /// highlighted, plus a hint line. Used to pick the repack codec / confirm.
    pub fn render_choice_menu(frame: &mut Frame, title: &str, options: &[&str], current: usize) {
        let mut strip: Vec<Span> = Vec::new();
        for (i, opt) in options.iter().enumerate() {
            let label = format!(" {opt} ");
            if i == current {
                strip.push(Span::styled(
                    label,
                    Style::default()
                        .fg(palette::SELECT_FG)
                        .bg(palette::SELECT_BG)
                        .add_modifier(Modifier::BOLD),
                ));
            } else {
                strip.push(dim_span(label));
            }
            strip.push(Span::raw(" "));
        }
        let lines: Vec<Line> = vec![
            Line::from(Span::raw(title.to_string())),
            Line::from(Span::raw("=".repeat(title.len().max(10)))),
            Line::default(),
            Line::from(strip),
            Line::default(),
            Line::from(hint_spans(&[
                ("← →", "move"),
                ("Enter", "select"),
                ("Esc", "cancel"),
            ])),
        ];
        Paragraph::new(lines).render(frame.area(), frame.buffer_mut());
    }

    /// The Ratatui port of [`Self::draw_progress`]: a full-screen progress view
    /// with a 40-cell bar, a `done/total` count and a detail line (e.g. the
    /// dataset currently being written).
    #[cfg(feature = "hdf5")]
    pub fn render_progress(
        frame: &mut Frame,
        title: &str,
        done: usize,
        total: usize,
        detail: &str,
    ) {
        let frac = if total > 0 {
            done as f64 / total as f64
        } else {
            0.0
        };
        let area = frame.area();
        // Title + rule on rows 0–1; a blank row 2; the gauge on row 3; the detail
        // line on row 4 — same layout as before, but the bar is a native LineGauge.
        Paragraph::new(vec![
            Line::from(Span::raw(title.to_string())),
            Line::from(Span::raw("=".repeat(title.len().max(10)))),
        ])
        .render(area, frame.buffer_mut());
        if area.height > 3 {
            render_line_gauge(
                frame,
                Rect {
                    x: 0,
                    y: 3,
                    width: area.width,
                    height: 1,
                },
                Line::from(format!("{done}/{total}")),
                frac,
                None,
            );
        }
        if area.height > 4 {
            Paragraph::new(Line::from(dim_span(detail.to_string()))).render(
                Rect {
                    x: 0,
                    y: 4,
                    width: area.width,
                    height: 1,
                },
                frame.buffer_mut(),
            );
        }
    }

    /// The Ratatui port of [`Self::draw_message`]: a simple full-screen message
    /// (title, underline rule, body, footer) over the pop-up panel surface.
    pub fn render_message(frame: &mut Frame, title: &str, message: &str) {
        render_popup_box(
            frame,
            title,
            vec![
                Line::from(Span::raw(message.to_string())),
                Line::default(),
                Line::from(dim_span("Click or press any key to return...")),
            ],
            Backdrop::Fill,
        );
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
    /// The Ratatui port of [`Self::draw_copied_flash`]: a bold green "✓ Copied …"
    /// confirmation composited over the frame's bottom row (clamped to the width
    /// so it never wraps and scrolls). Drawn last, over the live detail/data
    /// frame, so the content above it stays put.
    pub fn render_copied_flash(frame: &mut Frame, what: &str) {
        let area = frame.area();
        let width = area.width as usize;
        let full = format!("✓ Copied {what} to the clipboard");
        let msg: String = if full.chars().count() > width {
            full.chars()
                .take(width.saturating_sub(1))
                .chain(std::iter::once('…'))
                .collect()
        } else {
            full
        };
        Paragraph::new(Line::from(Span::styled(
            msg,
            Style::default()
                .fg(palette::SUCCESS)
                .add_modifier(Modifier::BOLD),
        )))
        .render(
            Rect {
                x: 0,
                y: area.height.saturating_sub(1),
                width: area.width,
                height: 1,
            },
            frame.buffer_mut(),
        );
    }

    /// Flash a transient warning `msg` on the bottom line (over whatever the view
    /// drew there), until the next redraw clears it — e.g. the wrong-keyboard-layout
    /// hint. Bold yellow, clamped to the width so it never wraps.
    pub fn render_notice(frame: &mut Frame, msg: &str) {
        let area = frame.area();
        let width = area.width as usize;
        let text: String = if msg.chars().count() > width {
            msg.chars()
                .take(width.saturating_sub(1))
                .chain(std::iter::once('…'))
                .collect()
        } else {
            msg.to_string()
        };
        Paragraph::new(Line::from(Span::styled(
            text,
            Style::default()
                .fg(palette::WARN)
                .add_modifier(Modifier::BOLD),
        )))
        .render(
            Rect {
                x: 0,
                y: area.height.saturating_sub(1),
                width: area.width,
                height: 1,
            },
            frame.buffer_mut(),
        );
    }

    /// The Ratatui port of [`Self::draw_health_warning`]: a full-screen warning
    /// panel summarising checkpoint health issues. Each category is capped so the
    /// panel stays small; missing items are red, extra items yellow.
    pub fn render_health_warning(frame: &mut Frame, reports: &[HealthReport]) {
        let mut lines: Vec<Line> = vec![
            Line::from(Span::styled(
                "⚠  Checkpoint health check",
                Style::default().fg(palette::WARN),
            )),
            Line::from(Span::styled(
                "=".repeat(60),
                Style::default().fg(palette::WARN),
            )),
        ];
        for report in reports {
            lines.push(Line::default());
            lines.push(Line::from(Span::styled(
                format!(
                    "{} does not match the .safetensors files on disk.",
                    report.index_path
                ),
                Style::default().fg(palette::WARN),
            )));
            lines.push(Line::default());
            health_section_lines(
                &mut lines,
                "Referenced by the index but MISSING",
                &report.missing_files,
                palette::ERROR,
            );
            health_section_lines(
                &mut lines,
                "Present on disk but NOT in the index",
                &report.extra_files,
                palette::WARN,
            );
            health_section_lines(
                &mut lines,
                "Expected by the index but absent from their file",
                &report.missing_tensors,
                palette::ERROR,
            );
            health_section_lines(
                &mut lines,
                "In files but not listed in the index",
                &report.extra_tensors,
                palette::WARN,
            );
        }
        lines.push(Line::from(dim_span(
            "The explorer scans the directory directly when the index is stale. Click or press any key to return.",
        )));
        Paragraph::new(lines).render(frame.area(), frame.buffer_mut());
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

/// One legend row as a styled [`Line`]: a two-space indent, the `symbol` (in
/// `color`, else default), then the description starting at absolute column
/// `desc_col`. The gap is filled with spaces sized to the symbol's *rendered*
/// display width, so the description lines up. An all-empty row is a blank
/// separator.
fn legend_row_line(color: Option<Color>, symbol: &str, desc: &str, desc_col: u16) -> Line<'static> {
    use unicode_width::UnicodeWidthStr;
    if symbol.is_empty() && desc.is_empty() {
        return Line::default();
    }
    let mut spans: Vec<Span> = vec![Span::raw("  ")];
    match color {
        Some(c) => spans.push(Span::styled(symbol.to_string(), Style::default().fg(c))),
        None => spans.push(Span::raw(symbol.to_string())),
    }
    // Pad from the current column (2 + rendered symbol width) to `desc_col`.
    let used = 2 + symbol.width();
    let pad = (desc_col as usize).saturating_sub(used).max(1);
    spans.push(Span::raw(" ".repeat(pad)));
    spans.push(Span::raw(desc.to_string()));
    Line::from(spans)
}

/// How a centred pop-up box treats the frame around it.
enum Backdrop {
    /// Leave the live frame intact around the box, clearing only the box's own
    /// rect — for a true pop-up (the legend `l`) that floats over a still-visible
    /// tree / detail view.
    Float,
    /// Wipe the whole frame to the [`palette::SCRIM`] first — for standalone
    /// message screens that own the frame (nothing is drawn beneath), so no
    /// terminal default background shows around the box.
    Fill,
}

/// A centred, content-sized pop-up over the frame: a rounded [`Block`] (accent
/// border, `title` on the top edge, panel background) wrapping `content`. With
/// [`Backdrop::Float`] the surrounding frame is left untouched (only the box rect
/// is cleared) so the screen behind stays visible — a real pop-up; with
/// [`Backdrop::Fill`] the whole frame is wiped to the scrim first, for standalone
/// message screens. Shared by the legend pop-up and message screens.
fn render_popup_box(
    frame: &mut Frame,
    title: &str,
    content: Vec<Line<'static>>,
    backdrop: Backdrop,
) {
    let area = frame.area();
    let inner_w = content
        .iter()
        .map(Line::width)
        .max()
        .unwrap_or(0)
        .max(title.chars().count() + 2);
    let box_w = ((inner_w + 4) as u16).min(area.width); // 2 borders + 2 padding
    let box_h = ((content.len() + 2) as u16).min(area.height); // 2 borders
    let rect = Rect {
        x: area.width.saturating_sub(box_w) / 2,
        y: area.height.saturating_sub(box_h) / 2,
        width: box_w,
        height: box_h,
    };
    match backdrop {
        // Float over the live frame: clear only the box's own rect so the block
        // paints on a clean surface, while the screen behind stays visible around it.
        Backdrop::Float => Clear.render(rect, frame.buffer_mut()),
        // Own the frame: wipe every glyph, then paint the scrim, so nothing shows
        // through around the box.
        Backdrop::Fill => {
            Clear.render(area, frame.buffer_mut());
            Block::default()
                .style(Style::default().bg(palette::SCRIM))
                .render(area, frame.buffer_mut());
        }
    }

    let panel = Style::default().bg(palette::PANEL_BG);
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(palette::ACCENT))
        .title(Span::styled(
            format!(" {title} "),
            Style::default()
                .fg(palette::KEY)
                .add_modifier(Modifier::BOLD),
        ))
        .padding(Padding::horizontal(1))
        .style(panel);
    let inner = block.inner(rect);
    block.render(rect, frame.buffer_mut());
    Paragraph::new(content)
        .style(panel)
        .render(inner, frame.buffer_mut());
}

/// A full-width pop-up framed with only top+bottom borders (the `title` rides the
/// top rule) over the live frame, centred vertically. Its body rows stay flush at
/// column 0 — used by the copied-command pop-up so the command can still be
/// selected cleanly by hand when the OSC-52 copy doesn't reach the terminal.
fn render_titled_bar(frame: &mut Frame, title: Line<'static>, content: Vec<Line<'static>>) {
    let area = frame.area();
    let box_h = ((content.len() + 2) as u16).min(area.height);
    let rect = Rect {
        x: 0,
        y: area.height.saturating_sub(box_h) / 2,
        width: area.width,
        height: box_h,
    };
    let panel = Style::default().bg(palette::PANEL_BG);
    let block = Block::default()
        .borders(Borders::TOP | Borders::BOTTOM)
        .border_style(Style::default().fg(palette::ACCENT))
        .title(title)
        .style(panel);
    let inner = block.inner(rect);
    Clear.render(rect, frame.buffer_mut());
    block.render(rect, frame.buffer_mut());
    Paragraph::new(content)
        .style(panel)
        .render(inner, frame.buffer_mut());
}

/// Composite a bottom-pinned two-row prompt (`prompt` on the second-to-last row,
/// `feedback` on the last) over the live frame — the Ratatui equivalent of the
/// raw prompts' `MoveTo(0, h-2)` / `MoveTo(0, h-1)` line writes. Each row is
/// cleared (its tail blanked) so a shorter new prompt leaves nothing stale behind.
fn render_bottom_band(frame: &mut Frame, prompt: Line<'static>, feedback: Line<'static>) {
    let area = frame.area();
    if area.height < 2 {
        return;
    }
    Paragraph::new(prompt).render(
        Rect {
            x: 0,
            y: area.height - 2,
            width: area.width,
            height: 1,
        },
        frame.buffer_mut(),
    );
    Paragraph::new(feedback).render(
        Rect {
            x: 0,
            y: area.height - 1,
            width: area.width,
            height: 1,
        },
        frame.buffer_mut(),
    );
}

/// The feedback line below a prompt: a red error message, or an empty line (which
/// still clears the row) when there's nothing to report.
fn error_line(error: Option<&str>) -> Line<'static> {
    match error {
        Some(msg) => Line::from(Span::styled(
            msg.to_string(),
            Style::default().fg(palette::ERROR),
        )),
        None => Line::default(),
    }
}

/// The input box as styled spans — the Ratatui port of [`input_box`]: a padded,
/// input-coloured field with the caret drawn as an inverted character (or a block
/// at the end), padded to at least `min_chars`.
fn input_box_spans(text: &str, cursor: usize, min_chars: usize) -> Vec<Span<'static>> {
    let field = Style::default().fg(palette::INPUT_FG).bg(palette::INPUT_BG);
    let caret = Style::default().fg(palette::INPUT_BG).bg(palette::INPUT_FG);
    let chars: Vec<char> = text.chars().collect();
    let cursor = cursor.min(chars.len());
    let mut spans: Vec<Span> = vec![Span::styled(" ", field)];
    for (i, ch) in chars.iter().enumerate() {
        let style = if i == cursor { caret } else { field };
        spans.push(Span::styled(ch.to_string(), style));
    }
    if cursor >= chars.len() {
        spans.push(Span::styled("█", field));
    }
    if chars.len() < min_chars {
        spans.push(Span::styled(" ".repeat(min_chars - chars.len()), field));
    }
    spans.push(Span::styled(" ", field));
    spans
}

/// Append one health-report section to `lines` (the Ratatui port of
/// [`health_section`]): a coloured title with the count, up to `CAP` items, and a
/// dimmed "… and N more" when capped, then a blank separator. Empty sections are
/// skipped.
fn health_section_lines(
    lines: &mut Vec<Line<'static>>,
    title: &str,
    items: &[String],
    color: Color,
) {
    if items.is_empty() {
        return;
    }
    const CAP: usize = 6;
    lines.push(Line::from(Span::styled(
        format!("{title} ({}):", items.len()),
        Style::default().fg(color),
    )));
    for item in items.iter().take(CAP) {
        lines.push(Line::from(Span::raw(format!("  {item}"))));
    }
    if items.len() > CAP {
        lines.push(Line::from(dim_span(format!(
            "  … and {} more",
            items.len() - CAP
        ))));
    }
    lines.push(Line::default());
}

/// The legend pop-up's box title, one per screen.
fn legend_title(legend: Legend) -> &'static str {
    match legend {
        Legend::Tree => "Legend — checkpoint tree",
        Legend::Detail => "Legend — tensor details",
        Legend::Heatmap => "Legend — heatmap",
        Legend::Values => "Legend — numeric values",
    }
}

/// The legend pop-up's body rows (the framing title comes from [`legend_title`]).
fn legend_band_lines(legend: Legend) -> Vec<Line<'static>> {
    let mut lines: Vec<Line> = Vec::new();

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
                    "≡ N",
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
                    Style::default().fg(heat_color(i as f64 / 23.0)),
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
                Span::styled(" 12 ", Style::default().bg(palette::STRIPE_DARK)),
                Span::styled(" 34 ", Style::default().bg(palette::STRIPE_LITE)),
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
    lines.push(Line::from(dim_span("Click or press any key to close.")));
    lines
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

/// The data-view title block as styled [`Line`]s — the Ratatui port of
/// [`write_data_view_title`]: the view label and tensor name, then a dimmed
/// `File:` and source path, each clipped (tail-kept) to `width` so both stay on
/// screen above a grid of any size. `kind` is the view label (`Values` / `Heatmap`).
fn data_view_title_lines(kind: &str, tensor: &TensorInfo, width: usize) -> Vec<Line<'static>> {
    vec![
        Line::from(vec![
            Span::raw(format!("{kind}: ")),
            Span::raw(truncate_keep_end(
                &tensor.name,
                width.saturating_sub(kind.len() + 2),
            )),
        ]),
        Line::from(vec![
            dim_span("File: "),
            Span::raw(truncate_keep_end(
                &tensor.source_path,
                width.saturating_sub(6),
            )),
        ]),
    ]
}

/// The data-view dtype span(s) — Ratatui port of [`write_view_dtype`]: just the
/// stored dtype, or a dimmed `stored as` + the bold reinterpretation label.
fn view_dtype_spans(
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

/// The data-view shape span(s) — Ratatui port of [`write_view_shape`].
fn view_shape_spans(stored: &[usize], logical: &[usize]) -> Vec<Span<'static>> {
    if stored == logical {
        vec![Span::raw(format_shape(logical))]
    } else {
        vec![
            dim_span(format!("{} as ", format_shape(stored))),
            key_span(format_shape(logical)),
        ]
    }
}

/// The one-line statistics view for a data screen as a styled [`Line`] — Ratatui
/// port of [`write_stats_view`]: the finished stats, a spinner while computing,
/// or `None` while pending (the raw path writes nothing then).
fn data_stats_view_line(stats: StatsView) -> Option<Line<'static>> {
    match stats {
        StatsView::Ready(s) => Some(Line::from(detail_stats_summary_spans(s))),
        StatsView::Computing {
            spinner,
            elapsed,
            progress,
        } => Some(Line::from(detail_computing_spans(
            spinner, elapsed, progress,
        ))),
        StatsView::Pending => None,
    }
}

/// The 3D slice-navigation header as a styled [`Line`] — Ratatui port of
/// [`write_slice_header`]. Only used when `sample.slices > 1`.
fn slice_header_line(sample: &Sample) -> Line<'static> {
    let mut spans: Vec<Span> = Vec::new();
    match sample.unpacked_field {
        Some(f) => spans.push(Span::raw(format!(
            "expert {} of {} — stored word {}, field {}/{} ({}-bit) — ",
            sample.slice, sample.slices, f.stored_slice, f.field, f.len_p, f.field_bits,
        ))),
        None => spans.push(Span::raw(format!(
            "slice {} of {} (fixed leading index) — ",
            sample.slice, sample.slices
        ))),
    }
    let items: &[(&str, &str)] = if matches!(sample.mode, SampleMode::Grid) {
        &[
            ("← →", "step"),
            ("Shift+← →", "jump 5% (both wrap)"),
            ("/", "index or %"),
        ]
    } else {
        &[("[ ]", "step"), ("/", "index or %")]
    };
    spans.extend(hint_spans(items));
    Line::from(spans)
}

/// A hint line `key label · key label · …` as styled spans, keys highlighted —
/// the Ratatui port of [`hint_line`]. An empty key writes the label plain; an
/// empty label writes just the key.
fn hint_spans(items: &[(&str, &str)]) -> Vec<Span<'static>> {
    let dim = Style::default().fg(palette::DIM);
    let mut spans: Vec<Span> = Vec::new();
    for (i, (key, label)) in items.iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled(" · ", dim));
        }
        if key.is_empty() {
            spans.push(Span::raw(label.to_string()));
        } else {
            spans.push(key_span(key.to_string()));
            if !label.is_empty() {
                spans.push(Span::raw(format!(" {label}")));
            }
        }
    }
    spans
}

/// The [`KeyEvent`] a click on a data-view footer chip synthesizes, or `None` for
/// the non-clickable chips (the multi-arrow nav hints `← →`, `[ ]`, `Home/End`, …,
/// and the trailing "any other key…" label). Every clickable chip is a single
/// glyph, so it never straddles a wrap boundary.
fn footer_key_event(key: &str) -> Option<KeyEvent> {
    if key == "⌫" {
        return Some(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE));
    }
    let mut chars = key.chars();
    match (chars.next(), chars.next()) {
        (Some(c), None) if "mvdrezbcyl/\\".contains(c) => Some(hint_key(c)),
        _ => None,
    }
}

/// The data-view footer as styled [`Line`]s — the Ratatui port of
/// [`write_view_footer`]. The raw footer is one logical line the terminal hard-
/// wraps at the screen edge (mid-chip, even mid-word), so we hard-wrap the styled
/// spans by character at `width` to match that exactly. Also returns the clickable
/// command chips: their char position in the unwrapped run maps to a (line, col)
/// by the same `width` hard wrap (each clickable chip is one glyph, so it never
/// splits across lines).
fn data_view_footer_wrapped_lines(
    mode: SampleMode,
    slices: usize,
    overridable: bool,
    heatmap: bool,
    stripe: StripeMode,
    base: NumBase,
    width: usize,
) -> (Vec<Line<'static>>, Vec<ChipHit>) {
    let items = view_footer_items(mode, slices, overridable, heatmap, stripe, base);
    let lines = wrap_spans_hard(hint_spans(&items), width);

    let w = width.max(1);
    let mut chips: Vec<ChipHit> = Vec::new();
    let mut pos = 0usize; // char offset into the unwrapped span run
    for (i, (key, label)) in items.iter().enumerate() {
        if i > 0 {
            pos += 3; // " · " (kept inert — not part of any chip)
        }
        let key_chars = key.chars().count();
        let advance = if key.is_empty() {
            label.chars().count()
        } else if label.is_empty() {
            key_chars
        } else {
            key_chars + 1 + label.chars().count()
        };
        // A single-action chip is clickable across its whole "key label" span.
        // The footer is hard-wrapped at `w`, so the span can straddle lines —
        // emit one region per line it covers.
        if let Some(kev) = footer_key_event(key) {
            let (start, end) = (pos, pos + advance);
            let mut c = start;
            while c < end {
                let line = c / w;
                let seg_end = end.min((line + 1) * w);
                chips.push(ChipHit {
                    line: line as u16,
                    col: (c % w) as u16,
                    width: (seg_end - c) as u16,
                    key: kev,
                });
                c = seg_end;
            }
        }
        pos += advance;
    }
    (lines, chips)
}

/// Hard-wrap a styled span run at `width` characters, splitting mid-span (and
/// mid-word) exactly where a terminal's line wrap would, so a footer built as
/// one logical line lays out identically across the migration.
fn wrap_spans_hard(spans: Vec<Span<'static>>, width: usize) -> Vec<Line<'static>> {
    let width = width.max(1);
    let mut lines: Vec<Line> = Vec::new();
    let mut cur: Vec<Span> = Vec::new();
    let mut col = 0usize;
    for span in spans {
        let style = span.style;
        let mut buf = String::new();
        for ch in span.content.chars() {
            if col == width {
                buf.clear_into_line(&mut cur, style);
                lines.push(Line::from(std::mem::take(&mut cur)));
                col = 0;
            }
            buf.push(ch);
            col += 1;
        }
        if !buf.is_empty() {
            cur.push(Span::styled(buf, style));
        }
    }
    if !cur.is_empty() {
        lines.push(Line::from(cur));
    }
    lines
}

/// Helper for [`wrap_spans_hard`]: flush the in-progress char buffer to the
/// current line as a styled span (no-op when empty), leaving the buffer cleared.
trait FlushSpan {
    fn clear_into_line(&mut self, cur: &mut Vec<Span<'static>>, style: Style);
}
impl FlushSpan for String {
    fn clear_into_line(&mut self, cur: &mut Vec<Span<'static>>, style: Style) {
        if !self.is_empty() {
            cur.push(Span::styled(std::mem::take(self), style));
        }
    }
}

/// One numeric-grid cell as styled span(s) — the Ratatui port of
/// [`write_grid_cell`]. `col_bg` is the column-stripe background (which, like the
/// raw path, bands all but the cell's first column so every stripe is the same
/// width and a one-space gutter separates neighbours); `row_bg` is the ambient
/// row-stripe background carried across the whole cell (incl. the gutter), set
/// once by the caller for a row band. `dim` dims the glyphs (the "⋯" gap marker).
fn grid_cell_spans(
    s: &str,
    col_bg: Option<ratatui::style::Color>,
    dim: bool,
    row_bg: Option<ratatui::style::Color>,
) -> Vec<Span<'static>> {
    let base = if dim {
        Style::default().fg(palette::DIM)
    } else {
        Style::default()
    };
    let with_bg = |style: Style, bg: Option<ratatui::style::Color>| match bg {
        Some(c) => style.bg(c),
        None => style,
    };
    match col_bg {
        // Leave the first column an uncoloured gutter (just the row band, if any)
        // and band the rest, so the stripe is the same width for every column.
        Some(c) => {
            let split = s.char_indices().nth(1).map_or(s.len(), |(i, _)| i);
            let (gutter, band) = s.split_at(split);
            vec![
                Span::styled(gutter.to_string(), with_bg(base, row_bg)),
                Span::styled(band.to_string(), with_bg(base, Some(c))),
            ]
        }
        None => vec![Span::styled(s.to_string(), with_bg(base, row_bg))],
    }
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
/// highlighter can be styled from the same constants as the rest of the UI. The
/// 16 ANSI-named indices map to yansi's named colors (so e.g. `Indexed(8)` emits
/// the bright-black SGR, not `38;5;8`); other indices use the 256-color cube.
fn to_yansi(color: Color) -> yansi::Color {
    use yansi::Color as Y;
    match color {
        Color::Black | Color::Indexed(0) => Y::Black,
        Color::Red | Color::Indexed(1) => Y::Red,
        Color::Green | Color::Indexed(2) => Y::Green,
        Color::Yellow | Color::Indexed(3) => Y::Yellow,
        Color::Blue | Color::Indexed(4) => Y::Blue,
        Color::Magenta | Color::Indexed(5) => Y::Magenta,
        Color::Cyan | Color::Indexed(6) => Y::Cyan,
        Color::Gray | Color::Indexed(7) => Y::White,
        Color::DarkGray | Color::Indexed(8) => Y::BrightBlack,
        Color::LightRed | Color::Indexed(9) => Y::Red,
        Color::LightGreen | Color::Indexed(10) => Y::Green,
        Color::LightYellow | Color::Indexed(11) => Y::Yellow,
        Color::LightBlue | Color::Indexed(12) => Y::Blue,
        Color::LightMagenta | Color::Indexed(13) => Y::Magenta,
        Color::LightCyan | Color::Indexed(14) => Y::Cyan,
        Color::White | Color::Indexed(15) => Y::White,
        Color::Indexed(n) => Y::Fixed(n),
        Color::Rgb(r, g, b) => Y::Rgb(r, g, b),
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
            .fg(palette::SELECT_FG)
            .bg(palette::SELECT_BG)
    } else {
        Style::default().fg(color)
    };
    Span::styled(text.into(), style)
}

/// The tree browser's key-hint line(s), word-wrapped to `width` on the
/// ` · `-separated `key label` chips (the long hint spills onto a second line).
fn tree_hint_lines(can_repack: bool, width: u16) -> (Vec<Line<'static>>, Vec<ChipHit>) {
    use KeyCode::{Backspace, Down, Enter, Left, Right, Up};
    let plain = KeyModifiers::NONE;
    let shift = KeyModifiers::SHIFT;
    // Each chip's key text is a list of segments; a `Seg::Key` glyph is clickable
    // (and synthesizes its key), a `Seg::Sep` (`/`, `Shift+`) is not. Both halves
    // of a dual chip are thus independently clickable.
    let mut items: Vec<(Vec<Seg>, &str)> = vec![
        (
            vec![
                Seg::Key("↑", KeyEvent::new(Up, plain)),
                Seg::Sep("/"),
                Seg::Key("↓", KeyEvent::new(Down, plain)),
            ],
            "navigate",
        ),
        (
            vec![
                Seg::Key("←", KeyEvent::new(Left, plain)),
                Seg::Sep("/"),
                Seg::Key("→", KeyEvent::new(Right, plain)),
            ],
            "parent/child",
        ),
        (
            vec![
                Seg::Sep("Shift+"),
                Seg::Key("↑", KeyEvent::new(Up, shift)),
                Seg::Sep("/"),
                Seg::Key("↓", KeyEvent::new(Down, shift)),
            ],
            "sibling",
        ),
        (
            vec![
                Seg::Key("Enter", KeyEvent::new(Enter, plain)),
                Seg::Sep("/"),
                Seg::Key("Space", hint_key(' ')),
            ],
            "expand",
        ),
        (
            vec![
                Seg::Key("E", hint_key('E')),
                Seg::Sep("/"),
                Seg::Key("C", hint_key('C')),
            ],
            "all",
        ),
        (vec![Seg::Key("/", hint_key('/'))], "search"),
        (vec![Seg::Key("l", hint_key('l'))], "legend"),
        (vec![Seg::Key("c", hint_key('c'))], "copy screen"),
        (vec![Seg::Key("f", hint_key('f'))], "copy file"),
        (vec![Seg::Key("n", hint_key('n'))], "copy name"),
        (vec![Seg::Key("y", hint_key('y'))], "copy command"),
        (
            vec![
                Seg::Key("⌫", KeyEvent::new(Backspace, plain)),
                Seg::Sep("/"),
                Seg::Key("\\", hint_key('\\')),
            ],
            "back/fwd",
        ),
    ];
    if can_repack {
        items.push((vec![Seg::Key("R", hint_key('R'))], "repack"));
    }
    items.push((vec![Seg::Key("q", hint_key('q'))], "quit"));

    let width = width as usize;
    let key_style = Style::default()
        .fg(palette::KEY)
        .add_modifier(Modifier::BOLD);
    let sep_style = Style::default().fg(palette::DIM);
    let mut lines: Vec<Line> = Vec::new();
    let mut chips: Vec<ChipHit> = Vec::new();
    let mut spans: Vec<Span> = Vec::new();
    let mut col = 0usize;
    for (segs, label) in items {
        let key_text: String = segs.iter().map(Seg::text).collect();
        let item_w = key_text.chars().count() + 1 + label.chars().count();
        let has_prev = !spans.is_empty();
        if has_prev && col + 3 + item_w > width {
            lines.push(Line::from(std::mem::take(&mut spans)));
            col = 0;
        }
        if !spans.is_empty() {
            spans.push(Span::styled(" · ", sep_style));
            col += 3;
        }
        // A single-action chip is clickable across its whole "key label"; a dual
        // chip (two keys sharing a label) keeps one region per glyph, since each
        // glyph is a different action and the label between them is ambiguous.
        let key_count = segs.iter().filter(|s| matches!(s, Seg::Key(..))).count();
        if key_count == 1 {
            let key = segs
                .iter()
                .find_map(|s| match s {
                    Seg::Key(_, k) => Some(*k),
                    Seg::Sep(_) => None,
                })
                .unwrap();
            chips.push(ChipHit {
                line: lines.len() as u16,
                col: col as u16,
                width: item_w as u16,
                key,
            });
        } else {
            let mut off = 0usize;
            for seg in &segs {
                let n = seg.text().chars().count();
                if let Seg::Key(_, key) = seg {
                    chips.push(ChipHit {
                        line: lines.len() as u16,
                        col: (col + off) as u16,
                        width: n as u16,
                        key: *key,
                    });
                }
                off += n;
            }
        }
        spans.push(Span::styled(key_text, key_style));
        spans.push(Span::raw(format!(" {label}")));
        col += item_w;
    }
    if !spans.is_empty() {
        lines.push(Line::from(spans));
    }
    (lines, chips)
}

/// The search bar header line: `Search [query▒]  N matches  Enter view · …`.
fn tree_search_line(config: &DrawConfig) -> Line<'static> {
    let dim = Style::default().fg(palette::DIM);
    let key_style = Style::default()
        .fg(palette::KEY)
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
        Style::default().bg(palette::INPUT_BG).fg(palette::INPUT_FG),
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

/// One tree row as a styled [`Line`]: group names in the accent and dtypes amber,
/// with the name, shape and size at full strength and only the leaf marker /
/// storage tag dimmed; a `selected` row is drawn plain so the caller's highlight
/// reads cleanly.
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
                Some(n) => format!("≡ {n}, "),
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
    Span::styled(text.into(), Style::default().fg(palette::DIM))
}

/// A bold bright-cyan key span (e.g. `s`, `d`) — the Ratatui equivalent of the
/// raw [`key_hint`].
fn key_span(key: impl Into<String>) -> Span<'static> {
    Span::styled(
        key.into(),
        Style::default()
            .fg(palette::KEY)
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
            Style::default().fg(palette::WARN),
        ));
    }
    spans.push(dim_span(format!("  ({})", fmt_duration(s.elapsed))));
    spans
}

/// The "scan in progress" stats segment as styled spans — Ratatui port of
/// [`write_computing`]: an accent spinner, a dimmed label, a progress bar with a
/// percentage (when the fraction is known), and the running elapsed time.
/// Render a native ratatui [`LineGauge`] into `area`: `label` at the left, then a
/// thick line filled to `ratio` — accent for the done part, dim for the rest. The
/// one progress-bar primitive, shared by the full-screen repack bar and the inline
/// "computing…" statistics line.
/// `max_line` caps the gauge *line* to that many cells (the widget draws the label
/// then the line): the inline "computing…" bar passes `Some(30)` so it doesn't
/// stretch across the whole screen; the full-screen bar passes `None` (full width).
fn render_line_gauge(
    frame: &mut Frame,
    area: Rect,
    label: Line<'static>,
    ratio: f64,
    max_line: Option<usize>,
) {
    let area = match max_line {
        // LineGauge lays out `label` then a space then the line, so bound the width
        // to the label plus the wanted line length (clamped to what's available).
        Some(cells) => Rect {
            width: ((label.width() + 1 + cells) as u16).min(area.width),
            ..area
        },
        None => area,
    };
    LineGauge::default()
        .line_set(ratatui::symbols::line::THICK)
        .filled_style(
            Style::default()
                .fg(palette::KEY)
                .add_modifier(Modifier::BOLD),
        )
        .unfilled_style(Style::default().fg(palette::DIM))
        .label(label)
        .ratio(ratio.clamp(0.0, 1.0))
        .render(area, frame.buffer_mut());
}

/// When statistics are computing *with a known fraction*, the `(ratio, label)` for
/// a [`render_line_gauge`] row; otherwise `None` (the caller shows the normal stats
/// text — the spinner-only "computing…", the finished stats, or the "press s" hint).
fn computing_gauge(stats: StatsView) -> Option<(f64, Line<'static>)> {
    match stats {
        StatsView::Computing {
            spinner,
            elapsed,
            progress: Some(frac),
        } => {
            let frac = frac.clamp(0.0, 1.0);
            let label = Line::from(format!(
                "{spinner} computing statistics… {:>3.0}% · {} ",
                frac * 100.0,
                fmt_duration(elapsed)
            ));
            Some((frac, label))
        }
        _ => None,
    }
}

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
) -> (Vec<Line<'static>>, Option<usize>) {
    let mut lines: Vec<Line> = Vec::new();

    lines.push(Line::from(Span::styled(
        "Tensor Details",
        Style::default().fg(palette::ACCENT),
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
            Style::default().fg(palette::UNINDEXED),
        )));
    }
    lines.push(Line::default());

    // Exact whole-tensor statistics: shown once computed, else a hint. While a
    // scan reports a fraction, the row is a native progress bar — reserve a blank
    // line here and hand the caller its index to render a `LineGauge` over.
    let stats_gauge_row = if computing_gauge(stats).is_some() {
        let row = lines.len();
        lines.push(Line::default());
        Some(row)
    } else {
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
            // Only the fraction-less "computing…" reaches here (the gauge handles
            // the case with a fraction above).
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
        None
    };
    lines.push(Line::default());

    (lines, stats_gauge_row)
}

/// The detail screen's footer hint as wrapped [`Line`]s, split into `key label,`
/// chips that wrap at `width` (so a chip is never broken across lines). Mirrors
/// the tree's [`tree_hint_lines`] wrapping.
fn detail_footer_lines(overridable: bool, width: u16) -> (Vec<Line<'static>>, Vec<ChipHit>) {
    // Each chunk is a run of spans that should not be split across lines; the
    // trailing text of each (the comma + space) keeps the on-screen wording. The
    // second field lists the chunk's clickable glyphs as `(char offset within the
    // chunk, glyph width, key)` so a click can be turned into the equivalent key.
    type Chunk = (Vec<Span<'static>>, Vec<(usize, u16, KeyEvent)>);
    let key_chunk = |glyph: &'static str, rest: &'static str, key: KeyEvent| -> Chunk {
        (
            vec![key_span(glyph), Span::raw(rest)],
            vec![(0, glyph.chars().count() as u16, key)],
        )
    };
    let mut chunks: Vec<Chunk> = vec![(vec![Span::raw("Press ")], vec![])];
    chunks.push(key_chunk("m", " for a heatmap, ", hint_key('m')));
    chunks.push(key_chunk("v", " for numeric values, ", hint_key('v')));
    chunks.push(key_chunk("h", " for a histogram, ", hint_key('h')));
    chunks.push(key_chunk("b", " to set its bin count, ", hint_key('b')));
    if overridable {
        chunks.push(key_chunk("d", " to reinterpret the dtype, ", hint_key('d')));
        chunks.push(key_chunk("r", " to reshape, ", hint_key('r')));
    }
    chunks.push(key_chunk("c", " to copy, ", hint_key('c')));
    chunks.push(key_chunk("y", " to copy the command, ", hint_key('y')));
    chunks.push(key_chunk("l", " for the legend, ", hint_key('l')));
    // The dual ⌫ / \ chunk: two independently clickable glyphs. `\` sits at char
    // offset 4 (after "⌫" + " / ").
    chunks.push((
        vec![
            key_span("⌫"),
            Span::raw(" / "),
            key_span("\\"),
            Span::raw(" to step back / forward, any other key to return..."),
        ],
        vec![
            (0, 1, KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE)),
            (4, 1, hint_key('\\')),
        ],
    ));

    // Greedily pack chunks onto lines, breaking before a chunk that would overflow
    // (each chunk already carries its own separator, so no extra is inserted).
    let width = (width as usize).max(1);
    let chunk_w = |c: &[Span]| -> usize { c.iter().map(|s| s.content.chars().count()).sum() };
    let mut lines: Vec<Line> = Vec::new();
    let mut chips: Vec<ChipHit> = Vec::new();
    let mut spans: Vec<Span> = Vec::new();
    let mut col = 0usize;
    for (chunk, keys) in chunks {
        let w = chunk_w(&chunk);
        if !spans.is_empty() && col + w > width {
            lines.push(Line::from(std::mem::take(&mut spans)));
            col = 0;
        }
        // This chunk starts at column `col` on line `lines.len()`. A single-action
        // chunk is clickable across its whole width (key + wording); the dual ⌫/\
        // chunk keeps one region per glyph (the shared wording is ambiguous).
        if keys.len() == 1 {
            chips.push(ChipHit {
                line: lines.len() as u16,
                col: col as u16,
                width: w as u16,
                key: keys[0].2,
            });
        } else {
            for (off, glyph_w, key) in keys {
                chips.push(ChipHit {
                    line: lines.len() as u16,
                    col: (col + off) as u16,
                    width: glyph_w,
                    key,
                });
            }
        }
        spans.extend(chunk);
        col += w;
    }
    if !spans.is_empty() {
        lines.push(Line::from(spans));
    }
    (lines, chips)
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
        head.push(Span::styled(s, Style::default().fg(palette::ACCENT)));
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

    let accent = Style::default().fg(palette::ACCENT);
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

/// Map a normalized value in `[0, 1]` to a blue→green→red 256-color ramp
/// (the 6×6×6 ANSI color cube, indices 16..=231).
fn heat_color(t: f64) -> Color {
    let t = t.clamp(0.0, 1.0);
    let r = (t * 5.0).round() as u8;
    let b = ((1.0 - t) * 5.0).round() as u8;
    let g = ((1.0 - (t - 0.5).abs() * 2.0) * 5.0).round() as u8;
    Color::Indexed(16 + 36 * r + 6 * g + b)
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
