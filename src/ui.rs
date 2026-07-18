use std::collections::{HashMap, HashSet};
use std::time::Duration;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, BorderType, Borders, Clear, LineGauge, Padding, Paragraph, Scrollbar,
    ScrollbarOrientation, ScrollbarState, StatefulWidget, Widget,
};

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
    let mut regions = chip_regions(chips, footer_top);
    regions.extend(close_button(
        frame,
        KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE),
    ));
    regions
}

/// True when `(col, row)` falls inside a clickable region.
pub fn region_hit(regions: &[(Rect, KeyEvent)], col: u16, row: u16) -> Option<KeyEvent> {
    region_at(regions, col, row).map(|(_, k)| k)
}

/// The clickable region (its rect and key) under `(col, row)`, if any — like
/// [`region_hit`] but keeps the rect too, so a hover can anchor a help bubble to
/// the chip it points at.
pub fn region_at(regions: &[(Rect, KeyEvent)], col: u16, row: u16) -> Option<(Rect, KeyEvent)> {
    regions
        .iter()
        .find(|(r, _)| col >= r.x && col < r.x + r.width && row >= r.y && row < r.y + r.height)
        .copied()
}

/// Map a footer's [`ChipHit`]s (line/col relative to their hint block) to absolute
/// screen [`Rect`]s paired with each chip's key. `base_row` is the block's first
/// screen row: `1` for the tree/files header hints, or `footer_top` for the modes
/// whose hints sit in the bottom footer. Replaces the per-mode remap that was
/// copy-pasted across every screen renderer.
pub fn chip_regions(chips: &[ChipHit], base_row: u16) -> ChipRegions {
    chips
        .iter()
        .map(|c| {
            (
                Rect {
                    x: c.col,
                    y: base_row + c.line,
                    width: c.width,
                    height: 1,
                },
                c.key,
            )
        })
        .collect()
}

/// Which screen a footer shortcut sits on, so [`shortcut_help`] can disambiguate
/// keys that mean different things per screen (`h`, `b`, `r`, the arrows).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum HelpCtx {
    Tree,
    Files,
    Layout,
    Detail,
    Data,
    Rename,
}

/// A one-line help description for a footer shortcut `key` on screen `ctx`, shown
/// as a bubble when the mouse hovers the chip. `None` for keys with no help.
pub fn shortcut_help(key: KeyEvent, ctx: HelpCtx) -> Option<&'static str> {
    use HelpCtx::{Data, Detail, Files, Layout, Rename, Tree};
    use KeyCode::{Backspace, Char, Down, Left, PageDown, PageUp, Right, Tab, Up};
    let shift = key.modifiers.contains(KeyModifiers::SHIFT);
    let help = match (ctx, key.code) {
        // File browser.
        (Tree, Tab) => "Switch to the file browser — the checkpoint's directory.",
        (Files, Tab) => "Switch back to the tensor tree.",
        // safetensors layout map.
        (Layout, Char(' ') | Char(':')) => "Open the command palette — search and run any command.",
        (Layout, Tab) => "Switch back to the tensor tree.",
        (Layout, Up | Down) => "Move the selection to the previous / next segment.",
        (Layout, PageUp | PageDown) => "Move the selection by one screenful.",
        (Layout, KeyCode::Enter) => "Jump to the selected tensor's place in the tensor tree.",
        (Detail, Tab) => "Show this tensor in its file's byte-layout map.",
        (Files, Up | Down) => "Move the selection up / down one row.",
        (Files, Left | Right) => "Collapse a directory / expand it (or step to its parent).",
        (Files, PageUp | PageDown) => "Scroll the listing by one screenful.",
        (Files, KeyCode::Enter) => {
            "Expand a directory, open a checkpoint file, or preview a text / JSON sidecar."
        }
        (Files, Char(' ') | Char(':')) => "Open the command palette — search and run any command.",
        (Files, Char('f')) => "Copy the selected file's path.",
        // Tree navigation.
        (Tree, Up | Down) if shift => "Jump to the previous / next sibling at this depth.",
        (Tree, Up | Down) => "Move the selection up / down one row.",
        (Tree, Left | Right) => "Collapse to the parent group, or step into the child.",
        (Tree, PageUp | PageDown) => "Scroll the tree by one screenful.",
        (Tree, KeyCode::Enter) => "Open the selected tensor, or expand / collapse a group.",
        (Tree, Char(' ') | Char(':')) => "Open the command palette — search and run any command.",
        (Tree, Char('E')) => "Expand every group in the tree.",
        (Tree, Char('C')) => "Collapse every group in the tree.",
        (Tree, Char('/')) => "Search: filter tensors by name as you type.",
        (Tree, Char('h')) => "Run the checkpoint health checks and show the report.",
        (Tree, Char('s')) => {
            "Show overall checkpoint stats: sizes, params, dtype mix, layers, experts."
        }
        (Tree, Char('t')) => "Copy the tree or a flat tensor list — text or JSON (opens a menu).",
        (Tree, Char('f')) => "Copy the selected row's file path.",
        (Tree, Char('n')) => "Copy the selected tensor's name.",
        (Tree, Char('r')) => "Repack this HDF5 checkpoint into a new file with another codec.",
        (Tree, Char('R')) => {
            "Rename tensors in place (safetensors): rewrites shard headers and the index."
        }
        (Tree, Char('q')) => "Quit the explorer.",
        // Detail screen.
        (Detail, Char(' ') | Char(':')) => "Open the command palette — search and run any command.",
        (Detail | Data, Char('m')) => "Show the tensor as a heatmap.",
        (Detail | Data, Char('v')) => "Show the tensor as a grid of numeric values.",
        (Detail, Char('h')) => "Compute and show the value histogram.",
        (Detail, Char('b' | 'B')) => "Set the histogram's bucket count.",
        (Detail, Char('s')) => "Compute exact whole-tensor statistics (min/max, mean, std, …).",
        (Detail | Data, Char('d')) => "Reinterpret the stored dtype (e.g. u4, i4, bf16, f32).",
        (Detail | Data, Char('r' | 'R')) => "Reshape the tensor's dimensions (row-major).",
        // Data view.
        (Data, Char(' ') | Char(':')) => "Open the command palette — search and run any command.",
        (Data, Char('e' | 'E')) => "Cycle the layout: overview → edges → window.",
        (Data, Char('z' | 'Z')) => "Cycle zebra striping: rows → columns → off.",
        (Data, Char('b' | 'B')) => "Cycle the numeral base: dec → hex → oct → bin.",
        (Data, Char(']') | Char('[')) => "Step to the next / previous slice.",
        (Data, Up | Down | Left | Right) => {
            "Pan the view (Shift = one screenful, Ctrl = to the edge)."
        }
        // Rename editor — palette commands, keyed by their registry sentinel char
        // (the palette maps each to `KeyCode::Char(sentinel)`; see `RENAME_COMMANDS`).
        (Rename, Char(' ') | Char(':')) => "Open the command palette — search and run any command.",
        (Rename, Char('r') | Char('\u{12}')) => {
            "Apply the rename in place (asks for confirmation first)."
        }
        (Rename, Char('\r')) => "Move to the next field (past the last field, add a new rule).",
        (Rename, Char('\u{e}')) => "Add another source → new-name rule.",
        (Rename, Char('\u{4}')) => "Remove the focused rule.",
        (Rename, Char('y') | Char('\u{19}')) => {
            "Copy the CLI command that reopens this rename editor."
        }
        (Rename, Char('a') | Char('\u{1}')) => {
            "Copy the `convert --map` command that applies this rename non-interactively."
        }
        (Rename, Char('s') | Char('\u{13}')) => "Copy the whole screen's text to the clipboard.",
        (Rename, Char('l') | Char('\u{c}')) => "Show the legend for the rename editor's symbols.",
        (Rename, Char('\u{1b}')) => "Go back to the previous view.",
        (Rename, Char('\u{3}')) => "Quit the explorer.",
        // Common to every screen.
        (_, Char('l')) => "Show the legend for this screen's symbols and keys.",
        (_, Char('c')) => "Copy the whole screen's text to the clipboard.",
        (_, Char('y')) => "Copy the CLI command that reopens this exact screen.",
        (_, Backspace) => "Step back through view history.",
        (_, Char('\\')) => "Step forward through view history.",
        _ => return None,
    };
    Some(help)
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
    /// Filled-red *background* for an alert badge (white text on it) — high
    /// luminance contrast that reads clearly on the grey status bar, where any
    /// red *foreground* stays muddy against the mid-grey. The health *error* badge.
    pub const ALERT: Color = Color::Indexed(160);
    /// Filled-orange *background* for the health badge when there are only
    /// warnings (e.g. extra files on disk) — a softer alert than the red [`ALERT`],
    /// which is reserved for real errors (missing files/tensors).
    pub const WARN_BG: Color = Color::Indexed(166);
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

/// Severity of the checkpoint-health badge on the tree's status line: a real
/// error (missing files/tensors — the checkpoint may be incomplete) shows a red
/// badge; warnings only (e.g. extra files on disk not in the index) show a softer
/// orange one, so the screaming red is reserved for genuine problems.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum HealthAlert {
    Warning,
    Error,
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
    /// Whether the loaded checkpoint can be repacked (a single HDF5 file), which
    /// gates the `r` hint.
    pub can_repack: bool,
    /// Whether the loaded checkpoint can be renamed in place (a writable local
    /// safetensors checkpoint), which gates the `R` hint.
    pub can_rename: bool,
    /// `source_path`s of tensors present on disk but not listed in the index
    /// (a stale `model.safetensors.index.json`), flagged in the tree.
    pub unindexed: &'a HashSet<String>,
    /// Per-tensor fused-codebook packing schemas, keyed by tensor name. A tensor
    /// with one shows its logical (unmerged) dtype and shape beside the physical.
    pub packing_schemas: &'a HashMap<String, PackingSchema>,
    /// A transient "✓ Copied …" confirmation to flash on the bottom line (over
    /// the secondary status), set by the tree's copy shortcuts.
    pub copied_flash: Option<&'a str>,
    /// Whether this frame is drawn to the live, interactive terminal. Gates the
    /// scroll bar: a headless `--plain` / screen-copy render is a static text
    /// dump with no viewport, so it shows no bar (see [`UI::tree_scrollbar`]).
    pub interactive: bool,
    /// The bottom-right status badges (access / health / metadata-only), from
    /// [`UI::status_badges`], right-to-left; and which one the mouse is over (for
    /// its hover bubble). One uniform bar — see [`UI::render_badge_bar`].
    pub badges: &'a [Badge],
    pub hovered_badge: Option<usize>,
}

/// A clickable link in the UI — the app-wide primitive for "click a name to jump".
/// A safetensors filename links to its byte-layout view; a *concrete* tensor name
/// links to its place in the tree (a schema with `{layer}`/`{expert}` placeholders
/// matches many tensors, so it is never a link). Recorded per screen and dispatched
/// by [`Explorer::open_link`](crate::explorer). See the `links` field on `Explorer`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Link {
    /// Open the byte-layout view of this safetensors file (its full path).
    Layout(String),
    /// Reveal this concrete tensor in the tree.
    Tree(String),
}

/// A frame's footer-chip click regions — each replays a [`KeyEvent`].
pub type ChipRegions = Vec<(Rect, KeyEvent)>;
/// A frame's navigation-link regions — each opens another view via a [`Link`].
pub type LinkRegions = Vec<(Rect, Link)>;

/// One rule's line in the rename preview: the before→after *schema* plus how many
/// tensors it touches and how they break down by [`RenameStatus`]. Summarising per
/// rule keeps the preview a few lines even when a rule matches every layer.
pub struct RenameRuleView {
    pub from: String,
    pub to: String,
    /// Tensors whose name the rule *changes* (the rows the preview lists).
    pub total: usize,
    /// Tensors the pattern *matches*, changed or not — so a no-op rule (a
    /// just-autocompleted source whose new name is still identical) reads as
    /// "matches N · unchanged", not the misleading "matches no tensors".
    pub matched: usize,
    pub ok: usize,
    pub collide: usize,
    pub wont_fit: usize,
    pub invalid: usize,
    /// Per changed shard: how the rewritten header sizes up (the detail behind a
    /// `won't fit` verdict — which file, and by how many bytes).
    pub shards: Vec<crate::rename::ShardFit>,
}

/// One entry in the rename editor's autocomplete dropdown: a tensor-family schema,
/// how many tensors it covers, and (optionally) the char range of the typed query
/// within it to embolden.
pub struct RenameCompletion {
    pub text: String,
    /// Tensors this family schema covers — shown as a dim `×N` metadata column.
    pub count: usize,
    /// `(start, end)` char range of the literal query match, to embolden; `None`
    /// for a number-agnostic match (where the query has no literal counterpart).
    pub hl: Option<(usize, usize)>,
}

/// Everything [`UI::render_rename`] draws: the dynamic list of source→new-name rule
/// pairs (with the focused field + its autocomplete) and a compact, per-rule
/// before→after preview marking each rule's in-place feasibility.
pub struct RenameView<'a> {
    pub root: &'a str,
    /// `(source, new-name)` for each rule pair, in order.
    pub pairs: &'a [(String, String)],
    pub focus_pair: usize,
    /// Which field of `focus_pair` has focus: `false` = source, `true` = new-name.
    pub on_target: bool,
    /// Caret position (char index) within the focused field.
    pub cursor: usize,
    /// Whether the autocomplete dropdown is open at the focused field.
    pub menu_open: bool,
    /// The highlighted candidate in the dropdown (an index into `completions`).
    pub menu_sel: usize,
    /// Autocomplete candidates for the focused field; empty ⇒ no dropdown drawn.
    pub completions: &'a [RenameCompletion],
    /// One summary per complete rule (the before→after preview).
    pub rules: &'a [RenameRuleView],
    /// Total tensors renamed across all rules (for the header).
    pub total: usize,
    pub warnings: &'a [String],
    /// Whether `model.safetensors.index.json` will be updated too.
    pub has_index: bool,
    /// Whether the rename can be applied (every affected tensor clean).
    pub applicable: bool,
    /// Preview-pane scroll offset.
    pub scroll: usize,
    pub error: Option<&'a str>,
    /// The `convert --map …` CLI command equivalent to the entered renames (shown
    /// above the footer, copyable with `^Y`), or `None` until a rule is complete.
    pub cli: Option<&'a str>,
    /// What was just copied to the clipboard (e.g. `"the apply command"`), shown as
    /// a `✓ copied …` flash on the command row; `None` when nothing was just copied.
    pub copied: Option<&'a str>,
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
    Rename,
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
    /// A metadata-only / unavailable notice (e.g. a remote `--ssh-read` source has
    /// no local bytes for data views); holds the message to display.
    Notice(String),
}

/// Rows of chrome above the tree list: the title, the search/hint line, and the
/// separator rule.
const TREE_HEADER_HEIGHT: usize = 3;
/// Rows of chrome below the tree list: the two-line status bar.
/// Footer rows below the tree list: the two-line status bar. (The metadata-only
/// state is now a badge on that bar, not a separate banner row.)
const TREE_FOOTER_HEIGHT: usize = 2;

/// Footer rows below the file-browser list: a one-line status bar (the selected
/// entry's path / size, or a copy confirmation).
const FILES_FOOTER_HEIGHT: usize = 1;

/// Header rows above the layout map's strip: the title, the size / tensor-count
/// summary, and the separator rule.
const LAYOUT_HEADER_ROWS: usize = 3;

/// Where the tree's vertical scroll bar sits and how a pointer over it maps to a
/// scroll offset. Built by [`UI::tree_scrollbar`]; consumed by the renderer (to
/// draw the bar) and the mouse handler (to scrub on click / drag).
pub struct TreeScrollbar {
    /// Rightmost terminal column, reserved for the bar.
    pub col: u16,
    /// First body row (the terminal row just below the header).
    pub top: u16,
    /// Track height in rows — the number of visible tree rows.
    pub rows: u16,
    /// The largest valid scroll offset (`total - visible`).
    pub max_offset: usize,
}

impl TreeScrollbar {
    /// The scroll offset a pointer at terminal `row` scrubs to: the top of the
    /// track maps to offset 0 and the bottom to `max_offset`, proportionally
    /// (rows above/below the track clamp to the ends).
    pub fn offset_at(&self, row: u16) -> usize {
        if self.rows <= 1 {
            return 0;
        }
        let rel = row.saturating_sub(self.top).min(self.rows - 1);
        let frac = f64::from(rel) / f64::from(self.rows - 1);
        (frac * self.max_offset as f64).round() as usize
    }

    /// Whether the terminal cell `(col, row)` lands on the scroll bar.
    pub fn hit(&self, col: u16, row: u16) -> bool {
        col == self.col && row >= self.top && row < self.top + self.rows
    }
}

pub struct UI;

/// What the bottom-right access badge advertises about the currently open
/// checkpoint: whether the tool can rewrite it in place. Only a **local
/// safetensors** checkpoint is [`Editable`](AccessBadge::Editable) — the in-place
/// rename (`convert --map` / the `R` action) is the one path that modifies it;
/// everything else (a remote `--ssh-read` read, an HDF5 file, plain exports) is
/// [`ReadOnly`](AccessBadge::ReadOnly), and browsing never modifies it either way.
/// It is the rightmost [`Badge`] in the [`status bar`](UI::render_badge_bar).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AccessBadge {
    ReadOnly,
    Editable,
}

impl AccessBadge {
    /// The chip text, symmetrically padded with one space on each side.
    const fn label(self) -> &'static str {
        match self {
            AccessBadge::ReadOnly => " read-only ",
            AccessBadge::Editable => " editable ",
        }
    }

    /// The chip foreground: reassuring green when read-only, attention-drawing
    /// amber when the checkpoint can be rewritten in place.
    fn color(self) -> Color {
        match self {
            AccessBadge::ReadOnly => palette::SUCCESS,
            AccessBadge::Editable => palette::WARN,
        }
    }

    /// The hover-bubble text explaining what the badge means.
    fn hover(self) -> &'static str {
        match self {
            AccessBadge::ReadOnly => {
                "The checkpoint you open is never modified — browsing and exports only \
                 ever read it. (Repack / convert write a new file, leaving the original \
                 untouched.)"
            }
            AccessBadge::Editable => {
                "Browsing and exports never modify this checkpoint. The one exception is \
                 the in-place rename (R / convert --map), which rewrites the headers \
                 after you confirm."
            }
        }
    }
}

const HEALTH_BADGE: &str = " ⚠ health ";
const METADATA_BADGE: &str = " metadata-only ";

/// Default-background columns left between adjacent badges (and between the
/// leftmost badge and the status text), so the `STATUS_BG` chips read as separate
/// badges rather than one bar.
const BADGE_GAP: u16 = 2;

/// One chip in the bottom-right **status bar** — the uniform model behind the
/// access / health / metadata-only badges. Built by [`UI::status_badges`] and laid
/// out / drawn / hit-tested by the `badge_bar_*` functions, so every badge shares
/// one path for width, gap, colour, hover bubble and click action (they used to be
/// three hand-threaded implementations that kept drifting).
#[derive(Clone, Copy)]
pub struct Badge {
    /// Chip text, already padded (`" read-only "`, `" ⚠ health "`, …); also the
    /// hover bubble's title.
    label: &'static str,
    fg: Color,
    bg: Color,
    /// The hover bubble's border / title colour.
    accent: Color,
    /// The hover bubble body (word-wrapped by [`render_hover_bubble`]).
    help: &'static str,
    /// The key a click on this badge synthesizes (e.g. `h` opens the health
    /// report), or `None` if the badge isn't actionable.
    action: Option<char>,
}

impl Badge {
    /// The chip's display width (the `⚠` glyph is wide).
    fn width(self) -> u16 {
        use unicode_width::UnicodeWidthStr;
        self.label.width() as u16
    }

    /// The key a click on this badge acts as, if any.
    pub fn action(self) -> Option<char> {
        self.action
    }
}

/// Build the bottom-right status badges in **right-to-left** order (index 0 is the
/// rightmost, hugging the edge): the access badge always, then the health badge
/// when the index/file check flagged something, then the metadata-only badge for a
/// remote source. This is the single source of truth both the renderer and the
/// hover / click hit-test build from, so they can't disagree.
pub fn status_badges(
    access: AccessBadge,
    health: Option<HealthAlert>,
    metadata_only: bool,
) -> Vec<Badge> {
    let mut badges = vec![Badge {
        label: access.label(),
        fg: access.color(),
        bg: palette::STATUS_BG,
        accent: access.color(),
        help: access.hover(),
        action: None,
    }];
    if let Some(alert) = health {
        let (bg, help) = match alert {
            HealthAlert::Error => (
                palette::ALERT,
                "Index / file mismatch — files or tensors the index references are \
                 missing on disk, so the checkpoint may be incomplete. Click (or press \
                 h) for the health report.",
            ),
            HealthAlert::Warning => (
                palette::WARN_BG,
                "Index / file mismatch (warnings only) — e.g. files on disk the index \
                 doesn't reference. Click (or press h) for the health report.",
            ),
        };
        badges.push(Badge {
            label: HEALTH_BADGE,
            fg: palette::STATUS_FG,
            bg,
            accent: bg,
            help,
            action: Some('h'),
        });
    }
    if metadata_only {
        badges.push(Badge {
            label: METADATA_BADGE,
            fg: palette::WARN,
            bg: palette::STATUS_BG,
            accent: palette::WARN,
            help: "A remote source: only header metadata is loaded, so the data views \
                   (heatmap / grid / histogram / statistics) need the file locally.",
            action: None,
        });
    }
    badges
}

/// The on-screen rect of each badge on the bottom row — right-aligned, index 0 at
/// the edge, each `BADGE_GAP` apart. `None` for a badge that doesn't fit the frame.
/// The one geometry the renderer, hit-test and reserve all share.
fn badge_rects(width: u16, height: u16, badges: &[Badge]) -> Vec<Option<Rect>> {
    let mut rects = Vec::with_capacity(badges.len());
    let mut right = 0u16; // columns already spoken for to the right (incl. gaps)
    for b in badges {
        let w = b.width();
        let rect = (height > 0 && width > right + w).then(|| Rect {
            x: width - right - w,
            y: height - 1,
            width: w,
            height: 1,
        });
        rects.push(rect);
        right += w + BADGE_GAP;
    }
    rects
}

impl UI {
    /// Draw the bottom-right **status bar** — every badge in `badges` (from
    /// [`status_badges`]) right-aligned on the last row, and, when `hovered` is
    /// `Some(i)`, that badge's hover bubble floated above it. Rendered last on a
    /// view so the chips sit over whatever occupies that row.
    pub fn render_badge_bar(frame: &mut Frame, badges: &[Badge], hovered: Option<usize>) {
        let area = frame.area();
        let rects = badge_rects(area.width, area.height, badges);
        for (b, rect) in badges.iter().zip(&rects) {
            if let Some(r) = rect {
                Paragraph::new(Line::from(Span::styled(
                    b.label,
                    Style::default()
                        .bg(b.bg)
                        .fg(b.fg)
                        .add_modifier(Modifier::BOLD),
                )))
                .render(*r, frame.buffer_mut());
            }
        }
        // The hover bubble goes last so it floats over the neighbouring chips.
        if let Some(i) = hovered
            && let (Some(b), Some(Some(r))) = (badges.get(i), rects.get(i))
        {
            render_hover_bubble(frame, *r, b.accent, Some(b.label), b.help);
        }
    }

    /// The index of the badge under `(col, row)`, if any — for the hover bubble and
    /// click actions. Shares [`badge_rects`] with the renderer, so they can't drift.
    pub fn badge_bar_hit(
        width: u16,
        height: u16,
        col: u16,
        row: u16,
        badges: &[Badge],
    ) -> Option<usize> {
        badge_rects(width, height, badges)
            .into_iter()
            .position(|r| r.is_some_and(|r| row == r.y && col >= r.x && col < r.x + r.width))
    }

    /// Columns the badge bar reserves on the right of the status line, so the
    /// status text never runs under it (a [`BADGE_GAP`] before each badge).
    pub fn badge_bar_width(badges: &[Badge]) -> u16 {
        badges.iter().map(|b| b.width() + BADGE_GAP).sum()
    }

    /// How many tree rows are visible at once (one screenful), used to size a
    /// PageUp/PageDown jump. `terminal_height` is the full terminal height.
    pub fn visible_tree_rows(terminal_height: u16) -> usize {
        (terminal_height as usize)
            .saturating_sub(TREE_HEADER_HEIGHT + TREE_FOOTER_HEIGHT)
            .max(1)
    }

    /// Rows the tree's bottom-pinned key-hint footer occupies (0 while searching —
    /// the search bar rides the header instead). Kept in sync with
    /// [`Self::render_tree`] so scroll / hit-testing align.
    pub fn tree_hint_rows(
        width: u16,
        search_mode: bool,
        can_repack: bool,
        can_rename: bool,
    ) -> usize {
        if search_mode {
            0
        } else {
            tree_hint_lines(can_repack, can_rename, width).0.len()
        }
    }

    /// Body rows visible in the tree at the given size — used to compute the
    /// scroll offset so it stays consistent with [`Self::render_tree`]'s layout
    /// (header = title + optional search line + rule; a bottom-pinned hint footer;
    /// then the two status lines).
    pub fn tree_visible_rows(
        width: u16,
        height: u16,
        search_mode: bool,
        can_repack: bool,
        can_rename: bool,
    ) -> usize {
        let header = Self::tree_header_rows(search_mode);
        let hints = Self::tree_hint_rows(width, search_mode, can_repack, can_rename);
        (height as usize)
            .saturating_sub(header + hints + TREE_FOOTER_HEIGHT)
            .max(1)
    }

    /// The first terminal row of the tree body — the header height (title + the
    /// search line while searching + rule; the key hints are a bottom footer now).
    /// Used for mouse hit-testing: a click at row `r >= tree_header_rows()` and above
    /// the hint footer lands on tree row `scroll_offset + (r - tree_header_rows())`.
    pub fn tree_header_rows(search_mode: bool) -> usize {
        if search_mode { 3 } else { 2 } // title + [search] + rule
    }

    /// Geometry of the tree's vertical scroll bar for this terminal size and a
    /// tree of `total` rows, or `None` when the whole tree fits the viewport (so
    /// no bar is drawn and no column reserved). Shared by [`Self::render_tree`]
    /// and the mouse handler, so click / drag hit-testing lines up with what's
    /// drawn. The bar rides the rightmost column of the body region.
    pub fn tree_scrollbar(
        width: u16,
        height: u16,
        search_mode: bool,
        can_repack: bool,
        can_rename: bool,
        total: usize,
    ) -> Option<TreeScrollbar> {
        let rows = Self::tree_visible_rows(width, height, search_mode, can_repack, can_rename);
        if width < 2 || total <= rows {
            return None; // nothing to scroll (or no room for a bar + content)
        }
        Some(TreeScrollbar {
            col: width - 1,
            top: Self::tree_header_rows(search_mode) as u16,
            rows: rows as u16,
            max_offset: total - rows,
        })
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

        // --- header + tree rows (the region above the 2-line status bar) ---
        let mut lines: Vec<Line> = Vec::new();

        // Title. (A health-check warning is surfaced on the status bar instead —
        // see the `⚠ health` alert beside the read-only badge below.)
        let title = vec![Span::raw(format!(
            "Checkpoint Explorer - {} ({}/{})",
            config.current_file,
            config.file_idx + 1,
            config.total_files
        ))];
        lines.push(Line::from(title));

        // The search bar rides the header while searching; the key hints are a
        // bottom-pinned footer (built below), so they don't push the tree down.
        if config.search_mode {
            lines.push(tree_search_line(config));
        }

        // Separator rule.
        lines.push(Line::from(Span::styled(
            "─".repeat(width as usize),
            Style::default().fg(palette::DIM),
        )));

        // The bottom hint footer (absent while searching — the search bar is the
        // input, and the status bar spells out Esc/Enter).
        let (hint_lines, chips) = if config.search_mode {
            (Vec::new(), Vec::new())
        } else {
            tree_hint_lines(config.can_repack, config.can_rename, width)
        };
        let hint_rows = hint_lines.len();

        let header_rows = lines.len();
        let footer_rows = TREE_FOOTER_HEIGHT;
        let body_rows = (height as usize).saturating_sub(header_rows + hint_rows + footer_rows);

        // A vertical scroll bar rides the rightmost column when the tree
        // overflows the viewport — but only in the live TUI; a headless
        // `--plain` / screen-copy render is a static dump with no viewport.
        let scrollbar = if config.interactive {
            Self::tree_scrollbar(
                width,
                height,
                config.search_mode,
                config.can_repack,
                config.can_rename,
                config.tree.len(),
            )
        } else {
            None
        };
        // Reserve the bar's column so long tree rows don't underlap it.
        let body_width = width.saturating_sub(if scrollbar.is_some() { 1 } else { 0 });

        // Header (title, hint(s), rule) spans the full width.
        Paragraph::new(lines).render(
            Rect {
                x: 0,
                y: 0,
                width,
                height: header_rows as u16,
            },
            frame.buffer_mut(),
        );

        // Visible tree rows from the (pre-computed) scroll offset, clipped to
        // `body_width` so the reserved scroll-bar column stays clear.
        if !(config.search_mode && config.tree.is_empty()) {
            let mut body: Vec<Line> = Vec::with_capacity(body_rows);
            for (idx, (node, depth)) in config
                .tree
                .iter()
                .enumerate()
                .skip(config.scroll_offset)
                .take(body_rows)
            {
                let selected = idx == config.selected_idx;
                body.push(tree_node_line(
                    node,
                    *depth,
                    selected,
                    config.unindexed,
                    config.packing_schemas,
                    MetaDisplay::Capped, // live tree keeps rows short
                ));
            }
            Paragraph::new(body).render(
                Rect {
                    x: 0,
                    y: header_rows as u16,
                    width: body_width,
                    height: body_rows as u16,
                },
                frame.buffer_mut(),
            );
        }

        // The scroll bar itself, over its reserved column. `content_length` is
        // the number of scroll positions (`max_offset + 1`) so the thumb reaches
        // the very bottom exactly when scrolled to the last row.
        if let Some(sb) = &scrollbar {
            let mut state = ScrollbarState::new(sb.max_offset + 1)
                .position(config.scroll_offset)
                .viewport_content_length(body_rows);
            StatefulWidget::render(
                Scrollbar::new(ScrollbarOrientation::VerticalRight)
                    .begin_symbol(None)
                    .end_symbol(None)
                    .track_symbol(Some("│"))
                    .thumb_symbol("█")
                    .track_style(Style::default().fg(palette::DIM))
                    .thumb_style(Style::default().fg(palette::ACCENT)),
                Rect {
                    x: sb.col,
                    y: sb.top,
                    width: 1,
                    height: sb.rows,
                },
                frame.buffer_mut(),
                &mut state,
            );
        }

        // --- key-hint footer, pinned just above the two-line status bar ---
        let hint_y = height.saturating_sub(TREE_FOOTER_HEIGHT as u16 + hint_rows as u16);
        if hint_rows > 0 {
            Paragraph::new(hint_lines).render(
                Rect {
                    x: 0,
                    y: hint_y,
                    width,
                    height: hint_rows as u16,
                },
                frame.buffer_mut(),
            );
        }

        // --- bottom two-line status bar ---
        // Reserve room on the right of the bottom status line for the persistent
        // badges drawn there (access, and any health / metadata-only badge), so the
        // status text never runs under them.
        let reserve = Self::badge_bar_width(config.badges) as usize;
        let max_text = (width as usize).saturating_sub(6 + reserve);
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

        // Second line: a transient copy confirmation (green, shown verbatim)
        // overrides the dimmed source file.
        let row1 = if let Some(flash) = config.copied_flash {
            Line::from(Span::styled(
                flash.to_string(),
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

        // Right-aligned status badges (access, and any health / metadata-only), with
        // the hovered one's bubble — all through the one uniform bar.
        Self::render_badge_bar(frame, config.badges, config.hovered_badge);

        // Clickable regions: each footer chip (the bottom-pinned hint block, at
        // `hint_y`) plus the top-right `[×]` (→ quit the tree).
        let mut regions = chip_regions(&chips, hint_y);
        regions.extend(close_button(frame, hint_key('q')));
        regions
    }

    /// The first terminal row of the file browser's body — its header height
    /// (title + separator rule; the key hints are a bottom-pinned footer now).
    /// Shared with the mouse handler so a click at row `r >= this` maps to file row
    /// `scroll + (r - this)`.
    pub fn files_header_rows(_width: u16) -> usize {
        2 // title + rule
    }

    /// Rows the bottom-pinned key-hint footer occupies (above the one-line status
    /// bar). Kept in sync with [`Self::render_files`] so scroll / hit-testing align.
    pub fn files_hint_rows(width: u16) -> usize {
        files_hint_lines(width).0.len()
    }

    /// Body rows visible in the file browser at the given size (header + the
    /// bottom-pinned hint footer + the one-line status bar), so the scroll offset
    /// stays consistent with [`Self::render_files`]'s layout.
    pub fn files_visible_rows(width: u16, height: u16) -> usize {
        (height as usize)
            .saturating_sub(
                Self::files_header_rows(width) + Self::files_hint_rows(width) + FILES_FOOTER_HEIGHT,
            )
            .max(1)
    }

    /// How many file rows fit one screenful — used to size a PageUp/PageDown jump.
    pub fn visible_file_rows(width: u16, height: u16) -> usize {
        Self::files_visible_rows(width, height)
    }

    /// The file browser's scroll-bar geometry (reusing [`TreeScrollbar`]) for this
    /// size and a listing of `total` rows, or `None` when it all fits.
    pub fn files_scrollbar(width: u16, height: u16, total: usize) -> Option<TreeScrollbar> {
        let rows = Self::files_visible_rows(width, height);
        if width < 2 || total <= rows {
            return None;
        }
        Some(TreeScrollbar {
            col: width - 1,
            top: Self::files_header_rows(width) as u16,
            rows: rows as u16,
            max_offset: total - rows,
        })
    }

    /// Render the file browser: header (title, hint line(s), rule), the visible
    /// file rows from `scroll`, a scroll bar when the listing overflows, and a
    /// one-line status bar showing the selected entry's path (or a copy
    /// confirmation). Returns the clickable footer chips + `[×]` close, like
    /// [`Self::render_tree`].
    // A flat render signature (frame + view state) — a config struct would just
    // move the same fields behind one more indirection for no clarity.
    #[allow(clippy::too_many_arguments)]
    pub fn render_files(
        frame: &mut Frame,
        root: &str,
        rows: &[crate::filetree::FileRow],
        selected: usize,
        scroll: usize,
        copied_flash: Option<&str>,
        interactive: bool,
        badges: &[Badge],
        hovered_badge: Option<usize>,
    ) -> Vec<(Rect, KeyEvent)> {
        let area = frame.area();
        let (width, height) = (area.width, area.height);
        if height < (FILES_FOOTER_HEIGHT as u16 + 1) {
            return Vec::new();
        }

        // --- header (title + rule); the key hints are a bottom-pinned footer ---
        let lines: Vec<Line> = vec![
            Line::from(Span::raw(format!("File browser - {root}"))),
            Line::from(Span::styled(
                "─".repeat(width as usize),
                Style::default().fg(palette::DIM),
            )),
        ];
        let (hint_lines, chips) = files_hint_lines(width);
        let hint_rows = hint_lines.len();

        let header_rows = lines.len();
        let body_rows =
            (height as usize).saturating_sub(header_rows + hint_rows + FILES_FOOTER_HEIGHT);

        let scrollbar = if interactive {
            Self::files_scrollbar(width, height, rows.len())
        } else {
            None
        };
        let body_width = width.saturating_sub(u16::from(scrollbar.is_some()));

        Paragraph::new(lines).render(
            Rect {
                x: 0,
                y: 0,
                width,
                height: header_rows as u16,
            },
            frame.buffer_mut(),
        );

        let mut body: Vec<Line> = Vec::with_capacity(body_rows);
        for (idx, row) in rows.iter().enumerate().skip(scroll).take(body_rows) {
            body.push(file_row_line(row, idx == selected));
        }
        Paragraph::new(body).render(
            Rect {
                x: 0,
                y: header_rows as u16,
                width: body_width,
                height: body_rows as u16,
            },
            frame.buffer_mut(),
        );

        if let Some(sb) = &scrollbar {
            let mut state = ScrollbarState::new(sb.max_offset + 1)
                .position(scroll)
                .viewport_content_length(body_rows);
            StatefulWidget::render(
                Scrollbar::new(ScrollbarOrientation::VerticalRight)
                    .begin_symbol(None)
                    .end_symbol(None)
                    .track_symbol(Some("│"))
                    .thumb_symbol("█")
                    .track_style(Style::default().fg(palette::DIM))
                    .thumb_style(Style::default().fg(palette::ACCENT)),
                Rect {
                    x: sb.col,
                    y: sb.top,
                    width: 1,
                    height: sb.rows,
                },
                frame.buffer_mut(),
                &mut state,
            );
        }

        // --- key-hint footer, pinned just above the one-line status bar ---
        let hint_y = height.saturating_sub(1 + hint_rows as u16);
        Paragraph::new(hint_lines).render(
            Rect {
                x: 0,
                y: hint_y,
                width,
                height: hint_rows as u16,
            },
            frame.buffer_mut(),
        );

        // --- one-line status bar (selected entry, or a copy confirmation) ---
        let reserve = Self::badge_bar_width(badges) as usize;
        let max_text = (width as usize).saturating_sub(6 + reserve);
        let status = if let Some(flash) = copied_flash {
            Line::from(Span::styled(
                flash.to_string(),
                Style::default()
                    .fg(palette::SUCCESS)
                    .add_modifier(Modifier::BOLD),
            ))
        } else if let Some(row) = rows.get(selected) {
            let text = truncate_keep_end(&row.path.to_string_lossy(), max_text);
            Line::from(Span::styled(
                format!(" ▪ {text} "),
                Style::default()
                    .bg(palette::STATUS_BG)
                    .fg(palette::STATUS_FG),
            ))
        } else {
            Line::default()
        };
        Paragraph::new(status).render(
            Rect {
                x: 0,
                y: height.saturating_sub(1),
                width,
                height: 1,
            },
            frame.buffer_mut(),
        );
        Self::render_badge_bar(frame, badges, hovered_badge);

        // Clickable footer chips (the bottom-pinned hint block, at `hint_y`)
        // plus the top-right `[×]` (→ switch back to the tensor tree).
        let mut regions = chip_regions(&chips, hint_y);
        regions.extend(close_button(
            frame,
            KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
        ));
        regions
    }

    /// Draw the in-place rename editor ([`Screen::Rename`](crate::explorer)): a
    /// title + rule header (the same borderless chrome as the tree / detail /
    /// layout views — it's a first-class view, not a pop-up dialog), the dynamic
    /// list of source→new-name rule pairs (with the focused field's autocomplete),
    /// a live before→after diff preview marking each tensor OK / collides /
    /// won't-fit, and the common footer / confirm bar. Returns the preview pane's
    /// max scroll offset, the footer chip regions (clickable, like the other
    /// views), and the preview's nav-link regions.
    pub fn render_rename(
        frame: &mut Frame,
        view: &RenameView,
    ) -> (usize, ChipRegions, LinkRegions, Vec<Rect>) {
        let area = frame.area();
        let (width, height) = (area.width, area.height);
        if height < 7 || width < 12 {
            return (0, Vec::new(), Vec::new(), Vec::new());
        }

        // Header: a title line then a full-width rule, matching the other views'
        // chrome (no surrounding border, no panel fill).
        let header = vec![
            Line::from(Span::styled(
                format!("Rename tensors in place — {}", view.root),
                Style::default()
                    .fg(palette::ACCENT)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                "─".repeat(width as usize),
                Style::default().fg(palette::DIM),
            )),
        ];
        let header_h = header.len() as u16;
        Paragraph::new(header).render(
            Rect {
                x: 0,
                y: 0,
                width,
                height: header_h,
            },
            frame.buffer_mut(),
        );

        // One field row: `label  value` — the focused field shows the caret via an
        // input box; others show their value plainly (not dimmed).
        let field_row = |label: &str, value: &str, focused: bool| -> Line<'static> {
            let mut spans = vec![Span::styled(
                format!("  {label:<4} "),
                Style::default().fg(palette::KEY),
            )];
            if focused {
                spans.extend(input_box_spans(value, view.cursor, 0));
            } else {
                spans.push(Span::raw(value.to_string()));
            }
            Line::from(spans)
        };

        // --- rule-pair editor lines ---
        // The focused field's row index within `editor` — the autocomplete dropdown
        // floats just beneath it (resolved to an absolute row once `editor` is laid
        // out). The dropdown itself is drawn last, over the content below.
        let mut editor: Vec<Line<'static>> = Vec::new();
        let mut focus_line = 0usize;
        for (i, (src, tgt)) in view.pairs.iter().enumerate() {
            if view.pairs.len() > 1 {
                editor.push(Line::from(Span::styled(
                    format!("rule {}", i + 1),
                    Style::default()
                        .fg(palette::DTYPE)
                        .add_modifier(Modifier::BOLD),
                )));
            }
            let focused_src = i == view.focus_pair && !view.on_target;
            let focused_tgt = i == view.focus_pair && view.on_target;
            if focused_src {
                focus_line = editor.len();
            }
            editor.push(field_row("from", src, focused_src));
            if focused_tgt {
                focus_line = editor.len();
            }
            editor.push(field_row("to", tgt, focused_tgt));
            editor.push(Line::from(Span::raw("")));
        }

        // --- compact, per-rule before → after preview ---
        let mut preview: Vec<Line<'static>> = Vec::new();
        let mut summary = format!(
            "Preview — {} tensor(s) across {} rule(s)",
            view.total,
            view.rules.len()
        );
        // Only claim the index will change when a rule actually matches something.
        if view.has_index && view.total > 0 {
            summary.push_str(" · updates index.json");
        }
        preview.push(Line::from(Span::styled(
            summary,
            Style::default()
                .fg(palette::KEY)
                .add_modifier(Modifier::BOLD),
        )));
        if view.rules.is_empty() {
            preview.push(Line::from(dim_span(
                "  autocomplete a source and edit its new name to preview the changes",
            )));
        }
        // Clickable links in the preview: (preview line index, x within inner,
        // width, target) — resolved to screen Rects once scroll is known. Shard
        // names open the layout view; a *concrete* source tensor opens the tree.
        let mut hits: Vec<(usize, u16, u16, Link)> = Vec::new();
        for (i, rule) in view.rules.iter().enumerate() {
            // A one-line status per rule (coloured by the worst outcome), then the
            // before → after schema on their own lines so nothing is truncated. The
            // count reflects tensors *changed*, except a matched-but-unchanged rule
            // (a just-autocompleted source whose new name is still identical), which
            // reports how many it *matches* so it doesn't read as "matches nothing".
            let (count, label, color) = if rule.collide > 0 || rule.invalid > 0 {
                let mut parts = Vec::new();
                if rule.collide > 0 {
                    parts.push(format!("{} collide", rule.collide));
                }
                if rule.invalid > 0 {
                    parts.push(format!("{} invalid target", rule.invalid));
                }
                (
                    rule.total,
                    format!("⚠ {}", parts.join(", ")),
                    palette::ERROR,
                )
            } else if rule.wont_fit > 0 {
                (
                    rule.total,
                    format!("⚠ {} won't fit in place", rule.wont_fit),
                    palette::WARN,
                )
            } else if rule.total == 0 {
                if rule.matched > 0 {
                    (
                        rule.matched,
                        "new name unchanged — edit the “to” field".to_string(),
                        palette::WARN,
                    )
                } else {
                    (0, "matches no tensors".to_string(), palette::DIM)
                }
            } else {
                (
                    rule.total,
                    "✓ applies cleanly".to_string(),
                    palette::SUCCESS,
                )
            };
            preview.push(Line::default());
            preview.push(Line::from(vec![
                Span::styled(
                    format!("rule {} · {} tensor(s) · ", i + 1, count),
                    Style::default().fg(palette::DTYPE),
                ),
                Span::styled(label, Style::default().fg(color)),
            ]));
            // A concrete source (no `{…}` placeholder) is one real tensor, so it's a
            // link to the tree; a schema source matches many, so it stays plain.
            if rule.from.contains('{') {
                preview.push(Line::from(Span::raw(format!("    {}", rule.from))));
            } else {
                hits.push((
                    preview.len(),
                    4, // "    " indent
                    rule.from.chars().count() as u16,
                    Link::Tree(rule.from.clone()),
                ));
                preview.push(Line::from(vec![
                    Span::raw("    "),
                    Span::styled(
                        rule.from.clone(),
                        Style::default()
                            .fg(palette::ACCENT)
                            .add_modifier(Modifier::UNDERLINED),
                    ),
                ]));
            }
            preview.push(Line::from(vec![
                Span::styled("  → ", Style::default().fg(palette::DIM)),
                Span::styled(rule.to.clone(), Style::default().fg(color)),
            ]));
            // Per-shard header sizing — which file fits in place and by how much.
            for sf in &rule.shards {
                let (mark, note, c) = if sf.fits() {
                    ("✓", format!("{} B to spare", sf.spare()), palette::SUCCESS)
                } else {
                    ("✗", format!("{} B over", sf.over()), palette::WARN)
                };
                // The filename is a link to that shard's layout view; record its
                // region. `"      {mark} "` = 6 spaces + mark + space = 8 columns.
                hits.push((
                    preview.len(),
                    8,
                    sf.file.chars().count() as u16,
                    Link::Layout(sf.path.clone()),
                ));
                preview.push(Line::from(vec![
                    Span::styled(format!("      {mark} "), Style::default().fg(palette::DIM)),
                    Span::styled(
                        sf.file.clone(),
                        Style::default()
                            .fg(palette::ACCENT)
                            .add_modifier(Modifier::UNDERLINED),
                    ),
                    Span::styled(
                        format!(
                            "  header {} B → {} B  ({note}, {} tensor(s))",
                            sf.current, sf.needed, sf.tensors
                        ),
                        Style::default().fg(c),
                    ),
                ]));
            }
        }
        if !view.warnings.is_empty() {
            preview.push(Line::default());
        }
        for w in view.warnings {
            preview.push(Line::from(Span::styled(
                format!("note: {w}"),
                Style::default().fg(palette::WARN),
            )));
        }

        // --- footer: the common clickable chip hint, or the error bar ---
        // Build it first so its (possibly wrapped) height reserves the bottom rows,
        // like the layout/file views size their footers. (Apply confirmation is a
        // floating modal now, not an inline footer bar.)
        let (footer_lines, chip_hits): (Vec<Line<'static>>, Vec<ChipHit>) =
            if let Some(err) = view.error {
                (
                    vec![Line::from(Span::styled(
                        format!("⚠ {err}"),
                        Style::default().fg(palette::ERROR),
                    ))],
                    Vec::new(),
                )
            } else {
                rename_hint_lines(width, view.applicable)
            };
        let footer_h = (footer_lines.len() as u16).max(1);
        let footer_top = height.saturating_sub(footer_h);

        // --- lay out: header, editor, preview (scroll), command row, footer ---
        // The apply-command row sits just above the footer when there's room.
        let cmd_y = footer_top.checked_sub(1).filter(|y| *y > header_h + 1);
        let editor_h = (editor.len() as u16).min(height.saturating_sub(header_h + footer_h + 2));
        Paragraph::new(editor).render(
            Rect {
                x: 0,
                y: header_h,
                width,
                height: editor_h,
            },
            frame.buffer_mut(),
        );

        let sep_y = header_h + editor_h;
        let preview_bottom = cmd_y.unwrap_or(footer_top);
        if sep_y < preview_bottom {
            Paragraph::new(Line::from(dim_span("─".repeat(width as usize)))).render(
                Rect {
                    x: 0,
                    y: sep_y,
                    width,
                    height: 1,
                },
                frame.buffer_mut(),
            );
        }
        let preview_y = sep_y + 1;
        let preview_h = preview_bottom.saturating_sub(preview_y);
        let visible = preview_h as usize;
        let max_scroll = preview.len().saturating_sub(visible);
        let scroll = view.scroll.min(max_scroll);
        let window: Vec<Line> = preview.iter().skip(scroll).take(visible).cloned().collect();
        Paragraph::new(window).render(
            Rect {
                x: 0,
                y: preview_y,
                width,
                height: preview_h,
            },
            frame.buffer_mut(),
        );
        // Map the visible link hits to on-screen Rects (target per region).
        let clicks: Vec<(Rect, Link)> = hits
            .into_iter()
            .filter(|(idx, ..)| *idx >= scroll && *idx < scroll + visible)
            .map(|(idx, x, w, target)| {
                (
                    Rect {
                        x,
                        y: preview_y + (idx - scroll) as u16,
                        width: w,
                        height: 1,
                    },
                    target,
                )
            })
            .collect();

        // The equivalent apply command (copyable with ^Y), just above the footer.
        if let Some(y) = cmd_y {
            let cmd_line = if let Some(what) = view.copied {
                Line::from(Span::styled(
                    format!("✓ copied {what} to the clipboard"),
                    Style::default()
                        .fg(palette::SUCCESS)
                        .add_modifier(Modifier::BOLD),
                ))
            } else if let Some(cmd) = view.cli {
                Line::from(vec![
                    Span::styled("apply: ", Style::default().fg(palette::DIM)),
                    Span::styled(cmd.to_string(), Style::default().fg(palette::META)),
                    Span::styled("   (^A copy)", Style::default().fg(palette::DIM)),
                ])
            } else {
                Line::from(dim_span(
                    "enter a rename above to get the `convert --map` command that applies it",
                ))
            };
            Paragraph::new(cmd_line).render(
                Rect {
                    x: 0,
                    y,
                    width,
                    height: 1,
                },
                frame.buffer_mut(),
            );
        }

        Paragraph::new(footer_lines).render(
            Rect {
                x: 0,
                y: footer_top,
                width,
                height: footer_h,
            },
            frame.buffer_mut(),
        );
        // Footer chips → absolute clickable regions (each replays its key).
        let chips = chip_regions(&chip_hits, footer_top);

        // The autocomplete dropdown floats over everything, anchored just below the
        // focused field (when it's on-screen) — drawn last so nothing overpaints it.
        let mut menu_rects = Vec::new();
        if view.menu_open && !view.completions.is_empty() && focus_line < editor_h as usize {
            menu_rects = render_completion_menu(
                frame,
                RENAME_MENU_X,
                header_h + focus_line as u16,
                view.completions,
                view.menu_sel,
            );
        }

        (max_scroll, chips, clicks, menu_rects)
    }

    /// Float a sidecar file preview over the file browser: a scrollable pop-up of
    /// the file's contents (JSON syntax-highlighted, other text plain) or an info
    /// line for a binary. Reuses the scroll-pop-up chrome; returns its max scroll
    /// and clickable regions so the caller can clamp/handle them.
    pub fn render_file_preview(
        frame: &mut Frame,
        title: &str,
        body: &[Line<'static>],
        footer: Line<'static>,
        scroll: usize,
    ) -> (usize, Vec<(Rect, KeyEvent)>) {
        render_scroll_popup(frame, title, body, footer, scroll, &[])
    }

    /// The first terminal row of the layout map's strip (its fixed header height),
    /// for the mouse click-to-select hit-test.
    pub fn layout_header_rows() -> usize {
        LAYOUT_HEADER_ROWS
    }

    /// Body rows the layout map's vertical strip occupies (total height minus the
    /// 3-row header and the footer hint line(s)).
    pub fn layout_visible_rows(width: u16, height: u16) -> usize {
        (height as usize)
            .saturating_sub(LAYOUT_HEADER_ROWS + layout_hint_lines(width).0.len())
            .max(1)
    }

    /// Render the safetensors **layout map** — a scrollable vertical strip of the
    /// file: a header (title + size / tensor-count / header-size summary), then one
    /// band per segment (header, each tensor by offset, any padding) whose height
    /// is proportional to its share of the file. Each band's first row carries its
    /// offset and a one-line label (name + dtype/shape + size); the header band's
    /// remaining rows list its `__metadata__` entries tree-like. The `selected`
    /// segment's label row is highlighted. Returns the max scroll offset (so the
    /// caller can clamp) and the clickable footer chips.
    pub fn render_layout(
        frame: &mut Frame,
        map: &crate::safelayout::LayoutMap,
        selected: usize,
        scroll: usize,
        copied: Option<&str>,
        interactive: bool,
    ) -> (usize, ChipRegions, LinkRegions) {
        use crate::safelayout::SegmentKind;
        let area = frame.area();
        let (width, height) = (area.width, area.height);
        if height < (LAYOUT_HEADER_ROWS as u16 + 2) {
            return (0, Vec::new(), Vec::new());
        }
        // A concrete tensor band's name links to that tensor in the tree; filled in
        // as the strip is drawn below.
        let mut links: Vec<(Rect, Link)> = Vec::new();

        // --- header (title, summary, rule) ---
        let dim = Style::default().fg(palette::DIM);
        let mut summary = vec![
            Span::styled(format_size(map.total_len as usize), Style::default()),
            Span::styled(" · ", dim),
            Span::raw(format!("{} tensors", map.tensor_count)),
            Span::styled(" · ", dim),
            Span::raw(format!("header {}", format_size(map.header_len as usize))),
        ];
        if map.metadata_entries() > 0 {
            summary.push(Span::styled(
                format!(" · {} metadata", map.metadata_entries()),
                dim,
            ));
        }
        let header_lines = vec![
            Line::from(Span::raw(format!("Layout - {}", map.name))),
            Line::from(summary),
            Line::from(Span::styled(
                "─".repeat(width as usize),
                Style::default().fg(palette::DIM),
            )),
        ];
        Paragraph::new(header_lines).render(
            Rect {
                x: 0,
                y: 0,
                width,
                height: LAYOUT_HEADER_ROWS as u16,
            },
            frame.buffer_mut(),
        );

        // --- footer hints (pinned to the bottom) ---
        // A copy confirmation temporarily takes over the footer's first line (its
        // own line, cleared full-width) so it never intermingles with the hints.
        let (mut hint_lines, chips) = layout_hint_lines(width);
        let footer_rows = hint_lines.len();
        let body_rows = (height as usize).saturating_sub(LAYOUT_HEADER_ROWS + footer_rows);
        if let Some(msg) = copied
            && let Some(first) = hint_lines.first_mut()
        {
            *first = Line::from(Span::styled(
                msg.to_string(),
                Style::default()
                    .fg(palette::SUCCESS)
                    .add_modifier(Modifier::BOLD),
            ));
        }
        Paragraph::new(hint_lines).render(
            Rect {
                x: 0,
                y: (height as usize - footer_rows) as u16,
                width,
                height: footer_rows as u16,
            },
            frame.buffer_mut(),
        );

        // --- the vertical strip (scrollable) ---
        let rows = band_rows(map, body_rows);
        let total_rows: usize = rows.iter().sum();
        let scrollbar =
            interactive.then(|| Self::files_scrollbar_like(width, total_rows, body_rows));
        let scrollbar = scrollbar.flatten();
        let strip_width = width.saturating_sub(u16::from(scrollbar.is_some()));

        let max_scroll = total_rows.saturating_sub(body_rows);
        let scroll = scroll.min(max_scroll);
        // Cumulative start row of each band.
        let mut starts = Vec::with_capacity(rows.len() + 1);
        let mut acc = 0usize;
        for &h in &rows {
            starts.push(acc);
            acc += h;
        }

        let sel = Style::default()
            .fg(palette::SELECT_FG)
            .bg(palette::SELECT_BG);
        let mut seg = 0usize; // segment whose band contains the current row
        let mut body: Vec<Line> = Vec::with_capacity(body_rows);
        for r in scroll..(scroll + body_rows).min(total_rows) {
            // Advance to the band containing global row `r`.
            while seg + 1 < starts.len() && r >= starts[seg] + rows[seg] {
                seg += 1;
            }
            let s = &map.segments[seg];
            let row_in = r - starts[seg];
            let first = row_in == 0;
            let selected_row = seg == selected && first;
            let rule = if r == 0 {
                '┬'
            } else if r == total_rows - 1 {
                '┴'
            } else {
                '│'
            };
            let (glyph, color) = band_style(s, map.total_len);
            let off = if first {
                format!("{:#014x}", s.start)
            } else {
                " ".repeat(14)
            };
            let mut spans = vec![
                Span::styled(off, dim),
                Span::raw(" "),
                Span::styled(rule.to_string(), dim),
                Span::raw(" "),
                Span::styled(glyph.to_string(), Style::default().fg(color)),
                Span::raw("  "),
            ];
            let label_w = strip_width.saturating_sub(20) as usize;
            if first {
                // One-line label: name (selection-highlighted), then a dim
                // dtype/shape + size, so nothing looks orphaned on a blank row.
                let name = truncate_keep_end(&s.name, label_w.saturating_sub(24));
                // A concrete tensor's name is a link to the tree (underlined, like
                // the other in-app links); the spans above are a fixed 20 columns
                // wide, so the name always starts at column 20.
                let is_tensor = s.kind == SegmentKind::Tensor;
                let name_style = if selected_row {
                    sel
                } else if is_tensor {
                    Style::default()
                        .fg(color)
                        .add_modifier(Modifier::UNDERLINED)
                } else {
                    Style::default().fg(color)
                };
                if is_tensor {
                    links.push((
                        Rect {
                            x: 20,
                            y: LAYOUT_HEADER_ROWS as u16 + (r - scroll) as u16,
                            width: name.chars().count() as u16,
                            height: 1,
                        },
                        Link::Tree(s.name.clone()),
                    ));
                }
                spans.push(Span::styled(name, name_style));
                let mut detail = String::new();
                if let (SegmentKind::Tensor, Some(dt)) = (s.kind, &s.dtype) {
                    detail.push_str(&format!("  {dt}"));
                    if !s.shape.is_empty() {
                        detail.push_str(&format!(" {}", format_shape(&s.shape)));
                    }
                }
                detail.push_str(&format!("  {}", format_size(s.len() as usize)));
                spans.push(Span::styled(detail, if selected_row { sel } else { dim }));
            } else if s.kind == SegmentKind::Header {
                // The header band's rows list its `__metadata__` entries tree-like.
                if let Some((k, v)) = map.metadata.get(row_in - 1) {
                    let val = truncate_keep_end(v, label_w.saturating_sub(k.len() + 6));
                    spans.push(Span::styled(
                        format!("† {k}  "),
                        Style::default().fg(palette::META),
                    ));
                    spans.push(Span::styled(val, dim));
                }
            }
            body.push(Line::from(spans));
        }
        Paragraph::new(body).render(
            Rect {
                x: 0,
                y: LAYOUT_HEADER_ROWS as u16,
                width: strip_width,
                height: body_rows as u16,
            },
            frame.buffer_mut(),
        );

        if let Some(sb) = &scrollbar {
            let mut state = ScrollbarState::new(sb.max_offset + 1)
                .position(scroll)
                .viewport_content_length(body_rows);
            StatefulWidget::render(
                Scrollbar::new(ScrollbarOrientation::VerticalRight)
                    .begin_symbol(None)
                    .end_symbol(None)
                    .track_symbol(Some("│"))
                    .thumb_symbol("█")
                    .track_style(Style::default().fg(palette::DIM))
                    .thumb_style(Style::default().fg(palette::ACCENT)),
                Rect {
                    x: sb.col,
                    y: LAYOUT_HEADER_ROWS as u16,
                    width: 1,
                    height: body_rows as u16,
                },
                frame.buffer_mut(),
                &mut state,
            );
        }

        // Clickable footer chips (hints start at the footer's top row) + `[×]`
        // (→ back to the tensor tree, like the file view's close).
        let footer_top = (height as usize - footer_rows) as u16;
        let mut regions = chip_regions(&chips, footer_top);
        regions.extend(close_button(
            frame,
            KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE),
        ));
        (max_scroll, regions, links)
    }

    /// A vertical scrollbar for a plain `total`-row list of `visible` rows sitting
    /// just below the layout-map header — reusing [`TreeScrollbar`].
    fn files_scrollbar_like(width: u16, total: usize, visible: usize) -> Option<TreeScrollbar> {
        if width < 2 || total <= visible {
            return None;
        }
        Some(TreeScrollbar {
            col: width - 1,
            top: LAYOUT_HEADER_ROWS as u16,
            rows: visible as u16,
            max_offset: total - visible,
        })
    }

    /// The cumulative start row of each layout-map band, plus the total row count
    /// as a trailing entry (so band `i` spans `[starts[i], starts[i+1])`). Lets the
    /// browsing loop map a click to a segment and snap the scroll to the selection,
    /// using the same band heights [`Self::render_layout`] draws.
    pub fn layout_band_starts(
        map: &crate::safelayout::LayoutMap,
        width: u16,
        height: u16,
    ) -> Vec<usize> {
        let body_rows = Self::layout_visible_rows(width, height);
        let mut starts = Vec::with_capacity(map.segments.len() + 1);
        let mut acc = 0usize;
        for h in band_rows(map, body_rows) {
            starts.push(acc);
            acc += h;
        }
        starts.push(acc);
        starts
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
    ) -> (ChipRegions, LinkRegions) {
        let area = frame.area();
        let (width, height) = (area.width, area.height);

        let (header, stats_gauge_row, links) =
            detail_field_lines(tensor, shape, view, unindexed, stats, schema, width);
        let remote = crate::remote::is_remote_source(&tensor.source_path);
        // The `Tab` → file-layout hint shows only for a local `.safetensors` shard
        // (the only source with a byte-layout map).
        let layout = !remote
            && std::path::Path::new(&tensor.source_path)
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"));
        let (footer, chips) = detail_footer_lines(overridable, remote, layout, width);
        let header_len = header.len();
        let footer_len = footer.len();

        // Header at the top; the footer is pinned to the **bottom** (above the remote
        // metadata-only banner), with any histogram filling the space between — the
        // same bottom-pinned footer every other view has. `footer_top` is the footer's
        // first screen row, so chip lines can be made absolute for hit-testing.
        let banner = usize::from(remote);
        let footer_top = (height as usize).saturating_sub(footer_len + banner) as u16;
        Paragraph::new(header).render(
            Rect {
                x: 0,
                y: 0,
                width,
                height: header_len as u16,
            },
            frame.buffer_mut(),
        );
        if let Some(hist) = histogram {
            // The histogram fills between the header and the footer (a blank spacer
            // row above the footer), so the screen fills exactly with no scroll.
            let section = (footer_top as usize).saturating_sub(header_len + 1).max(1);
            render_histogram(
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
        }
        Paragraph::new(footer).render(
            Rect {
                x: 0,
                y: footer_top,
                width,
                height: footer_len as u16,
            },
            frame.buffer_mut(),
        );

        // Metadata-only banner on the bottom row (remote `--ssh-read`) — the lower
        // part of the detail screen is otherwise blank, so it doesn't overlap.
        if crate::remote::is_remote_source(&tensor.source_path) {
            Paragraph::new(Line::from(Span::styled(
                " metadata-only — data views need the file locally ",
                Style::default()
                    .bg(palette::STATUS_BG)
                    .fg(palette::WARN)
                    .add_modifier(Modifier::BOLD),
            )))
            .render(
                Rect {
                    x: 0,
                    y: height.saturating_sub(1),
                    width,
                    height: 1,
                },
                frame.buffer_mut(),
            );
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
        let mut regions = chip_regions(&chips, footer_top);
        regions.extend(close_button(
            frame,
            KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE),
        ));

        // A pop-up overlay composites last, over the live frame, so the detail
        // (including a running scan's progress) keeps animating behind it.
        match overlay {
            Some(Overlay::Legend(l)) => Self::render_legend_band(frame, *l),
            Some(Overlay::Command(c)) => Self::render_command_band(frame, c),
            Some(Overlay::Notice(m)) => Self::render_notice_box(frame, m),
            None => {}
        }
        (regions, links)
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
            None,
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
        let value_lines: Vec<Line> = highlight_json_lines(&metadata.value).unwrap_or_else(|| {
            metadata
                .value
                .lines()
                .map(|l| Line::from(l.to_string()))
                .collect()
        });

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

        let (footer, chips) = data_view_footer_wrapped_lines(
            sample.mode,
            sample.slices,
            true,
            true,
            StripeMode::Off,
            NumBase::Decimal,
            width,
        );
        // Bottom-pin the footer; the sampled content fills the region above it
        // (clipped if it would overflow), like every other view.
        let footer_len = footer.len() as u16;
        // Reserve the bottom row for the access badge (drawn by render_data_frame),
        // so the footer's last chip never runs under it.
        let footer_top = area.height.saturating_sub(footer_len + 1);
        Paragraph::new(lines).render(
            Rect {
                x: 0,
                y: 0,
                width: area.width,
                height: footer_top,
            },
            frame.buffer_mut(),
        );
        Paragraph::new(footer).render(
            Rect {
                x: 0,
                y: footer_top,
                width: area.width,
                height: footer_len,
            },
            frame.buffer_mut(),
        );
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

        let (footer, chips) = data_view_footer_wrapped_lines(
            sample.mode,
            sample.slices,
            sample.overridable,
            false,
            stripe,
            base,
            width,
        );
        // Bottom-pin the footer; the value grid fills the region above it (clipped
        // if it would overflow), like every other view.
        let footer_len = footer.len() as u16;
        // Reserve the bottom row for the access badge (drawn by render_data_frame),
        // so the footer's last chip never runs under it.
        let footer_top = area.height.saturating_sub(footer_len + 1);
        Paragraph::new(lines).render(
            Rect {
                x: 0,
                y: 0,
                width: area.width,
                height: footer_top,
            },
            frame.buffer_mut(),
        );
        Paragraph::new(footer).render(
            Rect {
                x: 0,
                y: footer_top,
                width: area.width,
                height: footer_len,
            },
            frame.buffer_mut(),
        );
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

    /// A yes/no confirmation **floated over the live frame** (the screen behind stays
    /// visible): a title, the `body` summary lines, then an `[Apply] [Cancel]`-style
    /// choice strip (the `selected` option inverted) and a key hint. Drives the
    /// in-place rename apply confirmation.
    pub fn render_confirm_popup(
        frame: &mut Frame,
        title: &str,
        body: &[String],
        options: &[&str],
        selected: usize,
    ) {
        let mut content: Vec<Line> = body
            .iter()
            .map(|l| Line::from(Span::raw(l.clone())))
            .collect();
        content.push(Line::default());
        let mut strip: Vec<Span> = Vec::new();
        for (i, opt) in options.iter().enumerate() {
            let label = format!(" {opt} ");
            if i == selected {
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
            strip.push(Span::raw("  "));
        }
        content.push(Line::from(strip));
        content.push(Line::default());
        content.push(Line::from(hint_spans(&[
            ("← →", "move"),
            ("Enter", "select"),
            ("Y", "apply"),
            ("Esc", "cancel"),
        ])));
        render_popup_box(frame, title, content, Backdrop::Float, None);
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
            None,
        );
    }

    /// A metadata-only / unavailable notice **floated over** the live frame (the
    /// screen behind stays visible — unlike [`Self::render_message`]), dismissed by
    /// any key. Used for [`Overlay::Notice`].
    pub fn render_notice_box(frame: &mut Frame, message: &str) {
        render_popup_box(
            frame,
            "Metadata-only",
            vec![
                Line::from(Span::raw(message.to_string())),
                Line::default(),
                Line::from(dim_span("Click or press any key to dismiss")),
            ],
            Backdrop::Float,
            None,
        );
    }

    /// Float the health-check report (`h` in the tree) over the live tree. Built
    /// as styled lines directly from the [`CheckReport`](crate::check::CheckReport)
    /// (so every span sits on the popup's panel background, matching the box) —
    /// coloured marks per check, indented findings, a verdict, and a `state`-driven
    /// footer. While scanning, the "Value scan" row becomes an animated spinner.
    /// Render the health-check popup, its body scrolled by `scroll` rows (the
    /// footer stays pinned). Returns the max valid scroll so the caller can clamp.
    pub fn render_check_report(
        frame: &mut Frame,
        report: &crate::check::CheckReport,
        state: CheckPopup,
        scroll: usize,
        expanded: bool,
    ) -> (usize, Vec<(Rect, KeyEvent)>) {
        use crate::check::{Severity, Status, count_phrase, fmt_elapsed};
        let bg = palette::PANEL_BG;
        // Every span carries the panel background, so text and box match.
        let sty = |s: String, style: Style| Span::styled(s, style.bg(bg));
        // Body-line indices of the per-check findings toggles (all clickable → `f`).
        let mut fold_lines: Vec<usize> = Vec::new();

        // Title column width, including the synthetic "Value scan" row.
        let width = report
            .results
            .iter()
            .map(|r| r.title.len())
            .chain(std::iter::once("Value scan".len()))
            .max()
            .unwrap_or(0);

        let mut lines: Vec<Line> = vec![Line::from(sty(
            format!(
                "{} file(s) · {} tensors · {} params",
                report.n_files,
                report.n_tensors,
                crate::utils::format_parameters(report.params)
            ),
            Style::default().fg(palette::DIM),
        ))];

        for r in &report.results {
            let (mark, mc) = match r.status() {
                Status::Pass => ("✓", palette::SUCCESS),
                Status::Warn => ("⚠", palette::WARN),
                Status::Fail => ("✗", palette::ERROR),
                Status::Na => ("⊘", palette::DIM),
            };
            let mut trailer_text = match r.status() {
                Status::Pass => format!("— {}", r.summary.as_deref().unwrap_or(r.note)),
                Status::Na => "— n/a for this checkpoint".to_string(),
                _ => format!("({})", count_phrase(r.errors(), r.warnings())),
            };
            // The value scan carries its wall-clock time (like the CLI bar).
            if let Some(d) = r.elapsed {
                trailer_text.push_str(&format!("  ({})", fmt_elapsed(d)));
            }
            let trailer = sty(trailer_text, Style::default().fg(palette::DIM));
            lines.push(check_row(
                sty(mark.into(), Style::default().fg(mc)),
                r.title,
                width,
                trailer,
                bg,
            ));
            // The per-finding detail is folded away by default (like the stats
            // popup's per-shard list). Under each check with findings sits a
            // toggle aligned with the check title; `f` (or a click on it, either
            // state) reveals the full list. The `f` hint lives in the footer, with
            // the other keys, so it stays put and consistently styled.
            if !r.findings.is_empty() {
                let arrow = if expanded { "▾" } else { "▸" };
                fold_lines.push(lines.len());
                lines.push(Line::from(vec![
                    sty(
                        format!("    {arrow} "),
                        Style::default().fg(palette::ACCENT),
                    ),
                    sty(
                        format!(
                            "{} finding{}",
                            r.findings.len(),
                            if r.findings.len() == 1 { "" } else { "s" }
                        ),
                        Style::default().fg(palette::DIM),
                    ),
                ]));
                if expanded {
                    for f in &r.findings {
                        let (fm, fc) = match f.severity {
                            Severity::Error => ("✗", palette::ERROR),
                            Severity::Warning => ("⚠", palette::WARN),
                        };
                        let mut spans = vec![
                            sty("      ".into(), Style::default()),
                            sty(fm.into(), Style::default().fg(fc)),
                            sty(" ".into(), Style::default()),
                        ];
                        if let Some(subj) = &f.subject {
                            spans.push(sty(
                                format!("{subj}  "),
                                Style::default().add_modifier(Modifier::BOLD),
                            ));
                        }
                        spans.push(sty(f.message.clone(), Style::default()));
                        lines.push(Line::from(spans));
                    }
                }
            }
        }

        // The value tier isn't in `results` until it runs: show a spinner while
        // scanning, else a "not run" hint.
        if !report.values {
            let (mark, mc, trailer) = match state {
                // The count lives in the footer bar — don't repeat it here.
                CheckPopup::Scanning { frame, .. } => (
                    CHECK_SPINNER[frame % CHECK_SPINNER.len()].to_string(),
                    palette::ACCENT,
                    sty("— scanning…".into(), Style::default().fg(palette::DIM)),
                ),
                // Only suggest `v` when the scan is actually available — it isn't
                // for a remote checkpoint (data stays on the host).
                CheckPopup::Idle { can_scan, .. } => (
                    "·".into(),
                    palette::DIM,
                    sty(
                        if can_scan {
                            "— not run (press v)"
                        } else {
                            "— not run"
                        }
                        .into(),
                        Style::default().fg(palette::DIM),
                    ),
                ),
            };
            lines.push(check_row(
                sty(mark, Style::default().fg(mc)),
                "Value scan",
                width,
                trailer,
                bg,
            ));
        }

        let (e, w) = (report.errors(), report.warnings());
        let verdict = if e > 0 {
            sty(
                format!("FAIL — {}", count_phrase(e, w)),
                Style::default().fg(palette::ERROR),
            )
        } else if w > 0 {
            sty(
                format!("OK with warnings — {}", count_phrase(0, w)),
                Style::default().fg(palette::WARN),
            )
        } else if report.values {
            sty(
                "OK — no issues found".into(),
                Style::default().fg(palette::SUCCESS),
            )
        } else {
            sty(
                "OK — no metadata issues found".into(),
                Style::default().fg(palette::SUCCESS),
            )
        };
        lines.push(Line::from(vec![
            sty("  ".into(), Style::default()),
            verdict,
        ]));

        // Every per-check findings toggle is clickable (→ `f`), so a click folds
        // or unfolds in either state.
        let clickable: Vec<(usize, KeyEvent)> = fold_lines
            .iter()
            .map(|&i| (i, KeyEvent::new(KeyCode::Char('f'), KeyModifiers::NONE)))
            .collect();
        // The `f` fold hint goes in the footer (only when there are findings).
        let fold = (!fold_lines.is_empty()).then_some(expanded);
        // The key-hint footer stays pinned while the body (checks + findings)
        // scrolls, so a report with many findings never overflows the popup.
        render_scroll_popup(
            frame,
            "Health check",
            &lines,
            check_footer_line(&state, fold, bg),
            scroll,
            &clickable,
        )
    }

    /// The overall-checkpoint stats popup (the `s` key on the tree). Returns the
    /// max scroll offset, like [`Self::render_check_report`], so the caller can
    /// clamp its scroll state to what actually fit.
    pub fn render_stats(
        frame: &mut Frame,
        s: &crate::stats::CheckpointStats,
        copied: Option<&'static str>,
        scroll: usize,
        shards_expanded: bool,
    ) -> (usize, Vec<(Rect, KeyEvent)>) {
        use crate::stats::ExpertStorage;
        let bg = palette::PANEL_BG;
        let sty = |t: String, style: Style| Span::styled(t, style.bg(bg));
        let plain = |t: String| sty(t, Style::default());
        let dim = |t: String| sty(t, Style::default().fg(palette::DIM));

        // A section header, then indented "label   value" rows. Labels align to a
        // fixed column so the values line up down the popup.
        const LW: usize = 12;
        let header = |t: &str| {
            Line::from(sty(
                t.to_string(),
                Style::default()
                    .fg(palette::ACCENT)
                    .add_modifier(Modifier::BOLD),
            ))
        };
        // A glyphed section header like the tree — "▦ Tensors  ×116175" — with the
        // glyph + title in accent, the `count` plain (not dim, so it stands out),
        // and a dim `qualifier` (e.g. " per layer", " safetensors").
        let section = |glyph: &str, title: &str, count: String, qualifier: &str| {
            let accent = Style::default().fg(palette::ACCENT);
            let mut spans = vec![
                sty(format!("{glyph} "), accent),
                sty(title.to_string(), accent.add_modifier(Modifier::BOLD)),
            ];
            if !count.is_empty() {
                spans.push(plain(format!("  {count}")));
            }
            if !qualifier.is_empty() {
                spans.push(dim(qualifier.to_string()));
            }
            Line::from(spans)
        };
        // Pad the label to `LW`, then a guaranteed separator space — so a label
        // that exactly fills `LW` (e.g. "Architecture") still has a gap before it.
        let row = |label: &str, mut value: Vec<Span<'static>>| {
            let mut spans = vec![plain(format!("  {label:<LW$} "))];
            spans.append(&mut value);
            Line::from(spans)
        };
        // "<size> each · <size> total", the shared shape of the layer/expert rows.
        let each_total = |each: usize, total: usize, fmt: fn(usize) -> String| {
            vec![
                plain(fmt(each)),
                dim(" each · ".into()),
                plain(fmt(total)),
                dim(" total".into()),
            ]
        };

        let mut lines: Vec<Line> = Vec::new();
        // Body-line index of the per-shard fold toggle, once emitted (for click
        // hit-testing).
        let mut fold_line: Option<usize> = None;

        // ── Overview ──────────────────────────────────────────────────────────
        lines.push(header("Overview"));
        if let Some(mt) = &s.model_type {
            lines.push(row("Architecture", vec![plain(mt.clone())]));
        }
        lines.push(row("Parameters", vec![plain(format_parameters(s.params))]));
        // On-disk vs logical, with a compression ratio when they differ.
        let size_value = if s.compressed && s.disk_bytes > 0 {
            vec![
                plain(format_size(s.disk_bytes)),
                dim(" on disk · ".into()),
                plain(format_size(s.logical_bytes)),
                dim(format!(
                    " logical ({:.2}× smaller)",
                    s.logical_bytes as f64 / s.disk_bytes as f64
                )),
            ]
        } else {
            vec![plain(format_size(s.logical_bytes))]
        };
        lines.push(row("Size", size_value));

        // ── Files (per-shard logical size) ────────────────────────────────────
        lines.push(Line::from(sty(String::new(), Style::default())));
        let kind = if s.files.noun.starts_with("safetensors") {
            " safetensors"
        } else {
            ""
        };
        lines.push(section(
            crate::stats::GLYPH_FILES,
            "Files",
            format!("×{}", s.files.count),
            kind,
        ));
        // A `size  name` value, size padded and the name dimmed — shared by the
        // per-file and per-tensor largest/smallest rows so they read alike.
        let named = |n: &crate::stats::NamedSize| {
            vec![
                plain(format!("{:<9} ", format_size(n.bytes))),
                dim(n.name.clone()),
            ]
        };
        if let Some(l) = &s.files.largest {
            lines.push(row("Largest", named(l)));
        }
        if let Some(sm) = &s.files.smallest {
            lines.push(row("Smallest", named(sm)));
        }
        lines.push(row("Average", vec![plain(format_size(s.files.mean))]));
        lines.push(row("Median", vec![plain(format_size(s.files.median))]));

        // ── Tensors (count + size) ────────────────────────────────────────────
        lines.push(Line::from(sty(String::new(), Style::default())));
        lines.push(section(
            crate::stats::GLYPH_TENSORS,
            "Tensors",
            format!("×{}", s.n_tensors),
            "",
        ));
        if let Some(l) = &s.largest {
            lines.push(row("Largest", named(l)));
        }
        if let Some(sm) = &s.smallest {
            lines.push(row("Smallest", named(sm)));
        }
        lines.push(row("Average", vec![plain(format_size(s.mean_bytes))]));
        lines.push(row("Median", vec![plain(format_size(s.median_bytes))]));

        // ── Layers ───────────────────────────────────────────────────────────
        if let Some(l) = &s.layers {
            lines.push(Line::from(sty(String::new(), Style::default())));
            lines.push(section(
                crate::stats::GLYPH_LAYERS,
                "Layers",
                format!("×{}", l.count),
                "",
            ));
            lines.push(row(
                "Params",
                each_total(l.params_each(), l.params, format_parameters),
            ));
            lines.push(row(
                "Size",
                each_total(l.bytes_each(), l.bytes, format_size),
            ));
        }

        // ── Experts (MoE) ─────────────────────────────────────────────────────
        if let Some(x) = &s.experts {
            lines.push(Line::from(sty(String::new(), Style::default())));
            let (count, qualifier) = if x.per_layer > 0 {
                (format!("×{}", x.per_layer), " per layer")
            } else {
                (String::new(), "")
            };
            lines.push(section(
                crate::stats::GLYPH_EXPERTS,
                "Experts",
                count,
                qualifier,
            ));
            let mut storage = x.storage.label().to_string();
            if x.gate_up_fused {
                storage.push_str(" · gate+up fused");
            }
            lines.push(row("Storage", vec![plain(storage)]));
            // Per-expert averages are only meaningful once we know the layout.
            if x.per_layer > 0 || x.storage == ExpertStorage::Unfused {
                lines.push(row(
                    "Params",
                    each_total(x.params_each(), x.params, format_parameters),
                ));
                lines.push(row(
                    "Size",
                    each_total(x.bytes_each(), x.bytes, format_size),
                ));
            }
        }

        // ── dtype mix ─────────────────────────────────────────────────────────
        if !s.dtypes.is_empty() {
            lines.push(Line::from(sty(String::new(), Style::default())));
            lines.push(header("By dtype"));
            let dw = s.dtypes.iter().map(|d| d.dtype.len()).max().unwrap_or(0);
            for d in &s.dtypes {
                lines.push(Line::from(vec![
                    sty(
                        format!("  {:<dw$}  ", d.dtype),
                        Style::default().fg(palette::DTYPE),
                    ),
                    plain(format!("{:>7}", format_size(d.bytes))),
                    plain(format!("  {}", d.count)),
                    dim(format!(" tensor{}", if d.count == 1 { "" } else { "s" })),
                ]));
            }
        }

        // ── On disk (filesystem allocation) ────────────────────────────────────
        if let Some(d) = &s.disk {
            lines.push(Line::from(sty(String::new(), Style::default())));
            lines.push(header("On disk (filesystem)"));
            lines.push(row(
                "Allocated",
                vec![
                    plain(format_size(d.total_allocated as usize)),
                    dim(format!(
                        "  ({} apparent, {})",
                        format_size(d.total_apparent as usize),
                        crate::stats::ratio_phrase(d.total_apparent, d.total_allocated),
                    )),
                ],
            ));
            // The per-shard breakdown is folded away by default (a many-shard
            // model is otherwise a wall of rows); a click on this line or `f`
            // toggles it. Only shards the filesystem actually shrank are listed.
            if d.shards.len() > 1 {
                let savers: Vec<&crate::stats::ShardDisk> = d
                    .shards
                    .iter()
                    .filter(|sh| crate::stats::has_saving(sh.apparent, sh.allocated))
                    .collect();
                let arrow = if shards_expanded { "▾" } else { "▸" };
                // The `f` hint lives in the footer with the other keys; the toggle
                // itself just labels the breakdown (and, folded, the saver count).
                let tail = if shards_expanded {
                    String::new()
                } else {
                    format!("  ({} of {} smaller)", savers.len(), d.shards.len())
                };
                fold_line = Some(lines.len());
                lines.push(Line::from(vec![
                    sty(format!("  {arrow} "), Style::default().fg(palette::ACCENT)),
                    plain("per-shard breakdown".into()),
                    dim(tail),
                ]));
                if shards_expanded {
                    // Unfolding shows *every* shard (savers and not) — the folded
                    // summary already gave the "N of M smaller" headline, so the
                    // expanded view is the full breakdown, not a filtered one.
                    let nw = d.shards.iter().map(|sh| sh.name.len()).max().unwrap_or(0);
                    for sh in &d.shards {
                        lines.push(Line::from(vec![
                            sty(
                                format!("    {:<nw$}  ", sh.name),
                                Style::default().fg(palette::META),
                            ),
                            plain(format!("{:>9}", format_size(sh.apparent as usize))),
                            dim(" → ".into()),
                            plain(format!("{:>9}", format_size(sh.allocated as usize))),
                            dim(format!(
                                "  ({})",
                                crate::stats::ratio_phrase(sh.apparent, sh.allocated)
                            )),
                        ]));
                    }
                }
            }
        }

        // The fold toggle (when present) is the popup's one clickable body row.
        let clickable: Vec<(usize, KeyEvent)> = fold_line
            .map(|i| (i, KeyEvent::new(KeyCode::Char('f'), KeyModifiers::NONE)))
            .into_iter()
            .collect();
        // The `f` fold hint goes in the footer (only when there's a breakdown).
        let fold = fold_line.map(|_| shards_expanded);
        render_scroll_popup(
            frame,
            "Checkpoint stats",
            &lines,
            stats_footer_line(copied, fold, bg),
            scroll,
            &clickable,
        )
    }

    /// Borderless band shown when a chosen export is too big for the terminal
    /// clipboard: it copies the concrete CLI command that reproduces it instead
    /// and shows it on its own full-width line(s) at column 0 (so a long path
    /// stays selectable even past the terminal width). Mirrors
    /// [`Self::render_command_band`].
    pub fn render_export_band(frame: &mut Frame, command: &str) {
        let term_w = (frame.area().width as usize).max(1);
        let title = Line::from(vec![
            Span::styled(
                " Too large to copy ",
                Style::default()
                    .fg(palette::KEY)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "— export command copied to the clipboard ",
                Style::default().fg(palette::SUCCESS),
            ),
        ]);
        // The command, soft-wrapped at full width onto its own line(s), flush at
        // column 0 so it stays selectable by hand when OSC-52 can't reach the
        // terminal.
        let chars: Vec<char> = command.chars().collect();
        let cmd_rows = chars.len().div_ceil(term_w).max(1);
        let mut content: Vec<Line> = (0..cmd_rows)
            .map(|r| {
                let seg: String = chars.iter().skip(r * term_w).take(term_w).collect();
                Line::from(Span::raw(seg))
            })
            .collect();
        content.push(Line::from(dim_span(
            "run it to export  ·  any key dismisses",
        )));
        render_titled_bar(frame, title, content);
    }

    /// A floating selection menu: `items` numbered one per row with `selected`
    /// highlighted, a `preview` of the highlighted choice's output below, and a
    /// key hint. Used by the `t` copy-format picker; the caller drives selection
    /// and repaints. Returns each item's on-screen rect so clicks/hovers can be
    /// mapped back to a row.
    pub fn render_menu_box(
        frame: &mut Frame,
        title: &str,
        items: &[&str],
        selected: usize,
        preview: &[Line<'static>],
    ) -> Vec<Rect> {
        let mut content: Vec<Line> = items
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let row = format!("{}. {item}", i + 1);
                if i == selected {
                    Line::from(Span::styled(
                        format!("▸ {row}"),
                        Style::default()
                            .fg(palette::ACCENT)
                            .add_modifier(Modifier::BOLD),
                    ))
                } else {
                    Line::from(Span::raw(format!("  {row}")))
                }
            })
            .collect();
        // Live, tree-coloured preview of the highlighted export (from the current
        // checkpoint), each line indented under a "preview:" header.
        content.push(Line::default());
        content.push(Line::from(dim_span("preview:")));
        for line in preview {
            let mut spans = vec![Span::raw("  ")];
            spans.extend(line.spans.iter().cloned());
            content.push(Line::from(spans));
        }
        content.push(Line::default());
        content.push(Line::from(dim_span(
            "↑/↓ or 1–8 choose  ·  Enter/click copy  ·  Esc cancel",
        )));
        // A fixed inner width keeps the box a constant size across options (the
        // preview rows are already a fixed count); over-wide lines are clipped.
        let width = (frame.area().width as usize)
            .saturating_sub(4)
            .clamp(24, 110);
        let inner = render_popup_box(frame, title, content, Backdrop::Float, Some(width));
        // The items occupy the first `items.len()` inner rows.
        (0..items.len())
            .map(|i| Rect {
                x: inner.x,
                y: inner.y + i as u16,
                width: inner.width,
                height: 1,
            })
            .collect()
    }

    /// The command palette: a query line above a fuzzy-filtered list of commands
    /// (each `key`, `title`, and `help`), the selected row inverted. Returns the
    /// on-screen rect of every listed row so a click can pick it. Fixed width so
    /// the box doesn't jump as the query filters the list.
    pub fn render_command_palette(
        frame: &mut Frame,
        query: &str,
        rows: &[(String, String, String, String)],
        selected: usize,
    ) -> Vec<Rect> {
        let key_w = rows
            .iter()
            .map(|(k, ..)| k.chars().count())
            .max()
            .unwrap_or(1);
        // `Group: Title` in one column, aligned so the help lines up.
        let label = |group: &str, title: &str| format!("{group}: {title}");
        let label_w = rows
            .iter()
            .map(|(_, g, t, _)| label(g, t).chars().count())
            .max()
            .unwrap_or(0);

        let mut content: Vec<Line> = Vec::new();
        let mut query_line = vec![Span::styled(
            "❯ ",
            Style::default()
                .fg(palette::KEY)
                .add_modifier(Modifier::BOLD),
        )];
        // Caret at the end (the palette input only appends / backspaces).
        query_line.extend(input_box_spans(query, query.chars().count(), 24));
        content.push(Line::from(query_line));
        content.push(Line::default());

        if rows.is_empty() {
            content.push(Line::from(dim_span("  (no matching commands)")));
        }
        for (i, (key, group, title, help)) in rows.iter().enumerate() {
            let pad_k = " ".repeat(key_w.saturating_sub(key.chars().count()));
            let pad_l = " ".repeat(label_w.saturating_sub(label(group, title).chars().count()));
            if i == selected {
                content.push(Line::from(Span::styled(
                    format!("  {pad_k}{key}  {group}: {title}{pad_l}  {help} "),
                    Style::default()
                        .fg(palette::SELECT_FG)
                        .bg(palette::SELECT_BG)
                        .add_modifier(Modifier::BOLD),
                )));
            } else {
                content.push(Line::from(vec![
                    Span::raw(format!("  {pad_k}")),
                    Span::styled(
                        key.clone(),
                        Style::default()
                            .fg(palette::KEY)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw("  "),
                    // Category dimmed, command title normal — the VS Code look.
                    dim_span(format!("{group}: ")),
                    Span::raw(format!("{title}{pad_l}  ")),
                    dim_span(help.clone()),
                ]));
            }
        }
        content.push(Line::default());
        content.push(Line::from(dim_span(
            "↑/↓ move · Enter run · type to filter · Esc close",
        )));

        let width = (frame.area().width as usize)
            .saturating_sub(4)
            .clamp(30, 100);
        let inner = render_popup_box(frame, "Commands", content, Backdrop::Float, Some(width));
        // Rows start after the query line (0) and the blank separator (1).
        (0..rows.len())
            .map(|i| Rect {
                x: inner.x,
                y: inner.y + 2 + i as u16,
                width: inner.width,
                height: 1,
            })
            .collect()
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
        // The caller supplies the whole message (not just clipboard copies) — e.g.
        // "copied the screen to the clipboard" or "statistics already computed".
        let full = format!("✓ {what}");
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

/// The state of the health-check popup ([`UI::render_check_report`]).
#[derive(Clone, Copy)]
pub enum CheckPopup {
    /// Showing the report. `copied` briefly flashes what was just copied
    /// (`"command"` / `"report"` / `"screen"`); `can_scan` offers the `v` value
    /// scan (off for a remote source or once it has run).
    Idle {
        copied: Option<&'static str>,
        can_scan: bool,
    },
    /// A value scan is running: `done`/`total` tensors, `frame` animates the row
    /// spinner and drives the footer bar.
    Scanning {
        done: usize,
        total: usize,
        frame: usize,
    },
}

/// Braille spinner frames for the in-progress "Value scan" row.
const CHECK_SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// One check row: `  <mark> <title padded>  <trailer>`, all on the panel `bg`.
fn check_row(
    mark: Span<'static>,
    title: &str,
    width: usize,
    trailer: Span<'static>,
    bg: Color,
) -> Line<'static> {
    Line::from(vec![
        Span::styled("  ", Style::default().bg(bg)),
        mark,
        Span::styled(format!(" {title:<width$}  "), Style::default().bg(bg)),
        trailer,
    ])
}

/// The popup footer: the value-scan bar while scanning, a copy confirmation right
/// after `y`, or the key hints — with the key glyphs bold/accented (not dimmed)
/// so it's clear they're actionable.
fn check_footer_line(state: &CheckPopup, fold: Option<bool>, bg: Color) -> Line<'static> {
    let key = |k: &str| {
        Span::styled(
            k.to_string(),
            Style::default()
                .fg(palette::KEY)
                .add_modifier(Modifier::BOLD)
                .bg(bg),
        )
    };
    // Descriptions in the default foreground, only " · "/"cancel"-style connective
    // text dimmed — matching the tree view's footer style.
    let dim = |s: &str| Span::styled(s.to_string(), Style::default().fg(palette::DIM).bg(bg));
    let label = |s: &str| Span::styled(s.to_string(), Style::default().bg(bg));
    match *state {
        CheckPopup::Scanning { done, total, .. } => {
            const W: usize = 18;
            let filled = if total == 0 {
                0
            } else {
                (((done as f64 / total as f64) * W as f64).round() as usize).min(W)
            };
            Line::from(vec![
                Span::styled(
                    "━".repeat(filled),
                    Style::default().fg(palette::ACCENT).bg(bg),
                ),
                Span::styled(
                    "━".repeat(W - filled),
                    Style::default().fg(palette::DIM).bg(bg),
                ),
                Span::styled(format!("  {done}/{total}   "), Style::default().bg(bg)),
                key("Esc"),
                label(" cancel"),
            ])
        }
        CheckPopup::Idle {
            copied: Some(what), ..
        } => Line::from(Span::styled(
            format!("✓ copied {what} to the clipboard"),
            Style::default().fg(palette::SUCCESS).bg(bg),
        )),
        CheckPopup::Idle { can_scan, .. } => {
            let mut items: Vec<(&str, &str)> = Vec::new();
            // The findings-fold key, when there are findings to fold — a footer
            // hint (not inline text) so it matches the other keys and stays visible
            // whether folded or expanded.
            match fold {
                Some(true) => items.push(("f", " fold findings")),
                Some(false) => items.push(("f", " expand findings")),
                None => {}
            }
            if can_scan {
                items.push(("v", " value scan"));
            }
            items.push(("c", " copy screen"));
            items.push(("r", " copy report"));
            items.push(("y", " copy command"));
            items.push(("Esc", " dismiss"));
            let mut spans = Vec::new();
            for (i, (k, lbl)) in items.iter().enumerate() {
                if i > 0 {
                    spans.push(dim(" · "));
                }
                spans.push(key(k));
                spans.push(label(lbl));
            }
            Line::from(spans)
        }
    }
}

/// Footer for the stats popup: a "✓ copied …" flash, or the key hints.
fn stats_footer_line(copied: Option<&'static str>, fold: Option<bool>, bg: Color) -> Line<'static> {
    if let Some(what) = copied {
        return Line::from(Span::styled(
            format!("✓ copied {what} to the clipboard"),
            Style::default().fg(palette::SUCCESS).bg(bg),
        ));
    }
    let key = |k: &str| {
        Span::styled(
            k.to_string(),
            Style::default()
                .fg(palette::KEY)
                .add_modifier(Modifier::BOLD)
                .bg(bg),
        )
    };
    // Descriptions in the default foreground and only the " · " separators dimmed,
    // matching the tree view's footer (the other footers follow its style).
    let dim = |s: &str| Span::styled(s.to_string(), Style::default().fg(palette::DIM).bg(bg));
    let label = |s: &str| Span::styled(s.to_string(), Style::default().bg(bg));
    let mut items: Vec<(&str, &str)> = Vec::new();
    // The per-shard fold key, when there's a breakdown to fold — a footer hint
    // (not inline) so it matches the other keys and stays visible either way.
    match fold {
        Some(true) => items.push(("f", " fold shards")),
        Some(false) => items.push(("f", " expand shards")),
        None => {}
    }
    items.push(("c", " copy screen"));
    items.push(("r", " copy report"));
    items.push(("y", " copy command"));
    items.push(("Esc", " dismiss"));
    let mut spans = Vec::new();
    for (i, (k, lbl)) in items.iter().enumerate() {
        if i > 0 {
            spans.push(dim(" · "));
        }
        spans.push(key(k));
        spans.push(label(lbl));
    }
    Line::from(spans)
}

/// A centred, content-sized pop-up over the frame: a rounded [`Block`] (accent
/// border, `title` on the top edge, panel background) wrapping `content`. With
/// [`Backdrop::Float`] the surrounding frame is left untouched (only the box rect
/// is cleared) so the screen behind stays visible — a real pop-up; with
/// [`Backdrop::Fill`] the whole frame is wiped to the scrim first, for standalone
/// message screens. Shared by the legend pop-up and message screens.
/// Draw a centered popup box and return its inner (content) rect, so callers
/// that need to hit-test the content — e.g. a clickable menu — can map screen
/// coordinates to rows. `fixed_inner_w`, when set, pins the content width (lines
/// wider than it are clipped) so the box is a constant size regardless of its
/// content — otherwise the box sizes to the widest line.
/// The left column the rename editor's autocomplete dropdown anchors at — under
/// the field's value box (2-space indent + 4-wide label + space).
const RENAME_MENU_X: u16 = 7;

/// Draw the rename editor's autocomplete dropdown: a background-filled block (no
/// box-drawing border — it's outlined by its fill colour), floating just below the
/// focused field, one candidate per row with the highlighted one inverted, the
/// matched substring emboldened, and a dim right-aligned `×N` tensor count. A final
/// dim caption row spells out the keys. `field_row` is the focused field's absolute
/// row; the box drops beneath it, or flips above when it would overflow the frame.
/// Returns each candidate row's on-screen rect (the caption excluded) so a click
/// can accept it.
fn render_completion_menu(
    frame: &mut Frame,
    anchor_x: u16,
    field_row: u16,
    cands: &[RenameCompletion],
    selected: usize,
) -> Vec<Rect> {
    let area = frame.area();
    if cands.is_empty() || area.width <= anchor_x {
        return Vec::new();
    }
    const CAPTION: &str = "↑/↓ pick · ↵ accept · Tab complete · Esc close";
    let count_label = |n: usize| format!("×{n}");
    let name_w = cands
        .iter()
        .map(|c| c.text.chars().count())
        .max()
        .unwrap_or(0);
    let count_w = cands
        .iter()
        .map(|c| count_label(c.count).chars().count())
        .max()
        .unwrap_or(0);
    // 1 lead + name + 2 gap + count + 1 trail, but at least wide enough for the
    // caption, and never past the frame's right edge.
    let inner_w = (1 + name_w + 2 + count_w + 1).max(CAPTION.chars().count() + 2);
    let box_w = (inner_w as u16).min(area.width - anchor_x).max(1);
    let box_h = (cands.len() as u16 + 1).min(area.height); // +1 caption row
    // Prefer dropping below the field; flip above when there's no room beneath.
    let below = field_row + 1;
    let box_y = if below + box_h <= area.height {
        below
    } else {
        field_row.saturating_sub(box_h)
    };
    let box_x = anchor_x.min(area.width.saturating_sub(box_w));
    let rect = Rect {
        x: box_x,
        y: box_y,
        width: box_w,
        height: box_h,
    };
    let base = Style::default().fg(palette::INPUT_FG).bg(palette::PANEL_BG);
    let sel_style = Style::default()
        .fg(palette::SELECT_FG)
        .bg(palette::SELECT_BG);
    Clear.render(rect, frame.buffer_mut());
    Block::default()
        .style(base)
        .render(rect, frame.buffer_mut());

    let mut rects = Vec::new();
    for (i, c) in cands.iter().enumerate() {
        let row_y = box_y + i as u16;
        let picked = i == selected;
        let row = if picked { sel_style } else { base };
        let count_style = if picked {
            sel_style
        } else {
            Style::default().fg(palette::META).bg(palette::PANEL_BG)
        };
        let mut spans = vec![Span::styled(" ", row)];
        for (ci, ch) in c.text.chars().enumerate() {
            let mut st = row;
            if let Some((s, e)) = c.hl
                && ci >= s
                && ci < e
            {
                st = st.add_modifier(Modifier::BOLD);
                if !picked {
                    st = st.fg(palette::ACCENT);
                }
            }
            spans.push(Span::styled(ch.to_string(), st));
        }
        let name_len = c.text.chars().count();
        if name_len < name_w {
            spans.push(Span::styled(" ".repeat(name_w - name_len), row));
        }
        spans.push(Span::styled("  ", row));
        let cl = count_label(c.count);
        let cl_w = cl.chars().count();
        if cl_w < count_w {
            spans.push(Span::styled(" ".repeat(count_w - cl_w), count_style));
        }
        spans.push(Span::styled(cl, count_style));
        Paragraph::new(Line::from(spans)).render(
            Rect {
                x: box_x,
                y: row_y,
                width: box_w,
                height: 1,
            },
            frame.buffer_mut(),
        );
        rects.push(Rect {
            x: box_x,
            y: row_y,
            width: box_w,
            height: 1,
        });
    }
    // The key caption, dimmed, on the last row.
    Paragraph::new(Line::from(vec![
        Span::styled(" ", base),
        Span::styled(
            CAPTION,
            Style::default().fg(palette::DIM).bg(palette::PANEL_BG),
        ),
    ]))
    .render(
        Rect {
            x: box_x,
            y: box_y + cands.len() as u16,
            width: box_w,
            height: 1,
        },
        frame.buffer_mut(),
    );
    rects
}

fn render_popup_box(
    frame: &mut Frame,
    title: &str,
    content: Vec<Line<'static>>,
    backdrop: Backdrop,
    fixed_inner_w: Option<usize>,
) -> Rect {
    let area = frame.area();
    let inner_w = fixed_inner_w.unwrap_or_else(|| {
        content
            .iter()
            .map(Line::width)
            .max()
            .unwrap_or(0)
            .max(title.chars().count() + 2)
    });
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
    inner
}

/// A floating popup with a vertically-scrollable `body` and a pinned `footer`
/// row, sized to fit the frame (never taller than it). `scroll` is the first
/// visible body row (clamped internally); returns the maximum valid scroll so the
/// caller can clamp its own offset. When the body overflows, a dim indicator row
/// (range + scroll keys) sits just above the footer.
fn render_scroll_popup(
    frame: &mut Frame,
    title: &str,
    body: &[Line<'static>],
    footer: Line<'static>,
    scroll: usize,
    clickable: &[(usize, KeyEvent)],
) -> (usize, Vec<(Rect, KeyEvent)>) {
    let area = frame.area();
    let panel = Style::default().bg(palette::PANEL_BG);
    let total = body.len();

    // Height first (independent of width): fit the content, but never taller than
    // the frame (1-row margin top+bottom). The footer takes the last inner row;
    // when the body doesn't fit in the rest, reserve one more for the scroll
    // indicator.
    let max_box_h = area.height.saturating_sub(2).max(3);
    let box_h = ((total + 3) as u16).min(max_box_h); // body + footer + 2 borders
    let inner_h = box_h.saturating_sub(2) as usize;
    let overflow = total > inner_h.saturating_sub(1);
    let visible = inner_h.saturating_sub(1 + usize::from(overflow));
    let max_scroll = total.saturating_sub(visible);
    let scroll = scroll.min(max_scroll);
    let indicator = overflow.then(|| {
        format!(
            "↑↓ PgUp/PgDn scroll · {}–{} of {total}",
            scroll + 1,
            scroll + visible
        )
    });

    // Width sizes to the widest of the body, footer, title, and the indicator (so
    // the indicator isn't clipped when the body lines are short).
    let inner_w = body
        .iter()
        .chain(std::iter::once(&footer))
        .map(Line::width)
        .max()
        .unwrap_or(0)
        .max(title.chars().count() + 2)
        .max(
            indicator
                .as_deref()
                .map(str::chars)
                .map_or(0, |c| c.count()),
        );
    let box_w = ((inner_w + 4) as u16).min(area.width); // 2 borders + 2 padding

    let rect = Rect {
        x: area.width.saturating_sub(box_w) / 2,
        y: area.height.saturating_sub(box_h) / 2,
        width: box_w,
        height: box_h,
    };
    Clear.render(rect, frame.buffer_mut());
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

    // Clone only the visible window (not the whole body) so scrolling a large
    // pop-up — e.g. a big safetensors header — stays O(screen), not O(content).
    let window: Vec<Line> = body.iter().skip(scroll).take(visible).cloned().collect();
    Paragraph::new(window).style(panel).render(
        Rect {
            height: visible as u16,
            ..inner
        },
        frame.buffer_mut(),
    );
    if let Some(hint) = indicator {
        Paragraph::new(Line::from(dim_span(hint)))
            .style(panel)
            .render(
                Rect {
                    y: inner.y + visible as u16,
                    height: 1,
                    ..inner
                },
                frame.buffer_mut(),
            );
    }
    Paragraph::new(footer).style(panel).render(
        Rect {
            y: inner.y + inner.height - 1,
            height: 1,
            ..inner
        },
        frame.buffer_mut(),
    );
    // Map each requested body-line index to its on-screen row (when currently
    // visible in the scrolled window), so the caller can hit-test clicks on it.
    let regions: Vec<(Rect, KeyEvent)> = clickable
        .iter()
        .filter(|(idx, _)| *idx >= scroll && *idx < scroll + visible)
        .map(|(idx, key)| {
            let row = Rect {
                y: inner.y + (idx - scroll) as u16,
                height: 1,
                ..inner
            };
            (row, *key)
        })
        .collect();
    (max_scroll, regions)
}

/// Greedy word-wrap of a short help string into lines no wider than `width`.
fn wrap_help(text: &str, width: usize) -> Vec<Line<'static>> {
    let width = width.max(1);
    let mut lines = Vec::new();
    let mut cur = String::new();
    for word in text.split_whitespace() {
        if cur.is_empty() {
            cur.push_str(word);
        } else if cur.chars().count() + 1 + word.chars().count() <= width {
            cur.push(' ');
            cur.push_str(word);
        } else {
            lines.push(Line::from(std::mem::take(&mut cur)));
            cur.push_str(word);
        }
    }
    if !cur.is_empty() {
        lines.push(Line::from(cur));
    }
    lines
}

/// A hover help bubble floated adjacent to `anchor` — just above it, or below
/// when it hugs the top (as the tree's hints do) — with a `border` colour and an
/// optional `title` riding the border (both matching the element it describes).
/// Word-wrapped and clamped on-screen; the caller draws the screen first.
fn render_hover_bubble(
    frame: &mut Frame,
    anchor: Rect,
    border: Color,
    title: Option<&str>,
    help: &str,
) {
    let area = frame.area();
    let wrap_w = 52.min((area.width as usize).saturating_sub(4)).max(8);
    let lines = wrap_help(help, wrap_w);
    if lines.is_empty() {
        return;
    }
    let title_w = title.map(|t| t.chars().count() + 2).unwrap_or(0);
    let inner_w = lines
        .iter()
        .map(Line::width)
        .max()
        .unwrap_or(0)
        .max(title_w);
    let box_w = ((inner_w + 4) as u16).min(area.width); // 2 borders + 2 padding
    let box_h = ((lines.len() + 2) as u16).min(area.height); // 2 borders
    // Prefer just above the anchor; drop below it when there isn't room above.
    let y = if anchor.y >= box_h {
        anchor.y - box_h
    } else {
        (anchor.y + anchor.height).min(area.height.saturating_sub(box_h))
    };
    // Left-align to the anchor, nudged left as needed to stay fully on-screen.
    let x = anchor.x.min(area.width.saturating_sub(box_w));
    let rect = Rect {
        x,
        y,
        width: box_w,
        height: box_h,
    };
    Clear.render(rect, frame.buffer_mut());
    let panel = Style::default().bg(palette::PANEL_BG);
    let mut block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border))
        .padding(Padding::horizontal(1))
        .style(panel);
    if let Some(t) = title {
        block = block.title(Span::styled(
            t.to_string(),
            Style::default().fg(border).add_modifier(Modifier::BOLD),
        ));
    }
    let inner = block.inner(rect);
    block.render(rect, frame.buffer_mut());
    Paragraph::new(lines)
        .style(panel)
        .render(inner, frame.buffer_mut());
}

/// A help bubble for a footer shortcut chip (no title, key-cyan border), floated
/// adjacent to the chip. See [`render_hover_bubble`].
pub fn render_shortcut_bubble(frame: &mut Frame, anchor: Rect, help: &str) {
    render_hover_bubble(frame, anchor, palette::KEY, None, help);
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
    // Clear the two rows first: the band overlays the live data view, whose own
    // footer sits on these same rows. A `Paragraph` only paints its glyphs, so
    // without this the footer bled through past the (often shorter) band text —
    // e.g. the dtype menu left "…irst/last rows" showing behind it.
    Clear.render(
        Rect {
            x: 0,
            y: area.height - 2,
            width: area.width,
            height: 2,
        },
        frame.buffer_mut(),
    );
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

/// The legend pop-up's box title, one per screen.
fn legend_title(legend: Legend) -> &'static str {
    match legend {
        Legend::Tree => "Legend — checkpoint tree",
        Legend::Detail => "Legend — tensor details",
        Legend::Heatmap => "Legend — heatmap",
        Legend::Values => "Legend — numeric values",
        Legend::Rename => "Legend — rename tensors in place",
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
        Legend::Rename => {
            let rows = [
                (
                    Some(palette::ACCENT),
                    "{layer}",
                    "a numbered wildcard — matches any number and copies it into the new name (Tab inserts one)",
                ),
                (
                    None,
                    "12",
                    "a literal number matches only itself — so `…layers.0.…` renames just layer 0",
                ),
                (
                    Some(palette::SUCCESS),
                    "✓",
                    "the rule applies cleanly in place (the header fits the reserved space)",
                ),
                (
                    Some(palette::WARN),
                    "✗ won't fit",
                    "the rewritten header is larger than the shard's reserved space — shorten the new name",
                ),
                (
                    Some(palette::ERROR),
                    "⚠ collide",
                    "two tensors would end up with the same name — the rename is blocked",
                ),
                (
                    Some(palette::ACCENT),
                    "name",
                    "an underlined name is a link: a tensor opens the tree, a shard opens its byte-layout map",
                ),
            ];
            let col = legend_desc_col(&rows, 0);
            for (color, sym, desc) in rows {
                lines.push(legend_row_line(color, sym, desc, col));
            }
            lines.push(legend_row_line(None, "", "", col));
            lines.push(Line::from(dim_span(
                "  Space/: palette · Tab complete · ↑↓ fields · ↵ next field · ←→ caret · ^N add · ^D remove · ^R apply · ^S copy screen · ^Y copy cmd · ^A apply cmd",
            )));
        }
    }

    // Common to every screen: the persistent bottom status-line badges. The rename
    // editor draws its own footer (not the status bar), so skip them there.
    if matches!(legend, Legend::Rename) {
        return lines;
    }
    lines.push(Line::default());
    lines.push(Line::from(vec![
        Span::raw("  "),
        Span::styled(
            " read-only ",
            Style::default()
                .bg(palette::STATUS_BG)
                .fg(palette::SUCCESS)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(
            "      browsing never modifies the checkpoint — repack / convert write a new file",
        ),
    ]));
    lines.push(Line::from(vec![
        Span::raw("  "),
        Span::styled(
            " metadata-only ",
            Style::default()
                .bg(palette::STATUS_BG)
                .fg(palette::WARN)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(
            "  a remote source: only header metadata is loaded; data views need the file locally",
        ),
    ]));
    // The `⚠ health` alert badge only appears on the tree's status line, so it's
    // documented there: orange when the checkpoint has warnings only, red for a
    // real error. Aligned with the badges above.
    if matches!(legend, Legend::Tree) {
        use unicode_width::UnicodeWidthStr;
        let desc_col = 2 + METADATA_BADGE.width() + 2;
        let health = |bg: Color, desc: &str| -> Line<'static> {
            let pad = desc_col.saturating_sub(2 + HEALTH_BADGE.width()).max(1);
            Line::from(vec![
                Span::raw("  "),
                Span::styled(
                    HEALTH_BADGE.to_string(),
                    Style::default()
                        .bg(bg)
                        .fg(palette::STATUS_FG)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(" ".repeat(pad)),
                Span::raw(desc.to_string()),
            ])
        };
        lines.push(health(
            palette::WARN_BG,
            "health: warnings only (e.g. files on disk the index doesn't list)",
        ));
        lines.push(health(
            palette::ALERT,
            "health: an error — a referenced file or tensor is missing on disk",
        ));
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
) -> Vec<(Vec<Seg>, &'static str)> {
    use KeyCode::{Backspace, Down, End, Home, Left, PageDown, PageUp, Right, Up};
    let plain = KeyModifiers::NONE;
    let shift = KeyModifiers::SHIFT;
    // The other representation to switch to (heatmap ⇆ numeric values).
    let switch = if heatmap {
        (vec![Seg::Key("v", hint_key('v'))], "numeric values")
    } else {
        (vec![Seg::Key("m", hint_key('m'))], "heatmap")
    };
    let mut items: Vec<(Vec<Seg>, &str)> = vec![switch];
    let edges = matches!(mode, SampleMode::Edges { .. });
    let window = matches!(mode, SampleMode::Window { .. });
    // In the edges view the arrows rebalance first vs. last (Shift snaps to one
    // end); in the window view they pan the block (Shift a screenful, Ctrl to an
    // edge). Either way slice stepping moves to `[`/`]` so the arrows are free.
    if edges {
        items.push((
            vec![
                Seg::Key("←", KeyEvent::new(Left, plain)),
                Seg::Sep(" "),
                Seg::Key("→", KeyEvent::new(Right, plain)),
            ],
            "first/last cols",
        ));
        items.push((
            vec![
                Seg::Key("↑", KeyEvent::new(Up, plain)),
                Seg::Sep(" "),
                Seg::Key("↓", KeyEvent::new(Down, plain)),
            ],
            "first/last rows",
        ));
        items.push((vec![Seg::Sep("+Shift")], "one end"));
    }
    if window {
        items.push((vec![Seg::Sep("←↑↓→")], "pan"));
        items.push((vec![Seg::Sep("+Shift")], "page"));
        items.push((
            vec![
                Seg::Key("Home", KeyEvent::new(Home, plain)),
                Seg::Sep("/"),
                Seg::Key("End", KeyEvent::new(End, plain)),
            ],
            "col edge",
        ));
        items.push((
            vec![
                Seg::Key("PgUp", KeyEvent::new(PageUp, plain)),
                Seg::Sep("/"),
                Seg::Key("PgDn", KeyEvent::new(PageDown, plain)),
            ],
            "row edge",
        ));
    }
    if slices > 1 {
        if edges || window {
            items.push((
                vec![
                    Seg::Key("[", hint_key('[')),
                    Seg::Sep(" "),
                    Seg::Key("]", hint_key(']')),
                ],
                "slice",
            ));
        } else {
            items.push((
                vec![
                    Seg::Key("←", KeyEvent::new(Left, plain)),
                    Seg::Sep(" "),
                    Seg::Key("→", KeyEvent::new(Right, plain)),
                ],
                "step",
            ));
            items.push((
                vec![
                    Seg::Sep("Shift+"),
                    Seg::Key("←", KeyEvent::new(Left, shift)),
                    Seg::Sep(" "),
                    Seg::Key("→", KeyEvent::new(Right, shift)),
                ],
                "jump 5%",
            ));
        }
        items.push((vec![Seg::Key("/", hint_key('/'))], "index or %"));
    }
    if overridable {
        items.push((vec![Seg::Key("d", hint_key('d'))], "dtype"));
        items.push((vec![Seg::Key("r", hint_key('r'))], "reshape"));
    }
    // Cycle the layout overview → edges → window → overview; the label names the
    // layout `e` switches to next.
    items.push((
        vec![Seg::Key("e", hint_key('e'))],
        match mode {
            SampleMode::Grid => "edges",
            SampleMode::Edges { .. } => "window",
            SampleMode::Window { .. } => "overview",
        },
    ));
    // Cycle the zebra striping / numeral base (numeric grid only).
    if !heatmap {
        items.push((
            vec![Seg::Key("z", hint_key('z'))],
            match stripe {
                StripeMode::Rows => "zebra: rows",
                StripeMode::Cols => "zebra: cols",
                StripeMode::Off => "zebra: off",
            },
        ));
        items.push((
            vec![Seg::Key("b", hint_key('b'))],
            match base {
                NumBase::Decimal => "base: dec",
                NumBase::Hex => "base: hex",
                NumBase::Octal => "base: oct",
                NumBase::Binary => "base: bin",
            },
        ));
    }
    items.push((vec![Seg::Key("c", hint_key('c'))], "copy screen"));
    items.push((vec![Seg::Key("y", hint_key('y'))], "copy cmd"));
    items.push((vec![Seg::Key("l", hint_key('l'))], "legend"));
    items.push((
        vec![
            Seg::Key("Space", hint_key(' ')),
            Seg::Sep("/"),
            Seg::Key(":", hint_key(':')),
        ],
        "commands",
    ));
    items.push((
        vec![
            Seg::Key("⌫", KeyEvent::new(Backspace, plain)),
            Seg::Sep("/"),
            Seg::Key("\\", hint_key('\\')),
        ],
        "back/fwd",
    ));
    items
}

/// Physical lines the data view footer occupies at `width`: the blank spacer row
/// above it plus the wrapped hint line(s). Used to size the grid so the header
/// (tensor name + file) never scrolls off. Shares [`wrap_hint_items`] with
/// [`data_view_footer_wrapped_lines`] so the reservation can't drift from what's
/// drawn.
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
    1 + wrap_hint_items(items, width as u16).0.len().max(1)
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

/// The data-view footer as styled [`Line`]s + clickable chips — the shared
/// ` · `-separated `key label` chip format via [`wrap_hint_items`], identical to
/// every other view's footer (so the coloring / wrapping / hit-testing can't drift).
pub(crate) fn data_view_footer_wrapped_lines(
    mode: SampleMode,
    slices: usize,
    overridable: bool,
    heatmap: bool,
    stripe: StripeMode,
    base: NumBase,
    width: usize,
) -> (Vec<Line<'static>>, Vec<ChipHit>) {
    wrap_hint_items(
        view_footer_items(mode, slices, overridable, heatmap, stripe, base),
        width as u16,
    )
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
/// `{}`/`[]` brackets in the normal foreground — the same contrast as the commas
/// and other punctuation colored_json leaves unstyled — while the colons stay
/// dimmed so key/value separators recede behind the values.
fn json_styler() -> colored_json::Styler {
    let dim = to_yansi(palette::DIM).foreground();
    let bracket = to_yansi(Color::Reset).foreground();
    colored_json::Styler {
        object_brackets: bracket,
        object_colon: dim,
        array_brackets: bracket,
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
pub(crate) fn tree_hint_lines(
    can_repack: bool,
    can_rename: bool,
    width: u16,
) -> (Vec<Line<'static>>, Vec<ChipHit>) {
    use KeyCode::{Backspace, Down, Enter, Left, PageDown, PageUp, Right, Tab, Up};
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
                Seg::Key("PgUp", KeyEvent::new(PageUp, plain)),
                Seg::Sep("/"),
                Seg::Key("PgDn", KeyEvent::new(PageDown, plain)),
            ],
            "page",
        ),
        (vec![Seg::Key("Enter", KeyEvent::new(Enter, plain))], "open"),
        (vec![Seg::Key("Tab", KeyEvent::new(Tab, plain))], "files"),
        (
            vec![
                Seg::Key("Space", hint_key(' ')),
                Seg::Sep("/"),
                Seg::Key(":", hint_key(':')),
            ],
            "commands",
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
        (vec![Seg::Key("h", hint_key('h'))], "health"),
        (vec![Seg::Key("s", hint_key('s'))], "stats"),
        (vec![Seg::Key("c", hint_key('c'))], "copy screen"),
        (vec![Seg::Key("t", hint_key('t'))], "copy tree"),
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
        items.push((vec![Seg::Key("r", hint_key('r'))], "repack"));
    }
    if can_rename {
        items.push((vec![Seg::Key("R", hint_key('R'))], "rename"));
    }
    items.push((vec![Seg::Key("q", hint_key('q'))], "quit"));
    wrap_hint_items(items, width)
}

/// The in-place rename editor's footer hint chips — the same borderless,
/// clickable, hover-aware footer every other view has, opening with the common
/// `Space / :` command-palette chip. The `^N`/`^D`/`^Y` chips synthesize their
/// Ctrl combos; the rest are plain keys the editor loop already handles.
pub(crate) fn rename_hint_lines(
    width: u16,
    applicable: bool,
) -> (Vec<Line<'static>>, Vec<ChipHit>) {
    use KeyCode::{Char, Down, Enter, Esc, Left, PageDown, PageUp, Right, Tab, Up};
    let plain = KeyModifiers::NONE;
    let ctrl = KeyModifiers::CONTROL;
    // The apply chip's label reflects readiness (`^R` is blocked until clean).
    let apply_label = if applicable {
        "apply"
    } else {
        "apply (fix issues)"
    };
    let items: Vec<(Vec<Seg>, &str)> = vec![
        (
            vec![
                Seg::Key("Space", hint_key(' ')),
                Seg::Sep("/"),
                Seg::Key(":", hint_key(':')),
            ],
            "commands",
        ),
        (vec![Seg::Key("Tab", KeyEvent::new(Tab, plain))], "complete"),
        (
            vec![
                Seg::Key("↑", KeyEvent::new(Up, plain)),
                Seg::Sep("/"),
                Seg::Key("↓", KeyEvent::new(Down, plain)),
            ],
            "fields",
        ),
        (
            vec![Seg::Key("↵", KeyEvent::new(Enter, plain))],
            "next field",
        ),
        (
            vec![
                Seg::Key("←", KeyEvent::new(Left, plain)),
                Seg::Sep("/"),
                Seg::Key("→", KeyEvent::new(Right, plain)),
            ],
            "caret",
        ),
        (
            vec![Seg::Key("^N", KeyEvent::new(Char('n'), ctrl))],
            "add rule",
        ),
        (
            vec![Seg::Key("^D", KeyEvent::new(Char('d'), ctrl))],
            "remove",
        ),
        (
            vec![
                Seg::Key("PgUp", KeyEvent::new(PageUp, plain)),
                Seg::Sep("/"),
                Seg::Key("PgDn", KeyEvent::new(PageDown, plain)),
            ],
            "scroll",
        ),
        (
            vec![Seg::Key("^R", KeyEvent::new(Char('r'), ctrl))],
            apply_label,
        ),
        // The universal commands — bare `c`/`l`/`y` type into a field here, so they're
        // the Ctrl keys `^S`/`^L`/`^Y`, mirroring the non-editing modes' `c`/`l`/`y`.
        // `^A` copies the apply (`convert --map`) command.
        (
            vec![Seg::Key("^L", KeyEvent::new(Char('l'), ctrl))],
            "legend",
        ),
        (
            vec![Seg::Key("^S", KeyEvent::new(Char('s'), ctrl))],
            "copy screen",
        ),
        (
            vec![Seg::Key("^Y", KeyEvent::new(Char('y'), ctrl))],
            "copy command",
        ),
        (
            vec![Seg::Key("^A", KeyEvent::new(Char('a'), ctrl))],
            "copy apply cmd",
        ),
        (vec![Seg::Key("Esc", KeyEvent::new(Esc, plain))], "back"),
    ];
    wrap_hint_items(items, width)
}

/// Lay a list of key-hint chips (`key label`, ` · `-separated) into styled
/// [`Line`]s wrapped to `width`, tracking each clickable [`Seg::Key`]'s
/// [`ChipHit`] position. Shared by the tree ([`tree_hint_lines`]) and file
/// ([`files_hint_lines`]) footers so their wrapping and hit-testing match.
fn wrap_hint_items(items: Vec<(Vec<Seg>, &str)>, width: u16) -> (Vec<Line<'static>>, Vec<ChipHit>) {
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

/// The file browser's key-hint line(s), wrapped to `width` like
/// [`tree_hint_lines`] — the same `key label · …` chips and clickable
/// [`ChipHit`]s, for the file-view footer.
pub(crate) fn files_hint_lines(width: u16) -> (Vec<Line<'static>>, Vec<ChipHit>) {
    use KeyCode::{Backspace, Down, Enter, Left, PageDown, PageUp, Right, Tab, Up};
    let plain = KeyModifiers::NONE;
    let items: Vec<(Vec<Seg>, &str)> = vec![
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
            "collapse/expand",
        ),
        (
            vec![
                Seg::Key("PgUp", KeyEvent::new(PageUp, plain)),
                Seg::Sep("/"),
                Seg::Key("PgDn", KeyEvent::new(PageDown, plain)),
            ],
            "page",
        ),
        (
            vec![Seg::Key("Enter", KeyEvent::new(Enter, plain))],
            "open/preview",
        ),
        (
            vec![Seg::Key("Tab", KeyEvent::new(Tab, plain))],
            "tensor tree",
        ),
        (
            vec![
                Seg::Key("Space", hint_key(' ')),
                Seg::Sep("/"),
                Seg::Key(":", hint_key(':')),
            ],
            "commands",
        ),
        (vec![Seg::Key("l", hint_key('l'))], "legend"),
        (vec![Seg::Key("f", hint_key('f'))], "copy path"),
        (vec![Seg::Key("c", hint_key('c'))], "copy screen"),
        (vec![Seg::Key("y", hint_key('y'))], "copy command"),
        (
            vec![
                Seg::Key("⌫", KeyEvent::new(Backspace, plain)),
                Seg::Sep("/"),
                Seg::Key("\\", hint_key('\\')),
            ],
            "back/fwd",
        ),
        (vec![Seg::Key("q", hint_key('q'))], "quit"),
    ];
    wrap_hint_items(items, width)
}

/// The layout map's footer hints (`↑↓ select · ↵ in tree · …`), wrapped to
/// `width` like the tree's, with clickable [`ChipHit`]s.
pub(crate) fn layout_hint_lines(width: u16) -> (Vec<Line<'static>>, Vec<ChipHit>) {
    use KeyCode::{Backspace, Down, Enter, PageDown, PageUp, Tab, Up};
    let plain = KeyModifiers::NONE;
    let items: Vec<(Vec<Seg>, &str)> = vec![
        (
            vec![
                Seg::Key("↑", KeyEvent::new(Up, plain)),
                Seg::Sep("/"),
                Seg::Key("↓", KeyEvent::new(Down, plain)),
            ],
            "select",
        ),
        (
            vec![
                Seg::Key("PgUp", KeyEvent::new(PageUp, plain)),
                Seg::Sep("/"),
                Seg::Key("PgDn", KeyEvent::new(PageDown, plain)),
            ],
            "page",
        ),
        (vec![Seg::Key("↵", KeyEvent::new(Enter, plain))], "in tree"),
        (
            vec![Seg::Key("Tab", KeyEvent::new(Tab, plain))],
            "tensor tree",
        ),
        (
            vec![
                Seg::Key("Space", hint_key(' ')),
                Seg::Sep("/"),
                Seg::Key(":", hint_key(':')),
            ],
            "commands",
        ),
        (vec![Seg::Key("l", hint_key('l'))], "legend"),
        (vec![Seg::Key("c", hint_key('c'))], "copy screen"),
        (vec![Seg::Key("y", hint_key('y'))], "copy command"),
        (
            vec![
                Seg::Key("⌫", KeyEvent::new(Backspace, plain)),
                Seg::Sep("/"),
                Seg::Key("\\", hint_key('\\')),
            ],
            "back/fwd",
        ),
        (vec![Seg::Key("q", hint_key('q'))], "quit"),
    ];
    wrap_hint_items(items, width)
}

/// The band glyph (shaded by the segment's share of the file) and colour for the
/// layout-map strip: the header in the metadata violet, padding dim, tensors in
/// the dtype amber with a fuller block the larger they are — the shading is the
/// map's ASCII "graphic", so a big tensor reads as a solid column.
fn band_style(seg: &crate::safelayout::Segment, total_len: u64) -> (char, Color) {
    use crate::safelayout::SegmentKind;
    match seg.kind {
        SegmentKind::Header => ('█', palette::META),
        SegmentKind::Gap => ('░', palette::DIM),
        SegmentKind::Tensor => {
            let share = seg.len() as f64 / total_len.max(1) as f64;
            let glyph = if share >= 0.10 {
                '█'
            } else if share >= 0.02 {
                '▓'
            } else if share >= 0.005 {
                '▒'
            } else {
                '░'
            };
            (glyph, palette::DTYPE)
        }
    }
}

/// Per-segment band heights for the layout strip: proportional to each segment's
/// share of the file, at least one row each (so every tensor is labelled), summing
/// to a scrollable total. `body_rows` seeds the resolution so a small file fills
/// the viewport while a large one scrolls.
fn band_rows(map: &crate::safelayout::LayoutMap, body_rows: usize) -> Vec<usize> {
    use crate::safelayout::SegmentKind;
    let total_len = map.total_len.max(1) as f64;
    let target = map.segments.len().max(body_rows.max(1));
    map.segments
        .iter()
        .map(|s| {
            let share = s.len() as f64 / total_len;
            let proportional = (share * target as f64).round() as usize;
            match s.kind {
                // The header lists its `__metadata__` tree-like, so give it enough
                // rows to show them (a label row + one per entry) even when its
                // byte share is tiny — as it is for a multi-GB file.
                SegmentKind::Header => proportional.max(1 + map.metadata.len()),
                _ => proportional.max(1),
            }
        })
        .collect()
}

/// One file-browser row as a styled [`Line`]: a directory shows a fold arrow, its
/// name in the accent with a trailing `/`, and a dim size + file count; a file
/// shows a kind marker (a distinct glyph for openable checkpoints), its name
/// coloured by kind, and its dim size. `selected` draws the whole row in the
/// selection colours (via [`tree_span`], shared with the tensor tree).
fn file_row_line(row: &crate::filetree::FileRow, selected: bool) -> Line<'static> {
    use crate::filetree::FileKind;
    let indent = "  ".repeat(row.depth);
    let mut s: Vec<Span> = vec![tree_span(selected, Color::Reset, indent)];
    if row.is_dir {
        let arrow = if row.expanded { "▾" } else { "▸" };
        s.push(tree_span(selected, palette::ACCENT, arrow));
        s.push(tree_span(selected, Color::Reset, " "));
        s.push(tree_span(
            selected,
            palette::ACCENT,
            format!("{}/", row.name),
        ));
        let files = row.files;
        s.push(tree_span(
            selected,
            palette::DIM,
            format!(
                "  {} · {files} {}",
                format_size(row.size as usize),
                if files == 1 { "file" } else { "files" }
            ),
        ));
    } else {
        // A checkpoint gets the tensor glyph (it opens into the tree) and the amber
        // dtype accent; JSON/text/other stay quiet, so the openable ones stand out.
        let (marker, name_color) = match row.kind {
            FileKind::Checkpoint => ("▦", palette::DTYPE),
            FileKind::Json => ("{}", palette::META),
            FileKind::Text => ("·", Color::Reset),
            FileKind::Other => ("·", palette::DIM),
        };
        s.push(tree_span(selected, palette::DIM, marker));
        s.push(tree_span(selected, Color::Reset, " "));
        s.push(tree_span(selected, name_color, row.name.clone()));
        s.push(tree_span(
            selected,
            palette::DIM,
            format!("  {}", format_size(row.size as usize)),
        ));
    }
    Line::from(s)
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
/// The plain text of one tree row (no colour), exactly as [`tree_node_line`]
/// draws it — the shared building block for exporting the tree / a tensor list
/// (`t`, `--print-tree`, `--print-tensors`).
pub fn tree_row_text(
    node: &TreeNode,
    depth: usize,
    unindexed: &HashSet<String>,
    packing_schemas: &HashMap<String, PackingSchema>,
) -> String {
    line_to_text(&tree_row_line(node, depth, unindexed, packing_schemas))
}

/// The styled tree row (the colour the browser draws) — the building block for
/// the export text and the copy-menu preview.
pub fn tree_row_line(
    node: &TreeNode,
    depth: usize,
    unindexed: &HashSet<String>,
    packing_schemas: &HashMap<String, PackingSchema>,
) -> Line<'static> {
    tree_node_line(
        node,
        depth,
        false,
        unindexed,
        packing_schemas,
        MetaDisplay::Full,
    )
}

/// A tensor's row for the flat list: the same coloured fields as the tree, at
/// its full name, but without the leading `·` bullet a flat list doesn't need.
pub fn tensor_list_line(
    info: &TensorInfo,
    unindexed: &HashSet<String>,
    packing_schemas: &HashMap<String, PackingSchema>,
) -> Line<'static> {
    let node = TreeNode::Tensor {
        info: info.clone(),
        label: None,
    };
    without_bullet(tree_node_line(
        &node,
        0,
        false,
        unindexed,
        packing_schemas,
        MetaDisplay::Capped,
    ))
}

/// Drop the leading `·`/unindexed bullet from a depth-0 tensor row: span 0 is the
/// (empty) indent and span 1 is the bullet, so remove it and trim the space that
/// prefixes the name in the following span, leaving the coloured fields intact.
fn without_bullet(line: Line<'static>) -> Line<'static> {
    let mut spans = line.spans;
    if spans.len() >= 2 {
        spans.remove(1);
        if let Some(next) = spans.get_mut(1) {
            next.content = next.content.trim_start().to_string().into();
        }
    }
    Line::from(spans)
}

/// The plain text of a styled line (its span contents concatenated).
fn line_to_text(line: &Line<'static>) -> String {
    line.spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect()
}

/// How a metadata value is rendered in a tree row: capped to keep the live
/// tree's rows short, or in full for exports (`--print-tree`, the `t` preview)
/// where the whole value is wanted.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MetaDisplay {
    Capped,
    Full,
}

fn tree_node_line(
    node: &TreeNode,
    depth: usize,
    selected: bool,
    unindexed: &HashSet<String>,
    packing_schemas: &HashMap<String, PackingSchema>,
    meta: MetaDisplay,
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
            // Exports keep the whole value; the live tree caps it so rows stay short.
            let truncated_value = if meta == MetaDisplay::Full || flat.chars().count() <= 50 {
                flat
            } else {
                let head: String = flat.chars().take(47).collect();
                format!("{head}...")
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
pub fn dim_span(text: impl Into<String>) -> Span<'static> {
    Span::styled(text.into(), Style::default().fg(palette::DIM))
}

/// A bold green span — a "✓ copied" style confirmation, matching the copy
/// flashes elsewhere. Used by the preview pop-up's copy hint.
pub fn success_span(text: impl Into<String>) -> Span<'static> {
    Span::styled(
        text.into(),
        Style::default()
            .fg(palette::SUCCESS)
            .add_modifier(Modifier::BOLD),
    )
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
    width: u16,
) -> (Vec<Line<'static>>, Option<usize>, LinkRegions) {
    let mut lines: Vec<Line> = Vec::new();
    // Link regions in the header (currently just the `File:` path → layout map).
    // The header is rendered at `y = 0`, so a line's index is its screen row.
    let mut links: Vec<(Rect, Link)> = Vec::new();

    lines.push(Line::from(Span::styled(
        "Tensor Details",
        Style::default().fg(palette::ACCENT),
    )));
    lines.push(Line::from(dim_span("─".repeat(width as usize))));
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

    // Wrap the (possibly long, remote scp-style) path over several lines rather
    // than truncating it, so the whole path stays readable. Continuation lines are
    // indented to line up under the path after the "File: " label.
    let prefix = "File: ";
    let indent = " ".repeat(prefix.len());
    let avail = (width as usize).saturating_sub(prefix.len()).max(1);
    let path_chars: Vec<char> = tensor.source_path.chars().collect();
    // A local `.safetensors` shard's path is a link to its byte-layout map (accent
    // + underline, like the other in-app links); a remote / non-safetensors source
    // has no map, so it stays plain.
    let linkable = !crate::remote::is_remote_source(&tensor.source_path)
        && std::path::Path::new(&tensor.source_path)
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"));
    let path_style = if linkable {
        Style::default()
            .fg(palette::ACCENT)
            .add_modifier(Modifier::UNDERLINED)
    } else {
        Style::default()
    };
    if path_chars.is_empty() {
        lines.push(Line::from(dim_span(prefix)));
    } else {
        // `prefix` and `indent` are the same width, so the path always starts at the
        // same column on every (wrapped) line.
        let x = prefix.len() as u16;
        for (i, chunk) in path_chars.chunks(avail).enumerate() {
            let seg: String = chunk.iter().collect();
            let seg_w = seg.chars().count() as u16;
            if linkable {
                links.push((
                    Rect {
                        x,
                        y: lines.len() as u16,
                        width: seg_w,
                        height: 1,
                    },
                    Link::Layout(tensor.source_path.clone()),
                ));
            }
            if i == 0 {
                lines.push(Line::from(vec![
                    dim_span(prefix),
                    Span::styled(seg, path_style),
                ]));
            } else {
                lines.push(Line::from(vec![
                    Span::raw(indent.clone()),
                    Span::styled(seg, path_style),
                ]));
            }
        }
    }
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
            // A remote (`--ssh-read`) source has no local bytes to scan, so don't
            // offer the (non-working) `s` hint — say it's metadata-only instead.
            StatsView::Pending if crate::remote::is_remote_source(&tensor.source_path) => vec![
                dim_span("Statistics: "),
                Span::styled(
                    "unavailable — remote source, metadata-only",
                    Style::default().fg(palette::WARN),
                ),
            ],
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

    (lines, stats_gauge_row, links)
}

/// The detail screen's footer hint chips — the same borderless, ` · `-separated
/// `key label` format (and clickable [`ChipHit`]s) every other view uses, via the
/// shared [`wrap_hint_items`]. `overridable` gates `d`/`r`; `layout` gates `Tab`;
/// `remote` (metadata-only) hides `s` (there's nothing local to scan).
pub(crate) fn detail_footer_lines(
    overridable: bool,
    remote: bool,
    layout: bool,
    width: u16,
) -> (Vec<Line<'static>>, Vec<ChipHit>) {
    use KeyCode::{Backspace, Tab};
    let plain = KeyModifiers::NONE;
    let mut items: Vec<(Vec<Seg>, &str)> = vec![
        (vec![Seg::Key("m", hint_key('m'))], "heatmap"),
        (vec![Seg::Key("v", hint_key('v'))], "values"),
        (vec![Seg::Key("h", hint_key('h'))], "histogram"),
        (vec![Seg::Key("b", hint_key('b'))], "bins"),
    ];
    if !remote {
        items.push((vec![Seg::Key("s", hint_key('s'))], "stats"));
    }
    if overridable {
        items.push((vec![Seg::Key("d", hint_key('d'))], "dtype"));
        items.push((vec![Seg::Key("r", hint_key('r'))], "reshape"));
    }
    if layout {
        items.push((
            vec![Seg::Key("Tab", KeyEvent::new(Tab, plain))],
            "file layout",
        ));
    }
    items.push((
        vec![
            Seg::Key("Space", hint_key(' ')),
            Seg::Sep("/"),
            Seg::Key(":", hint_key(':')),
        ],
        "commands",
    ));
    items.push((vec![Seg::Key("l", hint_key('l'))], "legend"));
    items.push((vec![Seg::Key("c", hint_key('c'))], "copy screen"));
    items.push((vec![Seg::Key("y", hint_key('y'))], "copy command"));
    items.push((
        vec![
            Seg::Key("⌫", KeyEvent::new(Backspace, plain)),
            Seg::Sep("/"),
            Seg::Key("\\", hint_key('\\')),
        ],
        "back/fwd",
    ));
    wrap_hint_items(items, width)
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
fn highlight_json(raw: &str, inline_arrays: bool) -> Option<Vec<String>> {
    let value: serde_json::Value = serde_json::from_str(raw.trim()).ok()?;
    if !value.is_object() && !value.is_array() {
        return None;
    }
    // `colored_json` paints via yansi, whose default condition drops the ANSI
    // codes when stdout isn't a detected TTY (which would also make the result
    // non-deterministic). We render into our own buffer and own the terminal, so
    // force coloring on. `inline_arrays` swaps the layout formatter so a
    // safetensors header's flat arrays (shape / data_offsets) stay on one line.
    yansi::enable();
    let styler = json_styler();
    let on = colored_json::ColorMode::On;
    let pretty = if inline_arrays {
        colored_json::ColoredFormatter::with_styler(ObjectPrettyArrayInline::default(), styler)
            .to_colored_json(&value, on)
            .ok()?
    } else {
        colored_json::ColoredFormatter::with_styler(colored_json::PrettyFormatter::new(), styler)
            .to_colored_json(&value, on)
            .ok()?
    };
    Some(pretty.split('\n').map(str::to_string).collect())
}

/// Pretty-print `raw` JSON with flat scalar arrays inline (like
/// [`highlight_json_lines_inline`]) but **without** colour — as plain Ratatui
/// lines. Far cheaper than the highlighted path (no `colored_json` ANSI + no
/// `ansi-to-tui` parse), so a huge safetensors header renders instantly. Returns
/// `None` for non-JSON.
pub fn plain_json_lines_inline(raw: &str) -> Option<Vec<Line<'static>>> {
    let value: serde_json::Value = serde_json::from_str(raw.trim()).ok()?;
    if !value.is_object() && !value.is_array() {
        return None;
    }
    // `ColorMode::Off` keeps our layout formatter but emits no ANSI codes.
    let pretty = colored_json::ColoredFormatter::with_styler(
        ObjectPrettyArrayInline::default(),
        json_styler(),
    )
    .to_colored_json(&value, colored_json::ColorMode::Off)
    .ok()?;
    Some(
        pretty
            .split('\n')
            .map(|l| Line::from(Span::raw(l.to_string())))
            .collect(),
    )
}

/// [`highlight_json`] parsed back into styled Ratatui lines (via `ansi-to-tui`),
/// or `None` for non-JSON. Shared by the metadata detail view and the copy-menu
/// preview so both show the same `colored_json` palette.
pub fn highlight_json_lines(raw: &str) -> Option<Vec<Line<'static>>> {
    json_to_lines(raw, false)
}

/// Like [`highlight_json_lines`], but flat scalar arrays stay on one line — for
/// the safetensors header preview (`shape` / `data_offsets` read as `[a, b]`).
pub fn highlight_json_lines_inline(raw: &str) -> Option<Vec<Line<'static>>> {
    json_to_lines(raw, true)
}

fn json_to_lines(raw: &str, inline_arrays: bool) -> Option<Vec<Line<'static>>> {
    use ansi_to_tui::IntoText;
    let mut lines = highlight_json(raw, inline_arrays)?
        .join("\n")
        .into_text()
        .ok()?
        .lines;
    // `colored_json`'s resets parse to an explicit `bg = Reset`, which would
    // paint the terminal's default background over a panel (e.g. the copy-menu
    // pop-up). Drop it so each span inherits whatever container draws it.
    for span in lines.iter_mut().flat_map(|line| line.spans.iter_mut()) {
        span.style.bg = None;
    }
    Some(lines)
}

/// A `serde_json` formatter that pretty-prints objects (one key per line) but
/// keeps arrays inline (`[1, 2, 3]`). safetensors headers only contain flat
/// scalar arrays (a tensor's `shape` / `data_offsets`), so this reads far better
/// than the default element-per-line arrays. Fed to `colored_json`, which colours
/// the values while this controls the layout.
#[derive(Default)]
struct ObjectPrettyArrayInline {
    indent: usize,
    has_value: bool,
}

impl serde_json::ser::Formatter for ObjectPrettyArrayInline {
    fn begin_object<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        self.indent += 1;
        self.has_value = false;
        w.write_all(b"{")
    }
    fn end_object<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        self.indent -= 1;
        if self.has_value {
            w.write_all(b"\n")?;
            json_indent(w, self.indent)?;
        }
        w.write_all(b"}")
    }
    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        w.write_all(if first { b"\n" } else { b",\n" })?;
        json_indent(w, self.indent)
    }
    fn begin_object_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        w.write_all(b": ")
    }
    fn end_object_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        let _ = w;
        self.has_value = true;
        Ok(())
    }
    fn begin_array<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        w.write_all(b"[")
    }
    fn end_array<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        w.write_all(b"]")
    }
    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }
    fn end_array_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        let _ = w;
        self.has_value = true;
        Ok(())
    }
}

fn json_indent<W: ?Sized + std::io::Write>(w: &mut W, levels: usize) -> std::io::Result<()> {
    for _ in 0..levels {
        w.write_all(b"  ")?;
    }
    Ok(())
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
    fn command_palette_lists_commands_as_group_colon_title() {
        let rows = vec![
            (
                "c".to_string(),
                "Copy".to_string(),
                "Screen text".to_string(),
                "Copy the whole screen".to_string(),
            ),
            (
                "s".to_string(),
                "View".to_string(),
                "Checkpoint stats".to_string(),
                "Show stats".to_string(),
            ),
        ];
        let out = crate::tui::headless_render(90, 16, |f| {
            UI::render_command_palette(f, "cop", &rows, 0);
        })
        .unwrap();
        assert!(out.contains("Commands"), "titled box:\n{out}");
        // VS Code style: `Group: Title`, with the bound key beside it.
        assert!(out.contains("Copy: Screen text"), "{out}");
        assert!(out.contains("View: Checkpoint stats"), "{out}");
        assert!(out.contains("c  Copy: Screen text"), "key shown:\n{out}");
        // The query is echoed in the input line.
        assert!(out.contains("cop"), "query shown:\n{out}");
    }

    #[test]
    fn file_browser_shows_dirs_files_and_footer() {
        use crate::filetree::{FileKind, FileRow};
        let rows = vec![
            FileRow {
                depth: 0,
                name: "ckpt".into(),
                path: "/ckpt".into(),
                size: 100,
                is_dir: true,
                expanded: true,
                files: 2,
                kind: FileKind::Other,
            },
            FileRow {
                depth: 1,
                name: "model.safetensors".into(),
                path: "/ckpt/model.safetensors".into(),
                size: 90,
                is_dir: false,
                expanded: false,
                files: 0,
                kind: FileKind::Checkpoint,
            },
            FileRow {
                depth: 1,
                name: "config.json".into(),
                path: "/ckpt/config.json".into(),
                size: 10,
                is_dir: false,
                expanded: false,
                files: 0,
                kind: FileKind::Json,
            },
        ];
        let badges = status_badges(AccessBadge::ReadOnly, None, false);
        let out = crate::tui::headless_render(90, 16, |f| {
            UI::render_files(f, "/ckpt", &rows, 1, 0, None, true, &badges, None);
        })
        .unwrap();
        assert!(out.contains("File browser - /ckpt"), "title:\n{out}");
        assert!(out.contains("ckpt/"), "dir row with trailing slash:\n{out}");
        assert!(out.contains("model.safetensors"), "checkpoint row:\n{out}");
        assert!(out.contains("config.json"), "json row:\n{out}");
        // The selected row's path lands on the status bar.
        assert!(
            out.contains("/ckpt/model.safetensors"),
            "status bar path:\n{out}"
        );
        // Footer advertises the Tab toggle back to the tensor tree.
        assert!(out.contains("tensor tree"), "footer hint:\n{out}");
    }

    #[test]
    fn layout_map_renders_summary_and_bands() {
        use crate::safelayout::{LayoutMap, Segment, SegmentKind};
        let seg = |name: &str, dtype: Option<&str>, shape: Vec<usize>, start, end, kind| Segment {
            name: name.to_string(),
            dtype: dtype.map(str::to_string),
            shape,
            start,
            end,
            kind,
        };
        let map = LayoutMap {
            name: "model.safetensors".to_string(),
            total_len: 1_000_000,
            header_len: 200,
            tensor_count: 2,
            metadata: vec![("format".to_string(), "pt".to_string())],
            segments: vec![
                seg(
                    "header (8 B length + JSON metadata)",
                    None,
                    vec![],
                    0,
                    200,
                    SegmentKind::Header,
                ),
                seg(
                    "embed.weight",
                    Some("BF16"),
                    vec![1000, 256],
                    200,
                    800_200,
                    SegmentKind::Tensor,
                ),
                seg(
                    "norm.weight",
                    Some("F32"),
                    vec![256],
                    800_200,
                    1_000_000,
                    SegmentKind::Tensor,
                ),
            ],
        };
        let out = crate::tui::headless_render(90, 24, |f| {
            UI::render_layout(f, &map, 1, 0, None, true);
        })
        .unwrap();
        assert!(out.contains("Layout - model.safetensors"), "title:\n{out}");
        assert!(out.contains("2 tensors"), "summary:\n{out}");
        assert!(out.contains("1 metadata"), "metadata count:\n{out}");
        assert!(out.contains("embed.weight"), "tensor band:\n{out}");
        assert!(out.contains("BF16"), "dtype shown:\n{out}");
        // The header band lists its __metadata__ entries tree-like.
        assert!(out.contains("† format"), "metadata entry shown:\n{out}");
        // Absolute offsets are shown in hex.
        assert!(out.contains("0x00000000"), "header offset:\n{out}");
    }

    #[test]
    fn layout_tensor_band_names_are_tree_links() {
        use crate::safelayout::{LayoutMap, Segment, SegmentKind};
        let seg = |name: &str, dtype: Option<&str>, shape: Vec<usize>, start, end, kind| Segment {
            name: name.to_string(),
            dtype: dtype.map(str::to_string),
            shape,
            start,
            end,
            kind,
        };
        let map = LayoutMap {
            name: "model.safetensors".to_string(),
            total_len: 1_000_000,
            header_len: 200,
            tensor_count: 2,
            metadata: vec![],
            segments: vec![
                seg("header", None, vec![], 0, 200, SegmentKind::Header),
                seg(
                    "embed.weight",
                    Some("BF16"),
                    vec![1000, 256],
                    200,
                    800_200,
                    SegmentKind::Tensor,
                ),
                seg(
                    "norm.weight",
                    Some("F32"),
                    vec![256],
                    800_200,
                    1_000_000,
                    SegmentKind::Tensor,
                ),
            ],
        };
        let mut links = Vec::new();
        crate::tui::headless_render(90, 24, |f| {
            let (_, _, l) = UI::render_layout(f, &map, 1, 0, None, true);
            links = l;
        })
        .unwrap();
        // Each *tensor* band's name is a `Tree` link; the header band is not.
        let targets: Vec<&Link> = links.iter().map(|(_, l)| l).collect();
        assert!(
            targets
                .iter()
                .any(|l| matches!(l, Link::Tree(n) if n == "embed.weight")),
            "embed.weight should link to the tree: {targets:?}"
        );
        assert!(
            targets
                .iter()
                .any(|l| matches!(l, Link::Tree(n) if n == "norm.weight")),
            "norm.weight should link to the tree: {targets:?}"
        );
        assert!(
            !targets
                .iter()
                .any(|l| matches!(l, Link::Tree(n) if n == "header")),
            "the header band is not a tensor link: {targets:?}"
        );
        // The link starts at the name column (after the fixed 20-column prefix).
        assert!(
            links.iter().all(|(r, _)| r.x == 20),
            "name column: {links:?}"
        );
    }

    #[test]
    fn detail_file_path_links_to_the_layout_only_for_local_safetensors() {
        let ti = |source: &str| TensorInfo {
            name: "blk.0.weight".into(),
            dtype: "F32".into(),
            shape: vec![4],
            size_bytes: 16,
            num_elements: 4,
            storage: Storage::Unknown,
            source_path: source.into(),
            layout: Layout::None,
        };
        let links_for = |source: &str| -> LinkRegions {
            let t = ti(source);
            let (_, _, links) = detail_field_lines(
                &t,
                &t.shape,
                ViewDtype::Stored,
                false,
                StatsView::Pending,
                None,
                80,
            );
            links
        };

        // A local `.safetensors` shard's `File:` path links to its layout map.
        let local = links_for("/ckpt/model-00001.safetensors");
        assert_eq!(local.len(), 1, "one File link: {local:?}");
        assert!(
            matches!(&local[0].1, Link::Layout(p) if p == "/ckpt/model-00001.safetensors"),
            "links to the layout: {local:?}"
        );
        // A non-safetensors (or remote) source has no layout map — so no link.
        assert!(links_for("/ckpt/model.gguf").is_empty(), "gguf has no map");
    }

    #[test]
    fn bottom_band_clears_the_rows_it_overlays() {
        // The band (dtype menu / slice / reshape prompts) overlays the live data
        // view, whose footer sits on the same bottom rows. It must clear them, or
        // the footer bleeds through past the shorter band text.
        let out = crate::tui::headless_render(80, 8, |f| {
            let a = f.area();
            let footer = "LEFT_edge_of_footer ······································· TAIL_MARKER";
            Paragraph::new(footer).render(
                Rect {
                    x: 0,
                    y: a.height - 2,
                    width: a.width,
                    height: 1,
                },
                f.buffer_mut(),
            );
            Paragraph::new("SECOND_ROW_MARKER").render(
                Rect {
                    x: 0,
                    y: a.height - 1,
                    width: a.width,
                    height: 1,
                },
                f.buffer_mut(),
            );
            render_bottom_band(f, Line::from("short prompt"), Line::from("short feedback"));
        })
        .unwrap();
        assert!(out.contains("short prompt"), "band prompt shown:\n{out}");
        assert!(
            out.contains("short feedback"),
            "band feedback shown:\n{out}"
        );
        assert!(
            !out.contains("TAIL_MARKER"),
            "footer tail bled through the band:\n{out}"
        );
        assert!(
            !out.contains("SECOND_ROW_MARKER"),
            "footer second row bled through the band:\n{out}"
        );
    }

    #[test]
    fn highlight_json_colors_objects_and_arrays_only() {
        // Non-JSON text and bare scalars fall through to the raw path.
        assert!(highlight_json("just some text", false).is_none());
        assert!(highlight_json("\"a lone string\"", false).is_none());
        assert!(highlight_json("42", false).is_none());

        let raw = r#"{"b":[true,null,"x"],"a":1}"#;
        let lines = highlight_json(raw, false).expect("an object is highlighted");
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
    fn inline_arrays_keep_scalar_arrays_on_one_line() {
        // The safetensors-header variant renders a tensor's shape / data_offsets
        // inline — as they actually appear in the rendered (colored → Ratatui)
        // lines, not just in an isolated formatter.
        let raw = r#"{"w":{"dtype":"BF16","shape":[152064,4096],"data_offsets":[0,16]}}"#;
        let lines = highlight_json_lines_inline(raw).expect("json highlights");
        let text = |l: &Line| -> String { l.spans.iter().map(|s| s.content.as_ref()).collect() };
        // Some single rendered line carries the whole shape inline.
        assert!(
            lines.iter().any(|l| text(l).contains("[152064, 4096]")),
            "shape inline:\n{}",
            lines.iter().map(text).collect::<Vec<_>>().join("\n")
        );
        assert!(
            lines.iter().any(|l| text(l).contains("[0, 16]")),
            "offsets inline"
        );
        // The object is still expanded — dtype and shape land on different lines.
        assert!(lines.len() > 4, "object still multi-line: {}", lines.len());
    }

    #[test]
    fn plain_inline_matches_the_highlighted_layout_without_color() {
        // The fast (uncoloured) path used for large headers keeps arrays inline
        // and produces the same text as the highlighted path, minus the ANSI.
        let raw = r#"{"w":{"dtype":"BF16","shape":[152064,4096],"data_offsets":[0,16]}}"#;
        let text = |ls: &[Line]| {
            ls.iter()
                .map(|l| {
                    l.spans
                        .iter()
                        .map(|s| s.content.as_ref())
                        .collect::<String>()
                })
                .collect::<Vec<_>>()
                .join("\n")
        };
        let plain = plain_json_lines_inline(raw).expect("plain json");
        assert!(
            text(&plain).contains("[152064, 4096]"),
            "shape inline (plain)"
        );
        // No colour: every span uses the default (Reset) foreground.
        assert!(
            plain
                .iter()
                .flat_map(|l| l.spans.iter())
                .all(|s| s.style.fg.is_none() || s.style.fg == Some(Color::Reset)),
            "plain lines carry no colour"
        );
        // Same text as the highlighted variant (stripped of styling).
        assert_eq!(
            text(&plain),
            text(&highlight_json_lines_inline(raw).unwrap())
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

    #[test]
    fn stats_popup_renders_on_disk_section() {
        use crate::stats::{CheckpointStats, DiskUsage, ShardDisk};
        let tensors = vec![TensorInfo {
            name: "w".into(),
            dtype: "F32".into(),
            shape: vec![4],
            size_bytes: 16,
            num_elements: 4,
            storage: Storage::Unknown,
            source_path: "m.safetensors".into(),
            layout: Layout::None,
        }];
        // One shard squeezed 4× among two the filesystem left alone.
        let disk = DiskUsage::from_shards(vec![
            ShardDisk {
                name: "shard-saver.safetensors".into(),
                apparent: 4 << 20,
                allocated: 1 << 20,
            },
            ShardDisk {
                name: "shard-plain.safetensors".into(),
                apparent: 4 << 20,
                allocated: 4 << 20,
            },
        ]);
        let stats = CheckpointStats::compute(&tensors, None, disk);

        // Expanded: *every* shard is listed — the saver and the untouched one.
        let expanded = crate::tui::headless_render(100, 50, |f| {
            UI::render_stats(f, &stats, None, 0, true);
        })
        .unwrap();
        assert!(expanded.contains("On disk (filesystem)"), "{expanded}");
        assert!(expanded.contains("Allocated"), "{expanded}");
        assert!(expanded.contains("shard-saver.safetensors"), "{expanded}");
        assert!(expanded.contains("4.00×"), "{expanded}");
        assert!(expanded.contains("shard-plain.safetensors"), "{expanded}");
        assert!(
            !expanded.contains("shard with no filesystem saving"),
            "{expanded}"
        );

        // Folded (default): the shard list collapses to a single toggle line.
        let folded = crate::tui::headless_render(100, 50, |f| {
            UI::render_stats(f, &stats, None, 0, false);
        })
        .unwrap();
        assert!(folded.contains("per-shard breakdown"), "{folded}");
        assert!(!folded.contains("shard-saver.safetensors"), "{folded}");
    }

    #[test]
    fn tree_scrollbar_geometry_and_mapping() {
        // Everything fits the viewport → no bar.
        assert!(UI::tree_scrollbar(80, 40, false, false, false, 5).is_none());
        // Too narrow for a bar plus content → no bar.
        assert!(UI::tree_scrollbar(1, 40, false, false, false, 999).is_none());

        // Overflow → a bar in the rightmost column, tracking the visible rows.
        let visible = UI::tree_visible_rows(80, 20, false, false, false);
        let sb = UI::tree_scrollbar(80, 20, false, false, false, visible + 50)
            .expect("overflow shows bar");
        assert_eq!(sb.col, 79);
        assert_eq!(sb.rows as usize, visible);
        assert_eq!(sb.max_offset, 50);
        assert_eq!(sb.top as usize, UI::tree_header_rows(false));

        // Track top → offset 0, track bottom → max_offset; outside the track clamps.
        assert_eq!(sb.offset_at(sb.top), 0);
        assert_eq!(sb.offset_at(sb.top + sb.rows - 1), sb.max_offset);
        assert_eq!(sb.offset_at(0), 0);
        assert_eq!(sb.offset_at(sb.top + sb.rows + 99), sb.max_offset);
        // The middle track row maps near the middle offset (25 = max_offset/2).
        // A discrete bar only lands on multiples of one track-step, so the closest
        // position to the true centre is within a step — the tolerance tracks the
        // viewport height rather than assuming a fixed parity.
        let mid = sb.offset_at(sb.top + (sb.rows - 1) / 2);
        let step = (sb.max_offset as f64 / f64::from(sb.rows - 1)).ceil() as i64;
        assert!(
            (mid as i64 - 25).abs() <= step,
            "midpoint offset {mid} ≈ 25 (±{step})"
        );

        // Hit-testing: only the bar's own column, within the track rows.
        assert!(sb.hit(79, sb.top));
        assert!(sb.hit(79, sb.top + sb.rows - 1));
        assert!(!sb.hit(78, sb.top)); // wrong column
        assert!(!sb.hit(79, sb.top + sb.rows)); // just past the track
        assert!(!sb.hit(79, sb.top - 1)); // header row above the track
    }

    #[test]
    fn tree_scrollbar_drawn_only_when_interactive() {
        // A helper config over `nodes`, differing only in the `interactive` gate.
        fn cfg<'a>(
            nodes: &'a [(TreeNode, usize)],
            unindexed: &'a HashSet<String>,
            schemas: &'a HashMap<String, PackingSchema>,
            badges: &'a [Badge],
            interactive: bool,
        ) -> DrawConfig<'a> {
            DrawConfig {
                tree: nodes,
                current_file: "f",
                file_idx: 0,
                total_files: 1,
                selected_idx: 0,
                scroll_offset: 0,
                search_mode: false,
                search_query: "",
                search_cursor: 0,
                status_icon: "▪",
                status_bar: "",
                status_secondary: "",
                can_repack: false,
                can_rename: false,
                unindexed,
                packing_schemas: schemas,
                copied_flash: None,
                interactive,
                badges,
                hovered_badge: None,
            }
        }

        // 40 rows into a 20-row terminal → the tree overflows the viewport.
        let nodes: Vec<(TreeNode, usize)> = (0..40)
            .map(|i| {
                (
                    TreeNode::Metadata {
                        info: MetadataInfo {
                            name: format!("entry_{i}"),
                            value: "v".to_string(),
                            value_type: "str".to_string(),
                        },
                    },
                    0usize,
                )
            })
            .collect();
        let unindexed = HashSet::new();
        let schemas = HashMap::new();
        let badges = status_badges(AccessBadge::ReadOnly, None, false);

        // Interactive: the bar (thumb █ over a │ track) rides the right edge.
        let live = crate::tui::headless_render(80, 20, |f| {
            UI::render_tree(f, &cfg(&nodes, &unindexed, &schemas, &badges, true));
        })
        .unwrap();
        assert!(live.contains('█'), "expected a thumb:\n{live}");
        assert!(live.contains('│'), "expected a track:\n{live}");

        // Non-interactive (headless dump): no bar anywhere.
        let plain = crate::tui::headless_render(80, 20, |f| {
            UI::render_tree(f, &cfg(&nodes, &unindexed, &schemas, &badges, false));
        })
        .unwrap();
        assert!(
            !plain.contains('█') && !plain.contains('│'),
            "headless dump must show no scroll bar:\n{plain}"
        );
    }

    #[test]
    fn shortcut_help_is_context_aware() {
        // The same key means different things on different screens.
        assert_eq!(
            shortcut_help(hint_key('h'), HelpCtx::Tree),
            Some("Run the checkpoint health checks and show the report."),
        );
        assert_eq!(
            shortcut_help(hint_key('h'), HelpCtx::Detail),
            Some("Compute and show the value histogram."),
        );
        // A common key resolves on any screen; an unknown key has no bubble.
        assert!(shortcut_help(hint_key('l'), HelpCtx::Data).is_some());
        assert_eq!(shortcut_help(hint_key('☺'), HelpCtx::Tree), None);
    }

    #[test]
    fn shortcut_bubble_shows_the_help_text() {
        let anchor = Rect {
            x: 4,
            y: 1,
            width: 1,
            height: 1,
        };
        let out = crate::tui::headless_render(80, 20, |f| {
            render_shortcut_bubble(f, anchor, "Expand every group in the tree.");
        })
        .unwrap();
        let plain = strip_ansi_codes(&out);
        assert!(
            plain.contains("Expand every group in the tree."),
            "bubble should show the help text:\n{plain}"
        );
    }

    #[test]
    fn health_badge_sits_by_the_read_only_badge_with_a_hover_bubble() {
        let nodes: Vec<(TreeNode, usize)> = Vec::new();
        let unindexed = HashSet::new();
        let schemas = HashMap::new();
        // access (idx 0) + health (idx 1); hovering the health badge = Some(1).
        let badges = status_badges(AccessBadge::ReadOnly, Some(HealthAlert::Error), false);
        let mk = |hovered_badge: Option<usize>| DrawConfig {
            tree: &nodes,
            current_file: "model",
            file_idx: 0,
            total_files: 1,
            selected_idx: 0,
            scroll_offset: 0,
            search_mode: false,
            search_query: "",
            search_cursor: 0,
            status_icon: "▪",
            status_bar: "model.safetensors",
            status_secondary: "",
            can_repack: false,
            can_rename: false,
            unindexed: &unindexed,
            packing_schemas: &schemas,
            copied_flash: None,
            interactive: true,
            badges: &badges,
            hovered_badge,
        };

        // Not hovering: the short `⚠ health` badge shows on the bottom line, on the
        // same row as `read-only` and to its left — never in the title.
        let out = crate::tui::headless_render(120, 40, |f| {
            UI::render_tree(f, &mk(None));
        })
        .unwrap();
        let plain = strip_ansi_codes(&out);
        let lines: Vec<&str> = plain.lines().collect();
        assert!(
            !lines[0].contains('⚠'),
            "no alert in the title: {:?}",
            lines[0]
        );
        let badge_row = lines
            .iter()
            .find(|l| l.contains("read-only"))
            .expect("the read-only badge renders");
        assert!(
            badge_row.contains("⚠ health"),
            "the health badge should share the read-only line: {badge_row:?}"
        );
        assert!(
            badge_row.find("⚠ health") < badge_row.find("read-only"),
            "the health badge should sit left of read-only: {badge_row:?}"
        );
        // No hover → no help bubble.
        assert!(
            !plain.contains("Index / file mismatch"),
            "bubble only on hover:\n{plain}"
        );

        // Hovering the badge (index 1) floats its help bubble.
        let hovered = crate::tui::headless_render(120, 40, |f| {
            UI::render_tree(f, &mk(Some(1)));
        })
        .unwrap();
        assert!(
            strip_ansi_codes(&hovered).contains("Index / file mismatch"),
            "hovering the health badge should float its help bubble:\n{hovered}"
        );
    }

    #[test]
    fn access_badge_reflects_editability_with_symmetric_padding() {
        for mode in [AccessBadge::ReadOnly, AccessBadge::Editable] {
            // One space of padding on each side (the user flagged an asymmetric chip).
            let label = mode.label();
            assert!(
                label.starts_with(' ')
                    && label.ends_with(' ')
                    && !label.starts_with("  ")
                    && !label.ends_with("  "),
                "{mode:?} label should be symmetrically padded: {label:?}"
            );
        }
        for (mode, word) in [
            (AccessBadge::ReadOnly, "read-only"),
            (AccessBadge::Editable, "editable"),
        ] {
            let badges = status_badges(mode, None, false);
            let out =
                crate::tui::headless_render(120, 6, |f| UI::render_badge_bar(f, &badges, None))
                    .unwrap();
            let plain = strip_ansi_codes(&out);
            let last = plain.lines().last().unwrap_or_default();
            assert!(
                last.trim_end().ends_with(word),
                "{mode:?} should show {word:?}: {last:?}"
            );
        }
        // Hovering the editable badge (index 0) floats its in-place hint.
        let badges = status_badges(AccessBadge::Editable, None, false);
        let hint = strip_ansi_codes(
            &crate::tui::headless_render(120, 12, |f| UI::render_badge_bar(f, &badges, Some(0)))
                .unwrap(),
        );
        assert!(
            hint.contains("in-place") || hint.contains("convert --map"),
            "editable hint should mention the in-place exception:\n{hint}"
        );
    }

    #[test]
    fn render_rename_shows_fields_and_marks_the_diff() {
        let pairs = vec![(
            "model.layers.{layer}.self_attn.q_proj.weight".to_string(),
            "model.layers.{layer}.attn.q_proj.weight".to_string(),
        )];
        let rules = vec![RenameRuleView {
            from: "model.layers.{layer}.self_attn.q_proj.weight".to_string(),
            to: "model.layers.{layer}.attn.q_proj.weight".to_string(),
            total: 48,
            matched: 48,
            ok: 0,
            collide: 0,
            wont_fit: 48,
            invalid: 0,
            shards: vec![crate::rename::ShardFit {
                file: "model.safetensors".to_string(),
                path: "/ckpt/model.safetensors".to_string(),
                current: 512,
                needed: 560,
                tensors: 48,
            }],
        }];
        let view = RenameView {
            root: "/ckpt",
            pairs: &pairs,
            focus_pair: 0,
            on_target: true,
            cursor: 0,
            menu_open: false,
            menu_sel: 0,
            completions: &[],
            rules: &rules,
            total: 48,
            warnings: &[],
            has_index: true,
            applicable: false,
            scroll: 0,
            error: None,
            cli: Some("checkpoint-explorer convert /ckpt --map 'a=>b'"),
            copied: None,
        };
        let mut clicks = Vec::new();
        let out = crate::tui::headless_render(120, 30, |f| {
            let (_, _chips, c, _menu) = UI::render_rename(f, &view);
            clicks = c;
        })
        .unwrap();
        let plain = strip_ansi_codes(&out);
        assert!(plain.contains("Rename tensors in place"), "{plain}");
        assert!(plain.contains("from") && plain.contains("to"), "{plain}");
        // The per-rule schema before→after and its "won't fit" marker.
        assert!(
            plain.contains("model.layers.{layer}.attn.q_proj.weight"),
            "{plain}"
        );
        assert!(plain.contains("won't fit in place"), "{plain}");
        assert!(plain.contains("updates index.json"), "{plain}");
        // The shard name is a clickable region linking to that file's layout.
        assert!(
            clicks.iter().any(|(_, t)| matches!(
                t,
                crate::ui::Link::Layout(p) if p == "/ckpt/model.safetensors"
            )),
            "expected a clickable shard region"
        );
    }

    #[test]
    fn render_rename_dropdown_lists_candidates_with_counts_and_click_targets() {
        let pairs = vec![("q_proj".to_string(), String::new())];
        let cands = vec![
            RenameCompletion {
                text: "model.layers.{layer}.self_attn.q_proj.weight".into(),
                count: 32,
                hl: Some((0, 5)),
            },
            RenameCompletion {
                text: "model.layers.{layer}.self_attn.k_proj.weight".into(),
                count: 32,
                hl: None,
            },
        ];
        let view = RenameView {
            root: "/ckpt",
            pairs: &pairs,
            focus_pair: 0,
            on_target: false,
            cursor: 6,
            menu_open: true,
            menu_sel: 0,
            completions: &cands,
            rules: &[],
            total: 0,
            warnings: &[],
            has_index: false,
            applicable: false,
            scroll: 0,
            error: None,
            cli: None,
            copied: None,
        };
        let mut menu = Vec::new();
        let out = crate::tui::headless_render(120, 30, |f| {
            let (_, _chips, _links, m) = UI::render_rename(f, &view);
            menu = m;
        })
        .unwrap();
        let plain = strip_ansi_codes(&out);
        // Both candidates, the `×N` metadata column, and the key caption are drawn.
        assert!(plain.contains("self_attn.q_proj.weight"), "{plain}");
        assert!(plain.contains("self_attn.k_proj.weight"), "{plain}");
        assert!(plain.contains("×32"), "count column: {plain}");
        assert!(plain.contains("Tab complete"), "key caption: {plain}");
        // One click target per candidate row (the caption row is not clickable).
        assert_eq!(menu.len(), 2, "a click rect per candidate");
    }

    #[test]
    fn render_confirm_popup_shows_summary_and_choices() {
        let body = vec![
            "Rename 3 tensor(s) across 1 shard file(s):".to_string(),
            "Headers are rewritten in place — this cannot be undone.".to_string(),
        ];
        let out = crate::tui::headless_render(90, 20, |f| {
            UI::render_confirm_popup(f, "Apply rename in place?", &body, &["Apply", "Cancel"], 1);
        })
        .unwrap();
        let plain = strip_ansi_codes(&out);
        assert!(plain.contains("Apply rename in place?"), "{plain}");
        assert!(plain.contains("cannot be undone"), "{plain}");
        assert!(
            plain.contains("Apply") && plain.contains("Cancel"),
            "{plain}"
        );
    }

    #[test]
    fn badge_bar_hit_finds_the_badge_under_the_cursor() {
        // access (idx 0, rightmost) + health (idx 1) on a 120×40 frame.
        let badges = status_badges(AccessBadge::ReadOnly, Some(HealthAlert::Error), false);
        let rects = badge_rects(120, 40, &badges);
        let r0 = rects[0].expect("access fits");
        let r1 = rects[1].expect("health fits");
        assert_eq!(r0.y, 39, "on the bottom row");
        assert_eq!(r0.x + r0.width, 120, "access badge is flush right");
        assert!(
            r1.x + r1.width < r0.x,
            "health sits left of access: {r1:?} {r0:?}"
        );
        // A click maps to whichever badge is under it (and misses the row above).
        assert_eq!(UI::badge_bar_hit(120, 40, r0.x, 39, &badges), Some(0));
        assert_eq!(UI::badge_bar_hit(120, 40, r1.x, 39, &badges), Some(1));
        assert_eq!(UI::badge_bar_hit(120, 40, r1.x, 38, &badges), None);
        // Too narrow → nothing fits, nothing hits.
        assert!(badge_rects(8, 40, &badges).iter().all(Option::is_none));
        assert_eq!(UI::badge_bar_hit(8, 40, 4, 39, &badges), None);
    }

    #[test]
    fn badge_bar_lays_out_right_to_left_with_gaps() {
        let (w, h) = (120u16, 40u16);
        // access(0) + health(1) + metadata(2), each a BADGE_GAP left of the previous.
        let badges = status_badges(AccessBadge::ReadOnly, Some(HealthAlert::Error), true);
        let rects: Vec<Rect> = badge_rects(w, h, &badges)
            .into_iter()
            .map(|r| r.expect("fits"))
            .collect();
        assert_eq!(
            rects[0].x + rects[0].width,
            w,
            "rightmost badge is flush right"
        );
        for i in 1..rects.len() {
            assert_eq!(
                rects[i].x + rects[i].width + BADGE_GAP,
                rects[i - 1].x,
                "badge {i} sits a gap left of badge {}",
                i - 1
            );
            assert_eq!(rects[i].y, h - 1);
        }
        // Dropping the metadata badge leaves just the two.
        assert_eq!(
            badge_rects(
                w,
                h,
                &status_badges(AccessBadge::ReadOnly, Some(HealthAlert::Error), false)
            )
            .len(),
            2
        );
    }

    #[test]
    fn scroll_popup_reports_overflow() {
        let body: Vec<Line> = (0..50).map(|i| Line::from(format!("row {i}"))).collect();
        let footer = Line::from("footer");

        // Tall frame: the whole body fits → nothing to scroll.
        let mut fits_max = usize::MAX;
        crate::tui::headless_render(40, 60, |f| {
            fits_max = render_scroll_popup(f, "T", &body, footer.clone(), 0, &[]).0;
        })
        .unwrap();
        assert_eq!(fits_max, 0, "a 50-row body in a 60-row frame fits");

        // Short frame: the body overflows → scrollable, and the indicator shows.
        let mut small_max = 0;
        let out = crate::tui::headless_render(40, 12, |f| {
            small_max = render_scroll_popup(f, "T", &body, footer.clone(), 0, &[]).0;
        })
        .unwrap();
        assert!(small_max > 0, "a 50-row body in a 12-row frame must scroll");
        assert!(
            strip_ansi_codes(&out).contains("of 50"),
            "the overflow indicator shows the total:\n{out}"
        );
    }
}
