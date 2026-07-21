use anyhow::Result;
use crossterm::{
    cursor,
    event::{
        self, Event, KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
    },
    execute,
    terminal::{self, ClearType},
};
use fuzzy_matcher::FuzzyMatcher;
use fuzzy_matcher::skim::SkimMatcherV2;
use std::{
    cell::{Cell, RefCell},
    collections::{BTreeSet, HashMap, HashSet},
    fs::File,
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

use crate::sample::{HistShared, Histogram, PackingSchema, SampleMode, Stats, ViewDtype};

use crate::tree::{MetadataInfo, Storage, TensorInfo, TreeBuilder, TreeNode};
use crate::ui::{DrawConfig, HelpCtx, Legend, NumBase, Overlay, StatsView, StripeMode, UI};
use crate::utils::base64_encode;
use ratatui::text::{Line, Span};

// The data-view layout enum lives in core (frontend-free, so the kernel's
// data-view state can own it); re-exported here so `crate::explorer::DataLayout`
// keeps resolving for the renderers and the CLI.
pub use crate::viewstate::DataLayout;

/// Which screen to jump straight to for a `--tensor` opened from the CLI.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum OpenView {
    /// The tensor detail screen.
    Detail,
    /// The numeric values grid (`v`).
    Values,
    /// The ASCII heatmap (`m`).
    Heatmap,
    /// The tree browser, with the tensor revealed and highlighted (no view
    /// opened) — what `y` copies from the tree (`--tree`).
    Tree,
}

/// A bulk expansion state for the tree browser (`--tree-state`, the `E` / `C`
/// keys). Absent leaves the natural default (root expanded, layers collapsed).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TreeState {
    Expanded,
    Collapsed,
}

impl TreeState {
    /// The `--tree-state` value that names this state.
    pub fn label(self) -> &'static str {
        match self {
            TreeState::Expanded => "expanded",
            TreeState::Collapsed => "collapsed",
        }
    }
}

/// Parse a `--tree-state` value.
pub fn parse_tree_state(s: &str) -> Result<TreeState, String> {
    match s.to_ascii_lowercase().as_str() {
        "expanded" => Ok(TreeState::Expanded),
        "collapsed" => Ok(TreeState::Collapsed),
        other => Err(format!(
            "invalid tree state '{other}' (expected: expanded, collapsed)"
        )),
    }
}

/// Output format for `--print-tree`. (The `t` copy shortcut always uses `Text`.)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum TreeFormat {
    /// The grouped tree as text — one row per node, fully expanded, in the same
    /// layout the browser shows (no viewport limit, no header/footer chrome).
    #[default]
    Text,
    /// A `model.safetensors.index.json`-style object: a `metadata.total_size`
    /// and a `weight_map` of tensor name → its shard file. `-v` adds a `tensors`
    /// block with each tensor's dtype / shape / element count.
    Json,
}

/// How much per-tensor detail the tree export includes; raised by repeating `-v`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TreeDetail {
    /// Text: names + the browser's own fields. JSON: the bare index.json shape.
    Compact,
    /// Text: each tensor row also names its source file. JSON: adds a `tensors`
    /// block (dtype, shape, element count) alongside the `weight_map`.
    Full,
}

impl TreeDetail {
    /// Map a repeated-`-v` count to a detail level (0 → compact, ≥1 → full).
    pub fn from_verbosity(count: u8) -> Self {
        if count == 0 {
            TreeDetail::Compact
        } else {
            TreeDetail::Full
        }
    }
}

/// Which structure an export dumps: the grouped tree or a flat tensor list.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ExportShape {
    Tree,
    Tensors,
}

/// One entry in the `t` copy menu — a (shape, format, detail) combination, i.e.
/// one CLI `--print-tree`/`--print-tensors` [`--format json`] [`-v`] variant.
#[derive(Clone, Copy)]
struct ExportChoice {
    label: &'static str,
    shape: ExportShape,
    format: TreeFormat,
    detail: TreeDetail,
}

/// The eight export variants offered by `t`, one per CLI combination. `+ files`
/// (text) appends each tensor's source file; `+ details` (JSON) adds a
/// per-tensor block/objects — both what `-v` does.
const EXPORT_CHOICES: &[ExportChoice] = {
    use ExportShape::{Tensors, Tree};
    use TreeDetail::{Compact, Full};
    use TreeFormat::{Json, Text};
    &[
        ExportChoice {
            label: "tree · text",
            shape: Tree,
            format: Text,
            detail: Compact,
        },
        ExportChoice {
            label: "tree · text + files",
            shape: Tree,
            format: Text,
            detail: Full,
        },
        ExportChoice {
            label: "tree · JSON (index.json-style)",
            shape: Tree,
            format: Json,
            detail: Compact,
        },
        ExportChoice {
            label: "tree · JSON + tensor details",
            shape: Tree,
            format: Json,
            detail: Full,
        },
        ExportChoice {
            label: "tensors · text",
            shape: Tensors,
            format: Text,
            detail: Compact,
        },
        ExportChoice {
            label: "tensors · text + files",
            shape: Tensors,
            format: Text,
            detail: Full,
        },
        ExportChoice {
            label: "tensors · JSON (names)",
            shape: Tensors,
            format: Json,
            detail: Compact,
        },
        ExportChoice {
            label: "tensors · JSON + details",
            shape: Tensors,
            format: Json,
            detail: Full,
        },
    ]
};

/// How many lines of the highlighted export the `t` menu previews.
const MENU_PREVIEW_LINES: usize = 14;

/// What a `--tensor` / `--metadata` open targets — mutually exclusive by
/// construction (both-set was representable with two `Option`s), with "neither"
/// meaning the sole tensor of a single-tensor checkpoint.
pub enum OpenTarget {
    /// No `--tensor`/`--metadata`: the sole tensor (a single-tensor file — always
    /// the case for `.npy` — needs no flag); ambiguous when there's more than one.
    SoleTensor,
    /// An exact tensor name (`--tensor`).
    Tensor(String),
    /// An exact metadata entry to reveal in the tree (`--metadata`); metadata lives
    /// only in the tree, so there's no separate view.
    Metadata(String),
}

impl OpenTarget {
    /// The explicit tensor name, if `--tensor` named one.
    pub fn tensor(&self) -> Option<&str> {
        match self {
            OpenTarget::Tensor(n) => Some(n),
            _ => None,
        }
    }
    /// The metadata entry name, if `--metadata` named one.
    pub fn metadata(&self) -> Option<&str> {
        match self {
            OpenTarget::Metadata(n) => Some(n),
            _ => None,
        }
    }
}

/// The histogram request (`--histogram` / `--bins N`): a bucket count implies
/// showing the histogram, so it's one enum, not a bool + an `Option` that could
/// disagree (bins set but histogram off).
pub enum HistogramReq {
    Off,
    /// Show it, buckets chosen automatically.
    Auto,
    /// Show it with a fixed bucket count.
    Bins(usize),
}

impl HistogramReq {
    pub fn on(&self) -> bool {
        !matches!(self, HistogramReq::Off)
    }
    pub fn bins(&self) -> Option<usize> {
        match self {
            HistogramReq::Bins(n) => Some(*n),
            _ => None,
        }
    }
}

/// The health-popup request (`--health` / `--health-findings`): the findings
/// detail implies opening the popup, so one 3-state enum, not two bools where
/// `findings` could be set without `health`.
pub enum HealthReq {
    Off,
    Summary,
    Findings,
}

impl HealthReq {
    pub fn wants(&self) -> bool {
        !matches!(self, HealthReq::Off)
    }
    pub fn findings(&self) -> bool {
        matches!(self, HealthReq::Findings)
    }
}

/// The stats-view request (`--stats` / `--stats-shards`): the shard breakdown
/// implies opening the view — one enum, not two bools.
pub enum StatsReq {
    Off,
    Summary,
    Shards,
}

impl StatsReq {
    pub fn wants(&self) -> bool {
        !matches!(self, StatsReq::Off)
    }
    pub fn shards(&self) -> bool {
        matches!(self, StatsReq::Shards)
    }
}

/// A tensor + view to open on startup, from the CLI flags.
pub struct OpenRequest {
    /// What to open (tensor / metadata / the sole tensor).
    pub target: OpenTarget,
    /// Which screen to show.
    pub view: OpenView,
    /// The value-histogram request for the detail screen (the `h`/`b` keys).
    pub histogram: HistogramReq,
    /// Optional dtype reinterpretation to apply first.
    pub dtype: Option<ViewDtype>,
    /// Which data-view layout to force (`--edge`/`--overview`/`--window`);
    /// `None` keeps the session default.
    pub layout: Option<DataLayout>,
    /// The window layout's top-left corner (row, col), from `--window=ROW,COL`.
    pub window_at: Option<(usize, usize)>,
    /// The edges layout's head/tail split (row, col fractions in `0..=1`), from
    /// `--edge=RFRAC,CFRAC`.
    pub edge_split: Option<(f32, f32)>,
    /// Optional zebra-striping mode to apply (numeric grid).
    pub zebra: Option<StripeMode>,
    /// Optional numeral base for the numeric grid (`--base dec/hex/oct/bin`).
    pub base: Option<NumBase>,
    /// Optional starting slice (3D tensors), as a raw `N` or `N%` string
    /// resolved against the tensor's slice count.
    pub slice: Option<String>,
    /// Optional shape override (a reshape with a matching element count), as a
    /// raw string like `10,100` or `-1,768`.
    pub shape: Option<String>,
    /// Start the statistics scan immediately on the detail view.
    pub compute_stats: bool,
    /// Bulk tree expansion (`--tree-state`, the `E` / `C` keys); `None` keeps the
    /// natural default.
    pub tree_state: Option<TreeState>,
    /// Open the tree in search mode with this query (`--search`, the `/` key).
    pub search: Option<String>,
    /// Overlay the requested screen's legend (`--legend`, the `l` key). A
    /// render-time aid (for `--plain` / inspection); not part of `y`'s round-trip
    /// since the legend is a transient overlay you dismiss.
    pub legend: bool,
    /// The health-check popup request (`--health` / `--health-findings`, the `h`
    /// key + its `f` toggle). Round-trips through `y`.
    pub health: HealthReq,
    /// The checkpoint-stats view request (`--stats` / `--stats-shards`, the `s`
    /// key + its `f` toggle). Round-trips through `y`.
    pub stats: StatsReq,
    /// Render the view once and exit without interactive navigation.
    pub exit_after: bool,
    /// Land in the file browser (`--files`, the `Tab` toggle) once the tree is
    /// up. Round-trips through `y`: the file view's `y` copies `… --files`.
    pub files_view: bool,
    /// Open straight into the safetensors layout map for this file (`--layout
    /// PATH`). Round-trips through `y` from the layout view.
    pub layout_file: Option<String>,
    /// Preselect this tensor in the layout map (`--layout-select NAME`), so the
    /// layout view's `y` round-trips the selection.
    pub layout_select: Option<String>,
    /// Open straight into the in-place rename editor (`--rename`, the `R` key).
    /// Round-trips through `y`; only honoured for a local safetensors checkpoint.
    pub rename: bool,
    /// Seed the rename editor's rule pairs (`--rename-rule 'SRC=>TGT'`, repeatable),
    /// each a schema `source => new-name`. What the editor's `y` records.
    pub rename_rules: Vec<String>,
}

/// Which representation a tensor data view renders.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Representation {
    /// ASCII heatmap (`m`).
    Heatmap,
    /// Numeric values grid (`v`).
    Values,
}

/// An open reader for the tensor currently being viewed, kept across redraws so
/// panning / slice-stepping a data view doesn't re-open the file every frame
/// (re-opening dominates the cost and, for HDF5, also discards libhdf5's chunk
/// cache — see the `window_pan_open_cost` benchmark).
struct CachedReader {
    source_path: String,
    name: String,
    reader: Box<dyn crate::sample::TensorReader>,
}

/// A spinner cycled while a statistics scan runs (Braille dots).
const STATS_SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// The program name used when building the copyable CLI commands (`y`).
const PROGRAM: &str = "checkpoint-explorer";

// Data-view header rows above the grid. Summed to size the grid so the header
// (tensor name + file path) and the footer always stay on screen.
/// Tensor name line + source file line.
const HDR_TITLE_ROWS: usize = 2;
/// The dtype / shape / layout line.
const HDR_DTYPE_ROW: usize = 1;
/// The statistics line.
const HDR_STATS_ROW: usize = 1;
/// The slice line (3D tensors only).
const HDR_SLICE_ROW: usize = 1;
/// The blank spacer between the header and the grid.
const HDR_GRID_GAP_ROW: usize = 1;
/// The column-index row (the numeric grid only; the heatmap has none).
const HDR_COLINDEX_ROW: usize = 1;

/// How long the "✓ Copied …" confirmation stays on screen after `c` before it
/// auto-dismisses (it also clears on the next key press).
const COPY_FLASH: std::time::Duration = std::time::Duration::from_secs(2);

/// Two left-clicks on the same tree row within this window count as a double
/// click (which opens it); a lone click just selects it (visible feedback).
const DOUBLE_CLICK: std::time::Duration = std::time::Duration::from_millis(400);

/// What [`Explorer::gather_checkpoint`] returns: tensors, metadata, the parsed
/// `config.json`, the shards' on-disk footprint, and any remote health reports
/// (index/file mismatch — empty for a local read, whose health is gathered up
/// front instead).
type CheckpointParts = (
    Vec<TensorInfo>,
    Vec<MetadataInfo>,
    Option<crate::config::ModelConfig>,
    Option<crate::stats::DiskUsage>,
    Vec<crate::health::HealthReport>,
);

/// The bottom status bar's text: `(icon, primary line, secondary line)`.
type StatusBar = (&'static str, String, String);

/// What the current selection shows and copies, produced together by
/// [`Explorer::describe_selection`] — the single source of truth for the status
/// bar and the `f` copy, so the path shown and the path copied can't diverge.
struct SelectionView {
    /// The status bar `(glyph, primary, secondary)`.
    status: StatusBar,
    /// The path `f` copies (`None` for metadata / an empty selection). Always a
    /// substring of `status.1` for a group, so the two agree.
    copy_path: Option<String>,
}

impl SelectionView {
    /// The empty selection (nothing highlighted / an empty group).
    fn empty() -> Self {
        SelectionView {
            status: ("", String::new(), String::new()),
            copy_path: None,
        }
    }
}
/// The selected node's distinct source files, cached with its key — the
/// selection index, tree length, and search mode (see
/// [`Explorer::selected_source_files`]).
type GroupFilesCache = Option<(usize, usize, bool, std::collections::BTreeSet<String>)>;

/// Rows the tree viewport scrolls per mouse-wheel notch (independent of the
/// selection, like a normal scrollable list).
const WHEEL_STEP: usize = 3;
/// Rows a PageUp/PageDown scrolls the health-report popup body.
const SCROLL_PAGE: usize = 10;
/// Footer rows below the file-browser list (its one-line status bar) — the
/// explorer-side mirror of ui's `FILES_FOOTER_HEIGHT`, for the mouse row
/// hit-test in [`Explorer::run_files`].
const FILES_FOOTER_ROWS: usize = 1;

/// Max gap between two presses of the same navigation key for the second to count
/// as auto-repeat (a held key) rather than a fresh tap — comfortably above the OS
/// repeat interval (~30/s) plus a frame's render, below a human's tap cadence.
const SCROLL_REPEAT_WINDOW: std::time::Duration = std::time::Duration::from_millis(150);

/// Steps to move for a held **PageUp/PageDown** (per screenful), which cover a lot
/// of ground: a short grace at 1:1, then doubling every few repeats for as long as
/// it's held, with no low plateau — so velocity keeps building the longer you hold.
/// The only cap (`1 << 13`) is an overflow guard; a screenful times this already
/// crosses any real tree in a frame, and `move_selection` clamps to the ends.
fn accel_step_page(streak: u32) -> usize {
    let ramp = (streak.saturating_sub(2) / 3).min(13);
    1usize << ramp
}

/// Rows/cols to move for a held **arrow** (per row/column) — deliberately gentler
/// than [`accel_step_page`] so row-by-row movement stays controllable: a longer
/// grace, slower doubling, and a low cap (32) for a brisk-but-not-teleporting top
/// speed. Big jumps are what PageUp/PageDown are for.
fn accel_step_row(streak: u32) -> usize {
    let ramp = (streak.saturating_sub(3) / 4).min(5);
    1usize << ramp
}

/// A statistics scan running on a worker thread for a data view's current
/// `(tensor, view)`. The view stays fully interactive while it runs; the main
/// loop polls [`Self::handle`], caches the result when it lands, and animates the
/// spinner meanwhile. Dropping the job — because the view closed or the dtype
/// changed — cancels the worker at its next block boundary so no work is wasted.
struct ScanJob {
    view: ViewDtype,
    cancel: Arc<AtomicBool>,
    /// Set to make the worker wait between blocks (releasing the file lock) so a
    /// foreground read can run uncontended; cleared to resume where it left off.
    pause: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<Result<Stats, String>>>,
    started: std::time::Instant,
    /// Stored bytes the worker has scanned so far (it bumps this per block), and
    /// the total it will scan (`size_bytes`). Together they drive the progress bar.
    done: Arc<AtomicUsize>,
    total: usize,
}

impl ScanJob {
    /// Fraction of the tensor scanned so far (`0.0..=1.0`), or `None` when the
    /// total is unknown (empty tensor) so the caller shows just the spinner.
    fn progress(&self) -> Option<f64> {
        (self.total > 0)
            .then(|| (self.done.load(Ordering::Relaxed) as f64 / self.total as f64).min(1.0))
    }
}

impl Drop for ScanJob {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

/// The outcome of the reshape prompt (`r`).
enum ReshapeChoice {
    /// Apply this shape override.
    Set(Vec<usize>),
    /// Clear any override (entered empty).
    Clear,
    /// Leave the override unchanged (`Esc`).
    Cancel,
}

/// The outcome of the histogram bin-count prompt (`b`).
enum BinsChoice {
    /// Use this bucket count.
    Set(usize),
    /// Go back to the automatic count (entered empty).
    Clear,
    /// Leave the count unchanged (`Esc`).
    Cancel,
}

/// The last sample a data view rendered, reused when nothing that affects it
/// changed. This keeps the spinner-frame redraws during a stats scan from
/// re-reading (and, for HDF5, re-decompressing) the tensor every frame — those
/// reads block on the scan worker's HDF5 lock, which otherwise lags the UI and
/// lets keystrokes pile up. Keyed by everything the sampled grid depends on.
struct CachedSample {
    key: SampleKey,
    sample: crate::sample::Sample,
}

/// Everything that determines a data view's sampled grid. `max_rows`/`max_cols`
/// fold in the terminal size and (for the numeric grid) the stats-derived cell
/// width, so the key changes — and the grid re-samples once — when the exact
/// stats land.
type SampleKey = (
    String,         // tensor name
    Representation, // heatmap vs numeric grid
    usize,          // slice
    ViewDtype,      // dtype reinterpretation
    SampleMode,     // layout + offsets / tails
    usize,          // max_rows
    usize,          // max_cols
    Vec<usize>,     // effective shape (stored, or a shape override)
);

/// Cache key for a computed histogram: tensor name, view (dtype reinterpretation)
/// and the requested bucket count (`None` = automatic) — a different count caches
/// separately rather than reusing a stale layout.
type HistKey = (String, ViewDtype, Option<usize>);

/// Whether a screen waits for keys or renders once and returns (`--exit`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Interaction {
    Interactive,
    OneShot,
}

/// How an [`OpenRequest`] is being served, which decides how a failure (a tensor
/// that doesn't exist, an ambiguous request, bad `--shape`/`--slice`) is handled:
/// the navigator can recover, but the headless and one-shot modes must surface it
/// as a non-zero exit.
#[derive(Clone, Copy, PartialEq, Eq)]
enum OpenMode {
    /// Interactive navigator: show a recoverable message, wait for a key, then
    /// fall back to the tree. A failure is *not* fatal.
    Interactive,
    /// `--exit`: render the requested screen once. A failure is fatal — it
    /// propagates as an error so the process exits non-zero.
    OneShot,
    /// `--plain` / `--emit-command`: no terminal. Return the screen for the
    /// caller to render; a failure is fatal and reported on stderr.
    Headless,
}

/// Whether the detail view starts the stats scan on open (`--compute-stats`) or
/// leaves it for the user to trigger with `s`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum StatsStart {
    Auto,
    OnDemand,
}

/// How a statistics scan ended: it finished (result cached) or the user pressed
/// a key to abort it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ScanOutcome {
    Completed,
    Cancelled,
}

/// A screen in the navigation history. The tree is the root; opening a tensor
/// pushes a detail screen, and `m`/`v` push a data view.
#[derive(Clone)]
enum Screen {
    Tree,
    /// The file browser: a tree of the checkpoint's directory (files + sizes),
    /// toggled with `Tab`. Its state (selection, scroll, fold) lives on the
    /// [`Explorer`] like the tensor tree's, so the variant carries no fields.
    Files,
    /// The safetensors **layout map** for one file: a scrollable strip of its
    /// byte layout (header + each tensor's span). Opened from the file browser
    /// (`Enter` on a `.safetensors`) or `--layout PATH`. `selected` / `scroll` are
    /// recorded back into the history on leaving, so stepping away and back (e.g.
    /// `Enter` into the tree then Backspace) returns to the same segment.
    Layout {
        path: String,
        selected: usize,
        scroll: usize,
    },
    Detail {
        tensor: String,
        slice: usize,
    },
    Data {
        tensor: String,
        repr: Representation,
        slice: usize,
    },
    /// The in-place **rename** editor: a full-screen mode (opened with `R`) with a
    /// dynamic list of source→new-name rule pairs, live autocomplete, and a
    /// before→after diff preview. Carries its `(source, new-name)` pairs so that
    /// stepping away (e.g. clicking a shard to inspect its layout) and back restores
    /// what was typed.
    Rename {
        pairs: Vec<(String, String)>,
    },
    /// The full-screen **checkpoint stats** view (opened with `s` or `--stats`):
    /// a scrollable report (sizes, params, dtype mix, layers, experts, per-layer
    /// graphs) with the on-disk per-shard breakdown foldable via `f`. Carries the
    /// fold state + scroll so stepping away and back restores it; the reopen
    /// command round-trips it as `--stats` / `--stats-shards`.
    Stats {
        shards_expanded: bool,
        scroll: usize,
    },
}

/// Which live frame stays behind a [`Explorer::float_scroll_popup`] box — the
/// file browser, or a layout map (with the view state needed to redraw it).
enum PopupBackdrop<'a> {
    Files,
    Layout {
        map: &'a crate::safelayout::LayoutMap,
        selected: usize,
        scroll: usize,
    },
}

/// What a screen asks the navigator to do when the user leaves it.
enum Nav {
    /// Go to a new screen (truncates any forward history, then pushes).
    Open(Screen),
    /// Step back / forward through the visited-screen history (Backspace / `\`).
    Back,
    Forward,
    /// Quit the app.
    Quit,
}

/// The declarative facets of an interactive mode that the generic
/// [`Explorer::run_mode`] driver needs — the small, per-mode data that used to be
/// scattered across each hand-rolled `run_*` loop.
struct ModeSpec {
    /// Help / badge / hover context for this screen.
    id: HelpCtx,
    /// Ctrl-C quits the process immediately (detail / data / rename) vs returning
    /// `Nav::Quit` to the navigator (tree / files / layout).
    ctrlc_quits_immediately: bool,
}

/// The result of a mode handling a key (or `on_enter`): stay in the loop, or leave
/// with a [`Nav`].
enum Outcome {
    Stay,
    Leave(Nav),
}

/// What a mode's own mouse handler did with an event the driver didn't consume.
enum MouseOutcome {
    /// Not handled.
    Ignored,
    /// Handled; just redraw.
    Redraw,
    /// Treat it as this keypress (run it through `handle_key`).
    SynthKey(KeyEvent),
    /// Leave the mode.
    Leave(Nav),
}

/// What opening the command palette produced — folds the old per-mode dispatch
/// styles (return-a-`Nav`, synthesize-a-key, mutate-in-place) into one contract.
enum PaletteResult {
    /// Dismissed (Esc / click-off) or handled in place — stay.
    Handled,
    /// Leave the mode.
    Nav(Nav),
    /// Copy the reopen command — the engine does it (identical for every mode), so
    /// the palette's *Copy: Command to reopen this view* matches the `y` key exactly.
    CopyCommand,
    /// Re-feed this key through `handle_key` (the detail / data "synthesize a key" style).
    SynthKey(KeyEvent),
}

/// Whether the mode has live background work (a stats scan) whose spinner needs the
/// event loop to poll on a timeout, vs blocking on input.
enum Bg {
    Idle,
    /// A scan is running — poll on a timeout so its spinner animates.
    #[allow(dead_code)] // constructed once detail/data migrate (Step 4)
    Poll,
}

/// How often the event loop wakes to advance a background scan's spinner.
const SCAN_TICK: std::time::Duration = std::time::Duration::from_millis(80);

/// One interactive screen, driven by the generic [`Explorer::run_mode`]. A mode is
/// a small state struct plus these callbacks; the driver owns all the shared chrome
/// and event plumbing — the command palette, the copied-flash lifecycle, hover
/// bubbles, footer-chip / link / badge clicks, Ctrl-C, and the wrong-keyboard-layout
/// hint — so a mode physically cannot diverge from the others on those.
trait Mode {
    fn spec(&self) -> ModeSpec;
    /// Whether typed letters are field input here (rename, tree-in-search), so the
    /// driver skips the wrong-layout hint and badge-click actions.
    fn accepts_text(&self, _ex: &Explorer) -> bool {
        false
    }
    /// Draw the whole frame (chrome + body); records `self.clickable` / `self.links`.
    fn render_frame(&self, ex: &Explorer, f: &mut ratatui::Frame);
    /// Recompute pre-draw derived state that needs `&mut` (scroll clamping, keeping
    /// the selection visible). Runs before `render_frame` when input has settled.
    fn pre_draw(&mut self, _ex: &mut Explorer, _term: &mut crate::tui::LiveTerminal) {}
    /// One-time setup on entry (lazy build / deferred load / guard); may bail with a
    /// `Nav`. `Result` so it can propagate a load error.
    fn on_enter(
        &mut self,
        _ex: &mut Explorer,
        _term: &mut crate::tui::LiveTerminal,
    ) -> Result<Outcome> {
        Ok(Outcome::Stay)
    }
    /// Whether Space / `:` opens the command palette here. Default yes; the tree
    /// turns it off while searching (so `:` types into the query, Space is ignored).
    fn palette_on_space(&self, _ex: &Explorer) -> bool {
        true
    }
    /// Advance any background job (the detail / data stats scan); returns whether the
    /// loop should poll on a timeout so the spinner animates.
    fn tick_background(&mut self, _ex: &mut Explorer) -> Bg {
        Bg::Idle
    }
    /// Pause / resume a running background scan while input is pending, so a
    /// keypress's own file read isn't stuck behind the scan's block. No-op by default.
    fn set_background_paused(&self, _paused: bool) {}
    /// The in-frame overlay (legend / copied-command / notice), if one is up.
    fn overlay(&self) -> Option<&Overlay> {
        None
    }
    /// Dismiss any in-frame overlay; returns whether one was showing.
    fn dismiss_overlay(&mut self) -> bool {
        false
    }
    /// Open the command palette (Space / `:`) and dispatch the choice.
    fn open_palette(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
    ) -> PaletteResult;
    /// Handle a real or chip-synthesized keypress. `Result` so a handler that has to
    /// finish a deferred load (e.g. reveal a tensor) can propagate an I/O error.
    fn handle_key(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
        key: KeyEvent,
    ) -> Result<Outcome>;
    /// Handle a mouse event the driver didn't consume (rows / band / wheel).
    fn handle_mouse(
        &mut self,
        _ex: &mut Explorer,
        _term: &mut crate::tui::LiveTerminal,
        _m: MouseEvent,
    ) -> MouseOutcome {
        MouseOutcome::Ignored
    }
    /// Scrub this mode's body to a scroll `offset` — the engine calls this when the
    /// user clicks or drags the scroll bar. Default no-op (a mode that reports a
    /// [`crate::ui::VScrollbar`] overrides it to move its own offset).
    fn set_scroll(&mut self, _ex: &mut Explorer, _offset: usize) {}
    /// The screen to record in history for back / forward restore.
    fn residual(&self) -> Screen;
}

/// The file browser ([`Screen::Files`]) as a [`Mode`]: lists the checkpoint's
/// directory (fold with `←`/`→`, `Enter` opens a checkpoint / previews a sidecar),
/// `Tab`/Backspace return to the tree. Its selection/scroll live on [`Explorer`];
/// this holds only the transient click/drag bookkeeping the old `run_files` kept as
/// loop locals.
struct FilesMode {
    /// Last left-click (time + row) for double-click detection.
    last_click: Option<(std::time::Instant, u16)>,
    /// The selection the scroll was last kept-visible for (so a moved selection
    /// re-scrolls once). `usize::MAX` forces the first frame to update.
    last_sel: usize,
}

impl FilesMode {
    fn new() -> Self {
        Self {
            last_click: None,
            last_sel: usize::MAX,
        }
    }
}

impl Mode for FilesMode {
    fn spec(&self) -> ModeSpec {
        ModeSpec {
            id: HelpCtx::Files,
            ctrlc_quits_immediately: false,
        }
    }

    fn render_frame(&self, ex: &Explorer, f: &mut ratatui::Frame) {
        ex.render_files_frame(f, true);
    }

    fn on_enter(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
    ) -> Result<Outcome> {
        // Build the directory tree lazily on first entry, then keep it (fold state
        // persists across `Tab` toggles). Local walks the filesystem; a remote
        // source lists over SFTP / s3 — a listing failure floats an error and
        // drops back rather than showing an empty browser.
        if ex.file_state.tree.is_none() {
            match ex.build_browse_tree() {
                Ok(tree) => {
                    ex.file_state.tree = Some(tree);
                    ex.file_state.rebuild_rows();
                }
                Err(e) => {
                    let body = vec![
                        Line::from(Span::raw("Can't list the checkpoint directory:")),
                        Line::default(),
                        Line::from(crate::ui::dim_span(e)),
                    ];
                    ex.float_scroll_popup(term, "Files", body, PopupBackdrop::Files, None);
                    return Ok(Outcome::Leave(Nav::Back));
                }
            }
        }
        Ok(Outcome::Stay)
    }

    fn pre_draw(&mut self, ex: &mut Explorer, term: &mut crate::tui::LiveTerminal) {
        if let Ok(sz) = term.size() {
            if ex.file_state.selected != self.last_sel {
                ex.update_files_scroll(sz.width, sz.height);
                self.last_sel = ex.file_state.selected;
            }
            let body = UI::files_visible_rows(sz.width, sz.height);
            let total = ex.file_state.rows.len();
            ex.file_state.scroll = ex.file_state.scroll.min(total.saturating_sub(body));
        }
    }

    fn open_palette(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
    ) -> PaletteResult {
        let Some(cmd) = ex.file_command_palette(term) else {
            return PaletteResult::Handled;
        };
        if cmd == FileCmd::CopyCommand {
            return PaletteResult::CopyCommand; // engine-owned, like the `y` key
        }
        if let Some(nav) = ex.run_file_command(cmd, term) {
            return PaletteResult::Nav(nav);
        }
        PaletteResult::Handled
    }

    fn handle_key(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
        key: KeyEvent,
    ) -> Result<Outcome> {
        let total = ex.file_state.rows.len();
        match key.code {
            // Every lettered command dispatches through the registry (like the tree),
            // so key and palette entry can't drift.
            KeyCode::Char(c)
                if !key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
                    && file_command_for_key(c).is_some() =>
            {
                let cmd = file_command_for_key(c).expect("guarded by is_some");
                if let Some(nav) = ex.run_file_command(cmd, term) {
                    return Ok(Outcome::Leave(nav));
                }
            }
            KeyCode::Tab | KeyCode::Backspace => return Ok(Outcome::Leave(Nav::Back)),
            KeyCode::Char('\\') => return Ok(Outcome::Leave(Nav::Forward)),
            KeyCode::Up => {
                let step = ex.held_step(KeyCode::Up, accel_step_row) as i32;
                ex.file_state.move_selection(-step);
            }
            KeyCode::Down => {
                let step = ex.held_step(KeyCode::Down, accel_step_row) as i32;
                ex.file_state.move_selection(step);
            }
            KeyCode::PageUp => {
                let step =
                    (ex.file_page_rows() * ex.held_step(KeyCode::PageUp, accel_step_page)) as i32;
                ex.file_state.move_selection(-step);
            }
            KeyCode::PageDown => {
                let step =
                    (ex.file_page_rows() * ex.held_step(KeyCode::PageDown, accel_step_page)) as i32;
                ex.file_state.move_selection(step);
            }
            KeyCode::Home => ex.file_state.selected = 0,
            KeyCode::End => ex.file_state.selected = total.saturating_sub(1),
            KeyCode::Left => ex.file_state.collapse_or_parent(),
            KeyCode::Right => ex.file_state.expand_or_child(),
            KeyCode::Enter => {
                if let Some(nav) = ex.activate_file_selection() {
                    return Ok(Outcome::Leave(nav));
                }
            }
            _ => {}
        }
        Ok(Outcome::Stay)
    }

    fn handle_mouse(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
        m: MouseEvent,
    ) -> MouseOutcome {
        let Ok(sz) = term.size() else {
            return MouseOutcome::Ignored;
        };
        let (col, row) = (m.column, m.row);
        match m.kind {
            // (Scroll-bar clicks / drags are handled by the engine's `route_mouse`.)
            MouseEventKind::Down(MouseButton::Left) => {
                let body_top = UI::files_header_rows(sz.width) as u16;
                let body_bottom = sz.height.saturating_sub(FILES_FOOTER_ROWS as u16);
                if row >= body_top && row < body_bottom {
                    let idx = ex.file_state.scroll + (row - body_top) as usize;
                    if let Some(fr) = ex.file_state.rows.get(idx).cloned() {
                        // A click on a directory's ▸/▾ twisty (column `2*depth`)
                        // toggles it on a single click.
                        let on_arrow = fr.is_dir() && col == 2 * fr.depth as u16;
                        ex.file_state.selected = idx;
                        if on_arrow {
                            self.last_click = None;
                            ex.activate_file_selection();
                        } else {
                            let double = matches!(
                                self.last_click,
                                Some((t, r)) if r == row && t.elapsed() < DOUBLE_CLICK
                            );
                            if double {
                                self.last_click = None;
                                if let Some(nav) = ex.activate_file_selection() {
                                    return MouseOutcome::Leave(nav);
                                }
                            } else {
                                self.last_click = Some((std::time::Instant::now(), row));
                            }
                        }
                    }
                }
                MouseOutcome::Redraw
            }
            MouseEventKind::ScrollDown => {
                ex.file_state.scroll = ex.file_state.scroll.saturating_add(WHEEL_STEP);
                MouseOutcome::Redraw
            }
            MouseEventKind::ScrollUp => {
                ex.file_state.scroll = ex.file_state.scroll.saturating_sub(WHEEL_STEP);
                MouseOutcome::Redraw
            }
            _ => MouseOutcome::Ignored,
        }
    }

    fn set_scroll(&mut self, ex: &mut Explorer, offset: usize) {
        ex.file_state.scroll = offset;
    }

    fn residual(&self) -> Screen {
        Screen::Files
    }
}

/// The safetensors layout map ([`Screen::Layout`]) as a [`Mode`]: a scrollable
/// vertical strip of one file's byte layout. Its selection/scroll are the drill-down
/// residual (written back to history), and the parsed map lives here for the visit.
struct LayoutMode {
    path: String,
    /// The parsed map, or the parse error to report on entry.
    map: std::result::Result<crate::safelayout::LayoutMap, String>,
    selected: usize,
    scroll: usize,
    scroll_max: usize,
    last_sel: usize,
}

impl LayoutMode {
    /// The map is parsed by [`Explorer::layout_mode`] (which routes local vs
    /// remote), so the mode itself is source-agnostic — it just holds the result.
    fn new(
        path: String,
        map: std::result::Result<crate::safelayout::LayoutMap, String>,
        selected: usize,
        scroll: usize,
    ) -> Self {
        Self {
            path,
            map,
            selected,
            scroll,
            scroll_max: 0,
            last_sel: usize::MAX,
        }
    }

    /// The parsed map — only reached after `on_enter` has bailed on a parse error.
    fn map(&self) -> &crate::safelayout::LayoutMap {
        self.map.as_ref().expect("on_enter leaves on a parse error")
    }
}

impl Mode for LayoutMode {
    fn spec(&self) -> ModeSpec {
        ModeSpec {
            id: HelpCtx::Layout,
            ctrlc_quits_immediately: false,
        }
    }

    fn on_enter(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
    ) -> Result<Outcome> {
        match &self.map {
            Ok(map) => {
                self.selected = self.selected.min(map.segments.len().saturating_sub(1));
                Ok(Outcome::Stay)
            }
            Err(e) => {
                let body = vec![
                    Line::from(Span::raw(format!(
                        "Can't read the layout of {}:",
                        self.path
                    ))),
                    Line::default(),
                    Line::from(crate::ui::dim_span(e.clone())),
                ];
                ex.float_scroll_popup(term, "Layout", body, PopupBackdrop::Files, None);
                Ok(Outcome::Leave(Nav::Back))
            }
        }
    }

    fn render_frame(&self, ex: &Explorer, f: &mut ratatui::Frame) {
        let flash = ex.copied_flash.as_ref().map(|(w, _)| w.clone());
        let (_max, regions, links, vscroll) = UI::render_layout(
            f,
            self.map(),
            self.selected,
            self.scroll,
            flash.as_deref(),
            true,
        );
        *ex.clickable.borrow_mut() = regions;
        *ex.links.borrow_mut() = links; // tensor band name → tree
        *ex.vscrollbar.borrow_mut() = vscroll;
    }

    fn pre_draw(&mut self, _ex: &mut Explorer, term: &mut crate::tui::LiveTerminal) {
        // Compute the scroll bounds from the band layout up front, then snap so the
        // selected band's label row stays visible when the selection moved.
        let Ok(sz) = term.size() else { return };
        let starts = match &self.map {
            Ok(m) => UI::layout_band_starts(m, sz.width, sz.height),
            Err(_) => return,
        };
        let body = UI::layout_visible_rows(sz.width, sz.height);
        let total_rows = starts.last().copied().unwrap_or(0);
        self.scroll_max = total_rows.saturating_sub(body);
        if self.selected != self.last_sel {
            let band_start = starts.get(self.selected).copied().unwrap_or(0);
            if band_start < self.scroll {
                self.scroll = band_start;
            } else if band_start >= self.scroll + body {
                self.scroll = band_start + 1 - body;
            }
            self.last_sel = self.selected;
        }
        self.scroll = self.scroll.min(self.scroll_max);
    }

    fn open_palette(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
    ) -> PaletteResult {
        let map = match &self.map {
            Ok(m) => m,
            Err(_) => return PaletteResult::Handled,
        };
        let Some(cmd) = ex.layout_command_palette(term, map, self.selected, self.scroll) else {
            return PaletteResult::Handled;
        };
        if cmd == LayoutCmd::CopyCommand {
            return PaletteResult::CopyCommand; // engine-owned, like the `y` key
        }
        if let Some(nav) =
            ex.run_layout_command(cmd, &self.path, map, self.selected, self.scroll, term)
        {
            return PaletteResult::Nav(nav);
        }
        PaletteResult::Handled
    }

    fn handle_key(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
        key: KeyEvent,
    ) -> Result<Outcome> {
        let n = self.map().segments.len();
        let move_sel = |sel: usize, delta: i32| -> usize {
            if delta < 0 {
                sel.saturating_sub((-delta) as usize)
            } else {
                (sel + delta as usize).min(n.saturating_sub(1))
            }
        };
        match key.code {
            // Every lettered command dispatches through the registry (`q`/`l`/`c`/`y`)
            // so key and palette entry can't drift.
            KeyCode::Char(ch)
                if !key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
                    && layout_command_for_key(ch).is_some() =>
            {
                let cmd = layout_command_for_key(ch).expect("guarded by is_some");
                if let Some(nav) = ex.run_layout_command(
                    cmd,
                    &self.path,
                    self.map(),
                    self.selected,
                    self.scroll,
                    term,
                ) {
                    return Ok(Outcome::Leave(nav));
                }
            }
            KeyCode::Backspace | KeyCode::Tab | KeyCode::Esc => {
                return Ok(Outcome::Leave(Nav::Back));
            }
            KeyCode::Char('\\') => return Ok(Outcome::Leave(Nav::Forward)),
            KeyCode::Up => {
                let step = ex.held_step(KeyCode::Up, accel_step_row) as i32;
                self.selected = move_sel(self.selected, -step);
            }
            KeyCode::Down => {
                let step = ex.held_step(KeyCode::Down, accel_step_row) as i32;
                self.selected = move_sel(self.selected, step);
            }
            KeyCode::PageUp => {
                let page = ex.layout_page_segments(self.map(), term.size().ok());
                let step = (page * ex.held_step(KeyCode::PageUp, accel_step_page)) as i32;
                self.selected = move_sel(self.selected, -step);
            }
            KeyCode::PageDown => {
                let page = ex.layout_page_segments(self.map(), term.size().ok());
                let step = (page * ex.held_step(KeyCode::PageDown, accel_step_page)) as i32;
                self.selected = move_sel(self.selected, step);
            }
            KeyCode::Home => self.selected = 0,
            KeyCode::End => self.selected = n.saturating_sub(1),
            // Enter on the header previews the raw JSON header; on a tensor it jumps
            // to that tensor's place in the tree.
            KeyCode::Enter => match self.map().segments.get(self.selected).map(|s| &s.kind) {
                Some(crate::safelayout::SegmentKind::Header) => {
                    ex.preview_header_json(
                        term,
                        &self.path,
                        self.map(),
                        self.selected,
                        self.scroll,
                    );
                }
                Some(crate::safelayout::SegmentKind::Tensor { .. }) => {
                    if let Some(nav) = ex.reveal_layout_selection(self.map(), self.selected)? {
                        return Ok(Outcome::Leave(nav));
                    }
                }
                _ => {}
            },
            _ => {}
        }
        Ok(Outcome::Stay)
    }

    fn handle_mouse(
        &mut self,
        _ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
        m: MouseEvent,
    ) -> MouseOutcome {
        let (col, row) = (m.column, m.row);
        match m.kind {
            // A click on a band selects it (link / chip clicks are handled by the
            // driver's route_mouse before this).
            MouseEventKind::Down(MouseButton::Left) => {
                let _ = col;
                if let Ok(sz) = term.size() {
                    let top = UI::layout_header_rows() as u16;
                    let body = UI::layout_visible_rows(sz.width, sz.height);
                    if row >= top && (row as usize) < top as usize + body {
                        let content_row = self.scroll + (row - top) as usize;
                        let starts = UI::layout_band_starts(self.map(), sz.width, sz.height);
                        if let Some(seg) = starts
                            .windows(2)
                            .position(|w| content_row >= w[0] && content_row < w[1])
                        {
                            let n = self.map().segments.len();
                            self.selected = seg.min(n.saturating_sub(1));
                        }
                    }
                }
                MouseOutcome::Redraw
            }
            MouseEventKind::ScrollDown => {
                self.scroll = (self.scroll + WHEEL_STEP).min(self.scroll_max);
                MouseOutcome::Redraw
            }
            MouseEventKind::ScrollUp => {
                self.scroll = self.scroll.saturating_sub(WHEEL_STEP);
                MouseOutcome::Redraw
            }
            _ => MouseOutcome::Ignored,
        }
    }

    fn set_scroll(&mut self, _ex: &mut Explorer, offset: usize) {
        self.scroll = offset;
    }

    fn residual(&self) -> Screen {
        Screen::Layout {
            path: self.path.clone(),
            selected: self.selected,
            scroll: self.scroll,
        }
    }
}

/// The tensor tree ([`Screen::Tree`]) as a [`Mode`] — the root browser, including
/// the search sub-machine. Its selection/scroll/search state live on [`Explorer`];
/// this holds only the transient click/drag bookkeeping.
struct TreeMode {
    last_click: Option<(std::time::Instant, u16)>,
    last_sel: usize,
}

impl TreeMode {
    fn new() -> Self {
        Self {
            last_click: None,
            last_sel: usize::MAX,
        }
    }
}

impl Mode for TreeMode {
    fn spec(&self) -> ModeSpec {
        ModeSpec {
            id: HelpCtx::Tree,
            ctrlc_quits_immediately: false,
        }
    }

    // While searching, typed letters edit the query — skip the wrong-layout hint,
    // the badge-click actions, and the Space/`:` palette trigger.
    fn accepts_text(&self, ex: &Explorer) -> bool {
        ex.tree_state.search_mode()
    }
    fn palette_on_space(&self, ex: &Explorer) -> bool {
        !ex.tree_state.search_mode()
    }

    fn render_frame(&self, ex: &Explorer, f: &mut ratatui::Frame) {
        ex.render_tree_frame(f, true);
    }

    fn on_enter(
        &mut self,
        ex: &mut Explorer,
        _term: &mut crate::tui::LiveTerminal,
    ) -> Result<Outcome> {
        // The browser needs the whole checkpoint; finish a deferred `--tensor` load.
        ex.ensure_full_load()?;
        Ok(Outcome::Stay)
    }

    fn pre_draw(&mut self, ex: &mut Explorer, term: &mut crate::tui::LiveTerminal) {
        if let Ok(sz) = term.size() {
            if ex.tree_state.selected != self.last_sel {
                ex.update_tree_scroll(sz.width, sz.height); // snap to the moved selection
                self.last_sel = ex.tree_state.selected;
            }
            let body = UI::tree_visible_rows(
                sz.width,
                sz.height,
                ex.tree_state.search_mode(),
                ex.can_repack(),
                ex.can_rename(),
            );
            let total = ex.current_tree_len();
            ex.tree_state.scroll = ex.tree_state.scroll.min(total.saturating_sub(body));
        }
    }

    fn open_palette(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
    ) -> PaletteResult {
        let Some(cmd) = ex.command_palette(term) else {
            return PaletteResult::Handled;
        };
        if cmd == Cmd::CopyCommand {
            return PaletteResult::CopyCommand; // engine-owned, like the `y` key
        }
        if let Some(nav) = ex.run_command(cmd, term) {
            return PaletteResult::Nav(nav);
        }
        PaletteResult::Handled
    }

    fn handle_key(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
        key: KeyEvent,
    ) -> Result<Outcome> {
        match key {
            // Every tree command dispatches through the registry (the same path the
            // palette uses). In search mode the letters fall through to the query.
            KeyEvent {
                code: KeyCode::Char(c),
                modifiers,
                ..
            } if !ex.tree_state.search_mode()
                && !modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
                && tree_command_for_key(c).is_some() =>
            {
                let cmd = tree_command_for_key(c).expect("guarded by is_some");
                if let Some(nav) = ex.run_command(cmd, term) {
                    return Ok(Outcome::Leave(nav));
                }
            }
            // '/' is ignored rather than typed into the query.
            KeyEvent {
                code: KeyCode::Char('/'),
                ..
            } => {}
            KeyEvent {
                code: KeyCode::Esc, ..
            } if ex.tree_state.search_mode() => ex.exit_search_mode(),
            // Shift+↑/↓ jump to the previous/next sibling — before the plain arrows.
            KeyEvent {
                code: KeyCode::Up,
                modifiers: KeyModifiers::SHIFT,
                ..
            } => ex.tree_state.move_to_sibling(false),
            KeyEvent {
                code: KeyCode::Down,
                modifiers: KeyModifiers::SHIFT,
                ..
            } => ex.tree_state.move_to_sibling(true),
            KeyEvent {
                code: KeyCode::Up, ..
            } => {
                let step = ex.held_step(KeyCode::Up, accel_step_row) as i32;
                ex.tree_state.move_selection(-step);
            }
            KeyEvent {
                code: KeyCode::Down,
                ..
            } => {
                let step = ex.held_step(KeyCode::Down, accel_step_row) as i32;
                ex.tree_state.move_selection(step);
            }
            // While searching, ←/→ move the query caret (Shift = start/end).
            KeyEvent {
                code: KeyCode::Left,
                modifiers: KeyModifiers::SHIFT,
                ..
            } if ex.tree_state.search_mode() => ex.tree_state.search_cursor_home(),
            KeyEvent {
                code: KeyCode::Right,
                modifiers: KeyModifiers::SHIFT,
                ..
            } if ex.tree_state.search_mode() => ex.tree_state.search_cursor_end(),
            KeyEvent {
                code: KeyCode::Left,
                ..
            } if ex.tree_state.search_mode() => ex.tree_state.search_cursor_left(),
            KeyEvent {
                code: KeyCode::Right,
                ..
            } if ex.tree_state.search_mode() => ex.tree_state.search_cursor_right(),
            KeyEvent {
                code: KeyCode::Home,
                ..
            } if ex.tree_state.search_mode() => ex.tree_state.selected = 0,
            KeyEvent {
                code: KeyCode::End, ..
            } if ex.tree_state.search_mode() => {
                ex.tree_state.selected = ex.tree_state.visible().len().saturating_sub(1);
            }
            KeyEvent {
                code: KeyCode::PageUp,
                ..
            } => {
                let step = (ex.page_rows() * ex.held_step(KeyCode::PageUp, accel_step_page)) as i32;
                ex.tree_state.move_selection(-step);
            }
            KeyEvent {
                code: KeyCode::PageDown,
                ..
            } => {
                let step =
                    (ex.page_rows() * ex.held_step(KeyCode::PageDown, accel_step_page)) as i32;
                ex.tree_state.move_selection(step);
            }
            // ← jumps to the parent group; → enters the group's first child.
            KeyEvent {
                code: KeyCode::Left,
                ..
            } => ex.tree_state.move_to_parent(),
            KeyEvent {
                code: KeyCode::Right,
                ..
            } => ex.tree_state.move_to_first_child(),
            // While searching, Tab reveals the highlighted result in the tree
            // (leaving search); otherwise Tab toggles to the file browser.
            KeyEvent {
                code: KeyCode::Tab, ..
            } if ex.tree_state.search_mode() => ex.reveal_search_result(),
            KeyEvent {
                code: KeyCode::Tab, ..
            } => {
                if let Some(nav) = ex.run_command(Cmd::ViewFiles, term) {
                    return Ok(Outcome::Leave(nav));
                }
            }
            // Enter acts on the highlighted row: expand a group, or open a tensor.
            KeyEvent {
                code: KeyCode::Enter,
                ..
            } => {
                if let Some(nav) = ex.activate_selection() {
                    return Ok(Outcome::Leave(nav));
                }
            }
            // While searching, Space is ignored rather than typed into the query.
            KeyEvent {
                code: KeyCode::Char(' '),
                ..
            } => {}
            // Backspace edits the query while searching, else steps back.
            KeyEvent {
                code: KeyCode::Backspace,
                ..
            } if ex.tree_state.search_mode() => ex.search_backspace(),
            KeyEvent {
                code: KeyCode::Backspace,
                ..
            } => return Ok(Outcome::Leave(Nav::Back)),
            KeyEvent {
                code: KeyCode::Char('\\'),
                ..
            } if !ex.tree_state.search_mode() => return Ok(Outcome::Leave(Nav::Forward)),
            // Any other char while searching is inserted into the query.
            KeyEvent {
                code: KeyCode::Char(c),
                ..
            } if ex.tree_state.search_mode() => ex.search_insert(c),
            _ => {}
        }
        Ok(Outcome::Stay)
    }

    fn handle_mouse(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
        m: MouseEvent,
    ) -> MouseOutcome {
        let Ok(sz) = term.size() else {
            return MouseOutcome::Ignored;
        };
        let (col, row) = (m.column, m.row);
        match m.kind {
            // (Scroll-bar clicks / drags are handled by the engine's `route_mouse`.)
            MouseEventKind::Down(MouseButton::Left) => {
                let body_top = UI::tree_header_rows(ex.tree_state.search_mode()) as u16;
                // Body ends above the bottom-pinned hint footer + the 2-line status bar.
                let hint_rows = UI::tree_hint_rows(
                    sz.width,
                    ex.tree_state.search_mode(),
                    ex.can_repack(),
                    ex.can_rename(),
                ) as u16;
                let body_bottom = sz.height.saturating_sub(2 + hint_rows);
                if row >= body_top && row < body_bottom {
                    let idx = ex.tree_state.scroll + (row - body_top) as usize;
                    if idx < ex.current_tree_len() {
                        // A click exactly on a group's ▸/▾ twisty (column `2*depth`)
                        // toggles it on a single click.
                        let on_arrow = {
                            let tree = ex.tree_state.visible();
                            matches!(
                                tree.get(idx),
                                Some((TreeNode::Group { .. }, depth)) if col == 2 * *depth as u16
                            )
                        };
                        ex.tree_state.selected = idx;
                        if on_arrow {
                            self.last_click = None;
                            ex.activate_selection();
                        } else {
                            let double = matches!(
                                self.last_click,
                                Some((t, r)) if r == row && t.elapsed() < DOUBLE_CLICK
                            );
                            if double {
                                self.last_click = None;
                                if ex.tree_state.search_mode() {
                                    ex.reveal_search_result();
                                } else if let Some(nav) = ex.activate_selection() {
                                    return MouseOutcome::Leave(nav);
                                }
                            } else {
                                self.last_click = Some((std::time::Instant::now(), row));
                            }
                        }
                    }
                }
                MouseOutcome::Redraw
            }
            MouseEventKind::ScrollDown => {
                ex.tree_state.scroll = ex.tree_state.scroll.saturating_add(WHEEL_STEP);
                MouseOutcome::Redraw
            }
            MouseEventKind::ScrollUp => {
                ex.tree_state.scroll = ex.tree_state.scroll.saturating_sub(WHEEL_STEP);
                MouseOutcome::Redraw
            }
            _ => MouseOutcome::Ignored,
        }
    }

    fn set_scroll(&mut self, ex: &mut Explorer, offset: usize) {
        ex.tree_state.scroll = offset;
    }

    fn residual(&self) -> Screen {
        Screen::Tree
    }
}

/// The in-place rename editor ([`Screen::Rename`]) as a [`Mode`]. Owns the editor
/// model plus the cached shard headers and the dirty-gated preview (so pure caret /
/// focus moves don't re-scan the checkpoint). `scroll_max` is a `Cell` because it's
/// learned during the (`&self`) draw and read back by key/mouse handling.
struct RenameMode2 {
    /// Seed pairs from a prior visit / `--rename-rule`, consumed by `on_enter`.
    saved_pairs: Vec<(String, String)>,
    target: std::path::PathBuf,
    loaded: Option<crate::rename::Loaded>,
    /// The deduped generalized schemas the autocomplete offers, each with the count
    /// of tensors it covers (the dropdown's `×N` column).
    schemas: Vec<(String, usize)>,
    root: String,
    editor: RenameMode,
    /// What was last copied (the `✓ copied …` flash), cleared on the next key.
    copied: Option<&'static str>,
    /// The autocomplete dropdown's row rects from the last frame, so a click can
    /// accept the candidate under the cursor.
    menu_rects: std::cell::RefCell<Vec<ratatui::layout::Rect>>,
    // Derived, recomputed only when the rule pairs change (`dirty`).
    rules_view: Vec<crate::ui::RenameRuleView>,
    total: usize,
    warnings: Vec<String>,
    has_index: bool,
    applicable: bool,
    synth_err: Option<String>,
    cli: Option<String>,
    dirty: bool,
    scroll_max: std::cell::Cell<usize>,
    /// Set once a rename is applied — the rules are spent, so `residual` clears them.
    applied: bool,
}

impl RenameMode2 {
    fn new(saved_pairs: Vec<(String, String)>) -> Self {
        Self {
            saved_pairs,
            target: std::path::PathBuf::new(),
            loaded: None,
            schemas: Vec::new(),
            root: String::new(),
            editor: RenameMode::default(),
            copied: None,
            menu_rects: std::cell::RefCell::new(Vec::new()),
            rules_view: Vec::new(),
            total: 0,
            warnings: Vec::new(),
            has_index: false,
            applicable: false,
            synth_err: None,
            cli: None,
            dirty: true,
            scroll_max: std::cell::Cell::new(0),
            applied: false,
        }
    }

    fn loaded(&self) -> &crate::rename::Loaded {
        self.loaded.as_ref().expect("on_enter loads or leaves")
    }

    /// The current rules to persist / restore (dropping fully-blank pairs).
    fn pairs(&self) -> Vec<(String, String)> {
        self.editor
            .pairs
            .iter()
            .filter(|p| !(p.source.trim().is_empty() && p.target.trim().is_empty()))
            .map(|p| (p.source.clone(), p.target.clone()))
            .collect()
    }

    fn do_copy_apply(&mut self) {
        self.copied = (self.cli.is_some()
            && copy_to_clipboard(self.cli.as_deref().unwrap_or_default()))
        .then_some("the apply command");
    }

    fn do_copy_screen(&mut self) {
        let text = crate::tui::headless_render(120, 40, |f| {
            let _ = draw_rename_frame(
                f,
                &self.root,
                &self.editor,
                &self.schemas,
                &self.rules_view,
                self.total,
                &self.warnings,
                self.has_index,
                self.applicable,
                &self.synth_err,
                self.cli.as_deref(),
                None,
            );
        });
        if let Ok(text) = text {
            self.copied = copy_to_clipboard(&text).then_some("the screen text");
        }
    }

    /// Apply the rename (`R`): flash why it can't yet if it isn't clean, else float a
    /// confirmation modal and — only on an explicit confirm — rewrite the files.
    /// Returns `Some(nav)` to leave the editor once applied. Shared by the `R` key
    /// and the palette's *Apply* command.
    fn try_apply(&mut self, ex: &mut Explorer, term: &mut crate::tui::LiveTerminal) -> Option<Nav> {
        if !self.applicable {
            self.editor.error =
                Some("can't apply yet — fix the blocked rows / warnings above".to_string());
            return None;
        }
        if !self.confirm_apply(term) {
            return None;
        }
        match ex.apply_rename_mode(self.loaded(), &self.editor) {
            Ok(nav) => {
                self.applied = true; // rules spent → residual clears them
                Some(nav)
            }
            Err(e) => {
                self.editor.error = Some(e);
                None
            }
        }
    }

    /// Float the apply-confirmation modal over the live editor: a summary of what
    /// will change (from [`crate::rename::Plan::summary_lines`]) plus an
    /// `[Apply] [Cancel]` choice. Returns `true` only on an explicit confirm
    /// (`Enter` on *Apply*, or `Y`); `Esc` / `N` / *Cancel* return `false`.
    fn confirm_apply(&self, term: &mut crate::tui::LiveTerminal) -> bool {
        let fallback = || vec!["Apply the entered renames in place?".to_string()];
        let summary = match self.editor.build_map() {
            Ok((map, _)) => match self.loaded().plan(&map) {
                Ok(plan) => plan.summary_lines(8),
                Err(_) => fallback(),
            },
            Err(_) => fallback(),
        };
        let mut idx = 1usize; // default to the safe choice (Cancel)
        loop {
            if term
                .draw(|f| {
                    let _ = draw_rename_frame(
                        f,
                        &self.root,
                        &self.editor,
                        &self.schemas,
                        &self.rules_view,
                        self.total,
                        &self.warnings,
                        self.has_index,
                        self.applicable,
                        &self.synth_err,
                        self.cli.as_deref(),
                        self.copied,
                    );
                    UI::render_confirm_popup(
                        f,
                        "Apply rename in place?",
                        &summary,
                        &["Apply", "Cancel"],
                        idx,
                    );
                })
                .is_err()
            {
                return false;
            }
            match event::read() {
                Ok(Event::Key(key)) if is_ctrl_c(&key) => quit_immediately(),
                Ok(Event::Key(KeyEvent { code, .. })) => match code {
                    KeyCode::Left | KeyCode::Right | KeyCode::Tab => idx = 1 - idx,
                    KeyCode::Char('y' | 'Y') => return true,
                    KeyCode::Char('n' | 'N') => return false,
                    KeyCode::Enter => return idx == 0,
                    KeyCode::Esc => return false,
                    _ => {}
                },
                Ok(_) => {} // other mouse / resize: redraw
                Err(_) => return false,
            }
        }
    }
}

impl Mode for RenameMode2 {
    fn spec(&self) -> ModeSpec {
        ModeSpec {
            id: HelpCtx::Rename,
            ctrlc_quits_immediately: true,
        }
    }

    // The name fields take arbitrary text; skip the wrong-layout hint. Space / `:`
    // still open the palette (a tensor-name schema never contains either).
    fn accepts_text(&self, _ex: &Explorer) -> bool {
        true
    }

    fn on_enter(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
    ) -> Result<Outcome> {
        let Some(target) = ex.rename_target() else {
            return Ok(Outcome::Leave(Nav::Back)); // gated; bail safely if it slips
        };
        // Read every shard header once, so the preview is instant as the user types.
        let loaded = match crate::rename::load(&target) {
            Ok(l) => l,
            Err(e) => {
                let msg = format!("Cannot open the rename editor: {e:#}");
                ex.float_until_dismissed(term, |f| {
                    ex.render_tree_frame(f, true);
                    UI::render_notice(f, &msg);
                });
                return Ok(Outcome::Leave(Nav::Back));
            }
        };
        // Autocomplete over the deduped *generalized* schemas (one per tensor
        // family), each tagged with how many tensors it covers (the `×N` column).
        // One `generalize` per name (it's the hot part of opening a big checkpoint),
        // keeping first-seen order.
        let mut counts: HashMap<String, usize> = HashMap::new();
        let mut order: Vec<String> = Vec::new();
        for n in loaded.names() {
            let schema = crate::rename::generalize(n).0;
            let seen = counts.entry(schema.clone()).or_insert(0);
            if *seen == 0 {
                order.push(schema);
            }
            *seen += 1;
        }
        self.schemas = order
            .into_iter()
            .map(|s| {
                let c = counts[&s];
                (s, c)
            })
            .collect();
        self.root = loaded.root().display().to_string();
        if !self.saved_pairs.is_empty() {
            self.editor.pairs = std::mem::take(&mut self.saved_pairs)
                .into_iter()
                .map(|(source, target)| RenamePair { source, target })
                .collect();
        }
        self.target = target;
        self.loaded = Some(loaded);
        Ok(Outcome::Stay)
    }

    fn pre_draw(&mut self, ex: &mut Explorer, _term: &mut crate::tui::LiveTerminal) {
        if self.dirty {
            // Compute into locals first, then assign (so the `loaded` borrow ends
            // before the `&mut self` field writes).
            let (warnings, has_index, applicable, err, cli, rules_view, total) = {
                let loaded = self.loaded();
                let (preview, notes, err) = match self.editor.build_map() {
                    Ok((map, notes)) => (loaded.preview(&map), notes, None),
                    Err(e) => (crate::rename::RenamePreview::default(), Vec::new(), Some(e)),
                };
                let mut warnings = preview.warnings.clone();
                warnings.extend(notes);
                let has_index = preview.has_index;
                let applicable = err.is_none() && preview.applicable();
                let cli = ex.rename_cli_command(&self.target, &self.editor);
                let mut rules_view: Vec<crate::ui::RenameRuleView> = Vec::new();
                let mut total = 0usize;
                for p in &self.editor.pairs {
                    if p.source.trim().is_empty() || p.target.trim().is_empty() {
                        continue;
                    }
                    let Ok((pat, rep)) = crate::rename::rule_from_fields(&p.source, &p.target)
                    else {
                        continue;
                    };
                    let Ok(single) = crate::diff::NameMap::from_pairs([(pat, rep)]) else {
                        continue;
                    };
                    let pv = loaded.preview(&single);
                    let mut v = crate::ui::RenameRuleView {
                        from: p.source.clone(),
                        to: p.target.clone(),
                        total: pv.rows.len(),
                        matched: single.match_count(loaded.names().iter().map(String::as_str)),
                        ok: 0,
                        collide: 0,
                        wont_fit: 0,
                        invalid: 0,
                        shards: loaded.shard_fits(&single),
                    };
                    for r in &pv.rows {
                        match r.status {
                            crate::rename::RenameStatus::Ok => v.ok += 1,
                            crate::rename::RenameStatus::Collision => v.collide += 1,
                            crate::rename::RenameStatus::WontFit => v.wont_fit += 1,
                            crate::rename::RenameStatus::Invalid => v.invalid += 1,
                        }
                    }
                    total += v.total;
                    rules_view.push(v);
                }
                (warnings, has_index, applicable, err, cli, rules_view, total)
            };
            self.warnings = warnings;
            self.has_index = has_index;
            self.applicable = applicable;
            self.synth_err = err;
            self.cli = cli;
            self.rules_view = rules_view;
            self.total = total;
            self.dirty = false;
        }
        self.editor.scroll = self.editor.scroll.min(self.scroll_max.get());
    }

    fn render_frame(&self, ex: &Explorer, f: &mut ratatui::Frame) {
        let (max, chips, clicks, menu_rects, vscroll) = draw_rename_frame(
            f,
            &self.root,
            &self.editor,
            &self.schemas,
            &self.rules_view,
            self.total,
            &self.warnings,
            self.has_index,
            self.applicable,
            &self.synth_err,
            self.cli.as_deref(),
            self.copied,
        );
        self.scroll_max.set(max);
        *ex.vscrollbar.borrow_mut() = vscroll;
        // A preview link the open dropdown floats over must not steal the click that
        // was meant for a candidate row (the generic router tries links first).
        let clicks: crate::ui::LinkRegions = clicks
            .into_iter()
            .filter(|(r, _)| {
                !menu_rects
                    .iter()
                    .any(|mr| r.y == mr.y && r.x < mr.x + mr.width && mr.x < r.x + r.width)
            })
            .collect();
        *self.menu_rects.borrow_mut() = menu_rects; // dropdown rows (click to accept)
        *ex.clickable.borrow_mut() = chips; // footer chips (replay a key)
        *ex.links.borrow_mut() = clicks; // shard → layout, tensor → tree
    }

    fn open_palette(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
    ) -> PaletteResult {
        let entries =
            available_rename_commands(self.applicable, self.cli.is_some(), self.editor.pairs.len());
        let chosen = ex.run_palette(term, entries, HelpCtx::Rename, |_s, f| {
            let _ = draw_rename_frame(
                f,
                &self.root,
                &self.editor,
                &self.schemas,
                &self.rules_view,
                self.total,
                &self.warnings,
                self.has_index,
                self.applicable,
                &self.synth_err,
                self.cli.as_deref(),
                self.copied,
            );
        });
        match chosen {
            Some(RenameCmd::Back) => PaletteResult::Nav(Nav::Back),
            Some(RenameCmd::Quit) => PaletteResult::Nav(Nav::Quit),
            Some(RenameCmd::AddRule) => {
                self.editor.add_pair();
                self.editor.error = None;
                self.dirty = true;
                PaletteResult::Handled
            }
            Some(RenameCmd::RemoveRule) => {
                self.editor.remove_pair();
                self.editor.error = None;
                self.dirty = true;
                PaletteResult::Handled
            }
            Some(RenameCmd::Apply) => match self.try_apply(ex, term) {
                Some(nav) => PaletteResult::Nav(nav),
                None => PaletteResult::Handled,
            },
            Some(RenameCmd::CopyApplyCmd) => {
                self.do_copy_apply();
                PaletteResult::Handled
            }
            Some(RenameCmd::CopyReopenCmd) => PaletteResult::CopyCommand,
            Some(RenameCmd::CopyScreen) => {
                self.do_copy_screen();
                PaletteResult::Handled
            }
            Some(RenameCmd::Legend) => {
                ex.float_until_dismissed(term, |f| {
                    let _ = draw_rename_frame(
                        f,
                        &self.root,
                        &self.editor,
                        &self.schemas,
                        &self.rules_view,
                        self.total,
                        &self.warnings,
                        self.has_index,
                        self.applicable,
                        &self.synth_err,
                        self.cli.as_deref(),
                        self.copied,
                    );
                    UI::render_legend_band(f, Legend::Rename);
                });
                PaletteResult::Handled
            }
            None => PaletteResult::Handled,
        }
    }

    fn handle_key(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
        key: KeyEvent,
    ) -> Result<Outcome> {
        let KeyEvent {
            code, modifiers, ..
        } = key;
        // `^Y` (copy the command to reopen this editor — the `y` of the non-editing
        // modes) is handled by the engine (`do_copy_command`), so it's identical
        // everywhere. `^A` copies the `convert --map` command that *applies* the rename.
        if code == KeyCode::Char('a') && modifiers.contains(KeyModifiers::CONTROL) {
            self.do_copy_apply();
            return Ok(Outcome::Stay);
        }
        self.copied = None;
        // When the autocomplete dropdown is open, the arrows drive it and Enter
        // accepts the highlight (pgcli-style); otherwise Enter moves between fields.
        let cands = self.editor.completions(&self.schemas);
        let menu_open = self.editor.menu.is_some() && !cands.is_empty();
        match code {
            KeyCode::Esc if menu_open => self.editor.close_menu(),
            KeyCode::Esc => return Ok(Outcome::Leave(Nav::Back)),
            KeyCode::Char('n') if modifiers.contains(KeyModifiers::CONTROL) => {
                self.editor.add_pair();
                self.editor.error = None;
                self.dirty = true;
            }
            KeyCode::Char('d') if modifiers.contains(KeyModifiers::CONTROL) => {
                self.editor.remove_pair();
                self.editor.error = None;
                self.dirty = true;
            }
            // `^S` copies the whole screen (bare `c` types into a field here, so
            // copy-screen is a Ctrl key in the editor).
            KeyCode::Char('s') if modifiers.contains(KeyModifiers::CONTROL) => {
                self.do_copy_screen()
            }
            // `^L` shows the legend (bare `l` types into a field here).
            KeyCode::Char('l') if modifiers.contains(KeyModifiers::CONTROL) => {
                ex.float_until_dismissed(term, |f| {
                    let _ = draw_rename_frame(
                        f,
                        &self.root,
                        &self.editor,
                        &self.schemas,
                        &self.rules_view,
                        self.total,
                        &self.warnings,
                        self.has_index,
                        self.applicable,
                        &self.synth_err,
                        self.cli.as_deref(),
                        self.copied,
                    );
                    UI::render_legend_band(f, Legend::Rename);
                });
            }
            // Tab opens the dropdown and extends the field to the candidates' longest
            // common prefix (shell-style). Enter / a click accept the highlight — so
            // the two keys stay distinct.
            KeyCode::Tab => {
                self.editor.open_menu();
                self.editor.complete_prefix(&self.schemas);
                self.editor.error = None;
                self.dirty = true;
            }
            // Enter accepts a highlighted candidate; with the dropdown closed it
            // moves to the next field (adding a rule past the last) — it never
            // applies. Apply is `^R` (below).
            KeyCode::Enter if menu_open => {
                self.editor.accept(&self.schemas);
                self.editor.error = None;
                self.dirty = true;
            }
            KeyCode::Enter => self.editor.focus_down(),
            // `^R` applies the rename (a Ctrl key, so every character stays typeable),
            // after a confirmation pop-up.
            KeyCode::Char('r') if modifiers.contains(KeyModifiers::CONTROL) => {
                if let Some(nav) = self.try_apply(ex, term) {
                    return Ok(Outcome::Leave(nav));
                }
            }
            KeyCode::Up if menu_open => self.editor.menu_move(-1, cands.len()),
            KeyCode::Down if menu_open => self.editor.menu_move(1, cands.len()),
            KeyCode::Up => self.editor.focus_up(),
            KeyCode::Down => self.editor.focus_down(),
            KeyCode::Left => self.editor.left(),
            KeyCode::Right => self.editor.right(),
            KeyCode::Home => self.editor.cursor = 0,
            KeyCode::End => self.editor.caret_to_end(),
            KeyCode::PageUp => self.editor.scroll = self.editor.scroll.saturating_sub(SCROLL_PAGE),
            KeyCode::PageDown => {
                self.editor.scroll = (self.editor.scroll + SCROLL_PAGE).min(self.scroll_max.get());
            }
            KeyCode::Backspace => {
                self.editor.backspace();
                self.editor.remove_pair_if_empty();
                self.editor.error = None;
                self.dirty = true;
            }
            KeyCode::Delete => {
                self.editor.delete();
                self.editor.remove_pair_if_empty();
                self.editor.error = None;
                self.dirty = true;
            }
            KeyCode::Char(c)
                if !modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
            {
                self.editor.insert_char(c);
                self.editor.error = None;
                self.dirty = true;
            }
            _ => {}
        }
        Ok(Outcome::Stay)
    }

    fn handle_mouse(
        &mut self,
        _ex: &mut Explorer,
        _term: &mut crate::tui::LiveTerminal,
        m: MouseEvent,
    ) -> MouseOutcome {
        match m.kind {
            // A click on a dropdown row highlights and accepts that candidate.
            MouseEventKind::Down(MouseButton::Left) => {
                let hit = self.menu_rects.borrow().iter().position(|r| {
                    m.column >= r.x
                        && m.column < r.x + r.width
                        && m.row >= r.y
                        && m.row < r.y + r.height
                });
                if let Some(i) = hit {
                    self.editor.menu = Some(i);
                    self.editor.accept(&self.schemas);
                    self.editor.error = None;
                    self.dirty = true;
                    MouseOutcome::Redraw
                } else {
                    MouseOutcome::Ignored
                }
            }
            MouseEventKind::ScrollDown => {
                self.editor.scroll = (self.editor.scroll + 3).min(self.scroll_max.get());
                MouseOutcome::Redraw
            }
            MouseEventKind::ScrollUp => {
                self.editor.scroll = self.editor.scroll.saturating_sub(3);
                MouseOutcome::Redraw
            }
            _ => MouseOutcome::Ignored,
        }
    }

    fn set_scroll(&mut self, _ex: &mut Explorer, offset: usize) {
        self.editor.scroll = offset;
    }

    fn residual(&self) -> Screen {
        Screen::Rename {
            pairs: if self.applied {
                Vec::new()
            } else {
                self.pairs()
            },
        }
    }
}

/// The tensor detail screen ([`Screen::Detail`]) as a [`Mode`]. Runs the exact-stats
/// scan on a worker thread (via `tick_background` + `Bg::Poll`) and floats the legend
/// / copied-command as an in-frame `overlay` so a running scan animates behind it.
struct DetailMode {
    tensor_name: String,
    slice: usize,
    stats_start: StatsStart,
    interaction: Interaction,
    tensor: Option<TensorInfo>,
    overridable: bool,
    unindexed: bool,
    remote: bool,
    warm: bool,
    scan: Option<ScanJob>,
    spin: std::cell::Cell<usize>,
    overlay: Option<Overlay>,
}

impl DetailMode {
    fn new(
        tensor_name: String,
        slice: usize,
        stats_start: StatsStart,
        interaction: Interaction,
    ) -> Self {
        Self {
            tensor_name,
            slice,
            stats_start,
            interaction,
            tensor: None,
            overridable: false,
            unindexed: false,
            remote: false,
            warm: false,
            scan: None,
            spin: std::cell::Cell::new(0),
            overlay: None,
        }
    }

    fn tensor(&self) -> &TensorInfo {
        self.tensor.as_ref().expect("on_enter resolves or leaves")
    }

    fn shape(&self, ex: &Explorer) -> Vec<usize> {
        let t = self.tensor();
        ex.data_view
            .shape_overrides
            .borrow()
            .get(&t.name)
            .cloned()
            .unwrap_or_else(|| t.shape.clone())
    }

    /// The current statistics view — cached result, a live scan spinner, or pending.
    /// `stats` is the caller's local so the returned `StatsView` can borrow it.
    fn stats_view<'a>(&self, stats: &'a Option<Stats>) -> StatsView<'a> {
        match stats {
            Some(s) => StatsView::Ready(s),
            None if self.warm && self.scan.is_some() => {
                let job = self.scan.as_ref().unwrap();
                if job.started.elapsed() >= std::time::Duration::from_millis(120) {
                    self.spin.set(self.spin.get().wrapping_add(1));
                    StatsView::Computing {
                        spinner: STATS_SPINNER[self.spin.get() % STATS_SPINNER.len()],
                        elapsed: job.started.elapsed(),
                        progress: job.progress(),
                    }
                } else {
                    StatsView::Pending
                }
            }
            None => StatsView::Pending,
        }
    }

    fn layout_ok(&self) -> bool {
        !self.remote
            && std::path::Path::new(&self.tensor().source_path)
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"))
    }
}

impl Mode for DetailMode {
    fn spec(&self) -> ModeSpec {
        ModeSpec {
            id: HelpCtx::Detail,
            ctrlc_quits_immediately: true,
        }
    }

    fn set_background_paused(&self, paused: bool) {
        if let Some(job) = &self.scan {
            job.pause.store(paused, Ordering::Relaxed);
        }
    }

    fn overlay(&self) -> Option<&Overlay> {
        self.overlay.as_ref()
    }
    fn dismiss_overlay(&mut self) -> bool {
        self.overlay.take().is_some()
    }

    fn on_enter(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
    ) -> Result<Outcome> {
        let Some(tensor) = ex
            .tensors()
            .iter()
            .find(|t| t.name == self.tensor_name)
            .cloned()
        else {
            return Ok(Outcome::Leave(Nav::Open(Screen::Tree)));
        };
        self.overridable = dtype_overridable(&tensor);
        self.unindexed = ex.unindexed.contains(&tensor.source_path);
        self.remote = crate::remote::is_remote_source(&tensor.source_path);
        // Background pre-warm scan: only when interactive, overridable, local, and
        // not already doing a synchronous `--compute-stats` scan.
        self.warm = ex.preload
            && self.stats_start != StatsStart::Auto
            && self.interaction == Interaction::Interactive
            && self.overridable
            && !self.remote;
        // `--compute-stats`: kick off the scan synchronously on open, animating the
        // spinner right here.
        if self.stats_start == StatsStart::Auto && !self.remote {
            let view = ex.active_view(&tensor.name);
            let shape = ex
                .data_view
                .shape_overrides
                .borrow()
                .get(&tensor.name)
                .cloned()
                .unwrap_or_else(|| tensor.shape.clone());
            let (overridable, unindexed) = (self.overridable, self.unindexed);
            ex.compute_stats_animated(term, &tensor, view, |f, sv| {
                ex.render_detail_frame(
                    f,
                    &tensor,
                    &shape,
                    view,
                    overridable,
                    unindexed,
                    sv,
                    None,
                    None,
                    None,
                );
            });
        }
        self.tensor = Some(tensor);
        Ok(Outcome::Stay)
    }

    fn tick_background(&mut self, ex: &mut Explorer) -> Bg {
        if !self.warm {
            return Bg::Idle;
        }
        let tensor = self.tensor().clone();
        let view = ex.active_view(&tensor.name);
        if ex.cached_stats(&tensor, view).is_some() {
            self.scan = None;
            return Bg::Idle;
        }
        // (Re)start the scan for the current view; harvest it when finished.
        if self.scan.as_ref().is_none_or(|j| j.view != view) {
            self.scan = Some(ex.spawn_stats_scan(&tensor, view));
        }
        let finished = self
            .scan
            .as_ref()
            .and_then(|j| j.handle.as_ref())
            .is_some_and(|h| h.is_finished());
        if finished {
            let mut job = self.scan.take().unwrap();
            if let Some(h) = job.handle.take()
                && let Ok(Ok(s)) = h.join()
            {
                ex.stats_cache
                    .borrow_mut()
                    .insert((tensor.name.clone(), view), s);
            }
        }
        if self.scan.is_some() {
            Bg::Poll
        } else {
            Bg::Idle
        }
    }

    fn render_frame(&self, ex: &Explorer, f: &mut ratatui::Frame) {
        let tensor = self.tensor();
        let view = ex.active_view(&tensor.name);
        let shape = self.shape(ex);
        let stats = ex.cached_stats(tensor, view);
        let stats_view = self.stats_view(&stats);
        let hist = ex
            .histogram_cache
            .borrow()
            .get(&(tensor.name.clone(), view, ex.data_view.histogram_bins.get()))
            .cloned();
        ex.render_detail_frame(
            f,
            tensor,
            &shape,
            view,
            self.overridable,
            self.unindexed,
            stats_view,
            hist.as_ref(),
            None,
            self.overlay.as_ref(),
        );
        if let Some((what, _)) = &ex.copied_flash {
            UI::render_copied_flash(f, what);
        }
    }

    fn open_palette(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
    ) -> PaletteResult {
        let tensor = self.tensor();
        let view = ex.active_view(&tensor.name);
        let shape = self.shape(ex);
        let stats = ex.cached_stats(tensor, view);
        let stats_view = self.stats_view(&stats);
        let hist = ex
            .histogram_cache
            .borrow()
            .get(&(tensor.name.clone(), view, ex.data_view.histogram_bins.get()))
            .cloned();
        let entries = available_detail_commands(self.overridable, self.layout_ok());
        let (overridable, unindexed) = (self.overridable, self.unindexed);
        let chosen = ex.run_palette(term, entries, HelpCtx::Detail, |s, f| {
            s.render_detail_frame(
                f,
                tensor,
                &shape,
                view,
                overridable,
                unindexed,
                stats_view,
                hist.as_ref(),
                None,
                None,
            );
        });
        match chosen {
            Some(DetailCmd::CopyCommand) => PaletteResult::CopyCommand,
            Some(cmd) => PaletteResult::SynthKey(detail_cmd_key(cmd)),
            None => PaletteResult::Handled,
        }
    }

    fn handle_key(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
        key: KeyEvent,
    ) -> Result<Outcome> {
        let tensor = self.tensor().clone();
        let view = ex.active_view(&tensor.name);
        let shape = self.shape(ex);
        // Metadata-only (remote): the data keys can't run without local bytes, so
        // float a notice instead of a read that fails.
        if self.remote
            && matches!(
                key.code,
                KeyCode::Char('m' | 'v' | 'h' | 's' | 'S' | 'b' | 'B')
            )
        {
            self.overlay = Some(Overlay::Notice(
                "Read remotely with --ssh-read: only the structure is here. Data views \
                 (heatmap, values, histogram, statistics) need the file locally — copy the \
                 checkpoint down to preview its values."
                    .to_string(),
            ));
            return Ok(Outcome::Stay);
        }
        match key.code {
            KeyCode::Char('m') => {
                return Ok(Outcome::Leave(Nav::Open(Screen::Data {
                    tensor: tensor.name.clone(),
                    repr: Representation::Heatmap,
                    slice: self.slice,
                })));
            }
            KeyCode::Char('v') => {
                return Ok(Outcome::Leave(Nav::Open(Screen::Data {
                    tensor: tensor.name.clone(),
                    repr: Representation::Values,
                    slice: self.slice,
                })));
            }
            KeyCode::Tab => {
                if let Some(screen) = ex.tensor_layout_screen(&tensor) {
                    return Ok(Outcome::Leave(Nav::Open(screen)));
                }
            }
            KeyCode::Char('h') => {
                ex.ensure_detail_histogram(
                    term,
                    &tensor,
                    view,
                    &shape,
                    self.overridable,
                    self.unindexed,
                );
            }
            KeyCode::Char('b') | KeyCode::Char('B') => {
                let (overridable, unindexed) = (self.overridable, self.unindexed);
                let stats = ex.cached_stats(&tensor, view);
                let stats_view = self.stats_view(&stats);
                let hist = ex
                    .histogram_cache
                    .borrow()
                    .get(&(tensor.name.clone(), view, ex.data_view.histogram_bins.get()))
                    .cloned();
                let background = |f: &mut ratatui::Frame| {
                    ex.render_detail_frame(
                        f,
                        &tensor,
                        &shape,
                        view,
                        overridable,
                        unindexed,
                        stats_view,
                        hist.as_ref(),
                        None,
                        None,
                    );
                };
                let changed =
                    match ex.prompt_bins(term, background, ex.data_view.histogram_bins.get()) {
                        BinsChoice::Set(n) => {
                            ex.data_view.histogram_bins.set(Some(n));
                            true
                        }
                        BinsChoice::Clear => {
                            ex.data_view.histogram_bins.set(None);
                            true
                        }
                        BinsChoice::Cancel => false,
                    };
                if changed {
                    ex.ensure_detail_histogram(
                        term,
                        &tensor,
                        view,
                        &shape,
                        self.overridable,
                        self.unindexed,
                    );
                }
            }
            KeyCode::Char('s') | KeyCode::Char('S') => {
                // `s` is a no-op once the stats are cached — say so rather than
                // silently doing nothing (a key that appears not to work).
                if ex.cached_stats(&tensor, view).is_some() {
                    ex.copied_flash = Some((
                        "statistics already computed".to_string(),
                        std::time::Instant::now(),
                    ));
                } else {
                    let (overridable, unindexed) = (self.overridable, self.unindexed);
                    ex.compute_stats_animated(term, &tensor, view, |f, sv| {
                        ex.render_detail_frame(
                            f,
                            &tensor,
                            &shape,
                            view,
                            overridable,
                            unindexed,
                            sv,
                            None,
                            None,
                            None,
                        );
                    });
                }
            }
            KeyCode::Char('d') | KeyCode::Char('D') if self.overridable => {
                if let Some(chosen) = ex.prompt_dtype(term, &tensor, DtypePreview::Detail) {
                    let def = ex.default_view(&tensor.name);
                    let mut overrides = ex.data_view.dtype_overrides.borrow_mut();
                    if chosen == def {
                        overrides.remove(&tensor.name);
                    } else {
                        overrides.insert(tensor.name.clone(), chosen);
                    }
                }
            }
            KeyCode::Char('r') | KeyCode::Char('R') if self.overridable => {
                let current = ex
                    .data_view
                    .shape_overrides
                    .borrow()
                    .get(&tensor.name)
                    .cloned();
                let (overridable, unindexed) = (self.overridable, self.unindexed);
                let stats = ex.cached_stats(&tensor, view);
                let stats_view = self.stats_view(&stats);
                let hist = ex
                    .histogram_cache
                    .borrow()
                    .get(&(tensor.name.clone(), view, ex.data_view.histogram_bins.get()))
                    .cloned();
                let background = |f: &mut ratatui::Frame| {
                    ex.render_detail_frame(
                        f,
                        &tensor,
                        &shape,
                        view,
                        overridable,
                        unindexed,
                        stats_view,
                        hist.as_ref(),
                        None,
                        None,
                    );
                };
                match ex.prompt_reshape(term, background, &tensor, current.as_deref()) {
                    ReshapeChoice::Set(s) => {
                        ex.data_view
                            .shape_overrides
                            .borrow_mut()
                            .insert(tensor.name.clone(), s);
                    }
                    ReshapeChoice::Clear => {
                        ex.data_view
                            .shape_overrides
                            .borrow_mut()
                            .remove(&tensor.name);
                    }
                    ReshapeChoice::Cancel => {}
                }
            }
            KeyCode::Char('c') => {
                let stats = ex.cached_stats(&tensor, view);
                let stats_view = self.stats_view(&stats);
                let hist = ex
                    .histogram_cache
                    .borrow()
                    .get(&(tensor.name.clone(), view, ex.data_view.histogram_bins.get()))
                    .cloned();
                if let Ok(text) = ex.detail_plain(
                    &tensor,
                    &shape,
                    view,
                    self.overridable,
                    self.unindexed,
                    stats_view,
                    hist.as_ref(),
                    None,
                ) {
                    copy_to_clipboard(&text);
                }
                ex.copied_flash = Some((
                    "copied the screen to the clipboard".to_string(),
                    std::time::Instant::now(),
                ));
            }
            // `y` (copy the reopen command) is handled by the engine — see
            // `Explorer::do_copy_command` — so every mode does it identically.
            KeyCode::Char('l') => self.overlay = Some(Overlay::Legend(Legend::Detail)),
            KeyCode::Backspace => return Ok(Outcome::Leave(Nav::Back)),
            KeyCode::Char('\\') => return Ok(Outcome::Leave(Nav::Forward)),
            // Any other key goes back to the tree.
            _ => return Ok(Outcome::Leave(Nav::Open(Screen::Tree))),
        }
        Ok(Outcome::Stay)
    }

    fn residual(&self) -> Screen {
        Screen::Detail {
            tensor: self.tensor_name.clone(),
            slice: self.slice,
        }
    }
}

/// The full-screen checkpoint-stats view ([`Screen::Stats`]) as a [`Mode`]: a
/// scrollable report (sizes, params, dtype mix, layers, experts, per-layer graphs)
/// with the on-disk per-shard breakdown foldable via `f`. The stats are computed
/// once and cached on the [`Explorer`]; `scroll` / `shards_expanded` round-trip
/// through history and the `--stats` reopen command.
struct StatsMode {
    shards_expanded: bool,
    scroll: usize,
    /// The last render's maximum scroll (render is `&self`), so the key / wheel
    /// handlers can clamp downward scrolling to the content.
    scroll_max: std::cell::Cell<usize>,
    overlay: Option<Overlay>,
}

impl StatsMode {
    fn new(shards_expanded: bool, scroll: usize) -> Self {
        Self {
            shards_expanded,
            scroll,
            scroll_max: std::cell::Cell::new(0),
            overlay: None,
        }
    }

    /// The cached stats (computed in `on_enter`).
    fn stats(&self, ex: &Explorer) -> crate::stats::CheckpointStats {
        ex.checkpoint_stats_cache
            .borrow()
            .clone()
            .expect("on_enter computes and caches the stats")
    }

    /// Whether the report has a foldable breakdown for `f` / a click to toggle: a
    /// multi-shard on-disk section, or the S3 per-object list (an s3 source has no
    /// on-disk section, so the two never coexist and share the one fold state).
    fn has_fold(&self, ex: &Explorer) -> bool {
        let cache = ex.checkpoint_stats_cache.borrow();
        let Some(s) = cache.as_ref() else {
            return false;
        };
        let on_disk = s.disk().is_some_and(|d| d.shards.len() > 1);
        let s3 = s.s3().is_some_and(|x| !x.objects.is_empty());
        on_disk || s3
    }
}

impl Mode for StatsMode {
    fn spec(&self) -> ModeSpec {
        ModeSpec {
            id: HelpCtx::Stats,
            ctrlc_quits_immediately: true,
        }
    }

    fn overlay(&self) -> Option<&Overlay> {
        self.overlay.as_ref()
    }
    fn dismiss_overlay(&mut self) -> bool {
        self.overlay.take().is_some()
    }

    fn on_enter(
        &mut self,
        ex: &mut Explorer,
        _term: &mut crate::tui::LiveTerminal,
    ) -> Result<Outcome> {
        // Reuse the stats computed on a previous open (an O(tensors) header-only
        // pass over an immutable checkpoint), else compute + cache them now.
        if ex.checkpoint_stats_cache.borrow().is_none() {
            let s =
                crate::stats::CheckpointStats::compute(ex.tensors(), ex.config(), ex.disk_usage())
                    .with_s3(ex.s3_stats());
            *ex.checkpoint_stats_cache.borrow_mut() = Some(s);
        }
        Ok(Outcome::Stay)
    }

    fn pre_draw(&mut self, _ex: &mut Explorer, _term: &mut crate::tui::LiveTerminal) {
        self.scroll = self.scroll.min(self.scroll_max.get());
    }

    fn render_frame(&self, ex: &Explorer, f: &mut ratatui::Frame) {
        let stats = self.stats(ex);
        let max = ex.render_stats_screen(f, &stats, self.scroll, self.shards_expanded);
        self.scroll_max.set(max);
        if let Some((what, _)) = &ex.copied_flash {
            UI::render_copied_flash(f, what);
        }
    }

    fn open_palette(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
    ) -> PaletteResult {
        let stats = self.stats(ex);
        let entries = available_stats_commands(self.has_fold(ex));
        let (scroll, shards_expanded) = (self.scroll, self.shards_expanded);
        let chosen = ex.run_palette(term, entries, HelpCtx::Stats, |s, f| {
            s.render_stats_screen(f, &stats, scroll, shards_expanded);
        });
        match chosen {
            Some(StatsCmd::CopyCommand) => PaletteResult::CopyCommand,
            Some(cmd) => PaletteResult::SynthKey(stats_cmd_key(cmd)),
            None => PaletteResult::Handled,
        }
    }

    fn handle_key(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
        key: KeyEvent,
    ) -> Result<Outcome> {
        match key.code {
            // Fold / expand the on-disk per-shard breakdown (a no-op without one).
            KeyCode::Char('f') => {
                if self.has_fold(ex) {
                    self.shards_expanded = !self.shards_expanded;
                }
            }
            // Copy the report as plain text, matching the current fold state.
            KeyCode::Char('r') => {
                let report = self.stats(ex).render(self.shards_expanded);
                copy_to_clipboard(&report);
                ex.copied_flash = Some((
                    "copied the report to the clipboard".to_string(),
                    std::time::Instant::now(),
                ));
            }
            // Copy the whole screen's text at the live terminal size.
            KeyCode::Char('c') => {
                let (w, h) = term
                    .size()
                    .map(|s| (s.width, s.height))
                    .unwrap_or((120, 40));
                let stats = self.stats(ex);
                let (scroll, shards_expanded) = (self.scroll, self.shards_expanded);
                if let Ok(text) = crate::tui::headless_render(w, h, |f| {
                    UI::render_stats_frame(f, &stats, scroll, shards_expanded);
                }) {
                    copy_to_clipboard(&text);
                }
                ex.copied_flash = Some((
                    "copied the screen to the clipboard".to_string(),
                    std::time::Instant::now(),
                ));
            }
            KeyCode::Char('l') => self.overlay = Some(Overlay::Legend(Legend::Stats)),
            KeyCode::Char('q') => return Ok(Outcome::Leave(Nav::Quit)),
            KeyCode::Up => self.scroll = self.scroll.saturating_sub(1),
            KeyCode::Down => self.scroll = (self.scroll + 1).min(self.scroll_max.get()),
            KeyCode::PageUp => self.scroll = self.scroll.saturating_sub(SCROLL_PAGE),
            KeyCode::PageDown => {
                self.scroll = (self.scroll + SCROLL_PAGE).min(self.scroll_max.get())
            }
            KeyCode::Home => self.scroll = 0,
            KeyCode::End => self.scroll = self.scroll_max.get(),
            KeyCode::Esc | KeyCode::Backspace => return Ok(Outcome::Leave(Nav::Back)),
            KeyCode::Char('\\') => return Ok(Outcome::Leave(Nav::Forward)),
            _ => {}
        }
        Ok(Outcome::Stay)
    }

    fn handle_mouse(
        &mut self,
        _ex: &mut Explorer,
        _term: &mut crate::tui::LiveTerminal,
        m: MouseEvent,
    ) -> MouseOutcome {
        match m.kind {
            MouseEventKind::ScrollUp => {
                self.scroll = self.scroll.saturating_sub(WHEEL_STEP);
                MouseOutcome::Redraw
            }
            MouseEventKind::ScrollDown => {
                self.scroll = (self.scroll + WHEEL_STEP).min(self.scroll_max.get());
                MouseOutcome::Redraw
            }
            _ => MouseOutcome::Ignored,
        }
    }

    fn set_scroll(&mut self, _ex: &mut Explorer, offset: usize) {
        self.scroll = offset;
    }

    fn residual(&self) -> Screen {
        Screen::Stats {
            shards_expanded: self.shards_expanded,
            scroll: self.scroll,
        }
    }
}

/// The tensor data view ([`Screen::Data`]) as a [`Mode`] — the heatmap / numeric
/// grid. Like the detail screen it runs the exact-stats scan on a worker thread
/// (`tick_background`/`Bg::Poll`, paused while input flows). `slice`/`slices`/
/// `overridable` are `Cell`s because they're learned during the (`&self`) sample.
struct DataMode {
    tensor_name: String,
    repr: Representation,
    slice: std::cell::Cell<usize>,
    interaction: Interaction,
    tensor: Option<TensorInfo>,
    scan: Option<ScanJob>,
    spin: std::cell::Cell<usize>,
    overlay: Option<Overlay>,
    slices: std::cell::Cell<usize>,
    overridable: std::cell::Cell<bool>,
}

impl DataMode {
    fn new(
        tensor_name: String,
        repr: Representation,
        slice: usize,
        interaction: Interaction,
    ) -> Self {
        Self {
            tensor_name,
            repr,
            slice: std::cell::Cell::new(slice),
            interaction,
            tensor: None,
            scan: None,
            spin: std::cell::Cell::new(0),
            overlay: None,
            slices: std::cell::Cell::new(1),
            overridable: std::cell::Cell::new(false),
        }
    }

    fn tensor(&self) -> &TensorInfo {
        self.tensor.as_ref().expect("on_enter resolves or leaves")
    }

    /// The current statistics view — cached, a live scan spinner (data always
    /// scans when uncached), or pending. `stats` is the caller's local.
    fn stats_view<'a>(&self, stats: &'a Option<Stats>) -> StatsView<'a> {
        match stats {
            Some(s) => StatsView::Ready(s),
            None if self.scan.is_some() => {
                let job = self.scan.as_ref().unwrap();
                if job.started.elapsed() >= std::time::Duration::from_millis(120) {
                    self.spin.set(self.spin.get().wrapping_add(1));
                    StatsView::Computing {
                        spinner: STATS_SPINNER[self.spin.get() % STATS_SPINNER.len()],
                        elapsed: job.started.elapsed(),
                        progress: job.progress(),
                    }
                } else {
                    StatsView::Pending
                }
            }
            None => StatsView::Pending,
        }
    }
}

impl Mode for DataMode {
    fn spec(&self) -> ModeSpec {
        ModeSpec {
            id: HelpCtx::Data,
            ctrlc_quits_immediately: true,
        }
    }

    fn set_background_paused(&self, paused: bool) {
        if let Some(job) = &self.scan {
            job.pause.store(paused, Ordering::Relaxed);
        }
    }

    fn overlay(&self) -> Option<&Overlay> {
        self.overlay.as_ref()
    }
    fn dismiss_overlay(&mut self) -> bool {
        self.overlay.take().is_some()
    }

    fn on_enter(
        &mut self,
        ex: &mut Explorer,
        _term: &mut crate::tui::LiveTerminal,
    ) -> Result<Outcome> {
        let Some(tensor) = ex
            .tensors()
            .iter()
            .find(|t| t.name == self.tensor_name)
            .cloned()
        else {
            return Ok(Outcome::Leave(Nav::Back));
        };
        // One-shot (`--exit`): compute the stats synchronously so the single frame
        // shows them (interactively the scan runs in the background via tick).
        if self.interaction == Interaction::OneShot {
            let view = ex.active_view(&tensor.name);
            ex.compute_stats_sync(&tensor, view);
        }
        self.tensor = Some(tensor);
        Ok(Outcome::Stay)
    }

    fn tick_background(&mut self, ex: &mut Explorer) -> Bg {
        let tensor = self.tensor().clone();
        let view = ex.active_view(&tensor.name);
        if ex.cached_stats(&tensor, view).is_some() {
            self.scan = None;
            return Bg::Idle;
        }
        if self.scan.as_ref().is_none_or(|j| j.view != view) {
            self.scan = Some(ex.spawn_stats_scan(&tensor, view));
        }
        let finished = self
            .scan
            .as_ref()
            .and_then(|j| j.handle.as_ref())
            .is_some_and(|h| h.is_finished());
        if finished {
            let mut job = self.scan.take().unwrap();
            if let Some(h) = job.handle.take()
                && let Ok(Ok(s)) = h.join()
            {
                ex.stats_cache
                    .borrow_mut()
                    .insert((tensor.name.clone(), view), s);
            }
        }
        if self.scan.is_some() {
            Bg::Poll
        } else {
            Bg::Idle
        }
    }

    fn render_frame(&self, ex: &Explorer, f: &mut ratatui::Frame) {
        let tensor = self.tensor();
        let view = ex.active_view(&tensor.name);
        let mode = ex.data_sample_mode();
        let stats = ex.cached_stats(tensor, view);
        let stats_view = self.stats_view(&stats);
        let stripe = ex.data_view.data_view_stripe.get();
        let base = ex.data_view.data_view_base.get();
        match ex.render_data_frame(
            f,
            tensor,
            self.repr,
            self.slice.get(),
            view,
            mode,
            stats_view,
            stripe,
            base,
            self.overlay.as_ref(),
        ) {
            Ok((slices, overridable, clamped)) => {
                self.slices.set(slices);
                self.overridable.set(overridable);
                self.slice.set(clamped);
            }
            Err(msg) => UI::render_message(f, "Data preview unavailable", &msg),
        }
        if let Some((what, _)) = &ex.copied_flash {
            UI::render_copied_flash(f, what);
        }
    }

    fn open_palette(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
    ) -> PaletteResult {
        let tensor = self.tensor();
        let view = ex.active_view(&tensor.name);
        let mode = ex.data_sample_mode();
        let stats = ex.cached_stats(tensor, view);
        let stats_view = self.stats_view(&stats);
        let stripe = ex.data_view.data_view_stripe.get();
        let base = ex.data_view.data_view_base.get();
        let (repr, slice) = (self.repr, self.slice.get());
        let entries = available_data_commands(self.overridable.get());
        let chosen = ex.run_palette(term, entries, HelpCtx::Data, |s, f| {
            let _ = s.render_data_frame(
                f, tensor, repr, slice, view, mode, stats_view, stripe, base, None,
            );
        });
        match chosen {
            Some(DataCmd::CopyCommand) => PaletteResult::CopyCommand,
            Some(cmd) => PaletteResult::SynthKey(data_cmd_key(cmd)),
            None => PaletteResult::Handled,
        }
    }

    fn handle_key(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
        key: KeyEvent,
    ) -> Result<Outcome> {
        let tensor = self.tensor().clone();
        let view = ex.active_view(&tensor.name);
        let mode = ex.data_sample_mode();
        let slices = self.slices.get();
        let overridable = self.overridable.get();
        let stripe = ex.data_view.data_view_stripe.get();
        let base = ex.data_view.data_view_base.get();
        let KeyEvent {
            code, modifiers, ..
        } = key;
        let shift = modifiers.contains(KeyModifiers::SHIFT);
        let ctrl = modifiers.contains(KeyModifiers::CONTROL);
        let edges = matches!(mode, SampleMode::Edges { .. });
        let window = matches!(mode, SampleMode::Window { .. });
        // One arrow press moves the divider by a single index; Shift snaps to an end.
        let nudge = |cell: &Cell<f32>, toward_tail: bool, budget: usize| {
            let step = if shift {
                1.0
            } else {
                1.0 / budget.max(1) as f32
            };
            let delta = if toward_tail { step } else { -step };
            cell.set((cell.get() + delta).clamp(0.0, 1.0));
        };
        // Pan the window along one axis (Ctrl → edge, Shift → screenful, else plain).
        let pan = |cell: &Cell<usize>, forward: bool, page: usize, plain: usize| {
            let cur = cell.get();
            let next = if ctrl {
                if forward { usize::MAX } else { 0 }
            } else {
                let step = if shift { page.max(1) } else { plain.max(1) };
                if forward {
                    cur.saturating_add(step)
                } else {
                    cur.saturating_sub(step)
                }
            };
            cell.set(next);
        };
        match code {
            KeyCode::Char('m') => self.repr = Representation::Heatmap,
            KeyCode::Char('v') => self.repr = Representation::Values,
            KeyCode::Char('e') | KeyCode::Char('E') => ex
                .data_view
                .data_view_layout
                .set(ex.data_view.data_view_layout.get().next()),
            KeyCode::Char('z') | KeyCode::Char('Z') => ex
                .data_view
                .data_view_stripe
                .set(ex.data_view.data_view_stripe.get().next()),
            KeyCode::Char('b') | KeyCode::Char('B') => ex
                .data_view
                .data_view_base
                .set(ex.data_view.data_view_base.get().next()),
            KeyCode::Up if edges => nudge(
                &ex.data_view.data_view_row_tail,
                true,
                ex.data_view.edge_row_budget.get(),
            ),
            KeyCode::Down if edges => nudge(
                &ex.data_view.data_view_row_tail,
                false,
                ex.data_view.edge_row_budget.get(),
            ),
            KeyCode::Left if edges => nudge(
                &ex.data_view.data_view_col_tail,
                true,
                ex.data_view.edge_col_budget.get(),
            ),
            KeyCode::Right if edges => nudge(
                &ex.data_view.data_view_col_tail,
                false,
                ex.data_view.edge_col_budget.get(),
            ),
            KeyCode::Up if window => pan(
                &ex.data_view.data_view_win_row,
                false,
                ex.data_view.win_page_rows.get(),
                ex.held_step(KeyCode::Up, accel_step_row),
            ),
            KeyCode::Down if window => pan(
                &ex.data_view.data_view_win_row,
                true,
                ex.data_view.win_page_rows.get(),
                ex.held_step(KeyCode::Down, accel_step_row),
            ),
            KeyCode::Left if window => pan(
                &ex.data_view.data_view_win_col,
                false,
                ex.data_view.win_page_cols.get(),
                ex.held_step(KeyCode::Left, accel_step_row),
            ),
            KeyCode::Right if window => pan(
                &ex.data_view.data_view_win_col,
                true,
                ex.data_view.win_page_cols.get(),
                ex.held_step(KeyCode::Right, accel_step_row),
            ),
            KeyCode::Home if window => ex.data_view.data_view_win_col.set(0),
            KeyCode::End if window => ex.data_view.data_view_win_col.set(usize::MAX),
            KeyCode::PageUp if window => ex.data_view.data_view_win_row.set(0),
            KeyCode::PageDown if window => ex.data_view.data_view_win_row.set(usize::MAX),
            KeyCode::Char('d') | KeyCode::Char('D') if overridable => {
                if let Some(chosen) = ex.prompt_dtype(
                    term,
                    &tensor,
                    DtypePreview::Data {
                        repr: self.repr,
                        slice: self.slice.get(),
                        mode,
                    },
                ) {
                    let def = ex.default_view(&tensor.name);
                    let mut overrides = ex.data_view.dtype_overrides.borrow_mut();
                    if chosen == def {
                        overrides.remove(&tensor.name);
                    } else {
                        overrides.insert(tensor.name.clone(), chosen);
                    }
                }
            }
            KeyCode::Char('r') | KeyCode::Char('R') if overridable => {
                let current = ex
                    .data_view
                    .shape_overrides
                    .borrow()
                    .get(&tensor.name)
                    .cloned();
                let stats = ex.cached_stats(&tensor, view);
                let stats_view = self.stats_view(&stats);
                let repr = self.repr;
                let background = |f: &mut ratatui::Frame| {
                    ex.render_cached_data(f, &tensor, repr, stats_view, stripe, base);
                };
                match ex.prompt_reshape(term, background, &tensor, current.as_deref()) {
                    ReshapeChoice::Set(s) => {
                        ex.data_view
                            .shape_overrides
                            .borrow_mut()
                            .insert(tensor.name.clone(), s);
                        self.slice.set(0);
                    }
                    ReshapeChoice::Clear => {
                        ex.data_view
                            .shape_overrides
                            .borrow_mut()
                            .remove(&tensor.name);
                        self.slice.set(0);
                    }
                    ReshapeChoice::Cancel => {}
                }
            }
            KeyCode::Char('/') if slices > 1 => {
                let stats = ex.cached_stats(&tensor, view);
                let stats_view = self.stats_view(&stats);
                let repr = self.repr;
                let background = |f: &mut ratatui::Frame| {
                    ex.render_cached_data(f, &tensor, repr, stats_view, stripe, base);
                };
                if let Some(n) = ex.prompt_slice(term, background, slices) {
                    self.slice.set(n);
                }
            }
            KeyCode::Right if slices > 1 && shift => self
                .slice
                .set((self.slice.get() + slice_step(slices)) % slices),
            KeyCode::Left if slices > 1 && shift => self
                .slice
                .set((self.slice.get() + slices - slice_step(slices)) % slices),
            KeyCode::Char(']') | KeyCode::Right if slices > 1 => {
                self.slice.set((self.slice.get() + 1) % slices)
            }
            KeyCode::Char('[') | KeyCode::Left if slices > 1 => {
                self.slice.set((self.slice.get() + slices - 1) % slices)
            }
            KeyCode::Char('c') => {
                let stats = ex.cached_stats(&tensor, view);
                let stats_view = self.stats_view(&stats);
                if let Ok(text) = ex.data_plain(
                    &tensor,
                    self.repr,
                    self.slice.get(),
                    view,
                    mode,
                    stats_view,
                    stripe,
                    base,
                    None,
                ) {
                    copy_to_clipboard(&text);
                }
                ex.copied_flash = Some((
                    "copied the screen to the clipboard".to_string(),
                    std::time::Instant::now(),
                ));
            }
            // `y` (copy the reopen command) is engine-owned — see `do_copy_command`.
            KeyCode::Char('l') => {
                self.overlay = Some(Overlay::Legend(match self.repr {
                    Representation::Heatmap => Legend::Heatmap,
                    Representation::Values => Legend::Values,
                }));
            }
            KeyCode::Backspace => return Ok(Outcome::Leave(Nav::Back)),
            KeyCode::Char('\\') => return Ok(Outcome::Leave(Nav::Forward)),
            // Any other key goes back to the detail screen.
            _ => {
                return Ok(Outcome::Leave(Nav::Open(Screen::Detail {
                    tensor: tensor.name.clone(),
                    slice: self.slice.get(),
                })));
            }
        }
        Ok(Outcome::Stay)
    }

    fn handle_mouse(
        &mut self,
        _ex: &mut Explorer,
        _term: &mut crate::tui::LiveTerminal,
        m: MouseEvent,
    ) -> MouseOutcome {
        let slices = self.slices.get();
        match m.kind {
            MouseEventKind::ScrollDown if slices > 1 => {
                self.slice.set((self.slice.get() + 1) % slices);
                MouseOutcome::Redraw
            }
            MouseEventKind::ScrollUp if slices > 1 => {
                self.slice.set((self.slice.get() + slices - 1) % slices);
                MouseOutcome::Redraw
            }
            _ => MouseOutcome::Ignored,
        }
    }

    fn residual(&self) -> Screen {
        Screen::Data {
            tensor: self.tensor_name.clone(),
            repr: self.repr,
            slice: self.slice.get(),
        }
    }
}

/// A command the palette lists and runs, and the single action every tree
/// shortcut dispatches through ([`Explorer::run_command`]). Keeping the command
/// as data (rather than only a match arm) lets the palette enumerate the same set
/// the keys do, and leaves room for future palette-only or global commands.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Cmd {
    Search,
    ExpandAll,
    CollapseAll,
    ViewFiles,
    Stats,
    Health,
    Legend,
    CopyScreen,
    CopyTree,
    CopyPath,
    CopyName,
    CopyCommand,
    Repack,
    Rename,
    Quit,
}

/// A command the file browser's palette lists and runs — its own small registry,
/// since the file view acts on files (not tensors), so the tree's [`Cmd`] actions
/// (copy tensor name, expand groups, …) don't apply. See [`FILE_COMMANDS`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FileCmd {
    TensorTree,
    Legend,
    CopyPath,
    CopyScreen,
    CopyCommand,
    Quit,
}

/// A command the safetensors layout map's palette lists and runs. See
/// [`LAYOUT_COMMANDS`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum LayoutCmd {
    TensorTree,
    Legend,
    CopyScreen,
    CopyCommand,
    Quit,
}

/// A command the tensor **detail** view's palette lists. Each maps to the key
/// that already runs it (the palette synthesizes that key), so no separate
/// dispatch is needed. See [`DETAIL_COMMANDS`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum DetailCmd {
    Heatmap,
    Values,
    Histogram,
    Bins,
    Stats,
    Dtype,
    Reshape,
    FileLayout,
    Legend,
    CopyScreen,
    CopyCommand,
}

/// A command the in-place **rename editor**'s palette lists and runs. The editor
/// has mutable state (the rule pairs) and can leave the view, so it dispatches like
/// the tree/files/layout palettes (Style 1), not by synthesizing a key. Bare
/// letters can't be accelerators here — they're typed into the name fields — so the
/// palette is the primary way to reach copy / legend; the edit / apply commands
/// keep their Ctrl / Enter accelerators. See [`RENAME_COMMANDS`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RenameCmd {
    Apply,
    AddRule,
    RemoveRule,
    CopyScreen,
    CopyReopenCmd,
    CopyApplyCmd,
    Legend,
    Back,
    Quit,
}

/// A command the **checkpoint stats** view palette lists. Like [`DetailCmd`],
/// each maps to the key that already runs it. See [`STATS_COMMANDS`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum StatsCmd {
    FoldShards,
    CopyReport,
    CopyScreen,
    CopyCommand,
    Legend,
    Quit,
}

/// A command the **data view** (heatmap / numeric grid) palette lists. Like
/// [`DetailCmd`], each maps to the key that already runs it. See [`DATA_COMMANDS`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum DataCmd {
    Heatmap,
    Values,
    Layout,
    Zebra,
    Base,
    Dtype,
    Reshape,
    Legend,
    CopyScreen,
    CopyCommand,
}

/// The tree screen's command registry, in palette order: `(command, group,
/// title, the key that also runs it)`. The palette shows each as `Group: Title`
/// (VS Code style); the one-line help beside it is looked up from
/// [`crate::ui::shortcut_help`] by the key, so the palette and hover hints can't
/// drift.
const TREE_COMMANDS: &[(Cmd, &str, &str, char)] = &[
    (Cmd::Search, "Tree", "Search by name", '/'),
    (Cmd::ExpandAll, "Tree", "Expand all groups", 'E'),
    (Cmd::CollapseAll, "Tree", "Collapse all groups", 'C'),
    (Cmd::ViewFiles, "View", "File browser", '\t'),
    (Cmd::Stats, "View", "Checkpoint stats", 's'),
    (Cmd::Health, "View", "Health report", 'h'),
    (Cmd::Legend, "View", "Legend", 'l'),
    (Cmd::CopyScreen, "Copy", "Screen text", 'c'),
    (Cmd::CopyTree, "Copy", "Tree / tensor list…", 't'),
    (Cmd::CopyPath, "Copy", "File path", 'f'),
    (Cmd::CopyName, "Copy", "Tensor name", 'n'),
    (Cmd::CopyCommand, "Copy", "Command to reopen this view", 'y'),
    (Cmd::Repack, "File", "Repack HDF5 into a new codec…", 'r'),
    (Cmd::Rename, "File", "Rename tensors in place…", 'R'),
    (Cmd::Quit, "App", "Quit", 'q'),
];

/// The file browser's command registry, in palette order — the file-view analogue
/// of [`TREE_COMMANDS`]. `\t` (`Tab`) toggles back to the tensor tree.
const FILE_COMMANDS: &[(FileCmd, &str, &str, char)] = &[
    (FileCmd::TensorTree, "View", "Tensor tree", '\t'),
    (FileCmd::Legend, "View", "Legend", 'l'),
    (FileCmd::CopyPath, "Copy", "File path", 'f'),
    (FileCmd::CopyScreen, "Copy", "Screen text", 'c'),
    (
        FileCmd::CopyCommand,
        "Copy",
        "Command to reopen this view",
        'y',
    ),
    (FileCmd::Quit, "App", "Quit", 'q'),
];

/// The layout map's command registry, in palette order. `\t` (`Tab`) returns to
/// the tensor tree.
const LAYOUT_COMMANDS: &[(LayoutCmd, &str, &str, char)] = &[
    (LayoutCmd::TensorTree, "View", "Tensor tree", '\t'),
    (LayoutCmd::Legend, "View", "Legend", 'l'),
    (LayoutCmd::CopyScreen, "Copy", "Screen text", 'c'),
    (
        LayoutCmd::CopyCommand,
        "Copy",
        "Command to reopen this view",
        'y',
    ),
    (LayoutCmd::Quit, "App", "Quit", 'q'),
];

/// The detail view's command registry (palette order). Each `key` is the shortcut
/// the palette synthesizes when the command is chosen; `\t` is `Tab` (file layout).
const DETAIL_COMMANDS: &[(DetailCmd, &str, &str, char)] = &[
    (DetailCmd::Heatmap, "View", "Heatmap", 'm'),
    (DetailCmd::Values, "View", "Numeric values", 'v'),
    (DetailCmd::Histogram, "View", "Value histogram", 'h'),
    (DetailCmd::Bins, "View", "Histogram bin count…", 'b'),
    (DetailCmd::Stats, "View", "Compute statistics", 's'),
    (DetailCmd::Dtype, "View", "Reinterpret dtype…", 'd'),
    (DetailCmd::Reshape, "View", "Reshape…", 'r'),
    (DetailCmd::FileLayout, "View", "File layout", '\t'),
    (DetailCmd::Legend, "View", "Legend", 'l'),
    (DetailCmd::CopyScreen, "Copy", "Screen text", 'c'),
    (
        DetailCmd::CopyCommand,
        "Copy",
        "Command to reopen this view",
        'y',
    ),
];

/// The data view's command registry (palette order). Each `key` is the shortcut
/// the palette synthesizes when the command is chosen.
const DATA_COMMANDS: &[(DataCmd, &str, &str, char)] = &[
    (DataCmd::Heatmap, "View", "Heatmap", 'm'),
    (DataCmd::Values, "View", "Numeric values", 'v'),
    (
        DataCmd::Layout,
        "View",
        "Cycle layout (overview / edges / window)",
        'e',
    ),
    (DataCmd::Zebra, "View", "Cycle zebra striping", 'z'),
    (
        DataCmd::Base,
        "View",
        "Cycle numeral base (dec / hex / oct / bin)",
        'b',
    ),
    (DataCmd::Dtype, "View", "Reinterpret dtype…", 'd'),
    (DataCmd::Reshape, "View", "Reshape…", 'r'),
    (DataCmd::Legend, "View", "Legend", 'l'),
    (DataCmd::CopyScreen, "Copy", "Screen text", 'c'),
    (
        DataCmd::CopyCommand,
        "Copy",
        "Command to reopen this view",
        'y',
    ),
];

/// The checkpoint-stats view's command registry (palette order). `FoldShards`
/// is filtered out when there's no per-shard breakdown to fold (see
/// [`available_stats_commands`]); every shown key appears in the footer, enforced
/// by `every_static_mode_footer_shows_its_command_keys`.
const STATS_COMMANDS: &[(StatsCmd, &str, &str, char)] = &[
    (
        StatsCmd::FoldShards,
        "View",
        "Fold / expand the per-shard breakdown",
        'f',
    ),
    (StatsCmd::Legend, "View", "Legend", 'l'),
    (StatsCmd::CopyReport, "Copy", "Report text", 'r'),
    (StatsCmd::CopyScreen, "Copy", "Screen text", 'c'),
    (
        StatsCmd::CopyCommand,
        "Copy",
        "Command to reopen this view",
        'y',
    ),
    (StatsCmd::Quit, "App", "Quit", 'q'),
];

/// The rename editor's command registry (palette order). The `char` is a *sentinel*
/// naming the real **Ctrl** trigger — Ctrl keys so every character stays typeable in
/// the name fields, mirroring the non-editing modes' bare letters (`^R` apply, `^S`
/// copy screen, `^Y` copy command, `^A` copy apply-command, `^L` legend, `^N`/`^D`
/// add/remove, Esc back, `^C` quit). Every one has a footer key now — nothing is
/// palette-only. [`key_label`] renders them; [`crate::ui::shortcut_help`] (the
/// `Rename` arms) supplies each one's one-line help, keyed by the same char.
const RENAME_COMMANDS: &[(RenameCmd, &str, &str, char)] = &[
    (RenameCmd::Apply, "Rename", "Apply the rename", '\u{12}'), // ^R
    (RenameCmd::AddRule, "Rename", "Add a rule", '\u{e}'),      // ^N
    (
        RenameCmd::RemoveRule,
        "Rename",
        "Remove the focused rule",
        '\u{4}', // ^D
    ),
    (RenameCmd::CopyScreen, "Copy", "Screen text", '\u{13}'), // ^S
    (
        RenameCmd::CopyReopenCmd,
        "Copy",
        "Command to reopen this view",
        '\u{19}', // ^Y (the universal copy-command, mirrors the tree's `y`)
    ),
    (
        RenameCmd::CopyApplyCmd,
        "Copy",
        "Command to apply this rename",
        '\u{1}', // ^A
    ),
    (RenameCmd::Legend, "View", "Legend", '\u{c}'), // ^L
    (RenameCmd::Back, "App", "Back", '\u{1b}'),     // Esc
    (RenameCmd::Quit, "App", "Quit", '\u{3}'),      // ^C
];

/// A resolved palette entry for a command of type `T`: `(command, group, title,
/// key)`. Generic so every view's palette shares the picker
/// ([`Explorer::run_palette`]).
type PaletteRow<T> = (T, &'static str, &'static str, char);
type CmdEntry = PaletteRow<Cmd>;
type FileCmdEntry = PaletteRow<FileCmd>;
type LayoutCmdEntry = PaletteRow<LayoutCmd>;
type DetailCmdEntry = PaletteRow<DetailCmd>;
type DataCmdEntry = PaletteRow<DataCmd>;
type StatsCmdEntry = PaletteRow<StatsCmd>;
type RenameCmdEntry = PaletteRow<RenameCmd>;

/// The display label for a palette/footer key: `Tab` for the `\t` sentinel
/// ([`Cmd::ViewFiles`] / [`FileCmd::TensorTree`]), else the character itself.
/// The rename palette ([`RENAME_COMMANDS`]) uses control-char sentinels for the
/// commands whose real trigger is a Ctrl combo / Enter / Esc (bare letters can't
/// be accelerators there — they're typed into the name fields), and blank-label
/// sentinels for its palette-only commands.
fn key_label(c: char) -> String {
    match c {
        '\t' => "Tab".to_string(),
        '\r' => "Enter".to_string(),
        '\u{12}' => "^R".to_string(),
        '\u{13}' => "^S".to_string(),
        '\u{e}' => "^N".to_string(),
        '\u{4}' => "^D".to_string(),
        '\u{19}' => "^Y".to_string(),
        '\u{1}' => "^A".to_string(),
        '\u{c}' => "^L".to_string(),
        '\u{1b}' => "Esc".to_string(),
        '\u{3}' => "^C".to_string(),
        _ => c.to_string(),
    }
}

/// The tree command bound to key `c`, if any — so the key handler and the palette
/// share one key→command mapping (the registry table).
fn tree_command_for_key(c: char) -> Option<Cmd> {
    TREE_COMMANDS
        .iter()
        .find(|(_, _, _, key)| *key == c)
        .map(|(cmd, _, _, _)| *cmd)
}

/// The file-browser command bound to key `c`, if any — the file-view analogue of
/// [`tree_command_for_key`] (the `\t` sentinel is dispatched from the `Tab` arm,
/// not the `Char` handler, so it never resolves here).
fn file_command_for_key(c: char) -> Option<FileCmd> {
    FILE_COMMANDS
        .iter()
        .find(|(_, _, _, key)| *key == c)
        .map(|(cmd, _, _, _)| *cmd)
}

/// The layout-map command bound to key `c`, if any.
fn layout_command_for_key(c: char) -> Option<LayoutCmd> {
    LAYOUT_COMMANDS
        .iter()
        .find(|(_, _, _, key)| *key == c)
        .map(|(cmd, _, _, _)| *cmd)
}

/// The rename-editor commands available now: Apply only when the staged rename is
/// clean, the copy-apply command only when there's a complete rule (a `convert
/// --map` command exists), and Remove only when there's more than one rule.
fn available_rename_commands(
    applicable: bool,
    has_apply_cmd: bool,
    npairs: usize,
) -> Vec<RenameCmdEntry> {
    RENAME_COMMANDS
        .iter()
        .copied()
        .filter(|(cmd, _, _, _)| match cmd {
            RenameCmd::Apply => applicable,
            RenameCmd::CopyApplyCmd => has_apply_cmd,
            RenameCmd::RemoveRule => npairs > 1,
            _ => true,
        })
        .collect()
}

/// Build the rename editor's [`crate::ui::RenameView`] from the current staged
/// state and render it, returning the preview's max scroll, the footer chip regions
/// (clickable), and the link regions (shard → layout, concrete source → tree).
/// Shared by the live draw, the palette / legend backdrops, and the `c` copy-screen
/// (headless) so they can't drift.
#[allow(clippy::too_many_arguments)]
fn draw_rename_frame(
    f: &mut ratatui::Frame,
    root: &str,
    mode: &RenameMode,
    schemas: &[(String, usize)],
    rules_view: &[crate::ui::RenameRuleView],
    total: usize,
    warnings: &[String],
    has_index: bool,
    applicable: bool,
    synth_err: &Option<String>,
    cli: Option<&str>,
    copied: Option<&str>,
) -> (
    usize,
    crate::ui::ChipRegions,
    crate::ui::LinkRegions,
    Vec<ratatui::layout::Rect>,
    Option<crate::ui::VScrollbar>,
) {
    let completions = if mode.menu.is_some() {
        mode.completions(schemas)
    } else {
        Vec::new()
    };
    let menu_sel = mode
        .menu
        .unwrap_or(0)
        .min(completions.len().saturating_sub(1));
    let display_error = synth_err.clone().or_else(|| mode.error.clone());
    let pairs_disp: Vec<(String, String)> = mode
        .pairs
        .iter()
        .map(|p| (p.source.clone(), p.target.clone()))
        .collect();
    let view = crate::ui::RenameView {
        root,
        pairs: &pairs_disp,
        focus_pair: mode.focus_pair,
        on_target: mode.on_target,
        cursor: mode.cursor,
        menu_open: mode.menu.is_some(),
        menu_sel,
        completions: &completions,
        rules: rules_view,
        total,
        warnings,
        has_index,
        applicable,
        scroll: mode.scroll,
        error: display_error.as_deref(),
        cli,
        copied,
    };
    UI::render_rename(f, &view)
}

/// The detail-view commands available now: dtype / reshape only when the tensor's
/// dtype is reinterpretable, and the file layout only for a local `.safetensors`.
fn available_detail_commands(overridable: bool, layout: bool) -> Vec<DetailCmdEntry> {
    DETAIL_COMMANDS
        .iter()
        .copied()
        .filter(|(cmd, _, _, _)| match cmd {
            DetailCmd::Dtype | DetailCmd::Reshape => overridable,
            DetailCmd::FileLayout => layout,
            _ => true,
        })
        .collect()
}

/// The key a chosen [`DetailCmd`] maps to — the palette synthesizes this so the
/// detail loop's existing key handlers run it (`\t` → `Tab` = file layout).
fn detail_cmd_key(cmd: DetailCmd) -> KeyEvent {
    let entry = DETAIL_COMMANDS.iter().find(|(c, ..)| *c == cmd);
    match entry.map(|(_, _, _, key)| *key) {
        Some('\t') => KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
        Some(ch) => KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE),
        None => KeyEvent::new(KeyCode::Null, KeyModifiers::NONE),
    }
}

/// The data-view commands available now: dtype / reshape only when the tensor's
/// dtype is reinterpretable.
fn available_data_commands(overridable: bool) -> Vec<DataCmdEntry> {
    DATA_COMMANDS
        .iter()
        .copied()
        .filter(|(cmd, _, _, _)| match cmd {
            DataCmd::Dtype | DataCmd::Reshape => overridable,
            _ => true,
        })
        .collect()
}

/// The key a chosen [`DataCmd`] maps to — synthesized so the data loop's existing
/// key handlers run it.
fn data_cmd_key(cmd: DataCmd) -> KeyEvent {
    let key = DATA_COMMANDS
        .iter()
        .find(|(c, ..)| *c == cmd)
        .map_or('\0', |(_, _, _, key)| *key);
    KeyEvent::new(KeyCode::Char(key), KeyModifiers::NONE)
}

/// The stats-view commands available now: the per-shard fold only when there's a
/// breakdown to fold (more than one shard on disk).
fn available_stats_commands(has_fold: bool) -> Vec<StatsCmdEntry> {
    STATS_COMMANDS
        .iter()
        .copied()
        .filter(|(cmd, _, _, _)| match cmd {
            StatsCmd::FoldShards => has_fold,
            _ => true,
        })
        .collect()
}

/// The key a chosen [`StatsCmd`] maps to — synthesized so the stats mode's key
/// handlers run it.
fn stats_cmd_key(cmd: StatsCmd) -> KeyEvent {
    let key = STATS_COMMANDS
        .iter()
        .find(|(c, ..)| *c == cmd)
        .map_or('\0', |(_, _, _, key)| *key);
    KeyEvent::new(KeyCode::Char(key), KeyModifiers::NONE)
}

/// How the file browser (`Tab` / `--files`) browses a **remote** (`--ssh-read`)
/// checkpoint — it adapts to the source kind (`None` for a local checkpoint, which
/// walks the local filesystem):
/// - `Sftp(dir)`: a remote safetensors directory, listed recursively over SFTP
///   (full browse — layout map + sidecar preview via header/small-file reads).
/// - `S3(uri)`: an `s3://…` cstorch source, listed s3-natively (objects by key +
///   size via one boto3 `list_objects_v2`) — browse-only, no per-object view.
#[derive(Debug, Clone)]
enum RemoteBrowse {
    Sftp(String),
    S3(String),
}

/// Everything that makes a run **remote** (`--ssh-read`), bundled so it exists as
/// a unit or not at all: a local checkpoint is simply `remote: None`, and the SSH
/// session / cached password / captured on-disk usage / S3 metadata can't dangle
/// without a remote read behind them (the old six parallel `Option`/`RefCell`
/// fields let those states drift apart).
struct RemoteContext {
    /// The reader that runs cstorch/SFTP over SSH (`--ssh-read`/`--ssh-venv`).
    read: crate::remote::RemoteRead,
    /// The file browser's source kind, derived from the raw source at setup;
    /// `None` only for the degenerate empty-input case (nothing to browse).
    browse: Option<RemoteBrowse>,
    /// The one SSH session kept alive for the run: opened during the pre-TUI read
    /// and reused (reopened on idle-timeout with the cached password) by the file
    /// browser / remote layout / sidecar reads without a second auth prompt.
    session: RefCell<Option<crate::sftp::RemoteSession>>,
    /// The entered password, cached so a mid-session reopen doesn't re-prompt.
    password: RefCell<Option<String>>,
    /// The remote shards' on-disk footprint, captured during the read while the
    /// session was live (SFTP carries no block count, so it can't be re-derived
    /// later). `None` for `s3://` / hosts without GNU `stat`.
    disk: Option<crate::stats::DiskUsage>,
    /// The underlying S3 objects' metadata (checksums/ETags/tags/sizes) for an
    /// `s3://` source, shown in the stats report's S3 section. `None` for SFTP.
    s3_meta: Option<crate::remote::S3Meta>,
}

pub struct Explorer {
    /// The input path list (CLI args / globbed shard paths) — what to read. Not
    /// the same as the model's walked `FileEntry` list; this is the frontend's
    /// idea of "what the user asked to open".
    files: Vec<PathBuf>,
    /// The kernel session — the single authoritative owner of the checkpoint's
    /// canonical primary data (deduped + natural-sorted tensors, metadata, config,
    /// parameter count) and, for a local read, the serializable model. Populated
    /// once at load; `Some` afterwards. The Explorer reads its tensors/metadata/
    /// config through it (see [`Self::tensors`]/[`Self::metadata`]/[`Self::config`])
    /// so no frontend copy can drift. `Some` for a local
    /// checkpoint (built by `core::readers::read_local`); the remote readers fill
    /// it in a later step. The TUI is migrating to drive this `Session` and render
    /// its `ViewModel`; today its local views/reports derive from the model here.
    session: Option<crate::kernel::Session>,
    /// Everything remote (`--ssh-read`), or `None` for a local checkpoint. Bundles
    /// the reader, browse source, live SSH session + cached password, captured
    /// on-disk usage, and S3 metadata — see [`RemoteContext`].
    remote: Option<RemoteContext>,
    /// Whether the whole checkpoint structure has been read. A direct
    /// `--tensor X` open reads just that tensor first (fast path), leaving this
    /// `false` until the tree is shown and the full load runs.
    full_loaded: bool,
    /// The tensor-tree browser state (tree, flattened/filtered rows, selection,
    /// scroll, search) — owned by the kernel; the interactive tree screen drives
    /// and renders from it.
    tree_state: crate::kernel::TreeState,
    /// Transient "✓ Copied …" confirmation shown after a copy shortcut
    /// (`c`/`f`/`n`) as a bottom-line overlay — leaving the path/name in the
    /// status bar intact — paired with the time it was set so it clears on its
    /// own after `COPY_FLASH` (and on the next key press), like the data views.
    copied_flash: Option<(String, std::time::Instant)>,
    /// The live Ratatui terminal, owned for the duration of the interactive loop
    /// (`None` headlessly and before/after `run`).
    terminal: Option<crate::tui::LiveTerminal>,
    /// Clickable regions for the frame currently on screen: each footer key-hint
    /// chip and the `[×]` close control, paired with the `KeyEvent` a click on it
    /// synthesizes. Rebuilt every frame by the `render_*` functions; read by the
    /// loops' mouse handlers to turn a click into the equivalent keypress.
    clickable: RefCell<Vec<(ratatui::layout::Rect, KeyEvent)>>,
    /// Clickable **links** for the frame on screen: a safetensors filename or a
    /// concrete tensor name paired with where it jumps (a layout view / the tree).
    /// The app-wide counterpart to `clickable` — rebuilt each frame by the screens
    /// that show such names; a click routes through [`Self::open_link`].
    links: RefCell<Vec<(ratatui::layout::Rect, crate::ui::Link)>>,
    /// The current frame's vertical scroll bar, when the active mode's body
    /// overflows. Recorded by each mode's render (like `clickable`) and drawn +
    /// drag-scrubbed by the `run_mode` engine, so every mode gets a bar the same
    /// way — see [`crate::ui::VScrollbar`].
    vscrollbar: RefCell<Option<crate::ui::VScrollbar>>,
    /// Whether a scroll-bar drag is in progress (engine-owned, shared by every
    /// mode's bar).
    scrollbar_drag: bool,
    /// Index/file mismatches, shown as a warning panel. Populated in
    /// [`Self::finalize_load`] from [`Self::index_specs`] once the tensors are
    /// read (plus any remote index health folded in by the loader).
    health_reports: Vec<crate::health::HealthReport>,
    /// Parsed `model.safetensors.index.json`(s) to health-check against the loaded
    /// tensors — deferred to `finalize_load` so the shard headers are read once (by
    /// the loader), not again for the check.
    index_specs: Vec<crate::health::IndexSpec>,
    /// The **data-view presentation state** (dtype/shape overrides, histogram
    /// bucket count, layout + edge-split/window-offset knobs, zebra/base toggles)
    /// — session-remembered. Kernel-owned; see [`crate::kernel::DataViewState`].
    data_view: crate::kernel::DataViewState,
    /// Per-tensor fused-codebook packing schema parsed from metadata at load.
    /// A tensor with a schema defaults to the [`ViewDtype::Unpacked`] view.
    packing_schemas: HashMap<String, PackingSchema>,
    /// Exact whole-tensor statistics, cached per (tensor name, view) since the
    /// scan is expensive. Session-scoped.
    stats_cache: RefCell<HashMap<(String, ViewDtype), Stats>>,
    /// Cached whole-tensor histograms, keyed like the stats cache plus the
    /// requested bucket count (so a different `--bins` / `b` count caches and
    /// redraws separately rather than reusing a stale layout).
    histogram_cache: RefCell<HashMap<HistKey, Histogram>>,
    /// Held-key scroll acceleration: the last navigation key, when it fired, and
    /// how many consecutive fast repeats (terminal auto-repeat) it's had — so
    /// holding ↑/↓ (tree) or an arrow (data view) ramps the step up. See
    /// [`Explorer::held_step`].
    scroll_accel: Cell<Option<(KeyCode, std::time::Instant, u32)>>,
    /// A tensor/view to jump straight to on startup (from CLI flags); consumed
    /// once after loading, then normal browsing resumes.
    open: Option<OpenRequest>,
    /// The open reader for the data view's current tensor, reused across redraws
    /// (replaced when the viewed tensor changes). See [`CachedReader`].
    reader_cache: RefCell<Option<CachedReader>>,
    /// The last sampled grid a data view drew, reused across identical redraws
    /// (e.g. the spinner ticks during a stats scan). See [`CachedSample`].
    sample_cache: RefCell<Option<CachedSample>>,
    /// Whether to compute a tensor's exact stats in the background when its
    /// detail screen opens. Reading the whole tensor warms the OS/disk cache (the
    /// dominant cost on NFS), so the heatmap/numeric view opens fast; the scan is
    /// shown live on the detail screen's Statistics line. Off via `--no-preload`.
    preload: bool,
    /// `source_path`s of files present on disk but not referenced by a
    /// `model.safetensors.index.json` (derived from the health reports); their
    /// tensors are flagged in the tree and detail screens.
    unindexed: HashSet<String>,
    /// Which bottom-right status badge the mouse is over (index into the current
    /// screen's [`Self::screen_badges`]), which floats that badge's hover bubble.
    /// Set on mouse-move, `None` when over none. One field for all badges.
    hovered_badge: Cell<Option<usize>>,
    /// The footer shortcut chip the mouse is hovering (its on-screen rect + help
    /// text), which floats a help bubble beside it. Set on mouse-move, cleared on
    /// any other event so it never lingers onto the next screen.
    hovered_shortcut: Cell<Option<(ratatui::layout::Rect, &'static str)>>,
    /// Derived reports cached for the session — the loaded checkpoint is
    /// immutable, so the health-check report (`h`) and the stats (`s`) are each
    /// an O(tensors) pass computed once and reused, rather than recomputed every
    /// time the popup opens.
    cached_check: RefCell<Option<crate::check::CheckReport>>,
    checkpoint_stats_cache: RefCell<Option<crate::stats::CheckpointStats>>,
    /// Whether the checkpoint's shard files are actually writable — probed once (a
    /// local safetensors checkpoint on a read-only filesystem, e.g. an `ro` SSH
    /// mount, or a read-only file, looks local but can't be renamed in place).
    /// Gates the `editable` badge and the whole in-place-rename capability.
    writable: Cell<Option<bool>>,
    /// The selected node's distinct source files (keyed by selection index, tree
    /// length, and search mode), so a selected *group* isn't re-walked
    /// (`collect_source_paths`, O(tensors)) on every status-bar redraw *and* every
    /// `f`/`t` copy — the walk happens once per selection and both reuse it.
    cached_group_files: RefCell<GroupFilesCache>,
    /// The directory the file browser (`Tab`) lists — the checkpoint's own
    /// directory (the common parent of its shards). Fixed for the session.
    browse_root: PathBuf,
    /// The file-browser state (directory tree + flattened rows + selection/scroll)
    /// — kernel-owned, built lazily on first `Tab` and driven by the file screen.
    file_state: crate::kernel::FileState,
}

impl Explorer {
    pub fn new(
        files: Vec<PathBuf>,
        index_specs: Vec<crate::health::IndexSpec>,
        open: Option<OpenRequest>,
        preload: bool,
    ) -> Self {
        // `health_reports` / `unindexed` are computed in `finalize_load`, once the
        // tensors are read, from `index_specs` — so no shard header is read twice.
        let browse_root = browse_root_of(&files);
        Self {
            files,
            session: None,
            remote: None,
            full_loaded: false,
            tree_state: crate::kernel::TreeState::default(),
            copied_flash: None,
            terminal: None,
            clickable: RefCell::new(Vec::new()),
            links: RefCell::new(Vec::new()),
            vscrollbar: RefCell::new(None),
            scrollbar_drag: false,
            health_reports: Vec::new(),
            index_specs,
            data_view: crate::kernel::DataViewState::default(),
            packing_schemas: HashMap::new(),
            stats_cache: RefCell::new(HashMap::new()),
            histogram_cache: RefCell::new(HashMap::new()),
            scroll_accel: Cell::new(None),
            open,
            reader_cache: RefCell::new(None),
            sample_cache: RefCell::new(None),
            preload,
            unindexed: HashSet::new(),
            hovered_badge: Cell::new(None),
            hovered_shortcut: Cell::new(None),
            cached_check: RefCell::new(None),
            checkpoint_stats_cache: RefCell::new(None),
            writable: Cell::new(None),
            cached_group_files: RefCell::new(None),
            browse_root,
            file_state: crate::kernel::FileState::default(),
        }
    }

    /// Update the hovered-shortcut help from a mouse position: the footer chip
    /// under `(col, row)` on screen `ctx` (with a help string), else `None`.
    /// Feeds the help bubble drawn by the render paths.
    fn update_shortcut_hover(&self, ctx: HelpCtx, col: u16, row: u16) {
        let hovered = crate::ui::region_at(&self.clickable.borrow(), col, row)
            .and_then(|(rect, key)| crate::ui::shortcut_help(key, ctx).map(|h| (rect, h)));
        self.hovered_shortcut.set(hovered);
    }

    /// The bottom-right status badges for screen `ctx`, in right-to-left order — the
    /// single source of truth both the renderer and the hover / click hit-test use.
    /// The access badge shows on every browsing screen; the health and
    /// metadata-only badges only on the tree.
    fn screen_badges(&self, ctx: HelpCtx) -> Vec<crate::ui::Badge> {
        let (health, metadata_only) = if ctx == HelpCtx::Tree {
            (self.health_alert(), self.remote_read().is_some())
        } else {
            (None, false)
        };
        match ctx {
            // The layout map draws no status bar.
            HelpCtx::Layout => Vec::new(),
            _ => crate::ui::status_badges(self.access_badge(), health, metadata_only),
        }
    }

    /// Refresh every hover-bubble state from the mouse position `(col, row)` on a
    /// frame of `width`×`height`: the status badge under the cursor and the footer
    /// shortcut chip. Called on every mouse-move — by the browsing loops *and* the
    /// pop-up loops, so the bubbles stay live (rather than freezing) while a pop-up
    /// floats over the tree.
    fn update_hovers(&self, ctx: HelpCtx, width: u16, height: u16, col: u16, row: u16) {
        let badges = self.screen_badges(ctx);
        self.hovered_badge
            .set(UI::badge_bar_hit(width, height, col, row, &badges));
        self.update_shortcut_hover(ctx, col, row);
    }

    /// Float a tree pop-up until a key or click dismisses it, redrawing `draw`
    /// each iteration so the underlying hover bubbles stay live: a mouse-move
    /// refreshes them (read-only / health badge, footer chip) instead of freezing
    /// whatever the cursor was over when the pop-up opened. Ctrl-C still quits;
    /// wheel/drag are ignored (so the command text can be selected by hand).
    fn float_until_dismissed(
        &self,
        term: &mut crate::tui::LiveTerminal,
        mut draw: impl FnMut(&mut ratatui::Frame),
    ) {
        // A wrong-keyboard-layout key flashes the same hint as the main views
        // (rather than dismissing the pop-up), so a mistaken shortcut is explained
        // even with a pop-up up; cleared on the next input.
        let mut layout_hint: Option<char> = None;
        loop {
            let hint = layout_hint;
            if term
                .draw(|f| {
                    draw(f);
                    if let Some(c) = hint {
                        UI::render_notice(f, &layout_hint_msg(c));
                    }
                })
                .is_err()
            {
                return;
            }
            layout_hint = None;
            match event::read() {
                Ok(Event::Key(key)) => {
                    if is_ctrl_c(&key) {
                        quit_immediately();
                    }
                    if let Some(c) = wrong_layout_char(&key) {
                        layout_hint = Some(c);
                        continue; // warn, stay open
                    }
                    return;
                }
                Ok(Event::Mouse(m)) if matches!(m.kind, MouseEventKind::Down(_)) => return,
                Ok(Event::Mouse(m)) if matches!(m.kind, MouseEventKind::Moved) => {
                    if let Ok(sz) = term.size() {
                        self.update_hovers(HelpCtx::Tree, sz.width, sz.height, m.column, m.row);
                    }
                }
                Ok(_) => {}       // other mouse / resize: redraw
                Err(_) => return, // input closed
            }
        }
    }

    /// Composite the hovered shortcut's help bubble, if any — drawn last on every
    /// screen so it floats over the footer chips.
    fn render_shortcut_hover(&self, frame: &mut ratatui::Frame) {
        if let Some((rect, help)) = self.hovered_shortcut.get() {
            crate::ui::render_shortcut_bubble(frame, rect, help);
        }
    }

    /// Run `f` with an open reader for `t`, reusing the cached one when it is
    /// still for the same tensor and opening (and caching) a fresh one otherwise.
    /// Lets the data view re-sample on every pan / slice step without paying the
    /// file-open cost each frame.
    fn with_reader<R>(
        &self,
        t: &TensorInfo,
        f: impl FnOnce(&dyn crate::sample::TensorReader) -> Result<R, String>,
    ) -> Result<R, String> {
        {
            let mut cache = self.reader_cache.borrow_mut();
            let stale = cache
                .as_ref()
                .is_none_or(|c| c.source_path != t.source_path || c.name != t.name);
            if stale {
                let reader = crate::sample::open_reader(t)?;
                *cache = Some(CachedReader {
                    source_path: t.source_path.clone(),
                    name: t.name.clone(),
                    reader,
                });
            }
        }
        let cache = self.reader_cache.borrow();
        f(cache.as_ref().unwrap().reader.as_ref())
    }

    /// Cached exact statistics for `(tensor, view)`, or `None` if not yet
    /// computed (cheap lookup — never scans).
    fn cached_stats(&self, tensor: &TensorInfo, view: ViewDtype) -> Option<Stats> {
        self.stats_cache
            .borrow()
            .get(&(tensor.name.clone(), view))
            .copied()
    }

    /// Start a statistics scan for `(tensor, view)` on a worker thread. Used by
    /// the data view, which polls the returned [`ScanJob`] and stays interactive
    /// while it runs (see [`Self::run_data`]).
    fn spawn_stats_scan(&self, tensor: &TensorInfo, view: ViewDtype) -> ScanJob {
        let cancel = Arc::new(AtomicBool::new(false));
        let pause = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicUsize::new(0));
        let owned = tensor.clone();
        let schema = self.schema_for(&tensor.name).cloned();
        let worker_cancel = Arc::clone(&cancel);
        let worker_pause = Arc::clone(&pause);
        let worker_done = Arc::clone(&done);
        let handle = std::thread::spawn(move || {
            crate::sample::tensor_stats(
                &owned,
                view,
                schema.as_ref(),
                &worker_cancel,
                &worker_pause,
                Some(&*worker_done),
            )
        });
        ScanJob {
            view,
            cancel,
            pause,
            handle: Some(handle),
            started: std::time::Instant::now(),
            done,
            total: tensor.size_bytes,
        }
    }

    /// Compute and cache exact statistics for `(tensor, view)` on a miss. The
    /// scan runs on a worker thread; while it runs, `redraw` is called each frame
    /// with a [`StatsView::Computing`] state so the caller can animate a spinner
    /// *in place* on its own screen. Ctrl-C quits the app; **any other key
    /// cancels** the scan (the worker stops at the next block) and returns
    /// [`ScanOutcome::Cancelled`] right away, so a slow scan never traps the UI.
    /// Small tensors finish before the spinner ever appears.
    fn compute_stats_animated(
        &self,
        term: &mut crate::tui::LiveTerminal,
        tensor: &TensorInfo,
        view: ViewDtype,
        render: impl Fn(&mut ratatui::Frame, StatsView),
    ) -> ScanOutcome {
        if self.cached_stats(tensor, view).is_some() {
            return ScanOutcome::Completed;
        }

        // `cancel` lets a key press abort the scan cooperatively; we set it and
        // return without joining, so the UI is responsive and the worker winds
        // down on its own at the next block boundary.
        let cancel = Arc::new(AtomicBool::new(false));
        // The detail-screen scan has nothing to interleave with, so it never pauses.
        let pause = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicUsize::new(0));
        let total = tensor.size_bytes;
        let owned = tensor.clone();
        let schema = self.schema_for(&tensor.name).cloned();
        let worker_cancel = Arc::clone(&cancel);
        let worker_pause = Arc::clone(&pause);
        let worker_done = Arc::clone(&done);
        let handle = std::thread::spawn(move || {
            crate::sample::tensor_stats(
                &owned,
                view,
                schema.as_ref(),
                &worker_cancel,
                &worker_pause,
                Some(&*worker_done),
            )
        });

        const SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
        let started = std::time::Instant::now();
        let mut frame = 0usize;
        while !handle.is_finished() {
            // Only animate once it's clearly not instant, to avoid a flash for
            // small tensors (which return before the first frame).
            if started.elapsed() >= std::time::Duration::from_millis(120) {
                let sv = StatsView::Computing {
                    spinner: SPINNER[frame % SPINNER.len()],
                    elapsed: started.elapsed(),
                    progress: (total > 0)
                        .then(|| (done.load(Ordering::Relaxed) as f64 / total as f64).min(1.0)),
                };
                let _ = term.draw(|f| render(f, sv));
                frame += 1;
            }
            // Frame delay that also stays responsive to keys while we wait.
            if event::poll(std::time::Duration::from_millis(80)).unwrap_or(false)
                && let Ok(Event::Key(key)) = event::read()
            {
                if is_ctrl_c(&key) {
                    quit_immediately();
                }
                cancel.store(true, Ordering::Relaxed);
                return ScanOutcome::Cancelled;
            }
        }

        match handle.join() {
            Ok(Ok(s)) => {
                self.stats_cache
                    .borrow_mut()
                    .insert((tensor.name.clone(), view), s);
                ScanOutcome::Completed
            }
            // Surface a failure instead of silently doing nothing.
            Ok(Err(msg)) => {
                let _ = term.draw(|f| UI::render_message(f, "Statistics unavailable", &msg));
                let _ = event::read();
                ScanOutcome::Completed
            }
            Err(_) => {
                let _ = term.draw(|f| {
                    UI::render_message(f, "Statistics unavailable", "the scan thread panicked")
                });
                let _ = event::read();
                ScanOutcome::Completed
            }
        }
    }

    /// Read `s3://…` sources' metadata over SSH via cstorch on `host` (activating
    /// the venv at `venv`), instead of directly — so credentials stay on the
    /// remote (`--ssh-read` / `--ssh-venv`).
    pub fn set_remote_read(&mut self, host: String, venv: String) {
        // The file browser adapts to the remote source kind, derived from the raw
        // `--ssh-read` argument: an `s3://…` URI browses s3-natively; any other
        // path is the SFTP directory to browse (or, for a single shard, its
        // parent). `browse_root` (a local `.parent()`) doesn't apply remotely.
        let browse = self.files.first().map(|first| {
            let src = first.to_string_lossy().into_owned();
            if src.starts_with("s3://") {
                RemoteBrowse::S3(src)
            } else if src.ends_with(".safetensors") {
                let dir = src
                    .rsplit_once('/')
                    .map(|(d, _)| d.to_string())
                    .unwrap_or_else(|| ".".to_string());
                RemoteBrowse::Sftp(dir)
            } else {
                RemoteBrowse::Sftp(src.trim_end_matches('/').to_string())
            }
        });
        self.remote = Some(RemoteContext {
            read: crate::remote::RemoteRead::new(host, venv),
            browse,
            session: RefCell::new(None),
            password: RefCell::new(None),
            disk: None,
            s3_meta: None,
        });
    }

    /// The remote reader (`--ssh-read`), when this is a remote run.
    fn remote_read(&self) -> Option<&crate::remote::RemoteRead> {
        self.remote.as_ref().map(|r| &r.read)
    }

    /// The file browser's remote source kind, when browsing a remote checkpoint.
    fn remote_browse(&self) -> Option<&RemoteBrowse> {
        self.remote.as_ref().and_then(|r| r.browse.as_ref())
    }

    /// The captured remote on-disk usage, if any.
    fn remote_disk(&self) -> Option<crate::stats::DiskUsage> {
        self.remote.as_ref().and_then(|r| r.disk.clone())
    }

    /// The remote S3 object metadata, if any.
    fn remote_s3_meta(&self) -> Option<&crate::remote::S3Meta> {
        self.remote.as_ref().and_then(|r| r.s3_meta.as_ref())
    }

    /// Run `f` with the live remote session, reopening once (with the cached
    /// password) if the stored session errors — a `--ssh-read` connection can idle
    /// out between the initial read and a later browse. All remote file-browser /
    /// layout / sidecar reads go through this so they never re-prompt. Errors only
    /// when there's no `--ssh-read` configured or the (re)open itself fails.
    fn with_remote_session<T>(
        &self,
        f: impl Fn(&crate::sftp::RemoteSession) -> Result<T>,
    ) -> Result<T> {
        let rc = self
            .remote
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no --ssh-read session configured"))?;
        // Try the stored session first; a success returns immediately.
        if let Some(session) = rc.session.borrow().as_ref()
            && let Ok(v) = f(session)
        {
            return Ok(v);
        }
        // No session, or it errored (likely an idle timeout): reopen once with the
        // cached password (entered during the pre-TUI read, so no new prompt).
        let session = {
            let mut pw = rc.password.borrow_mut();
            rc.read.open_with(&mut pw)?
        };
        let out = f(&session);
        *rc.session.borrow_mut() = Some(session);
        out
    }

    fn load_all_files(&mut self) -> Result<()> {
        // Already loaded (e.g. a remote `--ssh-read` structure read synchronously
        // before the TUI started) — don't re-read.
        if self.full_loaded {
            return Ok(());
        }

        // Read the checkpoint structure on a worker thread so the UI stays
        // responsive: a cold file (e.g. a large HDF5 over the network) can take
        // seconds to enumerate, and we'd otherwise show an empty screen. Animate
        // a loading frame — the same header/footer chrome as the tree, with a
        // spinner in place of the rows — until the worker finishes.
        let files = self.files.clone();
        let remote = self.remote_read().cloned();
        let handle = std::thread::spawn(move || Self::gather_checkpoint(&files, remote.as_ref()));

        let label = self
            .files
            .first()
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        let total = self.files.len();
        let started = std::time::Instant::now();
        let mut frame = 0usize;
        loop {
            // Wait one ~12 fps tick — also catches `q` / Ctrl-C to abort. Polling
            // *before* drawing means a fast (cached) load finishes within the
            // first tick and never flashes the spinner; only a slow load reaches
            // the draw below.
            if event::poll(std::time::Duration::from_millis(80)).unwrap_or(false)
                && let Ok(Event::Key(key)) = event::read()
                && (is_ctrl_c(&key) || matches!(key.code, KeyCode::Char('q')))
            {
                quit_immediately();
            }
            if handle.is_finished() {
                break;
            }
            // Animate through the live terminal when one is up (the interactive
            // session); headless `--plain` uses `load_quiet` and never gets here.
            if let Some(term) = self.terminal.as_mut() {
                let spinner = STATS_SPINNER[frame % STATS_SPINNER.len()];
                let elapsed = started.elapsed();
                let _ = term.draw(|f| UI::render_loading(f, &label, total, spinner, elapsed));
            }
            frame += 1;
        }
        let ((tensors, metadata, config, disk, health), checkpoint) = handle
            .join()
            .map_err(|_| anyhow::anyhow!("checkpoint loader thread panicked"))??;
        if let Some(rc) = self.remote.as_mut() {
            rc.disk = disk;
        }
        // Remote index/file health (empty for a local read, whose reports were
        // gathered up front); fold it in so the popup and `⚠ health` badge show it.
        self.health_reports.extend(health);
        self.finalize_load(tensors, metadata, config, checkpoint);
        Ok(())
    }

    /// Read the checkpoint structure synchronously, with no loading animation —
    /// for `--plain`, which renders once to a buffer and must not write spinner
    /// frames to stdout.
    fn load_quiet(&mut self) -> Result<()> {
        // A remote read opens one SSH session and keeps it alive on `self`, so the
        // file browser (and remote layout / sidecar reads) reuse it without a
        // second auth prompt. A local read uses the plain static gatherer.
        let ((tensors, metadata, config, disk, health), checkpoint) =
            if self.remote_read().is_some() {
                self.gather_remote_keeping_session()?
            } else {
                Self::gather_checkpoint(&self.files, None)?
            };
        if let Some(rc) = self.remote.as_mut() {
            rc.disk = disk;
        }
        self.health_reports.extend(health);
        self.finalize_load(tensors, metadata, config, checkpoint);
        Ok(())
    }

    /// Remote (`--ssh-read`) structure read that **keeps** the SSH session alive on
    /// `self` — stashed in [`Self::remote_session`], its password in
    /// [`Self::remote_password`] — so the file browser and remote layout / sidecar
    /// reads reuse it without re-prompting. Mirrors
    /// [`crate::remote::RemoteRead::fetch_with_config`] but over one owned session.
    fn gather_remote_keeping_session(
        &mut self,
    ) -> Result<(CheckpointParts, Option<crate::model::Checkpoint>)> {
        let r = self
            .remote_read()
            .cloned()
            .expect("gather_remote_keeping_session requires --ssh-read");
        // One authenticated session for the whole run; the password entered here is
        // cached (any later reopen after an idle timeout reuses it silently).
        let session = {
            let mut pw = self.remote.as_ref().unwrap().password.borrow_mut();
            r.open_with(&mut pw)?
        };
        eprintln!("checkpoint-explorer: reading tensor metadata over ssh …");

        let mut tensors: Vec<TensorInfo> = Vec::new();
        let mut metadata: Vec<MetadataInfo> = Vec::new();
        let mut config: Option<crate::config::ModelConfig> = None;
        let mut disk_shards: Vec<crate::stats::ShardDisk> = Vec::new();
        let mut health: Vec<crate::health::HealthReport> = Vec::new();
        let mut s3_meta: Option<crate::remote::S3Meta> = None;
        for file_path in &self.files {
            let as_str = file_path.to_string_lossy().into_owned();
            let bars = crate::progress::Bars::start(vec![as_str.clone()]);
            let progress = bars.progress(0);
            // Fetch the S3 object metadata up front for an `s3://` source (checksums
            // /ETags/tags — a HEAD per object), so the stats report's S3 section is
            // ready; `want_s3` is a no-op for an SFTP source.
            let pw = self.remote.as_ref().unwrap().password.borrow().clone();
            let out = r.read(&session, &as_str, &pw, progress.as_deref(), true);
            bars.finish(0, out.is_ok());
            bars.join();
            let rc = out?;
            tensors.extend(rc.tensors);
            metadata.extend(rc.metadata);
            if let Some(d) = rc.disk {
                disk_shards.extend(d.shards);
            }
            health.extend(rc.health);
            if let Some(s3) = rc.s3 {
                s3_meta = Some(s3); // one `s3://` source per run
            }
            if config.is_none() {
                config = r.read_config(&session, &as_str);
            }
        }
        *self.remote.as_ref().unwrap().session.borrow_mut() = Some(session);

        // Build the central model from what was just read — no extra network I/O:
        // group tensors by their source file into shard headers, roll the on-disk
        // `stat` results into file entries, and carry config + S3 metadata. The
        // remote views still read over SSH lazily; this makes the model (and
        // `--print-model`, serialization) cover remote checkpoints too.
        let (root, source) = match self.remote_browse() {
            Some(RemoteBrowse::Sftp(dir)) => (
                format!("{}:{dir}", self.remote_host_label()),
                crate::model::Source::Sftp {
                    host: self.remote_host_label(),
                    root: dir.clone(),
                },
            ),
            Some(RemoteBrowse::S3(uri)) => {
                (uri.clone(), crate::model::Source::S3 { uri: uri.clone() })
            }
            None => (String::new(), crate::model::Source::Local),
        };
        let mut order: Vec<String> = Vec::new();
        let mut by_src: HashMap<String, Vec<TensorInfo>> = HashMap::new();
        for t in &tensors {
            if !by_src.contains_key(&t.source_path) {
                order.push(t.source_path.clone());
            }
            by_src
                .entry(t.source_path.clone())
                .or_default()
                .push(t.clone());
        }
        let mut shards: Vec<crate::model::ShardHeader> = order
            .iter()
            .enumerate()
            .map(|(i, src)| crate::model::ShardHeader {
                path: src.clone(),
                total_len: 0,
                header_len: 0,
                tensors: by_src.remove(src).unwrap_or_default(),
                // All `__metadata__` on the first shard (it isn't keyed per file).
                metadata: if i == 0 { metadata.clone() } else { Vec::new() },
            })
            .collect();
        if shards.is_empty() && !metadata.is_empty() {
            shards.push(crate::model::ShardHeader {
                path: root.clone(),
                total_len: 0,
                header_len: 0,
                tensors: Vec::new(),
                metadata: metadata.clone(),
            });
        }
        let files = disk_shards
            .iter()
            .map(|d| crate::model::FileEntry {
                rel_path: d.name.clone(),
                name: d.name.clone(),
                depth: 0,
                mode: None,
                mtime: None,
                inode: None, // remote reads carry no inode identity
                node: crate::model::FsNode::File {
                    apparent: d.apparent,
                    allocated: d.allocated,
                    kind: crate::filetree::FileKind::of(&d.name),
                    links: 1,
                },
            })
            .collect();
        let cp = crate::model::Checkpoint {
            source,
            root,
            files,
            shards,
            config: config.clone(),
            index: Vec::new(),
            s3: s3_meta.clone(),
        };

        if let Some(rc) = self.remote.as_mut() {
            rc.s3_meta = s3_meta;
        }
        let parts = (
            tensors,
            metadata,
            config,
            crate::stats::DiskUsage::from_shards(disk_shards),
            health,
        );
        Ok((parts, Some(cp)))
    }

    /// The health badge's alert level: red for a real error (a referenced file or
    /// tensor is missing on disk), a softer orange when there are only warnings
    /// (e.g. extra files on disk), `None` when there's nothing to flag.
    fn health_alert(&self) -> Option<crate::ui::HealthAlert> {
        if self.health_reports.is_empty() {
            None
        } else if self.health_reports.iter().any(|r| r.has_errors()) {
            Some(crate::ui::HealthAlert::Error)
        } else {
            Some(crate::ui::HealthAlert::Warning)
        }
    }

    /// Files on disk but absent from the index (per the health reports' extra
    /// files), resolved to absolute paths so they match each tensor's
    /// `source_path` — the tree dims their rows.
    fn unindexed_files(reports: &[crate::health::HealthReport]) -> HashSet<String> {
        let mut unindexed = HashSet::new();
        for report in reports {
            if let Some(dir) = Path::new(&report.index_path).parent() {
                for file in &report.extra_files {
                    unindexed.insert(absolute_path(&dir.join(file)));
                }
            }
        }
        unindexed
    }

    /// Shared post-read setup: dedup, sort, parameter/schema/tree build.
    fn finalize_load(
        &mut self,
        tensors: Vec<TensorInfo>,
        metadata: Vec<MetadataInfo>,
        config: Option<crate::config::ModelConfig>,
        model: Option<crate::model::Checkpoint>,
    ) {
        // Local index/file health, computed from the freshly-parsed tensors (before
        // dedup, so a name in two shards is seen in both) — the loader already read
        // every header, so this re-reads nothing. Remote index health was folded in
        // by the caller; append the local reports, then derive the unindexed-file
        // set (files on disk but absent from the index) for the tree's dimming.
        let local: Vec<crate::health::HealthReport> = self
            .index_specs
            .iter()
            .map(|spec| crate::health::check_loaded(spec, &tensors))
            .filter(|r| r.has_issues())
            .collect();
        self.health_reports.extend(local);
        self.unindexed = Self::unindexed_files(&self.health_reports);

        // The derived reports (health / stats) are keyed to the tensors — drop any
        // cached from a prior load so they're recomputed against the new set.
        *self.cached_check.borrow_mut() = None;
        *self.checkpoint_stats_cache.borrow_mut() = None;
        *self.cached_group_files.borrow_mut() = None;

        // Install the session — the single owner of the canonical (deduped +
        // natural-sorted) tensors/metadata/config. A local read hands over the
        // serializable model; a remote read without an assembled model supplies
        // the parts directly. Dedup + natural-sort now live in the kernel.
        self.session = Some(match model {
            Some(cp) => crate::kernel::Session::from_model(cp),
            None => crate::kernel::Session::from_parts(tensors, metadata, config),
        });

        let schemas = crate::sample::parse_packing_schemas(self.tensors(), self.metadata());
        self.packing_schemas = schemas;
        self.build_tree();
        self.full_loaded = true;
    }

    /// Run the full structure load if it hasn't happened yet. The fast `--tensor`
    /// path reads a single tensor and leaves the rest unread; this brings in the
    /// whole tree the first time it's needed (e.g. on returning to the browser),
    /// showing the loading spinner only then.
    fn ensure_full_load(&mut self) -> Result<()> {
        if !self.full_loaded {
            self.load_all_files()?;
        }
        Ok(())
    }

    /// Try to read just `name` (plus its packing schema) without enumerating the
    /// whole checkpoint, so a direct `--tensor X` view appears without the cold
    /// full-load spinner. Only the single-HDF5-file case is worth special-casing
    /// — other formats read their whole structure in one cheap header pass, and a
    /// multi-file checkpoint may hold the tensor in any shard. Returns whether the
    /// fast read succeeded; on `false` the caller does a normal full load.
    fn try_load_single_tensor(&mut self, name: &str) -> bool {
        #[cfg(feature = "hdf5")]
        {
            let [path] = self.files.as_slice() else {
                return false;
            };
            if !matches!(
                path.extension().and_then(|s| s.to_str()),
                Some("h5") | Some("hdf5")
            ) {
                return false;
            }
            match crate::hdf5::read_one(path, name) {
                Ok(Some((tensor, metadata))) => {
                    // A single-tensor fast open with no full model yet: install a
                    // parts-only session so the data view reads it like any other
                    // (the full load, which replaces this, still runs on `Tab`).
                    self.session = Some(crate::kernel::Session::from_parts(
                        vec![tensor],
                        metadata,
                        None,
                    ));
                    let schemas =
                        crate::sample::parse_packing_schemas(self.tensors(), self.metadata());
                    self.packing_schemas = schemas;
                    true
                }
                // Not found or a read error — let the full load handle it (and
                // surface the "tensor not found" message).
                _ => false,
            }
        }
        #[cfg(not(feature = "hdf5"))]
        {
            let _ = name;
            false
        }
    }

    /// The fused-codebook packing schema for `name`, if the checkpoint declared one.
    fn schema_for(&self, name: &str) -> Option<&PackingSchema> {
        self.packing_schemas.get(name)
    }

    /// The view a tensor opens in with no explicit override: the codebook
    /// [`ViewDtype::Unpacked`] when it carries a packing schema, else `Stored`.
    fn default_view(&self, name: &str) -> ViewDtype {
        if self.packing_schemas.contains_key(name) {
            ViewDtype::Unpacked
        } else {
            ViewDtype::Stored
        }
    }

    /// The active view for a tensor: an explicit `d`/`--dtype` override if set,
    /// otherwise its [`default_view`].
    fn active_view(&self, name: &str) -> ViewDtype {
        self.data_view
            .dtype_overrides
            .borrow()
            .get(name)
            .copied()
            .unwrap_or_else(|| self.default_view(name))
    }

    /// The value range to bin the histogram over: the intrinsic codebook span
    /// `0..=2^max_width-1` for the unmerged view (so every index gets a bar, even
    /// absent ones — like the 4-bit views show all 16), otherwise the tensor's
    /// exact min/max once a stats scan has cached it.
    fn histogram_range(&self, tensor: &TensorInfo, view: ViewDtype) -> Option<(f64, f64)> {
        if view == ViewDtype::Unpacked
            && let Some(s) = self.schema_for(&tensor.name)
        {
            return Some((0.0, ((1u64 << s.max_width()) - 1) as f64));
        }
        self.cached_stats(tensor, view).map(|s| (s.min, s.max))
    }

    /// Read every file's top-level structure (tensors + metadata) into owned
    /// vectors. A free function (no `&self`) so it can run on a worker thread
    /// while the UI animates a loading spinner, and so the `diff` subcommand can
    /// load a checkpoint's structure headlessly.
    pub(crate) fn gather_checkpoint(
        files: &[PathBuf],
        remote: Option<&crate::remote::RemoteRead>,
    ) -> Result<(CheckpointParts, Option<crate::model::Checkpoint>)> {
        // `--ssh-read`: every source is read on the remote (an s3:// cstorch
        // checkpoint, or a remote safetensors directory/file), keeping the
        // credentials and data there. (The central model is filled by the remote
        // reader in a later step; the remote path returns the parts tuple only.)
        if let Some(r) = remote {
            let mut tensors: Vec<TensorInfo> = Vec::new();
            let mut metadata: Vec<MetadataInfo> = Vec::new();
            let mut config: Option<crate::config::ModelConfig> = None;
            let mut disk_shards: Vec<crate::stats::ShardDisk> = Vec::new();
            let mut remote_health: Vec<crate::health::HealthReport> = Vec::new();
            for file_path in files {
                let as_str = file_path.to_string_lossy();
                let (t, m, cfg, disk, health) = r.fetch_with_config(&as_str)?;
                tensors.extend(t);
                metadata.extend(m);
                config = config.or(cfg);
                if let Some(d) = disk {
                    disk_shards.extend(d.shards);
                }
                remote_health.extend(health);
            }
            let parts = (
                tensors,
                metadata,
                config,
                crate::stats::DiskUsage::from_shards(disk_shards),
                remote_health,
            );
            return Ok((parts, None));
        }
        // Local: a bare s3:// URI has no local credentials to read.
        for file_path in files {
            let as_str = file_path.to_string_lossy();
            if crate::s3::is_uri(&as_str) {
                anyhow::bail!(
                    "{as_str}: reading an s3:// checkpoint needs --ssh-read <[user@]host> \
                     (its credentials stay on the remote)"
                );
            }
        }
        // Read the whole local checkpoint into the central model in one pass (fs
        // walk + every header + config + index); the tuple is derived from it.
        let cp = crate::readers::read_local(files)?;
        let parts = (
            cp.tensors_vec(),
            cp.metadata_vec(),
            cp.config.clone(),
            None,
            Vec::new(),
        );
        Ok((parts, Some(cp)))
    }

    fn build_tree(&mut self) {
        let children = if self.metadata().is_empty() {
            TreeBuilder::build_tree(self.tensors())
        } else {
            TreeBuilder::build_tree_mixed(self.tensors(), self.metadata())
        };
        // Everything hangs off a single root node summarising the whole
        // checkpoint (tensor count, parameters and size), so the tree reads
        // top-down from one place instead of from a separate footer.
        let total_size = self.tensors().iter().map(|t| t.size_bytes).sum();
        let stored_size = self.tensors().iter().map(|t| t.on_disk_size()).sum();
        let root = TreeNode::Group {
            name: self.root_label(),
            children,
            expanded: true,
            tensor_count: self.tensors().len(),
            params: self.total_parameters(),
            total_size,
            stored_size,
        };
        self.tree_state.tree = vec![root];
        self.flatten_tree();
    }

    /// A concise name for the checkpoint root: the file name for a single file,
    /// otherwise the shared parent directory's name (or "checkpoint").
    fn root_label(&self) -> String {
        let basename = |p: &Path| {
            p.file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "checkpoint".to_string())
        };
        match self.files.split_first() {
            None => "checkpoint".to_string(),
            Some((first, [])) => basename(first),
            Some((first, _)) => {
                let dir = first.parent();
                if dir.is_some() && self.files.iter().all(|f| f.parent() == dir) {
                    dir.map(basename)
                        .unwrap_or_else(|| "checkpoint".to_string())
                } else {
                    "checkpoint".to_string()
                }
            }
        }
    }

    /// Re-flatten the tree (fold-aware) and, if searching, refresh the filtered
    /// results. The reflatten is a kernel [`TreeState`](crate::kernel::TreeState)
    /// op; the search-filter refresh stays here since it reads the tensor list.
    fn flatten_tree(&mut self) {
        self.tree_state.reflatten();
        self.update_filtered_tree();
    }

    /// Dispatch a [`Link`](crate::ui::Link): open a safetensors file's layout view,
    /// or reveal a concrete tensor in the tree. `None` when a `Tree` link names a
    /// tensor that isn't in this checkpoint (a stray click).
    fn open_link(&mut self, link: &crate::ui::Link) -> Option<Nav> {
        match link {
            crate::ui::Link::Layout(path) => Some(Nav::Open(Screen::Layout {
                path: path.clone(),
                selected: 0,
                scroll: 0,
            })),
            crate::ui::Link::Tree(name) => {
                if self.tensors().iter().any(|t| &t.name == name) {
                    self.tree_state.reveal(name);
                    Some(Nav::Open(Screen::Tree))
                } else {
                    None
                }
            }
        }
    }

    /// Hit-test the current frame's [`Self::links`] at `(col, row)` and, on a hit,
    /// follow that link. The shared click path for every screen's clickable names.
    fn link_click(&mut self, col: u16, row: u16) -> Option<Nav> {
        let link = self.link_at(col, row)?;
        self.open_link(&link)
    }

    /// The link (if any) whose region covers `(col, row)` in the current frame.
    fn link_at(&self, col: u16, row: u16) -> Option<crate::ui::Link> {
        self.links
            .borrow()
            .iter()
            .find(|(r, _)| row == r.y && col >= r.x && col < r.x + r.width)
            .map(|(_, l)| l.clone())
    }

    fn update_filtered_tree(&mut self) {
        // Only meaningful while searching; the filtered list lives inside the
        // `search` state, so plain browsing never spends a clone of the (possibly
        // huge) flattened tree keeping it in sync.
        let Some(query) = self.tree_state.search.as_ref().map(|s| s.query.clone()) else {
            return;
        };
        let filtered: Vec<(TreeNode, usize)> = if query.is_empty() {
            self.tree_state.flattened.clone()
        } else {
            let matcher = SkimMatcherV2::default();
            let mut scored_results: Vec<(TreeNode, i64)> = Vec::new();

            // Search through ALL tensors, not just the flattened tree
            for tensor in self.tensors() {
                if let Some(score) = matcher.fuzzy_match(&tensor.name, &query) {
                    scored_results.push((
                        TreeNode::Tensor {
                            info: tensor.clone(),
                            label: None,
                        },
                        score,
                    ));
                }
            }

            // Also search through metadata if present
            for metadata in self.metadata() {
                if let Some(score) = matcher.fuzzy_match(&metadata.name, &query) {
                    scored_results.push((
                        TreeNode::Metadata {
                            info: metadata.clone(),
                        },
                        score,
                    ));
                }
            }

            // Sort by score (highest first), flat list at depth 0.
            scored_results.sort_by_key(|b| std::cmp::Reverse(b.1));
            scored_results
                .into_iter()
                .map(|(node, _)| (node, 0))
                .collect()
        };
        if let Some(s) = self.tree_state.search.as_mut() {
            s.filtered = filtered;
        }
    }

    /// Headless render (`--plain`): produce the requested screen once as plain
    /// text and print it — no raw mode, no alternate screen, no interactivity.
    /// Each screen renders through the same Ratatui code as the live loop, into a
    /// fixed-size in-memory [`TestBackend`](ratatui::backend::TestBackend) flattened
    /// to text (see [`crate::tui::headless_render`]), so the output is deterministic
    /// regardless of the ambient terminal and matches the interactive screen. For
    /// piping, `grep`, and end-to-end tests.
    pub fn render_plain(&mut self, emit_command: bool) -> Result<()> {
        self.load_quiet()?;

        // `--histogram`/`--bins` and `--compute-stats` are scanned synchronously
        // below: their interactive scans read key events, which a headless render
        // can't. Capture the intent and the bin count, then strip the histogram
        // from the request so `open_requested` only applies the dtype/shape/slice
        // overrides (and doesn't kick off the interactive scan).
        let (want_hist, want_stats, bins) = match &self.open {
            Some(r) => (r.histogram.on(), r.compute_stats, r.histogram.bins()),
            None => (false, false, None),
        };
        let want_legend = self.open.as_ref().is_some_and(|r| r.legend);
        let want_health_findings = self.open.as_ref().is_some_and(|r| r.health.findings());
        let want_health =
            want_health_findings || self.open.as_ref().is_some_and(|r| r.health.wants());
        let want_stats_shards = self.open.as_ref().is_some_and(|r| r.stats.shards());
        let want_stats_popup =
            want_stats_shards || self.open.as_ref().is_some_and(|r| r.stats.wants());
        // The file-browser / layout-map / rename-editor screens (headless): rendered
        // directly below via their own frames, so `--plain` covers every screen (not
        // just tree / detail / data). Captured before the request is consumed.
        let want_files = self.open.as_ref().is_some_and(|r| r.files_view);
        let want_layout = self.open.as_ref().and_then(|r| r.layout_file.clone());
        let want_rename = self.open.as_ref().is_some_and(|r| r.rename);
        if let Some(n) = bins {
            self.data_view.histogram_bins.set(Some(n));
        }
        let screen = match self.open.take() {
            Some(mut req) => {
                req.histogram = HistogramReq::Off;
                req.exit_after = false;
                // A failed request (unknown tensor/metadata, bad `--shape`/`--slice`)
                // is fatal here — propagate it so the headless render exits non-zero
                // rather than silently falling back to the tree.
                self.open_requested(req, OpenMode::Headless)?
                    .unwrap_or(Screen::Tree)
            }
            None => Screen::Tree,
        };

        // `--emit-command`: print the CLI that `y` would copy to reopen this exact
        // screen, instead of rendering. Used by the round-trip test (render the
        // screen, take this command, re-render, assert the two match).
        if emit_command {
            let mut cmd = self.reopen_command(&screen, want_stats, want_hist);
            if want_health {
                cmd = format!(
                    "{cmd} {}",
                    if want_health_findings {
                        "--health-findings"
                    } else {
                        "--health"
                    }
                );
            }
            if want_stats_popup {
                cmd = format!(
                    "{cmd} {}",
                    if want_stats_shards {
                        "--stats-shards"
                    } else {
                        "--stats"
                    }
                );
            }
            println!("{cmd}");
            return Ok(());
        }

        // The file browser (`--files`): build the directory rows, then render its
        // frame headlessly — so `--plain` covers this screen like the others.
        if want_files && self.file_view_available() {
            if self.file_state.tree.is_none() {
                self.file_state.tree =
                    Some(self.build_browse_tree().map_err(|e| anyhow::anyhow!(e))?);
                self.file_state.rebuild_rows();
            }
            let text = crate::tui::headless_render(120, 40, |f| self.render_files_frame(f, false))?;
            println!("{text}");
            return Ok(());
        }
        // The safetensors byte-layout map (`--layout <file>`).
        if let Some(path) = want_layout {
            let mode = self.layout_mode(path, 0, 0);
            if let Err(e) = &mode.map {
                anyhow::bail!("{e}");
            }
            let text = crate::tui::headless_render(120, 40, |f| mode.render_frame(self, f))?;
            println!("{text}");
            return Ok(());
        }
        // The in-place rename editor (`--rename`), with any `--rename-rule` seeds
        // applied. Rendered with an empty preview (headless can't run the live
        // per-keystroke recompute), which is the editor's initial state anyway.
        if want_rename && self.rename_target().is_some() {
            let target = self.rename_target().expect("checked above");
            let loaded = crate::rename::load(&target).map_err(|e| anyhow::anyhow!("{e:#}"))?;
            let mut counts: HashMap<String, usize> = HashMap::new();
            for n in loaded.names() {
                *counts.entry(crate::rename::generalize(n).0).or_default() += 1;
            }
            let mut seen = HashSet::new();
            let schemas: Vec<(String, usize)> = loaded
                .names()
                .iter()
                .map(|n| crate::rename::generalize(n).0)
                .filter(|s| seen.insert(s.clone()))
                .map(|s| {
                    let c = counts[&s];
                    (s, c)
                })
                .collect();
            let root = loaded.root().display().to_string();
            let editor = RenameMode::default();
            let rules_view: Vec<crate::ui::RenameRuleView> = Vec::new();
            let text = crate::tui::headless_render(120, 40, |f| {
                draw_rename_frame(
                    f,
                    &root,
                    &editor,
                    &schemas,
                    &rules_view,
                    0,
                    &[],
                    false,
                    false,
                    &None,
                    None,
                    None,
                );
            })?;
            println!("{text}");
            return Ok(());
        }

        // The tree (and its `--legend` overlay) renders via the in-memory backend:
        // draw the tree frame, then composite the legend band on top when asked —
        // mirroring the interactive `l` path (`show_legend`).
        if matches!(screen, Screen::Tree) {
            // `--health`: composite the (structural) health popup over the tree.
            let health = want_health.then(|| {
                crate::check::run(
                    self.files
                        .iter()
                        .map(|p| p.display().to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                    self.tensors(),
                    self.metadata(),
                    &self.files,
                    &self.health_reports,
                    self.config(),
                    &crate::filter::NameFilter::default(),
                    false,
                    1,
                )
            });
            // `--stats`: render the full-screen stats view (not composited over the
            // tree — it's a first-class screen now). The on-disk footprint is a
            // live, machine-specific measurement (block size / ZFS), so it's left
            // out of this deterministic headless render — the interactive mode and
            // its `r` report show it.
            if want_stats_popup {
                let stats =
                    crate::stats::CheckpointStats::compute(self.tensors(), self.config(), None)
                        .with_s3(self.s3_stats());
                let text = crate::tui::headless_render(120, 40, |f| {
                    UI::render_stats_frame(f, &stats, 0, want_stats_shards);
                })?;
                println!("{text}");
                return Ok(());
            }
            let text = crate::tui::headless_render(120, 40, |f| {
                self.render_tree_frame(f, false); // headless: no scroll bar
                if want_legend {
                    UI::render_legend_band(f, Legend::Tree);
                }
                if let Some(report) = &health {
                    UI::render_check_report(
                        f,
                        report,
                        crate::ui::CheckPopup::Idle {
                            copied: None,
                            can_scan: false,
                        },
                        0,
                        want_health_findings,
                    );
                }
            })?;
            println!("{text}");
            return Ok(());
        }
        // The detail screen (and its legend band) is migrated to Ratatui too:
        // render it — overlay included — via the in-memory backend.
        if let Screen::Detail { tensor, .. } = &screen {
            println!(
                "{}",
                self.draw_detail_plain(tensor, want_stats, want_hist, want_legend)?
            );
            return Ok(());
        }
        // The data views (and their legend band) are migrated to Ratatui too:
        // render the heatmap / numeric grid — overlay included — via the in-memory
        // backend.
        if let Screen::Data {
            tensor,
            repr,
            slice,
        } = &screen
        {
            println!(
                "{}",
                self.draw_data_plain(tensor, *repr, *slice, want_legend)?
            );
            return Ok(());
        }

        // All three screens (tree, detail, data) render and return above.
        unreachable!("every screen renders via the in-memory backend above");
    }

    /// The CLI command that reopens `screen` — what the `y` shortcut copies.
    /// Scans the statistics / histogram first when the screen would (so the
    /// command emits `--histogram` once it's been computed, mirroring `y`).
    fn reopen_command(&self, screen: &Screen, want_stats: bool, want_hist: bool) -> String {
        match screen {
            Screen::Tree => self.command_for_tree_selection(),
            Screen::Files => self.command_for_files(),
            Screen::Layout { path, selected, .. } => {
                // Resolve the selected segment's tensor name (parse the header) so
                // the reopen command restores the selection.
                let select = crate::safelayout::parse(Path::new(path))
                    .ok()
                    .and_then(|m| Self::layout_selected_tensor(&m, *selected));
                self.command_for_layout(path, select.as_deref())
            }
            Screen::Detail { tensor, .. } => {
                let Some(t) = self.tensors().iter().find(|t| &t.name == tensor).cloned() else {
                    return String::new();
                };
                let view = self.active_view(&t.name);
                if want_hist {
                    self.compute_histogram_sync(&t, view);
                }
                if want_stats {
                    self.compute_stats_sync(&t, view);
                }
                self.command_for_detail(&t)
            }
            Screen::Data {
                tensor,
                repr,
                slice,
            } => {
                let Some(t) = self.tensors().iter().find(|t| &t.name == tensor).cloned() else {
                    return String::new();
                };
                self.command_for_data(&t, *repr, *slice)
            }
            Screen::Rename { pairs } => self.command_for_rename(pairs),
            Screen::Stats {
                shards_expanded, ..
            } => {
                let flag = if *shards_expanded {
                    "--stats-shards"
                } else {
                    "--stats"
                };
                format!("{} {flag}", self.command_for_tree())
            }
        }
    }

    /// The `--rename [--rename-rule 'SRC=>TGT']…` command that reopens the in-place
    /// rename editor with the current rule pairs pre-seeded (what `y` copies). Each
    /// complete pair round-trips as a schema `source => new-name`, so restoring is
    /// lossless (no regex reversal). Mirrors [`Self::command_for_files`].
    fn command_for_rename(&self, pairs: &[(String, String)]) -> String {
        let mut parts = self.command_prefix();
        parts.extend(self.checkpoint_path_parts());
        parts.push("--rename".to_string());
        for (src, tgt) in pairs {
            if src.is_empty() || tgt.is_empty() {
                continue;
            }
            parts.push("--rename-rule".to_string());
            parts.push(shell_quote(&format!("{src}=>{tgt}")));
        }
        parts.join(" ")
    }

    /// Render a tensor's detail screen to plain text for [`Self::render_plain`],
    /// via the in-memory Ratatui backend (mirrors the live detail draw).
    /// Statistics and the histogram, when requested, are scanned synchronously
    /// here rather than animated on a worker thread; an optional `--legend`
    /// overlay composites the (now-Ratatui) legend band on top.
    fn draw_detail_plain(
        &self,
        tensor_name: &str,
        want_stats: bool,
        want_hist: bool,
        want_legend: bool,
    ) -> Result<String> {
        let Some(tensor) = self
            .tensors()
            .iter()
            .find(|t| t.name == tensor_name)
            .cloned()
        else {
            return Ok(String::new());
        };
        let view = self.active_view(&tensor.name);
        let shape = self
            .data_view
            .shape_overrides
            .borrow()
            .get(&tensor.name)
            .cloned()
            .unwrap_or_else(|| tensor.shape.clone());
        // The histogram is computed first because (for floats / wide ints) it
        // computes and caches the stats it needs for its range — which then
        // surface on the statistics line, matching the interactive screen.
        let hist = if want_hist {
            self.compute_histogram_sync(&tensor, view)
        } else {
            None
        };
        let stats = if want_stats {
            self.compute_stats_sync(&tensor, view)
        } else {
            self.cached_stats(&tensor, view)
        };
        let stats_view = match &stats {
            Some(s) => StatsView::Ready(s),
            None => StatsView::Pending,
        };
        let overlay = want_legend.then_some(Overlay::Legend(Legend::Detail));
        self.detail_plain(
            &tensor,
            &shape,
            view,
            dtype_overridable(&tensor),
            self.unindexed.contains(&tensor.source_path),
            stats_view,
            hist.as_ref(),
            overlay.as_ref(),
        )
    }

    /// Render a tensor's numeric / heatmap data view to plain text for
    /// [`Self::render_plain`], via the in-memory Ratatui backend (mirrors the live
    /// data draw). The layout (overview / edges / window) and position come from
    /// the request's flags (applied by `open_requested`); statistics — which set
    /// the value range and heatmap scale — are scanned synchronously. An optional
    /// `--legend` overlay composites the (now-Ratatui) legend band on top.
    fn draw_data_plain(
        &self,
        tensor_name: &str,
        repr: Representation,
        slice: usize,
        want_legend: bool,
    ) -> Result<String> {
        let Some(tensor) = self
            .tensors()
            .iter()
            .find(|t| t.name == tensor_name)
            .cloned()
        else {
            return Ok(String::new());
        };
        let view = self.active_view(&tensor.name);
        let mode = match self.data_view.data_view_layout.get() {
            DataLayout::Edges => SampleMode::Edges {
                row_tail: self.data_view.data_view_row_tail.get(),
                col_tail: self.data_view.data_view_col_tail.get(),
            },
            DataLayout::Overview => SampleMode::Grid,
            DataLayout::Window => SampleMode::Window {
                row_off: self.data_view.data_view_win_row.get(),
                col_off: self.data_view.data_view_win_col.get(),
            },
        };
        let stats = self.compute_stats_sync(&tensor, view);
        let stats_view = match &stats {
            Some(s) => StatsView::Ready(s),
            None => StatsView::Pending,
        };
        let legend = match repr {
            Representation::Heatmap => Legend::Heatmap,
            Representation::Values => Legend::Values,
        };
        let overlay = want_legend.then_some(Overlay::Legend(legend));
        self.data_plain(
            &tensor,
            repr,
            slice,
            view,
            mode,
            stats_view,
            self.data_view.data_view_stripe.get(),
            self.data_view.data_view_base.get(),
            overlay.as_ref(),
        )
    }

    /// Compute (and cache) exact statistics for `(tensor, view)` synchronously,
    /// for the headless `--plain` render. `None` if the format can't be byte-read.
    fn compute_stats_sync(&self, tensor: &TensorInfo, view: ViewDtype) -> Option<Stats> {
        if let Some(s) = self.cached_stats(tensor, view) {
            return Some(s);
        }
        let schema = self.schema_for(&tensor.name).cloned();
        let (cancel, pause) = (AtomicBool::new(false), AtomicBool::new(false));
        let s = crate::sample::tensor_stats(tensor, view, schema.as_ref(), &cancel, &pause, None)
            .ok()?;
        self.stats_cache
            .borrow_mut()
            .insert((tensor.name.clone(), view), s);
        Some(s)
    }

    /// Compute (and cache) the value histogram for `(tensor, view)` synchronously,
    /// for `--plain`. Mirrors [`Self::scan_histogram`] without the animation /
    /// cancellation: floats and wide integers need stats for their bin range, so
    /// those are computed first only when required.
    fn compute_histogram_sync(&self, tensor: &TensorInfo, view: ViewDtype) -> Option<Histogram> {
        let count = self.data_view.histogram_bins.get();
        let key = (tensor.name.clone(), view, count);
        if let Some(h) = self.histogram_cache.borrow().get(&key) {
            return Some(h.clone());
        }
        let range = self.histogram_range(tensor, view);
        if crate::sample::histogram_bins(view, &tensor.dtype, range, count).is_none() {
            self.compute_stats_sync(tensor, view); // populate the range, then retry
        }
        let range = self.histogram_range(tensor, view);
        let (bins, n) = crate::sample::histogram_bins(view, &tensor.dtype, range, count)?;
        let shared = HistShared::new(n);
        let (cancel, pause) = (AtomicBool::new(false), AtomicBool::new(false));
        let schema = self.schema_for(&tensor.name).cloned();
        let started = std::time::Instant::now();
        crate::sample::tensor_histogram_into(
            tensor,
            view,
            schema.as_ref(),
            bins,
            n,
            &shared,
            &cancel,
            &pause,
            None,
        )
        .ok()?;
        let mut hist = shared.snapshot(bins);
        hist.elapsed = started.elapsed();
        self.histogram_cache.borrow_mut().insert(key, hist.clone());
        Some(hist)
    }

    pub fn run(&mut self) -> Result<()> {
        if self.files.is_empty() {
            return Ok(());
        }

        // A remote (`--ssh-read`) structure read runs an interactive `ssh` that may
        // prompt for a password/2FA. Do it BEFORE taking over the screen, so the
        // prompt uses the normal terminal; `fetch` announces + shows a spinner
        // after the prompt. `load_all_files` then no-ops.
        if self.remote_read().is_some() {
            self.load_quiet()?;
        }

        // Set up the Ratatui terminal (raw mode, cleared screen, hidden cursor,
        // no alternate screen) and own it for the session.
        self.terminal = Some(crate::tui::init()?);

        let result = self.interactive_loop();

        // Restore the terminal: leave the last frame on screen and drop the shell
        // prompt onto a fresh line just below it (see `tui::restore`). Keeps what
        // you were looking at visible after quitting (and lets `--exit` output be
        // read / captured).
        if let Some(mut terminal) = self.terminal.take() {
            crate::tui::restore(&mut terminal)?;
        }

        result
    }

    fn interactive_loop(&mut self) -> Result<()> {
        // Browser-style screen history: Backspace steps back through the screens
        // you've visited, `\` steps forward, and any fresh navigation truncates
        // the forward tail. The tree is the root.
        let mut history = vec![Screen::Tree];
        let mut cursor = 0usize;

        // `--health` / `--stats` open straight into their popup once the tree is up.
        let want_health_findings = self.open.as_ref().is_some_and(|r| r.health.findings());
        let want_health =
            want_health_findings || self.open.as_ref().is_some_and(|r| r.health.wants());
        let want_stats_shards = self.open.as_ref().is_some_and(|r| r.stats.shards());
        let want_stats_popup =
            want_stats_shards || self.open.as_ref().is_some_and(|r| r.stats.wants());
        // `--files` lands in the file browser once the tree is up (like a seeded
        // `--tensor` screen, but pushed after the popups so it wins the landing).
        let want_files = self.open.as_ref().is_some_and(|r| r.files_view);
        // `--layout PATH` lands in that file's layout map, optionally preselecting
        // a tensor (`--layout-select NAME`).
        let want_layout = self.open.as_ref().and_then(|r| r.layout_file.clone());
        let want_layout_select = self.open.as_ref().and_then(|r| r.layout_select.clone());
        // `--rename` lands in the in-place rename editor, seeded with any
        // `--rename-rule 'SRC=>TGT'` pairs. Local safetensors only.
        let want_rename = self.open.as_ref().is_some_and(|r| r.rename);
        let want_rename_rules = self
            .open
            .as_ref()
            .map(|r| r.rename_rules.clone())
            .unwrap_or_default();

        // A `--tensor` request seeds the history with that screen — or, with
        // `--exit`, renders it once and quits without entering the navigator.
        if let Some(req) = self.open.take() {
            // Fast path: a single tensor's detail/data view doesn't need the
            // whole tree, so read just that tensor and defer the (potentially
            // slow) full structure load until the browser is actually shown.
            let fast = req.target.metadata().is_none()
                && !matches!(req.view, OpenView::Tree)
                && req
                    .target
                    .tensor()
                    .is_some_and(|name| self.try_load_single_tensor(name));
            if !fast {
                self.load_all_files()?;
            }

            let mode = if req.exit_after {
                OpenMode::OneShot
            } else {
                OpenMode::Interactive
            };
            match self.open_requested(req, mode) {
                Ok(seeded) => {
                    // `--exit` renders inside `open_requested`; never enter the navigator.
                    if mode == OpenMode::OneShot {
                        return Ok(());
                    }
                    if let Some(screen) = seeded {
                        history.push(screen);
                        cursor = 1;
                    }
                }
                // A one-shot failure is fatal (non-zero exit). Interactively, the
                // recoverable message was already shown and a key awaited inside
                // `open_requested`; fall through to the tree.
                Err(e) if mode == OpenMode::OneShot => return Err(e),
                Err(_) => {}
            }
        } else {
            self.load_all_files()?;
        }

        // `--health`: float the report popup over the tree before handing off to
        // the navigator (dismissing it drops into the normal tree).
        if want_health {
            self.ensure_full_load()?;
            let mut term = self.terminal.take().expect("interactive loop owns it");
            self.show_check_report(&mut term, want_health_findings);
            self.terminal = Some(term);
        }
        // `--stats` / `--stats-shards`: open straight into the full-screen stats
        // mode (Esc / `⌫` drops back to the tree).
        if want_stats_popup {
            self.ensure_full_load()?;
            history.push(Screen::Stats {
                shards_expanded: want_stats_shards,
                scroll: 0,
            });
            cursor = history.len() - 1;
        }
        // `--files`: open the file browser on top of the tree, so `Tab`/Backspace
        // drop back to it. Pushed last so it wins even alongside `--tensor`. The
        // file views are local-only, so skip them for a remote checkpoint.
        if want_files && self.file_view_available() {
            history.push(Screen::Files);
            cursor = history.len() - 1;
        }
        // `--layout PATH`: open that file's layout map on top of the tree. A
        // relative PATH resolves against the checkpoint directory, so the `--layout
        // <relative>` command that `y` copies round-trips.
        if let Some(path) = want_layout.filter(|_| self.file_view_available()) {
            let p = Path::new(&path);
            let abs = if p.is_absolute() {
                path
            } else if let Some(RemoteBrowse::Sftp(dir)) = self.remote_browse() {
                // A remote-relative `--layout` resolves against the SFTP browse dir.
                format!("{}/{path}", dir.trim_end_matches('/'))
            } else {
                self.browse_root.join(p).to_string_lossy().into_owned()
            };
            // Resolve `--layout-select NAME` to its segment index (parse the header,
            // locally or over SFTP per the source).
            let selected = want_layout_select
                .and_then(|name| {
                    self.parse_layout(&abs).ok().and_then(|m| {
                        m.segments.iter().position(|s| {
                            matches!(s.kind, crate::safelayout::SegmentKind::Tensor { .. })
                                && s.name == name
                        })
                    })
                })
                .unwrap_or(0);
            history.push(Screen::Layout {
                path: abs,
                selected,
                scroll: 0,
            });
            cursor = history.len() - 1;
        }
        // `--rename` (+ `--rename-rule 'SRC=>TGT'`): open the in-place rename editor
        // seeded with those schema pairs. Skip silently for a checkpoint that can't
        // be renamed in place (remote / non-safetensors / read-only) — the `R` gate.
        if want_rename && self.can_rename() {
            let pairs: Vec<(String, String)> = want_rename_rules
                .iter()
                .filter_map(|rule| {
                    rule.split_once("=>")
                        .map(|(src, tgt)| (src.trim().to_string(), tgt.trim().to_string()))
                })
                .collect();
            history.push(Screen::Rename { pairs });
            cursor = history.len() - 1;
        }

        loop {
            // The tensor the screen we're about to run belongs to (if any), so
            // that on returning to the tree we can land back on it.
            let screen_tensor = match &history[cursor] {
                Screen::Detail { tensor, .. } | Screen::Data { tensor, .. } => Some(tensor.clone()),
                Screen::Tree
                | Screen::Files
                | Screen::Layout { .. }
                | Screen::Rename { .. }
                | Screen::Stats { .. } => None,
            };

            let nav = match history[cursor].clone() {
                Screen::Tree => self.run_mode(&mut TreeMode::new())?,
                Screen::Files => self.run_mode(&mut FilesMode::new())?,
                Screen::Rename { pairs } => {
                    // Persist the typed rules so a round-trip (e.g. clicking a shard
                    // to view its layout) returns to the same editor state.
                    let mut mode = RenameMode2::new(pairs);
                    let nav = self.run_mode(&mut mode)?;
                    history[cursor] = mode.residual();
                    nav
                }
                Screen::Layout {
                    path,
                    selected,
                    scroll,
                } => {
                    // Record where the user left the layout map so back/forward
                    // returns to the same segment (like the data view's slice/repr).
                    let mut mode = self.layout_mode(path, selected, scroll);
                    let nav = self.run_mode(&mut mode)?;
                    history[cursor] = mode.residual();
                    nav
                }
                Screen::Detail { tensor, slice } => self.run_mode(&mut DetailMode::new(
                    tensor,
                    slice,
                    StatsStart::OnDemand,
                    Interaction::Interactive,
                ))?,
                Screen::Data {
                    tensor,
                    repr,
                    slice,
                } => {
                    // Re-record the screen with where the user left it (slice /
                    // representation), so back/forward returns there faithfully.
                    let mut mode = DataMode::new(tensor, repr, slice, Interaction::Interactive);
                    let nav = self.run_mode(&mut mode)?;
                    history[cursor] = mode.residual();
                    nav
                }
                Screen::Stats {
                    shards_expanded,
                    scroll,
                } => {
                    // Record the fold state / scroll where the user left them, so
                    // back/forward (and the `--stats` reopen command) restore it.
                    let mut mode = StatsMode::new(shards_expanded, scroll);
                    let nav = self.run_mode(&mut mode)?;
                    history[cursor] = mode.residual();
                    nav
                }
            };
            match nav {
                Nav::Quit => break,
                Nav::Open(screen) => {
                    history.truncate(cursor + 1);
                    history.push(screen);
                    cursor += 1;
                }
                Nav::Back => cursor = cursor.saturating_sub(1),
                Nav::Forward => {
                    if cursor + 1 < history.len() {
                        cursor += 1;
                    }
                }
            }

            // Returning to the tree from a tensor's detail/data view: select that
            // tensor (revealing it) so you land where you were. Revealing needs
            // the full tree, so finish the deferred load before locating it.
            if matches!(history[cursor], Screen::Tree) {
                self.ensure_full_load()?;
                if let Some(name) = screen_tensor {
                    self.tree_state.reveal(&name);
                }
            }
        }

        Ok(())
    }

    /// Build the tree's [`DrawConfig`] and render it into a Ratatui frame — the
    /// interactive and headless tree both go through this. `interactive` is true
    /// only for the live TUI (it gates the scroll bar; a headless `--plain` /
    /// screen-copy render passes false so its static dump shows no bar).
    fn render_tree_frame(&self, frame: &mut ratatui::Frame, interactive: bool) {
        let title = if self.files.len() == 1 {
            self.files[0].to_string_lossy().to_string()
        } else {
            "Multiple files".to_string()
        };
        let tree_to_display = self.tree_state.visible();
        let (status_icon, status_bar, status_secondary) = self.status_bar();
        let badges = self.screen_badges(HelpCtx::Tree);
        let config = DrawConfig {
            tree: tree_to_display,
            current_file: &title,
            file_idx: 0,
            total_files: 1,
            selected_idx: self.tree_state.selected,
            scroll_offset: self.tree_state.scroll,
            search_mode: self.tree_state.search_mode(),
            search_query: self.tree_state.search_query(),
            search_cursor: self.tree_state.search_cursor(),
            status_icon,
            status_bar: &status_bar,
            status_secondary: &status_secondary,
            can_repack: self.repack_input().is_some(),
            can_rename: self.can_rename(),
            unindexed: &self.unindexed,
            packing_schemas: &self.packing_schemas,
            copied_flash: self.copied_flash.as_ref().map(|(what, _)| what.as_str()),
            interactive,
            badges: &badges,
            hovered_badge: self.hovered_badge.get(),
        };
        *self.clickable.borrow_mut() = UI::render_tree(frame, &config);
        self.links.borrow_mut().clear(); // tree rows navigate on their own
        // The engine draws the scroll bar; report its geometry (live TUI only).
        *self.vscrollbar.borrow_mut() = interactive
            .then(|| {
                let area = frame.area();
                UI::tree_scrollbar(
                    area.width,
                    area.height,
                    self.tree_state.search_mode(),
                    self.repack_input().is_some(),
                    self.can_rename(),
                    tree_to_display.len(),
                    self.tree_state.scroll,
                )
            })
            .flatten();
        self.render_shortcut_hover(frame);
    }

    /// Render the tree to plain text via an in-memory Ratatui backend — the
    /// headless (`--plain`) tree and the `c` screen-copy share this.
    fn tree_plain(&self) -> Result<String> {
        crate::tui::headless_render(120, 40, |f| self.render_tree_frame(f, false))
    }

    /// Load the checkpoint and print the grouped tree to stdout, then return —
    /// `--print-tree`. Text is the fully-expanded browser tree; JSON is the
    /// `model.safetensors.index.json` shape (see [`TreeFormat`]).
    pub fn print_tree(
        &mut self,
        format: TreeFormat,
        detail: TreeDetail,
        filter: &crate::filter::NameFilter,
    ) -> Result<()> {
        self.load_quiet()?;
        self.apply_name_filter(filter);
        let out = match format {
            TreeFormat::Text => self.tree_text(detail),
            TreeFormat::Json => self.tree_json(detail),
        };
        emit_stdout(&out)
    }

    /// Load the checkpoint and print a flat list of every tensor to stdout, then
    /// return — `--print-tensors`.
    pub fn print_tensors(
        &mut self,
        format: TreeFormat,
        detail: TreeDetail,
        filter: &crate::filter::NameFilter,
    ) -> Result<()> {
        self.load_quiet()?;
        self.apply_name_filter(filter);
        let out = match format {
            TreeFormat::Text => self.tensors_text(detail),
            TreeFormat::Json => self.tensors_json(detail),
        };
        emit_stdout(&out)
    }

    /// Load the checkpoint and print the tensor-tree screen's serializable
    /// [`ViewModel`](crate::kernel::ViewModel) as JSON, then return — `--print-view`.
    /// This is the kernel's frontend-agnostic output contract exercised headlessly:
    /// exactly what a web server or MCP tool would serve for the tree screen,
    /// projected from the same live `tree_state` the interactive TUI renders.
    pub fn print_view(&mut self, filter: &crate::filter::NameFilter) -> Result<()> {
        self.load_quiet()?;
        self.apply_name_filter(filter);
        let vm = crate::kernel::ViewModel::from_tree(&self.root_label(), &self.tree_state);
        emit_stdout(&serde_json::to_string_pretty(&vm)?)
    }

    /// Drop the tensors and metadata whose names don't pass `filter`, then rebuild
    /// the tree — scoping a `--print-tree` / `--print-tensors` export to a subset.
    /// A no-op when the filter is inactive.
    fn apply_name_filter(&mut self, filter: &crate::filter::NameFilter) {
        if !filter.is_active() {
            return;
        }
        if let Some(session) = self.session.as_mut() {
            session.retain_named(|name| filter.matches(name));
        }
        self.build_tree();
    }

    /// The whole tree as text — every group and tensor in the browser's row
    /// layout, fully expanded regardless of the live collapse state, with no
    /// viewport limit or header/footer chrome. Backs the `t` copy and
    /// `--print-tree`. `Full` appends each tensor's source file.
    fn tree_text(&self, detail: TreeDetail) -> String {
        fn walk(
            node: &TreeNode,
            depth: usize,
            detail: TreeDetail,
            unindexed: &HashSet<String>,
            schemas: &HashMap<String, PackingSchema>,
            out: &mut Vec<String>,
        ) {
            let mut line = crate::ui::tree_row_text(node, depth, unindexed, schemas);
            if detail == TreeDetail::Full
                && let TreeNode::Tensor { info, .. } = node
            {
                line.push_str(&format!("  ← {}", file_basename(&info.source_path)));
            }
            out.push(line);
            if let TreeNode::Group { children, .. } = node {
                for child in children {
                    walk(child, depth + 1, detail, unindexed, schemas, out);
                }
            }
        }
        // Render a fully-expanded copy so every group's arrow (▾) matches the
        // listing below it, regardless of the live collapse state.
        let mut tree = self.tree_state.tree.clone();
        TreeBuilder::set_all_expanded(&mut tree, true);
        let mut out = Vec::new();
        for node in &tree {
            walk(
                node,
                0,
                detail,
                &self.unindexed,
                &self.packing_schemas,
                &mut out,
            );
        }
        out.join("\n")
    }

    /// A flat, one-line-per-tensor listing (in natural-sorted order), reusing the
    /// browser's tensor-row field layout but without its leading `·` bullet — a
    /// flat list needs no tree marker. `Full` appends each tensor's source file.
    fn tensors_text(&self, detail: TreeDetail) -> String {
        self.tensors()
            .iter()
            .map(|t| {
                let line = crate::ui::tensor_list_line(t, &self.unindexed, &self.packing_schemas);
                let mut text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
                if detail == TreeDetail::Full {
                    text.push_str(&format!("  ← {}", file_basename(&t.source_path)));
                }
                text
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// The tree as `model.safetensors.index.json`-style JSON: `metadata.total_size`
    /// (summed logical bytes) and a `weight_map` of tensor name → shard file.
    /// `Full` adds a `tensors` block with each tensor's dtype / shape / counts.
    fn tree_json(&self, detail: TreeDetail) -> String {
        let total_size: usize = self.tensors().iter().map(|t| t.size_bytes).sum();
        let weight_map: serde_json::Map<String, serde_json::Value> = self
            .tensors()
            .iter()
            .map(|t| (t.name.clone(), file_basename(&t.source_path).into()))
            .collect();
        let mut root = serde_json::Map::new();
        root.insert(
            "metadata".into(),
            serde_json::json!({ "total_size": total_size }),
        );
        root.insert("weight_map".into(), serde_json::Value::Object(weight_map));
        if detail == TreeDetail::Full {
            let tensors: serde_json::Map<String, serde_json::Value> = self
                .tensors()
                .iter()
                .map(|t| (t.name.clone(), tensor_facts(t)))
                .collect();
            root.insert("tensors".into(), serde_json::Value::Object(tensors));
        }
        serde_json::to_string_pretty(&serde_json::Value::Object(root)).unwrap_or_default()
    }

    /// A JSON list of tensors: bare names (`Compact`) or objects with name,
    /// dtype, shape, element count and source file (`Full`). Natural-sorted.
    fn tensors_json(&self, detail: TreeDetail) -> String {
        let items: Vec<serde_json::Value> = match detail {
            TreeDetail::Compact => self
                .tensors()
                .iter()
                .map(|t| t.name.clone().into())
                .collect(),
            TreeDetail::Full => self
                .tensors()
                .iter()
                .map(|t| {
                    let mut o = tensor_facts(t);
                    if let serde_json::Value::Object(m) = &mut o {
                        m.insert("name".into(), t.name.clone().into());
                        m.insert("file".into(), file_basename(&t.source_path).into());
                    }
                    o
                })
                .collect(),
        };
        serde_json::to_string_pretty(&serde_json::Value::Array(items)).unwrap_or_default()
    }

    /// The `t` shortcut: open a modal menu to pick which export variant to copy
    /// (tree / tensor list × text / JSON × plain / verbose — every CLI
    /// `--print-*` combination), then copy that. `↑`/`↓` (or `1`–`8`) move,
    /// Enter copies, Esc / click cancels. `term` is the borrowed live terminal.
    fn copy_menu(&mut self, term: &mut crate::tui::LiveTerminal) {
        let labels: Vec<&str> = EXPORT_CHOICES.iter().map(|c| c.label).collect();
        let last = EXPORT_CHOICES.len() - 1;
        let mut sel = 0usize;
        // The preview is regenerated only when the highlight moves (it renders
        // the real export, which is cheap but not free on a huge checkpoint).
        let mut previewed = usize::MAX;
        let mut preview: Vec<Line<'static>> = Vec::new();
        let mut item_rects: Vec<ratatui::layout::Rect> = Vec::new();
        // A wrong-keyboard-layout key flashes the shared hint rather than being
        // silently ignored; cleared on the next input.
        let mut layout_hint: Option<char> = None;
        loop {
            if sel != previewed {
                preview = self.export_preview(EXPORT_CHOICES[sel]);
                previewed = sel;
            }
            let hint = layout_hint;
            if term
                .draw(|f| {
                    self.render_tree_frame(f, true);
                    item_rects = UI::render_menu_box(f, "Copy as…", &labels, sel, &preview);
                    if let Some(c) = hint {
                        UI::render_notice(f, &layout_hint_msg(c));
                    }
                })
                .is_err()
            {
                return;
            }
            // Which menu row (if any) is under a mouse position.
            let hit = |col: u16, row: u16| -> Option<usize> {
                item_rects.iter().position(|r| {
                    row >= r.y && row < r.y + r.height && col >= r.x && col < r.x + r.width
                })
            };
            match event::read() {
                Ok(Event::Key(key)) => {
                    if is_ctrl_c(&key) {
                        quit_immediately();
                    }
                    if let Some(c) = wrong_layout_char(&key) {
                        layout_hint = Some(c);
                        continue;
                    }
                    layout_hint = None;
                    match key.code {
                        KeyCode::Up => sel = if sel == 0 { last } else { sel - 1 },
                        KeyCode::Down => sel = if sel == last { 0 } else { sel + 1 },
                        KeyCode::Home => sel = 0,
                        KeyCode::End => sel = last,
                        // 1–8 pick a row directly.
                        KeyCode::Char(d @ '1'..='9') => {
                            let i = d as usize - '1' as usize;
                            if i <= last {
                                self.copy_export(term, EXPORT_CHOICES[i]);
                                return;
                            }
                        }
                        KeyCode::Enter => {
                            self.copy_export(term, EXPORT_CHOICES[sel]);
                            return;
                        }
                        KeyCode::Esc | KeyCode::Char('q') => return,
                        _ => {}
                    }
                }
                Ok(Event::Mouse(m)) => match m.kind {
                    MouseEventKind::ScrollUp => sel = if sel == 0 { last } else { sel - 1 },
                    MouseEventKind::ScrollDown => sel = if sel == last { 0 } else { sel + 1 },
                    // Hover highlights the row under the cursor.
                    MouseEventKind::Moved | MouseEventKind::Drag(_) => {
                        if let Some(i) = hit(m.column, m.row) {
                            sel = i;
                        }
                    }
                    // Click a row to copy it; a click off the list cancels.
                    MouseEventKind::Down(_) => match hit(m.column, m.row) {
                        Some(i) => {
                            self.copy_export(term, EXPORT_CHOICES[i]);
                            return;
                        }
                        None => return,
                    },
                    _ => {}
                },
                Ok(_) => {}       // resize etc.: redraw
                Err(_) => return, // input closed
            }
        }
    }

    /// The head of a menu `choice`'s export, styled like the tree, for the
    /// picker's live preview — real output from this checkpoint. Always returns a
    /// fixed number of rows (blank-padded, then a "+N more" / blank summary) so
    /// the menu box is the same size for every option.
    fn export_preview(&self, choice: ExportChoice) -> Vec<Line<'static>> {
        let (mut lines, total) = match (choice.shape, choice.format) {
            (ExportShape::Tree, TreeFormat::Text) => self.tree_preview_lines(choice.detail),
            (ExportShape::Tensors, TreeFormat::Text) => self.tensors_preview_lines(choice.detail),
            // JSON: syntax-highlight it with the same palette as the metadata
            // view (falling back to plain lines if it somehow doesn't parse).
            (_, TreeFormat::Json) => {
                let full = self.export_text(choice);
                let styled = crate::ui::highlight_json_lines(&full).unwrap_or_else(|| {
                    full.lines()
                        .map(|l| Line::from(Span::raw(l.to_string())))
                        .collect()
                });
                let total = styled.len();
                (styled.into_iter().take(MENU_PREVIEW_LINES).collect(), total)
            }
        };
        lines.resize_with(MENU_PREVIEW_LINES, Line::default);
        lines.push(if total > MENU_PREVIEW_LINES {
            Line::from(crate::ui::dim_span(format!(
                "… (+{} more lines)",
                total - MENU_PREVIEW_LINES
            )))
        } else {
            Line::default()
        });
        lines
    }

    /// Styled preview rows for the tree export (first [`MENU_PREVIEW_LINES`]), plus
    /// the total row count. Walks fully expanded (forcing the open ▾ on collapsed
    /// groups) without cloning the tree.
    fn tree_preview_lines(&self, detail: TreeDetail) -> (Vec<Line<'static>>, usize) {
        fn walk(
            node: &TreeNode,
            depth: usize,
            detail: TreeDetail,
            unindexed: &HashSet<String>,
            schemas: &HashMap<String, PackingSchema>,
            out: &mut Vec<Line<'static>>,
            total: &mut usize,
        ) {
            *total += 1;
            if out.len() < MENU_PREVIEW_LINES {
                let mut line = crate::ui::tree_row_line(node, depth, unindexed, schemas);
                if let TreeNode::Group {
                    expanded: false, ..
                } = node
                {
                    for span in &mut line.spans {
                        if span.content == "▸" {
                            span.content = "▾".into();
                            break;
                        }
                    }
                }
                if detail == TreeDetail::Full
                    && let TreeNode::Tensor { info, .. } = node
                {
                    line.spans.push(crate::ui::dim_span(format!(
                        "  ← {}",
                        file_basename(&info.source_path)
                    )));
                }
                out.push(line);
            }
            if let TreeNode::Group { children, .. } = node {
                for child in children {
                    walk(child, depth + 1, detail, unindexed, schemas, out, total);
                }
            }
        }
        let mut out = Vec::new();
        let mut total = 0;
        for node in &self.tree_state.tree {
            walk(
                node,
                0,
                detail,
                &self.unindexed,
                &self.packing_schemas,
                &mut out,
                &mut total,
            );
        }
        (out, total)
    }

    /// Styled preview rows for the flat tensor list (first [`MENU_PREVIEW_LINES`]),
    /// plus the total tensor count.
    fn tensors_preview_lines(&self, detail: TreeDetail) -> (Vec<Line<'static>>, usize) {
        let lines = self
            .tensors()
            .iter()
            .take(MENU_PREVIEW_LINES)
            .map(|t| {
                let mut line =
                    crate::ui::tensor_list_line(t, &self.unindexed, &self.packing_schemas);
                if detail == TreeDetail::Full {
                    line.spans.push(crate::ui::dim_span(format!(
                        "  ← {}",
                        file_basename(&t.source_path)
                    )));
                }
                line
            })
            .collect();
        (lines, self.tensors().len())
    }

    /// Copy the export text for `choice`. If it fits the terminal clipboard, copy
    /// it directly (with a confirmation flash); otherwise copy the exact CLI
    /// command that reproduces it and show that in a dismissible band.
    fn copy_export(&mut self, term: &mut crate::tui::LiveTerminal, choice: ExportChoice) {
        let text = self.export_text(choice);
        if copy_to_clipboard(&text) {
            self.flash_copied(choice.label);
        } else {
            let command = self.export_command(choice);
            copy_to_clipboard(&command); // the command itself is small — always fits
            self.float_until_dismissed(term, |f| {
                self.render_tree_frame(f, true);
                UI::render_export_band(f, &command);
            });
        }
    }

    /// The exported text for a menu `choice`.
    fn export_text(&self, choice: ExportChoice) -> String {
        match (choice.shape, choice.format) {
            (ExportShape::Tree, TreeFormat::Text) => self.tree_text(choice.detail),
            (ExportShape::Tree, TreeFormat::Json) => self.tree_json(choice.detail),
            (ExportShape::Tensors, TreeFormat::Text) => self.tensors_text(choice.detail),
            (ExportShape::Tensors, TreeFormat::Json) => self.tensors_json(choice.detail),
        }
    }

    /// The concrete CLI command reproducing a menu `choice`, built the way `y`
    /// builds its reopen command (real paths, scp-style / `--ssh-read` for a
    /// remote source), so it runs as-is.
    fn export_command(&self, choice: ExportChoice) -> String {
        let mut parts = self.command_prefix();
        parts.extend(self.checkpoint_path_parts());
        parts.push(
            match choice.shape {
                ExportShape::Tree => "--print-tree",
                ExportShape::Tensors => "--print-tensors",
            }
            .to_string(),
        );
        if choice.format == TreeFormat::Json {
            parts.push("--format".to_string());
            parts.push("json".to_string());
        }
        if choice.detail == TreeDetail::Full {
            parts.push("-v".to_string());
        }
        parts.join(" ")
    }

    /// Build the detail-screen draw config and render it into `frame` — the
    /// Ratatui counterpart of [`Self::render_tree_frame`], shared by the live loop
    /// and the headless render.
    #[allow(clippy::too_many_arguments)] // mirrors the screen renderer's params
    fn render_detail_frame(
        &self,
        frame: &mut ratatui::Frame,
        tensor: &TensorInfo,
        shape: &[usize],
        view: ViewDtype,
        overridable: bool,
        unindexed: bool,
        stats: StatsView,
        histogram: Option<&Histogram>,
        hist_scanning: Option<crate::ui::ScanProgress>,
        overlay: Option<&Overlay>,
    ) {
        let (chips, links) = UI::render_detail(
            frame,
            tensor,
            shape,
            view,
            overridable,
            unindexed,
            stats,
            histogram,
            hist_scanning,
            self.schema_for(&tensor.name),
            overlay,
        );
        *self.clickable.borrow_mut() = chips;
        *self.links.borrow_mut() = links; // `File:` path → layout map
        let badges = self.screen_badges(HelpCtx::Detail);
        UI::render_badge_bar(frame, &badges, self.hovered_badge.get());
        self.render_shortcut_hover(frame);
    }

    /// Render the full-screen checkpoint-stats view (chrome + scrollable body +
    /// footer via [`UI::render_stats_frame`]), record its clickable regions, and
    /// draw the access badge on the reserved bottom row — the stats analogue of
    /// [`Self::render_detail_frame`]. Returns the maximum valid scroll offset so
    /// the mode can clamp its own.
    fn render_stats_screen(
        &self,
        frame: &mut ratatui::Frame,
        stats: &crate::stats::CheckpointStats,
        scroll: usize,
        shards_expanded: bool,
    ) -> usize {
        let (max, regions, vscroll) = UI::render_stats_frame(frame, stats, scroll, shards_expanded);
        *self.clickable.borrow_mut() = regions;
        *self.vscrollbar.borrow_mut() = vscroll;
        let badges = self.screen_badges(HelpCtx::Stats);
        UI::render_badge_bar(frame, &badges, self.hovered_badge.get());
        max
    }

    /// Render a tensor's detail screen to plain text via an in-memory Ratatui
    /// backend — the headless (`--plain`) detail and the `c` screen-copy share
    /// this. Mirrors [`Self::tree_plain`].
    #[allow(clippy::too_many_arguments)] // mirrors the screen renderer's params
    fn detail_plain(
        &self,
        tensor: &TensorInfo,
        shape: &[usize],
        view: ViewDtype,
        overridable: bool,
        unindexed: bool,
        stats: StatsView,
        histogram: Option<&Histogram>,
        overlay: Option<&Overlay>,
    ) -> Result<String> {
        crate::tui::headless_render(120, 40, |f| {
            self.render_detail_frame(
                f,
                tensor,
                shape,
                view,
                overridable,
                unindexed,
                stats,
                histogram,
                None,
                overlay,
            )
        })
    }

    /// Sample and render a tensor's data view (heatmap / numeric grid) into
    /// `frame`, compositing a pop-up `overlay` (legend / copied command) last —
    /// the Ratatui counterpart of [`Self::render_detail_frame`], shared by the
    /// live loop and the headless render. Returns `(slices, overridable,
    /// clamped_slice)` (or the error message [`Self::draw_data_view`] would
    /// have), so the loop can clamp the slice and gate slice/dtype hints.
    #[allow(clippy::too_many_arguments)] // mirrors the data-view sampler's params
    /// The data view's current sampling mode, from the session-remembered layout
    /// prefs (overview / edges split / window offset).
    fn data_sample_mode(&self) -> SampleMode {
        match self.data_view.data_view_layout.get() {
            DataLayout::Edges => SampleMode::Edges {
                row_tail: self.data_view.data_view_row_tail.get(),
                col_tail: self.data_view.data_view_col_tail.get(),
            },
            DataLayout::Overview => SampleMode::Grid,
            DataLayout::Window => SampleMode::Window {
                row_off: self.data_view.data_view_win_row.get(),
                col_off: self.data_view.data_view_win_col.get(),
            },
        }
    }

    #[allow(clippy::too_many_arguments)] // mirrors the data-view sampler's params
    fn render_data_frame(
        &self,
        frame: &mut ratatui::Frame,
        tensor: &TensorInfo,
        repr: Representation,
        slice: usize,
        view: ViewDtype,
        mode: SampleMode,
        stats: StatsView,
        stripe: StripeMode,
        base: NumBase,
        overlay: Option<&Overlay>,
    ) -> Result<(usize, bool, usize), String> {
        // Size the grid to the frame's render area — the live terminal size, or
        // the headless `TestBackend`'s fixed size, depending on the caller.
        let area = frame.area();
        let info = self.prepare_data_sample(
            tensor,
            repr,
            slice,
            view,
            mode,
            stats,
            area.width,
            area.height,
        )?;
        let cache = self.sample_cache.borrow();
        let sample = &cache.as_ref().unwrap().sample;
        *self.clickable.borrow_mut() = match repr {
            Representation::Heatmap => UI::render_heatmap(frame, tensor, sample, stats),
            Representation::Values => UI::render_values(frame, tensor, sample, stats, stripe, base),
        };
        match overlay {
            Some(Overlay::Legend(l)) => UI::render_legend_band(frame, *l),
            Some(Overlay::Command(c)) => UI::render_command_band(frame, c),
            Some(Overlay::Notice(m)) => UI::render_notice_box(frame, m),
            None => {}
        }
        let badges = self.screen_badges(HelpCtx::Data);
        UI::render_badge_bar(frame, &badges, self.hovered_badge.get());
        self.links.borrow_mut().clear(); // data view shows no linkable names
        self.render_shortcut_hover(frame);
        Ok(info)
    }

    /// Render the data view from the *already sampled* result in
    /// [`Self::sample_cache`] (no re-sampling), with no overlay — used as the
    /// static background behind the reshape / slice prompts, which float over the
    /// view that was just drawn. A no-op if the cache is somehow empty.
    fn render_cached_data(
        &self,
        frame: &mut ratatui::Frame,
        tensor: &TensorInfo,
        repr: Representation,
        stats: StatsView,
        stripe: StripeMode,
        base: NumBase,
    ) {
        let cache = self.sample_cache.borrow();
        let Some(cached) = cache.as_ref() else {
            return;
        };
        let sample = &cached.sample;
        // Drawn only as a static background behind a prompt (which runs its own
        // input loop), so the clickable map isn't updated here.
        let _regions = match repr {
            Representation::Heatmap => UI::render_heatmap(frame, tensor, sample, stats),
            Representation::Values => UI::render_values(frame, tensor, sample, stats, stripe, base),
        };
    }

    /// Render a tensor's data view to plain text via an in-memory Ratatui backend
    /// — the headless (`--plain`) data view and the `c` screen-copy share this.
    /// Mirrors [`Self::detail_plain`]. On a sampling error the message is rendered
    /// in place (matching the live "Data preview unavailable" path).
    #[allow(clippy::too_many_arguments)] // mirrors the data-view sampler's params
    fn data_plain(
        &self,
        tensor: &TensorInfo,
        repr: Representation,
        slice: usize,
        view: ViewDtype,
        mode: SampleMode,
        stats: StatsView,
        stripe: StripeMode,
        base: NumBase,
        overlay: Option<&Overlay>,
    ) -> Result<String> {
        crate::tui::headless_render(120, 40, |f| {
            if let Err(msg) = self.render_data_frame(
                f, tensor, repr, slice, view, mode, stats, stripe, base, overlay,
            ) {
                use ratatui::widgets::{Paragraph, Widget};
                Paragraph::new(format!("Data preview unavailable: {msg}"))
                    .render(f.area(), f.buffer_mut());
            }
        })
    }

    /// Recompute the scroll offset so the selected row stays visible, given the
    /// live terminal size (so it matches [`UI::render_tree`]'s body height).
    /// Whether the loaded checkpoint can be repacked (gates the `R` hint and is
    /// part of the tree's header height).
    fn can_repack(&self) -> bool {
        self.repack_input().is_some()
    }

    /// Number of rows in the tree currently shown (the search results when
    /// searching, else the full flattened tree).
    fn current_tree_len(&self) -> usize {
        if self.tree_state.search_mode() {
            self.tree_state.visible().len()
        } else {
            self.tree_state.flattened.len()
        }
    }

    fn update_tree_scroll(&mut self, width: u16, height: u16) {
        let body = UI::tree_visible_rows(
            width,
            height,
            self.tree_state.search_mode(),
            self.can_repack(),
            self.can_rename(),
        );
        let sel = self.tree_state.selected;
        self.tree_state.scroll = if sel >= self.tree_state.scroll + body {
            sel.saturating_sub(body - 1)
        } else if sel < self.tree_state.scroll {
            sel
        } else {
            self.tree_state.scroll
        };
    }

    /// Copy the current tree screen's text to the clipboard (the `c` shortcut).
    fn copy_tree_screen(&mut self) {
        if let Ok(text) = self.tree_plain() {
            copy_to_clipboard(&text);
        }
        self.flash_copied("screen contents");
    }

    /// Note a copy confirmation to flash on the bottom line for `COPY_FLASH`.
    fn flash_copied(&mut self, what: &str) {
        self.copied_flash = Some((
            format!("✓ Copied {what} to the clipboard"),
            std::time::Instant::now(),
        ));
    }

    /// Run `f` with the live terminal temporarily taken out of `self` and handed
    /// to it, then put back — the take/restore dance the pop-up commands share.
    fn with_terminal<R>(
        &mut self,
        f: impl FnOnce(&mut Self, &mut crate::tui::LiveTerminal) -> R,
    ) -> R {
        let mut term = self
            .terminal
            .take()
            .expect("interactive loop owns the terminal");
        let out = f(self, &mut term);
        self.terminal = Some(term);
        out
    }

    /// The generic interactive driver: run one [`Mode`] until it returns a [`Nav`].
    /// Owns the live terminal (taken into a local for the loop, like the old
    /// detail/data screens) and all the shared plumbing — the input-drain gate, the
    /// draw (frame + overlay + hover), the copied-flash lifecycle, footer-chip / link
    /// / badge clicks, hover, Ctrl-C, the wrong-layout hint, and the command palette
    /// — so a mode only supplies its content (`render_frame` / `handle_key` /
    /// `handle_mouse` / `open_palette`). Replaces the six hand-rolled `run_*` loops.
    /// Copy the command that reopens the current view (the `y` action) and float a
    /// borderless [`render_command_band`](UI::render_command_band) with it — the same
    /// for every mode (the command is `reopen_command(mode.residual())`), so the
    /// command is on the clipboard AND selectable by hand when OSC-52 can't reach the
    /// terminal. Engine-owned so no mode can diverge.
    fn do_copy_command(&self, mode: &dyn Mode, term: &mut crate::tui::LiveTerminal) {
        let cmd = self.reopen_command(&mode.residual(), false, false);
        copy_to_clipboard(&cmd);
        self.float_until_dismissed(term, |f| {
            mode.render_frame(self, f);
            UI::render_command_band(f, &cmd);
        });
    }

    fn run_mode(&mut self, mode: &mut dyn Mode) -> Result<Nav> {
        let spec = mode.spec();
        let mut term = self
            .terminal
            .take()
            .expect("interactive loop owns the terminal");
        match mode.on_enter(self, &mut term) {
            Ok(Outcome::Leave(nav)) => {
                self.terminal = Some(term);
                return Ok(nav);
            }
            Ok(Outcome::Stay) => {}
            Err(e) => {
                self.terminal = Some(term);
                return Err(e);
            }
        }
        let _ = term.clear();
        let mut layout_hint: Option<char> = None;

        let nav = loop {
            // Coalesce a burst of queued input before painting (held arrows stay
            // snappy), then advance any background scan.
            let input_pending = event::poll(std::time::Duration::ZERO).unwrap_or(false);
            let bg = mode.tick_background(self);

            if !input_pending {
                mode.pre_draw(self, &mut term);
                let hint = layout_hint;
                // Start each frame with no scroll bar; the mode's render records one
                // (via `self.vscrollbar`) when its body overflows, and the engine
                // draws it here — so a mode can't scroll without showing a bar.
                *self.vscrollbar.borrow_mut() = None;
                let drawn = term.draw(|f| {
                    mode.render_frame(self, f);
                    if let Some(sb) = *self.vscrollbar.borrow() {
                        UI::render_vscrollbar(f, &sb);
                    }
                    if let Some(o) = mode.overlay() {
                        match o {
                            Overlay::Legend(l) => UI::render_legend_band(f, *l),
                            Overlay::Command(c) => UI::render_command_band(f, c),
                            Overlay::Notice(m) => UI::render_notice_box(f, m),
                        }
                    }
                    if let Some(c) = hint {
                        UI::render_notice(f, &layout_hint_msg(c));
                    }
                    self.render_shortcut_hover(f);
                });
                if drawn.is_err() {
                    break Nav::Quit;
                }
            }
            layout_hint = None;

            // Wait for an event: the copied-flash expires on its own after
            // COPY_FLASH, and a live scan (Bg::Poll) ticks every SCAN_TICK so its
            // spinner animates — pausing the scan while input is pending so a
            // keypress's own file read isn't stuck behind the scan's block.
            let ev = if let Some((_, at)) = &self.copied_flash {
                let remaining = COPY_FLASH.saturating_sub(at.elapsed());
                if remaining.is_zero() || !event::poll(remaining).unwrap_or(false) {
                    self.copied_flash = None;
                    continue;
                }
                event::read()?
            } else if matches!(bg, Bg::Poll) {
                if event::poll(SCAN_TICK).unwrap_or(false) {
                    mode.set_background_paused(true);
                    event::read()?
                } else {
                    mode.set_background_paused(false);
                    continue; // tick → redraw (advance the spinner / harvest)
                }
            } else {
                event::read()?
            };

            // A floating overlay (detail/data legend/command/notice) swallows the
            // next input: any key or click closes it; Ctrl-C still quits.
            if mode.overlay().is_some() {
                match &ev {
                    Event::Key(k) if is_ctrl_c(k) => {
                        if spec.ctrlc_quits_immediately {
                            quit_immediately();
                        } else {
                            break Nav::Quit;
                        }
                    }
                    Event::Key(k) => {
                        if let Some(c) = wrong_layout_char(k) {
                            layout_hint = Some(c);
                        } else {
                            mode.dismiss_overlay();
                        }
                    }
                    Event::Mouse(m) if matches!(m.kind, MouseEventKind::Moved) => {
                        if let Ok(sz) = term.size() {
                            self.update_hovers(spec.id, sz.width, sz.height, m.column, m.row);
                        }
                    }
                    Event::Mouse(m) if matches!(m.kind, MouseEventKind::Down(_)) => {
                        mode.dismiss_overlay();
                    }
                    _ => {}
                }
                continue;
            }

            self.hovered_shortcut.set(None);

            // Resolve the event to a key: a real keypress, a footer-chip / badge
            // click (which replays a key), a link click (leaves), or a mode-specific
            // mouse action.
            let key = match ev {
                Event::Key(k) => k,
                Event::Mouse(m) => match self.route_mouse(&mut term, mode, &spec, m)? {
                    MouseOutcome::Leave(nav) => break nav,
                    MouseOutcome::SynthKey(k) => k,
                    MouseOutcome::Redraw | MouseOutcome::Ignored => continue,
                },
                _ => continue,
            };

            if is_ctrl_c(&key) {
                if spec.ctrlc_quits_immediately {
                    quit_immediately();
                } else {
                    break Nav::Quit;
                }
            }
            // A fresh key clears the copied-flash confirmation.
            self.copied_flash = None;

            // Copy the command that reopens this view (`y`, or `^Y` where letters are
            // field input). Owned by the engine so every mode behaves identically —
            // copy to the clipboard AND float a borderless band with the command so it
            // can still be selected by hand when OSC-52 can't reach the terminal.
            let copy_cmd = KeyEvent::new(
                KeyCode::Char('y'),
                if mode.accepts_text(self) {
                    KeyModifiers::CONTROL
                } else {
                    KeyModifiers::NONE
                },
            );
            if key == copy_cmd {
                self.do_copy_command(mode, &mut term);
                continue;
            }

            // Space / `:` opens the command palette (unless the mode takes it as
            // input — the tree while searching).
            if matches!(key.code, KeyCode::Char(' ') | KeyCode::Char(':'))
                && mode.palette_on_space(self)
            {
                match mode.open_palette(self, &mut term) {
                    PaletteResult::Nav(n) => break n,
                    PaletteResult::CopyCommand => self.do_copy_command(mode, &mut term),
                    PaletteResult::SynthKey(k) => {
                        if let Outcome::Leave(n) = mode.handle_key(self, &mut term, k)? {
                            break n;
                        }
                    }
                    PaletteResult::Handled => {}
                }
                continue;
            }

            // A non-Latin key (wrong keyboard layout) matches no shortcut — flash a
            // hint rather than silently ignoring it (unless the mode takes text).
            if !mode.accepts_text(self)
                && let Some(c) = wrong_layout_char(&key)
            {
                layout_hint = Some(c);
                continue;
            }

            if let Outcome::Leave(nav) = mode.handle_key(self, &mut term, key)? {
                break nav;
            }
        };

        self.terminal = Some(term);
        Ok(nav)
    }

    /// Render one [`Mode`] frame and return — the one-shot (`--exit`) path. Runs
    /// `on_enter` (which handles a `--compute-stats` synchronous scan) then draws a
    /// single frame, leaving it on screen. No event loop.
    fn run_mode_once(&mut self, mode: &mut dyn Mode) -> Result<()> {
        let mut term = self
            .terminal
            .take()
            .expect("interactive loop owns the terminal");
        let leave = matches!(mode.on_enter(self, &mut term), Ok(Outcome::Leave(_)));
        if !leave {
            let _ = term.draw(|f| mode.render_frame(self, f));
        }
        self.terminal = Some(term);
        Ok(())
    }

    /// The shared mouse routing every mode gets for free: hover on move; on a left
    /// click, follow an underlined link, else act a status-badge click as its key,
    /// else a footer chip as its key; anything else (rows / scrollbar / band / wheel)
    /// goes to the mode's own [`Mode::handle_mouse`].
    fn route_mouse(
        &mut self,
        term: &mut crate::tui::LiveTerminal,
        mode: &mut dyn Mode,
        spec: &ModeSpec,
        m: MouseEvent,
    ) -> Result<MouseOutcome> {
        let (col, row) = (m.column, m.row);
        match m.kind {
            MouseEventKind::Moved => {
                if let Ok(sz) = term.size() {
                    self.update_hovers(spec.id, sz.width, sz.height, col, row);
                }
                Ok(MouseOutcome::Redraw)
            }
            MouseEventKind::Down(MouseButton::Left) => {
                self.copied_flash = None;
                // An underlined link (a filename → layout, a tensor → tree).
                if let Some(nav) = self.link_click(col, row) {
                    return Ok(MouseOutcome::Leave(nav));
                }
                // A status badge (e.g. the health badge → `h`), unless letters are
                // field input here.
                if !mode.accepts_text(self)
                    && let Ok(sz) = term.size()
                    && let Some(k) = self.badge_action_at(spec.id, sz.width, sz.height, col, row)
                {
                    return Ok(MouseOutcome::SynthKey(k));
                }
                // A footer chip / `[×]`.
                if let Some(k) = crate::ui::region_hit(&self.clickable.borrow(), col, row) {
                    return Ok(MouseOutcome::SynthKey(k));
                }
                // The vertical scroll bar (any mode): click the track to scrub to
                // that offset and start a drag. `set_scroll` moves the mode's own
                // offset, so this works uniformly for every scrolling mode.
                let bar = *self.vscrollbar.borrow();
                if let Some(bar) = bar
                    && bar.hit(col, row)
                {
                    mode.set_scroll(self, bar.offset_at(row));
                    self.scrollbar_drag = true;
                    return Ok(MouseOutcome::Redraw);
                }
                Ok(mode.handle_mouse(self, term, m))
            }
            MouseEventKind::Drag(MouseButton::Left) if self.scrollbar_drag => {
                let bar = *self.vscrollbar.borrow();
                if let Some(bar) = bar {
                    mode.set_scroll(self, bar.offset_at(row));
                }
                Ok(MouseOutcome::Redraw)
            }
            MouseEventKind::Up(MouseButton::Left) => {
                self.scrollbar_drag = false;
                Ok(mode.handle_mouse(self, term, m))
            }
            MouseEventKind::ScrollDown | MouseEventKind::ScrollUp => {
                self.copied_flash = None;
                Ok(mode.handle_mouse(self, term, m))
            }
            _ => Ok(mode.handle_mouse(self, term, m)), // other drags / releases
        }
    }

    /// The key a click on a status badge synthesizes (the health badge → `h`), or
    /// `None` if the click missed the bar or the badge has no action.
    fn badge_action_at(&self, id: HelpCtx, w: u16, h: u16, col: u16, row: u16) -> Option<KeyEvent> {
        let badges = self.screen_badges(id);
        UI::badge_bar_hit(w, h, col, row, &badges)
            .and_then(|i| badges.get(i).and_then(|b| b.action()))
            .map(|k| KeyEvent::new(KeyCode::Char(k), KeyModifiers::NONE))
    }

    /// Whether the file browser / layout map are usable: a local checkpoint (walks
    /// the filesystem), or a remote source we know how to browse — a remote
    /// safetensors dir (SFTP) or an `s3://…` object list. Metadata-only either way.
    fn file_view_available(&self) -> bool {
        self.remote_read().is_none() || self.remote_browse().is_some()
    }

    /// Which access badge the bottom-right status line shows: `Editable` when the
    /// open checkpoint is one the in-place rename can actually rewrite (a local
    /// safetensors checkpoint whose files are writable — see [`Self::can_rename`]),
    /// else `ReadOnly` (including a read-only mount / file).
    fn access_badge(&self) -> crate::ui::AccessBadge {
        if self.can_rename() {
            crate::ui::AccessBadge::Editable
        } else {
            crate::ui::AccessBadge::ReadOnly
        }
    }

    /// Perform a tree command, from its key or the palette. Returns `Some(Nav)` for
    /// a command that leaves the tree (only `Quit` so far), else `None`. Pop-up
    /// commands borrow the terminal via [`Self::with_terminal`].
    fn run_command(&mut self, cmd: Cmd, term: &mut crate::tui::LiveTerminal) -> Option<Nav> {
        match cmd {
            Cmd::Search => self.enter_search_mode(),
            Cmd::ExpandAll => self.tree_state.set_all_expanded(true),
            Cmd::CollapseAll => self.tree_state.set_all_expanded(false),
            Cmd::ViewFiles => {
                if self.file_view_available() {
                    return Some(Nav::Open(Screen::Files));
                }
                // A remote source we can't browse (no SFTP dir / s3 listing) — say
                // so rather than opening an empty view.
                self.float_until_dismissed(term, |f| {
                    self.render_tree_frame(f, true);
                    UI::render_notice(f, "The file browser isn't available for this source.");
                });
            }
            Cmd::CopyScreen => self.copy_tree_screen(),
            Cmd::CopyPath => self.copy_selected_path(),
            Cmd::CopyName => self.copy_selected_name(),
            Cmd::Stats => {
                return Some(Nav::Open(Screen::Stats {
                    shards_expanded: false,
                    scroll: 0,
                }));
            }
            Cmd::Health => self.show_check_report(term, false),
            Cmd::Legend => self.show_legend(term, Legend::Tree, None),
            Cmd::CopyTree => self.copy_menu(term),
            // Copy-command is engine-owned (`do_copy_command` / the `y` key), reached
            // via `PaletteResult::CopyCommand`, so it never comes through here.
            Cmd::CopyCommand => {}
            Cmd::Repack => self.repack_checkpoint(term),
            Cmd::Rename => {
                if self.can_rename() {
                    return Some(Nav::Open(Screen::Rename { pairs: Vec::new() }));
                }
                // Can't rename — say why rather than opening an empty editor (the
                // palette already hides it; a bare `r` can still reach here). A local
                // safetensors checkpoint that's merely read-only gets a distinct hint
                // from a non-safetensors / remote source.
                let msg = if self.rename_target().is_some() {
                    "This checkpoint's files are read-only (a read-only filesystem or \
                     read-only files) — in-place rename can't rewrite them."
                } else {
                    "In-place rename is available for a local safetensors checkpoint only."
                };
                self.float_until_dismissed(term, |f| {
                    self.render_tree_frame(f, true);
                    UI::render_notice(f, msg);
                });
            }
            Cmd::Quit => return Some(Nav::Quit),
        }
        None
    }

    /// The commands available on the tree right now, in palette order — the static
    /// registry minus any whose precondition fails (e.g. Repack needs an HDF5
    /// source).
    fn available_commands(&self) -> Vec<CmdEntry> {
        TREE_COMMANDS
            .iter()
            .copied()
            .filter(|(cmd, _, _, _)| match cmd {
                Cmd::Repack => self.repack_input().is_some(),
                Cmd::Rename => self.can_rename(),
                Cmd::ViewFiles => self.file_view_available(),
                _ => true,
            })
            .collect()
    }

    /// Float the command palette over the tree (Space or `:`): a fuzzy-filtered
    /// picker of the available commands. Returns the chosen command — the caller
    /// runs it after the terminal is handed back, so a pop-up command can reclaim
    /// it — or `None` if dismissed.
    fn command_palette(&mut self, term: &mut crate::tui::LiveTerminal) -> Option<Cmd> {
        let entries = self.available_commands();
        self.run_palette(term, entries, HelpCtx::Tree, |s, f| {
            s.render_tree_frame(f, true)
        })
    }

    /// The file browser's command palette (Space or `:`) — the file-view analogue
    /// of [`Self::command_palette`], over [`FILE_COMMANDS`].
    fn file_command_palette(&mut self, term: &mut crate::tui::LiveTerminal) -> Option<FileCmd> {
        let entries = self.available_file_commands();
        self.run_palette(term, entries, HelpCtx::Files, |s, f| {
            s.render_files_frame(f, true)
        })
    }

    /// The layout map's command palette (Space or `:`), drawn over the strip.
    fn layout_command_palette(
        &self,
        term: &mut crate::tui::LiveTerminal,
        map: &crate::safelayout::LayoutMap,
        selected: usize,
        scroll: usize,
    ) -> Option<LayoutCmd> {
        let entries: Vec<LayoutCmdEntry> = LAYOUT_COMMANDS.to_vec();
        self.run_palette(term, entries, HelpCtx::Layout, move |_s, f| {
            UI::render_layout(f, map, selected, scroll, None, true);
        })
    }

    /// The shared fuzzy command-palette picker, generic over the command type so
    /// every view reuses one loop. `ctx` looks up each command's one-line help;
    /// `backdrop` draws the live frame behind the palette (passed `&self` so it can
    /// call the view's `render_*`). Returns the chosen command, or `None`.
    fn run_palette<T: Copy>(
        &self,
        term: &mut crate::tui::LiveTerminal,
        all: Vec<PaletteRow<T>>,
        ctx: HelpCtx,
        backdrop: impl Fn(&Self, &mut ratatui::Frame),
    ) -> Option<T> {
        // The `\t` sentinel (Tab) has no `Char` help entry — look it up by its real
        // key code, so `Tab`'s hint shows in the palette.
        let help = move |key: char| {
            let code = if key == '\t' {
                KeyCode::Tab
            } else {
                KeyCode::Char(key)
            };
            crate::ui::shortcut_help(KeyEvent::new(code, KeyModifiers::NONE), ctx).unwrap_or("")
        };
        let matcher = SkimMatcherV2::default();
        let mut query = String::new();
        let mut sel = 0usize;
        let mut row_rects: Vec<ratatui::layout::Rect> = Vec::new();
        loop {
            // Filter + rank by fuzzy score over "Group Title help".
            let filtered: Vec<PaletteRow<T>> = if query.is_empty() {
                all.clone()
            } else {
                let mut scored: Vec<(PaletteRow<T>, i64)> = all
                    .iter()
                    .filter_map(|&(cmd, group, title, key)| {
                        let hay = format!("{group} {title} {}", help(key));
                        matcher
                            .fuzzy_match(&hay, &query)
                            .map(|s| ((cmd, group, title, key), s))
                    })
                    .collect();
                // Highest fuzzy score first.
                scored.sort_by_key(|&(_, score)| std::cmp::Reverse(score));
                scored.into_iter().map(|(c, _)| c).collect()
            };
            if sel >= filtered.len() {
                sel = filtered.len().saturating_sub(1);
            }
            let rows: Vec<(String, String, String, String)> = filtered
                .iter()
                .map(|&(_, group, title, key)| {
                    (
                        key_label(key),
                        group.to_string(),
                        title.to_string(),
                        help(key).to_string(),
                    )
                })
                .collect();
            if term
                .draw(|f| {
                    backdrop(self, f);
                    row_rects = UI::render_command_palette(f, &query, &rows, sel);
                })
                .is_err()
            {
                return None;
            }
            let hit = |col: u16, row: u16| -> Option<usize> {
                row_rects
                    .iter()
                    .position(|r| row == r.y && col >= r.x && col < r.x + r.width)
            };
            match event::read() {
                Ok(Event::Key(key)) => {
                    if is_ctrl_c(&key) {
                        quit_immediately();
                    }
                    match key.code {
                        KeyCode::Esc => return None,
                        KeyCode::Up => sel = sel.saturating_sub(1),
                        KeyCode::Down if sel + 1 < filtered.len() => sel += 1,
                        KeyCode::Enter => return filtered.get(sel).map(|&(c, _, _, _)| c),
                        KeyCode::Backspace => {
                            query.pop();
                            sel = 0;
                        }
                        KeyCode::Char(c) => {
                            query.push(c);
                            sel = 0;
                        }
                        _ => {}
                    }
                }
                Ok(Event::Mouse(m)) => match m.kind {
                    MouseEventKind::ScrollUp => sel = sel.saturating_sub(1),
                    MouseEventKind::ScrollDown if sel + 1 < filtered.len() => sel += 1,
                    MouseEventKind::Moved => {
                        if let Some(i) = hit(m.column, m.row) {
                            sel = i;
                        }
                    }
                    MouseEventKind::Down(_) => {
                        // A click off any row dismisses (returns None).
                        let i = hit(m.column, m.row)?;
                        return filtered.get(i).map(|&(c, _, _, _)| c);
                    }
                    _ => {}
                },
                Ok(_) => {}
                Err(_) => return None,
            }
        }
    }

    /// Recompute the cached flattened file rows from the tree (after a build or a
    /// directory fold), clamping the selection into the new row count — the
    /// file-view analogue of the tensor tree's flatten-on-change.
    /// The header path shown atop the file browser: the local browse root, the
    /// remote SFTP directory, or the `s3://…` URI.
    fn browse_display_root(&self) -> String {
        match self.remote_browse() {
            Some(RemoteBrowse::Sftp(dir)) => format!("{}:{dir}", self.remote_host_label()),
            Some(RemoteBrowse::S3(uri)) => uri.clone(),
            None => self.browse_root.to_string_lossy().into_owned(),
        }
    }

    /// The remote host label for display (`host` from `--ssh-read`), or `remote`.
    fn remote_host_label(&self) -> String {
        self.remote_read()
            .map(|r| r.host.clone())
            .unwrap_or_else(|| "remote".to_string())
    }

    /// Build the file browser's directory tree for the current source: a local
    /// filesystem walk, a recursive SFTP listing, or an s3-native object tree —
    /// one [`crate::filetree::FileNode`] either way. Returns a readable error
    /// string (shown as a popup) when a remote listing fails.
    fn build_browse_tree(&self) -> std::result::Result<crate::filetree::FileNode, String> {
        match self.remote_browse() {
            // Local: assemble the browser from the cached directory walk in the
            // model (no `readdir`/`stat` on `Tab`). Falls back to a fresh walk only
            // if the model wasn't populated.
            None => match self.checkpoint() {
                Some(cp) => {
                    let root = &self.browse_root;
                    let mut listing: HashMap<PathBuf, Vec<crate::filetree::DirEntry>> =
                        HashMap::new();
                    for fe in &cp.files {
                        let full = root.join(&fe.rel_path);
                        let parent = full
                            .parent()
                            .map(Path::to_path_buf)
                            .unwrap_or_else(|| root.clone());
                        let entry = if fe.is_dir() {
                            crate::filetree::DirEntry::Directory {
                                name: fe.name.clone(),
                            }
                        } else {
                            crate::filetree::DirEntry::File {
                                name: fe.name.clone(),
                                size: fe.apparent(),
                            }
                        };
                        listing.entry(parent).or_default().push(entry);
                    }
                    let list = move |p: &Path| listing.get(p).cloned().unwrap_or_default();
                    Ok(crate::filetree::build_from(&list, root, 8))
                }
                None => Ok(crate::filetree::build(&self.browse_root, 8)),
            },
            Some(RemoteBrowse::Sftp(dir)) => {
                // Fetch the whole tree in one SFTP channel + one batch `stat -L`
                // (up front), then assemble it from the in-memory listing — so the
                // first `Tab` isn't a per-directory channel-open storm. A failed
                // root listing surfaces the error; deeper failures show empty.
                let listing = self
                    .with_remote_session(|s| s.walk_dir(dir, 8))
                    .map_err(|e| format!("{e:#}"))?;
                // `walk_dir` already yields the shared `filetree::DirEntry`, so the
                // listing feeds `build_from` directly — no per-entry conversion.
                let list = |p: &Path| -> Vec<crate::filetree::DirEntry> {
                    listing
                        .get(p.to_string_lossy().as_ref())
                        .cloned()
                        .unwrap_or_default()
                };
                Ok(crate::filetree::build_from(&list, Path::new(dir), 8))
            }
            Some(RemoteBrowse::S3(uri)) => {
                let r = self
                    .remote_read()
                    .ok_or_else(|| "no --ssh-read session configured".to_string())?;
                let objects = self
                    .with_remote_session(|s| r.list_s3(s, uri))
                    .map_err(|e| format!("{e:#}"))?;
                Ok(crate::filetree::build_from_keys(
                    &s3_root_label(uri),
                    &objects,
                ))
            }
        }
    }

    /// Draw the file browser: delegates to [`UI::render_files`] with the current
    /// (cached) rows and records the frame's clickable regions (footer chips +
    /// `[×]`).
    fn render_files_frame(&self, frame: &mut ratatui::Frame, interactive: bool) {
        let root = self.browse_display_root();
        let flash = self.copied_flash.as_ref().map(|(what, _)| what.as_str());
        let badges = self.screen_badges(HelpCtx::Files);
        let regions = UI::render_files(
            frame,
            &root,
            &self.file_state.rows,
            self.file_state.selected,
            self.file_state.scroll,
            flash,
            interactive,
            &badges,
            self.hovered_badge.get(),
        );
        *self.clickable.borrow_mut() = regions;
        self.links.borrow_mut().clear(); // file rows activate on their own
        // The engine draws the scroll bar; report its geometry (live TUI only).
        *self.vscrollbar.borrow_mut() = interactive
            .then(|| {
                let area = frame.area();
                UI::files_scrollbar(
                    area.width,
                    area.height,
                    self.file_state.rows.len(),
                    self.file_state.scroll,
                )
            })
            .flatten();
    }

    /// One screenful of file rows, to size a PageUp/PageDown jump.
    fn file_page_rows(&self) -> usize {
        let (w, h) = terminal::size().unwrap_or((80, 40));
        UI::visible_file_rows(w, h)
    }

    /// Keep the selected file row in view (snap the scroll offset), mirroring
    /// [`Self::update_tree_scroll`].
    fn update_files_scroll(&mut self, width: u16, height: u16) {
        let body = UI::files_visible_rows(width, height);
        let sel = self.file_state.selected;
        self.file_state.scroll = if sel >= self.file_state.scroll + body {
            sel.saturating_sub(body - 1)
        } else if sel < self.file_state.scroll {
            sel
        } else {
            self.file_state.scroll
        };
    }

    /// Whether `path` is one of the checkpoint files currently loaded (so opening
    /// it means switching back to the tensor tree, not loading a new checkpoint).
    fn is_loaded_checkpoint(&self, path: &Path) -> bool {
        // Remote paths can't be canonicalized locally — compare the strings.
        if self.remote_read().is_some() {
            let p = path.to_string_lossy();
            return self.files.iter().any(|f| f.to_string_lossy() == p);
        }
        let target = std::fs::canonicalize(path).ok();
        self.files
            .iter()
            .any(|f| f == path || (target.is_some() && std::fs::canonicalize(f).ok() == target))
    }

    /// Act on the highlighted file row (`Enter` / double-click): toggle a
    /// directory, open a checkpoint's file view (the layout map for safetensors,
    /// else the tensor tree / an info pop-up), or preview a text / JSON sidecar.
    /// Returns `Some(Nav)` only when it leaves the file view.
    fn activate_file_selection(&mut self) -> Option<Nav> {
        use crate::filetree::FileKind;
        let row = self.file_state.rows.get(self.file_state.selected)?.clone();
        if row.is_dir() {
            self.file_state.toggle_dir(self.file_state.selected);
            return None;
        }
        // Past the directory guard the row is a file; project its content kind
        // (the `Other` fallback is unreachable — a dir would have returned above).
        let kind = row.file_kind().unwrap_or(FileKind::Other);
        // s3-native browse is a read-only object listing: `Enter` on an object
        // shows an info pop-up (full key + size); there's no per-object layout map
        // or preview (cstorch objects aren't per-file safetensors).
        if let Some(RemoteBrowse::S3(uri)) = self.remote_browse() {
            let full = format!(
                "{}/{}",
                uri.trim_end_matches('/'),
                row.path.to_string_lossy()
            );
            let body = vec![
                Line::from(crate::ui::dim_span(format!(
                    "{}  ·  s3 object",
                    crate::utils::format_size(row.size as usize)
                ))),
                Line::default(),
                Line::from(Span::raw(full)),
                Line::default(),
                Line::from(crate::ui::dim_span(
                    "Remote object — browse-only (no per-object layout or preview).",
                )),
            ];
            self.with_terminal(|s, t| {
                s.float_scroll_popup(t, row.name.as_str(), body, PopupBackdrop::Files, None);
            });
            return None;
        }
        // A safetensors file opens its byte-layout map (the "proper" file view) —
        // for any such file, loaded or not (the map reads only its header).
        let is_safetensors = row
            .path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"));
        if kind == FileKind::Checkpoint && is_safetensors {
            return Some(Nav::Open(Screen::Layout {
                path: row.path.to_string_lossy().into_owned(),
                selected: 0,
                scroll: 0,
            }));
        }
        match kind {
            // A non-safetensors checkpoint that's the one we're exploring drops
            // back to its tensor tree (no per-file layout map for those formats).
            FileKind::Checkpoint if self.is_loaded_checkpoint(&row.path) => {
                return Some(Nav::Back);
            }
            FileKind::Checkpoint => {
                let cmd = format!("{PROGRAM} {}", shell_quote(&row.path.to_string_lossy()));
                let body = vec![
                    Line::from(Span::raw(
                        "This is a different checkpoint from the one open here.".to_string(),
                    )),
                    Line::default(),
                    Line::from(crate::ui::dim_span("Open it in its own view with:")),
                    Line::from(Span::raw(format!("  {cmd}"))),
                ];
                self.with_terminal(|s, t| {
                    s.float_scroll_popup(t, row.name.as_str(), body, PopupBackdrop::Files, None);
                });
            }
            FileKind::Json | FileKind::Text => {
                self.with_terminal(|s, t| s.preview_sidecar(t, &row.path, &row.name, kind));
            }
            FileKind::Other => {
                let body = vec![
                    Line::from(crate::ui::dim_span(format!(
                        "{}  ·  binary / unknown file",
                        crate::utils::format_size(row.size as usize)
                    ))),
                    Line::default(),
                    Line::from(Span::raw("No preview available.".to_string())),
                ];
                self.with_terminal(|s, t| {
                    s.float_scroll_popup(t, row.name.as_str(), body, PopupBackdrop::Files, None);
                });
            }
        }
        None
    }

    /// Read a text / JSON sidecar (capped) and float a scrollable preview over the
    /// file browser — JSON syntax-highlighted, other text plain. A non-UTF-8 file
    /// shows an info line instead.
    fn preview_sidecar(
        &self,
        term: &mut crate::tui::LiveTerminal,
        path: &Path,
        name: &str,
        kind: crate::filetree::FileKind,
    ) {
        const CAP: u64 = 4 << 20; // 4 MiB — plenty for config/tokenizer sidecars
        // Read the sidecar locally, or over SFTP for a remote checkpoint (small
        // file, read-only). s3 browse never previews (handled before this call).
        let read = match self.remote_browse() {
            Some(RemoteBrowse::Sftp(_)) => {
                self.read_remote_text_capped(&path.to_string_lossy(), CAP)
            }
            _ => read_text_capped(path, CAP),
        };
        // `copy` holds the raw text so `c` copies the file's contents verbatim.
        let (body, copy) = match read {
            Ok((text, truncated)) => {
                let mut lines = preview_lines(&text, kind);
                if lines.is_empty() {
                    lines.push(Line::from(crate::ui::dim_span("(empty file)")));
                }
                if truncated {
                    lines.push(Line::default());
                    lines.push(Line::from(crate::ui::dim_span(format!(
                        "… truncated at {}",
                        crate::utils::format_size(CAP as usize)
                    ))));
                }
                (lines, Some(text))
            }
            Err(msg) => (vec![Line::from(crate::ui::dim_span(msg))], None),
        };
        self.float_scroll_popup(term, name, body, PopupBackdrop::Files, copy);
    }

    /// Read a small remote sidecar over SFTP as UTF-8, capped at `cap` bytes —
    /// the remote analogue of [`read_text_capped`], returning `(text, truncated)`
    /// or a readable error. The whole (small) file is fetched, then capped.
    fn read_remote_text_capped(
        &self,
        path: &str,
        cap: u64,
    ) -> std::result::Result<(String, bool), String> {
        let bytes = self
            .with_remote_session(|s| s.read_file(path))
            .map_err(|e| format!("{e:#}"))?;
        let truncated = bytes.len() as u64 > cap;
        let end = if truncated { cap as usize } else { bytes.len() };
        // Cap on a char boundary so the UTF-8 decode can't split a multi-byte char.
        let mut cut = end.min(bytes.len());
        while cut > 0 && (bytes[cut - 1] & 0b1100_0000) == 0b1000_0000 {
            cut -= 1;
        }
        let text = String::from_utf8(bytes[..cut].to_vec())
            .map_err(|_| "Binary (non-UTF-8) file — no preview.".to_string())?;
        Ok((text, truncated))
    }

    /// The file browser's legend (glyphs + what `Enter` does), floated like a
    /// preview.
    fn show_files_legend(&self, term: &mut crate::tui::LiveTerminal) {
        let body = vec![
            Line::from(crate::ui::dim_span("Directories sort first, then files.")),
            Line::default(),
            Line::from(Span::raw(
                "▸ / ▾   collapsed / expanded directory".to_string(),
            )),
            Line::from(Span::raw(
                "▦        checkpoint — Enter opens its layout map (safetensors) or tensor tree"
                    .to_string(),
            )),
            Line::from(Span::raw(
                "{}       JSON — Enter previews it, highlighted".to_string(),
            )),
            Line::from(Span::raw(
                "·        text / other — Enter previews plain text".to_string(),
            )),
        ];
        self.float_scroll_popup(
            term,
            "File browser legend",
            body,
            PopupBackdrop::Files,
            None,
        );
    }

    /// A scrollable pop-up floated over the file browser (or the layout map):
    /// `↑↓`/`PgUp`/`PgDn`/`Home`/`End` scroll (held-key accelerated), any other
    /// key or a click dismisses, Ctrl-C quits. Shared by the sidecar preview, the
    /// legends, the copy-command panel, and the info pop-ups. `backdrop` chooses
    /// which frame stays live behind the box. When `copy` is `Some`, `c` copies
    /// that text to the clipboard (the footer advertises it) without dismissing.
    fn float_scroll_popup(
        &self,
        term: &mut crate::tui::LiveTerminal,
        title: &str,
        body: Vec<Line<'static>>,
        backdrop: PopupBackdrop,
        copy: Option<String>,
    ) {
        let mut scroll = 0usize;
        let mut scroll_max = 0usize;
        // When the last `c` copy happened — the "✓ copied" footer shows for
        // `COPY_FLASH`, then reverts on its own (like the copy flash elsewhere).
        let mut copied_at: Option<std::time::Instant> = None;
        // A wrong-keyboard-layout key flashes the shared hint (as in the main
        // views) instead of being silently ignored; cleared on the next input.
        let mut layout_hint: Option<char> = None;
        loop {
            let footer = if copied_at.is_some() {
                Line::from(crate::ui::success_span(
                    "✓ copied to the clipboard · Esc / click to close",
                ))
            } else if copy.is_some() {
                Line::from(crate::ui::dim_span(
                    "↑↓ PgUp/PgDn scroll · c copy · Esc / click to close",
                ))
            } else {
                files_dismiss_footer()
            };
            let hint = layout_hint;
            if term
                .draw(|f| {
                    match backdrop {
                        PopupBackdrop::Files => self.render_files_frame(f, true),
                        PopupBackdrop::Layout {
                            map,
                            selected,
                            scroll,
                        } => {
                            UI::render_layout(f, map, selected, scroll, None, true);
                        }
                    }
                    // Borrow the body (no per-frame clone) — only the visible window
                    // is copied inside, so a large header scrolls smoothly.
                    let (max, _) = UI::render_file_preview(f, title, &body, footer, scroll);
                    scroll_max = max;
                    if let Some(c) = hint {
                        UI::render_notice(f, &layout_hint_msg(c));
                    }
                })
                .is_err()
            {
                return;
            }
            // While the copy confirmation is up, wake when it expires so it clears
            // itself (not only on the next key press).
            if let Some(at) = copied_at {
                let remaining = COPY_FLASH.saturating_sub(at.elapsed());
                if remaining.is_zero() || !event::poll(remaining).unwrap_or(false) {
                    copied_at = None;
                    continue; // redraw with the confirmation gone
                }
            }
            match event::read() {
                Ok(Event::Key(key)) => {
                    if is_ctrl_c(&key) {
                        quit_immediately();
                    }
                    // A wrong-layout key flashes the hint and keeps the pop-up up.
                    if let Some(c) = wrong_layout_char(&key) {
                        layout_hint = Some(c);
                        continue;
                    }
                    layout_hint = None;
                    // Held-key acceleration, matching the tree / layout views.
                    match key.code {
                        KeyCode::Up => {
                            scroll =
                                scroll.saturating_sub(self.held_step(KeyCode::Up, accel_step_row));
                        }
                        KeyCode::Down => {
                            scroll = (scroll + self.held_step(KeyCode::Down, accel_step_row))
                                .min(scroll_max);
                        }
                        KeyCode::PageUp => {
                            let step =
                                SCROLL_PAGE * self.held_step(KeyCode::PageUp, accel_step_page);
                            scroll = scroll.saturating_sub(step);
                        }
                        KeyCode::PageDown => {
                            let step =
                                SCROLL_PAGE * self.held_step(KeyCode::PageDown, accel_step_page);
                            scroll = (scroll + step).min(scroll_max);
                        }
                        KeyCode::Home => scroll = 0,
                        KeyCode::End => scroll = scroll_max,
                        // `c` copies (when there's something to copy) and stays open;
                        // the confirmation clears itself after `COPY_FLASH`.
                        KeyCode::Char('c') if copy.is_some() => {
                            copy_to_clipboard(copy.as_deref().unwrap_or_default());
                            copied_at = Some(std::time::Instant::now());
                        }
                        // Only Esc closes (as the footer says); other keys —
                        // including a wrong keyboard layout's — are ignored rather
                        // than dismissing the pop-up unexpectedly.
                        KeyCode::Esc => return,
                        _ => {}
                    }
                }
                Ok(Event::Mouse(m)) => match m.kind {
                    MouseEventKind::ScrollUp => scroll = scroll.saturating_sub(WHEEL_STEP),
                    MouseEventKind::ScrollDown => scroll = (scroll + WHEEL_STEP).min(scroll_max),
                    MouseEventKind::Down(_) => return,
                    _ => {}
                },
                Ok(_) => {}
                Err(_) => return,
            }
        }
    }

    /// The file browser's commands available now (no preconditions — the whole
    /// registry), for its palette.
    fn available_file_commands(&self) -> Vec<FileCmdEntry> {
        FILE_COMMANDS.to_vec()
    }

    /// Run a file-browser command, from its key or the palette. Returns `Some(Nav)`
    /// for a command that leaves the file view (`Tab` → tensor tree, `q` → quit).
    fn run_file_command(
        &mut self,
        cmd: FileCmd,
        term: &mut crate::tui::LiveTerminal,
    ) -> Option<Nav> {
        match cmd {
            FileCmd::TensorTree => return Some(Nav::Back),
            FileCmd::Legend => self.show_files_legend(term),
            FileCmd::CopyPath => self.copy_file_path(),
            FileCmd::CopyScreen => self.copy_files_screen(),
            // Copy-command is engine-owned (`PaletteResult::CopyCommand` / `y`).
            FileCmd::CopyCommand => {}
            FileCmd::Quit => return Some(Nav::Quit),
        }
        None
    }

    /// Run a layout-map command, from its key or the palette. Returns `Some(Nav)`
    /// for a command that leaves the view (`Tab` → tensor tree, `q` → quit). The
    /// pop-ups float over the strip, so they need `map`/`selected`/`scroll`.
    fn run_layout_command(
        &mut self,
        cmd: LayoutCmd,
        _path: &str,
        map: &crate::safelayout::LayoutMap,
        selected: usize,
        scroll: usize,
        term: &mut crate::tui::LiveTerminal,
    ) -> Option<Nav> {
        match cmd {
            LayoutCmd::TensorTree => return Some(Nav::Back),
            LayoutCmd::Quit => return Some(Nav::Quit),
            LayoutCmd::Legend => {
                let body = layout_legend_lines();
                self.float_scroll_popup(
                    term,
                    "Layout legend",
                    body,
                    PopupBackdrop::Layout {
                        map,
                        selected,
                        scroll,
                    },
                    None,
                );
            }
            LayoutCmd::CopyScreen => {
                copy_to_clipboard(&layout_to_text(map));
                self.flash_copied("screen contents");
            }
            // Copy-command is engine-owned (`PaletteResult::CopyCommand` / `y`).
            LayoutCmd::CopyCommand => {}
        }
        None
    }

    /// Copy the selected file row's path (`f`).
    fn copy_file_path(&mut self) {
        let path = self
            .file_state
            .rows
            .get(self.file_state.selected)
            .map(|r| r.path.to_string_lossy().into_owned());
        if let Some(path) = path {
            copy_to_clipboard(&path);
            self.flash_copied("the file path");
        }
    }

    /// Copy the file browser's visible listing as plain text (`c`).
    fn copy_files_screen(&mut self) {
        let text = self
            .file_state
            .rows
            .iter()
            .map(|r| {
                let indent = "  ".repeat(r.depth);
                let name = if r.is_dir() {
                    format!("{}/", r.name)
                } else {
                    r.name.clone()
                };
                format!(
                    "{indent}{name}\t{}",
                    crate::utils::format_size(r.size as usize)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        copy_to_clipboard(&text);
        self.flash_copied("screen contents");
    }

    /// The command that reopens the layout map for `path` (`--layout`): the launch
    /// path(s) plus the flag. The file is emitted **relative to the checkpoint
    /// directory** (which the launch path already names), so the path isn't
    /// duplicated — `… <ckpt-dir> --layout model-00016.safetensors`. When a tensor
    /// is selected, `--layout-select <name>` restores the selection too.
    fn command_for_layout(&self, path: &str, select: Option<&str>) -> String {
        let mut parts = self.command_prefix();
        parts.extend(self.checkpoint_path_parts());
        parts.push("--layout".to_string());
        // Emit the path relative to the checkpoint dir (which the launch path names)
        // — the remote SFTP dir for a remote source, else the local browse root.
        let rel = match self.remote_browse() {
            Some(RemoteBrowse::Sftp(dir)) => path
                .strip_prefix(&format!("{}/", dir.trim_end_matches('/')))
                .unwrap_or(path)
                .to_string(),
            _ => Path::new(path)
                .strip_prefix(&self.browse_root)
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_else(|_| path.to_string()),
        };
        parts.push(shell_quote(&rel));
        if let Some(name) = select {
            parts.push("--layout-select".to_string());
            parts.push(shell_quote(name));
        }
        parts.join(" ")
    }

    /// Build a [`LayoutMode`] for `path`, parsing its layout locally or over the
    /// live SSH session per [`Self::remote_browse`] — so `Enter` on a remote shard
    /// shows the same byte-map as a local one (header-only either way).
    fn layout_mode(&self, path: String, selected: usize, scroll: usize) -> LayoutMode {
        let map = self.parse_layout(&path);
        LayoutMode::new(path, map, selected, scroll)
    }

    /// Parse the safetensors layout of `path`: locally for a local checkpoint, or
    /// over SFTP (header + file size, no tensor data) for a remote SFTP source. s3
    /// browse is object-list-only, so it never reaches here.
    fn parse_layout(
        &self,
        path: &str,
    ) -> std::result::Result<crate::safelayout::LayoutMap, String> {
        match self.remote_browse() {
            Some(RemoteBrowse::Sftp(_)) => {
                let (total_len, header) = self
                    .with_remote_session(|s| s.read_header_sized(path))
                    .map_err(|e| format!("{e:#}"))?;
                let name = path.rsplit('/').next().unwrap_or(path);
                crate::safelayout::parse_from(name, total_len, &header)
            }
            _ => {
                // Local: build from the cached shard header (no file read) when the
                // requested file is one of the loaded shards. Otherwise (a file not
                // in the model) fall back to reading its header.
                if let Some(cp) = self.checkpoint() {
                    let want = Path::new(path).file_name();
                    if want.is_some()
                        && let Some(sh) = cp
                            .shards
                            .iter()
                            .find(|s| Path::new(&s.path).file_name() == want)
                    {
                        let name = want.and_then(|n| n.to_str()).unwrap_or(path);
                        return Ok(crate::safelayout::from_tensors(
                            name,
                            sh.total_len,
                            sh.header_len,
                            &sh.tensors,
                            &sh.metadata,
                        ));
                    }
                }
                crate::safelayout::parse(Path::new(path)).map_err(|e| format!("{e:#}"))
            }
        }
    }

    /// The selected segment's tensor name in `map`, if the selection is a tensor
    /// (the header / a gap / out of range → `None`) — for `--layout-select`.
    fn layout_selected_tensor(
        map: &crate::safelayout::LayoutMap,
        selected: usize,
    ) -> Option<String> {
        map.segments
            .get(selected)
            .filter(|s| matches!(s.kind, crate::safelayout::SegmentKind::Tensor { .. }))
            .map(|s| s.name.clone())
    }

    /// How many segments to move the layout selection for one PageUp/PageDown —
    /// the number of bands currently on screen (at least one).
    fn layout_page_segments(
        &self,
        map: &crate::safelayout::LayoutMap,
        size: Option<ratatui::layout::Size>,
    ) -> usize {
        let Some(sz) = size else { return 1 };
        let starts = UI::layout_band_starts(map, sz.width, sz.height);
        let body = UI::layout_visible_rows(sz.width, sz.height);
        // Segments whose start row falls within one screenful — a rough page.
        let visible = starts
            .iter()
            .take(map.segments.len())
            .filter(|&&s| s < body)
            .count();
        visible.max(1)
    }

    /// `Enter` in the layout map: if the selected segment is a tensor that belongs
    /// to the loaded checkpoint, reveal it in the tensor tree (opening that screen);
    /// otherwise flash a note. Returns `Some(Nav)` when it navigates.
    fn reveal_layout_selection(
        &mut self,
        map: &crate::safelayout::LayoutMap,
        selected: usize,
    ) -> Result<Option<Nav>> {
        use crate::safelayout::SegmentKind;
        let Some(seg) = map.segments.get(selected) else {
            return Ok(None);
        };
        if !matches!(seg.kind, SegmentKind::Tensor { .. }) {
            return Ok(None);
        }
        let name = seg.name.clone();
        if self.tensors().iter().any(|t| t.name == name) {
            self.ensure_full_load()?;
            self.tree_state.reveal(&name);
            Ok(Some(Nav::Open(Screen::Tree)))
        } else {
            // A different checkpoint's file — its tensors aren't in this tree. Flash
            // a note (set directly, since it isn't a "copied" confirmation).
            self.copied_flash = Some((
                format!("{name} is not in the open checkpoint"),
                std::time::Instant::now(),
            ));
            Ok(None)
        }
    }

    /// `Enter` on the layout map's header band: float a scrollable, syntax-
    /// highlighted preview of the file's raw JSON header (the metadata that
    /// describes every tensor), pretty-printed for readability.
    fn preview_header_json(
        &self,
        term: &mut crate::tui::LiveTerminal,
        path: &str,
        map: &crate::safelayout::LayoutMap,
        selected: usize,
        scroll: usize,
    ) {
        const CAP: u64 = 2 << 20; // 2 MiB — a shard's header is far smaller
        // The JSON length `N` (the file's first 8 bytes hold this as a u64 LE).
        let n = map.header_len.saturating_sub(8);
        // `copy` holds the raw JSON so `c` copies the header verbatim.
        let (body, copy) = match crate::safelayout::read_header_json(Path::new(path), CAP) {
            Ok((json, truncated)) => {
                // Show the 8-byte length prefix first — the header is that u64
                // (little-endian) followed by `N` bytes of JSON.
                let mut lines = vec![
                    Line::from(crate::ui::dim_span(format!(
                        "8-byte length prefix (u64 LE) = {n}  ({})",
                        crate::utils::format_size(n as usize)
                    ))),
                    Line::from(crate::ui::dim_span(format!(
                        "followed by {n} bytes of JSON:"
                    ))),
                    Line::default(),
                ];
                // Keep a tensor's flat arrays (shape / data_offsets) on one line.
                // Syntax-highlighting a big header (colored_json + ANSI parse) is
                // slow, so only colour modest headers; a large one renders plain
                // (pretty-printed, uncoloured) — instant. `c` still copies the raw.
                const HIGHLIGHT_CAP: usize = 256 << 10; // 256 KiB of header JSON
                let json_lines = if json.len() <= HIGHLIGHT_CAP {
                    crate::ui::highlight_json_lines_inline(&json)
                } else {
                    crate::ui::plain_json_lines_inline(&json)
                };
                match json_lines {
                    Some(json_lines) => lines.extend(json_lines),
                    None => {
                        lines.extend(json.lines().map(|l| Line::from(Span::raw(l.to_string()))))
                    }
                }
                if truncated {
                    lines.push(Line::default());
                    lines.push(Line::from(crate::ui::dim_span(
                        "… header truncated for preview",
                    )));
                }
                (lines, Some(json))
            }
            Err(e) => (vec![Line::from(crate::ui::dim_span(e))], None),
        };
        let title = format!("{} — header", map.name);
        self.float_scroll_popup(
            term,
            &title,
            body,
            PopupBackdrop::Layout {
                map,
                selected,
                scroll,
            },
            copy,
        );
    }

    /// The layout-map screen for `tensor`'s shard, with that tensor preselected —
    /// `Some` only for a local `.safetensors` source (the layout map reads the
    /// file's header locally). Backs the detail view's `Tab` → file layout.
    fn tensor_layout_screen(&self, tensor: &TensorInfo) -> Option<Screen> {
        if self.remote_read().is_some() {
            return None;
        }
        let path = &tensor.source_path;
        let is_safetensors = Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"));
        if !is_safetensors {
            return None;
        }
        let map = crate::safelayout::parse(Path::new(path)).ok()?;
        let selected = map
            .segments
            .iter()
            .position(|s| {
                matches!(s.kind, crate::safelayout::SegmentKind::Tensor { .. })
                    && s.name == tensor.name
            })
            .unwrap_or(0);
        Some(Screen::Layout {
            path: path.clone(),
            selected,
            scroll: 0,
        })
    }

    /// Two-line status bar for the row under the cursor: a leading glyph, a
    /// primary line and a secondary line. For a tensor the primary is its full
    /// name (which the tree row may abbreviate) and the secondary is its source
    /// file; for a group the primary is its source file(s)/directory and the
    /// secondary is blank. (A copy confirmation flashes as a separate bottom-line
    /// overlay — see `copied_flash` — so it never hides this path/name.)
    /// The selected node's distinct source files — walked once per selection and
    /// cached, since a group selection otherwise re-walks its whole subtree
    /// (`collect_source_paths`, O(tensors)) on every status-bar render *and* every
    /// `f`/`t` copy. The key (selection index, tree length, search mode) changes
    /// whenever the selection or tree structure (expand/collapse/search) does.
    fn selected_source_files(&self) -> BTreeSet<String> {
        let tree = self.tree_state.visible();
        let key = (
            self.tree_state.selected,
            tree.len(),
            self.tree_state.search_mode(),
        );
        if let Some(c) = self.cached_group_files.borrow().as_ref()
            && (c.0, c.1, c.2) == key
        {
            return c.3.clone();
        }
        let mut files = BTreeSet::new();
        match tree.get(self.tree_state.selected) {
            Some((node @ TreeNode::Group { .. }, _)) => collect_source_paths(node, &mut files),
            Some((TreeNode::Tensor { info, .. }, _)) => {
                files.insert(info.source_path.clone());
            }
            _ => {}
        }
        *self.cached_group_files.borrow_mut() = Some((key.0, key.1, key.2, files.clone()));
        files
    }

    /// The bottom status bar for the current selection. The group case reuses the
    /// cached [`Self::selected_source_files`] walk, so it's cheap every frame.
    /// The status bar for the current selection — its display half of
    /// [`Self::describe_selection`], the single source both it and the `f` copy
    /// derive from (so the path shown and the path copied can never disagree).
    fn status_bar(&self) -> StatusBar {
        self.describe_selection().status
    }

    /// Copy the selected row's path to the clipboard (OSC 52) — the copy half of
    /// [`Self::describe_selection`], so `f` copies exactly the path the status bar
    /// shows.
    fn copy_selected_path(&mut self) {
        if let Some(path) = self.describe_selection().copy_path {
            copy_to_clipboard(&path);
            self.flash_copied("the source path");
        }
    }

    /// **The one place** that turns the current selection into what the UI shows
    /// and copies for it — the status-bar `(glyph, primary, secondary)` *and* the
    /// path `f` copies — computed together from a single analysis of the selected
    /// row, so the two can never drift apart (the class of "the status bar shows
    /// one path but `f` copies another" bug is made impossible by construction).
    ///
    /// Per row: a **tensor** → its own file; the **root** (the whole checkpoint) →
    /// its directory/prefix, never an arbitrary shard, even when one file is
    /// loaded; a non-root **group** → its single file, else the directory its files
    /// share (or a "stored across N files" span when they don't); **metadata** →
    /// no path.
    fn describe_selection(&self) -> SelectionView {
        let tree = self.tree_state.visible();
        let Some((node, depth)) = tree.get(self.tree_state.selected) else {
            return SelectionView::empty();
        };
        match node {
            // The full name on the first line (the tree row often abbreviates it —
            // last segment or a compacted path), the source file on the second.
            // `n` copies the name, `f` the file.
            TreeNode::Tensor { info, .. } => SelectionView {
                status: ("▪", info.name.clone(), info.source_path.clone()),
                copy_path: Some(info.source_path.clone()),
            },
            // The root (depth 0) is the whole checkpoint → its directory/prefix (so
            // `f` and the status bar both show the dir, never one shard like
            // `…/model-00016-of-00016.safetensors`). A file-count prefix adds
            // context, but the directory shown IS the directory copied.
            TreeNode::Group { .. } if *depth == 0 => {
                let dir = self.checkpoint_dir();
                let n = self.selected_source_files().len();
                let primary = if n > 1 {
                    format!("{n} files in {dir}")
                } else {
                    dir.clone()
                };
                SelectionView {
                    status: ("▸", primary, String::new()),
                    copy_path: Some(dir),
                }
            }
            // A non-root group: the file it lives in, else the directory its files
            // share. The `primary` string always contains `copy_path`, so the shown
            // and copied paths agree. (Cached per-selection walk — see
            // `selected_source_files` — so `f` on a big group doesn't re-traverse.)
            TreeNode::Group { .. } => {
                let files = self.selected_source_files();
                match files.len() {
                    0 => SelectionView::empty(),
                    1 => {
                        let file = files.into_iter().next().unwrap_or_default();
                        SelectionView {
                            status: ("▪", file.clone(), String::new()),
                            copy_path: Some(file),
                        }
                    }
                    // Files sharing a directory show + copy that directory; files
                    // that don't show a span and copy the first (nothing shared).
                    n => match common_dir(&files) {
                        Some(dir) => SelectionView {
                            status: ("▸", format!("{n} files in {dir}"), String::new()),
                            copy_path: Some(dir),
                        },
                        None => {
                            let first = files.iter().next().cloned().unwrap_or_default();
                            let first_name = file_name(&first);
                            let last_name = file_name(files.iter().next_back().unwrap());
                            SelectionView {
                                status: (
                                    "▸",
                                    format!("stored across {n} files: {first_name} … {last_name}"),
                                    String::new(),
                                ),
                                copy_path: Some(first),
                            }
                        }
                    },
                }
            }
            // The full metadata path on the first line (the tree row shows only
            // the short `…__metadata__` label); the value preview on the second.
            TreeNode::Metadata { info } => {
                let value = info.value.split_whitespace().collect::<Vec<_>>().join(" ");
                SelectionView {
                    status: ("†", info.name.clone(), value),
                    copy_path: None,
                }
            }
        }
    }

    /// The checkpoint's directory / prefix — what `f` copies for the root row: the
    /// local browse directory, the remote SFTP directory (scp-style `host:/dir`, so
    /// it's usable with `scp`/`rsync`), or the `s3://` prefix.
    fn checkpoint_dir(&self) -> String {
        match self.remote_browse() {
            Some(RemoteBrowse::S3(uri)) => uri.clone(),
            Some(RemoteBrowse::Sftp(dir)) => format!("{}:{dir}", self.remote_host_label()),
            None => self.browse_root.to_string_lossy().into_owned(),
        }
    }

    /// Copy the selected row's full name to the clipboard (the `n` shortcut): a
    /// tensor's complete name (e.g. `model.layers.0.self_attn.k_norm.weight`,
    /// which the tree may show abbreviated), or a group's path.
    fn copy_selected_name(&mut self) {
        let Some((node, _)) = self.tree_state.flattened.get(self.tree_state.selected) else {
            return;
        };
        // Name the thing copied so the confirmation is unambiguous — `n` yields a
        // tensor's full name, a group's path, or a metadata key depending on the
        // selected row.
        let what = match node {
            TreeNode::Tensor { .. } => "the full tensor name",
            TreeNode::Group { .. } => "the group path",
            TreeNode::Metadata { .. } => "the metadata key",
        };
        let name = node.name().to_string();
        if !name.is_empty() {
            copy_to_clipboard(&name);
            self.flash_copied(what);
        }
    }

    /// One screenful of tree rows, used to size a PageUp/PageDown jump so it
    /// matches what's currently visible.
    fn page_rows(&self) -> usize {
        let height = terminal::size().map(|(_, h)| h).unwrap_or(40);
        UI::visible_tree_rows(height)
    }

    /// The step (rows/cols) one press of navigation key `code` should move,
    /// accelerating while the key is held. A terminal has no key-up event —
    /// holding a key just streams repeats at the OS auto-repeat rate — so a run
    /// of the *same* key arriving faster than a human taps (within
    /// [`SCROLL_REPEAT_WINDOW`]) is treated as "held" and the step ramps up per
    /// `curve` ([`accel_step_row`] for arrows, [`accel_step_page`] for paging); a
    /// different key, or a pause, resets it to 1 so tapping stays 1:1.
    fn held_step(&self, code: KeyCode, curve: fn(u32) -> usize) -> usize {
        let now = std::time::Instant::now();
        let streak = match self.scroll_accel.get() {
            Some((last, at, n))
                if last == code && now.duration_since(at) <= SCROLL_REPEAT_WINDOW =>
            {
                n + 1
            }
            _ => 0,
        };
        self.scroll_accel.set(Some((code, now, streak)));
        curve(streak)
    }

    fn enter_search_mode(&mut self) {
        self.tree_state.search = Some(crate::kernel::SearchState::default());
        self.update_filtered_tree();
        self.tree_state.selected = 0;
        self.tree_state.scroll = 0;
    }

    /// Open the tree in search mode with a preset query (`--search`), cursor at
    /// the end — as if the query had just been typed into the search bar.
    fn open_search(&mut self, query: &str) {
        self.tree_state.search = Some(crate::kernel::SearchState {
            query: query.to_string(),
            cursor: query.chars().count(),
            filtered: Vec::new(),
        });
        self.update_filtered_tree();
        self.tree_state.selected = 0;
        self.tree_state.scroll = 0;
    }

    fn exit_search_mode(&mut self) {
        // Dropping the search state clears query/cursor/filtered atomically —
        // there's no stale search data left behind.
        self.tree_state.search = None;
        self.tree_state.selected = 0;
        self.tree_state.scroll = 0;
    }

    /// Insert a character into the query at the caret and advance past it.
    fn search_insert(&mut self, c: char) {
        if let Some(s) = self.tree_state.search.as_mut() {
            let byte = s
                .query
                .char_indices()
                .nth(s.cursor)
                .map(|(b, _)| b)
                .unwrap_or(s.query.len());
            s.query.insert(byte, c);
            s.cursor += 1;
        }
        self.update_filtered_tree();
        self.tree_state.selected = 0;
        self.tree_state.scroll = 0;
    }

    /// Delete the character before the caret (Backspace) and step the caret back.
    fn search_backspace(&mut self) {
        {
            let Some(s) = self.tree_state.search.as_mut() else {
                return;
            };
            if s.cursor == 0 {
                return;
            }
            let byte = s
                .query
                .char_indices()
                .nth(s.cursor - 1)
                .map(|(b, _)| b)
                .unwrap_or(0);
            s.query.remove(byte);
            s.cursor -= 1;
        }
        self.update_filtered_tree();
        self.tree_state.selected = 0;
        self.tree_state.scroll = 0;
    }

    /// Jump from the highlighted search result to that tensor's place in the
    /// tree: leave search mode, then expand to and select it so it's shown in
    /// context (a no-op if the highlighted result isn't a tensor).
    fn reveal_search_result(&mut self) {
        let sel = self.tree_state.selected;
        let name = match self
            .tree_state
            .search
            .as_ref()
            .and_then(|s| s.filtered.get(sel))
        {
            Some((TreeNode::Tensor { info, .. }, _)) => info.name.clone(),
            _ => return,
        };
        self.exit_search_mode();
        self.tree_state.reveal(&name);
    }

    /// Activate the highlighted tree row (shared by Enter, Space, and a left
    /// mouse click): open a tensor (returns `Nav::Open`), toggle a group, or show
    /// metadata in place. Returns `Some(nav)` when the caller should navigate.
    fn activate_selection(&mut self) -> Option<Nav> {
        match self.handle_selection() {
            (Some(screen), _) => Some(Nav::Open(screen)),
            (None, Some(info)) => {
                let mut term = self.terminal.take().expect("interactive loop owns it");
                self.show_metadata_detail(&mut term, &info);
                self.terminal = Some(term);
                None
            }
            (None, None) => None,
        }
    }

    /// Act on the highlighted tree row. Returns `Some(Screen::Detail)` when a
    /// tensor was selected (the navigator opens it); groups expand in place,
    /// returning `None`. The second element is the metadata entry to open in place
    /// (cloned out so the caller, which owns the live terminal, can draw it through
    /// Ratatui) — `None` for groups/tensors.
    fn handle_selection(&mut self) -> (Option<Screen>, Option<MetadataInfo>) {
        let tree = self.tree_state.visible();

        let Some((selected_node, _)) = tree.get(self.tree_state.selected) else {
            return (None, None);
        };
        // Tensor / metadata return owned data (the `tree` borrow ends here). A
        // group falls through to the in-place toggle below, which needs `&mut
        // self` — so it must run after the borrow, not inside the match.
        match selected_node {
            TreeNode::Tensor { info, .. } => {
                return (
                    Some(Screen::Detail {
                        tensor: info.name.clone(),
                        slice: 0,
                    }),
                    None,
                );
            }
            TreeNode::Metadata { info } => {
                return (None, Some(info.clone()));
            }
            // In search mode groups shouldn't appear, but if one does, do nothing.
            TreeNode::Group { .. } => {}
        }
        if !self.tree_state.search_mode() {
            self.tree_state.toggle_group_at(self.tree_state.selected);
        }
        (None, None)
    }

    /// Report a request that can't be honored. In the navigator this shows a
    /// recoverable message and waits for a key (the caller then falls back to the
    /// tree); the headless and one-shot modes draw nothing and rely on the
    /// returned error, which propagates to a non-zero process exit.
    fn reject_open(
        &mut self,
        mode: OpenMode,
        title: &str,
        screen_detail: &str,
        error: &str,
    ) -> anyhow::Error {
        // Interactively, the live terminal is up (set in `run`); draw the
        // recoverable message through it. Headless / one-shot draw nothing and
        // rely on the returned error.
        if mode == OpenMode::Interactive
            && let Some(term) = self.terminal.as_mut()
        {
            let _ = term.draw(|f| UI::render_message(f, title, screen_detail));
            let _ = event::read();
        }
        anyhow::anyhow!("{error}")
    }

    /// Apply a CLI open request: tree state/search, then locate the tensor and
    /// apply any dtype/shape/slice/layout overrides, and either render it once
    /// (`--exit`, [`OpenMode::OneShot`]) or return the [`Screen`] to render
    /// ([`OpenMode::Headless`]) or seed the navigator with ([`OpenMode::Interactive`]).
    /// `Ok(None)` means "show the tree" (no specific target requested); `Err`
    /// means the request couldn't be honored (unknown tensor/metadata, ambiguous,
    /// bad `--shape`/`--slice`) — fatal in the headless/one-shot modes, recoverable
    /// in the navigator (see [`Self::reject_open`]).
    fn open_requested(&mut self, req: OpenRequest, mode: OpenMode) -> Result<Option<Screen>> {
        // Tree-browser state applies whichever screen opens (and is what makes
        // these reachable headlessly): bulk expansion, then a search filter.
        match req.tree_state {
            Some(TreeState::Expanded) => self.tree_state.set_all_expanded(true),
            Some(TreeState::Collapsed) => self.tree_state.set_all_expanded(false),
            None => {}
        }
        if let Some(query) = &req.search {
            self.open_search(query);
        }

        // `--metadata`: metadata lives only in the tree, so reveal that entry and
        // stay on the tree (this is what `y` on a metadata row reproduces).
        if let Some(name) = req.target.metadata() {
            if self.metadata().iter().any(|m| m.name == name) {
                self.tree_state.reveal(name);
                return Ok(None);
            }
            return Err(self.reject_open(
                mode,
                "Metadata not found",
                &format!(
                    "No metadata entry named '{name}' in this checkpoint — opening the browser instead."
                ),
                &format!("no metadata entry named '{name}' in this checkpoint"),
            ));
        }
        // A tree screen with no specific tensor (`--tree-state` / `--search` /
        // `--legend` alone, or a bare launch routed to the tree): just show the
        // browser in whatever state the flags above set — don't demand a tensor.
        if req.view == OpenView::Tree && req.target.tensor().is_none() {
            return Ok(None);
        }
        // Resolve the target tensor: the named one, or — when `--tensor` is
        // omitted — the sole tensor if the checkpoint has exactly one (e.g. any
        // `.npy`, or a single-array `.npz`/HDF5/safetensors). Ambiguous otherwise.
        let tensor = match req.target.tensor() {
            Some(name) => match self.tensors().iter().find(|t| t.name == name) {
                Some(t) => t.clone(),
                None => {
                    return Err(self.reject_open(
                        mode,
                        "Tensor not found",
                        &format!(
                            "No tensor named '{name}' in this checkpoint — opening the browser instead."
                        ),
                        &format!("no tensor named '{name}' in this checkpoint"),
                    ));
                }
            },
            None => match self.tensors() {
                [only] => only.clone(),
                // No `--tensor` and not a single-tensor checkpoint: ambiguous when
                // there's more than one (an error), or nothing to open at all.
                _ if self.tensors().len() > 1 => {
                    return Err(self.reject_open(
                        mode,
                        "Which tensor?",
                        "This checkpoint has multiple tensors — name one with --tensor, or pick it in the browser.",
                        "this checkpoint has multiple tensors; name one with --tensor",
                    ));
                }
                _ => return Ok(None),
            },
        };
        // Apply the dtype override (skipped for formats that can't reinterpret,
        // so the header never claims a view that isn't actually applied).
        if let Some(dt) = req.dtype
            && dtype_overridable(&tensor)
        {
            let def = self.default_view(&tensor.name);
            // `--dtype unpacked` only applies to a tensor that carries a packing
            // schema; otherwise keep its default view.
            let dt = if dt == ViewDtype::Unpacked && self.schema_for(&tensor.name).is_none() {
                def
            } else {
                dt
            };
            let mut overrides = self.data_view.dtype_overrides.borrow_mut();
            // Record only an explicit non-default choice, so an unset tensor falls
            // back to its default (Unpacked for schema tensors) and `y` round-trips.
            if dt == def {
                overrides.remove(&tensor.name);
            } else {
                overrides.insert(tensor.name.clone(), dt);
            }
        }
        // Apply the shape override (a reshape with a matching element count).
        if let Some(s) = req.shape.as_deref()
            && dtype_overridable(&tensor)
        {
            match parse_shape_input(s, tensor.num_elements) {
                Ok(shape) => {
                    self.data_view
                        .shape_overrides
                        .borrow_mut()
                        .insert(tensor.name.clone(), shape);
                }
                Err(msg) => {
                    return Err(self.reject_open(
                        mode,
                        "Can't apply --shape",
                        &msg,
                        &format!("--shape: {msg}"),
                    ));
                }
            }
        }
        if let Some(layout) = req.layout {
            self.data_view.data_view_layout.set(layout);
        }
        // Position within the layout (clamped to valid bounds on the first draw).
        if let Some((row, col)) = req.window_at {
            self.data_view.data_view_win_row.set(row);
            self.data_view.data_view_win_col.set(col);
        }
        if let Some((row_tail, col_tail)) = req.edge_split {
            self.data_view.data_view_row_tail.set(row_tail);
            self.data_view.data_view_col_tail.set(col_tail);
        }
        if let Some(zebra) = req.zebra {
            self.data_view.data_view_stripe.set(zebra);
        }
        if let Some(base) = req.base {
            self.data_view.data_view_base.set(base);
        }
        // Resolve the starting slice against this tensor's slice count — the
        // leading dimension of the (possibly overridden) squeezed shape (so an
        // (N, M, 1, K) tensor pages through N slices, matching the data view);
        // 1D/2D have a single slice. Accepts an index or a percentage.
        let eff_shape = self
            .data_view
            .shape_overrides
            .borrow()
            .get(&tensor.name)
            .cloned()
            .unwrap_or_else(|| tensor.shape.clone());
        let slices = match crate::sample::squeezed_shape(&eff_shape).as_slice() {
            [d0, _, _] => *d0,
            _ => 1,
        };
        let start_slice = match req.slice.as_deref() {
            Some(s) => match parse_slice_input(s, slices) {
                Ok(Some(n)) => n,
                Ok(None) => 0,
                Err(msg) => {
                    return Err(self.reject_open(
                        mode,
                        "Can't apply --slice",
                        &msg,
                        &format!("--slice: {msg}"),
                    ));
                }
            },
            None => 0,
        };
        let stats_start = if req.compute_stats {
            StatsStart::Auto
        } else {
            StatsStart::OnDemand
        };
        let screen = match req.view {
            OpenView::Detail => Screen::Detail {
                tensor: tensor.name.clone(),
                slice: start_slice,
            },
            OpenView::Values => Screen::Data {
                tensor: tensor.name.clone(),
                repr: Representation::Values,
                slice: start_slice,
            },
            OpenView::Heatmap => Screen::Data {
                tensor: tensor.name.clone(),
                repr: Representation::Heatmap,
                slice: start_slice,
            },
            // `--tree`: don't open a view — land on the tree with this tensor
            // revealed and highlighted (the dtype/shape overrides applied above
            // stay set for when it's opened). Return `None` so the navigator
            // stays on the tree (cursor 0) with the selection we just set.
            OpenView::Tree => {
                self.tree_state.reveal(&tensor.name);
                return Ok(None);
            }
        };

        // `--histogram` / `--bins`: pre-compute the value histogram so the detail
        // screen shows it on first render — this is what makes `y`'s `--histogram`
        // (and `--bins N`) round-trip restore the view. A bucket count implies
        // showing the histogram. Done before the one-shot below so `--exit`
        // captures it too.
        if let Some(n) = req.histogram.bins() {
            self.data_view.histogram_bins.set(Some(n));
        }
        if req.histogram.on()
            && let Screen::Detail { .. } = screen
        {
            let view = self.active_view(&tensor.name);
            let shape = self
                .data_view
                .shape_overrides
                .borrow()
                .get(&tensor.name)
                .cloned()
                .unwrap_or_else(|| tensor.shape.clone());
            // The pre-warm animates through the live terminal (set in interactive
            // / one-shot modes; this block never runs headless, where the request
            // strips `--histogram`/`--bins`).
            let mut term = self
                .terminal
                .take()
                .expect("interactive loop owns the terminal");
            self.ensure_detail_histogram(
                &mut term,
                &tensor,
                view,
                &shape,
                dtype_overridable(&tensor),
                self.unindexed.contains(&tensor.source_path),
            );
            self.terminal = Some(term);
        }

        // One-shot (`--exit`): render the requested screen once and return (the
        // navigator is never entered).
        if mode == OpenMode::OneShot {
            match &screen {
                Screen::Detail { tensor, slice } => {
                    self.run_mode_once(&mut DetailMode::new(
                        tensor.clone(),
                        *slice,
                        stats_start,
                        Interaction::OneShot,
                    ))?;
                }
                Screen::Data {
                    tensor,
                    repr,
                    slice,
                } => {
                    self.run_mode_once(&mut DataMode::new(
                        tensor.clone(),
                        *repr,
                        *slice,
                        Interaction::OneShot,
                    ))?;
                }
                // The tree renders itself; the file browser, layout map, rename
                // editor and stats view are interactive-only (a `--files` / `--layout`
                // / `--stats` with `--exit` falls back to the headless render).
                Screen::Tree
                | Screen::Files
                | Screen::Layout { .. }
                | Screen::Rename { .. }
                | Screen::Stats { .. } => {}
            }
            return Ok(None);
        }

        // Headless (`--plain` / `--emit-command`): hand the resolved screen back
        // for the caller to render — no interactive drawing on this path.
        if mode == OpenMode::Headless {
            return Ok(Some(screen));
        }

        // Interactive: `--compute-stats` pre-warms the detail's stats so they
        // show on first render (the navigator itself always opens on-demand).
        if stats_start == StatsStart::Auto
            && let Screen::Detail { .. } = screen
        {
            let view = self.active_view(&tensor.name);
            let overridable = dtype_overridable(&tensor);
            let unindexed = self.unindexed.contains(&tensor.source_path);
            let shape = self
                .data_view
                .shape_overrides
                .borrow()
                .get(&tensor.name)
                .cloned()
                .unwrap_or_else(|| tensor.shape.clone());
            let mut term = self
                .terminal
                .take()
                .expect("interactive loop owns the terminal");
            self.compute_stats_animated(&mut term, &tensor, view, |f, sv| {
                self.render_detail_frame(
                    f,
                    &tensor,
                    &shape,
                    view,
                    overridable,
                    unindexed,
                    sv,
                    None,
                    None,
                    None,
                );
            });
            self.terminal = Some(term);
        }
        Ok(Some(screen))
    }

    /// Compute and show the detail screen's value histogram (animating the bars
    /// filling in below the statistics). Floats / wide integers need the value
    /// range, so stats are computed first when the bin layout can't be decided
    /// without them. Shared by the `h` key and the `--histogram` startup restore,
    /// so both produce the same result and `y` round-trips it.
    fn ensure_detail_histogram(
        &self,
        term: &mut crate::tui::LiveTerminal,
        tensor: &TensorInfo,
        view: ViewDtype,
        shape: &[usize],
        overridable: bool,
        unindexed: bool,
    ) {
        let range = self.histogram_range(tensor, view);
        let need_stats = crate::sample::histogram_bins(
            view,
            &tensor.dtype,
            range,
            self.data_view.histogram_bins.get(),
        )
        .is_none()
            && self.cached_stats(tensor, view).is_none();
        let ready = !need_stats
            || matches!(
                self.compute_stats_animated(term, tensor, view, |f, sv| {
                    self.render_detail_frame(
                        f,
                        tensor,
                        shape,
                        view,
                        overridable,
                        unindexed,
                        sv,
                        None,
                        None,
                        None,
                    );
                }),
                ScanOutcome::Completed
            );
        if ready {
            self.scan_histogram(term, tensor, view, |term, snap, scanning, overlay| {
                let stats = self.cached_stats(tensor, view);
                let sv = match &stats {
                    Some(s) => StatsView::Ready(s),
                    None => StatsView::Pending,
                };
                let _ = term.draw(|f| {
                    self.render_detail_frame(
                        f,
                        tensor,
                        shape,
                        view,
                        overridable,
                        unindexed,
                        sv,
                        Some(snap),
                        scanning,
                        overlay,
                    )
                });
            });
        }
    }

    /// Compute the whole-tensor value histogram for `(tensor, view)` into the
    /// cache, calling `redraw` with the running snapshot (and a spinner / elapsed
    /// / fraction, plus any pop-up overlay) each frame so the caller can animate
    /// the bins filling in on its own screen. A no-op if already cached. `l` / `y`
    /// raise the legend / copied-command pop-up over the still-filling bars
    /// (dismissed by any key); any other key cancels the scan (nothing is cached,
    /// so it recomputes next time); Ctrl-C quits. The bin layout needs the value
    /// range for floats / wide integers, taken from cached stats — the caller
    /// computes those first when required (the 4-bit views don't need them).
    fn scan_histogram(
        &self,
        term: &mut crate::tui::LiveTerminal,
        tensor: &TensorInfo,
        view: ViewDtype,
        mut redraw: impl FnMut(
            &mut crate::tui::LiveTerminal,
            &Histogram,
            Option<crate::ui::ScanProgress>,
            Option<&Overlay>,
        ),
    ) {
        let count = self.data_view.histogram_bins.get();
        let key = (tensor.name.clone(), view, count);
        if self.histogram_cache.borrow().contains_key(&key) {
            return;
        }
        let range = self.histogram_range(tensor, view);
        let Some((bins, n)) = crate::sample::histogram_bins(view, &tensor.dtype, range, count)
        else {
            return; // a range is required but stats aren't available
        };

        // Scan on a worker, accumulating into shared atomics so the bars can be
        // redrawn filling in.
        let shared = Arc::new(HistShared::new(n));
        let cancel = Arc::new(AtomicBool::new(false));
        let pause = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicUsize::new(0));
        let total = tensor.size_bytes;
        let owned = tensor.clone();
        let schema = self.schema_for(&tensor.name).cloned();
        let (wsh, wc, wp, wd) = (
            Arc::clone(&shared),
            Arc::clone(&cancel),
            Arc::clone(&pause),
            Arc::clone(&done),
        );
        let handle = std::thread::spawn(move || {
            crate::sample::tensor_histogram_into(
                &owned,
                view,
                schema.as_ref(),
                bins,
                n,
                &wsh,
                &wc,
                &wp,
                Some(&*wd),
            )
        });

        let started = std::time::Instant::now();
        let mut frame = 0usize;
        // A pop-up raised mid-scan (`l` / `y`); composited over the filling bars,
        // it never cancels the scan, which keeps computing behind it.
        let mut overlay: Option<Overlay> = None;
        while !handle.is_finished() {
            let progress =
                (total > 0).then(|| (done.load(Ordering::Relaxed) as f64 / total as f64).min(1.0));
            redraw(
                term,
                &shared.snapshot(bins),
                Some((
                    STATS_SPINNER[frame % STATS_SPINNER.len()],
                    started.elapsed(),
                    progress,
                )),
                overlay.as_ref(),
            );
            frame += 1;
            if event::poll(std::time::Duration::from_millis(80)).unwrap_or(false)
                && let Ok(Event::Key(kev)) = event::read()
            {
                if is_ctrl_c(&kev) {
                    quit_immediately();
                }
                if overlay.is_some() {
                    overlay = None; // dismiss the pop-up; the scan kept running
                } else {
                    match kev.code {
                        KeyCode::Char('l') => overlay = Some(Overlay::Legend(Legend::Detail)),
                        KeyCode::Char('y') => {
                            let cmd = self.command_for_detail(tensor);
                            copy_to_clipboard(&cmd);
                            overlay = Some(Overlay::Command(cmd));
                        }
                        // Any other key aborts the (possibly slow) scan.
                        _ => {
                            cancel.store(true, Ordering::Relaxed);
                            return; // cancelled — leave it uncached so `g` recomputes
                        }
                    }
                }
            }
        }
        match handle.join() {
            Ok(Ok(())) => {
                let mut hist = shared.snapshot(bins);
                // Record the scan time so the heading keeps showing it after the
                // bars have finished forming (mirroring the statistics line).
                hist.elapsed = started.elapsed();
                // A pop-up opened mid-scan stays up after the bars finish (rather
                // than vanishing the instant it completes); any key dismisses it.
                while overlay.is_some() {
                    redraw(term, &hist, None, overlay.as_ref());
                    if let Ok(Event::Key(kev)) = event::read() {
                        if is_ctrl_c(&kev) {
                            quit_immediately();
                        }
                        overlay = None;
                    }
                }
                self.histogram_cache.borrow_mut().insert(key, hist);
            }
            Ok(Err(msg)) => {
                let _ = term.draw(|f| UI::render_message(f, "Histogram unavailable", &msg));
                let _ = event::read();
            }
            Err(_) => {}
        }
    }

    /// Sample the heatmap / numeric grid for `(slice, view)`, sizing the grid to
    /// the `(cols, rows)` terminal size so the header and footer stay on screen,
    /// and leave the result in [`Self::sample_cache`]. Returns `(slices,
    /// overridable, clamped_slice)`. Shared by the Ratatui
    /// [`Self::render_data_frame`] / [`Self::data_plain`], so all three agree on
    /// the sample (and reuse the cache between a scan's spinner-frame redraws).
    #[allow(clippy::too_many_arguments)] // mirrors the data-view sampler's params
    fn prepare_data_sample(
        &self,
        tensor: &TensorInfo,
        repr: Representation,
        slice: usize,
        view: ViewDtype,
        mode: SampleMode,
        stats: StatsView,
        cols: u16,
        rows: u16,
    ) -> Result<(usize, bool, usize), String> {
        let width = cols as usize;
        let heatmap = matches!(repr, Representation::Heatmap);
        // Leading-index count of the (possibly reshaped) tensor — drives whether
        // a slice line appears in the header.
        let slices = {
            let overrides = self.data_view.shape_overrides.borrow();
            let eff = overrides
                .get(&tensor.name)
                .cloned()
                .unwrap_or_else(|| tensor.shape.clone());
            let stored = match crate::sample::squeezed_shape(&eff).as_slice() {
                [d0, _, _] => *d0,
                _ => 1,
            };
            // The codebook unmerge turns each stored expert into `lenP` slices.
            match (view, self.schema_for(&tensor.name)) {
                (ViewDtype::Unpacked, Some(s)) => stored * s.len_p(),
                _ => stored,
            }
        };
        // Size the grid to leave the header (tensor name + file path, dtype/
        // shape/layout, stats, a slice line for 3D, the blank spacer, and the
        // column-index row for the numeric grid) and the footer (blank + the
        // auto-wrapped hint line) on screen — so neither scrolls off the top.
        let header = HDR_TITLE_ROWS
            + HDR_DTYPE_ROW
            + HDR_STATS_ROW
            + HDR_GRID_GAP_ROW
            + if slices > 1 { HDR_SLICE_ROW } else { 0 }
            + if heatmap { 0 } else { HDR_COLINDEX_ROW };
        let footer = crate::ui::data_view_footer_lines(
            mode,
            slices,
            dtype_overridable(tensor),
            heatmap,
            self.data_view.data_view_stripe.get(),
            self.data_view.data_view_base.get(),
            width,
        );
        let text_rows = (rows as usize).saturating_sub(header + footer).max(1);
        let (max_rows, max_cols) = match repr {
            // The heatmap packs two data rows per text line (half blocks), so it
            // can sample twice as many rows as there are lines.
            Representation::Heatmap => (text_rows * 2, (cols as usize).saturating_sub(1).max(1)),
            // Numeric cell width depends on the base (hex/oct/bin are fixed,
            // wider) and, for decimal, the actual values (small ints — even in a
            // wide dtype — pack many columns); plus a 7-char row-index column.
            // The exact range comes from stats once computed. Must match the
            // width `draw_values` renders with, or the grid overflows the line.
            Representation::Values => {
                let cell = self.data_view.data_view_base.get().cell_width(
                    view,
                    &tensor.dtype,
                    stats.value_range(),
                );
                (text_rows, ((cols as usize).saturating_sub(7) / cell).max(1))
            }
        };
        // Remember the edges-view budgets so an arrow press can move the divider
        // by exactly one index (step = 1 / budget).
        self.data_view
            .edge_row_budget
            .set(crate::sample::edge_total(max_rows));
        self.data_view
            .edge_col_budget
            .set(crate::sample::edge_total(max_cols));
        // Reuse the last sample when nothing that affects the grid changed. This
        // is what keeps a stats scan's spinner-frame redraws from re-reading the
        // tensor every frame (which would block on the worker's HDF5 lock and lag
        // the UI); only an actual change — dtype, layout, slice, pan, resize, or
        // the exact stats landing — re-samples.
        // The effective shape: a session shape override if set, else the stored
        // shape. Region reads still use the real stored shape, so any reshape
        // with a matching element count is a valid row-major reinterpretation.
        let eff_shape = self
            .data_view
            .shape_overrides
            .borrow()
            .get(&tensor.name)
            .cloned()
            .unwrap_or_else(|| tensor.shape.clone());
        let key: SampleKey = (
            tensor.name.clone(),
            repr,
            slice,
            view,
            mode,
            max_rows,
            max_cols,
            eff_shape.clone(),
        );
        let hit = self
            .sample_cache
            .borrow()
            .as_ref()
            .is_some_and(|c| c.key == key);
        if !hit {
            let schema = self.schema_for(&tensor.name);
            let sample = self.with_reader(tensor, |reader| {
                crate::sample::sample_tensor_with(
                    reader, tensor, &eff_shape, max_rows, max_cols, slice, view, mode, schema,
                )
            })?;
            // In the window layout, read the clamped top-left corner and the
            // visible size back from the rendered sample, so panning stays in
            // bounds and a Shift+arrow strides exactly one screenful.
            if let SampleMode::Window { .. } = mode {
                self.data_view
                    .data_view_win_row
                    .set(sample.rows.first().copied().unwrap_or(0));
                self.data_view
                    .data_view_win_col
                    .set(sample.cols.first().copied().unwrap_or(0));
                self.data_view.win_page_rows.set(sample.rows.len().max(1));
                self.data_view.win_page_cols.set(sample.cols.len().max(1));
            }
            *self.sample_cache.borrow_mut() = Some(CachedSample { key, sample });
        }
        let cache = self.sample_cache.borrow();
        let sample = &cache.as_ref().unwrap().sample;
        Ok((sample.slices, sample.overridable, sample.slice))
    }

    /// Open the dtype-selection menu with a live preview, returning the chosen
    /// view or `None` if cancelled. `d`/→ move forward, `D`/← back (the menu is
    /// horizontal); Enter applies, Esc cancels. Ctrl-C quits the app. The
    /// preview re-renders whichever screen the menu was opened from.
    fn prompt_dtype(
        &self,
        term: &mut crate::tui::LiveTerminal,
        tensor: &TensorInfo,
        preview: DtypePreview,
    ) -> Option<ViewDtype> {
        let options =
            crate::sample::view_options_for(&tensor.dtype, self.schema_for(&tensor.name).is_some());
        if options.is_empty() {
            return None;
        }
        let current = self.active_view(&tensor.name);
        let mut idx = options.iter().position(|v| *v == current).unwrap_or(0);
        // The shape override (if any) is fixed while the dtype menu is open.
        let shape = self
            .data_view
            .shape_overrides
            .borrow()
            .get(&tensor.name)
            .cloned()
            .unwrap_or_else(|| tensor.shape.clone());
        let stripe = self.data_view.data_view_stripe.get();
        let base = self.data_view.data_view_base.get();
        loop {
            // Live preview of the highlighted view, then the menu band over it —
            // all in one Ratatui frame. Only read cached stats: navigating the menu
            // must never trigger a scan.
            let stats = self.cached_stats(tensor, options[idx]);
            let stats_view = stats.as_ref().map_or(StatsView::Pending, StatsView::Ready);
            let mut preview_ok = true;
            let drew = term.draw(|f| match preview {
                DtypePreview::Detail => {
                    self.render_detail_frame(
                        f,
                        tensor,
                        &shape,
                        options[idx],
                        true,
                        self.unindexed.contains(&tensor.source_path),
                        stats_view,
                        None,
                        None,
                        None,
                    );
                    UI::render_dtype_menu(f, &options, idx);
                }
                DtypePreview::Data { repr, slice, mode } => {
                    if self
                        .render_data_frame(
                            f,
                            tensor,
                            repr,
                            slice,
                            options[idx],
                            mode,
                            stats_view,
                            stripe,
                            base,
                            None,
                        )
                        .is_err()
                    {
                        preview_ok = false;
                        return;
                    }
                    UI::render_dtype_menu(f, &options, idx);
                }
            });
            if drew.is_err() || !preview_ok {
                return None;
            }
            match event::read() {
                Ok(Event::Key(key)) if is_ctrl_c(&key) => quit_immediately(),
                Ok(Event::Key(KeyEvent { code, .. })) => match code {
                    KeyCode::Right | KeyCode::Char('d') => idx = (idx + 1) % options.len(),
                    KeyCode::Left | KeyCode::Char('D') => {
                        idx = (idx + options.len() - 1) % options.len()
                    }
                    KeyCode::Enter => return Some(options[idx]),
                    KeyCode::Esc => return None,
                    _ => {}
                },
                Ok(_) => {}
                Err(_) => return None,
            }
        }
    }

    /// Prompt for a slice to jump to — either an absolute index (`123`) or a
    /// percentage of the way through (`50%`, where 0% is the first slice and
    /// 100% the last). Returns the chosen slice, or `None` if cancelled / left
    /// empty. Out-of-range entries are reported in the prompt, not jumped to.
    /// Ctrl-C quits the app outright.
    /// Prompt for a shape override (`r`). The entry is a list of dimensions
    /// (separated by `,`, space, or `x`) whose product must equal the tensor's
    /// element count. Enter on an empty entry clears any override; `Esc`
    /// cancels. Prefilled with the current override, if any.
    fn prompt_reshape(
        &self,
        term: &mut crate::tui::LiveTerminal,
        background: impl Fn(&mut ratatui::Frame),
        tensor: &TensorInfo,
        current: Option<&[usize]>,
    ) -> ReshapeChoice {
        let mut input = current
            .map(|s| {
                s.iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_default();
        let mut error: Option<String> = None;
        loop {
            if term
                .draw(|f| {
                    background(f);
                    UI::render_reshape_prompt(
                        f,
                        tensor.num_elements,
                        &tensor.shape,
                        &input,
                        error.as_deref(),
                    )
                })
                .is_err()
            {
                return ReshapeChoice::Cancel;
            }
            match event::read() {
                Ok(Event::Key(key)) if is_ctrl_c(&key) => quit_immediately(),
                Ok(Event::Key(KeyEvent { code, .. })) => match code {
                    KeyCode::Enter => {
                        if input.trim().is_empty() {
                            return ReshapeChoice::Clear;
                        }
                        match parse_shape_input(&input, tensor.num_elements) {
                            Ok(shape) => return ReshapeChoice::Set(shape),
                            Err(msg) => error = Some(msg),
                        }
                    }
                    KeyCode::Esc => return ReshapeChoice::Cancel,
                    KeyCode::Backspace => {
                        input.pop();
                        error = None;
                    }
                    // Accept digits, separators, and wildcard tokens (`*`, `-1`, `_`).
                    KeyCode::Char(c)
                        if c.is_ascii_digit() || matches!(c, ',' | ' ' | 'x' | '*' | '-' | '_') =>
                    {
                        input.push(c);
                        error = None;
                    }
                    _ => {}
                },
                Ok(_) => {}
                Err(_) => return ReshapeChoice::Cancel,
            }
        }
    }

    fn prompt_slice(
        &self,
        term: &mut crate::tui::LiveTerminal,
        background: impl Fn(&mut ratatui::Frame),
        slices: usize,
    ) -> Option<usize> {
        let mut input = String::new();
        let mut error: Option<String> = None;
        loop {
            if term
                .draw(|f| {
                    background(f);
                    UI::render_slice_prompt(f, slices, &input, error.as_deref());
                })
                .is_err()
            {
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

    fn show_metadata_detail(&self, term: &mut crate::tui::LiveTerminal, metadata: &MetadataInfo) {
        if term
            .draw(|f| UI::render_metadata_detail(f, metadata))
            .is_ok()
        {
            // Wait for a key (ignore mouse) so the value can be selected by mouse.
            wait_for_dismiss();
        }
    }

    /// The single HDF5 file backing this checkpoint, if repacking applies (one
    /// `.h5`/`.hdf5` file). `None` for safetensors/GGUF or multi-file views.
    fn repack_input(&self) -> Option<PathBuf> {
        match self.files.as_slice() {
            [f] if matches!(f.extension().and_then(|e| e.to_str()), Some("h5" | "hdf5")) => {
                Some(f.clone())
            }
            _ => None,
        }
    }

    /// The path an in-place rename would operate on: a *local* safetensors
    /// checkpoint — a single `.safetensors` file, or the directory holding its
    /// shards (so every shard *and* the index are renamed consistently). `None`
    /// (command hidden) for a remote source or any non-safetensors format.
    fn rename_target(&self) -> Option<PathBuf> {
        if self.remote_read().is_some() || self.files.is_empty() {
            return None;
        }
        if !self.files.iter().all(|f| {
            f.extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"))
        }) {
            return None;
        }
        match self.files.as_slice() {
            [one] => Some(one.clone()),
            many => Some(browse_root_of(many)),
        }
    }

    /// Whether the checkpoint's shard files can actually be written — a local
    /// safetensors checkpoint on a read-only filesystem (an `ro` SSH mount) or a
    /// read-only file exists and looks local but the in-place rename would fail.
    /// Uses the same [`crate::rename::is_writable`] probe the rename pre-flight does
    /// (one source of truth), and caches it since the badge / palette re-check it
    /// every frame.
    fn checkpoint_writable(&self) -> bool {
        if let Some(w) = self.writable.get() {
            return w;
        }
        let w = !self.files.is_empty() && self.files.iter().all(|f| crate::rename::is_writable(f));
        self.writable.set(Some(w));
        w
    }

    /// Whether the open checkpoint can be renamed in place: a local safetensors
    /// checkpoint ([`Self::rename_target`]) whose files are writable
    /// ([`Self::checkpoint_writable`]). The single gate for the `editable` badge,
    /// the `Rename` command, the `R` key, and `--rename`.
    fn can_rename(&self) -> bool {
        self.rename_target().is_some() && self.checkpoint_writable()
    }

    /// The `convert --map …` CLI command equivalent to the renames staged in `mode`
    /// (one `--map 'PATTERN=>REPLACEMENT'` per complete pair), or `None` until a pair
    /// is complete. Shown in the editor and copyable with `^Y`.
    fn rename_cli_command(&self, target: &Path, mode: &RenameMode) -> Option<String> {
        let mut rules = Vec::new();
        for p in &mode.pairs {
            if p.source.trim().is_empty() || p.target.trim().is_empty() {
                continue;
            }
            if let Ok((pat, rep)) = crate::rename::rule_from_fields(&p.source, &p.target) {
                rules.push((pat, rep));
            }
        }
        if rules.is_empty() {
            return None;
        }
        let mut parts = self.command_prefix(); // PROGRAM (rename is local → just it)
        parts.push("convert".to_string());
        parts.push(shell_quote(&target.to_string_lossy()));
        for (pat, rep) in rules {
            parts.push("--map".to_string());
            parts.push(shell_quote(&format!("{pat}=>{rep}")));
        }
        Some(parts.join(" "))
    }

    /// Apply the rename staged in `mode`: build the plan, write it, reload the tree
    /// so it shows the new names, and flash a confirmation. Returns the `Nav` back to
    /// the tree, or an error string to surface in the editor.
    fn apply_rename_mode(
        &mut self,
        loaded: &crate::rename::Loaded,
        mode: &RenameMode,
    ) -> std::result::Result<Nav, String> {
        let (map, _) = mode.build_map()?;
        let plan = loaded.plan(&map).map_err(|e| format!("{e:#}"))?;
        let count = plan.rename_count();
        crate::rename::apply(&plan).map_err(|e| format!("apply failed: {e:#}"))?;
        let msg = match self.reload_after_rename() {
            Ok(()) => format!("Renamed {count} tensor(s) in place"),
            Err(e) => format!("Renamed {count} tensor(s); reopen to refresh ({e:#})"),
        };
        self.copied_flash = Some((msg, std::time::Instant::now()));
        Ok(Nav::Back)
    }

    /// Re-read the checkpoint after an in-place rename so the tree reflects the new
    /// names. The index files were just rewritten, so the cached index specs are
    /// rebuilt from disk too — otherwise the health check would compare the new
    /// tensor names against the old index keys and flag spurious mismatches.
    fn reload_after_rename(&mut self) -> Result<()> {
        let mut specs = Vec::with_capacity(self.index_specs.len());
        for spec in &self.index_specs {
            specs.push(crate::health::parse_index_spec(
                &spec.dir,
                &spec.index_path,
            )?);
        }
        self.index_specs = specs;
        self.health_reports.clear();
        let ((tensors, metadata, config, disk, health), checkpoint) =
            Self::gather_checkpoint(&self.files, self.remote_read())?;
        if let Some(rc) = self.remote.as_mut() {
            rc.disk = disk;
        }
        self.health_reports.extend(health);
        self.finalize_load(tensors, metadata, config, checkpoint);
        Ok(())
    }

    /// Repack the current HDF5 checkpoint into a new file: prompt for the output
    /// name, then run the conversion with a progress screen.
    fn repack_checkpoint(&self, term: &mut crate::tui::LiveTerminal) {
        let Some(input) = self.repack_input() else {
            let _ = term.draw(|f| {
                UI::render_message(
                    f,
                    "Repack unavailable",
                    "Repacking is available only for a single HDF5 checkpoint (.h5/.hdf5).",
                )
            });
            let _ = event::read();
            return;
        };
        let default = default_repacked_name(&input);
        let Some(output) = self.prompt_output_path(term, &default) else {
            return;
        };
        let Some(codec) = self.prompt_codec(term) else {
            return;
        };
        if !self.confirm_same_codec(term, &input, codec) {
            return;
        }
        let Some(buffer_bytes) = self.prompt_buffer(term) else {
            return;
        };
        self.run_repack(term, &input, &output, codec, buffer_bytes);
    }

    /// If the source already uses `codec`, ask whether to re-encode anyway
    /// (a plain copy would be equivalent). Returns `true` to proceed.
    #[cfg(feature = "hdf5")]
    fn confirm_same_codec(
        &self,
        term: &mut crate::tui::LiveTerminal,
        input: &Path,
        codec: crate::codec::Codec,
    ) -> bool {
        if crate::convert::source_codec(input) != Some(codec) {
            return true;
        }
        let title = format!("Source is already {} — re-encode it anyway?", codec.label());
        let mut idx = 0; // 0 = repack anyway, 1 = cancel
        loop {
            if term
                .draw(|f| UI::render_choice_menu(f, &title, &["Repack anyway", "Cancel"], idx))
                .is_err()
            {
                return false;
            }
            match event::read() {
                Ok(Event::Key(key)) if is_ctrl_c(&key) => quit_immediately(),
                Ok(Event::Key(KeyEvent { code, .. })) => match code {
                    KeyCode::Left | KeyCode::Right => idx = 1 - idx,
                    KeyCode::Enter => return idx == 0,
                    KeyCode::Esc => return false,
                    _ => {}
                },
                Ok(_) => {}
                Err(_) => return false,
            }
        }
    }

    #[cfg(not(feature = "hdf5"))]
    fn confirm_same_codec(
        &self,
        _term: &mut crate::tui::LiveTerminal,
        _input: &Path,
        _codec: crate::codec::Codec,
    ) -> bool {
        true
    }

    /// Pick the output compression codec from a menu. Returns `None` if cancelled.
    fn prompt_codec(&self, term: &mut crate::tui::LiveTerminal) -> Option<crate::codec::Codec> {
        use crate::codec::Codec;
        let codecs = [Codec::Gzip, Codec::Zstd, Codec::Lz4, Codec::Uncompressed];
        let labels: Vec<&str> = codecs.iter().map(|c| c.label()).collect();
        let mut idx = 0;
        loop {
            if term
                .draw(|f| UI::render_choice_menu(f, "Repack — compression codec", &labels, idx))
                .is_err()
            {
                return None;
            }
            match event::read() {
                Ok(Event::Key(key)) if is_ctrl_c(&key) => quit_immediately(),
                Ok(Event::Key(KeyEvent { code, .. })) => match code {
                    KeyCode::Right => idx = (idx + 1) % codecs.len(),
                    KeyCode::Left => idx = (idx + codecs.len() - 1) % codecs.len(),
                    KeyCode::Enter => return Some(codecs[idx]),
                    KeyCode::Esc => return None,
                    _ => {}
                },
                Ok(_) => {}
                Err(_) => return None,
            }
        }
    }

    /// Prompt for the streaming buffer size (e.g. `256M`, `1G`), pre-filled with
    /// a default. Returns the size in bytes, or `None` if cancelled.
    /// Prompt for the histogram bucket count, pre-filled with the current count.
    /// An empty entry returns to the automatic count; `Esc` leaves it unchanged.
    fn prompt_bins(
        &self,
        term: &mut crate::tui::LiveTerminal,
        background: impl Fn(&mut ratatui::Frame),
        current: Option<usize>,
    ) -> BinsChoice {
        let mut input = current.map(|n| n.to_string()).unwrap_or_default();
        let mut error: Option<String> = None;
        loop {
            if term
                .draw(|f| {
                    background(f);
                    UI::render_text_prompt(
                        f,
                        "Histogram bin count (1–512, empty for automatic)",
                        &input,
                        error.as_deref(),
                    );
                })
                .is_err()
            {
                return BinsChoice::Cancel;
            }
            match event::read() {
                Ok(Event::Key(key)) if is_ctrl_c(&key) => quit_immediately(),
                Ok(Event::Key(KeyEvent { code, .. })) => match code {
                    KeyCode::Enter => {
                        let t = input.trim();
                        if t.is_empty() {
                            return BinsChoice::Clear;
                        }
                        match t.parse::<usize>() {
                            Ok(n) if (1..=512).contains(&n) => return BinsChoice::Set(n),
                            Ok(_) => error = Some("Enter a count between 1 and 512.".to_string()),
                            Err(_) => {
                                error =
                                    Some("Enter a whole number (empty = automatic).".to_string())
                            }
                        }
                    }
                    KeyCode::Esc => return BinsChoice::Cancel,
                    KeyCode::Backspace => {
                        input.pop();
                        error = None;
                    }
                    KeyCode::Char(c) => {
                        input.push(c);
                        error = None;
                    }
                    _ => {}
                },
                Ok(_) => {}
                Err(_) => return BinsChoice::Cancel,
            }
        }
    }

    fn prompt_buffer(&self, term: &mut crate::tui::LiveTerminal) -> Option<usize> {
        let mut input = "256M".to_string();
        let mut error: Option<String> = None;
        loop {
            if term
                .draw(|f| {
                    UI::render_text_prompt(
                        f,
                        "Streaming buffer size (e.g. 64M, 256M, 1G)",
                        &input,
                        error.as_deref(),
                    )
                })
                .is_err()
            {
                return None;
            }
            match event::read() {
                Ok(Event::Key(key)) if is_ctrl_c(&key) => quit_immediately(),
                Ok(Event::Key(KeyEvent { code, .. })) => match code {
                    KeyCode::Enter => match crate::utils::parse_size(input.trim()) {
                        Ok(n) if n > 0 => return Some(n),
                        Ok(_) => error = Some("Buffer must be greater than zero.".to_string()),
                        Err(e) => error = Some(e),
                    },
                    KeyCode::Esc => return None,
                    KeyCode::Backspace => {
                        input.pop();
                        error = None;
                    }
                    KeyCode::Char(c) => {
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

    /// Prompt for the repack output path, pre-filled with `default`, rejecting an
    /// empty name or an existing file. Returns `None` if cancelled.
    fn prompt_output_path(
        &self,
        term: &mut crate::tui::LiveTerminal,
        default: &Path,
    ) -> Option<PathBuf> {
        let mut input = default.to_string_lossy().into_owned();
        let mut error: Option<String> = None;
        loop {
            if term
                .draw(|f| {
                    UI::render_text_prompt(
                        f,
                        "Save repacked checkpoint as",
                        &input,
                        error.as_deref(),
                    )
                })
                .is_err()
            {
                return None;
            }
            match event::read() {
                Ok(Event::Key(key)) if is_ctrl_c(&key) => quit_immediately(),
                Ok(Event::Key(KeyEvent { code, .. })) => match code {
                    KeyCode::Enter => {
                        let trimmed = input.trim();
                        if trimmed.is_empty() {
                            error = Some("Enter a file name.".to_string());
                        } else if Path::new(trimmed).exists() {
                            error =
                                Some("That file already exists — choose another name.".to_string());
                        } else {
                            return Some(PathBuf::from(trimmed));
                        }
                    }
                    KeyCode::Esc => return None,
                    KeyCode::Backspace => {
                        input.pop();
                        error = None;
                    }
                    KeyCode::Char(c) => {
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

    #[cfg(feature = "hdf5")]
    fn run_repack(
        &self,
        term: &mut crate::tui::LiveTerminal,
        input: &Path,
        output: &Path,
        codec: crate::codec::Codec,
        buffer: usize,
    ) {
        let level = codec.clamp_level(codec.default_level());
        let opts = crate::convert::Options {
            codec,
            level,
            buffer_bytes: buffer,
        };
        let title = format!("Repacking → {} ({})", output.display(), codec.label());
        let _ = term.draw(|f| UI::render_progress(f, &title, 0, 1, "starting…"));
        // The conversion drives this callback per dataset; redraw the progress bar
        // through the live terminal each step (`term` is borrowed for the duration).
        let result = crate::convert::convert_hdf5(input, output, &opts, |done, total, name| {
            let _ = term.draw(|f| UI::render_progress(f, &title, done, total, name));
        });
        let level_note = if codec.uses_level() {
            format!(", level {level}")
        } else {
            String::new()
        };
        let (heading, body) = match result {
            Ok(rep) => (
                "Repack complete",
                format!("{} → {}{level_note}", rep.summary(codec), output.display()),
            ),
            Err(e) => ("Repack failed", format!("{e:#}")),
        };
        let _ = term.draw(|f| UI::render_message(f, heading, &body));
        let _ = event::read();
    }

    #[cfg(not(feature = "hdf5"))]
    fn run_repack(
        &self,
        term: &mut crate::tui::LiveTerminal,
        _input: &Path,
        _output: &Path,
        _codec: crate::codec::Codec,
        _buffer: usize,
    ) {
        let _ = term.draw(|f| {
            UI::render_message(
                f,
                "Repack unavailable",
                "Rebuild with `--features hdf5` to enable repacking.",
            )
        });
        let _ = event::read();
    }

    /// Show the context-sensitive legend for the current screen (`l`), then wait
    /// for any key to dismiss it (Ctrl-C still quits). The caller's loop redraws
    /// its own screen over the overlay on the next iteration.
    ///
    /// `resume` is the pause flag of a background scan that the caller paused for
    /// this keypress (or `None`). The overlay does no file I/O, so we clear it to
    /// let the scan keep computing while the legend is up — its result is
    /// harvested when the caller redraws — instead of stalling it until dismissal.
    fn show_legend(
        &self,
        term: &mut crate::tui::LiveTerminal,
        legend: Legend,
        resume: Option<&AtomicBool>,
    ) {
        if let Some(pause) = resume {
            pause.store(false, Ordering::Relaxed);
        }
        // Float the legend band over a fresh tree frame (the band composites last
        // so the tree stays visible behind it), keeping the hover bubbles live.
        self.float_until_dismissed(term, |f| {
            self.render_tree_frame(f, true);
            UI::render_legend_band(f, legend);
        });
    }

    /// Run the health checks and float the report over the tree: `v` runs the
    /// value-tier scan (progress bar; `Esc` cancels), `y` copies the CLI command
    /// that reopens the popup, `c` copies the whole screen, `r` copies just the
    /// report; `Esc` or a click dismisses and Ctrl-C quits (other keys are
    /// ignored, a non-Latin one showing the wrong-layout hint).
    ///
    /// The structural checks are header-only and run up front; the `--values`
    /// scan reads every tensor's bytes on a background thread (local only).
    /// `expanded` is the initial fold state of the per-finding detail (`f` toggles
    /// it, like the stats popup's per-shard fold; `--health-findings` opens it
    /// expanded).
    fn show_check_report(&self, term: &mut crate::tui::LiveTerminal, mut expanded: bool) {
        use crate::ui::CheckPopup;

        // Reuse the report computed on a previous open — the checkpoint is
        // immutable, so the structural checks (an O(tensors) pass) don't change.
        // The first open computes them; on a big checkpoint that's a beat, so draw
        // an immediate "running checks" notice for feedback rather than freezing on
        // the tree.
        let mut report = self.cached_check.borrow().clone();
        if report.is_none() {
            let _ = term.draw(|f| {
                self.render_tree_frame(f, true);
                UI::render_notice(f, "Running health checks…");
            });
            let label = self
                .files
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            let computed = crate::check::run(
                label,
                self.tensors(),
                self.metadata(),
                &self.files,
                &self.health_reports,
                self.config(),
                &crate::filter::NameFilter::default(),
                false,
                1,
            );
            *self.cached_check.borrow_mut() = Some(computed.clone());
            report = Some(computed);
        }
        let mut report = report.expect("just computed or cached");

        // What was just copied (and when), so the footer can flash it briefly.
        let mut copied_at: Option<(std::time::Instant, &'static str)> = None;
        // A non-Latin key (wrong keyboard layout) shows a hint, as on other screens.
        let mut layout_hint: Option<char> = None;
        // First visible body row, and the last-rendered max scroll (set by each
        // draw, then used to clamp scroll input).
        let mut scroll = 0usize;
        let mut scroll_max;

        loop {
            let copied = copied_at
                .filter(|(t, _)| t.elapsed() < COPY_FLASH)
                .map(|(_, what)| what);
            // Offer the value scan while it can still add something: a local
            // source that hasn't been scanned yet.
            let can_scan = self.remote_read().is_none() && !report.values;
            let state = CheckPopup::Idle { copied, can_scan };
            let hint = layout_hint;
            let max_cell = std::cell::Cell::new(0usize);
            if term
                .draw(|f| {
                    self.render_tree_frame(f, true);
                    let (max, regions) =
                        UI::render_check_report(f, &report, state, scroll, expanded);
                    max_cell.set(max);
                    // The popup owns the clickable map while up (the fold toggle).
                    *self.clickable.borrow_mut() = regions;
                    if let Some(c) = hint {
                        UI::render_notice(f, &layout_hint_msg(c));
                    }
                })
                .is_err()
            {
                break;
            }
            // Clamp to what actually fit (the report / terminal size can change).
            scroll_max = max_cell.get();
            scroll = scroll.min(scroll_max);

            // While the copy flash is up, wake to clear it when it expires; else
            // block until the next event.
            let event = if copied.is_some() {
                let left =
                    COPY_FLASH.saturating_sub(copied_at.map_or(COPY_FLASH, |(t, _)| t.elapsed()));
                if event::poll(left).unwrap_or(false) {
                    event::read().ok()
                } else {
                    copied_at = None; // flash expired — redraw without it
                    continue;
                }
            } else {
                event::read().ok()
            };

            // Any input clears a prior layout hint; a fresh wrong-layout key re-sets
            // it in the `_` arm below.
            layout_hint = None;
            match event {
                Some(Event::Key(key)) => {
                    if is_ctrl_c(&key) {
                        quit_immediately();
                    }
                    let now = std::time::Instant::now();
                    match key.code {
                        // `v` runs the value scan on a worker; the popup animates a
                        // spinner + progress bar meanwhile, and `Esc` cancels.
                        KeyCode::Char('v') if can_scan => {
                            self.run_value_scan(term, &mut report, expanded);
                            // Keep the cache in step, so a re-open shows the scan.
                            *self.cached_check.borrow_mut() = Some(report.clone());
                            copied_at = None;
                        }
                        // `f` folds / unfolds the per-finding detail.
                        KeyCode::Char('f') => expanded = !expanded,
                        // `y` copies the command that reopens this popup from the CLI
                        // (the app-wide "copy command" convention), in its fold state.
                        KeyCode::Char('y') => {
                            let flag = if expanded {
                                "--health-findings"
                            } else {
                                "--health"
                            };
                            let cmd = format!("{} {flag}", self.command_for_tree());
                            if copy_to_clipboard(&cmd) {
                                copied_at = Some((now, "command"));
                            }
                        }
                        // `c` copies the whole screen — the tree with this popup
                        // composited on top (not just the tree behind it), rendered
                        // at the live terminal size so the popup lands where it does
                        // on screen (a fixed size would misplace the centred box).
                        KeyCode::Char('c') => {
                            let (w, h) = term
                                .size()
                                .map(|s| (s.width, s.height))
                                .unwrap_or((120, 40));
                            let screen = crate::tui::headless_render(w, h, |f| {
                                self.render_tree_frame(f, false);
                                UI::render_check_report(
                                    f,
                                    &report,
                                    CheckPopup::Idle {
                                        copied: None,
                                        can_scan,
                                    },
                                    scroll,
                                    expanded,
                                );
                            });
                            if let Ok(text) = screen
                                && copy_to_clipboard(&text)
                            {
                                copied_at = Some((now, "screen"));
                            }
                        }
                        // `r` copies only the report.
                        KeyCode::Char('r') => {
                            let text = report.render(false);
                            if copy_to_clipboard(&text) {
                                copied_at = Some((now, "report"));
                            }
                        }
                        // Scroll the body when the report is taller than the popup.
                        KeyCode::Up => scroll = scroll.saturating_sub(1),
                        KeyCode::Down => scroll = (scroll + 1).min(scroll_max),
                        KeyCode::PageUp => scroll = scroll.saturating_sub(SCROLL_PAGE),
                        KeyCode::PageDown => scroll = (scroll + SCROLL_PAGE).min(scroll_max),
                        KeyCode::Home => scroll = 0,
                        KeyCode::End => scroll = scroll_max,
                        // `Esc` dismisses; other keys are ignored (it's a popup, not
                        // a modal — a stray key shouldn't close it) — but a non-Latin
                        // key (wrong layout) gets the same hint as elsewhere.
                        KeyCode::Esc => break,
                        _ => layout_hint = wrong_layout_char(&key),
                    }
                }
                // A click on the fold toggle folds/unfolds; a click elsewhere
                // dismisses (the popup convention).
                Some(Event::Mouse(m)) if matches!(m.kind, MouseEventKind::Down(_)) => {
                    if crate::ui::region_hit(&self.clickable.borrow(), m.column, m.row).is_some() {
                        expanded = !expanded;
                    } else {
                        break;
                    }
                }
                // The wheel scrolls the report body.
                Some(Event::Mouse(m)) if matches!(m.kind, MouseEventKind::ScrollUp) => {
                    scroll = scroll.saturating_sub(WHEEL_STEP)
                }
                Some(Event::Mouse(m)) if matches!(m.kind, MouseEventKind::ScrollDown) => {
                    scroll = (scroll + WHEEL_STEP).min(scroll_max)
                }
                // Motion refreshes the hover bubbles behind the popup, so they
                // stay live; drag/resize are ignored.
                Some(Event::Mouse(m)) if matches!(m.kind, MouseEventKind::Moved) => {
                    if let Ok(sz) = term.size() {
                        self.update_hovers(HelpCtx::Tree, sz.width, sz.height, m.column, m.row);
                    }
                }
                Some(_) => {}
                None => break,
            }
        }
    }

    /// The checkpoint's true on-disk footprint for the stats popup. Remote reads
    /// captured it at load time (the session is gone now); local checkpoints are
    /// statted here — cheap, and it picks up any change since load.
    /// The S3 objects for the stats report's S3 section — converting the remote
    /// read's [`crate::remote::S3Meta`] into the stats module's own
    /// [`crate::stats::S3Stats`] (so `stats` stays free of a remote dependency).
    /// `None` for a local / SFTP checkpoint.
    fn s3_stats(&self) -> Option<crate::stats::S3Stats> {
        let meta = self.remote_s3_meta()?;
        let objects = meta
            .objects
            .iter()
            .map(|o| crate::stats::S3ObjectStat {
                key: o.key.clone(),
                size: o.size,
                etag: o.etag.clone(),
                checksum: o.checksum.clone(),
                last_modified: o.last_modified.clone(),
                tags: o.tags.as_ref().map(|t| t.len()),
                user_meta: o.user_meta.len(),
            })
            .collect();
        Some(crate::stats::S3Stats {
            objects,
            warnings: meta.warnings.clone(),
        })
    }

    /// The cached checkpoint model (owned by the kernel session), when loaded.
    fn checkpoint(&self) -> Option<&crate::model::Checkpoint> {
        self.session.as_ref().and_then(|s| s.model())
    }

    /// The canonical tensor list (deduped by name + natural-sorted) — owned by the
    /// kernel [`Session`](crate::kernel::Session), the single source. Empty before
    /// a load completes.
    fn tensors(&self) -> &[TensorInfo] {
        self.session.as_ref().map_or(&[], |s| s.tensors())
    }

    /// The checkpoint's metadata entries (model / shard order), from the session.
    fn metadata(&self) -> &[MetadataInfo] {
        self.session.as_ref().map_or(&[], |s| s.metadata())
    }

    /// The checkpoint's `config.json`, when present — from the session.
    fn config(&self) -> Option<&crate::config::ModelConfig> {
        self.session.as_ref().and_then(|s| s.config())
    }

    /// Total element (parameter) count across the canonical tensors.
    fn total_parameters(&self) -> usize {
        self.session.as_ref().map_or(0, |s| s.total_parameters())
    }

    fn disk_usage(&self) -> Option<crate::stats::DiskUsage> {
        if self.remote_read().is_some() {
            return self.remote_disk();
        }
        // Local: derived from the cached model's directory walk (symlink-followed
        // sizes) — no live `stat`. Falls back to a fresh stat only if the model
        // wasn't populated (shouldn't happen on a local load).
        if let Some(cp) = self.checkpoint() {
            return cp.disk_usage();
        }
        let mut paths: Vec<&str> = self
            .tensors()
            .iter()
            .map(|t| t.source_path.as_str())
            .collect();
        paths.sort_unstable();
        paths.dedup();
        crate::stats::DiskUsage::from_local(&paths)
    }

    /// Run the value-tier scan on a worker thread, animating the popup (spinner +
    /// progress bar) until it finishes or `Esc` cancels. On completion the result
    /// is folded into `report` (the "Value scan" row fills in); a cancelled scan
    /// leaves `report` untouched.
    fn run_value_scan(
        &self,
        term: &mut crate::tui::LiveTerminal,
        report: &mut crate::check::CheckReport,
        expanded: bool,
    ) {
        use crate::ui::CheckPopup;

        let tensors = self.tensors().to_vec();
        let metadata = self.metadata().to_vec();
        let progress = Arc::new(crate::progress::LoadProgress::new());
        let cancel = Arc::new(AtomicBool::new(false));
        let jobs = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let (tx, rx) = std::sync::mpsc::channel();
        {
            let (p, c) = (Arc::clone(&progress), Arc::clone(&cancel));
            std::thread::spawn(move || {
                let res = crate::check::scan_values(
                    &tensors,
                    &metadata,
                    &crate::filter::NameFilter::default(),
                    jobs,
                    &p,
                    &c,
                );
                let _ = tx.send(res);
            });
        }

        let mut frame = 0usize;
        let result = loop {
            let (done, total) = progress.snapshot();
            if term
                .draw(|f| {
                    self.render_tree_frame(f, true);
                    UI::render_check_report(
                        f,
                        report,
                        CheckPopup::Scanning { done, total, frame },
                        0,
                        expanded,
                    );
                })
                .is_err()
            {
                cancel.store(true, Ordering::Relaxed);
            }
            frame = frame.wrapping_add(1);
            match rx.try_recv() {
                Ok(res) => break Some(res),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => break None,
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
            if event::poll(std::time::Duration::from_millis(80)).unwrap_or(false)
                && let Ok(Event::Key(k)) = event::read()
            {
                if is_ctrl_c(&k) {
                    quit_immediately();
                }
                if matches!(k.code, KeyCode::Esc) {
                    cancel.store(true, Ordering::Relaxed);
                }
            }
        };

        // Incorporate the result unless the scan was cancelled.
        if !cancel.load(Ordering::Relaxed)
            && let Some(res) = result
        {
            report.results.push(res);
            report.values = true;
        }
    }

    /// The host to fold into an scp-style positional (`host:/path`) so the reopen
    /// command matches the shorthand launch — `Some` only for a remote checkpoint
    /// whose source is a plain path (an `s3://…` cstorch source still needs
    /// `--ssh-read`, so `None` there and for local checkpoints).
    fn remote_scp_host(&self) -> Option<String> {
        let remote = self.remote_read()?;
        let any_s3 = self
            .files
            .iter()
            .any(|f| f.to_string_lossy().starts_with("s3://"));
        (!any_s3).then(|| remote.host.clone())
    }

    /// The path argument(s) that reopen this checkpoint the way it was launched:
    /// a single file as-is, or — when every loaded file lives in one directory (a
    /// sharded checkpoint opened as a folder) — that directory, so the command
    /// references the checkpoint rather than an arbitrary shard; otherwise the
    /// individual files. A remote path source is rendered scp-style (`host:/path`)
    /// so it reopens via the shorthand.
    fn checkpoint_path_parts(&self) -> Vec<String> {
        let host = self.remote_scp_host();
        let render = |s: &str| -> String {
            match &host {
                Some(h) => shell_quote(&format!("{h}:{s}")),
                None => shell_quote(s),
            }
        };
        match self.files.as_slice() {
            [] => Vec::new(),
            [one] => vec![render(&one.to_string_lossy())],
            many => {
                let set: BTreeSet<String> = many
                    .iter()
                    .map(|p| p.to_string_lossy().into_owned())
                    .collect();
                match common_dir(&set) {
                    Some(dir) => vec![render(&dir)],
                    None => many.iter().map(|p| render(&p.to_string_lossy())).collect(),
                }
            }
        }
    }

    /// The program name plus, for an `s3://…` remote source, `--ssh-read HOST`
    /// (and `--ssh-venv` when non-default). A remote path source carries its host
    /// scp-style in the path arg instead (see [`Self::checkpoint_path_parts`]), so
    /// it needs no flag; local checkpoints get just the program name.
    fn command_prefix(&self) -> Vec<String> {
        let mut parts = vec![PROGRAM.to_string()];
        if let Some(remote) = self.remote_read()
            && self.remote_scp_host().is_none()
        {
            parts.push("--ssh-read".to_string());
            parts.push(shell_quote(&remote.host));
            if remote.venv != "~/venv" {
                parts.push("--ssh-venv".to_string());
                parts.push(shell_quote(&remote.venv));
            }
        }
        parts
    }

    /// The command that reopens the current tree: the program and the file/dir
    /// arguments it was launched with.
    fn command_for_tree(&self) -> String {
        let mut parts = self.command_prefix();
        parts.extend(self.checkpoint_path_parts());
        parts.extend(self.tree_state_args());
        parts.join(" ")
    }

    /// The command that reopens the file browser (`--files`): the launch path(s)
    /// plus the flag — so the file view's `y` round-trips like every other view.
    fn command_for_files(&self) -> String {
        let mut parts = self.command_prefix();
        parts.extend(self.checkpoint_path_parts());
        parts.push("--files".to_string());
        parts.join(" ")
    }

    /// `--tree-state` args reproducing the current bulk expansion (`expanded` /
    /// `collapsed`), or nothing for the default / a mixed (per-group) state. The
    /// shared tail for the tree's reopen commands so `E` / `C` round-trip.
    fn tree_state_args(&self) -> Vec<String> {
        let state = if TreeBuilder::all_groups(&self.tree_state.tree, true) {
            Some(TreeState::Expanded)
        } else if TreeBuilder::all_groups(&self.tree_state.tree, false) {
            Some(TreeState::Collapsed)
        } else {
            None
        };
        match state {
            Some(s) => vec!["--tree-state".to_string(), s.label().to_string()],
            None => Vec::new(),
        }
    }

    /// The command `y` copies from the tree: when a tensor row is highlighted
    /// (e.g. after backing out of its data view, which re-selects it), reproduce
    /// *the tree with that tensor revealed* — `--tree`, plus any active
    /// dtype/shape override — rather than opening its detail; for a group/root
    /// row, the plain file list.
    fn command_for_tree_selection(&self) -> String {
        match self.tree_state.flattened.get(self.tree_state.selected) {
            Some((TreeNode::Tensor { info, .. }, _)) => {
                let mut parts = self.command_base(info);
                parts.push("--tree".to_string());
                parts.extend(self.tree_state_args());
                parts.join(" ")
            }
            // A metadata row reopens the tree with that entry revealed.
            Some((TreeNode::Metadata { info }, _)) => {
                let mut parts = self.command_prefix();
                parts.extend(self.checkpoint_path_parts());
                parts.push("--metadata".to_string());
                parts.push(shell_quote(&info.name));
                parts.push("--tree".to_string());
                parts.extend(self.tree_state_args());
                parts.join(" ")
            }
            _ => self.command_for_tree(),
        }
    }

    /// `checkpoint-explorer <file-or-dir> --tensor <name>`, plus the active dtype
    /// and shape overrides — the shared prefix for the detail and data-view
    /// commands. Uses the checkpoint's launch path(s), not the tensor's specific
    /// shard, so a directory checkpoint reopens whole.
    fn command_base(&self, tensor: &TensorInfo) -> Vec<String> {
        let mut parts = self.command_prefix();
        parts.extend(self.checkpoint_path_parts());
        parts.push("--tensor".to_string());
        parts.push(shell_quote(&tensor.name));
        if let Some(dt) = self.data_view.dtype_overrides.borrow().get(&tensor.name)
            && let Some(value) = dt.cli_value()
        {
            parts.push("--dtype".to_string());
            parts.push(value);
        }
        if let Some(shape) = self.data_view.shape_overrides.borrow().get(&tensor.name) {
            parts.push("--shape".to_string());
            parts.push(
                shape
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            );
        }
        parts
    }

    /// The command that reopens this tensor's detail screen.
    fn command_for_detail(&self, tensor: &TensorInfo) -> String {
        let mut parts = self.command_base(tensor);
        let view = self.active_view(&tensor.name);
        // Reopen with the histogram showing when it's been computed for this view.
        let bins = self.data_view.histogram_bins.get();
        if self
            .histogram_cache
            .borrow()
            .contains_key(&(tensor.name.clone(), view, bins))
        {
            parts.push("--histogram".to_string());
            // Carry a custom bucket count so the histogram round-trips exactly.
            if let Some(n) = bins {
                parts.push("--bins".to_string());
                parts.push(n.to_string());
            }
        }
        // Reopen with the statistics showing when they've been computed for this
        // view (the histogram already brings up the ones it needs for its range).
        if self.cached_stats(tensor, view).is_some() {
            parts.push("--compute-stats".to_string());
        }
        parts.join(" ")
    }

    /// The command that reopens this data view with its current representation,
    /// layout, zebra striping and slice.
    fn command_for_data(&self, tensor: &TensorInfo, repr: Representation, slice: usize) -> String {
        let mut parts = self.command_base(tensor);
        parts.push(
            match repr {
                Representation::Heatmap => "--heatmap",
                Representation::Values => "--values",
            }
            .to_string(),
        );
        // The layout, with its position: the window's top-left corner and the
        // edges head/tail split, so the command reopens the same view — not just
        // the same layout at its default position. The bare flag (no value) is
        // emitted at the default position to keep the command tidy.
        parts.push(match self.data_view.data_view_layout.get() {
            DataLayout::Overview => "--overview".to_string(),
            DataLayout::Edges => {
                let (rt, ct) = (
                    self.data_view.data_view_row_tail.get(),
                    self.data_view.data_view_col_tail.get(),
                );
                if rt == 0.5 && ct == 0.5 {
                    "--edge".to_string()
                } else {
                    format!("--edge={rt},{ct}")
                }
            }
            DataLayout::Window => {
                let (row, col) = (
                    self.data_view.data_view_win_row.get(),
                    self.data_view.data_view_win_col.get(),
                );
                if row == 0 && col == 0 {
                    "--window".to_string()
                } else {
                    format!("--window={row},{col}")
                }
            }
        });
        // Zebra applies only to the numeric grid; emit it only when it differs
        // from the default (rows), which a fresh launch already uses.
        if matches!(repr, Representation::Values) {
            let zebra = match self.data_view.data_view_stripe.get() {
                StripeMode::Rows => None,
                StripeMode::Cols => Some("cols"),
                StripeMode::Off => Some("off"),
            };
            if let Some(mode) = zebra {
                parts.push("--zebra".to_string());
                parts.push(mode.to_string());
            }
            // Base applies only to the numeric grid; emit it only when it differs
            // from the default (decimal).
            let base = self.data_view.data_view_base.get();
            if base != NumBase::Decimal {
                parts.push("--base".to_string());
                parts.push(base.label().to_string());
            }
        }
        // Slice 0 is the default, so only name a non-zero starting slice.
        if slice > 0 {
            parts.push("--slice".to_string());
            parts.push(slice.to_string());
        }
        parts.join(" ")
    }
}

/// Default output name for a repack: `<stem>.repacked.<ext>` beside the input.
fn default_repacked_name(input: &Path) -> PathBuf {
    let ext = input.extension().and_then(|e| e.to_str()).unwrap_or("h5");
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("checkpoint");
    input.with_file_name(format!("{stem}.repacked.{ext}"))
}

/// Which screen the dtype menu re-renders as its live preview.
#[derive(Clone, Copy)]
enum DtypePreview {
    Detail,
    Data {
        repr: Representation,
        slice: usize,
        mode: SampleMode,
    },
}

/// One source→new-name rule in the rename editor. The source is a concrete tensor
/// name (picked via autocomplete); the new name is its `{layer}`/`{expert}`/`{n0}`
/// *schema*, edited so the whole family renames at once.
#[derive(Default)]
struct RenamePair {
    source: String,
    target: String,
}

/// The in-place rename editor's live state ([`Screen::Rename`]): a dynamic list of
/// [`RenamePair`]s (grown with `↓`/`^N`), which field has focus, the caret position
/// in it, the preview scroll, and the confirm gate. See [`Explorer::run_rename`].
struct RenameMode {
    pairs: Vec<RenamePair>,
    /// Index of the pair whose field has focus.
    focus_pair: usize,
    /// Which field of `focus_pair` has focus: `false` = source, `true` = new-name.
    on_target: bool,
    /// Caret position (char index) within the focused field.
    cursor: usize,
    /// Scroll offset of the preview pane.
    scroll: usize,
    /// The autocomplete dropdown: `Some(i)` when open with candidate `i` highlighted,
    /// `None` when closed. Opens on typing, closes on accept / focus change / `Esc`.
    menu: Option<usize>,
    /// A transient error (a failed apply / rule synthesis) shown in the editor.
    error: Option<String>,
}

impl Default for RenameMode {
    fn default() -> Self {
        Self {
            pairs: vec![RenamePair::default()],
            focus_pair: 0,
            on_target: false,
            cursor: 0,
            scroll: 0,
            menu: None,
            error: None,
        }
    }
}

impl RenameMode {
    /// The focused field, mutably.
    fn field(&mut self) -> &mut String {
        let pair = &mut self.pairs[self.focus_pair];
        if self.on_target {
            &mut pair.target
        } else {
            &mut pair.source
        }
    }

    /// The focused field's text.
    fn field_ref(&self) -> &str {
        let pair = &self.pairs[self.focus_pair];
        if self.on_target {
            &pair.target
        } else {
            &pair.source
        }
    }

    /// Snap the caret to the end of the focused field (after a focus change).
    fn caret_to_end(&mut self) {
        self.cursor = self.field_ref().chars().count();
    }

    /// Insert a character at the caret and advance past it.
    fn insert_char(&mut self, c: char) {
        let cur = self.cursor;
        {
            let f = self.field();
            let byte = f.char_indices().nth(cur).map(|(b, _)| b).unwrap_or(f.len());
            f.insert(byte, c);
        }
        self.cursor += 1;
        self.menu = Some(0); // typing (re)opens the dropdown at its top match
    }

    /// Delete the character before the caret (Backspace).
    fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let cur = self.cursor;
        {
            let f = self.field();
            if let Some((byte, _)) = f.char_indices().nth(cur - 1) {
                f.remove(byte);
            }
        }
        self.cursor -= 1;
        self.menu = Some(0);
    }

    /// Delete the character at the caret (Delete).
    fn delete(&mut self) {
        let cur = self.cursor;
        {
            let f = self.field();
            if let Some((byte, _)) = f.char_indices().nth(cur) {
                f.remove(byte);
            }
        }
        self.menu = Some(0);
    }

    fn left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }
    fn right(&mut self) {
        self.cursor = (self.cursor + 1).min(self.field_ref().chars().count());
    }

    /// Autocomplete candidates for the *focused* field (source or new-name alike):
    /// the deduped generalized schemas (numbers → `{layer}`/`{expert}`/`{n0}`) that
    /// match the query number-agnostically, each with the count of tensors it covers
    /// and — for a literal match — the char range to embolden. An empty query lists
    /// the first few for discovery.
    fn completions(&self, schemas: &[(String, usize)]) -> Vec<crate::ui::RenameCompletion> {
        const MAX: usize = 8;
        let raw = self.field_ref().trim();
        let q = normalize_for_match(raw);
        let ql = raw.to_lowercase();
        let len = raw.chars().count();
        schemas
            .iter()
            .filter(|(s, _)| q.is_empty() || normalize_for_match(s).contains(&q))
            .take(MAX)
            .map(|(s, count)| {
                // Best-effort literal-substring highlight (skipped when the query
                // only matches number-agnostically, e.g. `layers.5` vs `{layer}`).
                let hl = (!ql.is_empty())
                    .then(|| s.to_lowercase().find(&ql))
                    .flatten()
                    .map(|b| {
                        let start = s.to_lowercase()[..b].chars().count();
                        (start, start + len)
                    });
                crate::ui::RenameCompletion {
                    text: s.clone(),
                    count: *count,
                    hl,
                }
            })
            .collect()
    }

    /// Open the dropdown at its top match.
    fn open_menu(&mut self) {
        self.menu = Some(0);
    }

    /// Close the dropdown.
    fn close_menu(&mut self) {
        self.menu = None;
    }

    /// Move the dropdown's highlight by `delta`, wrapping over `n` candidates (as
    /// prompt_toolkit / pgcli do). A no-op when the menu is closed or empty.
    fn menu_move(&mut self, delta: isize, n: usize) {
        if n == 0 {
            return;
        }
        let cur = self.menu.unwrap_or(0) as isize;
        self.menu = Some((cur + delta).rem_euclid(n as isize) as usize);
    }

    /// `Tab`: extend the focused field to the longest common prefix of every schema
    /// that matches it (shell-style — fill in as much as is unambiguous), leaving the
    /// dropdown open so ↑/↓ + `Enter` can pick from what's left. Distinct from
    /// `accept`, so Tab and Enter don't do the same thing. A no-op when the prefix
    /// wouldn't grow the field, or wouldn't keep the typed text (which would broaden
    /// the match rather than narrow it, since matching is by substring).
    fn complete_prefix(&mut self, schemas: &[(String, usize)]) {
        let q = self.field_ref().trim().to_string();
        let nq = normalize_for_match(&q);
        // The common prefix over *all* matches (uncapped, unlike `completions`).
        let matches: Vec<&str> = schemas
            .iter()
            .filter(|(s, _)| nq.is_empty() || normalize_for_match(s).contains(&nq))
            .map(|(s, _)| s.as_str())
            .collect();
        let prefix = longest_common_prefix(&matches);
        if prefix.chars().count() > q.chars().count() && (q.is_empty() || prefix.contains(&q)) {
            *self.field() = prefix;
            self.caret_to_end();
            self.menu = Some(0);
        }
    }

    /// Accept the highlighted candidate into the focused field: numbers stay
    /// generalized to `{layer}`/`{expert}` placeholders so one rule covers a whole
    /// family. Accepting into a source also prefills a still-empty new name from it.
    fn accept(&mut self, schemas: &[(String, usize)]) {
        let Some(sel) = self.menu else { return };
        let cands = self.completions(schemas);
        let Some(text) = cands.get(sel).map(|c| c.text.clone()) else {
            return;
        };
        let on_target = self.on_target;
        let pair = &mut self.pairs[self.focus_pair];
        if on_target {
            pair.target = text;
        } else {
            pair.source = text.clone();
            if pair.target.trim().is_empty() {
                pair.target = text;
            }
        }
        self.caret_to_end();
        self.menu = None; // accepting a completion closes the dropdown
    }

    /// Build the combined rename map from every complete pair (via
    /// [`rule_from_fields`](crate::rename::rule_from_fields)), plus notes about
    /// half-filled pairs. `Err` only on a rule-synthesis error (e.g. a bad
    /// placeholder), which the caller shows inline.
    fn build_map(&self) -> std::result::Result<(crate::diff::NameMap, Vec<String>), String> {
        let mut rules = Vec::new();
        let mut notes = Vec::new();
        for (i, p) in self.pairs.iter().enumerate() {
            match (p.source.trim().is_empty(), p.target.trim().is_empty()) {
                (true, true) => {} // blank pair — ignored
                (false, false) => {
                    rules.push(crate::rename::rule_from_fields(&p.source, &p.target)?)
                }
                _ => notes.push(format!(
                    "rule {}: fill both the source and the new name",
                    i + 1
                )),
            }
        }
        let map = crate::diff::NameMap::from_pairs(rules).map_err(|e| format!("{e:#}"))?;
        Ok((map, notes))
    }

    /// Append a fresh empty pair and focus its source.
    fn add_pair(&mut self) {
        self.pairs.push(RenamePair::default());
        self.focus_pair = self.pairs.len() - 1;
        self.on_target = false;
        self.caret_to_end();
        self.menu = None;
    }

    /// Remove the focused pair (keeping at least one), clamping focus.
    fn remove_pair(&mut self) {
        if self.pairs.len() > 1 {
            self.pairs.remove(self.focus_pair);
            self.focus_pair = self.focus_pair.min(self.pairs.len() - 1);
            self.on_target = false;
            self.caret_to_end();
            self.menu = None;
        }
    }

    /// After an edit empties a rule entirely (both fields blank), drop it and move to
    /// a neighbour — the end of the *previous* rule, or the *next* rule when the first
    /// is the one removed — so backspacing a just-added rule out of existence Just
    /// Works. Keeps at least one (blank) rule.
    fn remove_pair_if_empty(&mut self) {
        let p = &self.pairs[self.focus_pair];
        if !(p.source.trim().is_empty() && p.target.trim().is_empty()) || self.pairs.len() <= 1 {
            return;
        }
        let removing = self.focus_pair;
        self.pairs.remove(removing);
        if removing == 0 {
            // No previous rule — land on the new first rule's source.
            self.focus_pair = 0;
            self.on_target = false;
        } else {
            // Land at the end of the previous rule (its new-name field).
            self.focus_pair = removing - 1;
            self.on_target = true;
        }
        self.caret_to_end();
        self.menu = None;
    }

    /// The focused field's flat index (source = even, new-name = odd) — the order
    /// `↑`/`↓` step through.
    fn field_index(&self) -> usize {
        self.focus_pair * 2 + usize::from(self.on_target)
    }

    fn set_field_index(&mut self, idx: usize) {
        self.focus_pair = idx / 2;
        self.on_target = idx % 2 == 1;
        self.ensure_target_prefill();
        self.caret_to_end();
        self.menu = None; // don't pop the dropdown just by moving focus
    }

    /// `↑`: focus the previous field.
    fn focus_up(&mut self) {
        let i = self.field_index();
        if i > 0 {
            self.set_field_index(i - 1);
        }
    }

    /// `↓`: focus the next field, growing a new pair when stepping past the last.
    fn focus_down(&mut self) {
        let i = self.field_index();
        if i + 1 < self.pairs.len() * 2 {
            self.set_field_index(i + 1);
        } else {
            self.add_pair();
        }
    }

    /// When focus lands on an empty new-name field, prefill it with a copy of the
    /// source, so the user edits a copy (placeholders and concrete numbers kept).
    fn ensure_target_prefill(&mut self) {
        if self.on_target {
            let pair = &mut self.pairs[self.focus_pair];
            if pair.target.trim().is_empty() && !pair.source.trim().is_empty() {
                pair.target = pair.source.clone();
            }
        }
    }
}

/// The longest common prefix (by character) shared by every string in `items` —
/// what `Tab` extends a rename field to. Empty for an empty list.
fn longest_common_prefix(items: &[&str]) -> String {
    let Some((first, rest)) = items.split_first() else {
        return String::new();
    };
    let mut end = first.chars().count();
    for s in rest {
        let common = first
            .chars()
            .zip(s.chars())
            .take_while(|(a, b)| a == b)
            .count();
        end = end.min(common);
        if end == 0 {
            break;
        }
    }
    first.chars().take(end).collect()
}

/// Normalize a name / query for the rename autocomplete's number-agnostic match:
/// each `{token}` placeholder and each run of digits collapses to `#`, lowercased —
/// so typing `layers.5.q` matches the `layers.{layer}.…q…` family.
fn normalize_for_match(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '{'
            && let Some(rel) = chars[i + 1..].iter().position(|&c| c == '}')
        {
            out.push('#');
            i += 1 + rel + 1;
        } else if chars[i].is_ascii_digit() {
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            out.push('#');
        } else {
            out.push(chars[i].to_ascii_lowercase());
            i += 1;
        }
    }
    out
}

/// The directory shared by all `paths`, or `None` if they don't all share one.
/// The directory the file browser lists for a checkpoint launched as `files`:
/// the common parent of its shards (a directory checkpoint / sharded model), or
/// a single file's parent. Falls back to `.` when there's nothing to anchor to
/// (e.g. an empty list or a bare relative filename with no parent).
fn browse_root_of(files: &[PathBuf]) -> PathBuf {
    match files {
        [] => PathBuf::from("."),
        [one] => one
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf),
        many => {
            let set: BTreeSet<String> = many
                .iter()
                .map(|p| p.to_string_lossy().into_owned())
                .collect();
            common_dir(&set)
                .map(PathBuf::from)
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or_else(|| PathBuf::from("."))
        }
    }
}

/// The root label for the s3-native browse tree: the last non-empty component of
/// the URI's prefix (a `bucket/…/ckpt` → `ckpt`), or the bucket when the prefix is
/// empty. A short, meaningful heading; the full URI is still shown in the header.
fn s3_root_label(uri: &str) -> String {
    let rest = uri.strip_prefix("s3://").unwrap_or(uri);
    rest.trim_end_matches('/')
        .rsplit('/')
        .find(|s| !s.is_empty())
        .unwrap_or(rest)
        .to_string()
}

/// The `↑↓ … scroll · Esc / click to close` footer shared by the file view's
/// pop-ups (sidecar preview, legend, info).
fn files_dismiss_footer() -> Line<'static> {
    Line::from(crate::ui::dim_span(
        "↑↓ PgUp/PgDn scroll · Esc / click to close",
    ))
}

/// Read up to `cap` bytes of `path` as UTF-8, returning `(text, truncated)`.
/// A non-UTF-8 file yields an error message the caller shows in an info pop-up.
fn read_text_capped(path: &Path, cap: u64) -> Result<(String, bool), String> {
    let len = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let f = File::open(path).map_err(|e| e.to_string())?;
    let mut buf = Vec::new();
    f.take(cap)
        .read_to_end(&mut buf)
        .map_err(|e| e.to_string())?;
    let text =
        String::from_utf8(buf).map_err(|_| "Binary (non-UTF-8) file — no preview.".to_string())?;
    Ok((text, len > cap))
}

/// The layout map's legend (glyphs + a one-paragraph explainer of the safetensors
/// on-disk format), floated over the strip by `l`.
fn layout_legend_lines() -> Vec<Line<'static>> {
    vec![
        Line::from(crate::ui::dim_span(
            "How a .safetensors file is laid out on disk:",
        )),
        Line::default(),
        Line::from(Span::raw(
            "█ header    8-byte length + JSON metadata (dtype, shape, offsets)".to_string(),
        )),
        Line::from(Span::raw(
            "█ ▓ ▒ ░     one band per tensor, in file-offset order".to_string(),
        )),
        Line::from(crate::ui::dim_span(
            "            the fuller the block, the larger the tensor",
        )),
        Line::from(Span::raw(
            "░ padding   an unaccounted gap (alignment)".to_string(),
        )),
        Line::default(),
        Line::from(crate::ui::dim_span(
            "Each band's height ∝ its share of the file; offsets are absolute bytes.",
        )),
    ]
}

/// The layout map as plain text (offset, size, dtype/shape, name per segment) —
/// what `c` copies from the layout view.
fn layout_to_text(map: &crate::safelayout::LayoutMap) -> String {
    use crate::utils::{format_shape, format_size};
    let mut out = format!(
        "{}\n{} · {} tensors · header {}\n\n",
        map.name,
        format_size(map.total_len as usize),
        map.tensor_count,
        format_size(map.header_len as usize),
    );
    for s in &map.segments {
        let detail = match &s.kind {
            crate::safelayout::SegmentKind::Tensor { dtype, shape } => {
                format!("  {dtype} {}", format_shape(shape))
            }
            _ => String::new(),
        };
        out.push_str(&format!(
            "{:#014x}  {:>10}  {}{}\n",
            s.start,
            format_size(s.len() as usize),
            s.name,
            detail
        ));
    }
    out
}

/// Build a sidecar preview's lines: JSON syntax-highlighted (falling back to
/// plain text when it doesn't parse), any other text plain.
fn preview_lines(text: &str, kind: crate::filetree::FileKind) -> Vec<Line<'static>> {
    if kind == crate::filetree::FileKind::Json
        && let Some(lines) = crate::ui::highlight_json_lines(text)
    {
        return lines;
    }
    text.lines()
        .map(|l| Line::from(Span::raw(l.to_string())))
        .collect()
}

fn common_dir(paths: &BTreeSet<String>) -> Option<String> {
    let mut dirs = paths.iter().map(|p| {
        Path::new(p)
            .parent()
            .map(|d| d.to_string_lossy().into_owned())
    });
    let first = dirs.next().flatten()?;
    if dirs.all(|d| d.as_deref() == Some(first.as_str())) {
        Some(first)
    } else {
        None
    }
}

/// Parse a shape-override entry — dimensions separated by `,`, space, or `x`
/// (e.g. `10, 100` / `10x100`) — validating that the product equals `elements`
/// (the tensor's element count). One dimension may be a wildcard (`-1`, `*`, or
/// `_`), inferred from the count (like NumPy's `reshape(-1, …)`).
fn parse_shape_input(input: &str, elements: usize) -> Result<Vec<usize>, String> {
    let tokens: Vec<&str> = input
        .split(|c: char| c == ',' || c == 'x' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .collect();
    if tokens.is_empty() {
        return Err("enter one or more dimensions".to_string());
    }
    let mut dims: Vec<Option<usize>> = Vec::with_capacity(tokens.len());
    let mut wildcard: Option<usize> = None;
    for tok in &tokens {
        if matches!(*tok, "*" | "-1" | "_") {
            if wildcard.is_some() {
                return Err("only one inferred dimension (`-1`, `*`, `_`) is allowed".to_string());
            }
            wildcard = Some(dims.len());
            dims.push(None);
        } else {
            let d = tok
                .parse::<usize>()
                .map_err(|_| format!("'{tok}' is not a whole number"))?;
            if d == 0 {
                return Err("dimensions must be non-zero".to_string());
            }
            dims.push(Some(d));
        }
    }
    // Product of the explicitly-given dimensions.
    let known: usize = dims.iter().flatten().product();
    if let Some(w) = wildcard {
        if known == 0 || !elements.is_multiple_of(known) {
            return Err(format!(
                "can't infer a whole dimension for {elements} elements"
            ));
        }
        dims[w] = Some(elements / known);
    }
    let resolved: Vec<usize> = dims.into_iter().map(Option::unwrap).collect();
    let product: usize = resolved.iter().product();
    if product != elements {
        return Err(format!("{product} elements, but the tensor has {elements}"));
    }
    Ok(resolved)
}

/// Whether a tensor's dtype can be reinterpreted — formats whose raw stored
/// bytes we read ourselves (safetensors, NumPy, HDF5).
fn dtype_overridable(tensor: &TensorInfo) -> bool {
    matches!(
        std::path::Path::new(&tensor.source_path)
            .extension()
            .and_then(|e| e.to_str()),
        Some("safetensors" | "h5" | "hdf5" | "npy" | "npz")
    )
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
        let pct_str = pct_str.trim();
        let pct: f64 = pct_str
            .parse()
            .map_err(|_| format!("'{pct_str}' is not a number — write a percentage like 50%"))?;
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

/// A keypress that looks like the wrong keyboard layout: a plain non-ASCII *letter*
/// (e.g. Cyrillic/Greek produced by a non-Latin layout when the user meant a Latin
/// shortcut like `m`/`v`/`l`), with no Ctrl/Alt. Such a key can never match a
/// shortcut, so the loops surface a hint instead of silently doing nothing (or, on
/// the detail/data screens, treating it as "any other key" and navigating away).
/// Returns the character, to show it in the hint. Only meaningful outside text
/// input (search / prompts), where a non-ASCII character is legitimate.
fn wrong_layout_char(key: &KeyEvent) -> Option<char> {
    if key
        .modifiers
        .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
    {
        return None;
    }
    match key.code {
        KeyCode::Char(c) if !c.is_ascii() && c.is_alphabetic() => Some(c),
        _ => None,
    }
}

/// The bottom-line hint shown when [`wrong_layout_char`] fires.
fn layout_hint_msg(c: char) -> String {
    format!("⚠ '{c}' is not a shortcut — a non-US/Latin keyboard layout may be active")
}

/// Restore the terminal (leave raw mode, show the cursor) and exit the process
/// immediately, leaving the last frame on screen with the prompt just below it.
/// Used for Ctrl-C from any of the detail/data sub-screens so it quits outright
/// instead of stepping back one screen.
fn quit_immediately() -> ! {
    let mut stdout = io::stdout();
    // Clear below the cursor so no frame content lingers under the prompt (e.g.
    // the rows beneath a mid-screen overlay like the `y` command pop-up), then
    // drop the prompt onto a fresh line below the preserved frame.
    let _ = execute!(
        stdout,
        crossterm::event::DisableMouseCapture,
        terminal::Clear(ClearType::FromCursorDown),
        cursor::Show
    );
    let _ = terminal::disable_raw_mode();
    println!();
    std::process::exit(0);
}

/// Block until a key is pressed or the mouse is clicked, then dismiss. Mouse
/// motion / drag / wheel and resize are ignored, so a modifier-drag to select the
/// text (e.g. iTerm2 Option-drag, which bypasses capture entirely) doesn't close
/// the pop-up. Ctrl-C quits the app. Used by the "any key to dismiss" pop-ups.
fn wait_for_dismiss() {
    loop {
        match event::read() {
            Ok(Event::Key(key)) => {
                if is_ctrl_c(&key) {
                    quit_immediately();
                }
                return;
            }
            Ok(Event::Mouse(m)) if matches!(m.kind, MouseEventKind::Down(_)) => return,
            _ => {} // mouse motion / drag / wheel, resize: keep waiting
        }
    }
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
        TreeNode::Tensor { info, .. } => {
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

/// Write `out` (plus a trailing newline) to stdout, treating a closed pipe
/// (`| head`, `| grep -q`) as a normal, quiet exit rather than a panic — the
/// one-shot `--print-*` exports are meant to be piped.
fn emit_stdout(out: &str) -> Result<()> {
    use std::io::Write;
    let mut stdout = io::stdout();
    match stdout
        .write_all(out.as_bytes())
        .and_then(|()| stdout.write_all(b"\n"))
        .and_then(|()| stdout.flush())
    {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => std::process::exit(0),
        Err(e) => Err(e.into()),
    }
}

/// The final path component of a tensor's `source_path` (its shard file), for
/// the JSON `weight_map` and verbose listings. Works for local, scp-style
/// (`host:/…`) and `s3://` paths; falls back to the whole string if it ends in
/// no component (e.g. a trailing slash).
fn file_basename(source_path: &str) -> &str {
    source_path
        .rsplit(['/', '\\'])
        .next()
        .filter(|s| !s.is_empty())
        .unwrap_or(source_path)
}

/// The per-tensor detail object shared by the JSON exports: dtype, shape,
/// element count and logical byte size.
fn tensor_facts(t: &TensorInfo) -> serde_json::Value {
    let mut facts = serde_json::json!({
        "dtype": t.dtype,
        "shape": t.shape,
        "num_elements": t.num_elements,
        "size_bytes": t.size_bytes,
    });
    // For a compressed tensor (HDF5), also report the codec and on-disk size.
    if let Storage::Compressed {
        codec,
        stored_bytes,
    } = &t.storage
    {
        facts["codec"] = serde_json::Value::String(codec.clone());
        facts["stored_bytes"] = serde_json::json!(stored_bytes);
    }
    facts
}

/// Quote `s` as a single shell argument: left bare when it's only made of safe
/// characters (so plain tensor names and paths stay readable), else wrapped in
/// single quotes with any embedded quote escaped. Used to build copyable CLI
/// commands that survive paths/names containing spaces or shell metacharacters.
fn shell_quote(s: &str) -> String {
    let safe = !s.is_empty()
        && s.bytes().all(|b| {
            b.is_ascii_alphanumeric()
                || matches!(
                    b,
                    b'_' | b'-' | b'.' | b'/' | b'=' | b',' | b'%' | b'+' | b':' | b'@'
                )
        });
    if safe {
        s.to_string()
    } else {
        format!("'{}'", s.replace('\'', r"'\''"))
    }
}

/// Largest OSC 52 base64 payload we'll emit. Past this, terminals and tmux tend
/// to reject or truncate the sequence and spill the base64 into the display as
/// text — so we refuse rather than corrupt the screen, offering `--print-tree`
/// instead. Calibrated from a real ~30B checkpoint whose tree copies fine
/// (~186 KiB of base64); 1 MiB leaves generous headroom for much larger models
/// while still catching a pathological, terminal-breaking payload.
const OSC52_MAX_B64: usize = 1 << 20; // 1 MiB

/// Copy `text` to the clipboard via OSC 52 (reaches the local clipboard even
/// over SSH/tmux). Returns `false` — emitting nothing — when the encoded payload
/// exceeds [`OSC52_MAX_B64`], since a terminal that can't take it would otherwise
/// dump the raw base64 on screen. Callers can then offer a fallback. (All copies
/// but the whole-tree `t` are bounded by the viewport and comfortably fit.)
fn copy_to_clipboard(text: &str) -> bool {
    let b64 = base64_encode(text.as_bytes());
    if b64.len() > OSC52_MAX_B64 {
        return false;
    }
    let mut stdout = io::stdout();
    let _ = write!(stdout, "\x1b]52;c;{b64}\x07");
    let _ = stdout.flush();
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn browse_root_is_the_checkpoints_directory() {
        // A single file browses its parent directory.
        assert_eq!(
            browse_root_of(&[PathBuf::from("/models/m/model.safetensors")]),
            PathBuf::from("/models/m")
        );
        // Sharded files in one directory browse that common directory.
        assert_eq!(
            browse_root_of(&[
                PathBuf::from("/models/m/model-00001.safetensors"),
                PathBuf::from("/models/m/model-00002.safetensors"),
            ]),
            PathBuf::from("/models/m")
        );
        // A bare filename with no parent falls back to the current directory.
        assert_eq!(
            browse_root_of(&[PathBuf::from("model.safetensors")]),
            PathBuf::from(".")
        );
        assert_eq!(browse_root_of(&[]), PathBuf::from("."));
    }

    #[test]
    fn checkpoint_dir_is_the_directory_not_a_shard() {
        // `f` on the tree root copies the checkpoint directory — even when only one
        // shard is loaded, it must not yield that shard's file path.
        let e = Explorer::new(
            vec![PathBuf::from(
                "/cb/home/antont/ws/Qwen3-Coder-30B-A3B/model-00016-of-00016.safetensors",
            )],
            Vec::new(),
            None,
            false,
        );
        assert_eq!(e.checkpoint_dir(), "/cb/home/antont/ws/Qwen3-Coder-30B-A3B");

        // Remote SFTP: scp-style `host:/dir` (usable with scp/rsync).
        let mut e = Explorer::new(
            vec![PathBuf::from("/opt/models/ckpt")],
            Vec::new(),
            None,
            false,
        );
        e.set_remote_read("net004".into(), "~/venv".into());
        assert_eq!(e.checkpoint_dir(), "net004:/opt/models/ckpt");

        // Remote s3: the prefix URI is already the "directory".
        let mut e = Explorer::new(
            vec![PathBuf::from("s3://bucket/ckpt")],
            Vec::new(),
            None,
            false,
        );
        e.set_remote_read("host".into(), "~/venv".into());
        assert_eq!(e.checkpoint_dir(), "s3://bucket/ckpt");
    }

    #[test]
    fn status_bar_and_copy_path_share_one_source_for_the_root() {
        use crate::tree::{Layout, Storage, TensorInfo};
        let ti = |name: &str, shard: &str| TensorInfo {
            name: name.into(),
            dtype: "F32".into(),
            shape: vec![2],
            size_bytes: 8,
            num_elements: 2,
            storage: Storage::Unknown,
            source_path: format!("/ckpt/{shard}"),
            layout: Layout::None,
        };

        // A single-shard open: the root's status bar and `f` copy must BOTH show
        // the checkpoint directory, never the shard (the reported bug).
        let mut e = Explorer::new(
            vec![PathBuf::from("/ckpt/model-00016-of-00016.safetensors")],
            Vec::new(),
            None,
            false,
        );
        e.finalize_load(
            vec![ti("lm_head.weight", "model-00016-of-00016.safetensors")],
            Vec::new(),
            None,
            None,
        );
        e.tree_state.selected = 0; // the root row
        let view = e.describe_selection();
        assert_eq!(view.copy_path.as_deref(), Some("/ckpt"));
        assert_eq!(view.status.1, "/ckpt"); // status bar shows the dir, not the shard
        assert!(
            !view.status.1.contains(".safetensors"),
            "status: {:?}",
            view.status
        );

        // A two-shard directory open: still the directory, with a count prefix —
        // and the copied dir is a substring of what the status bar shows (they come
        // from the one `describe_selection`, so they can't disagree).
        let mut e = Explorer::new(
            vec![
                PathBuf::from("/ckpt/model-00001-of-00002.safetensors"),
                PathBuf::from("/ckpt/model-00002-of-00002.safetensors"),
            ],
            Vec::new(),
            None,
            false,
        );
        e.finalize_load(
            vec![
                ti(
                    "model.embed_tokens.weight",
                    "model-00001-of-00002.safetensors",
                ),
                ti("lm_head.weight", "model-00002-of-00002.safetensors"),
            ],
            Vec::new(),
            None,
            None,
        );
        e.tree_state.selected = 0;
        let view = e.describe_selection();
        assert_eq!(view.copy_path.as_deref(), Some("/ckpt"));
        assert_eq!(view.status.1, "2 files in /ckpt");
        assert!(view.status.1.contains(view.copy_path.as_deref().unwrap()));
    }

    #[test]
    fn layout_command_encodes_relative_path_and_selection() {
        // Launched with a shard file: the layout path is emitted relative to the
        // checkpoint directory (no duplication), and the selected tensor is
        // recorded so the precise view round-trips.
        let e = Explorer::new(
            vec![PathBuf::from("/ckpt/model-00016.safetensors")],
            Vec::new(),
            None,
            false,
        );
        let plain = e.command_for_layout("/ckpt/model-00016.safetensors", None);
        assert!(
            plain.contains("--layout model-00016.safetensors"),
            "relative path:\n{plain}"
        );
        assert!(!plain.contains("--layout-select"), "no selection:\n{plain}");

        let with_sel = e.command_for_layout(
            "/ckpt/model-00016.safetensors",
            Some("model.embed_tokens.weight"),
        );
        assert!(
            with_sel.contains("--layout-select model.embed_tokens.weight"),
            "selection encoded:\n{with_sel}"
        );
    }

    #[test]
    fn remote_browse_adapts_to_the_source_kind() {
        // An s3:// source browses s3-natively; the file view is available remotely.
        let mut e = Explorer::new(
            vec![PathBuf::from("s3://bucket/models/ckpt")],
            Vec::new(),
            None,
            false,
        );
        e.set_remote_read("host".into(), "~/venv".into());
        assert!(
            matches!(e.remote_browse(), Some(RemoteBrowse::S3(u)) if u == "s3://bucket/models/ckpt")
        );
        assert!(e.file_view_available());

        // A remote directory browses over SFTP as itself.
        let mut e = Explorer::new(
            vec![PathBuf::from("/opt/models/ckpt")],
            Vec::new(),
            None,
            false,
        );
        e.set_remote_read("host".into(), "~/venv".into());
        assert!(
            matches!(e.remote_browse(), Some(RemoteBrowse::Sftp(d)) if d == "/opt/models/ckpt")
        );

        // A single remote shard browses its parent directory.
        let mut e = Explorer::new(
            vec![PathBuf::from("/opt/models/ckpt/model-00001.safetensors")],
            Vec::new(),
            None,
            false,
        );
        e.set_remote_read("host".into(), "~/venv".into());
        assert!(
            matches!(e.remote_browse(), Some(RemoteBrowse::Sftp(d)) if d == "/opt/models/ckpt")
        );
    }

    #[test]
    fn remote_layout_command_is_relative_to_the_sftp_dir() {
        // A remote SFTP checkpoint: `--layout` emits the shard relative to the
        // browse dir, and the reopen carries the scp-style host:path (round-trip).
        let mut e = Explorer::new(
            vec![PathBuf::from("/opt/models/ckpt")],
            Vec::new(),
            None,
            false,
        );
        e.set_remote_read("net004".into(), "~/venv".into());
        let cmd = e.command_for_layout("/opt/models/ckpt/model-00001.safetensors", None);
        assert!(
            cmd.contains("--layout model-00001.safetensors"),
            "relative to the remote dir:\n{cmd}"
        );
        assert!(
            cmd.contains("net004:/opt/models/ckpt"),
            "scp-style host:\n{cmd}"
        );
    }

    #[test]
    fn s3_root_label_uses_the_last_prefix_component() {
        assert_eq!(s3_root_label("s3://bucket/a/b/ckpt"), "ckpt");
        assert_eq!(s3_root_label("s3://bucket/ckpt/"), "ckpt");
        assert_eq!(s3_root_label("s3://bucket"), "bucket");
    }

    #[test]
    fn file_command_registry_maps_keys() {
        // The file view's Tab (`\t`) toggles back to the tensor tree; other keys
        // map through the file registry, and the tree registry now offers Files.
        assert_eq!(file_command_for_key('l'), Some(FileCmd::Legend));
        assert_eq!(file_command_for_key('f'), Some(FileCmd::CopyPath));
        assert_eq!(file_command_for_key('q'), Some(FileCmd::Quit));
        assert_eq!(file_command_for_key('z'), None); // unbound
        assert_eq!(tree_command_for_key('\t'), Some(Cmd::ViewFiles));
        assert_eq!(key_label('\t'), "Tab");
        assert_eq!(key_label('q'), "q");
    }

    #[test]
    fn layout_and_detail_palettes_map_keys() {
        // Layout palette.
        assert_eq!(layout_command_for_key('l'), Some(LayoutCmd::Legend));
        assert_eq!(layout_command_for_key('y'), Some(LayoutCmd::CopyCommand));
        assert_eq!(layout_command_for_key('q'), Some(LayoutCmd::Quit));
        assert_eq!(layout_command_for_key('\t'), Some(LayoutCmd::TensorTree));
        assert_eq!(layout_command_for_key('z'), None);

        // Detail palette: each command's synthesized key round-trips, and the file
        // layout maps to Tab.
        assert_eq!(detail_cmd_key(DetailCmd::Heatmap).code, KeyCode::Char('m'));
        assert_eq!(detail_cmd_key(DetailCmd::FileLayout).code, KeyCode::Tab);

        // dtype/reshape only when overridable; file layout only when local .st.
        let full = available_detail_commands(true, true);
        assert!(full.iter().any(|(c, ..)| *c == DetailCmd::Dtype));
        assert!(full.iter().any(|(c, ..)| *c == DetailCmd::FileLayout));
        let bare = available_detail_commands(false, false);
        assert!(!bare.iter().any(|(c, ..)| *c == DetailCmd::Dtype));
        assert!(!bare.iter().any(|(c, ..)| *c == DetailCmd::Reshape));
        assert!(!bare.iter().any(|(c, ..)| *c == DetailCmd::FileLayout));
        // The data views / copies are always offered.
        assert!(bare.iter().any(|(c, ..)| *c == DetailCmd::Heatmap));
        assert!(bare.iter().any(|(c, ..)| *c == DetailCmd::CopyCommand));

        // Data view palette: keys synthesize back to their shortcut, and
        // dtype/reshape gate on overridable.
        assert_eq!(data_cmd_key(DataCmd::Values).code, KeyCode::Char('v'));
        assert_eq!(data_cmd_key(DataCmd::Base).code, KeyCode::Char('b'));
        assert!(
            available_data_commands(true)
                .iter()
                .any(|(c, ..)| *c == DataCmd::Reshape)
        );
        assert!(
            !available_data_commands(false)
                .iter()
                .any(|(c, ..)| *c == DataCmd::Dtype)
        );
    }

    #[test]
    fn command_registry_maps_keys_and_filters_unavailable() {
        // The key handler and the palette share one key→command table.
        assert_eq!(tree_command_for_key('s'), Some(Cmd::Stats));
        assert_eq!(tree_command_for_key('/'), Some(Cmd::Search));
        assert_eq!(tree_command_for_key('q'), Some(Cmd::Quit));
        assert_eq!(tree_command_for_key('z'), None); // unbound

        // With no files there's nothing to repack (needs an HDF5 source) or rename
        // (needs a local safetensors checkpoint), so the palette omits both but
        // still offers the rest.
        let e = Explorer::new(Vec::new(), Vec::new(), None, false);
        let available: Vec<Cmd> = e.available_commands().iter().map(|&(c, ..)| c).collect();
        assert!(available.contains(&Cmd::Stats));
        assert!(available.contains(&Cmd::Quit));
        assert!(
            !available.contains(&Cmd::Repack),
            "repack needs an HDF5 source: {available:?}"
        );
        assert!(
            !available.contains(&Cmd::Rename),
            "rename needs a local safetensors checkpoint: {available:?}"
        );
        assert_eq!(available.len(), TREE_COMMANDS.len() - 2);

        // A *writable* local safetensors checkpoint (even one shard) offers Rename.
        let dir = std::env::temp_dir().join(format!("ce_rename_gate_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("model.safetensors");
        std::fs::write(&f, b"").unwrap();
        let st = Explorer::new(vec![f.clone()], Vec::new(), None, false);
        assert!(st.rename_target().is_some());
        assert!(st.can_rename(), "a writable local safetensors is editable");
        assert!(
            st.available_commands()
                .iter()
                .any(|&(c, ..)| c == Cmd::Rename)
        );

        // A read-only file (the read-only-mount case) is structurally a local
        // safetensors but can't be renamed in place, so it's NOT editable and the
        // Rename command is hidden.
        let mut perms = std::fs::metadata(&f).unwrap().permissions();
        perms.set_readonly(true);
        std::fs::set_permissions(&f, perms.clone()).unwrap();
        let ro = Explorer::new(vec![f.clone()], Vec::new(), None, false);
        assert!(ro.rename_target().is_some(), "still local safetensors");
        assert!(
            !ro.can_rename(),
            "a read-only file can't be renamed in place"
        );
        assert!(
            !ro.available_commands()
                .iter()
                .any(|&(c, ..)| c == Cmd::Rename)
        );
        // Removing the file only needs a writable parent dir, so the read-only file
        // cleans up fine.
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Guardrail: every command a mode offers (in its registry, currently available)
    /// must be *shown* in that mode's footer with its real key — no bound-but-hidden
    /// or mislabeled bindings (the class of bug behind the tree's old `R repack`).
    /// Renders the tree with a writable safetensors so Rename is available, then
    /// checks each available `TREE_COMMANDS` key appears among the footer chips.
    #[test]
    fn tree_footer_advertises_every_available_command_key() {
        use crate::tree::Layout;
        let ti = |name: &str| TensorInfo {
            name: name.to_string(),
            dtype: "F32".into(),
            shape: vec![2, 2],
            size_bytes: 16,
            num_elements: 4,
            storage: Storage::Unknown,
            source_path: "/tmp/x.safetensors".into(),
            layout: Layout::None,
        };
        let dir = std::env::temp_dir().join(format!("ce_footer_enforce_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("model.safetensors");
        std::fs::write(&f, b"").unwrap();
        let mut e = Explorer::new(vec![f], Vec::new(), None, false);
        e.finalize_load(vec![ti("blk.0.a"), ti("blk.1.b")], Vec::new(), None, None);
        assert!(
            e.can_rename(),
            "a writable local safetensors → Rename available"
        );

        // Render populates `clickable` with the footer chips (Rect → KeyEvent).
        crate::tui::headless_render(140, 30, |frame| e.render_tree_frame(frame, true)).unwrap();
        let shown: HashSet<KeyEvent> = e.clickable.borrow().iter().map(|&(_, k)| k).collect();

        for &(cmd, _, _, c) in TREE_COMMANDS {
            if !e.available_commands().iter().any(|&(a, ..)| a == cmd) {
                continue; // gated off in this context (e.g. Repack needs HDF5)
            }
            let want = if c == '\t' {
                KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE)
            } else {
                KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)
            };
            assert!(
                shown.contains(&want),
                "tree command {cmd:?} (key {c:?}) is bound but not shown in the footer"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Guardrail across every static-footer mode (tree/files/layout/rename): each
    /// command in the registry (except the `App` group — Back/Quit are handled via
    /// Esc/⌫/`^C`/`q`, not a content chip) must appear in that mode's footer with its
    /// real key. Calls the footer builders directly and matches their chips against
    /// the registry keys — so a bound-but-hidden or mislabeled key fails CI (as the
    /// tree's old `R repack` and the layout's missing `Tab` did).
    #[test]
    fn every_static_mode_footer_shows_its_command_keys() {
        // A registry char → the KeyEvent its footer chip carries (matching
        // `hint_key` for bare letters, and the rename registry's Ctrl sentinels).
        fn footer_key(c: char) -> KeyEvent {
            match c {
                '\t' => KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
                '\r' => KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
                '\u{1b}' => KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE),
                c if (c as u32) < 32 => {
                    let letter = (b'a' + c as u8 - 1) as char;
                    KeyEvent::new(KeyCode::Char(letter), KeyModifiers::CONTROL)
                }
                c => KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE),
            }
        }
        fn check(name: &str, cmds: &[(&str, char)], chips: Vec<crate::ui::ChipHit>) {
            let shown: HashSet<KeyEvent> = chips.into_iter().map(|c| c.key).collect();
            for &(group, c) in cmds {
                if group == "App" || key_label(c).is_empty() {
                    continue; // Back/Quit are handled via Esc/⌫/^C/q, not a content chip
                }
                assert!(
                    shown.contains(&footer_key(c)),
                    "{name}: command key {c:?} ({}) is bound but not shown in the footer",
                    key_label(c),
                );
            }
        }
        check(
            "tree",
            &TREE_COMMANDS
                .iter()
                .map(|&(_, g, _, c)| (g, c))
                .collect::<Vec<_>>(),
            crate::ui::tree_hint_lines(true, true, 200).1,
        );
        check(
            "files",
            &FILE_COMMANDS
                .iter()
                .map(|&(_, g, _, c)| (g, c))
                .collect::<Vec<_>>(),
            crate::ui::files_hint_lines(200).1,
        );
        check(
            "layout",
            &LAYOUT_COMMANDS
                .iter()
                .map(|&(_, g, _, c)| (g, c))
                .collect::<Vec<_>>(),
            crate::ui::layout_hint_lines(200).1,
        );
        check(
            "rename",
            &RENAME_COMMANDS
                .iter()
                .map(|&(_, g, _, c)| (g, c))
                .collect::<Vec<_>>(),
            crate::ui::rename_hint_lines(200, true).1,
        );
        // Detail's footer is now the same chip format; render it with everything
        // available (overridable dtype, local file layout, non-remote so `s` shows).
        check(
            "detail",
            &DETAIL_COMMANDS
                .iter()
                .map(|&(_, g, _, c)| (g, c))
                .collect::<Vec<_>>(),
            crate::ui::detail_footer_lines(true, false, true, 200).1,
        );
        // The data view's footer is state-dependent: the `m`/`v` switch shows only the
        // *other* representation, and zebra/base only in the numeric grid — so no
        // single state shows every command. Union the two representations (numeric +
        // heatmap, both overridable) and require they cover every DATA_COMMANDS key.
        let data_chips = {
            let state = |heatmap: bool| {
                crate::ui::data_view_footer_wrapped_lines(
                    crate::sample::SampleMode::Grid,
                    1,
                    true,
                    heatmap,
                    crate::ui::StripeMode::Rows,
                    crate::ui::NumBase::Decimal,
                    200,
                )
                .1
            };
            let mut c = state(false);
            c.extend(state(true));
            c
        };
        check(
            "data",
            &DATA_COMMANDS
                .iter()
                .map(|&(_, g, _, c)| (g, c))
                .collect::<Vec<_>>(),
            data_chips,
        );
        // Stats footer rendered with the per-shard fold available, so `f` shows too.
        check(
            "stats",
            &STATS_COMMANDS
                .iter()
                .map(|&(_, g, _, c)| (g, c))
                .collect::<Vec<_>>(),
            crate::ui::stats_hint_lines(true, false, 200).1,
        );
    }

    #[test]
    fn command_for_rename_round_trips_the_rule_pairs() {
        // The `y` command for the rename editor: `--rename` plus one
        // `--rename-rule 'src=>tgt'` per complete pair, so it reopens the editor
        // with the same schema pairs (lossless — no regex reversal).
        let e = Explorer::new(
            vec![PathBuf::from("/ckpt/model.safetensors")],
            Vec::new(),
            None,
            false,
        );
        let pairs = vec![
            (
                "model.layers.{layer}.attn.q_proj.weight".to_string(),
                "model.layers.{layer}.self_attn.q_proj.weight".to_string(),
            ),
            // A blank pair is dropped (never emitted).
            (String::new(), String::new()),
            ("a.0.w".to_string(), "b.0.w".to_string()),
        ];
        let cmd = e.command_for_rename(&pairs);
        assert!(cmd.contains("--rename "), "opens the editor:\n{cmd}");
        assert_eq!(cmd.matches("--rename-rule").count(), 2, "two rules:\n{cmd}");
        assert!(
            cmd.contains("--rename-rule 'a.0.w=>b.0.w'"),
            "literal:\n{cmd}"
        );

        // Re-parse the emitted `--rename-rule` values the way `interactive_loop`
        // seeds them (split on the first `=>`) and confirm the pairs round-trip.
        let parsed: Vec<(String, String)> = cmd
            .split("--rename-rule ")
            .skip(1)
            .map(|rest| {
                let raw = rest.split(" --").next().unwrap_or(rest).trim();
                let unquoted = raw.trim_matches('\'');
                let (s, t) = unquoted.split_once("=>").unwrap();
                (s.to_string(), t.to_string())
            })
            .collect();
        assert_eq!(
            parsed,
            vec![
                (
                    "model.layers.{layer}.attn.q_proj.weight".to_string(),
                    "model.layers.{layer}.self_attn.q_proj.weight".to_string()
                ),
                ("a.0.w".to_string(), "b.0.w".to_string()),
            ]
        );
    }

    #[test]
    fn open_link_opens_layout_and_reveals_tensor() {
        use crate::tree::{Layout, Storage, TensorInfo};
        let ti = |name: &str| TensorInfo {
            name: name.to_string(),
            dtype: "F32".into(),
            shape: vec![2, 2],
            size_bytes: 16,
            num_elements: 4,
            storage: Storage::Unknown,
            source_path: "/tmp/x.safetensors".into(),
            layout: Layout::None,
        };
        let mut e = Explorer::new(Vec::new(), Vec::new(), None, false);
        e.finalize_load(vec![ti("blk.0.a"), ti("blk.1.b")], Vec::new(), None, None);

        // A `Layout` link opens that file's byte-layout view.
        assert!(matches!(
            e.open_link(&crate::ui::Link::Layout("/tmp/x.safetensors".into())),
            Some(Nav::Open(Screen::Layout { .. }))
        ));

        // A `Tree` link to a real tensor reveals it and lands on the tree.
        e.tree_state.set_all_expanded(true);
        assert!(matches!(
            e.open_link(&crate::ui::Link::Tree("blk.1.b".into())),
            Some(Nav::Open(Screen::Tree))
        ));
        assert_eq!(
            e.tree_state.flattened[e.tree_state.selected].0.name(),
            "blk.1.b"
        );

        // A `Tree` link to an absent tensor is a no-op (a stray click).
        assert!(e.open_link(&crate::ui::Link::Tree("nope".into())).is_none());
    }

    #[test]
    fn rename_palette_registry_labels_and_gating() {
        // The control-char sentinels render as their real Ctrl accelerators — every
        // rename command now has a shown footer key (none are palette-only).
        assert_eq!(key_label('\r'), "Enter");
        assert_eq!(key_label('\u{e}'), "^N");
        assert_eq!(key_label('\u{19}'), "^Y");
        assert_eq!(key_label('\u{1}'), "^A");
        assert_eq!(key_label('\u{c}'), "^L");
        assert_eq!(key_label('\u{1b}'), "Esc");

        // Every rename command has one-line help, looked up by its sentinel char.
        for &(_, _, _, key) in RENAME_COMMANDS {
            assert!(
                crate::ui::shortcut_help(
                    KeyEvent::new(KeyCode::Char(key), KeyModifiers::NONE),
                    HelpCtx::Rename,
                )
                .is_some(),
                "no help for rename key {key:?}"
            );
        }

        // Apply needs a clean staged rename; the apply-command copy needs a rule;
        // Remove needs more than one pair.
        let full = available_rename_commands(true, true, 2);
        assert!(full.iter().any(|(c, ..)| *c == RenameCmd::Apply));
        assert!(full.iter().any(|(c, ..)| *c == RenameCmd::CopyApplyCmd));
        assert!(full.iter().any(|(c, ..)| *c == RenameCmd::RemoveRule));
        let bare = available_rename_commands(false, false, 1);
        assert!(!bare.iter().any(|(c, ..)| *c == RenameCmd::Apply));
        assert!(!bare.iter().any(|(c, ..)| *c == RenameCmd::CopyApplyCmd));
        assert!(!bare.iter().any(|(c, ..)| *c == RenameCmd::RemoveRule));
        // Copy-screen / reopen-command / legend / back / quit are always offered.
        assert!(bare.iter().any(|(c, ..)| *c == RenameCmd::CopyScreen));
        assert!(bare.iter().any(|(c, ..)| *c == RenameCmd::CopyReopenCmd));
        assert!(bare.iter().any(|(c, ..)| *c == RenameCmd::Back));
    }

    #[test]
    fn rename_mode_suggests_by_substring_and_grows_pairs() {
        let names: Vec<String> = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.embed_tokens.weight",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        // Autocomplete matches the deduped, *generalized* schemas — number-agnostic,
        // so a concrete "Q_PROJ" query finds the {layer} family. Each schema carries
        // the count of tensors it covers (the dropdown's `×N` column).
        let schemas: Vec<(String, usize)> = {
            let mut counts: HashMap<String, usize> = HashMap::new();
            for n in &names {
                *counts.entry(crate::rename::generalize(n).0).or_default() += 1;
            }
            let mut seen = HashSet::new();
            names
                .iter()
                .map(|n| crate::rename::generalize(n).0)
                .filter(|s| seen.insert(s.clone()))
                .map(|s| {
                    let c = counts[&s];
                    (s, c)
                })
                .collect()
        };
        let texts = |mode: &RenameMode| -> Vec<String> {
            mode.completions(&schemas)
                .iter()
                .map(|c| c.text.clone())
                .collect()
        };
        let mut mode = RenameMode::default();
        mode.pairs[0].source = "Q_PROJ".to_string();
        assert_eq!(
            texts(&mode),
            vec!["model.layers.{layer}.self_attn.q_proj.weight"]
        );
        // Empty query lists all families for discovery.
        mode.pairs[0].source.clear();
        assert_eq!(mode.completions(&schemas).len(), 3);
        // The new-name field autocompletes too (both fields, pgcli-style).
        mode.on_target = true;
        mode.pairs[0].target = "up_proj".to_string();
        assert_eq!(
            texts(&mode),
            vec!["model.layers.{layer}.mlp.up_proj.weight"]
        );

        // ↓ past the last field grows a new pair and focuses its source.
        let before = mode.pairs.len();
        mode.focus_down();
        assert_eq!(mode.pairs.len(), before + 1);
        assert_eq!(mode.focus_pair, before);
        assert!(!mode.on_target);
    }

    #[test]
    fn rename_tab_completes_longest_common_prefix_not_the_first_match() {
        let names: Vec<String> = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let schemas: Vec<(String, usize)> = {
            let mut seen = HashSet::new();
            names
                .iter()
                .map(|n| crate::rename::generalize(n).0)
                .filter(|s| seen.insert(s.clone()))
                .map(|s| (s, 1))
                .collect()
        };
        let mut mode = RenameMode::default();

        // Two `self_attn` families share a stem → Tab fills up to it (not to either
        // full candidate — that's Enter's job), and leaves the dropdown open.
        mode.pairs[0].source = "self_attn".to_string();
        mode.complete_prefix(&schemas);
        assert_eq!(mode.pairs[0].source, "model.layers.{layer}.self_attn.");
        assert!(mode.menu.is_some(), "dropdown stays open after Tab");

        // Narrowed to a single family → Tab can complete the whole schema.
        mode.pairs[0].source = "q_proj".to_string();
        mode.complete_prefix(&schemas);
        assert_eq!(
            mode.pairs[0].source,
            "model.layers.{layer}.self_attn.q_proj.weight"
        );

        // A mid-token query whose matches' common prefix would *drop* it (broadening
        // the filter) is left untouched — Tab never clobbers the typed text.
        mode.pairs[0].source = "proj".to_string();
        mode.complete_prefix(&schemas);
        assert_eq!(mode.pairs[0].source, "proj");
    }

    #[test]
    fn rename_backspacing_a_rule_to_empty_removes_it_and_moves_to_a_neighbor() {
        let pair = |s: &str, t: &str| RenamePair {
            source: s.to_string(),
            target: t.to_string(),
        };

        // Deleting the last char of a middle/last rule (both fields then blank)
        // removes it and lands at the end of the *previous* rule's new-name field.
        let mut mode = RenameMode {
            pairs: vec![pair("a", "b"), pair("c", "d"), pair("x", "")],
            focus_pair: 2,
            cursor: 1,
            ..Default::default()
        };
        mode.backspace(); // "x" → ""
        mode.remove_pair_if_empty();
        assert_eq!(mode.pairs.len(), 2);
        assert_eq!(mode.focus_pair, 1);
        assert!(mode.on_target, "lands at the end of the previous rule");

        // Removing the *first* rule moves to the new first rule's source instead.
        let mut mode = RenameMode {
            pairs: vec![pair("e", ""), pair("f", "g")],
            cursor: 1,
            ..Default::default()
        };
        mode.backspace(); // "e" → ""
        mode.remove_pair_if_empty();
        assert_eq!(mode.pairs.len(), 1);
        assert_eq!(mode.focus_pair, 0);
        assert!(!mode.on_target);
        assert_eq!(mode.pairs[0].source, "f");

        // A rule with content still in the *other* field is NOT removed.
        let mut mode = RenameMode {
            pairs: vec![pair("a", "b"), pair("", "keep")],
            focus_pair: 1,
            ..Default::default()
        };
        mode.remove_pair_if_empty();
        assert_eq!(mode.pairs.len(), 2, "the new-name field still has content");

        // The last remaining rule is never removed (always ≥1).
        let mut mode = RenameMode::default();
        mode.remove_pair_if_empty();
        assert_eq!(mode.pairs.len(), 1);
    }

    #[test]
    fn rename_mode_build_map_combines_complete_pairs_and_notes_partial() {
        let mut mode = RenameMode::default();
        mode.pairs[0] = RenamePair {
            source: "model.layers.{layer}.self_attn.q_proj.weight".to_string(),
            target: "model.layers.{layer}.attn.q_proj.weight".to_string(),
        };
        // A half-filled pair becomes a note, not a rule.
        mode.pairs.push(RenamePair {
            source: "dangling".to_string(),
            target: String::new(),
        });
        let (map, notes) = mode.build_map().unwrap();
        assert_eq!(map.len(), 1, "only the complete pair becomes a rule");
        assert_eq!(notes.len(), 1, "the half-filled pair is noted");
        assert_eq!(
            map.map("model.layers.5.self_attn.q_proj.weight")
                .into_owned(),
            "model.layers.5.attn.q_proj.weight"
        );
    }

    #[test]
    fn held_key_scroll_accelerates_then_resets() {
        // Both curves start at a 1:1 grace and keep building the longer the key is
        // held (no low plateau); the page curve is far more aggressive than the
        // gentle arrow curve, which caps low so row movement stays controllable.
        assert_eq!(accel_step_row(0), 1);
        assert_eq!(accel_step_page(0), 1);
        assert!(accel_step_row(20) > accel_step_row(8));
        assert!(accel_step_page(20) > accel_step_page(8));
        assert_eq!(accel_step_row(1_000), 32, "arrows cap low");
        assert!(
            accel_step_page(40) > accel_step_row(1_000),
            "paging accelerates well past the arrow cap"
        );

        let e = Explorer::new(Vec::new(), Vec::new(), None, false);
        // Rapid repeats of the same key (all within the repeat window, since these
        // calls are microseconds apart) ramp the step up.
        let steps: Vec<usize> = (0..30)
            .map(|_| e.held_step(KeyCode::Down, accel_step_row))
            .collect();
        assert_eq!(steps[0], 1, "first press is 1:1");
        assert!(
            *steps.last().unwrap() > 1,
            "a held key accelerates: {steps:?}"
        );
        // A different key resets the streak — a tap the other way is precise again.
        assert_eq!(e.held_step(KeyCode::Up, accel_step_row), 1);
    }

    #[test]
    fn wrong_layout_char_flags_only_plain_non_latin_letters() {
        let k = |c: char| KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE);
        assert_eq!(wrong_layout_char(&k('м')), Some('м')); // Cyrillic (RU 'v')
        assert_eq!(wrong_layout_char(&k('ん')), Some('ん')); // Japanese
        assert_eq!(wrong_layout_char(&k('m')), None); // ASCII shortcut
        assert_eq!(wrong_layout_char(&k('/')), None); // ASCII punctuation
        assert_eq!(wrong_layout_char(&k('5')), None); // digit
        // Ctrl/Alt combinations are intentional, not layout mistakes.
        let ctrl_cyrillic = KeyEvent::new(KeyCode::Char('м'), KeyModifiers::CONTROL);
        assert_eq!(wrong_layout_char(&ctrl_cyrillic), None);
    }

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

    #[test]
    fn parse_shape_input_validates_and_infers() {
        // Explicit dims with assorted separators; product must match.
        assert_eq!(parse_shape_input("10, 100", 1000), Ok(vec![10, 100]));
        assert_eq!(parse_shape_input("2 3 4", 24), Ok(vec![2, 3, 4]));
        assert_eq!(parse_shape_input("4x5", 20), Ok(vec![4, 5]));
        // A single wildcard is inferred from the element count (`-1`, `*`, `_`).
        assert_eq!(parse_shape_input("-1, 100", 1000), Ok(vec![10, 100]));
        assert_eq!(parse_shape_input("100, *", 1000), Ok(vec![100, 10]));
        assert_eq!(parse_shape_input("_", 42), Ok(vec![42]));
        assert_eq!(parse_shape_input("2, _, 4", 24), Ok(vec![2, 3, 4]));
        // Errors: wrong product, non-divisible wildcard, two wildcards, zero, junk.
        assert!(parse_shape_input("10, 10", 1000).is_err());
        assert!(parse_shape_input("-1, 3", 1000).is_err()); // 1000 % 3 != 0
        assert!(parse_shape_input("-1, -1", 1000).is_err());
        assert!(parse_shape_input("0, 5", 0).is_err());
        assert!(parse_shape_input("", 10).is_err());
        assert!(parse_shape_input("abc", 10).is_err());
    }

    /// Build an explorer whose flattened tree has the given row depths (the
    /// node contents don't matter for coarse navigation, only the depths).
    fn explorer_with_depths(depths: &[usize]) -> Explorer {
        let mut e = Explorer::new(Vec::new(), Vec::new(), None, false);
        e.tree_state.flattened = depths
            .iter()
            .map(|&d| {
                (
                    TreeNode::Group {
                        name: String::new(),
                        children: Vec::new(),
                        expanded: false,
                        tensor_count: 0,
                        params: 0,
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

        e.tree_state.selected = 3;
        e.tree_state.move_to_parent();
        assert_eq!(e.tree_state.selected, 2);

        e.tree_state.selected = 1;
        e.tree_state.move_to_parent();
        assert_eq!(e.tree_state.selected, 0);

        // Top-level row has no parent.
        e.tree_state.selected = 0;
        e.tree_state.move_to_parent();
        assert_eq!(e.tree_state.selected, 0);
    }

    // reveal_tensor lands the cursor on a leaf whether or not its group is already
    // open — and when it's already visible it must NOT rebuild the flattened tree
    // (that rebuild was the lag returning to a big expanded remote tree).
    #[test]
    fn reveal_tensor_moves_cursor_and_only_reflattens_when_needed() {
        use crate::tree::{Layout, Storage, TensorInfo};
        let ti = |name: &str| TensorInfo {
            name: name.to_string(),
            dtype: "F32".into(),
            shape: vec![2, 2],
            size_bytes: 16,
            num_elements: 4,
            storage: Storage::Unknown,
            source_path: "/tmp/x.safetensors".into(),
            layout: Layout::None,
        };
        let mut e = Explorer::new(Vec::new(), Vec::new(), None, false);
        e.finalize_load(
            vec![ti("blk.0.a"), ti("blk.0.b"), ti("blk.1.a"), ti("blk.1.b")],
            Vec::new(),
            None,
            None,
        );

        // Fully expanded: the leaf is already visible, so revealing it just moves
        // the cursor onto that exact row without changing the flattened tree.
        e.tree_state.set_all_expanded(true);
        let before = e.tree_state.flattened.clone();
        e.tree_state.reveal("blk.1.b");
        assert_eq!(e.tree_state.flattened.len(), before.len());
        assert_eq!(
            e.tree_state.flattened[e.tree_state.selected].0.name(),
            "blk.1.b",
            "cursor should land on the revealed leaf"
        );

        // Collapsed: the leaf isn't visible, so reveal must expand its ancestors,
        // grow the flattened tree, and still land on it.
        e.tree_state.set_all_expanded(false);
        let collapsed_rows = e.tree_state.flattened.len();
        e.tree_state.selected = 0;
        e.tree_state.reveal("blk.1.b");
        assert!(
            e.tree_state.flattened.len() > collapsed_rows,
            "reveal expands to it"
        );
        assert_eq!(
            e.tree_state.flattened[e.tree_state.selected].0.name(),
            "blk.1.b"
        );
    }

    fn export_fixture() -> Explorer {
        use crate::tree::{Layout, Storage, TensorInfo};
        let ti = |name: &str, file: &str| TensorInfo {
            name: name.to_string(),
            dtype: "F32".into(),
            shape: vec![4, 8],
            size_bytes: 128,
            num_elements: 32,
            storage: Storage::Unknown,
            source_path: format!("/ckpt/{file}"),
            layout: Layout::None,
        };
        let mut e = Explorer::new(Vec::new(), Vec::new(), None, false);
        e.finalize_load(
            vec![
                ti("model.layers.0.a", "model-00001.safetensors"),
                ti("model.layers.0.b", "model-00001.safetensors"),
                ti("model.layers.1.a", "model-00002.safetensors"),
                ti("model.layers.1.b", "model-00002.safetensors"),
            ],
            Vec::new(),
            None,
            None,
        );
        e
    }

    #[test]
    fn tree_text_export_is_fully_expanded() {
        let e = export_fixture();
        let text = e.tree_text(TreeDetail::Compact);
        // Every group opens (▾, never a collapsed ▸) and the numeric layer group
        // is summarised (≡ 2), independent of the live collapse state.
        assert!(
            !text.contains('▸'),
            "export must be fully expanded:\n{text}"
        );
        assert!(text.contains("≡ 2"), "layer count shown:\n{text}");
        // All four leaves are listed.
        assert_eq!(text.matches(" [F32, ").count(), 4);
    }

    #[test]
    fn tree_text_export_keeps_full_metadata() {
        use crate::tree::{Layout, MetadataInfo, Storage, TensorInfo};
        // A metadata value well past the live tree's 50-char cap.
        let long = "A".repeat(300);
        let mut e = Explorer::new(Vec::new(), Vec::new(), None, false);
        e.finalize_load(
            vec![TensorInfo {
                name: "blk.weight".into(),
                dtype: "F32".into(),
                shape: vec![2, 2],
                size_bytes: 16,
                num_elements: 4,
                storage: Storage::Unknown,
                source_path: "/x.safetensors".into(),
                layout: Layout::None,
            }],
            vec![MetadataInfo {
                name: "blk.weight.__metadata__".into(),
                value: long.clone(),
                value_type: "string".into(),
            }],
            None,
            None,
        );
        let text = e.tree_text(TreeDetail::Compact);
        // The export shows the whole value — not the "…"-truncated tree-row form.
        assert!(
            text.contains(&long),
            "metadata truncated in export:\n{text}"
        );
    }

    #[test]
    fn tensors_text_export_is_flat_and_natural_sorted() {
        let e = export_fixture();
        let flat = e.tensors_text(TreeDetail::Compact);
        let lines: Vec<&str> = flat.lines().collect();
        assert_eq!(lines.len(), 4);
        // Full tensor names (not the abbreviated tree labels), in natural order.
        assert!(lines[0].contains("model.layers.0.a"));
        assert!(lines[3].contains("model.layers.1.b"));
        // --verbose appends the source file.
        let flat_v = e.tensors_text(TreeDetail::Full);
        assert!(flat_v.contains("← model-00001.safetensors"));
    }

    #[test]
    fn tree_json_export_matches_index_json_shape() {
        let e = export_fixture();
        let v: serde_json::Value = serde_json::from_str(&e.tree_json(TreeDetail::Compact)).unwrap();
        assert_eq!(v["metadata"]["total_size"], 512); // 4 × 128 bytes
        assert_eq!(
            v["weight_map"]["model.layers.1.a"],
            "model-00002.safetensors"
        );
        assert!(
            v.get("tensors").is_none(),
            "compact omits per-tensor detail"
        );
        // --verbose adds a tensors block keyed by name.
        let full: serde_json::Value = serde_json::from_str(&e.tree_json(TreeDetail::Full)).unwrap();
        assert_eq!(full["tensors"]["model.layers.0.a"]["dtype"], "F32");
        assert_eq!(
            full["tensors"]["model.layers.0.a"]["shape"],
            serde_json::json!([4, 8])
        );
    }

    #[test]
    fn oversized_clipboard_copy_is_refused_not_spilled() {
        // A payload whose base64 exceeds the OSC 52 ceiling is refused (returns
        // false, emits nothing) rather than dumped to the terminal as raw
        // base64 — the failure mode for copying a very large tree.
        let big = "x".repeat(OSC52_MAX_B64); // base64 is ~4/3 larger → over the cap
        assert!(!copy_to_clipboard(&big));
        // (The success path is not asserted here: it would emit the OSC 52
        // escape and clobber the clipboard of whoever runs the tests.)
    }

    #[test]
    fn tensors_json_export_is_names_then_objects() {
        let e = export_fixture();
        let names: serde_json::Value =
            serde_json::from_str(&e.tensors_json(TreeDetail::Compact)).unwrap();
        assert_eq!(names.as_array().unwrap().len(), 4);
        assert_eq!(names[0], "model.layers.0.a"); // natural-sorted array
        let full: serde_json::Value =
            serde_json::from_str(&e.tensors_json(TreeDetail::Full)).unwrap();
        assert_eq!(full[0]["name"], "model.layers.0.a");
        assert_eq!(full[0]["file"], "model-00001.safetensors");
        assert_eq!(full[0]["num_elements"], 32);
    }

    #[test]
    fn copy_menu_covers_all_eight_cli_variants() {
        let e = export_fixture();
        // One menu entry per (shape × format × verbosity) CLI combination.
        assert_eq!(EXPORT_CHOICES.len(), 8);
        for c in EXPORT_CHOICES {
            // The command carries exactly the flags for this choice…
            let cmd = e.export_command(*c);
            let shape_flag = match c.shape {
                ExportShape::Tree => "--print-tree",
                ExportShape::Tensors => "--print-tensors",
            };
            assert!(cmd.contains(shape_flag), "{cmd}");
            assert_eq!(
                cmd.contains("--format json"),
                c.format == TreeFormat::Json,
                "{cmd}"
            );
            assert_eq!(
                cmd.split_whitespace().any(|t| t == "-v"),
                c.detail == TreeDetail::Full,
                "{cmd}"
            );
            // …and export_text dispatches to the matching generator.
            let expected = match (c.shape, c.format) {
                (ExportShape::Tree, TreeFormat::Text) => e.tree_text(c.detail),
                (ExportShape::Tree, TreeFormat::Json) => e.tree_json(c.detail),
                (ExportShape::Tensors, TreeFormat::Text) => e.tensors_text(c.detail),
                (ExportShape::Tensors, TreeFormat::Json) => e.tensors_json(c.detail),
            };
            assert_eq!(e.export_text(*c), expected, "{}", c.label);
        }
    }

    #[test]
    fn move_selection_pages_through_the_browsing_tree_and_clamps() {
        // Not searching, so move_selection (what PageUp/PageDown call) walks the
        // full flattened tree rather than the filtered results.
        let mut e = explorer_with_depths(&vec![0; 100]);
        assert!(!e.tree_state.search_mode());

        // A page-sized jump down advances by the delta.
        e.tree_state.selected = 0;
        e.tree_state.move_selection(20);
        assert_eq!(e.tree_state.selected, 20);

        // Past the end it clamps to the last row rather than overshooting.
        e.tree_state.move_selection(1000);
        assert_eq!(e.tree_state.selected, 99);

        // A page-sized jump up steps back, and never underflows past the top.
        e.tree_state.move_selection(-20);
        assert_eq!(e.tree_state.selected, 79);
        e.tree_state.move_selection(-1000);
        assert_eq!(e.tree_state.selected, 0);
    }

    #[test]
    fn move_to_sibling_skips_descendants_and_stops_at_the_parent_boundary() {
        let mut e = explorer_with_depths(&[0, 1, 1, 2, 1, 0]);

        // Forward from idx2 (depth 1) skips the descendant idx3 (depth 2).
        e.tree_state.selected = 2;
        e.tree_state.move_to_sibling(true);
        assert_eq!(e.tree_state.selected, 4);

        // Forward from idx4: the next row (idx5) is shallower, so no sibling.
        e.tree_state.selected = 4;
        e.tree_state.move_to_sibling(true);
        assert_eq!(e.tree_state.selected, 4);

        // Backward from idx4 lands on idx2, skipping idx3.
        e.tree_state.selected = 4;
        e.tree_state.move_to_sibling(false);
        assert_eq!(e.tree_state.selected, 2);
    }

    fn group(depth: usize, expanded: bool, child: bool) -> (TreeNode, usize) {
        let children = if child {
            vec![TreeNode::Group {
                name: String::new(),
                children: Vec::new(),
                expanded: false,
                tensor_count: 0,
                params: 0,
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
                params: 0,
                total_size: 0,
                stored_size: 0,
            },
            depth,
        )
    }

    #[test]
    fn move_to_first_child_enters_an_expanded_group() {
        let mut e = Explorer::new(Vec::new(), Vec::new(), None, false);
        // idx0: expanded group with a child; idx1: that child (depth 1);
        // idx2: a childless group at depth 0.
        e.tree_state.flattened = vec![
            group(0, true, true),
            group(1, false, false),
            group(0, false, false),
        ];

        e.tree_state.selected = 0;
        e.tree_state.move_to_first_child();
        assert_eq!(e.tree_state.selected, 1);

        // A group with no children does not move.
        e.tree_state.selected = 2;
        e.tree_state.move_to_first_child();
        assert_eq!(e.tree_state.selected, 2);
    }
}
