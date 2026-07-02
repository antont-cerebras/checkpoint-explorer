use anyhow::{Context, Result};
use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEventKind},
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

use crate::gguf::GGUFFile;
use crate::sample::{HistShared, Histogram, PackingSchema, SampleMode, Stats, ViewDtype};

use crate::tree::{
    Layout, MetadataInfo, Storage, TensorInfo, TreeBuilder, TreeNode, natural_sort_key,
};
use crate::ui::{DrawConfig, Legend, NumBase, Overlay, StatsView, StripeMode, UI};
use crate::utils::base64_encode;

/// Whether the data views show the evenly-spaced overview or the first/last
/// How a data view lays out the values it shows: an evenly-spaced overview, the
/// first/last edges (padding) sample, or a contiguous pannable window. Cycled
/// with `e`, remembered for the session; defaults to the edges view (most useful
/// for inspecting padding).
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum DataLayout {
    Overview,
    #[default]
    Edges,
    Window,
}

impl DataLayout {
    /// The next layout in the `e` cycle: Overview → Edges → Window → Overview.
    fn next(self) -> Self {
        match self {
            DataLayout::Overview => DataLayout::Edges,
            DataLayout::Edges => DataLayout::Window,
            DataLayout::Window => DataLayout::Overview,
        }
    }
}

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

/// A tensor + view to open on startup, from the CLI flags.
pub struct OpenRequest {
    /// Exact tensor name to open. `None` targets the sole tensor when the
    /// checkpoint has exactly one (so a single-tensor file — always the case for
    /// `.npy` — needs no `--tensor`); ambiguous otherwise.
    pub tensor: Option<String>,
    /// Exact metadata entry name to reveal in the tree (`--metadata`). Mutually
    /// exclusive with `tensor`; when set, the tree opens with that entry
    /// selected (metadata lives only in the tree, so there's no separate view).
    pub metadata: Option<String>,
    /// Which screen to show.
    pub view: OpenView,
    /// Show the value histogram on the detail screen (the `h` key's result).
    pub histogram: bool,
    /// Requested histogram bucket count (`--bins N`, the `b` key's result);
    /// `None` leaves the count automatic. Implies showing the histogram.
    pub bins: Option<usize>,
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
    /// Render the view once and exit without interactive navigation.
    pub exit_after: bool,
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

/// Rows the tree viewport scrolls per mouse-wheel notch (independent of the
/// selection, like a normal scrollable list).
const WHEEL_STEP: usize = 3;

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
    Detail {
        tensor: String,
        slice: usize,
    },
    Data {
        tensor: String,
        repr: Representation,
        slice: usize,
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

pub struct Explorer {
    files: Vec<PathBuf>,
    tensors: Vec<TensorInfo>,
    metadata: Vec<MetadataInfo>,
    /// Whether the whole checkpoint structure has been read. A direct
    /// `--tensor X` open reads just that tensor first (fast path), leaving this
    /// `false` until the tree is shown and the full load runs.
    full_loaded: bool,
    tree: Vec<TreeNode>,
    selected_idx: usize,
    scroll_offset: usize,
    flattened_tree: Vec<(TreeNode, usize)>,
    total_parameters: usize,
    search_query: String,
    /// Caret position within `search_query`, as a character index in `0..=len`.
    search_cursor: usize,
    search_mode: bool,
    filtered_tree: Vec<(TreeNode, usize)>,
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
    /// Index/file mismatches detected at startup, shown as a warning panel.
    health_reports: Vec<crate::health::HealthReport>,
    /// Per-tensor dtype reinterpretation chosen in the data views, keyed by
    /// tensor name. Session-scoped: remembered until the app exits.
    dtype_overrides: RefCell<HashMap<String, ViewDtype>>,
    /// Per-tensor fused-codebook packing schema parsed from metadata at load.
    /// A tensor with a schema defaults to the [`ViewDtype::Unpacked`] view.
    packing_schemas: HashMap<String, PackingSchema>,
    /// Per-tensor shape override (a reshape with the same element count) chosen
    /// in the data views with `r`, keyed by tensor name. Session-scoped.
    shape_overrides: RefCell<HashMap<String, Vec<usize>>>,
    /// Exact whole-tensor statistics, cached per (tensor name, view) since the
    /// scan is expensive. Session-scoped.
    stats_cache: RefCell<HashMap<(String, ViewDtype), Stats>>,
    /// Cached whole-tensor histograms, keyed like the stats cache plus the
    /// requested bucket count (so a different `--bins` / `b` count caches and
    /// redraws separately rather than reusing a stale layout).
    histogram_cache: RefCell<HashMap<HistKey, Histogram>>,
    /// Requested histogram bucket count (the `b` key / `--bins`); `None` lets the
    /// layout pick automatically. Session-wide, like the other view toggles.
    histogram_bins: Cell<Option<usize>>,
    /// Which layout the data views use (overview / edges / window). Session-
    /// scoped: remembered as you move between tensors and in/out of the preview.
    data_view_layout: Cell<DataLayout>,
    /// In the edges view, how the fixed row/column budget is split between the
    /// first (head) and last (tail) indices: `0.0` shows only the first, `1.0`
    /// only the last, `0.5` is balanced. Adjustable with the arrow keys and
    /// session-remembered alongside [`Self::data_view_layout`].
    data_view_row_tail: Cell<f32>,
    data_view_col_tail: Cell<f32>,
    /// The last edges-view row/column budgets actually rendered, so an arrow
    /// press can move the divider by exactly one index (step = 1 / budget).
    edge_row_budget: Cell<usize>,
    edge_col_budget: Cell<usize>,
    /// The window view's top-left corner (row/column offset into the matrix).
    /// Clamped to a valid position on every draw (read back from the rendered
    /// sample), so panning behaves at the edges. Session-remembered.
    data_view_win_row: Cell<usize>,
    data_view_win_col: Cell<usize>,
    /// The last window's visible size (rows/cols actually shown), so a
    /// `Shift`+arrow press can stride by one screenful.
    win_page_rows: Cell<usize>,
    win_page_cols: Cell<usize>,
    /// The numeric grid's zebra striping (rows / columns / off). Session-
    /// remembered; cycled with `z`.
    data_view_stripe: Cell<StripeMode>,
    /// The numeric grid's numeral base (dec / hex / oct / bin). Session-
    /// remembered; cycled with `b`.
    data_view_base: Cell<NumBase>,
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
}

impl Explorer {
    pub fn new(
        files: Vec<PathBuf>,
        health_reports: Vec<crate::health::HealthReport>,
        open: Option<OpenRequest>,
        preload: bool,
    ) -> Self {
        // Files on disk but absent from the index (per the health reports),
        // resolved to absolute paths so they match each tensor's `source_path`.
        let mut unindexed = HashSet::new();
        for report in &health_reports {
            if let Some(dir) = Path::new(&report.index_path).parent() {
                for file in &report.extra_files {
                    unindexed.insert(absolute_path(&dir.join(file)));
                }
            }
        }
        Self {
            files,
            tensors: Vec::new(),
            metadata: Vec::new(),
            full_loaded: false,
            tree: Vec::new(),
            selected_idx: 0,
            scroll_offset: 0,
            flattened_tree: Vec::new(),
            total_parameters: 0,
            search_query: String::new(),
            search_cursor: 0,
            search_mode: false,
            filtered_tree: Vec::new(),
            copied_flash: None,
            terminal: None,
            clickable: RefCell::new(Vec::new()),
            health_reports,
            dtype_overrides: RefCell::new(HashMap::new()),
            packing_schemas: HashMap::new(),
            shape_overrides: RefCell::new(HashMap::new()),
            stats_cache: RefCell::new(HashMap::new()),
            histogram_cache: RefCell::new(HashMap::new()),
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
            open,
            reader_cache: RefCell::new(None),
            sample_cache: RefCell::new(None),
            preload,
            unindexed,
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

    fn load_all_files(&mut self) -> Result<()> {
        self.tensors.clear();
        self.metadata.clear();

        // Read the checkpoint structure on a worker thread so the UI stays
        // responsive: a cold file (e.g. a large HDF5 over the network) can take
        // seconds to enumerate, and we'd otherwise show an empty screen. Animate
        // a loading frame — the same header/footer chrome as the tree, with a
        // spinner in place of the rows — until the worker finishes.
        let files = self.files.clone();
        let handle = std::thread::spawn(move || Self::gather_checkpoint(&files));

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
        let (tensors, metadata) = handle
            .join()
            .map_err(|_| anyhow::anyhow!("checkpoint loader thread panicked"))??;
        self.finalize_load(tensors, metadata);
        Ok(())
    }

    /// Read the checkpoint structure synchronously, with no loading animation —
    /// for `--plain`, which renders once to a buffer and must not write spinner
    /// frames to stdout.
    fn load_quiet(&mut self) -> Result<()> {
        self.tensors.clear();
        self.metadata.clear();
        let (tensors, metadata) = Self::gather_checkpoint(&self.files)?;
        self.finalize_load(tensors, metadata);
        Ok(())
    }

    /// Shared post-read setup: dedup, sort, parameter/schema/tree build.
    fn finalize_load(&mut self, tensors: Vec<TensorInfo>, metadata: Vec<MetadataInfo>) {
        self.tensors = tensors;
        self.metadata = metadata;

        // Deduplicate tensors by name
        let mut seen_names = HashSet::new();
        self.tensors
            .retain(|tensor| seen_names.insert(tensor.name.clone()));

        self.tensors.sort_by_key(|a| natural_sort_key(&a.name));
        self.total_parameters = self.tensors.iter().map(|t| t.num_elements).sum::<usize>();
        self.packing_schemas = crate::sample::parse_packing_schemas(&self.tensors, &self.metadata);
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
                    self.total_parameters = tensor.num_elements;
                    self.tensors = vec![tensor];
                    self.metadata = metadata;
                    self.packing_schemas =
                        crate::sample::parse_packing_schemas(&self.tensors, &self.metadata);
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
        self.dtype_overrides
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

    fn read_safetensors_file(file_path: &Path) -> Result<(Vec<TensorInfo>, Vec<MetadataInfo>)> {
        let mut tensors: Vec<TensorInfo> = Vec::new();
        let mut metadata: Vec<MetadataInfo> = Vec::new();
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
                        metadata.push(MetadataInfo {
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

            tensors.push(TensorInfo {
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

        Ok((tensors, metadata))
    }

    /// Load a NumPy `.npy` file: one array behind a small header, then raw
    /// row-major little-endian data running to EOF. The byte range is absolute
    /// (the data follows the header), and the tensor is named after the file.
    fn read_numpy_file(file_path: &Path) -> Result<(Vec<TensorInfo>, Vec<MetadataInfo>)> {
        let source_path = absolute_path(file_path);
        let mut file = File::open(file_path)
            .with_context(|| format!("Failed to open file: {}", file_path.display()))?;
        let header = crate::npy::parse_header(&mut file)
            .map_err(|e| anyhow::anyhow!("{}: {e}", file_path.display()))?;
        let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
        let name = file_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("array")
            .to_string();
        let num_elements = header.shape.iter().product::<usize>();
        let tensor = TensorInfo {
            name,
            dtype: header.dtype,
            shape: header.shape,
            size_bytes: (file_len as usize).saturating_sub(header.data_offset),
            num_elements,
            storage: Storage::Unknown,
            source_path,
            layout: Layout::ByteRange {
                start: header.data_offset as u64,
                end: file_len,
            },
        };
        Ok((vec![tensor], Vec::new()))
    }

    /// Load a NumPy `.npz` archive: a ZIP whose `<name>.npy` entries are each a
    /// `.npy` array. We read each entry's header (decompressing only that much)
    /// to list the tensors; the reader decompresses the full entry on demand.
    fn read_npz_file(file_path: &Path) -> Result<(Vec<TensorInfo>, Vec<MetadataInfo>)> {
        let mut tensors: Vec<TensorInfo> = Vec::new();
        let source_path = absolute_path(file_path);
        let file = File::open(file_path)
            .with_context(|| format!("Failed to open file: {}", file_path.display()))?;
        let mut zip = zip::ZipArchive::new(file)
            .with_context(|| format!("Failed to read .npz archive: {}", file_path.display()))?;
        let entries: Vec<String> = zip.file_names().map(String::from).collect();
        for entry_name in entries {
            let Some(name) = entry_name.strip_suffix(".npy") else {
                continue; // not an array entry
            };
            let mut entry = zip.by_name(&entry_name).with_context(|| {
                format!("Failed to read {entry_name} in {}", file_path.display())
            })?;
            let stored_bytes = entry.compressed_size() as usize;
            let uncompressed = entry.size() as usize;
            let compressed = entry.compression() != zip::CompressionMethod::Stored;
            let header = crate::npy::parse_header(&mut entry)
                .map_err(|e| anyhow::anyhow!("{}: {entry_name}: {e}", file_path.display()))?;
            let num_elements = header.shape.iter().product::<usize>();
            let storage = if compressed {
                Storage::Compressed {
                    codec: "deflate".to_string(),
                    stored_bytes,
                }
            } else {
                Storage::Raw
            };
            tensors.push(TensorInfo {
                name: name.to_string(),
                dtype: header.dtype,
                shape: header.shape,
                // Data bytes = the entry's uncompressed size minus its header.
                size_bytes: uncompressed.saturating_sub(header.data_offset),
                num_elements,
                storage,
                source_path: source_path.clone(),
                layout: Layout::None,
            });
        }
        Ok((tensors, Vec::new()))
    }

    fn read_gguf_file(file_path: &Path) -> Result<(Vec<TensorInfo>, Vec<MetadataInfo>)> {
        let mut tensors: Vec<TensorInfo> = Vec::new();
        let mut metadata: Vec<MetadataInfo> = Vec::new();
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

            metadata.push(MetadataInfo {
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

            tensors.push(TensorInfo {
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

        Ok((tensors, metadata))
    }

    #[cfg(feature = "hdf5")]
    fn read_hdf5_file(file_path: &Path) -> Result<(Vec<TensorInfo>, Vec<MetadataInfo>)> {
        // Tensors plus root-attribute metadata (version, per-tensor and config
        // `__metadata__`) from a single file open.
        crate::hdf5::read(file_path)
    }

    /// Read every file's top-level structure (tensors + metadata) into owned
    /// vectors. A free function (no `&self`) so it can run on a worker thread
    /// while the UI animates a loading spinner, and so the `diff` subcommand can
    /// load a checkpoint's structure headlessly.
    pub(crate) fn gather_checkpoint(
        files: &[PathBuf],
    ) -> Result<(Vec<TensorInfo>, Vec<MetadataInfo>)> {
        let mut tensors: Vec<TensorInfo> = Vec::new();
        let mut metadata: Vec<MetadataInfo> = Vec::new();
        for file_path in files {
            let (t, m) = match file_path.extension().and_then(|s| s.to_str()) {
                Some("safetensors") => Self::read_safetensors_file(file_path)?,
                Some("gguf") => Self::read_gguf_file(file_path)?,
                Some("npy") => Self::read_numpy_file(file_path)?,
                Some("npz") => Self::read_npz_file(file_path)?,
                Some("h5") | Some("hdf5") => {
                    #[cfg(feature = "hdf5")]
                    {
                        Self::read_hdf5_file(file_path)?
                    }
                    #[cfg(not(feature = "hdf5"))]
                    {
                        eprintln!(
                            "Warning: HDF5 support is not compiled in; rebuild with `--features hdf5` to read {}",
                            file_path.display()
                        );
                        (Vec::new(), Vec::new())
                    }
                }
                _ => {
                    eprintln!("Warning: Unsupported file format: {}", file_path.display());
                    (Vec::new(), Vec::new())
                }
            };
            tensors.extend(t);
            metadata.extend(m);
        }
        Ok((tensors, metadata))
    }

    fn build_tree(&mut self) {
        let children = if self.metadata.is_empty() {
            TreeBuilder::build_tree(&self.tensors)
        } else {
            TreeBuilder::build_tree_mixed(&self.tensors, &self.metadata)
        };
        // Everything hangs off a single root node summarising the whole
        // checkpoint (tensor count, parameters and size), so the tree reads
        // top-down from one place instead of from a separate footer.
        let total_size = self.tensors.iter().map(|t| t.size_bytes).sum();
        let stored_size = self.tensors.iter().map(|t| t.on_disk_size()).sum();
        let root = TreeNode::Group {
            name: self.root_label(),
            children,
            expanded: true,
            tensor_count: self.tensors.len(),
            params: self.total_parameters,
            total_size,
            stored_size,
        };
        self.tree = vec![root];
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

    fn flatten_tree(&mut self) {
        self.flattened_tree = TreeBuilder::flatten_tree(&self.tree);
        self.update_filtered_tree();
    }

    /// Move the tree cursor onto the leaf named `name` — a tensor or a metadata
    /// entry — expanding any collapsed groups so it's visible. Used when
    /// returning to the tree from a detail/data view, and when the app was
    /// opened with `--tensor`/`--metadata`, so you land back on that row.
    fn reveal_tensor(&mut self, name: &str) {
        if !self.search_mode {
            TreeBuilder::expand_to_tensor(&mut self.tree, name);
            self.flatten_tree();
        }
        let tree = if self.search_mode {
            &self.filtered_tree
        } else {
            &self.flattened_tree
        };
        if let Some(idx) = tree.iter().position(|(node, _)| node.name() == name) {
            self.selected_idx = idx;
        }
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
                            label: None,
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
            Some(r) => (r.histogram || r.bins.is_some(), r.compute_stats, r.bins),
            None => (false, false, None),
        };
        let want_legend = self.open.as_ref().is_some_and(|r| r.legend);
        if let Some(n) = bins {
            self.histogram_bins.set(Some(n));
        }
        let screen = match self.open.take() {
            Some(mut req) => {
                req.histogram = false;
                req.bins = None;
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
            println!("{}", self.reopen_command(&screen, want_stats, want_hist));
            return Ok(());
        }

        // The tree (and its `--legend` overlay) renders via the in-memory backend:
        // draw the tree frame, then composite the legend band on top when asked —
        // mirroring the interactive `l` path (`show_legend`).
        if matches!(screen, Screen::Tree) {
            let text = crate::tui::headless_render(120, 40, |f| {
                self.render_tree_frame(f, false); // headless: no scroll bar
                if want_legend {
                    UI::render_legend_band(f, Legend::Tree);
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
            Screen::Detail { tensor, .. } => {
                let Some(t) = self.tensors.iter().find(|t| &t.name == tensor).cloned() else {
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
                let Some(t) = self.tensors.iter().find(|t| &t.name == tensor).cloned() else {
                    return String::new();
                };
                self.command_for_data(&t, *repr, *slice)
            }
        }
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
        let Some(tensor) = self.tensors.iter().find(|t| t.name == tensor_name).cloned() else {
            return Ok(String::new());
        };
        let view = self.active_view(&tensor.name);
        let shape = self
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
        let Some(tensor) = self.tensors.iter().find(|t| t.name == tensor_name).cloned() else {
            return Ok(String::new());
        };
        let view = self.active_view(&tensor.name);
        let mode = match self.data_view_layout.get() {
            DataLayout::Edges => SampleMode::Edges {
                row_tail: self.data_view_row_tail.get(),
                col_tail: self.data_view_col_tail.get(),
            },
            DataLayout::Overview => SampleMode::Grid,
            DataLayout::Window => SampleMode::Window {
                row_off: self.data_view_win_row.get(),
                col_off: self.data_view_win_col.get(),
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
            self.data_view_stripe.get(),
            self.data_view_base.get(),
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
        let count = self.histogram_bins.get();
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

        // A `--tensor` request seeds the history with that screen — or, with
        // `--exit`, renders it once and quits without entering the navigator.
        if let Some(req) = self.open.take() {
            // Fast path: a single tensor's detail/data view doesn't need the
            // whole tree, so read just that tensor and defer the (potentially
            // slow) full structure load until the browser is actually shown.
            let fast = req.metadata.is_none()
                && !matches!(req.view, OpenView::Tree)
                && req
                    .tensor
                    .as_deref()
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

        loop {
            // The tensor the screen we're about to run belongs to (if any), so
            // that on returning to the tree we can land back on it.
            let screen_tensor = match &history[cursor] {
                Screen::Detail { tensor, .. } | Screen::Data { tensor, .. } => Some(tensor.clone()),
                Screen::Tree => None,
            };

            let nav = match history[cursor].clone() {
                Screen::Tree => self.run_tree()?,
                Screen::Detail { tensor, slice } => self.run_detail(
                    &tensor,
                    slice,
                    StatsStart::OnDemand,
                    Interaction::Interactive,
                ),
                Screen::Data {
                    tensor,
                    repr,
                    slice,
                } => {
                    // Re-record the screen with where the user left it (slice /
                    // representation), so back/forward returns there faithfully.
                    let (nav, repr, slice) =
                        self.run_data(&tensor, repr, slice, Interaction::Interactive);
                    history[cursor] = Screen::Data {
                        tensor,
                        repr,
                        slice,
                    };
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
                    self.reveal_tensor(&name);
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
        let tree_to_display = if self.search_mode {
            &self.filtered_tree
        } else {
            &self.flattened_tree
        };
        let (status_icon, status_bar, status_secondary) = self.status_bar();
        let config = DrawConfig {
            tree: tree_to_display,
            current_file: &title,
            file_idx: 0,
            total_files: 1,
            selected_idx: self.selected_idx,
            scroll_offset: self.scroll_offset,
            search_mode: self.search_mode,
            search_query: &self.search_query,
            search_cursor: self.search_cursor,
            status_icon,
            status_bar: &status_bar,
            status_secondary: &status_secondary,
            health_warning: !self.health_reports.is_empty(),
            can_repack: self.repack_input().is_some(),
            unindexed: &self.unindexed,
            packing_schemas: &self.packing_schemas,
            copied_flash: self.copied_flash.as_ref().map(|(what, _)| what.as_str()),
            interactive,
        };
        *self.clickable.borrow_mut() = UI::render_tree(frame, &config);
    }

    /// Render the tree to plain text via an in-memory Ratatui backend — the
    /// headless (`--plain`) tree and the `c` screen-copy share this.
    fn tree_plain(&self) -> Result<String> {
        crate::tui::headless_render(120, 40, |f| self.render_tree_frame(f, false))
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
        *self.clickable.borrow_mut() = UI::render_detail(
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
            None => {}
        }
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
        if self.search_mode {
            self.filtered_tree.len()
        } else {
            self.flattened_tree.len()
        }
    }

    fn update_tree_scroll(&mut self, width: u16, height: u16) {
        let body = UI::tree_visible_rows(width, height, self.search_mode, self.can_repack());
        let sel = self.selected_idx;
        self.scroll_offset = if sel >= self.scroll_offset + body {
            sel.saturating_sub(body - 1)
        } else if sel < self.scroll_offset {
            sel
        } else {
            self.scroll_offset
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
        self.copied_flash = Some((what.to_string(), std::time::Instant::now()));
    }

    /// The tree browser. Handles in-place keys (navigation, search, expand) and
    /// returns a [`Nav`] when the user opens a tensor (`Enter`), moves through
    /// the screen history (Backspace / `\`), or quits.
    fn run_tree(&mut self) -> Result<Nav> {
        // The browser needs the whole checkpoint; bring it in now if a fast
        // `--tensor` open deferred the full load.
        self.ensure_full_load()?;
        // Force a full repaint on entry so the tree fully overwrites whatever
        // screen (detail/data view, or the loading frame) preceded it.
        let mut first = true;
        // The last left-click (time, terminal row), for double-click detection.
        let mut last_click: Option<(std::time::Instant, u16)> = None;
        // The selection as of the last frame: when it changes (arrows/click) we
        // snap the viewport to keep it visible; when it's unchanged we leave the
        // scroll offset alone so the wheel can scroll freely past the selection.
        let mut last_sel = usize::MAX;
        // A wrong-keyboard-layout hint to flash on the next frame (see
        // `wrong_layout_char`); cleared as soon as the next input arrives.
        let mut layout_hint: Option<char> = None;
        // True while the left button is held after pressing on the scroll bar, so
        // subsequent drag events keep scrubbing the viewport.
        let mut scrollbar_drag = false;
        loop {
            // Skip the (relatively expensive) repaint whenever more input is
            // already queued. A burst of mouse-wheel or motion events would
            // otherwise trigger one full redraw *each*, and those repaints back
            // the input queue up: scrolling lags behind, and later key presses —
            // Ctrl-C included — are stuck behind the pile of pending redraws. By
            // draining the whole burst first and painting only once it settles,
            // scrolling stays snappy and the keyboard stays responsive.
            let input_pending = event::poll(std::time::Duration::ZERO).unwrap_or(false);
            let mut term = self
                .terminal
                .take()
                .expect("interactive loop owns the terminal");
            if first {
                term.clear()?;
                first = false;
            }
            let size = term.size().ok();
            if !input_pending {
                if let Some(sz) = size {
                    if self.selected_idx != last_sel {
                        self.update_tree_scroll(sz.width, sz.height); // snap to the moved selection
                        last_sel = self.selected_idx;
                    }
                    // Clamp the (possibly wheel-scrolled) offset to the valid range.
                    let body = UI::tree_visible_rows(
                        sz.width,
                        sz.height,
                        self.search_mode,
                        self.can_repack(),
                    );
                    let total = self.current_tree_len();
                    self.scroll_offset = self.scroll_offset.min(total.saturating_sub(body));
                }
                let hint = layout_hint;
                term.draw(|f| {
                    self.render_tree_frame(f, true);
                    if let Some(c) = hint {
                        UI::render_notice(f, &layout_hint_msg(c));
                    }
                })?;
            }
            self.terminal = Some(term);

            // While a copy confirmation is up, wake when it expires so it clears
            // on its own after `COPY_FLASH` — not only on the next key press.
            let ev = if let Some((_, at)) = &self.copied_flash {
                let remaining = COPY_FLASH.saturating_sub(at.elapsed());
                if remaining.is_zero() || !event::poll(remaining).unwrap_or(false) {
                    self.copied_flash = None;
                    continue; // expired — redraw without the confirmation
                }
                event::read()?
            } else {
                event::read()?
            };
            // Any input clears a prior layout hint; a fresh wrong-layout key re-sets
            // it below.
            layout_hint = None;

            // Mouse: a click on a footer hint chip or the `[×]` acts like its key
            // (routed through the key match below); a click on a tree row selects
            // or opens it; the wheel scrolls the viewport.
            let mut synth: Option<KeyEvent> = None;
            if let Event::Mouse(m) = &ev {
                let (kind, row, col) = (m.kind, m.row, m.column);
                match kind {
                    MouseEventKind::Down(MouseButton::Left) => {
                        // A fresh click dismisses any lingering copy confirmation;
                        // if it lands on a copy chip the key match below re-sets it.
                        // (Don't clear on the button *release* or on drag/motion —
                        // that would wipe the "Copied" a copy-chip click just set,
                        // making it flicker.)
                        self.copied_flash = None;
                        // A press on the scroll bar scrubs the viewport (like the
                        // wheel — the selection stays put); holding the button then
                        // drags it. Checked before rows/chips so its column wins.
                        scrollbar_drag = false;
                        if let Some(sz) = size
                            && let Some(sb) = UI::tree_scrollbar(
                                sz.width,
                                sz.height,
                                self.search_mode,
                                self.can_repack(),
                                self.current_tree_len(),
                            )
                            && sb.hit(col, row)
                        {
                            self.scroll_offset = sb.offset_at(row);
                            scrollbar_drag = true;
                            continue;
                        }
                        let hit = crate::ui::region_hit(&self.clickable.borrow(), col, row);
                        if let Some(k) = hit {
                            synth = Some(k); // clicked a hint chip / [×]
                        } else if let Some(sz) = size {
                            let body_top =
                                UI::tree_header_rows(sz.width, self.search_mode, self.can_repack())
                                    as u16;
                            let body_bottom = sz.height.saturating_sub(2); // status bar
                            if row >= body_top && row < body_bottom {
                                let idx = self.scroll_offset + (row - body_top) as usize;
                                if idx < self.current_tree_len() {
                                    // A click exactly on a group's ▸/▾ twisty (the
                                    // arrow sits at column `2 * depth`) toggles it on
                                    // a single click, like clicking a folder's arrow.
                                    let on_arrow = {
                                        let tree = if self.search_mode {
                                            &self.filtered_tree
                                        } else {
                                            &self.flattened_tree
                                        };
                                        matches!(
                                            tree.get(idx),
                                            Some((TreeNode::Group { .. }, depth)) if col == 2 * *depth as u16
                                        )
                                    };
                                    self.selected_idx = idx;
                                    if on_arrow {
                                        last_click = None;
                                        self.activate_selection(); // group → toggle
                                    } else {
                                        // Otherwise a lone click just selects the row
                                        // (highlight + status bar move there — visible
                                        // feedback); a double-click opens it / toggles.
                                        let double = matches!(
                                            last_click,
                                            Some((t, r)) if r == row && t.elapsed() < DOUBLE_CLICK
                                        );
                                        if double {
                                            last_click = None;
                                            if self.search_mode {
                                                self.reveal_search_result();
                                            } else if let Some(nav) = self.activate_selection() {
                                                return Ok(nav);
                                            }
                                        } else {
                                            last_click = Some((std::time::Instant::now(), row));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // Dragging after a press on the scroll bar keeps scrubbing the
                    // viewport to wherever the pointer is (clamped to the track).
                    MouseEventKind::Drag(MouseButton::Left) if scrollbar_drag => {
                        if let Some(sz) = size
                            && let Some(sb) = UI::tree_scrollbar(
                                sz.width,
                                sz.height,
                                self.search_mode,
                                self.can_repack(),
                                self.current_tree_len(),
                            )
                        {
                            self.scroll_offset = sb.offset_at(row);
                        }
                    }
                    // Releasing ends any scroll-bar drag.
                    MouseEventKind::Up(MouseButton::Left) => scrollbar_drag = false,
                    // Wheel scrolls the viewport (not the selection); the offset
                    // is clamped to range before the next draw.
                    MouseEventKind::ScrollDown => {
                        self.copied_flash = None;
                        self.scroll_offset = self.scroll_offset.saturating_add(WHEEL_STEP)
                    }
                    MouseEventKind::ScrollUp => {
                        self.copied_flash = None;
                        self.scroll_offset = self.scroll_offset.saturating_sub(WHEEL_STEP)
                    }
                    _ => {}
                }
                // A clicked hint becomes a synthesized key handled below; any other
                // mouse event was handled inline here.
                if synth.is_none() {
                    continue;
                }
            }

            // Handle a real key press, or one synthesized from a clicked hint / [×].
            let key_event = match synth {
                Some(k) => k,
                None => match ev {
                    Event::Key(k) => k,
                    _ => continue,
                },
            };
            {
                // Any key also dismisses the copy confirmation.
                self.copied_flash = None;
                // A non-Latin key (wrong layout) can't match a shortcut; outside
                // search, flash a hint instead of silently ignoring it.
                if !self.search_mode
                    && let Some(c) = wrong_layout_char(&key_event)
                {
                    layout_hint = Some(c);
                    continue;
                }
                match key_event {
                    // `q` quits only outside search; while searching it is a
                    // normal character (typed via the search Char(c) arm below),
                    // so search is left with Esc.
                    KeyEvent {
                        code: KeyCode::Char('q'),
                        ..
                    } if !self.search_mode => return Ok(Nav::Quit),
                    KeyEvent {
                        code: KeyCode::Char('c'),
                        modifiers: KeyModifiers::CONTROL,
                        ..
                    } => return Ok(Nav::Quit),
                    // `c` (no modifier) copies the screen's text; `f` copies the
                    // selected row's source File. In search mode both fall
                    // through to be typed into the query.
                    KeyEvent {
                        code: KeyCode::Char('c'),
                        ..
                    } if !self.search_mode => self.copy_tree_screen(),
                    KeyEvent {
                        code: KeyCode::Char('f'),
                        ..
                    } if !self.search_mode => self.copy_selected_path(),
                    // `n` copies the selected tensor's full name (shown in the
                    // status bar; the tree row may abbreviate it).
                    KeyEvent {
                        code: KeyCode::Char('n'),
                        ..
                    } if !self.search_mode => self.copy_selected_name(),
                    // `y` shows and copies the CLI command that reopens this view
                    // — the highlighted tensor if one is selected, else the files.
                    KeyEvent {
                        code: KeyCode::Char('y'),
                        ..
                    } if !self.search_mode => {
                        let mut term = self.terminal.take().expect("interactive loop owns it");
                        self.copy_command(&mut term, &self.command_for_tree_selection(), None);
                        self.terminal = Some(term);
                    }
                    // `h` shows the checkpoint health report (when there is one).
                    KeyEvent {
                        code: KeyCode::Char('h'),
                        ..
                    } if !self.search_mode => {
                        let mut term = self.terminal.take().expect("interactive loop owns it");
                        self.show_health_report(&mut term);
                        self.terminal = Some(term);
                    }
                    // `l` opens the legend for the tree's glyphs.
                    KeyEvent {
                        code: KeyCode::Char('l'),
                        ..
                    } if !self.search_mode => {
                        let mut term = self.terminal.take().expect("interactive loop owns it");
                        self.show_legend(&mut term, Legend::Tree, None);
                        self.terminal = Some(term);
                    }
                    // `E` / `C` expand / collapse every group at once.
                    KeyEvent {
                        code: KeyCode::Char('E'),
                        ..
                    } if !self.search_mode => self.set_all_expanded(true),
                    KeyEvent {
                        code: KeyCode::Char('C'),
                        ..
                    } if !self.search_mode => self.set_all_expanded(false),
                    // `R` repacks the current HDF5 checkpoint into a new file.
                    KeyEvent {
                        code: KeyCode::Char('R'),
                        ..
                    } if !self.search_mode => {
                        let mut term = self.terminal.take().expect("interactive loop owns it");
                        self.repack_checkpoint(&mut term);
                        self.terminal = Some(term);
                    }
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
                    // While searching, the arrows edit the query: Shift+←/→ jump
                    // the text caret to the start/end, plain ←/→ move it one
                    // character at a time.
                    KeyEvent {
                        code: KeyCode::Left,
                        modifiers: KeyModifiers::SHIFT,
                        ..
                    } if self.search_mode => self.search_cursor = 0,
                    KeyEvent {
                        code: KeyCode::Right,
                        modifiers: KeyModifiers::SHIFT,
                        ..
                    } if self.search_mode => {
                        self.search_cursor = self.search_query.chars().count();
                    }
                    KeyEvent {
                        code: KeyCode::Left,
                        ..
                    } if self.search_mode => {
                        self.search_cursor = self.search_cursor.saturating_sub(1);
                    }
                    KeyEvent {
                        code: KeyCode::Right,
                        ..
                    } if self.search_mode => {
                        self.search_cursor =
                            (self.search_cursor + 1).min(self.search_query.chars().count());
                    }
                    // In the results list Home/End jump to the first/last match
                    // and PageUp/PageDown move by one screenful.
                    KeyEvent {
                        code: KeyCode::Home,
                        ..
                    } if self.search_mode => self.selected_idx = 0,
                    KeyEvent {
                        code: KeyCode::End, ..
                    } if self.search_mode => {
                        self.selected_idx = self.filtered_tree.len().saturating_sub(1);
                    }
                    KeyEvent {
                        code: KeyCode::PageUp,
                        ..
                    } if self.search_mode => self.move_selection(-(self.page_rows() as i32)),
                    KeyEvent {
                        code: KeyCode::PageDown,
                        ..
                    } if self.search_mode => self.move_selection(self.page_rows() as i32),
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
                    // While searching, Tab jumps to the highlighted result's
                    // place in the tree (leaving search), so you can see it in
                    // context — Enter still opens its detail.
                    KeyEvent {
                        code: KeyCode::Tab, ..
                    } if self.search_mode => self.reveal_search_result(),
                    // Enter acts on the highlighted row in both modes: expand a
                    // group, or open a tensor detail (returned to the navigator).
                    KeyEvent {
                        code: KeyCode::Enter,
                        ..
                    } => {
                        if let Some(nav) = self.activate_selection() {
                            return Ok(nav);
                        }
                    }
                    KeyEvent {
                        code: KeyCode::Char(' '),
                        ..
                    } if !self.search_mode => {
                        if let Some(nav) = self.activate_selection() {
                            return Ok(nav);
                        }
                    }
                    // While searching, space is ignored rather than typed into the query.
                    KeyEvent {
                        code: KeyCode::Char(' '),
                        ..
                    } => {}
                    // Backspace edits the query while searching, otherwise steps
                    // back through the screen history.
                    KeyEvent {
                        code: KeyCode::Backspace,
                        ..
                    } if self.search_mode => self.search_backspace(),
                    KeyEvent {
                        code: KeyCode::Backspace,
                        ..
                    } => return Ok(Nav::Back),
                    // `\` steps forward through the screen history.
                    KeyEvent {
                        code: KeyCode::Char('\\'),
                        ..
                    } if !self.search_mode => return Ok(Nav::Forward),
                    KeyEvent {
                        code: KeyCode::Char(c),
                        ..
                    } if self.search_mode => self.search_insert(c),
                    // Remove left/right file navigation since we're showing all files merged
                    _ => {}
                }
            }
        }
    }

    /// Two-line status bar for the row under the cursor: a leading glyph, a
    /// primary line and a secondary line. For a tensor the primary is its full
    /// name (which the tree row may abbreviate) and the secondary is its source
    /// file; for a group the primary is its source file(s)/directory and the
    /// secondary is blank. (A copy confirmation flashes as a separate bottom-line
    /// overlay — see `copied_flash` — so it never hides this path/name.)
    fn status_bar(&self) -> (&'static str, String, String) {
        let tree = if self.search_mode {
            &self.filtered_tree
        } else {
            &self.flattened_tree
        };
        let Some((node, _)) = tree.get(self.selected_idx) else {
            return ("", String::new(), String::new());
        };

        match node {
            // The full name on the first line (the tree row often abbreviates it —
            // last segment or a compacted path), the source file on the second.
            // `n` copies the name, `f` the file.
            TreeNode::Tensor { info, .. } => ("▪", info.name.clone(), info.source_path.clone()),
            TreeNode::Group { .. } => {
                let mut files = BTreeSet::new();
                collect_source_paths(node, &mut files);
                let primary = match files.len() {
                    0 => return ("", String::new(), String::new()),
                    1 => ("▪", files.into_iter().next().unwrap()),
                    n => match common_dir(&files) {
                        // When the files share a directory, show that instead of
                        // a long list — most checkpoints live in one folder.
                        Some(dir) => ("▸", format!("{n} files in {dir}")),
                        None => {
                            let first = file_name(files.iter().next().unwrap());
                            let last = file_name(files.iter().next_back().unwrap());
                            ("▸", format!("stored across {n} files: {first} … {last}"))
                        }
                    },
                };
                (primary.0, primary.1, String::new())
            }
            // The full metadata path on the first line (the tree row shows only
            // the short `…__metadata__` label); the value preview on the second.
            TreeNode::Metadata { info } => {
                let value = info.value.split_whitespace().collect::<Vec<_>>().join(" ");
                ("†", info.name.clone(), value)
            }
        }
    }

    /// Copy the selected row's path to the clipboard (OSC 52): a tensor's file,
    /// or for a group/root the single file it lives in, else the directory its
    /// files share (so copying the root yields the file or the checkpoint dir).
    fn copy_selected_path(&mut self) {
        let Some((node, _)) = self.flattened_tree.get(self.selected_idx) else {
            return;
        };
        let path = match node {
            TreeNode::Tensor { info, .. } => Some(info.source_path.clone()),
            TreeNode::Group { .. } => {
                let mut files = BTreeSet::new();
                collect_source_paths(node, &mut files);
                match files.len() {
                    0 => None,
                    1 => files.into_iter().next(),
                    _ => common_dir(&files).or_else(|| files.into_iter().next()),
                }
            }
            TreeNode::Metadata { .. } => None,
        };
        if let Some(path) = path {
            copy_to_clipboard(&path);
            self.flash_copied("the source path");
        }
    }

    /// Copy the selected row's full name to the clipboard (the `n` shortcut): a
    /// tensor's complete name (e.g. `model.layers.0.self_attn.k_norm.weight`,
    /// which the tree may show abbreviated), or a group's path.
    fn copy_selected_name(&mut self) {
        let Some((node, _)) = self.flattened_tree.get(self.selected_idx) else {
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
        self.search_cursor = 0;
        self.update_filtered_tree();
        self.selected_idx = 0;
        self.scroll_offset = 0;
    }

    /// Open the tree in search mode with a preset query (`--search`), cursor at
    /// the end — as if the query had just been typed into the search bar.
    fn open_search(&mut self, query: &str) {
        self.search_mode = true;
        self.search_query = query.to_string();
        self.search_cursor = self.search_query.chars().count();
        self.update_filtered_tree();
        self.selected_idx = 0;
        self.scroll_offset = 0;
    }

    fn exit_search_mode(&mut self) {
        self.search_mode = false;
        self.search_query.clear();
        self.search_cursor = 0;
        self.update_filtered_tree();
        self.selected_idx = 0;
        self.scroll_offset = 0;
    }

    /// Insert a character into the query at the caret and advance past it.
    fn search_insert(&mut self, c: char) {
        let byte = self
            .search_query
            .char_indices()
            .nth(self.search_cursor)
            .map(|(b, _)| b)
            .unwrap_or(self.search_query.len());
        self.search_query.insert(byte, c);
        self.search_cursor += 1;
        self.update_filtered_tree();
        self.selected_idx = 0;
        self.scroll_offset = 0;
    }

    /// Delete the character before the caret (Backspace) and step the caret back.
    fn search_backspace(&mut self) {
        if self.search_cursor == 0 {
            return;
        }
        let byte = self
            .search_query
            .char_indices()
            .nth(self.search_cursor - 1)
            .map(|(b, _)| b)
            .unwrap_or(0);
        self.search_query.remove(byte);
        self.search_cursor -= 1;
        self.update_filtered_tree();
        self.selected_idx = 0;
        self.scroll_offset = 0;
    }

    /// Jump from the highlighted search result to that tensor's place in the
    /// tree: leave search mode, then expand to and select it so it's shown in
    /// context (a no-op if the highlighted result isn't a tensor).
    fn reveal_search_result(&mut self) {
        let name = match self.filtered_tree.get(self.selected_idx) {
            Some((TreeNode::Tensor { info, .. }, _)) => info.name.clone(),
            _ => return,
        };
        self.exit_search_mode();
        self.reveal_tensor(&name);
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
            }
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
            Some(TreeState::Expanded) => self.set_all_expanded(true),
            Some(TreeState::Collapsed) => self.set_all_expanded(false),
            None => {}
        }
        if let Some(query) = &req.search {
            self.open_search(query);
        }

        // `--metadata`: metadata lives only in the tree, so reveal that entry and
        // stay on the tree (this is what `y` on a metadata row reproduces).
        if let Some(name) = &req.metadata {
            if self.metadata.iter().any(|m| &m.name == name) {
                self.reveal_tensor(name);
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
        if req.view == OpenView::Tree && req.tensor.is_none() {
            return Ok(None);
        }
        // Resolve the target tensor: the named one, or — when `--tensor` is
        // omitted — the sole tensor if the checkpoint has exactly one (e.g. any
        // `.npy`, or a single-array `.npz`/HDF5/safetensors). Ambiguous otherwise.
        let tensor = match &req.tensor {
            Some(name) => match self.tensors.iter().find(|t| t.name == *name) {
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
            None => match self.tensors.as_slice() {
                [only] => only.clone(),
                // No `--tensor` and not a single-tensor checkpoint: ambiguous when
                // there's more than one (an error), or nothing to open at all.
                _ if self.tensors.len() > 1 => {
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
            let mut overrides = self.dtype_overrides.borrow_mut();
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
                    self.shape_overrides
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
            self.data_view_layout.set(layout);
        }
        // Position within the layout (clamped to valid bounds on the first draw).
        if let Some((row, col)) = req.window_at {
            self.data_view_win_row.set(row);
            self.data_view_win_col.set(col);
        }
        if let Some((row_tail, col_tail)) = req.edge_split {
            self.data_view_row_tail.set(row_tail);
            self.data_view_col_tail.set(col_tail);
        }
        if let Some(zebra) = req.zebra {
            self.data_view_stripe.set(zebra);
        }
        if let Some(base) = req.base {
            self.data_view_base.set(base);
        }
        // Resolve the starting slice against this tensor's slice count — the
        // leading dimension of the (possibly overridden) squeezed shape (so an
        // (N, M, 1, K) tensor pages through N slices, matching the data view);
        // 1D/2D have a single slice. Accepts an index or a percentage.
        let eff_shape = self
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
                self.reveal_tensor(&tensor.name);
                return Ok(None);
            }
        };

        // `--histogram` / `--bins`: pre-compute the value histogram so the detail
        // screen shows it on first render — this is what makes `y`'s `--histogram`
        // (and `--bins N`) round-trip restore the view. A bucket count implies
        // showing the histogram. Done before the one-shot below so `--exit`
        // captures it too.
        if let Some(n) = req.bins {
            self.histogram_bins.set(Some(n));
        }
        if (req.histogram || req.bins.is_some())
            && let Screen::Detail { .. } = screen
        {
            let view = self.active_view(&tensor.name);
            let shape = self
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
                    self.run_detail(tensor, *slice, stats_start, Interaction::OneShot);
                }
                Screen::Data {
                    tensor,
                    repr,
                    slice,
                } => {
                    self.run_data(tensor, *repr, *slice, Interaction::OneShot);
                }
                Screen::Tree => {}
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

    /// The tensor detail screen. Sub-views: `m` heatmap, `v` numeric values
    /// (returned to the navigator as a new screen), `d` reinterpret dtype, `s`
    /// compute statistics. Backspace / `\` step through the screen history; any
    /// other key goes back to the tree. Returns the chosen [`Nav`].
    fn run_detail(
        &mut self,
        tensor_name: &str,
        start_slice: usize,
        stats_start: StatsStart,
        interaction: Interaction,
    ) -> Nav {
        // Own the live Ratatui terminal for the duration of the screen (taken out
        // of `self` so the immutable-borrow draw closures can coexist with it),
        // then hand it back — mirroring `run_tree`'s take/put-back.
        let mut term = self
            .terminal
            .take()
            .expect("interactive loop owns the terminal");
        let nav = self.run_detail_loop(
            &mut term,
            tensor_name,
            start_slice,
            stats_start,
            interaction,
        );
        self.terminal = Some(term);
        nav
    }

    /// The detail screen's interactive loop, drawing through the borrowed live
    /// `term`. Split out of [`Self::run_detail`] so the terminal can be lent as a
    /// `&mut` separate from `&self` (the cached-stats / override reads go through
    /// `RefCell`, so the loop only needs `&self`).
    fn run_detail_loop(
        &self,
        term: &mut crate::tui::LiveTerminal,
        tensor_name: &str,
        start_slice: usize,
        stats_start: StatsStart,
        interaction: Interaction,
    ) -> Nav {
        let Some(tensor) = self.tensors.iter().find(|t| t.name == tensor_name).cloned() else {
            return Nav::Open(Screen::Tree);
        };
        let tensor = &tensor;
        let overridable = dtype_overridable(tensor);
        let unindexed = self.unindexed.contains(&tensor.source_path);
        // While this screen is up, compute the tensor's exact stats in the
        // background and show the scan live on the Statistics line (a spinner +
        // timer) rather than silently claiming "press s". The reduction streams
        // the tensor in bounded blocks — never holding more than one block, so
        // memory stays flat even for a multi-GB tensor — and warms the OS/disk
        // cache (the dominant cost on NFS) as a side effect, so the heatmap /
        // numeric view then opens fast; only the tiny result is kept. Dropping
        // `scan` on any exit from this screen cancels the worker, so it never
        // contends with the data-view scan we navigate into (whatever it warmed
        // stays in the OS cache). Off via `--no-preload`; skipped when
        // `--compute-stats` scans synchronously below, in one-shot mode, or for
        // formats we don't byte-read (e.g. GGUF).
        let warm = self.preload
            && stats_start != StatsStart::Auto
            && interaction == Interaction::Interactive
            && overridable;
        let mut scan: Option<ScanJob> = None;
        let mut spin_frame = 0usize;
        let mut first = true;
        // A floating pop-up (legend `l` / copied command `y`) shown over the live
        // detail frame. While it's up the loop keeps redrawing and polling, so a
        // running scan's progress animates behind it; any key dismisses it.
        let mut overlay: Option<Overlay> = None;
        // Set right after `c` copies the screen; confirmed on the bottom line
        // until the next key or `COPY_FLASH` elapses.
        let mut copied_since: Option<std::time::Instant> = None;
        // A wrong-keyboard-layout hint to flash on the next frame; cleared on the
        // next input.
        let mut layout_hint: Option<char> = None;
        loop {
            let view = self.active_view(&tensor.name);
            let shape = self
                .shape_overrides
                .borrow()
                .get(&tensor.name)
                .cloned()
                .unwrap_or_else(|| tensor.shape.clone());
            // `--compute-stats` kicks off the scan synchronously on first open,
            // animating the spinner right here; normal browsing stays fast.
            if first && stats_start == StatsStart::Auto {
                self.compute_stats_animated(term, tensor, view, |f, sv| {
                    self.render_detail_frame(
                        f,
                        tensor,
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
            // Stats: shown if cached, else — while warming — a live spinner for
            // the background scan. Switching the dtype (`d`) restarts it for the
            // new view; the result is cached the moment it lands.
            let stats = self.cached_stats(tensor, view);
            let stats_view = match &stats {
                Some(s) => {
                    scan = None;
                    StatsView::Ready(s)
                }
                None if warm => {
                    if scan.as_ref().is_none_or(|j| j.view != view) {
                        scan = Some(self.spawn_stats_scan(tensor, view));
                    }
                    let finished = scan
                        .as_ref()
                        .and_then(|j| j.handle.as_ref())
                        .is_some_and(|h| h.is_finished());
                    if finished {
                        let mut job = scan.take().unwrap();
                        if let Some(h) = job.handle.take()
                            && let Ok(Ok(s)) = h.join()
                        {
                            self.stats_cache
                                .borrow_mut()
                                .insert((tensor.name.clone(), view), s);
                        }
                        continue; // redraw with the freshly cached stats
                    }
                    // Hold off the spinner briefly so a quick scan doesn't flash it.
                    let job = scan.as_ref().unwrap();
                    if job.started.elapsed() >= std::time::Duration::from_millis(120) {
                        spin_frame = spin_frame.wrapping_add(1);
                        StatsView::Computing {
                            spinner: STATS_SPINNER[spin_frame % STATS_SPINNER.len()],
                            elapsed: job.started.elapsed(),
                            progress: job.progress(),
                        }
                    } else {
                        StatsView::Pending
                    }
                }
                None => StatsView::Pending,
            };
            // Show the value histogram below the stats once it's been computed.
            let hist = self
                .histogram_cache
                .borrow()
                .get(&(tensor.name.clone(), view, self.histogram_bins.get()))
                .cloned();
            // Force a full repaint on the first frame so the detail fully
            // overwrites whatever screen preceded it.
            if first && term.clear().is_err() {
                return Nav::Quit;
            }
            // Confirm a screen copy on the bottom line while the flash is still
            // within its window (dismissed by the next key or the timed poll) —
            // composited over the detail frame in the same draw.
            let show_flash = copied_since.is_some_and(|t| t.elapsed() < COPY_FLASH);
            let hint = layout_hint;
            if term
                .draw(|f| {
                    self.render_detail_frame(
                        f,
                        tensor,
                        &shape,
                        view,
                        overridable,
                        unindexed,
                        stats_view,
                        hist.as_ref(),
                        None,
                        overlay.as_ref(),
                    );
                    if show_flash {
                        UI::render_copied_flash(f, "screen contents");
                    }
                    if let Some(c) = hint {
                        UI::render_notice(f, &layout_hint_msg(c));
                    }
                })
                .is_err()
            {
                return Nav::Quit;
            }
            // One-shot mode: leave this frame up and exit without reading keys.
            if first && interaction == Interaction::OneShot {
                return Nav::Quit;
            }
            first = false;
            // Block for a key when idle; while the background scan runs, poll so
            // the spinner keeps ticking and a finished scan is harvested promptly.
            // Pending input pauses the scan (it releases the file lock between
            // blocks) so the keypress's own work isn't stuck behind a block read.
            let ev = if overlay.is_some() {
                // A pop-up is up: never pause the scan (the pop-up does no file
                // I/O), and poll so its progress keeps animating behind it; with
                // no scan there's nothing to animate, so just wait for the key.
                if let Some(job) = &scan {
                    job.pause.store(false, Ordering::Relaxed);
                    if event::poll(std::time::Duration::from_millis(80)).unwrap_or(false) {
                        event::read()
                    } else {
                        continue;
                    }
                } else {
                    event::read()
                }
            } else if let Some(job) = &scan {
                if event::poll(std::time::Duration::from_millis(80)).unwrap_or(false) {
                    job.pause.store(true, Ordering::Relaxed);
                    event::read()
                } else {
                    job.pause.store(false, Ordering::Relaxed);
                    continue;
                }
            } else if let Some(t) = copied_since {
                // A copy confirmation is up: wake when it expires so it can be
                // cleared without a key press.
                let remaining = COPY_FLASH.saturating_sub(t.elapsed());
                if !remaining.is_zero() && event::poll(remaining).unwrap_or(false) {
                    event::read()
                } else {
                    copied_since = None;
                    continue; // flash window elapsed — redraw without it
                }
            } else {
                event::read()
            };
            // A fresh key or click dismisses a lingering copy confirmation (`c`
            // re-sets it below). The button *release* / drag / motion that follows
            // a click must NOT, or the "Copied" the click set would flicker away.
            let fresh = match &ev {
                Ok(Event::Key(_)) => true,
                Ok(Event::Mouse(m)) => matches!(
                    m.kind,
                    MouseEventKind::Down(_) | MouseEventKind::ScrollUp | MouseEventKind::ScrollDown
                ),
                _ => false,
            };
            if fresh {
                copied_since = None;
            }
            // While a pop-up overlay is up, any key dismisses it (Ctrl-C still
            // quits) rather than acting as a screen command; the loop then
            // redraws the detail without it.
            if overlay.is_some() {
                // A key or a mouse click dismisses the pop-up; mouse motion / drag
                // / wheel is ignored so a modifier-drag to select doesn't close it.
                match &ev {
                    Ok(Event::Key(key)) => {
                        if is_ctrl_c(key) {
                            quit_immediately();
                        }
                        overlay = None;
                    }
                    Ok(Event::Mouse(m)) if matches!(m.kind, MouseEventKind::Down(_)) => {
                        overlay = None;
                    }
                    _ => {}
                }
                continue;
            }
            // A click on a footer chip / `[×]` acts like its key, routed through the
            // match below; other mouse events (wheel, release, drag, motion) do
            // nothing on the detail screen.
            let ev = match ev {
                Ok(Event::Mouse(m)) => {
                    if let MouseEventKind::Down(MouseButton::Left) = m.kind {
                        match crate::ui::region_hit(&self.clickable.borrow(), m.column, m.row) {
                            Some(k) => Ok(Event::Key(k)),
                            None => continue,
                        }
                    } else {
                        continue;
                    }
                }
                other => other,
            };
            // Any input clears a prior layout hint; a non-Latin key (wrong layout)
            // flashes the hint instead of being treated as "any other key" (which
            // would navigate back to the tree).
            layout_hint = None;
            if let Ok(Event::Key(k)) = &ev
                && let Some(c) = wrong_layout_char(k)
            {
                layout_hint = Some(c);
                continue;
            }
            match ev {
                Ok(Event::Key(key)) if is_ctrl_c(&key) => quit_immediately(),
                Ok(Event::Key(KeyEvent {
                    code: KeyCode::Char('m'),
                    ..
                })) => {
                    return Nav::Open(Screen::Data {
                        tensor: tensor.name.clone(),
                        repr: Representation::Heatmap,
                        slice: start_slice,
                    });
                }
                Ok(Event::Key(KeyEvent {
                    code: KeyCode::Char('v'),
                    ..
                })) => {
                    return Nav::Open(Screen::Data {
                        tensor: tensor.name.clone(),
                        repr: Representation::Values,
                        slice: start_slice,
                    });
                }
                // Compute the whole-tensor value histogram, animating the bars
                // filling in below the statistics.
                Ok(Event::Key(KeyEvent {
                    code: KeyCode::Char('h'),
                    ..
                })) => {
                    self.ensure_detail_histogram(
                        term,
                        tensor,
                        view,
                        &shape,
                        overridable,
                        unindexed,
                    );
                }
                // Set the histogram's bucket count (then (re)compute and show it).
                Ok(Event::Key(KeyEvent {
                    code: KeyCode::Char('b') | KeyCode::Char('B'),
                    ..
                })) => {
                    // The bins prompt floats over the live detail frame; redraw it
                    // (no overlay) as the prompt's background.
                    let background = |f: &mut ratatui::Frame| {
                        self.render_detail_frame(
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
                    };
                    let changed =
                        match self.prompt_bins(term, background, self.histogram_bins.get()) {
                            BinsChoice::Set(n) => {
                                self.histogram_bins.set(Some(n));
                                true
                            }
                            BinsChoice::Clear => {
                                self.histogram_bins.set(None);
                                true
                            }
                            BinsChoice::Cancel => false,
                        };
                    if changed {
                        self.ensure_detail_histogram(
                            term,
                            tensor,
                            view,
                            &shape,
                            overridable,
                            unindexed,
                        );
                    }
                }
                // Compute exact whole-tensor statistics on demand, animating the
                // spinner in the detail screen's Statistics line.
                Ok(Event::Key(KeyEvent {
                    code: KeyCode::Char('s') | KeyCode::Char('S'),
                    ..
                })) => {
                    self.compute_stats_animated(term, tensor, view, |f, sv| {
                        self.render_detail_frame(
                            f,
                            tensor,
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
                // Reinterpret the dtype from the detail screen too.
                Ok(Event::Key(KeyEvent {
                    code: KeyCode::Char('d') | KeyCode::Char('D'),
                    ..
                })) if overridable => {
                    if let Some(chosen) = self.prompt_dtype(term, tensor, DtypePreview::Detail) {
                        let def = self.default_view(&tensor.name);
                        let mut overrides = self.dtype_overrides.borrow_mut();
                        if chosen == def {
                            overrides.remove(&tensor.name);
                        } else {
                            overrides.insert(tensor.name.clone(), chosen);
                        }
                    }
                }
                // Reshape (shape override) from the detail screen too.
                Ok(Event::Key(KeyEvent {
                    code: KeyCode::Char('r') | KeyCode::Char('R'),
                    ..
                })) if overridable => {
                    let current = self.shape_overrides.borrow().get(&tensor.name).cloned();
                    let background = |f: &mut ratatui::Frame| {
                        self.render_detail_frame(
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
                    };
                    match self.prompt_reshape(term, background, tensor, current.as_deref()) {
                        ReshapeChoice::Set(s) => {
                            self.shape_overrides
                                .borrow_mut()
                                .insert(tensor.name.clone(), s);
                        }
                        ReshapeChoice::Clear => {
                            self.shape_overrides.borrow_mut().remove(&tensor.name);
                        }
                        ReshapeChoice::Cancel => {}
                    }
                }
                // Copy the detail screen's text to the clipboard.
                Ok(Event::Key(KeyEvent {
                    code: KeyCode::Char('c'),
                    ..
                })) => {
                    let hist = self
                        .histogram_cache
                        .borrow()
                        .get(&(tensor.name.clone(), view, self.histogram_bins.get()))
                        .cloned();
                    // Capture the screen via the same Ratatui render as the live
                    // frame (no overlay), so the copied text matches what's shown.
                    if let Ok(text) = self.detail_plain(
                        tensor,
                        &shape,
                        view,
                        overridable,
                        unindexed,
                        stats_view,
                        hist.as_ref(),
                        None,
                    ) {
                        copy_to_clipboard(&text);
                    }
                    copied_since = Some(std::time::Instant::now());
                }
                // `y` copies the CLI command and raises it as a pop-up over the
                // live frame; the loop keeps animating any running scan behind it.
                Ok(Event::Key(KeyEvent {
                    code: KeyCode::Char('y'),
                    ..
                })) => {
                    let cmd = self.command_for_detail(tensor);
                    copy_to_clipboard(&cmd);
                    overlay = Some(Overlay::Command(cmd));
                }
                // `l` raises the legend as a pop-up over the live frame (same:
                // the detail, including a running scan's progress, animates on).
                Ok(Event::Key(KeyEvent {
                    code: KeyCode::Char('l'),
                    ..
                })) => overlay = Some(Overlay::Legend(Legend::Detail)),
                // History navigation.
                Ok(Event::Key(KeyEvent {
                    code: KeyCode::Backspace,
                    ..
                })) => return Nav::Back,
                Ok(Event::Key(KeyEvent {
                    code: KeyCode::Char('\\'),
                    ..
                })) => return Nav::Forward,
                // Any other key goes back to the tree.
                Ok(Event::Key(_)) => return Nav::Open(Screen::Tree),
                Ok(_) => {} // resize etc.: just redraw the detail
                Err(_) => return Nav::Quit,
            }
        }
    }

    /// Draw the heatmap or numeric grid for the tensor, sized to the terminal.
    /// `m`/`v` switch representation in place (no trip back to the detail
    /// screen). For 3D tensors this shows one 2D slice at a fixed first index
    /// (the 0th by default); `[`/`]` and the ← → arrows step through the slices,
    /// wrapping around at both ends. Any other key returns to the detail screen.
    /// A tensor data view (heatmap or numeric grid). Handles its keys in place
    /// (slice nav, `m`/`v`, `e`, `z`, `d`); Backspace / `\` step through the
    /// screen history and any other key goes back to the detail screen. Returns
    /// the chosen [`Nav`] plus where the user left it (representation, slice) so
    /// the navigator can record it for back/forward.
    ///
    /// Owns the live Ratatui terminal for the screen's duration (taken out of
    /// `self` so the immutable-borrow draw closures can coexist with it) and hands
    /// it back — mirroring [`Self::run_detail`].
    fn run_data(
        &mut self,
        tensor_name: &str,
        repr: Representation,
        start_slice: usize,
        interaction: Interaction,
    ) -> (Nav, Representation, usize) {
        let mut term = self
            .terminal
            .take()
            .expect("interactive loop owns the terminal");
        let out = self.run_data_loop(&mut term, tensor_name, repr, start_slice, interaction);
        self.terminal = Some(term);
        out
    }

    /// The data view's interactive loop, drawing through the borrowed live `term`.
    /// Split out of [`Self::run_data`] so the terminal can be lent as a `&mut`
    /// separate from `&self` (the cached-stats / sample reads go through
    /// `RefCell`, so the loop only needs `&self`).
    fn run_data_loop(
        &self,
        term: &mut crate::tui::LiveTerminal,
        tensor_name: &str,
        repr: Representation,
        start_slice: usize,
        interaction: Interaction,
    ) -> (Nav, Representation, usize) {
        let Some(tensor) = self.tensors.iter().find(|t| t.name == tensor_name).cloned() else {
            return (Nav::Back, repr, start_slice);
        };
        let tensor = &tensor;
        let mut repr = repr;
        let mut slice = start_slice;
        // The exact-stats scan for the current `(tensor, view)`, running on a
        // worker thread so the view stays interactive while it computes; `None`
        // once the stats are cached. `spin_frame` advances the spinner.
        let mut scan: Option<ScanJob> = None;
        let mut spin_frame = 0usize;
        // Set right after `c` copies the screen; shows a confirmation on the
        // bottom line until the next key or `COPY_FLASH` elapses.
        let mut copied_since: Option<std::time::Instant> = None;
        // A floating pop-up (legend `l` / copied command `y`) shown over the live
        // data frame; while it's up the loop keeps redrawing and polling so a
        // running scan animates behind it, and any key dismisses it.
        let mut overlay: Option<Overlay> = None;
        // Force a full repaint on entry so the data view fully overwrites whatever
        // screen preceded it.
        let mut first = true;
        // A wrong-keyboard-layout hint to flash on the next frame; cleared on the
        // next input.
        let mut layout_hint: Option<char> = None;
        loop {
            // The data-view layout is a session-remembered preference, so it
            // sticks as you move between tensors and in/out of the preview.
            let mode = match self.data_view_layout.get() {
                DataLayout::Edges => SampleMode::Edges {
                    row_tail: self.data_view_row_tail.get(),
                    col_tail: self.data_view_col_tail.get(),
                },
                DataLayout::Overview => SampleMode::Grid,
                DataLayout::Window => SampleMode::Window {
                    row_off: self.data_view_win_row.get(),
                    col_off: self.data_view_win_col.get(),
                },
            };
            // The dtype reinterpretation remembered for this tensor, if any.
            let view = self.active_view(&tensor.name);

            // Exact stats power the value range, heatmap scale and stats line, but
            // the scan can take a while on a big tensor. Run it on a worker thread
            // and keep the view fully interactive while it computes: switching the
            // dtype restarts the scan for the new view, while a layout/slice/pan
            // change just re-renders (the stats are whole-tensor, layout-agnostic).
            // The result is cached the moment it lands.
            let stats = self.cached_stats(tensor, view);
            let stats_view = match &stats {
                Some(s) => {
                    scan = None;
                    StatsView::Ready(s)
                }
                None => {
                    // (Re)start the scan when none is running or it's for a stale
                    // view; dropping the old job cancels its worker.
                    if scan.as_ref().is_none_or(|j| j.view != view) {
                        scan = Some(self.spawn_stats_scan(tensor, view));
                    }
                    let finished = scan
                        .as_ref()
                        .and_then(|j| j.handle.as_ref())
                        .is_some_and(|h| h.is_finished());
                    // One-shot mode (`--exit`) has no interactivity, so just wait
                    // for the scan; interactively, harvest it once it's finished.
                    if interaction == Interaction::OneShot || finished {
                        let mut job = scan.take().unwrap();
                        if let Some(h) = job.handle.take()
                            && let Ok(Ok(s)) = h.join()
                        {
                            self.stats_cache
                                .borrow_mut()
                                .insert((tensor.name.clone(), view), s);
                        }
                        continue; // redraw with the freshly cached stats
                    }
                    // Hold off the spinner briefly so a quick scan doesn't flash it.
                    let job = scan.as_ref().unwrap();
                    if job.started.elapsed() >= std::time::Duration::from_millis(120) {
                        spin_frame = spin_frame.wrapping_add(1);
                        StatsView::Computing {
                            spinner: STATS_SPINNER[spin_frame % STATS_SPINNER.len()],
                            elapsed: job.started.elapsed(),
                            progress: job.progress(),
                        }
                    } else {
                        StatsView::Pending
                    }
                }
            };

            let stripe = self.data_view_stripe.get();
            let base = self.data_view_base.get();
            // Force a full repaint on the first frame so the data view fully
            // overwrites whatever screen preceded it.
            if first && term.clear().is_err() {
                return (Nav::Quit, repr, slice);
            }
            // Confirm a screen copy on the bottom line while the flash is still
            // within its window — composited over the data frame in the same draw.
            let show_flash = copied_since.is_some_and(|t| t.elapsed() < COPY_FLASH);
            let hint = layout_hint;
            // Size the grid to the live terminal (the same size the draw below
            // renders into); fall back to a sane default if it can't be read.
            let (cols, rows) = term
                .size()
                .map(|s| (s.width, s.height))
                .unwrap_or((100, 40));
            // (slices, overridable, clamped slice) on success — sample once, then
            // draw the cached result through Ratatui (overlay composited last).
            let (slices, overridable) = match self
                .prepare_data_sample(tensor, repr, slice, view, mode, stats_view, cols, rows)
            {
                Ok((slices, overridable, clamped)) => {
                    slice = clamped;
                    if term
                        .draw(|f| {
                            let cache = self.sample_cache.borrow();
                            let sample = &cache.as_ref().unwrap().sample;
                            *self.clickable.borrow_mut() = match repr {
                                Representation::Heatmap => {
                                    UI::render_heatmap(f, tensor, sample, stats_view)
                                }
                                Representation::Values => {
                                    UI::render_values(f, tensor, sample, stats_view, stripe, base)
                                }
                            };
                            match overlay.as_ref() {
                                Some(Overlay::Legend(l)) => UI::render_legend_band(f, *l),
                                Some(Overlay::Command(c)) => UI::render_command_band(f, c),
                                None => {}
                            }
                            if show_flash {
                                UI::render_copied_flash(f, "screen contents");
                            }
                            if let Some(c) = hint {
                                UI::render_notice(f, &layout_hint_msg(c));
                            }
                        })
                        .is_err()
                    {
                        return (Nav::Quit, repr, slice);
                    }
                    (slices, overridable)
                }
                Err(msg) => {
                    let _ = term.draw(|f| UI::render_message(f, "Data preview unavailable", &msg));
                    if interaction == Interaction::Interactive
                        && let Ok(Event::Key(key)) = event::read()
                        && is_ctrl_c(&key)
                    {
                        quit_immediately();
                    }
                    return (
                        Nav::Open(Screen::Detail {
                            tensor: tensor.name.clone(),
                            slice,
                        }),
                        repr,
                        slice,
                    );
                }
            };

            first = false;

            // One-shot mode (`--exit`): stats are computed above and the final
            // frame is now drawn, so leave it up and exit without reading keys.
            if interaction == Interaction::OneShot {
                return (Nav::Quit, repr, slice);
            }

            // Read one event, then coalesce any buffered follow-ups (an arrow
            // key's auto-repeat) before redrawing. Each redraw re-samples the
            // tensor — slower than the key-repeat rate — so without draining, held
            // keys pile up and the separator keeps "coasting" through the backlog
            // after release. Applying the whole burst, then redrawing once, keeps
            // it smooth and stops the moment the key lifts. While a scan is
            // running we poll instead of blocking, so the spinner keeps ticking
            // and a finished scan is harvested promptly.
            let mut pending = if overlay.is_some() {
                // A pop-up is up: never pause the scan (the pop-up does no file
                // I/O), and poll so its progress keeps animating behind it; with
                // no scan there's nothing to animate, so just wait for the key.
                if let Some(job) = &scan {
                    job.pause.store(false, Ordering::Relaxed);
                    if event::poll(std::time::Duration::from_millis(80)).unwrap_or(false) {
                        event::read()
                    } else {
                        continue;
                    }
                } else {
                    event::read()
                }
            } else if let Some(job) = &scan {
                if event::poll(std::time::Duration::from_millis(80)).unwrap_or(false) {
                    // Input pending — give the foreground priority: pause the scan
                    // (it releases the HDF5 lock between blocks) so the re-sample
                    // this keypress triggers isn't stuck behind a long block read,
                    // which is what made dtype/layout changes lag and buffer keys.
                    // It resumes the moment input goes idle (the else branch).
                    job.pause.store(true, Ordering::Relaxed);
                    event::read()
                } else {
                    job.pause.store(false, Ordering::Relaxed);
                    continue; // idle — let the scan run; spinner reuses the cache
                }
            } else if let Some(t) = copied_since {
                // A copy confirmation is up: wake when it expires so it can be
                // cleared without a key press.
                let remaining = COPY_FLASH.saturating_sub(t.elapsed());
                if !remaining.is_zero() && event::poll(remaining).unwrap_or(false) {
                    event::read()
                } else {
                    copied_since = None;
                    continue; // flash window elapsed — redraw without it
                }
            } else {
                event::read()
            };
            // While a pop-up overlay is up, any key dismisses it (Ctrl-C still
            // quits) rather than acting as a screen command; the loop then redraws
            // the data view without it.
            if overlay.is_some() {
                // A key or a mouse click dismisses the pop-up; mouse motion / drag
                // / wheel is ignored so a modifier-drag to select doesn't close it.
                match &pending {
                    Ok(Event::Key(key)) => {
                        if is_ctrl_c(key) {
                            quit_immediately();
                        }
                        overlay = None;
                    }
                    Ok(Event::Mouse(m)) if matches!(m.kind, MouseEventKind::Down(_)) => {
                        overlay = None;
                    }
                    _ => {}
                }
                continue;
            }
            loop {
                // Any input clears a prior layout hint; a fresh wrong-layout key
                // re-sets it below.
                layout_hint = None;
                match pending {
                    Ok(Event::Key(key)) if is_ctrl_c(&key) => quit_immediately(),
                    // A non-Latin key (wrong layout) can't match a shortcut; flash a
                    // hint instead of treating it as "any other key" (→ back to detail).
                    Ok(Event::Key(key)) if wrong_layout_char(&key).is_some() => {
                        layout_hint = wrong_layout_char(&key);
                        break;
                    }
                    Ok(Event::Key(KeyEvent {
                        code, modifiers, ..
                    })) => {
                        // Any key dismisses a lingering copy confirmation; `c`
                        // sets it again below.
                        copied_since = None;
                        let shift = modifiers.contains(KeyModifiers::SHIFT);
                        let ctrl = modifiers.contains(KeyModifiers::CONTROL);
                        let edges = matches!(mode, SampleMode::Edges { .. });
                        let window = matches!(mode, SampleMode::Window { .. });
                        // One arrow press moves the divider by a single index
                        // (step = 1 / budget); Shift snaps the split to one end.
                        let nudge = |cell: &Cell<f32>, toward_tail: bool, budget: usize| {
                            let step = if shift {
                                1.0
                            } else {
                                1.0 / budget.max(1) as f32
                            };
                            let delta = if toward_tail { step } else { -step };
                            cell.set((cell.get() + delta).clamp(0.0, 1.0));
                        };
                        // Pan the window along one axis. Ctrl jumps fully to an
                        // edge (`usize::MAX` is clamped back to the last position
                        // on the next draw), Shift strides one screenful, a plain
                        // arrow moves a single row/column.
                        let pan = |cell: &Cell<usize>, forward: bool, page: usize| {
                            let cur = cell.get();
                            let next = if ctrl {
                                if forward { usize::MAX } else { 0 }
                            } else {
                                let step = if shift { page.max(1) } else { 1 };
                                if forward {
                                    cur.saturating_add(step)
                                } else {
                                    cur.saturating_sub(step)
                                }
                            };
                            cell.set(next);
                        };
                        match code {
                            // Switch representation in place, keeping the current slice.
                            KeyCode::Char('m') => repr = Representation::Heatmap,
                            KeyCode::Char('v') => repr = Representation::Values,
                            // Cycle the data-view layout overview → edges → window
                            // → overview; remembered for the session.
                            KeyCode::Char('e') | KeyCode::Char('E') => self
                                .data_view_layout
                                .set(self.data_view_layout.get().next()),
                            // Cycle the numeric grid's zebra striping rows → cols →
                            // off; remembered for the session.
                            KeyCode::Char('z') | KeyCode::Char('Z') => self
                                .data_view_stripe
                                .set(self.data_view_stripe.get().next()),
                            // Cycle the numeral base dec → hex → oct → bin
                            // (numeric grid); remembered for the session.
                            KeyCode::Char('b') | KeyCode::Char('B') => {
                                self.data_view_base.set(self.data_view_base.get().next())
                            }
                            // In the edges view the arrows move the divider between
                            // the first and last blocks (Shift pushes it fully to one
                            // end): e.g. `→` slides the column divider right, growing
                            // the first columns and shrinking the last; `↓` slides the
                            // row divider down. They take precedence over slice
                            // stepping, which stays on `[` / `]` and `/` while edges
                            // is active.
                            KeyCode::Up if edges => {
                                nudge(&self.data_view_row_tail, true, self.edge_row_budget.get())
                            }
                            KeyCode::Down if edges => {
                                nudge(&self.data_view_row_tail, false, self.edge_row_budget.get())
                            }
                            KeyCode::Left if edges => {
                                nudge(&self.data_view_col_tail, true, self.edge_col_budget.get())
                            }
                            KeyCode::Right if edges => {
                                nudge(&self.data_view_col_tail, false, self.edge_col_budget.get())
                            }
                            // In the window view the arrows pan the visible block
                            // (Shift strides a screenful; Ctrl+arrow also jumps to
                            // an edge on terminals that send it); slice stepping
                            // stays on `[` / `]` and `/`.
                            KeyCode::Up if window => {
                                pan(&self.data_view_win_row, false, self.win_page_rows.get())
                            }
                            KeyCode::Down if window => {
                                pan(&self.data_view_win_row, true, self.win_page_rows.get())
                            }
                            KeyCode::Left if window => {
                                pan(&self.data_view_win_col, false, self.win_page_cols.get())
                            }
                            KeyCode::Right if window => {
                                pan(&self.data_view_win_col, true, self.win_page_cols.get())
                            }
                            // Jump straight to an edge (clamped on the next draw):
                            // Home/End to the first/last column, PageUp/PageDown to
                            // the first/last row. Plain navigation keys, so they
                            // work everywhere (unlike Ctrl+arrow).
                            KeyCode::Home if window => self.data_view_win_col.set(0),
                            KeyCode::End if window => self.data_view_win_col.set(usize::MAX),
                            KeyCode::PageUp if window => self.data_view_win_row.set(0),
                            KeyCode::PageDown if window => self.data_view_win_row.set(usize::MAX),
                            // Open the dtype menu; `d` or `D`. The scan keeps
                            // running (paused while input flows, see the event
                            // wait below), so its live previews read uncontended
                            // and an accidental press you `Esc` out of never throws
                            // away a long computation. Picking a *different* dtype
                            // changes the view, which restarts the scan for it.
                            KeyCode::Char('d') | KeyCode::Char('D') if overridable => {
                                if let Some(chosen) = self.prompt_dtype(
                                    term,
                                    tensor,
                                    DtypePreview::Data { repr, slice, mode },
                                ) {
                                    let def = self.default_view(&tensor.name);
                                    let mut overrides = self.dtype_overrides.borrow_mut();
                                    if chosen == def {
                                        overrides.remove(&tensor.name);
                                    } else {
                                        overrides.insert(tensor.name.clone(), chosen);
                                    }
                                }
                            }
                            // Reshape: reinterpret the dimensions (`r`). The new
                            // shape must have the same element count; an empty
                            // entry clears the override. Reset the slice since the
                            // slice count can change.
                            KeyCode::Char('r') | KeyCode::Char('R') if overridable => {
                                let current =
                                    self.shape_overrides.borrow().get(&tensor.name).cloned();
                                // The prompt floats over the current data view; redraw
                                // it from the cached sample as the prompt's background.
                                let background = |f: &mut ratatui::Frame| {
                                    self.render_cached_data(
                                        f, tensor, repr, stats_view, stripe, base,
                                    );
                                };
                                match self.prompt_reshape(
                                    term,
                                    background,
                                    tensor,
                                    current.as_deref(),
                                ) {
                                    ReshapeChoice::Set(s) => {
                                        self.shape_overrides
                                            .borrow_mut()
                                            .insert(tensor.name.clone(), s);
                                        slice = 0;
                                    }
                                    ReshapeChoice::Clear => {
                                        self.shape_overrides.borrow_mut().remove(&tensor.name);
                                        slice = 0;
                                    }
                                    ReshapeChoice::Cancel => {}
                                }
                            }
                            // Jump straight to a slice by typing its index.
                            KeyCode::Char('/') if slices > 1 => {
                                let background = |f: &mut ratatui::Frame| {
                                    self.render_cached_data(
                                        f, tensor, repr, stats_view, stripe, base,
                                    );
                                };
                                if let Some(n) = self.prompt_slice(term, background, slices) {
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
                            // Copy the data view's text to the clipboard (the same
                            // Ratatui render the `--plain` path emits).
                            KeyCode::Char('c') => {
                                if let Ok(text) = self.data_plain(
                                    tensor, repr, slice, view, mode, stats_view, stripe, base, None,
                                ) {
                                    copy_to_clipboard(&text);
                                }
                                copied_since = Some(std::time::Instant::now());
                            }
                            // `y` copies the CLI command that reopens this exact
                            // view and floats it over the live frame as a Ratatui
                            // pop-up; the scan keeps running behind it.
                            KeyCode::Char('y') => {
                                let cmd = self.command_for_data(tensor, repr, slice);
                                copy_to_clipboard(&cmd);
                                overlay = Some(Overlay::Command(cmd));
                            }
                            // Open the legend for this representation as a pop-up
                            // band over the live data view; the background stats
                            // scan keeps running while it's up.
                            KeyCode::Char('l') => {
                                overlay = Some(Overlay::Legend(match repr {
                                    Representation::Heatmap => Legend::Heatmap,
                                    Representation::Values => Legend::Values,
                                }));
                            }
                            // History navigation: Backspace back, `\` forward.
                            KeyCode::Backspace => return (Nav::Back, repr, slice),
                            KeyCode::Char('\\') => return (Nav::Forward, repr, slice),
                            // Any other key goes back to the detail screen.
                            _ => {
                                return (
                                    Nav::Open(Screen::Detail {
                                        tensor: tensor.name.clone(),
                                        slice,
                                    }),
                                    repr,
                                    slice,
                                );
                            }
                        }
                    }
                    Ok(Event::Mouse(m)) => match m.kind {
                        // A click on a footer chip / `[×]` acts like its key: feed
                        // the synthesized key back through this match.
                        MouseEventKind::Down(MouseButton::Left) => {
                            if let Some(k) =
                                crate::ui::region_hit(&self.clickable.borrow(), m.column, m.row)
                            {
                                pending = Ok(Event::Key(k));
                                continue;
                            }
                        }
                        // The wheel pages through the slices (like `]` / `[`).
                        MouseEventKind::ScrollDown if slices > 1 => slice = (slice + 1) % slices,
                        MouseEventKind::ScrollUp if slices > 1 => {
                            slice = (slice + slices - 1) % slices
                        }
                        _ => {}
                    },
                    Ok(_) => {} // resize etc.: re-sample and redraw the same slice
                    Err(_) => return (Nav::Back, repr, slice),
                }
                // Drain the next buffered event without blocking; once the queue
                // is empty, fall out to redraw exactly once for the whole burst.
                // A just-opened pop-up stops the drain so the next iteration shows
                // it (and any further keys dismiss it rather than acting as commands).
                if overlay.is_none() && event::poll(std::time::Duration::ZERO).unwrap_or(false) {
                    pending = event::read();
                } else {
                    break;
                }
            }
        }
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
        let need_stats =
            crate::sample::histogram_bins(view, &tensor.dtype, range, self.histogram_bins.get())
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
        let count = self.histogram_bins.get();
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
            let overrides = self.shape_overrides.borrow();
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
            self.data_view_stripe.get(),
            self.data_view_base.get(),
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
                let cell =
                    self.data_view_base
                        .get()
                        .cell_width(view, &tensor.dtype, stats.value_range());
                (text_rows, ((cols as usize).saturating_sub(7) / cell).max(1))
            }
        };
        // Remember the edges-view budgets so an arrow press can move the divider
        // by exactly one index (step = 1 / budget).
        self.edge_row_budget
            .set(crate::sample::edge_total(max_rows));
        self.edge_col_budget
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
                self.data_view_win_row
                    .set(sample.rows.first().copied().unwrap_or(0));
                self.data_view_win_col
                    .set(sample.cols.first().copied().unwrap_or(0));
                self.win_page_rows.set(sample.rows.len().max(1));
                self.win_page_cols.set(sample.cols.len().max(1));
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
            .shape_overrides
            .borrow()
            .get(&tensor.name)
            .cloned()
            .unwrap_or_else(|| tensor.shape.clone());
        let stripe = self.data_view_stripe.get();
        let base = self.data_view_base.get();
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

    fn show_health_report(&self, term: &mut crate::tui::LiveTerminal) {
        if !self.health_reports.is_empty()
            && term
                .draw(|f| UI::render_health_warning(f, &self.health_reports))
                .is_ok()
        {
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
        // so the tree stays visible behind it).
        if term
            .draw(|f| {
                self.render_tree_frame(f, true);
                UI::render_legend_band(f, legend);
            })
            .is_ok()
        {
            wait_for_dismiss();
        }
    }

    /// Copy `command` to the clipboard and show it in a dismissible box, so the
    /// user can both see and paste the exact invocation that reopens this screen
    /// (the `y` shortcut). Any key returns; Ctrl-C still quits. `resume` keeps a
    /// caller-paused background scan running while the box is up (see
    /// [`Self::show_legend`]).
    fn copy_command(
        &self,
        term: &mut crate::tui::LiveTerminal,
        command: &str,
        resume: Option<&AtomicBool>,
    ) {
        copy_to_clipboard(command);
        if let Some(pause) = resume {
            pause.store(false, Ordering::Relaxed);
        }
        // Float the CLI-command band over a fresh tree frame (composited last so
        // the tree stays visible behind it).
        if term
            .draw(|f| {
                self.render_tree_frame(f, true);
                UI::render_command_band(f, command);
            })
            .is_ok()
        {
            // Wait for a key (mouse events are ignored) so the command text can be
            // selected with the mouse without the pop-up closing.
            wait_for_dismiss();
        }
    }

    /// The path argument(s) that reopen this checkpoint the way it was launched:
    /// a single file as-is, or — when every loaded file lives in one directory (a
    /// sharded checkpoint opened as a folder) — that directory, so the command
    /// references the checkpoint rather than an arbitrary shard; otherwise the
    /// individual files.
    fn checkpoint_path_parts(&self) -> Vec<String> {
        let quote = |p: &Path| shell_quote(&p.to_string_lossy());
        match self.files.as_slice() {
            [] => Vec::new(),
            [one] => vec![quote(one)],
            many => {
                let set: BTreeSet<String> = many
                    .iter()
                    .map(|p| p.to_string_lossy().into_owned())
                    .collect();
                match common_dir(&set) {
                    Some(dir) => vec![shell_quote(&dir)],
                    None => many.iter().map(|p| quote(p)).collect(),
                }
            }
        }
    }

    /// The command that reopens the current tree: the program and the file/dir
    /// arguments it was launched with.
    fn command_for_tree(&self) -> String {
        let mut parts = vec![PROGRAM.to_string()];
        parts.extend(self.checkpoint_path_parts());
        parts.extend(self.tree_state_args());
        parts.join(" ")
    }

    /// `--tree-state` args reproducing the current bulk expansion (`expanded` /
    /// `collapsed`), or nothing for the default / a mixed (per-group) state. The
    /// shared tail for the tree's reopen commands so `E` / `C` round-trip.
    fn tree_state_args(&self) -> Vec<String> {
        let state = if TreeBuilder::all_groups(&self.tree, true) {
            Some(TreeState::Expanded)
        } else if TreeBuilder::all_groups(&self.tree, false) {
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
        match self.flattened_tree.get(self.selected_idx) {
            Some((TreeNode::Tensor { info, .. }, _)) => {
                let mut parts = self.command_base(info);
                parts.push("--tree".to_string());
                parts.extend(self.tree_state_args());
                parts.join(" ")
            }
            // A metadata row reopens the tree with that entry revealed.
            Some((TreeNode::Metadata { info }, _)) => {
                let mut parts = vec![PROGRAM.to_string()];
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
        let mut parts = vec![PROGRAM.to_string()];
        parts.extend(self.checkpoint_path_parts());
        parts.push("--tensor".to_string());
        parts.push(shell_quote(&tensor.name));
        if let Some(dt) = self.dtype_overrides.borrow().get(&tensor.name)
            && let Some(value) = dt.cli_value()
        {
            parts.push("--dtype".to_string());
            parts.push(value);
        }
        if let Some(shape) = self.shape_overrides.borrow().get(&tensor.name) {
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
        let bins = self.histogram_bins.get();
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
        parts.push(match self.data_view_layout.get() {
            DataLayout::Overview => "--overview".to_string(),
            DataLayout::Edges => {
                let (rt, ct) = (self.data_view_row_tail.get(), self.data_view_col_tail.get());
                if rt == 0.5 && ct == 0.5 {
                    "--edge".to_string()
                } else {
                    format!("--edge={rt},{ct}")
                }
            }
            DataLayout::Window => {
                let (row, col) = (self.data_view_win_row.get(), self.data_view_win_col.get());
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
            let zebra = match self.data_view_stripe.get() {
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
            let base = self.data_view_base.get();
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

/// The directory shared by all `paths`, or `None` if they don't all share one.
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
        e.flattened_tree = depths
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
