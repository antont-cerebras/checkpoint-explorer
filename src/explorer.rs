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
    cell::{Cell, RefCell},
    collections::{BTreeSet, HashMap, HashSet},
    fs::File,
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

use crate::gguf::GGUFFile;
use crate::sample::{SampleMode, Stats, ViewDtype};

use crate::tree::{
    Layout, MetadataInfo, Storage, TensorInfo, TreeBuilder, TreeNode, natural_sort_key,
};
use crate::ui::{DrawConfig, Legend, StatsView, StripeMode, UI};
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

/// A tensor + view to open on startup, from the CLI flags.
pub struct OpenRequest {
    /// Exact tensor name to open. `None` targets the sole tensor when the
    /// checkpoint has exactly one (so a single-tensor file — always the case for
    /// `.npy` — needs no `--tensor`); ambiguous otherwise.
    pub tensor: Option<String>,
    /// Which screen to show.
    pub view: OpenView,
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
    /// Optional starting slice (3D tensors), as a raw `N` or `N%` string
    /// resolved against the tensor's slice count.
    pub slice: Option<String>,
    /// Optional shape override (a reshape with a matching element count), as a
    /// raw string like `10,100` or `-1,768`.
    pub shape: Option<String>,
    /// Start the statistics scan immediately on the detail view.
    pub compute_stats: bool,
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

/// Whether a screen waits for keys or renders once and returns (`--exit`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Interaction {
    Interactive,
    OneShot,
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
    /// Per-tensor dtype reinterpretation chosen in the data views, keyed by
    /// tensor name. Session-scoped: remembered until the app exits.
    dtype_overrides: RefCell<HashMap<String, ViewDtype>>,
    /// Per-tensor shape override (a reshape with the same element count) chosen
    /// in the data views with `r`, keyed by tensor name. Session-scoped.
    shape_overrides: RefCell<HashMap<String, Vec<usize>>>,
    /// Exact whole-tensor statistics, cached per (tensor name, view) since the
    /// scan is expensive. Session-scoped.
    stats_cache: RefCell<HashMap<(String, ViewDtype), Stats>>,
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
            dtype_overrides: RefCell::new(HashMap::new()),
            shape_overrides: RefCell::new(HashMap::new()),
            stats_cache: RefCell::new(HashMap::new()),
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
        let worker_cancel = Arc::clone(&cancel);
        let worker_pause = Arc::clone(&pause);
        let worker_done = Arc::clone(&done);
        let handle = std::thread::spawn(move || {
            crate::sample::tensor_stats(
                &owned,
                view,
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
        tensor: &TensorInfo,
        view: ViewDtype,
        mut redraw: impl FnMut(StatsView),
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
        let worker_cancel = Arc::clone(&cancel);
        let worker_pause = Arc::clone(&pause);
        let worker_done = Arc::clone(&done);
        let handle = std::thread::spawn(move || {
            crate::sample::tensor_stats(
                &owned,
                view,
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
                redraw(StatsView::Computing {
                    spinner: SPINNER[frame % SPINNER.len()],
                    elapsed: started.elapsed(),
                    progress: (total > 0)
                        .then(|| (done.load(Ordering::Relaxed) as f64 / total as f64).min(1.0)),
                });
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
                let _ = UI::draw_message("Statistics unavailable", &msg);
                let _ = event::read();
                ScanOutcome::Completed
            }
            Err(_) => {
                let _ = UI::draw_message("Statistics unavailable", "the scan thread panicked");
                let _ = event::read();
                ScanOutcome::Completed
            }
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
                Some("npy") => {
                    self.load_numpy_file(file_path)?;
                }
                Some("npz") => {
                    self.load_npz_file(file_path)?;
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

    /// Load a NumPy `.npy` file: one array behind a small header, then raw
    /// row-major little-endian data running to EOF. The byte range is absolute
    /// (the data follows the header), and the tensor is named after the file.
    fn load_numpy_file(&mut self, file_path: &PathBuf) -> Result<()> {
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
        self.tensors.push(TensorInfo {
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
        });
        Ok(())
    }

    /// Load a NumPy `.npz` archive: a ZIP whose `<name>.npy` entries are each a
    /// `.npy` array. We read each entry's header (decompressing only that much)
    /// to list the tensors; the reader decompresses the full entry on demand.
    fn load_npz_file(&mut self, file_path: &PathBuf) -> Result<()> {
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
            self.tensors.push(TensorInfo {
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

    /// Move the tree cursor onto the tensor named `name`, expanding any
    /// collapsed groups so it's visible. Used when returning to the tree from a
    /// tensor's detail/data view (and when the app was opened with `--tensor`),
    /// so you land back on the tensor you were looking at.
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

    pub fn run(&mut self) -> Result<()> {
        if self.files.is_empty() {
            return Ok(());
        }

        terminal::enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, terminal::Clear(ClearType::All), cursor::Hide)?;

        let result = self.interactive_loop();

        // Leave the last rendered frame on screen — don't clear it — and drop the
        // shell prompt onto a fresh line just below it. This keeps whatever you
        // were looking at visible after you quit (and lets `--exit` output be
        // read / captured). The newline lands the prompt at the bottom left;
        // disabling raw mode first so its `\n` becomes a CR+LF (column 0).
        execute!(stdout, cursor::Show)?;
        terminal::disable_raw_mode()?;
        println!();

        result
    }

    fn interactive_loop(&mut self) -> Result<()> {
        self.load_all_files()?;

        // Browser-style screen history: Backspace steps back through the screens
        // you've visited, `\` steps forward, and any fresh navigation truncates
        // the forward tail. The tree is the root.
        let mut history = vec![Screen::Tree];
        let mut cursor = 0usize;

        // A `--tensor` request seeds the history with that screen — or, with
        // `--exit`, renders it once and quits without entering the navigator.
        if let Some(req) = self.open.take() {
            let one_shot = req.exit_after;
            let seeded = self.open_requested(req);
            if one_shot {
                return Ok(());
            }
            if let Some(screen) = seeded {
                history.push(screen);
                cursor = 1;
            }
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
            // tensor (revealing it) so you land where you were.
            if matches!(history[cursor], Screen::Tree)
                && let Some(name) = screen_tensor
            {
                self.reveal_tensor(&name);
            }
        }

        Ok(())
    }

    /// Render the tree browser frame into `out`, returning the (possibly
    /// adjusted) scroll offset. Shared by the live loop and screen-copy.
    fn draw_tree(&self, out: &mut impl Write) -> Result<usize> {
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
        let (status_icon, status_ok, status_bar, status_secondary) = self.status_bar();
        let config = DrawConfig {
            tree: tree_to_display,
            current_file: &title,
            file_idx: 0,
            total_files: 1,
            selected_idx: self.selected_idx,
            scroll_offset: self.scroll_offset,
            search_mode: self.search_mode,
            search_query: &self.search_query,
            status_icon,
            status_ok,
            status_bar: &status_bar,
            status_secondary: &status_secondary,
            health_warning: !self.health_reports.is_empty(),
            can_repack: self.repack_input().is_some(),
            unindexed: &self.unindexed,
        };
        UI::draw_screen(out, &config)
    }

    /// Copy the current tree screen's text to the clipboard (the `c` shortcut).
    fn copy_tree_screen(&mut self) {
        let text = screen_text(|buf| {
            let _ = self.draw_tree(buf);
        });
        copy_to_clipboard(&text);
        self.copied_flash = Some("screen contents".to_string());
    }

    /// The tree browser. Handles in-place keys (navigation, search, expand) and
    /// returns a [`Nav`] when the user opens a tensor (`Enter`), moves through
    /// the screen history (Backspace / `\`), or quits.
    fn run_tree(&mut self) -> Result<Nav> {
        loop {
            self.scroll_offset = self.draw_tree(&mut live_out())?;

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
                            return Ok(Nav::Quit);
                        }
                    }
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
                    } if !self.search_mode => self.copy_command(&self.command_for_tree_selection()),
                    // `h` shows the checkpoint health report (when there is one).
                    KeyEvent {
                        code: KeyCode::Char('h'),
                        ..
                    } if !self.search_mode => self.show_health_report(),
                    // `l` opens the legend for the tree's glyphs.
                    KeyEvent {
                        code: KeyCode::Char('l'),
                        ..
                    } if !self.search_mode => self.show_legend(Legend::Tree),
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
                    } if !self.search_mode => self.repack_checkpoint(),
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
                        if let Some(screen) = self.handle_selection() {
                            return Ok(Nav::Open(screen));
                        }
                    }
                    KeyEvent {
                        code: KeyCode::Char(' '),
                        ..
                    } if !self.search_mode => {
                        if let Some(screen) = self.handle_selection() {
                            return Ok(Nav::Open(screen));
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
                    } if self.search_mode => {
                        self.search_query.pop();
                        self.update_filtered_tree();
                        self.selected_idx = 0;
                        self.scroll_offset = 0;
                    }
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
    }

    /// Two-line status bar for the row under the cursor: a leading glyph, an
    /// "is this a success message" flag (the copy confirmation), a primary line
    /// and a secondary line. For a tensor the primary is its full name (which the
    /// tree row may abbreviate) and the secondary is its source file; for a group
    /// the primary is its source file(s)/directory and the secondary is blank;
    /// the copy confirmation occupies the primary line alone.
    fn status_bar(&self) -> (&'static str, bool, String, String) {
        if let Some(flash) = &self.copied_flash {
            return ("✓", true, format!("Copied: {flash}"), String::new());
        }

        let tree = if self.search_mode {
            &self.filtered_tree
        } else {
            &self.flattened_tree
        };
        let Some((node, _)) = tree.get(self.selected_idx) else {
            return ("", false, String::new(), String::new());
        };

        match node {
            // The full name on the first line (the tree row often abbreviates it —
            // last segment or a compacted path), the source file on the second.
            // `n` copies the name, `f` the file.
            TreeNode::Tensor { info, .. } => {
                ("▪", false, info.name.clone(), info.source_path.clone())
            }
            TreeNode::Group { .. } => {
                let mut files = BTreeSet::new();
                collect_source_paths(node, &mut files);
                let primary = match files.len() {
                    0 => return ("", false, String::new(), String::new()),
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
                (primary.0, false, primary.1, String::new())
            }
            TreeNode::Metadata { .. } => ("", false, String::new(), String::new()),
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
            self.copied_flash = Some(path);
        }
    }

    /// Copy the selected row's full name to the clipboard (the `n` shortcut): a
    /// tensor's complete name (e.g. `model.layers.0.self_attn.k_norm.weight`,
    /// which the tree may show abbreviated), or a group's path.
    fn copy_selected_name(&mut self) {
        let Some((node, _)) = self.flattened_tree.get(self.selected_idx) else {
            return;
        };
        let name = node.name().to_string();
        if !name.is_empty() {
            copy_to_clipboard(&name);
            self.copied_flash = Some(name);
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

    /// Act on the highlighted tree row. Returns `Some(Screen::Detail)` when a
    /// tensor was selected (the navigator opens it); groups expand and metadata
    /// opens in place, returning `None`.
    fn handle_selection(&mut self) -> Option<Screen> {
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
                    return Some(Screen::Detail {
                        tensor: info.name.clone(),
                        slice: 0,
                    });
                }
                TreeNode::Metadata { info } => {
                    self.show_metadata_detail(info);
                }
            }
        }
        None
    }

    /// Apply a CLI `--tensor` request: locate the tensor, apply any dtype
    /// override and edges/overview choice, then either render it once (`--exit`)
    /// or return the [`Screen`] to seed the navigator with. Returns `None` when
    /// the tensor isn't found, the slice is invalid, or it was a one-shot render.
    fn open_requested(&mut self, req: OpenRequest) -> Option<Screen> {
        // Resolve the target tensor: the named one, or — when `--tensor` is
        // omitted — the sole tensor if the checkpoint has exactly one (e.g. any
        // `.npy`, or a single-array `.npz`/HDF5/safetensors). Ambiguous otherwise.
        let tensor = match &req.tensor {
            Some(name) => match self.tensors.iter().find(|t| t.name == *name) {
                Some(t) => t.clone(),
                None => {
                    let _ = UI::draw_message(
                        "Tensor not found",
                        &format!(
                            "No tensor named '{name}' in this checkpoint — opening the browser instead."
                        ),
                    );
                    if !req.exit_after {
                        let _ = event::read();
                    }
                    return None;
                }
            },
            None => match self.tensors.as_slice() {
                [only] => only.clone(),
                _ => {
                    if self.tensors.len() > 1 {
                        let _ = UI::draw_message(
                            "Which tensor?",
                            "This checkpoint has multiple tensors — name one with --tensor, or pick it in the browser.",
                        );
                        if !req.exit_after {
                            let _ = event::read();
                        }
                    }
                    return None;
                }
            },
        };
        // Apply the dtype override (skipped for formats that can't reinterpret,
        // so the header never claims a view that isn't actually applied).
        if let Some(dt) = req.dtype
            && dtype_overridable(&tensor)
        {
            let mut overrides = self.dtype_overrides.borrow_mut();
            if dt == ViewDtype::Stored {
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
                    let _ = UI::draw_message("Invalid --shape", &msg);
                    if !req.exit_after {
                        let _ = event::read();
                    }
                    return None;
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
                    let _ = UI::draw_message("Invalid --slice", &msg);
                    if !req.exit_after {
                        let _ = event::read();
                    }
                    return None;
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
                return None;
            }
        };

        // One-shot (`--exit`): render the requested screen once and return None
        // (the navigator is never entered).
        if req.exit_after {
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
            return None;
        }

        // Interactive: `--compute-stats` pre-warms the detail's stats so they
        // show on first render (the navigator itself always opens on-demand).
        if stats_start == StatsStart::Auto
            && let Screen::Detail { .. } = screen
        {
            let view = self
                .dtype_overrides
                .borrow()
                .get(&tensor.name)
                .copied()
                .unwrap_or(ViewDtype::Stored);
            let overridable = dtype_overridable(&tensor);
            let unindexed = self.unindexed.contains(&tensor.source_path);
            let shape = self
                .shape_overrides
                .borrow()
                .get(&tensor.name)
                .cloned()
                .unwrap_or_else(|| tensor.shape.clone());
            self.compute_stats_animated(&tensor, view, |sv| {
                let _ = UI::draw_tensor_detail(
                    &mut live_out(),
                    &tensor,
                    &shape,
                    view,
                    overridable,
                    unindexed,
                    sv,
                );
            });
        }
        Some(screen)
    }

    /// The tensor detail screen. Sub-views: `m` heatmap, `v` numeric values
    /// (returned to the navigator as a new screen), `d` reinterpret dtype, `s`
    /// compute statistics. Backspace / `\` step through the screen history; any
    /// other key goes back to the tree. Returns the chosen [`Nav`].
    fn run_detail(
        &self,
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
        loop {
            let view = self
                .dtype_overrides
                .borrow()
                .get(&tensor.name)
                .copied()
                .unwrap_or(ViewDtype::Stored);
            let shape = self
                .shape_overrides
                .borrow()
                .get(&tensor.name)
                .cloned()
                .unwrap_or_else(|| tensor.shape.clone());
            // `--compute-stats` kicks off the scan synchronously on first open,
            // animating the spinner right here; normal browsing stays fast.
            if first && stats_start == StatsStart::Auto {
                self.compute_stats_animated(tensor, view, |sv| {
                    let _ = UI::draw_tensor_detail(
                        &mut live_out(),
                        tensor,
                        &shape,
                        view,
                        overridable,
                        unindexed,
                        sv,
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
            if UI::draw_tensor_detail(
                &mut live_out(),
                tensor,
                &shape,
                view,
                overridable,
                unindexed,
                stats_view,
            )
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
            let ev = if let Some(job) = &scan {
                if event::poll(std::time::Duration::from_millis(80)).unwrap_or(false) {
                    job.pause.store(true, Ordering::Relaxed);
                    event::read()
                } else {
                    job.pause.store(false, Ordering::Relaxed);
                    continue;
                }
            } else {
                event::read()
            };
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
                // Compute exact whole-tensor statistics on demand, animating the
                // spinner in the detail screen's Statistics line.
                Ok(Event::Key(KeyEvent {
                    code: KeyCode::Char('s') | KeyCode::Char('S'),
                    ..
                })) => {
                    self.compute_stats_animated(tensor, view, |sv| {
                        let _ = UI::draw_tensor_detail(
                            &mut live_out(),
                            tensor,
                            &shape,
                            view,
                            overridable,
                            unindexed,
                            sv,
                        );
                    });
                }
                // Reinterpret the dtype from the detail screen too.
                Ok(Event::Key(KeyEvent {
                    code: KeyCode::Char('d') | KeyCode::Char('D'),
                    ..
                })) if overridable => {
                    if let Some(chosen) = self.prompt_dtype(tensor, DtypePreview::Detail) {
                        let mut overrides = self.dtype_overrides.borrow_mut();
                        if chosen == ViewDtype::Stored {
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
                    match self.prompt_reshape(tensor, current.as_deref()) {
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
                    let text = screen_text(|buf| {
                        let _ = UI::draw_tensor_detail(
                            buf,
                            tensor,
                            &shape,
                            view,
                            overridable,
                            unindexed,
                            stats_view,
                        );
                    });
                    copy_to_clipboard(&text);
                }
                // `y` shows and copies the CLI command that reopens this screen.
                Ok(Event::Key(KeyEvent {
                    code: KeyCode::Char('y'),
                    ..
                })) => self.copy_command(&self.command_for_detail(tensor)),
                // `l` opens the legend for the detail screen's glyphs, then
                // returns here (the loop redraws the detail over it).
                Ok(Event::Key(KeyEvent {
                    code: KeyCode::Char('l'),
                    ..
                })) => self.show_legend(Legend::Detail),
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
    fn run_data(
        &self,
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
            let view = self
                .dtype_overrides
                .borrow()
                .get(&tensor.name)
                .copied()
                .unwrap_or(ViewDtype::Stored);

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

            // (slices, overridable, clamped slice) on success.
            let (slices, overridable) = match self.draw_data_view(
                &mut live_out(),
                tensor,
                repr,
                slice,
                view,
                mode,
                stats_view,
            ) {
                Ok((slices, overridable, clamped)) => {
                    slice = clamped;
                    (slices, overridable)
                }
                Err(msg) => {
                    let _ = UI::draw_message("Data preview unavailable", &msg);
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
            let mut pending = if let Some(job) = &scan {
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
            } else {
                event::read()
            };
            loop {
                match pending {
                    Ok(Event::Key(key)) if is_ctrl_c(&key) => quit_immediately(),
                    Ok(Event::Key(KeyEvent {
                        code, modifiers, ..
                    })) => {
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
                                if let Some(chosen) = self
                                    .prompt_dtype(tensor, DtypePreview::Data { repr, slice, mode })
                                {
                                    let mut overrides = self.dtype_overrides.borrow_mut();
                                    if chosen == ViewDtype::Stored {
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
                                match self.prompt_reshape(tensor, current.as_deref()) {
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
                            // Copy the data view's text to the clipboard.
                            KeyCode::Char('c') => {
                                let text = screen_text(|buf| {
                                    let _ = self.draw_data_view(
                                        buf, tensor, repr, slice, view, mode, stats_view,
                                    );
                                });
                                copy_to_clipboard(&text);
                            }
                            // `y` shows and copies the CLI command that reopens
                            // this exact view.
                            KeyCode::Char('y') => {
                                self.copy_command(&self.command_for_data(tensor, repr, slice))
                            }
                            // Open the legend for this representation, then
                            // redraw the data view over it.
                            KeyCode::Char('l') => self.show_legend(match repr {
                                Representation::Heatmap => Legend::Heatmap,
                                Representation::Values => Legend::Values,
                            }),
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
                    Ok(_) => {} // resize etc.: re-sample and redraw the same slice
                    Err(_) => return (Nav::Back, repr, slice),
                }
                // Drain the next buffered event without blocking; once the queue
                // is empty, fall out to redraw exactly once for the whole burst.
                if event::poll(std::time::Duration::ZERO).unwrap_or(false) {
                    pending = event::read();
                } else {
                    break;
                }
            }
        }
    }

    /// Sample and draw the heatmap or numeric grid for `(slice, view)`, sized to
    /// the terminal. Returns `(slices, overridable, clamped_slice)` on success,
    /// or an error message for the caller to show. Shared by the data-view loop
    /// and the dtype menu's live preview.
    #[allow(clippy::too_many_arguments)] // a render helper; the params are all distinct
    fn draw_data_view(
        &self,
        out: &mut impl Write,
        tensor: &TensorInfo,
        repr: Representation,
        slice: usize,
        view: ViewDtype,
        mode: SampleMode,
        stats: StatsView,
    ) -> Result<(usize, bool, usize), String> {
        let (cols, rows) = terminal::size().unwrap_or((100, 40));
        let text_rows = (rows as usize).saturating_sub(8).max(1);
        let (max_rows, max_cols) = match repr {
            // The heatmap packs two data rows per text line (half blocks), so it
            // can sample twice as many rows as there are lines.
            Representation::Heatmap => (text_rows * 2, (cols as usize).saturating_sub(1).max(1)),
            // Numeric cell width depends on the actual values (small ints — even
            // in a wide dtype — pack many columns); plus a 7-char row-index
            // column. The exact range comes from stats once computed.
            Representation::Values => {
                let cell = view.cell_width(&tensor.dtype, stats.value_range());
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
            let sample = self.with_reader(tensor, |reader| {
                crate::sample::sample_tensor_with(
                    reader, tensor, &eff_shape, max_rows, max_cols, slice, view, mode,
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
        let info = (sample.slices, sample.overridable, sample.slice);
        match repr {
            Representation::Heatmap => {
                UI::draw_heatmap(out, tensor, sample, stats).map_err(|e| e.to_string())?;
            }
            Representation::Values => {
                UI::draw_values(out, tensor, sample, stats, self.data_view_stripe.get())
                    .map_err(|e| e.to_string())?;
            }
        }
        Ok(info)
    }

    /// Open the dtype-selection menu with a live preview, returning the chosen
    /// view or `None` if cancelled. `d`/→ move forward, `D`/← back (the menu is
    /// horizontal); Enter applies, Esc cancels. Ctrl-C quits the app. The
    /// preview re-renders whichever screen the menu was opened from.
    fn prompt_dtype(&self, tensor: &TensorInfo, preview: DtypePreview) -> Option<ViewDtype> {
        let options = crate::sample::view_options(&tensor.dtype);
        if options.is_empty() {
            return None;
        }
        let current = self
            .dtype_overrides
            .borrow()
            .get(&tensor.name)
            .copied()
            .unwrap_or(ViewDtype::Stored);
        let mut idx = options.iter().position(|v| *v == current).unwrap_or(0);
        // The shape override (if any) is fixed while the dtype menu is open.
        let shape = self
            .shape_overrides
            .borrow()
            .get(&tensor.name)
            .cloned()
            .unwrap_or_else(|| tensor.shape.clone());
        loop {
            // Live preview of the highlighted view, then the menu overlay.
            // Only read cached stats — navigating the menu must never trigger a scan.
            let stats = self.cached_stats(tensor, options[idx]);
            let stats_view = stats.as_ref().map_or(StatsView::Pending, StatsView::Ready);
            let preview_ok = match preview {
                DtypePreview::Detail => UI::draw_tensor_detail(
                    &mut live_out(),
                    tensor,
                    &shape,
                    options[idx],
                    true,
                    self.unindexed.contains(&tensor.source_path),
                    stats_view,
                )
                .is_ok(),
                DtypePreview::Data { repr, slice, mode } => self
                    .draw_data_view(
                        &mut live_out(),
                        tensor,
                        repr,
                        slice,
                        options[idx],
                        mode,
                        stats_view,
                    )
                    .is_ok(),
            };
            if !preview_ok {
                return None;
            }
            let _ = UI::draw_dtype_menu(&options, idx);
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
    fn prompt_reshape(&self, tensor: &TensorInfo, current: Option<&[usize]>) -> ReshapeChoice {
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
            if UI::draw_reshape_prompt(tensor.num_elements, &tensor.shape, &input, error.as_deref())
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
    fn repack_checkpoint(&self) {
        let Some(input) = self.repack_input() else {
            let _ = UI::draw_message(
                "Repack unavailable",
                "Repacking is available only for a single HDF5 checkpoint (.h5/.hdf5).",
            );
            let _ = event::read();
            return;
        };
        let default = default_repacked_name(&input);
        let Some(output) = self.prompt_output_path(&default) else {
            return;
        };
        let Some(codec) = self.prompt_codec() else {
            return;
        };
        if !self.confirm_same_codec(&input, codec) {
            return;
        }
        let Some(buffer_bytes) = self.prompt_buffer() else {
            return;
        };
        self.run_repack(&input, &output, codec, buffer_bytes);
    }

    /// If the source already uses `codec`, ask whether to re-encode anyway
    /// (a plain copy would be equivalent). Returns `true` to proceed.
    #[cfg(feature = "hdf5")]
    fn confirm_same_codec(&self, input: &Path, codec: crate::codec::Codec) -> bool {
        if crate::convert::source_codec(input) != Some(codec) {
            return true;
        }
        let title = format!("Source is already {} — re-encode it anyway?", codec.label());
        let mut idx = 0; // 0 = repack anyway, 1 = cancel
        loop {
            if UI::draw_choice_menu(&title, &["Repack anyway", "Cancel"], idx).is_err() {
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
    fn confirm_same_codec(&self, _input: &Path, _codec: crate::codec::Codec) -> bool {
        true
    }

    /// Pick the output compression codec from a menu. Returns `None` if cancelled.
    fn prompt_codec(&self) -> Option<crate::codec::Codec> {
        use crate::codec::Codec;
        let codecs = [Codec::Gzip, Codec::Zstd, Codec::Lz4, Codec::Uncompressed];
        let labels: Vec<&str> = codecs.iter().map(|c| c.label()).collect();
        let mut idx = 0;
        loop {
            if UI::draw_choice_menu("Repack — compression codec", &labels, idx).is_err() {
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
    fn prompt_buffer(&self) -> Option<usize> {
        let mut input = "256M".to_string();
        let mut error: Option<String> = None;
        loop {
            if UI::draw_text_prompt(
                "Streaming buffer size (e.g. 64M, 256M, 1G)",
                &input,
                error.as_deref(),
            )
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
    fn prompt_output_path(&self, default: &Path) -> Option<PathBuf> {
        let mut input = default.to_string_lossy().into_owned();
        let mut error: Option<String> = None;
        loop {
            if UI::draw_text_prompt("Save repacked checkpoint as", &input, error.as_deref())
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
    fn run_repack(&self, input: &Path, output: &Path, codec: crate::codec::Codec, buffer: usize) {
        let level = codec.clamp_level(codec.default_level());
        let opts = crate::convert::Options {
            codec,
            level,
            buffer_bytes: buffer,
        };
        let title = format!("Repacking → {} ({})", output.display(), codec.label());
        let _ = UI::draw_progress(&title, 0, 1, "starting…");
        let result = crate::convert::convert_hdf5(input, output, &opts, |done, total, name| {
            let _ = UI::draw_progress(&title, done, total, name);
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
        let _ = UI::draw_message(heading, &body);
        let _ = event::read();
    }

    #[cfg(not(feature = "hdf5"))]
    fn run_repack(
        &self,
        _input: &Path,
        _output: &Path,
        _codec: crate::codec::Codec,
        _buffer: usize,
    ) {
        let _ = UI::draw_message(
            "Repack unavailable",
            "Rebuild with `--features hdf5` to enable repacking.",
        );
        let _ = event::read();
    }

    /// Show the context-sensitive legend for the current screen (`l`), then wait
    /// for any key to dismiss it (Ctrl-C still quits). The caller's loop redraws
    /// its own screen over the overlay on the next iteration.
    fn show_legend(&self, legend: Legend) {
        if UI::draw_legend(legend).is_ok()
            && let Ok(Event::Key(key)) = event::read()
            && is_ctrl_c(&key)
        {
            quit_immediately();
        }
    }

    /// Copy `command` to the clipboard and show it in a dismissible box, so the
    /// user can both see and paste the exact invocation that reopens this screen
    /// (the `y` shortcut). Any key returns; Ctrl-C still quits.
    fn copy_command(&self, command: &str) {
        copy_to_clipboard(command);
        if UI::draw_command(command).is_ok()
            && let Ok(Event::Key(key)) = event::read()
            && is_ctrl_c(&key)
        {
            quit_immediately();
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
        parts.join(" ")
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
        self.command_base(tensor).join(" ")
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
                .map_err(|_| format!("invalid dimension: {tok}"))?;
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

/// Restore the terminal (leave raw mode, show the cursor) and exit the process
/// immediately, leaving the last frame on screen with the prompt just below it.
/// Used for Ctrl-C from any of the detail/data sub-screens so it quits outright
/// instead of stepping back one screen.
fn quit_immediately() -> ! {
    let mut stdout = io::stdout();
    let _ = execute!(stdout, cursor::Show);
    let _ = terminal::disable_raw_mode();
    // Drop the prompt onto a fresh line below the preserved frame.
    println!();
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

/// A buffered writer over the live stdout for the screen-draw functions. The
/// buffering makes each frame flush atomically (no progressive paint / flicker).
fn live_out() -> io::BufWriter<io::StdoutLock<'static>> {
    io::BufWriter::new(io::stdout().lock())
}

/// Render whatever screen is currently shown into a plain-text string (ANSI
/// escapes stripped), for the "copy screen contents" shortcut.
fn screen_text(render: impl FnOnce(&mut Vec<u8>)) -> String {
    let mut buf = Vec::new();
    render(&mut buf);
    strip_ansi(&buf)
}

/// Strip ANSI escape sequences (CSI / OSC / charset selects) and carriage
/// returns from terminal output, leaving the plain text the user sees.
fn strip_ansi(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\r' => {}
            '\x1b' => match chars.next() {
                // CSI: ESC [ … <final 0x40–0x7e>
                Some('[') => {
                    for d in chars.by_ref() {
                        if ('\x40'..='\x7e').contains(&d) {
                            break;
                        }
                    }
                }
                // OSC: ESC ] … terminated by BEL or ESC \
                Some(']') => {
                    while let Some(d) = chars.next() {
                        if d == '\x07' {
                            break;
                        }
                        if d == '\x1b' {
                            chars.next(); // consume the trailing '\'
                            break;
                        }
                    }
                }
                // Two-byte escapes (charset selects etc.): drop the next char.
                Some(_) => {}
                None => {}
            },
            other => out.push(other),
        }
    }
    // Trim trailing whitespace on each line and drop trailing blank lines.
    let mut lines: Vec<&str> = out.lines().map(|l| l.trim_end()).collect();
    while lines.last().is_some_and(|l| l.is_empty()) {
        lines.pop();
    }
    lines.join("\n")
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
