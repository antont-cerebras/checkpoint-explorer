mod codec;
#[cfg(feature = "hdf5")]
mod convert;
mod diff;
mod explorer;
mod filter;
mod gguf;
#[cfg(feature = "hdf5")]
mod hdf5;
#[cfg(feature = "hdf5")]
mod hdf5_lz4;
#[cfg(feature = "hdf5")]
mod hdf5_zstd;
mod health;
mod npy;
mod progress;
mod remote;
mod s3;
mod sample;
mod sftp;
mod tree;
mod tui;
mod ui;
mod utils;

use anyhow::{Context, Result};
use clap::{Args as ClapArgs, Parser, Subcommand};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::explorer::{DataLayout, Explorer, OpenRequest, OpenView};
use crate::tree::{MetadataInfo, TensorInfo};

/// Worked examples shown at the end of `--help` (not the terse `-h`), grouped by
/// the most useful things you can do. Written to read cleanly for both people
/// and coding agents: one commented, copy-pasteable command per line.
const EXAMPLES: &str = "\
Examples:
  Browse a checkpoint — a single file, a sharded directory, or a glob:
      checkpoint-explorer model.safetensors
      checkpoint-explorer /path/to/sharded-model/
      checkpoint-explorer 'model-*.safetensors'

  Look inside a tensor's data — heatmap, numeric grid, histogram, statistics:
      checkpoint-explorer model.safetensors --tensor model.layers.0.mlp.down_proj.weight --heatmap
      checkpoint-explorer model.safetensors --tensor NAME --values --dtype u4   # decode packed 4-bit

  Read a remote / S3 checkpoint over SSH (only metadata leaves the host):
      checkpoint-explorer --ssh-read user@host s3://bucket/model/checkpoint
      checkpoint-explorer user@host:/opt/models/some-model          # scp-style; a safetensors dir

  Export the structure for scripts / agents (text, or --format json):
      checkpoint-explorer model.safetensors --print-tree
      checkpoint-explorer model.safetensors --print-tensors --format json
      checkpoint-explorer model.safetensors --print-tree --name '*.mlp.*'   # !GLOB excludes

  Compare two checkpoints (exit 0 = identical, 1 = differ, 2 = error):
      checkpoint-explorer diff old.safetensors new.safetensors
      checkpoint-explorer diff old/ new/ --values --name '*.mlp.*'

  Repack an HDF5 checkpoint with an alternative codec — smaller on disk (hdf5 build only):
      checkpoint-explorer convert in.hdf5 out.hdf5 --codec zstd

  Per-subcommand help:  checkpoint-explorer diff --help  ·  checkpoint-explorer convert --help";

#[derive(Parser)]
#[command(name = "checkpoint-explorer")]
#[command(version)]
#[command(
    about = "Explore model checkpoints in the terminal — browse the tree, look inside tensor data, and diff (.safetensors / .gguf / .npy / .npz / .hdf5)"
)]
#[command(long_about = "\
Interactive terminal explorer for model checkpoints — .safetensors, .gguf, .npy/.npz, \
and (with the hdf5 build) .hdf5.

Beyond the tree of tensor names and shapes, it shows the actual data: ASCII heatmaps, \
numeric-value grids, value histograms, and exact whole-tensor statistics — streamed in \
bounded blocks, so multi-GB tensors work without loading them into RAM. Packed / \
quantized weights (4-bit, fused-codebook MoE) are decoded to their true values. \
Sharded / multi-file models, directories, and globs merge into one tree.

Remote checkpoints are read over SSH — a safetensors directory/file via SFTP, or an \
s3:// cstorch checkpoint via a remote venv — sending only metadata off the host, so \
data and credentials stay remote. For scripts and agents there are one-shot \
--print-tree / --print-tensors exports (text or JSON) and a `diff` subcommand with \
diff-style exit codes.

Give one or more paths to browse; press `l` in any screen for its key legend. See the \
examples below and `<command> --help`.")]
#[command(after_long_help = EXAMPLES)]
#[command(args_conflicts_with_subcommands = true)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Files/directories/globs to explore (the default action when no
    /// subcommand is given).
    #[command(flatten)]
    explore: ExploreArgs,
}

#[derive(ClapArgs)]
struct ExploreArgs {
    #[arg(
        help = "Checkpoint files, directories, or glob patterns to explore (e.g. *.safetensors, model-*.gguf, *.npy, *.npz, *.hdf5). Remote paths work too — an scp-style [USER@]HOST:/path (read over SSH, like --ssh-read), or an s3:// URI passed together with --ssh-read <HOST>; both browse-only, only metadata leaves the host"
    )]
    paths: Vec<PathBuf>,

    #[arg(
        short,
        long,
        help = "Recursively search directories for checkpoint files"
    )]
    recursive: bool,

    #[arg(
        long = "no-health-check",
        help = "Skip the checkpoint health check (index vs. files on disk)"
    )]
    no_health_check: bool,

    #[arg(
        long = "no-preload",
        help = "Don't compute a tensor's statistics in the background when its detail screen opens (the scan reads the tensor, warming the OS/disk cache to speed up the heatmap/values views especially over NFS; with this flag, statistics are computed only when you press s)"
    )]
    no_preload: bool,

    #[arg(
        long,
        value_name = "NAME",
        help = "Open a specific tensor on startup (exact name); optional when the checkpoint has only one tensor (e.g. a .npy). Combine with --dtype/--shape/--values/--heatmap/--edge"
    )]
    tensor: Option<String>,

    #[arg(
        long,
        value_name = "NAME",
        conflicts_with = "tensor",
        help = "Reveal a metadata entry on startup (exact name, e.g. model.norm.weight.__metadata__) — opens the tree with it selected"
    )]
    metadata: Option<String>,

    #[arg(
        long,
        value_name = "DTYPE",
        value_parser = sample::parse_view_dtype,
        help = "Reinterpret the tensor's dtype: u4, i4, unpacked (fused codebook, needs a packing schema), f16, bf16, i16, u16, f32, i32, u32, f64, i64, u64, i8, u8, stored"
    )]
    dtype: Option<sample::ViewDtype>,

    #[arg(
        long,
        conflicts_with = "heatmap",
        help = "Open straight into the tensor's numeric-values grid"
    )]
    values: bool,

    #[arg(long, help = "Open straight into the tensor's heatmap")]
    heatmap: bool,

    #[arg(
        long,
        conflicts_with_all = ["values", "heatmap", "tree"],
        help = "Show the tensor's value histogram on its detail screen"
    )]
    histogram: bool,

    #[arg(
        long,
        value_name = "N",
        value_parser = parse_bins,
        conflicts_with_all = ["values", "heatmap", "tree"],
        help = "Histogram bucket count (1–512); implies --histogram"
    )]
    bins: Option<usize>,

    #[arg(
        long,
        conflicts_with_all = ["values", "heatmap"],
        help = "Reveal the tensor highlighted in the tree browser instead of opening a view"
    )]
    tree: bool,

    #[arg(
        long,
        visible_alias = "edges",
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "0.5,0.5",
        value_name = "RFRAC,CFRAC",
        conflicts_with_all = ["overview", "window"],
        help = "Show the first/last edges (padding) submode; optional ROW,COL head/tail split fractions 0..1 (0=first, 1=last, 0.5=balanced)"
    )]
    edge: Option<String>,

    #[arg(long, help = "Show the evenly-spaced overview submode")]
    overview: bool,

    #[arg(
        long,
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "0,0",
        value_name = "ROW,COL",
        conflicts_with_all = ["edge", "overview"],
        help = "Show the contiguous pannable window submode; optional ROW,COL top-left corner (default 0,0)"
    )]
    window: Option<String>,

    #[arg(
        long,
        value_name = "MODE",
        value_parser = ui::parse_stripe_mode,
        help = "Zebra-stripe the numeric grid by rows, cols, or off"
    )]
    zebra: Option<ui::StripeMode>,

    #[arg(
        long,
        value_name = "BASE",
        value_parser = ui::parse_num_base,
        help = "Numeral base for the numeric grid: dec, hex, oct, or bin (non-decimal shows raw stored bits)"
    )]
    base: Option<ui::NumBase>,

    #[arg(
        long,
        value_name = "INDEX",
        help = "Starting slice for a 3D tensor: an index (e.g. 12) or a percentage (e.g. 50%)"
    )]
    slice: Option<String>,

    #[arg(
        long,
        value_name = "DIMS",
        help = "Reinterpret the tensor's shape (same element count); dims like 10,100 or -1,768 (one dim may be -1/*/_ to infer)"
    )]
    shape: Option<String>,

    #[arg(
        long,
        help = "Start computing statistics immediately when opening the detail view (data views always compute them)"
    )]
    compute_stats: bool,

    #[arg(
        long,
        value_name = "STATE",
        value_parser = explorer::parse_tree_state,
        help = "Open the tree fully expanded or collapsed (the `E` / `C` keys): expanded or collapsed"
    )]
    tree_state: Option<explorer::TreeState>,

    #[arg(
        long,
        value_name = "QUERY",
        help = "Open the tree in search mode filtered to QUERY (the `/` key)"
    )]
    search: Option<String>,

    #[arg(
        long,
        help = "Overlay the requested screen's legend (the `l` key) — useful with --plain"
    )]
    legend: bool,

    #[arg(
        long,
        help = "Render the requested view once and exit, without entering interactive navigation"
    )]
    exit: bool,

    #[arg(
        long,
        help = "Render the requested view once as plain text (no colour, no cursor control) and exit — for piping, grep, and end-to-end tests"
    )]
    plain: bool,

    #[arg(
        long,
        help = "Print the CLI command that reopens the requested view (what `y` copies) and exit, instead of rendering"
    )]
    emit_command: bool,

    #[arg(
        long = "print-tree",
        conflicts_with = "print_tensors",
        help = "Print the whole checkpoint tree (grouped, fully expanded) and exit — plain text, or a model.safetensors.index.json-style object with --format=json"
    )]
    print_tree: bool,

    #[arg(
        long = "print-tensors",
        help = "Print a flat list of every tensor and exit — plain text, or a JSON array with --format=json"
    )]
    print_tensors: bool,

    #[arg(
        long,
        value_enum,
        default_value_t = explorer::TreeFormat::default(),
        value_name = "FORMAT",
        help = "Output format for --print-tree / --print-tensors: text (default) or json"
    )]
    format: explorer::TreeFormat,

    #[arg(
        short = 'v',
        long = "verbose",
        action = clap::ArgAction::Count,
        help = "Add per-tensor detail to --print-tree / --print-tensors: the source file in text; a tensors block / detail objects in json"
    )]
    verbose: u8,

    #[arg(
        long = "name",
        value_name = "GLOB",
        help = "Filter --print-tree / --print-tensors to tensors whose name matches this glob (e.g. '*.mlp.*', 'model.layers.0.*'). Repeatable; prefix with ! to exclude ('!*.bias' = everything but biases)"
    )]
    name: Vec<String>,

    #[arg(
        long = "ssh-read",
        value_name = "[USER@]HOST",
        help = "Read a remote checkpoint's structure over SSH on [USER@]HOST (which has the access): an s3:// cstorch checkpoint, or a path to a safetensors directory/file on that host. Only the tensor metadata (names/dtypes/shapes) leaves the host — data/secrets stay remote. Browse-only"
    )]
    ssh_read: Option<String>,

    #[arg(
        long = "ssh-venv",
        value_name = "PATH",
        help = "Path to the cstorch virtualenv on the --ssh-read host, activated with `source <PATH>/bin/activate` (default: ~/venv)"
    )]
    ssh_venv: Option<String>,
}

#[derive(Subcommand)]
// Parsed once at startup; the size gap between `Convert` and the many-flag `Diff`
// variant doesn't matter here.
#[allow(clippy::large_enum_variant)]
enum Command {
    /// Repack an HDF5 checkpoint into a new file, re-compressing every dataset
    /// with the chosen codec (e.g. gzip/zstd are ~2× smaller than the LZ4 these
    /// checkpoints ship with).
    Convert {
        /// Source `.h5`/`.hdf5` checkpoint.
        input: PathBuf,
        /// Destination file to create.
        output: PathBuf,
        /// Compression codec for the output.
        #[arg(short, long, value_enum, default_value_t = codec::Codec::default())]
        codec: codec::Codec,
        /// Compression level (gzip 0–9, zstd 1–22; ignored for lz4/none).
        /// Defaults to a sensible level for the codec.
        #[arg(short, long)]
        level: Option<u8>,
        /// Streaming buffer per dataset block, e.g. `256M`, `1G` (default 256M).
        #[arg(short, long, default_value = "256M")]
        buffer: String,
        /// Overwrite the output file if it already exists.
        #[arg(short, long)]
        force: bool,
    },

    /// Compare two checkpoints and summarize their structural differences:
    /// tensors (by name, dtype, shape) and metadata (by name, value) that were
    /// added, removed, or changed. Tensor data/values are not compared.
    ///
    /// Exit status follows `diff`: 0 = structurally identical, 1 = differences
    /// found, 2 = trouble (a path couldn't be read).
    Diff {
        /// The baseline ("old") checkpoint — a file, directory, or glob.
        old: PathBuf,
        /// The checkpoint to compare against the baseline ("new").
        new: PathBuf,
        /// Recursively search directories for checkpoint files.
        #[arg(short, long)]
        recursive: bool,
        /// Compare only this one tensor (exact name) and, when it's present in
        /// both, also compare its element *values* (max/mean |Δ|), not just its
        /// dtype and shape. Without it, all tensors and metadata are compared
        /// structurally.
        #[arg(long, value_name = "NAME")]
        tensor: Option<String>,
        /// Compare only tensors — skip the checkpoints' metadata entirely.
        #[arg(long = "only-tensors")]
        only_tensors: bool,
        /// Also compare element values: read each tensor present in both (with a
        /// matching shape) and report max/mean |Δ| — turning a values-only change
        /// (same dtype & shape, different data) into a difference. Reads the whole
        /// checkpoint, so it's slower than the default structural diff.
        #[arg(long)]
        values: bool,
        /// Decode values under this view before comparing (with --values,
        /// --histogram, or --tensor): stored, u4, i4, unpacked (3-bit codebook, via
        /// the packing schema), f16, bf16, i16, u16, f32, i32, u32, f64, i64, u64,
        /// i8, u8.
        #[arg(long, value_name = "DTYPE", value_parser = sample::parse_view_dtype)]
        dtype: Option<sample::ViewDtype>,
        /// Compare value distributions: bin each common tensor's values (old & new
        /// over a shared layout) and report the total variation distance. With
        /// --tensor, prints the full bin-by-bin table. Reads the whole checkpoint.
        #[arg(long)]
        histogram: bool,
        /// Histogram bucket count (1–512) for --histogram; default picks a sensible
        /// count per dtype.
        #[arg(long, value_name = "N", value_parser = parse_bins)]
        bins: Option<usize>,
        /// List every changed entry instead of collapsing ones that share a name
        /// template and the same change (e.g. the same per-layer dtype change) into
        /// one line with a count and index range.
        #[arg(long)]
        full: bool,
        /// Never colorize the output (also off automatically when stdout isn't a
        /// terminal, or when `NO_COLOR` is set).
        #[arg(long = "no-color")]
        no_color: bool,
        /// Only diff tensors whose name matches this glob (e.g.
        /// '*.mlp.down_proj.weight', 'model.layers.*'). Repeatable — a tensor
        /// passes if it matches ANY; prefix with ! to exclude ('!*.bias' =
        /// everything but biases). Scopes the whole diff (structural + values)
        /// to the matching subset; metadata is not compared.
        #[arg(long = "name", value_name = "GLOB")]
        name: Vec<String>,
        /// Only diff these exact tensor names (comma-separated). Combine with
        /// --names-from; a tensor passes if it's in either list.
        #[arg(long = "names", value_name = "A,B,C")]
        names: Option<String>,
        /// Only diff the tensor names listed in this file (one per line; blank
        /// lines and '#' comments ignored).
        #[arg(long = "names-from", value_name = "FILE")]
        names_from: Option<PathBuf>,
        /// Only diff tensors whose stored dtype matches this glob, e.g. 'BF16',
        /// 'F*' (F16/F32/…). Case-insensitive.
        #[arg(long = "dtype-is", value_name = "GLOB")]
        dtype_is: Option<String>,
        /// Only diff tensors whose shape matches this glob. Dims are comma- or
        /// x-separated; '*' wildcards one dimension, '**' any number — e.g.
        /// '768,2048', '768,*', '*,2048', '768,**', '**,2048'.
        #[arg(long = "shape-is", value_name = "DIMS")]
        shape_is: Option<String>,
        /// Compare up to N tensors in parallel with --values / --histogram
        /// (default: number of logical CPUs; 1 = sequential). Reading tensor data
        /// is I/O-bound, so overlapping tensors speeds the whole run up.
        #[arg(short = 'j', long = "jobs", value_name = "N")]
        jobs: Option<usize>,
        /// Read each checkpoint's structure over SSH on [USER@]HOST (which holds the
        /// access): an s3:// cstorch checkpoint or a remote safetensors
        /// directory/file. Data/secrets stay remote; structural diff (dtype/shape).
        #[arg(long = "ssh-read", value_name = "[USER@]HOST")]
        ssh_read: Option<String>,
        /// Path to the cstorch virtualenv on the --ssh-read host (default: ~/venv).
        #[arg(long = "ssh-venv", value_name = "PATH")]
        ssh_venv: Option<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Command::Convert {
            input,
            output,
            codec,
            level,
            buffer,
            force,
        }) => run_convert(&input, &output, codec, level, &buffer, force),
        Some(Command::Diff {
            old,
            new,
            recursive,
            tensor,
            only_tensors,
            values,
            dtype,
            histogram,
            bins,
            full,
            no_color,
            name,
            names,
            names_from,
            dtype_is,
            shape_is,
            jobs,
            ssh_read,
            ssh_venv,
        }) => {
            // `diff`-style exit codes (0 same / 1 differ / 2 trouble) don't map to
            // the `Result` convention `main` uses elsewhere, so exit explicitly.
            // `--ssh-read`: read each checkpoint's structure via cstorch on the
            // remote (secrets stay there).
            let remote = ssh_read.map(|host| {
                crate::remote::RemoteRead::new(host, ssh_venv.unwrap_or_else(|| "~/venv".into()))
            });
            let filter = match build_tensor_filter(
                &name,
                names.as_deref(),
                names_from.as_deref(),
                dtype_is.as_deref(),
                shape_is.as_deref(),
            ) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("checkpoint-explorer diff: {e:#}");
                    std::process::exit(2);
                }
            };
            let filtered = filter.is_active();
            let opts = diff::DiffOpts {
                color: color_enabled(no_color),
                // A tensor filter scopes the diff to a subset, so metadata isn't
                // compared (like --only-tensors, but for a filtered run).
                metadata: !only_tensors && !filtered,
                group: !full,
                values,
                histogram,
                filtered,
            };
            let view = dtype.unwrap_or(sample::ViewDtype::Stored);
            // Default parallelism = logical CPUs; `--jobs 0` is treated as 1.
            let jobs = jobs.filter(|&j| j > 0).unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4)
            });
            let started = std::time::Instant::now();
            let code = run_diff(
                &old,
                &new,
                recursive,
                tensor.as_deref(),
                view,
                bins,
                opts,
                &filter,
                jobs,
                remote.as_ref(),
            );
            // Report how long it took, by default (on stderr, so a piped diff on
            // stdout stays clean). Skip on trouble (exit 2) — the error already said.
            // Dimmed (when stderr is a colour terminal) as a secondary footer line.
            if code != 2 {
                use std::io::IsTerminal;
                let msg = format!(
                    "checkpoint-explorer diff: done in {}",
                    format_elapsed(started.elapsed())
                );
                let dim = !no_color
                    && std::env::var_os("NO_COLOR").is_none()
                    && std::io::stderr().is_terminal();
                if dim {
                    eprintln!("\x1b[2m{msg}\x1b[0m");
                } else {
                    eprintln!("{msg}");
                }
            }
            std::process::exit(code)
        }
        None => run_explore(cli.explore),
    }
}

/// Compare two checkpoints' structure and print the summary. Returns the process
/// exit code: `0` identical, `1` differences found, `2` trouble (unreadable path).
/// Whether to colorize the diff: off when `--no-color`, when `NO_COLOR` is set
/// (https://no-color.org), or when stdout isn't a terminal (so pipes stay clean).
fn color_enabled(no_color: bool) -> bool {
    use std::io::IsTerminal;
    !no_color && std::env::var_os("NO_COLOR").is_none() && std::io::stdout().is_terminal()
}

/// The decode view, requested histogram bucket count, and each side's packing
/// schemas (for the `unpacked` view) used by the value / distribution comparison.
struct ValueCtx<'a> {
    view: sample::ViewDtype,
    bins: Option<usize>,
    old_schemas: &'a HashMap<String, sample::PackingSchema>,
    new_schemas: &'a HashMap<String, sample::PackingSchema>,
}

/// Build a [`diff::TensorFilter`] from the `diff` selection flags: compile each
/// `--name`/`--dtype-is`/`--shape-is` glob and merge `--names` + `--names-from`
/// into the exact-name set. `--shape-is` dims are comma/`x`-separated and joined
/// with `/` so the shape glob's `*`/`**` act per-dimension. Errors (bad glob or
/// unreadable names file) bubble up to a `2` exit.
fn build_tensor_filter(
    name: &[String],
    names: Option<&str>,
    names_from: Option<&Path>,
    dtype_is: Option<&str>,
    shape_is: Option<&str>,
) -> Result<diff::TensorFilter> {
    use glob::Pattern;

    let name_filter = filter::NameFilter::parse(name)?;

    let mut exact: HashSet<String> = HashSet::new();
    if let Some(list) = names {
        exact.extend(
            list.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string),
        );
    }
    if let Some(path) = names_from {
        let text = fs::read_to_string(path)
            .with_context(|| format!("reading --names-from {}", path.display()))?;
        exact.extend(
            text.lines()
                .map(str::trim)
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .map(str::to_string),
        );
    }
    let names_exact = (names.is_some() || names_from.is_some()).then_some(exact);

    let dtype = dtype_is
        .map(|d| {
            Pattern::new(&d.to_uppercase())
                .with_context(|| format!("invalid --dtype-is glob {d:?}"))
        })
        .transpose()?;

    let shape = shape_is
        .map(|s| {
            let path: String = s
                .chars()
                .map(|c| if matches!(c, ',' | 'x' | 'X') { '/' } else { c })
                .collect();
            Pattern::new(&path).with_context(|| format!("invalid --shape-is pattern {s:?}"))
        })
        .transpose()?;

    Ok(diff::TensorFilter {
        names: name_filter,
        names_exact,
        dtype,
        shape,
    })
}

/// Braille spinner frames (matches the interactive stats scan).
const DIFF_SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// Shared state between the value-comparison workers and the spinner thread.
struct CompareState {
    /// Tensors currently being compared (one entry per in-flight worker).
    inflight: std::sync::Mutex<Vec<String>>,
    done: std::sync::atomic::AtomicUsize,
    total: usize,
    stop: std::sync::atomic::AtomicBool,
}

/// Live progress for `diff --values` / `--histogram`: reading tensor data is the
/// slow part, so — only in an interactive terminal — a background thread renders
/// a spinner plus **every tensor currently being compared** (one per line) on
/// **stderr** (stdout stays a clean diff), cleared when done. Workers call
/// [`Self::track`] for an RAII guard that keeps a tensor listed while it's being
/// compared. A no-op when stderr isn't a TTY (piped / headless) or nothing will
/// be compared.
struct CompareProgress {
    state: Option<std::sync::Arc<CompareState>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

/// RAII marker: a tensor is being compared while this is alive; on drop it leaves
/// the in-flight list and bumps the done count.
struct InFlight<'a> {
    state: Option<&'a CompareState>,
    name: String,
}

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering;
        if let Some(st) = self.state {
            if let Ok(mut v) = st.inflight.lock()
                && let Some(i) = v.iter().position(|n| n == &self.name)
            {
                v.swap_remove(i);
            }
            st.done.fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl CompareProgress {
    fn start(total: usize) -> Self {
        use std::io::IsTerminal;
        if total == 0 || !std::io::stderr().is_terminal() {
            return Self {
                state: None,
                handle: None,
            };
        }
        let state = std::sync::Arc::new(CompareState {
            inflight: std::sync::Mutex::new(Vec::new()),
            done: std::sync::atomic::AtomicUsize::new(0),
            total,
            stop: std::sync::atomic::AtomicBool::new(false),
        });
        let worker = std::sync::Arc::clone(&state);
        let handle = std::thread::spawn(move || compare_spinner_loop(worker));
        Self {
            state: Some(state),
            handle: Some(handle),
        }
    }

    /// Mark `name` as being compared until the returned guard drops.
    fn track(&self, name: &str) -> InFlight<'_> {
        if let Some(st) = &self.state
            && let Ok(mut v) = st.inflight.lock()
        {
            v.push(name.to_string());
        }
        InFlight {
            state: self.state.as_deref(),
            name: name.to_string(),
        }
    }

    /// Stop the spinner thread (which erases its block on exit) and join it.
    fn finish(mut self) {
        use std::sync::atomic::Ordering;
        if let Some(st) = &self.state {
            st.stop.store(true, Ordering::Relaxed);
        }
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// The spinner thread body: ~10×/s redraw a block — a header plus one line per
/// in-flight tensor — in place on stderr until told to stop, then erase it.
fn compare_spinner_loop(st: std::sync::Arc<CompareState>) {
    use std::io::Write;
    use std::sync::atomic::Ordering;
    let (width, height) = match crossterm::terminal::size() {
        Ok((c, r)) if c > 0 && r > 0 => (c as usize, r as usize),
        _ => (100, 24),
    };
    let mut prev_lines = 0usize;
    let mut frame = 0usize;
    while !st.stop.load(Ordering::Relaxed) {
        let mut names = st.inflight.lock().map(|v| v.clone()).unwrap_or_default();
        names.sort_unstable();
        let done = st.done.load(Ordering::Relaxed);
        let block = compare_progress_block(frame, done, st.total, &names, width, height);
        draw_block(&block, &mut prev_lines);
        frame += 1;
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    erase_block(prev_lines);
    let _ = std::io::stderr().flush();
}

/// The progress block: a spinner + `done/total` header, then one indented line
/// per in-flight tensor (capped to the terminal height, name tails kept).
fn compare_progress_block(
    frame: usize,
    done: usize,
    total: usize,
    names: &[String],
    width: usize,
    height: usize,
) -> Vec<String> {
    let spin = DIFF_SPINNER[frame % DIFF_SPINNER.len()];
    let mut lines = vec![format!(
        "{spin} comparing tensors ({done}/{total}, {} in flight):",
        names.len()
    )];
    let indent = "    ";
    let max_rows = height.saturating_sub(2).max(1); // keep the block on screen
    let shown = names.len().min(max_rows);
    for name in &names[..shown] {
        let budget = width.saturating_sub(indent.chars().count());
        lines.push(format!("{indent}{}", truncate_tail(name, budget)));
    }
    if names.len() > shown {
        lines.push(format!("{indent}… and {} more", names.len() - shown));
    }
    lines
}

/// Redraw `lines` in place: move back to the previous block's top, clear
/// downward, and reprint. Leaves the cursor on the line just below the block.
fn draw_block(lines: &[String], prev_lines: &mut usize) {
    use std::io::Write;
    let mut out = String::new();
    if *prev_lines > 0 {
        out.push_str(&format!("\x1b[{prev_lines}A")); // up to the block's first line
    }
    out.push_str("\r\x1b[0J"); // column 0, clear to end of screen
    out.push_str(&lines.join("\n"));
    out.push('\n'); // rest on the line below, so the count is stable frame-to-frame
    eprint!("{out}");
    let _ = std::io::stderr().flush();
    *prev_lines = lines.len();
}

/// Erase a previously drawn block (on finish), leaving the cursor at its top.
fn erase_block(prev_lines: usize) {
    if prev_lines > 0 {
        eprint!("\x1b[{prev_lines}A\r\x1b[0J");
    }
}

/// Human-readable elapsed time: `850ms`, `12.3s`, or `2m3s`.
fn format_elapsed(d: std::time::Duration) -> String {
    let secs = d.as_secs_f64();
    if secs < 1.0 {
        format!("{}ms", d.as_millis())
    } else if secs < 60.0 {
        format!("{secs:.1}s")
    } else {
        let mins = d.as_secs() / 60;
        format!("{mins}m{}s", d.as_secs() % 60)
    }
}

/// Keep the tail of `s` (a tensor name's informative end) within `max` columns,
/// prefixing `…` when truncated.
fn truncate_tail(s: &str, max: usize) -> String {
    let n = s.chars().count();
    if n <= max {
        return s.to_string();
    }
    if max <= 1 {
        return "…".to_string();
    }
    let tail: String = s.chars().skip(n - (max - 1)).collect();
    format!("…{tail}")
}

/// The tensors + metadata read from one checkpoint (local or remote).
type Loaded = (Vec<TensorInfo>, Vec<MetadataInfo>);

#[allow(clippy::too_many_arguments)] // a CLI entry point; each arg is a distinct flag
fn run_diff(
    old: &Path,
    new: &Path,
    recursive: bool,
    tensor: Option<&str>,
    view: sample::ViewDtype,
    bins: Option<usize>,
    opts: diff::DiffOpts,
    filter: &diff::TensorFilter,
    jobs: usize,
    remote: Option<&crate::remote::RemoteRead>,
) -> i32 {
    let load_local = |path: &Path| -> Result<Loaded> {
        let (files, _health) =
            collect_safetensors_files(std::slice::from_ref(&path.to_path_buf()), recursive, true)?;
        if files.is_empty() {
            anyhow::bail!("no checkpoint files found at {}", path.display());
        }
        Explorer::gather_checkpoint(&files, None)
    };

    let (old_str, new_str) = (old.to_string_lossy(), new.to_string_lossy());
    // Remote: read both checkpoints in parallel, one over each of two SSH sessions
    // (ssh2 sessions aren't Sync, so a session per thread). The password is entered
    // once and reused for the second session, so it's still one prompt; agent/key
    // auth needs none. A spinner line animates for each read. Local: sequential.
    let loaded: Result<(Loaded, Loaded)> = match remote {
        Some(r) => (|| -> Result<(Loaded, Loaded)> {
            // Open both sessions up front so the one password prompt happens here,
            // before the spinner. Opening is silent, so nothing is printed until
            // we're actually connected — then announce the read (not before, when
            // we're still authenticating and nothing is being read yet).
            let mut password: Option<String> = None;
            let sa = r.open_with(&mut password)?;
            let sb = r.open_with(&mut password)?;
            eprintln!("checkpoint-explorer diff: reading tensor metadata over ssh …");
            let bars = progress::Bars::start(vec![old_str.to_string(), new_str.to_string()]);
            let read =
                |session: &crate::sftp::RemoteSession, src: &str, i: usize| -> Result<Loaded> {
                    let progress = bars.progress(i);
                    let out = r
                        .read(session, src, &password, progress.as_deref())
                        .with_context(|| format!("reading {src}"));
                    bars.finish(i, out.is_ok());
                    out
                };
            let (ra, rb) = std::thread::scope(|s| {
                let (oref, nref): (&str, &str) = (&old_str, &new_str);
                let ta = s.spawn(|| read(&sa, oref, 0));
                let tb = s.spawn(|| read(&sb, nref, 1));
                (ta.join(), tb.join())
            });
            bars.join();
            let ra = ra.map_err(|_| anyhow::anyhow!("remote read thread panicked"))??;
            let rb = rb.map_err(|_| anyhow::anyhow!("remote read thread panicked"))??;
            Ok((ra, rb))
        })(),
        None => (|| {
            Ok((
                load_local(old).with_context(|| format!("reading {}", old.display()))?,
                load_local(new).with_context(|| format!("reading {}", new.display()))?,
            ))
        })(),
    };
    let ((old_t, old_m), (new_t, new_m)) = match loaded {
        Ok(v) => v,
        Err(e) => {
            eprintln!("checkpoint-explorer diff: {e:#}");
            return 2;
        }
    };

    let (old_label, new_label) = (old.display().to_string(), new.display().to_string());

    // Packing schemas (for the `unpacked` view) come from the full metadata —
    // independent of `--only-tensors`, which only hides the metadata *diff*. Only
    // needed when values / distributions are compared.
    let compares_data = opts.values || opts.histogram || tensor.is_some();
    let (old_schemas, new_schemas) = if compares_data {
        (
            sample::parse_packing_schemas(&old_t, &old_m),
            sample::parse_packing_schemas(&new_t, &new_m),
        )
    } else {
        (HashMap::new(), HashMap::new())
    };
    let ctx = ValueCtx {
        view,
        bins,
        old_schemas: &old_schemas,
        new_schemas: &new_schemas,
    };

    // `--tensor NAME`: focus on one tensor and also compare its element values.
    // (This single-tensor mode is its own selection; the subset filters apply to
    // the whole-checkpoint diff below, so note if both were given.)
    if let Some(name) = tensor {
        if filter.is_active() {
            eprintln!("checkpoint-explorer diff: --tensor takes precedence; filters ignored");
        }
        return run_diff_tensor(&old_label, &new_label, name, &old_t, &new_t, &ctx, opts);
    }

    // `--only-tensors` / an active filter (opts.metadata == false): drop metadata
    // so it affects neither the report nor the exit code (its section becomes a
    // "not compared" note in the output).
    let empty: Vec<MetadataInfo> = Vec::new();
    let old_meta: &[MetadataInfo] = if opts.metadata { &old_m } else { &empty };
    let new_meta: &[MetadataInfo] = if opts.metadata { &new_m } else { &empty };
    let mut old_sum = diff::CheckpointSummary::from_loaded(&old_t, old_meta);
    let mut new_sum = diff::CheckpointSummary::from_loaded(&new_t, new_meta);
    // Total distinct tensors across both sides *before* filtering, so the filter's
    // match line can show "matched M of N" (context for whether M looks right).
    let total_tensors = old_sum
        .tensors
        .keys()
        .chain(new_sum.tensors.keys())
        .collect::<std::collections::HashSet<_>>()
        .len();
    // Scope the diff to the selected subset (no-op when no filter was given).
    filter.apply(&mut old_sum, &mut new_sum);

    let report = if opts.values || opts.histogram {
        use rayon::prelude::*;
        // Read each common same-shape tensor and compare values and/or distribution.
        let old_map: HashMap<&str, &TensorInfo> =
            old_t.iter().map(|t| (t.name.as_str(), t)).collect();
        let new_map: HashMap<&str, &TensorInfo> =
            new_t.iter().map(|t| (t.name.as_str(), t)).collect();
        // The tensors present on both sides — the ones we actually read/compare.
        let common: Vec<&str> = old_sum
            .tensors
            .keys()
            .filter(|k| new_sum.tensors.contains_key(*k))
            .map(String::as_str)
            .collect();
        let progress = CompareProgress::start(common.len());
        let compute = |name: &str| -> diff::TensorExtras {
            let _tracked = progress.track(name);
            let (Some(a), Some(b)) = (old_map.get(name), new_map.get(name)) else {
                return diff::TensorExtras::default();
            };
            if a.shape != b.shape {
                return diff::TensorExtras::default(); // needs matching shapes
            }
            diff::TensorExtras {
                values: opts.values.then(|| tensor_values(a, b, &ctx)).flatten(),
                histogram: opts
                    .histogram
                    .then(|| tensor_histogram(a, b, &ctx))
                    .flatten(),
            }
        };
        // Reading tensor data is I/O-bound, so compare up to `jobs` tensors at
        // once (the results are order-independent). `jobs == 1` stays sequential.
        let pairs: Vec<(&str, diff::TensorExtras)> = if jobs <= 1 {
            common.iter().map(|&n| (n, compute(n))).collect()
        } else {
            match rayon::ThreadPoolBuilder::new().num_threads(jobs).build() {
                Ok(pool) => pool.install(|| common.par_iter().map(|&n| (n, compute(n))).collect()),
                Err(_) => common.iter().map(|&n| (n, compute(n))).collect(),
            }
        };
        progress.finish();
        // Feed the precomputed extras into the (pure) comparison. Each common name
        // is requested exactly once, so `remove` moves the value out (no clone).
        let extras: std::cell::RefCell<HashMap<&str, diff::TensorExtras>> =
            std::cell::RefCell::new(pairs.into_iter().collect());
        diff::compare_with(&old_sum, &new_sum, |name| {
            extras.borrow_mut().remove(name).unwrap_or_default()
        })
    } else {
        diff::compare(&old_sum, &new_sum)
    };
    // When a filter scoped the diff, say what it selected on stderr (so the diff
    // on stdout stays clean for piping): the match count — disambiguating an empty
    // diff caused by "0 matched" from "all identical" — plus the matched names
    // collapsed into their index-templated schema, so it's clear which layers /
    // experts the filter actually covered.
    if let Some(desc) = filter.describe() {
        // The matched set is the union of both sides (so it includes structurally
        // unchanged tensors, which the report only counts, not names).
        let mut names: Vec<&str> = old_sum
            .tensors
            .keys()
            .chain(new_sum.tensors.keys())
            .map(String::as_str)
            .collect();
        names.sort_unstable();
        names.dedup();
        if names.is_empty() {
            eprintln!(
                "checkpoint-explorer diff: filter [{desc}] matched 0 of {total_tensors} tensor(s)"
            );
        } else {
            eprintln!(
                "checkpoint-explorer diff: filter [{desc}] matched {} of {total_tensors} tensor(s):",
                names.len()
            );
            let schema = diff::name_schema(&names);
            const MAX_SCHEMA_LINES: usize = 40;
            for (tmpl, count) in schema.iter().take(MAX_SCHEMA_LINES) {
                if *count > 1 {
                    eprintln!("    {tmpl}  (×{count})");
                } else {
                    eprintln!("    {tmpl}");
                }
            }
            if schema.len() > MAX_SCHEMA_LINES {
                eprintln!(
                    "    … and {} more template(s)",
                    schema.len() - MAX_SCHEMA_LINES
                );
            }
        }
    }
    print!("{}", report.render(&old_label, &new_label, opts));
    i32::from(report.has_differences())
}

/// `compare_values` for two same-shape tensors under `ctx`, as an `Option`.
fn tensor_values(a: &TensorInfo, b: &TensorInfo, ctx: &ValueCtx) -> Option<sample::ValueDiff> {
    sample::compare_values(
        a,
        ctx.old_schemas.get(&a.name),
        b,
        ctx.new_schemas.get(&b.name),
        ctx.view,
    )
    .ok()
}

/// `histogram_diff` for two same-shape tensors under `ctx`, summarized to a shift.
fn tensor_histogram(a: &TensorInfo, b: &TensorInfo, ctx: &ValueCtx) -> Option<diff::HistShift> {
    let hd = sample::histogram_diff(
        a,
        ctx.old_schemas.get(&a.name),
        b,
        ctx.new_schemas.get(&b.name),
        ctx.view,
        ctx.bins,
    )
    .ok()?;
    Some(diff::HistShift {
        tvd: hd.tvd(),
        bins: hd.n,
    })
}

/// The `diff --tensor NAME` path: compare one tensor's signature and, when it's
/// in both checkpoints, its element values. Exits 2 if the name is in neither.
fn run_diff_tensor(
    old_label: &str,
    new_label: &str,
    name: &str,
    old_t: &[TensorInfo],
    new_t: &[TensorInfo],
    ctx: &ValueCtx,
    opts: diff::DiffOpts,
) -> i32 {
    let old_info = old_t.iter().find(|t| t.name == name);
    let new_info = new_t.iter().find(|t| t.name == name);
    if old_info.is_none() && new_info.is_none() {
        eprintln!("checkpoint-explorer diff: tensor '{name}' not found in either checkpoint");
        return 2;
    }

    let old_sig = old_info.map(diff::TensorSig::of);
    let new_sig = new_info.map(diff::TensorSig::of);
    // Compare values only when the tensor is in both checkpoints.
    let values = match (old_info, new_info) {
        (Some(a), Some(b)) => Some(value_cmp(a, b, ctx)),
        _ => None,
    };

    print!(
        "{}",
        diff::render_tensor_focus(
            old_label,
            new_label,
            name,
            old_sig.as_ref(),
            new_sig.as_ref(),
            values.as_ref(),
            opts.color,
        )
    );

    // `--histogram`: append the full bin-by-bin distribution table when the tensor
    // is in both with a matching shape.
    let mut hist_differs = false;
    if opts.histogram
        && let (Some(a), Some(b)) = (old_info, new_info)
        && a.shape == b.shape
    {
        match sample::histogram_diff(
            a,
            ctx.old_schemas.get(&a.name),
            b,
            ctx.new_schemas.get(&b.name),
            ctx.view,
            ctx.bins,
        ) {
            Ok(hd) => {
                hist_differs = hd.differs();
                print!("{}", diff::render_histogram_table(name, &hd, opts.color));
            }
            Err(e) => eprintln!("checkpoint-explorer diff: histogram: {e}"),
        }
    }

    let differs = diff::tensor_focus_differs(old_sig.as_ref(), new_sig.as_ref(), values.as_ref())
        || hist_differs;
    i32::from(differs)
}

/// Compare two tensors' values, mapping the result into a [`diff::ValueCmp`].
/// Shapes must match for an element-wise comparison; a mismatch (or a read /
/// dtype error) is reported as skipped rather than failing the whole diff.
fn value_cmp(a: &TensorInfo, b: &TensorInfo, ctx: &ValueCtx) -> diff::ValueCmp {
    if a.shape != b.shape {
        return diff::ValueCmp::Skipped("shapes differ".to_string());
    }
    match sample::compare_values(
        a,
        ctx.old_schemas.get(&a.name),
        b,
        ctx.new_schemas.get(&b.name),
        ctx.view,
    ) {
        Ok(vd) if vd.differing == 0 => diff::ValueCmp::Identical,
        Ok(vd) => diff::ValueCmp::Differ(vd),
        Err(e) => diff::ValueCmp::Skipped(e),
    }
}

/// Parse a `ROW,COL` pair of non-negative integers (the `--window` top-left).
/// Parse and bound the `--bins` histogram bucket count to `1..=512`.
fn parse_bins(s: &str) -> std::result::Result<usize, String> {
    match s.trim().parse::<usize>() {
        Ok(n) if (1..=512).contains(&n) => Ok(n),
        Ok(_) => Err("must be between 1 and 512".to_string()),
        Err(_) => Err(format!("expected a whole number, got '{s}'")),
    }
}

fn parse_offset_pair(s: &str) -> Result<(usize, usize)> {
    let (r, c) = s
        .split_once(',')
        .with_context(|| format!("expected ROW,COL (two integers), got '{s}'"))?;
    let row = r
        .trim()
        .parse()
        .with_context(|| format!("invalid row '{r}'"))?;
    let col = c
        .trim()
        .parse()
        .with_context(|| format!("invalid col '{c}'"))?;
    Ok((row, col))
}

/// Parse a `RFRAC,CFRAC` pair of fractions in `0..=1` (the `--edge` head/tail
/// split: 0 keeps only the first indices, 1 only the last, 0.5 is balanced).
fn parse_fraction_pair(s: &str) -> Result<(f32, f32)> {
    let (r, c) = s
        .split_once(',')
        .with_context(|| format!("expected RFRAC,CFRAC (two fractions 0..1), got '{s}'"))?;
    let row: f32 = r
        .trim()
        .parse()
        .with_context(|| format!("invalid row '{r}'"))?;
    let col: f32 = c
        .trim()
        .parse()
        .with_context(|| format!("invalid col '{c}'"))?;
    if !(0.0..=1.0).contains(&row) || !(0.0..=1.0).contains(&col) {
        anyhow::bail!("edge split fractions must be between 0 and 1, got '{s}'");
    }
    Ok((row, col))
}

/// Split an scp-style `[user@]host:path` into (host, path). Returns `None` for a
/// local path or an `s3://…` URI (no host to derive — that needs `--ssh-read`).
/// Matches `scp`'s own rule: a `:` before any `/`, with a non-empty host to its
/// left.
fn split_scp(s: &str) -> Option<(String, String)> {
    if s.starts_with("s3://") {
        return None;
    }
    let colon = s.find(':')?;
    if colon == 0 || s[..colon].contains('/') {
        return None;
    }
    Some((s[..colon].to_string(), s[colon + 1..].to_string()))
}

fn run_explore(mut args: ExploreArgs) -> Result<()> {
    if args.paths.is_empty() {
        eprintln!("checkpoint-explorer: no checkpoint given.\n");
        eprintln!("Usage:");
        eprintln!(
            "  checkpoint-explorer <PATH>...            browse a checkpoint (file, directory, or glob)"
        );
        eprintln!(
            "  checkpoint-explorer <PATH> --print-tree  dump its structure (text, or --format json)"
        );
        eprintln!("  checkpoint-explorer diff <OLD> <NEW>     compare two checkpoints");
        eprintln!(
            "  checkpoint-explorer --ssh-read <HOST> <s3://…|/remote/path>   read a remote / S3 checkpoint"
        );
        eprintln!("\nRun `checkpoint-explorer --help` for all options and examples.");
        std::process::exit(1);
    }

    // Support scp-style positional paths (`[user@]host:/path`) without an explicit
    // --ssh-read: derive the host and read the path part remotely, so
    // `checkpoint-explorer host:/opt/model` just works.
    if args.ssh_read.is_none()
        && let Some((host, _)) = args
            .paths
            .iter()
            .find_map(|p| split_scp(&p.to_string_lossy()))
    {
        let mut remote_paths = Vec::with_capacity(args.paths.len());
        for p in &args.paths {
            match split_scp(&p.to_string_lossy()) {
                Some((h, path)) if h == host => remote_paths.push(PathBuf::from(path)),
                _ => anyhow::bail!(
                    "can't mix local and scp-style ({host}:…) paths (or different hosts); \
                     list paths from one host, or use --ssh-read"
                ),
            }
        }
        args.paths = remote_paths;
        args.ssh_read = Some(host);
    }

    // `--ssh-read` delegates the read to cstorch on a remote host, so the s3://
    // URIs are kept verbatim (no local listing). Otherwise list local sources.
    let (files, health_reports) = if args.ssh_read.is_some() {
        (args.paths.clone(), Vec::new())
    } else {
        collect_safetensors_files(&args.paths, args.recursive, args.no_health_check)?
    };

    if files.is_empty() {
        eprintln!("Error: No checkpoint files found in the specified paths.");
        std::process::exit(1);
    }

    // Reading HDF5 needs the `hdf5` build feature; without it these files would
    // load as empty and the tree would misleadingly read "0 tensors, 0 params,
    // 0 B". Say so plainly instead. Directory scans already skip HDF5 when the
    // feature is off, so this only fires for files the user named explicitly.
    #[cfg(not(feature = "hdf5"))]
    {
        let hdf5: Vec<&PathBuf> = files
            .iter()
            .filter(|p| matches!(p.extension().and_then(|s| s.to_str()), Some("h5" | "hdf5")))
            .collect();
        if !hdf5.is_empty() {
            eprintln!(
                "Error: this build of checkpoint-explorer was compiled without HDF5 support, so it cannot read:"
            );
            for path in &hdf5 {
                eprintln!("  {}", path.display());
            }
            eprintln!();
            eprintln!("Rebuild and reinstall with the `hdf5` feature, e.g.:");
            eprintln!("  cargo install --path . --features hdf5");
            std::process::exit(1);
        }
    }

    // Flags that target the tree browser rather than a tensor view: with no
    // `--tensor` (and no data view), they make the tree the opened screen, so
    // e.g. `--expand-all` or `--legend` alone don't demand a tensor.
    let tree_oriented =
        args.tree || args.tree_state.is_some() || args.search.is_some() || args.legend;
    let view = if args.values {
        OpenView::Values
    } else if args.heatmap {
        OpenView::Heatmap
    } else if args.tree || (tree_oriented && args.tensor.is_none()) {
        OpenView::Tree
    } else {
        OpenView::Detail
    };
    let layout = if args.window.is_some() {
        Some(DataLayout::Window)
    } else if args.edge.is_some() {
        Some(DataLayout::Edges)
    } else if args.overview {
        Some(DataLayout::Overview)
    } else {
        None
    };
    // Position within the layout: the window's top-left corner, or the edges
    // head/tail split — parsed from the optional `--window`/`--edge` value.
    let window_at = args.window.as_deref().map(parse_offset_pair).transpose()?;
    let edge_split = args.edge.as_deref().map(parse_fraction_pair).transpose()?;
    // Seed an open request when a tensor is named *or* any view/override flag is
    // given — the latter targets the sole tensor when the checkpoint has one.
    let wants_open = args.tensor.is_some()
        || args.metadata.is_some()
        || args.values
        || args.heatmap
        || args.tree
        || args.dtype.is_some()
        || args.edge.is_some()
        || args.overview
        || args.window.is_some()
        || args.zebra.is_some()
        || args.base.is_some()
        || args.slice.is_some()
        || args.shape.is_some()
        || args.compute_stats
        || args.histogram
        || args.bins.is_some()
        || args.tree_state.is_some()
        || args.search.is_some()
        || args.legend
        || args.exit;
    let open = wants_open.then_some(OpenRequest {
        tensor: args.tensor,
        metadata: args.metadata,
        view,
        histogram: args.histogram,
        bins: args.bins,
        dtype: args.dtype,
        layout,
        window_at,
        edge_split,
        zebra: args.zebra,
        base: args.base,
        slice: args.slice,
        shape: args.shape,
        compute_stats: args.compute_stats,
        tree_state: args.tree_state,
        search: args.search,
        legend: args.legend,
        exit_after: args.exit,
    });

    let mut explorer = Explorer::new(files, health_reports, open, !args.no_preload);
    if let Some(host) = args.ssh_read {
        let venv = args.ssh_venv.unwrap_or_else(|| "~/venv".to_string());
        explorer.set_remote_read(host, venv);
    }
    // One-shot exports: print the tree / tensor list and exit (honour --format,
    // -v, and the --name filter), before any interactive or --plain rendering.
    if args.print_tree || args.print_tensors {
        let detail = explorer::TreeDetail::from_verbosity(args.verbose);
        let filter = filter::NameFilter::parse(&args.name)?;
        return if args.print_tree {
            explorer.print_tree(args.format, detail, &filter)
        } else {
            explorer.print_tensors(args.format, detail, &filter)
        };
    }

    if args.emit_command {
        explorer.render_plain(true)
    } else if args.plain {
        explorer.render_plain(false)
    } else {
        explorer.run()
    }
}

#[cfg(feature = "hdf5")]
fn run_convert(
    input: &Path,
    output: &Path,
    codec: codec::Codec,
    level: Option<u8>,
    buffer: &str,
    force: bool,
) -> Result<()> {
    use anyhow::bail;
    use std::io::Write;

    let ext = input.extension().and_then(|e| e.to_str());
    if !matches!(ext, Some("h5" | "hdf5")) {
        bail!(
            "convert only supports HDF5 inputs (.h5/.hdf5), got: {}",
            input.display()
        );
    }
    // Refuse to read and write the same file (checked before --force removes the
    // output, so we never delete the input).
    if std::path::absolute(input).ok() == std::path::absolute(output).ok()
        && std::path::absolute(input).is_ok()
    {
        bail!("input and output are the same file: {}", input.display());
    }
    // Warn when the target codec is what the source already uses (a re-encode;
    // a plain file copy would be equivalent).
    if convert::source_codec(input) == Some(codec) {
        eprintln!(
            "warning: source is already {}; repacking just re-encodes it — a plain copy would be equivalent",
            codec.label()
        );
    }
    if force && output.exists() {
        fs::remove_file(output)
            .with_context(|| format!("removing existing {}", output.display()))?;
    }

    let level = codec.clamp_level(level.unwrap_or_else(|| codec.default_level()));
    let buffer_bytes = utils::parse_size(buffer).map_err(anyhow::Error::msg)?;
    let opts = convert::Options {
        codec,
        level,
        buffer_bytes,
    };
    let level_note = if codec.uses_level() {
        format!(" level {level}")
    } else {
        String::new()
    };
    eprintln!(
        "Repacking {} → {} ({}{level_note}, {} buffer)",
        input.display(),
        output.display(),
        codec.label(),
        utils::format_size(buffer_bytes),
    );

    let mut stderr = std::io::stderr();
    let report = convert::convert_hdf5(input, output, &opts, |done, total, name| {
        let bar = progress_bar(done, total, 28);
        let _ = write!(stderr, "\r{bar} [{done}/{total}] {name:.<48}\x1b[K");
        let _ = stderr.flush();
    })?;
    eprintln!("\rDone: {}\x1b[K", report.summary(codec));
    Ok(())
}

/// A `[####----]` progress bar of the given width.
#[cfg(feature = "hdf5")]
fn progress_bar(done: usize, total: usize, width: usize) -> String {
    let filled = (done * width).checked_div(total).unwrap_or(0);
    format!(
        "[{}{}]",
        "#".repeat(filled),
        "-".repeat(width.saturating_sub(filled))
    )
}

#[cfg(not(feature = "hdf5"))]
fn run_convert(
    _input: &Path,
    _output: &Path,
    _codec: codec::Codec,
    _level: Option<u8>,
    _buffer: &str,
    _force: bool,
) -> Result<()> {
    anyhow::bail!("`convert` requires building with `--features hdf5`")
}

fn collect_safetensors_files(
    paths: &[PathBuf],
    recursive: bool,
    no_health_check: bool,
) -> Result<(Vec<PathBuf>, Vec<health::HealthReport>)> {
    let mut files = Vec::new();
    let mut health_reports = Vec::new();

    for path in paths {
        // Remote checkpoints are read via `--ssh-read` (handled before this
        // function); a bare `s3://` here has no local credentials to read it with.
        let raw = path.to_string_lossy();
        if s3::is_uri(&raw) {
            eprintln!("Warning: {raw}: reading an s3:// checkpoint needs --ssh-read <[user@]host>");
            continue;
        }

        // Try to expand as glob pattern
        let expanded_paths: Vec<PathBuf> = match glob::glob(&path.to_string_lossy()) {
            Ok(paths) => paths.filter_map(Result::ok).collect(),
            Err(_) => vec![path.clone()], // Not a valid glob, treat as literal path
        };

        // Process each expanded path
        for expanded_path in expanded_paths {
            if !expanded_path.exists() {
                eprintln!("Warning: Path does not exist: {}", expanded_path.display());
                continue;
            }

            if expanded_path.is_file() {
                let ext = expanded_path.extension().and_then(|s| s.to_str());
                if matches!(
                    ext,
                    Some("safetensors" | "gguf" | "h5" | "hdf5" | "npy" | "npz")
                ) {
                    files.push(expanded_path.clone());
                } else {
                    eprintln!(
                        "Warning: Skipping unsupported file: {}",
                        expanded_path.display()
                    );
                }
            } else if expanded_path.is_dir() {
                // Check for SafeTensors index file first
                let index_path = expanded_path.join("model.safetensors.index.json");
                let mut found_from_index = false;
                if index_path.exists() {
                    let index_files = parse_safetensors_index(&index_path)?;
                    let mut missing = Vec::new();
                    for file in index_files {
                        let full_path = expanded_path.join(&file);
                        if full_path.exists() {
                            files.push(full_path);
                            found_from_index = true;
                        } else {
                            missing.push(file);
                        }
                    }
                    if !missing.is_empty() {
                        eprintln!(
                            "Warning: {} file(s) listed in {} were not found on disk (e.g. {}).",
                            missing.len(),
                            index_path.display(),
                            missing[0],
                        );
                    }
                    if !found_from_index {
                        eprintln!(
                            "Warning: index file references no existing files (it may be stale); scanning {} directly instead.",
                            expanded_path.display()
                        );
                    }

                    // Health check: compare the index against the files on disk
                    // and record any mismatch to surface in the UI.
                    if !no_health_check
                        && let Ok(report) = health::check(&expanded_path, &index_path)
                        && report.has_issues()
                    {
                        health_reports.push(report);
                    }
                }

                // Always scan the directory as well, so files present on disk
                // but missing from the index — a partially-stale index, e.g.
                // extra `codebooks`/`qscales` shards — still show up, alongside
                // the no-index and fully-stale cases. Duplicates (a shard listed
                // in the index *and* found by the scan) are removed below.
                scan_directory(&expanded_path, recursive, &mut files)?;
            }
        }
    }

    // Sort for consistent ordering and drop duplicates — the same file can be
    // collected both from the index and the directory scan (identical paths).
    files.sort();
    files.dedup();
    Ok((files, health_reports))
}

fn scan_directory(dir: &Path, recursive: bool, files: &mut Vec<PathBuf>) -> Result<()> {
    // HDF5 is only scanned for when compiled in, to avoid surfacing files the
    // build cannot read.
    let exts: &[&str] = if cfg!(feature = "hdf5") {
        &["safetensors", "gguf", "h5", "hdf5", "npy", "npz"]
    } else {
        &["safetensors", "gguf", "npy", "npz"]
    };
    let glob_prefix = if recursive { "**/" } else { "" };
    let patterns: Vec<String> = exts
        .iter()
        .map(|ext| format!("{}/{glob_prefix}*.{ext}", dir.display()))
        .collect();

    for pattern in patterns {
        for entry in glob::glob(&pattern).context("Failed to read glob pattern")? {
            match entry {
                Ok(file_path) => files.push(file_path),
                Err(e) => eprintln!("Warning: Error reading file: {e}"),
            }
        }
    }

    Ok(())
}

fn parse_safetensors_index(index_path: &PathBuf) -> Result<Vec<String>> {
    let content = fs::read_to_string(index_path)
        .with_context(|| format!("Failed to read index file: {}", index_path.display()))?;

    let index: serde_json::Value = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse index file: {}", index_path.display()))?;

    let mut files = Vec::new();

    if let Some(weight_map) = index.get("weight_map").and_then(|v| v.as_object()) {
        for file_name in weight_map.values() {
            if let Some(file_str) = file_name.as_str()
                && !files.iter().any(|existing| existing == file_str)
            {
                files.push(file_str.to_string());
            }
        }
    }

    files.sort();
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn splits_scp_style_paths_only() {
        assert_eq!(
            split_scp("net004:/opt/models/m"),
            Some(("net004".into(), "/opt/models/m".into()))
        );
        assert_eq!(
            split_scp("lab@host:rel/path"),
            Some(("lab@host".into(), "rel/path".into()))
        );
        // local paths and s3 URIs are not scp targets
        assert_eq!(split_scp("/opt/models/m"), None);
        assert_eq!(split_scp("./model.safetensors"), None);
        assert_eq!(split_scp("s3://bucket/key"), None);
        assert_eq!(split_scp("dir/a:b"), None); // colon after a slash → local
    }

    #[test]
    fn format_elapsed_scales() {
        use std::time::Duration;
        assert_eq!(format_elapsed(Duration::from_millis(850)), "850ms");
        assert_eq!(format_elapsed(Duration::from_millis(1500)), "1.5s");
        assert_eq!(format_elapsed(Duration::from_secs(125)), "2m5s");
    }

    #[test]
    fn truncate_tail_keeps_the_end() {
        assert_eq!(truncate_tail("short", 10), "short"); // fits
        assert_eq!(truncate_tail("abcdefgh", 8), "abcdefgh"); // exact
        assert_eq!(truncate_tail("abcdefgh", 4), "…fgh"); // …+tail, total == max
        assert_eq!(truncate_tail("abcdefgh", 1), "…");
        // A long tensor name keeps its most-specific tail within the budget.
        let name = "model.layers.0.block_sparse_moe.experts.down_proj.weight";
        let t = truncate_tail(name, 20);
        assert_eq!(t.chars().count(), 20);
        assert!(t.starts_with('…') && t.ends_with("down_proj.weight"), "{t}");
    }

    /// A directory whose index lists only some of the `.safetensors` on disk (a
    /// partially-stale index) must still surface the extra files — the bug where
    /// `codebooks`/`qscales` shards were silently dropped.
    #[test]
    fn collects_extra_files_absent_from_a_stale_index() {
        let dir = std::env::temp_dir().join("ckpt_explorer_stale_index_test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        // Three shards on disk; the index references only the first.
        fs::write(dir.join("model.safetensors"), b"x").unwrap();
        fs::write(dir.join("codebooks.safetensors"), b"x").unwrap();
        fs::write(dir.join("qscales.safetensors"), b"x").unwrap();
        fs::write(
            dir.join("model.safetensors.index.json"),
            br#"{"weight_map": {"w": "model.safetensors"}}"#,
        )
        .unwrap();

        let (files, _) =
            collect_safetensors_files(std::slice::from_ref(&dir), false, true).unwrap();
        let names: Vec<String> = files
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();

        for want in [
            "model.safetensors",
            "codebooks.safetensors",
            "qscales.safetensors",
        ] {
            assert!(
                names.iter().any(|n| n == want),
                "{want} should be collected; got {names:?}"
            );
        }
        // The shard listed in the index *and* found by the scan must appear once.
        let unique: std::collections::HashSet<_> = files.iter().collect();
        assert_eq!(files.len(), unique.len(), "duplicate paths: {names:?}");

        let _ = fs::remove_dir_all(&dir);
    }
}
