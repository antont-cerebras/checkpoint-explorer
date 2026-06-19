# Checkpoint Explorer

An interactive terminal-based explorer for [`safetensors`](https://huggingface.co/docs/safetensors) and [GGUF](https://huggingface.co/docs/hub/gguf) files, designed to help you visualize and navigate the structure of machine learning models.

![Demo](demo.gif)

## Features

- 🔍 **Interactive browsing** of `safetensors` and GGUF file structures
- 📁 **Hierarchical tree view** with expandable/collapsible groups
- 🔎 **Fuzzy search** - instantly filter tensors with fuzzy matching using `/` key
- 🔢 **Smart numeric sorting** for layer numbers (e.g., layer.0, layer.1, layer.2, ..., layer.10)
- 📊 **Tensor details** including shape, data type, and size
- 🔗 **Multi-file support** - automatically merges multiple files into a unified view
- 📂 **Directory support** - explore entire model directories with automatic `safetensors` index detection
- 🌟 **Glob pattern support** - use wildcards to select multiple files (e.g., `*.safetensors`, `model-*.gguf`)
- 📏 **Human-readable sizes** (B, KB, MB, GB)
- ⌨️ **Keyboard navigation** for smooth exploration
- 🧠 **GGUF support** - view GGML format tensors with quantization types
- 🧊 **HDF5 checkpoint support** (opt-in `--features hdf5`) - read Cerebras-style
  `.h5`/`.hdf5` checkpoints, showing compression status and both the logical and
  on-disk (compressed) sizes

## Installation

### Install
```bash
cargo install --git https://github.com/antont-cerebras/checkpoint-explorer
```

### Prerequisites
- Rust (1.70 or later)

### Build from source
```bash
git clone https://github.com/antont-cerebras/checkpoint-explorer
cd checkpoint-explorer
cargo build --release
```

### HDF5 checkpoint support (optional)
Reading Cerebras-style HDF5 checkpoints is behind the `hdf5` feature, which is
off by default so the standard build stays pure-Rust with no system
dependencies. Enabling it bundles and statically links libhdf5 (requires a C
toolchain + `cmake`; the first build is slower):
```bash
cargo install --git <repo-url> --features hdf5
# or from source:
cargo build --release --features hdf5
```

## Usage

### Basic usage
```bash
# Explore a single safetensors file
checkpoint-explorer model.safetensors

# Explore a GGUF file
checkpoint-explorer model.gguf

# Or if building from source
cargo run -- model.safetensors
cargo run -- model.gguf
```

### Directory exploration
```bash
# Explore all safetensors and GGUF files in a directory
checkpoint-explorer /path/to/model/directory

# Recursively search subdirectories
checkpoint-explorer -r /path/to/models

# The tool automatically detects and uses model.safetensors.index.json if present
checkpoint-explorer /path/to/huggingface/model
```

### Multi-file exploration
```bash
# Explore multiple files as a unified model
checkpoint-explorer model-00001-of-00003.safetensors model-00002-of-00003.safetensors model-00003-of-00003.safetensors

# Mix safetensors and GGUF files
checkpoint-explorer model.safetensors model.gguf

# Mix files and directories
checkpoint-explorer model.safetensors /path/to/additional/models
```

### Glob pattern support
```bash
# Use wildcards to select multiple files
checkpoint-explorer *.safetensors

# Match files with specific patterns
checkpoint-explorer model-*.gguf

# Match numbered checkpoint files
checkpoint-explorer checkpoint-[0-9]*.safetensors

# Combine multiple patterns
checkpoint-explorer *.safetensors *.gguf

# Mix glob patterns with explicit paths
checkpoint-explorer model.safetensors checkpoint-*.safetensors
```

### Open a tensor directly
Jump straight to a tensor's preview on startup instead of navigating the tree —
handy for scripting or revisiting a known tensor:
```bash
# Open a tensor's detail screen
checkpoint-explorer model.hdf5 --tensor model.layers.0.mlp.down_proj.weight

# Open straight into the numeric values grid, reinterpreted as packed 4-bit,
# in the first/last edges submode
checkpoint-explorer model.hdf5 \
  --tensor model.layers.0.block_sparse_moe.experts.down_proj.weight \
  --dtype u4-packed --values --edge
```
Flags (all require `--tensor`):

| Flag | Effect |
|------|--------|
| `--tensor <NAME>` | Open this tensor (exact name) on startup |
| `--values` / `--heatmap` | Open the numeric grid / the heatmap (default: the detail screen) |
| `--dtype <DTYPE>` | Reinterpret the dtype: `u4-packed`, `u4-lo`, `u4-hi`, `i4-packed`, `i4-lo`, `i4-hi`, or `f16`/`bf16`/`i16`/`u16`/`f32`/`i32`/`u32`/`f64`/`i64`/`u64`/`i8`/`u8`/`stored` |
| `--edge` (alias `--edges`) / `--overview` | Force the first/last edges submode / the evenly-spaced overview |
| `--zebra <rows\|cols\|off>` | Zebra-stripe the numeric grid by rows, columns, or off |
| `--slice <INDEX>` | Starting slice for a 3D tensor: an index (`12`) or a percentage (`50%`) |

Dismissing the opened screen drops you into the normal tree browser, so you can
keep exploring.

### Keyboard Controls

| Key | Action |
|-----|--------|
| `↑` / `↓` | Navigate up/down through the tree |
| `←` | Jump to the parent group |
| `→` | Enter the group (expand if needed) and select its first child |
| `Shift`+`↑` / `Shift`+`↓` | Jump to the previous / next sibling |
| `Enter` / `Space` | Expand/collapse groups, view tensor details |
| `E` / `C` | Expand all / collapse all groups |
| `/` | Enter search mode to filter tensors |
| `c` | Copy the selected row's path (tensor file, or a group/root's file or directory) |
| `h` | Show the checkpoint health report (when there is a mismatch) |
| `Esc` | Exit search mode |
| `q` | Quit the application (or exit search mode if active) |
| `Ctrl+C` | Force quit |

A status bar pinned to the bottom shows the file the selected tensor lives in
(or, for a group/root, the single file or the shared directory of its tensors).
Pressing `c` copies that path to the clipboard via the OSC 52 terminal escape —
so copying the root yields the file or the checkpoint directory — and it works
over SSH/tmux when the terminal supports it.

### Search Feature

Press `/` to enter search mode and start typing to filter tensors by name. The search:
- Uses **fuzzy matching** - find tensors even with typos or partial matches (e.g., "attnproj" will match "attn.c_proj.weight")
- Searches **all tensors** - not just visible ones, regardless of collapsed groups
- Shows results in a **flat list** with full tensor names
- Sorts by **relevance** - best matches appear first

Press `Enter` to open the highlighted result's details (you stay in search), and `Esc` or `q` to exit search mode and return to the full tree view.

### Tensor data preview

From a tensor's detail screen (open it with `Enter`/`Space`), you can preview the
actual data of 1D/2D/3D tensors:

- `m` — an **ASCII heatmap**: each sampled element is a colored block on a
  blue→green→red scale, with a min/max legend. Each character row packs two
  data rows (upper/lower half-block) for higher vertical resolution.
- `v` — a **numeric grid** of sampled values with row/column indices, including
  the edges. Column width adapts to the *actual* values (from the exact stats),
  so a tensor of small numbers packs many more columns onto the screen — even a
  16-bit dtype whose values happen to be one- or two-digit (common for sparse /
  quantized formats) gets as many columns as a 4-bit view, instead of always
  reserving room for `-32768`. Floats use a fixed scientific-notation width.
  When an index label is wider than its (narrow) cell, the labels are staggered
  across two header rows ("leap-frog"), and — for the most extreme cases —
  thinned so the ones shown don't collide. To make the grid easier on the eyes
  the indices are dimmed and the cells get a subtle alternating "zebra"
  background (a constant-width band per row or column); `z` cycles the striping
  between rows, columns, and off.

Both views open in the **edges view** by default — the first *and* last rows and
columns (as many as fit), with a dotted `⋯` / `⋮` separator marking the skipped
middle — handy for seeing how a tensor is padded at its edges (e.g. zero padding
vs. something else). Press `e` to toggle the evenly-spaced **overview** instead.
In the edges view the **arrow keys move the divider** between the first
and last blocks: `←` / `→` slide the column divider (so `→` grows the first
columns and shrinks the last, `←` the reverse), `↑` / `↓` slide the row divider,
and holding `Shift` pushes it all the way to one end (e.g. `Shift`+`←` to see
only the last columns). The header shows the current split (e.g.
`first 8 & last 18`). The edges choice and the split are remembered for the
session, so they stick as you move between tensors.

For **3D tensors** (e.g. stacked MoE experts, shape `[experts, rows, cols]`) the
preview shows a 2D matrix at a fixed leading index — the 0th by default. The
`←` / `→` arrows step through the slices one at a time and `Shift`+`←` / `→` jump
~5% at a time (both wrap around at the ends); `/` prompts for a slice to jump to —
either an exact index or a percentage like `50%` (0% = first, 100% = last).
Out-of-range entries are rejected with a message rather than jumping. (In the
edges view the arrows move the first/last divider instead, so there slices step
with `[` / `]` — `/` still works.)
Within either view, `m` and `v` switch between the heatmap and numeric
representations in place, `e` toggles the edges view, and `Ctrl+C` quits the app
from anywhere.

#### Statistics

The heatmap/numeric views show **exact whole-tensor statistics** — value range,
mean, standard deviation, % zeros (sparsity) and a non-finite (NaN/Inf) count —
computed by scanning every element once (`safetensors` are memory-mapped and
decoded in parallel with `rayon`; HDF5 datasets are streamed in row-blocks so
memory stays bounded regardless of tensor size — including LZ4-compressed
Cerebras checkpoints, which are decompressed in-process). The heatmap's color
scale uses the exact range, so colors mean the same thing across slices. The
detail screen shows the same stats on demand: press `s` to compute them (so
browsing the tree stays fast). Results are cached per tensor (and per dtype
override) for the session, and the scan time is shown dimmed next to the stats.
Scanning a multi-GB tensor takes a moment the first time (it's largely
disk/NFS-bound); the scan runs on a worker thread with an animated spinner and a
running timer, and `Ctrl+C` cancels.

Both views also sample a grid that fits the screen (they never read the whole
tensor for the *display* — only each sampled row's column span). Both the
statistics and the data preview work for `safetensors` and HDF5
(`--features hdf5`) of any size, reached through one format-agnostic reader; GGUF
data preview is not yet supported.

#### Dtype override

When the stored dtype misrepresents the data — common for quantized checkpoints
where 4-bit weights are packed into a `bf16`/`f16`/`i16` slot — press `d`
(safetensors or HDF5) to open a menu of alternative interpretations, just for
visualization. It reinterprets the raw stored bytes, so for HDF5 it applies to
both the statistics and (for previewable sizes) the data views.
This works from both the tensor **detail** screen and the heatmap/numeric views;
the detail screen updates its dtype, shape and parameter count to match. The menu
previews each option live as you move through it (`←`/`→` or `d`/`D` to move,
`Enter` to apply, `Esc` to cancel); the choice is remembered per tensor until you
quit. Options for a 16-bit tensor:

- another **same-width** dtype, e.g. view a `BF16` tensor as `F16` / `I16` / `U16`;
- **`u4`/`i4` (low nibble)** or **(high nibble)** — one 4-bit value from the
  low / high nibble of each slot (formats differ on which nibble holds the data);
- **`u4`/`i4` (packed)** — every nibble unpacked densely, so each 16-bit slot
  yields four values and the last dimension grows ×4.

The header shows the active reinterpretation (e.g. `BF16 as u4 (packed)`).

## Example Output

```
Checkpoint Explorer - model.safetensors (1/1)
Use ↑/↓ to navigate, Enter/Space to expand/collapse, / to search, c to copy path, q to quit
================================================================================

▼ 📦 model.safetensors (342 tensors, 1.5B params, 1.2 GiB)
  ▼ 📁 transformer (123 tensors, 1.2 GiB)
    ▼ 📁 h (32 layers, 120 tensors, 1.1 GiB)
      ▼ 📁 0 (5 tensors, 45.2 MiB)
        📄 attn.c_attn.weight [Float16, (4096, 3072), 25.2 MiB]
        📄 attn.c_proj.weight [Float16, (1024, 4096), 8.4 MiB]
        📄 ln_1.weight [Float16, (4096,), 8.2 KiB]
        📄 mlp.c_fc.weight [Float16, (4096, 11008), 90.1 MiB]
        📄 mlp.c_proj.weight [Float16, (11008, 4096), 90.1 MiB]
      ▶ 📁 1 (5 tensors, 45.2 MiB)
      ▶ 📁 2 (5 tensors, 45.2 MiB)
      ...
      ▶ 📁 31 (5 tensors, 45.2 MiB)
    📄 ln_f.weight [Float16, (4096,), 8.2 KiB]
    📄 wte.weight [Float16, (151936, 4096), 1.2 GiB]

/path/to/model.safetensors
```

The **root** node summarises the whole checkpoint (tensor count, parameters and
size). The bottom **status bar** shows the source file of the selected row — or,
for a directory of shards, `N files in <dir>`.

## How It Works

1. **Path Resolution**: Automatically discovers `safetensors` files from files, directories, or `safetensors` index files
2. **File Loading**: Loads one or more `safetensors` files and extracts tensor metadata
3. **Tree Building**: Organizes tensors into a hierarchical structure based on their names (split by '.')
4. **Smart Sorting**: Uses natural sorting to handle numeric components correctly
5. **Interactive Display**: Renders the tree with expansion/collapse functionality
6. **Tensor Details**: Shows detailed information when selecting individual tensors

## Technical Details

### Supported Formats
- `safetensors` files (`.safetensors`)
- GGUF files (`.gguf`) with GGML tensor types including quantized formats
- HDF5 checkpoints (`.h5`/`.hdf5`) when built with `--features hdf5` — Cerebras
  layout (URL-quoted tensor names as top-level datasets), with per-tensor
  compression markers (e.g. `lz4`, `gzip`) and on-disk sizes
- `safetensors` index files (`model.safetensors.index.json`)
- Directory scanning with recursive search option
- All tensor data types supported by the `safetensors` and GGML formats

### Performance
- Memory efficient: Only loads tensor metadata, not the actual tensor data
- Fast startup: Optimized for quick exploration of large models
- Responsive UI: Smooth navigation even with thousands of tensors

## Dependencies

- `safetensors` - For reading `safetensors` files
- `gguf` - For reading GGUF files
- `crossterm` - For terminal UI and keyboard input
- `clap` - For command-line argument parsing
- `anyhow` - For error handling
- `serde_json` - For parsing `safetensors` index files
- `glob` - For directory pattern matching

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
