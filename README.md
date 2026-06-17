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
| `c` | Copy the selected tensor's file path to the clipboard |
| `h` | Show the checkpoint health report (when there is a mismatch) |
| `Esc` | Exit search mode |
| `q` | Quit the application (or exit search mode if active) |
| `Ctrl+C` | Force quit |

A status bar above the footer always shows the file the tensor under the
cursor lives in (or, for a group, how many files its tensors span). Pressing
`c` copies that path to the clipboard via the OSC 52 terminal escape, so it
works over SSH/tmux when the terminal supports it.

### Search Feature

Press `/` to enter search mode and start typing to filter tensors by name. The search:
- Uses **fuzzy matching** - find tensors even with typos or partial matches (e.g., "attnproj" will match "attn.c_proj.weight")
- Searches **all tensors** - not just visible ones, regardless of collapsed groups
- Shows results in a **flat list** with full tensor names
- Sorts by **relevance** - best matches appear first

Press `Enter` or `Esc` to exit search mode and return to the full tree view.

## Example Output

```
Checkpoint Explorer - model.safetensors (1/1)
Use ↑/↓ to navigate, Enter/Space to expand/collapse, / to search, c to copy path, q to quit
================================================================================

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
Total Parameters: 1.5B | Selected: 1/342 | Scroll: 0 | Matches: 342
```

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
