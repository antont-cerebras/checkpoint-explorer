# demo.tape — regenerate demo.gif with:  vhs demo.tape
#
# Requires: vhs (https://github.com/charmbracelet/vhs), python3, and
# `checkpoint-explorer` on PATH (`cargo install --path . --features hdf5`).
#
# A short tour of the current feature set: browse the tree, search, a tensor's
# statistics + histogram, the heatmap and numeric-grid data views, decoding a
# packed 4-bit weight on the fly, and a coloured structural `diff`.
#
# Note: timings and navigation counts are a sensible starting point — tweak the
# Sleep durations / Down counts to taste, then commit the regenerated demo.gif.

Output demo.gif
Require checkpoint-explorer

Set Shell bash
Set FontSize 15
Set Width 1180
Set Height 700
Set Padding 16
Set Theme "Dracula"

# --- setup (hidden): build the small demo checkpoints -----------------------
Hide
Type "python3 gen_demo.py && clear" Enter
Sleep 2s
Show

# --- 1) Browse the tree: grouped layers, sizes, metadata --------------------
Type "checkpoint-explorer /tmp/ckpt-demo/model.safetensors --tree-state expanded" Enter
Sleep 2.5s
Down 8
Sleep 1.5s
Type "/"
Sleep 400ms
Type "down_proj"
Sleep 2s
Escape
Sleep 1s
Ctrl+C

# --- 2) A tensor's detail: stats + histogram, heatmap, numeric grid ---------
Type "checkpoint-explorer /tmp/ckpt-demo/model.safetensors --tensor model.layers.0.mlp.down_proj.weight --compute-stats" Enter
Sleep 2.5s
Type "h"
Sleep 2s
Type "m"
Sleep 2.5s
Backspace
Sleep 700ms
Type "v"
Sleep 2.5s
Ctrl+C

# --- 3) Decode a packed 4-bit weight on the fly (U8 -> u4) ------------------
Type "checkpoint-explorer /tmp/ckpt-demo/model.safetensors --tensor model.layers.0.mlp.gate_proj.qweight --dtype u4 --values" Enter
Sleep 3s
Ctrl+C

# --- 4) Compare two checkpoints: coloured structural diff -------------------
Type "checkpoint-explorer diff /tmp/ckpt-demo/old.safetensors /tmp/ckpt-demo/new.safetensors" Enter
Sleep 4s
