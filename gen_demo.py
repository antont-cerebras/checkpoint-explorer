#!/usr/bin/env python3
"""Generate the small demo checkpoints used by demo.tape.

Stdlib only (no numpy) — writes valid `.safetensors` files into /tmp/ckpt-demo:

  model.safetensors   a little transformer-ish checkpoint (real F32 data so the
                      heatmap / grid / histogram / stats views look interesting,
                      plus a packed U8 weight for the `--dtype u4` decode demo)
  old.safetensors     a two-checkpoint pair that differs in dtype, shape, an
  new.safetensors     added/removed tensor, and metadata — for the `diff` demo
"""

import json
import os
import random
import struct

random.seed(7)
OUT = "/tmp/ckpt-demo"
os.makedirs(OUT, exist_ok=True)

DT_SIZE = {"F32": 4, "F16": 2, "BF16": 2, "U16": 2, "U8": 1, "I8": 1}


def numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def write_safetensors(path, tensors, metadata):
    """tensors: list of (name, dtype, shape, data_bytes)."""
    header, blob = {}, bytearray()
    for name, dtype, shape, data in tensors:
        start = len(blob)
        blob += data
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, len(blob)],
        }
    if metadata:
        header["__metadata__"] = metadata
    hb = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(bytes(blob))


def f32(shape):
    """A real-valued F32 tensor (standard-normal) so the data views look alive."""
    return struct.pack("<%df" % numel(shape), *[random.gauss(0.0, 1.0) for _ in range(numel(shape))])


def u8(shape):
    return bytes(random.randrange(256) for _ in range(numel(shape)))


def zeros(dtype, shape):
    return b"\x00" * (numel(shape) * DT_SIZE[dtype])


# --- the explore-tour model: 6 layers (so names group as {0-5}) --------------
model = [
    ("model.embed_tokens.weight", "F32", (128, 48), f32((128, 48))),
    ("lm_head.weight", "F32", (128, 48), f32((128, 48))),
    ("model.norm.weight", "F32", (48,), f32((48,))),
]
for i in range(6):
    model.append((f"model.layers.{i}.self_attn.q_proj.weight", "F32", (48, 48), f32((48, 48))))
    model.append((f"model.layers.{i}.mlp.down_proj.weight", "F32", (48, 96), f32((48, 96))))
    # a 4-bit-packed weight stored as U8 (two nibbles/byte) — for `--dtype u4`
    model.append((f"model.layers.{i}.mlp.gate_proj.qweight", "U8", (48, 24), u8((48, 24))))
write_safetensors(
    f"{OUT}/model.safetensors",
    model,
    {"format": "pt", "producer": "checkpoint-explorer demo"},
)

# --- the diff pair: dtype change, shape change, add, remove, metadata edit ----
old = [
    ("model.embed_tokens.weight", "F32", (128, 48), zeros("F32", (128, 48))),
    ("model.layers.0.mlp.down_proj.weight", "F32", (48, 96), zeros("F32", (48, 96))),
    ("model.layers.0.mlp.gate_proj.weight", "F32", (48, 96), zeros("F32", (48, 96))),
]
new = [
    ("model.embed_tokens.weight", "BF16", (128, 48), zeros("BF16", (128, 48))),  # dtype F32→BF16
    ("model.layers.0.mlp.down_proj.weight", "F32", (48, 128), zeros("F32", (48, 128))),  # shape change
    ("model.layers.0.mlp.gate_up_proj.weight", "F32", (48, 192), zeros("F32", (48, 192))),  # renamed/added
]
write_safetensors(f"{OUT}/old.safetensors", old, {"format": "pt", "cstorch_version": "1.0.0"})
write_safetensors(f"{OUT}/new.safetensors", new, {"format": "pt", "cstorch_version": "2.0.0"})

print(f"wrote demo checkpoints to {OUT}")
