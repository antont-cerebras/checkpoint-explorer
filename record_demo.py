#!/usr/bin/env python3
"""Record demo.gif without vhs — pure stdlib PTY driver + asciinema `agg`.

`demo.tape` (vhs) is the recommended, portable way to regenerate the GIF. This
script is the no-vhs fallback that actually produced the committed demo.gif: it
drives a real bash PTY at a fixed 100x30, writes an asciicast, and you render it
with agg. It only needs python3 and agg (no vhs, no browser, no ffmpeg).

  python3 gen_demo.py                       # build /tmp/ckpt-demo/*.safetensors
  python3 record_demo.py                     # drive the TUI -> /tmp/demo.cast
  agg --theme dracula --font-size 14 /tmp/demo.cast demo.gif

Requires `checkpoint-explorer` on PATH (`cargo install --path . --features hdf5`)
and `agg` (`cargo install --git https://github.com/asciinema/agg`).

The tour mirrors demo.tape: browse the grouped tree + fuzzy search, a tensor's
stats/histogram + heatmap + numeric grid, an on-the-fly 4-bit (u4) decode, and a
coloured structural `diff`. Tweak the feed()/send() timings to taste.
"""

import fcntl
import json
import os
import pty
import select
import struct
import tempfile
import termios
import time

COLS, ROWS = 100, 30

# A minimal rcfile for a clean `$ ` prompt (bash inherits it via --rcfile). We
# also start bash with --noediting so readline doesn't emit its bracketed-paste
# init (which prints a stray char before the first prompt).
rc = tempfile.NamedTemporaryFile("w", suffix=".bashrc", delete=False)
rc.write("PS1='$ '\nHISTFILE=/dev/null\nunset PROMPT_COMMAND\n")
rc.close()

env = dict(os.environ, TERM="xterm-256color", COLUMNS=str(COLS), LINES=str(ROWS),
           PS1="$ ", HISTFILE="/dev/null")
pid, fd = pty.fork()
if pid == 0:
    os.execvpe("bash", ["bash", "--rcfile", rc.name, "--noprofile", "--noediting", "-i"], env)
fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))

events, start = [], time.time()


def feed(dur):
    end = time.time() + dur
    while True:
        rem = end - time.time()
        if rem <= 0:
            return True
        r, _, _ = select.select([fd], [], [], rem)
        if fd in r:
            try:
                data = os.read(fd, 65536)
            except OSError:
                return False
            if not data:
                return False
            events.append([round(time.time() - start, 3), "o", data.decode("utf-8", "replace")])


def send(s):
    os.write(fd, s.encode())


def cmd(s):
    send(s)
    feed(0.25)
    send("\r")  # show the command, then run it


ENTER, DOWN, ESC, BS, CTRLC = "\r", "\x1b[B", "\x1b", "\x7f", "\x03"

feed(0.4); send("clear\r"); feed(0.5)

# 1) browse the grouped tree + fuzzy search
cmd("checkpoint-explorer /tmp/ckpt-demo/model.safetensors --tree-state expanded"); feed(2.6)
send(DOWN * 8); feed(1.6)
send("/"); feed(0.4); send("down_proj"); feed(2.0); send(ESC); feed(1.0); send(CTRLC); feed(1.0)

# 2) a tensor's detail: stats + histogram, heatmap, numeric grid
cmd("checkpoint-explorer /tmp/ckpt-demo/model.safetensors --tensor model.layers.0.mlp.down_proj.weight --compute-stats"); feed(2.6)
send("h"); feed(2.2); send("m"); feed(2.6); send(BS); feed(0.7); send("v"); feed(2.6); send(CTRLC); feed(1.0)

# 3) decode a packed 4-bit weight (U8 -> u4)
cmd("checkpoint-explorer /tmp/ckpt-demo/model.safetensors --tensor model.layers.0.mlp.gate_proj.qweight --dtype u4 --values"); feed(3.0); send(CTRLC); feed(1.0)

# 4) coloured structural diff  (clear first — diff prints to the normal screen)
send("clear\r"); feed(0.5)
cmd("checkpoint-explorer diff /tmp/ckpt-demo/old.safetensors /tmp/ckpt-demo/new.safetensors"); feed(4.0)

send("exit\r"); feed(1.0)
try:
    os.close(fd)
except OSError:
    pass
os.waitpid(pid, 0)
os.unlink(rc.name)

header = {"version": 2, "width": COLS, "height": ROWS,
          "env": {"TERM": "xterm-256color", "SHELL": "/bin/bash"}}
with open("/tmp/demo.cast", "w") as f:
    f.write(json.dumps(header) + "\n")
    for ev in events:
        f.write(json.dumps(ev) + "\n")
print(f"cast: {len(events)} events, {events[-1][0] if events else 0:.1f}s, "
      f"{os.path.getsize('/tmp/demo.cast')} bytes")
