//! Live progress bars for remote reads: one colored spinner + elapsed timer per
//! read, settling to `✓` (green) or `✗` (red). Animated on a background thread —
//! off the main thread doing the blocking SSH reads, touching only shared atomics,
//! so it never races the sessions — and suppressed when stderr isn't a terminal
//! (escape codes never pollute a pipe/log). Callers must do any password prompt
//! *before* starting the bars.

use std::io::{IsTerminal, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Shared, thread-safe progress for the checkpoint-structure load: a count of
/// completed units (shards / files) the reader bumps as it goes, and a total
/// that starts at 0 and is set once known (e.g. after a remote directory is
/// listed). The loading screen polls [`LoadProgress::snapshot`] to draw a bar
/// instead of a bare spinner, so a slow SSH read visibly makes progress.
#[derive(Default)]
pub struct LoadProgress {
    done: AtomicUsize,
    total: AtomicUsize,
}

impl LoadProgress {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record how many units the load comprises, once that's known.
    pub fn set_total(&self, total: usize) {
        self.total.store(total, Ordering::Relaxed);
    }

    /// Mark one more unit complete.
    pub fn advance(&self) {
        self.done.fetch_add(1, Ordering::Relaxed);
    }

    /// Set the absolute completed count (for a reader that reports totals rather
    /// than ticking — e.g. the remote cstorch dump's progress lines).
    pub fn set_done(&self, done: usize) {
        self.done.store(done, Ordering::Relaxed);
    }

    /// `(done, total)` for rendering; `total` is 0 until [`Self::set_total`].
    pub fn snapshot(&self) -> (usize, usize) {
        (
            self.done.load(Ordering::Relaxed),
            self.total.load(Ordering::Relaxed),
        )
    }
}

const RUNNING: u8 = 0;
const OK: u8 = 1;
const ERR: u8 = 2;

/// A set of progress bars, one per labelled read. Create with [`Bars::start`],
/// call [`Bars::finish`] as each read lands, and [`Bars::join`] once all are done.
pub struct Bars {
    states: Vec<Arc<AtomicU8>>,
    durations: Vec<Arc<AtomicU64>>,
    progress: Vec<Arc<LoadProgress>>,
    start: Instant,
    handle: Option<JoinHandle<()>>,
}

impl Bars {
    /// Reserve one bar per label and (on a terminal) start animating them.
    pub fn start(labels: Vec<String>) -> Bars {
        let n = labels.len();
        let states: Vec<_> = (0..n).map(|_| Arc::new(AtomicU8::new(RUNNING))).collect();
        let durations: Vec<_> = (0..n).map(|_| Arc::new(AtomicU64::new(0))).collect();
        let progress: Vec<_> = (0..n).map(|_| Arc::new(LoadProgress::new())).collect();
        let start = Instant::now();
        let handle = std::io::stderr().is_terminal().then(|| {
            spawn(
                labels,
                states.clone(),
                durations.clone(),
                progress.clone(),
                start,
            )
        });
        Bars {
            states,
            durations,
            progress,
            start,
            handle,
        }
    }

    /// The shared progress handle for read `i` — hand it to the reader so it can
    /// report shard/file completion, and the bar fills in as they land.
    pub fn progress(&self, i: usize) -> Option<Arc<LoadProgress>> {
        self.progress.get(i).cloned()
    }

    /// Mark read `i` finished — freezing its timer and showing `✓` (ok) or `✗`.
    pub fn finish(&self, i: usize, ok: bool) {
        if let Some(d) = self.durations.get(i) {
            d.store(self.start.elapsed().as_millis() as u64, Ordering::Relaxed);
        }
        if let Some(s) = self.states.get(i) {
            s.store(if ok { OK } else { ERR }, Ordering::Release);
        }
    }

    /// Wait for the animation thread to draw the final state and exit.
    pub fn join(mut self) {
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Shorten `s` to at most `max` columns by replacing the middle with `…`, keeping
/// both ends — so a URI/path keeps its scheme/host prefix and its tail.
fn truncate_middle(s: &str, max: usize) -> String {
    let n = s.chars().count();
    if n <= max {
        return s.to_string();
    }
    if max <= 1 {
        return "…".to_string();
    }
    let keep = max - 1; // room for the ellipsis
    let head = keep.div_ceil(2);
    let tail = keep - head;
    let h: String = s.chars().take(head).collect();
    let t: String = s.chars().skip(n - tail).collect();
    format!("{h}…{t}")
}

/// Width of the drawn `━━━━━━` progress bar, in columns.
const BAR_COLS: usize = 16;

/// How many of `width` bar columns are filled for `done`/`total` (rounded,
/// clamped to `width`; empty when `total` is 0).
fn filled_cols(done: usize, total: usize, width: usize) -> usize {
    if total == 0 {
        return 0;
    }
    (((done as f64 / total as f64) * width as f64).round() as usize).min(width)
}

/// Start column of the indeterminate bar's bright `win`-wide window at animation
/// `frame`, ping-ponging across a `width`-wide bar (so it shows an alive bar
/// while the total is still unknown — connecting / listing the directory).
fn sweep_pos(frame: usize, width: usize, win: usize) -> usize {
    let span = width.saturating_sub(win);
    if span == 0 {
        return 0;
    }
    let t = frame % (span * 2);
    if t <= span { t } else { span * 2 - t }
}

fn spawn(
    labels: Vec<String>,
    states: Vec<Arc<AtomicU8>>,
    durations: Vec<Arc<AtomicU64>>,
    progress: Vec<Arc<LoadProgress>>,
    start: Instant,
) -> JoinHandle<()> {
    // Fit labels to the terminal width so a line (mark + path + bar + timer) can't
    // wrap and break the fixed-height redraw. Truncate in the *middle* so both
    // ends — the `s3://`/`host:` prefix and the checkpoint tail — stay visible.
    let cols = crossterm::terminal::size()
        .map(|(c, _)| c as usize)
        .unwrap_or(80);
    // "  ⠋ <label>  [bar]  123/456  12.3s"
    let budget = cols.saturating_sub(BAR_COLS + 30).max(20);
    let labels: Vec<String> = labels.iter().map(|l| truncate_middle(l, budget)).collect();
    std::thread::spawn(move || {
        const FRAMES: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
        // Bold cyan spinner, bold green ✓, bold red ✗; dimmed labels so the
        // coloured mark and the timer stand out.
        const RUN: &str = "\x1b[1;36m";
        const DONE: &str = "\x1b[1;32m";
        const FAIL: &str = "\x1b[1;31m";
        const DIM: &str = "\x1b[2m";
        const RESET: &str = "\x1b[0m";
        let n = labels.len();
        let mut err = std::io::stderr();
        for _ in 0..n {
            let _ = writeln!(err); // reserve n lines; cursor ends just below them
        }
        let mut i = 0usize;
        loop {
            let now: Vec<u8> = states.iter().map(|s| s.load(Ordering::Acquire)).collect();
            let _ = write!(err, "\x1b[{n}A"); // back up to the first reserved line
            for (k, &st) in now.iter().enumerate() {
                let (color, mark) = match st {
                    OK => (DONE, '✓'),
                    ERR => (FAIL, '✗'),
                    _ => (RUN, FRAMES[i % FRAMES.len()]),
                };
                let ms = if st == RUNNING {
                    start.elapsed().as_millis() as u64
                } else {
                    durations[k].load(Ordering::Relaxed)
                };
                let secs = ms as f64 / 1000.0;
                // A `[███░░░] done/total` bar once the total is known (e.g. after a
                // remote dir is listed); until then just the spinner + timer.
                let (done, total) = progress[k].snapshot();
                let bar = if total > 0 {
                    // Determinate: a thin bar in the TUI `LineGauge` style
                    // (`symbols::line::THICK`) — done part in the mark's colour, the
                    // rest dim — plus the `done/total` count.
                    let filled = filled_cols(done, total, BAR_COLS);
                    format!(
                        "  {color}{}{RESET}{DIM}{}{RESET} {done}/{total}",
                        "━".repeat(filled),
                        "━".repeat(BAR_COLS - filled),
                    )
                } else if st == RUNNING {
                    // Total not known yet (still connecting / listing the dir) or an
                    // `s3://` read with no per-shard count: an indeterminate bar with
                    // a bright window sweeping across, so a live bar shows from the
                    // start instead of a bare spinner.
                    let win = 3.min(BAR_COLS);
                    let pos = sweep_pos(i, BAR_COLS, win);
                    format!(
                        "  {DIM}{}{RESET}{color}{}{RESET}{DIM}{}{RESET}",
                        "━".repeat(pos),
                        "━".repeat(win),
                        "━".repeat(BAR_COLS - pos - win),
                    )
                } else {
                    String::new() // finished with no known total: mark + timer only
                };
                let _ = write!(
                    err,
                    "\r\x1b[2K  {color}{mark}{RESET} {DIM}{}{RESET}{bar} {color}{secs:.1}s{RESET}\n",
                    labels[k]
                );
            }
            let _ = err.flush();
            if now.iter().all(|&st| st != RUNNING) {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
            i += 1;
        }
    })
}

#[cfg(test)]
mod tests {
    use super::{LoadProgress, filled_cols, sweep_pos, truncate_middle};

    #[test]
    fn load_progress_tracks_done_and_total() {
        let p = LoadProgress::new();
        assert_eq!(p.snapshot(), (0, 0)); // total unknown until set
        p.set_total(48);
        p.advance();
        p.advance();
        assert_eq!(p.snapshot(), (2, 48));
    }

    #[test]
    fn sweep_window_ping_pongs_within_bounds() {
        let (w, win) = (16usize, 3usize);
        let span = w - win; // 13
        assert_eq!(sweep_pos(0, w, win), 0); // starts at the left
        assert_eq!(sweep_pos(span, w, win), span); // reaches the right edge
        assert_eq!(sweep_pos(span + 1, w, win), span - 1); // then reverses
        // Never runs the window past the bar.
        for f in 0..100 {
            assert!(sweep_pos(f, w, win) + win <= w, "frame {f} overflows");
        }
    }

    #[test]
    fn bar_fill_is_proportional_and_clamped() {
        assert_eq!(filled_cols(0, 48, 16), 0);
        assert_eq!(filled_cols(24, 48, 16), 8); // half
        assert_eq!(filled_cols(48, 48, 16), 16); // full
        assert_eq!(filled_cols(47, 48, 16), 16); // rounds up, still clamped
        assert_eq!(filled_cols(5, 0, 16), 0); // no total → empty, no divide-by-zero
    }

    #[test]
    fn middle_truncation_keeps_both_ends() {
        // Short enough → untouched.
        assert_eq!(truncate_middle("s3://bucket/key", 100), "s3://bucket/key");
        // Ellipsis goes in the middle, both ends kept.
        assert_eq!(truncate_middle("abcdefghij", 5), "ab…ij");
        assert_eq!(truncate_middle("abcdefghij", 1), "…");
        // A long string is elided in the middle: the kept head and tail are a real
        // prefix and suffix of the input, and the result fits the budget.
        let s = "s3://inference-opensource/minimax-m2.5/4bit/260402";
        let t = truncate_middle(s, 24);
        assert!(t.chars().count() <= 24 && t.contains('…'), "{t}");
        let (head, tail) = t.split_once('…').unwrap();
        assert!(s.starts_with(head) && s.ends_with(tail), "{t}");
    }
}
