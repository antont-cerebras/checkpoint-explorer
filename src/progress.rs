//! Live progress bars for remote reads: one colored spinner + elapsed timer per
//! read, settling to `✓` (green) or `✗` (red). Animated on a background thread —
//! off the main thread doing the blocking SSH reads, touching only shared atomics,
//! so it never races the sessions — and suppressed when stderr isn't a terminal
//! (escape codes never pollute a pipe/log). Callers must do any password prompt
//! *before* starting the bars.

use std::io::{IsTerminal, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const RUNNING: u8 = 0;
const OK: u8 = 1;
const ERR: u8 = 2;

/// A set of progress bars, one per labelled read. Create with [`Bars::start`],
/// call [`Bars::finish`] as each read lands, and [`Bars::join`] once all are done.
pub struct Bars {
    states: Vec<Arc<AtomicU8>>,
    durations: Vec<Arc<AtomicU64>>,
    start: Instant,
    handle: Option<JoinHandle<()>>,
}

impl Bars {
    /// Reserve one bar per label and (on a terminal) start animating them.
    pub fn start(labels: Vec<String>) -> Bars {
        let n = labels.len();
        let states: Vec<_> = (0..n).map(|_| Arc::new(AtomicU8::new(RUNNING))).collect();
        let durations: Vec<_> = (0..n).map(|_| Arc::new(AtomicU64::new(0))).collect();
        let start = Instant::now();
        let handle = std::io::stderr()
            .is_terminal()
            .then(|| spawn(labels, states.clone(), durations.clone(), start));
        Bars {
            states,
            durations,
            start,
            handle,
        }
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

fn spawn(
    labels: Vec<String>,
    states: Vec<Arc<AtomicU8>>,
    durations: Vec<Arc<AtomicU64>>,
    start: Instant,
) -> JoinHandle<()> {
    // Truncate labels so a line (mark + path + timer) can't wrap and break the
    // fixed-height redraw.
    let labels: Vec<String> = labels.iter().map(|l| crate::truncate_tail(l, 60)).collect();
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
                let _ = write!(
                    err,
                    "\r\x1b[2K  {color}{mark}{RESET} {DIM}{}{RESET} {color}{secs:.1}s{RESET}\n",
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
