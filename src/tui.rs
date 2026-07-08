//! Ratatui scaffolding shared by the interactive and headless render paths.
//!
//! This module owns the bits both the live loop and the `--plain` / screen-copy
//! paths need: the live [`Terminal`] lifecycle (deliberately *not* using the
//! alternate screen, so the last frame stays on exit) and an in-memory
//! [`TestBackend`] render for headless output.

use std::io::{self, Stdout};

use anyhow::Result;
use crossterm::{
    cursor,
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{self, ClearType},
};
use ratatui::{
    Frame, Terminal, TerminalOptions, Viewport,
    backend::{CrosstermBackend, TestBackend},
    buffer::Buffer,
};

/// The live terminal type owned by the interactive loop.
pub type LiveTerminal = Terminal<CrosstermBackend<Stdout>>;

/// Set up the live terminal: raw mode, a cleared screen, hidden cursor, and a
/// Ratatui terminal over stdout. Deliberately **no** alternate screen — quitting
/// leaves the last frame on screen (see [`restore`]).
pub fn init() -> Result<LiveTerminal> {
    terminal::enable_raw_mode()?;
    let mut out = io::stdout();
    // Capture the mouse so rows can be clicked and the wheel scrolls. (This means
    // the terminal's own text selection needs Shift held — the `y`/`c` shortcuts
    // and OSC-52 copy are the primary copy paths anyway.)
    // Clear the screen AND the scrollback, then home the cursor, before the
    // fullscreen viewport takes over. Any pre-TUI output (e.g. the `--ssh-read`
    // password prompt + read spinner) otherwise leaves lines the plain screen
    // clear (`\x1b[2J`) doesn't remove from the scrollback, so the tree appears
    // pushed down from the top.
    execute!(
        out,
        terminal::Clear(ClearType::All),
        terminal::Clear(ClearType::Purge),
        cursor::MoveTo(0, 0),
        cursor::Hide,
        EnableMouseCapture
    )?;
    let terminal = Terminal::with_options(
        CrosstermBackend::new(io::stdout()),
        TerminalOptions {
            viewport: Viewport::Fullscreen,
        },
    )?;
    Ok(terminal)
}

/// Restore the terminal after the interactive loop. Mirrors the previous
/// hand-rolled exit: leave the last rendered frame on screen, clear anything
/// below the cursor, show the cursor, leave raw mode, and drop the shell prompt
/// onto a fresh line just below the frame.
pub fn restore(terminal: &mut LiveTerminal) -> Result<()> {
    let height = terminal.size().map(|s| s.height).unwrap_or(0);
    let mut out = io::stdout();
    // Park the cursor at the bottom of the frame so the prompt lands below it.
    execute!(
        out,
        DisableMouseCapture,
        cursor::MoveTo(0, height.saturating_sub(1)),
        terminal::Clear(ClearType::FromCursorDown),
        cursor::Show
    )?;
    // Discard input still arriving before we hand the (cooked, echoing) terminal
    // back to the shell. Quitting mid-scroll (e.g. Ctrl-C during a mouse-wheel
    // burst) leaves a tail of unread SGR mouse reports in the buffer — plus, over
    // a laggy/remote link, more still in flight and any the terminal emits during
    // the round-trip before it processes `DisableMouseCapture` above. Left there,
    // they'd be echoed as `^[[<…M` garbage across the shell prompt. A single flush
    // only clears what's buffered *now*, so instead stay in raw mode (no echo) and
    // read+discard until the stream has been quiet for a short gap (outlasting the
    // disable's round-trip), capped so exit can't stall; then a final flush.
    #[cfg(unix)]
    // SAFETY: poll/read/tcflush operate only on our own stdin fd, reading into a
    // local buffer we own; errors (e.g. stdin isn't a tty) are handled by bailing.
    unsafe {
        use std::time::{Duration, Instant};
        let fd = libc::STDIN_FILENO;
        let mut pfd = libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        };
        // Always wait out a quiet window rather than checking "is something
        // queued right now" — at teardown the buffer is often momentarily empty
        // between arriving reports, and skipping the wait lets the next ones leak.
        let start = Instant::now();
        let mut buf = [0u8; 8192];
        loop {
            pfd.revents = 0;
            // A short gap with no input means the terminal has settled (the
            // disable took hold and the link drained); then we're done.
            if libc::poll(&mut pfd, 1, 60) <= 0 {
                break;
            }
            if libc::read(fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len()) <= 0 {
                break;
            }
            if start.elapsed() > Duration::from_millis(1200) {
                break; // cap: never stall exit, even under a relentless stream
            }
        }
        libc::tcflush(fd, libc::TCIFLUSH);
    }
    terminal::disable_raw_mode()?;
    println!();
    Ok(())
}

/// Render `f` once into an in-memory [`TestBackend`] of the given size and return
/// the resulting screen as plain text — the headless render path. Each row is the
/// buffer's cell symbols with trailing spaces trimmed, and trailing blank rows are
/// dropped, matching the shape the snapshot tests expect.
pub fn headless_render(width: u16, height: u16, f: impl FnOnce(&mut Frame)) -> Result<String> {
    let mut terminal = Terminal::new(TestBackend::new(width, height))?;
    terminal.draw(f)?;
    Ok(buffer_to_string(terminal.backend().buffer()))
}

/// Flatten a Ratatui [`Buffer`] to plain text: one line per row (cell symbols
/// concatenated; a wide glyph's trailing skip cell contributes nothing), trailing
/// spaces trimmed per row, trailing blank rows dropped.
pub fn buffer_to_string(buffer: &Buffer) -> String {
    use unicode_width::UnicodeWidthStr;
    let width = buffer.area.width as usize;
    let height = buffer.area.height as usize;
    let cells = buffer.content();
    let mut lines: Vec<String> = Vec::with_capacity(height);
    for row in 0..height {
        let mut line = String::new();
        // A wide glyph occupies several cells; emit its symbol once and skip the
        // continuation cells (same rule Ratatui's own buffer dump uses), so a
        // 2-cell emoji doesn't leak a stray space.
        let mut skip = 0usize;
        for col in 0..width {
            let symbol = cells[row * width + col].symbol();
            if skip == 0 {
                line.push_str(symbol);
            }
            skip = skip.max(symbol.width()).saturating_sub(1);
        }
        while line.ends_with(' ') {
            line.pop();
        }
        lines.push(line);
    }
    while lines.last().is_some_and(|l| l.is_empty()) {
        lines.pop();
    }
    lines.join("\n")
}
