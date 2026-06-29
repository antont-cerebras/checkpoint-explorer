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
    execute!(
        out,
        terminal::Clear(ClearType::All),
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
