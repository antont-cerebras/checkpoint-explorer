//! Ratatui scaffolding shared by the interactive and headless render paths.
//!
//! The app is migrating its hand-rolled crossterm renderer to Ratatui screen by
//! screen. This module owns the bits both the live loop and the `--plain` /
//! screen-copy paths need: the live [`Terminal`] lifecycle (deliberately *not*
//! using the alternate screen, so the last frame stays on exit), an in-memory
//! [`TestBackend`] render for headless output, and a crossterm→Ratatui color
//! shim so the existing `palette` constants keep working during the migration.

use std::io::{self, Stdout};

use anyhow::Result;
use crossterm::{
    cursor, execute,
    terminal::{self, ClearType},
};
use ratatui::{
    Frame, Terminal, TerminalOptions, Viewport,
    backend::{CrosstermBackend, TestBackend},
    buffer::Buffer,
};

/// The live terminal type owned by the interactive loop.
pub type LiveTerminal = Terminal<CrosstermBackend<Stdout>>;

/// Translate a crossterm [`Color`](crossterm::style::Color) — what the `palette`
/// constants use — into the equivalent Ratatui color, by ANSI index so the
/// on-screen color is unchanged. Lets the palette stay crossterm-typed while
/// screens migrate; the palette flips to native Ratatui colors once the last
/// raw screen is gone.
pub fn to_ratatui(color: crossterm::style::Color) -> ratatui::style::Color {
    use crossterm::style::Color as C;
    use ratatui::style::Color as R;
    match color {
        C::Reset => R::Reset,
        C::Black => R::Indexed(0),
        C::DarkRed => R::Indexed(1),
        C::DarkGreen => R::Indexed(2),
        C::DarkYellow => R::Indexed(3),
        C::DarkBlue => R::Indexed(4),
        C::DarkMagenta => R::Indexed(5),
        C::DarkCyan => R::Indexed(6),
        C::Grey => R::Indexed(7),
        C::DarkGrey => R::Indexed(8),
        C::Red => R::Indexed(9),
        C::Green => R::Indexed(10),
        C::Yellow => R::Indexed(11),
        C::Blue => R::Indexed(12),
        C::Magenta => R::Indexed(13),
        C::Cyan => R::Indexed(14),
        C::White => R::Indexed(15),
        C::AnsiValue(n) => R::Indexed(n),
        C::Rgb { r, g, b } => R::Rgb(r, g, b),
    }
}

/// Set up the live terminal: raw mode, a cleared screen, hidden cursor, and a
/// Ratatui terminal over stdout. Deliberately **no** alternate screen — quitting
/// leaves the last frame on screen (see [`restore`]).
pub fn init() -> Result<LiveTerminal> {
    terminal::enable_raw_mode()?;
    let mut out = io::stdout();
    execute!(out, terminal::Clear(ClearType::All), cursor::Hide)?;
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
        cursor::MoveTo(0, height.saturating_sub(1)),
        terminal::Clear(ClearType::FromCursorDown),
        cursor::Show
    )?;
    terminal::disable_raw_mode()?;
    println!();
    Ok(())
}

/// Render `f` once into an in-memory [`TestBackend`] of the given size and return
/// the resulting screen as plain text — the headless replacement for the old
/// ANSI-emulator (`plain::render`). Each row is the buffer's cell symbols with
/// trailing spaces trimmed, and trailing blank rows are dropped, matching the
/// shape the snapshot tests expect.
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
