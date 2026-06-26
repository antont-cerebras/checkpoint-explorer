//! Render the app's real terminal output to plain text.
//!
//! The draw code emits ANSI (cursor moves, clears, colour) for flicker-free
//! positioning. To get a stable, pipe-/test-friendly view we run that exact byte
//! stream through a tiny terminal emulator and read back the resulting character
//! grid — so `--plain` reflects what a terminal would actually show, rather than
//! a second rendering path that could drift from the interactive one.

use std::sync::atomic::{AtomicU32, Ordering};

/// A terminal size forced by `--plain` so headless rendering is deterministic
/// regardless of the ambient terminal (the binary's `terminal::size()` otherwise
/// reads the controlling tty, which varies by machine / CI). 0 means "unset".
static FORCED_SIZE: AtomicU32 = AtomicU32::new(0);

/// Force the virtual terminal size for subsequent [`term_size`] reads.
pub fn force_size(cols: u16, rows: u16) {
    FORCED_SIZE.store(((cols as u32) << 16) | rows as u32, Ordering::Relaxed);
}

/// The terminal size to render at: the `--plain` forced size when set, else the
/// real terminal (falling back to a sane default when there's no tty). Rendering
/// code should call this instead of `crossterm::terminal::size()` so `--plain`
/// can pin it.
pub fn term_size() -> (u16, u16) {
    match FORCED_SIZE.load(Ordering::Relaxed) {
        0 => crossterm::terminal::size().unwrap_or((100, 40)),
        v => ((v >> 16) as u16, (v & 0xFFFF) as u16),
    }
}

/// Emulate the ANSI `bytes` onto a `cols × rows` grid and return it as text:
/// one line per row, trailing spaces trimmed, trailing blank lines dropped.
/// Styling (SGR) is ignored; cursor moves, erases, and printable text are honoured.
pub fn render(bytes: &[u8], cols: usize, rows: usize) -> String {
    let mut grid = vec![vec![' '; cols]; rows];
    let (mut cr, mut cc) = (0usize, 0usize);
    // Deferred auto-wrap (like xterm with DECAWM): writing in the last column
    // leaves the cursor there with a pending wrap; the next printable char wraps
    // to the next line first. Any cursor move / CR / LF cancels the pending wrap.
    let mut pending_wrap = false;
    let text = String::from_utf8_lossy(bytes);
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        if ch == '\u{1b}' {
            // Only CSI (`ESC [ … final`) is interpreted; other escapes are skipped.
            if chars.get(i + 1) == Some(&'[') {
                let mut j = i + 2;
                let private = chars.get(j) == Some(&'?');
                let start = j;
                while j < chars.len() && !chars[j].is_ascii_alphabetic() {
                    j += 1;
                }
                if j >= chars.len() {
                    break;
                }
                let cmd = chars[j];
                let params: Vec<usize> = chars[start..j]
                    .iter()
                    .collect::<String>()
                    .trim_start_matches('?')
                    .split(';')
                    .filter_map(|p| p.parse::<usize>().ok())
                    .collect();
                if !private {
                    apply_csi(cmd, &params, &mut grid, &mut cr, &mut cc, cols, rows);
                    // A cursor move / erase cancels a pending wrap; SGR doesn't.
                    if cmd != 'm' {
                        pending_wrap = false;
                    }
                }
                i = j + 1;
                continue;
            }
            // Lone ESC or other escape (e.g. `ESC 7`): skip the next byte too.
            i += 2;
            continue;
        }
        match ch {
            '\r' => {
                cc = 0;
                pending_wrap = false;
            }
            '\n' => {
                cr = (cr + 1).min(rows.saturating_sub(1));
                pending_wrap = false;
            }
            '\t' => {
                cc = (((cc / 8) + 1) * 8).min(cols.saturating_sub(1));
                pending_wrap = false;
            }
            '\u{8}' => {
                cc = cc.saturating_sub(1);
                pending_wrap = false;
            }
            c if (c as u32) >= 0x20 => {
                if pending_wrap {
                    cc = 0;
                    cr = (cr + 1).min(rows.saturating_sub(1));
                    pending_wrap = false;
                }
                if cr < rows && cc < cols {
                    grid[cr][cc] = c;
                }
                if cc + 1 >= cols {
                    pending_wrap = true;
                } else {
                    cc += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    let mut lines: Vec<String> = grid
        .into_iter()
        .map(|row| row.into_iter().collect::<String>().trim_end().to_string())
        .collect();
    while lines.last().is_some_and(|l| l.is_empty()) {
        lines.pop();
    }
    lines.join("\n")
}

fn apply_csi(
    cmd: char,
    params: &[usize],
    grid: &mut [Vec<char>],
    cr: &mut usize,
    cc: &mut usize,
    cols: usize,
    rows: usize,
) {
    let p0 = params.first().copied();
    match cmd {
        // CUP / HVP: move to (row, col), 1-based.
        'H' | 'f' => {
            *cr = p0
                .unwrap_or(1)
                .saturating_sub(1)
                .min(rows.saturating_sub(1));
            *cc = params
                .get(1)
                .copied()
                .unwrap_or(1)
                .saturating_sub(1)
                .min(cols.saturating_sub(1));
        }
        'A' => *cr = cr.saturating_sub(p0.unwrap_or(1)),
        'B' => *cr = (*cr + p0.unwrap_or(1)).min(rows.saturating_sub(1)),
        'C' => *cc = (*cc + p0.unwrap_or(1)).min(cols.saturating_sub(1)),
        'D' => *cc = cc.saturating_sub(p0.unwrap_or(1)),
        // CHA: move to column, 1-based.
        'G' => {
            *cc = p0
                .unwrap_or(1)
                .saturating_sub(1)
                .min(cols.saturating_sub(1))
        }
        // ED: erase in display.
        'J' => match p0.unwrap_or(0) {
            2 | 3 => grid.iter_mut().for_each(|r| r.fill(' ')),
            1 => {
                grid[*cr][0..=(*cc).min(cols - 1)].fill(' ');
                grid.iter_mut().take(*cr).for_each(|r| r.fill(' '));
            }
            _ => {
                grid[*cr][(*cc).min(cols)..cols].fill(' ');
                grid.iter_mut()
                    .take(rows)
                    .skip(*cr + 1)
                    .for_each(|r| r.fill(' '));
            }
        },
        // EL: erase in line.
        'K' => match p0.unwrap_or(0) {
            2 => grid[*cr].fill(' '),
            1 => grid[*cr][0..=(*cc).min(cols - 1)].fill(' '),
            _ => grid[*cr][(*cc).min(cols)..cols].fill(' '),
        },
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::render;

    #[test]
    fn cursor_move_and_text() {
        // Write "B" at row 2 col 3 after a CUP, "A" at home.
        let s = render(b"A\x1b[3;4HB", 10, 5);
        assert_eq!(s, "A\n\n   B");
    }

    #[test]
    fn erase_to_end_of_line() {
        let s = render(b"hello\x1b[3G\x1b[K", 10, 1);
        assert_eq!(s, "he");
    }

    #[test]
    fn sgr_is_ignored() {
        let s = render(b"\x1b[31mred\x1b[0m", 10, 1);
        assert_eq!(s, "red");
    }
}
