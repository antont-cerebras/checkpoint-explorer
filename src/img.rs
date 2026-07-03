//! Render a sampled tensor grid as a **heatmap PNG**, and — for iTerm2 and
//! compatible terminals — an inline-image escape so the heatmap shows as a real
//! picture in the terminal (the protocol `imgcat` uses). Same blue→green→red map
//! as the ASCII heatmap, but at pixel resolution and smooth 24-bit colour.
//!
//! Backs the interactive `i` (show inline) / `p` (save) keys and the headless
//! `--image` / `--png` flags.

use crate::utils::base64_encode;

/// Continuous blue→green→red colormap for a normalized value in `[0, 1]` — the
/// smooth 24-bit version of the ASCII heatmap's ANSI ramp (cool = low, warm = high).
fn heat_rgb(t: f64) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    let r = t;
    let b = 1.0 - t;
    let g = (1.0 - (t - 0.5).abs() * 2.0).max(0.0);
    [
        (r * 255.0).round() as u8,
        (g * 255.0).round() as u8,
        (b * 255.0).round() as u8,
    ]
}

/// Upscale factor so the image's longer side is roughly `target_px`: each grid
/// cell becomes an N×N block of pixels (nearest-neighbour), clamped so tiny grids
/// don't explode and large ones stay 1:1. `target_px` is the caller's requested
/// image size (`--image-size` / the `+`/`-` keys); larger reads better on a
/// HiDPI/Retina display but makes a bigger payload.
fn scale_for(rows: usize, cols: usize, target_px: usize) -> usize {
    (target_px / rows.max(cols).max(1)).clamp(1, 64)
}

/// Render `values` (a sampled `rows × cols` grid) as an RGB PNG: each cell is
/// normalized over `[rmin, rmax]` through the heat colormap and upscaled to a
/// block, so the image keeps the sampled grid's orientation and aspect.
/// `target_px` sizes the image's longer side. With `legend`, a self-contained
/// colour-scale bar (blue→green→red, labelled `rmin`/`rmax`) is drawn on top —
/// used for the saved/`--png` file; the inline `i` viewer passes `false` and shows
/// the legend on its own line instead. Empty when the grid is empty.
pub fn heatmap_png(
    values: &[Vec<f64>],
    rmin: f64,
    rmax: f64,
    target_px: usize,
    legend: bool,
) -> Vec<u8> {
    let rows = values.len();
    let cols = values.first().map_or(0, Vec::len);
    if rows == 0 || cols == 0 {
        return Vec::new();
    }
    let scale = scale_for(rows, cols, target_px);
    let (hw, hh) = (cols * scale, rows * scale);

    // With a legend, reserve a band on top (sized off the image so it reads at any
    // resolution) and widen the canvas so the two labels don't collide. Without
    // one, the canvas is just the grid.
    let fs = (hw.max(hh) / 200).clamp(2, 8); // label pixel scale
    let (pad, bar_h, gap, text_h) = (4 * fs, 10 * fs, 3 * fs, 7 * fs);
    let (min_s, max_s) = (fmt_val(rmin), fmt_val(rmax));
    let (band_h, w) = if legend {
        (
            pad + bar_h + gap + text_h + pad,
            hw.max(text_w(&min_s, fs) + gap * 2 + text_w(&max_s, fs) + pad * 2),
        )
    } else {
        (0, hw)
    };
    let h = band_h + hh;
    let x0 = (w - hw) / 2; // heatmap centered (a no-op when w == hw)

    let mut px = vec![255u8; w * h * 3]; // white canvas (margins + band bg)

    if legend {
        // Gradient bar spanning the band, then the min/max labels beneath it.
        let (bar_x0, bar_x1) = (pad, w - pad);
        let bar_w = bar_x1.saturating_sub(bar_x0).max(1);
        for by in 0..bar_h {
            for bx in 0..bar_w {
                let rgb = heat_rgb(bx as f64 / (bar_w - 1).max(1) as f64);
                let i = ((pad + by) * w + bar_x0 + bx) * 3;
                px[i..i + 3].copy_from_slice(&rgb);
            }
        }
        let ty = pad + bar_h + gap;
        draw_text(&mut px, w, pad, ty, &min_s, fs, [0, 0, 0]);
        let mx = (w - pad).saturating_sub(text_w(&max_s, fs));
        draw_text(&mut px, w, mx, ty, &max_s, fs, [0, 0, 0]);
    }

    // Heatmap cells, upscaled, centered under the band.
    let range = rmax - rmin;
    for (ri, row) in values.iter().enumerate() {
        for cj in 0..cols {
            let v = row.get(cj).copied().unwrap_or(rmin);
            let t = if range > 0.0 { (v - rmin) / range } else { 0.5 };
            let rgb = heat_rgb(t);
            for dy in 0..scale {
                let y = band_h + ri * scale + dy;
                let start = (y * w + x0 + cj * scale) * 3;
                for dx in 0..scale {
                    px[start + dx * 3..start + dx * 3 + 3].copy_from_slice(&rgb);
                }
            }
        }
    }

    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, w as u32, h as u32);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        if let Ok(mut writer) = enc.write_header() {
            let _ = writer.write_image_data(&px);
        }
    }
    out
}

/// A one-line ANSI colour-scale legend for the inline viewer — the min value, a
/// truecolor blue→green→red gradient bar, then the max value. Kept out of the
/// image itself so the `i` view doesn't glue the legend onto the picture.
pub fn ansi_legend(rmin: f64, rmax: f64) -> String {
    let mut s = format!("{}  ", fmt_val(rmin));
    for i in 0..24 {
        let [r, g, b] = heat_rgb(i as f64 / 23.0);
        s.push_str(&format!("\x1b[38;2;{r};{g};{b}m\u{2588}"));
    }
    s.push_str("\x1b[0m  ");
    s.push_str(&fmt_val(rmax));
    s
}

/// Format a colour-scale bound compactly, using only characters [`glyph`] can
/// draw: fixed notation (trimmed) in the human range, else scientific (`e`).
fn fmt_val(v: f64) -> String {
    if v == 0.0 {
        return "0".into();
    }
    let a = v.abs();
    if (1e-3..1e5).contains(&a) {
        let s = format!("{v:.3}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        format!("{v:.2e}")
    }
}

/// Pixel width of `text` drawn at `scale` (6·scale per char: 5 wide + 1 spacing).
fn text_w(text: &str, scale: usize) -> usize {
    text.chars().count() * 6 * scale
}

/// Draw `text` into the RGB buffer of width `w` at `(x0, y0)`, each glyph pixel a
/// `scale`×`scale` block in `color`. Unknown chars advance but draw blank.
fn draw_text(
    px: &mut [u8],
    w: usize,
    x0: usize,
    y0: usize,
    text: &str,
    scale: usize,
    color: [u8; 3],
) {
    let mut cx = x0;
    for ch in text.chars() {
        if let Some(g) = glyph(ch) {
            for (ry, bits) in g.iter().enumerate() {
                for col in 0..5 {
                    if bits & (0x10 >> col) == 0 {
                        continue;
                    }
                    for dy in 0..scale {
                        for dx in 0..scale {
                            let (x, y) = (cx + col * scale + dx, y0 + ry * scale + dy);
                            let i = (y * w + x) * 3;
                            if x < w && i + 3 <= px.len() {
                                px[i..i + 3].copy_from_slice(&color);
                            }
                        }
                    }
                }
            }
        }
        cx += 6 * scale;
    }
}

/// Minimal 5×7 bitmap glyphs for the legend labels — only the characters
/// [`fmt_val`] emits (digits, `.`, `-`, `e`). Each row's low 5 bits are columns
/// (bit 4 = leftmost). `None` for anything else (drawn as a blank advance).
fn glyph(c: char) -> Option<[u8; 7]> {
    let g = match c {
        '0' => [
            0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
        ],
        '1' => [
            0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
        '2' => [
            0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111,
        ],
        '3' => [
            0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
        '4' => [
            0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
        ],
        '5' => [
            0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110,
        ],
        '6' => [
            0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
        ],
        '7' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
        ],
        '8' => [
            0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
        ],
        '9' => [
            0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100,
        ],
        '.' => [0, 0, 0, 0, 0, 0b00110, 0b00110],
        '-' => [0, 0, 0, 0b11111, 0, 0, 0],
        '+' => [0, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0],
        'e' => [0, 0, 0b01110, 0b10001, 0b11111, 0b10000, 0b01110],
        ' ' => [0, 0, 0, 0, 0, 0, 0],
        _ => return None,
    };
    Some(g)
}

/// Emit PNG bytes as an iTerm2 inline image (the protocol `imgcat` uses). Outside
/// tmux it's one `ESC ] 1337 ; File = … : <base64> BEL` escape; inside tmux the
/// base64 is split across a **multipart** transfer (`MultipartFile` / `FilePart` /
/// `FileEnd`), each part its own tmux-passthrough-wrapped escape — because a
/// single giant sequence overruns tmux's passthrough buffer and spills the tail
/// as literal base64 on screen. `name` (shown by iTerm2) is optional. `fit`, when
/// set to `(cols, rows)` terminal cells, sizes the image to fill that box while
/// preserving aspect ratio (the "as big as fits" display), so the picture tracks
/// the terminal instead of a fixed pixel size.
pub fn iterm2_inline(png: &[u8], name: Option<&str>, fit: Option<(usize, usize)>) -> String {
    if std::env::var_os("TMUX").is_some() {
        iterm2_multipart(png, name, fit)
    } else {
        iterm2_escape(png, name, fit)
    }
}

/// iTerm2 File arguments shared by the single-shot and multipart forms. `fit` adds
/// `width`/`height` in terminal cells (with `preserveAspectRatio`, iTerm2 scales to
/// fill that box keeping aspect).
fn file_args(len: usize, name: Option<&str>, fit: Option<(usize, usize)>) -> String {
    let mut args = format!("inline=1;preserveAspectRatio=1;size={len}");
    if let Some((cols, rows)) = fit {
        args = format!("{args};width={cols};height={rows}");
    }
    match name {
        Some(n) => format!("name={};{args}", base64_encode(n.as_bytes())),
        None => args,
    }
}

/// The bare single-shot iTerm2 `1337;File=` escape (no tmux wrapping) — split out
/// so it can be tested independent of the ambient `TMUX` environment.
fn iterm2_escape(png: &[u8], name: Option<&str>, fit: Option<(usize, usize)>) -> String {
    format!(
        "\x1b]1337;File={}:{}\x07",
        file_args(png.len(), name, fit),
        base64_encode(png)
    )
}

/// iTerm2's chunked (multipart) inline-image transfer, each OSC wrapped in a tmux
/// passthrough DCS. The base64 is ASCII, so slicing on byte boundaries is safe.
fn iterm2_multipart(png: &[u8], name: Option<&str>, fit: Option<(usize, usize)>) -> String {
    const CHUNK: usize = 2048; // base64 chars per part — well under tmux's limit
    let b64 = base64_encode(png);
    let wrap = |seq: &str| passthrough(seq, true);
    let mut out = wrap(&format!(
        "\x1b]1337;MultipartFile={}\x07",
        file_args(png.len(), name, fit)
    ));
    let bytes = b64.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let end = (i + CHUNK).min(bytes.len());
        out.push_str(&wrap(&format!("\x1b]1337;FilePart={}\x07", &b64[i..end])));
        i = end;
    }
    out.push_str(&wrap("\x1b]1337;FileEnd\x07"));
    out
}

/// Inside tmux, wrap a terminal-proprietary escape in tmux's passthrough DCS so
/// tmux forwards it verbatim to the outer terminal (iTerm2) instead of swallowing
/// it — every embedded `ESC` is doubled, per the protocol. Requires tmux
/// `allow-passthrough on` (the default since tmux 3.3). No-op when not in tmux.
fn passthrough(seq: &str, in_tmux: bool) -> String {
    if in_tmux {
        format!("\x1bPtmux;{}\x1b\\", seq.replace('\x1b', "\x1b\x1b"))
    } else {
        seq.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heat_rgb_spans_blue_to_red() {
        assert_eq!(heat_rgb(0.0), [0, 0, 255]); // low → blue
        assert_eq!(heat_rgb(1.0), [255, 0, 0]); // high → red
        let mid = heat_rgb(0.5);
        assert_eq!(mid[1], 255); // mid → green peak
        assert!(mid[0] < 255 && mid[2] < 255);
    }

    #[test]
    fn png_has_signature_and_scales_up() {
        // 2×3 grid → upscaled; valid 8-bit RGB PNG, taller than the bare grid
        // (the legend band sits on top) and at least as wide.
        let values = vec![vec![0.0, 0.5, 1.0], vec![1.0, 0.5, 0.0]];
        let png = heatmap_png(&values, 0.0, 1.0, 1600, true);
        assert_eq!(&png[..8], &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
        let decoder = png::Decoder::new(&png[..]);
        let reader = decoder.read_info().unwrap();
        let info = reader.info();
        let scale = scale_for(2, 3, 1600) as u32;
        assert!(info.width >= 3 * scale); // ≥ grid width (may widen for labels)
        assert!(info.height > 2 * scale); // grid + legend band
    }

    #[test]
    fn no_legend_is_exactly_the_grid() {
        // Without the legend the canvas is just the upscaled grid — no band, no
        // widening for labels.
        let values = vec![vec![0.0, 0.5, 1.0], vec![1.0, 0.5, 0.0]];
        let png = heatmap_png(&values, 0.0, 1.0, 1600, false);
        let r = png::Decoder::new(&png[..]).read_info().unwrap();
        let scale = scale_for(2, 3, 1600) as u32;
        assert_eq!((r.info().width, r.info().height), (3 * scale, 2 * scale));
    }

    #[test]
    fn target_px_controls_size() {
        // A grid big enough that the target divides below the per-cell clamp, so
        // the upscale factor (hence image size) actually tracks `target_px`.
        let values = vec![vec![0.5; 40]; 40];
        let dims = |target| {
            let png = heatmap_png(&values, 0.0, 1.0, target, true);
            let r = png::Decoder::new(&png[..]).read_info().unwrap();
            (r.info().width, r.info().height)
        };
        let (w_small, h_small) = dims(400);
        let (w_big, h_big) = dims(1600);
        assert!(w_big > w_small && h_big > h_small); // larger target → larger image
    }

    #[test]
    fn empty_grid_is_empty_png() {
        assert!(heatmap_png(&[], 0.0, 1.0, 1600, true).is_empty());
        assert!(heatmap_png(&[vec![]], 0.0, 1.0, 1600, true).is_empty());
    }

    #[test]
    fn ansi_legend_has_gradient_and_bounds() {
        let s = ansi_legend(-0.5, 1.25);
        assert!(s.contains("-0.5") && s.contains("1.25"));
        assert!(s.contains("\u{2588}") && s.contains("\x1b[38;2;")); // colored blocks
        assert!(s.ends_with("1.25")); // max on the right
    }

    #[test]
    fn fmt_val_stays_in_glyph_set() {
        // Every character we format must be drawable, or the legend shows gaps.
        for v in [0.0, 1.0, -0.5, 0.0123, -123.456, 1e-9, -4.2e12, 98765.0] {
            let s = fmt_val(v);
            assert!(
                s.chars().all(|c| glyph(c).is_some()),
                "fmt_val({v}) = {s:?} has an undrawable char"
            );
        }
    }

    #[test]
    fn iterm2_escape_shape() {
        let esc = iterm2_escape(b"\x89PNG", None, None);
        assert!(esc.starts_with("\x1b]1337;File=inline=1;"));
        assert!(esc.ends_with('\x07'));
        assert!(esc.contains(";size=4:")); // 4 bytes, then base64 payload
        assert!(!esc.contains("width=")); // no fit box requested
    }

    #[test]
    fn iterm2_escape_fit_adds_cell_box() {
        // A fit box adds width/height in cells so iTerm2 scales to fill it.
        let esc = iterm2_escape(b"\x89PNG", None, Some((80, 20)));
        assert!(esc.contains("width=80;height=20"));
        assert!(esc.contains("preserveAspectRatio=1"));
    }

    #[test]
    fn tmux_passthrough_doubles_esc_and_wraps() {
        let raw = "\x1b]1337;File=x\x07";
        // Outside tmux: untouched.
        assert_eq!(passthrough(raw, false), raw);
        // Inside tmux: wrapped in the DCS, every ESC doubled.
        let w = passthrough(raw, true);
        assert!(w.starts_with("\x1bPtmux;"));
        assert!(w.ends_with("\x1b\\"));
        assert!(w.contains("\x1b\x1b]1337;File=x\x07")); // inner ESC doubled
    }

    #[test]
    fn multipart_splits_into_passthrough_wrapped_parts() {
        // A payload larger than one chunk → announce + ≥2 parts + end, each its
        // own passthrough DCS (so no single sequence overruns tmux).
        let png = vec![0u8; 4096]; // ~5464 base64 chars > 2 * CHUNK
        let seq = iterm2_multipart(&png, Some("t"), None);
        assert_eq!(seq.matches("MultipartFile=").count(), 1);
        assert!(seq.matches("FilePart=").count() >= 2);
        assert_eq!(seq.matches("FileEnd").count(), 1);
        // Every emitted OSC is wrapped, so parts stay individually small.
        assert_eq!(
            seq.matches("\x1bPtmux;").count(),
            seq.matches("\x1b\\").count()
        );
    }
}
