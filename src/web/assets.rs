//! The embedded Svelte SPA (`web/dist`). rust-embed reads from disk in debug and
//! embeds the bytes in release builds, so `cargo run -- --web` always serves the
//! latest `npm run build` while a released binary is self-contained.

use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "web/dist"]
pub struct WebAssets;

/// Content-type for a served asset path, by extension (covers the SPA's outputs).
pub fn content_type(path: &str) -> &'static str {
    match path.rsplit('.').next() {
        Some("html") => "text/html; charset=utf-8",
        Some("js") | Some("mjs") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("json") | Some("map") => "application/json; charset=utf-8",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("ico") => "image/x-icon",
        Some("woff2") => "font/woff2",
        Some("woff") => "font/woff",
        _ => "application/octet-stream",
    }
}
