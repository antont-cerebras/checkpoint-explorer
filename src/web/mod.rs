//! `--web`: a headless HTTP server (sync/blocking, no async runtime) that serves
//! the checkpoint as JSON — the **data** — plus the embedded Svelte UI, which owns
//! its own **view state**. Local checkpoints only for now.
//!
//! `WebState` is read once at startup and shared read-only across worker threads
//! (`Arc`); every derived view/report is precomputed so request handling needs no
//! `&mut` (only the on-demand tensor-data scans touch disk, behind a small cache).

mod assets;
pub mod dto;
pub mod handlers;

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::Result;

use crate::{check, filetree, filter, health, kernel, model, safelayout, sample, stats, tree};
use handlers::{Query, Reply};

/// Everything the API serves, computed once from a local read. Shared read-only.
pub struct WebState {
    pub root: String,
    /// The full serializable model (backs `/api/model`).
    pub checkpoint: model::Checkpoint,
    /// The tensor-tree hierarchy (client folds/selects/searches it).
    pub tree: Vec<tree::TreeNode>,
    pub file_tree: dto::WebFileNode,
    pub stats: stats::CheckpointStats,
    pub health: Vec<health::HealthReport>,
    pub check: Option<check::CheckReport>,
    pub layouts: Vec<safelayout::LayoutMap>,
    /// Canonical (deduped, natural-sorted) tensors, for detail + data-view lookup.
    pub tensors: Vec<tree::TensorInfo>,
    tensor_index: HashMap<String, usize>,
    schemas: HashMap<String, sample::PackingSchema>,
    /// Per-`(tensor, view)` whole-tensor stats, memoized (also feeds histogram range).
    stats_cache: Mutex<HashMap<(String, String), dto::StatsDto>>,
}

impl WebState {
    /// Build the shared state from a local checkpoint read. `files`/`index_specs`
    /// are what `run_explore` already resolved (for the structural check + health).
    pub fn build(
        checkpoint: model::Checkpoint,
        files: &[PathBuf],
        index_specs: &[health::IndexSpec],
    ) -> Self {
        let root = checkpoint.root.clone();
        let disk = checkpoint.disk_usage();

        // Canonicalize through a Session (dedup + natural sort) so the tree and
        // tensor list match the TUI exactly. Clone the model so we keep it for
        // `/api/model`; `stats_with_disk` needs `&mut`, so call it last.
        let mut session = kernel::Session::from_model(checkpoint.clone());
        let tensors: Vec<tree::TensorInfo> = session.tensors().to_vec();
        let metadata: Vec<tree::MetadataInfo> = session.metadata().to_vec();
        let config = session.config().cloned();
        let tree = session.build_tree();
        let checkpoint_stats = session.stats_with_disk(disk).clone();

        let tensor_index = tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.clone(), i))
            .collect();
        let schemas = sample::parse_packing_schemas(&tensors, &metadata);

        let file_tree =
            dto::WebFileNode::from_node(&filetree::build(Path::new(&root), 8), Path::new(&root));

        let health: Vec<health::HealthReport> = index_specs
            .iter()
            .map(|spec| health::check_loaded(spec, &tensors))
            .collect();

        // Structural check only (values = false → no byte scan at startup).
        let all = filter::NameFilter::parse(&[]).expect("empty NameFilter is valid");
        let check = Some(check::run(
            root.clone(),
            &tensors,
            &metadata,
            files,
            &health,
            config.as_ref(),
            &all,
            false,
            1,
        ));

        let layouts = checkpoint
            .shards
            .iter()
            .map(|sh| {
                safelayout::from_tensors(
                    &sh.path,
                    sh.total_len,
                    sh.header_len,
                    &sh.tensors,
                    &sh.metadata,
                )
            })
            .collect();

        WebState {
            root,
            checkpoint,
            tree,
            file_tree,
            stats: checkpoint_stats,
            health,
            check,
            layouts,
            tensors,
            tensor_index,
            schemas,
            stats_cache: Mutex::new(HashMap::new()),
        }
    }
}

/// Start the server and block until the process is stopped (Ctrl-C). `host` is the
/// bind address (default `0.0.0.0` — all interfaces, so it's reachable at this
/// machine's hostname on the network, matching how VMs serve web apps here).
pub fn serve(state: Arc<WebState>, host: IpAddr, port: u16) -> Result<()> {
    let server = tiny_http::Server::http(SocketAddr::new(host, port))
        .map_err(|e| anyhow::anyhow!("failed to start web server on {host}:{port}: {e}"))?;
    let bound = server
        .server_addr()
        .to_ip()
        .map(|a| a.port())
        .unwrap_or(port);
    // Print a URL the browser can actually reach: a wildcard bind (0.0.0.0 / ::)
    // isn't clickable, so show this host's FQDN instead of the bind address.
    let display = if host.is_unspecified() {
        fqdn().unwrap_or_else(|| "localhost".to_string())
    } else {
        host.to_string()
    };
    let url = format!("http://{display}:{bound}/");
    println!("checkpoint-explorer web UI: {url}  (Ctrl-C to stop)");

    // A small worker pool so a static-asset / metadata request stays responsive
    // while another worker is inside a multi-second tensor scan.
    let server = Arc::new(server);
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(2, 8);
    let mut handles = Vec::with_capacity(workers);
    for _ in 0..workers {
        let server = Arc::clone(&server);
        let state = Arc::clone(&state);
        handles.push(std::thread::spawn(move || {
            while let Ok(req) = server.recv() {
                handle(&state, req);
            }
        }));
    }
    for h in handles {
        let _ = h.join();
    }
    Ok(())
}

fn handle(state: &WebState, req: tiny_http::Request) {
    let url = req.url().to_string();
    let (path, query_str) = url.split_once('?').unwrap_or((url.as_str(), ""));
    let gzip = accepts_gzip(&req);
    if let Some(api) = path.strip_prefix("/api/") {
        let q = parse_query(query_str);
        let (status, body) = route_api(state, api, &q);
        let data = serde_json::to_vec(&body).unwrap_or_default();
        send(req, status, data, "application/json; charset=utf-8", gzip);
    } else {
        respond_asset(req, path, gzip);
    }
}

fn route_api(s: &WebState, path: &str, q: &Query) -> Reply {
    match path {
        "tree" => handlers::tree(s),
        "files" => handlers::files(s),
        "stats" => handlers::stats(s),
        "health" => handlers::health(s),
        "check" => handlers::check(s),
        "model" => handlers::model(s),
        "tensor" => handlers::tensor(s, q),
        "layout" => handlers::layout(s, q),
        "file" => handlers::file(s, q),
        "tensor/stats" => handlers::tensor_stats(s, q),
        "tensor/sample" => handlers::tensor_sample(s, q),
        "tensor/histogram" => handlers::tensor_histogram(s, q),
        other => handlers::err(404, format!("no such endpoint: /api/{other}")),
    }
}

/// Parse `k=v&k=v` into a map, percent-decoding each **value** (tensor names carry
/// `/` and `.`, which the client sends `encodeURIComponent`-ed).
fn parse_query(qs: &str) -> Query {
    let mut map = HashMap::new();
    for pair in qs.split('&').filter(|p| !p.is_empty()) {
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        let val = percent_encoding::percent_decode_str(v)
            .decode_utf8_lossy()
            .into_owned();
        map.insert(k.to_string(), val);
    }
    map
}

fn respond_asset(req: tiny_http::Request, path: &str, gzip: bool) {
    let rel = path.trim_start_matches('/');
    let rel = if rel.is_empty() { "index.html" } else { rel };
    // Serve the asset, else fall back to index.html (client-side routing).
    let (data, ctype) = match assets::WebAssets::get(rel) {
        Some(f) => (f.data.into_owned(), assets::content_type(rel)),
        None => match assets::WebAssets::get("index.html") {
            Some(f) => (f.data.into_owned(), assets::content_type("index.html")),
            None => {
                let resp = tiny_http::Response::from_string(
                    "web UI not built — run `cd web && npm ci && npm run build`",
                )
                .with_status_code(404);
                let _ = req.respond(resp);
                return;
            }
        },
    };
    send(req, 200, data, ctype, gzip);
}

/// Send a response, gzip-compressing the body when the client accepts it and the
/// payload is large enough to be worth it (the tensor-tree JSON is tens of MB).
fn send(req: tiny_http::Request, status: u16, data: Vec<u8>, content_type: &str, gzip: bool) {
    let mut headers = vec![header("Content-Type", content_type)];
    let body = if gzip && data.len() > 1024 {
        match gzip_bytes(&data) {
            Ok(compressed) => {
                headers.push(header("Content-Encoding", "gzip"));
                compressed
            }
            Err(_) => data,
        }
    } else {
        data
    };
    let mut resp = tiny_http::Response::from_data(body).with_status_code(status);
    for h in headers {
        resp = resp.with_header(h);
    }
    let _ = req.respond(resp);
}

fn header(key: &str, value: &str) -> tiny_http::Header {
    tiny_http::Header::from_bytes(key.as_bytes(), value.as_bytes()).expect("valid header")
}

fn gzip_bytes(data: &[u8]) -> std::io::Result<Vec<u8>> {
    use flate2::{Compression, write::GzEncoder};
    use std::io::Write;
    let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
    encoder.write_all(data)?;
    encoder.finish()
}

fn accepts_gzip(req: &tiny_http::Request) -> bool {
    req.headers().iter().any(|h| {
        h.field.equiv("Accept-Encoding") && h.value.as_str().to_ascii_lowercase().contains("gzip")
    })
}

/// This machine's fully-qualified hostname (`hostname -f`), for the reachable URL
/// when bound to all interfaces. `None` if it can't be determined.
fn fqdn() -> Option<String> {
    let out = std::process::Command::new("hostname")
        .arg("-f")
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let name = String::from_utf8(out.stdout).ok()?.trim().to_string();
    if name.is_empty() { None } else { Some(name) }
}

#[cfg(test)]
mod tests {
    use super::parse_query;

    #[test]
    fn decodes_encoded_tensor_name() {
        // The client sends encodeURIComponent("model.layers.0/mlp.weight").
        let q = parse_query("name=model.layers.0%2Fmlp.weight&dtype=f16&rows=8");
        assert_eq!(
            q.get("name").map(String::as_str),
            Some("model.layers.0/mlp.weight")
        );
        assert_eq!(q.get("dtype").map(String::as_str), Some("f16"));
        assert_eq!(q.get("rows").map(String::as_str), Some("8"));
    }

    #[test]
    fn empty_query_is_empty_map() {
        assert!(parse_query("").is_empty());
    }
}
