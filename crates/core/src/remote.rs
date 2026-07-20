//! Read a remote checkpoint's **structure** (tensor names, dtypes, shapes) by
//! delegating to a machine that already has access to it, over one authenticated
//! SSH session ([`crate::sftp::RemoteSession`] — pure Rust, no external binary
//! runs locally or on the server):
//!
//! - a **safetensors directory or file** is read over SFTP — only each shard's
//!   header bytes are fetched, parsed with the local safetensors parser.
//! - an **`s3://…` cstorch checkpoint** is read by running a small
//!   `cerebras.pytorch` script (lazy load, metadata only) in the remote venv over
//!   an SSH exec channel — the one path that inherently needs Python/cstorch on
//!   the remote.
//!
//! Both share the one session, so a read — or `diff`'s two reads — costs a single
//! authentication / password prompt. Credentials/data stay on the remote (nothing
//! is copied locally). Metadata-only: only header/metadata bytes cross the wire.

use std::collections::HashSet;

use anyhow::{Context, Result, anyhow, bail};

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::progress::LoadProgress;
use crate::sftp::RemoteSession;
use crate::stats::{DiskUsage, ShardDisk};
use crate::tree::{Layout, MetadataInfo, Storage, TensorInfo};

/// A remote read's result: tensors, metadata, `config.json`, and the shards'
/// on-disk footprint (all but the tensors optional).
type FetchedCheckpoint = (
    Vec<TensorInfo>,
    Vec<MetadataInfo>,
    Option<crate::config::ModelConfig>,
    Option<DiskUsage>,
    Vec<crate::health::HealthReport>,
);

/// What [`RemoteRead::read`] returns: the tensors, metadata, the shards' on-disk
/// footprint, and the index/file health — all from one pass (shard headers and the
/// index read once), so the health check reuses what the read already parsed
/// rather than fetching headers or the index a second time.
pub struct RemoteCheckpoint {
    pub tensors: Vec<TensorInfo>,
    pub metadata: Vec<MetadataInfo>,
    pub disk: Option<DiskUsage>,
    pub health: Vec<crate::health::HealthReport>,
    /// The underlying S3 objects' metadata — `Some` only for an `s3://` source
    /// (fetched best-effort by the remote script); `None` for a local/SFTP read.
    pub s3: Option<S3Meta>,
}

/// One S3 object under a checkpoint's prefix, with the metadata `diff` compares.
/// Fetched best-effort by the remote dump script via boto3 (the remote's own AWS
/// credentials — nothing S3 happens locally).
#[derive(Debug, Clone)]
pub struct S3Object {
    /// Key relative to the checkpoint prefix, so two prefixes line up by shard.
    pub key: String,
    pub size: u64,
    pub etag: String,
    /// `(algorithm, value)` when the object stored an additional checksum.
    pub checksum: Option<(String, String)>,
    pub last_modified: String,
    /// User-defined `x-amz-meta-*` metadata.
    pub user_meta: BTreeMap<String, String>,
    /// Object tags, or `None` when they couldn't be read (permission) — distinct
    /// from `Some(empty)` meaning "read, no tags".
    pub tags: Option<BTreeMap<String, String>>,
}

/// The S3 objects under an `s3://` checkpoint's prefix, plus any warnings raised
/// while fetching them (e.g. tags denied). `Some` only for an `s3://` source.
#[derive(Debug, Clone, Default)]
pub struct S3Meta {
    pub objects: Vec<S3Object>,
    pub warnings: Vec<String>,
}

/// Line prefix the remote script tags its JSON with, so we can pick it out of any
/// motd / cstorch chatter on the SSH stdout.
const SENTINEL: &str = "CKPT_EXPLORER_META:";

/// Line prefix for the dump script's `done/total` progress reports, streamed
/// ahead of the final metadata so the load bar fills for an `s3://` read too.
const PROGRESS_TAG: &str = "CKPT_EXPLORER_PROG:";

/// Upper bound on SSH sessions used to read one safetensors dir's shards in
/// parallel (work-stealing) — roughly one per shard for a typical sharded model,
/// so no worker is more than ~1 shard deep. If opening this many trips sshd's
/// concurrent-connection limit (e.g. two dirs diffed at once), the refused opens
/// just mean fewer readers and the work-stealing counter still covers every shard.
const MAX_SHARD_SESSIONS: usize = 12;

/// Per-shard header parse output, tagged with the shard's index so results from
/// several parallel readers can be merged back into a deterministic order.
type ShardParse = (usize, Vec<TensorInfo>, Vec<MetadataInfo>);

/// Whether a tensor's `source_path` refers to a remote (`--ssh-read`) source — an
/// `s3://…` URI or an scp-style `[user@]host:path` — for which data views aren't
/// available locally. The scp test (a `:` before any `/`, with a non-empty host to
/// its left) matches how `scp` itself distinguishes a remote target from a local
/// path, so local absolute/relative paths are never misread as remote.
pub fn is_remote_source(source_path: &str) -> bool {
    if source_path.starts_with("s3://") {
        return true;
    }
    match source_path.find(':') {
        Some(colon) if colon > 0 => !source_path[..colon].contains('/'),
        _ => false,
    }
}

/// A remote host + cstorch venv to read checkpoint metadata through (`--ssh-read`
/// / `--ssh-venv`).
#[derive(Clone, Debug)]
pub struct RemoteRead {
    pub host: String,
    pub venv: String,
}

impl RemoteRead {
    pub fn new(host: String, venv: String) -> Self {
        RemoteRead { host, venv }
    }

    /// Read a remote checkpoint's structure over a fresh SSH session (one auth),
    /// with a progress spinner, and also fetch its `config.json` over the *same*
    /// session (no second auth prompt) so the `check`/health config-consistency
    /// check runs against a remote checkpoint too. `None` config for an `s3://`
    /// cstorch checkpoint (no HF `config.json`) or when the sidecar is
    /// absent/unreadable. For several reads sharing one session/prompt (e.g.
    /// `diff`), use [`Self::open_with`] + [`Self::read`] directly.
    pub fn fetch_with_config(&self, src: &str) -> Result<FetchedCheckpoint> {
        let mut password = None;
        let session = self.open_with(&mut password)?;
        eprintln!("checkpoint-explorer: reading tensor metadata over ssh …");
        let bars = crate::progress::Bars::start(vec![src.to_string()]);
        let progress = bars.progress(0);
        // Interactive browse doesn't use S3 object metadata (that's a `diff`-only
        // comparison), so skip the extra per-object HEADs here.
        let out = self.read(&session, src, &password, progress.as_deref(), false);
        bars.finish(0, out.is_ok());
        bars.join();
        let rc = out?;
        let config = self.read_config(&session, src);
        // The index/file health was computed by `read` from the same pass (no
        // second index read or header fetch).
        Ok((rc.tensors, rc.metadata, config, rc.disk, rc.health))
    }

    /// Fetch + parse the remote `config.json` for `src` over an already-open
    /// session. `None` for `s3://` (no HF config) or on any read/parse failure —
    /// the config check then reports `n/a` rather than erroring the whole load.
    pub fn read_config(
        &self,
        session: &RemoteSession,
        src: &str,
    ) -> Option<crate::config::ModelConfig> {
        let path = crate::config::remote_path(src)?;
        let bytes = session.read_file(&path).ok()?;
        let text = String::from_utf8(bytes).ok()?;
        crate::config::ModelConfig::parse(&text).filter(crate::config::ModelConfig::is_meaningful)
    }

    /// List the objects under an `s3://…` checkpoint's prefix over an already-open
    /// session — `(prefix-relative key, size)` per object — via a tiny boto3
    /// `list_objects_v2` (list only, **no** per-object HEAD, so it's one cheap
    /// paginated call). Read-only; the s3-native file browser turns these keys into
    /// a tree ([`crate::filetree::build_from_keys`]).
    pub fn list_s3(&self, session: &RemoteSession, uri: &str) -> Result<Vec<(String, u64)>> {
        let script = list_script(uri);
        let command = format!("source {}/bin/activate && python3 -", self.venv);
        let out = session.exec_capture(&command, &script, |_| {})?;
        let json = out
            .lines()
            .rev()
            .find_map(|l| l.strip_prefix(SENTINEL))
            .ok_or_else(|| {
                anyhow!(
                    "no object listing returned from {} — remote output was:\n{}",
                    self.host,
                    out.trim()
                )
            })?;
        parse_list(json)
    }

    /// Open an authenticated SSH session to the host, reusing/recording a password
    /// so a subsequent session to the same host authenticates without prompting
    /// again — used to read two checkpoints in parallel, and a dir's shards across
    /// a pool, all behind one prompt.
    pub fn open_with(&self, password: &mut Option<String>) -> Result<RemoteSession> {
        RemoteSession::connect_with(&self.host, password)
    }

    /// Read one checkpoint over an already-open session: an `s3://…` cstorch
    /// checkpoint via the cstorch dump over an SSH exec channel, or a remote
    /// safetensors dir/file over SFTP. Tensor data is never read. `password` (the
    /// one already entered for `session`) lets a multi-shard dir open a few more
    /// sessions to read shards in parallel without prompting again.
    pub fn read(
        &self,
        session: &RemoteSession,
        src: &str,
        password: &Option<String>,
        progress: Option<&LoadProgress>,
        want_s3: bool,
    ) -> Result<RemoteCheckpoint> {
        if src.starts_with("s3://") {
            // An s3:// cstorch checkpoint isn't a local filesystem path, so there's
            // no block allocation to measure, and it has no HF index to check. The
            // S3 object metadata (an extra HEAD per object) is fetched only when the
            // caller wants it (`diff`) — a plain browse skips that cost.
            let (tensors, metadata, s3) = self.read_cstorch(session, src, progress, want_s3)?;
            Ok(RemoteCheckpoint {
                tensors,
                metadata,
                disk: None,
                health: Vec::new(),
                s3: want_s3.then_some(s3),
            })
        } else {
            self.read_dir(session, src, password, progress)
        }
    }

    /// A remote safetensors directory/file over SFTP. Its shards' headers are read
    /// **in parallel** across a pool of sessions — `session` plus up to
    /// [`MAX_SHARD_SESSIONS`]`- 1` more opened here (reusing `password`, so no extra
    /// prompt) — sharing one **work-stealing** shard counter, then merged in shard
    /// order deduped by name.
    ///
    /// Work-stealing (rather than a fixed split) means `session` starts reading
    /// immediately while the extra sessions are still completing their SSH
    /// handshakes — hiding that setup latency — and a session drawing a slow or
    /// large-headered shard doesn't hold up the others. A shard is claimed with one
    /// atomic increment; a failed extra-open just means one fewer reader, not a
    /// failed read.
    fn read_dir(
        &self,
        session: &RemoteSession,
        path: &str,
        password: &Option<String>,
        progress: Option<&LoadProgress>,
    ) -> Result<RemoteCheckpoint> {
        use std::sync::atomic::AtomicUsize;

        // One pass over the directory: the shard read order plus the index +
        // listing the health check needs (read once, shared below).
        let crate::sftp::ShardListing {
            files,
            index_path,
            weight_map,
            actual,
        } = session.list_shards(path)?;
        if files.is_empty() {
            bail!("no safetensors files found at {}", self.source_path(path));
        }
        // Now the shard count is known — the bar switches from spinner to filling.
        if let Some(p) = progress {
            p.set_total(files.len());
            p.set_unit(crate::progress::Unit::Shards);
        }
        // Stamp each tensor with *its own* shard's scp-style path (not the dir), so
        // the status line / `f` shows the exact file and it's usable with scp.
        let displays: Vec<String> = files.iter().map(|f| self.source_path(f)).collect();

        let workers = files.len().min(MAX_SHARD_SESSIONS);
        let next = AtomicUsize::new(0);
        let parts: Vec<Result<Vec<ShardParse>>> = std::thread::scope(|s| {
            let (files, displays, next) = (&files, &displays, &next);
            let mut handles = Vec::with_capacity(workers);
            // The already-open session reads straight away.
            handles.push(s.spawn(move || session.read_shards(files, displays, next, progress)));
            // Extra sessions connect in parallel, then join the same queue.
            for _ in 1..workers {
                handles.push(s.spawn(move || {
                    let mut pw = password.clone();
                    match self.open_with(&mut pw) {
                        Ok(extra) => extra.read_shards(files, displays, next, progress),
                        Err(_) => Ok(Vec::new()), // one fewer reader; others cover it
                    }
                }));
            }
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err(anyhow!("shard read thread panicked")))
                })
                .collect()
        });

        let mut all: Vec<ShardParse> = Vec::new();
        for part in parts {
            all.extend(part?);
        }
        let (tensors, metadata) = merge_shards(all);
        if tensors.is_empty() {
            bail!(
                "no tensors in the safetensors headers at {}",
                self.source_path(path)
            );
        }
        // Best-effort filesystem footprint of the shards (one read-only `stat`
        // over SSH). A failure here — no `stat`, non-GNU, restricted shell — just
        // drops the on-disk section from the stats popup; it never fails the load.
        let disk = session
            .allocated_sizes(&files)
            .ok()
            .map(|rows| {
                rows.into_iter()
                    .map(|(p, apparent, allocated)| ShardDisk {
                        name: crate::stats::shard_name(&p),
                        apparent,
                        allocated,
                    })
                    .collect::<Vec<_>>()
            })
            .and_then(DiskUsage::from_shards);

        // Index/file health from the same pass: the index we already read (its
        // `weight_map`) and the directory listing (`actual`), compared against the
        // tensor names the shard read already parsed — grouped by their shard's file
        // name. No second index read, no re-read of any header. A botched/stale
        // index (references shards that aren't there, or lists tensors a shard
        // doesn't hold) surfaces in the tree's health popup and `⚠ health` badge,
        // just as for a local checkpoint.
        let health = match index_path {
            Some(index_path) => {
                let mut present_by_file: HashMap<String, BTreeSet<String>> = HashMap::new();
                for t in &tensors {
                    if let Some(name) = std::path::Path::new(&t.source_path)
                        .file_name()
                        .and_then(|n| n.to_str())
                    {
                        present_by_file
                            .entry(name.to_string())
                            .or_default()
                            .insert(t.name.clone());
                    }
                }
                let report =
                    crate::health::reconcile(&index_path, &weight_map, &actual, &present_by_file);
                if report.has_issues() {
                    vec![report]
                } else {
                    Vec::new()
                }
            }
            None => Vec::new(),
        };

        Ok(RemoteCheckpoint {
            tensors,
            metadata,
            disk,
            health,
            s3: None, // a safetensors dir/file has no S3 object metadata
        })
    }

    /// The `source_path` stamped on each remote tensor: an `s3://…` URI as-is, or a
    /// remote path in **scp form** `[user@]host:path` — so the status line and the
    /// `f` (copy file path) command yield something you can hand straight to
    /// `scp`/`rsync`, and [`is_remote_source`] can still tell it's remote (data
    /// views need the bytes locally).
    fn source_path(&self, src: &str) -> String {
        if src.starts_with("s3://") {
            src.to_string()
        } else {
            format!("{}:{}", self.host, src)
        }
    }

    /// `s3://` cstorch checkpoint: run the (lazy) cstorch dump script in the venv
    /// over an SSH exec channel and parse the sentinel-tagged JSON it prints.
    fn read_cstorch(
        &self,
        session: &RemoteSession,
        src: &str,
        progress: Option<&LoadProgress>,
        want_s3: bool,
    ) -> Result<(Vec<TensorInfo>, Vec<MetadataInfo>, S3Meta)> {
        let script = dump_script(src, want_s3);
        let command = format!("source {}/bin/activate && python3 -", self.venv);
        // Feed the streamed `PROG:done/total` lines into the load bar as they land.
        let out = session.exec_capture(&command, &script, |line| {
            // `PROG:done/total[/unit]` — the optional unit switches the bar's label
            // for the second (S3-metadata) phase; absent ⇒ tensors (back-compat).
            if let Some(rest) = line.strip_prefix(PROGRESS_TAG) {
                let mut parts = rest.splitn(3, '/');
                if let (Some(d), Some(t)) = (parts.next(), parts.next())
                    && let (Ok(done), Ok(total)) = (d.trim().parse(), t.trim().parse())
                    && let Some(p) = progress
                {
                    let unit = match parts.next().map(str::trim) {
                        Some("s3") => crate::progress::Unit::S3Objects,
                        _ => crate::progress::Unit::Tensors,
                    };
                    p.set_total(total);
                    p.set_done(done);
                    p.set_unit(unit);
                }
            }
        })?;
        let json = out
            .lines()
            .rev()
            .find_map(|l| l.strip_prefix(SENTINEL))
            .ok_or_else(|| {
                anyhow!(
                    "no metadata returned from {} — remote output was:\n{}",
                    self.host,
                    out.trim()
                )
            })?;
        parse_dump(json, &self.source_path(src))
    }
}

/// The cstorch dump script for an `s3://…` checkpoint: `cstorch.load` (lazy — no
/// tensor data) and emit each tensor's name/dtype/shape/itemsize as a
/// sentinel-tagged JSON line. The URI is embedded as a JSON string literal (valid
/// Python), so nothing needs quoting at the shell. (Safetensors dirs/files don't
/// use this — they're read over SFTP; see [`crate::sftp`].)
///
/// **Read-only:** the script only *loads* (lazily) and writes its output to
/// stdout — it never opens a file for writing, calls `cstorch.save`/`torch.save`,
/// or otherwise mutates the checkpoint. The remote checkpoint is never modified.
fn dump_script(src: &str, want_s3: bool) -> String {
    let src_lit = serde_json::to_string(src).unwrap_or_else(|_| "\"\"".into());
    const TEMPLATE: &str = r#"
import sys, json
SRC = __URI__
WANT_S3 = __WANT_S3__
S = "__SENTINEL__"
P = "__PROGRESS__"
def emit(obj):
    sys.stdout.write(S + json.dumps(obj) + "\n")
    sys.stdout.flush()
def prog(done, total, unit=None):
    tail = ("/" + unit) if unit else ""
    sys.stdout.write("%s%d/%d%s\n" % (P, done, total, tail))
    sys.stdout.flush()
try:
    import cerebras.pytorch as cstorch
except Exception as e:
    emit({"error": "import cerebras.pytorch failed: %r" % (e,)}); sys.exit(0)
try:
    sd = cstorch.load(SRC, map_location=None)   # lazy: metadata only, no data
except Exception as e:
    emit({"error": "cstorch.load failed: %r" % (e,)}); sys.exit(0)
keys = list(sd.keys())
total = len(keys)
prog(0, total)                                  # total known → bar goes determinate
step = max(1, total // 100)                     # ~100 updates, not one per tensor
tensors = []
for i, name in enumerate(keys):
    try:
        t = sd[name]
        it = int(t.element_size()) if hasattr(t, "element_size") else 0
        tensors.append({"name": str(name), "dtype": str(getattr(t, "dtype", "")), "shape": [int(d) for d in t.shape], "itemsize": it})
    except Exception:
        pass
    if (i + 1) % step == 0 or i + 1 == total:
        prog(i + 1, total)
# S3 object metadata (best-effort, read-only): list the objects under the prefix
# and HEAD each for size/etag/checksum/last-modified/user-metadata; tags need a
# separate (often ungranted) permission, so they're tried per object and a single
# warning is emitted if denied. Any failure degrades to fewer objects + a warning.
s3_objects = []
s3_warnings = []
if WANT_S3:
    try:
        import boto3
        from urllib.parse import urlparse
        u = urlparse(SRC)
        bucket, prefix = u.netloc, u.path.lstrip("/")
        cli = boto3.client("s3")
        keys, tok = [], None
        while True:
            kw = {"Bucket": bucket, "Prefix": prefix}
            if tok: kw["ContinuationToken"] = tok
            resp = cli.list_objects_v2(**kw)
            keys.extend(it["Key"] for it in resp.get("Contents", []))
            if resp.get("IsTruncated"): tok = resp.get("NextContinuationToken")
            else: break
        # Second progress phase: HEADing each object is the slow part, so drive the
        # bar off it (relabelled "S3 objects") instead of sitting at 100% tensors.
        nkeys = len(keys)
        s3_step = max(1, nkeys // 100)
        prog(0, nkeys, "s3")
        tags_denied = False
        for i, k in enumerate(keys):
            if i % s3_step == 0:
                prog(i, nkeys, "s3")
            try:
                h = cli.head_object(Bucket=bucket, Key=k, ChecksumMode="ENABLED")
            except Exception as e:
                s3_warnings.append("head_object failed for %s: %r" % (k, e)); continue
            rel = k[len(prefix):].lstrip("/") if k.startswith(prefix) else k
            lm = h.get("LastModified")
            obj = {"key": rel, "size": int(h.get("ContentLength", 0)),
                   "etag": str(h.get("ETag", "")).strip('"'),
                   "last_modified": lm.isoformat() if lm else "",
                   "metadata": {mk: str(mv) for mk, mv in (h.get("Metadata") or {}).items()}}
            for algo in ("CRC32", "CRC32C", "SHA1", "SHA256"):
                cv = h.get("Checksum" + algo)
                if cv:
                    obj["checksum"] = [algo.lower(), str(cv)]; break
            try:
                tg = cli.get_object_tagging(Bucket=bucket, Key=k)
                obj["tags"] = {t["Key"]: t["Value"] for t in tg.get("TagSet", [])}
            except Exception as e:
                if not tags_denied:
                    s3_warnings.append("tags unavailable (needs s3:GetObjectTagging): %r" % (e,))
                    tags_denied = True
            s3_objects.append(obj)
        prog(nkeys, nkeys, "s3")
    except Exception as e:
        s3_warnings.append("s3 metadata unavailable: %r" % (e,))
emit({"tensors": tensors, "metadata": [], "s3_objects": s3_objects, "s3_warnings": s3_warnings})
"#;
    TEMPLATE
        .replace("__URI__", &src_lit)
        .replace("__WANT_S3__", if want_s3 { "True" } else { "False" })
        .replace("__SENTINEL__", SENTINEL)
        .replace("__PROGRESS__", PROGRESS_TAG)
}

/// The boto3 object-listing script for an `s3://…` checkpoint: a single paginated
/// `list_objects_v2` emitting one sentinel-tagged JSON line
/// `{objects:[[rel_key,size],…]}` (or `{error:…}`). Keys are made
/// prefix-relative so the browser shows them s3-natively. Distinct from
/// [`dump_script`]'s S3 phase, which additionally HEADs each object for the diff
/// metadata compare — this is **list only**, no per-object request.
///
/// **Read-only:** `list_objects_v2` is a read; the script never writes.
fn list_script(uri: &str) -> String {
    let src_lit = serde_json::to_string(uri).unwrap_or_else(|_| "\"\"".into());
    const TEMPLATE: &str = r#"
import sys, json
SRC = __URI__
S = "__SENTINEL__"
def emit(obj):
    sys.stdout.write(S + json.dumps(obj) + "\n"); sys.stdout.flush()
try:
    import boto3
    from urllib.parse import urlparse
except Exception as e:
    emit({"error": "import boto3 failed: %r" % (e,)}); sys.exit(0)
try:
    u = urlparse(SRC)
    bucket, prefix = u.netloc, u.path.lstrip("/")
    cli = boto3.client("s3")
    objects, tok = [], None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if tok: kw["ContinuationToken"] = tok
        resp = cli.list_objects_v2(**kw)
        for it in resp.get("Contents", []):
            k = it["Key"]
            rel = k[len(prefix):].lstrip("/") if k.startswith(prefix) else k
            if rel:
                objects.append([rel, int(it.get("Size", 0))])
        if resp.get("IsTruncated"): tok = resp.get("NextContinuationToken")
        else: break
except Exception as e:
    emit({"error": "s3 list failed: %r" % (e,)}); sys.exit(0)
emit({"objects": objects})
"#;
    TEMPLATE
        .replace("__URI__", &src_lit)
        .replace("__SENTINEL__", SENTINEL)
}

/// Parse the object-listing JSON (`{objects:[[key,size],…]}` or `{error:…}`) into
/// `(prefix-relative key, size)` pairs. Malformed entries are skipped.
fn parse_list(json: &str) -> Result<Vec<(String, u64)>> {
    let v: serde_json::Value =
        serde_json::from_str(json).with_context(|| format!("parsing object listing: {json}"))?;
    if let Some(e) = v.get("error").and_then(serde_json::Value::as_str) {
        bail!("{e}");
    }
    let arr = v
        .get("objects")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow!("object listing had no `objects` array"))?;
    let mut out = Vec::with_capacity(arr.len());
    for it in arr {
        if let Some(pair) = it.as_array()
            && pair.len() == 2
            && let (Some(k), Some(sz)) = (pair[0].as_str(), pair[1].as_u64())
        {
            out.push((k.to_string(), sz));
        }
    }
    Ok(out)
}

/// Parse the remote JSON (`{tensors:[…], metadata:[…]}` or `{error:…}`) into
/// [`TensorInfo`]s + [`MetadataInfo`], stamping each tensor with `source_path`
/// (already remote-marked; see [`RemoteRead::source_path`]) so display, the `y`
/// command, and the data-view "local-only" guard all behave.
fn parse_dump(
    json: &str,
    source_path: &str,
) -> Result<(Vec<TensorInfo>, Vec<MetadataInfo>, S3Meta)> {
    let v: serde_json::Value =
        serde_json::from_str(json).context("parsing the remote metadata JSON")?;
    if let Some(err) = v.get("error").and_then(|e| e.as_str()) {
        bail!("remote: {err}");
    }
    let mut tensors = Vec::new();
    if let Some(arr) = v.get("tensors").and_then(|t| t.as_array()) {
        for item in arr {
            let name = item
                .get("name")
                .and_then(|x| x.as_str())
                .unwrap_or_default()
                .to_string();
            let dtype = map_dtype(
                item.get("dtype")
                    .and_then(|x| x.as_str())
                    .unwrap_or_default(),
            );
            let shape: Vec<usize> = item
                .get("shape")
                .and_then(|x| x.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|d| d.as_u64().map(|n| n as usize))
                        .collect()
                })
                .unwrap_or_default();
            let itemsize = item.get("itemsize").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
            let num_elements: usize = shape.iter().product();
            tensors.push(TensorInfo {
                name,
                dtype,
                shape,
                size_bytes: num_elements * itemsize,
                num_elements,
                storage: Storage::Unknown,
                source_path: source_path.to_string(),
                layout: Layout::None,
            });
        }
    }
    if tensors.is_empty() {
        bail!("the remote returned no tensors for {source_path}");
    }
    // safetensors `__metadata__` entries (name/value/value_type), when present.
    let metadata = v
        .get("metadata")
        .and_then(|m| m.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|e| {
                    let name = e.get("name").and_then(|x| x.as_str())?.to_string();
                    let value = e.get("value").and_then(|x| x.as_str())?.to_string();
                    let value_type = e
                        .get("value_type")
                        .and_then(|x| x.as_str())
                        .unwrap_or("string")
                        .to_string();
                    Some(MetadataInfo {
                        name,
                        value,
                        value_type,
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    let s3 = parse_s3_meta(&v);
    Ok((tensors, metadata, s3))
}

/// Parse the remote script's optional `s3_objects` / `s3_warnings` fields into
/// [`S3Meta`]. Missing / malformed fields degrade to empty (never an error) — the
/// tensor dump is what matters; S3 metadata is best-effort.
fn parse_s3_meta(v: &serde_json::Value) -> S3Meta {
    let str_map = |val: Option<&serde_json::Value>| -> BTreeMap<String, String> {
        val.and_then(|m| m.as_object())
            .map(|o| {
                o.iter()
                    .filter_map(|(k, v)| Some((k.clone(), v.as_str()?.to_string())))
                    .collect()
            })
            .unwrap_or_default()
    };
    let objects = v
        .get("s3_objects")
        .and_then(|a| a.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|o| {
                    let key = o.get("key").and_then(|x| x.as_str())?.to_string();
                    let checksum = o
                        .get("checksum")
                        .and_then(|c| c.as_array())
                        .and_then(|c| Some((c.first()?.as_str()?, c.get(1)?.as_str()?)))
                        .map(|(a, b)| (a.to_string(), b.to_string()));
                    Some(S3Object {
                        key,
                        size: o.get("size").and_then(|x| x.as_u64()).unwrap_or(0),
                        etag: o
                            .get("etag")
                            .and_then(|x| x.as_str())
                            .unwrap_or_default()
                            .to_string(),
                        checksum,
                        last_modified: o
                            .get("last_modified")
                            .and_then(|x| x.as_str())
                            .unwrap_or_default()
                            .to_string(),
                        user_meta: str_map(o.get("metadata")),
                        // `tags` absent in the JSON ⇒ couldn't be read (None);
                        // present (even empty) ⇒ read successfully.
                        tags: o.get("tags").map(|t| str_map(Some(t))),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    let warnings = v
        .get("s3_warnings")
        .and_then(|a| a.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|w| w.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    S3Meta { objects, warnings }
}

/// Map a torch dtype string (`torch.float16`) to the display name used elsewhere
/// (`F16`); unknown types pass through uppercased.
fn map_dtype(torch: &str) -> String {
    let s = torch.strip_prefix("torch.").unwrap_or(torch);
    match s {
        "float16" => "F16",
        "bfloat16" => "BF16",
        "float32" => "F32",
        "float64" => "F64",
        "float8_e4m3fn" => "F8_E4M3",
        "float8_e5m2" => "F8_E5M2",
        "int8" => "I8",
        "uint8" => "U8",
        "int16" => "I16",
        "uint16" => "U16",
        "int32" => "I32",
        "uint32" => "U32",
        "int64" => "I64",
        "uint64" => "U64",
        "bool" => "BOOL",
        other => return other.to_uppercase(),
    }
    .to_string()
}

/// Merge per-shard parse results into one checkpoint: order by shard index (so the
/// result is deterministic regardless of which parallel reader finished first),
/// then flatten, keeping the first occurrence of each tensor / metadata name.
fn merge_shards(mut shards: Vec<ShardParse>) -> (Vec<TensorInfo>, Vec<MetadataInfo>) {
    shards.sort_by_key(|(idx, _, _)| *idx);
    let (mut tensors, mut metadata) = (Vec::new(), Vec::new());
    let (mut seen_t, mut seen_m) = (HashSet::new(), HashSet::new());
    for (_, ts, ms) in shards {
        for t in ts {
            if seen_t.insert(t.name.clone()) {
                tensors.push(t);
            }
        }
        for m in ms {
            if seen_m.insert(m.name.clone()) {
                metadata.push(m);
            }
        }
    }
    (tensors, metadata)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cstorch_script_embeds_source_safely() {
        let s = dump_script("s3://b/k", false);
        assert!(s.contains("SRC = \"s3://b/k\""));
        assert!(s.contains("import cerebras.pytorch"));
        assert!(s.contains(SENTINEL));
    }

    #[test]
    fn cstorch_script_fetches_s3_metadata_only_when_wanted() {
        // Not wanted (interactive browse / check) → no boto3 work.
        let off = dump_script("s3://b/ckpt", false);
        assert!(off.contains("WANT_S3 = False"), "{off}");

        let s = dump_script("s3://b/ckpt", true);
        assert!(s.contains("WANT_S3 = True"), "{s}");
        // Fetches object metadata via boto3, read-only calls only.
        assert!(s.contains("boto3.client(\"s3\")"));
        assert!(s.contains("list_objects_v2"));
        assert!(s.contains("head_object"));
        assert!(s.contains("ChecksumMode=\"ENABLED\""));
        assert!(s.contains("get_object_tagging"));
        assert!(s.contains("s3_objects"));
        assert!(s.contains("s3_warnings"));
        // Reports S3 progress as a second phase (so the bar doesn't sit at 100%).
        assert!(s.contains("\"s3\""));
        // Read-only: never writes / uploads / deletes / puts tags.
        for forbidden in [
            "put_object",
            "upload",
            "delete_object",
            "put_object_tagging",
            "copy_object",
        ] {
            assert!(
                !s.contains(forbidden),
                "script must stay read-only: {forbidden}"
            );
        }
    }

    #[test]
    fn list_script_is_read_only_list_and_embeds_uri() {
        let s = list_script("s3://bucket/ckpt");
        assert!(s.contains("SRC = \"s3://bucket/ckpt\""));
        assert!(s.contains("boto3.client(\"s3\")"));
        assert!(s.contains("list_objects_v2"));
        assert!(s.contains(SENTINEL));
        // List only — no per-object HEAD/tag calls, and nothing that mutates.
        for forbidden in [
            "head_object",
            "get_object_tagging",
            "put_object",
            "upload",
            "delete_object",
            "copy_object",
        ] {
            assert!(
                !s.contains(forbidden),
                "list script must stay list-only + read-only: {forbidden}"
            );
        }
    }

    #[test]
    fn parse_list_reads_pairs_and_surfaces_errors() {
        let ok = parse_list(r#"{"objects":[["a/b.bin",100],["c.json",5]]}"#).unwrap();
        assert_eq!(
            ok,
            vec![("a/b.bin".to_string(), 100), ("c.json".to_string(), 5)]
        );
        // A remote error becomes an Err.
        assert!(parse_list(r#"{"error":"access denied"}"#).is_err());
        // Malformed entries are skipped, not fatal.
        let mixed = parse_list(r#"{"objects":[["good",1],["bad"],[1,2]]}"#).unwrap();
        assert_eq!(mixed, vec![("good".to_string(), 1)]);
    }

    #[test]
    fn parses_safetensors_dump_with_metadata_and_marks_source() {
        let json = r#"{"tensors":[
            {"name":"model.embed_tokens.weight","dtype":"BF16","shape":[151936,2048],"itemsize":2}
        ],"metadata":[{"name":"format","value":"pt","value_type":"string"}]}"#;
        let (t, m, s3) = parse_dump(json, "lab@host:/opt/models/ckpt").unwrap();
        assert_eq!(t[0].dtype, "BF16");
        assert_eq!(t[0].shape, vec![151936, 2048]);
        assert_eq!(t[0].source_path, "lab@host:/opt/models/ckpt");
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].name, "format");
        assert_eq!(m[0].value, "pt");
        // No `s3_objects` in the JSON → empty (this path is for safetensors dumps).
        assert!(s3.objects.is_empty() && s3.warnings.is_empty());
    }

    #[test]
    fn parses_s3_objects_and_warnings() {
        let json = r#"{"tensors":[
            {"name":"w","dtype":"F16","shape":[4],"itemsize":2}
        ],"metadata":[],"s3_objects":[
            {"key":"shard0.dat","size":1024,"etag":"abc","last_modified":"2026-01-02T03:04:05+00:00",
             "checksum":["sha256","deadbeef"],"metadata":{"run":"42"},"tags":{"env":"prod"}},
            {"key":"shard1.dat","size":2048,"etag":"def","last_modified":"2026-01-02T03:04:06+00:00"}
        ],"s3_warnings":["tags unavailable (needs s3:GetObjectTagging): AccessDenied"]}"#;
        let (_t, _m, s3) = parse_dump(json, "s3://b/ckpt").unwrap();
        assert_eq!(s3.objects.len(), 2);
        let o0 = &s3.objects[0];
        assert_eq!(o0.key, "shard0.dat");
        assert_eq!(o0.size, 1024);
        assert_eq!(o0.checksum, Some(("sha256".into(), "deadbeef".into())));
        assert_eq!(o0.user_meta.get("run").map(String::as_str), Some("42"));
        assert_eq!(
            o0.tags
                .as_ref()
                .and_then(|t| t.get("env"))
                .map(String::as_str),
            Some("prod")
        );
        // Second object had no `tags` key → None (couldn't be read), distinct from empty.
        assert!(s3.objects[1].tags.is_none());
        assert_eq!(s3.warnings.len(), 1);
    }

    #[test]
    fn source_path_is_scp_style_but_leaves_s3() {
        let r = RemoteRead::new("lab@host".into(), "~/venv".into());
        assert_eq!(r.source_path("s3://b/k"), "s3://b/k");
        assert_eq!(r.source_path("/opt/models/x"), "lab@host:/opt/models/x");
    }

    #[test]
    fn detects_remote_sources() {
        assert!(is_remote_source("s3://bucket/ckpt"));
        assert!(is_remote_source("lab@host:/opt/models/x"));
        assert!(is_remote_source("host:relative/path"));
        // local paths are never remote, even with a ':' inside a subdir
        assert!(!is_remote_source("/opt/models/x"));
        assert!(!is_remote_source("./model.safetensors"));
        assert!(!is_remote_source("dir/a:b"));
    }

    #[test]
    fn parses_dump_into_tensors() {
        let json = r#"{"tensors":[
            {"name":"a.weight","dtype":"torch.float16","shape":[6,4],"itemsize":2},
            {"name":"b","dtype":"torch.int32","shape":[5],"itemsize":4}
        ],"metadata":[]}"#;
        let (t, _m, _s3) = parse_dump(json, "s3://bucket/ckpt").unwrap();
        assert_eq!(t.len(), 2);
        assert_eq!(t[0].dtype, "F16");
        assert_eq!(t[0].shape, vec![6, 4]);
        assert_eq!(t[0].num_elements, 24);
        assert_eq!(t[0].size_bytes, 48);
        assert_eq!(t[0].source_path, "s3://bucket/ckpt");
        assert_eq!(t[1].dtype, "I32");
    }

    #[test]
    fn surfaces_remote_error() {
        let err = parse_dump(r#"{"error":"cstorch.load failed: boom"}"#, "s3://x/y");
        assert!(err.unwrap_err().to_string().contains("boom"));
    }

    #[test]
    fn maps_common_dtypes() {
        assert_eq!(map_dtype("torch.bfloat16"), "BF16");
        assert_eq!(map_dtype("torch.uint8"), "U8");
        assert_eq!(map_dtype("torch.weirdtype"), "WEIRDTYPE");
    }

    fn tensor(name: &str) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            dtype: "F16".into(),
            shape: vec![1],
            size_bytes: 2,
            num_elements: 1,
            storage: Storage::Unknown,
            source_path: "h:/p".into(),
            layout: Layout::None,
        }
    }
    fn meta(name: &str) -> MetadataInfo {
        MetadataInfo {
            name: name.to_string(),
            value: "v".into(),
            value_type: "string".into(),
        }
    }

    #[test]
    fn merge_shards_orders_by_index_and_dedups_first_seen() {
        // Deliberately out of order (as parallel readers may finish): shard 2 then
        // 0 then 1. `b` appears in shards 0 and 2 → the shard-0 copy wins.
        let parts = vec![
            (2, vec![tensor("c")], vec![meta("fmt")]),
            (0, vec![tensor("a"), tensor("b")], vec![meta("fmt")]),
            (1, vec![tensor("b")], vec![]),
        ];
        let (t, m) = merge_shards(parts);
        let names: Vec<&str> = t.iter().map(|x| x.name.as_str()).collect();
        assert_eq!(names, ["a", "b", "c"]); // shard order, `b` deduped
        assert_eq!(m.len(), 1); // duplicate `fmt` metadata collapsed
        assert_eq!(m[0].name, "fmt");
    }
}
