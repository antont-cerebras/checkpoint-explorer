//! Pure-Rust SSH client for reading remote checkpoint metadata over a single
//! authenticated session (via `ssh2`/libssh2) — no external binary runs, locally
//! or on the server, and the one session is reused for everything, so there's one
//! auth / one password prompt even for `diff`'s two reads:
//!
//! - a **safetensors directory/file** is read over SFTP — open each shard, read
//!   just its header (8-byte length + JSON; SFTP fetches only the bytes we ask
//!   for, never the multi-GB body) and parse with the shared safetensors parser
//!   ([`crate::stheader::parse_header`]).
//! - an **`s3://…` cstorch checkpoint** is read by running the lazy cstorch dump
//!   script over an SSH *exec* channel ([`RemoteSession::exec_capture`]); the
//!   caller ([`crate::remote`]) builds the script and parses the result.
//!
//! **Read-only, metadata-only.** Every remote access is a read: files are opened
//! read-only ([`open_readonly`] — `OpenFlags::READ`, no create/write/truncate
//! bits), and the module never issues `mkdir` / `remove` / `rename` / `setstat`
//! or an `s3://` command that writes. The cstorch path only *loads* the
//! checkpoint and prints metadata to stdout. So `--ssh-read` cannot create or
//! modify anything on the server — and no tensor data crosses the wire.

use std::collections::{BTreeSet, HashMap};
use std::io::{IsTerminal, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use ssh2::{CheckResult, KnownHostFileKind, Session};

use crate::tree::{MetadataInfo, TensorInfo};

/// What one pass over a remote directory (one `readdir`, one index read) yields:
/// the shard read order plus the pieces the index health check needs — so the
/// index and listing are read once, shared between reading the shards and checking
/// them (mirroring the local single-read path).
pub struct ShardListing {
    /// Shard files to read, in index order then any extras (full remote paths).
    pub files: Vec<String>,
    /// The `model.safetensors.index.json` path, when the directory has a usable
    /// one (a `weight_map`); `None` for a single file or a missing/unparsable index.
    pub index_path: Option<String>,
    /// The index's `weight_map` (tensor name -> shard file basename); empty without
    /// a usable index.
    pub weight_map: HashMap<String, String>,
    /// The `.safetensors` file basenames actually present in the directory.
    pub actual: BTreeSet<String>,
}

/// One entry in a remote directory listing: a file (with its size) or a
/// subdirectory — a 2-case sum instead of a `(name, size, is_dir)` tuple whose
/// bool gated the size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RemoteDirEntry {
    File { name: String, size: u64 },
    Directory { name: String },
}

/// One directory's entries — the file browser's view of a remote directory
/// (see [`RemoteSession::walk_dir`]).
pub type DirListing = Vec<RemoteDirEntry>;

/// A remote path's canonical filesystem metadata, with symlinks **followed** —
/// the value type of [`RemoteSession::stat_paths`], the single source of truth for
/// remote "file-like" sizes.
/// A followed (`stat -L`) remote path is either a file (with sizes) or a
/// directory (no size) — a 2-case sum instead of a size-carrying `is_dir` flag,
/// so "directory with a nonzero size" is unrepresentable. Symlinks are already
/// followed by `stat -L`, so there's no link case here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteStat {
    File {
        /// Apparent size (`st_size`) of the target, in bytes.
        apparent: u64,
        /// On-disk allocation (`st_blocks × block-size`) of the target, in bytes.
        allocated: u64,
    },
    Directory,
}

impl RemoteStat {
    pub fn is_dir(&self) -> bool {
        matches!(self, RemoteStat::Directory)
    }
    /// Apparent size in bytes (0 for a directory).
    pub fn apparent(&self) -> u64 {
        match self {
            RemoteStat::File { apparent, .. } => *apparent,
            RemoteStat::Directory => 0,
        }
    }
    /// On-disk allocation in bytes (0 for a directory).
    pub fn allocated(&self) -> u64 {
        match self {
            RemoteStat::File { allocated, .. } => *allocated,
            RemoteStat::Directory => 0,
        }
    }
}

/// An authenticated SSH session. Constructed once per host, then reused for every
/// read (SFTP for safetensors dirs, an exec channel for the cstorch dump), so a
/// checkpoint — or two, for `diff` — costs one authentication / one password
/// prompt.
pub struct RemoteSession {
    session: Session,
}

impl RemoteSession {
    /// Connect to `[user@]host[:port]`, verify the host key against
    /// `~/.ssh/known_hosts`, and authenticate (SSH agent → default identity files
    /// → password / keyboard-interactive prompt), reusing/recording a password in
    /// `password` so a second connection to the same host (another parallel `diff`
    /// side, or a shard-reading pool member) authenticates without prompting again.
    /// Agent/key auth needs no prompt regardless. Silent — the caller announces the
    /// connection once, so opening a pool of sessions doesn't spam the terminal.
    pub fn connect_with(target: &str, password: &mut Option<String>) -> Result<Self> {
        let (user, host, port) = parse_target(target);
        // A rejected password leaves the ssh2 session unusable for anything else, so
        // each attempt authenticates on a *fresh* connection (rather than retrying
        // on the tainted one, which then fails the read). A freshly-prompted
        // password gets a few tries — re-prompting on a new connection each time;
        // a cached one (reused for a second connection) gets a single try, no loop.
        let max_attempts = if password.is_some() { 1 } else { 3 };
        let mut last_err = None;
        for attempt in 1..=max_attempts {
            let session = handshake(&host, port)?;
            match authenticate(&session, &user, &host, password) {
                Ok(()) => return Ok(RemoteSession { session }),
                Err(e) => {
                    if attempt < max_attempts {
                        let msg = format!("checkpoint-explorer: {e} — try again");
                        // Bold yellow on a colour terminal (respecting NO_COLOR);
                        // plain when piped.
                        if std::io::stderr().is_terminal() && std::env::var_os("NO_COLOR").is_none()
                        {
                            eprintln!("\x1b[1;33m{msg}\x1b[0m");
                        } else {
                            eprintln!("{msg}");
                        }
                        *password = None; // re-prompt on the next attempt
                    }
                    last_err = Some(e);
                }
            }
        }
        Err(last_err.expect("at least one attempt was made"))
    }

    /// Run `command` on the host over an SSH exec channel, feed it `stdin`, and
    /// return its combined stdout+stderr. No local process is spawned and this
    /// reuses the session's authentication — used for the `s3://` cstorch dump.
    /// Run `command` with `stdin` piped in, streaming the merged stdout/stderr:
    /// `on_line` is called with each complete line *as it arrives* (so a remote
    /// script's progress lines can drive a live bar), and the full output is
    /// returned once the command exits.
    pub fn exec_capture(
        &self,
        command: &str,
        stdin: &str,
        mut on_line: impl FnMut(&str),
    ) -> Result<String> {
        let mut ch = self
            .session
            .channel_session()
            .context("opening an SSH exec channel")?;
        // Fold stderr into stdout so a single read can't deadlock on a full
        // stderr window (and cstorch chatter is captured for error messages).
        ch.handle_extended_data(ssh2::ExtendedData::Merge).ok();
        ch.exec(command)
            .with_context(|| format!("running `{command}` on the remote"))?;
        ch.write_all(stdin.as_bytes())
            .context("sending the script to the remote")?;
        ch.send_eof().ok();

        // Read incrementally and hand off complete lines as they land. Lines are
        // split on raw bytes and decoded whole, so a multi-byte char spanning a
        // read boundary is never cut mid-sequence.
        let mut out = String::new();
        let mut pending: Vec<u8> = Vec::new();
        let mut buf = [0u8; 8192];
        loop {
            let n = ch.read(&mut buf).context("reading the remote output")?;
            if n == 0 {
                break;
            }
            pending.extend_from_slice(&buf[..n]);
            while let Some(pos) = pending.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = pending.drain(..=pos).collect();
                let text = String::from_utf8_lossy(&line);
                on_line(text.trim_end_matches(['\n', '\r']));
                out.push_str(&text);
            }
        }
        if !pending.is_empty() {
            let text = String::from_utf8_lossy(&pending);
            on_line(&text);
            out.push_str(&text);
        }

        ch.wait_close().ok();
        let code = ch.exit_status().unwrap_or(0);
        if code != 0 {
            bail!("remote command exited with {code}:\n{}", out.trim());
        }
        Ok(out)
    }

    /// List the `.safetensors` shards of a directory (or a single file) over SFTP,
    /// together with the index/listing pieces the health check needs — one
    /// `readdir` and one index read, shared by the shard read and the check.
    pub fn list_shards(&self, path: &str) -> Result<ShardListing> {
        let sftp = self.session.sftp().context("opening the SFTP subsystem")?;
        list_shards(&sftp, path)
    }

    /// Read a whole small remote file — a metadata sidecar like `config.json` —
    /// over SFTP, strictly read-only ([`open_readonly`]). Never used for tensor
    /// data (that stays on the host); the file is expected to be a few KB.
    pub fn read_file(&self, path: &str) -> Result<Vec<u8>> {
        let sftp = self.session.sftp().context("opening the SFTP subsystem")?;
        read_all(&sftp, path)
    }

    /// Recursively list a remote directory tree (down to `max_depth`) over **one**
    /// SFTP channel plus **one** `stat -L` exec — the backend for the file browser
    /// on an SFTP checkpoint. Returns a map from each directory's path to its
    /// entries `(name, size, is_dir)`, which [`crate::filetree::build_from`] then
    /// assembles with zero further round-trips (so the first `Tab` isn't a per-
    /// directory channel-open storm). Dotfiles (and `.`/`..`) are skipped.
    ///
    /// **Symlinks are followed for sizing.** `readdir` reports an `lstat` size — for
    /// a symlinked shard that's the link *path* length (tens of bytes), not the
    /// target it opens to; so every symlink is resolved in one batched `stat -L`
    /// and its size becomes the target's. A symlink to a directory stays a **leaf**
    /// (not descended), matching the local walk, so the tree can't cycle. This
    /// keeps the browser's size equal to what the layout map reads when it opens
    /// the file (both follow the link) — see [`read_header_sized`].
    ///
    /// The root listing must succeed (surfaces an auth/permission error); a deeper
    /// directory that can't be listed degrades to empty.
    pub fn walk_dir(&self, root: &str, max_depth: usize) -> Result<HashMap<String, DirListing>> {
        let sftp = self.session.sftp().context("opening the SFTP subsystem")?;
        let root = root.trim_end_matches('/').to_string();
        let mut map: HashMap<String, DirListing> = HashMap::new();
        // Symlinks to resolve after the walk, in one batch: (dir, row index, full).
        let mut links: Vec<(String, usize, String)> = Vec::new();
        let mut stack = vec![(root.clone(), 0usize)];
        while let Some((dir, depth)) = stack.pop() {
            if depth >= max_depth {
                continue;
            }
            let entries = match sftp.readdir(Path::new(&dir)) {
                Ok(e) => e,
                Err(err) => {
                    if dir == root {
                        return Err(err).with_context(|| format!("listing {root}"));
                    }
                    map.insert(dir, Vec::new()); // an unreadable subdir shows empty
                    continue;
                }
            };
            let mut rows: Vec<RemoteDirEntry> = Vec::new();
            for (p, st) in entries {
                let name = match p.file_name().and_then(|n| n.to_str()) {
                    Some(n) if !n.starts_with('.') => n.to_string(),
                    _ => continue,
                };
                let full = format!("{dir}/{name}");
                // A symlink (S_IFLNK) is sized by a follow-up `stat -L`; the link's
                // own size is a fallback if that can't run (broken link / no GNU
                // stat). It stays a leaf, so it's never descended.
                let is_symlink = st.perm.is_some_and(|m| m & 0o170000 == 0o120000);
                if is_symlink {
                    links.push((dir.clone(), rows.len(), full));
                    // Pushed as a File leaf with a fallback size; the follow-up
                    // `stat -L` below replaces the size with the target's.
                    rows.push(RemoteDirEntry::File {
                        name,
                        size: st.size.unwrap_or(0),
                    });
                } else {
                    let is_dir = st.is_dir();
                    if is_dir && depth + 1 < max_depth {
                        stack.push((full, depth + 1));
                    }
                    rows.push(if is_dir {
                        RemoteDirEntry::Directory { name }
                    } else {
                        RemoteDirEntry::File {
                            name,
                            size: st.size.unwrap_or(0),
                        }
                    });
                }
            }
            map.insert(dir, rows);
        }
        // Follow every symlink in one `stat -L` and replace its fallback size with
        // the target's (a symlinked dir stays a size-0 leaf — never descended).
        if !links.is_empty() {
            let paths: Vec<String> = links.iter().map(|(_, _, p)| p.clone()).collect();
            let resolved = self.stat_paths(&paths);
            for (dir, idx, full) in &links {
                if let Some(st) = resolved.get(full)
                    && let Some(rows) = map.get_mut(dir)
                    && let Some(RemoteDirEntry::File { size, .. }) = rows.get_mut(*idx)
                {
                    // A symlinked dir stays a size-0 leaf (never descended).
                    *size = if st.is_dir() { 0 } else { st.apparent() };
                }
            }
        }
        Ok(map)
    }

    /// **The single source of truth for remote "file-like" sizes** — the file
    /// browser, the layout map, and the stats on-disk section all resolve a path's
    /// size through here, so they can't disagree. Stats `paths` in one `stat -L`
    /// exec with symlinks **followed**, so a linked shard reports its *target's*
    /// size and block allocation (never the ~50 B link stub), returning
    /// `path → RemoteStat`. Best-effort: a path that can't be resolved (broken
    /// link, no GNU `stat`) is simply absent, and the exit status is ignored so one
    /// bad path doesn't drop the rest.
    pub fn stat_paths(&self, paths: &[String]) -> HashMap<String, RemoteStat> {
        let mut out = HashMap::new();
        if paths.is_empty() {
            return out;
        }
        let args: String = paths
            .iter()
            .map(|p| format!(" {}", shell_single_quote(p)))
            .collect();
        // `-L` follows links; `%s` size · `%b` blocks · `%B` block-size · `%F` type
        // · `%n` the path as given (tab-separated so a spaced type like "regular
        // file" stays one field, and a path with spaces is the whole last field).
        let command = format!("stat -L -c '%s\t%b\t%B\t%F\t%n' --{args}");
        let _ = self.exec_capture(&command, "", |line| {
            if let Some((name, st)) = parse_stat_line(line) {
                out.insert(name, st);
            }
        });
        out
    }

    /// Read a remote safetensors header over SFTP together with the file's total
    /// size — the `(total_len, header_json)` a remote [`crate::safelayout`] map
    /// needs — without touching the tensor data. Read-only.
    pub fn read_header_sized(&self, path: &str) -> Result<(u64, Vec<u8>)> {
        let sftp = self.session.sftp().context("opening the SFTP subsystem")?;
        read_header_sized(&sftp, path)
    }

    /// The on-disk footprint of each path — `(path, apparent, allocated)` — for the
    /// stats "on disk" section, showing remote **ZFS/btrfs compression or sparse
    /// holes** (SFTP carries no block count). Derived from [`Self::stat_paths`],
    /// the one source of truth for remote file sizes, so it **follows symlinks**:
    /// a linked shard reports its target's real footprint, agreeing with the file
    /// browser and layout map (rather than the ~50 B link stub). Paths that don't
    /// resolve are omitted (treated as "unknown"), preserving the input order.
    pub fn allocated_sizes(&self, paths: &[String]) -> Result<Vec<(String, u64, u64)>> {
        let stats = self.stat_paths(paths);
        Ok(paths
            .iter()
            .filter_map(|p| {
                stats
                    .get(p)
                    .map(|st| (p.clone(), st.apparent(), st.allocated()))
            })
            .collect())
    }

    /// Claim shards from the shared `next` counter and read+parse each header over
    /// one SFTP channel, returning `(index, tensors, metadata)` per shard claimed.
    /// Several sessions sharing one `next` split the shards dynamically
    /// (work-stealing): each grabs the next unread shard with a single atomic
    /// increment, so a session that finishes early takes on more and a slow one
    /// doesn't hold up the rest. `displays[idx]` is the `source_path` stamped on the
    /// tensors of `files[idx]`.
    #[allow(clippy::type_complexity)]
    pub fn read_shards(
        &self,
        files: &[String],
        displays: &[String],
        next: &std::sync::atomic::AtomicUsize,
        progress: Option<&crate::progress::LoadProgress>,
    ) -> Result<Vec<(usize, Vec<TensorInfo>, Vec<MetadataInfo>)>> {
        use std::sync::atomic::Ordering;
        let sftp = self.session.sftp().context("opening the SFTP subsystem")?;
        let mut out = Vec::new();
        loop {
            let idx = next.fetch_add(1, Ordering::Relaxed);
            if idx >= files.len() {
                break;
            }
            let header = read_header(&sftp, &files[idx])?;
            let (ts, ms) = crate::stheader::parse_header(&header, &displays[idx])?;
            out.push((idx, ts, ms));
            if let Some(p) = progress {
                p.advance();
            }
        }
        Ok(out)
    }
}

/// Split `[user@]host[:port]` into its parts, defaulting the user to `$USER` and
/// the port to 22. A trailing `:NNNN` is only treated as a port when it parses as
/// a port number (so bare hostnames pass through untouched).
fn parse_target(target: &str) -> (String, String, u16) {
    let (user, rest) = match target.split_once('@') {
        Some((u, r)) => (u.to_string(), r),
        None => (default_user(), target),
    };
    let (host, port) = match rest.rsplit_once(':') {
        Some((h, p)) => match p.parse::<u16>() {
            Ok(n) => (h.to_string(), n),
            Err(_) => (rest.to_string(), 22),
        },
        None => (rest.to_string(), 22),
    };
    (user, host, port)
}

fn default_user() -> String {
    std::env::var("USER")
        .or_else(|_| std::env::var("LOGNAME"))
        .unwrap_or_else(|_| "root".to_string())
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/"))
}

/// Verify the presented host key against `~/.ssh/known_hosts`. Refuses to proceed
/// on a mismatch (possible MITM) or when the host is unknown — no blind
/// trust-on-first-use — so this stays as strict as the `ssh` client.
fn verify_host_key(session: &Session, host: &str, port: u16) -> Result<()> {
    let mut known = session.known_hosts().context("reading known hosts")?;
    // Missing/unreadable file is fine — we still reject NotFound below.
    let _ = known.read_file(
        &home_dir().join(".ssh/known_hosts"),
        KnownHostFileKind::OpenSSH,
    );
    let (key, _kind) = session
        .host_key()
        .ok_or_else(|| anyhow!("{host} presented no host key"))?;
    match known.check_port(host, port, key) {
        CheckResult::Match => Ok(()),
        CheckResult::Mismatch => bail!(
            "host key for {host} does NOT match ~/.ssh/known_hosts — refusing to connect \
             (possible man-in-the-middle). If the host legitimately changed, run \
             `ssh-keygen -R {host}` and reconnect with `ssh {host}` once."
        ),
        CheckResult::NotFound => bail!(
            "host key for {host} is not in ~/.ssh/known_hosts. Connect once with `ssh {host}` \
             to record it, then retry."
        ),
        CheckResult::Failure => bail!("could not check the host key for {host}"),
    }
}

/// TCP-connect, SSH-handshake, and verify the host key — a fresh, unauthenticated
/// session ready for one authentication attempt.
fn handshake(host: &str, port: u16) -> Result<Session> {
    let tcp =
        TcpStream::connect((host, port)).with_context(|| format!("connecting to {host}:{port}"))?;
    let mut session = Session::new().context("initialising the SSH session")?;
    session.set_tcp_stream(tcp);
    session.handshake().context("SSH handshake failed")?;
    verify_host_key(&session, host, port)?;
    Ok(session)
}

/// Authenticate the session: SSH agent, then unencrypted default identity files,
/// then a single password method — `password` auth if the server offers it, else
/// keyboard-interactive (covering plain passwords and 2FA). Exactly one password
/// method is tried per session: a rejected `userauth_password` taints the session,
/// so falling through to keyboard-interactive on it would authenticate a
/// connection that then can't run a channel. A wrong password fails here; the
/// caller retries on a fresh connection. A password entered here is cached in
/// `password` and reused on a subsequent call for the same host (no second prompt).
fn authenticate(
    session: &Session,
    user: &str,
    host: &str,
    password: &mut Option<String>,
) -> Result<()> {
    if session.userauth_agent(user).is_ok() && session.authenticated() {
        return Ok(());
    }
    for key in default_keys() {
        if session.userauth_pubkey_file(user, None, &key, None).is_ok() && session.authenticated() {
            return Ok(());
        }
    }
    let methods = session
        .auth_methods(user)
        .unwrap_or("password,keyboard-interactive");
    if methods.contains("password") {
        let pw = match password {
            Some(p) => p.clone(),
            None => {
                let pw = rpassword::prompt_password(format!("{user}@{host}'s password: "))
                    .context("reading password")?;
                reset_prompt_column();
                pw
            }
        };
        if session.userauth_password(user, &pw).is_ok() && session.authenticated() {
            *password = Some(pw);
            return Ok(());
        }
        bail!("authentication to {host} failed");
    }
    if methods.contains("keyboard-interactive") {
        let mut prompter = Prompter { password };
        if session
            .userauth_keyboard_interactive(user, &mut prompter)
            .is_ok()
            && session.authenticated()
        {
            return Ok(());
        }
        bail!("authentication to {host} failed");
    }
    bail!("no supported authentication method for {host} (server offers: {methods})");
}

/// The default identity files to try (unencrypted), most-preferred first.
fn default_keys() -> Vec<PathBuf> {
    let ssh = home_dir().join(".ssh");
    ["id_ed25519", "id_ecdsa", "id_rsa"]
        .iter()
        .map(|n| ssh.join(n))
        .filter(|p| p.exists())
        .collect()
}

/// Answers the server's keyboard-interactive prompts from the terminal: hidden
/// input for password-style prompts (reusing/caching `password` so a second
/// connection doesn't prompt again), echoed input for anything the server wants
/// shown (e.g. a visible OTP prompt).
struct Prompter<'a> {
    password: &'a mut Option<String>,
}

impl ssh2::KeyboardInteractivePrompt for Prompter<'_> {
    fn prompt<'a>(
        &mut self,
        _user: &str,
        instructions: &str,
        prompts: &[ssh2::Prompt<'a>],
    ) -> Vec<String> {
        if !instructions.trim().is_empty() {
            eprintln!("{}", instructions.trim());
        }
        prompts
            .iter()
            .map(|p| {
                if p.echo {
                    eprint!("{}", p.text);
                    let mut line = String::new();
                    let _ = std::io::stdin().read_line(&mut line);
                    line.trim_end_matches(['\n', '\r']).to_string()
                } else if let Some(cached) = self.password.as_ref() {
                    cached.clone()
                } else {
                    let entered = rpassword::prompt_password(p.text.as_ref()).unwrap_or_default();
                    reset_prompt_column();
                    *self.password = Some(entered.clone());
                    entered
                }
            })
            .collect()
    }
}

/// Enumerate the `.safetensors` shards of `path` (a directory or a single file):
/// the index's `weight_map` order first, then every other `*.safetensors` in the
/// directory (covers no-index dirs / extra shards like codebooks), deduped.
///
/// The directory listing is the source of truth for what exists — a stale
/// `model.safetensors.index.json` (e.g. a re-sharded checkpoint that kept an old
/// index, so `weight_map` names shards that were renamed away) must not make us
/// try to open files that aren't there. So index entries are kept only when the
/// listing confirms them; when the directory can't be listed at all, the index is
/// trusted as the only signal we have.
fn list_shards(sftp: &ssh2::Sftp, path: &str) -> Result<ShardListing> {
    if path.ends_with(".safetensors") {
        return Ok(ShardListing {
            files: vec![path.to_string()],
            index_path: None,
            weight_map: HashMap::new(),
            actual: BTreeSet::new(),
        });
    }
    let base = path.trim_end_matches('/');

    // The `.safetensors` files actually present — full paths (for the read order,
    // sorted) and basenames (for the health check). The listing is the truth.
    let (on_disk, actual, listed) = match sftp.readdir(Path::new(base)) {
        Ok(entries) => {
            let names: Vec<String> = entries
                .iter()
                .filter_map(|(p, _)| p.file_name().and_then(|n| n.to_str()))
                .filter(|n| n.ends_with(".safetensors"))
                .map(str::to_string)
                .collect();
            let mut full: Vec<String> = names.iter().map(|n| format!("{base}/{n}")).collect();
            full.sort();
            (full, names.into_iter().collect::<BTreeSet<_>>(), true)
        }
        Err(_) => (Vec::new(), BTreeSet::new(), false),
    };

    // The index gives the preferred shard order and the health check's weight_map.
    let mut indexed: Vec<String> = Vec::new();
    let mut weight_map: HashMap<String, String> = HashMap::new();
    let mut index_path = None;
    let index = format!("{base}/model.safetensors.index.json");
    if let Ok(bytes) = read_all(sftp, &index)
        && let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes)
        && let Some(wm) = v.get("weight_map").and_then(|w| w.as_object())
    {
        index_path = Some(index);
        for (tensor, file) in wm {
            if let Some(f) = file.as_str() {
                weight_map.insert(tensor.clone(), f.to_string());
            }
        }
        let mut shards: Vec<&str> = wm.values().filter_map(|s| s.as_str()).collect();
        shards.sort_unstable();
        shards.dedup();
        indexed = shards.into_iter().map(|s| format!("{base}/{s}")).collect();
    }
    Ok(ShardListing {
        files: merge_shard_lists(&indexed, &on_disk, listed),
        index_path,
        weight_map,
        actual,
    })
}

/// Combine the index's shard order with the directory listing: keep index
/// entries the listing confirms (or all of them, if the directory couldn't be
/// listed — the index is then all we have), then append every other present
/// `.safetensors`. Pure, so the stale-index handling is unit-tested without a
/// live SFTP server.
fn merge_shard_lists(indexed: &[String], on_disk: &[String], listed: bool) -> Vec<String> {
    let mut files: Vec<String> = Vec::new();
    for s in indexed {
        if (!listed || on_disk.contains(s)) && !files.contains(s) {
            files.push(s.clone());
        }
    }
    for f in on_disk {
        if !files.contains(f) {
            files.push(f.clone());
        }
    }
    files
}

/// After an interactive password prompt, return the cursor to column 0.
///
/// `rpassword` ends the prompt by writing a bare line-feed as it restores the
/// terminal — with `OPOST`/`ONLCR` off, that drops the cursor to the next row but
/// keeps the prompt's *column*. The following "reading …" notice would then print
/// indented under the prompt instead of at the left margin (the progress-bar
/// frames start with their own `\r`, so only the notice is affected). A carriage
/// return fixes it; it's a no-op when the prompt already returned to column 0.
/// Only on a terminal, so a redirected stderr log gets no stray control byte.
fn reset_prompt_column() {
    if std::io::stderr().is_terminal() {
        let mut err = std::io::stderr();
        let _ = err.write_all(b"\r");
        let _ = err.flush();
    }
}

/// Wrap a string in single quotes for a POSIX shell, escaping any embedded
/// single quote as `'\''` — so a path with spaces or shell metacharacters is
/// passed to the remote `stat` as one literal argument.
fn shell_single_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// Open a remote file **read-only**. This is the *only* way this module opens a
/// file, so the entire remote path is structurally incapable of creating,
/// truncating, or writing anything on the server: `OpenFlags::READ` carries no
/// write/create/append/truncate bit (adding one would have to be an explicit,
/// reviewable edit here). The `mode` argument only applies to file *creation*,
/// which READ never triggers.
fn open_readonly(sftp: &ssh2::Sftp, path: &str) -> Result<ssh2::File> {
    sftp.open_mode(
        Path::new(path),
        ssh2::OpenFlags::READ,
        0,
        ssh2::OpenType::File,
    )
    .with_context(|| format!("opening {path}"))
}

/// Read a shard's safetensors header over SFTP — the 8-byte little-endian length
/// then that many JSON bytes — without touching the tensor data that follows.
fn read_header(sftp: &ssh2::Sftp, path: &str) -> Result<Vec<u8>> {
    let mut f = open_readonly(sftp, path)?;
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf)
        .with_context(|| format!("reading header length of {path}"))?;
    let n = crate::stheader::header_len(u64::from_le_bytes(len_buf), path)?;
    let mut header = vec![0u8; n];
    f.read_exact(&mut header)
        .with_context(|| format!("reading header of {path}"))?;
    Ok(header)
}

/// Parse one `stat -L -c '%s\t%b\t%B\t%F\t%n'` line into `(path, RemoteStat)`:
/// size · blocks · block-size · type · path. The path (`%n`) is the whole last
/// field, so a path containing spaces (or the type's own space, "regular file")
/// survives. Pure, so the format is unit-tested without a live SFTP server.
fn parse_stat_line(line: &str) -> Option<(String, RemoteStat)> {
    let mut it = line.splitn(5, '\t');
    let (s, b, bs, kind, name) = (it.next()?, it.next()?, it.next()?, it.next()?, it.next()?);
    let apparent = s.trim().parse::<u64>().ok()?;
    let blocks = b.trim().parse::<u64>().ok()?;
    let block_size = bs.trim().parse::<u64>().ok()?;
    Some((
        name.to_string(),
        if kind == "directory" {
            RemoteStat::Directory
        } else {
            RemoteStat::File {
                apparent,
                allocated: blocks * block_size,
            }
        },
    ))
}

/// Like [`read_header`], but also returns the file's total size (from a `stat`
/// on the open handle) — the layout map needs it to size the trailing data gap.
fn read_header_sized(sftp: &ssh2::Sftp, path: &str) -> Result<(u64, Vec<u8>)> {
    let mut f = open_readonly(sftp, path)?;
    let total_len = f
        .stat()
        .with_context(|| format!("stat of {path}"))?
        .size
        .unwrap_or(0);
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf)
        .with_context(|| format!("reading header length of {path}"))?;
    let n = crate::stheader::header_len(u64::from_le_bytes(len_buf), path)?;
    let mut header = vec![0u8; n];
    f.read_exact(&mut header)
        .with_context(|| format!("reading header of {path}"))?;
    Ok((total_len, header))
}

/// Read a small remote file (the shard index) in full over SFTP.
fn read_all(sftp: &ssh2::Sftp, path: &str) -> Result<Vec<u8>> {
    let mut f = open_readonly(sftp, path)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)
        .with_context(|| format!("reading {path}"))?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_stat_line_reads_followed_size_and_allocation() {
        // size · blocks · block-size · type · path — a followed (`-L`) regular file.
        let (name, st) = parse_stat_line(
            "6127900160\t11968752\t512\tregular file\t/ckpt/model-00000.safetensors",
        )
        .unwrap();
        assert_eq!(name, "/ckpt/model-00000.safetensors");
        assert_eq!(st.apparent(), 6_127_900_160);
        assert_eq!(st.allocated(), 11_968_752 * 512); // st_blocks × block-size
        assert!(!st.is_dir());

        // A directory target (a followed symlink-to-dir).
        assert!(
            parse_stat_line("4096\t8\t512\tdirectory\t/d")
                .unwrap()
                .1
                .is_dir()
        );
        // The path (last field) keeps embedded spaces intact.
        assert_eq!(
            parse_stat_line("1\t1\t512\tregular file\t/a b/c d.safetensors")
                .unwrap()
                .0,
            "/a b/c d.safetensors"
        );
        // Garbage / short lines are ignored, not panics.
        assert!(parse_stat_line("nope").is_none());
        assert!(parse_stat_line("x\ty\tz\tregular file\t/p").is_none());
    }

    #[test]
    fn parses_targets() {
        assert_eq!(
            parse_target("lab@net004:2222"),
            ("lab".into(), "net004".into(), 2222)
        );
        assert_eq!(parse_target("lab@host"), ("lab".into(), "host".into(), 22));
        // a trailing non-numeric `:segment` is not a port
        assert_eq!(
            parse_target("host:notaport"),
            (default_user(), "host:notaport".into(), 22)
        );
        // bare host → default user + port 22
        let (_, h, p) = parse_target("box");
        assert_eq!((h.as_str(), p), ("box", 22));
    }

    fn v(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn stale_index_entries_are_dropped_for_the_real_files() {
        // The index names shards that were re-sharded away (`…-of-00014`), but the
        // directory holds a different set (`…-of-00073`, plus codebooks). Only the
        // files that actually exist are read.
        let indexed = v(&["d/model-00000-of-00014.safetensors"]);
        let on_disk = v(&[
            "d/codebooks.safetensors",
            "d/model-00001-of-00073.safetensors",
            "d/model-00002-of-00073.safetensors",
        ]);
        let got = merge_shard_lists(&indexed, &on_disk, true);
        assert_eq!(got, on_disk); // bogus index entry gone, order = on-disk
    }

    #[test]
    fn index_order_wins_then_extras_appended() {
        // A correct index fixes the shard order; files it doesn't mention
        // (codebooks) are appended after.
        let indexed = v(&["d/model-00001.safetensors", "d/model-00002.safetensors"]);
        let on_disk = v(&[
            "d/codebooks.safetensors",
            "d/model-00001.safetensors",
            "d/model-00002.safetensors",
        ]);
        let got = merge_shard_lists(&indexed, &on_disk, true);
        assert_eq!(
            got,
            v(&[
                "d/model-00001.safetensors",
                "d/model-00002.safetensors",
                "d/codebooks.safetensors",
            ])
        );
    }

    #[test]
    fn unlistable_dir_trusts_the_index() {
        // If the directory can't be listed, the index is the only signal — keep it.
        let indexed = v(&["d/model-00001.safetensors"]);
        let got = merge_shard_lists(&indexed, &[], false);
        assert_eq!(got, indexed);
    }
}
