//! Read a remote checkpoint's **structure** (tensor names, dtypes, shapes) by
//! delegating to a machine that already has access to it — over SSH, running a
//! small `cerebras.pytorch` (cstorch) script that opens the checkpoint lazily and
//! dumps metadata as JSON. Credentials stay on the remote (nothing is copied
//! locally), nothing is installed there beyond the existing cstorch venv +
//! `python3`, and it works for any URI `cstorch.load` accepts (S3/MinIO cstorch
//! checkpoints included). Browse-only: no tensor data crosses the wire.

use std::io::Write;
use std::process::{Command, Stdio};

use anyhow::{Context, Result, anyhow, bail};

use crate::tree::{Layout, MetadataInfo, Storage, TensorInfo};

/// Line prefix the remote script tags its JSON with, so we can pick it out of any
/// motd / cstorch chatter on the SSH stdout.
const SENTINEL: &str = "CKPT_EXPLORER_META:";

/// A remote host + cstorch venv to read checkpoint metadata through (`--ssh-read`
/// / `--ssh-venv`).
#[derive(Clone, Debug)]
pub struct RemoteRead {
    pub host: String,
    pub venv: String,
}

/// SSH options that reuse one authenticated connection across calls (so an
/// up-front auth, then each read, all share a single login — the password/2FA
/// prompt happens once, and reads can show a spinner without fighting it).
const CONTROL_ARGS: [&str; 6] = [
    "-o",
    "ControlMaster=auto",
    "-o",
    "ControlPath=/tmp/ckpt-explorer-ssh-%C",
    "-o",
    "ControlPersist=60",
];

impl RemoteRead {
    pub fn new(host: String, venv: String) -> Self {
        RemoteRead { host, venv }
    }

    /// Open (and authenticate) the shared SSH master connection up front. Any
    /// password/2FA prompt happens here, on the normal terminal, before any
    /// spinner; subsequent [`Self::fetch`] calls reuse it without prompting.
    /// Best-effort — if the master can't be established (e.g. multiplexing is
    /// disabled), the reads simply authenticate themselves.
    pub fn connect(&self) -> Result<()> {
        eprintln!("checkpoint-explorer: connecting to {} via ssh …", self.host);
        let status = Command::new("ssh")
            .args(CONTROL_ARGS)
            .arg("-N") // no remote command …
            .arg("-f") // … background the master once authenticated
            .arg(&self.host)
            .status()
            .with_context(|| format!("spawning `ssh {}`", self.host))?;
        if !status.success() {
            bail!(
                "ssh connection to {} failed (exit {:?})",
                self.host,
                status.code()
            );
        }
        Ok(())
    }

    /// SSH to the host, activate the cstorch venv, run the dump script, and parse
    /// the returned metadata into tensors. `uri` is whatever `cstorch.load`
    /// accepts (e.g. an `s3://…` cstorch checkpoint). Tensor data is never read.
    pub fn fetch(&self, uri: &str) -> Result<(Vec<TensorInfo>, Vec<MetadataInfo>)> {
        let script = dump_script(uri);
        let remote_cmd = format!("source {}/bin/activate && exec python3 -", self.venv);
        let mut child = Command::new("ssh")
            .args(CONTROL_ARGS)
            .arg(&self.host)
            .arg(&remote_cmd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit()) // show ssh auth prompts + cstorch errors live
            .spawn()
            .with_context(|| format!("spawning `ssh {}`", self.host))?;
        child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("ssh stdin unavailable"))?
            .write_all(script.as_bytes())
            .context("sending the metadata script to the remote")?;
        let out = child.wait_with_output().context("waiting for ssh")?;
        if !out.status.success() {
            bail!(
                "remote read via {} failed (ssh exit {:?}) — see the error above",
                self.host,
                out.status.code()
            );
        }
        let stdout = String::from_utf8_lossy(&out.stdout);
        let json = stdout
            .lines()
            .rev()
            .find_map(|l| l.strip_prefix(SENTINEL))
            .ok_or_else(|| {
                anyhow!(
                    "no metadata returned from {} (see any error above)",
                    self.host
                )
            })?;
        parse_dump(json, uri)
    }
}

/// The self-contained Python dumped over SSH. The URI is embedded as a JSON string
/// literal (valid Python), so there's nothing to quote-escape at the shell.
fn dump_script(uri: &str) -> String {
    let uri_lit = serde_json::to_string(uri).unwrap_or_else(|_| "\"\"".into());
    const TEMPLATE: &str = r#"
import sys, json
URI = __URI__
S = "__SENTINEL__"
def emit(obj):
    sys.stdout.write(S + json.dumps(obj) + "\n")
    sys.stdout.flush()
try:
    import cerebras.pytorch as cstorch
except Exception as e:
    emit({"error": "import cerebras.pytorch failed: %r" % (e,)}); sys.exit(0)
try:
    sd = cstorch.load(URI, map_location=None)   # lazy: metadata only, no data
except Exception as e:
    emit({"error": "cstorch.load failed: %r" % (e,)}); sys.exit(0)
try:
    keys = list(sd.keys())
except Exception as e:
    emit({"error": "listing keys failed: %r" % (e,)}); sys.exit(0)
tensors, meta = [], []
for name in keys:
    try:
        t = sd[name]
        shape = [int(d) for d in t.shape]
        dtype = str(getattr(t, "dtype", ""))
        try:
            itemsize = int(t.element_size())
        except Exception:
            itemsize = 0
        tensors.append({"name": str(name), "dtype": dtype, "shape": shape, "itemsize": itemsize})
    except Exception:
        meta.append({"name": str(name)})
emit({"tensors": tensors, "metadata": meta})
"#;
    TEMPLATE
        .replace("__URI__", &uri_lit)
        .replace("__SENTINEL__", SENTINEL)
}

/// Parse the remote JSON (`{tensors:[…], metadata:[…]}` or `{error:…}`) into
/// [`TensorInfo`]s whose `source_path` is the remote URI (so display and the `y`
/// command round-trip, and the data views correctly report themselves remote).
fn parse_dump(json: &str, uri: &str) -> Result<(Vec<TensorInfo>, Vec<MetadataInfo>)> {
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
                source_path: uri.to_string(),
                layout: Layout::None,
            });
        }
    }
    if tensors.is_empty() {
        bail!("the remote returned no tensors for {uri}");
    }
    Ok((tensors, Vec::new()))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn script_embeds_uri_safely() {
        let s = dump_script("s3://b/k");
        assert!(s.contains("URI = \"s3://b/k\""));
        assert!(s.contains("import cerebras.pytorch"));
        assert!(s.contains(SENTINEL));
    }

    #[test]
    fn parses_dump_into_tensors() {
        let json = r#"{"tensors":[
            {"name":"a.weight","dtype":"torch.float16","shape":[6,4],"itemsize":2},
            {"name":"b","dtype":"torch.int32","shape":[5],"itemsize":4}
        ],"metadata":[]}"#;
        let (t, _m) = parse_dump(json, "s3://bucket/ckpt").unwrap();
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
}
