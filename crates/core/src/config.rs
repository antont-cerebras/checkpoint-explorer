//! The subset of a Hugging Face `config.json` the `check` subcommand
//! cross-checks against the tensor tree (see `check::check_config`). Loaded from
//! next to a local checkpoint, or fetched over SFTP for a remote one.

use std::path::PathBuf;

/// The architecture fields we validate against the tensors. All optional — a
/// config that omits a field simply skips that check.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModelConfig {
    pub model_type: Option<String>,
    pub num_hidden_layers: Option<u64>,
    pub num_experts: Option<u64>,
    pub vocab_size: Option<u64>,
    pub hidden_size: Option<u64>,
    pub tie_word_embeddings: Option<bool>,
    pub use_qk_norm: Option<bool>,
}

impl ModelConfig {
    /// Parse the fields we care about from a `config.json` string. `None` only
    /// when the text isn't a JSON object at all; missing keys stay `None`.
    pub fn parse(json: &str) -> Option<ModelConfig> {
        let value: serde_json::Value = serde_json::from_str(json).ok()?;
        let obj = value.as_object()?;
        let uint = |k: &str| obj.get(k).and_then(serde_json::Value::as_u64);
        let boolean = |k: &str| obj.get(k).and_then(serde_json::Value::as_bool);
        Some(ModelConfig {
            model_type: obj
                .get("model_type")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            num_hidden_layers: uint("num_hidden_layers"),
            num_experts: uint("num_experts"),
            vocab_size: uint("vocab_size"),
            hidden_size: uint("hidden_size"),
            tie_word_embeddings: boolean("tie_word_embeddings"),
            use_qk_norm: boolean("use_qk_norm"),
        })
    }

    /// Whether at least one checkable field is present — so an unrelated JSON
    /// object sitting next to the weights isn't mistaken for a model config.
    pub fn is_meaningful(&self) -> bool {
        self.num_hidden_layers.is_some()
            || self.num_experts.is_some()
            || self.vocab_size.is_some()
            || self.tie_word_embeddings.is_some()
            || self.use_qk_norm.is_some()
    }
}

/// The `config.json` beside a local checkpoint — in the parent dir of its files
/// (shards share one dir), or in the dir itself when a directory was given.
/// `None` when there's no such file.
pub fn local_path(files: &[PathBuf]) -> Option<PathBuf> {
    let first = files.first()?;
    let dir = if first.is_dir() {
        first.clone()
    } else {
        first.parent()?.to_path_buf()
    };
    let path = dir.join("config.json");
    path.is_file().then_some(path)
}

/// Load + parse `config.json` for a local checkpoint, when present and it looks
/// like a real model config.
pub fn load_local(files: &[PathBuf]) -> Option<ModelConfig> {
    let text = std::fs::read_to_string(local_path(files)?).ok()?;
    ModelConfig::parse(&text).filter(ModelConfig::is_meaningful)
}

/// The remote `config.json` path for a checkpoint source: the checkpoint's
/// directory (a remote safetensors dir, or the parent of a single `.safetensors`
/// file) plus `config.json`. `None` for an `s3://` cstorch checkpoint, which has
/// no Hugging Face `config.json`.
pub fn remote_path(src: &str) -> Option<String> {
    if src.starts_with("s3://") {
        return None;
    }
    let dir = match src.rsplit_once('/') {
        // A trailing `*.safetensors` names a file — its parent dir holds the config.
        Some((dir, last)) if last.ends_with(".safetensors") => dir,
        // Otherwise `src` is the checkpoint directory itself.
        _ => src.trim_end_matches('/'),
    };
    Some(format!("{dir}/config.json"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_fields_we_check() {
        let cfg = ModelConfig::parse(
            r#"{"model_type":"qwen3_moe","num_hidden_layers":48,"num_experts":128,
                "vocab_size":151936,"hidden_size":2048,"tie_word_embeddings":false,
                "use_qk_norm":true,"unrelated":"ignored"}"#,
        )
        .unwrap();
        assert_eq!(cfg.model_type.as_deref(), Some("qwen3_moe"));
        assert_eq!(cfg.num_hidden_layers, Some(48));
        assert_eq!(cfg.num_experts, Some(128));
        assert_eq!(cfg.tie_word_embeddings, Some(false));
        assert_eq!(cfg.use_qk_norm, Some(true));
        assert!(cfg.is_meaningful());
    }

    #[test]
    fn non_config_json_is_not_meaningful() {
        assert!(!ModelConfig::parse(r#"{"foo":1}"#).unwrap().is_meaningful());
        assert!(ModelConfig::parse("not json").is_none());
    }

    #[test]
    fn remote_path_handles_dirs_files_and_s3() {
        assert_eq!(
            remote_path("/ckpts/qwen").as_deref(),
            Some("/ckpts/qwen/config.json")
        );
        assert_eq!(
            remote_path("/ckpts/qwen/model-00001-of-00003.safetensors").as_deref(),
            Some("/ckpts/qwen/config.json")
        );
        assert_eq!(remote_path("s3://bucket/ckpt"), None);
    }
}
