//! Recognise `s3://…` URIs. Remote checkpoints are read via [`crate::remote`]
//! (SSH-delegated cstorch), so this is just the scheme check shared by the CLI
//! and the loader — no S3 client of its own.

/// The remote-checkpoint URI scheme.
const SCHEME: &str = "s3://";

/// Is this CLI argument an `s3://…` URI?
pub fn is_uri(s: &str) -> bool {
    s.starts_with(SCHEME)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_s3_uris() {
        assert!(is_uri("s3://bucket/key"));
        assert!(is_uri("s3://bucket/prefix/"));
        assert!(!is_uri("model.safetensors"));
        assert!(!is_uri("/local/path"));
    }
}
