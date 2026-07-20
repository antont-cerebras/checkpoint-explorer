//! Shared tensor-name filtering. A filter is a set of glob patterns; any pattern
//! may be **negated** with a leading `!` to *exclude*. The same rules back
//! `diff --name` and the `--print-tree` / `--print-tensors` `--name` option, so a
//! pattern behaves identically wherever it's accepted.

use anyhow::{Context, Result};
use glob::Pattern;

/// A name filter: a name passes if it matches some **include** glob (or there are
/// none) and matches **no exclude** glob. An empty filter matches everything.
#[derive(Default, Clone)]
pub struct NameFilter {
    pub include: Vec<Pattern>,
    pub exclude: Vec<Pattern>,
}

impl NameFilter {
    /// Parse repeated `--name` values: a leading `!` marks a pattern as an
    /// exclude ("everything except …"), any other value is an include. Globs use
    /// the standard `*` / `?` / `[…]`; a bad glob is an error.
    pub fn parse(patterns: &[String]) -> Result<NameFilter> {
        let mut filter = NameFilter::default();
        for pattern in patterns {
            let (bucket, glob) = match pattern.strip_prefix('!') {
                Some(rest) => (&mut filter.exclude, rest),
                None => (&mut filter.include, pattern.as_str()),
            };
            bucket
                .push(Pattern::new(glob).with_context(|| format!("invalid --name glob {glob:?}"))?);
        }
        Ok(filter)
    }

    /// Whether the filter constrains anything (so callers can skip work / drop
    /// metadata when it doesn't).
    pub fn is_active(&self) -> bool {
        !self.include.is_empty() || !self.exclude.is_empty()
    }

    /// Whether `name` passes: it matches at least one include (or there are no
    /// includes) and matches none of the excludes.
    pub fn matches(&self, name: &str) -> bool {
        if !self.include.is_empty() && !self.include.iter().any(|p| p.matches(name)) {
            return false;
        }
        !self.exclude.iter().any(|p| p.matches(name))
    }
}

#[cfg(test)]
mod tests {
    use super::NameFilter;

    #[test]
    fn empty_filter_matches_everything() {
        let f = NameFilter::parse(&[]).unwrap();
        assert!(!f.is_active());
        assert!(f.matches("anything.at.all"));
    }

    #[test]
    fn include_globs_match_any() {
        let f = NameFilter::parse(&["*.mlp.*".into(), "*.norm.weight".into()]).unwrap();
        assert!(f.is_active());
        assert!(f.matches("model.layers.0.mlp.down_proj.weight"));
        assert!(f.matches("model.norm.weight"));
        assert!(!f.matches("model.embed_tokens.weight"));
    }

    #[test]
    fn bare_exclude_is_all_except() {
        let f = NameFilter::parse(&["!*.bias".into()]).unwrap();
        assert!(f.is_active());
        assert!(f.matches("model.layers.0.mlp.down_proj.weight")); // kept
        assert!(!f.matches("model.layers.0.mlp.down_proj.bias")); // excluded
    }

    #[test]
    fn include_minus_exclude() {
        let f = NameFilter::parse(&["*.weight".into(), "!*.norm.weight".into()]).unwrap();
        assert!(f.matches("model.layers.0.mlp.down_proj.weight"));
        assert!(!f.matches("model.norm.weight")); // matches include but excluded
        assert!(!f.matches("model.layers.0.mlp.down_proj.bias")); // no include match
    }
}
