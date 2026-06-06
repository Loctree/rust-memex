//! Shared helpers for e2e integration tests.
//!
//! Tests load the operator's canonical `~/.rmcp-servers/rust-memex/config.toml`
//! (or a path from `RUST_MEMEX_CONFIG`) and use its real provider cascade.
//! There is no synthetic fallback — e2e must exercise the runtime config path
//! end-to-end. If no config exists, tests fail-fast with an operator-actionable
//! message.

#![allow(dead_code)] // helpers are only used under `--features e2e-ollama`

use anyhow::{Context, Result, anyhow};
use rust_memex::EmbeddingConfig;
use serde::Deserialize;
use std::path::PathBuf;

/// Mirrors `bin/cli/config.rs::CONFIG_SEARCH_PATHS`. Keep in sync.
const CONFIG_SEARCH_PATHS: &[&str] = &[
    "~/.rmcp-servers/rust-memex/config.toml",
    "~/.config/rust-memex/config.toml",
];

/// Minimal mirror of `FileConfig` — only fields e2e tests need. Avoids
/// pulling the binary's `FileConfig` (which is `bin/cli/config.rs`-private)
/// into integration tests.
#[derive(Debug, Deserialize)]
struct FileConfigShim {
    #[serde(default)]
    db_path: Option<String>,
    #[serde(default)]
    embeddings: Option<EmbeddingConfig>,
}

#[derive(Debug, Clone)]
pub struct E2eConfig {
    pub source_path: PathBuf,
    pub db_path: Option<String>,
    pub embeddings: EmbeddingConfig,
}

/// Load operator's canonical rust-memex config.
///
/// Resolution order (mirrors runtime `discover_config`):
/// 1. `RUST_MEMEX_CONFIG` env var
/// 2. CONFIG_SEARCH_PATHS standard locations
///
/// Returns `Err` (not `Ok` with synthetic defaults) when nothing exists. The
/// e2e contract is "exercise the real config path"; a synthetic default would
/// silently mask broken config discovery.
pub fn load_e2e_config() -> Result<E2eConfig> {
    let path = resolve_config_path().ok_or_else(|| {
        anyhow!(
            "e2e tests need a real rust-memex config.toml. Set RUST_MEMEX_CONFIG \
             or place one at ~/.rmcp-servers/rust-memex/config.toml. Searched: \
             RUST_MEMEX_CONFIG env, {:?}",
            CONFIG_SEARCH_PATHS
        )
    })?;
    let contents = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read e2e config at {}", path.display()))?;
    let shim: FileConfigShim = toml::from_str(&contents)
        .with_context(|| format!("Failed to parse e2e config at {}", path.display()))?;
    let embeddings = shim.embeddings.ok_or_else(|| {
        anyhow!(
            "Config at {} has no [embeddings] section — e2e tests need a real \
             provider cascade.",
            path.display()
        )
    })?;
    if embeddings.providers.is_empty() {
        return Err(anyhow!(
            "Config at {} declares [embeddings] without any providers — \
             e2e tests need at least one [[embeddings.providers]].",
            path.display()
        ));
    }
    Ok(E2eConfig {
        source_path: path,
        db_path: shim.db_path,
        embeddings,
    })
}

fn resolve_config_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("RUST_MEMEX_CONFIG") {
        let pb = expand_tilde(&p);
        if pb.exists() {
            return Some(pb);
        }
    }
    for raw in CONFIG_SEARCH_PATHS {
        let pb = expand_tilde(raw);
        if pb.exists() {
            return Some(pb);
        }
    }
    None
}

fn expand_tilde(s: &str) -> PathBuf {
    if let Some(rest) = s.strip_prefix("~/")
        && let Ok(home) = std::env::var("HOME")
    {
        return PathBuf::from(home).join(rest);
    }
    PathBuf::from(s)
}
