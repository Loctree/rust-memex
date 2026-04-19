//! Namespace security configuration.
//!
//! Historically this module hosted `NamespaceAccessManager` + `TokenStore`: a
//! single-token-per-namespace auth layer with a bespoke on-disk schema. That
//! implementation was superseded by [`crate::auth::AuthManager`], which
//! provides per-token scopes, namespace ACLs, argon2id-hashed storage, and
//! rotation.
//!
//! What remains here is the runtime-configuration struct consumed by
//! [`crate::ServerConfig`]. The rest of the legacy surface was deleted along
//! with the Track C migration (v0.6.0). If you are looking for "how do I
//! check access to a namespace?" — go to `crate::auth::AuthManager`.
//!
//! Vibecrafted with AI Agents by Loctree (c)2024-2026 The LibraxisAI Team

use serde::{Deserialize, Serialize};

/// Configuration for namespace security (token-based access control).
///
/// Preserved as a config DTO so CLI/file-config surfaces keep working. The
/// actual enforcement now lives in [`crate::auth::AuthManager`]; this struct
/// only tells the runtime *whether* to spin up an auth manager and where the
/// token store lives on disk.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NamespaceSecurityConfig {
    /// Whether token-based access control is enabled.
    ///
    /// When `false`, the runtime wires up an `AuthManager` with an empty
    /// store and no legacy token — every request is allowed.
    #[serde(default)]
    pub enabled: bool,
    /// Path to the token store file (`tokens.json`, v2 schema).
    ///
    /// Defaults to `~/.rmcp-servers/rust-memex/tokens.json` when `enabled`
    /// is set but no path is configured.
    #[serde(default)]
    pub token_store_path: Option<String>,
}
