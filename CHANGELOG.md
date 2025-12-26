# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2025-12-26

### Breaking Changes
- **Binary renamed** from `rmcp_memex` to `rmcp-memex` (hyphenated).
- **Default paths changed** from `~/.rmcp_servers/rmcp_memex/` to `~/.rmcp-servers/rmcp-memex/`.
- **MCP server name** changed from `rmcp_memex` to `rmcp-memex`.

### Added
- **Namespace Security** - Token-based access control for protected namespaces.
  - `namespace_create_token` - Create access token for namespace
  - `namespace_revoke_token` - Revoke token (namespace becomes public)
  - `namespace_list_protected` - List protected namespaces
  - `namespace_security_status` - Security system status
- **Preprocessing Module** - Automatic noise filtering from conversation exports (~36-40% reduction).
  - Filters MCP tool artifacts (`<function_calls>`, `<invoke>`, etc.)
  - Removes CLI output (git status, cargo build, npm install)
  - Replaces metadata (UUIDs, timestamps) with placeholders
- **Onion Slice Architecture** - Hierarchical embeddings for better navigation.
  - OUTER (~100 chars) - Keywords + ultra-compression
  - MIDDLE (~300 chars) - Key sentences + context
  - INNER (~600 chars) - Expanded content
  - CORE (full text) - Complete document
- **Exact-Match Deduplication** - SHA256-based dedup for overlapping exports.
- **TUI Configuration Wizard** - Interactive setup for host configurations.
- **Comprehensive Documentation**:
  - `docs/01_security.md` - Namespace security guide
  - `docs/02_configuration.md` - CLI options and TOML config

### Changed
- Schema version bumped to v3 (added `content_hash` field).
- Embeddings configuration now uses `EmbeddingConfig` with multi-provider support.
- Updated to Rust Edition 2024 (stable).

### Fixed
- Consistent naming across package, binary, MCP server, and paths.

## [0.1.13] - 2025-12-25

### Added
- Initial standalone release synced from loctree-suite.
- LanceDB vector storage with 4096-dim embeddings.
- RAG tools: `rag_index`, `rag_index_text`, `rag_search`.
- Memory tools: `memory_upsert`, `memory_get`, `memory_search`, `memory_delete`, `memory_purge_namespace`.
- Multi-provider embeddings support (Ollama, MLX, OpenAI-compatible).

---

Created by M&K (c)2025 The LibraxisAI Team
