# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0] - 2025-12-28

### Added
- **MemexEngine** - High-level API for library consumers
  - `MemexEngine::for_app()` - quick setup for any application with auto-config
  - `MemexEngine::for_vista()` - Vista-optimized defaults (1024 dims, qwen3-embedding:0.6b)
  - CRUD operations: `store()`, `search()`, `get()`, `delete()`
  - Batch operations: `store_batch()` for efficient bulk inserts
  - Filtered operations: `search_filtered()`, `delete_by_filter()` for GDPR-compliant deletion
  - Hybrid search: `search_hybrid()` combining BM25 keyword + vector similarity
  - Builder pattern: `MemexConfig` with `with_dimension()`, `with_db_path()`, `with_bm25()`
- **Agent Tools API** - MCP-compatible tool functions for AI agents
  - `memory_store` - Store text with automatic embedding generation
  - `memory_search` - Semantic search with optional metadata filtering
  - `memory_get` - Retrieve document by ID
  - `memory_delete` - Delete document by ID
  - `memory_store_batch` - Efficient batch storage
  - `memory_delete_by_filter` - GDPR deletion by metadata filter
  - `tool_definitions()` - Returns all tool schemas for MCP registration
  - `ToolDefinition` struct matching MCP schema format
  - `ToolResult` struct for consistent tool responses
- **Feature Flags** - Library-only builds without CLI dependencies
  - `default = ["cli", "provider-cascade"]`
  - `cargo build --no-default-features` for minimal library build
  - CLI deps (clap, indicatif, ratatui, crossterm) now optional
- **MetaFilter** - Metadata filtering for searches and deletions
  - `for_patient()` - GDPR-compliant patient data filtering
  - `for_visit()` - Visit-specific filtering
  - `with_custom()` - Custom key-value filters
  - Date range support (`date_from`, `date_to`)
- **StoreItem** - Builder for batch storage items
- **BatchResult** - Result type for batch operations

### Changed
- CLI dependencies now optional (enabled by `cli` feature flag)
- Binary requires `cli` feature (`required-features = ["cli"]`)
- `progress` and `tui` modules conditional on `cli` feature
- Restructured lib.rs exports for lib-first usage

### Documentation
- **Configuration Guide** - Complete integration guide for any Rust project
  - Environment variables reference (`.env` configuration)
  - Quick start with `for_app()` auto-config
  - Custom `MemexConfig` with provider configuration
  - Provider cascade setup (multiple fallback providers)
  - Namespace strategy best practices
  - Embedding models reference table (dimensions, sizes, use cases)
  - Troubleshooting section for common errors
- **Library Usage** - Comprehensive examples
  - Basic CRUD operations
  - Batch operations with `store_batch()`
  - GDPR-compliant deletion with `MetaFilter`
  - Agent Tools API for MCP integration

## [0.2.4] - 2025-12-28

### Added
- **BM25 Keyword Search** - Hybrid search combining semantic + keyword (Tantivy-based).
  - Stemming and language support for improved recall
  - Token-aware validation utilities
- **Context-Aware Chunking** - Sentence-aware chunking in RAG pipeline.
  - Context prefix injection for enriched chunks
  - Improved boundary detection
- **Embedding Retry Logic** - Failed chunks retry individually with exponential backoff.

## [0.2.3] - 2025-12-27

### Added
- **Smart Progress Bar** - `--progress` flag for index command with ETA.
  - Phase 1: Pre-scan (file count, total size, estimated chunks)
  - Phase 2: Calibration (measure embedding speed on first file)
  - Phase 3: Progress bar with ETA based on calibration
- **Verbose Error Logging** - Detailed diagnostics for embedding operations.
  - Full error chain with URL, model, batch size
  - Per-chunk diagnostics with text preview
  - Response body dump on parse failures

### Changed
- Added `indicatif` crate for progress bar rendering.

## [0.2.2] - 2025-12-27

### Added
- **Path Sanitization** - Security hardening for TUI wizard (Semgrep findings).
  - `sanitize_existing_path()` / `sanitize_new_path()` functions
  - Paths validated against allowed directories (home, /tmp, /var/folders)
  - Traversal sequence detection
- **Release Workflow** - GitHub Actions for multi-platform binary releases.
- **Install Script** - `curl | sh` installer with platform detection.
  ```bash
  curl -LsSf https://raw.githubusercontent.com/VetCoders/rmcp-memex/main/install.sh | sh
  ```

## [0.2.1] - 2025-12-26

### Added
- **TUI Wizard** - Interactive configuration wizard with provider detection.
  - Auto-detect Ollama and MLX embedding providers
  - Health check system for embedder connectivity
  - Host detection for Codex, Cursor, Claude, JetBrains, VSCode
  - Config merge (preserves existing servers)
  - Dimension hints and extended host snippets (ClaudeCode, Junie)
- **Upsert Command** - Direct text chunk insertion via CLI.
- **Git Hooks** - Pre-commit (auto-fix) and pre-push (format, lint, test, build, Semgrep).

### Fixed
- **UTF-8 Boundary Panic** - Use `chars().count()` for accurate character counting in embedding batching.

### Removed
- **sled Dependency** - Eliminated dead code causing file corruption and crash loops.
  - Storage now uses only LanceDB + moka in-memory cache.

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
