use anyhow::Result;
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tokio::sync::{Mutex, Semaphore};
use tracing::{Level, info};
use tracing_subscriber::FmtSubscriber;
use walkdir::WalkDir;

use rmcp_memex::{
    EmbeddingClient, EmbeddingConfig, HybridConfig, HybridSearchResult, HybridSearcher,
    IndexProgressTracker, MlxConfig, NamespaceSecurityConfig, PreprocessingConfig, ProviderConfig,
    QueryRouter, RAGPipeline, RerankerConfig, SearchMode, SearchModeRecommendation, ServerConfig,
    SliceLayer, SliceMode, StorageManager, WizardConfig, create_server, run_wizard,
};

fn parse_features(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

/// Standard config discovery locations (in priority order)
const CONFIG_SEARCH_PATHS: &[&str] = &[
    "~/.rmcp-servers/rmcp-memex/config.toml",
    "~/.config/rmcp-memex/config.toml",
    "~/.rmcp_servers/rmcp_memex/config.toml", // legacy underscore path
];

/// Discover config file from standard locations
fn discover_config() -> Option<String> {
    // 1. Environment variable takes priority
    if let Ok(path) = std::env::var("RMCP_MEMEX_CONFIG") {
        let expanded = shellexpand::tilde(&path).to_string();
        if std::path::Path::new(&expanded).exists() {
            return Some(path);
        }
    }

    // 2. Check standard locations
    for path in CONFIG_SEARCH_PATHS {
        let expanded = shellexpand::tilde(path).to_string();
        if std::path::Path::new(&expanded).exists() {
            return Some(path.to_string());
        }
    }

    None
}

fn load_file_config(path: &str) -> Result<FileConfig> {
    let expanded = shellexpand::tilde(path).to_string();
    // This is the START of path validation - canonicalize resolves symlinks
    let canonical = std::path::Path::new(&expanded) // nosemgrep
        .canonicalize()
        .map_err(|e| anyhow::anyhow!("Cannot resolve config path '{}': {}", path, e))?;

    // Security: validate path is under home directory or current working directory
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(std::path::PathBuf::from)
        .ok();
    let cwd = std::env::current_dir().ok();

    let is_safe = home
        .as_ref()
        .map(|h| canonical.starts_with(h))
        .unwrap_or(false)
        || cwd
            .as_ref()
            .map(|c| canonical.starts_with(c))
            .unwrap_or(false);

    if !is_safe {
        return Err(anyhow::anyhow!(
            "Access denied: config path '{}' is outside allowed directories",
            path
        ));
    }

    // Path is validated above: canonicalized + checked against HOME/CWD
    let contents = std::fs::read_to_string(&canonical)?; // nosemgrep
    toml::from_str(&contents).map_err(Into::into)
}

/// Load config from explicit path or discover from standard locations
fn load_or_discover_config(explicit_path: Option<&str>) -> Result<(FileConfig, Option<String>)> {
    // Explicit path takes priority
    if let Some(path) = explicit_path {
        return Ok((load_file_config(path)?, Some(path.to_string())));
    }

    // Try to discover config
    if let Some(discovered) = discover_config() {
        return Ok((load_file_config(&discovered)?, Some(discovered)));
    }

    // No config found - use defaults
    Ok((FileConfig::default(), None))
}

#[derive(serde::Deserialize, Default, Clone)]
struct FileConfig {
    mode: Option<String>,
    features: Option<String>,
    cache_mb: Option<usize>,
    db_path: Option<String>,
    max_request_bytes: Option<usize>,
    log_level: Option<String>,
    allowed_paths: Option<Vec<String>>,
    /// Enable namespace token-based access control
    security_enabled: Option<bool>,
    /// Path to token store file
    token_store_path: Option<String>,
    /// Enable preprocessing to filter noise from documents before indexing
    preprocessing_enabled: Option<bool>,
    /// New: universal embedding config with provider cascade
    #[serde(default)]
    embeddings: Option<EmbeddingsFileConfig>,
    /// Legacy: MLX embedding server configuration (deprecated)
    #[serde(default)]
    mlx: Option<MlxFileConfig>,
}

/// New embedding configuration from TOML
#[derive(serde::Deserialize, Clone)]
struct EmbeddingsFileConfig {
    #[serde(default = "default_dimension")]
    required_dimension: usize,
    #[serde(default)]
    providers: Vec<ProviderFileConfig>,
    #[serde(default)]
    reranker: Option<RerankerFileConfig>,
}

fn default_dimension() -> usize {
    4096
}

impl Default for EmbeddingsFileConfig {
    fn default() -> Self {
        Self {
            required_dimension: 4096,
            providers: vec![],
            reranker: None,
        }
    }
}

#[derive(serde::Deserialize, Clone)]
struct ProviderFileConfig {
    name: String,
    base_url: String,
    model: String,
    #[serde(default = "default_priority")]
    priority: u8,
    #[serde(default = "default_endpoint")]
    endpoint: String,
}

fn default_priority() -> u8 {
    10
}

fn default_endpoint() -> String {
    "/v1/embeddings".to_string()
}

#[derive(serde::Deserialize, Clone)]
struct RerankerFileConfig {
    base_url: String,
    model: String,
    #[serde(default = "default_rerank_endpoint")]
    endpoint: String,
}

fn default_rerank_endpoint() -> String {
    "/v1/rerank".to_string()
}

/// Legacy MLX embedding server configuration from TOML (deprecated)
#[derive(serde::Deserialize, Default, Clone)]
struct MlxFileConfig {
    #[serde(default)]
    disabled: bool,
    local_port: Option<u16>,
    dragon_url: Option<String>,
    dragon_port: Option<u16>,
    embedder_model: Option<String>,
    reranker_model: Option<String>,
    reranker_port_offset: Option<u16>,
}

impl MlxFileConfig {
    /// Convert legacy config to MlxConfig for backward compat
    fn to_mlx_config(&self) -> MlxConfig {
        let mut config = MlxConfig::from_env();
        config.merge_file_config(
            Some(self.disabled),
            self.local_port,
            self.dragon_url.clone(),
            self.dragon_port,
            self.embedder_model.clone(),
            self.reranker_model.clone(),
            self.reranker_port_offset,
        );
        config
    }
}

impl FileConfig {
    /// Convert to EmbeddingConfig - new format takes precedence over legacy
    fn to_embedding_config(&self) -> EmbeddingConfig {
        // New format takes precedence
        if let Some(ref emb) = self.embeddings
            && !emb.providers.is_empty()
        {
            let providers = emb
                .providers
                .iter()
                .map(|p| ProviderConfig {
                    name: p.name.clone(),
                    base_url: p.base_url.clone(),
                    model: p.model.clone(),
                    priority: p.priority,
                    endpoint: p.endpoint.clone(),
                })
                .collect();

            let reranker = emb
                .reranker
                .as_ref()
                .map(|r| RerankerConfig {
                    base_url: Some(r.base_url.clone()),
                    model: Some(r.model.clone()),
                    endpoint: r.endpoint.clone(),
                })
                .unwrap_or_default();

            return EmbeddingConfig {
                required_dimension: emb.required_dimension,
                max_batch_chars: 32000, // ~8K tokens, safe for most models
                max_batch_items: 16,    // max texts per batch
                providers,
                reranker,
            };
        }

        // Fallback to legacy [mlx] config
        if let Some(ref mlx) = self.mlx {
            tracing::warn!("Using legacy [mlx] config - please migrate to [embeddings.providers]");
            return mlx.to_mlx_config().to_embedding_config();
        }

        // Use defaults from environment variables (legacy support)
        MlxConfig::from_env().to_embedding_config()
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about = "RAG/memory MCP server with LanceDB vector storage", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Optional config file (TOML) to load settings from; CLI flags override file when set.
    #[arg(long, global = true)]
    config: Option<String>,

    /// Server mode: "memory" (memory-only, no filesystem) or "full" (all features)
    #[arg(long, value_parser = ["memory", "full"], global = true)]
    mode: Option<String>,

    /// Enable specific features (comma-separated). Overrides --mode if set.
    #[arg(long, global = true)]
    features: Option<String>,

    /// Cache size in MB
    #[arg(long, global = true)]
    cache_mb: Option<usize>,

    /// Path for embedded vector store (LanceDB)
    #[arg(long, global = true)]
    db_path: Option<String>,

    /// Max allowed request size in bytes for JSON-RPC framing
    #[arg(long, global = true)]
    max_request_bytes: Option<usize>,

    /// Log level
    #[arg(long, global = true)]
    log_level: Option<String>,

    /// Allowed paths for file access (whitelist). Can be specified multiple times.
    /// If not set, defaults to $HOME and current working directory.
    /// Supports ~ expansion and absolute paths.
    #[arg(long, global = true, action = clap::ArgAction::Append)]
    allowed_paths: Option<Vec<String>>,

    /// Enable namespace token-based access control.
    /// When enabled, protected namespaces require a token for access.
    #[arg(long, global = true)]
    security_enabled: bool,

    /// Path to token store file for namespace access tokens.
    /// Defaults to ~/.rmcp-servers/rmcp-memex/tokens.json when security is enabled.
    #[arg(long, global = true)]
    token_store_path: Option<String>,

    /// HTTP/SSE server port for multi-agent access.
    /// When set, starts an HTTP server alongside MCP stdio.
    /// Agents can query via HTTP instead of holding LanceDB lock directly.
    /// Example: --http-port 6660
    #[arg(long, global = true)]
    http_port: Option<u16>,

    /// Run HTTP server only, without MCP stdio.
    /// Use this for daemon mode where agents connect via HTTP.
    /// Requires --http-port to be set.
    #[arg(long, global = true)]
    http_only: bool,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run the MCP server (default if no subcommand specified)
    Serve,

    /// Launch interactive configuration wizard
    #[command(alias = "config")]
    Wizard {
        /// Dry run mode - show changes without writing files
        #[arg(long)]
        dry_run: bool,
    },

    /// Quick stats and health check for namespaces
    ///
    /// Shows chunk count, date range, top topics, and storage info.
    ///
    /// Examples:
    ///   rmcp-memex overview           # All namespaces
    ///   rmcp-memex overview memories  # Specific namespace
    Overview {
        /// Namespace to get overview for (optional, shows all if not specified)
        namespace: Option<String>,

        /// Output as JSON instead of human-readable format
        #[arg(long)]
        json: bool,
    },

    /// Deep exploration with all details - drill into onion layers
    ///
    /// Shows ALL onion layers (outer/middle/inner/core), both BM25 and vector scores,
    /// full metadata, and related chunks.
    ///
    /// Examples:
    ///   rmcp-memex dive -n memories -q "dragon"
    ///   rmcp-memex dive -n memories -q "dragon" --verbose
    Dive {
        /// Namespace to search in
        #[arg(long, short = 'n', required = true)]
        namespace: String,

        /// Search query text
        #[arg(long, short = 'q', required = true)]
        query: String,

        /// Maximum number of results per layer
        #[arg(long, short = 'l', default_value = "5")]
        limit: usize,

        /// Show extra verbose output (full text, all metadata)
        #[arg(long, short = 'v')]
        verbose: bool,

        /// Output as JSON instead of human-readable format
        #[arg(long)]
        json: bool,
    },

    /// Batch index documents into vector store
    Index {
        /// Path to file or directory to index
        #[arg(required = true)]
        path: PathBuf,

        /// Namespace for indexed documents (default: "rag")
        #[arg(long, short = 'n')]
        namespace: Option<String>,

        /// Recursively walk subdirectories
        #[arg(long, short = 'r')]
        recursive: bool,

        /// Glob pattern to filter files (e.g. "*.md", "*.pdf")
        #[arg(long, short = 'g')]
        glob: Option<String>,

        /// Maximum depth when walking directories (0 = unlimited)
        #[arg(long, default_value = "0")]
        max_depth: usize,

        /// Enable preprocessing to filter noise (tool artifacts, CLI output)
        /// before indexing. Reduces vector storage size and improves search quality.
        /// Note: timestamps are preserved by default; use --sanitize-metadata to remove them.
        #[arg(long, short = 'p')]
        preprocess: bool,

        /// Sanitize timestamps, UUIDs, and session IDs from content.
        /// By default, these are preserved for temporal queries.
        /// Use this flag when you want to anonymize or normalize the data.
        #[arg(long)]
        sanitize_metadata: bool,

        /// Slicing mode for document chunking:
        /// - "onion" (default): Hierarchical slices (outer/middle/inner/core) for efficient context
        /// - "onion-fast" / "fast": Only outer+core layers (2x faster, good for large datasets)
        /// - "flat": Traditional fixed-size chunks with overlap
        #[arg(long, short = 's', default_value = "onion", value_parser = ["onion", "onion-fast", "fast", "flat"])]
        slice_mode: String,

        /// Enable exact-match deduplication (default: enabled).
        /// Skips indexing files whose content already exists in the namespace.
        /// Uses SHA256 hash of original content before any preprocessing.
        #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
        dedup: bool,

        /// Show smart progress bar with ETA based on calibration.
        /// Displays three phases: pre-scan, calibration, and indexing progress.
        #[arg(long)]
        progress: bool,

        /// Resume from last checkpoint if interrupted.
        /// Saves progress after each file to .index-checkpoint.json.
        /// On restart, skips already indexed files and continues.
        #[arg(long)]
        resume: bool,

        /// Enable async pipeline mode for concurrent indexing.
        /// Runs file reading, chunking, embedding, and storage in parallel
        /// using tokio channels. Can significantly speed up large batch operations.
        /// Note: Pipeline mode ignores --progress and --resume flags.
        #[arg(long)]
        pipeline: bool,

        /// Number of files to process in parallel (default: 4, max: 16).
        /// Higher values can speed up indexing on multi-core systems,
        /// but may increase memory usage and API pressure.
        /// Note: This is ignored when --pipeline is enabled.
        #[arg(long, short = 'P', default_value = "4", value_parser = clap::value_parser!(u8).range(1..=16))]
        parallel: u8,
    },

    /// Smart semantic search within a namespace
    ///
    /// Finds relevant information using vector similarity search with intelligent
    /// defaults. Results include relevance scores, timestamps, and metadata.
    ///
    /// Examples:
    ///   rmcp-memex search -n memories -q "when did we buy dragon"
    ///   rmcp-memex search -n memories -q "dragon" --deep
    ///   rmcp-memex search -n memories -q "dragon" -l 20
    ///   rmcp-memex search -n memories -q "dragon" --mode hybrid
    Search {
        /// Namespace to search in
        #[arg(long, short = 'n', required = true)]
        namespace: String,

        /// Search query text
        #[arg(long, short = 'q', required = true)]
        query: String,

        /// Maximum number of results to return (default: 10)
        #[arg(long, short = 'l', default_value = "10")]
        limit: usize,

        /// Output results as JSON instead of human-readable format
        #[arg(long)]
        json: bool,

        /// Deep search: include all layers (outer/middle/inner/core) instead of just outer
        #[arg(long)]
        deep: bool,

        /// Filter by specific layer (outer, middle, inner, core)
        #[arg(long, value_parser = ["outer", "middle", "inner", "core"])]
        layer: Option<String>,

        /// Search mode: vector (similarity only), keyword/bm25 (lexical only), or hybrid (default)
        /// Hybrid combines vector and BM25 using score fusion for best results.
        #[arg(long, short = 'm', default_value = "hybrid", value_parser = ["vector", "keyword", "bm25", "hybrid"])]
        mode: String,

        /// Auto-detect query intent and select optimal search mode.
        /// Overrides --mode when enabled. Uses QueryRouter to analyze query.
        #[arg(long)]
        auto_route: bool,

        /// Show relevance scores prominently (enabled by default)
        #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
        scores: bool,
    },

    /// Expand a slice to get its children (drill down in onion hierarchy)
    Expand {
        /// Namespace containing the slice
        #[arg(long, short = 'n', required = true)]
        namespace: String,

        /// Slice ID to expand
        #[arg(long, short = 'i', required = true)]
        id: String,

        /// Output results as JSON instead of human-readable format
        #[arg(long)]
        json: bool,
    },

    /// Get a specific chunk by namespace and ID
    Get {
        /// Namespace containing the chunk
        #[arg(long, short = 'n', required = true)]
        namespace: String,

        /// Chunk ID to retrieve
        #[arg(long, short = 'i', required = true)]
        id: String,

        /// Output result as JSON instead of human-readable format
        #[arg(long)]
        json: bool,
    },

    /// RAG search across all namespaces or a specific one
    RagSearch {
        /// Search query text
        #[arg(long, short = 'q', required = true)]
        query: String,

        /// Maximum number of results to return
        #[arg(long, short = 'l', default_value = "10")]
        limit: usize,

        /// Optional namespace to limit search to
        #[arg(long, short = 'n')]
        namespace: Option<String>,

        /// Output results as JSON instead of human-readable format
        #[arg(long)]
        json: bool,
    },

    /// List all namespaces with optional statistics
    Namespaces {
        /// Show statistics (document count, etc.)
        #[arg(long, short = 's')]
        stats: bool,

        /// Output as JSON instead of human-readable format
        #[arg(long)]
        json: bool,
    },

    /// Export a namespace to JSON file
    Export {
        /// Namespace to export
        #[arg(long, short = 'n', required = true)]
        namespace: String,

        /// Output file path (stdout if not specified)
        #[arg(long, short = 'o')]
        output: Option<PathBuf>,

        /// Include embeddings in export (large!)
        #[arg(long)]
        include_embeddings: bool,
    },

    /// Upsert a text chunk directly into vector memory (for hooks/scripts)
    Upsert {
        /// Namespace for the chunk
        #[arg(long, short = 'n', required = true)]
        namespace: String,

        /// Unique ID for the chunk
        #[arg(long, short = 'i', required = true)]
        id: String,

        /// Text content (if not provided, reads from stdin)
        #[arg(long, short = 't')]
        text: Option<String>,

        /// Optional metadata as JSON string
        #[arg(long, short = 'm', default_value = "{}")]
        metadata: String,
    },

    /// Optimize database: compact files and cleanup old versions
    ///
    /// Runs both compaction (merge small files) and pruning (remove old versions).
    /// Use this after large indexing operations to improve query performance
    /// and reduce file descriptor usage.
    Optimize,

    /// Compact database files into larger chunks
    ///
    /// Merges small data files into larger ones for better read performance.
    /// Run this after many small inserts to reduce "too many open files" errors.
    Compact,

    /// Cleanup old database versions
    ///
    /// Removes old versions of the data that are no longer needed.
    /// By default, keeps versions from the last 7 days.
    Cleanup {
        /// Remove versions older than N days (default: 7)
        #[arg(long, default_value = "7")]
        older_than_days: u64,
    },

    /// Show database statistics
    ///
    /// Displays row count, version count, and storage information.
    Stats,
}

impl Cli {
    fn into_server_config(self) -> Result<ServerConfig> {
        let (file_cfg, config_path) = load_or_discover_config(self.config.as_deref())?;
        if let Some(ref path) = config_path {
            eprintln!("Using config: {}", path);
        }

        // Extract embedding config first (before any moves from file_cfg)
        let embeddings = file_cfg.to_embedding_config();

        // Determine base config from mode (CLI > file > default)
        let mode = self.mode.as_deref().or(file_cfg.mode.as_deref());
        let base_cfg = match mode {
            Some("memory") => ServerConfig::for_memory_only(),
            Some("full") => ServerConfig::for_full_rag(),
            _ => ServerConfig::default(),
        };

        // CLI --features overrides mode-derived features
        let features = self
            .features
            .or(file_cfg.features)
            .map(|s| parse_features(&s))
            .unwrap_or(base_cfg.features);

        // Build security config from CLI and file settings
        let security_enabled = self.security_enabled || file_cfg.security_enabled.unwrap_or(false);
        let token_store_path = self.token_store_path.or(file_cfg.token_store_path);

        Ok(ServerConfig {
            features,
            cache_mb: self
                .cache_mb
                .or(file_cfg.cache_mb)
                .unwrap_or(base_cfg.cache_mb),
            db_path: self
                .db_path
                .or(file_cfg.db_path)
                .unwrap_or(base_cfg.db_path),
            max_request_bytes: self
                .max_request_bytes
                .or(file_cfg.max_request_bytes)
                .unwrap_or(base_cfg.max_request_bytes),
            log_level: self
                .log_level
                .or(file_cfg.log_level)
                .map(|s| parse_log_level(&s))
                .unwrap_or(base_cfg.log_level),
            allowed_paths: self
                .allowed_paths
                .or(file_cfg.allowed_paths)
                .unwrap_or(base_cfg.allowed_paths),
            security: NamespaceSecurityConfig {
                enabled: security_enabled,
                token_store_path,
            },
            embeddings,
            hybrid: base_cfg.hybrid,
        })
    }
}

fn parse_log_level(level: &str) -> Level {
    match level.to_ascii_lowercase().as_str() {
        "trace" => Level::TRACE,
        "debug" => Level::DEBUG,
        "info" => Level::INFO,
        "warn" => Level::WARN,
        "error" => Level::ERROR,
        _ => Level::INFO,
    }
}

/// Check if a path matches a glob pattern
fn matches_glob(path: &Path, pattern: &str) -> bool {
    let file_name = match path.file_name().and_then(|n| n.to_str()) {
        Some(n) => n,
        None => return false,
    };
    glob::Pattern::new(pattern)
        .map(|p| p.matches(file_name))
        .unwrap_or(false)
}

/// Collect files to index based on path, recursion, and glob settings
fn collect_files(
    path: &Path,
    recursive: bool,
    glob_pattern: Option<&str>,
    max_depth: usize,
) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    if path.is_file() {
        // Single file - check glob if provided
        if let Some(pattern) = glob_pattern {
            if matches_glob(path, pattern) {
                files.push(path.to_path_buf());
            }
        } else {
            files.push(path.to_path_buf());
        }
        return Ok(files);
    }

    // Directory walk
    let mut walker = WalkDir::new(path);
    if !recursive {
        walker = walker.max_depth(1);
    } else if max_depth > 0 {
        walker = walker.max_depth(max_depth);
    }

    for entry in walker.into_iter().filter_map(|e| e.ok()) {
        let entry_path = entry.path();
        if !entry_path.is_file() {
            continue;
        }

        // Filter by glob pattern if provided
        if glob_pattern.is_some_and(|pattern| !matches_glob(entry_path, pattern)) {
            continue;
        }

        files.push(entry_path.to_path_buf());
    }

    Ok(files)
}

/// Format and display search results (human-readable)
fn display_search_results(
    query: &str,
    namespace: Option<&str>,
    results: &[rmcp_memex::SearchResult],
    layer_filter: Option<SliceLayer>,
) {
    let ns_display = namespace.unwrap_or("all namespaces");
    let layer_display = layer_filter
        .map(|l| format!(" (layer: {})", l.name()))
        .unwrap_or_default();

    eprintln!(
        "\n-> Search Results for \"{}\" in [{}]{}\n",
        query, ns_display, layer_display
    );

    if results.is_empty() {
        eprintln!("No results found.");
        return;
    }

    for (i, result) in results.iter().enumerate() {
        // Truncate text for display
        let preview: String = result
            .text
            .chars()
            .take(100)
            .collect::<String>()
            .replace('\n', " ");
        let ellipsis = if result.text.len() > 100 { "..." } else { "" };

        // Layer info
        let layer_str = result
            .layer
            .map(|l| format!("[{}]", l.name()))
            .unwrap_or_default();

        eprintln!(
            "{}. [{:.2}] {} {}",
            i + 1,
            result.score,
            result.namespace,
            layer_str
        );
        eprintln!("   \"{}{ellipsis}\"", preview);
        eprintln!("   ID: {}", result.id);
        if !result.keywords.is_empty() {
            eprintln!("   Keywords: {}", result.keywords.join(", "));
        }
        if result.can_expand() {
            eprintln!("   [expandable: {} children]", result.children_ids.len());
        }
        if !result.metadata.is_null() && result.metadata != serde_json::json!({}) {
            eprintln!("   Metadata: {}", result.metadata);
        }
        eprintln!();
    }
}

/// Format search results as JSON
fn json_search_results(
    query: &str,
    namespace: Option<&str>,
    results: &[rmcp_memex::SearchResult],
    layer_filter: Option<SliceLayer>,
) -> serde_json::Value {
    serde_json::json!({
        "query": query,
        "namespace": namespace,
        "layer_filter": layer_filter.map(|l| l.name()),
        "count": results.len(),
        "results": results.iter().map(|r| serde_json::json!({
            "id": r.id,
            "namespace": r.namespace,
            "score": r.score,
            "text": r.text,
            "layer": r.layer.map(|l| l.name()),
            "keywords": r.keywords,
            "parent_id": r.parent_id,
            "children_ids": r.children_ids,
            "can_expand": r.can_expand(),
            "metadata": r.metadata
        })).collect::<Vec<_>>()
    })
}

/// Format and display hybrid search results (human-readable)
fn display_hybrid_search_results(
    query: &str,
    namespace: Option<&str>,
    results: &[HybridSearchResult],
    layer_filter: Option<SliceLayer>,
    search_mode: SearchMode,
) {
    let ns_display = namespace.unwrap_or("all namespaces");
    let layer_display = layer_filter
        .map(|l| format!(" (layer: {})", l.name()))
        .unwrap_or_default();
    let mode_display = match search_mode {
        SearchMode::Hybrid => "hybrid",
        SearchMode::Keyword => "keyword/bm25",
        SearchMode::Vector => "vector",
    };

    eprintln!(
        "\n-> Search Results for \"{}\" in [{}]{} [mode: {}]\n",
        query, ns_display, layer_display, mode_display
    );

    if results.is_empty() {
        eprintln!("No results found.");
        return;
    }

    for (i, result) in results.iter().enumerate() {
        // Truncate text for display
        let preview: String = result
            .document
            .chars()
            .take(100)
            .collect::<String>()
            .replace('\n', " ");
        let ellipsis = if result.document.len() > 100 {
            "..."
        } else {
            ""
        };

        // Layer info
        let layer_str = result
            .layer
            .map(|l| format!("[{}]", l.name()))
            .unwrap_or_default();

        // Score breakdown
        let score_details = match (result.vector_score, result.bm25_score) {
            (Some(v), Some(b)) => format!(
                "[combined: {:.2}, vec: {:.2}, bm25: {:.2}]",
                result.combined_score, v, b
            ),
            (Some(v), None) => format!("[vec: {:.2}]", v),
            (None, Some(b)) => format!("[bm25: {:.2}]", b),
            (None, None) => format!("[score: {:.2}]", result.combined_score),
        };

        eprintln!(
            "{}. {} {} {}",
            i + 1,
            score_details,
            result.namespace,
            layer_str
        );
        eprintln!("   \"{}{ellipsis}\"", preview);
        eprintln!("   ID: {}", result.id);
        if !result.keywords.is_empty() {
            eprintln!("   Keywords: {}", result.keywords.join(", "));
        }
        if !result.children_ids.is_empty() {
            eprintln!("   [expandable: {} children]", result.children_ids.len());
        }
        if !result.metadata.is_null() && result.metadata != serde_json::json!({}) {
            eprintln!("   Metadata: {}", result.metadata);
        }
        eprintln!();
    }
}

/// Format hybrid search results as JSON
fn json_hybrid_search_results(
    query: &str,
    namespace: Option<&str>,
    results: &[HybridSearchResult],
    layer_filter: Option<SliceLayer>,
    search_mode: SearchMode,
) -> serde_json::Value {
    serde_json::json!({
        "query": query,
        "namespace": namespace,
        "layer_filter": layer_filter.map(|l| l.name()),
        "search_mode": match search_mode {
            SearchMode::Hybrid => "hybrid",
            SearchMode::Keyword => "keyword",
            SearchMode::Vector => "vector",
        },
        "count": results.len(),
        "results": results.iter().map(|r| serde_json::json!({
            "id": r.id,
            "namespace": r.namespace,
            "combined_score": r.combined_score,
            "vector_score": r.vector_score,
            "bm25_score": r.bm25_score,
            "text": r.document,
            "layer": r.layer.map(|l| l.name()),
            "keywords": r.keywords,
            "parent_id": r.parent_id,
            "children_ids": r.children_ids,
            "metadata": r.metadata
        })).collect::<Vec<_>>()
    })
}

/// Run semantic search within a namespace
#[allow(clippy::too_many_arguments)] // CLI entry point - args from clap parser
async fn run_search(
    namespace: String,
    query: String,
    limit: usize,
    json_output: bool,
    db_path: String,
    layer_filter: Option<SliceLayer>,
    search_mode: SearchMode,
    embedding_config: &EmbeddingConfig,
) -> Result<()> {
    let embedding_client = Arc::new(Mutex::new(EmbeddingClient::new(embedding_config).await?));
    let storage = Arc::new(StorageManager::new_lance_only(&db_path).await?);

    // Use hybrid search if mode is not pure vector
    if search_mode != SearchMode::Vector {
        // Create hybrid config with specified mode
        let hybrid_config = HybridConfig {
            mode: search_mode,
            ..Default::default()
        };
        let hybrid_searcher = HybridSearcher::new(storage, hybrid_config).await?;

        // Get query embedding
        let query_embedding = embedding_client.lock().await.embed(&query).await?;

        let results = hybrid_searcher
            .search(
                &query,
                query_embedding,
                Some(&namespace),
                limit,
                layer_filter,
            )
            .await?;

        if json_output {
            let json = json_hybrid_search_results(
                &query,
                Some(&namespace),
                &results,
                layer_filter,
                search_mode,
            );
            println!("{}", serde_json::to_string_pretty(&json)?);
        } else {
            display_hybrid_search_results(
                &query,
                Some(&namespace),
                &results,
                layer_filter,
                search_mode,
            );
        }
    } else {
        // Legacy vector-only search
        let rag = RAGPipeline::new(embedding_client, storage).await?;
        let results = rag
            .memory_search_with_layer(&namespace, &query, limit, layer_filter)
            .await?;

        if json_output {
            let json = json_search_results(&query, Some(&namespace), &results, layer_filter);
            println!("{}", serde_json::to_string_pretty(&json)?);
        } else {
            display_search_results(&query, Some(&namespace), &results, layer_filter);
        }
    }

    Ok(())
}

/// Expand a slice to get its children (drill down in onion hierarchy)
async fn run_expand(
    namespace: String,
    id: String,
    json_output: bool,
    db_path: String,
    embedding_config: &EmbeddingConfig,
) -> Result<()> {
    let embedding_client = Arc::new(Mutex::new(EmbeddingClient::new(embedding_config).await?));
    let storage = Arc::new(StorageManager::new_lance_only(&db_path).await?);
    let rag = RAGPipeline::new(embedding_client, storage).await?;

    let results = rag.expand_result(&namespace, &id).await?;

    if json_output {
        let json = serde_json::json!({
            "parent_id": id,
            "namespace": namespace,
            "children_count": results.len(),
            "children": results.iter().map(|r| serde_json::json!({
                "id": r.id,
                "layer": r.layer.map(|l| l.name()),
                "text": r.text,
                "keywords": r.keywords,
                "parent_id": r.parent_id,
                "children_ids": r.children_ids,
            })).collect::<Vec<_>>()
        });
        println!("{}", serde_json::to_string_pretty(&json)?);
    } else {
        eprintln!("\n-> Children of slice \"{id}\" in [{namespace}]\n");

        if results.is_empty() {
            eprintln!("No children found (this may be a leaf/outer slice).");
        } else {
            for (i, result) in results.iter().enumerate() {
                let layer_str = result.layer.map(|l| l.name()).unwrap_or("flat");
                let preview: String = result
                    .text
                    .chars()
                    .take(100)
                    .collect::<String>()
                    .replace('\n', " ");
                let ellipsis = if result.text.len() > 100 { "..." } else { "" };

                eprintln!("{}. [{}] {}", i + 1, layer_str, result.id);
                eprintln!("   \"{}{ellipsis}\"", preview);
                if !result.keywords.is_empty() {
                    eprintln!("   Keywords: {}", result.keywords.join(", "));
                }
                eprintln!();
            }
        }
    }

    Ok(())
}

/// Get a specific chunk by namespace and ID
async fn run_get(
    namespace: String,
    id: String,
    json_output: bool,
    db_path: String,
    embedding_config: &EmbeddingConfig,
) -> Result<()> {
    let embedding_client = Arc::new(Mutex::new(EmbeddingClient::new(embedding_config).await?));
    let storage = Arc::new(StorageManager::new_lance_only(&db_path).await?);
    let rag = RAGPipeline::new(embedding_client, storage).await?;

    match rag.memory_get(&namespace, &id).await? {
        Some(result) => {
            if json_output {
                let json = serde_json::json!({
                    "found": true,
                    "id": result.id,
                    "namespace": result.namespace,
                    "text": result.text,
                    "metadata": result.metadata
                });
                println!("{}", serde_json::to_string_pretty(&json)?);
            } else {
                eprintln!("\n-> Found chunk in [{namespace}]\n");
                eprintln!("ID: {}", result.id);
                eprintln!("Namespace: {}", result.namespace);
                if !result.metadata.is_null() && result.metadata != serde_json::json!({}) {
                    eprintln!("Metadata: {}", result.metadata);
                }
                eprintln!("\n--- Content ---\n");
                println!("{}", result.text);
            }
        }
        None => {
            if json_output {
                let json = serde_json::json!({
                    "found": false,
                    "namespace": namespace,
                    "id": id
                });
                println!("{}", serde_json::to_string_pretty(&json)?);
            } else {
                eprintln!("Chunk '{}' not found in namespace '{}'", id, namespace);
            }
        }
    }

    Ok(())
}

/// Run RAG search (optionally across all namespaces)
async fn run_rag_search(
    query: String,
    limit: usize,
    namespace: Option<String>,
    json_output: bool,
    db_path: String,
    embedding_config: &EmbeddingConfig,
) -> Result<()> {
    let embedding_client = Arc::new(Mutex::new(EmbeddingClient::new(embedding_config).await?));
    let storage = Arc::new(StorageManager::new_lance_only(&db_path).await?);
    let rag = RAGPipeline::new(embedding_client, storage).await?;

    let results = rag
        .search_inner(namespace.as_deref(), &query, limit)
        .await?;

    if json_output {
        let json = json_search_results(&query, namespace.as_deref(), &results, None);
        println!("{}", serde_json::to_string_pretty(&json)?);
    } else {
        display_search_results(&query, namespace.as_deref(), &results, None);
    }

    Ok(())
}

/// List all namespaces with optional stats
async fn run_list_namespaces(stats: bool, json_output: bool, db_path: String) -> Result<()> {
    let storage = StorageManager::new_lance_only(&db_path).await?;

    // We need to query the LanceDB to get unique namespaces
    // This requires querying all documents and extracting unique namespaces
    let storage = Arc::new(storage);

    // Use a dummy embedding to get all documents (we'll use a zero vector)
    // This is a workaround since LanceDB doesn't have a direct "list all" method
    // We search with a large limit to get namespace statistics
    let zero_embedding = vec![0.0_f32; 4096]; // Max dimension for Qwen3
    let all_docs = storage.search_store(None, zero_embedding, 10000).await?;

    // Collect unique namespaces with counts
    let mut namespace_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for doc in &all_docs {
        *namespace_counts.entry(doc.namespace.clone()).or_insert(0) += 1;
    }

    let mut namespaces: Vec<_> = namespace_counts.into_iter().collect();
    namespaces.sort_by(|a, b| a.0.cmp(&b.0));

    if json_output {
        let json = if stats {
            serde_json::json!({
                "namespaces": namespaces.iter().map(|(ns, count)| serde_json::json!({
                    "name": ns,
                    "document_count": count
                })).collect::<Vec<_>>()
            })
        } else {
            serde_json::json!({
                "namespaces": namespaces.iter().map(|(ns, _)| ns).collect::<Vec<_>>()
            })
        };
        println!("{}", serde_json::to_string_pretty(&json)?);
    } else {
        eprintln!("\n-> Namespaces in {}\n", storage.lance_path());

        if namespaces.is_empty() {
            eprintln!("No namespaces found (database may be empty).");
        } else {
            for (ns, count) in &namespaces {
                if stats {
                    eprintln!("  {} ({} documents)", ns, count);
                } else {
                    eprintln!("  {}", ns);
                }
            }
            eprintln!();
            eprintln!("Total: {} namespace(s)", namespaces.len());
        }
    }

    Ok(())
}

/// Namespace overview stats
#[derive(Debug, Clone, serde::Serialize)]
struct NamespaceStats {
    name: String,
    total_chunks: usize,
    layer_counts: std::collections::HashMap<String, usize>,
    top_keywords: Vec<(String, usize)>,
    has_timestamps: bool,
    earliest_indexed: Option<String>,
    latest_indexed: Option<String>,
}

/// Run overview command - quick stats and health check
async fn run_overview(namespace: Option<String>, json_output: bool, db_path: String) -> Result<()> {
    let storage = StorageManager::new_lance_only(&db_path).await?;
    let storage = Arc::new(storage);

    // Use a zero embedding to get all documents
    let zero_embedding = vec![0.0_f32; 4096];
    let all_docs = storage
        .search_store(namespace.as_deref(), zero_embedding, 100000)
        .await?;

    if all_docs.is_empty() {
        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "status": "empty",
                    "message": "No documents found",
                    "namespace": namespace,
                    "db_path": db_path
                }))?
            );
        } else {
            eprintln!("\n-> Overview for {}\n", storage.lance_path());
            if let Some(ns) = &namespace {
                eprintln!("No documents found in namespace '{}'", ns);
            } else {
                eprintln!("Database is empty. Use 'rmcp-memex index' to add documents.");
            }
        }
        return Ok(());
    }

    // Group by namespace
    let mut by_namespace: std::collections::HashMap<String, Vec<_>> =
        std::collections::HashMap::new();
    for doc in &all_docs {
        by_namespace
            .entry(doc.namespace.clone())
            .or_default()
            .push(doc);
    }

    let mut stats_list: Vec<NamespaceStats> = Vec::new();

    for (ns_name, docs) in &by_namespace {
        // Count layers
        let mut layer_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for doc in docs {
            let layer_name = match doc.layer {
                1 => "outer",
                2 => "middle",
                3 => "inner",
                4 => "core",
                _ => "flat",
            };
            *layer_counts.entry(layer_name.to_string()).or_insert(0) += 1;
        }

        // Collect all keywords and count frequency
        let mut keyword_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for doc in docs {
            for kw in &doc.keywords {
                *keyword_counts.entry(kw.clone()).or_insert(0) += 1;
            }
        }
        let mut top_keywords: Vec<_> = keyword_counts.into_iter().collect();
        top_keywords.sort_by(|a, b| b.1.cmp(&a.1));
        let top_keywords: Vec<(String, usize)> = top_keywords.into_iter().take(10).collect();

        // Check for timestamps in metadata (look for common timestamp patterns)
        let has_timestamps = docs.iter().any(|d| {
            let meta_str = d.metadata.to_string();
            meta_str.contains("timestamp")
                || meta_str.contains("created_at")
                || meta_str.contains("indexed_at")
                || meta_str.contains("date")
        });

        // Try to extract date range from metadata
        let mut dates: Vec<String> = Vec::new();
        for doc in docs {
            if let Some(obj) = doc.metadata.as_object() {
                for (k, v) in obj {
                    if (k.contains("date") || k.contains("timestamp") || k.contains("time"))
                        && let Some(s) = v.as_str()
                    {
                        dates.push(s.to_string());
                    }
                }
            }
        }
        dates.sort();

        stats_list.push(NamespaceStats {
            name: ns_name.clone(),
            total_chunks: docs.len(),
            layer_counts,
            top_keywords,
            has_timestamps,
            earliest_indexed: dates.first().cloned(),
            latest_indexed: dates.last().cloned(),
        });
    }

    stats_list.sort_by(|a, b| a.name.cmp(&b.name));

    if json_output {
        let json = serde_json::json!({
            "db_path": db_path,
            "total_chunks": all_docs.len(),
            "namespace_count": stats_list.len(),
            "namespaces": stats_list
        });
        println!("{}", serde_json::to_string_pretty(&json)?);
    } else {
        eprintln!("\n=== RMCP-MEMEX OVERVIEW ===\n");
        eprintln!("Database: {}", db_path);
        eprintln!("Total chunks: {}", all_docs.len());
        eprintln!("Namespaces: {}\n", stats_list.len());

        for stats in &stats_list {
            eprintln!("--- {} ---", stats.name);
            eprintln!("  Chunks: {}", stats.total_chunks);

            // Layer breakdown
            if !stats.layer_counts.is_empty() {
                let layer_str: Vec<String> = stats
                    .layer_counts
                    .iter()
                    .map(|(k, v)| format!("{}:{}", k, v))
                    .collect();
                eprintln!("  Layers: {}", layer_str.join(", "));
            }

            // Top keywords
            if !stats.top_keywords.is_empty() {
                let kw_str: Vec<String> = stats
                    .top_keywords
                    .iter()
                    .take(5)
                    .map(|(k, v)| format!("{}({})", k, v))
                    .collect();
                eprintln!("  Top topics: {}", kw_str.join(", "));
            }

            // Date range
            if let (Some(earliest), Some(latest)) = (&stats.earliest_indexed, &stats.latest_indexed)
            {
                if earliest != latest {
                    eprintln!("  Date range: {} -> {}", earliest, latest);
                } else {
                    eprintln!("  Date: {}", earliest);
                }
            }

            // Timestamps warning
            if !stats.has_timestamps {
                eprintln!("  [!] No timestamp metadata found");
            }

            eprintln!();
        }

        eprintln!("Tip: Use 'rmcp-memex search -n <namespace> -q <query>' to search");
        eprintln!("     Use 'rmcp-memex dive -n <namespace> -q <query>' for deep exploration");
    }

    Ok(())
}

/// Run dive command - deep exploration with all onion layers
async fn run_dive(
    namespace: String,
    query: String,
    limit: usize,
    verbose: bool,
    json_output: bool,
    db_path: String,
    embedding_config: &EmbeddingConfig,
) -> Result<()> {
    let embedding_client = Arc::new(Mutex::new(EmbeddingClient::new(embedding_config).await?));
    let storage = Arc::new(StorageManager::new_lance_only(&db_path).await?);
    let rag = RAGPipeline::new(embedding_client, storage).await?;

    // Search each layer separately
    let layers = [
        (Some(SliceLayer::Outer), "OUTER"),
        (Some(SliceLayer::Middle), "MIDDLE"),
        (Some(SliceLayer::Inner), "INNER"),
        (Some(SliceLayer::Core), "CORE"),
    ];

    let mut all_results: Vec<serde_json::Value> = Vec::new();

    if !json_output {
        eprintln!("\n=== DEEP DIVE: \"{}\" in [{}] ===\n", query, namespace);
    }

    for (layer_filter, layer_name) in &layers {
        let results = rag
            .memory_search_with_layer(&namespace, &query, limit, *layer_filter)
            .await?;

        if json_output {
            let layer_results: Vec<serde_json::Value> = results
                .iter()
                .map(|r| {
                    let mut obj = serde_json::json!({
                        "id": r.id,
                        "score": r.score,
                        "keywords": r.keywords,
                        "layer": r.layer.map(|l| l.name()),
                        "can_expand": r.can_expand(),
                        "parent_id": r.parent_id,
                    });
                    if verbose {
                        obj["text"] = serde_json::json!(r.text);
                        obj["metadata"] = r.metadata.clone();
                        obj["children_ids"] = serde_json::json!(r.children_ids);
                    } else {
                        // Truncated preview
                        let preview: String = r.text.chars().take(200).collect();
                        obj["preview"] = serde_json::json!(preview);
                    }
                    obj
                })
                .collect();

            all_results.push(serde_json::json!({
                "layer": layer_name,
                "count": results.len(),
                "results": layer_results
            }));
        } else {
            eprintln!("--- {} LAYER ({} results) ---", layer_name, results.len());

            if results.is_empty() {
                eprintln!("  (no results)\n");
                continue;
            }

            for (i, result) in results.iter().enumerate() {
                eprintln!("  {}. [score: {:.3}] {}", i + 1, result.score, result.id);

                // Keywords
                if !result.keywords.is_empty() {
                    eprintln!("     Keywords: {}", result.keywords.join(", "));
                }

                // Text preview or full text
                if verbose {
                    eprintln!("     ---");
                    // Indent each line of text
                    for line in result.text.lines().take(20) {
                        eprintln!("     {}", line);
                    }
                    if result.text.lines().count() > 20 {
                        eprintln!("     ... ({} more lines)", result.text.lines().count() - 20);
                    }
                    eprintln!("     ---");

                    // Full metadata
                    if !result.metadata.is_null() && result.metadata != serde_json::json!({}) {
                        eprintln!("     Metadata: {}", result.metadata);
                    }
                } else {
                    // Short preview
                    let preview: String = result
                        .text
                        .chars()
                        .take(100)
                        .collect::<String>()
                        .replace('\n', " ");
                    let ellipsis = if result.text.len() > 100 { "..." } else { "" };
                    eprintln!("     \"{}{}\"", preview, ellipsis);
                }

                // Hierarchy info
                if result.can_expand() {
                    eprintln!("     [expandable: {} children]", result.children_ids.len());
                }
                if result.parent_id.is_some() {
                    eprintln!("     [has parent: can drill up]");
                }

                eprintln!();
            }
        }
    }

    if json_output {
        let output = serde_json::json!({
            "query": query,
            "namespace": namespace,
            "limit_per_layer": limit,
            "verbose": verbose,
            "layers": all_results
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        eprintln!("=== END DIVE ===\n");
        eprintln!(
            "Tip: Use 'rmcp-memex expand -n {} -i <id>' to expand a result",
            namespace
        );
    }

    Ok(())
}

/// Export a namespace to JSON file
async fn run_export(
    namespace: String,
    output: Option<PathBuf>,
    include_embeddings: bool,
    db_path: String,
) -> Result<()> {
    let storage = StorageManager::new_lance_only(&db_path).await?;

    // Get all documents in the namespace
    // Using a zero vector to search - this gets documents by namespace filter
    let zero_embedding = vec![0.0_f32; 4096]; // Max dimension
    let docs = storage
        .search_store(Some(&namespace), zero_embedding, 100000)
        .await?;

    if docs.is_empty() {
        eprintln!("No documents found in namespace '{}'", namespace);
        return Ok(());
    }

    // Build export structure
    let export_data: Vec<serde_json::Value> = docs
        .iter()
        .map(|doc| {
            let mut obj = serde_json::json!({
                "id": doc.id,
                "namespace": doc.namespace,
                "text": doc.document,
                "metadata": doc.metadata
            });

            if include_embeddings {
                obj["embedding"] = serde_json::json!(doc.embedding);
            }

            obj
        })
        .collect();

    let export_json = serde_json::json!({
        "namespace": namespace,
        "exported_at": chrono::Utc::now().to_rfc3339(),
        "document_count": export_data.len(),
        "include_embeddings": include_embeddings,
        "documents": export_data
    });

    let json_string = serde_json::to_string_pretty(&export_json)?;

    match output {
        Some(path) => {
            std::fs::write(&path, &json_string)?;
            eprintln!(
                "Exported {} documents from '{}' to {:?}",
                docs.len(),
                namespace,
                path
            );
        }
        None => {
            println!("{}", json_string);
        }
    }

    Ok(())
}

/// Checkpoint for resumable indexing
#[derive(Debug, Serialize, Deserialize)]
struct IndexCheckpoint {
    /// Namespace being indexed
    namespace: String,
    /// Files that have been successfully indexed (canonical paths)
    indexed_files: HashSet<String>,
    /// When checkpoint was last updated
    updated_at: String,
}

impl IndexCheckpoint {
    fn new(namespace: &str) -> Self {
        Self {
            namespace: namespace.to_string(),
            indexed_files: HashSet::new(),
            updated_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    fn checkpoint_path(db_path: &str, namespace: &str) -> PathBuf {
        let expanded = shellexpand::tilde(db_path).to_string();
        Path::new(&expanded)
            .parent()
            .unwrap_or(Path::new("."))
            .join(format!(".index-checkpoint-{}.json", namespace))
    }

    fn load(db_path: &str, namespace: &str) -> Option<Self> {
        let path = Self::checkpoint_path(db_path, namespace);
        if path.exists() {
            std::fs::read_to_string(&path)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
        } else {
            None
        }
    }

    fn save(&mut self, db_path: &str) -> Result<()> {
        self.updated_at = chrono::Utc::now().to_rfc3339();
        let path = Self::checkpoint_path(db_path, &self.namespace);
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, json)?;
        Ok(())
    }

    fn delete(db_path: &str, namespace: &str) {
        let path = Self::checkpoint_path(db_path, namespace);
        let _ = std::fs::remove_file(path);
    }

    fn mark_indexed(&mut self, file_path: &Path) {
        self.indexed_files
            .insert(file_path.to_string_lossy().to_string());
    }

    fn is_indexed(&self, file_path: &Path) -> bool {
        self.indexed_files
            .contains(&file_path.to_string_lossy().to_string())
    }
}

/// Configuration for batch indexing operation
struct BatchIndexConfig {
    path: PathBuf,
    namespace: Option<String>,
    recursive: bool,
    glob_pattern: Option<String>,
    max_depth: usize,
    db_path: String,
    preprocess: bool,
    /// Sanitize timestamps/UUIDs/session IDs (default: false = preserve for temporal queries)
    sanitize_metadata: bool,
    slice_mode: SliceMode,
    dedup: bool,
    embedding_config: EmbeddingConfig,
    /// Show smart progress bar with calibration-based ETA
    show_progress: bool,
    /// Resume from checkpoint if interrupted
    resume: bool,
    /// Enable async pipeline mode for concurrent stages
    pipeline: bool,
    /// Number of files to process in parallel (1-16, ignored in pipeline mode)
    parallel: u8,
}

/// Result of indexing a single file (for parallel processing)
#[derive(Debug)]
#[allow(dead_code)]
enum FileIndexResult {
    /// File was indexed successfully
    Indexed {
        file_path: PathBuf,
        chunks: usize,
        file_bytes: u64,
    },
    /// File was skipped (duplicate content)
    Skipped { file_path: PathBuf, reason: String },
    /// File was skipped (already in checkpoint)
    SkippedResume { file_path: PathBuf },
    /// Indexing failed
    Failed { file_path: PathBuf, error: String },
}

/// Run batch indexing with optional pipeline mode for concurrent processing
async fn run_batch_index(config: BatchIndexConfig) -> Result<()> {
    let BatchIndexConfig {
        path,
        namespace,
        recursive,
        glob_pattern,
        max_depth,
        db_path,
        preprocess,
        sanitize_metadata,
        slice_mode,
        dedup,
        embedding_config,
        show_progress,
        resume,
        pipeline,
        parallel,
    } = config;
    // Expand and canonicalize path - canonicalize validates path exists and resolves symlinks
    let expanded = shellexpand::tilde(path.to_str().unwrap_or("")).to_string();
    // nosemgrep: rust.actix.path-traversal.tainted-path.tainted-path
    let canonical = Path::new(&expanded).canonicalize()?;

    // Collect files
    let files = collect_files(&canonical, recursive, glob_pattern.as_deref(), max_depth)?;
    let total = files.len();

    if total == 0 {
        eprintln!("No files found matching criteria");
        return Ok(());
    }

    let mode_name = match slice_mode {
        SliceMode::Onion => "onion (hierarchical, 4 layers)",
        SliceMode::OnionFast => "onion-fast (outer+core, 2 layers)",
        SliceMode::Flat => "flat (traditional chunks)",
    };

    // Initialize progress tracker if --progress flag is set
    let tracker = if show_progress {
        let t = IndexProgressTracker::pre_scan(&files);
        t.display_pre_scan();
        Some(t)
    } else {
        eprintln!("Found {} files to index (slice mode: {})", total, mode_name);
        if preprocess {
            eprintln!("Preprocessing enabled: filtering tool artifacts, CLI output, and metadata");
        }
        if dedup {
            eprintln!("Deduplication enabled: skipping files with identical content");
        }
        None
    };

    // Initialize RAG pipeline - db_path is from CLI args or config, validated at load time
    let expanded_db = shellexpand::tilde(&db_path).to_string();
    // nosemgrep: rust.actix.path-traversal.tainted-path.tainted-path
    let db_dir = Path::new(&expanded_db);
    if let Some(parent) = db_dir.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Use lance-only storage for CLI - smaller cache for CLI use
    let embedding_client = Arc::new(Mutex::new(EmbeddingClient::new(&embedding_config).await?));
    let storage = Arc::new(StorageManager::new_lance_only(&expanded_db).await?);

    let ns_name = namespace.as_deref().unwrap_or("rag");

    // Pipeline mode: concurrent stages with channels
    if pipeline {
        if preprocess {
            eprintln!("Warning: --preprocess is not supported in pipeline mode (ignoring)");
        }
        if resume {
            eprintln!("Warning: --resume is not supported in pipeline mode (ignoring)");
        }
        if show_progress {
            eprintln!("Warning: --progress is not supported in pipeline mode (ignoring)");
        }

        eprintln!(
            "Pipeline mode: {} files, slice mode: {:?}",
            total, slice_mode
        );
        eprintln!("Running concurrent stages: reader -> chunker -> embedder -> storage");

        let pipeline_config = rmcp_memex::PipelineConfig {
            slice_mode,
            dedup_enabled: dedup,
            ..Default::default()
        };

        let result = rmcp_memex::run_pipeline(
            files,
            ns_name.to_string(),
            storage,
            embedding_client,
            pipeline_config,
        )
        .await?;

        eprintln!();
        eprintln!("Pipeline complete:");
        eprintln!("  Files read:        {}", result.stats.files_read);
        if result.stats.files_skipped > 0 {
            eprintln!("  Files skipped:     {}", result.stats.files_skipped);
        }
        eprintln!("  Chunks created:    {}", result.stats.chunks_created);
        eprintln!("  Chunks embedded:   {}", result.stats.chunks_embedded);
        eprintln!("  Chunks stored:     {}", result.stats.chunks_stored);
        if result.stats.errors > 0 {
            eprintln!("  Errors:            {}", result.stats.errors);
        }
        eprintln!("  Namespace:         {}", ns_name);
        eprintln!("  DB path:           {}", expanded_db);

        return Ok(());
    }

    // Standard (non-pipeline) mode with parallel file processing
    let rag = Arc::new(RAGPipeline::new(embedding_client, storage).await?);

    // Note: preprocessing currently uses flat mode
    let effective_mode = if preprocess {
        SliceMode::Flat
    } else {
        slice_mode
    };

    // Initialize checkpoint for resume capability (wrapped for thread-safe access)
    let checkpoint = if resume {
        if let Some(cp) = IndexCheckpoint::load(&db_path, ns_name) {
            let resumed_count = cp.indexed_files.len();
            eprintln!(
                "Resuming from checkpoint: {} files already indexed",
                resumed_count
            );
            Arc::new(Mutex::new(cp))
        } else {
            Arc::new(Mutex::new(IndexCheckpoint::new(ns_name)))
        }
    } else {
        // Clean start - remove any stale checkpoint
        IndexCheckpoint::delete(&db_path, ns_name);
        Arc::new(Mutex::new(IndexCheckpoint::new(ns_name)))
    };

    // Atomic counters for thread-safe progress tracking
    let indexed_count = Arc::new(AtomicUsize::new(0));
    let skipped_count = Arc::new(AtomicUsize::new(0));
    let skipped_resume_count = Arc::new(AtomicUsize::new(0));
    let failed_count = Arc::new(AtomicUsize::new(0));
    let total_chunks_count = Arc::new(AtomicUsize::new(0));
    let processed_count = Arc::new(AtomicUsize::new(0));

    // Semaphore to limit concurrent file processing
    let semaphore = Arc::new(Semaphore::new(parallel as usize));

    // Get embedder model name for calibration display
    let embedder_model = embedding_config
        .providers
        .first()
        .map(|p| p.model.clone())
        .unwrap_or_else(|| "unknown".to_string());

    // Flag to track if calibration is complete (for progress bar)
    let calibration_done = Arc::new(AtomicBool::new(false));

    // Wrap tracker for shared access
    let tracker = tracker.map(|t| Arc::new(Mutex::new(t)));

    // Start calibration if progress mode
    if let Some(ref t) = tracker {
        t.lock().await.start_calibration();
    }

    // Create task handles for parallel processing
    let mut handles = Vec::with_capacity(files.len());

    for file_path in files.into_iter() {
        // Clone shared resources for this task
        let semaphore = Arc::clone(&semaphore);
        let rag = Arc::clone(&rag);
        let checkpoint = Arc::clone(&checkpoint);
        let tracker = tracker.clone();
        let indexed_count = Arc::clone(&indexed_count);
        let skipped_count = Arc::clone(&skipped_count);
        let skipped_resume_count = Arc::clone(&skipped_resume_count);
        let failed_count = Arc::clone(&failed_count);
        let total_chunks_count = Arc::clone(&total_chunks_count);
        let processed_count = Arc::clone(&processed_count);
        let calibration_done = Arc::clone(&calibration_done);
        let db_path = db_path.clone();
        let ns = namespace.clone();
        let canonical = canonical.clone();
        let embedder_model = embedder_model.clone();
        let _ns_name = ns_name.to_string();

        let handle = tokio::spawn(async move {
            // Acquire semaphore permit to limit concurrency
            let _permit = semaphore.acquire().await.expect("semaphore closed");

            let display_path = file_path
                .strip_prefix(&canonical)
                .unwrap_or(&file_path)
                .display()
                .to_string();

            // Check if file already indexed (resume mode)
            if resume {
                let cp = checkpoint.lock().await;
                if cp.is_indexed(&file_path) {
                    drop(cp);
                    skipped_resume_count.fetch_add(1, Ordering::SeqCst);
                    processed_count.fetch_add(1, Ordering::SeqCst);
                    if let Some(ref t) = tracker {
                        t.lock().await.file_skipped();
                    }
                    return FileIndexResult::SkippedResume { file_path };
                }
            }

            // Get file size for calibration
            let file_bytes = std::fs::metadata(&file_path).map(|m| m.len()).unwrap_or(0);

            // Update progress display
            let current_processed = processed_count.load(Ordering::SeqCst);
            if let Some(ref t) = tracker {
                t.lock().await.set_message(&display_path);
            } else {
                let progress = format!("[{}/{}]", current_processed + 1, total);
                eprintln!("{} Indexing {}... ", progress, display_path);
            }

            // Build preprocessing config
            let preprocess_config = PreprocessingConfig {
                remove_metadata: sanitize_metadata,
                ..Default::default()
            };

            let result = if dedup {
                // Use dedup-enabled indexing
                if preprocess {
                    rag.index_document_with_preprocessing_and_dedup(
                        &file_path,
                        ns.as_deref(),
                        preprocess_config,
                    )
                    .await
                } else {
                    rag.index_document_with_dedup(&file_path, ns.as_deref(), effective_mode)
                        .await
                }
            } else {
                // Use original indexing without dedup (convert to IndexResult-like outcome)
                if preprocess {
                    rag.index_document_with_preprocessing(
                        &file_path,
                        ns.as_deref(),
                        preprocess_config,
                    )
                    .await
                    .map(|()| rmcp_memex::IndexResult::Indexed {
                        chunks_indexed: (file_bytes as usize / 500).max(1),
                        content_hash: String::new(),
                    })
                } else {
                    rag.index_document_with_mode(&file_path, ns.as_deref(), effective_mode)
                        .await
                        .map(|()| rmcp_memex::IndexResult::Indexed {
                            chunks_indexed: (file_bytes as usize / 500).max(1),
                            content_hash: String::new(),
                        })
                }
            };

            let file_result = match result {
                Ok(rmcp_memex::IndexResult::Indexed { chunks_indexed, .. }) => {
                    // Handle calibration on first completed file
                    if !calibration_done.swap(true, Ordering::SeqCst)
                        && let Some(ref t) = tracker
                    {
                        let mut guard = t.lock().await;
                        guard.finish_calibration(chunks_indexed, &embedder_model);
                        guard.adjust_estimate(file_bytes, chunks_indexed);
                        guard.start_progress_bar();
                    }

                    indexed_count.fetch_add(1, Ordering::SeqCst);
                    total_chunks_count.fetch_add(chunks_indexed, Ordering::SeqCst);

                    if let Some(ref t) = tracker {
                        t.lock().await.file_indexed(chunks_indexed);
                    } else {
                        eprintln!("  -> {} done ({} chunks)", display_path, chunks_indexed);
                    }

                    // Update checkpoint
                    if resume {
                        let mut cp = checkpoint.lock().await;
                        cp.mark_indexed(&file_path);
                        let _ = cp.save(&db_path);
                    }

                    FileIndexResult::Indexed {
                        file_path,
                        chunks: chunks_indexed,
                        file_bytes,
                    }
                }
                Ok(rmcp_memex::IndexResult::Skipped { reason, .. }) => {
                    // Handle calibration if this was the first file
                    if !calibration_done.swap(true, Ordering::SeqCst)
                        && let Some(ref t) = tracker
                    {
                        let mut guard = t.lock().await;
                        guard.finish_calibration(0, &embedder_model);
                        guard.start_progress_bar();
                    }

                    skipped_count.fetch_add(1, Ordering::SeqCst);

                    if let Some(ref t) = tracker {
                        t.lock().await.file_skipped();
                    } else {
                        eprintln!("  -> {} SKIPPED ({})", display_path, reason);
                    }

                    // Mark as indexed even if skipped (content exists)
                    if resume {
                        let mut cp = checkpoint.lock().await;
                        cp.mark_indexed(&file_path);
                        let _ = cp.save(&db_path);
                    }

                    FileIndexResult::Skipped { file_path, reason }
                }
                Err(e) => {
                    // Handle calibration if this was the first file
                    if !calibration_done.swap(true, Ordering::SeqCst)
                        && let Some(ref t) = tracker
                    {
                        let mut guard = t.lock().await;
                        guard.finish_calibration(0, &embedder_model);
                        guard.start_progress_bar();
                    }

                    failed_count.fetch_add(1, Ordering::SeqCst);

                    if let Some(ref t) = tracker {
                        t.lock().await.file_failed();
                    } else {
                        eprintln!("  -> {} FAILED: {}", display_path, e);
                    }

                    FileIndexResult::Failed {
                        file_path,
                        error: e.to_string(),
                    }
                }
            };

            processed_count.fetch_add(1, Ordering::SeqCst);
            file_result
        });

        handles.push(handle);
    }

    // Wait for all tasks to complete
    let mut results = Vec::with_capacity(handles.len());
    for handle in handles {
        match handle.await {
            Ok(result) => results.push(result),
            Err(e) => {
                // Task panicked - count as failure
                failed_count.fetch_add(1, Ordering::SeqCst);
                eprintln!("Task panicked: {}", e);
            }
        }
    }

    // Get final counts from atomics
    let indexed = indexed_count.load(Ordering::SeqCst);
    let skipped = skipped_count.load(Ordering::SeqCst);
    let skipped_resume = skipped_resume_count.load(Ordering::SeqCst);
    let failed = failed_count.load(Ordering::SeqCst);
    let total_chunks = total_chunks_count.load(Ordering::SeqCst);

    // Display summary
    if let Some(ref t) = tracker {
        let mut guard = t.lock().await;
        guard.finish();
        guard.display_summary();
        if skipped_resume > 0 {
            eprintln!("  Skipped (resumed): {}", skipped_resume);
        }
    } else {
        eprintln!();
        eprintln!("Indexing complete:");
        eprintln!("  New chunks:        {}", total_chunks);
        eprintln!("  Files indexed:     {}", indexed);
        if dedup && skipped > 0 {
            eprintln!("  Skipped (duplicate): {}", skipped);
        }
        if skipped_resume > 0 {
            eprintln!("  Skipped (resumed): {}", skipped_resume);
        }
        if failed > 0 {
            eprintln!("  Failed:            {}", failed);
        }
        eprintln!("  Total processed:   {}", total);
        if let Some(ref ns) = namespace {
            eprintln!("  Namespace:         {}", ns);
        }
        eprintln!("  Slice mode:        {}", mode_name);
        eprintln!("  Parallel workers:  {}", parallel);
        eprintln!(
            "  Deduplication:     {}",
            if dedup { "enabled" } else { "disabled" }
        );
        eprintln!("  DB path:           {}", expanded_db);
    }

    // Clean up checkpoint on successful completion
    if resume && failed == 0 {
        IndexCheckpoint::delete(&db_path, ns_name);
        eprintln!("Checkpoint cleared (all files indexed successfully)");
    } else if resume && failed > 0 {
        eprintln!(
            "Checkpoint preserved ({} files failed - rerun with --resume to retry)",
            failed
        );
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Wizard { dry_run }) => {
            // Run TUI wizard (no logging setup needed - TUI handles terminal)
            let wizard_config = WizardConfig {
                config_path: cli.config,
                dry_run,
            };
            run_wizard(wizard_config)
        }
        Some(Commands::Index {
            path,
            namespace,
            recursive,
            glob,
            max_depth,
            preprocess,
            sanitize_metadata,
            slice_mode,
            dedup,
            progress,
            resume,
            pipeline,
            parallel,
        }) => {
            // Get db_path and cache_mb from config or defaults
            let (file_cfg, config_path) = load_or_discover_config(cli.config.as_deref())?;
            if let Some(ref path) = config_path {
                eprintln!("Using config: {}", path);
            }

            // Extract embedding config before any moves
            let embedding_config = file_cfg.to_embedding_config();

            let db_path = cli
                .db_path
                .or(file_cfg.db_path)
                .unwrap_or_else(|| "~/.rmcp-servers/rmcp-memex/lancedb".to_string());
            let _cache_mb = cli.cache_mb.or(file_cfg.cache_mb).unwrap_or(4096);
            // CLI flag overrides file config
            let preprocess = preprocess || file_cfg.preprocessing_enabled.unwrap_or(false);
            let slice_mode: SliceMode = slice_mode.parse().unwrap_or_default();

            run_batch_index(BatchIndexConfig {
                path,
                namespace,
                recursive,
                glob_pattern: glob,
                max_depth,
                db_path,
                preprocess,
                sanitize_metadata,
                slice_mode,
                dedup,
                embedding_config,
                show_progress: progress,
                resume,
                pipeline,
                parallel,
            })
            .await
        }
        Some(Commands::Overview { namespace, json }) => {
            let (file_cfg, config_path) = load_or_discover_config(cli.config.as_deref())?;
            if let Some(ref path) = config_path {
                eprintln!("Using config: {}", path);
            }

            let db_path = cli
                .db_path
                .or(file_cfg.db_path)
                .unwrap_or_else(|| "~/.rmcp-servers/rmcp-memex/lancedb".to_string());
            let db_path = shellexpand::tilde(&db_path).to_string();

            run_overview(namespace, json, db_path).await
        }
        Some(Commands::Dive {
            namespace,
            query,
            limit,
            verbose,
            json,
        }) => {
            let (file_cfg, config_path) = load_or_discover_config(cli.config.as_deref())?;
            if let Some(ref path) = config_path {
                eprintln!("Using config: {}", path);
            }

            let embedding_config = file_cfg.to_embedding_config();

            let db_path = cli
                .db_path
                .or(file_cfg.db_path)
                .unwrap_or_else(|| "~/.rmcp-servers/rmcp-memex/lancedb".to_string());
            let db_path = shellexpand::tilde(&db_path).to_string();

            run_dive(
                namespace,
                query,
                limit,
                verbose,
                json,
                db_path,
                &embedding_config,
            )
            .await
        }
        Some(Commands::Search {
            namespace,
            query,
            limit,
            json,
            deep,
            layer,
            mode,
            auto_route,
            ..
        }) => {
            let (file_cfg, config_path) = load_or_discover_config(cli.config.as_deref())?;
            if let Some(ref path) = config_path {
                eprintln!("Using config: {}", path);
            }

            let embedding_config = file_cfg.to_embedding_config();

            let db_path = cli
                .db_path
                .or(file_cfg.db_path)
                .unwrap_or_else(|| "~/.rmcp-servers/rmcp-memex/lancedb".to_string());
            let db_path = shellexpand::tilde(&db_path).to_string();

            // Parse layer filter
            let layer_filter = if deep {
                None // All layers
            } else if let Some(layer_str) = layer {
                match layer_str.as_str() {
                    "outer" => Some(SliceLayer::Outer),
                    "middle" => Some(SliceLayer::Middle),
                    "inner" => Some(SliceLayer::Inner),
                    "core" => Some(SliceLayer::Core),
                    _ => None,
                }
            } else {
                None // Default: all layers (for backward compatibility)
            };

            // Parse search mode - use QueryRouter if --auto-route enabled
            let search_mode: SearchMode = if auto_route {
                let router = QueryRouter::new();
                let decision = router.route(&query);
                eprintln!(
                    "Query intent: {} (confidence: {:.2})",
                    decision.intent, decision.confidence
                );
                if let Some(ref suggestion) = decision.loctree_suggestion {
                    eprintln!(
                        "Consider: {} - {}",
                        suggestion.command, suggestion.explanation
                    );
                }
                if let Some(ref hints) = decision.temporal_hints
                    && !hints.date_references.is_empty()
                {
                    eprintln!("Date references: {}", hints.date_references.join(", "));
                }
                match decision.recommended_mode.mode {
                    SearchModeRecommendation::Vector => SearchMode::Vector,
                    SearchModeRecommendation::Bm25 => SearchMode::Keyword,
                    SearchModeRecommendation::Hybrid => SearchMode::Hybrid,
                }
            } else {
                mode.parse().unwrap_or_default()
            };

            run_search(
                namespace,
                query,
                limit,
                json,
                db_path,
                layer_filter,
                search_mode,
                &embedding_config,
            )
            .await
        }
        Some(Commands::Expand {
            namespace,
            id,
            json,
        }) => {
            let (file_cfg, config_path) = load_or_discover_config(cli.config.as_deref())?;
            if let Some(ref path) = config_path {
                eprintln!("Using config: {}", path);
            }

            let embedding_config = file_cfg.to_embedding_config();

            let db_path = cli
                .db_path
                .or(file_cfg.db_path)
                .unwrap_or_else(|| "~/.rmcp-servers/rmcp-memex/lancedb".to_string());
            let db_path = shellexpand::tilde(&db_path).to_string();

            run_expand(namespace, id, json, db_path, &embedding_config).await
        }
        Some(Commands::Get {
            namespace,
            id,
            json,
        }) => {
            let (file_cfg, config_path) = load_or_discover_config(cli.config.as_deref())?;
            if let Some(ref path) = config_path {
                eprintln!("Using config: {}", path);
            }

            let embedding_config = file_cfg.to_embedding_config();

            let db_path = cli
                .db_path
                .or(file_cfg.db_path)
                .unwrap_or_else(|| "~/.rmcp-servers/rmcp-memex/lancedb".to_string());
            let db_path = shellexpand::tilde(&db_path).to_string();

            run_get(namespace, id, json, db_path, &embedding_config).await
        }
        Some(Commands::RagSearch {
            query,
            limit,
            namespace,
            json,
        }) => {
            let (file_cfg, config_path) = load_or_discover_config(cli.config.as_deref())?;
            if let Some(ref path) = config_path {
                eprintln!("Using config: {}", path);
            }

            let embedding_config = file_cfg.to_embedding_config();

            let db_path = cli
                .db_path
                .or(file_cfg.db_path)
                .unwrap_or_else(|| "~/.rmcp-servers/rmcp-memex/lancedb".to_string());
            let db_path = shellexpand::tilde(&db_path).to_string();

            run_rag_search(query, limit, namespace, json, db_path, &embedding_config).await
        }
        Some(Commands::Namespaces { stats, json }) => {
            let (file_cfg, config_path) = load_or_discover_config(cli.config.as_deref())?;
            if let Some(ref path) = config_path {
                eprintln!("Using config: {}", path);
            }

            let db_path = cli
                .db_path
                .or(file_cfg.db_path)
                .unwrap_or_else(|| "~/.rmcp-servers/rmcp-memex/lancedb".to_string());
            let db_path = shellexpand::tilde(&db_path).to_string();

            run_list_namespaces(stats, json, db_path).await
        }
        Some(Commands::Export {
            namespace,
            output,
            include_embeddings,
        }) => {
            let (file_cfg, config_path) = load_or_discover_config(cli.config.as_deref())?;
            if let Some(ref path) = config_path {
                eprintln!("Using config: {}", path);
            }

            let db_path = cli
                .db_path
                .or(file_cfg.db_path)
                .unwrap_or_else(|| "~/.rmcp-servers/rmcp-memex/lancedb".to_string());
            let db_path = shellexpand::tilde(&db_path).to_string();

            run_export(namespace, output, include_embeddings, db_path).await
        }
        Some(Commands::Upsert {
            namespace,
            id,
            text,
            metadata,
        }) => {
            let (file_cfg, config_path) = load_or_discover_config(cli.config.as_deref())?;
            if let Some(ref path) = config_path {
                eprintln!("Using config: {}", path);
            }

            let embedding_config = file_cfg.to_embedding_config();

            let db_path = cli
                .db_path
                .or(file_cfg.db_path)
                .unwrap_or_else(|| "~/.rmcp-servers/rmcp-memex/lancedb".to_string());
            let db_path = shellexpand::tilde(&db_path).to_string();

            // Get text from argument or stdin
            let content = match text {
                Some(t) => t,
                None => {
                    use std::io::Read;
                    let mut buffer = String::new();
                    std::io::stdin().read_to_string(&mut buffer)?;
                    buffer
                }
            };

            if content.trim().is_empty() {
                return Err(anyhow::anyhow!(
                    "No text provided (use --text or pipe to stdin)"
                ));
            }

            // Parse metadata JSON
            let meta: serde_json::Value = serde_json::from_str(&metadata)
                .map_err(|e| anyhow::anyhow!("Invalid metadata JSON: {}", e))?;

            // Initialize RAG pipeline
            let embedding_client =
                Arc::new(Mutex::new(EmbeddingClient::new(&embedding_config).await?));
            let storage = Arc::new(StorageManager::new_lance_only(&db_path).await?);
            let rag = RAGPipeline::new(embedding_client, storage).await?;

            // Upsert
            rag.memory_upsert(&namespace, id.clone(), content.clone(), meta)
                .await?;

            eprintln!("✓ Upserted chunk '{}' to namespace '{}'", id, namespace);
            eprintln!("  Text: {} chars", content.len());
            eprintln!("  DB: {}", db_path);

            Ok(())
        }
        Some(Commands::Optimize) => {
            let (file_cfg, config_path) = load_or_discover_config(cli.config.as_deref())?;
            if let Some(ref path) = config_path {
                eprintln!("Using config: {}", path);
            }

            let db_path = cli
                .db_path
                .or(file_cfg.db_path)
                .unwrap_or_else(|| "~/.rmcp-servers/rmcp-memex/lancedb".to_string());
            let db_path = shellexpand::tilde(&db_path).to_string();

            eprintln!("Optimizing database at: {}", db_path);
            eprintln!("This may take a while for large databases...");

            let storage = StorageManager::new_lance_only(&db_path).await?;
            let stats = storage.optimize().await?;

            eprintln!();
            eprintln!("Optimization complete:");
            if let Some(ref c) = stats.compaction {
                eprintln!("  Files rewritten:    {}", c.files_removed);
                eprintln!("  Files added:        {}", c.files_added);
                eprintln!("  Fragments removed:  {}", c.fragments_removed);
                eprintln!("  Fragments added:    {}", c.fragments_added);
            }
            if let Some(ref p) = stats.prune {
                eprintln!("  Versions removed:   {}", p.old_versions);
                eprintln!("  Bytes freed:        {}", p.bytes_removed);
            }

            Ok(())
        }
        Some(Commands::Compact) => {
            let (file_cfg, config_path) = load_or_discover_config(cli.config.as_deref())?;
            if let Some(ref path) = config_path {
                eprintln!("Using config: {}", path);
            }

            let db_path = cli
                .db_path
                .or(file_cfg.db_path)
                .unwrap_or_else(|| "~/.rmcp-servers/rmcp-memex/lancedb".to_string());
            let db_path = shellexpand::tilde(&db_path).to_string();

            eprintln!("Compacting database at: {}", db_path);

            let storage = StorageManager::new_lance_only(&db_path).await?;
            let stats = storage.compact().await?;

            eprintln!();
            eprintln!("Compaction complete:");
            if let Some(ref c) = stats.compaction {
                eprintln!("  Files rewritten:    {}", c.files_removed);
                eprintln!("  Files added:        {}", c.files_added);
                eprintln!("  Fragments removed:  {}", c.fragments_removed);
                eprintln!("  Fragments added:    {}", c.fragments_added);
            } else {
                eprintln!("  No compaction needed");
            }

            Ok(())
        }
        Some(Commands::Cleanup { older_than_days }) => {
            let (file_cfg, config_path) = load_or_discover_config(cli.config.as_deref())?;
            if let Some(ref path) = config_path {
                eprintln!("Using config: {}", path);
            }

            let db_path = cli
                .db_path
                .or(file_cfg.db_path)
                .unwrap_or_else(|| "~/.rmcp-servers/rmcp-memex/lancedb".to_string());
            let db_path = shellexpand::tilde(&db_path).to_string();

            eprintln!(
                "Cleaning up versions older than {} days at: {}",
                older_than_days, db_path
            );

            let storage = StorageManager::new_lance_only(&db_path).await?;
            let stats = storage.cleanup(Some(older_than_days)).await?;

            eprintln!();
            eprintln!("Cleanup complete:");
            if let Some(ref p) = stats.prune {
                eprintln!("  Versions removed:   {}", p.old_versions);
                eprintln!("  Bytes freed:        {}", p.bytes_removed);
            } else {
                eprintln!("  No old versions to remove");
            }

            Ok(())
        }
        Some(Commands::Stats) => {
            let (file_cfg, config_path) = load_or_discover_config(cli.config.as_deref())?;
            if let Some(ref path) = config_path {
                eprintln!("Using config: {}", path);
            }

            let db_path = cli
                .db_path
                .or(file_cfg.db_path)
                .unwrap_or_else(|| "~/.rmcp-servers/rmcp-memex/lancedb".to_string());
            let db_path = shellexpand::tilde(&db_path).to_string();

            let storage = StorageManager::new_lance_only(&db_path).await?;
            let stats = storage.stats().await?;

            eprintln!("Database Statistics:");
            eprintln!("  Table:       {}", stats.table_name);
            eprintln!("  Path:        {}", stats.db_path);
            eprintln!("  Total rows:  {}", stats.row_count);
            eprintln!("  Versions:    {}", stats.version_count);

            // Also output as JSON for scripting
            println!("{}", serde_json::to_string_pretty(&stats)?);

            Ok(())
        }
        Some(Commands::Serve) | None => {
            // Run MCP server (and optionally HTTP/SSE server)
            let http_port = cli.http_port;
            let http_only = cli.http_only;

            // Validate http_only requires http_port
            if http_only && http_port.is_none() {
                return Err(anyhow::anyhow!(
                    "--http-only requires --http-port to be set"
                ));
            }

            let config = cli.into_server_config()?;

            // Send logs to stderr to keep stdout clean for JSON-RPC.
            let subscriber = FmtSubscriber::builder()
                .with_max_level(config.log_level)
                .with_writer(std::io::stderr)
                .with_ansi(false)
                .finish();
            tracing::subscriber::set_global_default(subscriber)?;

            info!("Starting RMCP Memex");
            info!("Features (informational): {:?}", config.features);
            info!("Cache: {}MB", config.cache_mb);
            info!("DB Path: {}", config.db_path);

            // Create MCP server (also initializes RAGPipeline)
            let server = create_server(config).await?;

            // HTTP-only mode: run HTTP server as main process (blocking)
            if http_only {
                let port = http_port.expect("validated above");
                let rag = server.rag();
                info!("Starting HTTP-only server on port {} (no MCP stdio)", port);
                rmcp_memex::http::start_server(rag, port).await?;
                return Ok(());
            }

            // If HTTP port specified, start HTTP/SSE server in background
            if let Some(port) = http_port {
                let rag = server.rag();
                info!("Starting HTTP/SSE server on port {}", port);
                tokio::spawn(async move {
                    if let Err(e) = rmcp_memex::http::start_server(rag, port).await {
                        tracing::error!("HTTP server error: {}", e);
                    }
                });
            }

            // Run MCP stdio server (blocking)
            server.run_stdio().await
        }
    }
}
