use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{Level, info};
use tracing_subscriber::FmtSubscriber;
use walkdir::WalkDir;

use rmcp_memex::{
    EmbeddingClient, EmbeddingConfig, MlxConfig, NamespaceSecurityConfig, PreprocessingConfig,
    ProviderConfig, RAGPipeline, RerankerConfig, ServerConfig, SliceLayer, SliceMode,
    StorageManager, WizardConfig, run_stdio_server, run_wizard,
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

        /// Enable preprocessing to filter noise (tool artifacts, CLI output, metadata)
        /// before indexing. Reduces vector storage size and improves search quality.
        #[arg(long, short = 'p')]
        preprocess: bool,

        /// Slicing mode for document chunking:
        /// - "onion" (default): Hierarchical slices (outer/middle/inner/core) for efficient context
        /// - "flat": Traditional fixed-size chunks with overlap
        #[arg(long, short = 's', default_value = "onion", value_parser = ["onion", "flat"])]
        slice_mode: String,

        /// Enable exact-match deduplication (default: enabled).
        /// Skips indexing files whose content already exists in the namespace.
        /// Uses SHA256 hash of original content before any preprocessing.
        #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
        dedup: bool,
    },

    /// Semantic search within a namespace
    Search {
        /// Namespace to search in
        #[arg(long, short = 'n', required = true)]
        namespace: String,

        /// Search query text
        #[arg(long, short = 'q', required = true)]
        query: String,

        /// Maximum number of results to return
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

/// Run semantic search within a namespace
async fn run_search(
    namespace: String,
    query: String,
    limit: usize,
    json_output: bool,
    db_path: String,
    layer_filter: Option<SliceLayer>,
    embedding_config: &EmbeddingConfig,
) -> Result<()> {
    let embedding_client = Arc::new(Mutex::new(EmbeddingClient::new(embedding_config).await?));
    let storage = Arc::new(StorageManager::new_lance_only(&db_path).await?);
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

/// Configuration for batch indexing operation
struct BatchIndexConfig {
    path: PathBuf,
    namespace: Option<String>,
    recursive: bool,
    glob_pattern: Option<String>,
    max_depth: usize,
    db_path: String,
    preprocess: bool,
    slice_mode: SliceMode,
    dedup: bool,
    embedding_config: EmbeddingConfig,
}

/// Run batch indexing
async fn run_batch_index(config: BatchIndexConfig) -> Result<()> {
    let BatchIndexConfig {
        path,
        namespace,
        recursive,
        glob_pattern,
        max_depth,
        db_path,
        preprocess,
        slice_mode,
        dedup,
        embedding_config,
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
        SliceMode::Onion => "onion (hierarchical)",
        SliceMode::Flat => "flat (traditional)",
    };
    eprintln!("Found {} files to index (slice mode: {})", total, mode_name);
    if preprocess {
        eprintln!("Preprocessing enabled: filtering tool artifacts, CLI output, and metadata");
    }
    if dedup {
        eprintln!("Deduplication enabled: skipping files with identical content");
    }

    // Initialize RAG pipeline - db_path is from CLI args or config, validated at load time
    let expanded_db = shellexpand::tilde(&db_path).to_string();
    // nosemgrep: rust.actix.path-traversal.tainted-path.tainted-path
    let db_dir = Path::new(&expanded_db);
    if let Some(parent) = db_dir.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Use lance-only storage for CLI - no sled lock conflict with running server
    let embedding_client = Arc::new(Mutex::new(EmbeddingClient::new(&embedding_config).await?));
    let storage = Arc::new(StorageManager::new_lance_only(&expanded_db).await?);
    let rag = RAGPipeline::new(embedding_client, storage).await?;

    // Note: preprocessing currently uses flat mode
    let effective_mode = if preprocess {
        SliceMode::Flat
    } else {
        slice_mode
    };

    let ns = namespace.as_deref();
    let mut indexed = 0;
    let mut skipped = 0;
    let mut failed = 0;
    let mut total_chunks = 0;

    for (i, file_path) in files.iter().enumerate() {
        let progress = format!("[{}/{}]", i + 1, total);
        let display_path = file_path
            .strip_prefix(&canonical)
            .unwrap_or(file_path)
            .display();

        eprint!("{} Indexing {}... ", progress, display_path);

        if dedup {
            // Use dedup-enabled indexing
            let result = if preprocess {
                rag.index_document_with_preprocessing_and_dedup(
                    file_path,
                    ns,
                    PreprocessingConfig::default(),
                )
                .await
            } else {
                rag.index_document_with_dedup(file_path, ns, effective_mode)
                    .await
            };

            match result {
                Ok(rmcp_memex::IndexResult::Indexed { chunks_indexed, .. }) => {
                    eprintln!("done ({} chunks)", chunks_indexed);
                    indexed += 1;
                    total_chunks += chunks_indexed;
                }
                Ok(rmcp_memex::IndexResult::Skipped { reason, .. }) => {
                    eprintln!("SKIPPED ({})", reason);
                    skipped += 1;
                }
                Err(e) => {
                    eprintln!("FAILED: {}", e);
                    failed += 1;
                }
            }
        } else {
            // Use original indexing without dedup
            let result = if preprocess {
                rag.index_document_with_preprocessing(file_path, ns, PreprocessingConfig::default())
                    .await
            } else {
                rag.index_document_with_mode(file_path, ns, effective_mode)
                    .await
            };

            match result {
                Ok(()) => {
                    eprintln!("done");
                    indexed += 1;
                }
                Err(e) => {
                    eprintln!("FAILED: {}", e);
                    failed += 1;
                }
            }
        }
    }

    eprintln!();
    eprintln!("Indexing complete:");
    eprintln!("  New chunks:        {}", total_chunks);
    eprintln!("  Files indexed:     {}", indexed);
    if dedup && skipped > 0 {
        eprintln!("  Skipped (duplicate): {}", skipped);
    }
    if failed > 0 {
        eprintln!("  Failed:            {}", failed);
    }
    eprintln!("  Total processed:   {}", total);
    if let Some(ns) = ns {
        eprintln!("  Namespace:         {}", ns);
    }
    eprintln!("  Slice mode:        {}", mode_name);
    eprintln!(
        "  Deduplication:     {}",
        if dedup { "enabled" } else { "disabled" }
    );
    eprintln!("  DB path:           {}", expanded_db);

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
            slice_mode,
            dedup,
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
                slice_mode,
                dedup,
                embedding_config,
            })
            .await
        }
        Some(Commands::Search {
            namespace,
            query,
            limit,
            json,
            deep,
            layer,
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

            run_search(
                namespace,
                query,
                limit,
                json,
                db_path,
                layer_filter,
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
        Some(Commands::Serve) | None => {
            // Run MCP server
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

            run_stdio_server(config).await
        }
    }
}
