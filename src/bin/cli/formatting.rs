use anyhow::Result;

use rust_memex::{HybridSearchResult, SearchMode, SliceLayer, path_utils};

#[allow(dead_code)]
fn parse_features(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

/// Standard config discovery locations (in priority order)
#[allow(dead_code)]
const CONFIG_SEARCH_PATHS: &[&str] = &[
    "~/.rmcp-servers/rmcp-memex/config.toml",
    "~/.config/rmcp-memex/config.toml",
    "~/.rmcp_servers/rmcp_memex/config.toml", // legacy underscore path
];

/// Discover config file from standard locations
#[allow(dead_code)]
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

#[allow(dead_code)]
fn load_file_config(path: &str) -> Result<FileConfig> {
    let (_canonical, contents) = path_utils::safe_read_to_string(path)
        .map_err(|e| anyhow::anyhow!("Cannot load config '{}': {}", path, e))?;
    toml::from_str(&contents).map_err(Into::into)
}

/// Load config from explicit path or discover from standard locations
#[allow(dead_code)]
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

use crate::cli::config::*;
/// Format and display search results (human-readable)
pub fn display_search_results(
    query: &str,
    namespace: Option<&str>,
    results: &[rust_memex::SearchResult],
    layer_filter: Option<SliceLayer>,
) {
    let ns_display = namespace.unwrap_or("all namespaces");
    let layer_display = layer_filter
        .map(|l| format!(" (layer: {})", l.name()))
        .unwrap_or_default();

    println!(
        "\n-> Search Results for \"{}\" in [{}]{}\n",
        query, ns_display, layer_display
    );

    if results.is_empty() {
        println!("No results found.");
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

        println!(
            "{}. [{:.2}] {} {}",
            i + 1,
            result.score,
            result.namespace,
            layer_str
        );
        println!("   \"{}{ellipsis}\"", preview);
        println!("   ID: {}", result.id);
        if !result.keywords.is_empty() {
            println!("   Keywords: {}", result.keywords.join(", "));
        }
        if result.can_expand() {
            println!("   [expandable: {} children]", result.children_ids.len());
        }
        if !result.metadata.is_null() && result.metadata != serde_json::json!({}) {
            println!("   Metadata: {}", result.metadata);
        }
        println!();
    }
}

/// Format search results as JSON
pub fn json_search_results(
    query: &str,
    namespace: Option<&str>,
    results: &[rust_memex::SearchResult],
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
pub fn display_hybrid_search_results(
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

    println!(
        "\n-> Search Results for \"{}\" in [{}]{} [mode: {}]\n",
        query, ns_display, layer_display, mode_display
    );

    if results.is_empty() {
        println!("No results found.");
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

        println!(
            "{}. {} {} {}",
            i + 1,
            score_details,
            result.namespace,
            layer_str
        );
        println!("   \"{}{ellipsis}\"", preview);
        println!("   ID: {}", result.id);
        if !result.keywords.is_empty() {
            println!("   Keywords: {}", result.keywords.join(", "));
        }
        if !result.children_ids.is_empty() {
            println!("   [expandable: {} children]", result.children_ids.len());
        }
        if !result.metadata.is_null() && result.metadata != serde_json::json!({}) {
            println!("   Metadata: {}", result.metadata);
        }
        println!();
    }
}

/// Format hybrid search results as JSON
pub fn json_hybrid_search_results(
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
