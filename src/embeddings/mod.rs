//! Universal embedding client with config-driven provider cascade.
//!
//! Supports any OpenAI-compatible embedding API (Ollama, vLLM, TEI, etc.)
//! Providers are tried in priority order until one responds.
//!
//! # Example config.toml
//! ```toml
//! [embeddings]
//! required_dimension = 4096
//! max_batch_chars = 32000
//! max_batch_items = 16
//!
//! [[embeddings.providers]]
//! name = "ollama-local"
//! base_url = "http://localhost:11434"
//! model = "qwen3-embedding:8b"
//! priority = 1
//!
//! [[embeddings.providers]]
//! name = "dragon"
//! base_url = "http://dragon:12345"
//! model = "Qwen/Qwen3-Embedding-4B"
//! priority = 2
//! ```

use anyhow::{Result, anyhow};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

// =============================================================================
// REQUEST/RESPONSE TYPES (OpenAI-compatible)
// =============================================================================

#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    input: Vec<String>,
    model: String,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct RerankRequest {
    query: String,
    documents: Vec<String>,
    model: String,
}

#[derive(Debug, Deserialize)]
struct RerankResponse {
    results: Vec<RerankResult>,
}

#[derive(Debug, Deserialize)]
struct RerankResult {
    index: usize,
    score: f32,
}

// =============================================================================
// PROVIDER CONFIGURATION
// =============================================================================

/// Single embedding provider configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ProviderConfig {
    /// Human-readable name for logging
    pub name: String,
    /// Base URL (e.g., "http://localhost:11434")
    pub base_url: String,
    /// Model name to use
    pub model: String,
    /// Priority (1 = highest, tried first)
    #[serde(default = "default_priority")]
    pub priority: u8,
    /// Embedding endpoint path (default: /v1/embeddings)
    #[serde(default = "default_embeddings_endpoint")]
    pub endpoint: String,
}

fn default_priority() -> u8 {
    10
}

fn default_embeddings_endpoint() -> String {
    "/v1/embeddings".to_string()
}

/// Reranker configuration (optional, separate from embedders)
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct RerankerConfig {
    /// Base URL for reranker service
    pub base_url: Option<String>,
    /// Model name
    pub model: Option<String>,
    /// Endpoint path (default: /v1/rerank)
    #[serde(default = "default_rerank_endpoint")]
    pub endpoint: String,
}

fn default_rerank_endpoint() -> String {
    "/v1/rerank".to_string()
}

fn default_dimension() -> usize {
    4096
}

fn default_max_batch_chars() -> usize {
    32000
}

fn default_max_batch_items() -> usize {
    16
}

/// Complete embedding configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EmbeddingConfig {
    /// Required vector dimension (mismatch corrupts database!)
    #[serde(default = "default_dimension")]
    pub required_dimension: usize,
    /// Maximum characters per embedding batch to avoid OOM (default: 32000)
    #[serde(default = "default_max_batch_chars")]
    pub max_batch_chars: usize,
    /// Maximum items per embedding batch (default: 16)
    #[serde(default = "default_max_batch_items")]
    pub max_batch_items: usize,
    /// List of providers to try in priority order
    #[serde(default)]
    pub providers: Vec<ProviderConfig>,
    /// Optional reranker configuration
    #[serde(default)]
    pub reranker: RerankerConfig,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            required_dimension: 4096,
            max_batch_chars: 32000,
            max_batch_items: 16,
            providers: vec![
                ProviderConfig {
                    name: "ollama-local".to_string(),
                    base_url: "http://localhost:11434".to_string(),
                    model: "qwen3-embedding:8b".to_string(),
                    priority: 1,
                    endpoint: default_embeddings_endpoint(),
                },
                ProviderConfig {
                    name: "dragon".to_string(),
                    base_url: "http://dragon:12345".to_string(),
                    model: "Qwen/Qwen3-Embedding-4B".to_string(),
                    priority: 2,
                    endpoint: default_embeddings_endpoint(),
                },
            ],
            reranker: RerankerConfig::default(),
        }
    }
}

// =============================================================================
// LEGACY CONFIG (backward compatibility)
// =============================================================================

/// Legacy MLX configuration - deprecated, use EmbeddingConfig instead
#[derive(Debug, Clone)]
pub struct MlxConfig {
    pub disabled: bool,
    pub local_port: u16,
    pub dragon_url: String,
    pub dragon_port: u16,
    pub embedder_model: String,
    pub reranker_model: String,
    pub reranker_port_offset: u16,
    pub max_batch_chars: usize,
    pub max_batch_items: usize,
}

impl Default for MlxConfig {
    fn default() -> Self {
        Self {
            disabled: false,
            local_port: 12345,
            dragon_url: "http://dragon".to_string(),
            dragon_port: 12345,
            embedder_model: "Qwen/Qwen3-Embedding-4B".to_string(),
            reranker_model: "Qwen/Qwen3-Reranker-4B".to_string(),
            reranker_port_offset: 1,
            max_batch_chars: 32000,
            max_batch_items: 16,
        }
    }
}

impl MlxConfig {
    /// Create config from environment variables (legacy support)
    pub fn from_env() -> Self {
        let disabled = std::env::var("DISABLE_MLX")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        let local_port = std::env::var("EMBEDDER_PORT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(12345);

        let dragon_url =
            std::env::var("DRAGON_BASE_URL").unwrap_or_else(|_| "http://dragon".to_string());

        let dragon_port = std::env::var("DRAGON_EMBEDDER_PORT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(local_port);

        let reranker_port_offset = std::env::var("RERANKER_PORT")
            .ok()
            .and_then(|s| s.parse::<u16>().ok())
            .map(|rp| rp.saturating_sub(local_port))
            .unwrap_or(1);

        let embedder_model = std::env::var("EMBEDDER_MODEL")
            .unwrap_or_else(|_| "Qwen/Qwen3-Embedding-4B".to_string());

        let reranker_model = std::env::var("RERANKER_MODEL")
            .unwrap_or_else(|_| "Qwen/Qwen3-Reranker-4B".to_string());

        let max_batch_chars = std::env::var("MLX_MAX_BATCH_CHARS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(32000);

        let max_batch_items = std::env::var("MLX_MAX_BATCH_ITEMS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(16);

        Self {
            disabled,
            local_port,
            dragon_url,
            dragon_port,
            embedder_model,
            reranker_model,
            reranker_port_offset,
            max_batch_chars,
            max_batch_items,
        }
    }

    /// Merge with values from file config
    #[allow(clippy::too_many_arguments)]
    pub fn merge_file_config(
        &mut self,
        disabled: Option<bool>,
        local_port: Option<u16>,
        dragon_url: Option<String>,
        dragon_port: Option<u16>,
        embedder_model: Option<String>,
        reranker_model: Option<String>,
        reranker_port_offset: Option<u16>,
    ) {
        if let Some(v) = disabled {
            self.disabled = v;
        }
        if let Some(v) = local_port {
            self.local_port = v;
        }
        if let Some(v) = dragon_url {
            self.dragon_url = v;
        }
        if let Some(v) = dragon_port {
            self.dragon_port = v;
        }
        if let Some(v) = embedder_model {
            self.embedder_model = v;
        }
        if let Some(v) = reranker_model {
            self.reranker_model = v;
        }
        if let Some(v) = reranker_port_offset {
            self.reranker_port_offset = v;
        }
    }

    /// Convert legacy config to new EmbeddingConfig
    pub fn to_embedding_config(&self) -> EmbeddingConfig {
        let reranker_port = self.local_port + self.reranker_port_offset;

        EmbeddingConfig {
            required_dimension: 4096,
            max_batch_chars: self.max_batch_chars,
            max_batch_items: self.max_batch_items,
            providers: vec![
                ProviderConfig {
                    name: "local".to_string(),
                    base_url: format!("http://localhost:{}", self.local_port),
                    model: self.embedder_model.clone(),
                    priority: 1,
                    endpoint: default_embeddings_endpoint(),
                },
                ProviderConfig {
                    name: "dragon".to_string(),
                    base_url: format!("{}:{}", self.dragon_url, self.dragon_port),
                    model: self.embedder_model.clone(),
                    priority: 2,
                    endpoint: default_embeddings_endpoint(),
                },
            ],
            reranker: RerankerConfig {
                base_url: Some(format!("{}:{}", self.dragon_url, reranker_port)),
                model: Some(self.reranker_model.clone()),
                endpoint: default_rerank_endpoint(),
            },
        }
    }

    /// Set batch limits
    pub fn with_batch_limits(mut self, max_chars: usize, max_items: usize) -> Self {
        self.max_batch_chars = max_chars;
        self.max_batch_items = max_items;
        self
    }
}

// =============================================================================
// EMBEDDING CLIENT
// =============================================================================

/// Universal embedding client with provider cascade
pub struct EmbeddingClient {
    client: Client,
    embedder_url: String,
    embedder_model: String,
    reranker_url: Option<String>,
    reranker_model: Option<String>,
    /// Which provider we're connected to
    connected_to: String,
    /// Expected dimension (for validation)
    required_dimension: usize,
    /// Maximum characters per embedding batch
    max_batch_chars: usize,
    /// Maximum items per embedding batch
    max_batch_items: usize,
}

// Type alias for backward compatibility
pub type MLXBridge = EmbeddingClient;

impl EmbeddingClient {
    /// Create client with config-driven provider cascade
    pub async fn new(config: &EmbeddingConfig) -> Result<Self> {
        if config.providers.is_empty() {
            return Err(anyhow!(
                "No embedding providers configured! Add providers to [embeddings.providers]"
            ));
        }

        // Long timeout for large embedding batches (100+ chunks can take minutes)
        let client = Client::builder()
            .timeout(Duration::from_secs(300))
            .connect_timeout(Duration::from_secs(10))
            .build()?;

        // Sort providers by priority
        let mut providers = config.providers.clone();
        providers.sort_by_key(|p| p.priority);

        // Try each provider in order
        let mut tried = Vec::new();
        for provider in &providers {
            let base_url = provider.base_url.trim_end_matches('/');

            match Self::health_check(&client, base_url).await {
                Ok(()) => {
                    tracing::info!("Embedding: Connected to {} ({})", provider.name, base_url);

                    let embedder_url = format!("{}{}", base_url, provider.endpoint);

                    // Build reranker URL if configured
                    let (reranker_url, reranker_model) =
                        if let Some(ref rr_base) = config.reranker.base_url {
                            (
                                Some(format!(
                                    "{}{}",
                                    rr_base.trim_end_matches('/'),
                                    config.reranker.endpoint
                                )),
                                config.reranker.model.clone(),
                            )
                        } else {
                            (None, None)
                        };

                    return Ok(Self {
                        client,
                        embedder_url,
                        embedder_model: provider.model.clone(),
                        reranker_url,
                        reranker_model,
                        connected_to: provider.name.clone(),
                        required_dimension: config.required_dimension,
                        max_batch_chars: config.max_batch_chars,
                        max_batch_items: config.max_batch_items,
                    });
                }
                Err(e) => {
                    tracing::warn!(
                        "Embedding: {} ({}) unavailable: {}",
                        provider.name,
                        base_url,
                        e
                    );
                    tried.push(format!("- {} ({}): {}", provider.name, base_url, e));
                }
            }
        }

        // All providers failed
        Err(anyhow!(
            "All embedding providers unavailable!\nTried:\n{}",
            tried.join("\n")
        ))
    }

    /// Create from legacy MlxConfig (backward compatibility)
    pub async fn from_legacy(config: &MlxConfig) -> Result<Self> {
        if config.disabled {
            return Err(anyhow!(
                "Embedding disabled via config. No fallback available!"
            ));
        }
        tracing::warn!("Using legacy [mlx] config - please migrate to [embeddings.providers]");
        let embedding_config = config.to_embedding_config();
        Self::new(&embedding_config).await
    }

    /// Legacy constructor from env vars only
    pub async fn from_env() -> Result<Self> {
        let config = MlxConfig::from_env();
        Self::from_legacy(&config).await
    }

    async fn health_check(client: &Client, base_url: &str) -> Result<()> {
        // Try /v1/models first (OpenAI-compat)
        let url = format!("{}/v1/models", base_url);
        let response = client
            .get(&url)
            .timeout(Duration::from_secs(5))
            .send()
            .await;

        match response {
            Ok(resp) if resp.status().is_success() => Ok(()),
            Ok(resp) if resp.status().as_u16() == 404 => {
                // Try Ollama-native endpoint
                let ollama_url = format!("{}/api/tags", base_url);
                let ollama_resp = client
                    .get(&ollama_url)
                    .timeout(Duration::from_secs(5))
                    .send()
                    .await?;
                if ollama_resp.status().is_success() {
                    Ok(())
                } else {
                    Err(anyhow!("Neither /v1/models nor /api/tags available"))
                }
            }
            Ok(resp) => Err(anyhow!("Health check failed: {}", resp.status())),
            Err(e) => Err(anyhow!("Connection failed: {}", e)),
        }
    }

    /// Get which provider we're connected to
    pub fn connected_to(&self) -> &str {
        &self.connected_to
    }

    /// Get required dimension
    pub fn required_dimension(&self) -> usize {
        self.required_dimension
    }

    pub async fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        let request = EmbeddingRequest {
            input: vec![text.to_string()],
            model: self.embedder_model.clone(),
        };

        let response = self
            .client
            .post(&self.embedder_url)
            .json(&request)
            .send()
            .await?
            .json::<EmbeddingResponse>()
            .await?;

        let embedding = response
            .data
            .into_iter()
            .next()
            .map(|d| d.embedding)
            .ok_or_else(|| anyhow!("No embedding returned"))?;

        // Validate dimension
        if embedding.len() != self.required_dimension {
            return Err(anyhow!(
                "Dimension mismatch! Expected {}, got {}. This would corrupt the database!",
                self.required_dimension,
                embedding.len()
            ));
        }

        Ok(embedding)
    }

    /// Embed a batch of texts with intelligent batching to avoid OOM.
    ///
    /// Large texts are chunked and only the first chunk is embedded.
    /// Batches are split to stay under max_batch_chars and max_batch_items.
    pub async fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let mut all_embeddings = Vec::with_capacity(texts.len());
        let mut current_batch: Vec<String> = Vec::new();
        let mut current_chars = 0;

        // Max chars per individual text (half of batch limit for safety)
        let max_text_chars = self.max_batch_chars / 2;

        for text in texts {
            // If single text exceeds limit, chunk it and use first chunk
            let text_to_embed = if text.len() > max_text_chars {
                tracing::debug!(
                    "Text too large ({} chars), truncating to {} chars",
                    text.len(),
                    max_text_chars
                );
                truncate_at_boundary(text, max_text_chars)
            } else {
                text.clone()
            };

            let text_len = text_to_embed.len();

            // Check if we need to flush current batch
            if !current_batch.is_empty()
                && (current_chars + text_len > self.max_batch_chars
                    || current_batch.len() >= self.max_batch_items)
            {
                // Flush current batch
                let batch_embeddings = self.embed_batch_internal(&current_batch).await?;
                all_embeddings.extend(batch_embeddings);
                current_batch.clear();
                current_chars = 0;
            }

            current_batch.push(text_to_embed);
            current_chars += text_len;
        }

        // Flush remaining batch
        if !current_batch.is_empty() {
            let batch_embeddings = self.embed_batch_internal(&current_batch).await?;
            all_embeddings.extend(batch_embeddings);
        }

        Ok(all_embeddings)
    }

    /// Internal batch embedding - sends directly to server
    async fn embed_batch_internal(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let request = EmbeddingRequest {
            input: texts.to_vec(),
            model: self.embedder_model.clone(),
        };

        tracing::debug!(
            "Embedding batch: {} texts, {} chars",
            texts.len(),
            texts.iter().map(|t| t.len()).sum::<usize>()
        );

        let response = self
            .client
            .post(&self.embedder_url)
            .json(&request)
            .send()
            .await?
            .json::<EmbeddingResponse>()
            .await?;

        let embeddings: Vec<Vec<f32>> = response.data.into_iter().map(|d| d.embedding).collect();

        // Validate dimensions
        for (i, emb) in embeddings.iter().enumerate() {
            if emb.len() != self.required_dimension {
                return Err(anyhow!(
                    "Dimension mismatch in batch[{}]! Expected {}, got {}",
                    i,
                    self.required_dimension,
                    emb.len()
                ));
            }
        }

        Ok(embeddings)
    }

    pub async fn rerank(&mut self, query: &str, documents: &[String]) -> Result<Vec<(usize, f32)>> {
        let reranker_url = self.reranker_url.as_ref().ok_or_else(|| {
            anyhow!("Reranker not configured. Add [embeddings.reranker] to config.")
        })?;
        let reranker_model = self
            .reranker_model
            .as_ref()
            .ok_or_else(|| anyhow!("Reranker model not configured."))?;

        let request = RerankRequest {
            query: query.to_string(),
            documents: documents.to_vec(),
            model: reranker_model.clone(),
        };

        let response = self
            .client
            .post(reranker_url)
            .json(&request)
            .send()
            .await?
            .json::<RerankResponse>()
            .await?;

        Ok(response
            .results
            .into_iter()
            .map(|r| (r.index, r.score))
            .collect())
    }
}

/// Truncate text at a word/sentence boundary to avoid cutting mid-word (UTF-8 safe)
fn truncate_at_boundary(text: &str, max_chars: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= max_chars {
        return text.to_string();
    }

    // Get byte index of max_chars-th character (UTF-8 safe)
    let byte_idx = text
        .char_indices()
        .nth(max_chars)
        .map(|(idx, _)| idx)
        .unwrap_or(text.len());

    let truncated = &text[..byte_idx];

    // Try to find a sentence boundary first (prefer complete sentences)
    let half_byte_idx = text
        .char_indices()
        .nth(max_chars / 2)
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    if let Some(pos) = truncated.rfind(['.', '!', '?', '\n'])
        && pos > half_byte_idx {
            return text[..=pos].to_string();
        }

    // Fall back to word boundary
    if let Some(pos) = truncated.rfind([' ', '\t', '\n']) {
        return text[..pos].to_string();
    }

    // Last resort: hard truncate
    truncated.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_sorting() {
        let mut providers = [
            ProviderConfig {
                name: "low".into(),
                base_url: "http://a".into(),
                model: "m".into(),
                priority: 10,
                endpoint: "/v1/embeddings".into(),
            },
            ProviderConfig {
                name: "high".into(),
                base_url: "http://b".into(),
                model: "m".into(),
                priority: 1,
                endpoint: "/v1/embeddings".into(),
            },
        ];
        providers.sort_by_key(|p| p.priority);
        assert_eq!(providers[0].name, "high");
        assert_eq!(providers[1].name, "low");
    }

    #[test]
    fn test_legacy_conversion() {
        let legacy = MlxConfig {
            disabled: false,
            local_port: 12345,
            dragon_url: "http://dragon".into(),
            dragon_port: 12345,
            embedder_model: "test-model".into(),
            reranker_model: "rerank-model".into(),
            reranker_port_offset: 1,
            max_batch_chars: 32000,
            max_batch_items: 16,
        };
        let config = legacy.to_embedding_config();
        assert_eq!(config.providers.len(), 2);
        assert_eq!(config.providers[0].base_url, "http://localhost:12345");
        assert!(config.reranker.base_url.is_some());
        assert_eq!(config.max_batch_chars, 32000);
        assert_eq!(config.max_batch_items, 16);
    }

    #[test]
    fn test_default_config() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.required_dimension, 4096);
        assert_eq!(config.max_batch_chars, 32000);
        assert_eq!(config.max_batch_items, 16);
        assert!(!config.providers.is_empty());
    }

    #[test]
    fn test_truncate_at_boundary() {
        // Test sentence boundary
        let text = "Hello world. This is a test.";
        let truncated = truncate_at_boundary(text, 15);
        assert_eq!(truncated, "Hello world.");

        // Test word boundary fallback
        let text = "Hello world this is a test";
        let truncated = truncate_at_boundary(text, 15);
        assert_eq!(truncated, "Hello world");

        // Test no truncation needed
        let text = "Short text";
        let truncated = truncate_at_boundary(text, 100);
        assert_eq!(truncated, "Short text");
    }
}
