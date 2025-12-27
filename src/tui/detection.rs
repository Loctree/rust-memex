//! Auto-detection module for embedding providers.
//!
//! Detects Ollama, MLX server, and other embedding providers automatically.

use anyhow::Result;
use reqwest::Client;
use serde::Deserialize;
use std::time::Duration;

/// Detected embedding provider information.
#[derive(Debug, Clone)]
pub struct DetectedProvider {
    /// Provider type
    pub kind: ProviderKind,
    /// Base URL where provider is running
    pub base_url: String,
    /// Port number
    pub port: u16,
    /// List of available models (if detected)
    pub models: Vec<String>,
    /// Suggested model to use
    pub suggested_model: Option<String>,
    /// Connection status
    pub status: ProviderStatus,
}

impl DetectedProvider {
    /// Get a human-readable description
    pub fn description(&self) -> String {
        match &self.status {
            ProviderStatus::Online(model) => {
                format!(
                    "{} at {} - {}",
                    self.kind.display_name(),
                    self.base_url,
                    model
                )
            }
            ProviderStatus::OnlineNoModel => {
                format!(
                    "{} at {} (no embedding model)",
                    self.kind.display_name(),
                    self.base_url
                )
            }
            ProviderStatus::Offline => {
                format!(
                    "{} at {} (offline)",
                    self.kind.display_name(),
                    self.base_url
                )
            }
        }
    }

    /// Get display name for UI
    pub fn display_name(&self) -> String {
        self.description()
    }

    /// Check if provider is usable
    pub fn is_usable(&self) -> bool {
        matches!(self.status, ProviderStatus::Online(_))
    }

    /// Get the model to use
    pub fn model(&self) -> Option<&str> {
        if let ProviderStatus::Online(ref model) = self.status {
            Some(model.as_str())
        } else {
            self.suggested_model.as_deref()
        }
    }

    /// Get embedding dimension for the detected model
    pub fn dimension(&self) -> usize {
        match self.model() {
            Some(m) if m.contains("qwen3-embedding") => 4096,
            Some(m) if m.contains("Qwen3-Embedding") => 4096,
            Some(m) if m.contains("bge-m3") => 1024,
            Some(m) if m.contains("nomic-embed") => 768,
            Some(m) if m.contains("mxbai-embed") => 1024,
            Some(m) if m.contains("all-minilm") => 384,
            _ => 4096, // Default to Qwen3 dimension
        }
    }
}

/// Type of embedding provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    /// Ollama with embedding models
    Ollama,
    /// MLX embedding server
    Mlx,
    /// Generic OpenAI-compatible endpoint
    OpenAICompat,
    /// Manual configuration
    Manual,
}

impl ProviderKind {
    pub fn display_name(&self) -> &'static str {
        match self {
            ProviderKind::Ollama => "Ollama",
            ProviderKind::Mlx => "MLX Server",
            ProviderKind::OpenAICompat => "OpenAI-Compatible",
            ProviderKind::Manual => "Manual",
        }
    }
}

/// Provider connection status.
#[derive(Debug, Clone)]
pub enum ProviderStatus {
    /// Provider is online with a usable model
    Online(String),
    /// Provider is online but no embedding model found
    OnlineNoModel,
    /// Provider is offline
    Offline,
}

/// Response from Ollama /api/tags endpoint.
#[derive(Debug, Deserialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaModel>,
}

#[derive(Debug, Deserialize)]
struct OllamaModel {
    name: String,
}

/// Response from OpenAI-compatible /v1/models endpoint.
#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<ModelInfo>,
}

#[derive(Debug, Deserialize)]
struct ModelInfo {
    id: String,
}

/// Detect embedding providers on standard ports.
pub async fn detect_providers() -> Vec<DetectedProvider> {
    let client = Client::builder()
        .timeout(Duration::from_secs(3))
        .connect_timeout(Duration::from_secs(2))
        .build()
        .unwrap_or_default();

    let mut providers = Vec::new();

    // Check Ollama on localhost:11434
    if let Some(provider) = detect_ollama(&client, "http://localhost", 11434).await {
        providers.push(provider);
    }

    // Check MLX on localhost:12345
    if let Some(provider) = detect_mlx(&client, "http://localhost", 12345).await {
        providers.push(provider);
    }

    // Check dragon:12345 (common remote MLX server)
    if let Some(provider) = detect_mlx(&client, "http://dragon", 12345).await {
        providers.push(provider);
    }

    providers
}

/// Check if a URL is reachable (simple health check).
async fn check_health(client: &Client, url: &str) -> bool {
    client.get(url).send().await.is_ok()
}

/// Detect Ollama on a given host/port.
async fn detect_ollama(client: &Client, host: &str, port: u16) -> Option<DetectedProvider> {
    let base_url = format!("{}:{}", host, port);
    let tags_url = format!("{}/api/tags", base_url);

    // Try to get list of models
    let response = match client.get(&tags_url).send().await {
        Ok(r) if r.status().is_success() => r,
        _ => {
            return Some(DetectedProvider {
                kind: ProviderKind::Ollama,
                base_url: base_url.clone(),
                port,
                models: vec![],
                suggested_model: Some("qwen3-embedding:8b".to_string()),
                status: ProviderStatus::Offline,
            });
        }
    };

    let tags: OllamaTagsResponse = match response.json().await {
        Ok(t) => t,
        Err(_) => {
            return Some(DetectedProvider {
                kind: ProviderKind::Ollama,
                base_url,
                port,
                models: vec![],
                suggested_model: Some("qwen3-embedding:8b".to_string()),
                status: ProviderStatus::OnlineNoModel,
            });
        }
    };

    let models: Vec<String> = tags.models.iter().map(|m| m.name.clone()).collect();

    // Look for embedding models (prefer qwen3-embedding)
    let embedding_model = models
        .iter()
        .find(|m| m.contains("qwen3-embedding"))
        .or_else(|| models.iter().find(|m| m.contains("embedding")))
        .or_else(|| models.iter().find(|m| m.contains("embed")))
        .or_else(|| models.iter().find(|m| m.contains("bge")))
        .or_else(|| models.iter().find(|m| m.contains("nomic")));

    let status = if let Some(model) = embedding_model {
        ProviderStatus::Online(model.clone())
    } else {
        ProviderStatus::OnlineNoModel
    };

    Some(DetectedProvider {
        kind: ProviderKind::Ollama,
        base_url,
        port,
        models,
        suggested_model: Some("qwen3-embedding:8b".to_string()),
        status,
    })
}

/// Detect MLX embedding server on a given host/port.
async fn detect_mlx(client: &Client, host: &str, port: u16) -> Option<DetectedProvider> {
    let base_url = format!("{}:{}", host, port);
    let models_url = format!("{}/v1/models", base_url);

    // Try OpenAI-compatible endpoint
    let response = match client.get(&models_url).send().await {
        Ok(r) if r.status().is_success() => r,
        _ => {
            // Don't report offline MLX servers unless we were explicitly looking for them
            return None;
        }
    };

    let models_resp: ModelsResponse = match response.json().await {
        Ok(m) => m,
        Err(_) => {
            return Some(DetectedProvider {
                kind: ProviderKind::Mlx,
                base_url,
                port,
                models: vec![],
                suggested_model: Some("Qwen/Qwen3-Embedding-4B".to_string()),
                status: ProviderStatus::OnlineNoModel,
            });
        }
    };

    let models: Vec<String> = models_resp.data.iter().map(|m| m.id.clone()).collect();

    // Look for embedding models
    let embedding_model = models
        .iter()
        .find(|m| m.contains("Embedding") || m.contains("embedding"))
        .or_else(|| models.iter().find(|m| m.contains("embed")));

    let status = if let Some(model) = embedding_model {
        ProviderStatus::Online(model.clone())
    } else {
        ProviderStatus::OnlineNoModel
    };

    Some(DetectedProvider {
        kind: ProviderKind::Mlx,
        base_url,
        port,
        models,
        suggested_model: Some("Qwen/Qwen3-Embedding-4B".to_string()),
        status,
    })
}

/// Check a custom endpoint for OpenAI-compatibility.
pub async fn check_custom_endpoint(url: &str) -> Result<DetectedProvider> {
    let client = Client::builder()
        .timeout(Duration::from_secs(5))
        .connect_timeout(Duration::from_secs(3))
        .build()?;

    let base_url = url.trim_end_matches('/');

    // Extract port from URL
    let port = reqwest::Url::parse(base_url)
        .ok()
        .and_then(|u| u.port())
        .unwrap_or(80);

    // Try /v1/models first
    let models_url = format!("{}/v1/models", base_url);
    if let Ok(resp) = client.get(&models_url).send().await
        && resp.status().is_success()
        && let Ok(models_resp) = resp.json::<ModelsResponse>().await
    {
        let models: Vec<String> = models_resp.data.iter().map(|m| m.id.clone()).collect();

        let embedding_model = models
            .iter()
            .find(|m| m.contains("embedding") || m.contains("Embedding"))
            .cloned();

        let status = if let Some(ref model) = embedding_model {
            ProviderStatus::Online(model.clone())
        } else {
            ProviderStatus::OnlineNoModel
        };

        return Ok(DetectedProvider {
            kind: ProviderKind::OpenAICompat,
            base_url: base_url.to_string(),
            port,
            models,
            suggested_model: embedding_model,
            status,
        });
    }

    // Try Ollama endpoint
    let tags_url = format!("{}/api/tags", base_url);
    if let Ok(resp) = client.get(&tags_url).send().await
        && resp.status().is_success()
        && let Ok(tags) = resp.json::<OllamaTagsResponse>().await
    {
        let models: Vec<String> = tags.models.iter().map(|m| m.name.clone()).collect();

        let embedding_model = models
            .iter()
            .find(|m| m.contains("embedding"))
            .or_else(|| models.iter().find(|m| m.contains("embed")))
            .cloned();

        let status = if let Some(ref model) = embedding_model {
            ProviderStatus::Online(model.clone())
        } else if !models.is_empty() {
            ProviderStatus::OnlineNoModel
        } else {
            ProviderStatus::Offline
        };

        return Ok(DetectedProvider {
            kind: ProviderKind::Ollama,
            base_url: base_url.to_string(),
            port,
            models,
            suggested_model: embedding_model.or(Some("qwen3-embedding:8b".to_string())),
            status,
        });
    }

    Ok(DetectedProvider {
        kind: ProviderKind::OpenAICompat,
        base_url: base_url.to_string(),
        port,
        models: vec![],
        suggested_model: None,
        status: ProviderStatus::Offline,
    })
}

/// Get dimension explanation for UI
pub fn dimension_explanation(dim: usize) -> &'static str {
    match dim {
        4096 => "Qwen3 models (4096 dims) - highest quality, recommended",
        1024 => "BGE-M3/mxbai-embed (1024 dims) - good balance",
        768 => "nomic-embed (768 dims) - fast and lightweight",
        384 => "all-minilm (384 dims) - fastest, lowest quality",
        _ => "Custom dimension - ensure all providers match",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_kind_display() {
        assert_eq!(ProviderKind::Ollama.display_name(), "Ollama");
        assert_eq!(ProviderKind::Mlx.display_name(), "MLX Server");
    }

    #[test]
    fn test_dimension_for_models() {
        let provider = DetectedProvider {
            kind: ProviderKind::Ollama,
            base_url: "http://localhost:11434".to_string(),
            port: 11434,
            models: vec![],
            suggested_model: Some("qwen3-embedding:8b".to_string()),
            status: ProviderStatus::Online("qwen3-embedding:8b".to_string()),
        };
        assert_eq!(provider.dimension(), 4096);
    }
}
