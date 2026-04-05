use anyhow::Result;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::{
    ServerConfig,
    embeddings::EmbeddingClient,
    mcp_protocol::{McpCore, McpTransport},
    search::HybridSearcher,
    security::NamespaceAccessManager,
    storage::StorageManager,
};

/// Build the shared MCP core used by both stdio and HTTP/SSE transports.
pub async fn build_mcp_core(config: ServerConfig) -> Result<Arc<McpCore>> {
    let embedding_client = EmbeddingClient::new(&config.embeddings).await?;
    tracing::info!(
        "Embedding: Connected to {}",
        embedding_client.connected_to()
    );
    let embedding_client = Arc::new(Mutex::new(embedding_client));

    let db_path = shellexpand::tilde(&config.db_path).to_string();
    let storage = Arc::new(StorageManager::new(&db_path).await?);
    let rag =
        Arc::new(crate::rag::RAGPipeline::new(embedding_client.clone(), storage.clone()).await?);

    let hybrid_searcher = if config.hybrid.mode != crate::search::SearchMode::Vector {
        tracing::info!("Hybrid search: mode={:?}", config.hybrid.mode);
        Some(Arc::new(
            HybridSearcher::new(storage, config.hybrid.clone()).await?,
        ))
    } else {
        tracing::info!("Hybrid search: disabled (vector-only mode)");
        None
    };

    let access_manager = NamespaceAccessManager::new(config.security.clone());
    access_manager.init().await?;
    let access_manager = Arc::new(access_manager);

    Ok(Arc::new(McpCore::new(
        rag,
        hybrid_searcher,
        embedding_client,
        config.max_request_bytes,
        config.allowed_paths,
        access_manager,
    )))
}

/// Dispatch a parsed JSON-RPC request through the shared MCP core.
pub async fn dispatch_mcp_request(
    mcp_core: &McpCore,
    request: Value,
    transport: McpTransport,
) -> Option<Value> {
    mcp_core.handle_request(request, transport).await
}

/// Dispatch a raw JSON-RPC payload through the shared MCP core.
pub async fn dispatch_mcp_payload(
    mcp_core: &McpCore,
    payload: &str,
    transport: McpTransport,
) -> Option<Value> {
    mcp_core.handle_payload(payload, transport).await
}
