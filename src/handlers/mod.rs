use anyhow::Result;
use serde_json::{Value, json};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Mutex;

use crate::{
    ServerConfig,
    embeddings::EmbeddingClient,
    mcp_protocol::{McpCore, McpTransport, jsonrpc_success},
    rag::RAGPipeline,
    search::HybridSearcher,
    security::NamespaceAccessManager,
    storage::StorageManager,
};

pub struct MCPServer {
    mcp_core: Arc<McpCore>,
    max_request_bytes: usize,
}

impl MCPServer {
    /// Get the RAGPipeline for sharing with the HTTP server.
    pub fn rag(&self) -> Arc<RAGPipeline> {
        self.mcp_core.rag()
    }

    /// Get the shared MCP core for reuse across transports.
    pub fn mcp_core(&self) -> Arc<McpCore> {
        self.mcp_core.clone()
    }

    pub async fn run_stdio(self) -> Result<()> {
        let stdin = tokio::io::stdin();
        let mut stdout = tokio::io::stdout();
        let mut reader = BufReader::new(stdin);
        let mut line = String::new();

        loop {
            line.clear();
            let read = reader.read_line(&mut line).await?;
            if read == 0 {
                break;
            }

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if trimmed.len() > self.max_request_bytes {
                let response = json!({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32600,
                        "message": format!(
                            "Request too large: {} bytes (max {})",
                            trimmed.len(),
                            self.max_request_bytes
                        )
                    }
                });
                write_json_line(&mut stdout, &response).await?;
                continue;
            }

            let request: Value = match serde_json::from_str(trimmed) {
                Ok(request) => request,
                Err(error) => {
                    let response = json!({
                        "jsonrpc": "2.0",
                        "error": {"code": -32700, "message": format!("Parse error: {}", error)}
                    });
                    write_json_line(&mut stdout, &response).await?;
                    continue;
                }
            };

            if let Some(response) = self
                .mcp_core
                .handle_jsonrpc_request(request, McpTransport::Stdio)
                .await
                .into_option()
            {
                write_json_line(&mut stdout, &response).await?;
            }
        }

        Ok(())
    }

    pub async fn run(self) -> Result<()> {
        self.run_stdio().await
    }

    pub async fn handle_request(&self, request: Value) -> Value {
        self.mcp_core
            .handle_jsonrpc_request(request, McpTransport::Stdio)
            .await
            .into_option()
            .unwrap_or_else(|| jsonrpc_success(&Value::Null, Value::Null))
    }
}

pub async fn create_server(config: ServerConfig) -> Result<MCPServer> {
    let embedding_client = EmbeddingClient::new(&config.embeddings).await?;
    tracing::info!(
        "Embedding: Connected to {}",
        embedding_client.connected_to()
    );
    let embedding_client = Arc::new(Mutex::new(embedding_client));

    let db_path = shellexpand::tilde(&config.db_path).to_string();
    let storage = Arc::new(StorageManager::new(&db_path).await?);
    let rag = Arc::new(RAGPipeline::new(embedding_client.clone(), storage.clone()).await?);

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

    let mcp_core = Arc::new(McpCore::new(
        rag,
        hybrid_searcher,
        embedding_client,
        config.allowed_paths,
        access_manager,
    ));

    Ok(MCPServer {
        mcp_core,
        max_request_bytes: config.max_request_bytes,
    })
}

async fn write_json_line(stdout: &mut tokio::io::Stdout, response: &Value) -> anyhow::Result<()> {
    let payload = serde_json::to_string(response)?;
    stdout.write_all(payload.as_bytes()).await?;
    stdout.write_all(b"\n").await?;
    stdout.flush().await?;
    Ok(())
}
