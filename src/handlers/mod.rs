use anyhow::Result;
use serde_json::Value;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use crate::{
    ServerConfig,
    mcp_protocol::{McpCore, McpTransport, jsonrpc_success},
    mcp_runtime::build_mcp_core,
};

pub struct MCPServer {
    mcp_core: Arc<McpCore>,
}

impl MCPServer {
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

            if let Some(response) = self
                .mcp_core
                .handle_payload(trimmed, McpTransport::Stdio)
                .await
            {
                write_json_line(&mut stdout, &response).await?;
            }
        }

        Ok(())
    }

    pub async fn run(self) -> Result<()> {
        self.run_stdio().await
    }

    pub async fn dispatch_request(&self, request: Value) -> Value {
        self.mcp_core
            .handle_request(request, McpTransport::Stdio)
            .await
            .unwrap_or_else(|| jsonrpc_success(&Value::Null, Value::Null))
    }
}

pub async fn create_server(config: ServerConfig) -> Result<MCPServer> {
    let mcp_core = build_mcp_core(config).await?;

    Ok(MCPServer { mcp_core })
}

async fn write_json_line(stdout: &mut tokio::io::Stdout, response: &Value) -> anyhow::Result<()> {
    let payload = serde_json::to_string(response)?;
    stdout.write_all(payload.as_bytes()).await?;
    stdout.write_all(b"\n").await?;
    stdout.flush().await?;
    Ok(())
}
