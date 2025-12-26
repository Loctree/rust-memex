use anyhow::{Result, anyhow};
use serde_json::json;
use std::path::Path;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::{
    ServerConfig, embeddings::EmbeddingClient, rag::RAGPipeline, security::NamespaceAccessManager,
    storage::StorageManager,
};

/// Validates a file path to prevent path traversal attacks.
/// Returns the canonicalized path if valid, or an error if the path is unsafe.
///
/// # Arguments
/// * `path_str` - The path string to validate
/// * `allowed_paths` - Whitelist of allowed paths. If empty, defaults to $HOME and cwd.
///   Supports ~ expansion and absolute paths.
fn validate_path(path_str: &str, allowed_paths: &[String]) -> Result<std::path::PathBuf> {
    if path_str.is_empty() {
        return Err(anyhow!("Path cannot be empty"));
    }

    // Expand ~ to home directory
    let expanded = shellexpand::tilde(path_str).to_string();
    // This IS the path validation/sanitization function - not a vulnerability
    // nosemgrep: rust.actix.path-traversal.tainted-path.tainted-path
    let path = Path::new(&expanded);

    // Check for obvious path traversal patterns before canonicalization
    let path_string = path_str.to_string();
    if path_string.contains("..") {
        return Err(anyhow!("Path traversal detected: '..' not allowed"));
    }

    // Canonicalize to resolve symlinks and get absolute path
    let canonical = path
        .canonicalize()
        .map_err(|e| anyhow!("Cannot resolve path '{}': {}", path_str, e))?;

    // Determine allowed base paths
    let is_safe = if allowed_paths.is_empty() {
        // Default behavior: allow paths under $HOME or current working directory
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .map(std::path::PathBuf::from)
            .ok();
        let cwd = std::env::current_dir().ok();

        home.as_ref()
            .map(|h| canonical.starts_with(h))
            .unwrap_or(false)
            || cwd
                .as_ref()
                .map(|c| canonical.starts_with(c))
                .unwrap_or(false)
    } else {
        // Use configured whitelist
        allowed_paths.iter().any(|allowed| {
            // Expand ~ in allowed path
            let expanded_allowed = shellexpand::tilde(allowed).to_string();
            // nosemgrep: rust.actix.path-traversal.tainted-path.tainted-path
            let allowed_path = Path::new(&expanded_allowed);

            // Try to canonicalize the allowed path, fall back to expanded path if it doesn't exist
            let canonical_allowed = allowed_path
                .canonicalize()
                .unwrap_or_else(|_| std::path::PathBuf::from(&expanded_allowed));

            canonical.starts_with(&canonical_allowed)
        })
    };

    if !is_safe {
        let allowed_info = if allowed_paths.is_empty() {
            "$HOME and current working directory".to_string()
        } else {
            format!("configured paths: {:?}", allowed_paths)
        };
        return Err(anyhow!(
            "Access denied: path '{}' is outside allowed directories ({})",
            path_str,
            allowed_info
        ));
    }

    Ok(canonical)
}

pub struct MCPServer {
    rag: Arc<RAGPipeline>,
    max_request_bytes: usize,
    allowed_paths: Vec<String>,
    access_manager: Arc<NamespaceAccessManager>,
}

impl MCPServer {
    pub async fn run_stdio(self) -> Result<()> {
        let stdin = tokio::io::stdin();
        let mut stdout = tokio::io::stdout();
        let mut reader = BufReader::new(stdin);
        let mut line = String::new();

        // Read newline-delimited JSON-RPC (standard MCP transport)
        loop {
            line.clear();
            let n = reader.read_line(&mut line).await?;
            if n == 0 {
                break; // EOF
            }

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue; // Skip empty lines
            }

            // Check size limit
            if trimmed.len() > self.max_request_bytes {
                let err = json!({
                    "jsonrpc": "2.0",
                    "error": {"code": -32600, "message": format!("Request too large: {} bytes (max {})", trimmed.len(), self.max_request_bytes)},
                    "id": serde_json::Value::Null
                });
                let payload = serde_json::to_string(&err)?;
                stdout.write_all(payload.as_bytes()).await?;
                stdout.write_all(b"\n").await?;
                stdout.flush().await?;
                continue;
            }

            let request: serde_json::Value = match serde_json::from_str(trimmed) {
                Ok(req) => req,
                Err(e) => {
                    let err = json!({
                        "jsonrpc": "2.0",
                        "error": {"code": -32700, "message": format!("Parse error: {}", e)},
                        "id": serde_json::Value::Null
                    });
                    let payload = serde_json::to_string(&err)?;
                    stdout.write_all(payload.as_bytes()).await?;
                    stdout.write_all(b"\n").await?;
                    stdout.flush().await?;
                    continue;
                }
            };

            let response = self.handle_request(request).await;
            let payload = serde_json::to_string(&response)?;
            stdout.write_all(payload.as_bytes()).await?;
            stdout.write_all(b"\n").await?;
            stdout.flush().await?;
        }

        Ok(())
    }

    pub async fn run(self) -> Result<()> {
        self.run_stdio().await
    }

    pub async fn handle_request(&self, request: serde_json::Value) -> serde_json::Value {
        let method = request["method"].as_str().unwrap_or("");
        let id = request["id"].clone();

        let result = match method {
            "initialize" => json!({
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": "rmcp-memex",
                    "version": env!("CARGO_PKG_VERSION")
                },
                "capabilities": {
                    "tools": {},
                    "resources": {}
                }
            }),

            "tools/list" => json!({
                "tools": [
                    {
                        "name": "health",
                        "description": "Health/status of rmcp-memex server",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    },
                    {
                        "name": "rag_index",
                        "description": "Index a document for RAG",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "namespace": {"type": "string"}
                            },
                            "required": ["path"]
                        }
                    },
                    {
                        "name": "rag_index_text",
                        "description": "Index raw text for RAG/memory",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "id": {"type": "string"},
                                "namespace": {"type": "string"},
                                "metadata": {"type": "object"}
                            },
                            "required": ["text"]
                        }
                    },
                    {
                        "name": "rag_search",
                        "description": "Search documents using RAG",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "k": {"type": "integer", "default": 10},
                                "namespace": {"type": "string"}
                            },
                            "required": ["query"]
                        }
                    },
                    {
                        "name": "memory_upsert",
                        "description": "Upsert a text chunk into vector memory. If the namespace is protected, provide the access token.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "namespace": {"type": "string"},
                                "id": {"type": "string"},
                                "text": {"type": "string"},
                                "metadata": {"type": "object"},
                                "token": {"type": "string", "description": "Access token for protected namespaces"}
                            },
                            "required": ["namespace", "id", "text"]
                        }
                    },
                    {
                        "name": "memory_get",
                        "description": "Get a stored chunk by namespace + id. If the namespace is protected, provide the access token.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "namespace": {"type": "string"},
                                "id": {"type": "string"},
                                "token": {"type": "string", "description": "Access token for protected namespaces"}
                            },
                            "required": ["namespace", "id"]
                        }
                    },
                    {
                        "name": "memory_search",
                        "description": "Semantic search within a namespace. If the namespace is protected, provide the access token.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "namespace": {"type": "string"},
                                "query": {"type": "string"},
                                "k": {"type": "integer", "default": 5},
                                "token": {"type": "string", "description": "Access token for protected namespaces"}
                            },
                            "required": ["namespace", "query"]
                        }
                    },
                    {
                        "name": "memory_delete",
                        "description": "Delete a chunk by namespace + id. If the namespace is protected, provide the access token.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "namespace": {"type": "string"},
                                "id": {"type": "string"},
                                "token": {"type": "string", "description": "Access token for protected namespaces"}
                            },
                            "required": ["namespace", "id"]
                        }
                    },
                    {
                        "name": "memory_purge_namespace",
                        "description": "Delete all chunks in a namespace. If the namespace is protected, provide the access token.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "namespace": {"type": "string"},
                                "token": {"type": "string", "description": "Access token for protected namespaces"}
                            },
                            "required": ["namespace"]
                        }
                    },
                    {
                        "name": "namespace_create_token",
                        "description": "Create an access token for a namespace. Once created, the namespace will require this token for access.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "namespace": {"type": "string", "description": "The namespace to protect with a token"},
                                "description": {"type": "string", "description": "Optional description for the token"}
                            },
                            "required": ["namespace"]
                        }
                    },
                    {
                        "name": "namespace_revoke_token",
                        "description": "Revoke the access token for a namespace, making it publicly accessible again.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "namespace": {"type": "string", "description": "The namespace to remove token protection from"}
                            },
                            "required": ["namespace"]
                        }
                    },
                    {
                        "name": "namespace_list_protected",
                        "description": "List all namespaces that have token protection enabled.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    },
                    {
                        "name": "namespace_security_status",
                        "description": "Check if namespace security (token-based access control) is enabled.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                ]
            }),

            "tools/call" => {
                let tool_name = request["params"]["name"].as_str().unwrap_or("");
                let args = &request["params"]["arguments"];

                match tool_name {
                    "health" => {
                        let status = json!({
                            "version": env!("CARGO_PKG_VERSION"),
                            "db_path": self.rag.storage().lance_path(),
                            "backend": "mlx",
                            "mlx_server": self.rag.mlx_connected_to(),
                        });
                        json!({
                            "content": [{"type": "text", "text": serde_json::to_string(&status).unwrap_or_default()}]
                        })
                    }
                    "rag_index" => {
                        let path_str = args["path"].as_str().unwrap_or("");
                        let namespace = args["namespace"].as_str();

                        // Validate path to prevent path traversal attacks
                        let validated_path = match validate_path(path_str, &self.allowed_paths) {
                            Ok(p) => p,
                            Err(e) => {
                                return json!({
                                    "jsonrpc": "2.0",
                                    "error": {"code": -32602, "message": e.to_string()},
                                    "id": id
                                });
                            }
                        };

                        match self.rag.index_document(&validated_path, namespace).await {
                            Ok(_) => json!({
                                "content": [{"type": "text", "text": format!("Indexed: {}", path_str)}]
                            }),
                            Err(e) => json!({
                                "error": {"message": e.to_string()}
                            }),
                        }
                    }
                    "rag_index_text" => {
                        let text = args["text"].as_str().unwrap_or("").to_string();
                        let namespace = args["namespace"].as_str();
                        let metadata = args.get("metadata").cloned().unwrap_or_else(|| json!({}));
                        let id = args
                            .get("id")
                            .and_then(|v| v.as_str().map(|s| s.to_string()))
                            .unwrap_or_else(|| Uuid::new_v4().to_string());

                        match self
                            .rag
                            .index_text(namespace, id.clone(), text, metadata)
                            .await
                        {
                            Ok(returned_id) => json!({
                                "content": [{"type": "text", "text": format!("Indexed text with id {}", returned_id)}]
                            }),
                            Err(e) => json!({
                                "error": {"message": e.to_string()}
                            }),
                        }
                    }
                    "rag_search" => {
                        let query = args["query"].as_str().unwrap_or("");
                        let k = args["k"].as_u64().unwrap_or(10) as usize;
                        let namespace = args["namespace"].as_str();

                        match self.rag.search_inner(namespace, query, k).await {
                            Ok(results) => json!({
                                "content": [{
                                    "type": "text",
                                    "text": serde_json::to_string(&results).unwrap_or_default()
                                }]
                            }),
                            Err(e) => json!({
                                "error": {"message": e.to_string()}
                            }),
                        }
                    }
                    "memory_upsert" => {
                        let namespace = args["namespace"].as_str().unwrap_or("default");
                        let token = args["token"].as_str();

                        // Verify namespace access
                        if let Err(e) = self.access_manager.verify_access(namespace, token).await {
                            return json!({
                                "jsonrpc": "2.0",
                                "error": {"code": -32603, "message": e.to_string()},
                                "id": id
                            });
                        }

                        let id_str = args["id"].as_str().unwrap_or("").to_string();
                        let text = args["text"].as_str().unwrap_or("").to_string();
                        let metadata = args.get("metadata").cloned().unwrap_or_else(|| json!({}));

                        match self
                            .rag
                            .memory_upsert(namespace, id_str.clone(), text, metadata)
                            .await
                        {
                            Ok(_) => json!({
                                "content": [{"type": "text", "text": format!("Upserted {}", id_str)}]
                            }),
                            Err(e) => json!({
                                "error": {"message": e.to_string()}
                            }),
                        }
                    }
                    "memory_get" => {
                        let namespace = args["namespace"].as_str().unwrap_or("default");
                        let token = args["token"].as_str();

                        // Verify namespace access
                        if let Err(e) = self.access_manager.verify_access(namespace, token).await {
                            return json!({
                                "jsonrpc": "2.0",
                                "error": {"code": -32603, "message": e.to_string()},
                                "id": id
                            });
                        }

                        let id_str = args["id"].as_str().unwrap_or("");
                        match self.rag.memory_get(namespace, id_str).await {
                            Ok(Some(doc)) => json!({
                                "content": [{"type": "text", "text": serde_json::to_string(&doc).unwrap_or_default()}]
                            }),
                            Ok(None) => json!({
                                "content": [{"type": "text", "text": "Not found"}]
                            }),
                            Err(e) => json!({
                                "error": {"message": e.to_string()}
                            }),
                        }
                    }
                    "memory_search" => {
                        let namespace = args["namespace"].as_str().unwrap_or("default");
                        let token = args["token"].as_str();

                        // Verify namespace access
                        if let Err(e) = self.access_manager.verify_access(namespace, token).await {
                            return json!({
                                "jsonrpc": "2.0",
                                "error": {"code": -32603, "message": e.to_string()},
                                "id": id
                            });
                        }

                        let query = args["query"].as_str().unwrap_or("");
                        let k = args["k"].as_u64().unwrap_or(5) as usize;
                        match self.rag.memory_search(namespace, query, k).await {
                            Ok(results) => json!({
                                "content": [{
                                    "type": "text",
                                    "text": serde_json::to_string(&results).unwrap_or_default()
                                }]
                            }),
                            Err(e) => json!({
                                "error": {"message": e.to_string()}
                            }),
                        }
                    }
                    "memory_delete" => {
                        let namespace = args["namespace"].as_str().unwrap_or("default");
                        let token = args["token"].as_str();

                        // Verify namespace access
                        if let Err(e) = self.access_manager.verify_access(namespace, token).await {
                            return json!({
                                "jsonrpc": "2.0",
                                "error": {"code": -32603, "message": e.to_string()},
                                "id": id
                            });
                        }

                        let id_str = args["id"].as_str().unwrap_or("");
                        match self.rag.memory_delete(namespace, id_str).await {
                            Ok(deleted) => json!({
                                "content": [{"type": "text", "text": format!("Deleted {} rows", deleted)}]
                            }),
                            Err(e) => json!({
                                "error": {"message": e.to_string()}
                            }),
                        }
                    }
                    "memory_purge_namespace" => {
                        let namespace = args["namespace"].as_str().unwrap_or("default");
                        let token = args["token"].as_str();

                        // Verify namespace access
                        if let Err(e) = self.access_manager.verify_access(namespace, token).await {
                            return json!({
                                "jsonrpc": "2.0",
                                "error": {"code": -32603, "message": e.to_string()},
                                "id": id
                            });
                        }

                        match self.rag.purge_namespace(namespace).await {
                            Ok(deleted) => json!({
                                "content": [{"type": "text", "text": format!("Purged namespace '{}', removed {} rows", namespace, deleted)}]
                            }),
                            Err(e) => json!({
                                "error": {"message": e.to_string()}
                            }),
                        }
                    }
                    "namespace_create_token" => {
                        let namespace = args["namespace"].as_str().unwrap_or("");
                        let description = args["description"].as_str().map(|s| s.to_string());

                        if namespace.is_empty() {
                            json!({
                                "error": {"message": "Namespace is required"}
                            })
                        } else {
                            match self
                                .access_manager
                                .create_token(namespace, description)
                                .await
                            {
                                Ok(token) => json!({
                                    "content": [{
                                        "type": "text",
                                        "text": format!(
                                            "Token created for namespace '{}'. Store this token securely - it won't be shown again!\n\nToken: {}",
                                            namespace, token
                                        )
                                    }]
                                }),
                                Err(e) => json!({
                                    "error": {"message": e.to_string()}
                                }),
                            }
                        }
                    }
                    "namespace_revoke_token" => {
                        let namespace = args["namespace"].as_str().unwrap_or("");

                        if namespace.is_empty() {
                            json!({
                                "error": {"message": "Namespace is required"}
                            })
                        } else {
                            match self.access_manager.revoke_token(namespace).await {
                                Ok(true) => json!({
                                    "content": [{"type": "text", "text": format!("Token revoked for namespace '{}'. The namespace is now publicly accessible.", namespace)}]
                                }),
                                Ok(false) => json!({
                                    "content": [{"type": "text", "text": format!("No token found for namespace '{}'.", namespace)}]
                                }),
                                Err(e) => json!({
                                    "error": {"message": e.to_string()}
                                }),
                            }
                        }
                    }
                    "namespace_list_protected" => {
                        let protected = self.access_manager.list_protected_namespaces().await;
                        if protected.is_empty() {
                            json!({
                                "content": [{"type": "text", "text": "No namespaces are currently protected with tokens."}]
                            })
                        } else {
                            let list: Vec<serde_json::Value> = protected
                                .iter()
                                .map(|(ns, created_at, desc)| {
                                    json!({
                                        "namespace": ns,
                                        "created_at": created_at,
                                        "description": desc
                                    })
                                })
                                .collect();
                            json!({
                                "content": [{"type": "text", "text": serde_json::to_string_pretty(&list).unwrap_or_default()}]
                            })
                        }
                    }
                    "namespace_security_status" => {
                        let enabled = self.access_manager.is_enabled();
                        let protected_count =
                            self.access_manager.list_protected_namespaces().await.len();
                        json!({
                            "content": [{
                                "type": "text",
                                "text": format!(
                                    "Namespace security: {}\nProtected namespaces: {}\n\nNote: When security is disabled, all namespaces are accessible without tokens.",
                                    if enabled { "ENABLED" } else { "DISABLED" },
                                    protected_count
                                )
                            }]
                        })
                    }
                    _ => {
                        return json!({
                            "jsonrpc": "2.0",
                            "error": {"code": -32601, "message": "Unknown tool"},
                            "id": id
                        });
                    }
                }
            }

            _ => {
                return json!({
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": "Unknown method"},
                    "id": id
                });
            }
        };

        json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": result
        })
    }
}

pub async fn create_server(config: ServerConfig) -> Result<MCPServer> {
    // Initialize embedding client with config-driven provider cascade
    let embedding_client = EmbeddingClient::new(&config.embeddings).await?;
    tracing::info!(
        "Embedding: Connected to {}",
        embedding_client.connected_to()
    );
    let embedding_client = Arc::new(Mutex::new(embedding_client));

    let db_path = shellexpand::tilde(&config.db_path).to_string();
    let storage = Arc::new(StorageManager::new(config.cache_mb, &db_path).await?);
    // NOTE: Removed ensure_collection() - table opens lazily on first use
    // This speeds up MCP server startup significantly
    let rag = Arc::new(RAGPipeline::new(embedding_client, storage).await?);

    // Initialize namespace access manager
    let access_manager = NamespaceAccessManager::new(config.security.clone());
    access_manager.init().await?;
    let access_manager = Arc::new(access_manager);

    Ok(MCPServer {
        rag,
        max_request_bytes: config.max_request_bytes,
        allowed_paths: config.allowed_paths,
        access_manager,
    })
}
