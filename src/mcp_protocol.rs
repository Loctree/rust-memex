use anyhow::{Result, anyhow};
use serde_json::{Value, json};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::{
    embeddings::EmbeddingClient,
    query::{QueryRouter, SearchModeRecommendation},
    rag::{RAGPipeline, SliceLayer},
    search::{HybridSearcher, SearchMode},
    security::NamespaceAccessManager,
};

pub const PROTOCOL_VERSION: &str = "2024-11-05";
pub const SERVER_NAME: &str = "rmcp-memex";

/// Build a JSON-RPC 2.0 error response.
/// Per JSON-RPC 2.0 spec, omits `id` field when it's null or absent.
pub fn jsonrpc_error(id: Option<&Value>, code: i32, message: impl Into<String>) -> Value {
    let message = message.into();

    match id {
        Some(id) if !id.is_null() => json!({
            "jsonrpc": "2.0",
            "error": {"code": code, "message": message},
            "id": id
        }),
        _ => json!({
            "jsonrpc": "2.0",
            "error": {"code": code, "message": message}
        }),
    }
}

/// Build a JSON-RPC 2.0 success response.
/// Per JSON-RPC 2.0 spec, omits `id` field when it's null.
pub fn jsonrpc_success(id: &Value, result: Value) -> Value {
    if id.is_null() {
        json!({
            "jsonrpc": "2.0",
            "result": result
        })
    } else {
        json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": result
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McpTransport {
    Stdio,
    HttpSse,
}

impl McpTransport {
    fn health_transport(self) -> Option<&'static str> {
        match self {
            Self::Stdio => None,
            Self::HttpSse => Some("mcp-over-sse"),
        }
    }
}

pub enum McpDispatch {
    Notification,
    Response(Value),
}

impl McpDispatch {
    pub fn into_option(self) -> Option<Value> {
        match self {
            Self::Notification => None,
            Self::Response(response) => Some(response),
        }
    }
}

#[derive(Clone)]
pub struct McpCore {
    rag: Arc<RAGPipeline>,
    hybrid_searcher: Option<Arc<HybridSearcher>>,
    embedding_client: Arc<Mutex<EmbeddingClient>>,
    allowed_paths: Vec<String>,
    access_manager: Arc<NamespaceAccessManager>,
}

impl McpCore {
    pub fn new(
        rag: Arc<RAGPipeline>,
        hybrid_searcher: Option<Arc<HybridSearcher>>,
        embedding_client: Arc<Mutex<EmbeddingClient>>,
        allowed_paths: Vec<String>,
        access_manager: Arc<NamespaceAccessManager>,
    ) -> Self {
        Self {
            rag,
            hybrid_searcher,
            embedding_client,
            allowed_paths,
            access_manager,
        }
    }

    pub fn rag(&self) -> Arc<RAGPipeline> {
        self.rag.clone()
    }

    pub async fn handle_jsonrpc_request(
        &self,
        request: Value,
        transport: McpTransport,
    ) -> McpDispatch {
        let method = request["method"].as_str().unwrap_or("");

        if method.starts_with("notifications/") {
            return McpDispatch::Notification;
        }

        let id = match request.get("id") {
            Some(value) if value.is_string() || value.is_number() => value.clone(),
            _ => {
                return McpDispatch::Response(json!({
                    "jsonrpc": "2.0",
                    "id": Value::Null,
                    "error": {
                        "code": -32600,
                        "message": "Invalid Request: missing or invalid 'id' field"
                    }
                }));
            }
        };

        let result = match method {
            "initialize" => Self::initialize_result(),
            "tools/list" => Self::tools_list_result(),
            "tools/call" => match self.handle_tool_call(&request, &id, transport).await {
                Ok(result) => result,
                Err(response) => return McpDispatch::Response(response),
            },
            _ => {
                return McpDispatch::Response(jsonrpc_error(
                    Some(&id),
                    -32601,
                    format!("Unknown method: {}", method),
                ));
            }
        };

        McpDispatch::Response(jsonrpc_success(&id, result))
    }

    fn initialize_result() -> Value {
        json!({
            "protocolVersion": PROTOCOL_VERSION,
            "serverInfo": {
                "name": SERVER_NAME,
                "version": env!("CARGO_PKG_VERSION")
            },
            "capabilities": {
                "tools": {},
                "resources": {}
            }
        })
    }

    fn tools_list_result() -> Value {
        json!({
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
                            "namespace": {"type": "string"},
                            "mode": {"type": "string", "enum": ["vector", "bm25", "hybrid"], "default": "hybrid", "description": "Search mode: vector (semantic), bm25 (keyword), hybrid (both)"},
                            "auto_route": {"type": "boolean", "default": false, "description": "Auto-detect query intent and select optimal search mode. Overrides mode when true."}
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
                            "mode": {"type": "string", "enum": ["vector", "bm25", "hybrid"], "default": "hybrid", "description": "Search mode: vector (semantic), bm25 (keyword), hybrid (both)"},
                            "auto_route": {"type": "boolean", "default": false, "description": "Auto-detect query intent and select optimal search mode. Overrides mode when true."},
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
                },
                {
                    "name": "dive",
                    "description": "Deep exploration with all onion layers. Shows ALL layers (outer/middle/inner/core), both BM25 and vector scores, full metadata, and related chunks.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "namespace": {"type": "string", "description": "Namespace to search in"},
                            "query": {"type": "string", "description": "Search query text"},
                            "limit": {"type": "integer", "default": 5, "description": "Maximum results per layer"},
                            "verbose": {"type": "boolean", "default": false, "description": "Show full text and metadata"}
                        },
                        "required": ["namespace", "query"]
                    }
                }
            ]
        })
    }

    async fn handle_tool_call(
        &self,
        request: &Value,
        id: &Value,
        transport: McpTransport,
    ) -> std::result::Result<Value, Value> {
        let tool_name = request["params"]["name"].as_str().unwrap_or("");
        let args = &request["params"]["arguments"];

        match tool_name {
            "health" => {
                let mut status = json!({
                    "version": env!("CARGO_PKG_VERSION"),
                    "db_path": self.rag.storage().lance_path(),
                    "backend": "mlx",
                    "mlx_server": self.rag.mlx_connected_to(),
                });

                if let Some(transport_name) = transport.health_transport() {
                    status["transport"] = json!(transport_name);
                }

                Ok(text_result_from_json(&status))
            }
            "rag_index" => {
                let path_str = args["path"].as_str().unwrap_or("");
                let namespace = args["namespace"].as_str();

                let validated_path = validate_path(path_str, &self.allowed_paths)
                    .map_err(|e| jsonrpc_error(Some(id), -32602, e.to_string()))?;

                match self.rag.index_document(&validated_path, namespace).await {
                    Ok(_) => Ok(text_result(format!("Indexed: {}", path_str))),
                    Err(e) => Ok(tool_error(e)),
                }
            }
            "rag_index_text" => {
                let text = args["text"].as_str().unwrap_or("").to_string();
                let namespace = args["namespace"].as_str();
                let metadata = args.get("metadata").cloned().unwrap_or_else(|| json!({}));
                let item_id = args
                    .get("id")
                    .and_then(|value| value.as_str().map(ToOwned::to_owned))
                    .unwrap_or_else(|| Uuid::new_v4().to_string());

                match self
                    .rag
                    .index_text(namespace, item_id.clone(), text, metadata)
                    .await
                {
                    Ok(returned_id) => {
                        Ok(text_result(format!("Indexed text with id {}", returned_id)))
                    }
                    Err(e) => Ok(tool_error(e)),
                }
            }
            "rag_search" => {
                let query = args["query"].as_str().unwrap_or("");
                let limit = args["k"].as_u64().unwrap_or(10) as usize;
                let namespace = args["namespace"].as_str();
                let mode = requested_search_mode(query, args);

                if let Some(hybrid_result) = self
                    .try_hybrid_search(query, namespace, limit, mode, id, SearchShape::Rag)
                    .await?
                {
                    return Ok(hybrid_result);
                }

                match self.rag.search_inner(namespace, query, limit).await {
                    Ok(results) => Ok(text_result_from_json(&results)),
                    Err(e) => Ok(tool_error(e)),
                }
            }
            "memory_upsert" => {
                let namespace = args["namespace"].as_str().unwrap_or("default");
                let token = args["token"].as_str();

                self.access_manager
                    .verify_access(namespace, token)
                    .await
                    .map_err(|e| jsonrpc_error(Some(id), -32603, e.to_string()))?;

                let item_id = args["id"].as_str().unwrap_or("").to_string();
                let text = args["text"].as_str().unwrap_or("").to_string();
                let metadata = args.get("metadata").cloned().unwrap_or_else(|| json!({}));

                match self
                    .rag
                    .memory_upsert(namespace, item_id.clone(), text, metadata)
                    .await
                {
                    Ok(_) => Ok(text_result(format!("Upserted {}", item_id))),
                    Err(e) => Ok(tool_error(e)),
                }
            }
            "memory_get" => {
                let namespace = args["namespace"].as_str().unwrap_or("default");
                let token = args["token"].as_str();

                self.access_manager
                    .verify_access(namespace, token)
                    .await
                    .map_err(|e| jsonrpc_error(Some(id), -32603, e.to_string()))?;

                let item_id = args["id"].as_str().unwrap_or("");
                match self.rag.memory_get(namespace, item_id).await {
                    Ok(Some(doc)) => Ok(text_result_from_json(&doc)),
                    Ok(None) => Ok(text_result("Not found")),
                    Err(e) => Ok(tool_error(e)),
                }
            }
            "memory_search" => {
                let namespace = args["namespace"].as_str().unwrap_or("default");
                let token = args["token"].as_str();

                self.access_manager
                    .verify_access(namespace, token)
                    .await
                    .map_err(|e| jsonrpc_error(Some(id), -32603, e.to_string()))?;

                let query = args["query"].as_str().unwrap_or("");
                let limit = args["k"].as_u64().unwrap_or(5) as usize;
                let mode = requested_search_mode(query, args);

                if let Some(hybrid_result) = self
                    .try_hybrid_search(query, Some(namespace), limit, mode, id, SearchShape::Memory)
                    .await?
                {
                    return Ok(hybrid_result);
                }

                match self.rag.memory_search(namespace, query, limit).await {
                    Ok(results) => Ok(text_result_from_json(&results)),
                    Err(e) => Ok(tool_error(e)),
                }
            }
            "memory_delete" => {
                let namespace = args["namespace"].as_str().unwrap_or("default");
                let token = args["token"].as_str();

                self.access_manager
                    .verify_access(namespace, token)
                    .await
                    .map_err(|e| jsonrpc_error(Some(id), -32603, e.to_string()))?;

                let item_id = args["id"].as_str().unwrap_or("");
                match self.rag.memory_delete(namespace, item_id).await {
                    Ok(deleted) => Ok(text_result(format!("Deleted {} rows", deleted))),
                    Err(e) => Ok(tool_error(e)),
                }
            }
            "memory_purge_namespace" => {
                let namespace = args["namespace"].as_str().unwrap_or("default");
                let token = args["token"].as_str();

                self.access_manager
                    .verify_access(namespace, token)
                    .await
                    .map_err(|e| jsonrpc_error(Some(id), -32603, e.to_string()))?;

                match self.rag.purge_namespace(namespace).await {
                    Ok(deleted) => Ok(text_result(format!(
                        "Purged namespace '{}', removed {} rows",
                        namespace, deleted
                    ))),
                    Err(e) => Ok(tool_error(e)),
                }
            }
            "namespace_create_token" => {
                let namespace = args["namespace"].as_str().unwrap_or("");
                let description = args["description"].as_str().map(ToOwned::to_owned);

                if namespace.is_empty() {
                    return Ok(tool_error_message("Namespace is required"));
                }

                match self
                    .access_manager
                    .create_token(namespace, description)
                    .await
                {
                    Ok(token) => Ok(text_result(format!(
                        "Token created for namespace '{}'. Store this token securely - it won't be shown again!\n\nToken: {}",
                        namespace, token
                    ))),
                    Err(e) => Ok(tool_error(e)),
                }
            }
            "namespace_revoke_token" => {
                let namespace = args["namespace"].as_str().unwrap_or("");

                if namespace.is_empty() {
                    return Ok(tool_error_message("Namespace is required"));
                }

                match self.access_manager.revoke_token(namespace).await {
                    Ok(true) => Ok(text_result(format!(
                        "Token revoked for namespace '{}'. The namespace is now publicly accessible.",
                        namespace
                    ))),
                    Ok(false) => Ok(text_result(format!(
                        "No token found for namespace '{}'.",
                        namespace
                    ))),
                    Err(e) => Ok(tool_error(e)),
                }
            }
            "namespace_list_protected" => {
                let protected = self.access_manager.list_protected_namespaces().await;
                if protected.is_empty() {
                    Ok(text_result(
                        "No namespaces are currently protected with tokens.",
                    ))
                } else {
                    let list: Vec<Value> = protected
                        .iter()
                        .map(|(namespace, created_at, description)| {
                            json!({
                                "namespace": namespace,
                                "created_at": created_at,
                                "description": description
                            })
                        })
                        .collect();
                    Ok(pretty_text_result_from_json(&list))
                }
            }
            "namespace_security_status" => {
                let enabled = self.access_manager.is_enabled();
                let protected_count = self.access_manager.list_protected_namespaces().await.len();

                Ok(text_result(format!(
                    "Namespace security: {}\nProtected namespaces: {}\n\nNote: When security is disabled, all namespaces are accessible without tokens.",
                    if enabled { "ENABLED" } else { "DISABLED" },
                    protected_count
                )))
            }
            "dive" => {
                let namespace = args["namespace"].as_str().unwrap_or("");
                let query = args["query"].as_str().unwrap_or("");
                let limit = args["limit"].as_u64().unwrap_or(5) as usize;
                let verbose = args["verbose"].as_bool().unwrap_or(false);

                if namespace.is_empty() || query.is_empty() {
                    return Err(jsonrpc_error(
                        Some(id),
                        -32602,
                        "namespace and query are required",
                    ));
                }

                let layers = [
                    (Some(SliceLayer::Outer), "outer"),
                    (Some(SliceLayer::Middle), "middle"),
                    (Some(SliceLayer::Inner), "inner"),
                    (Some(SliceLayer::Core), "core"),
                ];

                let mut all_results: Vec<Value> = Vec::new();

                for (layer_filter, layer_name) in &layers {
                    match self
                        .rag
                        .memory_search_with_layer(namespace, query, limit, *layer_filter)
                        .await
                    {
                        Ok(results) => {
                            let layer_results: Vec<Value> = results
                                .iter()
                                .map(|result| {
                                    let mut object = json!({
                                        "id": result.id,
                                        "score": result.score,
                                        "keywords": result.keywords,
                                        "layer": result.layer.map(|layer| layer.name()),
                                        "can_expand": result.can_expand(),
                                        "parent_id": result.parent_id,
                                    });

                                    if verbose {
                                        object["text"] = json!(result.text);
                                        object["metadata"] = result.metadata.clone();
                                        object["children_ids"] = json!(result.children_ids);
                                    } else {
                                        let preview: String =
                                            result.text.chars().take(200).collect();
                                        object["preview"] = json!(preview);
                                    }

                                    object
                                })
                                .collect();

                            all_results.push(json!({
                                "layer": layer_name,
                                "count": results.len(),
                                "results": layer_results
                            }));
                        }
                        Err(e) => {
                            all_results.push(json!({
                                "layer": layer_name,
                                "error": e.to_string()
                            }));
                        }
                    }
                }

                Ok(pretty_text_result_from_json(&json!({
                    "query": query,
                    "namespace": namespace,
                    "limit_per_layer": limit,
                    "verbose": verbose,
                    "layers": all_results
                })))
            }
            _ => Err(jsonrpc_error(
                Some(id),
                -32601,
                format!("Unknown tool: {}", tool_name),
            )),
        }
    }

    async fn try_hybrid_search(
        &self,
        query: &str,
        namespace: Option<&str>,
        limit: usize,
        mode: SearchMode,
        id: &Value,
        shape: SearchShape,
    ) -> std::result::Result<Option<Value>, Value> {
        if mode == SearchMode::Vector {
            return Ok(None);
        }

        let Some(hybrid_searcher) = &self.hybrid_searcher else {
            return Ok(None);
        };

        let query_embedding = self
            .embedding_client
            .lock()
            .await
            .embed(query)
            .await
            .map_err(|e| jsonrpc_error(Some(id), -32603, format!("Embedding failed: {}", e)))?;

        let results = hybrid_searcher
            .search(query, query_embedding, namespace, limit, None)
            .await
            .map_err(|e| jsonrpc_error(Some(id), -32603, format!("Hybrid search failed: {}", e)))?;

        let payload: Vec<Value> = match shape {
            SearchShape::Rag => results
                .iter()
                .map(|result| {
                    json!({
                        "id": result.id,
                        "namespace": result.namespace,
                        "document": result.document,
                        "combined_score": result.combined_score,
                        "vector_score": result.vector_score,
                        "bm25_score": result.bm25_score,
                        "metadata": result.metadata,
                        "layer": result.layer.as_ref().map(|layer| format!("{:?}", layer)),
                        "keywords": result.keywords
                    })
                })
                .collect(),
            SearchShape::Memory => results
                .iter()
                .map(|result| {
                    json!({
                        "id": result.id,
                        "namespace": result.namespace,
                        "text": result.document,
                        "score": result.combined_score,
                        "vector_score": result.vector_score,
                        "bm25_score": result.bm25_score,
                        "metadata": result.metadata
                    })
                })
                .collect(),
        };

        Ok(Some(text_result_from_json(&payload)))
    }
}

#[cfg(test)]
pub(crate) fn shared_initialize_result() -> Value {
    McpCore::initialize_result()
}

#[cfg(test)]
pub(crate) fn shared_tools_list_result() -> Value {
    McpCore::tools_list_result()
}

#[derive(Clone, Copy)]
enum SearchShape {
    Rag,
    Memory,
}

fn requested_search_mode(query: &str, args: &Value) -> SearchMode {
    if args["auto_route"].as_bool().unwrap_or(false) {
        let router = QueryRouter::new();
        let decision = router.route(query);
        match decision.recommended_mode.mode {
            SearchModeRecommendation::Vector => SearchMode::Vector,
            SearchModeRecommendation::Bm25 => SearchMode::Keyword,
            SearchModeRecommendation::Hybrid => SearchMode::Hybrid,
        }
    } else {
        match args["mode"].as_str() {
            Some("vector") => SearchMode::Vector,
            Some("bm25") | Some("keyword") => SearchMode::Keyword,
            _ => SearchMode::Hybrid,
        }
    }
}

fn tool_error(error: impl ToString) -> Value {
    tool_error_message(error.to_string())
}

fn tool_error_message(message: impl Into<String>) -> Value {
    json!({
        "error": {"message": message.into()}
    })
}

fn text_result(text: impl Into<String>) -> Value {
    json!({
        "content": [{"type": "text", "text": text.into()}]
    })
}

fn text_result_from_json<T: serde::Serialize>(value: &T) -> Value {
    text_result(serde_json::to_string(value).unwrap_or_default())
}

fn pretty_text_result_from_json<T: serde::Serialize>(value: &T) -> Value {
    text_result(serde_json::to_string_pretty(value).unwrap_or_default())
}

/// Validates a file path to prevent path traversal attacks.
/// Returns the canonicalized path if valid, or an error if the path is unsafe.
fn validate_path(path_str: &str, allowed_paths: &[String]) -> Result<std::path::PathBuf> {
    if path_str.is_empty() {
        return Err(anyhow!("Path cannot be empty"));
    }

    if path_str.contains("..") || path_str.contains('\0') || path_str.contains('\n') {
        return Err(anyhow!(
            "Path traversal detected: invalid sequences in '{}'",
            path_str
        ));
    }

    let canonical = crate::path_utils::sanitize_existing_path(path_str)?;

    let is_safe = if allowed_paths.is_empty() {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .map(std::path::PathBuf::from)
            .ok();
        let cwd = std::env::current_dir().ok();

        home.as_ref()
            .map(|path| canonical.starts_with(path))
            .unwrap_or(false)
            || cwd
                .as_ref()
                .map(|path| canonical.starts_with(path))
                .unwrap_or(false)
    } else {
        allowed_paths.iter().any(|allowed| {
            let expanded_allowed = shellexpand::tilde(allowed).to_string();
            let allowed_path = Path::new(&expanded_allowed);
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

#[cfg(test)]
mod tests {
    use super::{McpCore, jsonrpc_error, jsonrpc_success};
    use serde_json::{Value, json};

    #[test]
    fn jsonrpc_error_omits_missing_id() {
        let response = jsonrpc_error(None, -32600, "boom");
        assert_eq!(response["jsonrpc"], "2.0");
        assert_eq!(response["error"]["code"], -32600);
        assert_eq!(response.get("id"), None);
    }

    #[test]
    fn jsonrpc_success_omits_null_id() {
        let response = jsonrpc_success(&Value::Null, json!({"ok": true}));
        assert_eq!(response["jsonrpc"], "2.0");
        assert_eq!(response["result"]["ok"], true);
        assert_eq!(response.get("id"), None);
    }

    #[test]
    fn initialize_advertises_resources_and_tools() {
        let response = McpCore::initialize_result();
        assert_eq!(response["protocolVersion"], "2024-11-05");
        assert!(response["capabilities"]["tools"].is_object());
        assert!(response["capabilities"]["resources"].is_object());
    }

    #[test]
    fn tool_list_contains_extended_stdio_and_http_surface() {
        let result = McpCore::tools_list_result();
        let tools = result["tools"]
            .as_array()
            .expect("tools list should be an array");
        let names: Vec<&str> = tools
            .iter()
            .filter_map(|tool| tool["name"].as_str())
            .collect();

        assert!(names.contains(&"rag_index"));
        assert!(names.contains(&"memory_purge_namespace"));
        assert!(names.contains(&"namespace_create_token"));
        assert!(names.contains(&"dive"));
    }
}
