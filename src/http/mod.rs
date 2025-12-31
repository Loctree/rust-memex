//! HTTP/SSE server for rmcp-memex
//!
//! Provides HTTP endpoints for agents that can't hold LanceDB lock directly.
//! All database access goes through the single server instance.
//!
//! Uses RAGPipeline (same as MCPServer) for consistency and full feature support:
//! - Multi-namespace (each agent can have own namespace)
//! - Onion slices (expand/drill-down in SSE)
//! - Full indexing pipeline with dedup
//!
//! Endpoints:
//! - GET  /health           - Health check
//! - POST /search           - Search documents
//! - GET  /sse/search       - SSE streaming search
//! - POST /upsert           - Upsert document (memory_upsert)
//! - POST /index            - Index text with full pipeline
//! - GET  /expand/:ns/:id   - Expand onion slice (get children)
//! - GET  /parent/:ns/:id   - Get parent slice (drill up)
//! - DELETE /ns/:namespace  - Purge namespace
//!
//! Created by M&K (c)2025 The LibraxisAI Team

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::StatusCode,
    response::{
        IntoResponse,
        sse::{Event, Sse},
    },
    routing::{delete, get, post},
};
use serde::{Deserialize, Serialize};
use tower_http::cors::{Any, CorsLayer};
use tracing::{error, info};

use crate::rag::{RAGPipeline, SearchResult, SliceLayer};

/// Shared state for HTTP handlers - uses RAGPipeline like MCPServer
#[derive(Clone)]
pub struct HttpState {
    pub rag: Arc<RAGPipeline>,
}

/// Search request body
#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default)]
    pub namespace: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Optional layer filter for onion slices
    #[serde(default)]
    pub layer: Option<u8>,
}

fn default_limit() -> usize {
    10
}

/// Search result for JSON response
#[derive(Debug, Serialize)]
pub struct SearchResultJson {
    pub id: String,
    pub namespace: String,
    pub text: String,
    pub score: f32,
    pub metadata: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub children_ids: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub keywords: Vec<String>,
    /// Can expand to children (has children_ids)
    pub can_expand: bool,
    /// Can drill up to parent (has parent_id)
    pub can_drill_up: bool,
}

impl From<SearchResult> for SearchResultJson {
    fn from(r: SearchResult) -> Self {
        let can_expand = r.can_expand();
        let can_drill_up = r.can_drill_up();
        Self {
            id: r.id,
            namespace: r.namespace,
            text: r.text,
            score: r.score,
            metadata: r.metadata,
            layer: r.layer.map(|l| l.name().to_string()),
            parent_id: r.parent_id,
            children_ids: r.children_ids,
            keywords: r.keywords,
            can_expand,
            can_drill_up,
        }
    }
}

/// Search response
#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResultJson>,
    pub query: String,
    pub namespace: Option<String>,
    pub elapsed_ms: u64,
    pub count: usize,
}

/// Upsert request body (memory_upsert)
#[derive(Debug, Deserialize)]
pub struct UpsertRequest {
    pub namespace: String,
    pub id: String,
    pub content: String,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

/// Index text request (full pipeline)
#[derive(Debug, Deserialize)]
pub struct IndexRequest {
    pub namespace: String,
    pub content: String,
    /// Slice mode: "flat", "outer", "deep" (default: "flat")
    #[serde(default = "default_slice_mode")]
    pub slice_mode: String,
}

fn default_slice_mode() -> String {
    "flat".to_string()
}

/// SSE search query params
#[derive(Debug, Deserialize)]
pub struct SseSearchParams {
    pub query: String,
    #[serde(default)]
    pub namespace: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

/// Health check response
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub db_path: String,
    pub embedding_provider: String,
}

/// Create the HTTP router
pub fn create_router(state: HttpState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/health", get(health_handler))
        .route("/search", post(search_handler))
        .route("/sse/search", get(sse_search_handler))
        .route("/upsert", post(upsert_handler))
        .route("/index", post(index_handler))
        .route("/expand/{ns}/{id}", get(expand_handler))
        .route("/parent/{ns}/{id}", get(parent_handler))
        .route("/get/{ns}/{id}", get(get_handler))
        .route("/delete/{ns}/{id}", post(delete_handler))
        .route("/ns/{namespace}", delete(purge_namespace_handler))
        .layer(cors)
        .with_state(state)
}

/// Health check endpoint
async fn health_handler(State(state): State<HttpState>) -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok".to_string(),
        db_path: state.rag.storage().lance_path().to_string(),
        embedding_provider: state.rag.mlx_connected_to(),
    })
}

/// Search endpoint (POST /search)
async fn search_handler(
    State(state): State<HttpState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, String)> {
    let start = std::time::Instant::now();

    let results = if let Some(layer_u8) = req.layer {
        // Search with layer filter
        let layer = SliceLayer::from_u8(layer_u8);
        state
            .rag
            .memory_search_with_layer(
                req.namespace.as_deref().unwrap_or("default"),
                &req.query,
                req.limit,
                layer,
            )
            .await
    } else {
        // Regular search
        state
            .rag
            .memory_search(
                req.namespace.as_deref().unwrap_or("default"),
                &req.query,
                req.limit,
            )
            .await
    }
    .map_err(|e| {
        error!("Search error: {}", e);
        (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    })?;

    let count = results.len();
    let search_results: Vec<SearchResultJson> = results.into_iter().map(Into::into).collect();

    Ok(Json(SearchResponse {
        results: search_results,
        query: req.query,
        namespace: req.namespace,
        elapsed_ms: start.elapsed().as_millis() as u64,
        count,
    }))
}

/// SSE streaming search endpoint (GET /sse/search?query=...&namespace=...&limit=...)
async fn sse_search_handler(
    State(state): State<HttpState>,
    Query(params): Query<SseSearchParams>,
) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        // Send start event
        yield Ok(Event::default()
            .event("start")
            .data(serde_json::json!({
                "query": params.query,
                "namespace": params.namespace,
                "limit": params.limit
            }).to_string()));

        let namespace = params.namespace.as_deref().unwrap_or("default");

        match state.rag.memory_search(namespace, &params.query, params.limit).await {
            Ok(results) => {
                let total = results.len();

                for (i, r) in results.into_iter().enumerate() {
                    let result: SearchResultJson = r.into();

                    if let Ok(json) = serde_json::to_string(&result) {
                        yield Ok(Event::default()
                            .event("result")
                            .id(i.to_string())
                            .data(json));
                    }

                    // Small delay for streaming effect
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }

                yield Ok(Event::default()
                    .event("done")
                    .data(serde_json::json!({
                        "status": "complete",
                        "total": total
                    }).to_string()));
            }
            Err(e) => {
                yield Ok(Event::default()
                    .event("error")
                    .data(serde_json::json!({"error": e.to_string()}).to_string()));
            }
        }
    };

    Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("ping"),
    )
}

/// Upsert document endpoint (POST /upsert) - uses memory_upsert
async fn upsert_handler(
    State(state): State<HttpState>,
    Json(req): Json<UpsertRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let metadata = req.metadata.unwrap_or(serde_json::json!({}));

    state
        .rag
        .memory_upsert(
            &req.namespace,
            req.id.clone(),
            req.content.clone(),
            metadata,
        )
        .await
        .map_err(|e| {
            error!("Upsert error: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
        })?;

    Ok(Json(serde_json::json!({
        "status": "ok",
        "id": req.id,
        "namespace": req.namespace
    })))
}

/// Index text with full pipeline (POST /index)
async fn index_handler(
    State(state): State<HttpState>,
    Json(req): Json<IndexRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    use crate::rag::SliceMode;

    let mode = match req.slice_mode.as_str() {
        "onion" => SliceMode::Onion,
        "onion_fast" | "fast" => SliceMode::OnionFast,
        _ => SliceMode::Flat,
    };

    // Generate ID from content hash
    let id = format!(
        "idx_{}",
        uuid::Uuid::new_v4()
            .to_string()
            .split('-')
            .next()
            .unwrap_or("000")
    );

    let result_id = state
        .rag
        .index_text_with_mode(
            Some(&req.namespace),
            id,
            req.content.clone(),
            serde_json::json!({}),
            mode,
        )
        .await
        .map_err(|e| {
            error!("Index error: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
        })?;

    Ok(Json(serde_json::json!({
        "status": "indexed",
        "namespace": req.namespace,
        "id": result_id,
        "slice_mode": req.slice_mode
    })))
}

/// Expand onion slice - get children (GET /expand/:ns/:id)
async fn expand_handler(
    State(state): State<HttpState>,
    Path((ns, id)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let children = state.rag.expand_result(&ns, &id).await.map_err(|e| {
        error!("Expand error: {}", e);
        (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    })?;

    let results: Vec<SearchResultJson> = children.into_iter().map(Into::into).collect();

    Ok(Json(serde_json::json!({
        "parent_id": id,
        "namespace": ns,
        "children": results,
        "count": results.len()
    })))
}

/// Get parent slice - drill up (GET /parent/:ns/:id)
async fn parent_handler(
    State(state): State<HttpState>,
    Path((ns, id)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    match state.rag.get_parent_result(&ns, &id).await {
        Ok(Some(parent)) => {
            let result: SearchResultJson = parent.into();
            Ok(Json(serde_json::json!({
                "child_id": id,
                "namespace": ns,
                "parent": result
            })))
        }
        Ok(None) => Err((StatusCode::NOT_FOUND, format!("No parent for '{}'", id))),
        Err(e) => {
            error!("Parent error: {}", e);
            Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}

/// Get document by namespace and ID (GET /get/:ns/:id)
async fn get_handler(
    State(state): State<HttpState>,
    Path((ns, id)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    match state.rag.memory_get(&ns, &id).await {
        Ok(Some(r)) => {
            let result: SearchResultJson = r.into();
            Ok(Json(serde_json::json!(result)))
        }
        Ok(None) => Err((
            StatusCode::NOT_FOUND,
            format!("Document '{}' not found in '{}'", id, ns),
        )),
        Err(e) => {
            error!("Get error: {}", e);
            Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}

/// Delete document (POST /delete/:ns/:id)
async fn delete_handler(
    State(state): State<HttpState>,
    Path((ns, id)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    match state.rag.memory_delete(&ns, &id).await {
        Ok(deleted) => Ok(Json(serde_json::json!({
            "status": if deleted > 0 { "deleted" } else { "not_found" },
            "id": id,
            "namespace": ns
        }))),
        Err(e) => {
            error!("Delete error: {}", e);
            Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}

/// Purge entire namespace (DELETE /ns/:namespace)
async fn purge_namespace_handler(
    State(state): State<HttpState>,
    Path(namespace): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    match state.rag.purge_namespace(&namespace).await {
        Ok(deleted) => Ok(Json(serde_json::json!({
            "status": "purged",
            "namespace": namespace,
            "deleted_count": deleted
        }))),
        Err(e) => {
            error!("Purge error: {}", e);
            Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}

/// Start the HTTP server with shared RAGPipeline
pub async fn start_server(rag: Arc<RAGPipeline>, port: u16) -> anyhow::Result<()> {
    let state = HttpState { rag };
    let app = create_router(state);

    let addr = format!("0.0.0.0:{}", port);
    info!("🚀 HTTP/SSE server starting on http://{}", addr);
    info!("   Endpoints: /search, /sse/search, /upsert, /expand, /parent");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_request_defaults() {
        let json = r#"{"query": "test"}"#;
        let req: SearchRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.limit, 10);
        assert!(req.namespace.is_none());
        assert!(req.layer.is_none());
    }

    #[test]
    fn test_index_request_defaults() {
        let json = r#"{"namespace": "test", "content": "hello"}"#;
        let req: IndexRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.slice_mode, "flat");
    }
}
