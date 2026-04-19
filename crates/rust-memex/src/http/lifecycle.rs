use std::convert::Infallible;
use std::io;
use std::path::PathBuf;
use std::time::Duration;

use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::{Multipart, State},
    http::{HeaderValue, StatusCode, header},
    response::{
        IntoResponse,
        sse::{Event, Sse},
    },
    routing::post,
};
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::{
    ReindexJob, ReprocessJob, SliceMode, default_reindexed_namespace,
    export_namespace_jsonl_stream, import_jsonl_file, migrate_namespace_atomic, reindex_namespace,
    reprocess_jsonl_file,
};

use super::HttpState;

#[derive(Debug, Deserialize)]
struct ReprocessRequest {
    input_path: String,
    target_namespace: String,
    slice_mode: String,
    #[serde(default)]
    preprocess: bool,
    #[serde(default)]
    skip_existing: bool,
}

#[derive(Debug, Deserialize)]
struct ReindexRequest {
    source_namespace: String,
    target_namespace: Option<String>,
    slice_mode: String,
    #[serde(default)]
    preprocess: bool,
    #[serde(default)]
    skip_existing: bool,
}

#[derive(Debug, Deserialize)]
struct ExportRequest {
    namespace: String,
    #[serde(default)]
    include_embeddings: bool,
}

#[derive(Debug, Deserialize)]
struct MigrateNamespaceRequest {
    from: String,
    to: String,
}

pub(super) fn routes() -> Router<HttpState> {
    Router::new()
        .route("/sse/reprocess", post(sse_reprocess_handler))
        .route("/sse/reindex", post(sse_reindex_handler))
        .route("/api/export", post(export_handler))
        .route("/api/import", post(import_handler))
        .route("/api/migrate-namespace", post(migrate_namespace_handler))
}

async fn sse_reprocess_handler(
    State(state): State<HttpState>,
    Json(request): Json<ReprocessRequest>,
) -> Result<Sse<impl futures::Stream<Item = Result<Event, Infallible>>>, (StatusCode, String)> {
    let slice_mode: SliceMode = request.slice_mode.parse().map_err(|err| {
        (
            StatusCode::BAD_REQUEST,
            format!("invalid slice_mode '{}': {}", request.slice_mode, err),
        )
    })?;

    let input_path = PathBuf::from(request.input_path.clone());
    let target_namespace = request.target_namespace.clone();
    let slice_mode_name = request.slice_mode.clone();
    let preprocess = request.preprocess;
    let skip_existing = request.skip_existing;
    let (tx, mut rx) = mpsc::unbounded_channel();
    let rag = state.rag.clone();

    tokio::spawn(async move {
        let _ = tx.send(sse_event(
            "start",
            json!({
                "input_path": input_path.display().to_string(),
                "target_namespace": target_namespace.clone(),
                "slice_mode": slice_mode_name.clone(),
                "preprocess": preprocess,
                "skip_existing": skip_existing,
            }),
        ));

        let result = reprocess_jsonl_file(
            rag,
            ReprocessJob {
                input_path: input_path.clone(),
                target_namespace: target_namespace.clone(),
                slice_mode,
                preprocess,
                skip_existing,
                dry_run: false,
            },
            |progress| {
                let _ = tx.send(sse_event(
                    "progress",
                    serde_json::to_value(progress).unwrap_or(Value::Null),
                ));
            },
        )
        .await;

        match result {
            Ok(summary) => {
                let _ = tx.send(sse_event(
                    "result",
                    serde_json::to_value(summary).unwrap_or(Value::Null),
                ));
            }
            Err(err) => {
                let _ = tx.send(sse_event("error", json!({ "error": err.to_string() })));
            }
        }
    });

    let stream = async_stream::stream! {
        while let Some(event) = rx.recv().await {
            yield Ok(to_axum_event(event));
        }
    };

    Ok(Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("ping"),
    ))
}

async fn sse_reindex_handler(
    State(state): State<HttpState>,
    Json(request): Json<ReindexRequest>,
) -> Result<Sse<impl futures::Stream<Item = Result<Event, Infallible>>>, (StatusCode, String)> {
    let slice_mode: SliceMode = request.slice_mode.parse().map_err(|err| {
        (
            StatusCode::BAD_REQUEST,
            format!("invalid slice_mode '{}': {}", request.slice_mode, err),
        )
    })?;

    let source_namespace = request.source_namespace.clone();
    let target_namespace = request
        .target_namespace
        .clone()
        .unwrap_or_else(|| default_reindexed_namespace(&source_namespace));
    let slice_mode_name = request.slice_mode.clone();
    let preprocess = request.preprocess;
    let skip_existing = request.skip_existing;
    let (tx, mut rx) = mpsc::unbounded_channel();
    let rag = state.rag.clone();

    tokio::spawn(async move {
        let _ = tx.send(sse_event(
            "start",
            json!({
                "source_namespace": source_namespace.clone(),
                "target_namespace": target_namespace.clone(),
                "slice_mode": slice_mode_name.clone(),
                "preprocess": preprocess,
                "skip_existing": skip_existing,
            }),
        ));

        let result = reindex_namespace(
            rag,
            ReindexJob {
                source_namespace: source_namespace.clone(),
                target_namespace: target_namespace.clone(),
                slice_mode,
                preprocess,
                skip_existing,
                dry_run: false,
            },
            |progress| {
                let _ = tx.send(sse_event(
                    "progress",
                    serde_json::to_value(progress).unwrap_or(Value::Null),
                ));
            },
        )
        .await;

        match result {
            Ok(summary) => {
                let _ = tx.send(sse_event(
                    "result",
                    serde_json::to_value(summary).unwrap_or(Value::Null),
                ));
            }
            Err(err) => {
                let _ = tx.send(sse_event("error", json!({ "error": err.to_string() })));
            }
        }
    });

    let stream = async_stream::stream! {
        while let Some(event) = rx.recv().await {
            yield Ok(to_axum_event(event));
        }
    };

    Ok(Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("ping"),
    ))
}

async fn export_handler(
    State(state): State<HttpState>,
    Json(request): Json<ExportRequest>,
) -> impl IntoResponse {
    let namespace = request.namespace.clone();
    let stream = export_namespace_jsonl_stream(
        state.rag.storage_manager(),
        request.namespace.clone(),
        request.include_embeddings,
    )
    .map(|item| item.map(Bytes::from).map_err(io::Error::other));

    let mut headers = axum::http::HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/x-ndjson"),
    );
    if let Ok(value) = HeaderValue::from_str(&format!(
        "attachment; filename=\"{}.jsonl\"",
        sanitize_filename(&namespace)
    )) {
        headers.insert(header::CONTENT_DISPOSITION, value);
    }

    (StatusCode::OK, headers, Body::from_stream(stream))
}

async fn import_handler(
    State(state): State<HttpState>,
    mut multipart: Multipart,
) -> Result<Json<Value>, (StatusCode, String)> {
    let mut namespace = None;
    let mut skip_existing = false;
    let mut upload_path = None;

    while let Some(field) = multipart.next_field().await.map_err(internal_error)? {
        let field_name = field.name().unwrap_or_default().to_string();
        match field_name.as_str() {
            "namespace" => {
                namespace = Some(field.text().await.map_err(internal_error)?);
            }
            "skip_existing" => {
                let value = field.text().await.map_err(internal_error)?;
                skip_existing = parse_bool_field(&value)?;
            }
            "file" => {
                let path = temp_upload_path();
                let mut file = tokio::fs::File::create(&path)
                    .await
                    .map_err(internal_error)?;
                let mut field = field;
                while let Some(chunk) = field.chunk().await.map_err(internal_error)? {
                    file.write_all(&chunk).await.map_err(internal_error)?;
                }
                file.flush().await.map_err(internal_error)?;
                upload_path = Some(path);
            }
            _ => {}
        }
    }

    let namespace = namespace.ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            "missing multipart field 'namespace'".to_string(),
        )
    })?;
    let upload_path = upload_path.ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            "missing multipart file field 'file'".to_string(),
        )
    })?;

    let outcome = import_jsonl_file(state.rag.clone(), namespace, &upload_path, skip_existing)
        .await
        .map_err(internal_error)?;
    let _ = tokio::fs::remove_file(&upload_path).await;

    Ok(Json(json!({ "imported_count": outcome.imported_count })))
}

async fn migrate_namespace_handler(
    State(state): State<HttpState>,
    Json(request): Json<MigrateNamespaceRequest>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let storage = state.rag.storage_manager();
    let outcome = migrate_namespace_atomic(storage.as_ref(), &request.from, &request.to)
        .await
        .map_err(internal_error)?;
    Ok(Json(json!({ "migrated_chunks": outcome.migrated_chunks })))
}

fn sse_event(event: &str, data: Value) -> crate::contracts::progress::SseEvent {
    crate::contracts::progress::SseEvent {
        event: event.to_string(),
        id: None,
        data,
    }
}

fn to_axum_event(event: crate::contracts::progress::SseEvent) -> Event {
    let mut axum_event = Event::default()
        .event(event.event)
        .data(event.data.to_string());
    if let Some(id) = event.id {
        axum_event = axum_event.id(id);
    }
    axum_event
}

fn sanitize_filename(namespace: &str) -> String {
    namespace
        .chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' => ch,
            _ => '_',
        })
        .collect()
}

fn temp_upload_path() -> PathBuf {
    std::env::temp_dir().join(format!("rust-memex-import-{}.jsonl", Uuid::new_v4()))
}

fn parse_bool_field(value: &str) -> Result<bool, (StatusCode, String)> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" | "" => Ok(false),
        other => Err((
            StatusCode::BAD_REQUEST,
            format!("invalid boolean field value '{}'", other),
        )),
    }
}

fn internal_error(err: impl std::fmt::Display) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, err.to_string())
}
