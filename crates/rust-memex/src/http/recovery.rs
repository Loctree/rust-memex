use std::convert::Infallible;
use std::path::PathBuf;
use std::time::Duration;

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::sse::{Event, Sse},
    routing::post,
};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    GcConfig, cleanup_versions, collect_garbage, compact_database,
    contracts::progress::{CompactProgress, SseEvent},
    merge_databases,
    recovery::MaintenanceExecution,
    repair_writes,
};

use super::HttpState;

pub(super) fn routes() -> Router<HttpState> {
    Router::new()
        .route("/api/merge", post(merge_handler))
        .route("/api/repair-writes", post(repair_writes_handler))
        .route("/sse/compact", post(sse_compact_handler))
        .route("/sse/cleanup", post(sse_cleanup_handler))
        .route("/sse/gc", post(sse_gc_handler))
        .route("/sse/optimize", post(sse_optimize_handler))
}

#[derive(Debug, Deserialize)]
struct MergeRequest {
    sources: Vec<String>,
    target: String,
    #[serde(default)]
    namespace_prefix: Option<String>,
    #[serde(default)]
    dedup: bool,
    #[serde(default)]
    dry_run: bool,
}

#[derive(Debug, Deserialize)]
struct RepairWritesRequest {
    #[serde(default)]
    namespace: Option<String>,
    #[serde(default)]
    execute: bool,
    #[serde(default)]
    json_output: bool,
}

async fn merge_handler(
    Json(request): Json<MergeRequest>,
) -> Result<Json<Value>, (StatusCode, String)> {
    if request.sources.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "merge requires at least one source path".to_string(),
        ));
    }
    if request.target.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "merge requires a non-empty target path".to_string(),
        ));
    }

    let execution = merge_databases(
        request.sources.iter().map(PathBuf::from).collect(),
        PathBuf::from(&request.target),
        request.dedup,
        request.namespace_prefix.clone(),
        request.dry_run,
    )
    .await
    .map_err(internal_error)?;

    Ok(Json(json!({
        "target": execution.target_path.display().to_string(),
        "dry_run": request.dry_run,
        "dedup": request.dedup,
        "namespace_prefix": request.namespace_prefix,
        "progress": execution.progress,
    })))
}

async fn repair_writes_handler(
    State(state): State<HttpState>,
    Json(request): Json<RepairWritesRequest>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let execution = repair_writes(
        state.rag.storage_manager().lance_path(),
        request.namespace.as_deref(),
        request.execute,
    )
    .await
    .map_err(internal_error)?;

    let payload = if request.json_output {
        serde_json::to_value(&execution.report).map_err(internal_error)?
    } else {
        serde_json::to_value(&execution.result).map_err(internal_error)?
    };

    Ok(Json(payload))
}

async fn sse_compact_handler(
    State(state): State<HttpState>,
) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        let storage = state.rag.storage_manager();
        let pre_stats = storage.stats().await.ok();
        yield Ok(to_axum_event(sse_event("start", start_payload("compact", storage.lance_path(), pre_stats.as_ref()))));
        yield Ok(to_axum_event(running_phase_event("compact", "Merging small files into larger ones")));

        match compact_database(storage.as_ref()).await {
            Ok(outcome) => {
                yield Ok(to_axum_event(success_phase_event("compact_done", &outcome.progress, &outcome)));
                yield Ok(to_axum_event(done_event("compact", true, &outcome.progress, &outcome)));
            }
            Err(error) => {
                yield Ok(to_axum_event(error_phase_event("compact_error", "compact", &error.to_string())));
                yield Ok(to_axum_event(failed_done_event("compact", &error.to_string())));
            }
        }
    };

    sse_response(stream)
}

async fn sse_cleanup_handler(
    State(state): State<HttpState>,
) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        let storage = state.rag.storage_manager();
        let pre_stats = storage.stats().await.ok();
        yield Ok(to_axum_event(sse_event("start", start_payload("cleanup", storage.lance_path(), pre_stats.as_ref()))));
        yield Ok(to_axum_event(running_phase_event("cleanup", "Removing old versions (>7 days)")));

        match cleanup_versions(storage.as_ref(), Some(7)).await {
            Ok(outcome) => {
                yield Ok(to_axum_event(success_phase_event("cleanup_done", &outcome.progress, &outcome)));
                yield Ok(to_axum_event(done_event("cleanup", true, &outcome.progress, &outcome)));
            }
            Err(error) => {
                yield Ok(to_axum_event(error_phase_event("cleanup_error", "cleanup", &error.to_string())));
                yield Ok(to_axum_event(failed_done_event("cleanup", &error.to_string())));
            }
        }
    };

    sse_response(stream)
}

async fn sse_gc_handler(
    State(state): State<HttpState>,
) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        let storage = state.rag.storage_manager();
        let pre_stats = storage.stats().await.ok();
        let config = GcConfig {
            remove_orphans: true,
            dry_run: false,
            ..GcConfig::default()
        };

        yield Ok(to_axum_event(sse_event("start", start_payload("gc", storage.lance_path(), pre_stats.as_ref()))));
        yield Ok(to_axum_event(running_phase_event("gc", "Removing orphan embeddings")));

        match collect_garbage(storage.as_ref(), &config).await {
            Ok(outcome) => {
                yield Ok(to_axum_event(success_phase_event("gc_done", &outcome.progress, &outcome)));
                yield Ok(to_axum_event(done_event("gc", true, &outcome.progress, &outcome)));
            }
            Err(error) => {
                yield Ok(to_axum_event(error_phase_event("gc_error", "gc", &error.to_string())));
                yield Ok(to_axum_event(failed_done_event("gc", &error.to_string())));
            }
        }
    };

    sse_response(stream)
}

async fn sse_optimize_handler(
    State(state): State<HttpState>,
) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        let storage = state.rag.storage_manager();
        let start_stats = storage.stats().await.ok();
        yield Ok(to_axum_event(sse_event(
            "start",
            json!({
                "status": "starting_optimization",
                "db_path": storage.lance_path(),
                "pre_row_count": start_stats.as_ref().map(|s| s.row_count),
                "pre_version_count": start_stats.as_ref().map(|s| s.version_count),
            }),
        )));

        yield Ok(to_axum_event(running_phase_event("compact", "Merging small files into larger ones")));
        let compact_result = compact_database(storage.as_ref()).await;
        match &compact_result {
            Ok(outcome) => yield Ok(to_axum_event(success_phase_event("compact_done", &outcome.progress, outcome))),
            Err(error) => yield Ok(to_axum_event(error_phase_event("compact_error", "compact", &error.to_string()))),
        }

        tokio::time::sleep(Duration::from_millis(10)).await;

        yield Ok(to_axum_event(running_phase_event("prune", "Removing old versions (>7 days)")));
        let cleanup_result = cleanup_versions(storage.as_ref(), Some(7)).await;
        match &cleanup_result {
            Ok(outcome) => {
                let mut progress = outcome.progress.clone();
                progress.phase = "prune".to_string();
                yield Ok(to_axum_event(success_phase_event("prune_done", &progress, outcome)));
            }
            Err(error) => yield Ok(to_axum_event(error_phase_event("prune_error", "prune", &error.to_string()))),
        }

        let end_stats = storage.stats().await.ok();
        yield Ok(to_axum_event(sse_event(
            "done",
            json!({
                "status": "complete",
                "post_row_count": end_stats.as_ref().map(|s| s.row_count),
                "post_version_count": end_stats.as_ref().map(|s| s.version_count),
                "compact_ok": compact_result.is_ok(),
                "prune_ok": cleanup_result.is_ok(),
                "elapsed_ms": start_stats
                    .as_ref()
                    .and_then(|_| {
                        compact_result
                            .as_ref()
                            .ok()
                            .and_then(|outcome| outcome.progress.elapsed_ms)
                            .zip(cleanup_result.as_ref().ok().and_then(|outcome| outcome.progress.elapsed_ms))
                            .map(|(compact_ms, cleanup_ms)| compact_ms + cleanup_ms)
                    }),
            }),
        )));
    };

    sse_response(stream)
}

fn running_phase_event(phase: &str, description: &str) -> SseEvent {
    sse_event(
        "phase",
        json!({
            "phase": phase,
            "status": "running",
            "description": description,
        }),
    )
}

fn success_phase_event(
    event_name: &str,
    progress: &CompactProgress,
    outcome: &MaintenanceExecution,
) -> SseEvent {
    let mut data = compact_progress_payload(progress, outcome);
    if let Value::Object(ref mut object) = data {
        object.insert("status".to_string(), Value::String("complete".to_string()));
    }
    sse_event(event_name, data)
}

fn error_phase_event(event_name: &str, phase: &str, error: &str) -> SseEvent {
    sse_event(
        event_name,
        json!({
            "phase": phase,
            "status": "error",
            "error": error,
        }),
    )
}

fn done_event(
    action: &str,
    ok: bool,
    progress: &CompactProgress,
    outcome: &MaintenanceExecution,
) -> SseEvent {
    let mut payload = compact_progress_payload(progress, outcome);
    if let Value::Object(ref mut object) = payload {
        object.insert("action".to_string(), Value::String(action.to_string()));
        object.insert("ok".to_string(), Value::Bool(ok));
        object.insert("status".to_string(), Value::String("complete".to_string()));
    }
    sse_event("done", payload)
}

fn failed_done_event(action: &str, error: &str) -> SseEvent {
    sse_event(
        "done",
        json!({
            "action": action,
            "ok": false,
            "status": "error",
            "error": error,
        }),
    )
}

fn compact_progress_payload(progress: &CompactProgress, outcome: &MaintenanceExecution) -> Value {
    let mut value = serde_json::to_value(progress).expect("compact progress serializes");
    if let Value::Object(ref mut object) = value {
        object.insert(
            "pre_row_count".to_string(),
            json!(outcome.pre_stats.as_ref().map(|stats| stats.row_count)),
        );
        object.insert(
            "post_row_count".to_string(),
            json!(outcome.post_stats.as_ref().map(|stats| stats.row_count)),
        );
        object.insert(
            "pre_version_count".to_string(),
            json!(outcome.pre_stats.as_ref().map(|stats| stats.version_count)),
        );
        object.insert(
            "post_version_count".to_string(),
            json!(outcome.post_stats.as_ref().map(|stats| stats.version_count)),
        );
        object.insert(
            "row_delta".to_string(),
            json!(stat_delta(
                outcome.pre_stats.as_ref().map(|stats| stats.row_count),
                outcome.post_stats.as_ref().map(|stats| stats.row_count),
            )),
        );
        object.insert(
            "version_delta".to_string(),
            json!(stat_delta(
                outcome.pre_stats.as_ref().map(|stats| stats.version_count),
                outcome.post_stats.as_ref().map(|stats| stats.version_count),
            )),
        );
        if let Some(gc_stats) = outcome.gc_stats.as_ref() {
            object.insert("orphans_found".to_string(), json!(gc_stats.orphans_found));
            object.insert(
                "orphans_removed".to_string(),
                json!(gc_stats.orphans_removed),
            );
            object.insert(
                "empty_namespaces_found".to_string(),
                json!(gc_stats.empty_namespaces_found),
            );
            object.insert(
                "old_docs_removed".to_string(),
                json!(gc_stats.old_docs_removed),
            );
        }
    }
    value
}

fn stat_delta(before: Option<usize>, after: Option<usize>) -> Option<i64> {
    before
        .zip(after)
        .map(|(before, after)| after as i64 - before as i64)
}

fn start_payload(action: &str, db_path: &str, pre_stats: Option<&crate::TableStats>) -> Value {
    json!({
        "action": action,
        "db_path": db_path,
        "pre_row_count": pre_stats.map(|stats| stats.row_count),
        "pre_version_count": pre_stats.map(|stats| stats.version_count),
    })
}

fn sse_event(event: &str, data: Value) -> SseEvent {
    SseEvent {
        event: event.to_string(),
        id: None,
        data,
    }
}

fn to_axum_event(event: SseEvent) -> Event {
    let mut axum_event = Event::default()
        .event(event.event)
        .data(event.data.to_string());
    if let Some(id) = event.id {
        axum_event = axum_event.id(id);
    }
    axum_event
}

fn sse_response<S>(stream: S) -> Sse<axum::response::sse::KeepAliveStream<S>>
where
    S: futures::Stream<Item = Result<Event, Infallible>> + Send + 'static,
{
    Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("ping"),
    )
}

fn internal_error(err: impl std::fmt::Display) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, err.to_string())
}
