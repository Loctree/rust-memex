use axum::{
    Json, Router,
    body::{Body, to_bytes},
    extract::State,
    http::{Method, Request, StatusCode, header},
    routing::post,
};
use rust_memex::{
    AuthManager, ChromaDocument, EmbeddingClient, EmbeddingConfig, McpCore, ProviderConfig,
    RAGPipeline, StorageManager,
    contracts::progress::{ReindexProgress, ReprocessProgress},
    http::{HttpServerConfig, HttpState, create_router},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;
use tempfile::TempDir;
use tokio::{net::TcpListener, sync::Mutex, task::JoinHandle};
use tower::util::ServiceExt;

const AUTH_TOKEN: &str = "secret-token";
const TEST_DIMENSION: usize = 8;

#[derive(Clone)]
struct MockEmbeddingState {
    dimension: usize,
}

#[derive(Debug, Deserialize)]
struct MockEmbeddingRequest {
    input: Vec<String>,
}

#[derive(Debug, serde::Serialize)]
struct MockEmbeddingResponse {
    data: Vec<MockEmbeddingData>,
}

#[derive(Debug, serde::Serialize)]
struct MockEmbeddingData {
    embedding: Vec<f32>,
}

struct MockEmbeddingServer {
    base_url: String,
    handle: JoinHandle<()>,
}

impl Drop for MockEmbeddingServer {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

struct TestApp {
    app: Router,
    storage: Arc<StorageManager>,
    _tmp: TempDir,
    _mock_server: MockEmbeddingServer,
}

async fn mock_embeddings(
    State(state): State<MockEmbeddingState>,
    Json(request): Json<MockEmbeddingRequest>,
) -> Json<MockEmbeddingResponse> {
    let data = request
        .input
        .into_iter()
        .enumerate()
        .map(|(index, _)| MockEmbeddingData {
            embedding: vec![index as f32 + 0.25; state.dimension],
        })
        .collect();
    Json(MockEmbeddingResponse { data })
}

async fn start_mock_embedding_server() -> MockEmbeddingServer {
    let app = Router::new()
        .route("/v1/embeddings", post(mock_embeddings))
        .with_state(MockEmbeddingState {
            dimension: TEST_DIMENSION,
        });

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind mock");
    let address = listener.local_addr().expect("mock address");
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.expect("mock server");
    });

    MockEmbeddingServer {
        base_url: format!("http://{}", address),
        handle,
    }
}

fn test_embedding_config(base_url: &str) -> EmbeddingConfig {
    EmbeddingConfig {
        required_dimension: TEST_DIMENSION,
        max_batch_chars: 16_000,
        max_batch_items: 8,
        providers: vec![ProviderConfig {
            name: "mock".to_string(),
            base_url: base_url.to_string(),
            model: "mock-embed".to_string(),
            priority: 1,
            endpoint: "/v1/embeddings".to_string(),
        }],
        reranker: Default::default(),
    }
}

async fn build_test_app() -> TestApp {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db_path = tmp.path().join("lancedb");
    let mock_server = start_mock_embedding_server().await;
    let embedding_config = test_embedding_config(&mock_server.base_url);
    let embedding_client = Arc::new(Mutex::new(
        EmbeddingClient::new(&embedding_config)
            .await
            .expect("embedding client"),
    ));
    let storage = Arc::new(
        StorageManager::new(db_path.to_str().unwrap())
            .await
            .expect("storage"),
    );
    let rag = Arc::new(
        RAGPipeline::new(embedding_client.clone(), storage.clone())
            .await
            .expect("rag"),
    );

    let tokens_path = tmp.path().join("tokens.json");
    let auth_manager = Arc::new(AuthManager::new(
        tokens_path.to_string_lossy().to_string(),
        None,
    ));
    let mcp_core = Arc::new(McpCore::new(
        rag.clone(),
        None,
        embedding_client,
        1024 * 1024,
        vec![],
        auth_manager,
    ));
    let state = HttpState::new(rag, mcp_core);
    let config = HttpServerConfig {
        auth_token: Some(AUTH_TOKEN.to_string()),
        ..Default::default()
    };
    let app = create_router(state, &config);

    TestApp {
        app,
        storage,
        _tmp: tmp,
        _mock_server: mock_server,
    }
}

fn authed_json_request(method: Method, uri: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, format!("Bearer {AUTH_TOKEN}"))
        .body(Body::from(body.to_string()))
        .expect("request")
}

fn authed_multipart_request(uri: &str, boundary: &str, body: String) -> Request<Body> {
    Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header(
            header::CONTENT_TYPE,
            format!("multipart/form-data; boundary={boundary}"),
        )
        .header(header::AUTHORIZATION, format!("Bearer {AUTH_TOKEN}"))
        .body(Body::from(body))
        .expect("multipart request")
}

fn parse_sse_events(body: &str) -> Vec<(String, Value)> {
    body.split("\n\n")
        .filter_map(|chunk| {
            let mut event = None;
            let mut data = None;
            for line in chunk.lines() {
                if let Some(value) = line.strip_prefix("event:") {
                    event = Some(value.trim().to_string());
                } else if let Some(value) = line.strip_prefix("data:") {
                    data =
                        Some(serde_json::from_str::<Value>(value.trim()).expect("valid sse json"));
                }
            }

            match (event, data) {
                (Some(event), Some(data)) => Some((event, data)),
                _ => None,
            }
        })
        .collect()
}

fn multipart_body(
    namespace: &str,
    skip_existing: bool,
    file_contents: &str,
    boundary: &str,
) -> String {
    format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"namespace\"\r\n\r\n{namespace}\r\n\
--{boundary}\r\nContent-Disposition: form-data; name=\"skip_existing\"\r\n\r\n{skip_existing}\r\n\
--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"export.jsonl\"\r\nContent-Type: application/x-ndjson\r\n\r\n{file_contents}\r\n\
--{boundary}--\r\n"
    )
}

#[tokio::test]
async fn reprocess_endpoint_streams_progress_and_writes_target_namespace() {
    let test_app = build_test_app().await;
    let input_path = test_app._tmp.path().join("reprocess.jsonl");
    let payload = [
        json!({
            "id": "doc-1:outer",
            "text": "Short summary",
            "metadata": {"original_id": "doc-1", "layer": "outer"},
            "content_hash": "outer-hash"
        }),
        json!({
            "id": "doc-1:core",
            "text": "Longer full document content that should survive collapse.",
            "metadata": {"original_id": "doc-1", "layer": "core"},
            "content_hash": "core-hash"
        }),
    ]
    .into_iter()
    .map(|row| row.to_string())
    .collect::<Vec<_>>()
    .join("\n");
    tokio::fs::write(&input_path, payload)
        .await
        .expect("write jsonl");

    let response = test_app
        .app
        .clone()
        .oneshot(authed_json_request(
            Method::POST,
            "/sse/reprocess",
            json!({
                "input_path": input_path,
                "target_namespace": "reprocessed-ns",
                "slice_mode": "flat",
                "preprocess": false,
                "skip_existing": false
            }),
        ))
        .await
        .expect("sse response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("sse body");
    let text = String::from_utf8(body.to_vec()).expect("utf8");
    let events = parse_sse_events(&text);
    assert!(events.iter().any(|(event, _)| event == "start"));
    assert!(events.iter().any(|(event, _)| event == "progress"));
    let result = events
        .iter()
        .find(|(event, _)| event == "result")
        .map(|(_, data)| data.clone())
        .expect("result event");
    let progress = events
        .iter()
        .find(|(event, _)| event == "progress")
        .map(|(_, data)| {
            serde_json::from_value::<ReprocessProgress>(data.clone()).expect("progress")
        })
        .expect("typed progress");

    assert_eq!(progress.source_label, input_path.display().to_string());
    assert_eq!(result["indexed_documents"], 1);
    assert_eq!(
        test_app
            .storage
            .count_namespace("reprocessed-ns")
            .await
            .expect("count"),
        1
    );
}

#[tokio::test]
async fn reindex_endpoint_defaults_target_namespace() {
    let test_app = build_test_app().await;
    test_app
        .storage
        .add_to_store(vec![ChromaDocument::new_flat_with_hash(
            "src-doc".to_string(),
            "source-ns".to_string(),
            vec![0.4; TEST_DIMENSION],
            json!({"doc_id": "src-doc"}),
            "Reindex me".to_string(),
            "hash-src-doc".to_string(),
        )])
        .await
        .expect("seed source");

    let response = test_app
        .app
        .clone()
        .oneshot(authed_json_request(
            Method::POST,
            "/sse/reindex",
            json!({
                "source_namespace": "source-ns",
                "slice_mode": "flat",
                "preprocess": false,
                "skip_existing": false
            }),
        ))
        .await
        .expect("reindex response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("sse body");
    let text = String::from_utf8(body.to_vec()).expect("utf8");
    let events = parse_sse_events(&text);
    let progress = events
        .iter()
        .find(|(event, _)| event == "progress")
        .map(|(_, data)| serde_json::from_value::<ReindexProgress>(data.clone()).expect("progress"))
        .expect("typed progress");
    let result = events
        .iter()
        .find(|(event, _)| event == "result")
        .map(|(_, data)| data.clone())
        .expect("result event");

    assert_eq!(progress.namespace, "source-ns-reindexed");
    assert_eq!(result["target_namespace"], "source-ns-reindexed");
    assert_eq!(
        test_app
            .storage
            .count_namespace("source-ns-reindexed")
            .await
            .expect("count"),
        1
    );
}

#[tokio::test]
async fn export_import_round_trip_and_migrate_namespace_work() {
    let test_app = build_test_app().await;
    test_app
        .storage
        .add_to_store(vec![
            ChromaDocument::new_flat_with_hash(
                "doc-a".to_string(),
                "export-src".to_string(),
                vec![0.1; TEST_DIMENSION],
                json!({"kind": "note"}),
                "Alpha body".to_string(),
                "hash-alpha".to_string(),
            ),
            ChromaDocument::new_flat_with_hash(
                "doc-b".to_string(),
                "export-src".to_string(),
                vec![0.2; TEST_DIMENSION],
                json!({"kind": "note"}),
                "Beta body".to_string(),
                "hash-beta".to_string(),
            ),
        ])
        .await
        .expect("seed export source");

    let export_response = test_app
        .app
        .clone()
        .oneshot(authed_json_request(
            Method::POST,
            "/api/export",
            json!({
                "namespace": "export-src",
                "include_embeddings": true
            }),
        ))
        .await
        .expect("export response");

    assert_eq!(export_response.status(), StatusCode::OK);
    assert_eq!(
        export_response.headers().get(header::CONTENT_TYPE).unwrap(),
        "application/x-ndjson"
    );
    let exported_bytes = to_bytes(export_response.into_body(), 1024 * 1024)
        .await
        .expect("export body");
    let exported_jsonl = String::from_utf8(exported_bytes.to_vec()).expect("utf8");
    let exported_rows = exported_jsonl
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("json line"))
        .collect::<Vec<_>>();
    assert_eq!(exported_rows.len(), 2);
    let exported_hashes = exported_rows
        .iter()
        .map(|row| {
            row["content_hash"]
                .as_str()
                .expect("exported content hash")
                .to_string()
        })
        .collect::<Vec<_>>();
    assert_eq!(exported_hashes, vec!["hash-alpha", "hash-beta"]);

    let boundary = "memex-boundary";
    let import_body = multipart_body("import-copy", false, &exported_jsonl, boundary);
    let import_response = test_app
        .app
        .clone()
        .oneshot(authed_multipart_request(
            "/api/import",
            boundary,
            import_body,
        ))
        .await
        .expect("import response");

    assert_eq!(import_response.status(), StatusCode::OK);
    let import_json: Value = serde_json::from_slice(
        &to_bytes(import_response.into_body(), 64 * 1024)
            .await
            .expect("import body"),
    )
    .expect("import json");
    assert_eq!(import_json["imported_count"], 2);
    assert_eq!(
        import_json["imported_count"]
            .as_u64()
            .expect("imported count") as usize,
        exported_rows.len()
    );
    assert_eq!(
        test_app
            .storage
            .count_namespace("import-copy")
            .await
            .expect("import count"),
        2
    );

    test_app
        .storage
        .add_to_store(vec![
            ChromaDocument::new_flat_with_hash(
                "move-a".to_string(),
                "rename-me".to_string(),
                vec![0.3; TEST_DIMENSION],
                json!({"kind": "rename"}),
                "Move alpha".to_string(),
                "hash-move-a".to_string(),
            ),
            ChromaDocument::new_flat_with_hash(
                "move-b".to_string(),
                "rename-me".to_string(),
                vec![0.4; TEST_DIMENSION],
                json!({"kind": "rename"}),
                "Move beta".to_string(),
                "hash-move-b".to_string(),
            ),
        ])
        .await
        .expect("seed migrate");

    let migrate_response = test_app
        .app
        .clone()
        .oneshot(authed_json_request(
            Method::POST,
            "/api/migrate-namespace",
            json!({
                "from": "rename-me",
                "to": "renamed-ns"
            }),
        ))
        .await
        .expect("migrate response");

    assert_eq!(migrate_response.status(), StatusCode::OK);
    let migrate_json: Value = serde_json::from_slice(
        &to_bytes(migrate_response.into_body(), 64 * 1024)
            .await
            .expect("migrate body"),
    )
    .expect("migrate json");
    assert_eq!(migrate_json["migrated_chunks"], 2);
    assert_eq!(
        test_app
            .storage
            .count_namespace("rename-me")
            .await
            .expect("old count"),
        0
    );
    assert_eq!(
        test_app
            .storage
            .count_namespace("renamed-ns")
            .await
            .expect("new count"),
        2
    );
}
