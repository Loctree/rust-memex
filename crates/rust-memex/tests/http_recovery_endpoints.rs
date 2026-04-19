use axum::{
    Json, Router,
    body::{Body, to_bytes},
    extract::State,
    http::{Method, Request, StatusCode, header},
    routing::post,
};
use rust_memex::{
    AuthManager, BM25Config, BM25Index, ChromaDocument, CrossStoreRecoveryBatch, EmbeddingClient,
    EmbeddingConfig, McpCore, ProviderConfig, RAGPipeline, StorageManager,
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
    tmp: TempDir,
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
        tmp,
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

async fn seed_doc(storage: &StorageManager, id: &str, namespace: &str, text: &str, hash: &str) {
    storage
        .add_to_store(vec![ChromaDocument::new_flat_with_hash(
            id.to_string(),
            namespace.to_string(),
            vec![0.1; TEST_DIMENSION],
            json!({"kind": "test"}),
            text.to_string(),
            hash.to_string(),
        )])
        .await
        .expect("seed doc");
}

#[tokio::test]
async fn merge_endpoint_dry_run_and_execute_work() {
    let test_app = build_test_app().await;
    let source_one_path = test_app.tmp.path().join("merge-src-1");
    let source_two_path = test_app.tmp.path().join("merge-src-2");
    let target_path = test_app.tmp.path().join("merge-target");

    let source_one = StorageManager::new_lance_only(source_one_path.to_str().unwrap())
        .await
        .expect("source one");
    let source_two = StorageManager::new_lance_only(source_two_path.to_str().unwrap())
        .await
        .expect("source two");

    seed_doc(
        &source_one,
        "doc-a",
        "alpha",
        "Alpha merge payload",
        "merge-hash-a",
    )
    .await;
    seed_doc(
        &source_two,
        "doc-b",
        "beta",
        "Beta merge payload",
        "merge-hash-b",
    )
    .await;

    let dry_run_response = test_app
        .app
        .clone()
        .oneshot(authed_json_request(
            Method::POST,
            "/api/merge",
            json!({
                "sources": [
                    source_one_path.display().to_string(),
                    source_two_path.display().to_string()
                ],
                "target": target_path.display().to_string(),
                "namespace_prefix": "merged:",
                "dedup": false,
                "dry_run": true
            }),
        ))
        .await
        .expect("dry-run merge response");

    assert_eq!(dry_run_response.status(), StatusCode::OK);
    let dry_run_json: Value = serde_json::from_slice(
        &to_bytes(dry_run_response.into_body(), 128 * 1024)
            .await
            .expect("dry-run merge body"),
    )
    .expect("dry-run merge json");
    assert_eq!(dry_run_json["dry_run"], true);
    assert_eq!(dry_run_json["progress"]["total_docs"], 2);
    assert_eq!(dry_run_json["progress"]["docs_copied"], 2);
    assert_eq!(
        dry_run_json["progress"]["namespaces"],
        json!(["merged:alpha", "merged:beta"])
    );

    let target_storage = StorageManager::new_lance_only(target_path.to_str().unwrap())
        .await
        .expect("target storage");
    assert_eq!(
        target_storage
            .count_namespace("merged:alpha")
            .await
            .expect("count"),
        0
    );

    let execute_response = test_app
        .app
        .clone()
        .oneshot(authed_json_request(
            Method::POST,
            "/api/merge",
            json!({
                "sources": [
                    source_one_path.display().to_string(),
                    source_two_path.display().to_string()
                ],
                "target": target_path.display().to_string(),
                "namespace_prefix": "merged:",
                "dedup": false,
                "dry_run": false
            }),
        ))
        .await
        .expect("execute merge response");

    assert_eq!(execute_response.status(), StatusCode::OK);
    let execute_json: Value = serde_json::from_slice(
        &to_bytes(execute_response.into_body(), 128 * 1024)
            .await
            .expect("execute merge body"),
    )
    .expect("execute merge json");
    assert_eq!(execute_json["progress"]["sources_processed"], 2);
    assert_eq!(
        target_storage
            .count_namespace("merged:alpha")
            .await
            .expect("count"),
        1
    );
    assert_eq!(
        target_storage
            .count_namespace("merged:beta")
            .await
            .expect("count"),
        1
    );
}

#[tokio::test]
async fn repair_writes_endpoint_inspects_and_executes_recovery() {
    let test_app = build_test_app().await;
    let repair_doc = ChromaDocument::new_flat_with_hash(
        "repair-doc".to_string(),
        "repair-http".to_string(),
        vec![0.2; TEST_DIMENSION],
        json!({"kind": "repair"}),
        "Recoverable repair payload".to_string(),
        "repair-hash".to_string(),
    );
    let recovery_batch = CrossStoreRecoveryBatch::from_documents(std::slice::from_ref(&repair_doc));

    test_app
        .storage
        .persist_cross_store_recovery_batch(&recovery_batch)
        .expect("persist recovery batch");
    test_app
        .storage
        .add_to_store(vec![repair_doc])
        .await
        .expect("seed repair doc");

    let inspect_response = test_app
        .app
        .clone()
        .oneshot(authed_json_request(
            Method::POST,
            "/api/repair-writes",
            json!({
                "namespace": "repair-http",
                "execute": false,
                "json_output": true
            }),
        ))
        .await
        .expect("inspect response");

    assert_eq!(inspect_response.status(), StatusCode::OK);
    let inspect_json: Value = serde_json::from_slice(
        &to_bytes(inspect_response.into_body(), 128 * 1024)
            .await
            .expect("inspect body"),
    )
    .expect("inspect json");
    assert_eq!(inspect_json["pending_batches"], 1);
    assert_eq!(inspect_json["divergent_batches"], 1);
    assert_eq!(inspect_json["documents_missing_bm25"], 1);

    let execute_response = test_app
        .app
        .clone()
        .oneshot(authed_json_request(
            Method::POST,
            "/api/repair-writes",
            json!({
                "namespace": "repair-http",
                "execute": true,
                "json_output": false
            }),
        ))
        .await
        .expect("execute response");

    assert_eq!(execute_response.status(), StatusCode::OK);
    let execute_json: Value = serde_json::from_slice(
        &to_bytes(execute_response.into_body(), 64 * 1024)
            .await
            .expect("execute body"),
    )
    .expect("execute json");
    assert_eq!(execute_json["pending_batches"], 1);
    assert_eq!(execute_json["repaired_documents"], 1);
    assert_eq!(execute_json["batches_repaired"], 1);
    assert!(
        test_app
            .storage
            .list_cross_store_recovery_batches()
            .expect("recovery batches")
            .is_empty()
    );

    let bm25_path = test_app.tmp.path().join(".bm25");
    let bm25 =
        BM25Index::new(&BM25Config::default().with_path(bm25_path.to_string_lossy().into_owned()))
            .expect("bm25 index");
    let bm25_results = bm25
        .search("recoverable", Some("repair-http"), 10)
        .expect("bm25 search");
    assert_eq!(bm25_results.len(), 1);
    assert_eq!(bm25_results[0].0, "repair-doc");
}

#[tokio::test]
async fn recovery_sse_endpoints_stream_independent_operations_and_alias() {
    let test_app = build_test_app().await;
    seed_doc(
        test_app.storage.as_ref(),
        "compact-doc",
        "compact-http",
        "Compaction target document",
        "compact-hash",
    )
    .await;

    let compact_response = test_app
        .app
        .clone()
        .oneshot(authed_json_request(Method::POST, "/sse/compact", json!({})))
        .await
        .expect("compact response");
    assert_eq!(compact_response.status(), StatusCode::OK);
    let compact_text = String::from_utf8(
        to_bytes(compact_response.into_body(), 256 * 1024)
            .await
            .expect("compact body")
            .to_vec(),
    )
    .expect("compact utf8");
    let compact_events = parse_sse_events(&compact_text);
    assert!(compact_events.iter().any(|(event, _)| event == "start"));
    assert!(
        compact_events
            .iter()
            .any(|(event, _)| event == "compact_done")
    );
    assert!(
        compact_events
            .iter()
            .find(|(event, _)| event == "done")
            .map(|(_, data)| data["ok"].as_bool().unwrap_or(false))
            .unwrap_or(false)
    );

    let cleanup_response = test_app
        .app
        .clone()
        .oneshot(authed_json_request(Method::POST, "/sse/cleanup", json!({})))
        .await
        .expect("cleanup response");
    assert_eq!(cleanup_response.status(), StatusCode::OK);
    let cleanup_text = String::from_utf8(
        to_bytes(cleanup_response.into_body(), 256 * 1024)
            .await
            .expect("cleanup body")
            .to_vec(),
    )
    .expect("cleanup utf8");
    let cleanup_events = parse_sse_events(&cleanup_text);
    assert!(
        cleanup_events
            .iter()
            .any(|(event, _)| event == "cleanup_done")
    );

    let mut orphan_doc = ChromaDocument::new_flat_with_hash(
        "orphan-doc".to_string(),
        "gc-http".to_string(),
        vec![0.3; TEST_DIMENSION],
        json!({"kind": "orphan"}),
        "Orphan document".to_string(),
        "gc-hash".to_string(),
    );
    orphan_doc.parent_id = Some("missing-parent".to_string());
    test_app
        .storage
        .add_to_store(vec![orphan_doc])
        .await
        .expect("seed orphan");

    let gc_response = test_app
        .app
        .clone()
        .oneshot(authed_json_request(Method::POST, "/sse/gc", json!({})))
        .await
        .expect("gc response");
    assert_eq!(gc_response.status(), StatusCode::OK);
    let gc_text = String::from_utf8(
        to_bytes(gc_response.into_body(), 256 * 1024)
            .await
            .expect("gc body")
            .to_vec(),
    )
    .expect("gc utf8");
    let gc_events = parse_sse_events(&gc_text);
    let gc_done = gc_events
        .iter()
        .find(|(event, _)| event == "gc_done")
        .map(|(_, data)| data.clone())
        .expect("gc done event");
    assert_eq!(gc_done["orphans_found"], 1);
    assert_eq!(gc_done["orphans_removed"], 1);
    assert_eq!(
        test_app
            .storage
            .count_namespace("gc-http")
            .await
            .expect("gc namespace count"),
        0
    );

    let optimize_response = test_app
        .app
        .clone()
        .oneshot(authed_json_request(
            Method::POST,
            "/sse/optimize",
            json!({}),
        ))
        .await
        .expect("optimize response");
    assert_eq!(optimize_response.status(), StatusCode::OK);
    let optimize_text = String::from_utf8(
        to_bytes(optimize_response.into_body(), 256 * 1024)
            .await
            .expect("optimize body")
            .to_vec(),
    )
    .expect("optimize utf8");
    let optimize_events = parse_sse_events(&optimize_text);
    assert!(
        optimize_events
            .iter()
            .any(|(event, _)| event == "compact_done")
    );
    assert!(
        optimize_events
            .iter()
            .any(|(event, _)| event == "prune_done")
    );
    let optimize_done = optimize_events
        .iter()
        .find(|(event, _)| event == "done")
        .map(|(_, data)| data.clone())
        .expect("optimize done");
    assert_eq!(optimize_done["compact_ok"], true);
    assert_eq!(optimize_done["prune_ok"], true);
}
