use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema};
use axum::{
    Json, Router,
    body::{Body, to_bytes},
    extract::State,
    http::{Method, Request, StatusCode, header},
    routing::post,
};
use lancedb::connect;
use rust_memex::{
    AuthManager, DEFAULT_TABLE_NAME, EmbeddingClient, EmbeddingConfig, McpCore, ProviderConfig,
    RAGPipeline, SchemaVersion, StorageManager,
    http::{HttpServerConfig, HttpState, create_router},
};
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::{net::TcpListener, sync::Mutex, task::JoinHandle};
use tower::util::ServiceExt;

const AUTH_TOKEN: &str = "secret-token";
const TEST_DIMENSION: usize = 8;

struct TestApp {
    app: Router,
    db_path: String,
    storage: Arc<StorageManager>,
    _tmp: TempDir,
    _mock_server: MockEmbeddingServer,
}

#[derive(Clone)]
struct MockEmbeddingState {
    dimension: usize,
}

#[derive(Debug, serde::Deserialize)]
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
        db_path: db_path.to_string_lossy().to_string(),
        storage,
        _tmp: tmp,
        _mock_server: mock_server,
    }
}

async fn create_pre_v4_table(db_path: &str) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("namespace", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                TEST_DIMENSION as i32,
            ),
            false,
        ),
        Field::new("text", DataType::Utf8, true),
        Field::new("metadata", DataType::Utf8, true),
        Field::new("layer", DataType::UInt8, true),
        Field::new("parent_id", DataType::Utf8, true),
        Field::new("children_ids", DataType::Utf8, true),
        Field::new("keywords", DataType::Utf8, true),
        Field::new("content_hash", DataType::Utf8, true),
    ]));
    connect(db_path)
        .execute()
        .await
        .expect("connect lancedb")
        .create_empty_table(DEFAULT_TABLE_NAME, schema)
        .execute()
        .await
        .expect("create pre-v4 table");
}

fn request(method: Method, uri: &str, body: Option<Value>) -> Request<Body> {
    let mut builder = Request::builder().method(method).uri(uri);
    if body.is_some() {
        builder = builder
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::AUTHORIZATION, format!("Bearer {AUTH_TOKEN}"));
    }
    builder
        .body(Body::from(
            body.map(|value| value.to_string()).unwrap_or_default(),
        ))
        .expect("request")
}

async fn read_json(response: axum::response::Response) -> Value {
    let body = to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("body bytes");
    serde_json::from_slice(&body).expect("json body")
}

#[tokio::test]
async fn health_reports_pre_v4_table_as_needing_migration() {
    let test_app = build_test_app().await;
    create_pre_v4_table(&test_app.db_path).await;

    let response = test_app
        .app
        .oneshot(request(Method::GET, "/health", None))
        .await
        .expect("health response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = read_json(response).await;
    assert_eq!(body["status"], "needs_migration");
    assert_eq!(body["schema_version"], "v3-pre");
    assert_eq!(body["expected_schema"], "v4");
    assert_eq!(body["needs_migration"], true);
    assert_eq!(body["missing_columns"], json!(["source_hash"]));
    assert!(body["manifest_version"].as_u64().is_some(), "{body}");
    assert_eq!(body["last_successful_append_at"], Value::Null);
}

#[tokio::test]
async fn health_reports_ok_after_migration_and_successful_upsert() {
    let test_app = build_test_app().await;
    create_pre_v4_table(&test_app.db_path).await;
    StorageManager::migrate_lance_schema(&test_app.db_path, SchemaVersion::current(), false)
        .await
        .expect("migrate schema");
    test_app.storage.refresh().await.expect("refresh storage");

    let upsert = test_app
        .app
        .clone()
        .oneshot(request(
            Method::POST,
            "/upsert",
            Some(json!({
                "namespace": "kb:health",
                "id": "doc-1",
                "content": "health schema status verifies append freshness",
                "metadata": {"slice_mode": "flat"}
            })),
        ))
        .await
        .expect("upsert response");
    assert_eq!(upsert.status(), StatusCode::OK);

    let response = test_app
        .app
        .oneshot(request(Method::GET, "/health", None))
        .await
        .expect("health response");

    let body = read_json(response).await;
    assert_eq!(body["status"], "ok");
    assert_eq!(body["schema_version"], "v4");
    assert_eq!(body["needs_migration"], false);
    assert_eq!(body["missing_columns"], json!([]));
    assert!(
        body["last_successful_append_at"].as_str().is_some(),
        "{body}"
    );
    assert_eq!(body["namespaces"]["kb:health"]["chunks"], 1);
    assert!(
        body["namespaces"]["kb:health"]["last_indexed_at"]
            .as_str()
            .is_some(),
        "{body}"
    );
}

#[tokio::test]
async fn health_reports_empty_current_schema_before_first_append() {
    let test_app = build_test_app().await;

    let response = test_app
        .app
        .oneshot(request(Method::GET, "/health", None))
        .await
        .expect("health response");

    let body = read_json(response).await;
    assert_eq!(body["status"], "ok");
    assert_eq!(body["schema_version"], "v4");
    assert_eq!(body["expected_schema"], "v4");
    assert_eq!(body["needs_migration"], false);
    assert_eq!(body["missing_columns"], json!([]));
    assert_eq!(body["manifest_version"], Value::Null);
    assert_eq!(body["last_successful_append_at"], Value::Null);
    assert_eq!(body["namespaces"], json!({}));
}
