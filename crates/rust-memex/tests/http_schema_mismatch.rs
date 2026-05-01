use std::{
    io::{self, Write},
    sync::{Arc, Mutex as StdMutex},
};

use arrow_schema::{DataType, Field, Schema};
use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Method, Request, StatusCode, header},
};
use lancedb::connect;
use rust_memex::{
    AuthManager, DEFAULT_TABLE_NAME, EmbeddingClient, McpCore, RAGPipeline, StorageManager,
    http::{HttpServerConfig, HttpState, create_router},
};
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::sync::Mutex;
use tower::util::ServiceExt;
use tracing_subscriber::fmt::MakeWriter;

const AUTH_TOKEN: &str = "secret-token";
const TEST_DIMENSION: usize = 8;

struct TestApp {
    app: Router,
    db_path: String,
    _tmp: TempDir,
}

#[derive(Clone)]
struct CapturedLogs(Arc<StdMutex<Vec<u8>>>);

struct CapturedLogWriter(Arc<StdMutex<Vec<u8>>>);

impl Write for CapturedLogWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0
            .lock()
            .expect("log buffer poisoned")
            .extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<'a> MakeWriter<'a> for CapturedLogs {
    type Writer = CapturedLogWriter;

    fn make_writer(&'a self) -> Self::Writer {
        CapturedLogWriter(self.0.clone())
    }
}

async fn build_test_app() -> TestApp {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db_path = tmp.path().join("lancedb");
    let embedding_client = Arc::new(Mutex::new(EmbeddingClient::stub_for_tests()));
    let storage = Arc::new(
        StorageManager::new(db_path.to_str().unwrap())
            .await
            .expect("storage"),
    );
    let rag = Arc::new(
        RAGPipeline::new(embedding_client.clone(), storage)
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
        _tmp: tmp,
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

fn authed_json_request(method: Method, uri: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, format!("Bearer {AUTH_TOKEN}"))
        .body(Body::from(body.to_string()))
        .expect("request")
}

#[tokio::test]
async fn upsert_schema_mismatch_returns_412_with_structured_body_and_error_log() {
    let test_app = build_test_app().await;
    create_pre_v4_table(&test_app.db_path).await;

    let log_buffer = Arc::new(StdMutex::new(Vec::new()));
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::ERROR)
        .with_writer(CapturedLogs(log_buffer.clone()))
        .with_ansi(false)
        .finish();
    let _guard = tracing::subscriber::set_default(subscriber);

    let response = test_app
        .app
        .clone()
        .oneshot(authed_json_request(
            Method::POST,
            "/upsert",
            json!({
                "namespace": "legacy-ns",
                "id": "doc-1",
                "content": "schema mismatch should fail loudly",
                "metadata": {"slice_mode": "flat"}
            }),
        ))
        .await
        .expect("upsert response");

    assert_eq!(response.status(), StatusCode::PRECONDITION_FAILED);
    let body = to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("body bytes");
    let json: Value = serde_json::from_slice(&body).expect("json body");
    assert_eq!(json["error"], "schema_mismatch");
    assert_eq!(json["error_kind"], "schema_mismatch");
    assert_eq!(json["missing_columns"], json!(["source_hash"]));
    assert!(
        json["remediation"]
            .as_str()
            .expect("remediation")
            .contains("rust-memex migrate-schema --db-path")
    );

    let logs =
        String::from_utf8(log_buffer.lock().expect("log buffer poisoned").clone()).expect("utf8");
    assert!(logs.contains("ERROR"), "{logs}");
    assert!(logs.contains("schema_mismatch"), "{logs}");
    assert!(logs.contains("source_hash"), "{logs}");
    assert!(logs.contains("rust-memex migrate-schema"), "{logs}");
}
