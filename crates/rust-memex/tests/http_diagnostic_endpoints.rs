use axum::{
    Json, Router,
    body::{Body, to_bytes},
    extract::State,
    http::{Method, Request, StatusCode, header},
    routing::post,
};
use rust_memex::{
    AuthManager, ChromaDocument, EmbeddingClient, EmbeddingConfig, McpCore, ProviderConfig,
    RAGPipeline, SliceLayer, StorageManager,
    contracts::{
        audit::AuditResult,
        stats::{DatabaseStats, NamespaceStats},
        timeline::TimelineEntry,
    },
    http::{HttpServerConfig, HttpState, create_router},
};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use std::sync::Arc;
use tempfile::TempDir;
use tokio::{net::TcpListener, sync::Mutex, task::JoinHandle};
use tower::util::ServiceExt;

const AUTH_TOKEN: &str = "secret-token";
const EMBEDDING_DIMENSION: usize = 2;

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

struct TestApp {
    app: axum::Router,
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
            dimension: EMBEDDING_DIMENSION,
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
        required_dimension: EMBEDDING_DIMENSION,
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
    let app = create_router(
        state,
        &HttpServerConfig {
            auth_token: Some(AUTH_TOKEN.to_string()),
            ..Default::default()
        },
    );

    seed_documents(storage.as_ref()).await;

    TestApp {
        app,
        storage,
        _tmp: tmp,
        _mock_server: mock_server,
    }
}

async fn seed_documents(storage: &StorageManager) {
    let docs = vec![
        doc_with_layer_and_hash(
            "klaud-outer",
            "klaudiusz-memories",
            SliceLayer::Outer,
            "Structured summary about patient follow-up and medication changes.",
            "hash-klaud-outer",
            None,
            json!({
                "indexed_at": "2026-04-18T09:15:00Z",
                "source": "klaudiusz-summary.md"
            }),
            &["patient", "summary"],
        ),
        doc_with_layer_and_hash(
            "klaud-core",
            "klaudiusz-memories",
            SliceLayer::Core,
            "Detailed clinical note with multiple sentences. The record includes observations, medication timing, and recovery guidance.",
            "hash-klaud-core",
            None,
            json!({
                "indexed_at": "2026-04-18T09:25:00Z",
                "source": "klaudiusz-core.md"
            }),
            &["clinical", "recovery"],
        ),
        doc_with_layer_and_hash(
            "aicx-1",
            "aicx",
            SliceLayer::Outer,
            "Timeline entry for audit day one.",
            "hash-aicx-1",
            None,
            json!({
                "indexed_at": "2026-04-17T10:15:00Z",
                "source": "day-one.md"
            }),
            &["timeline"],
        ),
        doc_with_layer_and_hash(
            "aicx-2",
            "aicx",
            SliceLayer::Outer,
            "Timeline entry for audit day two.",
            "hash-aicx-2",
            None,
            json!({
                "indexed_at": "2026-04-18T11:45:00Z",
                "source": "day-two.md"
            }),
            &["timeline"],
        ),
        // The two `dup-*` rows share both `content_hash` AND `source_hash` (+ layer)
        // so the post-v4 default (`source-hash-layer`) and the legacy
        // (`content-hash`) modes both classify them as one duplicate cluster.
        // Spec P4: "dedup CLI: nowy default `--group-by source-hash-layer` zachowuje
        // onion (1 chunk per layer per source), stary `--group-by content-hash`
        // jako opt-in dla edge cases".
        doc_with_layer_and_hash(
            "dup-keep",
            "dup-ns",
            SliceLayer::Outer,
            "This duplicate content should collapse.",
            "dup-hash",
            Some("dup-source"),
            json!({
                "indexed_at": "2026-04-19T08:00:00Z",
                "source": "dup-a.md"
            }),
            &["dup"],
        ),
        doc_with_layer_and_hash(
            "dup-remove",
            "dup-ns",
            SliceLayer::Outer,
            "This duplicate content should collapse.",
            "dup-hash",
            Some("dup-source"),
            json!({
                "indexed_at": "2026-04-19T08:05:00Z",
                "source": "dup-b.md"
            }),
            &["dup"],
        ),
        doc_with_layer_and_hash(
            "dup-unique",
            "dup-ns",
            SliceLayer::Outer,
            "This entry is unique.",
            "dup-unique-hash",
            Some("dup-unique-source"),
            json!({
                "indexed_at": "2026-04-19T08:10:00Z",
                "source": "dup-c.md"
            }),
            &["unique"],
        ),
        // Pre-v4 row: `content_hash` populated but `source_hash` is null. Under the
        // post-v4 default the row contributes to `docs_without_hash`; under the
        // legacy `content-hash` opt-in the same row is a regular candidate.
        doc_with_layer_and_hash(
            "dup-pre-v4",
            "dup-ns",
            SliceLayer::Outer,
            "Legacy chunk written before source_hash was wired up.",
            "pre-v4-hash",
            None,
            json!({
                "indexed_at": "2026-04-19T08:15:00Z",
                "source": "dup-pre-v4.md"
            }),
            &["legacy"],
        ),
        doc_with_layer_and_hash(
            "purge-1",
            "purge-me",
            SliceLayer::Outer,
            "fragment",
            "purge-hash-1",
            None,
            json!({
                "indexed_at": "2026-04-19T07:00:00Z",
                "source": "broken-a.txt"
            }),
            &["broken"],
        ),
        doc_with_layer_and_hash(
            "purge-2",
            "purge-me",
            SliceLayer::Outer,
            "noise",
            "purge-hash-2",
            None,
            json!({
                "indexed_at": "2026-04-19T07:05:00Z",
                "source": "broken-b.txt"
            }),
            &["broken"],
        ),
    ];

    storage.add_to_store(docs).await.expect("seed docs");
}

#[allow(clippy::too_many_arguments)]
fn doc_with_layer_and_hash(
    id: &str,
    namespace: &str,
    layer: SliceLayer,
    text: &str,
    content_hash: &str,
    source_hash: Option<&str>,
    metadata: Value,
    keywords: &[&str],
) -> ChromaDocument {
    let mut doc = ChromaDocument::new_flat_with_hashes(
        id.to_string(),
        namespace.to_string(),
        vec![0.25; EMBEDDING_DIMENSION],
        metadata,
        text.to_string(),
        content_hash.to_string(),
        source_hash.map(ToString::to_string),
    );
    doc.layer = layer.as_u8();
    doc.keywords = keywords
        .iter()
        .map(|keyword| (*keyword).to_string())
        .collect();
    doc
}

fn authed_request(method: Method, uri: &str, body: Option<Value>) -> Request<Body> {
    let builder = Request::builder()
        .method(method)
        .uri(uri)
        .header(header::AUTHORIZATION, format!("Bearer {AUTH_TOKEN}"));

    if let Some(json_body) = body {
        builder
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(json_body.to_string()))
            .expect("authed json request")
    } else {
        builder.body(Body::empty()).expect("authed empty request")
    }
}

async fn response_json<T: DeserializeOwned>(response: axum::response::Response) -> T {
    let bytes = to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("body bytes");
    serde_json::from_slice(&bytes).expect("json body")
}

#[tokio::test]
async fn audit_endpoint_returns_contract_json() {
    let test_app = build_test_app().await;

    let response = test_app
        .app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/audit?ns=klaudiusz-memories&threshold=80")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("audit response");

    assert_eq!(response.status(), StatusCode::OK);
    let audit: AuditResult = response_json(response).await;
    assert_eq!(audit.namespace, "klaudiusz-memories");
    assert!(audit.document_count >= 2);
}

#[tokio::test]
async fn database_stats_endpoint_returns_rows() {
    let test_app = build_test_app().await;

    let response = test_app
        .app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("stats response");

    assert_eq!(response.status(), StatusCode::OK);
    let stats: DatabaseStats = response_json(response).await;
    assert!(stats.row_count >= 9);
    assert_eq!(stats.table_name, "mcp_documents");
}

#[tokio::test]
async fn namespace_stats_endpoint_returns_layer_distribution() {
    let test_app = build_test_app().await;

    let response = test_app
        .app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/stats/klaudiusz-memories")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("namespace stats response");

    assert_eq!(response.status(), StatusCode::OK);
    let stats: NamespaceStats = response_json(response).await;
    assert_eq!(stats.name, "klaudiusz-memories");
    assert!(stats.total_chunks >= 2);
    assert_eq!(stats.layer_counts.get("outer"), Some(&1));
    assert_eq!(stats.layer_counts.get("core"), Some(&1));
}

#[tokio::test]
async fn timeline_endpoint_returns_bucketed_entries() {
    let test_app = build_test_app().await;

    let response = test_app
        .app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/timeline?ns=aicx&bucket=day")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("timeline response");

    assert_eq!(response.status(), StatusCode::OK);
    let entries: Vec<TimelineEntry> = response_json(response).await;
    assert!(!entries.is_empty());
    assert!(entries.iter().all(|entry| entry.namespace == "aicx"));
    assert!(entries.iter().any(|entry| entry.date == "2026-04-17"));
    assert!(entries.iter().any(|entry| entry.date == "2026-04-18"));
}

#[tokio::test]
async fn purge_quality_endpoint_requires_dry_run_then_executes() {
    let test_app = build_test_app().await;

    let blocked_response = test_app
        .app
        .clone()
        .oneshot(authed_request(
            Method::POST,
            "/api/purge-quality",
            Some(json!({
                "namespace": "purge-me",
                "threshold": 90,
                "confirm": true,
                "dry_run": false
            })),
        ))
        .await
        .expect("blocked purge response");

    assert_eq!(blocked_response.status(), StatusCode::CONFLICT);

    let dry_run_response = test_app
        .app
        .clone()
        .oneshot(authed_request(
            Method::POST,
            "/api/purge-quality",
            Some(json!({
                "namespace": "purge-me",
                "threshold": 90,
                "confirm": false,
                "dry_run": true
            })),
        ))
        .await
        .expect("purge dry run");

    assert_eq!(dry_run_response.status(), StatusCode::OK);
    let dry_run_json: Value = response_json(dry_run_response).await;
    assert!(dry_run_json["dry_run"].as_bool().unwrap());
    assert_eq!(
        test_app
            .storage
            .count_namespace("purge-me")
            .await
            .expect("pre purge count"),
        2
    );

    let execute_response = test_app
        .app
        .clone()
        .oneshot(authed_request(
            Method::POST,
            "/api/purge-quality",
            Some(json!({
                "namespace": "purge-me",
                "threshold": 90,
                "confirm": true,
                "dry_run": false
            })),
        ))
        .await
        .expect("purge execute");

    assert_eq!(execute_response.status(), StatusCode::OK);
    let execute_json: Value = response_json(execute_response).await;
    assert_eq!(execute_json["purged_namespaces"], 1);
    assert_eq!(
        test_app
            .storage
            .count_namespace("purge-me")
            .await
            .expect("post purge count"),
        0
    );
}

/// Default endpoint behaviour after spec P4 ("dedup grouping") locked
/// `source-hash-layer` as the post-v4 default. The two `dup-*` rows share
/// content_hash AND source_hash AND layer, so the cluster collapses to one
/// group keyed by `<source_hash>|layer<N>`. The pre-v4 row (no source_hash)
/// is reported via `docs_without_hash`, never silently swept under default
/// grouping. Execute deletes exactly one chunk and the surviving namespace
/// keeps every distinct row (kept duplicate + unique + pre-v4).
#[tokio::test]
async fn dedup_endpoint_lists_duplicates_then_executes() {
    let test_app = build_test_app().await;

    let dry_run_response = test_app
        .app
        .clone()
        .oneshot(authed_request(Method::POST, "/api/dedup?ns=dup-ns", None))
        .await
        .expect("dedup dry run");

    assert_eq!(dry_run_response.status(), StatusCode::OK);
    let dry_run_json: Value = response_json(dry_run_response).await;
    assert!(dry_run_json["dry_run"].as_bool().unwrap());
    assert_eq!(
        dry_run_json["result"]["group_by"], "source-hash-layer",
        "post-v4 default must surface back to the operator on the wire"
    );
    assert_eq!(dry_run_json["result"]["duplicate_groups"], 1);
    assert_eq!(
        dry_run_json["result"]["docs_without_hash"], 1,
        "pre-v4 row with empty source_hash must be visible, not silently grouped"
    );

    let group = &dry_run_json["result"]["groups"][0];
    let expected_key = format!("dup-source|layer{}", SliceLayer::Outer.as_u8());
    assert_eq!(group["group_key"], expected_key);
    assert_eq!(
        group["content_hash"], expected_key,
        "legacy `content_hash` field must mirror `group_key` for wire-compat"
    );
    assert_eq!(group["kept_id"], "dup-keep");
    assert_eq!(group["kept_namespace"], "dup-ns");
    assert_eq!(group["removed"][0]["id"], "dup-remove");

    let execute_response = test_app
        .app
        .clone()
        .oneshot(authed_request(
            Method::POST,
            "/api/dedup?ns=dup-ns&execute=true",
            Some(json!({ "confirm": true })),
        ))
        .await
        .expect("dedup execute");

    assert_eq!(execute_response.status(), StatusCode::OK);
    let execute_json: Value = response_json(execute_response).await;
    assert!(execute_json["execute"].as_bool().unwrap());
    assert_eq!(execute_json["result"]["group_by"], "source-hash-layer");
    assert_eq!(execute_json["result"]["duplicates_removed"], 1);
    // 4 seeded - 1 removed = 3 (dup-keep + dup-unique + dup-pre-v4).
    assert_eq!(
        test_app
            .storage
            .count_namespace("dup-ns")
            .await
            .expect("post dedup count"),
        3
    );
}

/// Spec P4 escape hatch: operators can opt into the legacy `content-hash`
/// grouping for edge cases (e.g. backfilled corpora where source_hash was
/// never populated). Locks the wire shape: `?group-by=content-hash` flips
/// the default, `result.group_by` echoes the choice back, and a pre-v4 row
/// that shares no `content_hash` with anyone else is reported as a unique
/// doc rather than as `docs_without_hash`.
#[tokio::test]
async fn dedup_endpoint_supports_legacy_content_hash_grouping() {
    let test_app = build_test_app().await;

    let dry_run_response = test_app
        .app
        .clone()
        .oneshot(authed_request(
            Method::POST,
            "/api/dedup?ns=dup-ns&group-by=content-hash",
            None,
        ))
        .await
        .expect("dedup dry run");

    assert_eq!(dry_run_response.status(), StatusCode::OK);
    let dry_run_json: Value = response_json(dry_run_response).await;
    assert!(dry_run_json["dry_run"].as_bool().unwrap());
    assert_eq!(
        dry_run_json["result"]["group_by"], "content-hash",
        "legacy opt-in must echo back so operators can audit which mode ran"
    );
    assert_eq!(
        dry_run_json["result"]["docs_without_hash"], 0,
        "legacy mode treats every row with a content_hash as eligible"
    );
    assert_eq!(dry_run_json["result"]["duplicate_groups"], 1);
    let group = &dry_run_json["result"]["groups"][0];
    assert_eq!(group["group_key"], "dup-hash");
    assert_eq!(
        group["content_hash"], "dup-hash",
        "legacy mode keys clusters by per-chunk content_hash directly"
    );
    assert_eq!(group["kept_id"], "dup-keep");
    assert_eq!(group["removed"][0]["id"], "dup-remove");
}
