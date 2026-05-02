use axum::{Json, Router, extract::State, http::StatusCode, routing::post};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    fs,
    io::ErrorKind,
    path::{Path, PathBuf},
    process::{Command, Output},
};
use tempfile::TempDir;
use tokio::{net::TcpListener, task::JoinHandle};

const REQUIRED_DIMENSION: usize = 16;

#[derive(Clone)]
struct FailingEmbeddingState {
    dimension: usize,
}

#[derive(Debug, Deserialize)]
struct EmbeddingRequest {
    input: Vec<String>,
}

#[derive(Debug, Serialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Serialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

struct FailingEmbeddingServer {
    base_url: String,
    handle: JoinHandle<()>,
}

impl Drop for FailingEmbeddingServer {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn index_failure_policy_covers_strict_json_threshold_and_warning_modes() {
    let Some(server) = start_failing_embedding_server().await else {
        eprintln!("skipping CLI strict policy test: sandbox denied loopback bind");
        return;
    };

    let strict = run_index(&server.base_url, &["--strict"]);
    assert!(
        !strict.status.success(),
        "--strict must exit non-zero when every file fails\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&strict.stdout),
        String::from_utf8_lossy(&strict.stderr)
    );
    assert!(
        String::from_utf8_lossy(&strict.stderr).contains("1/1 files failed"),
        "strict stderr should include aggregate failed count, got:\n{}",
        String::from_utf8_lossy(&strict.stderr)
    );

    let json_output = run_index(&server.base_url, &["--json"]);
    assert!(
        json_output.status.success(),
        "--json without strict keeps backwards-compatible exit zero\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&json_output.stdout),
        String::from_utf8_lossy(&json_output.stderr)
    );
    let last_line = String::from_utf8_lossy(&json_output.stdout)
        .lines()
        .last()
        .expect("json summary line")
        .to_string();
    let summary: Value = serde_json::from_str(&last_line).expect("summary must be valid JSON");
    assert_eq!(summary["indexed"], 0);
    assert_eq!(summary["failed"], 1);
    assert_eq!(summary["total"], 1);
    assert_eq!(summary["failure_rate"], 1.0);
    assert!(
        summary["errors"]
            .as_array()
            .is_some_and(|errors| !errors.is_empty()),
        "summary should include the per-file embedding error"
    );

    let threshold = run_index(&server.base_url, &["--max-failure-rate", "0.05"]);
    assert!(
        !threshold.status.success(),
        "--max-failure-rate 0.05 must fail when 100% of files fail\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&threshold.stdout),
        String::from_utf8_lossy(&threshold.stderr)
    );
    assert!(
        String::from_utf8_lossy(&threshold.stderr).contains("1/1 files failed"),
        "threshold stderr should include aggregate failed count, got:\n{}",
        String::from_utf8_lossy(&threshold.stderr)
    );

    let warning = run_index(&server.base_url, &[]);
    assert!(
        warning.status.success(),
        "default mode should preserve exit zero despite per-file failures\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&warning.stdout),
        String::from_utf8_lossy(&warning.stderr)
    );
    assert!(
        String::from_utf8_lossy(&warning.stderr)
            .contains("WARNING: 1/1 files failed to index. See log above."),
        "default stderr should warn loudly, got:\n{}",
        String::from_utf8_lossy(&warning.stderr)
    );
}

async fn start_failing_embedding_server() -> Option<FailingEmbeddingServer> {
    let app = Router::new()
        .route("/v1/embeddings", post(mock_embeddings))
        .with_state(FailingEmbeddingState {
            dimension: REQUIRED_DIMENSION,
        });

    let listener = match TcpListener::bind("127.0.0.1:0").await {
        Ok(listener) => listener,
        Err(error) if error.kind() == ErrorKind::PermissionDenied => return None,
        Err(error) => panic!("bind mock embedding server: {error}"),
    };
    let address = listener.local_addr().expect("read mock address");
    let handle = tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("mock embedding server failed");
    });

    Some(FailingEmbeddingServer {
        base_url: format!("http://{}", address),
        handle,
    })
}

async fn mock_embeddings(
    State(state): State<FailingEmbeddingState>,
    Json(request): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, (StatusCode, String)> {
    if request.input.len() == 1 && request.input[0] == "dimension probe" {
        return Ok(Json(EmbeddingResponse {
            data: vec![EmbeddingData {
                embedding: vec![0.0; state.dimension],
            }],
        }));
    }

    Err((
        StatusCode::INTERNAL_SERVER_ERROR,
        "intentional per-file embedding failure".to_string(),
    ))
}

fn run_index(base_url: &str, extra_args: &[&str]) -> Output {
    let tmp = TempDir::new().expect("tempdir");
    let corpus = tmp.path().join("corpus");
    fs::create_dir_all(&corpus).expect("create corpus");
    fs::write(
        corpus.join("doc.md"),
        "# Broken embedding test document\n\nThis document exists solely to trigger an embedding failure.\nThe mock embedding server will return HTTP 500 for this content.\nWe need enough text to pass the 50-char minimum document threshold.\n",
    )
    .expect("write sample file");
    let db_path = tmp.path().join("lancedb");
    let config_path = write_config(tmp.path(), base_url).expect("write config");

    let mut args = vec![
        "--config".to_string(),
        config_path.to_string_lossy().to_string(),
        "--db-path".to_string(),
        db_path.to_string_lossy().to_string(),
        "--allowed-paths".to_string(),
        tmp.path().to_string_lossy().to_string(),
        "index".to_string(),
        corpus.to_string_lossy().to_string(),
        "--namespace".to_string(),
        "cli-index-strict".to_string(),
        "--recursive".to_string(),
        "--slice-mode".to_string(),
        "flat".to_string(),
        "--chunker".to_string(),
        "flat".to_string(),
        "--parallel".to_string(),
        "1".to_string(),
    ];
    args.extend(extra_args.iter().map(|arg| (*arg).to_string()));

    Command::new(env!("CARGO_BIN_EXE_rust-memex"))
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args(args)
        .output()
        .expect("run rust-memex index")
}

fn write_config(base: &Path, base_url: &str) -> std::io::Result<PathBuf> {
    let config_path = base.join("mock-embeddings.toml");
    let config = format!(
        "[embeddings]\nrequired_dimension = {REQUIRED_DIMENSION}\nmax_batch_chars = 32000\nmax_batch_items = 16\n\n[[embeddings.providers]]\nname = \"failing-test\"\nbase_url = \"{base_url}\"\nmodel = \"mock-embedder\"\npriority = 1\nendpoint = \"/v1/embeddings\"\n"
    );
    fs::write(&config_path, config)?;
    Ok(config_path)
}
