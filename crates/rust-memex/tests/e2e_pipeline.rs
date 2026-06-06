//! End-to-End Pipeline Tests
//!
//! Full lifecycle verification: load operator's canonical config.toml →
//! build embedder via real provider cascade → index → embed → store →
//! search → delete → confirm-removed → re-add → confirm-restored.
//!
//! These tests are gated behind the `e2e-ollama` feature. They consume the
//! same `~/.rmcp-servers/rust-memex/config.toml` (or `RUST_MEMEX_CONFIG`)
//! that runtime uses — no synthetic defaults, no inline hardcoded endpoints.
//! If no config exists or no provider in the cascade is reachable, tests
//! fail-fast (no silent SKIP).
//!
//! Run with: `make test-e2e`
//!         or `cargo test --features e2e-ollama --test e2e_pipeline`
//!
//! Vibecrafted with AI Agents by VetCoders (c)2024-2026 LibraxisAI

mod common;

use rust_memex::{ChromaDocument, StorageManager};
use serde_json::json;
use tempfile::TempDir;

/// Synthetic dimension for storage-validation tests that don't touch a real
/// provider. Storage rules don't care about the value, only consistency.
const SYNTHETIC_TEST_DIM: usize = 2560;

// =============================================================================
// E2E TEST: Full pipeline lifecycle (index → search → delete → restore)
// =============================================================================

/// Full lifecycle: index 5 docs, semantic search, delete one, confirm absent,
/// re-add, confirm restored. Exercises the config-driven provider cascade
/// for embedding + LanceDB storage end-to-end.
#[cfg(feature = "e2e-ollama")]
#[tokio::test]
async fn test_e2e_full_lifecycle_index_search_delete_restore() {
    use rust_memex::EmbeddingClient;

    let cfg = common::load_e2e_config().expect("e2e config required");
    eprintln!(
        "e2e config: {} (providers: {})",
        cfg.source_path.display(),
        cfg.embeddings.providers.len()
    );

    let mut embedder = EmbeddingClient::new(&cfg.embeddings)
        .await
        .expect("EmbeddingClient must connect to a configured provider");
    let dim = cfg.embeddings.required_dimension;
    let connected = embedder.connected_to().to_string();
    eprintln!("connected provider: {connected}");

    let tmp = TempDir::new().expect("tempdir");
    let db_path = tmp.path().join("lancedb");
    let storage = StorageManager::new_lance_only(db_path.to_str().unwrap())
        .await
        .expect("storage init");
    storage
        .ensure_collection()
        .await
        .expect("ensure_collection");

    let ns = "e2e-lifecycle";

    // Test corpus
    let corpus = [
        (
            "doc-rust",
            "Rust is a systems programming language focused on safety and performance.",
        ),
        (
            "doc-python",
            "Python is a high-level interpreted language known for its readability.",
        ),
        (
            "doc-javascript",
            "JavaScript is the language of the web, running in browsers and Node.js.",
        ),
        (
            "doc-veterinary",
            "Veterinary medicine involves the diagnosis and treatment of animals.",
        ),
        (
            "doc-ai",
            "Artificial intelligence enables machines to learn from experience.",
        ),
    ];

    // ---- INDEX --------------------------------------------------------------
    let mut docs = Vec::with_capacity(corpus.len());
    for (id, text) in corpus.iter() {
        let embedding = embedder.embed(text).await.expect("embed");
        assert_eq!(embedding.len(), dim, "embedding dim must match config");
        docs.push(ChromaDocument::new_flat(
            (*id).to_string(),
            ns.to_string(),
            embedding,
            json!({"source": "e2e"}),
            (*text).to_string(),
        ));
    }
    storage.add_to_store(docs).await.expect("add_to_store");

    // ---- SEARCH (initial) ---------------------------------------------------
    let query = "systems programming language with memory safety";
    let q_embedding = embedder.embed(query).await.expect("embed query");
    let results = storage
        .search_store(Some(ns), q_embedding.clone(), corpus.len())
        .await
        .expect("search_store initial");
    assert!(!results.is_empty(), "search must return results");
    let initial_ids: Vec<String> = results.iter().map(|d| d.id.clone()).collect();
    eprintln!("initial search top-{}: {:?}", results.len(), initial_ids);
    assert!(
        initial_ids.iter().any(|id| id == "doc-rust"),
        "doc-rust must be reachable before delete; got {:?}",
        initial_ids
    );

    // ---- DELETE -------------------------------------------------------------
    let removed = storage
        .delete_documents(ns, &["doc-rust"])
        .await
        .expect("delete_documents");
    assert_eq!(removed, 1, "delete must remove exactly 1 document");

    // ---- SEARCH (after delete) ---------------------------------------------
    let results_after_delete = storage
        .search_store(Some(ns), q_embedding.clone(), corpus.len())
        .await
        .expect("search_store after delete");
    let after_delete_ids: Vec<String> = results_after_delete.iter().map(|d| d.id.clone()).collect();
    eprintln!(
        "after-delete search top-{}: {:?}",
        results_after_delete.len(),
        after_delete_ids
    );
    assert!(
        !after_delete_ids.iter().any(|id| id == "doc-rust"),
        "doc-rust must be absent after delete; got {:?}",
        after_delete_ids
    );

    // ---- RESTORE ------------------------------------------------------------
    let rust_text = corpus
        .iter()
        .find(|(id, _)| *id == "doc-rust")
        .map(|(_, t)| *t)
        .expect("doc-rust in corpus");
    let restore_embedding = embedder.embed(rust_text).await.expect("re-embed");
    storage
        .add_to_store(vec![ChromaDocument::new_flat(
            "doc-rust".to_string(),
            ns.to_string(),
            restore_embedding,
            json!({"source": "e2e", "restored": true}),
            rust_text.to_string(),
        )])
        .await
        .expect("re-add doc-rust");

    let results_after_restore = storage
        .search_store(Some(ns), q_embedding, corpus.len())
        .await
        .expect("search_store after restore");
    let after_restore_ids: Vec<String> =
        results_after_restore.iter().map(|d| d.id.clone()).collect();
    eprintln!(
        "after-restore search top-{}: {:?}",
        results_after_restore.len(),
        after_restore_ids
    );
    assert!(
        after_restore_ids.iter().any(|id| id == "doc-rust"),
        "doc-rust must reappear after restore; got {:?}",
        after_restore_ids
    );
}

/// Batch embedding sanity: embed multiple texts in one call, verify dim
/// consistency and absence of NaN/Inf in every row.
#[cfg(feature = "e2e-ollama")]
#[tokio::test]
async fn test_e2e_batch_embedding() {
    use rust_memex::EmbeddingClient;

    let cfg = common::load_e2e_config().expect("e2e config required");
    let mut embedder = EmbeddingClient::new(&cfg.embeddings)
        .await
        .expect("EmbeddingClient must connect");
    let dim = cfg.embeddings.required_dimension;

    let texts: Vec<String> = vec![
        "First document about machine learning".into(),
        "Second document about natural language processing".into(),
        "Third document about computer vision".into(),
        "Fourth document about reinforcement learning".into(),
    ];

    let embeddings = embedder.embed_batch(&texts).await.expect("embed_batch");
    assert_eq!(
        embeddings.len(),
        texts.len(),
        "must return one embedding per input"
    );
    for (i, emb) in embeddings.iter().enumerate() {
        assert_eq!(emb.len(), dim, "embedding {i} dim mismatch");
        for (j, &val) in emb.iter().enumerate() {
            assert!(
                !val.is_nan() && !val.is_infinite(),
                "embedding {i}[{j}] is NaN/Inf: {val}"
            );
        }
    }
}

/// Dimension-mismatch fail-fast: bumping required_dimension above what the
/// provider returns must error at client construction (before any DB write).
/// Exercises the EmbeddingClient invariant: dim mismatch corrupts the DB if
/// it slips through, so it must abort early.
#[cfg(feature = "e2e-ollama")]
#[tokio::test]
async fn test_e2e_dim_mismatch_fails_fast() {
    use rust_memex::EmbeddingClient;

    let cfg = common::load_e2e_config().expect("e2e config required");
    // Sanity: real config must work first
    let _ok = EmbeddingClient::new(&cfg.embeddings)
        .await
        .expect("baseline config must succeed");

    // Bump dim by 1 — no real model returns this; client must reject.
    let mut bad = cfg.embeddings.clone();
    bad.required_dimension += 1;
    let err = EmbeddingClient::new(&bad)
        .await
        .err()
        .expect("dim mismatch must error at construction");
    let msg = err.to_string();
    assert!(
        msg.contains("required_dimension") || msg.contains("dim"),
        "error must mention dimension; got: {msg}"
    );
}

/// Content-hash deduplication: store doc with hash, observe presence; ensure
/// hash-check API stays consistent across roundtrip.
#[cfg(feature = "e2e-ollama")]
#[tokio::test]
async fn test_e2e_deduplication() {
    use rust_memex::{EmbeddingClient, compute_content_hash};

    let cfg = common::load_e2e_config().expect("e2e config required");
    let mut embedder = EmbeddingClient::new(&cfg.embeddings)
        .await
        .expect("EmbeddingClient must connect");

    let tmp = TempDir::new().expect("tempdir");
    let db_path = tmp.path().join("lancedb");
    let storage = StorageManager::new_lance_only(db_path.to_str().unwrap())
        .await
        .expect("storage init");
    storage
        .ensure_collection()
        .await
        .expect("ensure_collection");

    let ns = "e2e-dedup";
    let content = "Unique content for deduplication test";
    let hash = compute_content_hash(content);

    assert!(
        !storage
            .has_content_hash(ns, &hash)
            .await
            .expect("has_content_hash"),
        "hash must not exist before indexing"
    );

    let embedding = embedder.embed(content).await.expect("embed");
    let mut doc = ChromaDocument::new_flat(
        "dedup-doc-1".to_string(),
        ns.to_string(),
        embedding,
        json!({}),
        content.to_string(),
    );
    doc.content_hash = Some(hash.clone());
    storage.add_to_store(vec![doc]).await.expect("add_to_store");

    assert!(
        storage
            .has_content_hash(ns, &hash)
            .await
            .expect("has_content_hash post"),
        "hash must exist after indexing"
    );
}

// =============================================================================
// STORAGE INVARIANT (no embedding provider needed)
// =============================================================================

/// Storage-level validation: empty IDs, empty namespaces, NaN/Inf in
/// embeddings, and inconsistent batch dimensions must all be rejected before
/// any LanceDB write. This invariant doesn't need a real provider — it
/// exercises pure storage rules with synthetic dim.
#[tokio::test]
async fn test_storage_rejects_invalid_embeddings() {
    let tmp = TempDir::new().expect("tempdir");
    let db_path = tmp.path().join("lancedb");
    let storage = StorageManager::new_lance_only(db_path.to_str().unwrap())
        .await
        .expect("storage init");
    storage.ensure_collection().await.unwrap();

    // Empty ID rejected
    let doc_empty_id = ChromaDocument::new_flat(
        String::new(),
        "test-ns".to_string(),
        vec![0.1f32; SYNTHETIC_TEST_DIM],
        json!({}),
        "Content".to_string(),
    );
    assert!(
        storage.add_to_store(vec![doc_empty_id]).await.is_err(),
        "Empty ID must be rejected"
    );

    // Empty namespace rejected
    let doc_empty_ns = ChromaDocument::new_flat(
        "valid-id".to_string(),
        String::new(),
        vec![0.1f32; SYNTHETIC_TEST_DIM],
        json!({}),
        "Content".to_string(),
    );
    assert!(
        storage.add_to_store(vec![doc_empty_ns]).await.is_err(),
        "Empty namespace must be rejected"
    );

    // NaN in embedding rejected
    let mut nan_emb = vec![0.1f32; SYNTHETIC_TEST_DIM];
    nan_emb[100] = f32::NAN;
    let doc_nan = ChromaDocument::new_flat(
        "nan-doc".to_string(),
        "test-ns".to_string(),
        nan_emb,
        json!({}),
        "Content".to_string(),
    );
    assert!(
        storage.add_to_store(vec![doc_nan]).await.is_err(),
        "NaN in embedding must be rejected"
    );

    // Inf in embedding rejected
    let mut inf_emb = vec![0.1f32; SYNTHETIC_TEST_DIM];
    inf_emb[200] = f32::INFINITY;
    let doc_inf = ChromaDocument::new_flat(
        "inf-doc".to_string(),
        "test-ns".to_string(),
        inf_emb,
        json!({}),
        "Content".to_string(),
    );
    assert!(
        storage.add_to_store(vec![doc_inf]).await.is_err(),
        "Inf in embedding must be rejected"
    );

    // Inconsistent dims in same batch rejected
    let doc_good = ChromaDocument::new_flat(
        "doc-good".to_string(),
        "test-ns".to_string(),
        vec![0.1f32; SYNTHETIC_TEST_DIM],
        json!({}),
        "Content".to_string(),
    );
    let doc_short = ChromaDocument::new_flat(
        "doc-short".to_string(),
        "test-ns".to_string(),
        vec![0.1f32; 1024], // wrong dim within batch
        json!({}),
        "Content".to_string(),
    );
    assert!(
        storage
            .add_to_store(vec![doc_good, doc_short])
            .await
            .is_err(),
        "Inconsistent batch dims must be rejected"
    );
}
