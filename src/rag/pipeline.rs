//! Async pipeline for concurrent RAG indexing.
//!
//! This module provides an optional pipeline mode where file reading, chunking,
//! embedding, and storage run concurrently using tokio channels.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌─────────────┐
//! │ File Reader │ ──► │   Chunker    │ ──► │   Embedder    │ ──► │   Storage   │
//! └─────────────┘     └──────────────┘     └───────────────┘     └─────────────┘
//!       tx1                 rx1/tx2              rx2/tx3              rx3
//! ```
//!
//! Each stage runs in its own tokio::spawn, connected via bounded mpsc channels
//! for backpressure. This allows overlapping I/O, CPU, and GPU work.
//!
//! Pipeline mode is opt-in via the `--pipeline` CLI flag or by calling
//! `run_pipeline()` directly.

use anyhow::Result;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, mpsc};
use tracing::{debug, error, info, warn};

use crate::embeddings::EmbeddingClient;
use crate::rag::{
    OnionSlice, OnionSliceConfig, SliceMode, create_onion_slices, create_onion_slices_fast,
};
use crate::storage::{ChromaDocument, StorageManager};

/// Channel buffer size for backpressure.
/// 100 items provides good throughput while limiting memory usage.
const CHANNEL_BUFFER_SIZE: usize = 100;

/// Batch size for storage writes to avoid RAM explosion.
const STORAGE_BATCH_SIZE: usize = 100;

/// File content with metadata for pipeline processing.
#[derive(Debug, Clone)]
pub struct FileContent {
    /// Path to the source file.
    pub path: PathBuf,
    /// Extracted text content.
    pub text: String,
    /// Target namespace for storage.
    pub namespace: String,
    /// SHA256 content hash for deduplication.
    pub content_hash: String,
}

/// A chunk ready for embedding.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Chunk ID (generated from content).
    pub id: String,
    /// Chunk content text.
    pub content: String,
    /// Source file path.
    pub source_path: PathBuf,
    /// Target namespace.
    pub namespace: String,
    /// Content hash of source file (for dedup tracking).
    pub source_hash: String,
    /// Onion slice layer (if using onion mode).
    pub layer: u8,
    /// Parent slice ID (for onion hierarchy).
    pub parent_id: Option<String>,
    /// Children slice IDs (for onion hierarchy).
    pub children_ids: Vec<String>,
    /// Extracted keywords.
    pub keywords: Vec<String>,
    /// Additional metadata.
    pub metadata: serde_json::Value,
}

/// An embedded chunk ready for storage.
#[derive(Debug, Clone)]
pub struct EmbeddedChunk {
    /// Original chunk data.
    pub chunk: Chunk,
    /// Embedding vector.
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone)]
struct ChunkBatch {
    path: PathBuf,
    content_hash: String,
    chunks: Vec<Chunk>,
}

#[derive(Debug, Clone)]
struct EmbeddedFile {
    path: PathBuf,
    content_hash: String,
    chunks: Vec<EmbeddedChunk>,
}

/// Pipeline configuration.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Number of files to buffer between reader and chunker.
    pub reader_buffer: usize,
    /// Number of chunk batches to buffer between chunker and embedder.
    pub chunker_buffer: usize,
    /// Number of embedded batches to buffer between embedder and storage.
    pub embedder_buffer: usize,
    /// Slicing mode for chunking.
    pub slice_mode: SliceMode,
    /// Enable storage-backed deduplication.
    pub dedup_enabled: bool,
    /// Optional event stream for progress/reporting consumers.
    pub event_sender: Option<mpsc::UnboundedSender<PipelineEvent>>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            reader_buffer: CHANNEL_BUFFER_SIZE,
            chunker_buffer: CHANNEL_BUFFER_SIZE,
            embedder_buffer: CHANNEL_BUFFER_SIZE,
            slice_mode: SliceMode::default(),
            dedup_enabled: true,
            event_sender: None,
        }
    }
}

/// Pipeline statistics for progress reporting.
#[derive(Debug, Default, Clone)]
pub struct PipelineStats {
    /// Total files scheduled for this pipeline run.
    pub total_files: usize,
    /// Files successfully read and handed to the chunker.
    pub files_read: usize,
    /// Files skipped before chunking (for example exact duplicates).
    pub files_skipped: usize,
    /// Files durably committed to storage.
    pub files_committed: usize,
    /// Files that failed in any stage.
    pub files_failed: usize,
    /// Chunks created.
    pub chunks_created: usize,
    /// Chunks embedded.
    pub chunks_embedded: usize,
    /// Chunks durably stored.
    pub chunks_stored: usize,
    /// Stage-local errors encountered.
    pub errors: usize,
}

/// Runtime snapshot for live progress consumers.
#[derive(Debug, Clone, Default)]
pub struct PipelineSnapshot {
    pub total_files: usize,
    pub files_read: usize,
    pub files_skipped: usize,
    pub files_committed: usize,
    pub files_failed: usize,
    pub chunks_created: usize,
    pub chunks_embedded: usize,
    pub chunks_stored: usize,
    pub errors: usize,
    pub reader_queue_depth: usize,
    pub chunker_queue_depth: usize,
    pub storage_queue_depth: usize,
    pub current_embed_batch_items: usize,
    pub current_embed_batch_chars: usize,
    pub files_per_sec: f64,
    pub chunks_per_sec: f64,
    pub eta: Option<Duration>,
    pub elapsed: Duration,
    pub bottleneck: String,
}

impl PipelineSnapshot {
    pub fn to_stats(&self) -> PipelineStats {
        PipelineStats {
            total_files: self.total_files,
            files_read: self.files_read,
            files_skipped: self.files_skipped,
            files_committed: self.files_committed,
            files_failed: self.files_failed,
            chunks_created: self.chunks_created,
            chunks_embedded: self.chunks_embedded,
            chunks_stored: self.chunks_stored,
            errors: self.errors,
        }
    }
}

/// Progress and lifecycle events emitted by the pipeline.
#[derive(Debug, Clone)]
pub enum PipelineEvent {
    FileRead {
        path: PathBuf,
        content_hash: String,
        bytes: usize,
    },
    FileSkipped {
        path: PathBuf,
        content_hash: String,
        reason: String,
    },
    ChunksCreated {
        path: PathBuf,
        content_hash: String,
        count: usize,
    },
    ChunksEmbedded {
        path: PathBuf,
        content_hash: String,
        count: usize,
        chars: usize,
        elapsed: Duration,
    },
    FileCommitted {
        path: PathBuf,
        content_hash: String,
        chunk_count: usize,
    },
    Error {
        path: Option<PathBuf>,
        stage: &'static str,
        message: String,
    },
    Snapshot(PipelineSnapshot),
}

/// Result of pipeline execution.
#[derive(Debug)]
pub struct PipelineResult {
    /// Pipeline statistics.
    pub stats: PipelineStats,
    /// Error messages if any.
    pub errors: Vec<String>,
}

#[derive(Debug)]
struct PipelineProgressState {
    snapshot: PipelineSnapshot,
    started_at: Instant,
    failed_paths: HashSet<String>,
    error_messages: Vec<String>,
}

impl PipelineProgressState {
    fn new(total_files: usize) -> Self {
        let snapshot = PipelineSnapshot {
            total_files,
            bottleneck: "reader".to_string(),
            ..Default::default()
        };

        Self {
            snapshot,
            started_at: Instant::now(),
            failed_paths: HashSet::new(),
            error_messages: Vec::new(),
        }
    }

    fn record_failure(&mut self, path: &Path) {
        let key = path.to_string_lossy().to_string();
        if self.failed_paths.insert(key) {
            self.snapshot.files_failed += 1;
        }
    }

    fn apply(&mut self, event: &PipelineEvent) -> PipelineSnapshot {
        match event {
            PipelineEvent::FileRead { .. } => {
                self.snapshot.files_read += 1;
                self.snapshot.reader_queue_depth += 1;
            }
            PipelineEvent::FileSkipped { .. } => {
                self.snapshot.files_skipped += 1;
            }
            PipelineEvent::ChunksCreated { count, .. } => {
                self.snapshot.chunks_created += count;
                self.snapshot.reader_queue_depth =
                    self.snapshot.reader_queue_depth.saturating_sub(1);
                self.snapshot.chunker_queue_depth += 1;
            }
            PipelineEvent::ChunksEmbedded {
                count,
                chars,
                elapsed: _,
                ..
            } => {
                self.snapshot.chunks_embedded += count;
                self.snapshot.chunker_queue_depth =
                    self.snapshot.chunker_queue_depth.saturating_sub(1);
                self.snapshot.storage_queue_depth += 1;
                self.snapshot.current_embed_batch_items = *count;
                self.snapshot.current_embed_batch_chars = *chars;
            }
            PipelineEvent::FileCommitted { chunk_count, .. } => {
                self.snapshot.files_committed += 1;
                self.snapshot.chunks_stored += chunk_count;
                self.snapshot.storage_queue_depth =
                    self.snapshot.storage_queue_depth.saturating_sub(1);
            }
            PipelineEvent::Error {
                path,
                stage,
                message,
            } => {
                self.snapshot.errors += 1;
                match *stage {
                    "embedder" => {
                        self.snapshot.chunker_queue_depth =
                            self.snapshot.chunker_queue_depth.saturating_sub(1);
                    }
                    "storage" => {
                        self.snapshot.storage_queue_depth =
                            self.snapshot.storage_queue_depth.saturating_sub(1);
                    }
                    _ => {}
                }
                if let Some(path) = path {
                    self.record_failure(path);
                }
                self.error_messages.push(match path {
                    Some(path) => format!("{} [{}]: {}", stage, path.display(), message),
                    None => format!("{}: {}", stage, message),
                });
            }
            PipelineEvent::Snapshot(snapshot) => {
                self.snapshot = snapshot.clone();
            }
        }

        self.refresh_snapshot()
    }

    fn refresh_snapshot(&mut self) -> PipelineSnapshot {
        let elapsed = self.started_at.elapsed();
        let terminal_files = self.snapshot.files_committed
            + self.snapshot.files_skipped
            + self.snapshot.files_failed;
        let elapsed_secs = elapsed.as_secs_f64();

        self.snapshot.elapsed = elapsed;
        self.snapshot.files_per_sec = if elapsed_secs > 0.0 {
            terminal_files as f64 / elapsed_secs
        } else {
            0.0
        };
        self.snapshot.chunks_per_sec = if elapsed_secs > 0.0 {
            self.snapshot.chunks_stored as f64 / elapsed_secs
        } else {
            0.0
        };

        let remaining_files = self.snapshot.total_files.saturating_sub(terminal_files);
        self.snapshot.eta = if self.snapshot.files_per_sec > 0.0 && remaining_files > 0 {
            Some(Duration::from_secs_f64(
                remaining_files as f64 / self.snapshot.files_per_sec,
            ))
        } else {
            None
        };

        self.snapshot.bottleneck = determine_bottleneck(&self.snapshot);
        self.snapshot.clone()
    }
}

fn determine_bottleneck(snapshot: &PipelineSnapshot) -> String {
    let terminal_files = snapshot.files_committed + snapshot.files_skipped + snapshot.files_failed;
    if terminal_files >= snapshot.total_files {
        return "complete".to_string();
    }

    let mut stage = "idle";
    let mut depth = 0usize;

    if snapshot.reader_queue_depth > depth {
        stage = "chunker";
        depth = snapshot.reader_queue_depth;
    }
    if snapshot.chunker_queue_depth > depth {
        stage = "embedder";
        depth = snapshot.chunker_queue_depth;
    }
    if snapshot.storage_queue_depth > depth {
        stage = "storage";
        depth = snapshot.storage_queue_depth;
    }
    if depth == 0 && snapshot.files_read < snapshot.total_files {
        stage = "reader";
    }

    stage.to_string()
}

#[derive(Clone)]
struct PipelineObserver {
    sender: Option<mpsc::UnboundedSender<PipelineEvent>>,
    state: Arc<Mutex<PipelineProgressState>>,
}

impl PipelineObserver {
    fn new(total_files: usize, sender: Option<mpsc::UnboundedSender<PipelineEvent>>) -> Self {
        Self {
            sender,
            state: Arc::new(Mutex::new(PipelineProgressState::new(total_files))),
        }
    }

    async fn emit(&self, event: PipelineEvent) {
        let snapshot = {
            let mut state = self.state.lock().await;
            state.apply(&event)
        };

        if let Some(sender) = &self.sender {
            let _ = sender.send(event);
            let _ = sender.send(PipelineEvent::Snapshot(snapshot));
        }
    }

    async fn emit_initial_snapshot(&self) {
        let snapshot = {
            let mut state = self.state.lock().await;
            state.refresh_snapshot()
        };

        if let Some(sender) = &self.sender {
            let _ = sender.send(PipelineEvent::Snapshot(snapshot));
        }
    }

    async fn result(&self) -> PipelineResult {
        let state = self.state.lock().await;
        PipelineResult {
            stats: state.snapshot.to_stats(),
            errors: state.error_messages.clone(),
        }
    }
}

// =============================================================================
// STAGE 1: FILE READER
// =============================================================================

/// Stage 1: Read files and extract text content.
///
/// Reads each file, extracts text (supports PDF, text files), and sends
/// to the chunker stage. Handles deduplication check at the storage level.
async fn stage_read_files(
    files: Vec<PathBuf>,
    namespace: String,
    storage: Arc<StorageManager>,
    dedup_enabled: bool,
    tx: mpsc::Sender<FileContent>,
    observer: PipelineObserver,
) {
    for path in files {
        let text = match extract_file_text(&path).await {
            Ok(text) => text,
            Err(err) => {
                warn!("Failed to read file {:?}: {}", path, err);
                observer
                    .emit(PipelineEvent::Error {
                        path: Some(path.clone()),
                        stage: "reader",
                        message: err.to_string(),
                    })
                    .await;
                continue;
            }
        };

        let content_hash = crate::rag::compute_content_hash(&text);

        if dedup_enabled {
            match storage.has_content_hash(&namespace, &content_hash).await {
                Ok(true) => {
                    debug!(
                        "Skipping duplicate: {:?} (hash: {})",
                        path,
                        &content_hash[..16]
                    );
                    observer
                        .emit(PipelineEvent::FileSkipped {
                            path: path.clone(),
                            content_hash,
                            reason: "exact duplicate".to_string(),
                        })
                        .await;
                    continue;
                }
                Ok(false) => {}
                Err(err) => {
                    warn!("Dedup check failed for {:?}: {}", path, err);
                }
            }
        }

        let bytes = text.len();
        let content = FileContent {
            path: path.clone(),
            text,
            namespace: namespace.clone(),
            content_hash: content_hash.clone(),
        };

        if tx.send(content).await.is_err() {
            debug!("Reader: channel closed, stopping");
            break;
        }

        observer
            .emit(PipelineEvent::FileRead {
                path,
                content_hash,
                bytes,
            })
            .await;
    }

    info!("Reader stage complete");
}

/// Extract text content from a file (PDF or text).
async fn extract_file_text(path: &Path) -> Result<String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    if ext == "pdf" {
        let path = path.to_path_buf();
        let pdf_text =
            tokio::task::spawn_blocking(move || pdf_extract::extract_text(&path)).await??;
        return Ok(pdf_text);
    }

    let (_path, content) = crate::path_utils::safe_read_to_string_async(path).await?;
    Ok(content)
}

// =============================================================================
// STAGE 2: CHUNKER
// =============================================================================

/// Stage 2: Create chunks/slices from file content.
async fn stage_chunk_content(
    mut rx: mpsc::Receiver<FileContent>,
    tx: mpsc::Sender<ChunkBatch>,
    slice_mode: SliceMode,
    observer: PipelineObserver,
) {
    let config = OnionSliceConfig::default();

    while let Some(file_content) = rx.recv().await {
        let path = file_content.path.clone();
        let content_hash = file_content.content_hash.clone();
        let chunks = create_chunks_from_content(&file_content, slice_mode, &config);
        let count = chunks.len();

        if tx
            .send(ChunkBatch {
                path: path.clone(),
                content_hash: content_hash.clone(),
                chunks,
            })
            .await
            .is_err()
        {
            debug!("Chunker: channel closed, stopping");
            break;
        }

        observer
            .emit(PipelineEvent::ChunksCreated {
                path,
                content_hash,
                count,
            })
            .await;
    }

    info!("Chunker stage complete");
}

/// Create chunks from file content based on slicing mode.
fn create_chunks_from_content(
    content: &FileContent,
    slice_mode: SliceMode,
    config: &OnionSliceConfig,
) -> Vec<Chunk> {
    let metadata = serde_json::json!({
        "path": content.path.to_str(),
        "content_hash": &content.content_hash,
        "slice_mode": match slice_mode {
            SliceMode::Onion => "onion",
            SliceMode::OnionFast => "onion-fast",
            SliceMode::Flat => "flat",
        },
    });

    match slice_mode {
        SliceMode::Onion => {
            let slices = create_onion_slices(&content.text, &metadata, config);
            slices_to_chunks(slices, content)
        }
        SliceMode::OnionFast => {
            let slices = create_onion_slices_fast(&content.text, &metadata, config);
            slices_to_chunks(slices, content)
        }
        SliceMode::Flat => create_flat_chunks(&content.text, content, metadata),
    }
}

/// Convert onion slices to pipeline chunks.
fn slices_to_chunks(slices: Vec<OnionSlice>, content: &FileContent) -> Vec<Chunk> {
    slices
        .into_iter()
        .map(|slice| {
            let metadata = serde_json::json!({
                "path": content.path.to_str(),
                "content_hash": &content.content_hash,
                "layer": slice.layer.name(),
            });

            Chunk {
                id: slice.id,
                content: slice.content,
                source_path: content.path.clone(),
                namespace: content.namespace.clone(),
                source_hash: content.content_hash.clone(),
                layer: slice.layer.as_u8(),
                parent_id: slice.parent_id,
                children_ids: slice.children_ids,
                keywords: slice.keywords,
                metadata,
            }
        })
        .collect()
}

/// Create flat chunks from content.
fn create_flat_chunks(
    text: &str,
    content: &FileContent,
    base_metadata: serde_json::Value,
) -> Vec<Chunk> {
    let chunks = split_into_chunks(text, 512, 128);
    let total_chunks = chunks.len();

    chunks
        .into_iter()
        .enumerate()
        .map(|(idx, chunk_text)| {
            let mut metadata = base_metadata.clone();
            if let serde_json::Value::Object(ref mut map) = metadata {
                map.insert("chunk_index".to_string(), serde_json::json!(idx));
                map.insert("total_chunks".to_string(), serde_json::json!(total_chunks));
            }

            let id = format!(
                "{}_{}_{}",
                content.path.to_str().unwrap_or("unknown"),
                content.content_hash.get(..8).unwrap_or(""),
                idx
            );

            Chunk {
                id,
                content: chunk_text,
                source_path: content.path.clone(),
                namespace: content.namespace.clone(),
                source_hash: content.content_hash.clone(),
                layer: 0,
                parent_id: None,
                children_ids: vec![],
                keywords: vec![],
                metadata,
            }
        })
        .collect()
}

/// Simple chunking with overlap.
fn split_into_chunks(text: &str, target_size: usize, overlap: usize) -> Vec<String> {
    let mut char_offsets: Vec<usize> = text.char_indices().map(|(byte_idx, _)| byte_idx).collect();
    let len = char_offsets.len();

    if len <= target_size {
        return vec![text.to_string()];
    }

    char_offsets.push(text.len());

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < len {
        let end = (start + target_size).min(len);
        let start_byte = char_offsets[start];
        let end_byte = char_offsets[end];
        chunks.push(text[start_byte..end_byte].to_string());

        if end >= len {
            break;
        }

        start = end.saturating_sub(overlap);
    }

    chunks
}

// =============================================================================
// STAGE 3: EMBEDDER
// =============================================================================

/// Stage 3: Embed chunks using the embedding client.
async fn stage_embed_chunks(
    mut rx: mpsc::Receiver<ChunkBatch>,
    tx: mpsc::Sender<EmbeddedFile>,
    client: Arc<Mutex<EmbeddingClient>>,
    observer: PipelineObserver,
) {
    while let Some(chunk_batch) = rx.recv().await {
        if chunk_batch.chunks.is_empty() {
            continue;
        }

        let path = chunk_batch.path.clone();
        let content_hash = chunk_batch.content_hash.clone();
        let batch_chars: usize = chunk_batch
            .chunks
            .iter()
            .map(|chunk| chunk.content.chars().count())
            .sum();
        let texts: Vec<String> = chunk_batch
            .chunks
            .iter()
            .map(|chunk| chunk.content.clone())
            .collect();

        let start = Instant::now();
        let embeddings = match client.lock().await.embed_batch(&texts).await {
            Ok(embeddings) => embeddings,
            Err(err) => {
                error!("Embedding batch failed for {:?}: {}", path, err);
                observer
                    .emit(PipelineEvent::Error {
                        path: Some(path),
                        stage: "embedder",
                        message: err.to_string(),
                    })
                    .await;
                continue;
            }
        };
        let elapsed = start.elapsed();

        let count = embeddings.len();
        let embedded_chunks: Vec<EmbeddedChunk> = chunk_batch
            .chunks
            .into_iter()
            .zip(embeddings)
            .map(|(chunk, embedding)| EmbeddedChunk { chunk, embedding })
            .collect();

        if tx
            .send(EmbeddedFile {
                path: path.clone(),
                content_hash: content_hash.clone(),
                chunks: embedded_chunks,
            })
            .await
            .is_err()
        {
            debug!("Embedder: channel closed, stopping");
            break;
        }

        observer
            .emit(PipelineEvent::ChunksEmbedded {
                path,
                content_hash,
                count,
                chars: batch_chars,
                elapsed,
            })
            .await;
    }

    info!("Embedder stage complete");
}

// =============================================================================
// STAGE 4: STORAGE
// =============================================================================

/// Stage 4: Store embedded chunks to the database.
async fn stage_store_chunks(
    mut rx: mpsc::Receiver<EmbeddedFile>,
    storage: Arc<StorageManager>,
    observer: PipelineObserver,
) {
    while let Some(mut embedded_file) = rx.recv().await {
        let path = embedded_file.path.clone();
        let content_hash = embedded_file.content_hash.clone();
        let mut stored_for_file = 0usize;
        let mut storage_failed = false;

        while !embedded_file.chunks.is_empty() {
            let take = embedded_file.chunks.len().min(STORAGE_BATCH_SIZE);
            let batch: Vec<EmbeddedChunk> = embedded_file.chunks.drain(..take).collect();

            match store_batch(&storage, batch).await {
                Ok(count) => stored_for_file += count,
                Err(err) => {
                    error!("Storage batch failed for {:?}: {}", path, err);
                    observer
                        .emit(PipelineEvent::Error {
                            path: Some(path.clone()),
                            stage: "storage",
                            message: err.to_string(),
                        })
                        .await;
                    storage_failed = true;
                    break;
                }
            }
        }

        if !storage_failed {
            observer
                .emit(PipelineEvent::FileCommitted {
                    path,
                    content_hash,
                    chunk_count: stored_for_file,
                })
                .await;
        }
    }

    info!("Storage stage complete");
}

/// Store a batch of embedded chunks.
async fn store_batch(storage: &StorageManager, batch: Vec<EmbeddedChunk>) -> Result<usize> {
    let count = batch.len();

    let documents: Vec<ChromaDocument> = batch
        .into_iter()
        .map(|embedded| {
            if embedded.chunk.layer > 0 {
                ChromaDocument {
                    id: embedded.chunk.id,
                    namespace: embedded.chunk.namespace,
                    embedding: embedded.embedding,
                    metadata: embedded.chunk.metadata,
                    document: embedded.chunk.content,
                    layer: embedded.chunk.layer,
                    parent_id: embedded.chunk.parent_id,
                    children_ids: embedded.chunk.children_ids,
                    keywords: embedded.chunk.keywords,
                    content_hash: Some(embedded.chunk.source_hash),
                }
            } else {
                ChromaDocument::new_flat_with_hash(
                    embedded.chunk.id,
                    embedded.chunk.namespace,
                    embedded.embedding,
                    embedded.chunk.metadata,
                    embedded.chunk.content,
                    embedded.chunk.source_hash,
                )
            }
        })
        .collect();

    storage.add_to_store(documents).await?;
    debug!("Stored batch of {} chunks", count);
    Ok(count)
}

// =============================================================================
// PIPELINE COORDINATOR
// =============================================================================

/// Run the async pipeline for document indexing.
///
/// Spawns all stages concurrently and waits for completion.
pub async fn run_pipeline(
    files: Vec<PathBuf>,
    namespace: String,
    storage: Arc<StorageManager>,
    client: Arc<Mutex<EmbeddingClient>>,
    config: PipelineConfig,
) -> Result<PipelineResult> {
    let total_files = files.len();
    info!(
        "Starting pipeline: {} files, mode: {:?}",
        total_files, config.slice_mode
    );

    let observer = PipelineObserver::new(total_files, config.event_sender.clone());
    observer.emit_initial_snapshot().await;

    let (tx1, rx1) = mpsc::channel::<FileContent>(config.reader_buffer);
    let (tx2, rx2) = mpsc::channel::<ChunkBatch>(config.chunker_buffer);
    let (tx3, rx3) = mpsc::channel::<EmbeddedFile>(config.embedder_buffer);

    let storage_for_reader = storage.clone();
    let storage_for_storage = storage;
    let ns_for_reader = namespace.clone();
    let slice_mode = config.slice_mode;
    let dedup_enabled = config.dedup_enabled;

    let reader_handle = tokio::spawn(stage_read_files(
        files,
        ns_for_reader,
        storage_for_reader,
        dedup_enabled,
        tx1,
        observer.clone(),
    ));

    let chunker_handle = tokio::spawn(stage_chunk_content(rx1, tx2, slice_mode, observer.clone()));
    let embedder_handle = tokio::spawn(stage_embed_chunks(rx2, tx3, client, observer.clone()));
    let storage_handle = tokio::spawn(stage_store_chunks(
        rx3,
        storage_for_storage,
        observer.clone(),
    ));

    let (_reader_result, _chunker_result, _embedder_result, _storage_result) = tokio::try_join!(
        reader_handle,
        chunker_handle,
        embedder_handle,
        storage_handle
    )?;

    let result = observer.result().await;

    info!(
        "Pipeline complete: {} files -> {} chunks -> {} stored",
        result.stats.files_committed, result.stats.chunks_created, result.stats.chunks_stored
    );

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_split_into_chunks_short_text() {
        let text = "Hello world";
        let chunks = split_into_chunks(text, 100, 20);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Hello world");
    }

    #[test]
    fn test_split_into_chunks_with_overlap() {
        let text = "abcdefghijklmnopqrstuvwxyz";
        let chunks = split_into_chunks(text, 10, 3);
        assert!(chunks.len() > 1);
        assert_eq!(chunks[0].len(), 10);
        assert!(chunks[0].ends_with(&chunks[1][..3]));
    }

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert_eq!(config.reader_buffer, CHANNEL_BUFFER_SIZE);
        assert_eq!(config.slice_mode, SliceMode::default());
        assert!(config.dedup_enabled);
        assert!(config.event_sender.is_none());
    }

    #[tokio::test]
    async fn test_pipeline_observer_tracks_snapshot_and_failures() {
        let observer = PipelineObserver::new(3, None);
        observer.emit_initial_snapshot().await;

        let path_a = PathBuf::from("a.md");
        let path_b = PathBuf::from("b.md");

        observer
            .emit(PipelineEvent::FileRead {
                path: path_a.clone(),
                content_hash: "hash-a".to_string(),
                bytes: 10,
            })
            .await;
        observer
            .emit(PipelineEvent::ChunksCreated {
                path: path_a.clone(),
                content_hash: "hash-a".to_string(),
                count: 4,
            })
            .await;
        observer
            .emit(PipelineEvent::ChunksEmbedded {
                path: path_a.clone(),
                content_hash: "hash-a".to_string(),
                count: 4,
                chars: 1200,
                elapsed: Duration::from_millis(400),
            })
            .await;
        observer
            .emit(PipelineEvent::FileCommitted {
                path: path_a,
                content_hash: "hash-a".to_string(),
                chunk_count: 4,
            })
            .await;
        observer
            .emit(PipelineEvent::FileSkipped {
                path: path_b.clone(),
                content_hash: "hash-b".to_string(),
                reason: "duplicate".to_string(),
            })
            .await;
        observer
            .emit(PipelineEvent::Error {
                path: Some(path_b),
                stage: "embedder",
                message: "boom".to_string(),
            })
            .await;

        let result = observer.result().await;
        assert_eq!(result.stats.total_files, 3);
        assert_eq!(result.stats.files_read, 1);
        assert_eq!(result.stats.files_skipped, 1);
        assert_eq!(result.stats.files_committed, 1);
        assert_eq!(result.stats.files_failed, 1);
        assert_eq!(result.stats.chunks_created, 4);
        assert_eq!(result.stats.chunks_embedded, 4);
        assert_eq!(result.stats.chunks_stored, 4);
        assert_eq!(result.stats.errors, 1);
        assert_eq!(result.errors.len(), 1);
    }

    #[test]
    fn test_snapshot_to_stats_carries_runtime_truth() {
        let snapshot = PipelineSnapshot {
            total_files: 5,
            files_read: 4,
            files_skipped: 1,
            files_committed: 2,
            files_failed: 1,
            chunks_created: 12,
            chunks_embedded: 10,
            chunks_stored: 8,
            errors: 3,
            ..Default::default()
        };

        let stats = snapshot.to_stats();
        assert_eq!(stats.total_files, 5);
        assert_eq!(stats.files_read, 4);
        assert_eq!(stats.files_skipped, 1);
        assert_eq!(stats.files_committed, 2);
        assert_eq!(stats.files_failed, 1);
        assert_eq!(stats.chunks_created, 12);
        assert_eq!(stats.chunks_embedded, 10);
        assert_eq!(stats.chunks_stored, 8);
        assert_eq!(stats.errors, 3);
    }
}
