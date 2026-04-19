use anyhow::Result;
use indicatif::{HumanDuration, ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, Semaphore, mpsc};

pub use rust_memex::diagnostics::{DedupGroup, DedupResult, KeepStrategy};
use rust_memex::{
    CrossStoreRecoveryReport, EmbeddingClient, EmbeddingConfig, IndexProgressTracker,
    PipelineConfig, PipelineEvent, PipelineSnapshot, PreprocessingConfig, RAGPipeline, SliceMode,
    StorageManager, diagnostics, merge_databases, migrate_namespace_atomic,
    rag::PipelineGovernorConfig, repair_writes as execute_repair_writes,
};

use crate::cli::definition::*;

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct IndexCheckpointStats {
    pub total_files: usize,
    pub discovered_files: usize,
    pub resumed_files: usize,
    pub files_read: usize,
    pub files_skipped: usize,
    pub files_committed: usize,
    pub files_failed: usize,
    pub chunks_created: usize,
    pub chunks_embedded: usize,
    pub chunks_stored: usize,
    pub errors: usize,
}

impl From<&PipelineSnapshot> for IndexCheckpointStats {
    fn from(snapshot: &PipelineSnapshot) -> Self {
        Self {
            total_files: snapshot.total_files,
            discovered_files: snapshot.discovered_files,
            resumed_files: snapshot.resumed_files,
            files_read: snapshot.files_read,
            files_skipped: snapshot.files_skipped,
            files_committed: snapshot.files_committed,
            files_failed: snapshot.files_failed,
            chunks_created: snapshot.chunks_created,
            chunks_embedded: snapshot.chunks_embedded,
            chunks_stored: snapshot.chunks_stored,
            errors: snapshot.errors,
        }
    }
}

/// Checkpoint for resumable indexing
#[derive(Debug, Serialize, Deserialize)]
pub struct IndexCheckpoint {
    /// Namespace being indexed
    pub namespace: String,
    /// Database path tied to this checkpoint
    #[serde(default)]
    pub db_path: Option<String>,
    /// Files that have been successfully indexed (canonical paths)
    pub indexed_files: HashSet<String>,
    /// Content hashes that were durably committed or already satisfied
    #[serde(default)]
    pub indexed_hashes: HashSet<String>,
    /// When checkpoint was last updated
    pub updated_at: String,
    /// Optional runtime stats snapshot
    #[serde(default)]
    pub stats: Option<IndexCheckpointStats>,
}

impl IndexCheckpoint {
    pub fn new(namespace: &str, db_path: &str) -> Self {
        Self {
            namespace: namespace.to_string(),
            db_path: Some(db_path.to_string()),
            indexed_files: HashSet::new(),
            indexed_hashes: HashSet::new(),
            updated_at: chrono::Utc::now().to_rfc3339(),
            stats: None,
        }
    }

    pub fn checkpoint_path(db_path: &str, namespace: &str) -> PathBuf {
        let expanded = shellexpand::tilde(db_path).to_string();
        Path::new(&expanded)
            .parent()
            .unwrap_or(Path::new("."))
            .join(format!(".index-checkpoint-{}.json", namespace))
    }

    pub fn load(db_path: &str, namespace: &str) -> Option<Self> {
        let path = Self::checkpoint_path(db_path, namespace);
        if path.exists() {
            std::fs::read_to_string(&path)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
        } else {
            None
        }
    }

    pub fn save(&mut self, db_path: &str) -> Result<()> {
        self.db_path = Some(db_path.to_string());
        self.updated_at = chrono::Utc::now().to_rfc3339();
        let path = Self::checkpoint_path(db_path, &self.namespace);
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, json)?;
        Ok(())
    }

    pub fn delete(db_path: &str, namespace: &str) {
        let path = Self::checkpoint_path(db_path, namespace);
        let _ = std::fs::remove_file(path);
    }

    pub fn mark_indexed(&mut self, file_path: &Path) {
        self.mark_indexed_with_hash(file_path, None);
    }

    pub fn mark_indexed_with_hash(&mut self, file_path: &Path, content_hash: Option<&str>) {
        self.indexed_files
            .insert(file_path.to_string_lossy().to_string());
        if let Some(content_hash) = content_hash {
            self.indexed_hashes.insert(content_hash.to_string());
        }
    }

    pub fn is_indexed(&self, file_path: &Path) -> bool {
        self.indexed_files
            .contains(&file_path.to_string_lossy().to_string())
    }

    pub fn update_from_snapshot(&mut self, snapshot: &PipelineSnapshot) {
        self.stats = Some(IndexCheckpointStats::from(snapshot));
    }
}

/// Configuration for batch indexing operation
pub struct BatchIndexConfig {
    pub path: PathBuf,
    pub namespace: Option<String>,
    pub recursive: bool,
    pub glob_pattern: Option<String>,
    pub max_depth: usize,
    pub db_path: String,
    pub preprocess: bool,
    /// Sanitize timestamps/UUIDs/session IDs (default: false = preserve for temporal queries)
    pub sanitize_metadata: bool,
    pub slice_mode: SliceMode,
    pub dedup: bool,
    pub embedding_config: EmbeddingConfig,
    /// Show progress bar with calibration-based ETA
    pub show_progress: bool,
    /// Resume from checkpoint if interrupted
    pub resume: bool,
    /// Enable async pipeline mode for concurrent stages
    pub pipeline: bool,
    /// Maximum embedding requests in flight for pipeline mode.
    pub pipeline_embed_concurrency: u8,
    /// Enable adaptive governor for pipeline embed throughput.
    pub pipeline_governor: bool,
    /// Number of files to process in parallel (1-16, ignored in pipeline mode)
    pub parallel: u8,
}

/// Result of indexing a single file (for parallel processing)
#[derive(Debug)]
pub enum FileIndexResult {
    /// File was indexed successfully
    Indexed,
    /// File was skipped (duplicate content)
    Skipped,
    /// File was skipped (already in checkpoint)
    SkippedResume,
    /// Indexing failed
    Failed,
}

fn pipeline_embed_rate(snapshot: &PipelineSnapshot) -> f64 {
    let elapsed_secs = snapshot.elapsed.as_secs_f64();
    if elapsed_secs > 0.0 {
        snapshot.chunks_embedded as f64 / elapsed_secs
    } else {
        0.0
    }
}

fn format_pipeline_status_line(snapshot: &PipelineSnapshot, scheduled_files: usize) -> String {
    let terminal_files = snapshot.files_committed + snapshot.files_skipped + snapshot.files_failed;
    let discovered_files = snapshot
        .discovered_files
        .max(scheduled_files.saturating_add(snapshot.resumed_files))
        .max(scheduled_files);
    let completed_discovered = terminal_files
        .saturating_add(snapshot.resumed_files)
        .min(discovered_files);
    let eta = snapshot
        .eta
        .map(HumanDuration)
        .map(|value| value.to_string())
        .unwrap_or_else(|| "--".to_string());
    let avg_embed_ms = snapshot
        .avg_embed_batch_ms
        .map(|value| format!("{value:.0}ms"))
        .unwrap_or_else(|| "--".to_string());

    format!(
        "{}/{} discovered | scheduled {} resumed {} | read {} | committed {} skipped {} failed {} | chunks created {} embedded {} ({:.1}/s) stored {} ({:.1}/s) | eta {} | q {}/{}/{} | embed {}/{} req @ {} | batch {}/{} items {} / {} chars | gov {} ({}) | {}",
        completed_discovered,
        discovered_files,
        scheduled_files,
        snapshot.resumed_files,
        snapshot.files_read,
        snapshot.files_committed,
        snapshot.files_skipped,
        snapshot.files_failed,
        snapshot.chunks_created,
        snapshot.chunks_embedded,
        pipeline_embed_rate(snapshot),
        snapshot.chunks_stored,
        snapshot.chunks_per_sec,
        eta,
        snapshot.reader_queue_depth,
        snapshot.chunker_queue_depth,
        snapshot.storage_queue_depth,
        snapshot.embed_active_requests,
        snapshot.embed_concurrency_limit,
        avg_embed_ms,
        snapshot.current_embed_batch_items,
        snapshot.embed_batch_items_limit,
        snapshot.current_embed_batch_chars,
        snapshot.embed_batch_chars_limit,
        snapshot.governor_mode,
        snapshot.governor_reason,
        snapshot.bottleneck
    )
}

fn format_pipeline_error_line(path: Option<&Path>, stage: &str, message: &str) -> String {
    match path {
        Some(path) => format!(
            "[pipeline:error] {} [{}] {}",
            stage,
            path.display(),
            message
        ),
        None => format!("[pipeline:error] {} {}", stage, message),
    }
}

fn should_disable_pipeline_storage_dedup(existing_checkpoint_loaded: bool) -> bool {
    existing_checkpoint_loaded
}

struct PipelineProgressRenderer {
    total_files: usize,
    progress_bar: Option<ProgressBar>,
    last_line_at: Instant,
}

impl PipelineProgressRenderer {
    fn new(total_files: usize, interactive: bool) -> Self {
        let progress_bar = if interactive {
            let pb = ProgressBar::new(total_files as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} files | {msg}")
                    .expect("invalid pipeline progress template")
                    .progress_chars("#>-"),
            );
            Some(pb)
        } else {
            None
        };

        Self {
            total_files,
            progress_bar,
            last_line_at: Instant::now() - Duration::from_secs(5),
        }
    }

    fn println(&self, line: &str) {
        if let Some(progress_bar) = &self.progress_bar {
            progress_bar.println(line);
        } else {
            eprintln!("{}", line);
        }
    }

    fn render(&mut self, snapshot: &PipelineSnapshot, force: bool) {
        let terminal_files =
            snapshot.files_committed + snapshot.files_skipped + snapshot.files_failed;
        let discovered_files = snapshot
            .discovered_files
            .max(self.total_files.saturating_add(snapshot.resumed_files))
            .max(self.total_files);
        let completed_discovered = terminal_files
            .saturating_add(snapshot.resumed_files)
            .min(discovered_files);
        let message = format_pipeline_status_line(snapshot, self.total_files);

        if let Some(progress_bar) = &self.progress_bar {
            progress_bar.set_length(discovered_files as u64);
            progress_bar.set_position(completed_discovered as u64);
            progress_bar.set_message(message.clone());
            if force && completed_discovered >= discovered_files {
                progress_bar.finish_with_message(format!("complete | {}", message));
            }
            return;
        }

        if !force && self.last_line_at.elapsed() < Duration::from_secs(1) {
            return;
        }

        self.last_line_at = Instant::now();
        eprintln!("[pipeline] {}", message);
    }
}

struct PipelineEventConsumerConfig {
    checkpoint: Option<Arc<Mutex<IndexCheckpoint>>>,
    db_path: String,
    show_progress: bool,
    interactive_progress: bool,
    scheduled_files: usize,
    discovered_files: usize,
    resumed_files: usize,
}

async fn consume_pipeline_events(
    mut rx: mpsc::UnboundedReceiver<PipelineEvent>,
    config: PipelineEventConsumerConfig,
) -> PipelineSnapshot {
    let PipelineEventConsumerConfig {
        checkpoint,
        db_path,
        show_progress,
        interactive_progress,
        scheduled_files,
        discovered_files,
        resumed_files,
    } = config;

    let mut renderer =
        show_progress.then(|| PipelineProgressRenderer::new(scheduled_files, interactive_progress));
    let mut latest_snapshot = PipelineSnapshot {
        total_files: scheduled_files,
        discovered_files,
        resumed_files,
        ..Default::default()
    };

    while let Some(event) = rx.recv().await {
        match event {
            PipelineEvent::FileCommitted {
                path, content_hash, ..
            } => {
                if let Some(checkpoint) = &checkpoint {
                    let mut checkpoint = checkpoint.lock().await;
                    checkpoint.mark_indexed_with_hash(&path, Some(&content_hash));
                    let _ = checkpoint.save(&db_path);
                }
            }
            PipelineEvent::FileSkipped {
                path, content_hash, ..
            } => {
                if let Some(checkpoint) = &checkpoint {
                    let mut checkpoint = checkpoint.lock().await;
                    checkpoint.mark_indexed_with_hash(&path, Some(&content_hash));
                    let _ = checkpoint.save(&db_path);
                }
            }
            PipelineEvent::Snapshot(snapshot) => {
                latest_snapshot = *snapshot;
                if let Some(checkpoint) = &checkpoint {
                    let mut checkpoint = checkpoint.lock().await;
                    checkpoint.update_from_snapshot(&latest_snapshot);
                    let _ = checkpoint.save(&db_path);
                }
                if let Some(renderer) = &mut renderer {
                    renderer.render(&latest_snapshot, false);
                }
            }
            PipelineEvent::Error {
                path,
                stage,
                message,
            } => {
                if let Some(renderer) = &renderer {
                    renderer.println(&format_pipeline_error_line(
                        path.as_deref(),
                        stage,
                        &message,
                    ));
                } else {
                    eprintln!(
                        "{}",
                        format_pipeline_error_line(path.as_deref(), stage, &message)
                    );
                }
            }
            PipelineEvent::FileRead { .. }
            | PipelineEvent::EmbedStarted { .. }
            | PipelineEvent::ChunksCreated { .. }
            | PipelineEvent::ChunksEmbedded { .. }
            | PipelineEvent::GovernorAdjusted { .. } => {}
        }
    }

    if let Some(renderer) = &mut renderer {
        renderer.render(&latest_snapshot, true);
    }

    latest_snapshot
}

/// Run batch indexing with optional pipeline mode for concurrent processing
pub async fn run_batch_index(config: BatchIndexConfig) -> Result<()> {
    let BatchIndexConfig {
        path,
        namespace,
        recursive,
        glob_pattern,
        max_depth,
        db_path,
        preprocess,
        sanitize_metadata,
        slice_mode,
        dedup,
        embedding_config,
        show_progress,
        resume,
        pipeline,
        pipeline_embed_concurrency,
        pipeline_governor,
        parallel,
    } = config;
    // Expand and canonicalize path - canonicalize validates path exists and resolves symlinks
    let expanded = shellexpand::tilde(path.to_str().unwrap_or("")).to_string();
    let canonical = Path::new(&expanded).canonicalize()?;

    // Collect files
    let files = collect_files(&canonical, recursive, glob_pattern.as_deref(), max_depth)?;
    let total = files.len();

    if total == 0 {
        eprintln!("No files found matching criteria");
        return Ok(());
    }

    let mode_name = match slice_mode {
        SliceMode::Onion => "onion (hierarchical, 4 layers)",
        SliceMode::OnionFast => "onion-fast (outer+core, 2 layers)",
        SliceMode::Flat => "flat (traditional chunks)",
    };

    let use_progress_bar = show_progress && std::io::stderr().is_terminal();
    if show_progress && !use_progress_bar {
        eprintln!("Warning: --progress requires an interactive terminal (using line logs)");
    }

    let tracker = if use_progress_bar {
        let t = IndexProgressTracker::pre_scan(&files);
        t.display_pre_scan();
        Some(t)
    } else {
        eprintln!("Found {} files to index (slice mode: {})", total, mode_name);
        if preprocess {
            eprintln!("Preprocessing enabled: filtering tool artifacts, CLI output, and metadata");
        }
        if dedup {
            eprintln!("Deduplication enabled: skipping files with identical content");
        }
        None
    };

    // Initialize RAG pipeline - db_path is from CLI args or config, validated at load time
    let expanded_db = shellexpand::tilde(&db_path).to_string();
    let db_dir = Path::new(&expanded_db);
    if let Some(parent) = db_dir.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Use full storage for CLI batch indexing to ensure BM25 is written
    let embedding_client = Arc::new(Mutex::new(EmbeddingClient::new(&embedding_config).await?));
    let storage = Arc::new(StorageManager::new(&expanded_db).await?);

    let ns_name = namespace.as_deref().unwrap_or("rag");

    // Pipeline mode: concurrent stages with channels
    if pipeline {
        if preprocess {
            eprintln!("Warning: --preprocess is not supported in pipeline mode (ignoring)");
        }

        let (checkpoint, existing_checkpoint_loaded) = if resume {
            if let Some(cp) = IndexCheckpoint::load(&db_path, ns_name) {
                let resumed_count = cp.indexed_files.len();
                eprintln!(
                    "Resuming from checkpoint: {} files already committed",
                    resumed_count
                );
                (Arc::new(Mutex::new(cp)), true)
            } else {
                (
                    Arc::new(Mutex::new(IndexCheckpoint::new(ns_name, &db_path))),
                    false,
                )
            }
        } else {
            IndexCheckpoint::delete(&db_path, ns_name);
            (
                Arc::new(Mutex::new(IndexCheckpoint::new(ns_name, &db_path))),
                false,
            )
        };

        let (pipeline_files, resumed_count, disable_storage_dedup) = if resume {
            let checkpoint_guard = checkpoint.lock().await;
            let resumed_count = checkpoint_guard.indexed_files.len();
            let filtered_files: Vec<PathBuf> = files
                .iter()
                .filter(|path| !checkpoint_guard.is_indexed(path))
                .cloned()
                .collect();
            let disable_storage_dedup =
                should_disable_pipeline_storage_dedup(existing_checkpoint_loaded);
            (filtered_files, resumed_count, disable_storage_dedup)
        } else {
            (files.clone(), 0, false)
        };

        if pipeline_files.is_empty() {
            eprintln!("Pipeline resume complete: all files already committed");
            if resume {
                IndexCheckpoint::delete(&db_path, ns_name);
            }
            return Ok(());
        }

        if disable_storage_dedup {
            eprintln!(
                "Pipeline resume: using checkpoint truth for committed files to avoid partial-write false positives"
            );
        }

        eprintln!(
            "Pipeline mode: {} files ({} discovered, {} resumed), slice mode: {:?}, embed concurrency ceiling: {}, governor: {}",
            pipeline_files.len(),
            total,
            resumed_count,
            slice_mode,
            pipeline_embed_concurrency,
            if pipeline_governor {
                "adaptive"
            } else {
                "fixed"
            }
        );
        eprintln!("Running concurrent stages: reader -> chunker -> embedder -> storage");

        let (event_tx, event_rx) = mpsc::unbounded_channel();
        let event_task = tokio::spawn(consume_pipeline_events(
            event_rx,
            PipelineEventConsumerConfig {
                checkpoint: resume.then_some(Arc::clone(&checkpoint)),
                db_path: db_path.clone(),
                show_progress,
                interactive_progress: use_progress_bar,
                scheduled_files: pipeline_files.len(),
                discovered_files: total,
                resumed_files: resumed_count,
            },
        ));

        let pipeline_config = PipelineConfig {
            slice_mode,
            dedup_enabled: dedup && !disable_storage_dedup,
            embed_concurrency: pipeline_embed_concurrency as usize,
            governor: pipeline_governor.then(|| {
                PipelineGovernorConfig::adaptive(
                    embedding_config.max_batch_chars,
                    embedding_config.max_batch_items,
                    pipeline_embed_concurrency as usize,
                )
            }),
            event_sender: Some(event_tx),
            discovered_files: total,
            resumed_files: resumed_count,
            ..Default::default()
        };

        let pipeline_run = rust_memex::run_pipeline(
            pipeline_files,
            ns_name.to_string(),
            storage,
            embedding_client,
            pipeline_config,
        )
        .await;
        let snapshot = event_task.await?;
        let result = pipeline_run?;

        eprintln!();
        eprintln!("Pipeline complete:");
        eprintln!("  Files committed:   {}", result.stats.files_committed);
        eprintln!("  Files read:        {}", result.stats.files_read);
        if result.stats.files_skipped > 0 {
            eprintln!("  Files skipped:     {}", result.stats.files_skipped);
        }
        if result.stats.files_failed > 0 {
            eprintln!("  Files failed:      {}", result.stats.files_failed);
        }
        if resumed_count > 0 {
            eprintln!("  Skipped (resumed): {}", resumed_count);
        }
        eprintln!("  Chunks created:    {}", result.stats.chunks_created);
        eprintln!("  Chunks embedded:   {}", result.stats.chunks_embedded);
        eprintln!("  Chunks stored:     {}", result.stats.chunks_stored);
        if result.stats.errors > 0 {
            eprintln!("  Errors:            {}", result.stats.errors);
            for error in result.errors.iter().take(5) {
                eprintln!("    - {}", error);
            }
            if result.errors.len() > 5 {
                eprintln!("    - ... and {} more", result.errors.len() - 5);
            }
        }
        if let Some(avg_embed_ms) = snapshot.avg_embed_batch_ms {
            eprintln!("  Avg embed batch:   {:.0} ms", avg_embed_ms);
        }
        eprintln!(
            "  Embed ceiling:     {} req | {} items | {} chars",
            snapshot.embed_concurrency_limit,
            snapshot.embed_batch_items_limit,
            snapshot.embed_batch_chars_limit
        );
        eprintln!(
            "  Governor:          {} ({})",
            snapshot.governor_mode, snapshot.governor_reason
        );
        eprintln!("  Bottleneck:        {}", snapshot.bottleneck);
        eprintln!("  Namespace:         {}", ns_name);
        eprintln!("  DB path:           {}", expanded_db);

        if resume && result.stats.files_failed == 0 {
            IndexCheckpoint::delete(&db_path, ns_name);
            eprintln!("Checkpoint cleared (pipeline completed successfully)");
        } else if resume && result.stats.files_failed > 0 {
            eprintln!(
                "Checkpoint preserved ({} files failed - rerun with --resume to retry)",
                result.stats.files_failed
            );
        }

        return Ok(());
    }

    // Standard (non-pipeline) mode with parallel file processing
    let rag = Arc::new(RAGPipeline::new(embedding_client, storage).await?);

    // Note: preprocessing currently uses flat mode
    let effective_mode = if preprocess {
        SliceMode::Flat
    } else {
        slice_mode
    };

    // Initialize checkpoint for resume capability (wrapped for thread-safe access)
    let checkpoint = if resume {
        if let Some(cp) = IndexCheckpoint::load(&db_path, ns_name) {
            let resumed_count = cp.indexed_files.len();
            eprintln!(
                "Resuming from checkpoint: {} files already indexed",
                resumed_count
            );
            Arc::new(Mutex::new(cp))
        } else {
            Arc::new(Mutex::new(IndexCheckpoint::new(ns_name, &db_path)))
        }
    } else {
        // Clean start - remove any stale checkpoint
        IndexCheckpoint::delete(&db_path, ns_name);
        Arc::new(Mutex::new(IndexCheckpoint::new(ns_name, &db_path)))
    };

    // Atomic counters for thread-safe progress tracking
    let indexed_count = Arc::new(AtomicUsize::new(0));
    let skipped_count = Arc::new(AtomicUsize::new(0));
    let skipped_resume_count = Arc::new(AtomicUsize::new(0));
    let failed_count = Arc::new(AtomicUsize::new(0));
    let total_chunks_count = Arc::new(AtomicUsize::new(0));
    let processed_count = Arc::new(AtomicUsize::new(0));

    // Semaphore to limit concurrent file processing
    let semaphore = Arc::new(Semaphore::new(parallel as usize));

    // Get embedder model name for calibration display
    let embedder_model = embedding_config
        .providers
        .first()
        .map(|p| p.model.clone())
        .unwrap_or_else(|| "unknown".to_string());

    // Flag to track if calibration is complete (for progress bar)
    let calibration_done = Arc::new(AtomicBool::new(false));

    // Wrap tracker for shared access
    let tracker = tracker.map(|t| Arc::new(Mutex::new(t)));

    // Start calibration if progress mode
    if let Some(ref t) = tracker {
        t.lock().await.start_calibration();
    }

    // Create task handles for parallel processing
    let mut handles = Vec::with_capacity(files.len());

    for file_path in files.into_iter() {
        // Clone shared resources for this task
        let semaphore = Arc::clone(&semaphore);
        let rag = Arc::clone(&rag);
        let checkpoint = Arc::clone(&checkpoint);
        let tracker = tracker.clone();
        let indexed_count = Arc::clone(&indexed_count);
        let skipped_count = Arc::clone(&skipped_count);
        let skipped_resume_count = Arc::clone(&skipped_resume_count);
        let failed_count = Arc::clone(&failed_count);
        let total_chunks_count = Arc::clone(&total_chunks_count);
        let processed_count = Arc::clone(&processed_count);
        let calibration_done = Arc::clone(&calibration_done);
        let db_path = db_path.clone();
        let ns = namespace.clone();
        let canonical = canonical.clone();
        let embedder_model = embedder_model.clone();
        let _ns_name = ns_name.to_string();

        let handle = tokio::spawn(async move {
            // Acquire semaphore permit to limit concurrency
            let _permit = semaphore.acquire().await.expect("semaphore closed");

            let display_path = file_path
                .strip_prefix(&canonical)
                .unwrap_or(&file_path)
                .display()
                .to_string();

            // Check if file already indexed (resume mode)
            if resume {
                let cp = checkpoint.lock().await;
                if cp.is_indexed(&file_path) {
                    drop(cp);
                    skipped_resume_count.fetch_add(1, Ordering::SeqCst);
                    processed_count.fetch_add(1, Ordering::SeqCst);
                    if let Some(ref t) = tracker {
                        t.lock().await.file_skipped();
                    }
                    return FileIndexResult::SkippedResume;
                }
            }

            // Get file size for calibration
            let file_bytes = std::fs::metadata(&file_path).map(|m| m.len()).unwrap_or(0);

            // Update progress display
            let current_processed = processed_count.load(Ordering::SeqCst);
            if let Some(ref t) = tracker {
                t.lock().await.set_message(&display_path);
            } else {
                let progress = format!("[{}/{}]", current_processed + 1, total);
                eprintln!("{} Indexing {}... ", progress, display_path);
            }

            // Build preprocessing config
            let preprocess_config = PreprocessingConfig {
                remove_metadata: sanitize_metadata,
                ..Default::default()
            };

            let result = if dedup {
                // Use dedup-enabled indexing
                if preprocess {
                    rag.index_document_with_preprocessing_and_dedup(
                        &file_path,
                        ns.as_deref(),
                        preprocess_config,
                    )
                    .await
                } else {
                    rag.index_document_with_dedup(&file_path, ns.as_deref(), effective_mode)
                        .await
                }
            } else {
                // Use original indexing without dedup (convert to IndexResult-like outcome)
                if preprocess {
                    rag.index_document_with_preprocessing(
                        &file_path,
                        ns.as_deref(),
                        preprocess_config,
                    )
                    .await
                    .map(|()| rust_memex::IndexResult::Indexed {
                        chunks_indexed: (file_bytes as usize / 500).max(1),
                        content_hash: String::new(),
                        embedder_ms: None,
                        tokens_estimated: None,
                    })
                } else {
                    rag.index_document_with_mode(&file_path, ns.as_deref(), effective_mode)
                        .await
                        .map(|()| rust_memex::IndexResult::Indexed {
                            chunks_indexed: (file_bytes as usize / 500).max(1),
                            content_hash: String::new(),
                            embedder_ms: None,
                            tokens_estimated: None,
                        })
                }
            };

            let file_result = match result {
                Ok(rust_memex::IndexResult::Indexed { chunks_indexed, .. }) => {
                    // Handle calibration on first completed file
                    if !calibration_done.swap(true, Ordering::SeqCst)
                        && let Some(ref t) = tracker
                    {
                        let mut guard = t.lock().await;
                        guard.finish_calibration(chunks_indexed, &embedder_model);
                        guard.adjust_estimate(file_bytes, chunks_indexed);
                        guard.start_progress_bar();
                    }

                    indexed_count.fetch_add(1, Ordering::SeqCst);
                    total_chunks_count.fetch_add(chunks_indexed, Ordering::SeqCst);

                    if let Some(ref t) = tracker {
                        t.lock().await.file_indexed(chunks_indexed);
                    } else {
                        eprintln!("  -> {} done ({} chunks)", display_path, chunks_indexed);
                    }

                    // Update checkpoint
                    if resume {
                        let mut cp = checkpoint.lock().await;
                        cp.mark_indexed(&file_path);
                        let _ = cp.save(&db_path);
                    }

                    FileIndexResult::Indexed
                }
                Ok(rust_memex::IndexResult::Skipped { reason, .. }) => {
                    // Handle calibration if this was the first file
                    if !calibration_done.swap(true, Ordering::SeqCst)
                        && let Some(ref t) = tracker
                    {
                        let mut guard = t.lock().await;
                        guard.finish_calibration(0, &embedder_model);
                        guard.start_progress_bar();
                    }

                    skipped_count.fetch_add(1, Ordering::SeqCst);

                    if let Some(ref t) = tracker {
                        t.lock().await.file_skipped();
                    } else {
                        eprintln!("  -> {} SKIPPED ({})", display_path, reason);
                    }

                    // Mark as indexed even if skipped (content exists)
                    if resume {
                        let mut cp = checkpoint.lock().await;
                        cp.mark_indexed(&file_path);
                        let _ = cp.save(&db_path);
                    }

                    FileIndexResult::Skipped
                }
                Err(e) => {
                    // Handle calibration if this was the first file
                    if !calibration_done.swap(true, Ordering::SeqCst)
                        && let Some(ref t) = tracker
                    {
                        let mut guard = t.lock().await;
                        guard.finish_calibration(0, &embedder_model);
                        guard.start_progress_bar();
                    }

                    failed_count.fetch_add(1, Ordering::SeqCst);

                    if let Some(ref t) = tracker {
                        t.lock().await.file_failed();
                    } else {
                        eprintln!("  -> {} FAILED: {}", display_path, e);
                    }

                    FileIndexResult::Failed
                }
            };

            processed_count.fetch_add(1, Ordering::SeqCst);
            file_result
        });

        handles.push(handle);
    }

    // Wait for all tasks to complete
    let mut results = Vec::with_capacity(handles.len());
    for handle in handles {
        match handle.await {
            Ok(result) => results.push(result),
            Err(e) => {
                // Task panicked - count as failure
                failed_count.fetch_add(1, Ordering::SeqCst);
                eprintln!("Task panicked: {}", e);
            }
        }
    }

    // Get final counts from atomics
    let indexed = indexed_count.load(Ordering::SeqCst);
    let skipped = skipped_count.load(Ordering::SeqCst);
    let skipped_resume = skipped_resume_count.load(Ordering::SeqCst);
    let failed = failed_count.load(Ordering::SeqCst);
    let total_chunks = total_chunks_count.load(Ordering::SeqCst);

    // Display summary
    if let Some(ref t) = tracker {
        let mut guard = t.lock().await;
        guard.finish();
        guard.display_summary();
        if skipped_resume > 0 {
            eprintln!("  Skipped (resumed): {}", skipped_resume);
        }
    } else {
        eprintln!();

        // Determine outcome and show appropriate summary
        let all_skipped = indexed == 0 && skipped > 0 && failed == 0;
        let all_failed = indexed == 0 && skipped == 0 && failed > 0;

        if all_skipped {
            eprintln!("Indexing complete: All content already indexed");
            eprintln!();
            eprintln!("  Files checked:     {}", total);
            eprintln!("  Already indexed:   {} (skipped)", skipped);
            if skipped_resume > 0 {
                eprintln!("  Resumed from:      {} (checkpoint)", skipped_resume);
            }
            eprintln!();
            eprintln!("  [OK] No new content to index - your memory is up to date!");
        } else if all_failed {
            eprintln!("Indexing FAILED: No files were indexed");
            eprintln!();
            eprintln!("  Files attempted:   {}", total);
            eprintln!("  Failed:            {}", failed);
            eprintln!();
            eprintln!("  [!] Check file permissions and embedding server connectivity");
        } else {
            eprintln!("Indexing complete:");
            eprintln!();
            eprintln!("  New chunks:        {}", total_chunks);
            eprintln!("  Files indexed:     {}", indexed);
            if dedup && skipped > 0 {
                eprintln!("  Already indexed:   {} (skipped)", skipped);
            }
            if skipped_resume > 0 {
                eprintln!("  Resumed from:      {} (checkpoint)", skipped_resume);
            }
            if failed > 0 {
                eprintln!("  Failed:            {}", failed);
            }
            eprintln!("  Total processed:   {}", total);
        }

        eprintln!();
        eprintln!("Config:");
        if let Some(ref ns) = namespace {
            eprintln!("  Namespace:         {}", ns);
        }
        eprintln!("  Slice mode:        {}", mode_name);
        eprintln!("  Parallel workers:  {}", parallel);
        eprintln!(
            "  Deduplication:     {}",
            if dedup { "enabled" } else { "disabled" }
        );
        eprintln!("  DB path:           {}", expanded_db);
    }

    // Clean up checkpoint on successful completion
    if resume && failed == 0 {
        IndexCheckpoint::delete(&db_path, ns_name);
        eprintln!("Checkpoint cleared (all files indexed successfully)");
    } else if resume && failed > 0 {
        eprintln!(
            "Checkpoint preserved ({} files failed - rerun with --resume to retry)",
            failed
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_pipeline_status_line_surfaces_stage_flow() {
        let snapshot = PipelineSnapshot {
            total_files: 6,
            discovered_files: 8,
            resumed_files: 2,
            files_read: 4,
            files_committed: 2,
            files_skipped: 1,
            files_failed: 1,
            chunks_created: 24,
            chunks_embedded: 20,
            chunks_stored: 18,
            errors: 0,
            files_per_sec: 0.8,
            chunks_per_sec: 3.6,
            reader_queue_depth: 2,
            chunker_queue_depth: 1,
            storage_queue_depth: 3,
            embed_active_requests: 1,
            embed_concurrency_limit: 3,
            current_embed_batch_items: 8,
            current_embed_batch_chars: 4096,
            embed_batch_items_limit: 16,
            embed_batch_chars_limit: 8192,
            avg_embed_batch_ms: Some(640.0),
            bottleneck: "storage".to_string(),
            governor_mode: "adaptive".to_string(),
            governor_reason: "backlog sustained".to_string(),
            eta: Some(Duration::from_secs(12)),
            elapsed: Duration::from_secs(5),
        };

        let line = format_pipeline_status_line(&snapshot, 6);

        assert!(line.contains("6/8 discovered"));
        assert!(line.contains("scheduled 6 resumed 2"));
        assert!(line.contains("read 4"));
        assert!(line.contains("embedded 20 (4.0/s)"));
        assert!(line.contains("stored 18 (3.6/s)"));
        assert!(line.contains("q 2/1/3"));
        assert!(line.contains("embed 1/3 req @ 640ms"));
        assert!(line.contains("batch 8/16 items 4096 / 8192 chars"));
        assert!(line.contains("gov adaptive (backlog sustained)"));
        assert!(line.contains("storage"));
    }

    #[test]
    fn format_pipeline_error_line_keeps_stage_and_path_visible() {
        let line = format_pipeline_error_line(
            Some(Path::new("/tmp/corpus/a.md")),
            "embedder",
            "connection reset",
        );

        assert_eq!(
            line,
            "[pipeline:error] embedder [/tmp/corpus/a.md] connection reset"
        );
    }

    #[test]
    fn resume_checkpoint_disables_pipeline_storage_dedup_even_without_committed_files() {
        assert!(should_disable_pipeline_storage_dedup(true));
        assert!(!should_disable_pipeline_storage_dedup(false));
    }
}

fn print_cross_store_recovery_report(report: &CrossStoreRecoveryReport, execute: bool) {
    let mode = if execute { "EXECUTE" } else { "DRY RUN" };
    eprintln!("\n=== CROSS-STORE RECOVERY ({}) ===\n", mode);
    eprintln!("Recovery dir: {}", report.recovery_dir);
    eprintln!("Pending batches: {}", report.pending_batches);
    eprintln!("  Divergent:   {}", report.divergent_batches);
    eprintln!("  Rolled back: {}", report.rolled_back_batches);
    eprintln!("  Stale:       {}", report.stale_batches);
    eprintln!("  Clean:       {}", report.clean_batches);
    eprintln!("Documents examined: {}", report.documents_examined);
    eprintln!("Missing BM25 docs:  {}", report.documents_missing_bm25);
    eprintln!("Missing Lance docs: {}", report.documents_missing_lance);
    if execute {
        eprintln!("Repaired docs:      {}", report.repaired_documents);
        eprintln!("Skipped docs:       {}", report.skipped_documents);
        eprintln!("Cleared batches:    {}", report.cleared_batches);
    }

    if report.batches.is_empty() {
        eprintln!("\nNo recovery ledgers found.");
        return;
    }

    eprintln!();
    for batch in &report.batches {
        let state = match batch.state {
            rust_memex::CrossStoreRecoveryState::Clean => "clean",
            rust_memex::CrossStoreRecoveryState::Divergent => "divergent",
            rust_memex::CrossStoreRecoveryState::RolledBack => "rolled_back",
            rust_memex::CrossStoreRecoveryState::Stale => "stale",
        };
        eprintln!(
            "- {} [{}] state={} docs={} lance={} bm25={}",
            batch.batch_id,
            batch.namespace,
            state,
            batch.document_count,
            batch.lance_documents,
            batch.bm25_documents
        );
        if let Some(ref error) = batch.last_error {
            eprintln!("  last_error: {}", error);
        }
        if !batch.missing_bm25_ids.is_empty() {
            eprintln!("  missing_bm25: {}", batch.missing_bm25_ids.join(", "));
        }
        if !batch.missing_lance_ids.is_empty() {
            eprintln!("  missing_lance: {}", batch.missing_lance_ids.join(", "));
        }
    }
}

pub async fn run_repair_writes(
    db_path: String,
    namespace: Option<String>,
    execute: bool,
    json_output: bool,
) -> Result<()> {
    let repair = execute_repair_writes(&db_path, namespace.as_deref(), execute).await?;
    let report = repair.report;

    if json_output {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_cross_store_recovery_report(&report, execute);
    }

    Ok(())
}

/// Run deduplication on the database
pub async fn run_dedup(
    namespace: Option<String>,
    dry_run: bool,
    keep_strategy: KeepStrategy,
    cross_namespace: bool,
    json_output: bool,
    db_path: String,
) -> Result<()> {
    let storage = Arc::new(StorageManager::new_lance_only(&db_path).await?);
    let result = diagnostics::deduplicate_documents(
        storage.as_ref(),
        namespace.as_deref(),
        dry_run,
        keep_strategy,
        cross_namespace,
    )
    .await?;

    if result.total_docs == 0 {
        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "status": "empty",
                    "message": "No documents found",
                    "namespace": namespace,
                }))?
            );
        } else {
            eprintln!("No documents found in database.");
        }
        return Ok(());
    }

    if !json_output {
        eprintln!("Scanning {} documents for duplicates...", result.total_docs);
        if dry_run {
            eprintln!("(dry-run mode: no changes will be made)");
        }
    }

    // Output results
    if json_output {
        let output = serde_json::json!({
            "dry_run": dry_run,
            "namespace": namespace,
            "cross_namespace": cross_namespace,
            "keep_strategy": format!("{:?}", keep_strategy).to_lowercase(),
            "result": result,
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        eprintln!();
        eprintln!(
            "Deduplication {}:",
            if dry_run { "report" } else { "complete" }
        );
        eprintln!("  Total documents:     {}", result.total_docs);
        eprintln!("  Unique documents:    {}", result.unique_docs);
        eprintln!("  Duplicate groups:    {}", result.duplicate_groups);
        eprintln!(
            "  Duplicates {}:  {}",
            if dry_run { "found" } else { "removed" },
            result.duplicates_removed
        );
        if result.docs_without_hash > 0 {
            eprintln!(
                "  Without hash:        {} (cannot deduplicate)",
                result.docs_without_hash
            );
        }

        // Show some duplicate groups if any
        if !result.groups.is_empty() {
            eprintln!();
            let show_count = result.groups.len().min(5);
            eprintln!(
                "Sample duplicate groups ({} of {}):",
                show_count,
                result.groups.len()
            );
            for group in result.groups.iter().take(show_count) {
                eprintln!();
                eprintln!(
                    "  Hash: {}...",
                    &group.content_hash[..group.content_hash.len().min(16)]
                );
                eprintln!("  Kept: {} (ns: {})", group.kept_id, group.kept_namespace);
                for removed in &group.removed {
                    eprintln!(
                        "  {} {} (ns: {})",
                        if dry_run { "Would remove:" } else { "Removed:" },
                        removed.id,
                        removed.namespace
                    );
                }
            }
            if result.groups.len() > 5 {
                eprintln!();
                eprintln!("  ... and {} more groups", result.groups.len() - 5);
            }
        }

        if dry_run && result.duplicates_removed > 0 {
            eprintln!();
            eprintln!("To actually remove duplicates, run with: --dry-run false");
        }
    }

    Ok(())
}

/// Migration result for reporting
#[derive(Debug, Clone, Serialize)]
pub struct MigrationResult {
    pub from_namespace: String,
    pub to_namespace: String,
    pub docs_migrated: usize,
    pub docs_merged: usize,
    pub source_deleted: bool,
    pub dry_run: bool,
}

/// Migrate documents from one namespace to another
pub async fn run_migrate_namespace(
    from: String,
    to: String,
    db_path: String,
    merge: bool,
    delete_source: bool,
    dry_run: bool,
    json_output: bool,
) -> Result<()> {
    let db_path = shellexpand::tilde(&db_path).to_string();
    let storage = StorageManager::new_lance_only(&db_path).await?;

    // Edge case: same source and target
    if from == to {
        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "status": "no-op",
                    "message": "Source and target namespaces are the same",
                    "namespace": from
                }))?
            );
        } else {
            eprintln!(
                "Warning: Source and target namespaces are the same ('{}').",
                from
            );
            eprintln!("No migration needed.");
        }
        return Ok(());
    }

    // Check if source namespace exists
    let source_exists = storage.namespace_exists(&from).await?;
    if !source_exists {
        let msg = format!("Source namespace '{}' does not exist or is empty", from);
        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "status": "error",
                    "message": msg
                }))?
            );
        } else {
            eprintln!("Error: {}", msg);
        }
        return Err(anyhow::anyhow!(msg));
    }

    // Check if target namespace exists
    let target_exists = storage.namespace_exists(&to).await?;
    if target_exists && !merge {
        let msg = format!(
            "Target namespace '{}' already exists. Use --merge to merge documents.",
            to
        );
        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "status": "error",
                    "message": msg,
                    "hint": "Use --merge flag to merge into existing namespace"
                }))?
            );
        } else {
            eprintln!("Error: {}", msg);
        }
        return Err(anyhow::anyhow!(msg));
    }

    // Get all documents from source namespace
    let source_docs = storage.get_all_in_namespace(&from).await?;
    let source_count = source_docs.len();

    if source_count == 0 {
        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "status": "no-op",
                    "message": "Source namespace is empty",
                    "namespace": from
                }))?
            );
        } else {
            eprintln!("Source namespace '{}' is empty. Nothing to migrate.", from);
        }
        return Ok(());
    }

    // Get target document count for merge reporting
    let target_count_before = if target_exists {
        storage.count_namespace(&to).await?
    } else {
        0
    };

    if dry_run {
        // Report what would happen
        let result = MigrationResult {
            from_namespace: from.clone(),
            to_namespace: to.clone(),
            docs_migrated: source_count,
            docs_merged: if target_exists {
                target_count_before
            } else {
                0
            },
            source_deleted: delete_source,
            dry_run: true,
        };

        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "status": "dry-run",
                    "result": result,
                    "message": "No changes made"
                }))?
            );
        } else {
            eprintln!("\n-> Dry Run: Namespace Migration\n");
            eprintln!("  From:           '{}'", from);
            eprintln!("  To:             '{}'", to);
            eprintln!("  Docs to move:   {}", source_count);
            if target_exists {
                eprintln!("  Existing docs:  {} (will be merged)", target_count_before);
            }
            eprintln!(
                "  Delete source:  {}",
                if delete_source { "yes" } else { "no" }
            );
            eprintln!("\nNo changes made (dry run).");
        }
        return Ok(());
    }

    if !merge && delete_source {
        let outcome = migrate_namespace_atomic(&storage, &from, &to).await?;
        let result = MigrationResult {
            from_namespace: outcome.from_namespace.clone(),
            to_namespace: outcome.to_namespace.clone(),
            docs_migrated: outcome.migrated_chunks,
            docs_merged: 0,
            source_deleted: true,
            dry_run: false,
        };

        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "status": "success",
                    "result": result
                }))?
            );
        } else {
            eprintln!("\n-> Namespace Migration Complete\n");
            eprintln!("  From:           '{}'", outcome.from_namespace);
            eprintln!("  To:             '{}'", outcome.to_namespace);
            eprintln!("  Docs migrated:  {}", outcome.migrated_chunks);
            eprintln!("  Source '{}': atomically renamed", outcome.from_namespace);
            eprintln!("\n  DB path: {}", db_path);
        }

        return Ok(());
    }

    // Perform the migration
    // Create new documents with updated namespace
    let migrated_docs: Vec<rust_memex::ChromaDocument> = source_docs
        .into_iter()
        .map(|mut doc| {
            doc.namespace = to.clone();
            doc
        })
        .collect();

    // Insert into target namespace
    storage.add_to_store(migrated_docs).await?;

    // Delete source namespace if requested
    let source_deleted = if delete_source {
        storage.delete_namespace_documents(&from).await?;
        true
    } else {
        false
    };

    // Report results
    let result = MigrationResult {
        from_namespace: from.clone(),
        to_namespace: to.clone(),
        docs_migrated: source_count,
        docs_merged: if target_exists {
            target_count_before
        } else {
            0
        },
        source_deleted,
        dry_run: false,
    };

    if json_output {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "status": "success",
                "result": result
            }))?
        );
    } else {
        eprintln!("\n-> Namespace Migration Complete\n");
        eprintln!("  From:           '{}'", from);
        eprintln!("  To:             '{}'", to);
        eprintln!("  Docs migrated:  {}", source_count);
        if target_exists {
            eprintln!("  Merged with:    {} existing docs", target_count_before);
            eprintln!(
                "  Total in '{}': {}",
                to,
                source_count + target_count_before
            );
        }
        if source_deleted {
            eprintln!("  Source '{}': deleted", from);
        } else {
            eprintln!(
                "  Source '{}': preserved (use --delete-source to remove)",
                from
            );
        }
        eprintln!("\n  DB path: {}", db_path);
    }

    Ok(())
}

/// Purge (delete) all documents in a namespace
pub async fn run_purge_namespace(
    namespace: String,
    db_path: String,
    confirm: bool,
    json_output: bool,
) -> Result<()> {
    let db_path = shellexpand::tilde(&db_path).to_string();
    let storage = StorageManager::new_lance_only(&db_path).await?;

    // Check if namespace exists
    let exists = storage.namespace_exists(&namespace).await?;
    if !exists {
        let msg = format!("Namespace '{}' does not exist or is empty", namespace);
        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "status": "error",
                    "message": msg
                }))?
            );
        } else {
            eprintln!("Error: {}", msg);
        }
        return Err(anyhow::anyhow!(msg));
    }

    // Get count before purge
    let docs = storage.get_all_in_namespace(&namespace).await?;
    let doc_count = docs.len();

    // Confirmation prompt (unless --confirm flag)
    if !confirm && !json_output {
        eprintln!(
            "\n⚠️  WARNING: This will permanently delete {} documents from namespace '{}'",
            doc_count, namespace
        );
        eprintln!("   This action cannot be undone!\n");
        eprint!("   Type 'yes' to confirm: ");

        use std::io::{self, BufRead, Write};
        io::stderr().flush()?;
        let stdin = io::stdin();
        let mut input = String::new();
        stdin.lock().read_line(&mut input)?;

        if input.trim().to_lowercase() != "yes" {
            eprintln!("\n   Aborted. No changes made.");
            return Ok(());
        }
    }

    // Perform the purge
    let deleted = storage.delete_namespace_documents(&namespace).await?;

    // Report results
    if json_output {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "status": "success",
                "namespace": namespace,
                "documents_deleted": doc_count,
                "rows_deleted": deleted
            }))?
        );
    } else {
        eprintln!("\n✓ Purged namespace '{}'", namespace);
        eprintln!("  Documents deleted: {}", doc_count);
        eprintln!("  Rows deleted: {}", deleted);
        eprintln!("  DB path: {}", db_path);
    }

    Ok(())
}

/// Statistics for merge operation
#[derive(Debug, Clone, Default, Serialize)]
pub struct MergeStats {
    /// Total documents found in sources
    pub total_docs: usize,
    /// Documents copied to target
    pub docs_copied: usize,
    /// Documents skipped (duplicates)
    pub docs_skipped: usize,
    /// Namespaces merged
    pub namespaces: HashSet<String>,
    /// Source databases processed
    pub sources_processed: usize,
    /// Errors encountered (non-fatal)
    pub errors: usize,
}

/// Merge multiple LanceDB databases into one
pub async fn run_merge(
    source_paths: Vec<PathBuf>,
    target_path: PathBuf,
    dedup: bool,
    namespace_prefix: Option<String>,
    dry_run: bool,
    json_output: bool,
) -> Result<()> {
    let mut stats = MergeStats::default();
    let execution = merge_databases(
        source_paths.clone(),
        target_path,
        dedup,
        namespace_prefix.clone(),
        dry_run,
    )
    .await?;
    let validated_target = execution.target_path;
    stats.total_docs = execution.progress.total_docs;
    stats.docs_copied = execution.progress.docs_copied;
    stats.docs_skipped = execution.progress.docs_skipped;
    stats.namespaces = execution.progress.namespaces.into_iter().collect();
    stats.sources_processed = execution.progress.sources_processed;
    stats.errors = execution.progress.errors;

    if !json_output {
        eprintln!("\n=== RMCP-MEMEX MERGE ===\n");
        eprintln!("Sources: {} database(s)", source_paths.len());
        for src in &source_paths {
            eprintln!("  - {}", src.display());
        }
        eprintln!("Target:  {}", validated_target.display());
        if let Some(ref prefix) = namespace_prefix {
            eprintln!("Prefix:  {}", prefix);
        }
        eprintln!("Dedup:   {}", if dedup { "enabled" } else { "disabled" });
        if dry_run {
            eprintln!("\n[DRY RUN - no changes will be made]\n");
        }
        eprintln!();
    }

    // Output final summary
    if json_output {
        let output = serde_json::json!({
            "status": if dry_run { "dry_run" } else { "completed" },
            "sources_processed": stats.sources_processed,
            "total_docs": stats.total_docs,
            "docs_copied": stats.docs_copied,
            "docs_skipped": stats.docs_skipped,
            "namespaces": stats.namespaces.iter().collect::<Vec<_>>(),
            "namespace_count": stats.namespaces.len(),
            "errors": stats.errors,
            "target": validated_target.display().to_string(),
            "dedup_enabled": dedup,
            "namespace_prefix": namespace_prefix,
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        eprintln!(
            "=== MERGE {} ===\n",
            if dry_run { "PREVIEW" } else { "COMPLETE" }
        );
        eprintln!("  Sources processed: {}", stats.sources_processed);
        eprintln!("  Total documents:   {}", stats.total_docs);
        eprintln!("  Documents copied:  {}", stats.docs_copied);
        if dedup && stats.docs_skipped > 0 {
            eprintln!("  Skipped (dedup):   {}", stats.docs_skipped);
        }
        eprintln!("  Namespaces:        {}", stats.namespaces.len());
        if stats.errors > 0 {
            eprintln!("  Errors:            {}", stats.errors);
        }
        eprintln!("  Target database:   {}", validated_target.display());

        if dry_run {
            eprintln!("\n[DRY RUN - run without --dry-run to apply changes]");
        }
    }

    Ok(())
}
