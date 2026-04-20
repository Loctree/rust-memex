use anyhow::{Result, anyhow};
use async_stream::try_stream;
use axum::body::Bytes;
use futures::{Stream, StreamExt};
use memex_contracts::progress::{ReindexProgress, ReprocessProgress};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, BufReader};

use crate::{
    ChromaDocument, PreprocessingConfig, Preprocessor, RAGPipeline, SliceMode, StorageManager,
    compute_content_hash, path_utils,
};

const EXPORT_PAGE_SIZE: usize = 5_000;

/// JSONL export record structure shared across CLI and HTTP lifecycle flows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportRecord {
    pub id: String,
    pub text: String,
    pub metadata: Value,
    pub content_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embeddings: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ReprocessOutcome {
    pub target_namespace: String,
    pub source_label: String,
    pub source_records: usize,
    pub canonical_documents: usize,
    pub indexed_documents: usize,
    pub replaced_documents: usize,
    pub skipped_existing_documents: usize,
    pub skipped_empty_documents: usize,
    pub skipped_preprocess_short_documents: usize,
    pub failed_documents: usize,
    pub failed_ids: Vec<String>,
    pub parse_errors: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ReindexOutcome {
    pub source_namespace: String,
    pub target_namespace: String,
    pub source_records: usize,
    pub canonical_documents: usize,
    pub indexed_documents: usize,
    pub replaced_documents: usize,
    pub skipped_documents: usize,
    pub failed_documents: usize,
    pub failed_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ImportOutcome {
    pub imported_count: usize,
    pub skipped_count: usize,
    pub error_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct NamespaceMigrationOutcome {
    pub from_namespace: String,
    pub to_namespace: String,
    pub migrated_chunks: usize,
}

#[derive(Debug, Clone)]
pub struct ReprocessJob {
    pub input_path: PathBuf,
    pub target_namespace: String,
    pub slice_mode: SliceMode,
    pub preprocess: bool,
    pub skip_existing: bool,
    pub dry_run: bool,
}

#[derive(Debug, Clone)]
pub struct ReindexJob {
    pub source_namespace: String,
    pub target_namespace: String,
    pub slice_mode: SliceMode,
    pub preprocess: bool,
    pub skip_existing: bool,
    pub dry_run: bool,
}

#[derive(Debug, Clone)]
struct ReprocessDocument {
    canonical_id: String,
    source_record_id: String,
    text: String,
    metadata: Value,
    source_text_hash: String,
    collapsed_records: usize,
}

#[derive(Debug, Clone)]
struct RebuildPlan {
    namespace: String,
    source_label: String,
    source_records: usize,
    docs: Vec<ReprocessDocument>,
    slice_mode: SliceMode,
    preprocess: bool,
    skip_existing: bool,
    dry_run: bool,
    parse_errors: usize,
}

#[derive(Debug, Clone, Default)]
struct RebuildProgress {
    processed_documents: usize,
    indexed_documents: usize,
    skipped_documents: usize,
    failed_documents: usize,
}

#[derive(Debug, Clone, Default)]
struct RebuildStats {
    indexed_documents: usize,
    replaced_documents: usize,
    skipped_existing_documents: usize,
    skipped_empty_documents: usize,
    skipped_preprocess_short_documents: usize,
    failed_ids: Vec<String>,
}

pub fn default_reindexed_namespace(namespace: &str) -> String {
    format!("{namespace}-reindexed")
}

pub fn export_namespace_jsonl_stream(
    storage: Arc<StorageManager>,
    namespace: String,
    include_embeddings: bool,
) -> impl Stream<Item = Result<String>> + Send + 'static {
    try_stream! {
        let mut offset = 0usize;
        loop {
            let page = storage
                .all_documents_page(Some(&namespace), offset, EXPORT_PAGE_SIZE)
                .await?;
            let page_len = page.len();

            for doc in page {
                let record = ExportRecord {
                    id: doc.id,
                    text: doc.document,
                    metadata: doc.metadata,
                    content_hash: doc.content_hash,
                    embeddings: include_embeddings.then_some(doc.embedding),
                };

                let mut line = serde_json::to_string(&record)?;
                line.push('\n');
                yield line;
            }

            if page_len < EXPORT_PAGE_SIZE {
                break;
            }
            offset += page_len;
        }
    }
}

pub async fn import_jsonl_file(
    rag: Arc<RAGPipeline>,
    namespace: String,
    input: &Path,
    skip_existing: bool,
) -> Result<ImportOutcome> {
    let validated = path_utils::validate_read_path(input)?;
    let file = tokio::fs::File::open(&validated)
        .await
        .map_err(|err| anyhow!("Failed to open '{}': {}", validated.display(), err))?;
    let reader = BufReader::new(file);
    import_jsonl_reader(rag, namespace, skip_existing, reader).await
}

pub async fn import_jsonl_reader<R>(
    rag: Arc<RAGPipeline>,
    namespace: String,
    skip_existing: bool,
    reader: R,
) -> Result<ImportOutcome>
where
    R: AsyncBufRead + Unpin,
{
    let storage = rag.storage_manager();
    let mut outcome = ImportOutcome::default();
    let mut lines = reader.lines();

    while let Some(line) = lines.next_line().await? {
        if process_import_line(
            rag.as_ref(),
            storage.as_ref(),
            &namespace,
            skip_existing,
            line.as_bytes(),
            &mut outcome,
        )
        .await
        .is_err()
        {
            outcome.error_count += 1;
        }
    }

    Ok(outcome)
}

pub async fn import_jsonl_bytes_stream<S, E>(
    rag: Arc<RAGPipeline>,
    namespace: String,
    skip_existing: bool,
    mut stream: S,
) -> Result<ImportOutcome>
where
    S: Stream<Item = std::result::Result<Bytes, E>> + Unpin,
    E: std::fmt::Display,
{
    let storage = rag.storage_manager();
    let mut outcome = ImportOutcome::default();
    let mut buffer = Vec::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|err| anyhow!("multipart stream error: {err}"))?;
        buffer.extend_from_slice(&chunk);

        while let Some(position) = buffer.iter().position(|byte| *byte == b'\n') {
            let line = buffer.drain(..=position).collect::<Vec<_>>();
            if process_import_line(
                rag.as_ref(),
                storage.as_ref(),
                &namespace,
                skip_existing,
                &line,
                &mut outcome,
            )
            .await
            .is_err()
            {
                outcome.error_count += 1;
            }
        }
    }

    if !buffer.is_empty()
        && process_import_line(
            rag.as_ref(),
            storage.as_ref(),
            &namespace,
            skip_existing,
            &buffer,
            &mut outcome,
        )
        .await
        .is_err()
    {
        outcome.error_count += 1;
    }

    Ok(outcome)
}

pub async fn migrate_namespace_atomic(
    storage: &StorageManager,
    from: &str,
    to: &str,
) -> Result<NamespaceMigrationOutcome> {
    let migrated_chunks = storage.rename_namespace_atomic(from, to).await?;
    Ok(NamespaceMigrationOutcome {
        from_namespace: from.to_string(),
        to_namespace: to.to_string(),
        migrated_chunks,
    })
}

pub async fn reprocess_jsonl_file<F>(
    rag: Arc<RAGPipeline>,
    job: ReprocessJob,
    mut on_progress: F,
) -> Result<ReprocessOutcome>
where
    F: FnMut(ReprocessProgress),
{
    let ReprocessJob {
        input_path,
        target_namespace,
        slice_mode,
        preprocess,
        skip_existing,
        dry_run,
    } = job;
    let (_validated, content) = path_utils::safe_read_to_string_async(&input_path).await?;
    let (records, parse_errors) = parse_export_records(&content);

    if records.is_empty() {
        return Ok(ReprocessOutcome {
            target_namespace,
            source_label: input_path.display().to_string(),
            parse_errors,
            ..ReprocessOutcome::default()
        });
    }

    let source_records = records.len();
    let docs = collapse_export_records(records);
    let source_label = input_path.display().to_string();
    let canonical_documents = docs.len();

    let stats = run_rebuild_documents(
        rag,
        RebuildPlan {
            namespace: target_namespace.clone(),
            source_label: source_label.clone(),
            source_records,
            docs,
            slice_mode,
            preprocess,
            skip_existing,
            dry_run,
            parse_errors,
        },
        |progress| {
            on_progress(ReprocessProgress {
                source_label: source_label.clone(),
                processed_documents: progress.processed_documents,
                indexed_documents: progress.indexed_documents,
                skipped_documents: progress.skipped_documents,
                failed_documents: progress.failed_documents,
            });
        },
    )
    .await?;

    Ok(ReprocessOutcome {
        target_namespace,
        source_label,
        source_records,
        canonical_documents,
        indexed_documents: stats.indexed_documents,
        replaced_documents: stats.replaced_documents,
        skipped_existing_documents: stats.skipped_existing_documents,
        skipped_empty_documents: stats.skipped_empty_documents,
        skipped_preprocess_short_documents: stats.skipped_preprocess_short_documents,
        failed_documents: stats.failed_ids.len(),
        failed_ids: stats.failed_ids,
        parse_errors,
    })
}

pub async fn reindex_namespace<F>(
    rag: Arc<RAGPipeline>,
    job: ReindexJob,
    mut on_progress: F,
) -> Result<ReindexOutcome>
where
    F: FnMut(ReindexProgress),
{
    let ReindexJob {
        source_namespace,
        target_namespace,
        slice_mode,
        preprocess,
        skip_existing,
        dry_run,
    } = job;
    if source_namespace == target_namespace {
        return Err(anyhow!(
            "Source and target namespace are the same ('{}'). Use a different target namespace.",
            source_namespace
        ));
    }

    let storage = rag.storage_manager();
    let target_count = storage.count_namespace(&target_namespace).await?;
    if target_count > 0 && !skip_existing {
        return Err(anyhow!(
            "Target namespace '{}' already contains {} documents. Pass skip_existing to resume, or use a fresh target namespace.",
            target_namespace,
            target_count
        ));
    }

    let mut records = Vec::new();
    let mut offset = 0usize;
    loop {
        let page = storage
            .all_documents_page(Some(&source_namespace), offset, EXPORT_PAGE_SIZE)
            .await?;
        let page_len = page.len();

        for doc in page {
            records.push(ExportRecord {
                id: doc.id,
                text: doc.document,
                metadata: doc.metadata,
                content_hash: doc.content_hash,
                embeddings: None,
            });
        }

        if page_len < EXPORT_PAGE_SIZE {
            break;
        }
        offset += page_len;
    }

    if records.is_empty() {
        return Ok(ReindexOutcome {
            source_namespace,
            target_namespace,
            ..ReindexOutcome::default()
        });
    }

    let source_records = records.len();
    let docs = collapse_export_records(records);
    let canonical_documents = docs.len();
    let progress_namespace = target_namespace.clone();

    let stats = run_rebuild_documents(
        rag,
        RebuildPlan {
            namespace: target_namespace.clone(),
            source_label: format!("namespace:{source_namespace}"),
            source_records,
            docs,
            slice_mode,
            preprocess,
            skip_existing,
            dry_run,
            parse_errors: 0,
        },
        |progress| {
            on_progress(ReindexProgress {
                namespace: progress_namespace.clone(),
                total_files: canonical_documents,
                processed_files: progress.processed_documents,
                indexed_files: progress.indexed_documents,
                skipped_files: progress.skipped_documents,
                failed_files: progress.failed_documents,
                total_chunks: source_records,
            });
        },
    )
    .await?;

    Ok(ReindexOutcome {
        source_namespace,
        target_namespace,
        source_records,
        canonical_documents,
        indexed_documents: stats.indexed_documents,
        replaced_documents: stats.replaced_documents,
        skipped_documents: stats.skipped_existing_documents
            + stats.skipped_empty_documents
            + stats.skipped_preprocess_short_documents,
        failed_documents: stats.failed_ids.len(),
        failed_ids: stats.failed_ids,
    })
}

fn preferred_reprocess_id(record: &ExportRecord) -> String {
    record
        .metadata
        .get("original_id")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            record
                .metadata
                .get("doc_id")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
        })
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| record.id.clone())
}

fn reprocess_layer_rank(metadata: &Value) -> u8 {
    match metadata.get("layer").and_then(Value::as_str) {
        Some("core") => 4,
        Some("inner") => 3,
        Some("middle") => 2,
        Some("outer") => 1,
        _ => 0,
    }
}

fn should_replace_reprocess_candidate(
    current: &ReprocessDocument,
    candidate: &ReprocessDocument,
) -> bool {
    let current_rank = (current.text.len(), reprocess_layer_rank(&current.metadata));
    let candidate_rank = (
        candidate.text.len(),
        reprocess_layer_rank(&candidate.metadata),
    );
    candidate_rank > current_rank
}

fn collapse_export_records(records: Vec<ExportRecord>) -> Vec<ReprocessDocument> {
    let mut grouped: HashMap<String, ReprocessDocument> = HashMap::new();

    for record in records {
        let text = record.text.trim();
        if text.is_empty() {
            continue;
        }

        let candidate = ReprocessDocument {
            canonical_id: preferred_reprocess_id(&record),
            source_record_id: record.id,
            text: text.to_string(),
            metadata: record.metadata,
            source_text_hash: compute_content_hash(text),
            collapsed_records: 1,
        };

        match grouped.entry(candidate.canonical_id.clone()) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(candidate);
            }
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                let total_records = entry.get().collapsed_records + 1;
                if should_replace_reprocess_candidate(entry.get(), &candidate) {
                    let mut replacement = candidate;
                    replacement.collapsed_records = total_records;
                    entry.insert(replacement);
                } else {
                    entry.get_mut().collapsed_records = total_records;
                }
            }
        }
    }

    let mut docs: Vec<ReprocessDocument> = grouped.into_values().collect();
    docs.sort_by(|left, right| left.canonical_id.cmp(&right.canonical_id));
    docs
}

fn reprocess_slice_mode_name(slice_mode: SliceMode) -> &'static str {
    match slice_mode {
        SliceMode::Onion => "onion",
        SliceMode::OnionFast => "onion-fast",
        SliceMode::Flat => "flat",
    }
}

fn prepare_reprocess_metadata(
    metadata: &Value,
    source_record_id: &str,
    source_text_hash: &str,
    collapsed_records: usize,
    slice_mode: SliceMode,
    source_label: &str,
    preprocess: bool,
) -> Value {
    let mut map = match metadata.clone() {
        Value::Object(map) => map,
        _ => Map::new(),
    };

    for key in [
        "layer",
        "parent_id",
        "children_ids",
        "original_id",
        "slice_mode",
        "content_hash",
    ] {
        map.remove(key);
    }

    map.insert(
        "slice_mode".to_string(),
        json!(reprocess_slice_mode_name(slice_mode)),
    );
    map.insert(
        "reprocess_source_record_id".to_string(),
        json!(source_record_id),
    );
    map.insert("reprocess_source_hash".to_string(), json!(source_text_hash));
    map.insert(
        "reprocess_collapsed_records".to_string(),
        json!(collapsed_records),
    );
    map.insert("reprocess_source".to_string(), json!(source_label));
    if preprocess {
        map.insert("reprocess_preprocessed".to_string(), json!(true));
    }

    Value::Object(map)
}

fn parse_export_records(content: &str) -> (Vec<ExportRecord>, usize) {
    let mut parse_errors = 0usize;
    let mut records = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        match serde_json::from_str::<ExportRecord>(trimmed) {
            Ok(record) => records.push(record),
            Err(_) => {
                parse_errors += 1;
            }
        }
    }

    (records, parse_errors)
}

async fn process_import_line(
    rag: &RAGPipeline,
    storage: &StorageManager,
    namespace: &str,
    skip_existing: bool,
    raw_line: &[u8],
    outcome: &mut ImportOutcome,
) -> Result<()> {
    let text = std::str::from_utf8(raw_line)?.trim();
    if text.is_empty() {
        return Ok(());
    }

    let record: ExportRecord = serde_json::from_str(text)?;
    let content_hash = record
        .content_hash
        .clone()
        .unwrap_or_else(|| compute_content_hash(&record.text));

    if skip_existing && storage.has_content_hash(namespace, &content_hash).await? {
        outcome.skipped_count += 1;
        return Ok(());
    }

    if let Some(embedding) = record.embeddings {
        let document = ChromaDocument::new_flat_with_hash(
            record.id,
            namespace.to_string(),
            embedding,
            record.metadata,
            record.text,
            content_hash,
        );
        storage.add_to_store(vec![document]).await?;
    } else {
        rag.memory_upsert(namespace, record.id, record.text, record.metadata)
            .await?;
    }

    outcome.imported_count += 1;
    Ok(())
}

async fn run_rebuild_documents<F>(
    rag: Arc<RAGPipeline>,
    plan: RebuildPlan,
    mut emit_progress: F,
) -> Result<RebuildStats>
where
    F: FnMut(&RebuildProgress),
{
    let RebuildPlan {
        namespace,
        source_label,
        source_records,
        docs,
        slice_mode,
        preprocess,
        skip_existing,
        dry_run,
        parse_errors: _parse_errors,
    } = plan;

    if docs.is_empty() {
        return Ok(RebuildStats::default());
    }

    if dry_run {
        return Ok(RebuildStats::default());
    }

    rag.embedding_healthcheck().await?;

    let preprocessor = preprocess.then(|| Preprocessor::new(PreprocessingConfig::default()));
    let min_length = PreprocessingConfig::default().min_content_length;
    let mut stats = RebuildStats::default();
    let mut progress = RebuildProgress::default();

    for (idx, doc) in docs.iter().enumerate() {
        let existing = rag.lookup_memory(&namespace, &doc.canonical_id).await?;
        if let Some(existing_doc) = existing.as_ref()
            && skip_existing
            && existing_doc
                .metadata
                .get("reprocess_source_hash")
                .and_then(Value::as_str)
                == Some(doc.source_text_hash.as_str())
        {
            stats.skipped_existing_documents += 1;
            progress.processed_documents = idx + 1;
            progress.skipped_documents = stats.skipped_existing_documents
                + stats.skipped_empty_documents
                + stats.skipped_preprocess_short_documents;
            progress.failed_documents = stats.failed_ids.len();
            progress.indexed_documents = stats.indexed_documents;
            emit_progress(&progress);
            continue;
        }

        let text = if let Some(preprocessor) = &preprocessor {
            preprocessor.extract_semantic_content(&doc.text)
        } else {
            doc.text.clone()
        };

        if text.trim().is_empty() {
            stats.skipped_empty_documents += 1;
            progress.processed_documents = idx + 1;
            progress.skipped_documents = stats.skipped_existing_documents
                + stats.skipped_empty_documents
                + stats.skipped_preprocess_short_documents;
            progress.failed_documents = stats.failed_ids.len();
            progress.indexed_documents = stats.indexed_documents;
            emit_progress(&progress);
            continue;
        }

        if preprocess && text.trim().len() < min_length {
            stats.skipped_preprocess_short_documents += 1;
            progress.processed_documents = idx + 1;
            progress.skipped_documents = stats.skipped_existing_documents
                + stats.skipped_empty_documents
                + stats.skipped_preprocess_short_documents;
            progress.failed_documents = stats.failed_ids.len();
            progress.indexed_documents = stats.indexed_documents;
            emit_progress(&progress);
            continue;
        }

        let metadata = prepare_reprocess_metadata(
            &doc.metadata,
            &doc.source_record_id,
            &doc.source_text_hash,
            doc.collapsed_records,
            slice_mode,
            &source_label,
            preprocess,
        );

        if existing.is_some() {
            stats.replaced_documents += 1;
        }

        match rag
            .memory_upsert(&namespace, doc.canonical_id.clone(), text, metadata)
            .await
        {
            Ok(()) => {
                stats.indexed_documents += 1;
            }
            Err(_) => {
                stats.failed_ids.push(doc.canonical_id.clone());
            }
        }

        progress.processed_documents = idx + 1;
        progress.indexed_documents = stats.indexed_documents;
        progress.skipped_documents = stats.skipped_existing_documents
            + stats.skipped_empty_documents
            + stats.skipped_preprocess_short_documents;
        progress.failed_documents = stats.failed_ids.len();
        emit_progress(&progress);
    }

    let _ = source_records;
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapse_export_records_prefers_core_slice_for_same_original_id() {
        let records = vec![
            ExportRecord {
                id: "outer-1".to_string(),
                text: "Short summary".to_string(),
                metadata: json!({
                    "original_id": "doc-1",
                    "layer": "outer",
                    "project": "vista"
                }),
                content_hash: Some("outer-hash".to_string()),
                embeddings: None,
            },
            ExportRecord {
                id: "core-1".to_string(),
                text: "Longer full document content that should win during reprocess collapse."
                    .to_string(),
                metadata: json!({
                    "original_id": "doc-1",
                    "layer": "core",
                    "project": "vista"
                }),
                content_hash: Some("core-hash".to_string()),
                embeddings: None,
            },
        ];

        let docs = collapse_export_records(records);
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].canonical_id, "doc-1");
        assert_eq!(docs[0].source_record_id, "core-1");
        assert_eq!(docs[0].collapsed_records, 2);
        assert!(docs[0].text.contains("full document content"));
    }

    #[test]
    fn collapse_export_records_falls_back_to_doc_id_then_record_id() {
        let records = vec![
            ExportRecord {
                id: "record-a".to_string(),
                text: "alpha".to_string(),
                metadata: json!({"doc_id": "doc-a"}),
                content_hash: None,
                embeddings: None,
            },
            ExportRecord {
                id: "record-b".to_string(),
                text: "beta".to_string(),
                metadata: json!({}),
                content_hash: None,
                embeddings: None,
            },
        ];

        let docs = collapse_export_records(records);
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].canonical_id, "doc-a");
        assert_eq!(docs[1].canonical_id, "record-b");
    }

    #[test]
    fn prepare_reprocess_metadata_replaces_stale_chunk_fields() {
        let metadata = json!({
            "original_id": "old-doc",
            "layer": "outer",
            "slice_mode": "onion",
            "content_hash": "stale",
            "project": "vista"
        });

        let prepared = prepare_reprocess_metadata(
            &metadata,
            "core-1",
            "fresh-hash",
            4,
            SliceMode::OnionFast,
            "legacy.jsonl",
            true,
        );

        assert_eq!(prepared["project"], "vista");
        assert_eq!(prepared["slice_mode"], "onion-fast");
        assert_eq!(prepared["reprocess_source_record_id"], "core-1");
        assert_eq!(prepared["reprocess_source_hash"], "fresh-hash");
        assert_eq!(prepared["reprocess_collapsed_records"], 4);
        assert_eq!(prepared["reprocess_source"], "legacy.jsonl");
        assert_eq!(prepared["reprocess_preprocessed"], true);
        assert!(prepared.get("layer").is_none());
        assert!(prepared.get("original_id").is_none());
        assert!(prepared.get("content_hash").is_none());
    }

    #[test]
    fn default_reindexed_namespace_appends_suffix() {
        assert_eq!(
            default_reindexed_namespace("kodowanie"),
            "kodowanie-reindexed"
        );
    }
}
