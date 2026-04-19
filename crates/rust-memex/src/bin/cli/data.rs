use anyhow::Result;
use futures::{StreamExt, pin_mut};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

pub use rust_memex::contracts::audit::{
    AuditRecommendation, AuditResult as NamespaceAuditResult, ChunkQuality, QualityTier,
};
use rust_memex::{
    EmbeddingClient, EmbeddingConfig, RAGPipeline, ReindexJob, ReprocessJob, SliceMode,
    StorageManager, export_namespace_jsonl_stream, import_jsonl_file, reindex_namespace,
    reprocess_jsonl_file,
};

pub struct ReprocessConfig {
    pub namespace: String,
    pub input: PathBuf,
    pub slice_mode: SliceMode,
    pub preprocess: bool,
    pub skip_existing: bool,
    pub dry_run: bool,
    pub db_path: String,
}

pub struct ReindexConfig {
    pub source_namespace: String,
    pub target_namespace: String,
    pub slice_mode: SliceMode,
    pub preprocess: bool,
    pub skip_existing: bool,
    pub dry_run: bool,
    pub db_path: String,
}

fn slice_mode_name(slice_mode: SliceMode) -> &'static str {
    match slice_mode {
        SliceMode::Onion => "onion",
        SliceMode::OnionFast => "onion-fast",
        SliceMode::Flat => "flat",
    }
}

/// Export a namespace to JSONL file for portable backup
pub async fn run_export(
    namespace: String,
    output: Option<PathBuf>,
    include_embeddings: bool,
    db_path: String,
) -> Result<()> {
    let storage = Arc::new(StorageManager::new_lance_only(&db_path).await?);
    let stream = export_namespace_jsonl_stream(storage, namespace.clone(), include_embeddings);
    pin_mut!(stream);

    let mut exported_count = 0usize;
    match output {
        Some(path) => {
            use tokio::io::AsyncWriteExt;

            let mut file = tokio::fs::File::create(&path).await?;
            while let Some(line) = stream.next().await {
                let line = line?;
                file.write_all(line.as_bytes()).await?;
                exported_count += 1;
            }
            file.flush().await?;
            eprintln!(
                "Exported {} documents from '{}' to {:?}",
                exported_count, namespace, path
            );
        }
        None => {
            use tokio::io::AsyncWriteExt;

            let mut stdout = tokio::io::stdout();
            while let Some(line) = stream.next().await {
                let line = line?;
                stdout.write_all(line.as_bytes()).await?;
                exported_count += 1;
            }
            stdout.flush().await?;
        }
    }

    if include_embeddings && exported_count > 0 {
        eprintln!("  (embeddings included - file may be large)");
    }
    Ok(())
}

/// Import documents from JSONL file into a namespace
pub async fn run_import(
    namespace: String,
    input: PathBuf,
    skip_existing: bool,
    db_path: String,
    embedding_config: &EmbeddingConfig,
) -> Result<()> {
    let storage = Arc::new(StorageManager::new_lance_only(&db_path).await?);
    let embedding_client = Arc::new(Mutex::new(EmbeddingClient::new(embedding_config).await?));
    let rag = Arc::new(RAGPipeline::new(embedding_client, storage).await?);
    let outcome = import_jsonl_file(rag, namespace.clone(), &input, skip_existing).await?;

    eprintln!();
    eprintln!("Import complete:");
    eprintln!("  Imported: {} documents", outcome.imported_count);
    if outcome.skipped_count > 0 {
        eprintln!("  Skipped:  {} (already exist)", outcome.skipped_count);
    }
    if outcome.error_count > 0 {
        eprintln!("  Errors:   {}", outcome.error_count);
    }

    Ok(())
}

/// Rebuild exported JSONL documents into a fresh namespace with current chunking.
pub async fn run_reprocess(
    config: ReprocessConfig,
    embedding_config: &EmbeddingConfig,
) -> Result<()> {
    let ReprocessConfig {
        namespace,
        input,
        slice_mode,
        preprocess,
        skip_existing,
        dry_run,
        db_path,
    } = config;

    let storage = Arc::new(StorageManager::new_lance_only(&db_path).await?);
    let embedding_client = Arc::new(Mutex::new(EmbeddingClient::new(embedding_config).await?));
    let rag = Arc::new(RAGPipeline::new(embedding_client, storage).await?);

    let outcome = reprocess_jsonl_file(
        rag,
        ReprocessJob {
            input_path: input.clone(),
            target_namespace: namespace.clone(),
            slice_mode,
            preprocess,
            skip_existing,
            dry_run,
        },
        |_| {},
    )
    .await?;

    let collapsed_records = outcome
        .source_records
        .saturating_sub(outcome.canonical_documents);
    eprintln!(
        "Reprocessing {} source records into {} canonical documents for namespace '{}'...",
        outcome.source_records, outcome.canonical_documents, namespace
    );
    eprintln!("  Source:       {}", outcome.source_label);
    eprintln!("  Slice mode:   {}", slice_mode_name(slice_mode));
    eprintln!(
        "  Preprocess:   {}",
        if preprocess { "enabled" } else { "disabled" }
    );
    eprintln!(
        "  Collapsed:    {} duplicate slice records",
        collapsed_records
    );
    if dry_run {
        eprintln!();
        eprintln!("Dry run only: no documents were written.");
    }
    eprintln!();
    eprintln!("Reprocess complete:");
    eprintln!("  Indexed:         {}", outcome.indexed_documents);
    if outcome.replaced_documents > 0 {
        eprintln!("  Replaced:        {}", outcome.replaced_documents);
    }
    if outcome.skipped_existing_documents > 0 {
        eprintln!("  Skipped existing: {}", outcome.skipped_existing_documents);
    }
    if outcome.skipped_empty_documents > 0 {
        eprintln!("  Skipped empty:   {}", outcome.skipped_empty_documents);
    }
    if outcome.skipped_preprocess_short_documents > 0 {
        eprintln!(
            "  Skipped too short: {}",
            outcome.skipped_preprocess_short_documents
        );
    }
    if !outcome.failed_ids.is_empty() {
        eprintln!(
            "  FAILED:          {} (IDs: {})",
            outcome.failed_ids.len(),
            if outcome.failed_ids.len() <= 10 {
                outcome.failed_ids.join(", ")
            } else {
                format!(
                    "{}... and {} more",
                    outcome.failed_ids[..10].join(", "),
                    outcome.failed_ids.len() - 10
                )
            }
        );
    }
    if outcome.parse_errors > 0 {
        eprintln!("  Parse errors:    {}", outcome.parse_errors);
    }
    Ok(())
}

pub async fn run_reindex(config: ReindexConfig, embedding_config: &EmbeddingConfig) -> Result<()> {
    let ReindexConfig {
        source_namespace,
        target_namespace,
        slice_mode,
        preprocess,
        skip_existing,
        dry_run,
        db_path,
    } = config;

    let storage = Arc::new(StorageManager::new_lance_only(&db_path).await?);
    let embedding_client = Arc::new(Mutex::new(EmbeddingClient::new(embedding_config).await?));
    let rag = Arc::new(RAGPipeline::new(embedding_client, storage).await?);

    let outcome = reindex_namespace(
        rag,
        ReindexJob {
            source_namespace: source_namespace.clone(),
            target_namespace: target_namespace.clone(),
            slice_mode,
            preprocess,
            skip_existing,
            dry_run,
        },
        |_| {},
    )
    .await?;

    if dry_run {
        eprintln!("Dry run only: no documents were written.");
    }
    eprintln!(
        "Reindexed {} source rows into {} canonical documents from '{}' to '{}'",
        outcome.source_records,
        outcome.canonical_documents,
        outcome.source_namespace,
        outcome.target_namespace
    );
    eprintln!("  Indexed:  {}", outcome.indexed_documents);
    if outcome.replaced_documents > 0 {
        eprintln!("  Replaced: {}", outcome.replaced_documents);
    }
    if outcome.skipped_documents > 0 {
        eprintln!("  Skipped:  {}", outcome.skipped_documents);
    }
    if !outcome.failed_ids.is_empty() {
        eprintln!("  Failed:   {}", outcome.failed_ids.len());
    }
    Ok(())
}

// =============================================================================
// AUDIT & PURGE QUALITY COMMANDS
// =============================================================================

/// Run audit on namespaces to check quality metrics
pub async fn run_audit(
    namespace: Option<String>,
    threshold: u8,
    verbose: bool,
    json: bool,
    db_path: String,
) -> Result<()> {
    use rust_memex::{IntegrityRecommendation, TextIntegrityMetrics};

    let storage = StorageManager::new_lance_only(&db_path).await?;

    // Get namespaces to audit (list_namespaces returns Vec<(String, usize)>)
    let namespaces: Vec<String> = if let Some(ns) = namespace {
        vec![ns]
    } else {
        storage
            .list_namespaces()
            .await?
            .into_iter()
            .map(|(name, _count)| name)
            .collect()
    };

    if namespaces.is_empty() {
        if json {
            println!(r#"{{"namespaces": [], "summary": {{"total": 0}}}}"#);
        } else {
            eprintln!("No namespaces found in database");
        }
        return Ok(());
    }

    let threshold_f32 = threshold as f32 / 100.0;
    let mut results: Vec<NamespaceAuditResult> = Vec::new();

    if !json {
        eprintln!(
            "Auditing {} namespace(s) with {}% quality threshold...\n",
            namespaces.len(),
            threshold
        );
    }

    for ns in &namespaces {
        // Get all documents in namespace
        let docs = storage.get_all_in_namespace(ns).await?;

        if docs.is_empty() {
            results.push(NamespaceAuditResult {
                namespace: ns.clone(),
                document_count: 0,
                avg_chunk_length: 0,
                sentence_integrity: 0.0,
                word_integrity: 0.0,
                chunk_quality: 0.0,
                overall_score: 0.0,
                recommendation: AuditRecommendation::Empty,
                passes_threshold: false,
            });
            continue;
        }

        // Extract text from documents and compute metrics (ChromaDocument has `document` field)
        let chunks: Vec<String> = docs.iter().map(|d| d.document.clone()).collect();
        let combined_text = chunks.join(" ");

        let metrics = TextIntegrityMetrics::compute(&combined_text, &chunks);
        let passes = metrics.overall >= threshold_f32;

        let recommendation = match metrics.recommendation() {
            IntegrityRecommendation::Excellent => AuditRecommendation::Excellent,
            IntegrityRecommendation::Good => AuditRecommendation::Good,
            IntegrityRecommendation::Warn => AuditRecommendation::Warn,
            IntegrityRecommendation::Purge => AuditRecommendation::Purge,
        };

        results.push(NamespaceAuditResult {
            namespace: ns.clone(),
            document_count: docs.len(),
            avg_chunk_length: metrics.avg_chunk_length,
            sentence_integrity: metrics.sentence_integrity,
            word_integrity: metrics.word_integrity,
            chunk_quality: metrics.chunk_quality,
            overall_score: metrics.overall,
            recommendation,
            passes_threshold: passes,
        });

        if verbose && !json {
            eprintln!("Namespace: {}", ns);
            eprintln!("  Documents: {}", docs.len());
            eprintln!("  {}", metrics);
            eprintln!();
        }
    }

    // Output results
    if json {
        let passing = results.iter().filter(|r| r.passes_threshold).count();
        let failing = results.len() - passing;

        let output = serde_json::json!({
            "namespaces": results,
            "summary": {
                "total": results.len(),
                "passing": passing,
                "failing": failing,
                "threshold": threshold
            }
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        // Human-readable output
        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║                    NAMESPACE QUALITY AUDIT                     ║");
        println!("╠════════════════════════════════════════════════════════════════╣");
        println!(
            "║ {:30} │ {:>6} │ {:>6} │ {:>8} ║",
            "Namespace", "Docs", "Score", "Status"
        );
        println!("╠════════════════════════════════════════════════════════════════╣");

        for result in &results {
            let status_icon = if result.passes_threshold {
                "✅"
            } else {
                "❌"
            };
            let ns_display = if result.namespace.len() > 28 {
                format!("{}...", &result.namespace[..25])
            } else {
                result.namespace.clone()
            };

            println!(
                "║ {:30} │ {:>6} │ {:>5.1}% │ {} {:>6} ║",
                ns_display,
                result.document_count,
                result.overall_score * 100.0,
                status_icon,
                result.recommendation
            );
        }

        println!("╚════════════════════════════════════════════════════════════════╝");

        let passing = results.iter().filter(|r| r.passes_threshold).count();
        let failing = results.len() - passing;

        println!();
        println!(
            "Summary: {} passing, {} failing (threshold: {}%)",
            passing, failing, threshold
        );

        if failing > 0 {
            println!();
            println!("Namespaces below threshold:");
            for result in results.iter().filter(|r| !r.passes_threshold) {
                println!(
                    "  - {} ({:.1}% quality, {} docs)",
                    result.namespace,
                    result.overall_score * 100.0,
                    result.document_count
                );
            }
            println!();
            println!(
                "Run 'rust-memex purge-quality --threshold {}' to remove low-quality namespaces",
                threshold
            );
        }
    }

    Ok(())
}

/// Purge namespaces below quality threshold
pub async fn run_purge_quality(
    threshold: u8,
    confirm: bool,
    json: bool,
    db_path: String,
) -> Result<()> {
    use rust_memex::TextIntegrityMetrics;

    let storage = StorageManager::new_lance_only(&db_path).await?;
    // list_namespaces returns Vec<(String, usize)>
    let namespace_list = storage.list_namespaces().await?;

    if namespace_list.is_empty() {
        if json {
            println!(r#"{{"purged": [], "dry_run": {}}}"#, !confirm);
        } else {
            eprintln!("No namespaces found in database");
        }
        return Ok(());
    }

    let threshold_f32 = threshold as f32 / 100.0;
    let mut to_purge: Vec<(String, f32, usize)> = Vec::new();

    if !json {
        eprintln!(
            "Analyzing {} namespace(s) with {}% quality threshold...\n",
            namespace_list.len(),
            threshold
        );
    }

    for (ns, _count) in &namespace_list {
        let docs = storage.get_all_in_namespace(ns).await?;

        if docs.is_empty() {
            to_purge.push((ns.clone(), 0.0, 0));
            continue;
        }

        // ChromaDocument has `document` field, not `text`
        let chunks: Vec<String> = docs.iter().map(|d| d.document.clone()).collect();
        let combined_text = chunks.join(" ");
        let metrics = TextIntegrityMetrics::compute(&combined_text, &chunks);

        if metrics.overall < threshold_f32 {
            to_purge.push((ns.clone(), metrics.overall, docs.len()));
        }
    }

    if to_purge.is_empty() {
        if json {
            println!(r#"{{"purged": [], "message": "All namespaces pass quality threshold"}}"#);
        } else {
            println!(
                "All namespaces pass the {}% quality threshold. Nothing to purge.",
                threshold
            );
        }
        return Ok(());
    }

    if json {
        let output = serde_json::json!({
            "dry_run": !confirm,
            "threshold": threshold,
            "to_purge": to_purge.iter().map(|(ns, score, count)| {
                serde_json::json!({
                    "namespace": ns,
                    "quality_score": score,
                    "document_count": count
                })
            }).collect::<Vec<_>>()
        });
        println!("{}", serde_json::to_string_pretty(&output)?);

        if !confirm {
            return Ok(());
        }
    } else {
        println!(
            "Found {} namespace(s) below {}% quality threshold:",
            to_purge.len(),
            threshold
        );
        for (ns, score, count) in &to_purge {
            println!("  - {} ({:.1}% quality, {} docs)", ns, score * 100.0, count);
        }
        println!();

        if !confirm {
            println!("DRY RUN - No changes made.");
            println!("Run with --confirm to actually delete these namespaces.");
            return Ok(());
        }
    }

    // Actually purge if confirmed (use purge_namespace, not delete_namespace)
    let mut purged_count = 0;
    for (ns, _score, count) in &to_purge {
        if !json {
            eprint!("Purging '{}' ({} docs)... ", ns, count);
        }

        match storage.delete_namespace_documents(ns).await {
            Ok(_) => {
                purged_count += 1;
                if !json {
                    eprintln!("done");
                }
            }
            Err(e) => {
                if !json {
                    eprintln!("ERROR: {}", e);
                }
            }
        }
    }

    if !json {
        println!();
        println!(
            "Purged {} namespace(s) with quality below {}%",
            purged_count, threshold
        );
    }

    Ok(())
}
