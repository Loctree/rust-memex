use anyhow::{Result, anyhow};
use memex_contracts::progress::{CompactProgress, MergeProgress, RepairResult};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::{
    BM25Config, BM25Index, CrossStoreRecoveryReport, GcConfig, GcStats, StorageManager, TableStats,
    inspect_cross_store_recovery, path_utils, repair_cross_store_recovery,
};

#[derive(Debug, Clone)]
pub struct MergeExecution {
    pub progress: MergeProgress,
    pub target_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct RepairExecution {
    pub result: RepairResult,
    pub report: CrossStoreRecoveryReport,
}

#[derive(Debug, Clone)]
pub struct MaintenanceExecution {
    pub progress: CompactProgress,
    pub pre_stats: Option<TableStats>,
    pub post_stats: Option<TableStats>,
    pub gc_stats: Option<GcStats>,
}

pub async fn merge_databases(
    source_paths: Vec<PathBuf>,
    target_path: PathBuf,
    dedup: bool,
    namespace_prefix: Option<String>,
    dry_run: bool,
) -> Result<MergeExecution> {
    let mut validated_sources = Vec::new();
    let mut progress = MergeProgress::default();
    let mut namespaces = HashSet::new();

    for source in &source_paths {
        let source_str = source.to_str().unwrap_or("");
        match path_utils::sanitize_existing_path(source_str) {
            Ok(validated) => validated_sources.push(validated),
            Err(_) => progress.errors += 1,
        }
    }

    if validated_sources.is_empty() {
        return Err(anyhow!("No valid source databases found"));
    }

    let validated_target = path_utils::sanitize_new_path(target_path.to_str().unwrap_or(""))?;
    let target_storage = if dry_run {
        None
    } else {
        if let Some(parent) = validated_target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Some(StorageManager::new_lance_only(validated_target.to_str().unwrap_or("")).await?)
    };

    let mut seen_hashes = HashSet::new();
    if dedup
        && let Some(ref target) = target_storage
        && let Ok(existing_docs) = target.all_documents(None, 100_000).await
    {
        for doc in existing_docs {
            if let Some(hash) = doc.content_hash {
                seen_hashes.insert(hash);
            }
        }
    }

    for source_path in &validated_sources {
        let source_storage =
            match StorageManager::new_lance_only(source_path.to_str().unwrap_or("")).await {
                Ok(storage) => storage,
                Err(_) => {
                    progress.errors += 1;
                    continue;
                }
            };

        let source_docs = match source_storage.all_documents(None, 100_000).await {
            Ok(docs) => docs,
            Err(_) => {
                progress.errors += 1;
                continue;
            }
        };

        progress.total_docs += source_docs.len();

        let mut docs_by_namespace = std::collections::HashMap::new();
        for doc in source_docs {
            docs_by_namespace
                .entry(doc.namespace.clone())
                .or_insert_with(Vec::new)
                .push(doc);
        }

        for (namespace, docs) in docs_by_namespace {
            let target_namespace = if let Some(ref prefix) = namespace_prefix {
                format!("{prefix}{namespace}")
            } else {
                namespace
            };
            namespaces.insert(target_namespace.clone());

            let mut batch = Vec::new();
            for doc in docs {
                if dedup && let Some(ref hash) = doc.content_hash {
                    if seen_hashes.contains(hash) {
                        progress.docs_skipped += 1;
                        continue;
                    }
                    seen_hashes.insert(hash.clone());
                }

                batch.push(crate::ChromaDocument {
                    id: doc.id,
                    namespace: target_namespace.clone(),
                    embedding: doc.embedding,
                    metadata: doc.metadata,
                    document: doc.document,
                    layer: doc.layer,
                    parent_id: doc.parent_id,
                    children_ids: doc.children_ids,
                    keywords: doc.keywords,
                    content_hash: doc.content_hash,
                });
                progress.docs_copied += 1;
            }

            if !dry_run
                && !batch.is_empty()
                && let Some(ref target) = target_storage
                && target.add_to_store(batch).await.is_err()
            {
                progress.errors += 1;
            }
        }

        progress.sources_processed += 1;
    }

    let mut namespaces = namespaces.into_iter().collect::<Vec<_>>();
    namespaces.sort();
    progress.namespaces = namespaces;

    Ok(MergeExecution {
        progress,
        target_path: validated_target,
    })
}

pub async fn repair_writes(
    db_path: &str,
    namespace: Option<&str>,
    execute: bool,
) -> Result<RepairExecution> {
    let storage = StorageManager::new_lance_only(db_path).await?;
    let mut bm25_config = BM25Config::default().with_read_only(!execute);
    if let Some(path) = sibling_bm25_path(db_path) {
        bm25_config = bm25_config.with_path(path.to_string_lossy().into_owned());
    }
    let bm25 = BM25Index::new(&bm25_config)?;

    let report = if execute {
        repair_cross_store_recovery(&storage, &bm25, namespace).await?
    } else {
        inspect_cross_store_recovery(&storage, &bm25, namespace).await?
    };

    Ok(RepairExecution {
        result: RepairResult {
            recovery_dir: report.recovery_dir.clone(),
            pending_batches: report.pending_batches,
            repaired_documents: report.repaired_documents,
            skipped_documents: report.skipped_documents,
            batches_repaired: report.batches_repaired,
        },
        report,
    })
}

pub async fn compact_database(storage: &StorageManager) -> Result<MaintenanceExecution> {
    let started_at = Instant::now();
    let pre_stats = storage.stats().await.ok();
    let stats = storage.compact().await?;
    let post_stats = storage.stats().await.ok();

    Ok(MaintenanceExecution {
        progress: CompactProgress {
            phase: "compact".to_string(),
            status: "complete".to_string(),
            description: Some("Merging small files into larger ones".to_string()),
            files_removed: stats
                .compaction
                .as_ref()
                .map(|value| value.files_removed as u64),
            files_added: stats
                .compaction
                .as_ref()
                .map(|value| value.files_added as u64),
            fragments_removed: stats
                .compaction
                .as_ref()
                .map(|value| value.fragments_removed as u64),
            fragments_added: stats
                .compaction
                .as_ref()
                .map(|value| value.fragments_added as u64),
            old_versions: None,
            bytes_removed: None,
            elapsed_ms: Some(started_at.elapsed().as_millis() as u64),
        },
        pre_stats,
        post_stats,
        gc_stats: None,
    })
}

pub async fn cleanup_versions(
    storage: &StorageManager,
    older_than_days: Option<u64>,
) -> Result<MaintenanceExecution> {
    let started_at = Instant::now();
    let pre_stats = storage.stats().await.ok();
    let stats = storage.cleanup(older_than_days).await?;
    let post_stats = storage.stats().await.ok();
    let older_than_days = older_than_days.unwrap_or(7);

    Ok(MaintenanceExecution {
        progress: CompactProgress {
            phase: "cleanup".to_string(),
            status: "complete".to_string(),
            description: Some(format!(
                "Removing old versions older than {older_than_days} days"
            )),
            files_removed: None,
            files_added: None,
            fragments_removed: None,
            fragments_added: None,
            old_versions: stats.prune.as_ref().map(|value| value.old_versions),
            bytes_removed: stats.prune.as_ref().map(|value| value.bytes_removed),
            elapsed_ms: Some(started_at.elapsed().as_millis() as u64),
        },
        pre_stats,
        post_stats,
        gc_stats: None,
    })
}

pub async fn collect_garbage(
    storage: &StorageManager,
    config: &GcConfig,
) -> Result<MaintenanceExecution> {
    let started_at = Instant::now();
    let pre_stats = storage.stats().await.ok();
    let gc_stats = storage.garbage_collect(config).await?;
    let post_stats = storage.stats().await.ok();

    Ok(MaintenanceExecution {
        progress: CompactProgress {
            phase: "gc".to_string(),
            status: "complete".to_string(),
            description: Some(gc_description(config)),
            files_removed: None,
            files_added: None,
            fragments_removed: None,
            fragments_added: None,
            old_versions: None,
            bytes_removed: gc_stats.bytes_freed,
            elapsed_ms: Some(started_at.elapsed().as_millis() as u64),
        },
        pre_stats,
        post_stats,
        gc_stats: Some(gc_stats),
    })
}

fn gc_description(config: &GcConfig) -> String {
    let mut actions = Vec::new();
    if config.remove_orphans {
        actions.push("orphan embeddings".to_string());
    }
    if config.remove_empty {
        actions.push("empty namespaces".to_string());
    }
    if let Some(duration) = config.older_than.as_ref() {
        actions.push(format!("documents older than {} days", duration.num_days()));
    }
    if actions.is_empty() {
        "Running garbage collection".to_string()
    } else {
        format!("Removing {}", actions.join(", "))
    }
}

pub fn sibling_bm25_path(db_path: &str) -> Option<PathBuf> {
    let db_path = shellexpand::tilde(db_path).to_string();
    Path::new(&db_path)
        .parent()
        .map(|parent| parent.join(".bm25"))
}
