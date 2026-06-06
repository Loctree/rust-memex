use std::collections::{BTreeMap, BTreeSet, HashMap};

use anyhow::Result;
use chrono::{DateTime, Duration, NaiveDate, NaiveDateTime, TimeZone, Utc};
use memex_contracts::{
    audit::{AuditRecommendation, AuditResult},
    stats::{DatabaseStats, NamespaceStats},
    timeline::TimelineEntry,
};
use serde::{Deserialize, Serialize};

use crate::storage::ChromaDocument;
use crate::{IntegrityRecommendation, SliceLayer, StorageManager, TextIntegrityMetrics};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeepStrategy {
    /// Keep the document with the earliest ID (lexicographic).
    Oldest,
    /// Keep the document with the latest ID (lexicographic).
    Newest,
    /// Keep the first document returned from storage iteration.
    HighestScore,
}

impl From<&str> for KeepStrategy {
    fn from(value: &str) -> Self {
        match value {
            "newest" => Self::Newest,
            "highest-score" => Self::HighestScore,
            _ => Self::Oldest,
        }
    }
}

/// How to group chunks when looking for duplicates.
///
/// After the v4 schema (`source_hash` + per-chunk `content_hash`) the legacy
/// `content_hash`-only grouping is broken for onion namespaces: every onion
/// layer of the same source has a unique chunk hash, so naive grouping reports
/// zero duplicates. Spec `2026-04-27_kb-transcripts-onion-slicer-fix-spec.md`,
/// P4 fixes this by making `(source_hash, layer)` the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DedupGroupBy {
    /// `(source_hash, layer)` — preserves onion structure: keeps one chunk
    /// per layer per source, removing only true repeats (e.g. `__dupe__` +
    /// `__clean__` variants of the same file). This is the post-v4 default.
    #[default]
    SourceHashLayer,
    /// `source_hash` alone — collapses all layers of one source into a single
    /// group. Useful when callers already plan to re-slice and only want to
    /// deduplicate at the source-document level.
    SourceHash,
    /// `content_hash` (per-chunk text SHA256) — legacy v3 grouping. After P0
    /// every chunk is unique, so this finds duplicates only when two chunk
    /// texts are byte-identical. Kept as opt-in for force-reindex edge cases.
    ContentHash,
}

impl DedupGroupBy {
    /// Parse the CLI / HTTP string form. Unknown values fall through to the
    /// default (`source-hash-layer`) so older clients keep working safely.
    pub fn parse(value: &str) -> Self {
        match value {
            "content-hash" | "content_hash" => Self::ContentHash,
            "source-hash" | "source_hash" => Self::SourceHash,
            _ => Self::SourceHashLayer,
        }
    }

    /// Stable string label for logging / CLI display.
    pub fn label(self) -> &'static str {
        match self {
            Self::SourceHashLayer => "source-hash-layer",
            Self::SourceHash => "source-hash",
            Self::ContentHash => "content-hash",
        }
    }
}

impl From<&str> for DedupGroupBy {
    fn from(value: &str) -> Self {
        Self::parse(value)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DedupDuplicate {
    pub id: String,
    pub namespace: String,
}

/// One duplicate cluster.
///
/// `content_hash` keeps its legacy field name for wire-compat with older
/// callers. Its semantic is now "the value of the grouping key" — for the
/// post-v4 default this is `<source_hash>:layer<N>`, for `SourceHash` it is
/// the source hash alone, and for the legacy `ContentHash` mode it is the
/// per-chunk text hash. The new `group_key` field carries the same value with
/// a clearer name; new clients should prefer it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DedupGroup {
    pub content_hash: String,
    #[serde(default)]
    pub group_key: String,
    pub kept_id: String,
    pub kept_namespace: String,
    pub removed: Vec<DedupDuplicate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DedupResult {
    pub total_docs: usize,
    pub unique_docs: usize,
    pub duplicate_groups: usize,
    pub duplicates_removed: usize,
    pub docs_without_hash: usize,
    /// Strategy used to bucket chunks into duplicate groups.
    /// Defaults to `SourceHashLayer` so legacy serialized payloads keep
    /// deserializing into the spec-default shape.
    #[serde(default)]
    pub group_by: DedupGroupBy,
    pub groups: Vec<DedupGroup>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PurgeQualityCandidate {
    pub namespace: String,
    pub quality_score: f32,
    pub document_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PurgeQualityResult {
    pub namespace_filter: Option<String>,
    pub threshold: u8,
    pub dry_run: bool,
    pub purged_namespaces: usize,
    pub candidates: Vec<PurgeQualityCandidate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineCoverage {
    pub earliest: Option<String>,
    pub latest: Option<String>,
    pub total_days: usize,
    pub days_with_data: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineReport {
    pub namespaces: Vec<String>,
    pub entries: Vec<TimelineEntry>,
    pub coverage: TimelineCoverage,
    pub gaps: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum TimelineBucket {
    #[default]
    Day,
    Hour,
}

impl TimelineBucket {
    pub fn parse(value: &str) -> Self {
        match value {
            "hour" => Self::Hour,
            _ => Self::Day,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TimelineQuery {
    pub namespace: Option<String>,
    pub since: Option<String>,
    pub until: Option<String>,
    pub bucket: TimelineBucket,
}

pub async fn audit_namespaces(
    storage: &StorageManager,
    namespace: Option<&str>,
    threshold: u8,
) -> Result<Vec<AuditResult>> {
    let namespaces: Vec<String> = if let Some(ns) = namespace {
        vec![ns.to_string()]
    } else {
        storage
            .list_namespaces()
            .await?
            .into_iter()
            .map(|(name, _count)| name)
            .collect()
    };

    let threshold_f32 = threshold as f32 / 100.0;
    let mut results = Vec::with_capacity(namespaces.len());

    for ns in namespaces {
        let docs = storage.get_all_in_namespace(&ns).await?;
        if docs.is_empty() {
            results.push(AuditResult {
                namespace: ns,
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

        let chunks: Vec<String> = docs.iter().map(|doc| doc.document.clone()).collect();
        let combined_text = chunks.join(" ");
        let metrics = TextIntegrityMetrics::compute(&combined_text, &chunks);

        results.push(AuditResult {
            namespace: ns,
            document_count: docs.len(),
            avg_chunk_length: metrics.avg_chunk_length,
            sentence_integrity: metrics.sentence_integrity,
            word_integrity: metrics.word_integrity,
            chunk_quality: metrics.chunk_quality,
            overall_score: metrics.overall,
            recommendation: integrity_recommendation(metrics.recommendation()),
            passes_threshold: metrics.overall >= threshold_f32,
        });
    }

    Ok(results)
}

pub async fn purge_quality_namespaces(
    storage: &StorageManager,
    namespace: Option<&str>,
    threshold: u8,
    dry_run: bool,
) -> Result<PurgeQualityResult> {
    let candidates = audit_namespaces(storage, namespace, threshold)
        .await?
        .into_iter()
        .filter(|result| !result.passes_threshold)
        .map(|result| PurgeQualityCandidate {
            namespace: result.namespace,
            quality_score: result.overall_score,
            document_count: result.document_count,
        })
        .collect::<Vec<_>>();

    let mut purged_namespaces = 0usize;
    if !dry_run {
        for candidate in &candidates {
            storage
                .delete_namespace_documents(&candidate.namespace)
                .await?;
            purged_namespaces += 1;
        }
    }

    Ok(PurgeQualityResult {
        namespace_filter: namespace.map(ToOwned::to_owned),
        threshold,
        dry_run,
        purged_namespaces,
        candidates,
    })
}

pub async fn deduplicate_documents(
    storage: &StorageManager,
    namespace: Option<&str>,
    dry_run: bool,
    keep_strategy: KeepStrategy,
    cross_namespace: bool,
    group_by: DedupGroupBy,
) -> Result<DedupResult> {
    let all_docs = storage.all_documents(namespace, 1_000_000).await?;

    let mut hash_groups: HashMap<String, Vec<_>> = HashMap::new();
    let mut docs_without_hash = 0usize;

    for doc in &all_docs {
        // Build the bucket key per requested strategy. A doc is "without hash"
        // if the strategy's required field is empty for this row — the caller
        // can then decide whether to backfill or fall back.
        let raw_key: Option<String> = match group_by {
            DedupGroupBy::ContentHash => doc
                .content_hash
                .as_deref()
                .filter(|hash| !hash.is_empty())
                .map(ToOwned::to_owned),
            DedupGroupBy::SourceHash => doc
                .source_hash
                .as_deref()
                .filter(|hash| !hash.is_empty())
                .map(ToOwned::to_owned),
            DedupGroupBy::SourceHashLayer => doc
                .source_hash
                .as_deref()
                .filter(|hash| !hash.is_empty())
                .map(|hash| format!("{}|layer{}", hash, doc.layer)),
        };

        let Some(key) = raw_key else {
            docs_without_hash += 1;
            continue;
        };

        let scoped_key = if cross_namespace {
            key
        } else {
            format!("{}:{}", doc.namespace, key)
        };
        hash_groups.entry(scoped_key).or_default().push(doc);
    }

    let mut result = DedupResult {
        total_docs: all_docs.len(),
        unique_docs: 0,
        duplicate_groups: 0,
        duplicates_removed: 0,
        docs_without_hash,
        group_by,
        groups: Vec::new(),
    };

    for (key, mut docs) in hash_groups {
        if docs.len() == 1 {
            result.unique_docs += 1;
            continue;
        }

        match keep_strategy {
            KeepStrategy::Oldest => docs.sort_by(|left, right| left.id.cmp(&right.id)),
            KeepStrategy::Newest => docs.sort_by(|left, right| right.id.cmp(&left.id)),
            KeepStrategy::HighestScore => {}
        }

        let kept = docs[0];
        let removed_docs = docs.into_iter().skip(1).collect::<Vec<_>>();

        if !dry_run {
            for doc in &removed_docs {
                storage.delete_document(&doc.namespace, &doc.id).await?;
            }
        }

        // Strip the namespace prefix when reporting the group key so consumers
        // see the strategy-native value (e.g. `<source_hash>|layer3`).
        let display_key = if cross_namespace {
            key.clone()
        } else {
            key.split_once(':')
                .map(|(_ns, rest)| rest.to_string())
                .unwrap_or_else(|| key.clone())
        };

        result.unique_docs += 1;
        result.duplicate_groups += 1;
        result.duplicates_removed += removed_docs.len();
        result.groups.push(DedupGroup {
            content_hash: display_key.clone(),
            group_key: display_key,
            kept_id: kept.id.clone(),
            kept_namespace: kept.namespace.clone(),
            removed: removed_docs
                .iter()
                .map(|doc| DedupDuplicate {
                    id: doc.id.clone(),
                    namespace: doc.namespace.clone(),
                })
                .collect(),
        });
    }

    Ok(result)
}

/// Result of a backfill-hashes pass.
///
/// `content_hash` is per-chunk SHA256 (semantics introduced in schema v4).
/// `source_hash` is the SHA256 of the source document text — same value across
/// all four onion layers from one source. Pre-v4 namespaces stored the source
/// hash in `content_hash`; this backfill (a) re-derives a true per-chunk
/// `content_hash`, and (b) preserves the legacy hash as `source_hash` so
/// pre-index dedup works without re-reading source files.
///
/// Spec: `2026-04-27_kb-transcripts-onion-slicer-fix-spec.md`, P0 backfill.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BackfillHashesResult {
    /// Total documents inspected in the namespace (or across all namespaces).
    pub total_docs: usize,
    /// Documents that needed a `content_hash` write because the column was
    /// either empty or stored the source hash (pre-v4 schema).
    pub content_hash_backfilled: usize,
    /// Documents that needed a `source_hash` write because the column was
    /// empty (pre-v4 schema, or v4 chunk written before the field was wired
    /// up end-to-end).
    pub source_hash_backfilled: usize,
    /// Documents skipped because they already have correct per-chunk
    /// `content_hash` AND a populated `source_hash`.
    pub already_consistent: usize,
    /// Documents that could not be backfilled because the embedding column
    /// was missing or zero-length (extremely unlikely; logged in storage warn).
    pub skipped_no_embedding: usize,
    /// `true` when the caller asked for a dry run (no writes performed).
    pub dry_run: bool,
}

/// Recompute per-chunk `content_hash` and recover `source_hash` for legacy
/// chunks. Safe to run repeatedly; chunks that already match the v4 contract
/// are counted under `already_consistent` and left alone.
///
/// Strategy:
/// 1. For each chunk, recompute `chunk_hash = SHA256(document_text)`.
/// 2. If the stored `content_hash` differs from `chunk_hash`, the row is
///    pre-v4 — its `content_hash` was actually the source hash. Move it into
///    `source_hash` (when empty) before overwriting `content_hash`.
/// 3. If `source_hash` is still empty after step 2, fall back to copying the
///    new `content_hash` so dedup has *something* to key on (better than
///    nothing — operators can re-index from source for true provenance).
/// 4. Re-write the row by deleting + inserting (LanceDB has no per-row update
///    that accepts a fixed-size vector update for our schema).
///
fn fmt_duration(secs: f64) -> String {
    if secs > 3600.0 {
        format!("{:.0}h{:02.0}m", secs / 3600.0, (secs % 3600.0) / 60.0)
    } else if secs > 60.0 {
        format!("{:.0}m{:02.0}s", secs / 60.0, secs % 60.0)
    } else {
        format!("{:.0}s", secs)
    }
}

const SPINNER: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

fn emit_backfill_progress(
    processed: usize,
    total: usize,
    started: &std::time::Instant,
    last_report: &mut std::time::Instant,
) {
    if total == 0 || last_report.elapsed().as_secs() < 10 {
        return;
    }
    *last_report = std::time::Instant::now();
    let pct = (processed as f64 / total as f64 * 100.0).min(100.0);
    let elapsed = started.elapsed().as_secs_f64();
    let rate = processed as f64 / elapsed;
    let eta = if rate > 0.0 {
        (total - processed) as f64 / rate
    } else {
        0.0
    };
    let tick = (elapsed as usize / 10) % SPINNER.len();
    eprint!(
        "\r  {} [{:>6}/{:>6}] {:5.1}%  {:.0} docs/s  ETA {}   ",
        SPINNER[tick],
        processed,
        total,
        pct,
        rate,
        fmt_duration(eta)
    );
}

/// Spec: `2026-04-27_kb-transcripts-onion-slicer-fix-spec.md`, P0 backfill.
pub async fn backfill_chunk_and_source_hashes(
    storage: &StorageManager,
    namespace: Option<&str>,
    dry_run: bool,
) -> Result<BackfillHashesResult> {
    let mut result = BackfillHashesResult {
        dry_run,
        ..Default::default()
    };

    // Pre-count total documents for progress reporting.
    let total_count = match namespace {
        Some(ns) => storage.count_namespace(ns).await.unwrap_or(0),
        None => storage.stats().await.map(|s| s.row_count).unwrap_or(0),
    };

    const PAGE: usize = 5_000;
    let mut offset = 0;
    let mut processed = 0usize;
    let started = std::time::Instant::now();
    let mut last_report = started;

    loop {
        let page = storage.all_documents_page(namespace, offset, PAGE).await?;
        let page_len = page.len();
        if page_len == 0 {
            break;
        }
        result.total_docs += page_len;

        for doc in &page {
            processed += 1;

            if doc.embedding.is_empty() {
                result.skipped_no_embedding += 1;
                emit_backfill_progress(processed, total_count, &started, &mut last_report);
                continue;
            }

            let true_chunk_hash = crate::rag::compute_content_hash(&doc.document);
            let mut needs_content = false;
            let mut needs_source = false;

            let new_content_hash = match doc.content_hash.as_deref() {
                Some(stored) if stored == true_chunk_hash => stored.to_string(),
                Some(_) => {
                    needs_content = true;
                    true_chunk_hash.clone()
                }
                None => {
                    needs_content = true;
                    true_chunk_hash.clone()
                }
            };

            // Recover source_hash. Pre-v4 chunks stored the source hash under
            // `content_hash`; if that was the case, prefer it over the freshly
            // computed chunk hash. Otherwise fall back to chunk_hash so dedup
            // has a key (better than `None`).
            let recovered_source_hash = match (&doc.source_hash, &doc.content_hash) {
                (Some(s), _) if !s.is_empty() => s.clone(),
                (_, Some(legacy)) if legacy.as_str() != true_chunk_hash.as_str() => {
                    // Legacy content_hash was actually the source hash.
                    needs_source = true;
                    legacy.clone()
                }
                _ => {
                    needs_source = doc.source_hash.is_none();
                    new_content_hash.clone()
                }
            };

            if !needs_content && !needs_source {
                result.already_consistent += 1;
                emit_backfill_progress(processed, total_count, &started, &mut last_report);
                continue;
            }

            if needs_content {
                result.content_hash_backfilled += 1;
            }
            if needs_source {
                result.source_hash_backfilled += 1;
            }

            if dry_run {
                emit_backfill_progress(processed, total_count, &started, &mut last_report);
                continue;
            }

            // Atomic-per-row swap: delete then re-insert with corrected
            // hashes. LanceDB lacks an update-with-vector path for our schema.
            let new_doc = ChromaDocument {
                id: doc.id.clone(),
                namespace: doc.namespace.clone(),
                embedding: doc.embedding.clone(),
                metadata: doc.metadata.clone(),
                document: doc.document.clone(),
                layer: doc.layer,
                parent_id: doc.parent_id.clone(),
                children_ids: doc.children_ids.clone(),
                keywords: doc.keywords.clone(),
                content_hash: Some(new_content_hash.clone()),
                source_hash: Some(recovered_source_hash.clone()),
            };
            storage.delete_document(&doc.namespace, &doc.id).await?;
            storage.add_to_store(vec![new_doc]).await?;
            emit_backfill_progress(processed, total_count, &started, &mut last_report);
        }

        if page_len < PAGE {
            break;
        }
        offset += page_len;
    }

    if total_count > 0 && processed > 0 {
        eprintln!(
            "\r  [{0}/{0}] 100.0%  done in {1}                    ",
            processed,
            fmt_duration(started.elapsed().as_secs_f64())
        );
    }

    Ok(result)
}

pub async fn database_stats(storage: &StorageManager) -> Result<DatabaseStats> {
    match storage.stats().await {
        Ok(stats) => Ok(DatabaseStats {
            row_count: stats.row_count,
            version_count: stats.version_count,
            table_name: stats.table_name,
            db_path: stats.db_path,
        }),
        Err(_) => Ok(DatabaseStats {
            row_count: 0,
            version_count: 0,
            table_name: storage.get_collection_name().to_string(),
            db_path: storage.lance_path().to_string(),
        }),
    }
}

pub async fn namespace_stats(
    storage: &StorageManager,
    namespace: Option<&str>,
) -> Result<Vec<NamespaceStats>> {
    let all_docs = storage.all_documents(namespace, 100_000).await?;
    let mut by_namespace: HashMap<String, Vec<_>> = HashMap::new();
    for doc in &all_docs {
        by_namespace
            .entry(doc.namespace.clone())
            .or_default()
            .push(doc);
    }

    let mut stats_list = Vec::with_capacity(by_namespace.len());
    for (name, docs) in by_namespace {
        let total_chunks = docs.len();
        let mut layer_counts = HashMap::new();
        let mut keyword_counts = HashMap::new();
        let mut dates = Vec::new();

        for doc in docs {
            let layer_name = SliceLayer::from_u8(doc.layer)
                .map(|layer| layer.name().to_string())
                .unwrap_or_else(|| "flat".to_string());
            *layer_counts.entry(layer_name).or_insert(0) += 1;

            for keyword in &doc.keywords {
                *keyword_counts.entry(keyword.clone()).or_insert(0) += 1;
            }

            if let Some(timestamp) = extract_doc_timestamp_string(doc.metadata.as_object()) {
                dates.push(timestamp);
            }
        }

        let mut top_keywords = keyword_counts.into_iter().collect::<Vec<_>>();
        top_keywords.sort_by_key(|entry| std::cmp::Reverse(entry.1));
        top_keywords.truncate(10);
        dates.sort();

        stats_list.push(NamespaceStats {
            name,
            total_chunks,
            layer_counts,
            top_keywords,
            has_timestamps: !dates.is_empty(),
            earliest_indexed: dates.first().cloned(),
            latest_indexed: dates.last().cloned(),
        });
    }

    stats_list.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(stats_list)
}

pub async fn timeline_report(
    storage: &StorageManager,
    query: &TimelineQuery,
) -> Result<TimelineReport> {
    let namespaces: Vec<String> = if let Some(namespace) = query.namespace.as_deref() {
        vec![namespace.to_string()]
    } else {
        storage
            .list_namespaces()
            .await?
            .into_iter()
            .map(|(name, _count)| name)
            .collect()
    };

    let since = query.since.as_deref().and_then(parse_time_bound);
    let until = query.until.as_deref().and_then(parse_time_bound);

    let mut timeline: BTreeMap<String, BTreeMap<String, BTreeMap<String, usize>>> = BTreeMap::new();
    let mut all_dates = BTreeSet::new();

    for namespace in &namespaces {
        let docs = storage.get_all_in_namespace(namespace).await?;
        for doc in docs {
            let Some(timestamp) = extract_doc_timestamp(&doc) else {
                all_dates.insert("unknown".to_string());
                *timeline
                    .entry("unknown".to_string())
                    .or_default()
                    .entry(namespace.clone())
                    .or_default()
                    .entry("unknown".to_string())
                    .or_default() += 1;
                continue;
            };

            if since.is_some_and(|lower| timestamp < lower) {
                continue;
            }
            if until.is_some_and(|upper| timestamp > upper) {
                continue;
            }

            let bucket = match query.bucket {
                TimelineBucket::Day => timestamp.format("%Y-%m-%d").to_string(),
                TimelineBucket::Hour => timestamp.format("%Y-%m-%dT%H:00:00Z").to_string(),
            };
            let source = doc
                .metadata
                .get("source")
                .and_then(|value| value.as_str())
                .or_else(|| {
                    doc.metadata
                        .get("file_path")
                        .and_then(|value| value.as_str())
                })
                .map(filename_from_path)
                .unwrap_or_else(|| "unknown".to_string());

            all_dates.insert(bucket.clone());
            *timeline
                .entry(bucket)
                .or_default()
                .entry(namespace.clone())
                .or_default()
                .entry(source)
                .or_default() += 1;
        }
    }

    let entries = timeline
        .iter()
        .flat_map(|(date, namespace_map)| {
            namespace_map
                .iter()
                .flat_map(move |(namespace, source_map)| {
                    source_map
                        .iter()
                        .map(move |(source, chunk_count)| TimelineEntry {
                            date: date.clone(),
                            namespace: namespace.clone(),
                            source: Some(source.clone()),
                            chunk_count: *chunk_count,
                        })
                })
        })
        .collect::<Vec<_>>();

    let ordered_dates = all_dates
        .iter()
        .filter(|date| *date != "unknown")
        .collect::<Vec<_>>();
    let earliest = ordered_dates.first().map(|date| (*date).clone());
    let latest = ordered_dates.last().map(|date| (*date).clone());
    let gaps = compute_gaps(&ordered_dates, query.bucket);
    let total_days = match (ordered_dates.first(), ordered_dates.last()) {
        (Some(first), Some(last)) => {
            let first_date = timeline_gap_date(first, query.bucket);
            let last_date = timeline_gap_date(last, query.bucket);
            match (first_date, last_date) {
                (Some(first_date), Some(last_date)) => {
                    (last_date - first_date).num_days() as usize + 1
                }
                _ => 0,
            }
        }
        _ => 0,
    };

    Ok(TimelineReport {
        namespaces,
        entries,
        coverage: TimelineCoverage {
            earliest,
            latest,
            total_days,
            days_with_data: ordered_dates.len(),
        },
        gaps,
    })
}

fn integrity_recommendation(recommendation: IntegrityRecommendation) -> AuditRecommendation {
    match recommendation {
        IntegrityRecommendation::Excellent => AuditRecommendation::Excellent,
        IntegrityRecommendation::Good => AuditRecommendation::Good,
        IntegrityRecommendation::Warn => AuditRecommendation::Warn,
        IntegrityRecommendation::Purge => AuditRecommendation::Purge,
    }
}

fn extract_doc_timestamp(doc: &crate::ChromaDocument) -> Option<DateTime<Utc>> {
    doc.metadata
        .get("indexed_at")
        .and_then(|value| value.as_str())
        .or_else(|| {
            doc.metadata
                .get("timestamp")
                .and_then(|value| value.as_str())
        })
        .or_else(|| {
            doc.metadata
                .get("created_at")
                .and_then(|value| value.as_str())
        })
        .and_then(parse_iso_or_date)
        .or_else(|| {
            doc.metadata
                .get("path")
                .and_then(|value| value.as_str())
                .and_then(extract_date_from_path)
                .and_then(|d| parse_iso_or_date(&d))
        })
}

fn extract_doc_timestamp_string(
    metadata: Option<&serde_json::Map<String, serde_json::Value>>,
) -> Option<String> {
    metadata.and_then(|object| {
        object
            .iter()
            .find_map(|(key, value)| {
                if !(key.contains("date") || key.contains("timestamp") || key.contains("time")) {
                    return None;
                }
                value.as_str().map(ToOwned::to_owned)
            })
            .or_else(|| {
                object
                    .get("path")
                    .and_then(|value| value.as_str())
                    .and_then(extract_date_from_path)
            })
    })
}

fn extract_date_from_path(path: &str) -> Option<String> {
    use std::sync::OnceLock;

    // Look for YYYY-MM-DD or YYYY_MM_DD
    static RE_STANDARD: OnceLock<regex::Regex> = OnceLock::new();
    let re_standard = RE_STANDARD
        .get_or_init(|| regex::Regex::new(r"(19\d{2}|20\d{2})[_-](\d{2})[_-](\d{2})").unwrap());

    if let Some(caps) = re_standard.captures(path) {
        let year = caps.get(1)?.as_str();
        let month = caps.get(2)?.as_str();
        let day = caps.get(3)?.as_str();
        return Some(format!("{}-{}-{}", year, month, day));
    }

    // Also look for YYYY_MMDD or YYYYMMDD
    static RE_SHORT: OnceLock<regex::Regex> = OnceLock::new();
    let re_short =
        RE_SHORT.get_or_init(|| regex::Regex::new(r"(19\d{2}|20\d{2})_?(\d{2})(\d{2})").unwrap());

    if let Some(caps) = re_short.captures(path) {
        let year = caps.get(1)?.as_str();
        let month = caps.get(2)?.as_str();
        let day = caps.get(3)?.as_str();
        let month_num: u32 = month.parse().ok()?;
        let day_num: u32 = day.parse().ok()?;
        if (1..=12).contains(&month_num) && (1..=31).contains(&day_num) {
            return Some(format!("{}-{}-{}", year, month, day));
        }
    }

    None
}

fn parse_time_bound(input: &str) -> Option<DateTime<Utc>> {
    if let Some(days_str) = input.strip_suffix('d')
        && let Ok(days) = days_str.parse::<i64>()
    {
        return Some(Utc::now() - Duration::days(days));
    }

    if input.len() == 7
        && input.chars().nth(4) == Some('-')
        && let Ok(date) = NaiveDate::parse_from_str(&format!("{input}-01"), "%Y-%m-%d")
    {
        return date
            .and_hms_opt(0, 0, 0)
            .map(|dt| Utc.from_utc_datetime(&dt));
    }

    parse_iso_or_date(input)
}

fn parse_iso_or_date(input: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(input)
        .map(|dt| dt.with_timezone(&Utc))
        .ok()
        .or_else(|| {
            NaiveDateTime::parse_from_str(input, "%Y-%m-%dT%H:%M:%S")
                .ok()
                .map(|dt| Utc.from_utc_datetime(&dt))
        })
        .or_else(|| {
            NaiveDate::parse_from_str(input, "%Y-%m-%d")
                .ok()
                .and_then(|date| date.and_hms_opt(0, 0, 0))
                .map(|dt| Utc.from_utc_datetime(&dt))
        })
}

fn filename_from_path(path: &str) -> String {
    std::path::Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(path)
        .to_string()
}

fn compute_gaps(dates: &[&String], bucket: TimelineBucket) -> Vec<String> {
    let parsed_dates = dates
        .iter()
        .filter_map(|date| timeline_gap_date(date, bucket))
        .collect::<Vec<_>>();

    let mut gaps = Vec::new();
    for window in parsed_dates.windows(2) {
        let diff = window[1] - window[0];
        let missing_units = match bucket {
            TimelineBucket::Day => diff.num_days() - 1,
            TimelineBucket::Hour => diff.num_hours() - 1,
        };
        if missing_units > 0 {
            gaps.push(format!(
                "{} to {} ({} missing {})",
                format_gap_date(window[0], bucket),
                format_gap_date(window[1], bucket),
                missing_units,
                match bucket {
                    TimelineBucket::Day => "day(s)",
                    TimelineBucket::Hour => "hour(s)",
                }
            ));
        }
    }
    gaps
}

fn timeline_gap_date(date: &str, bucket: TimelineBucket) -> Option<DateTime<Utc>> {
    match bucket {
        TimelineBucket::Day => NaiveDate::parse_from_str(date, "%Y-%m-%d")
            .ok()
            .and_then(|date| date.and_hms_opt(0, 0, 0))
            .map(|dt| Utc.from_utc_datetime(&dt)),
        TimelineBucket::Hour => DateTime::parse_from_rfc3339(date)
            .map(|dt| dt.with_timezone(&Utc))
            .ok()
            .or_else(|| parse_iso_or_date(date)),
    }
}

fn format_gap_date(date: DateTime<Utc>, bucket: TimelineBucket) -> String {
    match bucket {
        TimelineBucket::Day => date.format("%Y-%m-%d").to_string(),
        TimelineBucket::Hour => date.format("%Y-%m-%dT%H:00:00Z").to_string(),
    }
}

#[cfg(test)]
mod backfill_tests {
    use super::*;
    use crate::rag::compute_content_hash;
    use crate::storage::ChromaDocument;
    use tempfile::TempDir;

    /// Pre-v4 row stored `content_hash = SHA256(source_doc)` for every layer
    /// of the same source. After backfill, `content_hash` must become
    /// SHA256(chunk_text) and `source_hash` must hold the legacy value so
    /// pre-index dedup keeps working.
    #[tokio::test]
    async fn backfill_promotes_legacy_content_hash_to_source_hash() {
        let tmp = TempDir::new().expect("temp dir");
        let db_path = tmp.path().join("lancedb");
        let storage = StorageManager::new_lance_only(db_path.to_str().expect("utf-8 temp db path"))
            .await
            .expect("storage");
        storage.ensure_collection().await.expect("collection");

        let namespace = "kb:transcripts-test".to_string();
        let source_text = "full source document text";
        let source_hash = compute_content_hash(source_text);

        // Two chunks from one source, written under the v3 contract:
        // both rows carry the source-text hash in `content_hash` and
        // leave `source_hash` empty.
        let chunk_a_text = "outer summary text";
        let chunk_b_text = "inner detailed text";
        let doc_a = ChromaDocument {
            id: "chunk-a".to_string(),
            namespace: namespace.clone(),
            embedding: vec![0.1_f32; 8],
            metadata: serde_json::json!({"path": "doc.md"}),
            document: chunk_a_text.to_string(),
            layer: 1,
            parent_id: None,
            children_ids: vec![],
            keywords: vec![],
            content_hash: Some(source_hash.clone()),
            source_hash: None,
        };
        let doc_b = ChromaDocument {
            id: "chunk-b".to_string(),
            namespace: namespace.clone(),
            embedding: vec![0.2_f32; 8],
            metadata: serde_json::json!({"path": "doc.md"}),
            document: chunk_b_text.to_string(),
            layer: 3,
            parent_id: None,
            children_ids: vec![],
            keywords: vec![],
            content_hash: Some(source_hash.clone()),
            source_hash: None,
        };
        storage
            .add_to_store(vec![doc_a, doc_b])
            .await
            .expect("seed pre-v4 rows");

        let dry = backfill_chunk_and_source_hashes(&storage, Some(&namespace), true)
            .await
            .expect("dry run");
        assert!(dry.dry_run);
        assert_eq!(dry.total_docs, 2);
        assert_eq!(dry.content_hash_backfilled, 2);
        assert_eq!(dry.source_hash_backfilled, 2);
        assert_eq!(dry.already_consistent, 0);

        let live = backfill_chunk_and_source_hashes(&storage, Some(&namespace), false)
            .await
            .expect("live run");
        assert!(!live.dry_run);
        assert_eq!(live.content_hash_backfilled, 2);
        assert_eq!(live.source_hash_backfilled, 2);

        let after_a = storage
            .get_document(&namespace, "chunk-a")
            .await
            .expect("lookup a")
            .expect("doc-a present");
        assert_eq!(
            after_a.content_hash.as_deref(),
            Some(compute_content_hash(chunk_a_text)).as_deref()
        );
        assert_eq!(after_a.source_hash.as_deref(), Some(source_hash.as_str()));

        let after_b = storage
            .get_document(&namespace, "chunk-b")
            .await
            .expect("lookup b")
            .expect("doc-b present");
        assert_eq!(
            after_b.content_hash.as_deref(),
            Some(compute_content_hash(chunk_b_text)).as_deref()
        );
        assert_eq!(after_b.source_hash.as_deref(), Some(source_hash.as_str()));

        // Idempotency: running again leaves the rows alone.
        let again = backfill_chunk_and_source_hashes(&storage, Some(&namespace), false)
            .await
            .expect("idempotent");
        assert_eq!(again.content_hash_backfilled, 0);
        assert_eq!(again.source_hash_backfilled, 0);
        assert_eq!(again.already_consistent, 2);
    }
}

#[cfg(test)]
mod dedup_grouping_tests {
    use super::*;
    use crate::rag::compute_content_hash;
    use crate::storage::ChromaDocument;
    use tempfile::TempDir;

    fn doc(
        id: &str,
        ns: &str,
        layer: u8,
        text: &str,
        source: &str,
        chunk_hash: &str,
    ) -> ChromaDocument {
        ChromaDocument {
            id: id.to_string(),
            namespace: ns.to_string(),
            embedding: vec![0.1_f32; 8],
            metadata: serde_json::json!({}),
            document: text.to_string(),
            layer,
            parent_id: None,
            children_ids: vec![],
            keywords: vec![],
            content_hash: Some(chunk_hash.to_string()),
            source_hash: Some(source.to_string()),
        }
    }

    /// Spec P4: post-v4 default (`source-hash-layer`) must keep the onion
    /// intact: with two `__dupe__` + `__clean__` variants of the same source,
    /// each layer collapses to a single survivor (4 dups removed for a 4-layer
    /// onion), while distinct sources remain untouched.
    #[tokio::test]
    async fn source_hash_layer_grouping_preserves_onion_structure() {
        let tmp = TempDir::new().expect("temp dir");
        let db_path = tmp.path().join("db");
        let storage = StorageManager::new_lance_only(db_path.to_str().expect("utf-8 temp db path"))
            .await
            .expect("storage");
        storage.ensure_collection().await.expect("collection");

        let ns = "kb:transcripts-test";
        let source_a = compute_content_hash("source document A — full transcript");
        let source_b = compute_content_hash("source document B — different transcript");

        // Source A indexed twice (`__dupe__` + `__clean__`): 4 layers × 2 = 8 chunks.
        // After source-hash-layer grouping: 4 groups of 2 → 4 duplicates removed.
        let mut docs = Vec::new();
        for (suffix, _variant) in [("clean", "clean"), ("dupe", "dupe")].iter() {
            for layer in 0u8..4 {
                let text = format!("source-A layer-{layer} variant-{suffix}");
                let chunk_hash = compute_content_hash(&text);
                docs.push(doc(
                    &format!("a-{suffix}-l{layer}"),
                    ns,
                    layer,
                    &text,
                    &source_a,
                    &chunk_hash,
                ));
            }
        }
        // Source B indexed once: 4 unique chunks, must survive.
        for layer in 0u8..4 {
            let text = format!("source-B layer-{layer}");
            let chunk_hash = compute_content_hash(&text);
            docs.push(doc(
                &format!("b-l{layer}"),
                ns,
                layer,
                &text,
                &source_b,
                &chunk_hash,
            ));
        }
        storage
            .add_to_store(docs)
            .await
            .expect("seed dedup fixture");

        let result = deduplicate_documents(
            &storage,
            Some(ns),
            true,
            KeepStrategy::Oldest,
            false,
            DedupGroupBy::SourceHashLayer,
        )
        .await
        .expect("dedup");

        assert_eq!(result.total_docs, 12);
        assert_eq!(result.duplicate_groups, 4, "one group per onion layer");
        assert_eq!(
            result.duplicates_removed, 4,
            "remove one variant per layer, keep the other"
        );
        assert_eq!(
            result.docs_without_hash, 0,
            "every chunk has source_hash populated"
        );
        // Source B contributes 4 singleton groups counted as unique_docs.
        assert_eq!(result.unique_docs, 4 + 4);
        // Group keys must contain the layer suffix so consumers can verify
        // per-layer onion preservation.
        assert!(
            result.groups.iter().all(|g| g.group_key.contains("|layer")),
            "source-hash-layer keys must encode layer: {:?}",
            result
                .groups
                .iter()
                .map(|g| &g.group_key)
                .collect::<Vec<_>>()
        );
    }

    /// Spec P4 invariant: post-P0 every chunk has a unique `content_hash`, so
    /// the legacy grouping must report zero duplicates on a freshly-indexed
    /// onion namespace. This test exists to lock that semantic in place — it
    /// is the symptom that motivated adding `--group-by`.
    #[tokio::test]
    async fn content_hash_grouping_finds_zero_duplicates_on_fresh_onion() {
        let tmp = TempDir::new().expect("temp dir");
        let db_path = tmp.path().join("db");
        let storage = StorageManager::new_lance_only(db_path.to_str().expect("utf-8 temp db path"))
            .await
            .expect("storage");
        storage.ensure_collection().await.expect("collection");

        let ns = "kb:transcripts-test";
        let source_a = compute_content_hash("source document A");
        let mut docs = Vec::new();
        for layer in 0u8..4 {
            // Distinct chunk text per layer → unique per-chunk content_hash.
            let text = format!("source-A unique-layer-{layer}");
            let chunk_hash = compute_content_hash(&text);
            docs.push(doc(
                &format!("a-l{layer}"),
                ns,
                layer,
                &text,
                &source_a,
                &chunk_hash,
            ));
        }
        storage
            .add_to_store(docs)
            .await
            .expect("seed unique chunks");

        let result = deduplicate_documents(
            &storage,
            Some(ns),
            true,
            KeepStrategy::Oldest,
            false,
            DedupGroupBy::ContentHash,
        )
        .await
        .expect("dedup");

        assert_eq!(result.total_docs, 4);
        assert_eq!(
            result.duplicate_groups, 0,
            "post-P0 each chunk has unique content_hash, legacy grouping must find none"
        );
        assert_eq!(result.duplicates_removed, 0);
    }

    #[test]
    fn dedup_group_by_parses_known_aliases_and_falls_back_to_default() {
        assert_eq!(
            DedupGroupBy::parse("source-hash-layer"),
            DedupGroupBy::SourceHashLayer
        );
        assert_eq!(DedupGroupBy::parse("source-hash"), DedupGroupBy::SourceHash);
        assert_eq!(
            DedupGroupBy::parse("source_hash"),
            DedupGroupBy::SourceHash,
            "underscore form accepted as alias"
        );
        assert_eq!(
            DedupGroupBy::parse("content-hash"),
            DedupGroupBy::ContentHash
        );
        assert_eq!(
            DedupGroupBy::parse("content_hash"),
            DedupGroupBy::ContentHash
        );
        // Unknown / empty values resolve to the post-v4 default rather than
        // erroring out — keeps older HTTP/CLI clients working.
        assert_eq!(DedupGroupBy::parse(""), DedupGroupBy::SourceHashLayer);
        assert_eq!(DedupGroupBy::parse("nope"), DedupGroupBy::SourceHashLayer);
    }
}
