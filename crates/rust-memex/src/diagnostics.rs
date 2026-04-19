use std::collections::{BTreeMap, BTreeSet, HashMap};

use anyhow::Result;
use chrono::{DateTime, Duration, NaiveDate, NaiveDateTime, TimeZone, Utc};
use memex_contracts::{
    audit::{AuditRecommendation, AuditResult},
    stats::{DatabaseStats, NamespaceStats},
    timeline::TimelineEntry,
};
use serde::{Deserialize, Serialize};

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DedupDuplicate {
    pub id: String,
    pub namespace: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DedupGroup {
    pub content_hash: String,
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
) -> Result<DedupResult> {
    let all_docs = storage.all_documents(namespace, 1_000_000).await?;

    let mut hash_groups: HashMap<String, Vec<_>> = HashMap::new();
    let mut docs_without_hash = 0usize;

    for doc in &all_docs {
        match &doc.content_hash {
            Some(hash) if !hash.is_empty() => {
                let key = if cross_namespace {
                    hash.clone()
                } else {
                    format!("{}:{}", doc.namespace, hash)
                };
                hash_groups.entry(key).or_default().push(doc);
            }
            _ => docs_without_hash += 1,
        }
    }

    let mut result = DedupResult {
        total_docs: all_docs.len(),
        unique_docs: 0,
        duplicate_groups: 0,
        duplicates_removed: 0,
        docs_without_hash,
        groups: Vec::new(),
    };

    for (_key, mut docs) in hash_groups {
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

        result.unique_docs += 1;
        result.duplicate_groups += 1;
        result.duplicates_removed += removed_docs.len();
        result.groups.push(DedupGroup {
            content_hash: kept.content_hash.clone().unwrap_or_default(),
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
}

fn extract_doc_timestamp_string(
    metadata: Option<&serde_json::Map<String, serde_json::Value>>,
) -> Option<String> {
    metadata.and_then(|object| {
        object.iter().find_map(|(key, value)| {
            if !(key.contains("date") || key.contains("timestamp") || key.contains("time")) {
                return None;
            }
            value.as_str().map(ToOwned::to_owned)
        })
    })
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
