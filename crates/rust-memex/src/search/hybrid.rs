//! Hybrid search combining BM25 keyword matching with vector similarity.
//!
//! Uses Reciprocal Rank Fusion (RRF) or weighted linear combination
//! to merge results from both search methods.

use anyhow::Result;
use chrono::{DateTime, Duration, NaiveDate, NaiveDateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::bm25::{BM25Config, BM25Index};
use crate::rag::{SearchOptions, SliceLayer};
use crate::storage::{ChromaDocument, StorageManager};

/// Search mode configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    /// Vector similarity only (legacy behavior)
    Vector,
    /// BM25 keyword search only
    Keyword,
    /// Combined vector + BM25 with score fusion (default)
    #[default]
    Hybrid,
}

impl std::str::FromStr for SearchMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "vector" => Ok(SearchMode::Vector),
            "keyword" | "bm25" => Ok(SearchMode::Keyword),
            "hybrid" => Ok(SearchMode::Hybrid),
            other => Err(format!(
                "Invalid search mode: '{}'. Use 'vector', 'keyword', or 'hybrid'",
                other
            )),
        }
    }
}

/// Configuration for hybrid search
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HybridConfig {
    /// Search mode (vector, keyword, or hybrid)
    #[serde(default)]
    pub mode: SearchMode,

    /// Weight for vector similarity scores (0.0-1.0)
    #[serde(default = "default_vector_weight")]
    pub vector_weight: f32,

    /// Weight for BM25 keyword scores (0.0-1.0)
    #[serde(default = "default_bm25_weight")]
    pub bm25_weight: f32,

    /// Use Reciprocal Rank Fusion instead of weighted linear
    #[serde(default)]
    pub use_rrf: bool,

    /// RRF constant k (typically 60)
    #[serde(default = "default_rrf_k")]
    pub rrf_k: f32,

    /// BM25 index configuration
    #[serde(default)]
    pub bm25: BM25Config,

    /// Maximum results per source file (0 = unlimited)
    #[serde(default = "default_max_per_source")]
    pub max_per_source: usize,
}

fn default_vector_weight() -> f32 {
    0.6
}
fn default_bm25_weight() -> f32 {
    0.4
}
fn default_rrf_k() -> f32 {
    60.0
}

fn default_max_per_source() -> usize {
    3
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            mode: SearchMode::default(),
            vector_weight: 0.6,
            bm25_weight: 0.4,
            use_rrf: false,
            rrf_k: 60.0,
            bm25: BM25Config::default(),
            max_per_source: default_max_per_source(),
        }
    }
}

/// Hybrid search result with combined scoring
#[derive(Debug, Clone)]
pub struct HybridSearchResult {
    pub id: String,
    pub namespace: String,
    pub document: String,
    pub combined_score: f32,
    pub vector_score: Option<f32>,
    pub bm25_score: Option<f32>,
    pub metadata: serde_json::Value,
    pub layer: Option<SliceLayer>,
    pub parent_id: Option<String>,
    pub children_ids: Vec<String>,
    pub keywords: Vec<String>,
}

type HybridResultKey = (String, String);
type HybridScoreTuple = (f32, Option<f32>, Option<f32>);
type HybridFusionMap = HashMap<HybridResultKey, HybridScoreTuple>;

/// Hybrid searcher combining vector and BM25 search
pub struct HybridSearcher {
    storage: Arc<StorageManager>,
    bm25_index: Option<Arc<BM25Index>>,
    config: HybridConfig,
}

impl HybridSearcher {
    /// Create a new hybrid searcher
    pub async fn new(storage: Arc<StorageManager>, config: HybridConfig) -> Result<Self> {
        let bm25_index = if config.mode != SearchMode::Vector {
            Some(Arc::new(BM25Index::new(&config.bm25)?))
        } else {
            None
        };

        Ok(Self {
            storage,
            bm25_index,
            config,
        })
    }

    /// Create a hybrid searcher with an existing BM25 index
    pub fn with_bm25_index(
        storage: Arc<StorageManager>,
        bm25_index: Arc<BM25Index>,
        config: HybridConfig,
    ) -> Self {
        Self {
            storage,
            bm25_index: Some(bm25_index),
            config,
        }
    }

    /// Get the BM25 index for direct operations
    pub fn bm25_index(&self) -> Option<&Arc<BM25Index>> {
        self.bm25_index.as_ref()
    }

    /// Index documents in both vector store and BM25 index
    pub async fn index_documents(&self, docs: &[ChromaDocument]) -> Result<()> {
        // Add to vector store
        self.storage.add_to_store(docs.to_vec()).await?;

        // Add to BM25 index if available
        if let Some(ref bm25) = self.bm25_index {
            let bm25_docs: Vec<(String, String, String)> = docs
                .iter()
                .map(|d| (d.id.clone(), d.namespace.clone(), d.document.clone()))
                .collect();
            bm25.add_documents(&bm25_docs).await?;
        }

        Ok(())
    }

    /// Perform hybrid search
    pub async fn search(
        &self,
        query: &str,
        query_embedding: Vec<f32>,
        namespace: Option<&str>,
        limit: usize,
        options: SearchOptions,
    ) -> Result<Vec<HybridSearchResult>> {
        match self.config.mode {
            SearchMode::Vector => {
                self.vector_only_search(query, query_embedding, namespace, limit, options)
                    .await
            }
            SearchMode::Keyword => {
                self.keyword_only_search(query, namespace, limit, options)
                    .await
            }
            SearchMode::Hybrid => {
                self.hybrid_search(query, query_embedding, namespace, limit, options)
                    .await
            }
        }
    }

    /// Vector-only search (legacy behavior)
    async fn vector_only_search(
        &self,
        query: &str,
        query_embedding: Vec<f32>,
        namespace: Option<&str>,
        limit: usize,
        options: SearchOptions,
    ) -> Result<Vec<HybridSearchResult>> {
        let candidate_limit = candidate_limit(limit, &options);
        let candidates = self
            .storage
            .search_store_with_layer(
                namespace,
                query_embedding,
                candidate_limit,
                options.layer_filter,
            )
            .await?;

        let mut results: Vec<HybridSearchResult> = candidates
            .into_iter()
            .map(|doc| {
                let layer = doc.slice_layer(); // Call before moving fields
                HybridSearchResult {
                    id: doc.id,
                    namespace: doc.namespace,
                    document: doc.document,
                    combined_score: 1.0, // Will be recalculated by reranker
                    vector_score: Some(1.0),
                    bm25_score: None,
                    metadata: doc.metadata,
                    layer,
                    parent_id: doc.parent_id,
                    children_ids: doc.children_ids,
                    keywords: doc.keywords,
                }
            })
            .collect();
        Self::apply_post_search_processing(query, &mut results, &options);
        Self::dedup_by_chunk_hash(&mut results);
        Self::dedup_by_source_path(&mut results);
        Self::enforce_source_diversity(&mut results, self.config.max_per_source);
        results.truncate(limit);
        Ok(results)
    }

    /// Keyword-only search using BM25
    async fn keyword_only_search(
        &self,
        query: &str,
        namespace: Option<&str>,
        limit: usize,
        options: SearchOptions,
    ) -> Result<Vec<HybridSearchResult>> {
        let bm25 = self
            .bm25_index
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("BM25 index not initialized for keyword search"))?;

        let bm25_results = bm25.search(query, namespace, candidate_limit(limit, &options))?;
        if bm25_results.is_empty() {
            return self
                .keyword_scan_fallback(query, namespace, limit, options)
                .await;
        }

        // Fetch full documents from storage
        let mut results = Vec::with_capacity(bm25_results.len());
        for (id, hit_namespace, score) in bm25_results {
            if let Some(doc) = self.storage.get_document(&hit_namespace, &id).await? {
                let layer = doc.slice_layer(); // Call before moving fields
                results.push(HybridSearchResult {
                    id: doc.id,
                    namespace: doc.namespace,
                    document: doc.document,
                    combined_score: score,
                    vector_score: None,
                    bm25_score: Some(score),
                    metadata: doc.metadata,
                    layer,
                    parent_id: doc.parent_id,
                    children_ids: doc.children_ids,
                    keywords: doc.keywords,
                });
            }
        }

        Self::apply_post_search_processing(query, &mut results, &options);
        Self::dedup_by_chunk_hash(&mut results);
        Self::dedup_by_source_path(&mut results);
        Self::enforce_source_diversity(&mut results, self.config.max_per_source);
        results.truncate(limit);
        Ok(results)
    }

    async fn keyword_scan_fallback(
        &self,
        query: &str,
        namespace: Option<&str>,
        limit: usize,
        options: SearchOptions,
    ) -> Result<Vec<HybridSearchResult>> {
        let terms = query
            .split_whitespace()
            .map(|term| {
                term.trim_matches(|ch: char| !ch.is_alphanumeric())
                    .to_ascii_lowercase()
            })
            .filter(|term| !term.is_empty())
            .collect::<Vec<_>>();
        if terms.is_empty() {
            return Ok(vec![]);
        }

        let docs = self.storage.all_documents(namespace, 100_000).await?;
        let mut results = Vec::new();
        for doc in docs {
            let searchable = format!(
                "{} {} {}",
                doc.document,
                doc.keywords.join(" "),
                doc.metadata
            )
            .to_ascii_lowercase();
            let score = terms
                .iter()
                .map(|term| searchable.matches(term).count())
                .sum::<usize>();
            if score == 0 {
                continue;
            }

            let layer = doc.slice_layer();
            let layer_matches = match options.layer_filter {
                Some(SliceLayer::Outer) => layer.is_none() || layer == Some(SliceLayer::Outer),
                Some(expected) => layer == Some(expected),
                None => true,
            };
            if !layer_matches {
                continue;
            }

            results.push(HybridSearchResult {
                id: doc.id,
                namespace: doc.namespace,
                document: doc.document,
                combined_score: score as f32,
                vector_score: None,
                bm25_score: Some(0.0),
                metadata: doc.metadata,
                layer,
                parent_id: doc.parent_id,
                children_ids: doc.children_ids,
                keywords: doc.keywords,
            });
        }

        Self::apply_post_search_processing(query, &mut results, &options);
        Self::dedup_by_chunk_hash(&mut results);
        Self::dedup_by_source_path(&mut results);
        Self::enforce_source_diversity(&mut results, self.config.max_per_source);
        results.truncate(limit);
        Ok(results)
    }

    /// Hybrid search combining vector and BM25
    async fn hybrid_search(
        &self,
        query: &str,
        query_embedding: Vec<f32>,
        namespace: Option<&str>,
        limit: usize,
        options: SearchOptions,
    ) -> Result<Vec<HybridSearchResult>> {
        let expanded_limit = candidate_limit(limit, &options); // Get more candidates for fusion
        let query_policy = QueryPolicy::from_query(query);

        // Run vector search
        let vector_results = self
            .storage
            .search_store_with_layer(
                namespace,
                query_embedding,
                expanded_limit,
                options.layer_filter,
            )
            .await?;

        // Run BM25 search if available
        let bm25_results = if let Some(ref bm25) = self.bm25_index {
            bm25.search(query, namespace, expanded_limit * 2)?
        } else {
            vec![]
        };

        // Fuse results
        let fused = if self.config.use_rrf {
            self.reciprocal_rank_fusion(&vector_results, &bm25_results)
        } else {
            self.weighted_linear_fusion(&vector_results, &bm25_results)
        };

        // Sort by combined score and take top-k
        let mut results: Vec<_> = fused.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(expanded_limit * 2);

        // Fetch full documents for final results
        let mut final_results = Vec::with_capacity(results.len());
        for ((result_namespace, id), (combined_score, vector_score, bm25_score)) in results {
            // Find in vector results first
            if let Some(doc) = vector_results
                .iter()
                .find(|d| d.id == id && d.namespace == result_namespace)
            {
                let layer = doc.slice_layer(); // Call before cloning metadata
                final_results.push(HybridSearchResult {
                    id: doc.id.clone(),
                    namespace: doc.namespace.clone(),
                    document: doc.document.clone(),
                    combined_score,
                    vector_score,
                    bm25_score,
                    metadata: doc.metadata.clone(),
                    layer,
                    parent_id: doc.parent_id.clone(),
                    children_ids: doc.children_ids.clone(),
                    keywords: doc.keywords.clone(),
                });
            } else if let Some(doc) = self.storage.get_document(&result_namespace, &id).await? {
                // BM25-only result, fetch from storage
                let layer = doc.slice_layer(); // Call before moving fields
                if options.layer_filter.is_some() && layer != options.layer_filter {
                    continue;
                }
                final_results.push(HybridSearchResult {
                    id: doc.id,
                    namespace: doc.namespace,
                    document: doc.document,
                    combined_score,
                    vector_score,
                    bm25_score,
                    metadata: doc.metadata,
                    layer,
                    parent_id: doc.parent_id,
                    children_ids: doc.children_ids,
                    keywords: doc.keywords,
                });
            }
        }

        if query_policy.precision_query {
            final_results.extend(
                self.lexical_precision_candidates(
                    &query_policy,
                    namespace,
                    expanded_limit,
                    &options,
                )
                .await?,
            );
        }

        Self::apply_post_search_processing(query, &mut final_results, &options);
        Self::dedup_by_chunk_hash(&mut final_results);
        Self::dedup_by_source_path(&mut final_results);
        Self::enforce_source_diversity(&mut final_results, self.config.max_per_source);
        final_results.truncate(limit);

        tracing::debug!(
            "Hybrid search: {} vector + {} BM25 -> {} fused (deduped) results",
            vector_results.len(),
            bm25_results.len(),
            final_results.len()
        );

        Ok(final_results)
    }

    async fn lexical_precision_candidates(
        &self,
        query_policy: &QueryPolicy,
        namespace: Option<&str>,
        limit: usize,
        options: &SearchOptions,
    ) -> Result<Vec<HybridSearchResult>> {
        let docs = self.storage.all_documents(namespace, 100_000).await?;
        let mut results = Vec::new();

        for doc in docs {
            let layer = doc.slice_layer();
            let layer_matches = match options.layer_filter {
                Some(SliceLayer::Outer) => layer.is_none() || layer == Some(SliceLayer::Outer),
                Some(expected) => layer == Some(expected),
                None => true,
            };
            if !layer_matches {
                continue;
            }

            let evidence = SearchEvidence::from_parts(&doc.document, &doc.keywords, &doc.metadata);
            let score = query_policy.lexical_score(&evidence);
            if score <= 0.0 {
                continue;
            }

            results.push(HybridSearchResult {
                id: doc.id,
                namespace: doc.namespace,
                document: doc.document,
                combined_score: score,
                vector_score: None,
                bm25_score: Some(score),
                metadata: doc.metadata,
                layer,
                parent_id: doc.parent_id,
                children_ids: doc.children_ids,
                keywords: doc.keywords,
            });
        }

        results.sort_by(|left, right| {
            right
                .combined_score
                .partial_cmp(&left.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);
        Ok(results)
    }

    fn apply_post_search_processing(
        query: &str,
        results: &mut Vec<HybridSearchResult>,
        options: &SearchOptions,
    ) {
        if let Some(project) = options.project_filter.as_deref() {
            results.retain(|result| matches_project_filter(&result.metadata, project));
        }

        let query_policy = QueryPolicy::from_query(query);
        for result in results.iter_mut() {
            let evidence =
                SearchEvidence::from_parts(&result.document, &result.keywords, &result.metadata);
            let lexical_score = query_policy.lexical_score(&evidence);
            let mut score =
                boosted_score(query, result.combined_score, &result.metadata, result.layer);

            if query_policy.precision_query {
                if lexical_score <= 0.0 {
                    score *= 0.12;
                } else {
                    score += lexical_score;
                }

                if result.layer == Some(SliceLayer::Outer)
                    && looks_like_keyword_stub(&result.document)
                    && lexical_score < query_policy.strong_match_threshold()
                {
                    score *= 0.35;
                }
            } else if lexical_score > 0.0 {
                score += lexical_score.min(0.25);
            }

            result.combined_score = score;
        }

        results.sort_by(|left, right| {
            right
                .combined_score
                .partial_cmp(&left.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Deduplicate results by per-chunk hash from metadata.
    ///
    /// Schema v4 (post spec P0) stores the per-chunk SHA256 under
    /// `metadata.chunk_hash`. The legacy field `metadata.content_hash` (v3 and
    /// earlier) actually held the *source* document hash for every onion
    /// layer, so reading it here would either no-op silently (post-v4 chunks
    /// have no `content_hash` field in metadata) or aggressively collapse
    /// every onion layer of a single source into one result (legacy chunks).
    /// Either outcome contradicts the function name.
    ///
    /// Chunk-level dedup therefore keys strictly on `metadata.chunk_hash`.
    /// Results without that field (legacy v3 chunks, no-hash flat indexers)
    /// are passed through untouched — `dedup_by_source_path` and
    /// `enforce_source_diversity` still bound the result list.
    ///
    /// `metadata.source_hash` is preserved for provenance and is not used
    /// here; `metadata.file_hash` is a deprecated alias of `source_hash` and
    /// is also ignored.
    fn dedup_by_chunk_hash(results: &mut Vec<HybridSearchResult>) {
        let mut seen: HashSet<String> = HashSet::new();
        let before = results.len();
        results.retain(|r| {
            match r.metadata.get("chunk_hash").and_then(|v| v.as_str()) {
                Some(hash) => seen.insert(hash.to_string()),
                None => true, // keep legacy/no-hash chunks; source-path dedup still applies
            }
        });
        let removed = before - results.len();
        if removed > 0 {
            tracing::debug!("Dedup: removed {} duplicate chunks by chunk_hash", removed);
        }
    }

    /// Deduplicate results by their source document path when present.
    ///
    /// This keeps only the highest-ranked hit per source file after chunk-level dedup,
    /// so one heavily sliced document does not dominate the result list.
    fn dedup_by_source_path(results: &mut Vec<HybridSearchResult>) {
        let mut seen: HashSet<String> = HashSet::new();
        let before = results.len();

        results.retain(|result| {
            let source_key = result
                .metadata
                .get("source_path")
                .and_then(|value| value.as_str())
                .or_else(|| result.metadata.get("path").and_then(|value| value.as_str()))
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| format!("{}::{}", result.namespace, result.id));

            seen.insert(source_key)
        });

        let removed = before - results.len();
        if removed > 0 {
            tracing::debug!(
                "Dedup: removed {} duplicate hits by source path/doc id",
                removed
            );
        }
    }

    /// Enforce source-file diversity: max N results per unique source path.
    /// Prevents a single large file from dominating all results.
    fn enforce_source_diversity(results: &mut Vec<HybridSearchResult>, max_per_source: usize) {
        if max_per_source == 0 {
            return; // 0 = unlimited
        }
        let mut source_counts: HashMap<String, usize> = HashMap::new();
        let before = results.len();
        results.retain(|r| {
            let path = r
                .metadata
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            let count = source_counts.entry(path).or_insert(0);
            *count += 1;
            *count <= max_per_source
        });
        let removed = before - results.len();
        if removed > 0 {
            tracing::debug!(
                "Diversity: capped {} results (max {} per source)",
                removed,
                max_per_source
            );
        }
    }

    /// Weighted linear combination of scores
    fn weighted_linear_fusion(
        &self,
        vector_results: &[ChromaDocument],
        bm25_results: &[(String, String, f32)],
    ) -> HybridFusionMap {
        let mut combined: HybridFusionMap = HashMap::new();

        // Use rank-based normalization for vector results
        for (idx, doc) in vector_results.iter().enumerate() {
            // Use rank-based scoring for vector results (higher rank = higher score)
            let normalized = 1.0 - (idx as f32 / vector_results.len().max(1) as f32);
            let weighted = normalized * self.config.vector_weight;

            combined.insert(
                (doc.namespace.clone(), doc.id.clone()),
                (weighted, Some(normalized), None),
            );
        }

        // Normalize and add BM25 scores
        let bm25_max = bm25_results
            .iter()
            .map(|(_, _, score)| *score)
            .fold(0.0_f32, f32::max);

        for (id, namespace, score) in bm25_results {
            let normalized = if bm25_max > 0.0 {
                score / bm25_max
            } else {
                0.0
            };
            let weighted = normalized * self.config.bm25_weight;

            combined
                .entry((namespace.clone(), id.clone()))
                .and_modify(|(total, _, bm25)| {
                    *total += weighted;
                    *bm25 = Some(normalized);
                })
                .or_insert((weighted, None, Some(normalized)));
        }

        combined
    }

    /// Reciprocal Rank Fusion (RRF)
    /// RRF(d) = sum(1 / (k + rank(d)))
    fn reciprocal_rank_fusion(
        &self,
        vector_results: &[ChromaDocument],
        bm25_results: &[(String, String, f32)],
    ) -> HybridFusionMap {
        let mut combined: HybridFusionMap = HashMap::new();
        let k = self.config.rrf_k;

        // Add vector results with RRF scoring
        for (rank, doc) in vector_results.iter().enumerate() {
            let rrf_score = 1.0 / (k + rank as f32 + 1.0);
            let weighted = rrf_score * self.config.vector_weight;

            combined.insert(
                (doc.namespace.clone(), doc.id.clone()),
                (weighted, Some(rrf_score), None),
            );
        }

        // Add BM25 results with RRF scoring
        for (rank, (id, namespace, _)) in bm25_results.iter().enumerate() {
            let rrf_score = 1.0 / (k + rank as f32 + 1.0);
            let weighted = rrf_score * self.config.bm25_weight;

            combined
                .entry((namespace.clone(), id.clone()))
                .and_modify(|(total, _, bm25)| {
                    *total += weighted;
                    *bm25 = Some(rrf_score);
                })
                .or_insert((weighted, None, Some(rrf_score)));
        }

        combined
    }

    /// Delete documents from both indices
    pub async fn delete_from_indices(&self, namespace: &str, ids: &[String]) -> Result<usize> {
        let mut deleted = 0;

        for id in ids {
            deleted += self.storage.delete_document(namespace, id).await?;
        }

        if let Some(ref bm25) = self.bm25_index {
            bm25.delete_documents(ids).await?;
        }

        Ok(deleted)
    }
}

fn candidate_limit(limit: usize, options: &SearchOptions) -> usize {
    let multiplier = if options.project_filter.is_some() {
        8
    } else {
        3
    };
    limit.max(1) * multiplier
}

struct SearchEvidence {
    text: String,
}

impl SearchEvidence {
    fn from_parts(document: &str, keywords: &[String], metadata: &Value) -> Self {
        Self {
            text: normalize_search_text(&format!(
                "{} {} {}",
                document,
                keywords.join(" "),
                metadata
            )),
        }
    }

    fn contains(&self, needle: &str) -> bool {
        self.text.contains(needle)
    }
}

#[derive(Debug, Clone)]
struct QueryPolicy {
    precision_query: bool,
    anchors: Vec<Vec<String>>,
}

impl QueryPolicy {
    fn from_query(query: &str) -> Self {
        let raw_terms = query
            .split_whitespace()
            .map(|term| term.trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '-'))
            .filter(|term| !term.is_empty())
            .collect::<Vec<_>>();
        let normalized_terms = raw_terms
            .iter()
            .map(|term| normalize_search_text(term))
            .filter(|term| !term.is_empty() && !is_search_stopword(term))
            .collect::<Vec<_>>();

        let has_acronym = raw_terms.iter().any(|term| {
            let alpha_count = term.chars().filter(|ch| ch.is_alphabetic()).count();
            let upper_count = term.chars().filter(|ch| ch.is_uppercase()).count();
            (2..=8).contains(&alpha_count) && upper_count >= 2
        });
        let has_definition_intent = normalized_terms.iter().any(|term| {
            matches!(
                term.as_str(),
                "definition" | "defined" | "define" | "definicja" | "definicje" | "znaczenie"
            )
        });
        let precision_query = normalized_terms.len() <= 4 && (has_acronym || has_definition_intent);

        let anchors = normalized_terms
            .iter()
            .map(|term| expanded_anchor_terms(term))
            .collect::<Vec<_>>();

        Self {
            precision_query,
            anchors,
        }
    }

    fn lexical_score(&self, evidence: &SearchEvidence) -> f32 {
        if self.anchors.is_empty() {
            return 0.0;
        }

        let mut matched = 0usize;
        let mut score = 0.0f32;
        for anchor_group in &self.anchors {
            let mut group_score = 0.0f32;
            for anchor in anchor_group {
                if evidence.contains(anchor) {
                    group_score = group_score.max(anchor_weight(anchor));
                }
            }
            if group_score > 0.0 {
                matched += 1;
                score += group_score;
            }
        }

        if self.precision_query && matched == self.anchors.len() {
            score += 1.0;
        }

        score
    }

    fn strong_match_threshold(&self) -> f32 {
        if self.anchors.len() >= 2 { 2.0 } else { 1.0 }
    }
}

fn expanded_anchor_terms(term: &str) -> Vec<String> {
    match term {
        "dou" => vec![
            "dou".to_string(),
            "vc dou".to_string(),
            "vc-dou".to_string(),
            "definition of undone".to_string(),
            "undone".to_string(),
        ],
        "definicja" | "definicje" | "znaczenie" => vec![
            term.to_string(),
            "definition".to_string(),
            "defined".to_string(),
            "define".to_string(),
        ],
        "definition" | "defined" | "define" => vec![
            term.to_string(),
            "definicja".to_string(),
            "definicje".to_string(),
        ],
        other => vec![other.to_string()],
    }
}

fn anchor_weight(anchor: &str) -> f32 {
    match anchor {
        "definition of undone" => 2.0,
        "vc dou" | "vc-dou" | "dou" => 1.4,
        "definition" | "definicja" | "definicje" => 1.2,
        _ => 1.0,
    }
}

fn normalize_search_text(value: &str) -> String {
    value
        .to_lowercase()
        .chars()
        .map(|ch| {
            if ch.is_alphanumeric() || ch.is_whitespace() || ch == '-' {
                ch
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn is_search_stopword(term: &str) -> bool {
    matches!(
        term,
        "the"
            | "a"
            | "an"
            | "and"
            | "or"
            | "of"
            | "in"
            | "o"
            | "i"
            | "w"
            | "we"
            | "na"
            | "do"
            | "to"
            | "jest"
            | "co"
            | "czym"
    )
}

fn looks_like_keyword_stub(document: &str) -> bool {
    let trimmed = document.trim();
    trimmed.starts_with('[')
        && trimmed
            .lines()
            .next()
            .is_some_and(|line| line.len() < 180 && line.contains(',') && line.contains(']'))
}

fn matches_project_filter(metadata: &Value, project: &str) -> bool {
    let needle = project.trim();
    if needle.is_empty() {
        return true;
    }

    let needle = canonical_project_identity(needle);

    metadata.as_object().is_some_and(|object| {
        ["project", "project_id", "source_project"]
            .iter()
            .filter_map(|key| object.get(*key))
            .filter_map(|value| value.as_str())
            .any(|value| canonical_project_identity(value) == needle)
    })
}

fn canonical_project_identity(value: &str) -> String {
    match value.trim().to_ascii_lowercase().as_str() {
        "loctree" | "vetcoders" => "vetcoders".to_string(),
        other => other.to_string(),
    }
}

fn boosted_score(query: &str, base_score: f32, metadata: &Value, layer: Option<SliceLayer>) -> f32 {
    (base_score * layer_multiplier(query, layer)) + recency_bonus(metadata)
}

fn layer_multiplier(query: &str, layer: Option<SliceLayer>) -> f32 {
    let Some(layer) = layer else {
        return 1.0;
    };

    if is_detailed_query(query) {
        match layer {
            SliceLayer::Outer => 0.96,
            SliceLayer::Middle => 1.03,
            SliceLayer::Inner => 1.08,
            SliceLayer::Core => 1.12,
        }
    } else {
        match layer {
            SliceLayer::Outer => 1.08,
            SliceLayer::Middle => 1.04,
            SliceLayer::Inner => 1.0,
            SliceLayer::Core => 0.96,
        }
    }
}

fn is_detailed_query(query: &str) -> bool {
    let lowered = query.to_ascii_lowercase();
    let word_count = lowered.split_whitespace().count();
    word_count >= 8
        || [
            "why",
            "how",
            "details",
            "detailed",
            "implementation",
            "stacktrace",
            "exact",
            "quote",
            "full",
            "error",
            "trace",
        ]
        .iter()
        .any(|needle| lowered.contains(needle))
}

fn recency_bonus(metadata: &Value) -> f32 {
    let Some(object) = metadata.as_object() else {
        return 0.0;
    };

    let timestamp = ["timestamp", "created_at", "indexed_at", "date", "time"]
        .iter()
        .filter_map(|key| object.get(*key))
        .filter_map(|value| value.as_str())
        .find_map(parse_timestamp);

    let Some(timestamp) = timestamp else {
        return 0.0;
    };

    let age = Utc::now().signed_duration_since(timestamp);
    if age < Duration::zero() || age > Duration::days(14) {
        return 0.0;
    }

    let remaining = (Duration::days(14) - age).num_seconds().max(0) as f32;
    let window = Duration::days(14).num_seconds().max(1) as f32;
    0.12 * (remaining / window)
}

fn parse_timestamp(value: &str) -> Option<DateTime<Utc>> {
    if let Ok(parsed) = DateTime::parse_from_rfc3339(value) {
        return Some(parsed.with_timezone(&Utc));
    }

    if let Ok(parsed) = NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S") {
        return Some(DateTime::<Utc>::from_naive_utc_and_offset(parsed, Utc));
    }

    if let Ok(parsed) = NaiveDate::parse_from_str(value, "%Y-%m-%d") {
        let at_midnight = parsed.and_hms_opt(0, 0, 0)?;
        return Some(DateTime::<Utc>::from_naive_utc_and_offset(at_midnight, Utc));
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    #[test]
    fn test_search_mode_parsing() {
        assert_eq!("vector".parse::<SearchMode>().unwrap(), SearchMode::Vector);
        assert_eq!(
            "keyword".parse::<SearchMode>().unwrap(),
            SearchMode::Keyword
        );
        assert_eq!("bm25".parse::<SearchMode>().unwrap(), SearchMode::Keyword);
        assert_eq!("hybrid".parse::<SearchMode>().unwrap(), SearchMode::Hybrid);
        assert!("invalid".parse::<SearchMode>().is_err());
    }

    #[test]
    fn test_default_config() {
        let config = HybridConfig::default();
        assert_eq!(config.mode, SearchMode::Hybrid);
        assert_eq!(config.vector_weight, 0.6);
        assert_eq!(config.bm25_weight, 0.4);
        assert!(!config.use_rrf);
    }

    #[test]
    fn precision_query_boosts_dou_definition_evidence() {
        let policy = QueryPolicy::from_query("DoU definicja");
        assert!(policy.precision_query);

        let good = SearchEvidence::from_parts(
            "8. vc-dou — Definition of Undone audit calej powierzchni produktu",
            &[],
            &json!({}),
        );
        let weak = SearchEvidence::from_parts(
            "[assistant, nie, search, wiec, juz] [project: VetCoders/ai-contexters]",
            &[],
            &json!({}),
        );

        assert!(policy.lexical_score(&good) >= policy.strong_match_threshold());
        assert_eq!(policy.lexical_score(&weak), 0.0);
        assert!(looks_like_keyword_stub(
            "[assistant, nie, search, wiec, juz] [project: x]"
        ));
    }

    #[tokio::test]
    async fn test_keyword_search_uses_bm25_hit_namespace() {
        let storage_dir = TempDir::new().unwrap();
        let bm25_dir = TempDir::new().unwrap();
        let storage = Arc::new(
            StorageManager::new_lance_only(storage_dir.path().to_str().unwrap())
                .await
                .unwrap(),
        );

        storage.ensure_collection().await.unwrap();

        let config = HybridConfig {
            mode: SearchMode::Keyword,
            bm25: BM25Config::default().with_path(bm25_dir.path().to_str().unwrap()),
            max_per_source: 0,
            ..Default::default()
        };
        let searcher = HybridSearcher::new(storage, config).await.unwrap();

        let embedding = vec![0.1f32; 4096];
        let docs = vec![
            ChromaDocument::new_flat(
                "shared-id".to_string(),
                "namespace-a".to_string(),
                embedding.clone(),
                json!({"path": "a.txt"}),
                "alpha shared term".to_string(),
            ),
            ChromaDocument::new_flat(
                "shared-id".to_string(),
                "namespace-b".to_string(),
                embedding,
                json!({"path": "b.txt"}),
                "beta shared term".to_string(),
            ),
        ];

        searcher.index_documents(&docs).await.unwrap();

        let results = searcher
            .search("shared", vec![], None, 10, SearchOptions::default())
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        assert!(
            results
                .iter()
                .any(|result| result.namespace == "namespace-a")
        );
        assert!(
            results
                .iter()
                .any(|result| result.namespace == "namespace-b")
        );
    }

    #[tokio::test]
    async fn keyword_search_falls_back_to_lance_text_when_bm25_is_empty() {
        let storage_dir = TempDir::new().unwrap();
        let bm25_dir = TempDir::new().unwrap();
        let storage = Arc::new(
            StorageManager::new_lance_only(storage_dir.path().to_str().unwrap())
                .await
                .unwrap(),
        );

        storage
            .add_to_store(vec![ChromaDocument::new_flat(
                "doc-1".to_string(),
                "namespace-a".to_string(),
                vec![0.1f32; 8],
                json!({"path": "fallback.txt"}),
                "needle term only exists in Lance".to_string(),
            )])
            .await
            .unwrap();

        let config = HybridConfig {
            mode: SearchMode::Keyword,
            bm25: BM25Config::default().with_path(bm25_dir.path().to_str().unwrap()),
            max_per_source: 0,
            ..Default::default()
        };
        let searcher = HybridSearcher::new(storage, config).await.unwrap();

        let results = searcher
            .search(
                "needle",
                vec![],
                Some("namespace-a"),
                10,
                SearchOptions::default(),
            )
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "doc-1");
        assert_eq!(results[0].namespace, "namespace-a");
    }

    #[tokio::test]
    async fn test_hybrid_search_keeps_duplicate_ids_across_namespaces() {
        let storage_dir = TempDir::new().unwrap();
        let bm25_dir = TempDir::new().unwrap();
        let storage = Arc::new(
            StorageManager::new_lance_only(storage_dir.path().to_str().unwrap())
                .await
                .unwrap(),
        );

        storage.ensure_collection().await.unwrap();

        let config = HybridConfig {
            mode: SearchMode::Hybrid,
            bm25: BM25Config::default().with_path(bm25_dir.path().to_str().unwrap()),
            max_per_source: 0,
            ..Default::default()
        };
        let searcher = HybridSearcher::new(storage, config).await.unwrap();

        let embedding = vec![0.25f32; 4096];
        let docs = vec![
            ChromaDocument::new_flat(
                "shared-id".to_string(),
                "namespace-a".to_string(),
                embedding.clone(),
                json!({"path": "a.txt"}),
                "alpha shared term".to_string(),
            ),
            ChromaDocument::new_flat(
                "shared-id".to_string(),
                "namespace-b".to_string(),
                embedding.clone(),
                json!({"path": "b.txt"}),
                "beta shared term".to_string(),
            ),
        ];

        searcher.index_documents(&docs).await.unwrap();

        let results = searcher
            .search("shared", embedding, None, 10, SearchOptions::default())
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        assert!(
            results
                .iter()
                .any(|result| result.namespace == "namespace-a")
        );
        assert!(
            results
                .iter()
                .any(|result| result.namespace == "namespace-b")
        );
    }

    #[test]
    fn test_dedup_by_source_path_keeps_highest_ranked_hit_per_file() {
        let mut results = vec![
            HybridSearchResult {
                id: "doc-outer".to_string(),
                namespace: "ns".to_string(),
                document: "outer".to_string(),
                combined_score: 0.9,
                vector_score: Some(0.9),
                bm25_score: Some(0.8),
                metadata: json!({"path": "/tmp/shared.md"}),
                layer: None,
                parent_id: None,
                children_ids: vec![],
                keywords: vec![],
            },
            HybridSearchResult {
                id: "doc-inner".to_string(),
                namespace: "ns".to_string(),
                document: "inner".to_string(),
                combined_score: 0.7,
                vector_score: Some(0.7),
                bm25_score: Some(0.6),
                metadata: json!({"path": "/tmp/shared.md"}),
                layer: None,
                parent_id: None,
                children_ids: vec![],
                keywords: vec![],
            },
            HybridSearchResult {
                id: "doc-other".to_string(),
                namespace: "ns".to_string(),
                document: "other".to_string(),
                combined_score: 0.5,
                vector_score: Some(0.5),
                bm25_score: Some(0.4),
                metadata: json!({"path": "/tmp/other.md"}),
                layer: None,
                parent_id: None,
                children_ids: vec![],
                keywords: vec![],
            },
        ];

        HybridSearcher::dedup_by_source_path(&mut results);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "doc-outer");
        assert_eq!(results[1].id, "doc-other");
    }

    /// Schema v4 (post spec P0) writes per-chunk SHA256 under
    /// `metadata.chunk_hash`. The pre-v4 dedup keyed off `metadata.content_hash`,
    /// which is either absent (v4) or holds the source-document hash (v3
    /// legacy) — neither matches the per-chunk semantic the function name
    /// promises. This test locks the v4 contract: identical `chunk_hash`
    /// collapses to one survivor, distinct hashes survive, missing field is
    /// passed through.
    #[test]
    fn dedup_by_chunk_hash_collapses_v4_chunk_duplicates_only() {
        fn fixture(id: &str, score: f32, chunk_hash: Option<&str>) -> HybridSearchResult {
            let metadata = match chunk_hash {
                Some(hash) => json!({
                    "path": "/tmp/transcripts/a.md",
                    "source_hash": "src-aaaa",
                    "chunk_hash": hash,
                    "layer": "outer",
                }),
                None => json!({
                    "path": "/tmp/transcripts/legacy.md",
                    "source_hash": "src-aaaa",
                    "layer": "outer",
                }),
            };
            HybridSearchResult {
                id: id.to_string(),
                namespace: "kb:transcripts".to_string(),
                document: format!("{id} document text"),
                combined_score: score,
                vector_score: Some(score),
                bm25_score: Some(score),
                metadata,
                layer: None,
                parent_id: None,
                children_ids: vec![],
                keywords: vec![],
            }
        }

        // Two identical-chunk hits arrive in score order (e.g. same chunk
        // surfaced through both vector and BM25 lanes); only the first must
        // survive. A distinct chunk and a legacy chunk (no `chunk_hash`) must
        // pass through untouched.
        let mut results = vec![
            fixture("dup-high", 0.9, Some("chunk-aaaa")),
            fixture("dup-low", 0.7, Some("chunk-aaaa")),
            fixture("unique", 0.6, Some("chunk-bbbb")),
            fixture("legacy", 0.5, None),
        ];

        HybridSearcher::dedup_by_chunk_hash(&mut results);

        assert_eq!(
            results.len(),
            3,
            "expected duplicate chunk_hash to collapse, got {:?}",
            results.iter().map(|r| &r.id).collect::<Vec<_>>()
        );
        assert_eq!(results[0].id, "dup-high", "first duplicate must win");
        assert!(
            results.iter().any(|r| r.id == "unique"),
            "distinct chunk_hash must survive"
        );
        assert!(
            results.iter().any(|r| r.id == "legacy"),
            "legacy chunk without chunk_hash must pass through"
        );
        assert!(
            !results.iter().any(|r| r.id == "dup-low"),
            "second duplicate must be removed"
        );
    }

    /// Pre-v4 metadata stored the *source* document hash under
    /// `content_hash`, so every onion layer of a single source shared the
    /// same value. The post-P0 dedup must NOT key on that legacy field —
    /// otherwise outer/middle/inner/core hits collapse to a single layer at
    /// search time, masking the onion structure operators rebuilt the
    /// namespace to expose. This test pins that boundary.
    #[test]
    fn dedup_by_chunk_hash_ignores_legacy_content_hash_field() {
        fn legacy(id: &str, layer: &str) -> HybridSearchResult {
            HybridSearchResult {
                id: id.to_string(),
                namespace: "kb:transcripts".to_string(),
                document: format!("{id} {layer}"),
                combined_score: 0.5,
                vector_score: Some(0.5),
                bm25_score: Some(0.4),
                metadata: json!({
                    "path": "/tmp/transcripts/legacy.md",
                    // Pre-v4 quirk: every layer carried source_hash here.
                    "content_hash": "shared-source-hash",
                    "layer": layer,
                }),
                layer: None,
                parent_id: None,
                children_ids: vec![],
                keywords: vec![],
            }
        }

        let mut results = vec![
            legacy("outer", "outer"),
            legacy("middle", "middle"),
            legacy("inner", "inner"),
            legacy("core", "core"),
        ];

        HybridSearcher::dedup_by_chunk_hash(&mut results);

        assert_eq!(
            results.len(),
            4,
            "legacy `content_hash` (== source hash) must not collapse onion layers"
        );
    }

    #[test]
    fn project_filter_matches_project_and_project_id() {
        assert!(matches_project_filter(
            &json!({"project": "Vista"}),
            "vista"
        ));
        assert!(matches_project_filter(
            &json!({"project_id": "Loctree"}),
            "vetcoders"
        ));
        assert!(!matches_project_filter(
            &json!({"project": "rust-memex"}),
            "vista"
        ));
    }

    #[test]
    fn recency_bonus_rewards_recent_documents() {
        let fresh = json!({"indexed_at": Utc::now().to_rfc3339()});
        let stale = json!({"indexed_at": (Utc::now() - Duration::days(30)).to_rfc3339()});

        assert!(recency_bonus(&fresh) > 0.0);
        assert_eq!(recency_bonus(&stale), 0.0);
    }

    #[test]
    fn detailed_queries_prefer_core_over_outer() {
        let query = "how exactly did the indexing pipeline fail with this stacktrace";
        assert!(layer_multiplier(query, Some(SliceLayer::Core)) > 1.0);
        assert!(layer_multiplier(query, Some(SliceLayer::Outer)) < 1.0);
    }
}
