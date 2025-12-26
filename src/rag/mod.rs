use anyhow::Result;
use pdf_extract;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::debug;

use crate::{
    embeddings::MLXBridge,
    preprocessing::{PreprocessingConfig, Preprocessor},
    storage::{ChromaDocument, StorageManager},
};

const DEFAULT_NAMESPACE: &str = "rag";

/// Storage batch size - write to LanceDB every N documents to avoid RAM explosion
/// and enable crash recovery for large file indexing.
const STORAGE_BATCH_SIZE: usize = 100;

// =============================================================================
// ONION SLICE ARCHITECTURE
// =============================================================================
//
// The onion-like slice architecture creates hierarchical embeddings:
//   OUTER  (~100 chars) - Keywords, topic, participants
//   MIDDLE (~300 chars) - Key points, decisions, summary
//   INNER  (~600 chars) - Detailed context, quotes, reasoning
//   CORE   (full text)  - Complete original content
//
// Philosophy: "Minimum info -> Maximum navigation paths"
// Search returns OUTER slices by default; user drills down as needed.
// =============================================================================

/// Layer in the onion-like slice hierarchy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum SliceLayer {
    /// ~100 chars - Keywords, topic, "What is this about?"
    Outer = 1,
    /// ~300 chars - Key points, summary, "What happened?"
    Middle = 2,
    /// ~600 chars - Detailed context, "How did it happen?"
    Inner = 3,
    /// Full content - Complete original text
    Core = 4,
}

impl SliceLayer {
    /// Target character count for this layer
    pub fn target_chars(&self) -> usize {
        match self {
            SliceLayer::Outer => 100,
            SliceLayer::Middle => 300,
            SliceLayer::Inner => 600,
            SliceLayer::Core => usize::MAX,
        }
    }

    /// Convert to u8 for storage
    pub fn as_u8(&self) -> u8 {
        *self as u8
    }

    /// Convert from u8
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(SliceLayer::Outer),
            2 => Some(SliceLayer::Middle),
            3 => Some(SliceLayer::Inner),
            4 => Some(SliceLayer::Core),
            _ => None,
        }
    }

    /// Human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            SliceLayer::Outer => "outer",
            SliceLayer::Middle => "middle",
            SliceLayer::Inner => "inner",
            SliceLayer::Core => "core",
        }
    }
}

impl std::fmt::Display for SliceLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// A slice in the onion hierarchy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnionSlice {
    /// Unique ID for this slice (hash-based)
    pub id: String,
    /// Layer in the hierarchy
    pub layer: SliceLayer,
    /// The slice content
    pub content: String,
    /// Parent slice ID (None for Core)
    pub parent_id: Option<String>,
    /// Children slice IDs (empty for Outer)
    pub children_ids: Vec<String>,
    /// Extracted keywords for this slice
    pub keywords: Vec<String>,
}

impl OnionSlice {
    /// Generate a deterministic ID from content hash
    pub fn generate_id(content: &str, layer: SliceLayer) -> String {
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        layer.as_u8().hash(&mut hasher);
        format!("slice_{:016x}_{}", hasher.finish(), layer.name())
    }
}

/// Slicing mode for document indexing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SliceMode {
    /// Hierarchical onion slicing (default)
    #[default]
    Onion,
    /// Traditional flat chunking (backward compatible)
    Flat,
}

impl std::str::FromStr for SliceMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "onion" => Ok(SliceMode::Onion),
            "flat" => Ok(SliceMode::Flat),
            other => Err(format!(
                "Invalid slice mode: '{}'. Use 'onion' or 'flat'",
                other
            )),
        }
    }
}

/// Result of indexing operation with deduplication
#[derive(Debug, Clone)]
pub enum IndexResult {
    /// Content was indexed successfully
    Indexed {
        /// Number of chunks/slices indexed
        chunks_indexed: usize,
        /// Content hash for the indexed content
        content_hash: String,
    },
    /// Content was skipped because it already exists (exact-match duplicate)
    Skipped {
        /// Reason for skipping
        reason: String,
        /// Content hash that was found as duplicate
        content_hash: String,
    },
}

impl IndexResult {
    /// Check if content was indexed
    pub fn is_indexed(&self) -> bool {
        matches!(self, IndexResult::Indexed { .. })
    }

    /// Check if content was skipped
    pub fn is_skipped(&self) -> bool {
        matches!(self, IndexResult::Skipped { .. })
    }

    /// Get the content hash
    pub fn content_hash(&self) -> &str {
        match self {
            IndexResult::Indexed { content_hash, .. } => content_hash,
            IndexResult::Skipped { content_hash, .. } => content_hash,
        }
    }
}

/// Compute SHA256 hash of content and return as hex string
pub fn compute_content_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let result = hasher.finalize();
    // Convert to hex string (64 chars for SHA256)
    result.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Configuration for onion slicing
#[derive(Debug, Clone)]
pub struct OnionSliceConfig {
    /// Target size for outer layer (~100 chars)
    pub outer_target: usize,
    /// Target size for middle layer (~300 chars)
    pub middle_target: usize,
    /// Target size for inner layer (~600 chars)
    pub inner_target: usize,
    /// Minimum content length to apply onion slicing (below this, use single Core slice)
    pub min_content_for_slicing: usize,
}

impl Default for OnionSliceConfig {
    fn default() -> Self {
        Self {
            outer_target: 100,
            middle_target: 300,
            inner_target: 600,
            min_content_for_slicing: 200,
        }
    }
}

/// Create onion slices from content
///
/// Algorithm:
/// 1. Full content -> CORE slice
/// 2. Extract key sentences -> INNER slice (~600 chars)
/// 3. Summarize to key points -> MIDDLE slice (~300 chars)
/// 4. Extract keywords/topic -> OUTER slice (~100 chars)
pub fn create_onion_slices(
    content: &str,
    _metadata: &serde_json::Value,
    config: &OnionSliceConfig,
) -> Vec<OnionSlice> {
    let content = content.trim();

    // For very short content, just create a single Core slice
    if content.len() < config.min_content_for_slicing {
        let core_id = OnionSlice::generate_id(content, SliceLayer::Core);
        let keywords = extract_keywords(content, 5);
        return vec![OnionSlice {
            id: core_id,
            layer: SliceLayer::Core,
            content: content.to_string(),
            parent_id: None,
            children_ids: vec![],
            keywords,
        }];
    }

    let mut slices = Vec::with_capacity(4);

    // 1. CORE slice - full content
    let core_id = OnionSlice::generate_id(content, SliceLayer::Core);
    let core_keywords = extract_keywords(content, 10);

    // 2. INNER slice - extract key sentences (~600 chars)
    let inner_content = extract_key_content(content, config.inner_target);
    let inner_id = OnionSlice::generate_id(&inner_content, SliceLayer::Inner);
    let inner_keywords = extract_keywords(&inner_content, 7);

    // 3. MIDDLE slice - summarize to key points (~300 chars)
    let middle_content = extract_key_content(&inner_content, config.middle_target);
    let middle_id = OnionSlice::generate_id(&middle_content, SliceLayer::Middle);
    let middle_keywords = extract_keywords(&middle_content, 5);

    // 4. OUTER slice - keywords and topic (~100 chars)
    let outer_content = create_outer_summary(&middle_content, &core_keywords, config.outer_target);
    let outer_id = OnionSlice::generate_id(&outer_content, SliceLayer::Outer);
    let outer_keywords = extract_keywords(&outer_content, 3);

    // Build hierarchy with parent/children links
    slices.push(OnionSlice {
        id: outer_id.clone(),
        layer: SliceLayer::Outer,
        content: outer_content,
        parent_id: Some(middle_id.clone()),
        children_ids: vec![],
        keywords: outer_keywords,
    });

    slices.push(OnionSlice {
        id: middle_id.clone(),
        layer: SliceLayer::Middle,
        content: middle_content,
        parent_id: Some(inner_id.clone()),
        children_ids: vec![outer_id],
        keywords: middle_keywords,
    });

    slices.push(OnionSlice {
        id: inner_id.clone(),
        layer: SliceLayer::Inner,
        content: inner_content,
        parent_id: Some(core_id.clone()),
        children_ids: vec![middle_id],
        keywords: inner_keywords,
    });

    slices.push(OnionSlice {
        id: core_id.clone(),
        layer: SliceLayer::Core,
        content: content.to_string(),
        parent_id: None,
        children_ids: vec![inner_id],
        keywords: core_keywords,
    });

    slices
}

/// Extract keywords from text using simple TF-based extraction
fn extract_keywords(text: &str, max_keywords: usize) -> Vec<String> {
    use std::collections::HashMap;

    // Common stop words to filter out
    const STOP_WORDS: &[&str] = &[
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "from", "as", "is", "was", "are", "were", "been", "be", "have", "has", "had", "do", "does",
        "did", "will", "would", "could", "should", "may", "might", "must", "shall", "can", "this",
        "that", "these", "those", "i", "you", "he", "she", "it", "we", "they", "what", "which",
        "who", "whom", "when", "where", "why", "how", "all", "each", "every", "both", "few",
        "more", "most", "other", "some", "such", "no", "not", "only", "own", "same", "so", "than",
        "too", "very", "just", "also", "now", "here", "there", "then", "once", "if", "into",
        "through", "during", "before", "after", "above", "below", "between", "under", "again",
        "further", "about", "out", "over", "up", "down", "off", "any", "because", "until", "while",
    ];

    let stop_set: std::collections::HashSet<&str> = STOP_WORDS.iter().copied().collect();

    // Tokenize and count word frequencies
    let mut word_counts: HashMap<String, usize> = HashMap::new();
    for word in text.split_whitespace() {
        let cleaned: String = word
            .chars()
            .filter(|c| c.is_alphanumeric())
            .collect::<String>()
            .to_lowercase();

        if cleaned.len() >= 3 && !stop_set.contains(cleaned.as_str()) {
            *word_counts.entry(cleaned).or_insert(0) += 1;
        }
    }

    // Sort by frequency and take top N
    let mut words: Vec<_> = word_counts.into_iter().collect();
    words.sort_by(|a, b| b.1.cmp(&a.1));

    words
        .into_iter()
        .take(max_keywords)
        .map(|(word, _)| word)
        .collect()
}

/// Extract key content from text, targeting a specific character count
/// Uses sentence-based extraction to maintain coherence
fn extract_key_content(text: &str, target_chars: usize) -> String {
    if text.len() <= target_chars {
        return text.to_string();
    }

    // Split into sentences (simple heuristic)
    let sentences: Vec<&str> = text
        .split(['.', '!', '?'])
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    if sentences.is_empty() {
        // Fallback: truncate with word boundary
        return truncate_at_word_boundary(text, target_chars);
    }

    // Score sentences by position and keyword density
    let keywords = extract_keywords(text, 10);
    let keyword_set: std::collections::HashSet<&str> =
        keywords.iter().map(|s| s.as_str()).collect();

    let mut scored_sentences: Vec<(usize, f32, &str)> = sentences
        .iter()
        .enumerate()
        .map(|(idx, sentence)| {
            let mut score = 0.0_f32;

            // Position score: first and last sentences are often important
            if idx == 0 {
                score += 2.0;
            } else if idx == sentences.len() - 1 {
                score += 1.5;
            }

            // Keyword density score
            let words: Vec<&str> = sentence.split_whitespace().collect();
            let keyword_count = words
                .iter()
                .filter(|w| {
                    let cleaned: String = w
                        .chars()
                        .filter(|c| c.is_alphanumeric())
                        .collect::<String>()
                        .to_lowercase();
                    keyword_set.contains(cleaned.as_str())
                })
                .count();

            if !words.is_empty() {
                score += (keyword_count as f32 / words.len() as f32) * 3.0;
            }

            // Length penalty for very short sentences
            if sentence.len() < 20 {
                score -= 0.5;
            }

            (idx, score, *sentence)
        })
        .collect();

    // Sort by score (descending)
    scored_sentences.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Collect sentences until we hit target length, maintaining original order
    let mut selected_indices: Vec<usize> = Vec::new();
    let mut total_len = 0;

    for (idx, _, sentence) in &scored_sentences {
        let sentence_len = sentence.len() + 2; // +2 for ". "
        if total_len + sentence_len > target_chars && !selected_indices.is_empty() {
            break;
        }
        selected_indices.push(*idx);
        total_len += sentence_len;
    }

    // Sort by original position to maintain text flow
    selected_indices.sort();

    // Reconstruct text
    let result: Vec<&str> = selected_indices
        .iter()
        .filter_map(|&idx| sentences.get(idx).copied())
        .collect();

    if result.is_empty() {
        truncate_at_word_boundary(text, target_chars)
    } else {
        result.join(". ") + "."
    }
}

/// Create an outer summary from middle content and keywords
fn create_outer_summary(middle_content: &str, keywords: &[String], target_chars: usize) -> String {
    // Start with top keywords
    let keyword_prefix = if !keywords.is_empty() {
        format!(
            "[{}] ",
            keywords
                .iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        )
    } else {
        String::new()
    };

    let remaining_chars = target_chars.saturating_sub(keyword_prefix.len());

    // Take first sentence or truncate
    let first_sentence = middle_content
        .split(['.', '!', '?'])
        .next()
        .unwrap_or(middle_content)
        .trim();

    let summary = if first_sentence.len() <= remaining_chars {
        first_sentence.to_string()
    } else {
        truncate_at_word_boundary(first_sentence, remaining_chars)
    };

    format!("{}{}", keyword_prefix, summary)
}

/// Truncate text at a word boundary
fn truncate_at_word_boundary(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        return text.to_string();
    }

    // Find the last space before max_chars
    let truncated = &text[..max_chars];
    if let Some(last_space) = truncated.rfind(' ') {
        format!("{}...", &text[..last_space])
    } else {
        format!("{}...", truncated)
    }
}

pub struct RAGPipeline {
    mlx_bridge: Arc<Mutex<MLXBridge>>,
    storage: Arc<StorageManager>,
}

impl RAGPipeline {
    /// Create new RAGPipeline with MLXBridge (required, no fallback!)
    pub async fn new(
        mlx_bridge: Arc<Mutex<MLXBridge>>,
        storage: Arc<StorageManager>,
    ) -> Result<Self> {
        Ok(Self {
            mlx_bridge,
            storage,
        })
    }

    pub fn storage(&self) -> Arc<StorageManager> {
        self.storage.clone()
    }

    /// Get which MLX server we're connected to (for health/status reporting)
    pub fn mlx_connected_to(&self) -> String {
        // This is safe because mlx_bridge is required and always initialized
        if let Ok(bridge) = self.mlx_bridge.try_lock() {
            bridge.connected_to().to_string()
        } else {
            "mlx (lock held)".to_string()
        }
    }

    pub async fn index_document(&self, path: &Path, namespace: Option<&str>) -> Result<()> {
        self.index_document_with_mode(path, namespace, SliceMode::default())
            .await
    }

    /// Index a document with explicit slice mode
    pub async fn index_document_with_mode(
        &self,
        path: &Path,
        namespace: Option<&str>,
        slice_mode: SliceMode,
    ) -> Result<()> {
        self.index_document_internal(path, namespace, None, slice_mode)
            .await
    }

    /// Index a document with optional preprocessing to filter noise
    pub async fn index_document_with_preprocessing(
        &self,
        path: &Path,
        namespace: Option<&str>,
        preprocess_config: PreprocessingConfig,
    ) -> Result<()> {
        self.index_document_internal(path, namespace, Some(preprocess_config), SliceMode::Flat)
            .await
    }

    /// Index a document with deduplication (skips if exact content already exists)
    pub async fn index_document_with_dedup(
        &self,
        path: &Path,
        namespace: Option<&str>,
        slice_mode: SliceMode,
    ) -> Result<IndexResult> {
        let text = self.extract_text(path).await?;
        let ns = namespace.unwrap_or(DEFAULT_NAMESPACE);

        // Compute content hash BEFORE any processing
        let content_hash = compute_content_hash(&text);

        // Check if this exact content already exists
        if self.storage.has_content_hash(ns, &content_hash).await? {
            debug!(
                "Skipping duplicate content: {} (hash: {})",
                path.display(),
                &content_hash[..16]
            );
            return Ok(IndexResult::Skipped {
                reason: "exact duplicate".to_string(),
                content_hash,
            });
        }

        let base_metadata = json!({
            "path": path.to_str(),
            "slice_mode": match slice_mode {
                SliceMode::Onion => "onion",
                SliceMode::Flat => "flat",
            },
            "content_hash": &content_hash,
        });

        let chunks_indexed = match slice_mode {
            SliceMode::Onion => {
                self.index_with_onion_slicing_and_hash(&text, ns, base_metadata, &content_hash)
                    .await?
            }
            SliceMode::Flat => {
                self.index_with_flat_chunking_and_hash(
                    &text,
                    ns,
                    path,
                    base_metadata,
                    &content_hash,
                )
                .await?
            }
        };

        Ok(IndexResult::Indexed {
            chunks_indexed,
            content_hash,
        })
    }

    /// Index a document with preprocessing and deduplication
    pub async fn index_document_with_preprocessing_and_dedup(
        &self,
        path: &Path,
        namespace: Option<&str>,
        preprocess_config: PreprocessingConfig,
    ) -> Result<IndexResult> {
        let text = self.extract_text(path).await?;
        let ns = namespace.unwrap_or(DEFAULT_NAMESPACE);

        // Compute content hash BEFORE preprocessing (hash original content)
        let content_hash = compute_content_hash(&text);

        // Check if this exact content already exists
        if self.storage.has_content_hash(ns, &content_hash).await? {
            debug!(
                "Skipping duplicate content: {} (hash: {})",
                path.display(),
                &content_hash[..16]
            );
            return Ok(IndexResult::Skipped {
                reason: "exact duplicate".to_string(),
                content_hash,
            });
        }

        // Now preprocess for indexing
        let preprocessor = Preprocessor::new(preprocess_config);
        let cleaned = preprocessor.extract_semantic_content(&text);
        tracing::info!(
            "Preprocessing: {} chars -> {} chars ({:.1}% reduction)",
            text.len(),
            cleaned.len(),
            (1.0 - (cleaned.len() as f32 / text.len() as f32)) * 100.0
        );

        let base_metadata = json!({
            "path": path.to_str(),
            "slice_mode": "flat",
            "content_hash": &content_hash,
        });

        let chunks_indexed = self
            .index_with_flat_chunking_and_hash(&cleaned, ns, path, base_metadata, &content_hash)
            .await?;

        Ok(IndexResult::Indexed {
            chunks_indexed,
            content_hash,
        })
    }

    async fn index_document_internal(
        &self,
        path: &Path,
        namespace: Option<&str>,
        preprocess_config: Option<PreprocessingConfig>,
        slice_mode: SliceMode,
    ) -> Result<()> {
        let text = self.extract_text(path).await?;

        // Optionally preprocess the text to remove noise
        let text = if let Some(config) = preprocess_config {
            let preprocessor = Preprocessor::new(config);
            let cleaned = preprocessor.extract_semantic_content(&text);
            tracing::info!(
                "Preprocessing: {} chars -> {} chars ({:.1}% reduction)",
                text.len(),
                cleaned.len(),
                (1.0 - (cleaned.len() as f32 / text.len() as f32)) * 100.0
            );
            cleaned
        } else {
            text
        };

        let ns = namespace.unwrap_or(DEFAULT_NAMESPACE);
        let base_metadata = json!({
            "path": path.to_str(),
            "slice_mode": match slice_mode {
                SliceMode::Onion => "onion",
                SliceMode::Flat => "flat",
            }
        });

        match slice_mode {
            SliceMode::Onion => {
                self.index_with_onion_slicing(&text, ns, base_metadata)
                    .await
            }
            SliceMode::Flat => {
                self.index_with_flat_chunking(&text, ns, path, base_metadata)
                    .await
            }
        }
    }

    /// Index using onion slice architecture (hierarchical embeddings)
    async fn index_with_onion_slicing(
        &self,
        text: &str,
        namespace: &str,
        base_metadata: serde_json::Value,
    ) -> Result<()> {
        let config = OnionSliceConfig::default();
        let slices = create_onion_slices(text, &base_metadata, &config);
        let total_slices = slices.len();

        tracing::info!(
            "Onion slicing: {} chars -> {} slices (outer/middle/inner/core)",
            text.len(),
            total_slices
        );

        // Process in batches to avoid RAM explosion for large files
        let mut total_stored = 0;
        for batch in slices.chunks(STORAGE_BATCH_SIZE) {
            // Embed this batch
            let batch_contents: Vec<String> = batch.iter().map(|s| s.content.clone()).collect();
            let embeddings = self.embed_chunks(&batch_contents).await?;

            // Create documents from this batch
            let mut batch_docs = Vec::with_capacity(batch.len());
            for (slice, embedding) in batch.iter().zip(embeddings.iter()) {
                let mut metadata = base_metadata.clone();
                if let serde_json::Value::Object(ref mut map) = metadata {
                    map.insert("layer".to_string(), json!(slice.layer.name()));
                    map.insert("keywords".to_string(), json!(slice.keywords));
                }

                let doc = ChromaDocument::from_onion_slice(
                    slice,
                    namespace.to_string(),
                    embedding.clone(),
                    metadata,
                );
                batch_docs.push(doc);
            }

            // Flush this batch to storage
            self.storage.add_to_store(batch_docs).await?;
            total_stored += batch.len();
            tracing::info!("Stored {}/{} slices", total_stored, total_slices);
        }

        Ok(())
    }

    /// Index using onion slice architecture with content hash for deduplication
    async fn index_with_onion_slicing_and_hash(
        &self,
        text: &str,
        namespace: &str,
        base_metadata: serde_json::Value,
        content_hash: &str,
    ) -> Result<usize> {
        let config = OnionSliceConfig::default();
        let slices = create_onion_slices(text, &base_metadata, &config);
        let total_slices = slices.len();

        tracing::info!(
            "Onion slicing: {} chars -> {} slices (outer/middle/inner/core)",
            text.len(),
            total_slices
        );

        // Process in batches to avoid RAM explosion for large files
        let mut total_stored = 0;
        for batch in slices.chunks(STORAGE_BATCH_SIZE) {
            // Embed this batch
            let batch_contents: Vec<String> = batch.iter().map(|s| s.content.clone()).collect();
            let embeddings = self.embed_chunks(&batch_contents).await?;

            // Create documents from this batch with content hash
            let mut batch_docs = Vec::with_capacity(batch.len());
            for (slice, embedding) in batch.iter().zip(embeddings.iter()) {
                let mut metadata = base_metadata.clone();
                if let serde_json::Value::Object(ref mut map) = metadata {
                    map.insert("layer".to_string(), json!(slice.layer.name()));
                    map.insert("keywords".to_string(), json!(slice.keywords));
                }

                let doc = ChromaDocument::from_onion_slice_with_hash(
                    slice,
                    namespace.to_string(),
                    embedding.clone(),
                    metadata,
                    content_hash.to_string(),
                );
                batch_docs.push(doc);
            }

            // Flush this batch to storage
            self.storage.add_to_store(batch_docs).await?;
            total_stored += batch.len();
            tracing::info!("Stored {}/{} slices", total_stored, total_slices);
        }

        Ok(total_slices)
    }

    /// Index using traditional flat chunking (backward compatible)
    async fn index_with_flat_chunking(
        &self,
        text: &str,
        namespace: &str,
        path: &Path,
        base_metadata: serde_json::Value,
    ) -> Result<()> {
        // Chunk the text
        let chunks = self.chunk_text(text, 512, 128)?;
        let total_chunks = chunks.len();

        tracing::info!(
            "Flat chunking: {} chars -> {} chunks",
            text.len(),
            total_chunks
        );

        // Process in batches to avoid RAM explosion for large files
        let mut total_stored = 0;
        let mut global_idx = 0;
        for batch in chunks.chunks(STORAGE_BATCH_SIZE) {
            // Embed this batch
            let embeddings = self.embed_chunks(batch).await?;

            // Create documents from this batch
            let mut batch_docs = Vec::with_capacity(batch.len());
            for (chunk, embedding) in batch.iter().zip(embeddings.iter()) {
                let mut metadata = base_metadata.clone();
                if let serde_json::Value::Object(ref mut map) = metadata {
                    map.insert("chunk_index".to_string(), json!(global_idx));
                    map.insert("total_chunks".to_string(), json!(total_chunks));
                }

                let doc = ChromaDocument::new_flat(
                    format!("{}_{}", path.to_str().unwrap_or("unknown"), global_idx),
                    namespace.to_string(),
                    embedding.clone(),
                    metadata,
                    chunk.clone(),
                );
                batch_docs.push(doc);
                global_idx += 1;
            }

            // Flush this batch to storage
            self.storage.add_to_store(batch_docs).await?;
            total_stored += batch.len();
            tracing::info!("Stored {}/{} chunks", total_stored, total_chunks);
        }

        Ok(())
    }

    /// Index using traditional flat chunking with content hash for deduplication
    async fn index_with_flat_chunking_and_hash(
        &self,
        text: &str,
        namespace: &str,
        path: &Path,
        base_metadata: serde_json::Value,
        content_hash: &str,
    ) -> Result<usize> {
        // Chunk the text
        let chunks = self.chunk_text(text, 512, 128)?;
        let total_chunks = chunks.len();

        tracing::info!(
            "Flat chunking: {} chars -> {} chunks",
            text.len(),
            total_chunks
        );

        // Process in batches to avoid RAM explosion for large files
        let mut total_stored = 0;
        let mut global_idx = 0;
        for batch in chunks.chunks(STORAGE_BATCH_SIZE) {
            // Embed this batch
            let embeddings = self.embed_chunks(batch).await?;

            // Create documents from this batch with content hash
            let mut batch_docs = Vec::with_capacity(batch.len());
            for (chunk, embedding) in batch.iter().zip(embeddings.iter()) {
                let mut metadata = base_metadata.clone();
                if let serde_json::Value::Object(ref mut map) = metadata {
                    map.insert("chunk_index".to_string(), json!(global_idx));
                    map.insert("total_chunks".to_string(), json!(total_chunks));
                }

                let doc = ChromaDocument::new_flat_with_hash(
                    format!("{}_{}", path.to_str().unwrap_or("unknown"), global_idx),
                    namespace.to_string(),
                    embedding.clone(),
                    metadata,
                    chunk.clone(),
                    content_hash.to_string(),
                );
                batch_docs.push(doc);
                global_idx += 1;
            }

            // Flush this batch to storage
            self.storage.add_to_store(batch_docs).await?;
            total_stored += batch.len();
            tracing::info!("Stored {}/{} chunks", total_stored, total_chunks);
        }

        Ok(total_chunks)
    }

    pub async fn index_text(
        &self,
        namespace: Option<&str>,
        id: String,
        text: String,
        metadata: serde_json::Value,
    ) -> Result<String> {
        self.index_text_with_mode(namespace, id, text, metadata, SliceMode::default())
            .await
    }

    /// Index text with explicit slice mode
    pub async fn index_text_with_mode(
        &self,
        namespace: Option<&str>,
        id: String,
        text: String,
        metadata: serde_json::Value,
        slice_mode: SliceMode,
    ) -> Result<String> {
        let ns = namespace.unwrap_or(DEFAULT_NAMESPACE).to_string();

        match slice_mode {
            SliceMode::Onion => {
                // For onion mode, ignore the provided ID and use generated slice IDs
                let config = OnionSliceConfig::default();
                let slices = create_onion_slices(&text, &metadata, &config);

                let slice_contents: Vec<String> =
                    slices.iter().map(|s| s.content.clone()).collect();
                let embeddings = self.embed_chunks(&slice_contents).await?;

                let mut documents = Vec::with_capacity(slices.len());
                for (slice, embedding) in slices.iter().zip(embeddings.iter()) {
                    let mut meta = metadata.clone();
                    if let serde_json::Value::Object(ref mut map) = meta {
                        map.insert("layer".to_string(), json!(slice.layer.name()));
                        map.insert("original_id".to_string(), json!(id));
                    }

                    let doc = ChromaDocument::from_onion_slice(
                        slice,
                        ns.clone(),
                        embedding.clone(),
                        meta,
                    );
                    documents.push(doc);
                }

                self.storage.add_to_store(documents).await?;

                // Return the outer slice ID (what search will hit first)
                Ok(slices
                    .iter()
                    .find(|s| s.layer == SliceLayer::Outer)
                    .map(|s| s.id.clone())
                    .unwrap_or(id))
            }
            SliceMode::Flat => {
                let embedding = self.embed_query(&text).await?;
                let doc = ChromaDocument::new_flat(id.clone(), ns, embedding, metadata, text);
                self.storage.add_to_store(vec![doc]).await?;
                Ok(id)
            }
        }
    }

    pub async fn memory_upsert(
        &self,
        namespace: &str,
        id: String,
        text: String,
        metadata: serde_json::Value,
    ) -> Result<()> {
        self.index_text(Some(namespace), id, text, metadata).await?;
        Ok(())
    }

    pub async fn memory_get(&self, namespace: &str, id: &str) -> Result<Option<SearchResult>> {
        if let Some(doc) = self.storage.get_document(namespace, id).await? {
            let layer = doc.slice_layer();
            return Ok(Some(SearchResult {
                id: doc.id,
                namespace: doc.namespace,
                text: doc.document,
                score: 1.0,
                metadata: doc.metadata,
                layer,
                parent_id: doc.parent_id,
                children_ids: doc.children_ids,
                keywords: doc.keywords,
            }));
        }
        Ok(None)
    }

    pub async fn memory_delete(&self, namespace: &str, id: &str) -> Result<usize> {
        self.storage.delete_document(namespace, id).await
    }

    pub async fn purge_namespace(&self, namespace: &str) -> Result<usize> {
        self.storage.purge_namespace(namespace).await
    }

    pub async fn memory_search(
        &self,
        namespace: &str,
        query: &str,
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        self.search_with_options(Some(namespace), query, k, SearchOptions::default())
            .await
    }

    /// Search with layer filter - returns only outer slices by default (efficient context usage)
    pub async fn memory_search_with_layer(
        &self,
        namespace: &str,
        query: &str,
        k: usize,
        layer: Option<SliceLayer>,
    ) -> Result<Vec<SearchResult>> {
        self.search_with_options(
            Some(namespace),
            query,
            k,
            SearchOptions {
                layer_filter: layer,
            },
        )
        .await
    }

    pub async fn search(&self, query: &str, k: usize) -> Result<Vec<SearchResult>> {
        self.search_inner(None, query, k).await
    }

    /// Legacy search method for backward compatibility
    pub async fn search_inner(
        &self,
        namespace: Option<&str>,
        query: &str,
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        self.search_with_options(namespace, query, k, SearchOptions::default())
            .await
    }

    /// Search with full options including layer filtering
    pub async fn search_with_options(
        &self,
        namespace: Option<&str>,
        query: &str,
        k: usize,
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        let query_embedding = self.embed_query(query).await?;

        let candidates = self
            .storage
            .search_store_with_layer(
                namespace,
                query_embedding.clone(),
                k * 3,
                options.layer_filter,
            )
            .await?;

        // Rerank if we have candidates
        if !candidates.is_empty() {
            let documents: Vec<String> = candidates.iter().map(|c| c.document.clone()).collect();
            let metadatas: Vec<serde_json::Value> =
                candidates.iter().map(|c| c.metadata.clone()).collect();

            // Try MLX reranker; fallback to cosine if rerank fails
            let reranked = match self.mlx_bridge.lock().await.rerank(query, &documents).await {
                Ok(r) => Some(r),
                Err(e) => {
                    tracing::warn!("MLX rerank failed, using cosine fallback: {}", e);
                    None
                }
            };

            let reranked = if let Some(r) = reranked {
                r
            } else {
                // Cosine fallback
                let doc_embeddings = self.ensure_doc_embeddings(&documents, &candidates).await?;
                let scores = doc_embeddings
                    .iter()
                    .enumerate()
                    .map(|(idx, emb)| (idx, cosine(&query_embedding, emb)))
                    .collect::<Vec<_>>();
                let mut scores = scores;
                scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                scores
            };

            // Return top-k reranked results with onion slice info
            let results: Vec<SearchResult> = reranked
                .into_iter()
                .take(k)
                .filter_map(|(idx, score)| {
                    candidates.get(idx).map(|candidate| {
                        SearchResult {
                            id: candidate.id.clone(),
                            namespace: candidate.namespace.clone(),
                            text: candidate.document.clone(),
                            score,
                            metadata: metadatas.get(idx).cloned().unwrap_or_else(|| json!({})),
                            // Onion slice fields
                            layer: candidate.slice_layer(),
                            parent_id: candidate.parent_id.clone(),
                            children_ids: candidate.children_ids.clone(),
                            keywords: candidate.keywords.clone(),
                        }
                    })
                })
                .collect();

            return Ok(results);
        }

        Ok(vec![])
    }

    /// Expand a search result to get its children (drill down in onion hierarchy)
    pub async fn expand_result(&self, namespace: &str, id: &str) -> Result<Vec<SearchResult>> {
        let children = self.storage.get_children(namespace, id).await?;
        Ok(children
            .into_iter()
            .map(|doc| {
                let layer = doc.slice_layer();
                SearchResult {
                    id: doc.id,
                    namespace: doc.namespace,
                    text: doc.document,
                    score: 1.0,
                    metadata: doc.metadata,
                    layer,
                    parent_id: doc.parent_id,
                    children_ids: doc.children_ids,
                    keywords: doc.keywords,
                }
            })
            .collect())
    }

    /// Get the parent of a search result (drill up in onion hierarchy)
    pub async fn get_parent_result(
        &self,
        namespace: &str,
        id: &str,
    ) -> Result<Option<SearchResult>> {
        if let Some(parent) = self.storage.get_parent(namespace, id).await? {
            let layer = parent.slice_layer();
            return Ok(Some(SearchResult {
                id: parent.id,
                namespace: parent.namespace,
                text: parent.document,
                score: 1.0,
                metadata: parent.metadata,
                layer,
                parent_id: parent.parent_id,
                children_ids: parent.children_ids,
                keywords: parent.keywords,
            }));
        }
        Ok(None)
    }

    async fn extract_text(&self, path: &Path) -> Result<String> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        if ext == "pdf" {
            // pdf_extract is blocking; offload to blocking thread
            let path = path.to_path_buf();
            let pdf_text =
                tokio::task::spawn_blocking(move || pdf_extract::extract_text(&path)).await??;
            return Ok(pdf_text);
        }

        // Default: treat as UTF-8 text
        // Path is validated by caller (handlers::validate_path) before reaching this private method
        // nosemgrep: rust.actix.path-traversal.tainted-path.tainted-path
        tokio::fs::read_to_string(path).await.map_err(|e| e.into())
    }

    async fn embed_chunks(&self, chunks: &[String]) -> Result<Vec<Vec<f32>>> {
        // Use MLX for all embeddings (no FastEmbed fallback!)
        self.mlx_bridge.lock().await.embed_batch(chunks).await
    }

    async fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        self.mlx_bridge.lock().await.embed(query).await
    }

    async fn ensure_doc_embeddings(
        &self,
        documents: &[String],
        candidates: &[ChromaDocument],
    ) -> Result<Vec<Vec<f32>>> {
        // If storage returned embeddings, use them; otherwise embed via MLX
        let has_all = candidates.iter().all(|c| !c.embedding.is_empty());
        if has_all {
            return Ok(candidates.iter().map(|c| c.embedding.clone()).collect());
        }

        self.mlx_bridge.lock().await.embed_batch(documents).await
    }

    fn chunk_text(&self, text: &str, chunk_size: usize, overlap: usize) -> Result<Vec<String>> {
        let mut chunks = Vec::new();
        let chars: Vec<char> = text.chars().collect();

        let mut start = 0;
        while start < chars.len() {
            let end = (start + chunk_size).min(chars.len());
            let chunk: String = chars[start..end].iter().collect();
            chunks.push(chunk);

            if end >= chars.len() {
                break;
            }

            start = end - overlap;
        }

        Ok(chunks)
    }
}

/// Options for search operations
#[derive(Debug, Clone, Default)]
pub struct SearchOptions {
    /// Filter by onion slice layer (None = all layers)
    pub layer_filter: Option<SliceLayer>,
}

impl SearchOptions {
    /// Search only outer slices (default for onion mode - minimum context, maximum navigation)
    pub fn outer_only() -> Self {
        Self {
            layer_filter: Some(SliceLayer::Outer),
        }
    }

    /// Deep search - include all layers including Core
    pub fn deep() -> Self {
        Self { layer_filter: None }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SearchResult {
    pub id: String,
    pub namespace: String,
    pub text: String,
    pub score: f32,
    pub metadata: serde_json::Value,
    /// Onion slice layer (None for legacy flat chunks)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer: Option<SliceLayer>,
    /// Parent slice ID for drilling up in hierarchy
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    /// Children slice IDs for drilling down in hierarchy
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub children_ids: Vec<String>,
    /// Keywords extracted from this slice
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub keywords: Vec<String>,
}

impl SearchResult {
    /// Create a legacy result without onion slice fields
    pub fn new_legacy(
        id: String,
        namespace: String,
        text: String,
        score: f32,
        metadata: serde_json::Value,
    ) -> Self {
        Self {
            id,
            namespace,
            text,
            score,
            metadata,
            layer: None,
            parent_id: None,
            children_ids: vec![],
            keywords: vec![],
        }
    }

    /// Check if this result can be expanded (has children)
    pub fn can_expand(&self) -> bool {
        !self.children_ids.is_empty()
    }

    /// Check if this result has a parent to drill up to
    pub fn can_drill_up(&self) -> bool {
        self.parent_id.is_some()
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}
