//! BM25 keyword search using Tantivy.
//!
//! Provides exact keyword matching to complement vector similarity search.
//! This helps distinguish between semantically similar but distinct terms
//! like "smutny" (sad) and "melancholijny" (melancholic).

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tantivy::{
    Index, IndexReader, IndexWriter, TantivyDocument,
    collector::TopDocs,
    query::QueryParser,
    schema::{
        Field, IndexRecordOption, STORED, Schema, TEXT, TextFieldIndexing, TextOptions, Value,
    },
    tokenizer::{Language, LowerCaser, RemoveLongFilter, SimpleTokenizer, Stemmer, TextAnalyzer},
};
use tokio::sync::Mutex;

/// Supported languages for stemming
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StemLanguage {
    #[default]
    English,
    German,
    French,
    Spanish,
    Italian,
    Portuguese,
    Russian,
    /// No stemming (for unsupported languages like Polish)
    None,
}

impl StemLanguage {
    fn to_tantivy_language(self) -> Option<Language> {
        match self {
            StemLanguage::English => Some(Language::English),
            StemLanguage::German => Some(Language::German),
            StemLanguage::French => Some(Language::French),
            StemLanguage::Spanish => Some(Language::Spanish),
            StemLanguage::Italian => Some(Language::Italian),
            StemLanguage::Portuguese => Some(Language::Portuguese),
            StemLanguage::Russian => Some(Language::Russian),
            StemLanguage::None => None,
        }
    }
}

/// Configuration for BM25 index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BM25Config {
    /// Path to store the Tantivy index
    #[serde(default = "default_bm25_path")]
    pub index_path: String,
    /// Heap size for index writer (bytes)
    #[serde(default = "default_heap_size")]
    pub writer_heap_size: usize,
    /// Enable stemming for better recall
    #[serde(default = "default_true")]
    pub enable_stemming: bool,
    /// Language for stemming
    #[serde(default)]
    pub language: StemLanguage,
}

fn default_bm25_path() -> String {
    "~/.rmcp-servers/rmcp-memex/bm25".to_string()
}

fn default_heap_size() -> usize {
    50_000_000
}

fn default_true() -> bool {
    true
}

impl Default for BM25Config {
    fn default() -> Self {
        Self {
            index_path: default_bm25_path(),
            writer_heap_size: default_heap_size(),
            enable_stemming: true,
            language: StemLanguage::English,
        }
    }
}

impl BM25Config {
    /// Create config for multilingual content (no stemming)
    pub fn multilingual() -> Self {
        Self {
            language: StemLanguage::None,
            enable_stemming: false,
            ..Self::default()
        }
    }

    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        self.index_path = path.into();
        self
    }
}

/// BM25 keyword search index using Tantivy
pub struct BM25Index {
    index: Index,
    reader: IndexReader,
    writer: Arc<Mutex<IndexWriter>>,
    content_field: Field,
    id_field: Field,
    namespace_field: Field,
}

impl BM25Index {
    /// Create or open a BM25 index at the given path
    pub fn new(config: &BM25Config) -> Result<Self> {
        let path = crate::path_utils::sanitize_new_path(&config.index_path)?;

        // Create directory if it doesn't exist
        if !path.exists() {
            std::fs::create_dir_all(&path)?;
        }

        // Build schema with text analysis
        let mut schema_builder = Schema::builder();

        // Configure text field with proper tokenization
        let text_options = TextOptions::default()
            .set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("custom_tokenizer")
                    .set_index_option(IndexRecordOption::WithFreqsAndPositions),
            )
            .set_stored();

        let content_field = schema_builder.add_text_field("content", text_options);
        let id_field = schema_builder.add_text_field("id", STORED);
        let namespace_field = schema_builder.add_text_field("namespace", TEXT | STORED);

        let schema = schema_builder.build();

        // Open or create index
        let index = if path.join("meta.json").exists() {
            Index::open_in_dir(path)?
        } else {
            Index::create_in_dir(path, schema.clone())?
        };

        // Register custom tokenizer with optional stemming
        let tokenizer = if config.enable_stemming {
            if let Some(lang) = config.language.to_tantivy_language() {
                TextAnalyzer::builder(SimpleTokenizer::default())
                    .filter(RemoveLongFilter::limit(40))
                    .filter(LowerCaser)
                    .filter(Stemmer::new(lang))
                    .build()
            } else {
                // No stemming for unsupported languages
                TextAnalyzer::builder(SimpleTokenizer::default())
                    .filter(RemoveLongFilter::limit(40))
                    .filter(LowerCaser)
                    .build()
            }
        } else {
            TextAnalyzer::builder(SimpleTokenizer::default())
                .filter(RemoveLongFilter::limit(40))
                .filter(LowerCaser)
                .build()
        };

        index.tokenizers().register("custom_tokenizer", tokenizer);

        let reader = index.reader()?;
        let writer = index.writer(config.writer_heap_size)?;

        Ok(Self {
            index,
            reader,
            writer: Arc::new(Mutex::new(writer)),
            content_field,
            id_field,
            namespace_field,
        })
    }

    /// Add documents to the BM25 index
    ///
    /// # Arguments
    /// * `docs` - List of (id, namespace, content) tuples
    pub async fn add_documents(&self, docs: &[(String, String, String)]) -> Result<()> {
        let mut writer = self.writer.lock().await;

        for (id, namespace, content) in docs {
            let mut doc = TantivyDocument::new();
            doc.add_text(self.content_field, content);
            doc.add_text(self.id_field, id);
            doc.add_text(self.namespace_field, namespace);
            writer.add_document(doc)?;
        }

        writer.commit()?;
        self.reader.reload()?;

        tracing::debug!("Added {} documents to BM25 index", docs.len());
        Ok(())
    }

    /// Search the BM25 index
    ///
    /// # Arguments
    /// * `query` - Search query string
    /// * `namespace` - Optional namespace filter
    /// * `limit` - Maximum number of results
    ///
    /// # Returns
    /// Vector of (document_id, score) tuples, sorted by score descending
    pub fn search(
        &self,
        query: &str,
        namespace: Option<&str>,
        limit: usize,
    ) -> Result<Vec<(String, f32)>> {
        let searcher = self.reader.searcher();

        // Build query - search in content field
        let query_parser = QueryParser::for_index(&self.index, vec![self.content_field]);

        // Escape special characters and parse query
        let escaped_query = Self::escape_query(query);
        let parsed_query = query_parser
            .parse_query(&escaped_query)
            .map_err(|e| anyhow!("Query parse error: {}", e))?;

        // Execute search
        let top_docs = searcher.search(&parsed_query, &TopDocs::with_limit(limit * 2))?;

        let mut results = Vec::with_capacity(limit);

        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_address)?;

            // Get document ID using Value trait
            let id = doc
                .get_first(self.id_field)
                .and_then(|v| Value::as_str(&v).map(|s| s.to_string()))
                .ok_or_else(|| anyhow!("Document missing ID field"))?;

            // Filter by namespace if specified
            if let Some(ns) = namespace {
                let doc_ns = doc
                    .get_first(self.namespace_field)
                    .and_then(|v| Value::as_str(&v))
                    .unwrap_or("");
                if doc_ns != ns {
                    continue;
                }
            }

            results.push((id, score));

            if results.len() >= limit {
                break;
            }
        }

        tracing::debug!("BM25 search '{}' returned {} results", query, results.len());

        Ok(results)
    }

    /// Delete documents by ID
    pub async fn delete_documents(&self, ids: &[String]) -> Result<usize> {
        let mut writer = self.writer.lock().await;
        let mut deleted = 0;

        for id in ids {
            // Delete by term query on id field
            let term = tantivy::Term::from_field_text(self.id_field, id);
            writer.delete_term(term);
            deleted += 1;
        }

        writer.commit()?;
        self.reader.reload()?;

        Ok(deleted)
    }

    /// Delete all documents in a namespace
    pub async fn purge_namespace(&self, namespace: &str) -> Result<usize> {
        let mut writer = self.writer.lock().await;

        let term = tantivy::Term::from_field_text(self.namespace_field, namespace);
        writer.delete_term(term);
        writer.commit()?;
        self.reader.reload()?;

        tracing::info!("Purged namespace '{}' from BM25 index", namespace);
        Ok(1) // Tantivy doesn't return exact count for term deletes
    }

    /// Escape special query characters
    fn escape_query(query: &str) -> String {
        // Tantivy query syntax special characters
        let special_chars = [
            '+', '-', '&', '|', '!', '(', ')', '{', '}', '[', ']', '^', '"', '~', '*', '?', ':',
            '\\', '/',
        ];

        let mut escaped = String::with_capacity(query.len() * 2);
        for c in query.chars() {
            if special_chars.contains(&c) {
                escaped.push('\\');
            }
            escaped.push(c);
        }
        escaped
    }

    /// Get document count in index
    pub fn doc_count(&self) -> u64 {
        let searcher = self.reader.searcher();
        searcher.num_docs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_bm25_basic() {
        let temp_dir = TempDir::new().unwrap();
        let config = BM25Config::default().with_path(temp_dir.path().to_str().unwrap());

        let index = BM25Index::new(&config).unwrap();

        // Add some documents
        let docs = vec![
            (
                "doc1".to_string(),
                "test".to_string(),
                "The quick brown fox jumps over the lazy dog".to_string(),
            ),
            (
                "doc2".to_string(),
                "test".to_string(),
                "A quick brown dog runs in the park".to_string(),
            ),
            (
                "doc3".to_string(),
                "test".to_string(),
                "The lazy cat sleeps all day".to_string(),
            ),
        ];

        index.add_documents(&docs).await.unwrap();

        // Search
        let results = index.search("quick brown", None, 10).unwrap();

        assert_eq!(results.len(), 2);
        // doc1 and doc2 should match, doc3 should not
        let ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
        assert!(ids.contains(&"doc1"));
        assert!(ids.contains(&"doc2"));
    }

    #[tokio::test]
    async fn test_bm25_namespace_filter() {
        let temp_dir = TempDir::new().unwrap();
        let config = BM25Config::default().with_path(temp_dir.path().to_str().unwrap());

        let index = BM25Index::new(&config).unwrap();

        let docs = vec![
            (
                "doc1".to_string(),
                "ns1".to_string(),
                "hello world".to_string(),
            ),
            (
                "doc2".to_string(),
                "ns2".to_string(),
                "hello universe".to_string(),
            ),
        ];

        index.add_documents(&docs).await.unwrap();

        // Search with namespace filter
        let results = index.search("hello", Some("ns1"), 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "doc1");
    }

    #[test]
    fn test_escape_query() {
        assert_eq!(BM25Index::escape_query("hello world"), "hello world");
        assert_eq!(BM25Index::escape_query("hello+world"), "hello\\+world");
        assert_eq!(BM25Index::escape_query("test:query"), "test\\:query");
    }
}
