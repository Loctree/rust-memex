//! Hybrid search combining BM25 keyword matching with vector similarity.
//!
//! This module implements semantic improvements that help distinguish between
//! semantically similar but distinct terms (e.g., "smutny" vs "melancholijny").
//!
//! # Architecture
//!
//! ```text
//! Query -> [BM25 Index] -> keyword matches (score 0-1)
//!       -> [Vector Store] -> semantic matches (score 0-1)
//!       -> [Score Fusion] -> combined ranking
//!       -> [Reranker] -> final precision boost
//! ```
//!
//! # Usage
//!
//! ```toml
//! [search]
//! mode = "hybrid"  # or "vector" for vector-only
//! bm25_weight = 0.4
//! vector_weight = 0.6
//! ```

pub mod bm25;
pub mod hybrid;

pub use bm25::{BM25Config, BM25Index, StemLanguage};
pub use hybrid::{HybridConfig, HybridSearchResult, HybridSearcher, SearchMode};
