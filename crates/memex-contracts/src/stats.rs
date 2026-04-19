use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct NamespaceStats {
    pub name: String,
    pub total_chunks: usize,
    pub layer_counts: HashMap<String, usize>,
    pub top_keywords: Vec<(String, usize)>,
    pub has_timestamps: bool,
    pub earliest_indexed: Option<String>,
    pub latest_indexed: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct DatabaseStats {
    pub row_count: usize,
    pub version_count: usize,
    pub table_name: String,
    pub db_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct StorageMetrics {
    pub total_namespaces: usize,
    pub total_documents: usize,
    pub bytes_used: Option<u64>,
    pub bytes_reclaimed: Option<u64>,
}
