use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SseEvent {
    pub event: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub data: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ReindexProgress {
    pub namespace: String,
    pub total_files: usize,
    pub processed_files: usize,
    pub indexed_files: usize,
    pub skipped_files: usize,
    pub failed_files: usize,
    pub total_chunks: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct MergeProgress {
    pub total_docs: usize,
    pub docs_copied: usize,
    pub docs_skipped: usize,
    pub namespaces: Vec<String>,
    pub sources_processed: usize,
    pub errors: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ReprocessProgress {
    pub source_label: String,
    pub processed_documents: usize,
    pub indexed_documents: usize,
    pub skipped_documents: usize,
    pub failed_documents: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct AuditProgress {
    pub processed_namespaces: usize,
    pub total_namespaces: usize,
    pub current_namespace: Option<String>,
    pub threshold: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct RepairResult {
    pub recovery_dir: String,
    pub pending_batches: usize,
    pub repaired_documents: usize,
    pub skipped_documents: usize,
    pub batches_repaired: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct CompactProgress {
    pub phase: String,
    pub status: String,
    pub description: Option<String>,
    pub files_removed: Option<u64>,
    pub files_added: Option<u64>,
    pub fragments_removed: Option<u64>,
    pub fragments_added: Option<u64>,
    pub old_versions: Option<u64>,
    pub bytes_removed: Option<u64>,
    pub elapsed_ms: Option<u64>,
}
