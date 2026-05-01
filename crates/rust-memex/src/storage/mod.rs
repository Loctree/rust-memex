use anyhow::{Result, anyhow};
use arrow_array::types::Float32Type;
use arrow_array::{
    Array, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray,
    UInt8Array,
};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::connection::Connection;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::table::{NewColumnTransform, OptimizeAction, OptimizeStats};
use lancedb::{Table, connect};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::rag::SliceLayer;

/// Schema version for LanceDB tables. Increment when changing table structure.
/// Version 2: Added onion slice fields (layer, parent_id, children_ids, keywords)
/// Version 3: Added content_hash for exact-match deduplication
/// Version 4: Split `content_hash` into per-chunk SHA256 (this column) and a
///            new `source_hash` column for the SHA256 of the source document
///            text. This enables both layer-aware chunk dedup and pre-index
///            source-level dedup. Pre-v4 rows store source-text hash in
///            `content_hash`; backfill via `/admin/backfill-hashes` corrects
///            this without breaking old data.
/// See docs/MIGRATION.md for migration procedures.
pub const SCHEMA_VERSION: u32 = 4;
pub const DEFAULT_TABLE_NAME: &str = "mcp_documents";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemaVersion {
    V3,
    V4,
}

impl SchemaVersion {
    pub fn current() -> Self {
        Self::V4
    }
}

impl fmt::Display for SchemaVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::V3 => f.write_str("v3"),
            Self::V4 => f.write_str("v4"),
        }
    }
}

impl FromStr for SchemaVersion {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "3" | "v3" => Ok(Self::V3),
            "4" | "v4" => Ok(Self::V4),
            other => Err(anyhow!(
                "unsupported schema target '{other}' (expected v3 or v4)"
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SchemaMigrationReport {
    pub target: SchemaVersion,
    pub missing_columns: Vec<Field>,
    pub applied: bool,
}

impl SchemaMigrationReport {
    pub fn missing_column_names(&self) -> Vec<&str> {
        self.missing_columns
            .iter()
            .map(|field| field.name().as_str())
            .collect()
    }
}

pub fn required_columns_for(version: SchemaVersion) -> Vec<Field> {
    let mut fields = vec![Field::new("content_hash", DataType::Utf8, true)];
    if matches!(version, SchemaVersion::V4) {
        fields.push(Field::new("source_hash", DataType::Utf8, true));
    }
    fields
}

#[derive(Debug, Clone)]
pub struct SchemaMismatchWriteError {
    table_name: String,
    db_path: String,
    missing_columns: Vec<String>,
    message: String,
}

impl SchemaMismatchWriteError {
    fn new(
        table_name: impl Into<String>,
        db_path: impl Into<String>,
        missing_columns: Vec<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            table_name: table_name.into(),
            db_path: db_path.into(),
            missing_columns,
            message: message.into(),
        }
    }

    pub fn missing_columns(&self) -> &[String] {
        &self.missing_columns
    }

    pub fn db_path(&self) -> &str {
        &self.db_path
    }

    pub fn table_name(&self) -> &str {
        &self.table_name
    }

    pub fn remediation(&self) -> String {
        format!("rust-memex migrate-schema --db-path {}", self.db_path)
    }
}

impl fmt::Display for SchemaMismatchWriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ERROR schema mismatch while writing Lance table '{}': missing columns {:?}. {}. Remediation: {}",
            self.table_name,
            self.missing_columns,
            self.message,
            self.remediation()
        )
    }
}

impl std::error::Error for SchemaMismatchWriteError {}

fn is_schema_mismatch_message(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    lower.contains("schema mismatch")
        || lower.contains("append with different schema")
        || lower.contains("fields did not match")
        || lower.contains("missing=[")
}

fn extract_missing_columns(message: &str) -> Vec<String> {
    let mut columns = Vec::new();
    let mut rest = message;

    while let Some(start) = rest.find("missing=[") {
        rest = &rest[start + "missing=[".len()..];
        let Some(end) = rest.find(']') else {
            break;
        };
        let list = &rest[..end];
        for item in list.split(',') {
            let column = item
                .trim()
                .trim_matches('`')
                .trim_matches('"')
                .trim_matches('\'');
            if !column.is_empty() && !columns.iter().any(|existing| existing == column) {
                columns.push(column.to_string());
            }
        }
        rest = &rest[end + 1..];
    }

    if columns.is_empty() {
        for field in required_columns_for(SchemaVersion::current()) {
            let name = field.name();
            if message.contains(name) {
                columns.push(name.to_string());
            }
        }
    }

    columns
}

// =============================================================================
// STORAGE BACKEND INTERFACE
// =============================================================================
//
// To add a new storage backend, implement a struct with the following methods:
//
//   async fn add_to_store(&self, documents: Vec<ChromaDocument>) -> Result<()>
//   async fn get_document(&self, namespace: &str, id: &str) -> Result<Option<ChromaDocument>>
//   async fn search(&self, namespace: Option<&str>, embedding: &[f32], k: usize) -> Result<Vec<ChromaDocument>>
//   async fn delete(&self, namespace: &str, id: &str) -> Result<usize>
//   async fn delete_namespace(&self, namespace: &str) -> Result<usize>
//
// Current implementation:
//   - `StorageManager`: LanceDB embedded vector store
//
// Future alternatives to consider:
//   - Qdrant, Milvus, Pinecone (external vector DBs)
//   - SQLite with vector extension
// =============================================================================

#[derive(Debug, Serialize, Clone)]
pub struct ChromaDocument {
    pub id: String,
    pub namespace: String,
    pub embedding: Vec<f32>,
    pub metadata: serde_json::Value,
    pub document: String,
    /// Onion slice layer (1=Outer, 2=Middle, 3=Inner, 4=Core, 0=legacy flat)
    pub layer: u8,
    /// Parent slice ID in the onion hierarchy (None for Core slices)
    pub parent_id: Option<String>,
    /// Children slice IDs in the onion hierarchy
    pub children_ids: Vec<String>,
    /// Extracted keywords for this slice
    pub keywords: Vec<String>,
    /// SHA256 hash of THIS chunk's text. Used for chunk-level deduplication.
    /// Pre-v4 rows may transitionally store the source-text hash here; the
    /// `/admin/backfill-hashes` endpoint corrects them.
    pub content_hash: Option<String>,
    /// SHA256 hash of the SOURCE document text (same value across all four
    /// onion layers from one source). Used for pre-index dedup so we never
    /// re-embed an already-ingested file. Optional for backward compatibility
    /// with v3 schemas — `None` means "unknown source provenance".
    pub source_hash: Option<String>,
}

impl ChromaDocument {
    /// Create a new document with default (legacy) slice values
    pub fn new_flat(
        id: String,
        namespace: String,
        embedding: Vec<f32>,
        metadata: serde_json::Value,
        document: String,
    ) -> Self {
        Self {
            id,
            namespace,
            embedding,
            metadata,
            document,
            layer: 0, // Legacy flat mode
            parent_id: None,
            children_ids: vec![],
            keywords: vec![],
            content_hash: None,
            source_hash: None,
        }
    }

    /// Create a new document with content hash for deduplication.
    /// Source hash is left empty — prefer `new_flat_with_hashes` so callers
    /// supply the source-document hash explicitly when they have it.
    pub fn new_flat_with_hash(
        id: String,
        namespace: String,
        embedding: Vec<f32>,
        metadata: serde_json::Value,
        document: String,
        content_hash: String,
    ) -> Self {
        Self::new_flat_with_hashes(
            id,
            namespace,
            embedding,
            metadata,
            document,
            content_hash,
            None,
        )
    }

    /// Create a flat (legacy) document with both per-chunk and source hashes.
    pub fn new_flat_with_hashes(
        id: String,
        namespace: String,
        embedding: Vec<f32>,
        metadata: serde_json::Value,
        document: String,
        content_hash: String,
        source_hash: Option<String>,
    ) -> Self {
        Self {
            id,
            namespace,
            embedding,
            metadata,
            document,
            layer: 0,
            parent_id: None,
            children_ids: vec![],
            keywords: vec![],
            content_hash: Some(content_hash),
            source_hash,
        }
    }

    /// Create a document from an onion slice
    pub fn from_onion_slice(
        slice: &crate::rag::OnionSlice,
        namespace: String,
        embedding: Vec<f32>,
        metadata: serde_json::Value,
    ) -> Self {
        Self {
            id: slice.id.clone(),
            namespace,
            embedding,
            metadata,
            document: slice.content.clone(),
            layer: slice.layer.as_u8(),
            parent_id: slice.parent_id.clone(),
            children_ids: slice.children_ids.clone(),
            keywords: slice.keywords.clone(),
            content_hash: None,
            source_hash: None,
        }
    }

    /// Create a document from an onion slice with content hash for deduplication.
    /// Source hash is left empty — prefer `from_onion_slice_with_hashes` to
    /// preserve source provenance for pre-index dedup.
    pub fn from_onion_slice_with_hash(
        slice: &crate::rag::OnionSlice,
        namespace: String,
        embedding: Vec<f32>,
        metadata: serde_json::Value,
        content_hash: String,
    ) -> Self {
        Self::from_onion_slice_with_hashes(
            slice,
            namespace,
            embedding,
            metadata,
            content_hash,
            None,
        )
    }

    /// Create an onion-slice document with both per-chunk and source hashes.
    pub fn from_onion_slice_with_hashes(
        slice: &crate::rag::OnionSlice,
        namespace: String,
        embedding: Vec<f32>,
        metadata: serde_json::Value,
        content_hash: String,
        source_hash: Option<String>,
    ) -> Self {
        Self {
            id: slice.id.clone(),
            namespace,
            embedding,
            metadata,
            document: slice.content.clone(),
            layer: slice.layer.as_u8(),
            parent_id: slice.parent_id.clone(),
            children_ids: slice.children_ids.clone(),
            keywords: slice.keywords.clone(),
            content_hash: Some(content_hash),
            source_hash,
        }
    }

    /// Check if this is a legacy flat chunk (not an onion slice)
    pub fn is_flat(&self) -> bool {
        self.layer == 0
    }

    /// Get the slice layer if this is an onion slice
    pub fn slice_layer(&self) -> Option<SliceLayer> {
        SliceLayer::from_u8(self.layer)
    }
}

pub struct StorageManager {
    lance: Connection,
    table: Arc<Mutex<Option<Table>>>,
    collection_name: String,
    lance_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum CrossStoreRecoveryStatus {
    #[default]
    Pending,
    RolledBack,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossStoreRecoveryDocumentRef {
    pub namespace: String,
    pub id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossStoreRecoveryBatch {
    pub batch_id: String,
    pub created_at: String,
    #[serde(default)]
    pub status: CrossStoreRecoveryStatus,
    #[serde(default)]
    pub last_error: Option<String>,
    pub documents: Vec<CrossStoreRecoveryDocumentRef>,
}

impl CrossStoreRecoveryBatch {
    pub fn from_documents(documents: &[ChromaDocument]) -> Self {
        Self {
            batch_id: Uuid::new_v4().to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
            status: CrossStoreRecoveryStatus::Pending,
            last_error: None,
            documents: documents
                .iter()
                .map(|document| CrossStoreRecoveryDocumentRef {
                    namespace: document.namespace.clone(),
                    id: document.id.clone(),
                })
                .collect(),
        }
    }
}

type BatchIter =
    RecordBatchIterator<std::vec::IntoIter<std::result::Result<RecordBatch, ArrowError>>>;

impl StorageManager {
    pub async fn new(db_path: &str) -> Result<Self> {
        // Embedded LanceDB path (expand ~, allow override via env)
        let lance_env = std::env::var("LANCEDB_PATH").unwrap_or_else(|_| db_path.to_string());
        let lance_path = if lance_env.trim().is_empty() {
            shellexpand::tilde("~/.rmcp-servers/rust-memex/lancedb").to_string()
        } else {
            shellexpand::tilde(&lance_env).to_string()
        };

        let lance = connect(&lance_path).execute().await?;

        Ok(Self {
            lance,
            table: Arc::new(Mutex::new(None)),
            collection_name: DEFAULT_TABLE_NAME.to_string(),
            lance_path,
        })
    }

    /// Create a storage manager for CLI tools.
    /// Use this for CLI tools that only need vector operations (index/search).
    pub async fn new_lance_only(db_path: &str) -> Result<Self> {
        let lance_path = shellexpand::tilde(db_path).to_string();
        let lance = connect(&lance_path).execute().await?;

        Ok(Self {
            lance,
            table: Arc::new(Mutex::new(None)),
            collection_name: DEFAULT_TABLE_NAME.to_string(),
            lance_path,
        })
    }

    pub fn lance_path(&self) -> &str {
        &self.lance_path
    }

    pub async fn require_current_schema_for_writes(&self) -> Result<()> {
        let Some(table) = self.open_table_if_exists().await? else {
            return Ok(());
        };
        self.ensure_hash_schema_columns(&table).await
    }

    pub async fn missing_required_columns(
        table: &Table,
        target: SchemaVersion,
    ) -> Result<Vec<Field>> {
        let schema = table.schema().await?;
        Ok(required_columns_for(target)
            .into_iter()
            .filter(|field| schema.field_with_name(field.name()).is_err())
            .collect())
    }

    pub async fn migrate_lance_schema(
        db_path: &str,
        target: SchemaVersion,
        check_only: bool,
    ) -> Result<SchemaMigrationReport> {
        let lance_path = shellexpand::tilde(db_path).to_string();
        let lance = connect(&lance_path).execute().await?;
        let table = lance.open_table(DEFAULT_TABLE_NAME).execute().await?;
        let missing = Self::missing_required_columns(&table, target).await?;

        if missing.is_empty() || check_only {
            return Ok(SchemaMigrationReport {
                target,
                missing_columns: missing,
                applied: false,
            });
        }

        let transform = NewColumnTransform::AllNulls(Arc::new(Schema::new(missing.clone())));
        if let Err(error) = table.add_columns(transform, None).await {
            let _ = table.checkout_latest().await;
            let remaining = Self::missing_required_columns(&table, target).await?;
            if remaining.is_empty() {
                warn!(
                    "Lance table '{}' schema migration raced with another writer and is already complete",
                    DEFAULT_TABLE_NAME
                );
                return Ok(SchemaMigrationReport {
                    target,
                    missing_columns: missing,
                    applied: true,
                });
            }
            return Err(anyhow!(
                "failed to migrate Lance table '{}' schema to {target}: {error}",
                DEFAULT_TABLE_NAME
            ));
        }

        let _ = table.checkout_latest().await;
        Ok(SchemaMigrationReport {
            target,
            missing_columns: missing,
            applied: true,
        })
    }

    pub fn cross_store_recovery_dir(&self) -> PathBuf {
        let db_path = Path::new(&self.lance_path);
        let parent = db_path.parent().unwrap_or_else(|| Path::new("."));
        let stem = db_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("lancedb");
        parent.join(format!(".{stem}-cross-store-recovery"))
    }

    fn cross_store_recovery_batch_path(&self, batch_id: &str) -> PathBuf {
        self.cross_store_recovery_dir()
            .join(format!("{batch_id}.json"))
    }

    fn write_cross_store_recovery_batch(&self, batch: &CrossStoreRecoveryBatch) -> Result<PathBuf> {
        let dir = self.cross_store_recovery_dir();
        std::fs::create_dir_all(&dir)?;

        let path = self.cross_store_recovery_batch_path(&batch.batch_id);
        let tmp_path = path.with_extension("json.tmp");
        let payload = serde_json::to_vec_pretty(batch)?;

        std::fs::write(&tmp_path, payload)?;
        std::fs::rename(&tmp_path, &path)?;

        Ok(path)
    }

    pub fn persist_cross_store_recovery_batch(
        &self,
        batch: &CrossStoreRecoveryBatch,
    ) -> Result<PathBuf> {
        self.write_cross_store_recovery_batch(batch)
    }

    pub fn update_cross_store_recovery_batch(
        &self,
        batch: &CrossStoreRecoveryBatch,
    ) -> Result<PathBuf> {
        self.write_cross_store_recovery_batch(batch)
    }

    pub fn clear_cross_store_recovery_batch(&self, batch_id: &str) -> Result<()> {
        let path = self.cross_store_recovery_batch_path(batch_id);
        match std::fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error.into()),
        }
    }

    pub fn list_cross_store_recovery_batches(&self) -> Result<Vec<CrossStoreRecoveryBatch>> {
        let dir = self.cross_store_recovery_dir();
        if !dir.exists() {
            return Ok(vec![]);
        }

        let mut batches = Vec::new();
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                continue;
            }

            let payload = std::fs::read(&path)?;
            let batch: CrossStoreRecoveryBatch = serde_json::from_slice(&payload)?;
            batches.push(batch);
        }

        batches.sort_by(|left, right| left.created_at.cmp(&right.created_at));
        Ok(batches)
    }

    /// Refresh the table connection to see new data written by other processes.
    /// This clears the cached table reference, forcing it to be re-opened on next query.
    pub async fn refresh(&self) -> Result<()> {
        let mut guard = self.table.lock().await;
        *guard = None;
        tracing::info!("LanceDB table cache cleared - will refresh on next query");
        Ok(())
    }

    pub async fn ensure_collection(&self) -> Result<()> {
        // Attempt to open; if missing, create empty table lazily on first add
        let mut guard = self.table.lock().await;
        if guard.is_some() {
            return Ok(());
        }
        match self
            .lance
            .open_table(self.collection_name.as_str())
            .execute()
            .await
        {
            Ok(table) => {
                *guard = Some(table);
                info!("Found existing Lance table '{}'", self.collection_name);
            }
            Err(_) => {
                info!(
                    "Lance table '{}' will be created on first insert",
                    self.collection_name
                );
            }
        }
        Ok(())
    }

    pub async fn add_to_store(&self, documents: Vec<ChromaDocument>) -> Result<()> {
        if documents.is_empty() {
            return Ok(());
        }

        // Pre-validation: check all documents before writing anything
        let dim = documents
            .first()
            .ok_or_else(|| anyhow!("No documents to add"))?
            .embedding
            .len();
        if dim == 0 {
            return Err(anyhow!("Embedding dimension is zero"));
        }

        // Validate ALL documents have consistent dimensions and required fields
        for (i, doc) in documents.iter().enumerate() {
            if doc.embedding.len() != dim {
                return Err(anyhow!(
                    "Document {} has inconsistent embedding dimension: expected {}, got {}. \
                     Aborting batch to prevent database corruption.",
                    i,
                    dim,
                    doc.embedding.len()
                ));
            }
            if doc.id.is_empty() {
                return Err(anyhow!("Document {} has empty ID. Aborting batch.", i));
            }
            if doc.namespace.is_empty() {
                return Err(anyhow!(
                    "Document {} has empty namespace. Aborting batch.",
                    i
                ));
            }
            // Check for NaN/Inf in embeddings
            for (j, &val) in doc.embedding.iter().enumerate() {
                if val.is_nan() || val.is_infinite() {
                    return Err(anyhow!(
                        "Document {} has invalid embedding value at index {}: {}. \
                         Aborting batch to prevent database corruption.",
                        i,
                        j,
                        val
                    ));
                }
            }
        }

        let table = self.ensure_table(dim).await?;
        self.ensure_hash_schema_columns(&table).await?;
        let batch = self.docs_to_batch(&documents, dim)?;
        if let Err(error) = table.add(batch).execute().await {
            return Err(self.map_lancedb_write_error(error));
        }
        debug!(
            "Inserted {} documents into Lance (validated)",
            documents.len()
        );
        Ok(())
    }

    pub async fn search_store(
        &self,
        namespace: Option<&str>,
        embedding: Vec<f32>,
        k: usize,
    ) -> Result<Vec<ChromaDocument>> {
        if embedding.is_empty() {
            return Ok(vec![]);
        }
        let dim = embedding.len();
        let table = self.ensure_table(dim).await?;

        let mut query = table.query();
        if let Some(ns) = namespace {
            query = query.only_if(self.namespace_filter(ns).as_str());
        }
        let mut stream = query.nearest_to(embedding)?.limit(k).execute().await?;

        let mut results = Vec::new();
        while let Some(batch) = stream.try_next().await? {
            let mut docs = self.batch_to_docs(&batch)?;
            results.append(&mut docs);
        }
        debug!("Lance returned {} results", results.len());
        Ok(results)
    }

    /// Return a single page of documents without running a vector search.
    ///
    /// Used by admin/reporting paths that need deterministic limit/offset
    /// behavior without assuming any embedding dimension or creating a table on
    /// read.
    pub async fn all_documents_page(
        &self,
        namespace: Option<&str>,
        offset: usize,
        limit: usize,
    ) -> Result<Vec<ChromaDocument>> {
        let table = match self.open_table_if_exists().await? {
            Some(t) => t,
            None => return Ok(vec![]),
        };

        let mut query = table.query().limit(limit).offset(offset);
        if let Some(ns) = namespace {
            query = query.only_if(self.namespace_filter(ns).as_str());
        }
        let mut stream = query.execute().await?;

        let mut results = Vec::new();
        while let Some(batch) = stream.try_next().await? {
            let mut docs = self.batch_to_docs(&batch)?;
            results.append(&mut docs);
        }

        Ok(results)
    }

    /// Return documents without running a vector search.
    /// Used by admin/reporting paths that need a bounded full-table scan
    /// starting from the first row.
    pub async fn all_documents(
        &self,
        namespace: Option<&str>,
        limit: usize,
    ) -> Result<Vec<ChromaDocument>> {
        self.all_documents_page(namespace, 0, limit).await
    }

    pub async fn get_document(&self, namespace: &str, id: &str) -> Result<Option<ChromaDocument>> {
        let table = match self.open_table_if_exists().await? {
            Some(t) => t,
            None => return Ok(None),
        };
        let filter = format!(
            "{} AND {}",
            self.namespace_filter(namespace),
            self.id_filter(id)
        );
        let mut stream = table
            .query()
            .only_if(filter.as_str())
            .limit(1)
            .execute()
            .await?;
        if let Some(batch) = stream.try_next().await? {
            let mut docs = self.batch_to_docs(&batch)?;
            if let Some(doc) = docs.pop() {
                return Ok(Some(doc));
            }
        }
        Ok(None)
    }

    pub async fn delete_document(&self, namespace: &str, id: &str) -> Result<usize> {
        let table = match self.open_table_if_exists().await? {
            Some(t) => t,
            None => return Ok(0),
        };
        let predicate = format!(
            "{} AND {}",
            self.namespace_filter(namespace),
            self.id_filter(id)
        );
        let pre_count = table
            .query()
            .only_if(predicate.as_str())
            .execute()
            .await?
            .try_fold(
                0usize,
                |acc, batch| async move { Ok(acc + batch.num_rows()) },
            )
            .await?;
        if pre_count == 0 {
            return Ok(0);
        }
        table.delete(predicate.as_str()).await?;
        Ok(pre_count)
    }

    /// Batch delete documents by IDs within a namespace.
    ///
    /// Issues a single `DELETE WHERE namespace = X AND id IN (...)` per chunk,
    /// avoiding the per-document table scan that `delete_document` incurs when
    /// called in a loop. Predicate is split into 500-id chunks to keep SQL
    /// length bounded regardless of caller batch size.
    pub async fn delete_documents(&self, namespace: &str, ids: &[&str]) -> Result<usize> {
        if ids.is_empty() {
            return Ok(0);
        }
        let table = match self.open_table_if_exists().await? {
            Some(t) => t,
            None => return Ok(0),
        };
        const CHUNK: usize = 500;
        let ns_filter = self.namespace_filter(namespace);
        let mut total_deleted = 0usize;
        for batch in ids.chunks(CHUNK) {
            let id_list = batch
                .iter()
                .map(|id| format!("'{}'", id.replace('\'', "''")))
                .collect::<Vec<_>>()
                .join(", ");
            let predicate = format!("{} AND id IN ({})", ns_filter, id_list);
            let pre_count = table.count_rows(Some(predicate.clone())).await?;
            if pre_count == 0 {
                continue;
            }
            table.delete(predicate.as_str()).await?;
            total_deleted += pre_count;
        }
        Ok(total_deleted)
    }

    pub async fn delete_namespace_documents(&self, namespace: &str) -> Result<usize> {
        let table = match self.open_table_if_exists().await? {
            Some(t) => t,
            None => return Ok(0),
        };
        let predicate = self.namespace_filter(namespace);
        let pre_count = table
            .query()
            .only_if(predicate.as_str())
            .execute()
            .await?
            .try_fold(
                0usize,
                |acc, batch| async move { Ok(acc + batch.num_rows()) },
            )
            .await?;
        if pre_count == 0 {
            return Ok(0);
        }
        table.delete(predicate.as_str()).await?;
        Ok(pre_count)
    }

    pub async fn rename_namespace_atomic(&self, from: &str, to: &str) -> Result<usize> {
        if from == to {
            return Ok(0);
        }

        let table = match self.open_table_if_exists().await? {
            Some(t) => t,
            None => return Ok(0),
        };

        let source_filter = self.namespace_filter(from);
        let source_count = table.count_rows(Some(source_filter.clone())).await?;
        if source_count == 0 {
            return Ok(0);
        }

        let target_filter = self.namespace_filter(to);
        let target_count = table.count_rows(Some(target_filter)).await?;
        if target_count > 0 {
            return Err(anyhow!(
                "Target namespace '{}' already exists with {} rows",
                to,
                target_count
            ));
        }

        let sql_literal = format!("'{}'", to.replace('\'', "''"));
        let update = table
            .update()
            .only_if(source_filter)
            .column("namespace", sql_literal)
            .execute()
            .await?;

        Ok(update.rows_updated as usize)
    }

    pub fn get_collection_name(&self) -> &str {
        &self.collection_name
    }

    async fn ensure_table(&self, dim: usize) -> Result<Table> {
        let mut guard = self.table.lock().await;
        if let Some(table) = guard.as_ref() {
            return Ok(table.clone());
        }

        let maybe_table = self
            .lance
            .open_table(self.collection_name.as_str())
            .execute()
            .await;

        let table = if let Ok(tbl) = maybe_table {
            tbl
        } else {
            if dim == 0 {
                return Err(anyhow!(
                    "Vector table '{}' not found and dimension is unknown",
                    self.collection_name
                ));
            }
            info!(
                "Creating Lance table '{}' with vector dimension {} (schema v{})",
                self.collection_name, dim, SCHEMA_VERSION
            );
            let schema = Arc::new(Self::create_schema(dim));
            self.lance
                .create_empty_table(self.collection_name.as_str(), schema)
                .execute()
                .await?
        };

        *guard = Some(table.clone());
        Ok(table)
    }

    /// Try to open the table without creating it.
    /// Returns `Ok(None)` when the table genuinely does not exist.
    /// Propagates real errors (I/O, corruption, permission) as `Err`.
    async fn open_table_if_exists(&self) -> Result<Option<Table>> {
        let mut guard = self.table.lock().await;
        if let Some(table) = guard.as_ref() {
            return Ok(Some(table.clone()));
        }

        match self
            .lance
            .open_table(self.collection_name.as_str())
            .execute()
            .await
        {
            Ok(tbl) => {
                *guard = Some(tbl.clone());
                Ok(Some(tbl))
            }
            Err(e) => {
                let msg = e.to_string().to_lowercase();
                if msg.contains("not found")
                    || msg.contains("does not exist")
                    || msg.contains("no such file")
                {
                    Ok(None)
                } else {
                    tracing::warn!(
                        "LanceDB error opening table '{}': {}",
                        self.collection_name,
                        e
                    );
                    Err(anyhow!(
                        "LanceDB error on table '{}': {}",
                        self.collection_name,
                        e
                    ))
                }
            }
        }
    }

    async fn open_existing_table(&self) -> Result<Table> {
        self.open_table_if_exists().await?.ok_or_else(|| {
            anyhow!(
                "Vector table '{}' not found at {}. Index data first so rust-memex can use the stored embedding dimension instead of guessing.",
                self.collection_name,
                self.lance_path
            )
        })
    }

    /// Create the LanceDB schema with onion slice fields and content hash
    fn create_schema(dim: usize) -> Schema {
        Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("namespace", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dim as i32,
                ),
                false,
            ),
            Field::new("text", DataType::Utf8, true),
            Field::new("metadata", DataType::Utf8, true),
            // Onion slice fields (v2 schema)
            Field::new("layer", DataType::UInt8, true), // 0=flat, 1=outer, 2=middle, 3=inner, 4=core
            Field::new("parent_id", DataType::Utf8, true), // Parent slice ID
            Field::new("children_ids", DataType::Utf8, true), // JSON array of children IDs
            Field::new("keywords", DataType::Utf8, true), // JSON array of keywords
            // Per-chunk dedup hash (v3 schema; v4 narrows semantic to chunk text)
            Field::new("content_hash", DataType::Utf8, true), // SHA256 of THIS chunk's text
            // Source-document hash (v4 schema) — same value across all 4 layers
            // from one source. Drives pre-index dedup so we never re-embed an
            // already-ingested file.
            Field::new("source_hash", DataType::Utf8, true), // SHA256 of source document text
        ])
    }

    async fn ensure_hash_schema_columns(&self, table: &Table) -> Result<()> {
        let missing = Self::missing_required_columns(table, SchemaVersion::current()).await?;

        if missing.is_empty() {
            return Ok(());
        }

        let missing_columns = missing
            .iter()
            .map(|field| field.name().to_string())
            .collect::<Vec<_>>();
        let error = SchemaMismatchWriteError::new(
            self.collection_name.clone(),
            self.lance_path.clone(),
            missing_columns,
            "table is older than the current writer schema",
        );
        self.log_schema_mismatch(&error);
        Err(error.into())
    }

    fn map_lancedb_write_error(&self, error: lancedb::error::Error) -> anyhow::Error {
        let message = match &error {
            lancedb::error::Error::Lance { source } => source.to_string(),
            lancedb::error::Error::Schema { message } => message.clone(),
            lancedb::error::Error::Arrow { source } => source.to_string(),
            _ => return error.into(),
        };

        if !is_schema_mismatch_message(&message) {
            return error.into();
        }

        let missing_columns = extract_missing_columns(&message);
        let error = SchemaMismatchWriteError::new(
            self.collection_name.clone(),
            self.lance_path.clone(),
            missing_columns,
            message,
        );
        self.log_schema_mismatch(&error);
        error.into()
    }

    fn log_schema_mismatch(&self, error: &SchemaMismatchWriteError) {
        error!(
            error_kind = "schema_mismatch",
            table = %error.table_name(),
            db_path = %error.db_path(),
            missing_columns = ?error.missing_columns(),
            remediation = %error.remediation(),
            file = file!(),
            line = line!(),
            "write-path schema mismatch"
        );
    }

    fn docs_to_batch(&self, documents: &[ChromaDocument], dim: usize) -> Result<BatchIter> {
        let ids = documents.iter().map(|d| d.id.as_str()).collect::<Vec<_>>();
        let namespaces = documents
            .iter()
            .map(|d| d.namespace.as_str())
            .collect::<Vec<_>>();
        let texts = documents
            .iter()
            .map(|d| d.document.as_str())
            .collect::<Vec<_>>();
        let metadata_strings = documents
            .iter()
            .map(|d| serde_json::to_string(&d.metadata).unwrap_or_else(|_| "{}".to_string()))
            .collect::<Vec<_>>();

        let vectors = documents.iter().map(|d| {
            if d.embedding.len() != dim {
                None
            } else {
                Some(d.embedding.iter().map(|v| Some(*v)).collect::<Vec<_>>())
            }
        });

        // Onion slice fields
        let layers: Vec<u8> = documents.iter().map(|d| d.layer).collect();
        let parent_ids: Vec<Option<&str>> =
            documents.iter().map(|d| d.parent_id.as_deref()).collect();
        let children_ids_json: Vec<String> = documents
            .iter()
            .map(|d| serde_json::to_string(&d.children_ids).unwrap_or_else(|_| "[]".to_string()))
            .collect();
        let keywords_json: Vec<String> = documents
            .iter()
            .map(|d| serde_json::to_string(&d.keywords).unwrap_or_else(|_| "[]".to_string()))
            .collect();
        // Per-chunk content hash for chunk-level dedup
        let content_hashes: Vec<Option<&str>> = documents
            .iter()
            .map(|d| d.content_hash.as_deref())
            .collect();
        // Source-document hash for pre-index source-level dedup (v4 schema)
        let source_hashes: Vec<Option<&str>> =
            documents.iter().map(|d| d.source_hash.as_deref()).collect();

        let schema = Arc::new(Self::create_schema(dim));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(ids)),
                Arc::new(StringArray::from(namespaces)),
                Arc::new(
                    FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                        vectors, dim as i32,
                    ),
                ),
                Arc::new(StringArray::from(texts)),
                Arc::new(StringArray::from(metadata_strings)),
                // Onion slice fields
                Arc::new(UInt8Array::from(layers)),
                Arc::new(StringArray::from(parent_ids)),
                Arc::new(StringArray::from(
                    children_ids_json
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    keywords_json.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                )),
                // Per-chunk content hash for chunk-level dedup
                Arc::new(StringArray::from(content_hashes)),
                // Source-document hash for source-level dedup (v4 schema)
                Arc::new(StringArray::from(source_hashes)),
            ],
        )?;

        Ok(RecordBatchIterator::new(
            vec![Ok(batch)].into_iter(),
            schema,
        ))
    }

    fn batch_to_docs(&self, batch: &RecordBatch) -> Result<Vec<ChromaDocument>> {
        let id_col = batch
            .column_by_name("id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| anyhow!("Missing id column"))?;
        let ns_col = batch
            .column_by_name("namespace")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| anyhow!("Missing namespace column"))?;
        let text_col = batch
            .column_by_name("text")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| anyhow!("Missing text column"))?;
        let metadata_col = batch
            .column_by_name("metadata")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| anyhow!("Missing metadata column"))?;
        let vector_col = batch
            .column_by_name("vector")
            .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>())
            .ok_or_else(|| anyhow!("Missing vector column"))?;

        // Onion slice fields (optional for backward compatibility with v1 schema)
        let layer_col = batch
            .column_by_name("layer")
            .and_then(|c| c.as_any().downcast_ref::<UInt8Array>());
        let parent_id_col = batch
            .column_by_name("parent_id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let children_ids_col = batch
            .column_by_name("children_ids")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let keywords_col = batch
            .column_by_name("keywords")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        // Content hash field (optional for backward compatibility with v2 schema)
        let content_hash_col = batch
            .column_by_name("content_hash")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        // Source hash column (v4 schema) — optional for pre-v4 tables.
        let source_hash_col = batch
            .column_by_name("source_hash")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());

        let dim = vector_col.value_length() as usize;
        let values = vector_col
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| anyhow!("Vector inner type mismatch"))?;

        let mut docs = Vec::new();
        for i in 0..batch.num_rows() {
            let id = id_col.value(i).to_string();
            let text = text_col.value(i).to_string();
            let namespace = ns_col.value(i).to_string();
            let meta_str = metadata_col.value(i);
            let metadata: Value = serde_json::from_str(meta_str).unwrap_or_else(|_| json!({}));

            let offset = i * dim;
            let mut emb = Vec::with_capacity(dim);
            for j in 0..dim {
                emb.push(values.value(offset + j));
            }

            // Read onion slice fields (with v1 schema compatibility)
            let layer = layer_col
                .and_then(|col| {
                    if col.is_null(i) {
                        None
                    } else {
                        Some(col.value(i))
                    }
                })
                .unwrap_or(0);

            let parent_id = parent_id_col.and_then(|col| {
                if col.is_null(i) {
                    None
                } else {
                    Some(col.value(i).to_string())
                }
            });

            let children_ids: Vec<String> = children_ids_col
                .and_then(|col| {
                    if col.is_null(i) {
                        None
                    } else {
                        serde_json::from_str(col.value(i)).ok()
                    }
                })
                .unwrap_or_default();

            let keywords: Vec<String> = keywords_col
                .and_then(|col| {
                    if col.is_null(i) {
                        None
                    } else {
                        serde_json::from_str(col.value(i)).ok()
                    }
                })
                .unwrap_or_default();

            let content_hash = content_hash_col.and_then(|col| {
                if col.is_null(i) {
                    None
                } else {
                    Some(col.value(i).to_string())
                }
            });

            let source_hash = source_hash_col.and_then(|col| {
                if col.is_null(i) {
                    None
                } else {
                    Some(col.value(i).to_string())
                }
            });

            docs.push(ChromaDocument {
                id,
                namespace,
                embedding: emb,
                metadata,
                document: text,
                layer,
                parent_id,
                children_ids,
                keywords,
                content_hash,
                source_hash,
            });
        }
        Ok(docs)
    }

    pub async fn get_filtered_in_namespace(
        &self,
        namespace: &str,
        filter: &str,
    ) -> Result<Vec<ChromaDocument>> {
        let table = match self.open_table_if_exists().await? {
            Some(t) => t,
            None => return Ok(vec![]),
        };
        let combined = format!("{} AND ({})", self.namespace_filter(namespace), filter);
        let mut stream = table.query().only_if(combined.as_str()).execute().await?;
        let mut results = Vec::new();
        while let Some(batch) = stream.try_next().await? {
            let mut docs = self.batch_to_docs(&batch)?;
            results.append(&mut docs);
        }
        Ok(results)
    }

    /// Search with optional layer filtering for onion slice architecture
    pub async fn search_store_with_layer(
        &self,
        namespace: Option<&str>,
        embedding: Vec<f32>,
        k: usize,
        layer_filter: Option<SliceLayer>,
    ) -> Result<Vec<ChromaDocument>> {
        if embedding.is_empty() {
            return Ok(vec![]);
        }
        let dim = embedding.len();
        let table = self.ensure_table(dim).await?;

        let mut query = table.query();

        // Build combined filter
        let mut filters = Vec::new();
        if let Some(ns) = namespace {
            filters.push(self.namespace_filter(ns));
        }
        if let Some(layer) = layer_filter {
            filters.push(self.layer_filter(layer));
        }

        if !filters.is_empty() {
            let combined = filters.join(" AND ");
            query = query.only_if(combined.as_str());
        }

        let mut stream = query.nearest_to(embedding)?.limit(k).execute().await?;

        let mut results = Vec::new();
        while let Some(batch) = stream.try_next().await? {
            let mut docs = self.batch_to_docs(&batch)?;
            results.append(&mut docs);
        }
        debug!(
            "Lance returned {} results (layer filter: {:?})",
            results.len(),
            layer_filter
        );
        Ok(results)
    }

    /// Get a document by ID and expand to get its children
    pub async fn get_children(
        &self,
        namespace: &str,
        parent_id: &str,
    ) -> Result<Vec<ChromaDocument>> {
        if self.open_table_if_exists().await?.is_none() {
            return Ok(vec![]);
        }

        // First get the parent document to find children IDs
        if let Some(parent) = self.get_document(namespace, parent_id).await? {
            if parent.children_ids.is_empty() {
                return Ok(vec![]);
            }

            // Query for all children
            let mut children = Vec::new();
            for child_id in &parent.children_ids {
                if let Some(child) = self.get_document(namespace, child_id).await? {
                    children.push(child);
                }
            }
            return Ok(children);
        }

        Ok(vec![])
    }

    /// Get the parent of a document (drill up in onion hierarchy)
    pub async fn get_parent(
        &self,
        namespace: &str,
        child_id: &str,
    ) -> Result<Option<ChromaDocument>> {
        if let Some(child) = self.get_document(namespace, child_id).await?
            && let Some(ref parent_id) = child.parent_id
        {
            return self.get_document(namespace, parent_id).await;
        }
        Ok(None)
    }

    fn namespace_filter(&self, namespace: &str) -> String {
        format!("namespace = '{}'", namespace.replace('\'', "''"))
    }

    fn id_filter(&self, id: &str) -> String {
        format!("id = '{}'", id.replace('\'', "''"))
    }

    fn layer_filter(&self, layer: SliceLayer) -> String {
        if layer == SliceLayer::Outer {
            // Default search should surface onion summaries while still seeing legacy flat chunks.
            "(layer = 0 OR layer = 1)".to_string()
        } else {
            format!("layer = {}", layer.as_u8())
        }
    }

    fn content_hash_filter(&self, hash: &str) -> String {
        format!("content_hash = '{}'", hash.replace('\'', "''"))
    }

    fn source_hash_filter(&self, hash: &str) -> String {
        format!("source_hash = '{}'", hash.replace('\'', "''"))
    }

    /// Check if the table schema has content_hash column (schema v3+)
    async fn table_has_content_hash(table: &Table) -> bool {
        table
            .schema()
            .await
            .map(|schema| schema.field_with_name("content_hash").is_ok())
            .unwrap_or(false)
    }

    /// Check if the table schema has source_hash column (schema v4+)
    async fn table_has_source_hash(table: &Table) -> bool {
        table
            .schema()
            .await
            .map(|schema| schema.field_with_name("source_hash").is_ok())
            .unwrap_or(false)
    }

    /// Check if a content hash already exists in a namespace (for exact-match deduplication)
    ///
    /// Returns Ok(false) if:
    /// - Table doesn't exist yet
    /// - Table has old schema without content_hash column (graceful degradation)
    pub async fn has_content_hash(&self, namespace: &str, hash: &str) -> Result<bool> {
        let table = match self.open_table_if_exists().await? {
            Some(t) => t,
            None => return Ok(false),
        };

        // Graceful handling of old schema without content_hash column
        if !Self::table_has_content_hash(&table).await {
            tracing::warn!(
                "Table '{}' has old schema without content_hash column. \
                 Deduplication disabled. Consider re-indexing with new schema.",
                self.collection_name
            );
            return Ok(false); // Can't check for duplicates, treat as new
        }

        let filter = format!(
            "{} AND {}",
            self.namespace_filter(namespace),
            self.content_hash_filter(hash)
        );

        let mut stream = table
            .query()
            .only_if(filter.as_str())
            .limit(1)
            .execute()
            .await?;

        if let Some(batch) = stream.try_next().await? {
            return Ok(batch.num_rows() > 0);
        }

        Ok(false)
    }

    /// Check if any chunk in `namespace` already references the given source-document
    /// hash. Used by the indexing pipeline to skip re-embedding files that were
    /// already ingested (P4 — pre-index source-level dedup).
    ///
    /// Returns Ok(false) if the table doesn't exist yet, or if the table is on a
    /// pre-v4 schema without the `source_hash` column (graceful degradation —
    /// older namespaces should be backfilled via `/admin/backfill-hashes`).
    pub async fn has_source_hash(&self, namespace: &str, hash: &str) -> Result<bool> {
        let table = match self.open_table_if_exists().await? {
            Some(t) => t,
            None => return Ok(false),
        };

        if !Self::table_has_source_hash(&table).await {
            tracing::debug!(
                "Table '{}' has pre-v4 schema without source_hash column. \
                 Source-level dedup disabled until backfill.",
                self.collection_name
            );
            return Ok(false);
        }

        let filter = format!(
            "{} AND {}",
            self.namespace_filter(namespace),
            self.source_hash_filter(hash)
        );

        let mut stream = table
            .query()
            .only_if(filter.as_str())
            .limit(1)
            .execute()
            .await?;

        if let Some(batch) = stream.try_next().await? {
            return Ok(batch.num_rows() > 0);
        }

        Ok(false)
    }

    /// Filter a list of hashes to return only those that don't exist in the namespace.
    /// This is more efficient than calling has_content_hash for each hash individually.
    ///
    /// Returns all hashes as "new" if table has old schema without content_hash column.
    pub async fn filter_existing_hashes<'a>(
        &self,
        namespace: &str,
        hashes: &'a [String],
    ) -> Result<Vec<&'a String>> {
        if hashes.is_empty() {
            return Ok(vec![]);
        }

        let table = match self.open_table_if_exists().await? {
            Some(t) => t,
            None => return Ok(hashes.iter().collect()),
        };

        // Graceful handling of old schema without content_hash column
        if !Self::table_has_content_hash(&table).await {
            tracing::warn!(
                "Table '{}' has old schema without content_hash column. \
                 Deduplication disabled. Consider re-indexing with new schema.",
                self.collection_name
            );
            return Ok(hashes.iter().collect()); // All are "new" since we can't check
        }

        // Query for existing hashes in this namespace
        // We build a filter with OR conditions for all hashes
        let hash_conditions: Vec<String> =
            hashes.iter().map(|h| self.content_hash_filter(h)).collect();

        let filter = format!(
            "{} AND ({})",
            self.namespace_filter(namespace),
            hash_conditions.join(" OR ")
        );

        let mut stream = table
            .query()
            .only_if(filter.as_str())
            .limit(hashes.len())
            .execute()
            .await?;

        // Collect existing hashes from results
        let mut existing_hashes = std::collections::HashSet::new();
        while let Some(batch) = stream.try_next().await? {
            if let Some(hash_col) = batch
                .column_by_name("content_hash")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            {
                for i in 0..batch.num_rows() {
                    if !hash_col.is_null(i) {
                        existing_hashes.insert(hash_col.value(i).to_string());
                    }
                }
            }
        }

        // Return only hashes that don't exist
        Ok(hashes
            .iter()
            .filter(|h| !existing_hashes.contains(h.as_str()))
            .collect())
    }

    // =========================================================================
    // MAINTENANCE OPERATIONS
    // =========================================================================

    /// Run all optimizations (compact + prune old versions)
    pub async fn optimize(&self) -> Result<OptimizeStats> {
        let table = self.open_existing_table().await?;
        let stats = table.optimize(OptimizeAction::All).await?;
        info!(
            "Optimize complete: compaction={:?}, prune={:?}",
            stats.compaction, stats.prune
        );
        Ok(stats)
    }

    /// Compact small files into larger ones for better performance
    pub async fn compact(&self) -> Result<OptimizeStats> {
        let table = self.open_existing_table().await?;
        let stats = table
            .optimize(OptimizeAction::Compact {
                options: Default::default(),
                remap_options: None,
            })
            .await?;
        info!("Compaction complete: {:?}", stats.compaction);
        Ok(stats)
    }

    /// Remove old versions older than specified duration (default: 7 days)
    pub async fn cleanup(&self, older_than_days: Option<u64>) -> Result<OptimizeStats> {
        let table = self.open_existing_table().await?;
        let days = older_than_days.unwrap_or(7) as i64;
        let duration = chrono::TimeDelta::days(days);
        let stats = table
            .optimize(OptimizeAction::Prune {
                older_than: Some(duration),
                delete_unverified: Some(false),
                error_if_tagged_old_versions: None,
            })
            .await?;
        info!("Cleanup complete: {:?}", stats.prune);
        Ok(stats)
    }

    /// Get table statistics (row count, fragments, etc.)
    pub async fn stats(&self) -> Result<TableStats> {
        let table = self.open_existing_table().await?;
        let row_count = table.count_rows(None).await?;

        // Get version count
        let versions = table.list_versions().await.unwrap_or_default();
        let version_count = versions.len();

        Ok(TableStats {
            row_count,
            version_count,
            table_name: self.collection_name.clone(),
            db_path: self.lance_path.clone(),
        })
    }

    /// Count rows in a specific namespace
    pub async fn count_namespace(&self, namespace: &str) -> Result<usize> {
        let table = match self.open_table_if_exists().await? {
            Some(table) => table,
            None => return Ok(0),
        };
        let filter = self.namespace_filter(namespace);
        let count = table.count_rows(Some(filter)).await?;
        Ok(count)
    }

    /// Get all documents from a namespace (for migration/export)
    ///
    /// Note: This uses a full table scan with namespace filter.
    /// For very large namespaces, consider batching.
    pub async fn get_all_in_namespace(&self, namespace: &str) -> Result<Vec<ChromaDocument>> {
        let table = match self.open_table_if_exists().await? {
            Some(t) => t,
            None => return Ok(vec![]),
        };

        let filter = self.namespace_filter(namespace);
        let mut stream = table.query().only_if(filter.as_str()).execute().await?;

        let mut results = Vec::new();
        while let Some(batch) = stream.try_next().await? {
            let mut docs = self.batch_to_docs(&batch)?;
            results.append(&mut docs);
        }

        debug!(
            "Retrieved {} documents from namespace '{}'",
            results.len(),
            namespace
        );
        Ok(results)
    }

    /// Check if a namespace exists (has any documents)
    pub async fn namespace_exists(&self, namespace: &str) -> Result<bool> {
        let count = self.count_namespace(namespace).await?;
        Ok(count > 0)
    }
}

/// Statistics about the LanceDB table
#[derive(Debug, Clone, Serialize)]
pub struct TableStats {
    pub row_count: usize,
    pub version_count: usize,
    pub table_name: String,
    pub db_path: String,
}

// =============================================================================
// GARBAGE COLLECTION
// =============================================================================

/// Statistics from garbage collection operations
#[derive(Debug, Clone, Default, Serialize)]
pub struct GcStats {
    /// Number of orphan embeddings found (embeddings without valid parent references)
    pub orphans_found: usize,
    /// Number of orphan embeddings removed
    pub orphans_removed: usize,
    /// Number of empty namespaces found
    pub empty_namespaces_found: usize,
    /// Number of empty namespaces removed (documents deleted)
    pub empty_namespaces_removed: usize,
    /// Number of old documents found (older than threshold)
    pub old_docs_found: usize,
    /// Number of old documents removed
    pub old_docs_removed: usize,
    /// Estimated space freed in bytes (if calculable)
    pub bytes_freed: Option<u64>,
    /// List of namespaces that were empty
    pub empty_namespace_names: Vec<String>,
    /// List of namespaces affected by old doc cleanup
    pub affected_namespaces: Vec<String>,
}

impl GcStats {
    /// Check if any issues were found
    pub fn has_issues(&self) -> bool {
        self.orphans_found > 0 || self.empty_namespaces_found > 0 || self.old_docs_found > 0
    }

    /// Check if any deletions occurred
    pub fn has_deletions(&self) -> bool {
        self.orphans_removed > 0 || self.empty_namespaces_removed > 0 || self.old_docs_removed > 0
    }
}

/// Configuration for garbage collection
#[derive(Debug, Clone)]
pub struct GcConfig {
    /// Remove orphan embeddings (embeddings with no parent document)
    pub remove_orphans: bool,
    /// Remove empty namespaces (namespaces with 0 documents)
    pub remove_empty: bool,
    /// Remove documents older than this duration
    pub older_than: Option<chrono::Duration>,
    /// Dry run mode - only report what would be removed
    pub dry_run: bool,
    /// Limit to specific namespace (None = all namespaces)
    pub namespace: Option<String>,
}

impl Default for GcConfig {
    fn default() -> Self {
        Self {
            remove_orphans: false,
            remove_empty: false,
            older_than: None,
            dry_run: true,
            namespace: None,
        }
    }
}

/// Parse a duration string like "30d", "6m", "1y"
pub fn parse_duration_string(s: &str) -> Result<chrono::Duration> {
    let s = s.trim().to_lowercase();
    if s.is_empty() {
        return Err(anyhow!("Empty duration string"));
    }

    // Extract numeric part and unit
    let (num_str, unit) = if s.ends_with('d') {
        (&s[..s.len() - 1], 'd')
    } else if s.ends_with('m') {
        (&s[..s.len() - 1], 'm')
    } else if s.ends_with('y') {
        (&s[..s.len() - 1], 'y')
    } else {
        return Err(anyhow!(
            "Invalid duration format '{}'. Use format like '30d', '6m', or '1y'",
            s
        ));
    };

    let num: i64 = num_str.parse().map_err(|_| {
        anyhow!(
            "Invalid number in duration '{}'. Use format like '30d', '6m', or '1y'",
            s
        )
    })?;

    if num <= 0 {
        return Err(anyhow!("Duration must be positive, got '{}'", s));
    }

    match unit {
        'd' => Ok(chrono::Duration::days(num)),
        'm' => Ok(chrono::Duration::days(num * 30)), // Approximate month
        'y' => Ok(chrono::Duration::days(num * 365)), // Approximate year
        _ => unreachable!(),
    }
}

impl StorageManager {
    // =========================================================================
    // GARBAGE COLLECTION OPERATIONS
    // =========================================================================

    /// Run garbage collection based on configuration
    #[doc(alias = "run_gc")]
    pub async fn garbage_collect(&self, config: &GcConfig) -> Result<GcStats> {
        let mut stats = GcStats::default();

        const PAGE_SIZE: usize = 5000;
        let mut all_docs: Vec<ChromaDocument> = Vec::new();
        let mut offset = 0;
        loop {
            let page = self
                .all_documents_page(config.namespace.as_deref(), offset, PAGE_SIZE)
                .await?;
            let page_len = page.len();
            all_docs.extend(page);
            if page_len < PAGE_SIZE {
                break;
            }
            offset += page_len;
        }

        if all_docs.is_empty() {
            return Ok(stats);
        }

        // Group documents by namespace
        let mut by_namespace: std::collections::HashMap<String, Vec<&ChromaDocument>> =
            std::collections::HashMap::new();
        for doc in &all_docs {
            by_namespace
                .entry(doc.namespace.clone())
                .or_default()
                .push(doc);
        }

        // 1. Find orphan embeddings (documents with parent_id that doesn't exist)
        if config.remove_orphans {
            let orphan_stats = self
                .find_and_remove_orphans(&all_docs, config.dry_run)
                .await?;
            stats.orphans_found = orphan_stats.0;
            stats.orphans_removed = orphan_stats.1;
        }

        // 2. Find and optionally remove empty namespaces
        if config.remove_empty {
            let empty_stats = self
                .find_and_remove_empty_namespaces(&by_namespace, config.dry_run)
                .await?;
            stats.empty_namespaces_found = empty_stats.0;
            stats.empty_namespaces_removed = empty_stats.1;
            stats.empty_namespace_names = empty_stats.2;
        }

        // 3. Find and optionally remove old documents
        if let Some(ref duration) = config.older_than {
            let old_stats = self
                .find_and_remove_old_docs(&all_docs, duration, config.dry_run)
                .await?;
            stats.old_docs_found = old_stats.0;
            stats.old_docs_removed = old_stats.1;
            stats.affected_namespaces = old_stats.2;
        }

        Ok(stats)
    }

    #[deprecated(note = "use garbage_collect")]
    pub async fn run_gc(&self, config: &GcConfig) -> Result<GcStats> {
        self.garbage_collect(config).await
    }

    /// Find orphan embeddings - documents with parent_id pointing to non-existent documents
    async fn find_and_remove_orphans(
        &self,
        docs: &[ChromaDocument],
        dry_run: bool,
    ) -> Result<(usize, usize)> {
        // Build a set of all document IDs
        let all_ids: std::collections::HashSet<&str> = docs.iter().map(|d| d.id.as_str()).collect();

        // Find documents with parent_id that doesn't exist in the ID set
        let mut orphans: Vec<(&str, &str)> = Vec::new(); // (namespace, id)
        for doc in docs {
            if let Some(ref parent_id) = doc.parent_id
                && !all_ids.contains(parent_id.as_str())
            {
                orphans.push((&doc.namespace, &doc.id));
            }
        }

        let found = orphans.len();
        let mut removed = 0;

        if !dry_run && !orphans.is_empty() {
            for (namespace, id) in &orphans {
                if self.delete_document(namespace, id).await.is_ok() {
                    removed += 1;
                }
            }
        }

        Ok((found, removed))
    }

    /// Find empty namespaces - this checks if namespaces have 0 documents
    /// Note: In LanceDB, namespaces are implicit (just a column value), so "removing"
    /// an empty namespace means there are no documents to delete
    async fn find_and_remove_empty_namespaces(
        &self,
        by_namespace: &std::collections::HashMap<String, Vec<&ChromaDocument>>,
        _dry_run: bool,
    ) -> Result<(usize, usize, Vec<String>)> {
        // Find namespaces with 0 documents
        let empty_namespaces: Vec<String> = by_namespace
            .iter()
            .filter(|(_, docs)| docs.is_empty())
            .map(|(ns, _)| ns.clone())
            .collect();

        let found = empty_namespaces.len();
        // Empty namespaces don't need deletion - they have no documents
        // Just report them
        let removed = 0;

        Ok((found, removed, empty_namespaces))
    }

    /// Find and optionally remove documents older than the specified duration
    async fn find_and_remove_old_docs(
        &self,
        docs: &[ChromaDocument],
        older_than: &chrono::Duration,
        dry_run: bool,
    ) -> Result<(usize, usize, Vec<String>)> {
        let cutoff = chrono::Utc::now() - *older_than;

        let mut old_docs: Vec<(&str, &str)> = Vec::new(); // (namespace, id)
        let mut affected_namespaces: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        for doc in docs {
            // Check for timestamp in metadata
            if let Some(obj) = doc.metadata.as_object() {
                let mut doc_timestamp: Option<String> = None;

                // Look for common timestamp field names
                for key in &["timestamp", "created_at", "indexed_at", "date", "time"] {
                    if let Some(value) = obj.get(*key)
                        && let Some(ts) = value.as_str()
                    {
                        doc_timestamp = Some(ts.to_string());
                        break;
                    }
                }

                // Check if document is older than cutoff
                if let Some(ts) = doc_timestamp {
                    // Parse the timestamp - try RFC3339 first, then other formats
                    let is_old = if let Ok(parsed) = chrono::DateTime::parse_from_rfc3339(&ts) {
                        parsed < cutoff
                    } else if let Ok(parsed) =
                        chrono::NaiveDateTime::parse_from_str(&ts, "%Y-%m-%d %H:%M:%S")
                    {
                        parsed < cutoff.naive_utc()
                    } else if let Ok(parsed) = chrono::NaiveDate::parse_from_str(&ts, "%Y-%m-%d") {
                        parsed < cutoff.date_naive()
                    } else {
                        // Can't parse timestamp, skip this document
                        false
                    };

                    if is_old {
                        old_docs.push((&doc.namespace, &doc.id));
                        affected_namespaces.insert(doc.namespace.clone());
                    }
                }
            }
        }

        let found = old_docs.len();
        let mut removed = 0;

        if !dry_run && !old_docs.is_empty() {
            for (namespace, id) in &old_docs {
                if self.delete_document(namespace, id).await.is_ok() {
                    removed += 1;
                }
            }
        }

        Ok((found, removed, affected_namespaces.into_iter().collect()))
    }

    /// List all unique namespaces in the database
    pub async fn list_namespaces(&self) -> Result<Vec<(String, usize)>> {
        self.refresh().await?;

        let mut namespace_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();

        const PAGE_SIZE: usize = 5000;
        let mut offset = 0;
        loop {
            let page = self.all_documents_page(None, offset, PAGE_SIZE).await?;
            let page_len = page.len();
            for doc in &page {
                *namespace_counts.entry(doc.namespace.clone()).or_insert(0) += 1;
            }
            if page_len < PAGE_SIZE {
                break;
            }
            offset += page_len;
        }

        let mut namespaces: Vec<(String, usize)> = namespace_counts.into_iter().collect();
        namespaces.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(namespaces)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn flat_documents_preserve_separate_chunk_and_source_hashes() {
        let doc = ChromaDocument::new_flat_with_hashes(
            "doc-1".to_string(),
            "kb:transcripts".to_string(),
            vec![0.0, 1.0],
            json!({"path": "sample.md"}),
            "outer summary chunk".to_string(),
            "chunk-sha256".to_string(),
            Some("source-sha256".to_string()),
        );

        assert_eq!(doc.content_hash.as_deref(), Some("chunk-sha256"));
        assert_eq!(doc.source_hash.as_deref(), Some("source-sha256"));
        assert_ne!(doc.content_hash, doc.source_hash);
    }
}
