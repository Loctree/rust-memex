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
use lancedb::{Table, connect};
use moka::future::Cache;
use serde::Serialize;
use serde_json::{Value, json};
use sled::Db;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::{debug, info};

use crate::rag::SliceLayer;

/// Schema version for LanceDB tables. Increment when changing table structure.
/// Version 2: Added onion slice fields (layer, parent_id, children_ids, keywords)
/// Version 3: Added content_hash for exact-match deduplication
/// See docs/MIGRATION.md for migration procedures.
pub const SCHEMA_VERSION: u32 = 3;

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
//   - `StorageManager`: LanceDB (vector store) + sled (KV) + moka (cache)
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
    /// SHA256 hash of original content for exact-match deduplication
    pub content_hash: Option<String>,
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
        }
    }

    /// Create a new document with content hash for deduplication
    pub fn new_flat_with_hash(
        id: String,
        namespace: String,
        embedding: Vec<f32>,
        metadata: serde_json::Value,
        document: String,
        content_hash: String,
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
        }
    }

    /// Create a document from an onion slice with content hash for deduplication
    pub fn from_onion_slice_with_hash(
        slice: &crate::rag::OnionSlice,
        namespace: String,
        embedding: Vec<f32>,
        metadata: serde_json::Value,
        content_hash: String,
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
    cache: Arc<Cache<String, Vec<u8>>>,
    db: Option<Db>,
    lance: Connection,
    table: Arc<Mutex<Option<Table>>>,
    collection_name: String,
    lance_path: String,
}

type BatchIter =
    RecordBatchIterator<std::vec::IntoIter<std::result::Result<RecordBatch, ArrowError>>>;

impl StorageManager {
    pub async fn new(cache_mb: usize, db_path: &str) -> Result<Self> {
        // In-memory cache for misc K/V usage
        let cache_bytes = cache_mb * 1024 * 1024;
        let cache = Cache::builder()
            .max_capacity(cache_bytes as u64)
            .time_to_live(Duration::from_secs(3600))
            .build();

        // Persistent K/V for auxiliary state
        // SLED_PATH env allows unique sled per instance when sharing LanceDB
        let sled_path = std::env::var("SLED_PATH")
            .unwrap_or_else(|_| format!("{}/.sled", shellexpand::tilde(db_path)));
        let sled_path = shellexpand::tilde(&sled_path).to_string();
        let db = sled::open(&sled_path)?;

        // Embedded LanceDB path (expand ~, allow override via env)
        let lance_env = std::env::var("LANCEDB_PATH").unwrap_or_else(|_| db_path.to_string());
        let lance_path = if lance_env.trim().is_empty() {
            shellexpand::tilde("~/.rmcp-servers/rmcp-memex/lancedb").to_string()
        } else {
            shellexpand::tilde(&lance_env).to_string()
        };

        let lance = connect(&lance_path).execute().await?;

        Ok(Self {
            cache: Arc::new(cache),
            db: Some(db),
            lance,
            table: Arc::new(Mutex::new(None)),
            collection_name: "mcp_documents".to_string(),
            lance_path,
        })
    }

    /// Create a LanceDB-only storage manager without sled K/V store.
    /// Use this for CLI tools that only need vector operations (index/search).
    /// This allows concurrent access with running MCP server.
    pub async fn new_lance_only(db_path: &str) -> Result<Self> {
        let cache = Cache::builder()
            .max_capacity(64 * 1024 * 1024) // 64MB default for CLI
            .time_to_live(Duration::from_secs(3600))
            .build();

        let lance_path = shellexpand::tilde(db_path).to_string();
        let lance = connect(&lance_path).execute().await?;

        Ok(Self {
            cache: Arc::new(cache),
            db: None,
            lance,
            table: Arc::new(Mutex::new(None)),
            collection_name: "mcp_documents".to_string(),
            lance_path,
        })
    }

    pub fn lance_path(&self) -> &str {
        &self.lance_path
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

    pub async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        if let Some(value) = self.cache.get(key).await {
            return Ok(Some(value));
        }
        if let Some(ref db) = self.db
            && let Some(value) = db.get(key)?
        {
            let vec = value.to_vec();
            self.cache.insert(key.to_string(), vec.clone()).await;
            return Ok(Some(vec));
        }
        Ok(None)
    }

    pub async fn set(&self, key: &str, value: Vec<u8>) -> Result<()> {
        self.cache.insert(key.to_string(), value.clone()).await;
        if let Some(ref db) = self.db {
            db.insert(key, value)?;
            db.flush()?;
        }
        Ok(())
    }

    pub async fn add_to_store(&self, documents: Vec<ChromaDocument>) -> Result<()> {
        if documents.is_empty() {
            return Ok(());
        }
        let dim = documents
            .first()
            .ok_or_else(|| anyhow!("No documents to add"))?
            .embedding
            .len();
        if dim == 0 {
            return Err(anyhow!("Embedding dimension is zero"));
        }

        let table = self.ensure_table(dim).await?;
        let batch = self.docs_to_batch(&documents, dim)?;
        table.add(batch).execute().await?;
        debug!("Inserted {} documents into Lance", documents.len());
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

    pub async fn get_document(&self, namespace: &str, id: &str) -> Result<Option<ChromaDocument>> {
        let table = match self.ensure_table(0).await {
            Ok(t) => t,
            Err(_) => return Ok(None),
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
        let table = match self.ensure_table(0).await {
            Ok(t) => t,
            Err(_) => return Ok(0),
        };
        let predicate = format!(
            "{} AND {}",
            self.namespace_filter(namespace),
            self.id_filter(id)
        );
        let deleted = table.delete(predicate.as_str()).await?;
        Ok(deleted.version as usize)
    }

    pub async fn purge_namespace(&self, namespace: &str) -> Result<usize> {
        let table = match self.ensure_table(0).await {
            Ok(t) => t,
            Err(_) => return Ok(0),
        };
        let predicate = self.namespace_filter(namespace);
        let deleted = table.delete(predicate.as_str()).await?;
        Ok(deleted.version as usize)
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
            // Deduplication field (v3 schema)
            Field::new("content_hash", DataType::Utf8, true), // SHA256 hash for exact-match dedup
        ])
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
        // Content hash for deduplication
        let content_hashes: Vec<Option<&str>> = documents
            .iter()
            .map(|d| d.content_hash.as_deref())
            .collect();

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
                // Content hash for deduplication
                Arc::new(StringArray::from(content_hashes)),
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
            });
        }
        Ok(docs)
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
        // Ensure table exists
        let _ = match self.ensure_table(0).await {
            Ok(t) => t,
            Err(_) => return Ok(vec![]),
        };

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
        format!("layer = {}", layer.as_u8())
    }

    fn content_hash_filter(&self, hash: &str) -> String {
        format!("content_hash = '{}'", hash.replace('\'', "''"))
    }

    /// Check if the table schema has content_hash column (schema v3+)
    async fn table_has_content_hash(table: &Table) -> bool {
        table
            .schema()
            .await
            .map(|schema| schema.field_with_name("content_hash").is_ok())
            .unwrap_or(false)
    }

    /// Check if a content hash already exists in a namespace (for exact-match deduplication)
    ///
    /// Returns Ok(false) if:
    /// - Table doesn't exist yet
    /// - Table has old schema without content_hash column (graceful degradation)
    pub async fn has_content_hash(&self, namespace: &str, hash: &str) -> Result<bool> {
        let table = match self.ensure_table(0).await {
            Ok(t) => t,
            Err(_) => return Ok(false), // Table doesn't exist yet, no duplicates possible
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

        let table = match self.ensure_table(0).await {
            Ok(t) => t,
            Err(_) => return Ok(hashes.iter().collect()), // Table doesn't exist, all are new
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
}
