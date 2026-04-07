//! Agent tools for MCP-compatible memory operations.
//!
//! These functions provide a tool-oriented interface to MemexEngine,
//! suitable for use with AI agents via MCP or similar protocols.
//!
//! # Example
//!
//! ```rust,ignore
//! use rmcp_memex::{MemexEngine, MemexConfig};
//! use rmcp_memex::tools::{memory_store, memory_search, tool_definitions, ToolResult};
//! use serde_json::json;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let engine = MemexEngine::for_app("my-app", "documents").await?;
//!
//!     // Store a document using the tool API
//!     let result = memory_store(
//!         &engine,
//!         "doc-1".to_string(),
//!         "Important patient notes about feline diabetes".to_string(),
//!         json!({"patient_id": "P-123", "doc_type": "notes"}),
//!     ).await?;
//!     assert!(result.success);
//!
//!     // Search for documents
//!     let results = memory_search(&engine, "diabetes".to_string(), 5, None).await?;
//!     println!("Found {} documents", results.len());
//!
//!     // Get tool definitions for MCP registration
//!     let tools = tool_definitions();
//!     println!("Available tools: {:?}", tools.iter().map(|t| &t.name).collect::<Vec<_>>());
//!
//!     Ok(())
//! }
//! ```

use crate::engine::{BatchResult, MemexEngine, MetaFilter, StoreItem};
use crate::rag::SearchResult;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

// =============================================================================
// TOOL RESULT
// =============================================================================

/// Result type for tool operations.
///
/// Provides a consistent response format for all tool operations,
/// suitable for MCP tool call responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Whether the operation succeeded
    pub success: bool,
    /// Human-readable message describing the result
    pub message: String,
    /// Optional data payload (operation-specific)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl ToolResult {
    /// Create a success result with just a message
    pub fn ok(message: impl Into<String>) -> Self {
        Self {
            success: true,
            message: message.into(),
            data: None,
        }
    }

    /// Create a success result with data
    pub fn ok_with_data(message: impl Into<String>, data: Value) -> Self {
        Self {
            success: true,
            message: message.into(),
            data: Some(data),
        }
    }

    /// Create an error result
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            success: false,
            message: message.into(),
            data: None,
        }
    }
}

// =============================================================================
// TOOL FUNCTIONS
// =============================================================================

/// Store text in memory with automatic embedding generation.
///
/// # Arguments
/// * `engine` - The MemexEngine instance
/// * `id` - Unique document identifier
/// * `text` - Text content to embed and store
/// * `metadata` - Additional metadata (JSON object)
///
/// # Returns
/// `ToolResult` indicating success or failure
///
/// # Example
///
/// ```rust,ignore
/// let result = memory_store(
///     &engine,
///     "visit-123".to_string(),
///     "Patient presented with lethargy...".to_string(),
///     json!({"patient_id": "P-456", "visit_type": "checkup"}),
/// ).await?;
/// ```
pub async fn memory_store(
    engine: &MemexEngine,
    id: String,
    text: String,
    metadata: Value,
) -> Result<ToolResult> {
    match engine.store(&id, &text, metadata).await {
        Ok(()) => Ok(ToolResult::ok(format!(
            "Document '{}' stored successfully",
            id
        ))),
        Err(e) => Ok(ToolResult::error(format!(
            "Failed to store document '{}': {}",
            id, e
        ))),
    }
}

/// Search memory semantically using vector similarity.
///
/// # Arguments
/// * `engine` - The MemexEngine instance
/// * `query` - Search query text
/// * `limit` - Maximum number of results to return
/// * `filter` - Optional metadata filter for narrowing results
///
/// # Returns
/// Vector of `SearchResult` ordered by relevance (highest score first)
///
/// # Example
///
/// ```rust,ignore
/// // Simple search
/// let results = memory_search(&engine, "diabetes symptoms".to_string(), 10, None).await?;
///
/// // Filtered search
/// let filter = MetaFilter::for_patient("P-456");
/// let results = memory_search(&engine, "diabetes".to_string(), 10, Some(filter)).await?;
/// ```
pub async fn memory_search(
    engine: &MemexEngine,
    query: String,
    limit: usize,
    filter: Option<MetaFilter>,
) -> Result<Vec<SearchResult>> {
    match filter {
        Some(f) => engine.search_filtered(&query, f, limit).await,
        None => engine.search(&query, limit).await,
    }
}

/// Get a document by ID.
///
/// # Arguments
/// * `engine` - The MemexEngine instance
/// * `id` - Document identifier to retrieve
///
/// # Returns
/// `Option<SearchResult>` - The document if found, None otherwise
///
/// # Example
///
/// ```rust,ignore
/// if let Some(doc) = memory_get(&engine, "visit-123".to_string()).await? {
///     println!("Found: {}", doc.text);
/// }
/// ```
pub async fn memory_get(engine: &MemexEngine, id: String) -> Result<Option<SearchResult>> {
    engine.get(&id).await
}

/// Delete a document by ID.
///
/// # Arguments
/// * `engine` - The MemexEngine instance
/// * `id` - Document identifier to delete
///
/// # Returns
/// `ToolResult` indicating success or failure, with deletion status
///
/// # Example
///
/// ```rust,ignore
/// let result = memory_delete(&engine, "visit-123".to_string()).await?;
/// if result.success {
///     println!("Document deleted");
/// }
/// ```
pub async fn memory_delete(engine: &MemexEngine, id: String) -> Result<ToolResult> {
    match engine.delete(&id).await {
        Ok(true) => Ok(ToolResult::ok(format!(
            "Document '{}' deleted successfully",
            id
        ))),
        Ok(false) => Ok(ToolResult::ok_with_data(
            format!("Document '{}' not found", id),
            json!({"deleted": false}),
        )),
        Err(e) => Ok(ToolResult::error(format!(
            "Failed to delete document '{}': {}",
            id, e
        ))),
    }
}

/// Batch store multiple documents efficiently.
///
/// More efficient than calling `memory_store()` multiple times as embeddings
/// are generated in batches.
///
/// # Arguments
/// * `engine` - The MemexEngine instance
/// * `items` - Vector of items to store
///
/// # Returns
/// `BatchResult` with success/failure counts
///
/// # Example
///
/// ```rust,ignore
/// let items = vec![
///     StoreItem::new("doc-1", "First document").with_metadata(json!({"type": "note"})),
///     StoreItem::new("doc-2", "Second document").with_metadata(json!({"type": "note"})),
/// ];
/// let result = memory_store_batch(&engine, items).await?;
/// println!("Stored {} documents", result.success_count);
/// ```
pub async fn memory_store_batch(
    engine: &MemexEngine,
    items: Vec<StoreItem>,
) -> Result<BatchResult> {
    engine.store_batch(items).await
}

/// Delete all documents matching a metadata filter.
///
/// This is the primary method for GDPR-compliant data deletion.
///
/// # Arguments
/// * `engine` - The MemexEngine instance
/// * `filter` - Metadata filter specifying which documents to delete
///
/// # Returns
/// `ToolResult` with count of deleted documents
///
/// # Example
///
/// ```rust,ignore
/// // GDPR request - delete all patient data
/// let filter = MetaFilter::for_patient("P-456");
/// let result = memory_delete_by_filter(&engine, filter).await?;
/// if let Some(data) = result.data {
///     println!("Deleted {} documents", data["deleted_count"]);
/// }
/// ```
pub async fn memory_delete_by_filter(
    engine: &MemexEngine,
    filter: MetaFilter,
) -> Result<ToolResult> {
    match engine.delete_by_filter(filter.clone()).await {
        Ok(count) => Ok(ToolResult::ok_with_data(
            format!("Deleted {} documents matching filter", count),
            json!({
                "deleted_count": count,
                "filter": filter,
            }),
        )),
        Err(e) => Ok(ToolResult::error(format!(
            "Failed to delete by filter: {}",
            e
        ))),
    }
}

// =============================================================================
// TOOL DEFINITIONS FOR MCP
// =============================================================================

/// MCP tool definition schema.
///
/// This structure matches the MCP (Model Context Protocol) tool definition format,
/// allowing these tools to be registered with MCP-compatible AI agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool name (used for invocation)
    pub name: String,
    /// Human-readable description of what the tool does
    pub description: String,
    /// JSON Schema for the tool's input parameters
    pub input_schema: Value,
}

impl ToolDefinition {
    /// Create a new tool definition
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
        }
    }
}

/// Get all tool definitions for MCP registration.
///
/// Returns a vector of `ToolDefinition` that can be used to register these
/// tools with an MCP server or any compatible AI agent framework.
///
/// # Example
///
/// ```rust,ignore
/// let tools = tool_definitions();
/// for tool in &tools {
///     println!("Tool: {} - {}", tool.name, tool.description);
/// }
/// ```
pub fn tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition::new(
            "memory_store",
            "Store text in memory with automatic embedding generation. Use for saving documents, notes, or any text that should be searchable later.",
            json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Unique document identifier"
                    },
                    "text": {
                        "type": "string",
                        "description": "Text content to embed and store"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Additional metadata (patient_id, visit_id, doc_type, etc.)",
                        "additionalProperties": true
                    }
                },
                "required": ["id", "text"]
            }),
        ),
        ToolDefinition::new(
            "memory_search",
            "Search memory semantically using natural language. Returns documents ranked by relevance to the query.",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 100
                    },
                    "filter": {
                        "type": "object",
                        "description": "Optional metadata filter",
                        "properties": {
                            "patient_id": {
                                "type": "string",
                                "description": "Filter by patient ID"
                            },
                            "visit_id": {
                                "type": "string",
                                "description": "Filter by visit ID"
                            },
                            "doc_type": {
                                "type": "string",
                                "description": "Filter by document type"
                            },
                            "date_from": {
                                "type": "string",
                                "description": "Filter by date range start (ISO 8601)"
                            },
                            "date_to": {
                                "type": "string",
                                "description": "Filter by date range end (ISO 8601)"
                            }
                        }
                    }
                },
                "required": ["query"]
            }),
        ),
        ToolDefinition::new(
            "memory_get",
            "Retrieve a specific document by its ID. Returns the full document with text, metadata, and similarity score.",
            json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Document identifier to retrieve"
                    }
                },
                "required": ["id"]
            }),
        ),
        ToolDefinition::new(
            "memory_delete",
            "Delete a specific document by its ID. Returns confirmation of deletion.",
            json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Document identifier to delete"
                    }
                },
                "required": ["id"]
            }),
        ),
        ToolDefinition::new(
            "memory_store_batch",
            "Store multiple documents in a single batch operation. More efficient than storing documents individually.",
            json!({
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "description": "Array of documents to store",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": "Unique document identifier"
                                },
                                "text": {
                                    "type": "string",
                                    "description": "Text content to embed and store"
                                },
                                "metadata": {
                                    "type": "object",
                                    "description": "Additional metadata",
                                    "additionalProperties": true
                                }
                            },
                            "required": ["id", "text"]
                        }
                    }
                },
                "required": ["items"]
            }),
        ),
        ToolDefinition::new(
            "memory_delete_by_filter",
            "Delete all documents matching a metadata filter. Primary method for GDPR-compliant data deletion (e.g., delete all patient data).",
            json!({
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "object",
                        "description": "Metadata filter specifying which documents to delete",
                        "properties": {
                            "patient_id": {
                                "type": "string",
                                "description": "Delete all documents for this patient (GDPR)"
                            },
                            "visit_id": {
                                "type": "string",
                                "description": "Delete all documents for this visit"
                            },
                            "doc_type": {
                                "type": "string",
                                "description": "Delete all documents of this type"
                            },
                            "date_from": {
                                "type": "string",
                                "description": "Delete documents from this date onwards"
                            },
                            "date_to": {
                                "type": "string",
                                "description": "Delete documents up to this date"
                            }
                        }
                    }
                },
                "required": ["filter"]
            }),
        ),
    ]
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_result_ok() {
        let result = ToolResult::ok("Success");
        assert!(result.success);
        assert_eq!(result.message, "Success");
        assert!(result.data.is_none());
    }

    #[test]
    fn test_tool_result_ok_with_data() {
        let result = ToolResult::ok_with_data("Success", json!({"count": 42}));
        assert!(result.success);
        assert_eq!(result.message, "Success");
        assert_eq!(result.data.unwrap()["count"], 42);
    }

    #[test]
    fn test_tool_result_error() {
        let result = ToolResult::error("Something went wrong");
        assert!(!result.success);
        assert_eq!(result.message, "Something went wrong");
        assert!(result.data.is_none());
    }

    #[test]
    fn test_tool_definitions_count() {
        let tools = tool_definitions();
        assert_eq!(tools.len(), 6);
    }

    #[test]
    fn test_tool_definitions_names() {
        let tools = tool_definitions();
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();

        assert!(names.contains(&"memory_store"));
        assert!(names.contains(&"memory_search"));
        assert!(names.contains(&"memory_get"));
        assert!(names.contains(&"memory_delete"));
        assert!(names.contains(&"memory_store_batch"));
        assert!(names.contains(&"memory_delete_by_filter"));
    }

    #[test]
    fn test_tool_definitions_have_required_fields() {
        let tools = tool_definitions();

        for tool in tools {
            assert!(!tool.name.is_empty(), "Tool name should not be empty");
            assert!(
                !tool.description.is_empty(),
                "Tool description should not be empty"
            );
            assert!(
                tool.input_schema.is_object(),
                "Input schema should be an object"
            );
            assert!(
                tool.input_schema.get("type").is_some(),
                "Input schema should have a type field"
            );
            assert!(
                tool.input_schema.get("properties").is_some(),
                "Input schema should have properties"
            );
        }
    }

    #[test]
    fn test_tool_result_serialization() {
        let result = ToolResult::ok_with_data("Success", json!({"id": "doc-1"}));
        let json_str = serde_json::to_string(&result).unwrap();

        assert!(json_str.contains("\"success\":true"));
        assert!(json_str.contains("\"message\":\"Success\""));
        assert!(json_str.contains("\"data\""));
    }

    #[test]
    fn test_tool_definition_creation() {
        let tool = ToolDefinition::new(
            "test_tool",
            "A test tool",
            json!({
                "type": "object",
                "properties": {
                    "input": { "type": "string" }
                }
            }),
        );

        assert_eq!(tool.name, "test_tool");
        assert_eq!(tool.description, "A test tool");
        assert!(tool.input_schema["properties"]["input"].is_object());
    }
}
