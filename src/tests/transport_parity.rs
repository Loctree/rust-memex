//! Transport parity tests for the shared MCP protocol core.
//!
//! Both stdio and HTTP/SSE route through `McpCore::handle_jsonrpc_request`.
//! These tests pin the single contract surface and prove that a future drift
//! between transports would fail loudly instead of silently shipping.
//!
//! ## Coverage matrix
//!
//! | Surface              | Static | Dispatch (requires MLX) |
//! |----------------------|--------|-------------------------|
//! | initialize           | ✓      | ✓                       |
//! | tools/list           | ✓      | ✓                       |
//! | tools/call (health)  |        | ✓ (known difference)    |
//! | tools/call (unknown) |        | ✓                       |
//! | unknown method       |        | ✓                       |
//! | missing id           |        | ✓                       |
//! | notifications        |        | ✓                       |
//! | JSON-RPC framing     | ✓      |                         |
//!
//! ## Known transport differences
//!
//! The `health` tool includes a `"transport": "mcp-over-sse"` field when called
//! via `McpTransport::HttpSse`. This is intentional — clients use it for
//! diagnostics. All other tool responses are transport-independent.

use crate::mcp_protocol::{
    McpCore, McpTransport, PROTOCOL_VERSION, SERVER_NAME, jsonrpc_error,
    jsonrpc_success, shared_initialize_result, shared_tools_list_result,
};
use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn tool_names() -> Vec<String> {
    shared_tools_list_result()["tools"]
        .as_array()
        .expect("tools list should be an array")
        .iter()
        .filter_map(|tool| tool["name"].as_str().map(ToOwned::to_owned))
        .collect()
}

/// The canonical tool surface. Every add or remove MUST update this list.
const EXPECTED_TOOLS: &[&str] = &[
    "health",
    "rag_index",
    "rag_index_text",
    "rag_search",
    "memory_upsert",
    "memory_get",
    "memory_search",
    "memory_delete",
    "memory_purge_namespace",
    "namespace_create_token",
    "namespace_revoke_token",
    "namespace_list_protected",
    "namespace_security_status",
    "dive",
];

// ---------------------------------------------------------------------------
// Static contract tests (no infrastructure required)
// ---------------------------------------------------------------------------

#[test]
fn initialize_result_has_correct_protocol_version() {
    let result = shared_initialize_result();
    assert_eq!(result["protocolVersion"], PROTOCOL_VERSION);
}

#[test]
fn initialize_result_has_server_info() {
    let result = shared_initialize_result();
    assert_eq!(result["serverInfo"]["name"], SERVER_NAME);
    assert!(
        result["serverInfo"]["version"].is_string(),
        "version must be a string"
    );
}

#[test]
fn initialize_result_has_tools_and_resources_capabilities() {
    let result = shared_initialize_result();
    assert!(result["capabilities"]["tools"].is_object());
    assert!(result["capabilities"]["resources"].is_object());
}

#[test]
fn initialize_result_has_no_extra_top_level_keys() {
    let result = shared_initialize_result();
    let obj = result.as_object().expect("initialize result should be an object");
    let keys: Vec<&String> = obj.keys().collect();
    assert!(keys.contains(&&"protocolVersion".to_string()));
    assert!(keys.contains(&&"serverInfo".to_string()));
    assert!(keys.contains(&&"capabilities".to_string()));
    assert_eq!(keys.len(), 3, "unexpected keys in initialize result: {:?}", keys);
}

#[test]
fn unified_tool_surface_matches_expected_names() {
    let names = tool_names();

    for tool in EXPECTED_TOOLS {
        assert!(
            names.contains(&tool.to_string()),
            "Expected shared tool '{}' not found in {:?}",
            tool,
            names
        );
    }

    assert_eq!(
        names.len(),
        EXPECTED_TOOLS.len(),
        "Tool count mismatch. Got {:?}, expected {:?}",
        names,
        EXPECTED_TOOLS
    );
}

#[test]
fn unified_tool_surface_includes_previously_drifted_http_gaps() {
    let names = tool_names();

    // These were historically missing from the HTTP surface before unification.
    assert!(names.contains(&"rag_index".to_string()));
    assert!(names.contains(&"memory_purge_namespace".to_string()));
    assert!(names.contains(&"namespace_create_token".to_string()));
    assert!(names.contains(&"dive".to_string()));
}

#[test]
fn all_tools_have_valid_input_schema() {
    for tool in shared_tools_list_result()["tools"]
        .as_array()
        .expect("tools list should be an array")
    {
        let name = tool["name"].as_str().unwrap_or("<unknown>");
        let schema = &tool["inputSchema"];

        assert_eq!(
            schema["type"], "object",
            "Tool '{}' inputSchema.type must be 'object'",
            name
        );
        assert!(
            schema["properties"].is_object(),
            "Tool '{}' must have properties object",
            name
        );
        assert!(
            schema["required"].is_array(),
            "Tool '{}' must have required array",
            name
        );
    }
}

#[test]
fn tools_with_required_fields_declare_them_in_properties() {
    for tool in shared_tools_list_result()["tools"]
        .as_array()
        .expect("tools list should be an array")
    {
        let name = tool["name"].as_str().unwrap_or("<unknown>");
        let properties = &tool["inputSchema"]["properties"];
        let required = tool["inputSchema"]["required"]
            .as_array()
            .expect("required should be array");

        for req in required {
            let req_name = req.as_str().unwrap_or("");
            assert!(
                properties.get(req_name).is_some(),
                "Tool '{}' declares required field '{}' but it is missing from properties",
                name,
                req_name
            );
        }
    }
}

#[test]
fn tools_list_result_wraps_in_tools_key() {
    let result = shared_tools_list_result();
    assert!(result["tools"].is_array());
    assert_eq!(
        result["tools"].as_array().unwrap().len(),
        EXPECTED_TOOLS.len()
    );
}

// ---------------------------------------------------------------------------
// JSON-RPC framing tests
// ---------------------------------------------------------------------------

#[test]
fn jsonrpc_success_includes_id_when_present() {
    let resp = jsonrpc_success(&json!(42), json!({"ok": true}));
    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(resp["id"], 42);
    assert_eq!(resp["result"]["ok"], true);
    assert!(resp.get("error").is_none());
}

#[test]
fn jsonrpc_success_omits_id_when_null() {
    let resp = jsonrpc_success(&Value::Null, json!({"ok": true}));
    assert_eq!(resp["jsonrpc"], "2.0");
    assert!(resp.get("id").is_none());
}

#[test]
fn jsonrpc_success_with_string_id() {
    let resp = jsonrpc_success(&json!("req-abc"), json!(42));
    assert_eq!(resp["id"], "req-abc");
    assert_eq!(resp["result"], 42);
}

#[test]
fn jsonrpc_error_includes_id_when_present() {
    let request_id = json!("req-1");
    let resp = jsonrpc_error(Some(&request_id), -32601, "Not found");
    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(resp["id"], "req-1");
    assert_eq!(resp["error"]["code"], -32601);
    assert_eq!(resp["error"]["message"], "Not found");
    assert!(resp.get("result").is_none());
}

#[test]
fn jsonrpc_error_omits_id_when_null() {
    let resp = jsonrpc_error(Some(&Value::Null), -32700, "Parse error");
    assert_eq!(resp["jsonrpc"], "2.0");
    assert!(resp.get("id").is_none());
    assert_eq!(resp["error"]["code"], -32700);
}

#[test]
fn jsonrpc_error_omits_id_when_none() {
    let resp = jsonrpc_error(None, -32600, "Bad request");
    assert_eq!(resp["jsonrpc"], "2.0");
    assert!(resp.get("id").is_none());
    assert_eq!(resp["error"]["code"], -32600);
}

// ---------------------------------------------------------------------------
// Dispatch parity tests (require McpCore — need real infrastructure)
// ---------------------------------------------------------------------------

/// Build a McpCore backed by a temporary LanceDB + the configured embedding server.
/// Returns None if the embedding server is unreachable (test should skip).
async fn try_build_mcp_core() -> Option<McpCore> {
    use crate::{
        EmbeddingConfig, EmbeddingClient, ProviderConfig,
        rag::RAGPipeline,
        security::{NamespaceAccessManager, NamespaceSecurityConfig},
        storage::StorageManager,
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;

    let config = EmbeddingConfig {
        required_dimension: 4096,
        max_batch_chars: 32000,
        max_batch_items: 16,
        providers: vec![ProviderConfig {
            name: "test-local".to_string(),
            base_url: "http://localhost:12345".to_string(),
            model: "test-model".to_string(),
            priority: 1,
            endpoint: "/v1/embeddings".to_string(),
        }],
        ..Default::default()
    };

    let embedding_client = match EmbeddingClient::new(&config).await {
        Ok(c) => Arc::new(Mutex::new(c)),
        Err(_) => {
            eprintln!("⚠ Skipping dispatch parity test: MLX server unavailable at localhost:12345");
            return None;
        }
    };

    let tmp = tempfile::tempdir().ok()?;
    let db_path = tmp.path().join(".lancedb");
    // Leak the tempdir so it outlives the test — the OS cleans up /tmp anyway
    let db_path_str = db_path.to_string_lossy().to_string();
    std::mem::forget(tmp);

    let storage = Arc::new(StorageManager::new(&db_path_str).await.ok()?);
    let rag = Arc::new(RAGPipeline::new(embedding_client.clone(), storage).await.ok()?);

    let access_manager = Arc::new(
        NamespaceAccessManager::new(NamespaceSecurityConfig::default()),
    );
    let _ = access_manager.init().await;

    Some(McpCore::new(
        rag,
        None, // no hybrid searcher needed for protocol tests
        embedding_client,
        vec![],
        access_manager,
    ))
}

macro_rules! require_mcp_core {
    ($core:expr) => {
        match $core {
            Some(core) => core,
            None => return,
        }
    };
}

/// Dispatch an `initialize` request through both transports and assert identical results.
#[tokio::test]
async fn dispatch_initialize_parity_across_transports() {
    let core = require_mcp_core!(try_build_mcp_core().await);

    let request = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {}
    });

    let stdio_resp = core
        .handle_jsonrpc_request(request.clone(), McpTransport::Stdio)
        .await
        .into_option()
        .expect("initialize must return a response");

    let http_resp = core
        .handle_jsonrpc_request(request, McpTransport::HttpSse)
        .await
        .into_option()
        .expect("initialize must return a response");

    assert_eq!(
        stdio_resp, http_resp,
        "initialize response must be identical across transports"
    );

    // Verify the content is correct
    assert_eq!(stdio_resp["result"]["protocolVersion"], PROTOCOL_VERSION);
    assert_eq!(stdio_resp["result"]["serverInfo"]["name"], SERVER_NAME);
    assert_eq!(stdio_resp["id"], 1);
}

/// Dispatch a `tools/list` request through both transports and assert identical results.
#[tokio::test]
async fn dispatch_tools_list_parity_across_transports() {
    let core = require_mcp_core!(try_build_mcp_core().await);

    let request = json!({
        "jsonrpc": "2.0",
        "id": "list-1",
        "method": "tools/list",
        "params": {}
    });

    let stdio_resp = core
        .handle_jsonrpc_request(request.clone(), McpTransport::Stdio)
        .await
        .into_option()
        .expect("tools/list must return a response");

    let http_resp = core
        .handle_jsonrpc_request(request, McpTransport::HttpSse)
        .await
        .into_option()
        .expect("tools/list must return a response");

    assert_eq!(
        stdio_resp, http_resp,
        "tools/list response must be identical across transports"
    );

    let tool_count = stdio_resp["result"]["tools"]
        .as_array()
        .expect("tools should be an array")
        .len();
    assert_eq!(tool_count, EXPECTED_TOOLS.len());
}

/// Unknown method returns the same error shape for both transports.
#[tokio::test]
async fn dispatch_unknown_method_parity() {
    let core = require_mcp_core!(try_build_mcp_core().await);

    let request = json!({
        "jsonrpc": "2.0",
        "id": 99,
        "method": "nonexistent/method",
        "params": {}
    });

    let stdio_resp = core
        .handle_jsonrpc_request(request.clone(), McpTransport::Stdio)
        .await
        .into_option()
        .expect("unknown method must return error");

    let http_resp = core
        .handle_jsonrpc_request(request, McpTransport::HttpSse)
        .await
        .into_option()
        .expect("unknown method must return error");

    assert_eq!(
        stdio_resp, http_resp,
        "unknown method error must be identical across transports"
    );
    assert_eq!(stdio_resp["error"]["code"], -32601);
    assert!(
        stdio_resp["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("nonexistent/method")
    );
}

/// Unknown tool name inside tools/call returns the same error for both transports.
#[tokio::test]
async fn dispatch_unknown_tool_parity() {
    let core = require_mcp_core!(try_build_mcp_core().await);

    let request = json!({
        "jsonrpc": "2.0",
        "id": "tool-1",
        "method": "tools/call",
        "params": {
            "name": "nonexistent_tool",
            "arguments": {}
        }
    });

    let stdio_resp = core
        .handle_jsonrpc_request(request.clone(), McpTransport::Stdio)
        .await
        .into_option()
        .expect("unknown tool must return error");

    let http_resp = core
        .handle_jsonrpc_request(request, McpTransport::HttpSse)
        .await
        .into_option()
        .expect("unknown tool must return error");

    assert_eq!(
        stdio_resp, http_resp,
        "unknown tool error must be identical across transports"
    );
    assert_eq!(stdio_resp["error"]["code"], -32601);
}

/// Missing `id` field produces the same error for both transports.
#[tokio::test]
async fn dispatch_missing_id_parity() {
    let core = require_mcp_core!(try_build_mcp_core().await);

    let request = json!({
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {}
    });

    let stdio_resp = core
        .handle_jsonrpc_request(request.clone(), McpTransport::Stdio)
        .await
        .into_option()
        .expect("missing id must return error");

    let http_resp = core
        .handle_jsonrpc_request(request, McpTransport::HttpSse)
        .await
        .into_option()
        .expect("missing id must return error");

    assert_eq!(
        stdio_resp, http_resp,
        "missing id error must be identical across transports"
    );
    assert_eq!(stdio_resp["error"]["code"], -32600);
}

/// Notifications are swallowed (return None) by both transports.
#[tokio::test]
async fn dispatch_notification_parity() {
    let core = require_mcp_core!(try_build_mcp_core().await);

    let request = json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {}
    });

    let stdio_result = core
        .handle_jsonrpc_request(request.clone(), McpTransport::Stdio)
        .await
        .into_option();

    let http_result = core
        .handle_jsonrpc_request(request, McpTransport::HttpSse)
        .await
        .into_option();

    assert!(stdio_result.is_none(), "stdio should swallow notification");
    assert!(http_result.is_none(), "http should swallow notification");
}

// ---------------------------------------------------------------------------
// Known transport difference: health tool `transport` field
// ---------------------------------------------------------------------------

/// The `health` tool intentionally differs between transports:
/// - Stdio: no `transport` field in the result
/// - HttpSse: includes `"transport": "mcp-over-sse"`
///
/// This test documents and pins that difference so it is not accidentally
/// removed or changed without updating both transports.
#[tokio::test]
async fn health_tool_transport_field_difference_is_intentional() {
    let core = require_mcp_core!(try_build_mcp_core().await);

    let request = json!({
        "jsonrpc": "2.0",
        "id": "health-1",
        "method": "tools/call",
        "params": {
            "name": "health",
            "arguments": {}
        }
    });

    let stdio_resp = core
        .handle_jsonrpc_request(request.clone(), McpTransport::Stdio)
        .await
        .into_option()
        .expect("health must return a response");

    let http_resp = core
        .handle_jsonrpc_request(request, McpTransport::HttpSse)
        .await
        .into_option()
        .expect("health must return a response");

    // Both must succeed with content array
    let stdio_text = stdio_resp["result"]["content"][0]["text"]
        .as_str()
        .expect("stdio health should have text");
    let http_text = http_resp["result"]["content"][0]["text"]
        .as_str()
        .expect("http health should have text");

    let stdio_json: Value = serde_json::from_str(stdio_text).expect("stdio health should be JSON");
    let http_json: Value = serde_json::from_str(http_text).expect("http health should be JSON");

    // Stdio: no transport field
    assert!(
        stdio_json.get("transport").is_none(),
        "stdio health should NOT have transport field"
    );

    // HttpSse: has transport field
    assert_eq!(
        http_json["transport"], "mcp-over-sse",
        "http health should have transport: mcp-over-sse"
    );

    // Everything else should match
    assert_eq!(stdio_json["version"], http_json["version"]);
    assert_eq!(stdio_json["backend"], http_json["backend"]);
    assert_eq!(stdio_json["db_path"], http_json["db_path"]);
}
