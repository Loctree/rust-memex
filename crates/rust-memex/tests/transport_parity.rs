//! Transport Parity Integration Tests
//!
//! Black-box tests proving that stdio and SSE/HTTP transports expose the
//! same MCP contract. Every test dispatches identical JSON-RPC payloads
//! through both `McpTransport::Stdio` and `McpTransport::HttpSse` and
//! asserts structurally equivalent responses.
//!
//! Gated by `e2e-ollama` feature: the full file is only compiled when
//! invoked via `make test-e2e` (or `cargo test --features e2e-ollama
//! --test transport_parity`). `build_mcp_core` requires a reachable
//! embedding provider; tests load the operator's canonical config.toml
//! and rely on its real provider cascade — no hardcoded localhost URLs.
//!
//! Vibecrafted with AI Agents by VetCoders (c)2024-2026 LibraxisAI

#![cfg(feature = "e2e-ollama")]

mod common;

use rust_memex::{
    HybridConfig, McpTransport, NamespaceSecurityConfig, ServerConfig, build_mcp_core,
    dispatch_mcp_payload,
};
use serde_json::{Value, json};
use tempfile::TempDir;

// =============================================================================
// Helpers
// =============================================================================

/// Build a `ServerConfig` from the operator's canonical config.toml,
/// pinned to a fresh temp `db_path` so tests don't touch the live DB.
fn test_server_config(db_path: &str) -> ServerConfig {
    let cfg = common::load_e2e_config().expect("e2e config required");
    ServerConfig {
        cache_mb: 64,
        db_path: db_path.to_string(),
        max_request_bytes: 1024 * 1024,
        log_level: tracing::Level::WARN,
        allowed_paths: vec![],
        security: NamespaceSecurityConfig::default(),
        embeddings: cfg.embeddings,
        hybrid: HybridConfig::default(),
    }
}

/// Dispatch the same payload through both transports.
async fn dispatch_both(
    core: &rust_memex::McpCore,
    payload: &str,
) -> (Option<Value>, Option<Value>) {
    let stdio = dispatch_mcp_payload(core, payload, McpTransport::Stdio).await;
    let sse = dispatch_mcp_payload(core, payload, McpTransport::HttpSse).await;
    (stdio, sse)
}

/// Strip the transport-dependent `transport` field from health results so
/// the rest can be compared.
fn strip_transport_field(mut value: Value) -> Value {
    if let Some(content) = value["result"]["content"].as_array()
        && let Some(first) = content.first()
        && let Some(text) = first["text"].as_str()
        && let Ok(mut parsed) = serde_json::from_str::<Value>(text)
    {
        parsed.as_object_mut().map(|o| o.remove("transport"));
        let stripped = serde_json::to_string(&parsed).unwrap();
        value["result"]["content"][0]["text"] = json!(stripped);
    }
    value
}

// =============================================================================
// Parity Tests
// =============================================================================

#[tokio::test]
async fn parity_initialize_identical_across_transports() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("lance");
    let config = test_server_config(db_path.to_str().unwrap());
    let core = build_mcp_core(config).await.expect("build_mcp_core");

    let payload = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
    let (stdio, sse) = dispatch_both(&core, payload).await;

    let stdio = stdio.expect("stdio must respond to initialize");
    let sse = sse.expect("sse must respond to initialize");

    assert_eq!(
        stdio, sse,
        "initialize responses must be identical across transports"
    );
    assert_eq!(stdio["jsonrpc"], "2.0");
    assert_eq!(stdio["id"], 1);
    assert!(stdio["result"]["protocolVersion"].is_string());
    assert_eq!(stdio["result"]["capabilities"], json!({ "tools": {} }));
}

#[tokio::test]
async fn parity_tools_list_identical_across_transports() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("lance");
    let config = test_server_config(db_path.to_str().unwrap());
    let core = build_mcp_core(config).await.expect("build_mcp_core");

    let payload = r#"{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}"#;
    let (stdio, sse) = dispatch_both(&core, payload).await;

    let stdio = stdio.expect("stdio must respond to tools/list");
    let sse = sse.expect("sse must respond to tools/list");

    assert_eq!(
        stdio, sse,
        "tools/list responses must be identical across transports"
    );

    let tools = stdio["result"]["tools"].as_array().expect("tools array");
    assert!(
        !tools.is_empty(),
        "tools/list must expose at least one tool"
    );
    eprintln!("tools/list exposes {} tools", tools.len());
    // Schema invariant: every tool has a string `name` and `description`.
    // This is robust to additions/removals (no magic count to drift) but
    // still catches malformed entries.
    for (i, tool) in tools.iter().enumerate() {
        assert!(
            tool["name"].is_string(),
            "tool[{i}] missing string name: {tool}"
        );
        assert!(
            tool["description"].is_string(),
            "tool[{i}] missing string description: {tool}"
        );
    }
}

#[tokio::test]
async fn parity_health_tool_structurally_equivalent() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("lance");
    let config = test_server_config(db_path.to_str().unwrap());
    let core = build_mcp_core(config).await.expect("build_mcp_core");

    let payload = r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"health","arguments":{}}}"#;
    let (stdio, sse) = dispatch_both(&core, payload).await;

    let stdio = stdio.expect("stdio must respond to health");
    let sse = sse.expect("sse must respond to health");

    let stdio_stripped = strip_transport_field(stdio.clone());
    let sse_stripped = strip_transport_field(sse.clone());
    assert_eq!(
        stdio_stripped, sse_stripped,
        "health responses must match after stripping transport field"
    );

    let stdio_text: Value =
        serde_json::from_str(stdio["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
    let sse_text: Value =
        serde_json::from_str(sse["result"]["content"][0]["text"].as_str().unwrap()).unwrap();

    assert!(
        stdio_text.get("transport").is_none(),
        "stdio health must NOT include transport field"
    );
    assert_eq!(
        sse_text["transport"], "mcp-over-sse",
        "SSE health must include transport: mcp-over-sse"
    );
}

#[tokio::test]
async fn parity_namespace_security_status_identical() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("lance");
    let config = test_server_config(db_path.to_str().unwrap());
    let core = build_mcp_core(config).await.expect("build_mcp_core");

    let payload = r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"namespace_security_status","arguments":{}}}"#;
    let (stdio, sse) = dispatch_both(&core, payload).await;

    let stdio = stdio.expect("stdio must respond");
    let sse = sse.expect("sse must respond");
    assert_eq!(
        stdio, sse,
        "namespace_security_status must be identical across transports"
    );
}

#[tokio::test]
async fn parity_resources_list_rejected_until_resources_exist() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("lance");
    let config = test_server_config(db_path.to_str().unwrap());
    let core = build_mcp_core(config).await.expect("build_mcp_core");

    let payload = r#"{"jsonrpc":"2.0","id":5,"method":"resources/list","params":{}}"#;
    let (stdio, sse) = dispatch_both(&core, payload).await;

    let stdio = stdio.expect("stdio must respond with error");
    let sse = sse.expect("sse must respond with error");
    assert_eq!(
        stdio, sse,
        "resources/list must fail identically across transports until resources are implemented"
    );
    assert_eq!(stdio["error"]["code"], -32601);
    assert_eq!(stdio["error"]["message"], "Unknown method: resources/list");
}

#[tokio::test]
async fn parity_unknown_tool_error_identical() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("lance");
    let config = test_server_config(db_path.to_str().unwrap());
    let core = build_mcp_core(config).await.expect("build_mcp_core");

    let payload = r#"{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"nonexistent_tool","arguments":{}}}"#;
    let (stdio, sse) = dispatch_both(&core, payload).await;

    let stdio = stdio.expect("stdio must respond with error");
    let sse = sse.expect("sse must respond with error");
    assert_eq!(stdio, sse, "unknown tool errors must be identical");
    assert_eq!(stdio["error"]["code"], -32601);
}

#[tokio::test]
async fn parity_invalid_json_error_identical() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("lance");
    let config = test_server_config(db_path.to_str().unwrap());
    let core = build_mcp_core(config).await.expect("build_mcp_core");

    let payload = r#"{not valid json"#;
    let (stdio, sse) = dispatch_both(&core, payload).await;

    let stdio = stdio.expect("stdio must respond with parse error");
    let sse = sse.expect("sse must respond with parse error");
    assert_eq!(
        stdio, sse,
        "parse errors must be identical across transports"
    );
    assert_eq!(stdio["error"]["code"], -32700);
}

#[tokio::test]
async fn parity_missing_id_error_identical() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("lance");
    let config = test_server_config(db_path.to_str().unwrap());
    let core = build_mcp_core(config).await.expect("build_mcp_core");

    let payload = r#"{"jsonrpc":"2.0","method":"initialize","params":{}}"#;
    let (stdio, sse) = dispatch_both(&core, payload).await;

    let stdio = stdio.expect("stdio must respond with error");
    let sse = sse.expect("sse must respond with error");
    assert_eq!(stdio, sse, "missing id errors must be identical");
    assert_eq!(stdio["error"]["code"], -32600);
}

#[tokio::test]
async fn parity_notification_silent_on_both_transports() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("lance");
    let config = test_server_config(db_path.to_str().unwrap());
    let core = build_mcp_core(config).await.expect("build_mcp_core");

    let payload = r#"{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}"#;
    let (stdio, sse) = dispatch_both(&core, payload).await;

    assert!(stdio.is_none(), "stdio must not respond to notifications");
    assert!(sse.is_none(), "sse must not respond to notifications");
}
