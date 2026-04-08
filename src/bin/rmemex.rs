//! rmemex CLI - Convenience alias
//!
//! This binary is strictly a convenience alias for `rmcp-memex`.
//! The canonical product name and primary entrypoint is `rmcp-memex`.
//! Both binaries share the exact same codebase and behavior (stdio MCP & HTTP/SSE daemon).
//!
//! Usage:
//!   rmemex <command> [args]
//!
//! Vibecrafted with AI Agents by VetCoders (c)2026 VetCoders

// Include the same main code as rmcp_memex.rs
// This allows both binaries to share the implementation
include!("rmcp_memex.rs");
