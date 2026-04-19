//! Shared types for MCP host detection.

use serde::{Deserialize, Serialize};

/// Supported MCP host application kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HostKind {
    Codex,
    Cursor,
    VSCode,
    Claude,
    JetBrains,
    Unknown,
}

impl HostKind {
    pub fn as_label(&self) -> &'static str {
        match self {
            HostKind::Codex => "codex",
            HostKind::Cursor => "cursor",
            HostKind::VSCode => "vscode",
            HostKind::Claude => "claude",
            HostKind::JetBrains => "jetbrains",
            HostKind::Unknown => "unknown",
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            HostKind::Codex => "Codex CLI",
            HostKind::Cursor => "Cursor",
            HostKind::VSCode => "VS Code",
            HostKind::Claude => "Claude Desktop",
            HostKind::JetBrains => "JetBrains IDEs",
            HostKind::Unknown => "Unknown",
        }
    }
}

/// Configuration file format for MCP hosts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HostFormat {
    Toml,
    Json,
}
