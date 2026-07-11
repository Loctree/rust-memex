# Configuration Guide

## Overview

rust-memex can be configured in three ways (in priority order):
1. **CLI flags** - highest priority
2. **TOML configuration file** - medium priority
3. **Default values** - lowest priority

## CLI Options

```bash
rust-memex [OPTIONS] [COMMAND]
```

### Commands

| Command | Description |
|---------|------|
| `serve` | Start the MCP server (default) |
| `wizard` | Interactive configuration wizard |
| `index` | Batch indexing of documents |

### Global Options

| Flag | Description | Default |
|-------|------|-----------|
| `--config <PATH>` | Path to the TOML configuration file | none |
| `--cache-mb <SIZE>` | Cache size in MB | `4096` |
| `--db-path <PATH>` | Path to LanceDB | `~/.rmcp-servers/rust-memex/lancedb` |
| `--max-request-bytes <SIZE>` | Max request size | `5242880` (5MB) |
| `--log-level <LEVEL>` | Log level | `info` |
| `--allowed-paths <PATH>` | Allowed paths (can be repeated) | `$HOME`, `cwd` |
| `--security-enabled` | Enable namespace security | `false` |
| `--token-store-path <PATH>` | Path to the token store | `~/.rmcp-servers/rust-memex/tokens.json` |

### CLI Examples

```bash
# Basic run
rust-memex serve

# With a custom configuration
rust-memex serve --config ~/.rmcp-servers/rust-memex/config.toml

# With security and custom paths
rust-memex serve \
  --security-enabled \
  --allowed-paths ~ \
  --allowed-paths /Volumes/Data \
  --log-level debug

# Batch indexing
rust-memex index ./documents --namespace docs --recursive --glob "*.md"
```

## Configuration File (TOML)

### Location

Default location: `~/.rmcp-servers/rust-memex/config.toml`

### Full Example

```toml
# Cache size in MB
cache_mb = 4096

# Path to the LanceDB vector store
db_path = "~/.rmcp-servers/rust-memex/lancedb"

# Maximum JSON-RPC request size (bytes)
max_request_bytes = 5242880

# Log level: trace, debug, info, warn, error
log_level = "info"

# Whitelist of allowed paths for file operations
# If empty, defaults to $HOME and the current working directory
allowed_paths = [
    "~",
    "/Volumes/Shared/notes",
    "/opt/shared/documents"
]

# Enable namespace token-based access control
security_enabled = true

# Path to the namespace tokens file
token_store_path = "~/.rmcp-servers/rust-memex/tokens.json"
```

### Minimal Configuration

```toml
# Only the essential settings
db_path = "~/.rmcp-servers/rust-memex/lancedb"
security_enabled = true
```

## Server Modes

`rust-memex` exposes a single canonical MCP surface. There is no longer a
separate `memory/full` switch, because it did not actually change the
server contract.

If you want to narrow the runtime:
- use `allowed_paths` to restrict filesystem access
- set `--security-enabled` to protect namespaces with tokens
- set `--auth-token` if you expose mutating HTTP endpoints

## Environment Variables

| Variable | Description |
|---------|------|
| `HOME` / `USERPROFILE` | Home directory (for ~ expansion) |
| `LANCEDB_PATH` | Override the LanceDB path |
| `SLED_PATH` | Override the sled K/V store path |
| `FASTEMBED_CACHE_PATH` | Cache for FastEmbed models |
| `HF_HUB_CACHE` | Cache for HuggingFace models |

## Configuration for Claude/MCP

### ~/.claude.json

```json
{
  "mcpServers": {
    "rust-memex": {
      "command": "rust-memex",
      "args": ["serve", "--config", "~/.rmcp-servers/rust-memex/config.toml"]
    }
  }
}
```

### With security enabled

```json
{
  "mcpServers": {
    "rust-memex": {
      "command": "rust-memex",
      "args": [
        "serve",
        "--security-enabled",
        "--allowed-paths", "~",
        "--allowed-paths", "/Volumes/Data"
      ]
    }
  }
}
```

## Batch Indexing

The `index` command allows bulk indexing of documents.

### Syntax

```bash
rust-memex index <PATH> [OPTIONS]
```

### Options

| Flag | Description |
|-------|------|
| `-n, --namespace <NAME>` | Namespace for the documents (default: `rag`) |
| `-r, --recursive` | Traverse subdirectories recursively |
| `-g, --glob <PATTERN>` | Filter files by a glob pattern |
| `--max-depth <N>` | Maximum depth (0 = no limit) |

### Examples

```bash
# Index a single file
rust-memex index ./README.md

# Index a folder recursively
rust-memex index ./docs --recursive --namespace documentation

# Only markdown files
rust-memex index ./notes --recursive --glob "*.md" --namespace notes

# With a depth limit
rust-memex index ./project --recursive --max-depth 3
```

## Wizard (Configuration Wizard)

An interactive wizard for generating a configuration.

```bash
rust-memex wizard

# Dry run - show changes without saving
rust-memex wizard --dry-run
```

The wizard helps you configure:
- The LanceDB path
- Allowed paths
- Security settings
- Claude integration

## Configuration Priority

When the same option is set in multiple places:

```
CLI flag > Config file > Default value
```

Example:
```bash
# Config file: log_level = "info"
# CLI: --log-level debug
# Result: debug (CLI wins)
rust-memex serve --config ~/.rmcp-servers/rust-memex/config.toml --log-level debug
```

## Configuration Validation

The server validates the configuration at startup:

1. **Paths** - checks that they exist and are accessible
2. **Allowed paths** - resolves ~ and checks permissions
3. **Token store** - creates the file if it does not exist (when security enabled)
4. **LanceDB** - initializes the database if it does not exist

Configuration errors are reported at startup with a clear message.

## Troubleshooting

### "Access denied: path outside allowed directories"

Add the path to `allowed_paths`:
```toml
allowed_paths = [
    "~",
    "/path/to/your/directory"
]
```

### "Cannot resolve config path"

Check that the configuration file exists:
```bash
ls -la ~/.rmcp-servers/rust-memex/config.toml
```

### "Token store not found"

On the first run with `--security-enabled`, the token store is created
automatically. Make sure the parent directory exists:
```bash
mkdir -p ~/.rmcp-servers/rust-memex
```

### Debug logs

```bash
rust-memex serve --log-level trace
```

---

Vibecrafted with AI Agents by Loctree (c)2025 Vetcoders
