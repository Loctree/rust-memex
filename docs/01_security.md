# Security - Namespace Access Control

## Problem

In a multi-agent environment, where many AI agents may use the same
rust-memex server, data isolation between agents is required. Without an
access-control mechanism:
- Agent A can read Agent B's data
- There is no way to protect sensitive namespaces
- Auditing data access is difficult

In addition, restricting file access to only `$HOME` and `cwd` was too
restrictive - it blocked legitimate use of external volumes (e.g.
`/Volumes/ExternalDrive`).

## Solution

A two-level security system was implemented:

### 1. Configurable Path Whitelist

Instead of a hardcoded restriction to `$HOME` and `cwd`, a configurable
list of allowed paths was introduced.

```toml
# ~/.rmcp-servers/rust-memex/config.toml
allowed_paths = [
    "~",                              # Home directory
    "/Volumes/ExternalDrive/data",    # External volume
    "/opt/shared/documents"           # Shared directory
]
```

**Behavior:**
- If `allowed_paths` is empty → defaults to `$HOME` + `cwd` (backward compatible)
- If `allowed_paths` is set → only those paths are allowed
- Supports `~` expansion to the home directory
- Validation via canonicalization (resolves symlinks)

### 2. Namespace Access Tokens

Token-based access control for namespaces. Protected namespaces require a
token for read/write.

```
┌─────────────────────────────────────────────────────────────┐
│                    Namespace Security                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Public Namespace          Protected Namespace               │
│  ┌─────────────┐          ┌─────────────────────┐           │
│  │  "default"  │          │    "diary"          │           │
│  │             │          │                     │           │
│  │  No token   │          │  Token: rmx_7f3a9b  │           │
│  │  required   │          │  required           │           │
│  └─────────────┘          └─────────────────────┘           │
│                                                              │
│  Agent A: ✓ access        Agent A: ✗ no token               │
│  Agent B: ✓ access        Agent B: ✓ has token              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Usage

### Enabling Security

```bash
# CLI
rust-memex serve --security-enabled

# Or in config.toml
security_enabled = true
token_store_path = "~/.rmcp-servers/rust-memex/tokens.json"
```

### Creating a Token for a Namespace

```json
// MCP Request
{
  "method": "tools/call",
  "params": {
    "name": "namespace_create_token",
    "arguments": {
      "namespace": "diary",
      "description": "Personal diary namespace"
    }
  }
}

// Response
{
  "content": [{
    "type": "text",
    "text": "Token created for namespace 'diary': rmx_a1b2c3d4e5f6..."
  }]
}
```

**IMPORTANT:** The token is returned only once, at creation time. Store it
in a safe place!

### Accessing a Protected Namespace

```json
// Without a token - ERROR
{
  "method": "tools/call",
  "params": {
    "name": "memory_search",
    "arguments": {
      "namespace": "diary",
      "query": "my memories"
    }
  }
}
// Error: "Access denied: namespace 'diary' requires a valid token"

// With a token - OK
{
  "method": "tools/call",
  "params": {
    "name": "memory_search",
    "arguments": {
      "namespace": "diary",
      "query": "my memories",
      "token": "rmx_a1b2c3d4e5f6..."
    }
  }
}
// Success: returns search results
```

### Revoking a Token

```json
{
  "method": "tools/call",
  "params": {
    "name": "namespace_revoke_token",
    "arguments": {
      "namespace": "diary"
    }
  }
}
```

After a token is revoked, the namespace becomes public again.

### Listing Protected Namespaces

```json
{
  "method": "tools/call",
  "params": {
    "name": "namespace_list_protected",
    "arguments": {}
  }
}

// Response
{
  "content": [{
    "type": "text",
    "text": "[\"diary\", \"projects\", \"finance\"]"
  }]
}
```

### Security Status

```json
{
  "method": "tools/call",
  "params": {
    "name": "namespace_security_status",
    "arguments": {}
  }
}

// Response (enabled)
{
  "content": [{
    "type": "text",
    "text": "{\"enabled\":true,\"token_store_path\":\"~/.rmcp-servers/rust-memex/tokens.json\"}"
  }]
}

// Response (disabled)
{
  "content": [{
    "type": "text",
    "text": "{\"enabled\":false,\"message\":\"Namespace security is disabled. All namespaces are publicly accessible.\"}"
  }]
}
```

## Token Format

Tokens have the format: `rmx_<32 hex chars>`

```
rmx_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6
│   └─────────────────────────────────┘
│              32 hex chars (128 bit)
└── prefix "rmx_" (rust-memex)
```

Tokens are generated cryptographically securely (`rand::thread_rng()`).

## Token Store

Tokens are stored in a JSON file:

```json
// ~/.rmcp-servers/rust-memex/tokens.json
{
  "diary": {
    "token_hash": "5e884898da28047d...",
    "created_at": "2024-12-22T10:30:00Z",
    "description": "Personal diary namespace"
  },
  "projects": {
    "token_hash": "d033e22ae348aeb5...",
    "created_at": "2024-12-22T11:00:00Z",
    "description": null
  }
}
```

**Security:**
- Only the **hash** of the token (SHA-256) is stored, not the token itself
- The plaintext token is returned only once, at creation time
- Verification is done by comparing hashes

## Path Validation

The `validate_path()` function protects against path traversal attacks:

```rust
// Blocked patterns
"../../../etc/asdpasswd"     // Path traversal
sd"/etc/passwd"             // Outside allowed paths
"~/../../root/.ssh"       // Traversal after expansion

// Allowed (if in allowed_paths)
"~/Documents/notes.md"    // Under home
"/Volumes/Data/file.txt"  // Configured external volume
```

**Validation:**
1. Check that the path is not empty
2. Expand `~` to the home directory
3. Check for the `..` pattern (path traversal)
4. Canonicalization (resolve symlinks)
5. Check that the canonical path is under an allowed path

## Implementation

### Source Files

| File | Description |
|------|------|
| `rust-memex/src/security/mod.rs` | `NamespaceAccessManager`, `TokenStore`, token generation/verification |
| `rust-memex/src/handlers/mod.rs` | `validate_path()`, integration with the access manager |
| `rust-memex/src/lib.rs` | `NamespaceSecurityConfig`, re-exports |
| `rust-memex/src/bin/rust-memex.rs` | CLI flags `--security-enabled`, `--token-store-path` |

### Key Structures

```rust
/// Security configuration
pub struct NamespaceSecurityConfig {
    pub enabled: bool,
    pub token_store_path: Option<String>,
}

/// Namespace access manager
pub struct NamespaceAccessManager {
    enabled: bool,
    store: Option<Arc<Mutex<TokenStore>>>,
}

/// Token storage
pub struct TokenStore {
    path: PathBuf,
    tokens: HashMap<String, TokenEntry>,
}

/// Token entry
pub struct TokenEntry {
    pub token_hash: String,
    pub created_at: DateTime<Utc>,
    pub description: Option<String>,
}
```

### Tests

```bash
cd rust-memex && cargo test security
```

The tests cover:
- `test_token_generation` - generating tokens in the correct format
- `test_access_manager_disabled` - behavior when security is disabled
- `test_token_store_create_and_verify` - creating and verifying tokens
- `test_access_manager_enabled` - full flow with security enabled

## Best Practices

### For administrators

1. **Enable security in production** - `--security-enabled`
2. **Restrict allowed_paths** - only the necessary paths
3. **Back up the token store** - tokens are irreversible
4. **Rotate tokens** - periodically revoke + create new ones

### For AI agents

1. **Store tokens securely** - env vars or secure storage
2. **Do not log tokens** - avoid displaying them in logs
3. **Use dedicated namespaces** - data isolation
4. **Check security_status** - make sure security is enabled

## Future Extensions

### Phase 3: Namespace Encryption (planned)

```toml
# Future configuration
[namespaces.diary]
encrypted = true
key_derivation = "argon2id"
```

- Data in LanceDB encrypted with the namespace key
- Without the key = data is useless
- For truly sensitive data

---

Vibecrafted with AI Agents by Loctree (c)2025 Vetcoders
