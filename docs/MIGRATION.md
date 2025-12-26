# Schema Migration Guide

This document describes how to handle schema changes in rmcp-memex storage layers.

## Schema Versioning

The current schema version is defined in `src/storage/mod.rs`:

```rust
pub const SCHEMA_VERSION: u32 = 3;
```

Increment this version whenever you make breaking changes to:
- LanceDB table structure (columns, embeddings dimension)
- Sled key-value schema
- ChromaDocument fields

## Current Schema (v3)

### LanceDB Table: `mcp_documents`

| Column | Type | Description |
|--------|------|-------------|
| `id` | String | Unique document identifier |
| `namespace` | String | Namespace for isolation |
| `embedding` | FixedSizeList[Float32, 4096] | MLX embedding vector |
| `metadata` | String (JSON) | Serialized metadata object |
| `document` | String | Original text content |
| `layer` | u8 | Onion slice layer (1=Outer, 2=Middle, 3=Inner, 4=Core, 0=legacy) |
| `parent_id` | String (nullable) | Parent slice ID in hierarchy |
| `children_ids` | List[String] | Children slice IDs |
| `keywords` | List[String] | Extracted keywords for this slice |
| `content_hash` | String (nullable) | SHA256 hash for exact-match dedup |

### Schema History

- **v1**: Initial schema (id, namespace, embedding, metadata, document)
- **v2**: Added onion slice fields (layer, parent_id, children_ids, keywords)
- **v3**: Added content_hash for exact-match deduplication

### Sled Keys

| Key Pattern | Value | Description |
|-------------|-------|-------------|
| `{namespace}:{id}` | JSON bytes | Document lookup cache |

## Migration Procedures

### Before Any Migration

1. **Backup your data**:
   ```bash
   cp -r ~/.rmcp-servers/rmcp-memex/lancedb ~/.rmcp-servers/rmcp-memex/lancedb.backup
   cp -r ~/.rmcp-servers/sled ~/.rmcp-servers/sled.backup
   ```

2. **Stop all rmcp-memex instances**

### Migration Strategies

#### Strategy A: Re-index (Recommended for small datasets)

Best for datasets < 100K documents or when embeddings model changes.

```bash
# 1. Backup
cp -r ~/.rmcp-servers/rmcp-memex/lancedb ~/.rmcp-servers/rmcp-memex/lancedb.backup

# 2. Delete old data
rm -rf ~/.rmcp-servers/rmcp-memex/lancedb

# 3. Re-index your documents
# (Use your indexing scripts or MCP tools)
```

#### Strategy B: In-place Migration (Advanced)

For large datasets where re-indexing is expensive.

```rust
// Example migration script (pseudocode)
use rmcp_memex::storage::StorageManager;

async fn migrate_v2_to_v3(storage: &StorageManager) -> Result<()> {
    // 1. Read all documents from old schema
    // 2. Compute content_hash for each document
    // 3. Write to new table with content_hash
    // 4. Swap tables atomically
    Ok(())
}
```

### Version-Specific Migrations

#### v2 → v3

Added `content_hash` field for exact-match deduplication. Documents without this field will have `None` and deduplication will fall back to embedding-based comparison.

```bash
# Simple migration: just restart - new documents get content_hash
rmcp-memex serve --db-path ~/.rmcp-servers/rmcp-memex/lancedb
```

## Adding a Migration

When changing the schema:

1. Increment `SCHEMA_VERSION` in `src/storage/mod.rs`
2. Document the new schema in this file
3. Add migration procedure from previous version
4. Add a test that creates old-schema data and verifies migration
5. Update CHANGELOG.md

## Testing Migrations

```bash
# Run migration tests
cargo test migration

# Manual verification
cargo run -- --db-path /tmp/test_migration_db
```

## Rollback

If migration fails:

```bash
# Restore from backup
rm -rf ~/.rmcp-servers/rmcp-memex/lancedb
cp -r ~/.rmcp-servers/rmcp-memex/lancedb.backup ~/.rmcp-servers/rmcp-memex/lancedb
```

## FAQ

**Q: Do I need to migrate if I only update rmcp-memex?**

A: Only if the SCHEMA_VERSION changes. Check the CHANGELOG for "breaking: schema change" entries.

**Q: Can I run different schema versions side by side?**

A: Not recommended. Use separate `--db-path` for each version during testing.

**Q: How do I check my current schema version?**

A: Currently there's no runtime check. The schema version is implied by the rmcp-memex version you're running. Future versions may store schema version in the database.
