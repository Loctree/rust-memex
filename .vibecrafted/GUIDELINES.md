# rmcp-memex Session Guidelines

## Current Truth

- `rmcp-memex` is a Rust MCP memory/RAG server with dual transport: stdio and HTTP/SSE.
- The highest-impact hubs are `src/rag/mod.rs`, `src/embeddings/mod.rs`, `src/storage/mod.rs`, and `src/http/mod.rs`.
- `src/mcp_protocol.rs` is the shared MCP contract surface. Treat it as the canonical transport truth.
- Embedding dimensions are config-driven and enforced against provider reality.

## Working Rules

- Run loctree first for structural context before broad edits or deletes.
- Treat `rag`, `embeddings`, `storage`, and `http` as high-risk modules. Prefer narrow delegations over sweeping rewrites there.
- Keep Semgrep aimed at runtime truth. Inline test harness noise is excluded from the local `no-unwrap` rule so the gate stays actionable.
- Prefer replacing duplicated behavior by delegating to a single runtime source of truth instead of growing parallel wrappers.

## Quality Gates

- `semgrep --config .semgrep.yaml --error`
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo test --all-features`

## Product Notes

- A green build is not enough; verify the shared MCP contract and install path remain coherent.
- Current sharp tech debt is twin surface area across large modules, especially around memory/search wrappers.
