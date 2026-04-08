# Release Runbook

`rmcp-memex` is ready to release when the code path, distribution path, and
public-facing path all agree on the same product shape: one canonical binary,
one installer, one transport contract, and one story for new users.

## Preflight

Run these commands from the repo root before cutting a tag:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo check --no-default-features
cargo package --allow-dirty
```

If Ollama is available locally with the release embedding model already pulled,
also run:

```bash
cargo test --test transport_parity -- --ignored
```

## Versioning

1. Update `Cargo.toml` with the release version.
2. Move the relevant notes from `Unreleased` in `CHANGELOG.md` into a dated release section.
3. Commit the release prep.
4. Tag with `vX.Y.Z` or `vX.Y.Z-rc.N`.

## Publish Path

Pushing the tag triggers `.github/workflows/release.yml`, which now:

- builds the canonical `rmcp-memex` binary for macOS and Linux targets
- packages the binary with `README`, licenses, and installer script
- generates `rmcp-memex-sha256sums.txt`
- publishes GitHub Release notes with install and MCP config snippets

## Smoke Tests After Publish

Verify the public release surface, not just the repo:

```bash
# Installer path
curl -LsSf https://raw.githubusercontent.com/VetCoders/rmcp-memex/main/install.sh | sh
rmcp-memex --version

# Manual artifact path
curl -LO https://github.com/VetCoders/rmcp-memex/releases/latest/download/rmcp-memex-x86_64-unknown-linux-gnu.tar.gz
curl -LO https://github.com/VetCoders/rmcp-memex/releases/latest/download/rmcp-memex-sha256sums.txt
grep rmcp-memex-x86_64-unknown-linux-gnu.tar.gz rmcp-memex-sha256sums.txt

# Cargo path
cargo install rmcp-memex --version <version>
```

Then confirm:

- `rmcp-memex serve` starts cleanly
- the aliases `rust-memex`, `rmmx`, and `rmemex` resolve after installer-based installation
- `/health` responds when running with `--http-port 6660`
- the landing page publishes via `.github/workflows/pages.yml`

## Public Surface

The project now ships with a static landing page source in `docs/index.html`.
Keep it synchronized with the actual install path, transport story, and release
badges. If the product story changes, update the page in the same PR as the code.
