#!/usr/bin/env bash
set -euo pipefail

echo "=== rust-memex source setup ==="
echo "For prebuilt GitHub Release bundles, use ../install.sh instead."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "Building release binary..."
cargo build --locked --release --bin rust-memex

if [[ "${1:-}" == "--bundle-macos" ]]; then
  echo "Creating macOS app bundle..."
  "$SCRIPT_DIR/build-macos.sh"
  BIN_PATH="$HOME/.mcp-servers/rust-memex.app/Contents/MacOS/rust-memex"
else
  BIN_PATH="$REPO_ROOT/target/release/rust-memex"
fi

echo ""
echo "=== Done ==="
echo "Binary: $BIN_PATH"
echo ""
echo "Example MCP host config:"
cat <<JSON
{
  "mcpServers": {
    "rust-memex": {
      "command": "$BIN_PATH",
      "args": ["serve"]
    }
  }
}
JSON

echo ""
echo "Notes:"
echo "- The MLX HTTP bridge is optional. By default, DRAGON_BASE_URL=http://localhost."
echo "- To force local-only embeddings, set DISABLE_MLX=1."
