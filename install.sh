#!/bin/bash
# rmcp-memex installer
# curl -sSf https://raw.githubusercontent.com/VetCoders/rmcp-memex/main/install.sh | sh
# or with custom version:
# RMCP_MEMEX_VERSION=v0.1.13 curl -sSf https://raw.githubusercontent.com/VetCoders/rmcp-memex/main/install.sh | sh

set -euo pipefail

# Configuration
VERSION="${RMCP_MEMEX_VERSION:-latest}"
INSTALL_DIR="${RMCP_MEMEX_INSTALL_DIR:-$HOME/.cargo/bin}"
GITHUB_REPO="VetCoders/rmcp-memex"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() {
    printf "${BLUE}==>${NC} %s\n" "$1"
}

success() {
    printf "${GREEN}==>${NC} %s\n" "$1"
}

warn() {
    printf "${YELLOW}Warning:${NC} %s\n" "$1"
}

error() {
    printf "${RED}Error:${NC} %s\n" "$1" >&2
    exit 1
}

# Detect OS and architecture
detect_platform() {
    local os arch

    os=$(uname -s | tr '[:upper:]' '[:lower:]')
    arch=$(uname -m)

    # Normalize architecture
    case "$arch" in
        x86_64|amd64)
            arch="x86_64"
            ;;
        aarch64|arm64)
            arch="aarch64"
            ;;
        *)
            error "Unsupported architecture: $arch"
            ;;
    esac

    # Map to Rust target triple
    case "$os-$arch" in
        darwin-aarch64)
            echo "aarch64-apple-darwin"
            ;;
        darwin-x86_64)
            echo "x86_64-apple-darwin"
            ;;
        linux-x86_64)
            echo "x86_64-unknown-linux-gnu"
            ;;
        linux-aarch64)
            echo "aarch64-unknown-linux-gnu"
            ;;
        *)
            error "Unsupported platform: $os-$arch"
            ;;
    esac
}

# Get the latest version from GitHub API
get_latest_version() {
    local api_url="https://api.github.com/repos/${GITHUB_REPO}/releases/latest"

    if command -v curl &> /dev/null; then
        curl -sSf "$api_url" 2>/dev/null | grep '"tag_name"' | sed -E 's/.*"([^"]+)".*/\1/' || echo ""
    elif command -v wget &> /dev/null; then
        wget -qO- "$api_url" 2>/dev/null | grep '"tag_name"' | sed -E 's/.*"([^"]+)".*/\1/' || echo ""
    else
        error "Neither curl nor wget found. Please install one of them."
    fi
}

# Download and extract binary
download_and_install() {
    local target="$1"
    local version="$2"
    local install_dir="$3"
    local url
    local temp_dir

    # Construct download URL
    if [ "$version" = "latest" ]; then
        url="https://github.com/${GITHUB_REPO}/releases/latest/download/rmcp-memex-${target}.tar.gz"
    else
        url="https://github.com/${GITHUB_REPO}/releases/download/${version}/rmcp-memex-${target}.tar.gz"
    fi

    info "Downloading from: $url"

    # Create temp directory
    temp_dir=$(mktemp -d)
    trap "rm -rf '$temp_dir'" EXIT

    # Download
    if command -v curl &> /dev/null; then
        if ! curl -sSfL "$url" -o "$temp_dir/rmcp-memex.tar.gz" 2>/dev/null; then
            error "Failed to download from $url"
        fi
    elif command -v wget &> /dev/null; then
        if ! wget -q "$url" -O "$temp_dir/rmcp-memex.tar.gz" 2>/dev/null; then
            error "Failed to download from $url"
        fi
    else
        error "Neither curl nor wget found."
    fi

    # Extract
    info "Extracting..."
    mkdir -p "$temp_dir/extract"
    tar xzf "$temp_dir/rmcp-memex.tar.gz" -C "$temp_dir/extract"

    # Install
    mkdir -p "$install_dir"

    # Find the binary (might be in subdirectory)
    local binary
    binary=$(find "$temp_dir/extract" -name "rmcp_memex" -type f | head -1)

    if [ -z "$binary" ]; then
        error "Binary not found in archive"
    fi

    cp "$binary" "$install_dir/rmcp_memex"
    chmod +x "$install_dir/rmcp_memex"

    success "Installed rmcp_memex to $install_dir/rmcp_memex"
}

# Check if command is available
check_command() {
    command -v "$1" &> /dev/null
}

# Check if directory is in PATH
is_in_path() {
    case ":$PATH:" in
        *":$1:"*) return 0 ;;
        *) return 1 ;;
    esac
}

# Main installation logic
main() {
    echo ""
    echo "  ____  __  __  ____ ____       __  __ _____ __  __ _______  __"
    echo " |  _ \\|  \\/  |/ ___|  _ \\     |  \\/  | ____|  \\/  | ____\\ \\/ /"
    echo " | |_) | |\\/| | |   | |_) |____| |\\/| |  _| | |\\/| |  _|  \\  / "
    echo " |  _ <| |  | | |___|  __/|____| |  | | |___| |  | | |___ /  \\ "
    echo " |_| \\_\\_|  |_|\\____|_|        |_|  |_|_____|_|  |_|_____/_/\\_\\"
    echo ""
    echo "  RAG Memory with Vector Search for MCP"
    echo ""

    # Detect platform
    info "Detecting platform..."
    local target
    target=$(detect_platform)
    success "Platform: $target"

    # Resolve version
    local version="$VERSION"
    if [ "$version" = "latest" ]; then
        info "Fetching latest version..."
        version=$(get_latest_version)
        if [ -z "$version" ]; then
            warn "Could not determine latest version, using 'latest' tag"
            version="latest"
        else
            success "Latest version: $version"
        fi
    fi

    # Download and install
    info "Installing rmcp-memex $version..."
    download_and_install "$target" "$version" "$INSTALL_DIR"

    # Verify installation
    if [ -x "$INSTALL_DIR/rmcp_memex" ]; then
        success "Installation successful!"
        echo ""

        # Get version info
        local installed_version
        installed_version=$("$INSTALL_DIR/rmcp_memex" --version 2>/dev/null || echo "unknown")
        info "Installed version: $installed_version"
    else
        error "Installation verification failed"
    fi

    # Check PATH
    echo ""
    if ! is_in_path "$INSTALL_DIR"; then
        warn "The install directory is not in your PATH."
        echo ""
        echo "  Add this to your shell profile (~/.bashrc, ~/.zshrc, etc.):"
        echo ""
        echo "    export PATH=\"$INSTALL_DIR:\$PATH\""
        echo ""
    fi

    # Next steps
    echo ""
    info "Next steps:"
    echo ""
    echo "  1. Run the configuration wizard:"
    echo "     rmcp_memex wizard"
    echo ""
    echo "  2. Or start the MCP server directly:"
    echo "     rmcp_memex serve"
    echo ""
    echo "  3. Add to your MCP host config (e.g., Claude Desktop):"
    echo ""
    echo "     {\"mcpServers\": {\"rmcp_memex\": {"
    echo "       \"command\": \"$INSTALL_DIR/rmcp_memex\","
    echo "       \"args\": [\"serve\"]"
    echo "     }}}"
    echo ""

    # Launch wizard if interactive
    if [ -t 0 ] && [ -t 1 ]; then
        echo ""
        read -p "Would you like to run the configuration wizard now? [y/N] " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            exec "$INSTALL_DIR/rmcp_memex" wizard
        fi
    fi
}

main "$@"
