//! TUI Configuration Wizard for rmcp_memex
//!
//! Interactive terminal UI for configuring MCP server and host integrations.
//!
//! ## Wizard Flow
//! 1. Welcome - Brief introduction
//! 2. EmbedderSetup - Auto-detect Ollama/MLX + configure providers
//! 3. MemexSettings - Database path, cache size, etc.
//! 4. HostSelection - Detect and select MCP hosts
//! 5. SnippetPreview - Preview config snippets
//! 6. HealthCheck - Verify embedder connectivity + dimension
//! 7. DataSetup - Optional directory indexing
//! 8. Summary - Write config and finish

mod app;
mod detection;
mod health;
mod host_detection;
mod indexer;
mod path_utils;
mod ui;

pub use app::{WizardConfig, run_wizard};
pub use detection::{
    DetectedProvider, ProviderKind, ProviderStatus, check_custom_endpoint, detect_providers,
};
pub use health::{CheckStatus, HealthCheckItem, HealthCheckResult, HealthChecker};
pub use host_detection::{HostDetection, HostFormat, HostKind, detect_hosts};
pub use indexer::{
    DataSetupOption, DataSetupState, DataSetupSubStep, ImportMode, IndexProgress, collect_files,
    import_lancedb, start_indexing, validate_path,
};
