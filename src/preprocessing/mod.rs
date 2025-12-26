//! Preprocessing module for filtering noise from conversation exports before embedding.
//!
//! Problem: ~36-40% of conversation exports are NOISE (tool outputs, metadata, CLI commands).
//! This module filters noise BEFORE embedding to save vector space and improve search quality.

use lazy_static::lazy_static;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use tracing::info;

#[cfg(test)]
mod tests;

/// Configuration for the preprocessing pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    /// Remove MCP tool artifacts (function_calls, invoke, parameter, etc.)
    #[serde(default = "default_true")]
    pub remove_tool_artifacts: bool,

    /// Remove CLI command outputs (git status, cargo build, file listings)
    #[serde(default = "default_true")]
    pub remove_cli_output: bool,

    /// Remove system metadata (timestamps, UUIDs, session IDs)
    #[serde(default = "default_true")]
    pub remove_metadata: bool,

    /// Minimum content length after cleaning (characters)
    #[serde(default = "default_min_length")]
    pub min_content_length: usize,

    /// Threshold for deduplication (0.0-1.0, based on content hash similarity)
    #[serde(default = "default_dedupe_threshold")]
    pub dedupe_threshold: f32,

    /// Remove empty/boilerplate content
    #[serde(default = "default_true")]
    pub remove_empty_content: bool,

    /// Remove duplicate headers and repeated system prompts
    #[serde(default = "default_true")]
    pub remove_duplicate_headers: bool,
}

fn default_true() -> bool {
    true
}

fn default_min_length() -> usize {
    50
}

fn default_dedupe_threshold() -> f32 {
    0.95
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            remove_tool_artifacts: true,
            remove_cli_output: true,
            remove_metadata: true,
            min_content_length: 50,
            dedupe_threshold: 0.95,
            remove_empty_content: true,
            remove_duplicate_headers: true,
        }
    }
}

lazy_static! {
    // MCP Tool Artifacts patterns - XML-like tags used by Claude/MCP
    // Using string concatenation to avoid XML tag interpretation
    static ref FUNCTION_CALLS_BLOCK: Regex = Regex::new(
        &format!(r"(?s)<{}>{}</{}>" , "function_calls", r".*?", "function_calls")
    ).unwrap();

    static ref ANTML_INVOKE_BLOCK: Regex = Regex::new(
        &format!(r"(?s)<{}:{}[^>]*>.*?</{}:{}>" , "antml", "invoke", "antml", "invoke")
    ).unwrap();

    static ref ANTML_PARAMETER_BLOCK: Regex = Regex::new(
        &format!(r"(?s)<{}:{}[^>]*>.*?</{}:{}>" , "antml", "parameter", "antml", "parameter")
    ).unwrap();

    static ref FUNCTION_RESULTS_BLOCK: Regex = Regex::new(
        &format!(r"(?s)<{}>{}</{}>" , "function_results", r".*?", "function_results")
    ).unwrap();

    static ref RESULT_BLOCK: Regex = Regex::new(
        &format!(r"(?s)<{}>{}</{}>" , "result", r".*?", "result")
    ).unwrap();

    // Generic XML-like tool output tags
    static ref TOOL_OUTPUT_TAGS: Regex = Regex::new(
        r"(?s)<(output|name|value)>.*?</(output|name|value)>"
    ).unwrap();

    // CLI Output patterns
    static ref GIT_STATUS_OUTPUT: Regex = Regex::new(
        r"(?m)^\s*(On branch|Your branch|Changes (?:not staged|to be committed)|Untracked files|nothing to commit|modified:|new file:|deleted:).*$"
    ).unwrap();

    // Git diff patterns - only match actual diff context, not markdown lists
    static ref GIT_DIFF_OUTPUT: Regex = Regex::new(
        r"(?m)^(diff --git|index [0-9a-f]+\.\.[0-9a-f]+|--- a/|--- /|\+\+\+ a/|\+\+\+ b/|@@\s*-\d+.*@@|Binary files).*$"
    ).unwrap();

    static ref CARGO_OUTPUT: Regex = Regex::new(
        r"(?m)^(\s*(Compiling|Finished|Running|warning:|error\[E|-->|note:|help:)).*$"
    ).unwrap();

    static ref NPM_OUTPUT: Regex = Regex::new(
        r"(?m)^(npm (WARN|ERR!|notice)|added \d+ packages|up to date|audited \d+ packages).*$"
    ).unwrap();

    static ref FILE_LISTING: Regex = Regex::new(
        r"(?m)^(total \d+|[drwx-]{10}\s+\d+|[-lrwx]{10}\s+\d+).*$"
    ).unwrap();

    static ref TREE_OUTPUT: Regex = Regex::new(
        r"(?m)^[│├└─\s]+[\w.-]+/?$"
    ).unwrap();

    // System Metadata patterns
    static ref UUID_PATTERN: Regex = Regex::new(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    ).unwrap();

    static ref TIMESTAMP_ISO: Regex = Regex::new(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:?\d{2})?"
    ).unwrap();

    static ref UNIX_TIMESTAMP: Regex = Regex::new(
        r"\b1[6-7]\d{8}\b"
    ).unwrap();

    static ref SESSION_ID_PATTERN: Regex = Regex::new(
        r#"(session_id|sessionId|session-id|conv_id|conversation_id)["']?\s*[:=]\s*["']?[\w-]+"#
    ).unwrap();

    static ref FILE_PATH_METADATA: Regex = Regex::new(
        r#""(path|file_path|filepath)"\s*:\s*"[^"]+""#
    ).unwrap();

    // Empty/Boilerplate patterns
    static ref EMPTY_CONTENT_JSON: Regex = Regex::new(
        r#""content"\s*:\s*\[\s*\]"#
    ).unwrap();

    static ref EMPTY_TEXT_JSON: Regex = Regex::new(
        r#""text"\s*:\s*"""#
    ).unwrap();

    static ref PLACEHOLDER_MESSAGE: Regex = Regex::new(
        r"(?i)(placeholder|lorem ipsum|TODO:|FIXME:|XXX:)"
    ).unwrap();

    // Duplicate Header patterns (system prompts, instructions)
    static ref SYSTEM_PROMPT_HEADER: Regex = Regex::new(
        r"(?i)(you are an? (AI|assistant|helpful)|as an AI|I am Claude|I'm Claude)"
    ).unwrap();

    // Match instruction blocks until double newline (no look-ahead - unsupported in regex crate)
    static ref INSTRUCTION_BLOCK: Regex = Regex::new(
        r"(?is)(instructions?:|guidelines?:|rules?:)[^\n]*(?:\n[^\n]+)*"
    ).unwrap();

    // Whitespace normalization
    static ref MULTIPLE_NEWLINES: Regex = Regex::new(r"\n{3,}").unwrap();
    static ref MULTIPLE_SPACES: Regex = Regex::new(r" {2,}").unwrap();
}

/// A simple hash for deduplication based on normalized content
fn content_hash(s: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let normalized = s
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");

    let mut hasher = DefaultHasher::new();
    normalized.hash(&mut hasher);
    hasher.finish()
}

/// Calculate similarity between two content hashes (Jaccard-like for words)
fn content_similarity(a: &str, b: &str) -> f32 {
    let words_a: HashSet<&str> = a.split_whitespace().collect();
    let words_b: HashSet<&str> = b.split_whitespace().collect();

    if words_a.is_empty() && words_b.is_empty() {
        return 1.0;
    }

    let intersection = words_a.intersection(&words_b).count();
    let union = words_a.union(&words_b).count();

    if union == 0 {
        return 0.0;
    }

    intersection as f32 / union as f32
}

/// Message structure for conversation filtering
#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
    pub metadata: Option<serde_json::Value>,
}

/// Statistics from preprocessing
#[derive(Debug, Clone, Default)]
pub struct PreprocessingStats {
    pub total_input: usize,
    pub filtered_tool_artifacts: usize,
    pub filtered_cli_output: usize,
    pub filtered_metadata: usize,
    pub filtered_empty: usize,
    pub filtered_duplicates: usize,
    pub filtered_below_min_length: usize,
    pub total_output: usize,
}

impl PreprocessingStats {
    pub fn total_filtered(&self) -> usize {
        self.filtered_tool_artifacts
            + self.filtered_cli_output
            + self.filtered_metadata
            + self.filtered_empty
            + self.filtered_duplicates
            + self.filtered_below_min_length
    }

    pub fn filter_rate(&self) -> f32 {
        if self.total_input == 0 {
            return 0.0;
        }
        self.total_filtered() as f32 / self.total_input as f32
    }
}

/// The main preprocessor for filtering conversation noise
pub struct Preprocessor {
    config: PreprocessingConfig,
    seen_hashes: HashSet<u64>,
}

impl Preprocessor {
    /// Create a new preprocessor with the given configuration
    pub fn new(config: PreprocessingConfig) -> Self {
        Self {
            config,
            seen_hashes: HashSet::new(),
        }
    }

    /// Create a preprocessor with default configuration
    pub fn with_defaults() -> Self {
        Self::new(PreprocessingConfig::default())
    }

    /// Reset the deduplication cache (useful between conversations)
    pub fn reset_dedupe_cache(&mut self) {
        self.seen_hashes.clear();
    }

    /// Filter a single message content. Returns None if the content should be skipped entirely.
    pub fn filter_message(&mut self, content: &str) -> Option<String> {
        // Extract semantic content first
        let cleaned = self.extract_semantic_content(content);

        // Check minimum length
        if cleaned.len() < self.config.min_content_length {
            return None;
        }

        // Check for duplicates
        if self.config.dedupe_threshold < 1.0 {
            let hash = content_hash(&cleaned);
            if self.seen_hashes.contains(&hash) {
                return None;
            }
            self.seen_hashes.insert(hash);
        }

        Some(cleaned)
    }

    /// Filter a conversation (list of messages), returning only meaningful content
    pub fn filter_conversation(
        &mut self,
        messages: Vec<Message>,
    ) -> (Vec<Message>, PreprocessingStats) {
        let mut stats = PreprocessingStats {
            total_input: messages.len(),
            ..Default::default()
        };

        let mut result = Vec::new();
        let mut previous_contents: Vec<String> = Vec::new();

        for msg in messages {
            // Skip if content is mostly tool artifacts
            if self.config.remove_tool_artifacts && self.is_mostly_tool_artifact(&msg.content) {
                stats.filtered_tool_artifacts += 1;
                continue;
            }

            // Skip if content is mostly CLI output
            if self.config.remove_cli_output && self.is_mostly_cli_output(&msg.content) {
                stats.filtered_cli_output += 1;
                continue;
            }

            // Extract semantic content
            let cleaned = self.extract_semantic_content(&msg.content);

            // Skip empty content
            if self.config.remove_empty_content && cleaned.trim().is_empty() {
                stats.filtered_empty += 1;
                continue;
            }

            // Skip if below minimum length
            if cleaned.len() < self.config.min_content_length {
                stats.filtered_below_min_length += 1;
                continue;
            }

            // Check for duplicates/near-duplicates
            if self.config.dedupe_threshold < 1.0 {
                let is_duplicate = previous_contents
                    .iter()
                    .any(|prev| content_similarity(prev, &cleaned) >= self.config.dedupe_threshold);

                if is_duplicate {
                    stats.filtered_duplicates += 1;
                    continue;
                }
            }

            previous_contents.push(cleaned.clone());

            result.push(Message {
                role: msg.role,
                content: cleaned,
                metadata: msg.metadata,
            });
        }

        stats.total_output = result.len();

        info!(
            "Preprocessing complete: {}/{} messages kept ({:.1}% filtered)",
            stats.total_output,
            stats.total_input,
            stats.filter_rate() * 100.0
        );

        (result, stats)
    }

    /// Extract semantic content from raw text, removing noise patterns
    pub fn extract_semantic_content(&self, raw: &str) -> String {
        let mut result = raw.to_string();

        // Remove tool artifacts
        if self.config.remove_tool_artifacts {
            result = FUNCTION_CALLS_BLOCK.replace_all(&result, "").to_string();
            result = ANTML_INVOKE_BLOCK.replace_all(&result, "").to_string();
            result = ANTML_PARAMETER_BLOCK.replace_all(&result, "").to_string();
            result = FUNCTION_RESULTS_BLOCK.replace_all(&result, "").to_string();
            result = RESULT_BLOCK.replace_all(&result, "").to_string();
            result = TOOL_OUTPUT_TAGS.replace_all(&result, "").to_string();
        }

        // Remove CLI output
        if self.config.remove_cli_output {
            result = GIT_STATUS_OUTPUT.replace_all(&result, "").to_string();
            result = GIT_DIFF_OUTPUT.replace_all(&result, "").to_string();
            result = CARGO_OUTPUT.replace_all(&result, "").to_string();
            result = NPM_OUTPUT.replace_all(&result, "").to_string();
            result = FILE_LISTING.replace_all(&result, "").to_string();
            result = TREE_OUTPUT.replace_all(&result, "").to_string();
        }

        // Remove metadata
        if self.config.remove_metadata {
            result = UUID_PATTERN.replace_all(&result, "[UUID]").to_string();
            result = TIMESTAMP_ISO
                .replace_all(&result, "[TIMESTAMP]")
                .to_string();
            result = UNIX_TIMESTAMP
                .replace_all(&result, "[TIMESTAMP]")
                .to_string();
            result = SESSION_ID_PATTERN.replace_all(&result, "").to_string();
            result = FILE_PATH_METADATA.replace_all(&result, "").to_string();
        }

        // Remove empty content patterns
        if self.config.remove_empty_content {
            result = EMPTY_CONTENT_JSON.replace_all(&result, "").to_string();
            result = EMPTY_TEXT_JSON.replace_all(&result, "").to_string();
            result = PLACEHOLDER_MESSAGE.replace_all(&result, "").to_string();
        }

        // Normalize whitespace
        result = MULTIPLE_NEWLINES.replace_all(&result, "\n\n").to_string();
        result = MULTIPLE_SPACES.replace_all(&result, " ").to_string();

        result.trim().to_string()
    }

    /// Check if content is predominantly tool artifacts
    fn is_mostly_tool_artifact(&self, content: &str) -> bool {
        let original_len = content.len();
        if original_len == 0 {
            return false;
        }

        let mut cleaned = content.to_string();
        cleaned = FUNCTION_CALLS_BLOCK.replace_all(&cleaned, "").to_string();
        cleaned = ANTML_INVOKE_BLOCK.replace_all(&cleaned, "").to_string();
        cleaned = ANTML_PARAMETER_BLOCK.replace_all(&cleaned, "").to_string();
        cleaned = FUNCTION_RESULTS_BLOCK.replace_all(&cleaned, "").to_string();
        cleaned = RESULT_BLOCK.replace_all(&cleaned, "").to_string();

        let remaining_len = cleaned.trim().len();
        let artifact_ratio = 1.0 - (remaining_len as f32 / original_len as f32);

        // If more than 80% was tool artifacts, consider it mostly artifacts
        artifact_ratio > 0.8
    }

    /// Check if content is predominantly CLI output
    fn is_mostly_cli_output(&self, content: &str) -> bool {
        let lines: Vec<&str> = content.lines().collect();
        if lines.is_empty() {
            return false;
        }

        let cli_lines = lines
            .iter()
            .filter(|line| {
                GIT_STATUS_OUTPUT.is_match(line)
                    || GIT_DIFF_OUTPUT.is_match(line)
                    || CARGO_OUTPUT.is_match(line)
                    || NPM_OUTPUT.is_match(line)
                    || FILE_LISTING.is_match(line)
                    || TREE_OUTPUT.is_match(line)
            })
            .count();

        let cli_ratio = cli_lines as f32 / lines.len() as f32;

        // If more than 70% of lines are CLI output, consider it mostly CLI
        cli_ratio > 0.7
    }
}

#[cfg(test)]
impl Message {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            metadata: None,
        }
    }
}
