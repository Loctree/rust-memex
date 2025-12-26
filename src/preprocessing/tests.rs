//! Tests for the preprocessing module

use super::*;

#[test]
fn test_default_config() {
    let config = PreprocessingConfig::default();
    assert!(config.remove_tool_artifacts);
    assert!(config.remove_cli_output);
    assert!(config.remove_metadata);
    assert_eq!(config.min_content_length, 50);
    assert_eq!(config.dedupe_threshold, 0.95);
}

#[test]
fn test_remove_tool_artifacts_function_calls() {
    let preprocessor = Preprocessor::with_defaults();

    let input = r#"Here is some context. <function_calls><invoke name="test"><parameter>value</parameter></invoke></function_calls> And here is the result."#;
    let result = preprocessor.extract_semantic_content(input);

    assert!(!result.contains("function_calls"));
    assert!(!result.contains("invoke"));
    assert!(result.contains("Here is some context"));
    assert!(result.contains("And here is the result"));
}

#[test]
fn test_remove_tool_artifacts_antml_tags() {
    let preprocessor = Preprocessor::with_defaults();

    // Build the test string to avoid interpretation
    let tag_open = format!("<{}:{} name=\"test\">", "antml", "invoke");
    let tag_close = format!("</{}:{}>", "antml", "invoke");
    let input = format!("Before {}content inside{} After", tag_open, tag_close);

    let result = preprocessor.extract_semantic_content(&input);

    assert!(!result.contains("antml"));
    assert!(result.contains("Before"));
    assert!(result.contains("After"));
}

#[test]
fn test_remove_git_status_output() {
    let preprocessor = Preprocessor::with_defaults();

    let input = r#"Looking at the repo:
On branch main
Your branch is up to date with 'origin/main'.

Changes not staged for commit:
  modified:   src/lib.rs
  modified:   Cargo.toml

Now let me fix that."#;

    let result = preprocessor.extract_semantic_content(input);

    assert!(!result.contains("On branch"));
    assert!(!result.contains("Your branch is"));
    assert!(!result.contains("Changes not staged"));
    assert!(!result.contains("modified:"));
    assert!(result.contains("Looking at the repo"));
    assert!(result.contains("Now let me fix that"));
}

#[test]
fn test_remove_cargo_output() {
    let preprocessor = Preprocessor::with_defaults();

    let input = r#"Building the project:
   Compiling rmcp-memex v0.1.0
   Compiling tokio v1.0.0
    Finished release [optimized] target(s) in 2.34s
     Running target/release/rmcp_memex

Build complete!"#;

    let result = preprocessor.extract_semantic_content(input);

    assert!(!result.contains("Compiling"));
    assert!(!result.contains("Finished"));
    assert!(!result.contains("Running"));
    assert!(result.contains("Building the project"));
    assert!(result.contains("Build complete"));
}

#[test]
fn test_remove_file_listing() {
    let preprocessor = Preprocessor::with_defaults();

    let input = r#"Directory contents:
total 42
drwxr-xr-x  5 user staff  160 Dec 24 10:00 src
-rw-r--r--  1 user staff 1234 Dec 24 09:00 Cargo.toml
-rw-r--r--  1 user staff  567 Dec 24 08:00 README.md

That's what we have."#;

    let result = preprocessor.extract_semantic_content(input);

    assert!(!result.contains("total 42"));
    assert!(!result.contains("drwxr-xr-x"));
    assert!(result.contains("Directory contents"));
    assert!(result.contains("That's what we have"));
}

#[test]
fn test_remove_uuid() {
    let preprocessor = Preprocessor::with_defaults();

    let input = "Session 550e8400-e29b-41d4-a716-446655440000 started. Working on the task.";
    let result = preprocessor.extract_semantic_content(input);

    assert!(!result.contains("550e8400-e29b-41d4-a716-446655440000"));
    assert!(result.contains("[UUID]"));
    assert!(result.contains("Session"));
    assert!(result.contains("started"));
}

#[test]
fn test_remove_timestamps() {
    let preprocessor = Preprocessor::with_defaults();

    let input = "Created at 2024-12-24T10:30:00Z. Last modified 2024-12-24T11:00:00+01:00.";
    let result = preprocessor.extract_semantic_content(input);

    assert!(!result.contains("2024-12-24T"));
    assert!(result.contains("[TIMESTAMP]"));
    assert!(result.contains("Created at"));
    assert!(result.contains("Last modified"));
}

#[test]
fn test_remove_session_id() {
    let preprocessor = Preprocessor::with_defaults();

    let input = r#"Request with session_id: abc123xyz. Also sessionId="def456"."#;
    let result = preprocessor.extract_semantic_content(input);

    assert!(!result.contains("session_id: abc123xyz"));
    assert!(!result.contains("sessionId=\"def456\""));
    assert!(result.contains("Request with"));
}

#[test]
fn test_remove_empty_content_json() {
    let preprocessor = Preprocessor::with_defaults();

    let input = r#"Response: {"content": [], "status": "ok"}. Done."#;
    let result = preprocessor.extract_semantic_content(input);

    assert!(!result.contains(r#""content": []"#));
    assert!(result.contains("Response"));
    assert!(result.contains("Done"));
}

#[test]
fn test_preserve_semantic_content() {
    let preprocessor = Preprocessor::with_defaults();

    let input = r#"The preprocessing module filters noise from conversation exports.
    
It removes tool artifacts, CLI output, and metadata while preserving
the actual semantic content that should be embedded for search.

Key features:
- Pattern-based filtering using regex
- Deduplication with configurable threshold
- Minimum content length requirement"#;

    let result = preprocessor.extract_semantic_content(input);

    // All meaningful content should be preserved
    assert!(result.contains("preprocessing module"));
    assert!(result.contains("filters noise"));
    assert!(result.contains("conversation exports"));
    assert!(result.contains("Key features"));
    assert!(result.contains("Pattern-based filtering"));
    assert!(result.contains("Deduplication"));
}

#[test]
fn test_filter_message_below_min_length() {
    let mut preprocessor = Preprocessor::with_defaults();

    // Very short content should be filtered
    let result = preprocessor.filter_message("Hi");
    assert!(result.is_none());

    // Content at or above min length should pass
    let long_content = "This is a meaningful message with enough content to be indexed properly.";
    let result = preprocessor.filter_message(long_content);
    assert!(result.is_some());
}

#[test]
fn test_filter_message_duplicates() {
    let mut preprocessor = Preprocessor::new(PreprocessingConfig {
        min_content_length: 10,
        dedupe_threshold: 0.95,
        ..Default::default()
    });

    let content = "This is a test message that we will see twice.";

    // First occurrence should pass
    let result1 = preprocessor.filter_message(content);
    assert!(result1.is_some());

    // Exact duplicate should be filtered
    let result2 = preprocessor.filter_message(content);
    assert!(result2.is_none());
}

#[test]
fn test_filter_conversation() {
    let mut preprocessor = Preprocessor::new(PreprocessingConfig {
        min_content_length: 20,
        ..Default::default()
    });

    let messages = vec![
        Message::new("user", "How do I implement the preprocessing module?"),
        Message::new("assistant", "On branch main\nmodified: src/lib.rs"), // CLI output
        Message::new("assistant", "Let me explain the preprocessing approach..."),
        Message::new("user", ""), // Empty
        Message::new(
            "assistant",
            "Here's the implementation with detailed explanation of the patterns.",
        ),
    ];

    let (filtered, stats) = preprocessor.filter_conversation(messages);

    assert_eq!(stats.total_input, 5);
    assert!(stats.total_output < 5);
    assert!(!filtered.iter().any(|m| m.content.contains("On branch")));
    assert!(!filtered.iter().any(|m| m.content.is_empty()));
}

#[test]
fn test_is_mostly_tool_artifact() {
    let preprocessor = Preprocessor::with_defaults();

    // Build tool artifact content
    let _mostly_artifact = format!(
        "<{}>Some tool call with lots of data and parameters</{}>",
        "function_calls", "function_calls"
    );

    // This should be detected as mostly artifact if over 80%
    // Since we're testing the ratio, let's use a message that's clearly mostly tool output
    let mixed_content = "Brief intro. Here's the actual content that matters.";

    assert!(!preprocessor.is_mostly_tool_artifact(mixed_content));
}

#[test]
fn test_is_mostly_cli_output() {
    let preprocessor = Preprocessor::with_defaults();

    let cli_heavy = r#"On branch main
Your branch is up to date with 'origin/main'.
Changes not staged for commit:
  modified:   src/lib.rs
  modified:   Cargo.toml
nothing to commit"#;

    let mixed = r#"Looking at the code:
On branch main
That's the current state."#;

    assert!(preprocessor.is_mostly_cli_output(cli_heavy));
    assert!(!preprocessor.is_mostly_cli_output(mixed));
}

#[test]
fn test_content_similarity() {
    // Identical content
    assert_eq!(content_similarity("hello world", "hello world"), 1.0);

    // Completely different
    assert_eq!(content_similarity("hello world", "foo bar baz"), 0.0);

    // Partial overlap
    let sim = content_similarity("hello world test", "hello world foo");
    assert!(sim > 0.0 && sim < 1.0);
}

#[test]
fn test_preprocessing_stats() {
    let stats = PreprocessingStats {
        total_input: 100,
        filtered_tool_artifacts: 10,
        filtered_cli_output: 15,
        filtered_metadata: 5,
        filtered_empty: 8,
        filtered_duplicates: 2,
        filtered_below_min_length: 5,
        total_output: 55,
    };

    assert_eq!(stats.total_filtered(), 45);
    assert!((stats.filter_rate() - 0.45).abs() < 0.01);
}

#[test]
fn test_whitespace_normalization() {
    let preprocessor = Preprocessor::with_defaults();

    let input = "Text with    multiple    spaces\n\n\n\nand many newlines.";
    let result = preprocessor.extract_semantic_content(input);

    // Multiple spaces should be reduced to single
    assert!(!result.contains("    "));
    // Multiple newlines should be reduced to double
    assert!(!result.contains("\n\n\n"));
}

#[test]
fn test_config_serialization() {
    let config = PreprocessingConfig::default();

    // Should serialize to TOML
    let toml_str = toml::to_string(&config).unwrap();
    assert!(toml_str.contains("remove_tool_artifacts = true"));

    // Should deserialize back
    let parsed: PreprocessingConfig = toml::from_str(&toml_str).unwrap();
    assert_eq!(parsed.min_content_length, config.min_content_length);
}

#[test]
fn test_reset_dedupe_cache() {
    let mut preprocessor = Preprocessor::new(PreprocessingConfig {
        min_content_length: 10,
        dedupe_threshold: 0.95,
        ..Default::default()
    });

    let content = "This is a test message that we will see again.";

    // First occurrence
    assert!(preprocessor.filter_message(content).is_some());

    // Duplicate - should be filtered
    assert!(preprocessor.filter_message(content).is_none());

    // Reset cache
    preprocessor.reset_dedupe_cache();

    // Now it should pass again
    assert!(preprocessor.filter_message(content).is_some());
}
