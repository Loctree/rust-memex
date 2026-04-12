use super::{OnionSlice, OnionSliceConfig, SliceLayer, extract_keywords};

#[derive(Debug)]
pub struct SemanticBlock {
    pub role: String,
    pub content: String,
}

pub fn is_structured_conversation(metadata: &serde_json::Value) -> bool {
    metadata.as_object().is_some_and(|object| {
        matches!(
            object.get("type").and_then(|value| value.as_str()),
            Some("conversation" | "transcript_turn")
        ) || matches!(
            object.get("format").and_then(|value| value.as_str()),
            Some("markdown_transcript" | "sessions" | "claude_web" | "chatgpt")
        )
    })
}

pub fn parse_blocks(content: &str, metadata: &serde_json::Value) -> Vec<SemanticBlock> {
    let mut blocks = Vec::new();
    let format = metadata.get("format").and_then(|v| v.as_str()).unwrap_or("");
    
    if format == "markdown_transcript" {
        let mut current_role = String::new();
        let mut current_content = String::new();
        for line in content.lines() {
            if line.starts_with("User request:") {
                if !current_role.is_empty() {
                    blocks.push(SemanticBlock { role: current_role, content: current_content.trim().to_string() });
                }
                current_role = "User".to_string();
                current_content = String::new();
            } else if line.starts_with("Assistant response:") {
                if !current_role.is_empty() {
                    blocks.push(SemanticBlock { role: current_role, content: current_content.trim().to_string() });
                }
                current_role = "Assistant".to_string();
                current_content = String::new();
            } else if line.starts_with("Reasoning focus:") {
                if !current_role.is_empty() {
                    blocks.push(SemanticBlock { role: current_role, content: current_content.trim().to_string() });
                }
                current_role = "Reasoning".to_string();
                current_content = String::new();
            } else {
                if !current_content.is_empty() {
                    current_content.push('\n');
                }
                current_content.push_str(line);
            }
        }
        if !current_role.is_empty() {
            blocks.push(SemanticBlock { role: current_role, content: current_content.trim().to_string() });
        }
        
        if blocks.is_empty() {
            blocks.push(SemanticBlock { role: "Transcript".to_string(), content: content.to_string() });
        }
    } else {
        let role = metadata.get("role").and_then(|v| v.as_str()).unwrap_or("Message");
        let mut role_capitalized = role.to_string();
        if let Some(r) = role_capitalized.get_mut(0..1) {
            r.make_ascii_uppercase();
        }
        blocks.push(SemanticBlock { role: role_capitalized, content: content.to_string() });
    }
    blocks
}

fn truncate_at_word_boundary(text: &str, max_chars: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= max_chars {
        return text.to_string();
    }
    let byte_idx = text
        .char_indices()
        .nth(max_chars)
        .map(|(idx, _)| idx)
        .unwrap_or(text.len());
    let truncated = &text[..byte_idx];
    if let Some(last_space) = truncated.rfind(' ') {
        format!("{}...", &text[..last_space])
    } else {
        format!("{}...", truncated)
    }
}

pub fn create_structured_outer(blocks: &[SemanticBlock], keywords: &[String], target_chars: usize) -> String {
    let keyword_prefix = if !keywords.is_empty() {
        format!("[{}] ", keywords.iter().take(5).cloned().collect::<Vec<_>>().join(", "))
    } else {
        String::new()
    };
    
    let mut parts = Vec::new();
    for block in blocks {
        let first_sentence = block.content.split(['.', '!', '?', '\n']).next().unwrap_or(&block.content).trim();
        parts.push(format!("{}: {}", block.role, truncate_at_word_boundary(first_sentence, 60)));
    }
    
    let mut summary = parts.join(" | ");
    if summary.len() + keyword_prefix.len() > target_chars {
        summary = truncate_at_word_boundary(&summary, target_chars.saturating_sub(keyword_prefix.len()));
    }
    
    format!("{}{}", keyword_prefix, summary)
}

pub fn create_structured_middle(blocks: &[SemanticBlock], target_chars: usize) -> String {
    let mut parts = Vec::new();
    let chars_per_block = (target_chars / blocks.len().max(1)).max(50);
    
    for block in blocks {
        let snippet = truncate_at_word_boundary(&block.content, chars_per_block);
        parts.push(format!("{}:\n{}", block.role, snippet));
    }
    
    let result = parts.join("\n\n");
    truncate_at_word_boundary(&result, target_chars)
}

pub fn create_structured_inner(blocks: &[SemanticBlock], target_chars: usize) -> String {
    let mut parts = Vec::new();
    let chars_per_block = (target_chars / blocks.len().max(1)).max(100);
    
    for block in blocks {
        let snippet = truncate_at_word_boundary(&block.content, chars_per_block);
        parts.push(format!("{}:\n{}", block.role, snippet));
    }
    
    let result = parts.join("\n\n");
    truncate_at_word_boundary(&result, target_chars)
}

pub fn create_structured_onion_slices(
    content: &str,
    metadata: &serde_json::Value,
    config: &OnionSliceConfig,
) -> Vec<OnionSlice> {
    let content = content.trim();
    let blocks = parse_blocks(content, metadata);

    if content.len() < config.min_content_for_slicing {
        return create_structured_outer_core_slices(content, &blocks, config);
    }

    let core_id = OnionSlice::generate_id(content, SliceLayer::Core);
    let core_keywords = extract_keywords(content, 10);

    let inner_content = create_structured_inner(&blocks, config.inner_target);
    let inner_id = OnionSlice::generate_id(&inner_content, SliceLayer::Inner);
    let inner_keywords = extract_keywords(&inner_content, 7);

    let middle_content = create_structured_middle(&blocks, config.middle_target);
    let middle_id = OnionSlice::generate_id(&middle_content, SliceLayer::Middle);
    let middle_keywords = extract_keywords(&middle_content, 5);

    let outer_content = create_structured_outer(&blocks, &core_keywords, config.outer_target);
    let outer_id = OnionSlice::generate_id(&outer_content, SliceLayer::Outer);
    let outer_keywords = extract_keywords(&outer_content, 3);

    vec![
        OnionSlice {
            id: outer_id.clone(),
            layer: SliceLayer::Outer,
            content: outer_content,
            parent_id: Some(middle_id.clone()),
            children_ids: vec![],
            keywords: outer_keywords,
        },
        OnionSlice {
            id: middle_id.clone(),
            layer: SliceLayer::Middle,
            content: middle_content,
            parent_id: Some(inner_id.clone()),
            children_ids: vec![outer_id],
            keywords: middle_keywords,
        },
        OnionSlice {
            id: inner_id.clone(),
            layer: SliceLayer::Inner,
            content: inner_content,
            parent_id: Some(core_id.clone()),
            children_ids: vec![middle_id],
            keywords: inner_keywords,
        },
        OnionSlice {
            id: core_id.clone(),
            layer: SliceLayer::Core,
            content: content.to_string(),
            parent_id: None,
            children_ids: vec![inner_id],
            keywords: core_keywords,
        },
    ]
}

pub fn create_structured_onion_slices_fast(
    content: &str,
    metadata: &serde_json::Value,
    config: &OnionSliceConfig,
) -> Vec<OnionSlice> {
    let content = content.trim();
    let blocks = parse_blocks(content, metadata);

    if content.len() < config.min_content_for_slicing {
        return create_structured_outer_core_slices(content, &blocks, config);
    }

    let core_id = OnionSlice::generate_id(content, SliceLayer::Core);
    let core_keywords = extract_keywords(content, 10);

    let outer_content = create_structured_outer(&blocks, &core_keywords, config.outer_target);
    let outer_id = OnionSlice::generate_id(&outer_content, SliceLayer::Outer);
    let outer_keywords = extract_keywords(&outer_content, 3);

    vec![
        OnionSlice {
            id: outer_id.clone(),
            layer: SliceLayer::Outer,
            content: outer_content,
            parent_id: Some(core_id.clone()),
            children_ids: vec![],
            keywords: outer_keywords,
        },
        OnionSlice {
            id: core_id,
            layer: SliceLayer::Core,
            content: content.to_string(),
            parent_id: None,
            children_ids: vec![outer_id],
            keywords: core_keywords,
        },
    ]
}

fn create_structured_outer_core_slices(content: &str, blocks: &[SemanticBlock], config: &OnionSliceConfig) -> Vec<OnionSlice> {
    let core_id = OnionSlice::generate_id(content, SliceLayer::Core);
    let core_keywords = extract_keywords(content, 10);
    
    let outer_content = create_structured_outer(blocks, &core_keywords, config.outer_target);
    let outer_id = OnionSlice::generate_id(&outer_content, SliceLayer::Outer);
    let outer_keywords = extract_keywords(&outer_content, 3);

    vec![
        OnionSlice {
            id: outer_id.clone(),
            layer: SliceLayer::Outer,
            content: outer_content,
            parent_id: Some(core_id.clone()),
            children_ids: vec![],
            keywords: outer_keywords,
        },
        OnionSlice {
            id: core_id,
            layer: SliceLayer::Core,
            content: content.to_string(),
            parent_id: None,
            children_ids: vec![outer_id],
            keywords: core_keywords,
        },
    ]
}
