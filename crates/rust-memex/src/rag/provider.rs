use anyhow::{Result, anyhow};
use chrono::{TimeZone, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::future::Future;
use std::path::Path;
use std::pin::Pin;

use crate::rag::{
    OnionSlice, OnionSliceConfig, OuterSynthesis, SliceMode, create_onion_slices_async,
    create_onion_slices_fast_async,
};
use crate::rag::{compute_content_hash, pipeline::Chunk, pipeline::FileContent};

pub type ChunkProviderFuture<'a> = Pin<Box<dyn Future<Output = Result<Vec<Chunk>>> + Send + 'a>>;

/// Options shared by all chunk providers.
#[derive(Debug, Clone)]
pub struct ChunkOpts {
    pub chunker: ChunkerKind,
    pub slice_mode: SliceMode,
    pub outer_synthesis: OuterSynthesis,
    pub flat_window: usize,
    pub flat_overlap: usize,
}

impl ChunkOpts {
    pub fn new(
        chunker: ChunkerKind,
        slice_mode: SliceMode,
        outer_synthesis: OuterSynthesis,
    ) -> Self {
        Self {
            chunker,
            slice_mode,
            outer_synthesis,
            flat_window: 512,
            flat_overlap: 128,
        }
    }
}

/// Trait abstracting chunk production strategies.
pub trait ChunkProvider: Send + Sync {
    fn name(&self) -> &'static str;
    fn chunk<'a>(&'a self, doc: &'a FileContent, opts: &'a ChunkOpts) -> ChunkProviderFuture<'a>;
}

/// Chunker strategy exposed through CLI/API routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[serde(rename_all = "kebab-case")]
pub enum ChunkerKind {
    Aicx,
    Onion,
    Flat,
}

impl ChunkerKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Aicx => "aicx",
            Self::Onion => "onion",
            Self::Flat => "flat",
        }
    }

    pub fn into_provider(self) -> Box<dyn ChunkProvider> {
        match self {
            Self::Aicx => Box::new(AicxChunkProvider::default()),
            Self::Onion => Box::new(OnionChunkProvider),
            Self::Flat => Box::new(FlatChunkProvider {
                window: 512,
                overlap: 128,
            }),
        }
    }

    pub fn slice_mode(self, requested: SliceMode) -> SliceMode {
        match self {
            Self::Aicx | Self::Flat => SliceMode::Flat,
            Self::Onion => match requested {
                SliceMode::OnionFast => SliceMode::OnionFast,
                _ => SliceMode::Onion,
            },
        }
    }
}

impl std::str::FromStr for ChunkerKind {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "aicx" => Ok(Self::Aicx),
            "onion" => Ok(Self::Onion),
            "flat" => Ok(Self::Flat),
            other => Err(format!(
                "Invalid chunker: '{}'. Use 'aicx', 'onion', or 'flat'",
                other
            )),
        }
    }
}

impl std::fmt::Display for ChunkerKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Existing onion-slicer logic. Deprecated for transcript-shaped content.
pub struct OnionChunkProvider;

/// Frame-aware transcript chunker via `aicx-parser`.
#[derive(Default)]
pub struct AicxChunkProvider {
    config: aicx_parser::ChunkerConfig,
}

/// Simple fixed-size sliding window with configurable overlap.
pub struct FlatChunkProvider {
    window: usize,
    overlap: usize,
}

impl ChunkProvider for OnionChunkProvider {
    fn name(&self) -> &'static str {
        "onion"
    }

    fn chunk<'a>(&'a self, doc: &'a FileContent, opts: &'a ChunkOpts) -> ChunkProviderFuture<'a> {
        Box::pin(async move {
            let metadata = base_metadata(doc, opts.chunker, opts.slice_mode);
            let config = OnionSliceConfig {
                outer_synthesis: opts.outer_synthesis.clone(),
                ..OnionSliceConfig::default()
            };
            let slices = match opts.slice_mode {
                SliceMode::OnionFast => {
                    create_onion_slices_fast_async(&doc.text, &metadata, &config).await
                }
                SliceMode::Onion | SliceMode::Flat => {
                    create_onion_slices_async(&doc.text, &metadata, &config).await
                }
            };

            Ok(slices_to_chunks(slices, doc, metadata))
        })
    }
}

impl ChunkProvider for AicxChunkProvider {
    fn name(&self) -> &'static str {
        "aicx"
    }

    fn chunk<'a>(&'a self, doc: &'a FileContent, opts: &'a ChunkOpts) -> ChunkProviderFuture<'a> {
        Box::pin(async move {
            let entries = doc_to_timeline_entries(doc)?;
            let project = project_label(&doc.namespace);
            let agent = first_agent(&entries).unwrap_or("rust-memex");
            let aicx_chunks =
                aicx_parser::chunker::chunk_entries(&entries, &project, agent, &self.config);
            Ok(aicx_chunks
                .into_iter()
                .map(|chunk| memex_chunk_from_aicx(chunk, doc, opts))
                .collect())
        })
    }
}

impl ChunkProvider for FlatChunkProvider {
    fn name(&self) -> &'static str {
        "flat"
    }

    fn chunk<'a>(&'a self, doc: &'a FileContent, opts: &'a ChunkOpts) -> ChunkProviderFuture<'a> {
        Box::pin(async move {
            let window = opts.flat_window.max(1).max(self.window);
            let overlap = opts
                .flat_overlap
                .min(window.saturating_sub(1))
                .max(self.overlap.min(window.saturating_sub(1)));
            let metadata = base_metadata(doc, opts.chunker, SliceMode::Flat);
            Ok(create_flat_chunks(
                &doc.text, doc, metadata, window, overlap,
            ))
        })
    }
}

pub fn detect_default_chunker(source_path: &Path, namespace: &str) -> ChunkerKind {
    let path_str = source_path.to_string_lossy();
    if namespace.starts_with("kb:transcripts")
        || namespace.starts_with("aicx")
        || namespace.starts_with("klaudiusz-")
        || path_str.contains("__clean.md")
        || path_str.contains("/.aicx/store/")
        || path_str.contains("/transcripts/")
    {
        ChunkerKind::Aicx
    } else {
        ChunkerKind::Onion
    }
}

fn base_metadata(doc: &FileContent, chunker: ChunkerKind, slice_mode: SliceMode) -> Value {
    let mut metadata = json!({
        "path": doc.path.to_str(),
        "content_hash": &doc.content_hash,
        "source_hash": &doc.content_hash,
        "chunker": chunker.name(),
        "slice_mode": match slice_mode {
            SliceMode::Onion => "onion",
            SliceMode::OnionFast => "onion-fast",
            SliceMode::Flat => "flat",
        },
    });

    if chunker == ChunkerKind::Aicx
        && let Value::Object(ref mut map) = metadata
    {
        map.insert("format".to_string(), json!("markdown_transcript"));
        map.insert("type".to_string(), json!("conversation"));
    }

    metadata
}

fn slices_to_chunks(
    slices: Vec<OnionSlice>,
    content: &FileContent,
    base_metadata: Value,
) -> Vec<Chunk> {
    slices
        .into_iter()
        .map(|slice| {
            let chunk_hash = compute_content_hash(&slice.content);
            let mut metadata = base_metadata.clone();
            if let Value::Object(ref mut map) = metadata {
                map.insert("chunk_hash".to_string(), json!(&chunk_hash));
                map.insert("layer".to_string(), json!(slice.layer.name()));
                map.insert("keywords".to_string(), json!(slice.keywords));
            }

            Chunk {
                id: slice.id,
                content: slice.content,
                source_path: content.path.clone(),
                namespace: content.namespace.clone(),
                chunk_hash,
                source_hash: content.content_hash.clone(),
                layer: slice.layer.as_u8(),
                parent_id: slice.parent_id,
                children_ids: slice.children_ids,
                keywords: slice.keywords,
                metadata,
            }
        })
        .collect()
}

fn create_flat_chunks(
    text: &str,
    content: &FileContent,
    base_metadata: Value,
    window: usize,
    overlap: usize,
) -> Vec<Chunk> {
    let chunks = split_into_chunks(text, window, overlap);
    let total_chunks = chunks.len();

    chunks
        .into_iter()
        .enumerate()
        .map(|(idx, chunk_text)| {
            let chunk_hash = compute_content_hash(&chunk_text);
            let mut metadata = base_metadata.clone();
            if let Value::Object(ref mut map) = metadata {
                map.insert("chunk_index".to_string(), json!(idx));
                map.insert("total_chunks".to_string(), json!(total_chunks));
                map.insert("chunk_hash".to_string(), json!(&chunk_hash));
            }

            let id = format!(
                "{}_{}_{}",
                content.path.to_str().unwrap_or("unknown"),
                content.content_hash.get(..8).unwrap_or(""),
                idx
            );

            Chunk {
                id,
                content: chunk_text,
                source_path: content.path.clone(),
                namespace: content.namespace.clone(),
                chunk_hash,
                source_hash: content.content_hash.clone(),
                layer: 0,
                parent_id: None,
                children_ids: vec![],
                keywords: vec![],
                metadata,
            }
        })
        .collect()
}

pub(crate) fn split_into_chunks(text: &str, target_size: usize, overlap: usize) -> Vec<String> {
    let mut char_offsets: Vec<usize> = text.char_indices().map(|(byte_idx, _)| byte_idx).collect();
    let len = char_offsets.len();

    if len <= target_size {
        return vec![text.to_string()];
    }

    char_offsets.push(text.len());

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < len {
        let end = (start + target_size).min(len);
        let start_byte = char_offsets[start];
        let end_byte = char_offsets[end];
        chunks.push(text[start_byte..end_byte].to_string());

        if end >= len {
            break;
        }

        start = end.saturating_sub(overlap);
    }

    chunks
}

fn doc_to_timeline_entries(doc: &FileContent) -> Result<Vec<aicx_parser::TimelineEntry>> {
    let timestamp = Utc
        .timestamp_opt(0, 0)
        .single()
        .ok_or_else(|| anyhow!("failed to construct epoch timestamp"))?;
    let session_id = compute_content_hash(&doc.text)
        .get(..12)
        .unwrap_or("session")
        .to_string();
    let cwd = doc
        .path
        .parent()
        .and_then(Path::to_str)
        .map(ToString::to_string);
    let preamble = extract_preamble(&doc.text);
    let mut entries = Vec::new();
    let mut current_role: Option<String> = None;
    let mut current_body = String::new();

    for line in doc.text.lines() {
        if let Some(role) = role_heading(line) {
            push_entry(
                &mut entries,
                current_role.take(),
                &mut current_body,
                timestamp,
                &session_id,
                cwd.clone(),
            );
            current_role = Some(role.to_string());
        } else {
            if !current_body.is_empty() {
                current_body.push('\n');
            }
            current_body.push_str(line);
        }
    }

    push_entry(
        &mut entries,
        current_role,
        &mut current_body,
        timestamp,
        &session_id,
        cwd,
    );

    if let Some(preamble) = preamble
        && let Some(first) = entries.first_mut()
        && !first.message.starts_with("---")
    {
        first.message = format!("{}\n\n{}", preamble.trim_end(), first.message.trim_start());
    }

    if entries.is_empty() {
        entries.push(aicx_parser::TimelineEntry {
            timestamp,
            agent: "rust-memex".to_string(),
            session_id,
            role: "assistant".to_string(),
            message: doc.text.clone(),
            frame_kind: Some(aicx_parser::FrameKind::AgentReply),
            branch: None,
            cwd: doc
                .path
                .parent()
                .and_then(Path::to_str)
                .map(ToString::to_string),
        });
    }

    Ok(entries)
}

fn extract_preamble(text: &str) -> Option<String> {
    let mut lines = Vec::new();
    for line in text.lines() {
        if role_heading(line).is_some() {
            break;
        }
        lines.push(line);
    }
    let preamble = lines.join("\n");
    (!preamble.trim().is_empty()).then_some(preamble)
}

fn push_entry(
    entries: &mut Vec<aicx_parser::TimelineEntry>,
    role: Option<String>,
    body: &mut String,
    timestamp: chrono::DateTime<Utc>,
    session_id: &str,
    cwd: Option<String>,
) {
    let Some(role) = role else {
        body.clear();
        return;
    };
    let message = body.trim().to_string();
    body.clear();
    if message.is_empty() {
        return;
    }
    let frame_kind = match role.as_str() {
        "user" => Some(aicx_parser::FrameKind::UserMsg),
        "assistant" => Some(aicx_parser::FrameKind::AgentReply),
        "tool" => Some(aicx_parser::FrameKind::ToolCall),
        _ => None,
    };
    entries.push(aicx_parser::TimelineEntry {
        timestamp,
        agent: "rust-memex".to_string(),
        session_id: session_id.to_string(),
        role,
        message,
        frame_kind,
        branch: None,
        cwd,
    });
}

fn role_heading(line: &str) -> Option<&'static str> {
    let lowered = line.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "## user" | "### user" | "[user]" | "user request:" => Some("user"),
        "## assistant" | "### assistant" | "[assistant]" | "assistant response:" => {
            Some("assistant")
        }
        "## tool" | "### tool" | "[tool]" | "tool:" | "tool result:" => Some("tool"),
        _ => None,
    }
}

fn first_agent(entries: &[aicx_parser::TimelineEntry]) -> Option<&str> {
    entries.first().map(|entry| entry.agent.as_str())
}

fn project_label(namespace: &str) -> String {
    namespace
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn memex_chunk_from_aicx(
    chunk: aicx_parser::Chunk,
    content: &FileContent,
    opts: &ChunkOpts,
) -> Chunk {
    let chunk_hash = compute_content_hash(&chunk.text);
    let id_prefix = content.content_hash.get(..8).unwrap_or("aicx");
    let mut metadata = base_metadata(content, opts.chunker, SliceMode::Flat);
    if let Value::Object(ref mut map) = metadata {
        map.insert("aicx_chunk_id".to_string(), json!(&chunk.id));
        map.insert("aicx_project".to_string(), json!(&chunk.project));
        map.insert("aicx_agent".to_string(), json!(&chunk.agent));
        map.insert("aicx_date".to_string(), json!(&chunk.date));
        map.insert("aicx_session_id".to_string(), json!(&chunk.session_id));
        map.insert("aicx_kind".to_string(), json!(chunk.kind));
        map.insert("aicx_frame_kind".to_string(), json!(chunk.frame_kind));
        map.insert("aicx_msg_start".to_string(), json!(chunk.msg_range.0));
        map.insert("aicx_msg_end".to_string(), json!(chunk.msg_range.1));
        map.insert("token_estimate".to_string(), json!(chunk.token_estimate));
        map.insert("highlights".to_string(), json!(&chunk.highlights));
        map.insert("chunk_hash".to_string(), json!(&chunk_hash));
        if let Some(cwd) = &chunk.cwd {
            map.insert("aicx_cwd".to_string(), json!(cwd));
        }
        if let Some(run_id) = &chunk.run_id {
            map.insert("run_id".to_string(), json!(run_id));
        }
        if let Some(prompt_id) = &chunk.prompt_id {
            map.insert("prompt_id".to_string(), json!(prompt_id));
        }
        if let Some(model) = &chunk.agent_model {
            map.insert("agent_model".to_string(), json!(model));
        }
    }

    Chunk {
        id: format!("{id_prefix}::{}", chunk.id),
        content: chunk.text,
        source_path: content.path.clone(),
        namespace: content.namespace.clone(),
        chunk_hash,
        source_hash: content.content_hash.clone(),
        layer: 0,
        parent_id: None,
        children_ids: vec![],
        keywords: chunk.highlights,
        metadata,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn doc(text: &str, namespace: &str, path: &str) -> FileContent {
        FileContent {
            path: PathBuf::from(path),
            text: text.to_string(),
            namespace: namespace.to_string(),
            content_hash: compute_content_hash(text),
        }
    }

    #[test]
    fn default_chunker_routes_transcripts_to_aicx() {
        assert_eq!(
            detect_default_chunker(Path::new("/tmp/transcripts/sample.md"), "kb:any"),
            ChunkerKind::Aicx
        );
        assert_eq!(
            detect_default_chunker(Path::new("/tmp/readme.md"), "kb:docs"),
            ChunkerKind::Onion
        );
    }

    #[tokio::test]
    async fn aicx_provider_chunks_markdown_transcript() {
        let doc = doc(
            "## user\nWhat is the meaning of life?\n\n## assistant\nBuild something useful.\n",
            "kb:transcripts-test",
            "/tmp/sample-transcript.md",
        );
        let opts = ChunkOpts::new(
            ChunkerKind::Aicx,
            SliceMode::Flat,
            OuterSynthesis::default(),
        );
        let chunks = AicxChunkProvider::default()
            .chunk(&doc, &opts)
            .await
            .unwrap();

        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].metadata["chunker"], "aicx");
        assert!(chunks[0].content.contains("meaning of life"));
    }

    #[test]
    fn split_into_chunks_preserves_overlap() {
        let chunks = split_into_chunks("abcdefghijklmnopqrstuvwxyz", 10, 3);

        assert!(chunks.len() > 1);
        assert_eq!(chunks[0].len(), 10);
        assert!(chunks[0].ends_with(&chunks[1][..3]));
    }
}
