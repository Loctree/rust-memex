use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use axum::{Json, extract::State, http::StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::rag::{SearchOptions, SliceLayer, compute_content_hash};
use crate::storage::ChromaDocument;

use super::{HttpState, SearchResultJson, http_search_mode, search_results_with_mode};

const DEFAULT_CONTEXT_LIMIT: usize = 12;
const DEFAULT_MAX_EVIDENCE_PER_CLUSTER: usize = 5;
const DEFAULT_MAX_SOURCE_CHUNKS: usize = 160;

#[derive(Debug, Clone, Serialize)]
pub struct SearchClusterJson {
    pub cluster_id: String,
    pub group_by: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_range: Option<String>,
    pub representative: SearchResultJson,
    pub evidence: Vec<SearchResultJson>,
    pub hidden_duplicate_count: usize,
    pub hidden_duplicate_ids: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ContextPackRequest {
    #[serde(default)]
    pub query: Option<String>,
    #[serde(default)]
    pub namespace: Option<String>,
    #[serde(default)]
    pub ids: Vec<String>,
    #[serde(default = "default_context_limit")]
    pub limit: usize,
    #[serde(default)]
    pub layer: Option<u8>,
    #[serde(default)]
    pub deep: bool,
    #[serde(default)]
    pub project: Option<String>,
    #[serde(default = "default_mode")]
    pub mode: String,
    #[serde(default = "default_view")]
    pub view: String,
    #[serde(default = "default_true")]
    pub show_raw_evidence: bool,
    #[serde(default)]
    pub show_decisions_only: bool,
    #[serde(default = "default_max_evidence_per_cluster")]
    pub max_evidence_per_cluster: usize,
    #[serde(default = "default_max_source_chunks")]
    pub max_source_chunks: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContextPackResponse {
    pub query: Option<String>,
    pub namespace: String,
    pub selected_ids: Vec<String>,
    pub clusters: Vec<SearchClusterJson>,
    pub duplicate_count: usize,
    pub sources: Vec<ContextPackSourceJson>,
    pub markdown: String,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContextPackSourceJson {
    pub cluster_id: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    pub indexed_chunk_count: usize,
    pub used_core_document: bool,
    pub markdown: String,
}

#[derive(Clone)]
struct ClusterIdentity {
    group_by: &'static str,
    key: String,
    source_hash: Option<String>,
    session_id: Option<String>,
    source_path: Option<String>,
    turn_range: Option<String>,
}

fn default_context_limit() -> usize {
    DEFAULT_CONTEXT_LIMIT
}

fn default_mode() -> String {
    "hybrid".to_string()
}

fn default_view() -> String {
    "full".to_string()
}

fn default_true() -> bool {
    true
}

fn default_max_evidence_per_cluster() -> usize {
    DEFAULT_MAX_EVIDENCE_PER_CLUSTER
}

fn default_max_source_chunks() -> usize {
    DEFAULT_MAX_SOURCE_CHUNKS
}

pub fn collapse_results(results: &[SearchResultJson]) -> Vec<SearchClusterJson> {
    collapse_results_with_limit(results, DEFAULT_MAX_EVIDENCE_PER_CLUSTER)
}

pub fn collapse_results_with_limit(
    results: &[SearchResultJson],
    max_evidence_per_cluster: usize,
) -> Vec<SearchClusterJson> {
    let mut order: Vec<String> = Vec::new();
    let mut grouped: BTreeMap<String, Vec<SearchResultJson>> = BTreeMap::new();
    let mut identities: HashMap<String, ClusterIdentity> = HashMap::new();
    let max_evidence_per_cluster = max_evidence_per_cluster.max(1);

    for result in results {
        let identity = cluster_identity(result);
        if !grouped.contains_key(&identity.key) {
            order.push(identity.key.clone());
            identities.insert(identity.key.clone(), identity.clone());
        }
        grouped
            .entry(identity.key)
            .or_default()
            .push(result.clone());
    }

    order
        .into_iter()
        .filter_map(|key| {
            let mut evidence = grouped.remove(&key)?;
            evidence.sort_by(|left, right| {
                right
                    .score
                    .partial_cmp(&left.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let representative = evidence.first()?.clone();
            let identity = identities.remove(&key)?;
            let hidden_duplicate_ids = evidence
                .iter()
                .skip(1)
                .map(|result| result.id.clone())
                .collect::<Vec<_>>();
            let hidden_duplicate_count = evidence.len().saturating_sub(1);
            evidence.truncate(max_evidence_per_cluster);

            Some(SearchClusterJson {
                cluster_id: identity.key,
                group_by: identity.group_by.to_string(),
                source_hash: identity.source_hash,
                session_id: identity.session_id,
                source_path: identity.source_path,
                turn_range: identity.turn_range,
                representative,
                evidence,
                hidden_duplicate_count,
                hidden_duplicate_ids,
            })
        })
        .collect()
}

pub async fn context_pack_handler(
    State(state): State<HttpState>,
    Json(req): Json<ContextPackRequest>,
) -> Result<Json<ContextPackResponse>, (StatusCode, String)> {
    let start = Instant::now();
    let namespace = req
        .namespace
        .clone()
        .unwrap_or_else(|| "default".to_string());

    let selected = if req.ids.is_empty() {
        let Some(query) = req.query.as_deref() else {
            return Err((
                StatusCode::BAD_REQUEST,
                "Provide either query or ids for context-pack".to_string(),
            ));
        };
        let layer_filter = if req.deep {
            None
        } else {
            req.layer
                .and_then(SliceLayer::from_u8)
                .or(Some(SliceLayer::Outer))
        };
        let options = SearchOptions {
            layer_filter,
            project_filter: req.project.clone().filter(|value| !value.trim().is_empty()),
        };
        search_results_with_mode(
            &state,
            Some(namespace.as_str()),
            query,
            req.limit,
            http_search_mode(req.mode.as_str()),
            options,
        )
        .await
        .map_err(internal_error)?
    } else {
        let mut fetched = Vec::with_capacity(req.ids.len());
        for id in &req.ids {
            if let Some(result) = state
                .rag
                .lookup_memory(namespace.as_str(), id)
                .await
                .map_err(internal_error)?
            {
                fetched.push(SearchResultJson::from(result));
            }
        }
        fetched
    };

    let selected_ids = selected
        .iter()
        .map(|result| result.id.clone())
        .collect::<Vec<_>>();
    let clusters = collapse_results_with_limit(&selected, req.max_evidence_per_cluster);
    let duplicate_count = selected.len().saturating_sub(clusters.len());
    let view = if req.show_decisions_only {
        "decisions"
    } else {
        req.view.as_str()
    };
    let all_docs = state
        .rag
        .storage_manager()
        .all_documents(Some(namespace.as_str()), 100_000)
        .await
        .map_err(internal_error)?;
    let sources = rebuild_sources(&clusters, &all_docs, req.max_source_chunks.max(1), view);
    let markdown = render_context_pack_markdown(
        req.query.as_deref(),
        namespace.as_str(),
        &clusters,
        &sources,
        req.show_raw_evidence,
        req.max_evidence_per_cluster.max(1),
    );

    Ok(Json(ContextPackResponse {
        query: req.query,
        namespace,
        selected_ids,
        clusters,
        duplicate_count,
        sources,
        markdown,
        elapsed_ms: start.elapsed().as_millis() as u64,
    }))
}

fn internal_error(error: anyhow::Error) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, error.to_string())
}

fn cluster_identity(result: &SearchResultJson) -> ClusterIdentity {
    let source_hash = metadata_text(&result.metadata, &["source_hash", "file_hash"]);
    let source_path = metadata_text(&result.metadata, &["source_path", "path"]);
    let exact_content_hash = metadata_text(&result.metadata, &["chunk_hash", "content_hash"]);
    let session_id = metadata_text(
        &result.metadata,
        &["session_id", "conversation_id", "run_id"],
    )
    .or_else(|| source_path.as_deref().and_then(session_id_from_path));
    let turn_range = metadata_text(
        &result.metadata,
        &["turn_range", "line_range", "chunk_range", "message_range"],
    )
    .or_else(|| numeric_range(&result.metadata, "turn_start", "turn_end"))
    .or_else(|| numeric_range(&result.metadata, "line_start", "line_end"));

    let (group_by, raw_key) = if let Some(value) = source_hash.as_deref() {
        ("source_hash", value.to_string())
    } else if let Some(value) = session_id.as_deref() {
        ("session_id", value.to_string())
    } else if let Some(value) = source_path.as_deref() {
        ("source_path", value.to_string())
    } else if let Some(value) = result.parent_id.as_deref() {
        ("turn_range", value.to_string())
    } else if let Some(value) = turn_range.as_deref() {
        ("turn_range", value.to_string())
    } else if let Some(value) = exact_content_hash.as_deref() {
        ("content_hash", value.to_string())
    } else {
        ("normalized_text", normalized_text_hash(&result.text))
    };

    ClusterIdentity {
        group_by,
        key: format!("{}:{}", group_by, raw_key),
        source_hash,
        session_id,
        source_path,
        turn_range,
    }
}

fn rebuild_sources(
    clusters: &[SearchClusterJson],
    all_docs: &[ChromaDocument],
    max_source_chunks: usize,
    view: &str,
) -> Vec<ContextPackSourceJson> {
    clusters
        .iter()
        .map(|cluster| {
            let mut matching = all_docs
                .iter()
                .filter(|doc| source_matches_cluster(doc, cluster))
                .cloned()
                .collect::<Vec<_>>();

            matching.sort_by_key(source_sort_key);

            let used_core_document = matching.iter().any(|doc| doc.layer == 4);
            if used_core_document {
                matching.retain(|doc| doc.layer == 4);
            }
            matching.truncate(max_source_chunks);

            let markdown = if matching.is_empty() {
                render_selected_evidence(cluster, DEFAULT_MAX_EVIDENCE_PER_CLUSTER)
            } else {
                render_rebuilt_source(&matching, view)
            };

            ContextPackSourceJson {
                cluster_id: cluster.cluster_id.clone(),
                status: if matching.is_empty() {
                    "source_missing_selected_evidence_only".to_string()
                } else if used_core_document {
                    "rebuilt_from_core_chunk".to_string()
                } else {
                    "rebuilt_from_index_chunks".to_string()
                },
                source_path: cluster.source_path.clone(),
                source_hash: cluster.source_hash.clone(),
                session_id: cluster.session_id.clone(),
                indexed_chunk_count: matching.len(),
                used_core_document,
                markdown,
            }
        })
        .collect()
}

fn source_matches_cluster(doc: &ChromaDocument, cluster: &SearchClusterJson) -> bool {
    if let Some(source_hash) = cluster.source_hash.as_deref() {
        return doc.source_hash.as_deref() == Some(source_hash)
            || metadata_text(&doc.metadata, &["source_hash", "file_hash"]).as_deref()
                == Some(source_hash);
    }

    if let Some(source_path) = cluster.source_path.as_deref() {
        return metadata_text(&doc.metadata, &["source_path", "path"]).as_deref()
            == Some(source_path);
    }

    if let Some(session_id) = cluster.session_id.as_deref() {
        return metadata_text(&doc.metadata, &["session_id", "conversation_id", "run_id"])
            .or_else(|| {
                metadata_text(&doc.metadata, &["source_path", "path"])
                    .as_deref()
                    .and_then(session_id_from_path)
            })
            .as_deref()
            == Some(session_id);
    }

    false
}

fn source_sort_key(doc: &ChromaDocument) -> (u8, String) {
    let layer_rank = match doc.layer {
        4 => 0,
        3 => 1,
        2 => 2,
        1 => 3,
        _ => 4,
    };
    (layer_rank, doc.id.clone())
}

fn render_context_pack_markdown(
    query: Option<&str>,
    namespace: &str,
    clusters: &[SearchClusterJson],
    sources: &[ContextPackSourceJson],
    show_raw_evidence: bool,
    max_evidence_per_cluster: usize,
) -> String {
    let mut out = String::new();
    out.push_str("# rust-memex Context Pack\n\n");
    out.push_str(&format!("- Namespace: `{}`\n", namespace));
    if let Some(query) = query {
        out.push_str(&format!("- Query: `{}`\n", query));
    }
    out.push_str(&format!("- Collapsed clusters: {}\n", clusters.len()));
    out.push_str(&format!(
        "- Hidden duplicate hits: {}\n\n",
        clusters
            .iter()
            .map(|cluster| cluster.hidden_duplicate_count)
            .sum::<usize>()
    ));

    for (idx, cluster) in clusters.iter().enumerate() {
        out.push_str(&format!("## {}. {}\n\n", idx + 1, cluster_title(cluster)));
        out.push_str(&format!("- Grouped by: `{}`\n", cluster.group_by));
        if let Some(path) = cluster.source_path.as_deref() {
            out.push_str(&format!("- Source path: `{}`\n", path));
        }
        if let Some(session_id) = cluster.session_id.as_deref() {
            out.push_str(&format!("- Session: `{}`\n", session_id));
        }
        if let Some(turn_range) = cluster.turn_range.as_deref() {
            out.push_str(&format!("- Turn range: `{}`\n", turn_range));
        }
        out.push_str(&format!(
            "- Hidden duplicates: {}\n\n",
            cluster.hidden_duplicate_count
        ));

        if show_raw_evidence {
            out.push_str("### Raw Evidence\n\n");
            out.push_str(&render_selected_evidence(cluster, max_evidence_per_cluster));
            out.push('\n');
        }

        if let Some(source) = sources
            .iter()
            .find(|source| source.cluster_id == cluster.cluster_id)
        {
            out.push_str("### Rebuilt Context\n\n");
            out.push_str(&format!("Status: `{}`\n\n", source.status));
            out.push_str(&source.markdown);
            out.push('\n');
        }
    }

    out
}

fn render_selected_evidence(cluster: &SearchClusterJson, limit: usize) -> String {
    let mut out = String::new();
    for evidence in cluster.evidence.iter().take(limit) {
        out.push_str(&format!(
            "- `{}` score={:.3} layer={}\n\n{}\n\n",
            evidence.id,
            evidence.score,
            evidence.layer.as_deref().unwrap_or("flat"),
            evidence.text.trim()
        ));
    }
    out
}

fn render_rebuilt_source(docs: &[ChromaDocument], view: &str) -> String {
    let decisions_only = view.eq_ignore_ascii_case("decisions");
    let mut out = String::new();
    for doc in docs {
        let text = if decisions_only {
            decision_lines(&doc.document)
        } else {
            doc.document.clone()
        };
        if text.trim().is_empty() {
            continue;
        }
        out.push_str(&format!(
            "#### `{}` layer={}\n\n{}\n\n",
            doc.id,
            doc.slice_layer()
                .map(|layer| layer.name().to_string())
                .unwrap_or_else(|| "flat".to_string()),
            text.trim()
        ));
    }
    if out.is_empty() && decisions_only {
        "No decision-like lines found in rebuilt chunks.\n".to_string()
    } else {
        out
    }
}

fn decision_lines(text: &str) -> String {
    text.lines()
        .filter(|line| {
            let lowered = line.to_ascii_lowercase();
            lowered.contains("decision")
                || lowered.contains("decided")
                || lowered.contains("wybór")
                || lowered.contains("ustal")
                || lowered.contains("wniosek")
                || lowered.contains("verdict")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn cluster_title(cluster: &SearchClusterJson) -> String {
    cluster
        .source_path
        .as_deref()
        .and_then(|path| std::path::Path::new(path).file_name())
        .and_then(|name| name.to_str())
        .map(ToOwned::to_owned)
        .or_else(|| cluster.session_id.clone())
        .unwrap_or_else(|| cluster.representative.id.clone())
}

fn metadata_text(metadata: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .filter_map(|key| metadata.get(*key))
        .filter_map(|value| value.as_str())
        .map(str::trim)
        .find(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn numeric_range(metadata: &Value, start: &str, end: &str) -> Option<String> {
    let start = metadata.get(start).and_then(Value::as_i64)?;
    let end = metadata.get(end).and_then(Value::as_i64)?;
    Some(format!("{}..{}", start, end))
}

fn session_id_from_path(path: &str) -> Option<String> {
    let file_name = std::path::Path::new(path).file_name()?.to_str()?;
    file_name
        .split("__")
        .find(|part| looks_like_session_id(part))
        .map(ToOwned::to_owned)
        .or_else(|| {
            file_name
                .split(|ch: char| !(ch.is_ascii_hexdigit() || ch == '-'))
                .find(|part| looks_like_session_id(part))
                .map(ToOwned::to_owned)
        })
}

fn looks_like_session_id(value: &str) -> bool {
    let hexish = value
        .chars()
        .filter(|ch| ch.is_ascii_hexdigit() || *ch == '-')
        .count();
    value.len() >= 20 && hexish == value.len() && value.contains('-')
}

fn normalized_text_hash(text: &str) -> String {
    let normalized = text
        .split_whitespace()
        .map(|word| {
            word.chars()
                .filter(|ch| ch.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect::<String>()
        })
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    compute_content_hash(&normalized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn result(id: &str, path: &str, score: f32, text: &str) -> SearchResultJson {
        SearchResultJson {
            id: id.to_string(),
            namespace: "kb:transcripts".to_string(),
            text: text.to_string(),
            score,
            metadata: json!({"path": path}),
            layer: Some("outer".to_string()),
            parent_id: None,
            children_ids: vec![],
            keywords: vec![],
            can_expand: false,
            can_drill_up: false,
        }
    }

    fn result_with_metadata(
        id: &str,
        score: f32,
        text: &str,
        metadata: serde_json::Value,
    ) -> SearchResultJson {
        SearchResultJson {
            id: id.to_string(),
            namespace: "kb:transcripts".to_string(),
            text: text.to_string(),
            score,
            metadata,
            layer: Some("outer".to_string()),
            parent_id: None,
            children_ids: vec![],
            keywords: vec![],
            can_expand: false,
            can_drill_up: false,
        }
    }

    #[test]
    fn collapse_groups_by_source_path_and_preserves_best_evidence() {
        let results = vec![
            result("a", "/tmp/session.md", 0.7, "first hit"),
            result("b", "/tmp/session.md", 0.9, "better hit"),
            result("c", "/tmp/other.md", 0.4, "other"),
        ];

        let clusters = collapse_results(&results);

        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[0].group_by, "source_path");
        assert_eq!(clusters[0].representative.id, "b");
        assert_eq!(clusters[0].hidden_duplicate_count, 1);
        assert_eq!(clusters[0].hidden_duplicate_ids, vec!["a"]);
    }

    #[test]
    fn collapse_prefers_session_over_chunk_paths_and_keeps_limited_evidence() {
        let results = vec![
            result_with_metadata(
                "a",
                0.8,
                "turn one",
                json!({"session_id": "019d749e-5b30-7f33-8bb4-a3a6e21b66c4", "path": "/tmp/chunk-1.md"}),
            ),
            result_with_metadata(
                "b",
                0.9,
                "turn two",
                json!({"session_id": "019d749e-5b30-7f33-8bb4-a3a6e21b66c4", "path": "/tmp/chunk-2.md"}),
            ),
            result_with_metadata(
                "c",
                0.7,
                "turn three",
                json!({"session_id": "019d749e-5b30-7f33-8bb4-a3a6e21b66c4", "path": "/tmp/chunk-3.md"}),
            ),
        ];

        let clusters = collapse_results_with_limit(&results, 2);

        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].group_by, "session_id");
        assert_eq!(clusters[0].representative.id, "b");
        assert_eq!(clusters[0].evidence.len(), 2);
        assert_eq!(clusters[0].hidden_duplicate_count, 2);
        assert_eq!(clusters[0].hidden_duplicate_ids, vec!["a", "c"]);
    }

    #[test]
    fn collapse_falls_back_to_content_hash_then_normalized_text() {
        let hashed = vec![
            result_with_metadata(
                "a",
                0.5,
                "same logical chunk",
                json!({"chunk_hash": "chunk-a"}),
            ),
            result_with_metadata(
                "b",
                0.6,
                "same logical chunk copy",
                json!({"chunk_hash": "chunk-a"}),
            ),
        ];
        let hashed_clusters = collapse_results(&hashed);

        assert_eq!(hashed_clusters.len(), 1);
        assert_eq!(hashed_clusters[0].group_by, "content_hash");
        assert_eq!(hashed_clusters[0].representative.id, "b");

        let normalized = vec![
            result_with_metadata("c", 0.4, "Auth, license: Vista!", json!({})),
            result_with_metadata("d", 0.7, "auth license vista", json!({})),
        ];
        let normalized_clusters = collapse_results(&normalized);

        assert_eq!(normalized_clusters.len(), 1);
        assert_eq!(normalized_clusters[0].group_by, "normalized_text");
    }

    #[test]
    fn rebuilt_source_prefers_core_chunk_for_full_markdown() {
        let cluster = collapse_results(&[result(
            "outer",
            "/tmp/codex__019d749e-5b30-7f33-8bb4-a3a6e21b66c4__clean.md",
            0.8,
            "outer",
        )])
        .pop()
        .unwrap();
        let docs = vec![
            ChromaDocument::new_flat(
                "outer".to_string(),
                "kb:transcripts".to_string(),
                vec![],
                json!({"path": "/tmp/codex__019d749e-5b30-7f33-8bb4-a3a6e21b66c4__clean.md"}),
                "outer summary".to_string(),
            ),
            ChromaDocument {
                id: "core".to_string(),
                namespace: "kb:transcripts".to_string(),
                embedding: vec![],
                metadata: json!({"path": "/tmp/codex__019d749e-5b30-7f33-8bb4-a3a6e21b66c4__clean.md"}),
                document: "# Full transcript\n\nDecision: keep chunks as index units.".to_string(),
                layer: 4,
                parent_id: None,
                children_ids: vec![],
                keywords: vec![],
                content_hash: None,
                source_hash: None,
            },
        ];

        let sources = rebuild_sources(&[cluster], &docs, 20, "full");

        assert_eq!(sources[0].status, "rebuilt_from_core_chunk");
        assert!(sources[0].markdown.contains("# Full transcript"));
        assert!(!sources[0].markdown.contains("outer summary"));
    }
}
