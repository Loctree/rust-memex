use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChunkQuality {
    pub avg_chunk_length: usize,
    pub sentence_integrity: f32,
    pub word_integrity: f32,
    pub chunk_quality: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum QualityTier {
    Empty,
    Purge,
    Warn,
    Good,
    Excellent,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AuditRecommendation {
    Empty,
    Purge,
    Warn,
    Good,
    Excellent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AuditResult {
    pub namespace: String,
    pub document_count: usize,
    pub avg_chunk_length: usize,
    pub sentence_integrity: f32,
    pub word_integrity: f32,
    pub chunk_quality: f32,
    pub overall_score: f32,
    pub recommendation: AuditRecommendation,
    pub passes_threshold: bool,
}

impl AuditResult {
    pub fn chunk_quality_metrics(&self) -> ChunkQuality {
        ChunkQuality {
            avg_chunk_length: self.avg_chunk_length,
            sentence_integrity: self.sentence_integrity,
            word_integrity: self.word_integrity,
            chunk_quality: self.chunk_quality,
        }
    }

    pub fn quality_tier(&self) -> QualityTier {
        match self.recommendation {
            AuditRecommendation::Empty => QualityTier::Empty,
            AuditRecommendation::Purge => QualityTier::Purge,
            AuditRecommendation::Warn => QualityTier::Warn,
            AuditRecommendation::Good => QualityTier::Good,
            AuditRecommendation::Excellent => QualityTier::Excellent,
        }
    }
}

impl AuditRecommendation {
    pub const fn as_str(self) -> &'static str {
        match self {
            AuditRecommendation::Empty => "EMPTY",
            AuditRecommendation::Purge => "PURGE",
            AuditRecommendation::Warn => "WARN",
            AuditRecommendation::Good => "GOOD",
            AuditRecommendation::Excellent => "EXCELLENT",
        }
    }
}

impl fmt::Display for AuditRecommendation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl QualityTier {
    pub const fn as_str(self) -> &'static str {
        match self {
            QualityTier::Empty => "EMPTY",
            QualityTier::Purge => "PURGE",
            QualityTier::Warn => "WARN",
            QualityTier::Good => "GOOD",
            QualityTier::Excellent => "EXCELLENT",
        }
    }
}

impl fmt::Display for QualityTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
