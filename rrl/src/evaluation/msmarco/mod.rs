//! MS MARCO v1.1 Evaluation Module
//!
//! Provides evaluation functionality for MS MARCO v1.1 passage re-ranking task.
//! Computes standard IR metrics: MRR@10, NDCG@10, Recall@10, Recall@100.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::time::Instant;

use crate::evaluation::retrieval::{QueryResult, RetrievalEvaluator, RetrievalMetrics};

/// MS MARCO validation example format
#[derive(Debug, Clone, Deserialize)]
pub struct MsMarcoExample {
    /// Query text
    pub query: String,
    /// List of passages to re-rank
    pub passages: Vec<String>,
    /// Binary relevance labels (1 = relevant, 0 = not relevant)
    pub labels: Vec<u8>,
}

impl MsMarcoExample {
    /// Get indices of relevant passages
    pub fn relevant_indices(&self) -> Vec<usize> {
        self.labels
            .iter()
            .enumerate()
            .filter(|(_, &label)| label == 1)
            .map(|(i, _)| i)
            .collect()
    }

    /// Get number of passages
    pub fn num_passages(&self) -> usize {
        self.passages.len()
    }
}

/// MS MARCO evaluation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MsMarcoEvalConfig {
    /// Path to MS MARCO validation data (JSONL format)
    pub data_path: String,
    /// Base model name (e.g., "bert-base-uncased")
    pub model_name: String,
    /// Optional LoRA checkpoint path
    pub checkpoint_path: Option<String>,
    /// Optional sample size (None = full dataset)
    pub sample_size: Option<usize>,
    /// LoRA rank
    pub lora_rank: usize,
    /// LoRA alpha
    pub lora_alpha: f32,
    /// Device string (e.g., "auto", "cuda:0", "cpu")
    pub device: String,
    /// Whether to output JSON progress updates
    pub json_progress: bool,
}

impl Default for MsMarcoEvalConfig {
    fn default() -> Self {
        Self {
            data_path: "data/msmarco_validation.jsonl".to_string(),
            model_name: "bert-base-uncased".to_string(),
            checkpoint_path: None,
            sample_size: None,
            lora_rank: 8,
            lora_alpha: 16.0,
            device: "auto".to_string(),
            json_progress: false,
        }
    }
}

/// Progress update for streaming to UI
#[derive(Debug, Clone, Serialize)]
pub struct EvalProgress {
    /// Number of queries processed
    pub processed: usize,
    /// Total number of queries
    pub total: usize,
    /// Current running MRR@10
    pub current_mrr: f64,
    /// Estimated time remaining in seconds
    pub eta_seconds: f64,
    /// Current status message
    pub status: String,
}

impl EvalProgress {
    /// Create a new progress update
    pub fn new(processed: usize, total: usize, current_mrr: f64, eta_seconds: f64) -> Self {
        Self {
            processed,
            total,
            current_mrr,
            eta_seconds,
            status: "running".to_string(),
        }
    }

    /// Output as JSON line to stdout
    pub fn print_json(&self) {
        if let Ok(json) = serde_json::to_string(self) {
            println!("{}", json);
            let _ = std::io::stdout().flush();
        }
    }
}

/// Final MS MARCO evaluation metrics
#[derive(Debug, Clone, Serialize)]
pub struct MsMarcoMetrics {
    /// MRR@10 - Mean Reciprocal Rank at cutoff 10
    pub mrr_at_10: f64,
    /// NDCG@10 - Normalized Discounted Cumulative Gain at 10
    pub ndcg_at_10: f64,
    /// Recall@10 - Proportion of relevant docs found in top 10
    pub recall_at_10: f64,
    /// Recall@100 - Proportion of relevant docs found in top 100
    pub recall_at_100: f64,
    /// Number of queries evaluated
    pub num_queries: usize,
    /// Total evaluation time in seconds
    pub elapsed_seconds: f64,
}

impl MsMarcoMetrics {
    /// Create metrics from RetrievalMetrics
    pub fn from_retrieval_metrics(metrics: &RetrievalMetrics, elapsed_seconds: f64) -> Self {
        // Extract MRR (already computed by evaluator)
        let mrr_at_10 = metrics.mrr;

        // Extract NDCG@10
        let ndcg_at_10 = metrics
            .ndcg
            .iter()
            .find(|(k, _)| *k == 10)
            .map(|(_, v)| *v)
            .unwrap_or(0.0);

        // Extract Recall@10
        let recall_at_10 = metrics
            .recall
            .iter()
            .find(|(k, _)| *k == 10)
            .map(|(_, v)| *v)
            .unwrap_or(0.0);

        // Extract Recall@100
        let recall_at_100 = metrics
            .recall
            .iter()
            .find(|(k, _)| *k == 100)
            .map(|(_, v)| *v)
            .unwrap_or(0.0);

        Self {
            mrr_at_10,
            ndcg_at_10,
            recall_at_10,
            recall_at_100,
            num_queries: metrics.num_queries,
            elapsed_seconds,
        }
    }

    /// Output as JSON line to stdout
    pub fn print_json(&self) {
        if let Ok(json) = serde_json::to_string(self) {
            println!("{}", json);
            let _ = std::io::stdout().flush();
        }
    }
}

impl std::fmt::Display for MsMarcoMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "MS MARCO Evaluation Results ({} queries):", self.num_queries)?;
        writeln!(f, "  MRR@10:     {:.4}", self.mrr_at_10)?;
        writeln!(f, "  NDCG@10:    {:.4}", self.ndcg_at_10)?;
        writeln!(f, "  Recall@10:  {:.4}", self.recall_at_10)?;
        writeln!(f, "  Recall@100: {:.4}", self.recall_at_100)?;
        writeln!(f, "  Time:       {:.1}s", self.elapsed_seconds)?;
        Ok(())
    }
}

/// Load MS MARCO validation examples from JSONL file
pub fn load_msmarco_examples(path: &Path, sample_size: Option<usize>) -> Result<Vec<MsMarcoExample>> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open MS MARCO data file: {:?}", path))?;
    let reader = BufReader::new(file);

    let mut examples = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("Failed to read line {} from MS MARCO file", i + 1))?;
        if line.trim().is_empty() {
            continue;
        }

        let example: MsMarcoExample = serde_json::from_str(&line)
            .with_context(|| format!("Failed to parse MS MARCO example at line {}", i + 1))?;
        examples.push(example);

        // Check sample size limit
        if let Some(max) = sample_size {
            if examples.len() >= max {
                break;
            }
        }
    }

    Ok(examples)
}

/// MS MARCO Evaluator
///
/// Evaluates a fine-tuned embedding model on MS MARCO v1.1 passage re-ranking task.
pub struct MsMarcoEvaluator {
    /// K values for metrics computation
    k_values: Vec<usize>,
    /// Whether to output JSON progress
    json_progress: bool,
}

impl MsMarcoEvaluator {
    /// Create a new evaluator with standard MS MARCO K values
    pub fn new(json_progress: bool) -> Self {
        Self {
            // Standard MS MARCO evaluation K values
            k_values: vec![10, 100],
            json_progress,
        }
    }

    /// Evaluate query results and compute metrics
    ///
    /// This method is called after all embeddings have been computed and rankings determined.
    /// It uses the existing RetrievalEvaluator to compute standard IR metrics.
    pub fn evaluate(&self, query_results: &[QueryResult]) -> RetrievalMetrics {
        let evaluator = RetrievalEvaluator::with_k_values(self.k_values.clone());
        evaluator.evaluate(query_results)
    }

    /// Create a QueryResult from ranking results
    ///
    /// # Arguments
    /// * `query_id` - Unique identifier for this query
    /// * `ranked_indices` - Passage indices sorted by similarity (descending)
    /// * `relevant_indices` - Indices of relevant passages (ground truth)
    pub fn create_query_result(
        query_id: impl Into<String>,
        ranked_indices: Vec<usize>,
        relevant_indices: Vec<usize>,
    ) -> QueryResult {
        // Convert indices to string IDs
        let retrieved: Vec<String> = ranked_indices.iter().map(|i| format!("p{}", i)).collect();
        let relevant: Vec<String> = relevant_indices.iter().map(|i| format!("p{}", i)).collect();

        QueryResult::new(query_id, retrieved, relevant)
    }

    /// Print progress update
    pub fn report_progress(&self, processed: usize, total: usize, current_mrr: f64, start_time: Instant) {
        let elapsed = start_time.elapsed().as_secs_f64();
        let queries_per_sec = if elapsed > 0.0 {
            processed as f64 / elapsed
        } else {
            0.0
        };
        let remaining = total - processed;
        let eta_seconds = if queries_per_sec > 0.0 {
            remaining as f64 / queries_per_sec
        } else {
            0.0
        };

        if self.json_progress {
            EvalProgress::new(processed, total, current_mrr, eta_seconds).print_json();
        }
    }
}

/// Compute cosine similarity between two vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Rank passages by similarity scores (descending order)
///
/// Returns indices sorted by score from highest to lowest
pub fn rank_by_similarity(scores: &[f32]) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = scores.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.into_iter().map(|(i, _)| i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);

        let a = vec![1.0, 1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let expected = 1.0 / 2.0_f32.sqrt();
        assert!((cosine_similarity(&a, &b) - expected).abs() < 1e-6);
    }

    #[test]
    fn test_rank_by_similarity() {
        let scores = vec![0.3, 0.8, 0.1, 0.9, 0.5];
        let ranked = rank_by_similarity(&scores);
        assert_eq!(ranked, vec![3, 1, 4, 0, 2]); // 0.9, 0.8, 0.5, 0.3, 0.1
    }

    #[test]
    fn test_msmarco_example() {
        let example = MsMarcoExample {
            query: "test query".to_string(),
            passages: vec!["p1".to_string(), "p2".to_string(), "p3".to_string()],
            labels: vec![0, 1, 0],
        };

        assert_eq!(example.relevant_indices(), vec![1]);
        assert_eq!(example.num_passages(), 3);
    }

    #[test]
    fn test_create_query_result() {
        let result = MsMarcoEvaluator::create_query_result(
            "q1",
            vec![2, 0, 1], // passages ranked: p2, p0, p1
            vec![1],       // p1 is relevant
        );

        assert_eq!(result.query_id, "q1");
        assert_eq!(result.retrieved, vec!["p2", "p0", "p1"]);
        assert!(result.relevant.contains("p1"));

        // MRR should be 1/3 since p1 is at rank 3
        let rr = result.reciprocal_rank();
        assert!((rr - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_msmarco_metrics_display() {
        let metrics = MsMarcoMetrics {
            mrr_at_10: 0.3456,
            ndcg_at_10: 0.4567,
            recall_at_10: 0.5678,
            recall_at_100: 0.8901,
            num_queries: 1000,
            elapsed_seconds: 123.4,
        };

        let display = format!("{}", metrics);
        assert!(display.contains("MRR@10"));
        assert!(display.contains("0.3456"));
    }
}
