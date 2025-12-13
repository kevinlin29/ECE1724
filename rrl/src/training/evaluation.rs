//! Evaluation utilities for fine-tuned models
//!
//! Provides evaluation metrics and utilities for assessing model performance
//! on various tasks including multiple choice, ranking, and retrieval.

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::training::models::{EmbeddingModel, TokenizerWrapper};

/// Recipe MPR (Multiple Choice) example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecipeMprExample {
    /// The query/question
    pub query: String,
    /// Map of option_id -> option_text
    pub options: HashMap<String, String>,
    /// The correct answer ID
    pub answer: String,
}

/// Evaluation result for multiple choice tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    /// Top-1 accuracy (correct answer is rank 1)
    pub accuracy: f64,
    /// Mean Reciprocal Rank
    pub mrr: f64,
    /// Top-3 accuracy (correct answer in top 3)
    pub top_3_accuracy: f64,
    /// Number of examples evaluated
    pub num_examples: usize,
    /// Per-example results (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub per_example: Option<Vec<ExampleResult>>,
}

/// Result for a single example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExampleResult {
    /// Query text
    pub query: String,
    /// Correct answer
    pub correct_answer: String,
    /// Rank of correct answer (1-based)
    pub rank: usize,
    /// Top-3 predictions
    pub top_3: Vec<String>,
}

impl std::fmt::Display for EvaluationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Evaluation Results ({} examples):", self.num_examples)?;
        writeln!(f, "  Top-1 Accuracy: {:.2}%", self.accuracy * 100.0)?;
        writeln!(f, "  Top-3 Accuracy: {:.2}%", self.top_3_accuracy * 100.0)?;
        writeln!(f, "  Mean Reciprocal Rank: {:.4}", self.mrr)?;
        Ok(())
    }
}

/// Load Recipe MPR examples from JSON file
///
/// # Arguments
/// * `path` - Path to JSON file containing examples
///
/// # Returns
/// * Vector of RecipeMprExample
///
/// # Example
/// ```ignore
/// let examples = load_recipe_mpr_examples("data/recipe_mpr_test.json")?;
/// println!("Loaded {} examples", examples.len());
/// ```
pub fn load_recipe_mpr_examples(path: impl AsRef<Path>) -> Result<Vec<RecipeMprExample>> {
    let content = std::fs::read_to_string(path.as_ref())
        .with_context(|| format!("Failed to read file: {:?}", path.as_ref()))?;

    let examples: Vec<RecipeMprExample> = serde_json::from_str(&content)
        .context("Failed to parse Recipe MPR JSON")?;

    tracing::info!("Loaded {} Recipe MPR examples", examples.len());
    Ok(examples)
}

/// Evaluate model on multiple choice task
///
/// # Arguments
/// * `model` - The embedding model to evaluate
/// * `tokenizer` - Tokenizer for encoding text
/// * `examples` - List of multiple choice examples
/// * `device` - Device to run evaluation on
///
/// # Returns
/// * EvaluationResult with accuracy and MRR metrics
///
/// # Example
/// ```ignore
/// let result = evaluate_multiple_choice(
///     &model,
///     &tokenizer,
///     &examples,
///     &device,
/// )?;
/// println!("{}", result);
/// ```
pub fn evaluate_multiple_choice(
    model: &dyn EmbeddingModel,
    tokenizer: &TokenizerWrapper,
    examples: &[RecipeMprExample],
    device: &Device,
) -> Result<EvaluationResult> {
    if examples.is_empty() {
        return Err(anyhow::anyhow!("No examples provided for evaluation"));
    }

    tracing::info!("Evaluating on {} examples...", examples.len());

    let mut correct_top1 = 0;
    let mut correct_top3 = 0;
    let mut total_reciprocal_rank = 0.0;
    let mut per_example_results = Vec::new();

    for (idx, example) in examples.iter().enumerate() {
        if (idx + 1) % 10 == 0 {
            tracing::debug!("Progress: {}/{}", idx + 1, examples.len());
        }

        // Encode query
        let query_batch = tokenizer.encode_batch(&[example.query.clone()], true)
            .context("Failed to tokenize query")?;
        let (query_ids, query_mask) = query_batch.to_tensors(device)?;
        let query_emb = model.forward(&query_ids, &query_mask)
            .context("Failed to encode query")?;

        // Encode all options
        let option_texts: Vec<String> = example.options.values().cloned().collect();
        let option_ids: Vec<String> = example.options.keys().cloned().collect();

        let options_batch = tokenizer.encode_batch(&option_texts, true)
            .context("Failed to tokenize options")?;
        let (options_ids, options_mask) = options_batch.to_tensors(device)?;
        let options_emb = model.forward(&options_ids, &options_mask)
            .context("Failed to encode options")?;

        // Compute similarities
        let similarities = compute_cosine_similarities(&query_emb, &options_emb)?;

        // Get rankings
        let mut scores_with_ids: Vec<(f32, String, String)> = similarities
            .iter()
            .zip(option_ids.iter())
            .zip(option_texts.iter())
            .map(|((score, id), text)| (*score, id.clone(), text.clone()))
            .collect();

        // Sort by score descending
        scores_with_ids.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Find rank of correct answer
        let correct_rank = scores_with_ids
            .iter()
            .position(|(_, id, _)| id == &example.answer)
            .map(|pos| pos + 1) // Convert to 1-based rank
            .unwrap_or(option_ids.len() + 1);

        // Update metrics
        if correct_rank == 1 {
            correct_top1 += 1;
        }
        if correct_rank <= 3 {
            correct_top3 += 1;
        }
        total_reciprocal_rank += 1.0 / correct_rank as f64;

        // Store per-example result
        let top_3: Vec<String> = scores_with_ids
            .iter()
            .take(3)
            .map(|(_, _, text)| text.clone())
            .collect();

        per_example_results.push(ExampleResult {
            query: example.query.clone(),
            correct_answer: example.options.get(&example.answer).unwrap().clone(),
            rank: correct_rank,
            top_3,
        });
    }

    let num_examples = examples.len();
    let accuracy = correct_top1 as f64 / num_examples as f64;
    let top_3_accuracy = correct_top3 as f64 / num_examples as f64;
    let mrr = total_reciprocal_rank / num_examples as f64;

    tracing::info!("Evaluation complete!");
    tracing::info!("  Top-1 Accuracy: {:.2}%", accuracy * 100.0);
    tracing::info!("  Top-3 Accuracy: {:.2}%", top_3_accuracy * 100.0);
    tracing::info!("  MRR: {:.4}", mrr);

    Ok(EvaluationResult {
        accuracy,
        mrr,
        top_3_accuracy,
        num_examples,
        per_example: Some(per_example_results),
    })
}

/// Compute cosine similarities between query and documents
fn compute_cosine_similarities(query: &Tensor, documents: &Tensor) -> Result<Vec<f32>> {
    // Normalize embeddings
    let query_norm = normalize_embedding(query)?;
    let docs_norm = normalize_embedding(documents)?;

    // Compute dot products (cosine similarity for normalized vectors)
    // query: [1, hidden_dim]
    // docs: [num_docs, hidden_dim]
    // result: [1, num_docs]
    let similarities = query_norm.matmul(&docs_norm.t()?)?;

    // Convert to Vec<f32>
    let sims = similarities.squeeze(0)?.to_vec1::<f32>()?;
    Ok(sims)
}

/// Normalize embedding to unit length
fn normalize_embedding(embedding: &Tensor) -> Result<Tensor> {
    let norm = embedding
        .sqr()?
        .sum_keepdim(candle_core::D::Minus1)?
        .sqrt()?;
    let norm = norm.clamp(1e-12, f64::MAX)?;
    Ok(embedding.broadcast_div(&norm)?)
}

/// Evaluate on retrieval task (generic)
///
/// # Arguments
/// * `model` - Embedding model
/// * `tokenizer` - Tokenizer
/// * `queries` - List of queries
/// * `documents` - List of documents
/// * `relevance` - Relevance judgments (query_idx, doc_idx, relevance_score)
/// * `device` - Device
///
/// # Returns
/// * Retrieval metrics (MRR, nDCG, etc.)
pub fn evaluate_retrieval(
    model: &dyn EmbeddingModel,
    tokenizer: &TokenizerWrapper,
    queries: &[String],
    documents: &[String],
    relevance: &[(usize, usize, f32)],
    device: &Device,
) -> Result<RetrievalMetrics> {
    tracing::info!(
        "Evaluating retrieval: {} queries, {} documents",
        queries.len(),
        documents.len()
    );

    // Encode all documents once
    let docs_batch = tokenizer.encode_batch(documents, true)?;
    let (docs_ids, docs_mask) = docs_batch.to_tensors(device)?;
    let docs_emb = model.forward(&docs_ids, &docs_mask)?;

    let mut mrr_sum = 0.0;
    let mut ndcg_sum = 0.0;

    for (query_idx, query) in queries.iter().enumerate() {
        // Encode query
        let query_batch = tokenizer.encode_batch(&[query.clone()], true)?;
        let (query_ids, query_mask) = query_batch.to_tensors(device)?;
        let query_emb = model.forward(&query_ids, &query_mask)?;

        // Compute similarities
        let similarities = compute_cosine_similarities(&query_emb, &docs_emb)?;

        // Get relevance for this query
        let query_relevance: HashMap<usize, f32> = relevance
            .iter()
            .filter(|(q_idx, _, _)| *q_idx == query_idx)
            .map(|(_, doc_idx, score)| (*doc_idx, *score))
            .collect();

        if query_relevance.is_empty() {
            continue;
        }

        // Rank documents
        let mut doc_scores: Vec<(usize, f32)> = similarities
            .iter()
            .enumerate()
            .map(|(idx, &score)| (idx, score))
            .collect();
        doc_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Compute MRR
        if let Some(rank) = doc_scores
            .iter()
            .position(|(doc_idx, _)| query_relevance.contains_key(doc_idx))
        {
            mrr_sum += 1.0 / (rank + 1) as f64;
        }

        // Compute nDCG@10
        let ndcg = compute_ndcg(&doc_scores, &query_relevance, 10);
        ndcg_sum += ndcg;
    }

    let num_queries = queries.len() as f64;
    Ok(RetrievalMetrics {
        mrr: mrr_sum / num_queries,
        ndcg_at_10: ndcg_sum / num_queries,
        num_queries: queries.len(),
    })
}

/// Retrieval evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalMetrics {
    /// Mean Reciprocal Rank
    pub mrr: f64,
    /// Normalized Discounted Cumulative Gain at 10
    pub ndcg_at_10: f64,
    /// Number of queries
    pub num_queries: usize,
}

/// Compute nDCG (Normalized Discounted Cumulative Gain)
fn compute_ndcg(
    ranked_docs: &[(usize, f32)],
    relevance: &HashMap<usize, f32>,
    k: usize,
) -> f64 {
    // DCG
    let dcg: f64 = ranked_docs
        .iter()
        .take(k)
        .enumerate()
        .map(|(rank, (doc_idx, _))| {
            let rel = relevance.get(doc_idx).copied().unwrap_or(0.0) as f64;
            rel / ((rank + 2) as f64).log2()
        })
        .sum();

    // Ideal DCG
    let mut ideal_rels: Vec<f32> = relevance.values().copied().collect();
    ideal_rels.sort_by(|a, b| b.partial_cmp(a).unwrap());

    let idcg: f64 = ideal_rels
        .iter()
        .take(k)
        .enumerate()
        .map(|(rank, &rel)| rel as f64 / ((rank + 2) as f64).log2())
        .sum();

    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recipe_mpr_example() {
        let mut options = HashMap::new();
        options.insert("a".to_string(), "Option A".to_string());
        options.insert("b".to_string(), "Option B".to_string());

        let example = RecipeMprExample {
            query: "Test query".to_string(),
            options,
            answer: "a".to_string(),
        };

        assert_eq!(example.query, "Test query");
        assert_eq!(example.answer, "a");
        assert_eq!(example.options.len(), 2);
    }

    #[test]
    fn test_evaluation_result_display() {
        let result = EvaluationResult {
            accuracy: 0.75,
            mrr: 0.8333,
            top_3_accuracy: 0.90,
            num_examples: 100,
            per_example: None,
        };

        let display = format!("{}", result);
        assert!(display.contains("75.00%"));
        assert!(display.contains("0.8333"));
    }

    #[test]
    fn test_ndcg_computation() {
        let ranked_docs = vec![(0, 0.9), (1, 0.8), (2, 0.7)];
        let mut relevance = HashMap::new();
        relevance.insert(0, 3.0);
        relevance.insert(1, 2.0);
        relevance.insert(2, 1.0);

        let ndcg = compute_ndcg(&ranked_docs, &relevance, 3);
        assert!(ndcg > 0.0);
        assert!(ndcg <= 1.0);
    }
}