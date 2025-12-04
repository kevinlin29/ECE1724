//! Evaluation utilities for fine-tuned models
//!
//! Provides accuracy metrics for multiple-choice retrieval tasks.

use anyhow::{Context, Result};
use candle_core::Device;
use std::path::Path;

use super::models::{BertLoraModel, TokenizerWrapper};

/// Result of multiple-choice evaluation
#[derive(Debug, Clone)]
pub struct MultipleChoiceResult {
    /// Total number of examples evaluated
    pub total: usize,
    /// Number of correct predictions (top-1)
    pub correct: usize,
    /// Accuracy (correct / total)
    pub accuracy: f64,
    /// Mean Reciprocal Rank
    pub mrr: f64,
    /// Top-3 accuracy
    pub top3_accuracy: f64,
}

impl std::fmt::Display for MultipleChoiceResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Accuracy: {:.2}% ({}/{}) | MRR: {:.4} | Top-3: {:.2}%",
            self.accuracy * 100.0,
            self.correct,
            self.total,
            self.mrr,
            self.top3_accuracy * 100.0
        )
    }
}

/// Multiple-choice example for evaluation
#[derive(Debug, Clone)]
pub struct MCExample {
    /// Query text
    pub query: String,
    /// List of options
    pub options: Vec<String>,
    /// Index of correct answer (0-based)
    pub correct_idx: usize,
}

/// Evaluate multiple-choice accuracy using cosine similarity
///
/// For each example:
/// 1. Encode the query
/// 2. Encode all options
/// 3. Compute cosine similarity between query and each option
/// 4. Check if the highest-scoring option is the correct answer
pub fn evaluate_multiple_choice(
    model: &BertLoraModel,
    tokenizer: &TokenizerWrapper,
    examples: &[MCExample],
    device: &Device,
) -> Result<MultipleChoiceResult> {
    let mut correct = 0;
    let mut mrr_sum = 0.0;
    let mut top3_correct = 0;

    for (i, example) in examples.iter().enumerate() {
        // Encode query
        let query_batch = tokenizer.encode_batch(&[example.query.clone()], true)?;
        let (query_ids, query_mask) = query_batch.to_tensors(device)?;
        let query_emb = model.encode_normalized(&query_ids, &query_mask)?;

        // Encode all options
        let option_batch = tokenizer.encode_batch(&example.options, true)?;
        let (option_ids, option_mask) = option_batch.to_tensors(device)?;
        let option_embs = model.encode_normalized(&option_ids, &option_mask)?;

        // Compute cosine similarities (query @ options^T)
        let similarities = query_emb.matmul(&option_embs.t()?)?;
        let sim_vec: Vec<f32> = similarities.squeeze(0)?.to_vec1()?;

        // Find ranking of correct answer
        let mut indexed: Vec<(usize, f32)> = sim_vec.iter().enumerate().map(|(i, &s)| (i, s)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let correct_rank = indexed
            .iter()
            .position(|(idx, _)| *idx == example.correct_idx)
            .unwrap_or(indexed.len());

        // Top-1 accuracy
        if correct_rank == 0 {
            correct += 1;
        }

        // Top-3 accuracy
        if correct_rank < 3 {
            top3_correct += 1;
        }

        // MRR
        mrr_sum += 1.0 / (correct_rank + 1) as f64;

        // Progress logging
        if (i + 1) % 100 == 0 {
            tracing::info!(
                "Evaluated {}/{} examples, current accuracy: {:.2}%",
                i + 1,
                examples.len(),
                correct as f64 / (i + 1) as f64 * 100.0
            );
        }
    }

    let total = examples.len();
    Ok(MultipleChoiceResult {
        total,
        correct,
        accuracy: correct as f64 / total as f64,
        mrr: mrr_sum / total as f64,
        top3_accuracy: top3_correct as f64 / total as f64,
    })
}

/// Load Recipe-MPR examples for evaluation
pub fn load_recipe_mpr_examples(path: impl AsRef<Path>) -> Result<Vec<MCExample>> {
    let path = path.as_ref();
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read file: {:?}", path))?;

    #[derive(serde::Deserialize)]
    struct RecipeMprItem {
        query: String,
        options: std::collections::HashMap<String, String>,
        answer: String,
    }

    let items: Vec<RecipeMprItem> = serde_json::from_str(&content)
        .context("Failed to parse Recipe-MPR JSON")?;

    let mut examples = Vec::with_capacity(items.len());

    for item in items {
        // Convert options HashMap to Vec, preserving the correct answer index
        let mut option_ids: Vec<&String> = item.options.keys().collect();
        option_ids.sort(); // Consistent ordering

        let options: Vec<String> = option_ids
            .iter()
            .map(|id| item.options.get(*id).unwrap().clone())
            .collect();

        let correct_idx = option_ids
            .iter()
            .position(|id| **id == item.answer)
            .ok_or_else(|| anyhow::anyhow!("Correct answer not found in options"))?;

        examples.push(MCExample {
            query: item.query.trim().to_string(),
            options,
            correct_idx,
        });
    }

    Ok(examples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mc_example() {
        let example = MCExample {
            query: "What is a dessert?".to_string(),
            options: vec![
                "Chocolate cake".to_string(),
                "Grilled chicken".to_string(),
                "Salad".to_string(),
            ],
            correct_idx: 0,
        };

        assert_eq!(example.options.len(), 3);
        assert_eq!(example.correct_idx, 0);
    }
}
