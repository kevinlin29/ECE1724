//! Generation metrics for RAG evaluation
//!
//! Provides metrics for evaluating the quality of generated text:
//! - Exact Match (EM): Binary measure of exact string match
//! - Token F1 Score: F1 score based on token overlap
//! - ROUGE-L: Longest Common Subsequence based metric

use std::collections::HashSet;

/// Metrics for evaluating generated text quality
#[derive(Debug, Clone, Default)]
pub struct GenerationMetrics {
    /// Exact match score (1.0 or 0.0)
    pub exact_match: f32,
    /// Token-level F1 score
    pub token_f1: f32,
    /// ROUGE-L score (based on longest common subsequence)
    pub rouge_l: f32,
    /// Number of tokens in prediction
    pub pred_tokens: usize,
    /// Number of tokens in reference
    pub ref_tokens: usize,
}

impl GenerationMetrics {
    /// Compute all metrics for a single prediction-reference pair
    pub fn compute(prediction: &str, reference: &str) -> Self {
        let pred_normalized = normalize_text(prediction);
        let ref_normalized = normalize_text(reference);

        let exact_match = if pred_normalized == ref_normalized {
            1.0
        } else {
            0.0
        };

        let pred_tokens = tokenize(&pred_normalized);
        let ref_tokens = tokenize(&ref_normalized);

        let token_f1 = compute_token_f1(&pred_tokens, &ref_tokens);
        let rouge_l = compute_rouge_l(&pred_tokens, &ref_tokens);

        Self {
            exact_match,
            token_f1,
            rouge_l,
            pred_tokens: pred_tokens.len(),
            ref_tokens: ref_tokens.len(),
        }
    }

    /// Compute average metrics over multiple examples
    pub fn average(metrics: &[Self]) -> Self {
        if metrics.is_empty() {
            return Self::default();
        }

        let n = metrics.len() as f32;

        Self {
            exact_match: metrics.iter().map(|m| m.exact_match).sum::<f32>() / n,
            token_f1: metrics.iter().map(|m| m.token_f1).sum::<f32>() / n,
            rouge_l: metrics.iter().map(|m| m.rouge_l).sum::<f32>() / n,
            pred_tokens: (metrics.iter().map(|m| m.pred_tokens).sum::<usize>() as f32 / n) as usize,
            ref_tokens: (metrics.iter().map(|m| m.ref_tokens).sum::<usize>() as f32 / n) as usize,
        }
    }
}

impl std::fmt::Display for GenerationMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Generation Metrics:")?;
        writeln!(f, "  Exact Match: {:.2}%", self.exact_match * 100.0)?;
        writeln!(f, "  Token F1:    {:.2}%", self.token_f1 * 100.0)?;
        writeln!(f, "  ROUGE-L:     {:.2}%", self.rouge_l * 100.0)?;
        writeln!(f, "  Pred tokens: {}", self.pred_tokens)?;
        writeln!(f, "  Ref tokens:  {}", self.ref_tokens)?;
        Ok(())
    }
}

/// Normalize text for comparison (lowercase, strip extra whitespace)
fn normalize_text(text: &str) -> String {
    text.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Tokenize text into words
fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|s| s.to_string())
        .collect()
}

/// Compute token-level F1 score
fn compute_token_f1(pred_tokens: &[String], ref_tokens: &[String]) -> f32 {
    if pred_tokens.is_empty() && ref_tokens.is_empty() {
        return 1.0;
    }
    if pred_tokens.is_empty() || ref_tokens.is_empty() {
        return 0.0;
    }

    let pred_set: HashSet<_> = pred_tokens.iter().collect();
    let ref_set: HashSet<_> = ref_tokens.iter().collect();

    let common = pred_set.intersection(&ref_set).count() as f32;
    let precision = common / pred_tokens.len() as f32;
    let recall = common / ref_tokens.len() as f32;

    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

/// Compute ROUGE-L score based on Longest Common Subsequence
fn compute_rouge_l(pred_tokens: &[String], ref_tokens: &[String]) -> f32 {
    if pred_tokens.is_empty() && ref_tokens.is_empty() {
        return 1.0;
    }
    if pred_tokens.is_empty() || ref_tokens.is_empty() {
        return 0.0;
    }

    let lcs_len = lcs_length(pred_tokens, ref_tokens);
    let precision = lcs_len as f32 / pred_tokens.len() as f32;
    let recall = lcs_len as f32 / ref_tokens.len() as f32;

    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

/// Compute length of Longest Common Subsequence using dynamic programming
fn lcs_length(a: &[String], b: &[String]) -> usize {
    let m = a.len();
    let n = b.len();

    // Use 1D DP to save space
    let mut dp = vec![0usize; n + 1];

    for i in 1..=m {
        let mut prev = 0;
        for j in 1..=n {
            let temp = dp[j];
            if a[i - 1] == b[j - 1] {
                dp[j] = prev + 1;
            } else {
                dp[j] = dp[j].max(dp[j - 1]);
            }
            prev = temp;
        }
    }

    dp[n]
}

/// Evaluate a batch of prediction-reference pairs
pub fn evaluate_generation_batch(
    predictions: &[&str],
    references: &[&str],
) -> GenerationMetrics {
    assert_eq!(
        predictions.len(),
        references.len(),
        "Predictions and references must have the same length"
    );

    let metrics: Vec<GenerationMetrics> = predictions
        .iter()
        .zip(references.iter())
        .map(|(pred, ref_)| GenerationMetrics::compute(pred, ref_))
        .collect();

    GenerationMetrics::average(&metrics)
}

/// Result of RAG evaluation including retrieval and generation metrics
#[derive(Debug, Clone)]
pub struct RagEvaluationResult {
    /// Generation quality metrics
    pub generation: GenerationMetrics,
    /// Number of examples evaluated
    pub num_examples: usize,
    /// Average retrieval time in ms
    pub avg_retrieval_time_ms: f32,
    /// Average generation time in ms
    pub avg_generation_time_ms: f32,
}

impl std::fmt::Display for RagEvaluationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "RAG Evaluation Results ({} examples)", self.num_examples)?;
        writeln!(f, "========================================")?;
        writeln!(f, "{}", self.generation)?;
        writeln!(f, "Timing:")?;
        writeln!(f, "  Avg Retrieval: {:.1}ms", self.avg_retrieval_time_ms)?;
        writeln!(f, "  Avg Generation: {:.1}ms", self.avg_generation_time_ms)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_match() {
        let metrics = GenerationMetrics::compute("hello world", "hello world");
        assert_eq!(metrics.exact_match, 1.0);

        let metrics = GenerationMetrics::compute("hello world", "hello there");
        assert_eq!(metrics.exact_match, 0.0);
    }

    #[test]
    fn test_exact_match_normalization() {
        // Should normalize whitespace and case
        let metrics = GenerationMetrics::compute("  Hello   World  ", "hello world");
        assert_eq!(metrics.exact_match, 1.0);
    }

    #[test]
    fn test_token_f1() {
        // Perfect match
        let metrics = GenerationMetrics::compute("a b c", "a b c");
        assert!((metrics.token_f1 - 1.0).abs() < 0.001);

        // Partial overlap
        let metrics = GenerationMetrics::compute("a b c", "a b d");
        assert!((metrics.token_f1 - 0.666).abs() < 0.01);

        // No overlap
        let metrics = GenerationMetrics::compute("a b c", "d e f");
        assert_eq!(metrics.token_f1, 0.0);
    }

    #[test]
    fn test_rouge_l() {
        // Perfect match
        let metrics = GenerationMetrics::compute("the cat sat", "the cat sat");
        assert!((metrics.rouge_l - 1.0).abs() < 0.001);

        // LCS of 2: "the" and "sat"
        let metrics = GenerationMetrics::compute("the cat sat", "the dog sat");
        assert!((metrics.rouge_l - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_lcs_length() {
        let a = vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string()];
        let b = vec!["a".to_string(), "c".to_string(), "d".to_string()];
        assert_eq!(lcs_length(&a, &b), 3); // "a", "c", "d"
    }

    #[test]
    fn test_empty_texts() {
        let metrics = GenerationMetrics::compute("", "");
        assert_eq!(metrics.exact_match, 1.0);
        assert_eq!(metrics.token_f1, 1.0);
        assert_eq!(metrics.rouge_l, 1.0);

        let metrics = GenerationMetrics::compute("hello", "");
        assert_eq!(metrics.exact_match, 0.0);
        assert_eq!(metrics.token_f1, 0.0);
        assert_eq!(metrics.rouge_l, 0.0);
    }

    #[test]
    fn test_batch_evaluation() {
        let predictions = vec!["a b c", "d e f", "g h i"];
        let references = vec!["a b c", "d e f", "g h j"];

        let metrics = evaluate_generation_batch(&predictions, &references);

        // Two exact matches out of three
        assert!((metrics.exact_match - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_average_metrics() {
        let metrics = vec![
            GenerationMetrics {
                exact_match: 1.0,
                token_f1: 0.8,
                rouge_l: 0.9,
                pred_tokens: 10,
                ref_tokens: 10,
            },
            GenerationMetrics {
                exact_match: 0.0,
                token_f1: 0.6,
                rouge_l: 0.7,
                pred_tokens: 12,
                ref_tokens: 8,
            },
        ];

        let avg = GenerationMetrics::average(&metrics);
        assert!((avg.exact_match - 0.5).abs() < 0.001);
        assert!((avg.token_f1 - 0.7).abs() < 0.001);
        assert!((avg.rouge_l - 0.8).abs() < 0.001);
    }
}
