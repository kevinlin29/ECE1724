//! Retrieval evaluation metrics
//!
//! Provides standard IR metrics:
//! - Recall@K: Proportion of relevant items found in top-K
//! - Precision@K: Proportion of top-K items that are relevant
//! - MRR (Mean Reciprocal Rank): Average of 1/rank of first relevant item
//! - NDCG@K: Normalized Discounted Cumulative Gain
//! - MAP: Mean Average Precision

use std::collections::HashSet;

/// Single query evaluation result
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Query identifier
    pub query_id: String,
    /// Retrieved document IDs in ranked order
    pub retrieved: Vec<String>,
    /// Relevant document IDs (ground truth)
    pub relevant: HashSet<String>,
}

impl QueryResult {
    /// Create a new query result
    pub fn new(query_id: impl Into<String>, retrieved: Vec<String>, relevant: Vec<String>) -> Self {
        Self {
            query_id: query_id.into(),
            retrieved,
            relevant: relevant.into_iter().collect(),
        }
    }

    /// Compute Recall@K
    ///
    /// Recall@K = |relevant ∩ retrieved@K| / |relevant|
    pub fn recall_at_k(&self, k: usize) -> f64 {
        if self.relevant.is_empty() {
            return 0.0;
        }

        let top_k: HashSet<_> = self.retrieved.iter().take(k).collect();
        let relevant_in_top_k = self.relevant.iter().filter(|r| top_k.contains(r)).count();

        relevant_in_top_k as f64 / self.relevant.len() as f64
    }

    /// Compute Precision@K
    ///
    /// Precision@K = |relevant ∩ retrieved@K| / K
    pub fn precision_at_k(&self, k: usize) -> f64 {
        if k == 0 {
            return 0.0;
        }

        let relevant_in_top_k = self
            .retrieved
            .iter()
            .take(k)
            .filter(|r| self.relevant.contains(*r))
            .count();

        relevant_in_top_k as f64 / k as f64
    }

    /// Compute Reciprocal Rank
    ///
    /// RR = 1 / rank of first relevant item (0 if none found)
    pub fn reciprocal_rank(&self) -> f64 {
        for (i, doc) in self.retrieved.iter().enumerate() {
            if self.relevant.contains(doc) {
                return 1.0 / (i + 1) as f64;
            }
        }
        0.0
    }

    /// Compute DCG@K (Discounted Cumulative Gain)
    ///
    /// DCG@K = Σ(rel_i / log2(i+1)) for i in 1..K
    pub fn dcg_at_k(&self, k: usize) -> f64 {
        self.retrieved
            .iter()
            .take(k)
            .enumerate()
            .map(|(i, doc)| {
                let relevance = if self.relevant.contains(doc) { 1.0 } else { 0.0 };
                relevance / (i as f64 + 2.0).log2()
            })
            .sum()
    }

    /// Compute Ideal DCG@K
    ///
    /// IDCG@K = DCG of ideal ranking (all relevant docs first)
    pub fn idcg_at_k(&self, k: usize) -> f64 {
        let num_relevant = self.relevant.len().min(k);
        (0..num_relevant)
            .map(|i| 1.0 / (i as f64 + 2.0).log2())
            .sum()
    }

    /// Compute NDCG@K (Normalized DCG)
    ///
    /// NDCG@K = DCG@K / IDCG@K
    pub fn ndcg_at_k(&self, k: usize) -> f64 {
        let idcg = self.idcg_at_k(k);
        if idcg == 0.0 {
            return 0.0;
        }
        self.dcg_at_k(k) / idcg
    }

    /// Compute Average Precision
    ///
    /// AP = (1/|relevant|) * Σ(Precision@k * rel(k)) for k in 1..n
    pub fn average_precision(&self) -> f64 {
        if self.relevant.is_empty() {
            return 0.0;
        }

        let mut sum = 0.0;
        let mut relevant_count = 0;

        for (i, doc) in self.retrieved.iter().enumerate() {
            if self.relevant.contains(doc) {
                relevant_count += 1;
                sum += relevant_count as f64 / (i + 1) as f64;
            }
        }

        sum / self.relevant.len() as f64
    }

    /// Check if any relevant document was retrieved
    pub fn has_hit(&self) -> bool {
        self.retrieved.iter().any(|d| self.relevant.contains(d))
    }

    /// Check if any relevant document was retrieved in top-K
    pub fn has_hit_at_k(&self, k: usize) -> bool {
        self.retrieved
            .iter()
            .take(k)
            .any(|d| self.relevant.contains(d))
    }
}

/// Aggregated evaluation metrics across multiple queries
#[derive(Debug, Clone, Default)]
pub struct RetrievalMetrics {
    /// Recall@K values for different K
    pub recall: Vec<(usize, f64)>,
    /// Precision@K values for different K
    pub precision: Vec<(usize, f64)>,
    /// Mean Reciprocal Rank
    pub mrr: f64,
    /// NDCG@K values for different K
    pub ndcg: Vec<(usize, f64)>,
    /// Mean Average Precision
    pub map: f64,
    /// Hit rate (proportion of queries with at least one relevant result)
    pub hit_rate: f64,
    /// Number of queries evaluated
    pub num_queries: usize,
}

impl std::fmt::Display for RetrievalMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Retrieval Metrics ({} queries):", self.num_queries)?;
        writeln!(f, "  MRR: {:.4}", self.mrr)?;
        writeln!(f, "  MAP: {:.4}", self.map)?;
        writeln!(f, "  Hit Rate: {:.4}", self.hit_rate)?;

        for (k, recall) in &self.recall {
            writeln!(f, "  Recall@{}: {:.4}", k, recall)?;
        }

        for (k, precision) in &self.precision {
            writeln!(f, "  Precision@{}: {:.4}", k, precision)?;
        }

        for (k, ndcg) in &self.ndcg {
            writeln!(f, "  NDCG@{}: {:.4}", k, ndcg)?;
        }

        Ok(())
    }
}

/// Retrieval evaluator
#[derive(Debug, Default)]
pub struct RetrievalEvaluator {
    /// K values to compute metrics for
    k_values: Vec<usize>,
}

impl RetrievalEvaluator {
    /// Create a new evaluator with default K values [1, 5, 10, 20]
    pub fn new() -> Self {
        Self {
            k_values: vec![1, 5, 10, 20],
        }
    }

    /// Create evaluator with custom K values
    pub fn with_k_values(k_values: Vec<usize>) -> Self {
        Self { k_values }
    }

    /// Evaluate a single query result
    pub fn evaluate_query(&self, result: &QueryResult) -> SingleQueryMetrics {
        SingleQueryMetrics {
            query_id: result.query_id.clone(),
            recall: self
                .k_values
                .iter()
                .map(|&k| (k, result.recall_at_k(k)))
                .collect(),
            precision: self
                .k_values
                .iter()
                .map(|&k| (k, result.precision_at_k(k)))
                .collect(),
            reciprocal_rank: result.reciprocal_rank(),
            ndcg: self
                .k_values
                .iter()
                .map(|&k| (k, result.ndcg_at_k(k)))
                .collect(),
            average_precision: result.average_precision(),
            has_hit: result.has_hit(),
        }
    }

    /// Evaluate multiple query results and aggregate metrics
    pub fn evaluate(&self, results: &[QueryResult]) -> RetrievalMetrics {
        if results.is_empty() {
            return RetrievalMetrics::default();
        }

        let query_metrics: Vec<_> = results.iter().map(|r| self.evaluate_query(r)).collect();
        let n = query_metrics.len() as f64;

        // Aggregate metrics
        let recall: Vec<(usize, f64)> = self
            .k_values
            .iter()
            .map(|&k| {
                let avg = query_metrics
                    .iter()
                    .filter_map(|m| m.recall.iter().find(|(kv, _)| *kv == k).map(|(_, v)| *v))
                    .sum::<f64>()
                    / n;
                (k, avg)
            })
            .collect();

        let precision: Vec<(usize, f64)> = self
            .k_values
            .iter()
            .map(|&k| {
                let avg = query_metrics
                    .iter()
                    .filter_map(|m| m.precision.iter().find(|(kv, _)| *kv == k).map(|(_, v)| *v))
                    .sum::<f64>()
                    / n;
                (k, avg)
            })
            .collect();

        let ndcg: Vec<(usize, f64)> = self
            .k_values
            .iter()
            .map(|&k| {
                let avg = query_metrics
                    .iter()
                    .filter_map(|m| m.ndcg.iter().find(|(kv, _)| *kv == k).map(|(_, v)| *v))
                    .sum::<f64>()
                    / n;
                (k, avg)
            })
            .collect();

        let mrr = query_metrics.iter().map(|m| m.reciprocal_rank).sum::<f64>() / n;
        let map = query_metrics.iter().map(|m| m.average_precision).sum::<f64>() / n;
        let hit_rate = query_metrics.iter().filter(|m| m.has_hit).count() as f64 / n;

        RetrievalMetrics {
            recall,
            precision,
            mrr,
            ndcg,
            map,
            hit_rate,
            num_queries: results.len(),
        }
    }
}

/// Metrics for a single query
#[derive(Debug, Clone)]
pub struct SingleQueryMetrics {
    /// Query identifier
    pub query_id: String,
    /// Recall@K values
    pub recall: Vec<(usize, f64)>,
    /// Precision@K values
    pub precision: Vec<(usize, f64)>,
    /// Reciprocal Rank
    pub reciprocal_rank: f64,
    /// NDCG@K values
    pub ndcg: Vec<(usize, f64)>,
    /// Average Precision
    pub average_precision: f64,
    /// Whether any relevant document was retrieved
    pub has_hit: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recall_at_k() {
        let result = QueryResult::new(
            "q1",
            vec!["d1", "d2", "d3", "d4", "d5"]
                .into_iter()
                .map(String::from)
                .collect(),
            vec!["d1", "d3", "d6", "d7"]
                .into_iter()
                .map(String::from)
                .collect(),
        );

        // d1, d3 are relevant and in top-5, d6, d7 are not retrieved
        assert!((result.recall_at_k(5) - 0.5).abs() < 0.001); // 2/4
        assert!((result.recall_at_k(3) - 0.5).abs() < 0.001); // 2/4
        assert!((result.recall_at_k(1) - 0.25).abs() < 0.001); // 1/4
    }

    #[test]
    fn test_precision_at_k() {
        let result = QueryResult::new(
            "q1",
            vec!["d1", "d2", "d3", "d4", "d5"]
                .into_iter()
                .map(String::from)
                .collect(),
            vec!["d1", "d3"]
                .into_iter()
                .map(String::from)
                .collect(),
        );

        assert!((result.precision_at_k(5) - 0.4).abs() < 0.001); // 2/5
        assert!((result.precision_at_k(3) - 2.0 / 3.0).abs() < 0.001); // 2/3
        assert!((result.precision_at_k(1) - 1.0).abs() < 0.001); // 1/1
    }

    #[test]
    fn test_reciprocal_rank() {
        let result1 = QueryResult::new(
            "q1",
            vec!["d1", "d2", "d3"]
                .into_iter()
                .map(String::from)
                .collect(),
            vec!["d1"].into_iter().map(String::from).collect(),
        );
        assert!((result1.reciprocal_rank() - 1.0).abs() < 0.001);

        let result2 = QueryResult::new(
            "q2",
            vec!["d1", "d2", "d3"]
                .into_iter()
                .map(String::from)
                .collect(),
            vec!["d3"].into_iter().map(String::from).collect(),
        );
        assert!((result2.reciprocal_rank() - 1.0 / 3.0).abs() < 0.001);

        let result3 = QueryResult::new(
            "q3",
            vec!["d1", "d2", "d3"]
                .into_iter()
                .map(String::from)
                .collect(),
            vec!["d4"].into_iter().map(String::from).collect(),
        );
        assert!((result3.reciprocal_rank()).abs() < 0.001);
    }

    #[test]
    fn test_ndcg() {
        let result = QueryResult::new(
            "q1",
            vec!["d1", "d2", "d3", "d4", "d5"]
                .into_iter()
                .map(String::from)
                .collect(),
            vec!["d1", "d3", "d5"]
                .into_iter()
                .map(String::from)
                .collect(),
        );

        // Perfect ranking would be d1, d3, d5, d2, d4
        // DCG = 1/log2(2) + 0/log2(3) + 1/log2(4) + 0/log2(5) + 1/log2(6)
        //     = 1.0 + 0 + 0.5 + 0 + 0.387 = 1.887
        // IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1.0 + 0.631 + 0.5 = 2.131
        // NDCG = 1.887 / 2.131 = 0.885

        let ndcg = result.ndcg_at_k(5);
        assert!(ndcg > 0.8 && ndcg < 1.0);
    }

    #[test]
    fn test_evaluator() {
        let results = vec![
            QueryResult::new(
                "q1",
                vec!["d1", "d2", "d3"]
                    .into_iter()
                    .map(String::from)
                    .collect(),
                vec!["d1"].into_iter().map(String::from).collect(),
            ),
            QueryResult::new(
                "q2",
                vec!["d1", "d2", "d3"]
                    .into_iter()
                    .map(String::from)
                    .collect(),
                vec!["d3"].into_iter().map(String::from).collect(),
            ),
        ];

        let evaluator = RetrievalEvaluator::with_k_values(vec![1, 3]);
        let metrics = evaluator.evaluate(&results);

        assert_eq!(metrics.num_queries, 2);
        // MRR = (1.0 + 1/3) / 2 = 0.667
        assert!((metrics.mrr - 0.667).abs() < 0.01);
    }
}
