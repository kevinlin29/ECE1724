//! Hybrid retrieval
//!
//! Weighted fusion of dense and sparse retrieval signals.

use crate::retrieval::{Retriever, SearchResult};
use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;

/// Hybrid retriever that combines multiple retrievers using Reciprocal Rank Fusion (RRF)
pub struct HybridRetriever {
    /// List of retrievers to combine
    retrievers: Vec<Arc<dyn Retriever>>,
    /// RRF constant (typically 60)
    k: f32,
}

impl HybridRetriever {
    /// Create a new hybrid retriever
    pub fn new(retrievers: Vec<Arc<dyn Retriever>>) -> Self {
        Self {
            retrievers,
            k: 60.0, // Standard RRF constant
        }
    }

    /// Create a new hybrid retriever with custom RRF constant
    pub fn with_k(retrievers: Vec<Arc<dyn Retriever>>, k: f32) -> Self {
        Self { retrievers, k }
    }

    /// Apply Reciprocal Rank Fusion to combine results from multiple retrievers
    ///
    /// RRF formula: score(d) = Σ(1 / (k + rank(d)))
    /// where k is typically 60 and rank is the position in each result list
    fn reciprocal_rank_fusion(
        &self,
        results_lists: Vec<Vec<SearchResult>>,
    ) -> Vec<SearchResult> {
        // Map chunk_id -> (SearchResult, RRF score)
        let mut chunk_scores: HashMap<String, (SearchResult, f32)> = HashMap::new();

        // Process each result list
        for results in results_lists {
            for result in results {
                let rrf_score = 1.0 / (self.k + result.rank as f32);

                chunk_scores
                    .entry(result.chunk_id.clone())
                    .and_modify(|(_, score)| *score += rrf_score)
                    .or_insert((result, rrf_score));
            }
        }

        // Convert to vector and sort by RRF score (descending)
        let mut final_results: Vec<(SearchResult, f32)> = chunk_scores.into_values().collect();
        final_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Update ranks and scores
        final_results
            .into_iter()
            .enumerate()
            .map(|(idx, (mut result, rrf_score))| {
                result.rank = idx + 1;
                result.score = rrf_score;
                result
            })
            .collect()
    }
}

impl Retriever for HybridRetriever {
    fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        // Retrieve from each retriever
        // Fetch more results than needed (2x) to ensure good fusion
        let fetch_k = top_k * 2;

        let mut all_results = Vec::new();
        for retriever in &self.retrievers {
            match retriever.retrieve(query, fetch_k) {
                Ok(results) => {
                    tracing::debug!(
                        "Retriever '{}' returned {} results",
                        retriever.name(),
                        results.len()
                    );
                    all_results.push(results);
                }
                Err(e) => {
                    tracing::warn!(
                        "Retriever '{}' failed: {}",
                        retriever.name(),
                        e
                    );
                }
            }
        }

        if all_results.is_empty() {
            anyhow::bail!("All retrievers failed");
        }

        // Apply RRF
        let mut fused_results = self.reciprocal_rank_fusion(all_results);

        // Truncate to top_k
        fused_results.truncate(top_k);

        Ok(fused_results)
    }

    fn name(&self) -> &str {
        "hybrid"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{Chunk, DocumentMetadata};

    // Mock retriever for testing
    struct MockRetriever {
        name: String,
        results: Vec<SearchResult>,
    }

    impl MockRetriever {
        fn new(name: &str, results: Vec<SearchResult>) -> Self {
            Self {
                name: name.to_string(),
                results,
            }
        }
    }

    impl Retriever for MockRetriever {
        fn retrieve(&self, _query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
            Ok(self.results.iter().take(top_k).cloned().collect())
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    #[test]
    fn test_rrf_fusion() {
        // Create mock chunks
        let chunk1 = Chunk {
            id: "chunk1".to_string(),
            document_id: "doc1".to_string(),
            content: "content1".to_string(),
            start_pos: 0,
            end_pos: 8,
            chunk_index: 0,
            metadata: DocumentMetadata::default(),
        };

        let chunk2 = Chunk {
            id: "chunk2".to_string(),
            document_id: "doc1".to_string(),
            content: "content2".to_string(),
            start_pos: 0,
            end_pos: 8,
            chunk_index: 1,
            metadata: DocumentMetadata::default(),
        };

        let chunk3 = Chunk {
            id: "chunk3".to_string(),
            document_id: "doc1".to_string(),
            content: "content3".to_string(),
            start_pos: 0,
            end_pos: 8,
            chunk_index: 2,
            metadata: DocumentMetadata::default(),
        };

        // Mock retriever 1: ranks chunk1 first, chunk2 second
        let retriever1 = Arc::new(MockRetriever::new(
            "retriever1",
            vec![
                SearchResult {
                    chunk_id: "chunk1".to_string(),
                    chunk: chunk1.clone(),
                    score: 0.9,
                    rank: 1,
                },
                SearchResult {
                    chunk_id: "chunk2".to_string(),
                    chunk: chunk2.clone(),
                    score: 0.7,
                    rank: 2,
                },
            ],
        )) as Arc<dyn Retriever>;

        // Mock retriever 2: ranks chunk2 first, chunk3 second
        let retriever2 = Arc::new(MockRetriever::new(
            "retriever2",
            vec![
                SearchResult {
                    chunk_id: "chunk2".to_string(),
                    chunk: chunk2.clone(),
                    score: 0.95,
                    rank: 1,
                },
                SearchResult {
                    chunk_id: "chunk3".to_string(),
                    chunk: chunk3.clone(),
                    score: 0.8,
                    rank: 2,
                },
            ],
        )) as Arc<dyn Retriever>;

        // Create hybrid retriever
        let hybrid = HybridRetriever::new(vec![retriever1, retriever2]);

        // Retrieve
        let results = hybrid.retrieve("test query", 3).unwrap();

        // chunk2 appears in both lists (rank 1 and 2), so it should be first
        // RRF score for chunk2: 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252
        // RRF score for chunk1: 1/(60+1) = 0.01639
        // RRF score for chunk3: 1/(60+2) = 0.01613

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].chunk_id, "chunk2"); // Highest RRF score
        assert_eq!(results[0].rank, 1);
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn test_hybrid_with_single_retriever() {
        let chunk = Chunk {
            id: "chunk1".to_string(),
            document_id: "doc1".to_string(),
            content: "content".to_string(),
            start_pos: 0,
            end_pos: 7,
            chunk_index: 0,
            metadata: DocumentMetadata::default(),
        };

        let retriever = Arc::new(MockRetriever::new(
            "retriever1",
            vec![SearchResult {
                chunk_id: "chunk1".to_string(),
                chunk: chunk,
                score: 0.9,
                rank: 1,
            }],
        )) as Arc<dyn Retriever>;

        let hybrid = HybridRetriever::new(vec![retriever]);
        let results = hybrid.retrieve("test", 1).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk_id, "chunk1");
    }
}
