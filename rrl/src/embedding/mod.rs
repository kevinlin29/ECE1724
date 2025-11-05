//! Embedding generation and caching
//!
//! Provides trait-based embedding interface with support for multiple backends
//! (tch, onnxruntime) and persistent caching with SQLite.

use anyhow::Result;
use serde::{Deserialize, Serialize};

pub mod backends;
pub mod cache;

// Re-exports
pub use backends::*;
pub use cache::*;

/// Represents an embedding vector
pub type Embedding = Vec<f32>;

/// Pooling strategy for combining token embeddings
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PoolingStrategy {
    /// Mean pooling across all tokens
    Mean,
    /// Use the CLS token embedding
    Cls,
    /// Max pooling across all tokens
    Max,
}

impl Default for PoolingStrategy {
    fn default() -> Self {
        Self::Mean
    }
}

/// Configuration for embedding generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Model name or path
    pub model_name: String,
    /// Pooling strategy
    pub pooling: PoolingStrategy,
    /// Whether to normalize embeddings
    pub normalize: bool,
    /// Maximum sequence length
    pub max_length: usize,
    /// Batch size for processing
    pub batch_size: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model_name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            pooling: PoolingStrategy::Mean,
            normalize: true,
            max_length: 512,
            batch_size: 32,
        }
    }
}

/// Trait for embedding models
pub trait Embedder: Send + Sync {
    /// Embed a single text
    fn embed(&self, text: &str) -> Result<Embedding>;

    /// Embed multiple texts in batch
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>>;

    /// Get the embedding dimension
    fn dimension(&self) -> usize;

    /// Get the model name
    fn model_name(&self) -> &str;
}

/// Apply pooling strategy to token embeddings
pub fn apply_pooling(
    token_embeddings: &[Vec<f32>],
    strategy: PoolingStrategy,
) -> Result<Embedding> {
    if token_embeddings.is_empty() {
        anyhow::bail!("Cannot pool empty token embeddings");
    }

    let dim = token_embeddings[0].len();

    match strategy {
        PoolingStrategy::Mean => {
            let mut result = vec![0.0; dim];
            let num_tokens = token_embeddings.len() as f32;

            for token_emb in token_embeddings {
                for (i, &val) in token_emb.iter().enumerate() {
                    result[i] += val / num_tokens;
                }
            }

            Ok(result)
        }
        PoolingStrategy::Cls => {
            // Return the first token (CLS token)
            Ok(token_embeddings[0].clone())
        }
        PoolingStrategy::Max => {
            let mut result = vec![f32::NEG_INFINITY; dim];

            for token_emb in token_embeddings {
                for (i, &val) in token_emb.iter().enumerate() {
                    result[i] = result[i].max(val);
                }
            }

            Ok(result)
        }
    }
}

/// Normalize an embedding vector (L2 normalization)
pub fn normalize_embedding(embedding: &mut Embedding) {
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm > 0.0 {
        for val in embedding.iter_mut() {
            *val /= norm;
        }
    }
}

/// Calculate cosine similarity between two embeddings
pub fn cosine_similarity(a: &Embedding, b: &Embedding) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot_product / (norm_a * norm_b)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_pooling() {
        let tokens = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];

        let result = apply_pooling(&tokens, PoolingStrategy::Mean).unwrap();
        assert_eq!(result, vec![2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_cls_pooling() {
        let tokens = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];

        let result = apply_pooling(&tokens, PoolingStrategy::Cls).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_max_pooling() {
        let tokens = vec![
            vec![1.0, 5.0, 3.0],
            vec![4.0, 2.0, 6.0],
        ];

        let result = apply_pooling(&tokens, PoolingStrategy::Max).unwrap();
        assert_eq!(result, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_normalize_embedding() {
        let mut emb = vec![3.0, 4.0];
        normalize_embedding(&mut emb);

        // 3-4-5 triangle, so normalized should be [0.6, 0.8]
        assert!((emb[0] - 0.6).abs() < 1e-6);
        assert!((emb[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![1.0, 0.0];
        let d = vec![0.0, 1.0];
        assert!((cosine_similarity(&c, &d) - 0.0).abs() < 1e-6);
    }
}
