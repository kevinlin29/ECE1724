//! Embedding backend implementations
//!
//! Supports tch (PyTorch) and onnxruntime backends with GPU acceleration.

use crate::embedding::{Embedder, Embedding, EmbeddingConfig, normalize_embedding};
use anyhow::Result;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::Arc;

#[cfg(feature = "onnx-backend")]
pub mod onnx;

#[cfg(feature = "onnx-backend")]
pub use onnx::{HardwareBackend, OnnxEmbedder};

/// Mock embedder for testing (generates random but deterministic embeddings)
pub struct MockEmbedder {
    config: EmbeddingConfig,
    dimension: usize,
}

impl MockEmbedder {
    /// Create a new mock embedder
    pub fn new(config: EmbeddingConfig, dimension: usize) -> Self {
        Self { config, dimension }
    }

    /// Generate a deterministic embedding based on text hash
    fn generate_embedding(&self, text: &str) -> Embedding {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let seed = hasher.finish();

        // Use a simple deterministic pseudo-random generation
        let mut embedding = Vec::with_capacity(self.dimension);
        let mut state = seed;

        for _ in 0..self.dimension {
            // Simple LCG (Linear Congruential Generator)
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let value = ((state / 65536) % 10000) as f32 / 10000.0 - 0.5;
            embedding.push(value);
        }

        if self.config.normalize {
            let mut emb = embedding;
            normalize_embedding(&mut emb);
            emb
        } else {
            embedding
        }
    }
}

impl Embedder for MockEmbedder {
    fn embed(&self, text: &str) -> Result<Embedding> {
        Ok(self.generate_embedding(text))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>> {
        Ok(texts.iter().map(|&text| self.generate_embedding(text)).collect())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        &self.config.model_name
    }
}

/// Simple token-based embedder (bags of tokens with TF-IDF-like weighting)
/// This is a fallback embedder that doesn't require ML models
pub struct TokenEmbedder {
    config: EmbeddingConfig,
    dimension: usize,
}

impl TokenEmbedder {
    /// Create a new token-based embedder
    pub fn new(config: EmbeddingConfig, dimension: usize) -> Self {
        Self { config, dimension }
    }

    /// Generate embeddings based on token hashing
    fn generate_embedding(&self, text: &str) -> Embedding {
        let mut embedding = vec![0.0; self.dimension];

        // Tokenize by whitespace and punctuation
        let tokens: Vec<&str> = text
            .split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
            .filter(|s| !s.is_empty())
            .collect();

        if tokens.is_empty() {
            return embedding;
        }

        // Hash each token to a position in the embedding
        for token in &tokens {
            let mut hasher = DefaultHasher::new();
            token.to_lowercase().hash(&mut hasher);
            let idx = (hasher.finish() as usize) % self.dimension;

            // Increment the count at this position (simple bag-of-words)
            embedding[idx] += 1.0;
        }

        // Apply TF normalization
        let total_tokens = tokens.len() as f32;
        for val in embedding.iter_mut() {
            *val /= total_tokens;
        }

        if self.config.normalize {
            normalize_embedding(&mut embedding);
        }

        embedding
    }
}

impl Embedder for TokenEmbedder {
    fn embed(&self, text: &str) -> Result<Embedding> {
        Ok(self.generate_embedding(text))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>> {
        Ok(texts.iter().map(|&text| self.generate_embedding(text)).collect())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        &self.config.model_name
    }
}

/// Create an embedder based on backend name
pub fn create_embedder(
    backend: &str,
    config: EmbeddingConfig,
    dimension: usize,
) -> Result<Arc<dyn Embedder>> {
    match backend {
        "mock" => Ok(Arc::new(MockEmbedder::new(config, dimension))),
        "token" => Ok(Arc::new(TokenEmbedder::new(config, dimension))),
        #[cfg(feature = "onnx-backend")]
        "onnx" | "onnx-cpu" | "onnx-cuda" | "onnx-metal" => {
            anyhow::bail!("ONNX backend requires model path. Use create_onnx_embedder() instead.");
        }
        #[cfg(feature = "tch-backend")]
        "tch" => {
            // TODO: Implement tch backend
            anyhow::bail!("Tch backend not yet implemented");
        }
        _ => {
            tracing::warn!("Unknown backend '{}', using token-based embedder", backend);
            Ok(Arc::new(TokenEmbedder::new(config, dimension)))
        }
    }
}

/// Create an ONNX embedder with a specific model path
#[cfg(feature = "onnx-backend")]
pub fn create_onnx_embedder(
    model_path: &Path,
    config: EmbeddingConfig,
    hardware: &str,
) -> Result<Box<dyn Embedder>> {
    let hardware_backend = HardwareBackend::from_str(hardware);
    let embedder = OnnxEmbedder::new(model_path, config, hardware_backend)?;
    Ok(Box::new(embedder))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_embedder() {
        let config = EmbeddingConfig {
            model_name: "test-model".to_string(),
            normalize: true,
            ..Default::default()
        };
        let embedder = MockEmbedder::new(config, 128);

        let text = "Hello, world!";
        let emb = embedder.embed(text).unwrap();

        assert_eq!(emb.len(), 128);

        // Should be deterministic
        let emb2 = embedder.embed(text).unwrap();
        assert_eq!(emb, emb2);

        // Different text should give different embedding
        let emb3 = embedder.embed("Different text").unwrap();
        assert_ne!(emb, emb3);
    }

    #[test]
    fn test_token_embedder() {
        let config = EmbeddingConfig {
            model_name: "token-model".to_string(),
            normalize: true,
            ..Default::default()
        };
        let embedder = TokenEmbedder::new(config, 256);

        let text = "The quick brown fox jumps over the lazy dog";
        let emb = embedder.embed(text).unwrap();

        assert_eq!(emb.len(), 256);

        // Similar texts should have similar embeddings (roughly)
        let similar_text = "The quick brown fox";
        let emb2 = embedder.embed(similar_text).unwrap();

        // Calculate cosine similarity
        let dot_product: f32 = emb.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();
        assert!(dot_product > 0.1); // Should have some similarity
    }

    #[test]
    fn test_embedder_batch() {
        let config = EmbeddingConfig::default();
        let embedder = MockEmbedder::new(config, 64);

        let texts = vec!["text1", "text2", "text3"];
        let embeddings = embedder.embed_batch(&texts).unwrap();

        assert_eq!(embeddings.len(), 3);
        assert_eq!(embeddings[0].len(), 64);
    }
}
