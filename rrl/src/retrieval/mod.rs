//! Retrieval engines
//!
//! Implements dense (HNSW), sparse (BM25), and hybrid retrieval strategies.

use crate::data::Chunk;
use anyhow::Result;
use serde::{Deserialize, Serialize};

pub mod dense;
pub mod sparse;
pub mod hybrid;

// Re-exports
pub use dense::*;
pub use sparse::*;
pub use hybrid::*;

/// Search result with chunk and relevance score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Chunk ID
    pub chunk_id: String,
    /// The actual chunk content and metadata
    pub chunk: Chunk,
    /// Relevance score (higher is better)
    pub score: f32,
    /// Rank in the result list (1-indexed)
    pub rank: usize,
}

/// Index metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    /// Model name used for embeddings
    pub model_name: String,
    /// Embedding dimension
    pub dimension: usize,
    /// Number of chunks indexed
    pub num_chunks: usize,
    /// Index creation timestamp
    pub created_at: String,
}

/// Trait for retrieval engines
pub trait Retriever: Send + Sync {
    /// Retrieve top-k most relevant chunks for a query
    fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>>;

    /// Get the name of this retriever
    fn name(&self) -> &str;
}
