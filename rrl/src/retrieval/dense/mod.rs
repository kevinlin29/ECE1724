//! Dense retrieval using HNSW
//!
//! Approximate nearest neighbor search via hnsw_rs.

use crate::data::Chunk;
use crate::embedding::{Embedder, Embedding};
use crate::retrieval::{IndexMetadata, Retriever, SearchResult};
use anyhow::{Context, Result};
use hnsw_rs::hnsw::{Hnsw, Neighbour};
use hnsw_rs::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

/// Dense retriever using HNSW for approximate nearest neighbor search
pub struct HnswRetriever {
    /// HNSW index for vector search
    hnsw: Hnsw<'static, f32, DistCosine>,
    /// Mapping from HNSW point ID to chunk ID
    id_to_chunk_id: HashMap<usize, String>,
    /// Mapping from chunk ID to chunk data
    chunks: HashMap<String, Chunk>,
    /// Embedder for query encoding
    embedder: Arc<dyn Embedder>,
    /// Index metadata
    metadata: IndexMetadata,
}

/// Configuration for HNSW index
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Maximum number of connections per layer (default: 16)
    pub max_connections: usize,
    /// Size of the dynamic candidate list (default: 200)
    pub ef_construction: usize,
    /// Maximum number of layers (default: 16)
    pub max_layers: u8,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            max_connections: 16,
            ef_construction: 200,
            max_layers: 16,
        }
    }
}

impl HnswRetriever {
    /// Build a new HNSW index from chunks and embeddings
    pub fn build(
        chunks: Vec<Chunk>,
        embeddings: Vec<Embedding>,
        embedder: Arc<dyn Embedder>,
        config: HnswConfig,
    ) -> Result<Self> {
        if chunks.len() != embeddings.len() {
            anyhow::bail!(
                "Chunk count ({}) doesn't match embedding count ({})",
                chunks.len(),
                embeddings.len()
            );
        }

        if chunks.is_empty() {
            anyhow::bail!("Cannot build index with empty chunks");
        }

        let dimension = embeddings[0].len();
        tracing::debug!(
            "Building HNSW index: {} chunks, {} dimensions",
            chunks.len(),
            dimension
        );

        // Create HNSW index
        let hnsw: Hnsw<f32, DistCosine> = Hnsw::new(
            config.max_connections,
            chunks.len(),
            config.max_layers as usize,
            config.ef_construction,
            DistCosine,
        );

        // Build mappings
        let mut id_to_chunk_id = HashMap::new();
        let mut chunks_map = HashMap::new();

        // Insert all embeddings into HNSW
        for (idx, (chunk, embedding)) in chunks.iter().zip(embeddings.iter()).enumerate() {
            let point_id = idx;
            hnsw.insert((embedding.as_slice(), point_id));
            id_to_chunk_id.insert(point_id, chunk.id.clone());
            chunks_map.insert(chunk.id.clone(), chunk.clone());
        }

        // Create metadata
        let metadata = IndexMetadata {
            model_name: embedder.model_name().to_string(),
            dimension,
            num_chunks: chunks.len(),
            created_at: chrono::Utc::now().to_rfc3339(),
        };

        tracing::debug!("HNSW index built successfully");

        Ok(Self {
            hnsw,
            id_to_chunk_id,
            chunks: chunks_map,
            embedder,
            metadata,
        })
    }

    /// Save the index to disk (saves metadata and chunks, HNSW rebuilt on load)
    pub fn save(&self, index_dir: &Path) -> Result<()> {
        fs::create_dir_all(index_dir)
            .context("Failed to create index directory")?;

        // Save chunks
        let chunks_path = index_dir.join("chunks.json");
        let chunks_json = serde_json::to_string_pretty(&self.chunks)?;
        fs::write(chunks_path, chunks_json)?;

        // Save ID mapping
        let mapping_path = index_dir.join("id_mapping.json");
        let mapping_json = serde_json::to_string_pretty(&self.id_to_chunk_id)?;
        fs::write(mapping_path, mapping_json)?;

        // Save metadata
        let metadata_path = index_dir.join("metadata.json");
        let metadata_json = serde_json::to_string_pretty(&self.metadata)?;
        fs::write(metadata_path, metadata_json)?;

        tracing::debug!("HNSW index metadata saved to {:?}", index_dir);
        tracing::warn!("Note: HNSW index will be rebuilt on next load from embeddings");
        Ok(())
    }

    /// Load index from disk and rebuild HNSW from embeddings
    /// Note: This requires re-embedding all chunks, which can be slow for large indexes
    pub fn load(index_dir: &Path, embedder: Arc<dyn Embedder>) -> Result<Self> {
        tracing::debug!("Loading HNSW index from {:?}", index_dir);

        // Load metadata first
        let metadata_path = index_dir.join("metadata.json");
        let metadata_json = fs::read_to_string(&metadata_path)
            .context("Failed to read metadata.json")?;
        let metadata: IndexMetadata = serde_json::from_str(&metadata_json)?;

        // Verify embedder matches
        if embedder.model_name() != metadata.model_name {
            tracing::warn!(
                "Embedder model mismatch: index={}, embedder={}",
                metadata.model_name,
                embedder.model_name()
            );
        }

        // Load chunks
        let chunks_path = index_dir.join("chunks.json");
        let chunks_json = fs::read_to_string(&chunks_path)
            .context("Failed to read chunks.json")?;
        let chunks_map: HashMap<String, Chunk> = serde_json::from_str(&chunks_json)?;

        // Load ID mapping
        let mapping_path = index_dir.join("id_mapping.json");
        let mapping_json = fs::read_to_string(&mapping_path)
            .context("Failed to read id_mapping.json")?;
        let _id_to_chunk_id: HashMap<usize, String> = serde_json::from_str(&mapping_json)?;

        // Rebuild HNSW index from chunks
        // We need to re-embed all chunks to rebuild the index
        tracing::debug!("Rebuilding HNSW index from {} chunks", chunks_map.len());

        let mut chunks_vec: Vec<Chunk> = chunks_map.values().cloned().collect();
        // Sort by chunk_id to ensure consistent ordering
        chunks_vec.sort_by(|a, b| a.id.cmp(&b.id));

        let embeddings: Vec<Embedding> = chunks_vec
            .iter()
            .map(|c| embedder.embed(&c.content))
            .collect::<Result<Vec<_>>>()?;

        // Rebuild index using the build method
        Self::build(chunks_vec, embeddings, embedder, HnswConfig::default())
    }

    /// Get index metadata
    pub fn metadata(&self) -> &IndexMetadata {
        &self.metadata
    }

    /// Set search quality parameter (ef: number of neighbors to explore)
    pub fn set_search_ef(&mut self, _ef: usize) {
        self.hnsw.set_searching_mode(true);
        // Note: hnsw_rs doesn't expose set_ef directly,
        // but searching_mode enables better search quality
    }
}

impl Retriever for HnswRetriever {
    fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        // Embed the query
        let query_embedding = self.embedder.embed(query)
            .context("Failed to embed query")?;

        // Search HNSW
        let neighbors: Vec<Neighbour> = self.hnsw.search(
            query_embedding.as_slice(),
            top_k,
            30, // ef_search parameter
        );

        // Convert to SearchResults
        let mut results = Vec::new();
        for (rank, neighbor) in neighbors.iter().enumerate() {
            let point_id = neighbor.d_id;

            if let Some(chunk_id) = self.id_to_chunk_id.get(&point_id) {
                if let Some(chunk) = self.chunks.get(chunk_id) {
                    // Convert distance to similarity score
                    // hnsw_rs returns distance, we want similarity (1 - distance for cosine)
                    let score = 1.0 - neighbor.distance;

                    results.push(SearchResult {
                        chunk_id: chunk_id.clone(),
                        chunk: chunk.clone(),
                        score,
                        rank: rank + 1,
                    });
                }
            }
        }

        Ok(results)
    }

    fn name(&self) -> &str {
        "hnsw"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{Document, DocumentMetadata};
    use crate::embedding::{EmbeddingConfig, MockEmbedder};

    #[test]
    fn test_hnsw_build_and_search() {
        // Create test chunks
        let chunks = vec![
            Chunk {
                id: "chunk1".to_string(),
                document_id: "doc1".to_string(),
                content: "Rust is a systems programming language".to_string(),
                start_pos: 0,
                end_pos: 40,
                chunk_index: 0,
                metadata: DocumentMetadata::default(),
            },
            Chunk {
                id: "chunk2".to_string(),
                document_id: "doc1".to_string(),
                content: "Python is great for data science".to_string(),
                start_pos: 0,
                end_pos: 33,
                chunk_index: 0,
                metadata: DocumentMetadata::default(),
            },
        ];

        // Create embedder and embeddings
        let embedder = Arc::new(MockEmbedder::new(
            EmbeddingConfig::default(),
            384,
        )) as Arc<dyn Embedder>;

        let embeddings: Vec<Embedding> = chunks
            .iter()
            .map(|c| embedder.embed(&c.content).unwrap())
            .collect();

        // Build index
        let retriever = HnswRetriever::build(
            chunks.clone(),
            embeddings,
            embedder.clone(),
            HnswConfig::default(),
        )
        .unwrap();

        // Search
        let results = retriever.retrieve("programming language", 2).unwrap();

        assert_eq!(results.len(), 2);
        // Score should be a reasonable value (HNSW distance-based)
        assert!(results[0].score.is_finite());
    }

    #[test]
    fn test_hnsw_save_load() {
        use tempfile::tempdir;

        let chunks = vec![
            Chunk {
                id: "chunk1".to_string(),
                document_id: "doc1".to_string(),
                content: "test content".to_string(),
                start_pos: 0,
                end_pos: 12,
                chunk_index: 0,
                metadata: DocumentMetadata::default(),
            },
        ];

        let embedder = Arc::new(MockEmbedder::new(
            EmbeddingConfig::default(),
            384,
        )) as Arc<dyn Embedder>;

        let embeddings = vec![embedder.embed("test content").unwrap()];

        let retriever = HnswRetriever::build(
            chunks,
            embeddings,
            embedder.clone(),
            HnswConfig::default(),
        )
        .unwrap();

        // Save
        let temp_dir = tempdir().unwrap();
        retriever.save(temp_dir.path()).unwrap();

        // Load
        let loaded = HnswRetriever::load(temp_dir.path(), embedder).unwrap();

        assert_eq!(loaded.metadata.num_chunks, 1);
        assert_eq!(loaded.chunks.len(), 1);
    }
}
