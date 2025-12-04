//! Sparse retrieval using BM25
//!
//! Full-text search via tantivy.

use crate::data::Chunk;
use crate::retrieval::{IndexMetadata, Retriever, SearchResult};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{doc, Index, IndexWriter, ReloadPolicy};

/// Sparse retriever using BM25 for keyword search
pub struct Bm25Retriever {
    /// Tantivy index
    index: Index,
    /// Tantivy reader
    reader: tantivy::IndexReader,
    /// Schema fields
    #[allow(dead_code)]
    schema: Schema,
    /// Content field
    content_field: Field,
    /// Chunk ID field
    chunk_id_field: Field,
    /// Mapping from chunk ID to chunk data
    chunks: HashMap<String, Chunk>,
    /// Index directory path
    #[allow(dead_code)]
    index_path: PathBuf,
    /// Index metadata
    metadata: IndexMetadata,
}

impl Bm25Retriever {
    /// Build a new BM25 index from chunks
    pub fn build(chunks: Vec<Chunk>, index_dir: &Path) -> Result<Self> {
        if chunks.is_empty() {
            anyhow::bail!("Cannot build index with empty chunks");
        }

        tracing::info!("Building BM25 index: {} chunks", chunks.len());

        // Create schema
        let mut schema_builder = Schema::builder();
        let chunk_id_field = schema_builder.add_text_field("chunk_id", STRING | STORED);
        let content_field = schema_builder.add_text_field("content", TEXT | STORED);
        let schema = schema_builder.build();

        // Create index directory
        fs::create_dir_all(index_dir)
            .context("Failed to create index directory")?;

        let tantivy_dir = index_dir.join("tantivy");
        fs::create_dir_all(&tantivy_dir)?;

        // Create index
        let index = Index::create_in_dir(&tantivy_dir, schema.clone())?;

        // Create index writer
        let mut index_writer: IndexWriter = index.writer(50_000_000)?; // 50MB heap

        // Index all chunks
        let mut chunks_map = HashMap::new();
        for chunk in &chunks {
            let doc = doc!(
                chunk_id_field => chunk.id.clone(),
                content_field => chunk.content.clone(),
            );
            index_writer.add_document(doc)?;
            chunks_map.insert(chunk.id.clone(), chunk.clone());
        }

        // Commit the index
        index_writer.commit()?;

        tracing::info!("BM25 index committed");

        // Create reader
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;

        // Save chunks
        let chunks_path = index_dir.join("chunks.json");
        let chunks_json = serde_json::to_string_pretty(&chunks_map)?;
        fs::write(chunks_path, chunks_json)?;

        // Create metadata
        let metadata = IndexMetadata {
            model_name: "bm25".to_string(),
            dimension: 0, // N/A for sparse retrieval
            num_chunks: chunks.len(),
            created_at: chrono::Utc::now().to_rfc3339(),
        };

        // Save metadata
        let metadata_path = index_dir.join("metadata.json");
        let metadata_json = serde_json::to_string_pretty(&metadata)?;
        fs::write(metadata_path, metadata_json)?;

        tracing::info!("BM25 index built successfully");

        Ok(Self {
            index,
            reader,
            schema,
            content_field,
            chunk_id_field,
            chunks: chunks_map,
            index_path: index_dir.to_path_buf(),  
            metadata,
        })
    }

    /// Load the index from disk
    pub fn load(index_dir: &Path) -> Result<Self> {
        tracing::info!("Loading BM25 index from {:?}", index_dir);

        // Load metadata
        let metadata_path = index_dir.join("metadata.json");
        let metadata_json = fs::read_to_string(&metadata_path)
            .context("Failed to read metadata.json")?;
        let metadata: IndexMetadata = serde_json::from_str(&metadata_json)?;

        // Load chunks
        let chunks_path = index_dir.join("chunks.json");
        let chunks_json = fs::read_to_string(&chunks_path)
            .context("Failed to read chunks.json")?;
        let chunks: HashMap<String, Chunk> = serde_json::from_str(&chunks_json)?;

        // Open Tantivy index
        let tantivy_dir = index_dir.join("tantivy");
        let index = Index::open_in_dir(&tantivy_dir)
            .context("Failed to open Tantivy index")?;

        let schema = index.schema();
        let chunk_id_field = schema
            .get_field("chunk_id")
            .context("chunk_id field not found in schema")?;
        let content_field = schema
            .get_field("content")
            .context("content field not found in schema")?;

        // Create reader
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;

        tracing::info!("BM25 index loaded: {} chunks", chunks.len());

        Ok(Self {
            index,
            reader,
            schema, 
            content_field,
            chunk_id_field,
            chunks,
            index_path: index_dir.to_path_buf(),
            metadata,
        })
    }

    /// Get index metadata
    pub fn metadata(&self) -> &IndexMetadata {
        &self.metadata
    }
}

impl Retriever for Bm25Retriever {
    fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        let searcher = self.reader.searcher();

        // Create query parser for content field
        let query_parser = QueryParser::for_index(&self.index, vec![self.content_field]);

        // Parse query
        let query = query_parser
            .parse_query(query)
            .context("Failed to parse query")?;

        // Search
        let top_docs = searcher.search(&query, &TopDocs::with_limit(top_k))?;

        // Convert to SearchResults
        let mut results = Vec::new();
        for (rank, (score, doc_address)) in top_docs.iter().enumerate() {
            let retrieved_doc: tantivy::TantivyDocument = searcher.doc(*doc_address)?;

            // Extract chunk_id from document
            if let Some(chunk_id_value) = retrieved_doc.get_first(self.chunk_id_field) {
                if let Some(chunk_id) = chunk_id_value.as_str() {
                    if let Some(chunk) = self.chunks.get(chunk_id) {
                        results.push(SearchResult {
                            chunk_id: chunk_id.to_string(),
                            chunk: chunk.clone(),
                            score: *score,
                            rank: rank + 1,
                        });
                    }
                }
            }
        }

        Ok(results)
    }

    fn name(&self) -> &str {
        "bm25"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DocumentMetadata;
    use tempfile::tempdir;

    #[test]
    fn test_bm25_build_and_search() {
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

        let temp_dir = tempdir().unwrap();
        let retriever = Bm25Retriever::build(chunks, temp_dir.path()).unwrap();

        // Search for "programming"
        let results = retriever.retrieve("programming", 2).unwrap();

        assert!(!results.is_empty());
        assert!(results[0].chunk_id == "chunk1");
    }

    #[test]
    fn test_bm25_save_load() {
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

        let temp_dir = tempdir().unwrap();
        Bm25Retriever::build(chunks, temp_dir.path()).unwrap();

        // Load
        let loaded = Bm25Retriever::load(temp_dir.path()).unwrap();

        assert_eq!(loaded.metadata.num_chunks, 1);
        assert_eq!(loaded.chunks.len(), 1);
    }
}
