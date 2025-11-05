//! Text chunking strategies
//!
//! Implements fixed-size, overlapping, and semantic-based chunking methods.

use crate::data::{Chunk, Document};
use anyhow::Result;
use unicode_segmentation::UnicodeSegmentation;

/// Trait for text chunking strategies
pub trait Chunker {
    /// Split a document into chunks
    fn chunk(&self, document: &Document) -> Result<Vec<Chunk>>;
}

/// Configuration for chunking
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Size of each chunk (in characters or tokens)
    pub chunk_size: usize,
    /// Overlap between chunks (in characters or tokens)
    pub chunk_overlap: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            chunk_overlap: 50,
        }
    }
}

/// Fixed-size chunker that splits text into equal-sized chunks
pub struct FixedSizeChunker {
    config: ChunkConfig,
}

impl FixedSizeChunker {
    /// Create a new fixed-size chunker with the given configuration
    pub fn new(config: ChunkConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(ChunkConfig::default())
    }
}

impl Chunker for FixedSizeChunker {
    fn chunk(&self, document: &Document) -> Result<Vec<Chunk>> {
        let mut chunks = Vec::new();
        let content = &document.content;
        let chars: Vec<char> = content.chars().collect();

        if chars.is_empty() {
            return Ok(chunks);
        }

        let mut start_pos = 0;
        let mut chunk_index = 0;

        while start_pos < chars.len() {
            let end_pos = (start_pos + self.config.chunk_size).min(chars.len());
            let chunk_text: String = chars[start_pos..end_pos].iter().collect();

            let chunk_id = format!("{}_{}", document.id, chunk_index);

            let chunk = Chunk::new(
                chunk_id,
                document.id.clone(),
                chunk_text,
                start_pos,
                end_pos,
                chunk_index,
                document.metadata.clone(),
            );

            chunks.push(chunk);
            chunk_index += 1;

            // Move start position forward, accounting for overlap
            if end_pos >= chars.len() {
                break;
            }
            start_pos = end_pos.saturating_sub(self.config.chunk_overlap);
            if start_pos >= end_pos {
                break; // Prevent infinite loop
            }
        }

        Ok(chunks)
    }
}

/// Overlapping chunker that splits text with configurable overlap
pub struct OverlappingChunker {
    config: ChunkConfig,
}

impl OverlappingChunker {
    /// Create a new overlapping chunker with the given configuration
    pub fn new(config: ChunkConfig) -> Self {
        Self { config }
    }
}

impl Chunker for OverlappingChunker {
    fn chunk(&self, document: &Document) -> Result<Vec<Chunk>> {
        let mut chunks = Vec::new();
        let content = &document.content;
        let chars: Vec<char> = content.chars().collect();

        if chars.is_empty() {
            return Ok(chunks);
        }

        let mut start_pos = 0;
        let mut chunk_index = 0;
        let step_size = self.config.chunk_size.saturating_sub(self.config.chunk_overlap);

        while start_pos < chars.len() {
            let end_pos = (start_pos + self.config.chunk_size).min(chars.len());
            let chunk_text: String = chars[start_pos..end_pos].iter().collect();

            let chunk_id = format!("{}_{}", document.id, chunk_index);

            let chunk = Chunk::new(
                chunk_id,
                document.id.clone(),
                chunk_text,
                start_pos,
                end_pos,
                chunk_index,
                document.metadata.clone(),
            );

            chunks.push(chunk);
            chunk_index += 1;

            // Move by step size
            start_pos += step_size;
            if start_pos >= chars.len() || step_size == 0 {
                break;
            }
        }

        Ok(chunks)
    }
}

/// Semantic chunker that splits text based on sentence boundaries
pub struct SemanticChunker {
    config: ChunkConfig,
}

impl SemanticChunker {
    /// Create a new semantic chunker with the given configuration
    pub fn new(config: ChunkConfig) -> Self {
        Self { config }
    }
}

impl Chunker for SemanticChunker {
    fn chunk(&self, document: &Document) -> Result<Vec<Chunk>> {
        let mut chunks = Vec::new();
        let content = &document.content;

        // Split by sentence boundaries
        let sentences: Vec<&str> = content.unicode_sentences().collect();

        if sentences.is_empty() {
            return Ok(chunks);
        }

        let mut current_chunk = String::new();
        let mut chunk_start_pos = 0;
        let mut chunk_index = 0;
        let mut current_pos = 0;

        for sentence in sentences {
            let sentence_len = sentence.len();

            // If adding this sentence would exceed chunk size and we already have content
            if current_chunk.len() + sentence_len > self.config.chunk_size
                && !current_chunk.is_empty()
            {
                // Create a chunk from accumulated sentences
                let chunk_id = format!("{}_{}", document.id, chunk_index);
                let chunk = Chunk::new(
                    chunk_id,
                    document.id.clone(),
                    current_chunk.trim().to_string(),
                    chunk_start_pos,
                    current_pos,
                    chunk_index,
                    document.metadata.clone(),
                );
                chunks.push(chunk);
                chunk_index += 1;

                // Start new chunk with overlap
                // For simplicity, we'll just start fresh (can be improved with sentence overlap)
                current_chunk.clear();
                chunk_start_pos = current_pos;
            }

            current_chunk.push_str(sentence);
            current_pos += sentence_len;
        }

        // Add the last chunk if there's remaining content
        if !current_chunk.is_empty() {
            let chunk_id = format!("{}_{}", document.id, chunk_index);
            let chunk = Chunk::new(
                chunk_id,
                document.id.clone(),
                current_chunk.trim().to_string(),
                chunk_start_pos,
                current_pos,
                chunk_index,
                document.metadata.clone(),
            );
            chunks.push(chunk);
        }

        Ok(chunks)
    }
}

/// Create a chunker based on strategy name
pub fn create_chunker(strategy: &str, config: ChunkConfig) -> Box<dyn Chunker> {
    match strategy {
        "fixed" => Box::new(FixedSizeChunker::new(config)),
        "overlapping" => Box::new(OverlappingChunker::new(config)),
        "semantic" => Box::new(SemanticChunker::new(config)),
        _ => {
            tracing::warn!("Unknown chunking strategy '{}', using fixed-size", strategy);
            Box::new(FixedSizeChunker::new(config))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DocumentMetadata;
    use std::collections::HashMap;

    fn create_test_document(content: &str) -> Document {
        Document {
            id: "test_doc".to_string(),
            source: "test.txt".to_string(),
            content: content.to_string(),
            metadata: DocumentMetadata {
                file_path: None,
                file_type: "txt".to_string(),
                size: Some(content.len()),
                custom: HashMap::new(),
            },
        }
    }

    #[test]
    fn test_fixed_size_chunker() {
        let doc = create_test_document("Hello world! This is a test document with some content.");
        let config = ChunkConfig {
            chunk_size: 20,
            chunk_overlap: 5,
        };
        let chunker = FixedSizeChunker::new(config);
        let chunks = chunker.chunk(&doc).unwrap();

        assert!(!chunks.is_empty());
        assert!(chunks[0].content.len() <= 20);
    }

    #[test]
    fn test_semantic_chunker() {
        let doc = create_test_document("First sentence. Second sentence. Third sentence. Fourth sentence.");
        let config = ChunkConfig {
            chunk_size: 30,
            chunk_overlap: 0,
        };
        let chunker = SemanticChunker::new(config);
        let chunks = chunker.chunk(&doc).unwrap();

        assert!(!chunks.is_empty());
        // Semantic chunker should respect sentence boundaries
        for chunk in &chunks {
            assert!(chunk.content.ends_with('.') || chunk.content.contains('.'));
        }
    }
}
