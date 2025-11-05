//! Document ingestion and chunking
//!
//! This module provides functionality for loading documents from various sources
//! (PDF, Markdown, plain text) and splitting them into manageable chunks for
//! embedding and retrieval.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub mod loaders;
pub mod chunkers;

// Re-exports for convenience
pub use loaders::*;
pub use chunkers::*;

/// Represents a loaded document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Unique identifier for the document
    pub id: String,
    /// The source path or identifier
    pub source: String,
    /// Full text content of the document
    pub content: String,
    /// Metadata associated with the document
    pub metadata: DocumentMetadata,
}

/// Metadata for a document
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DocumentMetadata {
    /// File path if applicable
    pub file_path: Option<PathBuf>,
    /// File type (pdf, md, txt, etc.)
    pub file_type: String,
    /// File size in bytes
    pub size: Option<usize>,
    /// Custom metadata fields
    #[serde(flatten)]
    pub custom: std::collections::HashMap<String, String>,
}

/// Represents a chunk of text from a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Unique identifier for the chunk
    pub id: String,
    /// Reference to the parent document ID
    pub document_id: String,
    /// The chunk text content
    pub content: String,
    /// Start position in the original document
    pub start_pos: usize,
    /// End position in the original document
    pub end_pos: usize,
    /// Chunk index in the document
    pub chunk_index: usize,
    /// Metadata from the parent document
    pub metadata: DocumentMetadata,
}

impl Document {
    /// Create a new document
    pub fn new(id: String, source: String, content: String, metadata: DocumentMetadata) -> Self {
        Self {
            id,
            source,
            content,
            metadata,
        }
    }
}

impl Chunk {
    /// Create a new chunk
    pub fn new(
        id: String,
        document_id: String,
        content: String,
        start_pos: usize,
        end_pos: usize,
        chunk_index: usize,
        metadata: DocumentMetadata,
    ) -> Self {
        Self {
            id,
            document_id,
            content,
            start_pos,
            end_pos,
            chunk_index,
            metadata,
        }
    }
}
