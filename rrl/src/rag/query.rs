//! RAG query and response types
//!
//! Defines the core data structures for RAG pipeline input/output.

use serde::{Deserialize, Serialize};

/// Query input for RAG pipeline
#[derive(Debug, Clone)]
pub struct RagQuery {
    /// The user's question
    pub query: String,
    /// Number of documents to retrieve
    pub top_k: usize,
    /// Optional conversation history (query, response) pairs
    pub history: Vec<(String, String)>,
    /// Whether to include citations in response
    pub include_citations: bool,
}

impl RagQuery {
    /// Create a new RAG query
    pub fn new(query: &str) -> Self {
        Self {
            query: query.to_string(),
            top_k: 5,
            history: Vec::new(),
            include_citations: true,
        }
    }

    /// Set the number of documents to retrieve
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Add conversation history
    pub fn with_history(mut self, history: Vec<(String, String)>) -> Self {
        self.history = history;
        self
    }

    /// Set whether to include citations
    pub fn with_citations(mut self, include: bool) -> Self {
        self.include_citations = include;
        self
    }
}

/// Source document reference in response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    /// Chunk identifier
    pub chunk_id: String,
    /// Parent document identifier
    pub document_id: String,
    /// Relevance score from retrieval
    pub score: f32,
    /// Text snippet from the source
    pub snippet: String,
}

impl Source {
    /// Create a new source reference
    pub fn new(chunk_id: &str, document_id: &str, score: f32, snippet: &str) -> Self {
        Self {
            chunk_id: chunk_id.to_string(),
            document_id: document_id.to_string(),
            score,
            snippet: snippet.to_string(),
        }
    }
}

/// Response from RAG pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagResponse {
    /// Generated answer
    pub answer: String,
    /// Source documents used
    pub sources: Vec<Source>,
    /// Raw context sent to LLM (for debugging)
    pub context: String,
    /// Retrieval time in milliseconds
    pub retrieval_time_ms: u64,
    /// Generation time in milliseconds
    pub generation_time_ms: u64,
    /// Total tokens used in prompt
    pub tokens_used: usize,
}

impl RagResponse {
    /// Create a new RAG response
    pub fn new(
        answer: String,
        sources: Vec<Source>,
        context: String,
        retrieval_time_ms: u64,
        generation_time_ms: u64,
        tokens_used: usize,
    ) -> Self {
        Self {
            answer,
            sources,
            context,
            retrieval_time_ms,
            generation_time_ms,
            tokens_used,
        }
    }

    /// Get total processing time in milliseconds
    pub fn total_time_ms(&self) -> u64 {
        self.retrieval_time_ms + self.generation_time_ms
    }
}

impl std::fmt::Display for RagResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Answer: {}", self.answer)?;
        writeln!(f, "\nSources ({}):", self.sources.len())?;
        for (i, source) in self.sources.iter().enumerate() {
            writeln!(
                f,
                "  [{}] {} (score: {:.4})",
                i + 1,
                source.document_id,
                source.score
            )?;
        }
        writeln!(
            f,
            "\nTiming: retrieval={}ms, generation={}ms, total={}ms",
            self.retrieval_time_ms,
            self.generation_time_ms,
            self.total_time_ms()
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rag_query_builder() {
        let query = RagQuery::new("What is Rust?")
            .with_top_k(10)
            .with_citations(false);

        assert_eq!(query.query, "What is Rust?");
        assert_eq!(query.top_k, 10);
        assert!(!query.include_citations);
    }

    #[test]
    fn test_rag_response_display() {
        let response = RagResponse {
            answer: "Test answer".to_string(),
            sources: vec![Source::new("c1", "doc1", 0.95, "snippet")],
            context: "context".to_string(),
            retrieval_time_ms: 100,
            generation_time_ms: 500,
            tokens_used: 256,
        };

        let display = format!("{}", response);
        assert!(display.contains("Test answer"));
        assert!(display.contains("doc1"));
        assert!(display.contains("600ms"));
    }
}
