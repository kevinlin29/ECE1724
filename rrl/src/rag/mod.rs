//! RAG (Retrieval-Augmented Generation) Pipeline
//!
//! This module provides a complete RAG pipeline for document Q&A,
//! integrating retrieval with local LLM generation.
//!
//! # Architecture
//!
//! ```text
//! User Query
//!     │
//!     ▼
//! ┌─────────────┐
//! │  Retriever  │  ← Uses embedder for dense retrieval
//! └─────────────┘
//!     │
//!     ▼ SearchResults
//! ┌─────────────┐
//! │   Context   │  ← Formats retrieved docs into prompt
//! │   Builder   │
//! └─────────────┘
//!     │
//!     ▼ Formatted Prompt
//! ┌─────────────┐
//! │  Generator  │  ← Local LLM (Qwen2, etc.)
//! └─────────────┘
//!     │
//!     ▼
//! RagResponse (answer + sources)
//! ```
//!
//! # Example
//!
//! ```ignore
//! use rrl::rag::{RagPipelineBuilder, RagQuery, RagConfig};
//!
//! // Build pipeline
//! let pipeline = RagPipelineBuilder::new()
//!     .embedder(embedder)
//!     .retriever(retriever)
//!     .generator(generator)
//!     .build()?;
//!
//! // Execute query
//! let query = RagQuery::new("What is the recipe for pasta?")
//!     .with_top_k(5);
//! let response = pipeline.query(query)?;
//!
//! println!("{}", response.answer);
//! ```

pub mod context;
pub mod generator;
pub mod pipeline;
pub mod query;

// Alias for TUI compatibility: `crate::rag::generation::GeneratorConfig`
pub mod generation {
    pub use super::generator::{GeneratorConfig, SamplingParams, CandleGenerator};
}

// Re-exports for convenience
pub use context::ContextBuilder;
pub use generator::{CandleGenerator, Generator as GeneratorTrait, GeneratorConfig, SamplingParams};
pub use pipeline::{RagConfig, RagPipeline, RagPipelineBuilder, RetrievalStrategy};
pub use query::{RagQuery, RagResponse, Source};

use anyhow::Result;

/// Generator wrapper type for TUI compatibility
///
/// This is a concrete type that wraps a boxed generator trait object,
/// allowing the TUI to use `Generator::new(config)` syntax.
pub struct Generator {
    inner: Box<dyn GeneratorTrait>,
}

impl Generator {
    /// Create a new generator from config
    pub fn new(config: GeneratorConfig) -> Result<Self> {
        let inner = CandleGenerator::new(config)?;
        Ok(Self {
            inner: Box::new(inner),
        })
    }

    /// Generate a response given a prompt
    pub fn generate(&self, prompt: &str, params: &SamplingParams) -> Result<String> {
        self.inner.generate(prompt, params)
    }

    /// Generate with streaming output
    pub fn generate_stream(
        &self,
        prompt: &str,
        params: &SamplingParams,
    ) -> Result<Box<dyn Iterator<Item = Result<String>> + Send + '_>> {
        self.inner.generate_stream(prompt, params)
    }

    /// Get the model name
    pub fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    /// Get max context length
    pub fn max_context_length(&self) -> usize {
        self.inner.max_context_length()
    }

    /// Count tokens in text
    pub fn count_tokens(&self, text: &str) -> Result<usize> {
        self.inner.count_tokens(text)
    }

    /// Get the underlying generator trait object
    pub fn into_inner(self) -> Box<dyn GeneratorTrait> {
        self.inner
    }
}
