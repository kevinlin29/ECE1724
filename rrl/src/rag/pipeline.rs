//! RAG Pipeline orchestration
//!
//! Provides the main pipeline that coordinates retrieval and generation
//! for question-answering with retrieved context.

use anyhow::{Context, Result};
use std::sync::Arc;
use std::time::Instant;

use crate::embedding::Embedder;
use crate::retrieval::Retriever;

use super::context::ContextBuilder;
use super::generator::{CandleGenerator, Generator as GeneratorTrait, SamplingParams};
use super::query::{RagQuery, RagResponse, Source};
use super::Generator;

/// Retrieval strategy for the RAG pipeline
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RetrievalStrategy {
    /// Dense vector retrieval only (HNSW)
    Dense,
    /// Sparse keyword retrieval only (BM25)
    Sparse,
    /// Hybrid combining dense and sparse
    #[default]
    Hybrid,
}

/// Configuration for the RAG pipeline
#[derive(Debug, Clone)]
pub struct RagConfig {
    /// Retrieval strategy
    pub retrieval_strategy: RetrievalStrategy,

    /// Number of documents to retrieve
    pub top_k: usize,

    /// Maximum characters in context
    pub max_context_chars: usize,

    /// Whether to include citations in response
    pub include_citations: bool,

    /// Prompt template to use
    pub template_name: String,

    /// Sampling parameters for generation
    pub sampling_params: SamplingParams,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            retrieval_strategy: RetrievalStrategy::Hybrid,
            top_k: 5,
            max_context_chars: 4000,
            include_citations: true,
            template_name: "default".to_string(),
            sampling_params: SamplingParams::default(),
        }
    }
}

impl RagConfig {
    /// Create a new config with specified top_k
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set retrieval strategy
    pub fn with_strategy(mut self, strategy: RetrievalStrategy) -> Self {
        self.retrieval_strategy = strategy;
        self
    }

    /// Set max context chars
    pub fn with_max_context_chars(mut self, chars: usize) -> Self {
        self.max_context_chars = chars;
        self
    }

    /// Set template name
    pub fn with_template(mut self, template: &str) -> Self {
        self.template_name = template.to_string();
        self
    }

    /// Set citations flag
    pub fn with_citations(mut self, include: bool) -> Self {
        self.include_citations = include;
        self
    }

    /// Set sampling parameters
    pub fn with_sampling_params(mut self, params: SamplingParams) -> Self {
        self.sampling_params = params;
        self
    }
}

/// RAG Pipeline for document question-answering
///
/// Orchestrates the full RAG workflow:
/// 1. Embed query
/// 2. Retrieve relevant documents
/// 3. Build context from retrieved docs
/// 4. Generate answer using LLM
pub struct RagPipeline {
    embedder: Option<Arc<dyn Embedder>>,
    retriever: Arc<dyn Retriever>,
    generator: Box<dyn GeneratorTrait>,
    context_builder: ContextBuilder,
    config: RagConfig,
}

impl RagPipeline {
    /// Create a new pipeline (use RagPipelineBuilder instead)
    pub fn new(
        embedder: Option<Arc<dyn Embedder>>,
        retriever: Arc<dyn Retriever>,
        generator: Box<dyn GeneratorTrait>,
        config: RagConfig,
    ) -> Self {
        Self {
            embedder,
            retriever,
            generator,
            context_builder: ContextBuilder::new(),
            config,
        }
    }

    /// Create a new pipeline with a Generator wrapper type
    pub fn with_generator(
        embedder: Option<Arc<dyn Embedder>>,
        retriever: Arc<dyn Retriever>,
        generator: Generator,
        config: RagConfig,
    ) -> Self {
        Self {
            embedder,
            retriever,
            generator: generator.into_inner(),
            context_builder: ContextBuilder::new(),
            config,
        }
    }

    /// Execute a RAG query
    pub fn query(&self, query: RagQuery) -> Result<RagResponse> {
        let retrieval_start = Instant::now();

        // Get top_k from query (has default value)
        let top_k = query.top_k;

        // Retrieve relevant documents
        let search_results = self.retriever.retrieve(&query.query, top_k)?;

        let retrieval_time_ms = retrieval_start.elapsed().as_millis() as u64;

        // Build context from retrieved documents
        let context = self.context_builder.build(&search_results, self.config.max_context_chars);

        // Format prompt
        let prompt = self.context_builder.format_prompt(
            &query.query,
            &context,
            &self.config.template_name,
            query.include_citations,
        );

        // Generate answer
        let generation_start = Instant::now();
        let answer = self.generator.generate(&prompt, &self.config.sampling_params)?;
        let generation_time_ms = generation_start.elapsed().as_millis() as u64;

        // Count tokens used
        let tokens_used = self.generator.count_tokens(&prompt)?
            + self.generator.count_tokens(&answer)?;

        // Build sources list
        let sources: Vec<Source> = search_results
            .iter()
            .map(|result| Source {
                chunk_id: result.chunk_id.clone(),
                document_id: result.chunk.document_id.clone(),
                score: result.score,
                snippet: truncate_snippet(&result.chunk.content, 200),
            })
            .collect();

        Ok(RagResponse {
            answer,
            sources,
            context,
            retrieval_time_ms,
            generation_time_ms,
            tokens_used,
        })
    }

    /// Execute a streaming query (returns token iterator)
    pub fn query_stream(
        &self,
        query: RagQuery,
    ) -> Result<(Box<dyn Iterator<Item = Result<String>> + Send + '_>, RagResponse)> {
        let retrieval_start = Instant::now();

        let top_k = query.top_k;
        let search_results = self.retriever.retrieve(&query.query, top_k)?;
        let retrieval_time_ms = retrieval_start.elapsed().as_millis() as u64;

        let context = self.context_builder.build(&search_results, self.config.max_context_chars);
        let prompt = self.context_builder.format_prompt(
            &query.query,
            &context,
            &self.config.template_name,
            query.include_citations,
        );

        let sources: Vec<Source> = search_results
            .iter()
            .map(|result| Source {
                chunk_id: result.chunk_id.clone(),
                document_id: result.chunk.document_id.clone(),
                score: result.score,
                snippet: truncate_snippet(&result.chunk.content, 200),
            })
            .collect();

        // Create streaming iterator
        let stream = self.generator.generate_stream(&prompt, &self.config.sampling_params)?;

        // Return partial response (answer will be empty, filled by stream consumer)
        let partial_response = RagResponse {
            answer: String::new(),
            sources,
            context,
            retrieval_time_ms,
            generation_time_ms: 0,
            tokens_used: 0,
        };

        Ok((stream, partial_response))
    }

    /// Get the generator reference
    pub fn generator(&self) -> &dyn GeneratorTrait {
        self.generator.as_ref()
    }

    /// Get the retriever reference
    pub fn retriever(&self) -> &dyn Retriever {
        self.retriever.as_ref()
    }

    /// Get the embedder reference (if set)
    pub fn embedder(&self) -> Option<&dyn Embedder> {
        self.embedder.as_ref().map(|e| e.as_ref())
    }

    /// Get the config
    pub fn config(&self) -> &RagConfig {
        &self.config
    }
}

/// Builder for RagPipeline
pub struct RagPipelineBuilder {
    embedder: Option<Arc<dyn Embedder>>,
    retriever: Option<Arc<dyn Retriever>>,
    generator: Option<Box<dyn GeneratorTrait>>,
    config: RagConfig,
}

impl RagPipelineBuilder {
    /// Create a new pipeline builder
    pub fn new() -> Self {
        Self {
            embedder: None,
            retriever: None,
            generator: None,
            config: RagConfig::default(),
        }
    }

    /// Set the embedder (optional - used for dense retrieval)
    pub fn embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Set the retriever
    pub fn retriever(mut self, retriever: Arc<dyn Retriever>) -> Self {
        self.retriever = Some(retriever);
        self
    }

    /// Set the generator using a boxed trait object
    pub fn generator_boxed(mut self, generator: Box<dyn GeneratorTrait>) -> Self {
        self.generator = Some(generator);
        self
    }

    /// Set the generator using the Generator wrapper type (TUI compatible)
    pub fn generator(mut self, generator: Generator) -> Self {
        self.generator = Some(generator.into_inner());
        self
    }

    /// Set the generator from a CandleGenerator directly
    pub fn candle_generator(mut self, generator: CandleGenerator) -> Self {
        self.generator = Some(Box::new(generator));
        self
    }

    /// Set the config
    pub fn config(mut self, config: RagConfig) -> Self {
        self.config = config;
        self
    }

    /// Set top_k directly
    pub fn top_k(mut self, top_k: usize) -> Self {
        self.config.top_k = top_k;
        self
    }

    /// Build the pipeline
    pub fn build(self) -> Result<RagPipeline> {
        let retriever = self.retriever
            .context("Retriever is required to build RagPipeline")?;

        let generator = self.generator
            .context("Generator is required to build RagPipeline")?;

        Ok(RagPipeline::new(
            self.embedder,
            retriever,
            generator,
            self.config,
        ))
    }
}

impl Default for RagPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Truncate a text snippet to max length, preserving word boundaries
fn truncate_snippet(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        return text.to_string();
    }

    let truncated = &text[..max_len];
    // Find last space to avoid cutting words
    if let Some(last_space) = truncated.rfind(' ') {
        format!("{}...", &truncated[..last_space])
    } else {
        format!("{}...", truncated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rag_config_defaults() {
        let config = RagConfig::default();

        assert_eq!(config.top_k, 5);
        assert_eq!(config.max_context_chars, 4000);
        assert!(config.include_citations);
    }

    #[test]
    fn test_rag_config_builder() {
        let config = RagConfig::default()
            .with_top_k(10)
            .with_strategy(RetrievalStrategy::Dense)
            .with_citations(false);

        assert_eq!(config.top_k, 10);
        assert_eq!(config.retrieval_strategy, RetrievalStrategy::Dense);
        assert!(!config.include_citations);
    }

    #[test]
    fn test_truncate_snippet() {
        let text = "This is a long piece of text that needs to be truncated";
        let truncated = truncate_snippet(text, 20);

        assert!(truncated.len() <= 23); // 20 + "..."
        assert!(truncated.ends_with("..."));
    }

    #[test]
    fn test_truncate_short_text() {
        let text = "Short";
        let truncated = truncate_snippet(text, 20);

        assert_eq!(truncated, "Short");
    }
}
