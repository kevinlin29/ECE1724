//! Generator module for text generation
//!
//! Provides trait-based abstraction for decoder LLMs with Candle implementations.

pub mod candle;
pub mod config;

pub use candle::CandleGenerator;
pub use config::{GeneratorConfig, SamplingParams};

use anyhow::Result;

/// Trait for text generation models
///
/// Implementations of this trait provide text generation capabilities
/// for the RAG pipeline, supporting both single-shot and streaming generation.
pub trait Generator: Send + Sync {
    /// Generate a response given a prompt
    ///
    /// # Arguments
    /// * `prompt` - The input prompt for generation
    /// * `params` - Sampling parameters controlling generation
    ///
    /// # Returns
    /// * Generated text response
    fn generate(&self, prompt: &str, params: &SamplingParams) -> Result<String>;

    /// Generate with streaming output
    ///
    /// Returns an iterator that yields tokens as they are generated.
    /// This enables real-time display of generation progress.
    ///
    /// # Arguments
    /// * `prompt` - The input prompt for generation
    /// * `params` - Sampling parameters controlling generation
    ///
    /// # Returns
    /// * Iterator yielding generated tokens
    fn generate_stream(
        &self,
        prompt: &str,
        params: &SamplingParams,
    ) -> Result<Box<dyn Iterator<Item = Result<String>> + Send + '_>>;

    /// Get the model name/identifier
    fn model_name(&self) -> &str;

    /// Get maximum context length (in tokens)
    fn max_context_length(&self) -> usize;

    /// Count tokens in text
    ///
    /// Used for context window management to ensure prompts
    /// don't exceed model limits.
    fn count_tokens(&self, text: &str) -> Result<usize>;
}

/// Factory function for creating generators
///
/// Creates a generator from the provided configuration.
pub fn create_generator(config: GeneratorConfig) -> Result<Box<dyn Generator>> {
    Ok(Box::new(CandleGenerator::new(config)?))
}
