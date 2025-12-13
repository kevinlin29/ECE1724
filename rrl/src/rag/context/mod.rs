//! Context building for RAG prompts
//!
//! Provides utilities for assembling retrieved documents into
//! prompts suitable for LLM generation.

mod builder;
mod templates;

pub use builder::ContextBuilder;
pub use templates::PromptTemplates;
