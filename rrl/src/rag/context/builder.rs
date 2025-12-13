//! Context builder for RAG prompts
//!
//! Assembles retrieved documents into formatted context
//! suitable for LLM generation.

use crate::retrieval::SearchResult;

use super::templates::PromptTemplates;

/// Builds context from retrieved documents for LLM prompts
pub struct ContextBuilder {
    templates: PromptTemplates,
}

impl ContextBuilder {
    /// Create a new context builder with default templates
    pub fn new() -> Self {
        Self {
            templates: PromptTemplates::default(),
        }
    }

    /// Create a context builder with custom templates
    pub fn with_templates(templates: PromptTemplates) -> Self {
        Self { templates }
    }

    /// Build context string from search results
    ///
    /// # Arguments
    /// * `results` - Retrieved search results
    /// * `max_chars` - Maximum characters to include in context
    ///
    /// # Returns
    /// * Formatted context string with document references
    pub fn build(&self, results: &[SearchResult], max_chars: usize) -> String {
        let mut context = String::new();
        let mut total_chars = 0;

        for (i, result) in results.iter().enumerate() {
            let doc_context = format!(
                "[Document {}] (Source: {}, Score: {:.4})\n{}\n\n",
                i + 1,
                result.chunk.document_id,
                result.score,
                result.chunk.content.trim()
            );

            if total_chars + doc_context.len() > max_chars {
                // Truncate last document to fit
                let remaining = max_chars.saturating_sub(total_chars);
                if remaining > 100 {
                    // Only include if we can fit meaningful content
                    let truncated = &doc_context[..remaining.min(doc_context.len())];
                    context.push_str(truncated);
                    context.push_str("...\n[truncated]");
                }
                break;
            }

            context.push_str(&doc_context);
            total_chars += doc_context.len();
        }

        context.trim().to_string()
    }

    /// Build context with simple formatting (no metadata)
    pub fn build_simple(&self, results: &[SearchResult], max_chars: usize) -> String {
        let mut context = String::new();
        let mut total_chars = 0;

        for (i, result) in results.iter().enumerate() {
            let doc_context = format!("[{}] {}\n\n", i + 1, result.chunk.content.trim());

            if total_chars + doc_context.len() > max_chars {
                break;
            }

            context.push_str(&doc_context);
            total_chars += doc_context.len();
        }

        context.trim().to_string()
    }

    /// Format the complete prompt with system message, context, and query
    ///
    /// # Arguments
    /// * `query` - User's question
    /// * `context` - Pre-built context string
    /// * `template_name` - Name of the template to use
    /// * `include_citations` - Whether to include citation instructions
    ///
    /// # Returns
    /// * Formatted prompt ready for LLM
    pub fn format_prompt(
        &self,
        query: &str,
        context: &str,
        template_name: &str,
        include_citations: bool,
    ) -> String {
        let template = self.templates.get(template_name);

        let citation_instruction = if include_citations {
            "When citing sources, reference them as [Document N] where N is the document number."
        } else {
            ""
        };

        template
            .replace("{context}", context)
            .replace("{query}", query)
            .replace("{citation_instruction}", citation_instruction)
    }

    /// Build and format a complete prompt from search results
    ///
    /// Convenience method that combines `build()` and `format_prompt()`.
    pub fn build_prompt(
        &self,
        query: &str,
        results: &[SearchResult],
        max_context_chars: usize,
        template_name: &str,
        include_citations: bool,
    ) -> String {
        let context = self.build(results, max_context_chars);
        self.format_prompt(query, &context, template_name, include_citations)
    }

    /// Get access to templates for customization
    pub fn templates_mut(&mut self) -> &mut PromptTemplates {
        &mut self.templates
    }
}

impl Default for ContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{Chunk, DocumentMetadata};

    fn make_search_result(chunk_id: &str, doc_id: &str, content: &str, score: f32) -> SearchResult {
        SearchResult {
            chunk_id: chunk_id.to_string(),
            chunk: Chunk {
                id: chunk_id.to_string(),
                document_id: doc_id.to_string(),
                content: content.to_string(),
                start_pos: 0,
                end_pos: content.len(),
                chunk_index: 0,
                metadata: DocumentMetadata::default(),
            },
            score,
            rank: 1,
        }
    }

    #[test]
    fn test_build_context() {
        let builder = ContextBuilder::new();
        let results = vec![
            make_search_result("c1", "doc1", "First document content", 0.95),
            make_search_result("c2", "doc2", "Second document content", 0.85),
        ];

        let context = builder.build(&results, 1000);

        assert!(context.contains("[Document 1]"));
        assert!(context.contains("[Document 2]"));
        assert!(context.contains("First document content"));
        assert!(context.contains("Second document content"));
        assert!(context.contains("doc1"));
        assert!(context.contains("0.95"));
    }

    #[test]
    fn test_build_context_truncation() {
        let builder = ContextBuilder::new();
        let results = vec![
            make_search_result("c1", "doc1", "A".repeat(500).as_str(), 0.95),
            make_search_result("c2", "doc2", "B".repeat(500).as_str(), 0.85),
        ];

        // Limit to 600 chars - should only include first document fully
        let context = builder.build(&results, 600);

        assert!(context.contains("[Document 1]"));
        // Second document may be truncated or omitted
        assert!(context.len() <= 700); // Some buffer for formatting
    }

    #[test]
    fn test_format_prompt() {
        let builder = ContextBuilder::new();

        let prompt = builder.format_prompt(
            "What is X?",
            "Context about X",
            "default",
            true,
        );

        assert!(prompt.contains("What is X?"));
        assert!(prompt.contains("Context about X"));
        assert!(prompt.contains("[Document N]"));
    }

    #[test]
    fn test_format_prompt_without_citations() {
        let builder = ContextBuilder::new();

        let prompt = builder.format_prompt(
            "What is X?",
            "Context about X",
            "concise",
            false,
        );

        assert!(prompt.contains("What is X?"));
        assert!(!prompt.contains("[Document N]"));
    }
}
