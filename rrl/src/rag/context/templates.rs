//! Prompt templates for RAG generation
//!
//! Provides configurable templates for formatting prompts with
//! retrieved context and user queries.

use std::collections::HashMap;

/// Prompt templates for different use cases
pub struct PromptTemplates {
    templates: HashMap<String, String>,
}

impl Default for PromptTemplates {
    fn default() -> Self {
        let mut templates = HashMap::new();

        // Default RAG template
        templates.insert(
            "default".to_string(),
            concat!(
                "You are a helpful assistant that answers questions based on the provided context. ",
                "Use only the information from the context to answer. If the answer cannot be found ",
                "in the context, say \"I cannot find this information in the provided documents.\"\n\n",
                "{citation_instruction}\n\n",
                "Context:\n{context}\n\n",
                "Question: {query}\n\n",
                "Answer:"
            ).to_string(),
        );

        // Concise template
        templates.insert(
            "concise".to_string(),
            concat!(
                "Answer the following question using only the provided context. Be concise and direct.\n\n",
                "Context:\n{context}\n\n",
                "Question: {query}\n\n",
                "Answer:"
            ).to_string(),
        );

        // Detailed template with reasoning
        templates.insert(
            "detailed".to_string(),
            concat!(
                "You are an expert assistant. Carefully analyze the provided context and give a ",
                "comprehensive answer to the question. Explain your reasoning step by step.\n\n",
                "{citation_instruction}\n\n",
                "Context:\n{context}\n\n",
                "Question: {query}\n\n",
                "Detailed Answer:"
            ).to_string(),
        );

        // Recipe-specific template
        templates.insert(
            "recipe".to_string(),
            concat!(
                "You are a culinary expert. Using the recipe information provided, answer the question ",
                "about cooking procedures, ingredients, or techniques. Be specific and helpful.\n\n",
                "{citation_instruction}\n\n",
                "Recipe Context:\n{context}\n\n",
                "Question: {query}\n\n",
                "Answer:"
            ).to_string(),
        );

        // Chat template (for conversational mode)
        templates.insert(
            "chat".to_string(),
            concat!(
                "You are a helpful AI assistant engaged in a conversation. Use the provided context ",
                "to inform your response, but maintain a natural conversational tone.\n\n",
                "{citation_instruction}\n\n",
                "Relevant Information:\n{context}\n\n",
                "User: {query}\n\n",
                "Assistant:"
            ).to_string(),
        );

        // Simple template for small models (0.5B-1.5B)
        templates.insert(
            "simple".to_string(),
            concat!(
                "Answer based on this info:\n\n",
                "{context}\n\n",
                "Q: {query}\n",
                "A:"
            ).to_string(),
        );

        // Qwen chat template (for Qwen models)
        templates.insert(
            "qwen".to_string(),
            concat!(
                "<|im_start|>system\n",
                "You are a helpful assistant. Answer questions based on the provided context.\n",
                "<|im_end|>\n",
                "<|im_start|>user\n",
                "Context:\n{context}\n\n",
                "Question: {query}\n",
                "<|im_end|>\n",
                "<|im_start|>assistant\n"
            ).to_string(),
        );

        Self { templates }
    }
}

impl PromptTemplates {
    /// Create a new empty template collection
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
        }
    }

    /// Get a template by name, falling back to "default" if not found
    pub fn get(&self, name: &str) -> &str {
        self.templates
            .get(name)
            .or_else(|| self.templates.get("default"))
            .map(|s| s.as_str())
            .unwrap_or("")
    }

    /// Register a custom template
    pub fn register(&mut self, name: &str, template: &str) {
        self.templates.insert(name.to_string(), template.to_string());
    }

    /// Check if a template exists
    pub fn contains(&self, name: &str) -> bool {
        self.templates.contains_key(name)
    }

    /// List all available template names
    pub fn names(&self) -> Vec<&str> {
        self.templates.keys().map(|s| s.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_templates() {
        let templates = PromptTemplates::default();

        assert!(templates.contains("default"));
        assert!(templates.contains("concise"));
        assert!(templates.contains("detailed"));
        assert!(templates.contains("recipe"));
        assert!(templates.contains("chat"));
    }

    #[test]
    fn test_get_template() {
        let templates = PromptTemplates::default();

        let default = templates.get("default");
        assert!(default.contains("{context}"));
        assert!(default.contains("{query}"));
    }

    #[test]
    fn test_fallback_to_default() {
        let templates = PromptTemplates::default();

        // Non-existent template should return default
        let unknown = templates.get("nonexistent");
        let default = templates.get("default");
        assert_eq!(unknown, default);
    }

    #[test]
    fn test_custom_template() {
        let mut templates = PromptTemplates::default();

        templates.register("custom", "Custom: {query}");
        assert!(templates.contains("custom"));
        assert_eq!(templates.get("custom"), "Custom: {query}");
    }
}
