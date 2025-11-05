//! Document loaders for various file formats
//!
//! Supports PDF, Markdown, and plain text documents.

use crate::data::{Document, DocumentMetadata};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Trait for loading documents from various sources
pub trait DocumentLoader {
    /// Load a document from the given path
    fn load(&self, path: &Path) -> Result<Document>;

    /// Check if this loader can handle the given file extension
    fn can_load(&self, path: &Path) -> bool;
}

/// Text file loader
pub struct TextLoader;

impl DocumentLoader for TextLoader {
    fn load(&self, path: &Path) -> Result<Document> {
        let content = fs::read_to_string(path)
            .context(format!("Failed to read text file: {:?}", path))?;

        let metadata = fs::metadata(path)?;
        let file_size = metadata.len() as usize;

        let id = generate_document_id(path);
        let source = path.to_string_lossy().to_string();

        let doc_metadata = DocumentMetadata {
            file_path: Some(path.to_path_buf()),
            file_type: "txt".to_string(),
            size: Some(file_size),
            custom: HashMap::new(),
        };

        Ok(Document::new(id, source, content, doc_metadata))
    }

    fn can_load(&self, path: &Path) -> bool {
        if let Some(ext) = path.extension() {
            ext == "txt"
        } else {
            false
        }
    }
}

/// Markdown file loader
pub struct MarkdownLoader;

impl DocumentLoader for MarkdownLoader {
    fn load(&self, path: &Path) -> Result<Document> {
        let content = fs::read_to_string(path)
            .context(format!("Failed to read markdown file: {:?}", path))?;

        let metadata = fs::metadata(path)?;
        let file_size = metadata.len() as usize;

        let id = generate_document_id(path);
        let source = path.to_string_lossy().to_string();

        let doc_metadata = DocumentMetadata {
            file_path: Some(path.to_path_buf()),
            file_type: "md".to_string(),
            size: Some(file_size),
            custom: HashMap::new(),
        };

        Ok(Document::new(id, source, content, doc_metadata))
    }

    fn can_load(&self, path: &Path) -> bool {
        if let Some(ext) = path.extension() {
            matches!(ext.to_str(), Some("md") | Some("markdown"))
        } else {
            false
        }
    }
}

/// PDF file loader
pub struct PdfLoader;

impl DocumentLoader for PdfLoader {
    fn load(&self, _path: &Path) -> Result<Document> {
        // For now, we'll use a simple approach with pdf-extract crate
        // This will be added as a dependency
        #[cfg(feature = "pdf")]
        {
            use pdf_extract::extract_text;
            let content = extract_text(_path)
                .context(format!("Failed to extract text from PDF: {:?}", _path))?;

            let metadata = fs::metadata(_path)?;
            let file_size = metadata.len() as usize;

            let id = generate_document_id(_path);
            let source = _path.to_string_lossy().to_string();

            let doc_metadata = DocumentMetadata {
                file_path: Some(_path.to_path_buf()),
                file_type: "pdf".to_string(),
                size: Some(file_size),
                custom: HashMap::new(),
            };

            Ok(Document::new(id, source, content, doc_metadata))
        }

        #[cfg(not(feature = "pdf"))]
        {
            anyhow::bail!("PDF support not enabled. Compile with --features pdf")
        }
    }

    fn can_load(&self, path: &Path) -> bool {
        if let Some(ext) = path.extension() {
            ext == "pdf"
        } else {
            false
        }
    }
}

/// Multi-format document loader that delegates to specific loaders
pub struct MultiFormatLoader {
    loaders: Vec<Box<dyn DocumentLoader>>,
}

impl MultiFormatLoader {
    /// Create a new multi-format loader with all supported loaders
    pub fn new() -> Self {
        let loaders: Vec<Box<dyn DocumentLoader>> = vec![
            Box::new(TextLoader),
            Box::new(MarkdownLoader),
            Box::new(PdfLoader),
        ];

        Self { loaders }
    }

    /// Load a document, automatically selecting the appropriate loader
    pub fn load(&self, path: &Path) -> Result<Document> {
        for loader in &self.loaders {
            if loader.can_load(path) {
                return loader.load(path);
            }
        }

        anyhow::bail!("No loader found for file: {:?}", path)
    }

    /// Load all documents from a directory recursively
    pub fn load_directory(&self, dir_path: &Path) -> Result<Vec<Document>> {
        let mut documents = Vec::new();

        for entry in fs::read_dir(dir_path)
            .context(format!("Failed to read directory: {:?}", dir_path))?
        {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                // Try to load the file
                if let Ok(doc) = self.load(&path) {
                    documents.push(doc);
                } else {
                    tracing::warn!("Failed to load file: {:?}", path);
                }
            } else if path.is_dir() {
                // Recursively load from subdirectories
                let mut sub_docs = self.load_directory(&path)?;
                documents.append(&mut sub_docs);
            }
        }

        Ok(documents)
    }
}

impl Default for MultiFormatLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate a unique document ID based on file path
fn generate_document_id(path: &Path) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    path.to_string_lossy().hash(&mut hasher);
    format!("doc_{:x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_text_loader() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "Hello, world!").unwrap();

        let loader = TextLoader;
        let doc = loader.load(file.path()).unwrap();

        assert!(doc.content.contains("Hello, world!"));
        assert_eq!(doc.metadata.file_type, "txt");
    }

    #[test]
    fn test_markdown_loader() {
        let mut file = NamedTempFile::with_suffix(".md").unwrap();
        writeln!(file, "# Header\n\nContent").unwrap();

        let loader = MarkdownLoader;
        let doc = loader.load(file.path()).unwrap();

        assert!(doc.content.contains("# Header"));
        assert_eq!(doc.metadata.file_type, "md");
    }
}
