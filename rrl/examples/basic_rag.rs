//! Basic RAG example
//!
//! This demonstrates the core RAG workflow:
//! - Document ingestion and chunking
//! - Embedding generation
//! - Index building
//! - Querying and retrieval

use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    println!("Basic RAG example");
    println!("This demonstrates document loading, chunking, embedding, and retrieval.");
    println!();
    println!("Usage:");
    println!("  # Ingest documents");
    println!("  rrl ingest --input ./docs --output ./data/chunks.json");
    println!();
    println!("  # Generate embeddings");
    println!("  rrl embed --input ./data/chunks.json --output ./data/embeddings.safetensors");
    println!();
    println!("  # Build index");
    println!("  rrl index build --embeddings ./data/embeddings.safetensors --output ./index");
    println!();
    println!("  # Query");
    println!("  rrl query --index ./index --query \"What is RAG?\"");

    Ok(())
}
