//! Command-line interface
//!
//! Provides CLI commands for ingest, embed, train, eval, and serve.

use crate::data::{Chunk, ChunkConfig, Chunker, MultiFormatLoader, OverlappingChunker};
use crate::embedding::{create_embedder, Embedder, Embedding, EmbeddingCache, EmbeddingConfig};
use crate::retrieval::{Bm25Retriever, HnswConfig, HnswRetriever, HybridRetriever, Retriever};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

#[cfg(feature = "onnx-backend")]
use crate::embedding::backends::create_onnx_embedder;

/// Execute the ingest command
pub async fn ingest(
    input: String,
    output: String,
    chunk_size: usize,
    chunk_overlap: usize,
) -> Result<()> {
    tracing::info!("Starting ingestion pipeline");
    tracing::info!("  Input: {}", input);
    tracing::info!("  Output: {}", output);
    tracing::info!("  Chunk size: {}", chunk_size);
    tracing::info!("  Chunk overlap: {}", chunk_overlap);

    // Create output directory if it doesn't exist
    let output_path = Path::new(&output);
    fs::create_dir_all(output_path)
        .context(format!("Failed to create output directory: {}", output))?;

    // Initialize loader and chunker
    let loader = MultiFormatLoader::new();
    let config = ChunkConfig {
        chunk_size,
        chunk_overlap,
    };
    let chunker = OverlappingChunker::new(config);

    // Load documents
    let input_path = Path::new(&input);
    let documents = if input_path.is_file() {
        tracing::info!("Loading single file: {:?}", input_path);
        vec![loader.load(input_path)?]
    } else if input_path.is_dir() {
        tracing::info!("Loading directory: {:?}", input_path);
        loader.load_directory(input_path)?
    } else {
        anyhow::bail!("Input path does not exist: {}", input);
    };

    let num_documents = documents.len();
    tracing::info!("Loaded {} documents", num_documents);

    // Process each document
    let mut total_chunks = 0;
    for document in documents {
        tracing::info!(
            "Processing document: {} ({})",
            document.id,
            document.source
        );

        // Chunk the document
        let chunks = chunker.chunk(&document)?;
        tracing::info!("  Created {} chunks", chunks.len());
        total_chunks += chunks.len();

        // Save chunks to output directory
        let doc_output_dir = output_path.join(&document.id);
        fs::create_dir_all(&doc_output_dir)?;

        // Save document metadata
        let doc_metadata_path = doc_output_dir.join("metadata.json");
        let doc_json = serde_json::to_string_pretty(&document)?;
        fs::write(doc_metadata_path, doc_json)?;

        // Save chunks
        let chunks_path = doc_output_dir.join("chunks.json");
        let chunks_json = serde_json::to_string_pretty(&chunks)?;
        fs::write(chunks_path, chunks_json)?;

        // Also save a summary file
        let summary_path = doc_output_dir.join("summary.txt");
        let summary = format!(
            "Document: {}\nSource: {}\nChunks: {}\nContent length: {} chars\n",
            document.id,
            document.source,
            chunks.len(),
            document.content.len()
        );
        fs::write(summary_path, summary)?;
    }

    tracing::info!("Ingestion complete!");
    tracing::info!("  Total documents: {}", num_documents);
    tracing::info!("  Total chunks: {}", total_chunks);
    tracing::info!("  Output directory: {}", output);

    println!("\nIngestion Summary:");
    println!("  Documents processed: {}", num_documents);
    println!("  Total chunks created: {}", total_chunks);
    println!("  Output directory: {}", output);

    Ok(())
}

/// Execute the embed command
pub async fn embed(
    input: String,
    output: String,
    model: String,
    backend: String,
    model_path: Option<String>,
    hardware: String,
) -> Result<()> {
    tracing::info!("Starting embedding pipeline");
    tracing::info!("  Input: {}", input);
    tracing::info!("  Output: {}", output);
    tracing::info!("  Model: {}", model);
    tracing::info!("  Backend: {}", backend);
    tracing::info!("  Hardware: {}", hardware);

    // Create output directory
    let output_path = Path::new(&output);
    fs::create_dir_all(output_path)
        .context(format!("Failed to create output directory: {}", output))?;

    // Initialize embedder
    let config = EmbeddingConfig {
        model_name: model.clone(),
        ..Default::default()
    };

    let dimension = 384; // Default dimension

    // Create embedder based on backend
    let embedder = if backend.starts_with("onnx") || model_path.is_some() {
        #[cfg(feature = "onnx-backend")]
        {
            let model_file = model_path.as_ref()
                .ok_or_else(|| anyhow::anyhow!("ONNX backend requires --model-path"))?;
            let model_file_path = Path::new(model_file);
            if !model_file_path.exists() {
                anyhow::bail!("Model file not found: {}", model_file);
            }
            tracing::info!("  Model path: {}", model_file);
            create_onnx_embedder(model_file_path, config, &hardware)?
        }
        #[cfg(not(feature = "onnx-backend"))]
        {
            anyhow::bail!("ONNX backend not enabled. Compile with --features onnx-backend");
        }
    } else {
        create_embedder(&backend, config, dimension)?
    };

    tracing::info!("  Using embedder: {} (dim={})", embedder.model_name(), embedder.dimension());

    // Initialize cache
    let cache_path = output_path.join("embeddings.db");
    let cache = EmbeddingCache::new(&cache_path, model.clone())?;

    // Load chunks from input directory
    let input_path = Path::new(&input);
    if !input_path.is_dir() {
        anyhow::bail!("Input must be a directory from 'rrl ingest' output");
    }

    let mut total_chunks = 0;
    let mut total_embeddings = 0;

    // Iterate through document directories
    for entry in fs::read_dir(input_path)? {
        let entry = entry?;
        let doc_dir = entry.path();

        if !doc_dir.is_dir() {
            continue;
        }

        let chunks_file = doc_dir.join("chunks.json");
        if !chunks_file.exists() {
            tracing::warn!("Skipping directory without chunks: {:?}", doc_dir);
            continue;
        }

        // Load chunks
        let chunks_json = fs::read_to_string(&chunks_file)?;
        let chunks: Vec<Chunk> = serde_json::from_str(&chunks_json)?;

        tracing::info!("Processing {} chunks from {:?}", chunks.len(), doc_dir.file_name());
        total_chunks += chunks.len();

        // Generate embeddings for each chunk
        let texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();

        // Use batch embedding with caching
        let mut embeddings = Vec::new();
        for text in &texts {
            let embedding = cache.get_or_compute(text, |t| embedder.embed(t))?;
            embeddings.push(embedding);
        }

        total_embeddings += embeddings.len();

        // Save embeddings
        let doc_id = doc_dir.file_name().unwrap().to_string_lossy();
        let embeddings_output = output_path.join(format!("{}_embeddings.json", doc_id));

        #[derive(serde::Serialize)]
        struct ChunkEmbedding {
            chunk_id: String,
            embedding: Embedding,
        }

        let chunk_embeddings: Vec<ChunkEmbedding> = chunks
            .iter()
            .zip(embeddings.iter())
            .map(|(chunk, emb)| ChunkEmbedding {
                chunk_id: chunk.id.clone(),
                embedding: emb.clone(),
            })
            .collect();

        let output_json = serde_json::to_string_pretty(&chunk_embeddings)?;
        fs::write(&embeddings_output, output_json)?;

        tracing::info!("  Saved embeddings to {:?}", embeddings_output);
    }

    // Save cache statistics
    let stats = cache.stats()?;
    tracing::info!("Cache statistics:");
    tracing::info!("  Total entries: {}", stats.total_entries);
    tracing::info!("  Model entries: {}", stats.model_entries);

    tracing::info!("Embedding complete!");
    tracing::info!("  Total chunks processed: {}", total_chunks);
    tracing::info!("  Total embeddings generated: {}", total_embeddings);
    tracing::info!("  Output directory: {}", output);

    println!("\nEmbedding Summary:");
    println!("  Chunks processed: {}", total_chunks);
    println!("  Embeddings generated: {}", total_embeddings);
    println!("  Embedding dimension: {}", dimension);
    println!("  Cache entries: {}", stats.model_entries);
    println!("  Output directory: {}", output);

    Ok(())
}

/// Execute the index command - build retrieval indexes
pub async fn index(
    chunks_dir: String,
    embeddings_dir: String,
    output_dir: String,
    model: String,
    backend: String,
    index_type: String,
) -> Result<()> {
    tracing::info!("Starting index building");
    tracing::info!("  Chunks: {}", chunks_dir);
    tracing::info!("  Embeddings: {}", embeddings_dir);
    tracing::info!("  Output: {}", output_dir);
    tracing::info!("  Model: {}", model);
    tracing::info!("  Index type: {}", index_type);

    let output_path = Path::new(&output_dir);
    fs::create_dir_all(output_path)?;

    // Load all chunks
    tracing::info!("Loading chunks...");
    let mut all_chunks = Vec::new();
    let chunks_path = Path::new(&chunks_dir);

    for entry in fs::read_dir(chunks_path)? {
        let entry = entry?;
        let doc_dir = entry.path();

        if !doc_dir.is_dir() {
            continue;
        }

        let chunks_file = doc_dir.join("chunks.json");
        if !chunks_file.exists() {
            continue;
        }

        let chunks_json = fs::read_to_string(&chunks_file)?;
        let chunks: Vec<Chunk> = serde_json::from_str(&chunks_json)?;
        all_chunks.extend(chunks);
    }

    tracing::info!("Loaded {} total chunks", all_chunks.len());

    // Build BM25 index if requested
    if index_type == "bm25" || index_type == "both" {
        tracing::info!("Building BM25 index...");
        let bm25_dir = output_path.join("bm25");
        let bm25 = Bm25Retriever::build(all_chunks.clone(), &bm25_dir)?;
        tracing::info!("BM25 index built: {} chunks indexed", bm25.metadata().num_chunks);
    }

    // Build HNSW index if requested
    if index_type == "hnsw" || index_type == "both" {
        tracing::info!("Building HNSW index...");

        // Load embeddings
        tracing::info!("Loading embeddings...");
        let mut embeddings_map: HashMap<String, Embedding> = HashMap::new();
        let embeddings_path = Path::new(&embeddings_dir);

        for entry in fs::read_dir(embeddings_path)? {
            let entry = entry?;
            let file_path = entry.path();

            if !file_path.is_file() || file_path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }

            if file_path.file_name().and_then(|s| s.to_str()) == Some("embeddings.db") {
                continue;
            }

            let embeddings_json = fs::read_to_string(&file_path)?;

            #[derive(serde::Deserialize)]
            struct ChunkEmbedding {
                chunk_id: String,
                embedding: Embedding,
            }

            let chunk_embeddings: Vec<ChunkEmbedding> = serde_json::from_str(&embeddings_json)?;
            for ce in chunk_embeddings {
                embeddings_map.insert(ce.chunk_id, ce.embedding);
            }
        }

        tracing::info!("Loaded {} embeddings", embeddings_map.len());

        // Match chunks with embeddings
        let mut matched_chunks = Vec::new();
        let mut matched_embeddings = Vec::new();

        for chunk in &all_chunks {
            if let Some(embedding) = embeddings_map.get(&chunk.id) {
                matched_chunks.push(chunk.clone());
                matched_embeddings.push(embedding.clone());
            }
        }

        tracing::info!("Matched {} chunks with embeddings", matched_chunks.len());

        // Create embedder for query-time use
        let config = EmbeddingConfig {
            model_name: model.clone(),
            ..Default::default()
        };
        let dimension = if !matched_embeddings.is_empty() {
            matched_embeddings[0].len()
        } else {
            384
        };
        let embedder = create_embedder(&backend, config, dimension)?;

        // Build HNSW index
        let hnsw_dir = output_path.join("hnsw");
        let hnsw = HnswRetriever::build(
            matched_chunks,
            matched_embeddings,
            embedder,
            HnswConfig::default(),
        )?;

        hnsw.save(&hnsw_dir)?;
        tracing::info!("HNSW index built: {} chunks indexed", hnsw.metadata().num_chunks);
    }

    println!("\nIndex Building Summary:");
    println!("  Total chunks: {}", all_chunks.len());
    println!("  Index type: {}", index_type);
    println!("  Output directory: {}", output_dir);

    Ok(())
}

/// Execute the query command - search the indexes
pub async fn query(
    index_dir: String,
    query: String,
    top_k: usize,
    retriever_type: String,
    model: String,
    backend: String,
) -> Result<()> {
    tracing::info!("Starting query");
    tracing::info!("  Index: {}", index_dir);
    tracing::info!("  Query: {}", query);
    tracing::info!("  Top-k: {}", top_k);
    tracing::info!("  Retriever: {}", retriever_type);

    let index_path = Path::new(&index_dir);

    // Create embedder
    let config = EmbeddingConfig {
        model_name: model.clone(),
        ..Default::default()
    };
    let embedder = create_embedder(&backend, config, 384)?;

    // Load and use the appropriate retriever
    let retriever: Arc<dyn Retriever> = match retriever_type.as_str() {
        "hnsw" => {
            let hnsw_dir = index_path.join("hnsw");
            Arc::new(HnswRetriever::load(&hnsw_dir, embedder)?)
        }
        "bm25" => {
            let bm25_dir = index_path.join("bm25");
            Arc::new(Bm25Retriever::load(&bm25_dir)?)
        }
        "hybrid" => {
            // Load both retrievers
            let hnsw_dir = index_path.join("hnsw");
            let bm25_dir = index_path.join("bm25");

            let hnsw: Arc<dyn Retriever> = Arc::new(HnswRetriever::load(&hnsw_dir, embedder)?);
            let bm25: Arc<dyn Retriever> = Arc::new(Bm25Retriever::load(&bm25_dir)?);

            Arc::new(HybridRetriever::new(vec![hnsw, bm25]))
        }
        _ => anyhow::bail!("Unknown retriever type: {}", retriever_type),
    };

    // Perform the query
    let results = retriever.retrieve(&query, top_k)?;

    // Display results
    println!("\nQuery: {}", query);
    println!("Retriever: {}", retriever_type);
    println!("Found {} results:\n", results.len());

    for result in &results {
        println!("Rank {}: {} (score: {:.4})", result.rank, result.chunk_id, result.score);
        println!("  Document: {}", result.chunk.document_id);
        println!("  Content: {}", result.chunk.content.chars().take(200).collect::<String>());
        if result.chunk.content.len() > 200 {
            println!("  ...");
        }
        println!();
    }

    Ok(())
}
