//! Command-line interface
//!
//! Provides CLI commands for ingest, embed, train, eval, and serve.
pub mod infer;

use crate::data::{Chunk, ChunkConfig, Chunker, MultiFormatLoader, OverlappingChunker};
use crate::embedding::{create_embedder, Embedding, EmbeddingCache, EmbeddingConfig};
use crate::evaluation::{QueryResult, RetrievalEvaluator};
use crate::retrieval::{Bm25Retriever, HnswConfig, HnswRetriever, HybridRetriever, Retriever};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;
pub use infer::InferArgs;

#[cfg(feature = "training")]
use crate::training::{
    select_device, DatasetConfig, DevicePreference, LoraConfig, TrainingConfig, TrainingDataset,
};


#[cfg(feature = "training")]
use crate::training::TokenizerWrapper;

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
    let _ = (model_path, hardware); // Unused parameters kept for CLI compatibility

    // Create embedder based on backend
    let embedder = create_embedder(&backend, config, dimension)?;

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

/// Execute the train command - fine-tune a model
#[allow(clippy::too_many_arguments)]
pub async fn train(
    data: String,
    output: String,
    model: String,
    epochs: usize,
    learning_rate: f64,
    batch_size: usize,
    lora_rank: usize,
    lora_alpha: f32,
    device: String,
    max_seq_length: usize,
    gradient_accumulation: usize,
    warmup_ratio: f64,
    val_data: Option<String>,
    save_steps: usize,
    logging_steps: usize,
    gradient_checkpointing: bool,
    checkpoint_segment_size: usize,
) -> Result<()> {
    #[cfg(not(feature = "training"))]
    {
        let _ = (
            data,
            output,
            model,
            epochs,
            learning_rate,
            batch_size,
            lora_rank,
            lora_alpha,
            device,
            max_seq_length,
            gradient_accumulation,
            warmup_ratio,
            val_data,
            save_steps,
            logging_steps,
            gradient_checkpointing,
            checkpoint_segment_size,
        );
        anyhow::bail!(
            "Training feature not enabled. Compile with: cargo build --features training"
        );
    }

    #[cfg(feature = "training")]
    {
        tracing::info!("Starting fine-tuning pipeline with LoRA");
        tracing::info!("  Data: {}", data);
        tracing::info!("  Output: {}", output);
        tracing::info!("  Model: {}", model);
        tracing::info!("  Epochs: {}", epochs);
        tracing::info!("  Learning rate: {}", learning_rate);
        tracing::info!("  Batch size: {}", batch_size);
        tracing::info!("  LoRA rank: {}", lora_rank);
        tracing::info!("  LoRA alpha: {}", lora_alpha);
        tracing::info!("  Device: {}", device);
        tracing::info!("  Max sequence length: {}", max_seq_length);
        tracing::info!("  Gradient accumulation: {}", gradient_accumulation);
        tracing::info!("  Warmup ratio: {}", warmup_ratio);
        tracing::info!("  Save steps: {}", save_steps);
        tracing::info!("  Logging steps: {}", logging_steps);
        tracing::info!("  Gradient checkpointing: {}", if gradient_checkpointing { "enabled" } else { "disabled" });
        if gradient_checkpointing {
            tracing::info!("  Checkpoint segment size: {}", checkpoint_segment_size);
        }

        // Create output directory
        let output_path = Path::new(&output);
        fs::create_dir_all(output_path)
            .context(format!("Failed to create output directory: {}", output))?;

        // Select device - parse device string (e.g., "cuda:0", "metal:1", "cpu", "auto")
        let device_pref: DevicePreference = device.parse()?;
        let candle_device = select_device(device_pref)?;
        tracing::info!("Using device: {:?}", candle_device);

        // Load training dataset
        tracing::info!("Loading training data from: {}", data);
        let dataset_config = DatasetConfig::default();
        let train_dataset = TrainingDataset::load(&data, dataset_config.clone())?;
        tracing::info!("Loaded {} training examples", train_dataset.len());

        // Show dataset statistics
        let stats = train_dataset.stats();
        tracing::info!("Dataset stats: {}", stats);

        // Load validation dataset if provided
        let val_dataset = if let Some(ref val_path) = val_data {
            tracing::info!("Loading validation data from: {}", val_path);
            let val_ds = TrainingDataset::load(val_path, dataset_config)?;
            tracing::info!("Loaded {} validation examples", val_ds.len());
            Some(val_ds)
        } else {
            None
        };

        // Create training config
        let training_config = TrainingConfig {
            batch_size,
            num_epochs: epochs,
            learning_rate,
            warmup_ratio,
            gradient_accumulation_steps: gradient_accumulation,
            save_steps,
            logging_steps,
            output_dir: output.clone(),
            max_seq_length,
            gradient_checkpointing,
            checkpoint_segment_size,
            ..Default::default()
        };

        // Create trainer FIRST (it has the VarMap for trainable parameters)
        let mut trainer = crate::training::Trainer::new(training_config, candle_device.clone());

        // Create LoRA config
        let lora_config = LoraConfig::new(lora_rank, lora_alpha);
        tracing::info!(
            "LoRA config: rank={}, alpha={}, scaling={}",
            lora_config.rank,
            lora_config.alpha,
            lora_config.scaling()
        );

        // Load model with LoRA using trainer's VarMap
        // This registers LoRA parameters as trainable
        // Supports both encoder (BERT, RoBERTa) and decoder (Qwen2, LLaMA, Mistral) models
        tracing::info!("Loading model with LoRA from: {}", model);
        let lora_model = crate::training::load_any_model_lora(&model, &lora_config, trainer.var_map(), &candle_device)?;

        tracing::info!(
            "Model loaded: {} trainable params / ~{} total params ({:.4}%)",
            lora_model.num_trainable_params(),
            lora_model.num_total_params(),
            lora_model.num_trainable_params() as f64 / lora_model.num_total_params() as f64 * 100.0
        );

        // Load tokenizer
        tracing::info!("Loading tokenizer...");
        let tokenizer = TokenizerWrapper::from_pretrained(&model)?
            .with_max_length(max_seq_length);

        // Train with progress callback
        let result = trainer.train(
            &*lora_model,
            &tokenizer,
            &train_dataset,
            val_dataset.as_ref(),
            Some(Box::new(|metrics| {
                println!(
                    "Step {} | Epoch {} | Loss: {:.4} | LR: {:.2e}",
                    metrics.global_step, metrics.epoch, metrics.train_loss, metrics.learning_rate
                );
            })),
        )?;

        println!("\n========================================");
        println!("Training Complete!");
        println!("========================================");
        println!("  Final loss: {:.4}", result.metrics.train_loss);
        println!("  Total steps: {}", result.metrics.global_step);
        println!("  Epochs completed: {}", result.metrics.epoch);
        println!("  Samples/sec: {:.1}", result.metrics.samples_per_second);
        if let Some(path) = &result.checkpoint_path {
            println!("  Final checkpoint: {}", path);
        }
        println!("  Output directory: {}", output);
        println!("========================================");

        Ok(())
    }
}

/// Execute the eval command - evaluate retrieval performance
#[allow(clippy::too_many_arguments)]
pub async fn eval(
    data: String,
    index: String,
    model: String,
    backend: String,
    retriever_type: String,
    top_k: usize,
    output: Option<String>,
) -> Result<()> {
    tracing::info!("Starting retrieval evaluation");
    tracing::info!("  Test data: {}", data);
    tracing::info!("  Index: {}", index);
    tracing::info!("  Model: {}", model);
    tracing::info!("  Backend: {}", backend);
    tracing::info!("  Retriever: {}", retriever_type);
    tracing::info!("  Top-K: {}", top_k);

    // Load test data (JSONL format with query and relevant_docs fields)
    let data_path = Path::new(&data);
    if !data_path.exists() {
        anyhow::bail!("Test data file not found: {}", data);
    }

    #[derive(serde::Deserialize)]
    struct EvalExample {
        query: String,
        relevant_docs: Vec<String>,
        #[serde(default)]
        query_id: Option<String>,
    }

    let file_content = fs::read_to_string(data_path)?;
    let examples: Vec<EvalExample> = file_content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
        .map(|(i, line)| {
            let mut example: EvalExample =
                serde_json::from_str(line).context(format!("Failed to parse line {}", i + 1))?;
            if example.query_id.is_none() {
                example.query_id = Some(format!("q{}", i + 1));
            }
            Ok(example)
        })
        .collect::<Result<Vec<_>>>()?;

    tracing::info!("Loaded {} test examples", examples.len());

    // Create embedder
    let config = EmbeddingConfig {
        model_name: model.clone(),
        ..Default::default()
    };
    let embedder = create_embedder(&backend, config, 384)?;

    // Load retriever
    let index_path = Path::new(&index);
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
            let hnsw_dir = index_path.join("hnsw");
            let bm25_dir = index_path.join("bm25");
            let hnsw: Arc<dyn Retriever> = Arc::new(HnswRetriever::load(&hnsw_dir, embedder)?);
            let bm25: Arc<dyn Retriever> = Arc::new(Bm25Retriever::load(&bm25_dir)?);
            Arc::new(HybridRetriever::new(vec![hnsw, bm25]))
        }
        _ => anyhow::bail!("Unknown retriever type: {}", retriever_type),
    };

    // Run evaluation
    tracing::info!("Running evaluation on {} queries...", examples.len());
    let mut query_results = Vec::new();

    for example in &examples {
        let results = retriever.retrieve(&example.query, top_k)?;
        let retrieved: Vec<String> = results.iter().map(|r| r.chunk_id.clone()).collect();

        query_results.push(QueryResult::new(
            example.query_id.as_ref().unwrap(),
            retrieved,
            example.relevant_docs.clone(),
        ));
    }

    // Compute metrics
    let evaluator = RetrievalEvaluator::with_k_values(vec![1, 5, 10, top_k]);
    let metrics = evaluator.evaluate(&query_results);

    // Display results
    println!("\n{}", metrics);

    // Save detailed results if output path provided
    if let Some(output_path) = output {
        #[derive(serde::Serialize)]
        struct DetailedResult {
            query_id: String,
            query: String,
            retrieved: Vec<String>,
            relevant: Vec<String>,
            recall_at_k: f64,
            reciprocal_rank: f64,
        }

        let detailed: Vec<DetailedResult> = examples
            .iter()
            .zip(query_results.iter())
            .map(|(ex, qr)| DetailedResult {
                query_id: qr.query_id.clone(),
                query: ex.query.clone(),
                retrieved: qr.retrieved.clone(),
                relevant: qr.relevant.iter().cloned().collect(),
                recall_at_k: qr.recall_at_k(top_k),
                reciprocal_rank: qr.reciprocal_rank(),
            })
            .collect();

        let output_json = serde_json::to_string_pretty(&detailed)?;
        fs::write(&output_path, output_json)?;
        tracing::info!("Saved detailed results to: {}", output_path);
    }

    println!("\nEvaluation Summary:");
    println!("  Queries evaluated: {}", metrics.num_queries);
    println!("  MRR: {:.4}", metrics.mrr);
    println!("  MAP: {:.4}", metrics.map);
    println!("  Hit Rate: {:.4}", metrics.hit_rate);

    Ok(())
}

/// Execute the eval-mc command - evaluate multiple-choice accuracy
#[allow(clippy::too_many_arguments)]
pub async fn eval_mc(
    data: String,
    model: String,
    checkpoint: Option<String>,
    lora_rank: usize,
    lora_alpha: f32,
    device: String,
    max_seq_length: usize,
) -> Result<()> {
    #[cfg(not(feature = "training"))]
    {
        let _ = (data, model, checkpoint, lora_rank, lora_alpha, device, max_seq_length);
        anyhow::bail!(
            "Training feature not enabled. Compile with: cargo build --features training"
        );
    }

    #[cfg(feature = "training")]
    {
        use crate::training::{
            evaluate_multiple_choice, load_recipe_mpr_examples, LoraConfig,
        };
        use candle_nn::VarMap;

        println!("\n========================================");
        println!("Multiple-Choice Evaluation");
        println!("========================================");
        tracing::info!("Starting multiple-choice evaluation");
        tracing::info!("  Data: {}", data);
        tracing::info!("  Model: {}", model);
        tracing::info!("  Checkpoint: {:?}", checkpoint);
        tracing::info!("  LoRA rank: {}", lora_rank);
        tracing::info!("  LoRA alpha: {}", lora_alpha);
        tracing::info!("  Device: {}", device);

        // Select device
        let device_pref: DevicePreference = device.parse()?;
        let candle_device = select_device(device_pref)?;
        tracing::info!("Using device: {:?}", candle_device);

        // Load test data
        tracing::info!("Loading evaluation data from: {}", data);
        let examples = load_recipe_mpr_examples(&data)?;
        tracing::info!("Loaded {} examples for evaluation", examples.len());

        // Create LoRA config
        let lora_config = LoraConfig::new(lora_rank, lora_alpha);

        // Load model using universal loader (supports BERT, RoBERTa, Qwen2, LLaMA, Mistral)
        let var_map = VarMap::new();
        let mut lora_model = crate::training::load_any_model_lora(&model, &lora_config, &var_map, &candle_device)?;

        // Load checkpoint if provided
        if let Some(ref ckpt_path) = checkpoint {
            tracing::info!("Loading LoRA checkpoint from: {}", ckpt_path);
            lora_model.load_lora_checkpoint(std::path::Path::new(ckpt_path))?;
            println!("  Status: Fine-tuned model (checkpoint loaded)");
        } else {
            println!("  Status: Baseline model (no checkpoint)");
        }

        tracing::info!(
            "Model loaded: {} trainable params",
            lora_model.num_trainable_params()
        );

        // Load tokenizer
        tracing::info!("Loading tokenizer...");
        let tokenizer = TokenizerWrapper::from_pretrained(&model)?
            .with_max_length(max_seq_length);

        // Run evaluation
        println!("\nRunning evaluation on {} examples...", examples.len());
        let result = evaluate_multiple_choice(&*lora_model, &tokenizer, &examples, &candle_device)?;

        // Display results
        println!("\n========================================");
        println!("Results");
        println!("========================================");
        println!("{}", result);
        println!("========================================\n");

        Ok(())
    }
}

/// Execute the eval-msmarco command - evaluate on MS MARCO v1.1
#[allow(clippy::too_many_arguments)]
pub async fn eval_msmarco(
    data: String,
    model: String,
    checkpoint: Option<String>,
    sample: Option<usize>,
    lora_rank: usize,
    lora_alpha: f32,
    device: String,
    max_seq_length: usize,
    json_progress: bool,
) -> Result<()> {
    #[cfg(not(feature = "training"))]
    {
        let _ = (data, model, checkpoint, sample, lora_rank, lora_alpha, device, max_seq_length, json_progress);
        anyhow::bail!(
            "Training feature not enabled. Compile with: cargo build --features training"
        );
    }

    #[cfg(feature = "training")]
    {
        use crate::evaluation::msmarco::{
            load_msmarco_examples, MsMarcoEvaluator, MsMarcoMetrics,
            cosine_similarity, rank_by_similarity,
        };
        use crate::training::LoraConfig;
        use candle_nn::VarMap;
        use std::time::Instant;

        if !json_progress {
            println!("\n========================================");
            println!("MS MARCO v1.1 Evaluation");
            println!("========================================");
        }

        tracing::info!("Starting MS MARCO evaluation");
        tracing::info!("  Data: {}", data);
        tracing::info!("  Model: {}", model);
        tracing::info!("  Checkpoint: {:?}", checkpoint);
        tracing::info!("  Sample: {:?}", sample);
        tracing::info!("  LoRA rank: {}", lora_rank);
        tracing::info!("  LoRA alpha: {}", lora_alpha);
        tracing::info!("  Device: {}", device);
        tracing::info!("  JSON progress: {}", json_progress);

        // Select device
        let device_pref: DevicePreference = device.parse()?;
        let candle_device = select_device(device_pref)?;
        tracing::info!("Using device: {:?}", candle_device);

        // Load MS MARCO examples
        let data_path = std::path::Path::new(&data);
        tracing::info!("Loading MS MARCO data from: {}", data);
        let examples = load_msmarco_examples(data_path, sample)?;
        let total_queries = examples.len();
        tracing::info!("Loaded {} examples", total_queries);

        if !json_progress {
            println!("  Queries to evaluate: {}", total_queries);
            if let Some(s) = sample {
                println!("  (sampled from full dataset, limit: {})", s);
            }
        }

        // Create LoRA config
        let lora_config = LoraConfig::new(lora_rank, lora_alpha);

        // Load model using universal loader (supports BERT, RoBERTa, Qwen2, LLaMA, Mistral)
        let var_map = VarMap::new();
        let mut lora_model = crate::training::load_any_model_lora(&model, &lora_config, &var_map, &candle_device)?;

        // Load checkpoint if provided
        if let Some(ref ckpt_path) = checkpoint {
            tracing::info!("Loading LoRA checkpoint from: {}", ckpt_path);
            lora_model.load_lora_checkpoint(std::path::Path::new(ckpt_path))?;
            if !json_progress {
                println!("  Status: Fine-tuned model (checkpoint loaded)");
            }
        } else if !json_progress {
            println!("  Status: Baseline model (no checkpoint)");
        }

        tracing::info!(
            "Model loaded: {} trainable params",
            lora_model.num_trainable_params()
        );

        // Load tokenizer
        tracing::info!("Loading tokenizer...");
        let tokenizer = TokenizerWrapper::from_pretrained(&model)?
            .with_max_length(max_seq_length);

        // Create evaluator
        let evaluator = MsMarcoEvaluator::new(json_progress);

        // Run evaluation
        if !json_progress {
            println!("\nRunning evaluation on {} queries...", total_queries);
        }

        let start_time = Instant::now();
        let mut query_results = Vec::new();
        let mut running_rr_sum = 0.0;

        for (idx, example) in examples.iter().enumerate() {
            // Tokenize query
            let query_encoding = tokenizer.encode(&example.query, true)?;
            let query_input_ids = candle_core::Tensor::new(
                query_encoding.input_ids.as_slice(),
                &candle_device,
            )?.unsqueeze(0)?;
            let query_attention_mask = candle_core::Tensor::new(
                query_encoding.attention_mask.as_slice(),
                &candle_device,
            )?.unsqueeze(0)?;

            // Get query embedding
            let query_embedding = lora_model.forward(&query_input_ids, &query_attention_mask)?;
            let query_vec: Vec<f32> = query_embedding.squeeze(0)?.to_vec1()?;

            // Compute passage embeddings and similarities
            let mut similarities = Vec::with_capacity(example.passages.len());

            for passage in &example.passages {
                let passage_encoding = tokenizer.encode(passage, true)?;
                let passage_input_ids = candle_core::Tensor::new(
                    passage_encoding.input_ids.as_slice(),
                    &candle_device,
                )?.unsqueeze(0)?;
                let passage_attention_mask = candle_core::Tensor::new(
                    passage_encoding.attention_mask.as_slice(),
                    &candle_device,
                )?.unsqueeze(0)?;

                let passage_embedding = lora_model.forward(&passage_input_ids, &passage_attention_mask)?;
                let passage_vec: Vec<f32> = passage_embedding.squeeze(0)?.to_vec1()?;

                let sim = cosine_similarity(&query_vec, &passage_vec);
                similarities.push(sim);
            }

            // Rank passages by similarity
            let ranked_indices = rank_by_similarity(&similarities);
            let relevant_indices = example.relevant_indices();

            // Create query result
            let query_result = MsMarcoEvaluator::create_query_result(
                format!("q{}", idx),
                ranked_indices,
                relevant_indices,
            );

            // Update running MRR
            running_rr_sum += query_result.reciprocal_rank();
            let current_mrr = running_rr_sum / (idx + 1) as f64;

            query_results.push(query_result);

            // Report progress
            let processed = idx + 1;
            if json_progress {
                evaluator.report_progress(processed, total_queries, current_mrr, start_time);
            } else if processed % 100 == 0 || processed == total_queries {
                let elapsed = start_time.elapsed().as_secs_f64();
                let qps = processed as f64 / elapsed;
                let eta = (total_queries - processed) as f64 / qps;
                println!(
                    "  Progress: {}/{} queries ({:.1}%), MRR@10: {:.4}, ETA: {:.0}s",
                    processed,
                    total_queries,
                    processed as f64 / total_queries as f64 * 100.0,
                    current_mrr,
                    eta
                );
            }
        }

        // Compute final metrics
        let elapsed_seconds = start_time.elapsed().as_secs_f64();
        let retrieval_metrics = evaluator.evaluate(&query_results);
        let metrics = MsMarcoMetrics::from_retrieval_metrics(&retrieval_metrics, elapsed_seconds);

        // Output results
        if json_progress {
            metrics.print_json();
        } else {
            println!("\n========================================");
            println!("Results");
            println!("========================================");
            println!("{}", metrics);
            println!("========================================\n");
        }

        Ok(())
    }
}

/// Execute the RAG (Retrieval-Augmented Generation) command
#[cfg(feature = "training")]
pub async fn rag(
    index: String,
    query: Option<String>,
    generator_model: String,
    embedder_model: String,
    embedder_checkpoint: Option<String>,
    generator_checkpoint: Option<String>,
    top_k: usize,
    retriever_type: String,
    temperature: f32,
    max_tokens: usize,
    template: String,
    format: String,
    device: String,
    dtype: String,
) -> Result<()> {
    use crate::embedding::backends::{CandleBertConfig, CandleBertEmbedder};
    use crate::rag::{
        Generator, GeneratorConfig, RagConfig, RagPipelineBuilder,
        RetrievalStrategy, SamplingParams,
    };

    tracing::debug!("Starting RAG pipeline");
    tracing::debug!("  Index: {}", index);
    tracing::debug!("  Generator: {}", generator_model);
    tracing::debug!("  Embedder: {}", embedder_model);
    tracing::debug!("  Top-K: {}", top_k);
    tracing::debug!("  Retriever: {}", retriever_type);
    tracing::debug!("  Device: {}", device);
    tracing::debug!("  Dtype: {}", dtype);

    // Parse device preference
    let device_pref: DevicePreference = device.parse()?;

    // Load embedder (silent)
    let mut embedder_config = CandleBertConfig::new(&embedder_model)
        .with_device(device_pref.clone());

    if let Some(ref ckpt) = embedder_checkpoint {
        embedder_config = embedder_config.with_lora_checkpoint(ckpt);
    }

    let embedder: Arc<dyn crate::embedding::Embedder> = Arc::new(CandleBertEmbedder::new(embedder_config)?);

    // Load retriever (silent)
    let index_path = Path::new(&index);
    let hnsw_dir = index_path.join("hnsw");
    let bm25_dir = index_path.join("bm25");

    let strategy = match retriever_type.as_str() {
        "hybrid" => RetrievalStrategy::Hybrid,
        "dense" => RetrievalStrategy::Dense,
        "sparse" => RetrievalStrategy::Sparse,
        _ => RetrievalStrategy::Hybrid,
    };

    let retriever: Arc<dyn Retriever> = match retriever_type.as_str() {
        "hybrid" => {
            if hnsw_dir.exists() && bm25_dir.exists() {
                let hnsw: Arc<dyn Retriever> = Arc::new(HnswRetriever::load(&hnsw_dir, embedder.clone())?);
                let bm25: Arc<dyn Retriever> = Arc::new(Bm25Retriever::load(&bm25_dir)?);
                Arc::new(HybridRetriever::new(vec![hnsw, bm25]))
            } else {
                anyhow::bail!("Hybrid retriever requires both HNSW and BM25 indexes");
            }
        }
        "dense" => {
            if hnsw_dir.exists() {
                Arc::new(HnswRetriever::load(&hnsw_dir, embedder.clone())?)
            } else {
                anyhow::bail!("Dense retriever requires HNSW index");
            }
        }
        "sparse" => {
            if bm25_dir.exists() {
                Arc::new(Bm25Retriever::load(&bm25_dir)?)
            } else {
                anyhow::bail!("Sparse retriever requires BM25 index");
            }
        }
        _ => anyhow::bail!("Unknown retriever type: {}", retriever_type),
    };

    // Load generator (silent)
    let mut generator_config = GeneratorConfig::new(&generator_model)
        .with_device(device_pref)
        .with_max_new_tokens(max_tokens)
        .with_dtype(&dtype);

    if let Some(ref ckpt) = generator_checkpoint {
        generator_config = generator_config.with_lora_checkpoint(ckpt);
    }

    let generator = Generator::new(generator_config)?;

    // Build pipeline (silent)
    let sampling_params = SamplingParams::default()
        .with_temperature(temperature)
        .with_max_new_tokens(max_tokens);

    let rag_config = RagConfig {
        retrieval_strategy: strategy,
        top_k,
        max_context_chars: 4000,
        include_citations: true,
        template_name: template,
        sampling_params,
    };

    let pipeline = RagPipelineBuilder::new()
        .embedder(embedder)
        .retriever(retriever)
        .generator(generator)
        .config(rag_config)
        .build()?;

    // Execute query or enter interactive mode
    if let Some(query_text) = query {
        // Single query mode
        execute_rag_query(&pipeline, &query_text, top_k, &format)?;
    } else {
        // Interactive mode
        println!("Entering interactive mode. Type 'exit' or 'quit' to stop.\n");

        let stdin = std::io::stdin();
        loop {
            print!("Query: ");
            std::io::Write::flush(&mut std::io::stdout())?;

            let mut input = String::new();
            stdin.read_line(&mut input)?;
            let input = input.trim();

            if input.is_empty() {
                continue;
            }

            if input == "exit" || input == "quit" {
                println!("Goodbye!");
                break;
            }

            execute_rag_query(&pipeline, input, top_k, &format)?;
            println!();
        }
    }

    Ok(())
}

#[cfg(feature = "training")]
fn execute_rag_query(
    pipeline: &crate::rag::RagPipeline,
    query_text: &str,
    top_k: usize,
    format: &str,
) -> Result<()> {
    use crate::rag::RagQuery;

    let query = RagQuery::new(query_text).with_top_k(top_k);
    let start = std::time::Instant::now();
    let response = pipeline.query(query)?;
    let total_time = start.elapsed().as_millis();

    match format {
        "json" => {
            let json = serde_json::to_string_pretty(&response)?;
            println!("{}", json);
        }
        _ => {
            println!("\n========================================");
            println!("Answer");
            println!("========================================");
            println!("{}", response.answer);

            println!("\n========================================");
            println!("Sources ({} documents)", response.sources.len());
            println!("========================================");
            for (i, source) in response.sources.iter().enumerate() {
                println!("[{}] {} (score: {:.4})", i + 1, source.document_id, source.score);
                println!("    {}", source.snippet);
            }

            println!("\n========================================");
            println!("Timing");
            println!("========================================");
            println!("  Retrieval: {}ms", response.retrieval_time_ms);
            println!("  Generation: {}ms", response.generation_time_ms);
            println!("  Total: {}ms", total_time);
            println!("  Tokens used: {}", response.tokens_used);
        }
    }

    Ok(())
}

/// Execute the infer command - run inference on a model
pub async fn infer(args: InferArgs) -> Result<()> {
    crate::cli::infer::run(args)
}

