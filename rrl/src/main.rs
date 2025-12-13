use clap::{Parser, Subcommand};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use rrl::cli;

#[derive(Parser)]
#[command(name = "rrl")]
#[command(about = "RustRAGLab - A Rust framework for RAG-aware fine-tuning and evaluation", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Load and chunk documents from various sources
    Ingest {
        /// Input directory or file path
        #[arg(short, long)]
        input: String,

        /// Output directory for chunked documents
        #[arg(short, long)]
        output: String,

        /// Chunk size (default: 512)
        #[arg(long, default_value = "512")]
        chunk_size: usize,

        /// Chunk overlap (default: 50)
        #[arg(long, default_value = "50")]
        chunk_overlap: usize,
    },

    /// Compute embeddings and build indexes
    Embed {
        /// Input directory with chunked documents
        #[arg(short, long)]
        input: String,

        /// Output directory for embeddings and indexes
        #[arg(short, long)]
        output: String,

        /// Model name or path
        #[arg(short, long, default_value = "sentence-transformers/all-MiniLM-L6-v2")]
        model: String,

        /// Backend: token, mock, or onnx
        #[arg(short, long, default_value = "token")]
        backend: String,

        /// Model file path (for ONNX backend)
        #[arg(long)]
        model_path: Option<String>,

        /// Hardware backend: cpu, cuda, or metal
        #[arg(long, default_value = "cpu")]
        hardware: String,
    },

    /// Fine-tune BERT models with LoRA adapters
    Train {
        /// Training data path (JSONL or CSV)
        #[arg(short, long)]
        data: String,

        /// Output directory for model checkpoints
        #[arg(short, long)]
        output: String,

        /// Base model name or HuggingFace model ID
        #[arg(short, long, default_value = "bert-base-uncased")]
        model: String,

        /// Number of epochs
        #[arg(long, default_value = "3")]
        epochs: usize,

        /// Learning rate
        #[arg(long, default_value = "5e-5")]
        learning_rate: f64,

        /// Batch size
        #[arg(long, default_value = "32")]
        batch_size: usize,

        /// LoRA rank (0 to disable LoRA)
        #[arg(long, default_value = "8")]
        lora_rank: usize,

        /// LoRA alpha
        #[arg(long, default_value = "16")]
        lora_alpha: f32,

        /// Device: auto, cpu, cuda, or metal
        #[arg(long, default_value = "auto")]
        device: String,

        /// Maximum sequence length
        #[arg(long, default_value = "512")]
        max_seq_length: usize,

        /// Gradient accumulation steps
        #[arg(long, default_value = "1")]
        gradient_accumulation: usize,

        /// Warmup ratio (fraction of total steps)
        #[arg(long, default_value = "0.1")]
        warmup_ratio: f64,

        /// Validation data path (optional)
        #[arg(long)]
        val_data: Option<String>,

        /// Save checkpoint every N steps (0 to disable)
        #[arg(long, default_value = "500")]
        save_steps: usize,

        /// Log every N steps
        #[arg(long, default_value = "100")]
        logging_steps: usize,
    },

    /// Build retrieval indexes from chunks and embeddings
    Index {
        /// Input directory with chunks from ingest
        #[arg(short, long)]
        chunks: String,

        /// Input directory with embeddings from embed
        #[arg(short, long)]
        embeddings: String,

        /// Output directory for indexes
        #[arg(short, long)]
        output: String,

        /// Model name used for embeddings
        #[arg(short, long)]
        model: String,

        /// Backend: token, mock
        #[arg(short, long, default_value = "token")]
        backend: String,

        /// Build only specific index: hnsw, bm25, or both
        #[arg(long, default_value = "both")]
        index_type: String,
    },

    /// Query retrieval indexes
    Query {
        /// Index directory
        #[arg(short, long)]
        index: String,

        /// Query text
        #[arg(short, long)]
        query: String,

        /// Number of results to return
        #[arg(short, long, default_value = "5")]
        top_k: usize,

        /// Retriever type: hnsw, bm25, or hybrid
        #[arg(short, long, default_value = "hybrid")]
        retriever: String,

        /// Model name for embeddings (for HNSW/hybrid)
        #[arg(short, long, default_value = "token-embedder")]
        model: String,

        /// Backend: token, mock
        #[arg(short, long, default_value = "token")]
        backend: String,
    },

    /// Evaluate retrieval performance
    Eval {
        /// Test data path (JSONL with query and relevant docs)
        #[arg(short, long)]
        data: String,

        /// Index directory
        #[arg(short, long)]
        index: String,

        /// Model name for embeddings
        #[arg(short, long, default_value = "token-embedder")]
        model: String,

        /// Backend: token, mock
        #[arg(short, long, default_value = "token")]
        backend: String,

        /// Retriever type: hnsw, bm25, or hybrid
        #[arg(short, long, default_value = "hybrid")]
        retriever: String,

        /// Top-K for retrieval
        #[arg(short = 'k', long, default_value = "10")]
        top_k: usize,

        /// Output file for detailed results (optional)
        #[arg(long)]
        output: Option<String>,
    },

    /// Launch Axum server with streaming responses
    Serve {
        /// Index directory
        #[arg(short, long)]
        index: String,

        /// Model path
        #[arg(short, long)]
        model: String,

        /// Server address
        #[arg(long, default_value = "127.0.0.1:3000")]
        addr: String,
    },

    /// Evaluate multiple-choice accuracy on Recipe-MPR dataset
    EvalMc {
        /// Test data path (JSON file with query, options, answer)
        #[arg(short, long)]
        data: String,

        /// Base model path or HuggingFace model ID
        #[arg(short, long)]
        model: String,

        /// LoRA checkpoint path (optional, for fine-tuned model)
        #[arg(long)]
        checkpoint: Option<String>,

        /// LoRA rank (must match checkpoint)
        #[arg(long, default_value = "8")]
        lora_rank: usize,

        /// LoRA alpha (must match checkpoint)
        #[arg(long, default_value = "16")]
        lora_alpha: f32,

        /// Device: auto, cpu, cuda, or metal
        #[arg(long, default_value = "cpu")]
        device: String,

        /// Maximum sequence length
        #[arg(long, default_value = "512")]
        max_seq_length: usize,
    },

    /// Launch interactive TUI (Terminal User Interface)
    Tui,

    /// Launch training configuration TUI
    TrainUi,

    /// Run RAG (Retrieval-Augmented Generation) query
    #[cfg(feature = "training")]
    Rag {
        /// Index directory containing HNSW and/or BM25 indexes
        #[arg(short, long)]
        index: String,

        /// Query text (interactive mode if not provided)
        #[arg(short, long)]
        query: Option<String>,

        /// Generator model (HuggingFace ID or local path)
        #[arg(short, long, default_value = "Qwen/Qwen2.5-0.5B")]
        generator: String,

        /// Embedder model (HuggingFace ID or local path)
        #[arg(short, long, default_value = "bert-base-uncased")]
        embedder: String,

        /// Embedder LoRA checkpoint path
        #[arg(long)]
        embedder_checkpoint: Option<String>,

        /// Generator LoRA checkpoint path
        #[arg(long)]
        generator_checkpoint: Option<String>,

        /// Number of documents to retrieve
        #[arg(short = 'k', long, default_value = "5")]
        top_k: usize,

        /// Retriever type: dense, sparse, or hybrid
        #[arg(short, long, default_value = "hybrid")]
        retriever: String,

        /// Sampling temperature (0 = greedy)
        #[arg(long, default_value = "0.7")]
        temperature: f32,

        /// Maximum new tokens to generate
        #[arg(long, default_value = "512")]
        max_tokens: usize,

        /// Prompt template: default, concise, detailed, recipe, chat
        #[arg(long, default_value = "default")]
        template: String,

        /// Output format: text or json
        #[arg(long, default_value = "text")]
        format: String,

        /// Device: auto, cpu, cuda, or metal
        #[arg(long, default_value = "auto")]
        device: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing subscriber for logging
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "rrl=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Ingest {
            input,
            output,
            chunk_size,
            chunk_overlap,
        } => {
            cli::ingest(input, output, chunk_size, chunk_overlap).await?;
        }

        Commands::Embed {
            input,
            output,
            model,
            backend,
            model_path,
            hardware,
        } => {
            cli::embed(input, output, model, backend, model_path, hardware).await?;
        }

        Commands::Index {
            chunks,
            embeddings,
            output,
            model,
            backend,
            index_type,
        } => {
            cli::index(chunks, embeddings, output, model, backend, index_type).await?;
        }

        Commands::Query {
            index,
            query,
            top_k,
            retriever,
            model,
            backend,
        } => {
            cli::query(index, query, top_k, retriever, model, backend).await?;
        }

        Commands::Train {
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
        } => {
            cli::train(
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
            )
            .await?;
        }

        Commands::Eval {
            data,
            index,
            model,
            backend,
            retriever,
            top_k,
            output,
        } => {
            cli::eval(data, index, model, backend, retriever, top_k, output).await?;
        }

        Commands::Serve { index, model, addr } => {
            tracing::info!("Running serve command");
            tracing::info!("  Index: {}", index);
            tracing::info!("  Model: {}", model);
            tracing::info!("  Address: {}", addr);
            println!("Serve command not yet implemented");
        }

        Commands::EvalMc {
            data,
            model,
            checkpoint,
            lora_rank,
            lora_alpha,
            device,
            max_seq_length,
        } => {
            cli::eval_mc(
                data,
                model,
                checkpoint,
                lora_rank,
                lora_alpha,
                device,
                max_seq_length,
            )
            .await?;
        }

        Commands::Tui => {
            rrl::tui::run_tui()?;
        }

        #[cfg(feature = "training")]
        Commands::Rag {
            index,
            query,
            generator,
            embedder,
            embedder_checkpoint,
            generator_checkpoint,
            top_k,
            retriever,
            temperature,
            max_tokens,
            template,
            format,
            device,
        } => {
            cli::rag(
                index,
                query,
                generator,
                embedder,
                embedder_checkpoint,
                generator_checkpoint,
                top_k,
                retriever,
                temperature,
                max_tokens,
                template,
                format,
                device,
            )
            .await?;
        }

        Commands::TrainUi => {
            // Launch the training configuration TUI
            if let Some(config) = rrl::tui::run_training_tui()? {
                println!("\n========================================");
                println!("  Starting Training Pipeline");
                println!("========================================\n");
                println!("Configuration:");
                println!("  Model:         {}", config.model);
                println!("  Train Data:    {}", config.train_data);
                println!("  Val Data:      {}", config.val_data.as_deref().unwrap_or("None"));
                println!("  Test Data:     {}", config.test_data);
                println!("  Output:        {}", config.output_dir);
                println!("  Device:        {}", config.device);
                println!("  Epochs:        {}", config.epochs);
                println!("  Batch Size:    {}", config.batch_size);
                println!("  Learning Rate: {}", config.learning_rate);
                println!("  LoRA:          rank={}, alpha={}", config.lora_rank, config.lora_alpha);
                println!();

                // Step 1: Run training
                println!("Step 1/2: Training...\n");
                cli::train(
                    config.train_data.clone(),
                    config.output_dir.clone(),
                    config.model.clone(),
                    config.epochs,
                    config.learning_rate,
                    config.batch_size,
                    config.lora_rank,
                    config.lora_alpha,
                    config.device.clone(),
                    config.max_seq_length,
                    config.gradient_accumulation,
                    config.warmup_ratio,
                    config.val_data.clone(),
                    config.save_steps,
                    config.logging_steps,
                )
                .await?;

                // Step 2: Run evaluation on test set
                println!("\n========================================");
                println!("Step 2/2: Evaluating on test set...");
                println!("========================================\n");

                // Find the checkpoint file in the final directory
                let final_dir = format!("{}/final", config.output_dir);
                let checkpoint_path = std::fs::read_dir(&final_dir)
                    .ok()
                    .and_then(|entries| {
                        entries
                            .filter_map(|e| e.ok())
                            .find(|e| e.path().extension().map_or(false, |ext| ext == "safetensors"))
                            .map(|e| e.path().to_string_lossy().to_string())
                    })
                    .unwrap_or_else(|| format!("{}/checkpoint.safetensors", final_dir));
                cli::eval_mc(
                    config.test_data.clone(),
                    config.model.clone(),
                    Some(checkpoint_path),
                    config.lora_rank,
                    config.lora_alpha,
                    config.device.clone(),
                    config.max_seq_length,
                )
                .await?;

                println!("\n========================================");
                println!("  Training & Evaluation Complete!");
                println!("========================================");
                println!("Checkpoint saved to: {}", config.output_dir);
            } else {
                println!("Training cancelled by user");
            }
        }
    }

    Ok(())
}
