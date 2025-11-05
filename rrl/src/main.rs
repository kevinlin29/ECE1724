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

    /// Fine-tune LoRA adapters
    Train {
        /// Training data path
        #[arg(short, long)]
        data: String,

        /// Output directory for model checkpoints
        #[arg(short, long)]
        output: String,

        /// Base model name or path
        #[arg(short, long)]
        model: String,

        /// Number of epochs
        #[arg(long, default_value = "3")]
        epochs: usize,

        /// Learning rate
        #[arg(long, default_value = "1e-4")]
        learning_rate: f64,
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

    /// Evaluate retrieval/generation pipelines
    Eval {
        /// Test data path
        #[arg(short, long)]
        data: String,

        /// Index directory
        #[arg(short, long)]
        index: String,

        /// Model path (optional for retrieval-only eval)
        #[arg(short, long)]
        model: Option<String>,

        /// Evaluation mode: retrieval, generation, or both
        #[arg(long, default_value = "both")]
        mode: String,
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

    /// Launch interactive TUI (Terminal User Interface)
    Tui,
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
        } => {
            tracing::info!("Running train command");
            tracing::info!("  Data: {}", data);
            tracing::info!("  Output: {}", output);
            tracing::info!("  Model: {}", model);
            tracing::info!("  Epochs: {}", epochs);
            tracing::info!("  Learning rate: {}", learning_rate);
            println!("Train command not yet implemented");
        }

        Commands::Eval {
            data,
            index,
            model,
            mode,
        } => {
            tracing::info!("Running eval command");
            tracing::info!("  Data: {}", data);
            tracing::info!("  Index: {}", index);
            tracing::info!("  Model: {:?}", model);
            tracing::info!("  Mode: {}", mode);
            println!("Eval command not yet implemented");
        }

        Commands::Serve { index, model, addr } => {
            tracing::info!("Running serve command");
            tracing::info!("  Index: {}", index);
            tracing::info!("  Model: {}", model);
            tracing::info!("  Address: {}", addr);
            println!("Serve command not yet implemented");
        }

        Commands::Tui => {
            rrl::tui::run_tui()?;
        }
    }

    Ok(())
}
