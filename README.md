# RustRAGLab (RRL): A Rust Framework for RAG-Aware Fine-Tuning and Evaluation

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## Team Members

| Name | Student Number | Email |
|------|----------------|-------|
| Qiwen Lin | 1012495104 | qw.lin@mail.utoronto.ca |
| Liz Zhu | 1011844943 | liz.zhu@mail.utoronto.ca |

---

## Video Demo
https://youtu.be/9J0RZt0xrDs?si=lE0CfrMaOJLiz8dg

## Video Slide Presentation
https://youtu.be/BYjaDZOB8Tw 

---

## Motivation

Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by connecting them to external knowledge sources, improving accuracy and grounding. However, there is currently **no Rust-native framework** that supports the entire RAG workflow from ingestion to evaluation.

Existing tools such as **LangChain**, **LlamaIndex**, and **Haystack** are Python-based, while Rust developers must piece together fragmented crates (e.g., `hnsw_rs`, `tantivy`, `candle`) without a standardized architecture.

**RustRAGLab (RRL)** fills this gap by providing a **unified, performant, and safe Rust framework** for building and evaluating RAG systems with **optimized training and RAG pipeline** implementations. Full **CUDA and CPU support** has been **verified and tested** for production use.

### Why Rust?

Rust provides **memory safety**, **low runtime overhead**, and **predictable concurrency**, making it ideal for building high-performance retrieval and training pipelines without Python dependencies. Our optimized implementation leverages Rust's zero-cost abstractions for maximum throughput.

---

## Objectives

Design and implement an **end-to-end Rust-native framework** that integrates retrieval, adapter fine-tuning, and evaluation for RAG systems—offering both developer APIs and a CLI tool (`rrl`) for streamlined workflows.

---

## Key Features

### 1. Data and Chunking
- **Document Loader** — Plain text (`.txt`), Markdown, and HTML files supported
- **Chunking Strategies** — Fixed-size and overlapping chunking methods with token-aware splitting
- **Preprocessing Pipeline** — Tokenization, stopword filtering, sentence segmentation
- **CLI Command:** `rrl ingest --input ./docs --output ./output/chunks`

### 2. Embeddings and Model Support
- **Trait-based Embedder Interface** for modular backend integration
- **Backend** — Candle (Rust-native ML framework)
- **Hardware Acceleration** — CUDA (NVIDIA) and CPU support fully verified
- **Encoder Models** — BERT, RoBERTa, BGE, E5, DistilBERT, ALBERT, DeBERTa
- **Decoder Models (LLM)** — Qwen2, LLaMA, Mistral for text generation
- **Persistent Cache** — SQLite storage with versioning
- **CLI Command:** `rrl embed --input ./data/chunks.json --output ./data/embeddings`

### 3. Indexing and Retrieval
- **Dense Retrieval** — HNSW via `hnsw_rs`
- **Sparse Retrieval** — BM25 via `tantivy`
- **Hybrid Retriever** — Reciprocal Rank Fusion of dense and sparse signals
- **Optimized Pipeline** — Rust-native implementation for maximum throughput
- **Evaluation Metrics** — Recall@K, Precision@K, MRR, NDCG, MAP
- **CLI Command:** `rrl query --index ./index --query "What is RAG?"`

### 4. Fine-Tuning (RAG-Aware)
- **LoRA Fine-Tuning** using Candle (CUDA + CPU backends)
- **Optimized Training Pipeline** — Rust-native implementation with verified CUDA and CPU support
- **Multi-Architecture Support:**
  - Encoder Models: BERT, RoBERTa, BGE, E5, DistilBERT, ALBERT, DeBERTa
  - Decoder Models: Qwen2, LLaMA, Mistral (for generation fine-tuning)
- **Training Optimizations:** Mixed Precision Training, Gradient Checkpointing
- **Contrastive Loss** — Aligns model representations for retrieval tasks
- **CLI Command:** `rrl train --data ./data/train.jsonl --model bert-base-uncased`

### 5. Evaluation
- **Retrieval Metrics** — Recall@K, Precision@K, MRR, NDCG, MAP
- **Generation Metrics** — Perplexity, Exact Match (EM), F1
- **MS MARCO Evaluation** — Standard passage re-ranking benchmark
- **CLI Command:** `rrl eval-mc --data ./data/test.json --model bert-base-uncased`

### 6. RAG Pipeline
- **End-to-End Generation** — Retrieval-augmented generation with local LLMs
- **Multiple Generators** — Qwen2, LLaMA, Mistral support
- **Streaming Output** — Memory-efficient token generation
- **CLI Command:** `rrl rag --index ./index --query "What is machine learning?"`

### 7. Developer Interfaces

**CLI Commands:**
```bash
rrl ingest       # Load and chunk documents
rrl embed        # Compute embeddings
rrl index        # Build retrieval indexes (HNSW, BM25)
rrl query        # Query retrieval indexes
rrl train        # Fine-tune LoRA adapters
rrl eval         # Evaluate retrieval performance
rrl eval-mc      # Evaluate multiple-choice accuracy
rrl eval-msmarco # Evaluate on MS MARCO benchmark
rrl rag          # Run full RAG pipeline with LLM generation
rrl serve        # Launch API server
rrl infer        # Run inference on a model
```

**Rust API / SDK:**
- Modular traits: `Embedder`, `Retriever`, `Trainer`, `Evaluator`
- Integration with other Rust-based ML systems
- Type-safe configuration and error handling

### 8. Web Interface
- **Live Training Dashboard** — Real-time metrics, charts, and logs via WebSocket
- **Model Browser** — Explore and configure model architectures
- **Training Launcher** — Interactive job configuration and management
- **Evaluation Dashboard** — Test model performance with detailed metrics
- **RAG Workflow** — 4-step pipeline (Ingest, Embed, Index, Query)
- **Access:** `http://localhost:5173` (frontend) and `http://localhost:8000` (API)

---

## Optimized Implementation

### Training Pipeline
- **Verified CUDA Support** — Full GPU acceleration tested and production-ready
- **Verified CPU Support** — Efficient CPU-only training for broader compatibility
- **Gradient Checkpointing** — Memory-efficient training for large models
- **Parallel Data Loading** — Multi-threaded batch preparation

### RAG Pipeline
- **Native Rust Performance** — No Python bottlenecks
- **Concurrent Retrieval** — Parallel HNSW and BM25 search
- **Streaming Generation** — Memory-efficient token generation
- **Batched Processing** — High-throughput query handling

### Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| CUDA (NVIDIA GPU) | Verified | Full training and inference support |
| CPU | Verified | Optimized for multi-core processors |
| Metal (Apple Silicon) | Experimental | Limited testing |

---

## Project Structure

```
rrl/
├── src/
│   ├── cli/                    # Command-line interface
│   ├── cuda/                   # CUDA kernels for GPU acceleration
│   ├── data/                   # Document loading and chunking
│   │   ├── loaders/            # File format loaders
│   │   └── chunkers/           # Chunking strategies
│   ├── embedding/              # Embedding generation
│   │   ├── backends/           # Candle BERT backend
│   │   └── cache/              # SQLite embedding cache
│   ├── evaluation/             # Model evaluation metrics
│   │   ├── retrieval/          # IR metrics (Recall, MRR, NDCG)
│   │   └── generation/         # Generation metrics
│   ├── rag/                    # RAG pipeline
│   │   ├── context/            # Context building
│   │   ├── generator/          # LLM generation (Qwen2, LLaMA)
│   │   └── pipeline.rs         # End-to-end pipeline
│   ├── retrieval/              # Vector search and indexing
│   │   ├── dense/              # HNSW index
│   │   ├── sparse/             # BM25 index (Tantivy)
│   │   └── hybrid/             # Hybrid retrieval with RRF
│   ├── training/               # Training system
│   │   ├── lora/               # LoRA adapter implementation
│   │   ├── loss/               # Contrastive loss functions
│   │   ├── models/             # Model architectures
│   │   ├── dataset.rs          # Dataset loading
│   │   ├── optimizer.rs        # AdamW optimizer
│   │   └── trainer.rs          # Training loop
│   ├── server/                 # Server utilities
│   ├── utils/                  # Utility functions
│   ├── lib.rs                  # Library exports
│   └── main.rs                 # CLI entry point
├── Cargo.toml                  # Rust dependencies
├── README.md                   # This file
├── Proposal.md                 # Original project proposal
├── server.py                   # FastAPI backend server
└── ui/                         # React frontend
    ├── src/
    │   ├── pages/              # UI page components
    │   ├── App.jsx             # Main application
    │   └── api.js              # API client
    └── package.json
```

---

## Quick Start

### Prerequisites

```bash
# Rust 1.70+
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Python 3.8+ (for Web UI backend)
python --version

# Node.js 16+ (for Web UI frontend)
node --version

# For CUDA support (optional, NVIDIA GPU users)
# Install CUDA Toolkit 12.x
```

### Installation

```bash
# 1. Clone repository
git clone https://github.com/kevinlin29/ECE1724.git
cd ECE1724/rrl

# 2. Build Rust backend (choose based on your hardware)

# CPU-only build (training features enabled)
cargo build --release --features training

# CUDA GPU build (NVIDIA GPUs)
cargo build --release --features cuda

# Metal GPU build (Apple Silicon) - Experimental
cargo build --release --features metal

# Recommended: CUDA + training (for full functionality)
cargo build --release --features cuda,training

# 3. Install Python dependencies (for Web UI backend)
pip install fastapi uvicorn websockets python-multipart

# 4. Start the backend API server
python server.py
# Runs on http://localhost:8000

# 5. Install and run UI (in another terminal)
cd ui && npm install && npm run dev
# Runs on http://localhost:5173
```

Access the Web UI at http://localhost:5173

### Sample Data Setup

Use the script under `data/` to download training data:

```bash
cd data
python download_data.py
```

### Notes

- **CUDA GPU support** is recommended for training and RAG generation
- **Models:** We support Qwen2.5 family and LLaMA family. Use the HuggingFace model ID (e.g., `Qwen/Qwen2.5-0.5B`) or a local path to downloaded weights
- **Data paths:** Use absolute paths or paths relative to the `rrl/` directory

### Build Feature Flags

| Feature | Description | Status |
|---------|-------------|--------|
| `training` | Enables fine-tuning capabilities (CPU) | Verified |
| `cuda` | CUDA GPU acceleration + training | Verified |
| `metal` | Metal GPU acceleration + training | Experimental |

---

## Usage Examples


### 1. Complete RAG Workflow (CLI)

```bash
# Step 1: Ingest documents
./target/release/rrl ingest \
  --input ./docs \
  --output ./output/chunks \
  --chunk-size 512 \
  --chunk-overlap 50

# Step 2: Generate embeddings
./target/release/rrl embed \
  --input ./output/chunks \
  --output ./output/embeddings \
  --model token-embedder \
  --backend token

# Step 3: Build indexes
./target/release/rrl index \
  --chunks ./output/chunks \
  --embeddings ./output/embeddings \
  --output ./output/indexes \
  --model token-embedder \
  --index-type both  # builds both HNSW and BM25

# Step 4: Query (retrieval only)
./target/release/rrl query \
  --index ./output/indexes \
  --query "What is RAG?" \
  --top-k 5 \
  --retriever hybrid
```

### 2. RAG with LLM Generation

```bash
# Single query with Qwen2 - CPU
./target/release/rrl rag \
  --index ./output/indexes \
  --query "What is machine learning?" \
  --generator Qwen/Qwen2.5-0.5B \
  --embedder bert-base-uncased \
  --top-k 5 \
  --device cpu

# With CUDA acceleration
./target/release/rrl rag \
  --index ./output/indexes \
  --query "Explain neural networks" \
  --generator Qwen/Qwen2.5-0.5B \
  --retriever hybrid \
  --temperature 0.7 \
  --max-tokens 512 \
  --device cuda
```

**RAG Command Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--generator` | LLM for generation (Qwen2, LLaMA, Mistral) | `Qwen/Qwen2.5-0.5B` |
| `--embedder` | Encoder model for retrieval | `bert-base-uncased` |
| `--retriever` | Retriever type: dense, sparse, hybrid | `hybrid` |
| `--device` | Device: cpu, cuda, auto | `auto` |
| `--top-k` | Number of documents to retrieve | `5` |
| `--temperature` | Sampling temperature (0 = greedy) | `0.7` |
| `--max-tokens` | Maximum tokens to generate | `512` |
| `--dtype` | Model dtype: f32, f16, bf16 | `f16` |

### 3. Train a Model (CLI)

```bash
# CPU training
./target/release/rrl train \
  --data ./data/train.jsonl \
  --output ./outputs \
  --model bert-base-uncased \
  --epochs 3 \
  --batch-size 32 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --learning-rate 5e-5 \
  --device cpu

# CUDA training
./target/release/rrl train \
  --data ./data/train.jsonl \
  --output ./outputs \
  --model bert-base-uncased \
  --epochs 3 \
  --batch-size 16 \
  --lora-rank 8 \
  --learning-rate 5e-5 \
  --device cuda
```

### 4. Evaluate Model

```bash
# Multiple-choice evaluation
./target/release/rrl eval-mc \
  --data ./data/test.json \
  --model bert-base-uncased \
  --checkpoint ./outputs/final/lora_checkpoint.safetensors \
  --device cuda

# MS MARCO evaluation
./target/release/rrl eval-msmarco \
  --data ./data/msmarco_validation.jsonl \
  --model bert-base-uncased \
  --checkpoint ./outputs/final/lora_checkpoint.safetensors \
  --sample 1000
```

### 5. Web Interface

```bash
# Start the backend API server
python server.py
# Runs on http://localhost:8000

# Start the frontend (in another terminal)
cd ui && npm install && npm run dev
# Runs on http://localhost:5173

# Open browser to http://localhost:5173
```

---

## Contributions by Team Member

| Team Member | Responsibilities |
|-------------|------------------|
| **Qiwen Lin** | Backend Systems and Embedding/Retrieval: Core framework architecture, data loaders, embedding engine, retrieval (HNSW + Tantivy), hybrid retriever design, performance optimization, CUDA verification, Video Slide Presentation |
| **Liz Zhu** | Training, Evaluation and Serving: LoRA fine-tuning pipeline, evaluation metrics, web interface, API server implementation, CPU/CUDA testing, video demo |

---

## Testing

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_bert_lora

# Run with output
cargo test -- --nocapture

# Test CUDA support (if available)
cargo test --features cuda

# Test CPU training
cargo test --features training
```

---

## Lessons Learned and Concluding Remarks

### Technical Lessons

1. **Rust ML Ecosystem Maturity**: The Rust ML ecosystem has matured significantly. Libraries like Candle provide production-quality tensor operations with GPU support. However, documentation can be sparse, and we often had to read source code to understand APIs.

2. **Memory Management in Training**: Training transformers in Rust requires careful memory management. We implemented gradient checkpointing to enable training on consumer GPUs with limited VRAM. Rust's ownership model made memory leaks impossible, but we had to be deliberate about tensor lifetimes.

3. **Hybrid Retrieval Effectiveness**: Combining dense (HNSW) and sparse (BM25) retrieval consistently outperformed either approach alone. The Reciprocal Rank Fusion algorithm proved simple yet effective for score combination.

4. **LoRA Efficiency**: LoRA fine-tuning is remarkably parameter-efficient. With rank=8, we trained less than 1% of BERT's parameters while achieving significant improvements on domain-specific tasks.

5. **Tokenization Complexity**: Properly handling tokenization for BERT models (WordPiece, special tokens, attention masks) was more complex than anticipated. The HuggingFace tokenizers crate was essential for correctness.

### Engineering Lessons

1. **Feature Flags for Optional Dependencies**: Using Cargo features (`training`, `cuda`, `metal`) allowed us to keep the core library lightweight while enabling heavy dependencies only when needed.

2. **Async Architecture**: While we used async Rust (Tokio) for I/O operations, ML inference is CPU/GPU-bound. We learned to carefully separate async I/O from synchronous compute paths.

3. **Testing ML Systems**: Testing ML systems requires both unit tests for components and integration tests with known inputs/outputs. We implemented comprehensive tests for retrieval metrics.

4. **Error Handling**: Rust's `Result` type forced us to handle errors explicitly throughout the codebase, resulting in more robust error messages and graceful failure modes.

### Concluding Remarks

RustRAGLab demonstrates that production-quality RAG systems can be built entirely in Rust, achieving both safety and performance. The project successfully integrates document processing, embedding generation, hybrid retrieval, LoRA fine-tuning, and LLM generation into a cohesive framework.

**Key achievements include:**
- A complete, working RAG pipeline from ingestion to generation
- GPU-accelerated inference and training via CUDA/Metal
- Comprehensive evaluation suite with standard IR metrics (Recall@K, Precision@K, MRR, NDCG, MAP)
- User-friendly CLI and web interfaces
- Modular architecture enabling easy extension
- Support for multiple model architectures (BERT, RoBERTa, Qwen2, LLaMA, Mistral)

**Future work** could extend RRL with additional model architectures, distributed training support, and integration with more document formats. We also envision tighter integration between the retrieval and generation components for end-to-end RAG training.

We hope RustRAGLab serves as both a practical tool and a reference implementation for building ML systems in Rust.

---

## Used Crates and Packages

- **Candle** — Rust ML framework by HuggingFace
- **HuggingFace** — Model hub and transformers
- **hnsw_rs** — HNSW implementation in Rust
- **Tantivy** — Full-text search engine in Rust
- **FastAPI** — Python web framework
- **React** — Frontend UI framework

---

## License

MIT License

Copyright (c) 2025 Qiwen Lin, Liz Zhu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
