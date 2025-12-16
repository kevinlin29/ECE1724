# RustRAGLab (RRL): A Rust Framework for RAG-Aware Fine-Tuning and Evaluation

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/react-18.2-61dafb.svg)](https://reactjs.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 🎯 Motivation

Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by connecting them to external knowledge sources, improving accuracy and grounding. However, there is currently **no Rust-native framework** that supports the entire RAG workflow from ingestion to evaluation.

Existing tools such as **LangChain**, **LlamaIndex**, and **Haystack** are Python-based, while Rust developers must piece together fragmented crates (e.g., `hnsw_rs`, `tantivy`, `candle`) without a standardized architecture.

**RustRAGLab (RRL)** fills this gap by providing a **unified, performant, and safe Rust framework** for building and evaluating RAG systems with **optimized training and RAG pipeline** implementations. Full **CUDA and CPU support** has been **verified and tested** for production use.

### Why Rust?
Rust provides **memory safety**, **low runtime overhead**, and **predictable concurrency**, making it ideal for building high-performance retrieval and training pipelines without Python dependencies. Our optimized implementation leverages Rust's zero-cost abstractions for maximum throughput.

---

## 🚀 Objective

Design and implement an **end-to-end Rust-native framework** that integrates retrieval, adapter fine-tuning, and evaluation for RAG systems—offering both developer APIs and a CLI tool (`rrl`) for streamlined workflows.

**NEW:** Complete web interface for training, evaluation, and RAG workflows with live monitoring.

---

## ✨ Key Features

### 1. Data & Chunking
- ✅ **Document Loader** — Plain text (`.txt`) files supported
- ✅ **Chunking Strategies** — Fixed-size and overlapping chunking methods
- ✅ **Preprocessing Pipeline** — Tokenization, stopword filtering, sentence segmentation
- ✅ **CLI Command:** `rrl ingest --input ./docs --output ./output/chunks`

### 2. Embeddings & Model Support
- ✅ **Trait-based Embedder Interface** for modular backend integration
- ✅ **Backend** — **Candle** (Rust-native ML framework)
- ✅ **Hardware Acceleration** — **CUDA (NVIDIA)** and **CPU** support fully verified
- ✅ **Encoder Models** — BERT, RoBERTa, BGE, E5, DistilBERT, ALBERT, DeBERTa
- ✅ **Decoder Models (LLM)** — **Qwen2**, **LLaMA**, **Mistral** for text generation
- ✅ **Persistent Cache** — SQLite storage with versioning
- ✅ **CLI Command:** `rrl embed --input ./data/chunks.json --output ./data/embeddings.safetensors`

### 3. Indexing & Retrieval
- ✅ **Dense Retrieval** — HNSW via `hnsw_rs`
- ✅ **Sparse Retrieval** — BM25 via `tantivy`
- ✅ **Hybrid Retriever** — Weighted fusion of dense and sparse signals
- ✅ **Optimized Pipeline** — Rust-native implementation for maximum throughput
- ✅ **Evaluation Metrics** — Recall@k, Mean Reciprocal Rank (MRR)
- ✅ **CLI Command:** `rrl query --index ./index --query "What is RAG?"`

### 4. Fine-Tuning (RAG-Aware)
- ✅ **LoRA / QLoRA / DoRA Fine-Tuning** using **Candle** (CUDA + CPU backends)
- ✅ **Optimized Training Pipeline** — Rust-native implementation with verified CUDA and CPU support
- ✅ **Multi-Architecture Support:**
  - **Encoder Models:** BERT, RoBERTa, BGE, E5, DistilBERT, ALBERT, DeBERTa
  - **Decoder Models:** **Qwen2**, **LLaMA**, **Mistral** (for generation fine-tuning)
- ✅ **Multi-Adapter Support** — Train and switch between task-specific adapters
- ✅ **Training Optimizations:**
  - Flash Attention
  - Mixed Precision Training
  - Gradient Checkpointing
  - Distributed Training Support
- ✅ **Memory Efficiency** — Train 7B-70B models on consumer GPUs with QLoRA
- ✅ **Grounding-Aware Loss** — Aligns model attention with retrieved chunks
- ✅ **Auto-Scaling** — Dynamic batch size, learning rate, and gradient accumulation optimization
- ✅ **CLI Command:** `rrl train --data ./data/train.jsonl --model BAAI/bge-base-en-v1.5`

### 5. Evaluation
- ✅ **Retrieval Metrics** — Recall@k, Mean Reciprocal Rank (MRR)
- ✅ **Generation Metrics** — Perplexity, Exact Match (EM), F1, ROUGE-L
- ✅ **Attribution Metrics** — Support fraction and citation precision/recall
- ✅ **CLI Command:** `rrl eval-mc --data ./data/test.json --model bert-base-uncased`

### 6. Developer Interfaces

**CLI Commands:**
```bash
rrl ingest    # Load and chunk documents
rrl embed     # Compute embeddings and build indexes
rrl index     # Build retrieval indexes (HNSW, BM25)
rrl query     # Query retrieval indexes
rrl train     # Fine-tune LoRA adapters (encoder/decoder models)
rrl eval      # Evaluate retrieval performance
rrl eval-mc   # Evaluate multiple-choice accuracy
rrl rag       # Run full RAG pipeline with LLM generation (Qwen2/LLaMA/Mistral)
rrl infer     # Run inference on a model
rrl serve     # Launch API server
```

**Rust API / SDK:**
- ✅ Modular traits: `Embedder`, `Retriever`, `Trainer`, `Evaluator`
- ✅ Integration with other Rust-based ML systems
- ✅ Type-safe configuration and error handling

### 7. Web Interface (Primary UI)

**Complete React-based UI with live monitoring** — the primary way to interact with RRL:
- ✅ **Live Training Dashboard** — Real-time metrics, charts, logs via WebSocket
- ✅ **Model Browser** — Explore and configure model architectures
- ✅ **Training Launcher** — Interactive job configuration and management
- ✅ **Evaluation Dashboard** — Test model performance with detailed metrics
- ✅ **Inference Playground** — Interactive model testing environment
- ✅ **RAG Workflow** — 4-step pipeline (Ingest → Embed → Index → Query)
- ✅ **Data Upload** — Drag-and-drop dataset management

> **Note:** The Web UI is the recommended interface. Terminal UI (ratatui) development has been transitioned to focus on the Web UI.

**Access:** `http://localhost:5173` (after running `npm run dev`)

### 8. Server & API

**FastAPI Backend:**
- ✅ **REST API** — Complete API for all RRL operations
- ✅ **WebSocket** — Live training updates and streaming
- ✅ **File Upload** — Dataset upload with progress tracking
- ✅ **Job Management** — Start, stop, monitor training jobs
- ✅ **Model Serving** — Inference endpoints for trained models

**Access:** `http://localhost:8000/docs` (API documentation)

---

## 🔧 Optimized Implementation

### Training Pipeline
**RustRAGLab** features an **optimized training pipeline** built entirely in Rust:
- ✅ **Verified CUDA Support** — Full GPU acceleration tested and production-ready
- ✅ **Verified CPU Support** — Efficient CPU-only training for broader compatibility
- ✅ **Zero-Copy Operations** — Minimized memory overhead
- ✅ **Parallel Data Loading** — Multi-threaded batch preparation
- ✅ **Auto-Scaling** — Dynamic resource optimization for maximum throughput

### RAG Pipeline
**End-to-end optimized RAG implementation:**
- ✅ **Native Rust Performance** — No Python bottlenecks
- ✅ **Concurrent Retrieval** — Parallel HNSW and BM25 search
- ✅ **Streaming Generation** — Memory-efficient token generation
- ✅ **Batched Processing** — High-throughput query handling
- ✅ **Hardware Acceleration** — Verified CUDA and CPU execution

### Verified Platform Support
| Platform | Status | Notes |
|----------|--------|-------|
| **CUDA (NVIDIA GPU)** | ✅ Verified | Full training and inference support |
| **CPU** | ✅ Verified | Optimized for multi-core processors |
| **Metal (Apple Silicon)** | ⚠️ Experimental | Limited testing, community feedback welcome |

---

## 📁 Project Structure

```
rrl/
├── src/                        # 🦀 Rust source code
│   ├── cli/                    # ✅ Command-line interface
│   ├── cuda/                   # ✅ CUDA kernels for GPU acceleration
│   ├── data/                   # ✅ Dataset handling
│   ├── embedding/              # ✅ Embedding generation
│   ├── evaluation/             # ✅ Model evaluation metrics
│   ├── rag/                    # ✅ RAG system implementation
│   ├── retrieval/              # ✅ Vector search and indexing
│   ├── server/                 # ✅ Server utilities
│   ├── training/               # ✅ Training system
│   │   ├── autoscaling.rs      # Auto-scaling optimizer
│   │   ├── dataset.rs          # Dataset loading
│   │   ├── device.rs           # Device management (CPU/CUDA)
│   │   ├── evaluation.rs       # Evaluation metrics
│   │   ├── optimizer.rs        # AdamW optimizer
│   │   ├── tokenizer.rs        # Tokenization
│   │   ├── trainer.rs          # Training loop
│   │   └── models/             # 10+ model architectures
│   ├── tui/                    # ✅ Terminal UI
│   ├── utils/                  # ✅ Utility functions
│   ├── lib.rs                  # Library exports
│   └── main.rs                 # CLI entry point
├── server.py                   # 🆕 FastAPI backend
├── ui/                         # 🆕 React frontend
│   ├── src/
│   │   ├── pages/              # UI page components
│   │   │   ├── Dashboard.jsx   # Live training monitor
│   │   │   ├── Training.jsx    # Training launcher
│   │   │   ├── Models.jsx      # Model browser
│   │   │   ├── Evaluation.jsx  # Evaluation dashboard
│   │   │   ├── Inference.jsx   # Inference playground
│   │   │   ├── RAG.jsx         # RAG workflow
│   │   │   └── DataUpload.jsx  # Dataset uploader
│   │   ├── App.jsx             # Main application
│   │   ├── api.js              # API client
│   │   └── main.jsx            # Entry point
│   └── package.json
├── test-docs/                  # 🆕 Sample documents for testing
│   ├── ml.txt
│   ├── rag.txt
│   └── rust.txt
├── Cargo.toml                  # Rust dependencies
├── README.md                   # This file
├── Proposal.md                 # Original project proposal
├── CODE_STANDARDS.md           # 🆕 Code formatting guidelines
└── TASK_MANAGEMENT.md          # 🆕 Development workflow
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Rust 1.70+
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Python 3.8+
python --version

# Node.js 16+
node --version

# For CUDA support (optional, NVIDIA GPU users)
# Install CUDA Toolkit 11.8+ and cuDNN
```

### Installation

```bash
# 1. Clone repository
git clone https://github.com/kevinlin29/ECE1724.git
cd ECE1724/rrl

# 2. Build Rust backend (choose one based on your hardware)

# CPU-only build (training features enabled) - VERIFIED ✅
cargo build --release --features training

# CUDA GPU build (NVIDIA GPUs) - VERIFIED ✅
cargo build --release --features cuda

# Metal GPU build (Apple Silicon) - EXPERIMENTAL ⚠️
cargo build --release --features metal

# 3. Install Python dependencies (for Web UI backend)
pip install fastapi uvicorn websockets python-multipart

# 4. Install UI dependencies
cd ui
npm install
```

### Build Feature Flags

| Feature | Description | Status |
|---------|-------------|--------|
| `training` | Enables fine-tuning capabilities (CPU) | ✅ Verified |
| `cuda` | CUDA GPU acceleration + training | ✅ Verified |
| `metal` | Metal GPU acceleration + training | ⚠️ Experimental |

### Run the Platform

**Terminal 1 - Backend API:**
```bash
python server.py
# Runs on http://localhost:8000
```

**Terminal 2 - Frontend UI:**
```bash
cd ui
npm install
npm run dev
# Runs on http://localhost:5173
```

**Open browser:** http://localhost:5173

---

## 📖 Usage Examples

### 1. RAG Workflow (Web UI - Recommended)

1. **Open RAG Interface:** http://localhost:5173/rag

2. **Tab 1: Ingest Documents**
   ```
   Input Directory: ./test-docs
   Chunk Size: 512
   Chunk Overlap: 50
   → Click "Ingest Documents"
   ```

3. **Tab 2: Generate Embeddings**
   ```
   Model: BAAI/bge-base-en-v1.5
   Batch Size: 32
   → Click "Generate Embeddings"
   ```

4. **Tab 3: Build Index**
   ```
   Index Type: HNSW (Fast)
   → Click "Build Index"
   ```

5. **Tab 4: Query**
   ```
   Query: "What is machine learning?"
   Top K: 5
   → Click "Search"
   → View ranked results with scores
   ```

### 2. RAG Workflow (CLI)

```bash
# Step 1: Ingest documents
rrl ingest --input ./test-docs --output ./output/chunks

# Step 2: Generate embeddings
rrl embed \
  --input ./output/chunks \
  --output ./output/embeddings \
  --model BAAI/bge-base-en-v1.5

# Step 3: Build indexes
rrl index \
  --chunks ./output/chunks \
  --embeddings ./output/embeddings \
  --output ./output/indexes \
  --model BAAI/bge-base-en-v1.5 \
  --index-type both  # builds both HNSW and BM25

# Step 4: Query (retrieval only)
rrl query \
  --index ./output/indexes \
  --query "What is RAG?" \
  --top-k 5 \
  --retriever hybrid
```

### 3. RAG with LLM Generation (CLI)

Use the `rrl rag` command for full retrieval-augmented generation with **Qwen2** or **LLaMA**:

```bash
# Single query with Qwen2 (default) - CPU
rrl rag \
  --index ./output/indexes \
  --query "What is machine learning?" \
  --generator Qwen/Qwen2.5-0.5B \
  --embedder bert-base-uncased \
  --top-k 5 \
  --device cpu

# With CUDA acceleration
rrl rag \
  --index ./output/indexes \
  --generator meta-llama/Llama-2-7b-hf \
  --embedder BAAI/bge-base-en-v1.5 \
  --retriever hybrid \
  --temperature 0.7 \
  --max-tokens 512 \
  --device cuda

# With fine-tuned checkpoints
rrl rag \
  --index ./output/indexes \
  --query "How do I make pasta?" \
  --generator Qwen/Qwen2.5-0.5B \
  --generator-checkpoint ./outputs/final/lora_weights.safetensors \
  --embedder bert-base-uncased \
  --embedder-checkpoint ./outputs/embedder/lora_weights.safetensors \
  --format json
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
| `--template` | Prompt template: default, concise, detailed | `default` |
| `--format` | Output format: text, json | `text` |
| `--dtype` | Model dtype: f32, f16, bf16 | `f16` |

### 4. Train a Model (Web UI)

1. **Open Training Interface:** http://localhost:5173/training
2. Select model (e.g., `BAAI/bge-base-en-v1.5`)
3. Upload dataset or specify path
4. Configure hyperparameters:
   - Epochs: 3
   - Batch Size: 32 (auto-scaling enabled)
   - Learning Rate: 5e-5
   - LoRA Rank: 16
   - Device: CUDA (if available) or CPU
5. Click "Start Training"
6. Watch live metrics and logs in real-time

### 5. Train a Model (CLI)

```bash
# CPU training
rrl train \
  --data ./data/train.jsonl \
  --output ./outputs \
  --model BAAI/bge-base-en-v1.5 \
  --epochs 3 \
  --batch-size 32 \
  --lora-rank 16 \
  --learning-rate 5e-5 \
  --device cpu

# CUDA training with auto-scaling
rrl train \
  --data ./data/train.jsonl \
  --output ./outputs \
  --model BAAI/bge-base-en-v1.5 \
  --epochs 3 \
  --batch-size 8 \
  --lora-rank 16 \
  --learning-rate 5e-5 \
  --device cuda \
  --enable-autoscaling
```

### 6. Evaluate Model

**Web UI:**
1. Go to http://localhost:5173/evaluation
2. Select model and checkpoint
3. Upload test data
4. Click "Run Evaluation"
5. View accuracy and MRR metrics

**CLI:**
```bash
rrl eval-mc \
  --data ./data/test.json \
  --model BAAI/bge-base-en-v1.5 \
  --checkpoint ./outputs/checkpoint-500/lora_weights.safetensors \
  --device cuda
```

---

## 🗓️ Development Timeline

### ✅ Week 1–2: System Architecture & Data Pipeline (COMPLETED)
- [x] Define high-level module layouts
- [x] Implement document loader interface (PDF, MD, text)
- [x] Implement text chunker and tokenizer
- [x] Design and test `rrl ingest` with CLI parsing
- [x] Validate functionality and performance

### ✅ Week 3–4: Embedding Engine (COMPLETED)
- [x] Implement `Embedder` trait abstraction
- [x] Integrate `tch` and `onnxruntime` backends
- [x] Add `rrl embed` for local embedding
- [x] Support pooling strategies (mean, CLS) and normalization
- [x] Create persistent embedding cache (SQLite)
- [x] Benchmark embedding throughput and GPU utilization

### ✅ Week 5–6: Indexing & Retrieval (COMPLETED)
- [x] Integrate HNSW and Tantivy for dense/sparse retrieval
- [x] Implement `Retriever` trait and hybrid retriever
- [x] Add `rrl query` and evaluation commands
- [x] Measure Recall@k and MRR metrics
- [x] Optimize query latency with multithreading

### ✅ Week 7–8: Fine-Tuning & Evaluation Framework (COMPLETED)
- [x] Implement LoRA/QLoRA/DoRA fine-tuning using Candle
- [x] Add grounding-aware loss function
- [x] Build `rrl train` and `rrl eval` commands
- [x] Integrate generation metrics (F1, EM, ROUGE-L, Perplexity)
- [x] Multi-adapter support
- [x] Auto-scaling implementation

### ✅ Week 8.5: Web Interface & API (COMPLETED)
- [x] FastAPI backend with REST API
- [x] React frontend with 7 pages
- [x] Live training dashboard with WebSocket
- [x] RAG workflow interface
- [x] Model browser and evaluation dashboard
- [x] Data upload with drag-and-drop

### ✅ Week 9: RAG Pipeline & LLM Integration (COMPLETED)
- [x] Implement `rrl rag` command with full RAG pipeline
- [x] Integrate decoder models: **Qwen2**, **LLaMA**, **Mistral**
- [x] Web-based dashboard (React UI - primary interface)
- [x] Transitioned from Terminal UI (ratatui) to Web UI
- [x] Support for fine-tuned checkpoints in RAG pipeline
- [x] Optimize training and RAG pipelines in Rust

### ✅ Week 10: Final Integration & Testing (COMPLETED)
- [x] Complete full end-to-end RAG workflow
- [x] MS MARCO evaluation support
- [x] Multi-architecture model loading (encoder + decoder)
- [x] Comprehensive CLI with all commands
- [x] **Verify CUDA and CPU support** across all components
- [x] Performance optimization and testing
- [x] Documentation and README updates

---

## 👥 Team Roles

| Team Member | Responsibilities |
|-------------|------------------|
| **Kevin Lin** | Backend Systems & Embedding/Retrieval: Core framework architecture, data loaders, embedding engine, retrieval (HNSW + Tantivy), hybrid retriever design, performance optimization, CUDA verification |
| **Liz Zhu** | Training, Evaluation & Serving: LoRA fine-tuning pipeline, evaluation metrics, web interface, API server implementation, dashboard, auto-scaling, CPU/CUDA testing, Docker packaging |

---

## 🧪 Testing

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_bert_lora

# Run with output
cargo test -- --nocapture

# Test frontend
cd ui && npm test

# Test CUDA support (if available)
cargo test --features cuda

# Test CPU training
cargo test --features training
```

---

## 📚 Documentation

- **[CODE_STANDARDS.md](CODE_STANDARDS.md)** — Code formatting, structure, and testing guidelines
- **[TASK_MANAGEMENT.md](TASK_MANAGEMENT.md)** — Development workflow and task management
- **[Proposal.md](Proposal.md)** — Original project proposal
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** — Training module implementation guide
- **[AUTOSCALING_GUIDE.md](AUTOSCALING_GUIDE.md)** — Auto-scaling configuration and usage

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Candle** — Rust ML framework by HuggingFace
- **HuggingFace** — Model hub and transformers
- **FastAPI** — Python web framework
- **React** — UI framework
- **TailwindCSS** — Styling framework
- **hnsw_rs** — HNSW implementation
- **tantivy** — Full-text search engine

---

## ⚡ Quick Links

- **GitHub:** https://github.com/kevinlin29/ECE1724
- **Web UI:** http://localhost:5173
- **API Docs:** http://localhost:8000/docs
- **Original Proposal:** [Proposal.md](Proposal.md)
- **Youtube Video:** https://youtu.be/0vOJnGV3A2s

---
