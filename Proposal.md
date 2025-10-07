# RustRAGLab (RRL): A Rust Framework for RAG-Aware Fine-Tuning and Evaluation

## Motivation
Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by connecting them to external knowledge sources, improving accuracy and grounding.  
However, there is currently **no Rust-native framework** that supports the entire RAG workflow from ingestion to evaluation.

Existing tools such as **LangChain**, **LlamaIndex**, and **Haystack** are Python-based, while Rust developers must piece together fragmented crates (e.g., `hnsw_rs`, `tantivy`, `candle`) without a standardized architecture.

**RustRAGLab (RRL)** aims to fill this gap by providing a **unified, performant, and safe Rust framework** for building and evaluating RAG systems.

### Why Rust?
Rust provides **memory safety**, **low runtime overhead**, and **predictable concurrency**, making it ideal for building high-performance retrieval and training pipelines without Python dependencies.

---

## Objective
Design and implement an **end-to-end Rust-native framework** that integrates retrieval, adapter fine-tuning, and evaluation for RAG systems—offering both developer APIs and a CLI tool (`rrl`) for streamlined workflows.

---

## Key Features

### 1. Data & Chunking
- **Flexible Loaders** – Support PDF, Markdown, and plain text documents.
- **Chunking Strategies** – Fixed-size, overlapping, and semantic-based methods to balance recall and efficiency.
- **Preprocessing Pipeline** – Tokenization, stopword filtering, and optional sentence segmentation using `rust-tokenizers` or `tiktoken-rs`.

### 2. Embeddings
- **Trait-based Embedder Interface** for modular backend integration.
- **Backends** – `tch` (Torch bindings) and `onnxruntime`.
- **Hardware Acceleration** – Support for **CUDA (NVIDIA)** and **Metal (Apple)** GPUs.
- **Persistent Cache** – Store embeddings in SQLite with versioning and reproducibility manifests.

### 3. Indexing & Retrieval
- **Dense Retrieval** – HNSW via `hnsw_rs`.
- **Sparse Retrieval** – BM25 via `tantivy`.
- **Hybrid Retriever** – Weighted fusion of dense and sparse signals for balanced performance.

### 4. Fine-Tuning (RAG-Aware)
- **LoRA / QLoRA Fine-Tuning** using **Candle** (CUDA + Metal backends).
- **Grounding-Aware Loss** – Encourages responses that cite retrieved passages and penalizes hallucinations by aligning model attention with relevant chunks.
- **Optional Contrastive Training** – Incorporate hard negatives to enhance retrieval–generation consistency.

### 5. Evaluation
- **Retrieval Metrics** – Recall@k, Mean Reciprocal Rank (MRR).
- **Generation Metrics** – Perplexity, Exact Match (EM), F1, ROUGE-L.
- **Attribution Metrics** – Support fraction (answers that cite retrieved docs) and citation precision/recall.

### 6. Developer Interfaces
- **CLI Commands**
  - `rrl ingest` – load and chunk documents.
  - `rrl embed` – compute embeddings and build indexes.
  - `rrl train` – fine-tune LoRA adapters.
  - `rrl eval` – evaluate retrieval/generation pipelines.
  - `rrl serve` – launch Axum server with streaming responses and hot reload.
- **Rust API / SDK**
  - Modular traits: `Embedder`, `Retriever`, `Trainer`, `Evaluator`.
  - Allows integration with other Rust-based ML systems.

### 7. Server & Visualization
- **REST API** via Axum with streaming generation.
- **Hot-Reloadable Indexes** – update sources without downtime.
- **Terminal Dashboard (ratatui)** – visualize retrieval stats, GPU utilization, latency trends, and training progress.

---

## Architecture Overview

