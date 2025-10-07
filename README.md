# RustRAGLab (RRL): A Rust Framework for RAG-Aware Fine-Tuning and Evaluation

## Motivation
Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by connecting them to external knowledge sources, improving accuracy and grounding. However, there is currently **no Rust-native framework** that supports the entire RAG workflow from ingestion to evaluation.

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
            ┌─────────────────────────────┐
            │        Data Sources         │
            │  (PDF / Markdown / Text)    │
            └──────────────┬──────────────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │  Ingestion &     │
                 │  Chunking Layer  │
                 │  (rrl ingest)    │
                 └───────┬──────────┘
                         │
                         ▼
              ┌─────────────────────────┐
              │   Embedding Engine      │
              │ (tch / onnxruntime)     │
              │  → generates vectors    │
              └────────┬────────────────┘
                       │
                       ▼
      ┌──────────────────────────────────────┐
      │   Indexing & Retrieval Engine        │
      │ (HNSW + BM25 Hybrid via hnsw_rs,     │
      │  tantivy)                            │
      └──────────┬───────────────────────────┘
                 │
                 ▼
      ┌─────────────────────────┐
      │   Fine-Tuning (LoRA)    │
      │   + Grounding Loss      │
      │   (Candle, CUDA/Metal)  │
      └──────────┬──────────────┘
                 │
                 ▼
      ┌─────────────────────────┐
      │   Evaluation Layer      │
      │ (Recall@k, MRR, F1, etc)│
      └──────────┬──────────────┘
                 │
                 ▼
      ┌─────────────────────────┐
      │  Axum REST Server       │
      │  + Streaming Responses  │
      │  (rrl serve)            │
      └─────────────────────────┘


## Tentative Plan
This project is to be developed over the next 10 weeks, with a modular approach. It will be developed in subsystems (data, embedding, retrieval, fine-tuning, evaluation, and serving) to ensure robustness and efficiency.

### Week 1–2: System Architecture & Data Pipeline
- Define high level module layouts
- Implement document loader interface to support PDF, md, and text
- Implement text chunker and tokenizer
- Design and test `rrl ingest` with CLI parsing
- Validate functionality and test for performance

### Week 3–4: Embedding Engine
- Implement the `Embedder` trait abstraction
- Integrate `tch` and/or `onnxruntime`backend for harware support
- Add `rrl embed` for local embedding
- Support basic pooling strategies (mean, CLS) and vector normalization.
- Create a persistent embedding cache (SQLite) with version tracking and reproducibility manifests.
- Benchmark embedding throughput and GPU/CPU utilization.

### Week 5–6: Indexing & Retrieval

- Integrate **HNSW** and **Tantivy** for dense/sparse retrieval.
- Implement `Retriever` trait and hybrid retriever with weighted fusion.  
- Add `rrl query` and evaluation commands.  
- Measure Recall@k and MRR metrics; optimize query latency with multithreading.

### Week 7–8: Fine-Tuning & Evaluation Framework
- Implement LoRA/QLoRA fine-tuning using **Candle** (CUDA + Metal).  
- Add grounding-aware loss function to align retriever and generator.  
- Build `rrl train` and `rrl eval` commands.  
- Integrate generation metrics (F1, EM, ROUGE-L, Perplexity).  

### Week 9: Serving & Visualization
- Develop `rrl serve` — an Axum-based inference server with REST endpoints
- Integrate a lightweight dashboard using ratatui to monitor: Retrieval statistics, Query throughput, GPU utilization and latency trends.

### Week 10 
- Complete full end-to-end test
- Profile latency and memory footprint across all backends for efficiency
- Complete Final report
- Prepare for final demo

## Team Roles

| Team Member | Responsibilities |
|--------------|------------------|
| **Kevin Lin – Backend Systems & Embedding/Retrieval** | Core framework architecture, data loaders, embedding engine, retrieval (HNSW + Tantivy), hybrid retriever design, and performance optimization. |
| **Liz Zhu – Training, Evaluation & Serving** | LoRA fine-tuning pipeline, evaluation metrics, Axum server implementation (`rrl serve`), terminal dashboard, and Docker packaging. |
