# RustRAGLab (RRL): A Rust Framework for RAG-Aware Fine-Tuning and Evaluation

## Motivation
Large language models often rely on Retrieval-Augmented Generation (RAG) to connect to external knowledge sources, making them more capable and accurate in a specified domain. Currently there are not a single rust-native framework which supports the RAG workflow of a LLM from start to finish, and that is where this project comes in. Most existing tools are python based, such as LangChain, LlamaIndex, and Haystack, while rust developers are given fragemented codes which are not connect to form a standarized architecture. Thus meaning that there are no current tools which support the building of retrieval pipelines, fine-tuning adapters, and their correspoding evaluation tools.

We are motivated to fill this gap by building RustRAGLab (RRL), which is designed to be a Rust-native framework to support the full life cycle of a RAG system, with the following key points in mind:
1. Enables safe, performant, and low-overhead pipelines from Rust's core strength.
2. Fills a real gap in the Rust Machine Learning environment
3. Provides broth a CLI and APIs to lower barriers for developers that seeks to experiment with RAG outside of python

## Objective and Features
### Objective:
Design and implement a end-to-end Rust-native framework that integrates retrieval, adapter fine-tuning, and evaluation for RAG systems, offering both developer APIs and a CLI tool (rrl) for streamlined use.

Data & Chunking: PDF/Markdown loaders → text → chunkers (fixed, overlap, semantic).

---

### Features:
- **Data & Chunking**
  - **Flexible Loaders** – Support for multiple document types such as PDF, Markdown, and plain text.
  - **Chunking Strategies** – Implement fixed-size, overlapping, and semantic based chunking methods to enable tunable trade-offs between retrieval accuracy and performance.
  - **Pre-processing Pipeline** – Includes tokenization, stopword filtering, and optional sentence segmentation using rust-tokenizers or tiktoken-rs.

- **Embeddings**
  - Trait-based embedding interface.  
  - Backends via `tch` (Torch bindings) and `onnxruntime`.  
  - Hardware acceleration support for **CUDA** (NVIDIA GPUs) and **Metal** (Apple GPUs). 
  
- **Indexing & Retrieval**
  - Dense retrieval using **HNSW** (`hnsw_rs`).  
  - Sparse retrieval using **BM25** (`tantivy`).  
  - Hybrid retriever that combines dense and sparse signals.  

- **Developer Interface**
  - **CLI tools**
    - `rrl ingest` → load and chunk documents
    - `rrl embed`: generate and store embeddings
    - `rrl train`: fine-tune adapter or retriever
    - `rrl eval` : evaluate retrieval or generation pipeline
    - `rrl serve`: launch Axum-based inference server with streaming responses and hot-reload for new documents.
  - **Rust API / SDK** – For integration into other Rust ML projects, exposing modular traits (Embedder, Retriever, Trainer, Evaluator)

- **Evaluation**
  - Retrieval metrics: Recall@k, Mean Reciprocal Rank (MRR).
  - Generation metrics: Perplexity, Exact Match, F1, ROUGE-L.
  - Attribution score: fraction of answers that explicitly cite retrieved documents.

- **Server & Dashboard**
  - REST API via **Axum**, with streaming output for interactive inference.
  - Index hot-reloading without downtime.
  - Terminal UI with **ratatui** for visualizing retrieval stats and training progress.

## Architecture Design
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
