# RustRAGLab (RRL): A Rust Framework for RAG-Aware Fine-Tuning and Evaluation

## Motivation
Retrieval-Augmented Generation (RAG) combines large language models (LLMs) with external knowledge sources, making them more accurate and tailored to the datasource. But in the Rust ecosystem, there are no complete frameworks that supports the full RAG workflow from start to finish. Most existing tools are Python-based—like LangChain, LlamaIndex, and Haystack—while Rust developers are left with separate, unconnected crates (such as hnsw_rs, tantivy, and candle). This means there’s no single, unified toolkit in Rust for building retrieval pipelines, fine-tuning adapters, and evaluating how retrieval and generation work together.

We are motivated to bridge this gap by building RustRAGLab (RRL), a Rust-native framework designed to support the full lifecycle of RAG systems, with the following in mind:
1. It enables safe, performant, and low-overhead pipelines using Rust’s strengths.
2. It fills a real gap in the Rust machine learning ecosystem.
3. It provides both CLI and library APIs, lowering barriers for developers who want to experiment with RAG without dropping back into Python. 

Our team aims to deliver an MVP that can ingest documents, build retrieval pipelines, fine-tune LoRA adapters with grounding-aware objectives, and evaluate retrieval+generation in a unified way.

## Objective and key features
### Objective:
Design and implement a end-to-end Rust-native framework that integrates retrieval, adapter fine-tuning, and evaluation for RAG systems, offering both developer APIs and a CLI tool (rrl) for streamlined use.

Data & Chunking: PDF/Markdown loaders → text → chunkers (fixed, overlap, semantic).

### Key Features:

- **Data & Chunking**
  - Load documents from PDF and Markdown.  
  - Chunk text using fixed-size, overlapping, or semantic-based strategies.  

- **Embeddings**
  - Trait-based embedding interface.  
  - Backends via `tch` (Torch bindings) and `onnxruntime`.  
  - Hardware acceleration support for **CUDA** (NVIDIA GPUs) and **Metal** (Apple GPUs).  

- **Indexing & Retrieval**
  - Dense retrieval using **HNSW** (`hnsw_rs`).  
  - Sparse retrieval using **BM25** (`tantivy`).  
  - Hybrid retriever that combines dense and sparse signals.  

- **RAG-Aware Training**
  - LoRA and QLoRA fine-tuning using **Candle**.  
  - Supports CUDA and Metal backends for efficient training.  
  - **Grounding-aware loss**: encourages responses that cite retrieved passages and penalizes hallucination.  
  - Optional contrastive training with hard negatives.  

- **Evaluation**
  - Retrieval metrics: Recall@k, Mean Reciprocal Rank (MRR).
  - Generation metrics: Perplexity, Exact Match, F1, ROUGE-L.
  - Attribution score: fraction of answers that explicitly cite retrieved documents.

- **CLI (`rrl`)**
  - `rrl ingest`: parse and chunk documents, build indexes.
  - `rrl retrain`: fine-tune LoRA/QLoRA adapters.
  - `rrl eval`: compute retrieval and generation metrics.
  - `rrl serve`: launch Axum-based inference server with streaming responses and hot-reload for new documents.

- **Server & Dashboard**
  - REST API via **Axum**, with streaming output for interactive inference.
  - Index hot-reloading without downtime.
  - Terminal UI with **ratatui** for visualizing retrieval stats and training progress.

## Tentative Plan

The project will be implemented as **modular crates** for clarity and maintainability. This modularity allows parallel work and makes the framework easier to extend after the course.

### Crate Structure

- `rrl-core`: Core traits, chunkers, embeddings, retrievers.
- `rrl-train`: LoRA/QLoRA adapter fine-tuning with Candle.
- `rrl-eval`: Retrieval + generation metrics.
- `rrl-cli`: CLI interface, YAML/Serde configuration.
- `rrl-serve`: Axum inference API with streaming and hot-reload.

### Team Roles

- **Kevin Lin (Retrieval & Evaluation)**  
  - Implement loaders and chunking strategies.  
  - Build HNSW + BM25 hybrid retriever.  
  - Develop evaluation metrics (retrieval + attribution).  

- **Liz Zhu (Training & CLI/Server)**  
  - Implement LoRA/QLoRA fine-tuning in Candle (CUDA + Metal).  
  - Add grounding-aware loss function.  
  - Build CLI commands (`ingest`, `retrain`, `eval`, `serve`).  
  - Implement Axum server with hot-reload.  

### Development Strategy

- **Week 1–2**: Set up repo and crate skeletons; implement loaders and chunking.  
- **Week 3–4**: Add embeddings + hybrid retrieval pipeline.  
- **Week 5–6**: Implement LoRA/QLoRA fine-tuning with grounding-aware loss.  
- **Week 7–8**: Implement evaluation crate (retrieval + generation).  
- **Week 9**: Integrate CLI + server.  
- **Week 10**: Testing, documentation, and polish.  
