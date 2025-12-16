//! # RustRAGLab (RRL)
//!
//! A Rust framework for RAG-aware fine-tuning and evaluation.
//!
//! ## Overview
//!
//! RustRAGLab provides a unified, performant, and safe Rust framework for building
//! and evaluating Retrieval-Augmented Generation (RAG) systems with support for:
//!
//! - Document ingestion and chunking
//! - Embedding generation with GPU acceleration
//! - Dense and sparse retrieval (HNSW + BM25)
//! - LoRA/QLoRA fine-tuning with grounding-aware loss
//! - Comprehensive evaluation metrics
//! - REST API server with streaming responses
//!
//! ## Architecture
//!
//! The framework is organized into modular components:
//!
//! - `data` - Document loading and chunking
//! - `embedding` - Embedding generation and caching
//! - `retrieval` - Dense, sparse, and hybrid retrieval
//! - `training` - LoRA fine-tuning with Candle
//! - `evaluation` - Retrieval and generation metrics
//! - `server` - Axum-based REST API
//! - `cli` - Command-line interface
//! - `utils` - Common utilities

// Core modules
pub mod data;
pub mod embedding;
pub mod retrieval;
pub mod evaluation;
pub mod server;
pub mod cli;
pub mod utils;

// Training module (requires candle)
#[cfg(feature = "training")]
pub mod training;

// RAG pipeline module
#[cfg(feature = "training")]
pub mod rag;

// Re-export commonly used types
pub use anyhow::{Error, Result};
