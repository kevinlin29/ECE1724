//! Terminal User Interface for RustRAGLab
//!
//! Provides an interactive TUI for managing the RAG pipeline:
//! - Pipeline visualization (Ingest → Embed → Index → Query)
//! - Interactive query interface
//! - Real-time statistics and monitoring
//! - Log viewing
//! - Training configuration and monitoring
//! - RAG question-answering with local LLMs

pub mod app;
pub mod ui;
pub mod events;
pub mod training;

#[cfg(feature = "training")]
pub mod rag;

pub use app::{App, run_tui};
pub use training::{run_training_tui, TrainingAppConfig, TrainingResults};

#[cfg(feature = "training")]
pub use rag::{RagApp, RagAppConfig};
