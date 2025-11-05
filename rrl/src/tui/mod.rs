//! Terminal User Interface for RustRAGLab
//!
//! Provides an interactive TUI for managing the RAG pipeline:
//! - Pipeline visualization (Ingest → Embed → Index → Query)
//! - Interactive query interface
//! - Real-time statistics and monitoring
//! - Log viewing

pub mod app;
pub mod ui;
pub mod events;

pub use app::{App, run_tui};
