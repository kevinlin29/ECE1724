//! Terminal User Interface for RustRAGLab
//!
//! Provides an interactive TUI for managing the RAG pipeline:
//! - Pipeline visualization (Ingest → Embed → Index → Query)
//! - Interactive query interface
//! - Real-time statistics and monitoring
//! - Log viewing
//! - Training configuration and monitoring

pub mod app;
pub mod ui;
pub mod events;
pub mod training;

pub use app::{App, run_tui};
pub use training::{run_training_tui, TrainingAppConfig, TrainingResults};
