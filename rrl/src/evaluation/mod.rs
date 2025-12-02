//! Evaluation metrics
//!
//! Provides comprehensive metrics for retrieval (Recall@k, MRR, NDCG, MAP) and
//! generation (F1, EM, ROUGE-L, perplexity).

pub mod generation;
pub mod retrieval;

// Re-exports
pub use retrieval::{QueryResult, RetrievalEvaluator, RetrievalMetrics, SingleQueryMetrics};
