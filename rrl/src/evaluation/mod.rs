//! Evaluation metrics
//!
//! Provides comprehensive metrics for retrieval (Recall@k, MRR, NDCG, MAP) and
//! generation (F1, EM, ROUGE-L).

pub mod generation;
pub mod retrieval;

// Re-exports
pub use retrieval::{QueryResult, RetrievalEvaluator, RetrievalMetrics, SingleQueryMetrics};
pub use generation::{
    GenerationMetrics, RagEvaluationResult, evaluate_generation_batch,
};
