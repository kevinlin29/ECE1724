//! Evaluation metrics
//!
//! Provides comprehensive metrics for retrieval (Recall@k, MRR, NDCG, MAP) and
//! generation (F1, EM, ROUGE-L).

pub mod generation;
pub mod msmarco;
pub mod retrieval;

// Re-exports
pub use retrieval::{QueryResult, RetrievalEvaluator, RetrievalMetrics, SingleQueryMetrics};
pub use generation::{
    GenerationMetrics, RagEvaluationResult, evaluate_generation_batch,
};
pub use msmarco::{
    MsMarcoExample, MsMarcoEvalConfig, MsMarcoEvaluator, MsMarcoMetrics,
    EvalProgress, load_msmarco_examples, cosine_similarity, rank_by_similarity,
};
