//! Shared utilities for encoder models

use anyhow::Result;
use candle_core::{Tensor, D};

use super::super::common::PoolingStrategy;

/// Apply pooling to hidden states
pub fn apply_pooling(
    hidden_states: &Tensor,
    attention_mask: &Tensor,
    strategy: PoolingStrategy,
) -> Result<Tensor> {
    match strategy {
        PoolingStrategy::Mean => mean_pool(hidden_states, attention_mask),
        PoolingStrategy::Cls => cls_pool(hidden_states),
        PoolingStrategy::Max => max_pool(hidden_states, attention_mask),
        PoolingStrategy::Last => last_pool(hidden_states),
    }
}

/// Mean pooling over non-padding tokens
fn mean_pool(hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    let mask = attention_mask.unsqueeze(2)?.to_dtype(hidden_states.dtype())?;
    let masked = hidden_states.broadcast_mul(&mask)?;
    let sum = masked.sum(1)?;
    let count = mask.sum(1)?.clamp(1e-9, f64::MAX)?;
    Ok(sum.broadcast_div(&count)?)
}

/// CLS token pooling
fn cls_pool(hidden_states: &Tensor) -> Result<Tensor> {
    Ok(hidden_states.narrow(1, 0, 1)?.squeeze(1)?)
}

/// Last token pooling (for GPT-style models)
fn last_pool(hidden_states: &Tensor) -> Result<Tensor> {
    let seq_len = hidden_states.dim(1)?;
    Ok(hidden_states.narrow(1, seq_len - 1, 1)?.squeeze(1)?)
}

/// Max pooling
fn max_pool(hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    let mask = attention_mask.unsqueeze(2)?.to_dtype(hidden_states.dtype())?;
    let neg_inf = Tensor::full(-1e9f32, hidden_states.shape(), hidden_states.device())?;
    let masked = hidden_states
        .broadcast_mul(&mask)?
        .broadcast_add(&neg_inf.broadcast_mul(&(1.0 - &mask)?)?)?;
    Ok(masked.max(1)?)
}

/// Normalize embeddings to unit length
pub fn normalize_embeddings(embeddings: &Tensor) -> Result<Tensor> {
    let norm = embeddings
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .sqrt()?;
    Ok(embeddings.broadcast_div(&norm.clamp(1e-12, f64::MAX)?)?)
}

/// Compute cosine similarity
pub fn cosine_similarity(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_norm = normalize_embeddings(a)?;
    let b_norm = normalize_embeddings(b)?;
    Ok(a_norm.matmul(&b_norm.t()?)?)
}