//! Loss functions for training embedding models
//!
//! Includes:
//! - Contrastive loss (InfoNCE) for learning discriminative embeddings
//! - In-batch negatives for efficient training
//! - Grounding-aware loss for RAG alignment

#[cfg(feature = "training")]
use anyhow::Result;
#[cfg(feature = "training")]
use candle_core::{DType, Tensor, D};

/// Contrastive loss configuration
#[derive(Debug, Clone)]
pub struct ContrastiveLossConfig {
    /// Temperature for softmax (lower = sharper distribution)
    pub temperature: f32,
    /// Whether to use in-batch negatives
    pub in_batch_negatives: bool,
}

impl Default for ContrastiveLossConfig {
    fn default() -> Self {
        Self {
            temperature: 0.05,
            in_batch_negatives: true,
        }
    }
}

/// Contrastive loss (InfoNCE)
///
/// Pulls positive pairs together and pushes negative pairs apart.
#[derive(Debug, Clone)]
pub struct ContrastiveLoss {
    config: ContrastiveLossConfig,
}

impl ContrastiveLoss {
    /// Create a new contrastive loss
    pub fn new(config: ContrastiveLossConfig) -> Self {
        Self { config }
    }

    /// Create with default config
    pub fn default_config() -> Self {
        Self::new(ContrastiveLossConfig::default())
    }

    /// Get the temperature
    pub fn temperature(&self) -> f32 {
        self.config.temperature
    }

    /// Compute InfoNCE loss
    ///
    /// # Arguments
    /// * `query` - Query embeddings [batch_size, hidden_dim]
    /// * `positives` - Positive embeddings [batch_size, hidden_dim]
    /// * `negatives` - Negative embeddings [batch_size, num_negatives, hidden_dim] (optional)
    ///
    /// # Returns
    /// * Loss scalar tensor
    #[cfg(feature = "training")]
    pub fn forward(
        &self,
        query: &Tensor,
        positives: &Tensor,
        negatives: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Normalize embeddings to unit vectors
        let q = normalize(query)?;
        let p = normalize(positives)?;

        let temperature = self.config.temperature as f64;

        if self.config.in_batch_negatives && negatives.is_none() {
            // Use in-batch negatives: each other sample in the batch is a negative
            return self.in_batch_loss(&q, &p, temperature);
        }

        // Standard InfoNCE with explicit negatives
        // Positive similarity: [batch_size]
        let pos_sim = element_wise_dot(&q, &p)?;
        let pos_sim = (pos_sim / temperature)?;

        if let Some(neg) = negatives {
            let n = normalize_3d(neg)?;

            // Negative similarities: [batch_size, num_negatives]
            // q: [batch, hidden] -> [batch, 1, hidden]
            // n: [batch, num_neg, hidden]
            // result: [batch, 1, num_neg] -> [batch, num_neg]
            let q_expanded = q.unsqueeze(1)?;
            let neg_sim = q_expanded.matmul(&n.transpose(1, 2)?)?.squeeze(1)?;
            let neg_sim = (neg_sim / temperature)?;

            // Concatenate: [batch, 1 + num_negatives]
            let logits = Tensor::cat(&[pos_sim.unsqueeze(1)?, neg_sim], 1)?;

            // Labels: positive is always at index 0
            let batch_size = query.dim(0)?;
            let labels = Tensor::zeros((batch_size,), DType::U32, query.device())?;

            // Cross-entropy loss
            cross_entropy(&logits, &labels)
        } else {
            // No negatives provided and in_batch_negatives is false
            // Use simple loss: maximize positive similarity
            let loss = (1.0 - pos_sim)?.mean_all()?;
            Ok(loss)
        }
    }

    /// Compute in-batch contrastive loss
    ///
    /// Each sample uses all other samples in the batch as negatives.
    #[cfg(feature = "training")]
    fn in_batch_loss(&self, q: &Tensor, p: &Tensor, temperature: f64) -> Result<Tensor> {
        let batch_size = q.dim(0)?;

        // Similarity matrix: [batch, batch]
        // Each row i contains similarities between q[i] and all positives
        let sim_matrix = q.matmul(&p.t()?)?;
        let sim_matrix = (sim_matrix / temperature)?;

        // Labels: diagonal elements are the positives
        let labels = Tensor::arange(0u32, batch_size as u32, q.device())?;

        cross_entropy(&sim_matrix, &labels)
    }
}

/// Normalize embeddings to unit length
#[cfg(feature = "training")]
fn normalize(embeddings: &Tensor) -> Result<Tensor> {
    let norm = embeddings.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    let norm = norm.clamp(1e-12, f64::MAX)?;
    Ok(embeddings.broadcast_div(&norm)?)
}

/// Normalize 3D tensor along last dimension
#[cfg(feature = "training")]
fn normalize_3d(embeddings: &Tensor) -> Result<Tensor> {
    let norm = embeddings.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    let norm = norm.clamp(1e-12, f64::MAX)?;
    Ok(embeddings.broadcast_div(&norm)?)
}

/// Element-wise dot product between two tensors
#[cfg(feature = "training")]
fn element_wise_dot(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // (a * b).sum(-1)
    Ok((a * b)?.sum(D::Minus1)?)
}

/// Cross-entropy loss
#[cfg(feature = "training")]
fn cross_entropy(logits: &Tensor, labels: &Tensor) -> Result<Tensor> {
    let log_softmax = candle_nn::ops::log_softmax(logits, D::Minus1)?;
    let labels_i64 = labels.to_dtype(DType::I64)?;
    Ok(candle_nn::loss::nll(&log_softmax, &labels_i64)?)
}

/// Triplet loss configuration
#[derive(Debug, Clone)]
pub struct TripletLossConfig {
    /// Margin for triplet loss
    pub margin: f32,
}

impl Default for TripletLossConfig {
    fn default() -> Self {
        Self { margin: 0.2 }
    }
}

/// Triplet margin loss
///
/// Ensures that anchor is closer to positive than to negative by a margin.
#[derive(Debug, Clone)]
pub struct TripletLoss {
    config: TripletLossConfig,
}

impl TripletLoss {
    /// Create a new triplet loss
    pub fn new(config: TripletLossConfig) -> Self {
        Self { config }
    }

    /// Compute triplet loss
    ///
    /// # Arguments
    /// * `anchor` - Anchor embeddings [batch_size, hidden_dim]
    /// * `positive` - Positive embeddings [batch_size, hidden_dim]
    /// * `negative` - Negative embeddings [batch_size, hidden_dim]
    ///
    /// # Returns
    /// * Loss scalar tensor
    #[cfg(feature = "training")]
    pub fn forward(
        &self,
        anchor: &Tensor,
        positive: &Tensor,
        negative: &Tensor,
    ) -> Result<Tensor> {
        // Normalize
        let a = normalize(anchor)?;
        let p = normalize(positive)?;
        let n = normalize(negative)?;

        // Cosine similarities
        let pos_sim = element_wise_dot(&a, &p)?;
        let neg_sim = element_wise_dot(&a, &n)?;

        // Loss = max(0, margin - pos_sim + neg_sim)
        let margin = self.config.margin as f64;
        let loss = (margin - &pos_sim + &neg_sim)?;
        let loss = loss.clamp(0.0, f64::MAX)?;
        Ok(loss.mean_all()?)
    }
}

/// Combined RAG loss
///
/// Combines contrastive loss with grounding-aware regularization.
#[derive(Debug, Clone)]
pub struct RagLoss {
    /// Weight for contrastive loss
    pub contrastive_weight: f32,
    /// Weight for grounding loss
    pub grounding_weight: f32,
    /// Contrastive loss
    contrastive: ContrastiveLoss,
}

impl RagLoss {
    /// Create a new RAG loss
    pub fn new(contrastive_weight: f32, grounding_weight: f32, temperature: f32) -> Self {
        Self {
            contrastive_weight,
            grounding_weight,
            contrastive: ContrastiveLoss::new(ContrastiveLossConfig {
                temperature,
                in_batch_negatives: true,
            }),
        }
    }

    /// Compute combined RAG loss
    ///
    /// # Arguments
    /// * `query` - Query embeddings
    /// * `positives` - Positive context embeddings
    /// * `negatives` - Negative context embeddings (optional)
    ///
    /// # Returns
    /// * `RagLossOutput` with breakdown of losses
    #[cfg(feature = "training")]
    pub fn forward(
        &self,
        query: &Tensor,
        positives: &Tensor,
        negatives: Option<&Tensor>,
    ) -> Result<RagLossOutput> {
        // Contrastive loss
        let contrastive_loss = self.contrastive.forward(query, positives, negatives)?;

        // Grounding loss: MSE between query and mean of positives
        // Encourages query to be close to its retrieved contexts
        let grounding_loss = self.grounding_loss(query, positives)?;

        // Weighted sum
        let total_loss = ((&contrastive_loss * self.contrastive_weight as f64)?
            + (&grounding_loss * self.grounding_weight as f64)?)?;

        Ok(RagLossOutput {
            total_loss,
            contrastive_loss: contrastive_loss.to_scalar::<f32>()?,
            grounding_loss: grounding_loss.to_scalar::<f32>()?,
        })
    }

    /// Grounding loss: MSE between query and mean positive
    #[cfg(feature = "training")]
    fn grounding_loss(&self, query: &Tensor, positives: &Tensor) -> Result<Tensor> {
        let q = normalize(query)?;
        let p = normalize(positives)?;

        // MSE loss
        let diff = (&q - &p)?;
        let mse = diff.sqr()?.mean_all()?;
        Ok(mse)
    }
}

/// Output from RAG loss computation
#[cfg(feature = "training")]
#[derive(Debug, Clone)]
pub struct RagLossOutput {
    /// Total weighted loss
    pub total_loss: Tensor,
    /// Contrastive loss component
    pub contrastive_loss: f32,
    /// Grounding loss component
    pub grounding_loss: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contrastive_loss_config() {
        let config = ContrastiveLossConfig::default();
        assert!((config.temperature - 0.05).abs() < 1e-6);
        assert!(config.in_batch_negatives);
    }

    #[test]
    fn test_triplet_loss_config() {
        let config = TripletLossConfig::default();
        assert!((config.margin - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_rag_loss_creation() {
        let loss = RagLoss::new(1.0, 0.1, 0.05);
        assert!((loss.contrastive_weight - 1.0).abs() < 1e-6);
        assert!((loss.grounding_weight - 0.1).abs() < 1e-6);
    }
}
