//! BERT-family model wrappers for sentence embeddings
//!
//! Provides unified interfaces for BERT, RoBERTa, DistilBERT, and ALBERT models.

mod bert;
mod bert_cuda;
mod bert_lora;
mod tokenizer;

pub use bert::BertForEmbedding;
pub use bert_cuda::{CudaBertConfig, CudaBertModel, CudaLayerNorm};
pub use bert_lora::{load_bert_lora, BertLoraModel};
pub use tokenizer::TokenizerWrapper;

use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};

use super::hub::{HubModelConfig, ModelLoader};

/// Supported model architectures
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArchitecture {
    Bert,
    Roberta,
    DistilBert,
    Albert,
}

impl std::fmt::Display for ModelArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelArchitecture::Bert => write!(f, "bert"),
            ModelArchitecture::Roberta => write!(f, "roberta"),
            ModelArchitecture::DistilBert => write!(f, "distilbert"),
            ModelArchitecture::Albert => write!(f, "albert"),
        }
    }
}

/// Trait for embedding models
///
/// All BERT-family models implement this trait for generating sentence embeddings.
pub trait EmbeddingModel: Send + Sync {
    /// Forward pass to get embeddings
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch_size, seq_len]
    /// * `attention_mask` - Attention mask [batch_size, seq_len]
    ///
    /// # Returns
    /// * Embeddings tensor [batch_size, hidden_size]
    fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor>;

    /// Get the hidden size (embedding dimension)
    fn hidden_size(&self) -> usize;

    /// Get the model architecture type
    fn architecture(&self) -> ModelArchitecture;

    /// Get the device this model is on
    fn device(&self) -> &Device;
}

/// Load a model from HuggingFace Hub or local path
///
/// # Arguments
/// * `model_id_or_path` - Model ID (e.g., "bert-base-uncased") or local path
/// * `device` - Device to load the model on
///
/// # Returns
/// * Box<dyn EmbeddingModel> - The loaded model
///
/// # Example
/// ```ignore
/// let model = load_model("bert-base-uncased", &device)?;
/// let embeddings = model.forward(&input_ids, &attention_mask)?;
/// ```
pub fn load_model(model_id_or_path: &str, device: &Device) -> Result<Box<dyn EmbeddingModel>> {
    let loader = ModelLoader::new()?;
    let model_path = loader.load_model_path(model_id_or_path)?;
    let config = HubModelConfig::from_file(&model_path.config_file)?;

    // Validate config
    config.validate_bert_compatibility()?;

    // Get model type
    let model_type = config.get_model_type().ok_or_else(|| {
        anyhow!(
            "Could not determine model type from config. Architectures: {:?}",
            config.architectures
        )
    })?;

    tracing::info!(
        "Loading {} model: {} (hidden_size: {}, layers: {})",
        model_type,
        model_id_or_path,
        config.hidden_size.unwrap_or(0),
        config.num_hidden_layers.unwrap_or(0)
    );

    match model_type.as_str() {
        "bert" => {
            let model = BertForEmbedding::from_model_path(&model_path, device)?;
            Ok(Box::new(model))
        }
        "roberta" | "xlm-roberta" | "camembert" => {
            // RoBERTa uses the same architecture as BERT with minor differences
            // For now, we use BERT loader which works for most cases
            tracing::info!("Loading RoBERTa-style model using BERT architecture");
            let model = BertForEmbedding::from_model_path(&model_path, device)?;
            Ok(Box::new(model))
        }
        "distilbert" => {
            // DistilBERT has a similar structure but fewer layers
            tracing::info!("Loading DistilBERT model using BERT architecture");
            let model = BertForEmbedding::from_model_path(&model_path, device)?;
            Ok(Box::new(model))
        }
        "albert" => {
            // ALBERT uses parameter sharing - structure is different
            // For now, we use BERT loader with a warning
            tracing::warn!("ALBERT support is experimental - using BERT loader");
            let model = BertForEmbedding::from_model_path(&model_path, device)?;
            Ok(Box::new(model))
        }
        _ => Err(anyhow!(
            "Unsupported model type: {}. Supported: bert, roberta, distilbert, albert",
            model_type
        )),
    }
}

/// Load a model with a specific architecture
pub fn load_model_with_arch(
    model_id_or_path: &str,
    arch: ModelArchitecture,
    device: &Device,
) -> Result<Box<dyn EmbeddingModel>> {
    let loader = ModelLoader::new()?;
    let model_path = loader.load_model_path(model_id_or_path)?;

    match arch {
        ModelArchitecture::Bert
        | ModelArchitecture::Roberta
        | ModelArchitecture::DistilBert
        | ModelArchitecture::Albert => {
            let model = BertForEmbedding::from_model_path(&model_path, device)?;
            Ok(Box::new(model))
        }
    }
}

/// Pooling strategies for converting hidden states to sentence embeddings
#[derive(Debug, Clone, Copy, Default)]
pub enum PoolingStrategy {
    /// Mean pooling over non-padding tokens (recommended)
    #[default]
    Mean,
    /// Use [CLS] token embedding
    Cls,
    /// Max pooling over all tokens
    Max,
}

/// Apply pooling to hidden states
///
/// # Arguments
/// * `hidden_states` - Hidden states from model [batch_size, seq_len, hidden_size]
/// * `attention_mask` - Attention mask [batch_size, seq_len]
/// * `strategy` - Pooling strategy to use
///
/// # Returns
/// * Pooled embeddings [batch_size, hidden_size]
pub fn apply_pooling(
    hidden_states: &Tensor,
    attention_mask: &Tensor,
    strategy: PoolingStrategy,
) -> Result<Tensor> {
    match strategy {
        PoolingStrategy::Mean => mean_pool(hidden_states, attention_mask),
        PoolingStrategy::Cls => cls_pool(hidden_states),
        PoolingStrategy::Max => max_pool(hidden_states, attention_mask),
    }
}

/// Mean pooling over non-padding tokens
fn mean_pool(hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    // Expand mask to hidden size: [batch, seq, 1] -> [batch, seq, hidden]
    let mask = attention_mask
        .unsqueeze(2)?
        .to_dtype(hidden_states.dtype())?;

    // Apply mask and sum
    let masked = hidden_states.broadcast_mul(&mask)?;
    let sum = masked.sum(1)?;

    // Count non-padding tokens per sample
    let count = mask.sum(1)?.clamp(1e-9, f64::MAX)?;

    // Divide by count
    Ok(sum.broadcast_div(&count)?)
}

/// CLS token pooling (first token)
fn cls_pool(hidden_states: &Tensor) -> Result<Tensor> {
    // Select first token: [batch, seq, hidden] -> [batch, hidden]
    Ok(hidden_states.narrow(1, 0, 1)?.squeeze(1)?)
}

/// Max pooling over all tokens
fn max_pool(hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    // Create large negative values for padding tokens
    let mask = attention_mask
        .unsqueeze(2)?
        .to_dtype(hidden_states.dtype())?;
    let neg_inf = Tensor::full(-1e9f32, hidden_states.shape(), hidden_states.device())?;
    let masked = hidden_states
        .broadcast_mul(&mask)?
        .broadcast_add(&neg_inf.broadcast_mul(&(1.0 - &mask)?)?)?;

    // Max over sequence dimension
    Ok(masked.max(1)?)
}

/// Normalize embeddings to unit length
pub fn normalize_embeddings(embeddings: &Tensor) -> Result<Tensor> {
    let norm = embeddings
        .sqr()?
        .sum_keepdim(candle_core::D::Minus1)?
        .sqrt()?;
    Ok(embeddings.broadcast_div(&norm.clamp(1e-12, f64::MAX)?)?)
}

/// Compute cosine similarity between two embedding tensors
pub fn cosine_similarity(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_norm = normalize_embeddings(a)?;
    let b_norm = normalize_embeddings(b)?;
    Ok(a_norm.matmul(&b_norm.t()?)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_architecture_display() {
        assert_eq!(ModelArchitecture::Bert.to_string(), "bert");
        assert_eq!(ModelArchitecture::Roberta.to_string(), "roberta");
        assert_eq!(ModelArchitecture::DistilBert.to_string(), "distilbert");
        assert_eq!(ModelArchitecture::Albert.to_string(), "albert");
    }
}
