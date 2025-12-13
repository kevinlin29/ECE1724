//! BERT model wrapper for sentence embeddings
//!
//! Uses candle-transformers' BERT implementation with mean pooling.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};

// use super::{apply_pooling, EmbeddingModel, ModelArchitecture, PoolingStrategy};
use super::super::common::{EmbeddingModel, ModelArchitecture, PoolingStrategy};
use super::encoder_utils::apply_pooling;
use crate::training::hub::ModelPath;

/// BERT model configured for generating sentence embeddings
pub struct BertForEmbedding {
    model: BertModel,
    config: BertConfig,
    device: Device,
    pooling: PoolingStrategy,
}

impl BertForEmbedding {
    /// Load a BERT model from a ModelPath
    pub fn from_model_path(model_path: &ModelPath, device: &Device) -> Result<Self> {
        model_path.validate()?;

        // Load config
        let config_str = std::fs::read_to_string(&model_path.config_file)
            .context("Failed to read config.json")?;
        let config: BertConfig =
            serde_json::from_str(&config_str).context("Failed to parse BERT config")?;

        tracing::debug!(
            "BERT config: hidden_size={}, layers={}, heads={}",
            config.hidden_size,
            config.num_hidden_layers,
            config.num_attention_heads
        );

        // Load weights
        let vb = if model_path.is_safetensors() {
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[&model_path.weights_file], DType::F32, device)
                    .context("Failed to load safetensors weights")?
            }
        } else {
            // For pytorch bin files, we need different loading logic
            // For now, only support safetensors
            return Err(anyhow::anyhow!(
                "Only safetensors format is currently supported. \
                Please use a model with model.safetensors file."
            ));
        };

        // Create model
        let model =
            BertModel::load(vb, &config).context("Failed to initialize BERT model from weights")?;

        tracing::info!(
            "Loaded BERT model: {} layers, {} hidden size",
            config.num_hidden_layers,
            config.hidden_size
        );

        Ok(Self {
            model,
            config,
            device: device.clone(),
            pooling: PoolingStrategy::Mean,
        })
    }

    /// Load a BERT model directly from HuggingFace
    pub fn from_pretrained(model_id: &str, device: &Device) -> Result<Self> {
        use crate::training::hub::ModelLoader;
        let loader = ModelLoader::new()?;
        let model_path = loader.load_model_path(model_id)?;
        Self::from_model_path(&model_path, device)
    }

    /// Set the pooling strategy
    pub fn with_pooling(mut self, pooling: PoolingStrategy) -> Self {
        self.pooling = pooling;
        self
    }

    /// Get the BERT config
    pub fn config(&self) -> &BertConfig {
        &self.config
    }

    /// Forward pass returning the last hidden states
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch_size, seq_len]
    /// * `attention_mask` - Attention mask [batch_size, seq_len] (optional)
    /// * `token_type_ids` - Token type IDs [batch_size, seq_len] (optional)
    ///
    /// # Returns
    /// * Hidden states tensor [batch_size, seq_len, hidden_size]
    pub fn forward_hidden(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Create default token_type_ids if not provided
        let default_type_ids;
        let type_ids = match token_type_ids {
            Some(ids) => ids,
            None => {
                default_type_ids = Tensor::zeros(input_ids.shape(), DType::U32, &self.device)?;
                &default_type_ids
            }
        };

        // Forward through BERT
        // The forward method returns the last hidden state
        let hidden_states = self.model.forward(input_ids, type_ids, attention_mask)?;

        Ok(hidden_states)
    }

    /// Encode texts to embeddings
    ///
    /// Note: This requires a tokenizer. Use TokenizerWrapper for full encoding.
    pub fn encode_tensors(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor> {
        // Get hidden states
        let hidden_states = self.forward_hidden(input_ids, Some(attention_mask), None)?;

        // Apply pooling
        apply_pooling(&hidden_states, attention_mask, self.pooling)
    }
}

impl EmbeddingModel for BertForEmbedding {
    fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        self.encode_tensors(input_ids, attention_mask)
    }

    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    fn architecture(&self) -> ModelArchitecture {
        ModelArchitecture::Bert
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require downloading models from HuggingFace
    // They are ignored by default and can be run with:
    // cargo test --features training -- --ignored

    #[test]
    #[ignore]
    fn test_bert_load() {
        let device = Device::Cpu;
        let model = BertForEmbedding::from_pretrained("bert-base-uncased", &device);
        assert!(model.is_ok(), "Failed to load BERT: {:?}", model.err());

        let model = model.unwrap();
        assert_eq!(model.hidden_size(), 768);
        assert_eq!(model.architecture(), ModelArchitecture::Bert);
    }
}
