//! BERT model with trainable LoRA adapters
//!
//! This module provides a BERT model with LoRA (Low-Rank Adaptation) for
//! parameter-efficient fine-tuning. The base BERT weights remain frozen
//! while small LoRA adapters are trained.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Init, VarBuilder, VarMap};
use std::path::Path;

use super::bert_cuda::{CudaBertConfig, CudaBertModel};
use super::{apply_pooling, normalize_embeddings, EmbeddingModel, ModelArchitecture, PoolingStrategy};
use crate::training::hub::ModelPath;
use crate::training::lora::LoraConfig;

/// BERT model with LoRA adapters for fine-tuning
///
/// The model architecture:
/// - Base BERT encoder (frozen weights) - Uses CUDA-compatible implementation
/// - LoRA projection applied after pooling (trainable)
///
/// This approach is simpler than injecting LoRA into attention layers
/// but still enables effective task-specific adaptation.
pub struct BertLoraModel {
    /// Base BERT model (frozen) - CUDA-compatible implementation
    base_model: CudaBertModel,
    /// BERT configuration
    config: CudaBertConfig,
    /// Device
    device: Device,
    /// Pooling strategy
    pooling: PoolingStrategy,
    /// LoRA down projection: hidden_size -> rank
    lora_down: Tensor,
    /// LoRA up projection: rank -> hidden_size
    lora_up: Tensor,
    /// LoRA scaling factor
    lora_scaling: f32,
    /// LoRA rank
    lora_rank: usize,
}

impl BertLoraModel {
    /// Create a new BERT model with LoRA from a model path
    ///
    /// # Arguments
    /// * `model_path` - Path to the pretrained model files
    /// * `lora_config` - LoRA configuration
    /// * `var_map` - VarMap for trainable parameters (LoRA weights will be registered here)
    /// * `device` - Device to load the model on
    pub fn from_model_path(
        model_path: &ModelPath,
        lora_config: &LoraConfig,
        var_map: &VarMap,
        device: &Device,
    ) -> Result<Self> {
        model_path.validate()?;

        // Load BERT config
        let config_str = std::fs::read_to_string(&model_path.config_file)
            .context("Failed to read config.json")?;
        let config: CudaBertConfig =
            serde_json::from_str(&config_str).context("Failed to parse BERT config")?;

        tracing::info!(
            "Loading BERT with LoRA: hidden_size={}, layers={}, lora_rank={}",
            config.hidden_size,
            config.num_hidden_layers,
            lora_config.rank
        );

        // Load base BERT weights (frozen)
        // Some HuggingFace models have "bert." prefix, some don't
        // We try without prefix first, then with prefix
        let base_vb = if model_path.is_safetensors() {
            unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &[&model_path.weights_file],
                    DType::F32,
                    device,
                )
                .context("Failed to load safetensors weights")?
            }
        } else {
            return Err(anyhow::anyhow!(
                "Only safetensors format is currently supported"
            ));
        };

        // Try loading without prefix first, then with "bert" prefix
        // Our CudaBertModel is CUDA-compatible (uses basic ops for layer norm)
        let base_model = CudaBertModel::load(base_vb.clone(), &config)
            .or_else(|_| {
                tracing::info!("Trying to load with 'bert' prefix...");
                CudaBertModel::load(base_vb.pp("bert"), &config)
            })
            .context("Failed to initialize BERT model")?;

        // Create trainable LoRA parameters
        let hidden_size = config.hidden_size;
        let rank = lora_config.rank;

        // Create VarBuilder from VarMap for trainable parameters
        let lora_vb = VarBuilder::from_varmap(var_map, DType::F32, device);

        // LoRA down projection: hidden_size -> rank (Kaiming init)
        let lora_down = lora_vb.get_with_hints(
            (rank, hidden_size),
            "lora_projection.down",
            Init::Kaiming {
                dist: candle_nn::init::NormalOrUniform::Uniform,
                fan: candle_nn::init::FanInOut::FanIn,
                non_linearity: candle_nn::init::NonLinearity::Linear,
            },
        )?;

        // LoRA up projection: rank -> hidden_size (zeros init)
        let lora_up = lora_vb.get_with_hints(
            (hidden_size, rank),
            "lora_projection.up",
            Init::Const(0.0),
        )?;

        let lora_scaling = lora_config.scaling();

        tracing::info!(
            "Created LoRA adapters: {} trainable parameters (rank={})",
            rank * hidden_size * 2,
            rank
        );

        Ok(Self {
            base_model,
            config,
            device: device.clone(),
            pooling: PoolingStrategy::Mean,
            lora_down,
            lora_up,
            lora_scaling,
            lora_rank: rank,
        })
    }

    /// Load from HuggingFace model ID
    pub fn from_pretrained(
        model_id: &str,
        lora_config: &LoraConfig,
        var_map: &VarMap,
        device: &Device,
    ) -> Result<Self> {
        use crate::training::hub::ModelLoader;
        let loader = ModelLoader::new()?;
        let model_path = loader.load_model_path(model_id)?;
        Self::from_model_path(&model_path, lora_config, var_map, device)
    }

    /// Set pooling strategy
    pub fn with_pooling(mut self, pooling: PoolingStrategy) -> Self {
        self.pooling = pooling;
        self
    }

    /// Get the BERT config
    pub fn config(&self) -> &CudaBertConfig {
        &self.config
    }

    /// Get number of trainable LoRA parameters
    pub fn num_trainable_params(&self) -> usize {
        self.lora_rank * self.config.hidden_size * 2
    }

    /// Get total model parameters (approximate)
    pub fn num_total_params(&self) -> usize {
        // BERT-base: ~110M parameters
        // This is an approximation
        let bert_params = self.config.hidden_size
            * self.config.hidden_size
            * 4  // Q, K, V, O projections
            * self.config.num_hidden_layers
            + self.config.hidden_size * self.config.intermediate_size * 2  // FFN
            * self.config.num_hidden_layers
            + self.config.vocab_size * self.config.hidden_size;  // Embeddings

        bert_params + self.num_trainable_params()
    }

    /// Forward pass through base BERT
    fn forward_base(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Create default token_type_ids
        let token_type_ids = Tensor::zeros(input_ids.shape(), DType::U32, &self.device)?;

        // Forward through frozen BERT
        let hidden_states = self.base_model.forward(input_ids, &token_type_ids, attention_mask)?;

        Ok(hidden_states)
    }

    /// Apply LoRA projection to embeddings
    ///
    /// LoRA: output = input + (input @ down^T @ up^T) * scaling
    fn apply_lora(&self, embeddings: &Tensor) -> Result<Tensor> {
        // LoRA contribution: embeddings @ lora_down^T @ lora_up^T * scaling
        let lora_out = embeddings
            .matmul(&self.lora_down.t()?)?
            .matmul(&self.lora_up.t()?)?;

        let scaled = (lora_out * self.lora_scaling as f64)?;
        let result = (embeddings + scaled)?;

        Ok(result)
    }

    /// Full forward pass with LoRA
    pub fn forward_with_lora(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor> {
        // Get hidden states from frozen BERT
        let hidden_states = self.forward_base(input_ids, Some(attention_mask))?;

        // Apply pooling to get sentence embeddings
        let pooled = apply_pooling(&hidden_states, attention_mask, self.pooling)?;

        // Apply LoRA adaptation
        let adapted = self.apply_lora(&pooled)?;

        Ok(adapted)
    }

    /// Get normalized embeddings (useful for cosine similarity)
    pub fn encode_normalized(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor> {
        let embeddings = self.forward_with_lora(input_ids, attention_mask)?;
        normalize_embeddings(&embeddings)
    }

    /// Load LoRA checkpoint weights
    ///
    /// This loads saved LoRA weights from a checkpoint file into the model.
    /// The checkpoint should contain `lora_projection.down` and `lora_projection.up` tensors.
    pub fn load_lora_checkpoint(&mut self, checkpoint_path: impl AsRef<Path>) -> Result<()> {
        let path = checkpoint_path.as_ref();
        tracing::info!("Loading LoRA checkpoint from: {:?}", path);

        // Load tensors from safetensors file
        let tensors = candle_core::safetensors::load(path, &self.device)
            .context("Failed to load checkpoint file")?;

        // Update LoRA weights
        if let Some(down) = tensors.get("lora_projection.down") {
            self.lora_down = down.clone();
            tracing::debug!("Loaded lora_projection.down: {:?}", down.shape());
        } else {
            return Err(anyhow::anyhow!("Checkpoint missing lora_projection.down"));
        }

        if let Some(up) = tensors.get("lora_projection.up") {
            self.lora_up = up.clone();
            tracing::debug!("Loaded lora_projection.up: {:?}", up.shape());
        } else {
            return Err(anyhow::anyhow!("Checkpoint missing lora_projection.up"));
        }

        tracing::info!("Successfully loaded LoRA checkpoint");
        Ok(())
    }

    /// Create model from pretrained + LoRA checkpoint
    ///
    /// This is a convenience method that loads the base model and then
    /// loads LoRA weights from a checkpoint file.
    pub fn from_pretrained_with_checkpoint(
        model_id: &str,
        lora_config: &LoraConfig,
        checkpoint_path: impl AsRef<Path>,
        device: &Device,
    ) -> Result<Self> {
        // Create a temporary VarMap (we'll replace LoRA weights anyway)
        let var_map = VarMap::new();
        let mut model = Self::from_pretrained(model_id, lora_config, &var_map, device)?;
        model.load_lora_checkpoint(checkpoint_path)?;
        Ok(model)
    }
}

impl EmbeddingModel for BertLoraModel {
    fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        self.forward_with_lora(input_ids, attention_mask)
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

/// Load a BERT model with LoRA for fine-tuning
///
/// # Arguments
/// * `model_id` - HuggingFace model ID or local path
/// * `lora_config` - LoRA configuration
/// * `var_map` - VarMap for trainable parameters
/// * `device` - Device to load on
///
/// # Example
/// ```ignore
/// let var_map = VarMap::new();
/// let lora_config = LoraConfig::new(8, 16.0);
/// let model = load_bert_lora("bert-base-uncased", &lora_config, &var_map, &device)?;
/// ```
pub fn load_bert_lora(
    model_id: &str,
    lora_config: &LoraConfig,
    var_map: &VarMap,
    device: &Device,
) -> Result<BertLoraModel> {
    BertLoraModel::from_pretrained(model_id, lora_config, var_map, device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_param_count() {
        // BERT-base with rank 8
        let hidden_size = 768;
        let rank = 8;
        let trainable = rank * hidden_size * 2;
        assert_eq!(trainable, 12288); // 12K trainable params

        // With rank 16
        let rank = 16;
        let trainable = rank * hidden_size * 2;
        assert_eq!(trainable, 24576); // 24K trainable params
    }
}
