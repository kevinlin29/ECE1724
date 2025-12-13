//! Candle-based BERT embedder with LoRA support
//!
//! Bridges the fine-tuned BertLoraModel from the training module
//! to the Embedder trait for use in retrieval pipelines.

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarMap;
use std::path::PathBuf;
use std::sync::Arc;

use crate::embedding::{Embedder, Embedding, PoolingStrategy};
use crate::training::{
    BertLoraModel, DevicePreference, EmbeddingModel, LoraConfig, LoraModel,
    PoolingStrategy as TrainingPoolingStrategy, TokenizerWrapper, select_device,
};

/// Configuration for the Candle BERT embedder
#[derive(Debug, Clone)]
pub struct CandleBertConfig {
    /// HuggingFace model ID or local path
    pub model_id: String,

    /// LoRA rank (must match checkpoint if loading)
    pub lora_rank: usize,

    /// LoRA alpha (must match checkpoint if loading)
    pub lora_alpha: f32,

    /// Path to LoRA checkpoint file (.safetensors)
    pub lora_checkpoint: Option<PathBuf>,

    /// Device preference (auto, cuda, metal, cpu)
    pub device: DevicePreference,

    /// Maximum sequence length
    pub max_length: usize,

    /// Pooling strategy
    pub pooling: PoolingStrategy,

    /// Whether to normalize embeddings
    pub normalize: bool,

    /// Batch size for embedding
    pub batch_size: usize,
}

impl Default for CandleBertConfig {
    fn default() -> Self {
        Self {
            model_id: "bert-base-uncased".to_string(),
            lora_rank: 8,
            lora_alpha: 16.0,
            lora_checkpoint: None,
            device: DevicePreference::Auto,
            max_length: 512,
            pooling: PoolingStrategy::Mean,
            normalize: true,
            batch_size: 32,
        }
    }
}

impl CandleBertConfig {
    /// Create a new config with the given model ID
    pub fn new(model_id: &str) -> Self {
        Self {
            model_id: model_id.to_string(),
            ..Default::default()
        }
    }

    /// Set LoRA configuration
    pub fn with_lora_config(mut self, rank: usize, alpha: f32) -> Self {
        self.lora_rank = rank;
        self.lora_alpha = alpha;
        self
    }

    /// Set LoRA checkpoint path
    pub fn with_lora_checkpoint(mut self, path: &str) -> Self {
        self.lora_checkpoint = Some(PathBuf::from(path));
        self
    }

    /// Set device preference
    pub fn with_device(mut self, device: DevicePreference) -> Self {
        self.device = device;
        self
    }

    /// Set max sequence length
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    /// Set pooling strategy
    pub fn with_pooling(mut self, pooling: PoolingStrategy) -> Self {
        self.pooling = pooling;
        self
    }

    /// Set normalization
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Convert to LoraConfig
    fn to_lora_config(&self) -> LoraConfig {
        LoraConfig {
            rank: self.lora_rank,
            alpha: self.lora_alpha,
            ..Default::default()
        }
    }

    /// Convert pooling strategy to training module's pooling strategy
    fn to_training_pooling(&self) -> TrainingPoolingStrategy {
        match self.pooling {
            PoolingStrategy::Mean => TrainingPoolingStrategy::Mean,
            PoolingStrategy::Cls => TrainingPoolingStrategy::Cls,
            PoolingStrategy::Max => TrainingPoolingStrategy::Max,
        }
    }
}

/// Candle-based BERT embedder with LoRA support
///
/// Wraps a BertLoraModel from the training module and implements
/// the Embedder trait for use in retrieval pipelines.
pub struct CandleBertEmbedder {
    model: BertLoraModel,
    tokenizer: TokenizerWrapper,
    config: CandleBertConfig,
    device: Device,
    hidden_size: usize,
}

impl CandleBertEmbedder {
    /// Create a new Candle BERT embedder from config
    pub fn new(config: CandleBertConfig) -> Result<Self> {
        let device = select_device(config.device.clone())?;

        tracing::info!("Loading Candle BERT embedder: {}", config.model_id);
        tracing::info!("  Device: {:?}", device);
        tracing::info!("  LoRA rank: {}, alpha: {}", config.lora_rank, config.lora_alpha);
        if let Some(ref ckpt) = config.lora_checkpoint {
            tracing::info!("  LoRA checkpoint: {:?}", ckpt);
        }

        // Load tokenizer
        let tokenizer = TokenizerWrapper::from_pretrained(&config.model_id)
            .context("Failed to load tokenizer")?
            .with_max_length(config.max_length);

        // Load model
        let lora_config = config.to_lora_config();
        let var_map = VarMap::new();

        let mut model = BertLoraModel::from_pretrained(
            &config.model_id,
            &lora_config,
            &var_map,
            &device,
        ).context("Failed to load BERT model")?;

        // Set pooling strategy
        model = model.with_pooling(config.to_training_pooling());

        // Load LoRA checkpoint if specified
        if let Some(ref checkpoint_path) = config.lora_checkpoint {
            model.load_lora_checkpoint(checkpoint_path)
                .context("Failed to load LoRA checkpoint")?;
            tracing::info!("Loaded fine-tuned LoRA weights");
        }

        let hidden_size = model.hidden_size();

        tracing::info!("Candle BERT embedder loaded (dim={})", hidden_size);

        Ok(Self {
            model,
            tokenizer,
            config,
            device,
            hidden_size,
        })
    }

    /// Create from a pre-loaded model (useful for sharing models)
    pub fn from_model(
        model: BertLoraModel,
        tokenizer: TokenizerWrapper,
        config: CandleBertConfig,
        device: Device,
    ) -> Self {
        let hidden_size = model.hidden_size();
        Self {
            model,
            tokenizer,
            config,
            device,
            hidden_size,
        }
    }

    /// Embed a single text, returning the raw tensor
    fn embed_tensor(&self, text: &str) -> Result<Tensor> {
        let encoded = self.tokenizer.encode(text, true)?;

        let input_ids = Tensor::new(&encoded.input_ids[..], &self.device)?
            .unsqueeze(0)?;
        let attention_mask = Tensor::new(&encoded.attention_mask[..], &self.device)?
            .unsqueeze(0)?
            .to_dtype(candle_core::DType::F32)?;

        let embeddings = if self.config.normalize {
            self.model.encode_normalized(&input_ids, &attention_mask)?
        } else {
            self.model.forward(&input_ids, &attention_mask)?
        };

        // Remove batch dimension
        Ok(embeddings.squeeze(0)?)
    }

    /// Embed multiple texts as tensors
    fn embed_batch_tensor(&self, texts: &[&str]) -> Result<Tensor> {
        if texts.is_empty() {
            anyhow::bail!("Cannot embed empty batch");
        }

        // Convert &[&str] to Vec<String> for tokenizer
        let texts_owned: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let batch_encoded = self.tokenizer.encode_batch(&texts_owned, true)?;

        let input_ids = Tensor::new(batch_encoded.input_ids.as_slice(), &self.device)?;
        let attention_mask = Tensor::new(batch_encoded.attention_mask.as_slice(), &self.device)?
            .to_dtype(candle_core::DType::F32)?;

        // Reshape to [batch_size, seq_len]
        let batch_size = texts.len();
        let seq_len = batch_encoded.input_ids.len() / batch_size;
        let input_ids = input_ids.reshape((batch_size, seq_len))?;
        let attention_mask = attention_mask.reshape((batch_size, seq_len))?;

        let embeddings = if self.config.normalize {
            self.model.encode_normalized(&input_ids, &attention_mask)?
        } else {
            self.model.forward(&input_ids, &attention_mask)?
        };

        Ok(embeddings)
    }

    /// Get the underlying model reference
    pub fn model(&self) -> &BertLoraModel {
        &self.model
    }

    /// Get the tokenizer reference
    pub fn tokenizer(&self) -> &TokenizerWrapper {
        &self.tokenizer
    }

    /// Get the device
    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl Embedder for CandleBertEmbedder {
    fn embed(&self, text: &str) -> Result<Embedding> {
        let tensor = self.embed_tensor(text)?;
        let embedding: Vec<f32> = tensor.to_vec1()?;
        Ok(embedding)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Process in batches
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(self.config.batch_size) {
            let batch_tensor = self.embed_batch_tensor(chunk)?;
            let batch_size = chunk.len();

            // Convert tensor to vec of embeddings
            for i in 0..batch_size {
                let embedding_tensor = batch_tensor.get(i)?;
                let embedding: Vec<f32> = embedding_tensor.to_vec1()?;
                all_embeddings.push(embedding);
            }
        }

        Ok(all_embeddings)
    }

    fn dimension(&self) -> usize {
        self.hidden_size
    }

    fn model_name(&self) -> &str {
        &self.config.model_id
    }
}

/// Auto-detect and load the best available embedder checkpoint
///
/// Searches for fine-tuned checkpoints in standard locations:
/// 1. Explicit checkpoint path (if provided)
/// 2. output/{model_name}-lora/lora_checkpoint.safetensors
/// 3. output/recipe-lora/lora_checkpoint.safetensors
/// 4. Falls back to base model without LoRA
pub fn auto_detect_embedder(
    model_id: &str,
    checkpoint_override: Option<&str>,
    device: DevicePreference,
) -> Result<Arc<dyn Embedder>> {
    let mut config = CandleBertConfig::new(model_id)
        .with_device(device);

    // Check explicit override first
    if let Some(ckpt_path) = checkpoint_override {
        let path = PathBuf::from(ckpt_path);
        if path.exists() {
            tracing::info!("Using specified checkpoint: {:?}", path);
            config = config.with_lora_checkpoint(ckpt_path);
            return Ok(Arc::new(CandleBertEmbedder::new(config)?));
        } else {
            tracing::warn!("Specified checkpoint not found: {:?}", path);
        }
    }

    // Try to find checkpoint in output directory
    let model_name = model_id.split('/').last().unwrap_or(model_id);
    let search_paths = [
        format!("output/{}-lora/lora_checkpoint.safetensors", model_name),
        format!("output/recipe-lora/lora_checkpoint.safetensors"),
        "output/bert-lora-cuda/lora_checkpoint.safetensors".to_string(),
    ];

    for search_path in &search_paths {
        let path = PathBuf::from(search_path);
        if path.exists() {
            tracing::info!("Auto-detected checkpoint: {:?}", path);
            config = config.with_lora_checkpoint(search_path);
            return Ok(Arc::new(CandleBertEmbedder::new(config)?));
        }
    }

    // Also try scanning output directory for any lora checkpoints
    if let Ok(entries) = std::fs::read_dir("output") {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let checkpoint = path.join("lora_checkpoint.safetensors");
                if checkpoint.exists() {
                    tracing::info!("Auto-detected checkpoint: {:?}", checkpoint);
                    config = config.with_lora_checkpoint(checkpoint.to_str().unwrap_or_default());
                    return Ok(Arc::new(CandleBertEmbedder::new(config)?));
                }
            }
        }
    }

    // Fall back to base model
    tracing::info!("No fine-tuned checkpoint found, using base model");
    Ok(Arc::new(CandleBertEmbedder::new(config)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder() {
        let config = CandleBertConfig::new("bert-base-uncased")
            .with_lora_config(16, 32.0)
            .with_device(DevicePreference::Cpu)
            .with_max_length(256)
            .with_normalize(false);

        assert_eq!(config.model_id, "bert-base-uncased");
        assert_eq!(config.lora_rank, 16);
        assert_eq!(config.lora_alpha, 32.0);
        assert_eq!(config.max_length, 256);
        assert!(!config.normalize);
    }

    #[test]
    fn test_lora_config_conversion() {
        let config = CandleBertConfig::new("test")
            .with_lora_config(8, 16.0);

        let lora_config = config.to_lora_config();
        assert_eq!(lora_config.rank, 8);
        assert_eq!(lora_config.alpha, 16.0);
    }

    #[test]
    fn test_pooling_conversion() {
        let config = CandleBertConfig::new("test")
            .with_pooling(PoolingStrategy::Cls);

        let training_pooling = config.to_training_pooling();
        assert!(matches!(training_pooling, TrainingPoolingStrategy::Cls));
    }
}
