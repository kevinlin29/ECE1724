//! Common traits and utilities for all model types

use anyhow::Result;
use candle_core::{Device, Tensor};
use std::path::Path;

/// Base trait for all embedding models
pub trait EmbeddingModel: Send + Sync {
    fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor>;
    fn hidden_size(&self) -> usize;
    fn architecture(&self) -> ModelArchitecture;
    fn device(&self) -> &Device;
    fn max_seq_length(&self) -> usize {
        512
    }
}

/// Trait for models that support LoRA fine-tuning
pub trait LoraModel: EmbeddingModel {
    fn num_trainable_params(&self) -> usize;
    fn num_total_params(&self) -> usize;
    
    // Use &Path instead of impl AsRef<Path> for object safety
    fn load_lora_checkpoint(&mut self, path: &Path) -> Result<()>;
    fn save_lora_checkpoint(&self, path: &Path) -> Result<()>;
}

/// Model architecture types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArchitecture {
    // Encoder models
    Bert,
    Roberta,
    DistilBert,
    Albert,
    DeBERTa,
    // Decoder models (LLMs)
    Qwen2,
    Llama,
    Mistral,
}

impl std::fmt::Display for ModelArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bert => write!(f, "bert"),
            Self::Roberta => write!(f, "roberta"),
            Self::DistilBert => write!(f, "distilbert"),
            Self::Albert => write!(f, "albert"),
            Self::DeBERTa => write!(f, "deberta"),
            Self::Qwen2 => write!(f, "qwen2"),
            Self::Llama => write!(f, "llama"),
            Self::Mistral => write!(f, "mistral"),
        }
    }
}

impl ModelArchitecture {
    /// Returns true if this is a decoder-only (causal LM) architecture
    pub fn is_decoder(&self) -> bool {
        matches!(self, Self::Qwen2 | Self::Llama | Self::Mistral)
    }

    /// Returns true if this is an encoder-only architecture
    pub fn is_encoder(&self) -> bool {
        !self.is_decoder()
    }
}

/// Pooling strategies for different model types
#[derive(Debug, Clone, Copy, Default)]
pub enum PoolingStrategy {
    #[default]
    Mean,
    Cls,
    Last,
    Max,
}

/// Detect model architecture from config
pub fn detect_architecture(
    model_type: Option<&str>,
    architectures: &[String],
) -> Result<ModelArchitecture> {
    // Check model_type field first
    if let Some(mt) = model_type {
        let mt_lower = mt.to_lowercase();
        // Decoder models (check first as they're more specific)
        if mt_lower.contains("qwen") {
            return Ok(ModelArchitecture::Qwen2);
        }
        if mt_lower.contains("llama") {
            return Ok(ModelArchitecture::Llama);
        }
        if mt_lower.contains("mistral") {
            return Ok(ModelArchitecture::Mistral);
        }
        // Encoder models
        match mt_lower.as_str() {
            "bert" => return Ok(ModelArchitecture::Bert),
            "roberta" | "xlm-roberta" | "camembert" => return Ok(ModelArchitecture::Roberta),
            "distilbert" => return Ok(ModelArchitecture::DistilBert),
            "albert" => return Ok(ModelArchitecture::Albert),
            "deberta" | "deberta-v2" | "deberta-v3" => return Ok(ModelArchitecture::DeBERTa),
            _ => {}
        }
    }

    // Check architectures list
    for arch in architectures {
        let arch_lower = arch.to_lowercase();
        // Decoder models
        if arch_lower.contains("qwen") {
            return Ok(ModelArchitecture::Qwen2);
        }
        if arch_lower.contains("llama") {
            return Ok(ModelArchitecture::Llama);
        }
        if arch_lower.contains("mistral") {
            return Ok(ModelArchitecture::Mistral);
        }
        // Encoder models
        if arch_lower.contains("roberta") {
            return Ok(ModelArchitecture::Roberta);
        }
        if arch_lower.contains("deberta") {
            return Ok(ModelArchitecture::DeBERTa);
        }
        if arch_lower.contains("distilbert") {
            return Ok(ModelArchitecture::DistilBert);
        }
        if arch_lower.contains("albert") {
            return Ok(ModelArchitecture::Albert);
        }
        if arch_lower.contains("bert") {
            return Ok(ModelArchitecture::Bert);
        }
    }

    anyhow::bail!(
        "Could not detect model architecture from type={:?}, architectures={:?}",
        model_type,
        architectures
    )
}