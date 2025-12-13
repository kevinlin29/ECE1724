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
    Bert,
    Roberta,
    DistilBert,
    Albert,
    DeBERTa,
}

impl std::fmt::Display for ModelArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bert => write!(f, "bert"),
            Self::Roberta => write!(f, "roberta"),
            Self::DistilBert => write!(f, "distilbert"),
            Self::Albert => write!(f, "albert"),
            Self::DeBERTa => write!(f, "deberta"),
        }
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
    if let Some(mt) = model_type {
        match mt.to_lowercase().as_str() {
            "bert" => return Ok(ModelArchitecture::Bert),
            "roberta" | "xlm-roberta" | "camembert" => return Ok(ModelArchitecture::Roberta),
            "distilbert" => return Ok(ModelArchitecture::DistilBert),
            "albert" => return Ok(ModelArchitecture::Albert),
            "deberta" | "deberta-v2" | "deberta-v3" => return Ok(ModelArchitecture::DeBERTa),
            _ => {}
        }
    }
    
    for arch in architectures {
        let arch_lower = arch.to_lowercase();
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