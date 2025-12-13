//! Common traits and utilities for all model types

use anyhow::Result;
use candle_core::{Device, Tensor};

/// Base trait for all embedding models
pub trait EmbeddingModel: Send + Sync {
    /// Forward pass to get embeddings
    fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor>;
    
    /// Get the hidden size (embedding dimension)
    fn hidden_size(&self) -> usize;
    
    /// Get the model architecture type
    fn architecture(&self) -> ModelArchitecture;
    
    /// Get the device this model is on
    fn device(&self) -> &Device;
    
    /// Get maximum sequence length
    fn max_seq_length(&self) -> usize {
        512
    }
}

/// Trait for models that support LoRA fine-tuning
pub trait LoraModel: EmbeddingModel {
    /// Get number of trainable LoRA parameters
    fn num_trainable_params(&self) -> usize;
    
    /// Get total model parameters
    fn num_total_params(&self) -> usize;
    
    /// Load LoRA checkpoint
    fn load_lora_checkpoint(&mut self, path: impl AsRef<std::path::Path>) -> Result<()>;
    
    /// Save LoRA checkpoint
    fn save_lora_checkpoint(&self, path: impl AsRef<std::path::Path>) -> Result<()>;
}

/// Model architecture types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArchitecture {
    // Encoder-only models
    Bert,
    Roberta,
    DistilBert,
    Albert,
    DeBERTa,
    EncoderV2,  // For BGE, E5, etc.
    
    // Decoder-only models
    Gpt2,
    GptNeo,
    LLaMA,
    Mistral,
    
    // Encoder-decoder models
    T5,
    Bart,
    MT5,
}

impl std::fmt::Display for ModelArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bert => write!(f, "bert"),
            Self::Roberta => write!(f, "roberta"),
            Self::DistilBert => write!(f, "distilbert"),
            Self::Albert => write!(f, "albert"),
            Self::DeBERTa => write!(f, "deberta"),
            Self::EncoderV2 => write!(f, "encoder-v2"),
            Self::Gpt2 => write!(f, "gpt2"),
            Self::GptNeo => write!(f, "gpt-neo"),
            Self::LLaMA => write!(f, "llama"),
            Self::Mistral => write!(f, "mistral"),
            Self::T5 => write!(f, "t5"),
            Self::Bart => write!(f, "bart"),
            Self::MT5 => write!(f, "mt5"),
        }
    }
}

/// Pooling strategies for different model types
#[derive(Debug, Clone, Copy, Default)]
pub enum PoolingStrategy {
    /// Mean pooling over non-padding tokens (most common)
    #[default]
    Mean,
    /// Use [CLS] token (BERT-style)
    Cls,
    /// Use last token (GPT-style)
    Last,
    /// Max pooling
    Max,
    /// Weighted mean (attention-based)
    WeightedMean,
    /// No pooling (use sequence output)
    None,
}

/// Model capabilities
#[derive(Debug, Clone)]
pub struct ModelCapabilities {
    /// Supports bidirectional attention
    pub bidirectional: bool,
    /// Has [CLS] token
    pub has_cls_token: bool,
    /// Supports causal masking
    pub causal: bool,
    /// Supports encoder-decoder
    pub encoder_decoder: bool,
    /// Recommended pooling strategy
    pub default_pooling: PoolingStrategy,
}

impl ModelCapabilities {
    /// Capabilities for BERT-family models
    pub fn bert_family() -> Self {
        Self {
            bidirectional: true,
            has_cls_token: true,
            causal: false,
            encoder_decoder: false,
            default_pooling: PoolingStrategy::Mean,
        }
    }
    
    /// Capabilities for GPT-family models
    pub fn gpt_family() -> Self {
        Self {
            bidirectional: false,
            has_cls_token: false,
            causal: true,
            encoder_decoder: false,
            default_pooling: PoolingStrategy::Last,
        }
    }
    
    /// Capabilities for T5-family models
    pub fn t5_family() -> Self {
        Self {
            bidirectional: true,
            has_cls_token: false,
            causal: false,
            encoder_decoder: true,
            default_pooling: PoolingStrategy::Mean,
        }
    }
}

/// Detect model architecture from config
pub fn detect_architecture(
    model_type: Option<&str>,
    architectures: &[String],
) -> Result<ModelArchitecture> {
    // Try model_type first
    if let Some(mt) = model_type {
        match mt.to_lowercase().as_str() {
            "bert" => return Ok(ModelArchitecture::Bert),
            "roberta" | "xlm-roberta" | "camembert" => return Ok(ModelArchitecture::Roberta),
            "distilbert" => return Ok(ModelArchitecture::DistilBert),
            "albert" => return Ok(ModelArchitecture::Albert),
            "deberta" | "deberta-v2" | "deberta-v3" => return Ok(ModelArchitecture::DeBERTa),
            "gpt2" => return Ok(ModelArchitecture::Gpt2),
            "gpt_neo" | "gpt-neo" => return Ok(ModelArchitecture::GptNeo),
            "llama" => return Ok(ModelArchitecture::LLaMA),
            "mistral" => return Ok(ModelArchitecture::Mistral),
            "t5" => return Ok(ModelArchitecture::T5),
            "bart" => return Ok(ModelArchitecture::Bart),
            "mt5" => return Ok(ModelArchitecture::MT5),
            _ => {}
        }
    }
    
    // Try architectures
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
        if arch_lower.contains("gpt2") {
            return Ok(ModelArchitecture::Gpt2);
        }
        if arch_lower.contains("gptneo") || arch_lower.contains("gpt-neo") {
            return Ok(ModelArchitecture::GptNeo);
        }
        if arch_lower.contains("llama") {
            return Ok(ModelArchitecture::LLaMA);
        }
        if arch_lower.contains("mistral") {
            return Ok(ModelArchitecture::Mistral);
        }
        if arch_lower.contains("t5") {
            return Ok(ModelArchitecture::T5);
        }
        if arch_lower.contains("bart") {
            return Ok(ModelArchitecture::Bart);
        }
    }
    
    anyhow::bail!(
        "Could not detect model architecture from type={:?}, architectures={:?}",
        model_type,
        architectures
    )
}

/// Get model capabilities from architecture
pub fn get_capabilities(arch: ModelArchitecture) -> ModelCapabilities {
    match arch {
        ModelArchitecture::Bert
        | ModelArchitecture::Roberta
        | ModelArchitecture::DistilBert
        | ModelArchitecture::Albert
        | ModelArchitecture::DeBERTa
        | ModelArchitecture::EncoderV2 => ModelCapabilities::bert_family(),
        
        ModelArchitecture::Gpt2
        | ModelArchitecture::GptNeo
        | ModelArchitecture::LLaMA
        | ModelArchitecture::Mistral => ModelCapabilities::gpt_family(),
        
        ModelArchitecture::T5
        | ModelArchitecture::Bart
        | ModelArchitecture::MT5 => ModelCapabilities::t5_family(),
    }
}