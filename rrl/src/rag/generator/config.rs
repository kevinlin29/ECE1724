//! Configuration for text generators
//!
//! Defines configuration structures for generator initialization
//! and sampling parameters for controlling generation behavior.

use crate::training::DevicePreference;
use serde::{Deserialize, Serialize};

/// Configuration for initializing a generator model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorConfig {
    /// HuggingFace model ID or local path
    pub model_id: String,

    /// Optional LoRA checkpoint path for fine-tuned models
    pub lora_checkpoint: Option<String>,

    /// LoRA rank (must match checkpoint if loading)
    pub lora_rank: usize,

    /// LoRA alpha (must match checkpoint if loading)
    pub lora_alpha: f32,

    /// Device preference (auto, cuda, metal, cpu)
    pub device: DevicePreference,

    /// Maximum new tokens to generate
    pub max_new_tokens: usize,

    /// Model data type ("f32", "f16", "bf16")
    pub dtype: String,

    /// Enable KV-cache for efficient generation
    pub use_kv_cache: bool,

    /// Maximum sequence length for the model
    pub max_seq_length: usize,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            model_id: "Qwen/Qwen2.5-0.5B".to_string(),
            lora_checkpoint: None,
            lora_rank: 8,
            lora_alpha: 16.0,
            device: DevicePreference::Auto,
            max_new_tokens: 512,
            dtype: "f32".to_string(),
            use_kv_cache: true,
            max_seq_length: 4096,
        }
    }
}

impl GeneratorConfig {
    /// Create a new generator config with the given model ID
    pub fn new(model_id: &str) -> Self {
        Self {
            model_id: model_id.to_string(),
            ..Default::default()
        }
    }

    /// Set the LoRA checkpoint path
    pub fn with_lora_checkpoint(mut self, path: &str) -> Self {
        self.lora_checkpoint = Some(path.to_string());
        self
    }

    /// Set the LoRA configuration
    pub fn with_lora_config(mut self, rank: usize, alpha: f32) -> Self {
        self.lora_rank = rank;
        self.lora_alpha = alpha;
        self
    }

    /// Set the device preference
    pub fn with_device(mut self, device: DevicePreference) -> Self {
        self.device = device;
        self
    }

    /// Set the maximum new tokens
    pub fn with_max_new_tokens(mut self, max_tokens: usize) -> Self {
        self.max_new_tokens = max_tokens;
        self
    }

    /// Set the data type
    pub fn with_dtype(mut self, dtype: &str) -> Self {
        self.dtype = dtype.to_string();
        self
    }

    /// Enable or disable KV-cache
    pub fn with_kv_cache(mut self, enable: bool) -> Self {
        self.use_kv_cache = enable;
        self
    }
}

/// Sampling parameters for text generation
///
/// Controls the randomness and diversity of generated text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Temperature (higher = more random, lower = more deterministic)
    /// Range: 0.0 to 2.0, default: 0.7
    pub temperature: f32,

    /// Top-p (nucleus sampling) - cumulative probability threshold
    /// Range: 0.0 to 1.0, default: 0.9
    pub top_p: f32,

    /// Top-k sampling (0 = disabled)
    /// Limits sampling to k most likely tokens
    pub top_k: usize,

    /// Repetition penalty (1.0 = no penalty)
    /// Values > 1.0 discourage repetition
    pub repetition_penalty: f32,

    /// Maximum new tokens to generate (overrides config if set)
    pub max_new_tokens: Option<usize>,

    /// Stop sequences - generation stops when any of these are produced
    pub stop_sequences: Vec<String>,

    /// Random seed for reproducibility (None = random)
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
            max_new_tokens: None,
            stop_sequences: vec![],
            seed: None,
        }
    }
}

impl SamplingParams {
    /// Create greedy decoding parameters (deterministic)
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 1,
            repetition_penalty: 1.0,
            ..Default::default()
        }
    }

    /// Create creative sampling parameters (more random)
    pub fn creative() -> Self {
        Self {
            temperature: 1.0,
            top_p: 0.95,
            top_k: 0,
            repetition_penalty: 1.2,
            ..Default::default()
        }
    }

    /// Create balanced sampling parameters
    pub fn balanced() -> Self {
        Self::default()
    }

    /// Set temperature
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /// Set top-p
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    /// Set top-k
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set max new tokens
    pub fn with_max_new_tokens(mut self, max_tokens: usize) -> Self {
        self.max_new_tokens = Some(max_tokens);
        self
    }

    /// Add stop sequences
    pub fn with_stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.stop_sequences = sequences;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_config_defaults() {
        let config = GeneratorConfig::default();

        assert_eq!(config.model_id, "Qwen/Qwen2.5-0.5B");
        assert!(config.lora_checkpoint.is_none());
        assert_eq!(config.max_new_tokens, 512);
        assert!(config.use_kv_cache);
    }

    #[test]
    fn test_generator_config_builder() {
        let config = GeneratorConfig::new("custom-model")
            .with_lora_checkpoint("/path/to/checkpoint")
            .with_max_new_tokens(1024)
            .with_device(DevicePreference::Cuda);

        assert_eq!(config.model_id, "custom-model");
        assert_eq!(config.lora_checkpoint, Some("/path/to/checkpoint".to_string()));
        assert_eq!(config.max_new_tokens, 1024);
    }

    #[test]
    fn test_sampling_params_presets() {
        let greedy = SamplingParams::greedy();
        assert_eq!(greedy.temperature, 0.0);
        assert_eq!(greedy.top_k, 1);

        let creative = SamplingParams::creative();
        assert_eq!(creative.temperature, 1.0);
        assert_eq!(creative.top_k, 0);
    }

    #[test]
    fn test_sampling_params_builder() {
        let params = SamplingParams::default()
            .with_temperature(0.5)
            .with_top_p(0.8)
            .with_max_new_tokens(256);

        assert_eq!(params.temperature, 0.5);
        assert_eq!(params.top_p, 0.8);
        assert_eq!(params.max_new_tokens, Some(256));
    }
}
