//! Candle-based decoder model implementation
//!
//! Supports Qwen2.5 and other decoder architectures via the Candle ML framework.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::qwen2::{Config as Qwen2Config, ModelForCausalLM as Qwen2Model};
use std::path::Path;
use std::sync::Mutex;

use super::{Generator, GeneratorConfig, SamplingParams};
use crate::training::{hub::ModelLoader, select_device, TokenizerWrapper};

/// Candle-based text generator supporting multiple architectures
pub struct CandleGenerator {
    /// Model wrapped in Mutex for thread-safe interior mutability (required for KV cache)
    model: Mutex<GeneratorModel>,
    tokenizer: TokenizerWrapper,
    config: GeneratorConfig,
    device: Device,
    eos_token_id: u32,
}

/// Enum for different model architectures
enum GeneratorModel {
    Qwen2(Qwen2Model),
}

impl CandleGenerator {
    /// Create a new Candle generator from config
    pub fn new(config: GeneratorConfig) -> Result<Self> {
        let device = select_device(config.device)?;

        tracing::info!("Loading generator model: {}", config.model_id);
        tracing::info!("  Device: {:?}", device);
        tracing::info!("  Max new tokens: {}", config.max_new_tokens);
        tracing::info!("  Dtype: {}", config.dtype);

        // Load tokenizer
        let tokenizer = TokenizerWrapper::from_pretrained(&config.model_id)
            .context("Failed to load tokenizer")?
            .with_max_length(config.max_seq_length);

        // Get EOS token ID
        let eos_token_id = tokenizer
            .eos_token_id()
            .unwrap_or(151643); // Qwen2 default

        // Load model
        let model = Self::load_model(&config, &device)?;

        tracing::info!("Generator loaded successfully");

        Ok(Self {
            model: Mutex::new(model),
            tokenizer,
            config,
            device,
            eos_token_id,
        })
    }

    fn load_model(config: &GeneratorConfig, device: &Device) -> Result<GeneratorModel> {
        let loader = ModelLoader::new()?;
        let model_path = loader.load_model_path(&config.model_id)?;

        // Read config to detect architecture
        let config_str = std::fs::read_to_string(&model_path.config_file)
            .context("Failed to read model config")?;
        let model_config: serde_json::Value = serde_json::from_str(&config_str)
            .context("Failed to parse model config")?;

        let dtype = match config.dtype.as_str() {
            "f16" => DType::F16,
            "bf16" => DType::BF16,
            _ => DType::F32,
        };

        // Detect and load appropriate model
        let arch = model_config["architectures"]
            .get(0)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_lowercase();

        let model_type = model_config["model_type"]
            .as_str()
            .unwrap_or("")
            .to_lowercase();

        tracing::info!("Detected architecture: {}, model_type: {}", arch, model_type);

        if arch.contains("qwen2") || model_type.contains("qwen2") {
            let qwen_config: Qwen2Config = serde_json::from_str(&config_str)
                .context("Failed to parse Qwen2 config")?;

            tracing::info!(
                "Loading Qwen2: vocab={}, hidden={}, layers={}",
                qwen_config.vocab_size,
                qwen_config.hidden_size,
                qwen_config.num_hidden_layers
            );

            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[&model_path.weights_file], dtype, device)
                    .context("Failed to load model weights")?
            };

            let model = Qwen2Model::new(&qwen_config, vb).context("Failed to create Qwen2 model")?;

            Ok(GeneratorModel::Qwen2(model))
        } else {
            anyhow::bail!(
                "Unsupported model architecture: {}. Supported: qwen2",
                arch
            )
        }
    }

    /// Load LoRA checkpoint if specified
    pub fn load_lora_checkpoint(&mut self, _path: impl AsRef<Path>) -> Result<()> {
        // LoRA loading for decoder models
        // This would require LoRA-aware model loading
        tracing::warn!("LoRA loading for decoder models not yet implemented");
        Ok(())
    }

    /// Internal generation with sampling
    fn generate_internal(&self, prompt: &str, params: &SamplingParams) -> Result<String> {
        // Encode prompt
        let encoded = self.tokenizer.encode(prompt, true)?;
        let prompt_tokens = encoded.input_ids.clone();
        let prompt_len = prompt_tokens.len();

        if prompt_len == 0 {
            anyhow::bail!("Empty prompt after tokenization");
        }

        // Create input tensor
        let mut all_tokens = prompt_tokens.clone();
        let max_tokens = params.max_new_tokens.unwrap_or(self.config.max_new_tokens);

        // Setup logits processor for sampling
        let seed = params.seed.unwrap_or(42);
        let temperature = if params.temperature > 0.0 {
            Some(params.temperature as f64)
        } else {
            None
        };
        let top_p = if params.top_p < 1.0 {
            Some(params.top_p as f64)
        } else {
            None
        };

        let mut logits_processor = LogitsProcessor::new(seed, temperature, top_p);

        // Generation loop
        let mut pos = 0;
        for _ in 0..max_tokens {
            // Create input tensor for current position
            let context_size = if pos == 0 { all_tokens.len() } else { 1 };
            let start_pos = all_tokens.len().saturating_sub(context_size);
            let input_ids: Vec<u32> = all_tokens[start_pos..].to_vec();

            let input_tensor = Tensor::new(&input_ids[..], &self.device)?
                .unsqueeze(0)?;

            // Forward pass (using Mutex for thread-safe interior mutability)
            let logits = {
                let mut model_guard = self.model.lock()
                    .map_err(|e| anyhow::anyhow!("Model lock poisoned: {}", e))?;
                match &mut *model_guard {
                    GeneratorModel::Qwen2(model) => {
                        model.forward(&input_tensor, pos)?
                    }
                }
            };

            // Get logits for last token
            let logits = logits.squeeze(0)?;
            let logits = if logits.dims().len() > 1 {
                logits.get(logits.dim(0)? - 1)?
            } else {
                logits
            };

            // Apply top-k if specified
            let logits = if params.top_k > 0 {
                self.apply_top_k(&logits, params.top_k)?
            } else {
                logits
            };

            // Apply repetition penalty
            let logits = if params.repetition_penalty != 1.0 {
                self.apply_repetition_penalty(&logits, &all_tokens, params.repetition_penalty)?
            } else {
                logits
            };

            // Sample next token
            let next_token = logits_processor.sample(&logits)?;

            all_tokens.push(next_token);
            pos += context_size;

            // Check for EOS
            if next_token == self.eos_token_id {
                tracing::debug!("Generation stopped: EOS token");
                break;
            }

            // Check stop sequences
            if !params.stop_sequences.is_empty() {
                let generated = self.tokenizer.decode(&all_tokens[prompt_len..], true)?;
                if params
                    .stop_sequences
                    .iter()
                    .any(|s| generated.contains(s))
                {
                    tracing::debug!("Generation stopped: stop sequence");
                    break;
                }
            }
        }

        // Decode generated tokens (excluding prompt)
        let generated_tokens = &all_tokens[prompt_len..];
        let output = self.tokenizer.decode(generated_tokens, true)?;

        Ok(output.trim().to_string())
    }

    /// Apply top-k filtering to logits
    fn apply_top_k(&self, logits: &Tensor, k: usize) -> Result<Tensor> {
        let vocab_size = logits.dim(0)?;
        if k >= vocab_size {
            return Ok(logits.clone());
        }

        // Get top-k indices
        let logits_vec: Vec<f32> = logits.to_vec1()?;
        let mut indexed: Vec<(usize, f32)> = logits_vec.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Set non-top-k to -inf
        let mut filtered = vec![f32::NEG_INFINITY; vocab_size];
        for (idx, val) in indexed.into_iter().take(k) {
            filtered[idx] = val;
        }

        Ok(Tensor::new(&filtered[..], logits.device())?)
    }

    /// Apply repetition penalty
    fn apply_repetition_penalty(
        &self,
        logits: &Tensor,
        tokens: &[u32],
        penalty: f32,
    ) -> Result<Tensor> {
        let mut logits_vec: Vec<f32> = logits.to_vec1()?;

        for &token in tokens {
            let idx = token as usize;
            if idx < logits_vec.len() {
                if logits_vec[idx] > 0.0 {
                    logits_vec[idx] /= penalty;
                } else {
                    logits_vec[idx] *= penalty;
                }
            }
        }

        Ok(Tensor::new(&logits_vec[..], logits.device())?)
    }
}

impl Generator for CandleGenerator {
    fn generate(&self, prompt: &str, params: &SamplingParams) -> Result<String> {
        self.generate_internal(prompt, params)
    }

    fn generate_stream(
        &self,
        prompt: &str,
        params: &SamplingParams,
    ) -> Result<Box<dyn Iterator<Item = Result<String>> + Send + '_>> {
        // For now, generate all at once and yield as chunks
        // True streaming would require restructuring the generation loop
        let result = self.generate(prompt, params)?;

        // Split into words for simulated streaming
        let words: Vec<String> = result
            .split_whitespace()
            .map(|s| format!("{} ", s))
            .collect();

        Ok(Box::new(words.into_iter().map(Ok)))
    }

    fn model_name(&self) -> &str {
        &self.config.model_id
    }

    fn max_context_length(&self) -> usize {
        self.config.max_seq_length
    }

    fn count_tokens(&self, text: &str) -> Result<usize> {
        let encoded = self.tokenizer.encode(text, false)?;
        Ok(encoded.input_ids.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::DevicePreference;

    #[test]
    fn test_sampling_params_defaults() {
        let params = SamplingParams::default();
        assert!(params.temperature > 0.0);
        assert!(params.top_p > 0.0);
    }

    #[test]
    fn test_generator_config() {
        let config = GeneratorConfig::new("test-model")
            .with_max_new_tokens(256)
            .with_device(DevicePreference::Cpu);

        assert_eq!(config.model_id, "test-model");
        assert_eq!(config.max_new_tokens, 256);
    }
}
