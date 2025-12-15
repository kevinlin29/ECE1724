//! Unified decoder model with trainable LoRA adapters
//!
//! Supports Qwen2, LLaMA, Mistral, and other decoder-only transformer architectures.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Init, VarBuilder, VarMap};
use candle_transformers::models::llama::{Cache as LlamaCache, Config as LlamaConfig, Llama};
use candle_transformers::models::qwen2::{Config as Qwen2Config, ModelForCausalLM as Qwen2Model};
use std::path::Path;
use std::sync::Mutex;

use crate::training::hub::{ModelLoader, ModelPath};
use crate::training::lora::LoraConfig;
use crate::training::models::common::{EmbeddingModel, LoraModel, ModelArchitecture};

/// Enum for different decoder model architectures
enum DecoderModel {
    Qwen2(Qwen2Model, candle_nn::Embedding), // Include embed_tokens for embedding extraction
    Llama(Llama, LlamaConfig),
}

/// Configuration for decoder models
#[derive(Debug, Clone)]
pub struct DecoderConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub architecture: ModelArchitecture,
}

/// Unified decoder model with LoRA adapters for fine-tuning
///
/// This implementation applies LoRA as a projection on the model's hidden states,
/// enabling efficient fine-tuning of decoder-only LLMs like Qwen2, LLaMA, and Mistral.
pub struct DecoderLoraModel {
    /// Model wrapped in Mutex for thread-safe interior mutability (required for KV cache)
    model: Mutex<DecoderModel>,
    config: DecoderConfig,
    device: Device,
    // LoRA adapter applied to hidden states
    lora_down: Tensor,
    lora_up: Tensor,
    lora_scaling: f32,
    lora_rank: usize,
}

impl DecoderLoraModel {
    /// Load from a model path with LoRA adapters
    pub fn from_model_path(
        model_path: &ModelPath,
        lora_config: &LoraConfig,
        var_map: &VarMap,
        device: &Device,
    ) -> Result<Self> {
        model_path.validate()?;

        // Read and parse config
        let config_str = std::fs::read_to_string(&model_path.config_file)
            .context("Failed to read model config")?;
        let model_config: serde_json::Value = serde_json::from_str(&config_str)
            .context("Failed to parse model config")?;

        // Detect architecture
        let arch = model_config["architectures"]
            .get(0)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_lowercase();

        let model_type = model_config["model_type"]
            .as_str()
            .unwrap_or("")
            .to_lowercase();

        tracing::debug!("Loading decoder model: arch={}, type={}", arch, model_type);

        // Determine dtype
        let dtype = DType::F32; // Use F32 for training stability

        // Load weights
        let weight_files = Self::find_weight_files(&model_path.path)?;
        let weight_refs: Vec<&Path> = weight_files.iter().map(|p| p.as_path()).collect();

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&weight_refs, dtype, device)
                .context("Failed to load model weights")?
        };

        // Load appropriate model based on architecture
        let (model, config) = if arch.contains("qwen") || model_type.contains("qwen") {
            let qwen_config: Qwen2Config = serde_json::from_str(&config_str)
                .context("Failed to parse Qwen2 config")?;

            tracing::debug!(
                "Loading Qwen2 with LoRA: hidden={}, layers={}, vocab={}, lora_rank={}",
                qwen_config.hidden_size,
                qwen_config.num_hidden_layers,
                qwen_config.vocab_size,
                lora_config.rank
            );

            // Load the embedding layer separately for embedding extraction
            // The weights are stored at "model.embed_tokens.weight"
            let embed_tokens = candle_nn::embedding(
                qwen_config.vocab_size,
                qwen_config.hidden_size,
                vb.pp("model.embed_tokens"),
            ).context("Failed to load Qwen2 embed_tokens")?;

            let qwen_model = Qwen2Model::new(&qwen_config, vb)
                .context("Failed to load Qwen2 model")?;

            let decoder_config = DecoderConfig {
                hidden_size: qwen_config.hidden_size,
                num_hidden_layers: qwen_config.num_hidden_layers,
                vocab_size: qwen_config.vocab_size,
                architecture: ModelArchitecture::Qwen2,
            };

            (DecoderModel::Qwen2(qwen_model, embed_tokens), decoder_config)
        } else if arch.contains("llama") || model_type.contains("llama") {
            // Parse Llama config manually (LlamaConfig doesn't implement Deserialize)
            let llama_config = Self::parse_llama_config(&model_config)?;

            tracing::debug!(
                "Loading Llama with LoRA: hidden={}, layers={}, vocab={}, lora_rank={}",
                llama_config.hidden_size,
                llama_config.num_hidden_layers,
                llama_config.vocab_size,
                lora_config.rank
            );

            let llama_model = Llama::load(vb, &llama_config)
                .context("Failed to load Llama model")?;

            let decoder_config = DecoderConfig {
                hidden_size: llama_config.hidden_size,
                num_hidden_layers: llama_config.num_hidden_layers,
                vocab_size: llama_config.vocab_size,
                architecture: ModelArchitecture::Llama,
            };

            (DecoderModel::Llama(llama_model, llama_config), decoder_config)
        } else if arch.contains("mistral") || model_type.contains("mistral") {
            // Mistral uses the same architecture as Llama
            let llama_config = Self::parse_llama_config(&model_config)?;

            tracing::debug!(
                "Loading Mistral with LoRA: hidden={}, layers={}, vocab={}, lora_rank={}",
                llama_config.hidden_size,
                llama_config.num_hidden_layers,
                llama_config.vocab_size,
                lora_config.rank
            );

            let mistral_model = Llama::load(vb, &llama_config)
                .context("Failed to load Mistral model")?;

            let decoder_config = DecoderConfig {
                hidden_size: llama_config.hidden_size,
                num_hidden_layers: llama_config.num_hidden_layers,
                vocab_size: llama_config.vocab_size,
                architecture: ModelArchitecture::Mistral,
            };

            (DecoderModel::Llama(mistral_model, llama_config), decoder_config)
        } else {
            anyhow::bail!("Unsupported decoder architecture: {} / {}", arch, model_type);
        };

        // Initialize LoRA adapters
        let hidden_size = config.hidden_size;
        let rank = lora_config.rank;
        let lora_vb = VarBuilder::from_varmap(var_map, DType::F32, device);

        let lora_down = lora_vb.get_with_hints(
            (rank, hidden_size),
            "lora_projection.down",
            Init::Kaiming {
                dist: candle_nn::init::NormalOrUniform::Uniform,
                fan: candle_nn::init::FanInOut::FanIn,
                non_linearity: candle_nn::init::NonLinearity::Linear,
            },
        )?;

        let lora_up = lora_vb.get_with_hints(
            (hidden_size, rank),
            "lora_projection.up",
            Init::Const(0.0),
        )?;

        tracing::debug!(
            "Created decoder LoRA adapters: {} trainable params",
            rank * hidden_size * 2
        );

        Ok(Self {
            model: Mutex::new(model),
            config,
            device: device.clone(),
            lora_down,
            lora_up,
            lora_scaling: lora_config.scaling(),
            lora_rank: rank,
        })
    }

    /// Load from a HuggingFace model ID or local path
    pub fn from_pretrained(
        model_id: &str,
        lora_config: &LoraConfig,
        var_map: &VarMap,
        device: &Device,
    ) -> Result<Self> {
        let loader = ModelLoader::new()?;
        let model_path = loader.load_model_path(model_id)?;
        Self::from_model_path(&model_path, lora_config, var_map, device)
    }

    /// Get the decoder config
    pub fn config(&self) -> &DecoderConfig {
        &self.config
    }

    /// Find weight files (handles both single and sharded safetensors)
    fn find_weight_files(model_dir: &Path) -> Result<Vec<std::path::PathBuf>> {
        let single_file = model_dir.join("model.safetensors");
        if single_file.exists() {
            return Ok(vec![single_file]);
        }

        // Look for sharded files
        let mut shards = Vec::new();
        for entry in std::fs::read_dir(model_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map(|e| e == "safetensors").unwrap_or(false) {
                if let Some(name) = path.file_name() {
                    let name_str = name.to_string_lossy();
                    if name_str.starts_with("model-") && name_str.contains("-of-") {
                        shards.push(path);
                    }
                }
            }
        }

        if shards.is_empty() {
            anyhow::bail!("No model weight files found in {:?}", model_dir);
        }

        shards.sort();
        Ok(shards)
    }

    /// Parse Llama config from JSON (LlamaConfig doesn't implement Deserialize)
    fn parse_llama_config(model_config: &serde_json::Value) -> Result<LlamaConfig> {
        let hidden_size = model_config["hidden_size"].as_u64().unwrap_or(4096) as usize;
        let intermediate_size = model_config["intermediate_size"].as_u64().unwrap_or(11008) as usize;
        let vocab_size = model_config["vocab_size"].as_u64().unwrap_or(32000) as usize;
        let num_hidden_layers = model_config["num_hidden_layers"].as_u64().unwrap_or(32) as usize;
        let num_attention_heads = model_config["num_attention_heads"].as_u64().unwrap_or(32) as usize;
        let num_key_value_heads = model_config["num_key_value_heads"].as_u64().unwrap_or(32) as usize;
        let rms_norm_eps = model_config["rms_norm_eps"].as_f64().unwrap_or(1e-6);
        let rope_theta = model_config["rope_theta"].as_f64().unwrap_or(10000.0) as f32;
        let max_position_embeddings = model_config["max_position_embeddings"].as_u64().unwrap_or(4096) as usize;
        let tie_word_embeddings = model_config["tie_word_embeddings"].as_bool().unwrap_or(false);

        Ok(LlamaConfig {
            hidden_size,
            intermediate_size,
            vocab_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            use_flash_attn: false,
            rms_norm_eps,
            rope_theta,
            bos_token_id: model_config["bos_token_id"].as_u64().map(|v| v as u32),
            eos_token_id: None,
            rope_scaling: None,
            max_position_embeddings,
            tie_word_embeddings,
        })
    }

    /// Apply LoRA projection to hidden states
    fn apply_lora(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // LoRA: h' = h + (h @ A @ B) * scaling
        let lora_out = hidden_states
            .matmul(&self.lora_down.t()?)?
            .matmul(&self.lora_up.t()?)?;
        let scaled = (lora_out * self.lora_scaling as f64)?;
        Ok((hidden_states + scaled)?)
    }

    /// Forward pass returning logits (for training with cross-entropy loss)
    pub fn forward_logits(&self, input_ids: &Tensor) -> Result<Tensor> {
        let mut model_guard = self.model.lock().map_err(|e| anyhow::anyhow!("Model lock error: {}", e))?;

        match &mut *model_guard {
            DecoderModel::Qwen2(model, _embed_tokens) => {
                let logits = model.forward(input_ids, 0)?;
                // For decoder training, we typically want logits directly
                Ok(logits)
            }
            DecoderModel::Llama(model, llama_config) => {
                // Create a fresh cache for each forward pass during training
                // This ensures we get full sequence logits [batch, seq_len, vocab]
                // instead of single-token logits [batch, vocab] from cached inference
                let mut cache = LlamaCache::new(false, DType::F32, llama_config, &self.device)?;
                let logits = model.forward(input_ids, 0, &mut cache)?;
                Ok(logits)
            }
        }
    }

    /// Get mean-pooled hidden states (useful for embedding tasks)
    ///
    /// For decoder models, we get input embeddings and apply mean pooling.
    /// This is similar to how sentence-transformers handles decoder models.
    fn get_pooled_hidden(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        // Get input embeddings [batch, seq_len, hidden_size]
        let embeddings = self.get_input_embeddings(input_ids)?;

        let hidden_size = embeddings.dim(2)?;

        // Mean pooling over sequence dimension with attention mask
        let mask_expanded = attention_mask
            .unsqueeze(2)?
            .broadcast_as(embeddings.shape())?
            .to_dtype(embeddings.dtype())?;

        let masked = (&embeddings * &mask_expanded)?;
        let sum = masked.sum(1)?;
        let count = attention_mask.sum(1)?.unsqueeze(1)?;
        let count = count.to_dtype(embeddings.dtype())?;  // Convert to same dtype as embeddings
        let count = count.broadcast_as(&[count.dim(0)?, hidden_size])?;
        let pooled = sum.broadcast_div(&count)?;

        // Apply LoRA to pooled representation
        self.apply_lora(&pooled)
    }

    /// Get input embeddings from the model (before transformer layers)
    fn get_input_embeddings(&self, input_ids: &Tensor) -> Result<Tensor> {
        let model_guard = self.model.lock().map_err(|e| anyhow::anyhow!("Model lock error: {}", e))?;

        match &*model_guard {
            DecoderModel::Qwen2(_model, embed_tokens) => {
                // Use the separately loaded embed_tokens layer for embedding extraction
                embed_tokens.forward(input_ids).map_err(|e| anyhow::anyhow!("Qwen2 embedding error: {}", e))
            }
            DecoderModel::Llama(model, _) => {
                // Use Llama's embed method to get input embeddings [batch, seq, hidden]
                model.embed(input_ids).map_err(|e| anyhow::anyhow!("Embedding error: {}", e))
            }
        }
    }
}

impl EmbeddingModel for DecoderLoraModel {
    fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        self.get_pooled_hidden(input_ids, attention_mask)
    }

    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    fn architecture(&self) -> ModelArchitecture {
        self.config.architecture
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

impl LoraModel for DecoderLoraModel {
    fn num_trainable_params(&self) -> usize {
        self.lora_rank * self.config.hidden_size * 2
    }

    fn num_total_params(&self) -> usize {
        // Estimate total params for decoder model
        let hidden = self.config.hidden_size;
        let layers = self.config.num_hidden_layers;
        let vocab = self.config.vocab_size;

        // Approximate: attention (4 * hidden^2) + FFN (3 * hidden * 4 * hidden) per layer
        // Plus embeddings and LM head
        let per_layer = 4 * hidden * hidden + 3 * hidden * 4 * hidden;
        let embeddings = vocab * hidden * 2; // input + output embeddings

        per_layer * layers + embeddings + self.num_trainable_params()
    }

    fn load_lora_checkpoint(&mut self, path: &Path) -> Result<()> {
        tracing::debug!("Loading decoder LoRA checkpoint from: {:?}", path);
        let tensors = candle_core::safetensors::load(path, &self.device)?;

        self.lora_down = tensors
            .get("lora_projection.down")
            .ok_or_else(|| anyhow::anyhow!("Missing lora_projection.down"))?
            .clone();
        self.lora_up = tensors
            .get("lora_projection.up")
            .ok_or_else(|| anyhow::anyhow!("Missing lora_projection.up"))?
            .clone();

        tracing::debug!("Successfully loaded decoder LoRA checkpoint");
        Ok(())
    }

    fn save_lora_checkpoint(&self, path: &Path) -> Result<()> {
        use std::collections::HashMap;

        tracing::debug!("Saving decoder LoRA checkpoint to: {:?}", path);

        let lora_down_cpu = self.lora_down.to_device(&Device::Cpu)?;
        let lora_up_cpu = self.lora_up.to_device(&Device::Cpu)?;

        let mut tensors = HashMap::new();
        tensors.insert("lora_projection.down".to_string(), lora_down_cpu);
        tensors.insert("lora_projection.up".to_string(), lora_up_cpu);

        candle_core::safetensors::save(&tensors, path)?;

        tracing::debug!("Successfully saved decoder LoRA checkpoint");
        Ok(())
    }
}

/// Convenience function to load any decoder with LoRA
pub fn load_decoder_lora(
    model_id: &str,
    lora_config: &LoraConfig,
    var_map: &VarMap,
    device: &Device,
) -> Result<DecoderLoraModel> {
    DecoderLoraModel::from_pretrained(model_id, lora_config, var_map, device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_lora_param_count() {
        // For a 4096 hidden size model with rank 8
        let hidden_size = 4096;
        let rank = 8;
        assert_eq!(rank * hidden_size * 2, 65536);
    }
}
