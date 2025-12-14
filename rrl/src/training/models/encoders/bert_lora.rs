//! BERT model with trainable LoRA adapters

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Init, VarBuilder, VarMap};
use std::path::Path;

use super::super::common::{EmbeddingModel, LoraModel, ModelArchitecture, PoolingStrategy};
use super::bert_cuda::{CudaBertConfig, CudaBertModel};
use super::encoder_utils::{apply_pooling, normalize_embeddings};
use crate::training::hub::ModelPath;
use crate::training::lora::LoraConfig;

/// BERT model with LoRA adapters for fine-tuning
pub struct BertLoraModel {
    base_model: CudaBertModel,
    config: CudaBertConfig,
    device: Device,
    pooling: PoolingStrategy,
    lora_down: Tensor,
    lora_up: Tensor,
    lora_scaling: f32,
    lora_rank: usize,
}

impl BertLoraModel {
    pub fn from_model_path(
        model_path: &ModelPath,
        lora_config: &LoraConfig,
        var_map: &VarMap,
        device: &Device,
    ) -> Result<Self> {
        model_path.validate()?;

        let config_str = std::fs::read_to_string(&model_path.config_file)?;
        let config: CudaBertConfig = serde_json::from_str(&config_str)?;

        tracing::debug!(
            "Loading BERT with LoRA: hidden_size={}, layers={}, lora_rank={}",
            config.hidden_size, config.num_hidden_layers, lora_config.rank
        );

        let base_vb = if model_path.is_safetensors() {
            unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &[&model_path.weights_file],
                    DType::F32,
                    device,
                )?
            }
        } else {
            return Err(anyhow::anyhow!("Only safetensors format supported"));
        };

        let base_model = CudaBertModel::load(base_vb.clone(), &config)
            .or_else(|_| CudaBertModel::load(base_vb.pp("bert"), &config))?;

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

        tracing::debug!("Created LoRA adapters: {} params", rank * hidden_size * 2);

        Ok(Self {
            base_model,
            config,
            device: device.clone(),
            pooling: PoolingStrategy::Mean,
            lora_down,
            lora_up,
            lora_scaling: lora_config.scaling(),
            lora_rank: rank,
        })
    }

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

    pub fn with_pooling(mut self, pooling: PoolingStrategy) -> Self {
        self.pooling = pooling;
        self
    }

    pub fn config(&self) -> &CudaBertConfig {
        &self.config
    }

    fn forward_base(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let token_type_ids = Tensor::zeros(input_ids.shape(), DType::U32, &self.device)?;
        self.base_model.forward(input_ids, &token_type_ids, attention_mask)
    }

    fn apply_lora(&self, embeddings: &Tensor) -> Result<Tensor> {
        let lora_out = embeddings
            .matmul(&self.lora_down.t()?)?
            .matmul(&self.lora_up.t()?)?;
        let scaled = (lora_out * self.lora_scaling as f64)?;
        Ok((embeddings + scaled)?)
    }

    pub fn forward_with_lora(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let hidden_states = self.forward_base(input_ids, Some(attention_mask))?;
        let pooled = apply_pooling(&hidden_states, attention_mask, self.pooling)?;
        self.apply_lora(&pooled)
    }

    pub fn encode_normalized(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let embeddings = self.forward_with_lora(input_ids, attention_mask)?;
        normalize_embeddings(&embeddings)
    }

    pub fn from_pretrained_with_checkpoint(
        model_id: &str,
        lora_config: &LoraConfig,
        checkpoint_path: impl AsRef<Path>,
        device: &Device,
    ) -> Result<Self> {
        let var_map = VarMap::new();
        let mut model = Self::from_pretrained(model_id, lora_config, &var_map, device)?;
        model.load_lora_checkpoint(checkpoint_path.as_ref())?;
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

impl LoraModel for BertLoraModel {
    fn num_trainable_params(&self) -> usize {
        self.lora_rank * self.config.hidden_size * 2
    }
    
    fn num_total_params(&self) -> usize {
        let bert_params = self.config.hidden_size
            * self.config.hidden_size
            * 4
            * self.config.num_hidden_layers
            + self.config.hidden_size * self.config.intermediate_size * 2
            * self.config.num_hidden_layers
            + self.config.vocab_size * self.config.hidden_size;
        bert_params + self.num_trainable_params()
    }
    
    fn load_lora_checkpoint(&mut self, path: &Path) -> Result<()> {
        tracing::debug!("Loading LoRA checkpoint from: {:?}", path);
        let tensors = candle_core::safetensors::load(path, &self.device)?;
        
        self.lora_down = tensors.get("lora_projection.down")
            .ok_or_else(|| anyhow::anyhow!("Missing lora_projection.down"))?
            .clone();
        self.lora_up = tensors.get("lora_projection.up")
            .ok_or_else(|| anyhow::anyhow!("Missing lora_projection.up"))?
            .clone();
        
        tracing::debug!("Successfully loaded LoRA checkpoint");
        Ok(())
    }
    
    fn save_lora_checkpoint(&self, path: &Path) -> Result<()> {
        use std::collections::HashMap;
        
        tracing::debug!("Saving LoRA checkpoint to: {:?}", path);
        
        let lora_down_cpu = self.lora_down.to_device(&Device::Cpu)?;
        let lora_up_cpu = self.lora_up.to_device(&Device::Cpu)?;
        
        let mut tensors = HashMap::new();
        tensors.insert("lora_projection.down".to_string(), lora_down_cpu);
        tensors.insert("lora_projection.up".to_string(), lora_up_cpu);
        
        candle_core::safetensors::save(&tensors, path)?;
        
        tracing::debug!("Successfully saved LoRA checkpoint");
        Ok(())
    }
}

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
        let hidden_size = 768;
        let rank = 8;
        assert_eq!(rank * hidden_size * 2, 12288);
    }
}