//! RoBERTa model with trainable LoRA adapters

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Init, VarBuilder, VarMap};
use std::path::Path;

use super::super::common::{EmbeddingModel, LoraModel, ModelArchitecture, PoolingStrategy};
use super::bert_cuda::{CudaBertConfig, CudaBertModel};
use super::encoder_utils::apply_pooling;
use crate::training::hub::ModelPath;
use crate::training::lora::LoraConfig;

/// RoBERTa model with LoRA adapters
pub struct RobertaLoraModel {
    base_model: CudaBertModel,
    config: CudaBertConfig,
    device: Device,
    pooling: PoolingStrategy,
    lora_down: Tensor,
    lora_up: Tensor,
    lora_scaling: f32,
    lora_rank: usize,
}

impl RobertaLoraModel {
    pub fn from_model_path(
        model_path: &ModelPath,
        lora_config: &LoraConfig,
        var_map: &VarMap,
        device: &Device,
    ) -> Result<Self> {
        model_path.validate()?;

        let config_str = std::fs::read_to_string(&model_path.config_file)?;
        let config: CudaBertConfig = serde_json::from_str(&config_str)?;

        tracing::info!(
            "Loading RoBERTa: hidden={}, layers={}, rank={}",
            config.hidden_size, config.num_hidden_layers, lora_config.rank
        );

        let base_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&model_path.weights_file], DType::F32, device)?
        };

        let base_model = CudaBertModel::load(base_vb.clone(), &config)
            .or_else(|_| CudaBertModel::load(base_vb.pp("roberta"), &config))
            .or_else(|_| CudaBertModel::load(base_vb.pp("encoder"), &config))?;

        let hidden_size = config.hidden_size;
        let rank = lora_config.rank;
        let lora_vb = VarBuilder::from_varmap(var_map, DType::F32, device);

        let lora_down = lora_vb.get_with_hints(
            (rank, hidden_size), "lora_projection.down",
            Init::Kaiming {
                dist: candle_nn::init::NormalOrUniform::Uniform,
                fan: candle_nn::init::FanInOut::FanIn,
                non_linearity: candle_nn::init::NonLinearity::Linear,
            },
        )?;

        let lora_up = lora_vb.get_with_hints(
            (hidden_size, rank), "lora_projection.up", Init::Const(0.0)
        )?;

        Ok(Self {
            base_model, config, device: device.clone(),
            pooling: PoolingStrategy::Mean,
            lora_down, lora_up,
            lora_scaling: lora_config.scaling(),
            lora_rank: rank,
        })
    }

    pub fn from_pretrained(model_id: &str, lora_config: &LoraConfig, var_map: &VarMap, device: &Device) -> Result<Self> {
        use crate::training::hub::ModelLoader;
        let loader = ModelLoader::new()?;
        Self::from_model_path(&loader.load_model_path(model_id)?, lora_config, var_map, device)
    }

    fn apply_lora(&self, embeddings: &Tensor) -> Result<Tensor> {
        let lora_out = embeddings.matmul(&self.lora_down.t()?)?.matmul(&self.lora_up.t()?)?;
        Ok((embeddings + (lora_out * self.lora_scaling as f64)?)?)
    }

    pub fn forward_with_lora(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let token_type_ids = Tensor::zeros(input_ids.shape(), DType::U32, &self.device)?;
        let hidden = self.base_model.forward(input_ids, &token_type_ids, Some(attention_mask))?;
        let pooled = apply_pooling(&hidden, attention_mask, self.pooling)?;
        self.apply_lora(&pooled)
    }
}

impl EmbeddingModel for RobertaLoraModel {
    fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        self.forward_with_lora(input_ids, attention_mask)
    }
    fn hidden_size(&self) -> usize { self.config.hidden_size }
    fn architecture(&self) -> ModelArchitecture { ModelArchitecture::Roberta }
    fn device(&self) -> &Device { &self.device }
}

impl LoraModel for RobertaLoraModel {
    fn num_trainable_params(&self) -> usize { self.lora_rank * self.config.hidden_size * 2 }
    fn num_total_params(&self) -> usize {
        self.config.hidden_size * self.config.hidden_size * 4 * self.config.num_hidden_layers
            + self.config.hidden_size * self.config.intermediate_size * 2 * self.config.num_hidden_layers
            + self.config.vocab_size * self.config.hidden_size
            + self.num_trainable_params()
    }
    
    fn load_lora_checkpoint(&mut self, path: &Path) -> Result<()> {
        let tensors = candle_core::safetensors::load(path, &self.device)?;
        self.lora_down = tensors.get("lora_projection.down").ok_or_else(|| anyhow::anyhow!("Missing down"))?.clone();
        self.lora_up = tensors.get("lora_projection.up").ok_or_else(|| anyhow::anyhow!("Missing up"))?.clone();
        Ok(())
    }
    
    fn save_lora_checkpoint(&self, path: &Path) -> Result<()> {
        let mut tensors = std::collections::HashMap::new();
        tensors.insert("lora_projection.down".to_string(), self.lora_down.to_device(&Device::Cpu)?);
        tensors.insert("lora_projection.up".to_string(), self.lora_up.to_device(&Device::Cpu)?);
        candle_core::safetensors::save(&tensors, path)?;
        Ok(())
    }
}