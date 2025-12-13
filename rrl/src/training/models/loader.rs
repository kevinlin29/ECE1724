//! Universal model loader supporting multiple architectures

use anyhow::Result;
use candle_core::Device;
use candle_nn::VarMap;

use super::common::{detect_architecture, LoraModel, ModelArchitecture};
use super::encoders::{BertLoraModel, RobertaLoraModel};
use crate::training::hub::{HubModelConfig, ModelLoader as HubLoader};
use crate::training::lora::LoraConfig;

pub struct UniversalModelLoader {
    hub_loader: HubLoader,
}

impl UniversalModelLoader {
    pub fn new() -> Result<Self> {
        Ok(Self { hub_loader: HubLoader::new()? })
    }
    
    pub fn load_lora(
        &self,
        model_id_or_path: &str,
        lora_config: &LoraConfig,
        var_map: &VarMap,
        device: &Device,
    ) -> Result<Box<dyn LoraModel>> {
        let model_path = self.hub_loader.load_model_path(model_id_or_path)?;
        let config = HubModelConfig::from_file(&model_path.config_file)?;
        
        let arch = detect_architecture(config.model_type.as_deref(), &config.architectures)?;
        
        tracing::info!("Loading {} model: {}", arch, model_id_or_path);
        
        match arch {
            ModelArchitecture::Bert => {
                let model = BertLoraModel::from_model_path(&model_path, lora_config, var_map, device)?;
                Ok(Box::new(model))
            }
            ModelArchitecture::Roberta => {
                let model = RobertaLoraModel::from_model_path(&model_path, lora_config, var_map, device)?;
                Ok(Box::new(model))
            }
            ModelArchitecture::DistilBert | ModelArchitecture::Albert => {
                tracing::warn!("{} using BERT implementation", arch);
                let model = BertLoraModel::from_model_path(&model_path, lora_config, var_map, device)?;
                Ok(Box::new(model))
            }
            _ => Err(anyhow::anyhow!("Architecture {} not yet supported", arch)),
        }
    }
    
    pub fn recommend_lora_config(&self, model_id_or_path: &str) -> Result<LoraConfig> {
        let config = self.hub_loader.load_config(model_id_or_path)?;
        let hidden_size = config.hidden_size.unwrap_or(768);
        
        let (rank, alpha) = match hidden_size {
            ..=768 => (8, 16.0),
            769..=1024 => (16, 32.0),
            1025..=1536 => (32, 64.0),
            _ => (64, 128.0),
        };
        
        Ok(LoraConfig::new(rank, alpha))
    }
}

pub fn load_any_model_lora(
    model_id: &str,
    lora_config: &LoraConfig,
    var_map: &VarMap,
    device: &Device,
) -> Result<Box<dyn LoraModel>> {
    UniversalModelLoader::new()?.load_lora(model_id, lora_config, var_map, device)
}

pub fn load_any_model_auto(
    model_id: &str,
    var_map: &VarMap,
    device: &Device,
) -> Result<Box<dyn LoraModel>> {
    let loader = UniversalModelLoader::new()?;
    let lora_config = loader.recommend_lora_config(model_id)?;
    loader.load_lora(model_id, &lora_config, var_map, device)
}