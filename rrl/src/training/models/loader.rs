//! Universal model loader

use anyhow::Result;
use candle_core::Device;
use candle_nn::VarMap;

use super::common::{detect_architecture, LoraModel, ModelArchitecture};
use super::encoders::BertLoraModel;
use crate::training::hub::{HubModelConfig, ModelLoader as HubLoader};
use crate::training::lora::LoraConfig;

/// Universal model loader that supports all architectures
pub struct UniversalModelLoader {
    hub_loader: HubLoader,
}

impl UniversalModelLoader {
    /// Create a new universal loader
    pub fn new() -> Result<Self> {
        Ok(Self {
            hub_loader: HubLoader::new()?,
        })
    }
    
    /// Load any model with LoRA for training
    pub fn load_lora(
        &self,
        model_id_or_path: &str,
        lora_config: &LoraConfig,
        var_map: &VarMap,
        device: &Device,
    ) -> Result<Box<dyn LoraModel>> {
        let model_path = self.hub_loader.load_model_path(model_id_or_path)?;
        let config = HubModelConfig::from_file(&model_path.config_file)?;
        
        let arch = detect_architecture(
            config.model_type.as_deref(),
            &config.architectures,
        )?;
        
        tracing::info!(
            "Detected architecture: {} for model {}",
            arch,
            model_id_or_path
        );
        
        match arch {
            ModelArchitecture::Bert
            | ModelArchitecture::Roberta
            | ModelArchitecture::DistilBert
            | ModelArchitecture::Albert => {
                // For now, all use BERT implementation
                let model = BertLoraModel::from_model_path(
                    &model_path,
                    lora_config,
                    var_map,
                    device,
                )?;
                Ok(Box::new(model))
            }
            
            _ => Err(anyhow::anyhow!(
                "Model architecture {} not yet supported",
                arch
            )),
        }
    }
    
    /// Get recommended LoRA configuration for a model
    pub fn recommend_lora_config(&self, model_id_or_path: &str) -> Result<LoraConfig> {
        let config = self.hub_loader.load_config(model_id_or_path)?;
        
        // Recommend based on model size
        let hidden_size = config.hidden_size.unwrap_or(768);
        
        let (rank, alpha) = if hidden_size <= 768 {
            (8, 16.0)
        } else if hidden_size <= 1024 {
            (16, 32.0)
        } else {
            (32, 64.0)
        };
        
        Ok(LoraConfig::new(rank, alpha))
    }
}

impl Default for UniversalModelLoader {
    fn default() -> Self {
        Self::new().expect("Failed to create UniversalModelLoader")
    }
}

/// Load any model with LoRA
pub fn load_any_model_lora(
    model_id: &str,
    lora_config: &LoraConfig,
    var_map: &VarMap,
    device: &Device,
) -> Result<Box<dyn LoraModel>> {
    UniversalModelLoader::new()?.load_lora(model_id, lora_config, var_map, device)
}

/// Load model with auto-detected LoRA config
pub fn load_any_model_auto(
    model_id: &str,
    var_map: &VarMap,
    device: &Device,
) -> Result<Box<dyn LoraModel>> {
    let loader = UniversalModelLoader::new()?;
    let lora_config = loader.recommend_lora_config(model_id)?;
    
    tracing::info!(
        "Auto-detected LoRA config for {}: rank={}, alpha={}",
        model_id,
        lora_config.rank,
        lora_config.alpha
    );
    
    loader.load_lora(model_id, &lora_config, var_map, device)
}