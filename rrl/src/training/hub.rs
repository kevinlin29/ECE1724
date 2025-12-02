//! HuggingFace Hub integration for model downloading
//!
//! Provides functionality to download models from HuggingFace Hub
//! and manage local model cache.

use anyhow::{anyhow, Context, Result};
use hf_hub::api::sync::Api;
use std::path::{Path, PathBuf};

/// HuggingFace Hub API wrapper
pub struct HubApi {
    api: Api,
}

impl HubApi {
    /// Create a new HubApi instance
    pub fn new() -> Result<Self> {
        let api = Api::new().context("Failed to initialize HuggingFace Hub API")?;
        Ok(Self { api })
    }

    /// Download a model from HuggingFace Hub
    ///
    /// # Arguments
    /// * `model_id` - The model identifier (e.g., "bert-base-uncased")
    ///
    /// # Returns
    /// * `ModelPath` - Information about the downloaded model location
    pub fn download_model(&self, model_id: &str) -> Result<ModelPath> {
        tracing::info!("Downloading model from HuggingFace Hub: {}", model_id);

        let repo = self.api.model(model_id.to_string());

        // Download required files
        let config_path = repo
            .get("config.json")
            .context("Failed to download config.json")?;

        tracing::debug!("Downloaded config.json: {:?}", config_path);

        // Try to download model weights (safetensors preferred)
        let weights_path = if let Ok(path) = repo.get("model.safetensors") {
            tracing::debug!("Downloaded model.safetensors: {:?}", path);
            path
        } else if let Ok(path) = repo.get("pytorch_model.bin") {
            tracing::warn!("Safetensors not available, using pytorch_model.bin");
            path
        } else {
            return Err(anyhow!(
                "No model weights found (tried model.safetensors and pytorch_model.bin)"
            ));
        };

        // Try to download tokenizer files
        let tokenizer_path = repo.get("tokenizer.json").ok();
        let tokenizer_config_path = repo.get("tokenizer_config.json").ok();

        if tokenizer_path.is_some() {
            tracing::debug!("Downloaded tokenizer.json");
        }

        // Get the model directory from the config path
        let model_dir = config_path
            .parent()
            .ok_or_else(|| anyhow!("Invalid config path"))?
            .to_path_buf();

        Ok(ModelPath {
            path: model_dir,
            model_id: model_id.to_string(),
            is_local: false,
            config_file: config_path,
            weights_file: weights_path,
            tokenizer_file: tokenizer_path,
            tokenizer_config_file: tokenizer_config_path,
        })
    }

    /// Get model path - downloads if not cached
    pub fn get_model_path(&self, model_id: &str) -> Result<ModelPath> {
        self.download_model(model_id)
    }

    /// Load config from a model
    pub fn load_config(&self, model_id: &str) -> Result<HubModelConfig> {
        let model_path = self.get_model_path(model_id)?;
        HubModelConfig::from_file(&model_path.config_file)
    }
}

impl Default for HubApi {
    fn default() -> Self {
        Self::new().expect("Failed to create HubApi")
    }
}

/// Represents a downloaded or local model path
#[derive(Debug, Clone)]
pub struct ModelPath {
    /// Root directory containing model files
    pub path: PathBuf,
    /// Original model ID (e.g., "bert-base-uncased")
    pub model_id: String,
    /// Whether this is a local path (not downloaded from Hub)
    pub is_local: bool,
    /// Path to config.json
    pub config_file: PathBuf,
    /// Path to model weights (safetensors or bin)
    pub weights_file: PathBuf,
    /// Path to tokenizer.json (optional)
    pub tokenizer_file: Option<PathBuf>,
    /// Path to tokenizer_config.json (optional)
    pub tokenizer_config_file: Option<PathBuf>,
}

impl ModelPath {
    /// Create a ModelPath from a local directory
    pub fn from_local(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        if !path.exists() {
            return Err(anyhow!("Model directory does not exist: {:?}", path));
        }

        let config_file = path.join("config.json");
        if !config_file.exists() {
            return Err(anyhow!("config.json not found in {:?}", path));
        }

        // Try safetensors first, then pytorch
        let weights_file = if path.join("model.safetensors").exists() {
            path.join("model.safetensors")
        } else if path.join("pytorch_model.bin").exists() {
            path.join("pytorch_model.bin")
        } else {
            return Err(anyhow!(
                "No model weights found in {:?} (tried model.safetensors and pytorch_model.bin)",
                path
            ));
        };

        let tokenizer_file = path.join("tokenizer.json");
        let tokenizer_config_file = path.join("tokenizer_config.json");

        Ok(Self {
            model_id: path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            path: path.clone(),
            is_local: true,
            config_file,
            weights_file,
            tokenizer_file: if tokenizer_file.exists() {
                Some(tokenizer_file)
            } else {
                None
            },
            tokenizer_config_file: if tokenizer_config_file.exists() {
                Some(tokenizer_config_file)
            } else {
                None
            },
        })
    }

    /// Check if this is a safetensors model
    pub fn is_safetensors(&self) -> bool {
        self.weights_file
            .extension()
            .map(|e| e == "safetensors")
            .unwrap_or(false)
    }

    /// Validate that all required files exist
    pub fn validate(&self) -> Result<()> {
        if !self.config_file.exists() {
            return Err(anyhow!("Config file not found: {:?}", self.config_file));
        }
        if !self.weights_file.exists() {
            return Err(anyhow!("Weights file not found: {:?}", self.weights_file));
        }
        Ok(())
    }
}

/// Configuration from HuggingFace model's config.json
#[derive(Debug, Clone, serde::Deserialize)]
pub struct HubModelConfig {
    /// Model architectures (e.g., ["BertForMaskedLM"])
    #[serde(default)]
    pub architectures: Vec<String>,

    /// Model type (e.g., "bert", "roberta")
    pub model_type: Option<String>,

    /// Vocabulary size
    pub vocab_size: Option<usize>,

    /// Hidden size (embedding dimension)
    pub hidden_size: Option<usize>,

    /// Number of hidden layers
    pub num_hidden_layers: Option<usize>,

    /// Number of attention heads
    pub num_attention_heads: Option<usize>,

    /// Intermediate size (FFN dimension)
    pub intermediate_size: Option<usize>,

    /// Hidden activation function
    pub hidden_act: Option<String>,

    /// Hidden dropout probability
    pub hidden_dropout_prob: Option<f64>,

    /// Attention dropout probability
    pub attention_probs_dropout_prob: Option<f64>,

    /// Maximum position embeddings
    pub max_position_embeddings: Option<usize>,

    /// Type vocabulary size
    pub type_vocab_size: Option<usize>,

    /// Layer norm epsilon
    pub layer_norm_eps: Option<f64>,

    /// Padding token ID
    pub pad_token_id: Option<usize>,

    /// Any extra fields we don't explicitly handle
    #[serde(flatten)]
    pub extra: serde_json::Value,
}

impl HubModelConfig {
    /// Load config from a JSON file
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("Failed to read config file: {:?}", path.as_ref()))?;
        serde_json::from_str(&content).context("Failed to parse config.json")
    }

    /// Check if this is a BERT-family model
    pub fn is_bert_family(&self) -> bool {
        if let Some(model_type) = &self.model_type {
            matches!(
                model_type.as_str(),
                "bert" | "roberta" | "distilbert" | "albert" | "camembert" | "xlm-roberta"
            )
        } else {
            // Check architectures
            self.architectures.iter().any(|arch| {
                arch.contains("Bert")
                    || arch.contains("Roberta")
                    || arch.contains("DistilBert")
                    || arch.contains("Albert")
            })
        }
    }

    /// Get the model type, inferring from architectures if needed
    pub fn get_model_type(&self) -> Option<String> {
        if let Some(model_type) = &self.model_type {
            return Some(model_type.clone());
        }

        // Try to infer from architectures
        for arch in &self.architectures {
            let arch_lower = arch.to_lowercase();
            if arch_lower.contains("roberta") {
                return Some("roberta".to_string());
            }
            if arch_lower.contains("distilbert") {
                return Some("distilbert".to_string());
            }
            if arch_lower.contains("albert") {
                return Some("albert".to_string());
            }
            if arch_lower.contains("bert") {
                return Some("bert".to_string());
            }
        }

        None
    }

    /// Validate that this config is compatible with our BERT implementation
    pub fn validate_bert_compatibility(&self) -> Result<()> {
        if !self.is_bert_family() {
            return Err(anyhow!(
                "Model is not BERT-family. Architectures: {:?}, model_type: {:?}",
                self.architectures,
                self.model_type
            ));
        }

        if self.hidden_size.is_none() {
            return Err(anyhow!("Config missing required field: hidden_size"));
        }

        if self.num_hidden_layers.is_none() {
            return Err(anyhow!("Config missing required field: num_hidden_layers"));
        }

        if self.num_attention_heads.is_none() {
            return Err(anyhow!("Config missing required field: num_attention_heads"));
        }

        Ok(())
    }
}

/// Model loader that handles both local and HuggingFace models
pub struct ModelLoader {
    hub: HubApi,
}

impl ModelLoader {
    /// Create a new ModelLoader
    pub fn new() -> Result<Self> {
        Ok(Self {
            hub: HubApi::new()?,
        })
    }

    /// Load model path - auto-detects local vs HuggingFace
    ///
    /// If the path exists locally (as a file or directory), treats it as a local path.
    /// Otherwise, downloads from HuggingFace Hub.
    pub fn load_model_path(&self, model_id_or_path: &str) -> Result<ModelPath> {
        // Check if it exists as a local path first
        let local_path = std::path::Path::new(model_id_or_path);
        let is_local = local_path.exists()
            || model_id_or_path.starts_with('.')
            || model_id_or_path.starts_with('/')
            || model_id_or_path.starts_with('~');

        if is_local && local_path.exists() {
            tracing::info!("Loading model from local path: {}", model_id_or_path);
            ModelPath::from_local(model_id_or_path)
        } else if is_local {
            // Starts with ., /, or ~ but doesn't exist
            Err(anyhow!(
                "Local model path does not exist: {}",
                model_id_or_path
            ))
        } else {
            // Treat as HuggingFace model ID (e.g., "bert-base-uncased" or "sentence-transformers/all-MiniLM-L6-v2")
            tracing::info!("Downloading model from HuggingFace Hub: {}", model_id_or_path);
            self.hub.download_model(model_id_or_path)
        }
    }

    /// Load config from a model (local or HuggingFace)
    pub fn load_config(&self, model_id_or_path: &str) -> Result<HubModelConfig> {
        let model_path = self.load_model_path(model_id_or_path)?;
        HubModelConfig::from_file(&model_path.config_file)
    }
}

impl Default for ModelLoader {
    fn default() -> Self {
        Self::new().expect("Failed to create ModelLoader")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_type_detection() {
        let config = HubModelConfig {
            architectures: vec!["BertForMaskedLM".to_string()],
            model_type: Some("bert".to_string()),
            vocab_size: Some(30522),
            hidden_size: Some(768),
            num_hidden_layers: Some(12),
            num_attention_heads: Some(12),
            intermediate_size: Some(3072),
            hidden_act: Some("gelu".to_string()),
            hidden_dropout_prob: Some(0.1),
            attention_probs_dropout_prob: Some(0.1),
            max_position_embeddings: Some(512),
            type_vocab_size: Some(2),
            layer_norm_eps: Some(1e-12),
            pad_token_id: Some(0),
            extra: serde_json::Value::Null,
        };

        assert!(config.is_bert_family());
        assert_eq!(config.get_model_type(), Some("bert".to_string()));
        assert!(config.validate_bert_compatibility().is_ok());
    }

    #[test]
    fn test_model_path_is_local() {
        let loader = ModelLoader::new().unwrap();

        // These should be treated as local
        assert!(loader
            .load_model_path("./my-model")
            .is_err_and(|e| e.to_string().contains("does not exist")));
        assert!(loader
            .load_model_path("/absolute/path")
            .is_err_and(|e| e.to_string().contains("does not exist")));
    }
}
