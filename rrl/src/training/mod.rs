//! Fine-tuning and training utilities

pub mod device;
pub mod lora;
pub mod loss;

#[cfg(feature = "training")]
pub mod dataset;
#[cfg(feature = "training")]
pub mod evaluation;
#[cfg(feature = "training")]
pub mod hub;
#[cfg(feature = "training")]
pub mod models;
#[cfg(feature = "training")]
pub mod optimizer;
#[cfg(feature = "training")]
pub mod trainer;

// Re-exports for convenience
pub use device::{DevicePreference, select_device, device_info, DeviceInfo};
pub use lora::{LoraConfig, LoraStats};
pub use loss::{ContrastiveLoss, ContrastiveLossConfig};

#[cfg(feature = "training")]
pub use dataset::{
    DatasetConfig, TrainingDataset, TrainingExample,
};

#[cfg(feature = "training")]
pub use evaluation::{
    evaluate_multiple_choice, evaluate_retrieval,
    load_recipe_mpr_examples,
    EvaluationResult, RecipeMprExample, RetrievalMetrics,
};

#[cfg(feature = "training")]
pub use models::{
    // Main traits
    EmbeddingModel, LoraModel, ModelArchitecture, PoolingStrategy,
    // Loaders
    UniversalModelLoader, load_any_model_lora, load_any_model_auto,
    // Specific models
    BertLoraModel, load_bert_lora,
    // Tokenizer
    TokenizerWrapper, EncodedInput, BatchEncodedInput,
    // Utilities
    detect_architecture,
};

#[cfg(feature = "training")]
pub use optimizer::AdamW;

#[cfg(feature = "training")]
pub use trainer::{
    Trainer, TrainingConfig, TrainingResult, TrainingMetrics,
};

#[cfg(feature = "training")]
pub use hub::{
    ModelLoader, ModelPath, HubModelConfig,
};