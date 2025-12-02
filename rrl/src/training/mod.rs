//! Fine-tuning with LoRA/QLoRA
//!
//! Implements RAG-aware fine-tuning with grounding-aware loss using Candle.
//!
//! # Features
//!
//! This module requires the `training` feature to be enabled:
//! ```bash
//! cargo build --features training
//! ```
//!
//! For GPU support:
//! ```bash
//! cargo build --features cuda    # NVIDIA GPU
//! cargo build --features metal   # Apple GPU
//! ```
//!
//! # Modules
//!
//! - `device` - CPU/CUDA/Metal device abstraction
//! - `hub` - HuggingFace Hub integration
//! - `models` - BERT-family model wrappers
//! - `lora` - LoRA adapter implementation
//! - `loss` - Loss functions (contrastive, grounding)
//! - `optimizer` - AdamW optimizer
//! - `trainer` - Training loop
//! - `dataset` - Dataset loading and batching

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

// Re-exports
pub use device::DevicePreference;
pub use lora::LoraConfig;
#[cfg(feature = "training")]
pub use device::select_device;

#[cfg(feature = "training")]
pub use dataset::{DatasetConfig, TrainingDataset, TrainingExample};
#[cfg(feature = "training")]
pub use evaluation::{evaluate_multiple_choice, load_recipe_mpr_examples, MCExample, MultipleChoiceResult};
#[cfg(feature = "training")]
pub use hub::HubApi;
#[cfg(feature = "training")]
pub use models::{load_bert_lora, load_model, BertLoraModel};
#[cfg(feature = "training")]
pub use trainer::{Trainer, TrainingConfig, TrainingResult};
