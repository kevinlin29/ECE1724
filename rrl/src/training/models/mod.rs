//! Model implementations for training

pub mod common;
pub mod loader;
pub mod tokenizer;
pub mod encoders;

// Future modules
// pub mod decoders;
// pub mod encoder_decoder;

// Re-exports for convenience
pub use common::{
    EmbeddingModel, LoraModel, ModelArchitecture, PoolingStrategy,
    detect_architecture,
};
pub use loader::{
    UniversalModelLoader, load_any_model_lora, load_any_model_auto,
};
pub use tokenizer::{TokenizerWrapper, EncodedInput, BatchEncodedInput};

// Re-export encoder models
pub use encoders::{
    BertForEmbedding, BertLoraModel, CudaBertModel,
    load_bert_lora,
};

// When you add RoBERTa:
// pub use encoders::RobertaLoraModel;