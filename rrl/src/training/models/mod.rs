//! Model implementations for training

pub mod common;
pub mod loader;
pub mod tokenizer;
pub mod encoders;
pub mod decoders;

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

// Re-export decoder models
pub use decoders::{DecoderConfig, DecoderLoraModel, load_decoder_lora};