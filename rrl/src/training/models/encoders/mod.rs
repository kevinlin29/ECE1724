// src/training/models/encoders/mod.rs
//! Encoder-only transformer models

mod bert;
mod bert_cuda;
mod bert_lora;
// mod roberta_lora;  // When you add it
mod encoder_utils;

// Re-exports
pub use bert::BertForEmbedding;
pub use bert_cuda::{CudaBertConfig, CudaBertModel, CudaLayerNorm};
pub use bert_lora::{BertLoraModel, load_bert_lora};
// pub use roberta_lora::RobertaLoraModel;  // When added
pub use encoder_utils::{apply_pooling, normalize_embeddings};