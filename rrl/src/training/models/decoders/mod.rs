//! Decoder model implementations for training
//!
//! This module provides LoRA-enabled decoder models for fine-tuning
//! causal language models like Qwen2, LLaMA, and Mistral.

mod decoder_lora;

pub use decoder_lora::{DecoderConfig, DecoderLoraModel, load_decoder_lora};
