//! Tokenizer wrapper for HuggingFace tokenizers
//!
//! Provides a convenient interface for tokenizing text for BERT-family models.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::path::Path;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::training::hub::{ModelLoader, ModelPath};

/// Wrapper around HuggingFace tokenizer
pub struct TokenizerWrapper {
    tokenizer: Tokenizer,
    max_length: usize,
}

impl TokenizerWrapper {
    /// Load tokenizer from a file path
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path.as_ref())
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(Self {
            tokenizer,
            max_length: 512,
        })
    }

    /// Load tokenizer from a ModelPath
    pub fn from_model_path(model_path: &ModelPath) -> Result<Self> {
        let tokenizer_path = model_path
            .tokenizer_file
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Tokenizer file not found in model path"))?;

        Self::from_file(tokenizer_path)
    }

    /// Load tokenizer from HuggingFace Hub or local path
    pub fn from_pretrained(model_id_or_path: &str) -> Result<Self> {
        let loader = ModelLoader::new()?;
        let model_path = loader.load_model_path(model_id_or_path)?;
        Self::from_model_path(&model_path)
    }

    /// Set maximum sequence length
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    /// Get the maximum sequence length
    pub fn max_length(&self) -> usize {
        self.max_length
    }

    /// Encode a single text
    ///
    /// # Arguments
    /// * `text` - Text to encode
    /// * `add_special_tokens` - Whether to add [CLS] and [SEP] tokens
    ///
    /// # Returns
    /// * `EncodedInput` - Encoded token IDs and attention mask
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<EncodedInput> {
        let encoding = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        Ok(EncodedInput {
            input_ids: encoding.get_ids().to_vec(),
            attention_mask: encoding.get_attention_mask().to_vec(),
            token_type_ids: encoding.get_type_ids().to_vec(),
        })
    }

    /// Encode a batch of texts with padding
    ///
    /// # Arguments
    /// * `texts` - Texts to encode
    /// * `add_special_tokens` - Whether to add special tokens
    ///
    /// # Returns
    /// * `BatchEncodedInput` - Batch of encoded inputs
    pub fn encode_batch(&self, texts: &[String], add_special_tokens: bool) -> Result<BatchEncodedInput> {
        // Configure padding
        let mut tokenizer = self.tokenizer.clone();
        tokenizer
            .with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                ..Default::default()
            }));

        // Configure truncation
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: self.max_length,
                ..Default::default()
            }))
            .map_err(|e| anyhow::anyhow!("Failed to set truncation: {}", e))?;

        // Encode batch
        let encodings = tokenizer
            .encode_batch(texts.to_vec(), add_special_tokens)
            .map_err(|e| anyhow::anyhow!("Batch tokenization failed: {}", e))?;

        // Convert to batch format
        let batch_size = encodings.len();
        let seq_len = encodings.first().map(|e| e.get_ids().len()).unwrap_or(0);

        let mut input_ids = Vec::with_capacity(batch_size * seq_len);
        let mut attention_mask = Vec::with_capacity(batch_size * seq_len);
        let mut token_type_ids = Vec::with_capacity(batch_size * seq_len);

        for encoding in &encodings {
            input_ids.extend(encoding.get_ids());
            attention_mask.extend(encoding.get_attention_mask());
            token_type_ids.extend(encoding.get_type_ids());
        }

        Ok(BatchEncodedInput {
            input_ids,
            attention_mask,
            token_type_ids,
            batch_size,
            seq_len,
        })
    }

    /// Encode texts to tensors ready for model input
    ///
    /// # Arguments
    /// * `texts` - Texts to encode
    /// * `device` - Device to create tensors on
    ///
    /// # Returns
    /// * Tuple of (input_ids, attention_mask) tensors
    pub fn encode_to_tensors(
        &self,
        texts: &[String],
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let batch = self.encode_batch(texts, true)?;
        batch.to_tensors(device)
    }

    /// Decode token IDs back to text
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }
}

/// Encoded input for a single text
#[derive(Debug, Clone)]
pub struct EncodedInput {
    /// Token IDs
    pub input_ids: Vec<u32>,
    /// Attention mask (1 for real tokens, 0 for padding)
    pub attention_mask: Vec<u32>,
    /// Token type IDs (for sentence pair tasks)
    pub token_type_ids: Vec<u32>,
}

impl EncodedInput {
    /// Get sequence length
    pub fn len(&self) -> usize {
        self.input_ids.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.input_ids.is_empty()
    }

    /// Convert to tensors
    pub fn to_tensors(&self, device: &Device) -> Result<(Tensor, Tensor)> {
        let input_ids = Tensor::new(&self.input_ids[..], device)?
            .to_dtype(DType::U32)?
            .unsqueeze(0)?; // Add batch dimension

        let attention_mask = Tensor::new(&self.attention_mask[..], device)?
            .to_dtype(DType::F32)?
            .unsqueeze(0)?;

        Ok((input_ids, attention_mask))
    }
}

/// Batch encoded input
#[derive(Debug, Clone)]
pub struct BatchEncodedInput {
    /// Flattened token IDs [batch_size * seq_len]
    pub input_ids: Vec<u32>,
    /// Flattened attention mask
    pub attention_mask: Vec<u32>,
    /// Flattened token type IDs
    pub token_type_ids: Vec<u32>,
    /// Batch size
    pub batch_size: usize,
    /// Sequence length (after padding)
    pub seq_len: usize,
}

impl BatchEncodedInput {
    /// Convert to tensors [batch_size, seq_len]
    pub fn to_tensors(&self, device: &Device) -> Result<(Tensor, Tensor)> {
        let input_ids = Tensor::new(&self.input_ids[..], device)?
            .to_dtype(DType::U32)?
            .reshape((self.batch_size, self.seq_len))?;

        let attention_mask = Tensor::new(&self.attention_mask[..], device)?
            .to_dtype(DType::F32)?
            .reshape((self.batch_size, self.seq_len))?;

        Ok((input_ids, attention_mask))
    }

    /// Get token type IDs as tensor
    pub fn token_type_ids_tensor(&self, device: &Device) -> Result<Tensor> {
        Ok(Tensor::new(&self.token_type_ids[..], device)?
            .to_dtype(DType::U32)?
            .reshape((self.batch_size, self.seq_len))?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_tokenizer_load() {
        let tokenizer = TokenizerWrapper::from_pretrained("bert-base-uncased");
        assert!(
            tokenizer.is_ok(),
            "Failed to load tokenizer: {:?}",
            tokenizer.err()
        );
    }

    #[test]
    #[ignore]
    fn test_tokenizer_encode() {
        let tokenizer = TokenizerWrapper::from_pretrained("bert-base-uncased").unwrap();
        let encoded = tokenizer.encode("Hello, world!", true).unwrap();

        assert!(!encoded.is_empty());
        assert_eq!(encoded.input_ids.len(), encoded.attention_mask.len());
    }
}
