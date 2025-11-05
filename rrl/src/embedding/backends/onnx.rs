//! ONNX Runtime backend for embeddings
//!
//! Provides hardware-accelerated inference using ONNX Runtime with support for
//! CUDA and Metal execution providers.

#[cfg(feature = "onnx-backend")]
use crate::embedding::{
    apply_pooling, normalize_embedding, Embedder, Embedding, EmbeddingConfig,
};
#[cfg(feature = "onnx-backend")]
use anyhow::Result;
#[cfg(feature = "onnx-backend")]
use ndarray::Array2;
#[cfg(feature = "onnx-backend")]
use once_cell::sync::Lazy;
#[cfg(feature = "onnx-backend")]
use onnxruntime::{
    environment::Environment, session::Session, GraphOptimizationLevel, LoggingLevel,
};
#[cfg(feature = "onnx-backend")]
use std::path::PathBuf;
#[cfg(feature = "onnx-backend")]
use std::sync::Arc;
#[cfg(feature = "onnx-backend")]
use tiktoken_rs::{cl100k_base, CoreBPE};

#[cfg(feature = "onnx-backend")]
/// Global ONNX environment (lazy initialized)
static ONNX_ENVIRONMENT: Lazy<Environment> = Lazy::new(|| {
    Environment::builder()
        .with_name("rrl")
        .with_log_level(LoggingLevel::Warning)
        .build()
        .expect("Failed to create ONNX environment")
});

#[cfg(feature = "onnx-backend")]
/// Hardware backend for ONNX execution
#[derive(Debug, Clone, Copy)]
pub enum HardwareBackend {
    /// CPU execution
    Cpu,
    /// NVIDIA CUDA GPU
    Cuda,
    /// Apple Metal GPU
    Metal,
}

#[cfg(feature = "onnx-backend")]
impl HardwareBackend {
    /// Get backend from string
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "cuda" => Self::Cuda,
            "metal" => Self::Metal,
            _ => Self::Cpu,
        }
    }
}

#[cfg(feature = "onnx-backend")]
/// ONNX-based embedder
pub struct OnnxEmbedder {
    session: Session<'static>,
    tokenizer: Arc<CoreBPE>,
    config: EmbeddingConfig,
    dimension: usize,
}

#[cfg(feature = "onnx-backend")]
impl OnnxEmbedder {
    /// Create a new ONNX embedder from model path
    pub fn new(
        model_path: &std::path::Path,
        config: EmbeddingConfig,
        hardware: HardwareBackend,
    ) -> Result<Self> {
        tracing::info!("Loading ONNX model from {:?}", model_path);
        tracing::info!("Hardware backend: {:?}", hardware);

        // Verify model file exists
        if !model_path.exists() {
            anyhow::bail!("Model file not found: {:?}", model_path);
        }

        // Initialize tokenizer
        let tokenizer = Arc::new(cl100k_base()?);

        // Use default dimension
        let dimension = 384;

        // Create session using global environment
        let mut session_builder = ONNX_ENVIRONMENT
            .new_session_builder()?
            .with_optimization_level(GraphOptimizationLevel::All)?
            .with_number_threads(4)?;

        // Add execution provider based on hardware
        session_builder = match hardware {
            #[cfg(feature = "cuda")]
            HardwareBackend::Cuda => {
                tracing::info!("Enabling CUDA execution provider");
                match session_builder.with_cuda(0) {
                    Ok(sb) => {
                        tracing::info!("CUDA enabled successfully");
                        sb
                    }
                    Err(e) => {
                        tracing::warn!("CUDA not available: {}. Falling back to CPU", e);
                        session_builder
                    }
                }
            }
            #[cfg(feature = "metal")]
            HardwareBackend::Metal => {
                tracing::info!("Enabling CoreML execution provider for Metal");
                match session_builder.with_coreml() {
                    Ok(sb) => {
                        tracing::info!("CoreML/Metal enabled successfully");
                        sb
                    }
                    Err(e) => {
                        tracing::warn!("CoreML not available: {}. Falling back to CPU", e);
                        session_builder
                    }
                }
            }
            HardwareBackend::Cpu => {
                tracing::info!("Using CPU execution provider");
                session_builder
            }
            #[allow(unreachable_patterns)]
            _ => session_builder,
        };

        let session = session_builder.with_model_from_file(model_path)?;

        tracing::info!("ONNX model loaded successfully (dimension: {})", dimension);

        Ok(Self {
            session,
            tokenizer,
            config,
            dimension,
        })
    }

    /// Tokenize text into input IDs
    fn tokenize(&self, text: &str) -> Result<Vec<i64>> {
        let tokens = self.tokenizer.encode_with_special_tokens(text);

        // Truncate to max length
        let mut token_ids: Vec<i64> = tokens
            .iter()
            .take(self.config.max_length)
            .map(|&t| t as i64)
            .collect();

        // Pad if necessary
        while token_ids.len() < self.config.max_length {
            token_ids.push(0); // PAD token
        }

        Ok(token_ids)
    }

    /// Create attention mask (1 for real tokens, 0 for padding)
    fn create_attention_mask(&self, token_ids: &[i64]) -> Vec<i64> {
        token_ids
            .iter()
            .map(|&id| if id == 0 { 0 } else { 1 })
            .collect()
    }

    /// Run inference and extract embeddings
    fn inference(&self, input_ids: &[i64], attention_mask: &[i64]) -> Result<Embedding> {
        // Prepare input tensors
        let batch_size = 1;
        let seq_length = input_ids.len();

        let input_ids_array = Array2::from_shape_vec(
            (batch_size, seq_length),
            input_ids.to_vec(),
        )?;

        let attention_mask_array = Array2::from_shape_vec(
            (batch_size, seq_length),
            attention_mask.to_vec(),
        )?;

        // Run inference
        let outputs = self.session.run(vec![
            input_ids_array.into_dyn().into(),
            attention_mask_array.into_dyn().into(),
        ])?;

        // Extract embeddings from output
        // Output shape is typically [batch_size, sequence_length, hidden_size]
        let output_tensor = outputs[0].extract_tensor::<f32>()?.view().to_owned();

        // Convert to Vec<Vec<f32>> for pooling
        let shape = output_tensor.shape();
        if shape.len() != 3 {
            anyhow::bail!("Unexpected output shape: {:?}", shape);
        }

        let seq_len = shape[1];
        let hidden_size = shape[2];

        let mut token_embeddings = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let start_idx = i * hidden_size;
            let end_idx = start_idx + hidden_size;
            let slice: Vec<f32> = output_tensor
                .as_slice()
                .unwrap()[start_idx..end_idx]
                .to_vec();
            token_embeddings.push(slice);
        }

        // Apply pooling
        let mut embedding = apply_pooling(&token_embeddings, self.config.pooling)?;

        // Normalize if configured
        if self.config.normalize {
            normalize_embedding(&mut embedding);
        }

        Ok(embedding)
    }
}

#[cfg(feature = "onnx-backend")]
impl Embedder for OnnxEmbedder {
    fn embed(&self, text: &str) -> Result<Embedding> {
        let input_ids = self.tokenize(text)?;
        let attention_mask = self.create_attention_mask(&input_ids);
        self.inference(&input_ids, &attention_mask)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>> {
        // For simplicity, process one by one
        // TODO: Implement true batching for better performance
        texts.iter().map(|&text| self.embed(text)).collect()
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        &self.config.model_name
    }
}

#[cfg(test)]
#[cfg(feature = "onnx-backend")]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_backend_parsing() {
        assert!(matches!(HardwareBackend::from_str("cpu"), HardwareBackend::Cpu));
        assert!(matches!(HardwareBackend::from_str("cuda"), HardwareBackend::Cuda));
        assert!(matches!(HardwareBackend::from_str("metal"), HardwareBackend::Metal));
        assert!(matches!(HardwareBackend::from_str("unknown"), HardwareBackend::Cpu));
    }
}
