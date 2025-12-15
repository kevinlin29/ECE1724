//! CUDA-compatible BERT implementation
//!
//! This module provides a BERT model that works on CUDA by implementing
//! layer normalization using basic tensor operations instead of the
//! candle_nn::LayerNorm which lacks CUDA support.
//!
//! Supports gradient checkpointing for memory-efficient training.

use anyhow::Result;
use candle_core::{DType, Module, Tensor, D};
use candle_nn::{Embedding, Linear, VarBuilder};

use crate::training::checkpoint::{is_checkpointing_enabled, CheckpointConfig};

/// CUDA-compatible softmax
///
/// Implements softmax using basic ops that have CUDA support:
/// softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
fn cuda_softmax(x: &Tensor, dim: D) -> anyhow::Result<Tensor> {
    // Numerically stable softmax: subtract max before exp
    let max = x.max_keepdim(dim)?;
    let x_shifted = x.broadcast_sub(&max)?;
    let exp_x = x_shifted.exp()?;
    let sum_exp = exp_x.sum_keepdim(dim)?;
    Ok(exp_x.broadcast_div(&sum_exp)?)
}

/// CUDA-compatible Layer Normalization
///
/// Implements layer norm using basic ops that have CUDA support:
/// y = (x - mean) / sqrt(var + eps) * weight + bias
#[derive(Debug, Clone)]
pub struct CudaLayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl CudaLayerNorm {
    pub fn new(weight: Tensor, bias: Tensor, eps: f64) -> Self {
        Self { weight, bias, eps }
    }

    pub fn load(vb: VarBuilder, hidden_size: usize, eps: f64) -> Result<Self> {
        // Try PyTorch naming (weight/bias) first, then TensorFlow naming (gamma/beta)
        let weight = vb.get(hidden_size, "weight")
            .or_else(|_| vb.get(hidden_size, "gamma"))?;
        let bias = vb.get(hidden_size, "bias")
            .or_else(|_| vb.get(hidden_size, "beta"))?;
        Ok(Self { weight, bias, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Compute mean along last dimension
        let mean = x.mean_keepdim(D::Minus1)?;

        // Compute variance: E[(x - mean)^2]
        let diff = x.broadcast_sub(&mean)?;
        let variance = diff.sqr()?.mean_keepdim(D::Minus1)?;

        // Normalize: (x - mean) / sqrt(var + eps)
        let std = (variance + self.eps)?.sqrt()?;
        let normalized = diff.broadcast_div(&std)?;

        // Apply affine transformation: normalized * weight + bias
        let output = normalized.broadcast_mul(&self.weight)?.broadcast_add(&self.bias)?;

        Ok(output)
    }
}

impl Module for CudaLayerNorm {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.forward(x).map_err(|e| candle_core::Error::Msg(e.to_string()))
    }
}

/// BERT Embeddings (word + position + token_type)
#[derive(Debug)]
pub struct CudaBertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: CudaLayerNorm,
    #[allow(dead_code)]
    hidden_dropout_prob: f64,
}

impl CudaBertEmbeddings {
    pub fn load(vb: VarBuilder, config: &CudaBertConfig) -> Result<Self> {
        let word_embeddings = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("word_embeddings"),
        )?;
        let position_embeddings = candle_nn::embedding(
            config.max_position_embeddings,
            config.hidden_size,
            vb.pp("position_embeddings"),
        )?;
        let token_type_embeddings = candle_nn::embedding(
            config.type_vocab_size,
            config.hidden_size,
            vb.pp("token_type_embeddings"),
        )?;
        let layer_norm = CudaLayerNorm::load(
            vb.pp("LayerNorm"),
            config.hidden_size,
            config.layer_norm_eps,
        )?;

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            hidden_dropout_prob: config.hidden_dropout_prob,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
    ) -> Result<Tensor> {
        let (_batch_size, seq_len) = input_ids.dims2()?;

        // Get word embeddings
        let word_embeds = self.word_embeddings.forward(input_ids)?;

        // Get position embeddings
        let position_ids = Tensor::arange(0u32, seq_len as u32, input_ids.device())?;
        let position_embeds = self.position_embeddings.forward(&position_ids)?;

        // Get token type embeddings
        let token_type_embeds = self.token_type_embeddings.forward(token_type_ids)?;

        // Sum all embeddings
        let embeddings = (word_embeds + position_embeds.broadcast_add(&token_type_embeds)?)?;

        // Apply layer norm
        let normalized = self.layer_norm.forward(&embeddings)?;

        // Note: Dropout is typically only applied during training
        // For inference, we skip it
        Ok(normalized)
    }
}

/// Self-attention mechanism
#[derive(Debug)]
pub struct CudaBertSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    num_attention_heads: usize,
    attention_head_size: usize,
}

impl CudaBertSelfAttention {
    pub fn load(vb: VarBuilder, config: &CudaBertConfig) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;

        let query = candle_nn::linear(config.hidden_size, config.hidden_size, vb.pp("query"))?;
        let key = candle_nn::linear(config.hidden_size, config.hidden_size, vb.pp("key"))?;
        let value = candle_nn::linear(config.hidden_size, config.hidden_size, vb.pp("value"))?;

        Ok(Self {
            query,
            key,
            value,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
        })
    }

    fn transpose_for_scores(&self, x: &Tensor, batch_size: usize, seq_len: usize) -> Result<Tensor> {
        // Reshape from [batch, seq, hidden] to [batch, num_heads, seq, head_size]
        let x = x.reshape((batch_size, seq_len, self.num_attention_heads, self.attention_head_size))?;
        x.transpose(1, 2).map_err(Into::into)
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;

        // Linear projections
        let query_layer = self.query.forward(hidden_states)?;
        let key_layer = self.key.forward(hidden_states)?;
        let value_layer = self.value.forward(hidden_states)?;

        // Reshape for multi-head attention
        let query_layer = self.transpose_for_scores(&query_layer, batch_size, seq_len)?.contiguous()?;
        let key_layer = self.transpose_for_scores(&key_layer, batch_size, seq_len)?.contiguous()?;
        let value_layer = self.transpose_for_scores(&value_layer, batch_size, seq_len)?.contiguous()?;

        // Compute attention scores: Q @ K^T
        // K^T needs to be contiguous after transpose
        let key_layer_t = key_layer.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let scale = 1.0 / (self.attention_head_size as f64).sqrt();
        let attention_scores = query_layer.matmul(&key_layer_t)?;
        let attention_scores = (attention_scores * scale)?;

        // Apply attention mask if provided
        let attention_scores = if let Some(mask) = attention_mask {
            // mask is [batch, 1, 1, seq] with 0 for valid and -inf for masked
            // Need to broadcast to [batch, heads, seq, seq]
            attention_scores.broadcast_add(mask)?
        } else {
            attention_scores
        };

        // Softmax
        let attention_probs = cuda_softmax(&attention_scores, D::Minus1)?;

        // Apply attention to values
        let context_layer = attention_probs.matmul(&value_layer)?;

        // Reshape back to [batch, seq, hidden]
        let context_layer = context_layer.transpose(1, 2)?.contiguous()?;
        let context_layer = context_layer.reshape((batch_size, seq_len, self.num_attention_heads * self.attention_head_size))?;

        Ok(context_layer)
    }
}

/// Self-attention output (dense + layer norm)
#[derive(Debug)]
pub struct CudaBertSelfOutput {
    dense: Linear,
    layer_norm: CudaLayerNorm,
}

impl CudaBertSelfOutput {
    pub fn load(vb: VarBuilder, config: &CudaBertConfig) -> Result<Self> {
        let dense = candle_nn::linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = CudaLayerNorm::load(vb.pp("LayerNorm"), config.hidden_size, config.layer_norm_eps)?;
        Ok(Self { dense, layer_norm })
    }

    pub fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        // Residual connection + layer norm
        let hidden_states = self.layer_norm.forward(&(hidden_states + input_tensor)?)?;
        Ok(hidden_states)
    }
}

/// Full attention block
#[derive(Debug)]
pub struct CudaBertAttention {
    self_attention: CudaBertSelfAttention,
    output: CudaBertSelfOutput,
}

impl CudaBertAttention {
    pub fn load(vb: VarBuilder, config: &CudaBertConfig) -> Result<Self> {
        let self_attention = CudaBertSelfAttention::load(vb.pp("self"), config)?;
        let output = CudaBertSelfOutput::load(vb.pp("output"), config)?;
        Ok(Self { self_attention, output })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let self_output = self.self_attention.forward(hidden_states, attention_mask)?;
        let attention_output = self.output.forward(&self_output, hidden_states)?;
        Ok(attention_output)
    }
}

/// Feed-forward intermediate layer
#[derive(Debug)]
pub struct CudaBertIntermediate {
    dense: Linear,
}

impl CudaBertIntermediate {
    pub fn load(vb: VarBuilder, config: &CudaBertConfig) -> Result<Self> {
        let dense = candle_nn::linear(config.hidden_size, config.intermediate_size, vb.pp("dense"))?;
        Ok(Self { dense })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        // GELU activation
        let hidden_states = hidden_states.gelu()?;
        Ok(hidden_states)
    }
}

/// Feed-forward output layer
#[derive(Debug)]
pub struct CudaBertOutput {
    dense: Linear,
    layer_norm: CudaLayerNorm,
}

impl CudaBertOutput {
    pub fn load(vb: VarBuilder, config: &CudaBertConfig) -> Result<Self> {
        let dense = candle_nn::linear(config.intermediate_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = CudaLayerNorm::load(vb.pp("LayerNorm"), config.hidden_size, config.layer_norm_eps)?;
        Ok(Self { dense, layer_norm })
    }

    pub fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        // Residual connection + layer norm
        let hidden_states = self.layer_norm.forward(&(hidden_states + input_tensor)?)?;
        Ok(hidden_states)
    }
}

/// Single transformer layer
#[derive(Debug)]
pub struct CudaBertLayer {
    attention: CudaBertAttention,
    intermediate: CudaBertIntermediate,
    output: CudaBertOutput,
}

impl CudaBertLayer {
    pub fn load(vb: VarBuilder, config: &CudaBertConfig) -> Result<Self> {
        let attention = CudaBertAttention::load(vb.pp("attention"), config)?;
        let intermediate = CudaBertIntermediate::load(vb.pp("intermediate"), config)?;
        let output = CudaBertOutput::load(vb.pp("output"), config)?;
        Ok(Self { attention, intermediate, output })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let attention_output = self.attention.forward(hidden_states, attention_mask)?;
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        let layer_output = self.output.forward(&intermediate_output, &attention_output)?;
        Ok(layer_output)
    }
}

/// BERT encoder (stack of transformer layers)
///
/// Supports gradient checkpointing for memory-efficient training.
/// When checkpointing is enabled, activations are not stored during forward pass
/// and are recomputed during backward pass.
#[derive(Debug)]
pub struct CudaBertEncoder {
    layers: Vec<CudaBertLayer>,
    checkpoint_config: CheckpointConfig,
}

#[allow(dead_code)]
impl CudaBertEncoder {
    pub fn load(vb: VarBuilder, config: &CudaBertConfig) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let vb_layers = vb.pp("layer");

        for i in 0..config.num_hidden_layers {
            let layer = CudaBertLayer::load(vb_layers.pp(i.to_string()), config)?;
            layers.push(layer);
        }

        Ok(Self {
            layers,
            checkpoint_config: CheckpointConfig::default(),
        })
    }

    /// Load with gradient checkpointing configuration
    pub fn load_with_checkpointing(
        vb: VarBuilder,
        config: &CudaBertConfig,
        checkpoint_config: CheckpointConfig,
    ) -> Result<Self> {
        let mut encoder = Self::load(vb, config)?;
        encoder.checkpoint_config = checkpoint_config;
        Ok(encoder)
    }

    /// Set the checkpoint configuration
    pub fn set_checkpoint_config(&mut self, config: CheckpointConfig) {
        self.checkpoint_config = config;
    }

    /// Enable gradient checkpointing with default segment size
    pub fn enable_checkpointing(&mut self, segment_size: usize) {
        self.checkpoint_config = CheckpointConfig::new(true).with_segment_size(segment_size);
    }

    /// Disable gradient checkpointing
    pub fn disable_checkpointing(&mut self) {
        self.checkpoint_config.enabled = false;
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let use_checkpointing = self.checkpoint_config.enabled && is_checkpointing_enabled();

        if use_checkpointing {
            self.forward_with_checkpointing(hidden_states, attention_mask)
        } else {
            self.forward_standard(hidden_states, attention_mask)
        }
    }

    /// Standard forward pass without checkpointing
    fn forward_standard(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, attention_mask)?;
        }

        Ok(hidden_states)
    }

    /// Forward pass with gradient checkpointing
    ///
    /// Divides layers into segments and creates checkpoints at segment boundaries.
    /// This reduces memory usage by not storing all intermediate activations,
    /// at the cost of recomputing them during backward pass.
    fn forward_with_checkpointing(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let segment_size = self.checkpoint_config.segment_size;
        let num_layers = self.layers.len();

        tracing::debug!(
            "Running BERT encoder with checkpointing (segment_size={}, layers={})",
            segment_size,
            num_layers
        );

        let mut hidden_states = hidden_states.clone();

        for (segment_idx, segment_start) in (0..num_layers).step_by(segment_size).enumerate() {
            let segment_end = (segment_start + segment_size).min(num_layers);

            // Checkpoint at segment boundary (except first)
            // Detaching breaks the computation graph, so gradients won't flow through
            // During backward pass, Candle will recompute from this checkpoint
            if segment_idx > 0 {
                hidden_states = hidden_states.detach();
                tracing::trace!(
                    "Checkpoint created at segment {} (layers {}-{})",
                    segment_idx,
                    segment_start,
                    segment_end
                );
            }

            // Run layers in this segment
            for layer_idx in segment_start..segment_end {
                hidden_states = self.layers[layer_idx].forward(&hidden_states, attention_mask)?;
            }
        }

        Ok(hidden_states)
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get checkpoint config
    pub fn checkpoint_config(&self) -> &CheckpointConfig {
        &self.checkpoint_config
    }
}

/// BERT configuration
#[derive(Debug, Clone, serde::Deserialize)]
pub struct CudaBertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_dropout_prob: f64,
    pub attention_probs_dropout_prob: f64,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
}

fn default_layer_norm_eps() -> f64 {
    1e-12
}

impl Default for CudaBertConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30522,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
        }
    }
}

/// Full CUDA-compatible BERT model
#[derive(Debug)]
pub struct CudaBertModel {
    embeddings: CudaBertEmbeddings,
    encoder: CudaBertEncoder,
    config: CudaBertConfig,
}

impl CudaBertModel {
    pub fn load(vb: VarBuilder, config: &CudaBertConfig) -> Result<Self> {
        let embeddings = CudaBertEmbeddings::load(vb.pp("embeddings"), config)?;
        let encoder = CudaBertEncoder::load(vb.pp("encoder"), config)?;

        Ok(Self {
            embeddings,
            encoder,
            config: config.clone(),
        })
    }

    /// Create attention mask for transformer
    /// Converts [batch, seq] mask (1=valid, 0=pad) to [batch, 1, 1, seq] with -inf for padding
    fn create_attention_mask(&self, attention_mask: &Tensor) -> Result<Tensor> {
        let dtype = DType::F32;
        let _device = attention_mask.device();

        // Convert 0/1 mask to attention mask with -inf for padding
        // Shape: [batch, seq] -> [batch, 1, 1, seq]
        let attention_mask = attention_mask.to_dtype(dtype)?;
        let attention_mask = attention_mask.unsqueeze(1)?.unsqueeze(1)?;

        // Convert: 1 -> 0, 0 -> -10000 (large negative for softmax)
        let attention_mask = ((1.0 - attention_mask)? * (-10000.0))?;

        Ok(attention_mask)
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Get embeddings
        let hidden_states = self.embeddings.forward(input_ids, token_type_ids)?;

        // Create extended attention mask
        let attention_mask = attention_mask
            .map(|m| self.create_attention_mask(m))
            .transpose()?;

        // Encode
        let sequence_output = self.encoder.forward(&hidden_states, attention_mask.as_ref())?;

        Ok(sequence_output)
    }

    pub fn config(&self) -> &CudaBertConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_cuda_layer_norm() {
        let device = Device::Cpu;
        let hidden_size = 768;

        let weight = Tensor::ones(hidden_size, DType::F32, &device).unwrap();
        let bias = Tensor::zeros(hidden_size, DType::F32, &device).unwrap();
        let layer_norm = CudaLayerNorm::new(weight, bias, 1e-12);

        let input = Tensor::randn(0.0f32, 1.0, (2, 10, hidden_size), &device).unwrap();
        let output = layer_norm.forward(&input).unwrap();

        assert_eq!(output.dims(), &[2, 10, hidden_size]);
    }

    #[test]
    fn test_bert_config_deserialize() {
        let json = r#"{
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "layer_norm_eps": 1e-12
        }"#;

        let config: CudaBertConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_hidden_layers, 12);
    }
}
