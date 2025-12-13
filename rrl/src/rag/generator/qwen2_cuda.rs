//! CUDA-compatible Qwen2 model implementation
//!
//! This module provides a Qwen2 model that works on CUDA by implementing
//! RMS normalization using basic tensor operations instead of the
//! candle operations which lack CUDA support.

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{Embedding, Linear, VarBuilder};
use std::sync::Arc;

/// CUDA-compatible softmax
///
/// Implements softmax using basic ops that have CUDA support:
/// softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
pub fn cuda_softmax(x: &Tensor, dim: D) -> Result<Tensor> {
    // Numerically stable softmax: subtract max before exp
    let max = x.max_keepdim(dim)?;
    let x_shifted = x.broadcast_sub(&max)?;
    let exp_x = x_shifted.exp()?;
    let sum_exp = exp_x.sum_keepdim(dim)?;
    Ok(exp_x.broadcast_div(&sum_exp)?)
}

/// CUDA-compatible RMS Normalization
///
/// Implements RMS norm using basic ops that have CUDA support:
/// y = x / sqrt(mean(x^2) + eps) * weight
///
/// Unlike LayerNorm, RMS norm doesn't subtract mean or add bias.
#[derive(Debug, Clone)]
pub struct CudaRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl CudaRmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    pub fn load(vb: VarBuilder, size: usize, eps: f64) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // RMS = sqrt(mean(x^2))
        let x_squared = x.sqr()?;
        let mean_squared = x_squared.mean_keepdim(D::Minus1)?;

        // Normalize: x / sqrt(mean(x^2) + eps)
        let rms = (mean_squared + self.eps)?.sqrt()?;
        let normalized = x.broadcast_div(&rms)?;

        // Scale by weight
        Ok(normalized.broadcast_mul(&self.weight)?)
    }
}

impl Module for CudaRmsNorm {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.forward(x).map_err(|e| candle_core::Error::Msg(e.to_string()))
    }
}

/// Rotary Position Embedding
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    pub fn new(dtype: DType, config: &CudaQwen2Config, device: &Device) -> Result<Self> {
        let dim = config.hidden_size / config.num_attention_heads;
        let max_seq_len = config.max_position_embeddings;
        let theta = config.rope_theta;

        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / theta.powf(i as f32 / dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?.to_dtype(dtype)?;

        let t: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let t = Tensor::from_vec(t, (max_seq_len, 1), device)?.to_dtype(dtype)?;

        let freqs = t.matmul(&inv_freq)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        Ok(Self { cos, sin })
    }

    pub fn apply_rotary_emb(&self, q: &Tensor, k: &Tensor, start_pos: usize) -> Result<(Tensor, Tensor)> {
        let (_b, _h, seq_len, _d) = q.dims4()?;
        let cos = self.cos.narrow(0, start_pos, seq_len)?;
        let sin = self.sin.narrow(0, start_pos, seq_len)?;

        let q_embed = self.apply_rotary_emb_single(q, &cos, &sin)?;
        let k_embed = self.apply_rotary_emb_single(k, &cos, &sin)?;

        Ok((q_embed, k_embed))
    }

    fn apply_rotary_emb_single(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b, h, seq_len, d) = x.dims4()?;
        let x = x.reshape((b, h, seq_len, d / 2, 2))?;
        let x0 = x.narrow(D::Minus1, 0, 1)?;
        let x1 = x.narrow(D::Minus1, 1, 1)?;

        let cos = cos.unsqueeze(0)?.unsqueeze(0)?.unsqueeze(D::Minus1)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?.unsqueeze(D::Minus1)?;

        let rotated_x0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
        let rotated_x1 = (x1.broadcast_mul(&cos)? + x0.broadcast_mul(&sin)?)?;

        let rotated = Tensor::cat(&[rotated_x0, rotated_x1], D::Minus1)?;
        Ok(rotated.reshape((b, h, seq_len, d))?)
    }
}

/// Qwen2 MLP (Feed-Forward Network)
#[derive(Debug)]
pub struct CudaQwen2Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl CudaQwen2Mlp {
    pub fn load(vb: VarBuilder, config: &CudaQwen2Config) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;

        let gate_proj = candle_nn::linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = candle_nn::linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = candle_nn::linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // SwiGLU activation: down(silu(gate(x)) * up(x))
        let gate = self.gate_proj.forward(x)?;
        let gate = candle_nn::ops::silu(&gate)?;
        let up = self.up_proj.forward(x)?;
        let hidden = (gate * up)?;
        self.down_proj.forward(&hidden).map_err(Into::into)
    }
}

/// Qwen2 Self Attention
#[derive(Debug)]
pub struct CudaQwen2Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl CudaQwen2Attention {
    pub fn load(vb: VarBuilder, config: &CudaQwen2Config, rotary_emb: Arc<RotaryEmbedding>) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = hidden_size / num_heads;

        let q_proj = candle_nn::linear_no_bias(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = candle_nn::linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_emb,
            kv_cache: None,
        })
    }

    pub fn forward(&mut self, x: &Tensor, start_pos: usize, mask: Option<&Tensor>) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape for multi-head attention
        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;

        // Apply rotary embeddings
        let (q, k) = self.rotary_emb.apply_rotary_emb(&q, &k, start_pos)?;

        // KV cache handling
        let (k, v) = match &self.kv_cache {
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, &k], 2)?;
                let v = Tensor::cat(&[prev_v, &v], 2)?;
                (k, v)
            }
            None => (k, v),
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        // Repeat KV for GQA (Grouped Query Attention)
        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        // Attention: softmax(Q @ K^T / sqrt(d)) @ V
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.contiguous()?.matmul(&k.t()?.contiguous()?)? * scale)?;

        // Apply causal mask
        let attn_weights = match mask {
            Some(m) => attn_weights.broadcast_add(m)?,
            None => attn_weights,
        };

        let attn_weights = cuda_softmax(&attn_weights, D::Minus1)?;
        let attn_output = attn_weights.matmul(&v.contiguous()?)?;

        // Reshape back
        let attn_output = attn_output.transpose(1, 2)?.reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

        self.o_proj.forward(&attn_output).map_err(Into::into)
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        let n_rep = self.num_heads / self.num_kv_heads;
        if n_rep == 1 {
            return Ok(x);
        }
        let (b, num_kv_heads, seq_len, head_dim) = x.dims4()?;
        let x = x.unsqueeze(2)?;
        let x = x.expand((b, num_kv_heads, n_rep, seq_len, head_dim))?;
        Ok(x.reshape((b, num_kv_heads * n_rep, seq_len, head_dim))?)
    }

    pub fn clear_cache(&mut self) {
        self.kv_cache = None;
    }
}

/// Qwen2 Decoder Layer
#[derive(Debug)]
pub struct CudaQwen2DecoderLayer {
    self_attn: CudaQwen2Attention,
    mlp: CudaQwen2Mlp,
    input_layernorm: CudaRmsNorm,
    post_attention_layernorm: CudaRmsNorm,
}

impl CudaQwen2DecoderLayer {
    pub fn load(vb: VarBuilder, config: &CudaQwen2Config, rotary_emb: Arc<RotaryEmbedding>) -> Result<Self> {
        let self_attn = CudaQwen2Attention::load(vb.pp("self_attn"), config, rotary_emb)?;
        let mlp = CudaQwen2Mlp::load(vb.pp("mlp"), config)?;
        let input_layernorm = CudaRmsNorm::load(vb.pp("input_layernorm"), config.hidden_size, config.rms_norm_eps)?;
        let post_attention_layernorm = CudaRmsNorm::load(vb.pp("post_attention_layernorm"), config.hidden_size, config.rms_norm_eps)?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(&mut self, x: &Tensor, start_pos: usize, mask: Option<&Tensor>) -> Result<Tensor> {
        // Pre-norm architecture
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward(&x, start_pos, mask)?;
        let x = (residual + x)?;

        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        (residual + x).map_err(Into::into)
    }

    pub fn clear_cache(&mut self) {
        self.self_attn.clear_cache();
    }
}

/// Qwen2 Configuration
#[derive(Debug, Clone, serde::Deserialize)]
pub struct CudaQwen2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

fn default_max_position_embeddings() -> usize { 32768 }
fn default_rms_norm_eps() -> f64 { 1e-6 }
fn default_rope_theta() -> f32 { 10000.0 }

/// CUDA-compatible Qwen2 Model for Causal LM
#[derive(Debug)]
pub struct CudaQwen2Model {
    embed_tokens: Embedding,
    layers: Vec<CudaQwen2DecoderLayer>,
    norm: CudaRmsNorm,
    lm_head: Linear,
    config: CudaQwen2Config,
    device: Device,
    dtype: DType,
}

impl CudaQwen2Model {
    pub fn load(vb: VarBuilder, config: &CudaQwen2Config) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();

        let embed_tokens = candle_nn::embedding(config.vocab_size, config.hidden_size, vb.pp("model.embed_tokens"))?;

        let rotary_emb = Arc::new(RotaryEmbedding::new(dtype, config, &device)?);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let vb_layers = vb.pp("model.layers");
        for i in 0..config.num_hidden_layers {
            let layer = CudaQwen2DecoderLayer::load(vb_layers.pp(i.to_string()), config, rotary_emb.clone())?;
            layers.push(layer);
        }

        let norm = CudaRmsNorm::load(vb.pp("model.norm"), config.hidden_size, config.rms_norm_eps)?;

        // lm_head may be tied to embed_tokens
        let lm_head = if config.tie_word_embeddings {
            // Create a linear layer from embedding weights
            let weight = embed_tokens.embeddings().clone();
            Linear::new(weight, None)
        } else {
            candle_nn::linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            config: config.clone(),
            device,
            dtype,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor> {
        let (_batch_size, seq_len) = input_ids.dims2()?;

        // Get embeddings
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        // Create causal mask
        let mask = if seq_len > 1 {
            let mask = Self::make_causal_mask(seq_len, start_pos, &self.device, self.dtype)?;
            Some(mask)
        } else {
            None
        };

        // Pass through decoder layers
        for layer in &mut self.layers {
            hidden_states = layer.forward(&hidden_states, start_pos, mask.as_ref())?;
        }

        // Final norm
        hidden_states = self.norm.forward(&hidden_states)?;

        // LM head
        let logits = self.lm_head.forward(&hidden_states)?;

        Ok(logits)
    }

    fn make_causal_mask(seq_len: usize, start_pos: usize, device: &Device, dtype: DType) -> Result<Tensor> {
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..(seq_len + start_pos)).map(move |j| {
                    if j > i + start_pos {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect();

        let mask = Tensor::from_vec(mask, (seq_len, seq_len + start_pos), device)?;
        Ok(mask.to_dtype(dtype)?.unsqueeze(0)?.unsqueeze(0)?)
    }

    pub fn clear_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_cache();
        }
    }

    pub fn config(&self) -> &CudaQwen2Config {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_rms_norm() {
        let device = Device::Cpu;
        let hidden_size = 64;

        let weight = Tensor::ones(hidden_size, DType::F32, &device).unwrap();
        let rms_norm = CudaRmsNorm::new(weight, 1e-6);

        let input = Tensor::randn(0.0f32, 1.0, (2, 10, hidden_size), &device).unwrap();
        let output = rms_norm.forward(&input).unwrap();

        assert_eq!(output.dims(), &[2, 10, hidden_size]);
    }

    #[test]
    fn test_qwen2_config_deserialize() {
        let json = r#"{
            "vocab_size": 151936,
            "hidden_size": 896,
            "intermediate_size": 4864,
            "num_hidden_layers": 24,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": true
        }"#;

        let config: CudaQwen2Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.hidden_size, 896);
        assert_eq!(config.num_hidden_layers, 24);
    }
}
