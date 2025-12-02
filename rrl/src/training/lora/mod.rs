//! LoRA/QLoRA fine-tuning
//!
//! Parameter-efficient fine-tuning using Candle.
//!
//! # LoRA (Low-Rank Adaptation)
//!
//! LoRA reduces the number of trainable parameters by decomposing weight updates
//! into low-rank matrices:
//!
//! ```text
//! W' = W + BA * (alpha / rank)
//! ```
//!
//! Where:
//! - W is the frozen pretrained weight
//! - B ∈ ℝ^(out × rank) initialized to zeros
//! - A ∈ ℝ^(rank × in) initialized with Kaiming uniform
//! - alpha is a scaling factor
//!
//! This typically reduces trainable parameters by 100-1000x.

#[cfg(feature = "training")]
use anyhow::Result;
#[cfg(feature = "training")]
use candle_core::Tensor;
#[cfg(feature = "training")]
use candle_nn::{Init, Linear, Module, VarBuilder};

/// LoRA configuration
#[derive(Debug, Clone)]
pub struct LoraConfig {
    /// Rank of the low-rank decomposition (typically 4-64)
    pub rank: usize,
    /// Scaling factor (typically rank * 2)
    pub alpha: f32,
    /// Dropout probability for LoRA layers (0.0-0.1)
    pub dropout: f32,
    /// Target modules to apply LoRA to (e.g., ["query", "value"])
    pub target_modules: Vec<String>,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            dropout: 0.0,
            target_modules: vec!["query".to_string(), "value".to_string()],
        }
    }
}

impl LoraConfig {
    /// Create a new LoRA config
    pub fn new(rank: usize, alpha: f32) -> Self {
        Self {
            rank,
            alpha,
            ..Default::default()
        }
    }

    /// Set dropout
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    /// Set target modules
    pub fn with_target_modules(mut self, modules: Vec<String>) -> Self {
        self.target_modules = modules;
        self
    }

    /// Get the scaling factor
    pub fn scaling(&self) -> f32 {
        self.alpha / self.rank as f32
    }
}

/// LoRA linear layer
///
/// Wraps a frozen base linear layer with trainable low-rank adapters.
#[cfg(feature = "training")]
pub struct LoraLinear {
    /// Base (frozen) linear layer
    base: Linear,
    /// Down projection: input_dim -> rank
    lora_a: Tensor,
    /// Up projection: rank -> output_dim
    lora_b: Tensor,
    /// Scaling factor (alpha / rank)
    scaling: f32,
    /// Whether LoRA weights are merged into base
    merged: bool,
    /// Input dimension
    in_features: usize,
    /// Output dimension
    out_features: usize,
    /// Rank
    rank: usize,
}

#[cfg(feature = "training")]
impl LoraLinear {
    /// Create a new LoRA linear layer
    pub fn new(
        in_features: usize,
        out_features: usize,
        config: &LoraConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Create base linear layer (frozen)
        let base = candle_nn::linear(in_features, out_features, vb.pp("base"))?;

        // Create LoRA matrices
        // A: rank x in_features (down projection)
        // B: out_features x rank (up projection)

        // A initialized with Kaiming uniform
        let lora_a = vb.get_with_hints(
            (config.rank, in_features),
            "lora_a",
            Init::Kaiming {
                dist: candle_nn::init::NormalOrUniform::Uniform,
                fan: candle_nn::init::FanInOut::FanIn,
                non_linearity: candle_nn::init::NonLinearity::Linear,
            },
        )?;

        // B initialized to zeros
        let lora_b = vb.get_with_hints(
            (out_features, config.rank),
            "lora_b",
            Init::Const(0.0),
        )?;

        Ok(Self {
            base,
            lora_a,
            lora_b,
            scaling: config.scaling(),
            merged: false,
            in_features,
            out_features,
            rank: config.rank,
        })
    }

    /// Forward pass
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Base output
        let base_out = self.base.forward(x)?;

        if self.merged {
            return Ok(base_out);
        }

        // LoRA output: x @ A^T @ B^T * scaling
        // Shape: [batch, seq, in] @ [in, rank] @ [rank, out] = [batch, seq, out]
        let lora_out = x
            .matmul(&self.lora_a.t()?)?
            .matmul(&self.lora_b.t()?)?;

        // Scale and add
        let scaled = (lora_out * self.scaling as f64)?;
        Ok((base_out + scaled)?)
    }

    /// Merge LoRA weights into base for efficient inference
    pub fn merge(&mut self) -> Result<()> {
        if self.merged {
            return Ok(());
        }

        // Compute merged weight: W' = W + B @ A * scaling
        // Note: This requires mutable access to base weights
        // For now, we just set the flag
        tracing::warn!("LoRA merge is not fully implemented - using flag only");
        self.merged = true;
        Ok(())
    }

    /// Unmerge LoRA weights (restore original base)
    pub fn unmerge(&mut self) -> Result<()> {
        if !self.merged {
            return Ok(());
        }

        tracing::warn!("LoRA unmerge is not fully implemented - using flag only");
        self.merged = false;
        Ok(())
    }

    /// Check if merged
    pub fn is_merged(&self) -> bool {
        self.merged
    }

    /// Get number of trainable parameters
    pub fn num_trainable_params(&self) -> usize {
        // Only A and B are trainable
        self.rank * self.in_features + self.out_features * self.rank
    }

    /// Get number of total parameters (including frozen)
    pub fn num_total_params(&self) -> usize {
        // Base: in * out + out (bias)
        // LoRA: rank * in + out * rank
        self.in_features * self.out_features + self.out_features + self.num_trainable_params()
    }

    /// Get parameter efficiency (trainable / total)
    pub fn param_efficiency(&self) -> f64 {
        self.num_trainable_params() as f64 / self.num_total_params() as f64 * 100.0
    }
}

/// Statistics about LoRA parameters
#[derive(Debug, Clone)]
pub struct LoraStats {
    /// Total parameters in the model
    pub total_params: usize,
    /// Trainable parameters (LoRA only)
    pub trainable_params: usize,
    /// Percentage of trainable parameters
    pub trainable_percent: f64,
    /// Number of LoRA layers
    pub num_lora_layers: usize,
}

impl LoraStats {
    /// Create stats from counts
    pub fn new(total: usize, trainable: usize, num_layers: usize) -> Self {
        Self {
            total_params: total,
            trainable_params: trainable,
            trainable_percent: trainable as f64 / total as f64 * 100.0,
            num_lora_layers: num_layers,
        }
    }
}

impl std::fmt::Display for LoraStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LoRA Stats: {}/{} params ({:.2}%) across {} layers",
            self.trainable_params, self.total_params, self.trainable_percent, self.num_lora_layers
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_config_default() {
        let config = LoraConfig::default();
        assert_eq!(config.rank, 8);
        assert_eq!(config.alpha, 16.0);
        assert_eq!(config.scaling(), 2.0);
    }

    #[test]
    fn test_lora_config_custom() {
        let config = LoraConfig::new(16, 32.0)
            .with_dropout(0.1)
            .with_target_modules(vec!["query".to_string(), "key".to_string(), "value".to_string()]);

        assert_eq!(config.rank, 16);
        assert_eq!(config.alpha, 32.0);
        assert_eq!(config.scaling(), 2.0);
        assert_eq!(config.dropout, 0.1);
        assert_eq!(config.target_modules.len(), 3);
    }

    #[test]
    fn test_lora_stats() {
        let stats = LoraStats::new(110_000_000, 295_000, 24);
        assert!((stats.trainable_percent - 0.268).abs() < 0.01);
    }
}
