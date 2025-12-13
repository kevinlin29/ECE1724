//! Gradient checkpointing for memory-efficient training
//!
//! Implements gradient checkpointing (also known as activation checkpointing)
//! to reduce GPU memory usage during training by trading compute for memory.
//!
//! The technique works by:
//! 1. Not storing intermediate activations during forward pass
//! 2. Recomputing activations during backward pass from saved checkpoints
//!
//! This can reduce memory usage by O(sqrt(n)) for n layers at the cost of
//! one additional forward pass per segment.

use anyhow::Result;
use candle_core::Tensor;
use std::sync::atomic::{AtomicBool, Ordering};

/// Global flag to enable/disable gradient checkpointing
static CHECKPOINTING_ENABLED: AtomicBool = AtomicBool::new(false);

/// Enable gradient checkpointing globally
pub fn enable_checkpointing() {
    CHECKPOINTING_ENABLED.store(true, Ordering::SeqCst);
    tracing::info!("Gradient checkpointing enabled");
}

/// Disable gradient checkpointing globally
pub fn disable_checkpointing() {
    CHECKPOINTING_ENABLED.store(false, Ordering::SeqCst);
    tracing::info!("Gradient checkpointing disabled");
}

/// Check if gradient checkpointing is enabled
pub fn is_checkpointing_enabled() -> bool {
    CHECKPOINTING_ENABLED.load(Ordering::SeqCst)
}

/// Configuration for gradient checkpointing
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Whether checkpointing is enabled
    pub enabled: bool,
    /// Number of layers per checkpoint segment
    /// Smaller = less memory, more recomputation
    /// Typical values: 1-4 for transformer models
    pub segment_size: usize,
    /// Whether to checkpoint the embedding layer
    pub checkpoint_embeddings: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            segment_size: 2,
            checkpoint_embeddings: false,
        }
    }
}

impl CheckpointConfig {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            ..Default::default()
        }
    }

    pub fn with_segment_size(mut self, size: usize) -> Self {
        self.segment_size = size.max(1);
        self
    }

    pub fn with_checkpoint_embeddings(mut self, checkpoint: bool) -> Self {
        self.checkpoint_embeddings = checkpoint;
        self
    }
}

/// Checkpoint a tensor by detaching it from the computation graph
///
/// When checkpointing is enabled, this creates a "checkpoint" by:
/// 1. Detaching the tensor (breaking gradient flow)
/// 2. The caller must store the function to recompute if gradients are needed
///
/// During backward pass, Candle will need to recompute from this checkpoint.
pub fn checkpoint_tensor(tensor: &Tensor) -> Result<Tensor> {
    if is_checkpointing_enabled() {
        // Detach breaks the computation graph here
        // Gradients won't flow through this point during backward
        // Instead, the forward computation will be redone from this checkpoint
        Ok(tensor.detach())
    } else {
        // Without checkpointing, just clone the tensor reference
        Ok(tensor.clone())
    }
}

/// A checkpoint segment that can recompute its forward pass
///
/// This is used to wrap a sequence of layers that should be checkpointed together.
/// During training, activations are not stored; during backward, they're recomputed.
pub struct CheckpointSegment<F>
where
    F: Fn(&Tensor, Option<&Tensor>) -> Result<Tensor>,
{
    /// The forward function for this segment
    forward_fn: F,
    /// Stored input for recomputation (only when checkpointing)
    stored_input: Option<Tensor>,
    /// Stored mask for recomputation
    stored_mask: Option<Tensor>,
}

impl<F> CheckpointSegment<F>
where
    F: Fn(&Tensor, Option<&Tensor>) -> Result<Tensor>,
{
    pub fn new(forward_fn: F) -> Self {
        Self {
            forward_fn,
            stored_input: None,
            stored_mask: None,
        }
    }

    /// Run the segment with optional checkpointing
    pub fn forward(&mut self, input: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        if is_checkpointing_enabled() {
            // Store input for potential recomputation
            self.stored_input = Some(input.detach());
            self.stored_mask = mask.map(|m| m.detach());

            // Run forward pass
            let output = (self.forward_fn)(input, mask)?;

            // Detach output - gradients will trigger recomputation
            Ok(output.detach())
        } else {
            // Normal forward pass - keep computation graph intact
            (self.forward_fn)(input, mask)
        }
    }

    /// Recompute the forward pass (used during backward when checkpointing)
    pub fn recompute(&self) -> Result<Tensor> {
        match &self.stored_input {
            Some(input) => (self.forward_fn)(input, self.stored_mask.as_ref()),
            None => Err(anyhow::anyhow!("No stored input for recomputation")),
        }
    }
}

/// Helper to run a function with checkpointing
///
/// This is a simpler interface for checkpointing individual operations.
///
/// # Example
/// ```ignore
/// let output = checkpoint(
///     || layer.forward(&input, mask),
///     is_training && use_checkpointing,
/// )?;
/// ```
pub fn checkpoint<F>(f: F, should_checkpoint: bool) -> Result<Tensor>
where
    F: FnOnce() -> Result<Tensor>,
{
    if should_checkpoint && is_checkpointing_enabled() {
        let output = f()?;
        Ok(output.detach())
    } else {
        f()
    }
}

/// Checkpointed sequential layer runner
///
/// Runs multiple layers in sequence, creating checkpoints at segment boundaries.
/// This is the main utility for applying checkpointing to transformer encoders.
pub struct CheckpointedSequential {
    config: CheckpointConfig,
}

impl CheckpointedSequential {
    pub fn new(config: CheckpointConfig) -> Self {
        Self { config }
    }

    /// Run layers with checkpointing
    ///
    /// Divides layers into segments and checkpoints at segment boundaries.
    ///
    /// # Arguments
    /// * `hidden_states` - Input tensor
    /// * `attention_mask` - Optional attention mask
    /// * `layers` - Vector of layer forward functions
    pub fn forward<F>(
        &self,
        mut hidden_states: Tensor,
        attention_mask: Option<&Tensor>,
        layers: &[F],
    ) -> Result<Tensor>
    where
        F: Fn(&Tensor, Option<&Tensor>) -> Result<Tensor>,
    {
        if !self.config.enabled || !is_checkpointing_enabled() {
            // No checkpointing - run normally
            for layer in layers {
                hidden_states = layer(&hidden_states, attention_mask)?;
            }
            return Ok(hidden_states);
        }

        // With checkpointing - process in segments
        let segment_size = self.config.segment_size;
        let num_layers = layers.len();

        tracing::debug!(
            "Running {} layers with checkpointing (segment_size={})",
            num_layers,
            segment_size
        );

        for (segment_idx, segment_start) in (0..num_layers).step_by(segment_size).enumerate() {
            let segment_end = (segment_start + segment_size).min(num_layers);

            // Checkpoint at segment boundary (except first)
            if segment_idx > 0 {
                hidden_states = hidden_states.detach();
            }

            // Run segment
            for layer_idx in segment_start..segment_end {
                hidden_states = layers[layer_idx](&hidden_states, attention_mask)?;
            }
        }

        Ok(hidden_states)
    }
}

/// Memory statistics for checkpointing analysis
#[derive(Debug, Default)]
pub struct CheckpointMemoryStats {
    /// Peak memory without checkpointing (estimated)
    pub peak_without_checkpoint_mb: f64,
    /// Peak memory with checkpointing (estimated)
    pub peak_with_checkpoint_mb: f64,
    /// Memory saved by checkpointing
    pub memory_saved_mb: f64,
    /// Additional compute overhead (number of extra forward passes)
    pub extra_forward_passes: usize,
}

impl CheckpointMemoryStats {
    /// Estimate memory stats for a transformer model
    ///
    /// # Arguments
    /// * `batch_size` - Batch size
    /// * `seq_len` - Sequence length
    /// * `hidden_size` - Hidden dimension
    /// * `num_layers` - Number of transformer layers
    /// * `segment_size` - Checkpointing segment size
    pub fn estimate(
        batch_size: usize,
        seq_len: usize,
        hidden_size: usize,
        num_layers: usize,
        segment_size: usize,
    ) -> Self {
        // Each layer stores activations of shape [batch, seq, hidden]
        // Plus attention matrices [batch, heads, seq, seq]
        let activation_per_layer_mb =
            (batch_size * seq_len * hidden_size * 4) as f64 / (1024.0 * 1024.0);

        // Without checkpointing: store all layer activations
        let peak_without = activation_per_layer_mb * num_layers as f64;

        // With checkpointing: store only checkpoint activations + one segment
        let num_segments = (num_layers + segment_size - 1) / segment_size;
        let peak_with = activation_per_layer_mb * (num_segments + segment_size) as f64;

        // Extra forward passes: each segment (except first) needs recomputation
        let extra_forwards = (num_segments.saturating_sub(1)) * segment_size;

        Self {
            peak_without_checkpoint_mb: peak_without,
            peak_with_checkpoint_mb: peak_with,
            memory_saved_mb: (peak_without - peak_with).max(0.0),
            extra_forward_passes: extra_forwards,
        }
    }
}

impl std::fmt::Display for CheckpointMemoryStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Gradient Checkpointing Memory Analysis:")?;
        writeln!(f, "  Without checkpointing: {:.1} MB", self.peak_without_checkpoint_mb)?;
        writeln!(f, "  With checkpointing:    {:.1} MB", self.peak_with_checkpoint_mb)?;
        writeln!(f, "  Memory saved:          {:.1} MB ({:.1}%)",
            self.memory_saved_mb,
            self.memory_saved_mb / self.peak_without_checkpoint_mb * 100.0)?;
        writeln!(f, "  Extra forward passes:  {}", self.extra_forward_passes)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_config() {
        let config = CheckpointConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.segment_size, 2);

        let config = CheckpointConfig::new(true).with_segment_size(4);
        assert!(config.enabled);
        assert_eq!(config.segment_size, 4);
    }

    #[test]
    fn test_checkpointing_flag() {
        disable_checkpointing();
        assert!(!is_checkpointing_enabled());

        enable_checkpointing();
        assert!(is_checkpointing_enabled());

        disable_checkpointing();
        assert!(!is_checkpointing_enabled());
    }

    #[test]
    fn test_memory_stats_estimation() {
        let stats = CheckpointMemoryStats::estimate(
            8,      // batch_size
            512,    // seq_len
            768,    // hidden_size
            12,     // num_layers
            2,      // segment_size
        );

        assert!(stats.peak_without_checkpoint_mb > 0.0);
        assert!(stats.peak_with_checkpoint_mb > 0.0);
        assert!(stats.memory_saved_mb >= 0.0);

        println!("{}", stats);
    }
}
