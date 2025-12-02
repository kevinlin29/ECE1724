//! Optimizers for training
//!
//! Implements AdamW optimizer for fine-tuning BERT-family models.

use anyhow::Result;
use candle_core::Tensor;
use candle_nn::optim::{Optimizer, ParamsAdamW};
use candle_nn::VarMap;

/// AdamW optimizer configuration
#[derive(Debug, Clone)]
pub struct AdamWConfig {
    /// Learning rate
    pub lr: f64,
    /// Beta1 (first moment decay)
    pub beta1: f64,
    /// Beta2 (second moment decay)
    pub beta2: f64,
    /// Epsilon for numerical stability
    pub eps: f64,
    /// Weight decay coefficient
    pub weight_decay: f64,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            lr: 5e-5,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

/// AdamW optimizer wrapper
pub struct AdamW {
    inner: candle_nn::optim::AdamW,
    config: AdamWConfig,
    step_count: usize,
}

impl AdamW {
    /// Create a new AdamW optimizer
    pub fn new(var_map: &VarMap, config: AdamWConfig) -> Result<Self> {
        let params = ParamsAdamW {
            lr: config.lr,
            beta1: config.beta1,
            beta2: config.beta2,
            eps: config.eps,
            weight_decay: config.weight_decay,
        };

        let inner = candle_nn::optim::AdamW::new(var_map.all_vars(), params)?;

        Ok(Self {
            inner,
            config,
            step_count: 0,
        })
    }

    /// Perform an optimization step
    pub fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        self.inner.step(grads)?;
        self.step_count += 1;
        Ok(())
    }

    /// Get current learning rate
    pub fn learning_rate(&self) -> f64 {
        self.config.lr
    }

    /// Set learning rate
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.config.lr = lr;
        self.inner.set_learning_rate(lr);
    }

    /// Get step count
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Zero gradients (no-op for candle, but kept for API compatibility)
    pub fn zero_grad(&mut self) {
        // Candle handles this automatically
    }
}

/// Learning rate scheduler
pub struct LearningRateScheduler {
    base_lr: f64,
    warmup_steps: usize,
    total_steps: usize,
    current_step: usize,
}

impl LearningRateScheduler {
    /// Create a new learning rate scheduler with warmup and cosine decay
    pub fn new(base_lr: f64, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            base_lr,
            warmup_steps,
            total_steps,
            current_step: 0,
        }
    }

    /// Get the learning rate for the current step
    pub fn get_lr(&self) -> f64 {
        if self.current_step < self.warmup_steps {
            // Linear warmup
            self.base_lr * (self.current_step as f64 / self.warmup_steps as f64)
        } else {
            // Cosine decay
            let progress =
                (self.current_step - self.warmup_steps) as f64 / (self.total_steps - self.warmup_steps) as f64;
            let decay = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
            self.base_lr * decay
        }
    }

    /// Step the scheduler and return the new learning rate
    pub fn step(&mut self) -> f64 {
        self.current_step += 1;
        self.get_lr()
    }

    /// Reset the scheduler
    pub fn reset(&mut self) {
        self.current_step = 0;
    }

    /// Get current step
    pub fn current_step(&self) -> usize {
        self.current_step
    }
}

/// Compute gradient norm for a set of tensors
///
/// # Arguments
/// * `grads` - GradStore containing computed gradients
/// * `params` - Parameters to compute gradient norm for
///
/// # Returns
/// * Total L2 norm of all gradients
pub fn compute_grad_norm(
    grads: &candle_core::backprop::GradStore,
    params: &[Tensor],
) -> Result<f64> {
    let mut total_norm_sq: f64 = 0.0;

    for param in params {
        if let Some(grad) = grads.get(param) {
            let grad_norm_sq = grad.sqr()?.sum_all()?.to_scalar::<f32>()?;
            total_norm_sq += grad_norm_sq as f64;
        }
    }

    Ok(total_norm_sq.sqrt())
}

/// Check if gradient clipping is needed and log if so
///
/// Note: Candle's GradStore is immutable, so actual clipping must be done
/// during the backward pass or by scaling the loss. This function only
/// computes the norm and logs a warning if clipping would be needed.
pub fn check_grad_norm(
    grads: &candle_core::backprop::GradStore,
    params: &[Tensor],
    max_norm: f64,
) -> Result<f64> {
    let total_norm = compute_grad_norm(grads, params)?;

    if total_norm > max_norm {
        let clip_coef = max_norm / (total_norm + 1e-6);
        tracing::debug!(
            "Gradient norm {:.4} > max_norm {:.4}, would clip with coef {:.4}",
            total_norm,
            max_norm,
            clip_coef
        );
    }

    Ok(total_norm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lr_scheduler_warmup() {
        let scheduler = LearningRateScheduler::new(1e-4, 100, 1000);
        assert_eq!(scheduler.get_lr(), 0.0);
    }

    #[test]
    fn test_lr_scheduler_decay() {
        let mut scheduler = LearningRateScheduler::new(1e-4, 0, 1000);

        // At step 0, should be full LR
        assert!((scheduler.get_lr() - 1e-4).abs() < 1e-10);

        // At step 500 (halfway), should be about 0.5 * base_lr
        for _ in 0..500 {
            scheduler.step();
        }
        let lr_500 = scheduler.get_lr();
        assert!((lr_500 - 0.5e-4).abs() < 1e-6);

        // At step 1000, should be near 0
        for _ in 0..500 {
            scheduler.step();
        }
        let lr_1000 = scheduler.get_lr();
        assert!(lr_1000 < 1e-8);
    }
}
