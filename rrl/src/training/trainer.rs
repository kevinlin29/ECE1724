//! Training loop for fine-tuning BERT-family models
//!
//! Provides a high-level API for training with:
//! - Gradient accumulation
//! - Learning rate scheduling
//! - Checkpointing
//! - Progress tracking

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarMap;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use super::checkpoint::{enable_checkpointing, disable_checkpointing, CheckpointMemoryStats};
use super::dataset::{Batched, TrainingDataset};
use super::loss::{ContrastiveLoss, ContrastiveLossConfig};
use super::models::{EmbeddingModel, LoraModel, TokenizerWrapper};
use super::optimizer::{AdamW, AdamWConfig, LearningRateScheduler};

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Batch size for training
    pub batch_size: usize,
    /// Number of training epochs
    pub num_epochs: usize,
    /// Base learning rate
    pub learning_rate: f64,
    /// Warmup steps (fraction of total)
    pub warmup_ratio: f64,
    /// Weight decay
    pub weight_decay: f64,
    /// Gradient accumulation steps
    pub gradient_accumulation_steps: usize,
    /// Maximum gradient norm for clipping
    pub max_grad_norm: f64,
    /// Save checkpoint every N steps (0 to disable)
    pub save_steps: usize,
    /// Evaluation steps (0 to disable)
    pub eval_steps: usize,
    /// Logging steps
    pub logging_steps: usize,
    /// Output directory for checkpoints
    pub output_dir: String,
    /// Temperature for contrastive loss
    pub temperature: f32,
    /// Maximum sequence length
    pub max_seq_length: usize,
    /// Enable gradient checkpointing for memory efficiency
    pub gradient_checkpointing: bool,
    /// Number of layers per checkpoint segment (smaller = less memory, more compute)
    pub checkpoint_segment_size: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            num_epochs: 3,
            learning_rate: 5e-5,
            warmup_ratio: 0.1,
            weight_decay: 0.01,
            gradient_accumulation_steps: 1,
            max_grad_norm: 1.0,
            save_steps: 500,
            eval_steps: 500,
            logging_steps: 100,
            output_dir: "./output".to_string(),
            temperature: 0.05,
            max_seq_length: 512,
            gradient_checkpointing: false,
            checkpoint_segment_size: 2,
        }
    }
}

impl TrainingConfig {
    /// Estimate memory savings from gradient checkpointing
    pub fn estimate_checkpoint_memory(
        &self,
        hidden_size: usize,
        num_layers: usize,
    ) -> CheckpointMemoryStats {
        CheckpointMemoryStats::estimate(
            self.batch_size,
            self.max_seq_length,
            hidden_size,
            num_layers,
            self.checkpoint_segment_size,
        )
    }
}

/// Training metrics
#[derive(Debug, Clone, Default)]
pub struct TrainingMetrics {
    /// Total training loss
    pub train_loss: f64,
    /// Number of training steps
    pub global_step: usize,
    /// Current epoch
    pub epoch: usize,
    /// Samples per second
    pub samples_per_second: f64,
    /// Current learning rate
    pub learning_rate: f64,
}

impl std::fmt::Display for TrainingMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Step {} | Epoch {} | Loss: {:.4} | LR: {:.2e} | {:.1} samples/s",
            self.global_step,
            self.epoch,
            self.train_loss,
            self.learning_rate,
            self.samples_per_second
        )
    }
}

/// Training result
#[derive(Debug)]
pub struct TrainingResult {
    /// Final metrics
    pub metrics: TrainingMetrics,
    /// Path to final checkpoint (if saved)
    pub checkpoint_path: Option<String>,
    /// Training history (loss per step)
    pub history: Vec<f64>,
}

/// Trainer for fine-tuning embedding models
pub struct Trainer {
    config: TrainingConfig,
    device: Device,
    var_map: VarMap,
}

impl Trainer {
    /// Create a new trainer
    pub fn new(config: TrainingConfig, device: Device) -> Self {
        Self {
            config,
            device,
            var_map: VarMap::new(),
        }
    }

    /// Get the device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the config
    pub fn config(&self) -> &TrainingConfig {
        &self.config
    }

    /// Get the var_map
    pub fn var_map(&self) -> &VarMap {
        &self.var_map
    }

    /// Get mutable reference to var_map for adding parameters
    pub fn var_map_mut(&mut self) -> &mut VarMap {
        &mut self.var_map
    }

    /// Create optimizer
    pub fn create_optimizer(&self) -> Result<AdamW> {
        let config = AdamWConfig {
            lr: self.config.learning_rate,
            weight_decay: self.config.weight_decay,
            ..Default::default()
        };
        AdamW::new(&self.var_map, config)
    }

    /// Create learning rate scheduler
    pub fn create_scheduler(&self, total_steps: usize) -> LearningRateScheduler {
        let warmup_steps = (total_steps as f64 * self.config.warmup_ratio) as usize;
        LearningRateScheduler::new(self.config.learning_rate, warmup_steps, total_steps)
    }

    /// Save checkpoint
    pub fn save_checkpoint(&self, path: impl AsRef<Path>, step: usize) -> Result<()> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)?;

        let checkpoint_file = path.join(format!("checkpoint-{}.safetensors", step));

        // Save var_map
        self.var_map
            .save(&checkpoint_file)
            .context("Failed to save checkpoint")?;

        tracing::info!("Saved checkpoint to {:?}", checkpoint_file);
        Ok(())
    }

    /// Load checkpoint
    pub fn load_checkpoint(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        self.var_map
            .load(path)
            .context("Failed to load checkpoint")?;
        tracing::info!("Loaded checkpoint from {:?}", path);
        Ok(())
    }

    /// Train the model on the dataset
    ///
    /// Supports both encoder (BERT, RoBERTa) and decoder (Qwen2, LLaMA, Mistral) models.
    #[allow(unused_variables, unused_assignments)]
    pub fn train(
        &mut self,
        model: &dyn LoraModel,
        tokenizer: &TokenizerWrapper,
        dataset: &TrainingDataset,
        eval_dataset: Option<&TrainingDataset>,
        progress_callback: Option<Box<dyn Fn(&TrainingMetrics)>>,
    ) -> Result<TrainingResult> {
        let total_batches = (dataset.len() + self.config.batch_size - 1) / self.config.batch_size;
        let total_steps = total_batches * self.config.num_epochs / self.config.gradient_accumulation_steps;

        tracing::info!("Starting training:");
        tracing::info!("  Dataset size: {}", dataset.len());
        tracing::info!("  Batch size: {}", self.config.batch_size);
        tracing::info!("  Gradient accumulation steps: {}", self.config.gradient_accumulation_steps);
        tracing::info!("  Effective batch size: {}", self.config.batch_size * self.config.gradient_accumulation_steps);
        tracing::info!("  Epochs: {}", self.config.num_epochs);
        tracing::info!("  Total optimization steps: {}", total_steps);
        tracing::info!("  Learning rate: {}", self.config.learning_rate);
        tracing::info!("  Max gradient norm: {}", self.config.max_grad_norm);

        // Enable/disable gradient checkpointing globally
        if self.config.gradient_checkpointing {
            enable_checkpointing();
            tracing::info!("  Gradient checkpointing: ENABLED (segment_size={})", self.config.checkpoint_segment_size);

            // Log memory estimation
            let mem_stats = self.config.estimate_checkpoint_memory(
                model.hidden_size(),
                12, // Typical BERT layers, could be made configurable
            );
            tracing::info!("  Estimated memory savings: {:.1} MB ({:.1}%)",
                mem_stats.memory_saved_mb,
                mem_stats.memory_saved_mb / mem_stats.peak_without_checkpoint_mb * 100.0
            );
        } else {
            disable_checkpointing();
            tracing::info!("  Gradient checkpointing: disabled");
        }

        // Create optimizer and scheduler
        let mut optimizer = self.create_optimizer()?;
        let mut scheduler = self.create_scheduler(total_steps);

        // Create loss function
        let loss_fn = ContrastiveLoss::new(ContrastiveLossConfig {
            temperature: self.config.temperature,
            in_batch_negatives: true,
        });

        let mut metrics = TrainingMetrics::default();
        let mut history = Vec::new();
        let mut accumulated_loss_value = 0.0;
        let mut accumulated_steps = 0;
        let mut accumulated_loss_tensor: Option<Tensor> = None;

        // Create output directory
        std::fs::create_dir_all(&self.config.output_dir)?;

        for epoch in 0..self.config.num_epochs {
            metrics.epoch = epoch + 1;
            let epoch_start = Instant::now();
            let mut epoch_loss = 0.0;
            let mut epoch_samples = 0;

            for (_batch_idx, batch) in dataset.batches(self.config.batch_size).enumerate() {
                let step_start = Instant::now();

                // Convert batch examples to training format
                let (queries, positives): (Vec<_>, Vec<_>) = batch
                    .iter()
                    .filter(|ex| !ex.positives.is_empty())
                    .map(|ex| {
                        // Take first positive for simplicity
                        (ex.query.clone(), ex.positives[0].clone())
                    })
                    .unzip();

                if queries.is_empty() {
                    continue;
                }

                // Compute loss for this batch
                let loss = self.compute_batch_loss(
                    model,
                    tokenizer,
                    &queries,
                    &positives,
                    &loss_fn,
                )?;

                let loss_value = loss.to_scalar::<f32>()? as f64;

                // Scale loss for gradient accumulation (average over accumulation steps)
                let scaled_loss = (&loss / self.config.gradient_accumulation_steps as f64)?;

                // Accumulate the loss tensor for proper gradient computation
                accumulated_loss_tensor = Some(match accumulated_loss_tensor {
                    Some(acc) => (&acc + &scaled_loss)?,
                    None => scaled_loss,
                });

                accumulated_loss_value += loss_value;
                accumulated_steps += 1;
                epoch_loss += loss_value;
                epoch_samples += queries.len();

                // Perform optimization step after accumulating enough gradients
                if accumulated_steps >= self.config.gradient_accumulation_steps {
                    if let Some(ref acc_loss) = accumulated_loss_tensor {
                        // Backward pass on accumulated loss
                        let grads = acc_loss.backward()?;

                        // Optimizer step with gradient clipping
                        let grad_norm = optimizer.step_with_clipping(
                            &grads,
                            self.config.max_grad_norm,
                        )?;

                        if grad_norm > self.config.max_grad_norm {
                            tracing::debug!(
                                "Step {}: Gradient norm {:.4} exceeded max {:.4}, clipping applied",
                                metrics.global_step,
                                grad_norm,
                                self.config.max_grad_norm
                            );
                        }

                        // Update learning rate
                        let new_lr = scheduler.step();
                        optimizer.set_learning_rate(new_lr);

                        metrics.global_step += 1;
                    }

                    // Reset accumulation
                    accumulated_loss_value = 0.0;
                    accumulated_steps = 0;
                    accumulated_loss_tensor = None;
                }

                // Update metrics
                let step_time = step_start.elapsed().as_secs_f64();
                metrics.train_loss = loss_value;
                metrics.learning_rate = scheduler.get_lr();
                metrics.samples_per_second = queries.len() as f64 / step_time;

                history.push(loss_value);

                // Logging
                if metrics.global_step > 0 && metrics.global_step % self.config.logging_steps == 0 {
                    tracing::info!("{}", metrics);
                    // Flush to ensure real-time log streaming to UI
                    let _ = std::io::stderr().flush();
                    if let Some(ref callback) = progress_callback {
                        callback(&metrics);
                    }
                }

                // Checkpointing
                if self.config.save_steps > 0
                    && metrics.global_step > 0
                    && metrics.global_step % self.config.save_steps == 0
                {
                    self.save_checkpoint(&self.config.output_dir, metrics.global_step)?;
                }

                // Evaluation
                if self.config.eval_steps > 0
                    && metrics.global_step > 0
                    && metrics.global_step % self.config.eval_steps == 0
                {
                    if let Some(eval_ds) = eval_dataset {
                        let eval_loss = self.evaluate(model, tokenizer, eval_ds, &loss_fn)?;
                        tracing::info!("Evaluation loss: {:.4}", eval_loss);
                        let _ = std::io::stderr().flush();
                    }
                }
            }

            let epoch_time = epoch_start.elapsed().as_secs_f64();
            let avg_epoch_loss = if epoch_samples > 0 {
                epoch_loss / (epoch_samples as f64 / self.config.batch_size as f64)
            } else {
                0.0
            };

            tracing::info!(
                "Epoch {} completed in {:.1}s | Avg loss: {:.4} | Samples: {}",
                epoch + 1,
                epoch_time,
                avg_epoch_loss,
                epoch_samples
            );
            let _ = std::io::stderr().flush();
        }

        // Save final checkpoint
        let checkpoint_path = if self.config.save_steps > 0 {
            let final_path = Path::new(&self.config.output_dir).join("final");
            self.save_checkpoint(&final_path, metrics.global_step)?;
            Some(final_path.to_string_lossy().to_string())
        } else {
            None
        };

        Ok(TrainingResult {
            metrics,
            checkpoint_path,
            history,
        })
    }

    /// Compute loss for a single batch
    fn compute_batch_loss(
        &self,
        model: &dyn EmbeddingModel,
        tokenizer: &TokenizerWrapper,
        queries: &[String],
        positives: &[String],
        loss_fn: &ContrastiveLoss,
    ) -> Result<Tensor> {
        // Tokenize queries
        let query_batch = tokenizer.encode_batch(queries, true)?;
        let (query_ids, query_mask) = query_batch.to_tensors(model.device())?;

        // Tokenize positives
        let pos_batch = tokenizer.encode_batch(positives, true)?;
        let (pos_ids, pos_mask) = pos_batch.to_tensors(model.device())?;

        // Forward pass - get embeddings
        let query_embeddings = model.forward(&query_ids, &query_mask)?;
        let pos_embeddings = model.forward(&pos_ids, &pos_mask)?;

        // Compute contrastive loss
        loss_fn.forward(&query_embeddings, &pos_embeddings, None)
    }

    /// Evaluate on a dataset
    fn evaluate(
        &self,
        model: &dyn EmbeddingModel,
        tokenizer: &TokenizerWrapper,
        dataset: &TrainingDataset,
        loss_fn: &ContrastiveLoss,
    ) -> Result<f64> {
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        for batch in dataset.batches(self.config.batch_size) {
            let (queries, positives): (Vec<_>, Vec<_>) = batch
                .iter()
                .filter(|ex| !ex.positives.is_empty())
                .map(|ex| (ex.query.clone(), ex.positives[0].clone()))
                .unzip();

            if queries.is_empty() {
                continue;
            }

            let loss = self.compute_batch_loss(model, tokenizer, &queries, &positives, loss_fn)?;
            total_loss += loss.to_scalar::<f32>()? as f64;
            num_batches += 1;
        }

        Ok(if num_batches > 0 {
            total_loss / num_batches as f64
        } else {
            0.0
        })
    }
}

/// Training batch
#[derive(Debug)]
pub struct TrainingBatch {
    /// Anchor/query texts
    pub queries: Vec<String>,
    /// Positive texts (for contrastive learning)
    pub positives: Vec<String>,
    /// Negative texts (optional, for hard negatives)
    pub negatives: Option<Vec<Vec<String>>>,
}

impl TrainingBatch {
    /// Create a new training batch
    pub fn new(queries: Vec<String>, positives: Vec<String>) -> Self {
        Self {
            queries,
            positives,
            negatives: None,
        }
    }

    /// Add negatives
    pub fn with_negatives(mut self, negatives: Vec<Vec<String>>) -> Self {
        self.negatives = Some(negatives);
        self
    }

    /// Batch size
    pub fn len(&self) -> usize {
        self.queries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.queries.is_empty()
    }
}

/// Progress callback for training
pub type ProgressCallback = Box<dyn Fn(&TrainingMetrics) + Send>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.num_epochs, 3);
        assert!((config.learning_rate - 5e-5).abs() < 1e-10);
    }

    #[test]
    fn test_training_batch() {
        let batch = TrainingBatch::new(
            vec!["query1".to_string(), "query2".to_string()],
            vec!["pos1".to_string(), "pos2".to_string()],
        );

        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());
        assert!(batch.negatives.is_none());
    }

    #[test]
    fn test_training_metrics_display() {
        let metrics = TrainingMetrics {
            train_loss: 0.5,
            global_step: 100,
            epoch: 1,
            samples_per_second: 32.5,
            learning_rate: 5e-5,
        };

        let display = format!("{}", metrics);
        assert!(display.contains("Step 100"));
        assert!(display.contains("Epoch 1"));
    }
}
