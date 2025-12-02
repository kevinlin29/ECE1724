//! Comprehensive Fine-Tuning Example for RRL
//!
//! This example demonstrates the fine-tuning workflow:
//! 1. Loading and preparing a dataset for RAG-aware fine-tuning
//! 2. Setting up LoRA adapters for efficient training
//! 3. Configuring contrastive loss
//! 4. Training with the trainer API
//!
//! Usage:
//!   cargo run --example fine_tuning --features training --release

use anyhow::Result;

#[cfg(feature = "training")]
use rrl::training::{
    device::{select_device, DevicePreference},
    lora::LoraConfig,
    loss::ContrastiveLossConfig,
    trainer::TrainingConfig,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing for logging
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    println!("=== RRL Fine-Tuning Example ===\n");

    #[cfg(feature = "training")]
    {
        // Step 1: Configure device
        println!("Step 1: Setting up device...");
        let device = select_device(DevicePreference::Auto)?;
        println!("  Using device: {:?}\n", device);

        // Step 2: Configure LoRA for efficient fine-tuning
        println!("Step 2: Configuring LoRA adapters...");
        let lora_config = configure_lora();
        println!("  - LoRA rank: {}", lora_config.rank);
        println!("  - LoRA alpha: {}", lora_config.alpha);
        println!("  - Scaling: {}", lora_config.scaling());
        println!("  - Dropout: {}\n", lora_config.dropout);

        // Step 3: Set up training configuration
        println!("Step 3: Configuring training...");
        let training_config = configure_training();
        println!("  - Batch size: {}", training_config.batch_size);
        println!("  - Learning rate: {}", training_config.learning_rate);
        println!("  - Num epochs: {}", training_config.num_epochs);
        println!("  - Warmup ratio: {}", training_config.warmup_ratio);
        println!("  - Weight decay: {}\n", training_config.weight_decay);

        // Step 4: Configure loss function
        println!("Step 4: Configuring contrastive loss...");
        let loss_config = ContrastiveLossConfig::default();
        println!("  - Temperature: {}", loss_config.temperature);
        println!("  - In-batch negatives: {}\n", loss_config.in_batch_negatives);

        // Step 5: Training overview
        println!("Step 5: Training workflow overview");
        println!("  The training loop will:");
        println!("  1. Load batches of query-document pairs");
        println!("  2. Tokenize inputs with the model's tokenizer");
        println!("  3. Forward pass through BERT model");
        println!("  4. Compute contrastive loss (InfoNCE)");
        println!("  5. Backward pass and optimizer step");
        println!("  6. Update learning rate scheduler");
        println!("  7. Save checkpoints periodically\n");

        println!("=== Configuration Complete ===");
        println!("To run actual training, implement the data loading");
        println!("and training loop using the rrl::training module.\n");
    }

    #[cfg(not(feature = "training"))]
    {
        println!("This example requires the 'training' feature.");
        println!("Run with: cargo run --example fine_tuning --features training");
    }

    Ok(())
}

#[cfg(feature = "training")]
fn configure_lora() -> LoraConfig {
    LoraConfig::new(8, 16.0)
        .with_dropout(0.1)
        .with_target_modules(vec!["query".to_string(), "value".to_string()])
}

#[cfg(feature = "training")]
fn configure_training() -> TrainingConfig {
    TrainingConfig {
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
    }
}
