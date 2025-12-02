//! Example: Fine-tuning BERT with LoRA on Recipe-MPR dataset
//!
//! This example demonstrates the workflow for fine-tuning:
//! 1. Load the Recipe-MPR dataset
//! 2. Create a BERT model with LoRA adapters
//! 3. Set up the training loop with RAG-aware losses
//! 4. Train the model and save checkpoints
//!
//! Usage:
//!   cargo run --example finetune_recipe_mpr --features training --release

use anyhow::Result;

#[cfg(feature = "training")]
use rrl::training::{
    device::{select_device, DevicePreference},
    lora::{LoraConfig, LoraStats},
    loss::RagLoss,
    models::load_model,
    trainer::TrainingConfig,
};

fn main() -> Result<()> {
    println!("Fine-tuning BERT with LoRA on Recipe-MPR Dataset\n");

    #[cfg(feature = "training")]
    {
        // 1. Setup device
        let device = select_device(DevicePreference::Auto)?;
        println!("Using device: {:?}", device);

        // 2. Configure LoRA
        println!("\nLoRA Configuration:");
        let lora_config = LoraConfig::new(8, 32.0)
            .with_dropout(0.1)
            .with_target_modules(vec!["query".to_string(), "value".to_string()]);

        println!("  - Rank: {}", lora_config.rank);
        println!("  - Alpha: {}", lora_config.alpha);
        println!("  - Scaling: {}", lora_config.scaling());
        println!("  - Target modules: {:?}", lora_config.target_modules);

        // 3. Calculate parameter statistics
        // For BERT-base: 768 hidden size, targeting query and value projections
        // Each LoRA layer: rank * hidden_size * 2 = 8 * 768 * 2 = 12,288 params per layer
        // 12 layers * 2 modules (query, value) = 24 LoRA modules
        // Total LoRA params: 24 * 12,288 = ~295K params vs 110M total

        let stats = LoraStats::new(110_000_000, 295_000, 24);
        println!("\nParameter Statistics:");
        println!("  {}", stats);

        // 4. Setup loss function
        println!("\nLoss Configuration:");
        let loss = RagLoss::new(
            1.0,  // contrastive weight
            0.1,  // grounding weight
            0.05, // temperature
        );
        println!("  - Contrastive weight: {}", loss.contrastive_weight);
        println!("  - Grounding weight: {}", loss.grounding_weight);

        // 5. Training configuration
        let training_config = TrainingConfig {
            batch_size: 32,
            num_epochs: 3,
            learning_rate: 5e-5,
            warmup_ratio: 0.1,
            weight_decay: 0.01,
            gradient_accumulation_steps: 4,
            max_grad_norm: 1.0,
            save_steps: 500,
            eval_steps: 500,
            logging_steps: 100,
            output_dir: "./checkpoints/recipe-mpr".to_string(),
            temperature: 0.05,
            max_seq_length: 512,
        };

        println!("\nTraining Configuration:");
        println!("  - Batch size: {}", training_config.batch_size);
        println!("  - Learning rate: {}", training_config.learning_rate);
        println!("  - Epochs: {}", training_config.num_epochs);
        println!("  - Gradient accumulation: {}", training_config.gradient_accumulation_steps);
        println!("  - Output directory: {}", training_config.output_dir);

        // 6. Training loop overview
        println!("\nTraining Workflow:");
        println!("  1. For each epoch:");
        println!("     a. Iterate over batches");
        println!("     b. Tokenize query-document pairs");
        println!("     c. Forward pass through BERT + LoRA");
        println!("     d. Compute contrastive loss");
        println!("     e. Backward pass (gradient computation)");
        println!("     f. Accumulate gradients");
        println!("     g. Optimizer step (every {} batches)", training_config.gradient_accumulation_steps);
        println!("     h. Log metrics every {} steps", training_config.logging_steps);
        println!("     i. Save checkpoint every {} steps", training_config.save_steps);
        println!();

        // 7. Load model (if available)
        println!("Loading model...");
        match load_model("bert-base-uncased", &device) {
            Ok(model) => {
                println!("  Model loaded: hidden_size={}, arch={}",
                    model.hidden_size(),
                    model.architecture()
                );
            }
            Err(e) => {
                println!("  Note: Model not loaded ({})", e);
                println!("  This is expected if model hasn't been downloaded yet.");
            }
        }

        println!("\n=== Example Complete ===");
        println!("To run actual training:");
        println!("  1. Download the Recipe-MPR dataset");
        println!("  2. Implement data loading for your dataset format");
        println!("  3. Use the Trainer with the configuration above");
    }

    #[cfg(not(feature = "training"))]
    {
        println!("This example requires the 'training' feature.");
        println!("Run with: cargo run --example finetune_recipe_mpr --features training");
    }

    Ok(())
}
