//! Example: Loading BERT models from HuggingFace Hub
//!
//! This example demonstrates how to:
//! 1. Load models directly from HuggingFace Hub (automatic download)
//! 2. Load models from local paths
//! 3. Use different BERT variants (base, large, cased, uncased)
//! 4. Add LoRA adapters to pretrained models
//!
//! Usage:
//!   cargo run --example load_from_huggingface --features training --release

#[cfg(feature = "training")]
use rrl::training::{
    device::{select_device, DevicePreference},
    hub::ModelLoader,
    lora::LoraConfig,
    models::load_model,
};

#[cfg(feature = "training")]
fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("HuggingFace Hub Integration Examples\n");
    println!("====================================\n");

    // Setup device
    let device = select_device(DevicePreference::Auto)?;
    println!("Using device: {:?}\n", device);

    // Example 1: Load BERT from HuggingFace Hub
    println!("Example 1: Loading BERT-base-uncased from HuggingFace");
    println!("-------------------------------------------------");

    match load_model("bert-base-uncased", &device) {
        Ok(model) => {
            println!("Model loaded successfully!");
            println!("  Hidden size: {}", model.hidden_size());
            println!("  Architecture: {}", model.architecture());
            println!();
        }
        Err(e) => {
            println!("Failed to load model: {}", e);
            println!("  (This is expected if you don't have internet or the model hasn't been downloaded yet.)");
            println!();
        }
    }

    // Example 2: List model path components
    println!("Example 2: Using ModelLoader directly");
    println!("-------------------------------------------------");

    let loader = ModelLoader::new()?;
    match loader.load_model_path("bert-base-uncased") {
        Ok(path) => {
            println!("Model files:");
            println!("  Config: {:?}", path.config_file);
            println!("  Weights: {:?}", path.weights_file);
            println!("  Tokenizer: {:?}", path.tokenizer_file);
            println!();
        }
        Err(e) => {
            println!("Failed to resolve model path: {}", e);
            println!();
        }
    }

    // Example 3: LoRA configuration
    println!("Example 3: LoRA Configuration");
    println!("-------------------------------------------------");

    let lora_config = LoraConfig::new(8, 16.0)
        .with_dropout(0.1)
        .with_target_modules(vec!["query".to_string(), "value".to_string()]);

    println!("LoRA Config:");
    println!("  Rank: {}", lora_config.rank);
    println!("  Alpha: {}", lora_config.alpha);
    println!("  Scaling: {}", lora_config.scaling());
    println!("  Dropout: {}", lora_config.dropout);
    println!("  Target modules: {:?}", lora_config.target_modules);
    println!();

    // Example 4: Supported models
    println!("Example 4: Supported BERT models from HuggingFace");
    println!("-------------------------------------------------");
    println!("The following models are supported:");
    println!();

    let supported_models = vec![
        ("bert-base-uncased", "12 layers, 110M params"),
        ("bert-base-cased", "12 layers, 110M params"),
        ("bert-large-uncased", "24 layers, 340M params"),
        ("bert-large-cased", "24 layers, 340M params"),
        ("sentence-transformers/all-MiniLM-L6-v2", "Optimized for embeddings"),
        ("distilbert-base-uncased", "DistilBERT - 6 layers"),
        ("roberta-base", "RoBERTa base model"),
    ];

    for (model_id, description) in supported_models {
        println!("  - {} - {}", model_id, description);
    }

    println!();
    println!("Usage Tips:");
    println!("-------------------------------------------------");
    println!("1. Models are cached in ~/.cache/huggingface/hub");
    println!("2. First download may take a while, subsequent loads are instant");
    println!("3. Compatible with any BERT-architecture model on HuggingFace");
    println!();

    Ok(())
}

#[cfg(not(feature = "training"))]
fn main() {
    println!("This example requires the 'training' feature.");
    println!("Run with: cargo run --example load_from_huggingface --features training");
}
