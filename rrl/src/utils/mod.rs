//! Common utilities
//!
//! Shared utilities for logging, error handling, and helper functions.

use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};

pub mod download;

pub use download::*;

/// Get the default models directory
pub fn get_models_dir() -> Result<PathBuf> {
    let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE"))?;
    let models_dir = Path::new(&home).join(".cache/rrl/models");
    fs::create_dir_all(&models_dir)
        .context(format!("Failed to create models directory: {:?}", models_dir))?;
    Ok(models_dir)
}

/// Get the default cache directory
pub fn get_cache_dir() -> Result<PathBuf> {
    let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE"))?;
    let cache_dir = Path::new(&home).join(".cache/rrl");
    fs::create_dir_all(&cache_dir)
        .context(format!("Failed to create cache directory: {:?}", cache_dir))?;
    Ok(cache_dir)
}
