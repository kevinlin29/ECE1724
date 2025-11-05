//! Model downloading utilities
//!
//! Download and cache ONNX models from Hugging Face or local paths.

use anyhow::{Context, Result};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Download a file from a URL to a local path
pub fn download_file(url: &str, output_path: &Path) -> Result<()> {
    tracing::info!("Downloading from {} to {:?}", url, output_path);

    // Create parent directory if it doesn't exist
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let response = reqwest::blocking::get(url)
        .context(format!("Failed to download from {}", url))?;

    if !response.status().is_success() {
        anyhow::bail!("Failed to download: HTTP {}", response.status());
    }

    let mut file = File::create(output_path)
        .context(format!("Failed to create file: {:?}", output_path))?;

    let content = response.bytes()?;
    file.write_all(&content)?;

    tracing::info!("Download complete: {:?}", output_path);
    Ok(())
}

/// Download a Hugging Face model file
pub fn download_hf_model(
    repo_id: &str,
    filename: &str,
    cache_dir: &Path,
) -> Result<PathBuf> {
    let model_dir = cache_dir.join(repo_id.replace('/', "_"));
    fs::create_dir_all(&model_dir)?;

    let output_path = model_dir.join(filename);

    // Check if already downloaded
    if output_path.exists() {
        tracing::info!("Model already cached at {:?}", output_path);
        return Ok(output_path);
    }

    // Construct Hugging Face URL
    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        repo_id, filename
    );

    download_file(&url, &output_path)?;
    Ok(output_path)
}

/// Model manifest for tracking downloaded files
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct ModelManifest {
    pub repo_id: String,
    pub files: Vec<String>,
    pub downloaded_at: String,
}

impl ModelManifest {
    /// Create a new manifest
    pub fn new(repo_id: String, files: Vec<String>) -> Self {
        let downloaded_at = chrono::Utc::now().to_rfc3339();
        Self {
            repo_id,
            files,
            downloaded_at,
        }
    }

    /// Save manifest to disk
    pub fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Load manifest from disk
    pub fn load(path: &Path) -> Result<Self> {
        let json = fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_roundtrip() {
        let manifest = ModelManifest::new(
            "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            vec!["model.onnx".to_string(), "tokenizer.json".to_string()],
        );

        let temp_dir = tempfile::tempdir().unwrap();
        let manifest_path = temp_dir.path().join("manifest.json");

        manifest.save(&manifest_path).unwrap();
        let loaded = ModelManifest::load(&manifest_path).unwrap();

        assert_eq!(manifest.repo_id, loaded.repo_id);
        assert_eq!(manifest.files, loaded.files);
    }
}
