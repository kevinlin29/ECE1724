//! Dataset loading and batching for training
//!
//! Supports loading query-document pairs from JSONL and CSV formats.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// A single training example with query and documents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    /// Query text
    pub query: String,
    /// Positive documents (relevant to the query)
    pub positives: Vec<String>,
    /// Negative documents (not relevant to the query)
    #[serde(default)]
    pub negatives: Vec<String>,
    /// Optional metadata
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

/// Dataset configuration
#[derive(Debug, Clone)]
pub struct DatasetConfig {
    /// Maximum number of positives per query
    pub max_positives: usize,
    /// Maximum number of negatives per query
    pub max_negatives: usize,
    /// Whether to shuffle the dataset
    pub shuffle: bool,
    /// Random seed for shuffling
    pub seed: Option<u64>,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            max_positives: 5,
            max_negatives: 5,
            shuffle: true,
            seed: Some(42),
        }
    }
}

/// Training dataset
#[derive(Debug)]
pub struct TrainingDataset {
    examples: Vec<TrainingExample>,
    config: DatasetConfig,
}

impl TrainingDataset {
    /// Create a new dataset from examples
    pub fn new(examples: Vec<TrainingExample>, config: DatasetConfig) -> Self {
        Self { examples, config }
    }

    /// Load dataset from a JSONL file
    ///
    /// Expected format (one JSON object per line):
    /// ```json
    /// {"query": "...", "positives": ["...", "..."], "negatives": ["...", "..."]}
    /// ```
    pub fn from_jsonl(path: impl AsRef<Path>, config: DatasetConfig) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)
            .with_context(|| format!("Failed to open dataset file: {:?}", path))?;
        let reader = BufReader::new(file);

        let mut examples = Vec::new();
        for (line_num, line) in reader.lines().enumerate() {
            let line = line.with_context(|| format!("Failed to read line {}", line_num + 1))?;

            // Skip empty lines
            if line.trim().is_empty() {
                continue;
            }

            let mut example: TrainingExample = serde_json::from_str(&line)
                .with_context(|| format!("Failed to parse JSON at line {}", line_num + 1))?;

            // Truncate to max counts
            example.positives.truncate(config.max_positives);
            example.negatives.truncate(config.max_negatives);

            examples.push(example);
        }

        tracing::info!("Loaded {} training examples from {:?}", examples.len(), path);

        let mut dataset = Self { examples, config };

        if dataset.config.shuffle {
            dataset.shuffle();
        }

        Ok(dataset)
    }

    /// Load dataset from a CSV file
    ///
    /// Expected columns: query, positive, negative (optional)
    /// Multiple positives/negatives can be separated by |
    pub fn from_csv(path: impl AsRef<Path>, config: DatasetConfig) -> Result<Self> {
        let path = path.as_ref();
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .flexible(true)
            .from_path(path)
            .with_context(|| format!("Failed to open CSV file: {:?}", path))?;

        let headers = reader.headers()?.clone();
        let query_idx = headers.iter().position(|h| h == "query")
            .ok_or_else(|| anyhow::anyhow!("CSV must have 'query' column"))?;
        let positive_idx = headers.iter().position(|h| h == "positive" || h == "positives");
        let negative_idx = headers.iter().position(|h| h == "negative" || h == "negatives");

        let mut examples = Vec::new();
        for (row_num, result) in reader.records().enumerate() {
            let record = result.with_context(|| format!("Failed to read CSV row {}", row_num + 1))?;

            let query = record.get(query_idx)
                .ok_or_else(|| anyhow::anyhow!("Missing query at row {}", row_num + 1))?
                .to_string();

            let positives = if let Some(idx) = positive_idx {
                record.get(idx)
                    .map(|s| s.split('|').map(|p| p.trim().to_string()).collect())
                    .unwrap_or_default()
            } else {
                Vec::new()
            };

            let negatives = if let Some(idx) = negative_idx {
                record.get(idx)
                    .map(|s| s.split('|').map(|n| n.trim().to_string()).collect())
                    .unwrap_or_default()
            } else {
                Vec::new()
            };

            let mut example = TrainingExample {
                query,
                positives,
                negatives,
                metadata: None,
            };

            example.positives.truncate(config.max_positives);
            example.negatives.truncate(config.max_negatives);

            examples.push(example);
        }

        tracing::info!("Loaded {} training examples from {:?}", examples.len(), path);

        let mut dataset = Self { examples, config };

        if dataset.config.shuffle {
            dataset.shuffle();
        }

        Ok(dataset)
    }

    /// Load dataset from file, auto-detecting format
    pub fn load(path: impl AsRef<Path>, config: DatasetConfig) -> Result<Self> {
        let path = path.as_ref();
        let extension = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        match extension.to_lowercase().as_str() {
            "jsonl" => Self::from_jsonl(path, config),
            "json" => {
                // Try recipe-mpr format first, then JSONL
                Self::from_recipe_mpr(path, config.clone())
                    .or_else(|_| Self::from_jsonl(path, config))
            }
            "csv" => Self::from_csv(path, config),
            _ => {
                // Try JSONL first, then CSV
                Self::from_jsonl(path, config.clone())
                    .or_else(|_| Self::from_csv(path, config))
            }
        }
    }

    /// Load dataset from Recipe-MPR format (JSON array with multiple choice QA)
    ///
    /// Expected format:
    /// ```json
    /// [
    ///   {
    ///     "query": "I want to make a warm dish containing oysters",
    ///     "options": {
    ///       "08cb462fdf": "Simple creamy oyster soup",
    ///       "5b9441298f": "Seasoned salted crackers shaped like oysters",
    ///       ...
    ///     },
    ///     "answer": "08cb462fdf"
    ///   },
    ///   ...
    /// ]
    /// ```
    ///
    /// This converts to contrastive learning format:
    /// - query: the question
    /// - positive: the correct answer's description
    /// - negatives: the other options' descriptions
    pub fn from_recipe_mpr(path: impl AsRef<Path>, config: DatasetConfig) -> Result<Self> {
        let path = path.as_ref();
        let file_content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read file: {:?}", path))?;

        #[derive(Deserialize)]
        struct RecipeMprExample {
            query: String,
            options: std::collections::HashMap<String, String>,
            answer: String,
        }

        let mpr_examples: Vec<RecipeMprExample> = serde_json::from_str(&file_content)
            .with_context(|| "Failed to parse Recipe-MPR JSON array")?;

        let mut examples = Vec::with_capacity(mpr_examples.len());

        for mpr in mpr_examples {
            // Get the correct answer's description
            let positive = mpr.options.get(&mpr.answer)
                .ok_or_else(|| anyhow::anyhow!(
                    "Answer key '{}' not found in options for query: {}",
                    mpr.answer,
                    mpr.query
                ))?
                .clone();

            // Get all other options as negatives
            let negatives: Vec<String> = mpr.options
                .iter()
                .filter(|(k, _)| **k != mpr.answer)
                .map(|(_, v)| v.clone())
                .collect();

            let mut example = TrainingExample {
                query: mpr.query.trim().to_string(),
                positives: vec![positive],
                negatives,
                metadata: None,
            };

            // Truncate to config limits
            example.positives.truncate(config.max_positives);
            example.negatives.truncate(config.max_negatives);

            examples.push(example);
        }

        tracing::info!(
            "Loaded {} training examples from Recipe-MPR format: {:?}",
            examples.len(),
            path
        );

        let mut dataset = Self { examples, config };

        if dataset.config.shuffle {
            dataset.shuffle();
        }

        Ok(dataset)
    }

    /// Get the number of examples
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Get an example by index
    pub fn get(&self, index: usize) -> Option<&TrainingExample> {
        self.examples.get(index)
    }

    /// Iterate over examples
    pub fn iter(&self) -> impl Iterator<Item = &TrainingExample> {
        self.examples.iter()
    }

    /// Shuffle the dataset
    pub fn shuffle(&mut self) {
        use rand::seq::SliceRandom;
        use rand::SeedableRng;

        let mut rng = if let Some(seed) = self.config.seed {
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::rngs::StdRng::from_entropy()
        };

        self.examples.shuffle(&mut rng);
    }

    /// Split dataset into train/validation sets
    pub fn split(&self, train_ratio: f64) -> (Vec<TrainingExample>, Vec<TrainingExample>) {
        let split_idx = (self.examples.len() as f64 * train_ratio) as usize;
        let train = self.examples[..split_idx].to_vec();
        let val = self.examples[split_idx..].to_vec();
        (train, val)
    }

    /// Get statistics about the dataset
    pub fn stats(&self) -> DatasetStats {
        let total = self.examples.len();
        let total_positives: usize = self.examples.iter().map(|e| e.positives.len()).sum();
        let total_negatives: usize = self.examples.iter().map(|e| e.negatives.len()).sum();
        let avg_query_len: f64 = if total > 0 {
            self.examples.iter().map(|e| e.query.len()).sum::<usize>() as f64 / total as f64
        } else {
            0.0
        };

        DatasetStats {
            total_examples: total,
            avg_positives: if total > 0 { total_positives as f64 / total as f64 } else { 0.0 },
            avg_negatives: if total > 0 { total_negatives as f64 / total as f64 } else { 0.0 },
            avg_query_length: avg_query_len,
        }
    }
}

/// Dataset statistics
#[derive(Debug, Clone)]
pub struct DatasetStats {
    /// Total number of examples
    pub total_examples: usize,
    /// Average positives per example
    pub avg_positives: f64,
    /// Average negatives per example
    pub avg_negatives: f64,
    /// Average query length in characters
    pub avg_query_length: f64,
}

impl std::fmt::Display for DatasetStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Dataset: {} examples, {:.1} avg positives, {:.1} avg negatives, {:.1} avg query chars",
            self.total_examples, self.avg_positives, self.avg_negatives, self.avg_query_length
        )
    }
}

/// Batch iterator for training
pub struct BatchIterator<'a> {
    dataset: &'a TrainingDataset,
    batch_size: usize,
    current_idx: usize,
}

impl<'a> BatchIterator<'a> {
    /// Create a new batch iterator
    pub fn new(dataset: &'a TrainingDataset, batch_size: usize) -> Self {
        Self {
            dataset,
            batch_size,
            current_idx: 0,
        }
    }
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = Vec<&'a TrainingExample>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.dataset.len() {
            return None;
        }

        let end_idx = (self.current_idx + self.batch_size).min(self.dataset.len());
        let batch: Vec<_> = (self.current_idx..end_idx)
            .filter_map(|i| self.dataset.get(i))
            .collect();

        self.current_idx = end_idx;

        if batch.is_empty() {
            None
        } else {
            Some(batch)
        }
    }
}

/// Extension trait for creating batches
pub trait Batched {
    fn batches(&self, batch_size: usize) -> BatchIterator<'_>;
}

impl Batched for TrainingDataset {
    fn batches(&self, batch_size: usize) -> BatchIterator<'_> {
        BatchIterator::new(self, batch_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_jsonl() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, r#"{{"query": "what is rust", "positives": ["Rust is a systems programming language"], "negatives": ["Python is interpreted"]}}"#).unwrap();
        writeln!(file, r#"{{"query": "what is python", "positives": ["Python is a programming language"]}}"#).unwrap();

        let config = DatasetConfig { shuffle: false, ..Default::default() };
        let dataset = TrainingDataset::from_jsonl(file.path(), config).unwrap();
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.get(0).unwrap().query, "what is rust");
    }

    #[test]
    fn test_dataset_stats() {
        let examples = vec![
            TrainingExample {
                query: "test query".to_string(),
                positives: vec!["pos1".to_string(), "pos2".to_string()],
                negatives: vec!["neg1".to_string()],
                metadata: None,
            },
        ];

        let dataset = TrainingDataset::new(examples, DatasetConfig::default());
        let stats = dataset.stats();

        assert_eq!(stats.total_examples, 1);
        assert!((stats.avg_positives - 2.0).abs() < 0.01);
        assert!((stats.avg_negatives - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_batch_iterator() {
        let examples: Vec<_> = (0..10)
            .map(|i| TrainingExample {
                query: format!("query {}", i),
                positives: vec![format!("pos {}", i)],
                negatives: vec![],
                metadata: None,
            })
            .collect();

        let dataset = TrainingDataset::new(examples, DatasetConfig { shuffle: false, ..Default::default() });
        let batches: Vec<_> = dataset.batches(3).collect();

        assert_eq!(batches.len(), 4); // 10 / 3 = 3 full + 1 partial
        assert_eq!(batches[0].len(), 3);
        assert_eq!(batches[3].len(), 1);
    }
}
