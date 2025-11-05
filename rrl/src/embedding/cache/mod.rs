//! Persistent embedding cache
//!
//! SQLite-based cache with versioning and reproducibility manifests.

use crate::embedding::Embedding;
use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;

/// Embedding cache backed by SQLite
pub struct EmbeddingCache {
    conn: Connection,
    model_name: String,
}

impl EmbeddingCache {
    /// Create a new embedding cache at the given path
    pub fn new(db_path: &Path, model_name: String) -> Result<Self> {
        let conn = Connection::open(db_path)
            .context(format!("Failed to open cache database: {:?}", db_path))?;

        // Create the embeddings table if it doesn't exist
        conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY,
                text_hash TEXT NOT NULL,
                model_name TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                dimension INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                UNIQUE(text_hash, model_name)
            )",
            [],
        )?;

        // Create index for faster lookups
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_text_hash_model
             ON embeddings(text_hash, model_name)",
            [],
        )?;

        // Create metadata table for versioning
        conn.execute(
            "CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )",
            [],
        )?;

        Ok(Self { conn, model_name })
    }

    /// Generate a hash for the text
    fn hash_text(text: &str) -> String {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Serialize an embedding to bytes
    fn serialize_embedding(embedding: &Embedding) -> Vec<u8> {
        embedding
            .iter()
            .flat_map(|&f| f.to_le_bytes())
            .collect()
    }

    /// Deserialize an embedding from bytes
    fn deserialize_embedding(bytes: &[u8]) -> Result<Embedding> {
        if bytes.len() % 4 != 0 {
            anyhow::bail!("Invalid embedding bytes length");
        }

        let mut embedding = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            let bytes: [u8; 4] = chunk.try_into()?;
            embedding.push(f32::from_le_bytes(bytes));
        }

        Ok(embedding)
    }

    /// Get an embedding from the cache
    pub fn get(&self, text: &str) -> Result<Option<Embedding>> {
        let text_hash = Self::hash_text(text);

        let mut stmt = self.conn.prepare(
            "SELECT embedding FROM embeddings
             WHERE text_hash = ?1 AND model_name = ?2",
        )?;

        let result = stmt.query_row(params![text_hash, self.model_name], |row| {
            let bytes: Vec<u8> = row.get(0)?;
            Ok(bytes)
        });

        match result {
            Ok(bytes) => Ok(Some(Self::deserialize_embedding(&bytes)?)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Store an embedding in the cache
    pub fn put(&self, text: &str, embedding: &Embedding) -> Result<()> {
        let text_hash = Self::hash_text(text);
        let embedding_bytes = Self::serialize_embedding(embedding);
        let dimension = embedding.len() as i32;
        let created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs() as i64;

        self.conn.execute(
            "INSERT OR REPLACE INTO embeddings
             (text_hash, model_name, text, embedding, dimension, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                text_hash,
                self.model_name,
                text,
                embedding_bytes,
                dimension,
                created_at
            ],
        )?;

        Ok(())
    }

    /// Get or compute an embedding (with caching)
    pub fn get_or_compute<F>(&self, text: &str, compute_fn: F) -> Result<Embedding>
    where
        F: FnOnce(&str) -> Result<Embedding>,
    {
        // Try to get from cache first
        if let Some(embedding) = self.get(text)? {
            tracing::debug!("Cache hit for text: {}", &text[..text.len().min(50)]);
            return Ok(embedding);
        }

        // Cache miss - compute the embedding
        tracing::debug!("Cache miss for text: {}", &text[..text.len().min(50)]);
        let embedding = compute_fn(text)?;

        // Store in cache
        self.put(text, &embedding)?;

        Ok(embedding)
    }

    /// Get cache statistics
    pub fn stats(&self) -> Result<CacheStats> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM embeddings", [], |row| row.get(0))?;

        let model_count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM embeddings WHERE model_name = ?1",
            params![self.model_name],
            |row| row.get(0),
        )?;

        Ok(CacheStats {
            total_entries: count as usize,
            model_entries: model_count as usize,
            model_name: self.model_name.clone(),
        })
    }

    /// Clear all cached embeddings for the current model
    pub fn clear_model(&self) -> Result<usize> {
        let deleted = self.conn.execute(
            "DELETE FROM embeddings WHERE model_name = ?1",
            params![self.model_name],
        )?;
        Ok(deleted)
    }

    /// Clear all cached embeddings
    pub fn clear_all(&self) -> Result<usize> {
        let deleted = self.conn.execute("DELETE FROM embeddings", [])?;
        Ok(deleted)
    }

    /// Set a metadata value
    pub fn set_metadata(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?1, ?2)",
            params![key, value],
        )?;
        Ok(())
    }

    /// Get a metadata value
    pub fn get_metadata(&self, key: &str) -> Result<Option<String>> {
        let result = self
            .conn
            .query_row("SELECT value FROM metadata WHERE key = ?1", params![key], |row| {
                row.get(0)
            });

        match result {
            Ok(value) => Ok(Some(value)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}

/// Cache statistics
#[derive(Debug)]
pub struct CacheStats {
    pub total_entries: usize,
    pub model_entries: usize,
    pub model_name: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_cache_put_get() {
        let temp_file = NamedTempFile::new().unwrap();
        let cache = EmbeddingCache::new(temp_file.path(), "test-model".to_string()).unwrap();

        let text = "Hello, world!";
        let embedding = vec![1.0, 2.0, 3.0];

        // Should not exist initially
        assert!(cache.get(text).unwrap().is_none());

        // Put the embedding
        cache.put(text, &embedding).unwrap();

        // Should now exist
        let retrieved = cache.get(text).unwrap().unwrap();
        assert_eq!(retrieved, embedding);
    }

    #[test]
    fn test_cache_get_or_compute() {
        let temp_file = NamedTempFile::new().unwrap();
        let cache = EmbeddingCache::new(temp_file.path(), "test-model".to_string()).unwrap();

        let text = "Test text";
        let expected = vec![4.0, 5.0, 6.0];

        // First call should compute
        let result1 = cache.get_or_compute(text, |_| Ok(expected.clone())).unwrap();
        assert_eq!(result1, expected);

        // Verify it was cached
        let cached = cache.get(text).unwrap();
        assert!(cached.is_some());
        assert_eq!(cached.unwrap(), expected);

        // Second call should use cache (verify by calling again)
        let result2 = cache.get_or_compute(text, |_| Ok(expected.clone())).unwrap();
        assert_eq!(result2, expected);
    }

    #[test]
    fn test_cache_stats() {
        let temp_file = NamedTempFile::new().unwrap();
        let cache = EmbeddingCache::new(temp_file.path(), "test-model".to_string()).unwrap();

        cache.put("text1", &vec![1.0, 2.0]).unwrap();
        cache.put("text2", &vec![3.0, 4.0]).unwrap();

        let stats = cache.stats().unwrap();
        assert_eq!(stats.model_entries, 2);
    }

    #[test]
    fn test_metadata() {
        let temp_file = NamedTempFile::new().unwrap();
        let cache = EmbeddingCache::new(temp_file.path(), "test-model".to_string()).unwrap();

        cache.set_metadata("version", "1.0").unwrap();
        let value = cache.get_metadata("version").unwrap();
        assert_eq!(value, Some("1.0".to_string()));

        let missing = cache.get_metadata("nonexistent").unwrap();
        assert_eq!(missing, None);
    }
}
