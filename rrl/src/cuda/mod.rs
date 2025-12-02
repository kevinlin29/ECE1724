//! Custom CUDA kernels for operations not available in candle
//!
//! This module provides CUDA implementations for:
//! - Layer normalization (forward pass)

pub mod kernels;
pub mod layer_norm;

pub use layer_norm::cuda_layer_norm;
