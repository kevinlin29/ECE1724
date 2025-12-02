//! Device abstraction layer for CPU/CUDA/Metal support
//!
//! Provides unified device selection across different hardware backends.

use anyhow::anyhow;
#[cfg(feature = "training")]
use anyhow::Result;
#[cfg(feature = "training")]
use candle_core::Device;

/// Device preference for training and inference
#[derive(Debug, Clone, PartialEq)]
pub enum DevicePreference {
    /// Automatically select the best available device (CUDA > Metal > CPU)
    Auto,
    /// Use a specific CUDA device by index
    Cuda(usize),
    /// Use a specific Metal device by index (macOS only)
    Metal(usize),
    /// Use CPU only
    Cpu,
}

impl Default for DevicePreference {
    fn default() -> Self {
        Self::Auto
    }
}

impl std::fmt::Display for DevicePreference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DevicePreference::Auto => write!(f, "auto"),
            DevicePreference::Cuda(id) => write!(f, "cuda:{}", id),
            DevicePreference::Metal(id) => write!(f, "metal:{}", id),
            DevicePreference::Cpu => write!(f, "cpu"),
        }
    }
}

impl std::str::FromStr for DevicePreference {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.to_lowercase();
        if s == "auto" {
            Ok(DevicePreference::Auto)
        } else if s == "cpu" {
            Ok(DevicePreference::Cpu)
        } else if s == "cuda" || s == "cuda:0" {
            Ok(DevicePreference::Cuda(0))
        } else if s.starts_with("cuda:") {
            let id: usize = s[5..].parse().map_err(|_| anyhow!("Invalid CUDA device ID"))?;
            Ok(DevicePreference::Cuda(id))
        } else if s == "metal" || s == "metal:0" {
            Ok(DevicePreference::Metal(0))
        } else if s.starts_with("metal:") {
            let id: usize = s[6..].parse().map_err(|_| anyhow!("Invalid Metal device ID"))?;
            Ok(DevicePreference::Metal(id))
        } else {
            Err(anyhow!("Unknown device: {}. Use 'auto', 'cpu', 'cuda', 'cuda:N', 'metal', or 'metal:N'", s))
        }
    }
}

/// Select a device based on preference
///
/// # Arguments
/// * `pref` - Device preference
///
/// # Returns
/// * `Result<Device>` - Selected device or error if unavailable
#[cfg(feature = "training")]
pub fn select_device(pref: DevicePreference) -> Result<Device> {
    match pref {
        DevicePreference::Auto => auto_select_device(),
        DevicePreference::Cuda(id) => {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(id).map_err(|e| anyhow!("Failed to initialize CUDA device {}: {}", id, e))
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = id;
                Err(anyhow!("CUDA support not enabled. Rebuild with --features cuda"))
            }
        }
        DevicePreference::Metal(id) => {
            #[cfg(feature = "metal")]
            {
                Device::new_metal(id).map_err(|e| anyhow!("Failed to initialize Metal device {}: {}", id, e))
            }
            #[cfg(not(feature = "metal"))]
            {
                let _ = id;
                Err(anyhow!("Metal support not enabled. Rebuild with --features metal"))
            }
        }
        DevicePreference::Cpu => Ok(Device::Cpu),
    }
}

/// Automatically select the best available device
///
/// Tries CUDA first, then Metal, then falls back to CPU.
#[cfg(feature = "training")]
pub fn auto_select_device() -> Result<Device> {
    // Try CUDA first
    #[cfg(feature = "cuda")]
    {
        if let Ok(device) = Device::new_cuda(0) {
            tracing::info!("Auto-selected CUDA device 0");
            return Ok(device);
        }
        tracing::debug!("CUDA not available, trying Metal...");
    }

    // Try Metal
    #[cfg(feature = "metal")]
    {
        if let Ok(device) = Device::new_metal(0) {
            tracing::info!("Auto-selected Metal device 0");
            return Ok(device);
        }
        tracing::debug!("Metal not available, falling back to CPU");
    }

    // Fallback to CPU
    tracing::info!("Using CPU device");
    Ok(Device::Cpu)
}

/// Get device information as a string
#[cfg(feature = "training")]
pub fn device_info(device: &Device) -> String {
    match device {
        Device::Cpu => "CPU".to_string(),
        Device::Cuda(_) => "CUDA GPU".to_string(),
        Device::Metal(_) => "Metal GPU".to_string(),
    }
}

/// Check if CUDA is available
#[cfg(feature = "training")]
pub fn cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        Device::new_cuda(0).is_ok()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Check if Metal is available
#[cfg(feature = "training")]
pub fn metal_available() -> bool {
    #[cfg(feature = "metal")]
    {
        Device::new_metal(0).is_ok()
    }
    #[cfg(not(feature = "metal"))]
    {
        false
    }
}

/// Count available CUDA devices
#[cfg(all(feature = "training", feature = "cuda"))]
pub fn cuda_device_count() -> usize {
    let mut count = 0;
    while Device::new_cuda(count).is_ok() {
        count += 1;
        if count > 16 {
            // Safety limit
            break;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_preference_from_str() {
        assert_eq!(
            "auto".parse::<DevicePreference>().unwrap(),
            DevicePreference::Auto
        );
        assert_eq!(
            "cpu".parse::<DevicePreference>().unwrap(),
            DevicePreference::Cpu
        );
        assert_eq!(
            "cuda".parse::<DevicePreference>().unwrap(),
            DevicePreference::Cuda(0)
        );
        assert_eq!(
            "cuda:0".parse::<DevicePreference>().unwrap(),
            DevicePreference::Cuda(0)
        );
        assert_eq!(
            "cuda:1".parse::<DevicePreference>().unwrap(),
            DevicePreference::Cuda(1)
        );
        assert_eq!(
            "metal".parse::<DevicePreference>().unwrap(),
            DevicePreference::Metal(0)
        );
        assert_eq!(
            "metal:0".parse::<DevicePreference>().unwrap(),
            DevicePreference::Metal(0)
        );
    }

    #[test]
    fn test_device_preference_display() {
        assert_eq!(DevicePreference::Auto.to_string(), "auto");
        assert_eq!(DevicePreference::Cpu.to_string(), "cpu");
        assert_eq!(DevicePreference::Cuda(0).to_string(), "cuda:0");
        assert_eq!(DevicePreference::Cuda(1).to_string(), "cuda:1");
        assert_eq!(DevicePreference::Metal(0).to_string(), "metal:0");
    }
}
