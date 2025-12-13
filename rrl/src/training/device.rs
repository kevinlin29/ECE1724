//! Device selection and management for training

use anyhow::Result;
use candle_core::Device;
use serde::{Deserialize, Serialize};

/// Device preference for training
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DevicePreference {
    Cuda,
    Metal,
    Cpu,
    Auto,
}

impl Default for DevicePreference {
    fn default() -> Self {
        Self::Auto
    }
}

impl std::str::FromStr for DevicePreference {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "cuda" | "gpu" => Ok(Self::Cuda),
            "metal" => Ok(Self::Metal),
            "cpu" => Ok(Self::Cpu),
            "auto" => Ok(Self::Auto),
            _ => Err(anyhow::anyhow!(
                "Invalid device preference: {}. Valid options: cuda, metal, cpu, auto",
                s
            )),
        }
    }
}

impl std::fmt::Display for DevicePreference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cuda => write!(f, "cuda"),
            Self::Metal => write!(f, "metal"),
            Self::Cpu => write!(f, "cpu"),
            Self::Auto => write!(f, "auto"),
        }
    }
}

/// Select device based on preference
pub fn select_device(preference: DevicePreference) -> Result<Device> {
    match preference {
        DevicePreference::Cuda => {
            #[cfg(feature = "cuda")]
            {
                tracing::info!("Attempting to use CUDA device...");
                match Device::new_cuda(0) {
                    Ok(device) => {
                        tracing::info!("✓ CUDA device selected");
                        Ok(device)
                    }
                    Err(e) => {
                        tracing::warn!("✗ CUDA initialization failed: {}", e);
                        tracing::warn!("Falling back to CPU");
                        Ok(Device::Cpu)
                    }
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                tracing::warn!("CUDA requested but not compiled with 'cuda' feature");
                tracing::warn!("Falling back to CPU");
                Ok(Device::Cpu)
            }
        }

        DevicePreference::Metal => {
            #[cfg(feature = "metal")]
            {
                tracing::info!("Attempting to use Metal device...");
                match Device::new_metal(0) {
                    Ok(device) => {
                        tracing::info!("✓ Metal device selected");
                        Ok(device)
                    }
                    Err(e) => {
                        tracing::warn!("✗ Metal initialization failed: {}", e);
                        tracing::warn!("Falling back to CPU");
                        Ok(Device::Cpu)
                    }
                }
            }
            #[cfg(not(feature = "metal"))]
            {
                tracing::warn!("Metal requested but not compiled with 'metal' feature");
                tracing::warn!("Falling back to CPU");
                Ok(Device::Cpu)
            }
        }

        DevicePreference::Cpu => {
            tracing::info!("✓ CPU device selected");
            Ok(Device::Cpu)
        }

        DevicePreference::Auto => {
            tracing::info!("Auto-selecting best available device...");

            #[cfg(feature = "cuda")]
            {
                if let Ok(device) = Device::new_cuda(0) {
                    tracing::info!("✓ Auto-selected: CUDA GPU");
                    return Ok(device);
                }
            }

            #[cfg(feature = "metal")]
            {
                if let Ok(device) = Device::new_metal(0) {
                    tracing::info!("✓ Auto-selected: Metal GPU (Apple Silicon)");
                    return Ok(device);
                }
            }

            tracing::info!("✓ Auto-selected: CPU");
            Ok(Device::Cpu)
        }
    }
}

/// Get information about the selected device
pub fn device_info(device: &Device) -> DeviceInfo {
    match device {
        Device::Cpu => DeviceInfo {
            device_type: "CPU".to_string(),
            name: "CPU".to_string(),
            memory_gb: None,
            is_gpu: false,
        },
        Device::Cuda(_) => DeviceInfo {
            device_type: "CUDA".to_string(),
            name: "CUDA Device".to_string(),
            memory_gb: None,
            is_gpu: true,
        },
        Device::Metal(_) => DeviceInfo {
            device_type: "Metal".to_string(),
            name: "Metal Device".to_string(),
            memory_gb: None,
            is_gpu: true,
        },
    }
}

/// Information about a device
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub device_type: String,
    pub name: String,
    pub memory_gb: Option<f32>,
    pub is_gpu: bool,
}

impl std::fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.name, self.device_type)?;
        if let Some(mem) = self.memory_gb {
            write!(f, " - {:.1} GB", mem)?;
        }
        Ok(())
    }
}

/// Check if CUDA is available
pub fn is_cuda_available() -> bool {
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
pub fn is_metal_available() -> bool {
    #[cfg(feature = "metal")]
    {
        Device::new_metal(0).is_ok()
    }
    #[cfg(not(feature = "metal"))]
    {
        false
    }
}

/// Print available devices
pub fn print_available_devices() {
    println!("Available devices:");
    println!("  CPU: ✓ Always available");

    #[cfg(feature = "cuda")]
    {
        if is_cuda_available() {
            println!("  CUDA: ✓ Available");
        } else {
            println!("  CUDA: ✗ Not available");
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        println!("  CUDA: ✗ Not compiled (use --features cuda)");
    }

    #[cfg(feature = "metal")]
    {
        if is_metal_available() {
            println!("  Metal: ✓ Available");
        } else {
            println!("  Metal: ✗ Not available");
        }
    }
    #[cfg(not(feature = "metal"))]
    {
        println!("  Metal: ✗ Not compiled (use --features metal)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_preference_from_str() {
        assert_eq!(
            "cuda".parse::<DevicePreference>().unwrap(),
            DevicePreference::Cuda
        );
        assert_eq!(
            "cpu".parse::<DevicePreference>().unwrap(),
            DevicePreference::Cpu
        );
    }

    #[test]
    fn test_cpu_always_available() {
        let device = select_device(DevicePreference::Cpu);
        assert!(device.is_ok());
    }
}