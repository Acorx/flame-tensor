//! CUDA backend stub (feature-gated).
//!
//! This module compiles when the `cuda` feature is enabled.
//! Actual CUDA operations require a CUDA runtime and are not yet implemented.

use crate::tensor::tensor::Tensor;
use crate::tensor::ops;
use super::Backend;

/// CUDA backend — placeholder.
///
/// All operations currently fall back to CPU. Real CUDA kernels
/// will be implemented in a future version using the cuda-open project.
pub struct CudaBackend {
    device_id: usize,
}

impl CudaBackend {
    pub fn new(device_id: usize) -> Self {
        CudaBackend { device_id }
    }

    pub fn device_id(&self) -> usize { self.device_id }

    pub fn device_count() -> usize {
        // Stub: no CUDA runtime detected
        0
    }
}

impl Backend for CudaBackend {
    fn add(&self, a: &Tensor, b: &Tensor) -> Tensor {
        // Fallback to CPU until CUDA kernels are implemented
        ops::add(a, b)
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        ops::mul(a, b)
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        ops::matmul(a, b)
    }

    fn sum(&self, a: &Tensor, _axis: Option<usize>) -> Tensor {
        ops::add(a, &Tensor::scalar(0.0f32)) // identity fallback
    }

    fn name(&self) -> &str { "CUDA(stub)" }
}
