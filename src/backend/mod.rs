//! Backend abstraction: CPU and optional CUDA.

pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;

pub use cpu::CpuBackend;
#[cfg(feature = "cuda")]
pub use cuda::CudaBackend;

use crate::tensor::tensor::Tensor;
use crate::tensor::dtype::DType;

/// Backend trait for tensor operations.
pub trait Backend {
    /// Element-wise addition.
    fn add(&self, a: &Tensor, b: &Tensor) -> Tensor;
    /// Element-wise multiplication.
    fn mul(&self, a: &Tensor, b: &Tensor) -> Tensor;
    /// Matrix multiplication.
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Tensor;
    /// Sum reduction.
    fn sum(&self, a: &Tensor, axis: Option<usize>) -> Tensor;
    /// Name of the backend.
    fn name(&self) -> &str;
}
