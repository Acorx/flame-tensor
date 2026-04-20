//! CPU backend with SIMD hints and rayon parallelism.

use crate::tensor::tensor::Tensor;
use crate::tensor::dtype::DType;
use crate::tensor::ops;
use crate::tensor::reduce;
use rayon::prelude::*;
use super::Backend;

/// CPU backend using standard operations with optional rayon parallelism.
pub struct CpuBackend {
    pub parallel_threshold: usize,
}

impl CpuBackend {
    pub fn new() -> Self {
        CpuBackend { parallel_threshold: 100_000 }
    }
}

impl Default for CpuBackend {
    fn default() -> Self { Self::new() }
}

impl Backend for CpuBackend {
    fn add(&self, a: &Tensor, b: &Tensor) -> Tensor {
        if a.numel() >= self.parallel_threshold {
            parallel_elementwise(a, b, |x, y| x + y)
        } else {
            ops::add(a, b)
        }
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        if a.numel() >= self.parallel_threshold {
            parallel_elementwise(a, b, |x, y| x * y)
        } else {
            ops::mul(a, b)
        }
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        ops::matmul(a, b)
    }

    fn sum(&self, a: &Tensor, axis: Option<usize>) -> Tensor {
        match axis {
            Some(ax) => reduce::sum(a, ax),
            None => reduce::sum_all(a),
        }
    }

    fn name(&self) -> &str { "CPU" }
}

/// Parallel element-wise operation for large tensors.
fn parallel_elementwise(a: &Tensor, b: &Tensor, op: impl Fn(f32, f32) -> f32 + Sync) -> Tensor {
    assert_eq!(a.shape(), b.shape(), "shapes must match for parallel element-wise op");
    let a_data = a.as_slice::<f32>();
    let b_data = b.as_slice::<f32>();
    let result: Vec<f32> = a_data.par_iter().enumerate()
        .map(|(i, &x)| op(x, b_data[i]))
        .collect();
    Tensor::from_vec(result, a.dims().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_add() {
        let backend = CpuBackend::new();
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
        let b = Tensor::from_vec(vec![4.0f32, 5.0, 6.0], vec![3]);
        let c = backend.add(&a, &b);
        let data = c.as_slice::<f32>();
        assert_eq!(data[0], 5.0);
        assert_eq!(data[2], 9.0);
    }

    #[test]
    fn test_cpu_backend_matmul() {
        let backend = CpuBackend::new();
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = backend.matmul(&a, &b);
        assert_eq!(c.dims(), &[2, 2]);
    }
}
