//! Activation functions as Module implementations.

use crate::tensor::tensor::Tensor;
use crate::tensor::ops;
use crate::nn::module::{Module, ModuleError};

/// ReLU activation.
pub struct ReLU;

impl Module for ReLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor, ModuleError> {
        Ok(ops::relu(input))
    }
    fn name(&self) -> &str { "ReLU" }
}

/// GELU activation (approximate).
pub struct GELU;

impl Module for GELU {
    fn forward(&self, input: &Tensor) -> Result<Tensor, ModuleError> {
        // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        Ok(ops::gelu(input))
    }
    fn name(&self) -> &str { "GELU" }
}

/// SiLU (Swish) activation: x * sigmoid(x).
pub struct SiLU;

impl Module for SiLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor, ModuleError> {
        let sig = ops::sigmoid(input);
        Ok(ops::mul(input, &sig))
    }
    fn name(&self) -> &str { "SiLU" }
}

/// Tanh activation.
pub struct Tanh;

impl Module for Tanh {
    fn forward(&self, input: &Tensor) -> Result<Tensor, ModuleError> {
        Ok(ops::tanh(input))
    }
    fn name(&self) -> &str { "Tanh" }
}

/// Sigmoid activation.
pub struct Sigmoid;

impl Module for Sigmoid {
    fn forward(&self, input: &Tensor) -> Result<Tensor, ModuleError> {
        Ok(ops::sigmoid(input))
    }
    fn name(&self) -> &str { "Sigmoid" }
}

/// Softmax along the last dimension.
pub struct Softmax;

impl Module for Softmax {
    fn forward(&self, input: &Tensor) -> Result<Tensor, ModuleError> {
        Ok(ops::softmax(input, -1))
    }
    fn name(&self) -> &str { "Softmax" }
}
/// LeakyReLU activation.
pub struct LeakyReLU {
    pub negative_slope: f32,
}

impl LeakyReLU {
    pub fn new(slope: f32) -> Self { LeakyReLU { negative_slope: slope } }
}

impl Module for LeakyReLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor, ModuleError> {
        Ok(ops::leaky_relu(input, self.negative_slope))
    }
    fn name(&self) -> &str { "LeakyReLU" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let x = Tensor::from_vec(vec![-1.0f32, 0.0, 1.0, 2.0], vec![4]);
        let y = ReLU.forward(&x).unwrap();
        let data = y.as_slice::<f32>();
        assert_eq!(data[0], 0.0);
        assert_eq!(data[2], 1.0);
    }

    #[test]
    fn test_sigmoid() {
        let x = Tensor::from_vec(vec![0.0f32], vec![1]);
        let y = Sigmoid.forward(&x).unwrap();
        let data = y.as_slice::<f32>();
        assert!((data[0] - 0.5).abs() < 1e-5);
    }
}
