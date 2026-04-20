//! Normalization layers: LayerNorm, BatchNorm1d, RMSNorm.

use crate::tensor::tensor::Tensor;
use crate::tensor::ops;
use crate::nn::module::{Module, ModuleError, Parameter};

/// Layer normalization over the last D dimensions.
pub struct LayerNorm {
    pub weight: Tensor, // gamma, shape [normalized_shape]
    pub bias: Tensor,   // beta, shape [normalized_shape]
    pub eps: f32,
    normalized_shape: Vec<usize>,
}

impl LayerNorm {
    pub fn new(normalized_shape: Vec<usize>, eps: f32) -> Self {
        let n: usize = normalized_shape.iter().product();
        LayerNorm {
            weight: Tensor::from_vec(vec![1.0f32; n], normalized_shape.clone()),
            bias: Tensor::from_vec(vec![0.0f32; n], normalized_shape.clone()),
            eps,
            normalized_shape,
        }
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor, ModuleError> {
        // Normalize over last dim, apply gamma + beta
        let last_dim = input.ndim().saturating_sub(1);
        let mean = ops::mean(input, last_dim);
        let diff = ops::sub(input, &mean);
        let var = ops::mean(&ops::mul(&diff, &diff), last_dim);
        let std = ops::add_scalar(&var, self.eps);
        let std = ops::sqrt(&std);
        let normed = ops::div(&diff, &std);
        let out = ops::mul(&normed, &self.weight);
        Ok(ops::add(&out, &self.bias))
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![
            Parameter::new("weight", self.weight.clone()),
            Parameter::new("bias", self.bias.clone()),
        ]
    }

    fn name(&self) -> &str { "LayerNorm" }
}

/// Batch normalization for 1D inputs.
pub struct BatchNorm1d {
    pub weight: Tensor,
    pub bias: Tensor,
    pub running_mean: Tensor,
    pub running_var: Tensor,
    pub eps: f32,
    pub momentum: f32,
    pub training: bool,
}

impl BatchNorm1d {
    pub fn new(num_features: usize, eps: f32, momentum: f32) -> Self {
        BatchNorm1d {
            weight: Tensor::from_vec(vec![1.0f32; num_features], vec![num_features]),
            bias: Tensor::from_vec(vec![0.0f32; num_features], vec![num_features]),
            running_mean: Tensor::from_vec(vec![0.0f32; num_features], vec![num_features]),
            running_var: Tensor::from_vec(vec![1.0f32; num_features], vec![num_features]),
            eps,
            momentum,
            training: true,
        }
    }
}

impl Module for BatchNorm1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor, ModuleError> {
        // Simplified: use running stats
        let normed = ops::div(
            &ops::sub(input, &self.running_mean),
            &ops::sqrt(&ops::add_scalar(&self.running_var, self.eps)),
        );
        Ok(ops::add(&ops::mul(&normed, &self.weight), &self.bias))
    }
    fn name(&self) -> &str { "BatchNorm1d" }
}

/// RMS normalization.
pub struct RMSNorm {
    pub weight: Tensor,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(num_features: usize, eps: f32) -> Self {
        RMSNorm {
            weight: Tensor::from_vec(vec![1.0f32; num_features], vec![num_features]),
            eps,
        }
    }
}

impl Module for RMSNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor, ModuleError> {
        let last_dim = input.ndim().saturating_sub(1);
        let pow2 = ops::mul(input, input);
        let mean = ops::mean(&pow2, last_dim);
        let rms = ops::sqrt(&ops::add_scalar(&mean, self.eps));
        let normed = ops::div(input, &rms);
        Ok(ops::mul(&normed, &self.weight))
    }
    fn name(&self) -> &str { "RMSNorm" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layernorm() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![1, 4]);
        let y = ln.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 4]);
    }
}
