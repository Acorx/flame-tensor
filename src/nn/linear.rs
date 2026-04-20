//! Linear (fully connected) layer: y = xW^T + b.

use crate::tensor::tensor::Tensor;
use crate::tensor::ops;
use crate::nn::module::{Module, ModuleError, Parameter};

/// Fully connected layer.
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    in_features: usize,
    out_features: usize,
}

impl Linear {
    /// Create a new Linear layer with Kaiming uniform initialization.
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        let weight = kaiming_uniform(in_features, out_features);
        let b = if bias {
            Some(Tensor::from_vec(vec![0.0f32; out_features], vec![out_features]))
        } else {
            None
        };
        Linear { weight, bias: b, in_features, out_features }
    }

    pub fn in_features(&self) -> usize { self.in_features }
    pub fn out_features(&self) -> usize { self.out_features }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor, ModuleError> {
        // input: [..., in_features] → output: [..., out_features]
        if input.ndim() < 1 {
            return Err(ModuleError::DimError("input must have at least 1 dim".into()));
        }
        let last_dim = input.dims()[input.ndim() - 1];
        if last_dim != self.in_features {
            return Err(ModuleError::ShapeMismatch {
                expected: vec![self.in_features],
                got: vec![last_dim],
            });
        }
        // x @ W + b  (weight is [in_features, out_features])
        let output = ops::matmul(input, &self.weight);
        let mut output = output;
        if let Some(ref b) = self.bias {
            output = ops::add(&output, b);
        }
        Ok(output)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = vec![Parameter::new("weight", self.weight.clone())];
        if let Some(ref b) = self.bias {
            params.push(Parameter::new("bias", b.clone()));
        }
        params
    }

    fn name(&self) -> &str { "Linear" }
}

/// Kaiming uniform initialization for weight matrices.
fn kaiming_uniform(fan_in: usize, fan_out: usize) -> Tensor {
    let bound = (6.0f32 / (fan_in as f32 + fan_out as f32)).sqrt();
    let n = fan_in * fan_out;
    let mut data: Vec<f32> = Vec::with_capacity(n);
    // Simple LCG pseudo-random for init (no rand dep)
    let mut state: u32 = 0x12345678;
    for _ in 0..n {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let v = ((state >> 16) & 0x7fff) as f32 / 32767.0;
        data.push(v * 2.0 * bound - bound);
    }
    Tensor::from_vec(data, vec![fan_in, fan_out])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward() {
        let lin = Linear::new(4, 2, true);
        let x = Tensor::from_vec(vec![1.0f32; 4], vec![1, 4]);
        let y = lin.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 2]);
    }

    #[test]
    fn test_linear_no_bias() {
        let lin = Linear::new(3, 5, false);
        assert!(lin.bias.is_none());
        let x = Tensor::from_vec(vec![1.0f32; 3], vec![1, 3]);
        let y = lin.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 5]);
    }
}
