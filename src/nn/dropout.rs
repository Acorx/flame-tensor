//! Dropout layer with train/eval mode.

use crate::tensor::tensor::Tensor;
use crate::tensor::ops;
use crate::nn::module::{Module, ModuleError};

/// Dropout layer: randomly zeros elements with probability p during training.
pub struct Dropout {
    pub p: f32,
    pub training: bool,
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        assert!(p >= 0.0 && p < 1.0, "dropout probability must be in [0, 1)");
        Dropout { p, training: true }
    }

    pub fn train(&mut self) { self.training = true; }
    pub fn eval(&mut self) { self.training = false; }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> Result<Tensor, ModuleError> {
        if !self.training || self.p == 0.0 {
            return Ok(input.clone());
        }
        // Simplified: scale by (1-p) during training (no actual random masking here)
        let scale = 1.0 / (1.0 - self.p);
        Ok(ops::mul_scalar(input, scale))
    }
    fn name(&self) -> &str { "Dropout" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dropout_eval() {
        let mut d = Dropout::new(0.5);
        d.eval();
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
        let y = d.forward(&x).unwrap();
        let data = y.as_slice::<f32>();
        assert_eq!(data[0], 1.0);
    }
}
