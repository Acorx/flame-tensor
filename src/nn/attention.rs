//! Multi-head self-attention mechanism.

use crate::tensor::tensor::Tensor;
use crate::tensor::ops;
use crate::nn::module::{Module, ModuleError, Parameter};
use crate::nn::linear::Linear;

/// Multi-head self-attention (v1: simplified for compilation).
pub struct MultiHeadSelfAttention {
    pub n_heads: usize,
    pub d_model: usize,
    pub d_head: usize,
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub causal: bool,
}

impl MultiHeadSelfAttention {
    pub fn new(d_model: usize, n_heads: usize, causal: bool) -> Self {
        let d_head = d_model / n_heads;
        assert_eq!(d_model % n_heads, 0, "d_model must be divisible by n_heads");
        MultiHeadSelfAttention {
            n_heads, d_model, d_head,
            wq: Linear::new(d_model, d_model, true),
            wk: Linear::new(d_model, d_model, true),
            wv: Linear::new(d_model, d_model, true),
            wo: Linear::new(d_model, d_model, true),
            causal,
        }
    }
}

impl Module for MultiHeadSelfAttention {
    fn forward(&self, input: &Tensor) -> Result<Tensor, ModuleError> {
        // input: [batch, seq_len, d_model]
        let batch = input.dims()[0];
        let seq_len = input.dims()[1];

        // Flatten to 2D for Linear
        let flat = input.reshape(vec![batch * seq_len, self.d_model]);

        // Project Q, K, V
        let q = self.wq.forward(&flat)?; // [batch*seq_len, d_model]
        let _k = self.wk.forward(&flat)?;
        let v = self.wv.forward(&flat)?;

        // v1 simplified: skip actual attention computation, just project V through wo
        // Full attention with batched matmul will come in v2
        let out = self.wo.forward(&v)?;

        Ok(out.reshape(vec![batch, seq_len, self.d_model]))
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.wq.parameters());
        p.extend(self.wk.parameters());
        p.extend(self.wv.parameters());
        p.extend(self.wo.parameters());
        p
    }

    fn name(&self) -> &str { "MultiHeadSelfAttention" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_shape() {
        let attn = MultiHeadSelfAttention::new(64, 4, true);
        let x = Tensor::from_vec(vec![0.0f32; 2 * 8 * 64], vec![2, 8, 64]);
        let y = attn.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, 8, 64]);
    }
}
