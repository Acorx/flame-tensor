//! Transformer block and GPT-2 style model.

use crate::tensor::tensor::Tensor;
use crate::tensor::ops;
use crate::nn::module::{Module, ModuleError, Parameter};
use crate::nn::attention::MultiHeadSelfAttention;
use crate::nn::linear::Linear;
use crate::nn::activation::GELU;
use crate::nn::norm::LayerNorm;
use crate::nn::dropout::Dropout;

/// Single transformer block: attention + FFN with residuals and LayerNorm.
pub struct TransformerBlock {
    pub attn: MultiHeadSelfAttention,
    pub ln1: LayerNorm,
    pub ln2: LayerNorm,
    pub ffn1: Linear,
    pub ffn2: Linear,
    pub dropout: Dropout,
    pub d_model: usize,
    pub d_ff: usize,
}

impl TransformerBlock {
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize, causal: bool, dropout_p: f32) -> Self {
        TransformerBlock {
            attn: MultiHeadSelfAttention::new(d_model, n_heads, causal),
            ln1: LayerNorm::new(vec![d_model], 1e-5),
            ln2: LayerNorm::new(vec![d_model], 1e-5),
            ffn1: Linear::new(d_model, d_ff, true),
            ffn2: Linear::new(d_ff, d_model, true),
            dropout: Dropout::new(dropout_p),
            d_model,
            d_ff,
        }
    }
}

impl Module for TransformerBlock {
    fn forward(&self, input: &Tensor) -> Result<Tensor, ModuleError> {
        // input: [batch, seq_len, d_model]
        let batch = input.dims()[0];
        let seq_len = input.dims()[1];

        // Flatten to 2D for layer norm and linear
        let flat = input.reshape(vec![batch * seq_len, self.d_model]);

        // Pre-norm + attention + residual (simplified for v1)
        let normed1 = self.ln1.forward(&flat)?;
        let attn_out = self.attn.forward(
            &normed1.reshape(vec![batch, seq_len, self.d_model])
        )?;
        let attn_flat = attn_out.reshape(vec![batch * seq_len, self.d_model]);
        let x = ops::add(&flat, &attn_flat);

        // Pre-norm + FFN + residual
        let normed2 = self.ln2.forward(&x)?;
        let ffn_h = self.ffn1.forward(&normed2)?;
        let ffn_act = GELU.forward(&ffn_h)?;
        let ffn_out = self.ffn2.forward(&ffn_act)?;
        let out = ops::add(&x, &ffn_out);

        Ok(out.reshape(vec![batch, seq_len, self.d_model]))
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.attn.parameters());
        p.extend(self.ln1.parameters());
        p.extend(self.ln2.parameters());
        p.extend(self.ffn1.parameters());
        p.extend(self.ffn2.parameters());
        p
    }

    fn name(&self) -> &str { "TransformerBlock" }
}

/// GPT-2 style model.
pub struct GPT2Model {
    pub n_layers: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub blocks: Vec<TransformerBlock>,
    pub ln_f: LayerNorm,
}

impl GPT2Model {
    pub fn new(vocab_size: usize, max_seq_len: usize, d_model: usize, n_heads: usize, n_layers: usize, d_ff: usize, dropout: f32) -> Self {
        let blocks = (0..n_layers)
            .map(|_| TransformerBlock::new(d_model, n_heads, d_ff, true, dropout))
            .collect();
        GPT2Model {
            n_layers, d_model, n_heads, d_ff, vocab_size, max_seq_len,
            blocks,
            ln_f: LayerNorm::new(vec![d_model], 1e-5),
        }
    }
}

impl Module for GPT2Model {
    fn forward(&self, input: &Tensor) -> Result<Tensor, ModuleError> {
        let mut x = input.clone();
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        let batch = x.dims()[0];
        let seq_len = x.dims()[1];
        let flat = x.reshape(vec![batch * seq_len, self.d_model]);
        let out = self.ln_f.forward(&flat)?;
        Ok(out.reshape(vec![batch, seq_len, self.d_model]))
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        for block in &self.blocks { p.extend(block.parameters()); }
        p.extend(self.ln_f.parameters());
        p
    }

    fn name(&self) -> &str { "GPT2Model" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_block() {
        let block = TransformerBlock::new(64, 4, 256, true, 0.1);
        let x = Tensor::from_vec(vec![0.0f32; 2 * 8 * 64], vec![2, 8, 64]);
        let y = block.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, 8, 64]);
    }
}
