//! Embedding layer and positional encoding.

use crate::tensor::tensor::Tensor;
use crate::tensor::ops;
use crate::nn::module::{Module, ModuleError, Parameter};

/// Token embedding layer: lookup table of shape [vocab_size, embedding_dim].
pub struct Embedding {
    pub weight: Tensor,
    vocab_size: usize,
    embedding_dim: usize,
}

impl Embedding {
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        // Initialize with small random values
        let n = vocab_size * embedding_dim;
        let mut data: Vec<f32> = Vec::with_capacity(n);
        let mut state: u32 = 0xDEADBEEF;
        for _ in 0..n {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let v = ((state >> 16) & 0x7fff) as f32 / 32767.0;
            data.push((v - 0.5) * 0.02);
        }
        Embedding {
            weight: Tensor::from_vec(data, vec![vocab_size, embedding_dim]),
            vocab_size,
            embedding_dim,
        }
    }

    /// Look up embeddings for a set of indices.
    pub fn lookup(&self, indices: &[usize]) -> Tensor {
        let batch = indices.len();
        let mut result = Vec::with_capacity(batch * self.embedding_dim);
        for &idx in indices {
            let row = self.weight.as_slice::<f32>()[idx * self.embedding_dim..(idx + 1) * self.embedding_dim].to_vec();
            result.extend_from_slice(&row);
        }
        Tensor::from_vec(result, vec![batch, self.embedding_dim])
    }
}

impl Module for Embedding {
    fn forward(&self, input: &Tensor) -> Result<Tensor, ModuleError> {
        // Input: [batch, seq_len] of token IDs (as f32 for now)
        let data = input.as_slice::<f32>();
        let n = input.numel();
        let mut indices: Vec<usize> = Vec::with_capacity(n);
        for &v in data { indices.push(v as usize); }
        let mut result = Vec::with_capacity(n * self.embedding_dim);
        for &idx in &indices {
            if idx >= self.vocab_size {
                return Err(ModuleError::DimError(format!("index {} >= vocab_size {}", idx, self.vocab_size)));
            }
            let row = self.weight.as_slice::<f32>()[idx * self.embedding_dim..(idx + 1) * self.embedding_dim].to_vec();
            result.extend_from_slice(&row);
        }
        let out_shape = if input.ndim() == 1 { vec![n, self.embedding_dim] }
                        else { vec![input.dims()[0], input.dims()[1], self.embedding_dim] };
        Ok(Tensor::from_vec(result, out_shape))
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![Parameter::new("weight", self.weight.clone())]
    }

    fn name(&self) -> &str { "Embedding" }
}

/// Sinusoidal positional encoding (as in "Attention Is All You Need").
pub struct PositionalEncoding {
    pub encoding: Tensor, // [max_len, d_model]
    pub d_model: usize,
    pub max_len: usize,
}

impl PositionalEncoding {
    pub fn new(d_model: usize, max_len: usize) -> Self {
        let mut data = vec![0.0f32; max_len * d_model];
        for pos in 0..max_len {
            for i in 0..d_model / 2 {
                let dim = 2 * i;
                let angle = pos as f32 / 10000.0f32.powi(dim as i32 / d_model as i32);
                data[pos * d_model + dim] = angle.sin();
                if dim + 1 < d_model {
                    data[pos * d_model + dim + 1] = angle.cos();
                }
            }
        }
        PositionalEncoding {
            encoding: Tensor::from_vec(data, vec![max_len, d_model]),
            d_model,
            max_len,
        }
    }
}

impl Module for PositionalEncoding {
    fn forward(&self, input: &Tensor) -> Result<Tensor, ModuleError> {
        // input: [batch, seq_len, d_model]
        let seq_len = if input.ndim() >= 2 { input.dims()[input.ndim() - 2] } else { 1 };
        if seq_len > self.max_len {
            return Err(ModuleError::DimError(format!("seq_len {} > max_len {}", seq_len, self.max_len)));
        }
        // Slice positional encoding up to seq_len
        let pe_data = self.encoding.as_slice::<f32>();
        let pe_slice = &pe_data[..seq_len * self.d_model];
        let pe = Tensor::from_vec(pe_slice.to_vec(), vec![seq_len, self.d_model]);
        Ok(ops::add(input, &pe))
    }
    fn name(&self) -> &str { "PositionalEncoding" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding() {
        let emb = Embedding::new(100, 32);
        let x = Tensor::from_vec(vec![0.0f32, 1.0, 2.0], vec![3]);
        let y = emb.forward(&x).unwrap();
        assert_eq!(y.dims(), &[3, 32]);
    }

    #[test]
    fn test_pos_encoding() {
        let pe = PositionalEncoding::new(64, 512);
        let x = Tensor::from_vec(vec![0.0f32; 2 * 10 * 64], vec![2, 10, 64]);
        let y = pe.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, 10, 64]);
    }
}
