//! Convolution layers: Conv1d, Conv2d, ConvTranspose2d.

use crate::tensor::tensor::Tensor;
use crate::tensor::ops;
use crate::nn::module::{Module, ModuleError, Parameter};

/// 1D convolution layer.
pub struct Conv1d {
    pub weight: Tensor, // [out_channels, in_channels/groups, kernel_size]
    pub bias: Option<Tensor>,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl Conv1d {
    pub fn new(in_ch: usize, out_ch: usize, kernel_size: usize, stride: usize, padding: usize) -> Self {
        let w = Tensor::from_vec(vec![0.0f32; out_ch * (in_ch / 1) * kernel_size], vec![out_ch, in_ch, kernel_size]);
        let b = Some(Tensor::from_vec(vec![0.0f32; out_ch], vec![out_ch]));
        Conv1d { weight: w, bias: b, stride, padding, dilation: 1, groups: 1 }
    }
}

impl Module for Conv1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor, ModuleError> {
        // input: [batch, in_channels, length]
        let out_len = (input.dims()[2] + 2 * self.padding - self.weight.dims()[2]) / self.stride + 1;
        let out_ch = self.weight.dims()[0];
        let batch = input.dims()[0];
        let out = Tensor::from_vec(vec![0.0f32; batch * out_ch * out_len], vec![batch, out_ch, out_len]);
        // Simplified: return zeros for now (full im2col in v2)
        Ok(out)
    }
    fn name(&self) -> &str { "Conv1d" }
}

/// 2D convolution layer.
pub struct Conv2d {
    pub weight: Tensor, // [out_channels, in_channels/groups, kH, kW]
    pub bias: Option<Tensor>,
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub dilation: (usize, usize),
    pub groups: usize,
}

impl Conv2d {
    pub fn new(in_ch: usize, out_ch: usize, kernel_h: usize, kernel_w: usize,
               stride: usize, padding: usize) -> Self {
        let w = Tensor::from_vec(
            vec![0.0f32; out_ch * in_ch * kernel_h * kernel_w],
            vec![out_ch, in_ch, kernel_h, kernel_w],
        );
        let b = Some(Tensor::from_vec(vec![0.0f32; out_ch], vec![out_ch]));
        Conv2d { weight: w, bias: b, stride: (stride, stride), padding: (padding, padding), dilation: (1, 1), groups: 1 }
    }
}

impl Module for Conv2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor, ModuleError> {
        // input: [batch, in_channels, H, W]
        if input.ndim() != 4 {
            return Err(ModuleError::DimError("Conv2d expects 4D input [N,C,H,W]".into()));
        }
        let (n, _ic, h, w) = (input.dims()[0], input.dims()[1], input.dims()[2], input.dims()[3]);
        let (kh, kw) = (self.weight.dims()[2], self.weight.dims()[3]);
        let out_h = (h + 2 * self.padding.0 - kh) / self.stride.0 + 1;
        let out_w = (w + 2 * self.padding.1 - kw) / self.stride.1 + 1;
        let out_ch = self.weight.dims()[0];
        let out = Tensor::from_vec(vec![0.0f32; n * out_ch * out_h * out_w], vec![n, out_ch, out_h, out_w]);
        Ok(out) // Simplified; full im2col in v2
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = vec![Parameter::new("weight", self.weight.clone())];
        if let Some(ref b) = self.bias { p.push(Parameter::new("bias", b.clone())); }
        p
    }

    fn name(&self) -> &str { "Conv2d" }
}

/// Transposed 2D convolution (deconvolution).
pub struct ConvTranspose2d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

impl ConvTranspose2d {
    pub fn new(in_ch: usize, out_ch: usize, kh: usize, kw: usize, stride: usize, padding: usize) -> Self {
        let w = Tensor::from_vec(vec![0.0f32; in_ch * out_ch * kh * kw], vec![in_ch, out_ch, kh, kw]);
        let b = Some(Tensor::from_vec(vec![0.0f32; out_ch], vec![out_ch]));
        ConvTranspose2d { weight: w, bias: b, stride: (stride, stride), padding: (padding, padding) }
    }
}

impl Module for ConvTranspose2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor, ModuleError> {
        // Simplified stub
        let batch = input.dims()[0];
        let out_ch = self.weight.dims()[1];
        Ok(Tensor::from_vec(vec![0.0f32; batch * out_ch], vec![batch, out_ch]))
    }
    fn name(&self) -> &str { "ConvTranspose2d" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv1d_shape() {
        let conv = Conv1d::new(3, 16, 3, 1, 1);
        let x = Tensor::from_vec(vec![0.0f32; 2*3*32], vec![2, 3, 32]);
        let y = conv.forward(&x).unwrap();
        assert_eq!(y.ndim(), 3);
    }

    #[test]
    fn test_conv2d_shape() {
        let conv = Conv2d::new(3, 16, 3, 3, 1, 1);
        let x = Tensor::from_vec(vec![0.0f32; 2*3*8*8], vec![2, 3, 8, 8]);
        let y = conv.forward(&x).unwrap();
        assert_eq!(y.ndim(), 4);
        assert_eq!(y.dims()[1], 16);
    }
}
