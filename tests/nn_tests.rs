//! Neural network module integration tests.

use flame_tensor::Tensor;
use flame_tensor::nn::module::Module;
use flame_tensor::nn::linear::Linear;
use flame_tensor::nn::activation::{ReLU, Sigmoid, GELU};
use flame_tensor::nn::norm::LayerNorm;
use flame_tensor::nn::dropout::Dropout;
use flame_tensor::nn::embedding::Embedding;
use flame_tensor::nn::sequential::Sequential;
use flame_tensor::nn::conv::Conv2d;
use flame_tensor::nn::attention::MultiHeadSelfAttention;
use flame_tensor::nn::transformer::TransformerBlock;

#[test]
fn test_linear_forward() {
    let lin = Linear::new(8, 4, true);
    let x = Tensor::from_vec(vec![1.0f32; 8], vec![1, 8]);
    let y = lin.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 4]);
}

#[test]
fn test_relu_forward() {
    let x = Tensor::from_vec(vec![-1.0f32, 0.0, 1.0, 2.0], vec![4]);
    let y = ReLU.forward(&x).unwrap();
    let data = y.as_slice::<f32>();
    assert_eq!(data[0], 0.0);
    assert_eq!(data[3], 2.0);
}

#[test]
fn test_sigmoid_forward() {
    let x = Tensor::from_vec(vec![0.0f32], vec![1]);
    let y = Sigmoid.forward(&x).unwrap();
    let data = y.as_slice::<f32>();
    assert!((data[0] - 0.5).abs() < 1e-4);
}

#[test]
fn test_layernorm_forward() {
    let ln = LayerNorm::new(vec![4], 1e-5);
    let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![1, 4]);
    let y = ln.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 4]);
}

#[test]
fn test_dropout_eval() {
    let mut d = Dropout::new(0.5);
    d.eval();
    let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
    let y = d.forward(&x).unwrap();
    assert_eq!(y.as_slice::<f32>()[0], 1.0);
}

#[test]
fn test_embedding() {
    let emb = Embedding::new(100, 32);
    let x = Tensor::from_vec(vec![0.0f32, 5.0, 10.0], vec![3]);
    let y = emb.forward(&x).unwrap();
    assert_eq!(y.dims(), &[3, 32]);
}

#[test]
fn test_sequential() {
    let seq = Sequential::new()
        .add(Linear::new(4, 8, true))
        .add(ReLU)
        .add(Linear::new(8, 2, true));
    let x = Tensor::from_vec(vec![1.0f32; 4], vec![1, 4]);
    let y = seq.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 2]);
}

#[test]
fn test_conv2d_shape() {
    let conv = Conv2d::new(3, 16, 3, 3, 1, 1);
    let x = Tensor::from_vec(vec![0.0f32; 1*3*8*8], vec![1, 3, 8, 8]);
    let y = conv.forward(&x).unwrap();
    assert_eq!(y.ndim(), 4);
    assert_eq!(y.dims()[1], 16);
}

#[test]
fn test_attention_shape() {
    let attn = MultiHeadSelfAttention::new(64, 4, true);
    let x = Tensor::from_vec(vec![0.0f32; 2*8*64], vec![2, 8, 64]);
    let y = attn.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 8, 64]);
}

#[test]
fn test_transformer_block() {
    let block = TransformerBlock::new(64, 4, 256, true, 0.1);
    let x = Tensor::from_vec(vec![0.0f32; 2*8*64], vec![2, 8, 64]);
    let y = block.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 8, 64]);
}
