//! Tensor integration tests.

use flame_tensor::{Tensor, DType, Shape};

#[test]
fn test_tensor_create_and_shape() {
    let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    assert_eq!(t.ndim(), 2);
    assert_eq!(t.dims(), &[2, 3]);
    assert_eq!(t.numel(), 6);
    assert!(t.is_contiguous());
}

#[test]
fn test_scalar_tensor() {
    let t = Tensor::scalar(42.0f32);
    assert!(t.shape().is_scalar());
    assert_eq!(t.numel(), 1);
}

#[test]
fn test_reshape() {
    let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let r = t.reshape(vec![3, 2]);
    assert_eq!(r.dims(), &[3, 2]);
}

#[test]
fn test_transpose() {
    let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let tt = t.transpose(0, 1);
    assert_eq!(tt.dims(), &[3, 2]);
}

#[test]
fn test_add() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
    let b = Tensor::from_vec(vec![4.0f32, 5.0, 6.0], vec![3]);
    let c = flame_tensor::tensor::ops::add(&a, &b);
    let data = c.as_slice::<f32>();
    assert_eq!(data[0], 5.0);
    assert_eq!(data[2], 9.0);
}

#[test]
fn test_mul() {
    let a = Tensor::from_vec(vec![2.0f32, 3.0], vec![2]);
    let b = Tensor::from_vec(vec![4.0f32, 5.0], vec![2]);
    let c = flame_tensor::tensor::ops::mul(&a, &b);
    let data = c.as_slice::<f32>();
    assert_eq!(data[0], 8.0);
    assert_eq!(data[1], 15.0);
}

#[test]
fn test_matmul() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]);
    let c = flame_tensor::tensor::ops::matmul(&a, &b);
    assert_eq!(c.dims(), &[2, 2]);
}

#[test]
fn test_dtype() {
    let t = Tensor::from_vec(vec![1.0f32], vec![1]);
    assert_eq!(t.dtype(), DType::F32);
}

#[test]
fn test_permute() {
    let t = Tensor::from_vec(vec![1.0f32; 24], vec![2, 3, 4]);
    let p = t.permute(&[2, 0, 1]);
    assert_eq!(p.dims(), &[4, 2, 3]);
}
