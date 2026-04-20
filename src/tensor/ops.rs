//! Element-wise and binary tensor operations.

use crate::tensor::tensor::Tensor;
use crate::tensor::dtype::Element;
use rayon::prelude::*;

/// Element-wise addition.
pub fn add(a: &Tensor, b: &Tensor) -> Tensor { binary_op(a, b, |x, y| x + y) }

/// Element-wise subtraction.
pub fn sub(a: &Tensor, b: &Tensor) -> Tensor { binary_op(a, b, |x, y| x - y) }

/// Element-wise multiplication.
pub fn mul(a: &Tensor, b: &Tensor) -> Tensor { binary_op(a, b, |x, y| x * y) }

/// Element-wise division.
pub fn div(a: &Tensor, b: &Tensor) -> Tensor { binary_op(a, b, |x, y| x / y) }

/// Element-wise negation.
pub fn neg(a: &Tensor) -> Tensor { unary_op(a, |x| -x) }

/// Element-wise exponent.
pub fn exp(a: &Tensor) -> Tensor { unary_op(a, |x| x.exp()) }

/// Element-wise natural log.
pub fn ln(a: &Tensor) -> Tensor { unary_op(a, |x| x.ln()) }

/// Element-wise square root.
pub fn sqrt(a: &Tensor) -> Tensor { unary_op(a, |x| x.sqrt()) }

/// Element-wise ReLU.
pub fn relu(a: &Tensor) -> Tensor { unary_op(a, |x| if x < 0.0 { 0.0 } else { x }) }

/// Element-wise sigmoid.
pub fn sigmoid(a: &Tensor) -> Tensor { unary_op(a, |x| 1.0 / (1.0 + (-x).exp())) }

/// Element-wise tanh.
pub fn tanh(a: &Tensor) -> Tensor { unary_op(a, |x| x.tanh()) }

/// Element-wise absolute value.
pub fn abs(a: &Tensor) -> Tensor { unary_op(a, |x| x.abs()) }

/// Add a scalar to every element.
pub fn add_scalar(a: &Tensor, s: f32) -> Tensor { unary_op(a, |x| x + s) }

/// Multiply every element by a scalar.
pub fn mul_scalar(a: &Tensor, s: f32) -> Tensor { unary_op(a, |x| x * s) }

/// GELU activation (approximate).
pub fn gelu(a: &Tensor) -> Tensor {
    unary_op(a, |x| 0.5 * x * (1.0 + (0.7978845688 * (x + 0.044715 * x * x * x)).tanh()))
}

/// LeakyReLU activation.
pub fn leaky_relu(a: &Tensor, negative_slope: f32) -> Tensor {
    unary_op(a, |x| if x > 0.0 { x } else { negative_slope * x })
}

/// ReLU mask: returns 1 where x > 0, 0 elsewhere (for backward pass).
pub fn relu_mask(a: &Tensor) -> Tensor {
    unary_op(a, |x| if x > 0.0 { 1.0 } else { 0.0 })
}

/// Softmax along the given axis (simplified: last dim).
pub fn softmax(a: &Tensor, _axis: isize) -> Tensor {
    let a = a.contiguous();
    let data = a.as_slice::<f32>();
    let ndim = a.ndim();
    let last_dim = a.dims().last().copied().unwrap_or(1);
    let outer: usize = if ndim > 1 { a.dims()[..ndim - 1].iter().product() } else { 1 };
    let mut out = Vec::with_capacity(a.numel());
    for i in 0..outer {
        let off = i * last_dim;
        let row = &data[off..off + last_dim];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        out.extend(exps.iter().map(|&x| x / sum));
    }
    Tensor::from_vec(out, a.dims().to_vec())
}

/// Apply causal mask: set upper triangle to -inf.
pub fn apply_causal_mask(scores: &Tensor) -> Tensor {
    let scores = scores.contiguous();
    let data = scores.as_slice::<f32>();
    let ndim = scores.ndim();
    let seq_q = scores.dims()[ndim - 2];
    let seq_k = scores.dims()[ndim - 1];
    let batch: usize = if ndim > 2 { scores.dims()[..ndim - 2].iter().product() } else { 1 };
    let mut out = data.to_vec();
    for b in 0..batch {
        for i in 0..seq_q {
            for j in 0..seq_k {
                if j > i {
                    out[b * seq_q * seq_k + i * seq_k + j] = f32::NEG_INFINITY;
                }
            }
        }
    }
    Tensor::from_vec(out, scores.dims().to_vec())
}

/// Mean reduction over the given axis.
pub fn mean(a: &Tensor, axis: usize) -> Tensor {
    crate::tensor::reduce::mean(a, axis)
}

/// Gradient seed helper (returns ones matching input shape).
pub fn mul_with_grad_seed(a: &Tensor) -> Tensor {
    Tensor::from_vec(vec![1.0f32; a.numel()], a.dims().to_vec())
}

/// Element-wise clip: clamp values to [min, max].
pub fn clip(a: &Tensor, min_val: f32, max_val: f32) -> Tensor {
    unary_op(a, |x| if x < min_val { min_val } else if x > max_val { max_val } else { x })
}

/// Element-wise power (integer exponent).
pub fn powi(a: &Tensor, exp: i32) -> Tensor { unary_op(a, |x| x.powi(exp)) }

/// Matrix multiplication for 2D tensors.
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let a = a.contiguous();
    let b = b.contiguous();
    let a_dims = a.dims();
    let b_dims = b.dims();
    assert!(a_dims.len() >= 2, "matmul requires >=2D tensors");
    assert!(b_dims.len() >= 2, "matmul requires >=2D tensors");
    let m = a_dims[a_dims.len() - 2];
    let k_a = a_dims[a_dims.len() - 1];
    let k_b = b_dims[b_dims.len() - 2];
    let n = b_dims[b_dims.len() - 1];
    assert_eq!(k_a, k_b, "matmul inner dims mismatch");
    let a_data = a.as_slice::<f32>();
    let b_data = b.as_slice::<f32>();
    let mut c_data = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k_a {
                sum += a_data[i * k_a + kk] * b_data[kk * n + j];
            }
            c_data[i * n + j] = sum;
        }
    }
    let mut out_shape = a_dims[..a_dims.len() - 2].to_vec();
    out_shape.push(m);
    out_shape.push(n);
    Tensor::from_vec(c_data, out_shape)
}

// --- Internal helpers ---

fn binary_op(a: &Tensor, b: &Tensor, op: impl Fn(f32, f32) -> f32 + Sync) -> Tensor {
    let a = a.contiguous();
    let b = b.contiguous();
    // Simple broadcasting: expand b to match a's shape if possible
    let (a_data, b_data, out_shape) = if a.dims() == b.dims() {
        (a.as_slice::<f32>().to_vec(), b.as_slice::<f32>().to_vec(), a.dims().to_vec())
    } else {
        // Try broadcasting b to a's shape
        let b_expanded = broadcast_to(&b, a.dims());
        (a.as_slice::<f32>().to_vec(), b_expanded.as_slice::<f32>().to_vec(), a.dims().to_vec())
    };
    let out: Vec<f32> = a_data.par_iter().enumerate()
        .map(|(i, &x)| op(x, b_data[i]))
        .collect();
    Tensor::from_vec(out, out_shape)
}

/// Simple broadcast: expand tensor to target shape by repeating along size-1 dims.
fn broadcast_to(t: &Tensor, target: &[usize]) -> Tensor {
    let t = t.contiguous();
    let src = t.dims();
    if src == target { return t.clone(); }
    let src_data = t.as_slice::<f32>();
    let numel: usize = target.iter().product();
    let mut out = vec![0.0f32; numel];
    // Compute broadcast strides for src
    let ndim_target = target.len();
    let ndim_src = src.len();
    let mut src_strides = vec![0usize; ndim_target];
    for i in 0..ndim_src {
        let target_idx = ndim_target - ndim_src + i;
        if src[i] == target[target_idx] {
            let s: usize = src[i+1..].iter().product::<usize>().max(1);
            src_strides[target_idx] = s;
        } else if src[i] == 1 {
            src_strides[target_idx] = 0; // broadcast this dim
        }
    }
    // Compute target strides
    let mut target_strides = vec![1usize; ndim_target];
    for i in (0..ndim_target.saturating_sub(1)).rev() {
        target_strides[i] = target_strides[i + 1] * target[i + 1];
    }
    for flat_idx in 0..numel {
        let mut src_flat = 0usize;
        let mut rem = flat_idx;
        for d in 0..ndim_target {
            let coord = rem / target_strides[d];
            rem %= target_strides[d];
            src_flat += coord * src_strides[d];
        }
        out[flat_idx] = src_data[src_flat.min(src_data.len() - 1)];
    }
    Tensor::from_vec(out, target.to_vec())
}

fn unary_op(a: &Tensor, op: impl Fn(f32) -> f32 + Sync) -> Tensor {
    let a = a.contiguous();
    let a_data = a.as_slice::<f32>();
    let out: Vec<f32> = a_data.par_iter().map(|&x| op(x)).collect();
    Tensor::from_vec(out, a.dims().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
        let b = Tensor::from_vec(vec![4.0f32, 5.0, 6.0], vec![3]);
        let c = add(&a, &b);
        let data = c.as_slice::<f32>();
        assert_eq!(data, &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_exp() {
        let a = Tensor::from_vec(vec![0.0f32, 1.0], vec![2]);
        let b = exp(&a);
        let data = b.as_slice::<f32>();
        assert!((data[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_matmul() {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = matmul(&a, &b);
        let data = c.as_slice::<f32>();
        assert_eq!(data, &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_add_scalar() {
        let a = Tensor::from_vec(vec![1.0f32, 2.0], vec![2]);
        let c = add_scalar(&a, 10.0);
        assert_eq!(c.as_slice::<f32>(), &[11.0, 12.0]);
    }

    #[test]
    fn test_softmax() {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
        let s = softmax(&a, -1);
        let data = s.as_slice::<f32>();
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
