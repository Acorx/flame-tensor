//! Reduction operations: sum, mean, max, min, argmax, argmin over axes.

use crate::tensor::tensor::Tensor;
use rayon::prelude::*;

/// Sum over all elements.
pub fn sum_all(a: &Tensor) -> Tensor {
    let a = a.contiguous();
    let data = a.as_slice::<f32>();
    let total: f32 = data.par_iter().sum();
    Tensor::scalar(total)
}

/// Sum along an axis.
pub fn sum(a: &Tensor, axis: usize) -> Tensor {
    let dims = a.dims();
    assert!(axis < dims.len(), "axis {} out of bounds for ndim {}", axis, dims.len());
    let a = a.contiguous();
    let data = a.as_slice::<f32>();
    let out_shape: Vec<usize> = dims.iter().enumerate().map(|(i, &d)| if i == axis { 1 } else { d }).collect();
    let outer: usize = dims[..axis].iter().product::<usize>().max(1);
    let inner: usize = dims[axis + 1..].iter().product::<usize>().max(1);
    let dim = dims[axis];
    let mut out = vec![0.0f32; outer * inner];
    for o in 0..outer {
        for i in 0..inner {
            let mut s = 0.0f32;
            for d in 0..dim {
                let idx = o * dim * inner + d * inner + i;
                s += data[idx];
            }
            out[o * inner + i] = s;
        }
    }
    Tensor::from_vec(out, out_shape)
}

/// Mean over all elements.
pub fn mean_all(a: &Tensor) -> Tensor {
    let s = sum_all(a);
    let sum_val = s.as_slice::<f32>()[0];
    Tensor::scalar(sum_val / a.numel() as f32)
}

/// Mean along an axis.
pub fn mean(a: &Tensor, axis: usize) -> Tensor {
    let dims = a.dims();
    let n = dims[axis] as f32;
    let summed = sum(a, axis);
    let data = summed.as_slice::<f32>();
    let out: Vec<f32> = data.iter().map(|&x| x / n).collect();
    let mut shape = dims.to_vec();
    shape[axis] = 1;
    Tensor::from_vec(out, shape)
}

/// Max over all elements.
pub fn max_all(a: &Tensor) -> Tensor {
    let a = a.contiguous();
    let data = a.as_slice::<f32>();
    let max_val = data.par_iter().copied().reduce(|| f32::NEG_INFINITY, f32::max);
    Tensor::scalar(max_val)
}

/// Min over all elements.
pub fn min_all(a: &Tensor) -> Tensor {
    let a = a.contiguous();
    let data = a.as_slice::<f32>();
    let min_val = data.par_iter().copied().reduce(|| f32::INFINITY, f32::min);
    Tensor::scalar(min_val)
}

/// Argmax: index of max value along an axis.
pub fn argmax(a: &Tensor, axis: usize) -> Tensor {
    let dims = a.dims();
    assert!(axis < dims.len());
    let a = a.contiguous();
    let data = a.as_slice::<f32>();
    let mut out_shape: Vec<usize> = dims.to_vec();
    out_shape[axis] = 1;
    let outer: usize = dims[..axis].iter().product::<usize>().max(1);
    let inner: usize = dims[axis + 1..].iter().product::<usize>().max(1);
    let dim = dims[axis];
    let mut out = vec![0i64; outer * inner];
    for o in 0..outer {
        for i in 0..inner {
            let mut best = 0usize;
            let mut best_val = data[o * dim * inner + 0 * inner + i];
            for d in 1..dim {
                let val = data[o * dim * inner + d * inner + i];
                if val > best_val { best_val = val; best = d; }
            }
            out[o * inner + i] = best as i64;
        }
    }
    Tensor::from_vec(out, out_shape)
}

/// Argmin: index of min value along an axis.
pub fn argmin(a: &Tensor, axis: usize) -> Tensor {
    let dims = a.dims();
    assert!(axis < dims.len());
    let a = a.contiguous();
    let data = a.as_slice::<f32>();
    let mut out_shape: Vec<usize> = dims.to_vec();
    out_shape[axis] = 1;
    let outer: usize = dims[..axis].iter().product::<usize>().max(1);
    let inner: usize = dims[axis + 1..].iter().product::<usize>().max(1);
    let dim = dims[axis];
    let mut out = vec![0i64; outer * inner];
    for o in 0..outer {
        for i in 0..inner {
            let mut best = 0usize;
            let mut best_val = data[o * dim * inner + 0 * inner + i];
            for d in 1..dim {
                let val = data[o * dim * inner + d * inner + i];
                if val < best_val { best_val = val; best = d; }
            }
            out[o * inner + i] = best as i64;
        }
    }
    Tensor::from_vec(out, out_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_all() {
        let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let s = sum_all(&t);
        assert!((s.as_slice::<f32>()[0] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_sum_axis() {
        let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let s = sum(&t, 1);
        let data = s.as_slice::<f32>();
        assert!((data[0] - 6.0).abs() < 1e-5);
        assert!((data[1] - 15.0).abs() < 1e-5);
    }

    #[test]
    fn test_mean_axis() {
        let t = Tensor::from_vec(vec![2.0f32, 4.0, 6.0], vec![3]);
        let m = mean(&t, 0);
        assert!((m.as_slice::<f32>()[0] - 4.0).abs() < 1e-5);
    }
}
