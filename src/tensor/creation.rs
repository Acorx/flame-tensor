//! Tensor creation utilities: zeros, ones, randn, arange, etc.

use crate::tensor::tensor::Tensor;
use crate::tensor::storage::Storage;
use crate::tensor::dtype::{DType, Element};

/// Create a tensor of zeros.
pub fn zeros<T: Element>(shape: Vec<usize>) -> Tensor {
    let numel: usize = shape.iter().product::<usize>().max(1);
    let storage = Storage::zeros(T::DTYPE, numel);
    Tensor::from_storage(storage, shape)
}

/// Create a tensor of ones.
pub fn ones<T: Element>(shape: Vec<usize>) -> Tensor {
    let numel: usize = shape.iter().product::<usize>().max(1);
    let mut storage = Storage::zeros(T::DTYPE, numel);
    let data = storage.as_mut_slice::<T>();
    data.iter_mut().for_each(|v| *v = T::one_val());
    Tensor::from_storage(storage, shape)
}

/// Create a tensor filled with a specific value.
pub fn full<T: Element>(shape: Vec<usize>, value: T) -> Tensor {
    let numel: usize = shape.iter().product::<usize>().max(1);
    let mut storage = Storage::zeros(T::DTYPE, numel);
    let data = storage.as_mut_slice::<T>();
    data.iter_mut().for_each(|v| *v = value);
    Tensor::from_storage(storage, shape)
}

/// Identity matrix.
pub fn eye<T: Element>(n: usize) -> Tensor {
    let mut data = vec![T::zero_val(); n * n];
    for i in 0..n {
        data[i * n + i] = T::one_val();
    }
    Tensor::from_vec(data, vec![n, n])
}

/// Range of values [start, end) with step (f32 only for simplicity).
pub fn arange(start: f32, end: f32, step: f32) -> Tensor {
    let mut data = Vec::new();
    let mut val = start;
    while val < end {
        data.push(val);
        val += step;
    }
    let len = data.len();
    Tensor::from_vec(data, vec![len])
}

/// Linearly spaced values (f32).
pub fn linspace(start: f32, end: f32, steps: usize) -> Tensor {
    if steps == 0 { return Tensor::from_vec(Vec::<f32>::new(), vec![0]); }
    if steps == 1 { return Tensor::from_vec(vec![start], vec![1]); }
    let step = (end - start) / (steps - 1) as f32;
    let mut data = Vec::with_capacity(steps);
    for i in 0..steps {
        data.push(start + step * i as f32);
    }
    Tensor::from_vec(data, vec![steps])
}

/// Random uniform [0, 1) using a simple LCG PRNG (no external rand dependency).
pub fn rand<T: Element>(shape: Vec<usize>) -> Tensor {
    let numel: usize = shape.iter().product::<usize>().max(1);
    let mut state: u64 = 0x853c49e6748fea9b;
    let mut data = Vec::with_capacity(numel);
    for _ in 0..numel {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u = state;
        let f = (u >> 40) as f32 / (1u32 << 24) as f32; // [0, 1)
        data.push(T::from(f).unwrap());
    }
    Tensor::from_vec(data, shape)
}

/// Random normal distribution (Box-Muller) using LCG.
pub fn randn<T: Element>(shape: Vec<usize>) -> Tensor {
    let numel: usize = shape.iter().product::<usize>().max(1);
    let mut state: u64 = 0x1234567890abcdef;
    let mut data = Vec::with_capacity(numel);

    let mut lcg = || -> f32 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u32 << 24) as f32
    };

    for _ in (0..numel).step_by(2) {
        let u1 = lcg().max(1e-10);
        let u2 = lcg();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        data.push(T::from(z0).unwrap());
        if data.len() < numel {
            let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).sin();
            data.push(T::from(z1).unwrap());
        }
    }
    data.truncate(numel);
    Tensor::from_vec(data, shape)
}

/// Trait for tensor creation methods on existing tensors.
pub trait CreationOps {
    fn zeros<T: Element>(shape: Vec<usize>) -> Tensor;
    fn ones<T: Element>(shape: Vec<usize>) -> Tensor;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let t = zeros::<f32>(vec![2, 3]);
        assert_eq!(t.numel(), 6);
        assert!(t.as_slice::<f32>().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ones() {
        let t = ones::<f32>(vec![3]);
        assert!(t.as_slice::<f32>().iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_eye() {
        let t = eye::<f32>(3);
        let data = t.as_slice::<f32>();
        assert_eq!(data[0], 1.0);
        assert_eq!(data[4], 1.0);
        assert_eq!(data[8], 1.0);
        assert_eq!(data[1], 0.0);
    }

    #[test]
    fn test_arange() {
        let t = arange(0.0f32, 5.0, 1.0);
        assert_eq!(t.as_slice::<f32>(), &[0.0, 1.0, 2.0, 3.0, 4.0]);
    }
    #[test]
    fn test_randn() {
        let t = randn::<f32>(vec![100]);
        assert_eq!(t.numel(), 100);
    }
}
