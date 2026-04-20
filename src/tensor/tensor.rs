//! Core Tensor struct: shape, strides, storage, dtype.

use crate::tensor::storage::Storage;
use crate::tensor::view::TensorView;
use crate::tensor::dtype::{DType, Element};
use crate::tensor::broadcast::broadcast_shapes;
use std::fmt;

/// Shape descriptor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape(Vec<usize>);

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self { Shape(dims) }
    pub fn dims(&self) -> &[usize] { &self.0 }
    pub fn ndim(&self) -> usize { self.0.len() }
    pub fn numel(&self) -> usize { self.0.iter().product::<usize>().max(1) }
    pub fn is_scalar(&self) -> bool { self.0.is_empty() }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self { Shape(dims) }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.0.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{}", d)?;
        }
        write!(f, "]")
    }
}

/// The core tensor — a multi-dimensional array with COW storage.
pub struct Tensor {
    pub storage: Storage,
    pub shape: Shape,
    pub strides: Vec<isize>,
    pub offset: usize,
}

impl Tensor {
    /// Create a tensor from a flat Vec and shape.
    pub fn from_vec<T: Element>(data: Vec<T>, shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(data.len(), expected, "data length {} != shape numel {}", data.len(), expected);
        let storage = Storage::from_slice(&data);
        let strides = crate::tensor::view::compute_strides(&shape);
        Tensor { storage, shape: Shape(shape), strides, offset: 0 }
    }

    /// Create a scalar tensor.
    pub fn scalar<T: Element>(v: T) -> Self {
        Tensor::from_vec(vec![v], vec![])
    }

    /// Create from raw storage + shape (zero-copy).
    pub fn from_storage(storage: Storage, shape: Vec<usize>) -> Self {
        let strides = crate::tensor::view::compute_strides(&shape);
        Tensor { storage, shape: Shape(shape), strides, offset: 0 }
    }

    /// Shape.
    pub fn shape(&self) -> &Shape { &self.shape }

    /// Raw dims.
    pub fn dims(&self) -> &[usize] { self.shape.dims() }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize { self.shape.ndim() }

    /// Total elements.
    pub fn numel(&self) -> usize { self.shape.numel() }

    /// Data type.
    pub fn dtype(&self) -> DType { self.storage.dtype() }

    /// Strides.
    pub fn strides(&self) -> &[isize] { &self.strides }

    /// Storage.
    pub fn storage(&self) -> &Storage { &self.storage }

    /// Mutable storage (COW).
    pub fn storage_mut(&mut self) -> &mut Storage { &mut self.storage }

    /// Is this contiguous?
    pub fn is_contiguous(&self) -> bool {
        let expected = crate::tensor::view::compute_strides(self.shape.dims());
        self.strides == expected
    }

    /// Get a view for this tensor.
    pub fn view(&self) -> TensorView {
        TensorView::new(self.storage.clone(), self.offset, self.shape.dims().to_vec(), self.strides.clone())
    }

    /// Access raw typed data (contiguous only).
    pub fn as_slice<T: Element>(&self) -> &[T] {
        assert!(self.is_contiguous(), "as_slice requires contiguous tensor");
        assert_eq!(T::DTYPE, self.dtype());
        self.storage.as_slice()
    }

    /// Mutable typed data (COW).
    pub fn as_mut_slice<T: Element>(&mut self) -> &mut [T] {
        assert!(self.is_contiguous(), "as_mut_slice requires contiguous tensor");
        assert_eq!(T::DTYPE, self.dtype());
        self.storage.as_mut_slice()
    }

    /// Reshape (contiguous only).
    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        assert!(self.is_contiguous(), "reshape requires contiguous tensor");
        let new_numel: usize = new_shape.iter().product::<usize>().max(1);
        assert_eq!(self.numel(), new_numel, "reshape numel mismatch: {} vs {}", self.numel(), new_numel);
        let strides = crate::tensor::view::compute_strides(&new_shape);
        Tensor {
            storage: self.storage.clone(),
            shape: Shape(new_shape),
            strides,
            offset: self.offset,
        }
    }

    /// Broadcast to target shape.
    pub fn broadcast_to(&self, target: &[usize]) -> Self {
        let _ = broadcast_shapes(self.dims(), target);
        // For broadcasting, we keep same storage but change shape/strides
        let mut strides = vec![0isize; target.len()];
        let off = target.len() - self.ndim();
        for (i, (&dim, &s)) in self.dims().iter().zip(self.strides.iter()).enumerate() {
            if dim == 1 {
                strides[off + i] = 0;
            } else {
                strides[off + i] = s;
            }
        }
        Tensor {
            storage: self.storage.clone(),
            shape: Shape(target.to_vec()),
            strides,
            offset: self.offset,
        }
    }

    /// Transpose (swap two dims).
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Self {
        let mut shape = self.shape.dims().to_vec();
        let mut strides = self.strides.clone();
        shape.swap(dim0, dim1);
        strides.swap(dim0, dim1);
        Tensor {
            storage: self.storage.clone(),
            shape: Shape(shape),
            strides,
            offset: self.offset,
        }
    }

    /// Permute dimensions.
    pub fn permute(&self, dims: &[usize]) -> Self {
        assert_eq!(dims.len(), self.ndim(), "permute dims length mismatch");
        let shape: Vec<usize> = dims.iter().map(|&d| self.shape.dims()[d]).collect();
        let strides: Vec<isize> = dims.iter().map(|&d| self.strides[d]).collect();
        Tensor {
            storage: self.storage.clone(),
            shape: Shape(shape),
            strides,
            offset: self.offset,
        }
    }

    /// Make contiguous copy if needed.
    pub fn contiguous(&self) -> Self {
        if self.is_contiguous() {
            return self.clone();
        }
        // Gather elements into contiguous layout
        let mut data: Vec<u8> = vec![0u8; self.numel() * self.dtype().size_of()];
        let src = self.storage.as_bytes();
        let elem_size = self.dtype().size_of();
        for i in 0..self.numel() {
            let multi_idx = flat_to_multi(i, self.shape.dims());
            let flat = self.offset + multi_to_flat(&multi_idx, &self.strides);
            let src_off = flat * elem_size;
            let dst_off = i * elem_size;
            data[dst_off..dst_off + elem_size].copy_from_slice(&src[src_off..src_off + elem_size]);
        }
        let new_storage = Storage { data: std::sync::Arc::new(data), dtype: self.dtype(), len: self.numel() };
        let strides = crate::tensor::view::compute_strides(self.shape.dims());
        Tensor { storage: new_storage, shape: self.shape.clone(), strides, offset: 0 }
    }
}

fn flat_to_multi(flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut idx = Vec::with_capacity(shape.len());
    let mut remaining = flat;
    for i in 0..shape.len() {
        let stride: usize = shape[i + 1..].iter().product::<usize>().max(1);
        idx.push(remaining / stride);
        remaining %= stride;
    }
    idx
}

fn multi_to_flat(indices: &[usize], strides: &[isize]) -> usize {
    indices.iter().zip(strides.iter()).map(|(&i, &s)| (i as isize * s) as usize).sum()
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Tensor {
            storage: self.storage.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
        }
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype())
            .field("contiguous", &self.is_contiguous())
            .finish()
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor({}, dtype={})", self.shape, self.dtype())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_vec() {
        let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.numel(), 6);
        assert!(t.is_contiguous());
    }

    #[test]
    fn test_reshape() {
        let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let t2 = t.reshape(vec![3, 2]);
        assert_eq!(t2.dims(), &[3, 2]);
        assert_eq!(t2.numel(), 6);
    }

    #[test]
    fn test_transpose() {
        let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let tt = t.transpose(0, 1);
        assert_eq!(tt.dims(), &[3, 2]);
        assert!(!tt.is_contiguous());
    }

    #[test]
    fn test_scalar() {
        let t = Tensor::scalar(42.0f32);
        assert!(t.shape().is_scalar());
        assert_eq!(t.numel(), 1);
    }
}
