//! Zero-copy strided views over tensor storage.

use crate::tensor::storage::Storage;
use crate::tensor::dtype::{DType, Element};

/// A view into tensor data without ownership. Zero-copy via offsets and strides.
#[derive(Clone, Debug)]
pub struct TensorView {
    storage: Storage,
    offset: usize,
    shape: Vec<usize>,
    strides: Vec<isize>,
}

impl TensorView {
    /// Create a view over the full storage.
    pub fn full(storage: Storage, shape: Vec<usize>) -> Self {
        let strides = compute_strides(&shape);
        TensorView {
            storage,
            offset: 0,
            shape,
            strides,
        }
    }

    /// Create a view with explicit offset, shape, and strides.
    pub fn new(storage: Storage, offset: usize, shape: Vec<usize>, strides: Vec<isize>) -> Self {
        TensorView { storage, offset, shape, strides }
    }

    /// Shape of the view.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Strides of the view.
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Offset in elements.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Access the underlying storage.
    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    /// Is this view contiguous in memory?
    pub fn is_contiguous(&self) -> bool {
        let expected = compute_strides(&self.shape);
        self.strides == expected
    }

    /// Get a single element (slow path — for debugging/small tensors).
    pub fn get<T: Element>(&self, indices: &[usize]) -> T {
        let flat = self.flat_index(indices);
        let data: &[T] = self.storage.as_slice();
        data[self.offset + flat]
    }

    /// Compute flat index from multi-dimensional indices.
    fn flat_index(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.shape.len(), "index dimension mismatch");
        let mut idx: isize = 0;
        for ((&i, &s), &dim) in indices.iter().zip(self.strides.iter()).zip(self.shape.iter()) {
            assert!(i < dim, "index {} out of bounds for dim {}", i, dim);
            idx += (i as isize) * s;
        }
        idx as usize
    }

    /// Slice along a dimension: `start..end` for `dim`.
    pub fn slice_dim(&self, dim: usize, start: usize, end: usize) -> Self {
        assert!(dim < self.shape.len());
        assert!(start <= end && end <= self.shape[dim]);
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        shape[dim] = end - start;
        let new_offset = self.offset + start * self.strides[dim] as usize;
        TensorView {
            storage: self.storage.clone(),
            offset: new_offset,
            shape,
            strides,
        }
    }
}

/// Compute row-major strides from shape.
pub fn compute_strides(shape: &[usize]) -> Vec<isize> {
    let ndim = shape.len();
    if ndim == 0 { return vec![]; }
    let mut strides = vec![1isize; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as isize;
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strides() {
        let strides = compute_strides(&[2, 3, 4]);
        assert_eq!(strides, vec![12, 4, 1]);
    }

    #[test]
    fn test_contiguous() {
        let storage = Storage::zeros(DType::F32, 24);
        let view = TensorView::full(storage, vec![2, 3, 4]);
        assert!(view.is_contiguous());
    }
}
