//! COW (Copy-On-Write) storage for tensors.
//!
//! Uses `Arc<Vec<u8>>` for shared byte storage. Mutation clones only when refcount > 1.

use std::sync::Arc;
use crate::tensor::dtype::DType;

/// Reference-counted byte storage with copy-on-write semantics.
#[derive(Clone)]
pub struct Storage {
    pub data: Arc<Vec<u8>>,
    pub dtype: DType,
    pub len: usize,
}

impl Storage {
    /// Create new storage from a typed slice.
    pub fn from_slice<T: crate::tensor::dtype::Element>(data: &[T]) -> Self {
        let bytes = bytemuck::cast_slice::<T, u8>(data).to_vec();
        let len = data.len();
        Storage { data: Arc::new(bytes), dtype: T::DTYPE, len }
    }

    /// Create storage from raw bytes.
    pub fn from_bytes(bytes: Vec<u8>, dtype: DType, len: usize) -> Self {
        Storage { data: Arc::new(bytes), dtype, len }
    }

    /// Create empty storage for `len` elements of `dtype`.
    pub fn zeros(dtype: DType, len: usize) -> Self {
        let byte_len = len * dtype.size_of();
        Storage { data: Arc::new(vec![0u8; byte_len]), dtype, len }
    }

    /// Number of elements.
    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }

    /// Data type.
    pub fn dtype(&self) -> DType { self.dtype }

    /// Raw byte slice.
    pub fn as_bytes(&self) -> &[u8] { &self.data }

    /// Cast to typed slice.
    pub fn as_slice<T: crate::tensor::dtype::Element>(&self) -> &[T] {
        bytemuck::cast_slice(&self.data)
    }

    /// Get mutable typed slice — clones if shared (COW).
    pub fn as_mut_slice<T: crate::tensor::dtype::Element>(&mut self) -> &mut [T] {
        if Arc::strong_count(&self.data) > 1 {
            self.data = Arc::new(self.data.as_slice().to_vec());
        }
        bytemuck::cast_slice_mut(Arc::get_mut(&mut self.data).expect("COW clone failed"))
    }

    /// Is the storage shared (refcount > 1)?
    pub fn is_shared(&self) -> bool { Arc::strong_count(&self.data) > 1 }

    /// Byte length.
    pub fn byte_len(&self) -> usize { self.data.len() }
}

impl std::fmt::Debug for Storage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Storage")
            .field("dtype", &self.dtype)
            .field("len", &self.len)
            .field("shared", &self.is_shared())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cow_clones_on_mut() {
        let mut s1 = Storage::from_slice(&[1.0f32, 2.0, 3.0]);
        let s2 = s1.clone();
        assert!(s1.is_shared());
        let data = s1.as_mut_slice::<f32>();
        data[0] = 99.0;
        assert!(!s1.is_shared());
        assert_eq!(s2.as_slice::<f32>()[0], 1.0);
    }

    #[test]
    fn test_zeros() {
        let s = Storage::zeros(DType::F32, 4);
        assert_eq!(s.len(), 4);
        assert_eq!(s.dtype(), DType::F32);
    }
}
