//! Indexing, slicing, gather, scatter.

use crate::tensor::tensor::Tensor;
use crate::tensor::dtype::Element;

/// Index into the first dimension.
pub fn index_select<T: Element>(src: &Tensor, indices: &[usize]) -> Tensor {
    let src = src.contiguous();
    let src_dims = src.dims();
    assert!(!src_dims.is_empty());
    let batch = src_dims[0];
    let rest: usize = src_dims[1..].iter().product::<usize>().max(1);
    let src_data = src.as_slice::<T>();

    let mut out = Vec::with_capacity(indices.len() * rest);
    for &idx in indices {
        assert!(idx < batch, "index {} out of bounds for dim 0 of size {}", idx, batch);
        let start = idx * rest;
        out.extend_from_slice(&src_data[start..start + rest]);
    }

    let mut out_shape = vec![indices.len()];
    out_shape.extend_from_slice(&src_dims[1..]);
    Tensor::from_vec(out, out_shape)
}

/// Gather elements along an axis using an index tensor.
pub fn gather<T: Element>(src: &Tensor, axis: usize, index: &Tensor) -> Tensor {
    let src = src.contiguous();
    let index = index.contiguous();
    let src_dims = src.dims();
    let idx_dims = index.dims();
    assert_eq!(src_dims.len(), idx_dims.len(), "gather: src and index must have same ndim");
    assert!(axis < src_dims.len());

    let src_data = src.as_slice::<T>();
    let idx_data = index.as_slice::<i64>();
    let numel = idx_data.len();

    let mut out = vec![T::zero_val(); numel];
    let outer: usize = src_dims[..axis].iter().product::<usize>().max(1);
    let dim = src_dims[axis];
    let inner: usize = src_dims[axis + 1..].iter().product::<usize>().max(1);

    for o in 0..outer {
        for i in 0..inner {
            for d in 0..dim {
                // Check if this position is requested
                let idx_pos = o * dim * inner + d * inner + i;
                let gather_idx = idx_data[idx_pos] as usize;
                assert!(gather_idx < src_dims[axis], "gather index out of bounds");
                let src_pos = o * dim * inner + gather_idx * inner + i;
                out[idx_pos] = src_data[src_pos];
            }
        }
    }

    Tensor::from_vec(out, idx_dims.to_vec())
}

/// Scatter values along an axis using an index tensor.
pub fn scatter<T: Element + Copy>(dst: &Tensor, axis: usize, index: &Tensor, src: &Tensor) -> Tensor {
    let mut dst = dst.contiguous();
    let index = index.contiguous();
    let src = src.contiguous();
    let dst_dims = dst.dims().to_vec();
    let dim = dst_dims[axis];

    let dst_data = dst.as_mut_slice::<T>();
    let idx_data = index.as_slice::<i64>();
    let src_data = src.as_slice::<T>();

    let outer: usize = dst_dims[..axis].iter().product::<usize>().max(1);
    let inner: usize = dst_dims[axis + 1..].iter().product::<usize>().max(1);

    for o in 0..outer {
        for i in 0..inner {
            for d in 0..dim {
                let idx_pos = o * dim * inner + d * inner + i;
                let scatter_idx = idx_data[idx_pos] as usize;
                assert!(scatter_idx < dim, "scatter index out of bounds");
                let dst_pos = o * dim * inner + scatter_idx * inner + i;
                dst_data[dst_pos] = src_data[idx_pos];
            }
        }
    }
    dst
}

/// Narrow along a dimension (equivalent to slice).
pub fn narrow(src: &Tensor, dim: usize, start: usize, length: usize) -> Tensor {
    let dims = src.dims();
    assert!(dim < dims.len());
    assert!(start + length <= dims[dim]);

    let src = src.contiguous();
    let mut out_shape = dims.to_vec();
    out_shape[dim] = length;

    let outer: usize = dims[..dim].iter().product::<usize>().max(1);
    let inner: usize = dims[dim + 1..].iter().product::<usize>().max(1);
    let src_data = src.as_slice::<u8>();
    let elem_size = src.dtype().size_of();

    let mut out_bytes = vec![0u8; length * outer * inner * elem_size];
    for o in 0..outer {
        for l in 0..length {
            let src_off = (o * dims[dim] * inner + (start + l) * inner) * elem_size;
            let dst_off = (o * length * inner + l * inner) * elem_size;
            out_bytes[dst_off..dst_off + inner * elem_size]
                .copy_from_slice(&src_data[src_off..src_off + inner * elem_size]);
        }
    }

    let new_storage = crate::tensor::storage::Storage {
        data: std::sync::Arc::new(out_bytes),
        dtype: src.dtype(),
        len: outer * length * inner,
    };
    Tensor::from_storage(new_storage, out_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_select() {
        let t = Tensor::from_vec(vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0], vec![3, 2]);
        let sel = index_select::<f32>(&t, &[0, 2]);
        let data = sel.as_slice::<f32>();
        assert_eq!(data, &[10.0, 20.0, 50.0, 60.0]);
        assert_eq!(sel.dims(), &[2, 2]);
    }
}
