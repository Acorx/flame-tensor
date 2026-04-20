//! NumPy-style broadcasting rules.

/// Compute the broadcast shape from two input shapes.
/// Returns the output shape or an error if shapes are incompatible.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
    let max_len = a.len().max(b.len());
    let mut result = vec![1usize; max_len];

    let a_pad = pad_left(a, max_len);
    let b_pad = pad_left(b, max_len);

    for i in 0..max_len {
        match (a_pad[i], b_pad[i]) {
            (x, y) if x == y => result[i] = x,
            (1, y) => result[i] = y,
            (x, 1) => result[i] = x,
            (x, y) => return Err(format!(
                "broadcast error: dim {} has size {} vs {}", i, x, y
            )),
        }
    }
    Ok(result)
}

/// Pad shape on the left with 1s to reach target length.
fn pad_left(shape: &[usize], target: usize) -> Vec<usize> {
    let mut padded = vec![1usize; target];
    let offset = target - shape.len();
    for (i, &d) in shape.iter().enumerate() {
        padded[offset + i] = d;
    }
    padded
}

/// Check if two shapes are broadcastable.
pub fn is_broadcastable(a: &[usize], b: &[usize]) -> bool {
    broadcast_shapes(a, b).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_broadcast() {
        let result = broadcast_shapes(&[3, 4], &[4]).unwrap();
        assert_eq!(result, vec![3, 4]);
    }

    #[test]
    fn test_scalar_broadcast() {
        let result = broadcast_shapes(&[3, 4], &[]).unwrap();
        assert_eq!(result, vec![3, 4]);
    }

    #[test]
    fn test_incompatible() {
        let result = broadcast_shapes(&[3, 4], &[5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_dim() {
        let result = broadcast_shapes(&[2, 1, 4], &[3, 4]).unwrap();
        assert_eq!(result, vec![2, 3, 4]);
    }
}
