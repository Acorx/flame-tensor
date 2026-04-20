//! Data types supported by Flame-Tensor.

use std::fmt;
use bytemuck::{Pod, Zeroable};
use num_traits::{Num, NumCast};

/// Dynamic data type enum for runtime dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F16,
    BF16,
    F32,
    F64,
    I32,
    I64,
    U8,
    Bool,
}

impl DType {
    /// Size in bytes for this dtype.
    pub fn size_of(&self) -> usize {
        match self {
            DType::F16 | DType::BF16 => 2,
            DType::F32 | DType::I32 => 4,
            DType::F64 | DType::I64 => 8,
            DType::U8 | DType::Bool => 1,
        }
    }

    /// Is this a float type?
    pub fn is_float(&self) -> bool {
        matches!(self, DType::F16 | DType::BF16 | DType::F32 | DType::F64)
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F16 => write!(f, "f16"),
            DType::BF16 => write!(f, "bf16"),
            DType::F32 => write!(f, "f32"),
            DType::F64 => write!(f, "f64"),
            DType::I32 => write!(f, "i32"),
            DType::I64 => write!(f, "i64"),
            DType::U8 => write!(f, "u8"),
            DType::Bool => write!(f, "bool"),
        }
    }
}

/// Trait bound for all numeric types supported as tensor elements.
pub trait Element: Copy + Clone + Default + Pod + Zeroable + Num + NumCast + Send + Sync + 'static {
    const DTYPE: DType;
    fn zero_val() -> Self;
    fn one_val() -> Self;
}

impl Element for f32 {
    const DTYPE: DType = DType::F32;
    fn zero_val() -> Self { 0.0 }
    fn one_val() -> Self { 1.0 }
}

impl Element for f64 {
    const DTYPE: DType = DType::F64;
    fn zero_val() -> Self { 0.0 }
    fn one_val() -> Self { 1.0 }
}

impl Element for i32 {
    const DTYPE: DType = DType::I32;
    fn zero_val() -> Self { 0 }
    fn one_val() -> Self { 1 }
}

impl Element for i64 {
    const DTYPE: DType = DType::I64;
    fn zero_val() -> Self { 0 }
    fn one_val() -> Self { 1 }
}

impl Element for u8 {
    const DTYPE: DType = DType::U8;
    fn zero_val() -> Self { 0 }
    fn one_val() -> Self { 1 }
}

// Note: bool is NOT an Element — too many trait issues.
// Use u8 for boolean tensors instead.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_sizes() {
        assert_eq!(DType::F32.size_of(), 4);
        assert_eq!(DType::F64.size_of(), 8);
        assert_eq!(DType::F16.size_of(), 2);
        assert_eq!(DType::I32.size_of(), 4);
        assert_eq!(DType::U8.size_of(), 1);
    }

    #[test]
    fn test_dtype_is_float() {
        assert!(DType::F32.is_float());
        assert!(DType::F64.is_float());
        assert!(!DType::I32.is_float());
    }
}
