//! Module trait and Parameter struct.

use crate::tensor::tensor::Tensor;
use std::collections::HashMap;

/// Core trait for neural network modules.
pub trait Module {
    /// Forward pass: input tensor → output tensor.
    fn forward(&self, input: &Tensor) -> Result<Tensor, ModuleError>;

    /// Collect all named parameters.
    fn parameters(&self) -> Vec<Parameter> {
        Vec::new()
    }

    /// Module name for debugging.
    fn name(&self) -> &str {
        "unnamed"
    }
}

/// A named parameter wrapping a Tensor.
#[derive(Clone, Debug)]
pub struct Parameter {
    pub name: String,
    pub data: Tensor,
    pub requires_grad: bool,
}

impl Parameter {
    pub fn new(name: impl Into<String>, data: Tensor) -> Self {
        Parameter { name: name.into(), data, requires_grad: true }
    }

    pub fn frozen(name: impl Into<String>, data: Tensor) -> Self {
        Parameter { name: name.into(), data, requires_grad: false }
    }
}

/// Error type for module operations.
#[derive(Debug, thiserror::Error)]
pub enum ModuleError {
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    #[error("dimension error: {0}")]
    DimError(String),
    #[error("computation error: {0}")]
    ComputationError(String),
}

/// Trait for modules that can list named parameters.
pub trait NamedParameters {
    fn named_parameters(&self) -> HashMap<String, Parameter>;
}
