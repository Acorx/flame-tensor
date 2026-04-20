//! # flame-tensor
//!
//! A full tensor framework in Rust — *le nouveau torch mais en rust et incroyablement efficace*.
//!
//! ## Features
//! - Multi-precision: f16, bf16, f32, f64
//! - Tape-based reverse-mode automatic differentiation
//! - COW (copy-on-write) storage with zero-copy strided views
//! - CPU SIMD-accelerated operations
//! - Full neural network module library (Linear, Conv, Attention, Transformer)
//! - Optimizers: SGD, Adam, AdamW with LR scheduling
//! - Safetensors serialization (PyTorch/HuggingFace interop)
//! - Optional CUDA backend (feature-gated)

pub mod tensor;
pub mod autodiff;
pub mod nn;
pub mod optim;
pub mod serialize;
#[cfg(feature = "cpu")]
pub mod backend;

pub use tensor::Tensor;
pub use tensor::DType;
pub use tensor::tensor::Shape;
pub use autodiff::{Var, Tape};
