//! Optimizers: SGD, Adam, AdamW, and LR schedulers.

pub mod sgd;
pub mod adam;
pub mod scheduler;

pub use sgd::SGD;
pub use adam::{Adam, AdamW};
pub use scheduler::{LRScheduler, CosineAnnealingLR, LinearWarmup, ReduceLROnPlateau};

use crate::tensor::tensor::Tensor;

/// Core optimizer trait.
pub trait Optimizer {
    /// Perform a single optimization step, updating parameters in-place.
    fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]);

    /// Reset all internal state (momentum buffers, etc.).
    fn zero_grad(&mut self);
}
