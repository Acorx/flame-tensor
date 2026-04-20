//! Var: a differentiable tensor with a tape pointer.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::tensor::tensor::Tensor;
use crate::autodiff::tape::{Tape, TapePtr};
use crate::autodiff::gradient::{GradMapPtr, new_grad_map, accumulate_grad, get_grad};

/// Global variable ID counter.
static NEXT_VAR_ID: AtomicUsize = AtomicUsize::new(0);

/// A differentiable variable wrapping a Tensor.
///
/// Each Var holds:
/// - A `Tensor` (the data)
/// - A unique `id` for gradient lookup
/// - A shared `TapePtr` for recording backward closures
/// - A shared `GradMapPtr` for gradient accumulation
pub struct Var {
    pub data: Tensor,
    pub id: usize,
    pub tape: TapePtr,
    pub grads: GradMapPtr,
}

impl Var {
    /// Create a new Var from a Tensor, attaching it to the given tape.
    pub fn new(data: Tensor, tape: &TapePtr) -> Self {
        let id = NEXT_VAR_ID.fetch_add(1, Ordering::Relaxed);
        Var {
            data,
            id,
            tape: Rc::clone(tape),
            grads: new_grad_map(),
        }
    }

    /// Create a Var with shared gradient map (for multi-variable graphs).
    pub fn with_grads(data: Tensor, tape: &TapePtr, grads: &GradMapPtr) -> Self {
        let id = NEXT_VAR_ID.fetch_add(1, Ordering::Relaxed);
        Var {
            data,
            id,
            tape: Rc::clone(tape),
            grads: Rc::clone(grads),
        }
    }

    /// Create a "leaf" variable — no tape, no gradients.
    pub fn leaf(data: Tensor) -> Self {
        let id = NEXT_VAR_ID.fetch_add(1, Ordering::Relaxed);
        Var {
            data,
            id,
            tape: Rc::new(Tape::new()),
            grads: new_grad_map(),
        }
    }

    /// Get the gradient for this variable.
    pub fn grad(&self) -> Option<Tensor> {
        get_grad(&self.grads, self.id)
    }

    /// Run backward pass: seed gradient of 1.0, play the tape.
    pub fn backward(&self) {
        let seed = Tensor::scalar(1.0f32);
        accumulate_grad(&self.grads, self.id, &seed);
        self.tape.backward();
    }

    /// Zero gradients for this variable.
    pub fn zero_grad(&self) {
        self.grads.borrow_mut().remove(&self.id);
    }

    /// Unique ID.
    pub fn id(&self) -> usize {
        self.id
    }

    /// Shape.
    pub fn shape(&self) -> &crate::tensor::tensor::Shape {
        self.data.shape()
    }
}

impl Clone for Var {
    fn clone(&self) -> Self {
        Var {
            data: self.data.clone(),
            id: self.id,
            tape: Rc::clone(&self.tape),
            grads: Rc::clone(&self.grads),
        }
    }
}

impl std::fmt::Debug for Var {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Var")
            .field("id", &self.id)
            .field("data", &self.data)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::tape::new_tape;

    #[test]
    fn test_var_create() {
        let tape = new_tape();
        let v = Var::new(Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]), &tape);
        assert_eq!(v.data.dims(), &[3]);
        assert_eq!(v.id(), v.id()); // stable
    }

    #[test]
    fn test_var_unique_ids() {
        let tape = new_tape();
        let v1 = Var::new(Tensor::scalar(1.0f32), &tape);
        let v2 = Var::new(Tensor::scalar(2.0f32), &tape);
        assert_ne!(v1.id(), v2.id());
    }
}
