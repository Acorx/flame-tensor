//! Differentiable tensor operations.
//!
//! Each operation records a backward closure on the shared tape.

use std::rc::Rc;
use std::cell::RefCell;

use crate::tensor::tensor::Tensor;
use crate::tensor::ops;
use crate::autodiff::var::Var;
use crate::autodiff::tape::TapePtr;
use crate::autodiff::gradient::{GradMapPtr, new_grad_map, accumulate_grad};

/// Differentiable addition: z = a + b.
pub fn add(a: &Var, b: &Var) -> Var {
    let data = ops::add(&a.data, &b.data);
    let id = next_id();
    let tape = Rc::clone(&a.tape);
    let grads = merge_grads(&a.grads, &b.grads);
    let a_id = a.id;
    let b_id = b.id;

    let g = Rc::clone(&grads);
    tape.push(Box::new(move || {
        let ones = Tensor::from_vec(vec![1.0f32], vec![]);
        accumulate_grad(&g, a_id, &ones);
        accumulate_grad(&g, b_id, &ones);
    }));

    Var { data, id, tape, grads }
}

/// Differentiable subtraction: z = a - b.
pub fn sub(a: &Var, b: &Var) -> Var {
    let data = ops::sub(&a.data, &b.data);
    let id = next_id();
    let tape = Rc::clone(&a.tape);
    let grads = merge_grads(&a.grads, &b.grads);
    let a_id = a.id;
    let b_id = b.id;

    let g = Rc::clone(&grads);
    tape.push(Box::new(move || {
        let ones = Tensor::from_vec(vec![1.0f32], vec![]);
        let neg = Tensor::from_vec(vec![-1.0f32], vec![]);
        accumulate_grad(&g, a_id, &ones);
        accumulate_grad(&g, b_id, &neg);
    }));

    Var { data, id, tape, grads }
}

/// Differentiable element-wise multiplication: z = a * b.
pub fn mul(a: &Var, b: &Var) -> Var {
    let data = ops::mul(&a.data, &b.data);
    let id = next_id();
    let tape = Rc::clone(&a.tape);
    let grads = merge_grads(&a.grads, &b.grads);
    let a_id = a.id;
    let b_id = b.id;

    let g = Rc::clone(&grads);
    tape.push(Box::new(move || {
        let ones = Tensor::from_vec(vec![1.0f32], vec![]);
        accumulate_grad(&g, a_id, &ones);
        accumulate_grad(&g, b_id, &ones);
    }));

    Var { data, id, tape, grads }
}

/// Differentiable ReLU: z = max(0, a).
pub fn relu(a: &Var) -> Var {
    let data = ops::relu(&a.data);
    let id = next_id();
    let tape = Rc::clone(&a.tape);
    let a_id = a.id;

    let g = Rc::clone(&a.grads);
    let a_data = a.data.clone();
    tape.push(Box::new(move || {
        let mask = ops::relu_mask(&a_data);
        accumulate_grad(&g, a_id, &mask);
    }));

    Var { data, id, tape, grads: Rc::clone(&a.grads) }
}

/// Differentiable sigmoid.
pub fn sigmoid(a: &Var) -> Var {
    let data = ops::sigmoid(&a.data);
    let id = next_id();
    let tape = Rc::clone(&a.tape);
    let a_id = a.id;

    let g = Rc::clone(&a.grads);
    tape.push(Box::new(move || {
        let ones = Tensor::scalar(1.0f32);
        accumulate_grad(&g, a_id, &ones);
    }));

    Var { data, id, tape, grads: Rc::clone(&a.grads) }
}

/// Differentiable tanh.
pub fn tanh(a: &Var) -> Var {
    let data = ops::tanh(&a.data);
    let id = next_id();
    let tape = Rc::clone(&a.tape);
    let a_id = a.id;

    let g = Rc::clone(&a.grads);
    tape.push(Box::new(move || {
        let ones = Tensor::scalar(1.0f32);
        accumulate_grad(&g, a_id, &ones);
    }));

    Var { data, id, tape, grads: Rc::clone(&a.grads) }
}

/// Differentiable matmul: z = a @ b.
pub fn matmul(a: &Var, b: &Var) -> Var {
    let data = ops::matmul(&a.data, &b.data);
    let id = next_id();
    let tape = Rc::clone(&a.tape);
    let grads = merge_grads(&a.grads, &b.grads);
    let a_id = a.id;
    let b_id = b.id;
    let a_shape = a.data.dims().to_vec();
    let b_shape = b.data.dims().to_vec();

    let g = Rc::clone(&grads);
    tape.push(Box::new(move || {
        let grad_a = Tensor::from_vec(vec![1.0f32; a_shape.iter().product()], a_shape.clone());
        let grad_b = Tensor::from_vec(vec![1.0f32; b_shape.iter().product()], b_shape.clone());
        accumulate_grad(&g, a_id, &grad_a);
        accumulate_grad(&g, b_id, &grad_b);
    }));

    Var { data, id, tape, grads }
}

fn next_id() -> usize {
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

fn merge_grads(a: &GradMapPtr, b: &GradMapPtr) -> GradMapPtr {
    Rc::clone(a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::tape::new_tape;

    #[test]
    fn test_diff_add() {
        let tape = new_tape();
        let grads = new_grad_map();
        let a = Var::with_grads(Tensor::from_vec(vec![1.0f32, 2.0], vec![2]), &tape, &grads);
        let b = Var::with_grads(Tensor::from_vec(vec![3.0f32, 4.0], vec![2]), &tape, &grads);
        let c = add(&a, &b);
        assert_eq!(c.data.dims(), &[2]);
    }
}
