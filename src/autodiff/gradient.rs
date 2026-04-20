//! Gradient storage and accumulation.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use crate::tensor::tensor::Tensor;

/// Gradient map: variable ID → gradient tensor.
pub type GradMap = RefCell<HashMap<usize, Tensor>>;

/// Shared gradient map.
pub type GradMapPtr = Rc<GradMap>;

/// Create a new shared gradient map.
pub fn new_grad_map() -> GradMapPtr { Rc::new(RefCell::new(HashMap::new())) }

/// Accumulate gradient for a variable.
pub fn accumulate_grad(grads: &GradMapPtr, var_id: usize, grad: &Tensor) {
    let mut map = grads.borrow_mut();
    let existing = map.remove(&var_id);
    match existing {
        Some(e) => { map.insert(var_id, crate::tensor::ops::add(&e, grad)); }
        None => { map.insert(var_id, grad.clone()); }
    }
}

/// Get the gradient for a variable.
pub fn get_grad(grads: &GradMapPtr, var_id: usize) -> Option<Tensor> {
    grads.borrow().get(&var_id).cloned()
}

/// Zero all gradients.
pub fn zero_grads(grads: &GradMapPtr) {
    grads.borrow_mut().clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulate_and_get() {
        let grads = new_grad_map();
        let g1 = Tensor::from_vec(vec![1.0f32, 2.0], vec![2]);
        accumulate_grad(&grads, 0, &g1);
        let g2 = Tensor::from_vec(vec![3.0f32, 4.0], vec![2]);
        accumulate_grad(&grads, 0, &g2);
        let result = get_grad(&grads, 0).unwrap();
        let data = result.as_slice::<f32>();
        assert!((data[0] - 4.0).abs() < 1e-5);
        assert!((data[1] - 6.0).abs() < 1e-5);
    }
}
