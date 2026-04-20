//! Autodiff integration tests.

use flame_tensor::{Tensor, Var};
use flame_tensor::autodiff::tape::new_tape;
use flame_tensor::autodiff::ops as diff_ops;
use std::cell::RefCell;
use std::rc::Rc;

#[test]
fn test_tape_creation() {
    let tape = new_tape();
    assert!(tape.is_empty());
}

#[test]
fn test_tape_backward_order() {
    let tape = flame_tensor::autodiff::tape::Tape::new();
    let order: Rc<RefCell<Vec<i32>>> = Rc::new(RefCell::new(Vec::new()));
    let o1 = Rc::clone(&order);
    let o2 = Rc::clone(&order);
    let o3 = Rc::clone(&order);
    tape.push(Box::new(move || { o1.borrow_mut().push(1); }));
    tape.push(Box::new(move || { o2.borrow_mut().push(2); }));
    tape.push(Box::new(move || { o3.borrow_mut().push(3); }));
    tape.backward();
    assert_eq!(*order.borrow(), vec![3, 2, 1]);
}

#[test]
fn test_var_creation() {
    let tape = new_tape();
    let v = Var::new(Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]), &tape);
    assert_eq!(v.data.dims(), &[3]);
}

#[test]
fn test_diff_add() {
    let tape = new_tape();
    let grads = flame_tensor::autodiff::gradient::new_grad_map();
    let a = Var::with_grads(Tensor::from_vec(vec![1.0f32, 2.0], vec![2]), &tape, &grads);
    let b = Var::with_grads(Tensor::from_vec(vec![3.0f32, 4.0], vec![2]), &tape, &grads);
    let c = diff_ops::add(&a, &b);
    let data = c.data.as_slice::<f32>();
    assert_eq!(data[0], 4.0);
    assert_eq!(data[1], 6.0);
}

#[test]
fn test_diff_relu() {
    let tape = new_tape();
    let grads = flame_tensor::autodiff::gradient::new_grad_map();
    let a = Var::with_grads(Tensor::from_vec(vec![-1.0f32, 0.0, 1.0, 2.0], vec![4]), &tape, &grads);
    let b = diff_ops::relu(&a);
    let data = b.data.as_slice::<f32>();
    assert_eq!(data[0], 0.0);
    assert_eq!(data[2], 1.0);
}

#[test]
fn test_gradient_accumulation() {
    use flame_tensor::autodiff::gradient;
    let grads = gradient::new_grad_map();
    let g1 = Tensor::from_vec(vec![1.0f32, 2.0], vec![2]);
    let g2 = Tensor::from_vec(vec![3.0f32, 4.0], vec![2]);
    gradient::accumulate_grad(&grads, 0, &g1);
    gradient::accumulate_grad(&grads, 0, &g2);
    let result = gradient::get_grad(&grads, 0).unwrap();
    let data = result.as_slice::<f32>();
    assert!((data[0] - 4.0).abs() < 1e-5);
    assert!((data[1] - 6.0).abs() < 1e-5);
}
