//! The tape: a linear list of backward closures.

use std::cell::RefCell;
use std::rc::Rc;

/// A backward closure that accumulates gradients when called.
type BackwardFn = Box<dyn FnMut()>;

/// The tape records operations during forward pass and replays them backward.
pub struct Tape {
    entries: RefCell<Vec<BackwardFn>>,
}

impl Tape {
    /// Create a new empty tape.
    pub fn new() -> Self {
        Tape { entries: RefCell::new(Vec::new()) }
    }

    /// Push a backward closure onto the tape.
    pub fn push(&self, backward: BackwardFn) {
        self.entries.borrow_mut().push(backward);
    }

    /// Play the tape backward: execute closures in reverse order.
    pub fn backward(&self) {
        let mut entries = self.entries.borrow_mut();
        while let Some(mut backward) = entries.pop() {
            backward();
        }
    }

    /// Number of recorded operations.
    pub fn len(&self) -> usize { self.entries.borrow().len() }
    pub fn is_empty(&self) -> bool { self.entries.borrow().is_empty() }

    /// Clear the tape.
    pub fn clear(&self) { self.entries.borrow_mut().clear(); }
}

impl Default for Tape {
    fn default() -> Self { Self::new() }
}

/// Shared tape pointer.
pub type TapePtr = Rc<Tape>;

/// Create a new shared tape.
pub fn new_tape() -> TapePtr { Rc::new(Tape::new()) }

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn test_tape_push_and_backward() {
        let call_order: Rc<RefCell<Vec<i32>>> = Rc::new(RefCell::new(Vec::new()));
        let co1 = Rc::clone(&call_order);
        let co2 = Rc::clone(&call_order);
        let co3 = Rc::clone(&call_order);
        let tape = Tape::new();
        tape.push(Box::new(move || { co1.borrow_mut().push(1); }));
        tape.push(Box::new(move || { co2.borrow_mut().push(2); }));
        tape.push(Box::new(move || { co3.borrow_mut().push(3); }));
        tape.backward();
        assert_eq!(*call_order.borrow(), vec![3, 2, 1]);
    }
}
