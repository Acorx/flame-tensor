//! Tape-based reverse-mode automatic differentiation.
//!
//! Unlike graph-based AD (PyTorch), we use a single tape per forward pass.
//! Each operation pushes a backward closure onto the tape. On `.backward()`,
//! the tape is played in reverse. No graph traversal overhead, no reference cycles.

pub mod tape;
pub mod var;
pub mod ops;
pub mod gradient;

pub use tape::Tape;
pub use var::Var;
