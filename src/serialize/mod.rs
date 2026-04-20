//! Serialization module.

pub mod safetensors;

pub use safetensors::{save, load};
