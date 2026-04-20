//! Sequential container: chain modules in order.

use crate::tensor::tensor::Tensor;
use crate::nn::module::{Module, ModuleError, Parameter};

/// Sequential container: runs modules in order, feeding output of one as input to next.
pub struct Sequential {
    modules: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new() -> Self {
        Sequential { modules: Vec::new() }
    }

    pub fn add<M: Module + 'static>(mut self, module: M) -> Self {
        self.modules.push(Box::new(module));
        self
    }

    pub fn push<M: Module + 'static>(&mut self, module: M) {
        self.modules.push(Box::new(module));
    }

    pub fn len(&self) -> usize { self.modules.len() }
    pub fn is_empty(&self) -> bool { self.modules.is_empty() }
}

impl Default for Sequential {
    fn default() -> Self { Self::new() }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Result<Tensor, ModuleError> {
        let mut x = input.clone();
        for module in &self.modules {
            x = module.forward(&x)?;
        }
        Ok(x)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        for module in &self.modules {
            params.extend(module.parameters());
        }
        params
    }

    fn name(&self) -> &str { "Sequential" }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::linear::Linear;
    use crate::nn::activation::ReLU;

    #[test]
    fn test_sequential() {
        let seq = Sequential::new()
            .add(Linear::new(4, 8, true))
            .add(ReLU)
            .add(Linear::new(8, 2, true));
        let x = Tensor::from_vec(vec![1.0f32; 4], vec![1, 4]);
        let y = seq.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 2]);
    }
}
