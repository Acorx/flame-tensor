//! SGD optimizer with momentum, weight decay, and Nesterov.

use crate::tensor::tensor::Tensor;
use crate::tensor::ops;
use crate::optim::Optimizer;

/// Stochastic Gradient Descent with optional momentum and weight decay.
pub struct SGD {
    pub lr: f64,
    pub momentum: f64,
    pub weight_decay: f64,
    pub nesterov: bool,
    velocity: Vec<Tensor>,
    initialized: bool,
}

impl SGD {
    pub fn new(lr: f64, momentum: f64, weight_decay: f64, nesterov: bool) -> Self {
        SGD { lr, momentum, weight_decay, nesterov, velocity: Vec::new(), initialized: false }
    }

    pub fn simple(lr: f64) -> Self {
        SGD::new(lr, 0.0, 0.0, false)
    }
}

impl Optimizer for SGD {
    fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) {
        assert_eq!(params.len(), grads.len(), "params and grads must have same length");

        if !self.initialized {
            self.velocity = params.iter().map(|p| {
                Tensor::from_vec(vec![0.0f32; p.numel()], p.dims().to_vec())
            }).collect();
            self.initialized = true;
        }

        for i in 0..params.len() {
            let mut g = grads[i].clone();

            // Weight decay
            if self.weight_decay != 0.0 {
                let wd = ops::mul_scalar(&params[i], self.weight_decay as f32);
                g = ops::add(&g, &wd);
            }

            // Momentum
            if self.momentum != 0.0 {
                self.velocity[i] = ops::add(
                    &ops::mul_scalar(&self.velocity[i], self.momentum as f32),
                    &g,
                );
                if self.nesterov {
                    g = ops::add(&g, &ops::mul_scalar(&self.velocity[i], self.momentum as f32));
                } else {
                    g = self.velocity[i].clone();
                }
            }

            // Update: p = p - lr * g
            let update = ops::mul_scalar(&g, self.lr as f32);
            params[i] = ops::sub(&params[i], &update);
        }
    }

    fn zero_grad(&mut self) {
        for v in &mut self.velocity {
            *v = Tensor::from_vec(vec![0.0f32; v.numel()], v.dims().to_vec());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_simple() {
        let mut opt = SGD::simple(0.1);
        let mut params = vec![Tensor::from_vec(vec![1.0f32, 2.0], vec![2])];
        let grads = vec![Tensor::from_vec(vec![0.1f32, 0.2], vec![2])];
        opt.step(&mut params, &grads);
        let data = params[0].as_slice::<f32>();
        assert!((data[0] - 0.99).abs() < 1e-5);
        assert!((data[1] - 1.98).abs() < 1e-5);
    }
}
