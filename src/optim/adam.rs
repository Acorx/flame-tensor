//! Adam and AdamW optimizers.

use crate::tensor::tensor::Tensor;
use crate::tensor::ops;
use crate::optim::Optimizer;

/// Adam optimizer.
pub struct Adam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub amsgrad: bool,
    m: Vec<Tensor>,  // first moment
    v: Vec<Tensor>,  // second moment
    step_count: usize,
    initialized: bool,
}

impl Adam {
    pub fn new(lr: f64, beta1: f64, beta2: f64, eps: f64, weight_decay: f64) -> Self {
        Adam { lr, beta1, beta2, eps, weight_decay, amsgrad: false, m: Vec::new(), v: Vec::new(), step_count: 0, initialized: false }
    }

    pub fn default(lr: f64) -> Self {
        Adam::new(lr, 0.9, 0.999, 1e-8, 0.0)
    }
}

impl Optimizer for Adam {
    fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) {
        assert_eq!(params.len(), grads.len());
        if !self.initialized {
            self.m = params.iter().map(|p| Tensor::from_vec(vec![0.0f32; p.numel()], p.dims().to_vec())).collect();
            self.v = params.iter().map(|p| Tensor::from_vec(vec![0.0f32; p.numel()], p.dims().to_vec())).collect();
            self.initialized = true;
        }
        self.step_count += 1;

        for i in 0..params.len() {
            let mut g = grads[i].clone();

            if self.weight_decay != 0.0 {
                let wd = ops::mul_scalar(&params[i], self.weight_decay as f32);
                g = ops::add(&g, &wd);
            }

            // m = beta1 * m + (1 - beta1) * g
            self.m[i] = ops::add(
                &ops::mul_scalar(&self.m[i], self.beta1 as f32),
                &ops::mul_scalar(&g, (1.0 - self.beta1) as f32),
            );

            // v = beta2 * v + (1 - beta2) * g^2
            let g2 = ops::mul(&g, &g);
            self.v[i] = ops::add(
                &ops::mul_scalar(&self.v[i], self.beta2 as f32),
                &ops::mul_scalar(&g2, (1.0 - self.beta2) as f32),
            );

            // Bias correction
            let bias1 = 1.0 - self.beta1.powi(self.step_count as i32);
            let bias2 = 1.0 - self.beta2.powi(self.step_count as i32);
            let m_hat = ops::mul_scalar(&self.m[i], (1.0 / bias1) as f32);
            let v_hat = ops::mul_scalar(&self.v[i], (1.0 / bias2) as f32);

            // p = p - lr * m_hat / (sqrt(v_hat) + eps)
            let v_sqrt = ops::sqrt(&v_hat);
            let v_eps = ops::add_scalar(&v_sqrt, self.eps as f32);
            let update = ops::div(&m_hat, &v_eps);
            let update = ops::mul_scalar(&update, self.lr as f32);
            params[i] = ops::sub(&params[i], &update);
        }
    }

    fn zero_grad(&mut self) {
        for m in &mut self.m { *m = Tensor::from_vec(vec![0.0f32; m.numel()], m.dims().to_vec()); }
        for v in &mut self.v { *v = Tensor::from_vec(vec![0.0f32; v.numel()], v.dims().to_vec()); }
        self.step_count = 0;
    }
}

/// AdamW optimizer (decoupled weight decay).
pub struct AdamW {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    m: Vec<Tensor>,
    v: Vec<Tensor>,
    step_count: usize,
    initialized: bool,
}

impl AdamW {
    pub fn new(lr: f64, beta1: f64, beta2: f64, eps: f64, weight_decay: f64) -> Self {
        AdamW { lr, beta1, beta2, eps, weight_decay, m: Vec::new(), v: Vec::new(), step_count: 0, initialized: false }
    }

    pub fn default(lr: f64, weight_decay: f64) -> Self {
        AdamW::new(lr, 0.9, 0.999, 1e-8, weight_decay)
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) {
        assert_eq!(params.len(), grads.len());
        if !self.initialized {
            self.m = params.iter().map(|p| Tensor::from_vec(vec![0.0f32; p.numel()], p.dims().to_vec())).collect();
            self.v = params.iter().map(|p| Tensor::from_vec(vec![0.0f32; p.numel()], p.dims().to_vec())).collect();
            self.initialized = true;
        }
        self.step_count += 1;

        for i in 0..params.len() {
            let g = grads[i].clone();

            // Decoupled weight decay: p = p * (1 - lr * wd)
            if self.weight_decay != 0.0 {
                let decay = ops::mul_scalar(&params[i], (1.0 - self.lr * self.weight_decay) as f32);
                params[i] = decay;
            }

            // Standard Adam update on g (no L2 regularization)
            self.m[i] = ops::add(
                &ops::mul_scalar(&self.m[i], self.beta1 as f32),
                &ops::mul_scalar(&g, (1.0 - self.beta1) as f32),
            );
            let g2 = ops::mul(&g, &g);
            self.v[i] = ops::add(
                &ops::mul_scalar(&self.v[i], self.beta2 as f32),
                &ops::mul_scalar(&g2, (1.0 - self.beta2) as f32),
            );

            let bias1 = 1.0 - self.beta1.powi(self.step_count as i32);
            let bias2 = 1.0 - self.beta2.powi(self.step_count as i32);
            let m_hat = ops::mul_scalar(&self.m[i], (1.0 / bias1) as f32);
            let v_hat = ops::mul_scalar(&self.v[i], (1.0 / bias2) as f32);

            let v_sqrt = ops::sqrt(&v_hat);
            let v_eps = ops::add_scalar(&v_sqrt, self.eps as f32);
            let update = ops::div(&m_hat, &v_eps);
            let update = ops::mul_scalar(&update, self.lr as f32);
            params[i] = ops::sub(&params[i], &update);
        }
    }

    fn zero_grad(&mut self) {
        for m in &mut self.m { *m = Tensor::from_vec(vec![0.0f32; m.numel()], m.dims().to_vec()); }
        for v in &mut self.v { *v = Tensor::from_vec(vec![0.0f32; v.numel()], v.dims().to_vec()); }
        self.step_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adam_step() {
        let mut opt = Adam::default(0.001);
        let mut params = vec![Tensor::from_vec(vec![1.0f32, 2.0], vec![2])];
        let grads = vec![Tensor::from_vec(vec![0.5f32, -0.5], vec![2])];
        opt.step(&mut params, &grads);
        // Should have updated
        let data = params[0].as_slice::<f32>();
        assert!(data[0] < 1.0);
        assert!(data[1] > 2.0);
    }

    #[test]
    fn test_adamw_step() {
        let mut opt = AdamW::default(0.001, 0.01);
        let mut params = vec![Tensor::from_vec(vec![1.0f32, 2.0], vec![2])];
        let grads = vec![Tensor::from_vec(vec![0.5f32, -0.5], vec![2])];
        opt.step(&mut params, &grads);
        let data = params[0].as_slice::<f32>();
        assert!(data[0] < 1.0);
    }
}
