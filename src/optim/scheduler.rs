//! Learning rate schedulers.

/// Trait for LR schedulers.
pub trait LRScheduler {
    /// Step the scheduler. Optional metric for ReduceLROnPlateau.
    fn step(&mut self, metric: Option<f64>) -> f64;

    /// Current learning rate.
    fn get_lr(&self) -> f64;
}

/// Cosine annealing learning rate scheduler.
pub struct CosineAnnealingLR {
    pub base_lr: f64,
    pub eta_min: f64,
    pub t_max: usize,
    current_epoch: usize,
}

impl CosineAnnealingLR {
    pub fn new(base_lr: f64, t_max: usize, eta_min: f64) -> Self {
        CosineAnnealingLR { base_lr, eta_min, t_max, current_epoch: 0 }
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn step(&mut self, _metric: Option<f64>) -> f64 {
        self.current_epoch += 1;
        let progress = self.current_epoch as f64 / self.t_max as f64;
        let lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1.0 + (std::f64::consts::PI * progress).cos());
        lr
    }

    fn get_lr(&self) -> f64 {
        let progress = self.current_epoch as f64 / self.t_max as f64;
        self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1.0 + (std::f64::consts::PI * progress).cos())
    }
}

/// Linear warmup then decay.
pub struct LinearWarmup {
    pub base_lr: f64,
    pub target_lr: f64,
    pub warmup_steps: usize,
    pub total_steps: usize,
    current_step: usize,
}

impl LinearWarmup {
    pub fn new(base_lr: f64, target_lr: f64, warmup_steps: usize, total_steps: usize) -> Self {
        LinearWarmup { base_lr, target_lr, warmup_steps, total_steps, current_step: 0 }
    }
}

impl LRScheduler for LinearWarmup {
    fn step(&mut self, _metric: Option<f64>) -> f64 {
        self.current_step += 1;
        self.get_lr()
    }

    fn get_lr(&self) -> f64 {
        if self.current_step <= self.warmup_steps {
            // Linear warmup from 0 to base_lr
            self.base_lr * (self.current_step as f64 / self.warmup_steps as f64)
        } else {
            // Linear decay from base_lr to target_lr
            let remaining = self.total_steps - self.warmup_steps;
            let progress = (self.current_step - self.warmup_steps) as f64 / remaining.max(1) as f64;
            self.base_lr + (self.target_lr - self.base_lr) * progress.min(1.0)
        }
    }
}

/// Reduce LR on plateau.
pub struct ReduceLROnPlateau {
    pub base_lr: f64,
    pub factor: f64,
    pub patience: usize,
    pub min_lr: f64,
    pub threshold: f64,
    current_lr: f64,
    best_metric: Option<f64>,
    bad_epochs: usize,
}

impl ReduceLROnPlateau {
    pub fn new(base_lr: f64, factor: f64, patience: usize, min_lr: f64) -> Self {
        ReduceLROnPlateau { base_lr, factor, patience, min_lr, threshold: 1e-4, current_lr: base_lr, best_metric: None, bad_epochs: 0 }
    }
}

impl LRScheduler for ReduceLROnPlateau {
    fn step(&mut self, metric: Option<f64>) -> f64 {
        if let Some(m) = metric {
            let improved = match self.best_metric {
                Some(best) => m < best - self.threshold,
                None => true,
            };
            if improved {
                self.best_metric = Some(m);
                self.bad_epochs = 0;
            } else {
                self.bad_epochs += 1;
                if self.bad_epochs >= self.patience {
                    self.current_lr = (self.current_lr * self.factor).max(self.min_lr);
                    self.bad_epochs = 0;
                }
            }
        }
        self.current_lr
    }

    fn get_lr(&self) -> f64 { self.current_lr }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_annealing() {
        let mut sched = CosineAnnealingLR::new(0.1, 100, 0.0);
        let lr1 = sched.step(None);
        assert!(lr1 < 0.1);
        assert!(lr1 > 0.0);
    }

    #[test]
    fn test_linear_warmup() {
        let mut sched = LinearWarmup::new(0.1, 0.0, 10, 100);
        // At step 0, lr should be ~0
        assert!(sched.get_lr() < 0.01);
        for _ in 0..10 { sched.step(None); }
        assert!((sched.get_lr() - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_reduce_on_plateau() {
        let mut sched = ReduceLROnPlateau::new(0.1, 0.5, 3, 1e-6);
        // Feed same metric for 4 steps (> patience of 3)
        for _ in 0..4 { sched.step(Some(1.0)); }
        assert!(sched.get_lr() < 0.1);
    }
}
