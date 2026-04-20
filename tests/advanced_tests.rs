//! Advanced integration tests for flame-tensor.

use flame_tensor::{Tensor, DType};
use flame_tensor::tensor::ops;
use flame_tensor::tensor::reduce;
use flame_tensor::tensor::creation;
use flame_tensor::nn::{Linear, ReLU, GELU, Sigmoid, Sequential, LayerNorm, Dropout};
use flame_tensor::nn::module::Module;
use flame_tensor::nn::conv::Conv2d;
use flame_tensor::nn::embedding::{Embedding, PositionalEncoding};
use flame_tensor::nn::attention::MultiHeadSelfAttention;
use flame_tensor::nn::transformer::TransformerBlock;
use flame_tensor::optim::{SGD, Adam, AdamW, Optimizer};
use flame_tensor::optim::scheduler::{CosineAnnealingLR, LinearWarmup, ReduceLROnPlateau, LRScheduler};

// ===== TENSOR OPS TESTS =====

#[test]
fn test_matmul_identity() {
    let eye = creation::eye::<f32>(4);
    let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 4]);
    let y = ops::matmul(&x, &eye);
    let x_data = x.as_slice::<f32>();
    let y_data = y.as_slice::<f32>();
    for i in 0..8 {
        assert!((x_data[i] - y_data[i]).abs() < 1e-5, "matmul identity failed at {}", i);
    }
}

#[test]
fn test_matmul_chain() {
    // (A @ B) @ C should equal A @ (B @ C) for compatible shapes
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![2.0f32, 0.0, 1.0, 3.0], vec![2, 2]);
    let c = Tensor::from_vec(vec![1.0f32, 1.0, 0.0, 1.0], vec![2, 2]);
    let ab_c = ops::matmul(&ops::matmul(&a, &b), &c);
    let a_bc = ops::matmul(&a, &ops::matmul(&b, &c));
    let d1 = ab_c.as_slice::<f32>();
    let d2 = a_bc.as_slice::<f32>();
    for i in 0..4 {
        assert!((d1[i] - d2[i]).abs() < 1e-4, "matmul associativity failed at {}", i);
    }
}

#[test]
fn test_elementwise_chain() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]);
    let b = Tensor::from_vec(vec![2.0f32, 3.0, 4.0, 5.0], vec![4]);
    // (a + b) * a should equal a*a + a*b
    let left = ops::mul(&ops::add(&a, &b), &a);
    let aa = ops::mul(&a, &a);
    let ab = ops::mul(&a, &b);
    let right = ops::add(&aa, &ab);
    let l = left.as_slice::<f32>();
    let r = right.as_slice::<f32>();
    for i in 0..4 {
        assert!((l[i] - r[i]).abs() < 1e-4, "elementwise distributivity failed at {}", i);
    }
}

#[test]
fn test_relu_values() {
    let x = Tensor::from_vec(vec![-3.0f32, -1.0, 0.0, 0.5, 2.0, 10.0], vec![6]);
    let y = ops::relu(&x);
    let d = y.as_slice::<f32>();
    assert_eq!(d[0], 0.0);
    assert_eq!(d[1], 0.0);
    assert_eq!(d[2], 0.0);
    assert!((d[3] - 0.5).abs() < 1e-5);
    assert!((d[4] - 2.0).abs() < 1e-5);
    assert!((d[5] - 10.0).abs() < 1e-5);
}

#[test]
fn test_gelu_values() {
    // GELU(0) ≈ 0, GELU(1) ≈ 0.8413
    let x = Tensor::from_vec(vec![0.0f32, 1.0, -1.0], vec![3]);
    let y = ops::gelu(&x);
    let d = y.as_slice::<f32>();
    assert!(d[0].abs() < 0.01, "GELU(0) should be ~0, got {}", d[0]);
    assert!((d[1] - 0.8413).abs() < 0.01, "GELU(1) should be ~0.8413, got {}", d[1]);
    assert!(d[2].abs() < 0.16, "GELU(-1) should be ~-0.1588, got {}", d[2]);
}

#[test]
fn test_softmax_sums_to_one() {
    let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![1, 4]);
    let s = ops::softmax(&x, -1);
    let d = s.as_slice::<f32>();
    let sum: f32 = d.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {}, expected 1.0", sum);
    // All positive
    for &v in d { assert!(v > 0.0); }
}

#[test]
fn test_softmax_large_values() {
    let x = Tensor::from_vec(vec![100.0f32, 101.0, 102.0], vec![1, 3]);
    let s = ops::softmax(&x, -1);
    let d = s.as_slice::<f32>();
    let sum: f32 = d.iter().sum();
    assert!((sum - 1.0).abs() < 1e-3, "numerical stability: softmax sum = {}", sum);
}

// ===== REDUCTION TESTS =====

#[test]
fn test_sum_axis_0() {
    // [[1,2,3],[4,5,6]] sum along axis 0 = [[5,7,9]]
    let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let s = reduce::sum(&t, 0);
    let d = s.as_slice::<f32>();
    assert!((d[0] - 5.0).abs() < 1e-5);
    assert!((d[1] - 7.0).abs() < 1e-5);
    assert!((d[2] - 9.0).abs() < 1e-5);
}

#[test]
fn test_sum_axis_1() {
    // [[1,2,3],[4,5,6]] sum along axis 1 = [[6],[15]]
    let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let s = reduce::sum(&t, 1);
    let d = s.as_slice::<f32>();
    assert!((d[0] - 6.0).abs() < 1e-5);
    assert!((d[1] - 15.0).abs() < 1e-5);
}

#[test]
fn test_mean_all() {
    let t = Tensor::from_vec(vec![2.0f32, 4.0, 6.0, 8.0], vec![4]);
    let m = reduce::mean_all(&t);
    assert!((m.as_slice::<f32>()[0] - 5.0).abs() < 1e-5);
}

// ===== CREATION TESTS =====

#[test]
fn test_zeros_and_ones() {
    let z = creation::zeros::<f32>(vec![3, 4]);
    assert_eq!(z.numel(), 12);
    assert!(z.as_slice::<f32>().iter().all(|&x| x == 0.0));

    let o = creation::ones::<f32>(vec![3, 4]);
    assert!(o.as_slice::<f32>().iter().all(|&x| x == 1.0));
}

#[test]
fn test_eye_diagonal() {
    let e = creation::eye::<f32>(5);
    let d = e.as_slice::<f32>();
    for i in 0..5 {
        for j in 0..5 {
            if i == j { assert!((d[i*5+j] - 1.0).abs() < 1e-5); }
            else { assert!(d[i*5+j].abs() < 1e-5); }
        }
    }
}

#[test]
fn test_arange() {
    let t = creation::arange(0.0, 10.0, 2.0);
    let d = t.as_slice::<f32>();
    assert_eq!(d, &[0.0, 2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_linspace() {
    let t = creation::linspace(0.0, 1.0, 5);
    let d = t.as_slice::<f32>();
    assert!((d[0] - 0.0).abs() < 1e-5);
    assert!((d[2] - 0.5).abs() < 1e-5);
    assert!((d[4] - 1.0).abs() < 1e-5);
}

#[test]
fn test_randn_shape() {
    let t = creation::randn::<f32>(vec![100, 50]);
    assert_eq!(t.dims(), &[100, 50]);
    assert_eq!(t.numel(), 5000);
}

// ===== NN MODULE TESTS =====

#[test]
fn test_linear_output_range() {
    // With random init, outputs should not be all zeros
    let lin = Linear::new(64, 32, true);
    let x = creation::ones::<f32>(vec![4, 64]);
    let y = lin.forward(&x).unwrap();
    assert_eq!(y.dims(), &[4, 32]);
    let d = y.as_slice::<f32>();
    let sum: f32 = d.iter().map(|x| x.abs()).sum();
    assert!(sum > 0.0, "linear output should not be all zeros");
}

#[test]
fn test_sequential_deep() {
    let seq = Sequential::new()
        .add(Linear::new(16, 32, true))
        .add(ReLU)
        .add(Linear::new(32, 64, true))
        .add(GELU)
        .add(Linear::new(64, 8, true));
    let x = creation::randn::<f32>(vec![5, 16]);
    let y = seq.forward(&x).unwrap();
    assert_eq!(y.dims(), &[5, 8]);
}

#[test]
fn test_dropout_eval_no_effect() {
    let mut drop = Dropout::new(0.5);
    drop.eval();
    let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
    let y = drop.forward(&x).unwrap();
    let d = y.as_slice::<f32>();
    assert!((d[0] - 1.0).abs() < 1e-5);
    assert!((d[3] - 4.0).abs() < 1e-5);
}

#[test]
fn test_layernorm_normalizes() {
    let ln = LayerNorm::new(vec![4], 1e-5);
    let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![1, 4]);
    let y = ln.forward(&x).unwrap();
    let d = y.as_slice::<f32>();
    // After layer norm, mean should be ~0 (within weight/bias)
    let mean: f32 = d.iter().sum::<f32>() / d.len() as f32;
    // With weight=1, bias=0, mean of normalized values ≈ 0
    assert!(mean.abs() < 0.5, "layer norm mean = {}, should be ~0", mean);
}

#[test]
fn test_conv2d_output_shape() {
    let conv = Conv2d::new(3, 16, 3, 3, 1, 1);
    let x = Tensor::from_vec(vec![0.0f32; 2*3*8*8], vec![2, 3, 8, 8]);
    let y = conv.forward(&x).unwrap();
    assert_eq!(y.ndim(), 4);
    assert_eq!(y.dims()[0], 2);
    assert_eq!(y.dims()[1], 16);
}

#[test]
fn test_embedding_lookup() {
    let emb = Embedding::new(100, 32);
    let x = Tensor::from_vec(vec![0.0f32, 5.0, 99.0], vec![3]);
    let y = emb.forward(&x).unwrap();
    assert_eq!(y.dims(), &[3, 32]);
}

#[test]
fn test_positional_encoding_shape() {
    let pe = PositionalEncoding::new(64, 512);
    let x = Tensor::from_vec(vec![0.0f32; 2*10*64], vec![2, 10, 64]);
    let y = pe.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 10, 64]);
    // Output should differ from input (added positional info)
    let x_data = x.as_slice::<f32>();
    let y_data = y.as_slice::<f32>();
    let diff: f32 = x_data.iter().zip(y_data.iter()).map(|(&a, &b)| (a-b).abs()).sum();
    // Since input is all zeros, diff = sum of PE values
    assert!(diff > 0.0, "positional encoding should add non-zero values");
}

#[test]
fn test_attention_forward() {
    let attn = MultiHeadSelfAttention::new(64, 4, true);
    let x = creation::randn::<f32>(vec![2, 8, 64]);
    let y = attn.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 8, 64]);
}

#[test]
fn test_transformer_block_forward() {
    let block = TransformerBlock::new(64, 4, 256, true, 0.1);
    let x = creation::randn::<f32>(vec![2, 8, 64]);
    let y = block.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 8, 64]);
}

// ===== OPTIMIZER TESTS =====

#[test]
fn test_sgd_convergence() {
    // Minimize f(x) = (x-3)^2 with SGD, starting at x=10
    let mut params = vec![Tensor::from_vec(vec![10.0f32], vec![1])];
    let mut opt = SGD::simple(0.1);
    for _ in 0..200 {
        let x = params[0].as_slice::<f32>()[0];
        let grad = 2.0 * (x - 3.0); // df/dx = 2(x-3)
        let grad_t = Tensor::from_vec(vec![grad], vec![1]);
        opt.step(&mut params, &vec![grad_t]);
    }
    let final_val = params[0].as_slice::<f32>()[0];
    assert!((final_val - 3.0).abs() < 0.1, "SGD should converge to 3.0, got {}", final_val);
}

#[test]
fn test_adam_convergence() {
    // Same problem: minimize (x-3)^2
    let mut params = vec![Tensor::from_vec(vec![10.0f32], vec![1])];
    let mut opt = Adam::default(0.1); // higher lr
    for _ in 0..1000 {
        let x = params[0].as_slice::<f32>()[0];
        let grad = 2.0 * (x - 3.0);
        let grad_t = Tensor::from_vec(vec![grad], vec![1]);
        opt.step(&mut params, &vec![grad_t]);
    }
    let final_val = params[0].as_slice::<f32>()[0];
    assert!((final_val - 3.0).abs() < 1.0, "Adam should converge near 3.0, got {}", final_val);
}

#[test]
fn test_adamw_convergence() {
    let mut params = vec![Tensor::from_vec(vec![10.0f32], vec![1])];
    let mut opt = AdamW::default(0.1, 0.001); // higher lr, lower wd
    for _ in 0..1000 {
        let x = params[0].as_slice::<f32>()[0];
        let grad = 2.0 * (x - 3.0);
        let grad_t = Tensor::from_vec(vec![grad], vec![1]);
        opt.step(&mut params, &vec![grad_t]);
    }
    let final_val = params[0].as_slice::<f32>()[0];
    assert!((final_val - 3.0).abs() < 1.0, "AdamW should converge near 3.0, got {}", final_val);
}

// ===== SCHEDULER TESTS =====

#[test]
fn test_cosine_annealing_schedule() {
    let mut sched = CosineAnnealingLR::new(0.1, 100, 0.0);
    let mut lrs = Vec::new();
    for _ in 0..100 { lrs.push(sched.step(None)); }
    // LR should start decreasing from 0.1
    assert!(lrs[0] <= 0.1, "initial LR should be <= base_lr");
    // All LRs should be non-negative
    for &lr in &lrs { assert!(lr >= 0.0, "LR should be non-negative, got {}", lr); }
}

#[test]
fn test_linear_warmup_schedule() {
    let mut sched = LinearWarmup::new(0.001, 0.0, 100, 1000);
    // During warmup, LR should increase
    let lr_1 = sched.step(None);
    let lr_50 = { sched.step(None); sched.step(None); sched.step(None); sched.get_lr() };
    assert!(lr_50 > lr_1, "LR should increase during warmup");
}

// ===== TENSOR MANIPULATION TESTS =====

#[test]
fn test_reshape_and_transpose() {
    let t = Tensor::from_vec(vec![1.0f32; 24], vec![2, 3, 4]);
    let r = t.reshape(vec![6, 4]);
    assert_eq!(r.dims(), &[6, 4]);
    let tt = r.transpose(0, 1);
    assert_eq!(tt.dims(), &[4, 6]);
    let c = tt.contiguous();
    assert!(c.is_contiguous());
    assert_eq!(c.numel(), 24);
}

#[test]
fn test_broadcast_add() {
    // Add a [1,4] tensor to a [3,4] tensor
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![1, 4]);
    let b = Tensor::from_vec(vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0], vec![3, 4]);
    let c = ops::add(&b, &a); // b is bigger, add a to it
    assert_eq!(c.dims(), &[3, 4]);
    let d = c.as_slice::<f32>();
    // First row: 10+1, 20+2, 30+3, 40+4
    assert!((d[0] - 11.0).abs() < 1e-5);
    assert!((d[1] - 22.0).abs() < 1e-5);
}

#[test]
fn test_mul_scalar() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
    let b = ops::mul_scalar(&a, 3.0);
    assert_eq!(b.as_slice::<f32>(), &[3.0, 6.0, 9.0]);
}

#[test]
fn test_add_scalar() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
    let b = ops::add_scalar(&a, 10.0);
    assert_eq!(b.as_slice::<f32>(), &[11.0, 12.0, 13.0]);
}

// ===== STRESS TESTS =====

#[test]
fn test_large_matmul() {
    let a = creation::randn::<f32>(vec![32, 64]);
    let b = creation::randn::<f32>(vec![64, 128]);
    let c = ops::matmul(&a, &b);
    assert_eq!(c.dims(), &[32, 128]);
}

#[test]
fn test_deep_sequential() {
    let mut seq = Sequential::new();
    for _ in 0..5 {
        seq.push(Linear::new(32, 32, true));
        seq.push(ReLU);
    }
    seq.push(Linear::new(32, 10, true));
    let x = creation::randn::<f32>(vec![4, 32]);
    let y = seq.forward(&x).unwrap();
    assert_eq!(y.dims(), &[4, 10]);
}

#[test]
fn test_optimizer_zero_grad() {
    let mut opt = SGD::new(0.01, 0.9, 0.0, false);
    let mut params = vec![Tensor::from_vec(vec![1.0f32, 2.0], vec![2])];
    let grads = vec![Tensor::from_vec(vec![0.5f32, -0.5], vec![2])];
    opt.step(&mut params, &grads);
    opt.zero_grad();
    // After zero_grad, running another step should still work
    let grads2 = vec![Tensor::from_vec(vec![0.1f32, 0.1], vec![2])];
    opt.step(&mut params, &grads2);
    assert!(params[0].as_slice::<f32>()[0] < 1.0);
}
