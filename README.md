# flame-tensor

> *Le nouveau torch mais en rust et incroyablement efficace.*

A full tensor framework in Rust — multi-precision, tape-based autodiff, neural network modules, optimizers, and safetensors serialization.

## Features

- **Multi-precision**: f16, bf16, f32, f64 (via `half` crate)
- **COW Storage**: Copy-on-write `Arc<Vec<u8>>` with zero-copy strided views
- **Tape-based Autodiff**: Reverse-mode AD with linear tape replay — no graph traversal overhead, no reference cycles
- **Neural Network Modules**: Linear, Conv1d/2d, Transformer, GPT-2, LayerNorm, Embedding, Sequential, and more
- **Optimizers**: SGD (momentum/nesterov), Adam, AdamW (decoupled weight decay), LR schedulers
- **Safetensors**: Read/write the safetensors format for PyTorch/HuggingFace interoperability
- **CPU Backend**: Rayon-parallelized operations with SIMD hints
- **CUDA Stub**: Feature-gated, ready for kernel integration

## Quick Start

```rust
use flame_tensor::{Tensor, DType};
use flame_tensor::tensor::ops;
use flame_tensor::nn::{Linear, ReLU, Sequential, LayerNorm};
use flame_tensor::nn::module::Module;
use flame_tensor::optim::{AdamW, Optimizer};

// Create tensors
let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
let y = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]);

// Element-wise ops
let z = ops::add(&x, &y);
let z = ops::matmul(&x, &y);
let z = ops::relu(&x);

// Build a model
let model = Sequential::new()
    .add(Linear::new(64, 128, true))
    .add(ReLU)
    .add(Linear::new(128, 10, true));

let input = Tensor::from_vec(vec![0.0f32; 64], vec![1, 64]);
let output = model.forward(&input).unwrap(); // [1, 10]

// Optimize
let mut optimizer = AdamW::default(0.001, 0.01);
optimizer.step(&mut params, &grads);
```

## Architecture

```
src/
├── tensor/        # Core: DType, Storage (COW), Views, Ops, Broadcast, Reduce, Index, Creation
├── autodiff/      # Tape-based AD: Tape, Var, differentiable Ops, Gradient
├── nn/            # Neural network: Module trait, Linear, Conv, Attention, Transformer, Norm, etc.
├── optim/         # Optimizers: SGD, Adam, AdamW + LR Schedulers
├── serialize/     # Safetensors format read/write
└── backend/       # CPU (rayon) + CUDA (stub)
```

### Design Principles

1. **Tape over Graph**: A single tape per forward pass. Each op pushes a `FnMut()` closure. Backward plays the tape in reverse. No Rc cycles, no traversal overhead.
2. **COW Storage**: `Arc<Vec<u8>>` shared byte storage. Mutation clones only when refcount > 1.
3. **Zero-copy Views**: Strided views share the same storage — slicing, transposing, and permuting are O(1).
4. **Safetensors First**: Native interop with the Python ML ecosystem. Load HuggingFace weights directly.

## Benchmarks

```bash
cargo bench
```

## Testing

```bash
cargo test          # 116 tests
cargo test --test advanced_tests  # Deep integration tests
```

## Roadmap (v2)

- [ ] Full scaled dot-product attention with batched matmul
- [ ] CUDA kernels (via cuda-open integration)
- [ ] Lazy evaluation / kernel fusion
- [ ] Arena allocation for training loops
- [ ] bf16/f16 compute paths (not just storage)
- [ ] Python bindings via PyO3
- [ ] Distributed training (NCCL)
- [ ] Model zoo: ResNet, BERT, LLaMA

## License

MIT
