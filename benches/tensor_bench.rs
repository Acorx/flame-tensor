//! Tensor operation benchmarks.

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use flame_tensor::Tensor;
use flame_tensor::tensor::ops;

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    for size in [64, 256].iter() {
        group.bench_with_input(BenchmarkId::new("f32", size), size, |b, &sz| {
            let a = Tensor::from_vec(vec![1.0f32; sz * sz], vec![sz, sz]);
            let b = Tensor::from_vec(vec![1.0f32; sz * sz], vec![sz, sz]);
            b.iter(|| ops::matmul(&a, &b));
        });
    }
    group.finish();
}

fn bench_elementwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise");
    let n = 1024 * 1024;
    let a = Tensor::from_vec(vec![1.0f32; n], vec![n]);
    let b = Tensor::from_vec(vec![2.0f32; n], vec![n]);

    group.bench_function("add_1M", |bencher| {
        bencher.iter(|| ops::add(&a, &b));
    });

    group.bench_function("mul_1M", |bencher| {
        bencher.iter(|| ops::mul(&a, &b));
    });
    group.finish();
}

fn bench_reduce(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduce");
    let t = Tensor::from_vec(vec![1.0f32; 1024 * 1024], vec![1024, 1024]);

    group.bench_function("sum_all_1M", |bencher| {
        bencher.iter(|| flame_tensor::tensor::reduce::sum_all(&t));
    });
    group.finish();
}

criterion_group!(benches, bench_matmul, bench_elementwise, bench_reduce);
criterion_main!(benches);
