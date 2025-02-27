/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use moxcms::{atanf, cbrtf, cosf, expf, logf, pow, powf, sinf};

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("libm::cbrtf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(libm::cbrtf(i as f32));
            }
        })
    });

    c.bench_function("moxcms: cbrtf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(cbrtf(i as f32));
            }
        })
    });

    c.bench_function("libm::cosf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(libm::cosf(i as f32));
            }
        })
    });

    c.bench_function("moxcms: cosf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(cosf(i as f32));
            }
        })
    });

    c.bench_function("libm::sinf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(libm::sinf(i as f32));
            }
        })
    });

    c.bench_function("moxcms: sinf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(sinf(i as f32));
            }
        })
    });

    c.bench_function("libm::expf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(libm::expf(i as f32));
            }
        })
    });

    c.bench_function("moxcms: expf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(expf(i as f32));
            }
        })
    });

    c.bench_function("libm::logf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(libm::logf(i as f32));
            }
        })
    });

    c.bench_function("moxcms: logf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(logf(i as f32));
            }
        })
    });

    c.bench_function("libm::powf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(libm::powf(i as f32, 0.323221324312f32 * i as f32));
            }
        })
    });

    c.bench_function("moxcms: powf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(powf(i as f32, 0.323221324312f32 * i as f32));
            }
        })
    });

    c.bench_function("libm::pow", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(libm::pow(i as f64, 0.323221324312f64 * i as f64));
            }
        })
    });

    c.bench_function("moxcms: pow", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(pow(i as f64, 0.323221324312f64 * i as f64));
            }
        })
    });

    c.bench_function("libm::atanf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(libm::atanf(i as f32));
            }
        })
    });

    c.bench_function("moxcms: atanf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(atanf(i as f32));
            }
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
