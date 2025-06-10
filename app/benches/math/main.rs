/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use criterion::{Criterion, criterion_group, criterion_main};
use moxcms::{
    exp, f_acos, f_acosf, f_asin, f_asinf, f_atan, f_atan2, f_atan2f, f_atanf, f_cbrt, f_cbrtf,
    f_cos, f_cosf, f_coshf, f_exp, f_exp2, f_exp2f, f_exp10f, f_expf, f_log, f_log2, f_log2f,
    f_log10, f_logf, f_pow, f_powf, f_sin, f_sincos, f_sincosf, f_sinf, f_sinhf, f_tan, f_tanf,
    f_tanhf, log10f, powf,
};
use std::hint::black_box;

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("libm::atan2", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(libm::atan2(i as f64 / 1000.0, i as f64 / 1000.0 + 0.5));
            }
        })
    });

    c.bench_function("system: atan2", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f64::atan2(i as f64 / 1000.0, i as f64 / 1000.0 + 0.5));
            }
        })
    });

    c.bench_function("moxcms: atan2", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f_atan2(i as f64 / 1000.0, i as f64 / 1000.0 + 0.5));
            }
        })
    });

    c.bench_function("libm::atan2f", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(libm::atan2f(i as f32 / 1000.0, i as f32 / 1000.0 + 0.5));
            }
        })
    });

    c.bench_function("system: atan2f", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f32::atan2(i as f32 / 1000.0, i as f32 / 1000.0 + 0.5));
            }
        })
    });

    c.bench_function("moxcms: atan2f", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f_atan2f(i as f32 / 1000.0, i as f32 / 1000.0 + 0.5));
            }
        })
    });

    c.bench_function("libm::acos", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(libm::acos(i as f64 / 1000.0));
            }
        })
    });

    c.bench_function("system: acos", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f64::acos(i as f64 / 1000.0));
            }
        })
    });

    c.bench_function("moxcms: acos", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f_acos(i as f64 / 1000.0));
            }
        })
    });

    c.bench_function("libm::sincos", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(libm::sincos(i as f64 * 1000.0));
            }
        })
    });

    c.bench_function("system: sincos", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f64::sin_cos(i as f64 * 1000.0));
            }
        })
    });

    c.bench_function("moxcms: FMA sincos", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f_sincos(i as f64 * 1000.0));
            }
        })
    });

    c.bench_function("libm::tan", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(libm::tan(i as f64 * 1000.0));
            }
        })
    });

    c.bench_function("system: tan", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f64::tan(i as f64 * 1000.0));
            }
        })
    });

    c.bench_function("moxcms: FMA tan", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f_tan(i as f64 * 1000.0));
            }
        })
    });

    c.bench_function("libm::sin", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(libm::sin(i as f64 * 1000.0));
            }
        })
    });

    c.bench_function("system: sin", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f64::sin(i as f64 * 1000.0));
            }
        })
    });

    c.bench_function("moxcms: FMA sin", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f_sin(i as f64 * 1000.0));
            }
        })
    });

    c.bench_function("libm::cos", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(libm::cos(i as f64 * 1000.0));
            }
        })
    });

    c.bench_function("system: cos", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f64::cos(i as f64 * 1000.0));
            }
        })
    });

    c.bench_function("moxcms: FMA cos", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f_cos(i as f64 * 1000.0));
            }
        })
    });

    c.bench_function("libm::sincosf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(libm::sincosf(i as f32));
            }
        })
    });

    c.bench_function("system: sincosf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f32::sin_cos(i as f32));
            }
        })
    });

    c.bench_function("moxcms: FMA sincosf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f_sincosf(i as f32));
            }
        })
    });

    c.bench_function("libm::tanf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(libm::tanf(i as f32 / 10000.0 - 1.));
            }
        })
    });

    c.bench_function("system::tanf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f32::tan(i as f32 / 10000.0 - 1.));
            }
        })
    });

    c.bench_function("moxcms::tanf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f_tanf(i as f32 / 10000.0 - 1.));
            }
        })
    });

    c.bench_function("libm::cbrt", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(libm::cbrt(i as f64));
            }
        })
    });

    c.bench_function("system: cbrt", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f64::cbrt(i as f64));
            }
        })
    });

    c.bench_function("moxcms: FMA cbrt", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f_cbrt(i as f64));
            }
        })
    });

    c.bench_function("libm::log10", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(libm::log10(i as f64));
            }
        })
    });

    c.bench_function("system: log10", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f64::log10(i as f64));
            }
        })
    });

    c.bench_function("moxcms: FMA log10", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f_log10(i as f64));
            }
        })
    });

    c.bench_function("libm::log10f", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(libm::log10f(i as f32));
            }
        })
    });

    c.bench_function("system: log10f", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f32::log10(i as f32));
            }
        })
    });

    c.bench_function("moxcms: FMA log10f", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(log10f(i as f32));
            }
        })
    });

    c.bench_function("libm::exp10f", |b| {
        b.iter(|| {
            for i in 1..10000 {
                black_box(libm::exp10f(i as f32 / 10000.0 - 1.));
            }
        })
    });

    c.bench_function("moxcms::exp10f", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f_exp10f(i as f32 / 10000.0 - 1.));
            }
        })
    });

    c.bench_function("libm::exp2f", |b| {
        b.iter(|| {
            for i in 1..10000 {
                black_box(libm::exp2f(i as f32 / 10000.0 - 1.));
            }
        })
    });

    c.bench_function("system::exp2f", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f32::exp2(i as f32 / 10000.0 - 1.));
            }
        })
    });

    c.bench_function("moxcms::exp2f", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f_exp2f(i as f32 / 10000.0 - 1.));
            }
        })
    });

    c.bench_function("libm::exp2", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(libm::exp2(i as f64 / 10000.0 - 1.));
            }
        })
    });

    c.bench_function("system::exp2", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f64::exp2(i as f64 / 10000.0 - 1.));
            }
        })
    });

    c.bench_function("moxcms::exp2", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f_exp2(i as f64 / 10000.0 - 1.));
            }
        })
    });

    c.bench_function("system::exp", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f64::exp(i as f64 / 10000.0 - 1.));
            }
        })
    });

    c.bench_function("moxcms::exp", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f_exp(i as f64 / 10000.0 - 1.));
            }
        })
    });

    c.bench_function("libm::cbrtf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(libm::cbrtf(i as f32));
            }
        })
    });

    c.bench_function("system: cbrtf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f32::cbrt(i as f32));
            }
        })
    });

    c.bench_function("moxcms: FMA cbrtf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f_cbrtf(i as f32));
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

    c.bench_function("system: cosf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f32::cos(i as f32));
            }
        })
    });

    c.bench_function("moxcms: FMA cosf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f_cosf(i as f32));
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

    c.bench_function("system: sinf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f32::sin(i as f32));
            }
        })
    });

    c.bench_function("moxcms: FMA sinf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f_sinf(i as f32));
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

    c.bench_function("system: expf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f32::exp(i as f32));
            }
        })
    });

    c.bench_function("moxcms: FMA expf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f_expf(i as f32));
            }
        })
    });

    c.bench_function("libm::exp", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(libm::exp(i as f64));
            }
        })
    });

    c.bench_function("moxcms: FMA exp", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f_exp(i as f64));
            }
        })
    });

    c.bench_function("moxcms: exp", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(exp(i as f64));
            }
        })
    });

    c.bench_function("libm::asinf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(libm::asinf(i as f32 / 1000.0));
            }
        })
    });

    c.bench_function("system::asinf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f32::asin(i as f32 / 1000.0));
            }
        })
    });

    c.bench_function("moxcms: FMA asinf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f_asinf(i as f32 / 1000.0));
            }
        })
    });

    c.bench_function("libm::acosf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(libm::acosf(i as f32 / 1000.0));
            }
        })
    });

    c.bench_function("system::acosf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f32::acos(i as f32 / 1000.0));
            }
        })
    });

    c.bench_function("moxcms: FMA acosf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f_acosf(i as f32 / 1000.0));
            }
        })
    });

    c.bench_function("libm::tanhf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(libm::tanhf(i as f32));
            }
        })
    });

    c.bench_function("system::tanhf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f32::tanh(i as f32));
            }
        })
    });

    c.bench_function("moxcms: FMA tanhf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f_tanhf(i as f32));
            }
        })
    });

    c.bench_function("libm::sinhf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(libm::sinhf(i as f32));
            }
        })
    });

    c.bench_function("system::sinhf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f32::sinh(i as f32));
            }
        })
    });

    c.bench_function("moxcms: FMA sinhf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f_sinhf(i as f32));
            }
        })
    });

    c.bench_function("libm::coshf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(libm::coshf(i as f32));
            }
        })
    });

    c.bench_function("system::coshf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f32::cosh(i as f32));
            }
        })
    });

    c.bench_function("moxcms: FMA coshf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f_coshf(i as f32));
            }
        })
    });

    c.bench_function("libm::log2f", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(libm::log2f(i as f32));
            }
        })
    });

    c.bench_function("system::log2f", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f32::log2(i as f32));
            }
        })
    });

    c.bench_function("moxcms: FMA log2f", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f_log2f(i as f32));
            }
        })
    });

    c.bench_function("libm::log2", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(libm::log2(i as f64));
            }
        })
    });

    c.bench_function("system::log2", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f64::log2(i as f64));
            }
        })
    });

    c.bench_function("moxcms: FMA log2", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f_log2(i as f64));
            }
        })
    });

    c.bench_function("libm::log", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(libm::log(i as f64));
            }
        })
    });

    c.bench_function("system: log", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f64::ln(i as f64));
            }
        })
    });

    c.bench_function("moxcms: FMA log", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f_log(i as f64));
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

    c.bench_function("system::logf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box((i as f32).ln());
            }
        })
    });

    c.bench_function("moxcms: FMA logf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f_logf(i as f32));
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

    c.bench_function("system: powf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f32::powf(i as f32, 0.323221324312f32 * i as f32));
            }
        })
    });

    c.bench_function("moxcms: FMA powf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f_powf(i as f32, 0.323221324312f32 * i as f32));
            }
        })
    });

    c.bench_function("libm::asin", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(libm::asin(i as f64 / 100.0));
            }
        })
    });

    c.bench_function("system: asin", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f64::asin(i as f64 / 100.0));
            }
        })
    });

    c.bench_function("moxcms: FMA asin", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f_asin(i as f64 / 100.0));
            }
        })
    });

    c.bench_function("libm::atan", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(libm::atan(i as f64 / 100.0));
            }
        })
    });

    c.bench_function("system: atan", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f64::atan(i as f64 / 100.0));
            }
        })
    });

    c.bench_function("moxcms: FMA atan", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f_atan(i as f64 / 100.0));
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

    c.bench_function("system: pow", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f64::powf(i as f64, 0.323221324312f64 * i as f64));
            }
        })
    });

    c.bench_function("moxcms: FMA pow", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f_pow(i as f64, 0.323221324312f64 * i as f64));
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

    c.bench_function("system: atanf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f32::atan(i as f32));
            }
        })
    });

    c.bench_function("moxcms: FMA atanf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(f_atanf(i as f32));
            }
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
