/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use moxcms::{
    atanf, cbrtf, cosf, exp, expf, f_atanf, f_cbrtf, f_cosf, f_exp, f_exp2, f_log, f_log2, f_log10,
    f_logf, f_pow, f_powf, f_sinf, log, logf, pow, powf, sinf,
};

pub fn criterion_benchmark(c: &mut Criterion) {
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

    /*c.bench_function("libm::exp2", |b| {
            b.iter(|| {
                for i in 1..10000 {
                    black_box(libm::exp2(i as f64 / 10000.0 - 1.));
                }
            })
        });

        c.bench_function("system::exp2", |b| {
            b.iter(|| {
                for i in 1..10000 {
                    black_box(f64::exp2(i as f64 / 10000.0 - 1.));
                }
            })
        });

        c.bench_function("moxcms::exp2", |b| {
            b.iter(|| {
                for i in 1..10000 {
                    black_box(f_exp2(i as f64 / 10000.0 - 1.));
                }
            })
        });

        c.bench_function("moxcms::exp", |b| {
            b.iter(|| {
                for i in 1..10000 {
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

        c.bench_function("moxcms: cbrtf", |b| {
            b.iter(|| {
                for i in 1..1000 {
                    black_box(cbrtf(i as f32));
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

        c.bench_function("moxcms: cosf", |b| {
            b.iter(|| {
                for i in 1..1000 {
                    black_box(cosf(i as f32));
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

        c.bench_function("moxcms: sinf", |b| {
            b.iter(|| {
                for i in 1..1000 {
                    black_box(sinf(i as f32));
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

        c.bench_function("moxcms: expf", |b| {
            b.iter(|| {
                for i in 1..1000 {
                    black_box(expf(i as f32));
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
    */
    c.bench_function("libm::log", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(libm::log(i as f64));
            }
        })
    });

    c.bench_function("moxcms: log", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(log(i as f64));
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

    c.bench_function("moxcms: logf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(logf(i as f32));
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

    c.bench_function("moxcms: atanf", |b| {
        b.iter(|| {
            for i in 1..1000 {
                black_box(atanf(i as f32));
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
