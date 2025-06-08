#![no_main]

use libfuzzer_sys::fuzz_target;
use moxcms::{
    f_acosf, f_asinf, f_atanf, f_cbrt, f_cbrtf, f_cosf, f_coshf, f_exp, f_exp2, f_exp2f, f_exp10,
    f_exp10f, f_expf, f_log, f_log2, f_log2f, f_log10, f_logf, f_sinf, f_sinhf, f_tanf, log10f,
};

fuzz_target!(|data: u64| {
    let lo = data.to_ne_bytes();

    let z_f32 = f32::from_bits(u32::from_ne_bytes([lo[0], lo[1], lo[2], lo[3]]));
    let z_f64 = f64::from_bits(data);

    _ = f_cbrtf(z_f32);
    _ = f_cbrt(z_f64);
    _ = f_atanf(z_f32);
    _ = f_cosf(z_f32);
    _ = f_exp(z_f64);
    _ = f_exp2(z_f64);
    _ = f_exp2f(z_f32);
    _ = f_exp10(z_f64);
    _ = f_exp10f(z_f32);
    _ = f_expf(z_f32);
    _ = f_log(z_f64);
    _ = f_log2(z_f64);
    _ = f_log10(z_f64);
    _ = f_logf(z_f32);
    _ = f_log2f(z_f32);
    _ = log10f(z_f32);
    _ = f_cosf(z_f32);
    _ = f_sinf(z_f32);
    _ = f_tanf(z_f32);
    _ = f_coshf(z_f32);
    _ = f_sinhf(z_f32);
    _ = f_acosf(z_f32);
    _ = f_asinf(z_f32);
});
