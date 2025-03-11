/*
 * // Copyright (c) Radzivon Bartoshyk 2/2025. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#![allow(clippy::approx_constant)]

use num_traits::Num;

#[inline(always)]
const fn halley_refine_f(x: f32, a: f32) -> f32 {
    let tx = x * x * x;
    x * (tx + 2f32 * a) / (2f32 * tx + a)
}

/// Computes Cube Root
#[inline]
pub const fn cbrtf(x: f32) -> f32 {
    if x == 0. {
        return x;
    }
    if x == f32::INFINITY {
        return f32::INFINITY;
    }
    if x == f32::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }

    const B1: u32 = 709958130;
    let mut t: f32;
    let mut ui: u32 = x.to_bits();
    let mut hx: u32 = ui & 0x7fffffff;

    hx = hx / 3 + B1;
    ui &= 0x80000000;
    ui |= hx;

    t = f32::from_bits(ui);
    t = halley_refine_f(t, x);
    halley_refine_f(t, x)
}

const PI_A2: f32 = 3.1414794921875f32;
const PI_B2: f32 = 0.00011315941810607910156f32;
const PI_C2: f32 = 1.9841872589410058936e-09f32;

#[inline]
const fn rintfk(x: f32) -> f32 {
    (if x < 0. { x - 0.5 } else { x + 0.5 }) as i32 as f32
}

#[inline(always)]
const fn fmlaf(a: f32, b: f32, c: f32) -> f32 {
    c + a * b
}

#[inline(always)]
const fn fmla(a: f64, b: f64, c: f64) -> f64 {
    c + a * b
}

#[inline]
const fn isnegzerof(x: f32) -> bool {
    x.to_bits() == (-0.0f32).to_bits()
}

/// Computes cosine for given value
#[inline]
pub const fn cosf(d: f32) -> f32 {
    let q = 1 + 2 * rintfk(std::f32::consts::FRAC_1_PI * d - 0.5) as i32;
    let qf = q as f32;
    let mut r = fmlaf(qf, -PI_A2 * 0.5, d);
    r = fmlaf(qf, -PI_B2 * 0.5, r);
    r = fmlaf(qf, -PI_C2 * 0.5, r);

    let x2 = r * r;

    if q & 2 == 0 {
        r = -r;
    }

    let mut u = 2.6083159809786593541503e-06f32;
    u = fmlaf(u, x2, -0.0001981069071916863322258f32);
    u = fmlaf(u, x2, 0.00833307858556509017944336f32);
    u = fmlaf(u, x2, -0.166666597127914428710938f32);
    u = fmlaf(u, x2 * r, r);
    if isnegzerof(d) {
        return -0.;
    }
    u
}

/// Sine function
#[inline]
pub const fn sinf(d: f32) -> f32 {
    let qf = rintfk(std::f32::consts::FRAC_1_PI * d);
    let q = qf as i32;
    let mut r = fmlaf(qf, -PI_A2, d);
    r = fmlaf(qf, -PI_B2, r);
    r = fmlaf(qf, -PI_C2, r);

    let x2 = r * r;

    if (q & 1) != 0 {
        r = -r;
    }

    let mut u = 2.6083159809786593541503e-06f32;
    u = fmlaf(u, x2, -0.0001981069071916863322258f32);
    u = fmlaf(u, x2, 0.00833307858556509017944336f32);
    u = fmlaf(u, x2, -0.166666597127914428710938f32);
    u = fmlaf(u, x2 * r, r);
    if isnegzerof(d) {
        return -0f32;
    }
    u
}

const L2U_F: f32 = 0.693_145_751_953_125;
const L2L_F: f32 = 1.428_606_765_330_187_045_e-6;
const R_LN2_F: f32 = std::f32::consts::LOG2_E;

#[inline]
const fn pow2if(q: i32) -> f32 {
    f32::from_bits(((q + 0x7f) as u32) << 23)
}

/// Computes exponent for given value
#[inline]
pub const fn expf(d: f32) -> f32 {
    const EXP_POLY_1_S: f32 = 2f32;
    const EXP_POLY_2_S: f32 = 0.16666707f32;
    const EXP_POLY_3_S: f32 = -0.002775669f32;
    const EXP_POLY_4_S: f32 = 6.6094115e-5f32;
    const EXP_POLY_5_S: f32 = 1.6546869e-6f32;
    let qf = rintfk(d * R_LN2_F);
    let q = qf as i32;
    let r = fmlaf(qf, -L2U_F, d);
    let r = fmlaf(qf, -L2L_F, r);

    let f = r * r;
    // Poly for u = r*(exp(r)+1)/(exp(r)-1)
    let mut u = EXP_POLY_5_S;
    u = fmlaf(u, f, EXP_POLY_4_S);
    u = fmlaf(u, f, EXP_POLY_3_S);
    u = fmlaf(u, f, EXP_POLY_2_S);
    u = fmlaf(u, f, EXP_POLY_1_S);
    let u = 1f32 + 2f32 * r / (u - r);
    let i2 = pow2if(q);
    let mut r = u * i2;
    if d < -87f32 {
        r = 0f32;
    }
    if d > 88f32 {
        r = f32::INFINITY;
    }
    r
}

#[inline]
// Founds n in ln(𝑥)=ln(𝑎)+𝑛ln(2)
const fn ilogb2kf(d: f32) -> i32 {
    (((d.to_bits() as i32) >> 23) & 0xff) - 0x7f
}

#[inline]
// Founds a in x=a+𝑛ln(2)
const fn ldexp3kf(d: f32, n: i32) -> f32 {
    f32::from_bits(((d.to_bits() as i32) + (n << 23)) as u32)
}

/// Natural logarithm
#[inline]
pub const fn logf(d: f32) -> f32 {
    const LN_POLY_1_F: f32 = 2f32;
    const LN_POLY_2_F: f32 = 0.6666677f32;
    const LN_POLY_3_F: f32 = 0.40017125f32;
    const LN_POLY_4_F: f32 = 0.28523374f32;
    const LN_POLY_5_F: f32 = 0.23616748f32;
    // ln(𝑥)=ln(𝑎)+𝑛ln(2)
    let n = ilogb2kf(d * (1. / 0.75));
    let a = ldexp3kf(d, -n);

    let x = (a - 1.) / (a + 1.);
    let x2 = x * x;
    let mut u = LN_POLY_5_F;
    u = fmlaf(u, x2, LN_POLY_4_F);
    u = fmlaf(u, x2, LN_POLY_3_F);
    u = fmlaf(u, x2, LN_POLY_2_F);
    u = fmlaf(u, x2, LN_POLY_1_F);
    if d == 0f32 {
        f32::NEG_INFINITY
    } else if (d < 0.) || d.is_nan() {
        f32::NAN
    } else if d.is_infinite() {
        f32::INFINITY
    } else {
        x * u + std::f32::consts::LN_2 * (n as f32)
    }
}

/// Copies sign from `y` to `x`
#[inline]
const fn copysignfk(x: f32, y: f32) -> f32 {
    f32::from_bits((x.to_bits() & !(1 << 31)) ^ (y.to_bits() & (1 << 31)))
}

/// Round to integer towards minus infinity
#[inline]
pub const fn floorf(x: f32) -> f32 {
    const F1_23: f32 = (1u32 << 23) as f32;
    let mut fr = x - (x as i32 as f32);
    fr = if fr < 0. { fr + 1. } else { fr };
    if x.is_infinite() || (x.abs() >= F1_23) {
        x
    } else {
        copysignfk(x - fr, x)
    }
}

/// Power function for given value
#[inline]
pub const fn powf(d: f32, n: f32) -> f32 {
    let value = d.abs();
    let mut c = expf(n * logf(value));
    c = copysignfk(c, d);
    if d < 0. && floorf(n) != n {
        return f32::NAN;
    }
    if n == f32::INFINITY || d.is_infinite() {
        f32::INFINITY
    } else if n == f32::NEG_INFINITY {
        0f32
    } else if n.is_nan() || d.is_nan() {
        f32::NAN
    } else {
        c
    }
}

/// Round towards whole integral number
#[inline]
const fn rintk(x: f64) -> f64 {
    (if x < 0. { x - 0.5 } else { x + 0.5 }) as i64 as f64
}

/// Computes 2^n
#[inline(always)]
const fn pow2i(q: i32) -> f64 {
    f64::from_bits(((q + 0x3ff) as u64) << 52)
}

/// Computes exponent for given value
#[inline]
pub const fn exp(d: f64) -> f64 {
    const EXP_POLY_1_D: f64 = 2f64;
    const EXP_POLY_2_D: f64 = 0.16666666666666674f64;
    const EXP_POLY_3_D: f64 = -0.0027777777777777614f64;
    const EXP_POLY_4_D: f64 = 6.613756613755705e-5f64;
    const EXP_POLY_5_D: f64 = -1.6534391534392554e-6f64;
    const EXP_POLY_6_D: f64 = 4.17535139757361979584e-8f64;
    const EXP_POLY_7_D: f64 = -1.05683802773749863697e-9f64;
    const EXP_POLY_8_D: f64 = 2.67650730613693576657e-11f64;
    const EXP_POLY_9_D: f64 = 1.71721241125556891283e-14;
    const EXP_POLY_10_D: f64 = -6.77936059264516573366e-13f64;

    const L2_U: f64 = 0.693_147_180_559_662_956_511_601_805_686_950_683_593_75;
    const L2_L: f64 = 0.282_352_905_630_315_771_225_884_481_750_134_360_255_254_120_68_e-12;
    const R_LN2: f64 =
        1.442_695_040_888_963_407_359_924_681_001_892_137_426_645_954_152_985_934_135_449_406_931;

    let qf = rintk(d * R_LN2);
    let q = qf as i32;

    let mut r = fmla(qf, -L2_U, d);
    r = fmla(qf, -L2_L, r);

    let f = r * r;
    // Poly for u = r*(exp(r)+1)/(exp(r)-1)
    let mut u = EXP_POLY_10_D;
    u = fmla(u, f, EXP_POLY_9_D);
    u = fmla(u, f, EXP_POLY_8_D);
    u = fmla(u, f, EXP_POLY_7_D);
    u = fmla(u, f, EXP_POLY_6_D);
    u = fmla(u, f, EXP_POLY_5_D);
    u = fmla(u, f, EXP_POLY_4_D);
    u = fmla(u, f, EXP_POLY_3_D);
    u = fmla(u, f, EXP_POLY_2_D);
    u = fmla(u, f, EXP_POLY_1_D);
    let u = 1f64 + 2f64 * r / (u - r);
    let i2 = pow2i(q);
    let mut r = u * i2;
    if d < -964f64 {
        r = 0f64;
    }
    if d > 709f64 {
        r = f64::INFINITY;
    }
    r
}

#[inline]
const fn ilogb2k(d: f64) -> i32 {
    (((d.to_bits() >> 52) & 0x7ff) as i32) - 0x3ff
}

#[inline]
const fn ldexp3k(d: f64, e: i32) -> f64 {
    f64::from_bits(((d.to_bits() as i64) + ((e as i64) << 52)) as u64)
}

/// Natural logarithm
#[inline]
pub const fn log(d: f64) -> f64 {
    const LN_POLY_1_D: f64 = 2.;
    const LN_POLY_2_D: f64 = 0.666_666_666_666_777_874_006_3;
    const LN_POLY_3_D: f64 = 0.399_999_999_950_799_600_689_777;
    const LN_POLY_4_D: f64 = 0.285_714_294_746_548_025_383_248;
    const LN_POLY_5_D: f64 = 0.222_221_366_518_767_365_905_163;
    const LN_POLY_6_D: f64 = 0.181_863_266_251_982_985_677_316;
    const LN_POLY_7_D: f64 = 0.152_519_917_006_351_951_593_857;
    const LN_POLY_8_D: f64 = 0.153_487_338_491_425_068_243_146;

    // ln(𝑥)=ln(𝑎)+𝑛ln(2)
    let n = ilogb2k(d * (1. / 0.75));
    let a = ldexp3k(d, -n);

    let x = (a - 1.) / (a + 1.);
    let x2 = x * x;
    let mut u = LN_POLY_8_D;
    u = fmla(u, x2, LN_POLY_7_D);
    u = fmla(u, x2, LN_POLY_6_D);
    u = fmla(u, x2, LN_POLY_5_D);
    u = fmla(u, x2, LN_POLY_4_D);
    u = fmla(u, x2, LN_POLY_3_D);
    u = fmla(u, x2, LN_POLY_2_D);
    u = fmla(u, x2, LN_POLY_1_D);

    if d == 0f64 {
        f64::NEG_INFINITY
    } else if (d < 0.) || d.is_nan() {
        f64::NAN
    } else if d.is_infinite() {
        f64::INFINITY
    } else {
        x * u + std::f64::consts::LN_2 * (n as f64)
    }
}

/// Copies sign from `y` to `x`
#[inline]
const fn copysignk(x: f64, y: f64) -> f64 {
    f64::from_bits((x.to_bits() & !(1 << 63)) ^ (y.to_bits() & (1 << 63)))
}

/// Floors value
#[inline]
pub const fn floor(x: f64) -> f64 {
    const D1_31: f64 = (1u64 << 31) as f64;
    const D1_52: f64 = (1u64 << 52) as f64;
    let mut fr = x - D1_31 * ((x * (1. / D1_31)) as i32 as f64);
    fr -= fr as i32 as f64;
    fr = if fr < 0. { fr + 1. } else { fr };
    if x.is_infinite() || (x.abs() >= D1_52) {
        x
    } else {
        copysignk(x - fr, x)
    }
}

/// Power function for given value
#[inline]
pub const fn pow(d: f64, n: f64) -> f64 {
    let value = d.abs();
    let mut c = exp(n * log(value));
    c = copysignk(c, d);
    if d < 0. && floor(n) != n {
        return f64::NAN;
    }
    if n == f64::INFINITY || d.is_infinite() {
        f64::INFINITY
    } else if n == f64::NEG_INFINITY {
        0f64
    } else if n.is_nan() || d.is_nan() {
        f64::NAN
    } else {
        c
    }
}

/// Computes Square root.
/// Most of CPU have built-in instruction with higher precision,
/// prefer use this only for const contexts.
#[inline]
pub const fn sqrtf(d: f32) -> f32 {
    let mut q = 1.0f32;

    let mut d = if d < 0f32 { f32::NAN } else { d };

    if d < 5.2939559203393770e-23f32 {
        d *= 1.8889465931478580e+22f32;
        q = 7.2759576141834260e-12f32;
    }

    if d > 1.8446744073709552e+19f32 {
        d *= 5.4210108624275220e-20f32;
        q = 4294967296.0f32;
    }

    // http://en.wikipedia.org/wiki/Fast_inverse_square_root
    let mut x = f32::from_bits(0x5f375a86 - ((d + 1e-45).to_bits() >> 1));

    x = x * (1.5f32 - 0.5f32 * d * x * x);
    x = x * (1.5f32 - 0.5f32 * d * x * x);
    x = x * (1.5f32 - 0.5f32 * d * x * x);
    x = x * (1.5f32 - 0.5f32 * d * x * x);

    if d.is_infinite() {
        return f32::INFINITY;
    }
    x * d * q
}

/// Hypot suitable for const context
#[inline]
pub const fn const_hypotf(x: f32, y: f32) -> f32 {
    let x = x.abs();
    let y = y.abs();
    let max = x.max(y);
    let min = x.min(y);
    let r = min / max;
    let ret = max * sqrtf(1f32 + r * r);

    if (x == f32::INFINITY) || (y == f32::INFINITY) {
        f32::INFINITY
    } else if x.is_nan() || y.is_nan() || ret.is_nan() {
        f32::NAN
    } else if min == 0. {
        max
    } else {
        ret
    }
}

/// Hypot function
#[inline]
pub fn hypotf(x: f32, y: f32) -> f32 {
    let x = x.abs();
    let y = y.abs();
    let max = x.max(y);
    let min = x.min(y);
    let r = min / max;
    let ret = max * (1f32 + r * r).sqrt();

    if (x == f32::INFINITY) || (y == f32::INFINITY) {
        f32::INFINITY
    } else if x.is_nan() || y.is_nan() || ret.is_nan() {
        f32::NAN
    } else if min == 0. {
        max
    } else {
        ret
    }
}

/// Computes Atan
#[inline]
pub const fn atanf(d: f32) -> f32 {
    let mut x = d;
    let q = if x < 0f32 {
        x = -x;
        1
    } else {
        0
    };
    let c = x;
    if x > 1f32 {
        x = 1f32 / x;
    }
    let x2 = x * x;

    let mut u = 0.00282363896258175373077393;
    u = fmlaf(u, x2, -0.0159569028764963150024414);
    u = fmlaf(u, x2, 0.0425049886107444763183594);
    u = fmlaf(u, x2, -0.0748900920152664184570312);
    u = fmlaf(u, x2, 0.106347933411598205566406);
    u = fmlaf(u, x2, -0.142027363181114196777344);
    u = fmlaf(u, x2, 0.199926957488059997558594);
    u = fmlaf(u, x2, -0.333331018686294555664062);

    u = x + x * (x2 * u);

    u = if c > 1f32 {
        std::f32::consts::FRAC_PI_2 - u
    } else {
        u
    };
    if q & 1 != 0 {
        u = -u;
    }
    u
}

/// Computes Atan2
#[inline]
pub const fn atan2f(y: f32, x: f32) -> f32 {
    if x == 0. {
        if y > 0. {
            return std::f32::consts::FRAC_PI_2;
        }
        if y < 0. {
            return -std::f32::consts::FRAC_PI_2;
        }
        if y == 0. {
            return 0f32;
        }
    }
    let rad = atanf(y / x);
    if x > 0f32 {
        rad
    } else if x < 0f32 && y >= 0f32 {
        std::f32::consts::PI + rad
    } else {
        // if x < 0. && y < 0.
        -std::f32::consts::PI + rad
    }
}

#[inline]
pub(crate) fn hypot3f(x: f32, y: f32, z: f32) -> f32 {
    let x = x.abs();
    let y = y.abs();
    let z = z.abs();

    let max = x.max(y).max(z);

    if max == 0.0 {
        return 0.0;
    }

    let recip_max = 1. / max;

    let norm_x = x * recip_max;
    let norm_y = y * recip_max;
    let norm_z = z * recip_max;

    let ret = max * (norm_x * norm_x + norm_y * norm_y + norm_z * norm_z).sqrt();

    if x == f32::INFINITY || y == f32::INFINITY || z == f32::INFINITY {
        f32::INFINITY
    } else if x.is_nan() || y.is_nan() || z.is_nan() || ret.is_nan() {
        f32::NAN
    } else {
        ret
    }
}

#[inline(always)]
pub const fn rounding_div_ceil(value: i32, div: i32) -> i32 {
    (value + div - 1) / div
}

// Generic function for max
#[inline(always)]
pub(crate) fn m_max<T: Num + PartialOrd>(a: T, b: T) -> T {
    if a > b { a } else { b }
}

// Generic function for min
#[inline(always)]
pub(crate) fn m_min<T: Num + PartialOrd>(a: T, b: T) -> T {
    if a < b { a } else { b }
}

#[inline]
pub(crate) fn m_clamp<T: Num + PartialOrd>(a: T, min: T, max: T) -> T {
    if a > max {
        max
    } else if a >= min {
        a
    } else {
        // a < min or a is NaN
        min
    }
}

pub(crate) trait FusedMultiplyAdd<T> {
    fn mla(&self, b: T, c: T) -> T;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cbrtf() {
        assert_eq!(cbrtf(0.0), 0.0);
        assert_eq!(cbrtf(-27.0), -3.0);
        assert_eq!(cbrtf(27.0), 3.0);
    }

    #[test]
    fn cosf_test() {
        assert_eq!(cosf(0.0), 1.0);
        assert_eq!(cosf(std::f32::consts::PI), -1f32);
    }

    #[test]
    fn sinf_test() {
        assert_eq!(sinf(0.0), 0.0);
        assert!((sinf(std::f32::consts::PI) - 0f32).abs() < 1e-6);
        assert!((sinf(std::f32::consts::FRAC_PI_2) - 1f32).abs() < 1e-6);
    }

    #[test]
    fn expf_test() {
        assert!(
            (expf(0f32) - 1f32).abs() < 1e-6,
            "Invalid result {}",
            expf(0f32)
        );
        assert!(
            (expf(5f32) - 148.4131591025766f32).abs() < 1e-6,
            "Invalid result {}",
            expf(5f32)
        );

        assert!(
            (exp(0f64) - 1f64).abs() < 1e-8,
            "Invalid result {}",
            exp(0f64)
        );
        assert!(
            (exp(5f64) - 148.4131591025766034211155800405522796f64).abs() < 1e-8,
            "Invalid result {}",
            exp(5f64)
        );
    }

    #[test]
    fn logf_test() {
        assert!(
            (logf(1f32) - 0f32).abs() < 1e-6,
            "Invalid result {}",
            logf(1f32)
        );
        assert!(
            (logf(5f32) - 1.60943791243410037460f32).abs() < 1e-6,
            "Invalid result {}",
            logf(5f32)
        );

        assert!(
            (log(1f64) - 0f64).abs() < 1e-8,
            "Invalid result {}",
            log(1f64)
        );
        assert!(
            (log(5f64) - 1.60943791243410037460f64).abs() < 1e-8,
            "Invalid result {}",
            log(5f64)
        );
    }

    #[test]
    fn powf_test() {
        assert!(
            (powf(2f32, 3f32) - 8f32).abs() < 1e-6,
            "Invalid result {}",
            powf(2f32, 3f32)
        );
        assert!(
            (powf(0.5f32, 2f32) - 0.25f32).abs() < 1e-6,
            "Invalid result {}",
            powf(0.5f32, 2f32)
        );

        assert!(
            (pow(2f64, 3f64) - 8f64).abs() < 1e-9,
            "Invalid result {}",
            pow(2f64, 3f64)
        );
        assert!(
            (pow(0.5f64, 2f64) - 0.25f64).abs() < 1e-9,
            "Invalid result {}",
            pow(0.5f64, 2f64)
        );
    }
    #[test]
    fn sqrtf_test() {
        assert!(
            (sqrtf(4f32) - 2f32).abs() < 1e-6,
            "Invalid result {}",
            sqrtf(4f32)
        );
        assert!(
            (sqrtf(9f32) - 3f32).abs() < 1e-6,
            "Invalid result {}",
            sqrtf(9f32)
        );
        println!("{}", sqrtf(4f32));
        println!("{}", sqrtf(9f32));
        println!("{}", sqrtf(12f32));
        println!("{}", sqrtf(25f32));
    }

    #[test]
    fn atan_test() {
        assert!(
            (atanf(2f32) - 1.107148717794090503017065f32).abs() < 1e-6,
            "Invalid result {}",
            atanf(2f32)
        );
        assert!(
            (atanf(5f32) - 1.3734007669450158608612719264f32).abs() < 1e-6,
            "Invalid result {}",
            atanf(5f32)
        );
    }
}
