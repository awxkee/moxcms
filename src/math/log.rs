/*
 * // Copyright (c) Radzivon Bartoshyk 4/2025. All rights reserved.
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
use crate::math::common::*;
use crate::math::dekker::Dekker;
use crate::math::log2::{LOG_COEFFS, f_polyeval4};
use crate::math::log10::LOG_R_DD;

/// Natural logarithm
#[inline]
pub const fn log(d: f64) -> f64 {
    const LN_POLY_2_D: f64 = 0.6666666666666762678e+0;
    const LN_POLY_3_D: f64 = 0.3999999999936908641e+0;
    const LN_POLY_4_D: f64 = 0.2857142874046159249e+0;
    const LN_POLY_5_D: f64 = 0.2222219947428228041e+0;
    const LN_POLY_6_D: f64 = 0.1818349302807168999e+0;
    const LN_POLY_7_D: f64 = 0.1531633000781658996e+0;
    const LN_POLY_8_D: f64 = 0.1476969208015536904e+0;

    let e = d.to_bits().wrapping_shr(52).wrapping_sub(0x3ff);
    if e >= 0x400 || e == 0x00000000fffffc01 {
        let minf = 0xfffu64 << 52;
        if e == 0x400 || (e == 0xc00 && d != f64::from_bits(minf)) {
            /* +Inf or NaN */
            return d + d;
        }
        if d <= 0. {
            return if d < 0. { f64::NAN } else { f64::NEG_INFINITY };
        }
    }

    // reduce into [sqrt(2)/2;sqrt(2)]
    let mut ui: u64 = d.to_bits();
    let mut hx = (ui >> 32) as u32;
    hx = hx.wrapping_add(0x3ff00000 - 0x3fe6a09e);
    let n = (hx >> 20) as i32 - 0x3ff;
    hx = (hx & 0x000fffff).wrapping_add(0x3fe6a09e);
    ui = (hx as u64) << 32 | (ui & 0xffffffff);
    let a = f64::from_bits(ui);

    let m = a - 1.;

    let x = m / (a + 1.);
    let x2 = x * x;
    let f = x2;

    const LN2_H: f64 = 0.6931471805599453;
    const LN2_L: f64 = 2.3190468138462996e-17;

    let mut u = LN_POLY_8_D;
    u = fmla(u, f, LN_POLY_7_D);
    u = fmla(u, f, LN_POLY_6_D);
    u = fmla(u, f, LN_POLY_5_D);
    u = fmla(u, f, LN_POLY_4_D);
    u = fmla(u, f, LN_POLY_3_D);
    u = fmla(u, f, LN_POLY_2_D);
    u *= f;

    let t = m * m * 0.5;
    let r = fmla(x, t, fmla(x, u, LN2_L * n as f64)) - t + m;
    fmla(LN2_H, n as f64, r)
}

/// Natural logarithm using FMA
///
/// Max found ULP 0.5
#[inline]
pub fn f_log(x: f64) -> f64 {
    let mut x_u = x.to_bits();

    const E_BIAS: u64 = (1u64 << (11 - 1u64)) - 1u64;

    let mut x_e: i64 = -(E_BIAS as i64);

    const MIN_NORMAL: u64 = f64::to_bits(f64::MIN_POSITIVE);
    const MAX_NORMAL: u64 = f64::to_bits(f64::MAX);

    if x_u == 1f64.to_bits() {
        // log2(1.0) = +0.0
        return 0.0;
    }
    if x_u < MIN_NORMAL || x_u > MAX_NORMAL {
        if x == 0.0 {
            return f64::NEG_INFINITY;
        }
        if x < 0. || x.is_nan() {
            return f64::NAN;
        }
        if x.is_infinite() || x.is_nan() {
            return x + x;
        }
        // Normalize denormal inputs.
        x_u = (x * f64::from_bits(0x4330000000000000)).to_bits();
        x_e -= 52;
    }

    // log2(x) = log2(2^x_e * x_m)
    //         = x_e + log2(x_m)
    // Range reduction for log2(x_m):
    // For each x_m, we would like to find r such that:
    //   -2^-8 <= r * x_m - 1 < 2^-7
    let shifted = (x_u >> 45) as i64;
    let index = shifted & 0x7F;
    let r = f64::from_bits(crate::math::log2::LOG_RANGE_REDUCTION[index as usize]);

    // Add unbiased exponent. Add an extra 1 if the 8 leading fractional bits are
    // all 1's.
    x_e = x_e.wrapping_add(x_u.wrapping_add(1u64 << 45).wrapping_shr(52) as i64);
    let e_x = x_e as f64;

    const LOG_2_HI: f64 = f64::from_bits(0x3fe62e42fefa3800);
    const LOG_2_LO: f64 = f64::from_bits(0x3d2ef35793c76730);

    let log_r_dd = LOG_R_DD[index as usize];

    // hi is exact
    let hi = f_fmla(e_x, LOG_2_HI, f64::from_bits(log_r_dd.1));
    // lo errors ~ e_x * LSB(LOG_2_LO) + LSB(LOG_R[index].lo) + rounding err
    //           <= 2 * (e_x * LSB(LOG_2_LO) + LSB(LOG_R[index].lo))
    let lo = f_fmla(e_x, LOG_2_LO, f64::from_bits(log_r_dd.0));

    // Set m = 1.mantissa.
    let x_m = (x_u & 0x000F_FFFF_FFFF_FFFFu64) | 0x3FF0_0000_0000_0000u64;
    let m = f64::from_bits(x_m);

    let u;
    #[cfg(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    ))]
    {
        u = f_fmla(r, m, -1.0); // exact   
    }
    #[cfg(not(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        use crate::math::log2::LOG_CD;
        let c_m = x_m & 0x3FFF_E000_0000_0000u64;
        let c = f64::from_bits(c_m);
        u = f_fmla(r, m - c, f64::from_bits(LOG_CD[index as usize])); // exact
    }

    let r1 = Dekker::from_exact_add(hi, u);

    let u_sq = u * u;
    // Degree-7 minimax polynomial
    let p0 = f_fmla(
        u,
        f64::from_bits(LOG_COEFFS[1]),
        f64::from_bits(LOG_COEFFS[0]),
    );
    let p1 = f_fmla(
        u,
        f64::from_bits(LOG_COEFFS[3]),
        f64::from_bits(LOG_COEFFS[2]),
    );
    let p2 = f_fmla(
        u,
        f64::from_bits(LOG_COEFFS[5]),
        f64::from_bits(LOG_COEFFS[4]),
    );
    let p = f_polyeval4(u_sq, lo + r1.lo, p0, p1, p2);

    r1.hi + p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_test() {
        println!("{}", log(2.));
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
    fn f_log_test() {
        println!("{}", f_log(2.));
        assert!(
            (f_log(1f64) - 0f64).abs() < 1e-8,
            "Invalid result {}",
            f_log(1f64)
        );
        assert!(
            (f_log(5f64) - 5f64.ln()).abs() < 1e-8,
            "Invalid result {}, expected {}",
            f_log(5f64),
            5f64.ln()
        );
        assert_eq!(f_log(0.), f64::NEG_INFINITY);
        assert!(f_log(-1.).is_nan());
        assert!(f_log(f64::NAN).is_nan());
        assert!(f_log(f64::NEG_INFINITY).is_nan());
        assert_eq!(f_log(f64::INFINITY), f64::INFINITY);
    }
}
