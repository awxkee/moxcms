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

    if d == 0f64 {
        f64::NEG_INFINITY
    } else if (d < 0.) || d.is_nan() {
        f64::NAN
    } else if d.is_infinite() {
        f64::INFINITY
    } else {
        let t = m * m * 0.5;
        let r = fmla(x, t, fmla(x, u, LN2_L * n as f64)) - t + m;
        fmla(LN2_H, n as f64, r)
    }
}

/// Natural logarithm using FMA
#[inline]
pub fn f_log(d: f64) -> f64 {
    const LN_POLY_2_D: f64 = 0.6666666666666762678e+0;
    const LN_POLY_3_D: f64 = 0.3999999999936908641e+0;
    const LN_POLY_4_D: f64 = 0.2857142874046159249e+0;
    const LN_POLY_5_D: f64 = 0.2222219947428228041e+0;
    const LN_POLY_6_D: f64 = 0.1818349302807168999e+0;
    const LN_POLY_7_D: f64 = 0.1531633000781658996e+0;
    const LN_POLY_8_D: f64 = 0.1476969208015536904e+0;

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

    const LN2_H: f64 = 0.6931471805599453;
    const LN2_L: f64 = 2.3190468138462996e-17;
    
    #[cfg(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    ))]
    {
        let mut u = LN_POLY_8_D;
        u = f_fmla(u, x2, LN_POLY_7_D);
        u = f_fmla(u, x2, LN_POLY_6_D);
        u = f_fmla(u, x2, LN_POLY_5_D);
        u = f_fmla(u, x2, LN_POLY_4_D);
        u = f_fmla(u, x2, LN_POLY_3_D);
        u = f_fmla(u, x2, LN_POLY_2_D);
        u *= x2;
        if d == 0f64 {
            f64::NEG_INFINITY
        } else if (d < 0.) || d.is_nan() {
            f64::NAN
        } else if d.is_infinite() {
            f64::INFINITY
        } else {
            let t = m * m * 0.5;
            let r = f_fmla(x, t, f_fmla(x, u, LN2_L * n as f64)) - t + m;
            f_fmla(LN2_H, n as f64, r)
        }
    }
    #[cfg(not(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        use crate::math::estrin::*;
        let x4 = x2 * x2;
        let x8 = x4 * x4;
        let mut u = poly7!(
            x2,
            x4,
            x8,
            LN_POLY_8_D,
            LN_POLY_7_D,
            LN_POLY_6_D,
            LN_POLY_5_D,
            LN_POLY_4_D,
            LN_POLY_3_D,
            LN_POLY_2_D
        );
        u *= x2;
        if d == 0f64 {
            f64::NEG_INFINITY
        } else if (d < 0.) || d.is_nan() {
            f64::NAN
        } else if d.is_infinite() {
            f64::INFINITY
        } else {
            let t = m * m * 0.5;
            let r = f_fmla(x, t, f_fmla(x, u, LN2_L * n as f64)) - t + m;
            f_fmla(LN2_H, n as f64, r)
        }
    }
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
    }
}
