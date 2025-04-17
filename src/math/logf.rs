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
pub const fn logf(d: f32) -> f32 {
    let mut ix = d.to_bits();
    /* reduce x into [sqrt(2)/2, sqrt(2)] */
    ix += 0x3f800000 - 0x3f3504f3;
    let n = (ix >> 23) as i32 - 0x7f;
    ix = (ix & 0x007fffff) + 0x3f3504f3;
    let a = f32::from_bits(ix) as f64;

    let x = (a - 1.) / (a + 1.);
    let x2 = x * x;
    let mut u = 0.2222220222147750310e+0;
    u = fmla(u, x2, 0.2857142871244668543e+0);
    u = fmla(u, x2, 0.3999999999950960318e+0);
    u = fmla(u, x2, 0.6666666666666734090e+0);
    u = fmla(u, x2, 2.);
    fmla(x, u, std::f64::consts::LN_2 * (n as f64)) as f32
    // if d == 0f32 {
    //     f32::NEG_INFINITY
    // } else if (d < 0.) || d.is_nan() {
    //     f32::NAN
    // } else if d.is_infinite() {
    //     f32::INFINITY
    // } else {
    // }
}

/// Natural logarithm using FMA
#[inline]
pub fn f_logf(d: f32) -> f32 {
    let mut ix = d.to_bits();
    /* reduce x into [sqrt(2)/2, sqrt(2)] */
    ix = ix.wrapping_add(0x3f800000 - 0x3f3504f3);
    let n = (ix >> 23) as i32 - 0x7f;
    ix = (ix & 0x007fffff).wrapping_add(0x3f3504f3);
    let a = f32::from_bits(ix) as f64;

    let x = (a - 1.) / (a + 1.);
    let x2 = x * x;
    #[cfg(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    ))]
    {
        let mut u = 0.2222220222147750310e+0;
        u = f_fmla(u, x2, 0.2857142871244668543e+0);
        u = f_fmla(u, x2, 0.3999999999950960318e+0);
        u = f_fmla(u, x2, 0.6666666666666734090e+0);
        u = f_fmla(u, x2, 2.);
        f_fmla(x, u, std::f64::consts::LN_2 * (n as f64)) as f32
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
        let rx2 = x2 * x2;
        let rx4 = rx2 * rx2;
        let u = poly5!(
            x2,
            rx2,
            rx4,
            0.2222220222147750310e+0,
            0.2857142871244668543e+0,
            0.3999999999950960318e+0,
            0.6666666666666734090e+0,
            2.
        );
        f_fmla(x, u, std::f64::consts::LN_2 * (n as f64)) as f32
    }
    // if d == 0f32 {
    //     f32::NEG_INFINITY
    // } else if (d < 0.) || d.is_nan() {
    //     f32::NAN
    // } else if d.is_infinite() {
    //     f32::INFINITY
    // } else {
    // }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logf() {
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
    }

    #[test]
    fn test_flogf() {
        assert!(
            (f_logf(1f32) - 0f32).abs() < 1e-6,
            "Invalid result {}",
            f_logf(1f32)
        );
        assert!(
            (f_logf(5f32) - 1.60943791243410037460f32).abs() < 1e-6,
            "Invalid result {}",
            f_logf(5f32)
        );
    }
}
