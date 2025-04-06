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
#[cfg(not(any(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "fma"
    ),
    all(target_arch = "aarch64", target_feature = "neon")
)))]
use crate::math::estrin::*;

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
    // if d == 0f32 {
    //     f32::NEG_INFINITY
    // } else if (d < 0.) || d.is_nan() {
    //     f32::NAN
    // } else if d.is_infinite() {
    //     f32::INFINITY
    // } else {
    x * u + std::f32::consts::LN_2 * (n as f32)
    // }
}

/// Natural logarithm using FMA
#[inline]
pub fn f_logf(d: f32) -> f32 {
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
    #[cfg(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    ))]
    {
        let mut u = LN_POLY_5_F;
        u = f_fmlaf(u, x2, LN_POLY_4_F);
        u = f_fmlaf(u, x2, LN_POLY_3_F);
        u = f_fmlaf(u, x2, LN_POLY_2_F);
        u = f_fmlaf(u, x2, LN_POLY_1_F);
        f_fmlaf(x, u, std::f32::consts::LN_2 * (n as f32))
    }
    #[cfg(not(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        let rx2 = x2 * x2;
        let rx4 = rx2 * rx2;
        let u = poly5!(
            x2,
            rx2,
            rx4,
            LN_POLY_5_F,
            LN_POLY_4_F,
            LN_POLY_3_F,
            LN_POLY_2_F,
            LN_POLY_1_F
        );
        f_fmlaf(x, u, std::f32::consts::LN_2 * (n as f32))
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
}
