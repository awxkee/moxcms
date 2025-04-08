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

/// Computes exponent for given value
#[inline]
pub const fn exp(d: f64) -> f64 {
    const EXP_POLY_1_D: f64 = 2f64;
    const EXP_POLY_2_D: f64 = 0.16666666666666674f64;
    const EXP_POLY_3_D: f64 = -0.0027777777777777614f64;
    const EXP_POLY_4_D: f64 = 6.613756613755705e-5f64;
    const EXP_POLY_5_D: f64 = -1.6534391534392554e-6f64;
    const EXP_POLY_6_D: f64 = 4.17535139757361979584e-8f64;

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
    let mut u = EXP_POLY_6_D;
    u = fmla(u, f, EXP_POLY_5_D);
    u = fmla(u, f, EXP_POLY_4_D);
    u = fmla(u, f, EXP_POLY_3_D);
    u = fmla(u, f, EXP_POLY_2_D);
    u = fmla(u, f, EXP_POLY_1_D);
    let u = 1f64 + 2f64 * r / (u - r);
    let i2 = pow2i(q);
    u * i2
    // if d < -964f64 {
    //     r = 0f64;
    // }
    // if d > 709f64 {
    //     r = f64::INFINITY;
    // }
}

/// Exp using FMA
#[inline]
pub fn f_exp(d: f64) -> f64 {
    const EXP_POLY_1_D: f64 = 2f64;
    const EXP_POLY_2_D: f64 = 0.16666666666666674f64;
    const EXP_POLY_3_D: f64 = -0.0027777777777777614f64;
    const EXP_POLY_4_D: f64 = 6.613756613755705e-5f64;
    const EXP_POLY_5_D: f64 = -1.6534391534392554e-6f64;
    const EXP_POLY_6_D: f64 = 4.17535139757361979584e-8f64;

    const L2_U: f64 = 0.693_147_180_559_662_956_511_601_805_686_950_683_593_75;
    const L2_L: f64 = 0.282_352_905_630_315_771_225_884_481_750_134_360_255_254_120_68_e-12;
    const R_LN2: f64 =
        1.442_695_040_888_963_407_359_924_681_001_892_137_426_645_954_152_985_934_135_449_406_931;

    let qf = (d * R_LN2).round();
    let q = qf as i32;

    let mut r = f_fmla(qf, -L2_U, d);
    r = f_fmla(qf, -L2_L, r);

    let f = r * r;
    // Poly for u = r*(exp(r)+1)/(exp(r)-1)
    let mut u;
    #[cfg(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    ))]
    {
        u = EXP_POLY_6_D;
        u = f_fmla(u, f, EXP_POLY_5_D);
        u = f_fmla(u, f, EXP_POLY_4_D);
        u = f_fmla(u, f, EXP_POLY_3_D);
        u = f_fmla(u, f, EXP_POLY_2_D);
        u = f_fmla(u, f, EXP_POLY_1_D);
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
        let x2 = f * f;
        let x4 = x2 * x2;
        u = poly6!(
            f,
            x2,
            x4,
            EXP_POLY_6_D,
            EXP_POLY_5_D,
            EXP_POLY_4_D,
            EXP_POLY_3_D,
            EXP_POLY_2_D,
            EXP_POLY_1_D
        );
    }
    u = f_fmla(2f64, r / (u - r), 1.);
    let i2 = pow2i(q);
    u * i2
    // if d < -964f64 {
    //     r = 0f64;
    // }
    // if d > 709f64 {
    //     r = f64::INFINITY;
    // }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_test() {
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

        assert!(
            (f_exp(0f64) - 1f64).abs() < 1e-8,
            "Invalid result {}",
            f_exp(0f64)
        );
        assert!(
            (f_exp(5f64) - 148.4131591025766034211155800405522796f64).abs() < 1e-8,
            "Invalid result {}",
            f_exp(5f64)
        );
    }
}
