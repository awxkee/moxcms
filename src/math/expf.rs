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

const L2U_F: f32 = 0.693_145_751_953_125;
const L2L_F: f32 = 1.428_606_765_330_187_045_e-6;
const R_LN2_F: f32 = std::f32::consts::LOG2_E;

/// Computes exponent for given value
#[inline]
pub const fn expf(d: f32) -> f32 {
    const EXP_POLY_1_S: f32 = 2f32;
    const EXP_POLY_2_S: f32 = 0.16666707f32;
    const EXP_POLY_3_S: f32 = -0.002775669f32;
    let qf = rintfk(d * R_LN2_F);
    let q = qf as i32;
    let r = fmlaf(qf, -L2U_F, d);
    let r = fmlaf(qf, -L2L_F, r);

    let f = r * r;
    // Poly for u = r*(exp(r)+1)/(exp(r)-1)
    let mut u = EXP_POLY_3_S;
    u = fmlaf(u, f, EXP_POLY_2_S);
    u = fmlaf(u, f, EXP_POLY_1_S);
    let u = 1f32 + 2f32 * r / (u - r);
    let i2 = pow2if(q);
    u * i2
    // if d < -87f32 {
    //     r = 0f32;
    // }
    // if d > 88f32 {
    //     r = f32::INFINITY;
    // }
}

/// Computes exponent for given value using FMA
#[inline]
pub fn f_expf(d: f32) -> f32 {
    const EXP_POLY_1_S: f32 = 2f32;
    const EXP_POLY_2_S: f32 = 0.16666707f32;
    const EXP_POLY_3_S: f32 = -0.002775669f32;
    let qf = (d * R_LN2_F).round();
    let q = qf as i32;
    let r = f_fmlaf(qf, -L2U_F, d);
    let r = f_fmlaf(qf, -L2L_F, r);

    let f = r * r;
    // Poly for u = r*(exp(r)+1)/(exp(r)-1)
    #[allow(unused_mut)]
    let mut u;
    #[cfg(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    ))]
    {
        u = EXP_POLY_3_S;
        u = f_fmlaf(u, f, EXP_POLY_2_S);
        u = f_fmlaf(u, f, EXP_POLY_1_S);
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
        u = poly3!(f, x2, EXP_POLY_3_S, EXP_POLY_2_S, EXP_POLY_1_S)
    }
    let u = f_fmlaf(2f32, r / (u - r), 1f32);
    let i2 = pow2if(q);
    u * i2
    // if d < -87f32 {
    //     r = 0f32;
    // }
    // if d > 88f32 {
    //     r = f32::INFINITY;
    // }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            (f_expf(0f32) - 1f32).abs() < 1e-6,
            "Invalid result {}",
            f_expf(0f32)
        );
        assert!(
            (f_expf(5f32) - 148.4131591025766f32).abs() < 1e-6,
            "Invalid result {}",
            f_expf(5f32)
        );
    }
}
