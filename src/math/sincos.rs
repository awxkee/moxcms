/*
 * // Copyright (c) Radzivon Bartoshyk 6/2025. All rights reserved.
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
use crate::math::common::f_fmla;
use crate::math::dekker::Dekker;
use crate::math::sin::{
    LargeArgumentReduction, SIN_K_PI_OVER_128, range_reduction_small, sincos_eval,
};

/// Sine and cosine for double precision
///
/// ULP 0.5
#[inline]
pub fn f_sincos(x: f64) -> (f64, f64) {
    let x_e = (x.to_bits() >> 52) & 0x7ff;
    const E_BIAS: u64 = (1u64 << (11 - 1u64)) - 1u64;

    let y: Dekker;
    let k;

    // |x| < 2^32 (with FMA) or |x| < 2^23 (w/o FMA)
    if x_e < E_BIAS + 22 {
        // |x| < 2^-26
        if x_e < E_BIAS - 7 {
            if x_e < E_BIAS - 26 {
                // Signed zeros.
                if x == 0.0 {
                    return (x, 1.0);
                }
                // For |x| < 2^-26, |sin(x) - x| < ulp(x)/2.
                let s_sin = f_fmla(x, f64::from_bits(0xbc90000000000000), x);
                let s_cos = 1.0 - f64::EPSILON;
                return (s_sin, s_cos);
            }
            k = 0;
            y = Dekker::new(0.0, x);
        } else {
            // // Small range reduction.
            (y, k) = range_reduction_small(x);
        }
    } else {
        // Inf or NaN
        if x_e > 2 * E_BIAS {
            // sin(+-Inf) = NaN
            return (x + f64::NAN, x + f64::NAN);
        }

        // Large range reduction.
        let mut argument_reduction = LargeArgumentReduction::default();
        k = argument_reduction.high_part(x);
        y = argument_reduction.reduce();
    }

    let r_sincos = sincos_eval(y);
    let (sin_y, cos_y) = (r_sincos.v_sin, r_sincos.v_cos);

    // Fast look up version, but needs 256-entry table.
    // cos(k * pi/128) = sin(k * pi/128 + pi/2) = sin((k + 64) * pi/128).
    let sk = SIN_K_PI_OVER_128[(k & 255) as usize];
    let ck = SIN_K_PI_OVER_128[((k.wrapping_add(64)) & 255) as usize];
    let sin_k = Dekker::new(f64::from_bits(sk.0), f64::from_bits(sk.1));
    let cos_k = Dekker::new(f64::from_bits(ck.0), f64::from_bits(ck.1));

    let msin_k = Dekker::new(-sin_k.lo, -sin_k.hi);

    // After range reduction, k = round(x * 128 / pi) and y = x - k * (pi / 128).
    // So k is an integer and -pi / 256 <= y <= pi / 256.
    // Then sin(x) = sin((k * pi/128 + y)
    //             = sin(y) * cos(k*pi/128) + cos(y) * sin(k*pi/128)
    let sin_k_cos_y = Dekker::quick_mult(cos_y, sin_k);
    let cos_k_sin_y = Dekker::quick_mult(sin_y, cos_k);
    //      cos(x) = cos((k * pi/128 + y)
    //             = cos(y) * cos(k*pi/128) - sin(y) * sin(k*pi/128)
    let cos_k_cos_y = Dekker::quick_mult(cos_y, cos_k);
    let msin_k_sin_y = Dekker::quick_mult(sin_y, msin_k);

    let mut sin_dd = Dekker::from_full_exact_add(sin_k_cos_y.hi, cos_k_sin_y.hi);
    let mut cos_dd = Dekker::from_full_exact_add(cos_k_cos_y.hi, msin_k_sin_y.hi);
    sin_dd.lo += sin_k_cos_y.lo + cos_k_sin_y.lo;
    cos_dd.lo += msin_k_sin_y.lo + cos_k_cos_y.lo;

    let sin_x = sin_dd.hi + sin_dd.lo;
    let cos_x = cos_dd.hi + cos_dd.lo;
    (sin_x, cos_x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sincos_test() {
        let zx_0 = f_sincos(0.0);
        assert_eq!(zx_0.0, 0.0);
        assert_eq!(zx_0.1, 1.0);
        let zx_1 = f_sincos(1.0);
        assert_eq!(zx_1.0, 0.8414709848078965);
        assert_eq!(zx_1.1, 0.5403023058681398);
        let zx_0_p5 = f_sincos(-0.5);
        assert_eq!(zx_0_p5.0, -0.479425538604203);
        assert_eq!(zx_0_p5.1, 0.8775825618903728);
    }
}
