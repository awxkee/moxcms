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
use crate::math::sin::{LargeArgumentReduction, range_reduction_small};

#[inline]
fn tan_eval(u: Dekker) -> Dekker {
    // Evaluate tan(y) = tan(x - k * (pi/128))
    // We use the degree-9 Taylor approximation:
    //   tan(y) ~ P(y) = y + y^3/3 + 2*y^5/15 + 17*y^7/315 + 62*y^9/2835
    // Then the error is bounded by:
    //   |tan(y) - P(y)| < 2^-6 * |y|^11 < 2^-6 * 2^-66 = 2^-72.
    // For y ~ u_hi + u_lo, fully expanding the polynomial and drop any terms
    // < ulp(u_hi^3) gives us:
    //   P(y) = y + y^3/3 + 2*y^5/15 + 17*y^7/315 + 62*y^9/2835 = ...
    // ~ u_hi + u_hi^3 * (1/3 + u_hi^2 * (2/15 + u_hi^2 * (17/315 +
    //                                                     + u_hi^2 * 62/2835))) +
    //        + u_lo (1 + u_hi^2 * (1 + u_hi^2 * 2/3))
    let u_hi_sq = u.hi * u.hi; // Error < ulp(u_hi^2) < 2^(-6 - 52) = 2^-58.
    // p1 ~ 17/315 + u_hi^2 62 / 2835.
    let p1 = f_fmla(
        u_hi_sq,
        f64::from_bits(0x3f9664f4882c10fa),
        f64::from_bits(0x3faba1ba1ba1ba1c),
    );
    // p2 ~ 1/3 + u_hi^2 2 / 15.
    let p2 = f_fmla(
        u_hi_sq,
        f64::from_bits(0x3fc1111111111111),
        f64::from_bits(0x3fd5555555555555),
    );
    // q1 ~ 1 + u_hi^2 * 2/3.
    let q1 = f_fmla(u_hi_sq, f64::from_bits(0x3fe5555555555555), 1.0);
    let u_hi_3 = u_hi_sq * u.hi;
    let u_hi_4 = u_hi_sq * u_hi_sq;
    // p3 ~ 1/3 + u_hi^2 * (2/15 + u_hi^2 * (17/315 + u_hi^2 * 62/2835))
    let p3 = f_fmla(u_hi_4, p1, p2);
    // q2 ~ 1 + u_hi^2 * (1 + u_hi^2 * 2/3)
    let q2 = f_fmla(u_hi_sq, q1, 1.0);
    let tan_lo = f_fmla(u_hi_3, p3, u.lo * q2);
    // Overall, |tan(y) - (u_hi + tan_lo)| < ulp(u_hi^3) <= 2^-71.
    // And the relative errors is:
    // |(tan(y) - (u_hi + tan_lo)) / tan(y) | <= 2*ulp(u_hi^2) < 2^-64
    Dekker::from_exact_add(u.hi, tan_lo)
}

/// Tan in double precision
///
/// ULP 0.5
pub fn f_tan(x: f64) -> f64 {
    let x_e = (x.to_bits() >> 52) & 0x7ff;
    const E_BIAS: u64 = (1u64 << (11 - 1u64)) - 1u64;

    let y: Dekker;
    let k;

    // |x| < 2^16
    if x_e < E_BIAS + 22 {
        // |x| < 2^-7
        if x_e < E_BIAS - 7 {
            // |x| < 2^-27, |tan(x) - x| < ulp(x)/2.
            if x_e < E_BIAS - 27 {
                // Signed zeros.
                if x == 0.0 {
                    return x + x;
                }
                return f_fmla(x, f64::from_bits(0x3c90000000000000), x);
            }
            // No range reduction needed.
            k = 0;
            y = Dekker::new(0., x);
        } else {
            // Small range reduction.
            (y, k) = range_reduction_small(x);
        }
    } else {
        // Inf or NaN
        if x_e > 2 * E_BIAS {
            if x.is_nan() {
                return f64::NAN;
            }
            // tan(+-Inf) = NaN
            return x + f64::NAN;
        }

        // Large range reduction.
        let mut argument_reduction = LargeArgumentReduction::default();
        k = argument_reduction.high_part(x);
        y = argument_reduction.reduce();
    }

    let tan_y = tan_eval(y);

    // Fast look up version, but needs 256-entry table.
    // cos(k * pi/128) = sin(k * pi/128 + pi/2) = sin((k + 64) * pi/128).
    let sk = crate::math::sin::SIN_K_PI_OVER_128[(k.wrapping_add(128) & 255) as usize];
    let ck = crate::math::sin::SIN_K_PI_OVER_128[((k.wrapping_add(64)) & 255) as usize];
    let msin_k = Dekker::new(f64::from_bits(sk.0), f64::from_bits(sk.1));
    let cos_k = Dekker::new(f64::from_bits(ck.0), f64::from_bits(ck.1));

    let cos_k_tan_y = Dekker::quick_mult(tan_y, cos_k);
    let msin_k_tan_y = Dekker::quick_mult(tan_y, msin_k);

    // num_dd = sin(k*pi/128) + tan(y) * cos(k*pi/128)
    let mut num_dd = Dekker::from_full_exact_add(cos_k_tan_y.hi, -msin_k.hi);
    // den_dd = cos(k*pi/128) - tan(y) * sin(k*pi/128)
    let mut den_dd = Dekker::from_full_exact_add(msin_k_tan_y.hi, cos_k.hi);
    num_dd.lo += cos_k_tan_y.lo - msin_k.lo;
    den_dd.lo += msin_k_tan_y.lo + cos_k.lo;

    Dekker::div(num_dd, den_dd).to_f64()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tan_test() {
        assert_eq!(f_tan(0.0), 0.0);
        assert_eq!(f_tan(1.0), 1.5574077246549023);
        assert_eq!(f_tan(-0.5), -0.5463024898437905);
    }
}
