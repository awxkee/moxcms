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
use crate::math::asin::asin_eval;
use crate::math::common::f_fmla;
use crate::math::dekker::Dekker;

#[inline]
pub fn f_acos(x: f64) -> f64 {
    let x_e = (x.to_bits() >> 52) & 0x7ff;
    const E_BIAS: u64 = (1u64 << (11 - 1u64)) - 1u64;

    const PI_OVER_TWO: Dekker = Dekker::new(
        f64::from_bits(0x3c91a62633145c07),
        f64::from_bits(0x3ff921fb54442d18),
    );

    let x_abs = f64::from_bits(x.to_bits() & 0x7fff_ffff_ffff_ffff);

    // |x| < 0.5.
    if x_e < E_BIAS - 1 {
        // |x| < 2^-55.
        if x_e < E_BIAS - 55 {
            // When |x| < 2^-55, acos(x) = pi/2
            return (x_abs + f64::from_bits(0x35f0000000000000)) + PI_OVER_TWO.hi;
        }

        let x_sq = Dekker::from_exact_mult(x, x);
        let err = x_abs * f64::from_bits(0x3cc0000000000000);
        // Polynomial approximation:
        //   p ~ asin(x)/x
        let (p, err) = asin_eval(x_sq, err);
        // asin(x) ~ x * p
        let r0 = Dekker::from_exact_mult(x, p.hi);
        // acos(x) = pi/2 - asin(x)
        //         ~ pi/2 - x * p
        //         = pi/2 - x * (p.hi + p.lo)
        let r_hi = f_fmla(-x, p.hi, PI_OVER_TWO.hi);
        // Use Dekker's 2SUM algorithm to compute the lower part.
        let mut r_lo = ((PI_OVER_TWO.hi - r_hi) - r0.hi) - r0.lo;
        r_lo = f_fmla(-x, p.lo, r_lo + PI_OVER_TWO.lo);
        return r_hi + (r_lo + err);
    }

    // |x| >= 0.5

    const SIGN: [f64; 2] = [1.0, -1.0];
    let x_sign = SIGN[if x.is_sign_negative() { 1 } else { 0 }];

    const PI: Dekker = Dekker::new(
        f64::from_bits(0x3ca1a62633145c07),
        f64::from_bits(0x400921fb54442d18),
    );

    // |x| >= 1
    if x_e >= E_BIAS {
        // x = +-1, asin(x) = +- pi/2
        if x_abs == 1.0 {
            // x = 1, acos(x) = 0,
            // x = -1, acos(x) = pi
            return if x == 1.0 {
                0.0
            } else {
                f_fmla(-x_sign, PI.hi, PI.lo)
            };
        }
        // |x| > 1, return NaN.
        return f64::NAN;
    }

    // When |x| >= 0.5, we perform range reduction as follow:
    //
    // When 0.5 <= x < 1, let:
    //   y = acos(x)
    // We will use the double angle formula:
    //   cos(2y) = 1 - 2 sin^2(y)
    // and the complement angle identity:
    //   x = cos(y) = 1 - 2 sin^2 (y/2)
    // So:
    //   sin(y/2) = sqrt( (1 - x)/2 )
    // And hence:
    //   y/2 = asin( sqrt( (1 - x)/2 ) )
    // Equivalently:
    //   acos(x) = y = 2 * asin( sqrt( (1 - x)/2 ) )
    // Let u = (1 - x)/2, then:
    //   acos(x) = 2 * asin( sqrt(u) )
    // Moreover, since 0.5 <= x < 1:
    //   0 < u <= 1/4, and 0 < sqrt(u) <= 0.5,
    // And hence we can reuse the same polynomial approximation of asin(x) when
    // |x| <= 0.5:
    //   acos(x) ~ 2 * sqrt(u) * P(u).
    //
    // When -1 < x <= -0.5, we reduce to the previous case using the formula:
    //   acos(x) = pi - acos(-x)
    //           = pi - 2 * asin ( sqrt( (1 + x)/2 ) )
    //           ~ pi - 2 * sqrt(u) * P(u),
    // where u = (1 - |x|)/2.

    // u = (1 - |x|)/2
    let u = f_fmla(x_abs, -0.5, 0.5);
    // v_hi + v_lo ~ sqrt(u).
    // Let:
    //   h = u - v_hi^2 = (sqrt(u) - v_hi) * (sqrt(u) + v_hi)
    // Then:
    //   sqrt(u) = v_hi + h / (sqrt(u) + v_hi)
    //            ~ v_hi + h / (2 * v_hi)
    // So we can use:
    //   v_lo = h / (2 * v_hi).
    let v_hi = u.sqrt();

    let h;
    #[cfg(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    ))]
    {
        h = f_fmla(v_hi, -v_hi, u);
    }
    #[cfg(not(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        let v_hi_sq = Dekker::from_exact_mult(v_hi, v_hi);
        h = (u - v_hi_sq.hi) - v_hi_sq.lo;
    }

    // Scale v_lo and v_hi by 2 from the formula:
    //   vh = v_hi * 2
    //   vl = 2*v_lo = h / v_hi.
    let vh = v_hi * 2.0;
    let vl = h / v_hi;

    // Polynomial approximation:
    //   p ~ asin(sqrt(u))/sqrt(u)
    let err = vh * f64::from_bits(0x3cc0000000000000);

    let (p, err) = asin_eval(Dekker::new(0.0, u), err);

    // Perform computations in double-double arithmetic:
    //   asin(x) = pi/2 - (v_hi + v_lo) * (ASIN_COEFFS[idx][0] + p)
    let r0 = Dekker::quick_mult(Dekker::new(vl, vh), p);

    let r_hi;
    let r_lo;
    if x.is_sign_positive() {
        r_hi = r0.hi;
        r_lo = r0.lo;
    } else {
        let r = Dekker::from_exact_add(PI.hi, -r0.hi);
        r_hi = r.hi;
        r_lo = (PI.lo - r0.lo) + r.lo;
    }

    r_hi + (r_lo + err)
}

#[cfg(test)]
mod tests {
    use crate::math::acos::f_acos;

    #[test]
    fn f_acos_test() {
        assert_eq!(f_acos(0.7), 0.7953988301841436);
        assert_eq!(f_acos(-0.1), 1.6709637479564565);
        assert_eq!(f_acos(-0.4), 1.9823131728623846);
    }
}
