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

/// Computes cube root
///
/// Max found ULP 0.4999999373385222
#[inline]
pub fn f_cbrt(x: f64) -> f64 {
    static ESCALE: [f64; 3] = [
        1.0,
        f64::from_bits(0x3ff428a2f98d728b),
        f64::from_bits(0x3ff965fea53d6e3d),
    ];
    const U0: f64 = f64::from_bits(0x3fd5555555555555);
    const U1: f64 = f64::from_bits(0x3fcc71c71c71c71c);

    static RSC: [f64; 6] = [1.0, -1.0, 0.5, -0.5, 0.25, -0.25];

    let bits = x.to_bits();
    let sign = bits >> 63;
    let mut exp = ((bits >> 52) & 0x7ff) as i32;
    let mut mant = bits & ((1u64 << 52) - 1);

    if exp == 0x7ff || x == 0.0 {
        return x + x;
    }

    // Normalize subnormal
    if exp == 0 && x != 0.0 {
        let norm = x * f64::from_bits(0x4350000000000000); // * 2^54
        let norm_bits = norm.to_bits();
        mant = norm_bits & ((1u64 << 52) - 1);
        exp = ((norm_bits >> 52) & 0x7ff) as i32 - 54;
    }

    exp += 3072;

    let cvt1 = mant | (0x3ff << 52);
    let mut cvt5 = cvt1;

    let et = exp / 3;
    let it = (exp % 3) as u64;

    cvt5 = cvt5.wrapping_add(it.wrapping_shl(52));
    cvt5 |= sign << 63;

    let zz = cvt5;
    /* cbrt(x) = cbrt(zz)*2^(et-1365) where 1 <= zz < 8 */
    let mut isc = ESCALE[it as usize].to_bits();
    isc |= sign << 63;
    let cvt2 = isc;
    let z = f64::from_bits(cvt1);
    let r = 1.0 / z;
    let rr = r * RSC[((it.wrapping_shl(1)) | sign) as usize];
    let z2 = z * z;
    let c0 = f_fmla(
        z,
        f64::from_bits(0x3fe2c9a3e94d1da5),
        f64::from_bits(0x3fe1b0babccfef9c),
    );
    let c2 = f_fmla(
        z,
        f64::from_bits(0x3f97a8d3e4ec9b07),
        f64::from_bits(0xbfc4dc30b1a1ddba),
    );
    let mut y = f_fmla(z2, c2, c0);
    let mut y2 = y * y;

    let mut h = f_fmla(y2, y * r, -1.0);
    /* h determines the error between y and z^(1/3) */
    y = f_fmla(-(h * y), f_fmla(-U1, h, U0), y);
    /* The correction y -= (h*y)*(u0 - u1*h) corresponds to a cubic variant
    of Newton's method, with the function f(y) = 1-z/y^3. */
    y *= f64::from_bits(cvt2);
    /* Now y is an approximation of zz^(1/3),
    and rr an approximation of 1/zz. We now perform another iteration of
    Newton-Raphson, this time with a linear approximation only. */
    y2 = y * y;
    let y2l = f_fmla(y, y, -y2);
    /* y2 + y2l = y^2 exactly */
    let y3 = y2 * y;
    let y3l = f_fmla(y, y2l, f_fmla(y, y2, -y3));
    /* y3 + y3l approximates y^3 with about 106 bits of accuracy */
    h = ((y3 - f64::from_bits(zz)) + y3l) * rr;
    /* the approximation of zz^(1/3) is y - dy */
    let y1 = f_fmla(-h, y * U0, y);
    let mut cvt3 = y1.to_bits();
    cvt3 = cvt3.wrapping_add(
        (et as u64)
            .wrapping_sub(342)
            .wrapping_sub(1023)
            .wrapping_shl(52),
    );
    f64::from_bits(cvt3)
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_cbrt() {
        assert_eq!(f_cbrt(27.0), 3.0);
        assert_eq!(f_cbrt(64.0), 4.0);
        assert_eq!(f_cbrt(125.0), 5.0);
        assert_eq!(f_cbrt(216.0), 6.0);
        assert_eq!(f_cbrt(343.0), 7.0);
        assert_eq!(f_cbrt(512.0), 8.0);
        assert_eq!(f_cbrt(729.0), 9.0);
        assert_eq!(f_cbrt(-729.0), -9.0);
        assert_eq!(f_cbrt(-512.0), -8.0);
        assert_eq!(f_cbrt(-343.0), -7.0);
        assert_eq!(f_cbrt(-216.0), -6.0);
        assert_eq!(f_cbrt(-125.0), -5.0);
        assert_eq!(f_cbrt(-64.0), -4.0);
        assert_eq!(f_cbrt(-27.0), -3.0);
        assert_eq!(f_cbrt(0.0), 0.0);
        assert_eq!(f_cbrt(f64::INFINITY), f64::INFINITY);
        assert_eq!(f_cbrt(f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert!(f_cbrt(f64::NAN).is_nan());
    }
}
