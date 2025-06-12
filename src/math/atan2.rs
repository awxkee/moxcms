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

static ATAN_I: [(u64, u64); 65] = [
    (0x0000000000000000, 0x0000000000000000),
    (0xbc2220c39d4dff50, 0x3f8fff555bbb729b),
    (0xbc35ec431444912c, 0x3f9ffd55bba97625),
    (0xbc086ef8f794f105, 0x3fa7fb818430da2a),
    (0xbc3c934d86d23f1d, 0x3faff55bb72cfdea),
    (0x3c5ac4ce285df847, 0x3fb3f59f0e7c559d),
    (0xbc5cfb654c0c3d98, 0x3fb7ee182602f10f),
    (0x3c5f7b8f29a05987, 0x3fbbe39ebe6f07c3),
    (0xbc4cd37686760c17, 0x3fbfd5ba9aac2f6e),
    (0xbc4b485914dacf8c, 0x3fc1e1fafb043727),
    (0x3c661a3b0ce9281b, 0x3fc3d6eee8c6626c),
    (0xbc5054ab2c010f3d, 0x3fc5c9811e3ec26a),
    (0x3c5347b0b4f881ca, 0x3fc7b97b4bce5b02),
    (0x3c4cf601e7b4348e, 0x3fc9a6a8e96c8626),
    (0x3c217b10d2e0e5ab, 0x3fcb90d7529260a2),
    (0x3c6c648d1534597e, 0x3fcd77d5df205736),
    (0x3c68ab6e3cf7afbd, 0x3fcf5b75f92c80dd),
    (0x3c762e47390cb865, 0x3fd09dc597d86362),
    (0x3c630ca4748b1bf9, 0x3fd18bf5a30bf178),
    (0xbc7077cdd36dfc81, 0x3fd278372057ef46),
    (0xbc6963a544b672d8, 0x3fd362773707ebcc),
    (0xbc75d5e43c55b3ba, 0x3fd44aa436c2af0a),
    (0xbc62566480884082, 0x3fd530ad9951cd4a),
    (0xbc7a725715711f00, 0x3fd614840309cfe2),
    (0xbc7c63aae6f6e918, 0x3fd6f61941e4def1),
    (0x3c769c885c2b249a, 0x3fd7d5604b63b3f7),
    (0x3c7b6d0ba3748fa8, 0x3fd8b24d394a1b25),
    (0x3c79e6c988fd0a77, 0x3fd98cd5454d6b18),
    (0xbc724dec1b50b7ff, 0x3fda64eec3cc23fd),
    (0x3c7ae187b1ca5040, 0x3fdb3a911da65c6c),
    (0xbc7cc1ce70934c34, 0x3fdc0db4c94ec9f0),
    (0xbc7a2cfa4418f1ad, 0x3fdcde53432c1351),
    (0x3c7a2b7f222f65e2, 0x3fddac670561bb4f),
    (0x3c70e53dc1bf3435, 0x3fde77eb7f175a34),
    (0xbc6a3992dc382a23, 0x3fdf40dd0b541418),
    (0xbc8b32c949c9d593, 0x3fe0039c73c1a40c),
    (0xbc7d5b495f6349e6, 0x3fe0657e94db30d0),
    (0x3c5974fa13b5404f, 0x3fe0c6145b5b43da),
    (0xbc52bdaee1c0ee35, 0x3fe1255d9bfbd2a9),
    (0x3c8c621cec00c301, 0x3fe1835a88be7c13),
    (0xbc5928df287a668f, 0x3fe1e00babdefeb4),
    (0x3c6c421c9f38224e, 0x3fe23b71e2cc9e6a),
    (0xbc709e73b0c6c087, 0x3fe2958e59308e31),
    (0x3c8c5d5e9ff0cf8d, 0x3fe2ee628406cbca),
    (0x3c81021137c71102, 0x3fe345f01cce37bb),
    (0xbc82304331d8bf46, 0x3fe39c391cd4171a),
    (0x3c7ecf8b492644f0, 0x3fe3f13fb89e96f4),
    (0xbc7f76d0163f79c8, 0x3fe445065b795b56),
    (0x3c72419a87f2a458, 0x3fe4978fa3269ee1),
    (0x3c84a33dbeb3796c, 0x3fe4e8de5bb6ec04),
    (0xbc81bb74abda520c, 0x3fe538f57b89061f),
    (0xbc75e5c9d8c5a950, 0x3fe587d81f732fbb),
    (0x3c60028e4bc5e7ca, 0x3fe5d58987169b18),
    (0xbc62b785350ee8c1, 0x3fe6220d115d7b8e),
    (0xbc76ea6febe8bbba, 0x3fe66d663923e087),
    (0xbc8a80386188c50e, 0x3fe6b798920b3d99),
    (0xbc78c34d25aadef6, 0x3fe700a7c5784634),
    (0x3c47b2a6165884a1, 0x3fe748978fba8e0f),
    (0x3c8406a089803740, 0x3fe78f6bbd5d315e),
    (0x3c8560821e2f3aa9, 0x3fe7d528289fa093),
    (0xbc7bf76229d3b917, 0x3fe819d0b7158a4d),
    (0x3c66b66e7fc8b8c3, 0x3fe85d69576cc2c5),
    (0xbc855b9a5e177a1b, 0x3fe89ff5ff57f1f8),
    (0xbc7ec182ab042f61, 0x3fe8e17aa99cc05e),
    (0x3c81a62633145c07, 0x3fe921fb54442d18),
];

// Approximate atan(x) for |x| <= 2^-7.
// Using degree-9 Taylor polynomial:
//  P = x - x^3/3 + x^5/5 -x^7/7 + x^9/9;
// Then the absolute error is bounded by:
//   |atan(x) - P(x)| < |x|^11/11 < 2^(-7*11) / 11 < 2^-80.
// And the relative error is bounded by:
//   |(atan(x) - P(x))/atan(x)| < |x|^10 / 10 < 2^-73.
// For x = x_hi + x_lo, fully expand the polynomial and drop any terms less than
//   ulp(x_hi^3 / 3) gives us:
// P(x) ~ x_hi - x_hi^3/3 + x_hi^5/5 - x_hi^7/7 + x_hi^9/9 +
//        + x_lo * (1 - x_hi^2 + x_hi^4)
// Since p.lo is ~ x^3/3, the relative error from rounding is bounded by:
//   |(atan(x) - P(x))/atan(x)| < ulp(x^2) <= 2^(-14-52) = 2^-66.
#[inline]
fn atan_eval(x: Dekker) -> Dekker {
    let p_hi = x.hi;
    let x_hi_sq = x.hi * x.hi;
    // c0 ~ x_hi^2 * 1/5 - 1/3
    let c0 = f_fmla(
        x_hi_sq,
        f64::from_bits(0x3fc999999999999a),
        f64::from_bits(0xbfd5555555555555),
    );
    // c1 ~ x_hi^2 * 1/9 - 1/7
    let c1 = f_fmla(
        x_hi_sq,
        f64::from_bits(0x3fbc71c71c71c71c),
        f64::from_bits(0xbfc2492492492492),
    );
    // x_hi^3
    let x_hi_3 = x_hi_sq * x.hi;
    // x_hi^4
    let x_hi_4 = x_hi_sq * x_hi_sq;
    // d0 ~ 1/3 - x_hi^2 / 5 + x_hi^4 / 7 - x_hi^6 / 9
    let d0 = f_fmla(x_hi_4, c1, c0);
    // x_lo - x_lo * x_hi^2 + x_lo * x_hi^4
    let d1 = f_fmla(x_hi_4 - x_hi_sq, x.lo, x.lo);
    // p.lo ~ -x_hi^3/3 + x_hi^5/5 - x_hi^7/7 + x_hi^9/9 +
    //        + x_lo * (1 - x_hi^2 + x_hi^4)
    let p_lo = f_fmla(x_hi_3, d0, d1);
    Dekker::new(p_lo, p_hi)
}

/// Computes atan in double precision
///
/// Max found ULP 0.5
#[inline]
pub fn f_atan2(y: f64, x: f64) -> f64 {
    const IS_NEG: [f64; 2] = [1.0, -1.0];
    const ZERO: Dekker = Dekker::new(0.0, 0.0);
    const MZERO: Dekker = Dekker::new(-0.0, -0.0);
    const PI: Dekker = Dekker::new(
        f64::from_bits(0x3ca1a62633145c07),
        f64::from_bits(0x400921fb54442d18),
    );
    const MPI: Dekker = Dekker::new(
        f64::from_bits(0xbca1a62633145c07),
        f64::from_bits(0xc00921fb54442d18),
    );
    const PI_OVER_2: Dekker = Dekker::new(
        f64::from_bits(0x3c91a62633145c07),
        f64::from_bits(0x3ff921fb54442d18),
    );
    const MPI_OVER_2: Dekker = Dekker::new(
        f64::from_bits(0xbc91a62633145c07),
        f64::from_bits(0xbff921fb54442d18),
    );
    const PI_OVER_4: Dekker = Dekker::new(
        f64::from_bits(0x3c81a62633145c07),
        f64::from_bits(0x3fe921fb54442d18),
    );
    const THREE_PI_OVER_4: Dekker = Dekker::new(
        f64::from_bits(0x3c9a79394c9e8a0a),
        f64::from_bits(0x4002d97c7f3321d2),
    );

    // Adjustment for constant term:
    //   CONST_ADJ[x_sign][y_sign][recip]
    const CONST_ADJ: [[[Dekker; 2]; 2]; 2] = [
        [[ZERO, MPI_OVER_2], [MZERO, MPI_OVER_2]],
        [[MPI, PI_OVER_2], [MPI, PI_OVER_2]],
    ];

    let x_sign = if x.is_sign_negative() { 1 } else { 0 };
    let y_sign = if y.is_sign_negative() { 1 } else { 0 };
    let x_bits = x.to_bits() & 0x7fff_ffff_ffff_ffff;
    let y_bits = y.to_bits() & 0x7fff_ffff_ffff_ffff;
    let x_abs = x_bits;
    let y_abs = y_bits;
    let recip = x_abs < y_abs;
    let mut min_abs = if recip { x_abs } else { y_abs };
    let mut max_abs = if !recip { x_abs } else { y_abs };
    let mut min_exp = min_abs.wrapping_shr(52);
    let mut max_exp = max_abs.wrapping_shr(52);

    let mut num = f64::from_bits(min_abs);
    let mut den = f64::from_bits(max_abs);

    // Check for exceptional cases, whether inputs are 0, inf, nan, or close to
    // overflow, or close to underflow.
    if max_exp > 0x7ffu64 - 128u64 || min_exp < 128u64 {
        if x.is_nan() || y.is_nan() {
            return f64::NAN;
        }
        let x_except = if x == 0.0 {
            0
        } else if x.is_infinite() {
            2
        } else {
            1
        };
        let y_except = if y == 0.0 {
            0
        } else if y.is_infinite() {
            2
        } else {
            1
        };

        // Exceptional cases:
        //   EXCEPT[y_except][x_except][x_is_neg]
        // with x_except & y_except:
        //   0: zero
        //   1: finite, non-zero
        //   2: infinity
        const EXCEPTS: [[[Dekker; 2]; 3]; 3] = [
            [[ZERO, PI], [ZERO, PI], [ZERO, PI]],
            [[PI_OVER_2, PI_OVER_2], [ZERO, ZERO], [ZERO, PI]],
            [
                [PI_OVER_2, PI_OVER_2],
                [PI_OVER_2, PI_OVER_2],
                [PI_OVER_4, THREE_PI_OVER_4],
            ],
        ];

        if (x_except != 1) || (y_except != 1) {
            let r = EXCEPTS[y_except][x_except][x_sign];
            return f_fmla(IS_NEG[y_sign], r.hi, IS_NEG[y_sign] * r.lo);
        }
        let scale_up = min_exp < 128u64;
        let scale_down = max_exp > 0x7ffu64 - 128u64;
        // At least one input is denormal, multiply both numerator and denominator
        // by some large enough power of 2 to normalize denormal inputs.
        if scale_up {
            num *= f64::from_bits(0x43f0000000000000);
            if !scale_down {
                den *= f64::from_bits(0x43f0000000000000)
            }
        } else if scale_down {
            den *= f64::from_bits(0x3bf0000000000000);
            if !scale_up {
                num *= f64::from_bits(0x3bf0000000000000);
            }
        }

        min_abs = num.to_bits();
        max_abs = den.to_bits();
        min_exp = min_abs.wrapping_shr(52);
        max_exp = max_abs.wrapping_shr(52);
    }
    let final_sign = IS_NEG[if (x_sign != y_sign) != recip { 1 } else { 0 }];
    let const_term = CONST_ADJ[x_sign][y_sign][if recip { 1 } else { 0 }];
    let exp_diff = max_exp - min_exp;
    // We have the following bound for normalized n and d:
    //   2^(-exp_diff - 1) < n/d < 2^(-exp_diff + 1).
    if exp_diff > 54 {
        return f_fmla(
            final_sign,
            const_term.hi,
            final_sign * (const_term.lo + num / den),
        );
    }

    let mut k = (64.0 * num / den).round();
    let idx = k as u64;
    // k = idx / 64
    k *= f64::from_bits(0x3f90000000000000);

    // Range reduction:
    // atan(n/d) - atan(k/64) = atan((n/d - k/64) / (1 + (n/d) * (k/64)))
    //                        = atan((n - d * k/64)) / (d + n * k/64))
    let num_k = Dekker::from_exact_mult(num, k);
    let den_k = Dekker::from_exact_mult(den, k);

    // num_dd = n - d * k
    let num_dd = Dekker::from_exact_add(num - den_k.hi, -den_k.lo);
    // den_dd = d + n * k
    let mut den_dd = Dekker::from_exact_add(den, num_k.hi);
    den_dd.lo += num_k.lo;

    // q = (n - d * k) / (d + n * k)
    let q = Dekker::div(num_dd, den_dd);
    // p ~ atan(q)
    let p = atan_eval(q);

    let vl = ATAN_I[idx as usize];
    let vlo = Dekker::new(f64::from_bits(vl.0), f64::from_bits(vl.1));
    let mut r = Dekker::add(const_term, Dekker::add(vlo, p));
    r.hi *= final_sign;
    r.lo *= final_sign;

    r.hi + r.lo
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atan2() {
        assert_eq!(f_atan2(-5., 2.), -1.1902899496825317);
        assert_eq!(f_atan2(2., -5.), 2.761086276477428);
    }
}
