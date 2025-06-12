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

pub(crate) static EXP_TABLE: [u64; 64] = [
    0x3ff0000000000000,
    0x3ff02c9a3e778061,
    0x3ff059b0d3158574,
    0x3ff0874518759bc8,
    0x3ff0b5586cf9890f,
    0x3ff0e3ec32d3d1a2,
    0x3ff11301d0125b51,
    0x3ff1429aaea92de0,
    0x3ff172b83c7d517b,
    0x3ff1a35beb6fcb75,
    0x3ff1d4873168b9aa,
    0x3ff2063b88628cd6,
    0x3ff2387a6e756238,
    0x3ff26b4565e27cdd,
    0x3ff29e9df51fdee1,
    0x3ff2d285a6e4030b,
    0x3ff306fe0a31b715,
    0x3ff33c08b26416ff,
    0x3ff371a7373aa9cb,
    0x3ff3a7db34e59ff7,
    0x3ff3dea64c123422,
    0x3ff4160a21f72e2a,
    0x3ff44e086061892d,
    0x3ff486a2b5c13cd0,
    0x3ff4bfdad5362a27,
    0x3ff4f9b2769d2ca7,
    0x3ff5342b569d4f82,
    0x3ff56f4736b527da,
    0x3ff5ab07dd485429,
    0x3ff5e76f15ad2148,
    0x3ff6247eb03a5585,
    0x3ff6623882552225,
    0x3ff6a09e667f3bcd,
    0x3ff6dfb23c651a2f,
    0x3ff71f75e8ec5f74,
    0x3ff75feb564267c9,
    0x3ff7a11473eb0187,
    0x3ff7e2f336cf4e62,
    0x3ff82589994cce13,
    0x3ff868d99b4492ed,
    0x3ff8ace5422aa0db,
    0x3ff8f1ae99157736,
    0x3ff93737b0cdc5e5,
    0x3ff97d829fde4e50,
    0x3ff9c49182a3f090,
    0x3ffa0c667b5de565,
    0x3ffa5503b23e255d,
    0x3ffa9e6b5579fdbf,
    0x3ffae89f995ad3ad,
    0x3ffb33a2b84f15fb,
    0x3ffb7f76f2fb5e47,
    0x3ffbcc1e904bc1d2,
    0x3ffc199bdd85529c,
    0x3ffc67f12e57d14b,
    0x3ffcb720dcef9069,
    0x3ffd072d4a07897c,
    0x3ffd5818dcfba487,
    0x3ffda9e603db3285,
    0x3ffdfc97337b9b5f,
    0x3ffe502ee78b3ff6,
    0x3ffea4afa2a490da,
    0x3ffefa1bee615a27,
    0x3fff50765b6e4540,
    0x3fffa7c1819e90d8,
];

/// Computes exp
///
/// Max found ULP 0.4999993
#[inline]
pub fn f_expf(x: f32) -> f32 {
    const C: [u64; 6] = [
        0x3fe62e42fefa39ef,
        0x3fcebfbdff82c58f,
        0x3fac6b08d702e0ed,
        0x3f83b2ab6fb92e5e,
        0x3f55d886e6d54203,
        0x3f2430976b8ce6ef,
    ];
    const B: [u64; 4] = [
        0x3ff0000000000000,
        0x3fe62e42fef4c4e7,
        0x3fcebfd1b232f475,
        0x3fac6b19384ecd93,
    ];
    const ILN2: f64 = f64::from_bits(0x3ff71547652b82fe);
    const BIG: f64 = f64::from_bits(0x42d8000000000000);
    let t = x.to_bits();
    let z = x;
    let a = ILN2 * z as f64;
    let u = (a + BIG).to_bits();
    let ux = t.wrapping_shl(1);
    if !(0x6f93813eu32..=0x8562e42eu32).contains(&ux) {
        // |x| > 88.7228 or x=nan or |x| < 2.40508e-05
        if ux < 0x6f93813eu32 {
            // |x| < 2.40508e-05
            return 1.0 + z * (1. + z * 0.5);
        }
        if ux >= 0xffu32 << 24 {
            // x is inf or nan
            if ux > 0xffu32 << 24 {
                return x + x;
            } // x = nan
            static IR: [f32; 2] = [f32::INFINITY, 0.0];
            return IR[t.wrapping_shr(31) as usize]; // x = +-inf
        }
        if t > 0xc2ce8ec0u32 {
            // x < -103.279
            let zz0 = f_fmla(
                f64::from_bits(0x4059d1d9fccf4770),
                f64::from_bits(0x36971547652b82ed),
                z as f64,
            );
            let mut y = f64::from_bits(0x36a0000000000000) + zz0;
            y = y.max(f64::from_bits(0x3680000000000000));
            return y as f32;
        }
        if (t >> 31) == 0 && t > 0x42b17217u32 {
            // x > 0x1.62e42ep+6
            let r = f64::from_bits(0x47e0000000000000) * f64::from_bits(0x47e0000000000000);
            return r as f32;
        }
    }
    let ia = BIG - f64::from_bits(u);
    let mut h = a + ia;
    let sv = EXP_TABLE[(u & 0x3f) as usize].wrapping_add(u.wrapping_shr(6).wrapping_shl(52));
    let mut h2 = h * h;

    let q0 = f_fmla(h, f64::from_bits(B[3]), f64::from_bits(B[2]));
    let q1 = f_fmla(h, f64::from_bits(B[1]), f64::from_bits(B[0]));

    let mut r = f_fmla(h2, q0, q1) * f64::from_bits(sv);
    let mut ub = r;
    let lb = f_fmla(-r, f64::from_bits(0x3de3edbbe4560327), r);
    // Ziv's accuracy test
    if ub != lb {
        const ILN2H: f64 = f64::from_bits(0x3ff7154765000000);
        const ILN2L: f64 = f64::from_bits(0x3e05c17f0bbbe880);
        let zz0 = f_fmla(ILN2H, z as f64, ia);
        h = f_fmla(ILN2L, z as f64, zz0);
        let s = f64::from_bits(sv);
        h2 = h * h;
        let w = s * h;
        let w0 = f_fmla(h, f64::from_bits(C[5]), f64::from_bits(C[4]));
        let w1 = f_fmla(h, f64::from_bits(C[3]), f64::from_bits(C[2]));
        let w2 = f_fmla(h, f64::from_bits(C[1]), f64::from_bits(C[0]));
        let kq0 = f_fmla(h2, w0, w1);
        let kq1 = f_fmla(h2, kq0, w2);
        r = f_fmla(w, kq1, s);
        ub = r;
    }
    ub as f32
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
    }

    #[test]
    fn f_expf_test() {
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
