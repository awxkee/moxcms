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
use crate::math::common::{f_fmla, f_fmlaf};
use crate::math::cosf::{sincos_reduce_big, sincos_reduce0, sincos_reduce1};

static TB: [u64; 32] = [
    0x0000000000000000,
    0x3fc8f8b83c69a60b,
    0x3fd87de2a6aea963,
    0x3fe1c73b39ae68c8,
    0x3fe6a09e667f3bcd,
    0x3fea9b66290ea1a3,
    0x3fed906bcf328d46,
    0x3fef6297cff75cb0,
    0x3ff0000000000000,
    0x3fef6297cff75cb0,
    0x3fed906bcf328d46,
    0x3fea9b66290ea1a3,
    0x3fe6a09e667f3bcd,
    0x3fe1c73b39ae68c8,
    0x3fd87de2a6aea963,
    0x3fc8f8b83c69a60b,
    0x0000000000000000,
    0xbfc8f8b83c69a60b,
    0xbfd87de2a6aea963,
    0xbfe1c73b39ae68c8,
    0xbfe6a09e667f3bcd,
    0xbfea9b66290ea1a3,
    0xbfed906bcf328d46,
    0xbfef6297cff75cb0,
    0xbff0000000000000,
    0xbfef6297cff75cb0,
    0xbfed906bcf328d46,
    0xbfea9b66290ea1a3,
    0xbfe6a09e667f3bcd,
    0xbfe1c73b39ae68c8,
    0xbfd87de2a6aea963,
    0xbfc8f8b83c69a60b,
];

#[inline]
fn add_sign(x: f32, rh: f32, rl: f32) -> f32 {
    let sgn = f32::copysign(1.0, x);
    f_fmlaf(sgn, rh, sgn * rl)
}

#[inline]
fn search_from_table(x: f32, r: f64) -> f32 {
    static ST: [(u32, u32, u32); 4] = [
        (0x46199998, 0xbeb1fa5d, 0xb2000000),
        (0x3f3adc51, 0x3f2ab445, 0xb2800000),
        (0x3fa7832a, 0x3f7741b6, 0xb2800000),
        (0x4116cbe4, 0xb2ccde2d, 0xa6000000),
    ];

    let t = x.to_bits();
    let ax = t & 0x7fffffff;
    for i in ST.iter() {
        if i.0 == ax {
            return add_sign(x, f32::from_bits(i.1), f32::from_bits(i.2));
        }
    }
    r as f32
}

#[inline]
fn as_sinf_big(x: f32) -> f32 {
    const B: [u64; 4] = [
        0x3f93bd3cc9be45dc,
        0xbf103c1f081b0833,
        0x3e755d3c6fc9ac1f,
        0xbdce1d3ff281b40d,
    ];
    const A: [u64; 4] = [
        0x3fc921fb54442d17,
        0xbf54abbce6256a39,
        0x3ec466bc5a518c16,
        0xbe232bdc61074ff6,
    ];
    let t = x.to_bits();
    let ax = t.wrapping_shl(1);
    if ax >= 0xffu32 << 24 {
        // nan or +-inf
        if ax.wrapping_shl(8) != 0 {
            return x + x;
        }; // nan
        return f32::NAN; // to raise FE_INVALID
    }
    let (z, ia) = sincos_reduce_big(t);
    let z2 = z * z;
    let z4 = z2 * z2;

    let w0 = f_fmla(z2, f64::from_bits(A[1]), f64::from_bits(A[0]));
    let w1 = f_fmla(z2, f64::from_bits(A[3]), f64::from_bits(A[2]));

    let aa = f_fmla(z4, w1, w0);

    let q0 = f_fmla(z2, f64::from_bits(B[1]), f64::from_bits(B[0]));
    let q1 = f_fmla(z2, f64::from_bits(B[3]), f64::from_bits(B[2]));

    let bb = f_fmla(z4, q1, q0);

    let s0 = f64::from_bits(TB[(ia & 31) as usize]);
    let c0 = f64::from_bits(TB[((ia.wrapping_add(8)) & 31) as usize]);

    let f0 = f_fmla(-bb, z * s0, aa * c0);
    let r = f_fmla(z, f0, s0);
    r as f32
}

/// Sine function using FMA
///
/// Max found ULP 0.4999996
#[inline]
pub fn f_sinf(x: f32) -> f32 {
    let t = x.to_bits();
    let ax = t.wrapping_shl(1);
    let ia;
    let z0 = x;
    let z: f64;
    #[allow(clippy::manual_range_contains)]
    if ax > 0x99000000u32 || ax < 0x73000000u32 {
        // |x| > 6.71089e+07 or |x| < 0.000244141
        if ax < 0x73000000u32 {
            // |x| < 0.000244141
            if ax < 0x66000000u32 {
                // |x| < 2.98023e-08
                if ax == 0u32 {
                    return x;
                }
                let res = f_fmlaf(-x, x.abs(), x);
                return res;
            }
            return (-f64::from_bits(0x3fc5555560000000) as f32 * x) * (x * x) + x;
        }
        return as_sinf_big(x);
    }

    const B: [u64; 4] = [
        0x3f93bd3cc9be45dc,
        0xbf103c1f081b0833,
        0x3e755d3c6fc9ac1f,
        0xbdce1d3ff281b40d,
    ];
    const A: [u64; 4] = [
        0x3fc921fb54442d17,
        0xbf54abbce6256a39,
        0x3ec466bc5a518c16,
        0xbe232bdc61074ff6,
    ];

    if ax < 0x822d97c8u32 {
        if ax == 0x7e75b8a2u32 || ax == 0x7f4f0654u32 {
            return search_from_table(x, 0.0);
        };
        (z, ia) = sincos_reduce0(z0);
    } else {
        if ax == 0x8c333330u32 {
            return search_from_table(x, 0.0);
        }
        (z, ia) = sincos_reduce1(z0);
    }
    let z2 = z * z;
    let z4 = z2 * z2;

    let w0 = f_fmla(z2, f64::from_bits(A[1]), f64::from_bits(A[0]));
    let w1 = f_fmla(z2, f64::from_bits(A[3]), f64::from_bits(A[2]));

    let aa = f_fmla(z4, w1, w0);

    let q0 = f_fmla(z2, f64::from_bits(B[1]), f64::from_bits(B[0]));
    let q1 = f_fmla(z2, f64::from_bits(B[3]), f64::from_bits(B[2]));

    let bb = f_fmla(z4, q1, q0);

    let s0 = f64::from_bits(TB[(ia & 31) as usize]);
    let c0 = f64::from_bits(TB[((ia.wrapping_add(8)) & 31) as usize]);

    let f0 = f_fmla(aa, z * c0, s0);
    let r = f_fmla(-bb, z2 * s0, f0);
    r as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f_sinf_test() {
        assert_eq!(f_sinf(0.0), 0.0);
        assert!((f_sinf(std::f32::consts::PI) - 0f32).abs() < 1e-6);
        assert!((f_sinf(std::f32::consts::FRAC_PI_2) - 1f32).abs() < 1e-6);
    }
}
