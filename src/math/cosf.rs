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

#[inline]
pub(crate) fn sincos_reduce1(z: f32) -> (f64, i32) {
    let x = z;
    let idl = -f64::from_bits(0x3e2b1bbead603d8b) * x as f64;
    let idh = f64::from_bits(0x40145f306e000000) * x as f64;
    let id = idh.round_ties_even();
    let q = (f64::from_bits(0x4338000000000000) + id).to_bits();
    ((idh - id) + idl, q as i32)
}

#[inline]
pub(crate) fn sincos_reduce0(x: f32) -> (f64, i32) {
    let idh = f64::from_bits(0x40145f306dc9c883) * x as f64;
    let id = idh.round_ties_even();
    let q = (f64::from_bits(0x4338000000000000) + id).to_bits();
    (idh - id, q as i32)
}

static TB: [u64; 32] = [
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
    0x0000000000000000,
    0x3fc8f8b83c69a60b,
    0x3fd87de2a6aea963,
    0x3fe1c73b39ae68c8,
    0x3fe6a09e667f3bcd,
    0x3fea9b66290ea1a3,
    0x3fed906bcf328d46,
    0x3fef6297cff75cb0,
];

#[inline]
fn search_from_table(x: f32, r: f64) -> f32 {
    static ST: [(u32, u32, u32); 5] = [
        (0x4096cbe4, 0x324cde2e, 0xa5800000),
        (0x5922aa80, 0x3f08aebf, 0xb2800000),
        (0x5aa4542c, 0x3efa40a4, 0x32000000),
        (0x5f18b878, 0x3f7f14bb, 0x32800000),
        (0x6115cb11, 0x3f78142f, 0xb2800000),
    ];

    let t = x.to_bits();
    let ax = t & 0x7fffffff;
    for i in ST.iter() {
        if i.0 == ax {
            return f32::from_bits(i.1) + f32::from_bits(i.2);
        }
    }
    r as f32
}

#[inline]
pub(crate) fn sincos_reduce_big(u: u32) -> (f64, i32) {
    const IPI: [u64; 4] = [
        0xfe5163abdebbc562,
        0xdb6295993c439041,
        0xfc2757d1f534ddc0,
        0xa2f9836e4e441529,
    ];
    let e = (u >> 23) & 0xff;
    let m: u64 = ((u as u64) & 0x7fffff) | (1 << 23);
    let p0 = m as u128 * IPI[0] as u128;
    let mut p1 = m as u128 * IPI[1] as u128;
    p1 = p1.wrapping_add(p0.wrapping_shr(64));
    let mut p2 = m as u128 * IPI[2] as u128;
    p2 = p2.wrapping_add(p1.wrapping_shr(64));
    let mut p3 = m as u128 * IPI[3] as u128;
    p3 = p3.wrapping_add(p2.wrapping_shr(64));
    let p3h = p3.wrapping_shr(64) as u64;
    let p3l = p3 as u64;
    let p2l = p2 as u64;
    let p1l = p1 as u64;
    let a: i64;
    let k = (e as i32).wrapping_sub(124);
    let s = k.wrapping_sub(23);
    /* in cr_cosf(), rbig() is called in the case 127+28 <= e < 0xff
    thus 155 <= e <= 254, which yields 28 <= k <= 127 and 5 <= s <= 104 */
    let mut i: i32;
    if s < 64 {
        i = (p3h << s | p3l >> (64 - s)) as i32;
        a = (p3l << s | p2l >> (64 - s)) as i64;
    } else if s == 64 {
        i = p3l as i32;
        a = p2l as i64;
    } else {
        /* s > 64 */
        i = (p3l << (s - 64) | p2l >> (128 - s)) as i32;
        a = (p2l << (s - 64) | p1l >> (128 - s)) as i64;
    }
    let sgn: i32 = (u as i32).wrapping_shr(31);
    let sm: i64 = a.wrapping_shr(63);
    i = i.wrapping_sub(sm as i32);
    let z = (a ^ sgn as i64) as f64 * f64::from_bits(0x3bf0000000000000);
    i = (i ^ sgn).wrapping_sub(sgn);
    (z, i)
}

#[inline]
fn as_cosf_big(x: f32) -> f32 {
    let t = x.to_bits();
    let ax = t.wrapping_shl(1);
    if ax >= 0xffu32 << 24 {
        if ax << 8 != 0 {
            return x + x;
        }

        return f32::NAN;
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
    let (z, ia) = sincos_reduce_big(t);
    let z2 = z * z;
    let z4 = z2 * z2;

    let w0 = f_fmla(z2, f64::from_bits(A[1]), f64::from_bits(A[0]));
    let w1 = f_fmla(z2, f64::from_bits(A[3]), f64::from_bits(A[2]));

    let aa = f_fmla(z4, w1, w0);

    let q0 = f_fmla(z2, f64::from_bits(B[1]), f64::from_bits(B[0]));
    let q1 = f_fmla(z2, f64::from_bits(B[3]), f64::from_bits(B[2]));

    let bb = f_fmla(z4, q1, q0);

    let s0 = f64::from_bits(TB[((ia.wrapping_add(8i32)) & 31) as usize]);
    let c0 = f64::from_bits(TB[(ia & 31) as usize]);

    let g0 = f_fmla(aa, s0, -bb * (z * c0));

    let r = f_fmla(z, g0, c0);
    let tr = r.to_bits();
    let tail: u64 = (tr.wrapping_add(6)) & 0xfffffff;
    if tail <= 12 {
        return search_from_table(x, r);
    }
    r as f32
}

/// Computes cosine function
///
/// Max found ULP 0.49999967
///
/// Working argument range [-1000000..1000000]
#[inline]
pub fn f_cosf(x: f32) -> f32 {
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
    let (z, ia);
    let z0 = x;
    if ax > 0x99000000u32 || ax < 0x73000000 {
        if ax < 0x73000000 {
            if ax < 0x66000000u32 {
                if ax == 0u32 {
                    return 1.0;
                };
                return 1.0 - f64::from_bits(0x3e60000000000000) as f32;
            }
            return f_fmlaf(-f64::from_bits(0x3fe0000000000000) as f32 * x, x, 1.0);
        }
        return as_cosf_big(x);
    }
    if ax < 0x82a41896u32 {
        if ax == 0x812d97c8u32 {
            return search_from_table(x, 0.0);
        };
        (z, ia) = sincos_reduce0(z0);
    } else {
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

    let c0 = f64::from_bits(TB[(ia & 31) as usize]);
    let s0 = f64::from_bits(TB[(ia.wrapping_add(8) & 31) as usize]);

    let n0 = f_fmla(bb, -(z2 * c0), c0);

    let r = f_fmla(aa, z * s0, n0);
    r as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f_cosf_test() {
        assert_eq!(f_cosf(0.0), 1.0);
        assert_eq!(f_cosf(std::f32::consts::PI), -1f32);
    }
}
