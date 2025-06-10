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
pub(crate) fn tan_reduce1(z: f32) -> (f64, i64) {
    let x = z;
    let idl = -f64::from_bits(0x3dfb1bbead603d8b) * x as f64;
    let idh = f64::from_bits(0x3fe45f306e000000) * x as f64;
    let id = idh.round_ties_even();
    ((idh - id) + idl, id as i64)
}

#[inline]
pub(crate) fn tan_reduce_big(u: u32) -> (f64, i64) {
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
    let k = (e as i32).wrapping_sub(127);
    let s = k.wrapping_sub(23);
    /* in tan is called in the case 127+28 <= e < 0xff
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
    (z, i as i64)
}

/// Computes tan
///
/// Max found ULP 0.4999999
#[inline]
pub fn f_tanf(x: f32) -> f32 {
    let t = x.to_bits();
    let e = (t.wrapping_shr(23)) & 0xff;
    let i;
    let z;
    if e < 127 + 28 {
        // |x| < 2^28
        if e < 115 {
            // |x| < 2^-13
            if e < 102 {
                // |x| < 2^-26
                return f_fmlaf(x, x.abs(), x);
            }
            let x2 = x * x;
            return f_fmlaf(x, f64::from_bits(0x3fd5555560000000) as f32 * x2, x);
        }
        (z, i) = tan_reduce1(x);
    } else if e < 0xff {
        (z, i) = tan_reduce_big(t);
    } else {
        if (t.wrapping_shl(9)) != 0 {
            return x + x;
        } // nan
        return f32::INFINITY; // inf
    }
    let z2 = z * z;
    let z4 = z2 * z2;
    const CN: [u64; 4] = [
        0x3ff921fb54442d18,
        0xbfdfd226e573289f,
        0x3f9b7a60c8dac9f6,
        0xbf2725beb40f33e5,
    ];
    const CD: [u64; 4] = [
        0x3ff0000000000000,
        0xbff2395347fb829d,
        0x3fc2313660f29c36,
        0xbf69a707ab98d1c1,
    ];
    const S: [f64; 2] = [0., 1.];
    let mut n = f_fmla(z2, f64::from_bits(CN[1]), f64::from_bits(CN[0]));
    let n2 = f_fmla(z2, f64::from_bits(CN[3]), f64::from_bits(CN[2]));
    n = f_fmla(z4, n2, n);
    let mut d = f_fmla(z2, f64::from_bits(CD[1]), f64::from_bits(CD[0]));
    let d2 = f_fmla(z2, f64::from_bits(CD[3]), f64::from_bits(CD[2]));
    d = f_fmla(z4, d2, d);
    n *= z;
    let s0 = S[(i & 1) as usize];
    let s1 = S[(1 - (i & 1)) as usize];
    let r1 = f_fmla(n, s1, -d * s0) / f_fmla(n, s0, d * s1);
    let tr = r1.to_bits();
    let tail = (tr.wrapping_add(7)) & 0x000000001fffffff;
    if tail <= 14 {
        static ST: [(u32, u32, u32); 8] = [
            (0x3f8a1f62, 0x3feefcfb, 0xa5c48e92),
            (0x4d56d355, 0x3e740182, 0x22a0cfa3),
            (0x57d7b0ed, 0x3eb068e4, 0xa416b61d),
            (0x5980445e, 0x3fe50f68, 0x257b0298),
            (0x63fc86fe, 0x3f2cbfce, 0x25492cbc),
            (0x6a662711, 0xc0799ac2, 0x266b92a5),
            (0x6ad36709, 0xbf62b097, 0xa513619f),
            (0x72b505bb, 0xbff2150f, 0xa58ee483),
        ];
        let ax: u32 = t & 0x000000007fffffff;
        let sgn = t.wrapping_shr(31);
        for i in ST.iter() {
            if i.0 == ax {
                return if sgn != 0 {
                    -f32::from_bits(i.1) - f32::from_bits(i.2)
                } else {
                    f32::from_bits(i.1) + f32::from_bits(i.2)
                };
            }
        }
    }
    r1 as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f_tanf_test() {
        assert_eq!(f_tanf(0.0), 0.0);
        assert_eq!(f_tanf(1.0), 1.5574077);
        assert_eq!(f_tanf(-1.0), -1.5574077);
        assert_eq!(f_tanf(10.0), 0.64836085);
        assert_eq!(f_tanf(-10.0), -0.64836085);
    }
}
