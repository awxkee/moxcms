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
use crate::math::expf::EXP_TABLE;

const TBLSIZE: usize = 64;

#[repr(align(64))]
struct Exp2Table([(u32, u32); TBLSIZE]);

#[rustfmt::skip]
static EXP2FT: Exp2Table = Exp2Table([(0x3F3504F3, 0xB2D4175E),(0x3F36FD92, 0x3268D5EF),(0x3F38FBAF, 0xB30E8719),(0x3F3AFF5B, 0x3319E7DA),(0x3F3D08A4, 0x333CD82F),(0x3F3F179A, 0x330E1902),(0x3F412C4D, 0x32CCF4D7),(0x3F4346CD, 0x328F330E),(0x3F45672A, 0xB201B5B7),(0x3F478D75, 0x32CCCE34),(0x3F49B9BE, 0x335E937C),(0x3F4BEC15, 0x2FF41909),(0x3F4E248C, 0xB21760EA),(0x3F506334, 0x3283628B),(0x3F52A81E, 0x3340F500),(0x3F54F35B, 0x331202BD),(0x3F5744FD, 0x32B66A3E),(0x3F599D16, 0x32D0D9B1),(0x3F5BFBB8, 0x332ED93F),(0x3F5E60F5, 0x3350A709),(0x3F60CCDF, 0x32025744),(0x3F633F89, 0xB33A7C4D),(0x3F65B907, 0x321DA4E9),(0x3F68396A, 0xB2FF36A7),(0x3F6AC0C7, 0x3217E40E),(0x3F6D4F30, 0xB2400CBB),(0x3F6FE4BA, 0x331A2ACC),(0x3F728177, 0xB2B7D3E5),(0x3F75257D, 0xB1FED2BE),(0x3F77D0DF, 0xB32B73BA),(0x3F7A83B3, 0x32579081),(0x3F7D3E0C, 0xB19726B5),(0x3F800000, 0x00000000),(0x3F8164D2, 0x320C09FB),(0x3F82CD87, 0x3391E031),(0x3F843A29, 0x33287EEF),(0x3F85AAC3, 0xB38F6665),(0x3F871F62, 0x339004AB),(0x3F88980F, 0x33AC4561),(0x3F8A14D5, 0xB39CDAEA),(0x3F8B95C2, 0x32949D5C),(0x3F8D1ADF, 0xB36F79FA),(0x3F8EA43A, 0x33971DC2),(0x3F9031DC, 0xB32BD022),(0x3F91C3D3, 0xB3928952),(0x3F935A2B, 0xB2EBFECF),(0x3F94F4F0, 0x3357B8BB),(0x3F96942D, 0xB307353B),(0x3F9837F0, 0xB345DFE9),(0x3F99E046, 0x3382A804),(0x3F9B8D3A, 0x3326993E),(0x3F9D3EDA, 0x3350A029),(0x3F9EF532, 0xB3605F62),(0x3FA0B051, 0xB210909B),(0x3FA27043, 0xB0DDC369),(0x3FA43516, 0x33385844),(0x3FA5FED7, 0x33400757),(0x3FA7CD94, 0x3325446E),(0x3FA9A15B, 0x33237A50),(0x3FAB7A3A, 0x33201CA4),(0x3FAD583F, 0x32394687),(0x3FAF3B79, 0x332E1225),(0x3FB123F6, 0x33838969),(0x3FB311C4, 0xB219F2BA)]);

/* ULP 0.508 method
  let redux = f32::from_bits(0x4b400000) / TBLSIZE as f32;

  let ui = f32::to_bits(d + redux);
  let mut i0 = ui;
  i0 = i0.wrapping_add(TBLSIZE as u32 / 2);
  let k = i0 / TBLSIZE as u32;
  i0 &= TBLSIZE as u32 - 1;
  let mut uf = f32::from_bits(ui);
  uf -= redux;

  let item = EXP2FT.0[i0 as usize];
  let z0: f32 = f32::from_bits(item.0);
  let z1: f32 = f32::from_bits(item.1);

  let f: f32 = d - uf - z1;

  let mut u = 0.055504108664458832;
  u = f_fmlaf(u, f, 0.24022650695908768);
  u = f_fmlaf(u, f, 0.69314718055994973);
  u *= f;

  let i2 = pow2if(k as i32);
  f_fmlaf(u, z0, z0) * i2
*/

/// Computing exp2f using FMA
/// ULP 0.4999994
#[inline]
pub fn f_exp2f(x: f32) -> f32 {
    let mut t = x.to_bits();
    if (t & 0xffff) == 0 {
        // x maybe integer
        let k: i32 = (((t >> 23) & 0xff) as i32).wrapping_sub(127); // 2^k <= |x| < 2^(k+1)
        if k >= 0 && k < 9 && (t << (9i32.wrapping_add(k))) == 0 {
            // x integer, with 1 <= |x| < 2^9
            let msk = (t as i32) >> 31;
            let mut m: i32 = (((t & 0x7fffff) | (1 << 23)) >> (23 - k)) as i32;
            m = (m ^ msk).wrapping_sub(msk).wrapping_add(127);
            if m > 0 && m < 255 {
                t = (m as u32).wrapping_shl(23);
                return f32::from_bits(t);
            } else if m <= 0 && m > -23 {
                t = 1i32.wrapping_shl(22i32.wrapping_add(m) as u32) as u32;
                return f32::from_bits(t);
            }
        }
    }
    let ux = t.wrapping_shl(1);
    if ux >= 0x86000000u32 || ux < 0x65000000u32 {
        // |x| >= 128 or x=nan or |x| < 0x1p-26
        if ux < 0x65000000u32 {
            return 1.0 + x;
        } // |x| < 0x1p-26
        // if x < -149 or 128 <= x is special
        if !(t >= 0xc3000000u32 && t < 0xc3150000u32) {
            if ux >= 0xffu32 << 24 {
                // x is inf or nan
                if ux > (0xffu32 << 24) {
                    return x + x;
                } // x = nan
                static IR: [f32; 2] = [f32::INFINITY, 0.];
                return IR[(t >> 31) as usize]; // x = +-inf
            }
            if t >= 0xc3150000u32 {
                // x < -149
                let z = x;
                let mut y = f_fmla(
                    z as f64 + 149.,
                    f64::from_bits(0x3690000000000000),
                    f64::from_bits(0x36a0000000000000),
                );
                y = y.max(f64::from_bits(0x3680000000000000));
                return y as f32;
            }
            // now x >= 128
            let r = f64::from_bits(0x47e0000000000000) * f64::from_bits(0x47e0000000000000);
            return r as f32;
        }
    }
    let offd = f64::from_bits(0x42d8000000000000);
    let xd: f64 = x as f64;
    let h: f64 = xd - ((xd + offd) - offd);
    let h2 = h * h;
    let u: u32 = (x + f32::from_bits(0x48400000)).to_bits();
    let mut sv = EXP_TABLE[(u & 0x3f) as usize];
    sv = sv.wrapping_add(((u as u64) >> 6).wrapping_shl(52));

    if ux <= 0x79e7526eu32 {
        if t == 0x3b429d37u32 {
            return f32::from_bits(0x3f804385) - f32::from_bits(0x33000000);
        }
        if t == 0xbcf3a937u32 {
            return f32::from_bits(0x3f7ac6b1) - f32::from_bits(0x32800000);
        }
        if t == 0xb8d3d026u32 {
            return f32::from_bits(0x3f7ffb69) + f32::from_bits(0x32800000);
        }
    }

    const C: [u64; 6] = [
        0x3fe62e42fefa39ef,
        0x3fcebfbdff82c58f,
        0x3fac6b08d702e0ed,
        0x3f83b2ab6fb92e5e,
        0x3f55d886e6d54203,
        0x3f2430976b8ce6ef,
    ];

    let ru0 = f_fmla(h, f64::from_bits(C[5]), f64::from_bits(C[4]));
    let ru1 = f_fmla(h, f64::from_bits(C[3]), f64::from_bits(C[2]));
    let ru2 = f_fmla(h, f64::from_bits(C[1]), f64::from_bits(C[0]));

    let rz0 = f_fmla(h2, ru0, ru1);
    let rz1 = f_fmla(h2, rz0, ru2);

    let r = f_fmla(f64::from_bits(sv) * h, rz1, f64::from_bits(sv));
    r as f32
}

#[inline]
pub(crate) fn dirty_exp2f(d: f32) -> f32 {
    let redux = f32::from_bits(0x4b400000) / TBLSIZE as f32;

    let ui = f32::to_bits(d + redux);
    let mut i0 = ui;
    i0 = i0.wrapping_add(TBLSIZE as u32 / 2);
    let k = i0 / TBLSIZE as u32;
    i0 &= TBLSIZE as u32 - 1;
    let mut uf = f32::from_bits(ui);
    uf -= redux;

    let item = EXP2FT.0[i0 as usize];
    let z0: f32 = f32::from_bits(item.0);

    let f: f32 = d - uf;

    let mut u = 0.24022650695908768;
    u = f_fmlaf(u, f, 0.69314718055994973);
    u *= f;

    let i2 = pow2if(k as i32);
    f_fmlaf(u, z0, z0) * i2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exp2f() {
        assert_eq!(f_exp2f(2.0), 4.0);
        assert_eq!(f_exp2f(3.0), 8.0);
        assert_eq!(f_exp2f(4.0), 16.0);
        assert_eq!(f_exp2f(10.0), 1024.0);
        assert_eq!(f_exp2f(-10.0), 0.0009765625);
        assert!(f_exp2f(f32::NAN).is_nan());
        assert_eq!(f_exp2f(-0.35), 0.7845841);
        assert_eq!(f_exp2f(0.35), 1.2745606);
        assert!(f_exp2f(f32::INFINITY).is_infinite());
        assert_eq!(f_exp2f(f32::NEG_INFINITY), 0.0);
    }

    #[test]
    fn test_dirty_exp2f() {
        assert!((dirty_exp2f(0.35f32) - 0.35f32.exp2()).abs() < 1e-5);
        assert!((dirty_exp2f(-0.6f32) - (-0.6f32).exp2()).abs() < 1e-5);
    }
}
