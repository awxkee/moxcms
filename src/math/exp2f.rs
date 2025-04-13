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

const TBLSIZE: usize = 64;

#[repr(align(64))]
struct Exp2Table([(u32, u32); TBLSIZE]);

#[rustfmt::skip]
static EXP2FT: Exp2Table = Exp2Table([(0x3F3504F3, 0xB2D4175E),(0x3F36FD92, 0x3268D5EF),(0x3F38FBAF, 0xB30E8719),(0x3F3AFF5B, 0x3319E7DA),(0x3F3D08A4, 0x333CD82F),(0x3F3F179A, 0x330E1902),(0x3F412C4D, 0x32CCF4D7),(0x3F4346CD, 0x328F330E),(0x3F45672A, 0xB201B5B7),(0x3F478D75, 0x32CCCE34),(0x3F49B9BE, 0x335E937C),(0x3F4BEC15, 0x2FF41909),(0x3F4E248C, 0xB21760EA),(0x3F506334, 0x3283628B),(0x3F52A81E, 0x3340F500),(0x3F54F35B, 0x331202BD),(0x3F5744FD, 0x32B66A3E),(0x3F599D16, 0x32D0D9B1),(0x3F5BFBB8, 0x332ED93F),(0x3F5E60F5, 0x3350A709),(0x3F60CCDF, 0x32025744),(0x3F633F89, 0xB33A7C4D),(0x3F65B907, 0x321DA4E9),(0x3F68396A, 0xB2FF36A7),(0x3F6AC0C7, 0x3217E40E),(0x3F6D4F30, 0xB2400CBB),(0x3F6FE4BA, 0x331A2ACC),(0x3F728177, 0xB2B7D3E5),(0x3F75257D, 0xB1FED2BE),(0x3F77D0DF, 0xB32B73BA),(0x3F7A83B3, 0x32579081),(0x3F7D3E0C, 0xB19726B5),(0x3F800000, 0x00000000),(0x3F8164D2, 0x320C09FB),(0x3F82CD87, 0x3391E031),(0x3F843A29, 0x33287EEF),(0x3F85AAC3, 0xB38F6665),(0x3F871F62, 0x339004AB),(0x3F88980F, 0x33AC4561),(0x3F8A14D5, 0xB39CDAEA),(0x3F8B95C2, 0x32949D5C),(0x3F8D1ADF, 0xB36F79FA),(0x3F8EA43A, 0x33971DC2),(0x3F9031DC, 0xB32BD022),(0x3F91C3D3, 0xB3928952),(0x3F935A2B, 0xB2EBFECF),(0x3F94F4F0, 0x3357B8BB),(0x3F96942D, 0xB307353B),(0x3F9837F0, 0xB345DFE9),(0x3F99E046, 0x3382A804),(0x3F9B8D3A, 0x3326993E),(0x3F9D3EDA, 0x3350A029),(0x3F9EF532, 0xB3605F62),(0x3FA0B051, 0xB210909B),(0x3FA27043, 0xB0DDC369),(0x3FA43516, 0x33385844),(0x3FA5FED7, 0x33400757),(0x3FA7CD94, 0x3325446E),(0x3FA9A15B, 0x33237A50),(0x3FAB7A3A, 0x33201CA4),(0x3FAD583F, 0x32394687),(0x3FAF3B79, 0x332E1225),(0x3FB123F6, 0x33838969),(0x3FB311C4, 0xB219F2BA)]);

/// Computing exp2f using FMA
/// ULP 0.50175375
#[inline]
pub fn f_exp2f(d: f32) -> f32 {
    let redux = f32::from_bits(0x4b400000) / TBLSIZE as f32;

    let ui = f32::to_bits(d + redux);
    let mut i0 = ui;
    i0 += TBLSIZE as u32 / 2;
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
}

#[inline]
pub(crate) fn dirty_exp2f(d: f32) -> f32 {
    let redux = f32::from_bits(0x4b400000) / TBLSIZE as f32;

    let ui = f32::to_bits(d + redux);
    let mut i0 = ui;
    i0 += TBLSIZE as u32 / 2;
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
        assert!((f_exp2f(0.35f32) - 0.35f32.exp2()).abs() < 1e-5);
        assert!((f_exp2f(-0.6f32) - (-0.6f32).exp2()).abs() < 1e-5);
    }

    #[test]
    fn test_dirty_exp2f() {
        assert!((dirty_exp2f(0.35f32) - 0.35f32.exp2()).abs() < 1e-5);
        assert!((dirty_exp2f(-0.6f32) - (-0.6f32).exp2()).abs() < 1e-5);
    }
}
