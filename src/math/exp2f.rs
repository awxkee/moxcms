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
use crate::Float48;
use crate::math::common::*;

/// Computing exp2f using FMA
#[inline]
pub fn f_exp2f(d: f32) -> f32 {
    const TBLSIZE: usize = 64;
    #[rustfmt::skip]
    const EXP2FT: [f32; TBLSIZE] = [0.70710677, 0.7148067, 0.7225904, 0.7304589, 0.7384131, 0.7464539, 0.7545822, 0.7627991, 0.7711054, 0.7795022, 0.78799045, 0.7965711, 0.80524516, 0.8140137, 0.82287776, 0.8318383, 0.8408964, 0.8500532, 0.8593097, 0.86866695, 0.8781261, 0.8876882, 0.89735454, 0.90712607, 0.91700405, 0.92698956, 0.93708384, 0.947288, 0.9576033, 0.96803087, 0.9785721, 0.989228, 1.0, 1.0108893, 1.0218972, 1.0330249, 1.0442737, 1.0556452, 1.0671405, 1.0787607, 1.0905077, 1.1023825, 1.1143868, 1.1265216, 1.1387886, 1.1511892, 1.1637249, 1.176397, 1.1892071, 1.2021568, 1.2152474, 1.2284806, 1.2418578, 1.2553807, 1.269051, 1.28287, 1.2968396, 1.3109612, 1.3252367, 1.3396676, 1.3542556, 1.3690025, 1.38391, 1.3989797];

    let redux = f32::from_bits(0x4b400000) / TBLSIZE as f32;

    let ui = f32::to_bits(d + redux);
    let mut i0 = ui;
    i0 += TBLSIZE as u32 / 2;
    let k = i0 / TBLSIZE as u32;
    i0 &= TBLSIZE as u32 - 1;
    let mut uf = f32::from_bits(ui);
    uf -= redux;
    let f: f32 = d - uf;

    let z: f32 = EXP2FT[i0 as usize];

    #[cfg(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    ))]
    {
        let mut u = 0.24022652;
        u = f_fmlaf(u, f, 0.69314718);
        u = f_fmlaf(u, f, 0.1000000000e+1);

        let i2 = pow2if(k as i32);
        u * i2 * z
    }
    #[cfg(not(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        use crate::math::estrin::*;
        let x2 = f * f;
        let u = poly3!(f, x2, 0.24022652, 0.69314718, 0.1000000000e+1);
        let i2 = pow2if(k as i32);
        u * i2 * z
    }
}

/// Computing exp2f using FMA for 32 bit
#[inline]
#[allow(dead_code)]
pub(crate) fn exp2f48(d: Float48) -> f32 {
    const TBLSIZE: usize = 64;
    #[rustfmt::skip]
    const EXP2FT: [(u32, u32); TBLSIZE] = [(0x3F3504F3, 0x324FE77A),(0x3F36FD92, 0xB1E6B974),(0x3F38FBAF, 0x328EC5F7),(0x3F3AFF5B, 0xB29BD983),(0x3F3D08A4, 0xB2C14FE8),(0x3F3F179A, 0xB2930B1A),(0x3F412C4D, 0xB256663E),(0x3F4346CD, 0xB2176DA2),(0x3F45672A, 0x318AA837),(0x3F478D75, 0xB25D5119),(0x3F49B9BE, 0xB2F323A2),(0x3F4BEC15, 0xAF86C6C2),(0x3F4E248C, 0x31A8FC24),(0x3F506334, 0xB2144353),(0x3F52A81E, 0xB2DC1DAA),(0x3F54F35B, 0xB2A86024),(0x3F5744FD, 0xB254A58A),(0x3F599D16, 0xB2761D41),(0x3F5BFBB8, 0xB2D04A1C),(0x3F5E60F5, 0xB2FB43E3),(0x3F60CCDF, 0xB19EAB59),(0x3F633F89, 0x32E57D15),(0x3F65B907, 0xB1C41BE6),(0x3F68396A, 0x32A07898),(0x3F6AC0C7, 0xB1C116DE),(0x3F6D4F30, 0x31F6CCA1),(0x3F6FE4BA, 0xB2C8464A),(0x3F728177, 0x327167FF),(0x3F75257D, 0x31A92436),(0x3F77D0DF, 0x32E615A2),(0x3F7A83B3, 0xB2123758),(0x3F7D3E0C, 0x314F486C),(0x3F800000, 0x00000000),(0x3F8164D2, 0xB1C43FD0),(0x3F82CD87, 0xB34EA7A9),(0x3F843A29, 0xB2F14C87),(0x3F85AAC3, 0x334F9891),(0x3F871F62, 0xB352C2E6),(0x3F88980F, 0xB37EDA4B),(0x3F8A14D5, 0x336A92DE),(0x3F8B95C2, 0xB260ABA1),(0x3F8D1ADF, 0x3336FCB7),(0x3F8EA43A, 0xB3697465),(0x3F9031DC, 0x330628CD),(0x3F91C3D3, 0x33675624),(0x3F935A2B, 0x32BC4F9C),(0x3F94F4F0, 0xB32E0212),(0x3F96942D, 0x32DC8061),(0x3F9837F0, 0x33231B71),(0x3F99E046, 0xB359BE90),(0x3F9B8D3A, 0xB30C5563),(0x3F9D3EDA, 0xB331A601),(0x3F9EF532, 0x33412342),(0x3FA0B051, 0x31FB9715),(0x3FA27043, 0x30C3125A),(0x3FA43516, 0xB323EC33),(0x3FA5FED7, 0xB32C9D5E),(0x3FA7CD94, 0xB3162D36),(0x3FA9A15B, 0xB3162B08),(0x3FAB7A3A, 0xB314AD82),(0x3FAD583F, 0xB22DEAF6),(0x3FAF3B79, 0xB3252DEB),(0x3FB123F6, 0xB37C5AA8),(0x3FB311C4, 0x32154889)];
    let redux = f32::from_bits(0x4b400000) / TBLSIZE as f32;

    let ui = f32::to_bits(d.v0 + redux);
    let mut i0 = ui;
    i0 += TBLSIZE as u32 / 2;
    let k = i0 / TBLSIZE as u32;
    i0 &= TBLSIZE as u32 - 1;
    let mut uf = f32::from_bits(ui);
    uf -= redux;
    let c = d - uf;

    let cl = EXP2FT[i0 as usize];
    let z = Float48::new(f32::from_bits(cl.0), f32::from_bits(cl.1));

    #[cfg(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    ))]
    {
        let mut u = 0.24022652;
        u = f_fmlaf(u, c.v0, 0.69314718);
        let u = u * c + 1.;

        let i2 = pow2if(k as i32);
        (u * i2 * z).to_f32()
    }
    #[cfg(not(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        use crate::math::estrin::*;
        let u = poly2!(c.v0, 0.24022652, 0.69314718);
        let u = u * c + 1.;

        let i2 = pow2if(k as i32);
        (u * i2 * z).to_f32()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_exp2f() {
        println!("{}", f_exp2f(5.4));
        println!("{}", 5.4f32.exp2());
        let mut max_diff = f32::MIN;
        let mut max_away = 0;
        for i in -10000..10000 {
            let my_expf = f_exp2f(i as f32 / 1000.);
            let system = (i as f32 / 1000.).exp2();
            max_diff = max_diff.max((my_expf - system).abs());
            max_away = (my_expf.to_bits() as i64 - system.to_bits() as i64)
                .abs()
                .max(max_away);
        }
        println!("{} max away {}", max_diff, max_away);
        assert!((f_exp2f(0.35f32) - 0.35f32.exp2()).abs() < 1e-5);
        assert!((f_exp2f(-0.6f32) - (-0.6f32).exp2()).abs() < 1e-5);
    }

    #[test]
    fn test_exp2f48() {
        println!("{}", exp2f48(Float48::from_f32(5.4)));
        println!("{}", 5.4f32.exp2());
        let mut max_diff = f32::MIN;
        let mut max_away = 0;
        for i in -10000..10000 {
            let my_expf = exp2f48(Float48::from_f32(i as f32 / 1000.));
            let system = (i as f32 / 1000.).exp2();
            max_diff = max_diff.max((my_expf - system).abs());
            max_away = (my_expf.to_bits() as i64 - system.to_bits() as i64)
                .abs()
                .max(max_away);
        }
        println!("{} max away {}", max_diff, max_away);
        assert!((f_exp2f(0.35f32) - 0.35f32.exp2()).abs() < 1e-5);
        assert!((f_exp2f(-0.6f32) - (-0.6f32).exp2()).abs() < 1e-5);
    }
}
