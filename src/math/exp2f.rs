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
#[cfg(not(any(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "fma"
    ),
    all(target_arch = "aarch64", target_feature = "neon")
)))]
use crate::math::estrin::*;

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
        let x2 = f * f;
        let u = poly3!(f, x2, 0.24022652, 0.69314718, 0.1000000000e+1);
        let i2 = pow2if(k as i32);
        u * i2 * z
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
}
