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
use crate::math::float106::Float106;

/// Natural logarithm using FMA
///
/// ULP under 1.0
#[inline(always)]
pub fn f_log10(d: f64) -> f64 {
    // reduce into [sqrt(2)/2;sqrt(2)]
    let mut ui: u64 = d.to_bits();
    let mut hx = (ui >> 32) as u32;
    hx = hx.wrapping_add(0x3ff00000 - 0x3fe6a09e);
    let n = (hx >> 20) as i32 - 0x3ff;
    hx = (hx & 0x000fffff).wrapping_add(0x3fe6a09e);
    ui = (hx as u64) << 32 | (ui & 0xffffffff);
    let a = f64::from_bits(ui);

    let a106 = Float106::from_f64(a);

    let x = (a106 - 1.) / (a106 + 1.);

    let rx2 = x.v0 * x.v0;
    let x2 = rx2;

    #[cfg(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    ))]
    {
        let mut u = 0.6408793956596637101e-1;
        u = f_fmla(u, x2, 0.6652520676426221507e-1);
        u = f_fmla(u, x2, 0.7896955655948678827e-1);
        u = f_fmla(u, x2, 0.9650979419682287602e-1);
        u = f_fmla(u, x2, 0.1240841383264986008e+0);
        u = f_fmla(u, x2, 0.1737177927590776300e+0);
        u = f_fmla(u, x2, 0.2895296546021709390e+0);
        let s = x.fast_mul_f64(0.8685889638065036542e+0)
            + Float106::new(0.3010299956639812, -2.8037281277851704e-18) * n as f64;
        (x.v0 * (x2 * u) + s).to_f64()
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
        let rx2 = x2 * x2;
        let rx4 = rx2 * rx2;
        let u = poly7!(
            x2,
            rx2,
            rx4,
            0.6408793956596637101e-1,
            0.6652520676426221507e-1,
            0.7896955655948678827e-1,
            0.9650979419682287602e-1,
            0.1240841383264986008e+0,
            0.1737177927590776300e+0,
            0.2895296546021709390e+0
        );
        let s = x.fast_mul_f64(0.8685889638065036542e+0)
            + Float106::new(0.3010299956639812, -2.8037281277851704e-18) * n as f64;
        (x.v0 * (x2 * u) + s).to_f64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log10d() {
        println!("{}", f_log10(10.));
        let mut max_diff = f64::MIN;
        let mut max_away = 0;

        for i in 1..20000 {
            let my_expf = f_log10(i as f64 / 1000.);
            let system = (i as f64 / 1000.).log10();
            max_diff = max_diff.max((my_expf - system).abs());
            max_away = (my_expf.to_bits() as i64 - system.to_bits() as i64)
                .abs()
                .max(max_away);
        }
        println!("{} max away {}", max_diff, max_away);
        assert!((f_log10(0.35) - 0.35f64.log10()).abs() < 1e-8);
        assert!((f_log10(0.9) - 0.9f64.log10()).abs() < 1e-8);
        assert!((f_log10(10.) - 10f64.log10()).abs() < 1e-8);
    }
}
