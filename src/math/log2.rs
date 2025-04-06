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

/// Natural logarithm using FMA
#[inline]
pub fn f_log2(d: f64) -> f64 {
    let n = ilogb2k(d * (1. / 0.75));
    let a = ldexp3k(d, -n);

    let x = (a - 1.) / (a + 1.);

    let x2 = x * x;
    #[cfg(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    ))]
    {
        let mut u = 0.2210319873572944675e+0;
        u = f_fmla(u, x2, 0.2201017466118781220e+0);
        u = f_fmla(u, x2, 0.2623693760780589253e+0);
        u = f_fmla(u, x2, 0.3205977867563723840e+0);
        u = f_fmla(u, x2, 0.4121985940253306314e+0);
        u = f_fmla(u, x2, 0.5770780163029655546e+0);
        u = f_fmla(u, x2, 0.9617966939260729972e+0);
        f_fmla(x2 * x, u, f_fmla(x, 0.2885390081777926774e+1, n as f64))
    }
    #[cfg(not(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        let rx2 = x2 * x2;
        let rx4 = rx2 * rx2;
        let u = poly7!(
            x2,
            rx2,
            rx4,
            0.2210319873572944675e+0,
            0.2201017466118781220e+0,
            0.2623693760780589253e+0,
            0.3205977867563723840e+0,
            0.4121985940253306314e+0,
            0.5770780163029655546e+0,
            0.9617966939260729972e+0
        );
        f_fmla(x2 * x, u, f_fmla(x, 0.2885390081777926774e+1, n as f64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log2d() {
        let mut max_diff = f64::MIN;
        let mut max_away = 0;
        for i in 1..20000 {
            let my_expf = f_log2(i as f64 / 1000.);
            let system = (i as f64 / 1000.).log2();
            max_diff = max_diff.max((my_expf - system).abs());
            max_away = (my_expf.to_bits() as i64 - system.to_bits() as i64)
                .abs()
                .max(max_away);
        }
        println!("{} max away {}", max_diff, max_away);
        assert!((f_log2(0.35) - 0.35f64.log2()).abs() < 1e-8);
        assert!((f_log2(0.9) - 0.9f64.log2()).abs() < 1e-8);
    }
}
