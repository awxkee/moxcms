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

/// Natural logarithm using FMA
///
/// Max found ULP 0.5055166
#[inline]
pub fn f_log2f(d: f32) -> f32 {
    f_log2fx(d) as f32
}

/// Natural logarithm using FMA
#[inline(always)]
#[allow(dead_code)]
pub(crate) fn f_log2fx(d: f32) -> f64 {
    let mut ix = d.to_bits();
    /* reduce x into [sqrt(2)/2, sqrt(2)] */
    ix = ix.wrapping_add(0x3f800000 - 0x3f3504f3);
    let n = (ix >> 23) as i32 - 0x7f;
    ix = (ix & 0x007fffff).wrapping_add(0x3f3504f3);
    let a = f32::from_bits(ix) as f64;

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
        let mut u = 0.3205986261348816382e+0;
        u = f_fmla(u, x2, 0.4121985850084821691e+0);
        u = f_fmla(u, x2, 0.5770780163490337802e+0);
        u = f_fmla(u, x2, 0.9617966939259845749e+0);
        f_fmla(x2 * x, u, f_fmla(x, 0.2885390081777926802e+1, n as f64))
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
        let u = poly4!(
            x2,
            rx2,
            0.3205986261348816382e+0,
            0.4121985850084821691e+0,
            0.5770780163490337802e+0,
            0.9617966939259845749e+0
        );
        f_fmla(x2 * x, u, f_fmla(x, 0.2885390081777926802e+1, n as f64))
    }
}

/// Natural logarithm using FMA
#[inline(always)]
#[allow(dead_code)]
pub(crate) fn dirty_log2f(d: f32) -> f32 {
    let mut ix = d.to_bits();
    /* reduce x into [sqrt(2)/2, sqrt(2)] */
    ix = ix.wrapping_add(0x3f800000 - 0x3f3504f3);
    let n = (ix >> 23) as i32 - 0x7f;
    ix = (ix & 0x007fffff).wrapping_add(0x3f3504f3);
    let a = f32::from_bits(ix);

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
        let mut u = 0.4121985850084821691e+0;
        u = f_fmlaf(u, x2, 0.5770780163490337802e+0);
        u = f_fmlaf(u, x2, 0.9617966939259845749e+0);
        f_fmlaf(x2 * x, u, f_fmlaf(x, 0.2885390081777926802e+1, n as f32))
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
        let u = poly3!(
            x2,
            rx2,
            0.4121985850084821691e+0,
            0.5770780163490337802e+0,
            0.9617966939259845749e+0
        );
        f_fmlaf(x2 * x, u, f_fmlaf(x, 0.2885390081777926802e+1, n as f32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log2f() {
        assert!((f_log2f(0.35f32) - 0.35f32.log2()).abs() < 1e-5);
        assert!((f_log2f(0.9f32) - 0.9f32.log2()).abs() < 1e-5);
    }

    #[test]
    fn test_dirty_log2f() {
        assert!((dirty_log2f(0.35f32) - 0.35f32.log2()).abs() < 1e-5);
        assert!((dirty_log2f(0.9f32) - 0.9f32.log2()).abs() < 1e-5);
    }
}
