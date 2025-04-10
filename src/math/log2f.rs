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

/// Natural logarithm using FMA
#[inline]
pub fn f_log2f(d: f32) -> f32 {
    #[cfg(native_64_word)]
    {
        f_log2fx(d) as f32
    }
    #[cfg(not(native_64_word))]
    {
        f_log2f48(d).to_f32()
    }
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
        f_fmla(x2 * x, u, f_fmla(x,0.2885390081777926802e+1, n as f64))
    }
}

/// Natural logarithm using FMA
#[inline(always)]
#[allow(dead_code)]
pub(crate) fn f_log2f48(d: f32) -> Float48 {
    let mut ix = d.to_bits();
    /* reduce x into [sqrt(2)/2, sqrt(2)] */
    ix = ix.wrapping_add(0x3f800000 - 0x3f3504f3);
    let n = (ix >> 23) as i32 - 0x7f;
    ix = (ix & 0x007fffff).wrapping_add(0x3f3504f3);
    let a = f32::from_bits(ix);
    
    let a48 = Float48::from_f32(a);

    let x = (a48 - 1.) / (a48 + 1.);

    let x2 = x.v0 * x.v0;
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
    x.fast_mul_f32(x2 * u) + (x.fast_mul_f32(0.2885390081777926802e+1) + n as f32)
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
    fn test_log2f48() {
        println!("{}", f_log2f48(0.35f32).to_f32());
        println!("{}", 0.35f32.log2());

        let mut max_diff = f32::MIN;
        let mut max_away = 0;
        for i in 1..20000 {
            let my_expf = f_log2f48(i as f32 / 1000.).to_f32();
            let system = (i as f32 / 1000.).log2();
            max_diff = max_diff.max((my_expf - system).abs());
            max_away = (my_expf.to_bits() as i64 - system.to_bits() as i64)
                .abs()
                .max(max_away);
        }
        println!("{} max away {}", max_diff, max_away);
        assert!((f_log2f48(0.35f32).to_f32() - 0.35f32.log2()).abs() < 1e-5);
        assert!((f_log2f48(0.9f32).to_f32() - 0.9f32.log2()).abs() < 1e-5);
    }
}
