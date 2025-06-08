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
/// Max found ULP 0.7263569920658313
#[inline]
pub fn f_log2(d: f64) -> f64 {
    const IVLN2HI: f64 = 1.44269504072144627571e+00;
    const IVLN2LO: f64 = 1.67517131648865118353e-10;

    const LG1: u64 = 0x3fe5555555555594;
    const LG2: u64 = 0x3fd999999997f6f8;
    const LG3: u64 = 0x3fd2492494241370;
    const LG4: u64 = 0x3fcc71c51d01b16c;
    const LG5: u64 = 0x3fc74664992e5112;
    const LG6: u64 = 0x3fc39a0bb5f6a888;
    const LG7: u64 = 0x3fc2f0edc7587e42;

    // reduce into [sqrt(2)/2;sqrt(2)]
    let mut ui: u64 = d.to_bits();
    let mut hx = (ui >> 32) as u32;
    hx = hx.wrapping_add(0x3ff00000 - 0x3fe6a09e);
    let n = (hx >> 20) as i32 - 0x3ff;
    hx = (hx & 0x000fffff).wrapping_add(0x3fe6a09e);
    ui = (hx as u64) << 32 | (ui & 0xffffffff);
    let a = f64::from_bits(ui);

    let f = a - 1.0;
    let hfsq = 0.5 * f * f;
    let s = f / (2.0 + f);
    let z = s * s;
    let w = z * z;
    let t1;
    let t2;
    #[cfg(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    ))]
    {
        let mut u = f_fmla(w, f64::from_bits(LG6), f64::from_bits(LG4));
        u = f_fmla(w, u, f64::from_bits(LG2));
        t1 = u * w;

        let mut u1 = f_fmla(w, f64::from_bits(LG7), f64::from_bits(LG5));
        u1 = f_fmla(u1, w, f64::from_bits(LG3));
        u1 = f_fmla(u1, w, f64::from_bits(LG1));
        t2 = z * u1;
    }
    #[cfg(not(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        t1 = w * (f64::from_bits(LG2) + w * (f64::from_bits(LG4) + w * f64::from_bits(LG6)));
        t2 = z
            * (f64::from_bits(LG1)
                + w * (f64::from_bits(LG3) + w * (f64::from_bits(LG5) + w * f64::from_bits(LG7))))
    }
    let r = t2 + t1;

    let mut hi = f - hfsq;
    ui = hi.to_bits();
    ui &= (-1i64 as u64) << 32;
    hi = f64::from_bits(ui);
    let lo = f_fmla(hfsq + r, s, f - hi - hfsq);

    /* val_hi+val_lo ~ log10(1+f) + k*log10(2) */
    let mut val_hi = hi * IVLN2HI;
    let dk = n as f64;
    let y = dk;
    let mut val_lo = f_fmla(lo + hi, IVLN2LO, lo * IVLN2HI);

    let w = y + val_hi;
    val_lo += (y - w) + val_hi;
    val_hi = w;

    if d == 0f64 {
        f64::NEG_INFINITY
    } else if (d < 0.) || d.is_nan() {
        f64::NAN
    } else if d.is_infinite() {
        f64::INFINITY
    } else {
        val_lo + val_hi
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log2d() {
        println!("f_log2 {}", f_log2(2.));
        println!("{}", f_log2(34.5));
        assert!((f_log2(0.35) - 0.35f64.log2()).abs() < 1e-8);
        assert!((f_log2(0.9) - 0.9f64.log2()).abs() < 1e-8);
    }
}
