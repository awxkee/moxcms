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

#[inline(always)]
const fn halley_refine(x: f32, a: f32) -> f32 {
    let tx = x * x * x;
    x * (tx + 2f32 * a) / (2f32 * tx + a)
}

#[allow(dead_code)]
#[inline(always)]
fn f_halley_refine(x: f32, a: f32) -> f32 {
    let tx = x * x * x;
    x * f_fmlaf(2f32, a, tx) / f_fmlaf(2f32, tx, a)
}

#[inline(always)]
fn halley_refine_d(x: f64, a: f64) -> f64 {
    let tx = x * x * x;
    x * f_fmla(2., a, tx) / f_fmla(2., tx, a)
}

/// Computes Cube Root
#[inline]
pub const fn cbrtf(x: f32) -> f32 {
    if x == 0. {
        return x;
    }
    // if x == f32::INFINITY {
    //     return f32::INFINITY;
    // }
    // if x == f32::NEG_INFINITY {
    //     return f32::NEG_INFINITY;
    // }

    const B1: u32 = 709958130;
    let mut t: f32;
    let mut ui: u32 = x.to_bits();
    let mut hx: u32 = ui & 0x7fffffff;

    hx = hx / 3 + B1;
    ui &= 0x80000000;
    ui |= hx;

    t = f32::from_bits(ui);
    t = halley_refine(t, x);
    halley_refine(t, x)
}

/// Computes Cube Root using FMA
///
/// Peak ULP on 64 bit = 0.49999577
#[inline]
pub fn f_cbrtf(x: f32) -> f32 {
    #[cfg(native_64_word)]
    {
        if x == 0. {
            return x;
        }
        // if x == f32::INFINITY {
        //     return f32::INFINITY;
        // }
        // if x == f32::NEG_INFINITY {
        //     return f32::NEG_INFINITY;
        // }

        const B1: u32 = 709958130;
        let mut ui: u32 = x.to_bits();
        let mut hx: u32 = ui & 0x7fffffff;

        hx = (hx / 3).overflowing_add(B1).0;
        ui &= 0x80000000;
        ui |= hx;

        let mut t = f32::from_bits(ui) as f64;
        t = halley_refine_d(t, x as f64);
        halley_refine_d(t, x as f64) as f32
    }
    #[cfg(not(native_64_word))]
    {
        if x == 0. {
            return x;
        }
        // if x == f32::INFINITY {
        //     return f32::INFINITY;
        // }
        // if x == f32::NEG_INFINITY {
        //     return f32::NEG_INFINITY;
        // }

        const B1: u32 = 709958130;
        let mut t: f32;
        let mut ui: u32 = x.to_bits();
        let mut hx: u32 = ui & 0x7fffffff;

        hx = (hx / 3).overflowing_add(B1).0;
        ui &= 0x80000000;
        ui |= hx;

        t = f32::from_bits(ui);
        t = f_halley_refine(t, x);
        f_halley_refine(t, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fcbrtf() {
        assert_eq!(f_cbrtf(0.0), 0.0);
        assert_eq!(f_cbrtf(-27.0), -3.0);
        assert_eq!(f_cbrtf(27.0), 3.0);
    }

    #[test]
    fn test_cbrtf() {
        assert_eq!(cbrtf(0.0), 0.0);
        assert_eq!(cbrtf(-27.0), -3.0);
        assert_eq!(cbrtf(27.0), 3.0);
    }
}
