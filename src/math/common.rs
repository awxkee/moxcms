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
use crate::mlaf::mlaf;
use num_traits::MulAdd;
use std::ops::{Add, Mul};

#[inline]
pub(crate) const fn rintfk(x: f32) -> f32 {
    (if x < 0. { x - 0.5 } else { x + 0.5 }) as i32 as f32
}

#[inline(always)]
pub(crate) const fn fmlaf(a: f32, b: f32, c: f32) -> f32 {
    c + a * b
}

#[inline(always)]
pub(crate) fn f_fmlaf(a: f32, b: f32, c: f32) -> f32 {
    mlaf(c, a, b)
}

#[inline(always)]
pub(crate) const fn fmla(a: f64, b: f64, c: f64) -> f64 {
    c + a * b
}

#[inline(always)]
pub(crate) fn f_fmla(a: f64, b: f64, c: f64) -> f64 {
    mlaf(c, a, b)
}

#[allow(dead_code)]
#[inline(always)]
pub(crate) fn c_mlaf<T: Copy + Mul<T, Output = T> + Add<T, Output = T> + MulAdd<T, Output = T>>(
    a: T,
    b: T,
    c: T,
) -> T {
    mlaf(c, a, b)
}

/// Copies sign from `y` to `x`
#[inline]
pub(crate) const fn copysignfk(x: f32, y: f32) -> f32 {
    f32::from_bits((x.to_bits() & !(1 << 31)) ^ (y.to_bits() & (1 << 31)))
}

/// Copies sign from `y` to `x`
#[inline]
pub(crate) const fn copysign(x: f64, y: f64) -> f64 {
    f64::from_bits((x.to_bits() & !(1 << 63)) ^ (y.to_bits() & (1 << 63)))
}

// #[inline]
// // Founds n in ln(𝑥)=ln(𝑎)+𝑛ln(2)
// pub(crate) const fn ilogb2kf(d: f32) -> i32 {
//     (((d.to_bits() as i32) >> 23) & 0xff) - 0x7f
// }
//
// #[inline]
// // Founds a in x=a+𝑛ln(2)
// pub(crate) const fn ldexp3kf(d: f32, n: i32) -> f32 {
//     f32::from_bits(((d.to_bits() as i32) + (n << 23)) as u32)
// }

#[inline]
pub(crate) const fn pow2if(q: i32) -> f32 {
    f32::from_bits((q.wrapping_add(0x7f) as u32) << 23)
}

/// Round towards whole integral number
#[inline]
pub(crate) const fn rintk(x: f64) -> f64 {
    (if x < 0. { x - 0.5 } else { x + 0.5 }) as i64 as f64
}

/// Computes 2^n
#[inline(always)]
pub(crate) const fn pow2i(q: i32) -> f64 {
    f64::from_bits((q.wrapping_add(0x3ff) as u64) << 52)
}

// #[inline]
// pub(crate) const fn ilogb2k(d: f64) -> i32 {
//     (((d.to_bits() >> 52) & 0x7ff) as i32) - 0x3ff
// }
//
// #[inline]
// pub(crate) const fn ldexp3k(d: f64, e: i32) -> f64 {
//     f64::from_bits(((d.to_bits() as i64) + ((e as i64) << 52)) as u64)
// }

/// Copies sign from `y` to `x`
#[inline]
pub(crate) const fn copysignk(x: f64, y: f64) -> f64 {
    f64::from_bits((x.to_bits() & !(1 << 63)) ^ (y.to_bits() & (1 << 63)))
}
