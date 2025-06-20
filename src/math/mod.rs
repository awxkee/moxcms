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
#![allow(clippy::approx_constant, clippy::manual_range_contains)]
mod acos;
mod acosf;
mod asin;
mod asinf;
mod atan;
mod atan2;
mod atan2f;
mod atanf;
mod cbrt;
mod cbrtf;
mod common;
mod cosf;
mod coshf;
mod dekker;
mod estrin;
mod exp;
mod exp10;
mod exp10f;
mod exp2;
mod exp2f;
mod expf;
mod float106;
mod float48;
mod floor;
mod hypot;
mod log;
mod log10;
mod log10f;
mod log2;
mod log2f;
mod logf;
mod pow;
mod powf;
mod sin;
mod sincos;
mod sincosf;
mod sinf;
mod sinhf;
mod sqrtf;
mod tan;
mod tanf;
mod tanhf;
mod trigo;

pub use acos::f_acos;
pub use acosf::f_acosf;
pub use asin::f_asin;
pub use asinf::f_asinf;
pub use atan::f_atan;
pub use atan2::f_atan2;
pub use atan2f::f_atan2f;
pub use atanf::{atan2f, atanf, f_atanf};
pub use cbrt::f_cbrt;
pub use cbrtf::{cbrtf, f_cbrtf};
pub(crate) use common::{copysign, copysignfk};
pub use cosf::f_cosf;
pub use coshf::f_coshf;
pub use exp::{exp, f_exp};
pub use exp2::f_exp2;
pub use exp2f::f_exp2f;
pub use exp10::f_exp10;
pub use exp10f::f_exp10f;
pub use expf::{expf, f_expf};
pub use float48::Float48;
pub use float106::Float106;
pub use floor::{floor, floorf};
pub(crate) use hypot::hypot3f;
pub use hypot::{const_hypotf, hypotf};
pub use log::{f_log, log};
pub use log2::f_log2;
pub use log2f::f_log2f;
pub use log10::f_log10;
pub use log10f::log10f;
pub use logf::{f_logf, logf};
use num_traits::Num;
pub use pow::{f_pow, pow};
pub(crate) use powf::dirty_powf;
pub use powf::{f_powf, powf};
pub use sin::{f_cos, f_sin};
pub use sincos::f_sincos;
pub use sincosf::f_sincosf;
pub use sinf::f_sinf;
pub use sinhf::f_sinhf;
pub use sqrtf::sqrtf;
pub use tan::f_tan;
pub use tanf::f_tanf;
pub use tanhf::f_tanhf;
pub use trigo::{cosf, sinf};

#[inline(always)]
pub const fn rounding_div_ceil(value: i32, div: i32) -> i32 {
    (value + div - 1) / div
}

// Generic function for max
#[inline(always)]
pub(crate) fn m_max<T: Num + PartialOrd>(a: T, b: T) -> T {
    if a > b { a } else { b }
}

// Generic function for min
#[inline(always)]
pub(crate) fn m_min<T: Num + PartialOrd>(a: T, b: T) -> T {
    if a < b { a } else { b }
}

#[inline]
pub(crate) fn m_clamp<T: Num + PartialOrd>(a: T, min: T, max: T) -> T {
    if a > max {
        max
    } else if a >= min {
        a
    } else {
        // a < min or a is NaN
        min
    }
}

pub trait FusedMultiplyAdd<T> {
    fn mla(&self, b: T, c: T) -> T;
}

pub(crate) trait FusedMultiplyNegAdd<T> {
    fn neg_mla(&self, b: T, c: T) -> T;
}
