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
use crate::sqrtf;

#[inline]
pub(crate) fn hypot3f(x: f32, y: f32, z: f32) -> f32 {
    let x = x.abs();
    let y = y.abs();
    let z = z.abs();

    let max = x.max(y).max(z);

    if max == 0.0 {
        return 0.0;
    }

    let recip_max = 1. / max;

    let norm_x = x * recip_max;
    let norm_y = y * recip_max;
    let norm_z = z * recip_max;

    max * (norm_x * norm_x + norm_y * norm_y + norm_z * norm_z).sqrt()

    // if x == f32::INFINITY || y == f32::INFINITY || z == f32::INFINITY {
    //     f32::INFINITY
    // } else if x.is_nan() || y.is_nan() || z.is_nan() || ret.is_nan() {
    //     f32::NAN
    // // } else {
    // ret
    // }
}

/// Hypot suitable for const context
#[inline]
pub const fn const_hypotf(x: f32, y: f32) -> f32 {
    let x = x.abs();
    let y = y.abs();
    let max = x.max(y);
    let min = x.min(y);
    let r = min / max;
    let ret = max * sqrtf(1f32 + r * r);

    if (x == f32::INFINITY) || (y == f32::INFINITY) {
        f32::INFINITY
    } else if x.is_nan() || y.is_nan() || ret.is_nan() {
        f32::NAN
    } else if min == 0. {
        max
    } else {
        ret
    }
}

/// Hypot function
#[inline]
pub fn hypotf(x: f32, y: f32) -> f32 {
    let x = x.abs();
    let y = y.abs();
    let max = x.max(y);
    let min = x.min(y);
    let r = min / max;
    let ret = max * mlaf(1., r, r).sqrt();

    // if (x == f32::INFINITY) || (y == f32::INFINITY) {
    //     f32::INFINITY
    // } else if x.is_nan() || y.is_nan() || ret.is_nan() {
    //     f32::NAN
    // } else if min == 0. {
    //     max
    // } else {
    if min == 0. { max } else { ret }
    // }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hypotf() {
        let dx = (hypotf(1f32, 1f32) - (1f32 * 1f32 + 1f32 * 1f32).sqrt()).abs();
        assert!(dx < 1e-5);
        let dx = (hypotf(5f32, 5f32) - (5f32 * 5f32 + 5f32 * 5f32).sqrt()).abs();
        assert!(dx < 1e-5);
    }

    #[test]
    fn test_c_hypotf() {
        let dx = (const_hypotf(1f32, 1f32) - (1f32 * 1f32 + 1f32 * 1f32).sqrt()).abs();
        assert!(dx < 1e-5);
        let dx = (const_hypotf(5f32, 5f32) - (5f32 * 5f32 + 5f32 * 5f32).sqrt()).abs();
        assert!(dx < 1e-5);
    }
}
