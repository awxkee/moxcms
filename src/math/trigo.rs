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

const PI_A2: f32 = 3.1414794921875f32;
const PI_B2: f32 = 0.00011315941810607910156f32;
const PI_C2: f32 = 1.9841872589410058936e-09f32;

#[inline]
const fn isnegzerof(x: f32) -> bool {
    x.to_bits() == (-0.0f32).to_bits()
}

/// Computes cosine for given value
#[inline]
pub const fn cosf(d: f32) -> f32 {
    let q = 1 + 2 * rintfk(std::f32::consts::FRAC_1_PI * d - 0.5) as i32;
    let qf = q as f32;
    let mut r = fmlaf(qf, -PI_A2 * 0.5, d);
    r = fmlaf(qf, -PI_B2 * 0.5, r);
    r = fmlaf(qf, -PI_C2 * 0.5, r);

    let x2 = r * r;

    if q & 2 == 0 {
        r = -r;
    }

    let mut u = 2.6083159809786593541503e-06f32;
    u = fmlaf(u, x2, -0.0001981069071916863322258f32);
    u = fmlaf(u, x2, 0.00833307858556509017944336f32);
    u = fmlaf(u, x2, -0.166666597127914428710938f32);
    u = fmlaf(u, x2 * r, r);
    if isnegzerof(d) {
        return -0.;
    }
    u
}

/// Computes cosine for given value
#[inline]
pub fn f_cosf(d: f32) -> f32 {
    let q = 1 + 2 * f_fmlaf(std::f32::consts::FRAC_1_PI, d, -0.5).round() as i32;
    let qf = q as f32;
    let mut r = f_fmlaf(qf, -PI_A2 * 0.5, d);
    r = f_fmlaf(qf, -PI_B2 * 0.5, r);
    r = f_fmlaf(qf, -PI_C2 * 0.5, r);

    let x2 = r * r;

    if q & 2 == 0 {
        r = -r;
    }

    let mut u;
    #[cfg(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    ))]
    {
        u = 2.6083159809786593541503e-06f32;
        u = f_fmlaf(u, x2, -0.0001981069071916863322258f32);
        u = f_fmlaf(u, x2, 0.00833307858556509017944336f32);
        u = f_fmlaf(u, x2, -0.166666597127914428710938f32);
    }
    #[cfg(not(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        u = poly4!(
            x2,
            x2 * x2,
            2.6083159809786593541503e-06f32,
            -0.0001981069071916863322258f32,
            0.00833307858556509017944336f32,
            -0.166666597127914428710938f32
        );
    }
    u = f_fmlaf(u, x2 * r, r);
    if isnegzerof(d) {
        return -0.;
    }
    u
}

/// Sine function
#[inline]
pub const fn sinf(d: f32) -> f32 {
    let qf = rintfk(std::f32::consts::FRAC_1_PI * d);
    let q = qf as i32;
    let mut r = fmlaf(qf, -PI_A2, d);
    r = fmlaf(qf, -PI_B2, r);
    r = fmlaf(qf, -PI_C2, r);

    let x2 = r * r;

    if (q & 1) != 0 {
        r = -r;
    }

    let mut u = 2.6083159809786593541503e-06f32;
    u = fmlaf(u, x2, -0.0001981069071916863322258f32);
    u = fmlaf(u, x2, 0.00833307858556509017944336f32);
    u = fmlaf(u, x2, -0.166666597127914428710938f32);
    u = fmlaf(u, x2 * r, r);
    if isnegzerof(d) {
        return -0f32;
    }
    u
}

/// Sine function using FMA
#[inline]
pub fn f_sinf(d: f32) -> f32 {
    let qf = (std::f32::consts::FRAC_1_PI * d).round();
    let q = qf as i32;
    let mut r = f_fmlaf(qf, -PI_A2, d);
    r = f_fmlaf(qf, -PI_B2, r);
    r = f_fmlaf(qf, -PI_C2, r);

    let x2 = r * r;

    if (q & 1) != 0 {
        r = -r;
    }

    let mut u;
    #[cfg(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    ))]
    {
        u = 2.6083159809786593541503e-06f32;
        u = f_fmlaf(u, x2, -0.0001981069071916863322258f32);
        u = f_fmlaf(u, x2, 0.00833307858556509017944336f32);
        u = f_fmlaf(u, x2, -0.166666597127914428710938f32);
    }
    #[cfg(not(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        u = poly4!(
            x2,
            x2 * x2,
            2.6083159809786593541503e-06f32,
            -0.0001981069071916863322258f32,
            0.00833307858556509017944336f32,
            -0.166666597127914428710938f32
        );
    }
    u = f_fmlaf(u, x2 * r, r);
    if isnegzerof(d) {
        return -0f32;
    }
    u
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn cosf_test() {
        assert_eq!(cosf(0.0), 1.0);
        assert_eq!(cosf(std::f32::consts::PI), -1f32);

        assert_eq!(f_cosf(0.0), 1.0);
        assert_eq!(f_cosf(std::f32::consts::PI), -1f32);
    }

    #[test]
    fn sinf_test() {
        assert_eq!(sinf(0.0), 0.0);
        assert!((sinf(std::f32::consts::PI) - 0f32).abs() < 1e-6);
        assert!((sinf(std::f32::consts::FRAC_PI_2) - 1f32).abs() < 1e-6);

        assert_eq!(f_sinf(0.0), 0.0);
        assert!((f_sinf(std::f32::consts::PI) - 0f32).abs() < 1e-6);
        assert!((f_sinf(std::f32::consts::FRAC_PI_2) - 1f32).abs() < 1e-6);
    }
}
