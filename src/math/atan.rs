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

/// Computes Atan
#[inline]
pub const fn atanf(d: f32) -> f32 {
    let mut x = d;
    let q = if x < 0f32 {
        x = -x;
        1
    } else {
        0
    };
    let c = x;
    if x > 1f32 {
        x = 1f32 / x;
    }
    let x2 = x * x;

    let mut u = 0.3057095382e-2;
    u = fmlaf(u, x2, -0.1684093114e-1);
    u = fmlaf(u, x2, 0.4385302239e-1);
    u = fmlaf(u, x2, -0.7594467979e-1);
    u = fmlaf(u, x2, 0.1067925170e+0);
    u = fmlaf(u, x2, -0.1421231870e+0);
    u = fmlaf(u, x2, 0.1999354698e+0);
    u = fmlaf(u, x2, -0.3333310690e+0);
    u = x + x * (x2 * u);

    u = if c > 1f32 {
        std::f32::consts::FRAC_PI_2 - u
    } else {
        u
    };
    if q & 1 != 0 {
        u = -u;
    }
    u
}

/// Computes Atan using FMA
#[inline]
pub fn f_atanf(d: f32) -> f32 {
    let mut x = d;
    let q = if x < 0f32 {
        x = -x;
        1
    } else {
        0
    };
    let c = x;
    if x > 1f32 {
        x = 1f32 / x;
    }
    let x2 = x * x;

    let mut u;
    #[cfg(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    ))]
    {
        u = 0.3057095382e-2;
        u = f_fmlaf(u, x2, -0.1684093114e-1);
        u = f_fmlaf(u, x2, 0.4385302239e-1);
        u = f_fmlaf(u, x2, -0.7594467979e-1);
        u = f_fmlaf(u, x2, 0.1067925170e+0);
        u = f_fmlaf(u, x2, -0.1421231870e+0);
        u = f_fmlaf(u, x2, 0.1999354698e+0);
        u = f_fmlaf(u, x2, -0.3333310690e+0);
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
        u = poly8!(
            x2,
            rx2,
            rx4,
            0.3057095382e-2,
            -0.1684093114e-1,
            0.4385302239e-1,
            -0.7594467979e-1,
            0.1067925170e+0,
            -0.1421231870e+0,
            0.1999354698e+0,
            -0.3333310690e+0
        );
    }
    u = f_fmlaf(x2 * u, x, x);

    u = if c > 1f32 {
        std::f32::consts::FRAC_PI_2 - u
    } else {
        u
    };
    if q & 1 != 0 {
        u = -u;
    }
    u
}

/// Computes Atan2
#[inline]
pub const fn atan2f(y: f32, x: f32) -> f32 {
    if x == 0. {
        if y > 0. {
            return std::f32::consts::FRAC_PI_2;
        }
        if y < 0. {
            return -std::f32::consts::FRAC_PI_2;
        }
        if y == 0. {
            return 0f32;
        }
    }
    let rad = atanf(y / x);
    if x > 0f32 {
        rad
    } else if x < 0f32 && y >= 0f32 {
        std::f32::consts::PI + rad
    } else {
        // if x < 0. && y < 0.
        -std::f32::consts::PI + rad
    }
}

/// Computes Atan2 using FMA
#[inline]
pub fn f_atan2f(y: f32, x: f32) -> f32 {
    if x == 0. {
        if y > 0. {
            return std::f32::consts::FRAC_PI_2;
        }
        if y < 0. {
            return -std::f32::consts::FRAC_PI_2;
        }
        if y == 0. {
            return 0f32;
        }
    }
    let rad = f_atanf(y / x);
    if x > 0f32 {
        rad
    } else if x < 0f32 && y >= 0f32 {
        std::f32::consts::PI + rad
    } else {
        // if x < 0. && y < 0.
        -std::f32::consts::PI + rad
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atan_test() {
        assert!(
            (atanf(1.0) - std::f32::consts::PI / 4f32).abs() < 1e-6,
            "Invalid result {}",
            atanf(1f32)
        );
        assert!(
            (atanf(2f32) - 1.107148717794090503017065f32).abs() < 1e-6,
            "Invalid result {}",
            atanf(2f32)
        );
        assert!(
            (atanf(5f32) - 1.3734007669450158608612719264f32).abs() < 1e-6,
            "Invalid result {}",
            atanf(5f32)
        );
    }
}
