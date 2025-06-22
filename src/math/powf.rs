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
use crate::math::logf::EXP_MASK_F32;
use crate::{expf, logf};

/// Power function for given value
#[inline]
pub const fn powf(d: f32, n: f32) -> f32 {
    let value = d.abs();
    let c = expf(n * logf(value));
    if n == 1. {
        return d;
    }
    if d < 0.0 {
        let y = n as i32;
        if y % 2 == 0 { c } else { -c }
    } else {
        c
    }
}

#[inline]
const fn is_integer(x: f32) -> bool {
    let x_u = x.to_bits();
    let x_e = (x_u & EXP_MASK_F32) >> 23;
    let lsb = (x_u | EXP_MASK_F32).trailing_zeros();
    const E_BIAS: u32 = (1u32 << (8 - 1u32)) - 1u32;
    const UNIT_EXPONENT: u32 = E_BIAS + 23;
    x_e + lsb >= UNIT_EXPONENT
}

/// Power function for given value using FMA
///
/// Max found ULP 0.4999
#[inline]
pub fn f_powf(x: f32, y: f32) -> f32 {
    use crate::f_exp2;
    use crate::math::log2f::f_log2fx;

    let x_u = x.to_bits();
    let x_abs = x_u & 0x7fff_ffff;
    let mut y = y;
    let y_u = y.to_bits();
    let y_abs = y_u & 0x7fff_ffff;

    if (y_abs & 0x0007_ffff == 0) || (y_abs > 0x4f170000) {
        // y is signaling NaN
        if x.is_nan() || y.is_nan() {
            return f32::NAN;
        }

        // Exceptional exponents.
        if y == 0.0 {
            return 1.0;
        }

        match y_abs {
            0x7f80_0000 => {
                if x_abs > 0x7f80_0000 {
                    // pow(NaN, +-Inf) = NaN
                    return x;
                }
                if x_abs == 0x3f80_0000 {
                    // pow(+-1, +-Inf) = 1.0f
                    return 1.0;
                }
                if x == 0.0 && y_u == 0xff80_0000 {
                    // pow(+-0, -Inf) = +inf and raise FE_DIVBYZERO
                    return f32::INFINITY;
                }
                // pow (|x| < 1, -inf) = +inf
                // pow (|x| < 1, +inf) = 0.0f
                // pow (|x| > 1, -inf) = 0.0f
                // pow (|x| > 1, +inf) = +inf
                return if (x_abs < 0x3f80_0000) == (y_u == 0xff80_0000) {
                    f32::INFINITY
                } else {
                    0.
                };
            }
            _ => {
                match y_u {
                    0x3f00_0000 => {
                        // pow(x, 1/2) = sqrt(x)
                        if x == 0.0 || x_u == 0xff80_0000 {
                            // pow(-0, 1/2) = +0
                            // pow(-inf, 1/2) = +inf
                            // Make sure it is correct for FTZ/DAZ.
                            return x * x;
                        }
                        let r = x.sqrt();
                        return if r.to_bits() != 0x8000_0000 { r } else { 0.0 };
                    }
                    0x3f80_0000 => {
                        return x;
                    } // y = 1.0f
                    0x4000_0000 => return x * x, // y = 2.0f
                    _ => {
                        let is_int = is_integer(y);
                        if is_int && (y_u > 0x4000_0000) && (y_u <= 0x41c0_0000) {
                            // Check for exact cases when 2 < y < 25 and y is an integer.
                            let mut msb: i32 = if x_abs == 0 {
                                32 - 2
                            } else {
                                x_abs.leading_zeros() as i32
                            };
                            msb = if msb > 8 { msb } else { 8 };
                            let mut lsb: i32 = if x_abs == 0 {
                                0
                            } else {
                                x_abs.trailing_zeros() as i32
                            };
                            lsb = if lsb > 23 { 23 } else { lsb };
                            let extra_bits: i32 = 32 - 2 - lsb - msb;
                            let iter = y as i32;

                            if extra_bits * iter <= 23 + 2 {
                                // The result is either exact or exactly half-way.
                                // But it is exactly representable in double precision.
                                let x_d = x as f64;
                                let mut result = x_d;
                                for _ in 1..iter {
                                    result *= x_d;
                                }
                                return result as f32;
                            }
                        }

                        if y_abs > 0x4f17_0000 {
                            // if y is NaN
                            if y_abs > 0x7f80_0000 {
                                if x_u == 0x3f80_0000 {
                                    // x = 1.0f
                                    // pow(1, NaN) = 1
                                    return 1.0;
                                }
                                // pow(x, NaN) = NaN
                                return y;
                            }
                            // x^y will be overflow / underflow in single precision.  Set y to a
                            // large enough exponent but not too large, so that the computations
                            // won't be overflow in double precision.
                            y = f32::from_bits((y_u & 0x7fff_ffff).wrapping_add(0x4f800000u32));
                        }
                    }
                }
            }
        }
    }

    let lg = f_log2fx(f32::from_bits(x_abs));
    let c = f_exp2(y as f64 * lg) as f32;
    if x < 0.0 {
        let y = y as i32;
        if y % 2 == 0 { c } else { -c }
    } else {
        c
    }
}

/// Power function for given value using FMA
#[inline]
pub(crate) fn dirty_powf(d: f32, n: f32) -> f32 {
    use crate::math::exp2f::dirty_exp2f;
    use crate::math::log2f::dirty_log2f;
    let value = d.abs();
    let lg = dirty_log2f(value);
    let c = dirty_exp2f(n * lg);
    if d < 0.0 {
        let y = n as i32;
        if y % 2 == 0 { c } else { -c }
    } else {
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn powf_test() {
        assert!(
            (powf(2f32, 3f32) - 8f32).abs() < 1e-6,
            "Invalid result {}",
            powf(2f32, 3f32)
        );
        assert!(
            (powf(0.5f32, 2f32) - 0.25f32).abs() < 1e-6,
            "Invalid result {}",
            powf(0.5f32, 2f32)
        );
    }

    #[test]
    fn f_powf_test() {
        println!("{}", f_powf(3., 3.));
        println!("{}", f_powf(27., 1. / 3.));
        assert!(
            (f_powf(2f32, 3f32) - 8f32).abs() < 1e-6,
            "Invalid result {}",
            f_powf(2f32, 3f32)
        );
        assert!(
            (f_powf(0.5f32, 2f32) - 0.25f32).abs() < 1e-6,
            "Invalid result {}",
            f_powf(0.5f32, 2f32)
        );
    }

    #[test]
    fn dirty_powf_test() {
        println!("{}", dirty_powf(3., 3.));
        println!("{}", dirty_powf(27., 1. / 3.));
        assert!(
            (dirty_powf(2f32, 3f32) - 8f32).abs() < 1e-6,
            "Invalid result {}",
            dirty_powf(2f32, 3f32)
        );
        assert!(
            (dirty_powf(0.5f32, 2f32) - 0.25f32).abs() < 1e-6,
            "Invalid result {}",
            dirty_powf(0.5f32, 2f32)
        );
    }
}
