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

use crate::{exp, f_exp2, f_log2, log};

/// Power function for given value
#[inline]
pub const fn pow(d: f64, n: f64) -> f64 {
    let value = d.abs();

    let r = n * log(value);
    let c = exp(r);
    if n == 0. {
        return 1.;
    }
    if d < 0.0 {
        let y = n as i32;
        if y % 2 == 0 { c } else { -c }
    } else {
        c
    }
}

#[inline]
fn is_integer(n: f64) -> bool {
    n == n.round_ties_even()
}

/// Power function for given value using FMA
#[inline]
pub fn f_pow(x: f64, y: f64) -> f64 {
    let x_u = x.to_bits();
    let y_u = y.to_bits();

    let mut x = x;

    let mut sign: f64 = 1.;

    if x_u >= 0x7ff0000000000000 || y_u >= 0x7ff0000000000000 {
        if x.is_nan() {
            // IEEE 754-2019: pow(x,+/-0) = 1 if x is not a signaling NaN
            if y == 0.0 {
                return 1.0;
            }
            // pow(sNaN, y) = qNaN. This is implicit in IEEE 754-2019
            return x + x;
        }

        if y.is_nan() {
            // IEEE 754-2019: pow(1,y) = 1 for any y (even a quiet NaN)
            if x == 1.0 {
                return 1.0;
            }

            // pow(x, sNaN) = qNaN (see above)
            return y + y;
        }

        match x_u {
            // x = +inf
            0x7ff0000000000000 => {
                if y == 0.0 {
                    return 1.0;
                }

                if y < 0.0 {
                    return 0.0;
                }

                if y > 0.0 {
                    return f64::INFINITY;
                }
            }
            // x = -inf
            0xfff0000000000000 => {
                // y is an odd integer
                if is_integer(y) && !is_integer(y * 0.5) {
                    // y is a negative odd integer
                    return if y < 0.0 {
                        -0.0
                    } else {
                        f64::NEG_INFINITY // y is a positive odd integer
                    };
                }

                // y is a negative even integer or is negative non-integer
                if y < 0.0 {
                    return 0.0;
                }

                // y is a positive even integer or is positive non-integer
                if y > 0.0 {
                    return f64::INFINITY;
                }
            }
            _ => {}
        }

        match y_u {
            // y = +inf
            0x7ff0000000000000 => {
                if x == 0.0 {
                    return 0.0;
                }

                if x == -1.0 || x == 1.0 {
                    return 1.0;
                }

                if -1.0 < x && x < 1.0 {
                    return 0.0;
                }

                if x < -1.0 || 1.0 < x {
                    return f64::INFINITY;
                }
            }
            // y = -inf
            0xfff0000000000000 => {
                if x == 0.0 {
                    return f64::INFINITY;
                }

                if x == -1.0 || x == 1.0 {
                    return 1.0;
                }

                if -1.0 < x && x < 1.0 {
                    return f64::INFINITY;
                }

                if x < -1.0 || 1.0 < x {
                    return 0.0;
                }
            }
            _ => {}
        }
    }

    /* first deal with the case x <= 0 */
    if x <= 0.0 {
        /* pow(x,+/-0) is 1 if x is not a signaling NaN. */
        if y == 0.0 {
            return 1.0;
        }

        match x_u {
            // x = +0.0
            0 => {
                if is_integer(y) && !is_integer(y * 0.5) {
                    // y is a negative odd integer
                    if y < 0.0 {
                        return f64::INFINITY;
                    }

                    // y is a positive odd integer
                    return 0.0;
                }

                // y is positive (non-integer or a positive even integer)
                if y > 0.0 {
                    return 0.0;
                }

                // y is negative, finite and an even integer or a non-integer
                return f64::INFINITY;
            }
            // x = -0.0
            0x8000000000000000 => {
                // y is an odd integer
                if is_integer(y) && !is_integer(y * 0.5) {
                    // y is a negative odd integer
                    if y < 0.0 {
                        return f64::NEG_INFINITY;
                    }

                    // y is a positive odd integer
                    return -0.0;
                }

                // y is positive (non-integer or a positive even integer)
                if y > 0.0 {
                    return 0.0;
                }

                // y is negative, finite and an even integer or a non-integer
                return f64::INFINITY;
            }
            _ => {}
        }

        if !is_integer(y) {
            return f64::NAN;
        }

        // set sign to 1 for y even, to -1 for y odd
        let y_parity = if y.abs() >= f64::from_bits(0x4340000000000000) {
            0
        } else {
            y as i64 & 0x1
        };
        sign = if y_parity == 0 { 1.0 } else { -1.0 };

        // Set x to |x| for the rest of the computation
        x = -x;
    }

    let r = f_log2(x);
    let c = f_exp2(y * r);
    f64::copysign(c, sign)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn powf_test() {
        println!("{}", pow(3., 3.));
        println!("{}", pow(27., 1. / 3.));

        assert!(
            (pow(2f64, 3f64) - 8f64).abs() < 1e-9,
            "Invalid result {}",
            pow(2f64, 3f64)
        );
        assert!(
            (pow(0.5f64, 2f64) - 0.25f64).abs() < 1e-9,
            "Invalid result {}",
            pow(0.5f64, 2f64)
        );
    }

    #[test]
    fn f_pow_test() {
        println!("{}", f_pow(3., 3.));
        println!("{}", f_pow(27., 1. / 3.));

        assert!(
            (f_pow(2f64, 3f64) - 8f64).abs() < 1e-9,
            "Invalid result {}",
            f_pow(2f64, 3f64)
        );
        assert!(
            (f_pow(0.5f64, 2f64) - 0.25f64).abs() < 1e-9,
            "Invalid result {}",
            f_pow(0.5f64, 2f64)
        );
    }
}
