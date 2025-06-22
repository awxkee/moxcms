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

use crate::math::common::f_fmla;
use crate::math::dekker::Dekker;
use crate::math::log2::LOG_RANGE_REDUCTION;
use crate::math::logf::f_polyeval3;
use crate::math::sin::EXP_MASK;
use crate::{exp, log};

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

static LOG2_R_DD: [(u64, u64); 128] = [
    (0x0000000000000000, 0x0000000000000000),
    (0xbd319b14945cf6ba, 0x3f872c7ba2100000),
    (0xbd495539356f93dc, 0x3f9743ee86200000),
    (0x3d4abe0a48f83604, 0x3fa184b8e4c50000),
    (0x3d4635577970e040, 0x3fa77394c9d90000),
    (0xbd2401fbaaa67e3c, 0x3fad6ebd1f200000),
    (0xbd45b1799ceaeb51, 0x3fb1bb32a6008000),
    (0x3d47c407050799bf, 0x3fb4c560fe688000),
    (0x3d4da6339da288fc, 0x3fb7d60496cf8000),
    (0x3d4be4f6f22dbbad, 0x3fb960caf9ab8000),
    (0xbd2c760bc9b188c4, 0x3fbc7b528b710000),
    (0x3d3164e932b2d51c, 0x3fbf9c95dc1d0000),
    (0x3d2924ae921f7eca, 0x3fc097e38ce60000),
    (0xbd36d25a5b8a19b2, 0x3fc22dadc2ab4000),
    (0x3d4e50a1644ac794, 0x3fc3c6fb650cc000),
    (0x3d4f34baa74a7942, 0x3fc494f863b8c000),
    (0xbd18f7aac147fdc1, 0x3fc633a8bf438000),
    (0x3d4f84be19cb9578, 0x3fc7046031c78000),
    (0xbd166cccab240e90, 0x3fc8a8980abfc000),
    (0xbd03f7a55cd2af4c, 0x3fc97c1cb13c8000),
    (0x3d43458cde69308c, 0x3fcb2602497d4000),
    (0xbd3667f21fa8423f, 0x3fcbfc67a8000000),
    (0x3d0d2fe4574e09b9, 0x3fcdac22d3e44000),
    (0x3d4367bde40c5e6d, 0x3fce857d3d360000),
    (0x3d1d45da26510033, 0x3fd01d9bbcfa6000),
    (0xbd37204f55bbf90d, 0x3fd08bce0d960000),
    (0xbd4d4f1b95e0ff45, 0x3fd169c053640000),
    (0x3d3c20d74c0211bf, 0x3fd1d982c9d52000),
    (0x3d4ad89a083e072a, 0x3fd249cd2b13c000),
    (0x3d4cd0cb4492f1bc, 0x3fd32bfee370e000),
    (0xbd02101a9685c779, 0x3fd39de8e155a000),
    (0x3d49451cd394fe8d, 0x3fd4106017c3e000),
    (0x3d3661e393a16b95, 0x3fd4f6fbb2cec000),
    (0xbd3c6d8d86531d56, 0x3fd56b22e6b58000),
    (0x3d4c1c885adb21d3, 0x3fd5dfdcf1eea000),
    (0x3d23bb5921006679, 0x3fd6552b49986000),
    (0x3d41d406db502403, 0x3fd6cb0f6865c000),
    (0x3d455a63e278bad5, 0x3fd7b89f02cf2000),
    (0xbce66ae2a7ada553, 0x3fd8304d90c12000),
    (0xbd266cccab240e90, 0x3fd8a8980abfc000),
    (0xbd262404772a151d, 0x3fd921800924e000),
    (0x3d3ac9bca36fd02e, 0x3fd99b072a96c000),
    (0x3d44bc302ffa76fb, 0x3fda8ff971810000),
    (0x3d401fea1ec47c71, 0x3fdb0b67f4f46000),
    (0xbd4f20203b3186a6, 0x3fdb877c57b1c000),
    (0xbd22642415d47384, 0x3fdc043859e30000),
    (0xbcdbc76a2753b99b, 0x3fdc819dc2d46000),
    (0xbd4da93ae3a5f451, 0x3fdcffae611ae000),
    (0xbd450e785694a8c6, 0x3fdd7e6c0abc4000),
    (0x3d4c56138c894641, 0x3fddfdd89d586000),
    (0x3d45669df6a2b592, 0x3fde7df5fe538000),
    (0xbcfea92d9e0e8ac2, 0x3fdefec61b012000),
    (0x3d4a0331af2e6fea, 0x3fdf804ae8d0c000),
    (0x3cf9518ce032f41d, 0x3fe0014332be0000),
    (0xbd3b3b3864c60011, 0x3fe042bd4b9a8000),
    (0xbd2103e8f00d41c8, 0x3fe08494c66b9000),
    (0x3d465be75cc3da17, 0x3fe0c6caaf0c5000),
    (0x3d43676289cd3dd4, 0x3fe1096015dee000),
    (0xbd441dfc7d7c3321, 0x3fe14c560fe69000),
    (0x3d3e0cda8bd74461, 0x3fe18fadb6e2d000),
    (0x3d32a606046ad444, 0x3fe1d368296b5000),
    (0x3d4f9ea977a639c0, 0x3fe217868b0c3000),
    (0xbd250520a377c7ec, 0x3fe25c0a0463c000),
    (0x3d06e3cb71b554e7, 0x3fe2a0f3c3407000),
    (0xbcf4275f1035e5e8, 0x3fe2e644fac05000),
    (0xbcf4275f1035e5e8, 0x3fe2e644fac05000),
    (0xbd2979a5db68721d, 0x3fe32bfee370f000),
    (0x3d41ee969a95f529, 0x3fe37222bb707000),
    (0x3d4bb4b69336b66e, 0x3fe3b8b1c68fa000),
    (0x3d2d5e6a8a4fb059, 0x3fe3ffad4e74f000),
    (0x3d33106e404cabb7, 0x3fe44716a2c08000),
    (0x3d33106e404cabb7, 0x3fe44716a2c08000),
    (0xbd49bcaf1aa4168a, 0x3fe48eef19318000),
    (0x3d31646b761c48de, 0x3fe4d7380dcc4000),
    (0x3d42f0c0bfe9dbec, 0x3fe51ff2e3021000),
    (0x3d429904613e33c0, 0x3fe5692101d9b000),
    (0x3d31d406db502403, 0x3fe5b2c3da197000),
    (0x3d31d406db502403, 0x3fe5b2c3da197000),
    (0xbd3125d6cbcd1095, 0x3fe5fcdce2728000),
    (0xbd4bd9b32266d92c, 0x3fe6476d98ada000),
    (0x3d354243b21709ce, 0x3fe6927781d93000),
    (0x3d354243b21709ce, 0x3fe6927781d93000),
    (0xbd3ce60916e52e91, 0x3fe6ddfc2a790000),
    (0x3d4f1f5ae718f241, 0x3fe729fd26b70000),
    (0xbd46eb9612e0b4f3, 0x3fe7767c12968000),
    (0xbd46eb9612e0b4f3, 0x3fe7767c12968000),
    (0x3d4fed21f9cb2cc5, 0x3fe7c37a9227e000),
    (0x3d47f5dc57266758, 0x3fe810fa51bf6000),
    (0x3d47f5dc57266758, 0x3fe810fa51bf6000),
    (0x3d45b338360c2ae2, 0x3fe85efd062c6000),
    (0xbd496fc8f4b56502, 0x3fe8ad846cf37000),
    (0xbd496fc8f4b56502, 0x3fe8ad846cf37000),
    (0xbd3bdc81c4db3134, 0x3fe8fc924c89b000),
    (0x3d436c101ee13440, 0x3fe94c287492c000),
    (0x3d436c101ee13440, 0x3fe94c287492c000),
    (0x3d3e41fa0a62e6ae, 0x3fe99c48be206000),
    (0xbd1d97ee9124773b, 0x3fe9ecf50bf44000),
    (0xbd1d97ee9124773b, 0x3fe9ecf50bf44000),
    (0xbd13f94e00e7d6bc, 0x3fea3e2f4ac44000),
    (0xbd46879fa00b120a, 0x3fea8ff971811000),
    (0xbd46879fa00b120a, 0x3fea8ff971811000),
    (0x3d31659d8e2d7d38, 0x3feae255819f0000),
    (0x3d41e5e0ae0d3f8a, 0x3feb35458761d000),
    (0x3d41e5e0ae0d3f8a, 0x3feb35458761d000),
    (0x3d4484a15babcf88, 0x3feb88cb9a2ab000),
    (0x3d4484a15babcf88, 0x3feb88cb9a2ab000),
    (0x3d2871a7610e40bd, 0x3febdce9dcc96000),
    (0xbd42d90e5edaecee, 0x3fec31a27dd01000),
    (0xbd42d90e5edaecee, 0x3fec31a27dd01000),
    (0xbd45dd31d962d373, 0x3fec86f7b7ea5000),
    (0xbd45dd31d962d373, 0x3fec86f7b7ea5000),
    (0xbd49ad57391924a7, 0x3fecdcebd2374000),
    (0xbd33167ccc538261, 0x3fed338120a6e000),
    (0xbd33167ccc538261, 0x3fed338120a6e000),
    (0x3d2c7a4ff65ddbc9, 0x3fed8aba045b0000),
    (0x3d2c7a4ff65ddbc9, 0x3fed8aba045b0000),
    (0xbd3f9ab3cf74baba, 0x3fede298ec0bb000),
    (0xbd3f9ab3cf74baba, 0x3fede298ec0bb000),
    (0x3d452842c1c1e586, 0x3fee3b20546f5000),
    (0x3d452842c1c1e586, 0x3fee3b20546f5000),
    (0x3cf3c6764fc87b4a, 0x3fee9452c8a71000),
    (0x3cf3c6764fc87b4a, 0x3fee9452c8a71000),
    (0xbd3a0976c0a2827d, 0x3feeee32e2aed000),
    (0xbd3a0976c0a2827d, 0x3feeee32e2aed000),
    (0xbd4a45314dc4fc42, 0x3fef48c34bd1f000),
    (0xbd4a45314dc4fc42, 0x3fef48c34bd1f000),
    (0x3d3ef5d00e390a00, 0x3fefa406bd244000),
    (0x0000000000000000, 0x3ff0000000000000),
];

static EXP2_MID1: [(u64, u64); 64] = [
    (0x0000000000000000, 0x3ff0000000000000),
    (0xbc719083535b085d, 0x3ff02c9a3e778061),
    (0x3c8d73e2a475b465, 0x3ff059b0d3158574),
    (0x3c6186be4bb284ff, 0x3ff0874518759bc8),
    (0x3c98a62e4adc610b, 0x3ff0b5586cf9890f),
    (0x3c403a1727c57b53, 0x3ff0e3ec32d3d1a2),
    (0xbc96c51039449b3a, 0x3ff11301d0125b51),
    (0xbc932fbf9af1369e, 0x3ff1429aaea92de0),
    (0xbc819041b9d78a76, 0x3ff172b83c7d517b),
    (0x3c8e5b4c7b4968e4, 0x3ff1a35beb6fcb75),
    (0x3c9e016e00a2643c, 0x3ff1d4873168b9aa),
    (0x3c8dc775814a8495, 0x3ff2063b88628cd6),
    (0x3c99b07eb6c70573, 0x3ff2387a6e756238),
    (0x3c82bd339940e9d9, 0x3ff26b4565e27cdd),
    (0x3c8612e8afad1255, 0x3ff29e9df51fdee1),
    (0x3c90024754db41d5, 0x3ff2d285a6e4030b),
    (0x3c86f46ad23182e4, 0x3ff306fe0a31b715),
    (0x3c932721843659a6, 0x3ff33c08b26416ff),
    (0xbc963aeabf42eae2, 0x3ff371a7373aa9cb),
    (0xbc75e436d661f5e3, 0x3ff3a7db34e59ff7),
    (0x3c8ada0911f09ebc, 0x3ff3dea64c123422),
    (0xbc5ef3691c309278, 0x3ff4160a21f72e2a),
    (0x3c489b7a04ef80d0, 0x3ff44e086061892d),
    (0x3c73c1a3b69062f0, 0x3ff486a2b5c13cd0),
    (0x3c7d4397afec42e2, 0x3ff4bfdad5362a27),
    (0xbc94b309d25957e3, 0x3ff4f9b2769d2ca7),
    (0xbc807abe1db13cad, 0x3ff5342b569d4f82),
    (0x3c99bb2c011d93ad, 0x3ff56f4736b527da),
    (0x3c96324c054647ad, 0x3ff5ab07dd485429),
    (0x3c9ba6f93080e65e, 0x3ff5e76f15ad2148),
    (0xbc9383c17e40b497, 0x3ff6247eb03a5585),
    (0xbc9bb60987591c34, 0x3ff6623882552225),
    (0xbc9bdd3413b26456, 0x3ff6a09e667f3bcd),
    (0xbc6bbe3a683c88ab, 0x3ff6dfb23c651a2f),
    (0xbc816e4786887a99, 0x3ff71f75e8ec5f74),
    (0xbc90245957316dd3, 0x3ff75feb564267c9),
    (0xbc841577ee04992f, 0x3ff7a11473eb0187),
    (0x3c705d02ba15797e, 0x3ff7e2f336cf4e62),
    (0xbc9d4c1dd41532d8, 0x3ff82589994cce13),
    (0xbc9fc6f89bd4f6ba, 0x3ff868d99b4492ed),
    (0x3c96e9f156864b27, 0x3ff8ace5422aa0db),
    (0x3c85cc13a2e3976c, 0x3ff8f1ae99157736),
    (0xbc675fc781b57ebc, 0x3ff93737b0cdc5e5),
    (0xbc9d185b7c1b85d1, 0x3ff97d829fde4e50),
    (0x3c7c7c46b071f2be, 0x3ff9c49182a3f090),
    (0xbc9359495d1cd533, 0x3ffa0c667b5de565),
    (0xbc9d2f6edb8d41e1, 0x3ffa5503b23e255d),
    (0x3c90fac90ef7fd31, 0x3ffa9e6b5579fdbf),
    (0x3c97a1cd345dcc81, 0x3ffae89f995ad3ad),
    (0xbc62805e3084d708, 0x3ffb33a2b84f15fb),
    (0xbc75584f7e54ac3b, 0x3ffb7f76f2fb5e47),
    (0x3c823dd07a2d9e84, 0x3ffbcc1e904bc1d2),
    (0x3c811065895048dd, 0x3ffc199bdd85529c),
    (0x3c92884dff483cad, 0x3ffc67f12e57d14b),
    (0x3c7503cbd1e949db, 0x3ffcb720dcef9069),
    (0xbc9cbc3743797a9c, 0x3ffd072d4a07897c),
    (0x3c82ed02d75b3707, 0x3ffd5818dcfba487),
    (0x3c9c2300696db532, 0x3ffda9e603db3285),
    (0xbc91a5cd4f184b5c, 0x3ffdfc97337b9b5f),
    (0x3c839e8980a9cc8f, 0x3ffe502ee78b3ff6),
    (0xbc9e9c23179c2893, 0x3ffea4afa2a490da),
    (0x3c9dc7f486a4b6b0, 0x3ffefa1bee615a27),
    (0x3c99d3e12dd8a18b, 0x3fff50765b6e4540),
    (0x3c874853f3a5931e, 0x3fffa7c1819e90d8),
];

#[inline]
fn is_integer(n: f64) -> bool {
    n == n.round_ties_even()
}

#[inline]
fn is_odd_integer(x: f64) -> bool {
    let x_u = x.to_bits();
    let x_e = x_u >> 52;
    let lsb = (x_u | EXP_MASK).trailing_zeros();
    const E_BIAS: u64 = (1u64 << (11 - 1u64)) - 1u64;

    const UNIT_EXPONENT: u64 = E_BIAS + 52;
    x_e + lsb as u64 == UNIT_EXPONENT
}

/// Power function for given value using FMA
///
/// max found ULP 0.5216031589654122
#[inline]
pub fn f_pow(x: f64, y: f64) -> f64 {
    let mut y = y;
    let x_sign = x.is_sign_negative();
    let y_sign = y.is_sign_negative();

    let x_abs = x.to_bits() & 0x7fff_ffff_ffff_ffff;
    let y_abs = y.to_bits() & 0x7fff_ffff_ffff_ffff;

    const MANTISSA_MASK: u64 = (1u64 << 52) - 1;

    let mut x_mant = x.to_bits() & MANTISSA_MASK;
    let y_mant = y.to_bits() & MANTISSA_MASK;
    let x_u = x.to_bits();
    let x_a = x_abs;
    let y_a = y_abs;

    let mut x = x;

    let mut sign: u64 = 0;

    // exponent
    let mut e_x = ((x.to_bits() as i64 & EXP_MASK as i64) >> 52).wrapping_sub(1023) as f64;

    // If x or y is signaling NaN
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }

    // The double precision number that is closest to 1 is (1 - 2^-53), which has
    //   log2(1 - 2^-53) ~ -1.715...p-53.
    // So if |y| > |1075 / log2(1 - 2^-53)|, and x is finite:
    //   |y * log2(x)| = 0 or > 1075.
    // Hence, x^y will either overflow or underflow if x is not zero.
    if y_mant == 0
        || y_a > 0x43d7_4910_d52d_3052
        || x_u == 1f64.to_bits()
        || x_u >= f64::INFINITY.to_bits()
        || x_u < f64::MIN.to_bits()
    {
        // Exceptional exponents.
        if y == 0.0 {
            return 1.0;
        }

        match y_a {
            0x3fe0_0000_0000_0000 => {
                // TODO: speed up x^(-1/2) with rsqrt(x) when available.
                if x == 0.0 || x_u == f64::NEG_INFINITY.to_bits() {
                    // pow(-0, 1/2) = +0
                    // pow(-inf, 1/2) = +inf
                    // Make sure it works correctly for FTZ/DAZ.
                    return if y_sign { 1.0 / (x * x) } else { x * x };
                }
                return if y_sign { 1.0 / x.sqrt() } else { x.sqrt() };
            }
            0x3ff0_0000_0000_0000 => {
                return if y_sign { 1.0 / x } else { x };
            }
            0x4000_0000_0000_0000 => {
                return if y_sign { 1.0 / (x * x) } else { x * x };
            }
            _ => {}
        }

        // |y| > |1075 / log2(1 - 2^-53)|.
        if y_a > 0x43d7_4910_d52d_3052 {
            if y_a >= 0x7ff0_0000_0000_0000 {
                // y is inf or nan
                if y_mant != 0 {
                    // y is NaN
                    // pow(1, NaN) = 1
                    // pow(x, NaN) = NaN
                    return if x_u == 1f64.to_bits() { 1.0 } else { y };
                }

                // Now y is +-Inf
                if f64::from_bits(x_abs).is_nan() {
                    // pow(NaN, +-Inf) = NaN
                    return x;
                }

                if x_a == 0x3ff0_0000_0000_0000 {
                    // pow(+-1, +-Inf) = 1.0
                    return 1.0;
                }

                if x == 0.0 && y_sign {
                    // pow(+-0, -Inf) = +inf and raise FE_DIVBYZERO
                    return f64::INFINITY;
                }
                // pow (|x| < 1, -inf) = +inf
                // pow (|x| < 1, +inf) = 0.0
                // pow (|x| > 1, -inf) = 0.0
                // pow (|x| > 1, +inf) = +inf
                return if (x_a < 1f64.to_bits()) == y_sign {
                    f64::INFINITY
                } else {
                    0.0
                };
            }
            // x^y will overflow / underflow in double precision.  Set y to a
            // large enough exponent but not too large, so that the computations
            // won't overflow in double precision.
            y = if y_sign {
                f64::from_bits(0xc630000000000000)
            } else {
                f64::from_bits(0x4630000000000000)
            };
        }

        // y is finite and non-zero.

        if x_u == 1f64.to_bits() {
            // pow(1, y) = 1
            return 1.0;
        }

        // TODO: Speed things up with pow(2, y) = exp2(y) and pow(10, y) = exp10(y).

        if x == 0.0 {
            let out_is_neg = x_sign && is_odd_integer(y);
            if y_sign {
                // pow(0, negative number) = inf
                return if out_is_neg {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                };
            }
            // pow(0, positive number) = 0
            return if out_is_neg { -0.0 } else { 0.0 };
        }

        if x_a == f64::INFINITY.to_bits() {
            let out_is_neg = x_sign && is_odd_integer(y);
            if y_sign {
                return if out_is_neg { -0.0 } else { 0.0 };
            }
            return if out_is_neg {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
        }

        if x_a > f64::INFINITY.to_bits() {
            // x is NaN.
            // pow (aNaN, 0) is already taken care above.
            return x;
        }

        // Normalize denormal inputs.
        if x_a < f64::MIN_POSITIVE.to_bits() {
            e_x -= 64.0;
            x_mant = (x * f64::from_bits(0x43f0000000000000)).to_bits() & MANTISSA_MASK;
        }

        // x is finite and negative, and y is a finite integer.
        if x_sign {
            if is_integer(y) {
                x = -x;
                if is_odd_integer(y) {
                    // sign = -1.0;
                    sign = 0x8000_0000_0000_0000;
                }
            } else {
                // pow( negative, non-integer ) = NaN
                return f64::NAN;
            }
        }

        // y is finite and non-zero.

        if x_u == 1f64.to_bits() {
            // pow(1, y) = 1
            return 1.0;
        }

        if x == 0.0 {
            let out_is_neg = x_sign && is_odd_integer(y);
            if y_sign {
                // pow(0, negative number) = inf
                return if out_is_neg {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                };
            }
            // pow(0, positive number) = 0
            return if out_is_neg { -0.0 } else { 0.0 };
        }

        if x_a == f64::INFINITY.to_bits() {
            let out_is_neg = x_sign && is_odd_integer(y);
            if y_sign {
                return if out_is_neg { -0.0 } else { 0.0 };
            }
            return if out_is_neg {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
        }

        if x_a > f64::INFINITY.to_bits() {
            // x is NaN.
            // pow (aNaN, 0) is already taken care above.
            return x;
        }

        // Normalize denormal inputs.
        if x_a < f64::MIN_POSITIVE.to_bits() {
            e_x -= 64.0;
            x_mant = (x * f64::from_bits(0x43f0000000000000)).to_bits() >> 52;
        }

        // x is finite and negative, and y is a finite integer.
        if x_sign {
            if is_integer(y) {
                if is_odd_integer(y) {
                    // sign = -1.0;
                    sign = 0x8000_0000_0000_0000;
                }
            } else {
                // pow( negative, non-integer ) = NaN
                return f64::NAN;
            }
        }
    }

    ///////// END - Check exceptional cases //////////////////////////////////////

    // x^y = 2^( y * log2(x) )
    //     = 2^( y * ( e_x + log2(m_x) ) )
    // First we compute log2(x) = e_x + log2(m_x)

    // Extract exponent field of x.

    // Use the highest 7 fractional bits of m_x as the index for look up tables.
    let idx_x = x_mant.wrapping_shr(52 - 7);
    // Add the hidden bit to the mantissa.
    // 1 <= m_x < 2
    let m_x = x_mant | 0x3ff0_0000_0000_0000;

    // Reduced argument for log2(m_x):
    //   dx = r * m_x - 1.
    // The computation is exact, and -2^-8 <= dx < 2^-7.
    // Then m_x = (1 + dx) / r, and
    //   log2(m_x) = log2( (1 + dx) / r )
    //             = log2(1 + dx) - log2(r).

    // In order for the overall computations x^y = 2^(y * log2(x)) to have the
    // relative errors < 2^-52 (1ULP), we will need to evaluate the exponent part
    // y * log2(x) with absolute errors < 2^-52 (or better, 2^-53).  Since the
    // whole exponent range for double precision is bounded by
    // |y * log2(x)| < 1076 ~ 2^10, we need to evaluate log2(x) with absolute
    // errors < 2^-53 * 2^-10 = 2^-63.

    // With that requirement, we use the following degree-6 polynomial
    // approximation:
    //   P(dx) ~ log2(1 + dx) / dx
    // Generated by Sollya with:
    // > P = fpminimax(log2(1 + x)/x, 6, [|D...|], [-2^-8, 2^-7]); P;
    // > dirtyinfnorm(log2(1 + x) - x*P, [-2^-8, 2^-7]);
    //   0x1.d03cc...p-66

    const COEFFS: [u64; 7] = [
        0x3ff71547652b82fe,
        0xbfe71547652b82e7,
        0x3fdec709dc3b1fd5,
        0xbfd7154766124215,
        0x3fd2776bd90259d8,
        0xbfcec586c6f3d311,
        0x3fc9c4775eccf524,
    ];

    // Error: ulp(dx^2) <= (2^-7)^2 * 2^-52 = 2^-66
    // Extra errors from various computations and rounding directions, the overall
    // errors we can be bounded by 2^-65.

    let dx: f64;
    let dx_c0: Dekker;

    #[cfg(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    ))]
    {
        dx = f_fmla(
            f64::from_bits(LOG_RANGE_REDUCTION[idx_x as usize]),
            f64::from_bits(m_x),
            -1.0,
        ); // Exact
        dx_c0 = Dekker::from_exact_mult(f64::from_bits(COEFFS[0]), dx);
    }

    #[cfg(not(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        use crate::math::log2::LOG_CD;
        let c = f64::from_bits(m_x & 0x3fff_e000_0000_0000);
        dx = f_fmla(
            f64::from_bits(LOG_RANGE_REDUCTION[idx_x as usize]),
            f64::from_bits(m_x) - c,
            f64::from_bits(LOG_CD[idx_x as usize]),
        ); // Exact
        dx_c0 = Dekker::from_exact_mult(dx, f64::from_bits(COEFFS[0])); // Exact
    }

    let dx2 = dx * dx;
    let c0 = f_fmla(dx, f64::from_bits(COEFFS[2]), f64::from_bits(COEFFS[1]));
    let c1 = f_fmla(dx, f64::from_bits(COEFFS[4]), f64::from_bits(COEFFS[3]));
    let c2 = f_fmla(dx, f64::from_bits(COEFFS[6]), f64::from_bits(COEFFS[5]));

    let p = f_polyeval3(dx2, c0, c1, c2);

    // s = e_x - log2(r) + dx * P(dx)
    // Absolute error bound:
    //   |log2(x) - log2_x.hi - log2_x.lo| < 2^-65.

    // Notice that e_x - log2(r).hi is exact, so we perform an exact sum of
    // e_x - log2(r).hi and the high part of the product dx * c0:
    //   log2_x_hi.hi + log2_x_hi.lo = e_x - log2(r).hi + (dx * c0).hi
    let log_r_dd = LOG2_R_DD[idx_x as usize];
    let log2_x_hi = Dekker::from_exact_add(e_x + f64::from_bits(log_r_dd.1), dx_c0.hi);
    // The low part is dx^2 * p + low part of (dx * c0) + low part of -log2(r).
    let log2_x_lo = f_fmla(dx2, p, dx_c0.lo + f64::from_bits(log_r_dd.0));
    // Perform accurate sums.
    let mut log2_x = Dekker::from_exact_add(log2_x_hi.hi, log2_x_lo);
    log2_x.lo += log2_x_hi.lo;

    // To compute 2^(y * log2(x)), we break the exponent into 3 parts:
    //   y * log(2) = hi + mid + lo, where
    //   hi is an integer
    //   mid * 2^6 is an integer
    //   |lo| <= 2^-7
    // Then:
    //   x^y = 2^(y * log2(x)) = 2^hi * 2^mid * 2^lo,
    // In which 2^mid is obtained from a look-up table of size 2^6 = 64 elements,
    // and 2^lo ~ 1 + lo * P(lo).
    // Thus, we have:
    //   hi + mid = 2^-6 * round( 2^6 * y * log2(x) )
    // If we restrict the output such that |hi| < 150, (hi + mid) uses (8 + 6)
    // bits, hence, if we use double precision to perform
    //   round( 2^6 * y * log2(x))
    // the lo part is bounded by 2^-7 + 2^(-(52 - 14)) = 2^-7 + 2^-38

    // In the following computations:
    //   y6  = 2^6 * y
    //   hm  = 2^6 * (hi + mid) = round(2^6 * y * log2(x)) ~ round(y6 * s)
    //   lo6 = 2^6 * lo = 2^6 * (y - (hi + mid)) = y6 * log2(x) - hm.
    let y6 = y * f64::from_bits(0x4050000000000000); // Exact.

    let mut y6_log2_x = Dekker::from_exact_mult(y6, log2_x.hi);
    y6_log2_x.lo = f_fmla(y6, log2_x.lo, y6_log2_x.lo);

    // Check overflow/underflow.
    let mut scale = 1.0;

    // |2^(hi + mid) - exp2_hi_mid| <= ulp(exp2_hi_mid) / 2
    // Clamp the exponent part into smaller range that fits double precision.
    // For those exponents that are out of range, the final conversion will round
    // them correctly to inf/max float or 0/min float accordingly.
    const UPPER_EXP_BOUND: f64 = 512.0 * f64::from_bits(0x4050000000000000);

    if y6_log2_x.hi.abs() >= UPPER_EXP_BOUND {
        if y6_log2_x.hi.is_sign_positive() {
            scale = f64::from_bits(0x5ff0000000000000);
            y6_log2_x.hi -= 512.0 * 64.0;
            if y6_log2_x.hi > 513.0 * 64.0 {
                y6_log2_x.hi = 513.0 * 64.0;
            }
        } else {
            scale = f64::from_bits(0x1ff0000000000000);
            y6_log2_x.hi += 512.0 * 64.0;
            if y6_log2_x.hi < (-1076.0 + 512.0) * 64.0 {
                y6_log2_x.hi = -564.0 * 64.0;
            }
        }
    }

    let hm = y6_log2_x.hi.round();

    // lo6 = 2^6 * lo.
    let lo6_hi = y6_log2_x.hi - hm;
    let lo6 = lo6_hi + y6_log2_x.lo;

    let hm_i: i64 = hm as i64;
    let idx_y = (hm_i as u64) & 0x3f;

    // 2^hi
    let exp2_hi_i = ((hm_i >> 6) as u64).wrapping_shl(52) as i64;

    let exp2_mid = EXP2_MID1[idx_y as usize];

    // 2^mid
    let exp2_mid_hi_i: i64 = exp2_mid.1 as i64;
    let exp2_mid_lo_i: i64 = exp2_mid.0 as i64;
    // (-1)^sign * 2^hi * 2^mid
    // Error <= 2^hi * 2^-53
    let exp2_hm_hi_i = (exp2_hi_i.wrapping_add(exp2_mid_hi_i) as u64).wrapping_add(sign);
    // The low part could be 0.
    let exp2_hm_lo_i = if idx_y != 0 {
        (exp2_hi_i.wrapping_add(exp2_mid_lo_i) as u64).wrapping_add(sign)
    } else {
        sign
    };
    let exp2_hm_hi = f64::from_bits(exp2_hm_hi_i);
    let exp2_hm_lo = f64::from_bits(exp2_hm_lo_i);

    // Degree-5 polynomial approximation P(lo6) ~ 2^(lo6 / 2^6) = 2^(lo).
    // Generated by Sollya with:
    // > P = fpminimax(2^(x/64), 5, [|1, D...|], [-2^-1, 2^-1]);
    // > dirtyinfnorm(2^(x/64) - P, [-0.5, 0.5]);
    // 0x1.a2b77e618f5c4c176fd11b7659016cde5de83cb72p-60
    const EXP2_COEFFS: [u64; 6] = [
        0x3ff0000000000000,
        0x3f862e42fefa39ef,
        0x3f0ebfbdff82a23a,
        0x3e8c6b08d7076268,
        0x3e03b2ad33f8b48b,
        0x3d75d870c4d84445,
    ];

    let lo6_sqr = lo6 * lo6;

    let d0 = f_fmla(
        lo6,
        f64::from_bits(EXP2_COEFFS[2]),
        f64::from_bits(EXP2_COEFFS[1]),
    );
    let d1 = f_fmla(
        lo6,
        f64::from_bits(EXP2_COEFFS[4]),
        f64::from_bits(EXP2_COEFFS[3]),
    );
    let pp = f_polyeval3(lo6_sqr, d0, d1, f64::from_bits(EXP2_COEFFS[5]));

    let mut r = f_fmla(exp2_hm_hi * lo6, pp, exp2_hm_lo);
    r += exp2_hm_hi;

    r * scale
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
