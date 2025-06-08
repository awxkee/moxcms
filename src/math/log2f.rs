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

static IX: [u64; 129] = [
    0x3ff0000000000000,
    0x3fefc07f01fc0000,
    0x3fef81f81f820000,
    0x3fef44659e4a0000,
    0x3fef07c1f07c0000,
    0x3feecc07b3020000,
    0x3fee9131abf00000,
    0x3fee573ac9020000,
    0x3fee1e1e1e1e0000,
    0x3fede5d6e3f80000,
    0x3fedae6076ba0000,
    0x3fed77b654b80000,
    0x3fed41d41d420000,
    0x3fed0cb58f6e0000,
    0x3fecd85689040000,
    0x3feca4b3055e0000,
    0x3fec71c71c720000,
    0x3fec3f8f01c40000,
    0x3fec0e0703820000,
    0x3febdd2b89940000,
    0x3febacf914c20000,
    0x3feb7d6c3dda0000,
    0x3feb4e81b4e80000,
    0x3feb2036406c0000,
    0x3feaf286bca20000,
    0x3feac5701ac60000,
    0x3fea98ef606a0000,
    0x3fea6d01a6d00000,
    0x3fea41a41a420000,
    0x3fea16d3f97a0000,
    0x3fe9ec8e95100000,
    0x3fe9c2d14ee40000,
    0x3fe99999999a0000,
    0x3fe970e4f80c0000,
    0x3fe948b0fcd60000,
    0x3fe920fb49d00000,
    0x3fe8f9c18f9c0000,
    0x3fe8d3018d300000,
    0x3fe8acb90f6c0000,
    0x3fe886e5f0ac0000,
    0x3fe8618618620000,
    0x3fe83c977ab20000,
    0x3fe8181818180000,
    0x3fe7f405fd020000,
    0x3fe7d05f417e0000,
    0x3fe7ad2208e00000,
    0x3fe78a4c81780000,
    0x3fe767dce4340000,
    0x3fe745d1745e0000,
    0x3fe724287f460000,
    0x3fe702e05c0c0000,
    0x3fe6e1f76b440000,
    0x3fe6c16c16c20000,
    0x3fe6a13cd1540000,
    0x3fe6816816820000,
    0x3fe661ec6a520000,
    0x3fe642c8590c0000,
    0x3fe623fa77020000,
    0x3fe6058160580000,
    0x3fe5e75bb8d00000,
    0x3fe5c9882b940000,
    0x3fe5ac056b020000,
    0x3fe58ed230820000,
    0x3fe571ed3c500000,
    0x3fe5555555560000,
    0x3fe5390948f40000,
    0x3fe51d07eae20000,
    0x3fe5015015020000,
    0x3fe4e5e0a7300000,
    0x3fe4cab887260000,
    0x3fe4afd6a0520000,
    0x3fe49539e3b20000,
    0x3fe47ae147ae0000,
    0x3fe460cbc7f60000,
    0x3fe446f865620000,
    0x3fe42d6625d60000,
    0x3fe4141414140000,
    0x3fe3fb013fb00000,
    0x3fe3e22cbce40000,
    0x3fe3c995a47c0000,
    0x3fe3b13b13b20000,
    0x3fe3991c2c180000,
    0x3fe3813813820000,
    0x3fe3698df3de0000,
    0x3fe3521cfb2c0000,
    0x3fe33ae45b580000,
    0x3fe323e34a2c0000,
    0x3fe30d1901300000,
    0x3fe2f684bda20000,
    0x3fe2e025c04c0000,
    0x3fe2c9fb4d820000,
    0x3fe2b404ad020000,
    0x3fe29e4129e40000,
    0x3fe288b012880000,
    0x3fe27350b8820000,
    0x3fe25e2270800000,
    0x3fe24924924a0000,
    0x3fe23456789a0000,
    0x3fe21fb781220000,
    0x3fe20b470c680000,
    0x3fe1f7047dc20000,
    0x3fe1e2ef3b400000,
    0x3fe1cf06ada20000,
    0x3fe1bb4a40460000,
    0x3fe1a7b9611a0000,
    0x3fe19453808c0000,
    0x3fe1811811820000,
    0x3fe16e0689420000,
    0x3fe15b1e5f760000,
    0x3fe1485f0e0a0000,
    0x3fe135c811360000,
    0x3fe12358e75e0000,
    0x3fe1111111120000,
    0x3fe0fef010fe0000,
    0x3fe0ecf56be60000,
    0x3fe0db20a8900000,
    0x3fe0c9714fbc0000,
    0x3fe0b7e6ec260000,
    0x3fe0a6810a680000,
    0x3fe0953f39020000,
    0x3fe0842108420000,
    0x3fe073260a480000,
    0x3fe0624dd2f20000,
    0x3fe05197f7d80000,
    0x3fe0410410420000,
    0x3fe03091b5200000,
    0x3fe0204081020000,
    0x3fe0101010100000,
    0x3fe0000000000000,
];
static LIX: [u64; 129] = [
    0x0000000000000000,
    0xbf7fe02a6b146789,
    0xbf8fc0a8b0fa03e4,
    0xbf97b91b07de311b,
    0xbf9f829b0e7c3300,
    0xbfa39e87b9fd7d60,
    0xbfa77458f63edcfc,
    0xbfab42dd7117b1bf,
    0xbfaf0a30c01362a6,
    0xbfb16536eea7fae1,
    0xbfb341d7961791d1,
    0xbfb51b073f07983f,
    0xbfb6f0d28ae3eb4c,
    0xbfb8c345d6383b21,
    0xbfba926d3a475563,
    0xbfbc5e548f63a743,
    0xbfbe27076e28f2e6,
    0xbfbfec9131dbaabb,
    0xbfc0d77e7ccf6e59,
    0xbfc1b72ad52f87a0,
    0xbfc29552f81eb523,
    0xbfc371fc201f7f74,
    0xbfc44d2b6ccbfd1e,
    0xbfc526e5e3a41438,
    0xbfc5ff3070a613d4,
    0xbfc6d60fe717221d,
    0xbfc7ab890212b909,
    0xbfc87fa065214911,
    0xbfc9525a9cf296b4,
    0xbfca23bc1fe42563,
    0xbfcaf3c94e81bff3,
    0xbfcbc2867430acd6,
    0xbfcc8ff7c7989a22,
    0xbfcd5c216b535b91,
    0xbfce27076e2f92e6,
    0xbfcef0adcbe0d936,
    0xbfcfb9186d5ebe2b,
    0xbfd0402594b51041,
    0xbfd0a324e27370e3,
    0xbfd1058bf9ad7ad5,
    0xbfd1675cabaa660e,
    0xbfd1c898c16b91fb,
    0xbfd22941fbcfb966,
    0xbfd2895a13dd2ea3,
    0xbfd2e8e2bade7d31,
    0xbfd347dd9a9afd55,
    0xbfd3a64c556b05ea,
    0xbfd40430868877e4,
    0xbfd4618bc219dec2,
    0xbfd4be5f9579e0a1,
    0xbfd51aad872c982d,
    0xbfd5767717432a6c,
    0xbfd5d1bdbf5669ca,
    0xbfd62c82f2b83795,
    0x3fd5d5bddf5b0f30,
    0x3fd57bf753cb49fb,
    0x3fd522ae073b23d8,
    0x3fd4c9e09e18f43c,
    0x3fd4718dc271841b,
    0x3fd419b423d5a8c7,
    0x3fd3c2527735f184,
    0x3fd36b6776bff917,
    0x3fd314f1e1d54ce4,
    0x3fd2bef07cdb5354,
    0x3fd269621136db92,
    0x3fd214456d0e88d4,
    0x3fd1bf9963577b95,
    0x3fd16b5ccbaf1373,
    0x3fd1178e822ae47c,
    0x3fd0c42d67625ae3,
    0x3fd07138604b0862,
    0x3fd01eae56243e91,
    0x3fcf991c6cb33379,
    0x3fcef5ade4de2fe6,
    0x3fce530effe1b012,
    0x3fcdb13db0da1940,
    0x3fcd1037f264de7b,
    0x3fcc6ffbc6ef8f71,
    0x3fcbd087383798ad,
    0x3fcb31d8575dee3d,
    0x3fca93ed3c8fd9e3,
    0x3fc9f6c407055664,
    0x3fc95a5adcfc217f,
    0x3fc8beafeb38ce8c,
    0x3fc823c1655523c2,
    0x3fc7898d85460c73,
    0x3fc6f0128b7baabc,
    0x3fc6574ebe86933a,
    0x3fc5bf406b59bdb2,
    0x3fc527e5e4a5158d,
    0x3fc4913d83395561,
    0x3fc3fb45a59ed8cc,
    0x3fc365fcb0151016,
    0x3fc2d1610c81c13a,
    0x3fc23d712a4fa202,
    0x3fc1aa2b7e1ff72a,
    0x3fc1178e822de47c,
    0x3fc08598b5990a07,
    0x3fbfe89139dc1566,
    0x3fbec739830d9120,
    0x3fbda7276390c6a2,
    0x3fbc885801c04b23,
    0x3fbb6ac88da61b1c,
    0x3fba4e7640a45c38,
    0x3fb9335e5d524989,
    0x3fb8197e2f37a3f0,
    0x3fb700d30af800e1,
    0x3fb5e95a4d90f1cb,
    0x3fb4d3115d2cfeac,
    0x3fb3bdf5a7c60e64,
    0x3fb2aa04a44a57a5,
    0x3fb1973bd1527567,
    0x3fb08598b5ac3a07,
    0x3faeea31bfea787c,
    0x3faccb73cdcb32cc,
    0x3faaaef2d11110fc,
    0x3fa894aa1485b343,
    0x3fa67c94f2e07b58,
    0x3fa466aed42be3ea,
    0x3fa252f32faad83f,
    0x3fa0415d89e54444,
    0x3f9c63d2ec16aaf2,
    0x3f98492528ddcabf,
    0x3f9432a925ca0cc1,
    0x3f90205658d15847,
    0x3f882448a3d8a2aa,
    0x3f8010157586de71,
    0x3f70080559488b35,
    0x0000000000000000,
];

/// Natural logarithm using FMA
///
/// Max found ULP 0.4999996
#[inline]
pub fn f_log2f(x: f32) -> f32 {
    let t = x.to_bits();
    let ux = t;
    let mut m = (ux & 0x007fffff) as u64;
    m = m.wrapping_shl(52 - 23);
    let mut e: i32 = (ux >> 23).wrapping_sub(0x7f) as i32;
    if ux < 1u32 << 23 || ux >= 0xffu32 << 23 {
        if ux == 0 || ux == (1u32 << 31) {
            // x = +/-0
            return -1.0 / 0.0; // should raise 'Divide by zero' exception.
        }
        let inf_or_nan = ((ux >> 23) & 0xff) == 0xff;
        let nan = inf_or_nan && (ux << 9) != 0;
        if ux >> 31 != 0 && !nan {
            // x < 0
            return f32::NAN; // should raise 'Invalid operation' exception.
        }
        if inf_or_nan {
            return x + x;
        }
        // subnormal
        let nz = m.leading_zeros() as i32;
        m = m.wrapping_shl((nz - 11) as u32);
        m &= 0x000fffffffffffffu64;
        e = e.wrapping_sub(nz - 12);
    }
    if m == 0 {
        return e as f32;
    }
    let j = (m.wrapping_add(1u64 << (52 - 8))) >> (52 - 7);
    let k = if j > 53 { 1 } else { 0 };
    e += k;
    let xd = m | (0x3ffu64 << 52);
    let z = f_fmla(f64::from_bits(xd), f64::from_bits(IX[j as usize]), -1.0); // z is exact
    const C: [u64; 6] = [
        0x3ff0000000000000,
        0xbfe0000000000000,
        0x3fd55555555030bc,
        0xbfcffffffff2b4e5,
        0x3fc999b5076a42f2,
        0xbfc55570c45a647d,
    ];
    let z2 = z * z;
    let mut c0 = f_fmla(z, f64::from_bits(C[1]), f64::from_bits(C[0]));
    let c2 = f_fmla(z, f64::from_bits(C[3]), f64::from_bits(C[2]));
    let c4 = f_fmla(z, f64::from_bits(C[5]), f64::from_bits(C[4]));
    c0 = f_fmla(z2, f_fmla(z2, c4, c2), c0);
    const I_LN2: f64 = f64::from_bits(0x3ff71547652b82fe);

    let q0 = f_fmla(f64::from_bits(LIX[j as usize]), -I_LN2, e as f64);
    f_fmla(z * I_LN2, c0, q0) as f32
}

/// Natural logarithm using FMA
///
/// Max ULP 0.5248262
#[inline(always)]
#[allow(dead_code)]
pub(crate) fn f_log2fx(d: f32) -> f64 {
    let mut ix = d.to_bits();
    /* reduce x into [sqrt(2)/2, sqrt(2)] */
    ix = ix.wrapping_add(0x3f800000 - 0x3f3504f3);
    let n = (ix >> 23) as i32 - 0x7f;
    ix = (ix & 0x007fffff).wrapping_add(0x3f3504f3);
    let a = f32::from_bits(ix) as f64;

    let x = (a - 1.) / (a + 1.);

    let x2 = x * x;
    #[cfg(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    ))]
    {
        let mut u = 0.3205986261348816382e+0;
        u = f_fmla(u, x2, 0.4121985850084821691e+0);
        u = f_fmla(u, x2, 0.5770780163490337802e+0);
        u = f_fmla(u, x2, 0.9617966939259845749e+0);
        f_fmla(x2 * x, u, f_fmla(x, 0.2885390081777926802e+1, n as f64))
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
        let u = poly4!(
            x2,
            rx2,
            0.3205986261348816382e+0,
            0.4121985850084821691e+0,
            0.5770780163490337802e+0,
            0.9617966939259845749e+0
        );
        f_fmla(x2 * x, u, f_fmla(x, 0.2885390081777926802e+1, n as f64))
    }
}

/// Natural logarithm using FMA
#[inline(always)]
#[allow(dead_code)]
pub(crate) fn dirty_log2f(d: f32) -> f32 {
    let mut ix = d.to_bits();
    /* reduce x into [sqrt(2)/2, sqrt(2)] */
    ix = ix.wrapping_add(0x3f800000 - 0x3f3504f3);
    let n = (ix >> 23) as i32 - 0x7f;
    ix = (ix & 0x007fffff).wrapping_add(0x3f3504f3);
    let a = f32::from_bits(ix);

    let x = (a - 1.) / (a + 1.);

    let x2 = x * x;
    #[cfg(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    ))]
    {
        let mut u = 0.4121985850084821691e+0;
        u = f_fmlaf(u, x2, 0.5770780163490337802e+0);
        u = f_fmlaf(u, x2, 0.9617966939259845749e+0);
        f_fmlaf(x2 * x, u, f_fmlaf(x, 0.2885390081777926802e+1, n as f32))
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
        let u = poly3!(
            x2,
            rx2,
            0.4121985850084821691e+0,
            0.5770780163490337802e+0,
            0.9617966939259845749e+0
        );
        f_fmlaf(x2 * x, u, f_fmlaf(x, 0.2885390081777926802e+1, n as f32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log2f() {
        assert!((f_log2f(0.35f32) - 0.35f32.log2()).abs() < 1e-5);
        assert!((f_log2f(0.9f32) - 0.9f32.log2()).abs() < 1e-5);
    }

    #[test]
    fn test_dirty_log2f() {
        assert!((dirty_log2f(0.35f32) - 0.35f32.log2()).abs() < 1e-5);
        assert!((dirty_log2f(0.9f32) - 0.9f32.log2()).abs() < 1e-5);
    }
}
