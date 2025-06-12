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
use crate::math::dekker::Dekker;

pub(crate) static LOG_RANGE_REDUCTION: [u64; 128] = [
    0x3ff0000000000000,
    0x3fefc00000000000,
    0x3fef800000000000,
    0x3fef400000000000,
    0x3fef000000000000,
    0x3feec00000000000,
    0x3fee800000000000,
    0x3fee400000000000,
    0x3fee000000000000,
    0x3fede00000000000,
    0x3feda00000000000,
    0x3fed600000000000,
    0x3fed400000000000,
    0x3fed000000000000,
    0x3fecc00000000000,
    0x3feca00000000000,
    0x3fec600000000000,
    0x3fec400000000000,
    0x3fec000000000000,
    0x3febe00000000000,
    0x3feba00000000000,
    0x3feb800000000000,
    0x3feb400000000000,
    0x3feb200000000000,
    0x3feae00000000000,
    0x3feac00000000000,
    0x3fea800000000000,
    0x3fea600000000000,
    0x3fea400000000000,
    0x3fea000000000000,
    0x3fe9e00000000000,
    0x3fe9c00000000000,
    0x3fe9800000000000,
    0x3fe9600000000000,
    0x3fe9400000000000,
    0x3fe9200000000000,
    0x3fe9000000000000,
    0x3fe8c00000000000,
    0x3fe8a00000000000,
    0x3fe8800000000000,
    0x3fe8600000000000,
    0x3fe8400000000000,
    0x3fe8000000000000,
    0x3fe7e00000000000,
    0x3fe7c00000000000,
    0x3fe7a00000000000,
    0x3fe7800000000000,
    0x3fe7600000000000,
    0x3fe7400000000000,
    0x3fe7200000000000,
    0x3fe7000000000000,
    0x3fe6e00000000000,
    0x3fe6c00000000000,
    0x3fe6a00000000000,
    0x3fe6800000000000,
    0x3fe6600000000000,
    0x3fe6400000000000,
    0x3fe6200000000000,
    0x3fe6000000000000,
    0x3fe5e00000000000,
    0x3fe5c00000000000,
    0x3fe5a00000000000,
    0x3fe5800000000000,
    0x3fe5600000000000,
    0x3fe5400000000000,
    0x3fe5400000000000,
    0x3fe5200000000000,
    0x3fe5000000000000,
    0x3fe4e00000000000,
    0x3fe4c00000000000,
    0x3fe4a00000000000,
    0x3fe4a00000000000,
    0x3fe4800000000000,
    0x3fe4600000000000,
    0x3fe4400000000000,
    0x3fe4200000000000,
    0x3fe4000000000000,
    0x3fe4000000000000,
    0x3fe3e00000000000,
    0x3fe3c00000000000,
    0x3fe3a00000000000,
    0x3fe3a00000000000,
    0x3fe3800000000000,
    0x3fe3600000000000,
    0x3fe3400000000000,
    0x3fe3400000000000,
    0x3fe3200000000000,
    0x3fe3000000000000,
    0x3fe3000000000000,
    0x3fe2e00000000000,
    0x3fe2c00000000000,
    0x3fe2c00000000000,
    0x3fe2a00000000000,
    0x3fe2800000000000,
    0x3fe2800000000000,
    0x3fe2600000000000,
    0x3fe2400000000000,
    0x3fe2400000000000,
    0x3fe2200000000000,
    0x3fe2000000000000,
    0x3fe2000000000000,
    0x3fe1e00000000000,
    0x3fe1c00000000000,
    0x3fe1c00000000000,
    0x3fe1a00000000000,
    0x3fe1a00000000000,
    0x3fe1800000000000,
    0x3fe1600000000000,
    0x3fe1600000000000,
    0x3fe1400000000000,
    0x3fe1400000000000,
    0x3fe1200000000000,
    0x3fe1000000000000,
    0x3fe1000000000000,
    0x3fe0e00000000000,
    0x3fe0e00000000000,
    0x3fe0c00000000000,
    0x3fe0c00000000000,
    0x3fe0a00000000000,
    0x3fe0a00000000000,
    0x3fe0800000000000,
    0x3fe0800000000000,
    0x3fe0600000000000,
    0x3fe0600000000000,
    0x3fe0400000000000,
    0x3fe0400000000000,
    0x3fe0200000000000,
    0x3fe0000000000000,
];

static LOG_R1: [(u64, u64); 128] = [
    (0x0000000000000000, 0x0000000000000000),
    (0x3c146662d417ced0, 0x3f8010157588de71),
    (0x3c327c8e8416e71f, 0x3f90205658935847),
    (0xbc3d192d0619fa67, 0x3f98492528c8cabf),
    (0x3c4c05cf1d753622, 0x3fa0415d89e74444),
    (0xbc4cdd6f7f4a137e, 0x3fa466aed42de3ea),
    (0x3c3a8be97660a23d, 0x3fa894aa149fb343),
    (0xbc4e48fb0500efd4, 0x3faccb73cdddb2cc),
    (0xbc5dd7009902bf32, 0x3fb08598b59e3a07),
    (0xbc47558367a6acf6, 0x3fb1973bd1465567),
    (0x3c47a976d3b5b45f, 0x3fb3bdf5a7d1ee64),
    (0x3c5f38745c5c450a, 0x3fb5e95a4d9791cb),
    (0xbc272566212cdd05, 0x3fb700d30aeac0e1),
    (0xbc5478a85704ccb7, 0x3fb9335e5d594989),
    (0xbc40057eed1ca59f, 0x3fbb6ac88dad5b1c),
    (0x3c5a38cb559a6706, 0x3fbc885801bc4b23),
    (0xbc4a2bf991780d3f, 0x3fbec739830a1120),
    (0xbc5ac9f4215f9393, 0x3fbfe89139dbd566),
    (0xbc50e63a5f01c691, 0x3fc1178e8227e47c),
    (0xbc4c6ef1d9b2ef7e, 0x3fc1aa2b7e23f72a),
    (0xbc5499a3f25af95f, 0x3fc2d1610c86813a),
    (0x3c57d411a5b944ad, 0x3fc365fcb0159016),
    (0xbc50d5604930f135, 0x3fc4913d8333b561),
    (0xbc271a9682395bfd, 0x3fc527e5e4a1b58d),
    (0xbc3d34f0f4621bed, 0x3fc6574ebe8c133a),
    (0xbc68de59c21e166c, 0x3fc6f0128b756abc),
    (0xbc61232ce70be781, 0x3fc823c16551a3c2),
    (0x3c555aa8b6997a40, 0x3fc8beafeb38fe8c),
    (0x3c5142c507fb7a3d, 0x3fc95a5adcf7017f),
    (0x3c6bcafa9de97203, 0x3fca93ed3c8ad9e3),
    (0xbc66353ab386a94d, 0x3fcb31d8575bce3d),
    (0x3c3dd355f6a516d7, 0x3fcbd087383bd8ad),
    (0x3c660629242471a2, 0x3fcd1037f2655e7b),
    (0x3c5aa11d49f96cb9, 0x3fcdb13db0d48940),
    (0x3c42276041f43042, 0x3fce530effe71012),
    (0xbc508ab2ddc708a0, 0x3fcef5ade4dcffe6),
    (0x3c6f665066f980a2, 0x3fcf991c6cb3b379),
    (0x3c7cdb16ed4e9138, 0x3fd07138604d5862),
    (0x3c5162c79d5d11ee, 0x3fd0c42d676162e3),
    (0xbc60e63a5f01c691, 0x3fd1178e8227e47c),
    (0x3c766fbd28b40935, 0x3fd16b5ccbacfb73),
    (0xbc612aeb84249223, 0x3fd1bf99635a6b95),
    (0x3c7e0efadd9db02b, 0x3fd269621134db92),
    (0xbc782dad7fd86088, 0x3fd2bef07cdc9354),
    (0xbc73d69909e5c3dc, 0x3fd314f1e1d35ce4),
    (0xbc5324f0e883858e, 0x3fd36b6776be1117),
    (0xbc72ad27e50a8ec6, 0x3fd3c25277333184),
    (0x3c60dbb243827392, 0x3fd419b423d5e8c7),
    (0x3c38fb4c14c56eef, 0x3fd4718dc271c41b),
    (0xbc5123615b147a5d, 0x3fd4c9e09e172c3c),
    (0xbc68f7e9b38a6979, 0x3fd522ae0738a3d8),
    (0xbc60908d15f88b63, 0x3fd57bf753c8d1fb),
    (0xbc76541148cbb8a2, 0x3fd5d5bddf595f30),
    (0x3c6dc18ce51fff99, 0x3fd630030b3aac49),
    (0x3c5a64eadd740178, 0x3fd68ac83e9c6a14),
    (0x3c5657c222d868cd, 0x3fd6e60ee6af1972),
    (0x3c784a4ee3059583, 0x3fd741d876c67bb1),
    (0xbc7c168817443f22, 0x3fd79e26687cfb3e),
    (0xbc5219024acd3b77, 0x3fd7fafa3bd8151c),
    (0xbc7486666443b153, 0x3fd85855776dcbfb),
    (0xbc770f2f38238303, 0x3fd8b639a88b2df5),
    (0xbc7ad4bb98c1f2c5, 0x3fd914a8635bf68a),
    (0xbc689d2816cf838f, 0x3fd973a3431356ae),
    (0x3c487bcbcfd3e187, 0x3fd9d32bea15ed3b),
    (0xbc6ba8062860ae23, 0x3fda33440224fa79),
    (0xbc6ba8062860ae23, 0x3fda33440224fa79),
    (0x3c7bcafa9de97203, 0x3fda93ed3c8ad9e3),
    (0x3c79d56c45dd3e86, 0x3fdaf5295248cdd0),
    (0x3c7494b610665378, 0x3fdb56fa04462909),
    (0x3c46fd02999b21e1, 0x3fdbb9611b80e2fb),
    (0xbc7bfc00b8f3feaa, 0x3fdc1c60693fa39e),
    (0xbc7bfc00b8f3feaa, 0x3fdc1c60693fa39e),
    (0x3c6223eadb651b4a, 0x3fdc7ff9c74554c9),
    (0x3c70798270b29f39, 0x3fdce42f18064743),
    (0x3c7d7f4d3b3d406b, 0x3fdd490246defa6b),
    (0xbc70b5837185a661, 0x3fddae75484c9616),
    (0xbc7ac81cc8a4dfb8, 0x3fde148a1a2726ce),
    (0xbc7ac81cc8a4dfb8, 0x3fde148a1a2726ce),
    (0x3c757d646a17bc6a, 0x3fde7b42c3ddad73),
    (0xbc174b71fb5e57e3, 0x3fdee2a156b413e5),
    (0xbc60d487f5aba5e5, 0x3fdf4aa7ee03192d),
    (0xbc60d487f5aba5e5, 0x3fdf4aa7ee03192d),
    (0x3c67e8f05924d259, 0x3fdfb358af7a4884),
    (0x3c61713a36138e19, 0x3fe00e5ae5b207ab),
    (0xbc617f9e54e78104, 0x3fe04360be7603ad),
    (0xbc617f9e54e78104, 0x3fe04360be7603ad),
    (0x3c62241edf5fd1f7, 0x3fe078bf0533c568),
    (0x3c80d710fcfc4e0d, 0x3fe0ae76e2d054fa),
    (0x3c80d710fcfc4e0d, 0x3fe0ae76e2d054fa),
    (0x3c83300f002e836e, 0x3fe0e4898611cce1),
    (0xbc891eee7772c7c2, 0x3fe11af823c75aa8),
    (0xbc891eee7772c7c2, 0x3fe11af823c75aa8),
    (0x3c7342eb628dba17, 0x3fe151c3f6f29612),
    (0x3c889df1568ca0b0, 0x3fe188ee40f23ca6),
    (0x3c889df1568ca0b0, 0x3fe188ee40f23ca6),
    (0x3c759bddae1ccce2, 0x3fe1c07849ae6007),
    (0xbc72164ff40e9817, 0x3fe1f8635fc61659),
    (0xbc72164ff40e9817, 0x3fe1f8635fc61659),
    (0xbc6fcc8dbccc25cb, 0x3fe230b0d8bebc98),
    (0x3c8e0efadd9db02b, 0x3fe269621134db92),
    (0x3c8e0efadd9db02b, 0x3fe269621134db92),
    (0xbc76a0c343be95dc, 0x3fe2a2786d0ec107),
    (0xbc7b941ee770436b, 0x3fe2dbf557b0df43),
    (0xbc7b941ee770436b, 0x3fe2dbf557b0df43),
    (0x3c66c3a5f12642c9, 0x3fe315da4434068b),
    (0x3c66c3a5f12642c9, 0x3fe315da4434068b),
    (0xbc7f01ab6065515c, 0x3fe35028ad9d8c86),
    (0x3c821512aa596ea3, 0x3fe38ae2171976e7),
    (0x3c821512aa596ea3, 0x3fe38ae2171976e7),
    (0x3c71930603d87b6e, 0x3fe3c6080c36bfb5),
    (0x3c71930603d87b6e, 0x3fe3c6080c36bfb5),
    (0x3c686cf0f38b461a, 0x3fe4019c2125ca93),
    (0xbc784f481051f71a, 0x3fe43d9ff2f923c5),
    (0xbc784f481051f71a, 0x3fe43d9ff2f923c5),
    (0x3c82541aca7d5844, 0x3fe47a1527e8a2d3),
    (0x3c82541aca7d5844, 0x3fe47a1527e8a2d3),
    (0x3c8c457b531506f6, 0x3fe4b6fd6f970c1f),
    (0x3c8c457b531506f6, 0x3fe4b6fd6f970c1f),
    (0x3c7d749362382a77, 0x3fe4f45a835a4e19),
    (0x3c7d749362382a77, 0x3fe4f45a835a4e19),
    (0x3c7988ba4aea614d, 0x3fe5322e26867857),
    (0x3c7988ba4aea614d, 0x3fe5322e26867857),
    (0x3c880bff3303dd48, 0x3fe5707a26bb8c66),
    (0x3c880bff3303dd48, 0x3fe5707a26bb8c66),
    (0xbc86714fbcd8135b, 0x3fe5af405c3649e0),
    (0xbc86714fbcd8135b, 0x3fe5af405c3649e0),
    (0x3c71c066d235ee63, 0x3fe5ee82aa241920),
    (0x0000000000000000, 0x0000000000000000),
];

#[cfg(not(any(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "fma"
    ),
    all(target_arch = "aarch64", target_feature = "neon")
)))]
pub(crate) static LOG_CD: [u64; 128] = [
    0x0000000000000000,
    0xbf10000000000000,
    0xbf30000000000000,
    0xbf42000000000000,
    0xbf50000000000000,
    0xbf59000000000000,
    0xbf62000000000000,
    0xbf68800000000000,
    0xbf70000000000000,
    0xbf49000000000000,
    0xbf5f000000000000,
    0xbf69c00000000000,
    0xbf30000000000000,
    0xbf5c000000000000,
    0xbf6b000000000000,
    0xbf45000000000000,
    0xbf64000000000000,
    0x3f10000000000000,
    0xbf60000000000000,
    0x3f3a000000000000,
    0xbf5e000000000000,
    0x3f38000000000000,
    0xbf61000000000000,
    0xbf00000000000000,
    0xbf66000000000000,
    0xbf4a000000000000,
    0xbf6e000000000000,
    0xbf5f800000000000,
    0xbf30000000000000,
    0xbf6c000000000000,
    0xbf5f000000000000,
    0xbf3c000000000000,
    0xbf70000000000000,
    0xbf65400000000000,
    0xbf56000000000000,
    0xbf24000000000000,
    0x3f50000000000000,
    0xbf68800000000000,
    0xbf60800000000000,
    0xbf52000000000000,
    0xbf30000000000000,
    0x3f42000000000000,
    0xbf70000000000000,
    0xbf6ac00000000000,
    0xbf66000000000000,
    0xbf61c00000000000,
    0xbf5c000000000000,
    0xbf55800000000000,
    0xbf50000000000000,
    0xbf47000000000000,
    0xbf40000000000000,
    0xbf36000000000000,
    0xbf30000000000000,
    0xbf2c000000000000,
    0xbf30000000000000,
    0xbf36000000000000,
    0xbf40000000000000,
    0xbf47000000000000,
    0xbf50000000000000,
    0xbf55800000000000,
    0xbf5c000000000000,
    0xbf61c00000000000,
    0xbf66000000000000,
    0xbf6ac00000000000,
    0xbf70000000000000,
    0x3f55000000000000,
    0x3f42000000000000,
    0xbf30000000000000,
    0xbf52000000000000,
    0xbf60800000000000,
    0xbf68800000000000,
    0x3f60c00000000000,
    0x3f50000000000000,
    0xbf24000000000000,
    0xbf56000000000000,
    0xbf65400000000000,
    0xbf70000000000000,
    0x3f50000000000000,
    0xbf3c000000000000,
    0xbf5f000000000000,
    0xbf6c000000000000,
    0x3f56800000000000,
    0xbf30000000000000,
    0xbf5f800000000000,
    0xbf6e000000000000,
    0x3f51000000000000,
    0xbf4a000000000000,
    0xbf66000000000000,
    0x3f60000000000000,
    0xbf00000000000000,
    0xbf61000000000000,
    0x3f64800000000000,
    0x3f38000000000000,
    0xbf5e000000000000,
    0x3f66000000000000,
    0x3f3a000000000000,
    0xbf60000000000000,
    0x3f64800000000000,
    0x3f10000000000000,
    0xbf64000000000000,
    0x3f60000000000000,
    0xbf45000000000000,
    0xbf6b000000000000,
    0x3f51000000000000,
    0xbf5c000000000000,
    0x3f65400000000000,
    0xbf30000000000000,
    0xbf69c00000000000,
    0x3f52000000000000,
    0xbf5f000000000000,
    0x3f63000000000000,
    0xbf49000000000000,
    0xbf70000000000000,
    0x3f30000000000000,
    0xbf68800000000000,
    0x3f52800000000000,
    0xbf62000000000000,
    0x3f5f000000000000,
    0xbf59000000000000,
    0x3f64c00000000000,
    0xbf50000000000000,
    0x3f69000000000000,
    0xbf42000000000000,
    0x3f6c400000000000,
    0xbf30000000000000,
    0x3f6e800000000000,
    0xbf10000000000000,
    0xbf70000000000000,
];

#[inline(always)]
pub(crate) fn f_polyeval4(x: f64, a0: f64, a1: f64, a2: f64, a3: f64) -> f64 {
    let t1 = f_fmla(x, a3, a2); // a3 * x + a2
    let t2 = f_fmla(x, t1, a1); // (a3 * x + a2) * x + a1
    f_fmla(x, t2, a0) // ((a3 * x + a2) * x + a1) * x + a0
}

pub(crate) const LOG_COEFFS: [u64; 6] = [
    0xbfdfffffffffffff,
    0x3fd5555555554a9b,
    0xbfd0000000094567,
    0x3fc99999dcc9823c,
    0xbfc55550ac2e537a,
    0x3fc21a02c4e624d7,
];

/// Natural logarithm using FMA
///
/// Max found ULP 0.5
#[inline]
pub fn f_log2(x: f64) -> f64 {
    let mut x_u = x.to_bits();

    const E_BIAS: u64 = (1u64 << (11 - 1u64)) - 1u64;

    let mut x_e: i64 = -(E_BIAS as i64);

    const MIN_NORMAL: u64 = f64::to_bits(f64::MIN_POSITIVE);
    const MAX_NORMAL: u64 = f64::to_bits(f64::MAX);

    if x_u == 1f64.to_bits() {
        // log2(1.0) = +0.0
        return 0.0;
    }
    if x_u < MIN_NORMAL || x_u > MAX_NORMAL {
        if x == 0.0 {
            return f64::NEG_INFINITY;
        }
        if x < 0. || x.is_nan() {
            return f64::NAN;
        }
        if x.is_infinite() || x.is_nan() {
            return x + x;
        }
        // Normalize denormal inputs.
        x_u = (x * f64::from_bits(0x4330000000000000)).to_bits();
        x_e -= 52;
    }

    // log2(x) = log2(2^x_e * x_m)
    //         = x_e + log2(x_m)
    // Range reduction for log2(x_m):
    // For each x_m, we would like to find r such that:
    //   -2^-8 <= r * x_m - 1 < 2^-7
    let shifted = (x_u >> 45) as i64;
    let index = shifted & 0x7F;

    // Add unbiased exponent. Add an extra 1 if the 8 leading fractional bits are
    // all 1's.
    x_e = x_e.wrapping_add(x_u.wrapping_add(1u64 << 45).wrapping_shr(52) as i64);
    let e_x = x_e as f64;

    // Set m = 1.mantissa.
    let x_m = (x_u & 0x000F_FFFF_FFFF_FFFFu64) | 0x3FF0_0000_0000_0000u64;
    let m = f64::from_bits(x_m);

    let mut r1;
    let u;

    let r = f64::from_bits(LOG_RANGE_REDUCTION[index as usize]);

    #[cfg(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    ))]
    {
        u = f_fmla(r, m, -1.0); // exact   
    }
    #[cfg(not(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        let c_m = x_m & 0x3FFF_E000_0000_0000u64;
        let c = f64::from_bits(c_m);
        u = f_fmla(r, m - c, f64::from_bits(LOG_CD[index as usize])); // exact
    }

    // Exact sum:
    let log_vals = LOG_R1[index as usize];

    r1 = Dekker::from_exact_add(f64::from_bits(log_vals.1), u);

    // Error of u_sq = ulp(u^2);
    let u_sq = u * u;
    // Degree-7 minimax polynomial
    let p0 = f_fmla(
        u,
        f64::from_bits(LOG_COEFFS[1]),
        f64::from_bits(LOG_COEFFS[0]),
    );
    let p1 = f_fmla(
        u,
        f64::from_bits(LOG_COEFFS[3]),
        f64::from_bits(LOG_COEFFS[2]),
    );
    let p2 = f_fmla(
        u,
        f64::from_bits(LOG_COEFFS[5]),
        f64::from_bits(LOG_COEFFS[4]),
    );
    let p = f_polyeval4(u_sq, f64::from_bits(log_vals.0), p0, p1, p2);

    r1.lo += p;

    // Quick double-double multiplication:
    //   r2.hi + r2.lo ~ r1 * log2(e),
    // with error bounded by:
    //   4*ulp( ulp(r2.hi) )

    const LOG2_E: Dekker = Dekker::new(
        f64::from_bits(0x3c7777d0ffda0d24),
        f64::from_bits(0x3ff71547652b82fe),
    );
    let r2 = Dekker::quick_mult(r1, LOG2_E);
    let mut r3 = Dekker::from_exact_add(e_x, r2.hi);
    r3.lo += r2.lo;

    // Overall, if we choose sufficiently large constant C, the total error is
    // bounded by (C * ulp(u^2)).

    r3.hi + r3.lo
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log2d() {
        assert_eq!(f_log2(24.), 4.584962500721156181453738943);
        assert!((f_log2(0.35) - 0.35f64.log2()).abs() < 1e-8);
        assert!((f_log2(0.9) - 0.9f64.log2()).abs() < 1e-8);
        assert_eq!(f_log2(0.), f64::NEG_INFINITY);
        assert!(f_log2(-1.).is_nan());
        assert!(f_log2(f64::NAN).is_nan());
        assert!(f_log2(f64::NEG_INFINITY).is_nan());
        assert_eq!(f_log2(f64::INFINITY), f64::INFINITY);
    }
}
