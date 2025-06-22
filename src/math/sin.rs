/*
 * // Copyright (c) Radzivon Bartoshyk 6/2025. All rights reserved.
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

// For 2^-7 < |x| < 2^16, return k and u such that:
//   k = round(x * 128/pi)
//   x mod pi/128 = x - k * pi/128 ~ u.hi + u.lo
// Error bound:
//   |(x - k * pi/128) - (u_hi + u_lo)| <= max(ulp(ulp(u_hi)), 2^-119)
//                                      <= 2^-111.
#[inline]
pub(crate) fn range_reduction_small(x: f64) -> (Dekker, u64) {
    const MPI_OVER_128: [u64; 3] = [0xbf9921fb54400000, 0xbd70b4611a600000, 0xbb43198a2e037073];
    const ONE_TWENTY_EIGHT_OVER_PI_D: f64 = f64::from_bits(0x40445f306dc9c883);
    let prod_hi = x * ONE_TWENTY_EIGHT_OVER_PI_D;
    let kd = prod_hi.round();

    // Let y = x - k * (pi/128)
    // Then |y| < pi / 256
    // With extra rounding errors, we can bound |y| < 1.6 * 2^-7.
    let y_hi = f_fmla(kd, f64::from_bits(MPI_OVER_128[0]), x); // Exact
    // |u.hi| < 1.6*2^-7
    let u_hi = f_fmla(kd, f64::from_bits(MPI_OVER_128[1]), y_hi);

    let u0 = y_hi - u_hi; // Exact
    // |u.lo| <= max(ulp(u.hi), |kd * MPI_OVER_128[2]|)
    let u1 = f_fmla(kd, f64::from_bits(MPI_OVER_128[1]), u0); // Exact
    let u_lo = f_fmla(kd, f64::from_bits(MPI_OVER_128[2]), u1);
    // Error bound:
    // |x - k * pi/128| - (u.hi + u.lo) <= ulp(u.lo)
    //                                  <= ulp(max(ulp(u.hi), kd*MPI_OVER_128[2]))
    //                                  <= 2^(-7 - 104) = 2^-111.
    (Dekker::new(u_lo, u_hi), (kd as i64) as u64)
}

static ONE_TWENTY_EIGHT_OVER_PI: [(u64, u64, u64, u64); 128] = [
    (
        0x4040000000000014,
        0x3ce7cc1b727220a9,
        0x3983f84eafa3ea6a,
        0xb6211f924eb53362,
    ),
    (
        0x4040000000145f30,
        0x3ceb727220a94fe1,
        0x397d5f47d4d37703,
        0x361b6295993c4390,
    ),
    (
        0x404000145f306dca,
        0xbcdbbead603d8a83,
        0x395f534ddc0db629,
        0x35f664f10e4107f9,
    ),
    (
        0x40445f306dc9c883,
        0xbce6b01ec5417056,
        0xb986447e493ad4ce,
        0x362e21c820ff28b2,
    ),
    (
        0xc03f246c6efab581,
        0x3ca3abe8fa9a6ee0,
        0x394b6c52b3278872,
        0x35b07f9458eaf7af,
    ),
    (
        0x403391054a7f09d6,
        0xbca70565911f924f,
        0x3942b32788720840,
        0xb5dae9c5421443aa,
    ),
    (
        0x401529fc2757d1f5,
        0x3caa6ee06db14acd,
        0xb948778df7c035d4,
        0x35ed5ef5de2b0db9,
    ),
    (
        0xbfeec54170565912,
        0x3c4b6c52b3278872,
        0x38b07f9458eaf7af,
        0xb52d4f246dc8e2df,
    ),
    (
        0xc04505c1596447e5,
        0x3ceb14acc9e21c82,
        0x395fe5163abdebbc,
        0x35f586dc91b8e909,
    ),
    (
        0xc00596447e493ad5,
        0x3c993c439041fe51,
        0x3938eaf7aef1586e,
        0xb5cb7238b7b645a4,
    ),
    (
        0x404bb81b6c52b328,
        0xbcede37df00d74e3,
        0x3987bd778ac36e49,
        0xb611c5bdb22d1ffa,
    ),
    (
        0x404b6c52b3278872,
        0x3cb07f9458eaf7af,
        0xb92d4f246dc8e2df,
        0x35b374b801924bbb,
    ),
    (
        0x4042b32788720840,
        0xbcdae9c5421443aa,
        0x395b7246e3a424dd,
        0x35e700324977504f,
    ),
    (
        0xc048778df7c035d4,
        0x3ced5ef5de2b0db9,
        0x3971b8e909374b80,
        0x35f924bba8274648,
    ),
    (
        0xc03bef806ba71508,
        0xbcd443a9e48db91c,
        0xb976f6c8b47fe6db,
        0xb61115f62e6de302,
    ),
    (
        0xbfdae9c5421443aa,
        0x3c5b7246e3a424dd,
        0x38e700324977504f,
        0xb58cdbc603c429c7,
    ),
    (
        0xc0438a84288753c9,
        0xbccb7238b7b645a4,
        0x38f924bba8274648,
        0x359cfe1deb1cb12a,
    ),
    (
        0xc020a21d4f246dc9,
        0x3cad2126e9700325,
        0xb94a22bec5cdbc60,
        0xb5de214e34ed658c,
    ),
    (
        0xc02d4f246dc8e2df,
        0x3cb374b801924bbb,
        0xb95f62e6de301e21,
        0xb5f38d3b5963045e,
    ),
    (
        0xc03236e4716f6c8b,
        0xbcd1ff9b6d115f63,
        0x395921cfe1deb1cb,
        0x35d29a73ee88235f,
    ),
    (
        0x403b8e909374b802,
        0xbcdb6d115f62e6de,
        0xb9680f10a71a76b3,
        0x35fcfba208d7d4bb,
    ),
    (
        0x40309374b801924c,
        0xbcd15f62e6de301e,
        0xb960a71a76b2c609,
        0x3601046bea5d7689,
    ),
    (
        0xc0268ffcdb688afb,
        0xbca736f180f10a72,
        0x39462534e7dd1047,
        0xb5e0568a25dbd8b3,
    ),
    (
        0x3ff924bba8274648,
        0x3c9cfe1deb1cb12a,
        0xb9363045df7282b4,
        0xb5d44bb7b16638fe,
    ),
    (
        0xc04a22bec5cdbc60,
        0xbcde214e34ed658c,
        0xb95177dca0ad144c,
        0x35f213a671c09ad1,
    ),
    (
        0x4003a32439fc3bd6,
        0x3c9cb129a73ee882,
        0x392afa975da24275,
        0xb5b8e3f652e82070,
    ),
    (
        0xc03b78c0788538d4,
        0x3cd29a73ee88235f,
        0x3974baed1213a672,
        0xb60fb29741037d8d,
    ),
    (
        0x404fc3bd63962535,
        0xbcc822efb9415a29,
        0x396a24274ce38136,
        0xb60741037d8cdc54,
    ),
    (
        0xc014e34ed658c117,
        0xbcbf7282b4512edf,
        0x394d338e04d68bf0,
        0xb5dbec66e29c67cb,
    ),
    (
        0x40462534e7dd1047,
        0xbce0568a25dbd8b3,
        0xb96c7eca5d040df6,
        0xb5f9b8a719f2b318,
    ),
    (
        0xc0363045df7282b4,
        0xbcd44bb7b16638fe,
        0x397ad17df904e647,
        0x361639835339f49d,
    ),
    (
        0x404d1046bea5d769,
        0xbcebd8b31c7eca5d,
        0xb94037d8cdc538d0,
        0x35ea99cfa4e422fc,
    ),
    (
        0x402afa975da24275,
        0xbcb8e3f652e82070,
        0x3953991d63983534,
        0xb5f82d8dee81d108,
    ),
    (
        0xc04a28976f62cc72,
        0x3ca35a2fbf209cc9,
        0xb924e33e566305b2,
        0x35c08bf177bf2507,
    ),
    (
        0xc0476f62cc71fb29,
        0xbced040df633714e,
        0xb979f2b3182d8def,
        0x361f8bbdf9283b20,
    ),
    (
        0x404d338e04d68bf0,
        0xbcdbec66e29c67cb,
        0x3969cfa4e422fc5e,
        0xb5e036be27003b40,
    ),
    (
        0x403c09ad17df904e,
        0x3cd91d639835339f,
        0x397272117e2ef7e5,
        0xb617c4e007680022,
    ),
    (
        0x40468befc827323b,
        0xbcdc67cacc60b638,
        0x39717e2ef7e4a0ec,
        0x361ff897ffde0598,
    ),
    (
        0xc04037d8cdc538d0,
        0x3cea99cfa4e422fc,
        0x39877bf250763ff1,
        0x3617ffde05980fef,
    ),
    (
        0xc048cdc538cf9599,
        0x3cdf49c845f8bbe0,
        0xb97b5f13801da001,
        0x361e05980fef2f12,
    ),
    (
        0xc024e33e566305b2,
        0x3cc08bf177bf2507,
        0x3968ffc4bffef02d,
        0xb5ffc04343b9d298,
    ),
    (
        0xc03f2b3182d8dee8,
        0xbcbd1081b5f13802,
        0x3942fffbc0b301fe,
        0xb5ca1dce94beb25c,
    ),
    (
        0xc048c16c6f740e88,
        0xbce036be27003b40,
        0xb920fd33f8086877,
        0xb5bd297d64b824b2,
    ),
    (
        0x4043908bf177bf25,
        0x3cad8ffc4bffef03,
        0xb939fc04343b9d29,
        0xb5df592e092c9813,
    ),
    (
        0x4037e2ef7e4a0ec8,
        0xbc7da00087e99fc0,
        0xb910d0ee74a5f593,
        0x359f6d367ecf27cb,
    ),
    (
        0xc03081b5f13801da,
        0xbc20fd33f8086877,
        0xb8bd297d64b824b2,
        0xb558130d834f648b,
    ),
    (
        0xc04af89c00ed0004,
        0xbcdfa67f010d0ee7,
        0xb97297d64b824b26,
        0xb5d30d834f648b0c,
    ),
    (
        0xc04c00ed00043f4d,
        0x3c8fde5e2316b415,
        0xb912e092c98130d8,
        0xb5aa7b24585ce04d,
    ),
    (
        0x4042fffbc0b301fe,
        0xbcca1dce94beb25c,
        0xb9425930261b069f,
        0x35db74f463f669e6,
    ),
    (
        0xc020fd33f8086877,
        0xbcbd297d64b824b2,
        0xb958130d834f648b,
        0xb5c738132c3402ba,
    ),
    (
        0xc039fc04343b9d29,
        0xbcdf592e092c9813,
        0xb94b069ec9161738,
        0xb5c32c3402ba515b,
    ),
    (
        0xc010d0ee74a5f593,
        0x3c9f6d367ecf27cb,
        0x39036e9e8c7ecd3d,
        0xb5a00ae9456c229c,
    ),
    (
        0xc04dce94beb25c12,
        0xbce64c0986c1a7b2,
        0xb98161738132c340,
        0xb615d28ad8453814,
    ),
    (
        0xc044beb25c125930,
        0xbcd30d834f648b0c,
        0x3978fd9a797fa8b6,
        0xb605b08a7028341d,
    ),
    (
        0x403b47db4d9fb3ca,
        0xbcaa7b24585ce04d,
        0x3943cbfd45aea4f7,
        0x35e63f5f2f8bd9e8,
    ),
    (
        0xc0425930261b069f,
        0x3cdb74f463f669e6,
        0xb915d28ad8453814,
        0xb59a0e84c2f8c608,
    ),
    (
        0x403fb3c9f2c26dd4,
        0xbcc738132c3402ba,
        0xb96456c229c0a0d0,
        0xb60d0985f18c10eb,
    ),
    (
        0xc04b069ec9161738,
        0xbcc32c3402ba515b,
        0xb9314e050683a131,
        0x35d0739f78a5292f,
    ),
    (
        0xc04ec9161738132c,
        0xbcda015d28ad8454,
        0x397faf97c5ecf41d,
        0xb5f821d6b5b45650,
    ),
    (
        0xc0461738132c3403,
        0x3ce16ba93dd63f5f,
        0x3977c5ecf41ce7de,
        0x3604a525d4d7f6bf,
    ),
    (
        0x402fb34f2ff516bb,
        0xbccb08a7028341d1,
        0x3969e839cfbc5295,
        0xb60a2b2809409dc1,
    ),
    (
        0x4043cbfd45aea4f7,
        0x3ce63f5f2f8bd9e8,
        0x397ce7de294a4baa,
        0xb61404a04ee072a3,
    ),
    (
        0xc015d28ad8453814,
        0xbc9a0e84c2f8c608,
        0xb93d6b5b45650128,
        0xb5b3b81ca8bdea7f,
    ),
    (
        0xc0415b08a7028342,
        0x3cd7b3d0739f78a5,
        0x396497535fdafd89,
        0xb5bca8bdea7f33ee,
    ),
    (
        0x0000000000000000,
        0x0000000000000000,
        0x3c5f938a73db97fb,
        0x3f992155f7a3667c,
    ),
    (
        0xbc2912bd0d569a90,
        0x3fa91f65f10dd814,
        0x3c7ccbeeeae8129a,
        0x3fb2d52092ce19f4,
    ),
    (
        0xbc3e2718d26ed688,
        0x3fb917a6bc29b42c,
        0xbc7cbb1f71aca352,
        0x3fbf564e56a97310,
    ),
    (
        0xbc8dd9ffeaecbdc4,
        0x3fc2c8106e8e613c,
        0xbc8ab3802218894f,
        0x3fc5e214448b3fc8,
    ),
    (
        0xbc849b466e7fe360,
        0x3fc8f8b83c69a60c,
        0xbc8035e2873ca432,
        0x3fcc0b826a7e4f64,
    ),
    (
        0xbc850b7bbc4768b1,
        0x3fcf19f97b215f1c,
        0xbc83ed9efaa42ab3,
        0x3fd111d262b1f678,
    ),
    (
        0x3c9a8b5c974ee7b5,
        0x3fd294062ed59f04,
        0x3c94325f12be8946,
        0x3fd4135c94176600,
    ),
    (
        0x3c8fc2047e54e614,
        0x3fd58f9a75ab1fdc,
        0xbc9512c678219317,
        0x3fd7088530fa45a0,
    ),
    (
        0xbc92e59dba7ab4c2,
        0x3fd87de2a6aea964,
        0xbc9d24afdade848b,
        0x3fd9ef7943a8ed8c,
    ),
    (
        0x3c65b362cb974183,
        0x3fdb5d1009e15cc0,
        0xbc9e97af1a63c807,
        0x3fdcc66e9931c460,
    ),
    (
        0xbc8c3e4edc5872f8,
        0x3fde2b5d3806f63c,
        0x3c9fb44f80f92225,
        0x3fdf8ba4dbf89ab8,
    ),
    (
        0x3ca9697faf2e2fe5,
        0x3fe073879922ffec,
        0xbca7bc8eda6af93c,
        0x3fe11eb3541b4b24,
    ),
    (
        0x3c8b25dd267f6600,
        0x3fe1c73b39ae68c8,
        0xbca5769d0fbcddc3,
        0x3fe26d054cdd12e0,
    ),
    (
        0x3c9c20673b2116b2,
        0x3fe30ff7fce17034,
        0x3ca3c7c4bc72a92c,
        0x3fe3affa292050b8,
    ),
    (
        0xbcae7f895d302395,
        0x3fe44cf325091dd8,
        0x3ca13c293edceb32,
        0x3fe4e6cabbe3e5e8,
    ),
    (
        0xbc875720992bfbb2,
        0x3fe57d69348ceca0,
        0xbca24a366a5fe547,
        0x3fe610b7551d2ce0,
    ),
    (
        0x3c921165f626cdd5,
        0x3fe6a09e667f3bcc,
        0xbcabcac43c389ba9,
        0x3fe72d0837efff98,
    ),
    (
        0xbca21ea6f59be15b,
        0x3fe7b5df226aafb0,
        0x3cad217be0e2b971,
        0x3fe83b0e0bff976c,
    ),
    (
        0x3c969d0f6897664a,
        0x3fe8bc806b151740,
        0xbc9615f32b6f907a,
        0x3fe93a22499263fc,
    ),
    (
        0x3c96788ebcc76dc6,
        0x3fe9b3e047f38740,
        0x3caddae89fd441d1,
        0x3fea29a7a0462780,
    ),
    (
        0xbc9f98273c5d2495,
        0x3fea9b66290ea1a4,
        0xbc8926da300ffcce,
        0x3feb090a58150200,
    ),
    (
        0x3ca90e58336c64a8,
        0x3feb728345196e3c,
        0x3ca9f6963354e3fe,
        0x3febd7c0ac6f9528,
    ),
    (
        0x3c9a47d3a2a0dcbe,
        0x3fec38b2f180bdb0,
        0x3c9ed0489e16b9a0,
        0x3fec954b213411f4,
    ),
    (
        0xbca0f3db5dad5ac5,
        0x3feced7af43cc774,
        0x3caac42b5a8b6943,
        0x3fed4134d14dc938,
    ),
    (
        0xbcad75033dfb9ca8,
        0x3fed906bcf328d48,
        0x3c883c37c6107db3,
        0x3feddb13b6ccc23c,
    ),
    (
        0x3c97f59c49f6cd6d,
        0x3fee212104f686e4,
        0x3caee94a90d7b88b,
        0x3fee6288ec48e110,
    ),
    (
        0xbcaa27d3874701f9,
        0x3fee9f4156c62ddc,
        0xbc985f4e1b8298d0,
        0x3feed740e7684964,
    ),
    (
        0xbc9ab4e148e52d9e,
        0x3fef0a7efb9230d8,
        0x3c98a11412b82346,
        0x3fef38f3ac64e588,
    ),
    (
        0x3c7562172a361fd3,
        0x3fef6297cff75cb0,
        0x3ca3564acef1ff97,
        0x3fef8764fa714ba8,
    ),
    (
        0xbca5e82a3284d5c8,
        0x3fefa7557f08a518,
        0xbc9709bccb89a989,
        0x3fefc26470e19fd4,
    ),
    (
        0x3ca9e082721dfb8e,
        0x3fefd88da3d12524,
        0xbcaeade132f3981d,
        0x3fefe9cdad01883c,
    ),
    (
        0x3cae3a843d1db55f,
        0x3feff621e3796d7c,
        0x3c9765595d548d9a,
        0x3feffd886084cd0c,
    ),
    (
        0x0000000000000000,
        0x3ff0000000000000,
        0x3c9765595d548d9a,
        0x3feffd886084cd0c,
    ),
    (
        0x3cae3a843d1db55f,
        0x3feff621e3796d7c,
        0xbcaeade132f3981d,
        0x3fefe9cdad01883c,
    ),
    (
        0x3ca9e082721dfb8e,
        0x3fefd88da3d12524,
        0xbc9709bccb89a989,
        0x3fefc26470e19fd4,
    ),
    (
        0xbca5e82a3284d5c8,
        0x3fefa7557f08a518,
        0x3ca3564acef1ff97,
        0x3fef8764fa714ba8,
    ),
    (
        0x3c7562172a361fd3,
        0x3fef6297cff75cb0,
        0x3c98a11412b82346,
        0x3fef38f3ac64e588,
    ),
    (
        0xbc9ab4e148e52d9e,
        0x3fef0a7efb9230d8,
        0xbc985f4e1b8298d0,
        0x3feed740e7684964,
    ),
    (
        0xbcaa27d3874701f9,
        0x3fee9f4156c62ddc,
        0x3caee94a90d7b88b,
        0x3fee6288ec48e110,
    ),
    (
        0x3c97f59c49f6cd6d,
        0x3fee212104f686e4,
        0x3c883c37c6107db3,
        0x3feddb13b6ccc23c,
    ),
    (
        0xbcad75033dfb9ca8,
        0x3fed906bcf328d48,
        0x3caac42b5a8b6943,
        0x3fed4134d14dc938,
    ),
    (
        0xbca0f3db5dad5ac5,
        0x3feced7af43cc774,
        0x3c9ed0489e16b9a0,
        0x3fec954b213411f4,
    ),
    (
        0x3c9a47d3a2a0dcbe,
        0x3fec38b2f180bdb0,
        0x3ca9f6963354e3fe,
        0x3febd7c0ac6f9528,
    ),
    (
        0x3ca90e58336c64a8,
        0x3feb728345196e3c,
        0xbc8926da300ffcce,
        0x3feb090a58150200,
    ),
    (
        0xbc9f98273c5d2495,
        0x3fea9b66290ea1a4,
        0x3caddae89fd441d1,
        0x3fea29a7a0462780,
    ),
    (
        0x3c96788ebcc76dc6,
        0x3fe9b3e047f38740,
        0xbc9615f32b6f907a,
        0x3fe93a22499263fc,
    ),
    (
        0x3c969d0f6897664a,
        0x3fe8bc806b151740,
        0x3cad217be0e2b971,
        0x3fe83b0e0bff976c,
    ),
    (
        0xbca21ea6f59be15b,
        0x3fe7b5df226aafb0,
        0xbcabcac43c389ba9,
        0x3fe72d0837efff98,
    ),
    (
        0x3c921165f626cdd5,
        0x3fe6a09e667f3bcc,
        0xbca24a366a5fe547,
        0x3fe610b7551d2ce0,
    ),
    (
        0xbc875720992bfbb2,
        0x3fe57d69348ceca0,
        0x3ca13c293edceb32,
        0x3fe4e6cabbe3e5e8,
    ),
    (
        0xbcae7f895d302395,
        0x3fe44cf325091dd8,
        0x3ca3c7c4bc72a92c,
        0x3fe3affa292050b8,
    ),
    (
        0x3c9c20673b2116b2,
        0x3fe30ff7fce17034,
        0xbca5769d0fbcddc3,
        0x3fe26d054cdd12e0,
    ),
    (
        0x3c8b25dd267f6600,
        0x3fe1c73b39ae68c8,
        0xbca7bc8eda6af93c,
        0x3fe11eb3541b4b24,
    ),
    (
        0x3ca9697faf2e2fe5,
        0x3fe073879922ffec,
        0x3c9fb44f80f92225,
        0x3fdf8ba4dbf89ab8,
    ),
    (
        0xbc8c3e4edc5872f8,
        0x3fde2b5d3806f63c,
        0xbc9e97af1a63c807,
        0x3fdcc66e9931c460,
    ),
    (
        0x3c65b362cb974183,
        0x3fdb5d1009e15cc0,
        0xbc9d24afdade848b,
        0x3fd9ef7943a8ed8c,
    ),
    (
        0xbc92e59dba7ab4c2,
        0x3fd87de2a6aea964,
        0xbc9512c678219317,
        0x3fd7088530fa45a0,
    ),
    (
        0x3c8fc2047e54e614,
        0x3fd58f9a75ab1fdc,
        0x3c94325f12be8946,
        0x3fd4135c94176600,
    ),
    (
        0x3c9a8b5c974ee7b5,
        0x3fd294062ed59f04,
        0xbc83ed9efaa42ab3,
        0x3fd111d262b1f678,
    ),
    (
        0xbc850b7bbc4768b1,
        0x3fcf19f97b215f1c,
        0xbc8035e2873ca432,
        0x3fcc0b826a7e4f64,
    ),
    (
        0xbc849b466e7fe360,
        0x3fc8f8b83c69a60c,
        0xbc8ab3802218894f,
        0x3fc5e214448b3fc8,
    ),
    (
        0xbc8dd9ffeaecbdc4,
        0x3fc2c8106e8e613c,
        0xbc7cbb1f71aca352,
        0x3fbf564e56a97310,
    ),
    (
        0xbc3e2718d26ed688,
        0x3fb917a6bc29b42c,
        0x3c7ccbeeeae8129a,
        0x3fb2d52092ce19f4,
    ),
    (
        0xbc2912bd0d569a90,
        0x3fa91f65f10dd814,
        0x3c5f938a73db97fb,
        0x3f992155f7a3667c,
    ),
];

#[derive(Default)]
pub(crate) struct LargeArgumentReduction {
    x_reduced: f64,
    idx: u64,
    y_hi: f64,
    // Low part of x * ONE_TWENTY_EIGHT_OVER_PI[idx][1].
    pm_lo: f64,
}

const fn mask_trailing_ones(len: u64) -> u64 {
    if len >= 64 {
        u64::MAX // All ones if length is 64 or more
    } else {
        (1u64 << len).wrapping_sub(1)
    }
}

pub(crate) const EXP_MASK: u64 = mask_trailing_ones(11) << 52;

#[inline]
fn set_exponent_f64(x: u64, new_exp: u64) -> u64 {
    let encoded_mask = new_exp.wrapping_shl(52) & EXP_MASK;
    x ^ ((x ^ encoded_mask) & EXP_MASK)
}

impl LargeArgumentReduction {
    #[inline]
    pub(crate) fn high_part(&mut self, x: f64) -> u64 {
        let mut xbits = x.to_bits();
        let x_e = (x.to_bits() >> 52) & 0x7ff;
        const E_BIAS: u64 = (1u64 << (11 - 1u64)) - 1u64;
        let x_e_m62: i64 = (x_e as i64).wrapping_sub(E_BIAS as i64 + 62);
        self.idx = x_e_m62.wrapping_shr(4).wrapping_add(3) as u64;
        // Scale x down by 2^(-(16 * (idx - 3))

        xbits = set_exponent_f64(xbits, ((x_e_m62 & 15) + E_BIAS as i64 + 62) as u64);
        // 2^62 <= |x_reduced| < 2^(62 + 16) = 2^78
        self.x_reduced = f64::from_bits(xbits);
        // x * c_hi = ph.hi + ph.lo exactly.
        let ph = Dekker::from_exact_mult(
            self.x_reduced,
            f64::from_bits(ONE_TWENTY_EIGHT_OVER_PI[self.idx as usize].0),
        );
        // x * c_mid = pm.hi + pm.lo exactly.
        let pm = Dekker::from_exact_mult(
            self.x_reduced,
            f64::from_bits(ONE_TWENTY_EIGHT_OVER_PI[self.idx as usize].1),
        );
        // Extract integral parts and fractional parts of (ph.lo + pm.hi).
        let kh = ph.lo.round();
        let ph_lo_frac = ph.lo - kh; // Exact
        let km = (pm.hi + ph_lo_frac).round();
        let pm_hi_frac = pm.hi - km; // Exact
        // x * 128/pi mod 1 ~ y_hi + y_lo
        self.y_hi = ph_lo_frac + pm_hi_frac; // Exact
        self.pm_lo = pm.lo;
        (kh as i64 + km as i64) as u64
    }

    #[inline]
    pub(crate) fn reduce(&mut self) -> Dekker {
        let y_lo = f_fmla(
            self.x_reduced,
            f64::from_bits(ONE_TWENTY_EIGHT_OVER_PI[self.idx as usize].2),
            self.pm_lo,
        );
        let y = Dekker::from_exact_add(self.y_hi, y_lo);

        // Digits of pi/128, generated by Sollya with:
        // > a = round(pi/128, D, RN);
        // > b = round(pi/128 - a, D, RN);
        const PI_OVER_128_DD: Dekker = Dekker::new(
            f64::from_bits(0x3c31a62633145c07),
            f64::from_bits(0x3f9921fb54442d18),
        );

        // Error bound: with {a} denote the fractional part of a, i.e.:
        //   {a} = a - round(a)
        // Then,
        //   | {x * 128/pi} - (y_hi + y_lo) | <  2 * ulp(x_reduced *
        //                                         * ONE_TWENTY_EIGHT_OVER_PI[idx][2])
        // For FMA:
        //   | {x * 128/pi} - (y_hi + y_lo) | <= 2 * 2^77 * 2^-103 * 2^-52
        //                                    =  2^-77.
        //   | {x mod pi/128} - (u.hi + u.lo) | < 2 * 2^-6 * 2^-77.
        //                                      = 2^-82.
        // For non-FMA:
        //   | {x * 128/pi} - (y_hi + y_lo) | <= 2 * 2^77 * 2^-99 * 2^-52
        //                                    =  2^-73.
        //   | {x mod pi/128} - (u.hi + u.lo) | < 2 * 2^-6 * 2^-73.
        //                                      = 2^-78.
        Dekker::quick_mult(y, PI_OVER_128_DD)
    }
}

pub(crate) struct SinCos {
    pub(crate) v_sin: Dekker,
    pub(crate) v_cos: Dekker,
}

#[inline]
pub(crate) fn sincos_eval(u: Dekker) -> SinCos {
    // Evaluate sin(y) = sin(x - k * (pi/128))
    // We use the degree-7 Taylor approximation:
    //   sin(y) ~ y - y^3/3! + y^5/5! - y^7/7!
    // Then the error is bounded by:
    //   |sin(y) - (y - y^3/3! + y^5/5! - y^7/7!)| < |y|^9/9! < 2^-54/9! < 2^-72.
    // For y ~ u_hi + u_lo, fully expanding the polynomial and drop any terms
    // < ulp(u_hi^3) gives us:
    //   y - y^3/3! + y^5/5! - y^7/7! = ...
    // ~ u_hi + u_hi^3 * (-1/6 + u_hi^2 * (1/120 - u_hi^2 * 1/5040)) +
    //        + u_lo (1 + u_hi^2 * (-1/2 + u_hi^2 / 24))
    let u_hi_sq = u.hi * u.hi; // Error < ulp(u_hi^2) < 2^(-6 - 52) = 2^-58.
    // p1 ~ 1/120 + u_hi^2 / 5040.
    let p1 = f_fmla(
        u_hi_sq,
        f64::from_bits(0xbf2a01a01a01a01a),
        f64::from_bits(0x3f81111111111111),
    );
    // q1 ~ -1/2 + u_hi^2 / 24.
    let q1 = f_fmla(
        u_hi_sq,
        f64::from_bits(0x3fa5555555555555),
        f64::from_bits(0xbfe0000000000000),
    );
    let u_hi_3 = u_hi_sq * u.hi;
    // p2 ~ -1/6 + u_hi^2 (1/120 - u_hi^2 * 1/5040)
    let p2 = f_fmla(u_hi_sq, p1, f64::from_bits(0xbfc5555555555555));
    // q2 ~ 1 + u_hi^2 (-1/2 + u_hi^2 / 24)
    let q2 = f_fmla(u_hi_sq, q1, 1.0);
    let sin_lo = f_fmla(u_hi_3, p2, u.lo * q2);
    // Overall, |sin(y) - (u_hi + sin_lo)| < 2*ulp(u_hi^3) < 2^-69.

    // Evaluate cos(y) = cos(x - k * (pi/128))
    // We use the degree-8 Taylor approximation:
    //   cos(y) ~ 1 - y^2/2 + y^4/4! - y^6/6! + y^8/8!
    // Then the error is bounded by:
    //   |cos(y) - (...)| < |y|^10/10! < 2^-81
    // For y ~ u_hi + u_lo, fully expanding the polynomial and drop any terms
    // < ulp(u_hi^3) gives us:
    //   1 - y^2/2 + y^4/4! - y^6/6! + y^8/8! = ...
    // ~ 1 - u_hi^2/2 + u_hi^4(1/24 + u_hi^2 (-1/720 + u_hi^2/40320)) +
    //     + u_hi u_lo (-1 + u_hi^2/6)
    // We compute 1 - u_hi^2 accurately:
    //   v_hi + v_lo ~ 1 - u_hi^2/2
    // with error <= 2^-105.
    let u_hi_neg_half = (-0.5) * u.hi;

    let v_hi = f_fmla(u.hi, u_hi_neg_half, 1.0);
    let mut v_lo = 1.0 - v_hi; // Exact
    v_lo = f_fmla(u.hi, u_hi_neg_half, v_lo);

    // r1 ~ -1/720 + u_hi^2 / 40320
    let r1 = f_fmla(
        u_hi_sq,
        f64::from_bits(0x3efa01a01a01a01a),
        f64::from_bits(0xbf56c16c16c16c17),
    );
    // s1 ~ -1 + u_hi^2 / 6
    let s1 = f_fmla(u_hi_sq, f64::from_bits(0x3fc5555555555555), -1.0);
    let u_hi_4 = u_hi_sq * u_hi_sq;
    let u_hi_u_lo = u.hi * u.lo;
    // r2 ~ 1/24 + u_hi^2 (-1/720 + u_hi^2 / 40320)
    let r2 = f_fmla(u_hi_sq, r1, f64::from_bits(0x3fa5555555555555));
    // s2 ~ v_lo + u_hi * u_lo * (-1 + u_hi^2 / 6)
    let s2 = f_fmla(u_hi_u_lo, s1, v_lo);
    let cos_lo = f_fmla(u_hi_4, r2, s2);
    // Overall, |cos(y) - (v_hi + cos_lo)| < 2*ulp(u_hi^4) < 2^-75.

    let sin_u = Dekker::from_exact_add(u.hi, sin_lo);
    let cos_u = Dekker::from_exact_add(v_hi, cos_lo);

    SinCos {
        v_sin: sin_u,
        v_cos: cos_u,
    }
}

pub(crate) static SIN_K_PI_OVER_128: [(u64, u64); 256] = [
    (0x0000000000000000, 0x0000000000000000),
    (0xbbfb1d63091a0130, 0x3f992155f7a3667e),
    (0xbc2912bd0d569a90, 0x3fa91f65f10dd814),
    (0xbc49a088a8bf6b2c, 0x3fb2d52092ce19f6),
    (0xbc3e2718d26ed688, 0x3fb917a6bc29b42c),
    (0x3c4a2704729ae56d, 0x3fbf564e56a9730e),
    (0x3c513000a89a11e0, 0x3fc2c8106e8e613a),
    (0x3c6531ff779ddac6, 0x3fc5e214448b3fc6),
    (0xbc626d19b9ff8d82, 0x3fc8f8b83c69a60b),
    (0xbc1af1439e521935, 0x3fcc0b826a7e4f63),
    (0xbc642deef11da2c4, 0x3fcf19f97b215f1b),
    (0x3c7824c20ab7aa9a, 0x3fd111d262b1f677),
    (0xbc75d28da2c4612d, 0x3fd294062ed59f06),
    (0x3c70c97c4afa2518, 0x3fd4135c94176601),
    (0xbc1efdc0d58cf620, 0x3fd58f9a75ab1fdd),
    (0xbc744b19e0864c5d, 0x3fd7088530fa459f),
    (0xbc672cedd3d5a610, 0x3fd87de2a6aea963),
    (0x3c66da81290bdbab, 0x3fd9ef7943a8ed8a),
    (0x3c65b362cb974183, 0x3fdb5d1009e15cc0),
    (0x3c56850e59c37f8f, 0x3fdcc66e9931c45e),
    (0x3c5e0d891d3c6841, 0x3fde2b5d3806f63b),
    (0xbc32ec1fc1b776b8, 0x3fdf8ba4dbf89aba),
    (0xbc8a5a014347406c, 0x3fe073879922ffee),
    (0xbc8ef23b69abe4f1, 0x3fe11eb3541b4b23),
    (0x3c8b25dd267f6600, 0x3fe1c73b39ae68c8),
    (0xbc85da743ef3770c, 0x3fe26d054cdd12df),
    (0xbc6efcc626f74a6f, 0x3fe30ff7fce17035),
    (0x3c7e3e25e3954964, 0x3fe3affa292050b9),
    (0x3c68076a2cfdc6b3, 0x3fe44cf325091dd6),
    (0x3c63c293edceb327, 0x3fe4e6cabbe3e5e9),
    (0xbc875720992bfbb2, 0x3fe57d69348ceca0),
    (0xbc7251b352ff2a37, 0x3fe610b7551d2cdf),
    (0xbc8bdd3413b26456, 0x3fe6a09e667f3bcd),
    (0x3c80d4ef0f1d915c, 0x3fe72d0837efff96),
    (0xbc70f537acdf0ad7, 0x3fe7b5df226aafaf),
    (0xbc76f420f8ea3475, 0x3fe83b0e0bff976e),
    (0xbc82c5e12ed1336d, 0x3fe8bc806b151741),
    (0x3c83d419a920df0b, 0x3fe93a22499263fb),
    (0xbc830ee286712474, 0x3fe9b3e047f38741),
    (0xbc7128bb015df175, 0x3fea29a7a0462782),
    (0x3c39f630e8b6dac8, 0x3fea9b66290ea1a3),
    (0xbc8926da300ffcce, 0x3feb090a58150200),
    (0xbc8bc69f324e6d61, 0x3feb728345196e3e),
    (0xbc8825a732ac700a, 0x3febd7c0ac6f952a),
    (0xbc76e0b1757c8d07, 0x3fec38b2f180bdb1),
    (0xbc52fb761e946603, 0x3fec954b213411f5),
    (0xbc5e7b6bb5ab58ae, 0x3feced7af43cc773),
    (0xbc84ef5295d25af2, 0x3fed4134d14dc93a),
    (0x3c7457e610231ac2, 0x3fed906bcf328d46),
    (0x3c883c37c6107db3, 0x3feddb13b6ccc23c),
    (0xbc8014c76c126527, 0x3fee212104f686e5),
    (0xbc616b56f2847754, 0x3fee6288ec48e112),
    (0x3c8760b1e2e3f81e, 0x3fee9f4156c62dda),
    (0x3c7e82c791f59cc2, 0x3feed740e7684963),
    (0x3c752c7adc6b4989, 0x3fef0a7efb9230d7),
    (0xbc7d7bafb51f72e6, 0x3fef38f3ac64e589),
    (0x3c7562172a361fd3, 0x3fef6297cff75cb0),
    (0x3c7ab256778ffcb6, 0x3fef8764fa714ba9),
    (0xbc87a0a8ca13571f, 0x3fefa7557f08a517),
    (0x3c81ec8668ecacee, 0x3fefc26470e19fd3),
    (0xbc887df6378811c7, 0x3fefd88da3d12526),
    (0x3c6521ecd0c67e35, 0x3fefe9cdad01883a),
    (0xbc6c57bc2e24aa15, 0x3feff621e3796d7e),
    (0xbc81354d4556e4cb, 0x3feffd886084cd0d),
    (0x0000000000000000, 0x3ff0000000000000),
    (0xbc81354d4556e4cb, 0x3feffd886084cd0d),
    (0xbc6c57bc2e24aa15, 0x3feff621e3796d7e),
    (0x3c6521ecd0c67e35, 0x3fefe9cdad01883a),
    (0xbc887df6378811c7, 0x3fefd88da3d12526),
    (0x3c81ec8668ecacee, 0x3fefc26470e19fd3),
    (0xbc87a0a8ca13571f, 0x3fefa7557f08a517),
    (0x3c7ab256778ffcb6, 0x3fef8764fa714ba9),
    (0x3c7562172a361fd3, 0x3fef6297cff75cb0),
    (0xbc7d7bafb51f72e6, 0x3fef38f3ac64e589),
    (0x3c752c7adc6b4989, 0x3fef0a7efb9230d7),
    (0x3c7e82c791f59cc2, 0x3feed740e7684963),
    (0x3c8760b1e2e3f81e, 0x3fee9f4156c62dda),
    (0xbc616b56f2847754, 0x3fee6288ec48e112),
    (0xbc8014c76c126527, 0x3fee212104f686e5),
    (0x3c883c37c6107db3, 0x3feddb13b6ccc23c),
    (0x3c7457e610231ac2, 0x3fed906bcf328d46),
    (0xbc84ef5295d25af2, 0x3fed4134d14dc93a),
    (0xbc5e7b6bb5ab58ae, 0x3feced7af43cc773),
    (0xbc52fb761e946603, 0x3fec954b213411f5),
    (0xbc76e0b1757c8d07, 0x3fec38b2f180bdb1),
    (0xbc8825a732ac700a, 0x3febd7c0ac6f952a),
    (0xbc8bc69f324e6d61, 0x3feb728345196e3e),
    (0xbc8926da300ffcce, 0x3feb090a58150200),
    (0x3c39f630e8b6dac8, 0x3fea9b66290ea1a3),
    (0xbc7128bb015df175, 0x3fea29a7a0462782),
    (0xbc830ee286712474, 0x3fe9b3e047f38741),
    (0x3c83d419a920df0b, 0x3fe93a22499263fb),
    (0xbc82c5e12ed1336d, 0x3fe8bc806b151741),
    (0xbc76f420f8ea3475, 0x3fe83b0e0bff976e),
    (0xbc70f537acdf0ad7, 0x3fe7b5df226aafaf),
    (0x3c80d4ef0f1d915c, 0x3fe72d0837efff96),
    (0xbc8bdd3413b26456, 0x3fe6a09e667f3bcd),
    (0xbc7251b352ff2a37, 0x3fe610b7551d2cdf),
    (0xbc875720992bfbb2, 0x3fe57d69348ceca0),
    (0x3c63c293edceb327, 0x3fe4e6cabbe3e5e9),
    (0x3c68076a2cfdc6b3, 0x3fe44cf325091dd6),
    (0x3c7e3e25e3954964, 0x3fe3affa292050b9),
    (0xbc6efcc626f74a6f, 0x3fe30ff7fce17035),
    (0xbc85da743ef3770c, 0x3fe26d054cdd12df),
    (0x3c8b25dd267f6600, 0x3fe1c73b39ae68c8),
    (0xbc8ef23b69abe4f1, 0x3fe11eb3541b4b23),
    (0xbc8a5a014347406c, 0x3fe073879922ffee),
    (0xbc32ec1fc1b776b8, 0x3fdf8ba4dbf89aba),
    (0x3c5e0d891d3c6841, 0x3fde2b5d3806f63b),
    (0x3c56850e59c37f8f, 0x3fdcc66e9931c45e),
    (0x3c65b362cb974183, 0x3fdb5d1009e15cc0),
    (0x3c66da81290bdbab, 0x3fd9ef7943a8ed8a),
    (0xbc672cedd3d5a610, 0x3fd87de2a6aea963),
    (0xbc744b19e0864c5d, 0x3fd7088530fa459f),
    (0xbc1efdc0d58cf620, 0x3fd58f9a75ab1fdd),
    (0x3c70c97c4afa2518, 0x3fd4135c94176601),
    (0xbc75d28da2c4612d, 0x3fd294062ed59f06),
    (0x3c7824c20ab7aa9a, 0x3fd111d262b1f677),
    (0xbc642deef11da2c4, 0x3fcf19f97b215f1b),
    (0xbc1af1439e521935, 0x3fcc0b826a7e4f63),
    (0xbc626d19b9ff8d82, 0x3fc8f8b83c69a60b),
    (0x3c6531ff779ddac6, 0x3fc5e214448b3fc6),
    (0x3c513000a89a11e0, 0x3fc2c8106e8e613a),
    (0x3c4a2704729ae56d, 0x3fbf564e56a9730e),
    (0xbc3e2718d26ed688, 0x3fb917a6bc29b42c),
    (0xbc49a088a8bf6b2c, 0x3fb2d52092ce19f6),
    (0xbc2912bd0d569a90, 0x3fa91f65f10dd814),
    (0xbbfb1d63091a0130, 0x3f992155f7a3667e),
    (0x0000000000000000, 0x0000000000000000),
    (0x3bfb1d63091a0130, 0xbf992155f7a3667e),
    (0x3c2912bd0d569a90, 0xbfa91f65f10dd814),
    (0x3c49a088a8bf6b2c, 0xbfb2d52092ce19f6),
    (0x3c3e2718d26ed688, 0xbfb917a6bc29b42c),
    (0xbc4a2704729ae56d, 0xbfbf564e56a9730e),
    (0xbc513000a89a11e0, 0xbfc2c8106e8e613a),
    (0xbc6531ff779ddac6, 0xbfc5e214448b3fc6),
    (0x3c626d19b9ff8d82, 0xbfc8f8b83c69a60b),
    (0x3c1af1439e521935, 0xbfcc0b826a7e4f63),
    (0x3c642deef11da2c4, 0xbfcf19f97b215f1b),
    (0xbc7824c20ab7aa9a, 0xbfd111d262b1f677),
    (0x3c75d28da2c4612d, 0xbfd294062ed59f06),
    (0xbc70c97c4afa2518, 0xbfd4135c94176601),
    (0x3c1efdc0d58cf620, 0xbfd58f9a75ab1fdd),
    (0x3c744b19e0864c5d, 0xbfd7088530fa459f),
    (0x3c672cedd3d5a610, 0xbfd87de2a6aea963),
    (0xbc66da81290bdbab, 0xbfd9ef7943a8ed8a),
    (0xbc65b362cb974183, 0xbfdb5d1009e15cc0),
    (0xbc56850e59c37f8f, 0xbfdcc66e9931c45e),
    (0xbc5e0d891d3c6841, 0xbfde2b5d3806f63b),
    (0x3c32ec1fc1b776b8, 0xbfdf8ba4dbf89aba),
    (0x3c8a5a014347406c, 0xbfe073879922ffee),
    (0x3c8ef23b69abe4f1, 0xbfe11eb3541b4b23),
    (0xbc8b25dd267f6600, 0xbfe1c73b39ae68c8),
    (0x3c85da743ef3770c, 0xbfe26d054cdd12df),
    (0x3c6efcc626f74a6f, 0xbfe30ff7fce17035),
    (0xbc7e3e25e3954964, 0xbfe3affa292050b9),
    (0xbc68076a2cfdc6b3, 0xbfe44cf325091dd6),
    (0xbc63c293edceb327, 0xbfe4e6cabbe3e5e9),
    (0x3c875720992bfbb2, 0xbfe57d69348ceca0),
    (0x3c7251b352ff2a37, 0xbfe610b7551d2cdf),
    (0x3c8bdd3413b26456, 0xbfe6a09e667f3bcd),
    (0xbc80d4ef0f1d915c, 0xbfe72d0837efff96),
    (0x3c70f537acdf0ad7, 0xbfe7b5df226aafaf),
    (0x3c76f420f8ea3475, 0xbfe83b0e0bff976e),
    (0x3c82c5e12ed1336d, 0xbfe8bc806b151741),
    (0xbc83d419a920df0b, 0xbfe93a22499263fb),
    (0x3c830ee286712474, 0xbfe9b3e047f38741),
    (0x3c7128bb015df175, 0xbfea29a7a0462782),
    (0xbc39f630e8b6dac8, 0xbfea9b66290ea1a3),
    (0x3c8926da300ffcce, 0xbfeb090a58150200),
    (0x3c8bc69f324e6d61, 0xbfeb728345196e3e),
    (0x3c8825a732ac700a, 0xbfebd7c0ac6f952a),
    (0x3c76e0b1757c8d07, 0xbfec38b2f180bdb1),
    (0x3c52fb761e946603, 0xbfec954b213411f5),
    (0x3c5e7b6bb5ab58ae, 0xbfeced7af43cc773),
    (0x3c84ef5295d25af2, 0xbfed4134d14dc93a),
    (0xbc7457e610231ac2, 0xbfed906bcf328d46),
    (0xbc883c37c6107db3, 0xbfeddb13b6ccc23c),
    (0x3c8014c76c126527, 0xbfee212104f686e5),
    (0x3c616b56f2847754, 0xbfee6288ec48e112),
    (0xbc8760b1e2e3f81e, 0xbfee9f4156c62dda),
    (0xbc7e82c791f59cc2, 0xbfeed740e7684963),
    (0xbc752c7adc6b4989, 0xbfef0a7efb9230d7),
    (0x3c7d7bafb51f72e6, 0xbfef38f3ac64e589),
    (0xbc7562172a361fd3, 0xbfef6297cff75cb0),
    (0xbc7ab256778ffcb6, 0xbfef8764fa714ba9),
    (0x3c87a0a8ca13571f, 0xbfefa7557f08a517),
    (0xbc81ec8668ecacee, 0xbfefc26470e19fd3),
    (0x3c887df6378811c7, 0xbfefd88da3d12526),
    (0xbc6521ecd0c67e35, 0xbfefe9cdad01883a),
    (0x3c6c57bc2e24aa15, 0xbfeff621e3796d7e),
    (0x3c81354d4556e4cb, 0xbfeffd886084cd0d),
    (0x0000000000000000, 0xbff0000000000000),
    (0x3c81354d4556e4cb, 0xbfeffd886084cd0d),
    (0x3c6c57bc2e24aa15, 0xbfeff621e3796d7e),
    (0xbc6521ecd0c67e35, 0xbfefe9cdad01883a),
    (0x3c887df6378811c7, 0xbfefd88da3d12526),
    (0xbc81ec8668ecacee, 0xbfefc26470e19fd3),
    (0x3c87a0a8ca13571f, 0xbfefa7557f08a517),
    (0xbc7ab256778ffcb6, 0xbfef8764fa714ba9),
    (0xbc7562172a361fd3, 0xbfef6297cff75cb0),
    (0x3c7d7bafb51f72e6, 0xbfef38f3ac64e589),
    (0xbc752c7adc6b4989, 0xbfef0a7efb9230d7),
    (0xbc7e82c791f59cc2, 0xbfeed740e7684963),
    (0xbc8760b1e2e3f81e, 0xbfee9f4156c62dda),
    (0x3c616b56f2847754, 0xbfee6288ec48e112),
    (0x3c8014c76c126527, 0xbfee212104f686e5),
    (0xbc883c37c6107db3, 0xbfeddb13b6ccc23c),
    (0xbc7457e610231ac2, 0xbfed906bcf328d46),
    (0x3c84ef5295d25af2, 0xbfed4134d14dc93a),
    (0x3c5e7b6bb5ab58ae, 0xbfeced7af43cc773),
    (0x3c52fb761e946603, 0xbfec954b213411f5),
    (0x3c76e0b1757c8d07, 0xbfec38b2f180bdb1),
    (0x3c8825a732ac700a, 0xbfebd7c0ac6f952a),
    (0x3c8bc69f324e6d61, 0xbfeb728345196e3e),
    (0x3c8926da300ffcce, 0xbfeb090a58150200),
    (0xbc39f630e8b6dac8, 0xbfea9b66290ea1a3),
    (0x3c7128bb015df175, 0xbfea29a7a0462782),
    (0x3c830ee286712474, 0xbfe9b3e047f38741),
    (0xbc83d419a920df0b, 0xbfe93a22499263fb),
    (0x3c82c5e12ed1336d, 0xbfe8bc806b151741),
    (0x3c76f420f8ea3475, 0xbfe83b0e0bff976e),
    (0x3c70f537acdf0ad7, 0xbfe7b5df226aafaf),
    (0xbc80d4ef0f1d915c, 0xbfe72d0837efff96),
    (0x3c8bdd3413b26456, 0xbfe6a09e667f3bcd),
    (0x3c7251b352ff2a37, 0xbfe610b7551d2cdf),
    (0x3c875720992bfbb2, 0xbfe57d69348ceca0),
    (0xbc63c293edceb327, 0xbfe4e6cabbe3e5e9),
    (0xbc68076a2cfdc6b3, 0xbfe44cf325091dd6),
    (0xbc7e3e25e3954964, 0xbfe3affa292050b9),
    (0x3c6efcc626f74a6f, 0xbfe30ff7fce17035),
    (0x3c85da743ef3770c, 0xbfe26d054cdd12df),
    (0xbc8b25dd267f6600, 0xbfe1c73b39ae68c8),
    (0x3c8ef23b69abe4f1, 0xbfe11eb3541b4b23),
    (0x3c8a5a014347406c, 0xbfe073879922ffee),
    (0x3c32ec1fc1b776b8, 0xbfdf8ba4dbf89aba),
    (0xbc5e0d891d3c6841, 0xbfde2b5d3806f63b),
    (0xbc56850e59c37f8f, 0xbfdcc66e9931c45e),
    (0xbc65b362cb974183, 0xbfdb5d1009e15cc0),
    (0xbc66da81290bdbab, 0xbfd9ef7943a8ed8a),
    (0x3c672cedd3d5a610, 0xbfd87de2a6aea963),
    (0x3c744b19e0864c5d, 0xbfd7088530fa459f),
    (0x3c1efdc0d58cf620, 0xbfd58f9a75ab1fdd),
    (0xbc70c97c4afa2518, 0xbfd4135c94176601),
    (0x3c75d28da2c4612d, 0xbfd294062ed59f06),
    (0xbc7824c20ab7aa9a, 0xbfd111d262b1f677),
    (0x3c642deef11da2c4, 0xbfcf19f97b215f1b),
    (0x3c1af1439e521935, 0xbfcc0b826a7e4f63),
    (0x3c626d19b9ff8d82, 0xbfc8f8b83c69a60b),
    (0xbc6531ff779ddac6, 0xbfc5e214448b3fc6),
    (0xbc513000a89a11e0, 0xbfc2c8106e8e613a),
    (0xbc4a2704729ae56d, 0xbfbf564e56a9730e),
    (0x3c3e2718d26ed688, 0xbfb917a6bc29b42c),
    (0x3c49a088a8bf6b2c, 0xbfb2d52092ce19f6),
    (0x3c2912bd0d569a90, 0xbfa91f65f10dd814),
    (0x3bfb1d63091a0130, 0xbf992155f7a3667e),
];

/// Sine for double precision
///
/// ULP 0.5
#[inline]
pub fn f_sin(x: f64) -> f64 {
    let x_e = (x.to_bits() >> 52) & 0x7ff;
    const E_BIAS: u64 = (1u64 << (11 - 1u64)) - 1u64;

    let y: Dekker;
    let k;

    // |x| < 2^32 (with FMA) or |x| < 2^23 (w/o FMA)
    if x_e < E_BIAS + 22 {
        // |x| < 2^-26
        if x_e < E_BIAS - 26 {
            // Signed zeros.
            if x == 0.0 {
                return x;
            }
            // For |x| < 2^-26, |sin(x) - x| < ulp(x)/2.
            return f_fmla(x, f64::from_bits(0xbc90000000000000), x);
        }

        // // Small range reduction.
        (y, k) = range_reduction_small(x);
    } else {
        // Inf or NaN
        if x_e > 2 * E_BIAS {
            // sin(+-Inf) = NaN
            return x + f64::NAN;
        }

        // Large range reduction.
        let mut argument_reduction = LargeArgumentReduction::default();
        k = argument_reduction.high_part(x);
        y = argument_reduction.reduce();
    }

    let r_sincos = sincos_eval(y);

    // Fast look up version, but needs 256-entry table.
    // cos(k * pi/128) = sin(k * pi/128 + pi/2) = sin((k + 64) * pi/128).
    let sk = SIN_K_PI_OVER_128[(k & 255) as usize];
    let ck = SIN_K_PI_OVER_128[((k.wrapping_add(64)) & 255) as usize];
    let sin_k = Dekker::new(f64::from_bits(sk.0), f64::from_bits(sk.1));
    let cos_k = Dekker::new(f64::from_bits(ck.0), f64::from_bits(ck.1));

    let sin_k_cos_y = Dekker::quick_mult(r_sincos.v_cos, sin_k);
    let cos_k_sin_y = Dekker::quick_mult(r_sincos.v_sin, cos_k);

    let mut rr = Dekker::from_exact_add(sin_k_cos_y.hi, cos_k_sin_y.hi);
    rr.lo += sin_k_cos_y.lo + cos_k_sin_y.lo;
    rr.to_f64()
}

/// Cosine for double precision
///
/// ULP 0.5
#[inline]
pub fn f_cos(x: f64) -> f64 {
    let x_e = (x.to_bits() >> 52) & 0x7ff;
    const E_BIAS: u64 = (1u64 << (11 - 1u64)) - 1u64;

    let y: Dekker;
    let k;

    // |x| < 2^32 (with FMA) or |x| < 2^23 (w/o FMA)
    if x_e < E_BIAS + 22 {
        // |x| < 2^-26
        if x_e < E_BIAS - 7 {
            // |x| < 2^-26
            if x_e < E_BIAS - 27 {
                // Signed zeros.
                if x == 0.0 {
                    return 1.0;
                }
                // For |x| < 2^-26, |sin(x) - x| < ulp(x)/2.
                return 1.0 - f64::EPSILON;
            }
            k = 0;
            y = Dekker::new(0.0, x);
        } else {
            // // Small range reduction.
            (y, k) = range_reduction_small(x);
        }
    } else {
        // Inf or NaN
        if x_e > 2 * E_BIAS {
            // sin(+-Inf) = NaN
            return x + f64::NAN;
        }

        // Large range reduction.
        let mut argument_reduction = LargeArgumentReduction::default();
        k = argument_reduction.high_part(x);
        y = argument_reduction.reduce();
    }
    let r_sincos = sincos_eval(y);

    // Fast look up version, but needs 256-entry table.
    // cos(k * pi/128) = sin(k * pi/128 + pi/2) = sin((k + 64) * pi/128).
    let sk = SIN_K_PI_OVER_128[(k.wrapping_add(128) & 255) as usize];
    let ck = SIN_K_PI_OVER_128[((k.wrapping_add(64)) & 255) as usize];
    let msin_k = Dekker::new(f64::from_bits(sk.0), f64::from_bits(sk.1));
    let cos_k = Dekker::new(f64::from_bits(ck.0), f64::from_bits(ck.1));

    let sin_k_cos_y = Dekker::quick_mult(r_sincos.v_cos, cos_k);
    let cos_k_sin_y = Dekker::quick_mult(r_sincos.v_sin, msin_k);

    let mut rr = Dekker::from_exact_add(sin_k_cos_y.hi, cos_k_sin_y.hi);
    rr.lo += sin_k_cos_y.lo + cos_k_sin_y.lo;
    rr.to_f64()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cos_test() {
        assert_eq!(f_cos(0.0), 1.0);
        assert_eq!(f_cos(1.0), 0.5403023058681398);
        assert_eq!(f_cos(-0.5), 0.8775825618903728);
    }

    #[test]
    fn sin_test() {
        assert_eq!(f_sin(0.0), 0.0);
        assert_eq!(f_sin(1.0), 0.8414709848078965);
        assert_eq!(f_sin(-0.5), -0.479425538604203);
    }
}
