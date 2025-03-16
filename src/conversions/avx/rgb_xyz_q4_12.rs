/*
 * // Copyright (c) Radzivon Bartoshyk 3/2025. All rights reserved.
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
use crate::conversions::avx::stages::AvxAlignedU16;
use crate::conversions::rgbxyz_fixed::TransformProfileRgb8Bit;
use crate::{CmsError, Layout, TransformExecutor};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) struct TransformProfilePcsXYZRgb8BitAvx<
    const SRC_LAYOUT: u8,
    const DST_LAYOUT: u8,
    const LINEAR_CAP: usize,
    const GAMMA_LUT: usize,
> {
    pub(crate) profile: TransformProfileRgb8Bit,
}

impl<const SRC_LAYOUT: u8, const DST_LAYOUT: u8, const LINEAR_CAP: usize, const GAMMA_LUT: usize>
    TransformProfilePcsXYZRgb8BitAvx<SRC_LAYOUT, DST_LAYOUT, LINEAR_CAP, GAMMA_LUT>
{
    #[target_feature(enable = "avx2")]
    unsafe fn transform_avx2(&self, src: &[u8], dst: &mut [u8]) -> Result<(), CmsError> {
        let src_cn = Layout::from(SRC_LAYOUT);
        let dst_cn = Layout::from(DST_LAYOUT);
        let src_channels = src_cn.channels();
        let dst_channels = dst_cn.channels();

        let mut temporary0 = AvxAlignedU16([0; 16]);

        if src.len() / src_channels != dst.len() / dst_channels {
            return Err(CmsError::LaneSizeMismatch);
        }
        if src.len() % src_channels != 0 {
            return Err(CmsError::LaneMultipleOfChannels);
        }
        if dst.len() % dst_channels != 0 {
            return Err(CmsError::LaneMultipleOfChannels);
        }

        let t = self.profile.adaptation_matrix.transpose();

        let max_colors = 255;

        unsafe {
            let m0 = _mm256_setr_epi16(
                t.v[0][0], t.v[1][0], t.v[0][1], t.v[1][1], t.v[0][2], t.v[1][2], 0, 0, t.v[0][0],
                t.v[1][0], t.v[0][1], t.v[1][1], t.v[0][2], t.v[1][2], 0, 0,
            );
            let m2 = _mm256_setr_epi32(
                t.v[2][0] as i32,
                t.v[2][1] as i32,
                t.v[2][2] as i32,
                0,
                t.v[2][0] as i32,
                t.v[2][1] as i32,
                t.v[2][2] as i32,
                0,
            );

            let zeros = _mm256_setzero_si256();

            let v_max_value = _mm256_set1_epi32((1 << 12) - 1);

            let mut src = src;
            let mut dst = dst;

            let mut src_iter = src.chunks_exact(src_channels * 2);
            let dst_iter = dst.chunks_exact_mut(dst_channels * 2);

            let (mut r0, mut b0, mut a0);
            let (mut r1, mut b1, mut a1);

            let shuffle_r_g = _mm256_setr_epi8(
                0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                0, 1, 2, 3,
            );

            if let Some(src) = src_iter.next() {
                r0 = _mm_loadu_si16(
                    (&self.profile.r_linear[src[src_cn.r_i()] as usize] as *const i16).cast(),
                );

                b0 = _mm_set1_epi16(self.profile.b_linear[src[src_cn.b_i()] as usize]);
                r1 = _mm_loadu_si16(
                    (&self.profile.r_linear[src[src_cn.r_i() + src_channels] as usize]
                        as *const i16)
                        .cast(),
                );
                b1 = _mm_set1_epi16(
                    self.profile.b_linear[src[src_cn.b_i() + src_channels] as usize],
                );
                r0 = _mm_insert_epi16::<1>(
                    r0,
                    self.profile.g_linear[src[src_cn.g_i()] as usize] as i32,
                );
                r1 = _mm_insert_epi16::<1>(
                    r1,
                    self.profile.g_linear[src[src_cn.g_i() + src_channels] as usize] as i32,
                );
                a0 = if src_channels == 4 {
                    src[src_cn.a_i()]
                } else {
                    max_colors
                };
                a1 = if src_channels == 4 {
                    src[src_cn.a_i() + src_channels]
                } else {
                    max_colors
                };
            } else {
                r0 = _mm_setzero_si128();
                b0 = _mm_setzero_si128();
                a0 = max_colors;
                r1 = _mm_setzero_si128();
                b1 = _mm_setzero_si128();
                a1 = max_colors;
            }

            for (src, dst) in src_iter.zip(dst_iter) {
                let mut r = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(r0), r1);
                let b = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(b0), b1);
                r = _mm256_shuffle_epi8(r, shuffle_r_g);

                let v0 = _mm256_madd_epi16(r, m0);
                let v2 = _mm256_madd_epi16(b, m2);

                let mut v = _mm256_add_epi32(v0, v2);
                v = _mm256_srai_epi32::<12>(v);
                v = _mm256_max_epi32(v, zeros);
                v = _mm256_min_epi32(v, v_max_value);

                _mm256_store_si256(temporary0.0.as_mut_ptr() as *mut _, v);

                let as0 = a0;
                let as1 = a1;

                r0 = _mm_loadu_si16(
                    (&self.profile.r_linear[src[src_cn.r_i()] as usize] as *const i16).cast(),
                );

                b0 = _mm_loadu_si16(
                    (&self.profile.b_linear[src[src_cn.b_i()] as usize] as *const i16).cast(),
                );
                r1 = _mm_loadu_si16(
                    (&self.profile.r_linear[src[src_cn.r_i() + src_channels] as usize]
                        as *const i16)
                        .cast(),
                );
                b1 = _mm_loadu_si16(
                    (&self.profile.b_linear[src[src_cn.b_i() + src_channels] as usize]
                        as *const i16)
                        .cast(),
                );
                r0 = _mm_insert_epi16::<1>(
                    r0,
                    self.profile.g_linear[src[src_cn.g_i()] as usize] as i32,
                );
                r1 = _mm_insert_epi16::<1>(
                    r1,
                    self.profile.g_linear[src[src_cn.g_i() + src_channels] as usize] as i32,
                );
                b0 = _mm_broadcastw_epi16(b0);
                b1 = _mm_broadcastw_epi16(b1);

                a0 = if src_channels == 4 {
                    src[src_cn.a_i()]
                } else {
                    max_colors
                };
                a1 = if src_channels == 4 {
                    src[src_cn.a_i() + src_channels]
                } else {
                    max_colors
                };

                dst[dst_cn.r_i()] = self.profile.r_gamma[temporary0.0[0] as usize];
                dst[dst_cn.g_i()] = self.profile.g_gamma[temporary0.0[2] as usize];
                dst[dst_cn.b_i()] = self.profile.b_gamma[temporary0.0[4] as usize];
                if dst_channels == 4 {
                    dst[dst_cn.a_i()] = as0;
                }

                dst[dst_cn.r_i() + dst_channels] = self.profile.r_gamma[temporary0.0[8] as usize];
                dst[dst_cn.g_i() + dst_channels] = self.profile.g_gamma[temporary0.0[10] as usize];
                dst[dst_cn.b_i() + dst_channels] = self.profile.b_gamma[temporary0.0[12] as usize];
                if dst_channels == 4 {
                    dst[dst_cn.a_i() + dst_channels] = as1;
                }
            }

            if let Some(dst) = dst.chunks_exact_mut(dst_channels * 2).last() {
                let mut r = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(r0), r1);
                let b = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(b0), b1);
                r = _mm256_shuffle_epi8(r, shuffle_r_g);

                let v0 = _mm256_madd_epi16(r, m0);
                let v2 = _mm256_madd_epi16(b, m2);

                let mut v = _mm256_add_epi32(v0, v2);
                v = _mm256_srai_epi32::<12>(v);
                v = _mm256_max_epi32(v, zeros);
                v = _mm256_min_epi32(v, v_max_value);

                _mm256_store_si256(temporary0.0.as_mut_ptr() as *mut _, v);

                dst[dst_cn.r_i()] = self.profile.r_gamma[temporary0.0[0] as usize];
                dst[dst_cn.g_i()] = self.profile.g_gamma[temporary0.0[2] as usize];
                dst[dst_cn.b_i()] = self.profile.b_gamma[temporary0.0[4] as usize];
                if dst_channels == 4 {
                    dst[dst_cn.a_i()] = a0;
                }

                dst[dst_cn.r_i() + dst_channels] = self.profile.r_gamma[temporary0.0[8] as usize];
                dst[dst_cn.g_i() + dst_channels] = self.profile.g_gamma[temporary0.0[10] as usize];
                dst[dst_cn.b_i() + dst_channels] = self.profile.b_gamma[temporary0.0[12] as usize];
                if dst_channels == 4 {
                    dst[dst_cn.a_i() + dst_channels] = a1;
                }
            }

            src = src.chunks_exact(src_channels * 2).remainder();
            dst = dst.chunks_exact_mut(dst_channels * 2).into_remainder();

            for (src, dst) in src
                .chunks_exact(src_channels)
                .zip(dst.chunks_exact_mut(dst_channels))
            {
                let mut r = _mm_loadu_si16(
                    (&self.profile.r_linear[src[src_cn.r_i()] as usize] as *const i16).cast(),
                );
                let b = _mm_set1_epi16(self.profile.b_linear[src[src_cn.b_i()] as usize]);
                r = _mm_insert_epi16::<1>(
                    r,
                    self.profile.g_linear[src[src_cn.g_i()] as usize] as i32,
                );
                let a = if src_channels == 4 {
                    src[src_cn.a_i()]
                } else {
                    max_colors
                };

                r = _mm_shuffle_epi8(r, _mm256_castsi256_si128(shuffle_r_g));

                let v0 = _mm_madd_epi16(r, _mm256_castsi256_si128(m0));
                let v2 = _mm_madd_epi16(b, _mm256_castsi256_si128(m2));

                let mut v = _mm_add_epi32(v0, v2);
                v = _mm_srai_epi32::<12>(v);
                v = _mm_max_epi32(v, _mm_setzero_si128());
                v = _mm_min_epi32(v, _mm256_castsi256_si128(v_max_value));

                _mm_store_si128(temporary0.0.as_mut_ptr() as *mut _, v);

                dst[dst_cn.r_i()] = self.profile.r_gamma[temporary0.0[0] as usize];
                dst[dst_cn.g_i()] = self.profile.g_gamma[temporary0.0[2] as usize];
                dst[dst_cn.b_i()] = self.profile.b_gamma[temporary0.0[4] as usize];
                if dst_channels == 4 {
                    dst[dst_cn.a_i()] = a;
                }
            }
        }

        Ok(())
    }
}

impl<const SRC_LAYOUT: u8, const DST_LAYOUT: u8, const LINEAR_CAP: usize, const GAMMA_LUT: usize>
    TransformExecutor<u8>
    for TransformProfilePcsXYZRgb8BitAvx<SRC_LAYOUT, DST_LAYOUT, LINEAR_CAP, GAMMA_LUT>
{
    fn transform(&self, src: &[u8], dst: &mut [u8]) -> Result<(), CmsError> {
        unsafe { self.transform_avx2(src, dst) }
    }
}
