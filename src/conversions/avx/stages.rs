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
use crate::conversions::TransformProfileRgb;
use crate::{CmsError, Layout, Matrix3f, TransformExecutor};
use num_traits::AsPrimitive;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
pub(crate) fn _mm_opt_fmlaf_ps<const FMA: bool>(a: __m128, b: __m128, c: __m128) -> __m128 {
    unsafe {
        if FMA {
            _mm_fmadd_ps(b, c, a)
        } else {
            _mm_add_ps(_mm_mul_ps(b, c), a)
        }
    }
}

#[repr(align(16), C)]
struct AvxAlignedU16([u16; 8]);

pub(crate) struct TransformProfilePcsXYZRgbAvx<
    T: Clone + AsPrimitive<usize> + Default,
    const SRC_LAYOUT: u8,
    const DST_LAYOUT: u8,
    const LINEAR_CAP: usize,
    const GAMMA_LUT: usize,
    const BIT_DEPTH: usize,
> {
    pub(crate) profile: TransformProfileRgb<T, LINEAR_CAP>,
}

impl<
    T: Clone + AsPrimitive<usize> + Default,
    const SRC_LAYOUT: u8,
    const DST_LAYOUT: u8,
    const LINEAR_CAP: usize,
    const GAMMA_LUT: usize,
    const BIT_DEPTH: usize,
> TransformProfilePcsXYZRgbAvx<T, SRC_LAYOUT, DST_LAYOUT, LINEAR_CAP, GAMMA_LUT, BIT_DEPTH>
where
    u32: AsPrimitive<T>,
{
    #[inline(always)]
    unsafe fn transform_impl<const FMA: bool>(
        &self,
        src: &[T],
        dst: &mut [T],
    ) -> Result<(), CmsError> {
        let src_cn = Layout::from(SRC_LAYOUT);
        let dst_cn = Layout::from(DST_LAYOUT);
        let src_channels = src_cn.channels();
        let dst_channels = dst_cn.channels();

        let mut temporary0 = AvxAlignedU16([0; 8]);
        let mut temporary1 = AvxAlignedU16([0; 8]);

        if src.len() / src_channels != dst.len() / dst_channels {
            return Err(CmsError::LaneSizeMismatch);
        }
        if src.len() % src_channels != 0 {
            return Err(CmsError::LaneMultipleOfChannels);
        }
        if dst.len() % dst_channels != 0 {
            return Err(CmsError::LaneMultipleOfChannels);
        }

        let t = self
            .profile
            .adaptation_matrix
            .unwrap_or(Matrix3f::IDENTITY)
            .transpose();

        let scale = (GAMMA_LUT - 1) as f32;
        let max_colors = (1 << BIT_DEPTH) - 1;

        unsafe {
            let m0 = _mm_setr_ps(t.v[0][0], t.v[0][1], t.v[0][2], 0f32);
            let m1 = _mm_setr_ps(t.v[1][0], t.v[1][1], t.v[1][2], 0f32);
            let m2 = _mm_setr_ps(t.v[2][0], t.v[2][1], t.v[2][2], 0f32);

            let zeros = _mm_setzero_ps();

            let v_scale = _mm_set1_ps(scale);
            let rnd = _mm_set1_ps(0.5f32);

            for (src, dst) in src
                .chunks_exact(src_channels * 2)
                .zip(dst.chunks_exact_mut(dst_channels * 2))
            {
                let r0 =
                    _mm_broadcast_ss(&self.profile.r_linear.get_unchecked(src[src_cn.r_i()].as_()));
                let g0 =
                    _mm_broadcast_ss(&self.profile.g_linear.get_unchecked(src[src_cn.g_i()].as_()));
                let b0 =
                    _mm_broadcast_ss(&self.profile.b_linear.get_unchecked(src[src_cn.b_i()].as_()));
                let a0 = if src_channels == 4 {
                    f32::from_bits(src[src_cn.a_i()].as_() as u32)
                } else {
                    f32::from_bits(max_colors)
                };

                let r1 = _mm_broadcast_ss(
                    &self
                        .profile
                        .r_linear
                        .get_unchecked(src[src_cn.r_i() + src_channels].as_()),
                );
                let g1 = _mm_broadcast_ss(
                    &self
                        .profile
                        .g_linear
                        .get_unchecked(src[src_cn.g_i() + src_channels].as_()),
                );
                let b1 = _mm_broadcast_ss(
                    &self
                        .profile
                        .b_linear
                        .get_unchecked(src[src_cn.b_i() + src_channels].as_()),
                );
                let a1 = if src_channels == 4 {
                    f32::from_bits(src[src_cn.a_i() + src_channels].as_() as u32)
                } else {
                    f32::from_bits(max_colors)
                };

                let v0_0 = _mm_mul_ps(r0, m0);
                let v0_1 = _mm_mul_ps(r1, m0);
                let v1_0 = _mm_mul_ps(g0, m1);
                let v1_1 = _mm_mul_ps(g1, m1);
                let v2_0 = _mm_mul_ps(b0, m2);
                let v2_1 = _mm_mul_ps(b1, m2);

                let mut vr_0 = _mm_add_ps(_mm_add_ps(v0_0, v1_0), v2_0);
                let mut vr_1 = _mm_add_ps(_mm_add_ps(v0_1, v1_1), v2_1);
                vr_0 = _mm_max_ps(vr_0, zeros);
                vr_1 = _mm_max_ps(vr_1, zeros);
                vr_0 = _mm_opt_fmlaf_ps::<FMA>(rnd, vr_0, v_scale);
                vr_1 = _mm_opt_fmlaf_ps::<FMA>(rnd, vr_1, v_scale);
                vr_0 = _mm_min_ps(vr_0, v_scale);
                vr_1 = _mm_min_ps(vr_1, v_scale);

                let zx0 = _mm_cvtps_epi32(vr_0);
                let zx1 = _mm_cvtps_epi32(vr_1);
                _mm_store_si128(temporary0.0.as_mut_ptr() as *mut _, zx0);
                _mm_store_si128(temporary1.0.as_mut_ptr() as *mut _, zx1);

                dst[dst_cn.r_i()] = self.profile.r_gamma[temporary0.0[0] as usize];
                dst[dst_cn.g_i()] = self.profile.g_gamma[temporary0.0[2] as usize];
                dst[dst_cn.b_i()] = self.profile.b_gamma[temporary0.0[4] as usize];
                if dst_channels == 4 {
                    dst[dst_cn.a_i()] = a0.to_bits().as_();
                }

                dst[dst_cn.r_i() + dst_channels] = self.profile.r_gamma[temporary1.0[0] as usize];
                dst[dst_cn.g_i() + dst_channels] = self.profile.g_gamma[temporary1.0[2] as usize];
                dst[dst_cn.b_i() + dst_channels] = self.profile.b_gamma[temporary1.0[4] as usize];
                if dst_channels == 4 {
                    dst[dst_cn.a_i() + dst_channels] = a1.to_bits().as_();
                }
            }

            let src = src.chunks_exact(src_channels * 2).remainder();
            let dst = dst.chunks_exact_mut(dst_channels * 2).into_remainder();

            for (src, dst) in src
                .chunks_exact(src_channels)
                .zip(dst.chunks_exact_mut(dst_channels))
            {
                let r =
                    _mm_broadcast_ss(&self.profile.r_linear.get_unchecked(src[src_cn.r_i()].as_()));
                let g =
                    _mm_broadcast_ss(&self.profile.g_linear.get_unchecked(src[src_cn.g_i()].as_()));
                let b =
                    _mm_broadcast_ss(&self.profile.b_linear.get_unchecked(src[src_cn.b_i()].as_()));
                let a = if src_channels == 4 {
                    f32::from_bits(src[src_cn.a_i()].as_() as u32)
                } else {
                    f32::from_bits(max_colors)
                };

                let v0 = _mm_mul_ps(r, m0);
                let v1 = _mm_mul_ps(g, m1);
                let v2 = _mm_mul_ps(b, m2);

                let mut v = _mm_add_ps(_mm_add_ps(v0, v1), v2);
                v = _mm_max_ps(v, zeros);
                v = _mm_opt_fmlaf_ps::<FMA>(rnd, v, v_scale);
                v = _mm_min_ps(v, v_scale);

                let zx = _mm_cvtps_epi32(v);
                _mm_store_si128(temporary0.0.as_mut_ptr() as *mut _, zx);

                dst[dst_cn.r_i()] = self.profile.r_gamma[temporary0.0[0] as usize];
                dst[dst_cn.g_i()] = self.profile.g_gamma[temporary0.0[2] as usize];
                dst[dst_cn.b_i()] = self.profile.b_gamma[temporary0.0[4] as usize];
                if dst_channels == 4 {
                    dst[dst_cn.a_i()] = a.to_bits().as_();
                }
            }
        }

        Ok(())
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn transform_fma(&self, src: &[T], dst: &mut [T]) -> Result<(), CmsError> {
        unsafe { self.transform_impl::<true>(src, dst) }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn transform_avx(&self, src: &[T], dst: &mut [T]) -> Result<(), CmsError> {
        unsafe { self.transform_impl::<false>(src, dst) }
    }
}

impl<
    T: Clone + AsPrimitive<usize> + Default,
    const SRC_LAYOUT: u8,
    const DST_LAYOUT: u8,
    const LINEAR_CAP: usize,
    const GAMMA_LUT: usize,
    const BIT_DEPTH: usize,
> TransformExecutor<T>
    for TransformProfilePcsXYZRgbAvx<T, SRC_LAYOUT, DST_LAYOUT, LINEAR_CAP, GAMMA_LUT, BIT_DEPTH>
where
    u32: AsPrimitive<T>,
{
    fn transform(&self, src: &[T], dst: &mut [T]) -> Result<(), CmsError> {
        unsafe {
            if std::arch::is_x86_feature_detected!("fma") {
                self.transform_fma(src, dst)
            } else {
                self.transform_avx(src, dst)
            }
        }
    }
}
