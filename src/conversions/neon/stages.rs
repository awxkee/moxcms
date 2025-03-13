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
use std::arch::aarch64::*;

#[repr(align(16), C)]
pub(crate) struct NeonAlignedU16([u16; 8]);

#[repr(align(16), C)]
pub(crate) struct NeonAlignedU32(pub(crate) [u32; 4]);

pub(crate) struct TransformProfilePcsXYZRgbNeon<
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
> TransformExecutor<T>
    for TransformProfilePcsXYZRgbNeon<T, SRC_LAYOUT, DST_LAYOUT, LINEAR_CAP, GAMMA_LUT, BIT_DEPTH>
where
    u32: AsPrimitive<T>,
{
    fn transform(&self, src: &[T], dst: &mut [T]) -> Result<(), CmsError> {
        let src_cn = Layout::from(SRC_LAYOUT);
        let dst_cn = Layout::from(DST_LAYOUT);
        let src_channels = src_cn.channels();
        let dst_channels = dst_cn.channels();

        let mut temporary0 = NeonAlignedU16([0; 8]);
        let mut temporary1 = NeonAlignedU16([0; 8]);

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
        let max_colors: T = ((1 << BIT_DEPTH) - 1).as_();

        unsafe {
            let m0 = vld1q_f32([t.v[0][0], t.v[0][1], t.v[0][2], 0f32].as_ptr());
            let m1 = vld1q_f32([t.v[1][0], t.v[1][1], t.v[1][2], 0f32].as_ptr());
            let m2 = vld1q_f32([t.v[2][0], t.v[2][1], t.v[2][2], 0f32].as_ptr());

            let zeros = vdupq_n_f32(0f32);

            let v_scale = vdupq_n_f32(scale);

            let rnd = vdupq_n_f32(0.5f32);

            for (src, dst) in src
                .chunks_exact(src_channels * 2)
                .zip(dst.chunks_exact_mut(dst_channels * 2))
            {
                let r0 =
                    vld1q_dup_f32(self.profile.r_linear.get_unchecked(src[src_cn.r_i()].as_()));
                let g0 =
                    vld1q_dup_f32(self.profile.g_linear.get_unchecked(src[src_cn.g_i()].as_()));
                let b0 =
                    vld1q_dup_f32(self.profile.b_linear.get_unchecked(src[src_cn.b_i()].as_()));

                let r1 = vld1q_dup_f32(
                    self.profile
                        .r_linear
                        .get_unchecked(src[src_cn.r_i() + src_channels].as_()),
                );
                let g1 = vld1q_dup_f32(
                    self.profile
                        .g_linear
                        .get_unchecked(src[src_cn.g_i() + src_channels].as_()),
                );
                let b1 = vld1q_dup_f32(
                    self.profile
                        .b_linear
                        .get_unchecked(src[src_cn.b_i() + src_channels].as_()),
                );

                let a0 = if src_channels == 4 {
                    src[src_cn.a_i()]
                } else {
                    max_colors
                };

                let a1 = if src_channels == 4 {
                    src[src_cn.a_i() + src_channels]
                } else {
                    max_colors
                };

                let v0_0 = vmulq_f32(r0, m0);
                let v1_0 = vmulq_f32(g0, m1);
                let v2_0 = vmulq_f32(b0, m2);

                let v0_1 = vmulq_f32(r1, m0);
                let v1_1 = vmulq_f32(g1, m1);
                let v2_1 = vmulq_f32(b1, m2);

                let mut vr0 = vaddq_f32(vaddq_f32(v0_0, v1_0), v2_0);
                let mut vr1 = vaddq_f32(vaddq_f32(v0_1, v1_1), v2_1);
                vr0 = vmaxq_f32(vr0, zeros);
                vr1 = vmaxq_f32(vr1, zeros);
                vr0 = vfmaq_f32(rnd, vr0, v_scale);
                vr1 = vfmaq_f32(rnd, vr1, v_scale);
                vr0 = vminq_f32(vr0, v_scale);
                vr1 = vminq_f32(vr1, v_scale);

                let zx0 = vcvtq_u32_f32(vr0);
                let zx1 = vcvtq_u32_f32(vr1);
                vst1q_u32(temporary0.0.as_mut_ptr() as *mut _, zx0);
                vst1q_u32(temporary1.0.as_mut_ptr() as *mut _, zx1);

                dst[dst_cn.r_i()] = self.profile.r_gamma[temporary0.0[0] as usize];
                dst[dst_cn.g_i()] = self.profile.g_gamma[temporary0.0[2] as usize];
                dst[dst_cn.b_i()] = self.profile.b_gamma[temporary0.0[4] as usize];
                if dst_channels == 4 {
                    dst[dst_cn.a_i()] = a0;
                }

                dst[dst_cn.r_i() + dst_channels] = self.profile.r_gamma[temporary1.0[0] as usize];
                dst[dst_cn.g_i() + dst_channels] = self.profile.g_gamma[temporary1.0[2] as usize];
                dst[dst_cn.b_i() + dst_channels] = self.profile.b_gamma[temporary1.0[4] as usize];
                if dst_channels == 4 {
                    dst[dst_cn.a_i() + dst_channels] = a1;
                }
            }

            let src = src.chunks_exact(src_channels * 2).remainder();
            let dst = dst.chunks_exact_mut(dst_channels * 2).into_remainder();

            for (src, dst) in src
                .chunks_exact(src_channels)
                .zip(dst.chunks_exact_mut(dst_channels))
            {
                let r = vld1q_dup_f32(self.profile.r_linear.get_unchecked(src[src_cn.r_i()].as_()));
                let g = vld1q_dup_f32(self.profile.g_linear.get_unchecked(src[src_cn.g_i()].as_()));
                let b = vld1q_dup_f32(self.profile.b_linear.get_unchecked(src[src_cn.b_i()].as_()));
                let a = if src_channels == 4 {
                    src[src_cn.a_i()]
                } else {
                    max_colors
                };

                let v0 = vmulq_f32(r, m0);
                let v1 = vmulq_f32(g, m1);
                let v2 = vmulq_f32(b, m2);

                let mut v = vaddq_f32(vaddq_f32(v0, v1), v2);
                v = vmaxq_f32(v, zeros);
                v = vfmaq_f32(rnd, v, v_scale);
                v = vminq_f32(v, v_scale);

                let zx = vcvtq_u32_f32(v);
                vst1q_u32(temporary0.0.as_mut_ptr() as *mut _, zx);

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
