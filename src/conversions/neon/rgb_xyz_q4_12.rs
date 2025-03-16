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
use crate::conversions::rgbxyz_fixed::TransformProfileRgbFixedPoint;
use crate::{CmsError, Layout, TransformExecutor};
use num_traits::AsPrimitive;
use std::arch::aarch64::*;

pub(crate) struct TransformProfileRgbQ12Neon<
    T: Copy,
    const SRC_LAYOUT: u8,
    const DST_LAYOUT: u8,
    const LINEAR_CAP: usize,
    const GAMMA_LUT: usize,
    const BIT_DEPTH: usize,
> {
    pub(crate) profile: TransformProfileRgbFixedPoint<i16, T, LINEAR_CAP>,
}

impl<
    T: Copy + AsPrimitive<usize> + 'static + Default,
    const SRC_LAYOUT: u8,
    const DST_LAYOUT: u8,
    const LINEAR_CAP: usize,
    const GAMMA_LUT: usize,
    const BIT_DEPTH: usize,
> TransformExecutor<T>
    for TransformProfileRgbQ12Neon<T, SRC_LAYOUT, DST_LAYOUT, LINEAR_CAP, GAMMA_LUT, BIT_DEPTH>
where
    u32: AsPrimitive<T>,
{
    fn transform(&self, src: &[T], dst: &mut [T]) -> Result<(), CmsError> {
        let src_cn = Layout::from(SRC_LAYOUT);
        let dst_cn = Layout::from(DST_LAYOUT);
        let src_channels = src_cn.channels();
        let dst_channels = dst_cn.channels();

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
        let max_colors: T = ((1 << BIT_DEPTH) - 1).as_();

        unsafe {
            let m0 = vld1_s16([t.v[0][0], t.v[0][1], t.v[0][2], 0].as_ptr());
            let m1 = vld1_s16([t.v[1][0], t.v[1][1], t.v[1][2], 0].as_ptr());
            let m2 = vld1_s16([t.v[2][0], t.v[2][1], t.v[2][2], 0].as_ptr());

            let v_max_value = vdup_n_u16(GAMMA_LUT as u16 - 1);

            let rnd = vdupq_n_s32((1 << (12 - 1)) - 1);

            let mut src_iter = src.chunks_exact(src_channels * 2);

            let (mut r0, mut g0, mut b0, mut a0);
            let (mut r1, mut g1, mut b1, mut a1);

            if let Some(src) = src_iter.next() {
                let r0p = &self.profile.r_linear[src[src_cn.r_i()].as_()];
                let g0p = &self.profile.g_linear[src[src_cn.g_i()].as_()];
                let b0p = &self.profile.b_linear[src[src_cn.b_i()].as_()];

                let r1p = &self.profile.r_linear[src[src_cn.r_i() + src_channels].as_()];
                let g1p = &self.profile.g_linear[src[src_cn.g_i() + src_channels].as_()];
                let b1p = &self.profile.b_linear[src[src_cn.b_i() + src_channels].as_()];
                r0 = vld1_dup_s16(r0p);
                g0 = vld1_dup_s16(g0p);
                b0 = vld1_dup_s16(b0p);

                r1 = vld1_dup_s16(r1p);
                g1 = vld1_dup_s16(g1p);
                b1 = vld1_dup_s16(b1p);

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
                r0 = vdup_n_s16(0);
                g0 = vdup_n_s16(0);
                b0 = vdup_n_s16(0);
                r1 = vdup_n_s16(0);
                g1 = vdup_n_s16(0);
                b1 = vdup_n_s16(0);
                a0 = max_colors;
                a1 = max_colors;
            }

            for (src, dst) in src_iter.zip(dst.chunks_exact_mut(dst_channels * 2)) {
                let v0_0 = vmlal_s16(rnd, r0, m0);
                let v0_1 = vmlal_s16(rnd, r1, m0);

                let v1_0 = vmlal_s16(v0_0, g0, m1);
                let v1_1 = vmlal_s16(v0_1, g1, m1);

                let vr0 = vmlal_s16(v1_0, b0, m2);
                let vr1 = vmlal_s16(v1_1, b1, m2);

                let mut vr0 = vqshrun_n_s32::<12>(vr0);
                let mut vr1 = vqshrun_n_s32::<12>(vr1);

                vr0 = vmin_u16(vr0, v_max_value);
                vr1 = vmin_u16(vr1, v_max_value);

                let r0p = &self.profile.r_linear[src[src_cn.r_i()].as_()];
                let g0p = &self.profile.g_linear[src[src_cn.g_i()].as_()];
                let b0p = &self.profile.b_linear[src[src_cn.b_i()].as_()];

                let r1p = &self.profile.r_linear[src[src_cn.r_i() + src_channels].as_()];
                let g1p = &self.profile.g_linear[src[src_cn.g_i() + src_channels].as_()];
                let b1p = &self.profile.b_linear[src[src_cn.b_i() + src_channels].as_()];
                r0 = vld1_dup_s16(r0p);
                g0 = vld1_dup_s16(g0p);
                b0 = vld1_dup_s16(b0p);

                r1 = vld1_dup_s16(r1p);
                g1 = vld1_dup_s16(g1p);
                b1 = vld1_dup_s16(b1p);

                dst[dst_cn.r_i()] = self.profile.r_gamma[vget_lane_u16::<0>(vr0) as usize];
                dst[dst_cn.g_i()] = self.profile.g_gamma[vget_lane_u16::<1>(vr0) as usize];
                dst[dst_cn.b_i()] = self.profile.b_gamma[vget_lane_u16::<2>(vr0) as usize];
                if dst_channels == 4 {
                    dst[dst_cn.a_i()] = a0;
                }

                dst[dst_cn.r_i() + dst_channels] =
                    self.profile.r_gamma[vget_lane_u16::<0>(vr1) as usize];
                dst[dst_cn.g_i() + dst_channels] =
                    self.profile.g_gamma[vget_lane_u16::<1>(vr1) as usize];
                dst[dst_cn.b_i() + dst_channels] =
                    self.profile.b_gamma[vget_lane_u16::<2>(vr0) as usize];
                if dst_channels == 4 {
                    dst[dst_cn.a_i() + dst_channels] = a1;
                }

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
            }

            if let Some(dst) = dst.chunks_exact_mut(dst_channels * 2).last() {
                let v0_0 = vmlal_s16(rnd, r0, m0);
                let v0_1 = vmlal_s16(rnd, r1, m0);

                let v1_0 = vmlal_s16(v0_0, g0, m1);
                let v1_1 = vmlal_s16(v0_1, g1, m1);

                let vr0 = vmlal_s16(v1_0, b0, m2);
                let vr1 = vmlal_s16(v1_1, b1, m2);

                let mut vr0 = vqshrun_n_s32::<12>(vr0);
                let mut vr1 = vqshrun_n_s32::<12>(vr1);

                vr0 = vmin_u16(vr0, v_max_value);
                vr1 = vmin_u16(vr1, v_max_value);

                dst[dst_cn.r_i()] = self.profile.r_gamma[vget_lane_u16::<0>(vr0) as usize];
                dst[dst_cn.g_i()] = self.profile.g_gamma[vget_lane_u16::<1>(vr0) as usize];
                dst[dst_cn.b_i()] = self.profile.b_gamma[vget_lane_u16::<2>(vr0) as usize];
                if dst_channels == 4 {
                    dst[dst_cn.a_i()] = a0;
                }

                dst[dst_cn.r_i() + dst_channels] =
                    self.profile.r_gamma[vget_lane_u16::<0>(vr1) as usize];
                dst[dst_cn.g_i() + dst_channels] =
                    self.profile.g_gamma[vget_lane_u16::<1>(vr1) as usize];
                dst[dst_cn.b_i() + dst_channels] =
                    self.profile.b_gamma[vget_lane_u16::<2>(vr0) as usize];
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
                let rp = &self.profile.r_linear[src[src_cn.r_i()].as_()];
                let gp = &self.profile.g_linear[src[src_cn.g_i()].as_()];
                let bp = &self.profile.b_linear[src[src_cn.b_i()].as_()];
                let r = vld1_dup_s16(rp);
                let g = vld1_dup_s16(gp);
                let b = vld1_dup_s16(bp);
                let a = if src_channels == 4 {
                    src[src_cn.a_i()]
                } else {
                    max_colors
                };

                let v0 = vmlal_s16(rnd, r, m0);
                let v1 = vmlal_s16(v0, g, m1);
                let v = vmlal_s16(v1, b, m2);

                let mut vr0 = vqshrun_n_s32::<12>(v);
                vr0 = vmin_u16(vr0, v_max_value);

                dst[dst_cn.r_i()] = self.profile.r_gamma[vget_lane_u16::<0>(vr0) as usize];
                dst[dst_cn.g_i()] = self.profile.g_gamma[vget_lane_u16::<1>(vr0) as usize];
                dst[dst_cn.b_i()] = self.profile.b_gamma[vget_lane_u16::<2>(vr0) as usize];
                if dst_channels == 4 {
                    dst[dst_cn.a_i()] = a;
                }
            }
        }

        Ok(())
    }
}
