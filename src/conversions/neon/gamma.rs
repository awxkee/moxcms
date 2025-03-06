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
use crate::Layout;
use std::arch::aarch64::*;

#[repr(align(16), C)]
struct NeonAlignedU16([u16; 8]);

pub(crate) fn gamma_search_8bit<const SRC_LAYOUT: u8, const DST_LAYOUT: u8>(
    working_set: &[f32],
    dst: &mut [u8],
    r_gamma: &[u8; 65536],
    g_gamma: &[u8; 65536],
    b_gamma: &[u8; 65536],
) {
    unsafe {
        const BIT_DEPTH: usize = 8;
        let max_value = ((1u32 << BIT_DEPTH) - 1) as u8;
        let src_cn = Layout::from(SRC_LAYOUT);
        let src_channels = src_cn.channels();

        let mut temporary0 = NeonAlignedU16([0; 8]);
        let mut temporary1 = NeonAlignedU16([0; 8]);

        let dst_cn = Layout::from(DST_LAYOUT);
        let dst_channels = dst_cn.channels();
        let samples = dst.len() / dst_channels;

        let mut x = 0usize;
        let mut src_x = 0usize;
        let mut dst_x = 0usize;
        if src_channels == 3 {
            while x + 2 < samples {
                let chunk = working_set.get_unchecked(src_x..);

                let dst = dst.get_unchecked_mut(dst_x..);

                let src_vl0 = vcombine_f32(
                    vld1_f32(chunk.as_ptr()),
                    vld1_lane_f32::<0>(chunk.get_unchecked(2..).as_ptr(), vdup_n_f32(0f32)),
                );

                let src_vl1 = vcombine_f32(
                    vld1_f32(chunk.get_unchecked(3..).as_ptr()),
                    vld1_lane_f32::<0>(chunk.get_unchecked(5..).as_ptr(), vdup_n_f32(0f32)),
                );

                let src_f0 = vcvtaq_u32_f32(src_vl0);
                let src_f1 = vcvtaq_u32_f32(src_vl1);

                vst1q_u32(temporary0.0.as_mut_ptr() as *mut _, src_f0);
                vst1q_u32(temporary1.0.as_mut_ptr() as *mut _, src_f1);

                *dst.get_unchecked_mut(dst_cn.r_i()) = r_gamma[temporary0.0[0] as usize];
                *dst.get_unchecked_mut(dst_cn.g_i()) = g_gamma[temporary0.0[2] as usize];
                *dst.get_unchecked_mut(dst_cn.b_i()) = b_gamma[temporary0.0[4] as usize];

                if dst_channels == 4 {
                    dst[dst_cn.a_i()] = max_value;
                }

                *dst.get_unchecked_mut(dst_cn.r_i() + dst_channels) =
                    r_gamma[temporary1.0[0] as usize];
                *dst.get_unchecked_mut(dst_cn.g_i() + dst_channels) =
                    g_gamma[temporary1.0[2] as usize];
                *dst.get_unchecked_mut(dst_cn.b_i() + dst_channels) =
                    b_gamma[temporary1.0[4] as usize];

                if dst_channels == 4 {
                    dst[dst_cn.a_i() + dst_channels] = max_value;
                }

                x += 2;
                src_x += src_channels * 2;
                dst_x += dst_channels * 2;
            }

            while x < samples {
                let chunk = working_set.get_unchecked(src_x..);

                let dst = dst.get_unchecked_mut(dst_x..);

                let src_vl = vcombine_f32(
                    vld1_f32(chunk.as_ptr()),
                    vld1_lane_f32::<0>(chunk.get_unchecked(2..).as_ptr(), vdup_n_f32(0f32)),
                );

                let src_f = vcvtaq_u32_f32(src_vl);

                vst1q_u32(temporary0.0.as_mut_ptr() as *mut _, src_f);

                *dst.get_unchecked_mut(dst_cn.r_i()) = r_gamma[temporary0.0[0] as usize];
                *dst.get_unchecked_mut(dst_cn.g_i()) = g_gamma[temporary0.0[2] as usize];
                *dst.get_unchecked_mut(dst_cn.b_i()) = b_gamma[temporary0.0[4] as usize];

                if dst_channels == 4 {
                    dst[dst_cn.a_i()] = max_value;
                }
                x += 1;
                src_x += src_channels;
                dst_x += dst_channels;
            }
        } else if src_channels == 4 {
            while x + 2 < samples {
                let chunk = working_set.get_unchecked(src_x..);

                let dst = dst.get_unchecked_mut(dst_x..);

                let src_vl0 = vld1q_f32(chunk.as_ptr());
                let src_vl1 = vld1q_f32(chunk.get_unchecked(4..).as_ptr());

                let src_f0 = vcvtaq_u32_f32(src_vl0);
                let src_f1 = vcvtaq_u32_f32(src_vl1);

                vst1q_u32(temporary0.0.as_mut_ptr() as *mut _, src_f0);
                vst1q_u32(temporary1.0.as_mut_ptr() as *mut _, src_f1);

                *dst.get_unchecked_mut(dst_cn.r_i()) = r_gamma[temporary0.0[0] as usize];
                *dst.get_unchecked_mut(dst_cn.g_i()) = g_gamma[temporary0.0[2] as usize];
                *dst.get_unchecked_mut(dst_cn.b_i()) = b_gamma[temporary0.0[4] as usize];
                if dst_channels == 4 {
                    *dst.get_unchecked_mut(dst_cn.a_i()) = temporary0.0[7] as u8;
                }

                *dst.get_unchecked_mut(dst_cn.r_i() + dst_channels) =
                    r_gamma[temporary1.0[0] as usize];
                *dst.get_unchecked_mut(dst_cn.g_i() + dst_channels) =
                    g_gamma[temporary1.0[2] as usize];
                *dst.get_unchecked_mut(dst_cn.b_i() + dst_channels) =
                    b_gamma[temporary1.0[4] as usize];
                if dst_channels == 4 {
                    *dst.get_unchecked_mut(dst_cn.a_i() + dst_channels) = temporary1.0[7] as u8;
                }

                x += 2;
                src_x += src_channels * 2;
                dst_x += dst_channels * 2;
            }

            while x < samples {
                let chunk = working_set.get_unchecked(src_x..);

                let dst = dst.get_unchecked_mut(dst_x..);

                let src_vl = vld1q_f32(chunk.as_ptr());

                let src_f = vcvtaq_u32_f32(src_vl);

                vst1q_u32(temporary0.0.as_mut_ptr() as *mut _, src_f);

                *dst.get_unchecked_mut(dst_cn.r_i()) = r_gamma[temporary0.0[0] as usize];
                *dst.get_unchecked_mut(dst_cn.g_i()) = g_gamma[temporary0.0[2] as usize];
                *dst.get_unchecked_mut(dst_cn.b_i()) = b_gamma[temporary0.0[4] as usize];
                if dst_channels == 4 {
                    *dst.get_unchecked_mut(dst_cn.a_i()) = temporary0.0[7] as u8;
                }

                x += 1;
                src_x += src_channels;
                dst_x += dst_channels;
            }
        }
    }
}
