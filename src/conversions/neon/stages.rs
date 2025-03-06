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
use crate::mlaf::mlaf;
use crate::{CmsError, InPlaceStage, Layout, Matrix3f};
use std::arch::aarch64::*;

pub(crate) struct MatrixClipScaleStageNeon<const LAYOUT: u8> {
    pub(crate) matrix: Matrix3f,
    pub(crate) scale: f32,
}

impl<const LAYOUT: u8> InPlaceStage for MatrixClipScaleStageNeon<LAYOUT> {
    #[inline]
    fn transform(&self, dst: &mut [f32]) -> Result<(), CmsError> {
        let scale = self.scale;
        let cn = Layout::from(LAYOUT);
        let channels = cn.channels();

        let transform = self.matrix;

        let full_length = dst.len();

        let mut x = 0usize;

        unsafe {
            let m0 = vld1q_f32(self.matrix.v.as_ptr() as *mut f32);
            let m1 = vld1q_f32((self.matrix.v.as_ptr() as *mut f32).add(4));
            let m2 = vdupq_n_f32(self.matrix.v[2][2]);
            let v_scale = vdupq_n_f32(scale);
            let zeros = vdupq_n_f32(0f32);

            while x + 4 * channels < full_length {
                let src_data = if channels == 4 {
                    vld4q_f32(dst.get_unchecked(x..).as_ptr())
                } else {
                    let values = vld3q_f32(dst.get_unchecked(x..).as_ptr());
                    float32x4x4_t(values.0, values.1, values.2, vdupq_n_f32(1f32))
                };

                let r = src_data.0;
                let g = src_data.1;
                let b = src_data.2;

                let mut r0 = vmulq_laneq_f32::<0>(r, m0);
                let mut g0 = vmulq_laneq_f32::<3>(r, m0);
                let mut b0 = vmulq_laneq_f32::<2>(r, m1);

                r0 = vfmaq_laneq_f32::<1>(r0, g, m0);
                g0 = vfmaq_laneq_f32::<0>(g0, g, m1);
                b0 = vfmaq_laneq_f32::<3>(b0, g, m1);

                r0 = vfmaq_laneq_f32::<2>(r0, b, m0);
                g0 = vfmaq_laneq_f32::<1>(g0, b, m1);
                b0 = vfmaq_f32(b0, b, m2);

                r0 = vmaxq_f32(r0, zeros);
                g0 = vmaxq_f32(g0, zeros);
                b0 = vmaxq_f32(b0, zeros);

                r0 = vfmaq_f32(vdupq_n_f32(0.5f32), r0, v_scale);
                g0 = vfmaq_f32(vdupq_n_f32(0.5f32), g0, v_scale);
                b0 = vfmaq_f32(vdupq_n_f32(0.5f32), b0, v_scale);

                r0 = vminq_f32(r0, v_scale);
                g0 = vminq_f32(g0, v_scale);
                b0 = vminq_f32(b0, v_scale);

                if channels == 4 {
                    vst4q_f32(
                        dst.get_unchecked_mut(x..).as_mut_ptr(),
                        float32x4x4_t(r0, g0, b0, src_data.3),
                    );
                } else if channels == 3 {
                    vst3q_f32(
                        dst.get_unchecked_mut(x..).as_mut_ptr(),
                        float32x4x3_t(r0, g0, b0),
                    );
                }

                x += 4 * channels;
            }
        }

        for chunk in dst.chunks_exact_mut(channels).skip(x / channels) {
            let r = chunk[0];
            let g = chunk[1];
            let b = chunk[2];

            chunk[0] = mlaf(
                0.5f32,
                mlaf(
                    mlaf(r * transform.v[0][0], g, transform.v[0][1]),
                    b,
                    transform.v[0][2],
                )
                .max(0f32)
                .min(1f32),
                scale,
            );

            chunk[1] = mlaf(
                0.5f32,
                mlaf(
                    mlaf(r * transform.v[1][0], g, transform.v[1][1]),
                    b,
                    transform.v[1][2],
                )
                .max(0f32)
                .min(1f32),
                scale,
            );

            chunk[2] = mlaf(
                0.5f32,
                mlaf(
                    mlaf(r * transform.v[2][0], g, transform.v[2][1]),
                    b,
                    transform.v[2][2],
                )
                .max(0f32)
                .min(1f32),
                scale,
            )
        }

        Ok(())
    }
}
