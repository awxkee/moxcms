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
use crate::conversions::avx::pack::{
    _mm256_deinterleave_rgb_ps, _mm256_deinterleave_rgba_ps, _mm256_interleave_rgb_ps,
    _mm256_interleave_rgba_ps,
};
use crate::conversions::avx::util::_mm256_opt_fmlaf_ps;
use crate::mlaf::mlaf;
use crate::{CmsError, InPlaceStage, Layout, Matrix3f};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) struct MatrixClipScaleStageAvx<const LAYOUT: u8> {
    pub(crate) matrix: Matrix3f,
    pub(crate) scale: f32,
}

pub(crate) struct MatrixClipScaleStageAvxFma<const LAYOUT: u8> {
    pub(crate) matrix: Matrix3f,
    pub(crate) scale: f32,
}

#[inline(always)]
fn transform_executor<const LAYOUT: u8, const FMA: bool>(
    dst: &mut [f32],
    scale: f32,
    matrix: Matrix3f,
) {
    let cn = Layout::from(LAYOUT);
    let channels = cn.channels();

    unsafe {
        let m0 = _mm256_broadcast_ss(&matrix.v[0][0]);
        let m1 = _mm256_broadcast_ss(&matrix.v[0][1]);
        let m2 = _mm256_broadcast_ss(&matrix.v[0][2]);

        let m3 = _mm256_broadcast_ss(&matrix.v[1][0]);
        let m4 = _mm256_broadcast_ss(&matrix.v[1][1]);
        let m5 = _mm256_broadcast_ss(&matrix.v[1][2]);

        let m6 = _mm256_broadcast_ss(&matrix.v[2][0]);
        let m7 = _mm256_broadcast_ss(&matrix.v[2][1]);
        let m8 = _mm256_broadcast_ss(&matrix.v[2][2]);

        let v_scale = _mm256_set1_ps(scale);

        let mut x = 0usize;
        let total_width = dst.len();

        while x + 8 * channels < total_width {
            let chunk = dst.get_unchecked_mut(x..);

            let x0 = _mm256_loadu_ps(chunk.as_ptr());
            let x1 = _mm256_loadu_ps(chunk.get_unchecked(8..).as_ptr());
            let x2 = _mm256_loadu_ps(chunk.get_unchecked(16..).as_ptr());
            let (r, g, b, a) = if channels == 3 {
                let xyz = _mm256_deinterleave_rgb_ps(x0, x1, x2);
                (xyz.0, xyz.1, xyz.2, _mm256_setzero_ps())
            } else {
                let x3 = _mm256_loadu_ps(chunk.get_unchecked(24..).as_ptr());
                _mm256_deinterleave_rgba_ps(x0, x1, x2, x3)
            };

            let mut new_r = _mm256_mul_ps(r, m0);
            let mut new_g = _mm256_mul_ps(r, m3);
            let mut new_b = _mm256_mul_ps(r, m6);

            new_r = _mm256_opt_fmlaf_ps::<FMA>(new_r, g, m1);
            new_g = _mm256_opt_fmlaf_ps::<FMA>(new_g, g, m4);
            new_b = _mm256_opt_fmlaf_ps::<FMA>(new_b, g, m7);

            new_r = _mm256_opt_fmlaf_ps::<FMA>(new_r, b, m2);
            new_g = _mm256_opt_fmlaf_ps::<FMA>(new_g, b, m5);
            new_b = _mm256_opt_fmlaf_ps::<FMA>(new_b, b, m8);

            new_r = _mm256_max_ps(new_r, _mm256_setzero_ps());
            new_g = _mm256_max_ps(new_g, _mm256_setzero_ps());
            new_b = _mm256_max_ps(new_b, _mm256_setzero_ps());

            new_r = _mm256_opt_fmlaf_ps::<FMA>(_mm256_set1_ps(0.5f32), new_r, v_scale);
            new_g = _mm256_opt_fmlaf_ps::<FMA>(_mm256_set1_ps(0.5f32), new_g, v_scale);
            new_b = _mm256_opt_fmlaf_ps::<FMA>(_mm256_set1_ps(0.5f32), new_b, v_scale);

            new_r = _mm256_min_ps(new_r, v_scale);
            new_g = _mm256_min_ps(new_g, v_scale);
            new_b = _mm256_min_ps(new_b, v_scale);

            if channels == 3 {
                let xyz = _mm256_interleave_rgb_ps(new_r, new_g, new_b);
                _mm256_storeu_ps(chunk.as_mut_ptr(), xyz.0);
                _mm256_storeu_ps(chunk.get_unchecked_mut(8..).as_mut_ptr(), xyz.1);
                _mm256_storeu_ps(chunk.get_unchecked_mut(16..).as_mut_ptr(), xyz.2);
            } else if channels == 4 {
                let xyz = _mm256_interleave_rgba_ps(new_r, new_g, new_b, a);
                _mm256_storeu_ps(chunk.as_mut_ptr(), xyz.0);
                _mm256_storeu_ps(chunk.get_unchecked_mut(8..).as_mut_ptr(), xyz.1);
                _mm256_storeu_ps(chunk.get_unchecked_mut(16..).as_mut_ptr(), xyz.2);
                _mm256_storeu_ps(chunk.get_unchecked_mut(24..).as_mut_ptr(), xyz.3);
            }

            x += 8 * channels;
        }
    }

    let rem = dst.chunks_exact_mut(channels * 8).into_remainder();

    for chunk in rem.chunks_exact_mut(channels) {
        let r = chunk[0];
        let g = chunk[1];
        let b = chunk[2];

        chunk[0] = mlaf(
            0.5f32,
            mlaf(
                mlaf(r * matrix.v[0][0], g, matrix.v[0][1]),
                b,
                matrix.v[0][2],
            )
            .max(0f32)
            .min(1f32),
            scale,
        );

        chunk[1] = mlaf(
            0.5f32,
            mlaf(
                mlaf(r * matrix.v[1][0], g, matrix.v[1][1]),
                b,
                matrix.v[1][2],
            )
            .max(0f32)
            .min(1f32),
            scale,
        );

        chunk[2] = mlaf(
            0.5f32,
            mlaf(
                mlaf(r * matrix.v[2][0], g, matrix.v[2][1]),
                b,
                matrix.v[2][2],
            )
            .max(0f32)
            .min(1f32),
            scale,
        )
    }
}

impl<const LAYOUT: u8> MatrixClipScaleStageAvx<LAYOUT> {
    #[target_feature(enable = "avx2")]
    unsafe fn transform_call(&self, dst: &mut [f32]) {
        transform_executor::<LAYOUT, false>(dst, self.scale, self.matrix)
    }
}

impl<const LAYOUT: u8> InPlaceStage for MatrixClipScaleStageAvx<LAYOUT> {
    #[inline]
    fn transform(&self, dst: &mut [f32]) -> Result<(), CmsError> {
        unsafe { self.transform_call(dst) }
        Ok(())
    }
}

impl<const LAYOUT: u8> MatrixClipScaleStageAvxFma<LAYOUT> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn transform_call(&self, dst: &mut [f32]) {
        transform_executor::<LAYOUT, true>(dst, self.scale, self.matrix)
    }
}

impl<const LAYOUT: u8> InPlaceStage for MatrixClipScaleStageAvxFma<LAYOUT> {
    #[inline]
    fn transform(&self, dst: &mut [f32]) -> Result<(), CmsError> {
        unsafe { self.transform_call(dst) }
        Ok(())
    }
}
