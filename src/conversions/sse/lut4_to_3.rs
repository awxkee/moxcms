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
use crate::conversions::CompressForLut;
use crate::conversions::sse::TetrahedralSse;
use crate::conversions::sse::transform_lut3_to_3::SseAlignedU32;
use crate::conversions::tetrahedral::TetrhedralInterpolation;
use crate::{CmsError, Layout, TransformExecutor, rounding_div_ceil};
use num_traits::AsPrimitive;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::marker::PhantomData;

pub(crate) struct TransformLut4XyzToRgbSse<
    T,
    const LAYOUT: u8,
    const GRID_SIZE: usize,
    const BIT_DEPTH: usize,
> {
    pub(crate) lut: Vec<f32>,
    pub(crate) _phantom: PhantomData<T>,
}

impl<
    T: Copy + AsPrimitive<f32> + Default + CompressForLut,
    const LAYOUT: u8,
    const GRID_SIZE: usize,
    const BIT_DEPTH: usize,
> TransformLut4XyzToRgbSse<T, LAYOUT, GRID_SIZE, BIT_DEPTH>
where
    f32: AsPrimitive<T>,
    u32: AsPrimitive<T>,
{
    #[allow(unused_unsafe)]
    #[target_feature(enable = "sse4.1")]
    unsafe fn transform_chunk(&self, src: &[T], dst: &mut [T]) {
        let cn = Layout::from(LAYOUT);
        let channels = cn.channels();
        let grid_size = GRID_SIZE as i32;
        let grid_size3 = grid_size * grid_size * grid_size;

        let value_scale = unsafe { _mm_set1_ps(((1 << BIT_DEPTH) - 1) as f32) };
        let max_value = ((1 << BIT_DEPTH) - 1u32).as_();

        let mut temporary0 = SseAlignedU32([0; 4]);

        for (src, dst) in src.chunks_exact(4).zip(dst.chunks_exact_mut(channels)) {
            let c = src[0].compress_lut::<BIT_DEPTH>();
            let m = src[1].compress_lut::<BIT_DEPTH>();
            let y = src[2].compress_lut::<BIT_DEPTH>();
            let k = src[3].compress_lut::<BIT_DEPTH>();
            let linear_k: f32 = k as i32 as f32 / 255.0;
            let w: i32 = k as i32 * (GRID_SIZE as i32 - 1) / 255;
            let w_n: i32 = rounding_div_ceil(k as i32 * (GRID_SIZE as i32 - 1), 255);
            let t: f32 = linear_k * (GRID_SIZE as i32 - 1) as f32 - w as f32;

            let table1 = &self.lut[(w * grid_size3 * 3) as usize..];
            let table2 = &self.lut[(w_n * grid_size3 * 3) as usize..];

            let tetrahedral1 = TetrahedralSse::<GRID_SIZE>::new(table1);
            let tetrahedral2 = TetrahedralSse::<GRID_SIZE>::new(table2);
            let a0 = tetrahedral1.inter3_sse(c, m, y).v;
            let b0 = tetrahedral2.inter3_sse(c, m, y).v;

            unsafe {
                let t0 = _mm_set1_ps(t);
                let ones = _mm_set1_ps(1f32);
                let hp = _mm_mul_ps(a0, _mm_sub_ps(ones, t0));
                let mut v = _mm_add_ps(_mm_mul_ps(b0, t0), hp);
                v = _mm_max_ps(v, _mm_setzero_ps());
                v = _mm_mul_ps(v, value_scale);
                v = _mm_min_ps(v, value_scale);
                _mm_store_si128(temporary0.0.as_mut_ptr() as *mut _, _mm_cvtps_epi32(v));
            }

            dst[cn.r_i()] = temporary0.0[0].as_();
            dst[cn.g_i()] = temporary0.0[1].as_();
            dst[cn.b_i()] = temporary0.0[2].as_();
            if channels == 4 {
                dst[cn.a_i()] = max_value;
            }
        }
    }
}

impl<
    T: Copy + AsPrimitive<f32> + Default + CompressForLut,
    const LAYOUT: u8,
    const GRID_SIZE: usize,
    const BIT_DEPTH: usize,
> TransformExecutor<T> for TransformLut4XyzToRgbSse<T, LAYOUT, GRID_SIZE, BIT_DEPTH>
where
    f32: AsPrimitive<T>,
    u32: AsPrimitive<T>,
{
    fn transform(&self, src: &[T], dst: &mut [T]) -> Result<(), CmsError> {
        let cn = Layout::from(LAYOUT);
        let channels = cn.channels();
        if src.len() % 4 != 0 {
            return Err(CmsError::LaneMultipleOfChannels);
        }
        if dst.len() % channels != 0 {
            return Err(CmsError::LaneMultipleOfChannels);
        }
        let src_chunks = src.len() / 4;
        let dst_chunks = dst.len() / channels;
        if src_chunks != dst_chunks {
            return Err(CmsError::LaneSizeMismatch);
        }

        unsafe {
            self.transform_chunk(src, dst);
        }

        Ok(())
    }
}
