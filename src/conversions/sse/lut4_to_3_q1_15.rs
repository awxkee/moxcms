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
use crate::conversions::LutBarycentricReduction;
use crate::conversions::interpolator::BarycentricWeight;
use crate::conversions::sse::interpolator_q1_15::{
    PrismaticSseQ1_15, PyramidalSseQ1_15, SseAlignedI16x4, SseMdInterpolationQ1_15,
    TetrahedralSseQ1_15, TrilinearSseQ1_15,
};
use crate::transform::PointeeSizeExpressible;
use crate::{CmsError, InterpolationMethod, Layout, TransformExecutor};
use num_traits::AsPrimitive;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::marker::PhantomData;

pub(crate) struct TransformLut4XyzToRgbSseQ1_15<
    T,
    U,
    const LAYOUT: u8,
    const GRID_SIZE: usize,
    const BIT_DEPTH: usize,
    const BINS: usize,
    const BARYCENTRIC_BINS: usize,
> {
    pub(crate) lut: Vec<SseAlignedI16x4>,
    pub(crate) _phantom: PhantomData<T>,
    pub(crate) _phantom1: PhantomData<U>,
    pub(crate) interpolation_method: InterpolationMethod,
    pub(crate) weights: Box<[BarycentricWeight<i16>; BINS]>,
}

impl<
    T: Copy + AsPrimitive<f32> + Default + PointeeSizeExpressible,
    U: AsPrimitive<usize>,
    const LAYOUT: u8,
    const GRID_SIZE: usize,
    const BIT_DEPTH: usize,
    const BINS: usize,
    const BARYCENTRIC_BINS: usize,
> TransformLut4XyzToRgbSseQ1_15<T, U, LAYOUT, GRID_SIZE, BIT_DEPTH, BINS, BARYCENTRIC_BINS>
where
    f32: AsPrimitive<T>,
    u32: AsPrimitive<T>,
    (): LutBarycentricReduction<T, U>,
{
    #[allow(unused_unsafe)]
    #[target_feature(enable = "sse4.1")]
    unsafe fn transform_chunk<'b, Interpolator: SseMdInterpolationQ1_15<'b, GRID_SIZE>>(
        &'b self,
        src: &[T],
        dst: &mut [T],
    ) {
        unsafe {
            let cn = Layout::from(LAYOUT);
            let channels = cn.channels();
            let grid_size = GRID_SIZE as i32;
            let grid_size3 = grid_size * grid_size * grid_size;

            let value_scale = _mm_set1_ps(1. / (1 << 15) as f32);
            let max_value = ((1u32 << BIT_DEPTH) - 1).as_();
            let v_max = _mm_set1_epi16(((1u32 << BIT_DEPTH) - 1) as i16);
            let rnd = if BIT_DEPTH == 12 {
                _mm_set1_epi16((1 << (3 - 1)) - 1)
            } else if BIT_DEPTH == 10 {
                _mm_set1_epi16((1 << (5 - 1)) - 1)
            } else {
                _mm_set1_epi16((1 << (7 - 1)) - 1)
            };

            for (src, dst) in src.chunks_exact(4).zip(dst.chunks_exact_mut(channels)) {
                let c = <() as LutBarycentricReduction<T, U>>::reduce::<BIT_DEPTH, BARYCENTRIC_BINS>(
                    src[0],
                );
                let m = <() as LutBarycentricReduction<T, U>>::reduce::<BIT_DEPTH, BARYCENTRIC_BINS>(
                    src[1],
                );
                let y = <() as LutBarycentricReduction<T, U>>::reduce::<BIT_DEPTH, BARYCENTRIC_BINS>(
                    src[2],
                );
                let k = <() as LutBarycentricReduction<T, U>>::reduce::<BIT_DEPTH, BARYCENTRIC_BINS>(
                    src[3],
                );

                let k_weights = self.weights[k.as_()];

                let w: i32 = k_weights.x;
                let w_n: i32 = k_weights.x_n;
                let t: i16 = k_weights.w;

                let t0 = _mm_set1_epi16(t);

                let table1 = &self.lut[(w * grid_size3) as usize..];
                let table2 = &self.lut[(w_n * grid_size3) as usize..];

                let tetrahedral1 = Interpolator::new(table1);
                let tetrahedral2 = Interpolator::new(table2);
                let a0 = tetrahedral1.inter3_sse(c, m, y, &self.weights).v;
                let b0 = tetrahedral2.inter3_sse(c, m, y, &self.weights).v;

                let j0 = _mm_mulhrs_epi16(t0, a0);
                let j1 = _mm_mulhrs_epi16(b0, t0);

                let hp = _mm_sub_epi16(a0, j0);
                let v = _mm_add_epi16(j1, hp);

                if T::FINITE {
                    if BIT_DEPTH == 8 {
                        let mut r = _mm_srai_epi16::<7>(_mm_adds_epi16(v, rnd));
                        r = _mm_packus_epi16(r, r);

                        let m_once = _mm_cvtsi128_si32(r).to_ne_bytes();
                        let x = m_once[0];
                        let y = m_once[1];
                        let z = m_once[2];

                        dst[cn.r_i()] = (x as u32).as_();
                        dst[cn.g_i()] = (y as u32).as_();
                        dst[cn.b_i()] = (z as u32).as_();
                    } else {
                        let mut r = if BIT_DEPTH == 12 {
                            _mm_srai_epi16::<3>(_mm_adds_epi16(v, rnd))
                        } else if BIT_DEPTH == 10 {
                            _mm_srai_epi16::<5>(_mm_adds_epi16(v, rnd))
                        } else {
                            _mm_srai_epi16::<7>(_mm_adds_epi16(v, rnd))
                        };
                        r = _mm_max_epi16(r, _mm_setzero_si128());
                        r = _mm_min_epi16(r, v_max);

                        let x = _mm_extract_epi16::<0>(r);
                        let y = _mm_extract_epi16::<1>(r);
                        let z = _mm_extract_epi16::<2>(r);

                        dst[cn.r_i()] = (x as u32).as_();
                        dst[cn.g_i()] = (y as u32).as_();
                        dst[cn.b_i()] = (z as u32).as_();
                    }
                } else {
                    let mut o = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v, _mm_setzero_si128()));
                    o = _mm_mul_ps(o, value_scale);
                    o = _mm_min_ps(o, value_scale);
                    o = _mm_max_ps(o, _mm_set1_ps(1.0));
                    dst[cn.r_i()] = f32::from_bits(_mm_extract_ps::<0>(o) as u32).as_();
                    dst[cn.g_i()] = f32::from_bits(_mm_extract_ps::<1>(o) as u32).as_();
                    dst[cn.b_i()] = f32::from_bits(_mm_extract_ps::<2>(o) as u32).as_();
                }
                if channels == 4 {
                    dst[cn.a_i()] = max_value;
                }
            }
        }
    }
}

impl<
    T: Copy + AsPrimitive<f32> + Default + PointeeSizeExpressible,
    U: AsPrimitive<usize>,
    const LAYOUT: u8,
    const GRID_SIZE: usize,
    const BIT_DEPTH: usize,
    const BINS: usize,
    const BARYCENTRIC_BINS: usize,
> TransformExecutor<T>
    for TransformLut4XyzToRgbSseQ1_15<T, U, LAYOUT, GRID_SIZE, BIT_DEPTH, BINS, BARYCENTRIC_BINS>
where
    f32: AsPrimitive<T>,
    u32: AsPrimitive<T>,
    (): LutBarycentricReduction<T, U>,
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
            match self.interpolation_method {
                InterpolationMethod::Tetrahedral => {
                    self.transform_chunk::<TetrahedralSseQ1_15<GRID_SIZE>>(src, dst);
                }
                InterpolationMethod::Pyramid => {
                    self.transform_chunk::<PyramidalSseQ1_15<GRID_SIZE>>(src, dst);
                }
                InterpolationMethod::Prism => {
                    self.transform_chunk::<PrismaticSseQ1_15<GRID_SIZE>>(src, dst);
                }
                InterpolationMethod::Linear => {
                    self.transform_chunk::<TrilinearSseQ1_15<GRID_SIZE>>(src, dst);
                }
            }
        }

        Ok(())
    }
}
