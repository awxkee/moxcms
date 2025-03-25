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
use crate::conversions::sse::interpolator_q1_15::*;
use crate::transform::PointeeSizeExpressible;
use crate::{CmsError, InterpolationMethod, Layout, TransformExecutor};
use num_traits::AsPrimitive;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::marker::PhantomData;

pub(crate) struct TransformLut3x3SseQ1_15<
    T,
    U,
    const SRC_LAYOUT: u8,
    const DST_LAYOUT: u8,
    const GRID_SIZE: usize,
    const BIT_DEPTH: usize,
    const BINS: usize,
    const BARYCENTRIC_BINS: usize,
> {
    pub(crate) lut: Vec<SseAlignedI16x4>,
    pub(crate) _phantom: PhantomData<T>,
    pub(crate) _phantom2: PhantomData<U>,
    pub(crate) interpolation_method: InterpolationMethod,
    pub(crate) weights: Box<[BarycentricWeight<i16>; BINS]>,
}

impl<
    T: Copy + AsPrimitive<f32> + Default + PointeeSizeExpressible,
    U: AsPrimitive<usize>,
    const SRC_LAYOUT: u8,
    const DST_LAYOUT: u8,
    const GRID_SIZE: usize,
    const BIT_DEPTH: usize,
    const BINS: usize,
    const BARYCENTRIC_BINS: usize,
>
    TransformLut3x3SseQ1_15<
        T,
        U,
        SRC_LAYOUT,
        DST_LAYOUT,
        GRID_SIZE,
        BIT_DEPTH,
        BINS,
        BARYCENTRIC_BINS,
    >
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
            let src_cn = Layout::from(SRC_LAYOUT);
            let src_channels = src_cn.channels();

            let dst_cn = Layout::from(DST_LAYOUT);
            let dst_channels = dst_cn.channels();

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

            for (src, dst) in src
                .chunks_exact(src_channels)
                .zip(dst.chunks_exact_mut(dst_channels))
            {
                let x = <() as LutBarycentricReduction<T, U>>::reduce::<BIT_DEPTH, BARYCENTRIC_BINS>(
                    src[src_cn.r_i()],
                );
                let y = <() as LutBarycentricReduction<T, U>>::reduce::<BIT_DEPTH, BARYCENTRIC_BINS>(
                    src[src_cn.g_i()],
                );
                let z = <() as LutBarycentricReduction<T, U>>::reduce::<BIT_DEPTH, BARYCENTRIC_BINS>(
                    src[src_cn.b_i()],
                );

                let a = if src_channels == 4 {
                    src[src_cn.a_i()]
                } else {
                    max_value
                };

                let tetrahedral = Interpolator::new(&self.lut);
                let v = tetrahedral.inter3_sse(x, y, z, &self.weights);
                if T::FINITE {
                    if BIT_DEPTH == 8 {
                        let mut r = _mm_srai_epi16::<7>(_mm_adds_epi16(v.v, rnd));
                        r = _mm_packus_epi16(r, r);

                        let m_once = _mm_cvtsi128_si32(r).to_ne_bytes();
                        let x = m_once[0];
                        let y = m_once[1];
                        let z = m_once[2];

                        dst[dst_cn.r_i()] = (x as u32).as_();
                        dst[dst_cn.g_i()] = (y as u32).as_();
                        dst[dst_cn.b_i()] = (z as u32).as_();
                    } else {
                        let mut r = if BIT_DEPTH == 12 {
                            _mm_srai_epi16::<3>(_mm_adds_epi16(v.v, rnd))
                        } else if BIT_DEPTH == 10 {
                            _mm_srai_epi16::<5>(_mm_adds_epi16(v.v, rnd))
                        } else {
                            _mm_srai_epi16::<7>(_mm_adds_epi16(v.v, rnd))
                        };
                        r = _mm_max_epi16(r, _mm_setzero_si128());
                        r = _mm_min_epi16(r, v_max);

                        let x = _mm_extract_epi16::<0>(r);
                        let y = _mm_extract_epi16::<1>(r);
                        let z = _mm_extract_epi16::<2>(r);

                        dst[dst_cn.r_i()] = (x as u32).as_();
                        dst[dst_cn.g_i()] = (y as u32).as_();
                        dst[dst_cn.b_i()] = (z as u32).as_();
                    }
                } else {
                    let mut o = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v.v, _mm_setzero_si128()));
                    o = _mm_mul_ps(o, value_scale);
                    o = _mm_min_ps(o, value_scale);
                    o = _mm_max_ps(o, _mm_set1_ps(1.0));
                    dst[dst_cn.r_i()] = f32::from_bits(_mm_extract_ps::<0>(o) as u32).as_();
                    dst[dst_cn.g_i()] = f32::from_bits(_mm_extract_ps::<1>(o) as u32).as_();
                    dst[dst_cn.b_i()] = f32::from_bits(_mm_extract_ps::<2>(o) as u32).as_();
                }
                if dst_channels == 4 {
                    dst[dst_cn.a_i()] = a;
                }
            }
        }
    }
}

impl<
    T: Copy + AsPrimitive<f32> + Default + PointeeSizeExpressible,
    U: AsPrimitive<usize>,
    const SRC_LAYOUT: u8,
    const DST_LAYOUT: u8,
    const GRID_SIZE: usize,
    const BIT_DEPTH: usize,
    const BINS: usize,
    const BARYCENTRIC_BINS: usize,
> TransformExecutor<T>
    for TransformLut3x3SseQ1_15<
        T,
        U,
        SRC_LAYOUT,
        DST_LAYOUT,
        GRID_SIZE,
        BIT_DEPTH,
        BINS,
        BARYCENTRIC_BINS,
    >
where
    f32: AsPrimitive<T>,
    u32: AsPrimitive<T>,
    (): LutBarycentricReduction<T, U>,
{
    fn transform(&self, src: &[T], dst: &mut [T]) -> Result<(), CmsError> {
        let src_cn = Layout::from(SRC_LAYOUT);
        let src_channels = src_cn.channels();

        let dst_cn = Layout::from(DST_LAYOUT);
        let dst_channels = dst_cn.channels();
        if src.len() % src_channels != 0 {
            return Err(CmsError::LaneMultipleOfChannels);
        }
        if dst.len() % dst_channels != 0 {
            return Err(CmsError::LaneMultipleOfChannels);
        }
        let src_chunks = src.len() / src_channels;
        let dst_chunks = dst.len() / dst_channels;
        if src_chunks != dst_chunks {
            return Err(CmsError::LaneSizeMismatch);
        }

        unsafe {
            match self.interpolation_method {
                #[cfg(feature = "options")]
                InterpolationMethod::Tetrahedral => {
                    self.transform_chunk::<TetrahedralSseQ1_15<GRID_SIZE>>(src, dst);
                }
                #[cfg(feature = "options")]
                InterpolationMethod::Pyramid => {
                    self.transform_chunk::<PyramidalSseQ1_15<GRID_SIZE>>(src, dst);
                }
                #[cfg(feature = "options")]
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
