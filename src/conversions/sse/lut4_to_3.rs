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
use crate::conversions::interpolator::{BarycentricWeight, BarycentricWeightQ1_15};
use crate::conversions::lut_transforms::Lut4x3Factory;
use crate::conversions::sse::TetrahedralSse;
use crate::conversions::sse::interpolator::{
    PrismaticSse, PyramidalSse, SseAlignedF32, SseMdInterpolation, TrilinearSse,
};
use crate::conversions::sse::interpolator_q1_15::SseAlignedI16x4;
use crate::conversions::sse::lut4_to_3_q1_15::TransformLut4XyzToRgbSseQ1_15;
use crate::transform::PointeeSizeExpressible;
use crate::{
    BarycentricWeightScale, CmsError, InterpolationMethod, Layout, TransformExecutor,
    TransformOptions,
};
use num_traits::AsPrimitive;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::marker::PhantomData;

struct TransformLut4XyzToRgbSse<
    T,
    U,
    const LAYOUT: u8,
    const GRID_SIZE: usize,
    const BIT_DEPTH: usize,
    const BINS: usize,
    const BARYCENTRIC_BINS: usize,
> {
    lut: Vec<SseAlignedF32>,
    _phantom: PhantomData<T>,
    _phantom1: PhantomData<U>,
    interpolation_method: InterpolationMethod,
    weights: Box<[BarycentricWeight; BINS]>,
}

impl<
    T: Copy + AsPrimitive<f32> + Default + PointeeSizeExpressible,
    U: AsPrimitive<usize>,
    const LAYOUT: u8,
    const GRID_SIZE: usize,
    const BIT_DEPTH: usize,
    const BINS: usize,
    const BARYCENTRIC_BINS: usize,
> TransformLut4XyzToRgbSse<T, U, LAYOUT, GRID_SIZE, BIT_DEPTH, BINS, BARYCENTRIC_BINS>
where
    f32: AsPrimitive<T>,
    u32: AsPrimitive<T>,
    (): LutBarycentricReduction<T, U>,
{
    #[allow(unused_unsafe)]
    #[target_feature(enable = "sse4.1")]
    unsafe fn transform_chunk<'b, Interpolator: SseMdInterpolation<'b, GRID_SIZE>>(
        &'b self,
        src: &[T],
        dst: &mut [T],
    ) {
        let cn = Layout::from(LAYOUT);
        let channels = cn.channels();
        let grid_size = GRID_SIZE as i32;
        let grid_size3 = grid_size * grid_size * grid_size;

        let value_scale = unsafe { _mm_set1_ps(((1 << BIT_DEPTH) - 1) as f32) };
        let max_value = ((1 << BIT_DEPTH) - 1u32).as_();

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
            let t: f32 = k_weights.w;

            let table1 = &self.lut[(w * grid_size3) as usize..];
            let table2 = &self.lut[(w_n * grid_size3) as usize..];

            let tetrahedral1 = Interpolator::new(table1);
            let tetrahedral2 = Interpolator::new(table2);
            let a0 = tetrahedral1.inter3_sse(c, m, y, &self.weights).v;
            let b0 = tetrahedral2.inter3_sse(c, m, y, &self.weights).v;

            if T::FINITE {
                unsafe {
                    let t0 = _mm_set1_ps(t);
                    let ones = _mm_set1_ps(1f32);
                    let hp = _mm_mul_ps(a0, _mm_sub_ps(ones, t0));
                    let mut v = _mm_add_ps(_mm_mul_ps(b0, t0), hp);
                    v = _mm_max_ps(v, _mm_setzero_ps());
                    v = _mm_mul_ps(v, value_scale);
                    v = _mm_min_ps(v, value_scale);
                    let jvz = _mm_cvtps_epi32(v);

                    let x = _mm_extract_epi32::<0>(jvz);
                    let y = _mm_extract_epi32::<1>(jvz);
                    let z = _mm_extract_epi32::<2>(jvz);

                    dst[cn.r_i()] = (x as u32).as_();
                    dst[cn.g_i()] = (y as u32).as_();
                    dst[cn.b_i()] = (z as u32).as_();
                }
            } else {
                unsafe {
                    let t0 = _mm_set1_ps(t);
                    let ones = _mm_set1_ps(1f32);
                    let hp = _mm_mul_ps(a0, _mm_sub_ps(ones, t0));
                    let mut v = _mm_add_ps(_mm_mul_ps(b0, t0), hp);
                    v = _mm_max_ps(v, _mm_setzero_ps());
                    v = _mm_min_ps(v, value_scale);

                    dst[cn.r_i()] = f32::from_bits(_mm_extract_ps::<0>(v) as u32).as_();
                    dst[cn.g_i()] = f32::from_bits(_mm_extract_ps::<1>(v) as u32).as_();
                    dst[cn.b_i()] = f32::from_bits(_mm_extract_ps::<2>(v) as u32).as_();
                }
            }
            if channels == 4 {
                dst[cn.a_i()] = max_value;
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
    for TransformLut4XyzToRgbSse<T, U, LAYOUT, GRID_SIZE, BIT_DEPTH, BINS, BARYCENTRIC_BINS>
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
                    self.transform_chunk::<TetrahedralSse<GRID_SIZE>>(src, dst);
                }
                InterpolationMethod::Pyramid => {
                    self.transform_chunk::<PyramidalSse<GRID_SIZE>>(src, dst);
                }
                InterpolationMethod::Prism => {
                    self.transform_chunk::<PrismaticSse<GRID_SIZE>>(src, dst);
                }
                InterpolationMethod::Linear => {
                    self.transform_chunk::<TrilinearSse<GRID_SIZE>>(src, dst);
                }
            }
        }

        Ok(())
    }
}

pub(crate) struct SseLut4x3Factory {}

impl Lut4x3Factory for SseLut4x3Factory {
    fn make_transform_4x3<
        T: Copy + AsPrimitive<f32> + Default + PointeeSizeExpressible + 'static + Send + Sync,
        const LAYOUT: u8,
        const GRID_SIZE: usize,
        const BIT_DEPTH: usize,
    >(
        lut: Vec<f32>,
        options: TransformOptions,
    ) -> Box<dyn TransformExecutor<T> + Sync + Send>
    where
        f32: AsPrimitive<T>,
        u32: AsPrimitive<T>,
        (): LutBarycentricReduction<T, u8>,
        (): LutBarycentricReduction<T, u16>,
    {
        if options.prefer_fixed_point && BIT_DEPTH < 15 {
            const Q_SCALE: f32 = ((1 << 15) - 1) as f32;
            let lut = lut
                .chunks_exact(3)
                .map(|x| {
                    SseAlignedI16x4([
                        (x[0] * Q_SCALE).round() as i16,
                        (x[1] * Q_SCALE).round() as i16,
                        (x[2] * Q_SCALE).round() as i16,
                        0,
                    ])
                })
                .collect::<Vec<_>>();
            return match options.barycentric_weight_scale {
                BarycentricWeightScale::Low => Box::new(TransformLut4XyzToRgbSseQ1_15::<
                    T,
                    u8,
                    LAYOUT,
                    GRID_SIZE,
                    BIT_DEPTH,
                    256,
                    256,
                > {
                    lut,
                    _phantom: PhantomData,
                    _phantom1: PhantomData,
                    interpolation_method: options.interpolation_method,
                    weights: BarycentricWeightQ1_15::create_ranged_256::<GRID_SIZE>(),
                }),
                BarycentricWeightScale::High => Box::new(TransformLut4XyzToRgbSseQ1_15::<
                    T,
                    u16,
                    LAYOUT,
                    GRID_SIZE,
                    BIT_DEPTH,
                    65536,
                    65536,
                > {
                    lut,
                    _phantom: PhantomData,
                    _phantom1: PhantomData,
                    interpolation_method: options.interpolation_method,
                    weights: BarycentricWeightQ1_15::create_binned::<GRID_SIZE, 65536>(),
                }),
            };
        }
        let lut = lut
            .chunks_exact(3)
            .map(|x| SseAlignedF32([x[0], x[1], x[2], 0f32]))
            .collect::<Vec<_>>();
        match options.barycentric_weight_scale {
            BarycentricWeightScale::Low => {
                Box::new(
                    TransformLut4XyzToRgbSse::<T, u8, LAYOUT, GRID_SIZE, BIT_DEPTH, 256, 256> {
                        lut,
                        _phantom: PhantomData,
                        _phantom1: PhantomData,
                        interpolation_method: options.interpolation_method,
                        weights: BarycentricWeight::create_ranged_256::<GRID_SIZE>(),
                    },
                )
            }
            BarycentricWeightScale::High => Box::new(TransformLut4XyzToRgbSse::<
                T,
                u16,
                LAYOUT,
                GRID_SIZE,
                BIT_DEPTH,
                65536,
                65536,
            > {
                lut,
                _phantom: PhantomData,
                _phantom1: PhantomData,
                interpolation_method: options.interpolation_method,
                weights: BarycentricWeight::create_binned::<GRID_SIZE, 65536>(),
            }),
        }
    }
}
