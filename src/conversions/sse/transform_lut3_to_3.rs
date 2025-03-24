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
use crate::conversions::interpolator::BarycentricWeight;
use crate::conversions::lut_transforms::Lut3x3Factory;
use crate::conversions::sse::TetrahedralSse;
use crate::conversions::sse::interpolator::{
    PrismaticSse, PyramidalSse, SseAlignedF32, SseMdInterpolation, TrilinearSse,
};
use crate::conversions::{CompressForLut, LutBarycentricReduction};
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

struct TransformLut3x3Sse<
    T,
    U,
    const SRC_LAYOUT: u8,
    const DST_LAYOUT: u8,
    const GRID_SIZE: usize,
    const BIT_DEPTH: usize,
    const BINS: usize,
    const BARYCENTRIC_BINS: usize,
> {
    lut: Vec<SseAlignedF32>,
    _phantom: PhantomData<T>,
    _phantom2: PhantomData<U>,
    interpolation_method: InterpolationMethod,
    weights: Box<[BarycentricWeight; BINS]>,
}

impl<
    T: Copy + AsPrimitive<f32> + Default + CompressForLut + PointeeSizeExpressible,
    U: AsPrimitive<usize>,
    const SRC_LAYOUT: u8,
    const DST_LAYOUT: u8,
    const GRID_SIZE: usize,
    const BIT_DEPTH: usize,
    const BINS: usize,
    const BARYCENTRIC_BINS: usize,
> TransformLut3x3Sse<T, U, SRC_LAYOUT, DST_LAYOUT, GRID_SIZE, BIT_DEPTH, BINS, BARYCENTRIC_BINS>
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
        let src_cn = Layout::from(SRC_LAYOUT);
        let src_channels = src_cn.channels();

        let dst_cn = Layout::from(DST_LAYOUT);
        let dst_channels = dst_cn.channels();

        let value_scale = unsafe { _mm_set1_ps(((1 << BIT_DEPTH) - 1) as f32) };
        let max_value = ((1u32 << BIT_DEPTH) - 1).as_();

        for (src, dst) in src
            .chunks_exact(src_channels)
            .zip(dst.chunks_exact_mut(dst_channels))
        {
            let x = src[src_cn.r_i()].compress_lut::<BIT_DEPTH>();
            let y = src[src_cn.g_i()].compress_lut::<BIT_DEPTH>();
            let z = src[src_cn.b_i()].compress_lut::<BIT_DEPTH>();

            let a = if src_channels == 4 {
                src[src_cn.a_i()]
            } else {
                max_value
            };

            let tetrahedral = Interpolator::new(&self.lut);
            let v = tetrahedral.inter3_sse(x, y, z, &self.weights);
            if T::FINITE {
                unsafe {
                    let mut r = _mm_mul_ps(v.v, value_scale);
                    r = _mm_max_ps(r, _mm_setzero_ps());
                    r = _mm_min_ps(r, value_scale);
                    let jvz = _mm_cvtps_epi32(r);

                    let x = _mm_extract_epi32::<0>(jvz);
                    let y = _mm_extract_epi32::<1>(jvz);
                    let z = _mm_extract_epi32::<2>(jvz);

                    dst[dst_cn.r_i()] = (x as u32).as_();
                    dst[dst_cn.g_i()] = (y as u32).as_();
                    dst[dst_cn.b_i()] = (z as u32).as_();
                }
            } else {
                unsafe {
                    let mut r = _mm_max_ps(v.v, _mm_setzero_ps());
                    r = _mm_min_ps(r, value_scale);
                    dst[dst_cn.r_i()] = f32::from_bits(_mm_extract_ps::<0>(r) as u32).as_();
                    dst[dst_cn.g_i()] = f32::from_bits(_mm_extract_ps::<1>(r) as u32).as_();
                    dst[dst_cn.b_i()] = f32::from_bits(_mm_extract_ps::<2>(r) as u32).as_();
                }
            }
            if dst_channels == 4 {
                dst[dst_cn.a_i()] = a;
            }
        }
    }
}

impl<
    T: Copy + AsPrimitive<f32> + Default + CompressForLut + PointeeSizeExpressible,
    U: AsPrimitive<usize>,
    const SRC_LAYOUT: u8,
    const DST_LAYOUT: u8,
    const GRID_SIZE: usize,
    const BIT_DEPTH: usize,
    const BINS: usize,
    const BARYCENTRIC_BINS: usize,
> TransformExecutor<T>
    for TransformLut3x3Sse<
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

pub(crate) struct SseLut3x3Factory {}

impl Lut3x3Factory for SseLut3x3Factory {
    fn make_transform_3x3<
        T: Copy
            + AsPrimitive<f32>
            + Default
            + CompressForLut
            + PointeeSizeExpressible
            + 'static
            + Send
            + Sync,
        const SRC_LAYOUT: u8,
        const DST_LAYOUT: u8,
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
        let lut = lut
            .chunks_exact(3)
            .map(|x| SseAlignedF32([x[0], x[1], x[2], 0f32]))
            .collect::<Vec<_>>();
        match options.barycentric_weight_scale {
            BarycentricWeightScale::Low => Box::new(TransformLut3x3Sse::<
                T,
                u8,
                SRC_LAYOUT,
                DST_LAYOUT,
                GRID_SIZE,
                BIT_DEPTH,
                256,
                256,
            > {
                lut,
                _phantom: PhantomData,
                _phantom2: PhantomData,
                interpolation_method: options.interpolation_method,
                weights: BarycentricWeight::create_ranged_256::<GRID_SIZE>(),
            }),
            BarycentricWeightScale::Medium => Box::new(TransformLut3x3Sse::<
                T,
                u16,
                SRC_LAYOUT,
                DST_LAYOUT,
                GRID_SIZE,
                BIT_DEPTH,
                65536,
                16384,
            > {
                lut,
                _phantom: PhantomData,
                _phantom2: PhantomData,
                interpolation_method: options.interpolation_method,
                weights: BarycentricWeight::create_binned::<GRID_SIZE, 16384>(),
            }),
            BarycentricWeightScale::High => Box::new(TransformLut3x3Sse::<
                T,
                u16,
                SRC_LAYOUT,
                DST_LAYOUT,
                GRID_SIZE,
                BIT_DEPTH,
                65536,
                65536,
            > {
                lut,
                _phantom: PhantomData,
                _phantom2: PhantomData,
                interpolation_method: options.interpolation_method,
                weights: BarycentricWeight::create_binned::<GRID_SIZE, 65536>(),
            }),
        }
    }
}
