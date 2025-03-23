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
#![allow(dead_code)]
use crate::conversions::CompressForLut;
use crate::conversions::interpolator::MultidimensionalInterpolation;
use crate::conversions::lut_transforms::Lut3x3Factory;
use crate::transform::PointeeSizeExpressible;
use crate::{CmsError, InterpolationMethod, Layout, TransformExecutor};
use num_traits::AsPrimitive;
use std::marker::PhantomData;

pub(crate) struct TransformLut3x3<
    T,
    const SRC_LAYOUT: u8,
    const DST_LAYOUT: u8,
    const GRID_SIZE: usize,
    const BIT_DEPTH: usize,
> {
    pub(crate) lut: Vec<f32>,
    pub(crate) _phantom: PhantomData<T>,
    pub(crate) interpolation_method: InterpolationMethod,
}

impl<
    T: Copy + AsPrimitive<f32> + Default + CompressForLut + PointeeSizeExpressible,
    const SRC_LAYOUT: u8,
    const DST_LAYOUT: u8,
    const GRID_SIZE: usize,
    const BIT_DEPTH: usize,
> TransformLut3x3<T, SRC_LAYOUT, DST_LAYOUT, GRID_SIZE, BIT_DEPTH>
where
    f32: AsPrimitive<T>,
    u32: AsPrimitive<T>,
{
    #[inline(always)]
    fn transform_chunk<'b, Tetrahedral: MultidimensionalInterpolation<'b, GRID_SIZE>>(
        &'b self,
        src: &[T],
        dst: &mut [T],
    ) {
        let src_cn = Layout::from(SRC_LAYOUT);
        let src_channels = src_cn.channels();

        let dst_cn = Layout::from(DST_LAYOUT);
        let dst_channels = dst_cn.channels();

        let value_scale = ((1 << BIT_DEPTH) - 1) as f32;
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

            let tetrahedral = Tetrahedral::new(&self.lut);
            let v = tetrahedral.inter3(x, y, z);
            let r = if T::FINITE {
                v * value_scale + 0.5f32
            } else {
                v
            };
            dst[dst_cn.r_i()] = r.v[0].min(value_scale).max(0f32).as_();
            dst[dst_cn.g_i()] = r.v[1].min(value_scale).max(0f32).as_();
            dst[dst_cn.b_i()] = r.v[2].min(value_scale).max(0f32).as_();
            if dst_channels == 4 {
                dst[dst_cn.a_i()] = a;
            }
        }
    }
}

impl<
    T: Copy + AsPrimitive<f32> + Default + CompressForLut + PointeeSizeExpressible,
    const SRC_LAYOUT: u8,
    const DST_LAYOUT: u8,
    const GRID_SIZE: usize,
    const BIT_DEPTH: usize,
> TransformExecutor<T> for TransformLut3x3<T, SRC_LAYOUT, DST_LAYOUT, GRID_SIZE, BIT_DEPTH>
where
    f32: AsPrimitive<T>,
    u32: AsPrimitive<T>,
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

        match self.interpolation_method {
            InterpolationMethod::Tetrahedral => {
                use crate::conversions::interpolator::Tetrahedral;
                self.transform_chunk::<Tetrahedral<GRID_SIZE>>(src, dst);
            }
            InterpolationMethod::Pyramid => {
                use crate::conversions::interpolator::Pyramidal;
                self.transform_chunk::<Pyramidal<GRID_SIZE>>(src, dst);
            }
            InterpolationMethod::Prism => {
                use crate::conversions::interpolator::Prismatic;
                self.transform_chunk::<Prismatic<GRID_SIZE>>(src, dst);
            }
            InterpolationMethod::Linear => {
                use crate::conversions::interpolator::Trilinear;
                self.transform_chunk::<Trilinear<GRID_SIZE>>(src, dst);
            }
        }

        Ok(())
    }
}

pub(crate) struct DefaultLut3x3Factory {}

impl Lut3x3Factory for DefaultLut3x3Factory {
    fn make_transform_3x3<
        T: Copy + AsPrimitive<f32> + Default + CompressForLut + PointeeSizeExpressible + 'static,
        const SRC_LAYOUT: u8,
        const DST_LAYOUT: u8,
        const GRID_SIZE: usize,
        const BIT_DEPTH: usize,
    >(
        lut: Vec<f32>,
        interpolation_method: InterpolationMethod,
    ) -> impl TransformExecutor<T>
    where
        f32: AsPrimitive<T>,
        u32: AsPrimitive<T>,
    {
        TransformLut3x3::<T, SRC_LAYOUT, DST_LAYOUT, GRID_SIZE, BIT_DEPTH> {
            lut,
            _phantom: PhantomData,
            interpolation_method,
        }
    }
}
