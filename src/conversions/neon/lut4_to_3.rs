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
use crate::conversions::interpolator::{BarycentricWeight, BarycentricWeightQ1_15};
use crate::conversions::lut_transforms::Lut4x3Factory;
use crate::conversions::neon::interpolator::{
    NeonMdInterpolationDouble, PrismaticNeonDouble, PyramidalNeonDouble, TetrahedralNeonDouble,
    TrilinearNeonDouble,
};
use crate::conversions::neon::interpolator_q1_15::NeonAlignedI16x4;
use crate::conversions::neon::lut4_to_3_q1_15::TransformLut4XyzToRgbNeonQ1_15;
use crate::conversions::neon::rgb_xyz::NeonAlignedF32;
use crate::transform::PointeeSizeExpressible;
use crate::{CmsError, InterpolationMethod, Layout, TransformExecutor, TransformOptions};
use num_traits::AsPrimitive;
use std::arch::aarch64::*;
use std::marker::PhantomData;

struct TransformLut4XyzToRgbNeon<
    T,
    const LAYOUT: u8,
    const GRID_SIZE: usize,
    const BIT_DEPTH: usize,
> {
    lut: Vec<NeonAlignedF32>,
    _phantom: PhantomData<T>,
    interpolation_method: InterpolationMethod,
    weights: Box<[BarycentricWeight; 256]>,
}

impl<
    T: Copy + AsPrimitive<f32> + Default + CompressForLut + PointeeSizeExpressible,
    const LAYOUT: u8,
    const GRID_SIZE: usize,
    const BIT_DEPTH: usize,
> TransformLut4XyzToRgbNeon<T, LAYOUT, GRID_SIZE, BIT_DEPTH>
where
    f32: AsPrimitive<T>,
    u32: AsPrimitive<T>,
{
    #[allow(unused_unsafe)]
    fn transform_chunk<'b, Interpolator: NeonMdInterpolationDouble<'b, GRID_SIZE>>(
        &'b self,
        src: &[T],
        dst: &mut [T],
    ) {
        let cn = Layout::from(LAYOUT);
        let channels = cn.channels();
        let grid_size = GRID_SIZE as i32;
        let grid_size3 = grid_size * grid_size * grid_size;

        let value_scale = unsafe { vdupq_n_f32(((1 << BIT_DEPTH) - 1) as f32) };
        let max_value = ((1 << BIT_DEPTH) - 1u32).as_();

        for (src, dst) in src.chunks_exact(4).zip(dst.chunks_exact_mut(channels)) {
            let c = src[0].compress_lut::<BIT_DEPTH>();
            let m = src[1].compress_lut::<BIT_DEPTH>();
            let y = src[2].compress_lut::<BIT_DEPTH>();
            let k = src[3].compress_lut::<BIT_DEPTH>();

            let k_weights = self.weights[k as usize];

            let w: i32 = k_weights.x;
            let w_n: i32 = k_weights.x_n;
            let t: f32 = k_weights.w;

            let table1 = &self.lut[(w * grid_size3) as usize..];
            let table2 = &self.lut[(w_n * grid_size3) as usize..];

            let tetrahedral1 = Interpolator::new(table1, table2);
            let (a0, b0) = tetrahedral1.inter3_neon(c, m, y, &self.weights);
            let (a0, b0) = (a0.v, b0.v);

            if T::FINITE {
                unsafe {
                    let t0 = vdupq_n_f32(t);
                    let hp = vfmsq_f32(a0, a0, t0);
                    let mut v = vfmaq_f32(hp, b0, t0);
                    v = vmulq_f32(v, value_scale);
                    v = vminq_f32(v, value_scale);

                    let jvx = vcvtaq_u32_f32(v);

                    dst[cn.r_i()] = vgetq_lane_u32::<0>(jvx).as_();
                    dst[cn.g_i()] = vgetq_lane_u32::<1>(jvx).as_();
                    dst[cn.b_i()] = vgetq_lane_u32::<2>(jvx).as_();
                }
            } else {
                unsafe {
                    let t0 = vdupq_n_f32(t);
                    let hp = vfmsq_f32(a0, a0, t0);
                    let mut v = vfmaq_f32(hp, b0, t0);
                    v = vminq_f32(v, value_scale);
                    v = vmaxq_f32(v, vdupq_n_f32(0.));

                    dst[cn.r_i()] = vgetq_lane_f32::<0>(v).as_();
                    dst[cn.g_i()] = vgetq_lane_f32::<1>(v).as_();
                    dst[cn.b_i()] = vgetq_lane_f32::<2>(v).as_();
                }
            }
            if channels == 4 {
                dst[cn.a_i()] = max_value;
            }
        }
    }
}

impl<
    T: Copy + AsPrimitive<f32> + Default + CompressForLut + PointeeSizeExpressible,
    const LAYOUT: u8,
    const GRID_SIZE: usize,
    const BIT_DEPTH: usize,
> TransformExecutor<T> for TransformLut4XyzToRgbNeon<T, LAYOUT, GRID_SIZE, BIT_DEPTH>
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

        match self.interpolation_method {
            InterpolationMethod::Tetrahedral => {
                self.transform_chunk::<TetrahedralNeonDouble<GRID_SIZE>>(src, dst);
            }
            InterpolationMethod::Pyramid => {
                self.transform_chunk::<PyramidalNeonDouble<GRID_SIZE>>(src, dst);
            }
            InterpolationMethod::Prism => {
                self.transform_chunk::<PrismaticNeonDouble<GRID_SIZE>>(src, dst);
            }
            InterpolationMethod::Linear => {
                self.transform_chunk::<TrilinearNeonDouble<GRID_SIZE>>(src, dst);
            }
        }

        Ok(())
    }
}

pub(crate) struct NeonLut4x3Factory {}

impl Lut4x3Factory for NeonLut4x3Factory {
    fn make_transform_4x3<
        T: Copy
            + AsPrimitive<f32>
            + Default
            + CompressForLut
            + PointeeSizeExpressible
            + 'static
            + Send
            + Sync,
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
    {
        if options.prefer_fixed_point
            && std::arch::is_aarch64_feature_detected!("rdm")
            && BIT_DEPTH < 15
        {
            const Q_SCALE: f32 = ((1 << 15) - 1) as f32;
            let lut = lut
                .chunks_exact(3)
                .map(|x| {
                    NeonAlignedI16x4([
                        (x[0] * Q_SCALE).round() as i16,
                        (x[1] * Q_SCALE).round() as i16,
                        (x[2] * Q_SCALE).round() as i16,
                        0,
                    ])
                })
                .collect::<Vec<_>>();
            return Box::new(
                TransformLut4XyzToRgbNeonQ1_15::<T, LAYOUT, GRID_SIZE, BIT_DEPTH> {
                    lut,
                    _phantom: PhantomData,
                    interpolation_method: options.interpolation_method,
                    weights: BarycentricWeightQ1_15::create_ranged_256::<GRID_SIZE>(),
                },
            );
        }
        let lut = lut
            .chunks_exact(3)
            .map(|x| NeonAlignedF32([x[0], x[1], x[2], 0f32]))
            .collect::<Vec<_>>();
        Box::new(
            TransformLut4XyzToRgbNeon::<T, LAYOUT, GRID_SIZE, BIT_DEPTH> {
                lut,
                _phantom: PhantomData,
                interpolation_method: options.interpolation_method,
                weights: BarycentricWeight::create_ranged_256::<GRID_SIZE>(),
            },
        )
    }
}
