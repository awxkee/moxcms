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
use crate::conversions::interpolator::BarycentricWeightQ1_15;
use crate::conversions::neon::interpolator_q1_15::{
    NeonAlignedI16x4, NeonMdInterpolationQ1_15Double, PrismaticNeonQ1_15Double,
    PyramidalNeonQ1_15Double, TetrahedralNeonQ1_15Double, TrilinearNeonQ1_15Double,
};
use crate::transform::PointeeSizeExpressible;
use crate::{CmsError, InterpolationMethod, Layout, TransformExecutor};
use num_traits::AsPrimitive;
use std::arch::aarch64::*;
use std::marker::PhantomData;

pub(crate) struct TransformLut4XyzToRgbNeonQ1_15<
    T,
    const LAYOUT: u8,
    const GRID_SIZE: usize,
    const BIT_DEPTH: usize,
> {
    pub(crate) lut: Vec<NeonAlignedI16x4>,
    pub(crate) _phantom: PhantomData<T>,
    pub(crate) interpolation_method: InterpolationMethod,
    pub(crate) weights: Box<[BarycentricWeightQ1_15; 256]>,
}

impl<
    T: Copy + AsPrimitive<f32> + Default + CompressForLut + PointeeSizeExpressible,
    const LAYOUT: u8,
    const GRID_SIZE: usize,
    const BIT_DEPTH: usize,
> TransformLut4XyzToRgbNeonQ1_15<T, LAYOUT, GRID_SIZE, BIT_DEPTH>
where
    f32: AsPrimitive<T>,
    u32: AsPrimitive<T>,
{
    #[allow(unused_unsafe)]
    #[target_feature(enable = "rdm")]
    unsafe fn transform_chunk<'b, Interpolator: NeonMdInterpolationQ1_15Double<'b, GRID_SIZE>>(
        &'b self,
        src: &[T],
        dst: &mut [T],
    ) {
        unsafe {
            let cn = Layout::from(LAYOUT);
            let channels = cn.channels();
            let grid_size = GRID_SIZE as i32;
            let grid_size3 = grid_size * grid_size * grid_size;

            const FLOAT_RECIP: f32 = 1. / ((1i32 << 15i32) - 1) as f32;
            let value_scale = vdupq_n_f32(FLOAT_RECIP);
            let b_max_value = ((1u32 << BIT_DEPTH) - 1) as i16;
            let max_value = ((1u32 << BIT_DEPTH) - 1).as_();
            let q_max = vdup_n_s16(((1i32 << 15i32) - 1) as i16);
            let zeros = vdup_n_s16(0);
            let v_local_max = vdup_n_s16(b_max_value);

            for (src, dst) in src.chunks_exact(4).zip(dst.chunks_exact_mut(channels)) {
                let c = src[0].compress_lut::<BIT_DEPTH>();
                let m = src[1].compress_lut::<BIT_DEPTH>();
                let y = src[2].compress_lut::<BIT_DEPTH>();
                let k = src[3].compress_lut::<BIT_DEPTH>();

                let k_weights = self.weights[k as usize];

                let w: i32 = k_weights.x;
                let w_n: i32 = k_weights.x_n;
                let t: i16 = k_weights.w;

                let table1 = &self.lut[(w * grid_size3) as usize..];
                let table2 = &self.lut[(w_n * grid_size3) as usize..];

                let tetrahedral1 = Interpolator::new(table1, table2);
                let (a0, b0) = tetrahedral1.inter3_neon(c, m, y, &self.weights);
                let (a0, b0) = (a0.v, b0.v);

                let t0 = vdup_n_s16(t);
                let hp = vqrdmlsh_s16(a0, a0, t0);
                let v = vqrdmlah_s16(hp, b0, t0);

                if T::FINITE {
                    if BIT_DEPTH == 8 {
                        let r = vqrshrun_n_s16::<7>(vcombine_s16(v, v));

                        dst[cn.r_i()] = (vget_lane_u8::<0>(r) as u32).as_();
                        dst[cn.g_i()] = (vget_lane_u8::<1>(r) as u32).as_();
                        dst[cn.b_i()] = (vget_lane_u8::<2>(r) as u32).as_();
                    } else {
                        let mut r = vmax_s16(v, zeros);
                        r = vmax_s16(r, zeros);
                        r = if BIT_DEPTH == 12 {
                            vrshr_n_s16::<3>(r)
                        } else if BIT_DEPTH == 10 {
                            vrshr_n_s16::<5>(r)
                        } else {
                            vrshr_n_s16::<7>(r)
                        };
                        r = vmin_s16(r, v_local_max);
                        let r = vreinterpret_u16_s16(r);

                        dst[cn.r_i()] = (vget_lane_u16::<0>(r) as u32).as_();
                        dst[cn.g_i()] = (vget_lane_u16::<1>(r) as u32).as_();
                        dst[cn.b_i()] = (vget_lane_u16::<2>(r) as u32).as_();
                    }
                } else {
                    unsafe {
                        let mut r = vmax_s16(v, zeros);
                        r = vmax_s16(r, zeros);
                        r = vmin_s16(r, q_max);
                        let mut v = vcvtq_f32_s32(vmovl_s16(r));
                        v = vmulq_f32(v, value_scale);
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
}

impl<
    T: Copy + AsPrimitive<f32> + Default + CompressForLut + PointeeSizeExpressible,
    const LAYOUT: u8,
    const GRID_SIZE: usize,
    const BIT_DEPTH: usize,
> TransformExecutor<T> for TransformLut4XyzToRgbNeonQ1_15<T, LAYOUT, GRID_SIZE, BIT_DEPTH>
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
            match self.interpolation_method {
                InterpolationMethod::Tetrahedral => {
                    self.transform_chunk::<TetrahedralNeonQ1_15Double<GRID_SIZE>>(src, dst);
                }
                InterpolationMethod::Pyramid => {
                    self.transform_chunk::<PyramidalNeonQ1_15Double<GRID_SIZE>>(src, dst);
                }
                InterpolationMethod::Prism => {
                    self.transform_chunk::<PrismaticNeonQ1_15Double<GRID_SIZE>>(src, dst);
                }
                InterpolationMethod::Linear => {
                    self.transform_chunk::<TrilinearNeonQ1_15Double<GRID_SIZE>>(src, dst);
                }
            }
        }

        Ok(())
    }
}
