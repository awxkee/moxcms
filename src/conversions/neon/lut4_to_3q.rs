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
use crate::conversions::neon::interpolator_q2_14::{
    NeonAlignedI16x4, NeonMdInterpolationQ2_14Double, PrismaticNeonQ2_14Double,
    PyramidalNeonQ2_14Double, TetrahedralNeonQ2_14Double,
};
use crate::conversions::CompressForLut;
use crate::{
    rounding_div_ceil, CmsError, InterpolationMethod, Layout, PointeeSizeExpressible,
    TransformExecutor,
};
use num_traits::AsPrimitive;
use std::arch::aarch64::*;
use std::marker::PhantomData;

pub(crate) struct TransformLut4XyzToRgbNeonQ2_14<
    T,
    const LAYOUT: u8,
    const GRID_SIZE: usize,
    const BIT_DEPTH: usize,
> {
    pub(crate) lut: Vec<NeonAlignedI16x4>,
    pub(crate)  _phantom: PhantomData<T>,
    pub(crate) interpolation_method: InterpolationMethod,
}

impl<
    T: Copy + AsPrimitive<f32> + Default + CompressForLut + PointeeSizeExpressible,
    const LAYOUT: u8,
    const GRID_SIZE: usize,
    const BIT_DEPTH: usize,
> TransformLut4XyzToRgbNeonQ2_14<T, LAYOUT, GRID_SIZE, BIT_DEPTH>
where
    f32: AsPrimitive<T>,
    u32: AsPrimitive<T>,
{
    #[allow(unused_unsafe)]
    fn transform_chunk<'b, Interpolator: NeonMdInterpolationQ2_14Double<'b, GRID_SIZE>>(
        &'b self,
        src: &[T],
        dst: &mut [T],
    ) {
        let cn = Layout::from(LAYOUT);
        let channels = cn.channels();
        let grid_size = GRID_SIZE as i32;
        let grid_size3 = grid_size * grid_size * grid_size;

        let value_scale_f = unsafe { vdupq_n_f32(1. / ((1 << 15) - 1) as f32) };
        let v_max_value = unsafe { vdup_n_s16((1 << BIT_DEPTH) - 1) };
        let max_value = ((1 << BIT_DEPTH) - 1u32).as_();

        for (src, dst) in src.chunks_exact(4).zip(dst.chunks_exact_mut(channels)) {
            let c = src[0].compress_lut::<BIT_DEPTH>();
            let m = src[1].compress_lut::<BIT_DEPTH>();
            let y = src[2].compress_lut::<BIT_DEPTH>();
            let k = src[3].compress_lut::<BIT_DEPTH>();
            let linear_k: f32 = k as i32 as f32 / 65535.0;
            let w: i32 = k as i32 * (GRID_SIZE as i32 - 1) / 65535;
            let w_n: i32 = rounding_div_ceil(k as i32 * (GRID_SIZE as i32 - 1), 65535);
            const Q_SCALE: f32 =  ((1 << 15) - 1) as f32;
            let t: i16 =
                ((linear_k * (GRID_SIZE as i32 - 1) as f32 - w as f32) * Q_SCALE) as i16;

            let table1 = &self.lut[(w * grid_size3) as usize..];
            let table2 = &self.lut[(w_n * grid_size3) as usize..];

            let tetrahedral1 = Interpolator::new(table1, table2);
            let (a0, b0) = tetrahedral1.inter3_neon(c, m, y);
            let (a0, b0) = (a0.v, b0.v);

            if T::FINITE {
                unsafe {
                    let t0 = vdup_n_s16(t);
                    let ones = vdup_n_s16(1 << 15);
                    let hp = vqdmulh_s16(a0, vsub_s16(ones, t0));
                    let mut v = vqrdmlah_s16(hp, b0, t0);
                    if BIT_DEPTH == 12 {
                        v = vshr_n_s16::<3>(v);
                    } else if BIT_DEPTH == 10 {
                        v = vshr_n_s16::<5>(v);
                    } else if BIT_DEPTH == 8 {
                        v = vshr_n_s16::<7>(v);
                    }
                    v = vmin_s16(v, v_max_value);
                    v = vmax_s16(v, vdup_n_s16(0));

                    let v = vreinterpret_u16_s16(v);

                    dst[cn.r_i()] = (vget_lane_u16::<0>(v) as u32).as_();
                    dst[cn.g_i()] = (vget_lane_u16::<1>(v) as u32).as_();
                    dst[cn.b_i()] = (vget_lane_u16::<2>(v) as u32).as_();
                }
            } else {
                unsafe {
                    let t0 = vdup_n_s16(t);
                    let ones = vdup_n_s16(((1i32 << 15i32) - 1i32) as i16);
                    let hp = vqdmulh_s16(a0, vsub_s16(ones, t0));
                    let mut v = vqrdmlah_s16(hp, b0, t0);
                    v = vmin_s16(v, ones);
                    v = vmax_s16(v, vdup_n_s16(0));
                    let mut x = vcvtq_f32_s32(vmovl_s16(v));
                    x = vmulq_f32(x, value_scale_f);

                    dst[cn.r_i()] = vgetq_lane_f32::<0>(x).as_();
                    dst[cn.g_i()] = vgetq_lane_f32::<1>(x).as_();
                    dst[cn.b_i()] = vgetq_lane_f32::<2>(x).as_();
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
> TransformExecutor<T> for TransformLut4XyzToRgbNeonQ2_14<T, LAYOUT, GRID_SIZE, BIT_DEPTH>
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
                self.transform_chunk::<TetrahedralNeonQ2_14Double<GRID_SIZE>>(src, dst);
            }
            InterpolationMethod::Pyramid => {
                self.transform_chunk::<PyramidalNeonQ2_14Double<GRID_SIZE>>(src, dst);
            }
            InterpolationMethod::Prism => {
                self.transform_chunk::<PrismaticNeonQ2_14Double<GRID_SIZE>>(src, dst);
            }
        }

        Ok(())
    }
}
