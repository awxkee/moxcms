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
use crate::conversions::interpolator::{
    MultidimensionalInterpolation, Prismatic, Pyramidal, Tetrahedral, Trilinear,
};
use crate::conversions::lut_transforms::Lut4x3Factory;
use crate::math::{FusedMultiplyAdd, m_clamp};
use crate::{
    CmsError, InterpolationMethod, Layout, PointeeSizeExpressible, TransformExecutor, Vector3f,
};
use num_traits::AsPrimitive;
use std::marker::PhantomData;

pub(crate) trait Vector3fCmykLerp {
    fn interpolate(a: Vector3f, b: Vector3f, t: f32, scale: f32) -> Vector3f;
}

#[allow(unused)]
#[derive(Copy, Clone, Default)]
struct DefaultVector3fLerp;

impl Vector3fCmykLerp for DefaultVector3fLerp {
    #[inline(always)]
    fn interpolate(a: Vector3f, b: Vector3f, t: f32, scale: f32) -> Vector3f {
        let t = Vector3f::from(t);
        let mut new_vec = (a * (Vector3f::from(1.0) - t)).mla(b, t) * scale + 0.5f32;
        new_vec.v[0] = m_clamp(new_vec.v[0], 0.0, scale);
        new_vec.v[1] = m_clamp(new_vec.v[1], 0.0, scale);
        new_vec.v[2] = m_clamp(new_vec.v[2], 0.0, scale);
        new_vec
    }
}

#[allow(unused)]
#[derive(Copy, Clone, Default)]
pub(crate) struct NonFiniteVector3fLerp;

impl Vector3fCmykLerp for NonFiniteVector3fLerp {
    #[inline(always)]
    fn interpolate(a: Vector3f, b: Vector3f, t: f32, _: f32) -> Vector3f {
        let t = Vector3f::from(t);
        let mut new_vec = (a * (Vector3f::from(1.0) - t)).mla(b, t);
        new_vec.v[0] = m_clamp(new_vec.v[0], 0.0, 1.0);
        new_vec.v[1] = m_clamp(new_vec.v[1], 0.0, 1.0);
        new_vec.v[2] = m_clamp(new_vec.v[2], 0.0, 1.0);
        new_vec
    }
}

#[allow(unused)]
struct TransformLut4XyzToRgb<T, const LAYOUT: u8, const GRID_SIZE: usize, const BIT_DEPTH: usize> {
    lut: Vec<f32>,
    _phantom: PhantomData<T>,
    interpolation_method: InterpolationMethod,
}

#[allow(unused)]
impl<
    T: Copy + AsPrimitive<f32> + Default + CompressForLut,
    const LAYOUT: u8,
    const GRID_SIZE: usize,
    const BIT_DEPTH: usize,
> TransformLut4XyzToRgb<T, LAYOUT, GRID_SIZE, BIT_DEPTH>
where
    f32: AsPrimitive<T>,
    u32: AsPrimitive<T>,
{
    #[inline(always)]
    fn transform_chunk<
        'k,
        Tetrahedral: MultidimensionalInterpolation<'k, GRID_SIZE>,
        Interpolation: Vector3fCmykLerp,
    >(
        &'k self,
        src: &[T],
        dst: &mut [T],
    ) {
        let cn = Layout::from(LAYOUT);
        let channels = cn.channels();
        let grid_size = GRID_SIZE as i32;
        let grid_size3 = grid_size * grid_size * grid_size;

        let value_scale = ((1 << BIT_DEPTH) - 1) as f32;
        let max_value = ((1 << BIT_DEPTH) - 1u32).as_();

        for (src, dst) in src.chunks_exact(4).zip(dst.chunks_exact_mut(channels)) {
            let c = src[0].compress_lut::<BIT_DEPTH>();
            let m = src[1].compress_lut::<BIT_DEPTH>();
            let y = src[2].compress_lut::<BIT_DEPTH>();
            let k = src[3].compress_lut::<BIT_DEPTH>();
            let linear_k: f32 = k as i32 as f32 * (1. / 65535.);
            let w: i32 = k as i32 * (GRID_SIZE as i32 - 1) / 65535;
            let w_n: i32 = (w + 1).min(GRID_SIZE as i32 - 1);
            let t: f32 = linear_k * (GRID_SIZE as i32 - 1) as f32 - w as f32;

            let table1 = &self.lut[(w * grid_size3 * 3) as usize..];
            let table2 = &self.lut[(w_n * grid_size3 * 3) as usize..];

            let tetrahedral1 = Tetrahedral::new(table1);
            let tetrahedral2 = Tetrahedral::new(table2);
            let r1 = tetrahedral1.inter3(c, m, y);
            let r2 = tetrahedral2.inter3(c, m, y);
            let r = Interpolation::interpolate(r1, r2, t, value_scale);
            dst[cn.r_i()] = r.v[0].as_();
            dst[cn.g_i()] = r.v[1].as_();
            dst[cn.b_i()] = r.v[2].as_();
            if channels == 4 {
                dst[cn.a_i()] = max_value;
            }
        }
    }
}

#[allow(unused)]
impl<
    T: Copy + AsPrimitive<f32> + Default + CompressForLut + PointeeSizeExpressible,
    const LAYOUT: u8,
    const GRID_SIZE: usize,
    const BIT_DEPTH: usize,
> TransformExecutor<T> for TransformLut4XyzToRgb<T, LAYOUT, GRID_SIZE, BIT_DEPTH>
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
                if T::FINITE {
                    self.transform_chunk::<Tetrahedral<GRID_SIZE>, DefaultVector3fLerp>(src, dst);
                } else {
                    self.transform_chunk::<Tetrahedral<GRID_SIZE>, NonFiniteVector3fLerp>(src, dst);
                }
            }
            InterpolationMethod::Pyramid => {
                if T::FINITE {
                    self.transform_chunk::<Pyramidal<GRID_SIZE>, DefaultVector3fLerp>(src, dst);
                } else {
                    self.transform_chunk::<Pyramidal<GRID_SIZE>, NonFiniteVector3fLerp>(src, dst);
                }
            }
            InterpolationMethod::Prism => {
                if T::FINITE {
                    self.transform_chunk::<Prismatic<GRID_SIZE>, DefaultVector3fLerp>(src, dst);
                } else {
                    self.transform_chunk::<Prismatic<GRID_SIZE>, NonFiniteVector3fLerp>(src, dst);
                }
            }
            InterpolationMethod::Linear => {
                if T::FINITE {
                    self.transform_chunk::<Trilinear<GRID_SIZE>, DefaultVector3fLerp>(src, dst);
                } else {
                    self.transform_chunk::<Trilinear<GRID_SIZE>, NonFiniteVector3fLerp>(src, dst);
                }
            }
        }

        Ok(())
    }
}

#[allow(dead_code)]
pub(crate) struct DefaultLut4x3Factory {}

#[allow(dead_code)]
impl Lut4x3Factory for DefaultLut4x3Factory {
    fn make_transform_4x3<
        T: Copy + AsPrimitive<f32> + Default + CompressForLut + PointeeSizeExpressible + 'static,
        const LAYOUT: u8,
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
        TransformLut4XyzToRgb::<T, LAYOUT, GRID_SIZE, BIT_DEPTH> {
            lut,
            _phantom: PhantomData,
            interpolation_method,
        }
    }
}
