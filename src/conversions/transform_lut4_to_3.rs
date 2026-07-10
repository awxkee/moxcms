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
#![cfg(feature = "lut")]
use crate::conversions::LutBarycentricReduction;
use crate::conversions::interpolator::*;
use crate::conversions::lut_transforms::Lut4x3Factory;
use crate::math::{FusedMultiplyAdd, FusedMultiplyNegAdd, m_clamp};
use crate::{
    BarycentricWeightScale, CmsError, DataColorSpace, InterpolationMethod, Layout,
    PointeeSizeExpressible, TransformExecutor, TransformOptions, Vector3f,
};
use num_traits::AsPrimitive;
use std::marker::PhantomData;
use std::sync::Arc;

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
        let inter = a.neg_mla(a, t).mla(b, t);
        let mut new_vec = Vector3f::from(0.5).mla(inter, Vector3f::from(scale));
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
        a.neg_mla(a, t).mla(b, t)
    }
}

#[allow(unused)]
#[derive(Copy, Clone, Default)]
pub(crate) struct NonFiniteVector3fLerpUnbound;

impl Vector3fCmykLerp for NonFiniteVector3fLerpUnbound {
    #[inline(always)]
    fn interpolate(a: Vector3f, b: Vector3f, t: f32, _: f32) -> Vector3f {
        let t = Vector3f::from(t);
        a.neg_mla(a, t).mla(b, t)
    }
}

#[allow(unused)]
struct TransformLut4To3<
    T,
    U,
    const LAYOUT: u8,
    const GRID_SIZE: usize,
    const BIT_DEPTH: usize,
    const BINS: usize,
    const BARYCENTRIC_BINS: usize,
> {
    lut: Vec<f32>,
    _phantom: PhantomData<T>,
    _phantom1: PhantomData<U>,
    interpolation_method: InterpolationMethod,
    weights: Box<[BarycentricWeight<f32>; BINS]>,
    color_space: DataColorSpace,
    is_linear: bool,
}

#[repr(align(8), C)]
struct AlignedI16x4([i16; 4]);

#[allow(unused)]
struct TransformLut4To3Q0_15<
    T,
    U,
    const LAYOUT: u8,
    const GRID_SIZE: usize,
    const BIT_DEPTH: usize,
    const BINS: usize,
    const BARYCENTRIC_BINS: usize,
> {
    lut: Vec<AlignedI16x4>,
    _phantom: PhantomData<T>,
    _phantom1: PhantomData<U>,
    weights: Box<[BarycentricWeight<i16>; BINS]>,
}

#[inline(always)]
fn q0_15_mul(a: i16, b: i16) -> i16 {
    let product = a as i32 * b as i32;
    (((product >> 14) + 1) >> 1) as i16
}

#[inline(always)]
fn q0_15_lerp(a: i16, b: i16, t: i16) -> i16 {
    const Q: i16 = ((1i32 << 15) - 1) as i16;
    q0_15_mul(a, Q - t) + q0_15_mul(b, t)
}

#[inline(always)]
fn q0_15_lerp3(a: [i16; 3], b: [i16; 3], t: i16) -> [i16; 3] {
    [
        q0_15_lerp(a[0], b[0], t),
        q0_15_lerp(a[1], b[1], t),
        q0_15_lerp(a[2], b[2], t),
    ]
}

#[allow(unused)]
impl<
    T: Copy + AsPrimitive<f32> + Default + PointeeSizeExpressible,
    U: AsPrimitive<usize>,
    const LAYOUT: u8,
    const GRID_SIZE: usize,
    const BIT_DEPTH: usize,
    const BINS: usize,
    const BARYCENTRIC_BINS: usize,
> TransformLut4To3Q0_15<T, U, LAYOUT, GRID_SIZE, BIT_DEPTH, BINS, BARYCENTRIC_BINS>
where
    f32: AsPrimitive<T>,
    u32: AsPrimitive<T>,
    (): LutBarycentricReduction<T, U>,
{
    #[inline(always)]
    fn fetch(&self, table: &[AlignedI16x4], x: i32, y: i32, z: i32) -> [i16; 3] {
        let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
            + y as u32 * GRID_SIZE as u32
            + z as u32) as usize;
        let [r, g, b, _] = table[offset].0;
        [r, g, b]
    }

    #[inline(always)]
    fn inter3(&self, table: &[AlignedI16x4], c: U, m: U, y: U) -> [i16; 3] {
        let c = self.weights[c.as_()];
        let m = self.weights[m.as_()];
        let y = self.weights[y.as_()];

        let c000 = self.fetch(table, c.x, m.x, y.x);
        let c100 = self.fetch(table, c.x_n, m.x, y.x);
        let c010 = self.fetch(table, c.x, m.x_n, y.x);
        let c110 = self.fetch(table, c.x_n, m.x_n, y.x);
        let c001 = self.fetch(table, c.x, m.x, y.x_n);
        let c101 = self.fetch(table, c.x_n, m.x, y.x_n);
        let c011 = self.fetch(table, c.x, m.x_n, y.x_n);
        let c111 = self.fetch(table, c.x_n, m.x_n, y.x_n);

        let c00 = q0_15_lerp3(c000, c100, c.w);
        let c10 = q0_15_lerp3(c010, c110, c.w);
        let c01 = q0_15_lerp3(c001, c101, c.w);
        let c11 = q0_15_lerp3(c011, c111, c.w);
        let c0 = q0_15_lerp3(c00, c10, m.w);
        let c1 = q0_15_lerp3(c01, c11, m.w);
        q0_15_lerp3(c0, c1, y.w)
    }

    #[inline(never)]
    fn transform_chunk(&self, src: &[T], dst: &mut [T]) {
        let cn = Layout::from(LAYOUT);
        let channels = cn.channels();
        let grid_size = GRID_SIZE as i32;
        let grid_size3 = grid_size * grid_size * grid_size;

        let max_value = ((1 << BIT_DEPTH) - 1u32).as_();
        let max_i16 = if T::FINITE {
            ((1i32 << BIT_DEPTH) - 1) as i16
        } else {
            ((1i32 << 14) - 1) as i16
        };

        for (src, dst) in src
            .as_chunks::<4>()
            .0
            .iter()
            .zip(dst.chunks_exact_mut(channels))
        {
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
            let table1 = &self.lut[(w * grid_size3) as usize..];
            let table2 = &self.lut[(w_n * grid_size3) as usize..];

            let r1 = self.inter3(table1, c, m, y);
            let r2 = self.inter3(table2, c, m, y);
            let r = q0_15_lerp3(r1, r2, k_weights.w);

            if T::FINITE {
                dst[cn.r_i()] = (r[0].clamp(0, max_i16) as u32).as_();
                dst[cn.g_i()] = (r[1].clamp(0, max_i16) as u32).as_();
                dst[cn.b_i()] = (r[2].clamp(0, max_i16) as u32).as_();
            } else {
                let scale = 1.0 / ((1 << 14) - 1) as f32;
                dst[cn.r_i()] = (r[0] as f32 * scale).as_();
                dst[cn.g_i()] = (r[1] as f32 * scale).as_();
                dst[cn.b_i()] = (r[2] as f32 * scale).as_();
            }
            if channels == 4 {
                dst[cn.a_i()] = max_value;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::q0_15_mul;

    #[inline]
    fn pmulhrsw_reference(a: i16, b: i16) -> i16 {
        let product = a as i32 * b as i32;
        (((product >> 14) + 1) >> 1) as i16
    }

    #[test]
    fn test_q0_15_mul_matches_pmulhrsw_rounding() {
        let values = [
            i16::MIN,
            -30000,
            -16384,
            -8192,
            -1,
            0,
            1,
            8191,
            8192,
            16383,
            16384,
            30000,
            i16::MAX,
        ];

        for &a in &values {
            for &b in &values {
                assert_eq!(
                    q0_15_mul(a, b),
                    pmulhrsw_reference(a, b),
                    "Q0.15 multiplication differs for {a} * {b}",
                );
            }
        }
    }

    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
    #[test]
    fn test_q0_15_mul_matches_x86_mulhrs_epi16() {
        if std::is_x86_feature_detected!("ssse3") {
            unsafe {
                assert_q0_15_mul_matches_x86_mulhrs_epi16();
            }
        }
    }

    #[cfg(all(target_arch = "x86", feature = "sse"))]
    #[target_feature(enable = "ssse3")]
    unsafe fn assert_q0_15_mul_matches_x86_mulhrs_epi16() {
        use std::arch::x86::*;

        assert_q0_15_mul_matches_x86_mulhrs_epi16_impl();

        unsafe fn assert_q0_15_mul_matches_x86_mulhrs_epi16_impl() {
            let a = _mm_setr_epi16(i16::MIN, -30000, -16384, -1, 0, 8192, 16384, i16::MAX);
            let b = _mm_setr_epi16(i16::MAX, 16384, 8192, -1, 0, -16384, -30000, i16::MIN);
            let r = _mm_mulhrs_epi16(a, b);
            let mut actual = [0i16; 8];
            _mm_storeu_si128(actual.as_mut_ptr().cast(), r);
            let expected = [
                q0_15_mul(i16::MIN, i16::MAX),
                q0_15_mul(-30000, 16384),
                q0_15_mul(-16384, 8192),
                q0_15_mul(-1, -1),
                q0_15_mul(0, 0),
                q0_15_mul(8192, -16384),
                q0_15_mul(16384, -30000),
                q0_15_mul(i16::MAX, i16::MIN),
            ];
            assert_eq!(actual, expected);
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "sse"))]
    #[target_feature(enable = "ssse3")]
    unsafe fn assert_q0_15_mul_matches_x86_mulhrs_epi16() {
        use std::arch::x86_64::*;

        let a = _mm_setr_epi16(i16::MIN, -30000, -16384, -1, 0, 8192, 16384, i16::MAX);
        let b = _mm_setr_epi16(i16::MAX, 16384, 8192, -1, 0, -16384, -30000, i16::MIN);
        let r = _mm_mulhrs_epi16(a, b);
        let mut actual = [0i16; 8];
        _mm_storeu_si128(actual.as_mut_ptr().cast(), r);
        let expected = [
            q0_15_mul(i16::MIN, i16::MAX),
            q0_15_mul(-30000, 16384),
            q0_15_mul(-16384, 8192),
            q0_15_mul(-1, -1),
            q0_15_mul(0, 0),
            q0_15_mul(8192, -16384),
            q0_15_mul(16384, -30000),
            q0_15_mul(i16::MAX, i16::MIN),
        ];
        assert_eq!(actual, expected);
    }
}

#[allow(unused)]
impl<
    T: Copy + AsPrimitive<f32> + Default + PointeeSizeExpressible,
    U: AsPrimitive<usize>,
    const LAYOUT: u8,
    const GRID_SIZE: usize,
    const BIT_DEPTH: usize,
    const BINS: usize,
    const BARYCENTRIC_BINS: usize,
> TransformExecutor<T>
    for TransformLut4To3Q0_15<T, U, LAYOUT, GRID_SIZE, BIT_DEPTH, BINS, BARYCENTRIC_BINS>
where
    f32: AsPrimitive<T>,
    u32: AsPrimitive<T>,
    (): LutBarycentricReduction<T, U>,
{
    // Keep these checks compatible with the crate's Rust 1.85 MSRV:
    // `usize::is_multiple_of` is only stable starting with Rust 1.87.
    #[allow(clippy::manual_is_multiple_of)]
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

        self.transform_chunk(src, dst);

        Ok(())
    }
}

#[allow(unused)]
impl<
    T: Copy + AsPrimitive<f32> + Default,
    U: AsPrimitive<usize>,
    const LAYOUT: u8,
    const GRID_SIZE: usize,
    const BIT_DEPTH: usize,
    const BINS: usize,
    const BARYCENTRIC_BINS: usize,
> TransformLut4To3<T, U, LAYOUT, GRID_SIZE, BIT_DEPTH, BINS, BARYCENTRIC_BINS>
where
    f32: AsPrimitive<T>,
    u32: AsPrimitive<T>,
    (): LutBarycentricReduction<T, U>,
{
    #[inline(never)]
    fn transform_chunk<Interpolation: Vector3fCmykLerp>(
        &self,
        src: &[T],
        dst: &mut [T],
        interpolator: Box<dyn MultidimensionalInterpolation + Send + Sync>,
    ) {
        let cn = Layout::from(LAYOUT);
        let channels = cn.channels();
        let grid_size = GRID_SIZE as i32;
        let grid_size3 = grid_size * grid_size * grid_size;

        let value_scale = ((1 << BIT_DEPTH) - 1) as f32;
        let max_value = ((1 << BIT_DEPTH) - 1u32).as_();

        for (src, dst) in src
            .as_chunks::<4>()
            .0
            .iter()
            .zip(dst.chunks_exact_mut(channels))
        {
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

            let table1 = &self.lut[(w * grid_size3 * 3) as usize..];
            let table2 = &self.lut[(w_n * grid_size3 * 3) as usize..];

            let r1 = interpolator.inter3(
                table1,
                &self.weights[c.as_()],
                &self.weights[m.as_()],
                &self.weights[y.as_()],
            );
            let r2 = interpolator.inter3(
                table2,
                &self.weights[c.as_()],
                &self.weights[m.as_()],
                &self.weights[y.as_()],
            );
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
    T: Copy + AsPrimitive<f32> + Default + PointeeSizeExpressible,
    U: AsPrimitive<usize>,
    const LAYOUT: u8,
    const GRID_SIZE: usize,
    const BIT_DEPTH: usize,
    const BINS: usize,
    const BARYCENTRIC_BINS: usize,
> TransformExecutor<T>
    for TransformLut4To3<T, U, LAYOUT, GRID_SIZE, BIT_DEPTH, BINS, BARYCENTRIC_BINS>
where
    f32: AsPrimitive<T>,
    u32: AsPrimitive<T>,
    (): LutBarycentricReduction<T, U>,
{
    fn transform(&self, src: &[T], dst: &mut [T]) -> Result<(), CmsError> {
        let cn = Layout::from(LAYOUT);
        let channels = cn.channels();
        if !src.len().is_multiple_of(4) {
            return Err(CmsError::LaneMultipleOfChannels);
        }
        if !dst.len().is_multiple_of(channels) {
            return Err(CmsError::LaneMultipleOfChannels);
        }
        let src_chunks = src.len() / 4;
        let dst_chunks = dst.len() / channels;
        if src_chunks != dst_chunks {
            return Err(CmsError::LaneSizeMismatch);
        }

        if self.color_space == DataColorSpace::Lab
            || (self.is_linear && self.color_space == DataColorSpace::Rgb)
            || self.color_space == DataColorSpace::Xyz
        {
            if T::FINITE {
                self.transform_chunk::<DefaultVector3fLerp>(
                    src,
                    dst,
                    Box::new(Trilinear::<GRID_SIZE> {}),
                );
            } else {
                self.transform_chunk::<NonFiniteVector3fLerp>(
                    src,
                    dst,
                    Box::new(Trilinear::<GRID_SIZE> {}),
                );
            }
        } else {
            match self.interpolation_method {
                #[cfg(feature = "options")]
                InterpolationMethod::Tetrahedral => {
                    if T::FINITE {
                        self.transform_chunk::<DefaultVector3fLerp>(
                            src,
                            dst,
                            Box::new(Tetrahedral::<GRID_SIZE> {}),
                        );
                    } else {
                        self.transform_chunk::<NonFiniteVector3fLerp>(
                            src,
                            dst,
                            Box::new(Tetrahedral::<GRID_SIZE> {}),
                        );
                    }
                }
                #[cfg(feature = "options")]
                InterpolationMethod::Pyramid => {
                    if T::FINITE {
                        self.transform_chunk::<DefaultVector3fLerp>(
                            src,
                            dst,
                            Box::new(Pyramidal::<GRID_SIZE> {}),
                        );
                    } else {
                        self.transform_chunk::<NonFiniteVector3fLerp>(
                            src,
                            dst,
                            Box::new(Pyramidal::<GRID_SIZE> {}),
                        );
                    }
                }
                #[cfg(feature = "options")]
                InterpolationMethod::Prism => {
                    if T::FINITE {
                        self.transform_chunk::<DefaultVector3fLerp>(
                            src,
                            dst,
                            Box::new(Prismatic::<GRID_SIZE> {}),
                        );
                    } else {
                        self.transform_chunk::<NonFiniteVector3fLerp>(
                            src,
                            dst,
                            Box::new(Prismatic::<GRID_SIZE> {}),
                        );
                    }
                }
                InterpolationMethod::Linear => {
                    if T::FINITE {
                        self.transform_chunk::<DefaultVector3fLerp>(
                            src,
                            dst,
                            Box::new(Trilinear::<GRID_SIZE> {}),
                        );
                    } else {
                        self.transform_chunk::<NonFiniteVector3fLerp>(
                            src,
                            dst,
                            Box::new(Trilinear::<GRID_SIZE> {}),
                        );
                    }
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
        T: Copy + AsPrimitive<f32> + Default + PointeeSizeExpressible + 'static + Send + Sync,
        const LAYOUT: u8,
        const GRID_SIZE: usize,
        const BIT_DEPTH: usize,
    >(
        lut: Vec<f32>,
        options: TransformOptions,
        color_space: DataColorSpace,
        is_linear: bool,
    ) -> Arc<dyn TransformExecutor<T> + Sync + Send>
    where
        f32: AsPrimitive<T>,
        u32: AsPrimitive<T>,
        (): LutBarycentricReduction<T, u8>,
        (): LutBarycentricReduction<T, u16>,
    {
        let use_fixed_point = options.prefer_fixed_point
            && BIT_DEPTH < 16
            && (color_space == DataColorSpace::Lab
                || (is_linear && color_space == DataColorSpace::Rgb)
                || color_space == DataColorSpace::Xyz
                || options.interpolation_method == InterpolationMethod::Linear);

        if use_fixed_point {
            let q = if T::FINITE {
                ((1i32 << BIT_DEPTH) - 1) as f32
            } else {
                ((1i32 << 14) - 1) as f32
            };
            let lut: Vec<AlignedI16x4> = lut
                .as_chunks::<3>()
                .0
                .iter()
                .map(|x| {
                    AlignedI16x4([
                        (x[0] * q).round() as i16,
                        (x[1] * q).round() as i16,
                        (x[2] * q).round() as i16,
                        0,
                    ])
                })
                .collect();
            return match options.barycentric_weight_scale {
                BarycentricWeightScale::Low => {
                    Arc::new(
                        TransformLut4To3Q0_15::<T, u8, LAYOUT, GRID_SIZE, BIT_DEPTH, 256, 256> {
                            lut,
                            _phantom: PhantomData,
                            _phantom1: PhantomData,
                            weights: BarycentricWeight::<i16>::create_ranged_256::<GRID_SIZE>(),
                        },
                    )
                }
                #[cfg(feature = "options")]
                BarycentricWeightScale::High => Arc::new(TransformLut4To3Q0_15::<
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
                    weights: BarycentricWeight::<i16>::create_binned::<GRID_SIZE, 65536>(),
                }),
            };
        }

        match options.barycentric_weight_scale {
            BarycentricWeightScale::Low => {
                Arc::new(
                    TransformLut4To3::<T, u8, LAYOUT, GRID_SIZE, BIT_DEPTH, 256, 256> {
                        lut,
                        _phantom: PhantomData,
                        _phantom1: PhantomData,
                        interpolation_method: options.interpolation_method,
                        weights: BarycentricWeight::<f32>::create_ranged_256::<GRID_SIZE>(),
                        color_space,
                        is_linear,
                    },
                )
            }
            #[cfg(feature = "options")]
            BarycentricWeightScale::High => {
                Arc::new(
                    TransformLut4To3::<T, u16, LAYOUT, GRID_SIZE, BIT_DEPTH, 65536, 65536> {
                        lut,
                        _phantom: PhantomData,
                        _phantom1: PhantomData,
                        interpolation_method: options.interpolation_method,
                        weights: BarycentricWeight::<f32>::create_binned::<GRID_SIZE, 65536>(),
                        color_space,
                        is_linear,
                    },
                )
            }
        }
    }
}
