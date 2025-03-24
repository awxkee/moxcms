/*
 * // Copyright (c) Radzivon Bartoshyk 2/2025. All rights reserved.
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
use crate::conversions::lut_transforms::LUT_SAMPLING;
use crate::math::{FusedMultiplyAdd, FusedMultiplyNegAdd};
use crate::{Vector3f, Vector4f, rounding_div_ceil};
use std::ops::{Add, Mul, Sub};

pub(crate) struct Tetrahedral<'a, const GRID_SIZE: usize> {
    pub(crate) cube: &'a [f32],
}

pub(crate) struct Pyramidal<'a, const GRID_SIZE: usize> {
    pub(crate) cube: &'a [f32],
}

pub(crate) struct Prismatic<'a, const GRID_SIZE: usize> {
    pub(crate) cube: &'a [f32],
}

pub(crate) struct Trilinear<'a, const GRID_SIZE: usize> {
    pub(crate) cube: &'a [f32],
}

trait Fetcher<T> {
    fn fetch(&self, x: i32, y: i32, z: i32) -> T;
}

struct TetrahedralFetchVector3f<'a, const GRID_SIZE: usize> {
    cube: &'a [f32],
}

pub(crate) trait MultidimensionalInterpolation<'a, const GRID_SIZE: usize> {
    fn new(table: &'a [f32]) -> Self;
    fn inter3(&self, in_r: u8, in_g: u8, in_b: u8) -> Vector3f;
    fn inter4(&self, in_r: u8, in_g: u8, in_b: u8) -> Vector4f;
}

impl<const GRID_SIZE: usize> Fetcher<Vector3f> for TetrahedralFetchVector3f<'_, GRID_SIZE> {
    #[inline(always)]
    fn fetch(&self, x: i32, y: i32, z: i32) -> Vector3f {
        let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
            + y as u32 * GRID_SIZE as u32
            + z as u32) as usize
            * 3;
        let jx = &self.cube[offset..offset + 3];
        Vector3f {
            v: [jx[0], jx[1], jx[2]],
        }
    }
}

struct TetrahedralFetchVector4f<'a, const GRID_SIZE: usize> {
    cube: &'a [f32],
}

impl<const GRID_SIZE: usize> Fetcher<Vector4f> for TetrahedralFetchVector4f<'_, GRID_SIZE> {
    #[inline(always)]
    fn fetch(&self, x: i32, y: i32, z: i32) -> Vector4f {
        let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
            + y as u32 * GRID_SIZE as u32
            + z as u32) as usize
            * 4;
        let jx = &self.cube[offset..offset + 4];
        Vector4f {
            v: [jx[0], jx[1], jx[2], jx[3]],
        }
    }
}

impl<const GRID_SIZE: usize> Tetrahedral<'_, GRID_SIZE> {
    #[inline]
    fn interpolate<
        T: Copy
            + Sub<T, Output = T>
            + Mul<T, Output = T>
            + Mul<f32, Output = T>
            + Add<T, Output = T>
            + From<f32>
            + FusedMultiplyAdd<T>,
    >(
        &self,
        in_r: u8,
        in_g: u8,
        in_b: u8,
        r: impl Fetcher<T>,
    ) -> T {
        const SCALE: f32 = 1.0 / LUT_SAMPLING as f32;
        let x: i32 = in_r as i32 * (GRID_SIZE as i32 - 1) / LUT_SAMPLING as i32;
        let y: i32 = in_g as i32 * (GRID_SIZE as i32 - 1) / LUT_SAMPLING as i32;
        let z: i32 = in_b as i32 * (GRID_SIZE as i32 - 1) / LUT_SAMPLING as i32;
        let x_n: i32 = rounding_div_ceil(in_r as i32 * (GRID_SIZE as i32 - 1), LUT_SAMPLING as i32);
        let y_n: i32 = rounding_div_ceil(in_g as i32 * (GRID_SIZE as i32 - 1), LUT_SAMPLING as i32);
        let z_n: i32 = rounding_div_ceil(in_b as i32 * (GRID_SIZE as i32 - 1), LUT_SAMPLING as i32);
        let rx = in_r as f32 * ((GRID_SIZE as i32 - 1) as f32 * SCALE) - x as f32;
        let ry = in_g as f32 * ((GRID_SIZE as i32 - 1) as f32 * SCALE) - y as f32;
        let rz = in_b as f32 * ((GRID_SIZE as i32 - 1) as f32 * SCALE) - z as f32;
        let c0 = r.fetch(x, y, z);
        let c2;
        let c1;
        let c3;
        if rx >= ry {
            if ry >= rz {
                //rx >= ry && ry >= rz
                c1 = r.fetch(x_n, y, z) - c0;
                c2 = r.fetch(x_n, y_n, z) - r.fetch(x_n, y, z);
                c3 = r.fetch(x_n, y_n, z_n) - r.fetch(x_n, y_n, z);
            } else if rx >= rz {
                //rx >= rz && rz >= ry
                c1 = r.fetch(x_n, y, z) - c0;
                c2 = r.fetch(x_n, y_n, z_n) - r.fetch(x_n, y, z_n);
                c3 = r.fetch(x_n, y, z_n) - r.fetch(x_n, y, z);
            } else {
                //rz > rx && rx >= ry
                c1 = r.fetch(x_n, y, z_n) - r.fetch(x, y, z_n);
                c2 = r.fetch(x_n, y_n, z_n) - r.fetch(x_n, y, z_n);
                c3 = r.fetch(x, y, z_n) - c0;
            }
        } else if rx >= rz {
            //ry > rx && rx >= rz
            c1 = r.fetch(x_n, y_n, z) - r.fetch(x, y_n, z);
            c2 = r.fetch(x, y_n, z) - c0;
            c3 = r.fetch(x_n, y_n, z_n) - r.fetch(x_n, y_n, z);
        } else if ry >= rz {
            //ry >= rz && rz > rx
            c1 = r.fetch(x_n, y_n, z_n) - r.fetch(x, y_n, z_n);
            c2 = r.fetch(x, y_n, z) - c0;
            c3 = r.fetch(x, y_n, z_n) - r.fetch(x, y_n, z);
        } else {
            //rz > ry && ry > rx
            c1 = r.fetch(x_n, y_n, z_n) - r.fetch(x, y_n, z_n);
            c2 = r.fetch(x, y_n, z_n) - r.fetch(x, y, z_n);
            c3 = r.fetch(x, y, z_n) - c0;
        }
        let s0 = c0.mla(c1, T::from(rx));
        let s1 = s0.mla(c2, T::from(ry));
        s1.mla(c3, T::from(rz))
    }
}

macro_rules! define_md_inter {
    ($interpolator: ident) => {
        impl<'a, const GRID_SIZE: usize> MultidimensionalInterpolation<'a, GRID_SIZE>
            for $interpolator<'a, GRID_SIZE>
        {
            fn new(table: &'a [f32]) -> Self {
                Self { cube: table }
            }

            #[inline(always)]
            fn inter3(&self, in_r: u8, in_g: u8, in_b: u8) -> Vector3f {
                self.interpolate(
                    in_r,
                    in_g,
                    in_b,
                    TetrahedralFetchVector3f::<GRID_SIZE> { cube: self.cube },
                )
            }

            #[inline(always)]
            fn inter4(&self, in_r: u8, in_g: u8, in_b: u8) -> Vector4f {
                self.interpolate(
                    in_r,
                    in_g,
                    in_b,
                    TetrahedralFetchVector4f::<GRID_SIZE> { cube: self.cube },
                )
            }
        }
    };
}

define_md_inter!(Tetrahedral);
define_md_inter!(Pyramidal);
define_md_inter!(Prismatic);
define_md_inter!(Trilinear);

impl<const GRID_SIZE: usize> Pyramidal<'_, GRID_SIZE> {
    #[inline]
    fn interpolate<
        T: Copy
            + Sub<T, Output = T>
            + Mul<T, Output = T>
            + Mul<f32, Output = T>
            + Add<T, Output = T>
            + From<f32>
            + FusedMultiplyAdd<T>,
    >(
        &self,
        in_r: u8,
        in_g: u8,
        in_b: u8,
        r: impl Fetcher<T>,
    ) -> T {
        const SCALE: f32 = 1.0 / LUT_SAMPLING as f32;
        let x: i32 = in_r as i32 * (GRID_SIZE as i32 - 1) / LUT_SAMPLING as i32;
        let y: i32 = in_g as i32 * (GRID_SIZE as i32 - 1) / LUT_SAMPLING as i32;
        let z: i32 = in_b as i32 * (GRID_SIZE as i32 - 1) / LUT_SAMPLING as i32;
        let x_n: i32 = rounding_div_ceil(in_r as i32 * (GRID_SIZE as i32 - 1), LUT_SAMPLING as i32);
        let y_n: i32 = rounding_div_ceil(in_g as i32 * (GRID_SIZE as i32 - 1), LUT_SAMPLING as i32);
        let z_n: i32 = rounding_div_ceil(in_b as i32 * (GRID_SIZE as i32 - 1), LUT_SAMPLING as i32);
        let dr = in_r as f32 * ((GRID_SIZE as i32 - 1) as f32 * SCALE) - x as f32;
        let dg = in_g as f32 * ((GRID_SIZE as i32 - 1) as f32 * SCALE) - y as f32;
        let db = in_b as f32 * ((GRID_SIZE as i32 - 1) as f32 * SCALE) - z as f32;
        let c0 = r.fetch(x, y, z);

        if dr > db && dg > db {
            let x0 = r.fetch(x_n, y_n, z_n);
            let x1 = r.fetch(x_n, y_n, z);
            let x2 = r.fetch(x_n, y, z);
            let x3 = r.fetch(x, y_n, z);

            let c1 = x0 - x1;
            let c2 = x2 - c0;
            let c3 = x3 - c0;
            let c4 = c0 - x3 - x2 + x1;

            let s0 = c0.mla(c1, T::from(db));
            let s1 = s0.mla(c2, T::from(dr));
            let s2 = s1.mla(c3, T::from(dg));
            s2.mla(c4, T::from(dr * dg))
        } else if db > dr && dg > dr {
            let x0 = r.fetch(x, y, z_n);
            let x1 = r.fetch(x_n, y_n, z_n);
            let x2 = r.fetch(x, y_n, z_n);
            let x3 = r.fetch(x, y_n, z);

            let c1 = x0 - c0;
            let c2 = x1 - x2;
            let c3 = x3 - c0;
            let c4 = c0 - x3 - x0 + x2;

            let s0 = c0.mla(c1, T::from(db));
            let s1 = s0.mla(c2, T::from(dr));
            let s2 = s1.mla(c3, T::from(dg));
            s2.mla(c4, T::from(dg * db))
        } else {
            let x0 = r.fetch(x, y, z_n);
            let x1 = r.fetch(x_n, y, z);
            let x2 = r.fetch(x_n, y, z_n);
            let x3 = r.fetch(x_n, y_n, z_n);

            let c1 = x0 - c0;
            let c2 = x1 - c0;
            let c3 = x3 - x2;
            let c4 = c0 - x1 - x0 + x2;

            let s0 = c0.mla(c1, T::from(db));
            let s1 = s0.mla(c2, T::from(dr));
            let s2 = s1.mla(c3, T::from(dg));
            s2.mla(c4, T::from(db * dr))
        }
    }
}

impl<const GRID_SIZE: usize> Prismatic<'_, GRID_SIZE> {
    #[inline]
    fn interpolate<
        T: Copy
            + Sub<T, Output = T>
            + Mul<T, Output = T>
            + Mul<f32, Output = T>
            + Add<T, Output = T>
            + From<f32>
            + FusedMultiplyAdd<T>,
    >(
        &self,
        in_r: u8,
        in_g: u8,
        in_b: u8,
        r: impl Fetcher<T>,
    ) -> T {
        const SCALE: f32 = 1.0 / LUT_SAMPLING as f32;
        let x: i32 = in_r as i32 * (GRID_SIZE as i32 - 1) / LUT_SAMPLING as i32;
        let y: i32 = in_g as i32 * (GRID_SIZE as i32 - 1) / LUT_SAMPLING as i32;
        let z: i32 = in_b as i32 * (GRID_SIZE as i32 - 1) / LUT_SAMPLING as i32;
        let x_n: i32 = rounding_div_ceil(in_r as i32 * (GRID_SIZE as i32 - 1), LUT_SAMPLING as i32);
        let y_n: i32 = rounding_div_ceil(in_g as i32 * (GRID_SIZE as i32 - 1), LUT_SAMPLING as i32);
        let z_n: i32 = rounding_div_ceil(in_b as i32 * (GRID_SIZE as i32 - 1), LUT_SAMPLING as i32);
        let dr = in_r as f32 * ((GRID_SIZE as i32 - 1) as f32 * SCALE) - x as f32;
        let dg = in_g as f32 * ((GRID_SIZE as i32 - 1) as f32 * SCALE) - y as f32;
        let db = in_b as f32 * ((GRID_SIZE as i32 - 1) as f32 * SCALE) - z as f32;
        let c0 = r.fetch(x, y, z);

        if db >= dr {
            let x0 = r.fetch(x, y, z_n);
            let x1 = r.fetch(x_n, y, z_n);
            let x2 = r.fetch(x, y_n, z);
            let x3 = r.fetch(x, y_n, z_n);
            let x4 = r.fetch(x_n, y_n, z_n);

            let c1 = x0 - c0;
            let c2 = x1 - x0;
            let c3 = x2 - c0;
            let c4 = c0 - x2 - x0 + x3;
            let c5 = x0 - x3 - x1 + x4;

            let s0 = c0.mla(c1, T::from(db));
            let s1 = s0.mla(c2, T::from(dr));
            let s2 = s1.mla(c3, T::from(dg));
            let s3 = s2.mla(c4, T::from(dg * db));
            s3.mla(c5, T::from(dr * dg))
        } else {
            let x0 = r.fetch(x_n, y, z);
            let x1 = r.fetch(x_n, y, z_n);
            let x2 = r.fetch(x, y_n, z);
            let x3 = r.fetch(x_n, y_n, z);
            let x4 = r.fetch(x_n, y_n, z_n);

            let c1 = x1 - x0;
            let c2 = x0 - c0;
            let c3 = x2 - c0;
            let c4 = x0 - x3 - x1 + x4;
            let c5 = c0 - x2 - x0 + x3;

            let s0 = c0.mla(c1, T::from(db));
            let s1 = s0.mla(c2, T::from(dr));
            let s2 = s1.mla(c3, T::from(dg));
            let s3 = s2.mla(c4, T::from(dg * db));
            s3.mla(c5, T::from(dr * dg))
        }
    }
}

impl<const GRID_SIZE: usize> Trilinear<'_, GRID_SIZE> {
    #[inline]
    fn interpolate<
        T: Copy
            + Sub<T, Output = T>
            + Mul<T, Output = T>
            + Mul<f32, Output = T>
            + Add<T, Output = T>
            + From<f32>
            + FusedMultiplyAdd<T>
            + FusedMultiplyNegAdd<T>,
    >(
        &self,
        in_r: u8,
        in_g: u8,
        in_b: u8,
        r: impl Fetcher<T>,
    ) -> T {
        const SCALE: f32 = 1.0 / LUT_SAMPLING as f32;
        let x: i32 = in_r as i32 * (GRID_SIZE as i32 - 1) / LUT_SAMPLING as i32;
        let y: i32 = in_g as i32 * (GRID_SIZE as i32 - 1) / LUT_SAMPLING as i32;
        let z: i32 = in_b as i32 * (GRID_SIZE as i32 - 1) / LUT_SAMPLING as i32;
        let x_n: i32 = rounding_div_ceil(in_r as i32 * (GRID_SIZE as i32 - 1), LUT_SAMPLING as i32);
        let y_n: i32 = rounding_div_ceil(in_g as i32 * (GRID_SIZE as i32 - 1), LUT_SAMPLING as i32);
        let z_n: i32 = rounding_div_ceil(in_b as i32 * (GRID_SIZE as i32 - 1), LUT_SAMPLING as i32);
        let dr = in_r as f32 * ((GRID_SIZE as i32 - 1) as f32 * SCALE) - x as f32;
        let dg = in_g as f32 * ((GRID_SIZE as i32 - 1) as f32 * SCALE) - y as f32;
        let db = in_b as f32 * ((GRID_SIZE as i32 - 1) as f32 * SCALE) - z as f32;
        let w0 = T::from(dr);
        let w1 = T::from(dg);
        let w2 = T::from(db);

        let c000 = r.fetch(x, y, z);
        let c100 = r.fetch(x_n, y, z);
        let c010 = r.fetch(x, y_n, z);
        let c110 = r.fetch(x_n, y_n, z);
        let c001 = r.fetch(x, y, z_n);
        let c101 = r.fetch(x_n, y, z_n);
        let c011 = r.fetch(x, y_n, z_n);
        let c111 = r.fetch(x_n, y_n, z_n);

        let dx = T::from(dr);

        let c00 = c000.neg_mla(c000, dx).mla(c100, w0);
        let c10 = c010.neg_mla(c010, dx).mla(c110, w0);
        let c01 = c001.neg_mla(c001, dx).mla(c101, w0);
        let c11 = c011.neg_mla(c011, dx).mla(c111, w0);

        let dy = T::from(dg);

        let c0 = c00.neg_mla(c00, dy).mla(c10, w1);
        let c1 = c01.neg_mla(c01, dy).mla(c11, w1);

        let dz = T::from(db);

        c0.neg_mla(c0, dz).mla(c1, w2)
    }
}
