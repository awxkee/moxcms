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
use crate::conversions::neon::stages::NeonAlignedF32;
use crate::math::FusedMultiplyAdd;
use crate::rounding_div_ceil;
use std::arch::aarch64::*;
use std::ops::{Add, Sub};

pub(crate) struct TetrahedralNeon<'a, const GRID_SIZE: usize> {
    pub(crate) cube: &'a [NeonAlignedF32],
}

pub(crate) struct PyramidalNeon<'a, const GRID_SIZE: usize> {
    pub(crate) cube: &'a [NeonAlignedF32],
}

pub(crate) struct PyramidalNeonDouble<'a, const GRID_SIZE: usize> {
    pub(crate) cube0: &'a [NeonAlignedF32],
    pub(crate) cube1: &'a [NeonAlignedF32],
}

pub(crate) struct PrismaticNeonDouble<'a, const GRID_SIZE: usize> {
    pub(crate) cube0: &'a [NeonAlignedF32],
    pub(crate) cube1: &'a [NeonAlignedF32],
}

pub(crate) struct TetrahedralNeonDouble<'a, const GRID_SIZE: usize> {
    pub(crate) cube0: &'a [NeonAlignedF32],
    pub(crate) cube1: &'a [NeonAlignedF32],
}

pub(crate) struct PrismaticNeon<'a, const GRID_SIZE: usize> {
    pub(crate) cube: &'a [NeonAlignedF32],
}

trait Fetcher<T> {
    fn fetch(&self, x: i32, y: i32, z: i32) -> T;
}

struct TetrahedralNeonFetchVector<'a, const GRID_SIZE: usize> {
    cube: &'a [NeonAlignedF32],
}

#[derive(Copy, Clone)]
pub(crate) struct NeonVector {
    pub(crate) v: float32x4_t,
}

impl From<f32> for NeonVector {
    #[inline(always)]
    fn from(v: f32) -> Self {
        NeonVector {
            v: unsafe { vdupq_n_f32(v) },
        }
    }
}

impl Sub<NeonVector> for NeonVector {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: NeonVector) -> Self::Output {
        NeonVector {
            v: unsafe { vsubq_f32(self.v, rhs.v) },
        }
    }
}

impl Add<NeonVector> for NeonVector {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: NeonVector) -> Self::Output {
        NeonVector {
            v: unsafe { vaddq_f32(self.v, rhs.v) },
        }
    }
}

impl FusedMultiplyAdd<NeonVector> for NeonVector {
    #[inline(always)]
    fn mla(&self, b: NeonVector, c: NeonVector) -> NeonVector {
        NeonVector {
            v: unsafe { vfmaq_f32(self.v, b.v, c.v) },
        }
    }
}

impl<const GRID_SIZE: usize> Fetcher<NeonVector> for TetrahedralNeonFetchVector<'_, GRID_SIZE> {
    fn fetch(&self, x: i32, y: i32, z: i32) -> NeonVector {
        let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
            + y as u32 * GRID_SIZE as u32
            + z as u32) as usize;
        let jx = unsafe { self.cube.get_unchecked(offset..) };
        NeonVector {
            v: unsafe { vld1q_f32(jx.as_ptr() as *const f32) },
        }
    }
}

pub(crate) trait NeonMdInterpolation<'a, const GRID_SIZE: usize> {
    fn new(table: &'a [NeonAlignedF32]) -> Self;
    fn inter3_neon(&self, in_r: u8, in_g: u8, in_b: u8) -> NeonVector;
}

pub(crate) trait NeonMdInterpolationDouble<'a, const GRID_SIZE: usize> {
    fn new(table0: &'a [NeonAlignedF32], table1: &'a [NeonAlignedF32]) -> Self;
    fn inter3_neon(&self, in_r: u8, in_g: u8, in_b: u8) -> (NeonVector, NeonVector);
}

impl<const GRID_SIZE: usize> TetrahedralNeon<'_, GRID_SIZE> {
    #[inline(always)]
    fn interpolate(&self, in_r: u8, in_g: u8, in_b: u8, r: impl Fetcher<NeonVector>) -> NeonVector {
        const SCALE: f32 = 1.0 / 255.0;
        let x: i32 = in_r as i32 * (GRID_SIZE as i32 - 1) / 255;
        let y: i32 = in_g as i32 * (GRID_SIZE as i32 - 1) / 255;
        let z: i32 = in_b as i32 * (GRID_SIZE as i32 - 1) / 255;

        let c0 = r.fetch(x, y, z);

        let x_n: i32 = rounding_div_ceil(in_r as i32 * (GRID_SIZE as i32 - 1), 255);
        let y_n: i32 = rounding_div_ceil(in_g as i32 * (GRID_SIZE as i32 - 1), 255);
        let z_n: i32 = rounding_div_ceil(in_b as i32 * (GRID_SIZE as i32 - 1), 255);

        let scale = (GRID_SIZE as i32 - 1) as f32 * SCALE;

        let rx = in_r as f32 * scale - x as f32;
        let ry = in_g as f32 * scale - y as f32;
        let rz = in_b as f32 * scale - z as f32;

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
        let s0 = c0.mla(c1, NeonVector::from(rx));
        let s1 = s0.mla(c2, NeonVector::from(ry));
        s1.mla(c3, NeonVector::from(rz))
    }
}

impl<const GRID_SIZE: usize> TetrahedralNeonDouble<'_, GRID_SIZE> {
    #[inline(always)]
    fn interpolate(
        &self,
        in_r: u8,
        in_g: u8,
        in_b: u8,
        r0: impl Fetcher<NeonVector>,
        r1: impl Fetcher<NeonVector>,
    ) -> (NeonVector, NeonVector) {
        const SCALE: f32 = 1.0 / 255.0;
        let x: i32 = in_r as i32 * (GRID_SIZE as i32 - 1) / 255;
        let y: i32 = in_g as i32 * (GRID_SIZE as i32 - 1) / 255;
        let z: i32 = in_b as i32 * (GRID_SIZE as i32 - 1) / 255;

        let c0_0 = r0.fetch(x, y, z);
        let c0_1 = r1.fetch(x, y, z);

        let x_n: i32 = rounding_div_ceil(in_r as i32 * (GRID_SIZE as i32 - 1), 255);
        let y_n: i32 = rounding_div_ceil(in_g as i32 * (GRID_SIZE as i32 - 1), 255);
        let z_n: i32 = rounding_div_ceil(in_b as i32 * (GRID_SIZE as i32 - 1), 255);

        let scale = (GRID_SIZE as i32 - 1) as f32 * SCALE;

        let rx = in_r as f32 * scale - x as f32;
        let ry = in_g as f32 * scale - y as f32;
        let rz = in_b as f32 * scale - z as f32;

        let c2_0;
        let c1_0;
        let c3_0;

        let c2_1;
        let c1_1;
        let c3_1;
        if rx >= ry {
            if ry >= rz {
                //rx >= ry && ry >= rz
                c1_0 = r0.fetch(x_n, y, z) - c0_0;
                c2_0 = r0.fetch(x_n, y_n, z) - r0.fetch(x_n, y, z);
                c3_0 = r0.fetch(x_n, y_n, z_n) - r0.fetch(x_n, y_n, z);

                c1_1 = r1.fetch(x_n, y, z) - c0_1;
                c2_1 = r1.fetch(x_n, y_n, z) - r1.fetch(x_n, y, z);
                c3_1 = r1.fetch(x_n, y_n, z_n) - r1.fetch(x_n, y_n, z);
            } else if rx >= rz {
                //rx >= rz && rz >= ry
                c1_0 = r0.fetch(x_n, y, z) - c0_0;
                c2_0 = r0.fetch(x_n, y_n, z_n) - r0.fetch(x_n, y, z_n);
                c3_0 = r0.fetch(x_n, y, z_n) - r0.fetch(x_n, y, z);

                c1_1 = r1.fetch(x_n, y, z) - c0_1;
                c2_1 = r1.fetch(x_n, y_n, z_n) - r1.fetch(x_n, y, z_n);
                c3_1 = r1.fetch(x_n, y, z_n) - r1.fetch(x_n, y, z);
            } else {
                //rz > rx && rx >= ry
                c1_0 = r0.fetch(x_n, y, z_n) - r0.fetch(x, y, z_n);
                c2_0 = r0.fetch(x_n, y_n, z_n) - r0.fetch(x_n, y, z_n);
                c3_0 = r0.fetch(x, y, z_n) - c0_0;

                c1_1 = r1.fetch(x_n, y, z_n) - r1.fetch(x, y, z_n);
                c2_1 = r1.fetch(x_n, y_n, z_n) - r1.fetch(x_n, y, z_n);
                c3_1 = r1.fetch(x, y, z_n) - c0_1;
            }
        } else if rx >= rz {
            //ry > rx && rx >= rz
            c1_0 = r0.fetch(x_n, y_n, z) - r0.fetch(x, y_n, z);
            c2_0 = r0.fetch(x, y_n, z) - c0_0;
            c3_0 = r0.fetch(x_n, y_n, z_n) - r0.fetch(x_n, y_n, z);

            c1_1 = r1.fetch(x_n, y_n, z) - r1.fetch(x, y_n, z);
            c2_1 = r1.fetch(x, y_n, z) - c0_1;
            c3_1 = r1.fetch(x_n, y_n, z_n) - r1.fetch(x_n, y_n, z);
        } else if ry >= rz {
            //ry >= rz && rz > rx
            c1_0 = r0.fetch(x_n, y_n, z_n) - r0.fetch(x, y_n, z_n);
            c2_0 = r0.fetch(x, y_n, z) - c0_0;
            c3_0 = r0.fetch(x, y_n, z_n) - r0.fetch(x, y_n, z);

            c1_1 = r1.fetch(x_n, y_n, z_n) - r1.fetch(x, y_n, z_n);
            c2_1 = r1.fetch(x, y_n, z) - c0_1;
            c3_1 = r1.fetch(x, y_n, z_n) - r1.fetch(x, y_n, z);
        } else {
            //rz > ry && ry > rx
            c1_0 = r0.fetch(x_n, y_n, z_n) - r0.fetch(x, y_n, z_n);
            c2_0 = r0.fetch(x, y_n, z_n) - r0.fetch(x, y, z_n);
            c3_0 = r0.fetch(x, y, z_n) - c0_0;

            c1_1 = r1.fetch(x_n, y_n, z_n) - r1.fetch(x, y_n, z_n);
            c2_1 = r1.fetch(x, y_n, z_n) - r1.fetch(x, y, z_n);
            c3_1 = r1.fetch(x, y, z_n) - c0_1;
        }

        let w0 = NeonVector::from(rx);
        let w1 = NeonVector::from(ry);
        let w2 = NeonVector::from(rz);

        let s0 = c0_0.mla(c1_0, w0);
        let s1 = s0.mla(c2_0, w1);
        let v0 = s1.mla(c3_0, w2);

        let s1 = c0_1.mla(c1_1, w0);
        let s1 = s1.mla(c2_1, w1);
        let v1 = s1.mla(c3_1, w2);
        (v0, v1)
    }
}

macro_rules! define_md_inter_neon {
    ($interpolator: ident) => {
        impl<'a, const GRID_SIZE: usize> NeonMdInterpolation<'a, GRID_SIZE>
            for $interpolator<'a, GRID_SIZE>
        {
            #[inline(always)]
            fn new(table: &'a [NeonAlignedF32]) -> Self {
                Self { cube: table }
            }

            #[inline(always)]
            fn inter3_neon(&self, in_r: u8, in_g: u8, in_b: u8) -> NeonVector {
                self.interpolate(
                    in_r,
                    in_g,
                    in_b,
                    TetrahedralNeonFetchVector::<GRID_SIZE> { cube: self.cube },
                )
            }
        }
    };
}

macro_rules! define_md_inter_neon_d {
    ($interpolator: ident) => {
        impl<'a, const GRID_SIZE: usize> NeonMdInterpolationDouble<'a, GRID_SIZE>
            for $interpolator<'a, GRID_SIZE>
        {
            #[inline(always)]
            fn new(table0: &'a [NeonAlignedF32], table1: &'a [NeonAlignedF32]) -> Self {
                Self {
                    cube0: table0,
                    cube1: table1,
                }
            }

            #[inline(always)]
            fn inter3_neon(&self, in_r: u8, in_g: u8, in_b: u8) -> (NeonVector, NeonVector) {
                self.interpolate(
                    in_r,
                    in_g,
                    in_b,
                    TetrahedralNeonFetchVector::<GRID_SIZE> { cube: self.cube0 },
                    TetrahedralNeonFetchVector::<GRID_SIZE> { cube: self.cube1 },
                )
            }
        }
    };
}

define_md_inter_neon!(TetrahedralNeon);
define_md_inter_neon!(PyramidalNeon);
define_md_inter_neon!(PrismaticNeon);
define_md_inter_neon_d!(PrismaticNeonDouble);
define_md_inter_neon_d!(PyramidalNeonDouble);
define_md_inter_neon_d!(TetrahedralNeonDouble);

impl<const GRID_SIZE: usize> PyramidalNeon<'_, GRID_SIZE> {
    #[inline(always)]
    fn interpolate(&self, in_r: u8, in_g: u8, in_b: u8, r: impl Fetcher<NeonVector>) -> NeonVector {
        const SCALE: f32 = 1.0 / 255.0;
        let x: i32 = in_r as i32 * (GRID_SIZE as i32 - 1) / 255;
        let y: i32 = in_g as i32 * (GRID_SIZE as i32 - 1) / 255;
        let z: i32 = in_b as i32 * (GRID_SIZE as i32 - 1) / 255;

        let c0 = r.fetch(x, y, z);

        let x_n: i32 = rounding_div_ceil(in_r as i32 * (GRID_SIZE as i32 - 1), 255);
        let y_n: i32 = rounding_div_ceil(in_g as i32 * (GRID_SIZE as i32 - 1), 255);
        let z_n: i32 = rounding_div_ceil(in_b as i32 * (GRID_SIZE as i32 - 1), 255);

        let scale = (GRID_SIZE as i32 - 1) as f32 * SCALE;

        let dr = in_r as f32 * scale - x as f32;
        let dg = in_g as f32 * scale - y as f32;
        let db = in_b as f32 * scale - z as f32;

        if dr > db && dg > db {
            let x0 = r.fetch(x_n, y_n, z_n);
            let x1 = r.fetch(x_n, y_n, z);
            let x2 = r.fetch(x_n, y, z);
            let x3 = r.fetch(x, y_n, z);

            let c1 = x0 - x1;
            let c2 = x2 - c0;
            let c3 = x3 - c0;
            let c4 = c0 - x3 - x2 + x1;

            let s0 = c0.mla(c1, NeonVector::from(db));
            let s1 = s0.mla(c2, NeonVector::from(dr));
            let s2 = s1.mla(c3, NeonVector::from(dg));
            s2.mla(c4, NeonVector::from(dr * dg))
        } else if db > dr && dg > dr {
            let x0 = r.fetch(x, y, z_n);
            let x1 = r.fetch(x_n, y_n, z_n);
            let x2 = r.fetch(x, y_n, z_n);
            let x3 = r.fetch(x, y_n, z);

            let c1 = x0 - c0;
            let c2 = x1 - x2;
            let c3 = x3 - c0;
            let c4 = c0 - x3 - x0 + x2;

            let s0 = c0.mla(c1, NeonVector::from(db));
            let s1 = s0.mla(c2, NeonVector::from(dr));
            let s2 = s1.mla(c3, NeonVector::from(dg));
            s2.mla(c4, NeonVector::from(dg * db))
        } else {
            let x0 = r.fetch(x, y, z_n);
            let x1 = r.fetch(x_n, y, z);
            let x2 = r.fetch(x_n, y_n, z);
            let x3 = r.fetch(x_n, y, z_n);

            let c1 = x0 - c0;
            let c2 = x1 - c0;
            let c3 = x2 - x3;
            let c4 = c0 - x1 - x0 + x3;

            let s0 = c0.mla(c1, NeonVector::from(db));
            let s1 = s0.mla(c2, NeonVector::from(dr));
            let s2 = s1.mla(c3, NeonVector::from(dg));
            s2.mla(c4, NeonVector::from(db * dr))
        }
    }
}

impl<const GRID_SIZE: usize> PyramidalNeonDouble<'_, GRID_SIZE> {
    #[inline(always)]
    fn interpolate(
        &self,
        in_r: u8,
        in_g: u8,
        in_b: u8,
        r0: impl Fetcher<NeonVector>,
        r1: impl Fetcher<NeonVector>,
    ) -> (NeonVector, NeonVector) {
        const SCALE: f32 = 1.0 / 255.0;
        let x: i32 = in_r as i32 * (GRID_SIZE as i32 - 1) / 255;
        let y: i32 = in_g as i32 * (GRID_SIZE as i32 - 1) / 255;
        let z: i32 = in_b as i32 * (GRID_SIZE as i32 - 1) / 255;

        let c0_0 = r0.fetch(x, y, z);
        let c0_1 = r1.fetch(x, y, z);

        let x_n: i32 = rounding_div_ceil(in_r as i32 * (GRID_SIZE as i32 - 1), 255);
        let y_n: i32 = rounding_div_ceil(in_g as i32 * (GRID_SIZE as i32 - 1), 255);
        let z_n: i32 = rounding_div_ceil(in_b as i32 * (GRID_SIZE as i32 - 1), 255);

        let scale = (GRID_SIZE as i32 - 1) as f32 * SCALE;

        let dr = in_r as f32 * scale - x as f32;
        let dg = in_g as f32 * scale - y as f32;
        let db = in_b as f32 * scale - z as f32;

        let w0 = NeonVector::from(db);
        let w1 = NeonVector::from(dr);
        let w2 = NeonVector::from(dg);

        if dr > db && dg > db {
            let x0_0 = r0.fetch(x_n, y_n, z_n);
            let x1_0 = r0.fetch(x_n, y_n, z);
            let x2_0 = r0.fetch(x_n, y, z);
            let x3_0 = r0.fetch(x, y_n, z);

            let x0_1 = r1.fetch(x_n, y_n, z_n);
            let x1_1 = r1.fetch(x_n, y_n, z);
            let x2_1 = r1.fetch(x_n, y, z);
            let x3_1 = r1.fetch(x, y_n, z);

            let c1_0 = x0_0 - x1_0;
            let c2_0 = x2_0 - c0_0;
            let c3_0 = x3_0 - c0_0;
            let c4_0 = c0_0 - x3_0 - x2_0 + x1_0;

            let c1_1 = x0_1 - x1_1;
            let c2_1 = x2_1 - c0_1;
            let c3_1 = x3_1 - c0_1;
            let c4_1 = c0_1 - x3_1 - x2_1 + x1_1;

            let w3 = NeonVector::from(dr * dg);

            let s0_0 = c0_0.mla(c1_0, w0);
            let s1_0 = s0_0.mla(c2_0, w1);
            let s2_0 = s1_0.mla(c3_0, w2);
            let v0 = s2_0.mla(c4_0, w3);

            let s0_1 = c0_1.mla(c1_1, w0);
            let s1_1 = s0_1.mla(c2_1, w1);
            let s2_1 = s1_1.mla(c3_1, w2);
            let v1 = s2_1.mla(c4_1, w3);
            (v0, v1)
        } else if db > dr && dg > dr {
            let x0_0 = r0.fetch(x, y, z_n);
            let x1_0 = r0.fetch(x_n, y_n, z_n);
            let x2_0 = r0.fetch(x, y_n, z_n);
            let x3_0 = r0.fetch(x, y_n, z);

            let x0_1 = r1.fetch(x, y, z_n);
            let x1_1 = r1.fetch(x_n, y_n, z_n);
            let x2_1 = r1.fetch(x, y_n, z_n);
            let x3_1 = r1.fetch(x, y_n, z);

            let c1_0 = x0_0 - c0_0;
            let c2_0 = x1_0 - x2_0;
            let c3_0 = x3_0 - c0_0;
            let c4_0 = c0_0 - x3_0 - x0_0 + x2_0;

            let c1_1 = x0_1 - c0_1;
            let c2_1 = x1_1 - x2_1;
            let c3_1 = x3_1 - c0_1;
            let c4_1 = c0_1 - x3_1 - x0_1 + x2_1;

            let w3 = NeonVector::from(dg * db);

            let s0_0 = c0_0.mla(c1_0, w0);
            let s1_0 = s0_0.mla(c2_0, w1);
            let s2_0 = s1_0.mla(c3_0, w2);
            let v0 = s2_0.mla(c4_0, w3);

            let s0_1 = c0_1.mla(c1_1, w0);
            let s1_1 = s0_1.mla(c2_1, w1);
            let s2_1 = s1_1.mla(c3_1, w2);
            let v1 = s2_1.mla(c4_1, w3);
            (v0, v1)
        } else {
            let x0_0 = r0.fetch(x, y, z_n);
            let x1_0 = r0.fetch(x_n, y, z);
            let x2_0 = r0.fetch(x_n, y_n, z);
            let x3_0 = r0.fetch(x_n, y, z_n);

            let x0_1 = r1.fetch(x, y, z_n);
            let x1_1 = r1.fetch(x_n, y, z);
            let x2_1 = r1.fetch(x_n, y_n, z);
            let x3_1 = r1.fetch(x_n, y, z_n);

            let c1_0 = x0_0 - c0_0;
            let c2_0 = x1_0 - c0_0;
            let c3_0 = x2_0 - x3_0;
            let c4_0 = c0_0 - x1_0 - x0_0 + x3_0;

            let c1_1 = x0_1 - c0_1;
            let c2_1 = x1_1 - c0_1;
            let c3_1 = x2_1 - x3_1;
            let c4_1 = c0_1 - x1_1 - x0_1 + x3_1;

            let w3 = NeonVector::from(db * dr);

            let s0_0 = c0_0.mla(c1_0, w0);
            let s1_0 = s0_0.mla(c2_0, w1);
            let s2_0 = s1_0.mla(c3_0, w2);
            let v0 = s2_0.mla(c4_0, w3);

            let s0 = c0_1.mla(c1_1, w0);
            let s1 = s0.mla(c2_1, w1);
            let s2 = s1.mla(c3_1, w2);
            let v1 = s2.mla(c4_1, w3);
            (v0, v1)
        }
    }
}

impl<const GRID_SIZE: usize> PrismaticNeon<'_, GRID_SIZE> {
    #[inline(always)]
    fn interpolate(&self, in_r: u8, in_g: u8, in_b: u8, r: impl Fetcher<NeonVector>) -> NeonVector {
        const SCALE: f32 = 1.0 / 255.0;
        let x: i32 = in_r as i32 * (GRID_SIZE as i32 - 1) / 255;
        let y: i32 = in_g as i32 * (GRID_SIZE as i32 - 1) / 255;
        let z: i32 = in_b as i32 * (GRID_SIZE as i32 - 1) / 255;

        let c0 = r.fetch(x, y, z);

        let x_n: i32 = rounding_div_ceil(in_r as i32 * (GRID_SIZE as i32 - 1), 255);
        let y_n: i32 = rounding_div_ceil(in_g as i32 * (GRID_SIZE as i32 - 1), 255);
        let z_n: i32 = rounding_div_ceil(in_b as i32 * (GRID_SIZE as i32 - 1), 255);

        let scale = (GRID_SIZE as i32 - 1) as f32 * SCALE;

        let dr = in_r as f32 * scale - x as f32;
        let dg = in_g as f32 * scale - y as f32;
        let db = in_b as f32 * scale - z as f32;

        if db > dr {
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

            let s0 = c0.mla(c1, NeonVector::from(db));
            let s1 = s0.mla(c2, NeonVector::from(dr));
            let s2 = s1.mla(c3, NeonVector::from(dg));
            let s3 = s2.mla(c4, NeonVector::from(dg * db));
            s3.mla(c5, NeonVector::from(dr * dg))
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

            let s0 = c0.mla(c1, NeonVector::from(db));
            let s1 = s0.mla(c2, NeonVector::from(dr));
            let s2 = s1.mla(c3, NeonVector::from(dg));
            let s3 = s2.mla(c4, NeonVector::from(dg * db));
            s3.mla(c5, NeonVector::from(dr * dg))
        }
    }
}

impl<const GRID_SIZE: usize> PrismaticNeonDouble<'_, GRID_SIZE> {
    #[inline(always)]
    fn interpolate(
        &self,
        in_r: u8,
        in_g: u8,
        in_b: u8,
        r0: impl Fetcher<NeonVector>,
        r1: impl Fetcher<NeonVector>,
    ) -> (NeonVector, NeonVector) {
        const SCALE: f32 = 1.0 / 255.0;
        let x: i32 = in_r as i32 * (GRID_SIZE as i32 - 1) / 255;
        let y: i32 = in_g as i32 * (GRID_SIZE as i32 - 1) / 255;
        let z: i32 = in_b as i32 * (GRID_SIZE as i32 - 1) / 255;

        let c0_0 = r0.fetch(x, y, z);
        let c0_1 = r1.fetch(x, y, z);

        let x_n: i32 = rounding_div_ceil(in_r as i32 * (GRID_SIZE as i32 - 1), 255);
        let y_n: i32 = rounding_div_ceil(in_g as i32 * (GRID_SIZE as i32 - 1), 255);
        let z_n: i32 = rounding_div_ceil(in_b as i32 * (GRID_SIZE as i32 - 1), 255);

        let scale = (GRID_SIZE as i32 - 1) as f32 * SCALE;

        let dr = in_r as f32 * scale - x as f32;
        let dg = in_g as f32 * scale - y as f32;
        let db = in_b as f32 * scale - z as f32;

        let w0 = NeonVector::from(db);
        let w1 = NeonVector::from(dr);
        let w2 = NeonVector::from(dg);
        let w3 = NeonVector::from(dg * db);
        let w4 = NeonVector::from(dr * dg);

        if db > dr {
            let x0_0 = r0.fetch(x, y, z_n);
            let x1_0 = r0.fetch(x_n, y, z_n);
            let x2_0 = r0.fetch(x, y_n, z);
            let x3_0 = r0.fetch(x, y_n, z_n);
            let x4_0 = r0.fetch(x_n, y_n, z_n);

            let x0_1 = r1.fetch(x, y, z_n);
            let x1_1 = r1.fetch(x_n, y, z_n);
            let x2_1 = r1.fetch(x, y_n, z);
            let x3_1 = r1.fetch(x, y_n, z_n);
            let x4_1 = r1.fetch(x_n, y_n, z_n);

            let c1_0 = x0_0 - c0_0;
            let c2_0 = x1_0 - x0_0;
            let c3_0 = x2_0 - c0_0;
            let c4_0 = c0_0 - x2_0 - x0_0 + x3_0;
            let c5_0 = x0_0 - x3_0 - x1_0 + x4_0;

            let c1_1 = x0_1 - c0_1;
            let c2_1 = x1_1 - x0_1;
            let c3_1 = x2_1 - c0_1;
            let c4_1 = c0_1 - x2_1 - x0_1 + x3_1;
            let c5_1 = x0_1 - x3_1 - x1_1 + x4_1;

            let s0_0 = c0_0.mla(c1_0, w0);
            let s1_0 = s0_0.mla(c2_0, w1);
            let s2_0 = s1_0.mla(c3_0, w2);
            let s3_0 = s2_0.mla(c4_0, w3);
            let v0 = s3_0.mla(c5_0, w4);

            let s0_1 = c0_1.mla(c1_1, w0);
            let s1_1 = s0_1.mla(c2_1, w1);
            let s2_1 = s1_1.mla(c3_1, w2);
            let s3_1 = s2_1.mla(c4_1, w3);
            let v1 = s3_1.mla(c5_1, w4);
            (v0, v1)
        } else {
            let x0_0 = r0.fetch(x_n, y, z);
            let x1_0 = r0.fetch(x_n, y, z_n);
            let x2_0 = r0.fetch(x, y_n, z);
            let x3_0 = r0.fetch(x_n, y_n, z);
            let x4_0 = r0.fetch(x_n, y_n, z_n);

            let x0_1 = r1.fetch(x_n, y, z);
            let x1_1 = r1.fetch(x_n, y, z_n);
            let x2_1 = r1.fetch(x, y_n, z);
            let x3_1 = r1.fetch(x_n, y_n, z);
            let x4_1 = r1.fetch(x_n, y_n, z_n);

            let c1_0 = x1_0 - x0_0;
            let c2_0 = x0_0 - c0_0;
            let c3_0 = x2_0 - c0_0;
            let c4_0 = x0_0 - x3_0 - x1_0 + x4_0;
            let c5_0 = c0_0 - x2_0 - x0_0 + x3_0;

            let c1_1 = x1_1 - x0_1;
            let c2_1 = x0_1 - c0_1;
            let c3_1 = x2_1 - c0_1;
            let c4_1 = x0_1 - x3_1 - x1_1 + x4_1;
            let c5_1 = c0_1 - x2_1 - x0_1 + x3_1;

            let s0_0 = c0_0.mla(c1_0, w0);
            let s1_0 = s0_0.mla(c2_0, w1);
            let s2_0 = s1_0.mla(c3_0, w2);
            let s3_0 = s2_0.mla(c4_0, w3);
            let v0 = s3_0.mla(c5_0, w4);

            let s0_1 = c0_1.mla(c1_1, w0);
            let s1_1 = s0_1.mla(c2_1, w1);
            let s2_1 = s1_1.mla(c3_1, w2);
            let s3_1 = s2_1.mla(c4_1, w3);
            let v1 = s3_1.mla(c5_1, w4);
            (v0, v1)
        }
    }
}
