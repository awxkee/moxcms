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
use crate::math::FusedMultiplyAdd;
use crate::rounding_div_ceil;
use std::arch::aarch64::*;
use std::ops::{Add, Sub};

#[repr(align(16), C)]
pub(crate) struct NeonAlignedI16x4(pub(crate) [i16; 4]);

pub(crate) struct TetrahedralNeonQ2_14<'a, const GRID_SIZE: usize> {
    pub(crate) cube: &'a [NeonAlignedI16x4],
}

pub(crate) struct PyramidalNeonQ2_14<'a, const GRID_SIZE: usize> {
    pub(crate) cube: &'a [NeonAlignedI16x4],
}

pub(crate) struct PyramidalNeonQ2_14Double<'a, const GRID_SIZE: usize> {
    pub(crate) cube0: &'a [NeonAlignedI16x4],
    pub(crate) cube1: &'a [NeonAlignedI16x4],
}

pub(crate) struct PrismaticNeonQ2_14Double<'a, const GRID_SIZE: usize> {
    pub(crate) cube0: &'a [NeonAlignedI16x4],
    pub(crate) cube1: &'a [NeonAlignedI16x4],
}

pub(crate) struct TetrahedralNeonQ2_14Double<'a, const GRID_SIZE: usize> {
    pub(crate) cube0: &'a [NeonAlignedI16x4],
    pub(crate) cube1: &'a [NeonAlignedI16x4],
}

pub(crate) struct PrismaticNeonQ2_14<'a, const GRID_SIZE: usize> {
    pub(crate) cube: &'a [NeonAlignedI16x4],
}

trait Fetcher<T> {
    fn fetch(&self, x: i32, y: i32, z: i32) -> T;
}

struct TetrahedralNeonQ2_14QFetchVector<'a, const GRID_SIZE: usize> {
    cube: &'a [NeonAlignedI16x4],
}

struct TetrahedralNeonQ2_14QFetchVectorDouble<'a, const GRID_SIZE: usize> {
    cube0: &'a [NeonAlignedI16x4],
    cube1: &'a [NeonAlignedI16x4],
}

#[derive(Copy, Clone)]
pub(crate) struct NeonVectorQ2_14 {
    pub(crate) v: int16x4_t,
}

#[derive(Copy, Clone)]
pub(crate) struct NeonVectorQ2_14Double {
    pub(crate) v0: int16x4_t,
    pub(crate) v1: int16x4_t,
}

impl From<i16> for NeonVectorQ2_14 {
    #[inline(always)]
    fn from(v: i16) -> Self {
        NeonVectorQ2_14 {
            v: unsafe { vdup_n_s16(v) },
        }
    }
}

impl From<i16> for NeonVectorQ2_14Double {
    #[inline(always)]
    fn from(v: i16) -> Self {
        NeonVectorQ2_14Double {
            v0: unsafe { vdup_n_s16(v) },
            v1: unsafe { vdup_n_s16(v) },
        }
    }
}

impl Sub<NeonVectorQ2_14> for NeonVectorQ2_14 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: NeonVectorQ2_14) -> Self::Output {
        NeonVectorQ2_14 {
            v: unsafe { vsub_s16(self.v, rhs.v) },
        }
    }
}

impl Sub<NeonVectorQ2_14Double> for NeonVectorQ2_14Double {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: NeonVectorQ2_14Double) -> Self::Output {
        NeonVectorQ2_14Double {
            v0: unsafe { vsub_s16(self.v0, rhs.v0) },
            v1: unsafe { vsub_s16(self.v1, rhs.v1) },
        }
    }
}

impl Add<NeonVectorQ2_14> for NeonVectorQ2_14 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: NeonVectorQ2_14) -> Self::Output {
        NeonVectorQ2_14 {
            v: unsafe { vadd_s16(self.v, rhs.v) },
        }
    }
}

impl Add<NeonVectorQ2_14Double> for NeonVectorQ2_14Double {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: NeonVectorQ2_14Double) -> Self::Output {
        NeonVectorQ2_14Double {
            v0: unsafe { vadd_s16(self.v0, rhs.v0) },
            v1: unsafe { vadd_s16(self.v1, rhs.v1) },
        }
    }
}

impl FusedMultiplyAdd<NeonVectorQ2_14> for NeonVectorQ2_14 {
    #[inline(always)]
    fn mla(&self, b: NeonVectorQ2_14, c: NeonVectorQ2_14) -> NeonVectorQ2_14 {
        NeonVectorQ2_14 {
            v: unsafe { vqrdmlah_s16(self.v, b.v, c.v) },
        }
    }
}

impl NeonVectorQ2_14Double {
    #[inline(always)]
    fn mla(&self, b: NeonVectorQ2_14Double, c: NeonVectorQ2_14) -> NeonVectorQ2_14Double {
        NeonVectorQ2_14Double {
            v0: unsafe { vqrdmlah_s16(self.v0, b.v0, c.v) },
            v1: unsafe { vqrdmlah_s16(self.v1, b.v1, c.v) },
        }
    }

    #[inline(always)]
    pub(crate) fn split(self) -> (NeonVectorQ2_14, NeonVectorQ2_14) {
        (
            NeonVectorQ2_14 { v: self.v0 },
            NeonVectorQ2_14 { v: self.v1 },
        )
    }
}

impl<const GRID_SIZE: usize> Fetcher<NeonVectorQ2_14>
    for TetrahedralNeonQ2_14QFetchVector<'_, GRID_SIZE>
{
    fn fetch(&self, x: i32, y: i32, z: i32) -> NeonVectorQ2_14 {
        let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
            + y as u32 * GRID_SIZE as u32
            + z as u32) as usize;
        let jx = unsafe { self.cube.get_unchecked(offset..) };
        NeonVectorQ2_14 {
            v: unsafe { vld1_s16(jx.as_ptr() as *const i16) },
        }
    }
}

impl<const GRID_SIZE: usize> Fetcher<NeonVectorQ2_14Double>
    for TetrahedralNeonQ2_14QFetchVectorDouble<'_, GRID_SIZE>
{
    fn fetch(&self, x: i32, y: i32, z: i32) -> NeonVectorQ2_14Double {
        let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
            + y as u32 * GRID_SIZE as u32
            + z as u32) as usize;
        let jx0 = unsafe { self.cube0.get_unchecked(offset..) };
        let jx1 = unsafe { self.cube1.get_unchecked(offset..) };
        NeonVectorQ2_14Double {
            v0: unsafe { vld1_s16(jx0.as_ptr() as *const i16) },
            v1: unsafe { vld1_s16(jx1.as_ptr() as *const i16) },
        }
    }
}

pub(crate) trait NeonMdInterpolationQ2_14<'a, const GRID_SIZE: usize> {
    fn new(table: &'a [NeonAlignedI16x4]) -> Self;
    fn inter3_neon(&self, in_r: u16, in_g: u16, in_b: u16) -> NeonVectorQ2_14;
}

pub(crate) trait NeonMdInterpolationQ2_14Double<'a, const GRID_SIZE: usize> {
    fn new(table0: &'a [NeonAlignedI16x4], table1: &'a [NeonAlignedI16x4]) -> Self;
    fn inter3_neon(&self, in_r: u16, in_g: u16, in_b: u16) -> (NeonVectorQ2_14, NeonVectorQ2_14);
}

impl<const GRID_SIZE: usize> TetrahedralNeonQ2_14<'_, GRID_SIZE> {
    #[inline(always)]
    fn interpolate(
        &self,
        in_r: u16,
        in_g: u16,
        in_b: u16,
        r: impl Fetcher<NeonVectorQ2_14>,
    ) -> NeonVectorQ2_14 {
        const SCALE: f32 = 1.0 / 65535.0;
        let x: i32 = in_r as i32 * (GRID_SIZE as i32 - 1) / 65535;
        let y: i32 = in_g as i32 * (GRID_SIZE as i32 - 1) / 65535;
        let z: i32 = in_b as i32 * (GRID_SIZE as i32 - 1) / 65535;

        let c0 = r.fetch(x, y, z);

        let x_n: i32 = rounding_div_ceil(in_r as i32 * (GRID_SIZE as i32 - 1), 65535);
        let y_n: i32 = rounding_div_ceil(in_g as i32 * (GRID_SIZE as i32 - 1), 65535);
        let z_n: i32 = rounding_div_ceil(in_b as i32 * (GRID_SIZE as i32 - 1), 65535);

        let scale = (GRID_SIZE as i32 - 1) as f32 * SCALE;

        const Q_SCALE: f32 = ((1 << 14) - 1) as f32;

        let rx = ((in_r as f32 * scale - x as f32) * Q_SCALE) as i16;
        let ry = ((in_g as f32 * scale - y as f32) * Q_SCALE) as i16;
        let rz = ((in_b as f32 * scale - z as f32) * Q_SCALE) as i16;

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
        let s0 = c0.mla(c1, NeonVectorQ2_14::from(rx));
        let s1 = s0.mla(c2, NeonVectorQ2_14::from(ry));
        s1.mla(c3, NeonVectorQ2_14::from(rz))
    }
}

impl<const GRID_SIZE: usize> TetrahedralNeonQ2_14Double<'_, GRID_SIZE> {
    #[inline(always)]
    fn interpolate(
        &self,
        in_r: u16,
        in_g: u16,
        in_b: u16,
        r: impl Fetcher<NeonVectorQ2_14Double>,
    ) -> (NeonVectorQ2_14, NeonVectorQ2_14) {
        const SCALE: f32 = 1.0 / 65535.0;
        let x: i32 = in_r as i32 * (GRID_SIZE as i32 - 1) / 65535;
        let y: i32 = in_g as i32 * (GRID_SIZE as i32 - 1) / 65535;
        let z: i32 = in_b as i32 * (GRID_SIZE as i32 - 1) / 65535;

        let c0 = r.fetch(x, y, z);

        let x_n: i32 = rounding_div_ceil(in_r as i32 * (GRID_SIZE as i32 - 1), 65535);
        let y_n: i32 = rounding_div_ceil(in_g as i32 * (GRID_SIZE as i32 - 1), 65535);
        let z_n: i32 = rounding_div_ceil(in_b as i32 * (GRID_SIZE as i32 - 1), 65535);

        let scale = (GRID_SIZE as i32 - 1) as f32 * SCALE;

        const Q_SCALE: f32 = ((1 << 14) - 1) as f32;
        
        let rx = ((in_r as f32 * scale - x as f32) * Q_SCALE) as i16;
        let ry = ((in_g as f32 * scale - y as f32) * Q_SCALE) as i16;
        let rz = ((in_b as f32 * scale - z as f32) * Q_SCALE) as i16;

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
        let s0 = c0.mla(c1, NeonVectorQ2_14::from(rx));
        let s1 = s0.mla(c2, NeonVectorQ2_14::from(ry));
        s1.mla(c3, NeonVectorQ2_14::from(rz)).split()
    }
}

macro_rules! define_md_inter_neon {
    ($interpolator: ident) => {
        impl<'a, const GRID_SIZE: usize> NeonMdInterpolationQ2_14<'a, GRID_SIZE>
            for $interpolator<'a, GRID_SIZE>
        {
            #[inline(always)]
            fn new(table: &'a [NeonAlignedI16x4]) -> Self {
                Self { cube: table }
            }

            #[inline(always)]
            fn inter3_neon(&self, in_r: u16, in_g: u16, in_b: u16) -> NeonVectorQ2_14 {
                self.interpolate(
                    in_r,
                    in_g,
                    in_b,
                    TetrahedralNeonQ2_14QFetchVector::<GRID_SIZE> { cube: self.cube },
                )
            }
        }
    };
}

macro_rules! define_md_inter_neon_d {
    ($interpolator: ident) => {
        impl<'a, const GRID_SIZE: usize> NeonMdInterpolationQ2_14Double<'a, GRID_SIZE>
            for $interpolator<'a, GRID_SIZE>
        {
            #[inline(always)]
            fn new(table0: &'a [NeonAlignedI16x4], table1: &'a [NeonAlignedI16x4]) -> Self {
                Self {
                    cube0: table0,
                    cube1: table1,
                }
            }

            #[inline(always)]
            fn inter3_neon(
                &self,
                in_r: u16,
                in_g: u16,
                in_b: u16,
            ) -> (NeonVectorQ2_14, NeonVectorQ2_14) {
                self.interpolate(
                    in_r,
                    in_g,
                    in_b,
                    TetrahedralNeonQ2_14QFetchVectorDouble::<GRID_SIZE> {
                        cube0: self.cube0,
                        cube1: self.cube1,
                    },
                )
            }
        }
    };
}

define_md_inter_neon!(TetrahedralNeonQ2_14);
define_md_inter_neon!(PyramidalNeonQ2_14);
define_md_inter_neon!(PrismaticNeonQ2_14);
define_md_inter_neon_d!(PrismaticNeonQ2_14Double);
define_md_inter_neon_d!(PyramidalNeonQ2_14Double);
define_md_inter_neon_d!(TetrahedralNeonQ2_14Double);

impl<const GRID_SIZE: usize> PyramidalNeonQ2_14<'_, GRID_SIZE> {
    #[inline(always)]
    fn interpolate(
        &self,
        in_r: u16,
        in_g: u16,
        in_b: u16,
        r: impl Fetcher<NeonVectorQ2_14>,
    ) -> NeonVectorQ2_14 {
        const SCALE: f32 = 1.0 / 65535.0;
        let x: i32 = in_r as i32 * (GRID_SIZE as i32 - 1) / 65535;
        let y: i32 = in_g as i32 * (GRID_SIZE as i32 - 1) / 65535;
        let z: i32 = in_b as i32 * (GRID_SIZE as i32 - 1) / 65535;

        let c0 = r.fetch(x, y, z);

        let x_n: i32 = rounding_div_ceil(in_r as i32 * (GRID_SIZE as i32 - 1), 65535);
        let y_n: i32 = rounding_div_ceil(in_g as i32 * (GRID_SIZE as i32 - 1), 65535);
        let z_n: i32 = rounding_div_ceil(in_b as i32 * (GRID_SIZE as i32 - 1), 65535);

        let scale = (GRID_SIZE as i32 - 1) as f32 * SCALE;

        const Q_SCALE: f32 = ((1 << 14) - 1) as f32;

        let dr = ((in_r as f32 * scale - x as f32) * Q_SCALE) as i16;
        let dg = ((in_g as f32 * scale - y as f32) * Q_SCALE) as i16;
        let db = ((in_b as f32 * scale - z as f32) * Q_SCALE) as i16;

        if dr > db && dg > db {
            let x0 = r.fetch(x_n, y_n, z_n);
            let x1 = r.fetch(x_n, y_n, z);
            let x2 = r.fetch(x_n, y, z);
            let x3 = r.fetch(x, y_n, z);

            let c1 = x0 - x1;
            let c2 = x2 - c0;
            let c3 = x3 - c0;
            let c4 = c0 - x3 - x2 + x1;

            let s0 = c0.mla(c1, NeonVectorQ2_14::from(db));
            let s1 = s0.mla(c2, NeonVectorQ2_14::from(dr));
            let s2 = s1.mla(c3, NeonVectorQ2_14::from(dg));
            s2.mla(c4, NeonVectorQ2_14::from(dr * dg))
        } else if db > dr && dg > dr {
            let x0 = r.fetch(x, y, z_n);
            let x1 = r.fetch(x_n, y_n, z_n);
            let x2 = r.fetch(x, y_n, z_n);
            let x3 = r.fetch(x, y_n, z);

            let c1 = x0 - c0;
            let c2 = x1 - x2;
            let c3 = x3 - c0;
            let c4 = c0 - x3 - x0 + x2;

            let s0 = c0.mla(c1, NeonVectorQ2_14::from(db));
            let s1 = s0.mla(c2, NeonVectorQ2_14::from(dr));
            let s2 = s1.mla(c3, NeonVectorQ2_14::from(dg));
            s2.mla(c4, NeonVectorQ2_14::from(dg * db))
        } else {
            let x0 = r.fetch(x, y, z_n);
            let x1 = r.fetch(x_n, y, z);
            let x2 = r.fetch(x_n, y, z_n);
            let x3 = r.fetch(x_n, y_n, z_n);

            let c1 = x0 - c0;
            let c2 = x1 - c0;
            let c3 = x3 - x2;
            let c4 = c0 - x1 - x0 + x2;

            let s0 = c0.mla(c1, NeonVectorQ2_14::from(db));
            let s1 = s0.mla(c2, NeonVectorQ2_14::from(dr));
            let s2 = s1.mla(c3, NeonVectorQ2_14::from(dg));
            s2.mla(c4, NeonVectorQ2_14::from(db * dr))
        }
    }
}

impl<const GRID_SIZE: usize> PyramidalNeonQ2_14Double<'_, GRID_SIZE> {
    #[inline(always)]
    fn interpolate(
        &self,
        in_r: u16,
        in_g: u16,
        in_b: u16,
        r: impl Fetcher<NeonVectorQ2_14Double>,
    ) -> (NeonVectorQ2_14, NeonVectorQ2_14) {
        const SCALE: f32 = 1.0 / 65535.0;
        let x: i32 = in_r as i32 * (GRID_SIZE as i32 - 1) / 65535;
        let y: i32 = in_g as i32 * (GRID_SIZE as i32 - 1) / 65535;
        let z: i32 = in_b as i32 * (GRID_SIZE as i32 - 1) / 65535;

        let c0 = r.fetch(x, y, z);

        let x_n: i32 = rounding_div_ceil(in_r as i32 * (GRID_SIZE as i32 - 1), 65535);
        let y_n: i32 = rounding_div_ceil(in_g as i32 * (GRID_SIZE as i32 - 1), 65535);
        let z_n: i32 = rounding_div_ceil(in_b as i32 * (GRID_SIZE as i32 - 1), 65535);

        let scale = (GRID_SIZE as i32 - 1) as f32 * SCALE;

        const Q_SCALE: f32 = ((1 << 14) - 1) as f32;

        let dr = ((in_r as f32 * scale - x as f32) * Q_SCALE) as i16;
        let dg = ((in_g as f32 * scale - y as f32) * Q_SCALE) as i16;
        let db = ((in_b as f32 * scale - z as f32) * Q_SCALE) as i16;

        let w0 = NeonVectorQ2_14::from(db);
        let w1 = NeonVectorQ2_14::from(dr);
        let w2 = NeonVectorQ2_14::from(dg);

        if dr > db && dg > db {
            let x0 = r.fetch(x_n, y_n, z_n);
            let x1 = r.fetch(x_n, y_n, z);
            let x2 = r.fetch(x_n, y, z);
            let x3 = r.fetch(x, y_n, z);

            let c1 = x0 - x1;
            let c2 = x2 - c0;
            let c3 = x3 - c0;
            let c4 = c0 - x3 - x2 + x1;

            let w3 = NeonVectorQ2_14::from(dr * dg);

            let s0 = c0.mla(c1, w0);
            let s1 = s0.mla(c2, w1);
            let s2 = s1.mla(c3, w2);
            s2.mla(c4, w3).split()
        } else if db > dr && dg > dr {
            let x0 = r.fetch(x, y, z_n);
            let x1 = r.fetch(x_n, y_n, z_n);
            let x2 = r.fetch(x, y_n, z_n);
            let x3 = r.fetch(x, y_n, z);

            let c1 = x0 - c0;
            let c2 = x1 - x2;
            let c3 = x3 - c0;
            let c4 = c0 - x3 - x0 + x2;

            let w3 = NeonVectorQ2_14::from(dg * db);

            let s0 = c0.mla(c1, w0);
            let s1 = s0.mla(c2, w1);
            let s2 = s1.mla(c3, w2);
            s2.mla(c4, w3).split()
        } else {
            let x0 = r.fetch(x, y, z_n);
            let x1 = r.fetch(x_n, y, z);
            let x2 = r.fetch(x_n, y, z_n);
            let x3 = r.fetch(x_n, y_n, z_n);

            let c1 = x0 - c0;
            let c2 = x1 - c0;
            let c3 = x3 - x2;
            let c4 = c0 - x1 - x0 + x2;

            let w3 = NeonVectorQ2_14::from(db * dr);

            let s0 = c0.mla(c1, w0);
            let s1 = s0.mla(c2, w1);
            let s2 = s1.mla(c3, w2);
            s2.mla(c4, w3).split()
        }
    }
}

impl<const GRID_SIZE: usize> PrismaticNeonQ2_14<'_, GRID_SIZE> {
    #[inline(always)]
    fn interpolate(
        &self,
        in_r: u16,
        in_g: u16,
        in_b: u16,
        r: impl Fetcher<NeonVectorQ2_14>,
    ) -> NeonVectorQ2_14 {
        const SCALE: f32 = 1.0 / 65535.0;
        let x: i32 = in_r as i32 * (GRID_SIZE as i32 - 1) / 65535;
        let y: i32 = in_g as i32 * (GRID_SIZE as i32 - 1) / 65535;
        let z: i32 = in_b as i32 * (GRID_SIZE as i32 - 1) / 65535;

        let c0 = r.fetch(x, y, z);

        let x_n: i32 = rounding_div_ceil(in_r as i32 * (GRID_SIZE as i32 - 1), 65535);
        let y_n: i32 = rounding_div_ceil(in_g as i32 * (GRID_SIZE as i32 - 1), 65535);
        let z_n: i32 = rounding_div_ceil(in_b as i32 * (GRID_SIZE as i32 - 1), 65535);

        let scale = (GRID_SIZE as i32 - 1) as f32 * SCALE;

        const Q_SCALE: f32 = ((1 << 14) - 1) as f32;

        let dr = ((in_r as f32 * scale - x as f32) * Q_SCALE) as i16;
        let dg = ((in_g as f32 * scale - y as f32) * Q_SCALE) as i16;
        let db = ((in_b as f32 * scale - z as f32) * Q_SCALE) as i16;

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

            let s0 = c0.mla(c1, NeonVectorQ2_14::from(db));
            let s1 = s0.mla(c2, NeonVectorQ2_14::from(dr));
            let s2 = s1.mla(c3, NeonVectorQ2_14::from(dg));
            let s3 = s2.mla(c4, NeonVectorQ2_14::from(dg * db));
            s3.mla(c5, NeonVectorQ2_14::from(dr * dg))
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

            let s0 = c0.mla(c1, NeonVectorQ2_14::from(db));
            let s1 = s0.mla(c2, NeonVectorQ2_14::from(dr));
            let s2 = s1.mla(c3, NeonVectorQ2_14::from(dg));
            let s3 = s2.mla(c4, NeonVectorQ2_14::from(dg * db));
            s3.mla(c5, NeonVectorQ2_14::from(dr * dg))
        }
    }
}

impl<const GRID_SIZE: usize> PrismaticNeonQ2_14Double<'_, GRID_SIZE> {
    #[inline(always)]
    fn interpolate(
        &self,
        in_r: u16,
        in_g: u16,
        in_b: u16,
        rv: impl Fetcher<NeonVectorQ2_14Double>,
    ) -> (NeonVectorQ2_14, NeonVectorQ2_14) {
        const SCALE: f32 = 1.0 / 65535.0;
        let x: i32 = in_r as i32 * (GRID_SIZE as i32 - 1) / 65535;
        let y: i32 = in_g as i32 * (GRID_SIZE as i32 - 1) / 65535;
        let z: i32 = in_b as i32 * (GRID_SIZE as i32 - 1) / 65535;

        let c0 = rv.fetch(x, y, z);

        let x_n: i32 = rounding_div_ceil(in_r as i32 * (GRID_SIZE as i32 - 1), 65535);
        let y_n: i32 = rounding_div_ceil(in_g as i32 * (GRID_SIZE as i32 - 1), 65535);
        let z_n: i32 = rounding_div_ceil(in_b as i32 * (GRID_SIZE as i32 - 1), 65535);

        let scale = (GRID_SIZE as i32 - 1) as f32 * SCALE;

        const Q_SCALE: f32 = ((1 << 14) - 1) as f32;

        let dr = ((in_r as f32 * scale - x as f32) * Q_SCALE) as i16;
        let dg = ((in_g as f32 * scale - y as f32) * Q_SCALE) as i16;
        let db = ((in_b as f32 * scale - z as f32) * Q_SCALE) as i16;

        let w0 = NeonVectorQ2_14::from(db);
        let w1 = NeonVectorQ2_14::from(dr);
        let w2 = NeonVectorQ2_14::from(dg);
        let w3 = NeonVectorQ2_14::from(dg * db);
        let w4 = NeonVectorQ2_14::from(dr * dg);

        if db > dr {
            let x0 = rv.fetch(x, y, z_n);
            let x1 = rv.fetch(x_n, y, z_n);
            let x2 = rv.fetch(x, y_n, z);
            let x3 = rv.fetch(x, y_n, z_n);
            let x4 = rv.fetch(x_n, y_n, z_n);

            let c1 = x0 - c0;
            let c2 = x1 - x0;
            let c3 = x2 - c0;
            let c4 = c0 - x2 - x0 + x3;
            let c5 = x0 - x3 - x1 + x4;

            let s0 = c0.mla(c1, w0);
            let s1 = s0.mla(c2, w1);
            let s2 = s1.mla(c3, w2);
            let s3 = s2.mla(c4, w3);
            s3.mla(c5, w4).split()
        } else {
            let x0 = rv.fetch(x_n, y, z);
            let x1 = rv.fetch(x_n, y, z_n);
            let x2 = rv.fetch(x, y_n, z);
            let x3 = rv.fetch(x_n, y_n, z);
            let x4 = rv.fetch(x_n, y_n, z_n);

            let c1 = x1 - x0;
            let c2 = x0 - c0;
            let c3 = x2 - c0;
            let c4 = x0 - x3 - x1 + x4;
            let c5 = c0 - x2 - x0 + x3;

            let s0 = c0.mla(c1, w0);
            let s1 = s0.mla(c2, w1);
            let s2 = s1.mla(c3, w2);
            let s3 = s2.mla(c4, w3);
            s3.mla(c5, w4).split()
        }
    }
}
