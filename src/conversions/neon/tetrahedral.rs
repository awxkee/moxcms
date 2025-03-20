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

define_md_inter_neon!(TetrahedralNeon);
define_md_inter_neon!(PyramidalNeon);
define_md_inter_neon!(PrismaticNeon);

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

        let c2;
        let c1;
        let c3;
        let c4;

        if db > dr && dg > dr {
            c1 = r.fetch(x_n, y_n, z_n) - r.fetch(x_n, y_n, z);
            c2 = r.fetch(x_n, y, z) - c0;
            c3 = r.fetch(x, y_n, z) - c0;
            c4 = c0 - r.fetch(x, y_n, z) - r.fetch(x_n, y, z) + r.fetch(x_n, y_n, z);

            let s0 = c0.mla(c1, NeonVector::from(db));
            let s1 = s0.mla(c2, NeonVector::from(dr));
            let s2 = s1.mla(c3, NeonVector::from(dg));
            s2.mla(c4, NeonVector::from(dr * dg))
        } else if db > dr && dg > dr {
            c1 = r.fetch(x, y, z_n) - c0;
            c2 = r.fetch(x_n, y_n, z_n) - r.fetch(x, y_n, z_n);
            c3 = r.fetch(x, y_n, z) - c0;
            c4 = c0 - r.fetch(x, y_n, z) - r.fetch(x, y, z_n) + r.fetch(x, y_n, z_n);

            let s0 = c0.mla(c1, NeonVector::from(db));
            let s1 = s0.mla(c2, NeonVector::from(dr));
            let s2 = s1.mla(c3, NeonVector::from(dg));
            s2.mla(c4, NeonVector::from(dg * db))
        } else {
            c1 = r.fetch(x, y, z_n) - c0;
            c2 = r.fetch(x_n, y, z) - c0;
            c3 = r.fetch(x_n, y_n, z) - r.fetch(x_n, y, z_n);
            c4 = c0 - r.fetch(x_n, y, z) - r.fetch(x, y, z_n) + r.fetch(x_n, y, z_n);

            let s0 = c0.mla(c1, NeonVector::from(db));
            let s1 = s0.mla(c2, NeonVector::from(dr));
            let s2 = s1.mla(c3, NeonVector::from(dg));
            s2.mla(c4, NeonVector::from(db * dr))
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
            let c1 = r.fetch(x, y, z_n) - c0;
            let c2 = r.fetch(x_n, y, z_n) - r.fetch(x, y, z_n);
            let c3 = r.fetch(x, y_n, z) - c0;
            let c4 = c0 - r.fetch(x, y_n, z) - r.fetch(x, y, z_n) + r.fetch(x, y_n, z_n);
            let c5 = r.fetch(x, y, z_n) - r.fetch(x, y_n, z_n) - r.fetch(x_n, y, z_n)
                + r.fetch(x_n, y_n, z_n);

            let s0 = c0.mla(c1, NeonVector::from(db));
            let s1 = s0.mla(c2, NeonVector::from(dr));
            let s2 = s1.mla(c3, NeonVector::from(dg));
            let s3 = s2.mla(c4, NeonVector::from(dg * db));
            s3.mla(c5, NeonVector::from(dr * dg))
        } else {
            let c1 = r.fetch(x_n, y, z) - r.fetch(x_n, y, z_n);
            let c2 = r.fetch(x_n, y, z) - c0;
            let c3 = r.fetch(x, y_n, z) - c0;
            let c4 = r.fetch(x_n, y, z) - r.fetch(x_n, y_n, z) - r.fetch(x_n, y, z_n)
                + r.fetch(x_n, y_n, z_n);
            let c5 = c0 - r.fetch(x, y_n, z) - r.fetch(x_n, y, z) + r.fetch(x_n, y_n, z);

            let s0 = c0.mla(c1, NeonVector::from(db));
            let s1 = s0.mla(c2, NeonVector::from(dr));
            let s2 = s1.mla(c3, NeonVector::from(dg));
            let s3 = s2.mla(c4, NeonVector::from(dg * db));
            s3.mla(c5, NeonVector::from(dr * dg))
        }
    }
}
