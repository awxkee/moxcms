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
use crate::conversions::tetrahedral::TetrhedralInterpolation;
use crate::math::FusedMultiplyAdd;
use crate::{Vector3f, Vector4f, rounding_div_ceil};
use std::arch::aarch64::*;
use std::ops::Sub;

pub(crate) struct TetrahedralNeon<'a, const GRID_SIZE: usize> {
    pub(crate) cube: &'a [f32],
}

trait Fetcher<T> {
    fn fetch(&self, x: i32, y: i32, z: i32) -> T;
}

struct TetrahedralNeonFetchVector3f<'a, const GRID_SIZE: usize> {
    cube: &'a [f32],
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

impl FusedMultiplyAdd<NeonVector> for NeonVector {
    #[inline(always)]
    fn mla(&self, b: NeonVector, c: NeonVector) -> NeonVector {
        NeonVector {
            v: unsafe { vfmaq_f32(self.v, b.v, c.v) },
        }
    }
}

impl<const GRID_SIZE: usize> Fetcher<NeonVector> for TetrahedralNeonFetchVector3f<'_, GRID_SIZE> {
    fn fetch(&self, x: i32, y: i32, z: i32) -> NeonVector {
        let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
            + y as u32 * GRID_SIZE as u32
            + z as u32) as usize
            * 3;
        let jx = unsafe { self.cube.get_unchecked(offset..) };
        let v0 = unsafe { vcombine_f32(vld1_f32(jx.as_ptr()), vdup_n_f32(0.0f32)) };
        let v1 = unsafe { vld1q_lane_f32::<2>(jx.get_unchecked(2..).as_ptr(), v0) };
        NeonVector { v: v1 }
    }
}

struct TetrahedralNeonFetchVector4f<'a, const GRID_SIZE: usize> {
    cube: &'a [f32],
}

impl<const GRID_SIZE: usize> Fetcher<NeonVector> for TetrahedralNeonFetchVector4f<'_, GRID_SIZE> {
    #[inline(always)]
    fn fetch(&self, x: i32, y: i32, z: i32) -> NeonVector {
        let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
            + y as u32 * GRID_SIZE as u32
            + z as u32) as usize
            * 4;
        let jx = unsafe { self.cube.get_unchecked(offset..) };
        NeonVector {
            v: unsafe { vld1q_f32(jx.as_ptr()) },
        }
    }
}

impl<const GRID_SIZE: usize> TetrahedralNeon<'_, GRID_SIZE> {
    #[inline(always)]
    fn interpolate(&self, in_r: u8, in_g: u8, in_b: u8, r: impl Fetcher<NeonVector>) -> NeonVector {
        const SCALE: f32 = 1.0 / 255.0;
        let linear_r: f32 = in_r as i32 as f32 * SCALE;
        let linear_g: f32 = in_g as i32 as f32 * SCALE;
        let linear_b: f32 = in_b as i32 as f32 * SCALE;
        let x: i32 = in_r as i32 * (GRID_SIZE as i32 - 1) / 255;
        let y: i32 = in_g as i32 * (GRID_SIZE as i32 - 1) / 255;
        let z: i32 = in_b as i32 * (GRID_SIZE as i32 - 1) / 255;
        let x_n: i32 = rounding_div_ceil(in_r as i32 * (GRID_SIZE as i32 - 1), 255);
        let y_n: i32 = rounding_div_ceil(in_g as i32 * (GRID_SIZE as i32 - 1), 255);
        let z_n: i32 = rounding_div_ceil(in_b as i32 * (GRID_SIZE as i32 - 1), 255);
        let rx = linear_r * (GRID_SIZE as i32 - 1) as f32 - x as f32;
        let ry = linear_g * (GRID_SIZE as i32 - 1) as f32 - y as f32;
        let rz = linear_b * (GRID_SIZE as i32 - 1) as f32 - z as f32;
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
        let s0 = c0.mla(c1, NeonVector::from(rx));
        let s1 = s0.mla(c2, NeonVector::from(ry));
        s1.mla(c3, NeonVector::from(rz))
    }
}

impl<const GRID_SIZE: usize> TetrahedralNeon<'_, GRID_SIZE> {
    #[inline(always)]
    pub(crate) fn inter3_neon(&self, in_r: u8, in_g: u8, in_b: u8) -> NeonVector {
        self.interpolate(
            in_r,
            in_g,
            in_b,
            TetrahedralNeonFetchVector3f::<GRID_SIZE> { cube: self.cube },
        )
    }
}

impl<'a, const GRID_SIZE: usize> TetrhedralInterpolation<'a, GRID_SIZE>
    for TetrahedralNeon<'a, GRID_SIZE>
{
    fn new(table: &'a [f32]) -> Self {
        Self { cube: table }
    }

    #[inline(always)]
    fn inter3(&self, in_r: u8, in_g: u8, in_b: u8) -> Vector3f {
        let v = self.interpolate(
            in_r,
            in_g,
            in_b,
            TetrahedralNeonFetchVector3f::<GRID_SIZE> { cube: self.cube },
        );
        let mut vector3 = Vector3f { v: [0f32; 3] };
        unsafe {
            vst1_f32(vector3.v.as_mut_ptr(), vget_low_f32(v.v));
            vst1q_lane_f32::<2>((vector3.v.as_mut_ptr()).add(2), v.v);
        }
        vector3
    }

    #[inline(always)]
    fn inter4(&self, in_r: u8, in_g: u8, in_b: u8) -> Vector4f {
        let v = self.interpolate(
            in_r,
            in_g,
            in_b,
            TetrahedralNeonFetchVector4f::<GRID_SIZE> { cube: self.cube },
        );
        let mut vector4 = Vector4f { v: [0f32; 4] };
        unsafe {
            vst1q_f32(vector4.v.as_mut_ptr(), v.v);
        }
        vector4
    }
}
