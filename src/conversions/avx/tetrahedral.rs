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
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ops::Sub;

#[repr(align(16), C)]
pub(crate) struct SseAlignedF32(pub(crate) [f32; 4]);

pub(crate) struct TetrahedralAvxFma<'a, const GRID_SIZE: usize> {
    pub(crate) cube: &'a [SseAlignedF32],
}

trait Fetcher<T> {
    fn fetch(&self, x: i32, y: i32, z: i32) -> T;
}

#[derive(Copy, Clone)]
#[repr(transparent)]
pub(crate) struct AvxVectorSse {
    pub(crate) v: __m128,
}

impl From<f32> for AvxVectorSse {
    #[inline(always)]
    fn from(v: f32) -> Self {
        AvxVectorSse {
            v: unsafe { _mm_set1_ps(v) },
        }
    }
}

impl Sub<AvxVectorSse> for AvxVectorSse {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: AvxVectorSse) -> Self::Output {
        AvxVectorSse {
            v: unsafe { _mm_sub_ps(self.v, rhs.v) },
        }
    }
}

impl FusedMultiplyAdd<AvxVectorSse> for AvxVectorSse {
    #[inline(always)]
    fn mla(&self, b: AvxVectorSse, c: AvxVectorSse) -> AvxVectorSse {
        AvxVectorSse {
            v: unsafe { _mm_fmadd_ps(b.v, c.v, self.v) },
        }
    }
}

struct TetrahedralAvxFmaFetchVector<'a, const GRID_SIZE: usize> {
    cube: &'a [SseAlignedF32],
}

impl<const GRID_SIZE: usize> Fetcher<AvxVectorSse> for TetrahedralAvxFmaFetchVector<'_, GRID_SIZE> {
    #[inline(always)]
    fn fetch(&self, x: i32, y: i32, z: i32) -> AvxVectorSse {
        let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
            + y as u32 * GRID_SIZE as u32
            + z as u32) as usize;
        let jx = unsafe { self.cube.get_unchecked(offset..) };
        AvxVectorSse {
            v: unsafe { _mm_load_ps(jx.as_ptr() as *const f32) },
        }
    }
}

impl<const GRID_SIZE: usize> TetrahedralAvxFma<'_, GRID_SIZE> {
    #[inline(always)]
    fn interpolate(
        &self,
        in_r: u8,
        in_g: u8,
        in_b: u8,
        r: impl Fetcher<AvxVectorSse>,
    ) -> AvxVectorSse {
        const SCALE: f32 = 1.0 / 255.0;
        let x: i32 = in_r as i32 * (GRID_SIZE as i32 - 1) / 255;
        let y: i32 = in_g as i32 * (GRID_SIZE as i32 - 1) / 255;
        let z: i32 = in_b as i32 * (GRID_SIZE as i32 - 1) / 255;

        let c0 = r.fetch(x, y, z);

        let x_n: i32 = (in_r as f32 * ((GRID_SIZE as i32 - 1) as f32 * SCALE)).ceil() as i32;
        let y_n: i32 = (in_g as f32 * ((GRID_SIZE as i32 - 1) as f32 * SCALE)).ceil() as i32;
        let z_n: i32 = (in_b as f32 * ((GRID_SIZE as i32 - 1) as f32 * SCALE)).ceil() as i32;

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
        let s0 = c0.mla(c1, AvxVectorSse::from(rx));
        let s1 = s0.mla(c2, AvxVectorSse::from(ry));
        s1.mla(c3, AvxVectorSse::from(rz))
    }
}

impl<const GRID_SIZE: usize> TetrahedralAvxFma<'_, GRID_SIZE> {
    #[inline(always)]
    pub(crate) fn inter3_sse(&self, in_r: u8, in_g: u8, in_b: u8) -> AvxVectorSse {
        self.interpolate(
            in_r,
            in_g,
            in_b,
            TetrahedralAvxFmaFetchVector::<GRID_SIZE> { cube: self.cube },
        )
    }
}

impl<'a, const GRID_SIZE: usize> TetrahedralAvxFma<'a, GRID_SIZE> {
    pub(crate) fn new(table: &'a [SseAlignedF32]) -> Self {
        Self { cube: table }
    }
}
