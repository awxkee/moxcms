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
use crate::math::FusedMultiplyAdd;
use crate::rounding_div_ceil;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ops::{Add, Sub};

#[repr(align(16), C)]
pub(crate) struct SseAlignedF32(pub(crate) [f32; 4]);

pub(crate) struct TetrahedralSse<'a, const GRID_SIZE: usize> {
    pub(crate) cube: &'a [SseAlignedF32],
}

pub(crate) struct PyramidalSse<'a, const GRID_SIZE: usize> {
    pub(crate) cube: &'a [SseAlignedF32],
}

trait Fetcher<T> {
    fn fetch(&self, x: i32, y: i32, z: i32) -> T;
}

#[derive(Copy, Clone)]
#[repr(transparent)]
pub(crate) struct SseVector {
    pub(crate) v: __m128,
}

impl From<f32> for SseVector {
    #[inline(always)]
    fn from(v: f32) -> Self {
        SseVector {
            v: unsafe { _mm_set1_ps(v) },
        }
    }
}

impl Sub<SseVector> for SseVector {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: SseVector) -> Self::Output {
        SseVector {
            v: unsafe { _mm_sub_ps(self.v, rhs.v) },
        }
    }
}

impl Add<SseVector> for SseVector {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: SseVector) -> Self::Output {
        SseVector {
            v: unsafe { _mm_add_ps(self.v, rhs.v) },
        }
    }
}

impl FusedMultiplyAdd<SseVector> for SseVector {
    #[inline(always)]
    fn mla(&self, b: SseVector, c: SseVector) -> SseVector {
        SseVector {
            v: unsafe { _mm_add_ps(self.v, _mm_mul_ps(b.v, c.v)) },
        }
    }
}

struct TetrahedralSseFetchVector<'a, const GRID_SIZE: usize> {
    cube: &'a [SseAlignedF32],
}

impl<const GRID_SIZE: usize> Fetcher<SseVector> for TetrahedralSseFetchVector<'_, GRID_SIZE> {
    #[inline(always)]
    fn fetch(&self, x: i32, y: i32, z: i32) -> SseVector {
        let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
            + y as u32 * GRID_SIZE as u32
            + z as u32) as usize;
        let jx = unsafe { self.cube.get_unchecked(offset..) };
        SseVector {
            v: unsafe { _mm_load_ps(jx.as_ptr() as *const _) },
        }
    }
}

pub(crate) trait SseMdInterpolation<'a, const GRID_SIZE: usize> {
    fn new(table: &'a [SseAlignedF32]) -> Self;
    fn inter3_sse(&self, in_r: u8, in_g: u8, in_b: u8) -> SseVector;
}

impl<const GRID_SIZE: usize> TetrahedralSse<'_, GRID_SIZE> {
    #[inline(always)]
    fn interpolate(&self, in_r: u8, in_g: u8, in_b: u8, r: impl Fetcher<SseVector>) -> SseVector {
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

        // This is stupid to return the same in each branch but Rust is not able
        // to optimize this appropriately.
        if rx >= ry {
            if ry >= rz {
                //rx >= ry && ry >= rz
                c1 = r.fetch(x_n, y, z) - c0;
                c2 = r.fetch(x_n, y_n, z) - r.fetch(x_n, y, z);
                c3 = r.fetch(x_n, y_n, z_n) - r.fetch(x_n, y_n, z);
                let s0 = c0.mla(c1, SseVector::from(rx));
                let s1 = s0.mla(c2, SseVector::from(ry));
                s1.mla(c3, SseVector::from(rz))
            } else if rx >= rz {
                //rx >= rz && rz >= ry
                c1 = r.fetch(x_n, y, z) - c0;
                c2 = r.fetch(x_n, y_n, z_n) - r.fetch(x_n, y, z_n);
                c3 = r.fetch(x_n, y, z_n) - r.fetch(x_n, y, z);
                let s0 = c0.mla(c1, SseVector::from(rx));
                let s1 = s0.mla(c2, SseVector::from(ry));
                s1.mla(c3, SseVector::from(rz))
            } else {
                //rz > rx && rx >= ry
                c1 = r.fetch(x_n, y, z_n) - r.fetch(x, y, z_n);
                c2 = r.fetch(x_n, y_n, z_n) - r.fetch(x_n, y, z_n);
                c3 = r.fetch(x, y, z_n) - c0;
                let s0 = c0.mla(c1, SseVector::from(rx));
                let s1 = s0.mla(c2, SseVector::from(ry));
                s1.mla(c3, SseVector::from(rz))
            }
        } else if rx >= rz {
            //ry > rx && rx >= rz
            c1 = r.fetch(x_n, y_n, z) - r.fetch(x, y_n, z);
            c2 = r.fetch(x, y_n, z) - c0;
            c3 = r.fetch(x_n, y_n, z_n) - r.fetch(x_n, y_n, z);
            let s0 = c0.mla(c1, SseVector::from(rx));
            let s1 = s0.mla(c2, SseVector::from(ry));
            s1.mla(c3, SseVector::from(rz))
        } else if ry >= rz {
            //ry >= rz && rz > rx
            c1 = r.fetch(x_n, y_n, z_n) - r.fetch(x, y_n, z_n);
            c2 = r.fetch(x, y_n, z) - c0;
            c3 = r.fetch(x, y_n, z_n) - r.fetch(x, y_n, z);
            let s0 = c0.mla(c1, SseVector::from(rx));
            let s1 = s0.mla(c2, SseVector::from(ry));
            s1.mla(c3, SseVector::from(rz))
        } else {
            //rz > ry && ry > rx
            c1 = r.fetch(x_n, y_n, z_n) - r.fetch(x, y_n, z_n);
            c2 = r.fetch(x, y_n, z_n) - r.fetch(x, y, z_n);
            c3 = r.fetch(x, y, z_n) - c0;
            let s0 = c0.mla(c1, SseVector::from(rx));
            let s1 = s0.mla(c2, SseVector::from(ry));
            s1.mla(c3, SseVector::from(rz))
        }
    }
}

impl<'a, const GRID_SIZE: usize> SseMdInterpolation<'a, GRID_SIZE>
    for TetrahedralSse<'a, GRID_SIZE>
{
    fn new(table: &'a [SseAlignedF32]) -> Self {
        Self { cube: table }
    }

    #[inline(always)]
    fn inter3_sse(&self, in_r: u8, in_g: u8, in_b: u8) -> SseVector {
        self.interpolate(
            in_r,
            in_g,
            in_b,
            TetrahedralSseFetchVector::<GRID_SIZE> { cube: self.cube },
        )
    }
}

impl<const GRID_SIZE: usize> PyramidalSse<'_, GRID_SIZE> {
    #[inline(always)]
    fn interpolate(&self, in_r: u8, in_g: u8, in_b: u8, r: impl Fetcher<SseVector>) -> SseVector {
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

            let s0 = c0.mla(c1, SseVector::from(db));
            let s1 = s0.mla(c2, SseVector::from(dr));
            let s2 = s1.mla(c3, SseVector::from(dg));
            s2.mla(c4, SseVector::from(dr * dg))
        } else if db > dr && dg > dr {
            c1 = r.fetch(x, y, z_n) - c0;
            c2 = r.fetch(x_n, y_n, z_n) - r.fetch(x, y_n, z_n);
            c3 = r.fetch(x, y_n, z) - c0;
            c4 = c0 - r.fetch(x, y_n, z) - r.fetch(x, y, z_n) + r.fetch(x, y_n, z_n);

            let s0 = c0.mla(c1, SseVector::from(db));
            let s1 = s0.mla(c2, SseVector::from(dr));
            let s2 = s1.mla(c3, SseVector::from(dg));
            s2.mla(c4, SseVector::from(dg * db))
        } else {
            c1 = r.fetch(x, y, z_n) - c0;
            c2 = r.fetch(x_n, y, z) - c0;
            c3 = r.fetch(x_n, y_n, z) - r.fetch(x_n, y, z_n);
            c4 = c0 - r.fetch(x_n, y, z) - r.fetch(x, y, z_n) + r.fetch(x_n, y, z_n);

            let s0 = c0.mla(c1, SseVector::from(db));
            let s1 = s0.mla(c2, SseVector::from(dr));
            let s2 = s1.mla(c3, SseVector::from(dg));
            s2.mla(c4, SseVector::from(db * dr))
        }
    }
}

impl<'a, const GRID_SIZE: usize> SseMdInterpolation<'a, GRID_SIZE> for PyramidalSse<'a, GRID_SIZE> {
    #[inline(always)]
    fn new(table: &'a [SseAlignedF32]) -> Self {
        Self { cube: table }
    }

    #[inline(always)]
    fn inter3_sse(&self, in_r: u8, in_g: u8, in_b: u8) -> SseVector {
        self.interpolate(
            in_r,
            in_g,
            in_b,
            TetrahedralSseFetchVector::<GRID_SIZE> { cube: self.cube },
        )
    }
}
