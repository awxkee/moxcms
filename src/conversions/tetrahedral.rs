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
use crate::{Vector3f, Vector4f, rounding_div_ceil};
use std::ops::{Add, Mul, Sub};

pub(crate) struct Tetrahedral<'a, const GRID_SIZE: usize> {
    pub(crate) cube: &'a [f32],
}

impl<'a, const GRID_SIZE: usize> Tetrahedral<'a, GRID_SIZE> {
    pub(crate) fn new(table: &'a [f32]) -> Self {
        Self { cube: table }
    }

    #[inline]
    fn find3(cube: &[f32], x: i32, y: i32, z: i32) -> Vector3f {
        let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
            + y as u32 * GRID_SIZE as u32
            + z as u32) as usize
            * 3;
        let jx = &cube[offset..offset + 3];
        Vector3f {
            v: [jx[0], jx[1], jx[2]],
        }
    }

    #[inline]
    fn find4(tab: &[f32], x: i32, y: i32, z: i32) -> Vector4f {
        let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
            + y as u32 * GRID_SIZE as u32
            + z as u32) as usize
            * 4;
        let jx = &tab[offset..offset + 4];
        Vector4f {
            v: [jx[0], jx[1], jx[2], jx[3]],
        }
    }

    #[inline]
    fn interpolate<
        T: Copy + Sub<T, Output = T> + Mul<T, Output = T> + Mul<f32, Output = T> + Add<T, Output = T>,
        Retriever: Fn(&[f32], i32, i32, i32) -> T,
    >(
        &self,
        in_r: u8,
        in_g: u8,
        in_b: u8,
        r: Retriever,
    ) -> T {
        let linear_r: f32 = in_r as i32 as f32 / 255.0;
        let linear_g: f32 = in_g as i32 as f32 / 255.0;
        let linear_b: f32 = in_b as i32 as f32 / 255.0;
        let x: i32 = in_r as i32 * (GRID_SIZE as i32 - 1) / 255;
        let y: i32 = in_g as i32 * (GRID_SIZE as i32 - 1) / 255;
        let z: i32 = in_b as i32 * (GRID_SIZE as i32 - 1) / 255;
        let x_n: i32 = rounding_div_ceil(in_r as i32 * (GRID_SIZE as i32 - 1), 255);
        let y_n: i32 = rounding_div_ceil(in_g as i32 * (GRID_SIZE as i32 - 1), 255);
        let z_n: i32 = rounding_div_ceil(in_b as i32 * (GRID_SIZE as i32 - 1), 255);
        let rx: f32 = linear_r * (GRID_SIZE as i32 - 1) as f32 - x as f32;
        let ry: f32 = linear_g * (GRID_SIZE as i32 - 1) as f32 - y as f32;
        let rz: f32 = linear_b * (GRID_SIZE as i32 - 1) as f32 - z as f32;
        let c0 = r(self.cube, x, y, z);
        let c2;
        let c1;
        let c3;
        if rx >= ry {
            if ry >= rz {
                //rx >= ry && ry >= rz
                c1 = r(self.cube, x_n, y, z) - c0;
                c2 = r(self.cube, x_n, y_n, z) - r(self.cube, x_n, y, z);
                c3 = r(self.cube, x_n, y_n, z_n) - r(self.cube, x_n, y_n, z);
            } else if rx >= rz {
                //rx >= rz && rz >= ry
                c1 = r(self.cube, x_n, y, z) - c0;
                c2 = r(self.cube, x_n, y_n, z_n) - r(self.cube, x_n, y, z_n);
                c3 = r(self.cube, x_n, y, z_n) - r(self.cube, x_n, y, z);
            } else {
                //rz > rx && rx >= ry
                c1 = r(self.cube, x_n, y, z_n) - r(self.cube, x, y, z_n);
                c2 = r(self.cube, x_n, y_n, z_n) - r(self.cube, x_n, y, z_n);
                c3 = r(self.cube, x, y, z_n) - c0;
            }
        } else if rx >= rz {
            //ry > rx && rx >= rz
            c1 = r(self.cube, x_n, y_n, z) - r(self.cube, x, y_n, z);
            c2 = r(self.cube, x, y_n, z) - c0;
            c3 = r(self.cube, x_n, y_n, z_n) - r(self.cube, x_n, y_n, z);
        } else if ry >= rz {
            //ry >= rz && rz > rx
            c1 = r(self.cube, x_n, y_n, z_n) - r(self.cube, x, y_n, z_n);
            c2 = r(self.cube, x, y_n, z) - c0;
            c3 = r(self.cube, x, y_n, z_n) - r(self.cube, x, y_n, z);
        } else {
            //rz > ry && ry > rx
            c1 = r(self.cube, x_n, y_n, z_n) - r(self.cube, x, y_n, z_n);
            c2 = r(self.cube, x, y_n, z_n) - r(self.cube, x, y, z_n);
            c3 = r(self.cube, x, y, z_n) - c0;
        }
        c0 + c1 * rx + c2 * ry + c3 * rz
    }

    #[inline]
    pub(crate) fn inter4(&self, in_r: u8, in_g: u8, in_b: u8) -> Vector4f {
        self.interpolate(in_r, in_g, in_b, Self::find4)
    }

    #[inline]
    pub(crate) fn inter3(&self, in_r: u8, in_g: u8, in_b: u8) -> Vector3f {
        self.interpolate(in_r, in_g, in_b, Self::find3)
    }
}
