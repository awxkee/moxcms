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
use crate::Vector3f;
use std::ops::{Add, Mul, Sub};

#[inline]
pub(crate) fn lerp<T: Mul<Output = T> + Sub<Output = T> + Add<Output = T> + From<f32> + Copy>(
    a: T,
    b: T,
    t: T,
) -> T {
    a * (T::from(1.0) - t) + b * t
}

/// 4D CLUT helper
pub struct Array4D<'a> {
    array: &'a [f32],
    x_stride: u32,
    y_stride: u32,
    z_stride: u32,
    grid_size: usize,
}

impl Array4D<'_> {
    pub fn new(array: &[f32], grid_size: usize) -> Array4D {
        let z_stride = grid_size as u32;
        let y_stride = z_stride * z_stride;
        let x_stride = z_stride * z_stride * z_stride;
        Array4D {
            array,
            x_stride,
            y_stride,
            z_stride,
            grid_size,
        }
    }

    #[inline]
    pub fn vec3(&self, x: i32, y: i32, z: i32, w: i32) -> Vector3f {
        let start = (x as u32 * self.x_stride
            + y as u32 * self.y_stride
            + z as u32 * self.z_stride
            + w as u32) as usize
            * 3;
        let k = &self.array[start..start + 3];
        Vector3f {
            v: [k[0], k[1], k[2]],
        }
    }
}

impl Array4D<'_> {
    #[inline]
    pub fn quadlinear_vec3(&self, lin_x: f32, lin_y: f32, lin_z: f32, lin_w: f32) -> Vector3f {
        let scale = (self.grid_size as i32 - 1) as f32;

        let x = (lin_x * scale).floor() as i32;
        let y = (lin_y * scale).floor() as i32;
        let z = (lin_z * scale).floor() as i32;
        let w = (lin_w * scale).floor() as i32;

        let x_n = (lin_x * scale).ceil() as i32;
        let y_n = (lin_y * scale).ceil() as i32;
        let z_n = (lin_z * scale).ceil() as i32;
        let w_n = (lin_w * scale).ceil() as i32;

        let x_d = Vector3f::from(lin_x * scale - x as f32);
        let y_d = Vector3f::from(lin_y * scale - y as f32);
        let z_d = Vector3f::from(lin_z * scale - z as f32);
        let w_d = Vector3f::from(lin_w * scale - w as f32);

        let r_x1 = lerp(self.vec3(x, y, z, w), self.vec3(x_n, y, z, w), x_d);
        let r_x2 = lerp(self.vec3(x, y_n, z, w), self.vec3(x_n, y_n, z, w), x_d);
        let r_y1 = lerp(r_x1, r_x2, y_d);
        let r_x3 = lerp(self.vec3(x, y, z_n, w), self.vec3(x_n, y, z_n, w), x_d);
        let r_x4 = lerp(self.vec3(x, y_n, z_n, w), self.vec3(x_n, y_n, z_n, w), x_d);
        let r_y2 = lerp(r_x3, r_x4, y_d);
        let r_z1 = lerp(r_y1, r_y2, z_d);

        let r_x1 = lerp(self.vec3(x, y, z, w_n), self.vec3(x_n, y, z, w_n), x_d);
        let r_x2 = lerp(self.vec3(x, y_n, z, w_n), self.vec3(x_n, y_n, z, w_n), x_d);
        let r_y1 = lerp(r_x1, r_x2, y_d);
        let r_x3 = lerp(self.vec3(x, y, z_n, w_n), self.vec3(x_n, y, z_n, w_n), x_d);
        let r_x4 = lerp(
            self.vec3(x, y_n, z_n, w_n),
            self.vec3(x_n, y_n, z_n, w_n),
            x_d,
        );
        let r_y2 = lerp(r_x3, r_x4, y_d);
        let r_z2 = lerp(r_y1, r_y2, z_d);
        lerp(r_z1, r_z2, w_d)
    }
}

/// 3D CLUT helper
pub struct Array3D<'a> {
    array: &'a [f32],
    x_stride: u32,
    y_stride: u32,
    grid_size: usize,
}

impl Array3D<'_> {
    pub fn new(array: &[f32], grid_size: usize) -> Array3D {
        let y_stride = grid_size;
        let x_stride = y_stride * y_stride;
        Array3D {
            array,
            x_stride: x_stride as u32,
            y_stride: y_stride as u32,
            grid_size,
        }
    }

    #[inline]
    pub fn vec3(&self, x: i32, y: i32, z: i32) -> Vector3f {
        let start = (x as u32 * self.x_stride + y as u32 * self.y_stride + z as u32) as usize * 3;
        let k = &self.array[start..start + 3];
        Vector3f {
            v: [k[0], k[1], k[2]],
        }
    }

    #[inline]
    pub fn trilinear_interpolation(&self, lin_x: f32, lin_y: f32, lin_z: f32) -> Vector3f {
        let scale = (self.grid_size as i32 - 1) as f32;

        let x = (lin_x * scale).floor() as i32;
        let y = (lin_y * scale).floor() as i32;
        let z = (lin_z * scale).floor() as i32;

        let x_n = (lin_x * scale).ceil() as i32;
        let y_n = (lin_y * scale).ceil() as i32;
        let z_n = (lin_z * scale).ceil() as i32;

        let x_d = Vector3f::from(lin_x * scale - x as f32);
        let y_d = Vector3f::from(lin_y * scale - y as f32);
        let z_d = Vector3f::from(lin_z * scale - z as f32);

        let c000 = self.vec3(x, y, z);
        let c100 = self.vec3(x_n, y, z);
        let c010 = self.vec3(x, y_n, z);
        let c110 = self.vec3(x_n, y_n, z);
        let c001 = self.vec3(x, y, z_n);
        let c101 = self.vec3(x_n, y, z_n);
        let c011 = self.vec3(x, y_n, z_n);
        let c111 = self.vec3(x_n, y_n, z_n);

        // Perform trilinear interpolation
        let c00 = c000 * (Vector3f::from(1.0) - x_d) + c100 * x_d;
        let c10 = c010 * (Vector3f::from(1.0) - x_d) + c110 * x_d;
        let c01 = c001 * (Vector3f::from(1.0) - x_d) + c101 * x_d;
        let c11 = c011 * (Vector3f::from(1.0) - x_d) + c111 * x_d;

        let c0 = c00 * (Vector3f::from(1.0) - y_d) + c10 * y_d;
        let c1 = c01 * (Vector3f::from(1.0) - y_d) + c11 * y_d;

        c0 * (Vector3f::from(1.0) - z_d) + c1 * z_d
    }
}
