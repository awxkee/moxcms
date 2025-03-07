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
use crate::math::FusedMultiplyAdd;
use crate::{Vector3f, Vector4f};
use std::ops::{Add, Mul, Sub};

#[inline(always)]
pub(crate) fn lerp<
    T: Mul<Output = T> + Sub<Output = T> + Add<Output = T> + From<f32> + Copy + FusedMultiplyAdd<T>,
>(
    a: T,
    b: T,
    t: T,
) -> T {
    (a * (T::from(1.0) - t)).mla(b, t)
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

trait ArrayFetch<T> {
    fn fetch(&self, x: i32, y: i32, z: i32) -> T;
}

struct ArrayFetchVector3f<'a> {
    array: &'a [f32],
    x_stride: u32,
    y_stride: u32,
}

impl ArrayFetch<Vector3f> for ArrayFetchVector3f<'_> {
    #[inline]
    fn fetch(&self, x: i32, y: i32, z: i32) -> Vector3f {
        let start = (x as u32 * self.x_stride + y as u32 * self.y_stride + z as u32) as usize * 3;
        let k = &self.array[start..start + 3];
        Vector3f {
            v: [k[0], k[1], k[2]],
        }
    }
}

struct ArrayFetchVector4f<'a> {
    array: &'a [f32],
    x_stride: u32,
    y_stride: u32,
}

impl ArrayFetch<Vector4f> for ArrayFetchVector4f<'_> {
    #[inline]
    fn fetch(&self, x: i32, y: i32, z: i32) -> Vector4f {
        let start = (x as u32 * self.x_stride + y as u32 * self.y_stride + z as u32) as usize * 4;
        let k = &self.array[start..start + 4];
        Vector4f {
            v: [k[0], k[1], k[2], k[3]],
        }
    }
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
    pub fn vec4(&self, x: i32, y: i32, z: i32) -> Vector4f {
        let start = (x as u32 * self.x_stride + y as u32 * self.y_stride + z as u32) as usize * 4;
        let k = &self.array[start..start + 4];
        Vector4f {
            v: [k[0], k[1], k[2], k[3]],
        }
    }

    #[inline]
    fn trilinear<
        T: Copy + From<f32> + Sub<T, Output = T> + Mul<T, Output = T> + Add<T, Output = T>,
    >(
        &self,
        lin_x: f32,
        lin_y: f32,
        lin_z: f32,
        fetch: impl ArrayFetch<T>,
    ) -> T {
        let scale = (self.grid_size as i32 - 1) as f32;

        let x = (lin_x * scale).floor() as i32;
        let y = (lin_y * scale).floor() as i32;
        let z = (lin_z * scale).floor() as i32;

        let x_n = (lin_x * scale).ceil() as i32;
        let y_n = (lin_y * scale).ceil() as i32;
        let z_n = (lin_z * scale).ceil() as i32;

        let x_d = T::from(lin_x * scale - x as f32);
        let y_d = T::from(lin_y * scale - y as f32);
        let z_d = T::from(lin_z * scale - z as f32);

        let c000 = fetch.fetch(x, y, z);
        let c100 = fetch.fetch(x_n, y, z);
        let c010 = fetch.fetch(x, y_n, z);
        let c110 = fetch.fetch(x_n, y_n, z);
        let c001 = fetch.fetch(x, y, z_n);
        let c101 = fetch.fetch(x_n, y, z_n);
        let c011 = fetch.fetch(x, y_n, z_n);
        let c111 = fetch.fetch(x_n, y_n, z_n);

        // Perform trilinear interpolation
        let c00 = c000 * (T::from(1.0) - x_d) + c100 * x_d;
        let c10 = c010 * (T::from(1.0) - x_d) + c110 * x_d;
        let c01 = c001 * (T::from(1.0) - x_d) + c101 * x_d;
        let c11 = c011 * (T::from(1.0) - x_d) + c111 * x_d;

        let c0 = c00 * (T::from(1.0) - y_d) + c10 * y_d;
        let c1 = c01 * (T::from(1.0) - y_d) + c11 * y_d;

        c0 * (T::from(1.0) - z_d) + c1 * z_d
    }

    #[inline]
    pub fn trilinear_vec3(&self, lin_x: f32, lin_y: f32, lin_z: f32) -> Vector3f {
        self.trilinear(
            lin_x,
            lin_y,
            lin_z,
            ArrayFetchVector3f {
                array: self.array,
                x_stride: self.x_stride,
                y_stride: self.y_stride,
            },
        )
    }

    #[inline]
    pub fn trilinear_vec4(&self, lin_x: f32, lin_y: f32, lin_z: f32) -> Vector4f {
        self.trilinear(
            lin_x,
            lin_y,
            lin_z,
            ArrayFetchVector4f {
                array: self.array,
                x_stride: self.x_stride,
                y_stride: self.y_stride,
            },
        )
    }
}
