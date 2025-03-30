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
use crate::math::{FusedMultiplyAdd, FusedMultiplyNegAdd};
use crate::matrix::Vector4i;
use crate::{Vector3f, Vector3i, Vector4f};
use std::ops::{Add, Mul, Shr, Sub};

const RND: i32 = (1 << 14) - 1;
const SCALE: i32 = (1 << 15) - 1;
const Q: i32 = 15;

#[inline(always)]
fn lerp16<
    T: Mul<Output = T>
        + Sub<Output = T>
        + Add<Output = T>
        + From<i32>
        + Copy
        + FusedMultiplyAdd<T>
        + FusedMultiplyNegAdd<T>
        + Shr<i32, Output = T>,
>(
    a: T,
    b: T,
    t: T,
) -> T {
    let n_t = T::from(SCALE) - t;
    let q = n_t * a;
    (q + b * t + T::from(RND)) >> Q
}

/// 4D CLUT helper for interpolate values in Q0.15
///
/// Represents hypercube.
pub struct Array4DS16<'a> {
    array: &'a [u16],
    x_stride: u32,
    y_stride: u32,
    z_stride: u32,
    grid_size: [u8; 4],
}

trait Fetcher4<T> {
    fn fetch(&self, x: i32, y: i32, z: i32, w: i32) -> T;
}

impl Array4DS16<'_> {
    pub fn new(array: &[u16], grid_size: usize) -> Array4DS16 {
        let z_stride = grid_size as u32;
        let y_stride = z_stride * z_stride;
        let x_stride = z_stride * z_stride * z_stride;
        Array4DS16 {
            array,
            x_stride,
            y_stride,
            z_stride,
            grid_size: [
                grid_size as u8,
                grid_size as u8,
                grid_size as u8,
                grid_size as u8,
            ],
        }
    }

    pub fn new_hypercube(array: &[u16], grid_size: [u8; 4]) -> Array4DS16 {
        let z_stride = grid_size[2] as u32;
        let y_stride = z_stride * grid_size[1] as u32;
        let x_stride = y_stride * grid_size[0] as u32;
        Array4DS16 {
            array,
            x_stride,
            y_stride,
            z_stride,
            grid_size,
        }
    }
}

struct Fetch4Vec3<'a> {
    array: &'a [u16],
    x_stride: u32,
    y_stride: u32,
    z_stride: u32,
}

struct Fetch4Vec4<'a> {
    array: &'a [u16],
    x_stride: u32,
    y_stride: u32,
    z_stride: u32,
}

impl Fetcher4<Vector3i> for Fetch4Vec3<'_> {
    #[inline(always)]
    fn fetch(&self, x: i32, y: i32, z: i32, w: i32) -> Vector3i {
        let start = (x as u32 * self.x_stride
            + y as u32 * self.y_stride
            + z as u32 * self.z_stride
            + w as u32) as usize
            * 3;
        let k = &self.array[start..start + 3];
        Vector3i {
            v: [k[0] as i32, k[1] as i32, k[2] as i32],
        }
    }
}

impl Fetcher4<Vector4i> for Fetch4Vec4<'_> {
    #[inline(always)]
    fn fetch(&self, x: i32, y: i32, z: i32, w: i32) -> Vector4i {
        let start = (x as u32 * self.x_stride
            + y as u32 * self.y_stride
            + z as u32 * self.z_stride
            + w as u32) as usize
            * 4;
        let k = &self.array[start..start + 4];
        Vector4i {
            v: [k[0] as i32, k[1] as i32, k[2] as i32, k[3] as i32],
        }
    }
}

impl Array4DS16<'_> {
    #[inline(always)]
    fn quadlinear<
        T: From<i32>
            + Add<T, Output = T>
            + Mul<T, Output = T>
            + FusedMultiplyAdd<T>
            + Sub<T, Output = T>
            + Copy
            + FusedMultiplyNegAdd<T>
            + Shr<i32, Output = T>,
    >(
        &self,
        lin_x: f32,
        lin_y: f32,
        lin_z: f32,
        lin_w: f32,
        r: impl Fetcher4<T>,
    ) -> T {
        let scale_x = (self.grid_size[0] as i32 - 1) as f32;
        let scale_y = (self.grid_size[1] as i32 - 1) as f32;
        let scale_z = (self.grid_size[2] as i32 - 1) as f32;
        let scale_w = (self.grid_size[3] as i32 - 1) as f32;

        let x = (lin_x * scale_x).floor() as i32;
        let y = (lin_y * scale_y).floor() as i32;
        let z = (lin_z * scale_z).floor() as i32;
        let w = (lin_w * scale_w).floor() as i32;

        let x_n = (lin_x * scale_x).ceil() as i32;
        let y_n = (lin_y * scale_y).ceil() as i32;
        let z_n = (lin_z * scale_z).ceil() as i32;
        let w_n = (lin_w * scale_w).ceil() as i32;

        let x_d = T::from(((lin_x * scale_x - x as f32) * SCALE as f32 + 0.5) as i32);
        let y_d = T::from(((lin_y * scale_y - y as f32) * SCALE as f32 + 0.5) as i32);
        let z_d = T::from(((lin_z * scale_z - z as f32) * SCALE as f32 + 0.5) as i32);
        let w_d = T::from(((lin_w * scale_w - w as f32) * SCALE as f32 + 0.5) as i32);

        let r_x1 = lerp16(r.fetch(x, y, z, w), r.fetch(x_n, y, z, w), x_d);
        let r_x2 = lerp16(r.fetch(x, y_n, z, w), r.fetch(x_n, y_n, z, w), x_d);
        let r_y1 = lerp16(r_x1, r_x2, y_d);
        let r_x3 = lerp16(r.fetch(x, y, z_n, w), r.fetch(x_n, y, z_n, w), x_d);
        let r_x4 = lerp16(r.fetch(x, y_n, z_n, w), r.fetch(x_n, y_n, z_n, w), x_d);
        let r_y2 = lerp16(r_x3, r_x4, y_d);
        let r_z1 = lerp16(r_y1, r_y2, z_d);

        let r_x1 = lerp16(r.fetch(x, y, z, w_n), r.fetch(x_n, y, z, w_n), x_d);
        let r_x2 = lerp16(r.fetch(x, y_n, z, w_n), r.fetch(x_n, y_n, z, w_n), x_d);
        let r_y1 = lerp16(r_x1, r_x2, y_d);
        let r_x3 = lerp16(r.fetch(x, y, z_n, w_n), r.fetch(x_n, y, z_n, w_n), x_d);
        let r_x4 = lerp16(r.fetch(x, y_n, z_n, w_n), r.fetch(x_n, y_n, z_n, w_n), x_d);
        let r_y2 = lerp16(r_x3, r_x4, y_d);
        let r_z2 = lerp16(r_y1, r_y2, z_d);
        lerp16(r_z1, r_z2, w_d)
    }

    #[inline]
    pub fn quadlinear_vec3(&self, lin_x: u16, lin_y: u16, lin_z: u16, lin_w: u16) -> Vector3i {
        self.quadlinear(
            lin_x as f32 * (1. / 65535.),
            lin_y as f32 * (1. / 65535.),
            lin_z as f32 * (1. / 65535.),
            lin_w as f32 * (1. / 65535.),
            Fetch4Vec3 {
                array: self.array,
                x_stride: self.x_stride,
                y_stride: self.y_stride,
                z_stride: self.z_stride,
            },
        )
    }

    #[inline]
    pub fn quadlinear_vec4(&self, lin_x: u16, lin_y: u16, lin_z: u16, lin_w: u16) -> Vector4i {
        self.quadlinear(
            lin_x as f32 * (1. / 65535.),
            lin_y as f32 * (1. / 65535.),
            lin_z as f32 * (1. / 65535.),
            lin_w as f32 * (1. / 65535.),
            Fetch4Vec4 {
                array: self.array,
                x_stride: self.x_stride,
                y_stride: self.y_stride,
                z_stride: self.z_stride,
            },
        )
    }

    #[cfg(feature = "options")]
    #[inline(always)]
    fn pyramid<
        T: From<i32>
            + Add<T, Output = T>
            + Mul<T, Output = T>
            + FusedMultiplyAdd<T>
            + Sub<T, Output = T>
            + Copy
            + FusedMultiplyNegAdd<T>
            + Shr<i32, Output = T>,
    >(
        &self,
        lin_x: f32,
        lin_y: f32,
        lin_z: f32,
        lin_w: f32,
        r: impl Fetcher4<T>,
    ) -> T {
        let scale_x = (self.grid_size[0] as i32 - 1) as f32;
        let scale_y = (self.grid_size[1] as i32 - 1) as f32;
        let scale_z = (self.grid_size[2] as i32 - 1) as f32;
        let scale_w = (self.grid_size[3] as i32 - 1) as f32;

        let x = (lin_x * scale_x).floor() as i32;
        let y = (lin_y * scale_y).floor() as i32;
        let z = (lin_z * scale_z).floor() as i32;
        let w = (lin_w * scale_w).floor() as i32;

        let x_n = (lin_x * scale_x).ceil() as i32;
        let y_n = (lin_y * scale_y).ceil() as i32;
        let z_n = (lin_z * scale_z).ceil() as i32;
        let w_n = (lin_w * scale_w).ceil() as i32;

        let dr = ((lin_x * scale_x - x as f32) * SCALE as f32 + 0.5) as i32;
        let dg = ((lin_y * scale_y - y as f32) * SCALE as f32 + 0.5) as i32;
        let db = ((lin_z * scale_z - z as f32) * SCALE as f32 + 0.5) as i32;
        let dw = ((lin_w * scale_w - w as f32) * SCALE as f32 + 0.5) as i32;

        let c0 = r.fetch(x, y, z, w);

        let w0 = if dr > db && dg > db {
            let x0 = r.fetch(x_n, y_n, z_n, w);
            let x1 = r.fetch(x_n, y_n, z, w);
            let x2 = r.fetch(x_n, y, z, w);
            let x3 = r.fetch(x, y_n, z, w);

            let c1 = x0 - x1;
            let c2 = x2 - c0;
            let c3 = x3 - c0;
            let c4 = c0 - x3 - x2 + x1;

            let s0 = c1 * T::from(db);
            let s1 = c2 * T::from(dr);
            let s2 = c3 * T::from(dg);
            (((c4 * (T::from(dr * dg + RND) >> Q)) + s0 + s1 + s2 + T::from(RND)) >> Q) + c0
        } else if db > dr && dg > dr {
            let x0 = r.fetch(x, y, z_n, w);
            let x1 = r.fetch(x_n, y_n, z_n, w);
            let x2 = r.fetch(x, y_n, z_n, w);
            let x3 = r.fetch(x, y_n, z, w);

            let c1 = x0 - c0;
            let c2 = x1 - x2;
            let c3 = x3 - c0;
            let c4 = c0 - x3 - x0 + x2;

            let s0 = c1 * T::from(db);
            let s1 = c2 * T::from(dr);
            let s2 = c3 * T::from(dg);
            ((c4 * (T::from((dg * db + RND) >> Q)) + s0 + s1 + s2 + T::from(RND)) >> Q) + c0
        } else {
            let x0 = r.fetch(x, y, z_n, w);
            let x1 = r.fetch(x_n, y, z, w);
            let x2 = r.fetch(x_n, y, z_n, w);
            let x3 = r.fetch(x_n, y_n, z_n, w);

            let c1 = x0 - c0;
            let c2 = x1 - c0;
            let c3 = x3 - x2;
            let c4 = c0 - x1 - x0 + x2;

            let s0 = c1 * T::from(db);
            let s1 = c2 * T::from(dr);
            let s2 = c3 * T::from(dg);
            ((c4 * (T::from((db * dr + RND) >> Q)) + s0 + s1 + s2 + T::from(RND)) >> Q) + c0
        };

        let c0 = r.fetch(x, y, z, w_n);

        let w1 = if dr > db && dg > db {
            let x0 = r.fetch(x_n, y_n, z_n, w_n);
            let x1 = r.fetch(x_n, y_n, z, w_n);
            let x2 = r.fetch(x_n, y, z, w_n);
            let x3 = r.fetch(x, y_n, z, w_n);

            let c1 = x0 - x1;
            let c2 = x2 - c0;
            let c3 = x3 - c0;
            let c4 = c0 - x3 - x2 + x1;

            let s0 = c1 * T::from(db);
            let s1 = c2 * T::from(dr);
            let s2 = c3 * T::from(dg);
            (((c4 * (T::from(dr * dg + RND) >> Q)) + s0 + s1 + s2 + T::from(RND)) >> Q) + c0
        } else if db > dr && dg > dr {
            let x0 = r.fetch(x, y, z_n, w_n);
            let x1 = r.fetch(x_n, y_n, z_n, w_n);
            let x2 = r.fetch(x, y_n, z_n, w_n);
            let x3 = r.fetch(x, y_n, z, w_n);

            let c1 = x0 - c0;
            let c2 = x1 - x2;
            let c3 = x3 - c0;
            let c4 = c0 - x3 - x0 + x2;

            let s0 = c1 * T::from(db);
            let s1 = c2 * T::from(dr);
            let s2 = c3 * T::from(dg);
            ((c4 * (T::from((dg * db + RND) >> Q)) + s0 + s1 + s2 + T::from(RND)) >> Q) + c0
        } else {
            let x0 = r.fetch(x, y, z_n, w_n);
            let x1 = r.fetch(x_n, y, z, w_n);
            let x2 = r.fetch(x_n, y, z_n, w_n);
            let x3 = r.fetch(x_n, y_n, z_n, w_n);

            let c1 = x0 - c0;
            let c2 = x1 - c0;
            let c3 = x3 - x2;
            let c4 = c0 - x1 - x0 + x2;

            let s0 = c1 * T::from(db);
            let s1 = c2 * T::from(dr);
            let s2 = c3 * T::from(dg);
            ((c4 * (T::from((db * dr + RND) >> Q)) + s0 + s1 + s2 + T::from(RND)) >> Q) + c0
        };
        let ndw = T::from(SCALE - dw);
        (w0 * ndw + w1 * T::from(dw) + T::from(RND)) >> Q
    }

    #[cfg(feature = "options")]
    #[inline]
    pub fn pyramid_vec3(&self, lin_x: u16, lin_y: u16, lin_z: u16, lin_w: u16) -> Vector3i {
        self.pyramid(
            lin_x as f32 * (1. / 65535.),
            lin_y as f32 * (1. / 65535.),
            lin_z as f32 * (1. / 65535.),
            lin_w as f32 * (1. / 65535.),
            Fetch4Vec3 {
                array: self.array,
                x_stride: self.x_stride,
                y_stride: self.y_stride,
                z_stride: self.z_stride,
            },
        )
    }

    #[cfg(feature = "options")]
    #[inline]
    pub fn pyramid_vec4(&self, lin_x: u16, lin_y: u16, lin_z: u16, lin_w: u16) -> Vector4i {
        self.pyramid(
            lin_x as f32 * (1. / 65535.),
            lin_y as f32 * (1. / 65535.),
            lin_z as f32 * (1. / 65535.),
            lin_w as f32 * (1. / 65535.),
            Fetch4Vec4 {
                array: self.array,
                x_stride: self.x_stride,
                y_stride: self.y_stride,
                z_stride: self.z_stride,
            },
        )
    }

    #[cfg(feature = "options")]
    #[inline(always)]
    fn prism<
        T: From<i32>
            + Add<T, Output = T>
            + Mul<T, Output = T>
            + FusedMultiplyAdd<T>
            + Sub<T, Output = T>
            + Copy
            + Shr<i32, Output = T>
            + FusedMultiplyNegAdd<T>,
    >(
        &self,
        lin_x: f32,
        lin_y: f32,
        lin_z: f32,
        lin_w: f32,
        r: impl Fetcher4<T>,
    ) -> T {
        let scale_x = (self.grid_size[0] as i32 - 1) as f32;
        let scale_y = (self.grid_size[1] as i32 - 1) as f32;
        let scale_z = (self.grid_size[2] as i32 - 1) as f32;
        let scale_w = (self.grid_size[3] as i32 - 1) as f32;

        let x = (lin_x * scale_x).floor() as i32;
        let y = (lin_y * scale_y).floor() as i32;
        let z = (lin_z * scale_z).floor() as i32;
        let w = (lin_w * scale_w).floor() as i32;

        let x_n = (lin_x * scale_x).ceil() as i32;
        let y_n = (lin_y * scale_y).ceil() as i32;
        let z_n = (lin_z * scale_z).ceil() as i32;
        let w_n = (lin_w * scale_w).ceil() as i32;

        let dr = ((lin_x * scale_x - x as f32) * SCALE as f32 + 0.5) as i32;
        let dg = ((lin_y * scale_y - y as f32) * SCALE as f32 + 0.5) as i32;
        let db = ((lin_z * scale_z - z as f32) * SCALE as f32 + 0.5) as i32;
        let dw = ((lin_w * scale_w - w as f32) * SCALE as f32 + 0.5) as i32;

        let c0 = r.fetch(x, y, z, w);

        let w0 = if db >= dr {
            let x0 = r.fetch(x, y, z_n, w);
            let x1 = r.fetch(x_n, y, z_n, w);
            let x2 = r.fetch(x, y_n, z, w);
            let x3 = r.fetch(x, y_n, z_n, w);
            let x4 = r.fetch(x_n, y_n, z_n, w);

            let c1 = x0 - c0;
            let c2 = x1 - x0;
            let c3 = x2 - c0;
            let c4 = c0 - x2 - x0 + x3;
            let c5 = x0 - x3 - x1 + x4;

            let s0 = c1 * T::from(db);
            let s1 = c2 * T::from(dr);
            let s2 = c3 * T::from(dg);
            let s3 = c4 * T::from((dg * db + RND) >> Q);
            ((c5 * T::from((dr * dg + RND) >> Q) + s0 + s1 + s2 + s3 + T::from(RND)) >> Q) + c0
        } else {
            let x0 = r.fetch(x_n, y, z, w);
            let x1 = r.fetch(x_n, y, z_n, w);
            let x2 = r.fetch(x, y_n, z, w);
            let x3 = r.fetch(x_n, y_n, z, w);
            let x4 = r.fetch(x_n, y_n, z_n, w);

            let c1 = x1 - x0;
            let c2 = x0 - c0;
            let c3 = x2 - c0;
            let c4 = x0 - x3 - x1 + x4;
            let c5 = c0 - x2 - x0 + x3;

            let s0 = c1 * T::from(db);
            let s1 = c2 * T::from(dr);
            let s2 = c3 * T::from(dg);
            let s3 = c4 * T::from((dg * db + RND) >> Q);
            ((c5 * T::from((dr * dg + RND) >> Q) + s0 + s1 + s2 + s3 + T::from(RND)) >> Q) + c0
        };

        let c0 = r.fetch(x, y, z, w_n);

        let w1 = if db >= dr {
            let x0 = r.fetch(x, y, z_n, w_n);
            let x1 = r.fetch(x_n, y, z_n, w_n);
            let x2 = r.fetch(x, y_n, z, w_n);
            let x3 = r.fetch(x, y_n, z_n, w_n);
            let x4 = r.fetch(x_n, y_n, z_n, w_n);

            let c1 = x0 - c0;
            let c2 = x1 - x0;
            let c3 = x2 - c0;
            let c4 = c0 - x2 - x0 + x3;
            let c5 = x0 - x3 - x1 + x4;

            let s0 = c1 * T::from(db);
            let s1 = c2 * T::from(dr);
            let s2 = c3 * T::from(dg);
            let s3 = c4 * T::from((dg * db + RND) >> Q);
            ((c5 * T::from((dr * dg + RND) >> Q) + s0 + s1 + s2 + s3 + T::from(RND)) >> Q) + c0
        } else {
            let x0 = r.fetch(x_n, y, z, w_n);
            let x1 = r.fetch(x_n, y, z_n, w_n);
            let x2 = r.fetch(x, y_n, z, w_n);
            let x3 = r.fetch(x_n, y_n, z, w_n);
            let x4 = r.fetch(x_n, y_n, z_n, w_n);

            let c1 = x1 - x0;
            let c2 = x0 - c0;
            let c3 = x2 - c0;
            let c4 = x0 - x3 - x1 + x4;
            let c5 = c0 - x2 - x0 + x3;

            let s0 = c1 * T::from(db);
            let s1 = c2 * T::from(dr);
            let s2 = c3 * T::from(dg);
            let s3 = c4 * T::from((dg * db + RND) >> Q);
            ((c5 * T::from((dr * dg + RND) >> Q) + s0 + s1 + s2 + s3 + T::from(RND)) >> Q) + c0
        };
        let ndw = T::from(SCALE - dw);
        (w0 * ndw + w1 * T::from(dw) + T::from(RND)) >> Q
    }

    #[cfg(feature = "options")]
    #[inline]
    pub fn prism_vec3(&self, lin_x: u16, lin_y: u16, lin_z: u16, lin_w: u16) -> Vector3i {
        self.prism(
            lin_x as f32 * (1. / 65535.),
            lin_y as f32 * (1. / 65535.),
            lin_z as f32 * (1. / 65535.),
            lin_w as f32 * (1. / 65535.),
            Fetch4Vec3 {
                array: self.array,
                x_stride: self.x_stride,
                y_stride: self.y_stride,
                z_stride: self.z_stride,
            },
        )
    }

    #[cfg(feature = "options")]
    #[inline]
    pub fn prism_vec4(&self, lin_x: u16, lin_y: u16, lin_z: u16, lin_w: u16) -> Vector4i {
        self.prism(
            lin_x as f32 * (1. / 65535.),
            lin_y as f32 * (1. / 65535.),
            lin_z as f32 * (1. / 65535.),
            lin_w as f32 * (1. / 65535.),
            Fetch4Vec4 {
                array: self.array,
                x_stride: self.x_stride,
                y_stride: self.y_stride,
                z_stride: self.z_stride,
            },
        )
    }

    #[cfg(feature = "options")]
    #[inline(always)]
    fn tetra<
        T: From<i32>
            + Add<T, Output = T>
            + Mul<T, Output = T>
            + FusedMultiplyAdd<T>
            + Sub<T, Output = T>
            + Copy
            + Shr<i32, Output = T>
            + FusedMultiplyNegAdd<T>,
    >(
        &self,
        lin_x: f32,
        lin_y: f32,
        lin_z: f32,
        lin_w: f32,
        r: impl Fetcher4<T>,
    ) -> T {
        let scale_x = (self.grid_size[0] as i32 - 1) as f32;
        let scale_y = (self.grid_size[1] as i32 - 1) as f32;
        let scale_z = (self.grid_size[2] as i32 - 1) as f32;
        let scale_w = (self.grid_size[3] as i32 - 1) as f32;

        let x = (lin_x * scale_x).floor() as i32;
        let y = (lin_y * scale_y).floor() as i32;
        let z = (lin_z * scale_z).floor() as i32;
        let w = (lin_w * scale_w).floor() as i32;

        let x_n = (lin_x * scale_x).ceil() as i32;
        let y_n = (lin_y * scale_y).ceil() as i32;
        let z_n = (lin_z * scale_z).ceil() as i32;
        let w_n = (lin_w * scale_w).ceil() as i32;

        let rx = ((lin_x * scale_x - x as f32) * SCALE as f32 + 0.5) as i32;
        let ry = ((lin_y * scale_y - y as f32) * SCALE as f32 + 0.5) as i32;
        let rz = ((lin_z * scale_z - z as f32) * SCALE as f32 + 0.5) as i32;
        let rw = ((lin_w * scale_w - w as f32) * SCALE as f32 + 0.5) as i32;

        let c0 = r.fetch(x, y, z, w);
        let c2;
        let c1;
        let c3;
        if rx >= ry {
            if ry >= rz {
                //rx >= ry && ry >= rz
                c1 = r.fetch(x_n, y, z, w) - c0;
                c2 = r.fetch(x_n, y_n, z, w) - r.fetch(x_n, y, z, w);
                c3 = r.fetch(x_n, y_n, z_n, w) - r.fetch(x_n, y_n, z, w);
            } else if rx >= rz {
                //rx >= rz && rz >= ry
                c1 = r.fetch(x_n, y, z, w) - c0;
                c2 = r.fetch(x_n, y_n, z_n, w) - r.fetch(x_n, y, z_n, w);
                c3 = r.fetch(x_n, y, z_n, w) - r.fetch(x_n, y, z, w);
            } else {
                //rz > rx && rx >= ry
                c1 = r.fetch(x_n, y, z_n, w) - r.fetch(x, y, z_n, w);
                c2 = r.fetch(x_n, y_n, z_n, w) - r.fetch(x_n, y, z_n, w);
                c3 = r.fetch(x, y, z_n, w) - c0;
            }
        } else if rx >= rz {
            //ry > rx && rx >= rz
            c1 = r.fetch(x_n, y_n, z, w) - r.fetch(x, y_n, z, w);
            c2 = r.fetch(x, y_n, z, w) - c0;
            c3 = r.fetch(x_n, y_n, z_n, w) - r.fetch(x_n, y_n, z, w);
        } else if ry >= rz {
            //ry >= rz && rz > rx
            c1 = r.fetch(x_n, y_n, z_n, w) - r.fetch(x, y_n, z_n, w);
            c2 = r.fetch(x, y_n, z, w) - c0;
            c3 = r.fetch(x, y_n, z_n, w) - r.fetch(x, y_n, z, w);
        } else {
            //rz > ry && ry > rx
            c1 = r.fetch(x_n, y_n, z_n, w) - r.fetch(x, y_n, z_n, w);
            c2 = r.fetch(x, y_n, z_n, w) - r.fetch(x, y, z_n, w);
            c3 = r.fetch(x, y, z_n, w) - c0;
        }
        let s0 = c1 * T::from(rx);
        let s1 = c2 * T::from(ry);
        let w0 = ((c3 * T::from(rz) + s0 + s1 + T::from(RND)) >> Q) + c0;

        let c0 = r.fetch(x, y, z, w_n);
        let c2;
        let c1;
        let c3;
        if rx >= ry {
            if ry >= rz {
                //rx >= ry && ry >= rz
                c1 = r.fetch(x_n, y, z, w_n) - c0;
                c2 = r.fetch(x_n, y_n, z, w_n) - r.fetch(x_n, y, z, w_n);
                c3 = r.fetch(x_n, y_n, z_n, w_n) - r.fetch(x_n, y_n, z, w_n);
            } else if rx >= rz {
                //rx >= rz && rz >= ry
                c1 = r.fetch(x_n, y, z, w_n) - c0;
                c2 = r.fetch(x_n, y_n, z_n, w_n) - r.fetch(x_n, y, z_n, w_n);
                c3 = r.fetch(x_n, y, z_n, w_n) - r.fetch(x_n, y, z, w_n);
            } else {
                //rz > rx && rx >= ry
                c1 = r.fetch(x_n, y, z_n, w_n) - r.fetch(x, y, z_n, w_n);
                c2 = r.fetch(x_n, y_n, z_n, w_n) - r.fetch(x_n, y, z_n, w_n);
                c3 = r.fetch(x, y, z_n, w_n) - c0;
            }
        } else if rx >= rz {
            //ry > rx && rx >= rz
            c1 = r.fetch(x_n, y_n, z, w_n) - r.fetch(x, y_n, z, w_n);
            c2 = r.fetch(x, y_n, z, w_n) - c0;
            c3 = r.fetch(x_n, y_n, z_n, w_n) - r.fetch(x_n, y_n, z, w_n);
        } else if ry >= rz {
            //ry >= rz && rz > rx
            c1 = r.fetch(x_n, y_n, z_n, w_n) - r.fetch(x, y_n, z_n, w_n);
            c2 = r.fetch(x, y_n, z, w_n) - c0;
            c3 = r.fetch(x, y_n, z_n, w_n) - r.fetch(x, y_n, z, w_n);
        } else {
            //rz > ry && ry > rx
            c1 = r.fetch(x_n, y_n, z_n, w_n) - r.fetch(x, y_n, z_n, w_n);
            c2 = r.fetch(x, y_n, z_n, w_n) - r.fetch(x, y, z_n, w_n);
            c3 = r.fetch(x, y, z_n, w_n) - c0;
        }
        let s0 = c1 * T::from(rx);
        let s1 = c2 * T::from(ry);
        let w1 = ((c3 * T::from(rz) + s0 + s1 + T::from(RND)) >> Q) + c0;

        let ndw = T::from(SCALE - rw);
        (w0 * ndw + w1 * T::from(rw) + T::from(RND)) >> Q
    }

    #[cfg(feature = "options")]
    #[inline]
    pub fn tetra_vec3(&self, lin_x: u16, lin_y: u16, lin_z: u16, lin_w: u16) -> Vector3i {
        self.tetra(
            lin_x as f32 * (1. / 65535.),
            lin_y as f32 * (1. / 65535.),
            lin_z as f32 * (1. / 65535.),
            lin_w as f32 * (1. / 65535.),
            Fetch4Vec3 {
                array: self.array,
                x_stride: self.x_stride,
                y_stride: self.y_stride,
                z_stride: self.z_stride,
            },
        )
    }

    #[cfg(feature = "options")]
    #[inline]
    pub fn tetra_vec4(&self, lin_x: u16, lin_y: u16, lin_z: u16, lin_w: u16) -> Vector4i {
        self.tetra(
            lin_x as f32 * (1. / 65535.),
            lin_y as f32 * (1. / 65535.),
            lin_z as f32 * (1. / 65535.),
            lin_w as f32 * (1. / 65535.),
            Fetch4Vec4 {
                array: self.array,
                x_stride: self.x_stride,
                y_stride: self.y_stride,
                z_stride: self.z_stride,
            },
        )
    }
}

/// 3D CLUT helper
///
/// Represents hexahedron.
pub struct Array3DS16<'a> {
    array: &'a [f32],
    x_stride: u32,
    y_stride: u32,
    grid_size: [u8; 3],
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
    #[inline(always)]
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
    #[inline(always)]
    fn fetch(&self, x: i32, y: i32, z: i32) -> Vector4f {
        let start = (x as u32 * self.x_stride + y as u32 * self.y_stride + z as u32) as usize * 4;
        let k = &self.array[start..start + 4];
        Vector4f {
            v: [k[0], k[1], k[2], k[3]],
        }
    }
}

impl Array3DS16<'_> {
    pub fn new(array: &[f32], grid_size: usize) -> Array3DS16 {
        let y_stride = grid_size;
        let x_stride = y_stride * y_stride;
        Array3DS16 {
            array,
            x_stride: x_stride as u32,
            y_stride: y_stride as u32,
            grid_size: [grid_size as u8, grid_size as u8, grid_size as u8],
        }
    }

    pub fn new_hexahedron(array: &[f32], grid_size: [u8; 3]) -> Array3DS16 {
        let y_stride = grid_size[1] as u32;
        let x_stride = y_stride * grid_size[0] as u32;
        Array3DS16 {
            array,
            x_stride,
            y_stride,
            grid_size,
        }
    }

    #[inline(always)]
    fn trilinear<
        T: Copy
            + From<f32>
            + Sub<T, Output = T>
            + Mul<T, Output = T>
            + Add<T, Output = T>
            + FusedMultiplyNegAdd<T>
            + FusedMultiplyAdd<T>,
    >(
        &self,
        lin_x: f32,
        lin_y: f32,
        lin_z: f32,
        fetch: impl ArrayFetch<T>,
    ) -> T {
        let scale_x = (self.grid_size[0] as i32 - 1) as f32;
        let scale_y = (self.grid_size[1] as i32 - 1) as f32;
        let scale_z = (self.grid_size[2] as i32 - 1) as f32;

        let x = (lin_x * scale_x).floor() as i32;
        let y = (lin_y * scale_y).floor() as i32;
        let z = (lin_z * scale_z).floor() as i32;

        let x_n = (lin_x * scale_x).ceil() as i32;
        let y_n = (lin_y * scale_y).ceil() as i32;
        let z_n = (lin_z * scale_z).ceil() as i32;

        let x_d = T::from(lin_x * scale_x - x as f32);
        let y_d = T::from(lin_y * scale_y - y as f32);
        let z_d = T::from(lin_z * scale_z - z as f32);

        let c000 = fetch.fetch(x, y, z);
        let c100 = fetch.fetch(x_n, y, z);
        let c010 = fetch.fetch(x, y_n, z);
        let c110 = fetch.fetch(x_n, y_n, z);
        let c001 = fetch.fetch(x, y, z_n);
        let c101 = fetch.fetch(x_n, y, z_n);
        let c011 = fetch.fetch(x, y_n, z_n);
        let c111 = fetch.fetch(x_n, y_n, z_n);

        let c00 = c000.neg_mla(c000, x_d).mla(c100, x_d);
        let c10 = c010.neg_mla(c010, x_d).mla(c110, x_d);
        let c01 = c001.neg_mla(c001, x_d).mla(c101, x_d);
        let c11 = c011.neg_mla(c011, x_d).mla(c111, x_d);

        let c0 = c00.neg_mla(c00, y_d).mla(c10, y_d);
        let c1 = c01.neg_mla(c01, y_d).mla(c11, y_d);

        c0.neg_mla(c0, z_d).mla(c1, z_d)
    }

    #[cfg(feature = "options")]
    #[inline]
    fn pyramid<
        T: Copy
            + From<f32>
            + Sub<T, Output = T>
            + Mul<T, Output = T>
            + Add<T, Output = T>
            + FusedMultiplyAdd<T>,
    >(
        &self,
        lin_x: f32,
        lin_y: f32,
        lin_z: f32,
        fetch: impl ArrayFetch<T>,
    ) -> T {
        let scale_x = (self.grid_size[0] as i32 - 1) as f32;
        let scale_y = (self.grid_size[1] as i32 - 1) as f32;
        let scale_z = (self.grid_size[2] as i32 - 1) as f32;

        let x = (lin_x * scale_x).floor() as i32;
        let y = (lin_y * scale_y).floor() as i32;
        let z = (lin_z * scale_z).floor() as i32;

        let x_n = (lin_x * scale_x).ceil() as i32;
        let y_n = (lin_y * scale_y).ceil() as i32;
        let z_n = (lin_z * scale_z).ceil() as i32;

        let dr = lin_x * scale_x - x as f32;
        let dg = lin_y * scale_y - y as f32;
        let db = lin_z * scale_z - z as f32;

        let c0 = fetch.fetch(x, y, z);

        if dr > db && dg > db {
            let x0 = fetch.fetch(x_n, y_n, z_n);
            let x1 = fetch.fetch(x_n, y_n, z);
            let x2 = fetch.fetch(x_n, y, z);
            let x3 = fetch.fetch(x, y_n, z);

            let c1 = x0 - x1;
            let c2 = x2 - c0;
            let c3 = x3 - c0;
            let c4 = c0 - x3 - x2 + x1;

            let s0 = c0.mla(c1, T::from(db));
            let s1 = s0.mla(c2, T::from(dr));
            let s2 = s1.mla(c3, T::from(dg));
            s2.mla(c4, T::from(dr * dg))
        } else if db > dr && dg > dr {
            let x0 = fetch.fetch(x, y, z_n);
            let x1 = fetch.fetch(x_n, y_n, z_n);
            let x2 = fetch.fetch(x, y_n, z_n);
            let x3 = fetch.fetch(x, y_n, z);

            let c1 = x0 - c0;
            let c2 = x1 - x2;
            let c3 = x3 - c0;
            let c4 = c0 - x3 - x0 + x2;

            let s0 = c0.mla(c1, T::from(db));
            let s1 = s0.mla(c2, T::from(dr));
            let s2 = s1.mla(c3, T::from(dg));
            s2.mla(c4, T::from(dg * db))
        } else {
            let x0 = fetch.fetch(x, y, z_n);
            let x1 = fetch.fetch(x_n, y, z);
            let x2 = fetch.fetch(x_n, y, z_n);
            let x3 = fetch.fetch(x_n, y_n, z_n);

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

    #[cfg(feature = "options")]
    #[inline]
    fn tetra<
        T: Copy
            + From<f32>
            + Sub<T, Output = T>
            + Mul<T, Output = T>
            + Add<T, Output = T>
            + FusedMultiplyAdd<T>,
    >(
        &self,
        lin_x: f32,
        lin_y: f32,
        lin_z: f32,
        fetch: impl ArrayFetch<T>,
    ) -> T {
        let scale_x = (self.grid_size[0] as i32 - 1) as f32;
        let scale_y = (self.grid_size[1] as i32 - 1) as f32;
        let scale_z = (self.grid_size[2] as i32 - 1) as f32;

        let x = (lin_x * scale_x).floor() as i32;
        let y = (lin_y * scale_y).floor() as i32;
        let z = (lin_z * scale_z).floor() as i32;

        let x_n = (lin_x * scale_x).ceil() as i32;
        let y_n = (lin_y * scale_y).ceil() as i32;
        let z_n = (lin_z * scale_z).ceil() as i32;

        let rx = lin_x * scale_x - x as f32;
        let ry = lin_y * scale_y - y as f32;
        let rz = lin_z * scale_z - z as f32;

        let c0 = fetch.fetch(x, y, z);
        let c2;
        let c1;
        let c3;
        if rx >= ry {
            if ry >= rz {
                //rx >= ry && ry >= rz
                c1 = fetch.fetch(x_n, y, z) - c0;
                c2 = fetch.fetch(x_n, y_n, z) - fetch.fetch(x_n, y, z);
                c3 = fetch.fetch(x_n, y_n, z_n) - fetch.fetch(x_n, y_n, z);
            } else if rx >= rz {
                //rx >= rz && rz >= ry
                c1 = fetch.fetch(x_n, y, z) - c0;
                c2 = fetch.fetch(x_n, y_n, z_n) - fetch.fetch(x_n, y, z_n);
                c3 = fetch.fetch(x_n, y, z_n) - fetch.fetch(x_n, y, z);
            } else {
                //rz > rx && rx >= ry
                c1 = fetch.fetch(x_n, y, z_n) - fetch.fetch(x, y, z_n);
                c2 = fetch.fetch(x_n, y_n, z_n) - fetch.fetch(x_n, y, z_n);
                c3 = fetch.fetch(x, y, z_n) - c0;
            }
        } else if rx >= rz {
            //ry > rx && rx >= rz
            c1 = fetch.fetch(x_n, y_n, z) - fetch.fetch(x, y_n, z);
            c2 = fetch.fetch(x, y_n, z) - c0;
            c3 = fetch.fetch(x_n, y_n, z_n) - fetch.fetch(x_n, y_n, z);
        } else if ry >= rz {
            //ry >= rz && rz > rx
            c1 = fetch.fetch(x_n, y_n, z_n) - fetch.fetch(x, y_n, z_n);
            c2 = fetch.fetch(x, y_n, z) - c0;
            c3 = fetch.fetch(x, y_n, z_n) - fetch.fetch(x, y_n, z);
        } else {
            //rz > ry && ry > rx
            c1 = fetch.fetch(x_n, y_n, z_n) - fetch.fetch(x, y_n, z_n);
            c2 = fetch.fetch(x, y_n, z_n) - fetch.fetch(x, y, z_n);
            c3 = fetch.fetch(x, y, z_n) - c0;
        }
        let s0 = c0.mla(c1, T::from(rx));
        let s1 = s0.mla(c2, T::from(ry));
        s1.mla(c3, T::from(rz))
    }

    #[cfg(feature = "options")]
    #[inline]
    fn prism<
        T: Copy
            + From<f32>
            + Sub<T, Output = T>
            + Mul<T, Output = T>
            + Add<T, Output = T>
            + FusedMultiplyAdd<T>,
    >(
        &self,
        lin_x: f32,
        lin_y: f32,
        lin_z: f32,
        fetch: impl ArrayFetch<T>,
    ) -> T {
        let scale_x = (self.grid_size[0] as i32 - 1) as f32;
        let scale_y = (self.grid_size[1] as i32 - 1) as f32;
        let scale_z = (self.grid_size[2] as i32 - 1) as f32;

        let x = (lin_x * scale_x).floor() as i32;
        let y = (lin_y * scale_y).floor() as i32;
        let z = (lin_z * scale_z).floor() as i32;

        let x_n = (lin_x * scale_x).ceil() as i32;
        let y_n = (lin_y * scale_y).ceil() as i32;
        let z_n = (lin_z * scale_z).ceil() as i32;

        let dr = lin_x * scale_x - x as f32;
        let dg = lin_y * scale_y - y as f32;
        let db = lin_z * scale_z - z as f32;

        let c0 = fetch.fetch(x, y, z);

        if db >= dr {
            let x0 = fetch.fetch(x, y, z_n);
            let x1 = fetch.fetch(x_n, y, z_n);
            let x2 = fetch.fetch(x, y_n, z);
            let x3 = fetch.fetch(x, y_n, z_n);
            let x4 = fetch.fetch(x_n, y_n, z_n);

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
            let x0 = fetch.fetch(x_n, y, z);
            let x1 = fetch.fetch(x_n, y, z_n);
            let x2 = fetch.fetch(x, y_n, z);
            let x3 = fetch.fetch(x_n, y_n, z);
            let x4 = fetch.fetch(x_n, y_n, z_n);

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

    #[cfg(feature = "options")]
    #[inline]
    pub fn prism_vec3(&self, lin_x: f32, lin_y: f32, lin_z: f32) -> Vector3f {
        self.prism(
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

    #[cfg(feature = "options")]
    #[inline]
    pub fn pyramid_vec3(&self, lin_x: f32, lin_y: f32, lin_z: f32) -> Vector3f {
        self.pyramid(
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

    #[cfg(feature = "options")]
    #[inline]
    pub fn tetra_vec3(&self, lin_x: f32, lin_y: f32, lin_z: f32) -> Vector3f {
        self.tetra(
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

    #[cfg(feature = "options")]
    #[inline]
    pub fn tetra_vec4(&self, lin_x: f32, lin_y: f32, lin_z: f32) -> Vector4f {
        self.tetra(
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

    #[cfg(feature = "options")]
    #[inline]
    pub fn pyramid_vec4(&self, lin_x: f32, lin_y: f32, lin_z: f32) -> Vector4f {
        self.pyramid(
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

    #[cfg(feature = "options")]
    #[inline]
    pub fn prism_vec4(&self, lin_x: f32, lin_y: f32, lin_z: f32) -> Vector4f {
        self.prism(
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
