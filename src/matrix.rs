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
use crate::err::CmsError;
use crate::mlaf::mlaf;
use crate::profile::s15_fixed16_number_to_float;
use num_traits::AsPrimitive;
use std::ops::{Add, Mul, Sub};

/// Vector math helper
#[derive(Copy, Clone, Debug, Default)]
pub struct Vector3<T> {
    pub v: [T; 3],
}

pub type Vector3f = Vector3<f32>;
pub type Vector3i = Vector3<i32>;
pub type Vector3u = Vector3<u32>;

impl<T> PartialEq<Self> for Vector3<T>
where
    T: AsPrimitive<f32>,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        const TOLERANCE: f32 = 0.0001f32;
        let dx = (self.v[0].as_() - other.v[0].as_()).abs();
        let dy = (self.v[1].as_() - other.v[1].as_()).abs();
        let dz = (self.v[2].as_() - other.v[2].as_()).abs();
        dx < TOLERANCE && dy < TOLERANCE && dz < TOLERANCE
    }
}

impl<T> Vector3<T> {
    #[inline]
    pub fn to_<Z: Copy + 'static>(self) -> Vector3<Z>
    where
        T: AsPrimitive<Z>,
    {
        Vector3 {
            v: [self.v[0].as_(), self.v[1].as_(), self.v[2].as_()],
        }
    }
}

impl<T> Mul<Vector3<T>> for Vector3<T>
where
    T: Mul<Output = T> + Copy,
{
    type Output = Vector3<T>;

    #[inline]
    fn mul(self, rhs: Vector3<T>) -> Self::Output {
        Self {
            v: [
                self.v[0] * rhs.v[0],
                self.v[1] * rhs.v[1],
                self.v[2] * rhs.v[2],
            ],
        }
    }
}

impl<T> Mul<T> for Vector3<T>
where
    T: Mul<Output = T> + Copy,
{
    type Output = Vector3<T>;

    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        Self {
            v: [self.v[0] * rhs, self.v[1] * rhs, self.v[2] * rhs],
        }
    }
}

impl<T> From<T> for Vector3<T>
where
    T: Copy,
{
    fn from(value: T) -> Self {
        Self {
            v: [value, value, value],
        }
    }
}

impl<T> Add<Vector3<T>> for Vector3<T>
where
    T: Add<Output = T> + Copy,
{
    type Output = Vector3<T>;

    #[inline]
    fn add(self, rhs: Vector3<T>) -> Self::Output {
        Self {
            v: [
                self.v[0] + rhs.v[0],
                self.v[1] + rhs.v[1],
                self.v[2] + rhs.v[2],
            ],
        }
    }
}

impl<T> Add<T> for Vector3<T>
where
    T: Add<Output = T> + Copy,
{
    type Output = Vector3<T>;

    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        Self {
            v: [self.v[0] + rhs, self.v[1] + rhs, self.v[2] + rhs],
        }
    }
}

impl<T> Sub<Vector3<T>> for Vector3<T>
where
    T: Sub<Output = T> + Copy,
{
    type Output = Vector3<T>;

    #[inline]
    fn sub(self, rhs: Vector3<T>) -> Self::Output {
        Self {
            v: [
                self.v[0] - rhs.v[0],
                self.v[1] - rhs.v[1],
                self.v[2] - rhs.v[2],
            ],
        }
    }
}

/// Matrix math helper
#[derive(Copy, Clone, Debug, Default)]
pub struct Matrix3f {
    pub v: [[f32; 3]; 3],
}

pub const SRGB_MATRIX: Matrix3f = Matrix3f {
    v: [
        [
            s15_fixed16_number_to_float(0x6FA2),
            s15_fixed16_number_to_float(0x6299),
            s15_fixed16_number_to_float(0x24A0),
        ],
        [
            s15_fixed16_number_to_float(0x38F5),
            s15_fixed16_number_to_float(0xB785),
            s15_fixed16_number_to_float(0x0F84),
        ],
        [
            s15_fixed16_number_to_float(0x0390),
            s15_fixed16_number_to_float(0x18DA),
            s15_fixed16_number_to_float(0xB6CF),
        ],
    ],
};

pub const DISPLAY_P3_MATRIX: Matrix3f = Matrix3f {
    v: [
        [0.515102f32, 0.291965f32, 0.157153f32],
        [0.241182f32, 0.692236f32, 0.0665819f32],
        [-0.00104941f32, 0.0418818f32, 0.784378f32],
    ],
};

pub const BT2020_MATRIX: Matrix3f = Matrix3f {
    v: [
        [0.673459f32, 0.165661f32, 0.125100f32],
        [0.279033f32, 0.675338f32, 0.0456288f32],
        [-0.00193139f32, 0.0299794f32, 0.797162f32],
    ],
};

impl Matrix3f {
    pub const IDENTITY: Matrix3f = Matrix3f {
        v: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };

    #[inline]
    pub const fn test_equality(&self, other: Matrix3f) -> bool {
        const TOLERANCE: f32 = 0.001f32;
        let diff_r_x = (self.v[0][0] - other.v[0][0]).abs();
        let diff_r_y = (self.v[0][1] - other.v[0][1]).abs();
        let diff_r_z = (self.v[0][2] - other.v[0][2]).abs();

        if diff_r_x > TOLERANCE || diff_r_y > TOLERANCE || diff_r_z > TOLERANCE {
            return false;
        }

        let diff_g_x = (self.v[1][0] - other.v[1][0]).abs();
        let diff_g_y = (self.v[1][1] - other.v[1][1]).abs();
        let diff_g_z = (self.v[1][2] - other.v[1][2]).abs();

        if diff_g_x > TOLERANCE || diff_g_y > TOLERANCE || diff_g_z > TOLERANCE {
            return false;
        }

        let diff_b_x = (self.v[2][0] - other.v[2][0]).abs();
        let diff_b_y = (self.v[2][1] - other.v[2][1]).abs();
        let diff_b_z = (self.v[2][2] - other.v[2][2]).abs();

        if diff_b_x > TOLERANCE || diff_b_y > TOLERANCE || diff_b_z > TOLERANCE {
            return false;
        }

        true
    }

    #[inline]
    pub const fn determinant(&self) -> Option<f32> {
        let v = self.v;
        let a0 = v[0][0] * v[1][1] * v[2][2];
        let a1 = v[0][1] * v[1][2] * v[2][0];
        let a2 = v[0][2] * v[1][0] * v[2][1];

        let s0 = v[0][2] * v[1][1] * v[2][0];
        let s1 = v[0][1] * v[1][0] * v[2][2];
        let s2 = v[0][0] * v[1][2] * v[2][1];

        let j = a0 + a1 + a2 - s0 - s1 - s2;
        if j == 0. {
            return None;
        }
        Some(j)
    }

    #[inline]
    pub fn inverse(&self) -> Option<Self> {
        let v = self.v;
        let det = 1. / self.determinant()?;
        let a = v[0][0];
        let b = v[0][1];
        let c = v[0][2];
        let d = v[1][0];
        let e = v[1][1];
        let f = v[1][2];
        let g = v[2][0];
        let h = v[2][1];
        let i = v[2][2];

        Some(Matrix3f {
            v: [
                [
                    (e * i - f * h) * det,
                    (c * h - b * i) * det,
                    (b * f - c * e) * det,
                ],
                [
                    (f * g - d * i) * det,
                    (a * i - c * g) * det,
                    (c * d - a * f) * det,
                ],
                [
                    (d * h - e * g) * det,
                    (b * g - a * h) * det,
                    (a * e - b * d) * det,
                ],
            ],
        })
    }

    #[inline]
    pub fn mul_row<const R: usize>(&self, rhs: f32) -> Self {
        if R == 0 {
            Self {
                v: [(Vector3f { v: self.v[0] } * rhs).v, self.v[1], self.v[2]],
            }
        } else if R == 1 {
            Self {
                v: [self.v[0], (Vector3f { v: self.v[1] } * rhs).v, self.v[2]],
            }
        } else if R == 2 {
            Self {
                v: [self.v[0], self.v[1], (Vector3f { v: self.v[2] } * rhs).v],
            }
        } else {
            unimplemented!()
        }
    }

    #[inline]
    pub fn mul_row_vector<const R: usize>(&self, rhs: Vector3f) -> Self {
        if R == 0 {
            Self {
                v: [(Vector3f { v: self.v[0] } * rhs).v, self.v[1], self.v[2]],
            }
        } else if R == 1 {
            Self {
                v: [self.v[0], (Vector3f { v: self.v[1] } * rhs).v, self.v[2]],
            }
        } else if R == 2 {
            Self {
                v: [self.v[0], self.v[1], (Vector3f { v: self.v[2] } * rhs).v],
            }
        } else {
            unimplemented!()
        }
    }

    #[inline]
    pub fn mul_vector(&self, other: Vector3f) -> Vector3f {
        let x = mlaf(
            mlaf(self.v[0][1] * other.v[1], self.v[0][2], other.v[2]),
            self.v[0][0],
            other.v[0],
        );
        let y = mlaf(
            mlaf(self.v[1][0] * other.v[0], self.v[1][1], other.v[1]),
            self.v[1][2],
            other.v[2],
        );
        let z = mlaf(
            mlaf(self.v[2][0] * other.v[0], self.v[2][1], other.v[1]),
            self.v[2][2],
            other.v[2],
        );
        Vector3f { v: [x, y, z] }
    }

    #[inline]
    pub fn mat_mul(&self, other: Matrix3f) -> Self {
        let mut result = Matrix3f::default();

        for i in 0..3 {
            for j in 0..3 {
                result.v[i][j] = mlaf(
                    mlaf(self.v[i][0] * other.v[0][j], self.v[i][1], other.v[1][j]),
                    self.v[i][2],
                    other.v[2][j],
                );
            }
        }

        result
    }
}

/// Holds CIE XYZ representation
#[derive(Clone, Debug, Copy, Default)]
pub struct Xyz {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl PartialEq<Self> for Xyz {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        const TOLERANCE: f32 = 0.0001f32;
        let dx = (self.x - other.x).abs();
        let dy = (self.y - other.y).abs();
        let dz = (self.z - other.z).abs();
        dx < TOLERANCE && dy < TOLERANCE && dz < TOLERANCE
    }
}

impl Xyz {
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub const fn to_vector(self) -> Vector3f {
        Vector3f {
            v: [self.x, self.y, self.z],
        }
    }
}

impl Mul<f32> for Xyz {
    type Output = Xyz;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl Mul<Xyz> for Xyz {
    type Output = Xyz;

    #[inline]
    fn mul(self, rhs: Xyz) -> Self::Output {
        Self {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
        }
    }
}

/// Holds CIE XyY representation
#[derive(Clone, Debug, Copy, Default)]
pub struct XyY {
    pub x: f32,
    pub y: f32,
    pub yb: f32,
}

impl XyY {
    #[inline]
    pub const fn to_xyz(self) -> Xyz {
        Xyz {
            x: self.x / self.y * self.yb,
            y: self.yb,
            z: (1. - self.x - self.y) / self.y * self.yb,
        }
    }
}

#[derive(Clone, Debug, Copy)]
pub struct Chromacity {
    pub x: f32,
    pub y: f32,
}

impl Chromacity {
    #[inline]
    pub const fn to_xyz(&self) -> Xyz {
        Xyz {
            x: self.x / self.y,
            y: 1f32,
            z: (1f32 - self.x - self.y) / self.y,
        }
    }

    #[inline]
    pub const fn to_xyyb(&self) -> XyY {
        XyY {
            x: self.x,
            y: self.y,
            yb: 1f32,
        }
    }

    pub const D65: Chromacity = Chromacity {
        x: 0.31272,
        y: 0.32903,
    };

    pub const D50: Chromacity = Chromacity {
        x: 0.34567,
        y: 0.35850,
    };
}

impl TryFrom<Xyz> for Chromacity {
    type Error = CmsError;

    #[inline]
    fn try_from(xyz: Xyz) -> Result<Self, Self::Error> {
        let sum = xyz.x + xyz.y + xyz.z;

        // Avoid division by zero or invalid XYZ values
        if sum == 0.0 {
            return Err(CmsError::DivisionByZero);
        }
        let rec = 1f32 / (xyz.x + xyz.y + xyz.z);

        let chromacity_x = xyz.x * rec;
        let chromacity_y = xyz.y * rec;

        Ok(Chromacity {
            x: chromacity_x,
            y: chromacity_y,
        })
    }
}
