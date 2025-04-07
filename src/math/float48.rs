/*
 * // Copyright (c) Radzivon Bartoshyk 4/2025. All rights reserved.
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

use crate::math::common::f_fmlaf;
use std::ops::{Add, Div, Mul, Neg, Sub};

#[inline(always)]
const fn upper(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0x_ffff_f000)
}

#[inline(always)]
const fn cn_fmlaf(a: f32, b: f32, c: f32) -> f32 {
    c + a * b
}

#[derive(Copy, Clone, Debug, Default)]
pub struct Float48 {
    pub v0: f32,
    pub v1: f32,
}

impl Float48 {
    #[inline(always)]
    pub const fn from_f32(v: f32) -> Float48 {
        Float48 { v0: v, v1: 0. }
    }

    #[inline(always)]
    pub const fn to_f32(self) -> f32 {
        self.v0 + self.v1
    }

    #[inline(always)]
    pub const fn new(v0: f32, v1: f32) -> Self {
        Self { v0, v1 }
    }

    #[inline(always)]
    pub const fn abs(self) -> Self {
        if self.v0 < 0. {
            Self::new(-self.v0, -self.v1)
        } else {
            self
        }
    }

    #[inline(always)]
    pub fn sqr(self) -> Self {
        let xh = upper(self.v0);
        let xl = self.v0 - xh;
        let r0 = self.v0 * self.v0;

        let w0 = f_fmlaf(xh, xh, -r0);
        let w1 = f_fmlaf(xh + xh, xl, w0);
        let w2 = f_fmlaf(xl, xl, w1);
        let w3 = f_fmlaf(self.v0, self.v1 + self.v1, w2);

        Self::new(r0, w3)
    }

    #[inline(always)]
    pub fn mul_as_f32(self, other: Self) -> f32 {
        let xh = upper(self.v0);
        let xl = self.v0 - xh;
        let yh = upper(other.v0);
        let yl = other.v0 - yh;
        self.v1 * yh + xh * other.v1 + xl * yl + xh * yl + xl * yh + xh * yh
    }

    #[inline(always)]
    pub const fn c_recip(self) -> Float48 {
        let t = 1. / self.v0;
        let dh = upper(self.v0);
        let dl = self.v0 - dh;
        let th = upper(t);
        let tl = t - th;
        let q0 = t;
        Self::new(
            q0,
            t * (1. - dh * th - dh * tl - dl * th - dl * tl - self.v1 * t),
        )
    }
    
    #[inline]
    pub const fn c_add(self, rhs: Self) -> Self {
        let rx = self.v0 + rhs.v0;
        let v = rx - self.v0;
        let mut ry = (self.v0 - (rx - v)) + (rhs.v0 - v);
        ry += self.v1 + rhs.v1;
        Self { v0: rx, v1: ry }
    }

    #[inline]
    pub const fn c_mul(self, rhs: Self) -> Self {
        let xh = upper(self.v0);
        let xl = self.v0 - xh;
        let yh = upper(rhs.v0);
        let yl = rhs.v0 - yh;
        let r0 = self.v0 * rhs.v0;

        let w0 = cn_fmlaf(xh, yh, -r0);
        let w1 = cn_fmlaf(xl, yh, w0);
        let w2 = cn_fmlaf(xh, yl, w1);
        let w3 = cn_fmlaf(xl, yl, w2);
        let w4 = cn_fmlaf(self.v0, rhs.v1, w3);
        let w5 = cn_fmlaf(self.v1, rhs.v0, w4);

        Self::new(r0, w5)
    }
}

impl Mul<Float48> for Float48 {
    type Output = Float48;

    #[inline(always)]
    fn mul(self, other: Float48) -> Self::Output {
        let xh = upper(self.v0);
        let xl = self.v0 - xh;
        let yh = upper(other.v0);
        let yl = other.v0 - yh;
        let r0 = self.v0 * other.v0;

        let w0 = f_fmlaf(xh, yh, -r0);
        let w1 = f_fmlaf(xl, yh, w0);
        let w2 = f_fmlaf(xh, yl, w1);
        let w3 = f_fmlaf(xl, yl, w2);
        let w4 = f_fmlaf(self.v0, other.v1, w3);
        let w5 = f_fmlaf(self.v1, other.v0, w4);

        Self::new(r0, w5)
    }
}

impl Mul<f32> for Float48 {
    type Output = Float48;

    #[inline(always)]
    fn mul(self, rhs: f32) -> Self::Output {
        let xh = upper(self.v0);
        let xl = self.v0 - xh;
        let yh = upper(rhs);
        let yl = rhs - yh;
        let r0 = self.v0 * rhs;

        let w0 = f_fmlaf(xh, yh, -r0);
        let w1 = f_fmlaf(xl, yh, w0);
        let w2 = f_fmlaf(xh, yl, w1);
        let w3 = f_fmlaf(xl, yl, w2);
        let w4 = f_fmlaf(self.v1, rhs, w3);

        Self::new(r0, w4)
    }
}

impl Mul<Float48> for f32 {
    type Output = Float48;

    #[inline(always)]
    fn mul(self, rhs: Float48) -> Self::Output {
        rhs * self
    }
}

impl Div<Float48> for Float48 {
    type Output = Float48;

    #[inline(always)]
    fn div(self, rhs: Float48) -> Self::Output {
        let t = 1. / rhs.v0;
        let dh = upper(rhs.v0);
        let dl = rhs.v0 - dh;
        let th = upper(t);
        let tl = t - th;
        let nhh = upper(self.v0);
        let nhl = self.v0 - nhh;

        let q0 = self.v0 * t;

        let w0 = f_fmlaf(nhh, th, -q0);
        let w1 = f_fmlaf(nhh, tl, w0);
        let w2 = f_fmlaf(nhl, th, w1);
        let w3 = f_fmlaf(nhl, tl, w2);

        let z0 = f_fmlaf(-dh, th, 1.);
        let z1 = f_fmlaf(-dh, tl, z0);
        let z2 = f_fmlaf(-dl, th, z1);
        let z3 = f_fmlaf(-dl, tl, z2);

        let u = f_fmlaf(q0, z3, w3);

        let b0 = f_fmlaf(-q0, rhs.v1, self.v1);
        let b1 = f_fmlaf(t, b0, u);

        Self::new(q0, b1)
    }
}

impl Add<f32> for Float48 {
    type Output = Float48;

    #[inline(always)]
    fn add(self, rhs: f32) -> Self::Output {
        let rx = self.v0 + rhs;
        let v = rx - self.v0;
        let mut ry = (self.v0 - (rx - v)) + (rhs - v);
        ry += self.v1;
        Float48 { v0: rx, v1: ry }
    }
}

impl Add<Float48> for f32 {
    type Output = Float48;

    #[inline(always)]
    fn add(self, rhs: Float48) -> Self::Output {
        let rx  = self + rhs.v0;
        let v = rhs.v0 - self;
        let ry = (self - (rx - v)) + (rhs.v0 - v) + rhs.v1;
        Float48 { v0: rx, v1: ry }
    }
}

impl Add<Float48> for Float48 {
    type Output = Float48;

    #[inline(always)]
    fn add(self, rhs: Float48) -> Self::Output {
        let rx = self.v0 + rhs.v0;
        let v = rx - self.v0;
        let mut ry = (self.v0 - (rx - v)) + (rhs.v0 - v);
        ry += self.v1 + rhs.v1;
        Self { v0: rx, v1: ry }
    }
}

impl Float48 {
    #[inline(always)]
    pub fn normalize(self) -> Float48 {
        let sx = self.v0 + self.v1;
        let sy = self.v0 - sx + self.v1;
        Float48 { v0: sx, v1: sy }
    }
}

impl Float48 {
    #[inline(always)]
    pub const fn c_from_mul_product(v0: f32, v1: f32) -> Self {
        let xh = upper(v0);
        let xl = v0 - xh;
        let yh = upper(v1);
        let yl = v1 - yh;
        let r0 = v0 * v1;
        Self {
            v0: r0,
            v1: xh * yh - r0 + xl * yh + xh * yl + xl * yl,
        }
    }
}

impl Neg for Float48 {
    type Output = Float48;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self {
            v0: -self.v0,
            v1: -self.v1,
        }
    }
}

impl Sub<f32> for Float48 {
    type Output = Float48;

    #[inline(always)]
    fn sub(self, rhs: f32) -> Self::Output {
        self + (-rhs)
    }
}