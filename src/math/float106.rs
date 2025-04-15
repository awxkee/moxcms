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
use crate::math::common::f_fmla;
use std::ops::{Add, Div, Mul, Neg, Sub};

trait Upper {
    fn upper(self) -> f64;
}

impl Upper for f64 {
    #[inline(always)]
    fn upper(self) -> f64 {
        f64::from_bits(self.to_bits() & 0x_ffff_ffff_f800_0000)
    }
}

#[inline(always)]
const fn upper(v: f64) -> f64 {
    f64::from_bits(v.to_bits() & 0x_ffff_ffff_f800_0000)
}

#[inline(always)]
const fn cn_fmla(a: f64, b: f64, c: f64) -> f64 {
    c + a * b
}

#[derive(Copy, Clone, Debug, Default)]
pub struct Float106 {
    pub v0: f64,
    pub v1: f64,
}

impl Float106 {
    #[inline(always)]
    pub const fn from_f64(v: f64) -> Float106 {
        Float106 { v0: v, v1: 0. }
    }

    #[inline(always)]
    pub const fn to_f64(self) -> f64 {
        self.v0 + self.v1
    }

    #[inline(always)]
    pub const fn new(v0: f64, v1: f64) -> Self {
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
        let xh = self.v0.upper();
        let xl = self.v0 - xh;
        let r0 = self.v0 * self.v0;

        let w0 = f_fmla(xh, xh, -r0);
        let w1 = f_fmla(xh + xh, xl, w0);
        let w2 = f_fmla(xl, xl, w1);
        let w3 = f_fmla(self.v0, self.v1 + self.v1, w2);

        Self::new(r0, w3)
    }

    #[inline(always)]
    pub const fn c_sqr(self) -> Self {
        let xh = upper(self.v0);
        let xl = self.v0 - xh;
        let r0 = self.v0 * self.v0;

        let w0 = cn_fmla(xh, xh, -r0);
        let w1 = cn_fmla(xh + xh, xl, w0);
        let w2 = cn_fmla(xl, xl, w1);
        let w3 = cn_fmla(self.v0, self.v1 + self.v1, w2);

        Self::new(r0, w3)
    }

    #[inline(always)]
    pub fn mul_as_f64(self, other: Self) -> f64 {
        let xh = self.v0.upper();
        let xl = self.v0 - xh;
        let yh = other.v0.upper();
        let yl = other.v0 - yh;
        let z0 = f_fmla(self.v1, yh, xh * other.v1);
        let z1 = f_fmla(xl, yl, z0);
        let z2 = f_fmla(xh, yl, z1);
        let z3 = f_fmla(xl, yh, z2);
        f_fmla(xh, yh, z3)
    }

    #[inline(always)]
    pub fn recip(self) -> Float106 {
        let t = 1. / self.v0;
        let dh = self.v0.upper();
        let dl = self.v0 - dh;
        let th = t.upper();
        let tl = t - th;
        let q0 = t;
        let z0 = f_fmla(-dh, th, 1.);
        let z1 = f_fmla(-dh, tl, z0);
        let z2 = f_fmla(-dl, th, z1);
        let z3 = f_fmla(-dl, tl, z2);
        let z4 = f_fmla(-self.v1, t, z3);
        Self::new(q0, t * z4)
    }

    #[inline(always)]
    pub const fn c_add_f64(self, rhs: f64) -> Self {
        let rx = self.v0 + rhs;
        let v = rx - self.v0;
        let mut ry = (self.v0 - (rx - v)) + (rhs - v);
        ry += self.v1;
        Float106 { v0: rx, v1: ry }
    }

    #[inline(always)]
    pub const fn c_sub_f64(self, rhs: f64) -> Self {
        self.c_add_f64(-rhs)
    }

    #[inline(always)]
    pub const fn c_div(self, rhs: Float106) -> Self {
        let t = 1. / rhs.v0;
        let dh = upper(rhs.v0);
        let dl = rhs.v0 - dh;
        let th = upper(t);
        let tl = t - th;
        let nhh = upper(self.v0);
        let nhl = self.v0 - nhh;

        let q0 = self.v0 * t;

        let w0 = cn_fmla(nhh, th, -q0);
        let w1 = cn_fmla(nhh, tl, w0);
        let w2 = cn_fmla(nhl, th, w1);
        let w3 = cn_fmla(nhl, tl, w2);

        let z0 = cn_fmla(-dh, th, 1.);
        let z1 = cn_fmla(-dh, tl, z0);
        let z2 = cn_fmla(-dl, th, z1);
        let z3 = cn_fmla(-dl, tl, z2);

        let u = cn_fmla(q0, z3, w3);

        let b0 = cn_fmla(-q0, rhs.v1, self.v1);
        let b1 = cn_fmla(t, b0, u);

        Self::new(q0, b1)
    }

    #[inline(always)]
    pub const fn c_mul(self, other: Float106) -> Self {
        let xh = upper(self.v0);
        let xl = self.v0 - xh;
        let yh = upper(other.v0);
        let yl = other.v0 - yh;
        let r0 = self.v0 * other.v0;

        let w0 = cn_fmla(xh, yh, -r0);
        let w1 = cn_fmla(xl, yh, w0);
        let w2 = cn_fmla(xh, yl, w1);
        let w3 = cn_fmla(xl, yl, w2);
        let w4 = cn_fmla(self.v0, other.v1, w3);
        let w5 = cn_fmla(self.v1, other.v0, w4);

        Self::new(r0, w5)
    }

    #[inline(always)]
    pub const fn c_mul_f64(self, rhs: f64) -> Self {
        let xh = upper(self.v0);
        let xl = self.v0 - xh;
        let yh = upper(rhs);
        let yl = rhs - yh;
        let r0 = self.v0 * rhs;

        let w0 = cn_fmla(xh, yh, -r0);
        let w1 = cn_fmla(xl, yh, w0);
        let w2 = cn_fmla(xh, yl, w1);
        let w3 = cn_fmla(xl, yl, w2);
        let w4 = cn_fmla(self.v1, rhs, w3);

        Self::new(r0, w4)
    }

    #[inline(always)]
    pub fn fast_mul_f64(self, rhs: f64) -> Self {
        let mut product = Self::from_mul_product(self.v0, rhs);
        product.v1 = f_fmla(rhs, self.v1, product.v1);
        product
    }
}

impl Mul<Float106> for Float106 {
    type Output = Float106;

    #[inline(always)]
    fn mul(self, other: Float106) -> Self::Output {
        let xh = self.v0.upper();
        let xl = self.v0 - xh;
        let yh = other.v0.upper();
        let yl = other.v0 - yh;
        let r0 = self.v0 * other.v0;

        let w0 = f_fmla(xh, yh, -r0);
        let w1 = f_fmla(xl, yh, w0);
        let w2 = f_fmla(xh, yl, w1);
        let w3 = f_fmla(xl, yl, w2);
        let w4 = f_fmla(self.v0, other.v1, w3);
        let w5 = f_fmla(self.v1, other.v0, w4);

        Self::new(r0, w5)
    }
}

impl Mul<f64> for Float106 {
    type Output = Float106;

    #[inline(always)]
    fn mul(self, rhs: f64) -> Self::Output {
        let xh = self.v0.upper();
        let xl = self.v0 - xh;
        let yh = rhs.upper();
        let yl = rhs - yh;
        let r0 = self.v0 * rhs;

        let w0 = f_fmla(xh, yh, -r0);
        let w1 = f_fmla(xl, yh, w0);
        let w2 = f_fmla(xh, yl, w1);
        let w3 = f_fmla(xl, yl, w2);
        let w4 = f_fmla(self.v1, rhs, w3);

        Self::new(r0, w4)
    }
}

impl Mul<Float106> for f64 {
    type Output = Float106;

    #[inline(always)]
    fn mul(self, rhs: Float106) -> Self::Output {
        rhs * self
    }
}

impl Div<Float106> for Float106 {
    type Output = Float106;

    #[inline(always)]
    fn div(self, rhs: Float106) -> Self::Output {
        let t = 1. / rhs.v0;
        let dh = rhs.v0.upper();
        let dl = rhs.v0 - dh;
        let th = t.upper();
        let tl = t - th;
        let nhh = self.v0.upper();
        let nhl = self.v0 - nhh;

        let q0 = self.v0 * t;

        let w0 = f_fmla(nhh, th, -q0);
        let w1 = f_fmla(nhh, tl, w0);
        let w2 = f_fmla(nhl, th, w1);
        let w3 = f_fmla(nhl, tl, w2);

        let z0 = f_fmla(-dh, th, 1.);
        let z1 = f_fmla(-dh, tl, z0);
        let z2 = f_fmla(-dl, th, z1);
        let z3 = f_fmla(-dl, tl, z2);

        let u = f_fmla(q0, z3, w3);

        let b0 = f_fmla(-q0, rhs.v1, self.v1);
        let b1 = f_fmla(t, b0, u);

        Self::new(q0, b1)
    }
}

impl Add<f64> for Float106 {
    type Output = Float106;

    #[inline(always)]
    fn add(self, rhs: f64) -> Self::Output {
        let rx = self.v0 + rhs;
        let v = rx - self.v0;
        let mut ry = (self.v0 - (rx - v)) + (rhs - v);
        ry += self.v1;
        Float106 { v0: rx, v1: ry }
    }
}

impl Add<Float106> for f64 {
    type Output = Float106;

    #[inline(always)]
    fn add(self, rhs: Float106) -> Self::Output {
        let rx = self + rhs.v0;
        let v = rhs.v0 - self;
        let ry = (self - (rx - v)) + (rhs.v0 - v) + rhs.v1;

        Float106 { v0: rx, v1: ry }
    }
}

impl Add<Float106> for Float106 {
    type Output = Float106;

    #[inline(always)]
    fn add(self, rhs: Float106) -> Self::Output {
        let rx = self.v0 + rhs.v0;
        let v = rx - self.v0;
        let mut ry = (self.v0 - (rx - v)) + (rhs.v0 - v);
        ry += self.v1 + rhs.v1;
        Self { v0: rx, v1: ry }
    }
}

impl Float106 {
    #[inline(always)]
    pub fn normalize(self) -> Float106 {
        let sx = self.v0 + self.v1;
        let sy = self.v0 - sx + self.v1;
        Float106 { v0: sx, v1: sy }
    }
}

impl Float106 {
    #[inline(always)]
    pub fn from_exact_add(a: f64, b: f64) -> Self {
        let rhi = a + b;
        let t1 = rhi - a;
        let t2 = rhi - t1;
        let t3 = b - t1;
        let t4 = a - t2;
        let rlo = t3 + t4;
        Self { v0: rhi, v1: rlo }
    }

    #[inline(always)]
    pub fn from_mul_product(v0: f64, v1: f64) -> Self {
        let xh = v0.upper();
        let xl = v0 - xh;
        let yh = v1.upper();
        let yl = v1 - yh;
        let r0 = v0 * v1;

        let z0 = f_fmla(xh, yh, -r0);
        let z1 = f_fmla(xl, yh, z0);
        let z2 = f_fmla(xh, yl, z1);
        let z3 = f_fmla(xl, yl, z2);

        Self { v0: r0, v1: z3 }
    }

    #[inline(always)]
    pub const fn c_from_mul_product(v0: f64, v1: f64) -> Self {
        let xh = upper(v0);
        let xl = v0 - xh;
        let yh = upper(v1);
        let yl = v1 - yh;
        let r0 = v0 * v1;

        let z0 = cn_fmla(xh, yh, -r0);
        let z1 = cn_fmla(xl, yh, z0);
        let z2 = cn_fmla(xh, yl, z1);
        let z3 = cn_fmla(xl, yl, z2);

        Self { v0: r0, v1: z3 }
    }
}

impl Neg for Float106 {
    type Output = Float106;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self {
            v0: -self.v0,
            v1: -self.v1,
        }
    }
}

impl Sub<f64> for Float106 {
    type Output = Float106;

    #[inline(always)]
    fn sub(self, rhs: f64) -> Self::Output {
        self + (-rhs)
    }
}
