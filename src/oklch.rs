/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::math::atan2f;
use crate::{Oklab, Rgb, cbrtf, const_hypotf, cosf, hypotf, powf, sinf};
use num_traits::Pow;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Represents *Oklch* colorspace
#[repr(C)]
#[derive(Copy, Clone, PartialOrd, PartialEq)]
pub struct Oklch {
    /// Lightness
    pub l: f32,
    /// Chroma
    pub c: f32,
    /// Hue
    pub h: f32,
}

impl Oklch {
    /// Creates new instance
    #[inline]
    pub const fn new(l: f32, c: f32, h: f32) -> Oklch {
        Oklch { l, c, h }
    }

    /// Converts Linear [Rgb] into [Oklch]
    ///
    /// # Arguments
    /// `transfer_function` - Transfer function into linear colorspace and its inverse
    #[inline]
    pub fn from_linear_rgb(rgb: Rgb<f32>) -> Oklch {
        let oklab = Oklab::from_linear_rgb(rgb);
        Oklch::from_oklab(oklab)
    }

    /// Converts [Oklch] into linear [Rgb]
    #[inline]
    pub fn to_linear_rgb(&self) -> Rgb<f32> {
        let oklab = self.to_oklab();
        oklab.to_linear_rgb()
    }

    /// Converts *Oklab* to *Oklch*
    #[inline]
    pub fn from_oklab(oklab: Oklab) -> Oklch {
        let chroma = hypotf(oklab.b, oklab.a);
        let hue = oklab.b.atan2(oklab.a);
        Oklch::new(oklab.l, chroma, hue)
    }

    /// Converts *Oklab* to *Oklch*
    #[inline]
    pub const fn const_from_oklab(oklab: Oklab) -> Oklch {
        let chroma = const_hypotf(oklab.b, oklab.a);
        let hue = atan2f(oklab.b, oklab.a);
        Oklch::new(oklab.l, chroma, hue)
    }

    /// Converts *Oklch* to *Oklab*
    #[inline]
    pub const fn to_oklab(&self) -> Oklab {
        let l = self.l;
        let a = self.c * cosf(self.h);
        let b = self.c * sinf(self.h);
        Oklab::new(l, a, b)
    }
}

impl Oklch {
    #[inline]
    pub fn euclidean_distance(&self, other: Self) -> f32 {
        let dl = self.l - other.l;
        let dc = self.c - other.c;
        let dh = self.h - other.h;
        (dl * dl + dc * dc + dh * dh).sqrt()
    }
}

impl Oklch {
    #[inline]
    pub fn taxicab_distance(&self, other: Self) -> f32 {
        let dl = self.l - other.l;
        let dc = self.c - other.c;
        let dh = self.h - other.h;
        dl.abs() + dc.abs() + dh.abs()
    }
}

impl Add<Oklch> for Oklch {
    type Output = Oklch;

    #[inline]
    fn add(self, rhs: Self) -> Oklch {
        Oklch::new(self.l + rhs.l, self.c + rhs.c, self.h + rhs.h)
    }
}

impl Add<f32> for Oklch {
    type Output = Oklch;

    #[inline]
    fn add(self, rhs: f32) -> Oklch {
        Oklch::new(self.l + rhs, self.c + rhs, self.h + rhs)
    }
}

impl AddAssign<Oklch> for Oklch {
    #[inline]
    fn add_assign(&mut self, rhs: Oklch) {
        self.l += rhs.l;
        self.c += rhs.c;
        self.h += rhs.h;
    }
}

impl AddAssign<f32> for Oklch {
    #[inline]
    fn add_assign(&mut self, rhs: f32) {
        self.l += rhs;
        self.c += rhs;
        self.h += rhs;
    }
}

impl Mul<f32> for Oklch {
    type Output = Oklch;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Oklch::new(self.l * rhs, self.c * rhs, self.h * rhs)
    }
}

impl Mul<Oklch> for Oklch {
    type Output = Oklch;

    #[inline]
    fn mul(self, rhs: Oklch) -> Self::Output {
        Oklch::new(self.l * rhs.l, self.c * rhs.c, self.h * rhs.h)
    }
}

impl MulAssign<f32> for Oklch {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        self.l *= rhs;
        self.c *= rhs;
        self.h *= rhs;
    }
}

impl MulAssign<Oklch> for Oklch {
    #[inline]
    fn mul_assign(&mut self, rhs: Oklch) {
        self.l *= rhs.l;
        self.c *= rhs.c;
        self.h *= rhs.h;
    }
}

impl Sub<f32> for Oklch {
    type Output = Oklch;

    #[inline]
    fn sub(self, rhs: f32) -> Self::Output {
        Oklch::new(self.l - rhs, self.c - rhs, self.h - rhs)
    }
}

impl Sub<Oklch> for Oklch {
    type Output = Oklch;

    #[inline]
    fn sub(self, rhs: Oklch) -> Self::Output {
        Oklch::new(self.l - rhs.l, self.c - rhs.c, self.h - rhs.h)
    }
}

impl SubAssign<f32> for Oklch {
    #[inline]
    fn sub_assign(&mut self, rhs: f32) {
        self.l -= rhs;
        self.c -= rhs;
        self.h -= rhs;
    }
}

impl SubAssign<Oklch> for Oklch {
    #[inline]
    fn sub_assign(&mut self, rhs: Oklch) {
        self.l -= rhs.l;
        self.c -= rhs.c;
        self.h -= rhs.h;
    }
}

impl Div<f32> for Oklch {
    type Output = Oklch;

    #[inline]
    fn div(self, rhs: f32) -> Self::Output {
        Oklch::new(self.l / rhs, self.c / rhs, self.h / rhs)
    }
}

impl Div<Oklch> for Oklch {
    type Output = Oklch;

    #[inline]
    fn div(self, rhs: Oklch) -> Self::Output {
        Oklch::new(self.l / rhs.l, self.c / rhs.c, self.h / rhs.h)
    }
}

impl DivAssign<f32> for Oklch {
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        self.l /= rhs;
        self.c /= rhs;
        self.h /= rhs;
    }
}

impl DivAssign<Oklch> for Oklch {
    #[inline]
    fn div_assign(&mut self, rhs: Oklch) {
        self.l /= rhs.l;
        self.c /= rhs.c;
        self.h /= rhs.h;
    }
}

impl Neg for Oklch {
    type Output = Oklch;

    #[inline]
    fn neg(self) -> Self::Output {
        Oklch::new(-self.l, -self.c, -self.h)
    }
}

impl Pow<f32> for Oklch {
    type Output = Oklch;

    #[inline]
    fn pow(self, rhs: f32) -> Self::Output {
        Oklch::new(powf(self.l, rhs), powf(self.c, rhs), powf(self.h, rhs))
    }
}

impl Pow<Oklch> for Oklch {
    type Output = Oklch;

    #[inline]
    fn pow(self, rhs: Oklch) -> Self::Output {
        Oklch::new(
            powf(self.l, rhs.l),
            powf(self.c, rhs.c),
            powf(self.h, rhs.h),
        )
    }
}

impl Oklch {
    #[inline]
    pub fn sqrt(&self) -> Oklch {
        Oklch::new(
            if self.l < 0. { 0. } else { self.l.sqrt() },
            if self.c < 0. { 0. } else { self.c.sqrt() },
            if self.h < 0. { 0. } else { self.h.sqrt() },
        )
    }

    #[inline]
    pub const fn cbrt(&self) -> Oklch {
        Oklch::new(cbrtf(self.l), cbrtf(self.c), cbrtf(self.h))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let xyz = Rgb::new(0.1, 0.2, 0.3);
        let lab = Oklch::from_linear_rgb(xyz);
        let rolled_back = lab.to_linear_rgb();
        let dx = (xyz.r - rolled_back.r).abs();
        let dy = (xyz.g - rolled_back.g).abs();
        let dz = (xyz.b - rolled_back.b).abs();
        assert!(dx < 1e-5);
        assert!(dy < 1e-5);
        assert!(dz < 1e-5);
    }
}
