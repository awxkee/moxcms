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
use crate::Xyz;
use crate::jzczhz::Jzczhz;
use crate::mlaf::mlaf;
use num_traits::Pow;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

#[inline]
fn perceptual_quantizer(x: f32) -> f32 {
    if x <= 0. {
        return 0.;
    }
    let xx = f32::powf(x * 1e-4, 0.1593017578125);
    let rs = f32::powf(
        (0.8359375 + 18.8515625 * xx) / (1. + 18.6875 * xx),
        134.034375,
    );
    if rs.is_nan() {
        return 0.;
    }
    rs
}

#[inline]
fn perceptual_quantizer_inverse(x: f32) -> f32 {
    if x <= 0. {
        return 0.;
    }
    let xx = f32::powf(x, 7.460772656268214e-03);
    let rs = 1e4
        * f32::powf(
            (0.8359375 - xx) / (18.6875 * xx - 18.8515625),
            6.277394636015326,
        );
    if rs.is_nan() {
        return 0.;
    }
    rs
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Default)]
/// Represents Jzazbz
pub struct Jzazbz {
    /// Jz(lightness) generally expects to be between [0;1]
    pub jz: f32,
    /// Az generally expects to be between [-0.5;0.5]
    pub az: f32,
    /// Bz generally expects to be between [-0.5;0.5]
    pub bz: f32,
}

impl Jzazbz {
    /// Constructs new instance
    #[inline]
    pub fn new(jz: f32, az: f32, bz: f32) -> Jzazbz {
        Jzazbz { jz, az, bz }
    }

    #[inline]
    pub fn from_xyz(xyz: Xyz) -> Jzazbz {
        Self::from_xyz_with_display_luminance(xyz, 200f32)
    }

    #[inline]
    pub fn from_xyz_with_display_luminance(xyz: Xyz, display_luminance: f32) -> Jzazbz {
        let abs_xyz = xyz * display_luminance;
        let lp = perceptual_quantizer(mlaf(
            mlaf(0.674207838 * abs_xyz.x, 0.382799340, abs_xyz.y),
            -0.047570458,
            abs_xyz.z,
        ));
        let mp = perceptual_quantizer(mlaf(
            mlaf(0.149284160 * abs_xyz.x, 0.739628340, abs_xyz.y),
            0.083327300,
            abs_xyz.z,
        ));
        let sp = perceptual_quantizer(mlaf(
            mlaf(0.070941080 * abs_xyz.x, 0.174768000, abs_xyz.y),
            0.670970020,
            abs_xyz.z,
        ));
        let iz = 0.5 * (lp + mp);
        let az = mlaf(mlaf(3.524000 * lp, -4.066708, mp), 0.542708, sp);
        let bz = mlaf(mlaf(0.199076 * lp, 1.096799, mp), -1.295875, sp);
        let jz = (0.44 * iz) / (1. - 0.56 * iz) - 1.6295499532821566e-11;
        Jzazbz::new(jz, az, bz)
    }

    /// Converts [Jzazbz] to [Xyz]
    #[inline]
    pub fn to_xyz(&self, display_luminance: f32) -> Xyz {
        let jz = self.jz + 1.6295499532821566e-11;

        let iz = jz / (0.44f32 + 0.56f32 * jz);
        let l = perceptual_quantizer_inverse(mlaf(
            mlaf(iz, 1.386050432715393e-1, self.az),
            5.804731615611869e-2,
            self.bz,
        ));
        let m = perceptual_quantizer_inverse(mlaf(
            mlaf(iz, -1.386050432715393e-1, self.az),
            -5.804731615611891e-2,
            self.bz,
        ));
        let s = perceptual_quantizer_inverse(mlaf(
            mlaf(iz, -9.601924202631895e-2, self.az),
            -8.118918960560390e-1,
            self.bz,
        ));
        let x = mlaf(
            mlaf(1.661373055774069e+00 * l, -9.145230923250668e-01, m),
            2.313620767186147e-01,
            s,
        );
        let y = mlaf(
            mlaf(-3.250758740427037e-01 * l, 1.571847038366936e+00, m),
            -2.182538318672940e-01,
            s,
        );
        let z = mlaf(
            mlaf(-9.098281098284756e-02 * l, -3.127282905230740e-01, m),
            1.522766561305260e+00,
            s,
        );
        let rel_luminance = 1f32 / display_luminance;
        Xyz::new(x, y, z) * rel_luminance
    }

    /// Converts into *Jzczhz*
    #[inline]
    pub fn to_jzczhz(&self) -> Jzczhz {
        Jzczhz::from_jzazbz(*self)
    }

    #[inline]
    pub fn euclidean_distance(&self, other: Self) -> f32 {
        let djz = self.jz - other.jz;
        let daz = self.az - other.az;
        let dbz = self.bz - other.bz;
        (djz * djz + daz * daz + dbz * dbz).sqrt()
    }

    #[inline]
    pub fn taxicab_distance(&self, other: Self) -> f32 {
        let djz = self.jz - other.jz;
        let daz = self.az - other.az;
        let dbz = self.bz - other.bz;
        djz.abs() + daz.abs() + dbz.abs()
    }
}

impl Index<usize> for Jzazbz {
    type Output = f32;

    #[inline]
    fn index(&self, index: usize) -> &f32 {
        match index {
            0 => &self.jz,
            1 => &self.az,
            2 => &self.bz,
            _ => panic!("Index out of bounds for Jzazbz"),
        }
    }
}

impl IndexMut<usize> for Jzazbz {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        match index {
            0 => &mut self.jz,
            1 => &mut self.az,
            2 => &mut self.bz,
            _ => panic!("Index out of bounds for Jzazbz"),
        }
    }
}

impl Add<f32> for Jzazbz {
    type Output = Jzazbz;

    #[inline]
    fn add(self, rhs: f32) -> Self::Output {
        Jzazbz::new(self.jz + rhs, self.az + rhs, self.bz + rhs)
    }
}

impl Sub<f32> for Jzazbz {
    type Output = Jzazbz;

    #[inline]
    fn sub(self, rhs: f32) -> Self::Output {
        Jzazbz::new(self.jz - rhs, self.az - rhs, self.bz - rhs)
    }
}

impl Mul<f32> for Jzazbz {
    type Output = Jzazbz;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Jzazbz::new(self.jz * rhs, self.az * rhs, self.bz * rhs)
    }
}

impl Div<f32> for Jzazbz {
    type Output = Jzazbz;

    #[inline]
    fn div(self, rhs: f32) -> Self::Output {
        Jzazbz::new(self.jz / rhs, self.az / rhs, self.bz / rhs)
    }
}

impl Add<Jzazbz> for Jzazbz {
    type Output = Jzazbz;

    #[inline]
    fn add(self, rhs: Jzazbz) -> Self::Output {
        Jzazbz::new(self.jz + rhs.jz, self.az + rhs.az, self.bz + rhs.bz)
    }
}

impl Sub<Jzazbz> for Jzazbz {
    type Output = Jzazbz;

    #[inline]
    fn sub(self, rhs: Jzazbz) -> Self::Output {
        Jzazbz::new(self.jz - rhs.jz, self.az - rhs.az, self.bz - rhs.bz)
    }
}

impl Mul<Jzazbz> for Jzazbz {
    type Output = Jzazbz;

    #[inline]
    fn mul(self, rhs: Jzazbz) -> Self::Output {
        Jzazbz::new(self.jz * rhs.jz, self.az * rhs.az, self.bz * rhs.bz)
    }
}

impl Div<Jzazbz> for Jzazbz {
    type Output = Jzazbz;

    #[inline]
    fn div(self, rhs: Jzazbz) -> Self::Output {
        Jzazbz::new(self.jz / rhs.jz, self.az / rhs.az, self.bz / rhs.bz)
    }
}

impl AddAssign<Jzazbz> for Jzazbz {
    #[inline]
    fn add_assign(&mut self, rhs: Jzazbz) {
        self.jz += rhs.jz;
        self.az += rhs.az;
        self.bz += rhs.bz;
    }
}

impl SubAssign<Jzazbz> for Jzazbz {
    #[inline]
    fn sub_assign(&mut self, rhs: Jzazbz) {
        self.jz -= rhs.jz;
        self.az -= rhs.az;
        self.bz -= rhs.bz;
    }
}

impl MulAssign<Jzazbz> for Jzazbz {
    #[inline]
    fn mul_assign(&mut self, rhs: Jzazbz) {
        self.jz *= rhs.jz;
        self.az *= rhs.az;
        self.bz *= rhs.bz;
    }
}

impl DivAssign<Jzazbz> for Jzazbz {
    #[inline]
    fn div_assign(&mut self, rhs: Jzazbz) {
        self.jz /= rhs.jz;
        self.az /= rhs.az;
        self.bz /= rhs.bz;
    }
}

impl AddAssign<f32> for Jzazbz {
    #[inline]
    fn add_assign(&mut self, rhs: f32) {
        self.jz += rhs;
        self.az += rhs;
        self.bz += rhs;
    }
}

impl SubAssign<f32> for Jzazbz {
    #[inline]
    fn sub_assign(&mut self, rhs: f32) {
        self.jz -= rhs;
        self.az -= rhs;
        self.bz -= rhs;
    }
}

impl MulAssign<f32> for Jzazbz {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        self.jz *= rhs;
        self.az *= rhs;
        self.bz *= rhs;
    }
}

impl DivAssign<f32> for Jzazbz {
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        self.jz /= rhs;
        self.az /= rhs;
        self.bz /= rhs;
    }
}

impl Neg for Jzazbz {
    type Output = Jzazbz;

    #[inline]
    fn neg(self) -> Self::Output {
        Jzazbz::new(-self.jz, -self.az, -self.bz)
    }
}

impl Jzazbz {
    #[inline]
    pub fn sqrt(&self) -> Jzazbz {
        Jzazbz::new(self.jz.sqrt(), self.az.sqrt(), self.bz.sqrt())
    }

    #[inline]
    pub fn cbrt(&self) -> Jzazbz {
        Jzazbz::new(self.jz.cbrt(), self.az.cbrt(), self.bz.cbrt())
    }
}

impl Pow<f32> for Jzazbz {
    type Output = Jzazbz;

    #[inline]
    fn pow(self, rhs: f32) -> Self::Output {
        Jzazbz::new(self.jz.powf(rhs), self.az.powf(rhs), self.bz.powf(rhs))
    }
}

impl Pow<Jzazbz> for Jzazbz {
    type Output = Jzazbz;

    #[inline]
    fn pow(self, rhs: Jzazbz) -> Self::Output {
        Jzazbz::new(
            self.jz.powf(rhs.jz),
            self.az.powf(self.az),
            self.bz.powf(self.bz),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jzazbz_round() {
        let xyz = Xyz::new(0.5, 0.4, 0.3);
        let jzazbz = Jzazbz::from_xyz_with_display_luminance(xyz, 253f32);
        let old_xyz = jzazbz.to_xyz(253f32);
        assert!(
            (xyz.x - old_xyz.x).abs() <= 1e-3,
            "{:?} != {:?}",
            xyz,
            old_xyz
        );
        assert!(
            (xyz.y - old_xyz.y).abs() <= 1e-3,
            "{:?} != {:?}",
            xyz,
            old_xyz
        );
        assert!(
            (xyz.z - old_xyz.z).abs() <= 1e-3,
            "{:?} != {:?}",
            xyz,
            old_xyz
        );
    }
}
