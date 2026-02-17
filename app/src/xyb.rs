/*
 * // Copyright (c) Radzivon Bartoshyk 2/2026. All rights reserved.
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
use crate::encode_gray_lut::{create_gray_to_xyz_samples, create_xyz_to_gray_samples};
use moxcms::{
    ColorProfile, DataColorSpace, Lab, LutMultidimensionalType, LutStore, LutWarehouse, Matrix3d,
    Rgb, ToneReprCurve, Vector3, WHITE_POINT_D65,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialOrd, PartialEq)]
pub struct Xyb {
    pub x: f32,
    pub y: f32,
    pub b: f32,
}

impl Xyb {
    #[inline]
    pub fn new(x: f32, y: f32, b: f32) -> Xyb {
        Xyb { x, y, b }
    }

    #[inline]
    /// Converts linear [Rgb] to [Xyb]
    pub fn from_linear_rgb(rgb: Rgb<f32>) -> Xyb {
        const BIAS_CBRT: f32 = 0.155954200549248620f32;
        const BIAS: f32 = 0.00379307325527544933;
        let lgamma = f32::mul_add(
            0.3f32,
            rgb.r,
            f32::mul_add(0.622f32, rgb.g, f32::mul_add(0.078f32, rgb.b, BIAS)),
        )
            .cbrt()
            - BIAS_CBRT;
        let mgamma = f32::mul_add(
            0.23f32,
            rgb.r,
            f32::mul_add(0.692f32, rgb.g, f32::mul_add(0.078f32, rgb.b, BIAS)),
        )
            .cbrt()
            - BIAS_CBRT;
        let sgamma = f32::mul_add(
            0.24342268924547819f32,
            rgb.r,
            f32::mul_add(
                0.20476744424496821f32,
                rgb.g,
                f32::mul_add(0.55180986650955360f32, rgb.b, BIAS),
            ),
        )
            .cbrt()
            - BIAS_CBRT;
        let x = (lgamma - mgamma) * 0.5f32;
        let y = (lgamma + mgamma) * 0.5f32;
        let b = sgamma - mgamma;
        Xyb::new(x, y, b)
    }

    #[inline]
    /// Converts [Xyb] to linear [Rgb]
    pub fn to_linear_rgb(&self) -> Rgb<f32> {
        const BIAS_CBRT: f32 = 0.155954200549248620f32;
        const BIAS: f32 = 0.00379307325527544933;
        let x_lms = (self.x + self.y) + BIAS_CBRT;
        let y_lms = (-self.x + self.y) + BIAS_CBRT;
        let b_lms = (-self.x + self.y + self.b) + BIAS_CBRT;
        let x_c_lms = (x_lms * x_lms * x_lms) - BIAS;
        let y_c_lms = (y_lms * y_lms * y_lms) - BIAS;
        let b_c_lms = (b_lms * b_lms * b_lms) - BIAS;
        let r = f32::mul_add(
            11.031566901960783,
            x_c_lms,
            f32::mul_add(-9.866943921568629, y_c_lms, -0.16462299647058826 * b_c_lms),
        );
        let g = f32::mul_add(
            -3.254147380392157,
            x_c_lms,
            f32::mul_add(4.418770392156863, y_c_lms, -0.16462299647058826 * b_c_lms),
        );
        let b = f32::mul_add(
            -3.6588512862745097,
            x_c_lms,
            f32::mul_add(2.7129230470588235, y_c_lms, 1.9459282392156863 * b_c_lms),
        );
        Rgb::new(r, g, b)
    }
}

#[inline]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

pub(crate) fn create_xyb_to_lab<const SAMPLES: usize>() -> Vec<u16> {
    let lut_size: u32 = (SAMPLES * SAMPLES * SAMPLES) as u32;

    assert!(SAMPLES >= 1);

    let srgb_to_xyz_d50 = ColorProfile::new_srgb().rgb_to_xyz_matrix().to_f32();

    let mut src = Vec::with_capacity(lut_size as usize);
    for x in 0..SAMPLES as u32 {
        for y in 0..SAMPLES as u32 {
            for b in 0..SAMPLES as u32 {
                let scale_x = lerp(-0.05, 0.05, x as f32 / (SAMPLES - 1) as f32);
                let scale_y = lerp(0., 0.845, y as f32 / (SAMPLES - 1) as f32);
                let scale_z = lerp(-0.45, 0.45, b as f32 / (SAMPLES - 1) as f32);

                let xyb = Xyb::new(scale_x, scale_y, scale_z);
                let xyz = xyb.to_linear_rgb().to_xyz(srgb_to_xyz_d50);
                let mut pcs_lab = Lab::from_pcs_xyz(xyz);
                pcs_lab.l = pcs_lab.l.max(0.0);
                pcs_lab.a = pcs_lab.a.max(0.0).min(1.0);
                pcs_lab.b = pcs_lab.b.max(0.0).min(1.0);
                src.push((pcs_lab.l * 65535.).round() as u16);
                src.push(((pcs_lab.a) * 65535.).round() as u16);
                src.push(((pcs_lab.b) * 65535.).round() as u16);
            }
        }
    }
    src
}

pub(crate) fn create_lab_to_xyb<const SAMPLES: usize>() -> Vec<u16> {
    let lut_size: u32 = (SAMPLES * SAMPLES * SAMPLES) as u32;

    assert!(SAMPLES >= 1);

    let scale = 1. / (SAMPLES as f32 - 1.0);

    let xyz_to_srgb_d50 = ColorProfile::new_srgb()
        .rgb_to_xyz_matrix()
        .inverse()
        .to_f32();

    let mut src = Vec::with_capacity(lut_size as usize);
    for l in 0..SAMPLES as u32 {
        for a in 0..SAMPLES as u32 {
            for b in 0..SAMPLES as u32 {
                let lab = Lab::new(l as f32 * scale * 100.0, a as f32 * scale * 255.0 - 128.0, b as f32 * scale * 255.0 - 128.0);
                let pcs_xyz = lab.to_xyz();
                let lin_rgb = pcs_xyz.to_linear_rgb(xyz_to_srgb_d50);
                let mut xyb = Xyb::from_linear_rgb(lin_rgb);
                xyb.x = ((xyb.x + 0.05) / 0.1).clamp(0.0, 1.0);
                xyb.y = (xyb.y / 0.845).clamp(0.0, 1.0);
                xyb.b = ((xyb.b + 0.45) / (0.45 * 2.)).clamp(0.0, 1.0);
                src.push((xyb.x * 65535.).round() as u16);
                src.push((xyb.y * 65535.).round() as u16);
                src.push((xyb.b * 65535.).round() as u16);
            }
        }
    }
    src
}

pub(crate) fn gen_xyb_icc() -> ColorProfile {
    let lab_to_xyb = create_lab_to_xyb::<33>();
    let xyb_to_lab = create_xyb_to_lab::<33>();

    let mut xyb_icc = ColorProfile::new_lab();
    xyb_icc.color_space = DataColorSpace::Color3;
    xyb_icc.pcs = DataColorSpace::Lab;

    let b_to_a_lut = LutWarehouse::Multidimensional(LutMultidimensionalType {
        num_input_channels: 3,
        num_output_channels: 3,
        grid_points: [33, 33, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        clut: Some(LutStore::Store16(lab_to_xyb.clone())),
        b_curves: vec![
            ToneReprCurve::Lut(vec![]),
            ToneReprCurve::Lut(vec![]),
            ToneReprCurve::Lut(vec![]),
        ],
        matrix: Matrix3d::IDENTITY,
        a_curves: vec![
            ToneReprCurve::Lut(vec![]),
            ToneReprCurve::Lut(vec![]),
            ToneReprCurve::Lut(vec![]),
        ],
        m_curves: vec![
            ToneReprCurve::Lut(vec![]),
            ToneReprCurve::Lut(vec![]),
            ToneReprCurve::Lut(vec![]),
        ],
        bias: Vector3::default(),
    });
    xyb_icc.lut_b_to_a_perceptual = Some(b_to_a_lut.clone());
    xyb_icc.lut_b_to_a_colorimetric = Some(b_to_a_lut);

    let a_to_b = LutWarehouse::Multidimensional(LutMultidimensionalType {
        num_input_channels: 3,
        num_output_channels: 3,
        grid_points: [33, 33, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        clut: Some(LutStore::Store16(xyb_to_lab.clone())),
        b_curves: vec![
            ToneReprCurve::Lut(vec![]),
            ToneReprCurve::Lut(vec![]),
            ToneReprCurve::Lut(vec![]),
        ],
        matrix: Matrix3d::IDENTITY,
        a_curves: vec![
            ToneReprCurve::Lut(vec![]),
            ToneReprCurve::Lut(vec![]),
            ToneReprCurve::Lut(vec![]),
        ],
        m_curves: vec![
            ToneReprCurve::Lut(vec![]),
            ToneReprCurve::Lut(vec![]),
            ToneReprCurve::Lut(vec![]),
        ],
        bias: Vector3::default(),
    });
    xyb_icc.lut_a_to_b_colorimetric = Some(a_to_b.clone());
    xyb_icc.lut_a_to_b_perceptual = Some(a_to_b);

    xyb_icc
}
