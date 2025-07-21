/*
 * // Copyright (c) Radzivon Bartoshyk 7/2025. All rights reserved.
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
use moxcms::{
    ColorProfile, DataColorSpace, LutMultidimensionalType, LutStore, LutWarehouse, Matrix3d,
    ToneReprCurve, Vector3, WHITE_POINT_D50, WHITE_POINT_D65, adaption_matrix_d,
};
use std::ops::Mul;

pub(crate) fn create_xyz_to_gray_samples<const SAMPLES: usize>() -> Vec<u16> {
    let lut_size: u32 = (SAMPLES * SAMPLES * SAMPLES) as u32;

    assert!(SAMPLES >= 1);

    let mut src = Vec::with_capacity(lut_size as usize);
    for x in 0..SAMPLES as u32 {
        for y in 0..SAMPLES as u32 {
            for z in 0..SAMPLES as u32 {
                src.push(((y as f32 / (SAMPLES - 1) as f32) * 65535.).round() as u16);
            }
        }
    }
    src
}

pub(crate) fn create_gray_to_xyz_samples<const SAMPLES: usize>() -> Vec<u16> {
    let lut_size: u32 = (3 * SAMPLES) as u32;

    assert!(SAMPLES >= 1);

    let mut src = Vec::with_capacity(lut_size as usize);
    for y in 0..SAMPLES as u32 {
        src.push(((y as f32 / (SAMPLES - 1) as f32) * 65535.).round() as u16);
        src.push(((y as f32 / (SAMPLES - 1) as f32) * 65535.).round() as u16);
        src.push(((y as f32 / (SAMPLES - 1) as f32) * 65535.).round() as u16);
    }
    src
}

pub(crate) fn encode_gray_lut() -> ColorProfile {
    // PCS XYZ is scaled by 32768.0/65535.0 so scaling back is necessary if connection happens on PCS XYZ instead of PCS LAB
    let scale_xyz_matrix = Matrix3d {
        v: [
            [65535.0 / 32768.0, 0.0, 0.0],
            [0.0, 65535.0 / 32768.0, 0.0],
            [0.0, 0.0, 65535.0 / 32768.0],
        ],
    };
    let scale_rgb_to_xyz_matrix = Matrix3d {
        v: [
            [32768.0 / 65535.0, 0.0, 0.0],
            [0.0, 32768.0 / 65535.0, 0.0],
            [0.0, 0.0, 32768.0 / 65535.0],
        ],
    };

    let d50_to_d65 = adaption_matrix_d(WHITE_POINT_D50.to_xyz(), WHITE_POINT_D65.to_xyz());
    let d65_to_d50 = adaption_matrix_d(WHITE_POINT_D65.to_xyz(), WHITE_POINT_D50.to_xyz());

    let srgb_profile = ColorProfile::new_srgb();
    let inverted_srgb = srgb_profile.red_trc.as_ref().unwrap().inverse().unwrap();

    let xyz_to_gray_samples = create_xyz_to_gray_samples::<17>();

    let gray_to_xyz = create_gray_to_xyz_samples::<17>();

    let mut gray_profile = ColorProfile::new_gray_with_gamma(1.0);
    gray_profile.white_point = WHITE_POINT_D65.to_xyzd();
    gray_profile.color_space = DataColorSpace::Gray;
    gray_profile.gray_trc = srgb_profile.red_trc.clone();

    let b_to_a_lut = LutWarehouse::Multidimensional(LutMultidimensionalType {
        num_input_channels: 3,
        num_output_channels: 1,
        grid_points: [17, 17, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        clut: Some(LutStore::Store16(xyz_to_gray_samples.clone())),
        b_curves: vec![
            ToneReprCurve::Lut(vec![]),
            ToneReprCurve::Lut(vec![]),
            ToneReprCurve::Lut(vec![]),
        ],
        matrix: scale_xyz_matrix.mul(d50_to_d65),
        a_curves: vec![inverted_srgb],
        m_curves: vec![
            ToneReprCurve::Lut(vec![]),
            ToneReprCurve::Lut(vec![]),
            ToneReprCurve::Lut(vec![]),
        ],
        bias: Vector3::default(),
    });
    gray_profile.lut_b_to_a_perceptual = Some(b_to_a_lut.clone());
    gray_profile.lut_b_to_a_colorimetric = Some(b_to_a_lut);

    let a_to_b = LutWarehouse::Multidimensional(LutMultidimensionalType {
        num_input_channels: 1,
        num_output_channels: 3,
        grid_points: [17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        clut: Some(LutStore::Store16(gray_to_xyz.clone())),
        b_curves: vec![
            ToneReprCurve::Lut(vec![]),
            ToneReprCurve::Lut(vec![]),
            ToneReprCurve::Lut(vec![]),
        ],
        matrix: d65_to_d50.mul(scale_rgb_to_xyz_matrix),
        a_curves: vec![srgb_profile.red_trc.clone().unwrap().clone()],
        m_curves: vec![
            ToneReprCurve::Lut(vec![]),
            ToneReprCurve::Lut(vec![]),
            ToneReprCurve::Lut(vec![]),
        ],
        bias: Vector3::default(),
    });
    gray_profile.lut_a_to_b_colorimetric = Some(a_to_b.clone());
    gray_profile.lut_a_to_b_perceptual = Some(a_to_b);

    gray_profile
}
