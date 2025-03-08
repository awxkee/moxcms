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
use crate::math::m_clamp;
use crate::profile::LutDataType;
use crate::trc::lut_interp_linear_float;
use crate::{Array3D, CmsError, Stage};
use num_traits::AsPrimitive;

#[derive(Default)]
struct Lut3x4 {
    input: [Vec<f32>; 3],
    clut: Vec<f32>,
    grid_size: u8,
    gamma: [Vec<f32>; 4],
}

fn stage_lut_3x4(lut: &LutDataType) -> Box<dyn Stage> {
    let clut_length: usize = (lut.num_clut_grid_points as usize).pow(lut.num_input_channels as u32)
        * lut.num_output_channels as usize;
    // the matrix of lutType is only used when the input color space is XYZ.

    // Prepare input curves
    let mut transform = Lut3x4::default();
    transform.input[0] = lut.input_table[0..lut.num_input_table_entries as usize].to_vec();
    transform.input[1] = lut.input_table
        [lut.num_input_table_entries as usize..lut.num_input_table_entries as usize * 2]
        .to_vec();
    transform.input[2] = lut.input_table
        [lut.num_input_table_entries as usize * 2..lut.num_input_table_entries as usize * 3]
        .to_vec();
    // Prepare table
    assert_eq!(clut_length, lut.clut_table.len());
    transform.clut = lut.clut_table.clone();

    transform.grid_size = lut.num_clut_grid_points;
    // Prepare output curves
    transform.gamma[0] = lut.output_table[0..lut.num_output_table_entries as usize].to_vec();
    transform.gamma[1] = lut.output_table
        [lut.num_output_table_entries as usize..lut.num_output_table_entries as usize * 2]
        .to_vec();
    transform.gamma[2] = lut.output_table
        [lut.num_output_table_entries as usize * 2..lut.num_output_table_entries as usize * 3]
        .to_vec();
    transform.gamma[3] = lut.output_table
        [lut.num_output_table_entries as usize * 3..lut.num_output_table_entries as usize * 4]
        .to_vec();
    Box::new(transform)
}

impl Stage for Lut3x4 {
    fn transform(&self, src: &[f32], dst: &mut [f32]) -> Result<(), CmsError> {
        let l_tbl = Array3D::new(&self.clut[0..], self.grid_size as usize);

        let linearization_0 = &self.input[0];
        let linearization_1 = &self.input[1];
        let linearization_2 = &self.input[2];
        for (dest, src) in dst.chunks_exact_mut(4).zip(src.chunks_exact(3)) {
            debug_assert!(self.grid_size as i32 >= 1);
            let linear_x = lut_interp_linear_float(src[0], linearization_0);
            let linear_y = lut_interp_linear_float(src[1], linearization_1);
            let linear_z = lut_interp_linear_float(src[2], linearization_2);

            let clut = l_tbl.trilinear_vec4(linear_x, linear_y, linear_z);

            let pcs_x = lut_interp_linear_float(clut.v[0], &self.gamma[0]);
            let pcs_y = lut_interp_linear_float(clut.v[1], &self.gamma[1]);
            let pcs_z = lut_interp_linear_float(clut.v[2], &self.gamma[2]);
            let pcs_w = lut_interp_linear_float(clut.v[3], &self.gamma[3]);
            dest[0] = m_clamp(pcs_x, 0.0, 1.0f32);
            dest[1] = m_clamp(pcs_y, 0.0, 1.0f32);
            dest[2] = m_clamp(pcs_z, 0.0, 1.0f32);
            dest[3] = m_clamp(pcs_w, 0.0, 1.0f32);
        }
        Ok(())
    }
}

pub(crate) fn create_lut3_samples<T: Copy + 'static, const SAMPLES: usize>() -> Vec<T>
where
    u32: AsPrimitive<T>,
{
    let lut_size: u32 = (3 * SAMPLES * SAMPLES * SAMPLES) as u32;

    let mut src = Vec::with_capacity(lut_size as usize);
    for x in 0..SAMPLES as u32 {
        for y in 0..SAMPLES as u32 {
            for z in 0..SAMPLES as u32 {
                src.push(x.as_());
                src.push(y.as_());
                src.push(z.as_());
            }
        }
    }
    src
}

pub(crate) fn create_lut3_samples_norm<const SAMPLES: usize>() -> Vec<f32> {
    let lut_size: u32 = (3 * SAMPLES * SAMPLES * SAMPLES) as u32;

    let scale = 1. / SAMPLES as f32;

    let mut src = Vec::with_capacity(lut_size as usize);
    for x in 0..SAMPLES as u32 {
        for y in 0..SAMPLES as u32 {
            for z in 0..SAMPLES as u32 {
                src.push(x as f32 * scale);
                src.push(y as f32 * scale);
                src.push(z as f32 * scale);
            }
        }
    }
    src
}

pub(crate) fn create_lut3x4<const SAMPLES: usize>(
    lut: &LutDataType,
    src: &[f32],
) -> Result<Vec<f32>, CmsError> {
    if lut.num_input_channels != 3 {
        return Err(CmsError::UnsupportedProfileConnection);
    }

    let mut dest = vec![0.; (src.len() / 3) * 4];

    let lut_stage = stage_lut_3x4(lut);
    lut_stage.transform(src, &mut dest)?;
    Ok(dest)
}
