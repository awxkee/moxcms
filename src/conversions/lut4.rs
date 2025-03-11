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
use crate::{Array4D, CmsError, Stage};

#[derive(Default)]
struct Lut4 {
    linearization: [Vec<f32>; 4],
    clut: Vec<f32>,
    grid_size: u8,
    output: [Vec<f32>; 3],
}

impl Stage for Lut4 {
    fn transform(&self, src: &[f32], dst: &mut [f32]) -> Result<(), CmsError> {
        let l_tbl = Array4D::new(&self.clut[0..], self.grid_size as usize);

        let linearization_0 = &self.linearization[0];
        let linearization_1 = &self.linearization[1];
        let linearization_2 = &self.linearization[2];
        let linearization_3 = &self.linearization[3];
        for (dest, src) in dst.chunks_exact_mut(3).zip(src.chunks_exact(4)) {
            debug_assert!(self.grid_size as i32 >= 1);
            let linear_x = lut_interp_linear_float(src[0], linearization_0);
            let linear_y = lut_interp_linear_float(src[1], linearization_1);
            let linear_z = lut_interp_linear_float(src[2], linearization_2);
            let linear_w = lut_interp_linear_float(src[3], linearization_3);

            let clut = l_tbl.quadlinear_vec3(linear_x, linear_y, linear_z, linear_w);

            let pcs_x = lut_interp_linear_float(clut.v[0], &self.output[0]);
            let pcs_y = lut_interp_linear_float(clut.v[1], &self.output[1]);
            let pcs_z = lut_interp_linear_float(clut.v[2], &self.output[2]);
            dest[0] = m_clamp(pcs_x, 0.0, 1.0f32);
            dest[1] = m_clamp(pcs_y, 0.0, 1.0f32);
            dest[2] = m_clamp(pcs_z, 0.0, 1.0f32);
        }
        Ok(())
    }
}

fn stage_lut_4x3(lut: &LutDataType) -> Box<dyn Stage> {
    let clut_length: usize = (lut.num_clut_grid_points as usize).pow(lut.num_input_channels as u32)
        * lut.num_output_channels as usize;

    let mut transform = Lut4::default();
    transform.linearization[0] = lut.input_table[0..lut.num_input_table_entries as usize].to_vec();
    transform.linearization[1] = lut.input_table
        [lut.num_input_table_entries as usize..lut.num_input_table_entries as usize * 2]
        .to_vec();
    transform.linearization[2] = lut.input_table
        [lut.num_input_table_entries as usize * 2..lut.num_input_table_entries as usize * 3]
        .to_vec();
    transform.linearization[3] = lut.input_table
        [lut.num_input_table_entries as usize * 3..lut.num_input_table_entries as usize * 4]
        .to_vec();
    // Prepare table
    assert_eq!(clut_length, lut.clut_table.len());
    transform.clut = lut.clut_table.clone();

    transform.grid_size = lut.num_clut_grid_points;
    // Prepare output curves
    transform.output[0] = lut.output_table[0..lut.num_output_table_entries as usize].to_vec();
    transform.output[1] = lut.output_table
        [lut.num_output_table_entries as usize..lut.num_output_table_entries as usize * 2]
        .to_vec();
    transform.output[2] = lut.output_table
        [lut.num_output_table_entries as usize * 2..lut.num_output_table_entries as usize * 3]
        .to_vec();
    Box::new(transform)
}

pub(crate) fn create_lut4<const SAMPLES: usize>(lut: &LutDataType) -> Result<Vec<f32>, CmsError> {
    if lut.num_input_channels != 4 {
        return Err(CmsError::UnsupportedProfileConnection);
    }
    let lut_size: u32 = (4 * SAMPLES * SAMPLES * SAMPLES * SAMPLES) as u32;

    let mut src = Vec::with_capacity(lut_size as usize);
    let mut dest = vec![0.; (lut_size as usize) / 4 * 3];

    let recpeq = 1f32 / (SAMPLES - 1) as f32;
    for k in 0..SAMPLES {
        for c in 0..SAMPLES {
            for m in 0..SAMPLES {
                for y in 0..SAMPLES {
                    src.push(c as f32 * recpeq);
                    src.push(m as f32 * recpeq);
                    src.push(y as f32 * recpeq);
                    src.push(k as f32 * recpeq);
                }
            }
        }
    }
    let lut_stage = stage_lut_4x3(lut);
    lut_stage.transform(&src, &mut dest)?;
    Ok(dest)
}
