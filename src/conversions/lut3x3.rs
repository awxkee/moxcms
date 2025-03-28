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

use crate::profile::LutDataType;
use crate::trc::lut_interp_linear_float;
use crate::{Array3D, CmsError, InterpolationMethod, Stage, TransformOptions, Vector3f};

#[derive(Default)]
struct Lut3x3 {
    input: [Vec<f32>; 3],
    clut: Vec<f32>,
    grid_size: u8,
    gamma: [Vec<f32>; 3],
    interpolation_method: InterpolationMethod,
}

fn stage_lut_3x3(lut: &LutDataType, options: TransformOptions) -> Box<dyn Stage> {
    let clut_length: usize = (lut.num_clut_grid_points as usize).pow(lut.num_input_channels as u32)
        * lut.num_output_channels as usize;
    // the matrix of lutType is only used when the input color space is XYZ.

    // Prepare input curves
    let mut transform = Lut3x3 {
        interpolation_method: options.interpolation_method,
        ..Default::default()
    };
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
    Box::new(transform)
}

impl Lut3x3 {
    fn transform_impl<Fetch: Fn(f32, f32, f32) -> Vector3f>(
        &self,
        src: &[f32],
        dst: &mut [f32],
        fetch: Fetch,
    ) -> Result<(), CmsError> {
        let linearization_0 = &self.input[0];
        let linearization_1 = &self.input[1];
        let linearization_2 = &self.input[2];
        for (dest, src) in dst.chunks_exact_mut(3).zip(src.chunks_exact(3)) {
            debug_assert!(self.grid_size as i32 >= 1);
            let linear_x = lut_interp_linear_float(src[0], linearization_0);
            let linear_y = lut_interp_linear_float(src[1], linearization_1);
            let linear_z = lut_interp_linear_float(src[2], linearization_2);

            let clut = fetch(linear_x, linear_y, linear_z);

            let pcs_x = lut_interp_linear_float(clut.v[0], &self.gamma[0]);
            let pcs_y = lut_interp_linear_float(clut.v[1], &self.gamma[1]);
            let pcs_z = lut_interp_linear_float(clut.v[2], &self.gamma[2]);
            dest[0] = pcs_x;
            dest[1] = pcs_y;
            dest[2] = pcs_z;
        }
        Ok(())
    }
}

impl Stage for Lut3x3 {
    fn transform(&self, src: &[f32], dst: &mut [f32]) -> Result<(), CmsError> {
        let l_tbl = Array3D::new(&self.clut, self.grid_size as usize);
        match self.interpolation_method {
            #[cfg(feature = "options")]
            InterpolationMethod::Tetrahedral => {
                self.transform_impl(src, dst, |x, y, z| l_tbl.tetra_vec3(x, y, z))?;
            }
            #[cfg(feature = "options")]
            InterpolationMethod::Pyramid => {
                self.transform_impl(src, dst, |x, y, z| l_tbl.pyramid_vec3(x, y, z))?;
            }
            #[cfg(feature = "options")]
            InterpolationMethod::Prism => {
                self.transform_impl(src, dst, |x, y, z| l_tbl.prism_vec3(x, y, z))?;
            }
            InterpolationMethod::Linear => {
                self.transform_impl(src, dst, |x, y, z| l_tbl.trilinear_vec3(x, y, z))?;
            }
        }
        Ok(())
    }
}

pub(crate) fn create_lut3x3(
    lut: &LutDataType,
    src: &[f32],
    options: TransformOptions,
) -> Result<Vec<f32>, CmsError> {
    if lut.num_input_channels != 3 || lut.num_output_channels != 3 {
        return Err(CmsError::UnsupportedProfileConnection);
    }

    let mut dest = vec![0.; src.len()];

    let lut_stage = stage_lut_3x3(lut, options);
    lut_stage.transform(src, &mut dest)?;
    Ok(dest)
}
