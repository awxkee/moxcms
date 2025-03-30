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
use crate::{
    Array4D, CmsError, DataColorSpace, InterpolationMethod, Stage, TransformOptions, Vector3f,
};

#[derive(Default)]
struct Lut4 {
    linearization: [Vec<f32>; 4],
    clut: Vec<f32>,
    grid_size: u8,
    output: [Vec<f32>; 3],
    interpolation_method: InterpolationMethod,
    pcs: DataColorSpace,
}

impl Lut4 {
    fn transform_impl<Fetch: Fn(f32, f32, f32, f32) -> Vector3f>(
        &self,
        src: &[f32],
        dst: &mut [f32],
        fetch: Fetch,
    ) -> Result<(), CmsError> {
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

            let clut = fetch(linear_x, linear_y, linear_z, linear_w);

            let pcs_x = lut_interp_linear_float(clut.v[0], &self.output[0]);
            let pcs_y = lut_interp_linear_float(clut.v[1], &self.output[1]);
            let pcs_z = lut_interp_linear_float(clut.v[2], &self.output[2]);
            dest[0] = pcs_x;
            dest[1] = pcs_y;
            dest[2] = pcs_z;
        }
        Ok(())
    }
}

macro_rules! define_lut4_dispatch {
    ($dispatcher: ident) => {
        impl Stage for $dispatcher {
            fn transform(&self, src: &[f32], dst: &mut [f32]) -> Result<(), CmsError> {
                let l_tbl = Array4D::new(&self.clut, self.grid_size as usize);

                // If Source PCS is LAB trilinear should be used
                if self.pcs == DataColorSpace::Lab {
                    return self
                        .transform_impl(src, dst, |x, y, z, w| l_tbl.quadlinear_vec3(x, y, z, w));
                }

                match self.interpolation_method {
                    #[cfg(feature = "options")]
                    InterpolationMethod::Tetrahedral => {
                        self.transform_impl(src, dst, |x, y, z, w| l_tbl.tetra_vec3(x, y, z, w))?;
                    }
                    #[cfg(feature = "options")]
                    InterpolationMethod::Pyramid => {
                        self.transform_impl(src, dst, |x, y, z, w| l_tbl.pyramid_vec3(x, y, z, w))?;
                    }
                    #[cfg(feature = "options")]
                    InterpolationMethod::Prism => {
                        self.transform_impl(src, dst, |x, y, z, w| l_tbl.prism_vec3(x, y, z, w))?
                    }
                    InterpolationMethod::Linear => {
                        self.transform_impl(src, dst, |x, y, z, w| {
                            l_tbl.quadlinear_vec3(x, y, z, w)
                        })?
                    }
                }
                Ok(())
            }
        }
    };
}

define_lut4_dispatch!(Lut4);

fn stage_lut_4x3(
    lut: &LutDataType,
    options: TransformOptions,
    pcs: DataColorSpace,
) -> Result<Box<dyn Stage>, CmsError> {
    // There is 4 possible cases:
    // - All curves are non-linear
    // - Linearization curves are non-linear, but gamma is linear
    // - Gamma curves are non-linear, but linearization is linear
    // - All curves linear
    // Currently not optimized
    let clut_length: usize = (lut.num_clut_grid_points as usize).pow(lut.num_input_channels as u32)
        * lut.num_output_channels as usize;

    let linearization_table = lut.input_table.to_clut_f32();

    let lin_curve0 = linearization_table[0..lut.num_input_table_entries as usize].to_vec();
    let lin_curve1 = linearization_table
        [lut.num_input_table_entries as usize..lut.num_input_table_entries as usize * 2]
        .to_vec();
    let lin_curve2 = linearization_table
        [lut.num_input_table_entries as usize * 2..lut.num_input_table_entries as usize * 3]
        .to_vec();
    let lin_curve3 = linearization_table
        [lut.num_input_table_entries as usize * 3..lut.num_input_table_entries as usize * 4]
        .to_vec();

    let gamma_table = lut.output_table.to_clut_f32();

    let gamma_curve0 = gamma_table[0..lut.num_output_table_entries as usize].to_vec();
    let gamma_curve1 = gamma_table
        [lut.num_output_table_entries as usize..lut.num_output_table_entries as usize * 2]
        .to_vec();
    let gamma_curve2 = gamma_table
        [lut.num_output_table_entries as usize * 2..lut.num_output_table_entries as usize * 3]
        .to_vec();

    let clut_table = lut.clut_table.to_clut_f32();
    assert_eq!(clut_length, clut_table.len());

    let transform = Lut4 {
        linearization: [lin_curve0, lin_curve1, lin_curve2, lin_curve3],
        interpolation_method: options.interpolation_method,
        pcs,
        clut: clut_table,
        grid_size: lut.num_clut_grid_points,
        output: [gamma_curve0, gamma_curve1, gamma_curve2],
    };
    Ok(Box::new(transform))
}

pub(crate) fn create_lut4_norm_samples<const SAMPLES: usize>() -> Vec<f32> {
    let lut_size: u32 = (4 * SAMPLES * SAMPLES * SAMPLES * SAMPLES) as u32;

    let mut src = Vec::with_capacity(lut_size as usize);

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
    src
}

pub(crate) fn create_lut4<const SAMPLES: usize>(
    lut: &LutDataType,
    options: TransformOptions,
    pcs: DataColorSpace,
) -> Result<Vec<f32>, CmsError> {
    if lut.num_input_channels != 4 {
        return Err(CmsError::UnsupportedProfileConnection);
    }
    let lut_size: u32 = (4 * SAMPLES * SAMPLES * SAMPLES * SAMPLES) as u32;

    let src = create_lut4_norm_samples::<SAMPLES>();
    let mut dest = vec![0.; (lut_size as usize) / 4 * 3];

    let lut_stage = stage_lut_4x3(lut, options, pcs)?;
    lut_stage.transform(&src, &mut dest)?;
    Ok(dest)
}
