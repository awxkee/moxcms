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
#![cfg(feature = "lut")]
use crate::err::try_vec;
use crate::profile::LutDataType;
use crate::safe_math::{SafeMul, SafePowi};
use crate::trc::lut_interp_linear_float;
use crate::{
    CmsError, Cube, DataColorSpace, InterpolationMethod, MalformedSize, Matrix3d, Matrix3f, Stage,
    TransformOptions, Vector3f, Vector4f,
};
use num_traits::AsPrimitive;

#[derive(Default)]
struct Lut3x4 {
    matrix: Matrix3f,
    has_matrix: bool,
    input: [Vec<f32>; 3],
    clut: Vec<f32>,
    grid_size: u8,
    gamma: [Vec<f32>; 4],
    interpolation_method: InterpolationMethod,
    pcs: DataColorSpace,
}

fn make_lut_3x4(
    lut: &LutDataType,
    options: TransformOptions,
    pcs: DataColorSpace,
    is_input_pcs: bool,
) -> Result<Lut3x4, CmsError> {
    let clut_length: usize = (lut.num_clut_grid_points as usize)
        .safe_powi(lut.num_input_channels as u32)?
        .safe_mul(lut.num_output_channels as usize)?;

    let clut_table = lut.clut_table.to_clut_f32();
    if clut_table.len() != clut_length {
        return Err(CmsError::MalformedClut(MalformedSize {
            size: clut_table.len(),
            expected: clut_length,
        }));
    }

    let linearization_table = lut.input_table.to_clut_f32();

    if linearization_table.len() < lut.num_input_table_entries as usize * 3 {
        return Err(CmsError::MalformedCurveLutTable(MalformedSize {
            size: linearization_table.len(),
            expected: lut.num_input_table_entries as usize * 3,
        }));
    }

    let linear_curve0 = linearization_table[..lut.num_input_table_entries as usize].to_vec();
    let linear_curve1 = linearization_table
        [lut.num_input_table_entries as usize..lut.num_input_table_entries as usize * 2]
        .to_vec();
    let linear_curve2 = linearization_table
        [lut.num_input_table_entries as usize * 2..lut.num_input_table_entries as usize * 3]
        .to_vec();

    let gamma_table = lut.output_table.to_clut_f32();

    if gamma_table.len() < lut.num_output_table_entries as usize * 4 {
        return Err(CmsError::MalformedCurveLutTable(MalformedSize {
            size: gamma_table.len(),
            expected: lut.num_output_table_entries as usize * 4,
        }));
    }

    let gamma_curve0 = gamma_table[..lut.num_output_table_entries as usize].to_vec();
    let gamma_curve1 = gamma_table
        [lut.num_output_table_entries as usize..lut.num_output_table_entries as usize * 2]
        .to_vec();
    let gamma_curve2 = gamma_table
        [lut.num_output_table_entries as usize * 2..lut.num_output_table_entries as usize * 3]
        .to_vec();
    let gamma_curve3 = gamma_table
        [lut.num_output_table_entries as usize * 3..lut.num_output_table_entries as usize * 4]
        .to_vec();

    let matrix_f = lut.matrix.to_f32();
    let has_matrix = is_input_pcs && lut.matrix != Matrix3d::IDENTITY && pcs == DataColorSpace::Xyz;

    let transform = Lut3x4 {
        matrix: matrix_f,
        has_matrix,
        input: [linear_curve0, linear_curve1, linear_curve2],
        interpolation_method: options.interpolation_method,
        clut: clut_table,
        grid_size: lut.num_clut_grid_points,
        pcs,
        gamma: [gamma_curve0, gamma_curve1, gamma_curve2, gamma_curve3],
    };
    Ok(transform)
}

pub(crate) fn stage_lut_3x4(
    lut: &LutDataType,
    options: TransformOptions,
    pcs: DataColorSpace,
    is_input_pcs: bool,
) -> Result<Box<dyn Stage>, CmsError> {
    let transform = make_lut_3x4(lut, options, pcs, is_input_pcs)?;
    Ok(Box::new(transform))
}

pub(crate) fn create_lut3x4(
    lut: &LutDataType,
    src: &[f32],
    options: TransformOptions,
    pcs: DataColorSpace,
    is_input_pcs: bool,
) -> Result<Vec<f32>, CmsError> {
    if lut.num_input_channels != 3 || lut.num_output_channels != 4 {
        return Err(CmsError::UnsupportedProfileConnection);
    }

    let mut dest = try_vec![0.; (src.len() / 3) * 4];

    let lut_stage = stage_lut_3x4(lut, options, pcs, is_input_pcs)?;
    lut_stage.transform(src, &mut dest)?;
    Ok(dest)
}

impl Lut3x4 {
    fn transform_impl<Fetch: Fn(f32, f32, f32) -> Vector4f>(
        &self,
        src: &[f32],
        dst: &mut [f32],
        fetch: Fetch,
    ) -> Result<(), CmsError> {
        if self.has_matrix {
            self.transform_loop::<Fetch, true>(src, dst, &fetch)
        } else {
            self.transform_loop::<Fetch, false>(src, dst, &fetch)
        }
    }

    #[inline(always)]
    fn transform_loop<Fetch: Fn(f32, f32, f32) -> Vector4f, const HAS_MATRIX: bool>(
        &self,
        src: &[f32],
        dst: &mut [f32],
        fetch: &Fetch,
    ) -> Result<(), CmsError> {
        let linearization_0 = &self.input[0];
        let linearization_1 = &self.input[1];
        let linearization_2 = &self.input[2];
        for (dest, src) in dst
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(src.as_chunks::<3>().0.iter())
        {
            debug_assert!(self.grid_size as i32 >= 1);
            let (in_x, in_y, in_z) = if HAS_MATRIX {
                let m = self.matrix.mul_vector(Vector3f {
                    v: [src[0], src[1], src[2]],
                });
                (
                    m.v[0].clamp(0.0, 1.0),
                    m.v[1].clamp(0.0, 1.0),
                    m.v[2].clamp(0.0, 1.0),
                )
            } else {
                (src[0], src[1], src[2])
            };
            let linear_x = lut_interp_linear_float(in_x, linearization_0);
            let linear_y = lut_interp_linear_float(in_y, linearization_1);
            let linear_z = lut_interp_linear_float(in_z, linearization_2);

            let clut = fetch(linear_x, linear_y, linear_z);

            let pcs_x = lut_interp_linear_float(clut.v[0], &self.gamma[0]);
            let pcs_y = lut_interp_linear_float(clut.v[1], &self.gamma[1]);
            let pcs_z = lut_interp_linear_float(clut.v[2], &self.gamma[2]);
            let pcs_w = lut_interp_linear_float(clut.v[3], &self.gamma[3]);
            dest[0] = pcs_x;
            dest[1] = pcs_y;
            dest[2] = pcs_z;
            dest[3] = pcs_w;
        }
        Ok(())
    }
}

impl Stage for Lut3x4 {
    fn transform(&self, src: &[f32], dst: &mut [f32]) -> Result<(), CmsError> {
        let l_tbl = Cube::new(&self.clut, self.grid_size as usize, 4)?;

        // If PCS is LAB then linear interpolation should be used
        if self.pcs == DataColorSpace::Lab || self.pcs == DataColorSpace::Xyz {
            return self.transform_impl(src, dst, |x, y, z| l_tbl.trilinear_vec4(x, y, z));
        }

        match self.interpolation_method {
            #[cfg(feature = "options")]
            InterpolationMethod::Tetrahedral => {
                self.transform_impl(src, dst, |x, y, z| l_tbl.tetra_vec4(x, y, z))?;
            }
            #[cfg(feature = "options")]
            InterpolationMethod::Pyramid => {
                self.transform_impl(src, dst, |x, y, z| l_tbl.pyramid_vec4(x, y, z))?;
            }
            #[cfg(feature = "options")]
            InterpolationMethod::Prism => {
                self.transform_impl(src, dst, |x, y, z| l_tbl.prism_vec4(x, y, z))?;
            }
            InterpolationMethod::Linear => {
                self.transform_impl(src, dst, |x, y, z| l_tbl.trilinear_vec4(x, y, z))?;
            }
        }
        Ok(())
    }
}

pub(crate) fn create_lut3_samples<T: Copy + 'static, const SAMPLES: usize>() -> Vec<T>
where
    u32: AsPrimitive<T>,
{
    let lut_size: u32 = (3 * SAMPLES * SAMPLES * SAMPLES) as u32;

    assert!(SAMPLES >= 1);

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

    assert!(SAMPLES >= 1);

    let scale = 1. / (SAMPLES as f32 - 1.0);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::{LutStore, LutType};

    #[test]
    fn test_lut3x4_matrix_application() {
        // Non-symmetric, non-diagonal matrix with both positive and negative off-diagonal terms
        let matrix = Matrix3d {
            v: [
                [1.2, -0.2, 0.1],
                [-0.1, 1.1, 0.2],
                [0.05, -0.15, 1.05],
            ],
        };
        let input_table = LutStore::Store16(vec![0, 65535, 0, 65535, 0, 65535]);
        let output_table =
            LutStore::Store16(vec![0, 65535, 0, 65535, 0, 65535, 0, 65535]);
        let mut clut = Vec::new();
        // 2x2x2 cube mapping (r, g, b) to (r, g, b, 0.5)
        for r in 0..2 {
            for g in 0..2 {
                for b in 0..2 {
                    clut.push(if r == 1 { 65535 } else { 0 });
                    clut.push(if g == 1 { 65535 } else { 0 });
                    clut.push(if b == 1 { 65535 } else { 0 });
                    clut.push(32768); // 0.5
                }
            }
        }
        let lut_data = LutDataType {
            num_input_channels: 3,
            num_output_channels: 4,
            num_clut_grid_points: 2,
            matrix,
            num_input_table_entries: 2,
            num_output_table_entries: 2,
            input_table,
            clut_table: LutStore::Store16(clut),
            output_table,
            lut_type: LutType::Lut16,
        };

        // 1. In-range matrix application and channel mixing
        // M * [0.4, 0.5, 0.6] = [0.44, 0.63, 0.575]
        let stage = stage_lut_3x4(
            &lut_data,
            TransformOptions::default(),
            DataColorSpace::Xyz,
            true,
        )
        .unwrap();
        let src = [0.4f32, 0.5f32, 0.6f32];
        let mut dst = [0.0f32; 4];
        stage.transform(&src, &mut dst).unwrap();
        assert!((dst[0] - 0.44).abs() < 1e-4, "Expected ~0.44 but got {}", dst[0]);
        assert!((dst[1] - 0.63).abs() < 1e-4, "Expected ~0.63 but got {}", dst[1]);
        assert!((dst[2] - 0.575).abs() < 1e-4, "Expected ~0.575 but got {}", dst[2]);
        assert!((dst[3] - 0.5).abs() < 1e-3, "Expected ~0.5 but got {}", dst[3]);

        // 2. Lower clamping to 0.0: M * [0.1, 0.9, 0.1] = [-0.05, 1.00, -0.025] -> [0.0, 1.0, 0.0]
        let src_lower = [0.1f32, 0.9f32, 0.1f32];
        stage.transform(&src_lower, &mut dst).unwrap();
        assert!((dst[0] - 0.0).abs() < 1e-4, "Expected 0.0 lower clamp but got {}", dst[0]);
        assert!((dst[1] - 1.0).abs() < 1e-4, "Expected 1.0 but got {}", dst[1]);
        assert!((dst[2] - 0.0).abs() < 1e-4, "Expected 0.0 lower clamp but got {}", dst[2]);

        // 3. Upper clamping to 1.0: M * [0.9, 0.1, 0.8] = [1.14, 0.18, 0.87] -> [1.0, 0.18, 0.87]
        let src_upper = [0.9f32, 0.1f32, 0.8f32];
        stage.transform(&src_upper, &mut dst).unwrap();
        assert!((dst[0] - 1.0).abs() < 1e-4, "Expected 1.0 upper clamp but got {}", dst[0]);
        assert!((dst[1] - 0.18).abs() < 1e-4, "Expected ~0.18 but got {}", dst[1]);
        assert!((dst[2] - 0.87).abs() < 1e-4, "Expected ~0.87 but got {}", dst[2]);

        // 4. Gating test: matrix NOT applied when pcs == DataColorSpace::Lab
        let stage_lab = stage_lut_3x4(
            &lut_data,
            TransformOptions::default(),
            DataColorSpace::Lab,
            true,
        )
        .unwrap();
        stage_lab.transform(&src, &mut dst).unwrap();
        assert!((dst[0] - 0.4).abs() < 1e-4, "Expected 0.4 without matrix but got {}", dst[0]);
        assert!((dst[1] - 0.5).abs() < 1e-4, "Expected 0.5 without matrix but got {}", dst[1]);
        assert!((dst[2] - 0.6).abs() < 1e-4, "Expected 0.6 without matrix but got {}", dst[2]);

        // 5. Gating test: matrix NOT applied when is_input_pcs == false
        let stage_not_pcs = stage_lut_3x4(
            &lut_data,
            TransformOptions::default(),
            DataColorSpace::Xyz,
            false,
        )
        .unwrap();
        stage_not_pcs.transform(&src, &mut dst).unwrap();
        assert!((dst[0] - 0.4).abs() < 1e-4, "Expected 0.4 without matrix but got {}", dst[0]);
        assert!((dst[1] - 0.5).abs() < 1e-4, "Expected 0.5 without matrix but got {}", dst[1]);
        assert!((dst[2] - 0.6).abs() < 1e-4, "Expected 0.6 without matrix but got {}", dst[2]);
    }
}
