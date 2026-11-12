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
#[cfg(feature = "any_to_any")]
use crate::conversions::katana::{KatanaFinalStage, KatanaInitialStage};
use crate::err::{MalformedSize, try_vec};
use crate::profile::LutDataType;
use crate::safe_math::{SafeMul, SafePowi};
use crate::trc::lut_interp_linear_float;
use crate::matrix::{Matrix3f, Vector3f};
use crate::*;
#[cfg(feature = "any_to_any")]
use num_traits::AsPrimitive;

#[derive(Default)]
struct Lut3x3 {
    matrix: Matrix3f,
    has_matrix: bool,
    input: [Vec<f32>; 3],
    clut: Vec<f32>,
    grid_size: u8,
    gamma: [Vec<f32>; 3],
    interpolation_method: InterpolationMethod,
    pcs: DataColorSpace,
}

#[cfg(feature = "any_to_any")]
#[derive(Default)]
struct KatanaLut3x3<T: Copy + Default> {
    matrix: Matrix3f,
    has_matrix: bool,
    input: [Vec<f32>; 3],
    clut: Vec<f32>,
    grid_size: u8,
    gamma: [Vec<f32>; 3],
    interpolation_method: InterpolationMethod,
    pcs: DataColorSpace,
    _phantom: std::marker::PhantomData<T>,
    bit_depth: usize,
}

fn make_lut_3x3(
    lut: &LutDataType,
    options: TransformOptions,
    pcs: DataColorSpace,
    is_input_pcs: bool,
) -> Result<Lut3x3, CmsError> {
    let clut_length: usize = (lut.num_clut_grid_points as usize)
        .safe_powi(lut.num_input_channels as u32)?
        .safe_mul(lut.num_output_channels as usize)?;

    let lin_table = lut.input_table.to_clut_f32();

    if lin_table.len() < lut.num_input_table_entries as usize * 3 {
        return Err(CmsError::MalformedCurveLutTable(MalformedSize {
            size: lin_table.len(),
            expected: lut.num_input_table_entries as usize * 3,
        }));
    }

    let lin_curve0 = lin_table[..lut.num_input_table_entries as usize].to_vec();
    let lin_curve1 = lin_table
        [lut.num_input_table_entries as usize..lut.num_input_table_entries as usize * 2]
        .to_vec();
    let lin_curve2 = lin_table
        [lut.num_input_table_entries as usize * 2..lut.num_input_table_entries as usize * 3]
        .to_vec();

    let clut_table = lut.clut_table.to_clut_f32();
    if clut_table.len() != clut_length {
        return Err(CmsError::MalformedClut(MalformedSize {
            size: clut_table.len(),
            expected: clut_length,
        }));
    }

    let gamma_curves = lut.output_table.to_clut_f32();

    if gamma_curves.len() < lut.num_output_table_entries as usize * 3 {
        return Err(CmsError::MalformedCurveLutTable(MalformedSize {
            size: gamma_curves.len(),
            expected: lut.num_output_table_entries as usize * 3,
        }));
    }

    let gamma_curve0 = gamma_curves[..lut.num_output_table_entries as usize].to_vec();
    let gamma_curve1 = gamma_curves
        [lut.num_output_table_entries as usize..lut.num_output_table_entries as usize * 2]
        .to_vec();
    let gamma_curve2 = gamma_curves
        [lut.num_output_table_entries as usize * 2..lut.num_output_table_entries as usize * 3]
        .to_vec();

    let matrix_f = Matrix3f {
        v: [
            [
                lut.matrix.v[0][0] as f32,
                lut.matrix.v[0][1] as f32,
                lut.matrix.v[0][2] as f32,
            ],
            [
                lut.matrix.v[1][0] as f32,
                lut.matrix.v[1][1] as f32,
                lut.matrix.v[1][2] as f32,
            ],
            [
                lut.matrix.v[2][0] as f32,
                lut.matrix.v[2][1] as f32,
                lut.matrix.v[2][2] as f32,
            ],
        ],
    };
    let has_matrix = is_input_pcs && lut.matrix != Matrix3d::IDENTITY && pcs == DataColorSpace::Xyz;

    let transform = Lut3x3 {
        matrix: matrix_f,
        has_matrix,
        input: [lin_curve0, lin_curve1, lin_curve2],
        gamma: [gamma_curve0, gamma_curve1, gamma_curve2],
        interpolation_method: options.interpolation_method,
        clut: clut_table,
        grid_size: lut.num_clut_grid_points,
        pcs,
    };

    Ok(transform)
}

pub(crate) fn stage_lut_3x3(
    lut: &LutDataType,
    options: TransformOptions,
    pcs: DataColorSpace,
    is_input_pcs: bool,
) -> Result<Box<dyn Stage>, CmsError> {
    let transform = make_lut_3x3(lut, options, pcs, is_input_pcs)?;
    Ok(Box::new(transform))
}

#[cfg(feature = "any_to_any")]
pub(crate) fn katana_input_stage_lut_3x3<
    T: Copy + Default + AsPrimitive<f32> + PointeeSizeExpressible + Send + Sync,
>(
    lut: &LutDataType,
    options: TransformOptions,
    pcs: DataColorSpace,
    bit_depth: usize,
) -> Result<Box<dyn KatanaInitialStage<f32, T> + Send + Sync>, CmsError>
where
    f32: AsPrimitive<T>,
{
    let lut = make_lut_3x3(lut, options, pcs, false)?;

    let transform = KatanaLut3x3::<T> {
        matrix: lut.matrix,
        has_matrix: lut.has_matrix,
        input: lut.input,
        gamma: lut.gamma,
        interpolation_method: lut.interpolation_method,
        clut: lut.clut,
        grid_size: lut.grid_size,
        pcs: lut.pcs,
        _phantom: std::marker::PhantomData,
        bit_depth,
    };

    Ok(Box::new(transform))
}

#[cfg(feature = "any_to_any")]
pub(crate) fn katana_output_stage_lut_3x3<
    T: Copy + Default + AsPrimitive<f32> + PointeeSizeExpressible + Send + Sync,
>(
    lut: &LutDataType,
    options: TransformOptions,
    pcs: DataColorSpace,
    bit_depth: usize,
) -> Result<Box<dyn KatanaFinalStage<f32, T> + Send + Sync>, CmsError>
where
    f32: AsPrimitive<T>,
{
    let lut = make_lut_3x3(lut, options, pcs, true)?;

    let transform = KatanaLut3x3::<T> {
        matrix: lut.matrix,
        has_matrix: lut.has_matrix,
        input: lut.input,
        gamma: lut.gamma,
        interpolation_method: lut.interpolation_method,
        clut: lut.clut,
        grid_size: lut.grid_size,
        pcs: lut.pcs,
        _phantom: std::marker::PhantomData,
        bit_depth,
    };

    Ok(Box::new(transform))
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
        for (dest, src) in dst
            .as_chunks_mut::<3>()
            .0
            .iter_mut()
            .zip(src.as_chunks::<3>().0.iter())
        {
            debug_assert!(self.grid_size as i32 >= 1);
            let (in_x, in_y, in_z) = if self.has_matrix {
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
            dest[0] = pcs_x;
            dest[1] = pcs_y;
            dest[2] = pcs_z;
        }
        Ok(())
    }
}

impl Stage for Lut3x3 {
    fn transform(&self, src: &[f32], dst: &mut [f32]) -> Result<(), CmsError> {
        let l_tbl = Cube::new(&self.clut, self.grid_size as usize, 3)?;

        // If PCS is LAB then linear interpolation should be used
        if self.pcs == DataColorSpace::Lab || self.pcs == DataColorSpace::Xyz {
            return self.transform_impl(src, dst, |x, y, z| l_tbl.trilinear_vec3(x, y, z));
        }

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

#[cfg(feature = "any_to_any")]
impl<T: Copy + Default + PointeeSizeExpressible + AsPrimitive<f32>> KatanaLut3x3<T>
where
    f32: AsPrimitive<T>,
{
    fn to_pcs_impl<Fetch: Fn(f32, f32, f32) -> Vector3f>(
        &self,
        input: &[T],
        fetch: Fetch,
    ) -> Result<Vec<f32>, CmsError> {
        if !input.len().is_multiple_of(3) {
            return Err(CmsError::LaneMultipleOfChannels);
        }
        let normalizing_value = if T::FINITE {
            1.0 / ((1u32 << self.bit_depth) - 1) as f32
        } else {
            1.0
        };
        let mut dst = try_vec![0.; input.len()];
        let linearization_0 = &self.input[0];
        let linearization_1 = &self.input[1];
        let linearization_2 = &self.input[2];
        for (dest, src) in dst
            .as_chunks_mut::<3>()
            .0
            .iter_mut()
            .zip(input.as_chunks::<3>().0.iter())
        {
            let linear_x =
                lut_interp_linear_float(src[0].as_() * normalizing_value, linearization_0);
            let linear_y =
                lut_interp_linear_float(src[1].as_() * normalizing_value, linearization_1);
            let linear_z =
                lut_interp_linear_float(src[2].as_() * normalizing_value, linearization_2);

            let clut = fetch(linear_x, linear_y, linear_z);

            let pcs_x = lut_interp_linear_float(clut.v[0], &self.gamma[0]);
            let pcs_y = lut_interp_linear_float(clut.v[1], &self.gamma[1]);
            let pcs_z = lut_interp_linear_float(clut.v[2], &self.gamma[2]);
            dest[0] = pcs_x;
            dest[1] = pcs_y;
            dest[2] = pcs_z;
        }
        Ok(dst)
    }

    fn to_output<Fetch: Fn(f32, f32, f32) -> Vector3f>(
        &self,
        src: &[f32],
        dst: &mut [T],
        fetch: Fetch,
    ) -> Result<(), CmsError> {
        if !src.len().is_multiple_of(3) {
            return Err(CmsError::LaneMultipleOfChannels);
        }
        if !dst.len().is_multiple_of(3) {
            return Err(CmsError::LaneMultipleOfChannels);
        }
        if dst.len() != src.len() {
            return Err(CmsError::LaneSizeMismatch);
        }
        let norm_value = if T::FINITE {
            ((1u32 << self.bit_depth) - 1) as f32
        } else {
            1.0
        };

        let linearization_0 = &self.input[0];
        let linearization_1 = &self.input[1];
        let linearization_2 = &self.input[2];
        for (dest, src) in dst
            .as_chunks_mut::<3>()
            .0
            .iter_mut()
            .zip(src.as_chunks::<3>().0.iter())
        {
            let (in_x, in_y, in_z) = if self.has_matrix {
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

            if T::FINITE {
                dest[0] = (pcs_x * norm_value).round().max(0.0).min(norm_value).as_();
                dest[1] = (pcs_y * norm_value).round().max(0.0).min(norm_value).as_();
                dest[2] = (pcs_z * norm_value).round().max(0.0).min(norm_value).as_();
            } else {
                dest[0] = pcs_x.as_();
                dest[1] = pcs_y.as_();
                dest[2] = pcs_z.as_();
            }
        }
        Ok(())
    }
}

#[cfg(feature = "any_to_any")]
impl<T: Copy + Default + PointeeSizeExpressible + AsPrimitive<f32>> KatanaInitialStage<f32, T>
    for KatanaLut3x3<T>
where
    f32: AsPrimitive<T>,
{
    fn to_pcs(&self, input: &[T]) -> Result<Vec<f32>, CmsError> {
        let l_tbl = Cube::new(&self.clut, self.grid_size as usize, 3)?;

        // If PCS is LAB then linear interpolation should be used
        if self.pcs == DataColorSpace::Lab || self.pcs == DataColorSpace::Xyz {
            return self.to_pcs_impl(input, |x, y, z| l_tbl.trilinear_vec3(x, y, z));
        }

        match self.interpolation_method {
            #[cfg(feature = "options")]
            InterpolationMethod::Tetrahedral => {
                self.to_pcs_impl(input, |x, y, z| l_tbl.tetra_vec3(x, y, z))
            }
            #[cfg(feature = "options")]
            InterpolationMethod::Pyramid => {
                self.to_pcs_impl(input, |x, y, z| l_tbl.pyramid_vec3(x, y, z))
            }
            #[cfg(feature = "options")]
            InterpolationMethod::Prism => {
                self.to_pcs_impl(input, |x, y, z| l_tbl.prism_vec3(x, y, z))
            }
            InterpolationMethod::Linear => {
                self.to_pcs_impl(input, |x, y, z| l_tbl.trilinear_vec3(x, y, z))
            }
        }
    }
}

#[cfg(feature = "any_to_any")]
impl<T: Copy + Default + PointeeSizeExpressible + AsPrimitive<f32>> KatanaFinalStage<f32, T>
    for KatanaLut3x3<T>
where
    f32: AsPrimitive<T>,
{
    fn to_output(&self, src: &mut [f32], dst: &mut [T]) -> Result<(), CmsError> {
        let l_tbl = Cube::new(&self.clut, self.grid_size as usize, 3)?;

        // If PCS is LAB then linear interpolation should be used
        if self.pcs == DataColorSpace::Lab || self.pcs == DataColorSpace::Xyz {
            return self.to_output(src, dst, |x, y, z| l_tbl.trilinear_vec3(x, y, z));
        }

        match self.interpolation_method {
            #[cfg(feature = "options")]
            InterpolationMethod::Tetrahedral => {
                self.to_output(src, dst, |x, y, z| l_tbl.tetra_vec3(x, y, z))
            }
            #[cfg(feature = "options")]
            InterpolationMethod::Pyramid => {
                self.to_output(src, dst, |x, y, z| l_tbl.pyramid_vec3(x, y, z))
            }
            #[cfg(feature = "options")]
            InterpolationMethod::Prism => {
                self.to_output(src, dst, |x, y, z| l_tbl.prism_vec3(x, y, z))
            }
            InterpolationMethod::Linear => {
                self.to_output(src, dst, |x, y, z| l_tbl.trilinear_vec3(x, y, z))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::{LutStore, LutType};

    #[test]
    fn test_lut3x3_matrix_application() {
        let matrix = Matrix3d {
            v: [
                [2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        };
        let input_table = LutStore::Store16(vec![0, 65535, 0, 65535, 0, 65535]);
        let output_table = LutStore::Store16(vec![0, 65535, 0, 65535, 0, 65535]);
        let mut clut = Vec::new();
        for r in 0..2 {
            for g in 0..2 {
                for b in 0..2 {
                    clut.push(if r == 1 { 65535 } else { 0 });
                    clut.push(if g == 1 { 65535 } else { 0 });
                    clut.push(if b == 1 { 65535 } else { 0 });
                }
            }
        }
        let lut_data = LutDataType {
            num_input_channels: 3,
            num_output_channels: 3,
            num_clut_grid_points: 2,
            matrix,
            num_input_table_entries: 2,
            num_output_table_entries: 2,
            input_table,
            clut_table: LutStore::Store16(clut),
            output_table,
            lut_type: LutType::Lut16,
        };
        let stage =
            stage_lut_3x3(&lut_data, TransformOptions::default(), DataColorSpace::Xyz, true).unwrap();
        let src = [0.25f32, 0.5f32, 0.5f32];
        let mut dst = [0.0f32; 3];
        stage.transform(&src, &mut dst).unwrap();
        assert!((dst[0] - 0.5).abs() < 0.02, "Expected ~0.5 but got {}", dst[0]);

        // Test clamping to 1.0 when matrix exceeds 1.0
        let src_clamp = [0.8f32, 0.5f32, 0.5f32]; // 0.8 * 2.0 = 1.6 -> clamped to 1.0
        stage.transform(&src_clamp, &mut dst).unwrap();
        assert!((dst[0] - 1.0).abs() < 0.02, "Expected ~1.0 but got {}", dst[0]);
    }
}

pub(crate) fn create_lut3x3(
    lut: &LutDataType,
    src: &[f32],
    options: TransformOptions,
    pcs: DataColorSpace,
    is_input_pcs: bool,
) -> Result<Vec<f32>, CmsError> {
    if lut.num_input_channels != 3 || lut.num_output_channels != 3 {
        return Err(CmsError::UnsupportedProfileConnection);
    }

    let mut dest = try_vec![0.; src.len()];

    let lut_stage = stage_lut_3x3(lut, options, pcs, is_input_pcs)?;
    lut_stage.transform(src, &mut dest)?;
    Ok(dest)
}
