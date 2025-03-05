/*
 * // Copyright (c) Radzivon Bartoshyk 2/2025. All rights reserved.
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
use crate::conversions::tetrahedral::Tetrahedral;
use crate::lab::Lab;
use crate::mlaf::mlaf;
use crate::nd_array::{Array4D, lerp};
use crate::profile::LutDataType;
use crate::trc::{clamp_float, lut_interp_linear_float};
use crate::{
    CmsError, ColorProfile, DataColorSpace, InPlaceStage, Layout, Matrix3f, Stage,
    TransformExecutor, Vector3f, Xyz, rounding_div_ceil,
};
use num_traits::AsPrimitive;
use std::marker::PhantomData;

#[derive(Default)]
struct Lut4 {
    linearization: [Vec<f32>; 4],
    clut: Vec<f32>,
    grid_size: u8,
    output: [Vec<f32>; 3],
}

#[derive(Default)]
pub(crate) struct StageLabToXyz {}

impl InPlaceStage for StageLabToXyz {
    fn transform(&self, dst: &mut [f32]) -> Result<(), CmsError> {
        for dst in dst.chunks_exact_mut(3) {
            let lab = Lab::new(dst[0], dst[1], dst[2]);
            let xyz = lab.to_pcs_xyz();
            dst[0] = xyz.x;
            dst[1] = xyz.y;
            dst[2] = xyz.z;
        }
        Ok(())
    }
}

#[derive(Default)]
pub(crate) struct StageXyzToLab {}

impl InPlaceStage for StageXyzToLab {
    fn transform(&self, dst: &mut [f32]) -> Result<(), CmsError> {
        for dst in dst.chunks_exact_mut(3) {
            let xyz = Xyz::new(dst[0], dst[1], dst[2]);
            let lab = Lab::from_pcs_xyz(xyz);
            dst[0] = lab.l;
            dst[1] = lab.a;
            dst[2] = lab.b;
        }
        Ok(())
    }
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
            dest[0] = clamp_float(pcs_x);
            dest[1] = clamp_float(pcs_y);
            dest[2] = clamp_float(pcs_z);
        }
        Ok(())
    }
}

fn stage_lut_4x3(lut: &LutDataType) -> Box<dyn Stage> {
    let clut_length: usize = (lut.num_clut_grid_points as usize).pow(lut.num_input_channels as u32)
        * lut.num_output_channels as usize;
    // the matrix of lutType is only used when the input color space is XYZ.

    // Prepare input curves
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

fn create_lut4<const SAMPLES: usize>(lut: &LutDataType) -> Result<Vec<f32>, CmsError> {
    if lut.num_input_channels != 4 {
        return Err(CmsError::UnsupportedProfileConnection);
    }
    let lut_size: u32 = (4 * SAMPLES * SAMPLES * SAMPLES * SAMPLES) as u32;

    let mut src = Vec::with_capacity(lut_size as usize);
    let mut dest = vec![0.; lut_size as usize];
    /* Prepare a list of points we want to sample */
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

struct TransformLut4XyzToRgb<T, const LAYOUT: u8, const GRID_SIZE: usize, const BIT_DEPTH: usize> {
    lut: Vec<f32>,
    _phantom: PhantomData<T>,
}

struct XyzToRgbStage<T: Clone, const BIT_DEPTH: usize, const GAMMA_LUT: usize> {
    r_gamma: Box<[T; 65536]>,
    g_gamma: Box<[T; 65536]>,
    b_gamma: Box<[T; 65536]>,
    matrices: Vec<Matrix3f>,
}

impl<T: Clone + AsPrimitive<f32>, const BIT_DEPTH: usize, const GAMMA_LUT: usize> InPlaceStage
    for XyzToRgbStage<T, BIT_DEPTH, GAMMA_LUT>
{
    fn transform(&self, dst: &mut [f32]) -> Result<(), CmsError> {
        assert!(BIT_DEPTH >= 8);
        if !self.matrices.is_empty() {
            let m = self.matrices[0];
            for dst in dst.chunks_exact_mut(3) {
                let x = dst[0];
                let y = dst[1];
                let z = dst[2];
                dst[0] = mlaf(mlaf(x * m.v[0][0], y, m.v[0][1]), z, m.v[0][2]);
                dst[1] = mlaf(mlaf(x * m.v[1][0], y, m.v[1][1]), z, m.v[1][2]);
                dst[2] = mlaf(mlaf(x * m.v[2][0], y, m.v[2][1]), z, m.v[2][2]);
            }
        }

        for m in self.matrices.iter().skip(1) {
            for dst in dst.chunks_exact_mut(3) {
                let x = dst[0];
                let y = dst[1];
                let z = dst[2];
                dst[0] = mlaf(mlaf(x * m.v[0][0], y, m.v[0][1]), z, m.v[0][2]);
                dst[1] = mlaf(mlaf(x * m.v[1][0], y, m.v[1][1]), z, m.v[1][2]);
                dst[2] = mlaf(mlaf(x * m.v[2][0], y, m.v[2][1]), z, m.v[2][2]);
            }
        }

        let max_colors = (1 << BIT_DEPTH) - 1;
        let color_scale = 1f32 / max_colors as f32;
        let lut_cap = (GAMMA_LUT - 1) as f32;

        for dst in dst.chunks_exact_mut(3) {
            let r = mlaf(0.5f32, dst[0], lut_cap).min(lut_cap).max(0f32) as u16;
            let g = mlaf(0.5f32, dst[1], lut_cap).min(lut_cap).max(0f32) as u16;
            let b = mlaf(0.5f32, dst[2], lut_cap).min(lut_cap).max(0f32) as u16;
            dst[0] = self.r_gamma[r as usize].as_() * color_scale;
            dst[1] = self.g_gamma[g as usize].as_() * color_scale;
            dst[2] = self.b_gamma[b as usize].as_() * color_scale;
        }

        Ok(())
    }
}

pub(crate) trait CompressCmykLut {
    fn compress_cmyk_lut<const BIT_DEPTH: usize>(self) -> u8;
}

impl CompressCmykLut for u8 {
    #[inline]
    fn compress_cmyk_lut<const BIT_DEPTH: usize>(self) -> u8 {
        self
    }
}

impl CompressCmykLut for u16 {
    #[inline]
    fn compress_cmyk_lut<const BIT_DEPTH: usize>(self) -> u8 {
        let scale = BIT_DEPTH - 8;
        (self >> scale).min(255) as u8
    }
}

impl<
    T: Copy + AsPrimitive<f32> + Default + CompressCmykLut,
    const LAYOUT: u8,
    const GRID_SIZE: usize,
    const BIT_DEPTH: usize,
> TransformLut4XyzToRgb<T, LAYOUT, GRID_SIZE, BIT_DEPTH>
where
    f32: AsPrimitive<T>,
    u32: AsPrimitive<T>,
{
    #[inline(always)]
    fn transform_chunk(&self, src: &[T], dst: &mut [T]) {
        let cn = Layout::from(LAYOUT);
        let channels = cn.channels();
        let grid_size = GRID_SIZE as i32;
        let grid_size3 = grid_size * grid_size * grid_size;

        let value_scale = ((1 << BIT_DEPTH) - 1) as f32;
        let max_value = ((1 << BIT_DEPTH) - 1u32).as_();

        for (src, dst) in src.chunks_exact(4).zip(dst.chunks_exact_mut(channels)) {
            let c = src[0].compress_cmyk_lut::<BIT_DEPTH>();
            let m = src[1].compress_cmyk_lut::<BIT_DEPTH>();
            let y = src[2].compress_cmyk_lut::<BIT_DEPTH>();
            let k = src[3].compress_cmyk_lut::<BIT_DEPTH>();
            let linear_k: f32 = k as i32 as f32 / 255.0;
            let w: i32 = k as i32 * (GRID_SIZE as i32 - 1) / 255;
            let w_n: i32 = rounding_div_ceil(k as i32 * (GRID_SIZE as i32 - 1), 255);
            let t: f32 = linear_k * (GRID_SIZE as i32 - 1) as f32 - w as f32;

            let table1 = &self.lut[(w * grid_size3 * 3) as usize..];
            let table2 = &self.lut[(w_n * grid_size3 * 3) as usize..];

            let tetrahedral1 = Tetrahedral::<GRID_SIZE>::new(table1);
            let tetrahedral2 = Tetrahedral::<GRID_SIZE>::new(table2);
            let r1 = tetrahedral1.interpolate(c, m, y);
            let r2 = tetrahedral2.interpolate(c, m, y);
            let r = lerp(r1, r2, Vector3f::from(t)) * value_scale + 0.5f32;
            dst[cn.r_i()] = r.v[0].as_();
            dst[cn.g_i()] = r.v[1].as_();
            dst[cn.b_i()] = r.v[2].as_();
            if channels == 4 {
                dst[cn.a_i()] = max_value;
            }
        }
    }
}

impl<
    T: Copy + AsPrimitive<f32> + Default + CompressCmykLut,
    const LAYOUT: u8,
    const GRID_SIZE: usize,
    const BIT_DEPTH: usize,
> TransformExecutor<T> for TransformLut4XyzToRgb<T, LAYOUT, GRID_SIZE, BIT_DEPTH>
where
    f32: AsPrimitive<T>,
    u32: AsPrimitive<T>,
{
    fn transform(&self, src: &[T], dst: &mut [T]) -> Result<(), CmsError> {
        let cn = Layout::from(LAYOUT);
        let channels = cn.channels();
        if src.len() % 4 != 0 {
            return Err(CmsError::LaneMultipleOfChannels);
        }
        if dst.len() % channels != 0 {
            return Err(CmsError::LaneMultipleOfChannels);
        }
        let src_chunks = src.len() / 4;
        let dst_chunks = dst.len() / channels;
        if src_chunks != dst_chunks {
            return Err(CmsError::LaneSizeMismatch);
        }

        self.transform_chunk(src, dst);

        Ok(())
    }
}

pub(crate) fn make_cmyk_to_rgb<
    T: Copy + Default + AsPrimitive<f32> + Send + Sync + CompressCmykLut,
    const BIT_DEPTH: usize,
    const GAMMA_LUT: usize,
>(
    src_layout: Layout,
    source: &ColorProfile,
    dst_layout: Layout,
    dest: &ColorProfile,
) -> Result<Box<dyn TransformExecutor<T> + Send + Sync>, CmsError>
where
    f32: AsPrimitive<T>,
    u32: AsPrimitive<T>,
{
    if src_layout != Layout::Rgba {
        return Err(CmsError::InvalidLayout);
    }
    if source.color_space != DataColorSpace::Cmyk && dest.color_space != DataColorSpace::Rgb {
        return Err(CmsError::UnsupportedProfileConnection);
    }
    if source.pcs != DataColorSpace::Xyz && source.pcs != DataColorSpace::Lab {
        return Err(CmsError::UnsupportedProfileConnection);
    }

    let lut_a_to_b = source
        .lut_a_to_b_perceptual
        .as_ref()
        .ok_or(CmsError::UnsupportedProfileConnection)?;

    if source.color_space == DataColorSpace::Cmyk && dest.color_space == DataColorSpace::Rgb {
        if dst_layout != Layout::Rgb && dst_layout != Layout::Rgba {
            return Err(CmsError::UnsupportedProfileConnection);
        }

        const GRID_SIZE: usize = 17;

        let mut lut = create_lut4::<GRID_SIZE>(lut_a_to_b)?;

        if source.pcs == DataColorSpace::Lab {
            let lab_to_xyz_stage = StageLabToXyz::default();
            lab_to_xyz_stage.transform(&mut lut)?;
        }

        if dest.pcs == DataColorSpace::Lab {
            let lab_to_xyz_stage = StageXyzToLab::default();
            lab_to_xyz_stage.transform(&mut lut)?;
        }

        if dest.color_space == DataColorSpace::Rgb {
            let gamma_map_r =
                dest.build_gamma_table::<T, 65536, GAMMA_LUT, BIT_DEPTH>(&dest.red_trc)?;
            let gamma_map_g =
                dest.build_gamma_table::<T, 65536, GAMMA_LUT, BIT_DEPTH>(&dest.green_trc)?;
            let gamma_map_b =
                dest.build_gamma_table::<T, 65536, GAMMA_LUT, BIT_DEPTH>(&dest.blue_trc)?;

            let xyz_to_rgb = dest
                .rgb_to_xyz_matrix()
                .ok_or(CmsError::UnsupportedProfileConnection)?
                .inverse()
                .ok_or(CmsError::UnsupportedProfileConnection)?;

            let mut matrices = vec![Matrix3f {
                v: [
                    [65535.0 / 32768.0, 0.0, 0.0],
                    [0.0, 65535.0 / 32768.0, 0.0],
                    [0.0, 0.0, 65535.0 / 32768.0],
                ],
            }];

            matrices.push(xyz_to_rgb);
            let xyz_to_rgb_stage = XyzToRgbStage::<T, BIT_DEPTH, GAMMA_LUT> {
                r_gamma: gamma_map_r,
                g_gamma: gamma_map_g,
                b_gamma: gamma_map_b,
                matrices,
            };
            xyz_to_rgb_stage.transform(&mut lut)?;

            return Ok(match dst_layout {
                Layout::Rgb => Box::new(TransformLut4XyzToRgb::<
                    T,
                    { Layout::Rgb as u8 },
                    GRID_SIZE,
                    BIT_DEPTH,
                > {
                    lut,
                    _phantom: PhantomData,
                }),
                Layout::Rgba => Box::new(TransformLut4XyzToRgb::<
                    T,
                    { Layout::Rgba as u8 },
                    GRID_SIZE,
                    BIT_DEPTH,
                > {
                    lut,
                    _phantom: PhantomData,
                }),
                _ => unimplemented!(),
            });
        }
    }

    Err(CmsError::UnsupportedProfileConnection)
}
