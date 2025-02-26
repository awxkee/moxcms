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
use crate::lab::Lab;
use crate::mlaf::mlaf;
use crate::nd_array::{Array4D, lerp};
use crate::profile::LutDataType;
use crate::trc::{clamp_float, lut_interp_linear_float};
use crate::{
    CmsError, ColorProfile, DataColorSpace, Layout, Matrix3f, Stage, Transform8BitExecutor,
    TransformExecutor, Vector3f, Xyz,
};

#[derive(Default)]
struct Lut4 {
    linearization: [Vec<f32>; 4],
    clut: Vec<f32>,
    grid_size: u8,
    output: [Vec<f32>; 3],
}

#[derive(Default)]
pub(crate) struct StageLabToXyz {}

impl Stage for StageLabToXyz {
    fn transform(&self, src: &[f32], dst: &mut [f32]) -> Result<(), CmsError> {
        if src.len() != dst.len() {
            return Err(CmsError::LaneSizeMismatch);
        }
        for (src, dst) in src.chunks_exact(3).zip(dst.chunks_exact_mut(3)) {
            let lab = Lab::new(src[0], src[1], src[2]);
            let xyz = lab.to_pcs_xyz(None);
            dst[0] = xyz.x;
            dst[1] = xyz.y;
            dst[2] = xyz.z;
        }
        Ok(())
    }
}

#[derive(Default)]
pub(crate) struct StageXyzToLab {}

impl Stage for StageXyzToLab {
    fn transform(&self, src: &[f32], dst: &mut [f32]) -> Result<(), CmsError> {
        if src.len() != dst.len() {
            return Err(CmsError::LaneSizeMismatch);
        }
        for (src, dst) in src.chunks_exact(3).zip(dst.chunks_exact_mut(3)) {
            let xyz = Xyz::new(src[0], src[1], src[2]);
            let lab = Lab::from_pcs_xyz(xyz, None);
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

fn create_lut<const SAMPLES: usize>(lut: &LutDataType) -> Result<Vec<f32>, CmsError> {
    if lut.num_input_channels != 4 {
        return Err(CmsError::UnsupportedProfileConnection);
    }
    let lut_size: u32 = (4 * SAMPLES * SAMPLES * SAMPLES * SAMPLES) as u32;

    let mut src = Vec::with_capacity(lut_size as usize);
    let mut dest = vec![0.; lut_size as usize];
    /* Prepare a list of points we want to sample */
    for k in 0..SAMPLES {
        for c in 0..SAMPLES {
            for m in 0..SAMPLES {
                for y in 0..SAMPLES {
                    src.push(c as f32 / (SAMPLES - 1) as f32);
                    src.push(m as f32 / (SAMPLES - 1) as f32);
                    src.push(y as f32 / (SAMPLES - 1) as f32);
                    src.push(k as f32 / (SAMPLES - 1) as f32);
                }
            }
        }
    }
    let lut_stage = stage_lut_4x3(lut);
    lut_stage.transform(&src, &mut dest)?;
    Ok(dest)
}

struct TransformLut4XyzToRgb<const LAYOUT: u8, const GRID_SIZE: usize> {
    lut: Vec<f32>,
}

struct XyzToRgbStage {
    r_gamma: Box<[u8; 65536]>,
    g_gamma: Box<[u8; 65536]>,
    b_gamma: Box<[u8; 65536]>,
    matrices: Vec<Matrix3f>,
}

impl Stage for XyzToRgbStage {
    fn transform(&self, src: &[f32], dst: &mut [f32]) -> Result<(), CmsError> {
        if src.len() != dst.len() {
            return Err(CmsError::LaneSizeMismatch);
        }

        if !self.matrices.is_empty() {
            let m = self.matrices[0];
            for (src, dst) in src.chunks_exact(3).zip(dst.chunks_exact_mut(3)) {
                let x = src[0];
                let y = src[1];
                let z = src[2];
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

        for dst in dst.chunks_exact_mut(3) {
            let r = mlaf(0.5f32, dst[0], 8191f32).min(8191f32).max(0f32) as u16;
            let g = mlaf(0.5f32, dst[1], 8191f32).min(8191f32).max(0f32) as u16;
            let b = mlaf(0.5f32, dst[2], 8191f32).min(8191f32).max(0f32) as u16;
            dst[0] = self.r_gamma[r as usize] as f32 * (1. / 255f32);
            dst[1] = self.g_gamma[g as usize] as f32 * (1. / 255f32);
            dst[2] = self.b_gamma[b as usize] as f32 * (1. / 255f32);
        }

        Ok(())
    }
}

#[inline]
fn rounding_div_ceil(value: i32, div: i32) -> i32 {
    (value + div - 1) / div
}

struct Tetrahedral<'a, const GRID_SIZE: usize> {
    cube: &'a [f32],
}

impl<'a, const GRID_SIZE: usize> Tetrahedral<'a, GRID_SIZE> {
    pub fn new(table: &'a [f32]) -> Self {
        Self { cube: table }
    }

    #[inline]
    fn lp(&self, tab: &[f32], x: i32, y: i32, z: i32) -> Vector3f {
        let offset = (x as u32 * (GRID_SIZE as u32 * GRID_SIZE as u32)
            + y as u32 * GRID_SIZE as u32
            + z as u32) as usize
            * 3;
        let jx = &tab[offset..offset + 3];
        Vector3f {
            v: [jx[0], jx[1], jx[2]],
        }
    }

    fn interpolate(&self, in_r: u8, in_g: u8, in_b: u8) -> Vector3f {
        let linear_r: f32 = in_r as i32 as f32 / 255.0;
        let linear_g: f32 = in_g as i32 as f32 / 255.0;
        let linear_b: f32 = in_b as i32 as f32 / 255.0;
        let x: i32 = in_r as i32 * (GRID_SIZE as i32 - 1) / 255;
        let y: i32 = in_g as i32 * (GRID_SIZE as i32 - 1) / 255;
        let z: i32 = in_b as i32 * (GRID_SIZE as i32 - 1) / 255;
        let x_n: i32 = rounding_div_ceil(in_r as i32 * (GRID_SIZE as i32 - 1), 255);
        let y_n: i32 = rounding_div_ceil(in_g as i32 * (GRID_SIZE as i32 - 1), 255);
        let z_n: i32 = rounding_div_ceil(in_b as i32 * (GRID_SIZE as i32 - 1), 255);
        let rx: f32 = linear_r * (GRID_SIZE as i32 - 1) as f32 - x as f32;
        let ry: f32 = linear_g * (GRID_SIZE as i32 - 1) as f32 - y as f32;
        let rz: f32 = linear_b * (GRID_SIZE as i32 - 1) as f32 - z as f32;
        let c0 = self.lp(self.cube, x, y, z);
        let c2;
        let c1;
        let c3;
        if rx >= ry {
            if ry >= rz {
                //rx >= ry && ry >= rz
                c1 = self.lp(self.cube, x_n, y, z) - c0;
                c2 = self.lp(self.cube, x_n, y_n, z) - self.lp(self.cube, x_n, y, z);
                c3 = self.lp(self.cube, x_n, y_n, z_n) - self.lp(self.cube, x_n, y_n, z);
            } else if rx >= rz {
                //rx >= rz && rz >= ry
                c1 = self.lp(self.cube, x_n, y, z) - c0;
                c2 = self.lp(self.cube, x_n, y_n, z_n) - self.lp(self.cube, x_n, y, z_n);
                c3 = self.lp(self.cube, x_n, y, z_n) - self.lp(self.cube, x_n, y, z);
            } else {
                //rz > rx && rx >= ry
                c1 = self.lp(self.cube, x_n, y, z_n) - self.lp(self.cube, x, y, z_n);
                c2 = self.lp(self.cube, x_n, y_n, z_n) - self.lp(self.cube, x_n, y, z_n);
                c3 = self.lp(self.cube, x, y, z_n) - c0;
            }
        } else if rx >= rz {
            //ry > rx && rx >= rz
            c1 = self.lp(self.cube, x_n, y_n, z) - self.lp(self.cube, x, y_n, z);
            c2 = self.lp(self.cube, x, y_n, z) - c0;
            c3 = self.lp(self.cube, x_n, y_n, z_n) - self.lp(self.cube, x_n, y_n, z);
        } else if ry >= rz {
            //ry >= rz && rz > rx
            c1 = self.lp(self.cube, x_n, y_n, z_n) - self.lp(self.cube, x, y_n, z_n);
            c2 = self.lp(self.cube, x, y_n, z) - c0;
            c3 = self.lp(self.cube, x, y_n, z_n) - self.lp(self.cube, x, y_n, z);
        } else {
            //rz > ry && ry > rx
            c1 = self.lp(self.cube, x_n, y_n, z_n) - self.lp(self.cube, x, y_n, z_n);
            c2 = self.lp(self.cube, x, y_n, z_n) - self.lp(self.cube, x, y, z_n);
            c3 = self.lp(self.cube, x, y, z_n) - c0;
        }
        c0 + c1 * rx + c2 * ry + c3 * rz
    }
}

impl<const LAYOUT: u8, const GRID_SIZE: usize> TransformLut4XyzToRgb<LAYOUT, GRID_SIZE> {
    #[inline(always)]
    fn transform_chunk(&self, src: &[u8], dst: &mut [u8]) {
        let cn = Layout::from(LAYOUT);
        let channels = cn.channels();
        let grid_size = GRID_SIZE as i32;
        let grid_size3 = grid_size * grid_size * grid_size;
        for (src, dst) in src.chunks_exact(4).zip(dst.chunks_exact_mut(channels)) {
            let c = src[0];
            let m = src[1];
            let y = src[2];
            let k = src[3];
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
            let r = lerp(r1, r2, Vector3f::from(t)) * 255.0f32 + 0.5f32;
            dst[cn.r_i()] = r.v[0] as u8;
            dst[cn.g_i()] = r.v[1] as u8;
            dst[cn.b_i()] = r.v[2] as u8;
            if channels == 4 {
                dst[cn.a_i()] = 255;
            }
        }
    }
}

impl<const LAYOUT: u8, const GRID_SIZE: usize> TransformExecutor<u8>
    for TransformLut4XyzToRgb<LAYOUT, GRID_SIZE>
{
    fn transform(&self, src: &[u8], dst: &mut [u8]) -> Result<(), CmsError> {
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

pub(crate) fn create_cmyk_to_rgb(
    source: &ColorProfile,
    dest: &ColorProfile,
    layout: Layout,
) -> Result<Box<Transform8BitExecutor>, CmsError> {
    if source.color_space != DataColorSpace::Cmyk && dest.color_space != DataColorSpace::Rgb {
        return Err(CmsError::UnsupportedProfileConnection);
    }
    if source.pcs != DataColorSpace::Xyz && source.pcs != DataColorSpace::Lab {
        return Err(CmsError::UnsupportedProfileConnection);
    }

    let lut_a_to_b = match &source.lut_a_to_b {
        None => return Err(CmsError::UnsupportedProfileConnection),
        Some(v) => v,
    };

    let lut: Option<Box<Transform8BitExecutor>> = None;

    if source.color_space == DataColorSpace::Cmyk && dest.color_space == DataColorSpace::Rgb {
        if layout != Layout::Rgb8 && layout != Layout::Rgba8 {
            return Err(CmsError::UnsupportedProfileConnection);
        }

        const GRID_SIZE: usize = 17;

        let mut lut = create_lut::<GRID_SIZE>(lut_a_to_b)?;

        if source.pcs == DataColorSpace::Lab {
            let mut working_buffer = vec![0f32; lut.len()];
            let lab_to_xyz_stage = StageLabToXyz::default();
            lab_to_xyz_stage.transform(&lut, &mut working_buffer)?;
            std::mem::swap(&mut working_buffer, &mut lut);
        }

        if dest.pcs == DataColorSpace::Lab {
            let mut working_buffer = vec![0f32; lut.len()];
            let lab_to_xyz_stage = StageXyzToLab::default();
            lab_to_xyz_stage.transform(&lut, &mut working_buffer)?;
            std::mem::swap(&mut working_buffer, &mut lut);
        }

        if dest.color_space == DataColorSpace::Rgb {
            let gamma_map_r: Box<[u8; 65536]> = dest.build_8bit_gamma_table(&dest.red_trc)?;
            let gamma_map_g: Box<[u8; 65536]> = dest.build_8bit_gamma_table(&dest.green_trc)?;
            let gamma_map_b: Box<[u8; 65536]> = dest.build_8bit_gamma_table(&dest.blue_trc)?;
            let xyz_to_rgb_opt = match dest.rgb_to_xyz_matrix() {
                None => return Err(CmsError::UnsupportedProfileConnection),
                Some(v) => v.inverse(),
            };
            let xyz_to_rgb = match xyz_to_rgb_opt {
                None => return Err(CmsError::UnsupportedProfileConnection),
                Some(v) => v,
            };
            let mut matrices = vec![Matrix3f {
                v: [
                    [65535.0 / 32768.0, 0.0, 0.0],
                    [0.0, 65535.0 / 32768.0, 0.0],
                    [0.0, 0.0, 65535.0 / 32768.0],
                ],
            }];

            matrices.push(xyz_to_rgb);
            let xyz_to_rgb_stage = XyzToRgbStage {
                r_gamma: gamma_map_r,
                g_gamma: gamma_map_g,
                b_gamma: gamma_map_b,
                matrices,
            };
            let mut working_buffer = vec![0f32; lut.len()];
            xyz_to_rgb_stage.transform(&lut, &mut working_buffer)?;
            std::mem::swap(&mut working_buffer, &mut lut);
        }

        return Ok(match layout {
            Layout::Rgb8 => {
                Box::new(TransformLut4XyzToRgb::<{ Layout::Rgb8 as u8 }, GRID_SIZE> { lut })
            }
            Layout::Rgba8 => {
                Box::new(TransformLut4XyzToRgb::<{ Layout::Rgba8 as u8 }, GRID_SIZE> { lut })
            }
            _ => unimplemented!(),
        });
    }

    lut.map(Ok)
        .unwrap_or(Err(CmsError::UnsupportedProfileConnection))
}
