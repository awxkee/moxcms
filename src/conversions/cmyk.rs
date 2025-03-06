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
use crate::conversions::lut3::{create_lut3_samples, create_lut3x4};
use crate::conversions::lut3_to_4::TransformLut3x4;
use crate::conversions::lut4::create_lut4;
use crate::conversions::tetrahedral::Tetrahedral;
use crate::lab::Lab;
use crate::mlaf::mlaf;
use crate::nd_array::lerp;
use crate::{
    CmsError, ColorProfile, DataColorSpace, InPlaceStage, Layout, Matrix3f, TransformExecutor,
    TransformOptions, Vector3f, Xyz, rounding_div_ceil,
};
use num_traits::AsPrimitive;
use std::marker::PhantomData;

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

struct MatrixStage {
    matrices: Vec<Matrix3f>,
}

impl InPlaceStage for MatrixStage {
    fn transform(&self, dst: &mut [f32]) -> Result<(), CmsError> {
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
            let r1 = tetrahedral1.inter3(c, m, y);
            let r2 = tetrahedral2.inter3(c, m, y);
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

struct RgbLinearizationStage<
    T: Clone,
    const BIT_DEPTH: usize,
    const LINEAR_CAP: usize,
    const SAMPLES: usize,
> {
    r_lin: Box<[f32; LINEAR_CAP]>,
    g_lin: Box<[f32; LINEAR_CAP]>,
    b_lin: Box<[f32; LINEAR_CAP]>,
    _phantom: PhantomData<T>,
}

impl<
    T: Clone + AsPrimitive<usize>,
    const BIT_DEPTH: usize,
    const LINEAR_CAP: usize,
    const SAMPLES: usize,
> RgbLinearizationStage<T, BIT_DEPTH, LINEAR_CAP, SAMPLES>
{
    fn transform(&self, src: &[T], dst: &mut [f32]) -> Result<(), CmsError> {
        if src.len() % 3 != 0 {
            return Err(CmsError::LaneMultipleOfChannels);
        }
        if dst.len() % 3 != 0 {
            return Err(CmsError::LaneMultipleOfChannels);
        }

        let scale = ((1 << BIT_DEPTH) - 1) as f32 / SAMPLES as f32;

        for (src, dst) in src.chunks_exact(3).zip(dst.chunks_exact_mut(3)) {
            let j_r = src[0].as_() as f32 * scale;
            let j_g = src[1].as_() as f32 * scale;
            let j_b = src[2].as_() as f32 * scale;
            dst[0] = self.r_lin[(j_r as u16) as usize];
            dst[1] = self.g_lin[(j_g as u16) as usize];
            dst[2] = self.b_lin[(j_b as u16) as usize];
        }
        Ok(())
    }
}

pub(crate) fn make_cmyk_luts<
    T: Copy + Default + AsPrimitive<f32> + Send + Sync + CompressCmykLut + AsPrimitive<usize>,
    const BIT_DEPTH: usize,
    const LINEAR_CAP: usize,
    const GAMMA_LUT: usize,
>(
    src_layout: Layout,
    source: &ColorProfile,
    dst_layout: Layout,
    dest: &ColorProfile,
    options: TransformOptions,
) -> Result<Box<dyn TransformExecutor<T> + Send + Sync>, CmsError>
where
    f32: AsPrimitive<T>,
    u32: AsPrimitive<T>,
{
    if source.color_space == DataColorSpace::Cmyk && dest.color_space == DataColorSpace::Rgb {
        if src_layout != Layout::Rgba {
            return Err(CmsError::InvalidLayout);
        }
        if source.pcs != DataColorSpace::Xyz && source.pcs != DataColorSpace::Lab {
            return Err(CmsError::UnsupportedProfileConnection);
        }

        let lut_a_to_b = source
            .get_device_to_pcs_lut(options.rendering_intent)
            .ok_or(CmsError::UnsupportedLutRenderingIntent(
                source.rendering_intent,
            ))?;

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
    } else if source.color_space == DataColorSpace::Rgb && dest.color_space == DataColorSpace::Cmyk
    {
        if src_layout != Layout::Rgba && src_layout != Layout::Rgb {
            return Err(CmsError::InvalidLayout);
        }
        if dst_layout != Layout::Rgba {
            return Err(CmsError::InvalidLayout);
        }
        if source.pcs != DataColorSpace::Xyz && source.pcs != DataColorSpace::Lab {
            return Err(CmsError::UnsupportedProfileConnection);
        }

        let lut_b_to_a = dest.get_pcs_to_device_lut(options.rendering_intent).ok_or(
            CmsError::UnsupportedLutRenderingIntent(source.rendering_intent),
        )?;

        const GRID_SIZE: usize = 33;

        let lut_origins = create_lut3_samples::<T, GRID_SIZE>();

        let lin_r = source.build_r_linearize_table::<LINEAR_CAP>()?;
        let lin_g = source.build_g_linearize_table::<LINEAR_CAP>()?;
        let lin_b = source.build_b_linearize_table::<LINEAR_CAP>()?;

        let lin_stage = RgbLinearizationStage::<T, BIT_DEPTH, LINEAR_CAP, GRID_SIZE> {
            r_lin: lin_r,
            g_lin: lin_g,
            b_lin: lin_b,
            _phantom: PhantomData,
        };

        let mut lut = vec![0f32; lut_origins.len()];
        lin_stage.transform(&lut_origins, &mut lut)?;

        let xyz_to_rgb = source
            .rgb_to_xyz_matrix()
            .ok_or(CmsError::UnsupportedProfileConnection)?;

        let matrices = vec![
            xyz_to_rgb,
            Matrix3f {
                v: [
                    [32768.0 / 65535.0, 0.0, 0.0],
                    [0.0, 32768.0 / 65535.0, 0.0],
                    [0.0, 0.0, 32768.0 / 65535.0],
                ],
            },
        ];

        let matrix_stage = MatrixStage { matrices };
        matrix_stage.transform(&mut lut)?;

        if dest.pcs == DataColorSpace::Lab {
            let xyz_to_lab = StageXyzToLab::default();
            xyz_to_lab.transform(&mut lut)?;
        }

        let lut = create_lut3x4::<GRID_SIZE>(lut_b_to_a, &lut)?;

        return Ok(match src_layout {
            Layout::Rgb => {
                Box::new(
                    TransformLut3x4::<T, { Layout::Rgb as u8 }, GRID_SIZE, BIT_DEPTH> {
                        lut,
                        _phantom: PhantomData,
                    },
                )
            }
            Layout::Rgba => {
                Box::new(
                    TransformLut3x4::<T, { Layout::Rgba as u8 }, GRID_SIZE, BIT_DEPTH> {
                        lut,
                        _phantom: PhantomData,
                    },
                )
            }
            _ => unimplemented!(),
        });
    }

    Err(CmsError::UnsupportedProfileConnection)
}
