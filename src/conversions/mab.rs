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
use crate::mlaf::mlaf;
use crate::{Array3D, CmsError, InPlaceStage, LutMCurvesType, Matrix3f, Vector3f};

struct ACurves3<'a, const DEPTH: usize> {
    curve0: Box<[f32; DEPTH]>,
    curve1: Box<[f32; DEPTH]>,
    curve2: Box<[f32; DEPTH]>,
    clut: &'a [f32],
    grid_size: usize,
}

impl<const DEPTH: usize> InPlaceStage for ACurves3<'_, DEPTH> {
    fn transform(&self, dst: &mut [f32]) -> Result<(), CmsError> {
        let scale_value = (DEPTH - 1) as f32;

        let lut = Array3D::new(self.clut, self.grid_size);

        for dst in dst.chunks_exact_mut(3) {
            let a0 = (dst[0] * scale_value).min(scale_value) as u8;
            let a1 = (dst[1] * scale_value).min(scale_value) as u8;
            let a2 = (dst[2] * scale_value).min(scale_value) as u8;
            let b0 = self.curve0[a0 as usize];
            let b1 = self.curve1[a1 as usize];
            let b2 = self.curve2[a2 as usize];
            let interpolated = lut.trilinear_vec3(b0, b1, b2);
            dst[0] = interpolated.v[0];
            dst[1] = interpolated.v[1];
            dst[2] = interpolated.v[2];
        }
        Ok(())
    }
}

struct ACurves3Inverse<'a, const DEPTH: usize> {
    curve0: Box<[f32; DEPTH]>,
    curve1: Box<[f32; DEPTH]>,
    curve2: Box<[f32; DEPTH]>,
    clut: &'a [f32],
    grid_size: usize,
}

impl<const DEPTH: usize> InPlaceStage for ACurves3Inverse<'_, DEPTH> {
    fn transform(&self, dst: &mut [f32]) -> Result<(), CmsError> {
        let scale_value = (DEPTH as u32 - 1u32) as f32;

        let lut = Array3D::new(self.clut, self.grid_size);

        for dst in dst.chunks_exact_mut(3) {
            let interpolated = lut.trilinear_vec3(dst[0], dst[1], dst[2]);
            let a0 = (interpolated.v[0] * scale_value).min(scale_value) as u8;
            let a1 = (interpolated.v[1] * scale_value).min(scale_value) as u8;
            let a2 = (interpolated.v[2] * scale_value).min(scale_value) as u8;
            let b0 = self.curve0[a0 as usize];
            let b1 = self.curve1[a1 as usize];
            let b2 = self.curve2[a2 as usize];
            dst[0] = b0;
            dst[1] = b1;
            dst[2] = b2;
        }
        Ok(())
    }
}

struct MCurves3<const DEPTH: usize> {
    curve0: Box<[f32; DEPTH]>,
    curve1: Box<[f32; DEPTH]>,
    curve2: Box<[f32; DEPTH]>,
    matrix: Matrix3f,
    bias: Vector3f,
    inverse: bool,
}

impl<const DEPTH: usize> InPlaceStage for MCurves3<DEPTH> {
    fn transform(&self, dst: &mut [f32]) -> Result<(), CmsError> {
        let scale_value = (DEPTH - 1) as f32;

        if self.inverse {
            let m = self.matrix;
            let b = self.bias;

            if !m.test_equality(Matrix3f::IDENTITY)
                || !b.eq(&Vector3f {
                    v: [0f32, 0f32, 0f32],
                })
            {
                for dst in dst.chunks_exact_mut(3) {
                    let x = dst[0];
                    let y = dst[1];
                    let z = dst[2];
                    dst[0] = mlaf(mlaf(mlaf(b.v[0], x, m.v[0][0]), y, m.v[0][1]), z, m.v[0][2]);
                    dst[1] = mlaf(mlaf(mlaf(b.v[1], x, m.v[1][0]), y, m.v[1][1]), z, m.v[1][2]);
                    dst[2] = mlaf(mlaf(mlaf(b.v[2], x, m.v[2][0]), y, m.v[2][1]), z, m.v[2][2]);
                }
            }
        }

        for dst in dst.chunks_exact_mut(3) {
            let a0 = (dst[0] * scale_value).min(scale_value) as u8;
            let a1 = (dst[1] * scale_value).min(scale_value) as u8;
            let a2 = (dst[2] * scale_value).min(scale_value) as u8;
            let b0 = self.curve0[a0 as usize];
            let b1 = self.curve1[a1 as usize];
            let b2 = self.curve2[a2 as usize];
            dst[0] = b0;
            dst[1] = b1;
            dst[2] = b2;
        }

        if !self.inverse {
            let m = self.matrix;
            let b = self.bias;

            if !m.test_equality(Matrix3f::IDENTITY)
                || !b.eq(&Vector3f {
                    v: [0f32, 0f32, 0f32],
                })
            {
                for dst in dst.chunks_exact_mut(3) {
                    let x = dst[0];
                    let y = dst[1];
                    let z = dst[2];
                    dst[0] = mlaf(mlaf(mlaf(b.v[0], x, m.v[0][0]), y, m.v[0][1]), z, m.v[0][2]);
                    dst[1] = mlaf(mlaf(mlaf(b.v[1], x, m.v[1][0]), y, m.v[1][1]), z, m.v[1][2]);
                    dst[2] = mlaf(mlaf(mlaf(b.v[2], x, m.v[2][0]), y, m.v[2][1]), z, m.v[2][2]);
                }
            }
        }

        Ok(())
    }
}

struct BCurves<const DEPTH: usize> {
    curve0: Box<[f32; DEPTH]>,
    curve1: Box<[f32; DEPTH]>,
    curve2: Box<[f32; DEPTH]>,
}

impl<const DEPTH: usize> InPlaceStage for BCurves<DEPTH> {
    fn transform(&self, dst: &mut [f32]) -> Result<(), CmsError> {
        let scale_value = (DEPTH - 1) as f32;

        for dst in dst.chunks_exact_mut(3) {
            let a0 = (dst[0] * scale_value).min(scale_value) as u8;
            let a1 = (dst[1] * scale_value).min(scale_value) as u8;
            let a2 = (dst[2] * scale_value).min(scale_value) as u8;
            let b0 = self.curve0[a0 as usize];
            let b1 = self.curve1[a1 as usize];
            let b2 = self.curve2[a2 as usize];
            dst[0] = b0;
            dst[1] = b1;
            dst[2] = b2;
        }

        Ok(())
    }
}

pub(crate) fn prepare_mab_3x3(mab: &LutMCurvesType, lut: &mut [f32]) -> Result<(), CmsError> {
    const LERP_DEPTH: usize = 256;
    const BP: usize = 8;
    if mab.num_input_channels != 3 && mab.num_output_channels != 3 {
        return Err(CmsError::UnsupportedProfileConnection);
    }
    if mab.grid_points[0] != mab.grid_points[1] || mab.grid_points[1] != mab.grid_points[2] {
        return Err(CmsError::UnsupportedProfileConnection);
    }
    if mab.a_curves.len() == 3 {
        let curve0 = mab.a_curves[0]
            .build_linearize_table::<LERP_DEPTH, BP>()
            .ok_or(CmsError::InvalicTrcCurve)?;
        let curve1 = mab.a_curves[1]
            .build_linearize_table::<LERP_DEPTH, BP>()
            .ok_or(CmsError::InvalicTrcCurve)?;
        let curve2 = mab.a_curves[2]
            .build_linearize_table::<LERP_DEPTH, BP>()
            .ok_or(CmsError::InvalicTrcCurve)?;
        let clut = &mab.clut;
        let a_curves = ACurves3::<LERP_DEPTH> {
            curve0,
            curve1,
            curve2,
            clut,
            grid_size: mab.grid_points[0] as usize,
        };
        a_curves.transform(lut)?;
    }

    if mab.m_curves.len() == 3 {
        let curve0 = mab.m_curves[0]
            .build_linearize_table::<LERP_DEPTH, BP>()
            .ok_or(CmsError::InvalicTrcCurve)?;
        let curve1 = mab.m_curves[1]
            .build_linearize_table::<LERP_DEPTH, BP>()
            .ok_or(CmsError::InvalicTrcCurve)?;
        let curve2 = mab.m_curves[2]
            .build_linearize_table::<LERP_DEPTH, BP>()
            .ok_or(CmsError::InvalicTrcCurve)?;
        let matrix = mab.matrix;
        let bias = mab.bias;
        let m_curves = MCurves3::<LERP_DEPTH> {
            curve0,
            curve1,
            curve2,
            matrix,
            bias,
            inverse: false,
        };
        m_curves.transform(lut)?;
    }

    if mab.b_curves.len() == 3 {
        let curve0 = mab.b_curves[0]
            .build_linearize_table::<LERP_DEPTH, BP>()
            .ok_or(CmsError::InvalicTrcCurve)?;
        let curve1 = mab.b_curves[1]
            .build_linearize_table::<LERP_DEPTH, BP>()
            .ok_or(CmsError::InvalicTrcCurve)?;
        let curve2 = mab.b_curves[2]
            .build_linearize_table::<LERP_DEPTH, BP>()
            .ok_or(CmsError::InvalicTrcCurve)?;
        let b_curves = BCurves::<LERP_DEPTH> {
            curve0,
            curve1,
            curve2,
        };
        b_curves.transform(lut)?;
    } else {
        return Err(CmsError::InvalidAtoBLut);
    }

    Ok(())
}

pub(crate) fn prepare_mba_3x3(mab: &LutMCurvesType, lut: &mut [f32]) -> Result<(), CmsError> {
    if mab.num_input_channels != 3 && mab.num_output_channels != 3 {
        return Err(CmsError::UnsupportedProfileConnection);
    }
    if mab.grid_points[0] != mab.grid_points[1] || mab.grid_points[1] != mab.grid_points[2] {
        return Err(CmsError::UnsupportedProfileConnection);
    }

    const LERP_DEPTH: usize = 256;
    const BP: usize = 8;

    if mab.b_curves.len() == 3 {
        let curve0 = mab.b_curves[0]
            .build_linearize_table::<LERP_DEPTH, BP>()
            .ok_or(CmsError::InvalicTrcCurve)?;
        let curve1 = mab.b_curves[1]
            .build_linearize_table::<LERP_DEPTH, BP>()
            .ok_or(CmsError::InvalicTrcCurve)?;
        let curve2 = mab.b_curves[2]
            .build_linearize_table::<LERP_DEPTH, BP>()
            .ok_or(CmsError::InvalicTrcCurve)?;
        let b_curves = BCurves::<LERP_DEPTH> {
            curve0,
            curve1,
            curve2,
        };
        b_curves.transform(lut)?;
    } else {
        return Err(CmsError::InvalidAtoBLut);
    }

    if mab.m_curves.len() == 3 {
        let curve0 = mab.m_curves[0]
            .build_linearize_table::<LERP_DEPTH, BP>()
            .ok_or(CmsError::InvalicTrcCurve)?;
        let curve1 = mab.m_curves[1]
            .build_linearize_table::<LERP_DEPTH, BP>()
            .ok_or(CmsError::InvalicTrcCurve)?;
        let curve2 = mab.m_curves[2]
            .build_linearize_table::<LERP_DEPTH, BP>()
            .ok_or(CmsError::InvalicTrcCurve)?;
        let matrix = mab.matrix;
        let bias = mab.bias;
        let m_curves = MCurves3::<LERP_DEPTH> {
            curve0,
            curve1,
            curve2,
            matrix,
            bias,
            inverse: true,
        };
        m_curves.transform(lut)?;
    }

    if mab.a_curves.len() == 3 {
        let curve0 = mab.a_curves[0]
            .build_linearize_table::<LERP_DEPTH, BP>()
            .ok_or(CmsError::InvalicTrcCurve)?;
        let curve1 = mab.a_curves[1]
            .build_linearize_table::<LERP_DEPTH, BP>()
            .ok_or(CmsError::InvalicTrcCurve)?;
        let curve2 = mab.a_curves[2]
            .build_linearize_table::<LERP_DEPTH, BP>()
            .ok_or(CmsError::InvalicTrcCurve)?;
        let clut = &mab.clut;
        let a_curves = ACurves3Inverse::<LERP_DEPTH> {
            curve0,
            curve1,
            curve2,
            clut,
            grid_size: mab.grid_points[0] as usize,
        };
        a_curves.transform(lut)?;
    }

    Ok(())
}
