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
use crate::conversions::mab::{BCurves3, MCurves3};
use crate::{
    Array4D, CmsError, DataColorSpace, InPlaceStage, InterpolationMethod, LutMCurvesType, Stage,
    TransformOptions, Vector3f,
};

struct ACurves4x3<'a, const DEPTH: usize, const GRID_SIZE: usize> {
    curve0: Box<[f32; 65536]>,
    curve1: Box<[f32; 65536]>,
    curve2: Box<[f32; 65536]>,
    curve3: Box<[f32; 65536]>,
    clut: &'a [f32],
    grid_size: [u8; 4],
    interpolation_method: InterpolationMethod,
    pcs: DataColorSpace,
}

impl<const DEPTH: usize, const GRID_SIZE: usize> ACurves4x3<'_, DEPTH, GRID_SIZE> {
    fn transform_impl<Fetch: Fn(f32, f32, f32, f32) -> Vector3f>(
        &self,
        src: &[f32],
        dst: &mut [f32],
        fetch: Fetch,
    ) -> Result<(), CmsError> {
        let scale_value = (DEPTH - 1) as f32;

        assert_eq!(src.len() / 4, dst.len() / 3);

        for (src, dst) in src.chunks_exact(4).zip(dst.chunks_exact_mut(3)) {
            let a0 = (src[0] * scale_value).round().min(scale_value) as u16;
            let a1 = (src[1] * scale_value).round().min(scale_value) as u16;
            let a2 = (src[2] * scale_value).round().min(scale_value) as u16;
            let a3 = (src[3] * scale_value).round().min(scale_value) as u16;
            let c = self.curve0[a0 as usize];
            let m = self.curve1[a1 as usize];
            let y = self.curve2[a2 as usize];
            let k = self.curve3[a3 as usize];

            let r = fetch(c, m, y, k);
            dst[0] = r.v[0];
            dst[1] = r.v[1];
            dst[2] = r.v[2];
        }
        Ok(())
    }
}

impl<const DEPTH: usize, const GRID_SIZE: usize> Stage for ACurves4x3<'_, DEPTH, GRID_SIZE> {
    fn transform(&self, src: &[f32], dst: &mut [f32]) -> Result<(), CmsError> {
        let lut = Array4D::new_hypercube(self.clut, self.grid_size);

        // If PCS is LAB then linear interpolation should be used
        if self.pcs == DataColorSpace::Lab {
            return self.transform_impl(src, dst, |x, y, z, w| lut.quadlinear_vec3(x, y, z, w));
        }

        match self.interpolation_method {
            #[cfg(feature = "options")]
            InterpolationMethod::Tetrahedral => {
                self.transform_impl(src, dst, |x, y, z, w| lut.tetra_vec3(x, y, z, w))?;
            }
            #[cfg(feature = "options")]
            InterpolationMethod::Pyramid => {
                self.transform_impl(src, dst, |x, y, z, w| lut.pyramid_vec3(x, y, z, w))?;
            }
            #[cfg(feature = "options")]
            InterpolationMethod::Prism => {
                self.transform_impl(src, dst, |x, y, z, w| lut.prism_vec3(x, y, z, w))?;
            }
            InterpolationMethod::Linear => {
                self.transform_impl(src, dst, |x, y, z, w| lut.quadlinear_vec3(x, y, z, w))?;
            }
        }
        Ok(())
    }
}

pub(crate) fn prepare_mab_4x3<const GRID_SIZE: usize>(
    mab: &LutMCurvesType,
    lut: &mut [f32],
    options: TransformOptions,
    pcs: DataColorSpace,
) -> Result<Vec<f32>, CmsError> {
    const LERP_DEPTH: usize = 65536;
    const BP: usize = 13;
    const DEPTH: usize = 8192;
    if mab.num_input_channels != 4 && mab.num_output_channels != 3 {
        return Err(CmsError::UnsupportedProfileConnection);
    }
    let mut new_lut = vec![0f32; (lut.len() / 4) * 3];
    if mab.a_curves.len() == 4 && mab.clut.is_some() {
        let curve0 = mab.a_curves[0]
            .build_linearize_table::<u16, LERP_DEPTH, BP>()
            .ok_or(CmsError::InvalidTrcCurve)?;
        let curve1 = mab.a_curves[1]
            .build_linearize_table::<u16, LERP_DEPTH, BP>()
            .ok_or(CmsError::InvalidTrcCurve)?;
        let curve2 = mab.a_curves[2]
            .build_linearize_table::<u16, LERP_DEPTH, BP>()
            .ok_or(CmsError::InvalidTrcCurve)?;
        let curve3 = mab.a_curves[3]
            .build_linearize_table::<u16, LERP_DEPTH, BP>()
            .ok_or(CmsError::InvalidTrcCurve)?;
        let clut = &mab.clut.as_ref().map(|x| x.to_clut_f32()).unwrap();
        let a_curves = ACurves4x3::<DEPTH, GRID_SIZE> {
            curve0,
            curve1,
            curve2,
            curve3,
            clut,
            grid_size: [
                mab.grid_points[0],
                mab.grid_points[1],
                mab.grid_points[2],
                mab.grid_points[3],
            ],
            interpolation_method: options.interpolation_method,
            pcs,
        };
        a_curves.transform(lut, &mut new_lut)?;
    } else {
        // Not supported
        return Err(CmsError::UnsupportedProfileConnection);
    }

    if mab.m_curves.len() == 3 {
        let curve0 = mab.m_curves[0]
            .build_linearize_table::<u16, LERP_DEPTH, BP>()
            .ok_or(CmsError::InvalidTrcCurve)?;
        let curve1 = mab.m_curves[1]
            .build_linearize_table::<u16, LERP_DEPTH, BP>()
            .ok_or(CmsError::InvalidTrcCurve)?;
        let curve2 = mab.m_curves[2]
            .build_linearize_table::<u16, LERP_DEPTH, BP>()
            .ok_or(CmsError::InvalidTrcCurve)?;
        let matrix = mab.matrix;
        let bias = mab.bias;
        let m_curves = MCurves3::<DEPTH> {
            curve0,
            curve1,
            curve2,
            matrix,
            bias,
            inverse: false,
        };
        m_curves.transform(&mut new_lut)?;
    }

    if mab.b_curves.len() == 3 {
        let curve0 = mab.b_curves[0]
            .build_linearize_table::<u16, LERP_DEPTH, BP>()
            .ok_or(CmsError::InvalidTrcCurve)?;
        let curve1 = mab.b_curves[1]
            .build_linearize_table::<u16, LERP_DEPTH, BP>()
            .ok_or(CmsError::InvalidTrcCurve)?;
        let curve2 = mab.b_curves[2]
            .build_linearize_table::<u16, LERP_DEPTH, BP>()
            .ok_or(CmsError::InvalidTrcCurve)?;
        let b_curves = BCurves3::<DEPTH> {
            curve0,
            curve1,
            curve2,
        };
        b_curves.transform(&mut new_lut)?;
    } else {
        return Err(CmsError::InvalidAtoBLut);
    }

    Ok(new_lut)
}
