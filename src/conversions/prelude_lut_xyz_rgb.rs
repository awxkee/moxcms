/*
 * // Copyright (c) Radzivon Bartoshyk 4/2025. All rights reserved.
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
use crate::trc::ExtendedGammaEvaluator;
use crate::{CmsError, InPlaceStage, Matrix3f, Rgb, filmlike_clip};
use num_traits::AsPrimitive;
use std::marker::PhantomData;

pub(crate) struct XyzToRgbStage<T: Clone, const BIT_DEPTH: usize, const GAMMA_LUT: usize> {
    pub(crate) r_gamma: Box<[T; 65536]>,
    pub(crate) g_gamma: Box<[T; 65536]>,
    pub(crate) b_gamma: Box<[T; 65536]>,
    pub(crate) matrices: Vec<Matrix3f>,
}

impl<T: Clone + AsPrimitive<f32>, const BIT_DEPTH: usize, const GAMMA_LUT: usize> InPlaceStage
    for XyzToRgbStage<T, BIT_DEPTH, GAMMA_LUT>
{
    fn transform(&self, dst: &mut [f32]) -> Result<(), CmsError> {
        assert!(BIT_DEPTH > 0);
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
            let mut rgb = Rgb::new(dst[0], dst[1], dst[2]);
            if rgb.is_out_of_gamut() {
                rgb = filmlike_clip(rgb);
            }
            let r = mlaf(0.5f32, rgb.r, lut_cap).min(lut_cap).max(0f32) as u16;
            let g = mlaf(0.5f32, rgb.g, lut_cap).min(lut_cap).max(0f32) as u16;
            let b = mlaf(0.5f32, rgb.b, lut_cap).min(lut_cap).max(0f32) as u16;

            dst[0] = self.r_gamma[r as usize].as_() * color_scale;
            dst[1] = self.g_gamma[g as usize].as_() * color_scale;
            dst[2] = self.b_gamma[b as usize].as_() * color_scale;
        }

        Ok(())
    }
}

pub(crate) struct XyzToRgbStageExtended<T: Clone> {
    pub(crate) gamma_evaluator: Box<dyn ExtendedGammaEvaluator>,
    pub(crate) matrices: Vec<Matrix3f>,
    pub(crate) phantom_data: PhantomData<T>,
}

impl<T: Clone + AsPrimitive<f32>> InPlaceStage for XyzToRgbStageExtended<T> {
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

        for dst in dst.chunks_exact_mut(3) {
            let mut rgb = Rgb::new(dst[0], dst[1], dst[2]);
            rgb = self.gamma_evaluator.evaluate(rgb);
            dst[0] = rgb.r.as_();
            dst[1] = rgb.g.as_();
            dst[2] = rgb.b.as_();
        }

        Ok(())
    }
}
