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
use crate::mlaf::mlaf;
use crate::{
    CmsError, InPlaceStage, Layout, Matrix3f, Rgb, gamut_clip_adaptive_l0_0_5,
    gamut_clip_preserve_chroma,
};
use std::ops::Mul;

pub(crate) struct MatrixClipScaleStage<const LAYOUT: u8> {
    pub(crate) matrix: Matrix3f,
    pub(crate) scale: f32,
}

impl<const LAYOUT: u8> InPlaceStage for MatrixClipScaleStage<LAYOUT> {
    #[inline]
    fn transform(&self, dst: &mut [f32]) -> Result<(), CmsError> {
        let scale = self.scale;
        let cn = Layout::from(LAYOUT);
        let channels = cn.channels();

        let transform = self.matrix;

        for chunk in dst.chunks_exact_mut(channels) {
            let r = chunk[0];
            let g = chunk[1];
            let b = chunk[2];

            chunk[0] = mlaf(
                mlaf(r * transform.v[0][0], g, transform.v[0][1]),
                b,
                transform.v[0][2],
            )
            .max(0f32)
            .min(1f32)
            .mul(scale)
            .round();

            chunk[1] = mlaf(
                mlaf(r * transform.v[1][0], g, transform.v[1][1]),
                b,
                transform.v[1][2],
            )
            .max(0f32)
            .min(1f32)
            .mul(scale)
            .round();

            chunk[2] = mlaf(
                mlaf(r * transform.v[2][0], g, transform.v[2][1]),
                b,
                transform.v[2][2],
            )
            .max(0f32)
            .min(1f32)
            .mul(scale)
            .round();
        }

        Ok(())
    }
}

pub(crate) struct MatrixStage<const LAYOUT: u8> {
    pub(crate) matrix: Matrix3f,
}

impl<const LAYOUT: u8> InPlaceStage for MatrixStage<LAYOUT> {
    #[inline]
    fn transform(&self, dst: &mut [f32]) -> Result<(), CmsError> {
        let cn = Layout::from(LAYOUT);
        let channels = cn.channels();

        let transform = self.matrix;

        for chunk in dst.chunks_exact_mut(channels) {
            let r = chunk[0];
            let g = chunk[1];
            let b = chunk[2];

            chunk[0] = mlaf(
                mlaf(r * transform.v[0][0], g, transform.v[0][1]),
                b,
                transform.v[0][2],
            );

            chunk[1] = mlaf(
                mlaf(r * transform.v[1][0], g, transform.v[1][1]),
                b,
                transform.v[1][2],
            );

            chunk[2] = mlaf(
                mlaf(r * transform.v[2][0], g, transform.v[2][1]),
                b,
                transform.v[2][2],
            );
        }

        Ok(())
    }
}

// pub(crate) struct ClipScaleStage<const LAYOUT: u8> {
//     pub(crate) scale: f32,
// }
//
// impl<const LAYOUT: u8> InPlaceStage for ClipScaleStage<LAYOUT> {
//     #[inline]
//     fn transform(&self, dst: &mut [f32]) -> Result<(), CmsError> {
//         let scale = self.scale;
//         let cn = Layout::from(LAYOUT);
//         let channels = cn.channels();
//
//         for chunk in dst.chunks_exact_mut(channels) {
//             let r = chunk[0];
//             let g = chunk[1];
//             let b = chunk[2];
//
//             chunk[0] = r.max(0f32).min(1f32).mul(scale).round();
//             chunk[1] = g.max(0f32).min(1f32).mul(scale).round();
//             chunk[2] = b.max(0f32).min(1f32).mul(scale).round();
//         }
//
//         Ok(())
//     }
// }

pub(crate) struct GamutClipScaleStage<const LAYOUT: u8> {
    pub(crate) scale: f32,
}

impl<const LAYOUT: u8> InPlaceStage for GamutClipScaleStage<LAYOUT> {
    #[inline]
    fn transform(&self, dst: &mut [f32]) -> Result<(), CmsError> {
        let cn = Layout::from(LAYOUT);
        let channels = cn.channels();

        for chunk in dst.chunks_exact_mut(channels) {
            let mut rgb = Rgb::new(chunk[0], chunk[1], chunk[2]);
            if rgb.is_out_of_gamut() {
                rgb = gamut_clip_adaptive_l0_0_5(rgb, 0.5f32);
            }
            rgb = rgb.clamp(0.0, 1.0) * Rgb::dup(self.scale);
            chunk[0] = rgb.r;
            chunk[1] = rgb.g;
            chunk[2] = rgb.b;
        }

        Ok(())
    }
}

pub(crate) struct RelativeColorMetricRgbXyz<const LAYOUT: u8> {
    pub(crate) matrix: Matrix3f,
    pub(crate) scale: f32,
}

impl<const LAYOUT: u8> InPlaceStage for RelativeColorMetricRgbXyz<LAYOUT> {
    #[inline]
    fn transform(&self, dst: &mut [f32]) -> Result<(), CmsError> {
        let cn = Layout::from(LAYOUT);
        let channels = cn.channels();

        let transform = self.matrix;

        for chunk in dst.chunks_exact_mut(channels) {
            let rgb = Rgb::new(chunk[0], chunk[1], chunk[2]);

            let mut new_rgb = rgb.apply(transform);
            if new_rgb.is_out_of_gamut() {
                new_rgb = gamut_clip_preserve_chroma(rgb);
            }
            new_rgb = new_rgb.clamp(0.0, 1.0);
            new_rgb *= self.scale;

            chunk[0] = new_rgb.r;
            chunk[1] = new_rgb.g;
            chunk[2] = new_rgb.b;
        }

        Ok(())
    }
}
