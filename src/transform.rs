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
use crate::conversions::{
    CompressCmykLut, ToneReproductionRgbToGray, TransformProfileRgb, make_cmyk_to_rgb,
    make_gray_to_x, make_rgb_to_gray, make_rgb_xyz_rgb_transform,
};
use crate::err::CmsError;
use crate::{ColorProfile, DataColorSpace, Vector3f};
use num_traits::AsPrimitive;

/// Transformation executor itself
pub trait TransformExecutor<V: Copy + Default> {
    /// Count of samples always must match.
    /// If there is N samples of *Cmyk* source then N samples of *Rgb* is expected as an output.
    fn transform(&self, src: &[V], dst: &mut [V]) -> Result<(), CmsError>;
}

/// Helper for intermediate transformation stages
pub trait Stage {
    fn transform(&self, src: &[f32], dst: &mut [f32]) -> Result<(), CmsError>;
}

/// Helper for intermediate transformation stages
pub trait InPlaceStage {
    fn transform(&self, dst: &mut [f32]) -> Result<(), CmsError>;
}

/// Declares additional transformation options
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash, Default)]
pub struct TransformOptions {
    /// If enabled in the transformation attempt to
    /// clip gamut chroma if it is out range will be performed.
    /// This is slow option. Transformation will be at least 2 times slower.
    pub allow_chroma_clipping: bool,
}

pub type Transform8BitExecutor = dyn TransformExecutor<u8> + Send + Sync;
pub type Transform16BitExecutor = dyn TransformExecutor<u16> + Send + Sync;

/// Layout declares a data layout.
/// For RGB it shows also the channel order.
/// 8, and 16 bits it is storage size, not a data size.
/// To handle different data bit-depth appropriate executor must be used.
/// Cmyk8 uses the same layout as Rgba8.
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub enum Layout {
    Rgb = 0,
    Rgba = 1,
    Gray = 2,
    GrayAlpha = 3,
}

impl Layout {
    /// Returns Red channel index
    #[inline(always)]
    pub const fn r_i(self) -> usize {
        match self {
            Layout::Rgb => 0,
            Layout::Rgba => 0,
            Layout::Gray => unimplemented!(),
            Layout::GrayAlpha => unimplemented!(),
        }
    }

    /// Returns Green channel index
    #[inline(always)]
    pub const fn g_i(self) -> usize {
        match self {
            Layout::Rgb => 1,
            Layout::Rgba => 1,
            Layout::Gray => unimplemented!(),
            Layout::GrayAlpha => unimplemented!(),
        }
    }

    /// Returns Blue channel index
    #[inline(always)]
    pub const fn b_i(self) -> usize {
        match self {
            Layout::Rgb => 2,
            Layout::Rgba => 2,
            Layout::Gray => unimplemented!(),
            Layout::GrayAlpha => unimplemented!(),
        }
    }

    #[inline(always)]
    pub const fn a_i(self) -> usize {
        match self {
            Layout::Rgb => unimplemented!(),
            Layout::Rgba => 3,
            Layout::Gray => unimplemented!(),
            Layout::GrayAlpha => 1,
        }
    }

    #[inline(always)]
    pub const fn has_alpha(self) -> bool {
        match self {
            Layout::Rgb => false,
            Layout::Rgba => true,
            Layout::Gray => false,
            Layout::GrayAlpha => true,
        }
    }

    #[inline]
    pub const fn channels(self) -> usize {
        match self {
            Layout::Rgb => 3,
            Layout::Rgba => 4,
            Layout::Gray => 1,
            Layout::GrayAlpha => 2,
        }
    }
}

impl From<u8> for Layout {
    fn from(value: u8) -> Self {
        match value {
            0 => Layout::Rgb,
            1 => Layout::Rgba,
            2 => Layout::Gray,
            3 => Layout::GrayAlpha,
            _ => unimplemented!(),
        }
    }
}

impl ColorProfile {
    /// Creates transform between source and destination profile
    /// Use for 16 bit-depth data bit-depth only.
    pub fn create_transform_16bit(
        &self,
        src_layout: Layout,
        dst_pr: &ColorProfile,
        dst_layout: Layout,
        options: TransformOptions,
    ) -> Result<Box<Transform16BitExecutor>, CmsError> {
        self.create_transform_nbit::<u16, 16, 65536, 65536>(src_layout, dst_pr, dst_layout, options)
    }

    /// Creates transform between source and destination profile
    /// Use for 12 bit-depth data bit-depth only.
    pub fn create_transform_12bit(
        &self,
        src_layout: Layout,
        dst_pr: &ColorProfile,
        dst_layout: Layout,
        options: TransformOptions,
    ) -> Result<Box<Transform16BitExecutor>, CmsError> {
        const CAP: usize = 1 << 12;
        self.create_transform_nbit::<u16, 12, CAP, 16384>(src_layout, dst_pr, dst_layout, options)
    }

    /// Creates transform between source and destination profile
    /// Use for 10 bit-depth data bit-depth only.
    pub fn create_transform_10bit(
        &self,
        src_layout: Layout,
        dst_pr: &ColorProfile,
        dst_layout: Layout,
        options: TransformOptions,
    ) -> Result<Box<Transform16BitExecutor>, CmsError> {
        const CAP: usize = 1 << 10;
        self.create_transform_nbit::<u16, 10, CAP, 8192>(src_layout, dst_pr, dst_layout, options)
    }

    fn create_transform_nbit<
        T: Copy + Default + AsPrimitive<usize> + Send + Sync + AsPrimitive<f32> + CompressCmykLut,
        const BIT_DEPTH: usize,
        const LINEAR_CAP: usize,
        const GAMMA_CAP: usize,
    >(
        &self,
        src_layout: Layout,
        dst_pr: &ColorProfile,
        dst_layout: Layout,
        options: TransformOptions,
    ) -> Result<Box<dyn TransformExecutor<T> + Send + Sync>, CmsError>
    where
        f32: AsPrimitive<T>,
        u32: AsPrimitive<T>,
    {
        if self.color_space == DataColorSpace::Rgb
            && dst_pr.pcs == DataColorSpace::Xyz
            && dst_pr.color_space == DataColorSpace::Rgb
            && self.pcs == DataColorSpace::Xyz
        {
            if src_layout == Layout::Gray || src_layout == Layout::GrayAlpha {
                return Err(CmsError::InvalidLayout);
            }
            if dst_layout == Layout::Gray || dst_layout == Layout::GrayAlpha {
                return Err(CmsError::InvalidLayout);
            }
            let transform = self.transform_matrix(dst_pr);

            let lin_r = self.build_r_linearize_table::<LINEAR_CAP>()?;
            let lin_g = self.build_g_linearize_table::<LINEAR_CAP>()?;
            let lin_b = self.build_b_linearize_table::<LINEAR_CAP>()?;

            let gamma_r =
                dst_pr.build_gamma_table::<T, 65536, GAMMA_CAP, BIT_DEPTH>(&self.red_trc)?;
            let gamma_g =
                dst_pr.build_gamma_table::<T, 65536, GAMMA_CAP, BIT_DEPTH>(&self.green_trc)?;
            let gamma_b =
                dst_pr.build_gamma_table::<T, 65536, GAMMA_CAP, BIT_DEPTH>(&self.blue_trc)?;

            let profile_transform = TransformProfileRgb {
                r_linear: lin_r,
                g_linear: lin_g,
                b_linear: lin_b,
                r_gamma: gamma_r,
                g_gamma: gamma_g,
                b_gamma: gamma_b,
                adaptation_matrix: transform,
            };

            return make_rgb_xyz_rgb_transform::<T, LINEAR_CAP, GAMMA_CAP, BIT_DEPTH>(
                src_layout,
                dst_layout,
                profile_transform,
                self.rendering_intent,
                options,
            );
        } else if self.color_space == DataColorSpace::Gray
            && (dst_pr.color_space == DataColorSpace::Rgb
                || dst_pr.color_space == DataColorSpace::Gray)
            && self.pcs == DataColorSpace::Xyz
            && dst_pr.pcs == DataColorSpace::Xyz
        {
            if src_layout != Layout::GrayAlpha && src_layout != Layout::Gray {
                return Err(CmsError::InvalidLayout);
            }
            let gray_linear = self.build_gray_linearize_table::<LINEAR_CAP>()?;
            let gray_gamma =
                dst_pr.build_gamma_table::<T, 65536, GAMMA_CAP, BIT_DEPTH>(&self.gray_trc)?;

            return make_gray_to_x::<T, LINEAR_CAP, BIT_DEPTH, GAMMA_CAP>(
                src_layout,
                dst_layout,
                gray_linear,
                gray_gamma,
            );
        } else if self.color_space == DataColorSpace::Rgb
            && dst_pr.color_space == DataColorSpace::Gray
            && dst_pr.pcs == DataColorSpace::Xyz
            && self.pcs == DataColorSpace::Xyz
        {
            if src_layout == Layout::Gray || src_layout == Layout::GrayAlpha {
                return Err(CmsError::InvalidLayout);
            }
            if dst_layout != Layout::Gray && dst_layout != Layout::GrayAlpha {
                return Err(CmsError::InvalidLayout);
            }

            let lin_r = self.build_r_linearize_table::<LINEAR_CAP>()?;
            let lin_g = self.build_g_linearize_table::<LINEAR_CAP>()?;
            let lin_b = self.build_b_linearize_table::<LINEAR_CAP>()?;
            let gray_linear =
                dst_pr.build_gamma_table::<T, 65536, GAMMA_CAP, BIT_DEPTH>(&dst_pr.gray_trc)?;

            let transform = self
                .rgb_to_xyz_matrix()
                .ok_or(CmsError::UnsupportedProfileConnection)?;

            let vector = Vector3f {
                v: [transform.v[1][0], transform.v[1][1], transform.v[1][2]],
            };

            let trc_box = ToneReproductionRgbToGray::<T, LINEAR_CAP> {
                r_linear: lin_r,
                g_linear: lin_g,
                b_linear: lin_b,
                gray_gamma: gray_linear,
            };

            return Ok(make_rgb_to_gray::<T, LINEAR_CAP, BIT_DEPTH, GAMMA_CAP>(
                src_layout, dst_layout, trc_box, vector,
            ));
        } else if self.color_space == DataColorSpace::Cmyk
            && dst_pr.color_space == DataColorSpace::Rgb
            && (dst_pr.pcs == DataColorSpace::Xyz || dst_pr.pcs == DataColorSpace::Lab)
            && (self.pcs == DataColorSpace::Xyz || self.pcs == DataColorSpace::Lab)
        {
            if src_layout == Layout::Gray || src_layout == Layout::GrayAlpha {
                return Err(CmsError::InvalidLayout);
            }
            if dst_layout == Layout::Gray || dst_layout == Layout::GrayAlpha {
                return Err(CmsError::InvalidLayout);
            }
            return make_cmyk_to_rgb::<T, BIT_DEPTH, GAMMA_CAP>(
                src_layout, self, dst_layout, dst_pr,
            );
        }

        Err(CmsError::UnsupportedProfileConnection)
    }

    /// Creates transform between source and destination profile
    /// Only 8 bit is supported.
    pub fn create_transform_8bit(
        &self,
        src_layout: Layout,
        dst_pr: &ColorProfile,
        dst_layout: Layout,
        options: TransformOptions,
    ) -> Result<Box<Transform8BitExecutor>, CmsError> {
        self.create_transform_nbit::<u8, 8, 256, 8192>(src_layout, dst_pr, dst_layout, options)
    }
}

#[cfg(test)]
mod tests {
    use crate::{ColorProfile, Layout, TransformOptions};
    use rand::Rng;

    #[test]
    fn test_transform_rgb8() {
        let srgb_profile = ColorProfile::new_srgb();
        let bt2020_profile = ColorProfile::new_bt2020();
        let random_point_x = rand::rng().random_range(0..255);
        let transform = bt2020_profile
            .create_transform_8bit(
                Layout::Rgb,
                &srgb_profile,
                Layout::Rgb,
                TransformOptions::default(),
            )
            .unwrap();
        let src = vec![random_point_x; 256 * 256 * 3];
        let mut dst = vec![random_point_x; 256 * 256 * 3];
        transform.transform(&src, &mut dst).unwrap();
    }

    #[test]
    fn test_transform_rgba8() {
        let srgb_profile = ColorProfile::new_srgb();
        let bt2020_profile = ColorProfile::new_bt2020();
        let random_point_x = rand::rng().random_range(0..255);
        let transform = bt2020_profile
            .create_transform_8bit(
                Layout::Rgba,
                &srgb_profile,
                Layout::Rgba,
                TransformOptions::default(),
            )
            .unwrap();
        let src = vec![random_point_x; 256 * 256 * 4];
        let mut dst = vec![random_point_x; 256 * 256 * 4];
        transform.transform(&src, &mut dst).unwrap();
    }

    #[test]
    fn test_transform_gray_to_rgb8() {
        let srgb_profile = ColorProfile::new_gray_with_gamma(2.2f32);
        let bt2020_profile = ColorProfile::new_bt2020();
        let random_point_x = rand::rng().random_range(0..255);
        let transform = srgb_profile
            .create_transform_8bit(
                Layout::Gray,
                &bt2020_profile,
                Layout::Rgb,
                TransformOptions::default(),
            )
            .unwrap();
        let src = vec![random_point_x; 256 * 256];
        let mut dst = vec![random_point_x; 256 * 256 * 3];
        transform.transform(&src, &mut dst).unwrap();
    }

    #[test]
    fn test_transform_gray_to_rgba8() {
        let srgb_profile = ColorProfile::new_gray_with_gamma(2.2f32);
        let bt2020_profile = ColorProfile::new_bt2020();
        let random_point_x = rand::rng().random_range(0..255);
        let transform = srgb_profile
            .create_transform_8bit(
                Layout::Gray,
                &bt2020_profile,
                Layout::Rgba,
                TransformOptions::default(),
            )
            .unwrap();
        let src = vec![random_point_x; 256 * 256];
        let mut dst = vec![random_point_x; 256 * 256 * 4];
        transform.transform(&src, &mut dst).unwrap();
    }

    #[test]
    fn test_transform_gray_to_gray_alpha8() {
        let srgb_profile = ColorProfile::new_gray_with_gamma(2.2f32);
        let bt2020_profile = ColorProfile::new_bt2020();
        let random_point_x = rand::rng().random_range(0..255);
        let transform = srgb_profile
            .create_transform_8bit(
                Layout::Gray,
                &bt2020_profile,
                Layout::GrayAlpha,
                TransformOptions::default(),
            )
            .unwrap();
        let src = vec![random_point_x; 256 * 256];
        let mut dst = vec![random_point_x; 256 * 256 * 2];
        transform.transform(&src, &mut dst).unwrap();
    }

    #[test]
    fn test_transform_rgb10() {
        let srgb_profile = ColorProfile::new_srgb();
        let bt2020_profile = ColorProfile::new_bt2020();
        let random_point_x = rand::rng().random_range(0..((1 << 10) - 1));
        let transform = bt2020_profile
            .create_transform_10bit(
                Layout::Rgb,
                &srgb_profile,
                Layout::Rgb,
                TransformOptions::default(),
            )
            .unwrap();
        let src = vec![random_point_x; 256 * 256 * 3];
        let mut dst = vec![random_point_x; 256 * 256 * 3];
        transform.transform(&src, &mut dst).unwrap();
    }

    #[test]
    fn test_transform_rgb12() {
        let srgb_profile = ColorProfile::new_srgb();
        let bt2020_profile = ColorProfile::new_bt2020();
        let random_point_x = rand::rng().random_range(0..((1 << 12) - 1));
        let transform = bt2020_profile
            .create_transform_12bit(
                Layout::Rgb,
                &srgb_profile,
                Layout::Rgb,
                TransformOptions::default(),
            )
            .unwrap();
        let src = vec![random_point_x; 256 * 256 * 3];
        let mut dst = vec![random_point_x; 256 * 256 * 3];
        transform.transform(&src, &mut dst).unwrap();
    }

    #[test]
    fn test_transform_rgb16() {
        let srgb_profile = ColorProfile::new_srgb();
        let bt2020_profile = ColorProfile::new_bt2020();
        let random_point_x = rand::rng().random_range(0..((1u32 << 16u32) - 1u32)) as u16;
        let transform = bt2020_profile
            .create_transform_16bit(
                Layout::Rgb,
                &srgb_profile,
                Layout::Rgb,
                TransformOptions::default(),
            )
            .unwrap();
        let src = vec![random_point_x; 256 * 256 * 3];
        let mut dst = vec![random_point_x; 256 * 256 * 3];
        transform.transform(&src, &mut dst).unwrap();
    }
}
