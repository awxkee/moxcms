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
    LutBarycentricReduction, RgbXyzFactory, ToneReproductionRgbToGray, TransformProfileRgb,
    make_gray_to_x, make_lut_transform, make_rgb_to_gray,
};
use crate::err::CmsError;
use crate::trc::GammaLutInterpolate;
use crate::{ColorProfile, DataColorSpace, LutWarehouse, RenderingIntent, Vector3f, Xyz};
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

/// Barycentric interpolation weights size.
///
/// Bigger weights increases precision.
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash, Default)]
pub enum BarycentricWeightScale {
    #[default]
    /// Low scale weights is enough for common case.
    ///
    /// However, it might crush dark zones and gradients.
    /// Weights increasing costs 5% performance.
    Low,
    High,
}

/// Declares additional transformation options
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct TransformOptions {
    pub rendering_intent: RenderingIntent,
    /// If set it will try to use Transfer Characteristics from CICP
    /// on transform. This might be more precise and faster.
    pub allow_use_cicp_transfer: bool,
    /// Prefers fixed point where implemented as default.
    /// Most of the applications actually do not need floating point.
    ///
    /// Do not change it if you're not sure that extreme precision is required,
    /// in most cases it is a simple way to spend energy to warming up environment
    /// a little.
    ///
    /// LUT interpolation in fixed point often have very different speed on
    /// x86 AMD and Intel CPUs.
    /// AMD tends to be more effective on fixed point on LUT interpolation whereas Intel is not.
    /// If you're targeting specific x86 CPU always benchmark first.
    pub prefer_fixed_point: bool,
    /// Interpolation method for 3D LUT
    pub interpolation_method: InterpolationMethod,
    pub barycentric_weight_scale: BarycentricWeightScale,
    // pub black_point_compensation: bool,
}

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash, Default)]
/// Defines the interpolation method.
///
/// All methods produce very close results that almost not possible to separate without
/// some automation tools.
///
/// This implementation chooses the fastest method as default.
pub enum InterpolationMethod {
    /// General Tetrahedron interpolation.
    /// This is used in lcms2 and others CMS.
    Tetrahedral,
    /// Divides cube into a pyramids and interpolate then in the pyramid.
    Pyramid,
    /// Interpolation by dividing cube into prisms.
    Prism,
    /// Trilinear/Quadlinear interpolation
    #[default]
    Linear,
}

impl Default for TransformOptions {
    fn default() -> Self {
        Self {
            rendering_intent: RenderingIntent::default(),
            allow_use_cicp_transfer: true,
            prefer_fixed_point: true,
            interpolation_method: InterpolationMethod::default(),
            barycentric_weight_scale: BarycentricWeightScale::default(),
            // black_point_compensation: false,
        }
    }
}

pub type Transform8BitExecutor = dyn TransformExecutor<u8> + Send + Sync;
pub type Transform16BitExecutor = dyn TransformExecutor<u16> + Send + Sync;
pub type TransformF32BitExecutor = dyn TransformExecutor<f32> + Send + Sync;
pub type TransformF64BitExecutor = dyn TransformExecutor<f64> + Send + Sync;

/// Layout declares a data layout.
/// For RGB it shows also the channel order.
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

impl Layout {
    pub const fn resolve(value: u8) -> Self {
        match value {
            0 => Layout::Rgb,
            1 => Layout::Rgba,
            2 => Layout::Gray,
            3 => Layout::GrayAlpha,
            _ => unimplemented!(),
        }
    }
}

#[doc(hidden)]
pub trait PointeeSizeExpressible {
    fn _as_usize(self) -> usize;
    const FINITE: bool;
    const NOT_FINITE_GAMMA_TABLE_SIZE: usize;
    const NOT_FINITE_LINEAR_TABLE_SIZE: usize;
    const IS_U8: bool;
    const IS_U16: bool;
}

impl PointeeSizeExpressible for u8 {
    #[inline(always)]
    fn _as_usize(self) -> usize {
        self as usize
    }

    const FINITE: bool = true;
    const NOT_FINITE_GAMMA_TABLE_SIZE: usize = 1;
    const NOT_FINITE_LINEAR_TABLE_SIZE: usize = 1;
    const IS_U8: bool = true;
    const IS_U16: bool = false;
}

impl PointeeSizeExpressible for u16 {
    #[inline(always)]
    fn _as_usize(self) -> usize {
        self as usize
    }

    const FINITE: bool = true;

    const NOT_FINITE_GAMMA_TABLE_SIZE: usize = 1;
    const NOT_FINITE_LINEAR_TABLE_SIZE: usize = 1;

    const IS_U8: bool = false;
    const IS_U16: bool = true;
}

impl PointeeSizeExpressible for f32 {
    #[inline(always)]
    fn _as_usize(self) -> usize {
        const MAX_14_BIT: f32 = ((1 << 14u32) - 1) as f32;
        ((self * MAX_14_BIT).max(0f32).min(MAX_14_BIT) as u16) as usize
    }

    const FINITE: bool = false;

    const NOT_FINITE_GAMMA_TABLE_SIZE: usize = 32768;
    const NOT_FINITE_LINEAR_TABLE_SIZE: usize = 1 << 14u32;
    const IS_U8: bool = false;
    const IS_U16: bool = false;
}

impl PointeeSizeExpressible for f64 {
    #[inline(always)]
    fn _as_usize(self) -> usize {
        const MAX_16_BIT: f64 = ((1 << 16u32) - 1) as f64;
        ((self * MAX_16_BIT).max(0.).min(MAX_16_BIT) as u16) as usize
    }

    const FINITE: bool = false;

    const NOT_FINITE_GAMMA_TABLE_SIZE: usize = 65536;
    const NOT_FINITE_LINEAR_TABLE_SIZE: usize = 1 << 16;
    const IS_U8: bool = false;
    const IS_U16: bool = false;
}

impl ColorProfile {
    pub(crate) fn has_full_colors_triplet(&self) -> bool {
        self.red_colorant != Xyz::default()
            && self.green_colorant != Xyz::default()
            && self.blue_colorant != Xyz::default()
            && self.red_trc.is_some()
            && self.green_trc.is_some()
            && self.blue_trc.is_some()
    }

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
        self.create_transform_nbit::<u16, 12, 65536, 16384>(src_layout, dst_pr, dst_layout, options)
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
        self.create_transform_nbit::<u16, 10, 65536, 8192>(src_layout, dst_pr, dst_layout, options)
    }

    /// Creates transform between source and destination profile
    /// Data has to be normalized into [0, 1] range.
    /// ICC profiles and LUT tables do not exist in infinite precision.
    /// Thus, this implementation considers `f32` as 14-bit values.
    pub fn create_transform_f32(
        &self,
        src_layout: Layout,
        dst_pr: &ColorProfile,
        dst_layout: Layout,
        options: TransformOptions,
    ) -> Result<Box<TransformF32BitExecutor>, CmsError> {
        self.create_transform_nbit::<f32, 1, 65536, 32768>(src_layout, dst_pr, dst_layout, options)
    }

    /// Creates transform between source and destination profile
    /// Data has to be normalized into [0, 1] range.
    /// ICC profiles and LUT tables do not exist in infinite precision.
    /// Thus, this implementation considers `f64` as 16-bit values.
    pub fn create_transform_f64(
        &self,
        src_layout: Layout,
        dst_pr: &ColorProfile,
        dst_layout: Layout,
        options: TransformOptions,
    ) -> Result<Box<TransformF64BitExecutor>, CmsError> {
        self.create_transform_nbit::<f64, 1, 65536, 65536>(src_layout, dst_pr, dst_layout, options)
    }

    fn create_transform_nbit<
        T: Copy
            + Default
            + AsPrimitive<usize>
            + PointeeSizeExpressible
            + Send
            + Sync
            + AsPrimitive<f32>
            + RgbXyzFactory<T>
            + GammaLutInterpolate,
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
        (): LutBarycentricReduction<T, u8>,
        (): LutBarycentricReduction<T, u16>,
    {
        if self.color_space == DataColorSpace::Rgb
            && dst_pr.pcs == DataColorSpace::Xyz
            && dst_pr.color_space == DataColorSpace::Rgb
            && self.pcs == DataColorSpace::Xyz
            && self.has_full_colors_triplet()
            && dst_pr.has_full_colors_triplet()
        {
            if src_layout == Layout::Gray || src_layout == Layout::GrayAlpha {
                return Err(CmsError::InvalidLayout);
            }
            if dst_layout == Layout::Gray || dst_layout == Layout::GrayAlpha {
                return Err(CmsError::InvalidLayout);
            }
            let transform = self.transform_matrix(dst_pr);

            let lin_r = self.build_r_linearize_table::<T, LINEAR_CAP, BIT_DEPTH>(
                options.allow_use_cicp_transfer,
            )?;
            let lin_g = self.build_g_linearize_table::<T, LINEAR_CAP, BIT_DEPTH>(
                options.allow_use_cicp_transfer,
            )?;
            let lin_b = self.build_b_linearize_table::<T, LINEAR_CAP, BIT_DEPTH>(
                options.allow_use_cicp_transfer,
            )?;

            let gamma_r = dst_pr.build_gamma_table::<T, 65536, GAMMA_CAP, BIT_DEPTH>(
                &self.red_trc,
                options.allow_use_cicp_transfer,
            )?;
            let gamma_g = dst_pr.build_gamma_table::<T, 65536, GAMMA_CAP, BIT_DEPTH>(
                &self.green_trc,
                options.allow_use_cicp_transfer,
            )?;
            let gamma_b = dst_pr.build_gamma_table::<T, 65536, GAMMA_CAP, BIT_DEPTH>(
                &self.blue_trc,
                options.allow_use_cicp_transfer,
            )?;

            let profile_transform = TransformProfileRgb {
                r_linear: lin_r,
                g_linear: lin_g,
                b_linear: lin_b,
                r_gamma: gamma_r,
                g_gamma: gamma_g,
                b_gamma: gamma_b,
                adaptation_matrix: transform,
            };

            return T::make_transform::<LINEAR_CAP, GAMMA_CAP, BIT_DEPTH>(
                src_layout,
                dst_layout,
                profile_transform,
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
            let gray_linear = self.build_gray_linearize_table::<T, LINEAR_CAP, BIT_DEPTH>()?;
            let gray_gamma = dst_pr.build_gamma_table::<T, 65536, GAMMA_CAP, BIT_DEPTH>(
                &self.gray_trc,
                options.allow_use_cicp_transfer,
            )?;

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

            let lin_r = self.build_r_linearize_table::<T, LINEAR_CAP, BIT_DEPTH>(
                options.allow_use_cicp_transfer,
            )?;
            let lin_g = self.build_g_linearize_table::<T, LINEAR_CAP, BIT_DEPTH>(
                options.allow_use_cicp_transfer,
            )?;
            let lin_b = self.build_b_linearize_table::<T, LINEAR_CAP, BIT_DEPTH>(
                options.allow_use_cicp_transfer,
            )?;
            let gray_linear = dst_pr.build_gamma_table::<T, 65536, GAMMA_CAP, BIT_DEPTH>(
                &dst_pr.gray_trc,
                options.allow_use_cicp_transfer,
            )?;

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
        } else if (self.color_space == DataColorSpace::Cmyk
            || self.color_space == DataColorSpace::Rgb
            || self.color_space == DataColorSpace::Lab
            || self.color_space == DataColorSpace::Color3
            || self.color_space == DataColorSpace::Color4)
            && (dst_pr.color_space == DataColorSpace::Rgb
                || dst_pr.color_space == DataColorSpace::Cmyk
                || dst_pr.color_space == DataColorSpace::Lab
                || dst_pr.color_space == DataColorSpace::Color3
                || dst_pr.color_space == DataColorSpace::Color4)
            && (dst_pr.pcs == DataColorSpace::Xyz || dst_pr.pcs == DataColorSpace::Lab)
            && (self.pcs == DataColorSpace::Xyz || self.pcs == DataColorSpace::Lab)
        {
            if src_layout == Layout::Gray || src_layout == Layout::GrayAlpha {
                return Err(CmsError::InvalidLayout);
            }
            if dst_layout == Layout::Gray || dst_layout == Layout::GrayAlpha {
                return Err(CmsError::InvalidLayout);
            }
            return make_lut_transform::<T, BIT_DEPTH, LINEAR_CAP, GAMMA_CAP>(
                src_layout, self, dst_layout, dst_pr, options,
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
        self.create_transform_nbit::<u8, 8, 256, 4096>(src_layout, dst_pr, dst_layout, options)
    }

    pub(crate) fn get_device_to_pcs(&self, intent: RenderingIntent) -> Option<&LutWarehouse> {
        match intent {
            RenderingIntent::AbsoluteColorimetric => self.lut_a_to_b_colorimetric.as_ref(),
            RenderingIntent::Saturation => self.lut_a_to_b_saturation.as_ref(),
            RenderingIntent::RelativeColorimetric => self.lut_a_to_b_colorimetric.as_ref(),
            RenderingIntent::Perceptual => self.lut_a_to_b_perceptual.as_ref(),
        }
    }

    pub(crate) fn get_pcs_to_device(&self, intent: RenderingIntent) -> Option<&LutWarehouse> {
        match intent {
            RenderingIntent::AbsoluteColorimetric => self.lut_b_to_a_colorimetric.as_ref(),
            RenderingIntent::Saturation => self.lut_b_to_a_saturation.as_ref(),
            RenderingIntent::RelativeColorimetric => self.lut_b_to_a_colorimetric.as_ref(),
            RenderingIntent::Perceptual => self.lut_b_to_a_perceptual.as_ref(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{ColorProfile, Layout, RenderingIntent, TransformOptions};
    use rand::Rng;

    #[test]
    fn test_transform_rgb8() {
        let mut srgb_profile = ColorProfile::new_srgb();
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

        let transform = bt2020_profile
            .create_transform_8bit(
                Layout::Rgb,
                &srgb_profile,
                Layout::Rgb,
                TransformOptions {
                    ..TransformOptions::default()
                },
            )
            .unwrap();
        transform.transform(&src, &mut dst).unwrap();
        srgb_profile.rendering_intent = RenderingIntent::RelativeColorimetric;
        let transform = bt2020_profile
            .create_transform_8bit(
                Layout::Rgb,
                &srgb_profile,
                Layout::Rgb,
                TransformOptions {
                    ..TransformOptions::default()
                },
            )
            .unwrap();
        transform.transform(&src, &mut dst).unwrap();
        srgb_profile.rendering_intent = RenderingIntent::Saturation;
        let transform = bt2020_profile
            .create_transform_8bit(
                Layout::Rgb,
                &srgb_profile,
                Layout::Rgb,
                TransformOptions {
                    ..TransformOptions::default()
                },
            )
            .unwrap();
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
