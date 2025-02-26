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
use crate::clut::create_cmyk_to_rgb;
use crate::err::CmsError;
use crate::mlaf::mlaf;
use crate::{ColorProfile, DataColorSpace, Matrix3f};
use num_traits::AsPrimitive;
use std::ops::Mul;

pub trait TransformExecutor<V: Copy + Default> {
    /// Count of samples always must match
    /// If there is N samples of *Cmyk* source then N samples of *Rgb* is expected as an output
    fn transform(&self, src: &[V], dst: &mut [V]) -> Result<(), CmsError>;
}

pub trait Stage {
    fn transform(&self, src: &[f32], dst: &mut [f32]) -> Result<(), CmsError>;
}

pub type Transform8BitExecutor = dyn TransformExecutor<u8> + Send + Sync;
pub type Transform16BitExecutor = dyn TransformExecutor<u16> + Send + Sync;

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub enum Layout {
    Rgb8 = 0,
    Rgba8 = 1,
    Rgb16 = 2,
    Rgba16 = 3,
    Gray8 = 4,
    GrayAlpha8 = 5,
    Gray16 = 6,
    GrayAlpha16 = 7,
}

impl Layout {
    /// Returns Red channel index
    #[inline(always)]
    pub const fn r_i(self) -> usize {
        match self {
            Layout::Rgb8 => 0,
            Layout::Rgba8 => 0,
            Layout::Rgb16 => 0,
            Layout::Rgba16 => 0,
            Layout::Gray8 => unimplemented!(),
            Layout::GrayAlpha8 => unimplemented!(),
            Layout::Gray16 => unimplemented!(),
            Layout::GrayAlpha16 => unimplemented!(),
        }
    }

    /// Returns Green channel index
    #[inline(always)]
    pub const fn g_i(self) -> usize {
        match self {
            Layout::Rgb8 => 1,
            Layout::Rgba8 => 1,
            Layout::Rgb16 => 1,
            Layout::Rgba16 => 1,
            Layout::Gray8 => unimplemented!(),
            Layout::GrayAlpha8 => unimplemented!(),
            Layout::Gray16 => unimplemented!(),
            Layout::GrayAlpha16 => unimplemented!(),
        }
    }

    /// Returns Blue channel index
    #[inline(always)]
    pub const fn b_i(self) -> usize {
        match self {
            Layout::Rgb8 => 2,
            Layout::Rgba8 => 2,
            Layout::Rgb16 => 2,
            Layout::Rgba16 => 2,
            Layout::Gray8 => unimplemented!(),
            Layout::GrayAlpha8 => unimplemented!(),
            Layout::Gray16 => unimplemented!(),
            Layout::GrayAlpha16 => unimplemented!(),
        }
    }

    #[inline(always)]
    pub const fn a_i(self) -> usize {
        match self {
            Layout::Rgb8 => unimplemented!(),
            Layout::Rgba8 => 3,
            Layout::Rgb16 => unimplemented!(),
            Layout::Rgba16 => 3,
            Layout::Gray8 => unimplemented!(),
            Layout::GrayAlpha8 => 1,
            Layout::Gray16 => unimplemented!(),
            Layout::GrayAlpha16 => 1,
        }
    }

    #[inline(always)]
    pub const fn has_alpha(self) -> bool {
        match self {
            Layout::Rgb8 => false,
            Layout::Rgba8 => true,
            Layout::Rgb16 => false,
            Layout::Rgba16 => true,
            Layout::Gray8 => false,
            Layout::GrayAlpha8 => true,
            Layout::Gray16 => false,
            Layout::GrayAlpha16 => true,
        }
    }

    #[inline]
    pub fn is_16_bit(self) -> bool {
        if self == Layout::Rgb16
            || self == Layout::Rgba16
            || self == Layout::Gray16
            || self == Layout::GrayAlpha16
        {
            return true;
        }
        false
    }

    #[inline]
    pub const fn channels(self) -> usize {
        match self {
            Layout::Rgb8 => 3,
            Layout::Rgba8 => 4,
            Layout::Rgb16 => 3,
            Layout::Rgba16 => 4,
            Layout::Gray8 => 1,
            Layout::GrayAlpha8 => 2,
            Layout::Gray16 => 1,
            Layout::GrayAlpha16 => 2,
        }
    }
}

impl From<u8> for Layout {
    fn from(value: u8) -> Self {
        match value {
            0 => Layout::Rgb8,
            1 => Layout::Rgba8,
            2 => Layout::Rgb16,
            3 => Layout::Rgba16,
            4 => Layout::Gray8,
            5 => Layout::GrayAlpha8,
            6 => Layout::Gray16,
            7 => Layout::GrayAlpha16,
            _ => unimplemented!(),
        }
    }
}

#[derive(Clone)]
struct TransformProfileRgb8Bit {
    r_linear: Box<[f32; 256]>,
    g_linear: Box<[f32; 256]>,
    b_linear: Box<[f32; 256]>,
    r_gamma: Box<[u8; 65536]>,
    g_gamma: Box<[u8; 65536]>,
    b_gamma: Box<[u8; 65536]>,
    adaptation_matrix: Option<Matrix3f>,
}

#[derive(Clone)]
struct TransformProfileRgb16Bit<const BUCKET: usize> {
    r_linear: Box<[f32; BUCKET]>,
    g_linear: Box<[f32; BUCKET]>,
    b_linear: Box<[f32; BUCKET]>,
    r_gamma: Box<[u16; 65536]>,
    g_gamma: Box<[u16; 65536]>,
    b_gamma: Box<[u16; 65536]>,
    adaptation_matrix: Option<Matrix3f>,
}

#[derive(Clone)]
struct TransformProfileGrayToRgb<
    T,
    const DEST_LAYOUT: u8,
    const BUCKET: usize,
    const BIT_DEPTH: usize,
    const GAMMA_LUT: usize,
> {
    gray_linear: Box<[f32; BUCKET]>,
    gray_gamma: Box<[T; 65536]>,
}

struct TransformProfilePcsXYZRgb8Bit<const LAYOUT: u8> {
    profile: TransformProfileRgb8Bit,
}

struct TransformProfilePcsXYZRgb16Bit<
    const LAYOUT: u8,
    const LINEAR_CAP: usize,
    const GAMMA_LUT: usize,
> {
    profile: TransformProfileRgb16Bit<LINEAR_CAP>,
}

impl ColorProfile {
    /// Use for 16 bit-depth only
    pub fn create_transform_16bit(
        &self,
        destination_profile: &ColorProfile,
        layout: Layout,
    ) -> Result<Box<Transform16BitExecutor>, CmsError> {
        self.create_transform_nbit::<16, 65536, 65536>(destination_profile, layout)
    }

    /// Use for 12 bit-depth only
    pub fn create_transform_12bit(
        &self,
        destination_profile: &ColorProfile,
        layout: Layout,
    ) -> Result<Box<Transform16BitExecutor>, CmsError> {
        const CAP: usize = 1 << 12;
        self.create_transform_nbit::<12, CAP, 16384>(destination_profile, layout)
    }

    /// Use for 10 bit-depth only
    pub fn create_transform_10bit(
        &self,
        destination_profile: &ColorProfile,
        layout: Layout,
    ) -> Result<Box<Transform16BitExecutor>, CmsError> {
        const CAP: usize = 1 << 10;
        self.create_transform_nbit::<10, CAP, 8192>(destination_profile, layout)
    }

    fn create_transform_nbit<
        const BIT_DEPTH: usize,
        const LINEAR_CAP: usize,
        const GAMMA_CAP: usize,
    >(
        &self,
        destination_profile: &ColorProfile,
        layout: Layout,
    ) -> Result<Box<Transform16BitExecutor>, CmsError> {
        if layout == Layout::Rgba8
            || layout == Layout::GrayAlpha8
            || layout == Layout::Rgb8
            || layout == Layout::Gray8
        {
            return Err(CmsError::InvalidLayout);
        }
        if self.color_space == DataColorSpace::Rgb
            && destination_profile.pcs == DataColorSpace::Xyz
            && destination_profile.color_space == DataColorSpace::Rgb
            && self.pcs == DataColorSpace::Xyz
        {
            if layout == Layout::Gray16 || layout == Layout::GrayAlpha16 {
                return Err(CmsError::InvalidLayout);
            }
            let transform = self.transform_matrix(destination_profile);

            let image_linearize_map_r = self.build_r_linearize_table::<LINEAR_CAP>()?;
            let image_linearize_map_g = self.build_g_linearize_table::<LINEAR_CAP>()?;
            let image_linearize_map_b = self.build_b_linearize_table::<LINEAR_CAP>()?;

            let output_gamma_map_r: Box<[u16; 65536]> =
                self.build_gamma_table::<u16, 65536, GAMMA_CAP, BIT_DEPTH>(&self.red_trc)?;
            let output_gamma_map_g: Box<[u16; 65536]> =
                self.build_gamma_table::<u16, 65536, GAMMA_CAP, BIT_DEPTH>(&self.green_trc)?;
            let output_gamma_map_b: Box<[u16; 65536]> =
                self.build_gamma_table::<u16, 65536, GAMMA_CAP, BIT_DEPTH>(&self.blue_trc)?;

            let profile_transform = TransformProfileRgb16Bit {
                r_linear: image_linearize_map_r,
                g_linear: image_linearize_map_g,
                b_linear: image_linearize_map_b,
                r_gamma: output_gamma_map_r,
                g_gamma: output_gamma_map_g,
                b_gamma: output_gamma_map_b,
                adaptation_matrix: transform,
            };

            let transformer: Box<Transform16BitExecutor> = match layout {
                Layout::Rgb16 => Box::new(TransformProfilePcsXYZRgb16Bit::<
                    { Layout::Rgb8 as u8 },
                    LINEAR_CAP,
                    GAMMA_CAP,
                > {
                    profile: profile_transform,
                }),
                Layout::Rgba16 => Box::new(TransformProfilePcsXYZRgb16Bit::<
                    { Layout::Rgba8 as u8 },
                    LINEAR_CAP,
                    GAMMA_CAP,
                > {
                    profile: profile_transform,
                }),
                _ => unimplemented!(),
            };
            return Ok(transformer);
        } else if self.color_space == DataColorSpace::Gray
            && destination_profile.color_space == DataColorSpace::Rgb
            && self.pcs == DataColorSpace::Xyz
            && destination_profile.pcs == DataColorSpace::Xyz
        {
            let linear_tab = self.build_gray_linearize_table::<LINEAR_CAP>()?;
            let output_gamma: Box<[u16; 65536]> =
                self.build_gamma_table::<u16, 65536, GAMMA_CAP, BIT_DEPTH>(&self.gray_trc)?;

            let transformer: Box<Transform16BitExecutor> = match layout {
                Layout::Rgb8 => {
                    let profile = TransformProfileGrayToRgb::<
                        u16,
                        { Layout::Rgb8 as u8 },
                        LINEAR_CAP,
                        BIT_DEPTH,
                        GAMMA_CAP,
                    > {
                        gray_linear: linear_tab,
                        gray_gamma: output_gamma,
                    };
                    Box::new(profile)
                }
                Layout::Rgba8 => {
                    let profile = TransformProfileGrayToRgb::<
                        u16,
                        { Layout::Rgba8 as u8 },
                        LINEAR_CAP,
                        BIT_DEPTH,
                        GAMMA_CAP,
                    > {
                        gray_linear: linear_tab,
                        gray_gamma: output_gamma,
                    };
                    Box::new(profile)
                }
                Layout::Gray8 => {
                    let profile = TransformProfileGrayToRgb::<
                        u16,
                        { Layout::Gray8 as u8 },
                        LINEAR_CAP,
                        BIT_DEPTH,
                        GAMMA_CAP,
                    > {
                        gray_linear: linear_tab,
                        gray_gamma: output_gamma,
                    };
                    Box::new(profile)
                }
                Layout::GrayAlpha8 => {
                    let profile = TransformProfileGrayToRgb::<
                        u16,
                        { Layout::GrayAlpha8 as u8 },
                        LINEAR_CAP,
                        BIT_DEPTH,
                        GAMMA_CAP,
                    > {
                        gray_linear: linear_tab,
                        gray_gamma: output_gamma,
                    };
                    Box::new(profile)
                }
                _ => unimplemented!(),
            };
            return Ok(transformer);
        }

        Err(CmsError::UnsupportedProfileConnection)
    }

    pub fn create_transform_8bit(
        &self,
        destination_profile: &ColorProfile,
        layout: Layout,
    ) -> Result<Box<Transform8BitExecutor>, CmsError> {
        if layout.is_16_bit() {
            return Err(CmsError::InvalidLayout);
        }

        if self.color_space == DataColorSpace::Rgb
            && destination_profile.pcs == DataColorSpace::Xyz
            && destination_profile.color_space == DataColorSpace::Rgb
            && self.pcs == DataColorSpace::Xyz
        {
            if layout == Layout::Gray8 || layout == Layout::GrayAlpha8 {
                return Err(CmsError::InvalidLayout);
            }
            let transform = self.transform_matrix(destination_profile);

            let image_linearize_map_r = self.build_8bit_lin_table(&self.red_trc)?;
            let image_linearize_map_g = self.build_8bit_lin_table(&self.green_trc)?;
            let image_linearize_map_b = self.build_8bit_lin_table(&self.blue_trc)?;

            let output_gamma_map_r: Box<[u8; 65536]> =
                self.build_8bit_gamma_table(&self.red_trc)?;
            let output_gamma_map_g: Box<[u8; 65536]> =
                self.build_8bit_gamma_table(&self.green_trc)?;
            let output_gamma_map_b: Box<[u8; 65536]> =
                self.build_8bit_gamma_table(&self.blue_trc)?;

            let profile_transform = TransformProfileRgb8Bit {
                r_linear: image_linearize_map_r,
                g_linear: image_linearize_map_g,
                b_linear: image_linearize_map_b,
                r_gamma: output_gamma_map_r,
                g_gamma: output_gamma_map_g,
                b_gamma: output_gamma_map_b,
                adaptation_matrix: transform,
            };

            let transformer: Box<Transform8BitExecutor> = match layout {
                Layout::Rgb8 => Box::new(TransformProfilePcsXYZRgb8Bit::<{ Layout::Rgb8 as u8 }> {
                    profile: profile_transform,
                }),
                Layout::Rgba8 => {
                    Box::new(TransformProfilePcsXYZRgb8Bit::<{ Layout::Rgba8 as u8 }> {
                        profile: profile_transform,
                    })
                }
                _ => unimplemented!(),
            };
            return Ok(transformer);
        } else if self.color_space == DataColorSpace::Gray
            && destination_profile.color_space == DataColorSpace::Rgb
            && self.pcs == DataColorSpace::Xyz
            && destination_profile.pcs == DataColorSpace::Xyz
        {
            let linear_tab = self.build_8bit_lin_table(&self.gray_trc)?;
            let output_gamma: Box<[u8; 65536]> = self.build_8bit_gamma_table(&self.gray_trc)?;

            let transformer: Box<Transform8BitExecutor> = match layout {
                Layout::Rgb8 => {
                    let profile =
                        TransformProfileGrayToRgb::<u8, { Layout::Rgb8 as u8 }, 256, 8, 8192> {
                            gray_linear: linear_tab,
                            gray_gamma: output_gamma,
                        };
                    Box::new(profile)
                }
                Layout::Rgba8 => {
                    let profile =
                        TransformProfileGrayToRgb::<u8, { Layout::Rgba8 as u8 }, 256, 8, 8192> {
                            gray_linear: linear_tab,
                            gray_gamma: output_gamma,
                        };
                    Box::new(profile)
                }
                Layout::Gray8 => {
                    let profile =
                        TransformProfileGrayToRgb::<u8, { Layout::Gray8 as u8 }, 256, 8, 8192> {
                            gray_linear: linear_tab,
                            gray_gamma: output_gamma,
                        };
                    Box::new(profile)
                }
                Layout::GrayAlpha8 => {
                    let profile = TransformProfileGrayToRgb::<
                        u8,
                        { Layout::GrayAlpha8 as u8 },
                        256,
                        8,
                        8192,
                    > {
                        gray_linear: linear_tab,
                        gray_gamma: output_gamma,
                    };
                    Box::new(profile)
                }
                _ => unimplemented!(),
            };
            return Ok(transformer);
        } else if self.color_space == DataColorSpace::Cmyk
            && destination_profile.color_space == DataColorSpace::Rgb
        {
            if layout == Layout::Gray8 || layout == Layout::GrayAlpha8 {
                return Err(CmsError::InvalidLayout);
            }
            return create_cmyk_to_rgb(self, destination_profile, layout);
        }

        Err(CmsError::UnsupportedProfileConnection)
    }
}

impl<const LAYOUT: u8> TransformProfilePcsXYZRgb8Bit<LAYOUT> {
    #[inline(always)]
    fn transform_chunk(&self, src: &[u8], dst: &mut [u8], working_set: &mut [f32; 672]) {
        let cn = Layout::from(LAYOUT);
        let channels = cn.channels();

        for (chunk, dst) in src
            .chunks_exact(cn.channels())
            .zip(working_set.chunks_exact_mut(cn.channels()))
        {
            dst[0] = self.profile.r_linear[chunk[cn.r_i()] as usize];
            dst[1] = self.profile.g_linear[chunk[cn.g_i()] as usize];
            dst[2] = self.profile.b_linear[chunk[cn.b_i()] as usize];
            if channels == 4 {
                dst[3] = f32::from_bits(chunk[cn.a_i()] as u32);
            }
        }

        if let Some(transform) = self.profile.adaptation_matrix {
            assert!(src.len() <= 672, "Received {}", src.len());
            let sliced = &mut working_set[..src.len()];
            for chunk in sliced.chunks_exact_mut(channels) {
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
                .mul(8191f32)
                .round();

                chunk[1] = mlaf(
                    mlaf(r * transform.v[1][0], g, transform.v[1][1]),
                    b,
                    transform.v[1][2],
                )
                .max(0f32)
                .min(1f32)
                .mul(8191f32)
                .round();

                chunk[2] = mlaf(
                    mlaf(r * transform.v[2][0], g, transform.v[2][1]),
                    b,
                    transform.v[2][2],
                )
                .max(0f32)
                .min(1f32)
                .mul(8191f32)
                .round();
            }
        }

        for (chunk, dst) in working_set
            .chunks_exact(cn.channels())
            .zip(dst.chunks_exact_mut(cn.channels()))
        {
            dst[cn.r_i()] = self.profile.r_gamma[chunk[0] as usize];
            dst[cn.g_i()] = self.profile.g_gamma[chunk[1] as usize];
            dst[cn.b_i()] = self.profile.b_gamma[chunk[2] as usize];
            if channels == 4 {
                dst[cn.a_i()] = chunk[3].to_bits() as u8;
            }
        }
    }
}

impl<const LAYOUT: u8> TransformExecutor<u8> for TransformProfilePcsXYZRgb8Bit<LAYOUT> {
    fn transform(&self, src: &[u8], dst: &mut [u8]) -> Result<(), CmsError> {
        let cn = Layout::from(LAYOUT);
        let channels = cn.channels();
        if src.len() != dst.len() {
            return Err(CmsError::LaneSizeMismatch);
        }
        if src.len() % channels != 0 {
            return Err(CmsError::LaneMultipleOfChannels);
        }
        if dst.len() % channels != 0 {
            return Err(CmsError::LaneMultipleOfChannels);
        }
        let mut working_set = [0f32; 672];

        let chunks = 672;

        for (src, dst) in src.chunks_exact(chunks).zip(dst.chunks_exact_mut(chunks)) {
            self.transform_chunk(src, dst, &mut working_set);
        }

        let rem = src.chunks_exact(chunks).remainder();
        let dst_rem = dst.chunks_exact_mut(chunks).into_remainder();

        if !rem.is_empty() {
            self.transform_chunk(rem, dst_rem, &mut working_set);
        }

        Ok(())
    }
}

impl<
    T: Copy + Default + AsPrimitive<usize>,
    const DEST_LAYOUT: u8,
    const BUCKET: usize,
    const BIT_DEPTH: usize,
    const GAMMA_LUT: usize,
> TransformProfileGrayToRgb<T, DEST_LAYOUT, BUCKET, BIT_DEPTH, GAMMA_LUT>
where
    u32: AsPrimitive<T>,
{
    #[inline(always)]
    fn transform_chunk(
        &self,
        src: &[T],
        dst: &mut [T],
        working_set: &mut [f32; 672],
    ) -> Result<(), CmsError> {
        let cn = Layout::from(DEST_LAYOUT);
        let channels = cn.channels();

        for (&chunk, dst) in src.iter().zip(working_set.iter_mut()) {
            *dst = self.gray_linear[chunk.as_()];
        }

        let max_value: T = ((1u32 << BIT_DEPTH as u32) - 1u32).as_();
        let max_lut_size = (GAMMA_LUT - 1) as f32;

        for (&chunk, dst) in working_set.iter().zip(dst.chunks_exact_mut(channels)) {
            let possible_value = (chunk * max_lut_size).round() as usize;
            let gamma_value = self.gray_gamma[possible_value];
            if cn == Layout::Gray8
                || cn == Layout::GrayAlpha8
                || cn == Layout::Gray16
                || cn == Layout::GrayAlpha16
            {
                dst[0] = gamma_value;
                if cn == Layout::GrayAlpha8 || cn == Layout::GrayAlpha16 {
                    dst[1] = max_value;
                }
            } else {
                dst[cn.r_i()] = gamma_value;
                dst[cn.g_i()] = gamma_value;
                dst[cn.b_i()] = gamma_value;
                if cn.has_alpha() {
                    dst[cn.a_i()] = max_value;
                }
            }
        }
        Ok(())
    }
}

impl<
    T: Copy + Default + AsPrimitive<usize>,
    const DEST_LAYOUT: u8,
    const BUCKET: usize,
    const BIT_DEPTH: usize,
    const GAMMA_LUT: usize,
> TransformExecutor<T> for TransformProfileGrayToRgb<T, DEST_LAYOUT, BUCKET, BIT_DEPTH, GAMMA_LUT>
where
    u32: AsPrimitive<T>,
{
    fn transform(&self, src: &[T], dst: &mut [T]) -> Result<(), CmsError> {
        let cn = Layout::from(DEST_LAYOUT);
        let channels = cn.channels();
        if src.len() != (dst.len() / channels) {
            return Err(CmsError::LaneSizeMismatch);
        }
        if dst.len() % channels != 0 {
            return Err(CmsError::LaneMultipleOfChannels);
        }
        let mut working_set = [0f32; 672];

        let chunks = 672 / channels;

        for (src, dst) in src.chunks_exact(672).zip(dst.chunks_exact_mut(chunks)) {
            self.transform_chunk(src, dst, &mut working_set)?;
        }

        let rem = src.chunks_exact(672).remainder();
        let dst_rem = dst.chunks_exact_mut(chunks).into_remainder();

        if !rem.is_empty() {
            self.transform_chunk(rem, dst_rem, &mut working_set)?;
        }

        Ok(())
    }
}

impl<const LAYOUT: u8, const LINEAR_CAP: usize, const GAMMA_LUT: usize>
    TransformProfilePcsXYZRgb16Bit<LAYOUT, LINEAR_CAP, GAMMA_LUT>
{
    #[inline(always)]
    fn transform_chunk(
        &self,
        src: &[u16],
        dst: &mut [u16],
        working_set: &mut [f32; 672],
    ) -> Result<(), CmsError> {
        let cn = Layout::from(LAYOUT);
        let channels = cn.channels();

        for (chunk, dst) in src
            .chunks_exact(channels)
            .zip(working_set.chunks_exact_mut(channels))
        {
            dst[0] = self.profile.r_linear[chunk[cn.r_i()] as usize];
            dst[1] = self.profile.g_linear[chunk[cn.g_i()] as usize];
            dst[2] = self.profile.b_linear[chunk[cn.b_i()] as usize];
            if channels == 4 {
                dst[3] = f32::from_bits(chunk[cn.a_i()] as u32);
            }
        }

        let cap_values = (GAMMA_LUT - 1) as f32;

        if let Some(transform) = self.profile.adaptation_matrix {
            assert!(src.len() <= 672, "Received {}", src.len());
            let sliced = &mut working_set[..src.len()];
            for chunk in sliced.chunks_exact_mut(channels) {
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
                .mul(cap_values)
                .round();

                chunk[1] = mlaf(
                    mlaf(r * transform.v[1][0], g, transform.v[1][1]),
                    b,
                    transform.v[1][2],
                )
                .max(0f32)
                .min(1f32)
                .mul(cap_values)
                .round();

                chunk[2] = mlaf(
                    mlaf(r * transform.v[2][0], g, transform.v[2][1]),
                    b,
                    transform.v[2][2],
                )
                .max(0f32)
                .min(1f32)
                .mul(cap_values)
                .round();
            }
        }

        for (chunk, dst) in working_set
            .chunks_exact(cn.channels())
            .zip(dst.chunks_exact_mut(cn.channels()))
        {
            dst[cn.r_i()] = self.profile.r_gamma[chunk[0] as usize];
            dst[cn.g_i()] = self.profile.g_gamma[chunk[1] as usize];
            dst[cn.b_i()] = self.profile.b_gamma[chunk[2] as usize];
            if channels == 4 {
                dst[cn.a_i()] = chunk[3].to_bits() as u16;
            }
        }

        Ok(())
    }
}

impl<const LAYOUT: u8, const LINEAR_CAP: usize, const GAMMA_LUT: usize> TransformExecutor<u16>
    for TransformProfilePcsXYZRgb16Bit<LAYOUT, LINEAR_CAP, GAMMA_LUT>
{
    fn transform(&self, src: &[u16], dst: &mut [u16]) -> Result<(), CmsError> {
        let cn = Layout::from(LAYOUT);
        let channels = cn.channels();
        if src.len() != dst.len() {
            return Err(CmsError::LaneSizeMismatch);
        }
        if src.len() % channels != 0 {
            return Err(CmsError::LaneMultipleOfChannels);
        }
        if dst.len() % channels != 0 {
            return Err(CmsError::LaneMultipleOfChannels);
        }
        let mut working_set = [0f32; 672];

        let chunks = 672;

        for (src, dst) in src.chunks_exact(chunks).zip(dst.chunks_exact_mut(chunks)) {
            self.transform_chunk(src, dst, &mut working_set)?;
        }

        let rem = src.chunks_exact(chunks).remainder();
        let dst_rem = dst.chunks_exact_mut(chunks).into_remainder();

        if !rem.is_empty() {
            self.transform_chunk(rem, dst_rem, &mut working_set)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{ColorProfile, Layout};
    use rand::Rng;

    #[test]
    fn test_transform_rgb8() {
        let srgb_profile = ColorProfile::new_srgb();
        let bt2020_profile = ColorProfile::new_bt2020();
        let random_point_x = rand::rng().random_range(0..255);
        let transform = bt2020_profile
            .create_transform_8bit(&srgb_profile, Layout::Rgb8)
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
            .create_transform_8bit(&srgb_profile, Layout::Rgba8)
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
            .create_transform_8bit(&bt2020_profile, Layout::Rgb8)
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
            .create_transform_8bit(&bt2020_profile, Layout::Rgba8)
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
            .create_transform_8bit(&bt2020_profile, Layout::GrayAlpha8)
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
            .create_transform_10bit(&srgb_profile, Layout::Rgb16)
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
            .create_transform_12bit(&srgb_profile, Layout::Rgb16)
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
            .create_transform_16bit(&srgb_profile, Layout::Rgb16)
            .unwrap();
        let src = vec![random_point_x; 256 * 256 * 3];
        let mut dst = vec![random_point_x; 256 * 256 * 3];
        transform.transform(&src, &mut dst).unwrap();
    }
}
