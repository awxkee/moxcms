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
use crate::Layout;
use crate::conversions::TransformProfileRgb;
use crate::matrix::Matrix3;
use crate::{CmsError, TransformExecutor};

/// Fixed point conversion for 8-bit
pub(crate) struct TransformProfileRgb8Bit {
    pub(crate) r_linear: Box<[i16; 256]>,
    pub(crate) g_linear: Box<[i16; 256]>,
    pub(crate) b_linear: Box<[i16; 256]>,
    pub(crate) r_gamma: Box<[u8; 65536]>,
    pub(crate) g_gamma: Box<[u8; 65536]>,
    pub(crate) b_gamma: Box<[u8; 65536]>,
    pub(crate) adaptation_matrix: Matrix3<i16>,
}

#[allow(unused)]
struct TransformProfilePcsXYZRgbQ4_12<
    const SRC_LAYOUT: u8,
    const DST_LAYOUT: u8,
    const LINEAR_CAP: usize,
    const GAMMA_LUT: usize,
> {
    pub(crate) profile: TransformProfileRgb8Bit,
}

#[allow(unused)]
impl<const SRC_LAYOUT: u8, const DST_LAYOUT: u8, const LINEAR_CAP: usize, const GAMMA_LUT: usize>
    TransformExecutor<u8>
    for TransformProfilePcsXYZRgbQ4_12<SRC_LAYOUT, DST_LAYOUT, LINEAR_CAP, GAMMA_LUT>
{
    fn transform(&self, src: &[u8], dst: &mut [u8]) -> Result<(), CmsError> {
        let src_cn = Layout::from(SRC_LAYOUT);
        let dst_cn = Layout::from(DST_LAYOUT);
        let src_channels = src_cn.channels();
        let dst_channels = dst_cn.channels();

        if src.len() / src_channels != dst.len() / dst_channels {
            return Err(CmsError::LaneSizeMismatch);
        }
        if src.len() % src_channels != 0 {
            return Err(CmsError::LaneMultipleOfChannels);
        }
        if dst.len() % dst_channels != 0 {
            return Err(CmsError::LaneMultipleOfChannels);
        }

        let transform = self.profile.adaptation_matrix;
        let max_colors: u8 = 255;
        const ROUNDING_Q4_12: i32 = (1 << (12 - 1)) - 1;
        const Q: i32 = 12;

        for (src, dst) in src
            .chunks_exact(src_channels)
            .zip(dst.chunks_exact_mut(dst_channels))
        {
            let r = self.profile.r_linear[src[src_cn.r_i()] as usize];
            let g = self.profile.g_linear[src[src_cn.g_i()] as usize];
            let b = self.profile.b_linear[src[src_cn.b_i()] as usize];
            let a = if src_channels == 4 {
                src[src_cn.a_i()]
            } else {
                max_colors
            };

            let new_r = r as i32 * transform.v[0][0] as i32
                + g as i32 * transform.v[0][1] as i32
                + b as i32 * transform.v[0][2] as i32
                + ROUNDING_Q4_12;

            let r_q4_12 = (new_r >> Q).min(4095).max(0) as u16;

            let new_g = r as i32 * transform.v[1][0] as i32
                + g as i32 * transform.v[1][1] as i32
                + b as i32 * transform.v[1][2] as i32
                + ROUNDING_Q4_12;

            let g_q4_12 = (new_g >> Q).min(4095).max(0) as u16;

            let new_b = r as i32 * transform.v[2][0] as i32
                + g as i32 * transform.v[2][1] as i32
                + b as i32 * transform.v[2][2] as i32
                + ROUNDING_Q4_12;

            let b_q4_12 = (new_b >> Q).min(4095).max(0) as u16;

            dst[dst_cn.r_i()] = self.profile.r_gamma[r_q4_12 as usize];
            dst[dst_cn.g_i()] = self.profile.g_gamma[g_q4_12 as usize];
            dst[dst_cn.b_i()] = self.profile.b_gamma[b_q4_12 as usize];
            if dst_channels == 4 {
                dst[dst_cn.a_i()] = a;
            }
        }
        Ok(())
    }
}

macro_rules! create_rgb_xyz_dependant_8bit_executor {
    ($dep_name: ident, $dependant: ident) => {
        pub(crate) fn $dep_name<const LINEAR_CAP: usize, const GAMMA_LUT: usize>(
            src_layout: Layout,
            dst_layout: Layout,
            profile: TransformProfileRgb<u8, LINEAR_CAP>,
        ) -> Result<Box<dyn TransformExecutor<u8> + Send + Sync>, CmsError> {
            let q4_12_profile = profile.to_q4_12();
            if (src_layout == Layout::Rgba) && (dst_layout == Layout::Rgba) {
                return Ok(Box::new($dependant::<
                    { Layout::Rgba as u8 },
                    { Layout::Rgba as u8 },
                    LINEAR_CAP,
                    GAMMA_LUT,
                > {
                    profile: q4_12_profile,
                }));
            } else if (src_layout == Layout::Rgb) && (dst_layout == Layout::Rgba) {
                return Ok(Box::new($dependant::<
                    { Layout::Rgb as u8 },
                    { Layout::Rgba as u8 },
                    LINEAR_CAP,
                    GAMMA_LUT,
                > {
                    profile: q4_12_profile,
                }));
            } else if (src_layout == Layout::Rgba) && (dst_layout == Layout::Rgb) {
                return Ok(Box::new($dependant::<
                    { Layout::Rgba as u8 },
                    { Layout::Rgb as u8 },
                    LINEAR_CAP,
                    GAMMA_LUT,
                > {
                    profile: q4_12_profile,
                }));
            } else if (src_layout == Layout::Rgb) && (dst_layout == Layout::Rgb) {
                return Ok(Box::new($dependant::<
                    { Layout::Rgb as u8 },
                    { Layout::Rgb as u8 },
                    LINEAR_CAP,
                    GAMMA_LUT,
                > {
                    profile: q4_12_profile,
                }));
            }
            Err(CmsError::UnsupportedProfileConnection)
        }
    };
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon", feature = "neon"))]
use crate::conversions::neon::TransformProfileRgb8BitNeon;

#[cfg(all(target_arch = "aarch64", target_feature = "neon", feature = "neon"))]
create_rgb_xyz_dependant_8bit_executor!(make_8bit_rgb_xyz, TransformProfileRgb8BitNeon);

#[cfg(not(all(target_arch = "aarch64", target_feature = "neon", feature = "neon")))]
create_rgb_xyz_dependant_8bit_executor!(make_8bit_rgb_xyz, TransformProfilePcsXYZRgbQ4_12);

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
use crate::conversions::sse::TransformProfileRgb8BitSse;

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
create_rgb_xyz_dependant_8bit_executor!(
    make_rgb_xyz_q4_12_transform_sse_41,
    TransformProfileRgb8BitSse
);

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "avx"))]
use crate::conversions::avx::TransformProfilePcsXYZRgb8BitAvx;

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "avx"))]
create_rgb_xyz_dependant_8bit_executor!(
    make_rgb_xyz_q4_12_transform_avx2,
    TransformProfilePcsXYZRgb8BitAvx
);
