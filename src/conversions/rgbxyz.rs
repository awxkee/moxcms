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
use crate::{CmsError, Layout, Matrix3f, TransformExecutor};
use num_traits::AsPrimitive;

pub(crate) struct TransformProfileRgb<T: Clone, const BUCKET: usize> {
    pub(crate) r_linear: Box<[f32; BUCKET]>,
    pub(crate) g_linear: Box<[f32; BUCKET]>,
    pub(crate) b_linear: Box<[f32; BUCKET]>,
    pub(crate) r_gamma: Box<[T; 65536]>,
    pub(crate) g_gamma: Box<[T; 65536]>,
    pub(crate) b_gamma: Box<[T; 65536]>,
    pub(crate) adaptation_matrix: Option<Matrix3f>,
}

struct TransformProfilePcsXYZRgb<
    T: Clone,
    const SRC_LAYOUT: u8,
    const DST_LAYOUT: u8,
    const LINEAR_CAP: usize,
    const GAMMA_LUT: usize,
    const BIT_DEPTH: usize,
> {
    pub(crate) profile: TransformProfileRgb<T, LINEAR_CAP>,
}

#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
macro_rules! create_rgb_xyz_dependant_executor {
    ($dep_name: ident, $dependant: ident) => {
        pub(crate) fn $dep_name<
            T: Clone + Send + Sync + AsPrimitive<usize> + Default,
            const LINEAR_CAP: usize,
            const GAMMA_LUT: usize,
            const BIT_DEPTH: usize,
        >(
            src_layout: Layout,
            dst_layout: Layout,
            profile: TransformProfileRgb<T, LINEAR_CAP>,
        ) -> Result<Box<dyn TransformExecutor<T> + Send + Sync>, CmsError>
        where
            u32: AsPrimitive<T>,
        {
            if (src_layout == Layout::Rgba) && (dst_layout == Layout::Rgba) {
                return Ok(Box::new($dependant::<
                    T,
                    { Layout::Rgba as u8 },
                    { Layout::Rgba as u8 },
                    LINEAR_CAP,
                    GAMMA_LUT,
                    BIT_DEPTH,
                > {
                    profile,
                }));
            } else if (src_layout == Layout::Rgb) && (dst_layout == Layout::Rgba) {
                return Ok(Box::new($dependant::<
                    T,
                    { Layout::Rgb as u8 },
                    { Layout::Rgba as u8 },
                    LINEAR_CAP,
                    GAMMA_LUT,
                    BIT_DEPTH,
                > {
                    profile,
                }));
            } else if (src_layout == Layout::Rgba) && (dst_layout == Layout::Rgb) {
                return Ok(Box::new($dependant::<
                    T,
                    { Layout::Rgba as u8 },
                    { Layout::Rgb as u8 },
                    LINEAR_CAP,
                    GAMMA_LUT,
                    BIT_DEPTH,
                > {
                    profile,
                }));
            } else if (src_layout == Layout::Rgb) && (dst_layout == Layout::Rgb) {
                return Ok(Box::new($dependant::<
                    T,
                    { Layout::Rgb as u8 },
                    { Layout::Rgb as u8 },
                    LINEAR_CAP,
                    GAMMA_LUT,
                    BIT_DEPTH,
                > {
                    profile,
                }));
            }
            Err(CmsError::UnsupportedProfileConnection)
        }
    };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::conversions::sse::TransformProfilePcsXYZRgbSse;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
create_rgb_xyz_dependant_executor!(
    make_rgb_xyz_rgb_transform_sse_41,
    TransformProfilePcsXYZRgbSse
);

#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
pub(crate) fn make_rgb_xyz_rgb_transform<
    T: Clone + Send + Sync + AsPrimitive<usize> + Default,
    const LINEAR_CAP: usize,
    const GAMMA_LUT: usize,
    const BIT_DEPTH: usize,
>(
    src_layout: Layout,
    dst_layout: Layout,
    profile: TransformProfileRgb<T, LINEAR_CAP>,
) -> Result<Box<dyn TransformExecutor<T> + Send + Sync>, CmsError>
where
    u32: AsPrimitive<T>,
{
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return make_rgb_xyz_rgb_transform_sse_41::<T, LINEAR_CAP, GAMMA_LUT, BIT_DEPTH>(
                src_layout, dst_layout, profile,
            );
        }
    }
    if (src_layout == Layout::Rgba) && (dst_layout == Layout::Rgba) {
        return Ok(Box::new(TransformProfilePcsXYZRgb::<
            T,
            { Layout::Rgba as u8 },
            { Layout::Rgba as u8 },
            LINEAR_CAP,
            GAMMA_LUT,
            BIT_DEPTH,
        > {
            profile,
        }));
    } else if (src_layout == Layout::Rgb) && (dst_layout == Layout::Rgba) {
        return Ok(Box::new(TransformProfilePcsXYZRgb::<
            T,
            { Layout::Rgb as u8 },
            { Layout::Rgba as u8 },
            LINEAR_CAP,
            GAMMA_LUT,
            BIT_DEPTH,
        > {
            profile,
        }));
    } else if (src_layout == Layout::Rgba) && (dst_layout == Layout::Rgb) {
        return Ok(Box::new(TransformProfilePcsXYZRgb::<
            T,
            { Layout::Rgba as u8 },
            { Layout::Rgb as u8 },
            LINEAR_CAP,
            GAMMA_LUT,
            BIT_DEPTH,
        > {
            profile,
        }));
    } else if (src_layout == Layout::Rgb) && (dst_layout == Layout::Rgb) {
        return Ok(Box::new(TransformProfilePcsXYZRgb::<
            T,
            { Layout::Rgb as u8 },
            { Layout::Rgb as u8 },
            LINEAR_CAP,
            GAMMA_LUT,
            BIT_DEPTH,
        > {
            profile,
        }));
    }
    Err(CmsError::UnsupportedProfileConnection)
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub(crate) fn make_rgb_xyz_rgb_transform<
    T: Clone + Send + Sync + AsPrimitive<usize> + Default,
    const LINEAR_CAP: usize,
    const GAMMA_LUT: usize,
    const BIT_DEPTH: usize,
>(
    src_layout: Layout,
    dst_layout: Layout,
    profile: TransformProfileRgb<T, LINEAR_CAP>,
) -> Result<Box<dyn TransformExecutor<T> + Send + Sync>, CmsError>
where
    u32: AsPrimitive<T>,
{
    use crate::conversions::neon::TransformProfilePcsXYZRgbNeon;
    if (src_layout == Layout::Rgba) && (dst_layout == Layout::Rgba) {
        return Ok(Box::new(TransformProfilePcsXYZRgbNeon::<
            T,
            { Layout::Rgba as u8 },
            { Layout::Rgba as u8 },
            LINEAR_CAP,
            GAMMA_LUT,
            BIT_DEPTH,
        > {
            profile,
        }));
    } else if (src_layout == Layout::Rgb) && (dst_layout == Layout::Rgba) {
        return Ok(Box::new(TransformProfilePcsXYZRgbNeon::<
            T,
            { Layout::Rgb as u8 },
            { Layout::Rgba as u8 },
            LINEAR_CAP,
            GAMMA_LUT,
            BIT_DEPTH,
        > {
            profile,
        }));
    } else if (src_layout == Layout::Rgba) && (dst_layout == Layout::Rgb) {
        return Ok(Box::new(TransformProfilePcsXYZRgbNeon::<
            T,
            { Layout::Rgba as u8 },
            { Layout::Rgb as u8 },
            LINEAR_CAP,
            GAMMA_LUT,
            BIT_DEPTH,
        > {
            profile,
        }));
    } else if (src_layout == Layout::Rgb) && (dst_layout == Layout::Rgb) {
        return Ok(Box::new(TransformProfilePcsXYZRgbNeon::<
            T,
            { Layout::Rgb as u8 },
            { Layout::Rgb as u8 },
            LINEAR_CAP,
            GAMMA_LUT,
            BIT_DEPTH,
        > {
            profile,
        }));
    }
    Err(CmsError::UnsupportedProfileConnection)
}

impl<
    T: Clone + AsPrimitive<usize> + Default,
    const SRC_LAYOUT: u8,
    const DST_LAYOUT: u8,
    const LINEAR_CAP: usize,
    const GAMMA_LUT: usize,
    const BIT_DEPTH: usize,
> TransformExecutor<T>
    for TransformProfilePcsXYZRgb<T, SRC_LAYOUT, DST_LAYOUT, LINEAR_CAP, GAMMA_LUT, BIT_DEPTH>
where
    u32: AsPrimitive<T>,
{
    fn transform(&self, src: &[T], dst: &mut [T]) -> Result<(), CmsError> {
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

        let transform = self.profile.adaptation_matrix.unwrap_or(Matrix3f::IDENTITY);
        let scale = (GAMMA_LUT - 1) as f32;
        let max_colors = (1 << BIT_DEPTH) - 1;

        for (src, dst) in src
            .chunks_exact(src_channels)
            .zip(dst.chunks_exact_mut(dst_channels))
        {
            let r = self.profile.r_linear[src[src_cn.r_i()].as_()];
            let g = self.profile.g_linear[src[src_cn.g_i()].as_()];
            let b = self.profile.b_linear[src[src_cn.b_i()].as_()];
            let a = if src_channels == 4 {
                f32::from_bits(src[src_cn.a_i()].as_() as u32)
            } else {
                f32::from_bits(max_colors)
            };

            let new_r = mlaf(
                0.5f32,
                mlaf(
                    mlaf(r * transform.v[0][0], g, transform.v[0][1]),
                    b,
                    transform.v[0][2],
                )
                .max(0f32)
                .min(1f32),
                scale,
            );

            let new_g = mlaf(
                0.5f32,
                mlaf(
                    mlaf(r * transform.v[1][0], g, transform.v[1][1]),
                    b,
                    transform.v[1][2],
                )
                .max(0f32)
                .min(1f32),
                scale,
            );

            let new_b = mlaf(
                0.5f32,
                mlaf(
                    mlaf(r * transform.v[2][0], g, transform.v[2][1]),
                    b,
                    transform.v[2][2],
                )
                .max(0f32)
                .min(1f32),
                scale,
            );

            dst[dst_cn.r_i()] = self.profile.r_gamma[(new_r as u16) as usize];
            dst[dst_cn.g_i()] = self.profile.g_gamma[(new_g as u16) as usize];
            dst[dst_cn.b_i()] = self.profile.b_gamma[(new_b as u16) as usize];
            if dst_channels == 4 {
                dst[dst_cn.a_i()] = a.to_bits().as_();
            }
        }

        Ok(())
    }
}
