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
use num_traits::AsPrimitive;

/// Fixed point conversion for 8-bit/10-bit
pub(crate) struct TransformProfileRgbFixedPoint<R, T, const LINEAR_CAP: usize> {
    pub(crate) r_linear: Box<[R; LINEAR_CAP]>,
    pub(crate) g_linear: Box<[R; LINEAR_CAP]>,
    pub(crate) b_linear: Box<[R; LINEAR_CAP]>,
    pub(crate) r_gamma: Box<[T; 65536]>,
    pub(crate) g_gamma: Box<[T; 65536]>,
    pub(crate) b_gamma: Box<[T; 65536]>,
    pub(crate) adaptation_matrix: Matrix3<i16>,
}

#[allow(unused)]
struct TransformProfilePcsXYZRgbQ4_12<
    T: Copy,
    const SRC_LAYOUT: u8,
    const DST_LAYOUT: u8,
    const LINEAR_CAP: usize,
    const GAMMA_LUT: usize,
    const BIT_DEPTH: usize,
    const PRECISION: i32,
> {
    pub(crate) profile: TransformProfileRgbFixedPoint<i16, T, LINEAR_CAP>,
}

#[allow(unused)]
impl<
    T: Clone + PointeeSizeExpressible + Copy + Default + 'static,
    const SRC_LAYOUT: u8,
    const DST_LAYOUT: u8,
    const LINEAR_CAP: usize,
    const GAMMA_LUT: usize,
    const BIT_DEPTH: usize,
    const PRECISION: i32,
> TransformExecutor<T>
    for TransformProfilePcsXYZRgbQ4_12<
        T,
        SRC_LAYOUT,
        DST_LAYOUT,
        LINEAR_CAP,
        GAMMA_LUT,
        BIT_DEPTH,
        PRECISION,
    >
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

        let transform = self.profile.adaptation_matrix;
        let max_colors: T = ((1 << BIT_DEPTH as u32) - 1u32).as_();
        let rnd: i32 = (1 << (PRECISION - 1)) - 1;

        let v_gamma_max = GAMMA_LUT as i32 - 1;

        for (src, dst) in src
            .chunks_exact(src_channels)
            .zip(dst.chunks_exact_mut(dst_channels))
        {
            let r = self.profile.r_linear[src[src_cn.r_i()]._as_usize()];
            let g = self.profile.g_linear[src[src_cn.g_i()]._as_usize()];
            let b = self.profile.b_linear[src[src_cn.b_i()]._as_usize()];
            let a = if src_channels == 4 {
                src[src_cn.a_i()]
            } else {
                max_colors
            };

            let new_r = r as i32 * transform.v[0][0] as i32
                + g as i32 * transform.v[0][1] as i32
                + b as i32 * transform.v[0][2] as i32
                + rnd;

            let r_q4_12 = (new_r >> PRECISION).min(v_gamma_max).max(0) as u16;

            let new_g = r as i32 * transform.v[1][0] as i32
                + g as i32 * transform.v[1][1] as i32
                + b as i32 * transform.v[1][2] as i32
                + rnd;

            let g_q4_12 = (new_g >> PRECISION).min(v_gamma_max).max(0) as u16;

            let new_b = r as i32 * transform.v[2][0] as i32
                + g as i32 * transform.v[2][1] as i32
                + b as i32 * transform.v[2][2] as i32
                + rnd;

            let b_q4_12 = (new_b >> PRECISION).min(v_gamma_max).max(0) as u16;

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

macro_rules! create_rgb_xyz_dependant_q4_12_executor {
    ($dep_name: ident, $dependant: ident, $resolution: ident) => {
        pub(crate) fn $dep_name<
            T: Clone + Send + Sync + AsPrimitive<usize> + Default + PointeeSizeExpressible,
            const LINEAR_CAP: usize,
            const GAMMA_LUT: usize,
            const BIT_DEPTH: usize,
            const PRECISION: i32,
        >(
            src_layout: Layout,
            dst_layout: Layout,
            profile: TransformProfileRgb<T, LINEAR_CAP>,
        ) -> Result<Box<dyn TransformExecutor<T> + Send + Sync>, CmsError>
        where
            u32: AsPrimitive<T>,
        {
            let q4_12_profile =
                profile.to_q4_12_n::<$resolution, PRECISION, LINEAR_CAP, GAMMA_LUT, BIT_DEPTH>();
            if (src_layout == Layout::Rgba) && (dst_layout == Layout::Rgba) {
                return Ok(Box::new($dependant::<
                    T,
                    { Layout::Rgba as u8 },
                    { Layout::Rgba as u8 },
                    LINEAR_CAP,
                    GAMMA_LUT,
                    BIT_DEPTH,
                    PRECISION,
                > {
                    profile: q4_12_profile,
                }));
            } else if (src_layout == Layout::Rgb) && (dst_layout == Layout::Rgba) {
                return Ok(Box::new($dependant::<
                    T,
                    { Layout::Rgb as u8 },
                    { Layout::Rgba as u8 },
                    LINEAR_CAP,
                    GAMMA_LUT,
                    BIT_DEPTH,
                    PRECISION,
                > {
                    profile: q4_12_profile,
                }));
            } else if (src_layout == Layout::Rgba) && (dst_layout == Layout::Rgb) {
                return Ok(Box::new($dependant::<
                    T,
                    { Layout::Rgba as u8 },
                    { Layout::Rgb as u8 },
                    LINEAR_CAP,
                    GAMMA_LUT,
                    BIT_DEPTH,
                    PRECISION,
                > {
                    profile: q4_12_profile,
                }));
            } else if (src_layout == Layout::Rgb) && (dst_layout == Layout::Rgb) {
                return Ok(Box::new($dependant::<
                    T,
                    { Layout::Rgb as u8 },
                    { Layout::Rgb as u8 },
                    LINEAR_CAP,
                    GAMMA_LUT,
                    BIT_DEPTH,
                    PRECISION,
                > {
                    profile: q4_12_profile,
                }));
            }
            Err(CmsError::UnsupportedProfileConnection)
        }
    };
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon", feature = "neon"))]
use crate::conversions::neon::TransformProfileRgbQ12Neon;

#[cfg(all(target_arch = "aarch64", target_feature = "neon", feature = "neon"))]
create_rgb_xyz_dependant_q4_12_executor!(make_rgb_xyz_q4_12, TransformProfileRgbQ12Neon, i16);

#[cfg(not(all(target_arch = "aarch64", target_feature = "neon", feature = "neon")))]
create_rgb_xyz_dependant_q4_12_executor!(make_rgb_xyz_q4_12, TransformProfilePcsXYZRgbQ4_12, i16);

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
use crate::conversions::sse::TransformProfileRgbQ12Sse;

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
create_rgb_xyz_dependant_q4_12_executor!(
    make_rgb_xyz_q4_12_transform_sse_41,
    TransformProfileRgbQ12Sse,
    i32
);

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "avx"))]
use crate::conversions::avx::TransformProfilePcsXYZRgbQ12Avx;
use crate::transform::PointeeSizeExpressible;

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "avx"))]
create_rgb_xyz_dependant_q4_12_executor!(
    make_rgb_xyz_q4_12_transform_avx2,
    TransformProfilePcsXYZRgbQ12Avx,
    i32
);
