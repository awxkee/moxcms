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
use num_traits::AsPrimitive;

#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
pub(crate) struct MatrixClipScaleStage<const LAYOUT: u8> {
    pub(crate) matrix: Matrix3f,
    pub(crate) scale: f32,
}

#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
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

            chunk[1] = mlaf(
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

            chunk[2] = mlaf(
                0.5f32,
                mlaf(
                    mlaf(r * transform.v[2][0], g, transform.v[2][1]),
                    b,
                    transform.v[2][2],
                )
                .max(0f32)
                .min(1f32),
                scale,
            )
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
            rgb = rgb.clamp(0.0, 1.0) * Rgb::dup(self.scale) + Rgb::dup(0.5f32);
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
            new_rgb += 0.5f32;

            chunk[0] = new_rgb.r;
            chunk[1] = new_rgb.g;
            chunk[2] = new_rgb.b;
        }

        Ok(())
    }
}

pub(crate) type GammaSearchRgbFunction<T> =
    fn(&[f32], &mut [T], &[T; 65536], &[T; 65536], &[T; 65536]);

pub(crate) type LinearSearchRgbFunction<T, const CAP: usize> =
    fn(&[T], &mut [f32], &Box<[f32; CAP]>, &Box<[f32; CAP]>, &Box<[f32; CAP]>);

fn gamma_search<
    T: Copy + 'static,
    const SRC_LAYOUT: u8,
    const DST_LAYOUT: u8,
    const BIT_DEPTH: usize,
>(
    working_set: &[f32],
    dst: &mut [T],
    r_gamma: &[T; 65536],
    g_gamma: &[T; 65536],
    b_gamma: &[T; 65536],
) where
    u32: AsPrimitive<T>,
{
    let max_value = ((1u32 << BIT_DEPTH) - 1).as_();
    let src_cn = Layout::from(SRC_LAYOUT);
    let src_channels = src_cn.channels();

    let dst_cn = Layout::from(DST_LAYOUT);
    let dst_channels = dst_cn.channels();
    for (chunk, dst) in working_set
        .chunks_exact(src_channels)
        .zip(dst.chunks_exact_mut(dst_channels))
    {
        let r = (chunk[0] as u16) as usize;
        let g = (chunk[1] as u16) as usize;
        let b = (chunk[2] as u16) as usize;
        dst[dst_cn.r_i()] = r_gamma[r];
        dst[dst_cn.g_i()] = g_gamma[g];
        dst[dst_cn.b_i()] = b_gamma[b];
        if src_channels == 4 && dst_channels == 4 {
            dst[dst_cn.a_i()] = chunk[3].to_bits().as_();
        } else if src_channels == 3 && dst_channels == 4 {
            dst[dst_cn.a_i()] = max_value;
        }
    }
}

pub(crate) trait GammaSearchFactory<T> {
    fn provide_rgb_gamma_search<
        const SRC_LAYOUT: u8,
        const DST_LAYOUT: u8,
        const BIT_DEPTH: usize,
    >() -> GammaSearchRgbFunction<T>;

    fn provide_rgb_linear_search<const CAP: usize, const SRC_LAYOUT: u8, const BIT_DEPTH: usize>()
    -> LinearSearchRgbFunction<T, CAP>;
}

impl GammaSearchFactory<u8> for u8 {
    fn provide_rgb_gamma_search<
        const SRC_LAYOUT: u8,
        const DST_LAYOUT: u8,
        const BIT_DEPTH: usize,
    >() -> GammaSearchRgbFunction<u8> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::arch::is_x86_feature_detected!("sse4.1") {
                use crate::conversions::sse::gamma_search_8bit;
                return gamma_search_8bit::<SRC_LAYOUT, DST_LAYOUT>;
            }
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            use crate::conversions::neon::gamma_search_8bit;
            return gamma_search_8bit::<SRC_LAYOUT, DST_LAYOUT>;
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
        gamma_search::<u8, SRC_LAYOUT, DST_LAYOUT, BIT_DEPTH>
    }

    fn provide_rgb_linear_search<const CAP: usize, const SRC_LAYOUT: u8, const BIT_DEPTH: usize>()
    -> LinearSearchRgbFunction<u8, CAP> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::arch::is_x86_feature_detected!("sse4.1") {
                use crate::conversions::sse::linear_search_rgb8;
                return linear_search_rgb8::<CAP, SRC_LAYOUT>;
            }
        }
        linear_search_rgb::<u8, CAP, SRC_LAYOUT, BIT_DEPTH>
    }
}

impl GammaSearchFactory<u16> for u16 {
    fn provide_rgb_gamma_search<
        const SRC_LAYOUT: u8,
        const DST_LAYOUT: u8,
        const BIT_DEPTH: usize,
    >() -> GammaSearchRgbFunction<u16> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::arch::is_x86_feature_detected!("sse4.1") {
                use crate::conversions::sse::gamma_search_16bit;
                return gamma_search_16bit::<SRC_LAYOUT, DST_LAYOUT, BIT_DEPTH>;
            }
        }
        gamma_search::<u16, SRC_LAYOUT, DST_LAYOUT, BIT_DEPTH>
    }

    fn provide_rgb_linear_search<const CAP: usize, const SRC_LAYOUT: u8, const BIT_DEPTH: usize>()
    -> LinearSearchRgbFunction<u16, CAP> {
        linear_search_rgb::<u16, CAP, SRC_LAYOUT, BIT_DEPTH>
    }
}

fn linear_search_rgb<
    T: Copy + 'static + AsPrimitive<usize>,
    const CAP: usize,
    const SRC_LAYOUT: u8,
    const BIT_DEPTH: usize,
>(
    src: &[T],
    working_set: &mut [f32],
    r_linear: &Box<[f32; CAP]>,
    g_linear: &Box<[f32; CAP]>,
    b_linear: &Box<[f32; CAP]>,
) where
    u32: AsPrimitive<T>,
{
    let src_cn = Layout::from(SRC_LAYOUT);
    let src_channels = src_cn.channels();
    if src_channels == 4 {
        for (chunk, dst) in src
            .chunks_exact(src_channels * 3)
            .zip(working_set.chunks_exact_mut(src_channels * 3))
        {
            let r0 = chunk[src_cn.r_i()].as_();
            let g0 = chunk[src_cn.g_i()].as_();
            let b0 = chunk[src_cn.b_i()].as_();
            let a0 = chunk[src_cn.a_i()].as_();

            let r1 = chunk[src_cn.r_i() + 4].as_();
            let g1 = chunk[src_cn.g_i() + 4].as_();
            let b1 = chunk[src_cn.b_i() + 4].as_();
            let a1 = chunk[src_cn.a_i() + 4].as_();

            let r2 = chunk[src_cn.r_i() + 8].as_();
            let g2 = chunk[src_cn.g_i() + 8].as_();
            let b2 = chunk[src_cn.b_i() + 8].as_();
            let a2 = chunk[src_cn.a_i() + 8].as_();

            dst[0] = r_linear[r0];
            dst[1] = g_linear[g0];
            dst[2] = b_linear[b0];
            dst[3] = f32::from_bits(a0 as u32);

            dst[4] = r_linear[r1];
            dst[5] = g_linear[g1];
            dst[6] = b_linear[b1];
            dst[7] = f32::from_bits(a1 as u32);

            dst[8] = r_linear[r2];
            dst[9] = g_linear[g2];
            dst[10] = b_linear[b2];
            dst[11] = f32::from_bits(a2 as u32);
        }

        let src = src.chunks_exact(src_channels * 3).remainder();
        let working_set = working_set
            .chunks_exact_mut(src_channels * 3)
            .into_remainder();

        for (chunk, dst) in src
            .chunks_exact(src_channels)
            .zip(working_set.chunks_exact_mut(src_channels))
        {
            let r0 = chunk[src_cn.r_i()].as_();
            let g0 = chunk[src_cn.g_i()].as_();
            let b0 = chunk[src_cn.b_i()].as_();
            let a0 = chunk[src_cn.a_i()].as_();

            dst[0] = r_linear[r0];
            dst[1] = g_linear[g0];
            dst[2] = b_linear[b0];
            dst[3] = f32::from_bits(a0 as u32);
        }
    } else if src_channels == 3 {
        for (chunk, dst) in src
            .chunks_exact(src_channels * 3)
            .zip(working_set.chunks_exact_mut(src_channels * 3))
        {
            let r0 = chunk[src_cn.r_i()].as_();
            let g0 = chunk[src_cn.g_i()].as_();
            let b0 = chunk[src_cn.b_i()].as_();

            let r1 = chunk[src_cn.r_i() + 3].as_();
            let g1 = chunk[src_cn.g_i() + 3].as_();
            let b1 = chunk[src_cn.b_i() + 3].as_();

            let r2 = chunk[src_cn.r_i() + 6].as_();
            let g2 = chunk[src_cn.g_i() + 6].as_();
            let b2 = chunk[src_cn.b_i() + 6].as_();

            dst[0] = r_linear[r0];
            dst[1] = g_linear[g0];
            dst[2] = b_linear[b0];

            dst[3] = r_linear[r1];
            dst[4] = g_linear[g1];
            dst[5] = b_linear[b1];

            dst[6] = r_linear[r2];
            dst[7] = g_linear[g2];
            dst[8] = b_linear[b2];
        }

        let src = src.chunks_exact(src_channels * 3).remainder();
        let working_set = working_set
            .chunks_exact_mut(src_channels * 3)
            .into_remainder();

        for (chunk, dst) in src
            .chunks_exact(src_channels)
            .zip(working_set.chunks_exact_mut(src_channels))
        {
            let r = chunk[src_cn.r_i()].as_();
            let g = chunk[src_cn.g_i()].as_();
            let b = chunk[src_cn.b_i()].as_();
            dst[0] = r_linear[r];
            dst[1] = g_linear[g];
            dst[2] = b_linear[b];
        }
    }
}
