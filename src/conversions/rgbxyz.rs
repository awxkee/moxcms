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
use crate::conversions::chunking::compute_chunk_sizes;
use crate::conversions::stages::{
    GammaSearchFactory, GammaSearchFunction, RelativeColorMetricRgbXyz,
};
use crate::conversions::{GamutClipScaleStage, MatrixStage};
use crate::profile::RenderingIntent;
use crate::{CmsError, InPlaceStage, Layout, Matrix3f, TransformExecutor, TransformOptions};
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
    pub(crate) rendering_intent: RenderingIntent,
    pub(crate) options: TransformOptions,
    pub(crate) matrix_clip_scale_stage: Box<dyn InPlaceStage + Send + Sync>,
    pub(crate) gamma_search: Box<GammaSearchFunction<T>>,
}

fn make_clip_scale_stage<const LAYOUT: u8, const GAMMA_LUT: usize>(
    matrix: Option<Matrix3f>,
) -> Box<dyn InPlaceStage + Send + Sync> {
    let scale = (GAMMA_LUT - 1) as f32;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            use crate::conversions::avx::{MatrixClipScaleStageAvx, MatrixClipScaleStageAvxFma};
            return if std::arch::is_x86_feature_detected!("fma") {
                Box::new(MatrixClipScaleStageAvxFma::<LAYOUT> {
                    scale,
                    matrix: matrix.unwrap_or(Matrix3f::IDENTITY),
                })
            } else {
                Box::new(MatrixClipScaleStageAvx::<LAYOUT> {
                    scale,
                    matrix: matrix.unwrap_or(Matrix3f::IDENTITY),
                })
            };
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        use crate::conversions::neon::MatrixClipScaleStageNeon;
        return Box::new(MatrixClipScaleStageNeon::<LAYOUT> {
            scale,
            matrix: matrix.unwrap_or(Matrix3f::IDENTITY),
        });
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        use crate::conversions::stages::MatrixClipScaleStage;
        Box::new(MatrixClipScaleStage::<LAYOUT> {
            scale,
            matrix: matrix.unwrap_or(Matrix3f::IDENTITY),
        })
    }
}

pub(crate) fn make_rgb_xyz_rgb_transform<
    T: Clone + Send + Sync + AsPrimitive<usize> + Default + GammaSearchFactory<T>,
    const LINEAR_CAP: usize,
    const GAMMA_LUT: usize,
    const BIT_DEPTH: usize,
>(
    src_layout: Layout,
    dst_layout: Layout,
    profile: TransformProfileRgb<T, LINEAR_CAP>,
    intent: RenderingIntent,
    options: TransformOptions,
) -> Result<Box<dyn TransformExecutor<T> + Send + Sync>, CmsError>
where
    u32: AsPrimitive<T>,
{
    if (src_layout == Layout::Rgba) && (dst_layout == Layout::Rgba) {
        let matrix_clip_stage =
            make_clip_scale_stage::<{ Layout::Rgba as u8 }, GAMMA_LUT>(profile.adaptation_matrix);
        return Ok(Box::new(TransformProfilePcsXYZRgb::<
            T,
            { Layout::Rgba as u8 },
            { Layout::Rgba as u8 },
            LINEAR_CAP,
            GAMMA_LUT,
            BIT_DEPTH,
        > {
            profile,
            rendering_intent: intent,
            options,
            matrix_clip_scale_stage: matrix_clip_stage,
            gamma_search: Box::new(T::provide_rgb_gamma_search::<
                { Layout::Rgba as u8 },
                { Layout::Rgba as u8 },
                BIT_DEPTH,
            >()),
        }));
    } else if (src_layout == Layout::Rgb) && (dst_layout == Layout::Rgba) {
        let matrix_clip_stage =
            make_clip_scale_stage::<{ Layout::Rgb as u8 }, GAMMA_LUT>(profile.adaptation_matrix);
        return Ok(Box::new(TransformProfilePcsXYZRgb::<
            T,
            { Layout::Rgb as u8 },
            { Layout::Rgba as u8 },
            LINEAR_CAP,
            GAMMA_LUT,
            BIT_DEPTH,
        > {
            profile,
            rendering_intent: intent,
            options,
            matrix_clip_scale_stage: matrix_clip_stage,
            gamma_search: Box::new(T::provide_rgb_gamma_search::<
                { Layout::Rgb as u8 },
                { Layout::Rgba as u8 },
                BIT_DEPTH,
            >()),
        }));
    } else if (src_layout == Layout::Rgba) && (dst_layout == Layout::Rgb) {
        let matrix_clip_stage =
            make_clip_scale_stage::<{ Layout::Rgba as u8 }, GAMMA_LUT>(profile.adaptation_matrix);

        return Ok(Box::new(TransformProfilePcsXYZRgb::<
            T,
            { Layout::Rgba as u8 },
            { Layout::Rgb as u8 },
            LINEAR_CAP,
            GAMMA_LUT,
            BIT_DEPTH,
        > {
            profile,
            rendering_intent: intent,
            matrix_clip_scale_stage: matrix_clip_stage,
            options,
            gamma_search: Box::new(T::provide_rgb_gamma_search::<
                { Layout::Rgba as u8 },
                { Layout::Rgb as u8 },
                BIT_DEPTH,
            >()),
        }));
    } else if (src_layout == Layout::Rgb) && (dst_layout == Layout::Rgb) {
        let matrix_clip_stage =
            make_clip_scale_stage::<{ Layout::Rgb as u8 }, GAMMA_LUT>(profile.adaptation_matrix);

        return Ok(Box::new(TransformProfilePcsXYZRgb::<
            T,
            { Layout::Rgb as u8 },
            { Layout::Rgb as u8 },
            LINEAR_CAP,
            GAMMA_LUT,
            BIT_DEPTH,
        > {
            profile,
            rendering_intent: intent,
            options,
            matrix_clip_scale_stage: matrix_clip_stage,
            gamma_search: Box::new(T::provide_rgb_gamma_search::<
                { Layout::Rgb as u8 },
                { Layout::Rgb as u8 },
                BIT_DEPTH,
            >()),
        }));
    }
    Err(CmsError::UnsupportedProfileConnection)
}

impl<
    T: Clone + AsPrimitive<usize>,
    const SRC_LAYOUT: u8,
    const DST_LAYOUT: u8,
    const LINEAR_CAP: usize,
    const GAMMA_LUT: usize,
    const BIT_DEPTH: usize,
> TransformProfilePcsXYZRgb<T, SRC_LAYOUT, DST_LAYOUT, LINEAR_CAP, GAMMA_LUT, BIT_DEPTH>
where
    u32: AsPrimitive<T>,
{
    #[inline(always)]
    fn transform_chunk(
        &self,
        src: &[T],
        dst: &mut [T],
        working_set: &mut [f32; 1992],
    ) -> Result<(), CmsError> {
        let src_cn = Layout::from(SRC_LAYOUT);
        let src_channels = src_cn.channels();

        // for (chunk, dst) in src
        //     .chunks_exact(src_channels)
        //     .zip(working_set.chunks_exact_mut(src_channels))
        // {
        //     dst[0] = self.profile.r_linear[chunk[src_cn.r_i()].as_()];
        //     dst[1] = self.profile.g_linear[chunk[src_cn.g_i()].as_()];
        //     dst[2] = self.profile.b_linear[chunk[src_cn.b_i()].as_()];
        //     if src_channels == 4 {
        //         dst[3] = f32::from_bits(chunk[src_cn.a_i()].as_() as u32);
        //     }
        // }

        let cap_values = (GAMMA_LUT - 1) as f32;

        if let Some(transform) = self.profile.adaptation_matrix {
            let sliced = &mut working_set[..src.len()];

            // Check if rendering intent is adequate for gamut chroma clipping
            if self.rendering_intent == RenderingIntent::Perceptual
                && self.options.allow_chroma_clipping
            {
                let stage = MatrixStage::<SRC_LAYOUT> { matrix: transform };
                stage.transform(sliced)?;

                let stage = GamutClipScaleStage::<SRC_LAYOUT> { scale: cap_values };
                stage.transform(sliced)?;
            } else if self.rendering_intent == RenderingIntent::RelativeColorimetric
                || self.rendering_intent == RenderingIntent::Saturation
            {
                let stage = RelativeColorMetricRgbXyz::<SRC_LAYOUT> {
                    matrix: transform,
                    scale: cap_values,
                };
                stage.transform(sliced)?;
            } else {
                self.matrix_clip_scale_stage.transform(sliced)?;
            }
        }

        let search_fn = self.gamma_search.as_ref();
        search_fn(
            working_set,
            dst,
            &self.profile.r_gamma,
            &self.profile.g_gamma,
            &self.profile.b_gamma,
        );

        Ok(())
    }
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
        let mut working_set = [0f32; 1992];

        let (src_chunks, dst_chunks) = compute_chunk_sizes(1992, src_channels, dst_channels);

        for (src, dst) in src
            .chunks_exact(src_chunks)
            .zip(dst.chunks_exact_mut(dst_chunks))
        {
            self.transform_chunk(src, dst, &mut working_set)?;
        }

        let rem = src.chunks_exact(src_chunks).remainder();
        let dst_rem = dst.chunks_exact_mut(dst_chunks).into_remainder();

        if !rem.is_empty() {
            self.transform_chunk(rem, dst_rem, &mut working_set)?;
        }

        Ok(())
    }
}
