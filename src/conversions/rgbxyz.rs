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
use crate::conversions::{GamutClipScaleStage, MatrixClipScaleStage, MatrixStage};
use crate::profile::RenderingIntent;
use crate::{CmsError, InPlaceStage, Layout, Matrix3f, TransformExecutor, TransformOptions};
use num_traits::AsPrimitive;

#[derive(Clone)]
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
}

pub(crate) fn make_rgb_xyz_rgb_transform<
    T: Clone + Send + Sync + AsPrimitive<usize> + Default,
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
            rendering_intent: intent,
            options,
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
            rendering_intent: intent,
            options,
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
            rendering_intent: intent,
            options,
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
        working_set: &mut [f32; 672],
    ) -> Result<(), CmsError> {
        let src_cn = Layout::from(SRC_LAYOUT);
        let src_channels = src_cn.channels();

        let dst_cn = Layout::from(DST_LAYOUT);
        let dst_channels = dst_cn.channels();

        for (chunk, dst) in src
            .chunks_exact(src_channels)
            .zip(working_set.chunks_exact_mut(src_channels))
        {
            dst[0] = self.profile.r_linear[chunk[src_cn.r_i()].as_()];
            dst[1] = self.profile.g_linear[chunk[src_cn.g_i()].as_()];
            dst[2] = self.profile.b_linear[chunk[src_cn.b_i()].as_()];
            if src_channels == 4 {
                dst[3] = f32::from_bits(chunk[src_cn.a_i()].as_() as u32);
            }
        }

        let cap_values = (GAMMA_LUT - 1) as f32;

        if let Some(transform) = self.profile.adaptation_matrix {
            let sliced = &mut working_set[..src.len()];
            let gamut_clipping_intent = self.rendering_intent == RenderingIntent::Perceptual
                || self.rendering_intent == RenderingIntent::RelativeColorimetric
                || self.rendering_intent == RenderingIntent::Saturation;

            // Check if rendering intent is adequate for gamut chroma clipping
            if gamut_clipping_intent && self.options.allow_chroma_clipping {
                let stage = MatrixStage::<SRC_LAYOUT> { matrix: transform };
                stage.transform(sliced)?;

                let stage = GamutClipScaleStage::<SRC_LAYOUT> { scale: cap_values };
                stage.transform(sliced)?;
            } else {
                let stage = MatrixClipScaleStage::<SRC_LAYOUT> {
                    matrix: transform,
                    scale: cap_values,
                };
                stage.transform(sliced)?;
            }
        }

        let max_value = ((1u32 << BIT_DEPTH) - 1).as_();

        for (chunk, dst) in working_set
            .chunks_exact(src_channels)
            .zip(dst.chunks_exact_mut(dst_channels))
        {
            dst[dst_cn.r_i()] = self.profile.r_gamma[chunk[0] as usize];
            dst[dst_cn.g_i()] = self.profile.g_gamma[chunk[1] as usize];
            dst[dst_cn.b_i()] = self.profile.b_gamma[chunk[2] as usize];
            if src_channels == 4 && dst_channels == 4 {
                dst[dst_cn.a_i()] = chunk[3].to_bits().as_();
            } else if src_channels == 3 && dst_channels == 4 {
                dst[dst_cn.a_i()] = max_value;
            }
        }

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
        let mut working_set = [0f32; 672];

        let (src_chunks, dst_chunks) = compute_chunk_sizes(672, src_channels, dst_channels);

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
