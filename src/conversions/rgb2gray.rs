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
use crate::mlaf::mlaf;
use crate::{CmsError, Layout, TransformExecutor, Vector3f};
use num_traits::AsPrimitive;
use std::ops::Mul;

#[derive(Clone)]
pub(crate) struct ToneReproductionRgbToGray<T, const BUCKET: usize> {
    pub(crate) r_linear: Box<[f32; BUCKET]>,
    pub(crate) g_linear: Box<[f32; BUCKET]>,
    pub(crate) b_linear: Box<[f32; BUCKET]>,
    pub(crate) gray_gamma: Box<[T; 65536]>,
}

#[derive(Clone)]
struct TransformProfileRgbToGray<
    T,
    const SRC_LAYOUT: u8,
    const DST_LAYOUT: u8,
    const BUCKET: usize,
    const BIT_DEPTH: usize,
    const GAMMA_LUT: usize,
> {
    trc_box: ToneReproductionRgbToGray<T, BUCKET>,
    weights: Vector3f,
}

pub(crate) fn make_rgb_to_gray<
    T: Copy + Default + AsPrimitive<usize> + Send + Sync,
    const BUCKET: usize,
    const BIT_DEPTH: usize,
    const GAMMA_LUT: usize,
>(
    src_layout: Layout,
    dst_layout: Layout,
    trc: ToneReproductionRgbToGray<T, BUCKET>,
    weights: Vector3f,
) -> Box<dyn TransformExecutor<T> + Send + Sync>
where
    u32: AsPrimitive<T>,
{
    match src_layout {
        Layout::Rgb => match dst_layout {
            Layout::Rgb => unreachable!(),
            Layout::Rgba => unreachable!(),
            Layout::Gray => Box::new(TransformProfileRgbToGray::<
                T,
                { Layout::Rgb as u8 },
                { Layout::Gray as u8 },
                BUCKET,
                BIT_DEPTH,
                GAMMA_LUT,
            > {
                trc_box: trc,
                weights,
            }),
            Layout::GrayAlpha => Box::new(TransformProfileRgbToGray::<
                T,
                { Layout::Rgb as u8 },
                { Layout::GrayAlpha as u8 },
                BUCKET,
                BIT_DEPTH,
                GAMMA_LUT,
            > {
                trc_box: trc,
                weights,
            }),
        },
        Layout::Rgba => match dst_layout {
            Layout::Rgb => unreachable!(),
            Layout::Rgba => unreachable!(),
            Layout::Gray => Box::new(TransformProfileRgbToGray::<
                T,
                { Layout::Rgba as u8 },
                { Layout::Gray as u8 },
                BUCKET,
                BIT_DEPTH,
                GAMMA_LUT,
            > {
                trc_box: trc,
                weights,
            }),
            Layout::GrayAlpha => Box::new(TransformProfileRgbToGray::<
                T,
                { Layout::Rgba as u8 },
                { Layout::GrayAlpha as u8 },
                BUCKET,
                BIT_DEPTH,
                GAMMA_LUT,
            > {
                trc_box: trc,
                weights,
            }),
        },
        Layout::Gray => unreachable!(),
        Layout::GrayAlpha => unreachable!(),
    }
}

impl<
    T: Copy + Default + AsPrimitive<usize>,
    const SRC_LAYOUT: u8,
    const DST_LAYOUT: u8,
    const BUCKET: usize,
    const BIT_DEPTH: usize,
    const GAMMA_LUT: usize,
> TransformProfileRgbToGray<T, SRC_LAYOUT, DST_LAYOUT, BUCKET, BIT_DEPTH, GAMMA_LUT>
where
    u32: AsPrimitive<T>,
{
    #[inline(always)]
    fn transform_chunk(
        &self,
        src: &[T],
        dst: &mut [T],
        working_set: &mut [f32; 672],
        working_set_v2: &mut [f32; 672],
    ) -> Result<(), CmsError> {
        let src_cn = Layout::from(SRC_LAYOUT);
        let dst_cn = Layout::from(DST_LAYOUT);
        let src_channels = src_cn.channels();
        let dst_channels = dst_cn.channels();

        let working_set = &mut working_set[..src.len()];
        let working_set_v2 = &mut working_set_v2[..dst.len()];

        for (chunk, dst) in src
            .chunks_exact(src_channels)
            .zip(working_set.chunks_exact_mut(src_channels))
        {
            dst[0] = self.trc_box.r_linear[chunk[src_cn.r_i()].as_()];
            dst[1] = self.trc_box.g_linear[chunk[src_cn.g_i()].as_()];
            dst[2] = self.trc_box.b_linear[chunk[src_cn.b_i()].as_()];
            if src_channels == 4 {
                dst[3] = f32::from_bits(chunk[src_cn.a_i()].as_() as u32);
            }
        }

        let scale_value = (GAMMA_LUT - 1) as f32;
        let max_value = (1u32 << BIT_DEPTH) - 1;

        for (chunk, dst) in working_set
            .chunks_exact(src_channels)
            .zip(working_set_v2.chunks_exact_mut(dst_channels))
        {
            dst[0] = mlaf(
                mlaf(self.weights.v[0] * chunk[0], self.weights.v[1], chunk[1]),
                self.weights.v[2],
                chunk[2],
            )
            .min(1f32)
            .max(0f32)
            .mul(scale_value);
            if dst_channels == 2 && src_channels == 4 {
                dst[1] = chunk[1];
            } else if dst_channels == 2 {
                dst[1] = f32::from_bits(max_value);
            }
        }

        for (chunk, dst) in working_set_v2
            .chunks_exact(dst_channels)
            .zip(dst.chunks_exact_mut(dst_channels))
        {
            dst[0] = self.trc_box.gray_gamma[(chunk[0] as u16) as usize];
            if dst_channels == 2 {
                dst[1] = chunk[1].to_bits().as_();
            }
        }

        Ok(())
    }
}

impl<
    T: Copy + Default + AsPrimitive<usize>,
    const SRC_LAYOUT: u8,
    const DST_LAYOUT: u8,
    const BUCKET: usize,
    const BIT_DEPTH: usize,
    const GAMMA_LUT: usize,
> TransformExecutor<T>
    for TransformProfileRgbToGray<T, SRC_LAYOUT, DST_LAYOUT, BUCKET, BIT_DEPTH, GAMMA_LUT>
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
        let mut working_set_v2 = [0f32; 672];

        let (src_chunks, dst_chunks) = compute_chunk_sizes(672, src_channels, dst_channels);

        for (src, dst) in src
            .chunks_exact(src_chunks)
            .zip(dst.chunks_exact_mut(dst_chunks))
        {
            self.transform_chunk(src, dst, &mut working_set, &mut working_set_v2)?;
        }

        let rem = src.chunks_exact(src_chunks).remainder();
        let dst_rem = dst.chunks_exact_mut(dst_chunks).into_remainder();

        if !rem.is_empty() {
            self.transform_chunk(rem, dst_rem, &mut working_set, &mut working_set_v2)?;
        }

        Ok(())
    }
}
