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
use crate::{CmsError, Layout, TransformExecutor};
use num_traits::AsPrimitive;

#[derive(Clone)]
struct TransformProfileGrayToRgb<
    T,
    const SRC_LAYOUT: u8,
    const DEST_LAYOUT: u8,
    const BUCKET: usize,
    const BIT_DEPTH: usize,
    const GAMMA_LUT: usize,
> {
    gray_linear: Box<[f32; BUCKET]>,
    gray_gamma: Box<[T; 65536]>,
}

pub(crate) fn make_gray_to_x<
    T: Copy + Default + AsPrimitive<usize> + Send + Sync,
    const BUCKET: usize,
    const BIT_DEPTH: usize,
    const GAMMA_LUT: usize,
>(
    src_layout: Layout,
    dst_layout: Layout,
    gray_linear: Box<[f32; BUCKET]>,
    gray_gamma: Box<[T; 65536]>,
) -> Result<Box<dyn TransformExecutor<T> + Sync + Send>, CmsError>
where
    u32: AsPrimitive<T>,
{
    if src_layout != Layout::Gray && src_layout != Layout::GrayAlpha {
        return Err(CmsError::UnsupportedProfileConnection);
    }
    match src_layout {
        Layout::Rgb => unreachable!(),
        Layout::Rgba => unreachable!(),
        Layout::Gray => match dst_layout {
            Layout::Rgb => Ok(Box::new(TransformProfileGrayToRgb::<
                T,
                { Layout::Gray as u8 },
                { Layout::Rgb as u8 },
                BUCKET,
                BIT_DEPTH,
                GAMMA_LUT,
            > {
                gray_linear,
                gray_gamma,
            })),
            Layout::Rgba => Ok(Box::new(TransformProfileGrayToRgb::<
                T,
                { Layout::Gray as u8 },
                { Layout::Rgba as u8 },
                BUCKET,
                BIT_DEPTH,
                GAMMA_LUT,
            > {
                gray_linear,
                gray_gamma,
            })),
            Layout::Gray => Ok(Box::new(TransformProfileGrayToRgb::<
                T,
                { Layout::Gray as u8 },
                { Layout::Gray as u8 },
                BUCKET,
                BIT_DEPTH,
                GAMMA_LUT,
            > {
                gray_linear,
                gray_gamma,
            })),
            Layout::GrayAlpha => Ok(Box::new(TransformProfileGrayToRgb::<
                T,
                { Layout::Gray as u8 },
                { Layout::GrayAlpha as u8 },
                BUCKET,
                BIT_DEPTH,
                GAMMA_LUT,
            > {
                gray_linear,
                gray_gamma,
            })),
        },
        Layout::GrayAlpha => match dst_layout {
            Layout::Rgb => Ok(Box::new(TransformProfileGrayToRgb::<
                T,
                { Layout::Gray as u8 },
                { Layout::GrayAlpha as u8 },
                BUCKET,
                BIT_DEPTH,
                GAMMA_LUT,
            > {
                gray_linear,
                gray_gamma,
            })),
            Layout::Rgba => Ok(Box::new(TransformProfileGrayToRgb::<
                T,
                { Layout::Gray as u8 },
                { Layout::Rgba as u8 },
                BUCKET,
                BIT_DEPTH,
                GAMMA_LUT,
            > {
                gray_linear,
                gray_gamma,
            })),
            Layout::Gray => Ok(Box::new(TransformProfileGrayToRgb::<
                T,
                { Layout::Gray as u8 },
                { Layout::Gray as u8 },
                BUCKET,
                BIT_DEPTH,
                GAMMA_LUT,
            > {
                gray_linear,
                gray_gamma,
            })),
            Layout::GrayAlpha => Ok(Box::new(TransformProfileGrayToRgb::<
                T,
                { Layout::GrayAlpha as u8 },
                { Layout::GrayAlpha as u8 },
                BUCKET,
                BIT_DEPTH,
                GAMMA_LUT,
            > {
                gray_linear,
                gray_gamma,
            })),
        },
    }
}

impl<
    T: Copy + Default + AsPrimitive<usize>,
    const SRC_LAYOUT: u8,
    const DST_LAYOUT: u8,
    const BUCKET: usize,
    const BIT_DEPTH: usize,
    const GAMMA_LUT: usize,
> TransformProfileGrayToRgb<T, SRC_LAYOUT, DST_LAYOUT, BUCKET, BIT_DEPTH, GAMMA_LUT>
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
        let dst_cn = Layout::from(DST_LAYOUT);
        let src_channels = src_cn.channels();
        let dst_channels = dst_cn.channels();

        let is_gray_alpha = src_cn == Layout::GrayAlpha;

        for (chunk, dst) in src
            .chunks_exact(src_channels)
            .zip(working_set.chunks_exact_mut(src_channels))
        {
            dst[0] = self.gray_linear[chunk[0].as_()];
            if is_gray_alpha {
                dst[1] = f32::from_bits(chunk[1].as_() as u32);
            }
        }

        let max_value: T = ((1u32 << BIT_DEPTH as u32) - 1u32).as_();
        let max_lut_size = (GAMMA_LUT - 1) as f32;

        for (chunk, dst) in working_set
            .chunks_exact(src_channels)
            .zip(dst.chunks_exact_mut(dst_channels))
        {
            let possible_value = (chunk[0] * max_lut_size).round() as usize;
            let gamma_value = self.gray_gamma[possible_value];

            let alpha_value = if src_cn == Layout::GrayAlpha {
                (chunk[1] as u32).as_()
            } else {
                max_value
            };

            dst[0] = gamma_value;
            if dst_cn == Layout::GrayAlpha {
                dst[1] = alpha_value;
            } else if dst_cn == Layout::Rgb {
                dst[1] = gamma_value;
                dst[2] = gamma_value;
            } else if dst_cn == Layout::Rgba {
                dst[1] = gamma_value;
                dst[2] = gamma_value;
                dst[3] = alpha_value;
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
    for TransformProfileGrayToRgb<T, SRC_LAYOUT, DST_LAYOUT, BUCKET, BIT_DEPTH, GAMMA_LUT>
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
