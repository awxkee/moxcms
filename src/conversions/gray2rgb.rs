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
use crate::transform::PointeeSizeExpressible;
use crate::{CmsError, Layout, TransformExecutor};
use num_traits::AsPrimitive;

#[derive(Clone)]
struct TransformGrayToRgbExecutor<
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
    T: Copy + Default + PointeeSizeExpressible + 'static + Send + Sync,
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
            Layout::Rgb => Ok(Box::new(TransformGrayToRgbExecutor::<
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
            Layout::Rgba => Ok(Box::new(TransformGrayToRgbExecutor::<
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
            Layout::Gray => Ok(Box::new(TransformGrayToRgbExecutor::<
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
            Layout::GrayAlpha => Ok(Box::new(TransformGrayToRgbExecutor::<
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
            Layout::Rgb => Ok(Box::new(TransformGrayToRgbExecutor::<
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
            Layout::Rgba => Ok(Box::new(TransformGrayToRgbExecutor::<
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
            Layout::Gray => Ok(Box::new(TransformGrayToRgbExecutor::<
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
            Layout::GrayAlpha => Ok(Box::new(TransformGrayToRgbExecutor::<
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
    T: Copy + Default + PointeeSizeExpressible + 'static,
    const SRC_LAYOUT: u8,
    const DST_LAYOUT: u8,
    const BUCKET: usize,
    const BIT_DEPTH: usize,
    const GAMMA_LUT: usize,
> TransformExecutor<T>
    for TransformGrayToRgbExecutor<T, SRC_LAYOUT, DST_LAYOUT, BUCKET, BIT_DEPTH, GAMMA_LUT>
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

        let is_gray_alpha = src_cn == Layout::GrayAlpha;

        let max_value: T = ((1u32 << BIT_DEPTH as u32) - 1u32).as_();
        let max_lut_size = (GAMMA_LUT - 1) as f32;

        for (src, dst) in src
            .chunks_exact(src_channels)
            .zip(dst.chunks_exact_mut(dst_channels))
        {
            let g = self.gray_linear[src[0]._as_usize()];
            let a = if is_gray_alpha { src[1] } else { max_value };

            let possible_value = ((g * max_lut_size).round() as u16) as usize;
            let gamma_value = self.gray_gamma[possible_value];

            dst[0] = gamma_value;
            if dst_cn == Layout::GrayAlpha {
                dst[1] = a;
            } else if dst_cn == Layout::Rgb {
                dst[1] = gamma_value;
                dst[2] = gamma_value;
            } else if dst_cn == Layout::Rgba {
                dst[1] = gamma_value;
                dst[2] = gamma_value;
                dst[3] = a;
            }
        }

        Ok(())
    }
}
