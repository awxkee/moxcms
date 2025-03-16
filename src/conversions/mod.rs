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
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "avx"))]
mod avx;
mod gray2rgb;
mod lut3x3;
mod lut3x4;
mod lut4;
mod lut_transforms;
mod mab;
#[cfg(all(target_arch = "aarch64", target_feature = "neon", feature = "neon"))]
mod neon;
mod rgb2gray;
mod rgbxyz;
mod rgbxyz_fixed;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
mod sse;
mod stages;
mod tetrahedral;
mod transform_lut3_to_3;
mod transform_lut3_to_4;

pub(crate) use gray2rgb::make_gray_to_x;
pub(crate) use lut_transforms::{CompressLut, make_lut_transform};
pub(crate) use rgb2gray::{ToneReproductionRgbToGray, make_rgb_to_gray};
pub(crate) use rgbxyz::RgbXyzFactory;
pub(crate) use rgbxyz::TransformProfileRgb;
