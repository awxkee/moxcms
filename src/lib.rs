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
#![allow(clippy::manual_clamp, clippy::excessive_precision)]
#![deny(unreachable_pub)]
mod chad;
mod cicp;
mod conversions;
mod err;
mod gamut;
mod lab;
mod luv;
/// One of main intent is to provide fast math available in const context
/// ULP most of the methods ~3.5
mod math;
mod matrix;
mod mlaf;
mod nd_array;
mod oklab;
mod oklch;
mod profile;
mod rgb;
mod transform;
mod trc;
mod yrg;

pub use cicp::{ChromacityTriple, ColorPrimaries, MatrixCoefficients, TransferCharacteristics};
pub use err::CmsError;
pub use gamut::{
    gamut_clip_adaptive_l0_0_5, gamut_clip_adaptive_l0_l_cusp, gamut_clip_preserve_chroma,
    gamut_clip_project_to_l_cusp,
};
pub use lab::Lab;
pub use luv::{LCh, Luv};
pub use math::{
    atan2f, atanf, cbrtf, const_hypotf, cosf, exp, expf, floor, floorf, hypotf, log, logf, pow,
    powf, rounding_div_ceil, sinf, sqrtf,
};
pub use matrix::{
    BT2020_MATRIX, Chromacity, DISPLAY_P3_MATRIX, Matrix3f, Matrix4f, SRGB_MATRIX, Vector3,
    Vector3f, Vector3i, Vector3u, Vector4, Vector4f, XyY, Xyz,
};
pub use nd_array::{Array3D, Array4D};
pub use oklab::Oklab;
pub use oklch::Oklch;
pub use profile::{
    CicpProfile, ColorProfile, DataColorSpace, LutMType, LutType, LutWarehouse, ProfileClass,
    RenderingIntent,
};
pub use rgb::Rgb;
pub use transform::{
    InPlaceStage, Layout, Stage, Transform8BitExecutor, Transform16BitExecutor, TransformExecutor,
    TransformOptions,
};
pub use yrg::{Ych, Yrg, cie_y_1931_to_cie_y_2006};
