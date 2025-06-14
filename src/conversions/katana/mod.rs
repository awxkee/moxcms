/*
 * // Copyright (c) Radzivon Bartoshyk 6/2025. All rights reserved.
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
mod md3x3;
mod md4x3;
mod pcs_stages;
mod rgb_xyz;
mod stages;
mod xyz_lab;
mod xyz_rgb;

pub(crate) use md3x3::{multi_dimensional_3x3_to_device, multi_dimensional_3x3_to_pcs};
pub(crate) use md4x3::multi_dimensional_4x3_to_pcs;
pub(crate) use pcs_stages::{
    KatanaDefaultIntermediate, katana_pcs_lab_v2_to_v4, katana_pcs_lab_v4_to_v2,
};
pub(crate) use rgb_xyz::katana_create_rgb_lin_lut;
pub(crate) use stages::{Katana, KatanaFinalStage, KatanaInitialStage, KatanaIntermediateStage};
pub(crate) use xyz_lab::{KatanaStageLabToXyz, KatanaStageXyzToLab};
pub(crate) use xyz_rgb::katana_prepare_inverse_lut_rgb_xyz;
