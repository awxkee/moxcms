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
use crate::conversions::LutBarycentricReduction;
use crate::conversions::katana::{
    Katana, KatanaInitialStage, KatanaIntermediateStage, KatanaStageLabToXyz, KatanaStageXyzToLab,
    katana_create_rgb_lin_lut, katana_input_make_lut_nx3, katana_multi_dimensional_3xn_to_device,
    katana_multi_dimensional_nx3_to_pcs, katana_output_make_lut_3xn, katana_pcs_lab_v2_to_v4,
    katana_pcs_lab_v4_to_v2, katana_prepare_inverse_lut_rgb_xyz,
};
use crate::{
    CmsError, ColorProfile, DataColorSpace, GammaLutInterpolate, Layout, LutWarehouse,
    PointeeSizeExpressible, TransformExecutor, TransformOptions,
};
use num_traits::AsPrimitive;

pub(crate) fn do_any_to_any<
    T: Copy
        + Default
        + AsPrimitive<f32>
        + Send
        + Sync
        + AsPrimitive<usize>
        + PointeeSizeExpressible
        + GammaLutInterpolate,
    const BIT_DEPTH: usize,
    const LINEAR_CAP: usize,
    const GAMMA_LUT: usize,
>(
    src_layout: Layout,
    source: &ColorProfile,
    dst_layout: Layout,
    dest: &ColorProfile,
    options: TransformOptions,
) -> Result<Box<dyn TransformExecutor<T> + Send + Sync>, CmsError>
where
    f32: AsPrimitive<T>,
    u32: AsPrimitive<T>,
    (): LutBarycentricReduction<T, u8>,
    (): LutBarycentricReduction<T, u16>,
{
    if src_layout.channels() < 3 {
        return Err(CmsError::UnsupportedProfileConnection);
    }
    if dst_layout.channels() < 3 {
        return Err(CmsError::UnsupportedProfileConnection);
    }

    let mut stages: Vec<Box<dyn KatanaIntermediateStage<f32> + Send + Sync>> = Vec::new();

    let initial_stage: Box<dyn KatanaInitialStage<f32, T> + Send + Sync> = match source
        .is_matrix_shaper()
    {
        true => {
            let state =
                katana_create_rgb_lin_lut::<T, BIT_DEPTH, LINEAR_CAP>(src_layout, source, options)?;
            stages.extend(state.stages);
            state.initial_stage
        }
        false => match source.get_device_to_pcs(options.rendering_intent).ok_or(
            CmsError::UnsupportedLutRenderingIntent(source.rendering_intent),
        )? {
            LutWarehouse::Lut(lut) => katana_input_make_lut_nx3::<T, BIT_DEPTH>(
                src_layout,
                src_layout.channels(),
                lut,
                options,
                source.pcs,
            )?,
            LutWarehouse::Multidimensional(mab) => {
                katana_multi_dimensional_nx3_to_pcs::<T, BIT_DEPTH>(
                    src_layout, mab, options, source.pcs,
                )?
            }
        },
    };

    stages.push(katana_pcs_lab_v2_to_v4(source));
    if source.pcs == DataColorSpace::Lab {
        stages.push(Box::new(KatanaStageLabToXyz::default()));
    }
    if dest.pcs == DataColorSpace::Lab {
        stages.push(Box::new(KatanaStageXyzToLab::default()));
    }
    stages.push(katana_pcs_lab_v4_to_v2(dest));

    let final_stage = if dest.has_pcs_to_device_lut() {
        let pcs_to_device = dest
            .get_pcs_to_device(options.rendering_intent)
            .ok_or(CmsError::UnsupportedProfileConnection)?;
        match pcs_to_device {
            LutWarehouse::Lut(lut) => katana_output_make_lut_3xn::<T, BIT_DEPTH>(
                dst_layout,
                lut,
                options,
                dest.color_space,
            )?,
            LutWarehouse::Multidimensional(mab) => {
                katana_multi_dimensional_3xn_to_device::<T, BIT_DEPTH>(
                    dst_layout,
                    mab,
                    options,
                    dest.pcs,
                    dest.color_space,
                )?
            }
        }
    } else if dest.is_matrix_shaper() {
        let state = katana_prepare_inverse_lut_rgb_xyz::<T, BIT_DEPTH, GAMMA_LUT>(
            dest, dst_layout, options,
        )?;
        stages.extend(state.stages);
        state.final_stage
    } else {
        return Err(CmsError::UnsupportedProfileConnection);
    };

    Ok(Box::new(Katana::<f32, T> {
        initial_stage,
        final_stage,
        stages,
    }))
}
