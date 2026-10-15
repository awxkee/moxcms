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
#![cfg(feature = "lut")]
use crate::conversions::lut3x3::{create_lut3x3, stage_lut_3x3};
#[cfg(feature = "any_to_any")]
use crate::conversions::lut3x3::{katana_input_stage_lut_3x3, katana_output_stage_lut_3x3};
use crate::conversions::lut3x4::{create_lut3_samples_norm, create_lut3x4};
use crate::conversions::lut4::*;
use crate::conversions::mab::{prepare_mab_3x3, prepare_mba_3x3};
use crate::conversions::transform_lut3_to_4::make_transform_3x4;
use crate::err::try_vec;
use crate::mlaf::mlaf;
use crate::{
    CmsError, ColorProfile, DataColorSpace, InPlaceStage, Layout, LutWarehouse, Matrix3f,
    ProfileVersion, RenderingIntent, Stage, TransformExecutor, TransformOptions,
};
use num_traits::AsPrimitive;
use std::sync::Arc;

pub(crate) struct MatrixStage {
    pub(crate) matrices: Vec<Matrix3f>,
}

impl InPlaceStage for MatrixStage {
    fn transform(&self, dst: &mut [f32]) -> Result<(), CmsError> {
        if !self.matrices.is_empty() {
            let m = self.matrices[0];
            for dst in dst.as_chunks_mut::<3>().0.iter_mut() {
                let x = dst[0];
                let y = dst[1];
                let z = dst[2];
                dst[0] = mlaf(mlaf(x * m.v[0][0], y, m.v[0][1]), z, m.v[0][2]);
                dst[1] = mlaf(mlaf(x * m.v[1][0], y, m.v[1][1]), z, m.v[1][2]);
                dst[2] = mlaf(mlaf(x * m.v[2][0], y, m.v[2][1]), z, m.v[2][2]);
            }
        }

        for m in self.matrices.iter().skip(1) {
            for dst in dst.as_chunks_mut::<3>().0.iter_mut() {
                let x = dst[0];
                let y = dst[1];
                let z = dst[2];
                dst[0] = mlaf(mlaf(x * m.v[0][0], y, m.v[0][1]), z, m.v[0][2]);
                dst[1] = mlaf(mlaf(x * m.v[1][0], y, m.v[1][1]), z, m.v[1][2]);
                dst[2] = mlaf(mlaf(x * m.v[2][0], y, m.v[2][1]), z, m.v[2][2]);
            }
        }

        Ok(())
    }
}

pub(crate) const LUT_SAMPLING: u16 = 255;

pub(crate) trait Lut3x3Factory {
    fn make_transform_3x3<
        T: Copy + AsPrimitive<f32> + Default + PointeeSizeExpressible + 'static + Send + Sync,
        const SRC_LAYOUT: u8,
        const DST_LAYOUT: u8,
        const GRID_SIZE: usize,
        const BIT_DEPTH: usize,
        const BINS: usize,
    >(
        lut: Vec<f32>,
        options: TransformOptions,
        color_space: DataColorSpace,
        is_linear: bool,
    ) -> Arc<dyn TransformExecutor<T> + Send + Sync>
    where
        f32: AsPrimitive<T>,
        u32: AsPrimitive<T>,
        (): LutBarycentricReduction<T, u8>,
        (): LutBarycentricReduction<T, u16>;
}

/// Fills the outputs of the nodes from the first given, three a node.
pub(crate) type FillNodes<'a> = Box<dyn FnMut(usize, &mut [f32]) -> Result<(), CmsError> + 'a>;

/// A 4D grid of three outputs a node, either whole or made a plane of the last input at a time,
/// so that a transform can lay it out in its own layout without holding it twice.
pub(crate) enum Grid4x3<'a> {
    Whole(Vec<f32>),
    Planes {
        nodes: usize,
        plane: usize,
        fill: FillNodes<'a>,
    },
}

impl Grid4x3<'_> {
    /// Each node's outputs mapped into a transform's layout.
    pub(crate) fn collect<E>(self, map: impl Fn(&[f32; 3]) -> E) -> Result<Vec<E>, CmsError> {
        match self {
            Grid4x3::Whole(lut) => Ok(lut.as_chunks::<3>().0.iter().map(map).collect()),
            Grid4x3::Planes {
                nodes,
                plane,
                mut fill,
            } => {
                let mut out = Vec::new();
                out.try_reserve_exact(nodes)
                    .map_err(|_| CmsError::OutOfMemory(nodes))?;
                let mut part = try_vec![0f32; plane * 3];
                for first in (0..nodes).step_by(plane) {
                    let part = &mut part[..plane.min(nodes - first) * 3];
                    fill(first, part)?;
                    out.extend(part.as_chunks::<3>().0.iter().map(&map));
                }
                Ok(out)
            }
        }
    }

    /// The outputs as they come, three a node.
    pub(crate) fn into_f32(self) -> Result<Vec<f32>, CmsError> {
        match self {
            Grid4x3::Whole(lut) => Ok(lut),
            Grid4x3::Planes {
                nodes,
                plane,
                mut fill,
            } => {
                let mut out = try_vec![0f32; nodes * 3];
                for (i, part) in out.chunks_mut(plane * 3).enumerate() {
                    fill(i * plane, part)?;
                }
                Ok(out)
            }
        }
    }
}

pub(crate) trait Lut4x3Factory {
    fn make_transform_4x3<
        T: Copy + AsPrimitive<f32> + Default + PointeeSizeExpressible + 'static + Send + Sync,
        const LAYOUT: u8,
        const GRID_SIZE: usize,
        const BIT_DEPTH: usize,
        const BINS: usize,
    >(
        lut: Grid4x3<'_>,
        options: TransformOptions,
        color_space: DataColorSpace,
        is_linear: bool,
    ) -> Result<Arc<dyn TransformExecutor<T> + Sync + Send>, CmsError>
    where
        f32: AsPrimitive<T>,
        u32: AsPrimitive<T>,
        (): LutBarycentricReduction<T, u8>,
        (): LutBarycentricReduction<T, u16>;
}

fn pcs_lab_v4_to_v2(profile: &ColorProfile, intent: RenderingIntent, lut: &mut [f32]) {
    if profile.pcs == DataColorSpace::Lab
        && profile.version_internal < ProfileVersion::V4_0
        && !profile.pcs_to_device_is_lut8(intent)
        && lut.len().is_multiple_of(3)
    {
        assert_eq!(
            lut.len() % 3,
            0,
            "Lut {:?} is not a multiple of 3, this should not happen for lab",
            lut.len()
        );
        let v_mat = vec![Matrix3f {
            v: [
                [65280.0 / 65535.0, 0f32, 0f32],
                [0f32, 65280.0 / 65535.0, 0f32],
                [0f32, 0f32, 65280.0 / 65535.0f32],
            ],
        }];
        let stage = MatrixStage { matrices: v_mat };
        stage.transform(lut).unwrap();
    }
}

fn pcs_lab_v2_to_v4(profile: &ColorProfile, intent: RenderingIntent, lut: &mut [f32]) {
    if profile.pcs == DataColorSpace::Lab
        && profile.version_internal < ProfileVersion::V4_0
        && !profile.device_to_pcs_is_lut8(intent)
        && lut.len().is_multiple_of(3)
    {
        assert_eq!(
            lut.len() % 3,
            0,
            "Lut {:?} is not a multiple of 3, this should not happen for lab",
            lut.len()
        );
        let v_mat = vec![Matrix3f {
            v: [
                [65535.0 / 65280.0f32, 0f32, 0f32],
                [0f32, 65535.0f32 / 65280.0f32, 0f32],
                [0f32, 0f32, 65535.0f32 / 65280.0f32],
            ],
        }];
        let stage = MatrixStage { matrices: v_mat };
        stage.transform(lut).unwrap();
    }
}

macro_rules! make_transform_3x3_fn {
    ($method_name: ident, $exec_impl: ident) => {
        fn $method_name<
            T: Copy
                + Default
                + AsPrimitive<f32>
                + Send
                + Sync
                + AsPrimitive<usize>
                + PointeeSizeExpressible,
            const GRID_SIZE: usize,
            const BIT_DEPTH: usize,
            const BINS: usize,
        >(
            src_layout: Layout,
            dst_layout: Layout,
            lut: Vec<f32>,
            options: TransformOptions,
            color_space: DataColorSpace,
            is_linear: bool,
        ) -> Arc<dyn TransformExecutor<T> + Send + Sync>
        where
            f32: AsPrimitive<T>,
            u32: AsPrimitive<T>,
            (): LutBarycentricReduction<T, u8>,
            (): LutBarycentricReduction<T, u16>,
        {
            match src_layout {
                Layout::Rgb => match dst_layout {
                    Layout::Rgb => $exec_impl::make_transform_3x3::<
                        T,
                        { Layout::Rgb as u8 },
                        { Layout::Rgb as u8 },
                        GRID_SIZE,
                        BIT_DEPTH,
                        BINS,
                    >(lut, options, color_space, is_linear),
                    Layout::Rgba => $exec_impl::make_transform_3x3::<
                        T,
                        { Layout::Rgb as u8 },
                        { Layout::Rgba as u8 },
                        GRID_SIZE,
                        BIT_DEPTH,
                        BINS,
                    >(lut, options, color_space, is_linear),
                    _ => unimplemented!(),
                },
                Layout::Rgba => match dst_layout {
                    Layout::Rgb => $exec_impl::make_transform_3x3::<
                        T,
                        { Layout::Rgba as u8 },
                        { Layout::Rgb as u8 },
                        GRID_SIZE,
                        BIT_DEPTH,
                        BINS,
                    >(lut, options, color_space, is_linear),
                    Layout::Rgba => $exec_impl::make_transform_3x3::<
                        T,
                        { Layout::Rgba as u8 },
                        { Layout::Rgba as u8 },
                        GRID_SIZE,
                        BIT_DEPTH,
                        BINS,
                    >(lut, options, color_space, is_linear),
                    _ => unimplemented!(),
                },
                _ => unimplemented!(),
            }
        }
    };
}

macro_rules! make_transform_4x3_fn {
    ($method_name: ident, $exec_name: ident) => {
        fn $method_name<
            T: Copy
                + Default
                + AsPrimitive<f32>
                + Send
                + Sync
                + AsPrimitive<usize>
                + PointeeSizeExpressible,
            const GRID_SIZE: usize,
            const BIT_DEPTH: usize,
            const BINS: usize,
        >(
            dst_layout: Layout,
            lut: Grid4x3<'_>,
            options: TransformOptions,
            data_color_space: DataColorSpace,
            is_linear: bool,
        ) -> Result<Arc<dyn TransformExecutor<T> + Send + Sync>, CmsError>
        where
            f32: AsPrimitive<T>,
            u32: AsPrimitive<T>,
            (): LutBarycentricReduction<T, u8>,
            (): LutBarycentricReduction<T, u16>,
        {
            match dst_layout {
                Layout::Rgb => $exec_name::make_transform_4x3::<
                    T,
                    { Layout::Rgb as u8 },
                    GRID_SIZE,
                    BIT_DEPTH,
                    BINS,
                >(lut, options, data_color_space, is_linear),
                Layout::Rgba => $exec_name::make_transform_4x3::<
                    T,
                    { Layout::Rgba as u8 },
                    GRID_SIZE,
                    BIT_DEPTH,
                    BINS,
                >(lut, options, data_color_space, is_linear),
                _ => unimplemented!(),
            }
        }
    };
}

#[cfg(all(target_arch = "aarch64", feature = "neon_luts"))]
use crate::conversions::neon::NeonLut3x3Factory;
#[cfg(all(target_arch = "aarch64", feature = "neon_luts"))]
make_transform_3x3_fn!(make_transformer_3x3, NeonLut3x3Factory);

#[cfg(not(all(target_arch = "aarch64", feature = "neon_luts")))]
use crate::conversions::transform_lut3_to_3::DefaultLut3x3Factory;
#[cfg(not(all(target_arch = "aarch64", feature = "neon_luts")))]
make_transform_3x3_fn!(make_transformer_3x3, DefaultLut3x3Factory);

#[cfg(all(target_arch = "x86_64", feature = "avx_luts"))]
use crate::conversions::avx::AvxLut3x3Factory;
#[cfg(all(target_arch = "x86_64", feature = "avx_luts"))]
make_transform_3x3_fn!(make_transformer_3x3_avx_fma, AvxLut3x3Factory);

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse_luts"))]
use crate::conversions::sse::SseLut3x3Factory;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse_luts"))]
make_transform_3x3_fn!(make_transformer_3x3_sse41, SseLut3x3Factory);

use crate::conversions::LutBarycentricReduction;
#[cfg(all(target_arch = "x86_64", feature = "avx_luts"))]
use crate::conversions::avx::AvxLut4x3Factory;
use crate::conversions::bpc::{TransformDirection, compensate_bpc_in_lut};
#[cfg(feature = "any_to_any")]
use crate::conversions::katana::{
    Katana, KatanaDefaultIntermediate, KatanaInitialStage, KatanaPostFinalizationStage,
    KatanaStageLabToXyz, KatanaStageXyzToLab, katana_create_rgb_lin_lut, katana_pcs_lab_v2_to_v4,
    katana_pcs_lab_v4_to_v2, katana_prepare_inverse_lut_rgb_xyz, multi_dimensional_3x3_to_device,
    multi_dimensional_3x3_to_pcs, multi_dimensional_4x3_to_pcs,
};
use crate::conversions::mab4x3::prepare_mab_4x3;
use crate::conversions::mba3x4::prepare_mba_3x4;
#[cfg(feature = "any_to_any")]
use crate::conversions::md_luts_factory::{do_any_to_any, prepare_alpha_finalizer};
// use crate::conversions::bpc::compensate_bpc_in_lut;

#[cfg(all(target_arch = "x86_64", feature = "avx_luts"))]
make_transform_4x3_fn!(make_transformer_4x3_avx_fma, AvxLut4x3Factory);

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse_luts"))]
use crate::conversions::sse::SseLut4x3Factory;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse_luts"))]
make_transform_4x3_fn!(make_transformer_4x3_sse41, SseLut4x3Factory);

#[cfg(not(all(target_arch = "aarch64", feature = "neon_luts")))]
use crate::conversions::transform_lut4_to_3::DefaultLut4x3Factory;

#[cfg(not(all(target_arch = "aarch64", feature = "neon_luts")))]
make_transform_4x3_fn!(make_transformer_4x3, DefaultLut4x3Factory);

use crate::conversions::prelude_lut_xyz_rgb::{
    InverseRgbXyzStage, create_rgb_lin_lut, inverse_lut_rgb_xyz_stage, prepare_inverse_lut_rgb_xyz,
};
use crate::conversions::xyz_lab::{StageLabToXyz, StageXyzToLab};
use crate::transform::PointeeSizeExpressible;
use crate::trc::GammaLutInterpolate;

#[cfg(all(target_arch = "aarch64", feature = "neon_luts"))]
use crate::conversions::neon::NeonLut4x3Factory;
#[cfg(all(target_arch = "aarch64", feature = "neon_luts"))]
make_transform_4x3_fn!(make_transformer_4x3, NeonLut4x3Factory);

/// What turns the PCS into the destination's device values in a grid made by planes.
enum PcsToDevice<T: Clone> {
    Rgb(InverseRgbXyzStage<T>),
    Lut(Box<dyn Stage>),
}

/// The 4-ink to RGB or Lab grid of [`make_lut_transform`] made a plane of the last ink at a time,
/// node by node as it is made whole: for a device to PCS table of lut8 or lut16 and a
/// matrix-shaper destination or one with a lut8 or lut16 PCS to device table, None for others.
fn grid_4x3_by_planes<
    'a,
    T: Copy
        + Default
        + AsPrimitive<f32>
        + Send
        + Sync
        + AsPrimitive<usize>
        + PointeeSizeExpressible
        + GammaLutInterpolate,
    const GRID_SIZE: usize,
    const BIT_DEPTH: usize,
    const GAMMA_LUT: usize,
>(
    source: &'a ColorProfile,
    dest: &'a ColorProfile,
    options: TransformOptions,
) -> Result<Option<Grid4x3<'a>>, CmsError>
where
    f32: AsPrimitive<T>,
    u32: AsPrimitive<T>,
{
    let Some(LutWarehouse::Lut(device_to_pcs)) = source.get_device_to_pcs(options.rendering_intent)
    else {
        return Ok(None);
    };
    let lut_stage = stage_lut_4x3(device_to_pcs, options, source.pcs)?;
    let last = match dest.pcs {
        DataColorSpace::Xyz if dest.is_matrix_shaper() => PcsToDevice::Rgb(
            inverse_lut_rgb_xyz_stage::<T, BIT_DEPTH, GAMMA_LUT>(dest, options)?,
        ),
        DataColorSpace::Lab => match dest
            .get_pcs_to_device(options.rendering_intent)
            .ok_or(CmsError::UnsupportedProfileConnection)?
        {
            LutWarehouse::Lut(lut) => PcsToDevice::Lut(stage_lut_3x3(lut, options, dest.pcs)?),
            LutWarehouse::Multidimensional(_) => return Ok(None),
        },
        _ => return Ok(None),
    };
    let black_points = match (
        source.detect_black_point(options.rendering_intent, TransformDirection::DeviceToPcs),
        dest.detect_black_point(options.rendering_intent, TransformDirection::PcsToDevice),
    ) {
        (Some(src_bp), Some(dest_bp)) => Some((src_bp, dest_bp)),
        _ => None,
    };
    let apply_bpc_scaling = move |lut: &mut [f32]| {
        if let Some((src_bp, dest_bp)) = black_points {
            compensate_bpc_in_lut(lut, src_bp, dest_bp);
        }
    };

    let plane = GRID_SIZE * GRID_SIZE * GRID_SIZE;
    let recpeq = 1f32 / (GRID_SIZE - 1) as f32;
    let mut samples = try_vec![0f32; plane * 4];
    let mut scratch = match last {
        PcsToDevice::Lut(_) => try_vec![0f32; plane * 3],
        PcsToDevice::Rgb(_) => Vec::new(),
    };
    let fill = move |first: usize, out: &mut [f32]| -> Result<(), CmsError> {
        let samples = &mut samples[..out.len() / 3 * 4];
        for (j, sample) in samples.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let i = first + j;
            let (k, c) = (i / plane, i / (GRID_SIZE * GRID_SIZE) % GRID_SIZE);
            let (m, y) = (i / GRID_SIZE % GRID_SIZE, i % GRID_SIZE);
            *sample = [
                c as f32 * recpeq,
                m as f32 * recpeq,
                y as f32 * recpeq,
                k as f32 * recpeq,
            ];
        }
        lut_stage.transform(samples, out)?;

        pcs_lab_v2_to_v4(source, options.rendering_intent, out);

        if source.pcs == DataColorSpace::Xyz && dest.pcs == DataColorSpace::Lab {
            apply_bpc_scaling(out);
            StageXyzToLab::default().transform(out)?;
        } else if source.pcs == DataColorSpace::Lab && dest.pcs == DataColorSpace::Xyz {
            StageLabToXyz::default().transform(out)?;
            apply_bpc_scaling(out);
        } else {
            apply_bpc_scaling(out);
        }

        pcs_lab_v4_to_v2(dest, options.rendering_intent, out);

        match &last {
            PcsToDevice::Rgb(stage) => stage.transform(out),
            PcsToDevice::Lut(stage) => {
                let scratch = &mut scratch[..out.len()];
                scratch.copy_from_slice(out);
                stage.transform(scratch, out)
            }
        }
    };
    Ok(Some(Grid4x3::Planes {
        nodes: plane * GRID_SIZE,
        plane,
        fill: Box::new(fill),
    }))
}

#[inline(never)]
#[cold]
pub(crate) fn make_lut_transform<
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
    const BINS: usize,
>(
    src_layout: Layout,
    source: &ColorProfile,
    dst_layout: Layout,
    dest: &ColorProfile,
    options: TransformOptions,
) -> Result<Arc<dyn TransformExecutor<T> + Send + Sync>, CmsError>
where
    f32: AsPrimitive<T>,
    u32: AsPrimitive<T>,
    (): LutBarycentricReduction<T, u8>,
    (): LutBarycentricReduction<T, u16>,
{
    let apply_bpc_scaling = |lut: &mut [f32]| {
        if let (Some(src_bp), Some(dest_bp)) = (
            source.detect_black_point(options.rendering_intent, TransformDirection::DeviceToPcs),
            dest.detect_black_point(options.rendering_intent, TransformDirection::PcsToDevice),
        ) {
            compensate_bpc_in_lut(lut, src_bp, dest_bp);
        }
    };

    if (source.color_space == DataColorSpace::Cmyk || source.color_space == DataColorSpace::Color4)
        && (dest.color_space == DataColorSpace::Rgb || dest.color_space == DataColorSpace::Lab)
    {
        source.color_space.check_layout(src_layout)?;
        dest.color_space.check_layout(dst_layout)?;
        if source.pcs != DataColorSpace::Xyz && source.pcs != DataColorSpace::Lab {
            return Err(CmsError::UnsupportedProfileConnection);
        }
        if dest.pcs != DataColorSpace::Lab && dest.pcs != DataColorSpace::Xyz {
            return Err(CmsError::UnsupportedProfileConnection);
        }

        const GRID_SIZE: usize = 17;

        #[cfg(feature = "any_to_any")]
        let is_katana_required_for_source = source
            .get_device_to_pcs(options.rendering_intent)
            .ok_or(CmsError::UnsupportedLutRenderingIntent(
                source.rendering_intent,
            ))
            .map(|x| x.is_katana_required())?;

        #[cfg(feature = "any_to_any")]
        let is_katana_required_for_destination =
            if dest.is_matrix_shaper() || dest.pcs == DataColorSpace::Xyz {
                false
            } else if dest.pcs == DataColorSpace::Lab {
                dest.get_pcs_to_device(options.rendering_intent)
                    .ok_or(CmsError::UnsupportedProfileConnection)
                    .map(|x| x.is_katana_required())?
            } else {
                return Err(CmsError::UnsupportedProfileConnection);
            };

        #[cfg(feature = "any_to_any")]
        if is_katana_required_for_source || is_katana_required_for_destination {
            let initial_stage: Box<dyn KatanaInitialStage<f32, T> + Send + Sync> =
                match source.get_device_to_pcs(options.rendering_intent).ok_or(
                    CmsError::UnsupportedLutRenderingIntent(source.rendering_intent),
                )? {
                    LutWarehouse::Lut(lut) => {
                        katana_input_stage_lut_4x3::<T>(lut, options, source.pcs, BIT_DEPTH)?
                    }
                    LutWarehouse::Multidimensional(mab) => {
                        multi_dimensional_4x3_to_pcs::<T>(mab, options, source.pcs, BIT_DEPTH)?
                    }
                };

            let mut stages = Vec::new();

            stages.push(katana_pcs_lab_v2_to_v4(source, options.rendering_intent));
            if source.pcs == DataColorSpace::Lab {
                stages.push(Box::new(KatanaStageLabToXyz::default()));
            }
            if dest.pcs == DataColorSpace::Lab {
                stages.push(Box::new(KatanaStageXyzToLab::default()));
            }
            stages.push(katana_pcs_lab_v4_to_v2(dest, options.rendering_intent));

            let final_stage = if dest.has_pcs_to_device_lut() {
                let pcs_to_device = dest
                    .get_pcs_to_device(options.rendering_intent)
                    .ok_or(CmsError::UnsupportedProfileConnection)?;
                match pcs_to_device {
                    LutWarehouse::Lut(lut) => {
                        katana_output_stage_lut_3x3::<T>(lut, options, dest.pcs, BIT_DEPTH)?
                    }
                    LutWarehouse::Multidimensional(mab) => {
                        multi_dimensional_3x3_to_device::<T>(mab, options, dest.pcs, BIT_DEPTH)?
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

            let mut post_finalization: Vec<Box<dyn KatanaPostFinalizationStage<T> + Send + Sync>> =
                Vec::new();
            if let Some(stage) =
                prepare_alpha_finalizer::<T>(src_layout, source, dst_layout, dest, BIT_DEPTH)
            {
                post_finalization.push(stage);
            }

            return Ok(Arc::new(Katana::<f32, T> {
                initial_stage,
                final_stage,
                stages,
                post_finalization,
            }));
        }

        let lut = match grid_4x3_by_planes::<T, GRID_SIZE, BIT_DEPTH, GAMMA_LUT>(
            source, dest, options,
        )? {
            Some(grid) => grid,
            None => {
                let mut lut = match source.get_device_to_pcs(options.rendering_intent).ok_or(
                    CmsError::UnsupportedLutRenderingIntent(source.rendering_intent),
                )? {
                    LutWarehouse::Lut(lut) => create_lut4::<GRID_SIZE>(lut, options, source.pcs)?,
                    LutWarehouse::Multidimensional(m_curves) => {
                        let mut samples = create_lut4_norm_samples::<GRID_SIZE>();
                        prepare_mab_4x3(m_curves, &mut samples, options, source.pcs)?
                    }
                };

                pcs_lab_v2_to_v4(source, options.rendering_intent, &mut lut);

                if source.pcs == DataColorSpace::Xyz && dest.pcs == DataColorSpace::Lab {
                    apply_bpc_scaling(&mut lut);
                    let xyz_to_lab = StageXyzToLab::default();
                    xyz_to_lab.transform(&mut lut)?;
                } else if source.pcs == DataColorSpace::Lab && dest.pcs == DataColorSpace::Xyz {
                    let lab_to_xyz_stage = StageLabToXyz::default();
                    lab_to_xyz_stage.transform(&mut lut)?;
                    apply_bpc_scaling(&mut lut);
                } else {
                    apply_bpc_scaling(&mut lut);
                }

                pcs_lab_v4_to_v2(dest, options.rendering_intent, &mut lut);

                if dest.pcs == DataColorSpace::Xyz {
                    if dest.is_matrix_shaper() {
                        prepare_inverse_lut_rgb_xyz::<T, BIT_DEPTH, GAMMA_LUT>(
                            dest, &mut lut, options,
                        )?;
                    } else {
                        return Err(CmsError::UnsupportedProfileConnection);
                    }
                } else if dest.pcs == DataColorSpace::Lab {
                    let pcs_to_device = dest
                        .get_pcs_to_device(options.rendering_intent)
                        .ok_or(CmsError::UnsupportedProfileConnection)?;
                    match pcs_to_device {
                        LutWarehouse::Lut(lut_data_type) => {
                            lut = create_lut3x3(lut_data_type, &lut, options, dest.pcs)?
                        }
                        LutWarehouse::Multidimensional(mab) => {
                            prepare_mba_3x3(mab, &mut lut, options, dest.pcs)?
                        }
                    }
                }
                Grid4x3::Whole(lut)
            }
        };

        let is_dest_linear_profile = dest.color_space == DataColorSpace::Rgb
            && dest.is_matrix_shaper()
            && dest.is_linear_matrix_shaper();

        #[cfg(all(target_arch = "x86_64", feature = "avx_luts"))]
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return make_transformer_4x3_avx_fma::<T, GRID_SIZE, BIT_DEPTH, BINS>(
                dst_layout,
                lut,
                options,
                dest.color_space,
                is_dest_linear_profile,
            );
        }
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse_luts"))]
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return make_transformer_4x3_sse41::<T, GRID_SIZE, BIT_DEPTH, BINS>(
                dst_layout,
                lut,
                options,
                dest.color_space,
                is_dest_linear_profile,
            );
        }

        make_transformer_4x3::<T, GRID_SIZE, BIT_DEPTH, BINS>(
            dst_layout,
            lut,
            options,
            dest.color_space,
            is_dest_linear_profile,
        )
    } else if (source.color_space == DataColorSpace::Rgb
        || source.color_space == DataColorSpace::Lab)
        && (dest.color_space == DataColorSpace::Cmyk || dest.color_space == DataColorSpace::Color4)
    {
        source.color_space.check_layout(src_layout)?;
        dest.color_space.check_layout(dst_layout)?;

        if source.pcs != DataColorSpace::Xyz && source.pcs != DataColorSpace::Lab {
            return Err(CmsError::UnsupportedProfileConnection);
        }

        const GRID_SIZE: usize = 33;

        let mut lut: Vec<f32>;

        if source.has_device_to_pcs_lut() {
            let device_to_pcs = source
                .get_device_to_pcs(options.rendering_intent)
                .ok_or(CmsError::UnsupportedProfileConnection)?;
            lut = create_lut3_samples_norm::<GRID_SIZE>();

            match device_to_pcs {
                LutWarehouse::Lut(lut_data_type) => {
                    lut = create_lut3x3(lut_data_type, &lut, options, source.pcs)?;
                }
                LutWarehouse::Multidimensional(mab) => {
                    prepare_mab_3x3(mab, &mut lut, options, source.pcs)?
                }
            }
        } else if source.is_matrix_shaper() {
            lut = create_rgb_lin_lut::<T, BIT_DEPTH, LINEAR_CAP, GRID_SIZE>(source, options)?;
        } else {
            return Err(CmsError::UnsupportedProfileConnection);
        }

        pcs_lab_v2_to_v4(source, options.rendering_intent, &mut lut);

        if source.pcs == DataColorSpace::Xyz && dest.pcs == DataColorSpace::Lab {
            apply_bpc_scaling(&mut lut);
            let xyz_to_lab = StageXyzToLab::default();
            xyz_to_lab.transform(&mut lut)?;
        } else if source.pcs == DataColorSpace::Lab && dest.pcs == DataColorSpace::Xyz {
            let lab_to_xyz_stage = StageLabToXyz::default();
            lab_to_xyz_stage.transform(&mut lut)?;
            apply_bpc_scaling(&mut lut);
        } else {
            apply_bpc_scaling(&mut lut);
        }

        pcs_lab_v4_to_v2(dest, options.rendering_intent, &mut lut);

        let lut = match dest
            .get_pcs_to_device(options.rendering_intent)
            .ok_or(CmsError::UnsupportedProfileConnection)?
        {
            LutWarehouse::Lut(lut_type) => create_lut3x4(lut_type, &lut, options, dest.pcs)?,
            LutWarehouse::Multidimensional(m_curves) => {
                prepare_mba_3x4(m_curves, &mut lut, options, dest.pcs)?
            }
        };

        let is_dest_linear_profile = dest.color_space == DataColorSpace::Rgb
            && dest.is_matrix_shaper()
            && dest.is_linear_matrix_shaper();

        Ok(make_transform_3x4::<T, GRID_SIZE, BIT_DEPTH, BINS>(
            src_layout,
            lut,
            options,
            dest.color_space,
            is_dest_linear_profile,
        ))
    } else if (source.color_space.is_three_channels()) && (dest.color_space.is_three_channels()) {
        source.color_space.check_layout(src_layout)?;
        dest.color_space.check_layout(dst_layout)?;

        const GRID_SIZE: usize = 33;

        #[cfg(feature = "any_to_any")]
        let is_katana_required_for_source = if source.is_matrix_shaper() {
            false
        } else {
            source
                .get_device_to_pcs(options.rendering_intent)
                .ok_or(CmsError::UnsupportedLutRenderingIntent(
                    source.rendering_intent,
                ))
                .map(|x| x.is_katana_required())?
        };

        #[cfg(feature = "any_to_any")]
        let is_katana_required_for_destination =
            if source.is_matrix_shaper() || dest.pcs == DataColorSpace::Xyz {
                false
            } else if dest.pcs == DataColorSpace::Lab {
                dest.get_pcs_to_device(options.rendering_intent)
                    .ok_or(CmsError::UnsupportedProfileConnection)
                    .map(|x| x.is_katana_required())?
            } else {
                return Err(CmsError::UnsupportedProfileConnection);
            };

        #[cfg(feature = "any_to_any")]
        let mut stages: Vec<Box<KatanaDefaultIntermediate>> = Vec::new();

        // Slow and accurate fallback if anything not acceptable is detected by curve analysis
        #[cfg(feature = "any_to_any")]
        if is_katana_required_for_source || is_katana_required_for_destination {
            let source_stage: Box<dyn KatanaInitialStage<f32, T> + Send + Sync> =
                if source.is_matrix_shaper() {
                    let state = katana_create_rgb_lin_lut::<T, BIT_DEPTH, LINEAR_CAP>(
                        src_layout, source, options,
                    )?;
                    stages.extend(state.stages);
                    state.initial_stage
                } else {
                    match source.get_device_to_pcs(options.rendering_intent).ok_or(
                        CmsError::UnsupportedLutRenderingIntent(source.rendering_intent),
                    )? {
                        LutWarehouse::Lut(lut) => {
                            katana_input_stage_lut_3x3::<T>(lut, options, source.pcs, BIT_DEPTH)?
                        }
                        LutWarehouse::Multidimensional(mab) => {
                            multi_dimensional_3x3_to_pcs::<T>(mab, options, source.pcs, BIT_DEPTH)?
                        }
                    }
                };

            stages.push(katana_pcs_lab_v2_to_v4(source, options.rendering_intent));
            if source.pcs == DataColorSpace::Lab {
                stages.push(Box::new(KatanaStageLabToXyz::default()));
            }
            if dest.pcs == DataColorSpace::Lab {
                stages.push(Box::new(KatanaStageXyzToLab::default()));
            }
            stages.push(katana_pcs_lab_v4_to_v2(dest, options.rendering_intent));

            let final_stage = if dest.has_pcs_to_device_lut() {
                let pcs_to_device = dest
                    .get_pcs_to_device(options.rendering_intent)
                    .ok_or(CmsError::UnsupportedProfileConnection)?;
                match pcs_to_device {
                    LutWarehouse::Lut(lut) => {
                        katana_output_stage_lut_3x3::<T>(lut, options, dest.pcs, BIT_DEPTH)?
                    }
                    LutWarehouse::Multidimensional(mab) => {
                        multi_dimensional_3x3_to_device::<T>(mab, options, dest.pcs, BIT_DEPTH)?
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

            let mut post_finalization: Vec<Box<dyn KatanaPostFinalizationStage<T> + Send + Sync>> =
                Vec::new();
            if let Some(stage) =
                prepare_alpha_finalizer::<T>(src_layout, source, dst_layout, dest, BIT_DEPTH)
            {
                post_finalization.push(stage);
            }

            return Ok(Arc::new(Katana::<f32, T> {
                initial_stage: source_stage,
                final_stage,
                stages,
                post_finalization,
            }));
        }

        let mut lut: Vec<f32>;

        if source.has_device_to_pcs_lut() {
            let device_to_pcs = source
                .get_device_to_pcs(options.rendering_intent)
                .ok_or(CmsError::UnsupportedProfileConnection)?;
            lut = create_lut3_samples_norm::<GRID_SIZE>();

            match device_to_pcs {
                LutWarehouse::Lut(lut_data_type) => {
                    lut = create_lut3x3(lut_data_type, &lut, options, source.pcs)?;
                }
                LutWarehouse::Multidimensional(mab) => {
                    prepare_mab_3x3(mab, &mut lut, options, source.pcs)?
                }
            }
        } else if source.is_matrix_shaper() {
            lut = create_rgb_lin_lut::<T, BIT_DEPTH, LINEAR_CAP, GRID_SIZE>(source, options)?;
        } else {
            return Err(CmsError::UnsupportedProfileConnection);
        }

        pcs_lab_v2_to_v4(source, options.rendering_intent, &mut lut);

        if source.pcs == DataColorSpace::Xyz && dest.pcs == DataColorSpace::Lab {
            apply_bpc_scaling(&mut lut);
            let xyz_to_lab = StageXyzToLab::default();
            xyz_to_lab.transform(&mut lut)?;
        } else if source.pcs == DataColorSpace::Lab && dest.pcs == DataColorSpace::Xyz {
            let lab_to_xyz_stage = StageLabToXyz::default();
            lab_to_xyz_stage.transform(&mut lut)?;
            apply_bpc_scaling(&mut lut);
        } else {
            apply_bpc_scaling(&mut lut);
        }

        pcs_lab_v4_to_v2(dest, options.rendering_intent, &mut lut);

        if dest.has_pcs_to_device_lut() {
            let pcs_to_device = dest
                .get_pcs_to_device(options.rendering_intent)
                .ok_or(CmsError::UnsupportedProfileConnection)?;
            match pcs_to_device {
                LutWarehouse::Lut(lut_data_type) => {
                    lut = create_lut3x3(lut_data_type, &lut, options, dest.pcs)?;
                }
                LutWarehouse::Multidimensional(mab) => {
                    prepare_mba_3x3(mab, &mut lut, options, dest.pcs)?
                }
            }
        } else if dest.is_matrix_shaper() {
            prepare_inverse_lut_rgb_xyz::<T, BIT_DEPTH, GAMMA_LUT>(dest, &mut lut, options)?;
        } else {
            return Err(CmsError::UnsupportedProfileConnection);
        }

        let is_dest_linear_profile = dest.color_space == DataColorSpace::Rgb
            && dest.is_matrix_shaper()
            && dest.is_linear_matrix_shaper();

        #[cfg(all(feature = "avx_luts", target_arch = "x86_64"))]
        if std::arch::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return Ok(
                make_transformer_3x3_avx_fma::<T, GRID_SIZE, BIT_DEPTH, BINS>(
                    src_layout,
                    dst_layout,
                    lut,
                    options,
                    dest.color_space,
                    is_dest_linear_profile,
                ),
            );
        }
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse_luts"))]
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return Ok(make_transformer_3x3_sse41::<T, GRID_SIZE, BIT_DEPTH, BINS>(
                src_layout,
                dst_layout,
                lut,
                options,
                dest.color_space,
                is_dest_linear_profile,
            ));
        }

        #[cfg(all(target_arch = "aarch64", feature = "neon_luts"))]
        {
            Ok(make_transformer_3x3::<T, GRID_SIZE, BIT_DEPTH, BINS>(
                src_layout,
                dst_layout,
                lut,
                options,
                dest.color_space,
                is_dest_linear_profile,
            ))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon_luts")))]
        {
            Ok(make_transformer_3x3::<T, GRID_SIZE, BIT_DEPTH, BINS>(
                src_layout,
                dst_layout,
                lut,
                options,
                dest.color_space,
                is_dest_linear_profile,
            ))
        }
    } else {
        #[cfg(feature = "any_to_any")]
        {
            do_any_to_any::<T, BIT_DEPTH, LINEAR_CAP, GAMMA_LUT>(
                src_layout, source, dst_layout, dest, options,
            )
        }
        #[cfg(not(feature = "any_to_any"))]
        {
            Err(CmsError::UnsupportedProfileConnection)
        }
    }
}
