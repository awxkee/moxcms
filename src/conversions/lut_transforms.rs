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
use crate::conversions::lut3x3::create_lut3x3;
use crate::conversions::lut3x4::{create_lut3_samples, create_lut3_samples_norm, create_lut3x4};
use crate::conversions::lut4::{create_lut4, create_lut4_norm_samples};
use crate::conversions::mab::{prepare_mab_3x3, prepare_mba_3x3};
use crate::conversions::transform_lut3_to_4::make_transform_3x4;
use crate::lab::Lab;
use crate::mlaf::mlaf;
use crate::{
    CmsError, ColorProfile, DataColorSpace, InPlaceStage, Layout, LutWarehouse, Matrix3f,
    ProfileVersion, TransformExecutor, TransformOptions, Xyz,
};
use num_traits::AsPrimitive;
use std::marker::PhantomData;

#[derive(Default)]
pub(crate) struct StageLabToXyz {}

impl InPlaceStage for StageLabToXyz {
    fn transform(&self, dst: &mut [f32]) -> Result<(), CmsError> {
        for dst in dst.chunks_exact_mut(3) {
            let lab = Lab::new(dst[0], dst[1], dst[2]);
            let xyz = lab.to_pcs_xyz();
            dst[0] = xyz.x;
            dst[1] = xyz.y;
            dst[2] = xyz.z;
        }
        Ok(())
    }
}

#[derive(Default)]
pub(crate) struct StageXyzToLab {}

impl InPlaceStage for StageXyzToLab {
    fn transform(&self, dst: &mut [f32]) -> Result<(), CmsError> {
        for dst in dst.chunks_exact_mut(3) {
            let xyz = Xyz::new(dst[0], dst[1], dst[2]);
            let lab = Lab::from_pcs_xyz(xyz);
            dst[0] = lab.l;
            dst[1] = lab.a;
            dst[2] = lab.b;
        }
        Ok(())
    }
}

struct MatrixStage {
    matrices: Vec<Matrix3f>,
}

impl InPlaceStage for MatrixStage {
    fn transform(&self, dst: &mut [f32]) -> Result<(), CmsError> {
        if !self.matrices.is_empty() {
            let m = self.matrices[0];
            for dst in dst.chunks_exact_mut(3) {
                let x = dst[0];
                let y = dst[1];
                let z = dst[2];
                dst[0] = mlaf(mlaf(x * m.v[0][0], y, m.v[0][1]), z, m.v[0][2]);
                dst[1] = mlaf(mlaf(x * m.v[1][0], y, m.v[1][1]), z, m.v[1][2]);
                dst[2] = mlaf(mlaf(x * m.v[2][0], y, m.v[2][1]), z, m.v[2][2]);
            }
        }

        for m in self.matrices.iter().skip(1) {
            for dst in dst.chunks_exact_mut(3) {
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
    >(
        lut: Vec<f32>,
        options: TransformOptions,
        color_space: DataColorSpace,
        is_linear: bool,
    ) -> Box<dyn TransformExecutor<T> + Send + Sync>
    where
        f32: AsPrimitive<T>,
        u32: AsPrimitive<T>,
        (): LutBarycentricReduction<T, u8>,
        (): LutBarycentricReduction<T, u16>;
}

pub(crate) trait Lut4x3Factory {
    fn make_transform_4x3<
        T: Copy + AsPrimitive<f32> + Default + PointeeSizeExpressible + 'static + Send + Sync,
        const LAYOUT: u8,
        const GRID_SIZE: usize,
        const BIT_DEPTH: usize,
    >(
        lut: Vec<f32>,
        options: TransformOptions,
        color_space: DataColorSpace,
        is_linear: bool,
    ) -> Box<dyn TransformExecutor<T> + Sync + Send>
    where
        f32: AsPrimitive<T>,
        u32: AsPrimitive<T>,
        (): LutBarycentricReduction<T, u8>,
        (): LutBarycentricReduction<T, u16>;
}

struct RgbLinearizationStage<
    T: Clone,
    const BIT_DEPTH: usize,
    const LINEAR_CAP: usize,
    const SAMPLES: usize,
> {
    r_lin: Box<[f32; LINEAR_CAP]>,
    g_lin: Box<[f32; LINEAR_CAP]>,
    b_lin: Box<[f32; LINEAR_CAP]>,
    _phantom: PhantomData<T>,
}

impl<
    T: Clone + AsPrimitive<usize> + PointeeSizeExpressible,
    const BIT_DEPTH: usize,
    const LINEAR_CAP: usize,
    const SAMPLES: usize,
> RgbLinearizationStage<T, BIT_DEPTH, LINEAR_CAP, SAMPLES>
{
    fn transform(&self, src: &[T], dst: &mut [f32]) -> Result<(), CmsError> {
        if src.len() % 3 != 0 {
            return Err(CmsError::LaneMultipleOfChannels);
        }
        if dst.len() % 3 != 0 {
            return Err(CmsError::LaneMultipleOfChannels);
        }

        let scale = if T::FINITE {
            ((1 << BIT_DEPTH) - 1) as f32 / (SAMPLES as f32 - 1f32)
        } else {
            (T::NOT_FINITE_LINEAR_TABLE_SIZE - 1) as f32 / (SAMPLES as f32 - 1f32)
        };

        for (src, dst) in src.chunks_exact(3).zip(dst.chunks_exact_mut(3)) {
            let j_r = src[0].as_() as f32 * scale;
            let j_g = src[1].as_() as f32 * scale;
            let j_b = src[2].as_() as f32 * scale;
            dst[0] = self.r_lin[(j_r.round() as u16) as usize];
            dst[1] = self.g_lin[(j_g.round() as u16) as usize];
            dst[2] = self.b_lin[(j_b.round() as u16) as usize];
        }
        Ok(())
    }
}

fn pcs_lab_v4_to_v2(profile: &ColorProfile, lut: &mut [f32]) {
    if profile.pcs == DataColorSpace::Lab
        && profile.version_internal <= ProfileVersion::V4_0
        && lut.len() % 3 == 0
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

fn pcs_lab_v2_to_v4(profile: &ColorProfile, lut: &mut [f32]) {
    if profile.pcs == DataColorSpace::Lab
        && profile.version_internal <= ProfileVersion::V4_0
        && lut.len() % 3 == 0
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
        >(
            src_layout: Layout,
            dst_layout: Layout,
            lut: Vec<f32>,
            options: TransformOptions,
            color_space: DataColorSpace,
            is_linear: bool,
        ) -> Box<dyn TransformExecutor<T> + Send + Sync>
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
                    >(lut, options, color_space, is_linear),
                    Layout::Rgba => $exec_impl::make_transform_3x3::<
                        T,
                        { Layout::Rgb as u8 },
                        { Layout::Rgba as u8 },
                        GRID_SIZE,
                        BIT_DEPTH,
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
                    >(lut, options, color_space, is_linear),
                    Layout::Rgba => $exec_impl::make_transform_3x3::<
                        T,
                        { Layout::Rgba as u8 },
                        { Layout::Rgba as u8 },
                        GRID_SIZE,
                        BIT_DEPTH,
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
        >(
            dst_layout: Layout,
            lut: Vec<f32>,
            options: TransformOptions,
            data_color_space: DataColorSpace,
            is_linear: bool,
        ) -> Box<dyn TransformExecutor<T> + Send + Sync>
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
                >(lut, options, data_color_space, is_linear),
                Layout::Rgba => $exec_name::make_transform_4x3::<
                    T,
                    { Layout::Rgba as u8 },
                    GRID_SIZE,
                    BIT_DEPTH,
                >(lut, options, data_color_space, is_linear),
                _ => unimplemented!(),
            }
        }
    };
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon", feature = "neon"))]
use crate::conversions::neon::NeonLut3x3Factory;
#[cfg(all(target_arch = "aarch64", target_feature = "neon", feature = "neon"))]
make_transform_3x3_fn!(make_transformer_3x3, NeonLut3x3Factory);

#[cfg(not(all(target_arch = "aarch64", target_feature = "neon", feature = "neon")))]
use crate::conversions::transform_lut3_to_3::DefaultLut3x3Factory;
#[cfg(not(all(target_arch = "aarch64", target_feature = "neon", feature = "neon")))]
make_transform_3x3_fn!(make_transformer_3x3, DefaultLut3x3Factory);

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
use crate::conversions::avx::AvxLut3x3Factory;
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
make_transform_3x3_fn!(make_transformer_3x3_avx_fma, AvxLut3x3Factory);

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
use crate::conversions::sse::SseLut3x3Factory;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
make_transform_3x3_fn!(make_transformer_3x3_sse41, SseLut3x3Factory);

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
use crate::conversions::avx::AvxLut4x3Factory;
use crate::conversions::interpolator::LutBarycentricReduction;
use crate::conversions::mab4x3::prepare_mab_4x3;
use crate::conversions::mba3x4::prepare_mba_3x4;
// use crate::conversions::bpc::compensate_bpc_in_lut;

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
make_transform_4x3_fn!(make_transformer_4x3_avx_fma, AvxLut4x3Factory);

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
use crate::conversions::sse::SseLut4x3Factory;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
make_transform_4x3_fn!(make_transformer_4x3_sse41, SseLut4x3Factory);

#[cfg(not(all(target_arch = "aarch64", target_feature = "neon", feature = "neon")))]
use crate::conversions::transform_lut4_to_3::DefaultLut4x3Factory;

#[cfg(not(all(target_arch = "aarch64", target_feature = "neon", feature = "neon")))]
make_transform_4x3_fn!(make_transformer_4x3, DefaultLut4x3Factory);

#[cfg(all(target_arch = "aarch64", target_feature = "neon", feature = "neon"))]
use crate::conversions::neon::NeonLut4x3Factory;
use crate::conversions::prelude_lut_xyz_rgb::{XyzToRgbStage, XyzToRgbStageExtended};
use crate::transform::PointeeSizeExpressible;
use crate::trc::GammaLutInterpolate;

#[cfg(all(target_arch = "aarch64", target_feature = "neon", feature = "neon"))]
make_transform_4x3_fn!(make_transformer_4x3, NeonLut4x3Factory);

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

        let mut lut = match source.get_device_to_pcs(options.rendering_intent).ok_or(
            CmsError::UnsupportedLutRenderingIntent(source.rendering_intent),
        )? {
            LutWarehouse::Lut(lut) => create_lut4::<GRID_SIZE>(lut, options, source.pcs)?,
            LutWarehouse::Multidimensional(m_curves) => {
                let mut samples = create_lut4_norm_samples::<GRID_SIZE>();
                prepare_mab_4x3(m_curves, &mut samples, options, source.pcs)?
            }
        };

        pcs_lab_v2_to_v4(source, &mut lut);

        if source.pcs == DataColorSpace::Lab {
            let lab_to_xyz_stage = StageLabToXyz::default();
            lab_to_xyz_stage.transform(&mut lut)?;
        }

        // if source.color_space == DataColorSpace::Cmyk
        //     && (options.rendering_intent == RenderingIntent::Perceptual
        //         || options.rendering_intent == RenderingIntent::RelativeColorimetric)
        //     && options.black_point_compensation
        // {
        //     if let (Some(src_bp), Some(dst_bp)) = (
        //         source.detect_black_point::<GRID_SIZE>(&lut),
        //         dest.detect_black_point::<GRID_SIZE>(&lut),
        //     ) {
        //         compensate_bpc_in_lut(&mut lut, src_bp, dst_bp);
        //     }
        // }

        if dest.pcs == DataColorSpace::Lab {
            let lab_to_xyz_stage = StageXyzToLab::default();
            lab_to_xyz_stage.transform(&mut lut)?;
        }

        pcs_lab_v4_to_v2(dest, &mut lut);

        if dest.pcs == DataColorSpace::Xyz {
            if dest.is_matrix_shaper() {
                prepare_inverse_lut_rgb_xyz::<T, BIT_DEPTH, GAMMA_LUT>(dest, &mut lut, options)?;
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

        let is_dest_linear_profile = dest.color_space == DataColorSpace::Rgb
            && dest.is_matrix_shaper()
            && dest.is_linear_matrix_shaper();

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return Ok(make_transformer_4x3_avx_fma::<T, GRID_SIZE, BIT_DEPTH>(
                dst_layout,
                lut,
                options,
                dest.color_space,
                is_dest_linear_profile,
            ));
        }
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return Ok(make_transformer_4x3_sse41::<T, GRID_SIZE, BIT_DEPTH>(
                dst_layout,
                lut,
                options,
                dest.color_space,
                is_dest_linear_profile,
            ));
        }

        return Ok(make_transformer_4x3::<T, GRID_SIZE, BIT_DEPTH>(
            dst_layout,
            lut,
            options,
            dest.color_space,
            is_dest_linear_profile,
        ));
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

        pcs_lab_v2_to_v4(source, &mut lut);

        if source.pcs == DataColorSpace::Xyz && dest.pcs == DataColorSpace::Lab {
            let xyz_to_lab = StageXyzToLab::default();
            xyz_to_lab.transform(&mut lut)?;
        } else if source.pcs == DataColorSpace::Lab && dest.pcs == DataColorSpace::Xyz {
            let lab_to_xyz_stage = StageLabToXyz::default();
            lab_to_xyz_stage.transform(&mut lut)?;
        }

        pcs_lab_v4_to_v2(dest, &mut lut);

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

        return Ok(make_transform_3x4::<T, GRID_SIZE, BIT_DEPTH>(
            src_layout,
            lut,
            options,
            dest.color_space,
            is_dest_linear_profile,
        ));
    } else if (source.color_space.is_three_channels()) && (dest.color_space.is_three_channels()) {
        source.color_space.check_layout(src_layout)?;
        dest.color_space.check_layout(dst_layout)?;

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

        pcs_lab_v2_to_v4(source, &mut lut);

        if source.pcs == DataColorSpace::Xyz && dest.pcs == DataColorSpace::Lab {
            let xyz_to_lab = StageXyzToLab::default();
            xyz_to_lab.transform(&mut lut)?;
        } else if source.pcs == DataColorSpace::Lab && dest.pcs == DataColorSpace::Xyz {
            let lab_to_xyz_stage = StageLabToXyz::default();
            lab_to_xyz_stage.transform(&mut lut)?;
        }

        pcs_lab_v4_to_v2(dest, &mut lut);

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

        #[cfg(all(feature = "avx", target_arch = "x86_64"))]
        if std::arch::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return Ok(make_transformer_3x3_avx_fma::<T, GRID_SIZE, BIT_DEPTH>(
                src_layout,
                dst_layout,
                lut,
                options,
                dest.color_space,
                is_dest_linear_profile,
            ));
        }
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return Ok(make_transformer_3x3_sse41::<T, GRID_SIZE, BIT_DEPTH>(
                src_layout,
                dst_layout,
                lut,
                options,
                dest.color_space,
                is_dest_linear_profile,
            ));
        }

        return Ok(make_transformer_3x3::<T, GRID_SIZE, BIT_DEPTH>(
            src_layout,
            dst_layout,
            lut,
            options,
            dest.color_space,
            is_dest_linear_profile,
        ));
    }

    Err(CmsError::UnsupportedProfileConnection)
}

fn create_rgb_lin_lut<
    T: Copy + Default + AsPrimitive<f32> + Send + Sync + AsPrimitive<usize> + PointeeSizeExpressible,
    const BIT_DEPTH: usize,
    const LINEAR_CAP: usize,
    const GRID_SIZE: usize,
>(
    source: &ColorProfile,
    opts: TransformOptions,
) -> Result<Vec<f32>, CmsError>
where
    u32: AsPrimitive<T>,
    f32: AsPrimitive<T>,
{
    let lut_origins = create_lut3_samples::<T, GRID_SIZE>();

    let lin_r =
        source.build_r_linearize_table::<T, LINEAR_CAP, BIT_DEPTH>(opts.allow_use_cicp_transfer)?;
    let lin_g =
        source.build_g_linearize_table::<T, LINEAR_CAP, BIT_DEPTH>(opts.allow_use_cicp_transfer)?;
    let lin_b =
        source.build_b_linearize_table::<T, LINEAR_CAP, BIT_DEPTH>(opts.allow_use_cicp_transfer)?;

    let lin_stage = RgbLinearizationStage::<T, BIT_DEPTH, LINEAR_CAP, GRID_SIZE> {
        r_lin: lin_r,
        g_lin: lin_g,
        b_lin: lin_b,
        _phantom: PhantomData,
    };

    let mut lut = vec![0f32; lut_origins.len()];
    lin_stage.transform(&lut_origins, &mut lut)?;

    let xyz_to_rgb = source.rgb_to_xyz_matrix();

    let matrices = vec![
        xyz_to_rgb.to_f32(),
        Matrix3f {
            v: [
                [32768.0 / 65535.0, 0.0, 0.0],
                [0.0, 32768.0 / 65535.0, 0.0],
                [0.0, 0.0, 32768.0 / 65535.0],
            ],
        },
    ];

    let matrix_stage = MatrixStage { matrices };
    matrix_stage.transform(&mut lut)?;
    Ok(lut)
}

fn prepare_inverse_lut_rgb_xyz<
    T: Copy
        + Default
        + AsPrimitive<f32>
        + Send
        + Sync
        + AsPrimitive<usize>
        + PointeeSizeExpressible
        + GammaLutInterpolate,
    const BIT_DEPTH: usize,
    const GAMMA_LUT: usize,
>(
    dest: &ColorProfile,
    lut: &mut [f32],
    options: TransformOptions,
) -> Result<(), CmsError>
where
    f32: AsPrimitive<T>,
    u32: AsPrimitive<T>,
{
    if !T::FINITE {
        if let Some(extended_gamma) = dest.try_extended_gamma_evaluator() {
            let xyz_to_rgb = dest.rgb_to_xyz_matrix().inverse();

            let mut matrices = vec![Matrix3f {
                v: [
                    [65535.0 / 32768.0, 0.0, 0.0],
                    [0.0, 65535.0 / 32768.0, 0.0],
                    [0.0, 0.0, 65535.0 / 32768.0],
                ],
            }];

            matrices.push(xyz_to_rgb.to_f32());
            let xyz_to_rgb_stage = XyzToRgbStageExtended::<T> {
                gamma_evaluator: extended_gamma,
                matrices,
                phantom_data: PhantomData,
            };
            xyz_to_rgb_stage.transform(lut)?;
            return Ok(());
        }
    }
    let gamma_map_r = dest.build_gamma_table::<T, 65536, GAMMA_LUT, BIT_DEPTH>(
        &dest.red_trc,
        options.allow_use_cicp_transfer,
    )?;
    let gamma_map_g = dest.build_gamma_table::<T, 65536, GAMMA_LUT, BIT_DEPTH>(
        &dest.green_trc,
        options.allow_use_cicp_transfer,
    )?;
    let gamma_map_b = dest.build_gamma_table::<T, 65536, GAMMA_LUT, BIT_DEPTH>(
        &dest.blue_trc,
        options.allow_use_cicp_transfer,
    )?;

    let xyz_to_rgb = dest.rgb_to_xyz_matrix().inverse();

    let mut matrices = vec![Matrix3f {
        v: [
            [65535.0 / 32768.0, 0.0, 0.0],
            [0.0, 65535.0 / 32768.0, 0.0],
            [0.0, 0.0, 65535.0 / 32768.0],
        ],
    }];

    matrices.push(xyz_to_rgb.to_f32());
    let xyz_to_rgb_stage = XyzToRgbStage::<T, BIT_DEPTH, GAMMA_LUT> {
        r_gamma: gamma_map_r,
        g_gamma: gamma_map_g,
        b_gamma: gamma_map_b,
        matrices,
        intent: options.rendering_intent,
    };
    xyz_to_rgb_stage.transform(lut)?;
    Ok(())
}
