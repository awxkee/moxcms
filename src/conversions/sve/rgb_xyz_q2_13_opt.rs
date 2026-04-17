/*
 * // Copyright (c) Radzivon Bartoshyk 3/2025. All rights reserved.
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
use crate::conversions::rgbxyz_fixed::TransformMatrixShaperFpOptVec;
use crate::conversions::sve::{split_by_twos, split_by_twos_mut};
use crate::transform::PointeeSizeExpressible;
use crate::{CmsError, Layout, TransformExecutor};
use std::arch::aarch64::*;

pub(crate) struct TransformShaperQ2_13SveOpt<
    T: Copy,
    const SRC_LAYOUT: u8,
    const DST_LAYOUT: u8,
    const PRECISION: i32,
> {
    pub(crate) profile: TransformMatrixShaperFpOptVec<i16, i16, T>,
    #[allow(unused)]
    pub(crate) bit_depth: usize,
    pub(crate) gamma_lut: usize,
}

impl<const SRC_LAYOUT: u8, const DST_LAYOUT: u8, const PRECISION: i32> TransformExecutor<u8>
    for TransformShaperQ2_13SveOpt<u8, SRC_LAYOUT, DST_LAYOUT, PRECISION>
{
    fn transform(&self, src: &[u8], dst: &mut [u8]) -> Result<(), CmsError> {
        unsafe { self.transform_impl(src, dst) }
    }
}

impl<const SRC_LAYOUT: u8, const DST_LAYOUT: u8, const PRECISION: i32>
    TransformShaperQ2_13SveOpt<u8, SRC_LAYOUT, DST_LAYOUT, PRECISION>
{
    #[target_feature(enable = "sve2", enable = "sve")]
    fn transform_impl(&self, src: &[u8], dst: &mut [u8]) -> Result<(), CmsError> {
        let src_cn = Layout::from(SRC_LAYOUT);
        let dst_cn = Layout::from(DST_LAYOUT);
        let src_channels = src_cn.channels();
        let dst_channels = dst_cn.channels();

        if src.len() / src_channels != dst.len() / dst_channels {
            return Err(CmsError::LaneSizeMismatch);
        }
        if !src.len().is_multiple_of(src_channels) {
            return Err(CmsError::LaneMultipleOfChannels);
        }
        if !dst.len().is_multiple_of(dst_channels) {
            return Err(CmsError::LaneMultipleOfChannels);
        }

        let t = self.profile.adaptation_matrix.transpose();

        let lut_lin = &self.profile.linear;
        assert_lut_min_len!(u8, lut_lin.len());

        let (src_chunks, src_remainder) = split_by_twos(src, src_channels);
        let (dst_chunks, dst_remainder) = split_by_twos_mut(dst, dst_channels);

        // RGB + G B R + B R G

        let pv4_32 = svwhilelt_b32_u64(0u64, 4u64);

        let pv_src = svwhilelt_b8_u64(0u64, src_channels as u64);
        let pv_dst = svwhilelt_b8_u64(0u64, dst_channels as u64);

        let v_max_value = svdup_n_u16((self.gamma_lut - 1) as u16);

        let rnd = svdup_n_s32(1 << (PRECISION - 1));

        let alpha_pos_mask = svcmpeq_n_u8(
            svptrue_b8(),
            svand_n_u8_x(svptrue_b8(), svindex_u8(0u8, 1u8), 3u8),
            3u8,
        );

        if !src_chunks.is_empty() {
            let pv_src2 = svwhilelt_b8_u64(0u64, src_channels as u64 * 2);
            let pv_dst2 = svwhilelt_b8_u64(0u64, dst_channels as u64 * 2);

            let rrr_tbl: [u8; 16] = [
                0,
                255,
                255,
                255,
                src_channels as u8,
                255,
                255,
                255,
                src_channels as u8 * 2,
                255,
                255,
                255,
                src_channels as u8 * 3,
                255,
                255,
                255,
            ];
            let rrr_tbl = unsafe { svld1_u8(svwhilelt_b8_u32(0u32, 16u32), rrr_tbl.as_ptr()) };

            let ggg_tbl: [u8; 16] = [
                1,
                255,
                255,
                255,
                src_channels as u8 + 1,
                255,
                255,
                255,
                src_channels as u8 * 2 + 1,
                255,
                255,
                255,
                src_channels as u8 * 3 + 1,
                255,
                255,
                255,
            ];
            let ggg_tbl = unsafe { svld1_u8(svwhilelt_b8_u32(0u32, 16u32), ggg_tbl.as_ptr()) };

            let bbb_tbl: [u8; 16] = [
                2,
                255,
                255,
                255,
                src_channels as u8 + 2,
                255,
                255,
                255,
                src_channels as u8 * 2 + 2,
                255,
                255,
                255,
                src_channels as u8 * 3 + 2,
                255,
                255,
                255,
            ];
            let bbb_tbl = unsafe { svld1_u8(svwhilelt_b8_u32(0u32, 16u32), bbb_tbl.as_ptr()) };

            static COMPRESS_TABLE2: [u8; 16] = [
                0, 1, 2, 4, 5, 6, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            ];
            let compress_tbl =
                unsafe { svld1_u8(svwhilelt_b8_u32(0u32, 16u32), COMPRESS_TABLE2.as_ptr()) };

            let pv4_16 = svwhilelt_b16_u64(0u64, 4u64);

            let m0 = unsafe { svld1_s16(pv4_16, [t.v[0][0], t.v[0][1], t.v[0][2], 0].as_ptr()) };
            let m1 = unsafe { svld1_s16(pv4_16, [t.v[1][0], t.v[1][1], t.v[1][2], 0].as_ptr()) };
            let m2 = unsafe { svld1_s16(pv4_16, [t.v[2][0], t.v[2][1], t.v[2][2], 0].as_ptr()) };

            let (src0, src1) = src_chunks.split_at(src_chunks.len() / 2);
            let (dst0, dst1) = dst_chunks.split_at_mut(dst_chunks.len() / 2);
            let mut src_iter0 = src0.chunks_exact(src_channels * 2);
            let mut src_iter1 = src1.chunks_exact(src_channels * 2);

            let mut full_lane;
            let mut lane0;
            let mut lane1;
            let mut lane2;

            if let (Some(src0), Some(src1)) = (src_iter0.next(), src_iter1.next()) {
                let full_lane0 = unsafe { svld1_u8(pv_src2, src0.as_ptr()) };
                let full_lane1 = unsafe { svld1_u8(pv_src2, src1.as_ptr()) };
                full_lane = svsplice_u8(pv_src2, full_lane0, full_lane1);

                let q0 = svreinterpret_u32_u8(svtbl_u8(full_lane, rrr_tbl));
                let q1 = svreinterpret_u32_u8(svtbl_u8(full_lane, ggg_tbl));
                let q2 = svreinterpret_u32_u8(svtbl_u8(full_lane, bbb_tbl));

                lane0 = svreinterpret_s16_u32(unsafe {
                    svld1uh_gather_u32index_u32(pv4_32, lut_lin.as_ptr().cast(), q0)
                });
                lane1 = svreinterpret_s16_u32(unsafe {
                    svld1uh_gather_u32index_u32(pv4_32, lut_lin.as_ptr().cast(), q1)
                });
                lane2 = svreinterpret_s16_u32(unsafe {
                    svld1uh_gather_u32index_u32(pv4_32, lut_lin.as_ptr().cast(), q2)
                });
            } else {
                full_lane = svdup_n_u8(0);
                lane0 = svdup_n_s16(0);
                lane1 = svdup_n_s16(0);
                lane2 = svdup_n_s16(0);
            }

            for (((src0, src1), dst0), dst1) in src_iter0
                .zip(src_iter1)
                .zip(dst0.chunks_exact_mut(dst_channels * 2))
                .zip(dst1.chunks_exact_mut(dst_channels * 2))
            {
                let mut v_0 = svmlalb_lane_s32::<0>(rnd, lane0, m0); // all RRRR
                v_0 = svmlalb_lane_s32::<0>(v_0, lane1, m1);
                v_0 = svmlalb_lane_s32::<0>(v_0, lane2, m2);

                let mut v_1 = svmlalb_lane_s32::<1>(rnd, lane0, m0); // all GGGG
                v_1 = svmlalb_lane_s32::<1>(v_1, lane1, m1);
                v_1 = svmlalb_lane_s32::<1>(v_1, lane2, m2);

                let mut v_2 = svmlalb_lane_s32::<2>(rnd, lane0, m0); // all BBBB
                v_2 = svmlalb_lane_s32::<2>(v_2, lane1, m1);
                v_2 = svmlalb_lane_s32::<2>(v_2, lane2, m2);

                let mut vr0 = svqshrunb_n_s32::<PRECISION>(v_0);
                let mut vr1 = svqshrunb_n_s32::<PRECISION>(v_1);
                let mut vr2 = svqshrunb_n_s32::<PRECISION>(v_2);

                vr0 = svmin_u16_x(svptrue_b16(), vr0, v_max_value);
                vr1 = svmin_u16_x(svptrue_b16(), vr1, v_max_value);
                vr2 = svmin_u16_x(svptrue_b16(), vr2, v_max_value);

                let qfl = full_lane;

                let full_lane0 = unsafe { svld1_u8(pv_src2, src0.as_ptr()) };
                let full_lane1 = unsafe { svld1_u8(pv_src2, src1.as_ptr()) };

                let vals0 = svreinterpret_u8_u32(unsafe {
                    svld1ub_gather_u32offset_u32(
                        pv4_32,
                        self.profile.gamma.as_ptr().cast(),
                        svreinterpret_u32_u16(vr0),
                    )
                });

                full_lane = svsplice_u8(pv_src2, full_lane0, full_lane1);

                let vals1 = svreinterpret_u8_u32(unsafe {
                    svld1ub_gather_u32offset_u32(
                        pv4_32,
                        self.profile.gamma.as_ptr().cast(),
                        svreinterpret_u32_u16(vr1),
                    )
                });

                let q0 = svreinterpret_u32_u8(svtbl_u8(full_lane, rrr_tbl));
                let q1 = svreinterpret_u32_u8(svtbl_u8(full_lane, ggg_tbl));
                let q2 = svreinterpret_u32_u8(svtbl_u8(full_lane, bbb_tbl));

                let vals2 = svreinterpret_u8_u32(unsafe {
                    svld1ub_gather_u32offset_u32(
                        pv4_32,
                        self.profile.gamma.as_ptr().cast(),
                        svreinterpret_u32_u16(vr2),
                    )
                });

                lane0 = svreinterpret_s16_u32(unsafe {
                    svld1uh_gather_u32index_u32(pv4_32, lut_lin.as_ptr().cast(), q0)
                });
                lane1 = svreinterpret_s16_u32(unsafe {
                    svld1uh_gather_u32index_u32(pv4_32, lut_lin.as_ptr().cast(), q1)
                });
                lane2 = svreinterpret_s16_u32(unsafe {
                    svld1uh_gather_u32index_u32(pv4_32, lut_lin.as_ptr().cast(), q2)
                });

                let r = svuzp1_u8(vals0, svdup_n_u8(0));
                let g = svuzp1_u8(vals1, svdup_n_u8(0));
                let b = svuzp1_u8(vals2, svdup_n_u8(0));

                let r = svuzp1_u8(r, svdup_n_u8(0));
                let g = svuzp1_u8(g, svdup_n_u8(0));
                let b = svuzp1_u8(b, svdup_n_u8(0));

                let rg = svzip1_u8(r, g);
                let ba = svzip1_u8(b, svdup_n_u8(255));
                let mut rgba = svreinterpret_u8_u16(svzip1_u16(
                    svreinterpret_u16_u8(rg),
                    svreinterpret_u16_u8(ba),
                ));

                if dst_channels == 3 {
                    let mut rs0p = rgba;
                    let mut rs1p = svext_u8::<8>(rgba, svdup_n_u8(0));
                    rs0p = svtbl_u8(rs0p, compress_tbl);
                    rs1p = svtbl_u8(rs1p, compress_tbl);
                    unsafe {
                        svst1_u8(pv_dst2, dst0.as_mut_ptr().cast(), rs0p);
                    }
                    unsafe {
                        svst1_u8(pv_dst2, dst1.as_mut_ptr().cast(), rs1p);
                    }
                } else if dst_channels == 4 {
                    if src_channels == 4 {
                        rgba = svsel_u8(alpha_pos_mask, qfl, rgba);
                    }
                    let rs0p = rgba;
                    let rs1p = svext_u8::<8>(rgba, svdup_n_u8(0));

                    unsafe {
                        svst1_u8(pv_dst2, dst0.as_mut_ptr().cast(), rs0p);
                    }
                    unsafe {
                        svst1_u8(pv_dst2, dst1.as_mut_ptr().cast(), rs1p);
                    }
                }
            }

            if let (Some(dst0), Some(dst1)) = (
                dst0.chunks_exact_mut(dst_channels * 2).last(),
                dst1.chunks_exact_mut(dst_channels * 2).last(),
            ) {
                let mut v_0 = svmlalb_lane_s32::<0>(rnd, lane0, m0); // all RRRR
                v_0 = svmlalb_lane_s32::<0>(v_0, lane1, m1);
                v_0 = svmlalb_lane_s32::<0>(v_0, lane2, m2);

                let mut v_1 = svmlalb_lane_s32::<1>(rnd, lane0, m0); // all GGGG
                v_1 = svmlalb_lane_s32::<1>(v_1, lane1, m1);
                v_1 = svmlalb_lane_s32::<1>(v_1, lane2, m2);

                let mut v_2 = svmlalb_lane_s32::<2>(rnd, lane0, m0); // all BBBB
                v_2 = svmlalb_lane_s32::<2>(v_2, lane1, m1);
                v_2 = svmlalb_lane_s32::<2>(v_2, lane2, m2);

                let mut vr0 = svqshrunb_n_s32::<PRECISION>(v_0);
                let mut vr1 = svqshrunb_n_s32::<PRECISION>(v_1);
                let mut vr2 = svqshrunb_n_s32::<PRECISION>(v_2);

                vr0 = svmin_u16_x(svptrue_b16(), vr0, v_max_value);
                vr1 = svmin_u16_x(svptrue_b16(), vr1, v_max_value);
                vr2 = svmin_u16_x(svptrue_b16(), vr2, v_max_value);

                let vals0 = svreinterpret_u8_u32(unsafe {
                    svld1ub_gather_u32offset_u32(
                        pv4_32,
                        self.profile.gamma.as_ptr().cast(),
                        svreinterpret_u32_u16(vr0),
                    )
                });

                let vals1 = svreinterpret_u8_u32(unsafe {
                    svld1ub_gather_u32offset_u32(
                        pv4_32,
                        self.profile.gamma.as_ptr().cast(),
                        svreinterpret_u32_u16(vr1),
                    )
                });

                let vals2 = svreinterpret_u8_u32(unsafe {
                    svld1ub_gather_u32offset_u32(
                        pv4_32,
                        self.profile.gamma.as_ptr().cast(),
                        svreinterpret_u32_u16(vr2),
                    )
                });

                let r = svuzp1_u8(vals0, svdup_n_u8(0));
                let g = svuzp1_u8(vals1, svdup_n_u8(0));
                let b = svuzp1_u8(vals2, svdup_n_u8(0));

                let r = svuzp1_u8(r, svdup_n_u8(0));
                let g = svuzp1_u8(g, svdup_n_u8(0));
                let b = svuzp1_u8(b, svdup_n_u8(0));

                let rg = svzip1_u8(r, g);
                let ba = svzip1_u8(b, svdup_n_u8(255));
                let mut rgba = svreinterpret_u8_u16(svzip1_u16(
                    svreinterpret_u16_u8(rg),
                    svreinterpret_u16_u8(ba),
                ));

                if dst_channels == 3 {
                    let mut rs0p = rgba;
                    let mut rs1p = svext_u8::<8>(rgba, svdup_n_u8(0));
                    rs0p = svtbl_u8(rs0p, compress_tbl);
                    rs1p = svtbl_u8(rs1p, compress_tbl);
                    unsafe {
                        svst1_u8(pv_dst2, dst0.as_mut_ptr().cast(), rs0p);
                    }
                    unsafe {
                        svst1_u8(pv_dst2, dst1.as_mut_ptr().cast(), rs1p);
                    }
                } else if dst_channels == 4 {
                    if src_channels == 4 {
                        rgba = svsel_u8(alpha_pos_mask, full_lane, rgba);
                    }
                    let rs0p = rgba;
                    let rs1p = svext_u8::<8>(rgba, svdup_n_u8(0));

                    unsafe {
                        svst1_u8(pv_dst2, dst0.as_mut_ptr().cast(), rs0p);
                    }
                    unsafe {
                        svst1_u8(pv_dst2, dst1.as_mut_ptr().cast(), rs1p);
                    }
                }
            }
        }

        static SHUF_TABLE_1: [u8; 16] = [
            0, 255, 255, 255, 1, 255, 255, 255, 2, 255, 255, 255, 255, 255, 255, 255,
        ];
        let shuf1_tbl = unsafe { svld1_u8(svwhilelt_b8_u32(0u32, 16u32), SHUF_TABLE_1.as_ptr()) };
        static PACK_TABLE_1: [u8; 16] = [
            0, 4, 8, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        ];
        let pack1_tbl = unsafe { svld1_u8(svwhilelt_b8_u32(0u32, 16u32), PACK_TABLE_1.as_ptr()) };
        static PERMI_1: [u16; 8] = [2, 255, 4, 255, 0, 255, 255, 255];
        let permi1 = unsafe { svld1_u16(svwhilelt_b16_u32(0u32, 8u32), PERMI_1.as_ptr()) };
        static PERMI_2: [u16; 8] = [4, 255, 0, 255, 2, 255, 255, 255];
        let permi2 = unsafe { svld1_u16(svwhilelt_b16_u32(0u32, 8u32), PERMI_2.as_ptr()) };

        let pv6_16 = svwhilelt_b16_u64(0u64, 6u64);
        let pv3_32 = svwhilelt_b32_u64(0u64, 3u64);

        let m0 = unsafe { svld1_s16(pv6_16, [t.v[0][0], 0, t.v[1][1], 0, t.v[2][2], 0].as_ptr()) }; // R G B
        let m1 = unsafe { svld1_s16(pv6_16, [t.v[1][0], 0, t.v[2][1], 0, t.v[0][2], 0].as_ptr()) }; // G B R
        let m2 = unsafe { svld1_s16(pv6_16, [t.v[2][0], 0, t.v[0][1], 0, t.v[1][2], 0].as_ptr()) }; // B R G

        for (src, dst) in src_remainder
            .chunks_exact(src_channels)
            .zip(dst_remainder.chunks_exact_mut(dst_channels))
        {
            let lane = unsafe { svld1_u8(pv_src, src.as_ptr()) };
            let vals32 = svreinterpret_u32_u8(svtbl_u8(lane, shuf1_tbl));
            let linearized = svreinterpret_s16_u32(unsafe {
                svld1uh_gather_u32index_u32(pv3_32, lut_lin.as_ptr().cast(), vals32)
            });
            let q0 = svtbl_s16(linearized, permi1);
            let q1 = svtbl_s16(linearized, permi2);
            let v0 = svmlalb_s32(rnd, linearized, m0);
            let v1 = svmlalb_s32(v0, q0, m1);
            let v = svmlalb_s32(v1, q1, m2);

            let mut vr0 = svqshrunb_n_s32::<PRECISION>(v);
            vr0 = svmin_u16_x(svptrue_b16(), vr0, v_max_value);
            let gamma = svreinterpret_u8_u32(unsafe {
                svld1ub_gather_u32offset_u32(
                    pv3_32,
                    self.profile.gamma.as_ptr().cast(),
                    svreinterpret_u32_u16(vr0),
                )
            });
            let mut result = svtbl_u8(gamma, pack1_tbl);
            if dst_channels == 4 {
                if src_channels == 4 {
                    result = svsel_u8(alpha_pos_mask, lane, result)
                } else {
                    result = svsel_u8(alpha_pos_mask, svdup_n_u8(255), result)
                }
            }
            unsafe {
                svst1_u8(pv_dst, dst.as_mut_ptr().cast(), result);
            }
        }

        Ok(())
    }
}

#[cfg(feature = "in_place")]
use crate::InPlaceTransformExecutor;

#[cfg(feature = "in_place")]
impl<const SRC_LAYOUT: u8, const DST_LAYOUT: u8, const PRECISION: i32>
    TransformShaperQ2_13SveOpt<u8, SRC_LAYOUT, DST_LAYOUT, PRECISION>
{
    #[target_feature(enable = "sve2", enable = "sve")]
    fn transform_impl_in_place(&self, dst: &mut [u8]) -> Result<(), CmsError> {
        let src_cn = Layout::from(SRC_LAYOUT);
        assert_eq!(
            SRC_LAYOUT, DST_LAYOUT,
            "This is in-place transform, layout must not diverge"
        );
        let src_channels = src_cn.channels();

        if !dst.len().is_multiple_of(src_channels) {
            return Err(CmsError::LaneMultipleOfChannels);
        }

        let t = self.profile.adaptation_matrix.transpose();

        let lut_lin = &self.profile.linear;
        assert_lut_min_len!(u8, lut_lin.len());

        let (dst_chunks, dst_remainder) = split_by_twos_mut(dst, src_channels);

        // RGB + G B R + B R G

        let pv4_32 = svwhilelt_b32_u64(0u64, 4u64);

        let pv_src = svwhilelt_b8_u64(0u64, src_channels as u64);

        let v_max_value = svdup_n_u16((self.gamma_lut - 1) as u16);

        let rnd = svdup_n_s32(1 << (PRECISION - 1));

        let alpha_pos_mask = svcmpeq_n_u8(
            svptrue_b8(),
            svand_n_u8_x(svptrue_b8(), svindex_u8(0u8, 1u8), 3u8),
            3u8,
        );

        if !dst_chunks.is_empty() {
            let pv_src2 = svwhilelt_b8_u64(0u64, src_channels as u64 * 2);
            let pv_dst2 = pv_src2;

            let rrr_tbl: [u8; 16] = [
                0,
                255,
                255,
                255,
                src_channels as u8,
                255,
                255,
                255,
                src_channels as u8 * 2,
                255,
                255,
                255,
                src_channels as u8 * 3,
                255,
                255,
                255,
            ];
            let rrr_tbl = unsafe { svld1_u8(svwhilelt_b8_u32(0u32, 16u32), rrr_tbl.as_ptr()) };

            let ggg_tbl: [u8; 16] = [
                1,
                255,
                255,
                255,
                src_channels as u8 + 1,
                255,
                255,
                255,
                src_channels as u8 * 2 + 1,
                255,
                255,
                255,
                src_channels as u8 * 3 + 1,
                255,
                255,
                255,
            ];
            let ggg_tbl = unsafe { svld1_u8(svwhilelt_b8_u32(0u32, 16u32), ggg_tbl.as_ptr()) };

            let bbb_tbl: [u8; 16] = [
                2,
                255,
                255,
                255,
                src_channels as u8 + 2,
                255,
                255,
                255,
                src_channels as u8 * 2 + 2,
                255,
                255,
                255,
                src_channels as u8 * 3 + 2,
                255,
                255,
                255,
            ];
            let bbb_tbl = unsafe { svld1_u8(svwhilelt_b8_u32(0u32, 16u32), bbb_tbl.as_ptr()) };

            static COMPRESS_TABLE2: [u8; 16] = [
                0, 1, 2, 4, 5, 6, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            ];
            let compress_tbl =
                unsafe { svld1_u8(svwhilelt_b8_u32(0u32, 16u32), COMPRESS_TABLE2.as_ptr()) };

            let pv4_16 = svwhilelt_b16_u64(0u64, 4u64);

            let m0 = unsafe { svld1_s16(pv4_16, [t.v[0][0], t.v[0][1], t.v[0][2], 0].as_ptr()) };
            let m1 = unsafe { svld1_s16(pv4_16, [t.v[1][0], t.v[1][1], t.v[1][2], 0].as_ptr()) };
            let m2 = unsafe { svld1_s16(pv4_16, [t.v[2][0], t.v[2][1], t.v[2][2], 0].as_ptr()) };

            let (chunk0, chunk1) = dst_chunks.split_at_mut(dst_chunks.len() / 2);
            let chunk_len = chunk0.len() / (src_channels * 2);
            let chunk0_ptr = chunk0;
            let chunk1_ptr = chunk1;

            let mut full_lane;
            let mut lane0;
            let mut lane1;
            let mut lane2;

            if chunk_len > 0 {
                let full_lane0 = unsafe { svld1_u8(pv_src2, chunk0_ptr.as_ptr()) };
                let full_lane1 = unsafe { svld1_u8(pv_src2, chunk1_ptr.as_ptr()) };
                full_lane = svsplice_u8(pv_src2, full_lane0, full_lane1);

                let q0 = svreinterpret_u32_u8(svtbl_u8(full_lane, rrr_tbl));
                let q1 = svreinterpret_u32_u8(svtbl_u8(full_lane, ggg_tbl));
                let q2 = svreinterpret_u32_u8(svtbl_u8(full_lane, bbb_tbl));

                lane0 = svreinterpret_s16_u32(unsafe {
                    svld1uh_gather_u32index_u32(pv4_32, lut_lin.as_ptr().cast(), q0)
                });
                lane1 = svreinterpret_s16_u32(unsafe {
                    svld1uh_gather_u32index_u32(pv4_32, lut_lin.as_ptr().cast(), q1)
                });
                lane2 = svreinterpret_s16_u32(unsafe {
                    svld1uh_gather_u32index_u32(pv4_32, lut_lin.as_ptr().cast(), q2)
                });
            } else {
                full_lane = svdup_n_u8(0);
                lane0 = svdup_n_s16(0);
                lane1 = svdup_n_s16(0);
                lane2 = svdup_n_s16(0);
            }

            for i in 1..chunk_len {
                let c0 = unsafe { chunk0_ptr.get_unchecked(i * src_channels * 2..) };
                let c1 = unsafe { chunk1_ptr.get_unchecked(i * src_channels * 2..) };

                let mut v_0 = svmlalb_lane_s32::<0>(rnd, lane0, m0); // all RRRR
                v_0 = svmlalb_lane_s32::<0>(v_0, lane1, m1);
                v_0 = svmlalb_lane_s32::<0>(v_0, lane2, m2);

                let mut v_1 = svmlalb_lane_s32::<1>(rnd, lane0, m0); // all GGGG
                v_1 = svmlalb_lane_s32::<1>(v_1, lane1, m1);
                v_1 = svmlalb_lane_s32::<1>(v_1, lane2, m2);

                let mut v_2 = svmlalb_lane_s32::<2>(rnd, lane0, m0); // all BBBB
                v_2 = svmlalb_lane_s32::<2>(v_2, lane1, m1);
                v_2 = svmlalb_lane_s32::<2>(v_2, lane2, m2);

                let mut vr0 = svqshrunb_n_s32::<PRECISION>(v_0);
                let mut vr1 = svqshrunb_n_s32::<PRECISION>(v_1);
                let mut vr2 = svqshrunb_n_s32::<PRECISION>(v_2);

                vr0 = svmin_u16_x(svptrue_b16(), vr0, v_max_value);
                vr1 = svmin_u16_x(svptrue_b16(), vr1, v_max_value);
                vr2 = svmin_u16_x(svptrue_b16(), vr2, v_max_value);

                let qfl = full_lane;

                let full_lane0 = unsafe { svld1_u8(pv_src2, c0.as_ptr()) };
                let full_lane1 = unsafe { svld1_u8(pv_src2, c1.as_ptr()) };

                let vals0 = svreinterpret_u8_u32(unsafe {
                    svld1ub_gather_u32offset_u32(
                        pv4_32,
                        self.profile.gamma.as_ptr().cast(),
                        svreinterpret_u32_u16(vr0),
                    )
                });

                full_lane = svsplice_u8(pv_src2, full_lane0, full_lane1);

                let vals1 = svreinterpret_u8_u32(unsafe {
                    svld1ub_gather_u32offset_u32(
                        pv4_32,
                        self.profile.gamma.as_ptr().cast(),
                        svreinterpret_u32_u16(vr1),
                    )
                });

                let q0 = svreinterpret_u32_u8(svtbl_u8(full_lane, rrr_tbl));
                let q1 = svreinterpret_u32_u8(svtbl_u8(full_lane, ggg_tbl));
                let q2 = svreinterpret_u32_u8(svtbl_u8(full_lane, bbb_tbl));

                let vals2 = svreinterpret_u8_u32(unsafe {
                    svld1ub_gather_u32offset_u32(
                        pv4_32,
                        self.profile.gamma.as_ptr().cast(),
                        svreinterpret_u32_u16(vr2),
                    )
                });

                lane0 = svreinterpret_s16_u32(unsafe {
                    svld1uh_gather_u32index_u32(pv4_32, lut_lin.as_ptr().cast(), q0)
                });
                lane1 = svreinterpret_s16_u32(unsafe {
                    svld1uh_gather_u32index_u32(pv4_32, lut_lin.as_ptr().cast(), q1)
                });
                lane2 = svreinterpret_s16_u32(unsafe {
                    svld1uh_gather_u32index_u32(pv4_32, lut_lin.as_ptr().cast(), q2)
                });

                let r = svuzp1_u8(vals0, svdup_n_u8(0));
                let g = svuzp1_u8(vals1, svdup_n_u8(0));
                let b = svuzp1_u8(vals2, svdup_n_u8(0));

                let r = svuzp1_u8(r, svdup_n_u8(0));
                let g = svuzp1_u8(g, svdup_n_u8(0));
                let b = svuzp1_u8(b, svdup_n_u8(0));

                let rg = svzip1_u8(r, g);
                let ba = svzip1_u8(b, svdup_n_u8(255));
                let mut rgba = svreinterpret_u8_u16(svzip1_u16(
                    svreinterpret_u16_u8(rg),
                    svreinterpret_u16_u8(ba),
                ));

                let w0 = unsafe { chunk0_ptr.get_unchecked_mut((i - 1) * src_channels * 2..) };
                let w1 = unsafe { chunk1_ptr.get_unchecked_mut((i - 1) * src_channels * 2..) };

                if src_channels == 3 {
                    let mut rs0p = rgba;
                    let mut rs1p = svext_u8::<8>(rgba, svdup_n_u8(0));
                    rs0p = svtbl_u8(rs0p, compress_tbl);
                    rs1p = svtbl_u8(rs1p, compress_tbl);
                    unsafe {
                        svst1_u8(pv_dst2, w0.as_mut_ptr(), rs0p);
                    }
                    unsafe {
                        svst1_u8(pv_dst2, w1.as_mut_ptr(), rs1p);
                    }
                } else if src_channels == 4 {
                    rgba = svsel_u8(alpha_pos_mask, qfl, rgba);
                    let rs0p = rgba;
                    let rs1p = svext_u8::<8>(rgba, svdup_n_u8(0));
                    unsafe {
                        svst1_u8(pv_dst2, w0.as_mut_ptr(), rs0p);
                    }
                    unsafe {
                        svst1_u8(pv_dst2, w1.as_mut_ptr(), rs1p);
                    }
                }
            }

            // epilogue: flush last primed chunk (index chunk_len-1)
            if chunk_len > 0 {
                let w0 =
                    unsafe { chunk0_ptr.get_unchecked_mut((chunk_len - 1) * src_channels * 2) };
                let w1 =
                    unsafe { chunk1_ptr.get_unchecked_mut((chunk_len - 1) * src_channels * 2) };

                let mut v_0 = svmlalb_lane_s32::<0>(rnd, lane0, m0);
                v_0 = svmlalb_lane_s32::<0>(v_0, lane1, m1);
                v_0 = svmlalb_lane_s32::<0>(v_0, lane2, m2);

                let mut v_1 = svmlalb_lane_s32::<1>(rnd, lane0, m0);
                v_1 = svmlalb_lane_s32::<1>(v_1, lane1, m1);
                v_1 = svmlalb_lane_s32::<1>(v_1, lane2, m2);

                let mut v_2 = svmlalb_lane_s32::<2>(rnd, lane0, m0);
                v_2 = svmlalb_lane_s32::<2>(v_2, lane1, m1);
                v_2 = svmlalb_lane_s32::<2>(v_2, lane2, m2);

                let mut vr0 = svqshrunb_n_s32::<PRECISION>(v_0);
                let mut vr1 = svqshrunb_n_s32::<PRECISION>(v_1);
                let mut vr2 = svqshrunb_n_s32::<PRECISION>(v_2);

                vr0 = svmin_u16_x(svptrue_b16(), vr0, v_max_value);
                vr1 = svmin_u16_x(svptrue_b16(), vr1, v_max_value);
                vr2 = svmin_u16_x(svptrue_b16(), vr2, v_max_value);

                let vals0 = svreinterpret_u8_u32(unsafe {
                    svld1ub_gather_u32offset_u32(
                        pv4_32,
                        self.profile.gamma.as_ptr().cast(),
                        svreinterpret_u32_u16(vr0),
                    )
                });
                let vals1 = svreinterpret_u8_u32(unsafe {
                    svld1ub_gather_u32offset_u32(
                        pv4_32,
                        self.profile.gamma.as_ptr().cast(),
                        svreinterpret_u32_u16(vr1),
                    )
                });
                let vals2 = svreinterpret_u8_u32(unsafe {
                    svld1ub_gather_u32offset_u32(
                        pv4_32,
                        self.profile.gamma.as_ptr().cast(),
                        svreinterpret_u32_u16(vr2),
                    )
                });

                let r = svuzp1_u8(vals0, svdup_n_u8(0));
                let g = svuzp1_u8(vals1, svdup_n_u8(0));
                let b = svuzp1_u8(vals2, svdup_n_u8(0));

                let r = svuzp1_u8(r, svdup_n_u8(0));
                let g = svuzp1_u8(g, svdup_n_u8(0));
                let b = svuzp1_u8(b, svdup_n_u8(0));

                let rg = svzip1_u8(r, g);
                let ba = svzip1_u8(b, svdup_n_u8(255));
                let mut rgba = svreinterpret_u8_u16(svzip1_u16(
                    svreinterpret_u16_u8(rg),
                    svreinterpret_u16_u8(ba),
                ));

                if src_channels == 3 {
                    let mut rs0p = rgba;
                    let mut rs1p = svext_u8::<8>(rgba, svdup_n_u8(0));
                    rs0p = svtbl_u8(rs0p, compress_tbl);
                    rs1p = svtbl_u8(rs1p, compress_tbl);
                    unsafe {
                        svst1_u8(pv_dst2, w0, rs0p);
                    }
                    unsafe {
                        svst1_u8(pv_dst2, w1, rs1p);
                    }
                } else if src_channels == 4 {
                    rgba = svsel_u8(alpha_pos_mask, full_lane, rgba);
                    let rs0p = rgba;
                    let rs1p = svext_u8::<8>(rgba, svdup_n_u8(0));
                    unsafe {
                        svst1_u8(pv_dst2, w0, rs0p);
                    }
                    unsafe {
                        svst1_u8(pv_dst2, w1, rs1p);
                    }
                }
            }
        }

        static SHUF_TABLE_1: [u8; 16] = [
            0, 255, 255, 255, 1, 255, 255, 255, 2, 255, 255, 255, 255, 255, 255, 255,
        ];
        let shuf1_tbl = unsafe { svld1_u8(svwhilelt_b8_u32(0u32, 16u32), SHUF_TABLE_1.as_ptr()) };
        static PACK_TABLE_1: [u8; 16] = [
            0, 4, 8, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        ];
        let pack1_tbl = unsafe { svld1_u8(svwhilelt_b8_u32(0u32, 16u32), PACK_TABLE_1.as_ptr()) };
        static PERMI_1: [u16; 8] = [2, 255, 4, 255, 0, 255, 255, 255];
        let permi1 = unsafe { svld1_u16(svwhilelt_b16_u32(0u32, 8u32), PERMI_1.as_ptr()) };
        static PERMI_2: [u16; 8] = [4, 255, 0, 255, 2, 255, 255, 255];
        let permi2 = unsafe { svld1_u16(svwhilelt_b16_u32(0u32, 8u32), PERMI_2.as_ptr()) };

        let pv6_16 = svwhilelt_b16_u64(0u64, 6u64);
        let pv3_32 = svwhilelt_b32_u64(0u64, 3u64);

        let m0 = unsafe { svld1_s16(pv6_16, [t.v[0][0], 0, t.v[1][1], 0, t.v[2][2], 0].as_ptr()) }; // R G B
        let m1 = unsafe { svld1_s16(pv6_16, [t.v[1][0], 0, t.v[2][1], 0, t.v[0][2], 0].as_ptr()) }; // G B R
        let m2 = unsafe { svld1_s16(pv6_16, [t.v[2][0], 0, t.v[0][1], 0, t.v[1][2], 0].as_ptr()) }; // B R G

        for dst in dst_remainder.chunks_exact_mut(src_channels) {
            let lane = unsafe { svld1_u8(pv_src, dst.as_ptr()) };
            let vals32 = svreinterpret_u32_u8(svtbl_u8(lane, shuf1_tbl));
            let linearized = svreinterpret_s16_u32(unsafe {
                svld1uh_gather_u32index_u32(pv3_32, lut_lin.as_ptr().cast(), vals32)
            });
            let q0 = svtbl_s16(linearized, permi1);
            let q1 = svtbl_s16(linearized, permi2);
            let v0 = svmlalb_s32(rnd, linearized, m0);
            let v1 = svmlalb_s32(v0, q0, m1);
            let v = svmlalb_s32(v1, q1, m2);

            let mut vr0 = svqshrunb_n_s32::<PRECISION>(v);
            vr0 = svmin_u16_x(svptrue_b16(), vr0, v_max_value);
            let gamma = svreinterpret_u8_u32(unsafe {
                svld1ub_gather_u32offset_u32(
                    pv3_32,
                    self.profile.gamma.as_ptr().cast(),
                    svreinterpret_u32_u16(vr0),
                )
            });
            let mut result = svtbl_u8(gamma, pack1_tbl);
            if src_channels == 4 {
                result = svsel_u8(alpha_pos_mask, lane, result)
            } else {
                result = svsel_u8(alpha_pos_mask, svdup_n_u8(255), result)
            }

            unsafe {
                svst1_u8(pv_src, dst.as_mut_ptr().cast(), result);
            }
        }

        Ok(())
    }
}

#[cfg(feature = "in_place")]
impl<const SRC_LAYOUT: u8, const DST_LAYOUT: u8, const PRECISION: i32> InPlaceTransformExecutor<u8>
    for TransformShaperQ2_13SveOpt<u8, SRC_LAYOUT, DST_LAYOUT, PRECISION>
{
    fn transform(&self, dst: &mut [u8]) -> Result<(), CmsError> {
        unsafe { self.transform_impl_in_place(dst) }
    }
}
