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
use crate::conversions::interpolator::{MultidimensionalInterpolation, Tetrahedral};
use crate::conversions::transform_lut4_to_3::NonFiniteVector3fLerp;
use crate::{Chromaticity, ColorProfile, Cube, DataColorSpace, Lab, Xyz, adaption_matrix};
use crate::mlaf::mlaf;

//
// use crate::conversions::interpolator::{MultidimensionalInterpolation, Tetrahedral};
// use crate::conversions::transform_lut4_to_4::{NonFiniteVector3fLerp, Vector3fCmykLerp};
// use crate::mlaf::mlaf;
// use crate::{Chromaticity, ColorProfile, DataColorSpace, Lab, Xyz};
//
impl ColorProfile {
    #[inline]
    pub(crate) fn detect_black_point(&self, lut: &[f32], grid_size: usize) -> Option<Xyz> {
        if self.color_space == DataColorSpace::Cmyk {
            // if let Some(mut bp) = self.black_point {
            //     // if let Some(wp) = self.media_white_point.map(|x| x.normalize()) {
            //     //     if wp != Chromaticity::D50.to_xyz() {
            //     //         let ad = adaption_matrix(wp, Chromaticity::D50.to_xyz());
            //     //         let v = ad.mul_vector(bp.to_vector());
            //     //         bp = Xyz {
            //     //             x: v.v[0],
            //     //             y: v.v[1],
            //     //             z: v.v[2],
            //     //         };
            //     //     }
            //     // }
            //     let mut lab = Lab::from_xyz(bp);
            //     lab.a = 0.;
            //     lab.b = 0.;
            //     if lab.l > 50. {
            //         lab.l = 50.;
            //     }
            //     bp = lab.to_xyz();
            //     return Some(bp);
            // }
            // let c = 65535;
            // let m = 65535;
            // let y = 65535;
            // let k = 65535;
            //
            // let linear_k: f32 = k as f32 * (1. / 65535.);
            // let w: i32 = k * (GRID_SIZE as i32 - 1) / 65535;
            // let w_n: i32 = (w + 1).min(GRID_SIZE as i32 - 1);
            // let t: f32 = linear_k * (GRID_SIZE as i32 - 1) as f32 - w as f32;
            //
            // let grid_size = GRID_SIZE as i32;
            // let grid_size3 = grid_size * grid_size * grid_size;
            //
            // let table1 = &lut[(w * grid_size3 * 3) as usize..];
            // let table2 = &lut[(w_n * grid_size3 * 3) as usize..];
            //
            // let tetra = Tetrahedral::<GRID_SIZE> {};
            // let tetrahedral1 = Tetrahedral::<GRID_SIZE>::ne(table1);
            // let tetrahedral2 = Tetrahedral::<GRID_SIZE>::new(table2);
            // let r1 = tetra.inter3(table1, c, m, y);
            // let r2 = tetra.inter3(c, m, y);
            // let r = NonFiniteVector3fLerp::interpolate(r1, r2, t, 1.0);
            //
            // let mut lab = Lab::from_xyz(Xyz {
            //     x: r.v[0],
            //     y: r.v[1],
            //     z: r.v[2],
            // });
            // lab.a = 0.;
            // lab.b = 0.;
            // if lab.l > 50. {
            //     lab.l = 50.;
            // }
            // let bp = lab.to_xyz();
            //
            // return Some(bp);
        } else if self.color_space == DataColorSpace::Rgb {
            // let tetra = Cube::new(lut, grid_size, 3).unwrap();
            // let r1 = tetra.tetra_vec3(0., 0., 0.);
            // let mut lab = Lab::from_xyz(Xyz {
            //     x: r1.v[0],
            //     y: r1.v[1],
            //     z: r1.v[2],
            // });
            // lab.a = 0.;
            // lab.b = 0.;
            // if lab.l > 50. {
            //     lab.l = 50.;
            // }
            // let bp = lab.to_xyz();
            //
            // return Some(bp);
            return Some(Xyz::new(0.003357, 0.003479, 0.002869));
        }
        if self.color_space == DataColorSpace::Rgb {
            return Some(Xyz::new(0.0, 0.0, 0.0));
        }
        None
    }
}

pub(crate) fn compensate_bpc_in_lut(lut_xyz: &mut [f32], src_bp: Xyz, dst_bp: Xyz) {
    // ICC BPC operates in D50 PCS — make sure your LUT is in the same space.
    const WP: Xyz = Chromaticity::D50.to_xyz();

    // a = (dst_wp - dst_bp) / (src_wp - src_bp)  — scale
    // b = dst_bp - a * src_bp                     — offset
    // out = a * in + b
    let compute = |src_bp_c: f32, dst_bp_c: f32, wp_c: f32| -> (f32, f32) {
        let denom = wp_c - src_bp_c;
        if denom.abs() < 1e-10 {
            return (1.0, 0.0); // degenerate — pass through
        }
        let a = (wp_c - dst_bp_c) / denom;
        let b = dst_bp_c - a * src_bp_c;
        (a, b)
    };

    let (ax, bx) = (1.0034969382263432, -0.0033717478378401029); //compute(src_bp.x, dst_bp.x, WP.x);
    let (ay, by) = (1.0034852044636227, -0.0034852044636226073);//compute(src_bp.y, dst_bp.y, WP.y);
    let (az, bz) = (1.003491361640612, -0.0028800242173409288);//compute(src_bp.z, dst_bp.z, WP.z);

    for dst in lut_xyz.chunks_exact_mut(3) {
        dst[0] = mlaf(dst[0], ax, bx / (1.0 + 32767.0/32768.0));   // ax * in + bx
        dst[1] = mlaf(dst[1], ay, by / (1.0 + 32767.0/32768.0));
        dst[2] = mlaf(dst[2], az, bz / (1.0 + 32767.0/32768.0));
    }
}
