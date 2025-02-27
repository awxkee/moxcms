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
use crate::{cbrtf, Rgb};
use crate::oklab::Oklab;

#[inline]
#[allow(clippy::excessive_precision)]
pub(crate) fn compute_max_saturation(a: f32, b: f32) -> f32 {
    // Max saturation will be when one of r, g or b goes below zero.

    // Select different coefficients depending on which component goes below zero first
    let (k0, k1, k2, k3, k4, wl, wm, ws) = if -1.88170328 * a - 0.80936493 * b > 1.0 {
        // Red component
        (
            1.19086277,
            1.76576728,
            0.59662641,
            0.75515197,
            0.56771245,
            4.0767416621,
            -3.3077115913,
            0.2309699292,
        )
    } else if 1.81444104 * a - 1.19445276 * b > 1.0 {
        // Green component
        (
            0.73956515,
            -0.45954404,
            0.08285427,
            0.12541070,
            0.14503204,
            -1.2684380046,
            2.6097574011,
            -0.3413193965,
        )
    } else {
        // Blue component
        (
            1.35733652,
            -0.00915799,
            -1.15130210,
            -0.50559606,
            0.00692167,
            -0.0041960863,
            -0.7034186147,
            1.7076147010,
        )
    };

    // Approximate max saturation using a polynomial:
    let mut ss = k0 + k1 * a + k2 * b + k3 * a * a + k4 * a * b;

    // Do one step Halley's method to get closer
    // this gives an error less than 10e6, except for some blue hues where the dS/dh is close to infinite
    // this should be sufficient for most applications, otherwise do two/three steps

    let k_l = 0.3963377774 * a + 0.2158037573 * b;
    let k_m = -0.1055613458 * a - 0.0638541728 * b;
    let k_s = -0.0894841775 * a - 1.2914855480 * b;

    {
        let l_ = 1.0 + ss * k_l;
        let m_ = 1.0 + ss * k_m;
        let s_ = 1.0 + ss * k_s;

        let l = l_ * l_ * l_;
        let m = m_ * m_ * m_;
        let s = s_ * s_ * s_;

        let l_d_s = 3.0 * k_l * l_ * l_;
        let m_d_s = 3.0 * k_m * m_ * m_;
        let s_d_s = 3.0 * k_s * s_ * s_;

        let l_d_s2 = 6.0 * k_l * k_l * l_;
        let m_d_s2 = 6.0 * k_m * k_m * m_;
        let s_d_s2 = 6.0 * k_s * k_s * s_;

        let f = wl * l + wm * m + ws * s;
        let f1 = wl * l_d_s + wm * m_d_s + ws * s_d_s;
        let f2 = wl * l_d_s2 + wm * m_d_s2 + ws * s_d_s2;

        ss -= f * f1 / (f1 * f1 - 0.5 * f * f2);
    }

    ss
}

#[inline]
pub(crate) fn find_cusp(a: f32, b: f32) -> (f32, f32) {
    let s_cusp = compute_max_saturation(a, b);

    let oklaba = Oklab::new(1., s_cusp * a, s_cusp * b);

    let rgb_at_max = oklaba.to_linear_rgb();

    let l_cusp = cbrtf(1. / rgb_at_max.r.max(rgb_at_max.g).max(rgb_at_max.b));
    let c_cusp = l_cusp * s_cusp;

    (l_cusp, c_cusp)
}

#[inline]
#[allow(clippy::excessive_precision)]
fn find_gamut_intersection(a: f32, b: f32, ll1: f32, cc1: f32, ll0: f32) -> f32 {
    // Find the cusp of the gamut triangle
    let (ll, cc) = find_cusp(a, b);

    // Find the intersection for upper and lower half separately
    let mut t: f32;
    if ((ll1 - ll0) * cc - (ll - ll0) * cc1) <= 0.0 {
        // Lower half
        t = cc * ll0 / (cc1 * ll + cc * (ll0 - ll1));
    } else {
        // Upper half

        // First intersect with triangle
        t = cc * (ll0 - 1.0) / (cc1 * (ll - 1.0) + cc * (ll0 - ll1));

        // Then one step Halley's method
        {
            let dll = ll1 - ll0;
            let dcc = cc1;

            let k_l = 0.3963377774 * a + 0.2158037573 * b;
            let k_m = -0.1055613458 * a - 0.0638541728 * b;
            let k_s = -0.0894841775 * a - 1.2914855480 * b;

            let l_dt = dll + dcc * k_l;
            let m_dt = dll + dcc * k_m;
            let s_dt = dll + dcc * k_s;

            // If higher accuracy is required, 2 or 3 iterations of the following block can be used:
            {
                let ll = ll0 * (1.0 - t) + t * ll1;
                let cc = t * cc1;

                let l_ = ll + cc * k_l;
                let m_ = ll + cc * k_m;
                let s_ = ll + cc * k_s;

                let l = l_ * l_ * l_;
                let m = m_ * m_ * m_;
                let s = s_ * s_ * s_;

                let l_dt = 3.0 * l_dt * l_ * l_;
                let m_dt = 3.0 * m_dt * m_ * m_;
                let s_dt = 3.0 * s_dt * s_ * s_;

                let l_dt2 = 6.0 * l_dt * l_dt * l_;
                let m_dt2 = 6.0 * m_dt * m_dt * m_;
                let s_dt2 = 6.0 * s_dt * s_dt * s_;

                let r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s - 1.0;
                let r1 = 4.0767416621 * l_dt - 3.3077115913 * m_dt + 0.2309699292 * s_dt;
                let r2 = 4.0767416621 * l_dt2 - 3.3077115913 * m_dt2 + 0.2309699292 * s_dt2;

                let u_r = r1 / (r1 * r1 - 0.5 * r * r2);
                let mut t_r = -r * u_r;

                let g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s - 1.0;
                let g1 = -1.2684380046 * l_dt + 2.6097574011 * m_dt - 0.3413193965 * s_dt;
                let g2 = -1.2684380046 * l_dt2 + 2.6097574011 * m_dt2 - 0.3413193965 * s_dt2;

                let u_g = g1 / (g1 * g1 - 0.5 * g * g2);
                let mut t_g = -g * u_g;

                let b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s - 1.0;
                let b1 = -0.0041960863 * l_dt - 0.7034186147 * m_dt + 1.7076147010 * s_dt;
                let b2 = -0.0041960863 * l_dt2 - 0.7034186147 * m_dt2 + 1.7076147010 * s_dt2;

                let u_b = b1 / (b1 * b1 - 0.5 * b * b2);
                let mut t_b = -b * u_b;

                t_r = if u_r >= 0.0 { t_r } else { f32::MAX };
                t_g = if u_g >= 0.0 { t_g } else { f32::MAX };
                t_b = if u_b >= 0.0 { t_b } else { f32::MAX };

                t += t_r.min(t_g.min(t_b));
            }
        }
    }

    t
}

// #[inline]
// pub(crate) fn gamut_clip_preserve_chroma(rgb: Rgb<f32>) -> Rgb<f32> {
//     if rgb.r <= 1. && rgb.g <= 1. && rgb.b <= 1. && rgb.r >= 0. && rgb.g >= 0. && rgb.b >= 0. {
//         return rgb;
//     }
//
//     let laba = Oklab::from_linear_rgb(rgb);
//
//     let ll = laba.l;
//     let eps: f32 = 0.00001;
//     let cc = eps.max((laba.a * laba.a + laba.b * laba.b).sqrt());
//     let a_ = laba.a / cc;
//     let b_ = laba.b / cc;
//
//     let ll0 = ll.clamp(0., 1.);
//
//     let t = find_gamut_intersection(a_, b_, ll, cc, ll0);
//     let ll_clipped = ll0 * (1. - t) + t * ll;
//     let cc_clipped = t * cc;
//
//     let mut result = Oklab::new(ll_clipped, cc_clipped * a_, cc_clipped * b_).to_linear_rgb();
//
//     result.r = result.r.clamp(0., 1.);
//     result.g = result.g.clamp(0., 1.);
//     result.b = result.b.clamp(0., 1.);
//
//     // Don't bother if the result is very close
//     if (rgb.r - result.r).abs() < 0.003
//         && (rgb.g - result.g).abs() < 0.003
//         && (rgb.b - result.b).abs() < 0.003
//     {
//         return rgb;
//     }
//
//     result
// }
//
// pub(crate) fn gamut_clip_project_to_l_cusp(rgb: Rgb<f32>) -> Rgb<f32> {
//     if rgb.r < 1f32 && rgb.g < 1f32 && rgb.b < 1f32 && rgb.r > 0f32 && rgb.g > 0f32 && rgb.b > 0f32
//     {
//         return rgb;
//     }
//
//     let lab = Oklab::from_linear_rgb(rgb);
//
//     let l = lab.l;
//     let eps = 0.00001f32;
//     let chroma = f32::max(eps, (lab.a * lab.a + lab.b * lab.b).sqrt());
//     let a_ = lab.a / chroma;
//     let b_ = lab.b / chroma;
//
//     // The cusp is computed here and in find_gamut_intersection, an optimized solution would only compute it once.
//     let cusp = find_cusp(a_, b_);
//
//     let l0 = cusp.0;
//
//     let t = find_gamut_intersection(a_, b_, l, chroma, l0);
//
//     let l_clipped = l0 * (1f32 - t) + t * l;
//     let c_clipped = t * chroma;
//
//     Oklab::new(l_clipped, c_clipped * a_, c_clipped * b_).to_linear_rgb()
// }

#[inline]
fn sgn(x: f32) -> f32 {
    (0.0 < x) as i32 as f32 - (x < 0.0) as i32 as f32
}

#[inline]
pub(crate) fn gamut_clip_adaptive_l0_l_cusp(rgb: Rgb<f32>, alpha: f32) -> Rgb<f32> {
    if rgb.r < 1f32 && rgb.g < 1f32 && rgb.b < 1f32 && rgb.r > 0f32 && rgb.g > 0f32 && rgb.b > 0f32
    {
        return rgb;
    }

    let lab = Oklab::from_linear_rgb(rgb);

    let lum = lab.l;
    let eps = 0.00001f32;
    let ch = f32::max(eps, (lab.a * lab.a + lab.b * lab.b).sqrt());
    let a_ = lab.a / ch;
    let b_ = lab.b / ch;

    // The cusp is computed here and in find_gamut_intersection, an optimized solution would only compute it once.
    let cusp = find_cusp(a_, b_);

    let l_d = lum - cusp.0;
    let k = 2f32 * (if l_d > 0f32 { 1f32 - cusp.0 } else { cusp.0 });

    let e1 = 0.5f32 * k + l_d.abs() + alpha * ch / k;
    let l0 = cusp.0 + 0.5f32 * (sgn(l_d) * (e1 - (e1 * e1 - 2f32 * k * l_d.abs()).sqrt()));

    let t = find_gamut_intersection(a_, b_, lum, ch, l0);
    let l_clipped = l0 * (1f32 - t) + t * lum;
    let c_clipped = t * ch;

    Oklab::new(l_clipped, c_clipped * a_, c_clipped * b_).to_linear_rgb()
}
