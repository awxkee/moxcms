/*
 * // Copyright (c) Radzivon Bartoshyk 4/2026. All rights reserved.
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

//! Minimal reproduction for moxcms#162 / zenpipe#15.
//!
//! The embedded ICC profile is a v4 Apple Wide Color scanner profile
//! (ICC-RGB v4.0, scnr, PCS=XYZ, 30252 bytes) that uses only A2B LUTs
//! — no matrix-shaper fallback. Both moxcms and lcms2 are given the
//! same profile and the same exhaustive 256³ RGB input, then compared.
//!
//! Run: cargo test --release -p app issue_162 -- --nocapture

use std::fs;
use lcms2::{Intent, PixelFormat, Profile, Transform as LcmsTransform};
use moxcms::{ColorProfile, InterpolationMethod, Layout, RenderingIntent, TransformOptions};

/// The ICC profile embedded in wmc_d4e6bfcba7ee8f83.jpg (Apple Wide Color, v4 A2B-only).
///
pub struct Stats {
    pub max: [i32; 3],
    pub p99: [i32; 3],
    pub avg: [f64; 3],
    pub above_2: usize,
    pub total: usize,
    pub max_pixel_rgb_src: [u8; 3],
    pub max_pixel_diffs: [i32; 3],
    pub max_pixel_a: [u8; 3],   // NEW
    pub max_pixel_b: [u8; 3],   // NEW
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "max={}/{}/{} p99={}/{}/{} avg={:.2}/{:.2}/{:.2} above_2={}/{} \
             diffs={}/{}/{} src=({:?}) max pixels a {:?} max pixels b {:?}",
            self.max[0],
            self.max[1],
            self.max[2],
            self.p99[0],
            self.p99[1],
            self.p99[2],
            self.avg[0],
            self.avg[1],
            self.avg[2],
            self.above_2,
            self.total,
            self.max_pixel_diffs[0],
            self.max_pixel_diffs[1],
            self.max_pixel_diffs[2],
            self.max_pixel_rgb_src,   // x
            self.max_pixel_a,
            self.max_pixel_b
        )
    }
}

fn compare(a: &[u8], b: &[u8], ch: usize) -> Stats {
    let n = a.len() / ch;
    let mut max = [0i32; 3];
    let mut sum = [0u64; 3];
    let mut above_2 = 0usize;
    let mut hist = [[0u64; 256]; 3];

    // NEW: track the pixel with the highest single-channel divergence
    let mut global_max_val = 0i32;
    let mut max_pixel_idx = 0usize;
    let mut max_pixel_diffs = [0i32; 3];
    let mut max_pixel_rgb_src = [0u8; 3];   // NEW
    let mut max_pixel_a = [0u8; 3];   // NEW
    let mut max_pixel_b = [0u8; 3];   // NEW

    for (i, (pa, pb)) in a.chunks_exact(ch).zip(b.chunks_exact(ch)).enumerate() {
        let mut px_max = 0i32;
        let mut px_diffs = [0i32; 3];

        for c in 0..3 {
            let d = (pa[c] as i32 - pb[c] as i32).abs();
            max[c] = max[c].max(d);
            sum[c] += d as u64;
            hist[c][d as usize] += 1;
            px_max = px_max.max(d);
            px_diffs[c] = d;
        }

        if px_max > global_max_val {
            global_max_val = px_max;
            max_pixel_rgb_src = [
                (i >> 16) as u8,
                ((i >> 8) & 0xFF) as u8,
                (i & 0xFF) as u8,
            ];
            max_pixel_diffs = px_diffs;
            max_pixel_a = [pa[0], pa[1], pa[2]];   // NEW
            max_pixel_b = [pb[0], pb[1], pb[2]];   // NEW
        }

        if px_max > 2 {
            above_2 += 1;
        }
    }

    let avg = std::array::from_fn(|c| sum[c] as f64 / n as f64);
    let p99 = std::array::from_fn(|c| {
        let thr = (n as f64 * 0.99).ceil() as u64;
        let mut cum = 0u64;
        for (v, &cnt) in hist[c].iter().enumerate() {
            cum += cnt;
            if cum >= thr { return v as i32; }
        }
        255
    });

    Stats {
        max,
        p99,
        avg,
        above_2,
        total: n,
        max_pixel_rgb_src,
        max_pixel_diffs,     // [dR, dG, dB] at that pixel
        max_pixel_a,   // NEW
        max_pixel_b,   // NEW
    }
}

/// Build exhaustive 256³ RGB source (48 MiB).
fn all_rgb() -> Vec<u8> {
    let n: usize = 256 * 256 * 256;
    let mut v = Vec::with_capacity(n * 3);
    for r in 0..=255u8 {
        for g in 0..=255u8 {
            for b in 0..=255u8 {
                v.push(r);
                v.push(g);
                v.push(b);
            }
        }
    }
    v
}

fn moxcms_transform(src: &[u8], fixed: bool, interp: InterpolationMethod) -> Vec<u8> {
    let icc_bytes = fs::read("/Users/radzivon/RustroverProjects/moxcms/assets/apple_wide_gamut.icc").unwrap();
    let prof = ColorProfile::new_from_slice(&icc_bytes).unwrap();
    let srgb = ColorProfile::new_srgb();
    let t = prof
        .create_transform_8bit(
            Layout::Rgb,
            &srgb,
            Layout::Rgba,
            TransformOptions {
                prefer_fixed_point: fixed,
                allow_use_cicp_transfer: false,
                interpolation_method: interp,
                rendering_intent: RenderingIntent::Perceptual,
                ..Default::default()
            },
        )
        .unwrap();
    let mut dst = vec![0u8; (src.len() / 3) * 4];
    t.transform(src, &mut dst).unwrap();
    dst
}

fn lcms2_transform(src: &[u8]) -> Vec<u8> {
    let icc_bytes = fs::read("/Users/radzivon/RustroverProjects/moxcms/assets/apple_wide_gamut.icc").unwrap();
    let sp = Profile::new_icc(&icc_bytes).unwrap();
    let dp = Profile::new_srgb();
    let t = LcmsTransform::new(
        &sp,
        PixelFormat::RGB_8,
        &dp,
        PixelFormat::RGBA_8,
        Intent::Perceptual,
    )
        .unwrap();
    let mut dst = vec![0u8; (src.len() / 3) * 4];
    t.transform_pixels(src, &mut dst);
    dst
}

#[cfg(test)]
mod tests {
    use moxcms::InterpolationMethod;
    use crate::comparison::{all_rgb, compare, lcms2_transform, moxcms_transform};

    #[test]
    fn issue_162_moxcms_default_vs_lcms2() {
        let src = all_rgb();
        let mox = moxcms_transform(&src, true, InterpolationMethod::Linear);
        let lcm = lcms2_transform(&src);
        let s = compare(&mox, &lcm, 4);
        eprintln!("moxcms default (fixed trilinear) vs lcms2: {s}");
        // This demonstrates the divergence — max ~14-18 on some channels.
    }

    #[test]
    fn issue_162_moxcms_float_tetra_vs_lcms2() {
        let src = all_rgb();
        let mox = moxcms_transform(&src, false, InterpolationMethod::Tetrahedral);
        let lcm = lcms2_transform(&src);
        let s = compare(&mox, &lcm, 4);
        eprintln!("moxcms float tetrahedral vs lcms2: {s}");
        // Still shows similar max — the divergence is in LUT interpretation, not precision.
    }

    #[test]
    fn issue_162_moxcms_internal_fixed_vs_float() {
        let src = all_rgb();
        let fixed = moxcms_transform(&src, true, InterpolationMethod::Linear);
        let float = moxcms_transform(&src, false, InterpolationMethod::Linear);
        let flt_src = moxcms_transform(&[55, 230, 55], false, InterpolationMethod::Linear);
        let fxd_src = moxcms_transform(&[55, 230, 55], true, InterpolationMethod::Linear);
        let mox_tetra = moxcms_transform(&[55, 230, 55], false, InterpolationMethod::Tetrahedral);
        let lcms2_px = lcms2_transform(&[55, 230, 55]);
        let s = compare(&fixed, &float, 4);
        eprintln!("moxcms internal (fixed trilinear vs float trilinear): {s}, trilinear_flt_src {:?}, trilinear_fxd_src {:?}, lcms2_px {:?}, mox_tetra_fxd {:?}", flt_src, fxd_src, lcms2_px, mox_tetra);
        // Max ~2 — moxcms is internally consistent.
    }
}