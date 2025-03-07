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
use crate::Vector3f;
use crate::conversions::cmyk::Vector3fCmykLerp;
use std::arch::aarch64::*;

#[derive(Copy, Clone, Default)]
pub(crate) struct Vector3fLerpCmykNeon;

impl Vector3fCmykLerp for Vector3fLerpCmykNeon {
    #[inline(always)]
    fn interpolate(a: Vector3f, b: Vector3f, t: f32, scale: f32) -> Vector3f {
        unsafe {
            let a0 = vcombine_f32(
                vld1_f32(a.v.as_ptr()),
                vld1_lane_f32::<0>(a.v.as_ptr().add(2), vdup_n_f32(0f32)),
            );
            let b0 = vcombine_f32(
                vld1_f32(b.v.as_ptr()),
                vld1_lane_f32::<0>(b.v.as_ptr().add(2), vdup_n_f32(0f32)),
            );
            let t0 = vdupq_n_f32(t);
            let ones = vdupq_n_f32(1f32);
            let hp = vmulq_f32(a0, vsubq_f32(ones, t0));
            let mut v = vfmaq_f32(hp, b0, t0);
            v = vfmaq_f32(vdupq_n_f32(0.5f32), v, vdupq_n_f32(scale));
            let mut vector3 = Vector3f { v: [0f32; 3] };
            vst1_f32(vector3.v.as_mut_ptr(), vget_low_f32(v));
            vst1q_lane_f32::<2>(vector3.v.as_mut_ptr().add(2), v);
            vector3
        }
    }
}
