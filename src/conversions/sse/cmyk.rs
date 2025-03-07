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
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[derive(Copy, Clone, Default)]
pub(crate) struct Vector3fLerpCmykSse;

#[inline(always)]
pub(crate) fn load_vector_3f(v: Vector3f) -> __m128 {
    let v0 = unsafe { _mm_loadu_si64(v.v.as_ptr() as *const _) };
    let v1 = unsafe {
        _mm_insert_epi32::<2>(
            v0,
            (v.v.get_unchecked(2..).as_ptr() as *const i32).read_unaligned(),
        )
    };
    unsafe { _mm_castsi128_ps(v1) }
}

impl Vector3fCmykLerp for Vector3fLerpCmykSse {
    #[inline(always)]
    fn interpolate(a: Vector3f, b: Vector3f, t: f32, scale: f32) -> Vector3f {
        unsafe {
            let a0 = load_vector_3f(a);
            let b0 = load_vector_3f(b);
            let t0 = _mm_set1_ps(t);
            let ones = _mm_set1_ps(1f32);
            let hp = _mm_mul_ps(a0, _mm_sub_ps(ones, t0));
            let mut v = _mm_add_ps(hp, _mm_mul_ps(b0, t0));
            v = _mm_add_ps(_mm_set1_ps(0.5f32), _mm_mul_ps(v, _mm_set1_ps(scale)));
            let mut vector3 = Vector3f { v: [0f32; 3] };
            _mm_storeu_si64(vector3.v.as_mut_ptr() as *mut _, _mm_castps_si128(v));
            vector3.v[2] = f32::from_bits((_mm_extract_epi32::<2>(_mm_castps_si128(v))) as u32);
            vector3
        }
    }
}
