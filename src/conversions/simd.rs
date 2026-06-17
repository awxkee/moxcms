/*
 * // Copyright (c) 2026 safe_unaligned_simd Authors. All rights reserved.
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod x86 {
    use core::ptr;

    #[cfg(target_arch = "x86")]
    use core::arch::x86::{self as arch, __m128, __m128i, __m256i};
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{self as arch, __m128, __m128i, __m256i};

    pub(crate) unsafe trait Is128BitsUnaligned {}
    pub(crate) unsafe trait Is256BitsUnaligned {}

    unsafe impl Is128BitsUnaligned for [u16; 8] {}
    unsafe impl Is256BitsUnaligned for [u16; 16] {}

    #[inline]
    #[target_feature(enable = "sse")]
    pub(crate) fn _mm_load_ss(mem_addr: &f32) -> __m128 {
        unsafe { arch::_mm_load_ss(mem_addr) }
    }

    // This can use the std intrinsic directly once MSRV is >1.90.
    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn _mm_broadcast_ss(mem_addr: &f32) -> __m128 {
        #[allow(unused_unsafe)]
        unsafe {
            arch::_mm_broadcast_ss(mem_addr)
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    pub(crate) fn _mm_storeu_si128<T: Is128BitsUnaligned>(mem_addr: &mut T, a: __m128i) {
        unsafe { arch::_mm_storeu_si128(ptr::from_mut(mem_addr).cast(), a) }
    }

    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn _mm256_storeu_si256<T: Is256BitsUnaligned>(mem_addr: &mut T, a: __m256i) {
        unsafe { arch::_mm256_storeu_si256(ptr::from_mut(mem_addr).cast(), a) }
    }
}

#[cfg(target_arch = "aarch64")]
pub(crate) mod aarch64 {
    use core::ptr;

    use core::arch::aarch64::{self as arch, float32x4_t, int16x4_t, int32x4_t, uint32x4_t};

    pub(crate) unsafe trait Is128BitsUnaligned {}

    unsafe impl Is128BitsUnaligned for [u16; 8] {}

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn vld1_s16(mem_addr: &[i16; 4]) -> int16x4_t {
        unsafe { arch::vld1_s16(mem_addr.as_ptr()) }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn vld1_dup_s16(mem_addr: &i16) -> int16x4_t {
        unsafe { arch::vld1_dup_s16(mem_addr) }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn vld1q_f32(mem_addr: &[f32; 4]) -> float32x4_t {
        unsafe { arch::vld1q_f32(mem_addr.as_ptr()) }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn vld1q_dup_f32(mem_addr: &f32) -> float32x4_t {
        unsafe { arch::vld1q_dup_f32(mem_addr) }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn vld1q_s32(mem_addr: &[i32; 4]) -> int32x4_t {
        unsafe { arch::vld1q_s32(mem_addr.as_ptr()) }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn vld1q_dup_s32(mem_addr: &i32) -> int32x4_t {
        unsafe { arch::vld1q_dup_s32(mem_addr) }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn vst1q_u32<T: Is128BitsUnaligned>(mem_addr: &mut T, a: uint32x4_t) {
        unsafe { arch::vst1q_u32(ptr::from_mut(mem_addr).cast(), a) }
    }
}
