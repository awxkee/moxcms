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
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
pub(crate) unsafe fn _mm256_deinterleave_rgb_ps(
    a0: __m256,
    a1: __m256,
    a2: __m256,
) -> (__m256, __m256, __m256) {
    unsafe {
        let (v0, v1, v2) = _mm256_deinterleave_rgb_epi32((
            _mm256_castps_si256(a0),
            _mm256_castps_si256(a1),
            _mm256_castps_si256(a2),
        ));
        (
            _mm256_castsi256_ps(v0),
            _mm256_castsi256_ps(v1),
            _mm256_castsi256_ps(v2),
        )
    }
}

#[inline(always)]
pub(crate) unsafe fn _mm256_interleave_rgb_epi32(
    vals: (__m256i, __m256i, __m256i),
) -> (__m256i, __m256i, __m256i) {
    unsafe {
        let b0 = _mm256_shuffle_epi32::<0x6c>(vals.0);
        let g0 = _mm256_shuffle_epi32::<0xb1>(vals.1);
        let r0 = _mm256_shuffle_epi32::<0xc6>(vals.2);

        let p0 = _mm256_blend_epi32::<0x24>(_mm256_blend_epi32::<0x92>(b0, g0), r0);
        let p1 = _mm256_blend_epi32::<0x24>(_mm256_blend_epi32::<0x92>(g0, r0), b0);
        let p2 = _mm256_blend_epi32::<0x24>(_mm256_blend_epi32::<0x92>(r0, b0), g0);

        let v0 = _mm256_permute2x128_si256::<32>(p0, p1);
        let v1 = p2;
        let v2 = _mm256_permute2x128_si256::<49>(p0, p1);
        (v0, v1, v2)
    }
}

#[inline(always)]
pub(crate) unsafe fn _mm256_deinterleave_rgb_epi32(
    vals: (__m256i, __m256i, __m256i),
) -> (__m256i, __m256i, __m256i) {
    unsafe {
        let s02_low = _mm256_permute2x128_si256::<32>(vals.0, vals.2);
        let s02_high = _mm256_permute2x128_si256::<49>(vals.0, vals.2);

        let b0 = _mm256_blend_epi32::<0x92>(_mm256_blend_epi32::<0x24>(s02_low, s02_high), vals.1);
        let g0 = _mm256_blend_epi32::<0x24>(_mm256_blend_epi32::<0x92>(s02_high, s02_low), vals.1);
        let r0 = _mm256_blend_epi32::<0x92>(_mm256_blend_epi32::<0x24>(vals.1, s02_low), s02_high);

        let v0 = _mm256_shuffle_epi32::<0x6c>(b0);
        let v1 = _mm256_shuffle_epi32::<0xb1>(g0);
        let v2 = _mm256_shuffle_epi32::<0xc6>(r0);
        (v0, v1, v2)
    }
}

#[inline(always)]
pub(crate) unsafe fn _mm256_interleave_rgb_ps(
    a0: __m256,
    a1: __m256,
    a2: __m256,
) -> (__m256, __m256, __m256) {
    unsafe {
        let (v0, v1, v2) = _mm256_interleave_rgb_epi32((
            _mm256_castps_si256(a0),
            _mm256_castps_si256(a1),
            _mm256_castps_si256(a2),
        ));
        (
            _mm256_castsi256_ps(v0),
            _mm256_castsi256_ps(v1),
            _mm256_castsi256_ps(v2),
        )
    }
}

#[inline(always)]
pub(crate) unsafe fn _mm256_deinterleave_rgba_ps(
    a0: __m256,
    a1: __m256,
    a2: __m256,
    a3: __m256,
) -> (__m256, __m256, __m256, __m256) {
    unsafe {
        let (v0, v1, v2, v3) = _mm256_deinterleave_rgba_epi32((
            _mm256_castps_si256(a0),
            _mm256_castps_si256(a1),
            _mm256_castps_si256(a2),
            _mm256_castps_si256(a3),
        ));
        (
            _mm256_castsi256_ps(v0),
            _mm256_castsi256_ps(v1),
            _mm256_castsi256_ps(v2),
            _mm256_castsi256_ps(v3),
        )
    }
}

#[inline(always)]
pub(crate) unsafe fn _mm256_deinterleave_rgba_epi32(
    vals: (__m256i, __m256i, __m256i, __m256i),
) -> (__m256i, __m256i, __m256i, __m256i) {
    unsafe {
        let p01l = _mm256_unpacklo_epi32(vals.0, vals.1);
        let p01h = _mm256_unpackhi_epi32(vals.0, vals.1);
        let p23l = _mm256_unpacklo_epi32(vals.2, vals.3);
        let p23h = _mm256_unpackhi_epi32(vals.2, vals.3);

        let pll = _mm256_permute2x128_si256::<32>(p01l, p23l);
        let plh = _mm256_permute2x128_si256::<49>(p01l, p23l);
        let phl = _mm256_permute2x128_si256::<32>(p01h, p23h);
        let phh = _mm256_permute2x128_si256::<49>(p01h, p23h);

        let v0 = _mm256_unpacklo_epi32(pll, plh);
        let v1 = _mm256_unpackhi_epi32(pll, plh);
        let v2 = _mm256_unpacklo_epi32(phl, phh);
        let v3 = _mm256_unpackhi_epi32(phl, phh);
        (v0, v1, v2, v3)
    }
}

#[inline(always)]
pub(crate) unsafe fn _mm256_interleave_rgba_epi32(
    vals: (__m256i, __m256i, __m256i, __m256i),
) -> (__m256i, __m256i, __m256i, __m256i) {
    unsafe {
        let bg0 = _mm256_unpacklo_epi32(vals.0, vals.1);
        let bg1 = _mm256_unpackhi_epi32(vals.0, vals.1);
        let ra0 = _mm256_unpacklo_epi32(vals.2, vals.3);
        let ra1 = _mm256_unpackhi_epi32(vals.2, vals.3);

        let bgra0_ = _mm256_unpacklo_epi64(bg0, ra0);
        let bgra1_ = _mm256_unpackhi_epi64(bg0, ra0);
        let bgra2_ = _mm256_unpacklo_epi64(bg1, ra1);
        let bgra3_ = _mm256_unpackhi_epi64(bg1, ra1);

        let v0 = _mm256_permute2x128_si256::<32>(bgra0_, bgra1_);
        let v1 = _mm256_permute2x128_si256::<32>(bgra2_, bgra3_);
        let v2 = _mm256_permute2x128_si256::<49>(bgra0_, bgra1_);
        let v3 = _mm256_permute2x128_si256::<49>(bgra2_, bgra3_);

        (v0, v1, v2, v3)
    }
}

#[inline(always)]
pub(crate) unsafe fn _mm256_interleave_rgba_ps(
    a0: __m256,
    a1: __m256,
    a2: __m256,
    a3: __m256,
) -> (__m256, __m256, __m256, __m256) {
    unsafe {
        let (v0, v1, v2, v3) = _mm256_interleave_rgba_epi32((
            _mm256_castps_si256(a0),
            _mm256_castps_si256(a1),
            _mm256_castps_si256(a2),
            _mm256_castps_si256(a3),
        ));
        (
            _mm256_castsi256_ps(v0),
            _mm256_castsi256_ps(v1),
            _mm256_castsi256_ps(v2),
            _mm256_castsi256_ps(v3),
        )
    }
}
