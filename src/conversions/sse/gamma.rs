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
use crate::Layout;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "sse4.1")]
unsafe fn gamma_search_8bit_impl<const SRC_LAYOUT: u8, const DST_LAYOUT: u8>(
    working_set: &[f32],
    dst: &mut [u8],
    r_gamma: &[u8; 65536],
    g_gamma: &[u8; 65536],
    b_gamma: &[u8; 65536],
) {
    unsafe {
        const BIT_DEPTH: usize = 8;
        let max_value = ((1u32 << BIT_DEPTH) - 1) as u8;
        let src_cn = Layout::from(SRC_LAYOUT);
        let src_channels = src_cn.channels();

        let mut temporary: [u16; 4] = [0; 4];

        let dst_cn = Layout::from(DST_LAYOUT);
        let dst_channels = dst_cn.channels();
        for (chunk, dst) in working_set
            .chunks_exact(src_channels)
            .zip(dst.chunks_exact_mut(dst_channels))
        {
            if src_channels == 3 {
                let mut src_vl = _mm_loadu_si64(chunk.as_ptr() as *const u8);
                src_vl = _mm_insert_epi32::<2>(src_vl, chunk[2].to_bits() as i32);
                let src_f = _mm_cvtps_epi32(_mm_castsi128_ps(src_vl));
                let packed_u16 = _mm_packus_epi32(src_f, src_vl);
                _mm_storeu_si64(temporary.as_mut_ptr() as *mut _, packed_u16);
                dst[dst_cn.r_i()] = r_gamma[temporary[0] as usize];
                dst[dst_cn.g_i()] = g_gamma[temporary[1] as usize];
                dst[dst_cn.b_i()] = b_gamma[temporary[2] as usize];
                if dst_channels == 4 {
                    dst[dst_cn.a_i()] = max_value;
                }
            } else if src_channels == 4 {
                let src_f = _mm_cvtps_epi32(_mm_loadu_ps(chunk.as_ptr()));
                let packed_u16 = _mm_packus_epi32(src_f, src_f);
                _mm_storeu_si64(temporary.as_mut_ptr() as *mut _, packed_u16);
                dst[dst_cn.r_i()] = r_gamma[temporary[0] as usize];
                dst[dst_cn.g_i()] = g_gamma[temporary[1] as usize];
                dst[dst_cn.b_i()] = b_gamma[temporary[2] as usize];
                if dst_channels == 4 {
                    dst[dst_cn.a_i()] = chunk[3].to_bits() as u8;
                }
            }
        }
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn gamma_search_16bit_impl<
    const SRC_LAYOUT: u8,
    const DST_LAYOUT: u8,
    const BIT_DEPTH: usize,
>(
    working_set: &[f32],
    dst: &mut [u16],
    r_gamma: &[u16; 65536],
    g_gamma: &[u16; 65536],
    b_gamma: &[u16; 65536],
) {
    unsafe {
        let max_value = ((1u32 << BIT_DEPTH) - 1) as u16;
        let src_cn = Layout::from(SRC_LAYOUT);
        let src_channels = src_cn.channels();

        let mut temporary: [u16; 4] = [0; 4];

        let dst_cn = Layout::from(DST_LAYOUT);
        let dst_channels = dst_cn.channels();
        let samples = dst.len() / dst_channels;

        let mut x = 0usize;
        let mut src_x = 0usize;
        let mut dst_x = 0usize;
        while x < samples {
            let chunk = working_set.get_unchecked(src_x..);
            let dst = dst.get_unchecked_mut(dst_x..);
            if src_channels == 3 {
                let mut src_vl = _mm_loadu_si64(chunk.as_ptr() as *const u8);
                src_vl = _mm_insert_epi32::<2>(src_vl, chunk[2].to_bits() as i32);
                let src_f = _mm_cvtps_epi32(_mm_castsi128_ps(src_vl));
                let packed_u16 = _mm_packus_epi32(src_f, src_vl);
                _mm_storeu_si64(temporary.as_mut_ptr() as *mut _, packed_u16);
                dst[dst_cn.r_i()] = r_gamma[temporary[0] as usize];
                dst[dst_cn.g_i()] = g_gamma[temporary[1] as usize];
                dst[dst_cn.b_i()] = b_gamma[temporary[2] as usize];
                if dst_channels == 4 {
                    dst[dst_cn.a_i()] = max_value;
                }
            } else if src_channels == 4 {
                let src_f = _mm_cvtps_epi32(_mm_loadu_ps(chunk.as_ptr()));
                let packed_u16 = _mm_packus_epi32(src_f, src_f);
                _mm_storeu_si64(temporary.as_mut_ptr() as *mut _, packed_u16);
                dst[dst_cn.r_i()] = r_gamma[temporary[0] as usize];
                dst[dst_cn.g_i()] = g_gamma[temporary[1] as usize];
                dst[dst_cn.b_i()] = b_gamma[temporary[2] as usize];
                if dst_channels == 4 {
                    dst[dst_cn.a_i()] = chunk[3].to_bits() as u16;
                }
            }

            x += 1;
            src_x += src_channels;
            dst_x += dst_channels;
        }
    }
}

pub(crate) fn gamma_search_8bit<const SRC_LAYOUT: u8, const DST_LAYOUT: u8>(
    working_set: &[f32],
    dst: &mut [u8],
    r_gamma: &[u8; 65536],
    g_gamma: &[u8; 65536],
    b_gamma: &[u8; 65536],
) {
    unsafe {
        gamma_search_8bit_impl::<SRC_LAYOUT, DST_LAYOUT>(
            working_set,
            dst,
            r_gamma,
            g_gamma,
            b_gamma,
        )
    }
}

pub(crate) fn gamma_search_16bit<
    const SRC_LAYOUT: u8,
    const DST_LAYOUT: u8,
    const BIT_DEPTH: usize,
>(
    working_set: &[f32],
    dst: &mut [u16],
    r_gamma: &[u16; 65536],
    g_gamma: &[u16; 65536],
    b_gamma: &[u16; 65536],
) {
    unsafe {
        gamma_search_16bit_impl::<SRC_LAYOUT, DST_LAYOUT, BIT_DEPTH>(
            working_set,
            dst,
            r_gamma,
            g_gamma,
            b_gamma,
        )
    }
}
