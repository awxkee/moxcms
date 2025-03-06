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

        let mut temporary: [u16; 8] = [0; 8];

        let dst_cn = Layout::from(DST_LAYOUT);
        let dst_channels = dst_cn.channels();
        let samples = dst.len() / dst_channels;

        let mut x = 0usize;
        let mut src_x = 0usize;
        let mut dst_x = 0usize;
        if src_channels == 3 {
            while x + 2 < samples {
                let chunk = working_set.get_unchecked(src_x..src_x + 6);

                let dst = dst.get_unchecked_mut(dst_x..);

                let mut src_vl0 = _mm_loadu_si64(chunk.as_ptr() as *const u8);
                let mut src_vl1 = _mm_loadu_si64(chunk.get_unchecked(3..).as_ptr() as *const u8);

                src_vl0 = _mm_insert_epi32::<2>(src_vl0, chunk[2].to_bits() as i32);
                src_vl1 = _mm_insert_epi32::<2>(src_vl1, chunk[5].to_bits() as i32);

                let src_f0 = _mm_cvtps_epi32(_mm_castsi128_ps(src_vl0));
                let src_f1 = _mm_cvtps_epi32(_mm_castsi128_ps(src_vl1));

                let packed_u16_0 = _mm_packus_epi32(src_f0, src_f0);
                let packed_u16_1 = _mm_packus_epi32(src_f1, src_f1);

                _mm_storeu_si64(temporary.as_mut_ptr() as *mut _, packed_u16_0);
                _mm_storeu_si64(
                    temporary.get_unchecked_mut(4..).as_mut_ptr() as *mut _,
                    packed_u16_1,
                );

                *dst.get_unchecked_mut(dst_cn.r_i()) = r_gamma[temporary[0] as usize];
                *dst.get_unchecked_mut(dst_cn.g_i()) = g_gamma[temporary[1] as usize];
                *dst.get_unchecked_mut(dst_cn.b_i()) = b_gamma[temporary[2] as usize];
                if dst_channels == 4 {
                    *dst.get_unchecked_mut(dst_cn.a_i()) = max_value;
                }

                *dst.get_unchecked_mut(dst_cn.r_i() + dst_channels) =
                    r_gamma[temporary[4] as usize];
                *dst.get_unchecked_mut(dst_cn.g_i() + dst_channels) =
                    g_gamma[temporary[5] as usize];
                *dst.get_unchecked_mut(dst_cn.b_i() + dst_channels) =
                    b_gamma[temporary[6] as usize];
                if dst_channels == 4 {
                    *dst.get_unchecked_mut(dst_cn.a_i() + dst_channels) = max_value;
                }

                x += 2;
                src_x += src_channels * 2;
                dst_x += dst_channels * 2;
            }

            while x < samples {
                let chunk = working_set.get_unchecked(src_x..src_x + 3);

                let dst = dst.get_unchecked_mut(dst_x..);

                let mut src_vl = _mm_loadu_si64(chunk.as_ptr() as *const u8);
                src_vl = _mm_insert_epi32::<2>(src_vl, chunk[2].to_bits() as i32);

                let src_f = _mm_cvtps_epi32(_mm_castsi128_ps(src_vl));
                let packed_u16 = _mm_packus_epi32(src_f, src_f);
                _mm_storeu_si64(temporary.as_mut_ptr() as *mut _, packed_u16);
                *dst.get_unchecked_mut(dst_cn.r_i()) = r_gamma[temporary[0] as usize];
                *dst.get_unchecked_mut(dst_cn.g_i()) = g_gamma[temporary[1] as usize];
                *dst.get_unchecked_mut(dst_cn.b_i()) = b_gamma[temporary[2] as usize];
                if dst_channels == 4 {
                    dst[dst_cn.a_i()] = max_value;
                }
                x += 1;
                src_x += src_channels;
                dst_x += dst_channels;
            }
        } else if src_channels == 4 {
            while x < samples {
                let chunk = working_set.get_unchecked(src_x..src_x + 4);
                let dst = dst.get_unchecked_mut(dst_x..);

                let src_f = _mm_cvtps_epi32(_mm_loadu_ps(chunk.as_ptr()));
                let packed_u16 = _mm_packus_epi32(src_f, src_f);
                _mm_storeu_si64(temporary.as_mut_ptr() as *mut _, packed_u16);
                *dst.get_unchecked_mut(dst_cn.r_i()) = r_gamma[temporary[0] as usize];
                *dst.get_unchecked_mut(dst_cn.g_i()) = g_gamma[temporary[1] as usize];
                *dst.get_unchecked_mut(dst_cn.b_i()) = b_gamma[temporary[2] as usize];
                if dst_channels == 4 {
                    *dst.get_unchecked_mut(dst_cn.a_i()) = chunk[3].to_bits() as u8;
                }

                x += 1;
                src_x += src_channels;
                dst_x += dst_channels;
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

        let mut temporary: [u16; 8] = [0; 8];

        let dst_cn = Layout::from(DST_LAYOUT);
        let dst_channels = dst_cn.channels();
        let samples = dst.len() / dst_channels;

        let mut x = 0usize;
        let mut src_x = 0usize;
        let mut dst_x = 0usize;
        if src_channels == 3 {
            while x + 2 < samples {
                let chunk = working_set.get_unchecked(src_x..src_x + 6);

                let dst = dst.get_unchecked_mut(dst_x..);

                let mut src_vl0 = _mm_loadu_si64(chunk.as_ptr() as *const u8);
                let mut src_vl1 = _mm_loadu_si64(chunk.get_unchecked(3..).as_ptr() as *const u8);

                src_vl0 = _mm_insert_epi32::<2>(src_vl0, chunk[2].to_bits() as i32);
                src_vl1 = _mm_insert_epi32::<2>(src_vl1, chunk[5].to_bits() as i32);

                let src_f0 = _mm_cvtps_epi32(_mm_castsi128_ps(src_vl0));
                let src_f1 = _mm_cvtps_epi32(_mm_castsi128_ps(src_vl1));

                let packed_u16_0 = _mm_packus_epi32(src_f0, src_vl0);
                let packed_u16_1 = _mm_packus_epi32(src_f0, src_f1);

                _mm_storeu_si64(temporary.as_mut_ptr() as *mut _, packed_u16_0);
                _mm_storeu_si64(
                    temporary.get_unchecked_mut(4..).as_mut_ptr() as *mut _,
                    packed_u16_1,
                );

                *dst.get_unchecked_mut(dst_cn.r_i()) = r_gamma[temporary[0] as usize];
                *dst.get_unchecked_mut(dst_cn.g_i()) = g_gamma[temporary[1] as usize];
                *dst.get_unchecked_mut(dst_cn.b_i()) = b_gamma[temporary[2] as usize];
                if dst_channels == 4 {
                    *dst.get_unchecked_mut(dst_cn.a_i()) = max_value;
                }

                *dst.get_unchecked_mut(dst_cn.r_i() + dst_channels) =
                    r_gamma[temporary[4] as usize];
                *dst.get_unchecked_mut(dst_cn.g_i() + dst_channels) =
                    g_gamma[temporary[5] as usize];
                *dst.get_unchecked_mut(dst_cn.b_i() + dst_channels) =
                    b_gamma[temporary[6] as usize];
                if dst_channels == 4 {
                    *dst.get_unchecked_mut(dst_cn.a_i() + dst_channels) = max_value;
                }

                x += 2;
                src_x += src_channels * 2;
                dst_x += dst_channels * 2;
            }

            while x < samples {
                let chunk = working_set.get_unchecked(src_x..src_x + 3);

                let dst = dst.get_unchecked_mut(dst_x..);

                let mut src_vl = _mm_loadu_si64(chunk.as_ptr() as *const u8);
                src_vl = _mm_insert_epi32::<2>(src_vl, chunk[2].to_bits() as i32);

                let src_f = _mm_cvtps_epi32(_mm_castsi128_ps(src_vl));
                let packed_u16 = _mm_packus_epi32(src_f, src_vl);
                _mm_storeu_si64(temporary.as_mut_ptr() as *mut _, packed_u16);
                *dst.get_unchecked_mut(dst_cn.r_i()) = r_gamma[temporary[0] as usize];
                *dst.get_unchecked_mut(dst_cn.g_i()) = g_gamma[temporary[1] as usize];
                *dst.get_unchecked_mut(dst_cn.b_i()) = b_gamma[temporary[2] as usize];
                if dst_channels == 4 {
                    dst[dst_cn.a_i()] = max_value;
                }
                x += 1;
                src_x += src_channels;
                dst_x += dst_channels;
            }
        } else if src_channels == 4 {
            while x < samples {
                let chunk = working_set.get_unchecked(src_x..src_x + 4);
                let dst = dst.get_unchecked_mut(dst_x..);

                let src_f = _mm_cvtps_epi32(_mm_loadu_ps(chunk.as_ptr()));
                let packed_u16 = _mm_packus_epi32(src_f, src_f);
                _mm_storeu_si64(temporary.as_mut_ptr() as *mut _, packed_u16);
                *dst.get_unchecked_mut(dst_cn.r_i()) = r_gamma[temporary[0] as usize];
                *dst.get_unchecked_mut(dst_cn.g_i()) = g_gamma[temporary[1] as usize];
                *dst.get_unchecked_mut(dst_cn.b_i()) = b_gamma[temporary[2] as usize];
                if dst_channels == 4 {
                    *dst.get_unchecked_mut(dst_cn.a_i()) = chunk[3].to_bits() as u16;
                }

                x += 1;
                src_x += src_channels;
                dst_x += dst_channels;
            }
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

#[target_feature(enable = "sse4.1")]
fn linear_search_rgb_impl<const CAP: usize, const SRC_LAYOUT: u8>(
    src: &[u8],
    working_set: &mut [f32],
    r_linear: &Box<[f32; CAP]>,
    g_linear: &Box<[f32; CAP]>,
    b_linear: &Box<[f32; CAP]>,
) {
    let src_cn = Layout::from(SRC_LAYOUT);
    let src_channels = src_cn.channels();
    unsafe {
        if src_channels == 4 {
            let mut x = 0usize;
            let total_length = src.len();
            assert!(src.len() <= working_set.len());

            while x + 8 < total_length {
                let chunk = src.get_unchecked(x..x + 8);
                let r0 = chunk[src_cn.r_i()];
                let g0 = chunk[src_cn.g_i()];
                let b0 = chunk[src_cn.b_i()];
                let a0 = _mm_set1_ps(f32::from_bits(chunk[src_cn.a_i()] as u32));

                let r1 = chunk[src_cn.r_i() + 4];
                let g1 = chunk[src_cn.g_i() + 4];
                let b1 = chunk[src_cn.b_i() + 4];
                let a1 = _mm_set1_ps(f32::from_bits(chunk[src_cn.a_i() + 4] as u32));

                let r_l0 = _mm_load_ss(r_linear.get_unchecked(r0 as usize..).as_ptr());
                let g_l0 = _mm_load_ss(g_linear.get_unchecked(g0 as usize..).as_ptr());
                let b_l0 = _mm_load_ss(b_linear.get_unchecked(b0 as usize..).as_ptr());

                let r_l1 = _mm_load_ss(r_linear.get_unchecked(r1 as usize..).as_ptr());
                let g_l1 = _mm_load_ss(g_linear.get_unchecked(g1 as usize..).as_ptr());
                let b_l1 = _mm_load_ss(b_linear.get_unchecked(b1 as usize..).as_ptr());

                let r_g0 = _mm_unpacklo_ps(r_l0, g_l0);
                let b_a0 = _mm_unpacklo_ps(b_l0, a0);

                let r_g1 = _mm_unpacklo_ps(r_l1, g_l1);
                let b_a1 = _mm_unpacklo_ps(b_l1, a1);

                let interleaved0 =
                    _mm_unpacklo_epi64(_mm_castps_si128(r_g0), _mm_castps_si128(b_a0));
                let interleaved1 =
                    _mm_unpacklo_epi64(_mm_castps_si128(r_g1), _mm_castps_si128(b_a1));
                _mm_storeu_ps(
                    working_set.get_unchecked_mut(x..).as_mut_ptr(),
                    _mm_castsi128_ps(interleaved0),
                );
                _mm_storeu_ps(
                    working_set.get_unchecked_mut(x + 4..).as_mut_ptr(),
                    _mm_castsi128_ps(interleaved1),
                );
                x += 8;
            }

            while x < total_length {
                let chunk = src.get_unchecked(x..x + 4);
                let r = chunk[src_cn.r_i()];
                let g = chunk[src_cn.g_i()];
                let b = chunk[src_cn.b_i()];
                let a = _mm_set1_ps(f32::from_bits(chunk[src_cn.a_i()] as u32));
                let r_l = _mm_load_ss(r_linear.get_unchecked(r as usize..).as_ptr());
                let g_l = _mm_load_ss(g_linear.get_unchecked(g as usize..).as_ptr());
                let b_l = _mm_load_ss(b_linear.get_unchecked(b as usize..).as_ptr());
                let r_g = _mm_unpacklo_ps(r_l, g_l);
                let b_a = _mm_unpacklo_ps(b_l, a);
                let interleaved = _mm_unpacklo_epi64(_mm_castps_si128(r_g), _mm_castps_si128(b_a));
                _mm_storeu_ps(
                    working_set.get_unchecked_mut(x..).as_mut_ptr(),
                    _mm_castsi128_ps(interleaved),
                );
                x += 4;
            }
        } else {
            let mut x = 0usize;
            let total_length = src.len();
            assert!(src.len() <= working_set.len());

            while x + 9 < total_length {
                let chunk = src.get_unchecked(x..x + 9);
                let r0 = chunk[src_cn.r_i()];
                let g0 = chunk[src_cn.g_i()];
                let b0 = chunk[src_cn.b_i()];

                let r1 = chunk[src_cn.r_i() + 3];
                let g1 = chunk[src_cn.g_i() + 3];
                let b1 = chunk[src_cn.b_i() + 3];

                let r2 = chunk[src_cn.r_i() + 6];
                let g2 = chunk[src_cn.g_i() + 6];
                let b2 = chunk[src_cn.b_i() + 6];

                let r_l0 = _mm_load_ss(r_linear.get_unchecked(r0 as usize..).as_ptr());
                let g_l0 = _mm_load_ss(g_linear.get_unchecked(g0 as usize..).as_ptr());
                let b_l0 = _mm_load_ss(b_linear.get_unchecked(b0 as usize..).as_ptr());

                let r_l1 = _mm_load_ss(r_linear.get_unchecked(r1 as usize..).as_ptr());
                let g_l1 = _mm_load_ss(g_linear.get_unchecked(g1 as usize..).as_ptr());
                let b_l1 = _mm_load_ss(b_linear.get_unchecked(b1 as usize..).as_ptr());

                let r_l2 = _mm_load_ss(r_linear.get_unchecked(r2 as usize..).as_ptr());
                let g_l2 = _mm_load_ss(g_linear.get_unchecked(g2 as usize..).as_ptr());
                let b_l2 = _mm_load_ss(b_linear.get_unchecked(b2 as usize..).as_ptr());

                let r_g0 = _mm_unpacklo_ps(r_l0, g_l0);
                let r_g1 = _mm_unpacklo_ps(r_l1, g_l1);
                let r_g2 = _mm_unpacklo_ps(r_l2, g_l2);

                _mm_storeu_si64(
                    working_set.get_unchecked_mut(x..).as_mut_ptr() as *mut _,
                    _mm_castps_si128(r_g0),
                );
                _mm_storeu_si64(
                    working_set.get_unchecked_mut(x + 3..).as_mut_ptr() as *mut _,
                    _mm_castps_si128(r_g1),
                );
                _mm_storeu_si64(
                    working_set.get_unchecked_mut(x + 6..).as_mut_ptr() as *mut _,
                    _mm_castps_si128(r_g2),
                );

                _mm_storeu_si32(
                    working_set.get_unchecked_mut(x + 2..).as_mut_ptr() as *mut _,
                    _mm_castps_si128(b_l0),
                );
                _mm_storeu_si32(
                    working_set.get_unchecked_mut(x + 5..).as_mut_ptr() as *mut _,
                    _mm_castps_si128(b_l1),
                );
                _mm_storeu_si32(
                    working_set.get_unchecked_mut(x + 8..).as_mut_ptr() as *mut _,
                    _mm_castps_si128(b_l2),
                );
                x += 9;
            }

            while x < total_length {
                let chunk = src.get_unchecked(x..x + 3);
                let r = chunk[src_cn.r_i()];
                let g = chunk[src_cn.g_i()];
                let b = chunk[src_cn.b_i()];
                let r_l = _mm_load_ss(r_linear.get_unchecked(r as usize..).as_ptr());
                let g_l = _mm_load_ss(g_linear.get_unchecked(g as usize..).as_ptr());
                let b_l = _mm_load_ss(b_linear.get_unchecked(b as usize..).as_ptr());
                let r_g = _mm_unpacklo_ps(r_l, g_l);
                _mm_storeu_si64(
                    working_set.get_unchecked_mut(x..).as_mut_ptr() as *mut _,
                    _mm_castps_si128(r_g),
                );
                _mm_storeu_si32(
                    working_set.get_unchecked_mut(x + 2..).as_mut_ptr() as *mut _,
                    _mm_castps_si128(b_l),
                );
                x += 3;
            }
        }
    }
}

pub(crate) fn linear_search_rgb8<const CAP: usize, const SRC_LAYOUT: u8>(
    src: &[u8],
    working_set: &mut [f32],
    r_linear: &Box<[f32; CAP]>,
    g_linear: &Box<[f32; CAP]>,
    b_linear: &Box<[f32; CAP]>,
) {
    assert!(CAP >= 256);
    unsafe {
        linear_search_rgb_impl::<CAP, SRC_LAYOUT>(src, working_set, r_linear, g_linear, b_linear);
    }
}
