/*
 * // Copyright (c) Radzivon Bartoshyk 6/2025. All rights reserved.
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
use crate::math::common::f_fmla;

#[derive(Copy, Clone, Default, Debug)]
pub(crate) struct Dekker {
    pub(crate) lo: f64,
    pub(crate) hi: f64,
}

impl Dekker {
    #[inline]
    pub(crate) const fn new(lo: f64, hi: f64) -> Self {
        Dekker { lo, hi }
    }

    // Non FMA helper
    #[allow(dead_code)]
    #[inline]
    pub(crate) const fn split(a: f64) -> Dekker {
        // CN = 2^N.
        const CN: f64 = (1 << 27) as f64;
        const C: f64 = CN + 1.0;
        let t1 = C * a;
        let t2 = a - t1;
        let r_hi = t1 + t2;
        let r_lo = a - r_hi;
        Dekker::new(r_lo, r_hi)
    }

    // Non FMA helper
    #[allow(dead_code)]
    #[inline]
    fn from_exact_mult_impl_non_fma(asz: Dekker, a: f64, b: f64) -> Self {
        let bs = Dekker::split(b);

        let r_hi = a * b;
        let t1 = asz.hi * bs.hi - r_hi;
        let t2 = asz.hi * bs.lo + t1;
        let t3 = asz.lo * bs.hi + t2;
        let r_lo = asz.lo * bs.lo + t3;
        Dekker::new(r_lo, r_hi)
    }

    #[inline]
    pub(crate) const fn from_exact_add(a: f64, b: f64) -> Dekker {
        let r_hi = a + b;
        let t = r_hi - a;
        let r_lo = b - t;
        Dekker::new(r_lo, r_hi)
    }

    #[inline]
    pub(crate) const fn from_full_exact_add(a: f64, b: f64) -> Dekker {
        let r_hi = a + b;
        let t1 = r_hi - a;
        let t2 = r_hi - t1;
        let t3 = b - t1;
        let t4 = a - t2;
        let r_lo = t3 + t4;
        Dekker::new(r_lo, r_hi)
    }

    #[inline]
    pub(crate) fn add(a: Dekker, b: Dekker) -> Dekker {
        let s = a.hi + b.hi;
        let d = s - a.hi;
        let l = ((b.hi - d) + (a.hi + (d - s))) + (a.lo + b.lo);
        Dekker::new(l, s)
    }

    #[inline]
    pub(crate) fn div(a: Dekker, b: Dekker) -> Dekker {
        let q = 1.0 / b.hi;
        let r_hi = a.hi * q;
        #[cfg(any(
            all(
                any(target_arch = "x86", target_arch = "x86_64"),
                target_feature = "fma"
            ),
            all(target_arch = "aarch64", target_feature = "neon")
        ))]
        {
            let e_hi = f_fmla(b.hi, -r_hi, a.hi);
            let e_lo = f_fmla(b.lo, -r_hi, a.lo);
            let r_lo = q * (e_hi + e_lo);
            Dekker::new(r_lo, r_hi)
        }

        #[cfg(not(any(
            all(
                any(target_arch = "x86", target_arch = "x86_64"),
                target_feature = "fma"
            ),
            all(target_arch = "aarch64", target_feature = "neon")
        )))]
        {
            let b_hi_r_hi = Dekker::from_exact_mult(b.hi, -r_hi);
            let b_lo_r_hi = Dekker::from_exact_mult(b.lo, -r_hi);
            let e_hi = (a.hi + b_hi_r_hi.hi) + b_hi_r_hi.lo;
            let e_lo = (a.lo + b_lo_r_hi.hi) + b_lo_r_hi.lo;
            let r_lo = q * (e_hi + e_lo);
            Dekker::new(r_lo, r_hi)
        }
    }

    #[inline]
    pub(crate) fn from_exact_mult(a: f64, b: f64) -> Self {
        #[cfg(any(
            all(
                any(target_arch = "x86", target_arch = "x86_64"),
                target_feature = "fma"
            ),
            all(target_arch = "aarch64", target_feature = "neon")
        ))]
        {
            let r_hi = a * b;
            let r_lo = f_fmla(a, b, -r_hi);
            Dekker::new(r_lo, r_hi)
        }
        #[cfg(not(any(
            all(
                any(target_arch = "x86", target_arch = "x86_64"),
                target_feature = "fma"
            ),
            all(target_arch = "aarch64", target_feature = "neon")
        )))]
        {
            let splat = Dekker::split(a);
            Dekker::from_exact_mult_impl_non_fma(splat, a, b)
        }
    }

    // #[inline]
    // pub(crate) fn add_f64(&self, other: f64) -> DoubleDouble {
    //     let r = DoubleDouble::from_exact_add(self.hi, other);
    //     Dekker::from_exact_add(r.hi, r.lo + self.lo)
    // }

    #[inline]
    pub(crate) fn quick_mult(a: Dekker, b: Dekker) -> Self {
        let mut r = Dekker::from_exact_mult(a.hi, b.hi);
        let t1 = f_fmla(a.hi, b.lo, r.lo);
        let t2 = f_fmla(a.lo, b.hi, t1);
        r.lo = t2;
        r
    }

    #[inline]
    pub(crate) const fn to_f64(self) -> f64 {
        self.lo + self.hi
    }
}
