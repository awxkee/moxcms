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
use crate::matan::{
    does_curve_have_discontinuity, is_curve_ascending, is_curve_degenerated, is_curve_descending,
    is_curve_linear8, is_curve_linear16, is_curve_monotonic,
};
use crate::reader::{uint8_number_to_float_fast, uint16_number_to_float_fast};
use crate::{LutStore, ToneReprCurve};

impl LutStore {
    pub fn to_clut_f32(&self) -> Vec<f32> {
        match self {
            LutStore::Store8(store) => store
                .iter()
                .map(|x| uint8_number_to_float_fast(*x))
                .collect(),
            LutStore::Store16(store) => store
                .iter()
                .map(|x| uint16_number_to_float_fast(*x as u32))
                .collect(),
        }
    }

    pub(crate) fn is_degenerated(&self, entries: usize, channel: usize) -> bool {
        let start = entries * channel;
        let end = start + entries;

        match &self {
            LutStore::Store8(v) => is_curve_degenerated(&v[start..end]),
            LutStore::Store16(v) => is_curve_degenerated(&v[start..end]),
        }
    }

    pub(crate) fn is_monotonic(&self, entries: usize, channel: usize) -> bool {
        let start = entries * channel;
        let end = start + entries;

        match &self {
            LutStore::Store8(v) => is_curve_monotonic(&v[start..end]),
            LutStore::Store16(v) => is_curve_monotonic(&v[start..end]),
        }
    }

    pub(crate) fn have_discontinuities(&self, entries: usize, channel: usize) -> bool {
        let start = entries * channel;
        let end = start + entries;

        match &self {
            LutStore::Store8(v) => does_curve_have_discontinuity(&v[start..end]),
            LutStore::Store16(v) => does_curve_have_discontinuity(&v[start..end]),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn is_linear(&self, entries: usize, channel: usize) -> bool {
        let start = entries * channel;
        let end = start + entries;

        match &self {
            LutStore::Store8(v) => is_curve_linear8(&v[start..end]),
            LutStore::Store16(v) => is_curve_linear16(&v[start..end]),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn is_descending(&self, entries: usize, channel: usize) -> bool {
        let start = entries * channel;
        let end = start + entries;

        match &self {
            LutStore::Store8(v) => is_curve_descending(&v[start..end]),
            LutStore::Store16(v) => is_curve_descending(&v[start..end]),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn is_ascending(&self, entries: usize, channel: usize) -> bool {
        let start = entries * channel;
        let end = start + entries;

        match &self {
            LutStore::Store8(v) => is_curve_ascending(&v[start..end]),
            LutStore::Store16(v) => is_curve_ascending(&v[start..end]),
        }
    }
}

impl ToneReprCurve {
    pub(crate) fn is_linear(&self) -> bool {
        match &self {
            ToneReprCurve::Lut(lut) => {
                if lut.is_empty() {
                    return true;
                }
                if lut.len() == 1 {
                    let gamma = 1. / crate::trc::u8_fixed_8number_to_float(lut[0]);
                    if (gamma - 1.).abs() < 1e-4 {
                        return true;
                    }
                }
                is_curve_linear16(lut)
            }
            ToneReprCurve::Parametric(parametric) => {
                if parametric.is_empty() {
                    return true;
                }
                if parametric.len() == 1 && parametric[0] == 1. {
                    return true;
                }
                false
            }
        }
    }

    pub(crate) fn is_monotonic(&self) -> bool {
        match &self {
            ToneReprCurve::Lut(lut) => is_curve_monotonic(lut),
            ToneReprCurve::Parametric(_) => true,
        }
    }

    pub(crate) fn is_degenerated(&self) -> bool {
        match &self {
            ToneReprCurve::Lut(lut) => is_curve_degenerated(lut),
            ToneReprCurve::Parametric(_) => false,
        }
    }

    pub(crate) fn have_discontinuities(&self) -> bool {
        match &self {
            ToneReprCurve::Lut(lut) => does_curve_have_discontinuity(lut),
            ToneReprCurve::Parametric(_) => false,
        }
    }
}
