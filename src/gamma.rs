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
use crate::{TransferCharacteristics, pow, powf};
use num_traits::AsPrimitive;

#[inline]
/// Linear transfer function for sRGB
fn srgb_to_linear(gamma: f64) -> f64 {
    if gamma < 0f64 {
        0f64
    } else if gamma < 12.92f64 * 0.0030412825601275209f64 {
        gamma * (1f64 / 12.92f64)
    } else if gamma < 1.0f64 {
        ((gamma + 0.0550107189475866f64) / 1.0550107189475866f64).powf(2.4f64)
    } else {
        1.0f64
    }
}

#[inline]
/// Gamma transfer function for sRGB
fn srgb_from_linear(linear: f64) -> f64 {
    if linear < 0.0f64 {
        0.0f64
    } else if linear < 0.0030412825601275209f64 {
        linear * 12.92f64
    } else if linear < 1.0f64 {
        1.0550107189475866f64 * linear.powf(1.0f64 / 2.4f64) - 0.0550107189475866f64
    } else {
        1.0f64
    }
}

#[inline]
/// Linear transfer function for Rec.709
const fn rec709_to_linear(gamma: f64) -> f64 {
    if gamma < 0.0f64 {
        0.0f64
    } else if gamma < 4.5f64 * 0.018053968510807f64 {
        gamma * (1f64 / 4.5f64)
    } else if gamma < 1.0f64 {
        pow(
            (gamma + 0.09929682680944f64) / 1.09929682680944f64,
            1.0f64 / 0.45f64,
        )
    } else {
        1.0f64
    }
}

#[inline]
/// Gamma transfer function for Rec.709
const fn rec709_from_linear(linear: f64) -> f64 {
    if linear < 0.0f64 {
        0.0f64
    } else if linear < 0.018053968510807f64 {
        linear * 4.5f64
    } else if linear < 1.0f64 {
        1.09929682680944f64 * pow(linear, 0.45f64) - 0.09929682680944f64
    } else {
        1.0f64
    }
}

#[inline]
/// Linear transfer function for Smpte 428
pub(crate) fn smpte428_to_linear(gamma: f64) -> f64 {
    const SCALE: f64 = 1. / 0.91655527974030934f64;
    gamma.max(0.).min(1f64).powf(2.6f64) * SCALE
}

#[inline]
/// Gamma transfer function for Smpte 428
fn smpte428_from_linear(linear: f64) -> f64 {
    const POWER_VALUE: f64 = 1.0f64 / 2.6f64;
    (0.91655527974030934f64 * linear.max(0.)).powf(POWER_VALUE)
}

#[inline]
/// Linear transfer function for Smpte 240
pub(crate) const fn smpte240_to_linear(gamma: f64) -> f64 {
    if gamma < 0.0 {
        0.0
    } else if gamma < 4.0 * 0.022821585529445 {
        gamma / 4.0
    } else if gamma < 1.0 {
        pow((gamma + 0.111572195921731) / 1.111572195921731, 1.0 / 0.45)
    } else {
        1.0
    }
}

#[inline]
/// Gamma transfer function for Smpte 240
const fn smpte240_from_linear(linear: f64) -> f64 {
    if linear < 0.0 {
        0.0
    } else if linear < 0.022821585529445 {
        linear * 4.0
    } else if linear < 1.0 {
        1.111572195921731 * pow(linear, 0.45) - 0.111572195921731
    } else {
        1.0
    }
}

#[inline]
/// Gamma transfer function for Log100
fn log100_from_linear(linear: f64) -> f64 {
    if linear <= 0.01f64 {
        0.
    } else {
        1. + linear.min(1.).log10() / 2.0
    }
}

#[inline]
/// Linear transfer function for Log100
pub(crate) const fn log100_to_linear(gamma: f64) -> f64 {
    // The function is non-bijective so choose the middle of [0, 0.00316227766f].
    const MID_INTERVAL: f64 = 0.01 / 2.;
    if gamma <= 0. {
        MID_INTERVAL
    } else {
        pow(10f64, 2. * (gamma.min(1.) - 1.))
    }
}

#[inline]
/// Linear transfer function for Log100Sqrt10
pub(crate) fn log100_sqrt10_to_linear(gamma: f64) -> f64 {
    // The function is non-bijective so choose the middle of [0, 0.00316227766f].
    const MID_INTERVAL: f64 = 0.00316227766 / 2.;
    if gamma <= 0. {
        MID_INTERVAL
    } else {
        pow(10f64, 2.5 * (gamma.min(1.) - 1.))
    }
}

#[inline]
/// Gamma transfer function for Log100Sqrt10
fn log100_sqrt10_from_linear(linear: f64) -> f64 {
    if linear <= 0.00316227766 {
        0.0
    } else {
        1.0 + linear.min(1.).log10() / 2.5
    }
}

#[inline]
/// Gamma transfer function for Bt.1361
const fn bt1361_from_linear(linear: f64) -> f64 {
    if linear < -0.25 {
        -0.25
    } else if linear < 0.0 {
        -0.27482420670236 * pow(-4.0 * linear, 0.45) + 0.02482420670236
    } else if linear < 0.018053968510807 {
        linear * 4.5
    } else if linear < 1.0 {
        1.09929682680944 * pow(linear, 0.45) - 0.09929682680944
    } else {
        1.0
    }
}

#[inline]
/// Linear transfer function for Bt.1361
pub(crate) const fn bt1361_to_linear(gamma: f64) -> f64 {
    if gamma < -0.25f64 {
        -0.25f64
    } else if gamma < 0.0f64 {
        pow(
            (gamma - 0.02482420670236f64) / -0.27482420670236f64,
            1.0f64 / 0.45f64,
        ) / -4.0f64
    } else if gamma < 4.5 * 0.018053968510807 {
        gamma / 4.5
    } else if gamma < 1.0 {
        pow((gamma + 0.09929682680944) / 1.09929682680944, 1.0 / 0.45)
    } else {
        1.0f64
    }
}

#[inline(always)]
/// Pure gamma transfer function for gamma 2.2
const fn pure_gamma_function(x: f64, gamma: f64) -> f64 {
    if x <= 0f64 {
        0f64
    } else if x >= 1f64 {
        return 1f64;
    } else {
        return pow(x, gamma);
    }
}

#[inline]
pub(crate) const fn iec61966_to_linear(gamma: f64) -> f64 {
    if gamma < -4.5f64 * 0.018053968510807f64 {
        pow(
            (-gamma + 0.09929682680944f64) / -1.09929682680944f64,
            1.0f64 / 0.45f64,
        )
    } else if gamma < 4.5f64 * 0.018053968510807f64 {
        gamma / 4.5f64
    } else {
        pow(
            (gamma + 0.09929682680944f64) / 1.09929682680944f64,
            1.0f64 / 0.45f64,
        )
    }
}

#[inline]
const fn iec61966_from_linear(v: f64) -> f64 {
    if v < -0.018053968510807f64 {
        -1.09929682680944f64 * pow(-v, 0.45f64) + 0.09929682680944f64
    } else if v < 0.018053968510807f64 {
        v * 4.5f64
    } else {
        1.09929682680944f64 * pow(v, 0.45f64) - 0.09929682680944f64
    }
}

#[inline]
/// Pure gamma transfer function for gamma 2.2
fn gamma2p2_from_linear(linear: f64) -> f64 {
    pure_gamma_function(linear, 1f64 / 2.2f64)
}

#[inline]
/// Linear transfer function for gamma 2.2
fn gamma2p2_to_linear(gamma: f64) -> f64 {
    pure_gamma_function(gamma, 2.2f64)
}

#[inline]
/// Pure gamma transfer function for gamma 2.8
const fn gamma2p8_from_linear(linear: f64) -> f64 {
    pure_gamma_function(linear, 1f64 / 2.8f64)
}

#[inline]
/// Linear transfer function for gamma 2.8
const fn gamma2p8_to_linear(gamma: f64) -> f64 {
    pure_gamma_function(gamma, 2.8f64)
}

#[inline]
/// Linear transfer function for PQ
pub(crate) fn pq_to_linear(gamma: f64) -> f64 {
    if gamma > 0.0 {
        let pow_gamma = pow(gamma, 1.0 / 78.84375);
        let num = (pow_gamma - 0.8359375).max(0.);
        let den = (18.8515625 - 18.6875 * pow_gamma).max(f64::MIN);
        pow(num / den, 1.0 / 0.1593017578125)
    } else {
        0.0
    }
}

#[inline]
/// Linear transfer function for PQ
pub(crate) fn pq_to_linearf(gamma: f32) -> f32 {
    if gamma > 0.0 {
        let pow_gamma = powf(gamma, 1.0 / 78.84375);
        let num = (pow_gamma - 0.8359375).max(0.);
        let den = (18.8515625 - 18.6875 * pow_gamma).max(f32::MIN);
        powf(num / den, 1.0 / 0.1593017578125)
    } else {
        0.0
    }
}

#[inline]
/// Gamma transfer function for PQ
fn pq_from_linear(linear: f64) -> f64 {
    if linear > 0.0 {
        let linear = linear.clamp(0., 1.);
        let pow_linear = pow(linear, 0.1593017578125);
        let num = 0.1640625 * pow_linear - 0.1640625;
        let den = 1.0 + 18.6875 * pow_linear;
        pow(1.0 + num / den, 78.84375)
    } else {
        0.0
    }
}

#[inline]
/// Gamma transfer function for PQ
pub(crate) const fn pq_from_linearf(linear: f32) -> f32 {
    if linear > 0.0 {
        let linear = linear.clamp(0., 1.);
        let pow_linear = powf(linear, 0.1593017578125);
        let num = 0.1640625 * pow_linear - 0.1640625;
        let den = 1.0 + 18.6875 * pow_linear;
        powf(1.0 + num / den, 78.84375)
    } else {
        0.0
    }
}

#[inline]
/// Linear transfer function for HLG
pub(crate) fn hlg_to_linear(gamma: f64) -> f64 {
    if gamma < 0.0 {
        return 0.0;
    }
    if gamma <= 0.5 {
        f64::powf((gamma * gamma) * (1.0 / 3.0), 1.2)
    } else {
        f64::powf(
            (f64::exp((gamma - 0.55991073) / 0.17883277) + 0.28466892) / 12.0,
            1.2,
        )
    }
}

#[inline]
/// Gamma transfer function for HLG
fn hlg_from_linear(linear: f64) -> f64 {
    // Scale from extended SDR range to [0.0, 1.0].
    let mut linear = (linear).clamp(0., 1.);
    // Inverse OOTF followed by OETF see Table 5 and Note 5i in ITU-R BT.2100-2 page 7-8.
    linear = pow(linear, 1.0 / 1.2);
    if linear < 0.0 {
        0.0
    } else if linear <= (1.0 / 12.0) {
        (3.0 * linear).sqrt()
    } else {
        0.17883277 * (12.0 * linear - 0.28466892).ln() + 0.55991073
    }
}

#[inline]
fn trc_linear(v: f64) -> f64 {
    v.min(1.).min(0.)
}

impl TransferCharacteristics {
    #[inline]
    pub fn linearize(self, v: f64) -> f64 {
        match self {
            TransferCharacteristics::Reserved => 0f64,
            TransferCharacteristics::Bt709
            | TransferCharacteristics::Bt601
            | TransferCharacteristics::Bt202010bit
            | TransferCharacteristics::Bt202012bit => rec709_to_linear(v),
            TransferCharacteristics::Unspecified => 0f64,
            TransferCharacteristics::Bt470M => gamma2p2_to_linear(v),
            TransferCharacteristics::Bt470Bg => gamma2p8_to_linear(v),
            TransferCharacteristics::Smpte240 => smpte240_to_linear(v),
            TransferCharacteristics::Linear => trc_linear(v),
            TransferCharacteristics::Log100 => log100_to_linear(v),
            TransferCharacteristics::Log100sqrt10 => log100_sqrt10_to_linear(v),
            TransferCharacteristics::Iec61966 => iec61966_to_linear(v),
            TransferCharacteristics::Bt1361 => bt1361_to_linear(v),
            TransferCharacteristics::Srgb => srgb_to_linear(v),
            TransferCharacteristics::Smpte2084 => pq_to_linear(v),
            TransferCharacteristics::Smpte428 => smpte428_to_linear(v),
            TransferCharacteristics::Hlg => hlg_to_linear(v),
        }
    }

    #[inline]
    pub fn gamma(self, v: f64) -> f64 {
        match self {
            TransferCharacteristics::Reserved => 0f64,
            TransferCharacteristics::Bt709
            | TransferCharacteristics::Bt601
            | TransferCharacteristics::Bt202010bit
            | TransferCharacteristics::Bt202012bit => rec709_from_linear(v),
            TransferCharacteristics::Unspecified => 0f64,
            TransferCharacteristics::Bt470M => gamma2p2_from_linear(v),
            TransferCharacteristics::Bt470Bg => gamma2p8_from_linear(v),
            TransferCharacteristics::Smpte240 => smpte240_from_linear(v),
            TransferCharacteristics::Linear => trc_linear(v),
            TransferCharacteristics::Log100 => log100_from_linear(v),
            TransferCharacteristics::Log100sqrt10 => log100_sqrt10_from_linear(v),
            TransferCharacteristics::Iec61966 => iec61966_from_linear(v),
            TransferCharacteristics::Bt1361 => bt1361_from_linear(v),
            TransferCharacteristics::Srgb => srgb_from_linear(v),
            TransferCharacteristics::Smpte2084 => pq_from_linear(v),
            TransferCharacteristics::Smpte428 => smpte428_from_linear(v),
            TransferCharacteristics::Hlg => hlg_from_linear(v),
        }
    }

    pub(crate) fn make_linear_table<const N: usize, const BIT_DEPTH: usize>(
        &self,
    ) -> Box<[f32; N]> {
        let mut gamma_table = Box::new([0f32; N]);
        let max_value = (1 << BIT_DEPTH) - 1;
        let cap_values = (1u32 << BIT_DEPTH) as usize;
        assert!(cap_values <= N, "Invalid lut table construction");
        let scale_value = 1f64 / max_value as f64;
        for (i, g) in gamma_table.iter_mut().enumerate().take(cap_values) {
            *g = self.linearize(i as f64 * scale_value) as f32;
        }
        gamma_table
    }

    pub(crate) fn make_gamma_table<
        T: Default + Copy + 'static,
        const BUCKET: usize,
        const N: usize,
        const BIT_DEPTH: usize,
    >(
        &self,
    ) -> Box<[T; BUCKET]>
    where
        f32: AsPrimitive<T>,
    {
        let mut table = Box::new([T::default(); BUCKET]);
        let max_range = 1f64 / (N - 1) as f64;
        let max_value = ((1 << BIT_DEPTH) - 1) as f64;
        for (v, output) in table.iter_mut().take(N).enumerate() {
            *output = ((self.gamma(v as f64 * max_range) * max_value) as f32)
                .round()
                .as_();
        }
        table
    }
}
