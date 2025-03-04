/*
 * // Copyright (c) Radzivon Bartoshyk 2/2025. All rights reserved.
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
use crate::err::CmsError;
use crate::trc::{Trc, build_srgb_gamma_table, build_trc_table, curve_from_gamma};
use crate::{Chromacity, pow};
use std::convert::TryFrom;

/// See [Rec. ITU-T H.273 (12/2016)](https://www.itu.int/rec/T-REC-H.273-201612-I/en) Table 2
/// Values 0, 3, 13–21, 23–255 are all reserved so all map to the same variant
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ColorPrimaries {
    /// For future use by ITU-T | ISO/IEC
    Reserved,
    /// Rec. ITU-R BT.709-6<br />
    /// Rec. ITU-R BT.1361-0 conventional colour gamut system and extended colour gamut system (historical)<br />
    /// IEC 61966-2-1 sRGB or sYCC IEC 61966-2-4<br />
    /// Society of Motion Picture and Television Engineers (MPTE) RP 177 (1993) Annex B<br />
    Bt709 = 1,
    /// Unspecified<br />
    /// Image characteristics are unknown or are determined by the application.
    Unspecified = 2,
    /// Rec. ITU-R BT.470-6 System M (historical)<br />
    /// United States National Television System Committee 1953 Recommendation for transmission standards for color television<br />
    /// United States Federal Communications Commission (2003) Title 47 Code of Federal Regulations 73.682 (a) (20)<br />
    Bt470M = 4,
    /// Rec. ITU-R BT.470-6 System B, G (historical) Rec. ITU-R BT.601-7 625<br />
    /// Rec. ITU-R BT.1358-0 625 (historical)<br />
    /// Rec. ITU-R BT.1700-0 625 PAL and 625 SECAM<br />
    Bt470Bg = 5,
    /// Rec. ITU-R BT.601-7 525<br />
    /// Rec. ITU-R BT.1358-1 525 or 625 (historical) Rec. ITU-R BT.1700-0 NTSC<br />
    /// SMPTE 170M (2004)<br />
    /// (functionally the same as the value 7)<br />
    Bt601 = 6,
    /// SMPTE 240M (1999) (historical) (functionally the same as the value 6)<br />
    Smpte240 = 7,
    /// Generic film (colour filters using Illuminant C)<br />
    GenericFilm = 8,
    /// Rec. ITU-R BT.2020-2<br />
    /// Rec. ITU-R BT.2100-0<br />
    Bt2020 = 9,
    /// SMPTE ST 428-1<br />
    /// (CIE 1931 XYZ as in ISO 11664-1)<br />
    Xyz = 10,
    /// SMPTE RP 431-2 (2011)<br />
    Smpte431 = 11,
    /// SMPTE EG 432-1 (2010)<br />
    Smpte432 = 12,
    /// EBU Tech. 3213-E (1975)<br />
    Ebu3213 = 22,
}

impl TryFrom<u8> for ColorPrimaries {
    type Error = CmsError;

    #[allow(unreachable_patterns)]
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 | 3 | 13..=21 | 23..=255 => Ok(Self::Reserved),
            1 => Ok(Self::Bt709),
            2 => Ok(Self::Unspecified),
            4 => Ok(Self::Bt470M),
            5 => Ok(Self::Bt470Bg),
            6 => Ok(Self::Bt601),
            7 => Ok(Self::Smpte240),
            8 => Ok(Self::GenericFilm),
            9 => Ok(Self::Bt2020),
            10 => Ok(Self::Xyz),
            11 => Ok(Self::Smpte431),
            12 => Ok(Self::Smpte432),
            22 => Ok(Self::Ebu3213),
            _ => Err(CmsError::InvalidCicp),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ChromacityTriple {
    pub red: Chromacity,
    pub green: Chromacity,
    pub blue: Chromacity,
}

impl TryFrom<ColorPrimaries> for ChromacityTriple {
    type Error = CmsError;

    fn try_from(value: ColorPrimaries) -> Result<Self, Self::Error> {
        let red;
        let green;
        let blue;

        match value {
            ColorPrimaries::Reserved => {
                return Err(CmsError::UnsupportedColorPrimaries(value as u8));
            }
            ColorPrimaries::Bt709 => {
                green = Chromacity { x: 0.300, y: 0.600 };
                blue = Chromacity { x: 0.150, y: 0.060 };
                red = Chromacity { x: 0.640, y: 0.330 };
            }
            ColorPrimaries::Unspecified => {
                return Err(CmsError::UnsupportedColorPrimaries(value as u8));
            }
            ColorPrimaries::Bt470M => {
                green = Chromacity { x: 0.21, y: 0.71 };
                blue = Chromacity { x: 0.14, y: 0.08 };
                red = Chromacity { x: 0.67, y: 0.33 };
            }
            ColorPrimaries::Bt470Bg => {
                green = Chromacity { x: 0.29, y: 0.60 };
                blue = Chromacity { x: 0.15, y: 0.06 };
                red = Chromacity { x: 0.64, y: 0.33 };
            }
            ColorPrimaries::Bt601 | ColorPrimaries::Smpte240 => {
                green = Chromacity { x: 0.310, y: 0.595 };
                blue = Chromacity { x: 0.155, y: 0.070 };
                red = Chromacity { x: 0.630, y: 0.340 };
            }
            ColorPrimaries::GenericFilm => {
                green = Chromacity { x: 0.243, y: 0.692 };
                blue = Chromacity { x: 0.145, y: 0.049 };
                red = Chromacity { x: 0.681, y: 0.319 };
            }
            ColorPrimaries::Bt2020 => {
                green = Chromacity { x: 0.170, y: 0.797 };
                blue = Chromacity { x: 0.131, y: 0.046 };
                red = Chromacity { x: 0.708, y: 0.292 };
            }
            ColorPrimaries::Xyz => {
                green = Chromacity { x: 0.0, y: 1.0 };
                blue = Chromacity { x: 0.0, y: 0.0 };
                red = Chromacity { x: 1.0, y: 0.0 };
            }
            // These two share primaries, but have distinct white points
            ColorPrimaries::Smpte431 | ColorPrimaries::Smpte432 => {
                green = Chromacity { x: 0.265, y: 0.690 };
                blue = Chromacity { x: 0.150, y: 0.060 };
                red = Chromacity { x: 0.680, y: 0.320 };
            }
            ColorPrimaries::Ebu3213 => {
                green = Chromacity { x: 0.295, y: 0.605 };
                blue = Chromacity { x: 0.155, y: 0.077 };
                red = Chromacity { x: 0.630, y: 0.340 };
            }
        }

        Ok(Self { red, green, blue })
    }
}

impl ColorPrimaries {
    pub(crate) fn has_chromacity(self) -> bool {
        self != Self::Reserved && self != Self::Unspecified
    }

    pub(crate) fn white_point(self) -> Result<Chromacity, CmsError> {
        Ok(match self {
            Self::Reserved => return Err(CmsError::UnsupportedColorPrimaries(self as u8)),
            Self::Bt709
            | Self::Bt470Bg
            | Self::Bt601
            | Self::Smpte240
            | Self::Bt2020
            | Self::Smpte432
            | Self::Ebu3213 => Chromacity::D65,
            Self::Unspecified => return Err(CmsError::UnsupportedColorPrimaries(self as u8)),
            Self::Bt470M => Chromacity { x: 0.310, y: 0.316 },
            Self::GenericFilm => Chromacity { x: 0.310, y: 0.316 },
            Self::Xyz => Chromacity {
                x: 1. / 3.,
                y: 1. / 3.,
            },
            Self::Smpte431 => Chromacity { x: 0.314, y: 0.351 },
        })
    }
}

/// See [Rec. ITU-T H.273 (12/2016)](https://www.itu.int/rec/T-REC-H.273-201612-I/en) Table 3
/// Values 0, 3, 19–255 are all reserved so all map to the same variant
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TransferCharacteristics {
    /// For future use by ITU-T | ISO/IEC
    Reserved,
    /// Rec. ITU-R BT.709-6<br />
    /// Rec. ITU-R BT.1361-0 conventional colour gamut system (historical)<br />
    /// (functionally the same as the values 6, 14 and 15)    <br />
    Bt709 = 1,
    /// Image characteristics are unknown or are determined by the application.<br />
    Unspecified = 2,
    /// Rec. ITU-R BT.470-6 System M (historical)<br />
    /// United States National Television System Committee 1953 Recommendation for transmission standards for color television<br />
    /// United States Federal Communications Commission (2003) Title 47 Code of Federal Regulations 73.682 (a) (20)<br />
    /// Rec. ITU-R BT.1700-0 625 PAL and 625 SECAM<br />
    Bt470M = 4,
    /// Rec. ITU-R BT.470-6 System B, G (historical)<br />
    Bt470Bg = 5,
    /// Rec. ITU-R BT.601-7 525 or 625<br />
    /// Rec. ITU-R BT.1358-1 525 or 625 (historical)<br />
    /// Rec. ITU-R BT.1700-0 NTSC SMPTE 170M (2004)<br />
    /// (functionally the same as the values 1, 14 and 15)<br />
    Bt601 = 6,
    /// SMPTE 240M (1999) (historical)<br />
    Smpte240 = 7,
    /// Linear transfer characteristics<br />
    Linear = 8,
    /// Logarithmic transfer characteristic (100:1 range)<br />
    Log100 = 9,
    /// Logarithmic transfer characteristic (100 * Sqrt( 10 ) : 1 range)<br />
    Log100sqrt10 = 10,
    /// IEC 61966-2-4<br />
    Iec61966 = 11,
    /// Rec. ITU-R BT.1361-0 extended colour gamut system (historical)<br />
    Bt1361 = 12,
    /// IEC 61966-2-1 sRGB or sYCC<br />
    Srgb = 13,
    /// Rec. ITU-R BT.2020-2 (10-bit system)<br />
    /// (functionally the same as the values 1, 6 and 15)<br />
    Bt202010bit = 14,
    /// Rec. ITU-R BT.2020-2 (12-bit system)<br />
    /// (functionally the same as the values 1, 6 and 14)<br />
    Bt202012bit = 15,
    /// SMPTE ST 2084 for 10-, 12-, 14- and 16-bitsystems<br />
    /// Rec. ITU-R BT.2100-0 perceptual quantization (PQ) system<br />
    Smpte2084 = 16,
    /// SMPTE ST 428-1<br />
    Smpte428 = 17,
    /// ARIB STD-B67<br />
    /// Rec. ITU-R BT.2100-0 hybrid log- gamma (HLG) system<br />
    Hlg = 18,
}

impl TryFrom<u8> for TransferCharacteristics {
    type Error = CmsError;

    #[allow(unreachable_patterns)]
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 | 3 | 19..=255 => Ok(Self::Reserved),
            1 => Ok(Self::Bt709),
            2 => Ok(Self::Unspecified),
            4 => Ok(Self::Bt470M),
            5 => Ok(Self::Bt470Bg),
            6 => Ok(Self::Bt601),
            7 => Ok(Self::Smpte240), // unimplemented
            8 => Ok(Self::Linear),
            9 => Ok(Self::Log100),
            10 => Ok(Self::Log100sqrt10),
            11 => Ok(Self::Iec61966), // unimplemented
            12 => Ok(Self::Bt1361),   // unimplemented
            13 => Ok(Self::Srgb),
            14 => Ok(Self::Bt202010bit),
            15 => Ok(Self::Bt202012bit),
            16 => Ok(Self::Smpte2084),
            17 => Ok(Self::Smpte428), // unimplemented
            18 => Ok(Self::Hlg),
            _ => Err(CmsError::InvalidCicp),
        }
    }
}

impl TransferCharacteristics {
    pub(crate) fn has_transfer_curve(self) -> bool {
        self != Self::Reserved
            && self != Self::Unspecified
            && self != Self::Smpte240
            && self != Self::Iec61966
            && self != Self::Bt1361
            && self != Self::Smpte428
    }
}

impl TryFrom<TransferCharacteristics> for Trc {
    type Error = CmsError;
    /// See [ICC.1:2010](https://www.color.org/specification/ICC1v43_2010-12.pdf)
    /// See [Rec. ITU-R BT.2100-2](https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2100-2-201807-I!!PDF-E.pdf)
    fn try_from(value: TransferCharacteristics) -> Result<Self, Self::Error> {
        const NUM_TRC_TABLE_ENTRIES: i32 = 1024;

        Ok(match value {
            TransferCharacteristics::Reserved => {
                return Err(CmsError::UnsupportedTrc(value as u8));
            }
            TransferCharacteristics::Bt709
            | TransferCharacteristics::Bt601
            | TransferCharacteristics::Bt202010bit
            | TransferCharacteristics::Bt202012bit => {
                // The opto-electronic transfer characteristic function (OETF)
                // as defined in ITU-T H.273 table 3, row 1:
                //
                // V = (α * Lc^0.45) − (α − 1)  for 1 >= Lc >= β
                // V = 4.500 * Lc               for β >  Lc >= 0
                //
                // Inverting gives the electro-optical transfer characteristic
                // function (EOTF) which can be represented as ICC
                // parametricCurveType with 4 parameters (ICC.1:2010 Table 5).
                // Converting between the two (Lc ↔︎ Y, V ↔︎ X):
                //
                // Y = (a * X + b)^g  for (X >= d)
                // Y = c * X          for (X < d)
                //
                // g, a, b, c, d can then be defined in terms of α and β:
                //
                // g = 1 / 0.45
                // a = 1 / α
                // b = 1 - α
                // c = 1 / 4.500
                // d = 4.500 * β
                //
                // α and β are determined by solving the piecewise equations to
                // ensure continuity of both value and slope at the value β.
                // We use the values specified for 10-bit systems in
                // https://www.itu.int/rec/R-REC-BT.2020-2-201510-I Table 4
                // since this results in the similar values as available ICC
                // profiles after converting to s15Fixed16Number, providing us
                // good test coverage.

                type Float = f32;

                const ALPHA: Float = 1.099;
                const BETA: Float = 0.018;

                const LINEAR_CONF: Float = 4.500;
                const POW_EXP: Float = 0.45;

                const G: Float = 1. / POW_EXP;
                const A: Float = 1. / ALPHA;
                const B: Float = 1. - A;
                const C: Float = 1. / LINEAR_CONF;
                const D: Float = LINEAR_CONF * BETA;

                Trc::Parametric(vec![G, A, B, C, D])
            }
            TransferCharacteristics::Unspecified => {
                return Err(CmsError::UnsupportedTrc(value as u8));
            }
            TransferCharacteristics::Bt470M => curve_from_gamma(2.2),
            TransferCharacteristics::Bt470Bg => curve_from_gamma(2.8),
            TransferCharacteristics::Smpte240 => {
                return Err(CmsError::UnsupportedTrc(value as u8));
            }
            TransferCharacteristics::Linear => curve_from_gamma(1.),
            TransferCharacteristics::Log100 => {
                // See log_100_transfer_characteristics() for derivation
                // The opto-electronic transfer characteristic function (OETF)
                // as defined in ITU-T H.273 table 3, row 9:
                //
                // V = 1.0 + Log10(Lc) ÷ 2  for 1    >= Lc >= 0.01
                // V = 0.0                  for 0.01 >  Lc >= 0
                //
                // Inverting this to give the EOTF required for the profile gives
                //
                // Lc = 10^(2*V - 2)  for 1 >= V >= 0
                let table = build_trc_table(NUM_TRC_TABLE_ENTRIES, |v| pow(10f64, 2. * v - 2.));
                Trc::Lut(table)
            }
            TransferCharacteristics::Log100sqrt10 => {
                // The opto-electronic transfer characteristic function (OETF)
                // as defined in ITU-T H.273 table 3, row 10:
                //
                // V = 1.0 + Log10(Lc) ÷ 2.5  for               1 >= Lc >= Sqrt(10) ÷ 1000
                // V = 0.0                    for Sqrt(10) ÷ 1000 >  Lc >= 0
                //
                // Inverting this to give the EOTF required for the profile gives
                //
                // Lc = 10^(2.5*V - 2.5)  for 1 >= V >= 0
                let table = build_trc_table(NUM_TRC_TABLE_ENTRIES, |v| pow(10f64, 2.5 * v - 2.5));
                Trc::Lut(table)
            }
            TransferCharacteristics::Iec61966 => {
                return Err(CmsError::UnsupportedTrc(value as u8));
            }
            TransferCharacteristics::Bt1361 => return Err(CmsError::UnsupportedTrc(value as u8)),
            TransferCharacteristics::Srgb => {
                // Should we prefer this or curveType::Parametric?
                Trc::Lut(build_srgb_gamma_table(NUM_TRC_TABLE_ENTRIES))
            }

            TransferCharacteristics::Smpte2084 => {
                // Despite using Lo rather than Lc, H.273 gives the OETF:
                //
                // V = ( ( c1 + c2 * (Lo)^n ) ÷ ( 1 + c3 * (Lo)^n ) )^m
                const C1: f64 = 0.8359375;
                const C2: f64 = 18.8515625;
                const C3: f64 = 18.6875;
                const M: f64 = 78.84375;
                const N: f64 = 0.1593017578125;

                // Inverting this to give the EOTF required for the profile
                // (and confirmed by Rec. ITU-R BT.2100-2, Table 4) gives
                //
                // Y = ( max[( X^(1/m) - c1 ), 0] ÷ ( c2 - c3 * X^(1/m) ) )^(1/n)
                let table = build_trc_table(NUM_TRC_TABLE_ENTRIES, |x| {
                    pow(
                        (pow(x, 1. / M) - C1).max(0.) / (C2 - C3 * pow(x, 1. / M)),
                        1. / N,
                    )
                });
                Trc::Lut(table)
            }
            TransferCharacteristics::Smpte428 => {
                return Err(CmsError::UnsupportedTrc(value as u8));
            }
            TransferCharacteristics::Hlg => {
                // The opto-electronic transfer characteristic function (OETF)
                // as defined in ITU-T H.273 table 3, row 18:
                //
                // V = a * Ln(12 * Lc - b) + c  for 1      >= Lc >  1 ÷ 12
                // V = Sqrt(3) * Lc^0.5         for 1 ÷ 12 >= Lc >= 0
                const A: f64 = 0.17883277;
                const B: f64 = 0.28466892;
                const C: f64 = 0.55991073;

                // Inverting this to give the EOTF required for the profile
                // (and confirmed by Rec. ITU-R BT.2100-2, Table 4) gives
                //
                // Y = (X^2) / 3             for 0   <= X <= 0.5
                // Y = ((e^((X-c)/a))+b)/12  for 0.5 <  X <= 1
                let table = build_trc_table(NUM_TRC_TABLE_ENTRIES, |x| {
                    if x <= 0.5 {
                        let y1 = (x * x) / 3.;
                        debug_assert!((0. ..=1. / 12.).contains(&y1));
                        y1
                    } else {
                        (pow(std::f64::consts::E, (x - C) / A) + B) / 12.
                    }
                });
                Trc::Lut(table)
            }
        })
    }
}

/// Matrix Coefficients Enum (from ISO/IEC 23091-4 / MPEG CICP)
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(C)]
pub enum MatrixCoefficients {
    Identity = 0,                // RGB (Identity matrix)
    Bt709 = 1,                   // Rec. 709
    Unspecified = 2,             // Unspecified
    Reserved = 3,                // Reserved
    Fcc = 4,                     // FCC
    Bt470Bg = 5,                 // BT.470BG / BT.601-625
    Smpte170m = 6,               // SMPTE 170M / BT.601-525
    Smpte240m = 7,               // SMPTE 240M
    YCgCo = 8,                   // YCgCo
    Bt2020Ncl = 9,               // BT.2020 (non-constant luminance)
    Bt2020Cl = 10,               // BT.2020 (constant luminance)
    Smpte2085 = 11,              // SMPTE ST 2085
    ChromaticityDerivedNCL = 12, // Chromaticity-derived non-constant luminance
    ChromaticityDerivedCL = 13,  // Chromaticity-derived constant luminance
    ICtCp = 14,                  // ICtCp
}

impl TryFrom<u8> for MatrixCoefficients {
    type Error = CmsError;

    fn try_from(value: u8) -> Result<Self, CmsError> {
        match value {
            0 => Ok(MatrixCoefficients::Identity),
            1 => Ok(MatrixCoefficients::Bt709),
            2 => Ok(MatrixCoefficients::Unspecified),
            3 => Ok(MatrixCoefficients::Reserved),
            4 => Ok(MatrixCoefficients::Fcc),
            5 => Ok(MatrixCoefficients::Bt470Bg),
            6 => Ok(MatrixCoefficients::Smpte170m),
            7 => Ok(MatrixCoefficients::Smpte240m),
            8 => Ok(MatrixCoefficients::YCgCo),
            9 => Ok(MatrixCoefficients::Bt2020Ncl),
            10 => Ok(MatrixCoefficients::Bt2020Cl),
            11 => Ok(MatrixCoefficients::Smpte2085),
            12 => Ok(MatrixCoefficients::ChromaticityDerivedNCL),
            13 => Ok(MatrixCoefficients::ChromaticityDerivedCL),
            14 => Ok(MatrixCoefficients::ICtCp),
            _ => Err(CmsError::InvalidCicp),
        }
    }
}
