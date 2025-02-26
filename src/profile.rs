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
use crate::Chromacity;
use crate::chad::adapt_to_d50;
use crate::cicp::{ChromacityTriple, ColorPrimaries, MatrixCoefficients, TransferCharacteristics};
use crate::err::CmsError;
use crate::matrix::{BT2020_MATRIX, DISPLAY_P3_MATRIX, Matrix3f, SRGB_MATRIX, XyY, Xyz};
use crate::trc::{Trc, curve_from_gamma};
use std::io::Read;

const ACSP_SIGNATURE: u32 = u32::from_ne_bytes(*b"acsp").to_be(); // 'acsp' signature for ICC

/// Constants representing the min and max values that fit in a signed 32-bit integer as a float
const MAX_S32_FITS_IN_FLOAT: f32 = 2_147_483_647.0; // i32::MAX as f32
const MIN_S32_FITS_IN_FLOAT: f32 = -2_147_483_648.0; // i32::MIN as f32

/// Fixed-point scaling factor (assuming Fixed1 = 65536 like in ICC profiles)
const FIXED1: f32 = 65536.0;
const MAX_PROFILE_SIZE: usize = 1024 * 1024 * 3;
const TAG_SIZE: usize = 12;
const MARK_TRC_CURV: u32 = u32::from_ne_bytes(*b"curv").to_be();
const MARK_TRC_PARAM: u32 = u32::from_ne_bytes(*b"para").to_be();

const R_TAG_XYZ: u32 = u32::from_ne_bytes(*b"rXYZ").to_be();
const G_TAG_XYZ: u32 = u32::from_ne_bytes(*b"gXYZ").to_be();
const B_TAG_XYZ: u32 = u32::from_ne_bytes(*b"bXYZ").to_be();
const R_TAG_TRC: u32 = u32::from_ne_bytes(*b"rTRC").to_be();
const G_TAG_TRC: u32 = u32::from_ne_bytes(*b"gTRC").to_be();
const B_TAG_TRC: u32 = u32::from_ne_bytes(*b"bTRC").to_be();
const K_TAG_TRC: u32 = u32::from_ne_bytes(*b"kTRC").to_be();
const WT_PT_TAG: u32 = u32::from_ne_bytes(*b"wtpt").to_be();
const CICP_TAG: u32 = u32::from_ne_bytes(*b"cicp").to_be();
const CHAD_TAG: u32 = u32::from_ne_bytes(*b"chad").to_be();
const BLACKPOINT_TAG: u32 = u32::from_ne_bytes(*b"bkpt").to_be();
const ATOB0_TAG: u32 = u32::from_ne_bytes(*b"A2B0").to_be();
const B2A0_TAG: u32 = u32::from_ne_bytes(*b"B2A0").to_be();
const CHROMATIC_TYPE: u32 = u32::from_ne_bytes(*b"sf32").to_be();

#[inline]
fn uint8_number_to_float(a: u8) -> f32 {
    a as f32 / 255.0
}

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Default, Hash)]
pub enum DataColorSpace {
    #[default]
    Xyz,
    Lab,
    Luv,
    YCbr,
    Yxy,
    Rgb,
    Gray,
    Hsv,
    Hls,
    Cmyk,
    Cmy,
    Color2,
    Color3,
    Color4,
    Color5,
    Color6,
    Color7,
    Color8,
    Color9,
    Color10,
    Color11,
    Color12,
    Color13,
    Color14,
    Color15,
}

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Default)]
pub enum ProfileClass {
    InputDevice,
    #[default]
    DisplayDevice,
    OutputDevice,
    DeviceLink,
    ColorSpace,
    Abstract,
    Named,
}

impl TryFrom<u32> for ProfileClass {
    type Error = CmsError;
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        if value == u32::from_ne_bytes(*b"scnr").to_be() {
            return Ok(ProfileClass::InputDevice);
        } else if value == u32::from_ne_bytes(*b"mntr").to_be() {
            return Ok(ProfileClass::DisplayDevice);
        } else if value == u32::from_ne_bytes(*b"prtr").to_be() {
            return Ok(ProfileClass::OutputDevice);
        } else if value == u32::from_ne_bytes(*b"link").to_be() {
            return Ok(ProfileClass::DeviceLink);
        } else if value == u32::from_ne_bytes(*b"spac").to_be() {
            return Ok(ProfileClass::ColorSpace);
        } else if value == u32::from_ne_bytes(*b"abst").to_be() {
            return Ok(ProfileClass::Abstract);
        } else if value == u32::from_ne_bytes(*b"nmcl").to_be() {
            return Ok(ProfileClass::Named);
        }
        Err(CmsError::InvalidIcc)
    }
}

impl From<ProfileClass> for u32 {
    fn from(val: ProfileClass) -> Self {
        match val {
            ProfileClass::InputDevice => u32::from_ne_bytes(*b"scnr").to_be(),
            ProfileClass::DisplayDevice => u32::from_ne_bytes(*b"mntr").to_be(),
            ProfileClass::OutputDevice => u32::from_ne_bytes(*b"prtr").to_be(),
            ProfileClass::DeviceLink => u32::from_ne_bytes(*b"link").to_be(),
            ProfileClass::ColorSpace => u32::from_ne_bytes(*b"spac").to_be(),
            ProfileClass::Abstract => u32::from_ne_bytes(*b"abst").to_be(),
            ProfileClass::Named => u32::from_ne_bytes(*b"nmcl").to_be(),
        }
    }
}

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub enum LutType {
    Lut8,
    Lut16,
    LutMab,
    LutMba,
}

impl TryFrom<u32> for LutType {
    type Error = CmsError;
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        if value == u32::from_ne_bytes(*b"mft1").to_be() {
            return Ok(LutType::Lut8);
        } else if value == u32::from_ne_bytes(*b"mft2").to_be() {
            return Ok(LutType::Lut16);
        } else if value == u32::from_ne_bytes(*b"mAB ").to_be() {
            return Ok(LutType::LutMab);
        } else if value == u32::from_ne_bytes(*b"mBA ").to_be() {
            return Ok(LutType::LutMba);
        }
        Err(CmsError::InvalidIcc)
    }
}

impl TryFrom<u32> for DataColorSpace {
    type Error = CmsError;
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        if value == u32::from_ne_bytes(*b"XYZ ").to_be() {
            return Ok(DataColorSpace::Xyz);
        } else if value == u32::from_ne_bytes(*b"Lab ").to_be() {
            return Ok(DataColorSpace::Lab);
        } else if value == u32::from_ne_bytes(*b"Luv ").to_be() {
            return Ok(DataColorSpace::Luv);
        } else if value == u32::from_ne_bytes(*b"YCbr").to_be() {
            return Ok(DataColorSpace::YCbr);
        } else if value == u32::from_ne_bytes(*b"Yxy ").to_be() {
            return Ok(DataColorSpace::Yxy);
        } else if value == u32::from_ne_bytes(*b"RGB ").to_be() {
            return Ok(DataColorSpace::Rgb);
        } else if value == u32::from_ne_bytes(*b"GRAY").to_be() {
            return Ok(DataColorSpace::Gray);
        } else if value == u32::from_ne_bytes(*b"HSV ").to_be() {
            return Ok(DataColorSpace::Hsv);
        } else if value == u32::from_ne_bytes(*b"HLS ").to_be() {
            return Ok(DataColorSpace::Hls);
        } else if value == u32::from_ne_bytes(*b"CMYK").to_be() {
            return Ok(DataColorSpace::Cmyk);
        } else if value == u32::from_ne_bytes(*b"CMY ").to_be() {
            return Ok(DataColorSpace::Cmy);
        } else if value == u32::from_ne_bytes(*b"2CLR").to_be() {
            return Ok(DataColorSpace::Color2);
        } else if value == u32::from_ne_bytes(*b"3CLR").to_be() {
            return Ok(DataColorSpace::Color3);
        } else if value == u32::from_ne_bytes(*b"4CLR").to_be() {
            return Ok(DataColorSpace::Color4);
        } else if value == u32::from_ne_bytes(*b"5CLR").to_be() {
            return Ok(DataColorSpace::Color5);
        } else if value == u32::from_ne_bytes(*b"6CLR").to_be() {
            return Ok(DataColorSpace::Color6);
        } else if value == u32::from_ne_bytes(*b"7CLR").to_be() {
            return Ok(DataColorSpace::Color7);
        } else if value == u32::from_ne_bytes(*b"8CLR").to_be() {
            return Ok(DataColorSpace::Color8);
        } else if value == u32::from_ne_bytes(*b"9CLR").to_be() {
            return Ok(DataColorSpace::Color9);
        } else if value == u32::from_ne_bytes(*b"ACLR").to_be() {
            return Ok(DataColorSpace::Color10);
        } else if value == u32::from_ne_bytes(*b"BCLR").to_be() {
            return Ok(DataColorSpace::Color11);
        } else if value == u32::from_ne_bytes(*b"CCLR").to_be() {
            return Ok(DataColorSpace::Color12);
        } else if value == u32::from_ne_bytes(*b"DCLR").to_be() {
            return Ok(DataColorSpace::Color13);
        } else if value == u32::from_ne_bytes(*b"ECLR").to_be() {
            return Ok(DataColorSpace::Color14);
        } else if value == u32::from_ne_bytes(*b"FCLR").to_be() {
            return Ok(DataColorSpace::Color15);
        }
        Err(CmsError::InvalidIcc)
    }
}

impl From<DataColorSpace> for u32 {
    fn from(val: DataColorSpace) -> Self {
        match val {
            DataColorSpace::Xyz => u32::from_ne_bytes(*b"XYZ ").to_be(),
            DataColorSpace::Lab => u32::from_ne_bytes(*b"Lab ").to_be(),
            DataColorSpace::Luv => u32::from_ne_bytes(*b"Luv ").to_be(),
            DataColorSpace::YCbr => u32::from_ne_bytes(*b"YCbr").to_be(),
            DataColorSpace::Yxy => u32::from_ne_bytes(*b"Yxy ").to_be(),
            DataColorSpace::Rgb => u32::from_ne_bytes(*b"RGB ").to_be(),
            DataColorSpace::Gray => u32::from_ne_bytes(*b"GRAY").to_be(),
            DataColorSpace::Hsv => u32::from_ne_bytes(*b"HSV ").to_be(),
            DataColorSpace::Hls => u32::from_ne_bytes(*b"HLS ").to_be(),
            DataColorSpace::Cmyk => u32::from_ne_bytes(*b"CMYK").to_be(),
            DataColorSpace::Cmy => u32::from_ne_bytes(*b"CMY ").to_be(),
            DataColorSpace::Color2 => u32::from_ne_bytes(*b"2CLR").to_be(),
            DataColorSpace::Color3 => u32::from_ne_bytes(*b"3CLR").to_be(),
            DataColorSpace::Color4 => u32::from_ne_bytes(*b"4CLR").to_be(),
            DataColorSpace::Color5 => u32::from_ne_bytes(*b"5CLR").to_be(),
            DataColorSpace::Color6 => u32::from_ne_bytes(*b"6CLR").to_be(),
            DataColorSpace::Color7 => u32::from_ne_bytes(*b"7CLR").to_be(),
            DataColorSpace::Color8 => u32::from_ne_bytes(*b"8CLR").to_be(),
            DataColorSpace::Color9 => u32::from_ne_bytes(*b"9CLR").to_be(),
            DataColorSpace::Color10 => u32::from_ne_bytes(*b"ACLR").to_be(),
            DataColorSpace::Color11 => u32::from_ne_bytes(*b"BCLR").to_be(),
            DataColorSpace::Color12 => u32::from_ne_bytes(*b"CCLR").to_be(),
            DataColorSpace::Color13 => u32::from_ne_bytes(*b"DCLR").to_be(),
            DataColorSpace::Color14 => u32::from_ne_bytes(*b"ECLR").to_be(),
            DataColorSpace::Color15 => u32::from_ne_bytes(*b"FCLR").to_be(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LutDataType {
    // used by lut8Type/lut16Type (mft2) only
    pub num_input_channels: u8,
    pub num_output_channels: u8,
    pub num_clut_grid_points: u8,
    pub matrix: Matrix3f,
    pub num_input_table_entries: u16,
    pub num_output_table_entries: u16,
    pub input_table: Vec<f32>,
    pub clut_table: Vec<f32>,
    pub output_table: Vec<f32>,
    pub lut_type: LutType,
}

/// Clamps the float value within the range of an `i32`
/// Returns `i32::MAX` for NaN values.
#[inline]
const fn float_saturate2int(x: f32) -> i32 {
    if x.is_nan() {
        return i32::MAX;
    }
    x.clamp(MIN_S32_FITS_IN_FLOAT, MAX_S32_FITS_IN_FLOAT) as i32
}

/// Converts a float to a fixed-point integer representation
#[inline]
const fn float_round_to_fixed(x: f32) -> i32 {
    float_saturate2int((x as f64 * FIXED1 as f64 + 0.5) as f32)
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub enum RenderingIntent {
    AbsoluteColorimetric = 3,
    Saturation = 2,
    RelativeColorimetric = 1,
    #[default]
    Perceptual = 0,
}

impl TryFrom<u32> for RenderingIntent {
    type Error = CmsError;
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(RenderingIntent::Perceptual),
            1 => Ok(RenderingIntent::RelativeColorimetric),
            2 => Ok(RenderingIntent::Saturation),
            3 => Ok(RenderingIntent::AbsoluteColorimetric),
            _ => Err(CmsError::InvalidRenderingIntent),
        }
    }
}

/// ICC Header
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct IccHeader {
    pub size: u32,                    // Size of the profile (computed)
    pub cmm_type: u32,                // Preferred CMM type (ignored)
    pub version: u32,                 // Version (4.3 or 4.4 if CICP is included)
    pub profile_class: u32,           // Display device profile
    pub data_color_space: u32,        // RGB input color space
    pub pcs: u32,                     // Profile connection space
    pub creation_date_time: [u8; 12], // Date and time (ignored)
    pub signature: u32,               // Profile signature
    pub platform: u32,                // Platform target (ignored)
    pub flags: u32,                   // Flags (not embedded, can be used independently)
    pub device_manufacturer: u32,     // Device manufacturer (ignored)
    pub device_model: u32,            // Device model (ignored)
    pub device_attributes: [u8; 8],   // Device attributes (ignored)
    pub rendering_intent: u32,        // Relative colorimetric rendering intent
    pub illuminant_x: i32,            // D50 standard illuminant X
    pub illuminant_y: i32,            // D50 standard illuminant Y
    pub illuminant_z: i32,            // D50 standard illuminant Z
    pub creator: u32,                 // Profile creator (ignored)
    pub profile_id: [u8; 16],         // Profile id checksum (ignored)
    pub reserved: [u8; 28],           // Reserved (ignored)
    pub tag_count: u32,               // Technically not part of header, but required
}

impl IccHeader {
    #[allow(dead_code)]
    pub(crate) fn new(size: u32) -> Self {
        Self {
            size,
            cmm_type: 0,
            version: 0x04300000u32.to_be(),
            profile_class: ProfileClass::DisplayDevice.into(),
            data_color_space: DataColorSpace::Rgb.into(),
            pcs: DataColorSpace::Xyz.into(),
            creation_date_time: [0; 12],
            signature: ACSP_SIGNATURE.to_be(),
            platform: 0,
            flags: 0x00000000,
            device_manufacturer: 0,
            device_model: 0,
            device_attributes: [0; 8],
            rendering_intent: 1u32.to_be(),
            illuminant_x: float_round_to_fixed(Chromacity::D50.to_xyz().x).to_be(),
            illuminant_y: float_round_to_fixed(Chromacity::D50.to_xyz().y).to_be(),
            illuminant_z: float_round_to_fixed(Chromacity::D50.to_xyz().z).to_be(),
            creator: 0,
            profile_id: [0; 16],
            reserved: [0; 28],
            tag_count: 0,
        }
    }

    pub(crate) fn new_from_slice(slice: &[u8]) -> Result<Self, CmsError> {
        if slice.len() < size_of::<IccHeader>() {
            return Err(CmsError::InvalidIcc);
        }
        let mut cursor = std::io::Cursor::new(slice);
        let mut buffer = [0u8; size_of::<IccHeader>()];
        cursor
            .read_exact(&mut buffer)
            .map_err(|_| CmsError::InvalidIcc)?;

        let header = Self {
            size: u32::from_be_bytes(buffer[0..4].try_into().unwrap()),
            cmm_type: u32::from_be_bytes(buffer[4..8].try_into().unwrap()),
            version: u32::from_be_bytes(buffer[8..12].try_into().unwrap()),
            profile_class: u32::from_be_bytes(buffer[12..16].try_into().unwrap()),
            data_color_space: u32::from_be_bytes(buffer[16..20].try_into().unwrap()),
            pcs: u32::from_be_bytes(buffer[20..24].try_into().unwrap()),
            creation_date_time: buffer[24..36].try_into().unwrap(),
            signature: u32::from_be_bytes(buffer[36..40].try_into().unwrap()),
            platform: u32::from_be_bytes(buffer[40..44].try_into().unwrap()),
            flags: u32::from_be_bytes(buffer[44..48].try_into().unwrap()),
            device_manufacturer: u32::from_be_bytes(buffer[48..52].try_into().unwrap()),
            device_model: u32::from_be_bytes(buffer[52..56].try_into().unwrap()),
            device_attributes: buffer[56..64].try_into().unwrap(),
            rendering_intent: u32::from_be_bytes(buffer[64..68].try_into().unwrap()),
            illuminant_x: i32::from_be_bytes(buffer[68..72].try_into().unwrap()),
            illuminant_y: i32::from_be_bytes(buffer[72..76].try_into().unwrap()),
            illuminant_z: i32::from_be_bytes(buffer[76..80].try_into().unwrap()),
            creator: u32::from_be_bytes(buffer[80..84].try_into().unwrap()),
            profile_id: buffer[84..100].try_into().unwrap(),
            reserved: buffer[100..128].try_into().unwrap(),
            tag_count: u32::from_be_bytes(buffer[128..132].try_into().unwrap()),
        };

        if header.signature != ACSP_SIGNATURE {
            return Err(CmsError::InvalidIcc);
        }
        Ok(header)
    }
}

/// Representation of Coding Independent Code Point
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CicpProfile {
    pub color_primaries: ColorPrimaries,
    pub transfer_characteristics: TransferCharacteristics,
    pub matrix_coefficients: MatrixCoefficients,
}

/// ICC Profile representation
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct ColorProfile {
    pub pcs: DataColorSpace,
    pub color_space: DataColorSpace,
    pub profile_class: ProfileClass,
    pub rendering_intent: RenderingIntent,
    pub red_colorant: Xyz,
    pub green_colorant: Xyz,
    pub blue_colorant: Xyz,
    pub white_point: Option<Xyz>,
    pub black_point: Option<Xyz>,
    pub image_white_point: Xyz,
    pub red_trc: Option<Trc>,
    pub green_trc: Option<Trc>,
    pub blue_trc: Option<Trc>,
    pub gray_trc: Option<Trc>,
    pub cicp: Option<CicpProfile>,
    pub chromatic_adaptation: Option<Matrix3f>,
    pub lut_a_to_b: Option<LutDataType>,
    pub lut_b_to_a: Option<LutDataType>,
}

/* produces the nearest float to 'a' with a maximum error
 * of 1/1024 which happens for large values like 0x40000040 */
#[inline]
pub(crate) const fn s15_fixed16_number_to_float(a: i32) -> f32 {
    a as f32 / 65536.
}

#[inline]
const fn uint16_number_to_float(a: u32) -> f32 {
    a as f32 / 65536.
}

impl ColorProfile {
    #[inline]
    fn read_trc_tag(slice: &[u8], entry: usize, tag_size: usize) -> Result<Option<Trc>, CmsError> {
        if tag_size < TAG_SIZE {
            return Ok(None);
        }
        let last_tag_offset = tag_size + entry;
        if last_tag_offset > slice.len() {
            return Err(CmsError::InvalidIcc);
        }
        let tag = &slice[entry..last_tag_offset];
        if tag.len() < TAG_SIZE {
            return Err(CmsError::InvalidIcc);
        }

        let curve_type = u32::from_be_bytes([tag[0], tag[1], tag[2], tag[3]]);
        if curve_type == MARK_TRC_CURV {
            let entry_count = u32::from_be_bytes([tag[8], tag[9], tag[10], tag[11]]) as usize;
            if entry_count == 0 {
                return Ok(Some(Trc::Lut(vec![])));
            }
            if entry_count > 40000 {
                return Err(CmsError::CurveLutIsTooLarge);
            }
            if tag.len() < 12 + entry_count * size_of::<u16>() {
                return Err(CmsError::InvalidIcc);
            }
            let curve_sliced = &tag[12..12 + entry_count * size_of::<u16>()];
            let mut curve_values = vec![0u16; entry_count];
            for (value, curve_value) in curve_sliced.chunks_exact(2).zip(curve_values.iter_mut()) {
                let gamma_s15 = u16::from_be_bytes([value[0], value[1]]);
                *curve_value = gamma_s15;
            }
            Ok(Some(Trc::Lut(curve_values)))
        } else if curve_type == MARK_TRC_PARAM {
            let entry_count = u16::from_be_bytes([tag[8], tag[9]]) as usize;
            if entry_count > 4 {
                return Err(CmsError::InvalidIcc);
            }

            const COUNT_TO_LENGTH: [usize; 5] = [1, 3, 4, 5, 7]; //PARAMETRIC_CURVE_TYPE

            if tag.len() < 12 + COUNT_TO_LENGTH[entry_count] * size_of::<u32>() {
                return Err(CmsError::InvalidIcc);
            }
            let curve_sliced = &tag[12..12 + COUNT_TO_LENGTH[entry_count] * size_of::<u32>()];
            let mut params = vec![0f32; COUNT_TO_LENGTH[entry_count]];
            for (value, param_value) in curve_sliced.chunks_exact(4).zip(params.iter_mut()) {
                let parametric_value = u32::from_be_bytes([value[0], value[1], value[2], value[3]]);
                *param_value = uint16_number_to_float(parametric_value);
            }
            if entry_count == 1 || entry_count == 2 {
                /* we have a type 1 or type 2 function that has a division by 'a' */
                let a: f32 = params[1];
                if a == 0.0 {
                    return Err(CmsError::ParametricCurveZeroDivision);
                }
            }
            return Ok(Some(Trc::Parametric(params)));
        } else {
            return Err(CmsError::InvalidIcc);
        }
    }

    #[inline]
    fn read_chad_tag(slice: &[u8], entry: usize, tag_size: usize) -> Result<Matrix3f, CmsError> {
        let tag_size = if tag_size == 0 { TAG_SIZE } else { tag_size };
        let last_tag_offset = tag_size + entry;
        if last_tag_offset > slice.len() {
            return Err(CmsError::InvalidIcc);
        }
        if slice[entry..].len() < 8 {
            return Err(CmsError::InvalidIcc);
        }
        let tag0 = &slice[entry..entry + 8];
        let c_type = u32::from_be_bytes([tag0[0], tag0[1], tag0[2], tag0[3]]);
        if c_type != CHROMATIC_TYPE {
            return Err(CmsError::InvalidIcc);
        }
        if slice.len() < 9 * size_of::<u32>() + 8 {
            return Err(CmsError::InvalidIcc);
        }
        let tag = &slice[entry + 8..last_tag_offset];
        if tag.len() != size_of::<Matrix3f>() {
            return Err(CmsError::InvalidIcc);
        }
        let mut matrix = Matrix3f::default();
        for (i, chunk) in tag.chunks_exact(4).enumerate() {
            let q15_16_x = i32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            matrix.v[i / 3][i % 3] = s15_fixed16_number_to_float(q15_16_x);
        }
        Ok(matrix)
    }

    #[inline]
    fn read_xyz_tag(slice: &[u8], entry: usize, tag_size: usize) -> Result<Xyz, CmsError> {
        if tag_size < TAG_SIZE {
            return Ok(Xyz::default());
        }
        let last_tag_offset = tag_size + entry;
        if last_tag_offset > slice.len() {
            return Err(CmsError::InvalidIcc);
        }
        let tag = &slice[entry..last_tag_offset];
        if tag.len() < 20 {
            return Err(CmsError::InvalidIcc);
        }
        let q15_16_x = i32::from_be_bytes([tag[8], tag[9], tag[10], tag[11]]);
        let q15_16_y = i32::from_be_bytes([tag[12], tag[13], tag[14], tag[15]]);
        let q15_16_z = i32::from_be_bytes([tag[16], tag[17], tag[18], tag[19]]);
        let x = s15_fixed16_number_to_float(q15_16_x);
        let y = s15_fixed16_number_to_float(q15_16_y);
        let z = s15_fixed16_number_to_float(q15_16_z);
        Ok(Xyz { x, y, z })
    }

    #[inline]
    fn read_cicp_tag(
        slice: &[u8],
        entry: usize,
        tag_size: usize,
    ) -> Result<Option<CicpProfile>, CmsError> {
        if tag_size < TAG_SIZE {
            return Ok(None);
        }
        let last_tag_offset = tag_size + entry;
        if last_tag_offset > slice.len() {
            return Err(CmsError::InvalidIcc);
        }
        let tag = &slice[entry..last_tag_offset];
        if tag.len() < 12 {
            return Err(CmsError::InvalidIcc);
        }
        let primaries = ColorPrimaries::try_from(tag[8])?;
        let transfer_characteristics = TransferCharacteristics::try_from(tag[9])?;
        let matrix_coefficients = MatrixCoefficients::try_from(tag[10])?;
        Ok(Some(CicpProfile {
            color_primaries: primaries,
            transfer_characteristics,
            matrix_coefficients,
        }))
    }

    #[inline]
    fn read_lut_type(slice: &[u8], entry: usize, tag_size: usize) -> Result<LutType, CmsError> {
        let tag_size = if tag_size == 0 { TAG_SIZE } else { tag_size };
        let last_tag_offset = tag_size + entry;
        if last_tag_offset > slice.len() {
            return Err(CmsError::InvalidIcc);
        }
        let tag = &slice[entry..last_tag_offset];
        if tag.len() < 48 {
            return Err(CmsError::InvalidIcc);
        }
        let tag_type = u32::from_be_bytes([tag[0], tag[1], tag[2], tag[3]]);
        LutType::try_from(tag_type)
    }

    #[inline]
    fn read_lut_table_f32(table: &[u8], output: &mut [f32], lut_type: LutType) {
        if lut_type == LutType::Lut16 {
            for (src, dst) in table.chunks_exact(2).zip(output.iter_mut()) {
                *dst = uint16_number_to_float(u16::from_be_bytes([src[0], src[1]]) as u32);
            }
        } else if lut_type == LutType::Lut8 {
            for (src, dst) in table.iter().zip(output.iter_mut()) {
                *dst = uint8_number_to_float(*src);
            }
        }
    }

    #[inline]
    fn read_lut_a_to_b_type(
        slice: &[u8],
        entry: usize,
        tag_size: usize,
    ) -> Result<Option<LutDataType>, CmsError> {
        if tag_size < 48 {
            return Ok(None);
        }
        let last_tag_offset = tag_size + entry;
        if last_tag_offset > slice.len() {
            return Err(CmsError::InvalidIcc);
        }
        let tag = &slice[entry..last_tag_offset];
        if tag.len() < 48 {
            return Err(CmsError::InvalidIcc);
        }
        let tag_type = u32::from_be_bytes([tag[0], tag[1], tag[2], tag[3]]);
        let lut_type = LutType::try_from(tag_type)?;
        assert!(lut_type == LutType::Lut8 || lut_type == LutType::Lut16);

        if lut_type == LutType::Lut16 && tag.len() < 52 {
            return Err(CmsError::InvalidIcc);
        }

        let num_input_table_entries: u16 = match lut_type {
            LutType::Lut8 => 256,
            LutType::Lut16 => u16::from_be_bytes([tag[48], tag[49]]),
            _ => unreachable!(),
        };
        let num_output_table_entries: u16 = match lut_type {
            LutType::Lut8 => 256,
            LutType::Lut16 => u16::from_be_bytes([tag[50], tag[51]]),
            _ => unreachable!(),
        };

        if !(2..=4096).contains(&num_input_table_entries)
            || !(2..=4096).contains(&num_output_table_entries)
        {
            return Err(CmsError::InvalidIcc);
        }

        let input_offset: usize = match lut_type {
            LutType::Lut8 => 48,
            LutType::Lut16 => 52,
            _ => unreachable!(),
        };
        let entry_size: usize = match lut_type {
            LutType::Lut8 => 1,
            LutType::Lut16 => 2,
            _ => unreachable!(),
        };

        let in_chan = tag[8];
        let out_chan = tag[9];
        let is_3_to_4 = in_chan == 3 || out_chan == 4;
        let is_4_to_3 = in_chan == 4 || out_chan == 3;
        if !is_3_to_4 && !is_4_to_3 {
            return Err(CmsError::InvalidIcc);
        }
        let grid_points = tag[10];
        let clut_size = match (grid_points as u32).checked_pow(in_chan as u32) {
            Some(clut_size) => clut_size as usize,
            _ => {
                return Err(CmsError::InvalidIcc);
            }
        };
        match clut_size {
            1..=500_000 => {} // OK
            0 => {
                return Err(CmsError::InvalidIcc);
            }
            _ => {
                return Err(CmsError::InvalidIcc);
            }
        }

        assert!(tag.len() >= 48);

        let e00 = i32::from_be_bytes([tag[12], tag[13], tag[14], tag[15]]);
        let e01 = i32::from_be_bytes([tag[16], tag[17], tag[18], tag[19]]);
        let e02 = i32::from_be_bytes([tag[20], tag[21], tag[22], tag[23]]);
        let e10 = i32::from_be_bytes([tag[24], tag[25], tag[26], tag[27]]);
        let e11 = i32::from_be_bytes([tag[28], tag[29], tag[30], tag[31]]);
        let e12 = i32::from_be_bytes([tag[32], tag[33], tag[34], tag[35]]);
        let e20 = i32::from_be_bytes([tag[36], tag[37], tag[38], tag[39]]);
        let e21 = i32::from_be_bytes([tag[40], tag[41], tag[42], tag[43]]);
        let e22 = i32::from_be_bytes([tag[44], tag[45], tag[46], tag[47]]);

        let transform = Matrix3f {
            v: [
                [
                    s15_fixed16_number_to_float(e00),
                    s15_fixed16_number_to_float(e01),
                    s15_fixed16_number_to_float(e02),
                ],
                [
                    s15_fixed16_number_to_float(e10),
                    s15_fixed16_number_to_float(e11),
                    s15_fixed16_number_to_float(e12),
                ],
                [
                    s15_fixed16_number_to_float(e20),
                    s15_fixed16_number_to_float(e21),
                    s15_fixed16_number_to_float(e22),
                ],
            ],
        };

        let lut_input_size = (num_input_table_entries * in_chan as u16) as usize;

        let mut input_table = vec![0f32; lut_input_size];
        if tag.len() < input_offset + lut_input_size * entry_size {
            return Err(CmsError::InvalidIcc);
        }
        let shaped_input_table = &tag[input_offset..input_offset + lut_input_size * entry_size];
        Self::read_lut_table_f32(shaped_input_table, &mut input_table, lut_type);

        let clut_size_table_size =
            (num_input_table_entries as usize * in_chan as usize) * entry_size;

        let clut_offset = input_offset + clut_size_table_size;

        if tag.len() < clut_offset + clut_size_table_size * out_chan as usize {
            return Err(CmsError::InvalidIcc);
        }

        let clut_data_size = (clut_size * out_chan as usize) * entry_size;

        let mut clut_table = vec![0f32; clut_size * out_chan as usize];

        let shaped_clut_table = &tag[clut_offset..clut_offset + clut_data_size];
        Self::read_lut_table_f32(shaped_clut_table, &mut clut_table, lut_type);

        let output_offset = clut_offset + clut_data_size;

        let output_size = num_output_table_entries as usize * out_chan as usize;

        let mut output_table = vec![0f32; output_size];
        let shaped_output_table = &tag[output_offset..output_offset + output_size * entry_size];
        Self::read_lut_table_f32(shaped_output_table, &mut output_table, lut_type);

        Ok(Some(LutDataType {
            num_input_table_entries,
            num_output_table_entries,
            num_input_channels: in_chan,
            num_output_channels: out_chan,
            num_clut_grid_points: grid_points,
            matrix: transform,
            input_table,
            clut_table,
            output_table,
            lut_type,
        }))
    }

    #[allow(clippy::field_reassign_with_default)]
    pub fn new_from_slice(slice: &[u8]) -> Result<Self, CmsError> {
        let header = IccHeader::new_from_slice(slice)?;
        let tags_count = header.tag_count as usize;
        if slice.len() >= MAX_PROFILE_SIZE {
            return Err(CmsError::InvalidIcc);
        }
        if slice.len() < tags_count * TAG_SIZE + size_of::<IccHeader>() {
            return Err(CmsError::InvalidIcc);
        }
        let tags_slice =
            &slice[size_of::<IccHeader>()..size_of::<IccHeader>() + tags_count * TAG_SIZE];
        let mut profile = ColorProfile::default();
        profile.rendering_intent = RenderingIntent::try_from(header.rendering_intent)?;
        profile.pcs = DataColorSpace::try_from(header.pcs)?;
        profile.profile_class = ProfileClass::try_from(header.profile_class)?;
        profile.color_space = DataColorSpace::try_from(header.data_color_space)?;
        profile.image_white_point = Xyz {
            x: s15_fixed16_number_to_float(header.illuminant_x),
            y: s15_fixed16_number_to_float(header.illuminant_y),
            z: s15_fixed16_number_to_float(header.illuminant_z),
        };
        let color_space = profile.color_space;
        let known_profile_class = profile.profile_class == ProfileClass::DisplayDevice
            || profile.profile_class == ProfileClass::InputDevice
            || profile.profile_class == ProfileClass::OutputDevice
            || profile.profile_class == ProfileClass::ColorSpace;
        if known_profile_class {
            for tag in tags_slice.chunks_exact(TAG_SIZE) {
                let tag_value = u32::from_be_bytes([tag[0], tag[1], tag[2], tag[3]]);
                let tag_entry = u32::from_be_bytes([tag[4], tag[5], tag[6], tag[7]]);
                let tag_size = u32::from_be_bytes([tag[8], tag[9], tag[10], tag[11]]) as usize;
                if tag_value == R_TAG_XYZ && color_space == DataColorSpace::Rgb {
                    profile.red_colorant = Self::read_xyz_tag(slice, tag_entry as usize, tag_size)?;
                } else if tag_value == G_TAG_XYZ && color_space == DataColorSpace::Rgb {
                    profile.green_colorant =
                        Self::read_xyz_tag(slice, tag_entry as usize, tag_size)?;
                } else if tag_value == B_TAG_XYZ && color_space == DataColorSpace::Rgb {
                    profile.blue_colorant =
                        Self::read_xyz_tag(slice, tag_entry as usize, tag_size)?;
                } else if tag_value == CICP_TAG {
                    profile.cicp = Self::read_cicp_tag(slice, tag_entry as usize, tag_size)?;
                } else if tag_value == R_TAG_TRC && color_space == DataColorSpace::Rgb {
                    match Self::read_trc_tag(slice, tag_entry as usize, tag_size) {
                        Ok(trc) => profile.red_trc = trc,
                        Err(err) => return Err(err),
                    }
                } else if tag_value == G_TAG_TRC && color_space == DataColorSpace::Rgb {
                    match Self::read_trc_tag(slice, tag_entry as usize, tag_size) {
                        Ok(trc) => profile.green_trc = trc,
                        Err(err) => return Err(err),
                    }
                } else if tag_value == B_TAG_TRC && color_space == DataColorSpace::Rgb {
                    match Self::read_trc_tag(slice, tag_entry as usize, tag_size) {
                        Ok(trc) => profile.blue_trc = trc,
                        Err(err) => return Err(err),
                    }
                } else if tag_value == K_TAG_TRC && color_space == DataColorSpace::Rgb {
                    match Self::read_trc_tag(slice, tag_entry as usize, tag_size) {
                        Ok(trc) => profile.gray_trc = trc,
                        Err(err) => return Err(err),
                    }
                } else if tag_value == WT_PT_TAG {
                    match Self::read_xyz_tag(slice, tag_entry as usize, tag_size) {
                        Ok(wt) => profile.white_point = Some(wt),
                        Err(err) => return Err(err),
                    }
                } else if tag_value == BLACKPOINT_TAG {
                    match Self::read_xyz_tag(slice, tag_entry as usize, tag_size) {
                        Ok(wt) => profile.black_point = Some(wt),
                        Err(err) => return Err(err),
                    }
                } else if tag_value == CHAD_TAG {
                    profile.chromatic_adaptation =
                        Some(Self::read_chad_tag(slice, tag_entry as usize, tag_size)?);
                } else if tag_value == ATOB0_TAG
                    && (profile.color_space == DataColorSpace::Rgb
                        || profile.color_space == DataColorSpace::Cmyk)
                {
                    let lut_type = Self::read_lut_type(slice, tag_entry as usize, tag_size)?;
                    if lut_type == LutType::Lut8 || lut_type == LutType::Lut16 {
                        match Self::read_lut_a_to_b_type(slice, tag_entry as usize, tag_size) {
                            Ok(v) => profile.lut_a_to_b = v,
                            Err(err) => return Err(err),
                        }
                    }
                } else if tag_value == B2A0_TAG {
                    let lut_type = Self::read_lut_type(slice, tag_entry as usize, tag_size)?;
                    if lut_type == LutType::Lut8 || lut_type == LutType::Lut16 {
                        match Self::read_lut_a_to_b_type(slice, tag_entry as usize, tag_size) {
                            Ok(v) => profile.lut_b_to_a = v,
                            Err(err) => return Err(err),
                        }
                    }
                }
            }
        }

        Ok(profile)
    }
}

impl ColorProfile {
    #[inline]
    pub fn colorant_matrix(&self) -> Matrix3f {
        if let Some(cicp) = self.cicp {
            if let ColorPrimaries::Bt709 = cicp.color_primaries {
                return SRGB_MATRIX;
            } else if let ColorPrimaries::Bt2020 = cicp.color_primaries {
                return BT2020_MATRIX;
            } else if let ColorPrimaries::Smpte432 = cicp.color_primaries {
                return DISPLAY_P3_MATRIX;
            }
        }
        Matrix3f {
            v: [
                [
                    self.red_colorant.x,
                    self.green_colorant.x,
                    self.blue_colorant.x,
                ],
                [
                    self.red_colorant.y,
                    self.green_colorant.y,
                    self.blue_colorant.y,
                ],
                [
                    self.red_colorant.z,
                    self.green_colorant.z,
                    self.blue_colorant.z,
                ],
            ],
        }
    }

    pub(crate) fn update_rgb_colorimetry(
        &mut self,
        white_point: XyY,
        primaries: ChromacityTriple,
    ) -> bool {
        let red_xyz = primaries.red.to_xyz();
        let green_xyz = primaries.green.to_xyz();
        let blue_xyz = primaries.blue.to_xyz();
        let xyz_matrix = Matrix3f {
            v: [
                [red_xyz.x, green_xyz.x, blue_xyz.x],
                [red_xyz.y, green_xyz.y, blue_xyz.y],
                [red_xyz.z, green_xyz.z, blue_xyz.z],
            ],
        };
        let colorants = match self.rgb_to_xyz(xyz_matrix, white_point.to_xyz()) {
            None => return false,
            Some(v) => v,
        };
        let colorants = match adapt_to_d50(Some(colorants), white_point) {
            Some(colorants) => colorants,
            None => return false,
        };

        /* note: there's a transpose type of operation going on here */
        self.red_colorant.x = colorants.v[0][0];
        self.red_colorant.y = colorants.v[1][0];
        self.red_colorant.z = colorants.v[2][0];
        self.green_colorant.x = colorants.v[0][1];
        self.green_colorant.y = colorants.v[1][1];
        self.green_colorant.z = colorants.v[2][1];
        self.blue_colorant.x = colorants.v[0][2];
        self.blue_colorant.y = colorants.v[1][2];
        self.blue_colorant.z = colorants.v[2][2];
        true
    }

    pub fn update_rgb_colorimetry_from_cicp(&mut self, cicp: CicpProfile) -> bool {
        self.cicp = Some(cicp);
        if !cicp.color_primaries.has_chromacity()
            || !cicp.transfer_characteristics.has_transfer_curve()
        {
            return false;
        }
        let primaries_xy: ChromacityTriple = match cicp.color_primaries.try_into() {
            Ok(primaries) => primaries,
            Err(_) => return false,
        };
        let white_point: Chromacity = match cicp.color_primaries.white_point() {
            Ok(v) => v,
            Err(_) => return false,
        };
        if !self.update_rgb_colorimetry(white_point.to_xyyb(), primaries_xy) {
            return false;
        }

        let red_trc: Trc = match cicp.transfer_characteristics.try_into() {
            Ok(trc) => trc,
            Err(_) => return false,
        };
        self.green_trc = Some(red_trc.clone());
        self.blue_trc = Some(red_trc.clone());
        self.red_trc = Some(red_trc);
        false
    }

    pub fn rgb_to_xyz(&self, xyz_matrix: Matrix3f, wp: Xyz) -> Option<Matrix3f> {
        let xyz_inverse = xyz_matrix.inverse()?;
        let s = xyz_inverse.mul_vector(wp.to_vector());
        let mut v = xyz_matrix.mul_row_vector::<0>(s);
        v = v.mul_row_vector::<1>(s);
        v = v.mul_row_vector::<2>(s);
        Some(v)
    }

    pub fn rgb_to_xyz_matrix(&self) -> Option<Matrix3f> {
        let xyz_matrix = self.colorant_matrix();
        let white_point = self.white_point.unwrap_or(Chromacity::D50.to_xyz());
        self.rgb_to_xyz(xyz_matrix, white_point)
    }

    /// Computes transform matrix RGB -> XYZ -> RGB
    /// Current profile is used as source, other as destination
    pub fn transform_matrix(&self, dest: &ColorProfile) -> Option<Matrix3f> {
        let source = self.rgb_to_xyz_matrix()?;
        let dst = dest.rgb_to_xyz_matrix()?;
        let dest_inverse = dst.inverse()?;
        Some(dest_inverse.mat_mul(source))
    }
}

/* from lcms: cmsWhitePointFromTemp */
/* tempK must be >= 4000. and <= 25000.
 * Invalid values of tempK will return
 * (x,y,Y) = (-1.0, -1.0, -1.0)
 * similar to argyll: icx_DTEMP2XYZ() */
const fn white_point_from_temperature(temp_k: i32) -> XyY {
    let mut white_point = XyY {
        x: 0f32,
        y: 0f32,
        yb: 0f32,
    };
    // No optimization provided.
    let temp_k = temp_k as f64; // Square
    let temp_k2 = temp_k * temp_k; // Cube
    let temp_k3 = temp_k2 * temp_k;
    // For correlated color temperature (T) between 4000K and 7000K:
    let x = if temp_k > 4000.0 && temp_k <= 7000.0 {
        -4.6070 * (1E9 / temp_k3) + 2.9678 * (1E6 / temp_k2) + 0.09911 * (1E3 / temp_k) + 0.244063
    } else if temp_k > 7000.0 && temp_k <= 25000.0 {
        -2.0064 * (1E9 / temp_k3) + 1.9018 * (1E6 / temp_k2) + 0.24748 * (1E3 / temp_k) + 0.237040
    } else {
        // or for correlated color temperature (T) between 7000K and 25000K:
        // Invalid tempK
        white_point.x = -1.0;
        white_point.y = -1.0;
        white_point.yb = -1.0;
        debug_assert!(false, "invalid temp");
        return white_point;
    };
    // Obtain y(x)
    let y = -3.000 * (x * x) + 2.870 * x - 0.275;
    // wave factors (not used, but here for futures extensions)
    // let M1 = (-1.3515 - 1.7703*x + 5.9114 *y)/(0.0241 + 0.2562*x - 0.7341*y);
    // let M2 = (0.0300 - 31.4424*x + 30.0717*y)/(0.0241 + 0.2562*x - 0.7341*y);
    // Fill white_point struct
    white_point.x = x as f32;
    white_point.y = y as f32;
    white_point.yb = 1.0;
    white_point
}

pub(crate) const fn white_point_srgb() -> XyY {
    white_point_from_temperature(6504)
}

impl ColorProfile {
    pub fn new_srgb() -> ColorProfile {
        let primaries = ChromacityTriple::try_from(ColorPrimaries::Bt709).unwrap();
        let white_point = white_point_srgb();
        let mut profile = ColorProfile::default();
        profile.update_rgb_colorimetry(white_point, primaries);

        let curve = Trc::Parametric(vec![2.4, 1. / 1.055, 0.055 / 1.055, 1. / 12.92, 0.04045]);
        profile.red_trc = Some(curve.clone());
        profile.blue_trc = Some(curve.clone());
        profile.green_trc = Some(curve);
        profile.profile_class = ProfileClass::DisplayDevice;
        profile.rendering_intent = RenderingIntent::Perceptual;
        profile.color_space = DataColorSpace::Rgb;
        profile.pcs = DataColorSpace::Xyz;
        profile.image_white_point = Chromacity::D65.to_xyz();
        profile.white_point = Some(Chromacity::D50.to_xyz());
        profile
    }

    pub fn new_display_p3() -> ColorProfile {
        let primaries = ChromacityTriple::try_from(ColorPrimaries::Smpte432).unwrap();
        let white_point = white_point_srgb();
        let mut profile = ColorProfile::default();
        profile.update_rgb_colorimetry(white_point, primaries);

        let curve = Trc::Parametric(vec![2.4, 1. / 1.055, 0.055 / 1.055, 1. / 12.92, 0.04045]);
        profile.red_trc = Some(curve.clone());
        profile.blue_trc = Some(curve.clone());
        profile.green_trc = Some(curve);
        profile.profile_class = ProfileClass::DisplayDevice;
        profile.rendering_intent = RenderingIntent::Perceptual;
        profile.color_space = DataColorSpace::Rgb;
        profile.pcs = DataColorSpace::Xyz;
        profile.image_white_point = Chromacity::D65.to_xyz();
        profile.white_point = Some(Chromacity::D50.to_xyz());
        profile
    }

    pub fn new_bt2020() -> ColorProfile {
        let primaries = ChromacityTriple::try_from(ColorPrimaries::Bt2020).unwrap();
        let white_point = white_point_srgb();
        let mut profile = ColorProfile::default();
        profile.update_rgb_colorimetry(white_point, primaries);

        let curve = Trc::Parametric(vec![2.4, 1. / 1.055, 0.055 / 1.055, 1. / 12.92, 0.04045]);
        profile.red_trc = Some(curve.clone());
        profile.blue_trc = Some(curve.clone());
        profile.green_trc = Some(curve);
        profile.profile_class = ProfileClass::DisplayDevice;
        profile.rendering_intent = RenderingIntent::Perceptual;
        profile.color_space = DataColorSpace::Rgb;
        profile.pcs = DataColorSpace::Xyz;
        profile.image_white_point = Chromacity::D65.to_xyz();
        profile.white_point = Some(Chromacity::D50.to_xyz());
        profile
    }

    pub fn new_gray_with_gamma(gamma: f32) -> ColorProfile {
        ColorProfile {
            gray_trc: Some(curve_from_gamma(gamma)),
            profile_class: ProfileClass::DisplayDevice,
            rendering_intent: RenderingIntent::Perceptual,
            color_space: DataColorSpace::Gray,
            image_white_point: Chromacity::D65.to_xyz(),
            white_point: Some(Chromacity::D50.to_xyz()),
            ..Default::default()
        }
    }
}
