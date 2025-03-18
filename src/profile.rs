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
use crate::chad::adapt_to_d50;
use crate::cicp::{
    CicpColorPrimaries, ColorPrimaries, MatrixCoefficients, TransferCharacteristics,
};
use crate::dat::ColorDateTime;
use crate::err::CmsError;
use crate::matrix::{BT2020_MATRIX, DISPLAY_P3_MATRIX, Matrix3f, SRGB_MATRIX, XyY, Xyz};
use crate::safe_reader::{SafeAdd, SafeMul};
use crate::tag::{TAG_SIZE, Tag, TagTypeDefinition};
use crate::trc::ToneReprCurve;
use crate::{Chromaticity, Vector3f};
use std::io::Read;

const MAX_PROFILE_SIZE: usize = 1024 * 1024 * 10; // 10 MB max, for Fogra39 etc

#[inline]
fn uint8_number_to_float(a: u8) -> f32 {
    a as f32 / 255.0
}

fn utf16be_to_utf16(slice: &[u8]) -> Vec<u16> {
    let mut vec = vec![0u16; slice.len() / 2];
    for (dst, chunk) in vec.iter_mut().zip(slice.chunks_exact(2)) {
        *dst = u16::from_be_bytes([chunk[0], chunk[1]]);
    }
    vec
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileSignature {
    Acsp,
}

impl TryFrom<u32> for ProfileSignature {
    type Error = CmsError;
    #[inline]
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        if value == u32::from_ne_bytes(*b"acsp").to_be() {
            return Ok(ProfileSignature::Acsp);
        }
        Err(CmsError::InvalidProfile)
    }
}

impl From<ProfileSignature> for u32 {
    #[inline]
    fn from(value: ProfileSignature) -> Self {
        match value {
            ProfileSignature::Acsp => u32::from_ne_bytes(*b"acsp").to_be(),
        }
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Ord, PartialOrd)]
pub enum ProfileVersion {
    V2_0 = 0x02000000,
    V2_1 = 0x02100000,
    V2_2 = 0x02200000,
    V2_3 = 0x02300000,
    V2_4 = 0x02400000,
    V4_0 = 0x04000000,
    V4_1 = 0x04100000,
    V4_2 = 0x04200000,
    V4_3 = 0x04300000,
    #[default]
    V4_4 = 0x04400000,
    Unknown,
}

impl TryFrom<u32> for ProfileVersion {
    type Error = CmsError;
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0x02000000 => Ok(ProfileVersion::V2_0),
            0x02100000 => Ok(ProfileVersion::V2_1),
            0x02200000 => Ok(ProfileVersion::V2_2),
            0x02300000 => Ok(ProfileVersion::V2_3),
            0x02400000 => Ok(ProfileVersion::V2_4),
            0x04000000 => Ok(ProfileVersion::V4_0),
            0x04100000 => Ok(ProfileVersion::V4_1),
            0x04200000 => Ok(ProfileVersion::V4_2),
            0x04300000 => Ok(ProfileVersion::V4_3),
            0x04400000 => Ok(ProfileVersion::V4_3),
            _ => Err(CmsError::InvalidProfile),
        }
    }
}

impl From<ProfileVersion> for u32 {
    fn from(value: ProfileVersion) -> Self {
        match value {
            ProfileVersion::V2_0 => 0x02000000,
            ProfileVersion::V2_1 => 0x02100000,
            ProfileVersion::V2_2 => 0x02200000,
            ProfileVersion::V2_3 => 0x02300000,
            ProfileVersion::V2_4 => 0x02400000,
            ProfileVersion::V4_0 => 0x04000000,
            ProfileVersion::V4_1 => 0x04100000,
            ProfileVersion::V4_2 => 0x04200000,
            ProfileVersion::V4_3 => 0x04300000,
            ProfileVersion::V4_4 => 0x04400000,
            ProfileVersion::Unknown => 0x02000000,
        }
    }
}

#[repr(u32)]
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

#[repr(u32)]
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
        Err(CmsError::InvalidProfile)
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
        Err(CmsError::InvalidProfile)
    }
}

impl From<LutType> for u32 {
    fn from(val: LutType) -> Self {
        match val {
            LutType::Lut8 => u32::from_ne_bytes(*b"mft1").to_be(),
            LutType::Lut16 => u32::from_ne_bytes(*b"mft2").to_be(),
            LutType::LutMab => u32::from_ne_bytes(*b"mAB ").to_be(),
            LutType::LutMba => u32::from_ne_bytes(*b"mBA ").to_be(),
        }
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
        Err(CmsError::InvalidProfile)
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

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub enum TechnologySignatures {
    FilmScanner,
    DigitalCamera,
    ReflectiveScanner,
    InkJetPrinter,
    ThermalWaxPrinter,
    ElectrophotographicPrinter,
    ElectrostaticPrinter,
    DyeSublimationPrinter,
    PhotographicPaperPrinter,
    FilmWriter,
    VideoMonitor,
    VideoCamera,
    ProjectionTelevision,
    CathodeRayTubeDisplay,
    PassiveMatrixDisplay,
    ActiveMatrixDisplay,
    LiquidCrystalDisplay,
    OrganicLedDisplay,
    PhotoCd,
    PhotographicImageSetter,
    Gravure,
    OffsetLithography,
    Silkscreen,
    Flexography,
    MotionPictureFilmScanner,
    MotionPictureFilmRecorder,
    DigitalMotionPictureCamera,
    DigitalCinemaProjector,
    Unknown(u32),
}

impl From<u32> for TechnologySignatures {
    fn from(value: u32) -> Self {
        if value == u32::from_ne_bytes(*b"fscn").to_be() {
            return TechnologySignatures::FilmScanner;
        } else if value == u32::from_ne_bytes(*b"dcam").to_be() {
            return TechnologySignatures::DigitalCamera;
        } else if value == u32::from_ne_bytes(*b"rscn").to_be() {
            return TechnologySignatures::ReflectiveScanner;
        } else if value == u32::from_ne_bytes(*b"ijet").to_be() {
            return TechnologySignatures::InkJetPrinter;
        } else if value == u32::from_ne_bytes(*b"twax").to_be() {
            return TechnologySignatures::ThermalWaxPrinter;
        } else if value == u32::from_ne_bytes(*b"epho").to_be() {
            return TechnologySignatures::ElectrophotographicPrinter;
        } else if value == u32::from_ne_bytes(*b"esta").to_be() {
            return TechnologySignatures::ElectrostaticPrinter;
        } else if value == u32::from_ne_bytes(*b"dsub").to_be() {
            return TechnologySignatures::DyeSublimationPrinter;
        } else if value == u32::from_ne_bytes(*b"rpho").to_be() {
            return TechnologySignatures::PhotographicPaperPrinter;
        } else if value == u32::from_ne_bytes(*b"fprn").to_be() {
            return TechnologySignatures::FilmWriter;
        } else if value == u32::from_ne_bytes(*b"vidm").to_be() {
            return TechnologySignatures::VideoMonitor;
        } else if value == u32::from_ne_bytes(*b"vidc").to_be() {
            return TechnologySignatures::VideoCamera;
        } else if value == u32::from_ne_bytes(*b"pjtv").to_be() {
            return TechnologySignatures::ProjectionTelevision;
        } else if value == u32::from_ne_bytes(*b"CRT ").to_be() {
            return TechnologySignatures::CathodeRayTubeDisplay;
        } else if value == u32::from_ne_bytes(*b"PMD ").to_be() {
            return TechnologySignatures::PassiveMatrixDisplay;
        } else if value == u32::from_ne_bytes(*b"AMD ").to_be() {
            return TechnologySignatures::ActiveMatrixDisplay;
        } else if value == u32::from_ne_bytes(*b"LCD ").to_be() {
            return TechnologySignatures::LiquidCrystalDisplay;
        } else if value == u32::from_ne_bytes(*b"OLED").to_be() {
            return TechnologySignatures::OrganicLedDisplay;
        } else if value == u32::from_ne_bytes(*b"KPCD").to_be() {
            return TechnologySignatures::PhotoCd;
        } else if value == u32::from_ne_bytes(*b"imgs").to_be() {
            return TechnologySignatures::PhotographicImageSetter;
        } else if value == u32::from_ne_bytes(*b"grav").to_be() {
            return TechnologySignatures::Gravure;
        } else if value == u32::from_ne_bytes(*b"offs").to_be() {
            return TechnologySignatures::OffsetLithography;
        } else if value == u32::from_ne_bytes(*b"silk").to_be() {
            return TechnologySignatures::Silkscreen;
        } else if value == u32::from_ne_bytes(*b"flex").to_be() {
            return TechnologySignatures::Flexography;
        } else if value == u32::from_ne_bytes(*b"mpfs").to_be() {
            return TechnologySignatures::MotionPictureFilmScanner;
        } else if value == u32::from_ne_bytes(*b"mpfr").to_be() {
            return TechnologySignatures::MotionPictureFilmRecorder;
        } else if value == u32::from_ne_bytes(*b"dmpc").to_be() {
            return TechnologySignatures::DigitalMotionPictureCamera;
        } else if value == u32::from_ne_bytes(*b"dcpj").to_be() {
            return TechnologySignatures::DigitalCinemaProjector;
        }
        TechnologySignatures::Unknown(value)
    }
}

#[derive(Debug, Clone)]
pub enum LutWarehouse {
    Lut(LutDataType),
    MCurves(LutMCurvesType),
}

impl LutWarehouse {
    pub(crate) fn as_lut(&self) -> Option<&LutDataType> {
        match self {
            LutWarehouse::Lut(lut) => Some(lut),
            LutWarehouse::MCurves(_) => None,
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

#[derive(Debug, Clone)]
pub struct LutMCurvesType {
    pub num_input_channels: u8,
    pub num_output_channels: u8,
    pub grid_points: [u8; 16],
    pub clut: Vec<f32>,
    pub a_curves: Vec<ToneReprCurve>,
    pub b_curves: Vec<ToneReprCurve>,
    pub m_curves: Vec<ToneReprCurve>,
    pub matrix: Matrix3f,
    pub bias: Vector3f,
}

#[repr(u32)]
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

    #[inline]
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

impl From<RenderingIntent> for u32 {
    #[inline]
    fn from(value: RenderingIntent) -> Self {
        match value {
            RenderingIntent::AbsoluteColorimetric => 3,
            RenderingIntent::Saturation => 2,
            RenderingIntent::RelativeColorimetric => 1,
            RenderingIntent::Perceptual => 0,
        }
    }
}

/// ICC Header
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct ProfileHeader {
    pub size: u32,                         // Size of the profile (computed)
    pub cmm_type: u32,                     // Preferred CMM type (ignored)
    pub version: ProfileVersion,           // Version (4.3 or 4.4 if CICP is included)
    pub profile_class: ProfileClass,       // Display device profile
    pub data_color_space: DataColorSpace,  // RGB input color space
    pub pcs: DataColorSpace,               // Profile connection space
    pub creation_date_time: ColorDateTime, // Date and time
    pub signature: ProfileSignature,       // Profile signature
    pub platform: u32,                     // Platform target (ignored)
    pub flags: u32,                        // Flags (not embedded, can be used independently)
    pub device_manufacturer: u32,          // Device manufacturer (ignored)
    pub device_model: u32,                 // Device model (ignored)
    pub device_attributes: [u8; 8],        // Device attributes (ignored)
    pub rendering_intent: RenderingIntent, // Relative colorimetric rendering intent
    pub illuminant: Xyz,                   // D50 standard illuminant X
    pub creator: u32,                      // Profile creator (ignored)
    pub profile_id: [u8; 16],              // Profile id checksum (ignored)
    pub reserved: [u8; 28],                // Reserved (ignored)
    pub tag_count: u32,                    // Technically not part of header, but required
}

impl ProfileHeader {
    #[allow(dead_code)]
    pub(crate) fn new(size: u32) -> Self {
        Self {
            size,
            cmm_type: 0,
            version: ProfileVersion::V4_3,
            profile_class: ProfileClass::DisplayDevice,
            data_color_space: DataColorSpace::Rgb,
            pcs: DataColorSpace::Xyz,
            creation_date_time: ColorDateTime::default(),
            signature: ProfileSignature::Acsp,
            platform: 0,
            flags: 0x00000000,
            device_manufacturer: 0,
            device_model: 0,
            device_attributes: [0; 8],
            rendering_intent: RenderingIntent::Perceptual,
            illuminant: Chromaticity::D50.to_xyz(),
            creator: 0,
            profile_id: [0; 16],
            reserved: [0; 28],
            tag_count: 0,
        }
    }

    /// Creates profile from the buffer
    pub(crate) fn new_from_slice(slice: &[u8]) -> Result<Self, CmsError> {
        if slice.len() < size_of::<ProfileHeader>() {
            return Err(CmsError::InvalidProfile);
        }
        let mut cursor = std::io::Cursor::new(slice);
        let mut buffer = [0u8; size_of::<ProfileHeader>()];
        cursor
            .read_exact(&mut buffer)
            .map_err(|_| CmsError::InvalidProfile)?;

        let header = Self {
            size: u32::from_be_bytes(buffer[0..4].try_into().unwrap()),
            cmm_type: u32::from_be_bytes(buffer[4..8].try_into().unwrap()),
            version: ProfileVersion::try_from(u32::from_be_bytes(
                buffer[8..12].try_into().unwrap(),
            ))?,
            profile_class: ProfileClass::try_from(u32::from_be_bytes(
                buffer[12..16].try_into().unwrap(),
            ))?,
            data_color_space: DataColorSpace::try_from(u32::from_be_bytes(
                buffer[16..20].try_into().unwrap(),
            ))?,
            pcs: DataColorSpace::try_from(u32::from_be_bytes(buffer[20..24].try_into().unwrap()))?,
            creation_date_time: ColorDateTime::new_from_slice(buffer[24..36].try_into().unwrap())?,
            signature: ProfileSignature::try_from(u32::from_be_bytes(
                buffer[36..40].try_into().unwrap(),
            ))?,
            platform: u32::from_be_bytes(buffer[40..44].try_into().unwrap()),
            flags: u32::from_be_bytes(buffer[44..48].try_into().unwrap()),
            device_manufacturer: u32::from_be_bytes(buffer[48..52].try_into().unwrap()),
            device_model: u32::from_be_bytes(buffer[52..56].try_into().unwrap()),
            device_attributes: buffer[56..64].try_into().unwrap(),
            rendering_intent: RenderingIntent::try_from(u32::from_be_bytes(
                buffer[64..68].try_into().unwrap(),
            ))?,
            illuminant: Xyz::new(
                s15_fixed16_number_to_float(i32::from_be_bytes(buffer[68..72].try_into().unwrap())),
                s15_fixed16_number_to_float(i32::from_be_bytes(buffer[72..76].try_into().unwrap())),
                s15_fixed16_number_to_float(i32::from_be_bytes(buffer[76..80].try_into().unwrap())),
            ),
            creator: u32::from_be_bytes(buffer[80..84].try_into().unwrap()),
            profile_id: buffer[84..100].try_into().unwrap(),
            reserved: buffer[100..128].try_into().unwrap(),
            tag_count: u32::from_be_bytes(buffer[128..132].try_into().unwrap()),
        };
        Ok(header)
    }
}

/// A [Coding Independent Code Point](https://en.wikipedia.org/wiki/Coding-independent_code_points).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CicpProfile {
    pub color_primaries: CicpColorPrimaries,
    pub transfer_characteristics: TransferCharacteristics,
    pub matrix_coefficients: MatrixCoefficients,
    pub full_range: bool,
}

#[derive(Debug, Clone)]
pub struct LocalizableString {
    /// An ISO 639-1 value is expected; any text w. more than two symbols will be truncated
    pub language: String,
    /// An ISO 3166-1 value is expected; any text w. more than two symbols will be truncated
    pub country: String,
    pub value: String,
}

impl LocalizableString {
    /// Creates new localizable string
    ///
    /// # Arguments
    ///
    /// * `language`: an ISO 639-1 value is expected, any text more than 2 symbols will be truncated
    /// * `country`: an ISO 3166-1 value is expected, any text more than 2 symbols will be truncated
    /// * `value`: String value
    ///
    pub fn new(language: String, country: String, value: String) -> Self {
        Self {
            language,
            country,
            value,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DescriptionString {
    pub ascii_string: String,
    pub unicode_language_code: u32,
    pub unicode_string: String,
    pub script_code_code: i8,
    pub mac_string: String,
}

#[derive(Debug, Clone)]
pub enum ProfileText {
    PlainString(String),
    Localizable(Vec<LocalizableString>),
    Description(DescriptionString),
}

impl ProfileText {
    pub(crate) fn has_values(&self) -> bool {
        match self {
            ProfileText::PlainString(_) => true,
            ProfileText::Localizable(lc) => !lc.is_empty(),
            ProfileText::Description(_) => true,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum StandardObserver {
    D50,
    D65,
    Unknown,
}

impl From<u32> for StandardObserver {
    fn from(value: u32) -> Self {
        if value == 1 {
            return StandardObserver::D50;
        } else if value == 2 {
            return StandardObserver::D65;
        }
        StandardObserver::Unknown
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ViewingConditions {
    pub illuminant: Xyz,
    pub surround: Xyz,
    pub observer: StandardObserver,
}

#[derive(Debug, Clone, Copy)]
pub enum MeasurementGeometry {
    Unknown,
    /// 0°:45° or 45°:0°
    D45to45,
    /// 0°:d or d:0°
    D0to0,
}

impl From<u32> for MeasurementGeometry {
    fn from(value: u32) -> Self {
        if value == 1 {
            Self::D45to45
        } else if value == 2 {
            Self::D0to0
        } else {
            Self::Unknown
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum StandardIlluminant {
    Unknown,
    D50,
    D65,
    D93,
    F2,
    D55,
    A,
    EquiPower,
    F8,
}

impl From<u32> for StandardIlluminant {
    fn from(value: u32) -> Self {
        match value {
            1 => StandardIlluminant::D50,
            2 => StandardIlluminant::D65,
            3 => StandardIlluminant::D93,
            4 => StandardIlluminant::F2,
            5 => StandardIlluminant::D55,
            6 => StandardIlluminant::A,
            7 => StandardIlluminant::EquiPower,
            8 => StandardIlluminant::F8,
            _ => Self::Unknown,
        }
    }
}

impl From<StandardIlluminant> for u32 {
    fn from(value: StandardIlluminant) -> Self {
        match value {
            StandardIlluminant::Unknown => 0u32,
            StandardIlluminant::D50 => 1u32,
            StandardIlluminant::D65 => 2u32,
            StandardIlluminant::D93 => 3,
            StandardIlluminant::F2 => 4,
            StandardIlluminant::D55 => 5,
            StandardIlluminant::A => 6,
            StandardIlluminant::EquiPower => 7,
            StandardIlluminant::F8 => 8,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Measurement {
    pub observer: StandardObserver,
    pub backing: Xyz,
    pub geometry: MeasurementGeometry,
    pub flare: f32,
    pub illuminant: StandardIlluminant,
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
    pub white_point: Xyz,
    pub black_point: Option<Xyz>,
    pub media_white_point: Option<Xyz>,
    pub luminance: Option<Xyz>,
    pub measurement: Option<Measurement>,
    pub red_trc: Option<ToneReprCurve>,
    pub green_trc: Option<ToneReprCurve>,
    pub blue_trc: Option<ToneReprCurve>,
    pub gray_trc: Option<ToneReprCurve>,
    pub cicp: Option<CicpProfile>,
    pub chromatic_adaptation: Option<Matrix3f>,
    pub lut_a_to_b_perceptual: Option<LutWarehouse>,
    pub lut_a_to_b_colorimetric: Option<LutWarehouse>,
    pub lut_a_to_b_saturation: Option<LutWarehouse>,
    pub lut_b_to_a_perceptual: Option<LutWarehouse>,
    pub lut_b_to_a_colorimetric: Option<LutWarehouse>,
    pub lut_b_to_a_saturation: Option<LutWarehouse>,
    pub gamut: Option<LutWarehouse>,
    pub copyright: Option<ProfileText>,
    pub description: Option<ProfileText>,
    pub device_manufacturer: Option<ProfileText>,
    pub device_model: Option<ProfileText>,
    pub char_target: Option<ProfileText>,
    pub viewing_conditions: Option<ViewingConditions>,
    pub viewing_conditions_description: Option<ProfileText>,
    pub technology: Option<TechnologySignatures>,
    pub calibration_date: Option<ColorDateTime>,
    /// Version for internal and viewing purposes only.
    /// When encoding will be added profile will always be encoded as V4.
    pub(crate) version_internal: ProfileVersion,
}

/// Produces the nearest float to `a` with a maximum error of 1/1024 which
/// happens for large values like 0x40000040.
#[inline]
pub(crate) const fn s15_fixed16_number_to_float(a: i32) -> f32 {
    a as f32 / 65536.
}

#[inline]
const fn uint16_number_to_float(a: u32) -> f32 {
    a as f32 / 65536.
}

impl ColorProfile {
    /// Returns profile version
    pub fn version(&self) -> ProfileVersion {
        self.version_internal
    }

    fn read_trc_tag_s(
        slice: &[u8],
        entry: usize,
        tag_size: usize,
    ) -> Result<Option<ToneReprCurve>, CmsError> {
        let mut _empty = 0usize;
        Self::read_trc_tag(slice, entry, tag_size, &mut _empty)
    }

    fn read_trc_tag(
        slice: &[u8],
        entry: usize,
        tag_size: usize,
        read_size: &mut usize,
    ) -> Result<Option<ToneReprCurve>, CmsError> {
        if slice.len() < entry.safe_add(4)? {
            return Ok(None);
        }
        let small_tag = &slice[entry..entry + 4];
        // We require always recognize tone curves.
        let curve_type = TagTypeDefinition::from(u32::from_be_bytes([
            small_tag[0],
            small_tag[1],
            small_tag[2],
            small_tag[3],
        ]));
        if tag_size != 0 && tag_size < TAG_SIZE {
            return Ok(None);
        }
        let last_tag_offset = if tag_size != 0 {
            tag_size + entry
        } else {
            slice.len()
        };
        if last_tag_offset > slice.len() {
            return Err(CmsError::InvalidProfile);
        }
        let tag = &slice[entry..last_tag_offset];
        if tag.len() < TAG_SIZE {
            return Err(CmsError::InvalidProfile);
        }
        if curve_type == TagTypeDefinition::LutToneCurve {
            let entry_count = u32::from_be_bytes([tag[8], tag[9], tag[10], tag[11]]) as usize;
            if entry_count == 0 {
                return Ok(Some(ToneReprCurve::Lut(vec![])));
            }
            if entry_count > 40000 {
                return Err(CmsError::CurveLutIsTooLarge);
            }
            let curve_end = entry_count.safe_mul(size_of::<u16>())?.safe_add(12)?;
            if tag.len() < curve_end {
                return Err(CmsError::InvalidProfile);
            }
            let curve_sliced = &tag[12..curve_end];
            let mut curve_values = vec![0u16; entry_count];
            for (value, curve_value) in curve_sliced.chunks_exact(2).zip(curve_values.iter_mut()) {
                let gamma_s15 = u16::from_be_bytes([value[0], value[1]]);
                *curve_value = gamma_s15;
            }
            *read_size = curve_end;
            Ok(Some(ToneReprCurve::Lut(curve_values)))
        } else if curve_type == TagTypeDefinition::ParametricToneCurve {
            let entry_count = u16::from_be_bytes([tag[8], tag[9]]) as usize;
            if entry_count > 4 {
                return Err(CmsError::InvalidProfile);
            }

            const COUNT_TO_LENGTH: [usize; 5] = [1, 3, 4, 5, 7]; //PARAMETRIC_CURVE_TYPE

            if tag.len() < 12 + COUNT_TO_LENGTH[entry_count] * size_of::<u32>() {
                return Err(CmsError::InvalidProfile);
            }
            let curve_sliced = &tag[12..12 + COUNT_TO_LENGTH[entry_count] * size_of::<u32>()];
            let mut params = vec![0f32; COUNT_TO_LENGTH[entry_count]];
            for (value, param_value) in curve_sliced.chunks_exact(4).zip(params.iter_mut()) {
                let parametric_value = i32::from_be_bytes([value[0], value[1], value[2], value[3]]);
                *param_value = s15_fixed16_number_to_float(parametric_value);
            }
            if entry_count == 1 || entry_count == 2 {
                // we have a type 1 or type 2 function that has a division by `a`
                let a: f32 = params[1];
                if a == 0.0 {
                    return Err(CmsError::ParametricCurveZeroDivision);
                }
            }
            *read_size = 12 + COUNT_TO_LENGTH[entry_count] * 4;
            return Ok(Some(ToneReprCurve::Parametric(params)));
        } else {
            return Err(CmsError::InvalidProfile);
        }
    }

    #[inline]
    fn read_chad_tag(
        slice: &[u8],
        entry: usize,
        tag_size: usize,
    ) -> Result<Option<Matrix3f>, CmsError> {
        let last_tag_offset = tag_size.safe_add(entry)?;
        if last_tag_offset > slice.len() {
            return Err(CmsError::InvalidProfile);
        }
        if slice[entry..].len() < 8 {
            return Err(CmsError::InvalidProfile);
        }
        if tag_size < 8 {
            return Ok(None);
        }
        if (tag_size - 8) / 4 != 9 {
            return Ok(None);
        }
        let tag0 = &slice[entry..entry.safe_add(8)?];
        let c_type =
            TagTypeDefinition::from(u32::from_be_bytes([tag0[0], tag0[1], tag0[2], tag0[3]]));
        if c_type != TagTypeDefinition::S15Fixed16Array {
            return Err(CmsError::InvalidProfile);
        }
        if slice.len() < 9 * size_of::<u32>() + 8 {
            return Err(CmsError::InvalidProfile);
        }
        let tag = &slice[entry + 8..last_tag_offset];
        if tag.len() != size_of::<Matrix3f>() {
            return Err(CmsError::InvalidProfile);
        }
        let mut matrix = Matrix3f::default();
        for (i, chunk) in tag.chunks_exact(4).enumerate() {
            let q15_16_x = i32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            matrix.v[i / 3][i % 3] = s15_fixed16_number_to_float(q15_16_x);
        }
        Ok(Some(matrix))
    }

    #[inline]
    fn read_tech_tag(
        slice: &[u8],
        entry: usize,
        tag_size: usize,
    ) -> Result<Option<TechnologySignatures>, CmsError> {
        if tag_size < TAG_SIZE {
            return Err(CmsError::InvalidProfile);
        }
        let last_tag_offset = tag_size.safe_add(entry)?;
        if last_tag_offset > slice.len() {
            return Err(CmsError::InvalidProfile);
        }
        let tag = &slice[entry..entry.safe_add(12)?];
        let tag_type = u32::from_be_bytes([tag[0], tag[1], tag[2], tag[3]]);
        let def = TagTypeDefinition::from(tag_type);
        if def == TagTypeDefinition::Signature {
            let sig = u32::from_be_bytes([tag[8], tag[9], tag[10], tag[11]]);
            let tech_sig = TechnologySignatures::from(sig);
            return Ok(Some(tech_sig));
        }
        Ok(None)
    }

    #[inline]
    fn read_date_time_tag(
        slice: &[u8],
        entry: usize,
        tag_size: usize,
    ) -> Result<Option<ColorDateTime>, CmsError> {
        if tag_size < 20 {
            return Ok(None);
        }
        let last_tag_offset = tag_size.safe_add(entry)?;
        if last_tag_offset > slice.len() {
            return Err(CmsError::InvalidProfile);
        }
        let tag = &slice[entry..entry.safe_add(20)?];
        let tag_type = u32::from_be_bytes([tag[0], tag[1], tag[2], tag[3]]);
        let def = TagTypeDefinition::from(tag_type);
        if def == TagTypeDefinition::DateTime {
            let tag_value = &slice[8..20];
            let time = ColorDateTime::new_from_slice(tag_value)?;
            return Ok(Some(time));
        }
        Ok(None)
    }

    #[inline]
    fn read_meas_tag(
        slice: &[u8],
        entry: usize,
        tag_size: usize,
    ) -> Result<Option<Measurement>, CmsError> {
        if tag_size < TAG_SIZE {
            return Ok(None);
        }
        let last_tag_offset = tag_size.safe_add(entry)?;
        if last_tag_offset > slice.len() {
            return Err(CmsError::InvalidProfile);
        }
        let tag = &slice[entry..entry + 12];
        let tag_type = u32::from_be_bytes([tag[0], tag[1], tag[2], tag[3]]);
        let def = TagTypeDefinition::from(tag_type);
        if def != TagTypeDefinition::Measurement {
            return Ok(None);
        }
        if 36 > slice.len() {
            return Err(CmsError::InvalidProfile);
        }
        let tag = &slice[entry..entry + 36];
        let observer =
            StandardObserver::from(u32::from_be_bytes([tag[8], tag[9], tag[10], tag[11]]));
        let q15_16_x = i32::from_be_bytes([tag[12], tag[13], tag[14], tag[15]]);
        let q15_16_y = i32::from_be_bytes([tag[16], tag[17], tag[18], tag[19]]);
        let q15_16_z = i32::from_be_bytes([tag[20], tag[21], tag[22], tag[23]]);
        let x = s15_fixed16_number_to_float(q15_16_x);
        let y = s15_fixed16_number_to_float(q15_16_y);
        let z = s15_fixed16_number_to_float(q15_16_z);
        let xyz = Xyz::new(x, y, z);
        let geometry =
            MeasurementGeometry::from(u32::from_be_bytes([tag[24], tag[25], tag[26], tag[27]]));
        let flare =
            uint16_number_to_float(u32::from_be_bytes([tag[28], tag[29], tag[30], tag[31]]));
        let illuminant =
            StandardIlluminant::from(u32::from_be_bytes([tag[32], tag[33], tag[34], tag[35]]));
        Ok(Some(Measurement {
            flare,
            illuminant,
            geometry,
            observer,
            backing: xyz,
        }))
    }

    #[inline]
    fn read_xyz_tag(slice: &[u8], entry: usize, tag_size: usize) -> Result<Xyz, CmsError> {
        if tag_size < TAG_SIZE {
            return Ok(Xyz::default());
        }
        let last_tag_offset = tag_size.safe_add(entry)?;
        if last_tag_offset > slice.len() {
            return Err(CmsError::InvalidProfile);
        }
        let tag = &slice[entry..entry + 12];
        let tag_type = u32::from_be_bytes([tag[0], tag[1], tag[2], tag[3]]);
        let def = TagTypeDefinition::from(tag_type);
        if def != TagTypeDefinition::Xyz {
            return Ok(Xyz::default());
        }

        let tag = &slice[entry..last_tag_offset];
        if tag.len() < 20 {
            return Err(CmsError::InvalidProfile);
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
        let last_tag_offset = tag_size.safe_add(entry)?;
        if last_tag_offset > slice.len() {
            return Err(CmsError::InvalidProfile);
        }
        let tag = &slice[entry..last_tag_offset];
        if tag.len() < 12 {
            return Err(CmsError::InvalidProfile);
        }
        let tag_type = u32::from_be_bytes([tag[0], tag[1], tag[2], tag[3]]);
        let def = TagTypeDefinition::from(tag_type);
        if def != TagTypeDefinition::Cicp {
            return Ok(None);
        }
        let primaries = CicpColorPrimaries::try_from(tag[8])?;
        let transfer_characteristics = TransferCharacteristics::try_from(tag[9])?;
        let matrix_coefficients = MatrixCoefficients::try_from(tag[10])?;
        let full_range = tag[11] == 1;
        Ok(Some(CicpProfile {
            color_primaries: primaries,
            transfer_characteristics,
            matrix_coefficients,
            full_range,
        }))
    }

    #[inline]
    fn read_lut_type(slice: &[u8], entry: usize, tag_size: usize) -> Result<LutType, CmsError> {
        let tag_size = if tag_size == 0 { TAG_SIZE } else { tag_size };
        let last_tag_offset = tag_size.safe_add(entry)?;
        if last_tag_offset > slice.len() {
            return Err(CmsError::InvalidProfile);
        }
        let tag = &slice[entry..last_tag_offset];
        if tag.len() < 48 {
            return Err(CmsError::InvalidProfile);
        }
        let tag_type = u32::from_be_bytes([tag[0], tag[1], tag[2], tag[3]]);
        LutType::try_from(tag_type)
    }

    #[inline]
    fn read_viewing_conditions(
        slice: &[u8],
        entry: usize,
        tag_size: usize,
    ) -> Result<Option<ViewingConditions>, CmsError> {
        if tag_size < 36 {
            return Ok(None);
        }
        if slice.len() < entry.safe_add(36)? {
            return Err(CmsError::InvalidProfile);
        }
        let tag = &slice[entry..entry.safe_add(36)?];
        let tag_type =
            TagTypeDefinition::from(u32::from_be_bytes([tag[0], tag[1], tag[2], tag[3]]));
        // Ignore unknown
        if tag_type != TagTypeDefinition::DefViewingConditions {
            return Ok(None);
        }
        let illuminant_x = i32::from_be_bytes([tag[8], tag[9], tag[10], tag[11]]);
        let illuminant_y = i32::from_be_bytes([tag[12], tag[13], tag[14], tag[15]]);
        let illuminant_z = i32::from_be_bytes([tag[16], tag[17], tag[18], tag[19]]);

        let surround_x = i32::from_be_bytes([tag[20], tag[21], tag[22], tag[23]]);
        let surround_y = i32::from_be_bytes([tag[24], tag[25], tag[26], tag[27]]);
        let surround_z = i32::from_be_bytes([tag[28], tag[29], tag[30], tag[31]]);

        let illuminant_type = u32::from_be_bytes([tag[32], tag[33], tag[34], tag[35]]);

        let illuminant = Xyz::new(
            s15_fixed16_number_to_float(illuminant_x),
            s15_fixed16_number_to_float(illuminant_y),
            s15_fixed16_number_to_float(illuminant_z),
        );

        let surround = Xyz::new(
            s15_fixed16_number_to_float(surround_x),
            s15_fixed16_number_to_float(surround_y),
            s15_fixed16_number_to_float(surround_z),
        );

        let observer = StandardObserver::from(illuminant_type);

        Ok(Some(ViewingConditions {
            illuminant,
            surround,
            observer,
        }))
    }

    fn read_string_tag(
        slice: &[u8],
        entry: usize,
        tag_size: usize,
    ) -> Result<Option<ProfileText>, CmsError> {
        let tag_size = if tag_size == 0 { TAG_SIZE } else { tag_size };
        if tag_size < 4 {
            return Ok(None);
        }
        let last_tag_offset = tag_size.safe_add(entry)?;
        if last_tag_offset > slice.len() {
            return Err(CmsError::InvalidProfile);
        }
        let tag = &slice[entry..last_tag_offset];
        if tag.len() < 8 {
            return Ok(None);
        }
        let tag_type =
            TagTypeDefinition::from(u32::from_be_bytes([tag[0], tag[1], tag[2], tag[3]]));
        // Ignore unknown
        if tag_type == TagTypeDefinition::Text {
            let sliced_from_to_end = &tag[8..tag.len()];
            let str = String::from_utf8_lossy(sliced_from_to_end);
            return Ok(Some(ProfileText::PlainString(str.to_string())));
        } else if tag_type == TagTypeDefinition::MultiLocalizedUnicode {
            if tag.len() < 28 {
                return Err(CmsError::InvalidProfile);
            }
            // let record_size = u32::from_be_bytes([tag[12], tag[13], tag[14], tag[15]]) as usize;
            // // Record size is reserved to be 12.
            // if record_size != 12 {
            //     return Err(CmsError::InvalidIcc);
            // }
            let records_count = u32::from_be_bytes([tag[8], tag[9], tag[10], tag[11]]) as usize;
            let primary_language_code = String::from_utf8_lossy(&[tag[16], tag[17]]).to_string();
            let primary_country_code = String::from_utf8_lossy(&[tag[18], tag[19]]).to_string();
            let first_string_record_length =
                u32::from_be_bytes([tag[20], tag[21], tag[22], tag[23]]) as usize;
            let first_record_offset =
                u32::from_be_bytes([tag[24], tag[25], tag[26], tag[27]]) as usize;

            if tag.len() < first_record_offset.safe_add(first_string_record_length)? {
                return Ok(None);
            }

            let resliced =
                &tag[first_record_offset..first_record_offset + first_string_record_length];
            let cvt = utf16be_to_utf16(resliced);
            let string_record = String::from_utf16_lossy(&cvt);

            let mut records = vec![LocalizableString {
                language: primary_language_code,
                country: primary_country_code,
                value: string_record,
            }];

            for record in 1..records_count {
                // Localizable header must be at least 12 bytes
                let localizable_header_offset = if record == 1 {
                    28
                } else {
                    28 + 12 * (record - 1)
                };
                if tag.len() < localizable_header_offset + 12 {
                    return Err(CmsError::InvalidProfile);
                }
                let choked = &tag[localizable_header_offset..localizable_header_offset + 12];

                let language_code = String::from_utf8_lossy(&[choked[0], choked[1]]).to_string();
                let country_code = String::from_utf8_lossy(&[choked[2], choked[3]]).to_string();
                let record_length =
                    u32::from_be_bytes([choked[4], choked[5], choked[6], choked[7]]) as usize;
                let string_offset =
                    u32::from_be_bytes([choked[8], choked[9], choked[10], choked[11]]) as usize;

                if tag.len() < string_offset.safe_add(record_length)? {
                    return Ok(None);
                }
                let resliced = &tag[string_offset..string_offset + record_length];
                let cvt = utf16be_to_utf16(resliced);
                let string_record = String::from_utf16_lossy(&cvt);
                records.push(LocalizableString {
                    country: country_code,
                    language: language_code,
                    value: string_record,
                });
            }

            return Ok(Some(ProfileText::Localizable(records)));
        } else if tag_type == TagTypeDefinition::Description {
            if tag.len() < 12 {
                return Err(CmsError::InvalidProfile);
            }
            let ascii_length = u32::from_be_bytes([tag[8], tag[9], tag[10], tag[11]]) as usize;
            if tag.len() < 12.safe_add(ascii_length)? {
                return Err(CmsError::InvalidProfile);
            }
            let sliced = &tag[12..12 + ascii_length];
            let ascii_string = String::from_utf8_lossy(sliced).to_string();

            let mut last_position = 12 + ascii_length;
            if tag.len() < last_position + 8 {
                return Err(CmsError::InvalidProfile);
            }
            let uc = &tag[last_position..last_position + 8];
            let unicode_code = u32::from_be_bytes([uc[0], uc[1], uc[2], uc[3]]);
            let unicode_length = u32::from_be_bytes([uc[4], uc[5], uc[6], uc[7]]) as usize * 2;
            if tag.len() < unicode_length.safe_add(8)?.safe_add(last_position)? {
                return Ok(None);
            }

            last_position += 8;
            let uc = &tag[last_position..last_position + unicode_length];
            let wc = utf16be_to_utf16(uc);
            let unicode_string = String::from_utf16_lossy(&wc).to_string();

            // last_position += unicode_length;
            //
            // if tag.len() < last_position + 2 {
            //     return Err(CmsError::InvalidIcc);
            // }

            // let uc = &tag[last_position..last_position + 2];
            // let script_code = uc[0];
            // let mac_length = uc[1] as usize;
            // last_position += 2;
            // if tag.len() < last_position + mac_length {
            //     return Err(CmsError::InvalidIcc);
            // }
            //
            // let uc = &tag[last_position..last_position + unicode_length];
            // let wc = utf16be_to_utf16(uc);
            // let mac_string = String::from_utf16_lossy(&wc).to_string();

            return Ok(Some(ProfileText::Description(DescriptionString {
                ascii_string,
                unicode_language_code: unicode_code,
                unicode_string,
                mac_string: "".to_string(),
                script_code_code: -1,
            })));
        }
        Ok(None)
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
    fn read_nested_tone_curves(
        slice: &[u8],
        offset: usize,
        length: usize,
    ) -> Result<Option<Vec<ToneReprCurve>>, CmsError> {
        let mut curve_offset: usize = offset;
        let mut curves = Vec::new();
        for _ in 0..length {
            if slice.len() < curve_offset.safe_add(12)? {
                return Err(CmsError::InvalidProfile);
            }
            let mut tag_size = 0usize;
            let new_curve = Self::read_trc_tag(slice, curve_offset, 0, &mut tag_size)?;
            match new_curve {
                None => return Err(CmsError::InvalidProfile),
                Some(curve) => curves.push(curve),
            }
            curve_offset += tag_size;
            // 4 byte aligned
            if tag_size % 4 != 0 {
                curve_offset += 4 - tag_size % 4;
            }
        }
        Ok(Some(curves))
    }

    #[inline]
    fn read_lut_abm_type(
        slice: &[u8],
        entry: usize,
        tag_size: usize,
    ) -> Result<Option<LutWarehouse>, CmsError> {
        if tag_size < 48 {
            return Ok(None);
        }
        let last_tag_offset = tag_size.safe_add(entry)?;
        if last_tag_offset > slice.len() {
            return Err(CmsError::InvalidProfile);
        }
        let tag = &slice[entry..last_tag_offset];
        if tag.len() < 48 {
            return Err(CmsError::InvalidProfile);
        }
        let tag_type = u32::from_be_bytes([tag[0], tag[1], tag[2], tag[3]]);
        let tag_type_definition = TagTypeDefinition::from(tag_type);
        if tag_type_definition != TagTypeDefinition::MabLut
            && tag_type_definition != TagTypeDefinition::MbaLut
        {
            return Ok(None);
        }
        let in_channels = tag[8];
        let out_channels = tag[9];
        if in_channels > 4 && out_channels > 4 {
            return Ok(None);
        }
        let a_curve_offset = u32::from_be_bytes([tag[28], tag[29], tag[30], tag[31]]) as usize;
        let clut_offset = u32::from_be_bytes([tag[24], tag[25], tag[26], tag[27]]) as usize;
        let m_curve_offset = u32::from_be_bytes([tag[20], tag[21], tag[22], tag[23]]) as usize;
        let matrix_offset = u32::from_be_bytes([tag[16], tag[17], tag[18], tag[19]]) as usize;
        let b_curve_offset = u32::from_be_bytes([tag[12], tag[13], tag[14], tag[15]]) as usize;

        let matrix_end = matrix_offset.safe_add(12 * 4)?;
        if tag.len() < matrix_end {
            return Err(CmsError::InvalidProfile);
        }

        let m_tag = &tag[matrix_offset..matrix_end];

        let e00 = i32::from_be_bytes([m_tag[0], m_tag[1], m_tag[2], m_tag[3]]);
        let e01 = i32::from_be_bytes([m_tag[4], m_tag[5], m_tag[6], m_tag[7]]);
        let e02 = i32::from_be_bytes([m_tag[8], m_tag[9], m_tag[10], m_tag[11]]);

        let e10 = i32::from_be_bytes([m_tag[12], m_tag[13], m_tag[14], m_tag[15]]);
        let e11 = i32::from_be_bytes([m_tag[16], m_tag[17], m_tag[18], m_tag[19]]);
        let e12 = i32::from_be_bytes([m_tag[20], m_tag[21], m_tag[22], m_tag[23]]);

        let e20 = i32::from_be_bytes([m_tag[24], m_tag[25], m_tag[26], m_tag[27]]);
        let e21 = i32::from_be_bytes([m_tag[28], m_tag[29], m_tag[30], m_tag[31]]);
        let e22 = i32::from_be_bytes([m_tag[32], m_tag[33], m_tag[34], m_tag[35]]);

        let b0 = i32::from_be_bytes([m_tag[36], m_tag[37], m_tag[38], m_tag[39]]);
        let b1 = i32::from_be_bytes([m_tag[40], m_tag[41], m_tag[42], m_tag[43]]);
        let b2 = i32::from_be_bytes([m_tag[44], m_tag[45], m_tag[46], m_tag[47]]);

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

        let bias = Vector3f {
            v: [
                s15_fixed16_number_to_float(b0),
                s15_fixed16_number_to_float(b1),
                s15_fixed16_number_to_float(b2),
            ],
        };

        // Check if CLUT formed correctly
        if clut_offset.safe_add(20)? > tag.len() {
            return Err(CmsError::InvalidProfile);
        }

        let clut_sizes_slice = &tag[clut_offset..clut_offset.safe_add(16)?];
        let mut grid_points: [u8; 16] = [0; 16];
        for (&s, v) in clut_sizes_slice.iter().zip(grid_points.iter_mut()) {
            *v = s;
        }

        let mut clut_size = 1u32;
        for &i in grid_points.iter().take(in_channels as usize) {
            clut_size *= i as u32;
        }
        clut_size *= out_channels as u32;

        if clut_size == 0 {
            return Err(CmsError::InvalidProfile);
        }

        if clut_size > 10_000_000 {
            return Err(CmsError::InvalidProfile);
        }

        let clut_offset20 = clut_offset.safe_add(20)?;

        let clut_header = &tag[clut_offset..clut_offset20];
        let entry_size = clut_header[16];
        if entry_size != 1 && entry_size != 2 {
            return Err(CmsError::InvalidProfile);
        }

        let clut_end = clut_offset20.safe_add(clut_size.safe_mul(entry_size as u32)? as usize)?;

        if tag.len() < clut_end {
            return Err(CmsError::InvalidProfile);
        }

        let mut clut_table = vec![0f32; clut_size as usize];

        let shaped_clut_table = &tag[clut_offset20..clut_end];
        Self::read_lut_table_f32(
            shaped_clut_table,
            &mut clut_table,
            if entry_size == 1 {
                LutType::Lut8
            } else {
                LutType::Lut16
            },
        );

        let a_curves = if a_curve_offset == 0 {
            Vec::new()
        } else {
            Self::read_nested_tone_curves(tag, a_curve_offset, in_channels as usize)?
                .ok_or(CmsError::InvalidProfile)?
        };

        let m_curves = if m_curve_offset == 0 {
            Vec::new()
        } else {
            Self::read_nested_tone_curves(tag, m_curve_offset, out_channels as usize)?
                .ok_or(CmsError::InvalidProfile)?
        };

        let b_curves = if b_curve_offset == 0 {
            Vec::new()
        } else {
            Self::read_nested_tone_curves(tag, b_curve_offset, out_channels as usize)?
                .ok_or(CmsError::InvalidProfile)?
        };

        let wh = LutWarehouse::MCurves(LutMCurvesType {
            num_input_channels: in_channels,
            num_output_channels: out_channels,
            matrix: transform,
            clut: clut_table,
            a_curves,
            b_curves,
            m_curves,
            grid_points,
            bias,
        });
        Ok(Some(wh))
    }

    #[inline]
    fn read_lut_a_to_b_type(
        slice: &[u8],
        entry: usize,
        tag_size: usize,
    ) -> Result<Option<LutWarehouse>, CmsError> {
        if tag_size < 48 {
            return Ok(None);
        }
        let last_tag_offset = tag_size.safe_add(entry)?;
        if last_tag_offset > slice.len() {
            return Err(CmsError::InvalidProfile);
        }
        let tag = &slice[entry..last_tag_offset];
        if tag.len() < 48 {
            return Err(CmsError::InvalidProfile);
        }
        let tag_type = u32::from_be_bytes([tag[0], tag[1], tag[2], tag[3]]);
        let lut_type = LutType::try_from(tag_type)?;
        assert!(lut_type == LutType::Lut8 || lut_type == LutType::Lut16);

        if lut_type == LutType::Lut16 && tag.len() < 52 {
            return Err(CmsError::InvalidProfile);
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
            return Err(CmsError::InvalidProfile);
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
            return Err(CmsError::InvalidProfile);
        }
        let grid_points = tag[10];
        let clut_size = match (grid_points as u32).checked_pow(in_chan as u32) {
            Some(clut_size) => clut_size as usize,
            _ => {
                return Err(CmsError::InvalidProfile);
            }
        };
        match clut_size {
            1..=500_000 => {} // OK
            0 => {
                return Err(CmsError::InvalidProfile);
            }
            _ => {
                return Err(CmsError::InvalidProfile);
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

        let mut linearization_table = vec![0f32; lut_input_size];
        let linearization_table_end = lut_input_size
            .safe_mul(entry_size)?
            .safe_add(input_offset)?;
        if tag.len() < linearization_table_end {
            return Err(CmsError::InvalidProfile);
        }
        let shaped_input_table = &tag[input_offset..linearization_table_end];
        Self::read_lut_table_f32(shaped_input_table, &mut linearization_table, lut_type);

        let clut_offset = linearization_table_end;

        let clut_data_size = (clut_size * out_chan as usize) * entry_size;

        if tag.len() < clut_offset.safe_add(clut_data_size)? {
            return Err(CmsError::InvalidProfile);
        }

        let mut clut_table = vec![0f32; clut_size * out_chan as usize];

        let shaped_clut_table = &tag[clut_offset..clut_offset + clut_data_size];
        Self::read_lut_table_f32(shaped_clut_table, &mut clut_table, lut_type);

        let output_offset = clut_offset.safe_add(clut_data_size)?;

        let output_size = num_output_table_entries as usize * out_chan as usize;

        let mut out_gamma_table = vec![0f32; output_size];
        let shaped_output_table =
            &tag[output_offset..output_offset.safe_add(output_size.safe_mul(entry_size)?)?];
        Self::read_lut_table_f32(shaped_output_table, &mut out_gamma_table, lut_type);

        let wh = LutWarehouse::Lut(LutDataType {
            num_input_table_entries,
            num_output_table_entries,
            num_input_channels: in_chan,
            num_output_channels: out_chan,
            num_clut_grid_points: grid_points,
            matrix: transform,
            input_table: linearization_table,
            clut_table,
            output_table: out_gamma_table,
            lut_type,
        });
        Ok(Some(wh))
    }

    fn read_lut_tag(
        slice: &[u8],
        tag_entry: u32,
        tag_size: usize,
    ) -> Result<Option<LutWarehouse>, CmsError> {
        let lut_type = Self::read_lut_type(slice, tag_entry as usize, tag_size)?;
        Ok(if lut_type == LutType::Lut8 || lut_type == LutType::Lut16 {
            Self::read_lut_a_to_b_type(slice, tag_entry as usize, tag_size)?
        } else if lut_type == LutType::LutMba || lut_type == LutType::LutMab {
            Self::read_lut_abm_type(slice, tag_entry as usize, tag_size)?
        } else {
            None
        })
    }

    pub fn new_from_slice(slice: &[u8]) -> Result<Self, CmsError> {
        let header = ProfileHeader::new_from_slice(slice)?;
        let tags_count = header.tag_count as usize;
        if slice.len() >= MAX_PROFILE_SIZE {
            return Err(CmsError::InvalidProfile);
        }
        let tags_end = tags_count
            .safe_mul(TAG_SIZE)?
            .safe_add(size_of::<ProfileHeader>())?;
        if slice.len() < tags_end {
            return Err(CmsError::InvalidProfile);
        }
        let tags_slice = &slice[size_of::<ProfileHeader>()..tags_end];
        let mut profile = ColorProfile {
            rendering_intent: header.rendering_intent,
            pcs: header.pcs,
            profile_class: header.profile_class,
            color_space: header.data_color_space,
            white_point: header.illuminant,
            version_internal: header.version,
            ..Default::default()
        };
        let color_space = profile.color_space;
        for tag in tags_slice.chunks_exact(TAG_SIZE) {
            let tag_value = u32::from_be_bytes([tag[0], tag[1], tag[2], tag[3]]);
            let tag_entry = u32::from_be_bytes([tag[4], tag[5], tag[6], tag[7]]);
            let tag_size = u32::from_be_bytes([tag[8], tag[9], tag[10], tag[11]]) as usize;
            // Just ignore unknown tags
            if let Ok(tag) = Tag::try_from(tag_value) {
                match tag {
                    Tag::RedXyz => {
                        if color_space == DataColorSpace::Rgb {
                            profile.red_colorant =
                                Self::read_xyz_tag(slice, tag_entry as usize, tag_size)?;
                        }
                    }
                    Tag::GreenXyz => {
                        if color_space == DataColorSpace::Rgb {
                            profile.green_colorant =
                                Self::read_xyz_tag(slice, tag_entry as usize, tag_size)?;
                        }
                    }
                    Tag::BlueXyz => {
                        if color_space == DataColorSpace::Rgb {
                            profile.blue_colorant =
                                Self::read_xyz_tag(slice, tag_entry as usize, tag_size)?;
                        }
                    }
                    Tag::RedToneReproduction => {
                        if color_space == DataColorSpace::Rgb {
                            profile.red_trc =
                                Self::read_trc_tag_s(slice, tag_entry as usize, tag_size)?;
                        }
                    }
                    Tag::GreenToneReproduction => {
                        if color_space == DataColorSpace::Rgb {
                            profile.green_trc =
                                Self::read_trc_tag_s(slice, tag_entry as usize, tag_size)?;
                        }
                    }
                    Tag::BlueToneReproduction => {
                        if color_space == DataColorSpace::Rgb {
                            profile.blue_trc =
                                Self::read_trc_tag_s(slice, tag_entry as usize, tag_size)?;
                        }
                    }
                    Tag::GreyToneReproduction => {
                        if color_space == DataColorSpace::Rgb {
                            profile.gray_trc =
                                Self::read_trc_tag_s(slice, tag_entry as usize, tag_size)?;
                        }
                    }
                    Tag::MediaWhitePoint => {
                        match Self::read_xyz_tag(slice, tag_entry as usize, tag_size) {
                            Ok(wt) => profile.media_white_point = Some(wt),
                            Err(err) => return Err(err),
                        }
                    }
                    Tag::Luminance => {
                        match Self::read_xyz_tag(slice, tag_entry as usize, tag_size) {
                            Ok(wt) => profile.luminance = Some(wt),
                            Err(err) => return Err(err),
                        }
                    }
                    Tag::Measurement => {
                        profile.measurement =
                            Self::read_meas_tag(slice, tag_entry as usize, tag_size)?;
                    }
                    Tag::CodeIndependentPoints => {
                        profile.cicp = Self::read_cicp_tag(slice, tag_entry as usize, tag_size)?;
                    }
                    Tag::ChromaticAdaptation => {
                        profile.chromatic_adaptation =
                            Self::read_chad_tag(slice, tag_entry as usize, tag_size)?;
                    }
                    Tag::BlackPoint => {
                        match Self::read_xyz_tag(slice, tag_entry as usize, tag_size) {
                            Ok(wt) => profile.black_point = Some(wt),
                            Err(err) => return Err(err),
                        }
                    }
                    Tag::DeviceToPcsLutPerceptual => {
                        profile.lut_a_to_b_perceptual =
                            Self::read_lut_tag(slice, tag_entry, tag_size)?;
                    }
                    Tag::DeviceToPcsLutColorimetric => {
                        profile.lut_a_to_b_colorimetric =
                            Self::read_lut_tag(slice, tag_entry, tag_size)?;
                    }
                    Tag::DeviceToPcsLutSaturation => {
                        profile.lut_a_to_b_saturation =
                            Self::read_lut_tag(slice, tag_entry, tag_size)?;
                    }
                    Tag::PcsToDeviceLutPerceptual => {
                        profile.lut_b_to_a_perceptual =
                            Self::read_lut_tag(slice, tag_entry, tag_size)?;
                    }
                    Tag::PcsToDeviceLutColorimetric => {
                        profile.lut_b_to_a_colorimetric =
                            Self::read_lut_tag(slice, tag_entry, tag_size)?;
                    }
                    Tag::PcsToDeviceLutSaturation => {
                        profile.lut_b_to_a_saturation =
                            Self::read_lut_tag(slice, tag_entry, tag_size)?;
                    }
                    Tag::Gamut => {
                        profile.gamut = Self::read_lut_tag(slice, tag_entry, tag_size)?;
                    }
                    Tag::Copyright => {
                        profile.copyright =
                            Self::read_string_tag(slice, tag_entry as usize, tag_size)?;
                    }
                    Tag::ProfileDescription => {
                        profile.description =
                            Self::read_string_tag(slice, tag_entry as usize, tag_size)?;
                    }
                    Tag::ViewingConditionsDescription => {
                        profile.viewing_conditions_description =
                            Self::read_string_tag(slice, tag_entry as usize, tag_size)?;
                    }
                    Tag::DeviceModel => {
                        profile.device_model =
                            Self::read_string_tag(slice, tag_entry as usize, tag_size)?;
                    }
                    Tag::DeviceManufacturer => {
                        profile.device_manufacturer =
                            Self::read_string_tag(slice, tag_entry as usize, tag_size)?;
                    }
                    Tag::CharTarget => {
                        profile.char_target =
                            Self::read_string_tag(slice, tag_entry as usize, tag_size)?;
                    }
                    Tag::Chromaticity => {}
                    Tag::ObserverConditions => {
                        profile.viewing_conditions =
                            Self::read_viewing_conditions(slice, tag_entry as usize, tag_size)?;
                    }
                    Tag::Technology => {
                        profile.technology =
                            Self::read_tech_tag(slice, tag_entry as usize, tag_size)?;
                    }
                    Tag::CalibrationDateTime => {
                        profile.calibration_date =
                            Self::read_date_time_tag(slice, tag_entry as usize, tag_size)?;
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
            if CicpColorPrimaries::Bt709 == cicp.color_primaries {
                return SRGB_MATRIX;
            } else if CicpColorPrimaries::Bt2020 == cicp.color_primaries {
                return BT2020_MATRIX;
            } else if CicpColorPrimaries::Smpte240 == cicp.color_primaries {
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

    /// Computes colorants matrix. Returns not transposed matrix.
    ///
    /// To work on `const` context this method does have restrictions.
    /// If invalid values were provided it may return invalid matrix or NaNs.
    pub const fn colorants_matrix(white_point: XyY, primaries: ColorPrimaries) -> Matrix3f {
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
        let colorants = ColorProfile::rgb_to_xyz_const(xyz_matrix, white_point.to_xyz());
        adapt_to_d50(colorants, white_point)
    }

    /// Updates RGB triple colorimetry from 3 [Chromaticity] and white point
    pub const fn update_rgb_colorimetry(&mut self, white_point: XyY, primaries: ColorPrimaries) {
        let red_xyz = primaries.red.to_xyz();
        let green_xyz = primaries.green.to_xyz();
        let blue_xyz = primaries.blue.to_xyz();

        self.update_rgb_colorimetry_triplet(white_point, red_xyz, green_xyz, blue_xyz)
    }

    /// Updates RGB triple colorimetry from 3 [Xyz] and white point
    ///
    /// To work on `const` context this method does have restrictions.
    /// If invalid values were provided it may return invalid matrix or NaNs.
    pub const fn update_rgb_colorimetry_triplet(
        &mut self,
        white_point: XyY,
        red_xyz: Xyz,
        green_xyz: Xyz,
        blue_xyz: Xyz,
    ) {
        let xyz_matrix = Matrix3f {
            v: [
                [red_xyz.x, green_xyz.x, blue_xyz.x],
                [red_xyz.y, green_xyz.y, blue_xyz.y],
                [red_xyz.z, green_xyz.z, blue_xyz.z],
            ],
        };
        let colorants = ColorProfile::rgb_to_xyz_const(xyz_matrix, white_point.to_xyz());
        let colorants = adapt_to_d50(colorants, white_point);

        self.update_colorants(colorants);
    }

    pub(crate) const fn update_colorants(&mut self, colorants: Matrix3f) {
        // note: there's a transpose type of operation going on here
        self.red_colorant.x = colorants.v[0][0];
        self.red_colorant.y = colorants.v[1][0];
        self.red_colorant.z = colorants.v[2][0];
        self.green_colorant.x = colorants.v[0][1];
        self.green_colorant.y = colorants.v[1][1];
        self.green_colorant.z = colorants.v[2][1];
        self.blue_colorant.x = colorants.v[0][2];
        self.blue_colorant.y = colorants.v[1][2];
        self.blue_colorant.z = colorants.v[2][2];
    }

    /// Updates RGB triple colorimetry from CICP
    pub fn update_rgb_colorimetry_from_cicp(&mut self, cicp: CicpProfile) -> bool {
        self.cicp = Some(cicp);
        if !cicp.color_primaries.has_chromaticity()
            || !cicp.transfer_characteristics.has_transfer_curve()
        {
            return false;
        }
        let primaries_xy: ColorPrimaries = match cicp.color_primaries.try_into() {
            Ok(primaries) => primaries,
            Err(_) => return false,
        };
        let white_point: Chromaticity = match cicp.color_primaries.white_point() {
            Ok(v) => v,
            Err(_) => return false,
        };
        self.update_rgb_colorimetry(white_point.to_xyyb(), primaries_xy);

        let red_trc: ToneReprCurve = match cicp.transfer_characteristics.try_into() {
            Ok(trc) => trc,
            Err(_) => return false,
        };
        self.green_trc = Some(red_trc.clone());
        self.blue_trc = Some(red_trc.clone());
        self.red_trc = Some(red_trc);
        false
    }

    pub fn rgb_to_xyz(&self, xyz_matrix: Matrix3f, wp: Xyz) -> Option<Matrix3f> {
        let xyz_inverse = xyz_matrix.inverse();
        let s = xyz_inverse.mul_vector(wp.to_vector());
        let mut v = xyz_matrix.mul_row_vector::<0>(s);
        v = v.mul_row_vector::<1>(s);
        v = v.mul_row_vector::<2>(s);
        Some(v)
    }

    /// If Primaries is invalid will return invalid matrix on const context
    pub const fn rgb_to_xyz_const(xyz_matrix: Matrix3f, wp: Xyz) -> Matrix3f {
        let xyz_inverse = xyz_matrix.inverse();
        let s = xyz_inverse.mul_vector(wp.to_vector());
        let mut v = xyz_matrix.mul_row_vector::<0>(s);
        v = v.mul_row_vector::<1>(s);
        v = v.mul_row_vector::<2>(s);
        v
    }

    pub fn rgb_to_xyz_matrix(&self) -> Option<Matrix3f> {
        let xyz_matrix = self.colorant_matrix();
        let white_point = Chromaticity::D50.to_xyz();
        self.rgb_to_xyz(xyz_matrix, white_point)
    }

    /// Computes transform matrix RGB -> XYZ -> RGB
    /// Current profile is used as source, other as destination
    pub fn transform_matrix(&self, dest: &ColorProfile) -> Option<Matrix3f> {
        let source = self.rgb_to_xyz_matrix()?;
        let dst = dest.rgb_to_xyz_matrix()?;
        let dest_inverse = dst.inverse();
        Some(dest_inverse.mat_mul(source))
    }

    /// Returns volume of colors stored in profile
    pub fn profile_volume(&self) -> Option<f32> {
        let red_prim = self.red_colorant;
        let green_prim = self.green_colorant;
        let blue_prim = self.blue_colorant;
        let tetrahedral_vertices = Matrix3f {
            v: [
                [red_prim.x, red_prim.y, red_prim.z],
                [green_prim.x, green_prim.y, green_prim.z],
                [blue_prim.x, blue_prim.y, blue_prim.z],
            ],
        };
        let det = tetrahedral_vertices.determinant()?;
        Some(det / 6.0f32)
    }

    pub(crate) fn has_device_to_pcs_lut(&self) -> bool {
        self.lut_a_to_b_perceptual.is_some()
            || self.lut_a_to_b_saturation.is_some()
            || self.lut_a_to_b_colorimetric.is_some()
    }

    pub(crate) fn has_pcs_to_device_lut(&self) -> bool {
        self.lut_b_to_a_perceptual.is_some()
            || self.lut_b_to_a_saturation.is_some()
            || self.lut_b_to_a_colorimetric.is_some()
    }
}
