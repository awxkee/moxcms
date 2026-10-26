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
use crate::profile::{LutDataType, ProfileHeader};
use crate::tag::{TAG_SIZE, Tag, TagTypeDefinition};
use crate::trc::ToneReprCurve;
use crate::{
    CicpProfile, CmsError, ColorProfile, DataColorSpace, LocalizableString,
    LutMultidimensionalType, LutStore, LutType, LutWarehouse, Matrix3d, ProfileClass,
    ProfileSignature, ProfileText, ProfileVersion, Vector3d, ViewingConditions, Xyz, Xyzd,
};

pub(crate) trait FloatToFixedS15Fixed16 {
    fn to_s15_fixed16(self) -> i32;
}

pub(crate) trait FloatToFixedU8Fixed8 {
    fn to_u8_fixed8(self) -> u16;
}

// pub(crate) trait FloatToFixedU16 {
//     fn to_fixed_u16(self) -> u16;
// }

// impl FloatToFixedU16 for f32 {
//     #[inline]
//     fn to_fixed_u16(self) -> u16 {
//         const SCALE: f64 = (1 << 16) as f64;
//         (self as f64 * SCALE + 0.5)
//             .floor()
//             .clamp(u16::MIN as f64, u16::MAX as f64) as u16
//     }
// }

impl FloatToFixedS15Fixed16 for f32 {
    #[inline]
    fn to_s15_fixed16(self) -> i32 {
        const SCALE: f64 = (1 << 16) as f64;
        (self as f64 * SCALE + 0.5)
            .floor()
            .clamp(i32::MIN as f64, i32::MAX as f64) as i32
    }
}

impl FloatToFixedS15Fixed16 for f64 {
    #[inline]
    fn to_s15_fixed16(self) -> i32 {
        const SCALE: f64 = (1 << 16) as f64;
        (self * SCALE + 0.5)
            .floor()
            .clamp(i32::MIN as f64, i32::MAX as f64) as i32
    }
}

#[inline]
fn write_u32_be(into: &mut Vec<u8>, value: u32) {
    let bytes = value.to_be_bytes();
    into.push(bytes[0]);
    into.push(bytes[1]);
    into.push(bytes[2]);
    into.push(bytes[3]);
}

#[inline]
pub(crate) fn write_u16_be(into: &mut Vec<u8>, value: u16) {
    let bytes = value.to_be_bytes();
    into.push(bytes[0]);
    into.push(bytes[1]);
}

#[inline]
fn write_i32_be(into: &mut Vec<u8>, value: i32) {
    let bytes = value.to_be_bytes();
    into.push(bytes[0]);
    into.push(bytes[1]);
    into.push(bytes[2]);
    into.push(bytes[3]);
}

fn first_two_ascii_bytes(s: &String) -> [u8; 2] {
    let bytes = s.as_bytes();
    if bytes.len() >= 2 {
        bytes[0..2].try_into().unwrap()
    } else if bytes.len() == 1 {
        let vec = vec![bytes[0], 0u8];
        vec.try_into().unwrap()
    } else {
        let vec = vec![0u8, 0u8];
        vec.try_into().unwrap()
    }
}

/// Writes Multi Localized Unicode
#[inline]
fn write_mluc(into: &mut Vec<u8>, strings: &[LocalizableString]) -> usize {
    assert!(!strings.is_empty());
    let start = into.len();
    let tag_def: u32 = TagTypeDefinition::MultiLocalizedUnicode.into();
    write_u32_be(into, tag_def);
    write_u32_be(into, 0);
    let number_of_records = strings.len();
    write_u32_be(into, number_of_records as u32);
    write_u32_be(into, 12); // Record size, must be 12
    let lang = first_two_ascii_bytes(&strings[0].language);
    into.extend_from_slice(&lang);
    let country = first_two_ascii_bytes(&strings[0].country);
    into.extend_from_slice(&country);
    let first_string_len = strings[0].value.len() * 2;
    write_u32_be(into, first_string_len as u32);
    let mut first_string_offset = 16 + 12 * strings.len();
    write_u32_be(into, first_string_offset as u32);
    first_string_offset += first_string_len;
    for record in strings.iter().skip(1) {
        let lang = first_two_ascii_bytes(&record.language);
        into.extend_from_slice(&lang);
        let country = first_two_ascii_bytes(&record.country);
        into.extend_from_slice(&country);
        let first_string_len = record.value.len() * 2;
        write_u32_be(into, first_string_len as u32);
        write_u32_be(into, first_string_offset as u32);
        first_string_offset += first_string_len;
    }
    for record in strings.iter() {
        for chunk in record.value.encode_utf16() {
            write_u16_be(into, chunk);
        }
    }
    let end = into.len();
    end - start
}

#[inline]
fn write_string_value(into: &mut Vec<u8>, text: &ProfileText) -> usize {
    match text {
        ProfileText::PlainString(text) => {
            let vec = vec![LocalizableString {
                language: "en".to_string(),
                country: "US".to_string(),
                value: text.clone(),
            }];
            write_mluc(into, &vec)
        }
        ProfileText::Localizable(localizable) => {
            if localizable.is_empty() {
                return 0;
            }
            write_mluc(into, localizable)
        }
        ProfileText::Description(description) => {
            let value = if description.unicode_string.is_empty() {
                description.ascii_string.clone()
            } else {
                description.unicode_string.clone()
            };
            let vec = vec![LocalizableString {
                language: "en".to_string(),
                country: "US".to_string(),
                value,
            }];
            write_mluc(into, &vec)
        }
    }
}

#[inline]
fn write_xyz_tag_value(into: &mut Vec<u8>, xyz: Xyzd) {
    let tag_definition: u32 = TagTypeDefinition::Xyz.into();
    write_u32_be(into, tag_definition);
    write_u32_be(into, 0);
    let x_fixed = xyz.x.to_s15_fixed16();
    write_i32_be(into, x_fixed);
    let y_fixed = xyz.y.to_s15_fixed16();
    write_i32_be(into, y_fixed);
    let z_fixed = xyz.z.to_s15_fixed16();
    write_i32_be(into, z_fixed);
}

struct ProfileWriter {
    tags: Vec<u8>,
    entries: Vec<u8>,
    base_offset: usize,
}

impl ProfileWriter {
    fn new(tags_count: usize) -> Self {
        Self {
            tags: Vec::with_capacity(TAG_SIZE * tags_count),
            entries: Vec::new(),
            base_offset: size_of::<ProfileHeader>() + TAG_SIZE * tags_count,
        }
    }

    // Called after writing the payload; the tag's recorded size excludes padding.
    fn finish_tag(&mut self, tag: Tag, tag_size: usize) {
        write_u32_be(&mut self.tags, tag.into());
        write_u32_be(&mut self.tags, self.base_offset as u32);
        write_u32_be(&mut self.tags, tag_size as u32);

        let padding = (4 - tag_size % 4) % 4;
        self.entries.resize(self.entries.len() + padding, 0);
        self.base_offset += tag_size + padding;
    }

    fn finish(self, profile: &ColorProfile) -> Vec<u8> {
        let data_size = self.tags.len() + self.entries.len();
        let profile_header = ProfileHeader {
            size: (size_of::<ProfileHeader>() + data_size) as u32,
            pcs: profile.pcs,
            profile_class: profile.profile_class,
            rendering_intent: profile.rendering_intent,
            cmm_type: 0,
            version: if profile.cicp.is_some() {
                ProfileVersion::V4_3
            } else if profile.version_internal < ProfileVersion::V4_0 {
                ProfileVersion::V4_0
            } else {
                profile.version_internal
            },
            data_color_space: profile.color_space,
            creation_date_time: profile.creation_date_time,
            signature: ProfileSignature::Acsp,
            platform: 0u32,
            flags: 0u32,
            device_manufacturer: 0u32,
            device_model: 0u32,
            device_attributes: [0u8; 8],
            illuminant: profile.white_point.to_xyz(),
            creator: 0u32,
            profile_id: [0u8; 16],
            reserved: [0u8; 28],
            tag_count: (self.tags.len() / TAG_SIZE) as u32,
        };
        let mut encoded = profile_header.encode();
        encoded.reserve(data_size);
        encoded.extend(self.tags);
        encoded.extend(self.entries);
        encoded
    }
}

#[inline]
fn write_xyz(into: &mut Vec<u8>, xyz: Xyz) {
    let x_fixed = xyz.x.to_s15_fixed16();
    write_i32_be(into, x_fixed);
    let y_fixed = xyz.y.to_s15_fixed16();
    write_i32_be(into, y_fixed);
    let z_fixed = xyz.z.to_s15_fixed16();
    write_i32_be(into, z_fixed);
}

#[inline]
fn write_viewing_conditions_value(
    into: &mut Vec<u8>,
    viewing_conditions: &ViewingConditions,
) -> usize {
    let tag_definition: u32 = TagTypeDefinition::DefViewingConditions.into();
    write_u32_be(into, tag_definition);
    write_u32_be(into, 0);
    write_xyz(into, viewing_conditions.illuminant);
    write_xyz(into, viewing_conditions.surround);
    write_u32_be(into, viewing_conditions.observer.into());
    36
}

fn write_trc_entry(into: &mut Vec<u8>, trc: &ToneReprCurve) -> Result<usize, CmsError> {
    match trc {
        ToneReprCurve::Lut(lut) => {
            let curv: u32 = TagTypeDefinition::LutToneCurve.into();
            write_u32_be(into, curv);
            write_u32_be(into, 0);
            write_u32_be(into, lut.len() as u32);
            for item in lut.iter() {
                write_u16_be(into, *item);
            }
            Ok(12 + lut.len() * 2)
        }
        ToneReprCurve::Parametric(parametric_curve) => {
            if parametric_curve.len() > 7
                || parametric_curve.len() == 6
                || parametric_curve.len() == 2
            {
                return Err(CmsError::InvalidProfile);
            }
            let para: u32 = TagTypeDefinition::ParametricToneCurve.into();
            write_u32_be(into, para);
            write_u32_be(into, 0);
            if parametric_curve.len() == 1 {
                write_u16_be(into, 0);
            } else if parametric_curve.len() == 3 {
                write_u16_be(into, 1);
            } else if parametric_curve.len() == 4 {
                write_u16_be(into, 2);
            } else if parametric_curve.len() == 5 {
                write_u16_be(into, 3);
            } else if parametric_curve.len() == 7 {
                write_u16_be(into, 4);
            }
            write_u16_be(into, 0);
            for item in parametric_curve.iter() {
                write_i32_be(into, item.to_s15_fixed16());
            }
            Ok(12 + 4 * parametric_curve.len())
        }
    }
}

#[inline]
fn write_cicp_entry(into: &mut Vec<u8>, cicp: &CicpProfile) {
    let cicp_tag: u32 = TagTypeDefinition::Cicp.into();
    write_u32_be(into, cicp_tag);
    write_u32_be(into, 0);
    into.push(cicp.color_primaries as u8);
    into.push(cicp.transfer_characteristics as u8);
    into.push(cicp.matrix_coefficients as u8);
    into.push(if cicp.full_range { 1 } else { 0 });
}

fn write_chad(into: &mut Vec<u8>, matrix: Matrix3d) {
    let arr_type: u32 = TagTypeDefinition::S15Fixed16Array.into();
    write_u32_be(into, arr_type);
    write_u32_be(into, 0);
    write_matrix3d(into, matrix);
}

#[inline]
fn write_matrix3d(into: &mut Vec<u8>, v: Matrix3d) {
    write_i32_be(into, v.v[0][0].to_s15_fixed16());
    write_i32_be(into, v.v[0][1].to_s15_fixed16());
    write_i32_be(into, v.v[0][2].to_s15_fixed16());

    write_i32_be(into, v.v[1][0].to_s15_fixed16());
    write_i32_be(into, v.v[1][1].to_s15_fixed16());
    write_i32_be(into, v.v[1][2].to_s15_fixed16());

    write_i32_be(into, v.v[2][0].to_s15_fixed16());
    write_i32_be(into, v.v[2][1].to_s15_fixed16());
    write_i32_be(into, v.v[2][2].to_s15_fixed16());
}

#[inline]
fn write_vector3d(into: &mut Vec<u8>, v: Vector3d) {
    write_i32_be(into, v.v[0].to_s15_fixed16());
    write_i32_be(into, v.v[1].to_s15_fixed16());
    write_i32_be(into, v.v[2].to_s15_fixed16());
}

#[inline]
fn write_lut_entry(into: &mut Vec<u8>, lut: &LutDataType) -> Result<usize, CmsError> {
    if !lut.has_same_kind() {
        return Err(CmsError::InvalidProfile);
    }
    let start = into.len();
    let lut16_tag: u32 = match &lut.input_table {
        LutStore::Store8(_) => LutType::Lut8.into(),
        LutStore::Store16(_) => LutType::Lut16.into(),
    };
    write_u32_be(into, lut16_tag);
    write_u32_be(into, 0);
    into.push(lut.num_input_channels);
    into.push(lut.num_output_channels);
    into.push(lut.num_clut_grid_points);
    into.push(0);
    write_matrix3d(into, lut.matrix);
    write_u16_be(into, lut.num_input_table_entries);
    write_u16_be(into, lut.num_output_table_entries);
    match &lut.input_table {
        LutStore::Store8(input_table) => {
            for &item in input_table.iter() {
                into.push(item);
            }
        }
        LutStore::Store16(input_table) => {
            for &item in input_table.iter() {
                write_u16_be(into, item);
            }
        }
    }
    match &lut.clut_table {
        LutStore::Store8(input_table) => {
            for &item in input_table.iter() {
                into.push(item);
            }
        }
        LutStore::Store16(input_table) => {
            for &item in input_table.iter() {
                write_u16_be(into, item);
            }
        }
    }
    match &lut.output_table {
        LutStore::Store8(input_table) => {
            for &item in input_table.iter() {
                into.push(item);
            }
        }
        LutStore::Store16(input_table) => {
            for &item in input_table.iter() {
                write_u16_be(into, item);
            }
        }
    }
    let end = into.len();
    Ok(end - start)
}

#[inline]
fn write_mab_entry(
    into: &mut Vec<u8>,
    lut: &LutMultidimensionalType,
    is_a_to_b: bool,
) -> Result<usize, CmsError> {
    let start = into.len();
    let lut16_tag: u32 = if is_a_to_b {
        LutType::LutMab.into()
    } else {
        LutType::LutMba.into()
    };
    write_u32_be(into, lut16_tag);
    write_u32_be(into, 0);
    into.push(lut.num_input_channels);
    into.push(lut.num_output_channels);
    write_u16_be(into, 0);
    let mut working_offset = 32usize;

    let mut data = Vec::new();

    // Offset to "B curves"
    if !lut.b_curves.is_empty() {
        while !working_offset.is_multiple_of(4) {
            data.push(0);
            working_offset += 1;
        }

        write_u32_be(into, working_offset as u32);

        for trc in lut.b_curves.iter() {
            let curve_size = write_trc_entry(&mut data, trc)?;
            working_offset += curve_size;
            while !working_offset.is_multiple_of(4) {
                data.push(0);
                working_offset += 1;
            }
        }
    } else {
        write_u32_be(into, 0);
    }

    // Offset to matrix
    if !lut.m_curves.is_empty() {
        while !working_offset.is_multiple_of(4) {
            data.push(0);
            working_offset += 1;
        }

        write_u32_be(into, working_offset as u32);
        write_matrix3d(&mut data, lut.matrix);
        write_vector3d(&mut data, lut.bias);
        working_offset += 9 * 4 + 3 * 4;
        // Offset to "M curves"
        write_u32_be(into, working_offset as u32);
        for trc in lut.m_curves.iter() {
            let curve_size = write_trc_entry(&mut data, trc)?;
            working_offset += curve_size;
            while !working_offset.is_multiple_of(4) {
                data.push(0);
                working_offset += 1;
            }
        }
    } else {
        // Offset to matrix
        write_u32_be(into, 0);
        // Offset to "M curves"
        write_u32_be(into, 0);
    }

    let mut clut_start = data.len();

    // Offset to CLUT
    if let Some(clut) = &lut.clut {
        while !working_offset.is_multiple_of(4) {
            data.push(0);
            working_offset += 1;
        }

        clut_start = data.len();

        write_u32_be(into, working_offset as u32);

        // Writing CLUT
        for &pt in lut.grid_points.iter() {
            data.push(pt);
        }
        data.push(match clut {
            LutStore::Store8(_) => 1,
            LutStore::Store16(_) => 2,
        }); // Entry size
        data.push(0);
        data.push(0);
        data.push(0);
        match clut {
            LutStore::Store8(store) => {
                for &element in store.iter() {
                    data.push(element)
                }
            }
            LutStore::Store16(store) => {
                for &element in store.iter() {
                    write_u16_be(&mut data, element);
                }
            }
        }
    } else {
        write_u32_be(into, 0);
    }

    let clut_size = data.len() - clut_start;
    working_offset += clut_size;

    // Offset to "A curves"
    if !lut.a_curves.is_empty() {
        while !working_offset.is_multiple_of(4) {
            data.push(0);
            working_offset += 1;
        }

        write_u32_be(into, working_offset as u32);

        for trc in lut.a_curves.iter() {
            let curve_size = write_trc_entry(&mut data, trc)?;
            working_offset += curve_size;
            while !working_offset.is_multiple_of(4) {
                data.push(0);
                working_offset += 1;
            }
        }
    } else {
        write_u32_be(into, 0);
    }

    into.extend(data);

    let end = into.len();
    Ok(end - start)
}

fn write_lut(into: &mut Vec<u8>, lut: &LutWarehouse, is_a_to_b: bool) -> Result<usize, CmsError> {
    match lut {
        LutWarehouse::Lut(lut) => Ok(write_lut_entry(into, lut)?),
        LutWarehouse::Multidimensional(mab) => write_mab_entry(into, mab, is_a_to_b),
    }
}

impl ProfileHeader {
    fn encode(&self) -> Vec<u8> {
        let mut encoder: Vec<u8> = Vec::with_capacity(size_of::<ProfileHeader>());
        write_u32_be(&mut encoder, self.size); // Size
        write_u32_be(&mut encoder, 0); // CMM Type
        write_u32_be(&mut encoder, self.version.into()); // Version Number Type
        write_u32_be(&mut encoder, self.profile_class.into()); // Profile class
        write_u32_be(&mut encoder, self.data_color_space.into()); // Data color space
        write_u32_be(&mut encoder, self.pcs.into()); // PCS
        self.creation_date_time.encode(&mut encoder); // Date time
        write_u32_be(&mut encoder, self.signature.into()); // Profile signature
        write_u32_be(&mut encoder, self.platform);
        write_u32_be(&mut encoder, self.flags);
        write_u32_be(&mut encoder, self.device_manufacturer);
        write_u32_be(&mut encoder, self.device_model);
        for &i in self.device_attributes.iter() {
            encoder.push(i);
        }
        write_u32_be(&mut encoder, self.rendering_intent.into());
        write_i32_be(&mut encoder, self.illuminant.x.to_s15_fixed16());
        write_i32_be(&mut encoder, self.illuminant.y.to_s15_fixed16());
        write_i32_be(&mut encoder, self.illuminant.z.to_s15_fixed16());
        write_u32_be(&mut encoder, self.creator);
        for &i in self.profile_id.iter() {
            encoder.push(i);
        }
        for &i in self.reserved.iter() {
            encoder.push(i);
        }
        write_u32_be(&mut encoder, self.tag_count);
        encoder
    }
}

impl ColorProfile {
    fn writable_tags_count(&self) -> usize {
        let mut tags_count = 0usize;
        if self.red_colorant != Xyzd::default() {
            tags_count += 1;
        }
        if self.green_colorant != Xyzd::default() {
            tags_count += 1;
        }
        if self.blue_colorant != Xyzd::default() {
            tags_count += 1;
        }
        if self.red_trc.is_some() {
            tags_count += 1;
        }
        if self.green_trc.is_some() {
            tags_count += 1;
        }
        if self.blue_trc.is_some() {
            tags_count += 1;
        }
        if self.gray_trc.is_some() {
            tags_count += 1;
        }
        if self.cicp.is_some()
            && (self.profile_class == ProfileClass::InputDevice
                || self.profile_class == ProfileClass::DisplayDevice)
            && (self.color_space == DataColorSpace::Rgb
                || self.color_space == DataColorSpace::YCbr
                || self.color_space == DataColorSpace::Xyz)
        {
            tags_count += 1;
        }
        if self.media_white_point.is_some() {
            tags_count += 1;
        }
        if self.gamut.is_some() {
            tags_count += 1;
        }
        if self.chromatic_adaptation.is_some() {
            tags_count += 1;
        }
        if self.lut_a_to_b_perceptual.is_some() {
            tags_count += 1;
        }
        if self.lut_a_to_b_colorimetric.is_some() {
            tags_count += 1;
        }
        if self.lut_a_to_b_saturation.is_some() {
            tags_count += 1;
        }
        if self.lut_b_to_a_perceptual.is_some() {
            tags_count += 1;
        }
        if self.lut_b_to_a_colorimetric.is_some() {
            tags_count += 1;
        }
        if self.lut_b_to_a_saturation.is_some() {
            tags_count += 1;
        }
        if self.luminance.is_some() {
            tags_count += 1;
        }
        if let Some(description) = &self.description
            && description.has_values()
        {
            tags_count += 1;
        }
        if let Some(copyright) = &self.copyright
            && copyright.has_values()
        {
            tags_count += 1;
        }
        if self.viewing_conditions.is_some() {
            tags_count += 1;
        }
        if let Some(vd) = &self.viewing_conditions_description
            && vd.has_values()
        {
            tags_count += 1;
        }
        if let Some(vd) = &self.device_model
            && vd.has_values()
        {
            tags_count += 1;
        }
        if let Some(vd) = &self.device_manufacturer
            && vd.has_values()
        {
            tags_count += 1;
        }
        tags_count
    }

    /// Encodes profile
    pub fn encode(&self) -> Result<Vec<u8>, CmsError> {
        let mut writer = ProfileWriter::new(self.writable_tags_count());
        if self.red_colorant != Xyzd::default() {
            write_xyz_tag_value(&mut writer.entries, self.red_colorant);
            writer.finish_tag(Tag::RedXyz, 20);
        }
        if self.green_colorant != Xyzd::default() {
            write_xyz_tag_value(&mut writer.entries, self.green_colorant);
            writer.finish_tag(Tag::GreenXyz, 20);
        }
        if self.blue_colorant != Xyzd::default() {
            write_xyz_tag_value(&mut writer.entries, self.blue_colorant);
            writer.finish_tag(Tag::BlueXyz, 20);
        }
        if let Some(chad) = self.chromatic_adaptation {
            write_chad(&mut writer.entries, chad);
            writer.finish_tag(Tag::ChromaticAdaptation, 8 + 9 * 4);
        }
        if let Some(trc) = &self.red_trc {
            let entry_size = write_trc_entry(&mut writer.entries, trc)?;
            writer.finish_tag(Tag::RedToneReproduction, entry_size);
        }
        if let Some(trc) = &self.green_trc {
            let entry_size = write_trc_entry(&mut writer.entries, trc)?;
            writer.finish_tag(Tag::GreenToneReproduction, entry_size);
        }
        if let Some(trc) = &self.blue_trc {
            let entry_size = write_trc_entry(&mut writer.entries, trc)?;
            writer.finish_tag(Tag::BlueToneReproduction, entry_size);
        }
        if let Some(trc) = &self.gray_trc {
            let entry_size = write_trc_entry(&mut writer.entries, trc)?;
            writer.finish_tag(Tag::GreyToneReproduction, entry_size);
        }

        if let Some(media_wp) = self.media_white_point {
            write_xyz_tag_value(&mut writer.entries, media_wp);
            writer.finish_tag(Tag::MediaWhitePoint, 20);
        }

        // This tag may be present when the data colour space in the profile header is RGB, YCbCr, or XYZ, and the
        // profile class in the profile header is Input or Display. The tag shall not be present for other data colour spaces
        // or profile classes indicated in the profile header.

        if let Some(cicp) = &self.cicp
            && (self.profile_class == ProfileClass::InputDevice
                || self.profile_class == ProfileClass::DisplayDevice)
            && (self.color_space == DataColorSpace::Rgb
                || self.color_space == DataColorSpace::YCbr
                || self.color_space == DataColorSpace::Xyz)
        {
            write_cicp_entry(&mut writer.entries, cicp);
            writer.finish_tag(Tag::CodeIndependentPoints, 12);
        }

        if let Some(lut) = &self.lut_a_to_b_perceptual {
            let entry_size = write_lut(&mut writer.entries, lut, true)?;
            writer.finish_tag(Tag::DeviceToPcsLutPerceptual, entry_size);
        }

        if let Some(lut) = &self.lut_a_to_b_colorimetric {
            let entry_size = write_lut(&mut writer.entries, lut, true)?;
            writer.finish_tag(Tag::DeviceToPcsLutColorimetric, entry_size);
        }

        if let Some(lut) = &self.lut_a_to_b_saturation {
            let entry_size = write_lut(&mut writer.entries, lut, true)?;
            writer.finish_tag(Tag::DeviceToPcsLutSaturation, entry_size);
        }

        if let Some(lut) = &self.lut_b_to_a_perceptual {
            let entry_size = write_lut(&mut writer.entries, lut, false)?;
            writer.finish_tag(Tag::PcsToDeviceLutPerceptual, entry_size);
        }

        if let Some(lut) = &self.lut_b_to_a_colorimetric {
            let entry_size = write_lut(&mut writer.entries, lut, false)?;
            writer.finish_tag(Tag::PcsToDeviceLutColorimetric, entry_size);
        }

        if let Some(lut) = &self.lut_b_to_a_saturation {
            let entry_size = write_lut(&mut writer.entries, lut, false)?;
            writer.finish_tag(Tag::PcsToDeviceLutSaturation, entry_size);
        }

        if let Some(lut) = &self.gamut {
            let entry_size = write_lut(&mut writer.entries, lut, false)?;
            writer.finish_tag(Tag::Gamut, entry_size);
        }

        if let Some(luminance) = self.luminance {
            write_xyz_tag_value(&mut writer.entries, luminance);
            writer.finish_tag(Tag::Luminance, 20);
        }

        if let Some(description) = &self.description
            && description.has_values()
        {
            let entry_size = write_string_value(&mut writer.entries, description);
            writer.finish_tag(Tag::ProfileDescription, entry_size);
        }

        if let Some(copyright) = &self.copyright
            && copyright.has_values()
        {
            let entry_size = write_string_value(&mut writer.entries, copyright);
            writer.finish_tag(Tag::Copyright, entry_size);
        }

        if let Some(vc) = &self.viewing_conditions {
            let entry_size = write_viewing_conditions_value(&mut writer.entries, vc);
            writer.finish_tag(Tag::ObserverConditions, entry_size);
        }

        if let Some(vd) = &self.viewing_conditions_description
            && vd.has_values()
        {
            let entry_size = write_string_value(&mut writer.entries, vd);
            writer.finish_tag(Tag::ViewingConditionsDescription, entry_size);
        }

        if let Some(vd) = &self.device_model
            && vd.has_values()
        {
            let entry_size = write_string_value(&mut writer.entries, vd);
            writer.finish_tag(Tag::DeviceModel, entry_size);
        }

        if let Some(vd) = &self.device_manufacturer
            && vd.has_values()
        {
            let entry_size = write_string_value(&mut writer.entries, vd);
            writer.finish_tag(Tag::DeviceManufacturer, entry_size);
        }

        Ok(writer.finish(self))
    }
}

impl FloatToFixedU8Fixed8 for f32 {
    #[inline]
    fn to_u8_fixed8(self) -> u16 {
        if self > 255.0 + 255.0 / 256f32 {
            0xffffu16
        } else if self < 0.0 {
            0u16
        } else {
            (self * 256.0 + 0.5).floor() as u16
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CicpColorPrimaries, MatrixCoefficients, TransferCharacteristics};

    #[test]
    fn to_u8_fixed8() {
        assert_eq!(0, 0f32.to_u8_fixed8());
        assert_eq!(0x0100, 1f32.to_u8_fixed8());
        assert_eq!(u16::MAX, (255f32 + (255f32 / 256f32)).to_u8_fixed8());
    }

    #[test]
    fn to_s15_fixed16() {
        assert_eq!(0x80000000u32 as i32, (-32768f32).to_s15_fixed16());
        assert_eq!(0, 0f32.to_s15_fixed16());
        assert_eq!(0x10000, 1.0f32.to_s15_fixed16());
        assert_eq!(
            i32::MAX,
            (32767f32 + (65535f32 / 65536f32)).to_s15_fixed16()
        );
    }

    #[test]
    fn can_roundtrip_gray_with_cicp() {
        let mut basic_gray = ColorProfile::new_gray_with_gamma(2.2);
        basic_gray.cicp = Some(CicpProfile {
            color_primaries: CicpColorPrimaries::Bt709,
            transfer_characteristics: TransferCharacteristics::Srgb,
            matrix_coefficients: MatrixCoefficients::Smpte170m,
            full_range: true,
        });
        let encoded = basic_gray.encode().unwrap();
        let decoded_rountrip = ColorProfile::new_from_slice(&encoded).unwrap();
        // Gray profiles with cicp are not conformant, so we should remove it during encoding
        assert!(decoded_rountrip.cicp.is_none());
    }

    #[test]
    fn encoded_tags_are_aligned_and_roundtrip() {
        let mut profile = ColorProfile::new_srgb();
        profile.red_trc = Some(ToneReprCurve::Lut(vec![0, 32768, 65535]));
        profile.green_trc = Some(ToneReprCurve::Lut(vec![0, 65535]));
        profile.blue_trc = Some(ToneReprCurve::Lut(vec![0, 10000, 30000, 50000, 65535]));
        profile.description = Some(ProfileText::PlainString("x".into()));
        profile.copyright = Some(ProfileText::PlainString("yyy".into()));
        profile.device_model = Some(ProfileText::PlainString("mm".into()));
        profile.device_manufacturer = Some(ProfileText::PlainString("v".into()));

        let encoded = profile.encode().unwrap();
        let read_u32 =
            |offset| u32::from_be_bytes(encoded[offset..offset + 4].try_into().unwrap()) as usize;
        let expected_tags = [
            (Tag::RedXyz, 20),
            (Tag::GreenXyz, 20),
            (Tag::BlueXyz, 20),
            (Tag::ChromaticAdaptation, 44),
            (Tag::RedToneReproduction, 18),
            (Tag::GreenToneReproduction, 16),
            (Tag::BlueToneReproduction, 22),
            (Tag::MediaWhitePoint, 20),
            (Tag::CodeIndependentPoints, 12),
            (Tag::ProfileDescription, 30),
            (Tag::Copyright, 34),
            (Tag::DeviceModel, 32),
            (Tag::DeviceManufacturer, 30),
        ];
        assert_eq!(read_u32(0), encoded.len());
        assert_eq!(read_u32(128), expected_tags.len());
        let tags_end = 132 + expected_tags.len() * TAG_SIZE;
        let mut next_offset = tags_end;
        for (entry, (tag, size)) in encoded[132..tags_end]
            .as_chunks::<TAG_SIZE>()
            .0
            .iter()
            .zip(expected_tags)
        {
            let signature = u32::from_be_bytes(entry[0..4].try_into().unwrap());
            let offset = u32::from_be_bytes(entry[4..8].try_into().unwrap()) as usize;
            let encoded_size = u32::from_be_bytes(entry[8..12].try_into().unwrap()) as usize;
            assert_eq!(signature, u32::from(tag));
            assert_eq!(encoded_size, size);
            assert_eq!(offset, next_offset);
            assert_eq!(offset % 4, 0);
            let end = offset + size;
            next_offset = end.next_multiple_of(4);
            assert!(encoded[end..next_offset].iter().all(|&byte| byte == 0));
        }
        assert_eq!(next_offset, encoded.len());

        let decoded = ColorProfile::new_from_slice(&encoded).unwrap();
        assert_eq!(decoded.red_trc, profile.red_trc);
        assert_eq!(decoded.green_trc, profile.green_trc);
        assert_eq!(decoded.blue_trc, profile.blue_trc);
        for (text, expected) in [
            (decoded.description, "x"),
            (decoded.copyright, "yyy"),
            (decoded.device_model, "mm"),
            (decoded.device_manufacturer, "v"),
        ] {
            let Some(ProfileText::Localizable(records)) = text else {
                panic!("expected localized text");
            };
            assert_eq!(records.len(), 1);
            assert_eq!(records[0].value, expected);
        }
    }
}
