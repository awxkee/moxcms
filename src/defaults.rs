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
use crate::math::copysign;
use crate::trc::{Trc, build_trc_table, curve_from_gamma};
use crate::{
    Chromacity, ChromacityTriple, ColorPrimaries, ColorProfile, DataColorSpace, LocalizableString,
    ProfileClass, ProfileText, RenderingIntent, XyY, pow,
};
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

pub(crate) const fn white_point_d50() -> XyY {
    white_point_from_temperature(5003)
}

// https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2100-2-201807-I!!PDF-F.pdf
// Perceptual Quantization / SMPTE standard ST.2084
#[inline]
const fn pq_curve(x: f64) -> f64 {
    const M1: f64 = 2610.0 / 16384.0;
    const M2: f64 = (2523.0 / 4096.0) * 128.0;
    const C1: f64 = 3424.0 / 4096.0;
    const C2: f64 = (2413.0 / 4096.0) * 32.0;
    const C3: f64 = (2392.0 / 4096.0) * 32.0;

    if x == 0.0 {
        return 0.0;
    }
    let sign = x;
    let x = x.abs();

    let xpo = pow(x, 1.0 / M2);
    let num = (xpo - C1).max(0.0);
    let den = C2 - C3 * xpo;
    let res = pow(num / den, 1.0 / M1);

    copysign(res, sign)
}

impl ColorProfile {
    /// Creates new sRGB profile
    pub fn new_srgb() -> ColorProfile {
        let primaries = ChromacityTriple::try_from(ColorPrimaries::Bt709).unwrap();
        const WHITE_POINT: XyY = white_point_srgb();
        let mut profile = ColorProfile::default();
        profile.update_rgb_colorimetry(WHITE_POINT, primaries);

        let curve = Trc::Parametric(vec![2.4, 1. / 1.055, 0.055 / 1.055, 1. / 12.92, 0.04045]);
        profile.red_trc = Some(curve.clone());
        profile.blue_trc = Some(curve.clone());
        profile.green_trc = Some(curve);
        profile.profile_class = ProfileClass::DisplayDevice;
        profile.rendering_intent = RenderingIntent::Perceptual;
        profile.color_space = DataColorSpace::Rgb;
        profile.pcs = DataColorSpace::Xyz;
        profile.media_white_point = Some(Chromacity::D65.to_xyz());
        profile.white_point = Chromacity::D50.to_xyz();
        profile.description = Some(ProfileText::Localizable(vec![LocalizableString::new(
            "en".to_string(),
            "US".to_string(),
            "sRGB IEC61966-2.1".to_string(),
        )]));
        profile.copyright = Some(ProfileText::Localizable(vec![LocalizableString::new(
            "en".to_string(),
            "US".to_string(),
            "Public Domain".to_string(),
        )]));
        profile
    }

    /// Creates new Adobe RGB profile
    pub fn new_adobe_rgb() -> ColorProfile {
        let triplet = ChromacityTriple {
            red: Chromacity::new(0.6400, 0.3300),
            green: Chromacity::new(0.2100, 0.7100),
            blue: Chromacity::new(0.1500, 0.0600),
        };
        const WHITE_POINT: XyY = white_point_srgb();
        let mut profile = ColorProfile::default();
        profile.update_rgb_colorimetry(WHITE_POINT, triplet);

        let curve = curve_from_gamma(2.19921875f32);
        profile.red_trc = Some(curve.clone());
        profile.blue_trc = Some(curve.clone());
        profile.green_trc = Some(curve);
        profile.profile_class = ProfileClass::DisplayDevice;
        profile.rendering_intent = RenderingIntent::Perceptual;
        profile.color_space = DataColorSpace::Rgb;
        profile.pcs = DataColorSpace::Xyz;
        profile.media_white_point = Some(Chromacity::D65.to_xyz());
        profile.white_point = Chromacity::D50.to_xyz();
        profile.description = Some(ProfileText::Localizable(vec![LocalizableString::new(
            "en".to_string(),
            "US".to_string(),
            "Adobe RGB 1998".to_string(),
        )]));
        profile.copyright = Some(ProfileText::Localizable(vec![LocalizableString::new(
            "en".to_string(),
            "US".to_string(),
            "Public Domain".to_string(),
        )]));
        profile
    }

    /// Creates new Display P3 profile
    pub fn new_display_p3() -> ColorProfile {
        let primaries = ChromacityTriple::try_from(ColorPrimaries::Smpte432).unwrap();
        const WHITE_POINT: XyY = white_point_srgb();
        let mut profile = ColorProfile::default();
        profile.update_rgb_colorimetry(WHITE_POINT, primaries);

        let curve = Trc::Parametric(vec![2.4, 1. / 1.055, 0.055 / 1.055, 1. / 12.92, 0.04045]);
        profile.red_trc = Some(curve.clone());
        profile.blue_trc = Some(curve.clone());
        profile.green_trc = Some(curve);
        profile.profile_class = ProfileClass::DisplayDevice;
        profile.rendering_intent = RenderingIntent::Perceptual;
        profile.color_space = DataColorSpace::Rgb;
        profile.pcs = DataColorSpace::Xyz;
        profile.media_white_point = Some(Chromacity::D65.to_xyz());
        profile.white_point = Chromacity::D50.to_xyz();
        profile.description = Some(ProfileText::Localizable(vec![LocalizableString::new(
            "en".to_string(),
            "US".to_string(),
            "Display P3".to_string(),
        )]));
        profile.copyright = Some(ProfileText::Localizable(vec![LocalizableString::new(
            "en".to_string(),
            "US".to_string(),
            "Public Domain".to_string(),
        )]));
        profile
    }

    /// Creates new Display P3 PQ profile
    pub fn new_display_p3_pq() -> ColorProfile {
        let primaries = ChromacityTriple::try_from(ColorPrimaries::Smpte432).unwrap();
        const WHITE_POINT: XyY = white_point_srgb();
        let mut profile = ColorProfile::default();
        profile.update_rgb_colorimetry(WHITE_POINT, primaries);

        let table = build_trc_table(4096, pq_curve);
        let curve = Trc::Lut(table);

        profile.red_trc = Some(curve.clone());
        profile.blue_trc = Some(curve.clone());
        profile.green_trc = Some(curve);
        profile.profile_class = ProfileClass::DisplayDevice;
        profile.rendering_intent = RenderingIntent::Perceptual;
        profile.color_space = DataColorSpace::Rgb;
        profile.pcs = DataColorSpace::Xyz;
        profile.media_white_point = Some(Chromacity::D65.to_xyz());
        profile.white_point = Chromacity::D50.to_xyz();
        profile.description = Some(ProfileText::Localizable(vec![LocalizableString::new(
            "en".to_string(),
            "US".to_string(),
            "Display P3 PQ".to_string(),
        )]));
        profile.copyright = Some(ProfileText::Localizable(vec![LocalizableString::new(
            "en".to_string(),
            "US".to_string(),
            "Public Domain".to_string(),
        )]));
        profile
    }

    /// Creates new DCI P3 profile
    pub fn new_dci_p3() -> ColorProfile {
        let triplet = ChromacityTriple {
            red: Chromacity::new(0.680, 0.320),
            green: Chromacity::new(0.265, 0.690),
            blue: Chromacity::new(0.150, 0.060),
        };
        const WHITE_POINT: XyY = white_point_srgb();
        let mut profile = ColorProfile::default();
        profile.update_rgb_colorimetry(WHITE_POINT, triplet);

        let curve = curve_from_gamma(2.6f32);
        profile.red_trc = Some(curve.clone());
        profile.blue_trc = Some(curve.clone());
        profile.green_trc = Some(curve);
        profile.profile_class = ProfileClass::DisplayDevice;
        profile.rendering_intent = RenderingIntent::Perceptual;
        profile.color_space = DataColorSpace::Rgb;
        profile.pcs = DataColorSpace::Xyz;
        profile.media_white_point = Some(Chromacity::D65.to_xyz());
        profile.white_point = Chromacity::D50.to_xyz();
        profile.description = Some(ProfileText::Localizable(vec![LocalizableString::new(
            "en".to_string(),
            "US".to_string(),
            "DCI P3".to_string(),
        )]));
        profile.copyright = Some(ProfileText::Localizable(vec![LocalizableString::new(
            "en".to_string(),
            "US".to_string(),
            "Public Domain".to_string(),
        )]));
        profile
    }

    /// Creates new ProPhoto RGB profile
    pub fn new_pro_photo_rgb() -> ColorProfile {
        let triplet = ChromacityTriple {
            red: Chromacity::new(0.734699, 0.265301),
            green: Chromacity::new(0.159597, 0.840403),
            blue: Chromacity::new(0.036598, 0.000105),
        };
        const WHITE_POINT: XyY = white_point_d50();
        let mut profile = ColorProfile::default();
        profile.update_rgb_colorimetry(WHITE_POINT, triplet);

        let curve = curve_from_gamma(1.8f32);
        profile.red_trc = Some(curve.clone());
        profile.blue_trc = Some(curve.clone());
        profile.green_trc = Some(curve);
        profile.profile_class = ProfileClass::DisplayDevice;
        profile.rendering_intent = RenderingIntent::Perceptual;
        profile.color_space = DataColorSpace::Rgb;
        profile.pcs = DataColorSpace::Xyz;
        profile.media_white_point = Some(Chromacity::D50.to_xyz());
        profile.white_point = Chromacity::D50.to_xyz();
        profile.description = Some(ProfileText::Localizable(vec![LocalizableString::new(
            "en".to_string(),
            "US".to_string(),
            "ProPhoto RGB".to_string(),
        )]));
        profile.copyright = Some(ProfileText::Localizable(vec![LocalizableString::new(
            "en".to_string(),
            "US".to_string(),
            "Public Domain".to_string(),
        )]));
        profile
    }

    /// Creates new Bt.2020 profile
    pub fn new_bt2020() -> ColorProfile {
        let primaries = ChromacityTriple::try_from(ColorPrimaries::Bt2020).unwrap();
        const WHITE_POINT: XyY = white_point_srgb();
        let mut profile = ColorProfile::default();
        profile.update_rgb_colorimetry(WHITE_POINT, primaries);

        let curve = Trc::Parametric(vec![2.4, 1. / 1.055, 0.055 / 1.055, 1. / 12.92, 0.04045]);
        profile.red_trc = Some(curve.clone());
        profile.blue_trc = Some(curve.clone());
        profile.green_trc = Some(curve);
        profile.profile_class = ProfileClass::DisplayDevice;
        profile.rendering_intent = RenderingIntent::Perceptual;
        profile.color_space = DataColorSpace::Rgb;
        profile.pcs = DataColorSpace::Xyz;
        profile.media_white_point = Some(Chromacity::D65.to_xyz());
        profile.white_point = Chromacity::D50.to_xyz();
        profile.description = Some(ProfileText::Localizable(vec![LocalizableString::new(
            "en".to_string(),
            "US".to_string(),
            "Rec.2020".to_string(),
        )]));
        profile.copyright = Some(ProfileText::Localizable(vec![LocalizableString::new(
            "en".to_string(),
            "US".to_string(),
            "Public Domain".to_string(),
        )]));
        profile
    }

    /// Creates new Bt.2020 PQ profile
    pub fn new_bt2020_pq() -> ColorProfile {
        let primaries = ChromacityTriple::try_from(ColorPrimaries::Bt2020).unwrap();
        const WHITE_POINT: XyY = white_point_srgb();
        let mut profile = ColorProfile::default();
        profile.update_rgb_colorimetry(WHITE_POINT, primaries);

        let table = build_trc_table(4096, pq_curve);
        let curve = Trc::Lut(table);

        profile.red_trc = Some(curve.clone());
        profile.blue_trc = Some(curve.clone());
        profile.green_trc = Some(curve);
        profile.profile_class = ProfileClass::DisplayDevice;
        profile.rendering_intent = RenderingIntent::Perceptual;
        profile.color_space = DataColorSpace::Rgb;
        profile.pcs = DataColorSpace::Xyz;
        profile.media_white_point = Some(Chromacity::D65.to_xyz());
        profile.white_point = Chromacity::D50.to_xyz();
        profile.description = Some(ProfileText::Localizable(vec![LocalizableString::new(
            "en".to_string(),
            "US".to_string(),
            "Rec.2020 PQ".to_string(),
        )]));
        profile.copyright = Some(ProfileText::Localizable(vec![LocalizableString::new(
            "en".to_string(),
            "US".to_string(),
            "Public Domain".to_string(),
        )]));
        profile
    }

    /// Creates new Monochrome profile
    pub fn new_gray_with_gamma(gamma: f32) -> ColorProfile {
        ColorProfile {
            gray_trc: Some(curve_from_gamma(gamma)),
            profile_class: ProfileClass::DisplayDevice,
            rendering_intent: RenderingIntent::Perceptual,
            color_space: DataColorSpace::Gray,
            media_white_point: Some(Chromacity::D65.to_xyz()),
            white_point: Chromacity::D50.to_xyz(),
            copyright: Some(ProfileText::Localizable(vec![LocalizableString::new(
                "en".to_string(),
                "US".to_string(),
                "Public Domain".to_string(),
            )])),
            ..Default::default()
        }
    }
}
