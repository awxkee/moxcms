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
// use jxl_oxide::{JxlImage, JxlThreadPool, Lcms2, Moxcms};
use image::DynamicImage;
use moxcms::{
    BarycentricWeightScale, ColorProfile, InterpolationMethod, Layout, LutStore, LutWarehouse,
    RenderingIntent, ToneReprCurve, TransformOptions, Vector3d,
};
use std::fs;
use std::time::Instant;

fn compute_abs_diff4(src: &[f32], dst: &[[f32; 4]], highlights: &mut [f32]) {
    let mut abs_r = f32::MIN;
    let mut abs_g = f32::MIN;
    let mut abs_b = f32::MIN;
    let mut abs_a = f32::MIN;
    let mut mean_r = 0f32;
    let mut mean_g = 0f32;
    let mut mean_b = 0f32;
    for ((src, dst), h) in src
        .chunks_exact(4)
        .zip(dst.iter())
        .zip(highlights.chunks_exact_mut(4))
    {
        let dr = (src[0] - dst[0]).abs();
        abs_r = dr.max(abs_r);
        mean_r += dr.abs();
        abs_g = (src[1] - dst[1]).abs().max(abs_g);
        mean_g += (src[1] - dst[1]).abs();
        abs_b = (src[2] - dst[2]).abs().max(abs_b);
        mean_b += (src[2] - dst[2]).abs();
        abs_a = (src[3] - dst[3]).abs().max(abs_a);
        if dr > 0.1 {
            h[0] = 1.0f32;
            h[3] = 1.0f32;
        } else if dr < 0.2 {
            h[1] = 1.0f32;
            h[3] = 1.0f32;
        }
    }
    mean_r /= dst.len() as f32;
    mean_g /= dst.len() as f32;
    mean_b /= dst.len() as f32;
    println!("Abs R {} Mean R {}", abs_r, mean_r);
    println!("Abs G {} Mean G {}", abs_g, mean_g);
    println!("Abs B {} Mean G {}", abs_b, mean_b);
    println!("Abs A {}", abs_a);
}

fn compute_abs_diff42(src: &[f32], dst: &[f32]) {
    let mut abs_r = f32::MIN;
    let mut abs_g = f32::MIN;
    let mut abs_b = f32::MIN;
    let mut abs_a = f32::MIN;
    let mut mean_r = 0f32;
    let mut mean_g = 0f32;
    let mut mean_b = 0f32;
    for (src, dst) in src.chunks_exact(4).zip(dst.chunks_exact(4)) {
        let dr = (src[0] - dst[0]).abs();
        abs_r = dr.max(abs_r);
        mean_r += dr.abs();
        abs_g = (src[1] - dst[1]).abs().max(abs_g);
        mean_g += (src[1] - dst[1]).abs();
        abs_b = (src[2] - dst[2]).abs().max(abs_b);
        mean_b += (src[2] - dst[2]).abs();
        abs_a = (src[3] - dst[3]).abs().max(abs_a);
    }
    mean_r /= dst.len() as f32;
    mean_g /= dst.len() as f32;
    mean_b /= dst.len() as f32;
    println!("Abs R {} Mean R {}", abs_r, mean_r);
    println!("Abs G {} Mean G {}", abs_g, mean_g);
    println!("Abs B {} Mean G {}", abs_b, mean_b);
    println!("Abs A {}", abs_a);
}

fn check_gray() {
    let gray_icc = fs::read("./assets/Generic Gray Gamma 2.2 Profile.icc").unwrap();
    let gray_target = ColorProfile::new_from_slice(&gray_icc).unwrap();
    let srgb_source = ColorProfile::new_srgb();

    let f_str = "./assets/bench.jpg";
    let img = image::ImageReader::open(f_str).unwrap().decode().unwrap();
    let rgb_img = img.to_rgb8();

    let transform = srgb_source
        .create_transform_8bit(
            Layout::Rgb,
            &gray_target,
            Layout::Gray,
            TransformOptions {
                rendering_intent: RenderingIntent::Perceptual,
                allow_use_cicp_transfer: false,
                prefer_fixed_point: true,
                interpolation_method: InterpolationMethod::Linear,
                barycentric_weight_scale: BarycentricWeightScale::Low,
                allow_extended_range_rgb_xyz: false,
            },
        )
        .unwrap();

    let mut gray_target = vec![0u8; rgb_img.width() as usize * rgb_img.height() as usize];
    transform.transform(&rgb_img, &mut gray_target).unwrap();
    image::save_buffer(
        "gray.png",
        &gray_target,
        img.width(),
        img.height(),
        image::ExtendedColorType::L8,
    )
    .unwrap();
}

fn to_lut_v4(lut: &Option<LutWarehouse>, to_pcs: bool) -> Option<LutWarehouse> {
    if lut.is_none() {
        return None;
    }
    let lut = lut.as_ref().unwrap();
    match lut {
        LutWarehouse::Lut(lut) => {
            let mut grid_points: [u8; 16] = [0; 16];

            for i in 0..lut.num_input_channels {
                grid_points[i as usize] = lut.num_clut_grid_points;
            }

            let mut a_curves: Vec<ToneReprCurve> = vec![];

            let a_multiplier = if to_pcs {
                lut.num_input_table_entries as usize
            } else {
                lut.num_output_table_entries as usize
            };

            let a_channels = if to_pcs {
                lut.num_input_channels as usize
            } else {
                lut.num_output_channels as usize
            };

            let a_table = if to_pcs {
                &lut.input_table
            } else {
                &lut.output_table
            };

            for i in 0..a_channels {
                match a_table {
                    LutStore::Store8(lut8) => {
                        let lc = &lut8[i * a_multiplier..(i + 1) * a_multiplier];
                        let remapped = lc
                            .iter()
                            .map(|&x| u16::from_ne_bytes([x, x]))
                            .collect::<Vec<_>>();
                        a_curves.push(ToneReprCurve::Lut(remapped.to_vec()))
                    }
                    LutStore::Store16(lut16) => a_curves.push(ToneReprCurve::Lut(
                        lut16[i * a_multiplier..(i + 1) * a_multiplier].to_vec(),
                    )),
                }
            }

            let mut b_curves: Vec<ToneReprCurve> = vec![];

            let b_multiplier = if !to_pcs {
                lut.num_input_table_entries as usize
            } else {
                lut.num_output_table_entries as usize
            };

            let b_channels = if !to_pcs {
                lut.num_input_channels as usize
            } else {
                lut.num_output_channels as usize
            };

            let b_table = if to_pcs {
                &lut.output_table
            } else {
                &lut.input_table
            };

            for i in 0..b_channels {
                match b_table {
                    LutStore::Store8(lut8) => {
                        let lc = &lut8[i * b_multiplier..(i + 1) * b_multiplier];
                        let remapped = lc
                            .iter()
                            .map(|&x| u16::from_ne_bytes([x, x]))
                            .collect::<Vec<_>>();
                        b_curves.push(ToneReprCurve::Lut(remapped.to_vec()))
                    }
                    LutStore::Store16(lut16) => b_curves.push(ToneReprCurve::Lut(
                        lut16[i * b_multiplier..(i + 1) * b_multiplier].to_vec(),
                    )),
                }
            }

            let data_type = moxcms::LutMultidimensionalType {
                num_input_channels: lut.num_input_channels,
                num_output_channels: lut.num_output_channels,
                grid_points,
                clut: Some(lut.clut_table.clone()),
                matrix: lut.matrix,
                bias: Vector3d::default(),
                m_curves: vec![],
                b_curves,
                a_curves,
            };
            Some(LutWarehouse::Multidimensional(data_type))
        }
        LutWarehouse::Multidimensional(md) => Some(LutWarehouse::Multidimensional(md.clone())),
    }
}

fn main() {
    let reader = image::ImageReader::open("./assets/bench.jpg").unwrap();
    let mut decoder = reader.into_decoder().unwrap();
    // let icc_profile =
    //     moxcms::ColorProfile::new_from_slice(&decoder.icc_profile().unwrap().unwrap()).unwrap();
    // let custom_profile = Profile::new_icc(&decoder.icc_profile().unwrap().unwrap()).unwrap();

    let gray_icc = fs::read("./assets/FOGRA55.icc").unwrap();
    let md_icc_v4 = fs::read("./assets/FOGRA55.icc").unwrap();
    let gray_target = ColorProfile::new_from_slice(&md_icc_v4).unwrap();

    // let mut profile_clone = gray_target.clone();
    // profile_clone.lut_a_to_b_colorimetric = to_lut_v4(&profile_clone.lut_a_to_b_colorimetric, true);
    // profile_clone.lut_a_to_b_perceptual = to_lut_v4(&profile_clone.lut_a_to_b_perceptual, true);
    // profile_clone.lut_a_to_b_saturation = to_lut_v4(&profile_clone.lut_a_to_b_saturation, true);
    //
    // profile_clone.lut_b_to_a_perceptual = to_lut_v4(&profile_clone.lut_b_to_a_perceptual, false);
    // profile_clone.lut_b_to_a_colorimetric =
    //     to_lut_v4(&profile_clone.lut_b_to_a_colorimetric, false);
    // profile_clone.lut_b_to_a_saturation = to_lut_v4(&profile_clone.lut_b_to_a_saturation, false);

    // let encoded = profile_clone.encode().unwrap();
    // fs::write("./assets/FOGRA55_v4.icc", encoded).unwrap();

    let img = DynamicImage::from_decoder(decoder).unwrap();
    let rgb_f32 = img.to_rgb16();

    let srgb = moxcms::ColorProfile::new_srgb();

    let transform = srgb
        .create_transform_16bit(
            moxcms::Layout::Rgb,
            &gray_target,
            moxcms::Layout::Inks7,
            TransformOptions {
                prefer_fixed_point: false,
                ..Default::default()
            },
        )
        .unwrap();

    let mut new_img_bytes = vec![0u16; (img.as_bytes().len() / 3) * 7];
    transform.transform(&rgb_f32, &mut new_img_bytes).unwrap();

    // let profile = lcms2::Profile::new_icc(&md_icc_v4).unwrap();
    // let srgb_lcms = lcms2::Profile::new_srgb();
    // let transform = lcms2::Transform::new(
    //     &srgb_lcms,
    //     lcms2::PixelFormat::RGB_8,
    //     &profile,
    //     lcms2::PixelFormat::CMYK_8,
    //     lcms2::Intent::RelativeColorimetric,
    // )
    // .unwrap();

    // transform.transform_pixels(img.as_bytes(), &mut new_img_bytes);

    let inverse_transform = gray_target
        .create_transform_16bit(
            moxcms::Layout::Inks7,
            &srgb,
            moxcms::Layout::Rgb,
            TransformOptions {
                prefer_fixed_point: false,
                rendering_intent: RenderingIntent::RelativeColorimetric,
                ..Default::default()
            },
        )
        .unwrap();

    let mut new_img_bytes2 = vec![0u16; img.as_bytes().len()];
    let instant = Instant::now();
    inverse_transform
        .transform(&new_img_bytes, &mut new_img_bytes2)
        .unwrap();
    println!("moxcms inverse {:?}", instant.elapsed());

    let recollected = new_img_bytes2
        .iter()
        .map(|&x| (x >> 8).min(255).max(0) as u8)
        .collect::<Vec<_>>();

    let new_img = DynamicImage::ImageRgb8(
        image::RgbImage::from_raw(img.width(), img.height(), recollected).unwrap(),
    );
    new_img.save("converted.png").unwrap();

    // let profile = lcms2::Profile::new_icc(&gray_icc).unwrap();
    // let srgb = lcms2::Profile::new_srgb();
    // let transform = lcms2::Transform::new(
    //     &profile,
    //     lcms2::PixelFormat::RGB_8,
    //     &srgb,
    //     lcms2::PixelFormat::RGB_8,
    //     lcms2::Intent::Perceptual,
    // )
    // .unwrap();
    // let inverse_transform = lcms2::Transform::new(
    //     &profile,
    //     lcms2::PixelFormat::CMYK_8,
    //     &srgb_lcms,
    //     lcms2::PixelFormat::RGB_8,
    //     lcms2::Intent::RelativeColorimetric,
    // )
    // .unwrap();

    let mut new_img = vec![0u8; img.as_bytes().len()];
    // transform.transform_in_place(new_img.as_mut_slice());
    // inverse_transform.transform_in_place(new_img.as_mut_slice());
    let instant = Instant::now();
    // inverse_transform.transform_pixels(&new_img_bytes, &mut new_img);
    println!("LCMS inverse {:?}", instant.elapsed());

    let new_img = DynamicImage::ImageRgb8(
        image::RgbImage::from_raw(img.width(), img.height(), new_img).unwrap(),
    );
    new_img.save("converted_lcms2.png").unwrap();
}

// fn main() {
//     let us_swop_icc = fs::read("./assets/srgb_perceptual.icc").unwrap();
//
//     let width = 1920;
//     let height = 1080;
//
//     let cmyk = vec![0u8; width * height * 4];
//
//     let color_profile = ColorProfile::new_display_p3();// ColorProfile::new_from_slice(&us_swop_icc).unwrap();
//     let dest_profile = ColorProfile::new_srgb();
//     let mut dst = vec![32u8; width * height * 4];
//     for dst in dst.chunks_exact_mut(4) {
//         let v = rand::rng().random_range(0..255) as u8;
//         dst[0] = v;
//         dst[1] =v;
//         dst[2] =v;
//         dst[3] = 255;
//     }
//     let transform = color_profile
//         .create_transform_8bit(
//             Layout::Rgba,
//             &dest_profile,
//             Layout::Rgba,
//             TransformOptions {
//                 interpolation_method: InterpolationMethod::Pyramid,
//                 prefer_fixed_point: true,
//                 ..Default::default()
//             },
//         )
//         .unwrap();
//     transform.transform(&cmyk, &mut dst).unwrap();
// }
