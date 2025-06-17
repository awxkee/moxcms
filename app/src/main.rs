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
use image::ImageDecoder;
use lcms2::Profile;
use moxcms::{
    BarycentricWeightScale, ColorProfile, InterpolationMethod, Layout, RenderingIntent,
    TransformOptions, Xyz,
};
use std::fs;

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

fn main() {
    let reader = image::ImageReader::open("./assets/bench.jpg").unwrap();
    let mut decoder = reader.into_decoder().unwrap();
    let icc_profile =
        moxcms::ColorProfile::new_from_slice(&decoder.icc_profile().unwrap().unwrap()).unwrap();
    let custom_profile = Profile::new_icc(&decoder.icc_profile().unwrap().unwrap()).unwrap();

    let gray_icc = fs::read("./assets/bt2020_pq.icc").unwrap();
    let gray_target = ColorProfile::new_from_slice(&gray_icc).unwrap();

    let img = DynamicImage::from_decoder(decoder).unwrap();

    let srgb = moxcms::ColorProfile::new_srgb();

    let transform = srgb
        .create_transform_8bit(
            moxcms::Layout::Rgb,
            &gray_target,
            moxcms::Layout::Rgba,
            TransformOptions {
                prefer_fixed_point: false,
                ..Default::default()
            },
        )
        .unwrap();

    let mut new_img_bytes = vec![0u8; (img.as_bytes().len() / 3) * 4];
    transform
        .transform(img.as_bytes(), &mut new_img_bytes)
        .unwrap();

    let inverse_transform = gray_target
        .create_transform_8bit(
            moxcms::Layout::Rgba,
            &srgb,
            moxcms::Layout::Rgb,
            TransformOptions {
                prefer_fixed_point: false,
                ..Default::default()
            },
        )
        .unwrap();

    let mut new_img_bytes2 = vec![0u8; img.as_bytes().len()];
    inverse_transform
        .transform(&new_img_bytes, &mut new_img_bytes2)
        .unwrap();

    let new_img = DynamicImage::ImageRgb8(
        image::RgbImage::from_raw(img.width(), img.height(), new_img_bytes2).unwrap(),
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
    //     &srgb,
    //     lcms2::PixelFormat::RGB_8,
    //     &profile,
    //     lcms2::PixelFormat::RGB_8,
    //     lcms2::Intent::Perceptual,
    // )
    // .unwrap();
    //
    // let mut new_img = img.to_rgb8().to_vec();
    // transform.transform_in_place(new_img.as_mut_slice());
    // inverse_transform.transform_in_place(new_img.as_mut_slice());
    //
    // let new_img = DynamicImage::ImageRgb8(
    //     image::RgbImage::from_raw(img.width(), img.height(), new_img).unwrap(),
    // );
    // new_img.save("converted_lcms2.png").unwrap();
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
