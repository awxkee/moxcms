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
use image::GenericImageView;
use lcms2::Profile;
use moxcms::{
    pow, ColorProfile, InterpolationMethod, Layout, RenderingIntent, TransformOptions,
};
use std::fs;
use std::fs::File;
use std::io::{BufReader, Read};
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::options::DecoderOptions;
use zune_jpeg::JpegDecoder;

#[inline]
/// Gamma transfer function for sRGB
fn srgb_from_linear(linear: f64) -> f64 {
    if linear < 0.0f64 {
        0.0f64
    } else if linear < 0.0030412825601275209f64 {
        linear * 12.92f64
    } else if linear < 1.0f64 {
        1.0550107189475866f64 * pow(linear, 1.0f64 / 2.4f64) - 0.0550107189475866f64
    } else {
        1.0f64
    }
}

fn main() {
    let funny_icc = fs::read("./assets/rendered.jxl").unwrap();

    // println!("{:?}", decoded);

    let decoded = ColorProfile::new_bt2020_pq();
    let encoded = decoded.encode().unwrap();

    fs::write("./bt2020.icc", encoded).unwrap();

    let srgb_perceptual_icc = fs::read("./assets/srgb_perceptual.icc").unwrap();
    let out_icc = fs::read("./assets/output.icc").unwrap();

    let funny_profile = ColorProfile::new_from_slice(&funny_icc).unwrap();

    let srgb_perceptual_profile = ColorProfile::new_from_slice(&srgb_perceptual_icc).unwrap();
    let out_profile = ColorProfile::new_from_slice(&out_icc).unwrap();

    let f_str = "./assets/test1.jpg";
    let file = File::open(f_str).expect("Failed to open file");

    let img = image::ImageReader::open(f_str).unwrap().decode().unwrap();
    let rgb = img.to_rgb8();

    let reader = BufReader::new(file);
    let ref_reader = &reader;

    let options = DecoderOptions::new_fast().jpeg_set_out_colorspace(ColorSpace::RGB);

    let mut decoder = JpegDecoder::new_with_options(reader, options);

    // let mut decoder = JpegDecoder::new(reader);
    decoder.options().set_use_unsafe(true);
    decoder.decode_headers().unwrap();
    let mut real_dst = vec![0u8; decoder.output_buffer_size().unwrap()];

    let custom_profile = Profile::new_icc(&srgb_perceptual_icc).unwrap();
    //
    let srgb_profile = Profile::new_srgb();

    decoder.decode_into(&mut real_dst).unwrap();

    let real_dst = real_dst
        .chunks_exact(3)
        .flat_map(|x| [x[0], x[1], x[2], 255u8])
        .map(|x| x as f32 * (1.0 / 255.0))
        .collect::<Vec<_>>();

    let pr1 = lcms2::Profile::new_icc(&funny_icc).unwrap();

    // let t1 = Transform::new(
    //     &lcms2::Profile::new_srgb(),
    //     PixelFormat::RGBA_8,
    //     &pr1,
    //     PixelFormat::RGBA_8,
    //     Intent::Perceptual,
    // )
    // .unwrap();
    //
    // let t2 = Transform::new(
    //     &custom_profile,
    //     PixelFormat::RGBA_8,
    //     &srgb_profile,
    //     PixelFormat::RGBA_8,
    //     Intent::Perceptual,
    // )
    //     .unwrap();

    let mut cmyk = vec![0f32; (decoder.output_buffer_size().unwrap() / 3) * 4];

    let icc = decoder.icc_profile().unwrap();
    let color_profile = ColorProfile::new_from_slice(&srgb_perceptual_icc).unwrap();
    let cmyk_profile = ColorProfile::new_from_slice(&funny_icc).unwrap();
    // let color_profile = ColorProfile::new_gray_with_gamma(2.2);
    let mut dest_profile = ColorProfile::new_srgb();

    // t1.transform_pixels(&real_dst, &mut cmyk);

    let transform = dest_profile
        .create_transform_f32(
            Layout::Rgba,
            &funny_profile,
            Layout::Rgba,
            TransformOptions {
                rendering_intent: RenderingIntent::Perceptual,
                allow_use_cicp_transfer: false,
                prefer_fixed_point: false,
                interpolation_method: InterpolationMethod::Prism,
            },
        )
        .unwrap();

    transform.transform(&real_dst, &mut cmyk).unwrap();

    dest_profile.rendering_intent = RenderingIntent::Perceptual;
    let transform = funny_profile
        .create_transform_f32(
            Layout::Rgba,
            &out_profile,
            Layout::Rgba,
            TransformOptions {
                rendering_intent: RenderingIntent::Perceptual,
                allow_use_cicp_transfer: false,
                prefer_fixed_point: false,
                interpolation_method: InterpolationMethod::Prism,
            },
        )
        .unwrap();
    let mut dst = vec![0f32; real_dst.len()];
    //
    // let instant = Instant::now();

    for (src, dst) in cmyk
        .chunks_exact(img.width() as usize * 4)
        .zip(dst.chunks_exact_mut(img.width() as usize * 4))
    {
        transform
            .transform(
                &src[..img.width() as usize * 4],
                &mut dst[..img.width() as usize * 4],
            )
            .unwrap();
    }

    // t2.transform_pixels(&cmyk, &mut dst);

    // println!("Estimated time: {:?}", instant.elapsed());

    // let mut image = JxlImage::builder()
    //     .pool(JxlThreadPool::none())
    //     .read(std::io::Cursor::new(
    //         fs::read("./assets/input(1).jxl").unwrap(),
    //     ))
    //     .unwrap();
    // image.set_cms(Lcms2);
    //
    // let render = image.render_frame(0).unwrap();
    // let rendered_icc = image.rendered_icc();
    // let image = render.image_all_channels();
    // let img_buf = image.buf();
    //
    // let real_img_data = img_buf
    //     .chunks_exact(5)
    //     .flat_map(|x| [x[0], x[1], x[2], x[3]])
    //     // .map(|x| (x * 65535.0 + 0.5) as u16)
    //     .collect::<Vec<_>>();
    //
    // let jxl_profile = ColorProfile::new_from_slice(&rendered_icc).unwrap();
    // fs::write("./assets/rendered.jxl", rendered_icc).unwrap();
    // println!("jxl_profile: {:?}", jxl_profile.color_space);
    // let mut dst2 = vec![0u16; real_img_data.len()];
    // let transform2 = jxl_profile
    //     .create_transform_16bit(
    //         Layout::Rgba,
    //         &dest_profile,
    //         Layout::Rgba,
    //         TransformOptions::default(),
    //     )
    //     .unwrap();
    //
    // for (src, dst) in real_img_data
    //     .chunks_exact(img.width() as usize * 4)
    //     .zip(dst2.chunks_exact_mut(img.dimensions().0 as usize * 4))
    // {
    //     // ot.transform_pixels(src, dst);
    //
    //     transform2
    //         .transform(
    //             &src[..img.dimensions().0 as usize * 4],
    //             &mut dst[..img.dimensions().0 as usize * 4],
    //         )
    //         .unwrap();
    // }
    //
    // image::save_buffer(
    //     "jx.jpg",
    //     &dst2,
    //     float_image.dimensions().0,
    //     float_image.dimensions().1,
    //     image::ExtendedColorType::Rgb8,
    // )
    // .unwrap();

    // let dst = dst.chunks_exact(4).map(|x| {
    //     [x[0], x[1], x[2], 255]
    // }).flat_map(|x| x).collect::<Vec<u8>>();

    let dst = dst
        .iter()
        .map(|&x| (srgb_from_linear(x as f64) * 255.) as u8)
        .collect::<Vec<_>>();
    image::save_buffer(
        "v_new_satf32.png",
        &dst,
        img.width(),
        img.height(),
        image::ExtendedColorType::Rgba8,
    )
    .unwrap();
}

// fn main() {
//     let us_swop_icc = fs::read("./assets/us_swop_coated.icc").unwrap();
//
//     let width = 5000;
//     let height = 5000;
//
//     let cmyk = vec![0u8; width * height * 4];
//
//     let color_profile = ColorProfile::new_from_slice(&us_swop_icc).unwrap();
//     let dest_profile = ColorProfile::new_srgb();
//     let mut dst = vec![0u8; width * height * 4];
//     let transform = color_profile
//         .create_transform_8bit(
//             Layout::Rgba,
//             &dest_profile,
//             Layout::Rgba,
//             TransformOptions {
//                 interpolation_method: InterpolationMethod::Prism,
//                 ..Default::default()
//             },
//         )
//         .unwrap();
//     transform.transform(&cmyk, &mut dst).unwrap();
// }
