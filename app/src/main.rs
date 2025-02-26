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
use image::codecs::jpeg::JpegDecoder;
use image::{GenericImageView, ImageDecoder, RgbImage};
use jxl_oxide::{AllocTracker, JxlImage, JxlThreadPool};
use lcms2::{Intent, Profile, Transform};
use moxcms::{ColorProfile, Layout};
use std::fs;
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;
use turbojpeg::PixelFormat;

pub fn cmyk_to_rgb(c: f32, m: f32, y: f32, k: f32) -> (u8, u8, u8) {
    let r = (1.0 - c) * (1.0 - k);
    let g = (1.0 - m) * (1.0 - k);
    let b = (1.0 - y) * (1.0 - k);

    // Convert to 8-bit (0-255) range
    (
        (r * 255.0).round() as u8,
        (g * 255.0).round() as u8,
        (b * 255.0).round() as u8,
    )
}

fn main() {
    let f_str = "./assets/mountains_cmyk.jpg";
    let file = File::open(f_str).expect("Failed to open file");
    let image = JxlImage::builder()
        .pool(JxlThreadPool::none())
        .read(std::io::Cursor::new(fs::read("./assets/test.jxl").unwrap()))
        .unwrap();

    let render = image.render_frame(0).unwrap();
    let rendered_icc = image.rendered_icc();
    let image = render.image_all_channels();
    let float_image = RgbImage::from_raw(
        image.width() as u32,
        image.height() as u32,
        image
            .buf()
            .iter()
            .map(|x| x * 255. + 0.5)
            .map(|x| x as u8)
            .collect::<Vec<_>>(),
    )
    .unwrap();

    let img = image::ImageReader::open(f_str).unwrap().decode().unwrap();
    let rgb = img.to_rgb8();

    let mut decoder = JpegDecoder::new(BufReader::new(file)).unwrap();
    let icc = decoder.icc_profile().unwrap().unwrap();
    // let color_profile = ColorProfile::new_from_slice(&icc).unwrap();
    let dest_profile = ColorProfile::new_srgb();
    // let transform = color_profile
    //     .create_transform_8bit(&dest_profile, Layout::Rgb8)
    //     .unwrap();
    // let mut dst = vec![0u8; rgb.len()];
    // let mut cmyk = vec![0u8; img.width() as usize * img.height() as usize * 4];
    //
    // let custom_profile = Profile::new_icc(&icc).unwrap();
    //
    // let srgb_profile = Profile::new_srgb();
    //
    // let t = Transform::new(
    //     &srgb_profile,
    //     lcms2::PixelFormat::RGB_8,
    //     &custom_profile,
    //     lcms2::PixelFormat::CMYK_8,
    //     Intent::Perceptual,
    // )
    // .unwrap();
    // // let ot = Transform::new(&custom_profile, lcms2::PixelFormat::CMYK_8, &srgb_profile, lcms2::PixelFormat::RGB_8, Intent::Perceptual).unwrap();
    //
    // for (dst, src) in cmyk
    //     .chunks_exact_mut(img.width() as usize * 4)
    //     .zip(rgb.chunks_exact(img.dimensions().0 as usize * 3))
    // {
    //     t.transform_pixels(src, dst);
    // }
    // // Pixel struct must have layout compatible with PixelFormat specified in new()
    //
    // let instant = Instant::now();
    //
    // for (src, dst) in cmyk
    //     .chunks_exact(img.width() as usize * 4)
    //     .zip(dst.chunks_exact_mut(img.dimensions().0 as usize * 3))
    // {
    //     // ot.transform_pixels(src, dst);
    //
    //     transform
    //         .transform(
    //             &src[..img.dimensions().0 as usize * 4],
    //             &mut dst[..img.dimensions().0 as usize * 3],
    //         )
    //         .unwrap();
    // }
    // println!("Exec time {:?}", instant.elapsed());

    let jxl_profile = ColorProfile::new_from_slice(&rendered_icc).unwrap();
    let mut dst2 = vec![0u8; float_image.len()];
    let transform2 = jxl_profile
        .create_transform_8bit(&dest_profile, Layout::Rgb8)
        .unwrap();

    for (src, dst) in float_image
        .chunks_exact(float_image.width() as usize * 3)
        .zip(dst2.chunks_exact_mut(float_image.dimensions().0 as usize * 3))
    {
        // ot.transform_pixels(src, dst);

        transform2
            .transform(
                &src[..float_image.dimensions().0 as usize * 3],
                &mut dst[..float_image.dimensions().0 as usize * 3],
            )
            .unwrap();
    }

    image::save_buffer(
        "jx.jpg",
        &dst2,
        float_image.dimensions().0,
        float_image.dimensions().1,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();

    // image::save_buffer(
    //     "v1.jpg",
    //     &dst,
    //     img.dimensions().0,
    //     img.dimensions().1,
    //     image::ExtendedColorType::Rgb8,
    // )
    // .unwrap();
}
