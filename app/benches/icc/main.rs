/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use criterion::{Criterion, criterion_group, criterion_main};
use lcms2::{Intent, PixelFormat, Profile, Transform};
use moxcms::{ColorProfile, Layout, TransformOptions};
use std::fs;
use std::fs::File;
use std::io::BufReader;
use zune_jpeg::JpegDecoder;

pub fn criterion_benchmark(c: &mut Criterion) {
    let f_str = "../assets/bench.jpg";
    let file = File::open(f_str).expect("Failed to open file");
    let reader = BufReader::new(file);
    let mut jpeg_reader = JpegDecoder::new(reader);
    jpeg_reader.decode_headers().unwrap();
    let src_icc_profile = jpeg_reader.icc_profile().unwrap();

    let us_swop_icc = fs::read("../assets/us_swop_coated.icc").unwrap();

    let cmyk_profile = Profile::new_icc(&us_swop_icc).unwrap();
    let srgb_profile = Profile::new_srgb();

    let img = image::ImageReader::open(f_str).unwrap().decode().unwrap();
    let rgb = img.to_rgb8();
    let rgba = img.to_rgba8();

    let mut cmyk = vec![0u8; rgba.len()];
    let t = Transform::new(
        &srgb_profile,
        PixelFormat::RGBA_8,
        &cmyk_profile,
        PixelFormat::CMYK_8,
        Intent::Perceptual,
    )
    .unwrap();
    t.transform_pixels(&rgba, &mut cmyk);

    c.bench_function("moxcms: RGB -> RGB", |b| {
        let color_profile = ColorProfile::new_from_slice(&src_icc_profile).unwrap();
        let dest_profile = ColorProfile::new_srgb();
        let mut dst = vec![0u8; rgb.len()];
        let transform = color_profile
            .create_transform_8bit(
                Layout::Rgb,
                &dest_profile,
                Layout::Rgb,
                TransformOptions {
                    prefer_fixed_point: true,
                    ..Default::default()
                },
            )
            .unwrap();
        b.iter(|| {
            transform.transform(&rgb, &mut dst).unwrap();
        })
    });

    c.bench_function("moxcms: inplace RGB -> RGB", |b| {
        let color_profile = ColorProfile::new_from_slice(&src_icc_profile).unwrap();
        let dest_profile = ColorProfile::new_srgb();
        let mut dst = vec![0u8; rgb.len()];
        let transform = color_profile
            .create_in_place_transform_8bit(
                Layout::Rgb,
                &dest_profile,
                TransformOptions {
                    prefer_fixed_point: true,
                    ..Default::default()
                },
            )
            .unwrap();
        b.iter(|| {
            transform.transform(&mut dst).unwrap();
        })
    });

    c.bench_function("moxcms: RGBA -> RGBA", |b| {
        let color_profile = ColorProfile::new_from_slice(&src_icc_profile).unwrap();
        let dest_profile = ColorProfile::new_srgb();
        let mut dst = vec![0u8; rgba.len()];
        let transform = color_profile
            .create_transform_8bit(
                Layout::Rgba,
                &dest_profile,
                Layout::Rgba,
                TransformOptions {
                    prefer_fixed_point: true,
                    ..Default::default()
                },
            )
            .unwrap();
        b.iter(|| {
            transform.transform(&rgba, &mut dst).unwrap();
        })
    });

    c.bench_function("moxcms: in_place RGBA -> RGBA", |b| {
        let color_profile = ColorProfile::new_from_slice(&src_icc_profile).unwrap();
        let dest_profile = ColorProfile::new_srgb();
        let mut dst = vec![0u8; rgba.len()];
        let transform = color_profile
            .create_in_place_transform_8bit(
                Layout::Rgba,
                &dest_profile,
                TransformOptions {
                    prefer_fixed_point: true,
                    ..Default::default()
                },
            )
            .unwrap();
        b.iter(|| {
            transform.transform(&mut dst).unwrap();
        })
    });

    c.bench_function("lcms2: RGB -> RGB", |b| {
        let custom_profile = Profile::new_icc(&src_icc_profile).unwrap();
        let profile_bytes = fs::read("../assets/bt_2020.icc").unwrap();
        let dest_profile = Profile::new_icc(&profile_bytes).unwrap();
        let mut dst = vec![0u8; rgb.len()];
        let t = Transform::new(
            &custom_profile,
            PixelFormat::RGB_8,
            &dest_profile,
            PixelFormat::RGB_8,
            Intent::Perceptual,
        )
        .unwrap();

        b.iter(|| {
            t.transform_pixels(&rgb, &mut dst);
        })
    });

    c.bench_function("lcms2: RGBA -> RGBA", |b| {
        let custom_profile = Profile::new_icc(&src_icc_profile).unwrap();
        let profile_bytes = fs::read("../assets/bt_2020.icc").unwrap();
        let dest_profile = Profile::new_icc(&profile_bytes).unwrap();
        let mut dst = vec![0u8; rgba.len()];
        let t = Transform::new(
            &custom_profile,
            PixelFormat::RGBA_8,
            &dest_profile,
            PixelFormat::RGBA_8,
            Intent::Perceptual,
        )
        .unwrap();

        b.iter(|| {
            t.transform_pixels(&rgba, &mut dst);
        })
    });

    c.bench_function("moxcms: RGB16 -> RGB16", |b| {
        let rgb = rgb
            .iter()
            .map(|&x| u16::from_ne_bytes([x, x]))
            .collect::<Vec<_>>();
        let color_profile = ColorProfile::new_from_slice(&src_icc_profile).unwrap();
        let dest_profile = ColorProfile::new_srgb();
        let mut dst = vec![0u16; rgb.len()];
        let transform = color_profile
            .create_transform_16bit(
                Layout::Rgb,
                &dest_profile,
                Layout::Rgb,
                TransformOptions {
                    prefer_fixed_point: false,
                    ..Default::default()
                },
            )
            .unwrap();
        b.iter(|| {
            transform.transform(&rgb, &mut dst).unwrap();
        })
    });

    c.bench_function("moxcms: Fixed RGB16 -> RGB16", |b| {
        let rgb = rgb
            .iter()
            .map(|&x| u16::from_ne_bytes([x, x]))
            .collect::<Vec<_>>();
        let color_profile = ColorProfile::new_from_slice(&src_icc_profile).unwrap();
        let dest_profile = ColorProfile::new_srgb();
        let mut dst = vec![0u16; rgb.len()];
        let transform = color_profile
            .create_transform_16bit(
                Layout::Rgb,
                &dest_profile,
                Layout::Rgb,
                TransformOptions {
                    prefer_fixed_point: true,
                    ..Default::default()
                },
            )
            .unwrap();
        b.iter(|| {
            transform.transform(&rgb, &mut dst).unwrap();
        })
    });

    c.bench_function("qcms: RGB -> RGB", |b| {
        let custom_profile = qcms::Profile::new_from_slice(&src_icc_profile, false).unwrap();
        let profile_bytes = fs::read("../assets/bt_2020.icc").unwrap();
        let mut srgb_profile = qcms::Profile::new_from_slice(&profile_bytes, false).unwrap();
        let mut dst = vec![0u8; rgb.len()];
        srgb_profile.precache_output_transform();
        let xfm = qcms::Transform::new(
            &custom_profile,
            &srgb_profile,
            qcms::DataType::RGB8,
            qcms::Intent::default(),
        )
        .unwrap();

        b.iter(|| {
            xfm.convert(&rgb, &mut dst);
        })
    });

    c.bench_function("qcms: RGBA -> RGBA", |b| {
        let custom_profile = qcms::Profile::new_from_slice(&src_icc_profile, false).unwrap();
        let profile_bytes = fs::read("../assets/bt_2020.icc").unwrap();
        let mut srgb_profile = qcms::Profile::new_from_slice(&profile_bytes, false).unwrap();
        let mut dst = vec![0u8; rgba.len()];
        srgb_profile.precache_output_transform();
        let xfm = qcms::Transform::new(
            &custom_profile,
            &srgb_profile,
            qcms::DataType::RGBA8,
            qcms::Intent::default(),
        )
        .unwrap();

        b.iter(|| {
            xfm.convert(&rgba, &mut dst);
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
