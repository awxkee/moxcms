#![no_main]

use libfuzzer_sys::fuzz_target;
use moxcms::{ColorProfile, InterpolationMethod, Layout, TransformOptions};

fuzz_target!(|data: (u8, u8, u16, u8, u8, u8, u8, f32)| {
    let src_layout = if data.3 % 2 == 0 {
        Layout::Rgba
    } else {
        Layout::Rgb
    };
    let dst_layout = if data.4 % 2 == 0 {
        Layout::Rgba
    } else {
        Layout::Rgb
    };
    let bp = data.6 % 3;
    let bit_depth = if bp == 0 {
        10
    } else if bp == 1 {
        12
    } else {
        16
    };
    let inter = data.5 % 4;
    let interpolation_method = if inter == 0 {
        InterpolationMethod::Tetrahedral
    } else if inter == 1 {
        InterpolationMethod::Pyramid
    } else if inter == 2 {
        InterpolationMethod::Prism
    } else {
        InterpolationMethod::Linear
    };
    fuzz_8_bit(
        data.0 as usize,
        data.1 as usize,
        (data.2 >> 8) as u8,
        src_layout,
        dst_layout,
        interpolation_method,
    );
    fuzz_8_bit_in_place(
        data.0 as usize,
        data.1 as usize,
        (data.2 >> 8) as u8,
        src_layout,
    );
    fuzz_16_bit(
        data.0 as usize,
        data.1 as usize,
        data.2,
        src_layout,
        dst_layout,
        interpolation_method,
        bit_depth,
    );
});

fn fuzz_8_bit(
    width: usize,
    height: usize,
    px: u8,
    src_layout: Layout,
    dst_layout: Layout,
    interpolation_method: InterpolationMethod,
) {
    if width == 0 || height == 0 {
        return;
    }
    let src_image_rgb = vec![px; width * height * src_layout.channels()];
    let mut dst_image_rgb = vec![px; width * height * dst_layout.channels()];
    let src_profile = ColorProfile::new_srgb();
    let dst_profile = ColorProfile::new_bt2020();
    let transform = src_profile
        .create_transform_8bit(
            src_layout,
            &dst_profile,
            dst_layout,
            TransformOptions {
                interpolation_method,
                ..Default::default()
            },
        )
        .unwrap();
    transform
        .transform(&src_image_rgb, &mut dst_image_rgb)
        .unwrap();
}

fn fuzz_8_bit_in_place(width: usize, height: usize, px: u8, src_layout: Layout) {
    if width == 0 || height == 0 {
        return;
    }
    let mut src_image_rgb = vec![px; width * height * src_layout.channels()];
    let src_profile = ColorProfile::new_srgb();
    let dst_profile = ColorProfile::new_bt2020();
    let transform = src_profile
        .create_in_place_transform_8bit(
            src_layout,
            &dst_profile,
            TransformOptions {
                prefer_fixed_point: true,
                ..Default::default()
            },
        )
        .unwrap();
    transform.transform(&mut src_image_rgb).unwrap();
}

fn fuzz_16_bit(
    width: usize,
    height: usize,
    px: u16,
    src_layout: Layout,
    dst_layout: Layout,
    interpolation_method: InterpolationMethod,
    bp: usize,
) {
    if width == 0 || height == 0 {
        return;
    }
    let src_image_rgb = vec![px; width * height * src_layout.channels()];
    let mut dst_image_rgb = vec![px; width * height * dst_layout.channels()];
    let src_profile = ColorProfile::new_srgb();
    let dst_profile = ColorProfile::new_bt2020();
    let transform = if bp == 10 {
        src_profile
            .create_transform_10bit(
                src_layout,
                &dst_profile,
                dst_layout,
                TransformOptions {
                    interpolation_method,
                    ..Default::default()
                },
            )
            .unwrap()
    } else if bp == 12 {
        src_profile
            .create_transform_12bit(
                src_layout,
                &dst_profile,
                dst_layout,
                TransformOptions {
                    interpolation_method,
                    ..Default::default()
                },
            )
            .unwrap()
    } else {
        src_profile
            .create_transform_16bit(
                src_layout,
                &dst_profile,
                dst_layout,
                TransformOptions {
                    interpolation_method,
                    ..Default::default()
                },
            )
            .unwrap()
    };
    transform
        .transform(&src_image_rgb, &mut dst_image_rgb)
        .unwrap();
}
