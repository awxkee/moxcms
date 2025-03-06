#![no_main]

use libfuzzer_sys::fuzz_target;
use moxcms::{ColorProfile, Layout, RenderingIntent, TransformOptions};

fuzz_target!(|data: (u8, u8, u16, u8, u8,)| {
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
    fuzz_8_bit(
        data.0 as usize,
        data.1 as usize,
        (data.2 >> 8) as u8,
        src_layout,
        dst_layout,
    );
    fuzz_16_bit(
        data.0 as usize,
        data.1 as usize,
        data.2,
        src_layout,
        dst_layout,
    );
});

fn fuzz_8_bit(width: usize, height: usize, px: u8, src_layout: Layout, dst_layout: Layout) {
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
                rendering_intent: RenderingIntent::Perceptual,
            },
        )
        .unwrap();
    transform
        .transform(&src_image_rgb, &mut dst_image_rgb)
        .unwrap();
}

fn fuzz_16_bit(width: usize, height: usize, px: u16, src_layout: Layout, dst_layout: Layout) {
    if width == 0 || height == 0 {
        return;
    }
    let src_image_rgb = vec![px; width * height * src_layout.channels()];
    let mut dst_image_rgb = vec![px; width * height * dst_layout.channels()];
    let src_profile = ColorProfile::new_srgb();
    let dst_profile = ColorProfile::new_bt2020();
    let transform = src_profile
        .create_transform_16bit(
            src_layout,
            &dst_profile,
            dst_layout,
            TransformOptions {
                rendering_intent: RenderingIntent::Perceptual,
            },
        )
        .unwrap();
    transform
        .transform(&src_image_rgb, &mut dst_image_rgb)
        .unwrap();
}
