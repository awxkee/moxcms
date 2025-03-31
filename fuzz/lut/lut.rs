#![no_main]

use libfuzzer_sys::fuzz_target;
use moxcms::{BarycentricWeightScale, ColorProfile, InterpolationMethod, Layout, TransformOptions};
use std::fs;
use std::sync::LazyLock;

static STATIC_US_SWOP: LazyLock<ColorProfile> = LazyLock::new(|| {
    let cmyk_icc = fs::read("./assets/us_swop_coated.icc").unwrap();
    ColorProfile::new_from_slice(&cmyk_icc).unwrap()
});

static STATIC_SRGB_PERCEPTUAL: LazyLock<ColorProfile> = LazyLock::new(|| {
    let cmyk_icc = fs::read("./assets/srgb_perceptual.icc").unwrap();
    ColorProfile::new_from_slice(&cmyk_icc).unwrap()
});

fuzz_target!(|data: (u8, u8, u16, u8, u8, u8, f32, bool, bool)| {
    let dst_layout = if data.3 % 2 == 0 {
        Layout::Rgba
    } else {
        Layout::Rgb
    };
    let bp = data.5 % 3;
    let bit_depth = if bp == 0 {
        10
    } else if bp == 1 {
        12
    } else {
        16
    };
    let barycentric_high = if data.7 {
        BarycentricWeightScale::High
    } else {
        BarycentricWeightScale::Low
    };
    let inter = data.4 % 4;
    let interpolation_method = if inter == 0 {
        InterpolationMethod::Tetrahedral
    } else if inter == 1 {
        InterpolationMethod::Pyramid
    } else if inter == 2 {
        InterpolationMethod::Prism
    } else {
        InterpolationMethod::Linear
    };
    let fixed_point = data.8;
    fuzz_cmyk_8_bit(
        data.0 as usize,
        data.1 as usize,
        (data.2 >> 8) as u8,
        dst_layout,
        interpolation_method,
        barycentric_high,
        fixed_point,
    );
    fuzz_lut_rgb_8_bit(
        data.0 as usize,
        data.1 as usize,
        (data.2 >> 8) as u8,
        dst_layout,
        interpolation_method,
        barycentric_high,
        fixed_point,
    );
    fuzz_lut_f32(
        data.0 as usize,
        data.1 as usize,
        data.6,
        dst_layout,
        interpolation_method,
        barycentric_high,
        fixed_point,
    );
    fuzz_lut_16(
        data.0 as usize,
        data.1 as usize,
        data.2,
        dst_layout,
        interpolation_method,
        barycentric_high,
        bit_depth,
        fixed_point,
    );
});

fn fuzz_lut_f32(
    width: usize,
    height: usize,
    px: f32,
    dst_layout: Layout,
    interpolation_method: InterpolationMethod,
    barycentric_weight_scale: BarycentricWeightScale,
    fixed_point: bool,
) {
    if width == 0 || height == 0 {
        return;
    }

    let src_image_rgb = vec![px; width * height * 4];
    let mut dst_image_rgb = vec![px; width * height * dst_layout.channels()];
    let dst_profile = ColorProfile::new_display_p3();
    let transform = STATIC_US_SWOP
        .create_transform_f32(
            Layout::Rgba,
            &dst_profile,
            dst_layout,
            TransformOptions {
                interpolation_method,
                barycentric_weight_scale,
                prefer_fixed_point: fixed_point,
                ..Default::default()
            },
        )
        .unwrap();
    transform
        .transform(&src_image_rgb, &mut dst_image_rgb)
        .unwrap();
}

fn fuzz_lut_16(
    width: usize,
    height: usize,
    px: u16,
    dst_layout: Layout,
    interpolation_method: InterpolationMethod,
    barycentric_weight_scale: BarycentricWeightScale,
    bit_depth: usize,
    fixed_point: bool,
) {
    if width == 0 || height == 0 {
        return;
    }

    let src_image_rgb = vec![px; width * height * 4];
    let mut dst_image_rgb = vec![px; width * height * dst_layout.channels()];
    let dst_profile = ColorProfile::new_display_p3();
    let transform = if bit_depth == 10 {
        STATIC_US_SWOP
            .create_transform_10bit(
                Layout::Rgba,
                &dst_profile,
                dst_layout,
                TransformOptions {
                    interpolation_method,
                    barycentric_weight_scale,
                    prefer_fixed_point: fixed_point,
                    ..Default::default()
                },
            )
            .unwrap()
    } else if bit_depth == 12 {
        STATIC_US_SWOP
            .create_transform_12bit(
                Layout::Rgba,
                &dst_profile,
                dst_layout,
                TransformOptions {
                    interpolation_method,
                    barycentric_weight_scale,
                    prefer_fixed_point: fixed_point,
                    ..Default::default()
                },
            )
            .unwrap()
    } else {
        STATIC_US_SWOP
            .create_transform_16bit(
                Layout::Rgba,
                &dst_profile,
                dst_layout,
                TransformOptions {
                    interpolation_method,
                    barycentric_weight_scale,
                    prefer_fixed_point: fixed_point,
                    ..Default::default()
                },
            )
            .unwrap()
    };
    transform
        .transform(&src_image_rgb, &mut dst_image_rgb)
        .unwrap();
}

fn fuzz_cmyk_8_bit(
    width: usize,
    height: usize,
    px: u8,
    dst_layout: Layout,
    interpolation_method: InterpolationMethod,
    barycentric_weight_scale: BarycentricWeightScale,
    fixed_point: bool,
) {
    if width == 0 || height == 0 {
        return;
    }

    let src_image_rgb = vec![px; width * height * 4];
    let mut dst_image_rgb = vec![px; width * height * dst_layout.channels()];
    let dst_profile = ColorProfile::new_srgb();
    let transform = STATIC_US_SWOP
        .create_transform_8bit(
            Layout::Rgba,
            &dst_profile,
            dst_layout,
            TransformOptions {
                interpolation_method,
                barycentric_weight_scale,
                prefer_fixed_point: fixed_point,
                ..Default::default()
            },
        )
        .unwrap();
    transform
        .transform(&src_image_rgb, &mut dst_image_rgb)
        .unwrap();
}

fn fuzz_lut_rgb_8_bit(
    width: usize,
    height: usize,
    px: u8,
    dst_layout: Layout,
    interpolation_method: InterpolationMethod,
    barycentric_weight_scale: BarycentricWeightScale,
    fixed_point: bool,
) {
    if width == 0 || height == 0 {
        return;
    }

    let src_image_rgb = vec![px; width * height * 4];
    let mut dst_image_rgb = vec![px; width * height * dst_layout.channels()];
    let dst_profile = ColorProfile::new_display_p3();
    let transform = STATIC_SRGB_PERCEPTUAL
        .create_transform_8bit(
            Layout::Rgba,
            &dst_profile,
            dst_layout,
            TransformOptions {
                interpolation_method,
                barycentric_weight_scale,
                prefer_fixed_point: fixed_point,
                ..Default::default()
            },
        )
        .unwrap();
    transform
        .transform(&src_image_rgb, &mut dst_image_rgb)
        .unwrap();
}
