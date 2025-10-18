#![no_main]

use libfuzzer_sys::fuzz_target;
use moxcms::{ColorProfile, Layout, TransformOptions};

fuzz_target!(|data: &[u8]| {
    // Never panic expected
    let profile = ColorProfile::new_from_slice(data);

    const ALL_LAYOUTS: &[Layout] = &[
        Layout::Rgb,
        Layout::Rgba,
        Layout::Gray,
        Layout::GrayAlpha,
        Layout::Inks5,
        Layout::Inks6,
        Layout::Inks7,
        Layout::Inks8,
        Layout::Inks9,
        Layout::Inks10,
        Layout::Inks11,
        Layout::Inks12,
        Layout::Inks13,
        Layout::Inks14,
        Layout::Inks15,
    ];

    match profile {
        Ok(profile) => {
            let new_srgb = ColorProfile::new_srgb();
            for &src_layout in ALL_LAYOUTS {
                for &dst_layout in ALL_LAYOUTS {
                    _ = profile.create_transform_8bit(
                        src_layout,
                        &profile,
                        dst_layout,
                        TransformOptions::default(),
                    );
                    _ = new_srgb.create_transform_8bit(
                        src_layout,
                        &profile,
                        dst_layout,
                        TransformOptions::default(),
                    );
                }
            }
        }
        Err(_) => {}
    }
});
