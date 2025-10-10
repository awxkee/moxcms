#![no_main]

use libfuzzer_sys::fuzz_target;
use moxcms::{ColorProfile, Layout, TransformOptions};

fuzz_target!(|data: &[u8]| {
    // Never panic expected
    let profile = ColorProfile::new_from_slice(data);
    match profile {
        Ok(profile) => {
            let new_srgb = ColorProfile::new_srgb();
            _ = profile.create_transform_8bit(
                Layout::Rgba,
                &new_srgb,
                Layout::Rgb,
                TransformOptions::default(),
            );
            _ = new_srgb.create_transform_8bit(
                Layout::Rgb,
                &profile,
                Layout::Rgb,
                TransformOptions::default(),
            );
        }
        Err(_) => {}
    }
});
