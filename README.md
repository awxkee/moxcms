# Rust ICC Management

Fast and safe conversion between ICC profiles; in pure Rust.

Supports CMYK⬌RGBX, RGBX⬌RGBX, RGBX⬌GRAY, LAB⬌RGBX and CMYK⬌LAB, GRAY⬌RGB, any 3/4 color profiles to RGB and vice versa. Also supports almost any to any Display Class ICC profiles up to 16 inks.

## Example

```rust
let f_str = "./assets/dci_p3_profile.jpeg";
let file = File::open(f_str).expect("Failed to open file");

let img = image::ImageReader::open(f_str).unwrap().decode().unwrap();
let rgb = img.to_rgb8();

let mut decoder = JpegDecoder::new(BufReader::new(file)).unwrap();
let icc = decoder.icc_profile().unwrap().unwrap();
let color_profile = ColorProfile::new_from_slice(&icc).unwrap();
let dest_profile = ColorProfile::new_srgb();
let transform = color_profile
    .create_transform_8bit(&dest_profile, Layout::Rgb8, TransformOptions::default())
    .unwrap();
let mut dst = vec![0u8; rgb.len()];

for (src, dst) in rgb
    .chunks_exact(img.width() as usize * 3)
    .zip(dst.chunks_exact_mut(img.dimensions().0 as usize * 3))
{
    transform
        .transform(
            &src[..img.dimensions().0 as usize * 3],
            &mut dst[..img.dimensions().0 as usize * 3],
        )
        .unwrap();
}
image::save_buffer(
    "v1.jpg",
    &dst,
    img.dimensions().0,
    img.dimensions().1,
    image::ExtendedColorType::Rgb8,
)
    .unwrap();
```

## License

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.
