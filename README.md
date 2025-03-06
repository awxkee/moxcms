# Rust ICC management

Fast and safe converting between ICC profiles in pure Rust.

Supports CMYK -> RGBX and RGBX -> RGBX, Gray -> RGBX, RGBX -> GRAY, RGBX -> CMYK

# Example

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

# Benchmarks

### ICC transform 8-bit 

Test made on the image 1997x1331 size

| Conversion        | time(NEON) | Time(AVX2) |
|-------------------|:----------:|:----------:|
| moxcms RGB->RGB   |   3.95ms   |   5.13ms   |
| moxcms RGBA->RGBA |   4.31ms   |   5.87ms   |
| lcms2 RGB->RGB    |   13.1ms   |  27.73ms   |
| lcms2 RGB->RGB    |  21.97ms   |  35.70ms   |
| qcms RGB->RGB     |   6.47ms   |   4.59ms   |
| qcms RGBA->RGBA   |   6.83ms   |   4.99ms   |

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.
