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
use crate::math::m_clamp;
use crate::mlaf::mlaf;
use crate::{CmsError, ColorProfile, pow, powf};
use num_traits::AsPrimitive;

#[derive(Clone, Debug)]
pub enum Trc {
    Lut(Vec<u16>),
    Parametric(Vec<f32>),
}

#[allow(clippy::many_single_char_names)]
pub(crate) fn build_srgb_gamma_table(num_entries: i32) -> Vec<u16> {
    let gamma: f64 = 2.4;
    let a: f64 = 1.0 / 1.055;
    let b: f64 = 0.055 / 1.055;
    let c: f64 = 1.0 / 12.92;
    let d: f64 = 0.04045;
    build_parametric_table(num_entries, a, b, c, d, gamma)
}

#[allow(clippy::many_single_char_names)]
#[inline]
pub(crate) fn build_parametric_table(
    num_entries: i32,
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    g: f64,
) -> Vec<u16> {
    build_trc_table(
        num_entries,
        // IEC 61966-2.1 (sRGB)
        // Y = (aX + b)^Gamma | X >= d
        // Y = cX             | X < d
        |x| {
            if x >= d {
                let e: f64 = a * x + b;
                if e > 0. { pow(e, g) } else { 0. }
            } else {
                c * x
            }
        },
    )
}

pub(crate) fn build_trc_table(num_entries: i32, eotf: impl Fn(f64) -> f64) -> Vec<u16> {
    let mut table = vec![0u16; num_entries as usize];

    for (i, table_value) in table.iter_mut().enumerate() {
        let x: f64 = i as f64 / (num_entries - 1) as f64;
        let y: f64 = eotf(x);
        let mut output: f64;
        output = y * 65535.0 + 0.5;
        if output > 65535.0 {
            output = 65535.0
        }
        if output < 0.0 {
            output = 0.0
        }
        *table_value = output.floor() as u16;
    }
    table
}

pub(crate) fn float_to_u8_fixed_8_number(a: f32) -> u16 {
    if a > 255.0 + 255.0 / 256f32 {
        0xffffu16
    } else if a < 0.0 {
        0u16
    } else {
        (a * 256.0 + 0.5).floor() as u16
    }
}

pub(crate) fn curve_from_gamma(gamma: f32) -> Trc {
    Trc::Lut(vec![float_to_u8_fixed_8_number(gamma)])
}

#[derive(Debug)]
struct ParametricCurve {
    g: f32,
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    e: f32,
    f: f32,
}

impl ParametricCurve {
    #[allow(clippy::many_single_char_names)]
    fn new(params: &[f32]) -> Option<ParametricCurve> {
        // convert from the variable number of parameters
        // contained in profiles to a unified representation.
        let g: f32 = params[0];
        match params[1..] {
            [] => Some(ParametricCurve {
                g,
                a: 1.,
                b: 0.,
                c: 1.,
                d: 0.,
                e: 0.,
                f: 0.,
            }),
            [a, b] => Some(ParametricCurve {
                g,
                a,
                b,
                c: 0.,
                d: -b / a,
                e: 0.,
                f: 0.,
            }),
            [a, b, c] => Some(ParametricCurve {
                g,
                a,
                b,
                c: 0.,
                d: -b / a,
                e: c,
                f: c,
            }),
            [a, b, c, d] => Some(ParametricCurve {
                g,
                a,
                b,
                c,
                d,
                e: 0.,
                f: 0.,
            }),
            [a, b, c, d, e, f] => Some(ParametricCurve {
                g,
                a,
                b,
                c,
                d,
                e,
                f,
            }),
            _ => None,
        }
    }

    fn eval(&self, x: f32) -> f32 {
        if x < self.d {
            self.c * x + self.f
        } else {
            powf(self.a * x + self.b, self.g) + self.e
        }
    }

    #[allow(dead_code)]
    #[allow(clippy::many_single_char_names)]
    fn invert(&self) -> Option<ParametricCurve> {
        // First check if the function is continuous at the cross-over point d.
        let d1 = powf(self.a * self.d + self.b, self.g) + self.e;
        let d2 = self.c * self.d + self.f;

        if (d1 - d2).abs() > 0.1 {
            return None;
        }
        let d = d1;

        // y = (a * x + b)^g + e
        // y - e = (a * x + b)^g
        // (y - e)^(1/g) = a*x + b
        // (y - e)^(1/g) - b = a*x
        // (y - e)^(1/g)/a - b/a = x
        // ((y - e)/a^g)^(1/g) - b/a = x
        // ((1/(a^g)) * y - e/(a^g))^(1/g) - b/a = x
        let a = 1. / powf(self.a, self.g);
        let b = -self.e / powf(self.a, self.g);
        let g = 1. / self.g;
        let e = -self.b / self.a;

        // y = c * x + f
        // y - f = c * x
        // y/c - f/c = x
        let (c, f);
        if d <= 0. {
            c = 1.;
            f = 0.;
        } else {
            c = 1. / self.c;
            f = -self.f / self.c;
        }

        // if self.d > 0. and self.c == 0 as is likely with type 1 and 2 parametric function
        // then c and f will not be finite.
        if !(g.is_finite()
            && a.is_finite()
            && b.is_finite()
            && c.is_finite()
            && d.is_finite()
            && e.is_finite()
            && f.is_finite())
        {
            return None;
        }

        Some(ParametricCurve {
            g,
            a,
            b,
            c,
            d,
            e,
            f,
        })
    }
}

#[inline]
fn u8_fixed_8number_to_float(x: u16) -> f32 {
    // 0x0000 = 0.
    // 0x0100 = 1.
    // 0xffff = 255  + 255/256
    (x as i32 as f64 / 256.0) as f32
}

fn passthrough_table<const N: usize, const BIT_DEPTH: usize>() -> Box<[f32; N]> {
    let mut gamma_table = Box::new([0f32; N]);
    let max_value = (1 << BIT_DEPTH) - 1;
    let cap_values = (1u32 << BIT_DEPTH) as usize;
    assert!(cap_values <= N, "Invalid lut table construction");
    let scale_value = 1f64 / max_value as f64;
    for (i, g) in gamma_table.iter_mut().enumerate().take(cap_values) {
        *g = (i as f64 * scale_value) as f32;
    }
    gamma_table
}

fn linear_forward_table<const N: usize, const BIT_DEPTH: usize>(gamma: u16) -> Box<[f32; N]> {
    let mut gamma_table = Box::new([0f32; N]);
    let gamma_float: f32 = u8_fixed_8number_to_float(gamma);
    let max_value = (1 << BIT_DEPTH) - 1;
    let cap_values = (1u32 << BIT_DEPTH) as usize;
    assert!(cap_values <= N, "Invalid lut table construction");
    let scale_value = 1f64 / max_value as f64;
    for (i, g) in gamma_table.iter_mut().enumerate().take(cap_values) {
        *g = pow(i as f64 * scale_value, gamma_float as f64) as f32;
    }
    gamma_table
}

#[inline]
pub(crate) fn lut_interp_linear_float(x: f32, table: &[f32]) -> f32 {
    let value = x * (table.len() - 1) as f32;

    let upper: i32 = value.ceil() as i32;
    let lower: i32 = value.floor() as i32;

    let diff = upper as f32 - value;
    mlaf(
        table[upper as usize] * (1.0f32 - diff),
        table[lower as usize],
        diff,
    )
}

#[inline]
pub(crate) fn lut_interp_linear(input_value: f64, table: &[u16]) -> f32 {
    let mut input_value = input_value;
    if table.is_empty() {
        return input_value as f32;
    }

    input_value *= (table.len() - 1) as f64;

    let upper: i32 = input_value.ceil() as i32;
    let lower: i32 = input_value.floor() as i32;
    let value: f32 = ((table[(upper as usize).min(table.len() - 1)] as f64)
        * (1. - (upper as f64 - input_value))
        + (table[(lower as usize).min(table.len() - 1)] as f64 * (upper as f64 - input_value)))
        as f32;
    /* scale the value */
    value * (1.0 / 65535.0)
}

fn linear_lut_interpolate<const N: usize, const BIT_DEPTH: usize>(table: &[u16]) -> Box<[f32; N]> {
    let mut gamma_table = Box::new([0f32; N]);
    let max_value = (1 << BIT_DEPTH) - 1;
    let cap_values = (1u32 << BIT_DEPTH) as usize;
    assert!(cap_values <= N, "Invalid lut table construction");
    let scale_value = 1f64 / max_value as f64;
    for (i, g) in gamma_table.iter_mut().enumerate().take(cap_values) {
        *g = lut_interp_linear(i as f64 * scale_value, table);
    }
    gamma_table
}

fn linear_curve_parametric<const N: usize, const BIT_DEPTH: usize>(
    params: &[f32],
) -> Option<Box<[f32; N]>> {
    let params = ParametricCurve::new(params)?;
    let mut gamma_table = Box::new([0f32; N]);
    let max_value = (1 << BIT_DEPTH) - 1;
    let scale_value = 1f32 / max_value as f32;
    for (i, g) in gamma_table.iter_mut().enumerate().take(N) {
        let x = i as f32 * scale_value;
        *g = m_clamp(params.eval(x), 0.0, 1.0);
    }
    Some(gamma_table)
}

fn linear_curve_parametric_s<const N: usize>(params: &[f32]) -> Option<Box<[f32; N]>> {
    let params = ParametricCurve::new(params)?;
    let mut gamma_table = Box::new([0f32; N]);
    let scale_value = 1f32 / (N - 1) as f32;
    for (i, g) in gamma_table.iter_mut().enumerate().take(N) {
        let x = i as f32 * scale_value;
        *g = m_clamp(params.eval(x), 0.0, 1.0);
    }
    Some(gamma_table)
}

pub(crate) fn make_gamma_linear_table<
    T: Default + Copy + 'static,
    const BUCKET: usize,
    const N: usize,
    const BIT_DEPTH: usize,
>() -> Box<[T; BUCKET]>
where
    f32: AsPrimitive<T>,
{
    let mut table = Box::new([T::default(); BUCKET]);
    let max_range = (1f64 / (N as f64 / (1 << BIT_DEPTH) as f64)) as f32;
    for (v, output) in table.iter_mut().take(N).enumerate() {
        *output = (v as f32 * max_range).round().as_();
    }
    table
}

#[inline]
fn lut_interp_linear_gamma<T: Default + Copy + 'static, const N: usize, const BIT_DEPTH: usize>(
    input_value: u32,
    table: &[u16],
) -> T
where
    u32: AsPrimitive<T>,
{
    /* Start scaling input_value to the length of the array: PRECACHE_OUTPUT_MAX*(length-1).
     * We'll divide out the PRECACHE_OUTPUT_MAX next */
    let mut value: u32 = input_value * (table.len() - 1) as u32;
    let cap_value = N - 1;
    /* equivalent to ceil(value/PRECACHE_OUTPUT_MAX) */
    let upper: u32 = value.div_ceil(cap_value as u32);
    /* equivalent to floor(value/PRECACHE_OUTPUT_MAX) */
    let lower: u32 = value / cap_value as u32;
    /* interp is the distance from upper to value scaled to 0..PRECACHE_OUTPUT_MAX */
    let interp: u32 = value % cap_value as u32;
    let lw_value = table[lower as usize];
    let hw_value = table[upper as usize];
    /* the table values range from 0..65535 */
    value = hw_value as u32 * interp + lw_value as u32 * ((N - 1) as u32 - interp); // 0..(65535*PRECACHE_OUTPUT_MAX)

    /* round and scale */
    let max_colors = (1 << BIT_DEPTH) - 1;
    value += (cap_value * 65535 / max_colors / 2) as u32; // scale to 0..255
    value /= (cap_value * 65535 / max_colors) as u32;
    value.as_()
}

pub(crate) fn make_gamma_lut<
    T: Default + Copy + 'static,
    const BUCKET: usize,
    const N: usize,
    const BIT_DEPTH: usize,
>(
    table: &[u16],
) -> Box<[T; BUCKET]>
where
    u32: AsPrimitive<T>,
{
    let mut new_table = Box::new([T::default(); BUCKET]);
    for (v, output) in new_table.iter_mut().take(N).enumerate() {
        *output = lut_interp_linear_gamma::<T, N, BIT_DEPTH>(v as u32, table);
    }
    new_table
}

pub(crate) fn lut_interp_linear16(input_value: u16, table: &[u16]) -> u16 {
    /* Start scaling input_value to the length of the array: 65535*(length-1).
     * We'll divide out the 65535 next */
    let mut value: u32 = input_value as u32 * (table.len() as u32 - 1); /* equivalent to ceil(value/65535) */
    let upper: u32 = value.div_ceil(65535); /* equivalent to floor(value/65535) */
    let lower: u32 = value / 65535;
    /* interp is the distance from upper to value scaled to 0..65535 */
    let interp: u32 = value % 65535; // 0..65535*65535
    value = (table[upper as usize] as u32 * interp
        + table[lower as usize] as u32 * (65535 - interp))
        / 65535;
    value as u16
}

fn make_gamma_pow_table<
    T: Default + Copy + 'static,
    const BUCKET: usize,
    const N: usize,
    const BIT_DEPTH: usize,
>(
    gamma: f32,
) -> Box<[T; BUCKET]>
where
    f32: AsPrimitive<T>,
{
    let mut table = Box::new([T::default(); BUCKET]);
    let scale = 1f32 / (N - 1) as f32;
    let cap = ((1 << BIT_DEPTH) - 1) as f32;
    for (v, output) in table.iter_mut().take(N).enumerate() {
        *output = (cap * powf(v as f32 * scale, gamma)).round().as_();
    }
    table
}

fn lut_inverse_interp16(value: u16, lut_table: &[u16]) -> u16 {
    let mut l: i32 = 1; // 'int' Give spacing for negative values
    let mut r: i32 = 0x10000;
    let mut x: i32 = 0;
    let mut res: i32;
    let length = lut_table.len() as i32;

    let mut num_zeroes: i32 = 0;
    while lut_table[num_zeroes as usize] as i32 == 0 && num_zeroes < length - 1 {
        num_zeroes += 1
    }

    if num_zeroes == 0 && value as i32 == 0 {
        return 0u16;
    }
    let mut num_of_polys: i32 = 0;
    while lut_table[(length - 1 - num_of_polys) as usize] as i32 == 0xffff
        && num_of_polys < length - 1
    {
        num_of_polys += 1
    }
    // Does the curve belong to this case?
    if num_zeroes > 1 || num_of_polys > 1 {
        let a_0: i32;
        let b_0: i32;
        // Identify if value fall downto 0 or FFFF zone
        if value as i32 == 0 {
            return 0u16;
        }
        // if (Value == 0xFFFF) return 0xFFFF;
        // else restrict to valid zone
        if num_zeroes > 1 {
            a_0 = (num_zeroes - 1) * 0xffff / (length - 1);
            l = a_0 - 1
        }
        if num_of_polys > 1 {
            b_0 = (length - 1 - num_of_polys) * 0xffff / (length - 1);
            r = b_0 + 1
        }
    }
    if r <= l {
        // If this happens LutTable is not invertible
        return 0u16;
    }

    while r > l {
        x = (l + r) / 2;
        res = lut_interp_linear16((x - 1) as u16, lut_table) as i32;
        if res == value as i32 {
            // Found exact match.
            return (x - 1) as u16;
        }
        if res > value as i32 {
            r = x - 1
        } else {
            l = x + 1
        }
    }

    // Not found, should we interpolate?

    // Get surrounding nodes
    debug_assert!(x >= 1);

    let val2: f64 = (length - 1) as f64 * ((x - 1) as f64 / 65535.0);
    let cell0: i32 = val2.floor() as i32;
    let cell1: i32 = val2.ceil() as i32;
    if cell0 == cell1 {
        return x as u16;
    }

    let y0: f64 = lut_table[cell0 as usize] as f64;
    let x0: f64 = 65535.0 * cell0 as f64 / (length - 1) as f64;
    let y1: f64 = lut_table[cell1 as usize] as f64;
    let x1: f64 = 65535.0 * cell1 as f64 / (length - 1) as f64;
    let a: f64 = (y1 - y0) / (x1 - x0);
    let b: f64 = y0 - a * x0;
    if a.abs() < 0.01f64 {
        return x as u16;
    }
    let f: f64 = (value as i32 as f64 - b) / a;
    if f < 0.0 {
        return 0u16;
    }
    if f >= 65535.0 {
        return 0xffffu16;
    }
    (f + 0.5f64).floor() as u16
}

fn invert_lut(table: &[u16], out_length: usize) -> Vec<u16> {
    /* for now we invert the lut by creating a lut of size out_length
     * and attempting to lookup a value for each entry using lut_inverse_interp16 */
    let mut output = vec![0u16; out_length];
    let scale_value = 65535f64 / (out_length - 1) as f64;
    for (i, out) in output.iter_mut().enumerate() {
        let x: f64 = i as f64 * scale_value;
        let input: u16 = (x + 0.5f64).floor() as u16;
        *out = lut_inverse_interp16(input, table);
    }
    output
}

impl Trc {
    #[inline(always)]
    pub(crate) fn build_linearize_table<const N: usize, const BIT_DEPTH: usize>(
        &self,
    ) -> Option<Box<[f32; N]>> {
        match self {
            Trc::Parametric(params) => linear_curve_parametric::<N, BIT_DEPTH>(params),
            Trc::Lut(data) => match data.len() {
                0 => Some(passthrough_table::<N, BIT_DEPTH>()),
                1 => Some(linear_forward_table::<N, BIT_DEPTH>(data[0])),
                _ => Some(linear_lut_interpolate::<N, BIT_DEPTH>(data)),
            },
        }
    }

    #[inline]
    pub(crate) fn build_gamma_table<
        T: Default + Copy + 'static,
        const BUCKET: usize,
        const N: usize,
        const BIT_DEPTH: usize,
    >(
        &self,
    ) -> Option<Box<[T; BUCKET]>>
    where
        f32: AsPrimitive<T>,
        u32: AsPrimitive<T>,
    {
        match self {
            Trc::Parametric(params) => {
                let mut gamma_table_uint = Box::new([0; N]);

                let inverted_size: usize = N;
                let gamma_table = linear_curve_parametric_s::<N>(params)?;
                for (&src, dst) in gamma_table.iter().zip(gamma_table_uint.iter_mut()) {
                    *dst = (src * 65535f32) as u16;
                }
                let inverted = invert_lut(gamma_table_uint.as_slice(), inverted_size);
                Some(make_gamma_lut::<T, BUCKET, N, BIT_DEPTH>(&inverted))
            }
            Trc::Lut(data) => match data.len() {
                0 => Some(make_gamma_linear_table::<T, BUCKET, N, BIT_DEPTH>()),
                1 => Some(make_gamma_pow_table::<T, BUCKET, N, BIT_DEPTH>(
                    1. / u8_fixed_8number_to_float(data[0]),
                )),
                _ => {
                    let mut inverted_size = data.len();
                    if inverted_size < 256 {
                        inverted_size = 256
                    }
                    let inverted = invert_lut(data, inverted_size);
                    Some(make_gamma_lut::<T, BUCKET, N, BIT_DEPTH>(&inverted))
                }
            },
        }
    }
}

impl ColorProfile {
    /// Produces LUT for 8 bit tone linearization
    pub fn build_8bit_lin_table(&self, trc: &Option<Trc>) -> Result<Box<[f32; 256]>, CmsError> {
        trc.as_ref()
            .and_then(|trc| trc.build_linearize_table::<256, 8>())
            .ok_or(CmsError::BuildTransferFunction)
    }

    /// Produces LUT for Gray transfer curve with N depth
    pub fn build_gray_linearize_table<const N: usize, const BIT_DEPTH: usize>(
        &self,
    ) -> Result<Box<[f32; N]>, CmsError> {
        self.gray_trc
            .as_ref()
            .and_then(|trc| trc.build_linearize_table::<N, BIT_DEPTH>())
            .ok_or(CmsError::BuildTransferFunction)
    }

    /// Produces LUT for Red transfer curve with N depth
    pub fn build_r_linearize_table<const N: usize, const BIT_DEPTH: usize>(
        &self,
    ) -> Result<Box<[f32; N]>, CmsError> {
        self.red_trc
            .as_ref()
            .and_then(|trc| trc.build_linearize_table::<N, BIT_DEPTH>())
            .ok_or(CmsError::BuildTransferFunction)
    }

    /// Produces LUT for Green transfer curve with N depth
    pub fn build_g_linearize_table<const N: usize, const BIT_DEPTH: usize>(
        &self,
    ) -> Result<Box<[f32; N]>, CmsError> {
        self.green_trc
            .as_ref()
            .and_then(|trc| trc.build_linearize_table::<N, BIT_DEPTH>())
            .ok_or(CmsError::BuildTransferFunction)
    }

    /// Produces LUT for Blue transfer curve with N depth
    pub fn build_b_linearize_table<const N: usize, const BIT_DEPTH: usize>(
        &self,
    ) -> Result<Box<[f32; N]>, CmsError> {
        self.blue_trc
            .as_ref()
            .and_then(|trc| trc.build_linearize_table::<N, BIT_DEPTH>())
            .ok_or(CmsError::BuildTransferFunction)
    }

    /// Build gamma table for 8 bit depth
    /// Only 4092 first bins are used and values scaled in 0..255
    pub fn build_8bit_gamma_table(&self, trc: &Option<Trc>) -> Result<Box<[u16; 65536]>, CmsError> {
        self.build_gamma_table::<u16, 65536, 4092, 8>(trc)
    }

    /// Build gamma table for 10 bit depth
    /// Only 8192 first bins are used and values scaled in 0..1023
    pub fn build_10bit_gamma_table(
        &self,
        trc: &Option<Trc>,
    ) -> Result<Box<[u16; 65536]>, CmsError> {
        self.build_gamma_table::<u16, 65536, 8192, 10>(trc)
    }

    /// Build gamma table for 12 bit depth
    /// Only 16384 first bins are used and values scaled in 0..4095
    pub fn build_12bit_gamma_table(
        &self,
        trc: &Option<Trc>,
    ) -> Result<Box<[u16; 65536]>, CmsError> {
        self.build_gamma_table::<u16, 65536, 16384, 12>(trc)
    }

    /// Build gamma table for 16 bit depth
    /// Only 16384 first bins are used and values scaled in 0..65535
    pub fn build_16bit_gamma_table(
        &self,
        trc: &Option<Trc>,
    ) -> Result<Box<[u16; 65536]>, CmsError> {
        self.build_gamma_table::<u16, 65536, 65536, 16>(trc)
    }

    #[inline]
    pub fn build_gamma_table<
        T: Default + Copy + 'static,
        const BUCKET: usize,
        const N: usize,
        const BIT_DEPTH: usize,
    >(
        &self,
        trc: &Option<Trc>,
    ) -> Result<Box<[T; BUCKET]>, CmsError>
    where
        f32: AsPrimitive<T>,
        u32: AsPrimitive<T>,
    {
        trc.as_ref()
            .and_then(|trc| trc.build_gamma_table::<T, BUCKET, N, BIT_DEPTH>())
            .ok_or(CmsError::BuildTransferFunction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_16_bit_parametric() {
        let mut gamma_table_uint: [u16; 65536] = [0; 65536];

        let curve = vec![2.4, 1. / 1.055, 0.055 / 1.055, 1. / 12.92, 0.04045f32];

        let inverted_size: usize = 65536;
        let gamma_table = linear_curve_parametric::<65536, 16>(&curve).unwrap();
        for (&src, dst) in gamma_table.iter().zip(gamma_table_uint.iter_mut()) {
            *dst = (src * 65535f32) as u16;
        }
        let inverted = invert_lut(&gamma_table_uint, inverted_size);
        let value = lut_interp_linear_gamma::<u16, 65536, 16>(65535, &inverted);
        assert_ne!(value, 0);
        let inverted_lut = make_gamma_lut::<u16, 65536, 65536, 16>(&inverted);
        let last100 = &inverted[inverted_lut.len() - 50..inverted_lut.len() - 1];
        for &item in last100.iter() {
            assert_ne!(item, 0);
        }

        let last_inverted_100 = &inverted_lut[inverted_lut.len() - 50..inverted_lut.len() - 1];
        for &item in last_inverted_100.iter() {
            assert_ne!(item, 0);
        }
    }
}
