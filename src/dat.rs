/*
 * // Copyright (c) Radzivon Bartoshyk 3/2025. All rights reserved.
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
use crate::CmsError;
use crate::writer::write_u16_be;
use chrono::{Datelike, Timelike, Utc};

#[repr(C)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Default)]
pub struct ColorDateTime {
    pub year: u16,
    pub month: u16,
    pub day_of_the_month: u16,
    pub hours: u16,
    pub minutes: u16,
    pub seconds: u16,
}

impl ColorDateTime {
    /// Parses slice for date time
    pub fn new_from_slice(slice: &[u8]) -> Result<ColorDateTime, CmsError> {
        if slice.len() != 12 {
            return Err(CmsError::InvalidProfile);
        }
        let year = u16::from_be_bytes([slice[0], slice[1]]);
        let month = u16::from_be_bytes([slice[2], slice[3]]);
        let day_of_the_month = u16::from_be_bytes([slice[4], slice[5]]);
        let hours = u16::from_be_bytes([slice[6], slice[7]]);
        let minutes = u16::from_be_bytes([slice[8], slice[9]]);
        let seconds = u16::from_be_bytes([slice[10], slice[11]]);
        Ok(ColorDateTime {
            year,
            month,
            day_of_the_month,
            hours,
            minutes,
            seconds,
        })
    }

    /// Creates a new `ColorDateTime` from the current system time (UTC)
    pub fn now() -> Self {
        let now = Utc::now();
        Self {
            year: now.year() as u16,
            month: now.month() as u16,
            day_of_the_month: now.day() as u16,
            hours: now.hour() as u16,
            minutes: now.minute() as u16,
            seconds: now.second() as u16,
        }
    }

    #[inline]
    pub(crate) fn encode(&self, into: &mut Vec<u8>) {
        let year = self.year;
        let month = self.month;
        let day_of_the_month = self.day_of_the_month;
        let hours = self.hours;
        let minutes = self.minutes;
        let seconds = self.seconds;
        write_u16_be(into, year);
        write_u16_be(into, month);
        write_u16_be(into, day_of_the_month);
        write_u16_be(into, hours);
        write_u16_be(into, minutes);
        write_u16_be(into, seconds);
    }
}
