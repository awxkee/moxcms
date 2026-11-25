/*
 * // Copyright (c) Radzivon Bartoshyk 6/2025. All rights reserved.
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
use crate::LutWarehouse;

impl LutWarehouse {
    /// Method tests if mathematical fusion on LUT table is allowed.
    /// If it's not, full brute-force pass in [Katana] is required.
    #[cfg(feature = "any_to_any")]
    pub(crate) fn is_katana_required(&self) -> bool {
        match self {
            LutWarehouse::Lut(lut) => {
                let input_channels = lut.num_input_channels as usize;
                let input_entries = lut.num_input_table_entries as usize;
                let output_channels = lut.num_output_channels as usize;
                let output_entries = lut.num_output_table_entries as usize;

                if input_channels
                    .checked_mul(input_entries)
                    .is_some_and(|needed| needed <= lut.input_table.len())
                {
                    for i in 0..input_channels {
                        if lut.input_table.is_degenerated(input_entries, i) {
                            return true;
                        }
                        if !lut.input_table.is_monotonic(input_entries, i) {
                            return true;
                        }
                        if lut.input_table.have_discontinuities(input_entries, i) {
                            return true;
                        }
                    }
                }

                if output_channels
                    .checked_mul(output_entries)
                    .is_some_and(|needed| needed <= lut.output_table.len())
                {
                    for i in 0..output_channels {
                        if lut.output_table.is_degenerated(output_entries, i) {
                            return true;
                        }
                        if !lut.output_table.is_monotonic(output_entries, i) {
                            return true;
                        }
                        if lut.output_table.have_discontinuities(output_entries, i) {
                            return true;
                        }
                    }
                }

                false
            }
            LutWarehouse::Multidimensional(mab) => {
                for curve in mab.a_curves.iter() {
                    if curve.is_degenerated() {
                        return true;
                    }
                    if !curve.is_monotonic() {
                        return true;
                    }
                    if curve.have_discontinuities() {
                        return true;
                    }
                }

                for curve in mab.m_curves.iter() {
                    if curve.is_degenerated() {
                        return true;
                    }
                    if !curve.is_monotonic() {
                        return true;
                    }
                    if curve.have_discontinuities() {
                        return true;
                    }
                }

                for curve in mab.b_curves.iter() {
                    if curve.is_degenerated() {
                        return true;
                    }
                    if !curve.is_monotonic() {
                        return true;
                    }
                    if curve.have_discontinuities() {
                        return true;
                    }
                }

                false
            }
        }
    }
}

#[cfg(all(test, feature = "any_to_any"))]
mod tests {
    use crate::profile::{LutDataType, LutType};
    use crate::{LutStore, LutWarehouse, Matrix3d};

    #[test]
    fn lut16_two_entry_tables_do_not_panic_or_false_trigger_katana() {
        let base = LutDataType {
            num_input_channels: 4,
            num_output_channels: 3,
            num_clut_grid_points: 2,
            matrix: Matrix3d::IDENTITY,
            num_input_table_entries: 2,
            num_output_table_entries: 2,
            // Monotonically ascending across all 4 channels (len = 8 < 4 * 4).
            input_table: LutStore::Store16(vec![
                0, 10000, 10000, 20000, 20000, 40000, 40000, 65535,
            ]),
            clut_table: LutStore::Store16([0u16, 32768, 32768].repeat(16)),
            output_table: LutStore::Store16(vec![0, 65535, 0, 65535, 0, 65535]),
            lut_type: LutType::Lut16,
        };
        assert!(!LutWarehouse::Lut(base.clone()).is_katana_required());

        // Standard 2-entry identity tables [0, 65535] per channel should not trigger Katana.
        let identity = LutDataType {
            input_table: LutStore::Store16(vec![0, 65535, 0, 65535, 0, 65535, 0, 65535]),
            ..base
        };
        assert!(!LutWarehouse::Lut(identity).is_katana_required());
    }

    #[test]
    fn lut8_degenerated_later_channel_triggers_katana() {
        let identity: Vec<u8> = (0..=255).collect();
        let mut degenerated = identity.clone();
        degenerated[..50].fill(0);

        let mut input = Vec::with_capacity(256 * 4);
        input.extend_from_slice(&identity);
        input.extend_from_slice(&degenerated);
        input.extend_from_slice(&identity);
        input.extend_from_slice(&identity);

        let lut = LutDataType {
            num_input_channels: 4,
            num_output_channels: 3,
            num_clut_grid_points: 2,
            matrix: Matrix3d::IDENTITY,
            num_input_table_entries: 256,
            num_output_table_entries: 256,
            input_table: LutStore::Store8(input),
            clut_table: LutStore::Store8([0u8, 128, 128].repeat(16)),
            output_table: LutStore::Store8(identity.repeat(3)),
            lut_type: LutType::Lut8,
        };
        assert!(LutWarehouse::Lut(lut).is_katana_required());
    }
}
