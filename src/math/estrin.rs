/*
 * // Copyright (c) Radzivon Bartoshyk 4/2025. All rights reserved.
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

#![allow(unused)]

#[allow(unused_macros)]
macro_rules! poly2 {
    ($x:expr, $c1:expr, $c0:expr) => {
        c_mlaf($x, $c1, $c0)
    };
}
pub(crate) use poly2;

#[allow(unused_macros)]
macro_rules! poly3 {
    ($x:expr, $x2:expr, $c2:expr, $c1:expr, $c0:expr) => {
        c_mlaf($x2, $c2, poly2!($x, $c1, $c0))
    };
}

pub(crate) use poly3;

#[allow(unused_macros)]
macro_rules! poly4 {
    ($x:expr, $x2:expr, $c3:expr, $c2:expr, $c1:expr, $c0:expr) => {
        c_mlaf($x2, poly2!($x, $c3, $c2), poly2!($x, $c1, $c0))
    };
}

pub(crate) use poly4;

#[allow(unused_macros)]
macro_rules! poly5 {
    ($x:expr, $x2:expr, $x4:expr, $c4:expr, $c3:expr, $c2:expr, $c1:expr, $c0:expr) => {
        c_mlaf($x4, $c4, poly4!($x, $x2, $c3, $c2, $c1, $c0))
    };
}

pub(crate) use poly5;

#[allow(unused_macros)]
macro_rules! poly6 {
    ($x:expr, $x2:expr, $x4:expr, $c5:expr, $c4:expr, $c3:expr, $c2:expr, $c1:expr, $c0:expr) => {
        c_mlaf(
            $x4,
            poly2!($x, $c5, $c4),
            poly4!($x, $x2, $c3, $c2, $c1, $c0),
        )
    };
}

pub(crate) use poly6;

#[allow(unused_macros)]
macro_rules! poly7 {
    ($x:expr, $x2:expr, $x4:expr, $c6:expr, $c5:expr, $c4:expr, $c3:expr, $c2:expr, $c1:expr, $c0:expr) => {
        c_mlaf(
            $x4,
            poly3!($x, $x2, $c6, $c5, $c4),
            poly4!($x, $x2, $c3, $c2, $c1, $c0),
        )
    };
}

pub(crate) use poly7;

#[allow(unused_macros)]
macro_rules! poly8 {
    ($x:expr, $x2:expr, $x4:expr, $c7:expr, $c6:expr, $c5:expr, $c4:expr, $c3:expr, $c2:expr, $c1:expr, $c0:expr) => {
        c_mlaf(
            $x4,
            poly4!($x, $x2, $c7, $c6, $c5, $c4),
            poly4!($x, $x2, $c3, $c2, $c1, $c0),
        )
    };
}

pub(crate) use poly8;

#[allow(unused_macros)]
macro_rules! poly9 {
    ($x:expr, $x2:expr, $x4:expr, $x8:expr, $c8:expr, $c7:expr, $c6:expr, $c5:expr, $c4:expr, $c3:expr, $c2:expr, $c1:expr, $c0:expr) => {
        c_mlaf(
            $x8,
            $c8,
            poly8!($x, $x2, $x4, $c7, $c6, $c5, $c4, $c3, $c2, $c1, $c0),
        )
    };
}

pub(crate) use poly9;

#[allow(unused_macros)]
macro_rules! poly10 {
    ($x:expr, $x2:expr, $x4:expr, $x8:expr, $c9:expr, $c8:expr, $c7:expr, $c6:expr, $c5:expr, $c4:expr, $c3:expr, $c2:expr, $c1:expr, $c0:expr) => {
        c_mlaf(
            $x8,
            poly2!($x, $c9, $c8),
            poly8!($x, $x2, $x4, $c7, $c6, $c5, $c4, $c3, $c2, $c1, $c0),
        )
    };
}

pub(crate) use poly10;

#[allow(unused_macros)]
macro_rules! poly11 {
    ($x:expr, $x2:expr, $x4:expr, $x8:expr, $ca:expr, $c9:expr, $c8:expr, $c7:expr, $c6:expr, $c5:expr, $c4:expr, $c3:expr, $c2:expr, $c1:expr, $c0:expr) => {
        c_mlaf(
            $x8,
            poly3!($x, $x2, $ca, $c9, $c8),
            poly8!($x, $x2, $x4, $c7, $c6, $c5, $c4, $c3, $c2, $c1, $c0),
        )
    };
}

pub(crate) use poly11;
