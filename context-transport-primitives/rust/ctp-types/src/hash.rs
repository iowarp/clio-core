// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
// Port of `clio_ctp/types/hash.h` (ctp::hash / ctp::equal_to).
//
// C++ → Rust parity map
// ---------------------
// | C++ name                                             | Rust name                     |
// |------------------------------------------------------|-------------------------------|
// | `ctp::hash<T>::FNV_OFFSET_BASIS`                     | `FNV_OFFSET_BASIS`            |
// | `ctp::hash<T>::FNV_PRIME`                            | `FNV_PRIME`                   |
// | `ctp::hash<T>::fnv1a_hash(data, size)` (generic/GPU) | `fnv1a_hash(&[u8])`           |
// | `ctp::hash<uintN_t> / ctp::hash<intN_t>`             | `CtpHash` impls for u8..=u64, |
// |   (identity cast to `std::size_t`)                   |   i8..=i64, usize, isize      |
// | `ctp::hash<ctp::priv::basic_string<char, ...>>`      | `CtpHash` impls for `str`,    |
// |   (FNV-1a specialization in priv/string.h)           |   `String`, `[u8]`, `Vec<u8>` |
// | `ctp::equal_to<T>`                                   | `equal_to<T: PartialEq>()`    |
//
// Bit-for-bit parity notes (placement / DirectHash relies on this):
// * Strings: the C++ side hashes placement-relevant strings with the explicit
//   64-bit FNV-1a specialization (priv/string.h). `fnv1a_hash` and the string
//   `CtpHash` impls reproduce it exactly on every platform.
// * Integers: the C++ GPU path is `static_cast<std::size_t>(key)` — identity
//   with sign extension for signed types. The Rust impls match that (and
//   therefore libstdc++/libc++ `std::hash`, which is also identity).
//
// Semantic divergences (explicit):
// 1. The C++ CPU path delegates to `std::hash`, which is implementation-
//    defined. libstdc++/libc++ hash integers by identity (matches this port);
//    MSVC's `std::hash` FNV-1a-hashes the value's bytes instead. Cross-
//    language equality is guaranteed against the C++ GPU path, the explicit
//    FNV-1a string specialization, and libstdc++/libc++ integer hashing — but
//    NOT against MSVC-compiled `std::hash<intN_t>`.
// 2. The C++ generic `ctp::hash<T>` (GPU path) hashes the raw object bytes of
//    an arbitrary `T`. Rust has no safe byte reinterpretation, so callers
//    hash an explicit byte view via `fnv1a_hash`; to match the C++ in-memory
//    representation on little-endian hosts, feed `to_le_bytes()` output
//    (see `generic_template_byte_parity` test).
// 3. Hash width is always 64 bits (`u64`). C++ `std::size_t` is 32-bit on
//    32-bit hosts; CTP only targets 64-bit platforms, so we fix the width.
// 4. No pair/tuple hash exists in the C++ header; combined keys (e.g.
//    `MemoryBackendId` in allocator.h) pack fields into a `u64` before
//    hashing, which composes with `CtpHash for u64` unchanged.

/// FNV-1a 64-bit offset basis (C++ `FNV_OFFSET_BASIS`).
pub const FNV_OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;

/// FNV-1a 64-bit prime (C++ `FNV_PRIME`).
pub const FNV_PRIME: u64 = 1_099_511_628_211;

/// 64-bit FNV-1a over a byte slice.
///
/// Bit-for-bit identical to `ctp::hash<T>::fnv1a_hash` (types/hash.h) and to
/// the `ctp::hash` specialization for `ctp::priv::basic_string`
/// (data_structures/priv/string.h). Multiplication wraps modulo 2^64 exactly
/// as unsigned C++ arithmetic does.
#[inline]
pub fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// CTP hash functor (Rust rendering of `ctp::hash<T>`).
///
/// Implementations mirror the deterministic C++ semantics:
/// * integers hash to themselves (sign-extended for signed types), matching
///   `static_cast<std::size_t>(key)`;
/// * strings and byte sequences hash with 64-bit FNV-1a, matching the
///   `priv::basic_string` specialization.
pub trait CtpHash {
    /// Compute the CTP hash of `self`.
    fn ctp_hash(&self) -> u64;
}

macro_rules! impl_ctp_hash_unsigned {
    ($($t:ty),*) => {$(
        impl CtpHash for $t {
            #[inline]
            fn ctp_hash(&self) -> u64 {
                // C++: static_cast<std::size_t>(key) — zero-extends.
                *self as u64
            }
        }
    )*};
}

macro_rules! impl_ctp_hash_signed {
    ($($t:ty),*) => {$(
        impl CtpHash for $t {
            #[inline]
            fn ctp_hash(&self) -> u64 {
                // C++: static_cast<std::size_t>(key) — sign-extends to
                // 64 bits first, then reinterprets as unsigned.
                *self as i64 as u64
            }
        }
    )*};
}

impl_ctp_hash_unsigned!(u8, u16, u32, u64, usize);
impl_ctp_hash_signed!(i8, i16, i32, i64, isize);

impl CtpHash for [u8] {
    #[inline]
    fn ctp_hash(&self) -> u64 {
        fnv1a_hash(self)
    }
}

impl CtpHash for Vec<u8> {
    #[inline]
    fn ctp_hash(&self) -> u64 {
        fnv1a_hash(self)
    }
}

impl CtpHash for str {
    #[inline]
    fn ctp_hash(&self) -> u64 {
        fnv1a_hash(self.as_bytes())
    }
}

impl CtpHash for String {
    #[inline]
    fn ctp_hash(&self) -> u64 {
        fnv1a_hash(self.as_bytes())
    }
}

impl<T: CtpHash + ?Sized> CtpHash for &T {
    #[inline]
    fn ctp_hash(&self) -> u64 {
        (**self).ctp_hash()
    }
}

/// Rust rendering of `ctp::equal_to<T>`: plain `==` comparison.
///
/// Kept as a named function for parity with the C++ functor; idiomatic Rust
/// call sites should just use `PartialEq` directly.
#[inline]
pub fn equal_to<T: PartialEq + ?Sized>(a: &T, b: &T) -> bool {
    a == b
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference values below were computed independently (Python big-int
    // implementation of the C++ loop) and cross-checked against the
    // published FNV-1a 64 test vectors, i.e. by hand-executing the C++
    // algorithm rather than by calling `fnv1a_hash` itself.

    #[test]
    fn fnv1a_empty_is_offset_basis() {
        assert_eq!(fnv1a_hash(b""), FNV_OFFSET_BASIS);
        assert_eq!(fnv1a_hash(b""), 0xcbf2_9ce4_8422_2325);
    }

    #[test]
    fn fnv1a_known_string_vectors() {
        assert_eq!(fnv1a_hash(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a_hash(b"foobar"), 0x8594_4171_f739_67e8);
        assert_eq!(fnv1a_hash(b"hello"), 0xa430_d846_80aa_bd0b);
        assert_eq!(fnv1a_hash(b"chi"), 0xf5fa_f619_0cf9_1825);
        assert_eq!(fnv1a_hash(b"/home/user/data.bin"), 0x3db7_6222_3870_d661);
    }

    #[test]
    fn fnv1a_binary_bytes() {
        // A single NUL byte still mutates the state (XOR with 0, then *prime).
        assert_eq!(fnv1a_hash(&[0u8]), 0xaf63_bd4c_8601_b7df);
        assert_ne!(fnv1a_hash(&[0u8]), fnv1a_hash(b""));
        // High bytes exercise the full wrapping-multiply path.
        assert_eq!(fnv1a_hash(&[0xff; 8]), 0x8cf5_1a8b_fca3_883d);
    }

    #[test]
    fn fnv1a_is_case_sensitive() {
        // The C++ hash has no case folding; "A" and "a" must differ.
        assert_ne!(fnv1a_hash(b"A"), fnv1a_hash(b"a"));
    }

    #[test]
    fn string_impls_agree_with_byte_hash() {
        let s = "placement-key";
        assert_eq!(s.ctp_hash(), fnv1a_hash(s.as_bytes()));
        assert_eq!(String::from(s).ctp_hash(), s.ctp_hash());
        assert_eq!(s.as_bytes().ctp_hash(), s.ctp_hash());
        assert_eq!(s.as_bytes().to_vec().ctp_hash(), s.ctp_hash());
        // Blanket &T impl.
        assert_eq!((&s).ctp_hash(), s.ctp_hash());
    }

    #[test]
    fn unsigned_integers_hash_to_identity() {
        assert_eq!(0u8.ctp_hash(), 0);
        assert_eq!(0xabu8.ctp_hash(), 0xab);
        assert_eq!(u8::MAX.ctp_hash(), 0xff);
        assert_eq!(u16::MAX.ctp_hash(), 0xffff);
        assert_eq!(0xdead_beefu32.ctp_hash(), 0xdead_beef);
        assert_eq!(u32::MAX.ctp_hash(), 0xffff_ffff);
        assert_eq!(u64::MAX.ctp_hash(), u64::MAX);
        assert_eq!(0x0123_4567_89ab_cdefu64.ctp_hash(), 0x0123_4567_89ab_cdef);
        assert_eq!(42usize.ctp_hash(), 42);
    }

    #[test]
    fn signed_integers_sign_extend() {
        // C++: static_cast<std::size_t>((int8_t)-1) == 0xFFFF...FFFF.
        assert_eq!((-1i8).ctp_hash(), u64::MAX);
        assert_eq!((-1i16).ctp_hash(), u64::MAX);
        assert_eq!((-1i32).ctp_hash(), u64::MAX);
        assert_eq!((-1i64).ctp_hash(), u64::MAX);
        assert_eq!(i8::MIN.ctp_hash(), 0xffff_ffff_ffff_ff80);
        assert_eq!(i16::MIN.ctp_hash(), 0xffff_ffff_ffff_8000);
        assert_eq!(i32::MIN.ctp_hash(), 0xffff_ffff_8000_0000);
        assert_eq!(i64::MIN.ctp_hash(), 0x8000_0000_0000_0000);
        assert_eq!(i64::MAX.ctp_hash(), 0x7fff_ffff_ffff_ffff);
        assert_eq!(0i32.ctp_hash(), 0);
        assert_eq!(127i8.ctp_hash(), 127);
        assert_eq!((-5isize).ctp_hash(), (-5i64) as u64);
    }

    #[test]
    fn generic_template_byte_parity() {
        // C++ generic ctp::hash<T> (GPU path) hashes the raw object bytes.
        // On a little-endian host, that byte view equals to_le_bytes().
        // Reference values hand-computed from the C++ loop.
        assert_eq!(
            fnv1a_hash(&0x0123_4567_89ab_cdefu64.to_le_bytes()),
            0x37eb_3f33_4776_1c55
        );
        assert_eq!(
            fnv1a_hash(&0xdead_beefu32.to_le_bytes()),
            0xa44e_2de0_7150_f42b
        );
    }

    #[test]
    fn memory_backend_id_combine_pattern() {
        // allocator.h: hash<MemoryBackendId> = hash<u64>((major << 32) | minor).
        let (major, minor) = (3u32, 7u32);
        let combined = (u64::from(major) << 32) | u64::from(minor);
        assert_eq!(combined.ctp_hash(), 0x0000_0003_0000_0007);
    }

    #[test]
    fn equal_to_matches_cpp_functor() {
        assert!(equal_to(&5u32, &5u32));
        assert!(!equal_to(&5u32, &6u32));
        assert!(equal_to("abc", "abc"));
        assert!(!equal_to("abc", "abd"));
        assert!(equal_to::<[u8]>(&[], &[]));
    }
}
