// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
// Core CTP types. Mirrors clio_ctp/types/{numbers,bitfield}.h. The numeric
// aliases exist so FFI signatures and ported code read like their C++
// counterparts; the bitfields mirror ctp::bitfield32_t / ctp::abitfield32_t
// (used pervasively for task flags and container state).

#![deny(unsafe_op_in_unsafe_fn)]

pub mod hash;

use std::sync::atomic::{AtomicU32, Ordering};

// ---------------------------------------------------------------------------
// numbers.h equivalents
// ---------------------------------------------------------------------------
pub type U8 = u8;
pub type U16 = u16;
pub type U32 = u32;
pub type U64 = u64;
pub type I8 = i8;
pub type I16 = i16;
pub type I32 = i32;
pub type I64 = i64;
pub type F32 = f32;
pub type F64 = f64;

/// `BIT_OPT(u32, n)` equivalent: a single-bit mask.
#[inline]
pub const fn bit_opt(n: u32) -> u32 {
    1u32 << n
}

// ---------------------------------------------------------------------------
// bitfield.h equivalents
// ---------------------------------------------------------------------------

/// Plain 32-bit bitfield (mirrors `ctp::bitfield32_t`).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Bitfield32 {
    bits: u32,
}

impl Bitfield32 {
    #[inline]
    pub const fn new(bits: u32) -> Self {
        Self { bits }
    }

    /// Set the given flag bits.
    #[inline]
    pub fn set_bits(&mut self, mask: u32) {
        self.bits |= mask;
    }

    /// Clear the given flag bits.
    #[inline]
    pub fn unset_bits(&mut self, mask: u32) {
        self.bits &= !mask;
    }

    /// True if ANY of the given bits are set (mirrors `Any()`).
    #[inline]
    pub const fn any(&self, mask: u32) -> bool {
        (self.bits & mask) != 0
    }

    /// True if ALL of the given bits are set (mirrors `All()`).
    #[inline]
    pub const fn all(&self, mask: u32) -> bool {
        (self.bits & mask) == mask
    }

    /// Clear every bit (mirrors `Clear()`).
    #[inline]
    pub fn clear(&mut self) {
        self.bits = 0;
    }

    #[inline]
    pub const fn bits(&self) -> u32 {
        self.bits
    }
}

/// Atomic 32-bit bitfield (mirrors `ctp::abitfield32_t`).
///
/// Uses acquire/release ordering: flag publications must be visible to
/// readers on other threads the same way the C++ std::atomic default
/// (seq_cst) guarantees, and AcqRel is sufficient for flag semantics.
#[derive(Debug, Default)]
pub struct AtomicBitfield32 {
    bits: AtomicU32,
}

impl AtomicBitfield32 {
    #[inline]
    pub const fn new(bits: u32) -> Self {
        Self {
            bits: AtomicU32::new(bits),
        }
    }

    #[inline]
    pub fn set_bits(&self, mask: u32) {
        self.bits.fetch_or(mask, Ordering::AcqRel);
    }

    #[inline]
    pub fn unset_bits(&self, mask: u32) {
        self.bits.fetch_and(!mask, Ordering::AcqRel);
    }

    #[inline]
    pub fn any(&self, mask: u32) -> bool {
        (self.bits.load(Ordering::Acquire) & mask) != 0
    }

    #[inline]
    pub fn all(&self, mask: u32) -> bool {
        (self.bits.load(Ordering::Acquire) & mask) == mask
    }

    #[inline]
    pub fn clear(&self) {
        self.bits.store(0, Ordering::Release);
    }

    #[inline]
    pub fn bits(&self) -> u32 {
        self.bits.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitfield_set_unset_any_all() {
        let mut f = Bitfield32::default();
        let a = bit_opt(0);
        let b = bit_opt(5);
        f.set_bits(a | b);
        assert!(f.any(a));
        assert!(f.all(a | b));
        f.unset_bits(a);
        assert!(!f.any(a));
        assert!(f.any(b));
        assert!(!f.all(a | b));
        f.clear();
        assert_eq!(f.bits(), 0);
    }

    #[test]
    fn atomic_bitfield_concurrent_sets() {
        use std::sync::Arc;
        let f = Arc::new(AtomicBitfield32::default());
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let f = Arc::clone(&f);
                std::thread::spawn(move || f.set_bits(bit_opt(i)))
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert!(f.all((0..8).fold(0, |m, i| m | bit_opt(i))));
    }
}
