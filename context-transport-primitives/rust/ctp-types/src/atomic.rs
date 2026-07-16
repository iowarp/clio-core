// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
//! Atomic wrappers — port of `clio_ctp/types/atomic.h` (`ctp::ipc`).
//!
//! The C++ header exists to give three storage strategies **one API**, so that
//! locks, allocators and ring buffers can be written once and instantiated
//! either atomically or not (`opt_atomic<T, is_atomic>`), on host or device.
//! This module keeps that shape: [`Atomic<T>`] and [`Nonatomic<T>`] both
//! implement [`AtomicApi<T>`], so generic code compiles against either.
//!
//! # C++ → Rust name mapping
//!
//! | C++ (`ctp::ipc`) | Rust (`ctp_types::atomic`) | Notes |
//! |---|---|---|
//! | `nonatomic<T>` | [`Nonatomic<T>`] | `Cell`-like value with the atomic API; memory orders ignored |
//! | `std_atomic<T>` | [`Atomic<T>`] | wraps the `std::sync::atomic::Atomic*` matching `T` |
//! | `atomic<T>` (host alias of `std_atomic<T>`) | [`Atomic<T>`] | host is the only Rust target; see divergences |
//! | `rocm_atomic<T>` (device alias of `atomic<T>`) | — | not ported; kernels stay CUDA C++ (`ctp-gpu`) |
//! | `atomic_ref<T>` | [`AtomicRef<'a, T>`] | atomic ops over pre-existing plain storage |
//! | `opt_atomic<T, is_atomic>` | [`OptAtomic<IS_ATOMIC>`] + [`OptAtomicSelect`] | `std::conditional` → associated-type selector |
//! | `x` (public member) | private; [`AtomicApi::load`] / [`AtomicApi::get_mut`] | |
//! | `T ctor` / `operator=(U)` | `From<T>` / [`AtomicApi::store`]`(v, SeqCst)` | |
//! | `operator T()` | [`AtomicApi::load`]`(SeqCst)` | no implicit conversions in Rust |
//! | `load(order)` / `store(v, order)` | [`AtomicApi::load`] / [`AtomicApi::store`] | order is explicit (no default args) |
//! | `exchange(v, order)` | [`AtomicApi::exchange`] | returns the old value (see divergences) |
//! | `fetch_add` / `fetch_sub` | [`AtomicApi::fetch_add`] / [`AtomicApi::fetch_sub`] | |
//! | `compare_exchange_weak/strong(T &expected, U desired, order)` | [`AtomicApi::compare_exchange_weak`] / [`AtomicApi::compare_exchange_strong`] | `&mut T` expected, `bool` result — same protocol |
//! | `operator&= \| \|= \| ^=` | [`AtomicApi::fetch_and`] / [`fetch_or`](AtomicApi::fetch_or) / [`fetch_xor`](AtomicApi::fetch_xor) | return the previous value |
//! | `operator++ / --` (pre) | [`AtomicApi::pre_inc`] / [`AtomicApi::pre_dec`] | return the NEW value |
//! | `operator++ / --` (post) | [`AtomicApi::post_inc`] / [`AtomicApi::post_dec`] | return the OLD value |
//! | `operator+ - & \| ^` (non-mutating) | `a.load(o) + n`, … | see divergences |
//! | `operator== / !=` | `PartialEq<T>` / `PartialEq<Self>` | compares loaded values, as C++ does |
//! | `ref()` | [`AtomicApi::get_mut`] | `&mut self` instead of aliasing `T&` |
//! | `fetch_add_system` / `fetch_sub_system` | [`AtomicApi::fetch_add_system`] / [`AtomicApi::fetch_sub_system`] | |
//! | `store_system` / `load_system` / `load_device` | [`AtomicApi::store_system`] / [`load_system`](AtomicApi::load_system) / [`load_device`](AtomicApi::load_device) | |
//! | `or_system` | [`AtomicApi::or_system`] | |
//! | `compare_exchange_strong_system` | [`AtomicApi::compare_exchange_strong_system`] | |
//! | `threadfence()` | [`threadfence`] | `fence(Release)`, as the host branch |
//! | `threadfence_system()` | [`threadfence_system`] | `fence(SeqCst)`, as the host branch |
//! | `shfl_sync_u64()` | [`shfl_sync_u64`] | host fallback only (returns `val`) |
//! | `serialize` / `save` / `load` (cereal) | — | see divergences |
//!
//! # Semantic divergences from the C++
//!
//! 1. **No device backend.** `rocm_atomic<T>` and the `#if CTP_IS_GPU` branches
//!    (`atomicAdd`, `atomicCAS`, `atomicExch`, `__threadfence*`, `__shfl_sync`)
//!    are NOT ported: per `MIGRATION.md` GPU kernels remain CUDA C++ behind
//!    `ctp-gpu`, and Rust never runs the device pass. [`Atomic<T>`] therefore
//!    always corresponds to the C++ **host** selection (`atomic<T>` =
//!    `std_atomic<T>`), and the `*_system` / `*_device` methods reproduce the
//!    C++ **host** definitions exactly (plain `seq_cst` ops for `std_atomic`,
//!    plain non-atomic ops for `nonatomic`). [`threadfence`] /
//!    [`threadfence_system`] / [`shfl_sync_u64`] likewise port only the host
//!    branch — `shfl_sync_u64` is an identity function here, as it is on host.
//! 2. **Scope methods are not stronger than host C++.** Cross-device (CPU↔GPU)
//!    visibility beyond what `SeqCst` gives is unrepresentable in Rust std;
//!    the names are kept so ported call sites read identically, and so a future
//!    GPU-aware backend can specialize them.
//! 3. **`T` is restricted to the integer primitives** with std atomic
//!    counterparts (`u8..u64`, `i8..i64`, `usize`, `isize`), via the sealed
//!    [`AtomicPrimitive`] trait. The C++ template also accepts `float`, `bool`,
//!    enums and pointers; those have no uniform std atomic RMW surface and are
//!    deliberately out of scope. `T`'s std atomic must exist on the target
//!    (`target_has_atomic`), otherwise the impl is `cfg`-ed out.
//! 4. **`exchange` returns the old value** (`T`). The C++ is internally
//!    inconsistent here: `nonatomic::exchange` and `std_atomic::exchange`
//!    return `void` (the `std::atomic::exchange` result is discarded) while
//!    `rocm_atomic::exchange` returns `T`. Rust follows the `rocm_atomic` /
//!    `std::atomic` contract, which strictly adds information.
//! 5. **Post-increment/decrement are correct here.** C++
//!    `nonatomic::operator++(int)` and `std_atomic::operator++(int)` are
//!    `return atomic(x + 1);` — they return `x+1` and **never mutate `x`**
//!    (only `rocm_atomic`'s versions actually increment). [`AtomicApi::post_inc`]
//!    implements the intended semantics: increment, return the OLD value.
//!    Ported call sites relying on the C++ host bug must be re-read.
//! 6. **Overflow wraps, always.** C++ `nonatomic` does `x += count` — UB on
//!    signed overflow — while `std::atomic` is defined as modular. Both Rust
//!    types use wrapping arithmetic (the `std::atomic` rule), so nothing is UB
//!    and nothing panics in debug builds.
//! 7. **Single-order CAS.** The C++ passes one `memory_order` to
//!    `compare_exchange_*`, letting the STL derive the failure order. Rust
//!    requires both, so the C++ derivation is reproduced explicitly
//!    (`Release → Relaxed`, `AcqRel → Acquire`, else unchanged); this also
//!    prevents the panic Rust raises on an invalid failure order.
//! 8. **Non-mutating operators dropped.** C++ `operator+`, `-`, `&`, `|`, `^`
//!    return a *new wrapper* holding `load() OP n`. In Rust that would silently
//!    construct atomics; write `a.load(order) + n` instead.
//! 9. **`ref()` is `get_mut(&mut self)`.** The C++ hands out a `T&` aliasing
//!    live atomic storage (UB to use concurrently). Rust requires exclusive
//!    access, which is the only sound reading of that API.
//! 10. **No cereal `serialize`/`save`/`load`.** `ctp-types` has no serde
//!     dependency (and none may be added); serialize `load()`/`store()` values
//!     at the call site — the C++ `save`/`load` do exactly that (relaxed).
//! 11. **No `const` constructor for the generic types.** `Atomic::<T>::new` is
//!     a trait-dispatched call, so it cannot be `const fn`; `static`s needing
//!     const init should use `std::sync::atomic` types directly.
//! 12. **`Nonatomic<T>` is `!Sync`** (it is `Cell`-shaped). This is the honest
//!     encoding of "provides the API of an atomic, without being atomic": the
//!     C++ type is freely shareable and races are the caller's problem.
//! 13. **`ShmSafe` is not implemented** even though both types satisfy the
//!     shared-memory contract in `MEMORY_DESIGN.md` (layout-identical to `T`,
//!     no `Drop`, no pointers/references, position independent): the marker
//!     lives in `ctp-memory`, which depends on this crate, and `ctp-types` may
//!     not depend back. `ctp-memory` should add the impls (or re-export a
//!     marker) when it wires these types in.

use std::cell::Cell;
use std::fmt;
use std::sync::atomic::{fence, Ordering};

#[cfg(target_has_atomic = "16")]
use std::sync::atomic::{AtomicI16, AtomicU16};
#[cfg(target_has_atomic = "32")]
use std::sync::atomic::{AtomicI32, AtomicU32};
#[cfg(target_has_atomic = "64")]
use std::sync::atomic::{AtomicI64, AtomicU64};
#[cfg(target_has_atomic = "8")]
use std::sync::atomic::{AtomicI8, AtomicU8};
#[cfg(target_has_atomic = "ptr")]
use std::sync::atomic::{AtomicIsize, AtomicUsize};

mod sealed {
    pub trait Sealed {}
}

/// The C++ single-`memory_order` `compare_exchange_*` overloads let the STL
/// derive the failure order from the success order; Rust demands both. This is
/// that derivation (C++20 [atomics.types.operations]): the failure order is the
/// success order with the release component stripped.
#[inline]
const fn failure_order(order: Ordering) -> Ordering {
    match order {
        Ordering::Release => Ordering::Relaxed,
        Ordering::AcqRel => Ordering::Acquire,
        other => other,
    }
}

// ---------------------------------------------------------------------------
// AtomicPrimitive: T → its std::sync::atomic counterpart
// ---------------------------------------------------------------------------

/// A primitive that has a `std::sync::atomic` counterpart on this target.
///
/// Mirrors the implicit "arithmetic `T`" constraint of the C++ templates.
/// Sealed: [`AtomicRef`] relies on the size/bit-validity relationship between
/// `Self` and `Self::Repr` that std guarantees only for its own atomics.
pub trait AtomicPrimitive:
    sealed::Sealed + Copy + Default + PartialEq + Eq + fmt::Debug + Send + Sync + 'static
{
    /// The `std::sync::atomic` type with the same layout as `Self`.
    type Repr: fmt::Debug + Send + Sync;

    /// Additive identity (`T{}` in C++).
    const ZERO: Self;
    /// The `1` used by the increment/decrement operators.
    const ONE: Self;

    fn atomic_new(val: Self) -> Self::Repr;

    /// # Safety
    /// `ptr` must be valid and aligned for `Self::Repr` for `'a`, and for `'a`
    /// the pointee must be accessed only through the returned atomic.
    unsafe fn atomic_from_ptr<'a>(ptr: *mut Self) -> &'a Self::Repr;

    fn atomic_get_mut(a: &mut Self::Repr) -> &mut Self;
    fn atomic_load(a: &Self::Repr, order: Ordering) -> Self;
    fn atomic_store(a: &Self::Repr, val: Self, order: Ordering);
    fn atomic_swap(a: &Self::Repr, val: Self, order: Ordering) -> Self;
    fn atomic_fetch_add(a: &Self::Repr, val: Self, order: Ordering) -> Self;
    fn atomic_fetch_sub(a: &Self::Repr, val: Self, order: Ordering) -> Self;
    fn atomic_fetch_and(a: &Self::Repr, val: Self, order: Ordering) -> Self;
    fn atomic_fetch_or(a: &Self::Repr, val: Self, order: Ordering) -> Self;
    fn atomic_fetch_xor(a: &Self::Repr, val: Self, order: Ordering) -> Self;
    fn atomic_compare_exchange_weak(
        a: &Self::Repr,
        current: Self,
        new: Self,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Self, Self>;
    fn atomic_compare_exchange(
        a: &Self::Repr,
        current: Self,
        new: Self,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Self, Self>;

    // Plain (non-atomic) arithmetic used by `Nonatomic` and by the
    // increment/decrement defaults. Wrapping, per divergence 6.
    fn wrapping_add(self, other: Self) -> Self;
    fn wrapping_sub(self, other: Self) -> Self;
    fn bit_and(self, other: Self) -> Self;
    fn bit_or(self, other: Self) -> Self;
    fn bit_xor(self, other: Self) -> Self;
}

macro_rules! impl_atomic_primitive {
    ($prim:ty, $atom:ty) => {
        impl sealed::Sealed for $prim {}

        impl AtomicPrimitive for $prim {
            type Repr = $atom;

            const ZERO: Self = 0;
            const ONE: Self = 1;

            #[inline]
            fn atomic_new(val: Self) -> Self::Repr {
                <$atom>::new(val)
            }

            #[inline]
            unsafe fn atomic_from_ptr<'a>(ptr: *mut Self) -> &'a Self::Repr {
                // SAFETY: the caller guarantees `ptr` is valid and aligned for
                // `$atom` for `'a` and that the pointee is touched only through
                // the returned atomic for `'a` — exactly `from_ptr`'s contract.
                // `$atom` has the same size and bit validity as `$prim`.
                unsafe { <$atom>::from_ptr(ptr) }
            }

            #[inline]
            fn atomic_get_mut(a: &mut Self::Repr) -> &mut Self {
                a.get_mut()
            }

            #[inline]
            fn atomic_load(a: &Self::Repr, order: Ordering) -> Self {
                a.load(order)
            }

            #[inline]
            fn atomic_store(a: &Self::Repr, val: Self, order: Ordering) {
                a.store(val, order)
            }

            #[inline]
            fn atomic_swap(a: &Self::Repr, val: Self, order: Ordering) -> Self {
                a.swap(val, order)
            }

            #[inline]
            fn atomic_fetch_add(a: &Self::Repr, val: Self, order: Ordering) -> Self {
                a.fetch_add(val, order)
            }

            #[inline]
            fn atomic_fetch_sub(a: &Self::Repr, val: Self, order: Ordering) -> Self {
                a.fetch_sub(val, order)
            }

            #[inline]
            fn atomic_fetch_and(a: &Self::Repr, val: Self, order: Ordering) -> Self {
                a.fetch_and(val, order)
            }

            #[inline]
            fn atomic_fetch_or(a: &Self::Repr, val: Self, order: Ordering) -> Self {
                a.fetch_or(val, order)
            }

            #[inline]
            fn atomic_fetch_xor(a: &Self::Repr, val: Self, order: Ordering) -> Self {
                a.fetch_xor(val, order)
            }

            #[inline]
            fn atomic_compare_exchange_weak(
                a: &Self::Repr,
                current: Self,
                new: Self,
                success: Ordering,
                failure: Ordering,
            ) -> Result<Self, Self> {
                a.compare_exchange_weak(current, new, success, failure)
            }

            #[inline]
            fn atomic_compare_exchange(
                a: &Self::Repr,
                current: Self,
                new: Self,
                success: Ordering,
                failure: Ordering,
            ) -> Result<Self, Self> {
                a.compare_exchange(current, new, success, failure)
            }

            #[inline]
            fn wrapping_add(self, other: Self) -> Self {
                <$prim>::wrapping_add(self, other)
            }

            #[inline]
            fn wrapping_sub(self, other: Self) -> Self {
                <$prim>::wrapping_sub(self, other)
            }

            #[inline]
            fn bit_and(self, other: Self) -> Self {
                self & other
            }

            #[inline]
            fn bit_or(self, other: Self) -> Self {
                self | other
            }

            #[inline]
            fn bit_xor(self, other: Self) -> Self {
                self ^ other
            }
        }
    };
}

#[cfg(target_has_atomic = "8")]
impl_atomic_primitive!(u8, AtomicU8);
#[cfg(target_has_atomic = "8")]
impl_atomic_primitive!(i8, AtomicI8);
#[cfg(target_has_atomic = "16")]
impl_atomic_primitive!(u16, AtomicU16);
#[cfg(target_has_atomic = "16")]
impl_atomic_primitive!(i16, AtomicI16);
#[cfg(target_has_atomic = "32")]
impl_atomic_primitive!(u32, AtomicU32);
#[cfg(target_has_atomic = "32")]
impl_atomic_primitive!(i32, AtomicI32);
#[cfg(target_has_atomic = "64")]
impl_atomic_primitive!(u64, AtomicU64);
#[cfg(target_has_atomic = "64")]
impl_atomic_primitive!(i64, AtomicI64);
#[cfg(target_has_atomic = "ptr")]
impl_atomic_primitive!(usize, AtomicUsize);
#[cfg(target_has_atomic = "ptr")]
impl_atomic_primitive!(isize, AtomicIsize);

// ---------------------------------------------------------------------------
// AtomicApi: the one API both storage strategies provide
// ---------------------------------------------------------------------------

/// The common surface of `nonatomic<T>` / `atomic<T>`.
///
/// Generic code (locks, ring buffers, allocators) takes `A: AtomicApi<T>` and
/// instantiates with [`Atomic<T>`] or [`Nonatomic<T>`] — the Rust counterpart
/// of instantiating a C++ template with `opt_atomic<T, is_atomic>`.
///
/// All methods take `&self`: `Nonatomic` gets interior mutability from `Cell`
/// so the two types stay substitutable, exactly as the C++ non-const members
/// are on a shared object.
pub trait AtomicApi<T: AtomicPrimitive>: From<T> + Default + Sized {
    /// `load(order)`.
    fn load(&self, order: Ordering) -> T;
    /// `store(val, order)`; also the C++ `operator=(U)` (which is `seq_cst`).
    fn store(&self, val: T, order: Ordering);
    /// `exchange(val, order)`, returning the previous value (divergence 4).
    fn exchange(&self, val: T, order: Ordering) -> T;
    /// `fetch_add(count, order)`, returning the previous value. Wraps.
    fn fetch_add(&self, count: T, order: Ordering) -> T;
    /// `fetch_sub(count, order)`, returning the previous value. Wraps.
    fn fetch_sub(&self, count: T, order: Ordering) -> T;
    /// `operator&=`, returning the previous value.
    fn fetch_and(&self, mask: T, order: Ordering) -> T;
    /// `operator|=`, returning the previous value.
    fn fetch_or(&self, mask: T, order: Ordering) -> T;
    /// `operator^=`, returning the previous value.
    fn fetch_xor(&self, mask: T, order: Ordering) -> T;

    /// `compare_exchange_weak(T &expected, desired, order)`.
    ///
    /// Returns `true` on success. On failure (including a spurious one)
    /// `*expected` is overwritten with the value actually read, matching C++.
    fn compare_exchange_weak(&self, expected: &mut T, desired: T, order: Ordering) -> bool;

    /// `compare_exchange_strong(T &expected, desired, order)`. See
    /// [`compare_exchange_weak`](AtomicApi::compare_exchange_weak).
    fn compare_exchange_strong(&self, expected: &mut T, desired: T, order: Ordering) -> bool;

    /// `ref()` — exclusive access to the underlying value (divergence 9).
    fn get_mut(&mut self) -> &mut T;

    // -- scope variants: the C++ HOST definitions (divergences 1 & 2) --

    /// `fetch_add_system(count)`.
    #[inline]
    fn fetch_add_system(&self, count: T) -> T {
        self.fetch_add(count, Ordering::SeqCst)
    }

    /// `fetch_sub_system(count)`.
    #[inline]
    fn fetch_sub_system(&self, count: T) -> T {
        self.fetch_sub(count, Ordering::SeqCst)
    }

    /// `store_system(val)`.
    #[inline]
    fn store_system(&self, val: T) {
        self.store(val, Ordering::SeqCst)
    }

    /// `load_system()`.
    #[inline]
    fn load_system(&self) -> T {
        self.load(Ordering::SeqCst)
    }

    /// `load_device()`.
    #[inline]
    fn load_device(&self) -> T {
        self.load(Ordering::SeqCst)
    }

    /// `or_system(other)`. Returns the previous value (the C++ returns `*this`).
    #[inline]
    fn or_system(&self, mask: T) -> T {
        self.fetch_or(mask, Ordering::SeqCst)
    }

    /// `compare_exchange_strong_system(expected, desired, order)`.
    #[inline]
    fn compare_exchange_strong_system(
        &self,
        expected: &mut T,
        desired: T,
        order: Ordering,
    ) -> bool {
        self.compare_exchange_strong(expected, desired, order)
    }

    // -- increment / decrement (divergence 5) --

    /// `++x` — increments, returns the NEW value.
    #[inline]
    fn pre_inc(&self) -> T {
        self.fetch_add(T::ONE, Ordering::SeqCst).wrapping_add(T::ONE)
    }

    /// `x++` — increments, returns the OLD value.
    #[inline]
    fn post_inc(&self) -> T {
        self.fetch_add(T::ONE, Ordering::SeqCst)
    }

    /// `--x` — decrements, returns the NEW value.
    #[inline]
    fn pre_dec(&self) -> T {
        self.fetch_sub(T::ONE, Ordering::SeqCst).wrapping_sub(T::ONE)
    }

    /// `x--` — decrements, returns the OLD value.
    #[inline]
    fn post_dec(&self) -> T {
        self.fetch_sub(T::ONE, Ordering::SeqCst)
    }
}

// ---------------------------------------------------------------------------
// Nonatomic<T> — `ctp::ipc::nonatomic<T>`
// ---------------------------------------------------------------------------

/// "Provides the API of an atomic, without being atomic" (`nonatomic<T>`).
///
/// Layout-identical to `T` (`repr(transparent)` over `Cell<T>`), no `Drop`, no
/// pointers — shared-memory placeable per `MEMORY_DESIGN.md` (see divergence
/// 13). Memory orders are accepted and ignored, as in C++. `!Sync`
/// (divergence 12).
#[repr(transparent)]
#[derive(Default)]
pub struct Nonatomic<T: AtomicPrimitive> {
    x: Cell<T>,
}

impl<T: AtomicPrimitive> Nonatomic<T> {
    /// `nonatomic(U def)`.
    #[inline]
    pub const fn new(val: T) -> Self {
        Self { x: Cell::new(val) }
    }
}

impl<T: AtomicPrimitive> AtomicApi<T> for Nonatomic<T> {
    #[inline]
    fn load(&self, _order: Ordering) -> T {
        self.x.get()
    }

    #[inline]
    fn store(&self, val: T, _order: Ordering) {
        self.x.set(val);
    }

    #[inline]
    fn exchange(&self, val: T, _order: Ordering) -> T {
        self.x.replace(val)
    }

    #[inline]
    fn fetch_add(&self, count: T, _order: Ordering) -> T {
        let orig = self.x.get();
        self.x.set(orig.wrapping_add(count));
        orig
    }

    #[inline]
    fn fetch_sub(&self, count: T, _order: Ordering) -> T {
        let orig = self.x.get();
        self.x.set(orig.wrapping_sub(count));
        orig
    }

    #[inline]
    fn fetch_and(&self, mask: T, _order: Ordering) -> T {
        let orig = self.x.get();
        self.x.set(orig.bit_and(mask));
        orig
    }

    #[inline]
    fn fetch_or(&self, mask: T, _order: Ordering) -> T {
        let orig = self.x.get();
        self.x.set(orig.bit_or(mask));
        orig
    }

    #[inline]
    fn fetch_xor(&self, mask: T, _order: Ordering) -> T {
        let orig = self.x.get();
        self.x.set(orig.bit_xor(mask));
        orig
    }

    #[inline]
    fn compare_exchange_weak(&self, expected: &mut T, desired: T, order: Ordering) -> bool {
        self.compare_exchange_strong(expected, desired, order)
    }

    #[inline]
    fn compare_exchange_strong(&self, expected: &mut T, desired: T, _order: Ordering) -> bool {
        let cur = self.x.get();
        if cur == *expected {
            self.x.set(desired);
            true
        } else {
            *expected = cur;
            false
        }
    }

    #[inline]
    fn get_mut(&mut self) -> &mut T {
        self.x.get_mut()
    }
}

impl<T: AtomicPrimitive> From<T> for Nonatomic<T> {
    #[inline]
    fn from(val: T) -> Self {
        Self::new(val)
    }
}

impl<T: AtomicPrimitive> Clone for Nonatomic<T> {
    /// `nonatomic(const nonatomic &other)`.
    #[inline]
    fn clone(&self) -> Self {
        Self::new(self.x.get())
    }
}

impl<T: AtomicPrimitive> fmt::Debug for Nonatomic<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Nonatomic").field(&self.x.get()).finish()
    }
}

impl<T: AtomicPrimitive> PartialEq<T> for Nonatomic<T> {
    #[inline]
    fn eq(&self, other: &T) -> bool {
        self.x.get() == *other
    }
}

impl<T: AtomicPrimitive> PartialEq for Nonatomic<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.x.get() == other.x.get()
    }
}

// ---------------------------------------------------------------------------
// Atomic<T> — `ctp::ipc::std_atomic<T>` (== host `ctp::ipc::atomic<T>`)
// ---------------------------------------------------------------------------

/// A real atomic (`std_atomic<T>`; the host spelling of `atomic<T>`).
///
/// Layout-identical to `T` (`repr(transparent)` over the matching
/// `std::sync::atomic` type, which std guarantees has `T`'s size and bit
/// validity), no `Drop` — shared-memory placeable per `MEMORY_DESIGN.md`
/// (see divergence 13), which is what makes it usable for in-segment
/// cross-process state.
#[repr(transparent)]
pub struct Atomic<T: AtomicPrimitive> {
    x: T::Repr,
}

impl<T: AtomicPrimitive> Atomic<T> {
    /// `std_atomic(U def)`.
    #[inline]
    pub fn new(val: T) -> Self {
        Self {
            x: T::atomic_new(val),
        }
    }

    /// Borrow the underlying `std::sync::atomic` type, for code that wants the
    /// std API directly (e.g. `fetch_update`, `compare_exchange` as `Result`).
    #[inline]
    pub fn inner(&self) -> &T::Repr {
        &self.x
    }
}

impl<T: AtomicPrimitive> AtomicApi<T> for Atomic<T> {
    #[inline]
    fn load(&self, order: Ordering) -> T {
        T::atomic_load(&self.x, order)
    }

    #[inline]
    fn store(&self, val: T, order: Ordering) {
        T::atomic_store(&self.x, val, order);
    }

    #[inline]
    fn exchange(&self, val: T, order: Ordering) -> T {
        T::atomic_swap(&self.x, val, order)
    }

    #[inline]
    fn fetch_add(&self, count: T, order: Ordering) -> T {
        T::atomic_fetch_add(&self.x, count, order)
    }

    #[inline]
    fn fetch_sub(&self, count: T, order: Ordering) -> T {
        T::atomic_fetch_sub(&self.x, count, order)
    }

    #[inline]
    fn fetch_and(&self, mask: T, order: Ordering) -> T {
        T::atomic_fetch_and(&self.x, mask, order)
    }

    #[inline]
    fn fetch_or(&self, mask: T, order: Ordering) -> T {
        T::atomic_fetch_or(&self.x, mask, order)
    }

    #[inline]
    fn fetch_xor(&self, mask: T, order: Ordering) -> T {
        T::atomic_fetch_xor(&self.x, mask, order)
    }

    #[inline]
    fn compare_exchange_weak(&self, expected: &mut T, desired: T, order: Ordering) -> bool {
        match T::atomic_compare_exchange_weak(
            &self.x,
            *expected,
            desired,
            order,
            failure_order(order),
        ) {
            Ok(_) => true,
            Err(actual) => {
                *expected = actual;
                false
            }
        }
    }

    #[inline]
    fn compare_exchange_strong(&self, expected: &mut T, desired: T, order: Ordering) -> bool {
        match T::atomic_compare_exchange(&self.x, *expected, desired, order, failure_order(order)) {
            Ok(_) => true,
            Err(actual) => {
                *expected = actual;
                false
            }
        }
    }

    #[inline]
    fn get_mut(&mut self) -> &mut T {
        T::atomic_get_mut(&mut self.x)
    }
}

impl<T: AtomicPrimitive> Default for Atomic<T> {
    #[inline]
    fn default() -> Self {
        Self::new(T::ZERO)
    }
}

impl<T: AtomicPrimitive> From<T> for Atomic<T> {
    #[inline]
    fn from(val: T) -> Self {
        Self::new(val)
    }
}

impl<T: AtomicPrimitive> Clone for Atomic<T> {
    /// `std_atomic(const std_atomic &other) : x(other.x.load())`.
    #[inline]
    fn clone(&self) -> Self {
        Self::new(self.load(Ordering::SeqCst))
    }
}

impl<T: AtomicPrimitive> fmt::Debug for Atomic<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Atomic")
            .field(&self.load(Ordering::SeqCst))
            .finish()
    }
}

impl<T: AtomicPrimitive> PartialEq<T> for Atomic<T> {
    #[inline]
    fn eq(&self, other: &T) -> bool {
        self.load(Ordering::SeqCst) == *other
    }
}

impl<T: AtomicPrimitive> PartialEq for Atomic<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.load(Ordering::SeqCst) == other.load(Ordering::SeqCst)
    }
}

// ---------------------------------------------------------------------------
// opt_atomic<T, is_atomic>
// ---------------------------------------------------------------------------

/// Selector tag for [`OptAtomicSelect`] — the `is_atomic` bool of the C++
/// `opt_atomic<T, is_atomic>` alias.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OptAtomic<const IS_ATOMIC: bool>;

/// `std::conditional<is_atomic, atomic<T>, nonatomic<T>>` as an associated type.
///
/// Rust has no type-level `if`, so the choice is made by an impl per `bool`:
///
/// ```
/// use std::sync::atomic::Ordering;
/// use ctp_types::atomic::{AtomicApi, OptAtomic, OptAtomicSelect};
///
/// struct Queue<const IS_ATOMIC: bool>
/// where
///     OptAtomic<IS_ATOMIC>: OptAtomicSelect<u64>,
/// {
///     head: <OptAtomic<IS_ATOMIC> as OptAtomicSelect<u64>>::Type,
/// }
///
/// let q = Queue::<true> { head: 7u64.into() };
/// assert_eq!(q.head.fetch_add(1, Ordering::SeqCst), 7);
/// ```
pub trait OptAtomicSelect<T: AtomicPrimitive> {
    /// [`Atomic<T>`] when `IS_ATOMIC`, else [`Nonatomic<T>`].
    type Type: AtomicApi<T>;
}

impl<T: AtomicPrimitive> OptAtomicSelect<T> for OptAtomic<true> {
    type Type = Atomic<T>;
}

impl<T: AtomicPrimitive> OptAtomicSelect<T> for OptAtomic<false> {
    type Type = Nonatomic<T>;
}

// ---------------------------------------------------------------------------
// atomic_ref<T>
// ---------------------------------------------------------------------------

/// `ctp::ipc::atomic_ref<T>` — lock-free atomic ops on existing plain storage,
/// without changing the field's type.
///
/// The C++ exists to paper over `std::atomic_ref` availability (MSVC vs Apple
/// libc++); Rust needs no such dispatch — `Atomic*::from_ptr` is stable and
/// compiles to the same instructions everywhere.
///
/// [`AtomicRef::new`] is the safe constructor: an exclusive `&'a mut T` proves
/// nothing else touches the storage non-atomically for `'a`, while the
/// `AtomicRef` itself is `Send + Sync` and can be shared across (scoped)
/// threads. [`AtomicRef::from_ptr`] is the unchecked form for foreign storage
/// (e.g. a field inside a mapped shared-memory segment).
pub struct AtomicRef<'a, T: AtomicPrimitive> {
    inner: &'a T::Repr,
}

impl<'a, T: AtomicPrimitive> AtomicRef<'a, T> {
    /// `atomic_ref(T &value)`.
    ///
    /// # Panics
    /// If `value` is not aligned for `T`'s atomic counterpart. (Equal
    /// alignment holds on every target CTP supports; a few 32-bit targets give
    /// `AtomicU64` a stricter alignment than `u64`, and there the C++
    /// `std::atomic_ref` has an identical, merely unchecked, requirement.)
    #[inline]
    pub fn new(value: &'a mut T) -> Self {
        let ptr: *mut T = value;
        assert!(
            (ptr as usize).is_multiple_of(std::mem::align_of::<T::Repr>()),
            "atomic_ref: storage is under-aligned for its atomic counterpart"
        );
        // SAFETY: `ptr` comes from a live `&'a mut T`, so it is non-null,
        // dereferenceable and unaliased for `'a`; the exclusive borrow is moved
        // into this value, so for `'a` the storage is reachable only through the
        // returned atomic. `T::Repr` has `T`'s size and bit validity, and the
        // alignment requirement is asserted above.
        Self {
            inner: unsafe { T::atomic_from_ptr(ptr) },
        }
    }

    /// Construct over storage this reference does not own — the direct
    /// counterpart of the C++, which takes any `T&`.
    ///
    /// # Safety
    /// * `ptr` must be non-null, valid for reads and writes, and aligned for
    ///   `T`'s atomic counterpart, for all of `'a`.
    /// * For `'a`, the pointee must be accessed **only** through atomic
    ///   operations (this `AtomicRef`, another `AtomicRef`, or another
    ///   process's atomics on the same shared-memory bytes). Any concurrent
    ///   non-atomic access is a data race.
    #[inline]
    pub unsafe fn from_ptr(ptr: *mut T) -> Self {
        // SAFETY: forwarded to this function's own contract.
        Self {
            inner: unsafe { T::atomic_from_ptr(ptr) },
        }
    }

    /// `load(order)`.
    #[inline]
    pub fn load(&self, order: Ordering) -> T {
        T::atomic_load(self.inner, order)
    }

    /// `store(v, order)`.
    #[inline]
    pub fn store(&self, val: T, order: Ordering) {
        T::atomic_store(self.inner, val, order);
    }

    /// `fetch_add(v, order)`, returning the previous value.
    #[inline]
    pub fn fetch_add(&self, val: T, order: Ordering) -> T {
        T::atomic_fetch_add(self.inner, val, order)
    }

    /// `fetch_sub(v, order)`, returning the previous value.
    #[inline]
    pub fn fetch_sub(&self, val: T, order: Ordering) -> T {
        T::atomic_fetch_sub(self.inner, val, order)
    }

    /// `compare_exchange_weak(T &expected, desired, order)`; `*expected` is
    /// refreshed on failure, as in C++.
    #[inline]
    pub fn compare_exchange_weak(&self, expected: &mut T, desired: T, order: Ordering) -> bool {
        match T::atomic_compare_exchange_weak(
            self.inner,
            *expected,
            desired,
            order,
            failure_order(order),
        ) {
            Ok(_) => true,
            Err(actual) => {
                *expected = actual;
                false
            }
        }
    }

    /// `compare_exchange_strong(T &expected, desired, order)`.
    #[inline]
    pub fn compare_exchange_strong(&self, expected: &mut T, desired: T, order: Ordering) -> bool {
        match T::atomic_compare_exchange(
            self.inner,
            *expected,
            desired,
            order,
            failure_order(order),
        ) {
            Ok(_) => true,
            Err(actual) => {
                *expected = actual;
                false
            }
        }
    }
}

impl<T: AtomicPrimitive> fmt::Debug for AtomicRef<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("AtomicRef")
            .field(&self.load(Ordering::SeqCst))
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Fences and warp helpers (host branches only — divergence 1)
// ---------------------------------------------------------------------------

/// `ctp::ipc::threadfence()` — device-scope fence; host branch is a release
/// fence.
#[inline]
pub fn threadfence() {
    fence(Ordering::Release);
}

/// `ctp::ipc::threadfence_system()` — system-scope fence; host branch is a
/// seq_cst fence.
#[inline]
pub fn threadfence_system() {
    fence(Ordering::SeqCst);
}

/// `ctp::ipc::shfl_sync_u64()` — 64-bit warp broadcast. Host fallback: a warp
/// of one, so the value is returned unchanged (identical to the C++ host
/// branch, which also ignores `mask`/`src_lane`).
#[inline]
pub fn shfl_sync_u64(_mask: u32, val: u64, _src_lane: i32) -> u64 {
    val
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    // -- generic code compiles against BOTH storage strategies ---------------

    /// Stands in for a ported C++ template taking `opt_atomic<T, is_atomic>`.
    fn bump_twice<T: AtomicPrimitive, A: AtomicApi<T>>(a: &A) -> T {
        a.fetch_add(T::ONE, Ordering::Relaxed);
        a.fetch_add(T::ONE, Ordering::Relaxed)
    }

    fn cas_swap<T: AtomicPrimitive, A: AtomicApi<T>>(a: &A, from: T, to: T) -> bool {
        let mut expected = from;
        a.compare_exchange_strong(&mut expected, to, Ordering::SeqCst)
    }

    #[test]
    fn generic_over_both_storage_strategies() {
        let at = Atomic::<u64>::new(0);
        let non = Nonatomic::<u64>::new(0);
        assert_eq!(bump_twice(&at), 1);
        assert_eq!(bump_twice(&non), 1);
        assert_eq!(at.load(Ordering::SeqCst), 2);
        assert_eq!(non.load(Ordering::SeqCst), 2);
        assert!(cas_swap(&at, 2u64, 9));
        assert!(cas_swap(&non, 2u64, 9));
        assert!(at == 9u64);
        assert!(non == 9u64);
    }

    #[test]
    fn opt_atomic_selector_picks_the_right_type() {
        struct Ring<const IS_ATOMIC: bool>
        where
            OptAtomic<IS_ATOMIC>: OptAtomicSelect<u32>,
        {
            head: <OptAtomic<IS_ATOMIC> as OptAtomicSelect<u32>>::Type,
        }

        impl<const IS_ATOMIC: bool> Ring<IS_ATOMIC>
        where
            OptAtomic<IS_ATOMIC>: OptAtomicSelect<u32>,
        {
            fn new(v: u32) -> Self {
                Self { head: v.into() }
            }
            fn push(&self) -> u32 {
                self.head.fetch_add(1, Ordering::SeqCst)
            }
        }

        let atomic_ring = Ring::<true>::new(5);
        let plain_ring = Ring::<false>::new(5);
        assert_eq!(atomic_ring.push(), 5);
        assert_eq!(plain_ring.push(), 5);
        assert_eq!(atomic_ring.head.load(Ordering::SeqCst), 6);
        assert_eq!(plain_ring.head.load(Ordering::SeqCst), 6);

        // The selector really does resolve to distinct types.
        assert_eq!(
            std::any::type_name::<<OptAtomic<true> as OptAtomicSelect<u32>>::Type>(),
            std::any::type_name::<Atomic<u32>>()
        );
        assert_eq!(
            std::any::type_name::<<OptAtomic<false> as OptAtomicSelect<u32>>::Type>(),
            std::any::type_name::<Nonatomic<u32>>()
        );
    }

    // -- layout: shared-memory placement ------------------------------------

    #[test]
    fn layout_matches_the_underlying_primitive() {
        use std::mem::{align_of, size_of};
        assert_eq!(size_of::<Atomic<u64>>(), size_of::<u64>());
        assert_eq!(size_of::<Nonatomic<u64>>(), size_of::<u64>());
        assert_eq!(size_of::<Atomic<u8>>(), 1);
        assert_eq!(size_of::<Nonatomic<u8>>(), 1);
        assert_eq!(align_of::<Nonatomic<u32>>(), align_of::<u32>());
        // No Drop glue: required for shared-memory residency.
        assert!(!std::mem::needs_drop::<Atomic<u64>>());
        assert!(!std::mem::needs_drop::<Nonatomic<u64>>());
    }

    // -- nonatomic ----------------------------------------------------------

    #[test]
    fn nonatomic_basic_ops_ignore_ordering() {
        let n = Nonatomic::<u32>::new(10);
        assert_eq!(n.load(Ordering::Relaxed), 10);
        n.store(20, Ordering::Release);
        assert_eq!(n.load(Ordering::Acquire), 20);
        assert_eq!(n.fetch_add(5, Ordering::AcqRel), 20);
        assert_eq!(n.load(Ordering::SeqCst), 25);
        assert_eq!(n.fetch_sub(5, Ordering::Relaxed), 25);
        assert_eq!(n.load(Ordering::SeqCst), 20);
        assert_eq!(n.exchange(1, Ordering::SeqCst), 20);
        assert_eq!(n.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn nonatomic_bitwise_ops_return_previous() {
        let n = Nonatomic::<u32>::new(0b1100);
        assert_eq!(n.fetch_or(0b0011, Ordering::SeqCst), 0b1100);
        assert_eq!(n.load(Ordering::SeqCst), 0b1111);
        assert_eq!(n.fetch_and(0b0110, Ordering::SeqCst), 0b1111);
        assert_eq!(n.load(Ordering::SeqCst), 0b0110);
        assert_eq!(n.fetch_xor(0b0100, Ordering::SeqCst), 0b0110);
        assert_eq!(n.load(Ordering::SeqCst), 0b0010);
        assert_eq!(n.or_system(0b1000), 0b0010);
        assert_eq!(n.load(Ordering::SeqCst), 0b1010);
    }

    #[test]
    fn nonatomic_cas_refreshes_expected_on_failure() {
        let n = Nonatomic::<u32>::new(7);

        let mut expected = 7;
        assert!(n.compare_exchange_strong(&mut expected, 8, Ordering::SeqCst));
        assert_eq!(expected, 7); // untouched on success, as in C++
        assert_eq!(n.load(Ordering::SeqCst), 8);

        let mut expected = 7; // stale
        assert!(!n.compare_exchange_weak(&mut expected, 9, Ordering::SeqCst));
        assert_eq!(expected, 8); // refreshed with the actual value
        assert_eq!(n.load(Ordering::SeqCst), 8); // unchanged
    }

    #[test]
    fn nonatomic_scope_methods_are_plain() {
        let n = Nonatomic::<u64>::new(0);
        n.store_system(3);
        assert_eq!(n.load_system(), 3);
        assert_eq!(n.load_device(), 3);
        assert_eq!(n.fetch_add_system(2), 3);
        assert_eq!(n.fetch_sub_system(1), 5);
        assert_eq!(n.load(Ordering::SeqCst), 4);
        let mut expected = 4u64;
        assert!(n.compare_exchange_strong_system(&mut expected, 0, Ordering::SeqCst));
        assert_eq!(n.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn nonatomic_clone_eq_debug_default_from() {
        let a = Nonatomic::<i32>::new(-5);
        let b = a.clone();
        assert!(a == b);
        assert!(a == -5i32);
        assert!(a != Nonatomic::<i32>::new(-6));
        assert_eq!(Nonatomic::<i32>::default().load(Ordering::SeqCst), 0);
        assert_eq!(Nonatomic::from(42i32).load(Ordering::SeqCst), 42);
        assert_eq!(format!("{:?}", a), "Nonatomic(-5)");
        // Clone is a snapshot, not an alias.
        b.store(1, Ordering::SeqCst);
        assert_eq!(a.load(Ordering::SeqCst), -5);
    }

    #[test]
    fn nonatomic_get_mut_is_the_ref_accessor() {
        let mut n = Nonatomic::<u32>::new(1);
        *n.get_mut() += 41;
        assert_eq!(n.load(Ordering::SeqCst), 42);
    }

    // -- atomic -------------------------------------------------------------

    #[test]
    fn atomic_basic_ops() {
        let a = Atomic::<u32>::new(10);
        assert_eq!(a.load(Ordering::SeqCst), 10);
        a.store(20, Ordering::SeqCst);
        assert_eq!(a.fetch_add(5, Ordering::SeqCst), 20);
        assert_eq!(a.fetch_sub(5, Ordering::SeqCst), 25);
        assert_eq!(a.exchange(1, Ordering::SeqCst), 20);
        assert_eq!(a.load(Ordering::SeqCst), 1);
        assert_eq!(a.fetch_or(0b10, Ordering::SeqCst), 1);
        assert_eq!(a.fetch_and(0b10, Ordering::SeqCst), 0b11);
        assert_eq!(a.fetch_xor(0b10, Ordering::SeqCst), 0b10);
        assert_eq!(a.load(Ordering::SeqCst), 0);
        assert_eq!(a.inner().load(Ordering::SeqCst), 0);
    }

    #[test]
    fn atomic_cas_refreshes_expected_on_failure() {
        let a = Atomic::<u64>::new(7);

        let mut expected = 7;
        assert!(a.compare_exchange_strong(&mut expected, 8, Ordering::SeqCst));
        assert_eq!(expected, 7);

        let mut expected = 7; // stale
        assert!(!a.compare_exchange_strong(&mut expected, 9, Ordering::SeqCst));
        assert_eq!(expected, 8);
        assert_eq!(a.load(Ordering::SeqCst), 8);

        // Weak may fail spuriously; loop as real call sites do.
        let mut expected = 8;
        while !a.compare_exchange_weak(&mut expected, 10, Ordering::SeqCst) {
            assert_eq!(expected, 8); // no other writer: only spurious failures
        }
        assert_eq!(a.load(Ordering::SeqCst), 10);
    }

    /// The C++ single-order CAS overload derives the failure order; passing
    /// Release/AcqRel straight through to Rust's two-order API would panic.
    #[test]
    fn atomic_cas_accepts_release_orders_like_cpp() {
        assert_eq!(failure_order(Ordering::Release), Ordering::Relaxed);
        assert_eq!(failure_order(Ordering::AcqRel), Ordering::Acquire);
        assert_eq!(failure_order(Ordering::SeqCst), Ordering::SeqCst);
        assert_eq!(failure_order(Ordering::Acquire), Ordering::Acquire);
        assert_eq!(failure_order(Ordering::Relaxed), Ordering::Relaxed);

        for order in [Ordering::Release, Ordering::AcqRel] {
            let a = Atomic::<u32>::new(1);
            let mut expected = 1;
            assert!(a.compare_exchange_strong(&mut expected, 2, order));
            let mut stale = 1;
            assert!(!a.compare_exchange_strong(&mut stale, 3, order));
            assert_eq!(stale, 2);

            // ... and the same for the non-atomic twin and atomic_ref.
            let n = Nonatomic::<u32>::new(1);
            let mut expected = 1;
            assert!(n.compare_exchange_weak(&mut expected, 2, order));

            let mut storage = 1u32;
            let r = AtomicRef::new(&mut storage);
            let mut expected = 1;
            assert!(r.compare_exchange_strong(&mut expected, 2, order));
        }
    }

    #[test]
    fn atomic_scope_methods_match_host_cpp() {
        let a = Atomic::<u64>::new(0);
        a.store_system(3);
        assert_eq!(a.load_system(), 3);
        assert_eq!(a.load_device(), 3);
        assert_eq!(a.fetch_add_system(2), 3);
        assert_eq!(a.fetch_sub_system(1), 5);
        assert_eq!(a.or_system(0b1000), 4);
        assert_eq!(a.load(Ordering::SeqCst), 0b1100);
        let mut expected = 0b1100u64;
        assert!(a.compare_exchange_strong_system(&mut expected, 0, Ordering::SeqCst));
        assert_eq!(a.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn atomic_clone_eq_debug_default_from() {
        let a = Atomic::<i64>::new(-5);
        let b = a.clone();
        assert!(a == b);
        assert!(a == -5i64);
        assert!(a != Atomic::<i64>::new(-6));
        assert_eq!(Atomic::<i64>::default().load(Ordering::SeqCst), 0);
        assert_eq!(Atomic::from(42i64).load(Ordering::SeqCst), 42);
        assert_eq!(format!("{:?}", a), "Atomic(-5)");
        b.store(1, Ordering::SeqCst); // clone is a snapshot, not an alias
        assert_eq!(a.load(Ordering::SeqCst), -5);
    }

    #[test]
    fn atomic_get_mut() {
        let mut a = Atomic::<u32>::new(1);
        *a.get_mut() += 41;
        assert_eq!(a.load(Ordering::SeqCst), 42);
    }

    // -- increment / decrement (the C++ host post-inc bug is NOT ported) -----

    #[test]
    fn inc_dec_return_new_and_old_values() {
        for_each_variant(|a: &dyn AtomicApiDyn| {
            assert_eq!(a.pre_inc_(), 1); // ++x -> new
            assert_eq!(a.post_inc_(), 1); // x++ -> old
            assert_eq!(a.load_(), 2); // ... and it really incremented
            assert_eq!(a.pre_dec_(), 1); // --x -> new
            assert_eq!(a.post_dec_(), 1); // x-- -> old
            assert_eq!(a.load_(), 0);
        });
    }

    /// Tiny object-safe shim so the inc/dec table runs over both types.
    trait AtomicApiDyn {
        fn pre_inc_(&self) -> u32;
        fn post_inc_(&self) -> u32;
        fn pre_dec_(&self) -> u32;
        fn post_dec_(&self) -> u32;
        fn load_(&self) -> u32;
    }

    impl<A: AtomicApi<u32>> AtomicApiDyn for A {
        fn pre_inc_(&self) -> u32 {
            self.pre_inc()
        }
        fn post_inc_(&self) -> u32 {
            self.post_inc()
        }
        fn pre_dec_(&self) -> u32 {
            self.pre_dec()
        }
        fn post_dec_(&self) -> u32 {
            self.post_dec()
        }
        fn load_(&self) -> u32 {
            self.load(Ordering::SeqCst)
        }
    }

    fn for_each_variant(f: impl Fn(&dyn AtomicApiDyn)) {
        f(&Atomic::<u32>::new(0));
        f(&Nonatomic::<u32>::new(0));
    }

    // -- edge cases: zero, boundaries, wraparound ---------------------------

    #[test]
    fn zero_deltas_are_no_ops() {
        let a = Atomic::<u64>::new(5);
        let n = Nonatomic::<u64>::new(5);
        assert_eq!(a.fetch_add(0, Ordering::SeqCst), 5);
        assert_eq!(n.fetch_add(0, Ordering::SeqCst), 5);
        assert_eq!(a.fetch_sub(0, Ordering::SeqCst), 5);
        assert_eq!(n.fetch_sub(0, Ordering::SeqCst), 5);
        assert_eq!(a.load(Ordering::SeqCst), 5);
        assert_eq!(n.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn unsigned_overflow_and_underflow_wrap() {
        let a = Atomic::<u8>::new(u8::MAX);
        let n = Nonatomic::<u8>::new(u8::MAX);
        assert_eq!(a.fetch_add(1, Ordering::SeqCst), u8::MAX);
        assert_eq!(n.fetch_add(1, Ordering::SeqCst), u8::MAX);
        assert_eq!(a.load(Ordering::SeqCst), 0);
        assert_eq!(n.load(Ordering::SeqCst), 0);

        // Underflow: the ring-buffer "empty minus one" case.
        assert_eq!(a.fetch_sub(1, Ordering::SeqCst), 0);
        assert_eq!(n.fetch_sub(1, Ordering::SeqCst), 0);
        assert_eq!(a.load(Ordering::SeqCst), u8::MAX);
        assert_eq!(n.load(Ordering::SeqCst), u8::MAX);

        let big = Atomic::<u64>::new(u64::MAX);
        let big_n = Nonatomic::<u64>::new(u64::MAX);
        assert_eq!(big.post_inc(), u64::MAX);
        assert_eq!(big_n.post_inc(), u64::MAX);
        assert_eq!(big.load(Ordering::SeqCst), 0);
        assert_eq!(big_n.load(Ordering::SeqCst), 0);
    }

    /// C++ `nonatomic` would be UB here (`x += count` on a signed overflow);
    /// both Rust variants wrap, and neither panics in debug (divergence 6).
    #[test]
    fn signed_overflow_wraps_in_both_variants() {
        let a = Atomic::<i8>::new(i8::MAX);
        let n = Nonatomic::<i8>::new(i8::MAX);
        assert_eq!(a.fetch_add(1, Ordering::SeqCst), i8::MAX);
        assert_eq!(n.fetch_add(1, Ordering::SeqCst), i8::MAX);
        assert_eq!(a.load(Ordering::SeqCst), i8::MIN);
        assert_eq!(n.load(Ordering::SeqCst), i8::MIN);
        assert_eq!(a.pre_dec(), i8::MAX);
        assert_eq!(n.pre_dec(), i8::MAX);
    }

    #[test]
    fn every_primitive_width_round_trips() {
        macro_rules! check {
            ($t:ty) => {{
                let a = Atomic::<$t>::new(<$t>::MAX);
                assert_eq!(a.load(Ordering::SeqCst), <$t>::MAX);
                assert_eq!(a.exchange(<$t>::MIN, Ordering::SeqCst), <$t>::MAX);
                assert_eq!(a.load(Ordering::SeqCst), <$t>::MIN);
                let n = Nonatomic::<$t>::new(<$t>::MAX);
                assert_eq!(n.exchange(<$t>::MIN, Ordering::SeqCst), <$t>::MAX);
                assert_eq!(n.load(Ordering::SeqCst), <$t>::MIN);
            }};
        }
        check!(u8);
        check!(u16);
        check!(u32);
        check!(u64);
        check!(usize);
        check!(i8);
        check!(i16);
        check!(i32);
        check!(i64);
        check!(isize);
    }

    // -- concurrency --------------------------------------------------------

    #[test]
    fn atomic_fetch_add_is_race_free() {
        const THREADS: u64 = 8;
        const ITERS: u64 = 10_000;
        let counter = Arc::new(Atomic::<u64>::new(0));
        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let counter = Arc::clone(&counter);
                thread::spawn(move || {
                    for _ in 0..ITERS {
                        counter.fetch_add(1, Ordering::Relaxed);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(counter.load(Ordering::SeqCst), THREADS * ITERS);
    }

    #[test]
    fn atomic_cas_loop_elects_exactly_one_winner() {
        const THREADS: u32 = 8;
        let state = Arc::new(Atomic::<u32>::new(0));
        let wins = Arc::new(Atomic::<u32>::new(0));
        let handles: Vec<_> = (0..THREADS)
            .map(|i| {
                let state = Arc::clone(&state);
                let wins = Arc::clone(&wins);
                thread::spawn(move || {
                    let mut expected = 0;
                    if state.compare_exchange_strong(&mut expected, i + 1, Ordering::AcqRel) {
                        wins.fetch_add(1, Ordering::SeqCst);
                    } else {
                        // Loser sees the winner's value, never the stale 0.
                        assert_ne!(expected, 0);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(wins.load(Ordering::SeqCst), 1);
        assert_ne!(state.load(Ordering::SeqCst), 0);
    }

    /// Release/Acquire publication: the C++ relies on this for queue handoff.
    #[test]
    fn release_acquire_publishes_prior_writes() {
        for _ in 0..64 {
            let data = Arc::new(Atomic::<u64>::new(0));
            let flag = Arc::new(Atomic::<u32>::new(0));
            let (d, f) = (Arc::clone(&data), Arc::clone(&flag));
            let writer = thread::spawn(move || {
                d.store(0xDEAD_BEEF, Ordering::Relaxed);
                f.store(1, Ordering::Release);
            });
            let (d, f) = (Arc::clone(&data), Arc::clone(&flag));
            let reader = thread::spawn(move || {
                while f.load(Ordering::Acquire) == 0 {
                    std::hint::spin_loop();
                }
                assert_eq!(d.load(Ordering::Relaxed), 0xDEAD_BEEF);
            });
            writer.join().unwrap();
            reader.join().unwrap();
        }
    }

    // -- atomic_ref ---------------------------------------------------------

    #[test]
    fn atomic_ref_operates_on_plain_storage() {
        let mut storage: u64 = 5;
        {
            let r = AtomicRef::new(&mut storage);
            assert_eq!(r.load(Ordering::SeqCst), 5);
            r.store(7, Ordering::SeqCst);
            assert_eq!(r.fetch_add(3, Ordering::SeqCst), 7);
            assert_eq!(r.fetch_sub(2, Ordering::SeqCst), 10);
            let mut expected = 8;
            assert!(r.compare_exchange_strong(&mut expected, 100, Ordering::SeqCst));
            let mut stale = 8;
            assert!(!r.compare_exchange_weak(&mut stale, 1, Ordering::SeqCst));
            assert_eq!(stale, 100);
            assert_eq!(format!("{:?}", r), "AtomicRef(100)");
        }
        // The plain variable itself was updated.
        assert_eq!(storage, 100);
    }

    #[test]
    fn atomic_ref_is_shareable_across_threads() {
        const THREADS: u64 = 8;
        const ITERS: u64 = 5_000;
        let mut storage: u64 = 0;
        {
            let r = AtomicRef::new(&mut storage);
            thread::scope(|s| {
                for _ in 0..THREADS {
                    let r = &r;
                    s.spawn(move || {
                        for _ in 0..ITERS {
                            r.fetch_add(1, Ordering::Relaxed);
                        }
                    });
                }
            });
        }
        assert_eq!(storage, THREADS * ITERS);
    }

    #[test]
    fn atomic_ref_from_ptr_matches_the_safe_ctor() {
        let mut storage: u32 = 1;
        let ptr: *mut u32 = &mut storage;
        {
            // SAFETY: `storage` outlives `r`, is aligned (it is a `u32` and
            // AtomicU32 has u32's alignment on all supported targets), and is
            // only accessed through `r` for `r`'s lifetime.
            let r = unsafe { AtomicRef::from_ptr(ptr) };
            assert_eq!(r.fetch_add(41, Ordering::SeqCst), 1);
        }
        assert_eq!(storage, 42);
    }

    // -- fences / warp helpers ---------------------------------------------

    #[test]
    fn fences_and_shfl_host_fallbacks() {
        threadfence();
        threadfence_system();
        assert_eq!(shfl_sync_u64(0xFFFF_FFFF, 0xDEAD_BEEF_CAFE_F00D, 3), 0xDEAD_BEEF_CAFE_F00D);
        assert_eq!(shfl_sync_u64(0, 0, 0), 0);
    }
}
