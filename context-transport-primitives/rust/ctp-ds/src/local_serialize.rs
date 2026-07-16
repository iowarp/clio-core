// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
//! `ctp::ipc::{LocalSerialize, LocalDeserialize, CalculateSizeArchive}` →
//! [`LocalSaveArchive`], [`LocalLoadArchive`], [`CalculateSizeArchive`]: the
//! **local** (same-process) byte-buffer archives underneath
//! `clio::run::Default{Save,Load}Archive` / `LocalSaveTask` / `LocalLoadTask`.
//!
//! These archives are deliberately *not* a wire format. They exist to move a
//! task's fields between two threads of one process (or one host), so they
//! `memcpy` machine representations: **native endianness, native `size_t`
//! width, no versioning, no tags, no self-description**. Both C++ and Rust
//! describe the same bytes only because both run on the same machine — this is
//! the same contract the C++ side already offers, not a new limitation. No
//! `serde`: the trait below is hand-rolled, mirroring the C++ free-function
//! `save`/`load` customization points.
//!
//! # Byte layout (matches `local_serialize.h` / `serialize_common.h`)
//!
//! ```text
//! arithmetic T   -> sizeof(T) native-endian bytes             (memcpy of &obj)
//! bool           -> 1 byte (0 | 1)
//! string         -> [len: usize][len bytes]                   (save_string_fused)
//! vector<T>      -> [len: usize][elem_0][elem_1]...[elem_n-1]
//! list<T>        -> [len: usize][elem_0]...[elem_n-1]
//! unordered_map  -> [len: usize][k_0][v_0][k_1][v_1]...        (iteration order
//!                                                               unspecified —
//!                                                               as in C++)
//! ```
//!
//! `len` is a `usize` == C++ `size_t` (8 bytes on 64-bit, 4 on 32-bit): the C++
//! writes `sizeof(size_t)` bytes, so `usize::to_ne_bytes` is byte-exact on any
//! target where the two agree.
//!
//! The C++ `save_vec`/`load_vec` bulk-`memcpy` arithmetic element types instead
//! of looping. That is a *speed* specialization only: for arithmetic `T` the
//! bytes are identical either way, so the element-wise Rust port stays
//! wire-compatible with both C++ paths (see divergence 9).
//!
//! # Buffer growth (byte-exact port of `write_binary`)
//!
//! `LocalSaveArchive` grows the *length* (not just capacity) of the buffer by
//! doubling from a 64-byte floor, and tracks the true write head in `cur_off`.
//! So mid-serialization `buf.len()` is the grown capacity, **not** the number of
//! bytes written — [`finalize`](LocalSaveArchive::finalize) (C++ `Finalize()`)
//! truncates it back to `cur_off`. Read
//! [`serialized_size`](LocalSaveArchive::serialized_size) for the live count.
//!
//! # C++ → Rust name mapping
//!
//! | C++ (`ctp::ipc`)                             | Rust                                             |
//! |----------------------------------------------|--------------------------------------------------|
//! | `LocalSerialize<DataT>` (the *archive* class) | [`LocalSaveArchive<'a>`]                         |
//! | `LocalDeserialize<DataT>`                    | [`LocalLoadArchive<'a>`]                         |
//! | `CalculateSizeArchive`                       | [`CalculateSizeArchive`]                         |
//! | `is_serializeable_v<Ar, T>` (SFINAE concept) | [`LocalSerialize`] (the *trait*)                 |
//! | `save(Ar&, const T&)` free fn                | [`LocalSerialize::save`]                         |
//! | `load(Ar&, T&)` free fn                      | [`LocalSerialize::load`]                         |
//! | save-side archive API (duck-typed `Ar`)      | [`SaveArchive`] trait                            |
//! | load-side archive API (duck-typed `Ar`)      | [`LoadArchive`] trait                            |
//! | `LocalSerialize(DataT&)`                     | [`LocalSaveArchive::new`]                        |
//! | `LocalSerialize(DataT&, bool)` (append ctor) | [`LocalSaveArchive::new_append`]                 |
//! | `Finalize()`                                 | [`LocalSaveArchive::finalize`]                   |
//! | `cur_off_` (public field)                    | [`serialized_size`](LocalSaveArchive::serialized_size) / [`reset`](LocalSaveArchive::reset) |
//! | `operator<<` / `operator&` (saving)          | [`LocalSaveArchive::save_value`]                 |
//! | `operator>>` / `operator&` (loading)         | [`LocalLoadArchive::load_value`]                 |
//! | `operator()(Args&&...)` (argpack fold)       | call `save_value`/`load_value` per field         |
//! | `write_binary(const char*, size_t)`          | [`SaveArchive::write_binary`]                    |
//! | `read_binary(char*, size_t)`                 | [`LoadArchive::read_binary`] (returns `bool`)    |
//! | `save_string_fused(const char*, size_t)`     | [`SaveArchive::save_string_fused`]               |
//! | `save_string` / `load_string`                | `impl LocalSerialize for String`                 |
//! | `save_vec` / `load_vec`                      | `impl LocalSerialize for Vec<T>`                 |
//! | `save_list` / `load_list`                    | `impl LocalSerialize for LinkedList<T>`          |
//! | `save_map` / `load_map`                      | `impl LocalSerialize for HashMap<K, V>`          |
//! | `size()` (CalculateSizeArchive)              | [`CalculateSizeArchive::size`]                   |
//! | `resize_for_overwrite`                       | (internal; no public analogue)                   |
//! | `has_{serialize,save,load}_{fun,cls}_v`      | (unneeded — trait resolution replaces SFINAE)    |
//! | `has_range_ops_v` / `supports_range_ops`     | (not ported — see divergence 4)                  |
//! | `range` / `write_range` / `read_range`       | (not ported — see divergence 4)                  |
//! | `warp_converged_` / `SetWarpConverged`       | (not ported — see divergence 5)                  |
//!
//! # Semantic divergences from the C++
//!
//! 1. **Name collision, resolved.** C++ `ctp::ipc::LocalSerialize` is the
//!    *save archive*; the Rust trait named [`LocalSerialize`] is the
//!    *serializable-type* trait (C++'s `is_serializeable_v` concept). The C++
//!    class is [`LocalSaveArchive`] here. This is the one rename that could
//!    confuse a reader porting call sites, hence the top row of the table.
//! 2. **Read overrun is a flag, not a log.** C++ `read_binary` logs
//!    `HLOG(kError, ...)` and returns, leaving the destination *and* `cur_off_`
//!    untouched. Rust has no `HLOG` here: [`LoadArchive::read_binary`] returns
//!    `false` and latches [`LocalLoadArchive::has_error`]. The
//!    destination/offset semantics are preserved exactly — a failed load leaves
//!    the target value unmodified (so `size_t size = 0; ar >> size;` still
//!    yields `0` on a truncated buffer, as in C++).
//! 3. **Errors latch; they never unwind.** No panics, no `Result` in the hot
//!    path — matching C++, which cannot throw from these archives. Check
//!    `has_error()` after a load sequence; it stays set until [`LocalLoadArchive::reset`].
//! 4. **`range()`/`write_range()`/`read_range()` are not ported.** They
//!    `reinterpret_cast` a *span of struct fields* (`&first_field` ..
//!    `&last_field + sizeof(last)`) into one `memcpy`. That is inexpressible in
//!    safe Rust (no generic address-of-field), and it is **not** a wire-format
//!    equivalent of per-field saving: the span *includes inter-field padding*,
//!    so it produces different bytes than the field-by-field path whenever the
//!    C++ struct has padding. Rust callers save field-by-field, which matches
//!    the C++ **non**-`is_pod_` path (`PushPod(false)`). A caller holding a byte
//!    view of a POD may still call [`SaveArchive::write_binary`] directly.
//!    Consequently `supports_range_ops`/`has_range_ops_v` have no analogue.
//! 5. **GPU warp paths dropped.** `warp_converged_`, `threadIdx`-lane-strided
//!    `memcpy` and `__syncwarp()` are CUDA device code (`#if CTP_IS_GPU`). This
//!    is a host-only port; the sequential branch — the one the `#else` compiles
//!    on host — is what is reproduced. Byte output is identical (the warp path
//!    is a parallel `memcpy`, not a different format).
//! 6. **`bool` is normalized.** C++ `memcpy`s a raw byte into a `bool`; a byte
//!    of `2` yields an invalid `bool` (UB on read). Rust loads `byte != 0`, so
//!    every byte pattern maps to a valid `bool`. Saving is identical (`0`/`1`).
//! 7. **`String` is UTF-8 validated.** C++ `std::string` is a byte bag. On
//!    invalid UTF-8 the Rust load latches the error flag and stores the
//!    `from_utf8_lossy` replacement (U+FFFD). Use `Vec<u8>` for byte strings
//!    that are not text — it is byte-exact with the C++ `std::string` format.
//! 8. **Corrupt length prefixes are bounded, not trusted.** C++ `load_string`
//!    does `resize(size)` *before* the read, so a corrupt `size` either throws
//!    `bad_alloc` or leaves a zero-filled string of that length. Rust checks
//!    `len > remaining()` first: it latches the error and clears the string
//!    rather than attempt an unbounded allocation. Container loads likewise cap
//!    their pre-reservation at `remaining()` (a hint only — element count still
//!    drives the loop, so an absurd length over a *long* buffer still allocates,
//!    aborting on OOM where C++ would throw the catchable `bad_alloc`).
//! 9. **No arithmetic bulk-`memcpy` specialization.** `Vec<T>` always loops
//!    per element (Rust has no stable specialization). Bytes are identical
//!    (divergence-free *format*); only throughput differs for large
//!    `Vec<u8>`/`Vec<f64>`. Callers needing the bulk path can
//!    `write_binary`/`read_binary` a byte view directly.
//! 10. **Container loads follow C++ container-by-container.** `Vec` load
//!     replaces (C++ `resize` + overwrite every slot), while `LinkedList`
//!     **appends** (C++ `emplace_back` in a loop — it does *not* clear) and
//!     `HashMap` **merges/overwrites keys** (C++ `obj[key] = val` — also no
//!     clear). These asymmetries are faithful, not oversights: loading twice
//!     into one list doubles it in both languages. Sub-case: C++ `load_vec`
//!     reuses existing elements as load targets, so loading a
//!     `vector<unordered_map<..>>` into a *non-empty* vector merges into the
//!     surviving maps; Rust clears the vector first and loads into fresh
//!     `Default` elements. Reachable only via that reuse-into-dirty-vector path.
//! 11. **Overflow saturates instead of wrapping.** `CalculateSizeArchive`
//!     accumulates `cur_off` with `saturating_add` (C++ `+=` wraps silently on
//!     `size_t` overflow); the growth loop uses `saturating_mul`, which also
//!     rules out C++'s potential infinite `new_cap *= 2` at `SIZE_MAX`.
//!     `LocalSaveArchive::cur_off` cannot overflow: it is bounded by an actual
//!     allocation (`<= isize::MAX`).
//! 12. **Enums.** C++ auto-serializes any `enum` as its underlying type via
//!     `if constexpr (std::is_enum_v<T>)`. Rust has no such reflection: `impl
//!     LocalSerialize` by hand and save the discriminant (e.g. `(*self as
//!     u8).save(ar)`), which is byte-compatible with a `enum class : uint8_t`.
//!     See `tests::msg_type_enum_roundtrips_as_u8` for the `LocalMsgType` shape.
//! 13. **Rust-only conveniences** (no C++ counterpart): [`to_bytes`],
//!     [`from_bytes`], [`calculate_size`], [`LoadArchive::remaining`].
//!
//! # Not in this module (the task-archive layer)
//!
//! `clio::run::Default{Save,Load}Archive` = `Local{Save,Load}TaskArchive<
//! priv::vector<char>>`, which *wraps* these archives and adds `Task` dispatch
//! (`SerializeIn`/`SerializeOut`), `LocalTaskInfo` recording, `LbmMeta`
//! send/recv bulk lists, and `bulk()` `ShmPtr`/`FullPtr` modes 0–3. Those live
//! in `context-runtime` and depend on `Task`, the allocator and `ShmPtr`; they
//! belong to a `ctp-runtime`-side port, not to `ctp-ds`. This module is the
//! complete `ctp::ipc` substrate they are built from.

use std::collections::{HashMap, LinkedList};
use std::hash::Hash;

/// Width of the length prefix — C++ writes `sizeof(size_t)` bytes.
const LEN_BYTES: usize = core::mem::size_of::<usize>();

/// Initial buffer length used by the C++ growth loop (`if (new_cap == 0) new_cap = 64;`).
const MIN_CAP: usize = 64;

// ---------------------------------------------------------------------------
// Archive traits (the duck-typed `Ar` template parameter of the C++)
// ---------------------------------------------------------------------------

/// Save-side archive API — the C++ `Ar` requirements under `is_saving`.
///
/// Implemented by [`LocalSaveArchive`] (writes bytes) and
/// [`CalculateSizeArchive`] (counts them). Any `impl` must keep the two in
/// agreement: a dry run's [`CalculateSizeArchive::size`] must equal the real
/// archive's [`LocalSaveArchive::serialized_size`] for the same value, which is
/// exactly what the C++ pre-sizing path in `LocalSaveTaskArchive` relies on.
pub trait SaveArchive {
    /// C++ `write_binary(const char *data, size_t size)`.
    fn write_binary(&mut self, data: &[u8]);

    /// C++ `save_string_fused` — length prefix + payload under one capacity
    /// check. The default impl emits identical bytes to the fused overrides.
    fn save_string_fused(&mut self, str_data: &[u8]) {
        self.write_binary(&str_data.len().to_ne_bytes());
        if !str_data.is_empty() {
            self.write_binary(str_data);
        }
    }

    /// Emit a `size_t` length prefix (C++ `ar << obj.size()`).
    fn save_len(&mut self, len: usize) {
        self.write_binary(&len.to_ne_bytes());
    }
}

/// Load-side archive API — the C++ `Ar` requirements under `is_loading`.
pub trait LoadArchive {
    /// C++ `read_binary(char *data, size_t size)`.
    ///
    /// Returns `false` and latches the error flag if fewer than `out.len()`
    /// bytes remain; on failure `out` and the read offset are **both** left
    /// untouched, matching the C++ early-return.
    fn read_binary(&mut self, out: &mut [u8]) -> bool;

    /// Bytes left to read. (Rust-only; used to bound allocations.)
    fn remaining(&self) -> usize;

    /// Whether any read has failed (or invalid UTF-8 was seen) since `reset`.
    fn has_error(&self) -> bool;

    /// Latch the error flag.
    fn set_error(&mut self);

    /// Read a `size_t` length prefix (C++ `size_t size = 0; ar >> size;`).
    ///
    /// Yields `0` on a truncated buffer — the C++ leaves its zero-initialized
    /// local untouched, so an empty container is the shared outcome.
    fn load_len(&mut self) -> usize {
        let mut buf = [0u8; LEN_BYTES];
        self.read_binary(&mut buf);
        usize::from_ne_bytes(buf)
    }
}

/// A type that can be saved to / loaded from a local archive.
///
/// Replaces the C++ `is_serializeable_v<Ar, T>` SFINAE cluster
/// (`has_serialize_fun_v`, `has_load_save_fun_v`, `has_serialize_cls_v`,
/// `has_load_save_cls_v`): where C++ probes for a `save`/`load`/`serialize`
/// free function or member, Rust resolves one trait impl.
///
/// `load` takes `&mut self` (not `-> Self`) to mirror the C++ `load(Ar&, T&)`
/// out-parameter, which lets container loads reuse storage — and preserves the
/// append/merge semantics of divergence 10.
///
/// # Example
///
/// ```
/// use ctp_ds::local_serialize::{LoadArchive, LocalSerialize, SaveArchive, to_bytes, from_bytes};
///
/// #[derive(Default, PartialEq, Debug)]
/// struct Blob { id: u64, name: String, tags: Vec<u32> }
///
/// impl LocalSerialize for Blob {
///     fn save<A: SaveArchive + ?Sized>(&self, ar: &mut A) {
///         self.id.save(ar);
///         self.name.save(ar);
///         self.tags.save(ar);
///     }
///     fn load<A: LoadArchive + ?Sized>(&mut self, ar: &mut A) {
///         self.id.load(ar);
///         self.name.load(ar);
///         self.tags.load(ar);
///     }
/// }
///
/// let blob = Blob { id: 7, name: "hi".into(), tags: vec![1, 2, 3] };
/// let bytes = to_bytes(&blob);
/// assert_eq!(from_bytes::<Blob>(&bytes).unwrap(), blob);
/// ```
pub trait LocalSerialize {
    /// C++ `save(Ar &ar, const T &obj)`.
    fn save<A: SaveArchive + ?Sized>(&self, ar: &mut A);

    /// C++ `load(Ar &ar, T &obj)`.
    fn load<A: LoadArchive + ?Sized>(&mut self, ar: &mut A);
}

// ---------------------------------------------------------------------------
// LocalSaveArchive  (C++ ctp::ipc::LocalSerialize<DataT>)
// ---------------------------------------------------------------------------

/// Serializes values into a caller-owned `Vec<u8>`.
///
/// Port of `ctp::ipc::LocalSerialize<DataT>`. Like the C++ (`DataT &data_`) it
/// **borrows** the buffer rather than owning it — the archive is a cursor.
///
/// See the module docs on buffer growth: `buf.len()` over-reports until
/// [`finalize`](Self::finalize).
#[derive(Debug)]
pub struct LocalSaveArchive<'a> {
    data: &'a mut Vec<u8>,
    cur_off: usize,
}

impl<'a> LocalSaveArchive<'a> {
    /// C++ `LocalSerialize(DataT &data)` — writes from offset 0, overwriting.
    pub fn new(data: &'a mut Vec<u8>) -> Self {
        Self { data, cur_off: 0 }
    }

    /// C++ `LocalSerialize(DataT &data, bool)` — the append ctor
    /// (`cur_off_(data.size())`). The tag argument exists only to disambiguate
    /// the C++ overload; the Rust name carries that meaning instead.
    pub fn new_append(data: &'a mut Vec<u8>) -> Self {
        let cur_off = data.len();
        Self { data, cur_off }
    }

    /// C++ `Finalize()` — commit `cur_off_` to the buffer's length.
    ///
    /// Must be called when serialization is complete so `len()` reflects what
    /// was written; capacity is retained (C++ `resize` never shrinks capacity).
    pub fn finalize(&mut self) {
        // cur_off <= data.len() always holds (write_binary grows first), so the
        // C++ `resize(cur_off_)` can only ever shrink here.
        self.data.truncate(self.cur_off);
    }

    /// C++ `GetSerializedSize()` — bytes actually written, not buffer capacity.
    pub fn serialized_size(&self) -> usize {
        self.cur_off
    }

    /// C++ `Reset()`'s `serializer_.cur_off_ = 0` — rewind without freeing.
    pub fn reset(&mut self) {
        self.cur_off = 0;
    }

    /// C++ `operator<<` / `operator&`. Chainable, mirroring `ar << a << b`.
    pub fn save_value<T: LocalSerialize + ?Sized>(&mut self, value: &T) -> &mut Self {
        value.save(self);
        self
    }

    /// Grow `data` so that `new_off` bytes are addressable, byte-exactly
    /// reproducing the C++ doubling loop (64-byte floor, resize-not-reserve).
    fn grow_for(&mut self, new_off: usize) {
        if new_off > self.data.len() {
            let mut new_cap = self.data.len();
            if new_cap == 0 {
                new_cap = MIN_CAP;
            }
            while new_cap < new_off {
                // C++ `new_cap *= 2` would wrap (and spin forever) at SIZE_MAX.
                new_cap = new_cap.saturating_mul(2);
            }
            self.data.resize(new_cap, 0);
        }
    }
}

impl SaveArchive for LocalSaveArchive<'_> {
    fn write_binary(&mut self, data: &[u8]) {
        // cur_off <= data.len() <= isize::MAX, so this cannot overflow.
        let new_off = self.cur_off + data.len();
        self.grow_for(new_off);
        if !data.is_empty() {
            self.data[self.cur_off..new_off].copy_from_slice(data);
        }
        self.cur_off = new_off;
    }

    fn save_string_fused(&mut self, str_data: &[u8]) {
        let len = str_data.len();
        let new_off = self.cur_off + LEN_BYTES + len;
        self.grow_for(new_off);
        let body = self.cur_off + LEN_BYTES;
        self.data[self.cur_off..body].copy_from_slice(&len.to_ne_bytes());
        if len > 0 {
            self.data[body..new_off].copy_from_slice(str_data);
        }
        self.cur_off = new_off;
    }
}

// ---------------------------------------------------------------------------
// LocalLoadArchive  (C++ ctp::ipc::LocalDeserialize<DataT>)
// ---------------------------------------------------------------------------

/// Deserializes values from a caller-owned byte buffer.
///
/// Port of `ctp::ipc::LocalDeserialize<DataT>` (`const DataT &data_`).
#[derive(Debug)]
pub struct LocalLoadArchive<'a> {
    data: &'a [u8],
    cur_off: usize,
    error: bool,
}

impl<'a> LocalLoadArchive<'a> {
    /// C++ `LocalDeserialize(const DataT &data)`.
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            cur_off: 0,
            error: false,
        }
    }

    /// Bytes consumed so far (C++ `cur_off_`).
    pub fn cur_off(&self) -> usize {
        self.cur_off
    }

    /// C++ `Reset()` — placement-new of a fresh deserializer over the same
    /// buffer. Also clears the (Rust-only) error latch.
    pub fn reset(&mut self) {
        self.cur_off = 0;
        self.error = false;
    }

    /// C++ `operator>>` / `operator&`. Chainable.
    pub fn load_value<T: LocalSerialize + ?Sized>(&mut self, value: &mut T) -> &mut Self {
        value.load(self);
        self
    }
}

impl LoadArchive for LocalLoadArchive<'_> {
    fn read_binary(&mut self, out: &mut [u8]) -> bool {
        match self.cur_off.checked_add(out.len()) {
            Some(end) if end <= self.data.len() => {
                if !out.is_empty() {
                    out.copy_from_slice(&self.data[self.cur_off..end]);
                }
                self.cur_off = end;
                true
            }
            // C++: HLOG(kError, "...read beyond end of data"); return *this;
            // — destination and cur_off_ both left untouched.
            _ => {
                self.error = true;
                false
            }
        }
    }

    fn remaining(&self) -> usize {
        self.data.len() - self.cur_off
    }

    fn has_error(&self) -> bool {
        self.error
    }

    fn set_error(&mut self) {
        self.error = true;
    }
}

// ---------------------------------------------------------------------------
// CalculateSizeArchive  (C++ ctp::ipc::CalculateSizeArchive)
// ---------------------------------------------------------------------------

/// Dry-run archive: computes the serialized size without copying data.
///
/// Port of `ctp::ipc::CalculateSizeArchive`. Implements the same save-side API
/// as [`LocalSaveArchive`] so one `save` impl serves both, which is how the C++
/// pre-sizes a task buffer before serializing into it.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CalculateSizeArchive {
    cur_off: usize,
}

impl CalculateSizeArchive {
    /// C++ `CalculateSizeArchive() = default`.
    pub fn new() -> Self {
        Self { cur_off: 0 }
    }

    /// C++ `size()` — the total computed size.
    pub fn size(&self) -> usize {
        self.cur_off
    }

    /// Rewind the counter for reuse.
    pub fn reset(&mut self) {
        self.cur_off = 0;
    }

    /// C++ `operator<<` / `operator&`. Chainable.
    pub fn save_value<T: LocalSerialize + ?Sized>(&mut self, value: &T) -> &mut Self {
        value.save(self);
        self
    }
}

impl SaveArchive for CalculateSizeArchive {
    fn write_binary(&mut self, data: &[u8]) {
        // C++ `cur_off_ += size` wraps on overflow; saturate instead.
        self.cur_off = self.cur_off.saturating_add(data.len());
    }

    fn save_string_fused(&mut self, str_data: &[u8]) {
        self.cur_off = self
            .cur_off
            .saturating_add(LEN_BYTES)
            .saturating_add(str_data.len());
    }
}

// ---------------------------------------------------------------------------
// Primitive impls  (C++ `if constexpr (std::is_arithmetic<T>::value)`)
// ---------------------------------------------------------------------------

macro_rules! impl_local_serialize_arithmetic {
    ($($t:ty),* $(,)?) => {$(
        impl LocalSerialize for $t {
            fn save<A: SaveArchive + ?Sized>(&self, ar: &mut A) {
                ar.write_binary(&self.to_ne_bytes());
            }

            fn load<A: LoadArchive + ?Sized>(&mut self, ar: &mut A) {
                let mut buf = [0u8; core::mem::size_of::<$t>()];
                // On a short read the C++ leaves `obj` untouched; so do we.
                if ar.read_binary(&mut buf) {
                    *self = <$t>::from_ne_bytes(buf);
                }
            }
        }
    )*};
}

impl_local_serialize_arithmetic!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64
);

/// C++ `bool` is arithmetic and `memcpy`ed raw; see divergence 6 on
/// normalization of non-`0`/`1` bytes.
impl LocalSerialize for bool {
    fn save<A: SaveArchive + ?Sized>(&self, ar: &mut A) {
        ar.write_binary(&[u8::from(*self)]);
    }

    fn load<A: LoadArchive + ?Sized>(&mut self, ar: &mut A) {
        let mut buf = [0u8; 1];
        if ar.read_binary(&mut buf) {
            *self = buf[0] != 0;
        }
    }
}

// ---------------------------------------------------------------------------
// String  (C++ save_string / load_string)
// ---------------------------------------------------------------------------

/// C++ `std::string` via `save_string`/`load_string`. See divergences 7 and 8:
/// UTF-8 is validated, and a length prefix exceeding the remaining bytes clears
/// rather than allocating.
impl LocalSerialize for String {
    fn save<A: SaveArchive + ?Sized>(&self, ar: &mut A) {
        ar.save_string_fused(self.as_bytes());
    }

    fn load<A: LoadArchive + ?Sized>(&mut self, ar: &mut A) {
        let len = ar.load_len();
        if len > ar.remaining() {
            // C++ would resize(len) then fail the read, leaving a zero-filled
            // string of that length (or throw bad_alloc for an absurd len).
            ar.set_error();
            self.clear();
            return;
        }
        let mut buf = vec![0u8; len];
        if !ar.read_binary(&mut buf) {
            self.clear();
            return;
        }
        match String::from_utf8(buf) {
            Ok(s) => *self = s,
            Err(e) => {
                ar.set_error();
                *self = String::from_utf8_lossy(e.as_bytes()).into_owned();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Containers  (C++ save_vec/load_vec, save_list/load_list, save_map/load_map)
// ---------------------------------------------------------------------------

/// C++ `std::vector<T>` via `save_vec`/`load_vec`.
///
/// `Vec<u8>` is the byte-exact analogue of a C++ `std::string` payload, and the
/// right choice for non-UTF-8 bytes.
impl<T: LocalSerialize + Default> LocalSerialize for Vec<T> {
    fn save<A: SaveArchive + ?Sized>(&self, ar: &mut A) {
        ar.save_len(self.len());
        for elem in self {
            elem.save(ar);
        }
    }

    fn load<A: LoadArchive + ?Sized>(&mut self, ar: &mut A) {
        let len = ar.load_len();
        self.clear();
        // Reservation is a hint only — cap it so a corrupt prefix cannot
        // demand an unbounded allocation up front (divergence 8).
        self.reserve(len.min(ar.remaining()));
        for _ in 0..len {
            let mut elem = T::default();
            elem.load(ar);
            self.push(elem);
        }
    }
}

/// C++ `std::list<T>` via `save_list`/`load_list`.
///
/// Note the C++ `load_list` `emplace_back`s without clearing — loading
/// **appends**. Faithfully preserved (divergence 10).
impl<T: LocalSerialize + Default> LocalSerialize for LinkedList<T> {
    fn save<A: SaveArchive + ?Sized>(&self, ar: &mut A) {
        ar.save_len(self.len());
        for elem in self {
            elem.save(ar);
        }
    }

    fn load<A: LoadArchive + ?Sized>(&mut self, ar: &mut A) {
        let len = ar.load_len();
        for _ in 0..len {
            // C++: obj.emplace_back(); ar >> obj.back();
            let mut elem = T::default();
            elem.load(ar);
            self.push_back(elem);
        }
    }
}

/// C++ `std::unordered_map<K, V>` via `save_map`/`load_map`.
///
/// Save order is unspecified in both languages (hash order). The C++ `load_map`
/// does `obj[key] = val` without clearing — loading **merges**, overwriting
/// duplicate keys. Faithfully preserved (divergence 10).
impl<K, V> LocalSerialize for HashMap<K, V>
where
    K: LocalSerialize + Default + Eq + Hash,
    V: LocalSerialize + Default,
{
    fn save<A: SaveArchive + ?Sized>(&self, ar: &mut A) {
        ar.save_len(self.len());
        for (key, val) in self {
            key.save(ar);
            val.save(ar);
        }
    }

    fn load<A: LoadArchive + ?Sized>(&mut self, ar: &mut A) {
        let len = ar.load_len();
        for _ in 0..len {
            let mut key = K::default();
            let mut val = V::default();
            key.load(ar);
            val.load(ar);
            self.insert(key, val);
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience helpers (Rust-only — divergence 13)
// ---------------------------------------------------------------------------

/// Serialize `value` into a fresh, exactly-sized `Vec<u8>`.
pub fn to_bytes<T: LocalSerialize + ?Sized>(value: &T) -> Vec<u8> {
    let mut buf = Vec::new();
    let mut ar = LocalSaveArchive::new(&mut buf);
    ar.save_value(value);
    ar.finalize();
    buf
}

/// Deserialize a `T` from `bytes`.
///
/// `Err(())` when a read ran past the end or a `String` failed UTF-8
/// validation — i.e. exactly the conditions the C++ reports via `HLOG(kError)`
/// and then ignores.
#[allow(clippy::result_unit_err)]
pub fn from_bytes<T: LocalSerialize + Default>(bytes: &[u8]) -> Result<T, ()> {
    let mut ar = LocalLoadArchive::new(bytes);
    let mut value = T::default();
    ar.load_value(&mut value);
    if ar.has_error() {
        return Err(());
    }
    Ok(value)
}

/// Dry-run size of `value` (C++: build a `CalculateSizeArchive`, `<<`, `size()`).
pub fn calculate_size<T: LocalSerialize + ?Sized>(value: &T) -> usize {
    let mut ar = CalculateSizeArchive::new();
    ar.save_value(value);
    ar.size()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors `clio::run::LocalTaskInfo` — the exact shape the task archives
    /// push through this substrate.
    #[derive(Default, Debug, PartialEq, Eq, Clone)]
    struct TaskInfo {
        pid: u32,
        tid: u32,
        major: u64,
        unique: u64,
        node_id: u32,
        method_id: u32,
    }

    impl LocalSerialize for TaskInfo {
        fn save<A: SaveArchive + ?Sized>(&self, ar: &mut A) {
            self.pid.save(ar);
            self.tid.save(ar);
            self.major.save(ar);
            self.unique.save(ar);
            self.node_id.save(ar);
            self.method_id.save(ar);
        }

        fn load<A: LoadArchive + ?Sized>(&mut self, ar: &mut A) {
            self.pid.load(ar);
            self.tid.load(ar);
            self.major.load(ar);
            self.unique.load(ar);
            self.node_id.load(ar);
            self.method_id.load(ar);
        }
    }

    #[derive(Default, Debug, PartialEq)]
    struct Nested {
        name: String,
        payload: Vec<u8>,
        infos: Vec<TaskInfo>,
        flag: bool,
    }

    impl LocalSerialize for Nested {
        fn save<A: SaveArchive + ?Sized>(&self, ar: &mut A) {
            self.name.save(ar);
            self.payload.save(ar);
            self.infos.save(ar);
            self.flag.save(ar);
        }

        fn load<A: LoadArchive + ?Sized>(&mut self, ar: &mut A) {
            self.name.load(ar);
            self.payload.load(ar);
            self.infos.load(ar);
            self.flag.load(ar);
        }
    }

    fn roundtrip<T: LocalSerialize + Default + PartialEq + std::fmt::Debug>(value: &T) {
        let bytes = to_bytes(value);
        assert_eq!(
            bytes.len(),
            calculate_size(value),
            "CalculateSizeArchive disagrees with LocalSaveArchive"
        );
        let back: T = from_bytes(&bytes).expect("clean roundtrip");
        assert_eq!(&back, value);
    }

    // --- primitives ---------------------------------------------------------

    #[test]
    fn arithmetic_roundtrips_all_widths() {
        roundtrip(&0u8);
        roundtrip(&u8::MAX);
        roundtrip(&u16::MAX);
        roundtrip(&u32::MAX);
        roundtrip(&u64::MAX);
        roundtrip(&u128::MAX);
        roundtrip(&usize::MAX);
        roundtrip(&i8::MIN);
        roundtrip(&i16::MIN);
        roundtrip(&i32::MIN);
        roundtrip(&i64::MIN);
        roundtrip(&i128::MIN);
        roundtrip(&isize::MIN);
        roundtrip(&f32::MIN);
        roundtrip(&f64::MAX);
        roundtrip(&0.0f64);
        roundtrip(&true);
        roundtrip(&false);
    }

    #[test]
    fn float_edge_values_survive_bit_exactly() {
        // NaN != NaN, so compare bits rather than values.
        let bytes = to_bytes(&f64::NAN);
        let back: f64 = from_bytes(&bytes).unwrap();
        assert!(back.is_nan());

        let neg_zero: f64 = from_bytes(&to_bytes(&-0.0f64)).unwrap();
        assert!(neg_zero.is_sign_negative());
        assert_eq!(neg_zero, 0.0);

        let inf: f32 = from_bytes(&to_bytes(&f32::INFINITY)).unwrap();
        assert_eq!(inf, f32::INFINITY);
    }

    #[test]
    fn arithmetic_wire_format_is_native_memcpy() {
        // The C++ memcpys &obj for sizeof(T) bytes — no tag, no endian swap.
        assert_eq!(to_bytes(&0xDEAD_BEEFu32), 0xDEAD_BEEFu32.to_ne_bytes());
        assert_eq!(to_bytes(&(-1i64)), (-1i64).to_ne_bytes());
        assert_eq!(to_bytes(&1.5f64), 1.5f64.to_ne_bytes());
        assert_eq!(to_bytes(&true), vec![1u8]);
        assert_eq!(to_bytes(&false), vec![0u8]);
        assert_eq!(calculate_size(&0u128), 16);
    }

    #[test]
    fn bool_normalizes_nonzero_byte_to_true() {
        // Divergence 6: C++ memcpys the raw byte (UB for 2); we normalize.
        let mut ar = LocalLoadArchive::new(&[2u8]);
        let mut b = false;
        b.load(&mut ar);
        assert!(b);
        assert!(!ar.has_error());
    }

    #[test]
    fn short_read_leaves_value_and_offset_untouched() {
        // C++ read_binary logs and early-returns without advancing cur_off_.
        let buf = [1u8, 2, 3];
        let mut ar = LocalLoadArchive::new(&buf);
        let mut v = 0xAAAA_AAAAu32; // needs 4 bytes, only 3 available
        v.load(&mut ar);
        assert_eq!(v, 0xAAAA_AAAA, "destination must be untouched");
        assert_eq!(ar.cur_off(), 0, "offset must not advance");
        assert!(ar.has_error());
        assert_eq!(ar.remaining(), 3);
    }

    #[test]
    fn error_flag_latches_until_reset() {
        let buf = [0u8; 1];
        let mut ar = LocalLoadArchive::new(&buf);
        let mut v = 0u32;
        v.load(&mut ar); // fails
        assert!(ar.has_error());
        let mut ok = 0u8;
        ok.load(&mut ar); // succeeds, but the latch stays
        assert!(ar.has_error());
        ar.reset();
        assert!(!ar.has_error());
        assert_eq!(ar.cur_off(), 0);
    }

    #[test]
    fn zero_length_read_at_exact_end_succeeds() {
        let buf = [7u8];
        let mut ar = LocalLoadArchive::new(&buf);
        let mut v = 0u8;
        v.load(&mut ar);
        assert_eq!(v, 7);
        assert_eq!(ar.remaining(), 0);
        assert!(
            ar.read_binary(&mut []),
            "empty read at EOF is not an overrun"
        );
        assert!(!ar.has_error());
    }

    // --- strings ------------------------------------------------------------

    #[test]
    fn string_roundtrips_including_empty_and_unicode() {
        roundtrip(&String::new());
        roundtrip(&String::from("hello"));
        roundtrip(&String::from("naïve 🚀 日本語"));
        roundtrip(&"x".repeat(10_000));
    }

    #[test]
    fn string_wire_format_is_len_prefix_plus_bytes() {
        let bytes = to_bytes(&String::from("abc"));
        assert_eq!(bytes.len(), LEN_BYTES + 3);
        assert_eq!(&bytes[..LEN_BYTES], &3usize.to_ne_bytes());
        assert_eq!(&bytes[LEN_BYTES..], b"abc");

        // Empty string: prefix only, no payload.
        let empty = to_bytes(&String::new());
        assert_eq!(empty, 0usize.to_ne_bytes().to_vec());
        assert_eq!(calculate_size(&String::new()), LEN_BYTES);
    }

    #[test]
    fn fused_string_save_matches_unfused_default_impl() {
        // CalculateSizeArchive and LocalSaveArchive both override
        // save_string_fused; the trait default must agree byte-for-byte.
        struct Plain(Vec<u8>);
        impl SaveArchive for Plain {
            fn write_binary(&mut self, data: &[u8]) {
                self.0.extend_from_slice(data);
            }
            // deliberately does NOT override save_string_fused
        }
        let mut plain = Plain(Vec::new());
        String::from("fused?").save(&mut plain);
        assert_eq!(plain.0, to_bytes(&String::from("fused?")));
    }

    #[test]
    fn string_with_corrupt_length_prefix_clears_and_flags() {
        // Divergence 8: a huge prefix must not trigger a huge allocation.
        let mut bytes = usize::MAX.to_ne_bytes().to_vec();
        bytes.extend_from_slice(b"ab");
        let mut ar = LocalLoadArchive::new(&bytes);
        let mut s = String::from("previous");
        s.load(&mut ar);
        assert!(s.is_empty());
        assert!(ar.has_error());
    }

    #[test]
    fn string_with_truncated_length_prefix_yields_empty() {
        // load_len leaves its zero-init local alone on a short read → len 0.
        let bytes = [0u8; 2];
        let mut ar = LocalLoadArchive::new(&bytes);
        let mut s = String::from("previous");
        s.load(&mut ar);
        assert!(s.is_empty());
        assert!(ar.has_error());
    }

    #[test]
    fn string_with_invalid_utf8_flags_and_uses_replacement() {
        // Divergence 7: C++ std::string would accept these bytes verbatim.
        let mut bytes = 2usize.to_ne_bytes().to_vec();
        bytes.extend_from_slice(&[0xFF, 0xFE]);
        let mut ar = LocalLoadArchive::new(&bytes);
        let mut s = String::new();
        s.load(&mut ar);
        assert!(ar.has_error());
        assert_eq!(s, "\u{FFFD}\u{FFFD}");
        // Vec<u8> is the byte-exact escape hatch for the same payload.
        let raw: Vec<u8> = from_bytes(&bytes).unwrap();
        assert_eq!(raw, vec![0xFF, 0xFE]);
    }

    // --- containers ---------------------------------------------------------

    #[test]
    fn vec_roundtrips_including_empty_and_nested() {
        roundtrip(&Vec::<u32>::new());
        roundtrip(&vec![1u32, 2, 3]);
        roundtrip(&vec![0u8; 1000]);
        roundtrip(&vec![String::from("a"), String::new(), String::from("ccc")]);
        roundtrip(&vec![vec![1u64, 2], Vec::new(), vec![3]]);
    }

    #[test]
    fn vec_wire_format_is_len_prefix_plus_elements() {
        let bytes = to_bytes(&vec![1u16, 2u16]);
        assert_eq!(bytes.len(), LEN_BYTES + 4);
        assert_eq!(&bytes[..LEN_BYTES], &2usize.to_ne_bytes());
        assert_eq!(&bytes[LEN_BYTES..LEN_BYTES + 2], &1u16.to_ne_bytes());
        assert_eq!(&bytes[LEN_BYTES + 2..], &2u16.to_ne_bytes());
        // Matches C++ save_vec's arithmetic bulk-memcpy path byte-for-byte
        // (divergence 9): [len][raw element bytes].
        assert_eq!(to_bytes(&Vec::<u16>::new()), 0usize.to_ne_bytes().to_vec());
    }

    #[test]
    fn vec_load_replaces_existing_contents() {
        // C++ load_vec resizes then overwrites every slot.
        let bytes = to_bytes(&vec![9u32]);
        let mut ar = LocalLoadArchive::new(&bytes);
        let mut v = vec![1u32, 2, 3, 4];
        v.load(&mut ar);
        assert_eq!(v, vec![9]);
    }

    #[test]
    fn vec_with_corrupt_length_does_not_over_reserve() {
        // The reservation is capped by remaining(); the element loop then fails
        // each read, leaving Defaults — no unbounded allocation up front.
        let bytes = 4usize.to_ne_bytes().to_vec(); // says 4 elements, provides 0
        let mut ar = LocalLoadArchive::new(&bytes);
        let mut v: Vec<u32> = Vec::new();
        v.load(&mut ar);
        assert!(ar.has_error());
        assert_eq!(v, vec![0u32; 4], "failed reads leave Default elements");
    }

    #[test]
    fn list_load_appends_without_clearing() {
        // Divergence 10 / C++ load_list: emplace_back in a loop, no clear.
        let mut src = LinkedList::new();
        src.push_back(3u32);
        src.push_back(4u32);
        let bytes = to_bytes(&src);

        let mut dst = LinkedList::new();
        dst.push_back(1u32);
        dst.push_back(2u32);
        let mut ar = LocalLoadArchive::new(&bytes);
        dst.load(&mut ar);
        assert_eq!(dst.into_iter().collect::<Vec<_>>(), vec![1, 2, 3, 4]);

        // Empty list roundtrips through the same path.
        roundtrip(&LinkedList::<u32>::new());
        roundtrip(&src);
    }

    #[test]
    fn map_roundtrips_and_load_merges_overwriting_keys() {
        let mut src: HashMap<u32, String> = HashMap::new();
        src.insert(1, "one".into());
        src.insert(2, "two".into());
        roundtrip(&src);
        roundtrip(&HashMap::<u32, u64>::new());

        // Divergence 10 / C++ load_map: obj[key] = val, no clear.
        let bytes = to_bytes(&src);
        let mut dst: HashMap<u32, String> = HashMap::new();
        dst.insert(2, "STALE".into()); // overwritten by the load
        dst.insert(9, "kept".into()); // survives the load
        let mut ar = LocalLoadArchive::new(&bytes);
        dst.load(&mut ar);
        assert_eq!(dst.len(), 3);
        assert_eq!(dst[&1], "one");
        assert_eq!(dst[&2], "two");
        assert_eq!(dst[&9], "kept");
    }

    #[test]
    fn nested_struct_roundtrips_and_matches_dry_run_size() {
        let value = Nested {
            name: "task".into(),
            payload: vec![1, 2, 3, 4, 5],
            infos: vec![
                TaskInfo {
                    pid: 1,
                    tid: 2,
                    major: 3,
                    unique: 4,
                    node_id: 5,
                    method_id: 6,
                },
                TaskInfo::default(),
            ],
            flag: true,
        };
        roundtrip(&value);
        roundtrip(&Nested::default());

        // Hand-computed, to pin the format rather than restate the impl.
        // TaskInfo is packed on the wire: 4+4+8+8+4+4 = 32 bytes, with none of
        // the padding its in-memory layout carries (see divergence 4).
        const TASK_INFO: usize = 32;
        let expect = (LEN_BYTES + 4)                    // name: "task"
            + (LEN_BYTES + 5)                           // payload: 5 bytes
            + (LEN_BYTES + 2 * TASK_INFO)               // infos: 2 elements
            + 1; // flag
        assert_eq!(calculate_size(&value), expect);
        assert_eq!(expect, 98);
    }

    #[test]
    fn msg_type_enum_roundtrips_as_u8() {
        // Divergence 12: the C++ auto-serializes `enum class LocalMsgType :
        // uint8_t` via its underlying type; Rust does it by hand.
        #[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
        #[repr(u8)]
        enum LocalMsgType {
            #[default]
            SerializeIn = 0,
            SerializeOut = 1,
        }

        impl LocalSerialize for LocalMsgType {
            fn save<A: SaveArchive + ?Sized>(&self, ar: &mut A) {
                (*self as u8).save(ar);
            }
            fn load<A: LoadArchive + ?Sized>(&mut self, ar: &mut A) {
                let mut raw = 0u8;
                raw.load(ar);
                *self = match raw {
                    1 => LocalMsgType::SerializeOut,
                    _ => LocalMsgType::SerializeIn,
                };
            }
        }

        assert_eq!(to_bytes(&LocalMsgType::SerializeOut), vec![1u8]);
        roundtrip(&LocalMsgType::SerializeIn);
        roundtrip(&LocalMsgType::SerializeOut);
    }

    // --- buffer growth / archive mechanics ----------------------------------

    #[test]
    fn growth_doubles_from_64_and_finalize_truncates() {
        let mut buf = Vec::new();
        let mut ar = LocalSaveArchive::new(&mut buf);
        ar.save_value(&1u32);
        // C++: empty buffer -> new_cap = 64; the *length* (not just capacity)
        // becomes 64 while only 4 bytes are live.
        assert_eq!(ar.serialized_size(), 4);
        ar.finalize();
        assert_eq!(buf.len(), 4);

        // Force the doubling loop: 64 -> 128 -> 256.
        let mut buf2 = Vec::new();
        let mut ar2 = LocalSaveArchive::new(&mut buf2);
        ar2.save_value(&vec![0u8; 200]);
        assert_eq!(ar2.serialized_size(), LEN_BYTES + 200);
        ar2.finalize();
        assert_eq!(buf2.len(), LEN_BYTES + 200);
    }

    #[test]
    fn buffer_len_over_reports_until_finalize() {
        let mut buf = Vec::new();
        {
            let mut ar = LocalSaveArchive::new(&mut buf);
            ar.save_value(&7u8);
            assert_eq!(ar.serialized_size(), 1);
        }
        // finalize() not called: the grown length is still visible (C++ parity).
        assert_eq!(buf.len(), MIN_CAP);
        assert_eq!(buf[0], 7);
    }

    #[test]
    fn single_write_larger_than_double_grows_in_one_step() {
        let mut buf = Vec::new();
        let mut ar = LocalSaveArchive::new(&mut buf);
        let big = vec![0xABu8; 5000];
        ar.write_binary(&big);
        assert_eq!(ar.serialized_size(), 5000);
        ar.finalize();
        assert_eq!(buf.len(), 5000);
        assert!(buf.iter().all(|&b| b == 0xAB));
    }

    #[test]
    fn append_ctor_starts_at_existing_len() {
        // C++ LocalSerialize(DataT&, bool) — cur_off_(data.size()).
        let mut buf = vec![0xFFu8, 0xFE];
        {
            let mut ar = LocalSaveArchive::new_append(&mut buf);
            assert_eq!(ar.serialized_size(), 2);
            ar.save_value(&1u8);
            ar.finalize();
        }
        assert_eq!(buf, vec![0xFF, 0xFE, 1]);
    }

    #[test]
    fn plain_ctor_overwrites_from_offset_zero() {
        let mut buf = vec![9u8; 16];
        {
            let mut ar = LocalSaveArchive::new(&mut buf);
            ar.save_value(&1u8);
            ar.finalize();
        }
        assert_eq!(buf, vec![1u8]);
    }

    #[test]
    fn save_archive_reset_rewinds_and_overwrites() {
        let mut buf = Vec::new();
        let mut ar = LocalSaveArchive::new(&mut buf);
        ar.save_value(&String::from("first"));
        ar.reset();
        assert_eq!(ar.serialized_size(), 0);
        ar.save_value(&String::from("second"));
        ar.finalize();
        assert_eq!(buf, to_bytes(&String::from("second")));
    }

    #[test]
    fn finalize_is_idempotent_and_empty_save_yields_empty_buffer() {
        let mut buf = Vec::new();
        let mut ar = LocalSaveArchive::new(&mut buf);
        ar.finalize();
        ar.finalize();
        assert!(buf.is_empty());
    }

    #[test]
    fn chained_saves_and_loads_are_sequential() {
        // C++ `ar << a << b << c;` / `ar >> a >> b >> c;`
        let mut buf = Vec::new();
        let mut ar = LocalSaveArchive::new(&mut buf);
        ar.save_value(&1u32)
            .save_value(&String::from("mid"))
            .save_value(&2u64);
        ar.finalize();

        let mut lar = LocalLoadArchive::new(&buf);
        let (mut a, mut b, mut c) = (0u32, String::new(), 0u64);
        lar.load_value(&mut a).load_value(&mut b).load_value(&mut c);
        assert_eq!((a, b.as_str(), c), (1, "mid", 2));
        assert!(!lar.has_error());
        assert_eq!(lar.remaining(), 0);
    }

    // --- CalculateSizeArchive -----------------------------------------------

    #[test]
    fn dry_run_matches_real_size_for_every_shape() {
        macro_rules! agrees {
            ($v:expr) => {{
                let v = $v;
                assert_eq!(calculate_size(&v), to_bytes(&v).len());
            }};
        }
        agrees!(0u8);
        agrees!(0u128);
        agrees!(true);
        agrees!(String::new());
        agrees!(String::from("some text"));
        agrees!(Vec::<u32>::new());
        agrees!(vec![1u32; 50]);
        agrees!(vec![String::from("a"), String::from("bb")]);
        let mut m = HashMap::new();
        m.insert(1u32, String::from("v"));
        agrees!(m);
        let mut l = LinkedList::new();
        l.push_back(1u8);
        agrees!(l);
    }

    #[test]
    fn dry_run_accumulates_and_resets() {
        let mut ar = CalculateSizeArchive::new();
        assert_eq!(ar.size(), 0);
        ar.save_value(&1u32).save_value(&1u64);
        assert_eq!(ar.size(), 12);
        ar.reset();
        assert_eq!(ar.size(), 0);
        assert_eq!(CalculateSizeArchive::default(), CalculateSizeArchive::new());
    }

    #[test]
    fn dry_run_saturates_instead_of_wrapping() {
        // Divergence 11: C++ `cur_off_ += size` would wrap past SIZE_MAX.
        let mut ar = CalculateSizeArchive::new();
        ar.write_binary(&[0u8; 8]);
        // Simulate an absurd accumulated size without allocating one.
        for _ in 0..4 {
            ar.save_string_fused(&[]);
        }
        assert_eq!(ar.size(), 8 + 4 * LEN_BYTES);

        let mut sat = CalculateSizeArchive::new();
        // Drive cur_off to the ceiling via the (allocation-free) counter.
        sat.cur_off = usize::MAX - 1;
        sat.write_binary(&[0u8; 8]);
        assert_eq!(sat.size(), usize::MAX, "must saturate, not wrap");
        sat.save_string_fused(&[0u8; 4]);
        assert_eq!(sat.size(), usize::MAX);
    }

    // --- concurrency --------------------------------------------------------

    #[test]
    fn archives_are_independent_across_threads() {
        // The archives hold no global/thread-local state (project rule: never
        // thread_local!) — each is a self-contained cursor over its own buffer.
        let handles: Vec<_> = (0..8u32)
            .map(|i| {
                std::thread::spawn(move || {
                    let value = Nested {
                        name: format!("task-{i}"),
                        payload: vec![i as u8; i as usize],
                        infos: vec![TaskInfo {
                            pid: i,
                            tid: i * 2,
                            major: u64::from(i),
                            unique: u64::from(i) * 7,
                            node_id: i,
                            method_id: i,
                        }],
                        flag: i % 2 == 0,
                    };
                    let bytes = to_bytes(&value);
                    assert_eq!(bytes.len(), calculate_size(&value));
                    let back: Nested = from_bytes(&bytes).unwrap();
                    assert_eq!(back, value);
                })
            })
            .collect();
        for h in handles {
            h.join().expect("worker panicked");
        }
    }

    // --- helpers ------------------------------------------------------------

    #[test]
    fn from_bytes_reports_overrun_and_trailing_bytes_are_ignored() {
        let bytes = to_bytes(&1u32);
        assert!(
            from_bytes::<u64>(&bytes).is_err(),
            "overrun must be an error"
        );

        // Trailing bytes are not an error — the C++ never checks for them.
        let mut extended = bytes.clone();
        extended.extend_from_slice(&[0xAA; 4]);
        assert_eq!(from_bytes::<u32>(&extended).unwrap(), 1);
        assert!(from_bytes::<u32>(&[]).is_err());
    }
}
