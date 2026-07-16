// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).

//! Architecture-portable **global** (cross-node) serialization archives.
//!
//! Ports `clio_ctp/data_structures/serialization/global_serialize.h` +
//! `serialize_common.h` (the `GlobalSerialize`/`GlobalDeserialize` byte
//! engines) together with the archive shape layered on top of them in
//! `clio_runtime/task_archives.h` (`NetTaskArchive`, `SaveTaskArchive`,
//! `LoadTaskArchive`), including the bulk-transfer concept
//! (`BULK_EXPOSE`/`BULK_XFER`, defined in `clio_ctp/lightbeam/lightbeam.h`)
//! and `PushPod`/`PopPod`.
//!
//! "Global" = the wire format used between nodes. Unlike the local (SHM)
//! archives, it never batch-`memcpy`s across struct fields: every field is
//! written individually, so the format is independent of struct layout,
//! padding and alignment. This module reproduces that byte-for-byte.
//!
//! # C++ → Rust name mapping
//!
//! | C++ | Rust |
//! |---|---|
//! | `ctp::ipc::GlobalSerialize<std::vector<char>>` | [`GlobalSerialize`] |
//! | `ctp::ipc::GlobalDeserialize<std::vector<char>>` | [`GlobalDeserialize`] |
//! | `ar << obj` / `ar & obj` / `ar.base(obj)` (saving) | [`GlobalSerialize::save`] |
//! | `ar >> obj` / `ar & obj` / `ar.base(obj)` (loading) | [`GlobalDeserialize::load`] |
//! | `ar(a, b, c)` / `ar.range(a, b, c)` | successive `save`/`load` calls (divergence 1) |
//! | `Ar::is_saving` / `Ar::is_loading` (tag types) | `IS_SAVING` / `IS_LOADING` consts |
//! | `has_save_fun_v` / `has_save_cls_v` (SFINAE detect) | [`GlobalSave`] trait bound |
//! | `has_load_fun_v` / `has_load_cls_v` (SFINAE detect) | [`GlobalLoad`] trait bound |
//! | `is_serializeable_v<Ar, T>` | `T: GlobalSave + GlobalLoad` |
//! | `write_binary(const char*, size_t)` | [`GlobalSerialize::write_binary`] |
//! | `read_binary(char*, size_t)` | [`GlobalDeserialize::read_binary`] |
//! | `save_string_fused(const char*, size_t)` | [`GlobalSerialize::save_string_fused`] |
//! | `save_string` / `load_string` | `impl GlobalSave/GlobalLoad for String` |
//! | `save_vec` / `load_vec` | `impl GlobalSave/GlobalLoad for Vec<T>` |
//! | `save_list` / `load_list` | `impl GlobalSave/GlobalLoad for VecDeque<T>` |
//! | `save_map` / `load_map` | `impl GlobalSave/GlobalLoad for HashMap<K, V>` |
//! | `Finalize()` | [`GlobalSerialize::finalize`] |
//! | `resize_for_overwrite` | (not needed — Rust `Vec` has no zero-fill obligation) |
//! | `BULK_EXPOSE` / `BULK_XFER` | [`BULK_EXPOSE`] / [`BULK_XFER`] |
//! | `ctp::lbm::Bulk` | [`Bulk`] |
//! | `clio::run::MsgType` | [`MsgType`] |
//! | `clio::run::TaskInfo` | [`TaskInfo`] |
//! | `clio::run::TaskId` | [`TaskId`] |
//! | `clio::run::PoolId` (= `UniqueId`) | [`PoolId`] (= [`UniqueId`]) |
//! | `clio::run::NetTaskArchive` (base class) | [`NetTaskArchive`] (composed field `meta`) |
//! | `clio::run::SaveTaskArchive` | [`SaveTaskArchive`] |
//! | `clio::run::LoadTaskArchive` | [`LoadTaskArchive`] |
//! | `PushPod(bool)` / `PopPod()` | [`SaveTaskArchive::push_pod`] / [`SaveTaskArchive::pop_pod`] |
//! | `SaveTaskArchive::bulk(ptr, size, flags)` | [`SaveTaskArchive::bulk`] |
//! | `LoadTaskArchive::bulk(ptr&, size, flags)` | [`LoadTaskArchive::bulk`] |
//! | `SaveTaskArchive::serialize(Ar&)` | [`SaveTaskArchive::save_archive`] |
//! | `LoadTaskArchive::serialize(Ar&)` | [`LoadTaskArchive::load_archive`] |
//! | `daemon_allocated_bulk_count_` | [`LoadTaskArchive::daemon_allocated_bulk_count`] |
//! | `CLIO_IPC->AllocateBuffer(size)` | [`BulkAllocator::allocate_buffer`] (injected) |
//!
//! # Wire format (exact byte order)
//!
//! All scalars are written **native-endian**, with **no padding, no
//! alignment and no tags** — exactly `sizeof(T)` bytes of the object
//! representation, mirroring the C++ `memcpy(&obj, sizeof(T))`. Fields are
//! emitted strictly in declaration order.
//!
//! Primitives:
//!
//! | Type | Bytes |
//! |---|---|
//! | `bool` | 1 (`0`/`1`) |
//! | `u8`/`i8` | 1 |
//! | `u16`/`i16` | 2 |
//! | `u32`/`i32`/`f32` | 4 |
//! | `u64`/`i64`/`f64`/`usize` | 8 |
//! | enum (`MsgType`) | size of the C++ underlying type (`u8` here) |
//!
//! Composites (`LEN` is always a `usize`/`size_t`, i.e. 8 bytes):
//!
//! ```text
//! String / Vec<u8>    : [LEN:8][bytes:LEN]
//! Vec<T> / VecDeque<T>: [LEN:8][elem_0][elem_1]...[elem_{LEN-1}]
//! HashMap<K, V>       : [LEN:8][(k_0)(v_0)][(k_1)(v_1)]...
//! ```
//!
//! `Vec<T>` for arithmetic `T` is byte-identical whether written per-element
//! or as one `memcpy` of the array (arrays are packed), so the single generic
//! Rust impl matches both C++ `save_vec` branches. `std::list` and
//! `std::vector` share a wire format (`save_list` == `save_vec`'s
//! non-arithmetic branch), hence `VecDeque` maps onto the same bytes.
//!
//! Domain structs, in declaration order:
//!
//! ```text
//! UniqueId / PoolId (8B)  : [major:u32][minor:u32]
//! TaskId (32B)            : [pid:u32][tid:u32][major:u32][replica_id:u32]
//!                           [unique:u32][node_id:u32][net_key:usize:8]
//! TaskInfo (44B)          : [task_id:32][pool_id:8][method_id:u32]
//! Bulk (12B)              : [size:usize:8][flags:u32]      // data ptr NOT sent
//! ```
//!
//! The whole-archive form — `SaveTaskArchive::serialize` on the sending node,
//! `LoadTaskArchive::serialize` on the receiving node — is the payload that
//! actually crosses the network. Both C++ classes emit the identical
//! sequence (`SaveTaskArchive` writes `buffer_` where `LoadTaskArchive`
//! writes `data_`), which is what makes Save→wire→Load work:
//!
//! ```text
//! [send.len       : usize:8][Bulk × send.len         ]  // 12B each
//! [recv.len       : usize:8][Bulk × recv.len         ]
//! [send_bulks     : usize:8]                            // count of BULK_XFER in send
//! [recv_bulks     : usize:8]                            // count of BULK_XFER in recv
//! [task_infos.len : usize:8][TaskInfo × task_infos.len]  // 44B each
//! [msg_type       : u8:1   ]
//! [client_port    : i32:4  ]
//! [payload.len    : usize:8][payload bytes           ]  // buffer_ / data_
//! ```
//!
//! Note the payload is `Finalize()`d (truncated to the write cursor) *before*
//! being length-prefixed, so the over-allocated tail never reaches the wire.
//!
//! # Semantic divergences from the C++
//!
//! 1. **No variadic `operator()`/`range()`.** C++ folds an argpack over
//!    `base(arg)`. Rust has no variadic generics, so callers make successive
//!    [`GlobalSerialize::save`] / [`GlobalDeserialize::load`] calls, which
//!    produce the identical byte stream. `range()` and `operator()` are the
//!    same function in the C++ `Global*` archives (`range` forwards to
//!    `operator()`), so nothing is lost beyond call syntax.
//! 2. **`save`/`load` split into two traits.** C++ detects any of four shapes
//!    (`serialize` free fn / `save`+`load` free fns / `serialize` method /
//!    `save`+`load` methods) via SFINAE and reuses one `serialize` body for
//!    both directions. Rust has no `if constexpr` on archive direction, so
//!    [`GlobalSave`] and [`GlobalLoad`] are separate traits; the
//!    bidirectional single-body idiom is not expressible and each type
//!    implements both sides explicitly.
//! 3. **Read overflow is an error, not log-and-continue.** C++
//!    `GlobalDeserialize::read_binary` logs `HLOG(kError)` and *returns
//!    without advancing*, leaving the destination object at whatever value it
//!    had — silent corruption the caller cannot detect. Rust returns
//!    [`SerializeError::UnexpectedEof`]; the cursor likewise does not advance.
//! 4. **`usize` is always 8 bytes on the wire.** C++ writes `sizeof(size_t)`,
//!    which is 8 on every supported (64-bit) target but would be 4 on a
//!    32-bit build, silently changing the format. We pin `usize` to 8 bytes
//!    so a 32-bit Rust build stays wire-compatible with 64-bit C++. Loading
//!    errors with [`SerializeError::WidthOverflow`] if a received value
//!    exceeds `usize::MAX` on a 32-bit target.
//! 5. **Native endianness is preserved (not fixed to LE).** Faithful to the
//!    C++ `memcpy`: a big-endian peer is already incompatible with the C++
//!    implementation, and byte-swapping here would break parity with it.
//!    Cross-endian transport is therefore unsupported, exactly as in C++.
//! 6. **Invalid `MsgType` is rejected.** C++ `base()` `static_cast`s the
//!    underlying byte to the enum without validation (values ≥ 3 flow onward
//!    as an invalid enumerator). Rust returns
//!    [`SerializeError::InvalidMsgType`].
//! 7. **Growth policy: overflow saturates instead of looping.** C++
//!    `while (new_cap < new_off) new_cap *= 2;` would spin/overflow for a
//!    `new_off` near `SIZE_MAX`. The doubling schedule (`0 → 64`, then ×2) is
//!    reproduced exactly, but capacity is clamped to `new_off` on overflow
//!    rather than wrapping. Allocation failure aborts (Rust `Vec`) where C++
//!    would throw `std::bad_alloc`.
//! 8. **`LoadTaskArchive` is not self-referential.** C++ holds a
//!    `GlobalDeserialize` pointing into its own `data_` member (and
//!    placement-`new`s it after a move / after `serialize` refills `data_`).
//!    Rust stores `data: Vec<u8>` plus an explicit `cur_off` cursor and builds
//!    a borrowing [`GlobalDeserialize`] per call, removing the self-reference,
//!    the placement-new, and the move-constructor hazard.
//! 9. **`Bulk` drops `desc`/`mr`/`FullPtr`.** C++ `Bulk` holds
//!    `FullPtr<char> data` (raw pointer *and* `ShmPtr` in one) plus
//!    `void* desc` / `void* mr` for RDMA registration. Neither raw host
//!    pointers nor RDMA handles are portable into this crate, so [`Bulk`]
//!    keeps `data: ShmPtr<u8>` and models `desc != nullptr` as the boolean
//!    [`Bulk::has_desc`]. None of these fields are serialized in C++ either
//!    (`Bulk::serialize` sends only `size` + `flags`), so the **wire format is
//!    unaffected**.
//! 10. **Transport `Expose()` is not called.** C++ `SaveTaskArchive::bulk`
//!     calls `CLIO_IPC->ToFullPtr(ptr)` then, if a `ctp::lbm::Transport*` is
//!     attached, replaces the descriptor with `lbm_transport_->Expose(...)`
//!     for RDMA registration. `ctp::lbm::Transport` and the IPC manager live
//!     in layers above this crate; [`SaveTaskArchive::bulk`] performs the
//!     descriptor bookkeeping (push + `send_bulks` accounting) and leaves
//!     registration to the caller. `SetTransport` is therefore absent.
//! 11. **`LoadTaskArchive::bulk` takes an injected allocator.** C++ calls the
//!     global `CLIO_IPC->AllocateBuffer` and `memcpy`s through raw pointers.
//!     Rust routes both through the [`BulkAllocator`] trait so the branch
//!     structure (zmq-`desc` copy / socket-sentinel copy / SHM zero-copy /
//!     null→allocate) is ported verbatim while allocation and copying remain
//!     the runtime's job. With no allocator attached, paths that C++ would
//!     allocate on return [`SerializeError::NoBulkAllocator`] rather than
//!     silently yielding a null pointer.
//! 12. **Bulk exhaustion returns `Err` *and* nulls the pointer.** C++ logs
//!     `HLOG(kError)` and sets the pointer null, which callers routinely
//!     ignore. Rust does both: the out-pointer is nulled (faithful) and
//!     [`SerializeError::BulkExhausted`] is returned (detectable).
//! 13. **`PushPod`/`PopPod` are not a stack** — faithful to C++, where
//!     `PopPod()` unconditionally assigns `is_pod_ = false` rather than
//!     restoring a previous value, so nesting does not work. Reproduced
//!     exactly (see [`SaveTaskArchive::pop_pod`]); the names are the C++
//!     names, not a description of the behavior.
//! 14. **POD mode does not alter the byte layout.** In C++, `is_pod_` only
//!     selects `range(args...)` over the `SerializeArg` fold; for non-`Task`
//!     arguments both paths emit identical bytes. The sole difference is that
//!     POD mode skips the `std::is_base_of_v<Task, T>` dispatch that would
//!     recurse into `Task::SerializeIn`/`SerializeOut`. `Task` lives in
//!     context-runtime, so this crate stores the flag
//!     ([`SaveTaskArchive::is_pod`]) for the runtime layer to honor and never
//!     recurses itself.
//! 15. **`std::unordered_map` iteration order.** [`HashMap`] and C++
//!     `unordered_map` enumerate entries in different orders, so the *byte
//!     sequence* of a serialized map differs between implementations even
//!     though the decoded contents match. This is inherent to the C++ format
//!     (itself unstable across libstdc++/libc++/rehashes) and matters only if
//!     bytes are compared directly. Duplicate keys resolve last-wins on load,
//!     matching C++ `obj[key] = val`.
//! 16. **Move semantics / copy deletion are not modeled.** C++ deletes the
//!     copy ctor and hand-writes move ctors to re-seat the archive's
//!     serializer. Rust moves are memcpy-and-invalidate and nothing
//!     self-references, so no such machinery exists. Notably the C++
//!     `SaveTaskArchive(SaveTaskArchive&&)` re-seats the serializer with
//!     `cur_off_ = buffer_.size()` — the *capacity*-inflated size, not the
//!     logical length — so appending to a moved archive that was never
//!     `Finalize()`d embeds the zero-filled tail in the stream.
//!     [`GlobalSerialize::from_buffer`] reproduces the append-mode
//!     constructor and documents the same hazard; this port simply has no
//!     move ctor that can trip it.
//! 17. **`GetData()` returns `Vec<u8>`, not `std::string`.** The payload is
//!     arbitrary binary; C++ round-trips it through `std::string` only
//!     because that is its byte-buffer of convenience. `String` in Rust must
//!     be UTF-8, so `Vec<u8>` is the faithful type. Relatedly,
//!     `GlobalLoad for String` decodes non-UTF-8 bytes lossily (C++
//!     `std::string` accepts any bytes) — use `Vec<u8>`, which has an
//!     identical wire format, when the bytes are not text.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::hash::Hash;

use ctp_types::Bitfield32;
use ctp_memory::AllocatorId;

// ---------------------------------------------------------------------------
// Bulk flags (clio_ctp/lightbeam/lightbeam.h)
// ---------------------------------------------------------------------------

/// `BULK_EXPOSE` — bulk metadata is sent, but no data transfer occurs.
/// The sender exposes the region for the peer to read/fill.
pub const BULK_EXPOSE: u32 = 1 << 0;

/// `BULK_XFER` — bulk is marked for data transmission (payload rides along).
pub const BULK_XFER: u32 = 1 << 1;

/// Sentinel `AllocatorId` stamped by `SocketTransport::RecvBulks` on buffers
/// obtained from `std::malloc` (C++ `AllocatorId(UINT32_MAX - 1, UINT32_MAX - 1)`).
/// Such buffers are freed by `ClearRecvHandles` right after the load, so the
/// task must copy out of them rather than alias them.
pub const SOCKET_RECV_SENTINEL: AllocatorId = AllocatorId::new(u32::MAX - 1, u32::MAX - 1);

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Failure modes of the global archives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SerializeError {
    /// Read past the end of the buffer. C++ logs and leaves the destination
    /// untouched (divergence 3).
    UnexpectedEof {
        /// Bytes the read requested.
        needed: usize,
        /// Bytes actually available from the cursor.
        remaining: usize,
    },
    /// A `size_t` on the wire (8 bytes) does not fit this target's `usize`
    /// (divergence 4). Only reachable on 32-bit targets.
    WidthOverflow(u64),
    /// `MsgType` byte outside `0..=2` (divergence 6).
    InvalidMsgType(u8),
    /// [`LoadTaskArchive::bulk`] called more times than there are `recv`
    /// descriptors (divergence 12).
    BulkExhausted {
        /// The index that was requested.
        index: usize,
        /// Number of available `recv` descriptors.
        len: usize,
    },
    /// A bulk path that must allocate was taken with no [`BulkAllocator`]
    /// attached (divergence 11).
    NoBulkAllocator,
}

impl fmt::Display for SerializeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof { needed, remaining } => write!(
                f,
                "read beyond end of data: needed {needed} byte(s), {remaining} remaining"
            ),
            Self::WidthOverflow(v) => {
                write!(f, "wire size_t {v} does not fit in this target's usize")
            }
            Self::InvalidMsgType(v) => write!(f, "invalid MsgType discriminant: {v}"),
            Self::BulkExhausted { index, len } => write!(
                f,
                "recv bulk vector exhausted: index {index} of {len} descriptor(s)"
            ),
            Self::NoBulkAllocator => {
                f.write_str("bulk path requires a BulkAllocator but none is attached")
            }
        }
    }
}

impl std::error::Error for SerializeError {}

/// Convenience alias for archive results.
pub type Result<T> = std::result::Result<T, SerializeError>;

// ---------------------------------------------------------------------------
// Traits (C++ has_save_fun_v / has_load_fun_v SFINAE detection)
// ---------------------------------------------------------------------------

/// Types writable by [`GlobalSerialize`] (C++ `save`/`serialize` detection).
pub trait GlobalSave {
    /// Append `self` to the archive in C++ wire order.
    fn global_save(&self, ar: &mut GlobalSerialize);
}

/// Types readable by [`GlobalDeserialize`] (C++ `load`/`serialize` detection).
pub trait GlobalLoad: Sized {
    /// Decode one value from the archive, advancing its cursor.
    fn global_load(ar: &mut GlobalDeserialize<'_>) -> Result<Self>;
}

// ---------------------------------------------------------------------------
// GlobalSerialize
// ---------------------------------------------------------------------------

/// Architecture-portable binary serializer (C++ `ctp::ipc::GlobalSerialize`).
///
/// Serializes each field individually — no batch `memcpy` across struct
/// fields — so the wire format is independent of struct layout, padding and
/// alignment on any particular architecture.
///
/// Like the C++, the backing buffer grows on a doubling schedule while the
/// logical length is tracked separately in `cur_off`; [`finalize`] commits the
/// cursor to the buffer's length.
///
/// [`finalize`]: GlobalSerialize::finalize
#[derive(Debug, Default, Clone)]
pub struct GlobalSerialize {
    data: Vec<u8>,
    cur_off: usize,
}

impl GlobalSerialize {
    /// C++ `using is_saving = std::true_type;`
    pub const IS_SAVING: bool = true;
    /// C++ `using is_loading = std::false_type;`
    pub const IS_LOADING: bool = false;
    /// C++ `using supports_range_ops = std::true_type;`
    pub const SUPPORTS_RANGE_OPS: bool = true;

    /// C++ `GlobalSerialize(DataT &data)` — resets the buffer to empty.
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            cur_off: 0,
        }
    }

    /// C++ `GlobalSerialize(DataT &data, bool)` — *append* mode: the cursor
    /// starts at `data.len()`.
    ///
    /// # Hazard (faithful to C++)
    ///
    /// The cursor is seeded from the buffer's **length**, which for a buffer
    /// produced by this archive and never [`finalize`]d includes the
    /// zero-filled growth tail. Appending then writes *after* that tail,
    /// embedding the zeros in the stream. Pass a buffer obtained from
    /// [`into_bytes`] / [`get_buffer`] (both finalize) to avoid this.
    ///
    /// [`finalize`]: GlobalSerialize::finalize
    /// [`into_bytes`]: GlobalSerialize::into_bytes
    /// [`get_buffer`]: GlobalSerialize::get_buffer
    pub fn from_buffer(data: Vec<u8>) -> Self {
        let cur_off = data.len();
        Self { data, cur_off }
    }

    /// C++ `buffer_.reserve(n)`.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            data: Vec::with_capacity(cap),
            cur_off: 0,
        }
    }

    /// Current write cursor — the number of meaningful bytes written
    /// (C++ `cur_off_`).
    pub fn len(&self) -> usize {
        self.cur_off
    }

    /// True when nothing has been written yet.
    pub fn is_empty(&self) -> bool {
        self.cur_off == 0
    }

    /// C++ `Finalize()` — commit the local offset to the buffer's size,
    /// discarding the over-allocated tail.
    pub fn finalize(&mut self) {
        self.data.truncate(self.cur_off);
    }

    /// Finalize and borrow the exact bytes written.
    pub fn get_buffer(&mut self) -> &[u8] {
        self.finalize();
        &self.data
    }

    /// Finalize and take ownership of the exact bytes written.
    pub fn into_bytes(mut self) -> Vec<u8> {
        self.finalize();
        self.data
    }

    /// Reproduces the C++ capacity schedule: `0 → 64`, then repeated doubling
    /// until `new_off` fits. Saturates instead of overflowing (divergence 7).
    fn grow_for(&mut self, new_off: usize) {
        if new_off <= self.data.len() {
            return;
        }
        let mut new_cap = self.data.len();
        if new_cap == 0 {
            new_cap = 64;
        }
        while new_cap < new_off {
            new_cap = match new_cap.checked_mul(2) {
                Some(c) => c,
                None => new_off,
            };
        }
        // C++ std::vector::resize value-initializes the new tail to 0.
        self.data.resize(new_cap, 0);
    }

    /// C++ `write_binary(const char *data, size_t size)` — raw bytes, no
    /// length prefix.
    pub fn write_binary(&mut self, data: &[u8]) -> &mut Self {
        let new_off = self.cur_off + data.len();
        self.grow_for(new_off);
        if !data.is_empty() {
            self.data[self.cur_off..new_off].copy_from_slice(data);
        }
        self.cur_off = new_off;
        self
    }

    /// C++ `save_string_fused(const char *str_data, size_t len)` — length
    /// prefix (`size_t`, 8 bytes) + character data under one capacity check.
    pub fn save_string_fused(&mut self, str_data: &[u8]) -> &mut Self {
        let len = str_data.len();
        let total = USIZE_WIRE_BYTES + len;
        let new_off = self.cur_off + total;
        self.grow_for(new_off);
        let prefix_end = self.cur_off + USIZE_WIRE_BYTES;
        self.data[self.cur_off..prefix_end].copy_from_slice(&usize_to_wire(len));
        if len > 0 {
            self.data[prefix_end..new_off].copy_from_slice(str_data);
        }
        self.cur_off = new_off;
        self
    }

    /// C++ `operator<<` / `operator&` / `base(obj)` on the saving side.
    pub fn save<T: GlobalSave + ?Sized>(&mut self, obj: &T) -> &mut Self {
        obj.global_save(self);
        self
    }
}

// ---------------------------------------------------------------------------
// GlobalDeserialize
// ---------------------------------------------------------------------------

/// Architecture-portable binary deserializer
/// (C++ `ctp::ipc::GlobalDeserialize`).
///
/// Reads each field individually, matching [`GlobalSerialize`]'s format.
#[derive(Debug, Clone)]
pub struct GlobalDeserialize<'a> {
    data: &'a [u8],
    cur_off: usize,
}

impl<'a> GlobalDeserialize<'a> {
    /// C++ `using is_saving = std::false_type;`
    pub const IS_SAVING: bool = false;
    /// C++ `using is_loading = std::true_type;`
    pub const IS_LOADING: bool = true;
    /// C++ `using supports_range_ops = std::true_type;`
    pub const SUPPORTS_RANGE_OPS: bool = true;

    /// C++ `GlobalDeserialize(const DataT &data)` — cursor at 0.
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, cur_off: 0 }
    }

    /// Start reading at an explicit cursor (used by [`LoadTaskArchive`], which
    /// owns its buffer and tracks the cursor itself — divergence 8).
    pub fn at(data: &'a [u8], cur_off: usize) -> Self {
        Self { data, cur_off }
    }

    /// Current read cursor (C++ `cur_off_`).
    pub fn cur_off(&self) -> usize {
        self.cur_off
    }

    /// Bytes left to read.
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.cur_off)
    }

    /// C++ `read_binary(char *data, size_t size)`.
    ///
    /// Unlike C++ (which logs `HLOG(kError)` and returns silently), an
    /// over-read is reported as [`SerializeError::UnexpectedEof`] and the
    /// cursor does not advance (divergence 3).
    pub fn read_binary(&mut self, out: &mut [u8]) -> Result<()> {
        let size = out.len();
        let new_off = self.bounded_end(size)?;
        if size > 0 {
            out.copy_from_slice(&self.data[self.cur_off..new_off]);
        }
        self.cur_off = new_off;
        Ok(())
    }

    /// Bounds-check a read of `size` bytes and return its end offset.
    ///
    /// Uses `checked_add`: a hostile/corrupt length prefix can be as large as
    /// `usize::MAX`, and `cur_off + size` would then overflow (panic in debug,
    /// wrap into a *passing* bounds check in release). Overflow is reported as
    /// EOF — a buffer that large cannot exist.
    fn bounded_end(&self, size: usize) -> Result<usize> {
        match self.cur_off.checked_add(size) {
            Some(end) if end <= self.data.len() => Ok(end),
            _ => Err(SerializeError::UnexpectedEof {
                needed: size,
                remaining: self.remaining(),
            }),
        }
    }

    /// Borrow `size` bytes without copying, advancing the cursor.
    fn take(&mut self, size: usize) -> Result<&'a [u8]> {
        let new_off = self.bounded_end(size)?;
        let out = &self.data[self.cur_off..new_off];
        self.cur_off = new_off;
        Ok(out)
    }

    /// C++ `operator>>` / `operator&` / `base(obj)` on the loading side.
    pub fn load<T: GlobalLoad>(&mut self) -> Result<T> {
        T::global_load(self)
    }
}

// ---------------------------------------------------------------------------
// Primitive impls (C++ std::is_arithmetic branch of base())
// ---------------------------------------------------------------------------

/// `size_t` is 8 bytes on every supported target (divergence 4).
const USIZE_WIRE_BYTES: usize = 8;

fn usize_to_wire(v: usize) -> [u8; USIZE_WIRE_BYTES] {
    (v as u64).to_ne_bytes()
}

macro_rules! impl_arithmetic {
    ($($t:ty),* $(,)?) => {$(
        impl GlobalSave for $t {
            #[inline]
            fn global_save(&self, ar: &mut GlobalSerialize) {
                ar.write_binary(&self.to_ne_bytes());
            }
        }

        impl GlobalLoad for $t {
            #[inline]
            fn global_load(ar: &mut GlobalDeserialize<'_>) -> Result<Self> {
                let mut buf = [0u8; std::mem::size_of::<$t>()];
                ar.read_binary(&mut buf)?;
                Ok(<$t>::from_ne_bytes(buf))
            }
        }
    )*};
}

impl_arithmetic!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

/// C++ `bool` is arithmetic and `sizeof(bool) == 1`.
impl GlobalSave for bool {
    #[inline]
    fn global_save(&self, ar: &mut GlobalSerialize) {
        ar.write_binary(&[u8::from(*self)]);
    }
}

impl GlobalLoad for bool {
    #[inline]
    fn global_load(ar: &mut GlobalDeserialize<'_>) -> Result<Self> {
        let mut buf = [0u8; 1];
        ar.read_binary(&mut buf)?;
        // C++ memcpy's the byte into a bool object; any non-zero reads true.
        Ok(buf[0] != 0)
    }
}

/// `usize` maps to C++ `size_t`, pinned to 8 wire bytes (divergence 4).
impl GlobalSave for usize {
    #[inline]
    fn global_save(&self, ar: &mut GlobalSerialize) {
        ar.write_binary(&usize_to_wire(*self));
    }
}

impl GlobalLoad for usize {
    #[inline]
    fn global_load(ar: &mut GlobalDeserialize<'_>) -> Result<Self> {
        let v = u64::global_load(ar)?;
        usize::try_from(v).map_err(|_| SerializeError::WidthOverflow(v))
    }
}

// ---------------------------------------------------------------------------
// Container impls (C++ serialize_common.h)
// ---------------------------------------------------------------------------

/// C++ `save_string` → `save_string_fused`: `[len:8][bytes]`.
impl GlobalSave for str {
    fn global_save(&self, ar: &mut GlobalSerialize) {
        ar.save_string_fused(self.as_bytes());
    }
}

impl GlobalSave for String {
    fn global_save(&self, ar: &mut GlobalSerialize) {
        ar.save_string_fused(self.as_bytes());
    }
}

/// C++ `load_string`: read `size`, resize, read bytes.
///
/// Non-UTF-8 payloads are decoded lossily rather than failing, matching the
/// C++ "always produces a string" behavior (divergence 17).
impl GlobalLoad for String {
    fn global_load(ar: &mut GlobalDeserialize<'_>) -> Result<Self> {
        let size = usize::global_load(ar)?;
        let bytes = ar.take(size)?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }
}

/// C++ `save_vec`: `[len:8]` then, for arithmetic `T`, one `memcpy` of the
/// array; otherwise element-by-element. Both are byte-identical to this
/// per-element loop because arrays are packed (see module docs).
impl<T: GlobalSave> GlobalSave for Vec<T> {
    fn global_save(&self, ar: &mut GlobalSerialize) {
        ar.save(&self.len());
        for item in self {
            ar.save(item);
        }
    }
}

impl<T: GlobalLoad> GlobalLoad for Vec<T> {
    fn global_load(ar: &mut GlobalDeserialize<'_>) -> Result<Self> {
        let size = usize::global_load(ar)?;
        // Deliberately NOT pre-reserving `size`: a corrupt or hostile length
        // would let an 8-byte header request an arbitrary allocation. C++
        // resizes eagerly and is exposed to exactly that; growing on demand is
        // bounded by the bytes actually present.
        let mut out = Vec::new();
        for _ in 0..size {
            out.push(T::global_load(ar)?);
        }
        Ok(out)
    }
}

/// C++ `save_list` — same bytes as `save_vec`'s non-arithmetic branch.
impl<T: GlobalSave> GlobalSave for VecDeque<T> {
    fn global_save(&self, ar: &mut GlobalSerialize) {
        ar.save(&self.len());
        for item in self {
            ar.save(item);
        }
    }
}

impl<T: GlobalLoad> GlobalLoad for VecDeque<T> {
    fn global_load(ar: &mut GlobalDeserialize<'_>) -> Result<Self> {
        let size = usize::global_load(ar)?;
        let mut out = VecDeque::new();
        for _ in 0..size {
            // C++ load_list: emplace_back() then load into back().
            out.push_back(T::global_load(ar)?);
        }
        Ok(out)
    }
}

/// C++ `save_map`: `[len:8]` then `(key, value)` pairs. Iteration order is
/// implementation-defined on both sides (divergence 15).
impl<K: GlobalSave, V: GlobalSave> GlobalSave for HashMap<K, V> {
    fn global_save(&self, ar: &mut GlobalSerialize) {
        ar.save(&self.len());
        for (k, v) in self {
            ar.save(k);
            ar.save(v);
        }
    }
}

impl<K: GlobalLoad + Eq + Hash, V: GlobalLoad> GlobalLoad for HashMap<K, V> {
    fn global_load(ar: &mut GlobalDeserialize<'_>) -> Result<Self> {
        let size = usize::global_load(ar)?;
        let mut out = HashMap::new();
        for _ in 0..size {
            let key = K::global_load(ar)?;
            let val = V::global_load(ar)?;
            // C++ load_map: obj[key] = val — duplicates are last-wins.
            out.insert(key, val);
        }
        Ok(out)
    }
}

impl<T: GlobalSave + ?Sized> GlobalSave for &T {
    fn global_save(&self, ar: &mut GlobalSerialize) {
        (**self).global_save(ar);
    }
}

/// C++ `ctp::bitfield32_t::serialize` forwards to its `nonatomic<u32>`, so a
/// bitfield is exactly its `u32` on the wire.
///
/// The impl lives here rather than in `ctp-types` because the trait is local
/// here and the type is foreign — the orphan rule allows that direction, and
/// it keeps `ctp-types` free of a serialization dependency.
impl GlobalSave for Bitfield32 {
    fn global_save(&self, ar: &mut GlobalSerialize) {
        ar.save(&self.bits());
    }
}

impl GlobalLoad for Bitfield32 {
    fn global_load(ar: &mut GlobalDeserialize<'_>) -> Result<Self> {
        Ok(Bitfield32::new(ar.load()?))
    }
}

// ---------------------------------------------------------------------------
// (The runtime's task archives used to live here.)
// ---------------------------------------------------------------------------
//
// `NetTaskArchive`, `SaveTaskArchive`, `LoadTaskArchive`, `TaskInfo`,
// `MsgType`, `BulkAllocator`, a `Bulk`, the `BULK_*` flags, the socket-recv
// sentinel and copies of `UniqueId`/`TaskId` were all defined in this module.
// None of them are CTP: they are `clio::run` types and `ctp::lbm` types, and
// the C++ keeps this layer clear of both — no CTP header mentions `clio::run`,
// because serialization there is a template and needs no concrete type.
//
// Rust has no such escape: concrete `GlobalSave`/`GlobalLoad` impls need
// concrete types, so the types were declared next to the traits and the
// archives followed them down a layer. They now live in
// `context-runtime/rust/clio-run-archives`, and each type's wire format is
// implemented in its own crate — which the orphan rule requires anyway.
//
// What stays here is the byte engine: `GlobalSerialize`, `GlobalDeserialize`,
// the `GlobalSave`/`GlobalLoad` traits, and impls for primitives, `String`,
// collections and `Bitfield32`.

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // Byte engine
    // -----------------------------------------------------------------

    #[test]
    fn write_binary_empty_writes_nothing() {
        let mut ar = GlobalSerialize::new();
        ar.write_binary(&[]);
        assert_eq!(ar.len(), 0);
        assert!(ar.is_empty());
        assert_eq!(ar.into_bytes(), Vec::<u8>::new());
    }

    #[test]
    fn scalars_are_native_endian_and_exactly_sized() {
        let mut ar = GlobalSerialize::new();
        ar.save(&1u8).save(&2u16).save(&3u32).save(&4u64);
        assert_eq!(ar.len(), 1 + 2 + 4 + 8);
        let bytes = ar.into_bytes();
        let mut expect = Vec::new();
        expect.extend_from_slice(&1u8.to_ne_bytes());
        expect.extend_from_slice(&2u16.to_ne_bytes());
        expect.extend_from_slice(&3u32.to_ne_bytes());
        expect.extend_from_slice(&4u64.to_ne_bytes());
        assert_eq!(bytes, expect);
    }

    #[test]
    fn signed_float_and_bool_round_trip() {
        let mut ar = GlobalSerialize::new();
        ar.save(&-1i8)
            .save(&-2i16)
            .save(&-3i32)
            .save(&-4i64)
            .save(&1.5f32)
            .save(&-2.5f64)
            .save(&true)
            .save(&false);
        let bytes = ar.into_bytes();
        assert_eq!(bytes.len(), 1 + 2 + 4 + 8 + 4 + 8 + 1 + 1);
        let mut de = GlobalDeserialize::new(&bytes);
        assert_eq!(de.load::<i8>().unwrap(), -1);
        assert_eq!(de.load::<i16>().unwrap(), -2);
        assert_eq!(de.load::<i32>().unwrap(), -3);
        assert_eq!(de.load::<i64>().unwrap(), -4);
        assert_eq!(de.load::<f32>().unwrap(), 1.5);
        assert_eq!(de.load::<f64>().unwrap(), -2.5);
        assert!(de.load::<bool>().unwrap());
        assert!(!de.load::<bool>().unwrap());
        assert_eq!(de.remaining(), 0);
    }

    #[test]
    fn bool_nonzero_byte_loads_true() {
        // C++ memcpy's the byte into a bool; any non-zero is true.
        let bytes = [0x7fu8];
        let mut de = GlobalDeserialize::new(&bytes);
        assert!(de.load::<bool>().unwrap());
    }

    #[test]
    fn usize_is_eight_bytes_on_the_wire() {
        let mut ar = GlobalSerialize::new();
        ar.save(&42usize);
        let bytes = ar.into_bytes();
        assert_eq!(bytes.len(), 8);
        assert_eq!(bytes, 42u64.to_ne_bytes());
    }

    #[test]
    fn float_edge_values_round_trip_bitwise() {
        for v in [
            0.0f64,
            -0.0,
            f64::MIN,
            f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            let mut ar = GlobalSerialize::new();
            ar.save(&v);
            let bytes = ar.into_bytes();
            let got: f64 = GlobalDeserialize::new(&bytes).load().unwrap();
            assert_eq!(got.to_bits(), v.to_bits(), "value {v}");
        }
        // NaN: compare by predicate, not value.
        let mut ar = GlobalSerialize::new();
        ar.save(&f64::NAN);
        let bytes = ar.into_bytes();
        let got: f64 = GlobalDeserialize::new(&bytes).load().unwrap();
        assert!(got.is_nan());
    }

    #[test]
    fn integer_boundaries_round_trip() {
        let mut ar = GlobalSerialize::new();
        ar.save(&u64::MAX)
            .save(&i64::MIN)
            .save(&i64::MAX)
            .save(&0u32)
            .save(&u32::MAX);
        let bytes = ar.into_bytes();
        let mut de = GlobalDeserialize::new(&bytes);
        assert_eq!(de.load::<u64>().unwrap(), u64::MAX);
        assert_eq!(de.load::<i64>().unwrap(), i64::MIN);
        assert_eq!(de.load::<i64>().unwrap(), i64::MAX);
        assert_eq!(de.load::<u32>().unwrap(), 0);
        assert_eq!(de.load::<u32>().unwrap(), u32::MAX);
    }

    // -----------------------------------------------------------------
    // Growth policy (C++ capacity doubling + Finalize)
    // -----------------------------------------------------------------

    #[test]
    fn growth_follows_cpp_doubling_schedule() {
        // C++: new_cap starts at 64 when empty, then doubles.
        let mut ar = GlobalSerialize::new();
        ar.write_binary(&[0u8; 1]);
        assert_eq!(ar.data.len(), 64, "0 -> 64");
        ar.write_binary(&[0u8; 63]);
        assert_eq!(ar.data.len(), 64, "exactly fills 64, no growth");
        ar.write_binary(&[0u8; 1]);
        assert_eq!(ar.data.len(), 128, "64 -> 128");
        ar.write_binary(&[0u8; 200]);
        assert_eq!(ar.data.len(), 512, "128 -> 256 -> 512");
        assert_eq!(ar.len(), 265);
    }

    #[test]
    fn finalize_truncates_the_growth_tail() {
        let mut ar = GlobalSerialize::new();
        ar.save(&7u32);
        // Buffer over-allocated to 64, cursor at 4.
        assert_eq!(ar.data.len(), 64);
        assert_eq!(ar.len(), 4);
        ar.finalize();
        assert_eq!(ar.data.len(), 4);
        assert_eq!(ar.into_bytes(), 7u32.to_ne_bytes().to_vec());
    }

    #[test]
    fn finalize_is_idempotent_and_writes_may_continue() {
        let mut ar = GlobalSerialize::new();
        ar.save(&1u32);
        ar.finalize();
        ar.finalize();
        ar.save(&2u32);
        let bytes = ar.into_bytes();
        let mut expect = Vec::new();
        expect.extend_from_slice(&1u32.to_ne_bytes());
        expect.extend_from_slice(&2u32.to_ne_bytes());
        assert_eq!(bytes, expect);
    }

    #[test]
    fn from_buffer_appends_at_the_end() {
        let mut first = GlobalSerialize::new();
        first.save(&1u32);
        let finalized = first.into_bytes();
        assert_eq!(finalized.len(), 4);

        let mut second = GlobalSerialize::from_buffer(finalized);
        assert_eq!(second.len(), 4);
        second.save(&2u32);
        let bytes = second.into_bytes();
        let mut expect = Vec::new();
        expect.extend_from_slice(&1u32.to_ne_bytes());
        expect.extend_from_slice(&2u32.to_ne_bytes());
        assert_eq!(bytes, expect);
    }

    #[test]
    fn with_capacity_does_not_affect_length_or_bytes() {
        let mut ar = GlobalSerialize::with_capacity(256);
        assert_eq!(ar.len(), 0);
        ar.save(&5u8);
        assert_eq!(ar.into_bytes(), vec![5u8]);
    }

    // -----------------------------------------------------------------
    // Read overflow (divergence 3)
    // -----------------------------------------------------------------

    #[test]
    fn read_past_end_errors_and_does_not_advance() {
        let bytes = [1u8, 2, 3];
        let mut de = GlobalDeserialize::new(&bytes);
        let err = de.load::<u32>().unwrap_err();
        assert_eq!(
            err,
            SerializeError::UnexpectedEof {
                needed: 4,
                remaining: 3
            }
        );
        // Cursor unmoved: the partial read consumed nothing.
        assert_eq!(de.cur_off(), 0);
        assert_eq!(de.remaining(), 3);
    }

    #[test]
    fn read_from_empty_buffer_errors() {
        let mut de = GlobalDeserialize::new(&[]);
        assert!(de.load::<u8>().is_err());
        assert_eq!(de.remaining(), 0);
        // Zero-length read on an empty buffer is fine.
        assert!(de.read_binary(&mut []).is_ok());
    }

    #[test]
    fn truncated_string_payload_errors() {
        let mut ar = GlobalSerialize::new();
        ar.save(&String::from("hello"));
        let mut bytes = ar.into_bytes();
        bytes.truncate(bytes.len() - 1); // chop the last char
        let err = GlobalDeserialize::new(&bytes).load::<String>().unwrap_err();
        assert_eq!(
            err,
            SerializeError::UnexpectedEof {
                needed: 5,
                remaining: 4
            }
        );
    }

    #[test]
    fn hostile_length_prefix_errors_rather_than_allocating() {
        // len = u64::MAX with no payload; must not try to reserve it.
        let bytes = u64::MAX.to_ne_bytes();
        assert!(GlobalDeserialize::new(&bytes).load::<Vec<u32>>().is_err());
        assert!(GlobalDeserialize::new(&bytes).load::<String>().is_err());
        assert!(GlobalDeserialize::new(&bytes).load::<Vec<u8>>().is_err());
    }

    // -----------------------------------------------------------------
    // Strings & containers
    // -----------------------------------------------------------------

    #[test]
    fn string_wire_is_len_prefix_plus_bytes() {
        let mut ar = GlobalSerialize::new();
        ar.save(&String::from("abc"));
        let bytes = ar.into_bytes();
        let mut expect = Vec::new();
        expect.extend_from_slice(&3u64.to_ne_bytes());
        expect.extend_from_slice(b"abc");
        assert_eq!(bytes, expect);
        assert_eq!(
            GlobalDeserialize::new(&bytes).load::<String>().unwrap(),
            "abc"
        );
    }

    #[test]
    fn empty_string_is_just_a_zero_length() {
        let mut ar = GlobalSerialize::new();
        ar.save(&String::new());
        let bytes = ar.into_bytes();
        assert_eq!(bytes, 0u64.to_ne_bytes().to_vec());
        assert_eq!(GlobalDeserialize::new(&bytes).load::<String>().unwrap(), "");
    }

    #[test]
    fn str_and_string_produce_identical_bytes() {
        let mut a = GlobalSerialize::new();
        a.save("xyz");
        let mut b = GlobalSerialize::new();
        b.save(&String::from("xyz"));
        assert_eq!(a.into_bytes(), b.into_bytes());
    }

    #[test]
    fn string_and_vec_u8_share_a_wire_format() {
        // C++ save_string_fused and save_vec<char> both emit [len][bytes].
        let mut a = GlobalSerialize::new();
        a.save(&String::from("hi"));
        let mut b = GlobalSerialize::new();
        b.save(&b"hi".to_vec());
        assert_eq!(a.into_bytes(), b.into_bytes());
    }

    #[test]
    fn vec_of_arithmetic_matches_a_flat_memcpy() {
        // C++ save_vec takes the memcpy branch for arithmetic T; the
        // per-element loop must be byte-identical.
        let v: Vec<u32> = vec![1, 2, 3];
        let mut ar = GlobalSerialize::new();
        ar.save(&v);
        let bytes = ar.into_bytes();
        let mut expect = Vec::new();
        expect.extend_from_slice(&3u64.to_ne_bytes());
        for x in &v {
            expect.extend_from_slice(&x.to_ne_bytes());
        }
        assert_eq!(bytes, expect);
        assert_eq!(GlobalDeserialize::new(&bytes).load::<Vec<u32>>().unwrap(), v);
    }

    #[test]
    fn empty_vec_round_trips() {
        let mut ar = GlobalSerialize::new();
        ar.save(&Vec::<u64>::new());
        let bytes = ar.into_bytes();
        assert_eq!(bytes, 0u64.to_ne_bytes().to_vec());
        assert!(GlobalDeserialize::new(&bytes)
            .load::<Vec<u64>>()
            .unwrap()
            .is_empty());
    }

    #[test]
    fn nested_vec_round_trips() {
        let v: Vec<Vec<u16>> = vec![vec![], vec![1], vec![2, 3]];
        let mut ar = GlobalSerialize::new();
        ar.save(&v);
        let bytes = ar.into_bytes();
        assert_eq!(
            GlobalDeserialize::new(&bytes)
                .load::<Vec<Vec<u16>>>()
                .unwrap(),
            v
        );
    }

    #[test]
    fn vec_of_strings_round_trips() {
        let v = vec![String::from(""), String::from("a"), String::from("bcd")];
        let mut ar = GlobalSerialize::new();
        ar.save(&v);
        let bytes = ar.into_bytes();
        assert_eq!(
            GlobalDeserialize::new(&bytes).load::<Vec<String>>().unwrap(),
            v
        );
    }

    #[test]
    fn vecdeque_list_shares_the_vector_wire_format() {
        let list: VecDeque<u32> = VecDeque::from(vec![7, 8, 9]);
        let mut a = GlobalSerialize::new();
        a.save(&list);
        let mut b = GlobalSerialize::new();
        b.save(&vec![7u32, 8, 9]);
        let b_bytes = b.into_bytes();
        assert_eq!(a.into_bytes(), b_bytes);
        assert_eq!(
            GlobalDeserialize::new(&b_bytes)
                .load::<VecDeque<u32>>()
                .unwrap(),
            list
        );
    }

    #[test]
    fn map_round_trips_and_is_len_prefixed() {
        let mut m = HashMap::new();
        m.insert(1u32, String::from("one"));
        m.insert(2u32, String::from("two"));
        let mut ar = GlobalSerialize::new();
        ar.save(&m);
        let bytes = ar.into_bytes();
        // First 8 bytes are the entry count regardless of iteration order.
        assert_eq!(bytes[..8], 2u64.to_ne_bytes());
        assert_eq!(
            GlobalDeserialize::new(&bytes)
                .load::<HashMap<u32, String>>()
                .unwrap(),
            m
        );
    }

    #[test]
    fn empty_map_round_trips() {
        let m: HashMap<u32, u32> = HashMap::new();
        let mut ar = GlobalSerialize::new();
        ar.save(&m);
        let bytes = ar.into_bytes();
        assert_eq!(bytes, 0u64.to_ne_bytes().to_vec());
        assert!(GlobalDeserialize::new(&bytes)
            .load::<HashMap<u32, u32>>()
            .unwrap()
            .is_empty());
    }

    #[test]
    fn map_duplicate_keys_are_last_wins() {
        // C++ load_map does obj[key] = val.
        let mut ar = GlobalSerialize::new();
        ar.save(&2usize); // count
        ar.save(&5u32).save(&100u32);
        ar.save(&5u32).save(&200u32);
        let bytes = ar.into_bytes();
        let m: HashMap<u32, u32> = GlobalDeserialize::new(&bytes).load().unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(m[&5], 200);
    }

    // -----------------------------------------------------------------
    // Domain types
    // -----------------------------------------------------------------

    #[test]
    fn error_messages_render() {
        let errs = [
            SerializeError::UnexpectedEof {
                needed: 4,
                remaining: 1,
            },
            SerializeError::WidthOverflow(u64::MAX),
            SerializeError::InvalidMsgType(9),
            SerializeError::BulkExhausted { index: 2, len: 1 },
            SerializeError::NoBulkAllocator,
        ];
        for e in errs {
            assert!(!e.to_string().is_empty());
        }
    }
}
