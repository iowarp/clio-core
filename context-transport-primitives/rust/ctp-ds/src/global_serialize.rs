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

use ctp_memory::{AllocatorId, ShmPtr};

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

// ---------------------------------------------------------------------------
// Domain types (clio_runtime/types.h)
// ---------------------------------------------------------------------------

/// C++ `clio::run::UniqueId` — `serialize` emits `major` then `minor`.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UniqueId {
    /// Major component.
    pub major: u32,
    /// Minor component.
    pub minor: u32,
}

impl UniqueId {
    /// C++ `UniqueId(u32, u32)`.
    pub const fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }

    /// C++ `GetNull()` — both components zero.
    pub const fn null() -> Self {
        Self::new(0, 0)
    }

    /// C++ `IsNull()`.
    pub const fn is_null(&self) -> bool {
        self.major == 0 && self.minor == 0
    }

    /// C++ `ToU64()` — `(major << 32) | minor`.
    pub const fn to_u64(&self) -> u64 {
        ((self.major as u64) << 32) | (self.minor as u64)
    }

    /// C++ `FromU64(u64)`.
    pub const fn from_u64(value: u64) -> Self {
        Self::new((value >> 32) as u32, (value & 0xFFFF_FFFF) as u32)
    }
}

/// C++ `using PoolId = UniqueId;`
pub type PoolId = UniqueId;

impl GlobalSave for UniqueId {
    fn global_save(&self, ar: &mut GlobalSerialize) {
        ar.save(&self.major);
        ar.save(&self.minor);
    }
}

impl GlobalLoad for UniqueId {
    fn global_load(ar: &mut GlobalDeserialize<'_>) -> Result<Self> {
        Ok(Self {
            major: ar.load()?,
            minor: ar.load()?,
        })
    }
}

/// C++ `clio::run::TaskId` — `serialize` emits the seven fields in
/// declaration order (32 wire bytes).
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskId {
    /// Process ID.
    pub pid: u32,
    /// Thread ID.
    pub tid: u32,
    /// Major sequence number (monotonically increasing per thread).
    pub major: u32,
    /// Replica identifier (for replicated tasks).
    pub replica_id: u32,
    /// Unique identifier, incremented for root tasks and subtasks alike.
    pub unique: u32,
    /// Node identifier for distributed execution.
    pub node_id: u32,
    /// Network key for send/recv map lookup (pointer-based).
    pub net_key: usize,
}

impl TaskId {
    /// C++ `ToU64()` — XOR-folds the fields for hashing. `net_key` is
    /// excluded and `node_id` is masked to 32 bits, exactly as in C++.
    pub const fn to_u64(&self) -> u64 {
        let hash1 = ((self.pid as u64) << 32) | (self.tid as u64);
        let hash2 = ((self.major as u64) << 32) | (self.replica_id as u64);
        let hash3 = ((self.unique as u64) << 32) | ((self.node_id as u64) & 0xFFFF_FFFF);
        hash1 ^ hash2 ^ hash3
    }
}

impl GlobalSave for TaskId {
    fn global_save(&self, ar: &mut GlobalSerialize) {
        ar.save(&self.pid);
        ar.save(&self.tid);
        ar.save(&self.major);
        ar.save(&self.replica_id);
        ar.save(&self.unique);
        ar.save(&self.node_id);
        ar.save(&self.net_key);
    }
}

impl GlobalLoad for TaskId {
    fn global_load(ar: &mut GlobalDeserialize<'_>) -> Result<Self> {
        Ok(Self {
            pid: ar.load()?,
            tid: ar.load()?,
            major: ar.load()?,
            replica_id: ar.load()?,
            unique: ar.load()?,
            node_id: ar.load()?,
            net_key: ar.load()?,
        })
    }
}

/// C++ `clio::run::TaskInfo` — task metadata carried per serialized task
/// (44 wire bytes).
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct TaskInfo {
    /// Identity of the task.
    pub task_id: TaskId,
    /// Pool the task belongs to.
    pub pool_id: PoolId,
    /// Method being invoked.
    pub method_id: u32,
}

impl GlobalSave for TaskInfo {
    fn global_save(&self, ar: &mut GlobalSerialize) {
        ar.save(&self.task_id);
        ar.save(&self.pool_id);
        ar.save(&self.method_id);
    }
}

impl GlobalLoad for TaskInfo {
    fn global_load(ar: &mut GlobalDeserialize<'_>) -> Result<Self> {
        Ok(Self {
            task_id: ar.load()?,
            pool_id: ar.load()?,
            method_id: ar.load()?,
        })
    }
}

/// C++ `clio::run::MsgType : uint8_t` — the type of message being sent.
#[repr(u8)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MsgType {
    /// Serialize task inputs for remote execution.
    #[default]
    SerializeIn = 0,
    /// Serialize task outputs back to origin.
    SerializeOut = 1,
    /// Heartbeat message (no task data).
    Heartbeat = 2,
}

impl MsgType {
    /// The `uint8_t` underlying value written on the wire.
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// Decode a wire byte. Unlike the C++ `static_cast`, out-of-range values
    /// are rejected (divergence 6).
    pub const fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(Self::SerializeIn),
            1 => Ok(Self::SerializeOut),
            2 => Ok(Self::Heartbeat),
            other => Err(SerializeError::InvalidMsgType(other)),
        }
    }
}

impl GlobalSave for MsgType {
    fn global_save(&self, ar: &mut GlobalSerialize) {
        // C++ base(): write the enum's underlying type (uint8_t).
        ar.save(&self.as_u8());
    }
}

impl GlobalLoad for MsgType {
    fn global_load(ar: &mut GlobalDeserialize<'_>) -> Result<Self> {
        Self::from_u8(u8::global_load(ar)?)
    }
}

// ---------------------------------------------------------------------------
// Bulk
// ---------------------------------------------------------------------------

/// C++ `ctp::lbm::Bulk` — a bulk-transfer descriptor.
///
/// Only `size` and `flags` are serialized (C++ `Bulk::serialize` does
/// `ar(size, flags)`); the data pointer is meaningful only within the process
/// that owns it, and the payload rides the transport separately. See
/// divergence 9 for the dropped `desc`/`mr`/`FullPtr` fields.
#[derive(Debug, Clone, Copy)]
pub struct Bulk {
    /// Shared-memory location of the data (never serialized).
    pub data: ShmPtr<u8>,
    /// Payload size in bytes.
    pub size: u64,
    /// [`BULK_EXPOSE`] and/or [`BULK_XFER`].
    pub flags: u32,
    /// Models C++ `Bulk::desc != nullptr`: the buffer is owned by the
    /// transport (e.g. a libzmq `zmq_msg_t`) and must be copied out of, never
    /// aliased.
    pub has_desc: bool,
}

impl Default for Bulk {
    fn default() -> Self {
        Self {
            data: ShmPtr::null(),
            size: 0,
            flags: 0,
            has_desc: false,
        }
    }
}

impl Bulk {
    /// A descriptor with no attached buffer.
    pub fn new(size: u64, flags: u32) -> Self {
        Self {
            data: ShmPtr::null(),
            size,
            flags,
            has_desc: false,
        }
    }

    /// C++ `flags.Any(BULK_XFER)`.
    pub const fn is_xfer(&self) -> bool {
        (self.flags & BULK_XFER) != 0
    }

    /// C++ `flags.Any(BULK_EXPOSE)`.
    pub const fn is_expose(&self) -> bool {
        (self.flags & BULK_EXPOSE) != 0
    }
}

impl PartialEq for Bulk {
    fn eq(&self, other: &Self) -> bool {
        // ShmPtr has no PartialEq; compare its two public fields.
        self.data.alloc_id == other.data.alloc_id
            && self.data.off == other.data.off
            && self.size == other.size
            && self.flags == other.flags
            && self.has_desc == other.has_desc
    }
}

impl Eq for Bulk {}

impl GlobalSave for Bulk {
    fn global_save(&self, ar: &mut GlobalSerialize) {
        // C++ Bulk::serialize: ar(size, flags). `size` is size_t; `flags` is a
        // bitfield32_t whose serialize forwards to nonatomic<u32> → u32.
        ar.save(&self.size);
        ar.save(&self.flags);
    }
}

impl GlobalLoad for Bulk {
    fn global_load(ar: &mut GlobalDeserialize<'_>) -> Result<Self> {
        Ok(Self {
            data: ShmPtr::null(),
            size: ar.load()?,
            flags: ar.load()?,
            has_desc: false,
        })
    }
}

/// Supplies the buffer-management operations that C++ reaches for through the
/// global `CLIO_IPC` manager (divergence 11).
pub trait BulkAllocator {
    /// C++ `CLIO_IPC->AllocateBuffer(size)`. Returns a null [`ShmPtr`] on
    /// failure, matching the C++ contract.
    fn allocate_buffer(&mut self, size: u64) -> ShmPtr<u8>;

    /// C++ `memcpy(dst.ptr_, src.ptr_, size)` after `ToFullPtr` resolution.
    /// Implementations must tolerate unresolvable pointers by doing nothing
    /// (C++ guards both pointers before copying).
    fn copy_bulk(&mut self, dst: ShmPtr<u8>, src: ShmPtr<u8>, size: u64);
}

// ---------------------------------------------------------------------------
// NetTaskArchive
// ---------------------------------------------------------------------------

/// C++ `clio::run::NetTaskArchive` — the metadata common to both network
/// archives, plus the `LbmMeta` bulk vectors it inherits.
///
/// C++ uses inheritance (`NetTaskArchive : public ctp::lbm::LbmMeta<>`, then
/// `SaveTaskArchive : public NetTaskArchive`); Rust composes this as the
/// `meta` field of each archive.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct NetTaskArchive {
    /// `LbmMeta::send` — sender's bulk descriptors.
    pub send: Vec<Bulk>,
    /// `LbmMeta::recv` — receiver's bulk descriptors (copy of `send` with
    /// local pointers).
    pub recv: Vec<Bulk>,
    /// `LbmMeta::send_bulks` — count of [`BULK_XFER`] entries in `send`.
    pub send_bulks: usize,
    /// `LbmMeta::recv_bulks` — count of [`BULK_XFER`] entries in `recv`.
    pub recv_bulks: usize,
    /// Task metadata for each serialized task.
    pub task_infos: Vec<TaskInfo>,
    /// Message type: `SerializeIn`, `SerializeOut` or `Heartbeat`.
    pub msg_type: MsgType,
    /// Ephemeral port on the requesting client where responses should be
    /// returned. Set on the `SerializeIn` (request) path; the receiver pairs
    /// it with the sender's transport identity to open/cache a dedicated
    /// dial-back connection for the `SerializeOut` (response).
    /// `0` = unset (legacy path).
    pub client_port: i32,
}

impl NetTaskArchive {
    /// C++ `NetTaskArchive(MsgType)`.
    pub fn new(msg_type: MsgType) -> Self {
        Self {
            msg_type,
            ..Default::default()
        }
    }

    /// C++ `GetSendBulkCount()` — note this returns `send.size()`, *not*
    /// `send_bulks` (which counts only [`BULK_XFER`] entries).
    pub fn get_send_bulk_count(&self) -> usize {
        self.send.len()
    }

    /// C++ `GetRecvBulkCount()` — returns `recv.size()`, not `recv_bulks`.
    pub fn get_recv_bulk_count(&self) -> usize {
        self.recv.len()
    }

    /// C++ `GetTaskInfos()`.
    pub fn get_task_infos(&self) -> &[TaskInfo] {
        &self.task_infos
    }

    /// C++ `GetMsgType()`.
    pub fn get_msg_type(&self) -> MsgType {
        self.msg_type
    }

    /// The shared prefix of both archives' `serialize`:
    /// `send, recv, send_bulks, recv_bulks, task_infos, msg_type, client_port`.
    fn save_header(&self, ar: &mut GlobalSerialize) {
        ar.save(&self.send);
        ar.save(&self.recv);
        ar.save(&self.send_bulks);
        ar.save(&self.recv_bulks);
        ar.save(&self.task_infos);
        ar.save(&self.msg_type);
        ar.save(&self.client_port);
    }

    fn load_header(ar: &mut GlobalDeserialize<'_>) -> Result<Self> {
        Ok(Self {
            send: ar.load()?,
            recv: ar.load()?,
            send_bulks: ar.load()?,
            recv_bulks: ar.load()?,
            task_infos: ar.load()?,
            msg_type: ar.load()?,
            client_port: ar.load()?,
        })
    }
}

// ---------------------------------------------------------------------------
// SaveTaskArchive
// ---------------------------------------------------------------------------

/// C++ `clio::run::SaveTaskArchive` — saves task inputs or outputs for
/// network transfer. One archive handles both `SerializeIn` and
/// `SerializeOut` (selected by [`NetTaskArchive::msg_type`]).
#[derive(Debug, Default)]
pub struct SaveTaskArchive {
    /// The `NetTaskArchive`/`LbmMeta` state (C++ base classes).
    pub meta: NetTaskArchive,
    ser: GlobalSerialize,
    is_pod: bool,
}

impl SaveTaskArchive {
    /// C++ `using is_saving = std::true_type;`
    pub const IS_SAVING: bool = true;
    /// C++ `using is_loading = std::false_type;`
    pub const IS_LOADING: bool = false;
    /// C++ `using supports_range_ops = std::true_type;`
    pub const SUPPORTS_RANGE_OPS: bool = true;

    /// C++ `SaveTaskArchive(MsgType, Transport* = nullptr)`.
    /// The transport parameter is not ported (divergence 10).
    pub fn new(msg_type: MsgType) -> Self {
        Self {
            meta: NetTaskArchive::new(msg_type),
            // C++ ctor: buffer_.reserve(256).
            ser: GlobalSerialize::with_capacity(256),
            is_pod: false,
        }
    }

    /// C++ `PushPod(bool val)` — `is_pod_ = val`.
    pub fn push_pod(&mut self, val: bool) {
        self.is_pod = val;
    }

    /// C++ `PopPod()` — `is_pod_ = false`.
    ///
    /// Despite the name this is **not** a stack pop: it clears the flag
    /// unconditionally rather than restoring a saved value, so
    /// `push_pod(true); push_pod(true); pop_pod();` leaves POD mode off.
    /// Reproduced verbatim (divergence 13).
    pub fn pop_pod(&mut self) {
        self.is_pod = false;
    }

    /// Current POD mode (C++ `is_pod_`). See divergence 14: this flag never
    /// changes the byte layout; it tells the runtime layer to skip
    /// `Task`-derived dispatch.
    pub fn is_pod(&self) -> bool {
        self.is_pod
    }

    /// C++ `operator<<` for non-`Task` values. `Task`-derived values are
    /// dispatched by the runtime layer (divergence 14).
    pub fn save<T: GlobalSave + ?Sized>(&mut self, value: &T) -> &mut Self {
        self.ser.save(value);
        self
    }

    /// C++ `operator<<` for a `Task`: records its [`TaskInfo`], leaving the
    /// `SerializeIn`/`SerializeOut` body to the runtime layer.
    pub fn push_task_info(&mut self, info: TaskInfo) {
        self.meta.task_infos.push(info);
    }

    /// C++ `write_binary(const char*, size_t)`.
    pub fn write_binary(&mut self, data: &[u8]) {
        self.ser.write_binary(data);
    }

    /// C++ `SaveTaskArchive::bulk(ShmPtr<> ptr, size_t size, uint32_t flags)`.
    ///
    /// Pushes the descriptor onto `send` and, when [`BULK_XFER`] is set,
    /// increments `send_bulks` (which drives ZMQ `ZMQ_SNDMORE` framing). The
    /// `ToFullPtr`/`Expose` RDMA-registration step is the caller's
    /// responsibility here (divergence 10).
    pub fn bulk(&mut self, ptr: ShmPtr<u8>, size: u64, flags: u32) {
        let bulk = Bulk {
            data: ptr,
            size,
            flags,
            has_desc: false,
        };
        self.meta.send.push(bulk);
        // C++: track count of BULK_XFER entries for ZMQ_SNDMORE handling.
        if flags & BULK_XFER != 0 {
            self.meta.send_bulks += 1;
        }
    }

    /// C++ `GetBuffer()` — finalize and borrow the payload.
    pub fn get_buffer(&mut self) -> &[u8] {
        self.ser.get_buffer()
    }

    /// C++ `GetData()` — finalize and copy the payload out.
    /// Returns `Vec<u8>` rather than `std::string` (divergence 17).
    pub fn get_data(&mut self) -> Vec<u8> {
        self.ser.get_buffer().to_vec()
    }

    /// C++ `SaveTaskArchive::serialize(Ar &ar)` — emit the whole archive
    /// (header + finalized payload) in wire order.
    ///
    /// A method rather than a [`GlobalSave`] impl because `Finalize()` needs
    /// `&mut self` (divergences 2 and 16).
    pub fn save_archive(&mut self, ar: &mut GlobalSerialize) {
        self.meta.save_header(ar);
        // C++: serializer_.Finalize(); ar(buffer_);
        // save_string_fused is exactly save_vec's arithmetic branch: [len][bytes].
        let payload = self.ser.get_buffer().to_vec();
        ar.save_string_fused(&payload);
    }

    /// Convenience: [`save_archive`] into a fresh buffer.
    ///
    /// [`save_archive`]: SaveTaskArchive::save_archive
    pub fn serialize_archive(&mut self) -> Vec<u8> {
        let mut ar = GlobalSerialize::new();
        self.save_archive(&mut ar);
        ar.into_bytes()
    }
}

// ---------------------------------------------------------------------------
// LoadTaskArchive
// ---------------------------------------------------------------------------

/// C++ `clio::run::LoadTaskArchive` — loads task inputs or outputs from
/// network transfer.
///
/// Unlike C++, this holds an owned buffer plus an explicit cursor instead of a
/// self-referential `GlobalDeserialize` (divergence 8).
#[derive(Debug, Default)]
pub struct LoadTaskArchive {
    /// The `NetTaskArchive`/`LbmMeta` state (C++ base classes).
    pub meta: NetTaskArchive,
    data: Vec<u8>,
    cur_off: usize,
    current_task_index: usize,
    current_bulk_index: usize,
    is_pod: bool,
    /// C++ `daemon_allocated_bulk_count_`.
    ///
    /// Number of bulk buffers that [`bulk`] allocated locally
    /// ([`BULK_EXPOSE`] on receive). The receiver owns these and must free
    /// them after the task completes — otherwise cross-node `GetBlob`
    /// responses leak the output buffer per call and the daemon's SHM segment
    /// fills up after a few thousand cross-node reads. `RecvIn` promotes this
    /// to a `TASK_DATA_OWNER` flag on the task so the task's destructor frees
    /// the buffer; `SendOut` skips clearing the flag for tasks that actually
    /// have owned buffers.
    ///
    /// [`bulk`]: LoadTaskArchive::bulk
    pub daemon_allocated_bulk_count: usize,
}

impl LoadTaskArchive {
    /// C++ `using is_saving = std::false_type;`
    pub const IS_SAVING: bool = false;
    /// C++ `using is_loading = std::true_type;`
    pub const IS_LOADING: bool = true;
    /// C++ `using supports_range_ops = std::true_type;`
    pub const SUPPORTS_RANGE_OPS: bool = true;

    /// C++ `LoadTaskArchive()` — empty buffer, `msg_type_ = kSerializeIn`.
    pub fn new() -> Self {
        Self::default()
    }

    /// C++ `LoadTaskArchive(const char *data, size_t size)` /
    /// `LoadTaskArchive(const std::string&)`.
    pub fn from_bytes(data: &[u8]) -> Self {
        Self::from_vec(data.to_vec())
    }

    /// C++ `LoadTaskArchive(std::vector<char> &&data)`.
    pub fn from_vec(data: Vec<u8>) -> Self {
        Self {
            meta: NetTaskArchive::new(MsgType::SerializeIn),
            data,
            cur_off: 0,
            current_task_index: 0,
            current_bulk_index: 0,
            is_pod: false,
            daemon_allocated_bulk_count: 0,
        }
    }

    /// C++ `PushPod(bool val)`.
    pub fn push_pod(&mut self, val: bool) {
        self.is_pod = val;
    }

    /// C++ `PopPod()` — clears the flag unconditionally (divergence 13).
    pub fn pop_pod(&mut self) {
        self.is_pod = false;
    }

    /// Current POD mode (C++ `is_pod_`).
    pub fn is_pod(&self) -> bool {
        self.is_pod
    }

    /// The undecoded payload (C++ `data_`).
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Current read cursor into the payload.
    pub fn cur_off(&self) -> usize {
        self.cur_off
    }

    /// C++ `operator>>` for non-`Task` values.
    pub fn load<T: GlobalLoad>(&mut self) -> Result<T> {
        // Disjoint field borrows: `de` borrows `self.data`; the cursor
        // write-back touches only `self.cur_off`.
        let mut de = GlobalDeserialize::at(&self.data, self.cur_off);
        let out = T::global_load(&mut de);
        self.cur_off = de.cur_off();
        out
    }

    /// C++ `read_binary(char*, size_t)`.
    pub fn read_binary(&mut self, out: &mut [u8]) -> Result<()> {
        let mut de = GlobalDeserialize::at(&self.data, self.cur_off);
        let r = de.read_binary(out);
        self.cur_off = de.cur_off();
        r
    }

    /// C++ `GetCurrentTaskInfo()`.
    ///
    /// C++ indexes `task_infos_[current_task_index_]` unchecked (UB past the
    /// end); Rust returns `None`.
    pub fn get_current_task_info(&self) -> Option<&TaskInfo> {
        self.meta.task_infos.get(self.current_task_index)
    }

    /// C++ `current_task_index_++` — performed by `operator>>(T*&)` after a
    /// `Task` pointer is deserialized. `Task` lives in context-runtime, so the
    /// advance is exposed for the runtime layer to drive.
    pub fn advance_task_index(&mut self) {
        self.current_task_index += 1;
    }

    /// C++ `ResetTaskIndex()`.
    pub fn reset_task_index(&mut self) {
        self.current_task_index = 0;
    }

    /// C++ `ResetBulkIndex()`.
    pub fn reset_bulk_index(&mut self) {
        self.current_bulk_index = 0;
    }

    /// Current bulk cursor (C++ `current_bulk_index_`).
    pub fn current_bulk_index(&self) -> usize {
        self.current_bulk_index
    }

    /// Current task cursor (C++ `current_task_index_`).
    pub fn current_task_index(&self) -> usize {
        self.current_task_index
    }

    /// C++ `LoadTaskArchive::bulk(ShmPtr<> &ptr, size_t size, uint32_t flags)`.
    ///
    /// Resolves the next `recv` descriptor into `ptr`, following the C++
    /// branch structure exactly:
    ///
    /// **`SerializeIn`** (inbound request — the task has no valid pointer yet):
    /// - `recv[i].data` non-null and `has_desc` — the payload lives in a
    ///   transport-owned buffer (libzmq `zmq_msg_t`). Copy into an owned buffer
    ///   so the `TASK_DATA_OWNER` path can free it; aliasing would leak the
    ///   `zmq_msg_t`, which `FreeBuffer` cannot reclaim.
    /// - `recv[i].data` non-null with [`SOCKET_RECV_SENTINEL`] — the buffer was
    ///   `malloc`'d by `SocketTransport::RecvBulks` and `ClearRecvHandles`
    ///   frees it immediately after the load; copy out or the task is left with
    ///   a dangling pointer (heap-use-after-free).
    /// - `recv[i].data` non-null otherwise — SHM transport, data is already in
    ///   shared memory: alias it (zero-copy).
    /// - `recv[i].data` null — [`BULK_EXPOSE`] with no data sent; allocate a
    ///   buffer for the receiver to fill, and record it in `recv[i]`.
    ///
    /// **`SerializeOut`** (inbound response): if the descriptor carries
    /// [`BULK_XFER`], copy into the caller's buffer when it supplied one
    /// (keeping the caller's pointer valid, matching SHM behavior), else alias
    /// the receive buffer.
    ///
    /// Every path that allocates bumps [`daemon_allocated_bulk_count`].
    ///
    /// [`daemon_allocated_bulk_count`]: LoadTaskArchive::daemon_allocated_bulk_count
    pub fn bulk(
        &mut self,
        ptr: &mut ShmPtr<u8>,
        size: u64,
        _flags: u32,
        alloc: Option<&mut dyn BulkAllocator>,
    ) -> Result<()> {
        match self.meta.msg_type {
            MsgType::SerializeIn => self.bulk_serialize_in(ptr, size, alloc),
            MsgType::SerializeOut => self.bulk_serialize_out(ptr, alloc),
            // C++ has no branch for kHeartbeat: neither `if` matches, so the
            // call is a no-op leaving `ptr` untouched.
            MsgType::Heartbeat => Ok(()),
        }
    }

    fn bulk_serialize_in(
        &mut self,
        ptr: &mut ShmPtr<u8>,
        size: u64,
        alloc: Option<&mut dyn BulkAllocator>,
    ) -> Result<()> {
        let idx = self.current_bulk_index;
        let len = self.meta.recv.len();
        if idx >= len {
            // C++: HLOG(kError) + ptr = GetNull(). We do both, and report.
            *ptr = ShmPtr::null();
            return Err(SerializeError::BulkExhausted { index: idx, len });
        }

        let entry = self.meta.recv[idx];
        if !entry.data.is_null() {
            if entry.has_desc {
                // ZMQ zero-copy recv: copy into an owned buffer so the
                // TASK_DATA_OWNER destructor path can reclaim it, then leave
                // recv[i].data pointing at the owned buffer.
                let alloc = alloc.ok_or(SerializeError::NoBulkAllocator)?;
                let buf = alloc.allocate_buffer(size);
                alloc.copy_bulk(buf, entry.data, size);
                *ptr = buf;
                self.meta.recv[idx].data = buf;
                self.daemon_allocated_bulk_count += 1;
            } else if entry.data.alloc_id == SOCKET_RECV_SENTINEL {
                // SocketTransport recv: mirror the ZMQ path, but leave
                // recv[i].data alone so ClearRecvHandles still frees the
                // malloc'd buffer.
                let alloc = alloc.ok_or(SerializeError::NoBulkAllocator)?;
                let buf = alloc.allocate_buffer(size);
                alloc.copy_bulk(buf, entry.data, size);
                *ptr = buf;
                self.daemon_allocated_bulk_count += 1;
            } else {
                // Valid ShmPtr, no transport handle: SHM transport — data is
                // already in shared memory, keep zero-copy.
                *ptr = entry.data;
            }
        } else {
            // Null ShmPtr: BULK_EXPOSE via ZMQ/socket where no data was sent.
            // Allocate a buffer for the receiver to fill (e.g. ReadTask).
            let alloc = alloc.ok_or(SerializeError::NoBulkAllocator)?;
            let buf = alloc.allocate_buffer(size);
            *ptr = buf;
            self.meta.recv[idx].data = buf;
            self.daemon_allocated_bulk_count += 1;
        }
        self.current_bulk_index += 1;
        Ok(())
    }

    fn bulk_serialize_out(
        &mut self,
        ptr: &mut ShmPtr<u8>,
        alloc: Option<&mut dyn BulkAllocator>,
    ) -> Result<()> {
        let idx = self.current_bulk_index;
        // C++ silently does nothing when the recv vector is exhausted on the
        // SerializeOut path (no else branch, not even a log).
        if idx >= self.meta.recv.len() {
            return Ok(());
        }
        let entry = self.meta.recv[idx];
        if entry.is_xfer() {
            if !ptr.is_null() {
                // Caller supplied a buffer: copy the received data into it so
                // the caller's pointer stays valid (matches SHM behavior).
                let alloc = alloc.ok_or(SerializeError::NoBulkAllocator)?;
                alloc.copy_bulk(*ptr, entry.data, entry.size);
            } else {
                // No original buffer — zero-copy, point at the recv buffer.
                *ptr = entry.data;
            }
        }
        self.current_bulk_index += 1;
        Ok(())
    }

    /// C++ `LoadTaskArchive::serialize(Ar &ar)` — decode a whole archive
    /// (header + payload) produced by [`SaveTaskArchive::save_archive`].
    ///
    /// C++ placement-`new`s the deserializer over the refilled `data_`; here
    /// the cursor simply starts at 0 (divergence 8).
    pub fn load_archive(ar: &mut GlobalDeserialize<'_>) -> Result<Self> {
        let meta = NetTaskArchive::load_header(ar)?;
        let data: Vec<u8> = ar.load()?;
        Ok(Self {
            meta,
            data,
            cur_off: 0,
            current_task_index: 0,
            current_bulk_index: 0,
            is_pod: false,
            daemon_allocated_bulk_count: 0,
        })
    }

    /// Convenience: [`load_archive`] from a byte slice.
    ///
    /// [`load_archive`]: LoadTaskArchive::load_archive
    pub fn deserialize_archive(bytes: &[u8]) -> Result<Self> {
        let mut ar = GlobalDeserialize::new(bytes);
        Self::load_archive(&mut ar)
    }
}

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
    fn unique_id_is_eight_bytes_major_then_minor() {
        let id = UniqueId::new(0xAAAA_AAAA, 0xBBBB_BBBB);
        let mut ar = GlobalSerialize::new();
        ar.save(&id);
        let bytes = ar.into_bytes();
        assert_eq!(bytes.len(), 8);
        assert_eq!(bytes[..4], 0xAAAA_AAAAu32.to_ne_bytes());
        assert_eq!(bytes[4..], 0xBBBB_BBBBu32.to_ne_bytes());
        assert_eq!(
            GlobalDeserialize::new(&bytes).load::<UniqueId>().unwrap(),
            id
        );
    }

    #[test]
    fn unique_id_u64_helpers_match_cpp() {
        let id = UniqueId::new(7, 9);
        assert_eq!(id.to_u64(), (7u64 << 32) | 9);
        assert_eq!(UniqueId::from_u64(id.to_u64()), id);
        assert!(UniqueId::null().is_null());
        assert!(!id.is_null());
    }

    #[test]
    fn task_id_is_thirty_two_bytes_in_declaration_order() {
        let id = TaskId {
            pid: 1,
            tid: 2,
            major: 3,
            replica_id: 4,
            unique: 5,
            node_id: 6,
            net_key: 7,
        };
        let mut ar = GlobalSerialize::new();
        ar.save(&id);
        let bytes = ar.into_bytes();
        assert_eq!(bytes.len(), 32, "6 u32 + 1 size_t");
        let mut expect = Vec::new();
        for f in [1u32, 2, 3, 4, 5, 6] {
            expect.extend_from_slice(&f.to_ne_bytes());
        }
        expect.extend_from_slice(&7u64.to_ne_bytes());
        assert_eq!(bytes, expect);
        assert_eq!(GlobalDeserialize::new(&bytes).load::<TaskId>().unwrap(), id);
    }

    #[test]
    fn task_id_to_u64_matches_cpp_fold() {
        let id = TaskId {
            pid: 1,
            tid: 2,
            major: 3,
            replica_id: 4,
            unique: 5,
            node_id: 6,
            net_key: 0xDEAD,
        };
        // net_key is deliberately excluded from the C++ fold.
        let expect = (((1u64 << 32) | 2) ^ ((3u64 << 32) | 4)) ^ ((5u64 << 32) | 6);
        assert_eq!(id.to_u64(), expect);
    }

    #[test]
    fn task_info_is_forty_four_bytes() {
        let info = TaskInfo {
            task_id: TaskId {
                pid: 11,
                ..Default::default()
            },
            pool_id: PoolId::new(200, 0),
            method_id: 3,
        };
        let mut ar = GlobalSerialize::new();
        ar.save(&info);
        let bytes = ar.into_bytes();
        assert_eq!(bytes.len(), 32 + 8 + 4);
        assert_eq!(
            GlobalDeserialize::new(&bytes).load::<TaskInfo>().unwrap(),
            info
        );
    }

    #[test]
    fn msg_type_is_one_byte_and_validated() {
        for (m, b) in [
            (MsgType::SerializeIn, 0u8),
            (MsgType::SerializeOut, 1),
            (MsgType::Heartbeat, 2),
        ] {
            let mut ar = GlobalSerialize::new();
            ar.save(&m);
            let bytes = ar.into_bytes();
            assert_eq!(bytes, vec![b]);
            assert_eq!(GlobalDeserialize::new(&bytes).load::<MsgType>().unwrap(), m);
        }
        // Divergence 6: C++ static_casts blindly; we reject.
        let err = GlobalDeserialize::new(&[3u8]).load::<MsgType>().unwrap_err();
        assert_eq!(err, SerializeError::InvalidMsgType(3));
        assert_eq!(MsgType::default(), MsgType::SerializeIn);
    }

    // -----------------------------------------------------------------
    // Bulk
    // -----------------------------------------------------------------

    #[test]
    fn bulk_serializes_size_and_flags_only() {
        let b = Bulk {
            data: ShmPtr::new(AllocatorId::new(9, 9), 4096),
            size: 1024,
            flags: BULK_XFER,
            has_desc: true,
        };
        let mut ar = GlobalSerialize::new();
        ar.save(&b);
        let bytes = ar.into_bytes();
        assert_eq!(bytes.len(), 12, "size_t + u32, no pointer");
        assert_eq!(bytes[..8], 1024u64.to_ne_bytes());
        assert_eq!(bytes[8..], BULK_XFER.to_ne_bytes());

        // Loading drops the (process-local) pointer and desc flag.
        let got: Bulk = GlobalDeserialize::new(&bytes).load().unwrap();
        assert_eq!(got.size, 1024);
        assert_eq!(got.flags, BULK_XFER);
        assert!(got.data.is_null());
        assert!(!got.has_desc);
    }

    #[test]
    fn bulk_flag_predicates() {
        assert_eq!(BULK_EXPOSE, 1);
        assert_eq!(BULK_XFER, 2);
        let b = Bulk::new(8, BULK_EXPOSE | BULK_XFER);
        assert!(b.is_expose());
        assert!(b.is_xfer());
        let e = Bulk::new(8, BULK_EXPOSE);
        assert!(e.is_expose());
        assert!(!e.is_xfer());
        assert!(!Bulk::default().is_xfer());
    }

    // -----------------------------------------------------------------
    // PushPod / PopPod
    // -----------------------------------------------------------------

    #[test]
    fn push_pod_pop_pod_is_not_a_stack() {
        // Divergence 13: PopPod() unconditionally clears the flag.
        let mut ar = SaveTaskArchive::new(MsgType::SerializeIn);
        assert!(!ar.is_pod());
        ar.push_pod(true);
        assert!(ar.is_pod());
        ar.push_pod(true); // "nested"
        ar.pop_pod();
        assert!(!ar.is_pod(), "pop clears rather than restoring");
        ar.push_pod(false);
        assert!(!ar.is_pod());

        let mut lo = LoadTaskArchive::new();
        assert!(!lo.is_pod());
        lo.push_pod(true);
        assert!(lo.is_pod());
        lo.pop_pod();
        assert!(!lo.is_pod());
    }

    #[test]
    fn pod_mode_does_not_change_the_bytes() {
        // Divergence 14: is_pod only gates Task dispatch, never layout.
        let mut a = SaveTaskArchive::new(MsgType::SerializeIn);
        a.save(&1u32);
        a.save(&2u64);
        let plain = a.get_data();

        let mut b = SaveTaskArchive::new(MsgType::SerializeIn);
        b.push_pod(true);
        b.save(&1u32);
        b.save(&2u64);
        assert_eq!(b.get_data(), plain);
    }

    // -----------------------------------------------------------------
    // SaveTaskArchive
    // -----------------------------------------------------------------

    #[test]
    fn save_archive_bulk_counts_only_xfer_entries() {
        let mut ar = SaveTaskArchive::new(MsgType::SerializeIn);
        let p = ShmPtr::new(AllocatorId::new(1, 0), 64);
        ar.bulk(p, 100, BULK_EXPOSE);
        ar.bulk(p, 200, BULK_XFER);
        ar.bulk(p, 300, BULK_EXPOSE | BULK_XFER);
        // send_bulks counts BULK_XFER only; GetSendBulkCount() is send.size().
        assert_eq!(ar.meta.send_bulks, 2);
        assert_eq!(ar.meta.get_send_bulk_count(), 3);
        assert_eq!(ar.meta.send[0].size, 100);
        assert_eq!(ar.meta.send[2].flags, BULK_EXPOSE | BULK_XFER);
    }

    #[test]
    fn save_archive_payload_matches_a_bare_serializer() {
        let mut ta = SaveTaskArchive::new(MsgType::SerializeOut);
        ta.save(&7u32);
        ta.save(&String::from("payload"));

        let mut bare = GlobalSerialize::new();
        bare.save(&7u32);
        bare.save(&String::from("payload"));

        assert_eq!(ta.get_buffer(), bare.get_buffer());
    }

    #[test]
    fn save_archive_get_data_and_get_buffer_agree_and_are_repeatable() {
        let mut ta = SaveTaskArchive::new(MsgType::SerializeIn);
        ta.save(&123u64);
        let d1 = ta.get_data();
        let d2 = ta.get_data();
        assert_eq!(d1, d2, "GetData is idempotent");
        assert_eq!(ta.get_buffer().to_vec(), d1);
        assert_eq!(d1.len(), 8);
    }

    #[test]
    fn write_binary_bypasses_length_prefixing() {
        let mut ta = SaveTaskArchive::new(MsgType::SerializeIn);
        ta.write_binary(b"raw");
        assert_eq!(ta.get_buffer(), b"raw");
    }

    #[test]
    fn push_task_info_accumulates() {
        let mut ta = SaveTaskArchive::new(MsgType::SerializeIn);
        assert!(ta.meta.get_task_infos().is_empty());
        ta.push_task_info(TaskInfo::default());
        ta.push_task_info(TaskInfo {
            method_id: 9,
            ..Default::default()
        });
        assert_eq!(ta.meta.get_task_infos().len(), 2);
        assert_eq!(ta.meta.task_infos[1].method_id, 9);
    }

    // -----------------------------------------------------------------
    // Whole-archive wire format
    // -----------------------------------------------------------------

    fn sample_save_archive() -> SaveTaskArchive {
        let mut ta = SaveTaskArchive::new(MsgType::SerializeOut);
        ta.meta.client_port = 4242;
        ta.push_task_info(TaskInfo {
            task_id: TaskId {
                pid: 10,
                tid: 20,
                major: 30,
                replica_id: 0,
                unique: 40,
                node_id: 2,
                net_key: 0x1234,
            },
            pool_id: PoolId::new(200, 0),
            method_id: 5,
        });
        ta.bulk(ShmPtr::new(AllocatorId::new(1, 0), 512), 1024, BULK_XFER);
        ta.bulk(ShmPtr::null(), 64, BULK_EXPOSE);
        ta.meta.recv.push(Bulk::new(1024, BULK_XFER));
        ta.meta.recv_bulks = 1;
        ta.save(&0xDEAD_BEEFu32);
        ta.save(&String::from("body"));
        ta
    }

    #[test]
    fn whole_archive_byte_order_is_exact() {
        let mut ta = sample_save_archive();
        let payload = ta.get_data();
        let bytes = ta.serialize_archive();

        let mut expect = Vec::new();
        // send: len + 2 Bulks (12B each)
        expect.extend_from_slice(&2u64.to_ne_bytes());
        expect.extend_from_slice(&1024u64.to_ne_bytes());
        expect.extend_from_slice(&BULK_XFER.to_ne_bytes());
        expect.extend_from_slice(&64u64.to_ne_bytes());
        expect.extend_from_slice(&BULK_EXPOSE.to_ne_bytes());
        // recv: len + 1 Bulk
        expect.extend_from_slice(&1u64.to_ne_bytes());
        expect.extend_from_slice(&1024u64.to_ne_bytes());
        expect.extend_from_slice(&BULK_XFER.to_ne_bytes());
        // send_bulks, recv_bulks
        expect.extend_from_slice(&1u64.to_ne_bytes());
        expect.extend_from_slice(&1u64.to_ne_bytes());
        // task_infos: len + 1 TaskInfo (44B)
        expect.extend_from_slice(&1u64.to_ne_bytes());
        for f in [10u32, 20, 30, 0, 40, 2] {
            expect.extend_from_slice(&f.to_ne_bytes());
        }
        expect.extend_from_slice(&0x1234u64.to_ne_bytes());
        expect.extend_from_slice(&200u32.to_ne_bytes());
        expect.extend_from_slice(&0u32.to_ne_bytes());
        expect.extend_from_slice(&5u32.to_ne_bytes());
        // msg_type (u8), client_port (i32)
        expect.push(1u8);
        expect.extend_from_slice(&4242i32.to_ne_bytes());
        // payload: len + bytes
        expect.extend_from_slice(&(payload.len() as u64).to_ne_bytes());
        expect.extend_from_slice(&payload);

        assert_eq!(bytes, expect);
        assert_eq!(
            bytes.len(),
            8 + 24 + 8 + 12 + 8 + 8 + 8 + 44 + 1 + 4 + 8 + payload.len()
        );
    }

    #[test]
    fn save_archive_round_trips_into_load_archive() {
        let mut ta = sample_save_archive();
        let payload = ta.get_data();
        let bytes = ta.serialize_archive();

        let lo = LoadTaskArchive::deserialize_archive(&bytes).unwrap();
        assert_eq!(lo.meta.msg_type, MsgType::SerializeOut);
        assert_eq!(lo.meta.client_port, 4242);
        assert_eq!(lo.meta.send_bulks, 1);
        assert_eq!(lo.meta.recv_bulks, 1);
        assert_eq!(lo.meta.send.len(), 2);
        assert_eq!(lo.meta.send[0].size, 1024);
        assert_eq!(lo.meta.send[0].flags, BULK_XFER);
        assert_eq!(lo.meta.send[1].size, 64);
        // Pointers do not cross the wire.
        assert!(lo.meta.send[0].data.is_null());
        assert_eq!(lo.meta.recv.len(), 1);
        assert_eq!(lo.meta.task_infos.len(), 1);
        assert_eq!(lo.meta.task_infos[0].task_id.pid, 10);
        assert_eq!(lo.meta.task_infos[0].task_id.net_key, 0x1234);
        assert_eq!(lo.meta.task_infos[0].pool_id, PoolId::new(200, 0));
        assert_eq!(lo.meta.task_infos[0].method_id, 5);
        assert_eq!(lo.data(), payload.as_slice());
    }

    #[test]
    fn loaded_archive_replays_the_payload_fields() {
        let mut ta = sample_save_archive();
        let bytes = ta.serialize_archive();
        let mut lo = LoadTaskArchive::deserialize_archive(&bytes).unwrap();
        assert_eq!(lo.cur_off(), 0);
        assert_eq!(lo.load::<u32>().unwrap(), 0xDEAD_BEEF);
        assert_eq!(lo.load::<String>().unwrap(), "body");
        assert_eq!(lo.cur_off(), lo.data().len());
        // Reading past the payload errors.
        assert!(lo.load::<u8>().is_err());
    }

    #[test]
    fn empty_archive_round_trips() {
        let mut ta = SaveTaskArchive::new(MsgType::Heartbeat);
        let bytes = ta.serialize_archive();
        // 5 empty length prefixes/counts + msg_type + port + payload len
        assert_eq!(bytes.len(), 8 + 8 + 8 + 8 + 8 + 1 + 4 + 8);
        let lo = LoadTaskArchive::deserialize_archive(&bytes).unwrap();
        assert_eq!(lo.meta.msg_type, MsgType::Heartbeat);
        assert!(lo.meta.send.is_empty());
        assert!(lo.meta.recv.is_empty());
        assert!(lo.meta.task_infos.is_empty());
        assert_eq!(lo.meta.client_port, 0);
        assert!(lo.data().is_empty());
    }

    #[test]
    fn negative_client_port_round_trips() {
        let mut ta = SaveTaskArchive::new(MsgType::SerializeIn);
        ta.meta.client_port = -1;
        let bytes = ta.serialize_archive();
        let lo = LoadTaskArchive::deserialize_archive(&bytes).unwrap();
        assert_eq!(lo.meta.client_port, -1);
    }

    #[test]
    fn truncated_archive_errors_cleanly() {
        let mut ta = sample_save_archive();
        let bytes = ta.serialize_archive();
        for cut in [0usize, 1, 8, 20, 40, bytes.len() - 1] {
            assert!(
                LoadTaskArchive::deserialize_archive(&bytes[..cut]).is_err(),
                "truncation to {cut} bytes must error"
            );
        }
        assert!(LoadTaskArchive::deserialize_archive(&bytes).is_ok());
    }

    #[test]
    fn archive_with_invalid_msg_type_byte_is_rejected() {
        let mut ta = SaveTaskArchive::new(MsgType::SerializeIn);
        let mut bytes = ta.serialize_archive();
        // msg_type sits after send(8) + recv(8) + send_bulks(8) +
        // recv_bulks(8) + task_infos(8) = 40 bytes of empty prefixes.
        assert_eq!(bytes[40], 0);
        bytes[40] = 7;
        assert_eq!(
            LoadTaskArchive::deserialize_archive(&bytes).unwrap_err(),
            SerializeError::InvalidMsgType(7)
        );
    }

    // -----------------------------------------------------------------
    // LoadTaskArchive::bulk
    // -----------------------------------------------------------------

    /// Records allocate/copy calls so the ported branch structure is
    /// observable without a real IPC manager.
    #[derive(Default)]
    struct MockAlloc {
        next_off: u64,
        allocs: Vec<u64>,
        copies: Vec<(u64, u64, u64)>,
    }

    impl BulkAllocator for MockAlloc {
        fn allocate_buffer(&mut self, size: u64) -> ShmPtr<u8> {
            self.allocs.push(size);
            self.next_off += 4096;
            ShmPtr::new(AllocatorId::new(7, 7), self.next_off)
        }

        fn copy_bulk(&mut self, dst: ShmPtr<u8>, src: ShmPtr<u8>, size: u64) {
            self.copies.push((dst.off, src.off, size));
        }
    }

    fn load_with_recv(msg_type: MsgType, recv: Vec<Bulk>) -> LoadTaskArchive {
        let mut lo = LoadTaskArchive::new();
        lo.meta.msg_type = msg_type;
        lo.meta.recv = recv;
        lo
    }

    #[test]
    fn bulk_in_shm_path_is_zero_copy() {
        let shm = ShmPtr::new(AllocatorId::new(1, 0), 2048);
        let mut lo = load_with_recv(
            MsgType::SerializeIn,
            vec![Bulk {
                data: shm,
                size: 128,
                flags: BULK_XFER,
                has_desc: false,
            }],
        );
        let mut alloc = MockAlloc::default();
        let mut ptr = ShmPtr::null();
        lo.bulk(&mut ptr, 128, BULK_XFER, Some(&mut alloc)).unwrap();
        assert_eq!(ptr.off, 2048, "aliases the SHM buffer");
        assert!(alloc.allocs.is_empty(), "no allocation on the SHM path");
        assert_eq!(lo.daemon_allocated_bulk_count, 0);
        assert_eq!(lo.current_bulk_index(), 1);
    }

    #[test]
    fn bulk_in_zmq_desc_path_allocates_and_copies() {
        let zmq_buf = ShmPtr::new(AllocatorId::new(3, 3), 900);
        let mut lo = load_with_recv(
            MsgType::SerializeIn,
            vec![Bulk {
                data: zmq_buf,
                size: 256,
                flags: BULK_XFER,
                has_desc: true,
            }],
        );
        let mut alloc = MockAlloc::default();
        let mut ptr = ShmPtr::null();
        lo.bulk(&mut ptr, 256, BULK_XFER, Some(&mut alloc)).unwrap();
        assert_eq!(alloc.allocs, vec![256]);
        assert_eq!(alloc.copies, vec![(4096, 900, 256)]);
        assert_eq!(ptr.off, 4096);
        // C++ re-points recv[i].data at the owned buffer on this path.
        assert_eq!(lo.meta.recv[0].data.off, 4096);
        assert_eq!(lo.daemon_allocated_bulk_count, 1);
    }

    #[test]
    fn bulk_in_socket_sentinel_path_copies_but_keeps_recv_pointer() {
        let malloc_buf = ShmPtr::new(SOCKET_RECV_SENTINEL, 700);
        let mut lo = load_with_recv(
            MsgType::SerializeIn,
            vec![Bulk {
                data: malloc_buf,
                size: 64,
                flags: BULK_XFER,
                has_desc: false,
            }],
        );
        let mut alloc = MockAlloc::default();
        let mut ptr = ShmPtr::null();
        lo.bulk(&mut ptr, 64, BULK_XFER, Some(&mut alloc)).unwrap();
        assert_eq!(alloc.allocs, vec![64]);
        assert_eq!(alloc.copies, vec![(4096, 700, 64)]);
        assert_eq!(ptr.off, 4096);
        // C++ leaves recv[i].data alone so ClearRecvHandles still frees it.
        assert_eq!(lo.meta.recv[0].data.off, 700);
        assert_eq!(lo.meta.recv[0].data.alloc_id, SOCKET_RECV_SENTINEL);
        assert_eq!(lo.daemon_allocated_bulk_count, 1);
    }

    #[test]
    fn bulk_in_null_expose_path_allocates_receive_buffer() {
        let mut lo = load_with_recv(
            MsgType::SerializeIn,
            vec![Bulk {
                data: ShmPtr::null(),
                size: 1024,
                flags: BULK_EXPOSE,
                has_desc: false,
            }],
        );
        let mut alloc = MockAlloc::default();
        let mut ptr = ShmPtr::null();
        lo.bulk(&mut ptr, 1024, BULK_EXPOSE, Some(&mut alloc))
            .unwrap();
        assert_eq!(alloc.allocs, vec![1024]);
        assert!(alloc.copies.is_empty(), "nothing to copy: no data was sent");
        assert_eq!(ptr.off, 4096);
        assert_eq!(lo.meta.recv[0].data.off, 4096);
        assert_eq!(lo.daemon_allocated_bulk_count, 1);
    }

    #[test]
    fn bulk_in_advances_the_index_across_calls() {
        let mut lo = load_with_recv(
            MsgType::SerializeIn,
            vec![
                Bulk {
                    data: ShmPtr::new(AllocatorId::new(1, 0), 100),
                    size: 8,
                    flags: BULK_XFER,
                    has_desc: false,
                },
                Bulk {
                    data: ShmPtr::new(AllocatorId::new(1, 0), 200),
                    size: 8,
                    flags: BULK_XFER,
                    has_desc: false,
                },
            ],
        );
        let mut alloc = MockAlloc::default();
        let mut a = ShmPtr::null();
        let mut b = ShmPtr::null();
        lo.bulk(&mut a, 8, BULK_XFER, Some(&mut alloc)).unwrap();
        lo.bulk(&mut b, 8, BULK_XFER, Some(&mut alloc)).unwrap();
        assert_eq!(a.off, 100);
        assert_eq!(b.off, 200);
        assert_eq!(lo.current_bulk_index(), 2);

        lo.reset_bulk_index();
        assert_eq!(lo.current_bulk_index(), 0);
        let mut c = ShmPtr::null();
        lo.bulk(&mut c, 8, BULK_XFER, Some(&mut alloc)).unwrap();
        assert_eq!(c.off, 100, "reset replays from the start");
    }

    #[test]
    fn bulk_in_exhausted_nulls_pointer_and_errors() {
        let mut lo = load_with_recv(MsgType::SerializeIn, vec![]);
        let mut alloc = MockAlloc::default();
        let mut ptr = ShmPtr::new(AllocatorId::new(5, 5), 99);
        let err = lo
            .bulk(&mut ptr, 16, BULK_XFER, Some(&mut alloc))
            .unwrap_err();
        assert_eq!(err, SerializeError::BulkExhausted { index: 0, len: 0 });
        assert!(ptr.is_null(), "C++ sets the out-pointer null");
        assert_eq!(lo.current_bulk_index(), 0, "index does not advance");
    }

    #[test]
    fn bulk_without_allocator_errors_only_on_allocating_paths() {
        // SHM path needs no allocator.
        let mut lo = load_with_recv(
            MsgType::SerializeIn,
            vec![Bulk {
                data: ShmPtr::new(AllocatorId::new(1, 0), 32),
                size: 8,
                flags: BULK_XFER,
                has_desc: false,
            }],
        );
        let mut ptr = ShmPtr::null();
        assert!(lo.bulk(&mut ptr, 8, BULK_XFER, None).is_ok());
        assert_eq!(ptr.off, 32);

        // The BULK_EXPOSE/null path does.
        let mut lo2 = load_with_recv(MsgType::SerializeIn, vec![Bulk::new(8, BULK_EXPOSE)]);
        let mut p2 = ShmPtr::null();
        assert_eq!(
            lo2.bulk(&mut p2, 8, BULK_EXPOSE, None).unwrap_err(),
            SerializeError::NoBulkAllocator
        );
    }

    #[test]
    fn bulk_out_copies_into_a_caller_supplied_buffer() {
        let mut lo = load_with_recv(
            MsgType::SerializeOut,
            vec![Bulk {
                data: ShmPtr::new(AllocatorId::new(2, 2), 800),
                size: 512,
                flags: BULK_XFER,
                has_desc: false,
            }],
        );
        let mut alloc = MockAlloc::default();
        let mut ptr = ShmPtr::new(AllocatorId::new(1, 0), 64); // caller's buffer
        lo.bulk(&mut ptr, 512, BULK_XFER, Some(&mut alloc)).unwrap();
        assert_eq!(alloc.copies, vec![(64, 800, 512)]);
        assert_eq!(ptr.off, 64, "caller's pointer stays valid");
        assert!(alloc.allocs.is_empty());
        assert_eq!(lo.current_bulk_index(), 1);
    }

    #[test]
    fn bulk_out_aliases_when_caller_has_no_buffer() {
        let mut lo = load_with_recv(
            MsgType::SerializeOut,
            vec![Bulk {
                data: ShmPtr::new(AllocatorId::new(2, 2), 800),
                size: 512,
                flags: BULK_XFER,
                has_desc: false,
            }],
        );
        let mut alloc = MockAlloc::default();
        let mut ptr = ShmPtr::null();
        lo.bulk(&mut ptr, 512, BULK_XFER, Some(&mut alloc)).unwrap();
        assert!(alloc.copies.is_empty());
        assert_eq!(ptr.off, 800, "zero-copy onto the recv buffer");
    }

    #[test]
    fn bulk_out_ignores_non_xfer_descriptors_but_advances() {
        let mut lo = load_with_recv(
            MsgType::SerializeOut,
            vec![Bulk {
                data: ShmPtr::new(AllocatorId::new(2, 2), 800),
                size: 512,
                flags: BULK_EXPOSE, // not XFER: no data came back
                has_desc: false,
            }],
        );
        let mut alloc = MockAlloc::default();
        let mut ptr = ShmPtr::null();
        lo.bulk(&mut ptr, 512, BULK_EXPOSE, Some(&mut alloc))
            .unwrap();
        assert!(ptr.is_null(), "untouched");
        assert!(alloc.copies.is_empty());
        assert_eq!(lo.current_bulk_index(), 1, "index still advances");
    }

    #[test]
    fn bulk_out_exhausted_is_silently_ignored() {
        // C++ has no else branch on the SerializeOut path.
        let mut lo = load_with_recv(MsgType::SerializeOut, vec![]);
        let mut ptr = ShmPtr::new(AllocatorId::new(1, 0), 5);
        assert!(lo.bulk(&mut ptr, 8, BULK_XFER, None).is_ok());
        assert_eq!(ptr.off, 5, "left untouched");
    }

    #[test]
    fn bulk_on_heartbeat_is_a_noop() {
        // C++ matches neither `if`, so nothing happens.
        let mut lo = load_with_recv(MsgType::Heartbeat, vec![Bulk::new(8, BULK_XFER)]);
        let mut ptr = ShmPtr::new(AllocatorId::new(1, 0), 77);
        assert!(lo.bulk(&mut ptr, 8, BULK_XFER, None).is_ok());
        assert_eq!(ptr.off, 77);
        assert_eq!(lo.current_bulk_index(), 0);
    }

    // -----------------------------------------------------------------
    // Task index
    // -----------------------------------------------------------------

    #[test]
    fn task_index_walks_and_resets() {
        let mut lo = LoadTaskArchive::new();
        assert!(lo.get_current_task_info().is_none(), "empty: None not UB");
        lo.meta.task_infos = vec![
            TaskInfo {
                method_id: 1,
                ..Default::default()
            },
            TaskInfo {
                method_id: 2,
                ..Default::default()
            },
        ];
        assert_eq!(lo.get_current_task_info().unwrap().method_id, 1);
        lo.advance_task_index();
        assert_eq!(lo.get_current_task_info().unwrap().method_id, 2);
        lo.advance_task_index();
        assert!(lo.get_current_task_info().is_none(), "past the end: None");
        lo.reset_task_index();
        assert_eq!(lo.current_task_index(), 0);
        assert_eq!(lo.get_current_task_info().unwrap().method_id, 1);
    }

    // -----------------------------------------------------------------
    // Misc
    // -----------------------------------------------------------------

    #[test]
    fn from_bytes_and_from_vec_agree() {
        let a = LoadTaskArchive::from_bytes(b"abc");
        let b = LoadTaskArchive::from_vec(b"abc".to_vec());
        assert_eq!(a.data(), b.data());
        assert_eq!(a.meta.msg_type, MsgType::SerializeIn, "C++ ctor default");
    }

    #[test]
    fn read_binary_on_load_archive_advances_the_cursor() {
        let mut lo = LoadTaskArchive::from_bytes(b"abcdef");
        let mut out = [0u8; 3];
        lo.read_binary(&mut out).unwrap();
        assert_eq!(&out, b"abc");
        assert_eq!(lo.cur_off(), 3);
        let mut rest = [0u8; 4];
        assert!(lo.read_binary(&mut rest).is_err(), "only 3 bytes remain");
        assert_eq!(lo.cur_off(), 3, "failed read does not advance");
    }

    /// The C++ `is_saving`/`is_loading`/`supports_range_ops` tag types are
    /// compile-time facts, so check them at compile time too: this whole block
    /// is a static assertion that fails the build, not the test run.
    const _: () = {
        assert!(GlobalSerialize::IS_SAVING && !GlobalSerialize::IS_LOADING);
        assert!(GlobalDeserialize::IS_LOADING && !GlobalDeserialize::IS_SAVING);
        assert!(SaveTaskArchive::IS_SAVING && !SaveTaskArchive::IS_LOADING);
        assert!(LoadTaskArchive::IS_LOADING && !LoadTaskArchive::IS_SAVING);
        assert!(GlobalSerialize::SUPPORTS_RANGE_OPS);
        assert!(GlobalDeserialize::SUPPORTS_RANGE_OPS);
    };

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
