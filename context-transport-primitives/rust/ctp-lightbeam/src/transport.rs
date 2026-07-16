// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
//! The lightbeam **transport abstraction**: the `Transport` interface, the
//! `Bulk`/`LbmMeta`/`LbmContext` currency it moves, the event types
//! transports register with, and the factory shape that builds backends.
//!
//! Ported from `clio_ctp/lightbeam/lightbeam.h`,
//! `lightbeam/transport_factory_impl.h`, `lightbeam/utils.h`, and the
//! platform-neutral half of `lightbeam/event_manager.h`.
//!
//! Concrete backends live in sibling modules (`shm_transport`,
//! `socket_transport`); ZMQ / thallium / NIXL are **explicitly out of scope**
//! here — they are wrapper-crate work over libzmq / Mochi-Argobots / NIXL
//! (see `rust/MIGRATION.md`: "thallium stays C++ behind a wrapper crate").
//! Their [`TransportType`] variants exist so a wrapper crate can register a
//! constructor with [`TransportFactory`] without this file changing.
//!
//! # C++ → Rust name mapping
//!
//! | C++ (`ctp::lbm`) | Rust | Notes |
//! |---|---|---|
//! | `BULK_EXPOSE` / `BULK_XFER` | [`BULK_EXPOSE`] / [`BULK_XFER`] | same bit positions (0, 1) |
//! | `LBM_SYNC` | [`LBM_SYNC`] | retained no-op flag; Send is unconditionally sync |
//! | `ctp::ipc::FullPtr<char>` | [`FullPtr`] | local mirror — divergence 5 |
//! | `Bulk` | [`Bulk`] | `desc`/`mr` become opaque handles — divergence 6 |
//! | `ClientInfo` | [`ClientInfo`] | `rc` / `fd_` / `identity_` → `rc` / `fd` / `identity` |
//! | `LbmMeta<AllocT>` | [`LbmMeta`] | allocator parameter dropped — divergence 3 |
//! | `LbmMeta::send_bulks` / `recv_bulks` | [`LbmMeta::send_bulks`] / [`LbmMeta::recv_bulks`] | plus [`LbmMeta::refresh_bulk_counts`] (addition) |
//! | `LbmContext` | [`LbmContext`] | `IsSync`/`HasTimeout`/`HasFileDst` → [`is_sync`](LbmContext::is_sync)/[`has_timeout`](LbmContext::has_timeout)/[`has_file_dst`](LbmContext::has_file_dst) |
//! | `TransportType::k*` | [`TransportType`] | discriminants preserved (kZeroMq = 0 … kThallium = 4) |
//! | `TransportMode::kClient/kServer` | [`TransportMode`] | discriminants preserved (0, 1) |
//! | `class Transport` | [`Transport`] trait | manual `type_` switch → vtable — divergence 1 |
//! | `Transport::Expose` | [`Transport::expose`] | required method |
//! | `Transport::Send` / `Recv` | [`Transport::send`] / [`Transport::recv`] | de-templated — divergence 2 |
//! | `Transport::GetAddress` | [`Transport::address`] | defaults to `""` |
//! | `Transport::ClearRecvHandles` | [`Transport::clear_recv_handles`] | defaults to no-op |
//! | `Transport::RegisterEventManager` | [`Transport::register_event_manager`] | `EventManager&` → `Arc<dyn EventManager>` — divergence 7 |
//! | `Transport::UnregisterEventManager` | [`Transport::unregister_event_manager`] | drops the `Arc` |
//! | `Transport::PollRecv` | [`Transport::poll_recv`] | defaults to 0 (non-ZMQ), ms |
//! | `Transport::GetBoundPort` | [`Transport::bound_port`] | defaults to 0 |
//! | `Transport::IsServerAlive` | [`Transport::is_server_alive`] | defaults to `false` |
//! | `TransportDeleter` | — | no counterpart — divergence 1 |
//! | `TransportPtr` (`unique_ptr`) | [`TransportPtr`] (`Box<dyn Transport>`) | |
//! | `TransportFactory::Get` (2 overloads) | [`TransportFactory::get`] / [`TransportFactory::get_with_domain`] | Rust has no default args |
//! | (ctor arg defaulting inside `Get`) | [`TransportParams::resolve`] | the defaulting table, testable on its own |
//! | `#if CTP_ENABLE_*` backend gating | [`TransportFactory::register`] | runtime registry — divergence 9 |
//! | `kDefaultReadEvent` | [`DEFAULT_READ_EVENT`] | `POLLRDNORM` on Windows, `EPOLLIN` elsewhere |
//! | `EventTrigger` | [`EventTrigger`] | `fd_`/`event_id_` → `fd`/`event_id` |
//! | `EventAction` (abstract) | [`EventAction`] trait | `Run` → `run(&self, ..)` — divergence 8 |
//! | `EventInfo` | [`EventInfo`] | owns `Option<Arc<dyn EventAction>>` — divergence 8 |
//! | `EventManager` (platform classes) | [`EventManager`] trait | only the surface transports call — divergence 7 |
//! | `utils::ParseUrl` | [`parse_url`] | declaration-only in C++ — divergence 12 |
//! | `utils::GetDefaultBufferSize` | [`default_buffer_size`] | 1024 |
//! | `utils::GetDefaultCqSize` | [`default_cq_size`] | 10 |
//!
//! # Semantic divergences
//!
//! 1. **Dispatch.** C++ avoids vtables: `Transport` has a non-virtual dtor, a
//!    `type_` discriminant, and every method is a `switch (type_)` that
//!    `static_cast`s to the backend (`transport_factory_impl.h`);
//!    `TransportDeleter` hand-rolls the destructor dispatch. Rust uses a trait
//!    object: `Box<dyn Transport>` drops through the vtable, so
//!    `TransportDeleter` has no counterpart. This also removes a live C++
//!    hazard — `Transport(TransportMode)` never initializes `type_`, leaving
//!    it indeterminate until a derived ctor sets it; here
//!    [`Transport::transport_type`] is an implemented method and cannot be
//!    uninitialized.
//! 2. **De-templated `Send`/`Recv`.** C++ templates them on `MetaT`; a generic
//!    method is not object-safe, so both take `&mut LbmMeta` (the `LbmMeta<>`
//!    host instantiation, which is also what the non-template
//!    `ClearRecvHandles` already pins). Backends may still offer generic
//!    inherent methods. Device-side metas (`CTP_IS_GPU`, allocator-templated
//!    `LbmMeta<AllocT>`) are not modeled: this port is the host path only,
//!    mirroring the `#if CTP_IS_HOST` guard around the C++ dispatch.
//! 3. **Allocator parameterization dropped.** `LbmMeta<AllocT>` holds
//!    `ctp::priv::vector<Bulk, AllocT>` plus `AllocT* alloc_` so bulk vectors
//!    can sit in GPU-accessible memory. [`LbmMeta`] uses `Vec<Bulk>`; the
//!    GPU-allocator variant is deferred with the rest of the device path.
//! 4. **Serialization not implemented.** `ctp-serialize` is not a dependency
//!    of this crate and Cargo.toml is not mine to edit. The C++ wire contract
//!    is preserved as documentation and honored by the field layout:
//!    `Bulk::serialize` writes **only** `(size, flags)` — never `data`,
//!    `desc`, or `mr`; `LbmMeta::serialize` writes
//!    `(send, recv, send_bulks, recv_bulks)` — never `client_info_` or
//!    `alloc_`; `FullPtr::serialize` writes only `shm_`. See
//!    [`Bulk::SERIALIZED_FIELDS`] / [`LbmMeta::SERIALIZED_FIELDS`].
//! 5. **`FullPtr` comes from `ctp-memory`.** `Bulk::data` needs both halves
//!    (a socket transport may expose plain heap memory that lives in no
//!    registered segment). This module used to mirror `FullPtr` locally
//!    because `ctp-memory` had none; it has one now — in the same layer the
//!    C++ keeps it (`memory/allocator/allocator.h`) — so the mirror is gone
//!    and [`FullPtr`] is an alias pinning the `FullPtr<char>` instantiation
//!    lightbeam uses. Resolved: there is one `FullPtr` again.
//! 6. **Raw backend pointers → opaque handles.** `Bulk::desc`/`Bulk::mr`
//!    (libfabric `fid_mr*` etc.) and `LbmContext::copy_space`/`shm_info_`/
//!    `meta_buf_` are `void*`/`ShmTransferInfo*` in C++. Their pointee types
//!    live in backend modules this file does not own, so they cross this
//!    abstraction as `usize` addresses (0 = null), keeping it `unsafe`-free
//!    and `Send`. Backends recover typed pointers at their own unsafe
//!    boundary (exposed-provenance cast). Consequence: the compiler cannot
//!    check these handles — the same guarantee the C++ `void*` gives, no worse.
//! 7. **EventManager.** The platform classes (epoll / kqueue / IOCP, plus
//!    `AddSignalEvent`, `Signal`, `GetEpollFd`, `GetSignalFd`) are OS code and
//!    are **not** ported here. This file defines only the surface transports
//!    actually call: [`add_event`](EventManager::add_event),
//!    [`remove_event`](EventManager::remove_event),
//!    [`wait_us`](EventManager::wait_us). Three sub-divergences:
//!    (a) `Wait(int timeout_us = -1)` is in **microseconds** (`shm_transport.h`
//!    passes `200` as a 200us re-check) — the crate convention is ms, so the
//!    method is named `wait_us` to make the unit impossible to misread;
//!    (b) C++ holds a non-owning `EventManager*` that outlives nothing in
//!    particular — `UnregisterEventManager` exists precisely because the
//!    caller destroys the manager under the transport. `Arc<dyn EventManager>`
//!    makes that detach memory-safe rather than merely conventional;
//!    (c) C++ reaches the per-thread manager through `IpcManager::GetTls`
//!    (a `thread_local`). The project rule bans `thread_local!`, so the
//!    manager is passed explicitly (register call / [`LbmContext`]).
//!    Because the handle is shared, the trait methods take `&self` (C++'s are
//!    non-const); implementors use interior mutability.
//! 8. **Event actions.** `EventAction::Run` is a non-const virtual and
//!    `EventInfo::action_` is a raw non-owning pointer. Rust: `run(&self,
//!    &EventInfo)` on an `Arc`-shared handler owned by [`EventInfo`].
//! 9. **Factory.** C++ selects backends with `#if CTP_ENABLE_*` and `new`s the
//!    concrete type inside the switch. The abstraction cannot name sibling
//!    backends, so a process-global constructor registry
//!    (`OnceLock<RwLock<HashMap>>`, not `thread_local`) replaces the
//!    compile-time switch: an unregistered type yields `None`, exactly as the
//!    C++ `default: return nullptr` arm does for a compiled-out backend. The
//!    ctor-argument defaulting is ported verbatim in
//!    [`TransportParams::resolve`], **including** its asymmetries: ZMQ
//!    defaults port 0 → 8192, socket 0 → 8193, but **thallium's port is passed
//!    through undefaulted** (a 0 stays 0) and SHM/NIXL ignore protocol/port
//!    entirely.
//! 10. **`domain`.** The C++ 6-arg overload discards it (`(void)domain`) and is
//!     otherwise identical to the 5-arg one. [`TransportParams`] carries it —
//!     a superset; no backend ctor consumes it today.
//! 11. **`ClientInfo::identity`.** C++ guards it with `#if !CTP_IS_GPU`; this
//!     host-only port keeps it unconditional. `rc` stays a raw errno-style
//!     int — `EAGAIN` is not re-exported because `libc` is not a dependency
//!     of this crate.
//! 12. **`parse_url`.** `utils::ParseUrl` is **declared but never defined and
//!     never called** anywhere in the C++ tree — there is no behavior to match.
//!     The port implements the documented `"protocol://host:port"` contract and
//!     *defines* the edges C++ left open: no scheme, no port, empty input, and
//!     bracketed IPv6. Unbracketed IPv6 is not disambiguable and splits at the
//!     last colon, per the usual URL rule.
//! 13. **Trait defaults = the C++ `default:` arms that are real behavior**:
//!     [`poll_recv`](Transport::poll_recv) → 0 ("non-ZMQ transports use the
//!     EventManager path"), [`bound_port`](Transport::bound_port) → 0
//!     ("clients / non-TCP / non-ZMQ"), [`address`](Transport::address) → `""`,
//!     [`is_server_alive`](Transport::is_server_alive) → `false`,
//!     [`register_event_manager`](Transport::register_event_manager) /
//!     [`unregister_event_manager`](Transport::unregister_event_manager) /
//!     [`clear_recv_handles`](Transport::clear_recv_handles) → no-ops (the SHM
//!     arm). `expose`/`send`/`recv` are **required**: their `default:` arms
//!     (`Bulk{}` / `-1` / `ClientInfo{-1,-1,{}}`) only fired for a
//!     compiled-out backend, which trait objects make unreachable.
//! 14. **Zero-init.** C++ `Bulk::size`/`flags` have no default member
//!     initializers, so `Bulk b;` leaves `size` indeterminate while `Bulk{}`
//!     zero-initializes. Rust's [`Default`] is always the null/zero form.
//! 15. **`Transport: Send`.** C++ states no such constraint but moves
//!     `TransportPtr` onto recv threads; the supertrait encodes that. `Sync` is
//!     deliberately not required (backends hold non-shareable handles).
//! 16. **No separate Client/Server traits.** C++ lightbeam has one `Transport`
//!     class discriminated by a runtime `mode_`, not a client and a server
//!     type; splitting it here would invent API the C++ does not have. The
//!     mode split survives as [`TransportMode`] plus the
//!     [`is_server`](Transport::is_server)/[`is_client`](Transport::is_client)
//!     predicates, and server-only methods are documented as such.
//! 17. **`char` → `u8`.** `FullPtr<char>` addresses bytes, not text.


/// The `(process-local address, shared-memory pointer)` pair lightbeam
/// exposes for bulk transfer.
///
/// Lives in `ctp-memory` now, where the C++ also keeps it
/// (`memory/allocator/allocator.h`); this module used to carry a local mirror
/// because `ctp-memory` had no `FullPtr` yet, so there were two. This alias
/// pins the instantiation lightbeam uses, mirroring the C++ `FullPtr<char>`
/// (Rust spells the byte `u8`), and lets `FullPtr::null()` and friends infer
/// their type parameter as they did when this was a concrete struct.
pub type FullPtr = ctp_memory::FullPtr<u8>;
use crate::shm_transport::ShmRing;
use ctp_memory::AllocatorId;
use ctp_types::{bit_opt, Bitfield32};
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, OnceLock, RwLock};

// ---------------------------------------------------------------------------
// Bulk flags (lightbeam.h "--- Bulk Flags ---")
// ---------------------------------------------------------------------------

/// Bulk metadata is sent, but no data transfer occurs (C++ `BULK_EXPOSE`).
pub const BULK_EXPOSE: u32 = bit_opt(0);

/// Bulk is marked for data transmission (C++ `BULK_XFER`).
pub const BULK_XFER: u32 = bit_opt(1);

/// Retained no-op flag (C++ `LBM_SYNC`).
///
/// Send is unconditionally synchronous; the flag exists only so callers that
/// still pass `LbmContext(LBM_SYNC)` keep working.
pub const LBM_SYNC: u32 = 0x1;

// ---------------------------------------------------------------------------
// FullPtr (mirror of ctp::ipc::FullPtr<char> — divergence 5)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Bulk (lightbeam.h "--- Types ---")
// ---------------------------------------------------------------------------

/// A registered memory region participating in a transfer (C++ `Bulk`).
///
/// Deliberately **not** `ShmSafe`: `data.ptr`, `desc`, and `mr` are
/// process-local addresses/handles, so a `Bulk` must never be placed in a
/// shared segment. Only `(size, flags)` is serialized — see
/// [`Bulk::SERIALIZED_FIELDS`] and divergence 4.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bulk {
    /// C++ `ctp::ipc::FullPtr<char> data`.
    pub data: FullPtr,
    /// Region length in bytes (C++ `size_t size`).
    pub size: usize,
    /// [`BULK_EXPOSE`] or [`BULK_XFER`] (C++ `ctp::bitfield32_t flags`).
    pub flags: Bitfield32,
    /// RDMA memory-registration descriptor (C++ `void* desc`); 0 = null.
    pub desc: usize,
    /// RDMA memory-region handle, e.g. `fid_mr*` (C++ `void* mr`); 0 = null.
    pub mr: usize,
}

/// Marks a receive buffer the **transport itself** allocated, and which
/// [`Transport::clear_recv_handles`] must therefore free (C++
/// `AllocatorId(UINT32_MAX - 1, UINT32_MAX - 1)`).
///
/// This is not a decoration on "is it owned" — it records *which allocator*
/// owns the buffer, and that distinction is load-bearing. A recv `Bulk` can
/// end up holding either:
///
/// - a buffer this transport allocated on the raw system allocator, which
///   `clear_recv_handles` frees; or
/// - a buffer belonging to a CTP allocator, swapped in by the task archive's
///   bulk path for the `BULK_EXPOSE` / copy routes. Those are reclaimed by the
///   task instead (`daemon_allocated_bulk_count_` / `TASK_DATA_OWNER`).
///
/// Freeing the second kind as if it were the first is a real crash, not a
/// leak: a CTP `MallocAllocator`'s user pointer sits 16 bytes inside the true
/// malloc region, so handing it to `free` gives glibc "free(): invalid
/// pointer" (ASan: bad-free). The sentinel is what keeps the two apart, which
/// is why it is a distinctive id rather than a bool.
pub const RECV_ALLOCATED_ID: AllocatorId = AllocatorId::new(u32::MAX - 1, u32::MAX - 1);

impl Bulk {
    /// The exact field set C++ `Bulk::serialize` writes, in order
    /// (divergence 4). `data`, `desc`, and `mr` are local state and stay home.
    pub const SERIALIZED_FIELDS: [&'static str; 2] = ["size", "flags"];

    /// True when this descriptor's buffer was allocated by the transport's
    /// receive path and so must be freed by `clear_recv_handles`.
    ///
    /// Mirrors the C++ guard: a non-null local half **and** the
    /// [`RECV_ALLOCATED_ID`] sentinel. Both halves matter — a null pointer
    /// has nothing to free, and a different allocator id means somebody else
    /// owns it.
    pub fn is_recv_allocated(&self) -> bool {
        self.data.ptr != 0 && self.data.shm.alloc_id == RECV_ALLOCATED_ID
    }

    /// A bulk descriptor over `data` of `size` bytes with `flags`.
    pub fn new(data: FullPtr, size: usize, flags: u32) -> Self {
        Self {
            data,
            size,
            flags: Bitfield32::new(flags),
            desc: 0,
            mr: 0,
        }
    }

    /// True if this descriptor is marked for data transmission
    /// ([`BULK_XFER`]).
    pub fn is_xfer(&self) -> bool {
        self.flags.any(BULK_XFER)
    }

    /// True if this descriptor is metadata-only ([`BULK_EXPOSE`]).
    pub fn is_expose(&self) -> bool {
        self.flags.any(BULK_EXPOSE)
    }
}

impl Default for Bulk {
    /// C++ `Bulk{}` (the zero-initializing form — see divergence 14).
    fn default() -> Self {
        Self {
            data: FullPtr::null(),
            size: 0,
            flags: Bitfield32::new(0),
            desc: 0,
            mr: 0,
        }
    }
}

/// Allocate a zeroed receive buffer of exactly `len` bytes and leak it,
/// returning its address (0 is never returned).
///
/// C++ `RecvBulks` does `malloc(size)` here. The buffer is owned by the `Bulk`
/// that points at it and is reclaimed by [`SocketTransport::clear_recv_handles`]
/// via [`free_recv_buffer`] — so `Bulk`, like the C++ one, holds a location
/// rather than storage.
///
/// Fallible (`try_reserve_exact`): a bad `size` off the wire must not abort.
/// `into_boxed_slice` pins capacity to `len`, which is what makes freeing with
/// the recorded size sound.
pub(crate) fn alloc_recv_buffer(len: usize) -> Option<usize> {
    let mut v: Vec<u8> = Vec::new();
    v.try_reserve_exact(len).ok()?;
    v.resize(len, 0);
    let mut b = v.into_boxed_slice();
    let addr = b.as_mut_ptr() as usize;
    std::mem::forget(b);
    Some(addr)
}

/// Free a buffer from [`alloc_recv_buffer`].
///
/// # Safety
///
/// `addr` must come from [`alloc_recv_buffer`], and `len` must be the exact
/// length it was allocated with — `Bulk::size`, which must not have been
/// changed since. Nothing else may free it. C++ gets to use size-agnostic
/// `free` here; Rust's allocator wants the layout back, so the size invariant
/// is the price of the same operation.
pub(crate) unsafe fn free_recv_buffer(addr: usize, len: usize) {
    let slice = std::ptr::slice_from_raw_parts_mut(addr as *mut u8, len);
    drop(unsafe { Box::from_raw(slice) });
}

/// The bytes a `Bulk` points at.
///
/// # Safety
///
/// `bulk.data.ptr` must address at least `bulk.size` valid, initialized bytes
/// that stay alive and unaliased for `'a`. This is the C++ contract verbatim:
/// `Bulk` carries a location and the sender guarantees the memory. It is also
/// the one place the safety has to be given up — a `FullPtr` may point into a
/// shared segment, and a cross-process pointer cannot be a safe slice.
pub(crate) unsafe fn bulk_bytes<'a>(bulk: &Bulk) -> &'a [u8] {
    // A zero-length bulk carries no bytes and may legitimately have a null
    // pointer. `from_raw_parts` demands non-null even for length 0, so this
    // must be handled here rather than at each call site — C++ would simply
    // pass (nullptr, 0) to writev and never notice.
    if bulk.size == 0 {
        return &[];
    }
    unsafe { std::slice::from_raw_parts(bulk.data.ptr as *const u8, bulk.size) }
}

// ---------------------------------------------------------------------------
// ClientInfo (lightbeam.h "--- Client Info ---")
// ---------------------------------------------------------------------------

/// A socket handle, as an opaque routing id (C++ `sock::socket_t`).
///
/// One `u64` on every platform, from `AsRawFd`/`AsRawSocket`. The C++ stores
/// this in `ClientInfo` as an `int`, which **truncates a Windows `SOCKET`**
/// (a `UINT_PTR`); that is a latent bug rather than a contract, and it costs
/// nothing to fix here because `ClientInfo` is never serialized — the C++
/// marks the field "not serialized, host-only" — so its width is an
/// implementation choice with no ABI consequence.
///
/// Ids are handles for routing and lookup only; the sockets themselves are
/// owned `TcpStream`/`UnixStream` values.
pub type SocketId = u64;

/// "No socket" (C++ `sock::kInvalidSocket`, the Windows `INVALID_SOCKET`/`~0`
/// convention). This is what the C++ `fd_ = -1` initializer means.
pub const INVALID_SOCKET: SocketId = u64::MAX;

/// Routing info returned by [`Transport::recv`], consumed by
/// [`Transport::send`] (C++ `ClientInfo`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientInfo {
    /// Return code: 0 = success, `EAGAIN` = no data, etc. (divergence 11).
    pub rc: i32,
    /// Socket handle (socket-transport server mode); [`INVALID_SOCKET`] =
    /// none. C++ `fd_`, an `int` there — see [`SocketId`].
    pub fd: SocketId,
    /// ZMQ identity (ZMQ server mode). C++ `identity_`, host-only.
    pub identity: String,
}

impl Default for ClientInfo {
    /// The C++ default member initializers: `rc = 0`, **`fd_ = -1`**, empty
    /// identity.
    ///
    /// Hand-written rather than derived, because a derived `Default` would
    /// zero `fd` — and `fd = 0` is not "no fd", it is standard input. C++
    /// default-constructs a `ClientInfo` with `fd_ = -1`, so this must too;
    /// the two differ in exactly the case that matters and agree everywhere
    /// else, which is how the derive survived: nothing called it.
    fn default() -> Self {
        Self::new()
    }
}

impl ClientInfo {
    /// C++ default member initializers: `rc = 0`, `fd_ = -1` (spelled
    /// [`INVALID_SOCKET`] here), empty identity. Same as [`Default::default`].
    pub fn new() -> Self {
        Self {
            rc: 0,
            fd: INVALID_SOCKET,
            identity: String::new(),
        }
    }

    /// The C++ dispatch's failure value, `ClientInfo{-1, -1, {}}`.
    pub fn failed() -> Self {
        Self {
            rc: -1,
            fd: INVALID_SOCKET,
            identity: String::new(),
        }
    }

    /// The C++ error convention: a non-zero errno-style `rc`.
    pub fn err(rc: i32) -> Self {
        Self {
            rc,
            fd: INVALID_SOCKET,
            identity: String::new(),
        }
    }

    /// True if `rc == 0`.
    pub fn is_ok(&self) -> bool {
        self.rc == 0
    }
}

// ---------------------------------------------------------------------------
// LbmMeta (lightbeam.h "--- Metadata Base Class ---")
// ---------------------------------------------------------------------------

/// Per-transfer metadata: the sender's and receiver's bulk descriptors
/// (C++ `LbmMeta<AllocT>`; the allocator parameter is dropped — divergence 3).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LbmMeta {
    /// Sender's bulk descriptors ([`BULK_EXPOSE`] or [`BULK_XFER`]).
    pub send: Vec<Bulk>,
    /// Receiver's descriptors: a copy of `send` with local pointers.
    pub recv: Vec<Bulk>,
    /// Count of [`BULK_XFER`] entries in [`LbmMeta::send`].
    pub send_bulks: usize,
    /// Count of [`BULK_XFER`] entries in [`LbmMeta::recv`].
    pub recv_bulks: usize,
    /// Client routing info — **not serialized**, host-only (C++
    /// `client_info_`).
    pub client_info: ClientInfo,
}

impl LbmMeta {
    /// The exact field set C++ `LbmMeta::serialize` writes, in order
    /// (divergence 4). `client_info_` and `alloc_` are local state.
    pub const SERIALIZED_FIELDS: [&'static str; 4] = ["send", "recv", "send_bulks", "recv_bulks"];

    /// Empty metadata (C++ default ctor, minus the allocator plumbing).
    pub fn new() -> Self {
        Self::default()
    }

    /// Count the [`BULK_XFER`]-marked descriptors in `bulks`.
    ///
    /// **Rust addition** (no C++ counterpart): C++ callers maintain
    /// `send_bulks`/`recv_bulks` by hand. This is the invariant those fields
    /// document, made executable.
    pub fn count_xfer(bulks: &[Bulk]) -> usize {
        bulks.iter().filter(|b| b.is_xfer()).count()
    }

    /// Recompute [`LbmMeta::send_bulks`] / [`LbmMeta::recv_bulks`] from the
    /// vectors. **Rust addition** — see [`LbmMeta::count_xfer`].
    pub fn refresh_bulk_counts(&mut self) {
        self.send_bulks = Self::count_xfer(&self.send);
        self.recv_bulks = Self::count_xfer(&self.recv);
    }
}

// ---------------------------------------------------------------------------
// Events (event_manager.h — platform-neutral half; divergences 7 and 8)
// ---------------------------------------------------------------------------

/// Platform-neutral read-event constant (C++ `kDefaultReadEvent`):
/// `POLLRDNORM` on Windows, `EPOLLIN` elsewhere.
#[cfg(windows)]
pub const DEFAULT_READ_EVENT: u32 = 0x0100;

/// Platform-neutral read-event constant (C++ `kDefaultReadEvent`):
/// `POLLRDNORM` on Windows, `EPOLLIN` elsewhere.
#[cfg(not(windows))]
pub const DEFAULT_READ_EVENT: u32 = 0x001;

/// What an event registration is keyed on (C++ `EventTrigger`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EventTrigger {
    /// C++ `fd_`.
    pub fd: i32,
    /// C++ `event_id_`.
    pub event_id: i32,
}

/// A handler invoked when its registration fires (C++ abstract `EventAction`).
///
/// `run` takes `&self` (C++ `Run` is non-const): the handler is shared through
/// an [`Arc`], so implementors that mutate use interior mutability
/// (divergence 8).
pub trait EventAction: Send + Sync {
    /// C++ `virtual void Run(const EventInfo& event) = 0`.
    fn run(&self, event: &EventInfo);
}

/// A fired (or registered) event (C++ `EventInfo`).
#[derive(Clone, Default)]
pub struct EventInfo {
    /// C++ `trigger_`.
    pub trigger: EventTrigger,
    /// Event mask, e.g. [`DEFAULT_READ_EVENT`] (C++ `events_`).
    pub events: u32,
    /// C++ `EventAction* action_` — owning and optional here (divergence 8).
    pub action: Option<Arc<dyn EventAction>>,
}

impl EventInfo {
    /// An event with no handler attached (C++ `action_ = nullptr`).
    pub fn new(trigger: EventTrigger, events: u32) -> Self {
        Self {
            trigger,
            events,
            action: None,
        }
    }

    /// Attach a handler.
    pub fn with_action(mut self, action: Arc<dyn EventAction>) -> Self {
        self.action = Some(action);
        self
    }

    /// Invoke the handler, if any — what the C++ EventManager's dispatch loop
    /// does (`action_->Run(event)`, guarded on a null `action_`).
    pub fn dispatch(&self) {
        if let Some(action) = &self.action {
            action.run(self);
        }
    }
}

impl fmt::Debug for EventInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EventInfo")
            .field("trigger", &self.trigger)
            .field("events", &self.events)
            .field("action", &self.action.is_some())
            .finish()
    }
}

/// The readiness multiplexer a transport registers its fds with — only the
/// surface transports call (divergence 7). Platform implementations
/// (epoll / kqueue / IOCP, plus `AddSignalEvent`/`Signal`/`GetEpollFd`) are
/// not part of this abstraction.
pub trait EventManager: Send + Sync {
    /// C++ `int AddEvent(int fd, uint32_t events, EventAction* action)`.
    /// Returns the event id, or -1 on failure.
    fn add_event(&self, fd: i32, events: u32, action: Option<Arc<dyn EventAction>>) -> i32;

    /// C++ `void RemoveEvent(int fd)`. Must be called *before* closing `fd`,
    /// or a recycled fd number trips a stale registration.
    fn remove_event(&self, fd: i32);

    /// C++ `int Wait(int timeout_us = -1)` — **microseconds**, negative =
    /// wait forever. Returns the number of events handled, or -1.
    fn wait_us(&self, timeout_us: i32) -> i32;
}

// ---------------------------------------------------------------------------
// LbmContext (lightbeam.h "--- LbmContext ---")
// ---------------------------------------------------------------------------

/// Per-call transport options (C++ `LbmContext`).
///
/// The remaining raw backend pointer (`meta_buf`) crosses as an opaque
/// address — divergence 6.
#[derive(Clone, Default)]
pub struct LbmContext {
    /// Combination of `LBM_*` flags (C++ `flags`).
    pub flags: u32,
    /// Timeout in **milliseconds**; 0 = no timeout (C++ `timeout_ms`).
    pub timeout_ms: i32,
    /// The SPSC ring this transfer runs over: C++ `char* copy_space` and
    /// `ShmTransferInfo* shm_info_` fused into one handle.
    ///
    /// The two C++ fields are a pair — neither is meaningful without the
    /// other — so they cross as a single [`ShmRing`] that cannot be
    /// half-populated, rather than as two independent opaque addresses.
    ///
    /// That `lightbeam.h`'s context knows a type from `shm_transport.h` is not
    /// an inversion this port invents: the C++ header forward-declares
    /// `struct ShmTransferInfo;` and holds a pointer to it for exactly this.
    pub ring: Option<ShmRing>,
    /// Pre-allocated framing buffer for the SHM transport, avoiding a heap
    /// allocation (C++ `char* meta_buf_`); 0 = none.
    pub meta_buf: usize,
    /// Capacity of [`LbmContext::meta_buf`] (C++ `meta_buf_size_`).
    pub meta_buf_size: usize,
    /// True = all 32 lanes participate in the copy (C++ `warp_parallel_`).
    pub warp_parallel: bool,
    /// Server PID for the SHM liveness check (C++ `server_pid_`).
    pub server_pid: i32,
    /// Destination fd for CPU→storage; -1 = none (C++ `dst_fd_`).
    pub dst_fd: i32,
    /// Offset within the destination file (C++ `dst_offset_`).
    pub dst_offset: usize,
    /// Waiter PID for Send→`EventManager::Signal`; 0 = none (C++
    /// `signal_pid_`).
    pub signal_pid: i32,
    /// Waiter TID for Send→`EventManager::Signal` (C++ `signal_tid_`).
    pub signal_tid: i32,
    /// The waiter's own manager for [`Transport::recv`] to block on; `None` on
    /// the runtime/Send side (C++ `EventManager* event_manager_`, fetched from
    /// a thread_local — divergence 7c).
    pub event_manager: Option<Arc<dyn EventManager>>,
}

impl LbmContext {
    /// C++ `LbmContext()` — no flags, no timeout.
    pub fn new() -> Self {
        Self {
            flags: 0,
            timeout_ms: 0,
            ring: None,
            meta_buf: 0,
            meta_buf_size: 0,
            warp_parallel: false,
            server_pid: 0,
            dst_fd: -1,
            dst_offset: 0,
            signal_pid: 0,
            signal_tid: 0,
            event_manager: None,
        }
    }

    /// C++ `explicit LbmContext(uint32_t f)` — timeout 0.
    pub fn with_flags(flags: u32) -> Self {
        Self {
            flags,
            ..Self::new()
        }
    }

    /// Attach the SPSC ring this transfer streams through (builder style).
    ///
    /// **Rust addition**: C++ callers assign `copy_space` and `shm_info_`
    /// separately, which allows setting one and forgetting the other. One
    /// handle, set once, cannot be half-populated.
    pub fn with_ring(mut self, ring: ShmRing) -> Self {
        self.ring = Some(ring);
        self
    }

    /// C++ `LbmContext(uint32_t f, int timeout)`. `timeout_ms` is
    /// milliseconds.
    pub fn with_timeout(flags: u32, timeout_ms: i32) -> Self {
        Self {
            flags,
            timeout_ms,
            ..Self::new()
        }
    }

    /// C++ `LbmContext(uint32_t f, int timeout, int dst_fd, size_t
    /// dst_offset)` — a CPU→storage transfer.
    pub fn with_file_dst(flags: u32, timeout_ms: i32, dst_fd: i32, dst_offset: usize) -> Self {
        Self {
            flags,
            timeout_ms,
            dst_fd,
            dst_offset,
            ..Self::new()
        }
    }

    /// C++ `IsSync()`.
    pub const fn is_sync(&self) -> bool {
        (self.flags & LBM_SYNC) != 0
    }

    /// C++ `HasTimeout()`: strictly positive — 0 *and negatives* mean "no
    /// timeout".
    pub const fn has_timeout(&self) -> bool {
        self.timeout_ms > 0
    }

    /// C++ `HasFileDst()`: `dst_fd_ >= 0` (fd 0 counts).
    pub const fn has_file_dst(&self) -> bool {
        self.dst_fd >= 0
    }
}

impl fmt::Debug for LbmContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LbmContext")
            .field("flags", &self.flags)
            .field("timeout_ms", &self.timeout_ms)
            .field("ring", &self.ring)
            .field("meta_buf", &self.meta_buf)
            .field("meta_buf_size", &self.meta_buf_size)
            .field("warp_parallel", &self.warp_parallel)
            .field("server_pid", &self.server_pid)
            .field("dst_fd", &self.dst_fd)
            .field("dst_offset", &self.dst_offset)
            .field("signal_pid", &self.signal_pid)
            .field("signal_tid", &self.signal_tid)
            .field("event_manager", &self.event_manager.is_some())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Transport type / mode enums
// ---------------------------------------------------------------------------

/// Which backend implements a transport (C++ `enum class TransportType`).
/// Discriminants match the C++ declaration order for FFI parity.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransportType {
    /// C++ `kZeroMq` — out of scope here (wrapper crate over libzmq).
    ZeroMq = 0,
    /// C++ `kSocket`.
    Socket = 1,
    /// C++ `kShm`.
    Shm = 2,
    /// C++ `kNixl` — out of scope here (wrapper crate over NIXL).
    Nixl = 3,
    /// C++ `kThallium` — out of scope here; stays C++ behind a wrapper crate
    /// (Mochi/Argobots), per `MIGRATION.md`.
    Thallium = 4,
}

/// Which end of a connection a transport is (C++ `enum class TransportMode`).
///
/// This is a *runtime* discriminant, exactly as in C++: lightbeam has no
/// separate Client/Server classes, so this port has no separate client/server
/// traits either (divergence 16).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransportMode {
    /// C++ `kClient`.
    Client = 0,
    /// C++ `kServer`.
    Server = 1,
}

// ---------------------------------------------------------------------------
// Transport (lightbeam.h "--- Unified Transport Interface ---")
// ---------------------------------------------------------------------------

/// The unified transport interface (C++ `class Transport`).
///
/// See divergence 1 (dispatch), 2 (de-templated send/recv), 13 (which methods
/// carry defaults) and 16 (why there is no client/server split).
pub trait Transport: Send {
    /// C++ `type_`.
    fn transport_type(&self) -> TransportType;

    /// C++ `mode_`.
    fn mode(&self) -> TransportMode;

    /// C++ `IsServer()`.
    fn is_server(&self) -> bool {
        self.mode() == TransportMode::Server
    }

    /// C++ `IsClient()`.
    fn is_client(&self) -> bool {
        self.mode() == TransportMode::Client
    }

    /// Register `data_size` bytes at `ptr` for transfer (C++ `Expose`).
    /// `flags` is [`BULK_EXPOSE`] or [`BULK_XFER`].
    fn expose(&mut self, ptr: FullPtr, data_size: usize, flags: u32) -> Bulk;

    /// Send `meta` (C++ `Send`). Returns 0 on success, negative on failure.
    fn send(&mut self, meta: &mut LbmMeta, ctx: &LbmContext) -> i32;

    /// Receive into `meta` (C++ `Recv`), returning routing info for a
    /// subsequent [`Transport::send`].
    fn recv(&mut self, meta: &mut LbmMeta, ctx: &LbmContext) -> ClientInfo;

    /// Server-only: this transport's address (C++ `GetAddress`). Defaults to
    /// `""`.
    fn address(&self) -> String {
        String::new()
    }

    /// Release receive-side handles held in `meta` (C++ `ClearRecvHandles`).
    fn clear_recv_handles(&mut self, meta: &mut LbmMeta) {
        let _ = meta;
    }

    /// Register this transport's fds with `em` (C++ `RegisterEventManager`).
    /// Defaults to a no-op — the SHM arm.
    fn register_event_manager(&mut self, em: Arc<dyn EventManager>) {
        let _ = em;
    }

    /// Detach from the event manager (C++ `UnregisterEventManager`).
    ///
    /// Deliberately does **not** remove registrations: the caller is typically
    /// destroying the manager itself, and touching it here would be a
    /// use-after-free in C++ (here, dropping the [`Arc`] is enough).
    fn unregister_event_manager(&mut self) {}

    /// Block up to `timeout_ms` **milliseconds** until readable, using the
    /// backend's native readiness primitive (C++ `PollRecv`).
    ///
    /// Defaults to 0: non-ZMQ transports block through their [`EventManager`]
    /// registration instead.
    fn poll_recv(&mut self, timeout_ms: i32) -> i32 {
        let _ = timeout_ms;
        0
    }

    /// Server-only: the actual bound TCP port, resolving ephemeral port-0
    /// binds (C++ `GetBoundPort`). Defaults to 0 for clients / non-TCP /
    /// backends without the notion.
    fn bound_port(&self) -> i32 {
        0
    }

    /// Liveness check (C++ `IsServerAlive`). Defaults to `false`.
    fn is_server_alive(&self, ctx: &LbmContext) -> bool {
        let _ = ctx;
        false
    }
}

/// C++ `TransportPtr` (`std::unique_ptr<Transport, TransportDeleter>`).
/// The deleter's hand-rolled type switch is the vtable here — divergence 1.
pub type TransportPtr = Box<dyn Transport>;

// ---------------------------------------------------------------------------
// Factory (lightbeam.h + transport_factory_impl.h)
// ---------------------------------------------------------------------------

/// Default protocol when the caller passes an empty string (C++
/// `protocol.empty() ? "tcp" : protocol`).
pub const DEFAULT_PROTOCOL: &str = "tcp";

/// Default ZMQ port when the caller passes 0 (C++ `port == 0 ? 8192 : port`).
pub const DEFAULT_ZMQ_PORT: i32 = 8192;

/// Default socket port when the caller passes 0 (C++ `port == 0 ? 8193 :
/// port`).
pub const DEFAULT_SOCKET_PORT: i32 = 8193;

/// Backend constructor arguments, after the factory's defaulting rules.
///
/// This is the C++ `TransportFactory::Get` switch body turned into data
/// (divergence 9), so the defaulting table is testable without any backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransportParams {
    /// Address the backend binds/connects to. Unused by SHM.
    pub addr: String,
    /// Which backend to build.
    pub ttype: TransportType,
    /// Client or server.
    pub mode: TransportMode,
    /// Resolved protocol. Unused by SHM/NIXL.
    pub protocol: String,
    /// Resolved port. Unused by SHM/NIXL.
    pub port: i32,
    /// Accepted and ignored by every C++ ctor (`(void)domain`) — divergence 10.
    pub domain: String,
}

impl TransportParams {
    /// Apply the C++ factory's per-type defaulting, verbatim:
    ///
    /// | type | protocol | port |
    /// |---|---|---|
    /// | `ZeroMq` | `""` → `"tcp"` | `0` → 8192 |
    /// | `Socket` | `""` → `"tcp"` | `0` → 8193 |
    /// | `Thallium` | `""` → `"tcp"` | **passed through** (0 stays 0) |
    /// | `Shm` | unused by the ctor — passed through | unused — passed through |
    /// | `Nixl` | unused by the ctor — passed through | unused — passed through |
    ///
    /// Non-zero ports (including negatives) are always passed through: the C++
    /// test is `port == 0`, not `port <= 0`.
    pub fn resolve(
        addr: &str,
        ttype: TransportType,
        mode: TransportMode,
        protocol: &str,
        port: i32,
        domain: &str,
    ) -> Self {
        let defaulted_protocol = || {
            if protocol.is_empty() {
                DEFAULT_PROTOCOL.to_string()
            } else {
                protocol.to_string()
            }
        };
        let (protocol, port) = match ttype {
            TransportType::ZeroMq => (
                defaulted_protocol(),
                if port == 0 { DEFAULT_ZMQ_PORT } else { port },
            ),
            TransportType::Socket => (
                defaulted_protocol(),
                if port == 0 { DEFAULT_SOCKET_PORT } else { port },
            ),
            // ThalliumTransport(mode, addr, protocol.empty() ? "tcp" :
            // protocol, port) — the C++ does NOT default thallium's port.
            TransportType::Thallium => (defaulted_protocol(), port),
            // ShmTransport(mode) / NixlTransport(mode, addr): the ctors take
            // neither protocol nor port, so nothing is defaulted.
            TransportType::Shm | TransportType::Nixl => (protocol.to_string(), port),
        };
        Self {
            addr: addr.to_string(),
            ttype,
            mode,
            protocol,
            port,
            domain: domain.to_string(),
        }
    }
}

/// A backend constructor. Returning `None` mirrors the C++ factory yielding
/// `nullptr`.
pub type TransportCtor = fn(&TransportParams) -> Option<TransportPtr>;

fn registry() -> &'static RwLock<HashMap<TransportType, TransportCtor>> {
    // Process-global, NOT thread_local (project rule); mirrors the
    // ctp-memory allocator registry pattern from MEMORY_DESIGN.md.
    static REGISTRY: OnceLock<RwLock<HashMap<TransportType, TransportCtor>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// C++ `class TransportFactory`.
///
/// Where C++ picks a backend with `#if CTP_ENABLE_*` inside the switch, this
/// keeps a runtime constructor registry: each backend module (or wrapper
/// crate) registers itself, and an unregistered type yields `None` — the same
/// answer the C++ `default: return nullptr` arm gives for a compiled-out
/// backend (divergence 9).
pub struct TransportFactory;

impl TransportFactory {
    /// Register `ctor` for `ttype`, returning the constructor it replaced.
    pub fn register(ttype: TransportType, ctor: TransportCtor) -> Option<TransportCtor> {
        let mut map = registry().write().unwrap_or_else(|e| e.into_inner());
        map.insert(ttype, ctor)
    }

    /// Remove `ttype`'s constructor, returning it if one was registered.
    pub fn unregister(ttype: TransportType) -> Option<TransportCtor> {
        let mut map = registry().write().unwrap_or_else(|e| e.into_inner());
        map.remove(&ttype)
    }

    /// True if a backend is available for `ttype`.
    pub fn is_registered(ttype: TransportType) -> bool {
        let map = registry().read().unwrap_or_else(|e| e.into_inner());
        map.contains_key(&ttype)
    }

    /// Every currently registered type (unordered).
    pub fn registered_types() -> Vec<TransportType> {
        let map = registry().read().unwrap_or_else(|e| e.into_inner());
        map.keys().copied().collect()
    }

    /// C++ `TransportFactory::Get(addr, t, mode, protocol = "", port = 0)`.
    /// Rust has no default arguments: pass `""` / `0` for the defaults.
    pub fn get(
        addr: &str,
        ttype: TransportType,
        mode: TransportMode,
        protocol: &str,
        port: i32,
    ) -> Option<TransportPtr> {
        Self::get_with_domain(addr, ttype, mode, protocol, port, "")
    }

    /// C++ `TransportFactory::Get(addr, t, mode, protocol, port, domain)`.
    /// `domain` is carried into [`TransportParams`]; no C++ ctor consumes it
    /// (divergence 10).
    pub fn get_with_domain(
        addr: &str,
        ttype: TransportType,
        mode: TransportMode,
        protocol: &str,
        port: i32,
        domain: &str,
    ) -> Option<TransportPtr> {
        let params = TransportParams::resolve(addr, ttype, mode, protocol, port, domain);
        // Copy the fn pointer out and drop the lock *before* constructing: a
        // ctor that registers another backend would otherwise deadlock.
        let ctor = {
            let map = registry().read().unwrap_or_else(|e| e.into_inner());
            map.get(&ttype).copied()
        }?;
        ctor(&params)
    }
}

// ---------------------------------------------------------------------------
// utils.h
// ---------------------------------------------------------------------------

/// C++ `utils::ParseUrl` — split `"protocol://host:port"` into `(host, port)`.
///
/// The C++ declaration has **no definition anywhere in the tree** and no
/// caller, so this defines the edges it left open (divergence 12):
///
/// - no `"://"`  → the whole input is `host:port`
/// - no `:`      → `(input, "")`
/// - empty input → `("", "")`
/// - `"[::1]:80"` → `("::1", "80")` — bracketed IPv6 is unwrapped
/// - unbracketed IPv6 is ambiguous and splits at the **last** colon
pub fn parse_url(url: &str) -> (String, String) {
    let rest = match url.find("://") {
        Some(i) => &url[i + "://".len()..],
        None => url,
    };
    // Bracketed IPv6: the colons inside the brackets are not separators.
    if let Some(after_open) = rest.strip_prefix('[') {
        if let Some(close) = after_open.find(']') {
            let host = &after_open[..close];
            let tail = &after_open[close + 1..];
            let port = tail.strip_prefix(':').unwrap_or("");
            return (host.to_string(), port.to_string());
        }
    }
    match rest.rfind(':') {
        Some(i) => (rest[..i].to_string(), rest[i + 1..].to_string()),
        None => (rest.to_string(), String::new()),
    }
}

/// C++ `utils::GetDefaultBufferSize()`.
pub const fn default_buffer_size() -> usize {
    1024
}

/// C++ `utils::GetDefaultCqSize()`.
pub const fn default_cq_size() -> usize {
    10
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    // FullPtr's halves are only constructed in tests here; the library code
    // passes FullPtr values around whole. (AllocatorId comes via super::*.)
    use ctp_memory::ShmPtr;

    #[test]
    fn recv_allocated_sentinel_distinguishes_owners() {
        assert_eq!(RECV_ALLOCATED_ID, AllocatorId::new(u32::MAX - 1, u32::MAX - 1));
        // Distinct from null: private memory is NOT transport-allocated.
        assert_ne!(RECV_ALLOCATED_ID, AllocatorId::null());

        // A buffer the transport's recv path allocated: free it.
        let mine = Bulk::new(
            FullPtr::new(0x1000, ShmPtr::new(RECV_ALLOCATED_ID, 0x1000)),
            64,
            BULK_XFER,
        );
        assert!(mine.is_recv_allocated());

        // A CTP-allocator buffer swapped in by the archive: NOT ours. Freeing
        // it would be a bad-free, not a leak — its user pointer sits inside a
        // larger malloc region.
        let theirs = Bulk::new(
            FullPtr::new(0x2000, ShmPtr::new(AllocatorId::new(1, 0), 16)),
            64,
            BULK_XFER,
        );
        assert!(!theirs.is_recv_allocated());

        // Private (stack/heap) memory: null allocator id, address as offset.
        assert!(!Bulk::new(FullPtr::from_local(0x3000), 64, BULK_XFER).is_recv_allocated());

        // Nothing to free when the local half is null, even if tagged.
        let empty = Bulk::new(
            FullPtr::new(0, ShmPtr::new(RECV_ALLOCATED_ID, 0)),
            0,
            BULK_XFER,
        );
        assert!(!empty.is_recv_allocated());
    }
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Barrier, Mutex, MutexGuard};

    // -- helpers ------------------------------------------------------------

    /// Serializes tests that mutate the process-global factory registry.
    fn registry_guard() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// A transport implementing only the required methods, so the trait
    /// defaults (divergence 13) are what gets exercised.
    struct MinimalTransport {
        ttype: TransportType,
        mode: TransportMode,
        em: Option<Arc<dyn EventManager>>,
        sent: usize,
    }

    impl MinimalTransport {
        fn new(ttype: TransportType, mode: TransportMode) -> Self {
            Self {
                ttype,
                mode,
                em: None,
                sent: 0,
            }
        }
    }

    impl Transport for MinimalTransport {
        fn transport_type(&self) -> TransportType {
            self.ttype
        }
        fn mode(&self) -> TransportMode {
            self.mode
        }
        fn expose(&mut self, ptr: FullPtr, data_size: usize, flags: u32) -> Bulk {
            Bulk::new(ptr, data_size, flags)
        }
        fn send(&mut self, meta: &mut LbmMeta, _ctx: &LbmContext) -> i32 {
            self.sent += 1;
            meta.refresh_bulk_counts();
            0
        }
        fn recv(&mut self, meta: &mut LbmMeta, _ctx: &LbmContext) -> ClientInfo {
            meta.recv = meta.send.clone();
            meta.refresh_bulk_counts();
            ClientInfo::new()
        }
    }

    /// A transport that overrides the event-manager methods, to prove the
    /// register/detach lifetime (divergence 7b).
    struct EmTransport {
        em: Option<Arc<dyn EventManager>>,
    }

    impl Transport for EmTransport {
        fn transport_type(&self) -> TransportType {
            TransportType::Socket
        }
        fn mode(&self) -> TransportMode {
            TransportMode::Server
        }
        fn expose(&mut self, ptr: FullPtr, data_size: usize, flags: u32) -> Bulk {
            Bulk::new(ptr, data_size, flags)
        }
        fn send(&mut self, _meta: &mut LbmMeta, _ctx: &LbmContext) -> i32 {
            0
        }
        fn recv(&mut self, _meta: &mut LbmMeta, _ctx: &LbmContext) -> ClientInfo {
            ClientInfo::new()
        }
        fn register_event_manager(&mut self, em: Arc<dyn EventManager>) {
            em.add_event(7, DEFAULT_READ_EVENT, None);
            self.em = Some(em);
        }
        fn unregister_event_manager(&mut self) {
            // Mirrors socket_transport.h: detach WITHOUT RemoveEvent.
            self.em = None;
        }
    }

    #[derive(Default)]
    struct CountingEm {
        added: AtomicUsize,
        removed: AtomicUsize,
        waits: AtomicUsize,
    }

    impl EventManager for CountingEm {
        fn add_event(&self, fd: i32, _events: u32, _action: Option<Arc<dyn EventAction>>) -> i32 {
            self.added.fetch_add(1, Ordering::SeqCst);
            fd
        }
        fn remove_event(&self, _fd: i32) {
            self.removed.fetch_add(1, Ordering::SeqCst);
        }
        fn wait_us(&self, _timeout_us: i32) -> i32 {
            self.waits.fetch_add(1, Ordering::SeqCst);
            0
        }
    }

    #[derive(Default)]
    struct CountingAction {
        runs: AtomicUsize,
        last_fd: AtomicUsize,
    }

    impl EventAction for CountingAction {
        fn run(&self, event: &EventInfo) {
            self.runs.fetch_add(1, Ordering::SeqCst);
            self.last_fd
                .store(event.trigger.fd as usize, Ordering::SeqCst);
        }
    }

    fn shm(off: u64) -> ShmPtr<u8> {
        ShmPtr::new(AllocatorId::new(1, 2), off)
    }

    // -- flags --------------------------------------------------------------

    #[test]
    fn bulk_flag_bit_positions_match_cpp() {
        assert_eq!(BULK_EXPOSE, 1 << 0);
        assert_eq!(BULK_XFER, 1 << 1);
        assert_ne!(BULK_EXPOSE, BULK_XFER);
        assert_eq!(LBM_SYNC, 0x1);
    }

    // -- FullPtr ------------------------------------------------------------

    #[test]
    fn full_ptr_null_is_either_half_null() {
        // C++ IsNull(): ptr_ == nullptr || shm_.IsNull() — an OR, not an AND.
        assert!(FullPtr::null().is_null());
        assert!(FullPtr::new(0, shm(64)).is_null(), "null local half");
        assert!(
            FullPtr::new(0x1000, ShmPtr::null()).is_null(),
            "null shm half"
        );
        assert!(!FullPtr::new(0x1000, shm(64)).is_null());
    }

    #[test]
    fn full_ptr_from_local_uses_address_as_offset() {
        // C++ explicit FullPtr(T*): {AllocatorId::GetNull(), (size_t)ptr}.
        let p = FullPtr::from_local(0xDEAD_BEEF);
        assert_eq!(p.ptr, 0xDEAD_BEEF);
        assert_eq!(p.shm.alloc_id, AllocatorId::null());
        assert_eq!(p.shm.off, 0xDEAD_BEEF);
        assert!(!p.is_null(), "both halves set");
    }

    #[test]
    fn full_ptr_zero_local_address_is_null_pointer() {
        // Address 0 is nullptr; from_local(0) must therefore read as null.
        assert!(FullPtr::from_local(0).is_null());
    }

    #[test]
    fn full_ptr_set_null_clears_both_halves() {
        let mut p = FullPtr::from_local(0x2000);
        p.set_null();
        assert_eq!(p, FullPtr::null());
        assert!(p.is_null());
    }

    #[test]
    fn full_ptr_offset_advances_both_halves() {
        let p = FullPtr::new(0x1000, shm(64));
        let q = p.offset_by(16);
        assert_eq!(q.ptr, 0x1010);
        assert_eq!(q.shm.off, 80);
        assert_eq!(q.shm.alloc_id, p.shm.alloc_id, "identity preserved");
        assert_eq!(q.rewind_by(16), p, "rewind is the inverse");
    }

    #[test]
    fn full_ptr_offset_by_zero_is_identity() {
        let p = FullPtr::new(0x1000, shm(64));
        assert_eq!(p.offset_by(0), p);
        assert_eq!(p.rewind_by(0), p);
    }

    #[test]
    fn full_ptr_arithmetic_wraps_instead_of_panicking() {
        // Divergence 5: C++ unsigned overflow wraps; a debug-build panic here
        // would be a behavior change, so wrapping_add/sub is the faithful pick.
        // Each half wraps independently at its own width.
        let q = FullPtr::new(usize::MAX, shm(u64::MAX - 1)).offset_by(2);
        assert_eq!(q.ptr, 1, "usize::MAX + 2");
        assert_eq!(q.shm.off, 0, "(u64::MAX - 1) + 2");
        let r = FullPtr::new(0, shm(0)).rewind_by(1);
        assert_eq!(r.ptr, usize::MAX);
        // Underflow lands on the NULL_OFFSET sentinel — the C++ does exactly
        // this too; offsetting past a segment's start is caller error either
        // way, and neither language reports it.
        assert_eq!(r.shm.off, u64::MAX);
        assert!(r.shm.is_null());
    }

    #[test]
    fn full_ptr_equality_compares_both_halves() {
        let a = FullPtr::new(0x1000, shm(64));
        assert_eq!(a, FullPtr::new(0x1000, shm(64)));
        assert_ne!(a, FullPtr::new(0x1001, shm(64)), "local half differs");
        assert_ne!(a, FullPtr::new(0x1000, shm(65)), "offset differs");
        assert_ne!(
            a,
            FullPtr::new(0x1000, ShmPtr::new(AllocatorId::new(9, 9), 64)),
            "alloc id differs"
        );
    }

    #[test]
    fn full_ptr_field_order_matches_cpp_layout() {
        // C++ FullPtr<char>: { T* ptr_; ShmPtrBase<T> shm_; } over an empty
        // base — 8 + 16 bytes, ptr first.
        assert_eq!(std::mem::size_of::<FullPtr>(), 24);
        assert_eq!(std::mem::align_of::<FullPtr>(), 8);
    }

    // -- Bulk ---------------------------------------------------------------

    #[test]
    fn bulk_default_is_null_and_zero() {
        let b = Bulk::default();
        assert!(b.data.is_null());
        assert_eq!(b.size, 0);
        assert_eq!(b.flags.bits(), 0);
        assert_eq!(b.desc, 0);
        assert_eq!(b.mr, 0);
        assert!(!b.is_xfer());
        assert!(!b.is_expose());
    }

    #[test]
    fn bulk_flag_predicates() {
        let p = FullPtr::from_local(0x1000);
        assert!(Bulk::new(p, 8, BULK_XFER).is_xfer());
        assert!(!Bulk::new(p, 8, BULK_XFER).is_expose());
        assert!(Bulk::new(p, 8, BULK_EXPOSE).is_expose());
        assert!(!Bulk::new(p, 8, BULK_EXPOSE).is_xfer());
        // Both bits set: each predicate is an Any() test, so both hold.
        let both = Bulk::new(p, 8, BULK_EXPOSE | BULK_XFER);
        assert!(both.is_xfer() && both.is_expose());
        // Neither.
        assert!(!Bulk::new(p, 8, 0).is_xfer());
        assert!(!Bulk::new(p, 8, 0).is_expose());
    }

    #[test]
    fn bulk_zero_and_max_sizes_are_carried_verbatim() {
        let p = FullPtr::from_local(0x1000);
        assert_eq!(Bulk::new(p, 0, BULK_XFER).size, 0);
        assert_eq!(Bulk::new(p, usize::MAX, BULK_XFER).size, usize::MAX);
    }

    #[test]
    fn bulk_serialized_field_contract() {
        // Divergence 4: data/desc/mr never go on the wire.
        assert_eq!(Bulk::SERIALIZED_FIELDS, ["size", "flags"]);
        assert_eq!(
            LbmMeta::SERIALIZED_FIELDS,
            ["send", "recv", "send_bulks", "recv_bulks"]
        );
    }

    // -- ClientInfo ---------------------------------------------------------

    #[test]
    fn client_info_defaults_match_cpp_initializers() {
        let ci = ClientInfo::new();
        assert_eq!(ci.rc, 0);
        assert_eq!(ci.fd, INVALID_SOCKET, "C++ fd_ = -1 means 'no socket'");
        assert!(ci.identity.is_empty());
        assert!(ci.is_ok());
    }

    #[test]
    fn client_info_failure_matches_cpp_dispatch_default_arm() {
        // C++: return ClientInfo{-1, -1, {}};
        let ci = ClientInfo::failed();
        assert_eq!((ci.rc, ci.fd), (-1, INVALID_SOCKET));
        assert!(ci.identity.is_empty());
        assert!(!ci.is_ok());
        // EAGAIN-style: a non-zero rc is not ok, and fd defaults to -1.
        assert!(!ClientInfo::err(11).is_ok());
        assert_eq!(ClientInfo::err(11).fd, INVALID_SOCKET);
    }

    #[test]
    fn client_info_default_matches_the_cpp_default_ctor() {
        // C++: `int rc = 0; int fd_ = -1; std::string identity_;` — so a
        // default-constructed ClientInfo has fd_ == -1.
        //
        // This previously derived Default (zeroing fd) and a test asserted
        // the difference rather than fixing it. fd = 0 is not "no fd", it is
        // standard input, so that gap was a bug waiting for its first caller.
        // Default is hand-written now and agrees with new().
        assert_eq!(ClientInfo::default().rc, 0);
        assert_eq!(
            ClientInfo::default().fd,
            INVALID_SOCKET,
            "C++ initializes fd_ to -1: no socket"
        );
        assert!(ClientInfo::default().identity.is_empty());
        assert_eq!(ClientInfo::default(), ClientInfo::new());

        // The dispatch's failure value stays distinct from the default.
        assert_eq!(ClientInfo::failed().rc, -1);
        assert_ne!(ClientInfo::failed(), ClientInfo::default());
    }

    // -- LbmMeta ------------------------------------------------------------

    #[test]
    fn lbm_meta_default_is_empty() {
        let m = LbmMeta::new();
        assert!(m.send.is_empty() && m.recv.is_empty());
        assert_eq!(m.send_bulks, 0);
        assert_eq!(m.recv_bulks, 0);
        assert_eq!(m.client_info, ClientInfo::default());
    }

    #[test]
    fn lbm_meta_count_xfer_counts_only_xfer_entries() {
        let p = FullPtr::from_local(0x1000);
        assert_eq!(LbmMeta::count_xfer(&[]), 0, "empty");
        let bulks = vec![
            Bulk::new(p, 1, BULK_EXPOSE),             // metadata only
            Bulk::new(p, 2, BULK_XFER),               // counts
            Bulk::new(p, 3, BULK_EXPOSE | BULK_XFER), // counts once
            Bulk::new(p, 4, 0),                       // no flags
        ];
        assert_eq!(LbmMeta::count_xfer(&bulks), 2);
    }

    #[test]
    fn lbm_meta_refresh_bulk_counts_tracks_both_vectors() {
        let p = FullPtr::from_local(0x1000);
        let mut m = LbmMeta::new();
        m.send = vec![Bulk::new(p, 1, BULK_XFER), Bulk::new(p, 2, BULK_EXPOSE)];
        m.recv = vec![Bulk::new(p, 1, BULK_XFER), Bulk::new(p, 2, BULK_XFER)];
        m.refresh_bulk_counts();
        assert_eq!(m.send_bulks, 1);
        assert_eq!(m.recv_bulks, 2);
        // Idempotent.
        m.refresh_bulk_counts();
        assert_eq!((m.send_bulks, m.recv_bulks), (1, 2));
        // Clearing the vectors zeroes the counts.
        m.send.clear();
        m.recv.clear();
        m.refresh_bulk_counts();
        assert_eq!((m.send_bulks, m.recv_bulks), (0, 0));
    }

    // -- LbmContext ---------------------------------------------------------

    #[test]
    fn lbm_context_defaults_match_cpp_initializers() {
        let c = LbmContext::new();
        assert_eq!(c.flags, 0);
        assert_eq!(c.timeout_ms, 0);
        // C++ `copy_space = nullptr` + `shm_info_ = nullptr`, fused.
        assert!(c.ring.is_none());
        assert_eq!(c.meta_buf, 0);
        assert_eq!(c.meta_buf_size, 0);
        assert!(!c.warp_parallel);
        assert_eq!(c.server_pid, 0);
        assert_eq!(c.dst_fd, -1, "C++ dst_fd_ = -1");
        assert_eq!(c.dst_offset, 0);
        assert_eq!(c.signal_pid, 0);
        assert_eq!(c.signal_tid, 0);
        assert!(c.event_manager.is_none());
        assert!(!c.is_sync());
        assert!(!c.has_timeout());
        assert!(!c.has_file_dst(), "-1 is not a file destination");
    }

    #[test]
    fn lbm_context_is_sync_reads_the_flag_bit() {
        assert!(LbmContext::with_flags(LBM_SYNC).is_sync());
        assert!(!LbmContext::with_flags(0).is_sync());
        // Other bits must not be mistaken for LBM_SYNC.
        assert!(!LbmContext::with_flags(0x2).is_sync());
        assert!(LbmContext::with_flags(0x2 | LBM_SYNC).is_sync());
        assert!(LbmContext::with_flags(u32::MAX).is_sync());
    }

    #[test]
    fn lbm_context_has_timeout_is_strictly_positive() {
        // C++ HasTimeout(): timeout_ms > 0 — 0 AND negatives mean no timeout.
        assert!(!LbmContext::with_timeout(0, 0).has_timeout());
        assert!(!LbmContext::with_timeout(0, -1).has_timeout());
        assert!(!LbmContext::with_timeout(0, i32::MIN).has_timeout());
        assert!(LbmContext::with_timeout(0, 1).has_timeout());
        assert!(LbmContext::with_timeout(0, i32::MAX).has_timeout());
    }

    #[test]
    fn lbm_context_has_file_dst_includes_fd_zero() {
        // C++ HasFileDst(): dst_fd_ >= 0 — fd 0 IS a destination.
        assert!(LbmContext::with_file_dst(0, 0, 0, 0).has_file_dst());
        assert!(LbmContext::with_file_dst(0, 0, 3, 4096).has_file_dst());
        assert!(!LbmContext::with_file_dst(0, 0, -1, 0).has_file_dst());
        assert!(!LbmContext::with_file_dst(0, 0, i32::MIN, 0).has_file_dst());
    }

    #[test]
    fn lbm_context_file_dst_ctor_keeps_other_defaults() {
        let c = LbmContext::with_file_dst(LBM_SYNC, 50, 9, 4096);
        assert!(c.is_sync());
        assert_eq!(c.timeout_ms, 50);
        assert_eq!(c.dst_fd, 9);
        assert_eq!(c.dst_offset, 4096);
        assert!(c.has_file_dst() && c.has_timeout());
        // Everything the ctor doesn't touch stays at its default.
        assert_eq!(c.server_pid, 0);
        assert!(c.ring.is_none());
        assert!(c.event_manager.is_none());
    }

    #[test]
    fn lbm_context_with_flags_leaves_timeout_zero() {
        let c = LbmContext::with_flags(LBM_SYNC);
        assert!(c.is_sync());
        assert_eq!(c.timeout_ms, 0);
        assert!(!c.has_timeout());
        assert_eq!(c.dst_fd, -1);
    }

    #[test]
    fn lbm_context_carries_an_event_manager_explicitly_not_via_tls() {
        // Divergence 7c: the C++ pulls this from IpcManager::GetTls.
        let em = Arc::new(CountingEm::default());
        let mut c = LbmContext::new();
        c.event_manager = Some(em.clone());
        assert_eq!(c.event_manager.as_ref().unwrap().wait_us(200), 0);
        assert_eq!(em.waits.load(Ordering::SeqCst), 1);
        // Cloning the context shares the manager rather than copying it.
        let c2 = c.clone();
        c2.event_manager.as_ref().unwrap().wait_us(200);
        assert_eq!(em.waits.load(Ordering::SeqCst), 2);
    }

    // -- events -------------------------------------------------------------

    #[test]
    fn default_read_event_is_platform_correct() {
        #[cfg(windows)]
        assert_eq!(DEFAULT_READ_EVENT, 0x0100, "POLLRDNORM");
        #[cfg(not(windows))]
        assert_eq!(DEFAULT_READ_EVENT, 0x001, "EPOLLIN");
    }

    #[test]
    fn event_info_dispatch_runs_the_action_with_the_event() {
        let action = Arc::new(CountingAction::default());
        let info = EventInfo::new(
            EventTrigger {
                fd: 42,
                event_id: 7,
            },
            DEFAULT_READ_EVENT,
        )
        .with_action(action.clone());
        info.dispatch();
        info.dispatch();
        assert_eq!(action.runs.load(Ordering::SeqCst), 2);
        assert_eq!(action.last_fd.load(Ordering::SeqCst), 42, "event passed in");
        assert_eq!(info.events, DEFAULT_READ_EVENT);
    }

    #[test]
    fn event_info_dispatch_without_action_is_a_noop() {
        // C++ guards on a null action_; None must not panic.
        let info = EventInfo::new(EventTrigger::default(), DEFAULT_READ_EVENT);
        info.dispatch();
        assert!(info.action.is_none());
        assert_eq!(info.trigger.fd, 0);
        assert_eq!(info.trigger.event_id, 0);
    }

    #[test]
    fn event_info_debug_does_not_require_action_debug() {
        let info = EventInfo::new(EventTrigger { fd: 1, event_id: 2 }, DEFAULT_READ_EVENT)
            .with_action(Arc::new(CountingAction::default()));
        assert!(format!("{info:?}").contains("action: true"));
    }

    // -- Transport trait defaults ------------------------------------------

    #[test]
    fn transport_mode_helpers() {
        let s = MinimalTransport::new(TransportType::Shm, TransportMode::Server);
        assert!(s.is_server() && !s.is_client());
        let c = MinimalTransport::new(TransportType::Shm, TransportMode::Client);
        assert!(c.is_client() && !c.is_server());
    }

    #[test]
    fn transport_defaults_match_cpp_switch_default_arms() {
        let mut t = MinimalTransport::new(TransportType::Shm, TransportMode::Client);
        // PollRecv: "Non-ZMQ transports return 0".
        assert_eq!(t.poll_recv(1000), 0);
        assert_eq!(t.poll_recv(0), 0);
        assert_eq!(t.poll_recv(-1), 0);
        // GetBoundPort: "Returns 0 for clients / non-TCP / non-ZMQ".
        assert_eq!(t.bound_port(), 0);
        // GetAddress default arm: "".
        assert!(t.address().is_empty());
        // IsServerAlive default arm: false.
        assert!(!t.is_server_alive(&LbmContext::new()));
        // No-op arms must not panic or disturb state.
        let mut meta = LbmMeta::new();
        t.clear_recv_handles(&mut meta);
        t.register_event_manager(Arc::new(CountingEm::default()));
        t.unregister_event_manager();
        assert!(t.em.is_none(), "default register is a no-op (the SHM arm)");
        assert!(meta.send.is_empty());
        assert_eq!(t.sent, 0);
    }

    #[test]
    fn transport_round_trip_through_the_trait_object() {
        // Exercises the vtable dispatch that replaces the C++ type_ switch.
        let mut t: TransportPtr = Box::new(MinimalTransport::new(
            TransportType::Socket,
            TransportMode::Server,
        ));
        assert_eq!(t.transport_type(), TransportType::Socket);
        assert_eq!(t.mode(), TransportMode::Server);

        let bulk = t.expose(FullPtr::from_local(0x4000), 128, BULK_XFER);
        assert_eq!(bulk.size, 128);
        assert!(bulk.is_xfer());

        let mut meta = LbmMeta::new();
        meta.send.push(bulk);
        meta.send
            .push(Bulk::new(FullPtr::from_local(0x8000), 8, BULK_EXPOSE));
        assert_eq!(t.send(&mut meta, &LbmContext::new()), 0);
        assert_eq!(meta.send_bulks, 1, "only the XFER entry counts");

        let ci = t.recv(&mut meta, &LbmContext::new());
        assert!(ci.is_ok());
        assert_eq!(meta.recv.len(), 2);
        assert_eq!(meta.recv_bulks, 1);
    }

    #[test]
    fn event_manager_register_then_detach_drops_the_handle() {
        // Divergence 7b: UnregisterEventManager detaches WITHOUT RemoveEvent,
        // and the Arc makes the caller's subsequent drop safe.
        let em = Arc::new(CountingEm::default());
        let mut t = EmTransport { em: None };
        t.register_event_manager(em.clone());
        assert_eq!(em.added.load(Ordering::SeqCst), 1);
        assert_eq!(Arc::strong_count(&em), 2, "transport holds a reference");

        t.unregister_event_manager();
        assert_eq!(Arc::strong_count(&em), 1, "detached");
        assert_eq!(em.removed.load(Ordering::SeqCst), 0, "no RemoveEvent");
    }

    #[test]
    fn event_manager_wait_is_microseconds() {
        // shm_transport.h passes 200 for a "200us bounded re-check".
        let em = CountingEm::default();
        assert_eq!(em.wait_us(200), 0);
        assert_eq!(em.wait_us(-1), 0, "negative = wait forever");
        assert_eq!(em.waits.load(Ordering::SeqCst), 2);
        assert_eq!(em.add_event(5, DEFAULT_READ_EVENT, None), 5);
        em.remove_event(5);
        assert_eq!(em.added.load(Ordering::SeqCst), 1);
        assert_eq!(em.removed.load(Ordering::SeqCst), 1);
    }

    // -- enums --------------------------------------------------------------

    #[test]
    fn enum_discriminants_match_cpp() {
        assert_eq!(TransportType::ZeroMq as i32, 0);
        assert_eq!(TransportType::Socket as i32, 1);
        assert_eq!(TransportType::Shm as i32, 2);
        assert_eq!(TransportType::Nixl as i32, 3);
        assert_eq!(TransportType::Thallium as i32, 4);
        assert_eq!(TransportMode::Client as i32, 0);
        assert_eq!(TransportMode::Server as i32, 1);
    }

    // -- factory: parameter resolution --------------------------------------

    #[test]
    fn resolve_zmq_defaults() {
        let p =
            TransportParams::resolve("h", TransportType::ZeroMq, TransportMode::Client, "", 0, "");
        assert_eq!(p.protocol, "tcp");
        assert_eq!(p.port, DEFAULT_ZMQ_PORT);
        assert_eq!(p.port, 8192);
        assert_eq!(p.addr, "h");
        assert_eq!(p.mode, TransportMode::Client);
    }

    #[test]
    fn resolve_socket_defaults() {
        let p =
            TransportParams::resolve("h", TransportType::Socket, TransportMode::Server, "", 0, "");
        assert_eq!(p.protocol, "tcp");
        assert_eq!(p.port, DEFAULT_SOCKET_PORT);
        assert_eq!(p.port, 8193, "socket's default differs from ZMQ's");
    }

    #[test]
    fn resolve_thallium_defaults_protocol_but_not_port() {
        // The C++ ThalliumTransport ctor takes `port` unmodified — a 0 stays 0.
        let p = TransportParams::resolve(
            "h",
            TransportType::Thallium,
            TransportMode::Client,
            "",
            0,
            "",
        );
        assert_eq!(p.protocol, "tcp");
        assert_eq!(p.port, 0, "thallium port is NOT defaulted");
    }

    #[test]
    fn resolve_shm_and_nixl_default_nothing() {
        // ShmTransport(mode) / NixlTransport(mode, addr) take no protocol/port.
        for ttype in [TransportType::Shm, TransportType::Nixl] {
            let p = TransportParams::resolve("h", ttype, TransportMode::Client, "", 0, "");
            assert_eq!(p.protocol, "", "no protocol defaulting for {ttype:?}");
            assert_eq!(p.port, 0, "no port defaulting for {ttype:?}");
        }
    }

    #[test]
    fn resolve_preserves_explicit_values() {
        for ttype in [
            TransportType::ZeroMq,
            TransportType::Socket,
            TransportType::Thallium,
        ] {
            let p = TransportParams::resolve("h", ttype, TransportMode::Client, "ib", 5555, "");
            assert_eq!(p.protocol, "ib");
            assert_eq!(p.port, 5555);
        }
    }

    #[test]
    fn resolve_passes_non_zero_ports_through_including_negatives() {
        // C++ tests `port == 0`, not `port <= 0`.
        let p =
            TransportParams::resolve("h", TransportType::ZeroMq, TransportMode::Client, "", -1, "");
        assert_eq!(p.port, -1);
        let p = TransportParams::resolve(
            "h",
            TransportType::Socket,
            TransportMode::Client,
            "",
            i32::MAX,
            "",
        );
        assert_eq!(p.port, i32::MAX);
    }

    #[test]
    fn resolve_carries_addr_and_domain_verbatim() {
        let p = TransportParams::resolve(
            "node7",
            TransportType::ZeroMq,
            TransportMode::Server,
            "tcp",
            1,
            "mlx5_0",
        );
        assert_eq!(p.addr, "node7");
        // Divergence 10: the C++ overload does `(void)domain`; we carry it.
        assert_eq!(p.domain, "mlx5_0");
    }

    #[test]
    fn resolve_empty_addr_is_allowed() {
        let p = TransportParams::resolve("", TransportType::Shm, TransportMode::Server, "", 0, "");
        assert!(p.addr.is_empty());
        assert_eq!(p.ttype, TransportType::Shm);
    }

    // -- factory: registry --------------------------------------------------

    fn make_minimal(params: &TransportParams) -> Option<TransportPtr> {
        Some(Box::new(MinimalTransport::new(params.ttype, params.mode)))
    }

    fn make_nothing(_params: &TransportParams) -> Option<TransportPtr> {
        None
    }

    #[test]
    fn factory_unregistered_type_returns_none() {
        let _g = registry_guard();
        TransportFactory::unregister(TransportType::Nixl);
        assert!(!TransportFactory::is_registered(TransportType::Nixl));
        // C++: `default: return nullptr` for a compiled-out backend.
        assert!(
            TransportFactory::get("h", TransportType::Nixl, TransportMode::Client, "", 0).is_none()
        );
    }

    #[test]
    fn factory_registered_ctor_builds_the_transport() {
        let _g = registry_guard();
        TransportFactory::register(TransportType::Shm, make_minimal);

        let t = TransportFactory::get("h", TransportType::Shm, TransportMode::Server, "", 0)
            .expect("registered");
        assert_eq!(t.transport_type(), TransportType::Shm);
        assert!(t.is_server());

        TransportFactory::unregister(TransportType::Shm);
    }

    #[test]
    fn factory_ctor_returning_none_propagates() {
        let _g = registry_guard();
        TransportFactory::register(TransportType::Socket, make_nothing);
        assert!(
            TransportFactory::get("h", TransportType::Socket, TransportMode::Client, "", 0)
                .is_none()
        );
        TransportFactory::unregister(TransportType::Socket);
    }

    #[test]
    fn factory_register_replaces_and_reports_the_previous_ctor() {
        let _g = registry_guard();
        TransportFactory::unregister(TransportType::Thallium);
        assert!(TransportFactory::register(TransportType::Thallium, make_minimal).is_none());
        // Replacing returns the old ctor.
        assert!(TransportFactory::register(TransportType::Thallium, make_nothing).is_some());
        assert!(
            TransportFactory::get("h", TransportType::Thallium, TransportMode::Client, "", 0)
                .is_none(),
            "the replacement is what runs"
        );
        assert!(TransportFactory::is_registered(TransportType::Thallium));
        assert!(TransportFactory::registered_types().contains(&TransportType::Thallium));

        assert!(TransportFactory::unregister(TransportType::Thallium).is_some());
        assert!(!TransportFactory::is_registered(TransportType::Thallium));
        assert!(
            TransportFactory::unregister(TransportType::Thallium).is_none(),
            "idempotent"
        );
    }

    #[test]
    fn factory_passes_resolved_params_to_the_ctor() {
        let _g = registry_guard();
        static SEEN: OnceLock<Mutex<Option<TransportParams>>> = OnceLock::new();
        fn recording(params: &TransportParams) -> Option<TransportPtr> {
            *SEEN
                .get_or_init(|| Mutex::new(None))
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(params.clone());
            Some(Box::new(MinimalTransport::new(params.ttype, params.mode)))
        }

        TransportFactory::register(TransportType::ZeroMq, recording);
        // Ask for the defaults; the ctor must see them already resolved.
        TransportFactory::get_with_domain(
            "n1",
            TransportType::ZeroMq,
            TransportMode::Server,
            "",
            0,
            "mlx5_0",
        )
        .expect("registered");

        let seen = SEEN
            .get_or_init(|| Mutex::new(None))
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
            .expect("ctor ran");
        assert_eq!(seen.addr, "n1");
        assert_eq!(seen.protocol, "tcp");
        assert_eq!(seen.port, 8192);
        assert_eq!(seen.mode, TransportMode::Server);
        assert_eq!(seen.domain, "mlx5_0");

        TransportFactory::unregister(TransportType::ZeroMq);
    }

    #[test]
    fn factory_get_matches_get_with_domain_apart_from_the_domain() {
        // The two C++ overloads are identical apart from `(void)domain`.
        let a =
            TransportParams::resolve("h", TransportType::Socket, TransportMode::Client, "", 0, "");
        let b =
            TransportParams::resolve("h", TransportType::Socket, TransportMode::Client, "", 0, "x");
        assert_eq!(a.protocol, b.protocol);
        assert_eq!(a.port, b.port);
        assert_ne!(a.domain, b.domain, "domain is the only difference");
    }

    #[test]
    fn factory_get_is_safe_under_concurrent_use() {
        let _g = registry_guard();
        TransportFactory::register(TransportType::Shm, make_minimal);

        const THREADS: usize = 8;
        let barrier = Arc::new(Barrier::new(THREADS));
        let built = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::with_capacity(THREADS);
        for i in 0..THREADS {
            let barrier = barrier.clone();
            let built = built.clone();
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                // Half construct; half hammer the registry's read path.
                if i % 2 == 0 {
                    let t =
                        TransportFactory::get("h", TransportType::Shm, TransportMode::Client, "", 0)
                            .expect("registered");
                    assert_eq!(t.transport_type(), TransportType::Shm);
                    built.fetch_add(1, Ordering::SeqCst);
                } else {
                    assert!(TransportFactory::is_registered(TransportType::Shm));
                    let _ = TransportFactory::registered_types();
                }
            }));
        }
        for h in handles {
            h.join().expect("no thread panicked");
        }
        assert_eq!(built.load(Ordering::SeqCst), THREADS / 2);

        TransportFactory::unregister(TransportType::Shm);
    }

    #[test]
    fn transport_ptr_is_send() {
        // Divergence 15: the C++ moves TransportPtr onto recv threads.
        fn assert_send<T: Send>() {}
        assert_send::<TransportPtr>();
        let t: TransportPtr = Box::new(MinimalTransport::new(
            TransportType::Shm,
            TransportMode::Client,
        ));
        let ttype = std::thread::spawn(move || t.transport_type())
            .join()
            .expect("moved across threads");
        assert_eq!(ttype, TransportType::Shm);
    }

    // -- utils --------------------------------------------------------------

    #[test]
    fn parse_url_splits_scheme_host_port() {
        assert_eq!(
            parse_url("tcp://127.0.0.1:8192"),
            ("127.0.0.1".to_string(), "8192".to_string())
        );
        assert_eq!(
            parse_url("verbs://node7:1234"),
            ("node7".to_string(), "1234".to_string())
        );
    }

    #[test]
    fn parse_url_without_scheme() {
        assert_eq!(
            parse_url("localhost:8080"),
            ("localhost".to_string(), "8080".to_string())
        );
    }

    #[test]
    fn parse_url_without_port() {
        assert_eq!(parse_url("tcp://host"), ("host".to_string(), String::new()));
        assert_eq!(parse_url("host"), ("host".to_string(), String::new()));
    }

    #[test]
    fn parse_url_edge_cases() {
        assert_eq!(parse_url(""), (String::new(), String::new()));
        assert_eq!(parse_url("tcp://"), (String::new(), String::new()));
        // Trailing colon: empty port, not a panic.
        assert_eq!(parse_url("host:"), ("host".to_string(), String::new()));
        // Leading colon: empty host.
        assert_eq!(parse_url(":8080"), (String::new(), "8080".to_string()));
        // Scheme with no host.
        assert_eq!(parse_url("://:80"), (String::new(), "80".to_string()));
    }

    #[test]
    fn parse_url_ipv6_forms() {
        assert_eq!(
            parse_url("tcp://[::1]:8192"),
            ("::1".to_string(), "8192".to_string())
        );
        assert_eq!(parse_url("[::1]"), ("::1".to_string(), String::new()));
        assert_eq!(
            parse_url("[fe80::1%eth0]:80"),
            ("fe80::1%eth0".to_string(), "80".to_string())
        );
        // Unterminated bracket falls back to the last-colon rule.
        assert_eq!(parse_url("[::1:80"), ("[::1".to_string(), "80".to_string()));
        // Unbracketed IPv6 is ambiguous: split at the LAST colon.
        assert_eq!(
            parse_url("::1:8192"),
            ("::1".to_string(), "8192".to_string())
        );
    }

    #[test]
    fn parse_url_only_the_first_scheme_separator_is_stripped() {
        // A path-ish tail keeps its "://" — only the leading scheme goes.
        assert_eq!(
            parse_url("tcp://host/a://b"),
            ("host/a".to_string(), "//b".to_string()),
            "the tail's colon is still the last-colon split point"
        );
        // No colon after the scheme: everything is host.
        assert_eq!(
            parse_url("tcp://host/path"),
            ("host/path".to_string(), String::new())
        );
    }

    #[test]
    fn util_constants_match_cpp() {
        assert_eq!(default_buffer_size(), 1024);
        assert_eq!(default_cq_size(), 10);
        // constexpr in C++ — usable in a const context here too.
        const BUF: usize = default_buffer_size();
        assert_eq!(BUF, 1024);
    }
}
