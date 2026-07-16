// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).

//! Shared-memory lightbeam transports — Rust port of
//! `clio_ctp/lightbeam/shm_transport.h` (the per-task SPSC ring transport) and
//! `clio_ctp/lightbeam/shm_mpsc_transport.h` (the named multi-producer /
//! single-consumer segment transport, issue #642).
//!
//! Both backends move bytes through a ring carved out of a shared-memory
//! segment obtained from [`ctp_memory::SharedMemBackend`] (MEMORY_DESIGN.md):
//! in-segment control state is `#[repr(C)]`, has no `Drop`, stores offsets
//! rather than pointers, and every cross-process handshake is an in-segment
//! atomic. The two transports differ in shape:
//!
//! * [`ShmTransport`] — one SPSC ring per caller ([`ShmTransferInfo`] +
//!   copy space), byte-streamed, chunked around the ring's wraparound.
//! * [`ShmMpscTransport`] — ONE named segment; many producers `send_bytes`
//!   into fixed [`SHM_MPSC_CHUNK_SIZE`] slots, a single consumer `recv_bytes`
//!   de-multiplexes them by connection id.
//!
//! # C++ → Rust name mapping
//!
//! ## `shm_transport.h`
//!
//! | C++ | Rust |
//! |---|---|
//! | `ctp::lbm::ShmTransferInfo` | [`ShmTransferInfo`] |
//! | `ShmTransferInfo::total_written_` / `total_read_` / `copy_space_size_` | `ShmTransferInfo::total_written` / `total_read` / `copy_space_size` |
//! | `ctp::ipc::atomic<T>::load_system()` / `store_system()` | [`load_system`] / [`store_system`] |
//! | `LbmContext::copy_space` + `LbmContext::shm_info_` | [`ShmRing`] (fused handle), held by `LbmContext::ring` |
//! | *(caller-owned `FutureShm` copy space)* | [`ShmRingBuffer`] (heap) / [`ShmRingSegment`] (named SHM, via `ctp-memory`) |
//! | `ctp::lbm::ShmTransport` | [`ShmTransport`] |
//! | `ShmTransport::Expose` | [`ShmTransport::expose`] |
//! | `ShmTransport::GetAddress` | [`ShmTransport::address`] |
//! | `ShmTransport::IsServerAlive` | [`ShmTransport::is_server_alive`] |
//! | `ShmTransport::ClearRecvHandles` | [`ShmTransport::clear_recv_handles`] |
//! | `ShmTransport::Send` (static template) | [`ShmTransport::send`] |
//! | `ShmTransport::Recv` (static template) | [`ShmTransport::recv`] |
//! | `ShmTransport::WaitForTransferStart` | [`wait_for_transfer_start`] |
//! | `ShmTransport::RecvBulksImpl` | [`recv_bulks_impl`] |
//! | `ShmTransport::WriteTransfer` / `ReadTransfer` | [`write_transfer`] / [`read_transfer`] |
//! | `ShmTransport::Min3` | `min3` (private) |
//! | `ShmTransport::MemCopy` | `ptr::copy_nonoverlapping` (inline) |
//! | `ShmTransport::WarpMemCpy` | *(not ported — divergence 1)* |
//! | `WriteTransferDevice` / `ReadTransferDevice` / `RecvBulksImplDevice` | *(not ported — divergence 1)* |
//!
//! ## `shm_mpsc_transport.h`
//!
//! | C++ | Rust |
//! |---|---|
//! | `kShmMpscDefaultSegmentSize` | [`SHM_MPSC_DEFAULT_SEGMENT_SIZE`] |
//! | `kShmMpscChunkSize` | [`SHM_MPSC_CHUNK_SIZE`] |
//! | `kShmMpscMaxXfers` | [`SHM_MPSC_MAX_XFERS`] |
//! | `SHM_MPSC_DONTWAIT` | [`SHM_MPSC_DONTWAIT`] |
//! | `kShmMpscDeadXferUs` / `kShmMpscLivenessUs` | `SHM_MPSC_DEAD_XFER_US` / `SHM_MPSC_LIVENESS_US` (private) |
//! | `ShmXferHeader` | [`ShmXferHeader`] |
//! | `ShmTransportHeader` | [`ShmTransportHeader`] |
//! | `ShmMpscTransport` | [`ShmMpscTransport`] |
//! | `ShmMpscTransport::backend_` (`ctp::ipc::PosixShmMmap`) | `ShmMpscTransport::backend` ([`ctp_memory::SharedMemBackend`]) |
//! | `ServerInit` / `ClientInit` / `Shutdown` | [`ShmMpscTransport::server_init`] / [`ShmMpscTransport::client_init`] / [`ShmMpscTransport::shutdown`] |
//! | `SendBytes` / `RecvBytes` | [`ShmMpscTransport::send_bytes`] / [`ShmMpscTransport::recv_bytes`] |
//! | `ShmMpscTransport::Send` / `Recv` / `Expose` | [`ShmMpscTransport::send`] / [`ShmMpscTransport::recv`] / [`ShmMpscTransport::expose`] |
//! | `AppendRaw` / `ReadRaw` | `Vec::extend_from_slice` / `read_raw` (private) |
//! | `ComputeSlots` / `WaitSlotFree` / `WaitChunkReady` / `ServerAlive` | `compute_slots` / `wait_slot_free` / `wait_chunk_ready` / `server_alive` (private) |
//! | `kBackendHeaderSizeApprox()` | *(dropped — divergence 5)* |
//! | `RecvConn` | `RecvConn` (private) |
//! | `send_mu_` (`std::mutex`) | `send_mu` (`std::sync::Mutex<()>`) |
//! | `recv_conns_` (`std::unordered_map`) | `recv_conns` (`HashMap`) |
//!
//! ## Shared lightbeam surface (defined here — see divergence 2)
//!
//! | C++ (`lightbeam.h`) | Rust |
//! |---|---|
//! | `BULK_EXPOSE` / `BULK_XFER` | [`BULK_EXPOSE`] / [`BULK_XFER`] |
//! | `ctp::lbm::Bulk` + `ctp::ipc::FullPtr<char>` | [`Bulk`] (`data: Vec<u8>` + `shm: ShmPtr<u8>`) |
//! | `ctp::lbm::ClientInfo` | [`ClientInfo`] |
//! | `ctp::lbm::LbmMeta<AllocT>` | [`LbmMeta`] |
//! | `ctp::lbm::LbmContext` / `LBM_SYNC` | [`LbmContext`] / [`LBM_SYNC`] |
//! | `ctp::lbm::TransportType` / `TransportMode` | [`TransportType`] / [`TransportMode`] |
//! | `ctp::ipc::LocalSerialize<CharVec>` over `LbmMeta` | [`serialize_meta`] / [`deserialize_meta`] |
//! | `CTP_THREAD_MODEL->Yield()` | `ctp_thread::thread_model::thread_model().yield_now()` |
//! | `ctp::Timepoint` / `GetUsecFromStart` | `std::time::Instant` / `Instant::elapsed` |
//!
//! # Units
//!
//! Per the migration rules API timings are **milliseconds**: [`LbmContext`]'s
//! `timeout_ms` is ms, matching the C++ field of the same name. The C++
//! *internal* spin constants are microseconds and stay that way
//! (`SHM_MPSC_DEAD_XFER_US` = 1 s, `SHM_MPSC_LIVENESS_US` = 50 ms, the 10 µs
//! busy-wait in [`wait_for_transfer_start`]).
//!
//! # Semantic divergences from the C++
//!
//! 1. **No device (GPU) path.** `WarpMemCpy`, `WriteTransferDevice`,
//!    `ReadTransferDevice`, `RecvBulksImplDevice` and the `load_device()` /
//!    `threadfence()` device-scope atomics exist only in `CTP_IS_GPU` builds.
//!    Per MIGRATION.md device code stays in `ctp-gpu`; only the host pass is
//!    ported. [`load_system`] / [`store_system`] keep the C++ host lowering
//!    exactly (relaxed volatile load; seq_cst fence + store + seq_cst fence).
//! 2. **The `lightbeam.h` common surface is defined locally — TO RECONCILE.**
//!    `transport.rs` was a stub when this module was written, so [`Bulk`],
//!    [`ClientInfo`], [`LbmMeta`], [`LbmContext`], [`TransportType`],
//!    [`TransportMode`] and the `BULK_*` flags are declared here. It has since
//!    landed (parallel agent) with its own copies, so the crate now carries two
//!    parallel surfaces that compile independently but do not interoperate.
//!    The deltas a reconciliation must resolve, in the order they bite:
//!    * `transport::Bulk::data` is a `FullPtr { ptr: usize, shm }` — an opaque
//!      address that module never dereferences — whereas [`Bulk`] here owns its
//!      private payload as a `Vec<u8>` (divergence 3), because this module is
//!      the one that must *allocate* received private memory. Adopting
//!      `transport::Bulk` verbatim reintroduces the C++ malloc/`ClearRecvHandles`
//!      lifetime; the recommended landing is to keep the owned buffer and let
//!      `FullPtr::ptr` be derived from it where a raw address is genuinely
//!      needed (RDMA/NIXL), not the other way round.
//!    * `transport::LbmContext` carries `copy_space: usize` + `shm_info: usize`
//!      as opaque addresses; [`ShmRing`] fuses that pair. The bridge is one
//!      `unsafe ShmRing::from_raw(ctx.shm_info as *const _, ctx.copy_space as
//!      *mut _)` at the boundary, guarded by its documented SPSC contract.
//!    * `transport::Transport` is a trait with `expose`/`send`/`recv`/`address`;
//!      [`ShmTransport`]'s equivalents are inherent (C++ derives from
//!      `Transport` only under `CTP_IS_HOST`), and its `send`/`recv` are
//!      associated fns because the C++ ones are `static`. Implementing the
//!      trait needs `&mut self` wrappers that forward to them.
//!    * `transport::EventManager` exists as a trait, so divergence 7's
//!      signal/wait path becomes implementable at that point.
//! 3. **`Bulk` owns its private payload.** C++ `FullPtr<char>` carries a raw
//!    `char* ptr_` + `ShmPtr shm_`; received private memory is `std::malloc`ed
//!    and later released by `ClearRecvHandles` (guarded by `!bulk.desc`). Here
//!    the process-local payload is a `Vec<u8>` (`Bulk::data`), so receive
//!    buffers free themselves; [`ShmTransport::clear_recv_handles`] is retained
//!    for API parity and just releases them eagerly. Consequently:
//!    (a) the C++ `recv[i].data.shm_.off_ = (size_t)buf` raw-address stuffing is
//!    NOT reproduced — an allocated private bulk keeps `shm.off == NULL_OFFSET`
//!    (a local address inside a `ShmPtr` violates MEMORY_DESIGN pillar 3);
//!    (b) `Bulk::desc` / `Bulk::mr` (RDMA registration handles, unused by the
//!    SHM path) are omitted.
//! 4. **`LbmMeta` is not allocator-parameterised.** The C++ templates on
//!    `MetaT::allocator_type` so bulk vectors can live in GPU-visible memory
//!    (`CTP_MALLOC` on host). Metadata/framing buffers here are host-private and
//!    use the Rust global allocator; `ctp-memory` allocators address *segment*
//!    memory, which is not what `meta.alloc_` is for. `Send` therefore also
//!    drops `meta_buf.reserve(copy_space_size)` (a pure pre-sizing hint).
//! 5. **Segment sizing is exact.** `ShmMpscTransport::ServerInit` pads its
//!    request by `kBackendHeaderSizeApprox()` (64 KB) and clamps against
//!    `PosixShmMmap::data_capacity_` because that backend reserves a header page
//!    and enforces a ≥1 MB minimum. `ctp_memory::SharedMemBackend` maps exactly
//!    what is asked for, so the padding, the `data_capacity_` clamp and the
//!    `cap_ < kShmMpscChunkSize` bump collapse into one up-front guarantee:
//!    `xfer_space >= size_of::<ShmTransportHeader>() + SHM_MPSC_CHUNK_SIZE`,
//!    which is what those clauses existed to ensure (the C++ leans on the ≥1 MB
//!    minimum for it; without the bump a sub-chunk ring would let `SendBytes`
//!    write past `cap_`).
//! 6. **`ClientInit` probes before mapping.** C++ `shm_attach(name)` learns the
//!    size from the backend's own header page. `SharedMemBackend::open` needs a
//!    size, so [`ShmMpscTransport::client_init`] maps
//!    `size_of::<ShmTransportHeader>()` bytes first, reads `max_capacity_`, then
//!    re-opens the full mapping; [`ShmRingSegment::open`] does the same. Both
//!    are coherent views of one kernel object; the probe is dropped before the
//!    real map.
//! 7. **No `EventManager` signalling.** The host `Send` fires
//!    `EventManager::Signal(ctx.signal_pid_, ctx.signal_tid_)` after the
//!    metadata write and `WaitForTransferStart` sleeps on
//!    `ctx.event_manager_->Wait(200)`. `event_manager.h` is not ported (no crate
//!    owns it yet), so this port takes the C++ **`event_manager_ == nullptr`
//!    branch verbatim**: a 10 µs busy-wait, then a `Yield()` loop bounded by
//!    `ctx.timeout_ms` → `EAGAIN`. Semantics (including the timeout) are
//!    preserved; only wake latency changes. `LbmContext::signal_pid` /
//!    `signal_tid` are retained as inert fields.
//! 8. **No PID liveness probe.** `IsServerAlive` / `ShmMpscTransport::ServerAlive`
//!    call `ctp::SystemInfo::IsProcessAlive` under `#ifndef _WIN32`;
//!    `ctp-introspect` is not a dependency of this crate (Cargo.toml may not be
//!    edited), so both return `true` unconditionally — exactly the C++ behaviour
//!    on Windows. `send_bytes`'s `-EPIPE` path is therefore unreachable here; it
//!    is kept so wiring `IsProcessAlive` in later is a one-line change.
//! 9. **Ring geometry and descriptors are validated, not assumed.** C++
//!    `WriteTransfer` on a zero-sized ring spins forever (`space` is always 0)
//!    and would then `%` by zero; [`ShmTransport::send`] / [`ShmTransport::recv`]
//!    return `EINVAL` for a missing or zero-capacity ring. A send bulk whose
//!    `data` is shorter than its declared `size` is rejected with `EINVAL`
//!    **before any bytes are written** (C++ reads out of bounds). And
//!    `recv_bytes` drops a chunk whose `xfer_off_`/`xfer_size_` fall outside the
//!    ring (C++ trusts the producer; in Rust that read would be UB).
//! 10. **`send_bytes` rejects messages > `u32::MAX`.** `ShmXferHeader::rem_size_`
//!     is a `u32`; C++ truncates `size` into it while looping over the full
//!     `size_t`, so a >4 GB message silently corrupts the consumer's reassembly.
//!     This port returns `-EINVAL` up front.
//! 11. **Descriptor fields are relaxed atomics.** C++ `ShmXferHeader`'s
//!     `conn_id_`/`xfer_*`/`rem_*` and `ShmTransportHeader`'s `pid_`/`tid_`/
//!     `max_capacity_` are plain fields published across processes by the
//!     `ready_` store; plain cross-process reads/writes are a data race in Rust
//!     (UB), so they are `Atomic{U32,U64,I32}` accessed `Relaxed` and still
//!     ordered by the `ready_` store/load. Same size, align and offsets → the C++
//!     layout is unchanged (asserted in tests). Other atomics keep the
//!     `ctp::ipc::atomic` default of `SeqCst`.
//! 12. **No `ShmSafe` impls for the control structs.** MEMORY_DESIGN's `ShmSafe`
//!     requires `Copy`; [`ShmTransferInfo`], [`ShmXferHeader`] and
//!     [`ShmTransportHeader`] contain atomics (not `Copy`) and are mutated in
//!     place, never copied. They honour the substantive rules: `repr(C)`, no
//!     `Drop`, no absolute pointers. [`ShmPtr`] values crossing the wire are
//!     moved as bytes in the frozen `(major, minor, off)` ABI order rather than
//!     by `memcpy` of the C++ struct.
//! 13. **`serialize_meta` is a hand-rolled `LocalSerialize`.** `ctp-serialize`
//!     is not a dependency, so the archive is open-coded. The wire format matches
//!     `local_serialize.h` for this type: `size_t` (u64) vector length prefixes,
//!     per-`Bulk` `size` (u64) then `flags` (u32), in the C++ member order
//!     `(send, recv, send_bulks, recv_bulks)`, native endianness. A truncated
//!     frame yields `EIO` rather than C++'s unchecked read.
//! 14. **Rings/counters use wrapping arithmetic.** C++ `size_t` differences wrap;
//!     Rust would panic in debug builds, so `wrapping_sub` is used at the same
//!     spots (`total_written - total_read`, `xfer_id - xfer_id_head`).
//! 15. **`errno` values are literals.** `libc` is not a dependency of this crate;
//!     `EAGAIN`/`EIO`/`EPIPE`/`EINVAL` are declared here. The values agree on
//!     Linux, macOS and MSVC.

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::collections::HashMap;
use std::io;
use std::sync::atomic::{fence, AtomicBool, AtomicI32, AtomicU32, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use ctp_memory::{AllocatorId, SharedMemBackend, ShmPtr, NULL_OFFSET};
use ctp_thread::thread_model::{thread_model, ThreadModel};
use ctp_types::Bitfield32;

// ---------------------------------------------------------------------------
// errno (divergence 15)
// ---------------------------------------------------------------------------

/// `EIO` — malformed frame.
pub const EIO: i32 = 5;
/// `EAGAIN` — nothing available (deadline hit, or `SHM_MPSC_DONTWAIT`).
pub const EAGAIN: i32 = 11;
/// `EINVAL` — invalid ring geometry or bulk descriptor (divergences 9, 10).
pub const EINVAL: i32 = 22;
/// `EPIPE` — the consumer process died mid-transfer.
pub const EPIPE: i32 = 32;

// ---------------------------------------------------------------------------
// lightbeam.h surface (divergence 2 — move to transport.rs when it lands)
// ---------------------------------------------------------------------------

/// `BULK_EXPOSE`: bulk metadata is sent, no data transfer.
pub const BULK_EXPOSE: u32 = ctp_types::bit_opt(0);
/// `BULK_XFER`: bulk is marked for data transmission.
pub const BULK_XFER: u32 = ctp_types::bit_opt(1);

/// `LBM_SYNC` — a no-op flag, retained exactly as in C++ (Send is always
/// synchronous).
pub const LBM_SYNC: u32 = 0x1;

/// C++ `ctp::lbm::TransportType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportType {
    ZeroMq,
    Socket,
    Shm,
    Nixl,
    Thallium,
}

/// C++ `ctp::lbm::TransportMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportMode {
    Client,
    Server,
}

/// C++ `ctp::lbm::Bulk` fused with `ctp::ipc::FullPtr<char>` (divergence 3).
///
/// `data` is the process-local payload (empty when the bulk is shared-memory
/// backed); `shm` is the cross-process `(alloc_id, off)` pointer — a null
/// `alloc_id` means "private memory", which is what makes `send` stream the
/// bytes instead of passing the pointer through.
#[derive(Debug, Clone)]
pub struct Bulk {
    pub data: Vec<u8>,
    pub shm: ShmPtr<u8>,
    pub size: usize,
    pub flags: Bitfield32,
}

impl Default for Bulk {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            shm: ShmPtr::null(),
            size: 0,
            flags: Bitfield32::default(),
        }
    }
}

/// C++ `ctp::lbm::ClientInfo`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientInfo {
    /// 0 = success, `EAGAIN` = no data, …
    pub rc: i32,
    /// Socket fd (socket transport, server mode).
    pub fd: i32,
    /// ZMQ identity (ZMQ transport, server mode).
    pub identity: String,
}

impl Default for ClientInfo {
    fn default() -> Self {
        Self {
            rc: 0,
            fd: -1,
            identity: String::new(),
        }
    }
}

/// C++ `ctp::lbm::LbmMeta<AllocT>` (divergence 4: no allocator parameter).
#[derive(Debug, Default, Clone)]
pub struct LbmMeta {
    /// Sender's bulk descriptors (`BULK_EXPOSE` or `BULK_XFER`).
    pub send: Vec<Bulk>,
    /// Receiver's bulk descriptors (copy of `send` with local payloads).
    pub recv: Vec<Bulk>,
    /// Count of `BULK_XFER` entries in `send`.
    pub send_bulks: usize,
    /// Count of `BULK_XFER` entries in `recv`.
    pub recv_bulks: usize,
    /// Client routing info (not serialized).
    pub client_info: ClientInfo,
}

impl LbmMeta {
    pub fn new() -> Self {
        Self::default()
    }
}

/// C++ `ctp::lbm::LbmContext`. Fields meaningless to the SHM path (`meta_buf_`,
/// `warp_parallel_`, `dst_fd_`, `dst_offset_`) are omitted; `signal_pid` /
/// `signal_tid` are retained but inert (divergence 7).
#[derive(Debug, Default, Clone, Copy)]
pub struct LbmContext {
    /// Combination of `LBM_*` flags.
    pub flags: u32,
    /// Timeout in **milliseconds** (0 = no timeout).
    pub timeout_ms: i32,
    /// The SPSC ring this transfer runs over (C++ `copy_space` + `shm_info_`).
    pub ring: Option<ShmRing>,
    /// Server PID for the SHM liveness check (divergence 8).
    pub server_pid: i32,
    /// Waiter PID for `EventManager::Signal` (divergence 7).
    pub signal_pid: i32,
    /// Waiter TID for `EventManager::Signal` (divergence 7).
    pub signal_tid: i32,
}

impl LbmContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// C++ `LbmContext(uint32_t f)`.
    pub fn with_flags(flags: u32) -> Self {
        Self {
            flags,
            ..Self::default()
        }
    }

    /// C++ `LbmContext(uint32_t f, int timeout)`.
    pub fn with_timeout(flags: u32, timeout_ms: i32) -> Self {
        Self {
            flags,
            timeout_ms,
            ..Self::default()
        }
    }

    /// Attach the ring this transfer streams through (builder style).
    pub fn with_ring(mut self, ring: ShmRing) -> Self {
        self.ring = Some(ring);
        self
    }

    /// C++ `IsSync()`.
    pub fn is_sync(&self) -> bool {
        (self.flags & LBM_SYNC) != 0
    }

    /// C++ `HasTimeout()`.
    pub fn has_timeout(&self) -> bool {
        self.timeout_ms > 0
    }
}

// ---------------------------------------------------------------------------
// LocalSerialize equivalent (divergence 13)
// ---------------------------------------------------------------------------

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_ne_bytes());
}

fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_ne_bytes());
}

/// C++ `save_vec` for `priv::vector<Bulk>`: a `size_t` count, then each
/// `Bulk::serialize` = `(size, flags)`.
fn put_bulk_vec(out: &mut Vec<u8>, v: &[Bulk]) {
    put_u64(out, v.len() as u64);
    for b in v {
        put_u64(out, b.size as u64);
        put_u32(out, b.flags.bits());
    }
}

/// Wire size of one serialized `Bulk`: `size` (u64) + `flags` (u32).
const BULK_WIRE_LEN: usize = 12;

fn read_raw<'a>(buf: &'a [u8], pos: &mut usize, n: usize) -> Option<&'a [u8]> {
    let end = pos.checked_add(n)?;
    if end > buf.len() {
        return None;
    }
    let out = &buf[*pos..end];
    *pos = end;
    Some(out)
}

fn get_u32(buf: &[u8], pos: &mut usize) -> Option<u32> {
    let b = read_raw(buf, pos, 4)?;
    Some(u32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
}

fn get_u64(buf: &[u8], pos: &mut usize) -> Option<u64> {
    let b = read_raw(buf, pos, 8)?;
    Some(u64::from_ne_bytes([
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
    ]))
}

fn get_bulk_vec(buf: &[u8], pos: &mut usize) -> Option<Vec<Bulk>> {
    let n = get_u64(buf, pos)? as usize;
    // A bogus count must be rejected, never pre-allocated.
    if n.checked_mul(BULK_WIRE_LEN)? > buf.len().saturating_sub(*pos) {
        return None;
    }
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        let size = get_u64(buf, pos)? as usize;
        let flags = Bitfield32::new(get_u32(buf, pos)?);
        v.push(Bulk {
            data: Vec::new(),
            shm: ShmPtr::null(),
            size,
            flags,
        });
    }
    Some(v)
}

/// C++ `LbmMeta::serialize` under `LocalSerialize`: `(send, recv, send_bulks,
/// recv_bulks)`.
pub fn serialize_meta(meta: &LbmMeta) -> Vec<u8> {
    let mut out = Vec::new();
    put_bulk_vec(&mut out, &meta.send);
    put_bulk_vec(&mut out, &meta.recv);
    put_u64(&mut out, meta.send_bulks as u64);
    put_u64(&mut out, meta.recv_bulks as u64);
    out
}

#[allow(clippy::type_complexity)]
fn parse_meta(buf: &[u8]) -> Option<(Vec<Bulk>, Vec<Bulk>, u64, u64)> {
    let mut pos = 0usize;
    let send = get_bulk_vec(buf, &mut pos)?;
    let recv = get_bulk_vec(buf, &mut pos)?;
    let send_bulks = get_u64(buf, &mut pos)?;
    let recv_bulks = get_u64(buf, &mut pos)?;
    Some((send, recv, send_bulks, recv_bulks))
}

/// C++ `LocalDeserialize` of `LbmMeta`. Returns false on a truncated or bogus
/// frame (divergence 13); `meta` is then left untouched.
pub fn deserialize_meta(buf: &[u8], meta: &mut LbmMeta) -> bool {
    match parse_meta(buf) {
        Some((send, recv, send_bulks, recv_bulks)) => {
            meta.send = send;
            meta.recv = recv;
            meta.send_bulks = send_bulks as usize;
            meta.recv_bulks = recv_bulks as usize;
            true
        }
        None => false,
    }
}

/// Serialized length of a `ShmPtr` (C++ `sizeof(ShmPtr<char>)`).
const SHM_PTR_WIRE_LEN: usize = 16;

/// The frozen `ShmPtr` ABI as bytes: `major`, `minor`, `off` (divergence 12).
fn shm_to_bytes(p: &ShmPtr<u8>) -> [u8; SHM_PTR_WIRE_LEN] {
    let mut b = [0u8; SHM_PTR_WIRE_LEN];
    b[0..4].copy_from_slice(&p.alloc_id.major.to_ne_bytes());
    b[4..8].copy_from_slice(&p.alloc_id.minor.to_ne_bytes());
    b[8..16].copy_from_slice(&p.off.to_ne_bytes());
    b
}

fn shm_from_bytes(b: &[u8; SHM_PTR_WIRE_LEN]) -> ShmPtr<u8> {
    let major = u32::from_ne_bytes([b[0], b[1], b[2], b[3]]);
    let minor = u32::from_ne_bytes([b[4], b[5], b[6], b[7]]);
    let off = u64::from_ne_bytes([b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]]);
    ShmPtr::new(AllocatorId::new(major, minor), off)
}

// ---------------------------------------------------------------------------
// ShmTransferInfo — the SPSC ring's in-segment control block
// ---------------------------------------------------------------------------

/// C++ `ctp::ipc::atomic<T>::load_system()` on host: a volatile (relaxed) read,
/// no fence.
#[inline]
pub fn load_system(a: &AtomicU64) -> u64 {
    a.load(Ordering::Relaxed)
}

/// C++ `ctp::ipc::atomic<T>::store_system()` on host: seq_cst fence, volatile
/// write, seq_cst fence — prior writes are globally visible before the signal
/// value lands.
#[inline]
pub fn store_system(a: &AtomicU64, v: u64) {
    fence(Ordering::SeqCst);
    a.store(v, Ordering::Relaxed);
    fence(Ordering::SeqCst);
}

/// C++ `ctp::lbm::ShmTransferInfo`: SPSC ring metadata. The copy space is a
/// ring indexed by `total_written` / `total_read` modulo `copy_space_size`.
#[repr(C)]
pub struct ShmTransferInfo {
    /// Total bytes written by the producer.
    pub total_written: AtomicU64,
    /// Total bytes read by the consumer.
    pub total_read: AtomicU64,
    /// Ring capacity (atomic for cross-SM L2 visibility on GPU).
    pub copy_space_size: AtomicU64,
}

impl ShmTransferInfo {
    /// C++ `ShmTransferInfo()` — all counters zero.
    pub const fn new() -> Self {
        Self {
            total_written: AtomicU64::new(0),
            total_read: AtomicU64::new(0),
            copy_space_size: AtomicU64::new(0),
        }
    }

    /// Zeroed counters with the ring capacity published.
    pub const fn with_capacity(capacity: u64) -> Self {
        Self {
            total_written: AtomicU64::new(0),
            total_read: AtomicU64::new(0),
            copy_space_size: AtomicU64::new(capacity),
        }
    }

    /// Bytes written but not yet consumed (C++ `avail` lambda; wrapping per
    /// divergence 14).
    #[inline]
    pub fn avail(&self) -> u64 {
        load_system(&self.total_written).wrapping_sub(load_system(&self.total_read))
    }

    #[inline]
    pub fn capacity(&self) -> u64 {
        load_system(&self.copy_space_size)
    }
}

impl Default for ShmTransferInfo {
    fn default() -> Self {
        Self::new()
    }
}

/// A handle to one SPSC ring: C++ `LbmContext::shm_info_` + `copy_space` fused,
/// since neither is meaningful without the other.
#[derive(Clone, Copy)]
pub struct ShmRing {
    info: *const ShmTransferInfo,
    space: *mut u8,
}

// The ring is shared state by construction; all coordination happens through
// the in-segment atomics of `ShmTransferInfo` (MEMORY_DESIGN pillar 3).
unsafe impl Send for ShmRing {}
unsafe impl Sync for ShmRing {}

impl std::fmt::Debug for ShmRing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShmRing")
            .field("capacity", &self.capacity())
            .field("avail", &self.info().avail())
            .finish()
    }
}

impl ShmRing {
    /// Build a ring handle from a control block and its copy space.
    ///
    /// # Safety
    /// * `info` must point to a live, initialised [`ShmTransferInfo`] that stays
    ///   valid (and mapped) for as long as this handle is used.
    /// * `space` must point to at least `info.copy_space_size` writable bytes.
    /// * At most one producer and one consumer may use the ring (SPSC), as in
    ///   the C++.
    pub const unsafe fn from_raw(info: *const ShmTransferInfo, space: *mut u8) -> Self {
        Self { info, space }
    }

    #[inline]
    pub fn info(&self) -> &ShmTransferInfo {
        // SAFETY: `from_raw`'s contract — `info` points at a live
        // ShmTransferInfo for this handle's lifetime. Only shared (atomic)
        // access is handed out.
        unsafe { &*self.info }
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.info().capacity() as usize
    }
}

/// A heap-backed ring (header + copy space in one allocation), mirroring the
/// C++ caller-owned `FutureShm` copy space. Used for in-process transfers and
/// tests; [`ShmRingSegment`] is the cross-process form.
pub struct ShmRingBuffer {
    base: *mut u8,
    layout: Layout,
    capacity: usize,
}

// Same rationale as `ShmRing`: coordination is via the in-block atomics.
unsafe impl Send for ShmRingBuffer {}
unsafe impl Sync for ShmRingBuffer {}

impl ShmRingBuffer {
    /// Allocate a ring with `capacity` bytes of copy space.
    ///
    /// # Panics
    /// If `capacity` is 0 (a zero-sized ring makes C++ `WriteTransfer` spin
    /// forever — divergence 9) or the allocation fails.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "ring capacity must be non-zero");
        let size = std::mem::size_of::<ShmTransferInfo>() + capacity;
        let layout = Layout::from_size_align(size, std::mem::align_of::<ShmTransferInfo>())
            .expect("valid ring layout");
        // SAFETY: the layout has non-zero size; the result is null-checked.
        let base = unsafe { alloc_zeroed(layout) };
        assert!(!base.is_null(), "ring allocation failed");
        // SAFETY: `base` is a fresh, suitably aligned block of `size` bytes that
        // nothing else references yet, so placing the header (the C++
        // placement-new) is sound.
        unsafe {
            std::ptr::write(
                base as *mut ShmTransferInfo,
                ShmTransferInfo::with_capacity(capacity as u64),
            );
        }
        Self {
            base,
            layout,
            capacity,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// A (copyable) handle both the producer and the consumer can hold.
    pub fn ring(&self) -> ShmRing {
        // SAFETY: the header was initialised in `new`, the copy space is the
        // `capacity` bytes that follow it in the same allocation, and both
        // outlive every handle (the block is freed only in `Drop`).
        unsafe {
            ShmRing::from_raw(
                self.base as *const ShmTransferInfo,
                self.base.add(std::mem::size_of::<ShmTransferInfo>()),
            )
        }
    }
}

impl Drop for ShmRingBuffer {
    fn drop(&mut self) {
        // SAFETY: `base`/`layout` are exactly what `new` allocated.
        // `ShmTransferInfo` has no Drop (MEMORY_DESIGN), so the block can be
        // released without dropping it in place.
        unsafe { dealloc(self.base, self.layout) };
    }
}

/// A ring living in a named shared-memory segment from `ctp-memory`
/// (`[ ShmTransferInfo | copy space ]`), so producer and consumer can be
/// different processes — the deployment the C++ `ShmTransport` targets.
pub struct ShmRingSegment {
    backend: SharedMemBackend,
    is_owner: bool,
    capacity: usize,
}

impl ShmRingSegment {
    /// Create the segment and publish its ring capacity (server side).
    pub fn create(name: &str, capacity: usize) -> io::Result<Self> {
        if capacity == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "ring capacity must be non-zero",
            ));
        }
        let size = std::mem::size_of::<ShmTransferInfo>() + capacity;
        let backend = SharedMemBackend::create(name, size)?;
        // SAFETY: the mapping is `size` bytes and page-aligned (so suitably
        // aligned for the u64 atomics); no peer can observe the header before
        // this write — the capacity it publishes is what tells a client the
        // segment is ready.
        unsafe {
            std::ptr::write(
                backend.base() as *mut ShmTransferInfo,
                ShmTransferInfo::with_capacity(capacity as u64),
            );
        }
        Ok(Self {
            backend,
            is_owner: true,
            capacity,
        })
    }

    /// Attach to an existing segment; the capacity is read from its header
    /// (divergence 6: probe-then-map).
    pub fn open(name: &str) -> io::Result<Self> {
        let probe = SharedMemBackend::open(name, std::mem::size_of::<ShmTransferInfo>())?;
        // SAFETY: the probe maps exactly one ShmTransferInfo, written by the
        // creator before the segment was advertised.
        let capacity = unsafe { (*(probe.base() as *const ShmTransferInfo)).capacity() } as usize;
        drop(probe);
        if capacity == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "segment has no ring capacity",
            ));
        }
        let size = std::mem::size_of::<ShmTransferInfo>() + capacity;
        let backend = SharedMemBackend::open(name, size)?;
        Ok(Self {
            backend,
            is_owner: false,
            capacity,
        })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn name(&self) -> &str {
        self.backend.name()
    }

    /// A handle onto the segment's ring.
    pub fn ring(&self) -> ShmRing {
        // SAFETY: the header sits at the mapping's base (written by `create` or
        // validated by `open`); the copy space is the `capacity` bytes that
        // follow, all inside the mapping, which outlives the handle.
        unsafe {
            ShmRing::from_raw(
                self.backend.base() as *const ShmTransferInfo,
                self.backend
                    .base()
                    .add(std::mem::size_of::<ShmTransferInfo>()),
            )
        }
    }
}

impl Drop for ShmRingSegment {
    fn drop(&mut self) {
        // The creator unlinks the name (C++ shm_destroy); attachers only unmap
        // (shm_detach) — the backend's own Drop does that.
        if self.is_owner {
            self.backend.destroy();
        }
    }
}

// ---------------------------------------------------------------------------
// SPSC ring transfer primitives (C++ ShmTransport private statics)
// ---------------------------------------------------------------------------

/// C++ `ShmTransport::Min3`.
#[inline]
fn min3(a: usize, b: usize, c: usize) -> usize {
    a.min(b).min(c)
}

/// C++ `ShmTransport::WriteTransfer`: stream `data` into the ring, yielding
/// whenever it is full. The caller must have validated `ring.capacity() > 0`
/// (divergence 9).
pub fn write_transfer(data: &[u8], ring: &ShmRing) {
    let info = ring.info();
    let ring_size = info.capacity() as usize;
    debug_assert!(ring_size > 0, "zero-capacity ring (see divergence 9)");
    let size = data.len();
    let mut offset = 0usize;
    let mut total_written = load_system(&info.total_written);
    while offset < size {
        let total_read = load_system(&info.total_read);
        let space =
            (ring_size as u64).wrapping_sub(total_written.wrapping_sub(total_read)) as usize;
        if space == 0 {
            thread_model().yield_now();
            continue;
        }
        let write_pos = (total_written % ring_size as u64) as usize;
        let contig = ring_size - write_pos;
        let chunk = min3(size - offset, space, contig);
        // SAFETY: `chunk <= contig` keeps the write inside [write_pos,
        // ring_size) of the copy space, and `chunk <= space` keeps it off bytes
        // the consumer has yet to read; source and destination are distinct
        // allocations. The `store_system` below (seq_cst fenced) publishes these
        // bytes before the counter that reveals them.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr().add(offset),
                ring.space.add(write_pos),
                chunk,
            );
        }
        offset += chunk;
        total_written += chunk as u64;
        store_system(&info.total_written, total_written);
    }
}

/// C++ `ShmTransport::ReadTransfer`: drain `buf.len()` bytes from the ring,
/// yielding whenever it is empty.
pub fn read_transfer(buf: &mut [u8], ring: &ShmRing) {
    let info = ring.info();
    let ring_size = info.capacity() as usize;
    debug_assert!(ring_size > 0, "zero-capacity ring (see divergence 9)");
    let size = buf.len();
    let mut offset = 0usize;
    let mut total_read = load_system(&info.total_read);
    while offset < size {
        let total_written = load_system(&info.total_written);
        let avail = total_written.wrapping_sub(total_read) as usize;
        if avail == 0 {
            thread_model().yield_now();
            continue;
        }
        let read_pos = (total_read % ring_size as u64) as usize;
        let contig = ring_size - read_pos;
        let chunk = min3(size - offset, avail, contig);
        // SAFETY: `chunk <= contig` keeps the read inside [read_pos, ring_size)
        // of the copy space and `chunk <= avail` keeps it to bytes the producer
        // has already published (its `store_system` of total_written is seq_cst
        // fenced); the two buffers are distinct allocations.
        unsafe {
            std::ptr::copy_nonoverlapping(
                ring.space.add(read_pos),
                buf.as_mut_ptr().add(offset),
                chunk,
            );
        }
        offset += chunk;
        total_read += chunk as u64;
        store_system(&info.total_read, total_read);
    }
}

/// C++ `ShmTransport::WaitForTransferStart` — the `event_manager_ == nullptr`
/// branch (divergence 7): a ~10 µs busy-wait, then a `Yield()` loop bounded by
/// `timeout_ms`. Returns false only on a deadline with no data.
pub fn wait_for_transfer_start(ring: &ShmRing, timeout_ms: i32) -> bool {
    let info = ring.info();
    if info.avail() != 0 {
        return true;
    }
    let start = Instant::now();
    // Phase 1: short busy-wait — covers a fast sender with no syscall.
    while info.avail() == 0 {
        if start.elapsed() >= Duration::from_micros(10) {
            break;
        }
        thread_model().yield_now();
    }
    if info.avail() != 0 {
        return true;
    }
    // Phase 2: yield until the sender starts writing. `timeout_ms` (when > 0)
    // bounds the total wait so a never-produced result can't hang the caller.
    while info.avail() == 0 {
        if timeout_ms > 0 && start.elapsed() >= Duration::from_millis(timeout_ms as u64) {
            return false; // timed out with no data
        }
        thread_model().yield_now();
    }
    true
}

/// C++ `ShmTransport::RecvBulksImpl`: rebuild each recv bulk from the stream.
pub fn recv_bulks_impl(meta: &mut LbmMeta, ring: &ShmRing) -> i32 {
    for bulk in meta.recv.iter_mut() {
        if bulk.flags.any(BULK_EXPOSE) {
            // BULK_EXPOSE: read only the ShmPtr (no data transfer).
            let mut raw = [0u8; SHM_PTR_WIRE_LEN];
            read_transfer(&mut raw, ring);
            bulk.shm = shm_from_bytes(&raw);
            bulk.data.clear();
        } else if bulk.flags.any(BULK_XFER) {
            // BULK_XFER: read the ShmPtr first, then data if private memory.
            let mut raw = [0u8; SHM_PTR_WIRE_LEN];
            read_transfer(&mut raw, ring);
            let shm = shm_from_bytes(&raw);
            if !shm.alloc_id.is_null() {
                // Shared memory — ShmPtr passthrough, no data transfer.
                bulk.shm = shm;
                bulk.data.clear();
            } else {
                // Private memory — read the full data bytes. The C++ reuses a
                // caller-provided `ptr_` when present, else mallocs; here the Vec
                // is grown only when it cannot already hold `size`
                // (divergence 3).
                if bulk.data.len() < bulk.size {
                    bulk.data.resize(bulk.size, 0);
                }
                read_transfer(&mut bulk.data[..bulk.size], ring);
                bulk.shm = ShmPtr::new(AllocatorId::null(), NULL_OFFSET);
            }
        }
    }
    0
}

// ---------------------------------------------------------------------------
// ShmTransport
// ---------------------------------------------------------------------------

/// C++ `ctp::lbm::ShmTransport`: the per-task SPSC ring transport.
///
/// `send` / `recv` are associated functions (C++ statics); the instance only
/// carries the `Transport` base's mode/type (divergence 2).
#[derive(Debug, Clone, Copy)]
pub struct ShmTransport {
    pub mode: TransportMode,
    pub transport_type: TransportType,
}

impl ShmTransport {
    /// C++ `ShmTransport(TransportMode mode)` — sets `type_ = kShm`.
    pub fn new(mode: TransportMode) -> Self {
        Self {
            mode,
            transport_type: TransportType::Shm,
        }
    }

    /// C++ `Transport::IsServer()`.
    pub fn is_server(&self) -> bool {
        self.mode == TransportMode::Server
    }

    /// C++ `Transport::IsClient()`.
    pub fn is_client(&self) -> bool {
        self.mode == TransportMode::Client
    }

    /// C++ `ShmTransport::Expose`.
    pub fn expose(&self, data: Vec<u8>, shm: ShmPtr<u8>, data_size: usize, flags: u32) -> Bulk {
        Bulk {
            data,
            shm,
            size: data_size,
            flags: Bitfield32::new(flags),
        }
    }

    /// C++ `ShmTransport::GetAddress`.
    pub fn address(&self) -> &'static str {
        "shm"
    }

    /// C++ `ShmTransport::IsServerAlive` — always true here (divergence 8).
    pub fn is_server_alive(&self, _ctx: &LbmContext) -> bool {
        true
    }

    /// C++ `ShmTransport::ClearRecvHandles`: release received private-memory
    /// buffers. Dropping the `Vec`s is the `std::free` (divergence 3).
    pub fn clear_recv_handles(meta: &mut LbmMeta) {
        for bulk in meta.recv.iter_mut() {
            bulk.data = Vec::new();
        }
    }

    /// C++ `ShmTransport::Send`: serialize the metadata and bulk descriptors
    /// through the SPSC ring. Returns 0 on success, `EINVAL` on a bad ring or
    /// bulk descriptor (divergence 9).
    pub fn send(meta: &mut LbmMeta, ctx: &LbmContext) -> i32 {
        let ring = match ctx.ring {
            Some(r) => r,
            None => return EINVAL,
        };
        if ring.capacity() == 0 {
            return EINVAL;
        }
        // Validate every private-memory bulk BEFORE writing anything, so a short
        // payload can't leave a half-framed message in the ring.
        for bulk in &meta.send {
            if bulk_needs_data(bulk) && bulk.data.len() < bulk.size {
                return EINVAL;
            }
        }

        // 1. Serialize the metadata.
        let meta_buf = serialize_meta(meta);

        // 2. Transfer the serialized size, then the metadata.
        let meta_len = meta_buf.len() as u32;
        write_transfer(&meta_len.to_ne_bytes(), &ring);
        write_transfer(&meta_buf, &ring);

        // (C++ signals the waiter's EventManager here so it wakes with the
        // leading bytes already visible; not ported — divergence 7.)

        // 3. Send each bulk with the BULK_XFER or BULK_EXPOSE flag.
        for bulk in &meta.send {
            if bulk.flags.any(BULK_EXPOSE) {
                // BULK_EXPOSE: send only the ShmPtr (no data transfer).
                write_transfer(&shm_to_bytes(&bulk.shm), &ring);
            } else if bulk.flags.any(BULK_XFER) {
                // BULK_XFER: send the ShmPtr first, then data if private memory.
                write_transfer(&shm_to_bytes(&bulk.shm), &ring);
                if bulk.shm.alloc_id.is_null() {
                    write_transfer(&bulk.data[..bulk.size], &ring);
                }
            }
        }
        0
    }

    /// C++ `ShmTransport::Recv`: deserialize the metadata and receive bulk data
    /// through the SPSC ring. `rc = 0` on success; `EAGAIN` on a
    /// `ctx.timeout_ms` deadline; `EINVAL` / `EIO` per divergences 9 and 13.
    pub fn recv(meta: &mut LbmMeta, ctx: &LbmContext) -> ClientInfo {
        let mut info = ClientInfo::default();
        let ring = match ctx.ring {
            Some(r) => r,
            None => {
                info.rc = EINVAL;
                return info;
            }
        };
        if ring.capacity() == 0 {
            info.rc = EINVAL;
            return info;
        }

        // Block (after a short busy-wait) until the sender starts the transfer,
        // instead of busy-polling an empty ring.
        if !wait_for_transfer_start(&ring, ctx.timeout_ms) {
            info.rc = EAGAIN;
            return info;
        }

        // 1. Receive the 4-byte size prefix.
        let mut len_bytes = [0u8; 4];
        read_transfer(&mut len_bytes, &ring);
        let meta_len = u32::from_ne_bytes(len_bytes) as usize;

        // 2. Receive the metadata bytes.
        let mut meta_buf = vec![0u8; meta_len];
        read_transfer(&mut meta_buf, &ring);

        // 3. Deserialize.
        if !deserialize_meta(&meta_buf, meta) {
            info.rc = EIO;
            return info;
        }

        // 4. Set up recv entries from the send descriptors.
        let new_recv: Vec<Bulk> = meta
            .send
            .iter()
            .map(|b| Bulk {
                data: Vec::new(),
                shm: ShmPtr::null(),
                size: b.size,
                flags: b.flags,
            })
            .collect();
        meta.recv.extend(new_recv);

        // 5. Receive the bulk data.
        recv_bulks_impl(meta, &ring);

        info.rc = 0;
        info
    }
}

/// True when `send` must stream this bulk's bytes (private memory transfer).
fn bulk_needs_data(bulk: &Bulk) -> bool {
    !bulk.flags.any(BULK_EXPOSE) && bulk.flags.any(BULK_XFER) && bulk.shm.alloc_id.is_null()
}

// ---------------------------------------------------------------------------
// shm_mpsc_transport.h — tunables
// ---------------------------------------------------------------------------

/// C++ `kShmMpscDefaultSegmentSize`: default transfer space (header + ring).
pub const SHM_MPSC_DEFAULT_SEGMENT_SIZE: usize = 128 * 1024;
/// C++ `kShmMpscChunkSize`: per-chunk cap for `send_bytes`/`recv_bytes`.
pub const SHM_MPSC_CHUNK_SIZE: usize = 32 * 1024;
/// C++ `kShmMpscMaxXfers`: number of in-flight transfer slots the ring tracks.
pub const SHM_MPSC_MAX_XFERS: usize = 256;

/// C++ `SHM_MPSC_DONTWAIT`: `recv_bytes` returns `-EAGAIN` immediately when
/// nothing is in flight, instead of blocking. (It still blocks once a transfer
/// has been detected, to finish receiving that message.)
pub const SHM_MPSC_DONTWAIT: u32 = 0x1;

/// C++ `kShmMpscDeadXferUs`: how long the consumer waits on a not-yet-ready
/// chunk before declaring the producing connection dead (microseconds).
const SHM_MPSC_DEAD_XFER_US: u64 = 1_000_000; // 1 second
/// C++ `kShmMpscLivenessUs`: how long a producer waits for a free slot before
/// re-checking that the consumer is still alive (microseconds).
const SHM_MPSC_LIVENESS_US: u64 = 50_000; // 50 ms

/// C++ `ctp::lbm::ShmXferHeader`: one chunk's descriptor, living in the SHM
/// header. Layout-compatible with the C++ (divergence 11).
#[repr(C)]
pub struct ShmXferHeader {
    /// Producing connection's id.
    pub conn_id: AtomicU64,
    /// Absolute ring byte offset of this chunk's slot.
    pub xfer_off: AtomicU32,
    /// Number of bytes in this chunk.
    pub xfer_size: AtomicU32,
    /// Offset of this chunk within the producer's message.
    pub rem_off: AtomicU32,
    /// Total size of the producer's message.
    pub rem_size: AtomicU32,
    /// Producer sets after the memcpy; consumer clears.
    pub ready: AtomicBool,
}

impl ShmXferHeader {
    /// C++ `ShmXferHeader()` — zeroed, `ready_ = false`.
    pub const fn new() -> Self {
        Self {
            conn_id: AtomicU64::new(0),
            xfer_off: AtomicU32::new(0),
            xfer_size: AtomicU32::new(0),
            rem_off: AtomicU32::new(0),
            rem_size: AtomicU32::new(0),
            ready: AtomicBool::new(false),
        }
    }
}

impl Default for ShmXferHeader {
    fn default() -> Self {
        Self::new()
    }
}

/// C++ `ctp::lbm::ShmTransportHeader`: created once by the server at the start
/// of the segment; the ring is the bytes immediately following it.
///
/// The ring is divided into fixed [`SHM_MPSC_CHUNK_SIZE`] slots; slot id
/// `xfer_id` owns ring position `(xfer_id % num_slots) * SHM_MPSC_CHUNK_SIZE`.
/// Because the single `xfer_id_tail` counter assigns both the descriptor slot
/// AND the ring position, slot-id order and ring-offset order can never diverge
/// — which is what lets the consumer free a slot simply by advancing
/// `xfer_id_head`.
#[repr(C)]
pub struct ShmTransportHeader {
    /// Next client connection id.
    pub connection_id: AtomicU64,
    /// Next xfer slot to consume.
    pub xfer_id_head: AtomicU64,
    /// Next xfer slot to reserve.
    pub xfer_id_tail: AtomicU64,
    /// Server pid (liveness probe — divergence 8).
    pub pid: AtomicI32,
    /// Server tid.
    pub tid: AtomicI32,
    /// Ring capacity (segment - header).
    pub max_capacity: AtomicU64,
    /// In-flight chunk descriptors.
    pub xfers: [ShmXferHeader; SHM_MPSC_MAX_XFERS],
}

impl ShmTransportHeader {
    /// C++ `ShmTransportHeader()`; `xfers_[]` are default-constructed
    /// (`ready_ = false`).
    pub fn new() -> Self {
        Self {
            connection_id: AtomicU64::new(0),
            xfer_id_head: AtomicU64::new(0),
            xfer_id_tail: AtomicU64::new(0),
            pid: AtomicI32::new(0),
            tid: AtomicI32::new(0),
            max_capacity: AtomicU64::new(0),
            // `[ShmXferHeader::new(); N]` needs Copy; this is the non-Copy
            // array-init idiom.
            xfers: std::array::from_fn(|_| ShmXferHeader::new()),
        }
    }
}

impl Default for ShmTransportHeader {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ShmMpscTransport
// ---------------------------------------------------------------------------

/// Consumer-side per-connection reassembly state (C++ `RecvConn`).
#[derive(Debug, Default)]
struct RecvConn {
    buf: Vec<u8>,
    total: u32,
    received: u32,
}

/// C++ `ctp::lbm::ShmMpscTransport`: ONE named segment, many producers, one
/// consumer.
pub struct ShmMpscTransport {
    backend: Option<SharedMemBackend>,
    hdr: *mut ShmTransportHeader,
    ring: *mut u8,
    /// == `hdr.max_capacity`.
    pub cap: usize,
    /// Ring capacity in [`SHM_MPSC_CHUNK_SIZE`] slots.
    pub num_slots: usize,
    /// `min(SHM_MPSC_MAX_XFERS, num_slots)`.
    pub max_inflight: usize,
    pub is_server: bool,
    pub inited: bool,
    /// This client's connection id (server: 0).
    pub conn_id: u64,
    pub name: String,

    /// Serializes this connection's producers. A conn_id is one ordered stream
    /// at the consumer (which reassembles a single message per conn_id at a
    /// time), so when one client object is shared by multiple sender threads
    /// their messages must not interleave chunks under the shared conn_id. Held
    /// across the whole message; a blocking mutex (not a spinlock) because
    /// `send_bytes` yields waiting for ring capacity. Uncontended when each
    /// thread owns its own client.
    send_mu: Mutex<()>,

    /// Consumer-side per-connection reassembly state (single-threaded: the one
    /// Recv consumer owns this map).
    recv_conns: HashMap<u64, RecvConn>,
}

// The mapping is process-wide state; every cross-process handshake goes through
// the segment's atomics, and this type only ever hands out `&ShmTransportHeader`
// (atomic access) into it.
unsafe impl Send for ShmMpscTransport {}
unsafe impl Sync for ShmMpscTransport {}

impl Default for ShmMpscTransport {
    fn default() -> Self {
        Self {
            backend: None,
            hdr: std::ptr::null_mut(),
            ring: std::ptr::null_mut(),
            cap: 0,
            num_slots: 0,
            max_inflight: 0,
            is_server: false,
            inited: false,
            conn_id: 0,
            name: String::new(),
            send_mu: Mutex::new(()),
            recv_conns: HashMap::new(),
        }
    }
}

impl ShmMpscTransport {
    /// C++ `ShmMpscTransport() = default`.
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    fn hdr(&self) -> &ShmTransportHeader {
        debug_assert!(!self.hdr.is_null(), "transport not initialised");
        // SAFETY: `hdr` is non-null only between a successful `server_init` /
        // `client_init` and `shutdown`, where it points at the header placed at
        // the base of a mapping this object keeps alive in `backend`. Only
        // shared (atomic) access is handed out.
        unsafe { &*self.hdr }
    }

    /// Server init: create the named segment and place the header at its start.
    ///
    /// `xfer_space` is the total transfer space (header + ring); pass
    /// [`SHM_MPSC_DEFAULT_SEGMENT_SIZE`] for the C++ default. It is raised to
    /// `size_of::<ShmTransportHeader>() + SHM_MPSC_CHUNK_SIZE` when smaller, so
    /// the ring always holds at least one whole chunk (divergence 5).
    pub fn server_init(&mut self, name: &str, xfer_space: usize) -> bool {
        self.name = name.to_string();
        let hdr_size = std::mem::size_of::<ShmTransportHeader>();
        let min_space = hdr_size + SHM_MPSC_CHUNK_SIZE;
        let xfer_space = xfer_space.max(min_space);
        let backend = match SharedMemBackend::create(name, xfer_space) {
            Ok(b) => b,
            Err(_) => return false,
        };
        // SAFETY: the mapping is `xfer_space >= hdr_size` bytes and page-aligned
        // (so aligned for the header's atomics); nothing else has seen it yet,
        // so placing the header (the C++ placement-new) is sound.
        unsafe {
            std::ptr::write(
                backend.base() as *mut ShmTransportHeader,
                ShmTransportHeader::new(),
            )
        };
        self.hdr = backend.base() as *mut ShmTransportHeader;
        // C++ records the server pid/tid here for the liveness probe; without
        // ctp-introspect there is nothing to probe (divergence 8), so they stay
        // 0 and `server_alive` returns true — exactly as the C++ does on
        // Windows.
        self.cap = xfer_space - hdr_size;
        self.hdr()
            .max_capacity
            .store(self.cap as u64, Ordering::SeqCst);
        self.compute_slots();
        // SAFETY: the ring is the `cap` bytes after the header, inside the
        // mapping about to be owned by `self.backend`.
        self.ring = unsafe { backend.base().add(hdr_size) };
        self.backend = Some(backend);
        self.is_server = true;
        self.inited = true;
        true
    }

    /// Client init: attach to the named segment and take a connection id.
    pub fn client_init(&mut self, name: &str) -> bool {
        self.name = name.to_string();
        let hdr_size = std::mem::size_of::<ShmTransportHeader>();
        // Probe the header for the ring capacity, then map the whole segment
        // (divergence 6).
        let cap = match SharedMemBackend::open(name, hdr_size) {
            Ok(probe) => {
                // SAFETY: the probe maps exactly one ShmTransportHeader, which
                // `server_init` wrote before advertising the segment.
                let hdr = unsafe { &*(probe.base() as *const ShmTransportHeader) };
                hdr.max_capacity.load(Ordering::SeqCst) as usize
            }
            Err(_) => return false,
        };
        if cap < SHM_MPSC_CHUNK_SIZE {
            return false; // not a segment this transport can drive
        }
        let backend = match SharedMemBackend::open(name, hdr_size + cap) {
            Ok(b) => b,
            Err(_) => return false,
        };
        self.hdr = backend.base() as *mut ShmTransportHeader;
        self.cap = cap;
        self.compute_slots();
        // SAFETY: as in `server_init` — the ring follows the header inside the
        // mapping about to be owned by `self.backend`.
        self.ring = unsafe { backend.base().add(hdr_size) };
        self.backend = Some(backend);
        self.conn_id = self.hdr().connection_id.fetch_add(1, Ordering::SeqCst) + 1; // 0 is reserved
        self.is_server = false;
        self.inited = true;
        true
    }

    /// C++ `Shutdown()`: the server destroys (unmap + unlink), a client
    /// detaches (unmap).
    pub fn shutdown(&mut self) {
        if !self.inited {
            return;
        }
        self.inited = false;
        if let Some(backend) = self.backend.take() {
            if self.is_server {
                backend.destroy(); // unlink the name; Drop unmaps
            }
            drop(backend);
        }
        self.hdr = std::ptr::null_mut();
        self.ring = std::ptr::null_mut();
    }

    // --- Producer ---------------------------------------------------------

    /// C++ `SendBytes`: send `data` as one logical message from this
    /// connection. Returns 0 on success, `-EPIPE` if the consumer died
    /// mid-transfer (divergence 8), `-EINVAL` for a message > `u32::MAX`
    /// (divergence 10).
    pub fn send_bytes(&self, data: &[u8]) -> i32 {
        if data.len() > u32::MAX as usize {
            return -EINVAL;
        }
        // One message at a time per connection (see `send_mu`): keeps a shared
        // client's threads from interleaving chunks under the same conn_id.
        let _lk = self.send_mu.lock().unwrap_or_else(|e| e.into_inner());
        let hdr = self.hdr();
        let size = data.len();
        let rem_size = size as u32;
        let mut rem_off: u32 = 0;
        while (rem_off as usize) < size {
            // 1. Reserve an xfer slot; the SAME id picks both the descriptor and
            //    the fixed ring slot, so slot-id order == ring-offset order by
            //    construction. Wait until the slot is inside the in-flight
            //    window (which also means its ring slot has been drained).
            let xfer_id = hdr.xfer_id_tail.fetch_add(1, Ordering::SeqCst);
            if !self.wait_slot_free(xfer_id) {
                return -EPIPE;
            }
            // 2. Chunk size (each chunk fits wholly inside one slot, so no ring
            //    wraparound is possible within a chunk).
            let xfer_size = (size - rem_off as usize).min(SHM_MPSC_CHUNK_SIZE);
            // 3. Ring position derived from the slot id — no second counter to
            //    drift.
            let ring_pos = (xfer_id % self.num_slots as u64) as usize * SHM_MPSC_CHUNK_SIZE;
            debug_assert!(ring_pos + xfer_size <= self.cap);
            // SAFETY: `ring_pos < num_slots * CHUNK <= cap` and `xfer_size <=
            // CHUNK`, so the write stays inside the ring; `wait_slot_free`
            // guarantees no consumer is still draining this slot; source and
            // destination are distinct allocations.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr().add(rem_off as usize),
                    self.ring.add(ring_pos),
                    xfer_size,
                );
            }
            // 4. Publish the descriptor, then mark ready (release).
            let slot = &hdr.xfers[(xfer_id % SHM_MPSC_MAX_XFERS as u64) as usize];
            slot.conn_id.store(self.conn_id, Ordering::Relaxed);
            slot.xfer_off.store(ring_pos as u32, Ordering::Relaxed);
            slot.xfer_size.store(xfer_size as u32, Ordering::Relaxed);
            slot.rem_off.store(rem_off, Ordering::Relaxed);
            slot.rem_size.store(rem_size, Ordering::Relaxed);
            slot.ready.store(true, Ordering::SeqCst);
            // 5. Advance.
            rem_off += xfer_size as u32;
        }
        0
    }

    // --- Consumer ---------------------------------------------------------

    /// C++ `RecvBytes`: receive one complete message into `out` (resized to the
    /// message size) and report the producing connection id via `conn_out`.
    /// Returns 0 on success; `-EAGAIN` if `SHM_MPSC_DONTWAIT` and nothing is in
    /// flight.
    pub fn recv_bytes(&mut self, out: &mut Vec<u8>, conn_out: Option<&mut u64>, flags: u32) -> i32 {
        // Read the header pointer once: taking `&self.hdr()` here would borrow
        // all of `self` and lock out `self.recv_conns` below.
        let hdr_ptr = self.hdr;
        debug_assert!(!hdr_ptr.is_null(), "transport not initialised");
        // SAFETY: same contract as `hdr()` — a live header inside the mapping
        // `self.backend` keeps alive for as long as `inited`. Only shared
        // (atomic) access is made through it.
        let hdr: &ShmTransportHeader = unsafe { &*hdr_ptr };
        loop {
            let id_head = hdr.xfer_id_head.load(Ordering::SeqCst);
            let id_tail = hdr.xfer_id_tail.load(Ordering::SeqCst);
            if id_head == id_tail && self.recv_conns.is_empty() {
                if flags & SHM_MPSC_DONTWAIT != 0 {
                    return -EAGAIN;
                }
                thread_model().yield_now();
                continue;
            }
            if id_head == id_tail {
                // Nothing reserved yet but a message is partially received.
                thread_model().yield_now();
                continue;
            }
            let slot = &hdr.xfers[(id_head % SHM_MPSC_MAX_XFERS as u64) as usize];
            // Wait for the producer to publish this chunk; skip if it never
            // arrives.
            if !wait_chunk_ready(slot) {
                // Producer presumed dead: skip this slot. We cannot trust its
                // size, so its ring bytes are not reclaimed (a small leak for a
                // dead conn) — only the xfer-id cursor advances so the consumer
                // makes progress.
                hdr.xfer_id_head.fetch_add(1, Ordering::SeqCst);
                continue;
            }
            let conn = slot.conn_id.load(Ordering::Relaxed);
            let total = slot.rem_size.load(Ordering::Relaxed);
            let off = slot.rem_off.load(Ordering::Relaxed) as usize;
            let xsize = slot.xfer_size.load(Ordering::Relaxed) as usize;
            let xoff = slot.xfer_off.load(Ordering::Relaxed) as usize;
            // A corrupt descriptor must not read outside the ring
            // (divergence 9): drop the chunk and move on.
            if xoff.saturating_add(xsize) > self.cap {
                slot.ready.store(false, Ordering::SeqCst);
                hdr.xfer_id_head.fetch_add(1, Ordering::SeqCst);
                continue;
            }
            // De-mux into the per-connection buffer.
            let ring_ptr = self.ring;
            let st = self.recv_conns.entry(conn).or_default();
            if st.total == 0 {
                st.total = total;
                st.buf.resize(total as usize, 0);
                st.received = 0;
            }
            if off + xsize <= st.buf.len() {
                // The chunk lives wholly within one fixed slot, so a flat copy
                // suffices.
                // SAFETY: the source is bounds-checked against `cap` above and
                // the destination by the `if`; the producer's seq_cst `ready`
                // store happens-before the load in `wait_chunk_ready`, so the
                // bytes are visible; ring and buf are distinct allocations.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        ring_ptr.add(xoff),
                        st.buf.as_mut_ptr().add(off),
                        xsize,
                    );
                }
            }
            st.received += xsize as u32;
            let complete = st.received >= st.total;
            slot.ready.store(false, Ordering::SeqCst);
            // Advancing the cursor frees the ring slot: the next producer to
            // reuse this slot id (id + num_slots) waits on xfer_id_head via
            // `wait_slot_free`.
            hdr.xfer_id_head.fetch_add(1, Ordering::SeqCst);
            if complete {
                let st = self.recv_conns.remove(&conn).expect("entry just inserted");
                *out = st.buf;
                if let Some(c) = conn_out {
                    *c = conn;
                }
                return 0;
            }
        }
    }

    // --- High-level metadata + bulk API (mirrors ShmTransport::Send/Recv) ---

    /// C++ `ShmMpscTransport::Expose` (Transport::Expose parity).
    pub fn expose(&self, data: Vec<u8>, shm: ShmPtr<u8>, data_size: usize, flags: u32) -> Bulk {
        Bulk {
            data,
            shm,
            size: data_size,
            flags: Bitfield32::new(flags),
        }
    }

    /// C++ `ShmMpscTransport::Send`: serialize metadata + bulk data into ONE
    /// message and `send_bytes` it, so a producer's framing can't interleave
    /// with another's at the consumer.
    pub fn send(&self, meta: &LbmMeta) -> i32 {
        for bulk in &meta.send {
            if bulk_needs_data(bulk) && bulk.data.len() < bulk.size {
                return -EINVAL; // divergence 9
            }
        }
        let meta_buf = serialize_meta(meta);
        let mut msg: Vec<u8> = Vec::new();
        put_u32(&mut msg, meta_buf.len() as u32);
        msg.extend_from_slice(&meta_buf);
        for bulk in &meta.send {
            if bulk.flags.any(BULK_EXPOSE) {
                msg.extend_from_slice(&shm_to_bytes(&bulk.shm));
            } else if bulk.flags.any(BULK_XFER) {
                msg.extend_from_slice(&shm_to_bytes(&bulk.shm));
                if bulk.shm.alloc_id.is_null() {
                    msg.extend_from_slice(&bulk.data[..bulk.size]);
                }
            }
        }
        self.send_bytes(&msg)
    }

    /// C++ `ShmMpscTransport::Recv`: receive one complete message and rebuild
    /// metadata + bulks. `rc = 0` on success; `rc = EAGAIN` when
    /// `SHM_MPSC_DONTWAIT` and nothing is in flight (surfaced positive, as in
    /// C++); `rc = EIO` on a malformed frame.
    pub fn recv(&mut self, meta: &mut LbmMeta, flags: u32) -> ClientInfo {
        let mut info = ClientInfo::default();
        let mut msg: Vec<u8> = Vec::new();
        let mut conn: u64 = 0;
        let rc = self.recv_bytes(&mut msg, Some(&mut conn), flags);
        if rc != 0 {
            info.rc = if rc < 0 { -rc } else { rc }; // surface EAGAIN as positive errno
            return info;
        }
        let mut pos = 0usize;
        let meta_len = match get_u32(&msg, &mut pos) {
            Some(v) => v as usize,
            None => {
                info.rc = EIO;
                return info;
            }
        };
        let meta_buf = match read_raw(&msg, &mut pos, meta_len) {
            Some(b) => b.to_vec(),
            None => {
                info.rc = EIO;
                return info;
            }
        };
        if !deserialize_meta(&meta_buf, meta) {
            info.rc = EIO;
            return info;
        }
        let descs: Vec<(usize, Bitfield32)> =
            meta.send.iter().map(|b| (b.size, b.flags)).collect();
        for (size, flags) in descs {
            let mut recv_bulk = Bulk {
                data: Vec::new(),
                shm: ShmPtr::null(),
                size,
                flags,
            };
            if flags.any(BULK_EXPOSE) {
                match read_shm(&msg, &mut pos) {
                    Some(shm) => recv_bulk.shm = shm,
                    None => {
                        info.rc = EIO;
                        return info;
                    }
                }
            } else if flags.any(BULK_XFER) {
                let shm = match read_shm(&msg, &mut pos) {
                    Some(s) => s,
                    None => {
                        info.rc = EIO;
                        return info;
                    }
                };
                if !shm.alloc_id.is_null() {
                    recv_bulk.shm = shm;
                } else {
                    match read_raw(&msg, &mut pos, recv_bulk.size) {
                        Some(b) => {
                            recv_bulk.data = b.to_vec();
                            recv_bulk.shm = ShmPtr::new(AllocatorId::null(), NULL_OFFSET);
                        }
                        None => {
                            info.rc = EIO;
                            return info;
                        }
                    }
                }
            }
            meta.recv.push(recv_bulk);
        }
        info.rc = 0;
        info
    }

    // --- private ----------------------------------------------------------

    /// C++ `ServerAlive` — always true here (divergence 8).
    fn server_alive(&self) -> bool {
        true
    }

    /// C++ `ComputeSlots`: derive the ring's fixed-slot geometry from `cap`. The
    /// in-flight window is capped by BOTH the descriptor array size and the ring
    /// slot count so a reserved id never aliases a still-busy descriptor or a
    /// still-busy ring slot.
    fn compute_slots(&mut self) {
        self.num_slots = self.cap / SHM_MPSC_CHUNK_SIZE;
        if self.num_slots == 0 {
            // `server_init`/`client_init` guarantee cap >= CHUNK, so this is
            // unreachable; kept for parity with the C++ guard.
            self.num_slots = 1;
        }
        self.max_inflight = self.num_slots.min(SHM_MPSC_MAX_XFERS);
    }

    /// C++ `WaitSlotFree`: wait until this xfer slot is inside the bounded
    /// in-flight window. Because `max_inflight <= num_slots`, returning true
    /// also guarantees the ring slot owned by this id (its prior occupant
    /// `id - num_slots`) was consumed.
    fn wait_slot_free(&self, xfer_id: u64) -> bool {
        let hdr = self.hdr();
        let mut start = Instant::now();
        while xfer_id.wrapping_sub(hdr.xfer_id_head.load(Ordering::SeqCst))
            >= self.max_inflight as u64
        {
            if start.elapsed() >= Duration::from_micros(SHM_MPSC_LIVENESS_US) {
                if !self.server_alive() {
                    return false;
                }
                start = Instant::now();
            }
            thread_model().yield_now();
        }
        true
    }
}

fn read_shm(msg: &[u8], pos: &mut usize) -> Option<ShmPtr<u8>> {
    let b = read_raw(msg, pos, SHM_PTR_WIRE_LEN)?;
    let mut raw = [0u8; SHM_PTR_WIRE_LEN];
    raw.copy_from_slice(b);
    Some(shm_from_bytes(&raw))
}

/// C++ `WaitChunkReady`: wait for a chunk's ready flag; false => the producer is
/// presumed dead, skip it.
fn wait_chunk_ready(slot: &ShmXferHeader) -> bool {
    let start = Instant::now();
    while !slot.ready.load(Ordering::SeqCst) {
        if start.elapsed() >= Duration::from_micros(SHM_MPSC_DEAD_XFER_US) {
            return false;
        }
        thread_model().yield_now();
    }
    true
}

impl Drop for ShmMpscTransport {
    fn drop(&mut self) {
        // C++ `~ShmMpscTransport() { Shutdown(); }`.
        self.shutdown();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    static NAME_COUNTER: AtomicU32 = AtomicU32::new(0);

    fn unique_name(tag: &str) -> String {
        let n = NAME_COUNTER.fetch_add(1, Ordering::SeqCst);
        format!("ctp_rs_lbm_{}_{}_{}", tag, std::process::id(), n)
    }

    // --- ABI ---------------------------------------------------------------

    #[test]
    fn abi_layout_matches_cpp() {
        // ShmTransferInfo: three size_t atomics.
        assert_eq!(std::mem::size_of::<ShmTransferInfo>(), 24);
        assert_eq!(std::mem::align_of::<ShmTransferInfo>(), 8);
        // ShmXferHeader: u64 + 4*u32 + atomic<bool>, padded to 8.
        assert_eq!(std::mem::size_of::<ShmXferHeader>(), 32);
        assert_eq!(std::mem::align_of::<ShmXferHeader>(), 8);
        // ShmTransportHeader: 3*u64 + 2*int + size_t + 256 descriptors.
        assert_eq!(
            std::mem::size_of::<ShmTransportHeader>(),
            8 * 3 + 4 * 2 + 8 + SHM_MPSC_MAX_XFERS * 32
        );
        assert_eq!(std::mem::size_of::<ShmTransportHeader>(), 8232);

        // Field offsets (the C++ order is load-bearing: a client reads a header
        // the other language's server wrote).
        let h = ShmTransportHeader::new();
        let base = &h as *const _ as usize;
        assert_eq!(&h.connection_id as *const _ as usize - base, 0);
        assert_eq!(&h.xfer_id_head as *const _ as usize - base, 8);
        assert_eq!(&h.xfer_id_tail as *const _ as usize - base, 16);
        assert_eq!(&h.pid as *const _ as usize - base, 24);
        assert_eq!(&h.tid as *const _ as usize - base, 28);
        assert_eq!(&h.max_capacity as *const _ as usize - base, 32);
        assert_eq!(&h.xfers as *const _ as usize - base, 40);

        let x = ShmXferHeader::new();
        let xb = &x as *const _ as usize;
        assert_eq!(&x.conn_id as *const _ as usize - xb, 0);
        assert_eq!(&x.xfer_off as *const _ as usize - xb, 8);
        assert_eq!(&x.xfer_size as *const _ as usize - xb, 12);
        assert_eq!(&x.rem_off as *const _ as usize - xb, 16);
        assert_eq!(&x.rem_size as *const _ as usize - xb, 20);
        assert_eq!(&x.ready as *const _ as usize - xb, 24);
    }

    #[test]
    fn transfer_info_ctor_zeroes_counters() {
        let info = ShmTransferInfo::new();
        assert_eq!(info.avail(), 0);
        assert_eq!(info.capacity(), 0);
        let info = ShmTransferInfo::with_capacity(64);
        assert_eq!(info.capacity(), 64);
        assert_eq!(info.avail(), 0);
        // store_system / load_system round-trip (the C++ volatile pair).
        store_system(&info.total_written, 42);
        assert_eq!(load_system(&info.total_written), 42);
        assert_eq!(info.avail(), 42);
    }

    #[test]
    fn transfer_info_avail_wraps_like_size_t() {
        // Divergence 14: C++ size_t differences wrap; Rust must not panic.
        let info = ShmTransferInfo::with_capacity(8);
        store_system(&info.total_read, 4);
        store_system(&info.total_written, 2);
        assert_eq!(info.avail(), 2u64.wrapping_sub(4));
    }

    // --- ring primitives ---------------------------------------------------

    #[test]
    fn ring_roundtrip_small() {
        let buf = ShmRingBuffer::new(256);
        assert_eq!(buf.capacity(), 256);
        let ring = buf.ring();
        assert_eq!(ring.capacity(), 256);
        let msg = b"hello lightbeam";
        write_transfer(msg, &ring);
        assert_eq!(ring.info().avail(), msg.len() as u64);
        let mut out = vec![0u8; msg.len()];
        read_transfer(&mut out, &ring);
        assert_eq!(&out, msg);
        assert_eq!(ring.info().avail(), 0);
    }

    #[test]
    fn ring_zero_length_transfer_is_a_noop() {
        let buf = ShmRingBuffer::new(16);
        let ring = buf.ring();
        write_transfer(&[], &ring);
        assert_eq!(ring.info().avail(), 0);
        // Reading zero bytes from an empty ring must not spin.
        let mut out: Vec<u8> = Vec::new();
        read_transfer(&mut out, &ring);
        assert!(out.is_empty());
    }

    #[test]
    fn ring_single_byte_capacity() {
        let buf = ShmRingBuffer::new(1);
        let ring = buf.ring();
        for i in 0..4u8 {
            write_transfer(&[i], &ring);
            let mut out = [0u8; 1];
            read_transfer(&mut out, &ring);
            assert_eq!(out[0], i);
        }
    }

    #[test]
    fn ring_wraps_around_capacity() {
        let buf = ShmRingBuffer::new(16);
        let ring = buf.ring();
        // Fill part-way, drain, then write across the seam.
        let first: Vec<u8> = (0..10u8).collect();
        write_transfer(&first, &ring);
        let mut out = vec![0u8; 10];
        read_transfer(&mut out, &ring);
        assert_eq!(out, first);
        // write_pos is now 10; a 12-byte write must wrap (6 + 6).
        let second: Vec<u8> = (100..112u8).collect();
        write_transfer(&second, &ring);
        let mut out2 = vec![0u8; 12];
        read_transfer(&mut out2, &ring);
        assert_eq!(out2, second);
        // Counters keep growing; positions are modular.
        assert_eq!(load_system(&ring.info().total_written), 22);
        assert_eq!(load_system(&ring.info().total_read), 22);
    }

    #[test]
    fn ring_exact_capacity_fill_and_drain() {
        let cap = 8usize;
        let buf = ShmRingBuffer::new(cap);
        let ring = buf.ring();
        let msg: Vec<u8> = (0..cap as u8).collect();
        write_transfer(&msg, &ring); // fills the ring exactly (space == 0 after)
        assert_eq!(ring.info().avail(), cap as u64);
        let mut out = vec![0u8; cap];
        read_transfer(&mut out, &ring);
        assert_eq!(out, msg);
        assert_eq!(ring.info().avail(), 0);
    }

    #[test]
    fn ring_spsc_payload_larger_than_capacity() {
        // A payload many times the ring size only completes if producer and
        // consumer interleave — the back-pressure path.
        let buf = ShmRingBuffer::new(64);
        let payload: Vec<u8> = (0..100_000u32).map(|i| (i % 251) as u8).collect();
        let expect = payload.clone();
        let producer_ring = buf.ring();
        let producer = std::thread::spawn(move || {
            write_transfer(&payload, &producer_ring);
        });
        let mut out = vec![0u8; 100_000];
        read_transfer(&mut out, &buf.ring());
        producer.join().unwrap();
        assert_eq!(out, expect);
    }

    // --- serialization -----------------------------------------------------

    #[test]
    fn serialize_meta_wire_format() {
        let mut meta = LbmMeta::new();
        meta.send.push(Bulk {
            size: 7,
            flags: Bitfield32::new(BULK_XFER),
            ..Default::default()
        });
        meta.send_bulks = 1;
        let bytes = serialize_meta(&meta);
        // send: count(8) + [size(8) + flags(4)]; recv: count(8); then 2 * u64.
        assert_eq!(bytes.len(), 8 + BULK_WIRE_LEN + 8 + 8 + 8);
        assert_eq!(&bytes[0..8], &1u64.to_ne_bytes());
        assert_eq!(&bytes[8..16], &7u64.to_ne_bytes());
        assert_eq!(&bytes[16..20], &BULK_XFER.to_ne_bytes());
        assert_eq!(&bytes[20..28], &0u64.to_ne_bytes()); // recv count
        assert_eq!(&bytes[28..36], &1u64.to_ne_bytes()); // send_bulks
        assert_eq!(&bytes[36..44], &0u64.to_ne_bytes()); // recv_bulks
    }

    #[test]
    fn serialize_meta_roundtrip() {
        let mut meta = LbmMeta::new();
        for i in 0..3usize {
            meta.send.push(Bulk {
                data: vec![0u8; i],
                shm: ShmPtr::null(),
                size: i * 10,
                flags: Bitfield32::new(if i % 2 == 0 { BULK_XFER } else { BULK_EXPOSE }),
            });
        }
        meta.send_bulks = 2;
        meta.recv_bulks = 5;
        let bytes = serialize_meta(&meta);
        let mut got = LbmMeta::new();
        assert!(deserialize_meta(&bytes, &mut got));
        assert_eq!(got.send.len(), 3);
        assert!(got.recv.is_empty());
        assert_eq!(got.send_bulks, 2);
        assert_eq!(got.recv_bulks, 5);
        for i in 0..3usize {
            assert_eq!(got.send[i].size, i * 10);
            assert_eq!(got.send[i].flags, meta.send[i].flags);
            // Payloads are NOT part of the metadata frame.
            assert!(got.send[i].data.is_empty());
        }
    }

    #[test]
    fn serialize_empty_meta_roundtrip() {
        let meta = LbmMeta::new();
        let bytes = serialize_meta(&meta);
        assert_eq!(bytes.len(), 8 + 8 + 8 + 8);
        let mut got = LbmMeta::new();
        assert!(deserialize_meta(&bytes, &mut got));
        assert!(got.send.is_empty());
        assert!(got.recv.is_empty());
        assert_eq!(got.send_bulks, 0);
    }

    #[test]
    fn deserialize_meta_rejects_truncation_and_bogus_counts() {
        let mut meta = LbmMeta::new();
        meta.send.push(Bulk {
            size: 4,
            flags: Bitfield32::new(BULK_XFER),
            ..Default::default()
        });
        let bytes = serialize_meta(&meta);
        let mut got = LbmMeta::new();
        for cut in 0..bytes.len() {
            assert!(
                !deserialize_meta(&bytes[..cut], &mut got),
                "truncation at {cut} must not deserialize"
            );
        }
        assert!(deserialize_meta(&bytes, &mut got));
        // A huge vector count must be rejected, not pre-allocated (OOM).
        let mut bogus = u64::MAX.to_ne_bytes().to_vec();
        bogus.extend_from_slice(&[0u8; 8]);
        assert!(!deserialize_meta(&bogus, &mut got));
    }

    #[test]
    fn shm_ptr_wire_roundtrip() {
        let p = ShmPtr::<u8>::new(AllocatorId::new(9, 4), 0x1234_5678_9abc);
        let raw = shm_to_bytes(&p);
        assert_eq!(raw.len(), SHM_PTR_WIRE_LEN);
        let q = shm_from_bytes(&raw);
        assert_eq!(q.alloc_id, p.alloc_id);
        assert_eq!(q.off, p.off);
        // Frozen ABI order: major, minor, off.
        assert_eq!(&raw[0..4], &9u32.to_ne_bytes());
        assert_eq!(&raw[4..8], &4u32.to_ne_bytes());
        // Null round-trips as null.
        let n = shm_from_bytes(&shm_to_bytes(&ShmPtr::<u8>::null()));
        assert!(n.is_null());
        assert!(n.alloc_id.is_null());
    }

    // --- ShmTransport ------------------------------------------------------

    #[test]
    fn transport_identity() {
        let t = ShmTransport::new(TransportMode::Server);
        assert_eq!(t.address(), "shm");
        assert_eq!(t.transport_type, TransportType::Shm);
        assert!(t.is_server());
        assert!(!t.is_client());
        assert!(t.is_server_alive(&LbmContext::new()));
        assert!(ShmTransport::new(TransportMode::Client).is_client());
    }

    #[test]
    fn lbm_context_flags_and_timeout() {
        let c = LbmContext::with_flags(LBM_SYNC);
        assert!(c.is_sync());
        assert!(!c.has_timeout());
        let c = LbmContext::with_timeout(0, 5);
        assert!(!c.is_sync());
        assert!(c.has_timeout());
        assert_eq!(c.timeout_ms, 5);
    }

    #[test]
    fn transport_send_recv_metadata_only() {
        let buf = ShmRingBuffer::new(4096);
        let ctx = LbmContext::new().with_ring(buf.ring());
        let mut meta = LbmMeta::new();
        meta.send_bulks = 3;
        assert_eq!(ShmTransport::send(&mut meta, &ctx), 0);
        let mut got = LbmMeta::new();
        let info = ShmTransport::recv(&mut got, &ctx);
        assert_eq!(info.rc, 0);
        assert_eq!(info.fd, -1);
        assert_eq!(got.send_bulks, 3);
        assert!(got.recv.is_empty());
        assert_eq!(buf.ring().info().avail(), 0);
    }

    #[test]
    fn transport_send_recv_private_bulk_xfer() {
        let buf = ShmRingBuffer::new(4096);
        let ctx = LbmContext::new().with_ring(buf.ring());
        let payload: Vec<u8> = (0..64u8).collect();
        let t = ShmTransport::new(TransportMode::Client);
        let mut meta = LbmMeta::new();
        meta.send.push(t.expose(
            payload.clone(),
            ShmPtr::new(AllocatorId::null(), NULL_OFFSET),
            payload.len(),
            BULK_XFER,
        ));
        assert_eq!(ShmTransport::send(&mut meta, &ctx), 0);

        let mut got = LbmMeta::new();
        let info = ShmTransport::recv(&mut got, &ctx);
        assert_eq!(info.rc, 0);
        assert_eq!(got.recv.len(), 1);
        assert_eq!(got.recv[0].size, payload.len());
        assert_eq!(got.recv[0].data, payload);
        // Private memory: alloc_id stays null and the C++ address stuffing is
        // NOT reproduced (divergence 3a).
        assert!(got.recv[0].shm.alloc_id.is_null());
        assert_eq!(got.recv[0].shm.off, NULL_OFFSET);

        // clear_recv_handles releases the private buffers (C++ std::free).
        ShmTransport::clear_recv_handles(&mut got);
        assert!(got.recv[0].data.is_empty());
    }

    #[test]
    fn transport_send_recv_zero_sized_bulk() {
        let buf = ShmRingBuffer::new(256);
        let ctx = LbmContext::new().with_ring(buf.ring());
        let mut meta = LbmMeta::new();
        meta.send.push(Bulk {
            data: Vec::new(),
            shm: ShmPtr::new(AllocatorId::null(), NULL_OFFSET),
            size: 0,
            flags: Bitfield32::new(BULK_XFER),
        });
        assert_eq!(ShmTransport::send(&mut meta, &ctx), 0);
        let mut got = LbmMeta::new();
        assert_eq!(ShmTransport::recv(&mut got, &ctx).rc, 0);
        assert_eq!(got.recv.len(), 1);
        assert_eq!(got.recv[0].size, 0);
        assert!(got.recv[0].data.is_empty());
        assert_eq!(buf.ring().info().avail(), 0);
    }

    #[test]
    fn transport_send_recv_bulk_expose_and_shm_passthrough() {
        let buf = ShmRingBuffer::new(4096);
        let ctx = LbmContext::new().with_ring(buf.ring());
        let shm_a = ShmPtr::<u8>::new(AllocatorId::new(2, 1), 4096);
        let shm_b = ShmPtr::<u8>::new(AllocatorId::new(3, 7), 8192);
        let mut meta = LbmMeta::new();
        // BULK_EXPOSE: pointer only — `data` must NOT be transferred.
        meta.send.push(Bulk {
            data: vec![0xFF; 32],
            shm: shm_a,
            size: 32,
            flags: Bitfield32::new(BULK_EXPOSE),
        });
        // BULK_XFER with a real alloc_id: pointer passthrough, no data.
        meta.send.push(Bulk {
            data: Vec::new(),
            shm: shm_b,
            size: 16,
            flags: Bitfield32::new(BULK_XFER),
        });
        assert_eq!(ShmTransport::send(&mut meta, &ctx), 0);

        let mut got = LbmMeta::new();
        assert_eq!(ShmTransport::recv(&mut got, &ctx).rc, 0);
        assert_eq!(got.recv.len(), 2);
        assert_eq!(got.recv[0].shm.alloc_id, shm_a.alloc_id);
        assert_eq!(got.recv[0].shm.off, shm_a.off);
        assert!(got.recv[0].data.is_empty());
        assert_eq!(got.recv[1].shm.alloc_id, shm_b.alloc_id);
        assert_eq!(got.recv[1].shm.off, shm_b.off);
        assert!(got.recv[1].data.is_empty());
        assert_eq!(buf.ring().info().avail(), 0);
    }

    #[test]
    fn transport_rejects_missing_ring() {
        let mut meta = LbmMeta::new();
        let ctx = LbmContext::new();
        assert_eq!(ShmTransport::send(&mut meta, &ctx), EINVAL);
        assert_eq!(ShmTransport::recv(&mut meta, &ctx).rc, EINVAL);
    }

    #[test]
    fn transport_rejects_short_payload_before_writing() {
        let buf = ShmRingBuffer::new(256);
        let ctx = LbmContext::new().with_ring(buf.ring());
        let mut meta = LbmMeta::new();
        meta.send.push(Bulk {
            data: vec![1, 2, 3], // shorter than `size`
            shm: ShmPtr::new(AllocatorId::null(), NULL_OFFSET),
            size: 64,
            flags: Bitfield32::new(BULK_XFER),
        });
        assert_eq!(ShmTransport::send(&mut meta, &ctx), EINVAL);
        // Nothing was written: the ring is untouched (divergence 9).
        assert_eq!(buf.ring().info().avail(), 0);
    }

    #[test]
    fn transport_recv_times_out_with_eagain() {
        let buf = ShmRingBuffer::new(64);
        let ctx = LbmContext::with_timeout(0, 5).with_ring(buf.ring());
        let mut meta = LbmMeta::new();
        let start = Instant::now();
        let info = ShmTransport::recv(&mut meta, &ctx);
        assert_eq!(info.rc, EAGAIN);
        assert!(start.elapsed() >= Duration::from_millis(5));
    }

    #[test]
    fn wait_for_transfer_start_returns_immediately_when_data_present() {
        let buf = ShmRingBuffer::new(64);
        let ring = buf.ring();
        write_transfer(b"x", &ring);
        assert!(wait_for_transfer_start(&ring, 1));
        // With no timeout and data present it must not block either.
        assert!(wait_for_transfer_start(&ring, 0));
    }

    #[test]
    fn transport_across_threads_over_a_named_segment() {
        // The real deployment shape: producer and consumer hold independent
        // mappings of one named segment (here, two mappings in one process).
        let name = unique_name("seg");
        let server = ShmRingSegment::create(&name, 128).unwrap();
        assert_eq!(server.capacity(), 128);
        assert_eq!(server.name(), name);
        let client = ShmRingSegment::open(&name).unwrap();
        assert_eq!(client.capacity(), 128);

        let payload: Vec<u8> = (0..5_000u32).map(|i| (i % 97) as u8).collect();
        let expect = payload.clone();
        let send_ring = client.ring();
        let sender = std::thread::spawn(move || {
            let mut meta = LbmMeta::new();
            let len = payload.len();
            meta.send.push(Bulk {
                data: payload,
                shm: ShmPtr::new(AllocatorId::null(), NULL_OFFSET),
                size: len,
                flags: Bitfield32::new(BULK_XFER),
            });
            let ctx = LbmContext::new().with_ring(send_ring);
            ShmTransport::send(&mut meta, &ctx)
        });

        let mut got = LbmMeta::new();
        let ctx = LbmContext::new().with_ring(server.ring());
        let info = ShmTransport::recv(&mut got, &ctx);
        assert_eq!(sender.join().unwrap(), 0);
        assert_eq!(info.rc, 0);
        assert_eq!(got.recv.len(), 1);
        assert_eq!(got.recv[0].data, expect);
    }

    #[test]
    fn ring_segment_open_requires_a_creator() {
        assert!(ShmRingSegment::open(&unique_name("nosuch")).is_err());
        assert!(ShmRingSegment::create(&unique_name("zero"), 0).is_err());
    }

    // --- ShmMpscTransport --------------------------------------------------

    fn server(name: &str) -> ShmMpscTransport {
        let mut s = ShmMpscTransport::new();
        assert!(s.server_init(name, SHM_MPSC_DEFAULT_SEGMENT_SIZE));
        s
    }

    fn client(name: &str) -> ShmMpscTransport {
        let mut c = ShmMpscTransport::new();
        assert!(c.client_init(name));
        c
    }

    #[test]
    fn mpsc_geometry_and_connection_ids() {
        let name = unique_name("geom");
        let s = server(&name);
        assert!(s.is_server);
        assert!(s.inited);
        assert_eq!(s.conn_id, 0); // the server is conn 0
        assert_eq!(
            s.cap,
            SHM_MPSC_DEFAULT_SEGMENT_SIZE - std::mem::size_of::<ShmTransportHeader>()
        );
        assert_eq!(s.num_slots, s.cap / SHM_MPSC_CHUNK_SIZE);
        assert!(s.num_slots >= 1);
        assert_eq!(s.max_inflight, s.num_slots.min(SHM_MPSC_MAX_XFERS));

        // Connection ids start at 1 and increment per client (0 is reserved).
        let c1 = client(&name);
        let c2 = client(&name);
        assert_eq!(c1.conn_id, 1);
        assert_eq!(c2.conn_id, 2);
        assert_eq!(c1.cap, s.cap);
        assert_eq!(c1.num_slots, s.num_slots);
        assert!(!c1.is_server);
    }

    #[test]
    fn mpsc_tiny_segment_is_raised_to_hold_one_chunk() {
        // Divergence 5: a sub-chunk ring would let send_bytes write past cap.
        let name = unique_name("tiny");
        let mut s = ShmMpscTransport::new();
        assert!(s.server_init(&name, 8));
        assert_eq!(s.cap, SHM_MPSC_CHUNK_SIZE);
        assert_eq!(s.num_slots, 1);
        assert_eq!(s.max_inflight, 1);
    }

    #[test]
    fn mpsc_client_init_fails_without_a_server() {
        let mut c = ShmMpscTransport::new();
        assert!(!c.client_init(&unique_name("missing")));
        assert!(!c.inited);
    }

    #[test]
    fn mpsc_send_recv_single_chunk_message() {
        let name = unique_name("one");
        let mut s = server(&name);
        let c = client(&name);
        let msg = b"the quick brown fox".to_vec();
        assert_eq!(c.send_bytes(&msg), 0);
        let mut out = Vec::new();
        let mut conn = 0u64;
        assert_eq!(s.recv_bytes(&mut out, Some(&mut conn), 0), 0);
        assert_eq!(out, msg);
        assert_eq!(conn, c.conn_id);
        // The slot was released.
        assert_eq!(
            s.hdr().xfer_id_head.load(Ordering::SeqCst),
            s.hdr().xfer_id_tail.load(Ordering::SeqCst)
        );
    }

    #[test]
    fn mpsc_zero_length_send_puts_nothing_in_flight() {
        let name = unique_name("zero");
        let mut s = server(&name);
        let c = client(&name);
        assert_eq!(c.send_bytes(&[]), 0); // C++ parity: the loop never runs
        let mut out = Vec::new();
        assert_eq!(s.recv_bytes(&mut out, None, SHM_MPSC_DONTWAIT), -EAGAIN);
    }

    #[test]
    fn mpsc_dontwait_returns_eagain_when_idle() {
        let name = unique_name("dw");
        let mut s = server(&name);
        let _c = client(&name);
        let mut out = Vec::new();
        let mut conn = 7u64;
        assert_eq!(
            s.recv_bytes(&mut out, Some(&mut conn), SHM_MPSC_DONTWAIT),
            -EAGAIN
        );
        assert_eq!(conn, 7); // untouched
        assert!(out.is_empty());
    }

    #[test]
    fn mpsc_multi_chunk_message_needs_back_pressure() {
        // > CHUNK bytes and > the in-flight window: this only completes if the
        // consumer drains while the producer blocks in wait_slot_free.
        let name = unique_name("multi");
        let mut s = server(&name);
        let size = SHM_MPSC_CHUNK_SIZE * 5 + 123;
        let payload: Vec<u8> = (0..size).map(|i| (i % 253) as u8).collect();
        let expect = payload.clone();
        let cname = name.clone();
        let producer = std::thread::spawn(move || {
            let c = client(&cname);
            c.send_bytes(&payload)
        });
        let mut out = Vec::new();
        let mut conn = 0u64;
        assert_eq!(s.recv_bytes(&mut out, Some(&mut conn), 0), 0);
        assert_eq!(producer.join().unwrap(), 0);
        assert_eq!(out.len(), expect.len());
        assert_eq!(out, expect);
        assert_eq!(conn, 1);
    }

    #[test]
    fn mpsc_exact_chunk_multiple_message() {
        let name = unique_name("exact");
        let mut s = server(&name);
        let payload = vec![0xA5u8; SHM_MPSC_CHUNK_SIZE * 2];
        let expect = payload.clone();
        let cname = name.clone();
        let producer = std::thread::spawn(move || {
            let c = client(&cname);
            c.send_bytes(&payload)
        });
        let mut out = Vec::new();
        assert_eq!(s.recv_bytes(&mut out, None, 0), 0);
        assert_eq!(producer.join().unwrap(), 0);
        assert_eq!(out, expect);
    }

    #[test]
    fn mpsc_demuxes_concurrent_producers() {
        let name = unique_name("demux");
        let mut s = server(&name);
        const PRODUCERS: usize = 4;
        const PER_PRODUCER: usize = 8;
        let mut handles = Vec::new();
        for _ in 0..PRODUCERS {
            let cname = name.clone();
            handles.push(std::thread::spawn(move || {
                let c = client(&cname);
                let id = c.conn_id;
                for m in 0..PER_PRODUCER {
                    // Sizes straddle the chunk boundary so messages interleave
                    // in the ring and must be de-muxed by conn_id.
                    let len = SHM_MPSC_CHUNK_SIZE + m * 97 + 1;
                    let payload: Vec<u8> = (0..len).map(|i| ((i + m) % 251) as u8).collect();
                    assert_eq!(c.send_bytes(&payload), 0);
                }
                id
            }));
        }
        let mut seen: HashMap<u64, usize> = HashMap::new();
        for _ in 0..(PRODUCERS * PER_PRODUCER) {
            let mut out = Vec::new();
            let mut conn = 0u64;
            assert_eq!(s.recv_bytes(&mut out, Some(&mut conn), 0), 0);
            // Recover which message this was from its length and verify the
            // whole payload: a de-mux bug would splice two producers' bytes.
            let m = (out.len() - 1 - SHM_MPSC_CHUNK_SIZE) / 97;
            let expect: Vec<u8> = (0..out.len()).map(|i| ((i + m) % 251) as u8).collect();
            assert_eq!(out, expect, "conn {conn} message {m} was spliced");
            *seen.entry(conn).or_default() += 1;
        }
        let ids: Vec<u64> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(ids.len(), PRODUCERS);
        for id in ids {
            assert_eq!(seen.get(&id).copied().unwrap_or(0), PER_PRODUCER);
        }
    }

    #[test]
    fn mpsc_high_level_send_recv_with_bulks() {
        let name = unique_name("meta");
        let mut s = server(&name);
        let c = client(&name);
        let payload: Vec<u8> = (0..200u32).map(|i| (i % 255) as u8).collect();
        let shm = ShmPtr::<u8>::new(AllocatorId::new(5, 6), 1024);
        let mut meta = LbmMeta::new();
        meta.send.push(c.expose(
            payload.clone(),
            ShmPtr::new(AllocatorId::null(), NULL_OFFSET),
            payload.len(),
            BULK_XFER,
        ));
        meta.send.push(c.expose(Vec::new(), shm, 64, BULK_EXPOSE));
        meta.send_bulks = 1;
        assert_eq!(c.send(&meta), 0);

        let mut got = LbmMeta::new();
        let info = s.recv(&mut got, 0);
        assert_eq!(info.rc, 0);
        assert_eq!(got.send.len(), 2);
        assert_eq!(got.recv.len(), 2);
        assert_eq!(got.send_bulks, 1);
        assert_eq!(got.recv[0].data, payload);
        assert!(got.recv[0].shm.alloc_id.is_null());
        assert!(got.recv[1].data.is_empty());
        assert_eq!(got.recv[1].shm.alloc_id, shm.alloc_id);
        assert_eq!(got.recv[1].shm.off, shm.off);
    }

    #[test]
    fn mpsc_recv_dontwait_surfaces_eagain_positively() {
        // C++ parity: RecvBytes returns -EAGAIN, Recv reports it as +EAGAIN.
        let name = unique_name("eagain");
        let mut s = server(&name);
        let mut meta = LbmMeta::new();
        let info = s.recv(&mut meta, SHM_MPSC_DONTWAIT);
        assert_eq!(info.rc, EAGAIN);
        assert!(meta.recv.is_empty());
    }

    #[test]
    fn mpsc_high_level_send_rejects_short_payload() {
        let name = unique_name("short");
        let _s = server(&name);
        let c = client(&name);
        let mut meta = LbmMeta::new();
        meta.send.push(Bulk {
            data: vec![1, 2],
            shm: ShmPtr::new(AllocatorId::null(), NULL_OFFSET),
            size: 99,
            flags: Bitfield32::new(BULK_XFER),
        });
        assert_eq!(c.send(&meta), -EINVAL);
    }

    #[test]
    fn mpsc_recv_rejects_truncated_frame() {
        let name = unique_name("trunc");
        let mut s = server(&name);
        let c = client(&name);
        // A message whose meta_len prefix outruns the frame.
        let mut msg = Vec::new();
        put_u32(&mut msg, 999);
        msg.extend_from_slice(&[0u8; 4]);
        assert_eq!(c.send_bytes(&msg), 0);
        let mut meta = LbmMeta::new();
        assert_eq!(s.recv(&mut meta, 0).rc, EIO);
    }

    #[test]
    fn mpsc_shutdown_is_idempotent_and_drop_safe() {
        let name = unique_name("shut");
        let mut s = server(&name);
        {
            let mut c = client(&name);
            c.shutdown();
            assert!(!c.inited);
            c.shutdown(); // a second call is a no-op
        } // Drop runs shutdown a third time
        s.shutdown();
        assert!(!s.inited);
        assert!(s.hdr.is_null());
        s.shutdown();
    }

    #[test]
    fn mpsc_skips_a_dead_producers_slot() {
        // A producer that reserved an id and died: the consumer waits
        // kShmMpscDeadXferUs, advances past the slot, and still delivers the
        // next real message. (~1s by construction.)
        let name = unique_name("dead");
        let mut s = server(&name);
        let c = client(&name);
        // Simulate the dead reservation: take id 0 and never publish it.
        let dead_id = s.hdr().xfer_id_tail.fetch_add(1, Ordering::SeqCst);
        assert_eq!(dead_id, 0);
        let msg = b"after the dead slot".to_vec();
        assert_eq!(c.send_bytes(&msg), 0);
        let start = Instant::now();
        let mut out = Vec::new();
        let mut conn = 0u64;
        assert_eq!(s.recv_bytes(&mut out, Some(&mut conn), 0), 0);
        assert_eq!(out, msg);
        assert_eq!(conn, c.conn_id);
        assert!(start.elapsed() >= Duration::from_micros(SHM_MPSC_DEAD_XFER_US));
    }
}
