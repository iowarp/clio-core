// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core - context-runtime Rust adaptation (issue #756).

//! Network task archives - how a task crosses a process boundary.
//!
//! Ports `clio_runtime/task_archives.h`: [`NetTaskArchive`],
//! [`SaveTaskArchive`], [`LoadTaskArchive`], [`TaskInfo`] and [`MsgType`].
//!
//! This is the mechanism behind the fact that **tasks are not stored in
//! shared memory**: a task lives in private memory, is serialized into a
//! buffer by [`SaveTaskArchive`], and that buffer is copied through shared
//! memory (or a socket) and rebuilt by [`LoadTaskArchive`] on the far side.
//!
//! # Why this crate exists
//!
//! These types were ported into `ctp-ds`'s `global_serialize` module - the
//! layer *below* the runtime that owns them. The C++ keeps CTP generic: no
//! CTP header mentions `clio::run`, because serialization there is a template
//! and needs no concrete type. Rust has no such escape, so the porting agent
//! needed real types to write `GlobalSave`/`GlobalLoad` against, defined them
//! next to the traits, and the archives followed their types down a layer.
//!
//! The consequence was not cosmetic: `TaskId` and `UniqueId` ended up
//! declared twice, and only the `clio-run-types` copies were checked against
//! the C++ headers by `cpp_abi_conformance`. See CTP's `PORT_AUDIT.md` §2.
//!
//! So the split is: `ctp-ds` keeps the byte engine (`GlobalSerialize`,
//! `GlobalDeserialize`, and the `GlobalSave`/`GlobalLoad` traits), the runtime
//! types live here and in `clio-run-types`, and each type's wire format is
//! implemented in its own crate - which the orphan rule requires anyway, and
//! which happens to be the correct direction.

#![deny(unsafe_op_in_unsafe_fn)]

use clio_run_types::{PoolId, TaskId};
use ctp_ds::global_serialize::{
    GlobalDeserialize, GlobalLoad, GlobalSave, GlobalSerialize, Result, SerializeError,
};
use ctp_lightbeam::transport::{
    Bulk, LbmMeta, BULK_EXPOSE, BULK_XFER, RECV_ALLOCATED_ID as SOCKET_RECV_SENTINEL,
};
use ctp_memory::{FullPtr, ShmPtr};

/// How the archives obtain and copy bulk buffers — C++ `CLIO_IPC->*`.
///
/// This is the runtime's `IpcManager` seen from the archives, which is why it
/// has no implementor yet: the thing that implements it is not ported. It
/// lived in `ctp-ds` until now, where its emptiness looked like a hollow
/// abstraction; it is really just a not-yet, and it belongs on this side of
/// the layer boundary regardless.
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
// Domain types (clio_runtime/types.h)
// ---------------------------------------------------------------------------

/// The local archives (`local_task_archives.h`) are not ported yet. When they
/// are, they should reuse [`TaskInfo`] rather than port `LocalTaskInfo`
/// alongside it: the two C++ structs are field-for-field identical
/// (`TaskId task_id_; PoolId pool_id_; u32 method_id_;`) and differ only in
/// how they serialize — `TaskInfo::serialize` delegates (`ar(task_id_,
/// pool_id_, method_id_)`), while `LocalTaskInfo` expands the fields by hand
/// and adds `ctp::ipc::save`/`load` overloads. That is one type with two wire
/// encodings, which in Rust is one struct with two trait impls.
///
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
// NetTaskArchive
// ---------------------------------------------------------------------------

/// C++ `clio::run::NetTaskArchive` — the metadata common to both network
/// archives, plus the `LbmMeta` bulk vectors it inherits.
///
/// C++ uses inheritance twice over — `NetTaskArchive : public
/// ctp::lbm::LbmMeta<>`, then `SaveTaskArchive : public NetTaskArchive` — and
/// Rust composes both levels: the archives hold a `meta: NetTaskArchive`, and
/// this holds an [`LbmMeta`].
///
/// It previously inlined `LbmMeta`'s four fields instead of holding one, which
/// made it a fourth copy of that type in a third crate. Holding the real thing
/// is what lets a transport's `LbmMeta` and an archive's be the same value.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct NetTaskArchive {
    /// The `LbmMeta` base: bulk descriptors and their XFER counts.
    pub lbm: LbmMeta,
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
        self.lbm.send.len()
    }

    /// C++ `GetRecvBulkCount()` — returns `recv.size()`, not `recv_bulks`.
    pub fn get_recv_bulk_count(&self) -> usize {
        self.lbm.recv.len()
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
        ar.save(&self.lbm.send);
        ar.save(&self.lbm.recv);
        ar.save(&self.lbm.send_bulks);
        ar.save(&self.lbm.recv_bulks);
        ar.save(&self.task_infos);
        ar.save(&self.msg_type);
        ar.save(&self.client_port);
    }

    fn load_header(ar: &mut GlobalDeserialize<'_>) -> Result<Self> {
        Ok(Self {
            lbm: LbmMeta {
                send: ar.load()?,
                recv: ar.load()?,
                send_bulks: ar.load()?,
                recv_bulks: ar.load()?,
                ..LbmMeta::default()
            },
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
        // The archive knows only the shared half; the local address is the
        // receiver's to resolve. `desc`/`mr` stay 0 — RDMA registration is the
        // caller's job here (divergence 10).
        let bulk = Bulk::new(FullPtr::new(0, ptr), size as usize, flags);
        self.meta.lbm.send.push(bulk);
        // C++: track count of BULK_XFER entries for ZMQ_SNDMORE handling.
        if flags & BULK_XFER != 0 {
            self.meta.lbm.send_bulks += 1;
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
        let len = self.meta.lbm.recv.len();
        if idx >= len {
            // C++: HLOG(kError) + ptr = GetNull(). We do both, and report.
            *ptr = ShmPtr::null();
            return Err(SerializeError::BulkExhausted { index: idx, len });
        }

        let entry = self.meta.lbm.recv[idx];
        if !entry.data.shm.is_null() {
            if entry.desc != 0 {
                // ZMQ zero-copy recv: copy into an owned buffer so the
                // TASK_DATA_OWNER destructor path can reclaim it, then leave
                // recv[i].data pointing at the owned buffer.
                let alloc = alloc.ok_or(SerializeError::NoBulkAllocator)?;
                let buf = alloc.allocate_buffer(size);
                alloc.copy_bulk(buf, entry.data.shm, size);
                *ptr = buf;
                self.meta.lbm.recv[idx].data = FullPtr::new(0, buf);
                self.daemon_allocated_bulk_count += 1;
            } else if entry.data.shm.alloc_id == SOCKET_RECV_SENTINEL {
                // SocketTransport recv: mirror the ZMQ path, but leave
                // recv[i].data alone so ClearRecvHandles still frees the
                // malloc'd buffer.
                let alloc = alloc.ok_or(SerializeError::NoBulkAllocator)?;
                let buf = alloc.allocate_buffer(size);
                alloc.copy_bulk(buf, entry.data.shm, size);
                *ptr = buf;
                self.daemon_allocated_bulk_count += 1;
            } else {
                // Valid ShmPtr, no transport handle: SHM transport — data is
                // already in shared memory, keep zero-copy.
                *ptr = entry.data.shm;
            }
        } else {
            // Null ShmPtr: BULK_EXPOSE via ZMQ/socket where no data was sent.
            // Allocate a buffer for the receiver to fill (e.g. ReadTask).
            let alloc = alloc.ok_or(SerializeError::NoBulkAllocator)?;
            let buf = alloc.allocate_buffer(size);
            *ptr = buf;
            self.meta.lbm.recv[idx].data = FullPtr::new(0, buf);
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
        if idx >= self.meta.lbm.recv.len() {
            return Ok(());
        }
        let entry = self.meta.lbm.recv[idx];
        if entry.is_xfer() {
            if !ptr.is_null() {
                // Caller supplied a buffer: copy the received data into it so
                // the caller's pointer stays valid (matches SHM behavior).
                let alloc = alloc.ok_or(SerializeError::NoBulkAllocator)?;
                alloc.copy_bulk(*ptr, entry.data.shm, entry.size as u64);
            } else {
                // No original buffer — zero-copy, point at the recv buffer.
                *ptr = entry.data.shm;
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
    use clio_run_types::UniqueId;
    use ctp_lightbeam::transport::FullPtr as LbFullPtr;
    use ctp_memory::AllocatorId;

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
        let b = { let mut b = Bulk::new(FullPtr::new(0, ShmPtr::new(AllocatorId::new(9, 9), 4096)), 1024, BULK_XFER); b.desc = 1; b };
        let mut ar = GlobalSerialize::new();
        ar.save(&b);
        let bytes = ar.into_bytes();
        assert_eq!(bytes.len(), 12, "size_t + u32, no pointer");
        assert_eq!(bytes[..8], 1024u64.to_ne_bytes());
        assert_eq!(bytes[8..], BULK_XFER.to_ne_bytes());

        // Loading drops the (process-local) pointer and desc flag.
        let got: Bulk = GlobalDeserialize::new(&bytes).load().unwrap();
        assert_eq!(got.size, 1024);
        assert_eq!(got.flags.bits(), BULK_XFER);
        assert!(got.data.shm.is_null());
        assert!(!got.desc != 0);
    }

    #[test]
    fn bulk_flag_predicates() {
        assert_eq!(BULK_EXPOSE, 1);
        assert_eq!(BULK_XFER, 2);
        let b = Bulk::new(FullPtr::null(), 8, BULK_EXPOSE | BULK_XFER);
        assert!(b.is_expose());
        assert!(b.is_xfer());
        let e = Bulk::new(FullPtr::null(), 8, BULK_EXPOSE);
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
        assert_eq!(ar.meta.lbm.send_bulks, 2);
        assert_eq!(ar.meta.get_send_bulk_count(), 3);
        assert_eq!(ar.meta.lbm.send[0].size, 100);
        assert_eq!(ar.meta.lbm.send[2].flags.bits(), BULK_EXPOSE | BULK_XFER);
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
        ta.meta.lbm.recv.push(Bulk::new(FullPtr::null(), 1024, BULK_XFER));
        ta.meta.lbm.recv_bulks = 1;
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
        assert_eq!(lo.meta.lbm.send_bulks, 1);
        assert_eq!(lo.meta.lbm.recv_bulks, 1);
        assert_eq!(lo.meta.lbm.send.len(), 2);
        assert_eq!(lo.meta.lbm.send[0].size, 1024);
        assert_eq!(lo.meta.lbm.send[0].flags.bits(), BULK_XFER);
        assert_eq!(lo.meta.lbm.send[1].size, 64);
        // Pointers do not cross the wire.
        assert!(lo.meta.lbm.send[0].data.shm.is_null());
        assert_eq!(lo.meta.lbm.recv.len(), 1);
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
        assert!(lo.meta.lbm.send.is_empty());
        assert!(lo.meta.lbm.recv.is_empty());
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
        lo.meta.lbm.recv = recv;
        lo
    }

    #[test]
    fn bulk_in_shm_path_is_zero_copy() {
        let shm = ShmPtr::new(AllocatorId::new(1, 0), 2048);
        let mut lo = load_with_recv(
            MsgType::SerializeIn,
            vec![Bulk::new(FullPtr::new(0, shm), 128, BULK_XFER)],
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
            vec![{ let mut b = Bulk::new(FullPtr::new(0, zmq_buf), 256, BULK_XFER); b.desc = 1; b }],
        );
        let mut alloc = MockAlloc::default();
        let mut ptr = ShmPtr::null();
        lo.bulk(&mut ptr, 256, BULK_XFER, Some(&mut alloc)).unwrap();
        assert_eq!(alloc.allocs, vec![256]);
        assert_eq!(alloc.copies, vec![(4096, 900, 256)]);
        assert_eq!(ptr.off, 4096);
        // C++ re-points recv[i].data at the owned buffer on this path.
        assert_eq!(lo.meta.lbm.recv[0].data.shm.off, 4096);
        assert_eq!(lo.daemon_allocated_bulk_count, 1);
    }

    #[test]
    fn bulk_in_socket_sentinel_path_copies_but_keeps_recv_pointer() {
        let malloc_buf = ShmPtr::new(SOCKET_RECV_SENTINEL, 700);
        let mut lo = load_with_recv(
            MsgType::SerializeIn,
            vec![Bulk::new(FullPtr::new(0, malloc_buf), 64, BULK_XFER)],
        );
        let mut alloc = MockAlloc::default();
        let mut ptr = ShmPtr::null();
        lo.bulk(&mut ptr, 64, BULK_XFER, Some(&mut alloc)).unwrap();
        assert_eq!(alloc.allocs, vec![64]);
        assert_eq!(alloc.copies, vec![(4096, 700, 64)]);
        assert_eq!(ptr.off, 4096);
        // C++ leaves recv[i].data alone so ClearRecvHandles still frees it.
        assert_eq!(lo.meta.lbm.recv[0].data.shm.off, 700);
        assert_eq!(lo.meta.lbm.recv[0].data.shm.alloc_id, SOCKET_RECV_SENTINEL);
        assert_eq!(lo.daemon_allocated_bulk_count, 1);
    }

    #[test]
    fn bulk_in_null_expose_path_allocates_receive_buffer() {
        let mut lo = load_with_recv(
            MsgType::SerializeIn,
            vec![Bulk::new(FullPtr::new(0, ShmPtr::null()), 1024, BULK_EXPOSE)],
        );
        let mut alloc = MockAlloc::default();
        let mut ptr = ShmPtr::null();
        lo.bulk(&mut ptr, 1024, BULK_EXPOSE, Some(&mut alloc))
            .unwrap();
        assert_eq!(alloc.allocs, vec![1024]);
        assert!(alloc.copies.is_empty(), "nothing to copy: no data was sent");
        assert_eq!(ptr.off, 4096);
        assert_eq!(lo.meta.lbm.recv[0].data.shm.off, 4096);
        assert_eq!(lo.daemon_allocated_bulk_count, 1);
    }

    #[test]
    fn bulk_in_advances_the_index_across_calls() {
        let mut lo = load_with_recv(
            MsgType::SerializeIn,
            vec![
                Bulk::new(FullPtr::new(0, ShmPtr::new(AllocatorId::new(1, 0), 100)), 8, BULK_XFER),
                Bulk::new(FullPtr::new(0, ShmPtr::new(AllocatorId::new(1, 0), 200)), 8, BULK_XFER),
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
            vec![Bulk::new(FullPtr::new(0, ShmPtr::new(AllocatorId::new(1, 0), 32)), 8, BULK_XFER)],
        );
        let mut ptr = ShmPtr::null();
        assert!(lo.bulk(&mut ptr, 8, BULK_XFER, None).is_ok());
        assert_eq!(ptr.off, 32);

        // The BULK_EXPOSE/null path does.
        let mut lo2 = load_with_recv(MsgType::SerializeIn, vec![Bulk::new(FullPtr::null(), 8, BULK_EXPOSE)]);
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
            vec![Bulk::new(FullPtr::new(0, ShmPtr::new(AllocatorId::new(2, 2), 800)), 512, BULK_XFER)],
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
            vec![Bulk::new(FullPtr::new(0, ShmPtr::new(AllocatorId::new(2, 2), 800)), 512, BULK_XFER)],
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
            vec![Bulk::new(FullPtr::new(0, ShmPtr::new(AllocatorId::new(2, 2), 800)), 512, BULK_EXPOSE)],
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
        let mut lo = load_with_recv(MsgType::Heartbeat, vec![Bulk::new(FullPtr::null(), 8, BULK_XFER)]);
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
}
