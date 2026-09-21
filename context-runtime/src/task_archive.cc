/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Task archive implementations for bulk transfer support
 * Contains SaveTaskArchive and LoadTaskArchive bulk() method implementations
 */

#include "clio_runtime/task_archives.h"
#include "clio_runtime/ipc_manager.h"
#include "clio_ctp/util/logging.h"
#include <clio_ctp/util/gpu_api.h>

namespace clio::run {

/**
 * SaveTaskArchive bulk transfer implementation
 * Adds bulk descriptor to send vector with proper Expose handling
 * @param ptr Shared memory pointer to the data
 * @param size Size of the data in bytes
 * @param flags Transfer flags (BULK_XFER or BULK_EXPOSE)
 */
void SaveTaskArchive::bulk(ctp::ipc::ShmPtr<> ptr, size_t size, uint32_t flags) {
  ctp::ipc::FullPtr<char> full_ptr = CLIO_IPC->ToFullPtr(ptr).template Cast<char>();
  // DEVICE MEMORY IS STAGED THROUGH THE HOST. A task bound for another node
  // can carry a bulk buffer that lives in GPU memory -- a paged vector's
  // frame being written back to the node that owns its blob -- and the
  // transport copies bulk bytes with a host memcpy inside Send. On Aurora
  // that was zmq_send(0xff000000112e0000, 65536) dying in memmove, on the
  // first cross-node page flush of a two-node kmeans. Managed memory is
  // host-readable and needs nothing; device-only memory is copied once into
  // a buffer this archive owns for as long as the send needs it.
  if (size != 0 && full_ptr.ptr_ != nullptr &&
      ctp::IsDevicePointer(full_ptr.ptr_)) {
    staged_.emplace_back(new char[size]);
    char *host = staged_.back().get();
    ctp::DeviceAwareMemcpy(host, full_ptr.ptr_, size);
    full_ptr.ptr_ = host;
    full_ptr.shm_ = ctp::ipc::ShmPtr<>::GetNull();
  }
  ctp::lbm::Bulk bulk;
  bulk.data = full_ptr;
  bulk.size = size;
  bulk.flags.bits_ = flags;

  // If lbm_client is provided, automatically call Expose for RDMA registration
  if (lbm_transport_) {
    bulk = lbm_transport_->Expose(bulk.data, bulk.size, bulk.flags.bits_);
  }

  send.push_back(bulk);

  // Track count of BULK_XFER entries for proper ZMQ_SNDMORE handling
  if (flags & BULK_XFER) {
    send_bulks++;
  }
}

/**
 * LoadTaskArchive bulk transfer implementation
 * Handles both SerializeIn and SerializeOut modes
 * @param ptr Reference to shared memory pointer (output parameter for SerializeIn)
 * @param size Size of the data in bytes
 * @param flags Transfer flags (BULK_XFER or BULK_EXPOSE)
 */
void LoadTaskArchive::bulk(ctp::ipc::ShmPtr<> &ptr, size_t size, uint32_t flags) {
  if (msg_type_ == MsgType::kSerializeIn) {
    // SerializeIn mode (input) - Get pointer from recv vector at current index
    // The task itself doesn't have a valid pointer during deserialization,
    // so we look into the recv vector and use the FullPtr at the current index
    if (current_bulk_index_ < recv.size()) {
      if (!recv[current_bulk_index_].data.shm_.IsNull()) {
        if (recv[current_bulk_index_].desc != nullptr) {
          // ZMQ zero-copy recv: the data currently lives in a
          // libzmq-owned zmq_msg_t buffer (RecvBulks stored its handle
          // in Bulk::desc). Pointing the task directly at it leaks that
          // zmq_msg_t on every inbound BULK_XFER: FreeBuffer cannot free
          // libzmq memory (so the TASK_DATA_OWNER destructor path can't
          // reclaim it — see admin_runtime.cc RecvIn), and the only
          // ClearRecvHandles call sites are the client *response* path,
          // never this server inbound path. Instead, copy into an owned
          // CHI buffer so the existing daemon_allocated_bulk_count_ /
          // TASK_DATA_OWNER machinery frees it safely; the caller then
          // frees the now-unreferenced zmq_msg_t via ClearRecvHandles
          // immediately after AllocLoadTask. One extra 1 MiB memcpy on
          // the TCP path (which already memcpys); SHM path (desc==null)
          // stays zero-copy and untouched.
          ctp::ipc::FullPtr<char> buf = CLIO_IPC->AllocateBuffer(size);
          char *src = recv[current_bulk_index_].data.ptr_;
          if (buf.ptr_ && src) {
            memcpy(buf.ptr_, src, size);
          }
          ptr = buf.shm_.template Cast<void>();
          recv[current_bulk_index_].data = buf;
          ++daemon_allocated_bulk_count_;
        } else if (recv[current_bulk_index_].data.shm_.alloc_id_ ==
                       ctp::ipc::AllocatorId(UINT32_MAX - 1,
                                              UINT32_MAX - 1)) {
          // SocketTransport (IPC/TCP-socket) recv: the buffer was
          // std::malloc'd in SocketTransport::RecvBulks and tagged with
          // the (UINT32_MAX-1, UINT32_MAX-1) sentinel. ClearRecvHandles
          // will std::free it immediately after AllocLoadTask returns —
          // pointing the task straight at recv[i].data leaves a dangling
          // ptr that the worker later memcpy()s in e.g.
          // bdev::Runtime::WriteToRam (ASan: heap-use-after-free). Mirror
          // the ZMQ path: copy into a CTP buffer the task owns via
          // TASK_DATA_OWNER, leaving recv[i].data alone so
          // ClearRecvHandles still frees the malloc'd buffer.
          ctp::ipc::FullPtr<char> buf = CLIO_IPC->AllocateBuffer(size);
          char *src = recv[current_bulk_index_].data.ptr_;
          if (buf.ptr_ && src) {
            memcpy(buf.ptr_, src, size);
          }
          ptr = buf.shm_.template Cast<void>();
          ++daemon_allocated_bulk_count_;
        } else {
          // Valid ShmPtr, no zmq handle: SHM transport (data already in
          // shared memory) — keep zero-copy.
          ptr = recv[current_bulk_index_].data.shm_.template Cast<void>();
        }
      } else {
        // Null ShmPtr: BULK_EXPOSE via ZMQ/socket where no data was sent.
        // Allocate a buffer for the receiver to fill (e.g., ReadTask).
        // This buffer is owned by the daemon and must be freed when the
        // task is deleted (see daemon_allocated_bulk_count_ usage in
        // admin_runtime.cc RecvIn / SendOut).
        ctp::ipc::FullPtr<char> buf = CLIO_IPC->AllocateBuffer(size);
        ptr = buf.shm_.template Cast<void>();
        recv[current_bulk_index_].data = buf;
        ++daemon_allocated_bulk_count_;
      }
      current_bulk_index_++;
    } else {
      // Error: not enough bulk transfers in recv vector
      ptr = ctp::ipc::ShmPtr<>::GetNull();
      HLOG(kError, "[LoadTaskArchive::bulk] SerializeIn - recv vector empty or exhausted");
    }
  } else if (msg_type_ == MsgType::kSerializeOut) {
    if (current_bulk_index_ < recv.size()) {
      // Post-receive (TCP/IPC path): data arrived in recv buffer
      if (recv[current_bulk_index_].flags.Any(BULK_XFER)) {
        // If the task already has a valid buffer (caller-provided),
        // copy received data into it so the caller's pointer stays valid.
        // This handles the TCP case where the caller allocated a read buffer
        // and expects data to appear there (matching SHM behavior).
        // Note: MallocAllocator uses null alloc_id_, so check IsNull() on
        // the ShmPtr (which checks offset) rather than alloc_id_.
        if (!ptr.IsNull()) {
          ctp::ipc::FullPtr<char> dst = CLIO_IPC->ToFullPtr(ptr).template Cast<char>();
          char *src = recv[current_bulk_index_].data.ptr_;
          size_t copy_size = recv[current_bulk_index_].size;
          if (dst.ptr_ && src) {
            memcpy(dst.ptr_, src, copy_size);
          }
        } else {
          // No original buffer — zero-copy, point directly at recv buffer
          ptr = recv[current_bulk_index_].data.shm_.template Cast<void>();
        }
      }
      current_bulk_index_++;
    } else if (lbm_transport_) {
      // Pre-receive: expose task's buffer for RecvBulks (existing RecvOut pattern)
      ctp::ipc::FullPtr<char> buffer = CLIO_IPC->ToFullPtr(ptr).template Cast<char>();
      ctp::lbm::Bulk bulk = lbm_transport_->Expose(buffer, size, flags);
      recv.push_back(bulk);
      if (flags & BULK_XFER) {
        recv_bulks++;
      }
    }
  }
  // kHeartbeat has no bulk transfers
}

}  // namespace clio::run
