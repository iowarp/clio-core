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

#pragma once

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <csignal>

#include "hermes_shm/data_structures/serialization/local_serialize.h"
#include "hermes_shm/thread/thread_model_manager.h"
#include "lightbeam.h"

namespace hshm::lbm {

// --- ShmTransferInfo ---
// SPSC ring buffer metadata for shared memory transport.
// The copy space is treated as a ring buffer indexed by total_written_ and
// total_read_ modulo copy_space_size_.
struct ShmTransferInfo {
  hipc::atomic<size_t> total_written_;  // Total bytes written by producer
  hipc::atomic<size_t> total_read_;     // Total bytes read by consumer
  size_t copy_space_size_;              // Ring buffer capacity

  HSHM_CROSS_FUN ShmTransferInfo() {
    total_written_.store(0);
    total_read_.store(0);
    copy_space_size_ = 0;
  }
};

class ShmTransport
#if HSHM_IS_HOST
  : public Transport
#endif
{
 public:
#if HSHM_IS_HOST
  explicit ShmTransport(TransportMode mode) : Transport(mode) {
    type_ = TransportType::kShm;
  }

  ~ShmTransport() = default;

  Bulk Expose(const hipc::FullPtr<char>& ptr, size_t data_size,
              u32 flags) {
    Bulk bulk;
    bulk.data = ptr;
    bulk.size = data_size;
    bulk.flags = hshm::bitfield32_t(flags);
    return bulk;
  }

  std::string GetAddress() const { return "shm"; }

  /** Check if the server is still alive via PID probe. */
  bool IsServerAlive(const LbmContext& ctx = LbmContext()) const {
    if (ctx.server_pid_ > 0) {
      if (kill(ctx.server_pid_, 0) == -1 && errno == ESRCH) return false;
    }
    return true;
  }

  void ClearRecvHandles(LbmMeta<>& meta) {
    for (auto& bulk : meta.recv) {
      if (bulk.data.ptr_ && !bulk.desc) {
        std::free(bulk.data.ptr_);
        bulk.data.ptr_ = nullptr;
      }
    }
  }
#endif

  /**
   * GPU-compatible static Send.
   * Serializes metadata and bulk descriptors through the SPSC ring buffer.
   * Uses the meta's allocator for temporary serialization buffers.
   *
   * @tparam MetaT Metadata type (must have allocator_type and alloc_ member)
   * @param meta Metadata with send bulk descriptors to transfer
   * @param ctx LbmContext with copy_space and shm_info_ for the ring buffer
   * @return 0 on success
   */
  template <typename MetaT>
  HSHM_CROSS_FUN
  static int Send(MetaT& meta, const LbmContext& ctx) {
    using AllocT = typename MetaT::allocator_type;
    using CharVec = hshm::priv::vector<char, AllocT>;

    // 1. Serialize metadata using LocalSerialize with allocator-backed buffer
    CharVec meta_buf(meta.alloc_);
    meta_buf.reserve(ctx.shm_info_->copy_space_size_);
    hshm::ipc::LocalSerialize<CharVec> ar(meta_buf);
    ar(meta);

    // 2. Transfer serialized size then metadata
    uint32_t meta_len = static_cast<uint32_t>(meta_buf.size());
    WriteTransfer(reinterpret_cast<const char*>(&meta_len), sizeof(meta_len),
                  ctx);
    WriteTransfer(meta_buf.data(), meta_buf.size(), ctx);

    // 3. Send each bulk with BULK_XFER or BULK_EXPOSE flag
    for (size_t i = 0; i < meta.send.size(); ++i) {
      if (meta.send[i].flags.Any(BULK_EXPOSE)) {
        // BULK_EXPOSE: Send only the ShmPtr (no data transfer)
        WriteTransfer(
            reinterpret_cast<const char*>(&meta.send[i].data.shm_),
            sizeof(meta.send[i].data.shm_), ctx);
      } else if (meta.send[i].flags.Any(BULK_XFER)) {
        // BULK_XFER: Send ShmPtr first, then data if private memory
        WriteTransfer(
            reinterpret_cast<const char*>(&meta.send[i].data.shm_),
            sizeof(meta.send[i].data.shm_), ctx);
        if (meta.send[i].data.shm_.alloc_id_.IsNull()) {
          // Private memory — also send full data bytes
          WriteTransfer(meta.send[i].data.ptr_, meta.send[i].size, ctx);
        }
      }
    }
    return 0;
  }

  /**
   * GPU-compatible static Recv.
   * Deserializes metadata and receives bulk data through the SPSC ring buffer.
   * Uses the meta's allocator for temporary deserialization buffers and
   * for allocating received private-memory bulk data.
   *
   * @tparam MetaT Metadata type (must have allocator_type and alloc_ member)
   * @param meta Metadata to populate with received data
   * @param ctx LbmContext with copy_space and shm_info_ for the ring buffer
   * @return ClientInfo with rc=0 on success
   */
  template <typename MetaT>
  HSHM_CROSS_FUN
  static ClientInfo Recv(MetaT& meta, const LbmContext& ctx) {
    using AllocT = typename MetaT::allocator_type;
    using CharVec = hshm::priv::vector<char, AllocT>;
    ClientInfo info;

    // 1. Receive 4-byte size prefix
    uint32_t meta_len = 0;
    ReadTransfer(reinterpret_cast<char*>(&meta_len), sizeof(meta_len), ctx);

    // 2. Receive metadata bytes into allocator-backed buffer
    CharVec meta_buf(meta_len, meta.alloc_);
    ReadTransfer(meta_buf.data(), meta_len, ctx);

    // 3. Deserialize using LocalDeserialize
    hshm::ipc::LocalDeserialize<CharVec> ar(meta_buf);
    ar(meta);

    // 4. Set up recv entries from send descriptors
    for (size_t i = 0; i < meta.send.size(); ++i) {
      Bulk recv_bulk;
      recv_bulk.size = meta.send[i].size;
      recv_bulk.flags = meta.send[i].flags;
      recv_bulk.data = hipc::FullPtr<char>::GetNull();
      meta.recv.push_back(recv_bulk);
    }

    // 5. Receive bulk data
    RecvBulksImpl(meta, ctx);
    info.rc = 0;
    return info;
  }

 private:
  /**
   * GPU-compatible static bulk data receiver.
   * Allocates receive buffers using the meta's allocator for private memory.
   */
  template <typename MetaT>
  HSHM_CROSS_FUN
  static int RecvBulksImpl(MetaT& meta, const LbmContext& ctx) {
    for (size_t i = 0; i < meta.recv.size(); ++i) {
      if (meta.recv[i].flags.Any(BULK_EXPOSE)) {
        // BULK_EXPOSE: Read only the ShmPtr (no data transfer)
        hipc::ShmPtr<char> shm;
        ReadTransfer(reinterpret_cast<char*>(&shm), sizeof(shm), ctx);
        meta.recv[i].data.shm_ = shm;
        meta.recv[i].data.ptr_ = nullptr;
      } else if (meta.recv[i].flags.Any(BULK_XFER)) {
        // BULK_XFER: Read ShmPtr first, then data if private memory
        hipc::ShmPtr<char> shm;
        ReadTransfer(reinterpret_cast<char*>(&shm), sizeof(shm), ctx);

        if (!shm.alloc_id_.IsNull()) {
          // Shared memory — ShmPtr passthrough, no data transfer
          meta.recv[i].data.shm_ = shm;
          meta.recv[i].data.ptr_ = nullptr;
        } else {
          // Private memory — read full data bytes
          char* buf = meta.recv[i].data.ptr_;
          bool allocated = false;
          if (!buf) {
#if HSHM_IS_HOST
            buf = static_cast<char*>(std::malloc(meta.recv[i].size));
#else
            auto alloc_ptr =
                meta.alloc_->template AllocateObjs<char>(meta.recv[i].size);
            buf = alloc_ptr.ptr_;
#endif
            allocated = true;
          }

          ReadTransfer(buf, meta.recv[i].size, ctx);

          if (allocated) {
            meta.recv[i].data.ptr_ = buf;
            meta.recv[i].data.shm_.alloc_id_ = hipc::AllocatorId::GetNull();
            meta.recv[i].data.shm_.off_ = reinterpret_cast<size_t>(buf);
          }
        }
      }
    }
    return 0;
  }

 public:
  // GPU-safe min of three values
  HSHM_CROSS_FUN
  static size_t Min3(size_t a, size_t b, size_t c) {
    size_t m = (a < b) ? a : b;
    return (m < c) ? m : c;
  }

  // GPU-safe memcpy
  HSHM_CROSS_FUN
  static void MemCopy(char* dst, const char* src, size_t n) {
#if HSHM_IS_HOST
    std::memcpy(dst, src, n);
#else
    for (size_t i = 0; i < n; ++i) {
      dst[i] = src[i];
    }
#endif
  }

  // SPSC ring buffer write (system-scope atomics for GPU/CPU visibility)
  HSHM_CROSS_FUN
  static void WriteTransfer(const char* data, size_t size, const LbmContext& ctx) {
    size_t offset = 0;
    size_t total_written = ctx.shm_info_->total_written_.load_system();
    while (offset < size) {
      size_t total_read = ctx.shm_info_->total_read_.load_system();
      size_t space =
          ctx.shm_info_->copy_space_size_ - (total_written - total_read);
      if (space == 0) {
#if HSHM_IS_HOST
        HSHM_THREAD_MODEL->Yield();
#endif
        continue;
      }
      size_t write_pos = total_written % ctx.shm_info_->copy_space_size_;
      size_t contig = ctx.shm_info_->copy_space_size_ - write_pos;
      size_t chunk = Min3(size - offset, space, contig);
      MemCopy(ctx.copy_space + write_pos, data + offset, chunk);
      offset += chunk;
      total_written += chunk;
      ctx.shm_info_->total_written_.store_system(total_written);
    }
  }

  // SPSC ring buffer read (system-scope atomics for GPU/CPU visibility)
  HSHM_CROSS_FUN
  static void ReadTransfer(char* buf, size_t size, const LbmContext& ctx) {
    size_t offset = 0;
    size_t total_read = ctx.shm_info_->total_read_.load_system();
    while (offset < size) {
      size_t total_written = ctx.shm_info_->total_written_.load_system();
      size_t avail = total_written - total_read;
      if (avail == 0) {
#if HSHM_IS_HOST
        HSHM_THREAD_MODEL->Yield();
#endif
        continue;
      }
      size_t read_pos = total_read % ctx.shm_info_->copy_space_size_;
      size_t contig = ctx.shm_info_->copy_space_size_ - read_pos;
      size_t chunk = Min3(size - offset, avail, contig);
      MemCopy(buf + offset, ctx.copy_space + read_pos, chunk);
      offset += chunk;
      total_read += chunk;
      ctx.shm_info_->total_read_.store_system(total_read);
    }
  }
};

}  // namespace hshm::lbm
