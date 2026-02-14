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

#include <algorithm>
#include <atomic>
#include <cstring>

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

class ShmTransport : public Transport {
 public:
  explicit ShmTransport(TransportMode mode) : Transport(mode) {
    type_ = TransportType::kShm;
  }

  ~ShmTransport() override = default;

  Bulk Expose(const hipc::FullPtr<char>& ptr, size_t data_size,
              u32 flags) override {
    Bulk bulk;
    bulk.data = ptr;
    bulk.size = data_size;
    bulk.flags = hshm::bitfield32_t(flags);
    return bulk;
  }

  std::string GetAddress() const override { return "shm"; }

  template <typename MetaT>
  int Send(MetaT& meta, const LbmContext& ctx = LbmContext()) {
    // 1. Serialize metadata using LocalSerialize
    std::vector<char> meta_buf;
    meta_buf.reserve(ctx.shm_info_->copy_space_size_);
    hshm::ipc::LocalSerialize<> ar(meta_buf);
    ar(meta);

    // 2. Transfer serialized size then metadata
    uint32_t meta_len = static_cast<uint32_t>(meta_buf.size());
    WriteTransfer(reinterpret_cast<const char*>(&meta_len), sizeof(meta_len), ctx);
    WriteTransfer(meta_buf.data(), meta_buf.size(), ctx);

    // 3. Send each bulk with BULK_XFER or BULK_EXPOSE flag
    for (size_t i = 0; i < meta.send.size(); ++i) {
      if (meta.send[i].flags.Any(BULK_EXPOSE)) {
        // BULK_EXPOSE: Send only the ShmPtr (no data transfer)
        WriteTransfer(reinterpret_cast<const char*>(&meta.send[i].data.shm_),
                 sizeof(meta.send[i].data.shm_), ctx);
      } else if (meta.send[i].flags.Any(BULK_XFER)) {
        // BULK_XFER: Send ShmPtr first, then data if private memory
        WriteTransfer(reinterpret_cast<const char*>(&meta.send[i].data.shm_),
                 sizeof(meta.send[i].data.shm_), ctx);
        if (meta.send[i].data.shm_.alloc_id_.IsNull()) {
          // Private memory — also send full data bytes
          WriteTransfer(meta.send[i].data.ptr_, meta.send[i].size, ctx);
        }
      }
    }
    return 0;
  }

  template <typename MetaT>
  ClientInfo Recv(MetaT& meta, const LbmContext& ctx = LbmContext()) {
    ClientInfo info;
    info.rc = RecvMetadata(meta, ctx);
    if (info.rc != 0) return info;
    // Set up recv entries from send descriptors
    for (const auto& send_bulk : meta.send) {
      Bulk recv_bulk;
      recv_bulk.size = send_bulk.size;
      recv_bulk.flags = send_bulk.flags;
      recv_bulk.data = hipc::FullPtr<char>::GetNull();
      meta.recv.push_back(recv_bulk);
    }
    info.rc = RecvBulks(meta, ctx);
    return info;
  }

 private:
  template <typename MetaT>
  int RecvMetadata(MetaT& meta, const LbmContext& ctx = LbmContext()) {
    // 1. Receive 4-byte size prefix
    uint32_t meta_len = 0;
    ReadTransfer(reinterpret_cast<char*>(&meta_len), sizeof(meta_len), ctx);

    // 2. Receive metadata bytes
    std::vector<char> meta_buf(meta_len);
    ReadTransfer(meta_buf.data(), meta_len, ctx);

    // 3. Deserialize using LocalDeserialize
    hshm::ipc::LocalDeserialize<> ar(meta_buf);
    ar(meta);
    return 0;
  }

  template <typename MetaT>
  int RecvBulks(MetaT& meta, const LbmContext& ctx = LbmContext()) {
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
            buf = static_cast<char*>(std::malloc(meta.recv[i].size));
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

 private:
  // SPSC ring buffer write
  static void WriteTransfer(const char* data, size_t size, const LbmContext& ctx) {
    size_t offset = 0;
    size_t total_written = ctx.shm_info_->total_written_.load();
    while (offset < size) {
      size_t total_read = ctx.shm_info_->total_read_.load();
      size_t space =
          ctx.shm_info_->copy_space_size_ - (total_written - total_read);
      if (space == 0) {
        HSHM_THREAD_MODEL->Yield();
        continue;
      }
      size_t write_pos = total_written % ctx.shm_info_->copy_space_size_;
      size_t contig = ctx.shm_info_->copy_space_size_ - write_pos;
      size_t chunk = std::min({size - offset, space, contig});
      std::memcpy(ctx.copy_space + write_pos, data + offset, chunk);
      offset += chunk;
      total_written += chunk;
      ctx.shm_info_->total_written_.store(total_written,
                                          std::memory_order_release);
    }
  }

  // SPSC ring buffer read
  static void ReadTransfer(char* buf, size_t size, const LbmContext& ctx) {
    size_t offset = 0;
    size_t total_read = ctx.shm_info_->total_read_.load();
    while (offset < size) {
      size_t total_written = ctx.shm_info_->total_written_.load();
      size_t avail = total_written - total_read;
      if (avail == 0) {
        HSHM_THREAD_MODEL->Yield();
        continue;
      }
      size_t read_pos = total_read % ctx.shm_info_->copy_space_size_;
      size_t contig = ctx.shm_info_->copy_space_size_ - read_pos;
      size_t chunk = std::min({size - offset, avail, contig});
      std::memcpy(buf + offset, ctx.copy_space + read_pos, chunk);
      offset += chunk;
      total_read += chunk;
      ctx.shm_info_->total_read_.store(total_read, std::memory_order_release);
    }
  }
};

}  // namespace hshm::lbm
