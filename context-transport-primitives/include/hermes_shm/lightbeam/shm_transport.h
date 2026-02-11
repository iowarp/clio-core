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
#include <cstring>
#include <thread>

#include "lightbeam.h"

namespace hshm::lbm {

static constexpr u32 SHM_DATA_READY = BIT_OPT(u32, 1);

class ShmClient : public Client {
 public:
  ShmClient() { type_ = Transport::kShm; }

  ~ShmClient() override = default;

  Bulk Expose(const hipc::FullPtr<char>& ptr, size_t data_size,
              u32 flags) override {
    Bulk bulk;
    bulk.data = ptr;
    bulk.size = data_size;
    bulk.flags = hshm::bitfield32_t(flags);
    return bulk;
  }

  template <typename MetaT>
  int Send(MetaT& meta, const LbmContext& ctx = LbmContext()) {
    (void)ctx;
    // 1. Serialize metadata via cereal
    std::ostringstream oss(std::ios::binary);
    {
      cereal::BinaryOutputArchive ar(oss);
      ar(meta);
    }
    std::string meta_str = oss.str();

    // 2. Send 4-byte size prefix then metadata
    uint32_t meta_len = static_cast<uint32_t>(meta_str.size());
    Transfer(reinterpret_cast<const char*>(&meta_len), sizeof(meta_len));
    Transfer(meta_str.data(), meta_str.size());

    // 3. Send each bulk with BULK_XFER flag
    for (size_t i = 0; i < meta.send.size(); ++i) {
      if (!meta.send[i].flags.Any(BULK_XFER)) continue;
      if (!meta.send[i].data.shm_.alloc_id_.IsNull()) {
        // Data lives in shared memory — send ShmPtr only
        uint8_t mode = 1;
        Transfer(reinterpret_cast<const char*>(&mode), sizeof(mode));
        Transfer(reinterpret_cast<const char*>(&meta.send[i].data.shm_),
                 sizeof(meta.send[i].data.shm_));
      } else {
        // Private memory — full data copy
        uint8_t mode = 0;
        Transfer(reinterpret_cast<const char*>(&mode), sizeof(mode));
        Transfer(meta.send[i].data.ptr_, meta.send[i].size);
      }
    }
    return 0;
  }

 private:
  void Transfer(const char* data, size_t size) {
    size_t offset = 0;
    while (offset < size) {
      // Wait until server consumed previous chunk
      while (ctx_.copy_flags_->Any(SHM_DATA_READY)) {
        std::this_thread::yield();
      }

      size_t chunk_size = std::min(size - offset, ctx_.copy_space_size);
      std::memcpy(ctx_.copy_space, data + offset, chunk_size);
      ctx_.transfer_size_->store(chunk_size);
      std::atomic_thread_fence(std::memory_order_release);
      ctx_.copy_flags_->SetBits(SHM_DATA_READY);
      offset += chunk_size;
    }
  }
};

class ShmServer : public Server {
 public:
  ShmServer() { type_ = Transport::kShm; }

  ~ShmServer() override = default;

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
  int RecvMetadata(MetaT& meta) {
    // 1. Receive 4-byte size prefix
    uint32_t meta_len = 0;
    Transfer(reinterpret_cast<char*>(&meta_len), sizeof(meta_len));

    // 2. Receive metadata bytes
    std::string meta_str(meta_len, '\0');
    Transfer(&meta_str[0], meta_len);

    // 3. Deserialize
    std::istringstream iss(meta_str, std::ios::binary);
    cereal::BinaryInputArchive ar(iss);
    ar(meta);
    return 0;
  }

  template <typename MetaT>
  int RecvBulks(MetaT& meta) {
    for (size_t i = 0; i < meta.recv.size(); ++i) {
      if (!meta.recv[i].flags.Any(BULK_XFER)) continue;

      // Read transfer mode: 0 = full data copy, 1 = ShmPtr only
      uint8_t mode = 0;
      Transfer(reinterpret_cast<char*>(&mode), sizeof(mode));

      if (mode == 1) {
        // ShmPtr-only transfer — read the ShmPtr, leave ptr_ null
        hipc::ShmPtr<char> shm;
        Transfer(reinterpret_cast<char*>(&shm), sizeof(shm));
        meta.recv[i].data.shm_ = shm;
        meta.recv[i].data.ptr_ = nullptr;
      } else {
        // Full data copy
        char* buf = meta.recv[i].data.ptr_;
        bool allocated = false;
        if (!buf) {
          buf = static_cast<char*>(std::malloc(meta.recv[i].size));
          allocated = true;
        }

        Transfer(buf, meta.recv[i].size);

        if (allocated) {
          meta.recv[i].data.ptr_ = buf;
          meta.recv[i].data.shm_.alloc_id_ = hipc::AllocatorId::GetNull();
          meta.recv[i].data.shm_.off_ = reinterpret_cast<size_t>(buf);
        }
      }
    }
    return 0;
  }

 private:
  void Transfer(char* buf, size_t size) {
    size_t offset = 0;
    while (offset < size) {
      // Wait until client wrote a chunk
      while (!ctx_.copy_flags_->Any(SHM_DATA_READY)) {
        std::this_thread::yield();
      }

      std::atomic_thread_fence(std::memory_order_acquire);
      size_t chunk_size = ctx_.transfer_size_->load();
      std::memcpy(buf + offset, ctx_.copy_space, chunk_size);
      ctx_.copy_flags_->UnsetBits(SHM_DATA_READY);
      offset += chunk_size;
    }
  }
};

}  // namespace hshm::lbm
