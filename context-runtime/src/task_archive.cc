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

#include "chimaera/task_archives.h"
#include "chimaera/ipc_manager.h"
#include "hermes_shm/util/logging.h"

namespace chi {

/**
 * SaveTaskArchive bulk transfer implementation
 * Adds bulk descriptor to send vector with proper Expose handling
 * @param ptr Shared memory pointer to the data
 * @param size Size of the data in bytes
 * @param flags Transfer flags (BULK_XFER or BULK_EXPOSE)
 */
void SaveTaskArchive::bulk(hipc::ShmPtr<> ptr, size_t size, uint32_t flags) {
  hipc::FullPtr<char> full_ptr = CHI_IPC->ToFullPtr(ptr).template Cast<char>();
  hshm::lbm::Bulk bulk;
  bulk.data = full_ptr;
  bulk.size = size;
  bulk.flags.bits_ = flags;

  // If lbm_client is provided, automatically call Expose for RDMA registration
  if (lbm_client_) {
    bulk = lbm_client_->Expose(bulk.data, bulk.size, bulk.flags.bits_);
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
void LoadTaskArchive::bulk(hipc::ShmPtr<> &ptr, size_t size, uint32_t flags) {
  HLOG(kDebug, "[LoadTaskArchive::bulk] Called with size={}, flags={}, msg_type_={}",
       size, flags, static_cast<int>(msg_type_));

  if (msg_type_ == MsgType::kSerializeIn) {
    // SerializeIn mode (input) - Get pointer from recv vector at current index
    // The task itself doesn't have a valid pointer during deserialization,
    // so we look into the recv vector and use the FullPtr at the current index
    if (current_bulk_index_ < recv.size()) {
      // Cast FullPtr<char>'s shm_ to ShmPtr<>
      ptr = recv[current_bulk_index_].data.shm_.template Cast<void>();
      current_bulk_index_++;
      HLOG(kDebug, "[LoadTaskArchive::bulk] SerializeIn - used recv[{}]", current_bulk_index_ - 1);
    } else {
      // Error: not enough bulk transfers in recv vector
      ptr = hipc::ShmPtr<>::GetNull();
      HLOG(kError, "[LoadTaskArchive::bulk] SerializeIn - recv vector empty or exhausted");
    }
  } else if (msg_type_ == MsgType::kSerializeOut) {
    if (current_bulk_index_ < recv.size()) {
      // Post-receive: point task's ShmPtr directly at recv buffer (zero-copy)
      if (recv[current_bulk_index_].flags.Any(BULK_XFER)) {
        ptr = recv[current_bulk_index_].data.shm_.template Cast<void>();
      }
      current_bulk_index_++;
    } else if (lbm_server_) {
      // Pre-receive: expose task's buffer for RecvBulks (existing RecvOut pattern)
      hipc::FullPtr<char> buffer = CHI_IPC->ToFullPtr(ptr).template Cast<char>();
      hshm::lbm::Bulk bulk = lbm_server_->Expose(buffer, size, flags);
      recv.push_back(bulk);
      if (flags & BULK_XFER) {
        recv_bulks++;
      }
    }
  }
  // kHeartbeat has no bulk transfers
}

} // namespace chi
