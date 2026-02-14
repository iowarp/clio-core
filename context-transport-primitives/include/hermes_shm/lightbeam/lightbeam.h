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
// Common types, interfaces, and factory for lightbeam transports.
// Users must include the appropriate transport header (zmq_transport.h,
// socket_transport.h) before using the factory for that transport.
#include <cassert>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <sstream>

#if HSHM_ENABLE_CEREAL
#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#endif

#include "hermes_shm/lightbeam/event_manager.h"
#include "hermes_shm/memory/allocator/allocator.h"
#include "hermes_shm/types/bitfield.h"

namespace hshm::lbm {

// Forward declaration — full definition in shm_transport.h
struct ShmTransferInfo;

// --- Bulk Flags ---
#define BULK_EXPOSE \
  BIT_OPT(hshm::u32, 0)                  // Bulk metadata sent, no data transfer
#define BULK_XFER BIT_OPT(hshm::u32, 1)  // Bulk marked for data transmission

// --- Types ---
struct Bulk {
  hipc::FullPtr<char> data;
  size_t size;
  hshm::bitfield32_t flags;  // BULK_EXPOSE or BULK_XFER
  void* desc = nullptr;      // For RDMA memory registration
  void* mr = nullptr;        // For RDMA memory region handle (fid_mr*)

#if HSHM_ENABLE_CEREAL
  template <typename Ar>
  void serialize(Ar& ar) {
    ar(size, flags);
  }
#endif
};

// --- Client Info (returned by Recv, used by Send for routing) ---
struct ClientInfo {
  int rc = 0;               // Return code (0 = success, EAGAIN = no data, etc.)
  int fd_ = -1;             // Socket fd (SocketTransport server mode)
  std::string identity_;    // ZMQ identity (ZeroMqTransport server mode)
};

// --- Metadata Base Class ---
class LbmMeta {
 public:
  std::vector<Bulk>
      send;  // Sender's bulk descriptors (can have BULK_EXPOSE or BULK_XFER)
  std::vector<Bulk>
      recv;  // Receiver's bulk descriptors (copy of send with local pointers)
  size_t send_bulks = 0;  // Count of BULK_XFER entries in send vector
  size_t recv_bulks = 0;  // Count of BULK_XFER entries in recv vector
  ClientInfo client_info_;  // Client routing info (not serialized)

#if HSHM_ENABLE_CEREAL
  template <typename Ar>
  void serialize(Ar& ar) {
    ar(send, recv, send_bulks, recv_bulks);
  }
#endif
};

// --- LbmContext ---
constexpr uint32_t LBM_SYNC =
    0x1; /**< Synchronous send (wait for completion) */

struct LbmContext {
  uint32_t flags;      /**< Combination of LBM_* flags */
  int timeout_ms;      /**< Timeout in milliseconds (0 = no timeout) */
  char* copy_space = nullptr;                      /**< Shared buffer for chunked transfer */
  ShmTransferInfo* shm_info_ = nullptr;            /**< Transfer info in shared memory */

  LbmContext() : flags(0), timeout_ms(0) {}

  explicit LbmContext(uint32_t f) : flags(f), timeout_ms(0) {}

  LbmContext(uint32_t f, int timeout) : flags(f), timeout_ms(timeout) {}

  bool IsSync() const { return (flags & LBM_SYNC) != 0; }
  bool HasTimeout() const { return timeout_ms > 0; }
};

// --- Transport Type Enum ---
enum class TransportType { kZeroMq, kSocket, kShm };

// --- Transport Mode Enum ---
enum class TransportMode { kClient, kServer };

// --- Unified Transport Interface ---
class Transport {
 public:
  TransportType type_;
  TransportMode mode_;

  Transport(TransportMode mode) : mode_(mode) {}
  virtual ~Transport() = default;

  bool IsServer() const { return mode_ == TransportMode::kServer; }
  bool IsClient() const { return mode_ == TransportMode::kClient; }

  // Shared APIs (both client and server)
  virtual Bulk Expose(const hipc::FullPtr<char>& ptr, size_t data_size,
                      u32 flags) = 0;

  template <typename MetaT>
  int Send(MetaT& meta, const LbmContext& ctx = LbmContext());

  template <typename MetaT>
  ClientInfo Recv(MetaT& meta, const LbmContext& ctx = LbmContext());

  // Server-only APIs (no-op defaults for client mode)
  virtual std::string GetAddress() const { return ""; }
  virtual int GetFd() const { return -1; }

  virtual void ClearRecvHandles(LbmMeta& meta) {
    for (auto& bulk : meta.recv) {
      if (bulk.data.ptr_ && !bulk.desc) {
        std::free(bulk.data.ptr_);
        bulk.data.ptr_ = nullptr;
      }
    }
  }

  // Event registration API
  virtual void RegisterEventManager(EventManager &em) { (void)em; }
};

// --- Factory ---
class TransportFactory {
 public:
  static std::unique_ptr<Transport> Get(const std::string& addr,
                                        TransportType t, TransportMode mode,
                                        const std::string& protocol = "",
                                        int port = 0);
  static std::unique_ptr<Transport> Get(const std::string& addr,
                                        TransportType t, TransportMode mode,
                                        const std::string& protocol, int port,
                                        const std::string& domain);
};

}  // namespace hshm::lbm
