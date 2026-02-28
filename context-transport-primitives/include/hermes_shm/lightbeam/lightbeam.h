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

#include "hermes_shm/data_structures/priv/vector.h"
#include "hermes_shm/lightbeam/event_manager.h"
#include "hermes_shm/memory/allocator/allocator.h"
#include "hermes_shm/memory/allocator/malloc_allocator.h"
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

  /** Serialize bulk descriptor metadata (size and flags only) */
  template <typename Ar>
  HSHM_CROSS_FUN void serialize(Ar& ar) {
    ar(size, flags);
  }
};

// --- Client Info (returned by Recv, used by Send for routing) ---
struct ClientInfo {
  int rc = 0;               // Return code (0 = success, EAGAIN = no data, etc.)
  int fd_ = -1;             // Socket fd (SocketTransport server mode)
#if !HSHM_IS_GPU
  std::string identity_;    // ZMQ identity (ZeroMqTransport server mode)
#endif
};

// --- Metadata Base Class ---
/**
 * GPU-compatible metadata for lightbeam transports.
 * Uses hshm::priv::vector with a configurable allocator so that
 * bulk descriptor vectors can be managed in GPU-accessible memory.
 *
 * @tparam AllocT Allocator type for bulk descriptor vectors.
 *                Defaults to MallocAllocator for host-side usage.
 */
template <typename AllocT = hshm::ipc::MallocAllocator>
class LbmMeta {
 public:
  using allocator_type = AllocT;
  using BulkVector = hshm::priv::vector<Bulk, AllocT>;

  BulkVector
      send;  // Sender's bulk descriptors (can have BULK_EXPOSE or BULK_XFER)
  BulkVector
      recv;  // Receiver's bulk descriptors (copy of send with local pointers)
  size_t send_bulks = 0;  // Count of BULK_XFER entries in send vector
  size_t recv_bulks = 0;  // Count of BULK_XFER entries in recv vector
  AllocT* alloc_;          // Allocator used for internal vectors
#if !HSHM_IS_GPU
  ClientInfo client_info_;  // Client routing info (not serialized, host-only)
#endif

  /** Default constructor (uses HSHM_MALLOC allocator) */
  HSHM_CROSS_FUN LbmMeta()
      : send(HSHM_MALLOC), recv(HSHM_MALLOC), alloc_(HSHM_MALLOC) {}

  /** Constructor with custom allocator */
  HSHM_CROSS_FUN explicit LbmMeta(AllocT* alloc)
      : send(alloc), recv(alloc), alloc_(alloc) {}

  /** Move constructor */
  HSHM_CROSS_FUN LbmMeta(LbmMeta&& other) noexcept
      : send(std::move(other.send)),
        recv(std::move(other.recv)),
        send_bulks(other.send_bulks),
        recv_bulks(other.recv_bulks),
        alloc_(other.alloc_) {}

  /** Move assignment operator */
  HSHM_CROSS_FUN LbmMeta& operator=(LbmMeta&& other) noexcept {
    if (this != &other) {
      send = std::move(other.send);
      recv = std::move(other.recv);
      send_bulks = other.send_bulks;
      recv_bulks = other.recv_bulks;
      alloc_ = other.alloc_;
    }
    return *this;
  }

  /** Serialize metadata for LocalSerialize/LocalDeserialize */
  template <typename Ar>
  HSHM_CROSS_FUN void serialize(Ar& ar) {
    ar(send, recv, send_bulks, recv_bulks);
  }
};

// --- LbmContext ---
constexpr uint32_t LBM_SYNC =
    0x1; /**< Synchronous send (wait for completion) */

struct LbmContext {
  uint32_t flags;      /**< Combination of LBM_* flags */
  int timeout_ms;      /**< Timeout in milliseconds (0 = no timeout) */
  char* copy_space = nullptr;                      /**< Shared buffer for chunked transfer */
  ShmTransferInfo* shm_info_ = nullptr;            /**< Transfer info in shared memory */
  int server_pid_ = 0;                             /**< Server PID for SHM liveness check */

  HSHM_CROSS_FUN LbmContext() : flags(0), timeout_ms(0) {}

  HSHM_CROSS_FUN explicit LbmContext(uint32_t f) : flags(f), timeout_ms(0) {}

  HSHM_CROSS_FUN LbmContext(uint32_t f, int timeout) : flags(f), timeout_ms(timeout) {}

  HSHM_CROSS_FUN bool IsSync() const { return (flags & LBM_SYNC) != 0; }
  HSHM_CROSS_FUN bool HasTimeout() const { return timeout_ms > 0; }
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
  ~Transport() = default;

  bool IsServer() const { return mode_ == TransportMode::kServer; }
  bool IsClient() const { return mode_ == TransportMode::kClient; }

  // Shared APIs (both client and server)
  Bulk Expose(const hipc::FullPtr<char>& ptr, size_t data_size, u32 flags);

  template <typename MetaT>
  int Send(MetaT& meta, const LbmContext& ctx = LbmContext());

  template <typename MetaT>
  ClientInfo Recv(MetaT& meta, const LbmContext& ctx = LbmContext());

  // Server-only APIs
  std::string GetAddress() const;

  void ClearRecvHandles(LbmMeta<>& meta);

  // Event registration API
  void RegisterEventManager(EventManager &em);

  // Liveness check
  bool IsServerAlive(const LbmContext& ctx = LbmContext()) const;
};

// --- Transport custom deleter (dispatches via type_ instead of vtable) ---
struct TransportDeleter {
  inline void operator()(Transport* t) const;
};
using TransportPtr = std::unique_ptr<Transport, TransportDeleter>;

// --- Factory ---
class TransportFactory {
 public:
  static TransportPtr Get(const std::string& addr,
                          TransportType t, TransportMode mode,
                          const std::string& protocol = "",
                          int port = 0);
  static TransportPtr Get(const std::string& addr,
                          TransportType t, TransportMode mode,
                          const std::string& protocol, int port,
                          const std::string& domain);
};

}  // namespace hshm::lbm
