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

#include "clio_ctp/data_structures/serialization/global_serialize.h"

#include "clio_ctp/data_structures/priv/vector.h"
#include "clio_ctp/lightbeam/event_manager.h"
#include "clio_ctp/memory/allocator/allocator.h"
#include "clio_ctp/memory/allocator/malloc_allocator.h"
#include "clio_ctp/types/bitfield.h"

namespace ctp::lbm {

// Forward declaration — full definition in shm_transport.h
struct ShmTransferInfo;
// Forward declaration — full definition in event_manager.h (host-only).
class EventManager;

// --- Bulk Flags ---
#define BULK_EXPOSE \
  BIT_OPT(ctp::u32, 0)                  // Bulk metadata sent, no data transfer
#define BULK_XFER BIT_OPT(ctp::u32, 1)  // Bulk marked for data transmission

// --- Types ---
struct Bulk {
  ctp::ipc::FullPtr<char> data;
  size_t size;
  ctp::bitfield32_t flags;  // BULK_EXPOSE or BULK_XFER
  void* desc = nullptr;      // For RDMA memory registration
  void* mr = nullptr;        // For RDMA memory region handle (fid_mr*)

  /** Serialize bulk descriptor metadata (size and flags only) */
  template <typename Ar>
  CTP_CROSS_FUN void serialize(Ar& ar) {
    ar(size, flags);
  }
};

// --- Client Info (returned by Recv, used by Send for routing) ---
struct ClientInfo {
  int rc = 0;               // Return code (0 = success, EAGAIN = no data, etc.)
  int fd_ = -1;             // Socket fd (SocketTransport server mode)
#if !CTP_IS_GPU
  std::string identity_;    // ZMQ identity (ZeroMqTransport server mode)
#endif
};

// --- Metadata Base Class ---
/**
 * GPU-compatible metadata for lightbeam transports.
 * Uses ctp::priv::vector with a configurable allocator so that
 * bulk descriptor vectors can be managed in GPU-accessible memory.
 *
 * @tparam AllocT Allocator type for bulk descriptor vectors.
 *                Defaults to MallocAllocator for host-side usage.
 */
template <typename AllocT = ctp::ipc::MallocAllocator>
class LbmMeta {
 public:
  using allocator_type = AllocT;
  using BulkVector = ctp::priv::vector<Bulk, AllocT>;

  BulkVector
      send;  // Sender's bulk descriptors (can have BULK_EXPOSE or BULK_XFER)
  BulkVector
      recv;  // Receiver's bulk descriptors (copy of send with local pointers)
  size_t send_bulks = 0;  // Count of BULK_XFER entries in send vector
  size_t recv_bulks = 0;  // Count of BULK_XFER entries in recv vector
  AllocT* alloc_;          // Allocator used for internal vectors
#if !CTP_IS_GPU
  ClientInfo client_info_;  // Client routing info (not serialized, host-only)
#endif

  /** Default constructor (uses CTP_MALLOC on host, nullptr on GPU) */
  CTP_CROSS_FUN LbmMeta()
#if CTP_IS_HOST
      : send(CTP_MALLOC), recv(CTP_MALLOC), alloc_(CTP_MALLOC) {}
#else
      : send(nullptr), recv(nullptr), alloc_(nullptr) {}
#endif

  /** Constructor with custom allocator */
  CTP_CROSS_FUN explicit LbmMeta(AllocT* alloc)
      : send(alloc), recv(alloc), alloc_(alloc) {}

  /** Move constructor */
  CTP_CROSS_FUN LbmMeta(LbmMeta&& other) noexcept
      : send(std::move(other.send)),
        recv(std::move(other.recv)),
        send_bulks(other.send_bulks),
        recv_bulks(other.recv_bulks),
        alloc_(other.alloc_) {}

  /** Move assignment operator */
  CTP_CROSS_FUN LbmMeta& operator=(LbmMeta&& other) noexcept {
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
  CTP_CROSS_FUN void serialize(Ar& ar) {
    ar(send, recv, send_bulks, recv_bulks);
  }
};

// --- LbmContext ---
// LBM_SYNC is now a no-op flag retained only so callers that still pass
// LbmContext(LBM_SYNC) keep compiling.  Send is unconditionally
// synchronous; the previous async-completion-callback path was removed.
constexpr uint32_t LBM_SYNC = 0x1;

struct LbmContext {
  uint32_t flags;      /**< Combination of LBM_* flags */
  int timeout_ms;      /**< Timeout in milliseconds (0 = no timeout) */
  char* copy_space = nullptr;                      /**< Shared buffer for chunked transfer */
  ShmTransferInfo* shm_info_ = nullptr;            /**< Transfer info in shared memory */
  char* meta_buf_ = nullptr;                       /**< Pre-allocated buffer for ShmTransport framing (avoids heap alloc) */
  size_t meta_buf_size_ = 0;                       /**< Capacity of meta_buf_ */
  bool warp_parallel_ = false;                     /**< True = all 32 lanes participate in copy */
  int server_pid_ = 0;                             /**< Server PID for SHM liveness check */
  int dst_fd_ = -1;                                /**< Destination file descriptor for CPU→storage (-1 = none) */
  size_t dst_offset_ = 0;                          /**< Offset within destination file for CPU→storage */
  int signal_pid_ = 0;                             /**< Waiter PID for Send to EventManager::Signal (0 = none) */
  int signal_tid_ = 0;                             /**< Waiter TID for Send to EventManager::Signal */
  EventManager* event_manager_ = nullptr;          /**< Waiter's own EventManager (from IpcManager::GetTls) for Recv to block on; null on the runtime/Send side */

  CTP_CROSS_FUN LbmContext() : flags(0), timeout_ms(0) {}

  CTP_CROSS_FUN explicit LbmContext(uint32_t f) : flags(f), timeout_ms(0) {}

  CTP_CROSS_FUN LbmContext(uint32_t f, int timeout) : flags(f), timeout_ms(timeout) {}

  /** Construct context for CPU→storage transfers via file descriptor. */
  CTP_CROSS_FUN LbmContext(uint32_t f, int timeout, int dst_fd, size_t dst_offset)
      : flags(f), timeout_ms(timeout), dst_fd_(dst_fd), dst_offset_(dst_offset) {}

  CTP_CROSS_FUN bool IsSync() const { return (flags & LBM_SYNC) != 0; }
  CTP_CROSS_FUN bool HasTimeout() const { return timeout_ms > 0; }
  /** Returns true if this context targets a file descriptor destination. */
  CTP_CROSS_FUN bool HasFileDst() const { return dst_fd_ >= 0; }
};

// --- Transport Type Enum ---
enum class TransportType { kZeroMq, kSocket, kShm, kNixl, kThallium };

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
  Bulk Expose(const ctp::ipc::FullPtr<char>& ptr, size_t data_size, u32 flags);

  template <typename MetaT>
  int Send(MetaT& meta, const LbmContext& ctx = LbmContext());

  template <typename MetaT>
  ClientInfo Recv(MetaT& meta, const LbmContext& ctx = LbmContext());

  // Server-only APIs
  std::string GetAddress() const;

  void ClearRecvHandles(LbmMeta<>& meta);

  // Event registration API
  void RegisterEventManager(EventManager &em);
  void UnregisterEventManager();

  // Block up to timeout_ms until readable, using the transport's native
  // readiness primitive (ZMQ: zmq_poll). Non-ZMQ transports return 0 — they
  // use the EventManager path instead. Used by recv threads to block instead
  // of spin-polling.
  int PollRecv(int timeout_ms);

  // Actual bound TCP port for a server socket (resolves ephemeral port-0
  // binds via ZMQ_LAST_ENDPOINT). Returns 0 for clients / non-TCP / non-ZMQ.
  int GetBoundPort() const;

  // Liveness check
  bool IsServerAlive(const LbmContext& ctx = LbmContext()) const;
};

// --- Transport custom deleter (dispatches via type_ instead of vtable) ---
struct TransportDeleter {
  void operator()(Transport* t) const;
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

}  // namespace ctp::lbm
