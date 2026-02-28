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
#if HSHM_ENABLE_LIGHTBEAM
#include "lightbeam.h"
#include "shm_transport.h"
#include "socket_transport.h"
#if HSHM_ENABLE_ZMQ
#include "zmq_transport.h"
#endif
#if HSHM_ENABLE_THALLIUM
#include "thallium_transport.h"
#endif
#if HSHM_ENABLE_LIBFABRIC
#include "libfabric_transport.h"
#endif

namespace hshm::lbm {

#if HSHM_IS_HOST
// --- TransportDeleter implementation ---
inline void TransportDeleter::operator()(Transport* t) const {
  if (!t) return;
  switch (t->type_) {
#if HSHM_ENABLE_ZMQ
    case TransportType::kZeroMq:
      delete static_cast<ZeroMqTransport*>(t);
      break;
#endif
    case TransportType::kSocket:
      delete static_cast<SocketTransport*>(t);
      break;
    case TransportType::kShm:
      delete static_cast<ShmTransport*>(t);
      break;
    default:
      delete t;
      break;
  }
}

// --- Unified Transport Non-Template Dispatch ---
inline Bulk Transport::Expose(const hipc::FullPtr<char>& ptr, size_t data_size, u32 flags) {
  switch (type_) {
#if HSHM_ENABLE_ZMQ
    case TransportType::kZeroMq:
      return static_cast<ZeroMqTransport*>(this)->Expose(ptr, data_size, flags);
#endif
    case TransportType::kSocket:
      return static_cast<SocketTransport*>(this)->Expose(ptr, data_size, flags);
    case TransportType::kShm:
      return static_cast<ShmTransport*>(this)->Expose(ptr, data_size, flags);
    default:
      return Bulk{};
  }
}

inline std::string Transport::GetAddress() const {
  switch (type_) {
#if HSHM_ENABLE_ZMQ
    case TransportType::kZeroMq:
      return static_cast<const ZeroMqTransport*>(this)->GetAddress();
#endif
    case TransportType::kSocket:
      return static_cast<const SocketTransport*>(this)->GetAddress();
    case TransportType::kShm:
      return static_cast<const ShmTransport*>(this)->GetAddress();
    default:
      return "";
  }
}

inline void Transport::ClearRecvHandles(LbmMeta<>& meta) {
  switch (type_) {
#if HSHM_ENABLE_ZMQ
    case TransportType::kZeroMq:
      static_cast<ZeroMqTransport*>(this)->ClearRecvHandles(meta);
      break;
#endif
    case TransportType::kSocket:
      static_cast<SocketTransport*>(this)->ClearRecvHandles(meta);
      break;
    case TransportType::kShm:
      static_cast<ShmTransport*>(this)->ClearRecvHandles(meta);
      break;
    default:
      break;
  }
}

inline void Transport::RegisterEventManager(EventManager &em) {
  switch (type_) {
#if HSHM_ENABLE_ZMQ
    case TransportType::kZeroMq:
      static_cast<ZeroMqTransport*>(this)->RegisterEventManager(em);
      break;
#endif
    case TransportType::kSocket:
      static_cast<SocketTransport*>(this)->RegisterEventManager(em);
      break;
    case TransportType::kShm:
      /* no-op for SHM */
      break;
    default:
      break;
  }
}

inline bool Transport::IsServerAlive(const LbmContext& ctx) const {
  switch (type_) {
#if HSHM_ENABLE_ZMQ
    case TransportType::kZeroMq:
      return static_cast<const ZeroMqTransport*>(this)->IsServerAlive(ctx);
#endif
    case TransportType::kSocket:
      return static_cast<const SocketTransport*>(this)->IsServerAlive(ctx);
    case TransportType::kShm:
      return static_cast<const ShmTransport*>(this)->IsServerAlive(ctx);
    default:
      return false;
  }
}
#endif  // HSHM_IS_HOST

#if HSHM_IS_HOST
// --- Unified Transport Template Dispatch ---
template <typename MetaT>
int Transport::Send(MetaT& meta, const LbmContext& ctx) {
  switch (type_) {
#if HSHM_ENABLE_ZMQ
    case TransportType::kZeroMq:
      return static_cast<ZeroMqTransport*>(this)->Send(meta, ctx);
#endif
    case TransportType::kSocket:
      return static_cast<SocketTransport*>(this)->Send(meta, ctx);
    case TransportType::kShm:
      return static_cast<ShmTransport*>(this)->Send(meta, ctx);
    default:
      return -1;
  }
}

template <typename MetaT>
ClientInfo Transport::Recv(MetaT& meta, const LbmContext& ctx) {
  switch (type_) {
#if HSHM_ENABLE_ZMQ
    case TransportType::kZeroMq:
      return static_cast<ZeroMqTransport*>(this)->Recv(meta, ctx);
#endif
    case TransportType::kSocket:
      return static_cast<SocketTransport*>(this)->Recv(meta, ctx);
    case TransportType::kShm:
      return static_cast<ShmTransport*>(this)->Recv(meta, ctx);
    default:
      return ClientInfo{-1, -1, {}};
  }
}

// --- TransportFactory Implementations ---
inline TransportPtr TransportFactory::Get(
    const std::string& addr, TransportType t, TransportMode mode,
    const std::string& protocol, int port) {
  switch (t) {
#if HSHM_ENABLE_ZMQ
    case TransportType::kZeroMq:
      return TransportPtr(new ZeroMqTransport(
          mode, addr, protocol.empty() ? "tcp" : protocol,
          port == 0 ? 8192 : port));
#endif
    case TransportType::kSocket:
      return TransportPtr(new SocketTransport(
          mode, addr, protocol.empty() ? "tcp" : protocol,
          port == 0 ? 8193 : port));
    case TransportType::kShm:
      return TransportPtr(new ShmTransport(mode));
    default:
      return nullptr;
  }
}

inline TransportPtr TransportFactory::Get(
    const std::string& addr, TransportType t, TransportMode mode,
    const std::string& protocol, int port, const std::string& domain) {
  (void)domain;
  switch (t) {
#if HSHM_ENABLE_ZMQ
    case TransportType::kZeroMq:
      return TransportPtr(new ZeroMqTransport(
          mode, addr, protocol.empty() ? "tcp" : protocol,
          port == 0 ? 8192 : port));
#endif
    case TransportType::kSocket:
      return TransportPtr(new SocketTransport(
          mode, addr, protocol.empty() ? "tcp" : protocol,
          port == 0 ? 8193 : port));
    case TransportType::kShm:
      return TransportPtr(new ShmTransport(mode));
    default:
      return nullptr;
  }
}
#endif  // HSHM_IS_HOST

}  // namespace hshm::lbm
#endif  // HSHM_ENABLE_LIGHTBEAM
