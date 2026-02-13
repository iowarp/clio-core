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

// --- Base Class Template Dispatch ---
template <typename MetaT>
int Client::Send(MetaT& meta, const LbmContext& ctx) {
  switch (type_) {
#if HSHM_ENABLE_ZMQ
    case Transport::kZeroMq:
      return static_cast<ZeroMqClient*>(this)->Send(meta, ctx);
#endif
    case Transport::kSocket:
      return static_cast<SocketClient*>(this)->Send(meta, ctx);
    case Transport::kShm:
      return static_cast<ShmClient*>(this)->Send(meta, ctx);
    default:
      return -1;
  }
}

template <typename MetaT>
int Server::RecvMetadata(MetaT& meta, const LbmContext& ctx) {
  switch (type_) {
#if HSHM_ENABLE_ZMQ
    case Transport::kZeroMq:
      return static_cast<ZeroMqServer*>(this)->RecvMetadata(meta, ctx);
#endif
    case Transport::kSocket:
      return static_cast<SocketServer*>(this)->RecvMetadata(meta, ctx);
    case Transport::kShm:
      return static_cast<ShmServer*>(this)->RecvMetadata(meta, ctx);
    default:
      return -1;
  }
}

template <typename MetaT>
int Server::RecvBulks(MetaT& meta, const LbmContext& ctx) {
  switch (type_) {
#if HSHM_ENABLE_ZMQ
    case Transport::kZeroMq:
      return static_cast<ZeroMqServer*>(this)->RecvBulks(meta, ctx);
#endif
    case Transport::kSocket:
      return static_cast<SocketServer*>(this)->RecvBulks(meta, ctx);
    case Transport::kShm:
      return static_cast<ShmServer*>(this)->RecvBulks(meta, ctx);
    default:
      return -1;
  }
}

// --- TransportFactory Implementations ---
inline std::unique_ptr<Client> TransportFactory::GetClient(
    const std::string& addr, Transport t, const std::string& protocol,
    int port) {
  switch (t) {
#if HSHM_ENABLE_ZMQ
    case Transport::kZeroMq:
      return std::make_unique<ZeroMqClient>(
          addr, protocol.empty() ? "tcp" : protocol, port == 0 ? 8192 : port);
#endif
    case Transport::kSocket:
      return std::make_unique<SocketClient>(
          addr, protocol.empty() ? "tcp" : protocol, port == 0 ? 8193 : port);
    case Transport::kShm:
      return std::make_unique<ShmClient>();
#if HSHM_ENABLE_THALLIUM
    case Transport::kThallium:
      return std::make_unique<ThalliumClient>(
          addr, protocol.empty() ? "ofi+sockets" : protocol,
          port == 0 ? 8200 : port);
#endif
#if HSHM_ENABLE_LIBFABRIC
    case Transport::kLibfabric:
      return std::make_unique<LibfabricClient>(
          addr, protocol.empty() ? "tcp" : protocol, port == 0 ? 9222 : port);
#endif
    default:
      return nullptr;
  }
}

inline std::unique_ptr<Client> TransportFactory::GetClient(
    const std::string& addr, Transport t, const std::string& protocol, int port,
    const std::string& domain) {
  switch (t) {
#if HSHM_ENABLE_ZMQ
    case Transport::kZeroMq:
      return std::make_unique<ZeroMqClient>(
          addr, protocol.empty() ? "tcp" : protocol, port == 0 ? 8192 : port);
#endif
    case Transport::kSocket:
      return std::make_unique<SocketClient>(
          addr, protocol.empty() ? "tcp" : protocol, port == 0 ? 8193 : port);
    case Transport::kShm:
      return std::make_unique<ShmClient>();
#if HSHM_ENABLE_THALLIUM
    case Transport::kThallium:
      return std::make_unique<ThalliumClient>(
          addr, protocol.empty() ? "ofi+sockets" : protocol,
          port == 0 ? 8200 : port, domain);
#endif
#if HSHM_ENABLE_LIBFABRIC
    case Transport::kLibfabric:
      return std::make_unique<LibfabricClient>(
          addr, protocol.empty() ? "tcp" : protocol, port == 0 ? 9222 : port);
#endif
    default:
      return nullptr;
  }
}

inline std::unique_ptr<Server> TransportFactory::GetServer(
    const std::string& addr, Transport t, const std::string& protocol,
    int port) {
  switch (t) {
#if HSHM_ENABLE_ZMQ
    case Transport::kZeroMq:
      return std::make_unique<ZeroMqServer>(
          addr, protocol.empty() ? "tcp" : protocol, port == 0 ? 8192 : port);
#endif
    case Transport::kSocket:
      return std::make_unique<SocketServer>(
          addr, protocol.empty() ? "tcp" : protocol, port == 0 ? 8193 : port);
    case Transport::kShm:
      return std::make_unique<ShmServer>();
#if HSHM_ENABLE_THALLIUM
    case Transport::kThallium:
      return std::make_unique<ThalliumServer>(
          addr, protocol.empty() ? "ofi+sockets" : protocol,
          port == 0 ? 8200 : port);
#endif
#if HSHM_ENABLE_LIBFABRIC
    case Transport::kLibfabric:
      return std::make_unique<LibfabricServer>(
          addr, protocol.empty() ? "tcp" : protocol, port == 0 ? 9222 : port);
#endif
    default:
      return nullptr;
  }
}

inline std::unique_ptr<Server> TransportFactory::GetServer(
    const std::string& addr, Transport t, const std::string& protocol, int port,
    const std::string& domain) {
  switch (t) {
#if HSHM_ENABLE_ZMQ
    case Transport::kZeroMq:
      return std::make_unique<ZeroMqServer>(
          addr, protocol.empty() ? "tcp" : protocol, port == 0 ? 8192 : port);
#endif
    case Transport::kSocket:
      return std::make_unique<SocketServer>(
          addr, protocol.empty() ? "tcp" : protocol, port == 0 ? 8193 : port);
    case Transport::kShm:
      return std::make_unique<ShmServer>();
#if HSHM_ENABLE_THALLIUM
    case Transport::kThallium:
      return std::make_unique<ThalliumServer>(
          addr, protocol.empty() ? "ofi+sockets" : protocol,
          port == 0 ? 8200 : port, domain);
#endif
#if HSHM_ENABLE_LIBFABRIC
    case Transport::kLibfabric:
      return std::make_unique<LibfabricServer>(
          addr, protocol.empty() ? "tcp" : protocol, port == 0 ? 9222 : port);
#endif
    default:
      return nullptr;
  }
}

}  // namespace hshm::lbm
#endif  // HSHM_ENABLE_LIGHTBEAM
