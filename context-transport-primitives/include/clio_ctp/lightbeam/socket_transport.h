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
#include <cstring>
#include <stdexcept>
#include <vector>

#include "clio_ctp/util/logging.h"
#include "lightbeam.h"
#include "posix_socket.h"

namespace ctp::lbm {

/** Action that records a fired event into a SocketTransport's vector */
class SocketFiredAction : public EventAction {
 public:
  std::vector<EventInfo> *events_;
  explicit SocketFiredAction(std::vector<EventInfo> *events)
      : events_(events) {}
  void Run(const EventInfo &event) override {
    events_->push_back(event);
  }
};

class SocketTransport : public Transport {
 public:
  explicit SocketTransport(TransportMode mode, const std::string& addr,
                           const std::string& protocol = "tcp", int port = 8193)
      : Transport(mode),
        addr_(addr), protocol_(protocol), port_(port),
        fd_(sock::kInvalidSocket),
        listen_fd_(sock::kInvalidSocket),
        em_(nullptr),
        fired_action_(&fired_events_) {
    type_ = TransportType::kSocket;
    sock::InitSocketLib();

    if (mode == TransportMode::kClient) {
      // Client mode: connect. All socket-API use lives in
      // socket_{win,posix}.cc so this header pulls no platform socket headers.
      fd_ = sock::Connect(addr_, port_, protocol_);
      if (fd_ == sock::kInvalidSocket) {
        throw std::runtime_error(
            "SocketTransport: failed to connect to " + addr_ + ":" +
            std::to_string(port_));
      }
      sock::SetNonBlocking(fd_, true);
      HLOG(kDebug, "SocketTransport(client) connected to {}:{}", addr_, port_);
    } else {
      // Server mode: bind + listen (socket-API use lives in the .cc).
      listen_fd_ = sock::Listen(addr_, port_, protocol_);
      if (listen_fd_ == sock::kInvalidSocket) {
        throw std::runtime_error(
            "SocketTransport: failed to listen on " + addr_ + ":" +
            std::to_string(port_));
      }
      sock::SetNonBlocking(listen_fd_, true);
      HLOG(kDebug, "SocketTransport(server) listening on {}:{}", addr_, port_);
    }
  }

  ~SocketTransport() {
    if (IsClient()) {
      if (em_) em_->RemoveEvent(fd_);
      sock::Close(fd_);
    } else {
      for (auto fd : client_fds_) {
        if (em_) em_->RemoveEvent(fd);
        sock::Close(fd);
      }
      if (em_) em_->RemoveEvent(listen_fd_);
      sock::Close(listen_fd_);
      if (protocol_ == "ipc") {
        sock::UnlinkPath(addr_.c_str());
      }
    }
  }

  Bulk Expose(const ctp::ipc::FullPtr<char>& ptr, size_t data_size,
              u32 flags) {
    Bulk bulk;
    bulk.data = ptr;
    bulk.size = data_size;
    bulk.flags = ctp::bitfield32_t(flags);
    return bulk;
  }

  void ClearRecvHandles(LbmMeta<>& meta) {
    for (auto& bulk : meta.recv) {
      // Only std::free buffers RecvBulks allocated itself (tagged with
      // the (UINT32_MAX-1, UINT32_MAX-1) sentinel). LoadTaskArchive::bulk
      // may have swapped in a CTP MallocAllocator FullPtr (whose user
      // pointer is offset 16 bytes inside the real malloc region) for the
      // BULK_EXPOSE / ZMQ-copy paths — passing that to std::free triggers
      // glibc "free(): invalid pointer" (ASan: bad-free). Those CTP
      // buffers are reclaimed by the task destructor via
      // daemon_allocated_bulk_count_ / TASK_DATA_OWNER instead.
      if (bulk.data.ptr_ &&
          bulk.data.shm_.alloc_id_ ==
              ctp::ipc::AllocatorId(UINT32_MAX - 1, UINT32_MAX - 1)) {
        std::free(bulk.data.ptr_);
        bulk.data.ptr_ = nullptr;
      }
    }
  }

  std::string GetAddress() const { return addr_; }

  /** Check if the server is still alive via a connect probe. The socket API
   *  lives in socket_{win,posix}.cc (no platform socket headers in this file). */
  bool IsServerAlive(const LbmContext& ctx = LbmContext()) const {
    (void)ctx;
    return sock::IsServerAlive(addr_, port_, protocol_);
  }

  void RegisterEventManager(EventManager &em) {
    em_ = &em;
    if (IsClient()) {
      em.AddEvent(fd_, kDefaultReadEvent, &fired_action_);
    } else {
      // listen_fd_: no action — just wakes epoll for new connections
      em.AddEvent(listen_fd_, kDefaultReadEvent, nullptr);
      // client fds: action populates fired_events_ for recv
      for (auto fd : client_fds_) {
        em.AddEvent(fd, kDefaultReadEvent, &fired_action_);
      }
    }
  }

  void UnregisterEventManager() {
    // Detach without RemoveEvent — caller (e.g. RecvZmqClientThread) is
    // about to destroy the EventManager itself; touching em_ here would
    // be UAF. ~SocketTransport will see em_ == nullptr and skip the
    // (now-stale) RemoveEvent calls.
    em_ = nullptr;
  }

  template <typename MetaT>
  int Send(MetaT& meta, const LbmContext& ctx = LbmContext()) {
    // Determine which fd to send on
    sock::socket_t send_fd;
    if (IsClient()) {
      send_fd = fd_;
    } else {
      // Server mode: use client_info_.fd_ set by Recv
      send_fd = meta.client_info_.fd_;
      if (send_fd == sock::kInvalidSocket) {
        HLOG(kError, "SocketTransport::Send(server) - no client_info_.fd_ set");
        return -1;
      }
    }

    // 1. Serialize metadata via GlobalSerialize directly into the buffer
    //    we will hand to writev — no intermediate std::string copy.
    std::vector<char> meta_buf;
    {
      ctp::ipc::GlobalSerialize<std::vector<char>> ar(meta_buf);
      ar(meta);
      ar.Finalize();
    }

    // 2. Build iovec: [4-byte BE length prefix][metadata][bulk0][bulk1]...
    uint32_t meta_len = sock::HostToNet32(static_cast<uint32_t>(meta_buf.size()));

    int iov_count = 2;
    for (size_t i = 0; i < meta.send.size(); ++i) {
      if (meta.send[i].flags.Any(BULK_XFER)) {
        iov_count++;
      }
    }

    std::vector<sock::IoBuffer> iov(iov_count);
    int idx = 0;
    iov[idx].base = &meta_len;
    iov[idx].len = sizeof(meta_len);
    idx++;
    iov[idx].base = meta_buf.data();
    iov[idx].len = meta_buf.size();
    idx++;

    for (size_t i = 0; i < meta.send.size(); ++i) {
      if (!meta.send[i].flags.Any(BULK_XFER)) continue;
      iov[idx].base = meta.send[i].data.ptr_;
      iov[idx].len = meta.send[i].size;
      idx++;
    }

    // 3. Single writev syscall
    ssize_t sent = sock::SendV(send_fd, iov.data(), idx);
    if (sent < 0) {
      HLOG(kError, "SocketTransport::Send - writev failed: {}", sock::GetErrorString());
      return sock::GetError();
    }
    return 0;
  }

  /** Unified Recv: accept, iterate events, recv metadata + bulks */
  template <typename MetaT>
  ClientInfo Recv(MetaT& meta, const LbmContext& ctx = LbmContext()) {
    ClientInfo info;
    if (IsClient()) {
      if (em_) {
        auto it = std::find_if(fired_events_.begin(), fired_events_.end(),
            [this](const EventInfo &e) { return e.trigger_.fd_ == fd_; });
        if (it != fired_events_.end()) {
          fired_events_.erase(it);
        }
      }
      info.rc = RecvAll(fd_, meta, ctx);
      info.fd_ = fd_;
      return info;
    }

    // Server mode: accept new clients (non-blocking)
    AcceptNewClient();

    if (em_) {
      // Iterate fired_events_ (only client fds — listen_fd_ has no action)
      for (auto it = fired_events_.begin(); it != fired_events_.end();) {
        sock::socket_t fd = it->trigger_.fd_;
        info.rc = RecvAll(fd, meta, ctx);
        info.fd_ = fd;
        if (info.rc == EAGAIN) {
          it = fired_events_.erase(it);
          continue;
        }
        if (info.rc != 0) {
          // Drop epoll registration BEFORE close so the kernel can't
          // recycle the fd number and leave a stale entry in
          // EventManager::fd_to_reg_ (which would later trip MOD->ENOENT
          // when a fresh accept() picked the same number).
          em_->RemoveEvent(fd);
          sock::Close(fd);
          client_fds_.erase(
              std::remove(client_fds_.begin(), client_fds_.end(), fd),
              client_fds_.end());
          it = fired_events_.erase(it);
          return info;
        }
        return info;
      }
      // Fall through to poll all client fds directly when fired_events_
      // is exhausted (data may be on sockets that epoll hasn't re-armed yet)
    }

    // Poll all client fds directly
    for (size_t i = 0; i < client_fds_.size(); ++i) {
      sock::socket_t fd = client_fds_[i];
      info.rc = RecvAll(fd, meta, ctx);
      info.fd_ = fd;
      if (info.rc == EAGAIN) continue;
      if (info.rc != 0) {
        if (em_) em_->RemoveEvent(fd);
        sock::Close(fd);
        client_fds_.erase(client_fds_.begin() + i);
        return info;
      }
      return info;
    }
    info.rc = EAGAIN;
    return info;
  }

 private:
  /** Recv metadata + bulks on a specific fd */
  template <typename MetaT>
  int RecvAll(sock::socket_t fd, MetaT& meta, const LbmContext& ctx) {
    int rc = RecvMetadata(fd, meta, ctx);
    if (rc != 0) return rc;
    meta.client_info_.fd_ = fd;
    // Set up recv entries from send descriptors
    for (const auto& send_bulk : meta.send) {
      Bulk recv_bulk;
      recv_bulk.size = send_bulk.size;
      recv_bulk.flags = send_bulk.flags;
      recv_bulk.data = ctp::ipc::FullPtr<char>::GetNull();
      meta.recv.push_back(recv_bulk);
    }
    return RecvBulks(fd, meta, ctx);
  }

  /** Pure recv metadata on a specific fd */
  template <typename MetaT>
  int RecvMetadata(sock::socket_t fd, MetaT& meta, const LbmContext& ctx) {
    (void)ctx;
    uint32_t net_len = 0;
    int rc = sock::RecvExact(fd, reinterpret_cast<char*>(&net_len),
                             sizeof(net_len));
    if (rc == EAGAIN) return EAGAIN;
    if (rc != 0) return -1;

    uint32_t meta_len = sock::NetToHost32(net_len);
    // Recv straight into the deserialize buffer — no string intermediate.
    std::vector<char> meta_buf(meta_len);
    rc = sock::RecvExact(fd, meta_buf.data(), meta_len);
    if (rc != 0) return -1;

    try {
      ctp::ipc::GlobalDeserialize<std::vector<char>> ar(meta_buf);
      ar(meta);
    } catch (const std::exception& e) {
      HLOG(kFatal, "Socket RecvMetadata: Deserialization failed - {} (len={})",
           e.what(), meta_len);
      return -1;
    }
    return 0;
  }

  /** Pure recv bulks on a specific fd */
  template <typename MetaT>
  int RecvBulks(sock::socket_t fd, MetaT& meta, const LbmContext& ctx) {
    (void)ctx;
    for (size_t i = 0; i < meta.recv.size(); ++i) {
      if (!meta.recv[i].flags.Any(BULK_XFER)) continue;

      char* buf = meta.recv[i].data.ptr_;
      bool allocated = false;
      if (!buf) {
        buf = static_cast<char*>(std::malloc(meta.recv[i].size));
        allocated = true;
      }

      int rc;
      while (true) {
        rc = sock::RecvExact(fd, buf, meta.recv[i].size);
        if (rc != EAGAIN) break;
        if (sock::PollRead(fd, 1000) <= 0) {
          rc = -1;
          break;
        }
      }

      if (rc != 0) {
        if (allocated) std::free(buf);
        return sock::GetError();
      }

      if (allocated) {
        meta.recv[i].data.ptr_ = buf;
        // Use a distinct sentinel (UINT32_MAX-1, UINT32_MAX-1) — not
        // AllocatorId::GetNull() (UINT32_MAX, UINT32_MAX) — so
        // LoadTaskArchive::bulk and ClearRecvHandles can distinguish a
        // SocketTransport-owned raw std::malloc'd buffer from a CTP
        // MallocAllocator buffer (whose backend id is also Null but whose
        // ptr_ is offset 16 bytes inside the real malloc region).
        meta.recv[i].data.shm_.alloc_id_ =
            ctp::ipc::AllocatorId(UINT32_MAX - 1, UINT32_MAX - 1);
        meta.recv[i].data.shm_.off_ = reinterpret_cast<size_t>(buf);
      }
    }
    return 0;
  }

  /** Accept a single pending connection (non-blocking, no loop) */
  void AcceptNewClient() {
    if (IsClient()) return;
    sock::socket_t fd = sock::Accept(listen_fd_);
    if (fd == sock::kInvalidSocket) return;
    if (protocol_ != "ipc") {
      sock::SetTcpNoDelay(fd);
    }
    sock::SetRecvBuf(fd, 4 * 1024 * 1024);
    sock::SetNonBlocking(fd, true);
    client_fds_.push_back(fd);
    if (em_) {
      em_->AddEvent(fd, kDefaultReadEvent, &fired_action_);
    }
  }

  /** Remove all fired events for a given fd */
  void RemoveFiredEvent(int fd) {
    fired_events_.erase(
        std::remove_if(fired_events_.begin(), fired_events_.end(),
            [fd](const EventInfo &e) { return e.trigger_.fd_ == fd; }),
        fired_events_.end());
  }

  std::string addr_;
  std::string protocol_;
  int port_;
  sock::socket_t fd_;           // Client mode: connected socket
  sock::socket_t listen_fd_;    // Server mode: listening socket
  std::vector<sock::socket_t> client_fds_;  // Server mode: accepted client fds
  EventManager *em_;
  std::vector<EventInfo> fired_events_;  // Events fired by EventManager
  SocketFiredAction fired_action_;       // Action that populates fired_events_
};

}  // namespace ctp::lbm
