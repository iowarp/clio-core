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

#include <cstring>
#include <stdexcept>
#include <vector>

#include "hermes_shm/util/logging.h"
#include "lightbeam.h"
#include "posix_socket.h"

namespace hshm::lbm {

class SocketClient : public Client {
 public:
  explicit SocketClient(const std::string& addr,
                        const std::string& protocol = "tcp", int port = 8193)
      : addr_(addr), protocol_(protocol), port_(port),
        fd_(sock::kInvalidSocket), epoll_fd_(-1) {
    type_ = Transport::kSocket;

    if (protocol_ == "ipc") {
      // Unix domain socket
      fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
      if (fd_ == sock::kInvalidSocket) {
        throw std::runtime_error("SocketClient: failed to create Unix socket");
      }
      struct sockaddr_un sun;
      std::memset(&sun, 0, sizeof(sun));
      sun.sun_family = AF_UNIX;
      std::strncpy(sun.sun_path, addr_.c_str(), sizeof(sun.sun_path) - 1);
      if (::connect(fd_, reinterpret_cast<struct sockaddr*>(&sun),
                    sizeof(sun)) < 0) {
        sock::Close(fd_);
        throw std::runtime_error("SocketClient: failed to connect to Unix socket " + addr_);
      }
    } else {
      // TCP socket
      fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
      if (fd_ == sock::kInvalidSocket) {
        throw std::runtime_error("SocketClient: failed to create TCP socket");
      }
      sock::SetTcpNoDelay(fd_);
      sock::SetSendBuf(fd_, 4 * 1024 * 1024);

      struct sockaddr_in sin;
      std::memset(&sin, 0, sizeof(sin));
      sin.sin_family = AF_INET;
      sin.sin_port = htons(static_cast<uint16_t>(port_));
      if (::inet_pton(AF_INET, addr_.c_str(), &sin.sin_addr) <= 0) {
        sock::Close(fd_);
        throw std::runtime_error("SocketClient: invalid address " + addr_);
      }
      if (::connect(fd_, reinterpret_cast<struct sockaddr*>(&sin),
                    sizeof(sin)) < 0) {
        sock::Close(fd_);
        throw std::runtime_error(
            "SocketClient: failed to connect to " + addr_ + ":" +
            std::to_string(port_));
      }
    }

    HLOG(kDebug, "SocketClient connected to {}:{}", addr_, port_);
  }

  ~SocketClient() override {
    sock::Close(fd_);
  }

  void PollConnect(int epoll_fd) override {
    epoll_fd_ = epoll_fd;
    sock::EpollAdd(epoll_fd_, fd_);
  }

  void PollWait(int timeout_ms = 10) override {
    if (epoll_fd_ < 0) return;
    struct epoll_event events[4];
    sock::EpollWait(epoll_fd_, events, 4, timeout_ms);
  }

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
    // 1. Serialize metadata via cereal
    std::ostringstream oss(std::ios::binary);
    {
      cereal::BinaryOutputArchive ar(oss);
      ar(meta);
    }
    std::string meta_str = oss.str();

    // 2. Build iovec: [4-byte BE length prefix][metadata][bulk0][bulk1]...
    uint32_t meta_len = htonl(static_cast<uint32_t>(meta_str.size()));

    // Count iovecs: length prefix + metadata + bulks
    int iov_count = 2;  // length prefix + metadata
    for (size_t i = 0; i < meta.send.size(); ++i) {
      if (meta.send[i].flags.Any(BULK_XFER)) {
        iov_count++;
      }
    }

    std::vector<struct iovec> iov(iov_count);
    int idx = 0;
    iov[idx].iov_base = &meta_len;
    iov[idx].iov_len = sizeof(meta_len);
    idx++;
    iov[idx].iov_base = const_cast<char*>(meta_str.data());
    iov[idx].iov_len = meta_str.size();
    idx++;

    for (size_t i = 0; i < meta.send.size(); ++i) {
      if (!meta.send[i].flags.Any(BULK_XFER)) continue;
      iov[idx].iov_base = meta.send[i].data.ptr_;
      iov[idx].iov_len = meta.send[i].size;
      idx++;
    }

    // 3. Single writev syscall
    ssize_t sent = sock::SendV(fd_, iov.data(), idx);
    if (sent < 0) {
      HLOG(kError, "SocketClient::Send - writev failed: {}", strerror(errno));
      return errno;
    }
    return 0;
  }

 private:
  std::string addr_;
  std::string protocol_;
  int port_;
  sock::socket_t fd_;
  int epoll_fd_;
};

class SocketServer : public Server {
 public:
  explicit SocketServer(const std::string& addr,
                        const std::string& protocol = "tcp", int port = 8193)
      : addr_(addr), protocol_(protocol), port_(port),
        listen_fd_(sock::kInvalidSocket),
        last_recv_fd_(sock::kInvalidSocket),
        epoll_fd_(-1) {
    type_ = Transport::kSocket;

    if (protocol_ == "ipc") {
      // Remove stale socket file
      ::unlink(addr_.c_str());
      listen_fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
      if (listen_fd_ == sock::kInvalidSocket) {
        throw std::runtime_error("SocketServer: failed to create Unix socket");
      }
      struct sockaddr_un sun;
      std::memset(&sun, 0, sizeof(sun));
      sun.sun_family = AF_UNIX;
      std::strncpy(sun.sun_path, addr_.c_str(), sizeof(sun.sun_path) - 1);
      if (::bind(listen_fd_, reinterpret_cast<struct sockaddr*>(&sun),
                 sizeof(sun)) < 0) {
        sock::Close(listen_fd_);
        throw std::runtime_error("SocketServer: failed to bind Unix socket " + addr_);
      }
    } else {
      listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
      if (listen_fd_ == sock::kInvalidSocket) {
        throw std::runtime_error("SocketServer: failed to create TCP socket");
      }
      sock::SetReuseAddr(listen_fd_);
      sock::SetRecvBuf(listen_fd_, 4 * 1024 * 1024);

      struct sockaddr_in sin;
      std::memset(&sin, 0, sizeof(sin));
      sin.sin_family = AF_INET;
      sin.sin_port = htons(static_cast<uint16_t>(port_));
      sin.sin_addr.s_addr = INADDR_ANY;
      if (::bind(listen_fd_, reinterpret_cast<struct sockaddr*>(&sin),
                 sizeof(sin)) < 0) {
        sock::Close(listen_fd_);
        throw std::runtime_error(
            "SocketServer: failed to bind to port " + std::to_string(port_));
      }
    }

    if (::listen(listen_fd_, 16) < 0) {
      sock::Close(listen_fd_);
      throw std::runtime_error("SocketServer: listen failed");
    }

    // Set listen socket non-blocking for AcceptPending
    sock::SetNonBlocking(listen_fd_, true);

    HLOG(kDebug, "SocketServer listening on {}:{}", addr_, port_);
  }

  ~SocketServer() override {
    for (auto fd : client_fds_) {
      sock::Close(fd);
    }
    sock::Close(listen_fd_);
    if (protocol_ == "ipc") {
      ::unlink(addr_.c_str());
    }
  }

  Bulk Expose(const hipc::FullPtr<char>& ptr, size_t data_size,
              u32 flags) override {
    Bulk bulk;
    bulk.data = ptr;
    bulk.size = data_size;
    bulk.flags = hshm::bitfield32_t(flags);
    return bulk;
  }

  void ClearRecvHandles(LbmMeta& meta) override {
    for (auto& bulk : meta.recv) {
      if (bulk.data.ptr_) {
        std::free(bulk.data.ptr_);
        bulk.data.ptr_ = nullptr;
      }
    }
  }

  std::string GetAddress() const override { return addr_; }

  int GetFd() const override { return listen_fd_; }

  void PollConnect(int epoll_fd) override {
    epoll_fd_ = epoll_fd;
    sock::EpollAdd(epoll_fd_, listen_fd_);
    for (auto fd : client_fds_) {
      sock::EpollAdd(epoll_fd_, fd);
    }
  }

  void PollWait(int timeout_ms = 10) override {
    if (epoll_fd_ < 0) return;
    struct epoll_event events[16];
    sock::EpollWait(epoll_fd_, events, 16, timeout_ms);
  }

  std::vector<ClientInfo> AcceptNewClients() override {
    std::vector<ClientInfo> new_clients;
    while (true) {
      sock::socket_t fd = ::accept(listen_fd_, nullptr, nullptr);
      if (fd == sock::kInvalidSocket) break;
      if (protocol_ != "ipc") {
        sock::SetTcpNoDelay(fd);
      }
      sock::SetRecvBuf(fd, 4 * 1024 * 1024);
      sock::SetNonBlocking(fd, true);
      client_fds_.push_back(fd);
      if (epoll_fd_ >= 0) {
        sock::EpollAdd(epoll_fd_, fd);
      }
      new_clients.push_back(ClientInfo{fd});
    }
    return new_clients;
  }

  template <typename MetaT>
  int RecvMetadata(MetaT& meta, const LbmContext& ctx = LbmContext()) {
    (void)ctx;
    // Accept any pending connections (needed for standalone unit tests)
    AcceptPending();

    if (client_fds_.empty()) {
      return EAGAIN;
    }

    // Try recv directly on each non-blocking client fd (no poll() needed)
    for (size_t i = 0; i < client_fds_.size(); ++i) {
      sock::socket_t fd = client_fds_[i];

      // Read 4-byte BE length prefix (non-blocking)
      uint32_t net_len = 0;
      int rc = sock::RecvExact(fd, reinterpret_cast<char*>(&net_len),
                               sizeof(net_len));
      if (rc == EAGAIN) continue;  // No data on this fd, try next
      if (rc != 0) {
        // Client disconnected or error — remove from list
        sock::Close(fd);
        client_fds_.erase(client_fds_.begin() + i);
        return EAGAIN;
      }
      uint32_t meta_len = ntohl(net_len);

      // Read metadata bytes (may poll internally for partial reads)
      std::string meta_str(meta_len, '\0');
      rc = sock::RecvExact(fd, &meta_str[0], meta_len);
      if (rc != 0) {
        sock::Close(fd);
        client_fds_.erase(client_fds_.begin() + i);
        return -1;
      }

      // Deserialize
      try {
        std::istringstream iss(meta_str, std::ios::binary);
        cereal::BinaryInputArchive ar(iss);
        ar(meta);
      } catch (const std::exception& e) {
        HLOG(kFatal, "Socket RecvMetadata: Deserialization failed - {} (len={})",
             e.what(), meta_len);
        return -1;
      }

      last_recv_fd_ = fd;
      return 0;
    }
    return EAGAIN;
  }

  template <typename MetaT>
  int RecvBulks(MetaT& meta, const LbmContext& ctx = LbmContext()) {
    (void)ctx;
    for (size_t i = 0; i < meta.recv.size(); ++i) {
      if (!meta.recv[i].flags.Any(BULK_XFER)) continue;

      char* buf = meta.recv[i].data.ptr_;
      bool allocated = false;
      if (!buf) {
        buf = static_cast<char*>(std::malloc(meta.recv[i].size));
        allocated = true;
      }

      // Bulk data follows metadata on the same stream — retry on EAGAIN
      int rc;
      while (true) {
        rc = sock::RecvExact(last_recv_fd_, buf, meta.recv[i].size);
        if (rc != EAGAIN) break;
        if (sock::PollRead(last_recv_fd_, 1000) <= 0) {
          rc = -1;
          break;
        }
      }

      if (rc != 0) {
        if (allocated) std::free(buf);
        return errno;
      }

      if (allocated) {
        meta.recv[i].data.ptr_ = buf;
        meta.recv[i].data.shm_.alloc_id_ = hipc::AllocatorId::GetNull();
        meta.recv[i].data.shm_.off_ = reinterpret_cast<size_t>(buf);
      }
    }
    return 0;
  }

 private:
  void AcceptPending() {
    AcceptNewClients();
  }

  std::string addr_;
  std::string protocol_;
  int port_;
  sock::socket_t listen_fd_;
  std::vector<sock::socket_t> client_fds_;
  sock::socket_t last_recv_fd_;
  int epoll_fd_;
};

}  // namespace hshm::lbm
