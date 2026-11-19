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

#ifndef _WIN32

#include "clio_ctp/lightbeam/posix_socket.h"

#include <netdb.h>

#include <atomic>
#include <cerrno>
#include <cstring>
#include <mutex>
#include <unordered_map>

// Linux suppresses SIGPIPE per-write via this send() flag; macOS/BSD lack it
// and use the SO_NOSIGPIPE socket option (see SetNoSigPipe) instead, so define
// it to 0 there to keep SendV() portable.
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

namespace ctp::lbm::sock {

void InitSocketLib() {
  // No-op on POSIX
}

void CleanupSocketLib() {
  // No-op on POSIX
}

void SetSocketLibShutdown() {
  // No-op on POSIX: zmq sockets close cleanly there (no WSASTARTUP teardown
  // hazard), so there is nothing to leak.
}

bool IsSocketLibShutdown() { return false; }

void Close(socket_t fd) {
  if (fd != kInvalidSocket) {
    ::close(fd);
  }
}

int GetError() {
  return errno;
}

std::string GetErrorString() {
  return std::string(strerror(errno));
}

void SetNonBlocking(socket_t fd, bool enable) {
  int flags = ::fcntl(fd, F_GETFL, 0);
  if (enable) {
    ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);
  } else {
    ::fcntl(fd, F_SETFL, flags & ~O_NONBLOCK);
  }
}

void SetTcpNoDelay(socket_t fd) {
  int flag = 1;
  ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
}

void SetReuseAddr(socket_t fd) {
  int flag = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &flag, sizeof(flag));
}

void SetSendBuf(socket_t fd, int size) {
  ::setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &size, sizeof(size));
}

void SetRecvBuf(socket_t fd, int size) {
  ::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &size, sizeof(size));
}

void SetNoSigPipe(socket_t fd) {
#ifdef SO_NOSIGPIPE
  // macOS / *BSD: make writes on this fd fail with EPIPE rather than raise
  // SIGPIPE (whose default disposition terminates the process) when the peer
  // has closed. Linux has no SO_NOSIGPIPE and relies on MSG_NOSIGNAL in
  // SendV() instead.
  int on = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &on, sizeof(on));
#else
  (void)fd;
#endif
}

void UnlinkPath(const char* path) {
  ::unlink(path);
}

ssize_t SendV(socket_t fd, const IoBuffer* iov, int count) {
  ssize_t total = 0;
  for (int i = 0; i < count; ++i) {
    total += static_cast<ssize_t>(iov[i].len);
  }

  // Convert IoBuffer to iovec for writev
  struct iovec local_iov[64];
  int local_count = count < 64 ? count : 64;
  for (int i = 0; i < local_count; ++i) {
    local_iov[i].iov_base = iov[i].base;
    local_iov[i].iov_len = iov[i].len;
  }

  ssize_t sent = 0;
  int iov_idx = 0;

  while (sent < total) {
    // sendmsg (vs writev) lets us pass MSG_NOSIGNAL so a write to a peer that
    // has closed returns EPIPE instead of raising SIGPIPE on Linux; macOS/BSD
    // get the same guarantee from SO_NOSIGPIPE set in SetNoSigPipe().
    struct msghdr msg;
    std::memset(&msg, 0, sizeof(msg));
    msg.msg_iov = local_iov + iov_idx;
    msg.msg_iovlen = local_count - iov_idx;
    ssize_t n = ::sendmsg(fd, &msg, MSG_NOSIGNAL);
    if (n < 0) {
      if (errno == EINTR) continue;
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        struct pollfd pfd;
        pfd.fd = fd;
        pfd.events = POLLOUT;
        pfd.revents = 0;
        int pr = ::poll(&pfd, 1, 1000);
        if (pr <= 0) return -1;
        continue;
      }
      return -1;
    }
    sent += n;
    while (iov_idx < local_count && n >= static_cast<ssize_t>(local_iov[iov_idx].iov_len)) {
      n -= static_cast<ssize_t>(local_iov[iov_idx].iov_len);
      iov_idx++;
    }
    if (iov_idx < local_count && n > 0) {
      local_iov[iov_idx].iov_base =
          static_cast<char*>(local_iov[iov_idx].iov_base) + n;
      local_iov[iov_idx].iov_len -= n;
    }
  }
  return sent;
}

int RecvExact(socket_t fd, char* buf, size_t len) {
  size_t received = 0;
  while (received < len) {
    ssize_t n = ::recv(fd, buf + received, len - received, 0);
    if (n < 0) {
      if (errno == EINTR) continue;
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        if (received == 0) return EAGAIN;
        if (PollRead(fd, 1000) <= 0) return -1;
        continue;
      }
      return -1;
    }
    if (n == 0) {
      return -1;
    }
    received += static_cast<size_t>(n);
  }
  return 0;
}

int PollRead(socket_t fd, int timeout_ms) {
  struct pollfd pfd;
  pfd.fd = fd;
  pfd.events = POLLIN;
  pfd.revents = 0;
  return ::poll(&pfd, 1, timeout_ms);
}

int PollReadMulti(const socket_t* fds, int count, int timeout_ms) {
  struct pollfd pfds[128];
  int n = count < 128 ? count : 128;
  for (int i = 0; i < n; ++i) {
    pfds[i].fd = fds[i];
    pfds[i].events = POLLIN;
    pfds[i].revents = 0;
  }
  int rc = ::poll(pfds, n, timeout_ms);
  if (rc <= 0) return -1;
  for (int i = 0; i < n; ++i) {
    if (pfds[i].revents & POLLIN) {
      return i;
    }
  }
  return -1;
}

/** Per-process cache of host name -> IPv4 address for the connect path. */
struct HostCache {
  std::mutex mtx_;                                    /**< guards map_ */
  std::unordered_map<std::string, struct in_addr> map_; /**< resolved hosts */
  std::atomic<uint64_t> lookups_{0};                  /**< getaddrinfo calls */
};

/** @return The process-wide host cache (never destroyed: dials may run
 *  from static destructors at exit). */
static HostCache& GetHostCache() {
  static HostCache* cache = new HostCache();
  return *cache;
}

/**
 * Resolve a dial target to an IPv4 address. Literals are parsed directly;
 * names are looked up once and cached (a failed lookup is not cached).
 * @param host IPv4 literal or host name.
 * @param out Receives the address (network byte order).
 * @return 0 on success, else a getaddrinfo EAI_* code.
 */
static int ResolveHostV4(const std::string& host, struct in_addr* out) {
  if (::inet_pton(AF_INET, host.c_str(), out) > 0) return 0;
  HostCache& cache = GetHostCache();
  {
    std::lock_guard<std::mutex> lock(cache.mtx_);
    auto it = cache.map_.find(host);
    if (it != cache.map_.end()) {
      *out = it->second;
      return 0;
    }
  }
  struct addrinfo hints;
  std::memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  struct addrinfo* res = nullptr;
  cache.lookups_.fetch_add(1, std::memory_order_relaxed);
  int rc = ::getaddrinfo(host.c_str(), nullptr, &hints, &res);
  if (rc != 0 || res == nullptr) {
    if (res != nullptr) ::freeaddrinfo(res);
    return rc != 0 ? rc : EAI_NONAME;
  }
  *out = reinterpret_cast<struct sockaddr_in*>(res->ai_addr)->sin_addr;
  ::freeaddrinfo(res);
  std::lock_guard<std::mutex> lock(cache.mtx_);
  cache.map_[host] = *out;
  return 0;
}

uint64_t GetHostLookupCount() {
  return GetHostCache().lookups_.load(std::memory_order_relaxed);
}

void ForgetHost(const std::string& host) {
  HostCache& cache = GetHostCache();
  std::lock_guard<std::mutex> lock(cache.mtx_);
  cache.map_.erase(host);
}

bool IsServerAlive(const std::string& addr, int port,
                   const std::string& protocol) {
  if (protocol == "ipc") {
    // sun_path holds a short fixed-size path; a longer one used to be
    // silently truncated, binding or dialing the wrong name (#1356).
    if (addr.size() >= sizeof(sockaddr_un::sun_path)) {
      errno = ENAMETOOLONG;
      return false;
    }
    socket_t fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd == kInvalidSocket) return false;
    struct sockaddr_un sun;
    std::memset(&sun, 0, sizeof(sun));
    sun.sun_family = AF_UNIX;
    std::strncpy(sun.sun_path, addr.c_str(), sizeof(sun.sun_path) - 1);
    int rc = ::connect(fd, reinterpret_cast<struct sockaddr*>(&sun),
                       sizeof(sun));
    Close(fd);
    return rc == 0;
  }
  socket_t fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd == kInvalidSocket) return false;
  struct timeval tv;
  tv.tv_sec = 0;
  tv.tv_usec = 500000;
  setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO,
             reinterpret_cast<const char*>(&tv), sizeof(tv));
  struct sockaddr_in sa;
  std::memset(&sa, 0, sizeof(sa));
  sa.sin_family = AF_INET;
  sa.sin_port = htons(static_cast<uint16_t>(port));
  ::inet_pton(AF_INET, addr.c_str(), &sa.sin_addr);
  int rc = ::connect(fd, reinterpret_cast<struct sockaddr*>(&sa), sizeof(sa));
  Close(fd);
  return rc == 0;
}

socket_t Connect(const std::string& addr, int port,
                 const std::string& protocol) {
  if (protocol == "ipc") {
    // sun_path holds a short fixed-size path; a longer one used to be
    // silently truncated, binding or dialing the wrong name (#1356).
    if (addr.size() >= sizeof(sockaddr_un::sun_path)) {
      errno = ENAMETOOLONG;
      return kInvalidSocket;
    }
    socket_t fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd == kInvalidSocket) return kInvalidSocket;
    struct sockaddr_un sun;
    std::memset(&sun, 0, sizeof(sun));
    sun.sun_family = AF_UNIX;
    std::strncpy(sun.sun_path, addr.c_str(), sizeof(sun.sun_path) - 1);
    if (::connect(fd, reinterpret_cast<struct sockaddr*>(&sun),
                  sizeof(sun)) < 0) {
      Close(fd);
      return kInvalidSocket;
    }
    SetNoSigPipe(fd);
    return fd;
  }
  socket_t fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd == kInvalidSocket) return kInvalidSocket;
  SetTcpNoDelay(fd);
  SetSendBuf(fd, 4 * 1024 * 1024);
  SetNoSigPipe(fd);
  struct sockaddr_in sin;
  std::memset(&sin, 0, sizeof(sin));
  sin.sin_family = AF_INET;
  sin.sin_port = htons(static_cast<uint16_t>(port));
  // Not a dotted IPv4 literal: resolve it (cached). Hostfiles name hosts
  // (e.g. ares-comp-31-40g), and ZeroMQ resolves names, so a peer path
  // switched to this transport could otherwise never connect.
  if (ResolveHostV4(addr, &sin.sin_addr) != 0) {
    Close(fd);
    return kInvalidSocket;
  }
  if (::connect(fd, reinterpret_cast<struct sockaddr*>(&sin),
                sizeof(sin)) < 0) {
    int saved = errno;
    ForgetHost(addr);
    Close(fd);
    errno = saved;
    return kInvalidSocket;
  }
  return fd;
}

std::string ConnectError::ToString() const {
  switch (stage_) {
    case kNone:
      return "ok";
    case kResolve:
      return std::string("host lookup: ") + ::gai_strerror(code_);
    case kSocket:
      return std::string("socket: ") + std::strerror(code_);
    case kConnect:
      return std::string("connect: ") + std::strerror(code_);
    case kTimeout:
      return "connect: timed out";
  }
  return "unknown";
}

/**
 * Wait for a non-blocking connect() to finish.
 * @param fd Socket with a connect in progress.
 * @param timeout_ms Longest wait; <= 0 waits indefinitely.
 * @param err Receives the failure reason.
 * @return true if the connection is established.
 */
static bool FinishConnect(socket_t fd, int timeout_ms, ConnectError* err) {
  struct pollfd pfd;
  pfd.fd = fd;
  pfd.events = POLLOUT;
  pfd.revents = 0;
  int pr;
  do {
    pr = ::poll(&pfd, 1, timeout_ms > 0 ? timeout_ms : -1);
  } while (pr < 0 && errno == EINTR);
  if (pr == 0) {
    err->stage_ = ConnectError::kTimeout;
    err->code_ = ETIMEDOUT;
    return false;
  }
  int so_err = 0;
  socklen_t len = sizeof(so_err);
  if (pr < 0) {
    so_err = errno;
  } else if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_err, &len) < 0) {
    so_err = errno;
  }
  if (so_err != 0) {
    err->stage_ = ConnectError::kConnect;
    err->code_ = so_err;
    return false;
  }
  return true;
}

socket_t ConnectTimed(const std::string& addr, int port, int timeout_ms,
                      ConnectError* err) {
  ConnectError local;
  if (err == nullptr) err = &local;
  *err = ConnectError();
  struct sockaddr_in sin;
  std::memset(&sin, 0, sizeof(sin));
  sin.sin_family = AF_INET;
  sin.sin_port = htons(static_cast<uint16_t>(port));
  int gai = ResolveHostV4(addr, &sin.sin_addr);
  if (gai != 0) {
    err->stage_ = ConnectError::kResolve;
    err->code_ = gai;
    return kInvalidSocket;
  }
  socket_t fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd == kInvalidSocket) {
    err->stage_ = ConnectError::kSocket;
    err->code_ = errno;
    return kInvalidSocket;
  }
  SetTcpNoDelay(fd);
  SetNoSigPipe(fd);
  SetNonBlocking(fd, true);
  bool ok = true;
  if (::connect(fd, reinterpret_cast<struct sockaddr*>(&sin), sizeof(sin)) <
      0) {
    if (errno == EINPROGRESS || errno == EINTR) {
      ok = FinishConnect(fd, timeout_ms, err);
    } else {
      err->stage_ = ConnectError::kConnect;
      err->code_ = errno;
      ok = false;
    }
  }
  if (!ok) {
    ForgetHost(addr);
    Close(fd);
    return kInvalidSocket;
  }
  SetNonBlocking(fd, false);
  return fd;
}

/**
 * Resolve the IPv4 address a TCP listener binds.
 *
 * An empty address, "*" or "0.0.0.0" means every interface. Anything else is
 * a dotted IPv4 literal or a host name, so a caller that asks for loopback
 * gets loopback rather than a wildcard listener reachable from the network.
 *
 * @param addr address requested by the caller
 * @param out receives the resolved address in network byte order
 * @return true on success, false when `addr` cannot be resolved
 */
static bool ResolveListenAddr(const std::string& addr, struct in_addr* out) {
  if (addr.empty() || addr == "*" || addr == "0.0.0.0") {
    out->s_addr = htonl(INADDR_ANY);
    return true;
  }
  if (::inet_pton(AF_INET, addr.c_str(), out) > 0) return true;
  struct addrinfo hints;
  std::memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  struct addrinfo* res = nullptr;
  if (::getaddrinfo(addr.c_str(), nullptr, &hints, &res) != 0 ||
      res == nullptr) {
    return false;
  }
  *out = reinterpret_cast<struct sockaddr_in*>(res->ai_addr)->sin_addr;
  ::freeaddrinfo(res);
  return true;
}

socket_t Listen(const std::string& addr, int port,
                const std::string& protocol) {
  socket_t fd;
  if (protocol == "ipc") {
    UnlinkPath(addr.c_str());
    // sun_path holds a short fixed-size path; a longer one used to be
    // silently truncated, binding or dialing the wrong name (#1356).
    if (addr.size() >= sizeof(sockaddr_un::sun_path)) {
      errno = ENAMETOOLONG;
      return kInvalidSocket;
    }
    fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd == kInvalidSocket) return kInvalidSocket;
    struct sockaddr_un sun;
    std::memset(&sun, 0, sizeof(sun));
    sun.sun_family = AF_UNIX;
    std::strncpy(sun.sun_path, addr.c_str(), sizeof(sun.sun_path) - 1);
    if (::bind(fd, reinterpret_cast<struct sockaddr*>(&sun), sizeof(sun)) < 0) {
      Close(fd);
      return kInvalidSocket;
    }
  } else {
    struct sockaddr_in sin;
    std::memset(&sin, 0, sizeof(sin));
    sin.sin_family = AF_INET;
    sin.sin_port = htons(static_cast<uint16_t>(port));
    if (!ResolveListenAddr(addr, &sin.sin_addr)) return kInvalidSocket;
    fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd == kInvalidSocket) return kInvalidSocket;
    SetReuseAddr(fd);
    SetRecvBuf(fd, 4 * 1024 * 1024);
    if (::bind(fd, reinterpret_cast<struct sockaddr*>(&sin), sizeof(sin)) < 0) {
      Close(fd);
      return kInvalidSocket;
    }
  }
  if (::listen(fd, kListenBacklog) < 0) {
    Close(fd);
    return kInvalidSocket;
  }
  return fd;
}

socket_t Accept(socket_t listen_fd) {
  socket_t fd = ::accept(listen_fd, nullptr, nullptr);
  if (fd != kInvalidSocket) {
    SetNoSigPipe(fd);
  }
  return fd;
}

uint32_t HostToNet32(uint32_t host) { return htonl(host); }
uint32_t NetToHost32(uint32_t net) { return ntohl(net); }

}  // namespace ctp::lbm::sock

#endif  // !_WIN32
