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

#include "hermes_shm/lightbeam/posix_socket.h"

#include <cerrno>
#include <cstring>

namespace hshm::lbm::sock {

void InitSocketLib() {
  // No-op on POSIX
}

void CleanupSocketLib() {
  // No-op on POSIX
}

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
    ssize_t n = ::writev(fd, local_iov + iov_idx, local_count - iov_idx);
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

}  // namespace hshm::lbm::sock

#endif  // !_WIN32
