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

#ifdef _WIN32

#include "hermes_shm/lightbeam/posix_socket.h"

#include <cstring>

namespace hshm::lbm::sock {

void InitSocketLib() {
  static bool initialized = false;
  if (initialized) return;
  initialized = true;
  WSADATA wsa_data;
  WSAStartup(MAKEWORD(2, 2), &wsa_data);
}

void CleanupSocketLib() {
  WSACleanup();
}

void Close(socket_t fd) {
  if (fd != kInvalidSocket) {
    ::closesocket(fd);
  }
}

int GetError() {
  return WSAGetLastError();
}

std::string GetErrorString() {
  int err = WSAGetLastError();
  char buf[256];
  FormatMessageA(
      FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
      nullptr, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      buf, sizeof(buf), nullptr);
  return std::string(buf);
}

void SetNonBlocking(socket_t fd, bool enable) {
  u_long mode = enable ? 1 : 0;
  ioctlsocket(fd, FIONBIO, &mode);
}

void SetTcpNoDelay(socket_t fd) {
  int flag = 1;
  ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY,
               reinterpret_cast<const char*>(&flag), sizeof(flag));
}

void SetReuseAddr(socket_t fd) {
  int flag = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR,
               reinterpret_cast<const char*>(&flag), sizeof(flag));
}

void SetSendBuf(socket_t fd, int size) {
  ::setsockopt(fd, SOL_SOCKET, SO_SNDBUF,
               reinterpret_cast<const char*>(&size), sizeof(size));
}

void SetRecvBuf(socket_t fd, int size) {
  ::setsockopt(fd, SOL_SOCKET, SO_RCVBUF,
               reinterpret_cast<const char*>(&size), sizeof(size));
}

void UnlinkPath(const char* path) {
  DeleteFileA(path);
}

ssize_t SendV(socket_t fd, const IoBuffer* iov, int count) {
  // Convert IoBuffer to WSABUF
  WSABUF wsa_bufs[64];
  int local_count = count < 64 ? count : 64;
  DWORD total_expected = 0;
  for (int i = 0; i < local_count; ++i) {
    wsa_bufs[i].buf = static_cast<char*>(iov[i].base);
    wsa_bufs[i].len = static_cast<ULONG>(iov[i].len);
    total_expected += wsa_bufs[i].len;
  }

  DWORD bytes_sent = 0;
  DWORD total_sent = 0;
  int buf_idx = 0;

  while (total_sent < total_expected) {
    int rc = WSASend(fd, wsa_bufs + buf_idx, local_count - buf_idx,
                     &bytes_sent, 0, nullptr, nullptr);
    if (rc == SOCKET_ERROR) {
      int err = WSAGetLastError();
      if (err == WSAEINTR) continue;
      if (err == WSAEWOULDBLOCK) {
        WSAPOLLFD pfd;
        pfd.fd = fd;
        pfd.events = POLLWRNORM;
        pfd.revents = 0;
        int pr = WSAPoll(&pfd, 1, 1000);
        if (pr <= 0) return -1;
        continue;
      }
      return -1;
    }
    total_sent += bytes_sent;
    DWORD remaining = bytes_sent;
    while (buf_idx < local_count &&
           remaining >= static_cast<DWORD>(wsa_bufs[buf_idx].len)) {
      remaining -= wsa_bufs[buf_idx].len;
      buf_idx++;
    }
    if (buf_idx < local_count && remaining > 0) {
      wsa_bufs[buf_idx].buf += remaining;
      wsa_bufs[buf_idx].len -= remaining;
    }
  }
  return static_cast<ssize_t>(total_sent);
}

int RecvExact(socket_t fd, char* buf, size_t len) {
  size_t received = 0;
  while (received < len) {
    int n = ::recv(fd, buf + received, static_cast<int>(len - received), 0);
    if (n == SOCKET_ERROR) {
      int err = WSAGetLastError();
      if (err == WSAEINTR) continue;
      if (err == WSAEWOULDBLOCK) {
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
  WSAPOLLFD pfd;
  pfd.fd = fd;
  pfd.events = POLLRDNORM;
  pfd.revents = 0;
  return WSAPoll(&pfd, 1, timeout_ms);
}

int PollReadMulti(const socket_t* fds, int count, int timeout_ms) {
  WSAPOLLFD pfds[128];
  int n = count < 128 ? count : 128;
  for (int i = 0; i < n; ++i) {
    pfds[i].fd = fds[i];
    pfds[i].events = POLLRDNORM;
    pfds[i].revents = 0;
  }
  int rc = WSAPoll(pfds, n, timeout_ms);
  if (rc <= 0) return -1;
  for (int i = 0; i < n; ++i) {
    if (pfds[i].revents & POLLRDNORM) {
      return i;
    }
  }
  return -1;
}

}  // namespace hshm::lbm::sock

#endif  // _WIN32
