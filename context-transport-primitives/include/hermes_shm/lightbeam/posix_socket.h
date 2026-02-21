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

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <string>

#include "hermes_shm/constants/macros.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <afunix.h>
#include <BaseTsd.h>
#ifdef Yield
#undef Yield
#endif
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
using ssize_t = SSIZE_T;
#else
#include <sys/socket.h>
#include <sys/uio.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <poll.h>
#include <fcntl.h>
#include <sys/epoll.h>
#endif

namespace hshm::lbm::sock {

#ifdef _WIN32
using socket_t = SOCKET;
constexpr socket_t kInvalidSocket = INVALID_SOCKET;
#else
using socket_t = int;
constexpr socket_t kInvalidSocket = -1;
#endif

/** Platform-neutral scatter-gather buffer (replaces struct iovec) */
struct IoBuffer {
  void* base;
  size_t len;
};

HSHM_DLL void Close(socket_t fd);
HSHM_DLL int GetError();
HSHM_DLL std::string GetErrorString();
HSHM_DLL void SetNonBlocking(socket_t fd, bool enable);
HSHM_DLL void SetTcpNoDelay(socket_t fd);
HSHM_DLL void SetReuseAddr(socket_t fd);
HSHM_DLL void SetSendBuf(socket_t fd, int size);
HSHM_DLL void SetRecvBuf(socket_t fd, int size);

/** Initialize platform socket library (WSAStartup on Windows, no-op on POSIX) */
HSHM_DLL void InitSocketLib();

/** Cleanup platform socket library (WSACleanup on Windows, no-op on POSIX) */
HSHM_DLL void CleanupSocketLib();

/** Scatter-gather send. Returns total bytes sent or -1 on error. */
HSHM_DLL ssize_t SendV(socket_t fd, const IoBuffer* iov, int count);

/** Receive exactly len bytes. Returns 0 on success, -1 on error/short read. */
HSHM_DLL int RecvExact(socket_t fd, char* buf, size_t len);

/** Poll a single fd for readability. Returns >0 if ready, 0 on timeout, -1 on error. */
HSHM_DLL int PollRead(socket_t fd, int timeout_ms);

/** Poll multiple fds for readability. Returns index of first ready fd, -1 if none/error. */
HSHM_DLL int PollReadMulti(const socket_t* fds, int count, int timeout_ms);

/** Remove a file path (unlink on POSIX, DeleteFileA on Windows) */
HSHM_DLL void UnlinkPath(const char* path);

#ifndef _WIN32
/** Create an epoll file descriptor. Returns epoll fd or -1 on error. */
int EpollCreate();

/** Add a socket fd to an epoll instance for EPOLLIN events. Returns 0 on success. */
int EpollAdd(int epoll_fd, socket_t fd);

/** Wait on an epoll instance. Returns number of ready events. */
int EpollWait(int epoll_fd, struct epoll_event* events, int max_events,
              int timeout_ms);

/** Close an epoll file descriptor. */
void EpollClose(int epoll_fd);
#endif

}  // namespace hshm::lbm::sock
