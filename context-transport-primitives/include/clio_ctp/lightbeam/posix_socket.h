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

#include "clio_ctp/constants/macros.h"
#include "clio_ctp/types/numbers.h"

#ifdef _WIN32
// IMPORTANT: never include Windows API headers (<winsock2.h>, <windows.h>, ...)
// from a C++ header. They leak function-like macros (Yield, min, max, ...) into
// every including translation unit, which breaks e.g. StdThread::Yield() /
// Cuda::Yield() — and especially the CUDA build, where nvcc force-includes
// cuda_runtime.h -> windows.h before our headers. The Windows socket API lives
// entirely in socket_win.cc; here we only need portable mirrors of SOCKET /
// INVALID_SOCKET / SSIZE_T (a pointer-sized handle, ~0, and a signed size).
using ssize_t = intptr_t;
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
#ifdef __linux__
#include <sys/epoll.h>
#endif
#endif

namespace ctp::lbm::sock {

#ifdef _WIN32
using socket_t = uintptr_t;  // Windows SOCKET is UINT_PTR
constexpr socket_t kInvalidSocket =
    static_cast<socket_t>(~static_cast<uintptr_t>(0));  // INVALID_SOCKET
#else
using socket_t = int;
constexpr socket_t kInvalidSocket = -1;
#endif

/** Platform-neutral scatter-gather buffer (replaces struct iovec) */
struct IoBuffer {
  void* base;
  size_t len;
};

CTP_DLL void Close(socket_t fd);
CTP_DLL int GetError();
CTP_DLL std::string GetErrorString();
CTP_DLL void SetNonBlocking(socket_t fd, bool enable);
CTP_DLL void SetTcpNoDelay(socket_t fd);
CTP_DLL void SetReuseAddr(socket_t fd);
CTP_DLL void SetSendBuf(socket_t fd, int size);
CTP_DLL void SetRecvBuf(socket_t fd, int size);

/** Prevent writes to this socket from raising SIGPIPE when the peer has
 *  closed the connection. On macOS/BSD this sets the SO_NOSIGPIPE socket
 *  option; on Linux SendV() passes MSG_NOSIGNAL per write instead, so this is
 *  a no-op there (and on Windows, which has no SIGPIPE). Without it, writing
 *  to a dead peer terminates the process via SIGPIPE's default disposition on
 *  macOS. */
CTP_DLL void SetNoSigPipe(socket_t fd);

/** Initialize platform socket library (WSAStartup on Windows, no-op on POSIX) */
CTP_DLL void InitSocketLib();

/** Cleanup platform socket library (WSACleanup on Windows, no-op on POSIX) */
CTP_DLL void CleanupSocketLib();

/** Scatter-gather send. Returns total bytes sent or -1 on error. */
CTP_DLL ssize_t SendV(socket_t fd, const IoBuffer* iov, int count);

/** Receive exactly len bytes. Returns 0 on success, -1 on error/short read. */
CTP_DLL int RecvExact(socket_t fd, char* buf, size_t len);

/** Poll a single fd for readability. Returns >0 if ready, 0 on timeout, -1 on error. */
CTP_DLL int PollRead(socket_t fd, int timeout_ms);

/** Poll multiple fds for readability. Returns index of first ready fd, -1 if none/error. */
CTP_DLL int PollReadMulti(const socket_t* fds, int count, int timeout_ms);

/** Remove a file path (unlink on POSIX, DeleteFileA on Windows) */
CTP_DLL void UnlinkPath(const char* path);

/** Probe whether a server is accepting connections. protocol == "ipc" uses a
 *  Unix-domain socket at `addr`; otherwise a TCP connect to addr:port. Returns
 *  true if connect() succeeds. Keeps all socket-API use out of headers. */
CTP_DLL bool IsServerAlive(const std::string& addr, int port,
                           const std::string& protocol);

/** Create a socket and connect. protocol == "ipc" uses a Unix-domain socket at
 *  `addr`; otherwise TCP to addr:port (with TCP_NODELAY + a large send buffer).
 *  Returns a connected fd, or kInvalidSocket on failure. */
CTP_DLL socket_t Connect(const std::string& addr, int port,
                         const std::string& protocol);

/** Create a socket, bind, and listen. protocol == "ipc" uses a Unix-domain
 *  socket at `addr` (unlinking any stale path first); otherwise TCP on `port`
 *  (with SO_REUSEADDR + a large recv buffer). Returns a listening fd, or
 *  kInvalidSocket on failure. */
CTP_DLL socket_t Listen(const std::string& addr, int port,
                        const std::string& protocol);

/** Accept one pending connection on a listening fd (non-blocking). Returns the
 *  accepted fd, or kInvalidSocket if none pending / on error. */
CTP_DLL socket_t Accept(socket_t listen_fd);

/** Host<->network byte order for the 4-byte framing length prefix. */
CTP_DLL uint32_t HostToNet32(uint32_t host);
CTP_DLL uint32_t NetToHost32(uint32_t net);

#ifdef __linux__
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

}  // namespace ctp::lbm::sock
