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

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_IO_ASYNC_IO_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_IO_ASYNC_IO_H_

#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <string>

#include "hermes_shm/introspect/system_info.h"

namespace hshm {

/** Opaque token returned by Submit, used to check completion */
using IoToken = uint64_t;
static constexpr IoToken kInvalidIoToken = 0;

struct IoResult {
  ssize_t bytes_transferred;  /**< Bytes actually transferred (-1 on error) */
  int error_code;             /**< 0 on success, errno on failure */
};

class AsyncIO {
 public:
  AsyncIO() = default;
  virtual ~AsyncIO() = default;

  /** Open a file. Returns true on success. AsyncIO owns the fds internally.
   *  Opens two fds: one with O_DIRECT, one without.
   *  @param path File path
   *  @param flags O_RDWR, O_CREAT, etc. (O_DIRECT is managed internally)
   *  @param mode File creation mode (e.g., 0644) */
  virtual bool Open(const std::string &path, int flags, mode_t mode) = 0;

  /** Get file size. Returns -1 on error. */
  virtual ssize_t GetAsyncFileSize() const = 0;

  /** Truncate/extend file. Returns true on success. */
  virtual bool Truncate(size_t size) = 0;

  /** Submit async write. Automatically selects O_DIRECT fd if buffer and
   *  size are aligned, otherwise uses regular fd.
   *  @return IoToken for tracking, or kInvalidIoToken on failure */
  virtual IoToken Write(void *buffer, size_t size, off_t offset) = 0;

  /** Submit async read. Same alignment-adaptive logic as Write. */
  virtual IoToken Read(void *buffer, size_t size, off_t offset) = 0;

  /** Non-blocking check: is this I/O operation complete?
   *  @param token Token from Write/Read
   *  @param result Filled on completion
   *  @return true if complete, false if still in progress */
  virtual bool IsComplete(IoToken token, IoResult &result) = 0;

  /** Close all file descriptors and clean up resources */
  virtual void Close() = 0;

  /** Get the eventfd for epoll integration (returns -1 if not supported) */
  virtual int GetEventFd() const = 0;
};

}  // namespace hshm

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_IO_ASYNC_IO_H_
