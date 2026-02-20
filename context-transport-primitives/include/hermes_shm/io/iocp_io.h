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

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_IO_IOCP_IO_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_IO_IOCP_IO_H_

#ifdef _WIN32

#include "async_io.h"

// TODO: #include <windows.h>
// TODO: Windows IOCP implementation

namespace hshm {

class IocpAsyncIO : public AsyncIO {
 public:
  IocpAsyncIO(uint32_t io_depth) {
    (void)io_depth;
  }

  ~IocpAsyncIO() override {
    Close();
  }

  bool Open(const std::string &path, int flags, mode_t mode) override {
    // TODO: CreateFileA with FILE_FLAG_OVERLAPPED
    (void)path; (void)flags; (void)mode;
    return false;
  }

  ssize_t GetAsyncFileSize() const override {
    // TODO: GetFileSizeEx
    return -1;
  }

  bool Truncate(size_t size) override {
    // TODO: SetFilePointerEx + SetEndOfFile
    (void)size;
    return false;
  }

  IoToken Write(void *buffer, size_t size, off_t offset) override {
    // TODO: WriteFile with OVERLAPPED + IOCP
    (void)buffer; (void)size; (void)offset;
    return kInvalidIoToken;
  }

  IoToken Read(void *buffer, size_t size, off_t offset) override {
    // TODO: ReadFile with OVERLAPPED + IOCP
    (void)buffer; (void)size; (void)offset;
    return kInvalidIoToken;
  }

  bool IsComplete(IoToken token, IoResult &result) override {
    // TODO: GetQueuedCompletionStatus (non-blocking)
    (void)token; (void)result;
    return false;
  }

  void Close() override {
    // TODO: CloseHandle for file handles and IOCP
  }

  int GetEventFd() const override {
    return -1;  // Not applicable on Windows
  }
};

}  // namespace hshm

#endif  // _WIN32

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_IO_IOCP_IO_H_
