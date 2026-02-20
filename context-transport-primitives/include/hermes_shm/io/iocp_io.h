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
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <memory>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace hshm {

struct IocpOperation {
  OVERLAPPED overlapped;
  IoToken token;
  bool completed;
  ::DWORD bytes_transferred;
  ::DWORD error_code;
};

class IocpAsyncIO : public AsyncIO {
 public:
  IocpAsyncIO(uint32_t io_depth)
      : file_handle_(INVALID_HANDLE_VALUE),
        iocp_handle_(nullptr),
        next_token_(1) {
    (void)io_depth;
  }

  ~IocpAsyncIO() override {
    Close();
  }

  bool Open(const std::string &path, int flags, mode_t mode) override {
    (void)mode;
    std::lock_guard<std::mutex> lock(mutex_);

    ::DWORD access = 0;
    if (flags & O_RDWR) {
      access = GENERIC_READ | GENERIC_WRITE;
    } else if (flags & O_WRONLY) {
      access = GENERIC_WRITE;
    } else {
      access = GENERIC_READ;
    }

    ::DWORD creation = OPEN_EXISTING;
    if (flags & O_CREAT) {
      creation = OPEN_ALWAYS;
    }

    file_handle_ = CreateFileA(
        path.c_str(),
        access,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        nullptr,
        creation,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
        nullptr);

    if (file_handle_ == INVALID_HANDLE_VALUE) {
      return false;
    }

    iocp_handle_ = CreateIoCompletionPort(
        file_handle_, nullptr, 0, 0);
    if (iocp_handle_ == nullptr) {
      CloseHandle(file_handle_);
      file_handle_ = INVALID_HANDLE_VALUE;
      return false;
    }

    return true;
  }

  ssize_t GetAsyncFileSize() const override {
    if (file_handle_ == INVALID_HANDLE_VALUE) return -1;
    LARGE_INTEGER size;
    if (!GetFileSizeEx(file_handle_, &size)) return -1;
    return static_cast<ssize_t>(size.QuadPart);
  }

  bool Truncate(size_t size) override {
    if (file_handle_ == INVALID_HANDLE_VALUE) return false;
    FILE_END_OF_FILE_INFO eofi;
    eofi.EndOfFile.QuadPart = static_cast<LONGLONG>(size);
    return ::SetFileInformationByHandle(
        file_handle_, FileEndOfFileInfo, &eofi, sizeof(eofi)) != 0;
  }

  IoToken Write(void *buffer, size_t size, off_t offset) override {
    return SubmitIO(buffer, size, offset, true);
  }

  IoToken Read(void *buffer, size_t size, off_t offset) override {
    return SubmitIO(buffer, size, offset, false);
  }

  bool IsComplete(IoToken token, IoResult &result) override {
    // First drain any completions from the IOCP queue
    DrainCompletions();

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = pending_.find(token);
    if (it == pending_.end()) return false;

    if (!it->second->completed) return false;

    if (it->second->error_code == 0) {
      result.bytes_transferred =
          static_cast<ssize_t>(it->second->bytes_transferred);
      result.error_code = 0;
    } else {
      result.bytes_transferred = -1;
      result.error_code = static_cast<int>(it->second->error_code);
    }

    pending_.erase(it);
    return true;
  }

  void Close() override {
    std::lock_guard<std::mutex> lock(mutex_);

    // Cancel pending operations
    if (file_handle_ != INVALID_HANDLE_VALUE) {
      CancelIo(file_handle_);
    }
    pending_.clear();

    if (iocp_handle_ != nullptr) {
      CloseHandle(iocp_handle_);
      iocp_handle_ = nullptr;
    }
    if (file_handle_ != INVALID_HANDLE_VALUE) {
      CloseHandle(file_handle_);
      file_handle_ = INVALID_HANDLE_VALUE;
    }
  }

  int GetEventFd() const override {
    return -1;  // Not applicable on Windows
  }

 private:
  IoToken SubmitIO(void *buffer, size_t size, off_t offset, bool is_write) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_handle_ == INVALID_HANDLE_VALUE) return kInvalidIoToken;

    IoToken token = next_token_.fetch_add(1);
    auto op = std::make_unique<IocpOperation>();
    memset(&op->overlapped, 0, sizeof(OVERLAPPED));
    op->overlapped.Offset = static_cast<DWORD>(offset & 0xFFFFFFFF);
    op->overlapped.OffsetHigh = static_cast<DWORD>(
        (static_cast<uint64_t>(offset) >> 32) & 0xFFFFFFFF);
    op->token = token;
    op->completed = false;
    op->bytes_transferred = 0;
    op->error_code = 0;

    BOOL ok;
    if (is_write) {
      ok = WriteFile(file_handle_, buffer, static_cast<::DWORD>(size),
                     nullptr, &op->overlapped);
    } else {
      ok = ReadFile(file_handle_, buffer, static_cast<::DWORD>(size),
                    nullptr, &op->overlapped);
    }

    if (!ok) {
      ::DWORD err = ::GetLastError();
      if (err != ERROR_IO_PENDING) {
        return kInvalidIoToken;
      }
    } else {
      // Completed synchronously - mark it
      op->completed = true;
      ::DWORD transferred = 0;
      ::GetOverlappedResult(file_handle_, &op->overlapped, &transferred, FALSE);
      op->bytes_transferred = transferred;
      op->error_code = 0;
    }

    pending_[token] = std::move(op);
    return token;
  }

  void DrainCompletions() {
    if (iocp_handle_ == nullptr) return;

    ::DWORD bytes = 0;
    ULONG_PTR key = 0;
    LPOVERLAPPED ov = nullptr;

    // Non-blocking: drain all ready completions
    while (::GetQueuedCompletionStatus(iocp_handle_, &bytes, &key, &ov, 0)) {
      if (ov == nullptr) break;
      std::lock_guard<std::mutex> lock(mutex_);
      // Find the operation containing this OVERLAPPED
      for (auto &kv : pending_) {
        if (&kv.second->overlapped == ov) {
          kv.second->completed = true;
          kv.second->bytes_transferred = bytes;
          kv.second->error_code = 0;
          break;
        }
      }
    }

    // Check if there was an error completion
    if (ov != nullptr) {
      ::DWORD err = ::GetLastError();
      std::lock_guard<std::mutex> lock(mutex_);
      for (auto &kv : pending_) {
        if (&kv.second->overlapped == ov) {
          kv.second->completed = true;
          kv.second->bytes_transferred = bytes;
          kv.second->error_code = err;
          break;
        }
      }
    }
  }

  HANDLE file_handle_;
  HANDLE iocp_handle_;
  std::atomic<IoToken> next_token_;
  std::mutex mutex_;
  std::unordered_map<IoToken, std::unique_ptr<IocpOperation>> pending_;
};

}  // namespace hshm

#endif  // _WIN32

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_IO_IOCP_IO_H_
