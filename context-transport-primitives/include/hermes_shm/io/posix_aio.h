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

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_IO_POSIX_AIO_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_IO_POSIX_AIO_H_

#if !defined(_WIN32)

#include "async_io.h"
#include <aio.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <memory>

namespace hshm {

class PosixAsyncIO : public AsyncIO {
 public:
  PosixAsyncIO(uint32_t io_depth)
      : direct_fd_(-1), regular_fd_(-1), next_token_(1) {
    (void)io_depth;
  }

  ~PosixAsyncIO() override {
    Close();
  }

  bool Open(const std::string &path, int flags, mode_t mode) override {
    std::lock_guard<std::mutex> lock(mutex_);

    int base_flags = flags & ~O_DIRECT;

    regular_fd_ = open(path.c_str(), base_flags, mode);
    if (regular_fd_ < 0) return false;

    direct_fd_ = open(path.c_str(), base_flags | O_DIRECT, mode);

    return true;
  }

  ssize_t GetAsyncFileSize() const override {
    int fd = regular_fd_ >= 0 ? regular_fd_ : direct_fd_;
    if (fd < 0) return -1;
    struct stat st;
    if (fstat(fd, &st) != 0) return -1;
    return st.st_size;
  }

  bool Truncate(size_t size) override {
    int fd = regular_fd_ >= 0 ? regular_fd_ : direct_fd_;
    if (fd < 0) return false;
    return ftruncate(fd, static_cast<off_t>(size)) == 0;
  }

  IoToken Write(void *buffer, size_t size, off_t offset) override {
    return SubmitIO(buffer, size, offset, true);
  }

  IoToken Read(void *buffer, size_t size, off_t offset) override {
    return SubmitIO(buffer, size, offset, false);
  }

  bool IsComplete(IoToken token, IoResult &result) override {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = pending_.find(token);
    if (it == pending_.end()) return false;

    int err = aio_error(it->second.get());
    if (err == EINPROGRESS) return false;

    if (err == 0) {
      ssize_t bytes = aio_return(it->second.get());
      result.bytes_transferred = bytes;
      result.error_code = 0;
    } else {
      aio_return(it->second.get());
      result.bytes_transferred = -1;
      result.error_code = err;
    }

    pending_.erase(it);
    return true;
  }

  void Close() override {
    std::lock_guard<std::mutex> lock(mutex_);

    // Cancel pending operations
    for (auto &kv : pending_) {
      aio_cancel(regular_fd_, kv.second.get());
    }
    pending_.clear();

    if (direct_fd_ >= 0) {
      close(direct_fd_);
      direct_fd_ = -1;
    }
    if (regular_fd_ >= 0) {
      close(regular_fd_);
      regular_fd_ = -1;
    }
  }

  int GetEventFd() const override {
    return -1;  // POSIX AIO does not support eventfd
  }

 private:
  IoToken SubmitIO(void *buffer, size_t size, off_t offset, bool is_write) {
    std::lock_guard<std::mutex> lock(mutex_);

    int fd = SelectFd(buffer, size);
    IoToken token = next_token_.fetch_add(1);

    auto cb = std::make_unique<struct aiocb>();
    memset(cb.get(), 0, sizeof(struct aiocb));

    cb->aio_fildes = fd;
    cb->aio_buf = buffer;
    cb->aio_nbytes = size;
    cb->aio_offset = offset;

    int ret;
    if (is_write) {
      ret = aio_write(cb.get());
    } else {
      ret = aio_read(cb.get());
    }

    if (ret != 0) {
      return kInvalidIoToken;
    }

    pending_[token] = std::move(cb);
    return token;
  }

  int SelectFd(void *buffer, size_t size) const {
    if (direct_fd_ >= 0 &&
        (reinterpret_cast<uintptr_t>(buffer) % 4096 == 0) &&
        (size % 4096 == 0)) {
      return direct_fd_;
    }
    return regular_fd_;
  }

  int direct_fd_;
  int regular_fd_;
  std::atomic<IoToken> next_token_;
  std::mutex mutex_;
  std::unordered_map<IoToken, std::unique_ptr<struct aiocb>> pending_;
};

}  // namespace hshm

#endif  // !defined(_WIN32)

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_IO_POSIX_AIO_H_
