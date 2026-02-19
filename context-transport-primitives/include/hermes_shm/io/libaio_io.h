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

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_IO_LIBAIO_IO_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_IO_LIBAIO_IO_H_

#if HSHM_ENABLE_LIBAIO

#include "async_io.h"
#include <libaio.h>
#include <sys/eventfd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cerrno>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace hshm {

class LinuxAioAsyncIO : public AsyncIO {
 public:
  LinuxAioAsyncIO(uint32_t io_depth)
      : aio_ctx_(0), event_fd_(-1), direct_fd_(-1), regular_fd_(-1),
        io_depth_(io_depth), next_token_(1) {}

  ~LinuxAioAsyncIO() override {
    Close();
  }

  bool Open(const std::string &path, int flags, mode_t mode) override {
    std::lock_guard<std::mutex> lock(mutex_);

    // Strip O_DIRECT from caller flags — we manage it internally
    int base_flags = flags & ~O_DIRECT;

    // Open regular fd (without O_DIRECT)
    regular_fd_ = open(path.c_str(), base_flags, mode);
    if (regular_fd_ < 0) {
      return false;
    }

    // Open direct fd (with O_DIRECT) — may fail on some filesystems
    direct_fd_ = open(path.c_str(), base_flags | O_DIRECT, mode);
    // If O_DIRECT fails, we'll just use regular_fd_ for everything

    // Initialize libaio context
    aio_ctx_ = 0;
    int ret = io_setup(io_depth_, &aio_ctx_);
    if (ret < 0) {
      close(regular_fd_);
      regular_fd_ = -1;
      if (direct_fd_ >= 0) { close(direct_fd_); direct_fd_ = -1; }
      return false;
    }

    // Create eventfd for epoll integration
    event_fd_ = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (event_fd_ < 0) {
      io_destroy(aio_ctx_);
      aio_ctx_ = 0;
      close(regular_fd_);
      regular_fd_ = -1;
      if (direct_fd_ >= 0) { close(direct_fd_); direct_fd_ = -1; }
      return false;
    }

    return true;
  }

  ssize_t GetFileSize() const override {
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

    // Check if already completed
    auto it = completed_.find(token);
    if (it != completed_.end()) {
      result = it->second;
      completed_.erase(it);
      in_flight_.erase(token);
      return true;
    }

    // Not yet completed — try to harvest completions
    HarvestCompletions();

    // Check again after harvesting
    it = completed_.find(token);
    if (it != completed_.end()) {
      result = it->second;
      completed_.erase(it);
      in_flight_.erase(token);
      return true;
    }

    return false;
  }

  void Close() override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (event_fd_ >= 0) {
      close(event_fd_);
      event_fd_ = -1;
    }
    if (aio_ctx_ != 0) {
      io_destroy(aio_ctx_);
      aio_ctx_ = 0;
    }
    if (direct_fd_ >= 0) {
      close(direct_fd_);
      direct_fd_ = -1;
    }
    if (regular_fd_ >= 0) {
      close(regular_fd_);
      regular_fd_ = -1;
    }
    in_flight_.clear();
    completed_.clear();
  }

  int GetEventFd() const override {
    return event_fd_;
  }

 private:
  IoToken SubmitIO(void *buffer, size_t size, off_t offset, bool is_write) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Select fd based on alignment
    int fd = SelectFd(buffer, size);

    IoToken token = next_token_.fetch_add(1);

    struct iocb iocb_storage;
    struct iocb *iocb = &iocb_storage;
    memset(iocb, 0, sizeof(struct iocb));

    if (is_write) {
      io_prep_pwrite(iocb, fd, buffer, size, offset);
    } else {
      io_prep_pread(iocb, fd, buffer, size, offset);
    }

    io_set_eventfd(iocb, event_fd_);
    iocb->data = reinterpret_cast<void *>(token);

    struct iocb *iocbs[1] = {iocb};
    int submitted = io_submit(aio_ctx_, 1, iocbs);
    if (submitted != 1) {
      return kInvalidIoToken;
    }

    in_flight_.insert(token);
    return token;
  }

  int SelectFd(void *buffer, size_t size) const {
    // Use O_DIRECT fd if available and buffer+size are page-aligned
    if (direct_fd_ >= 0 &&
        (reinterpret_cast<uintptr_t>(buffer) % 4096 == 0) &&
        (size % 4096 == 0)) {
      return direct_fd_;
    }
    return regular_fd_;
  }

  void HarvestCompletions() {
    // Called with mutex_ held
    const int kMaxEvents = 32;
    struct io_event events[kMaxEvents];
    struct timespec timeout = {0, 0};  // Non-blocking

    int completed = io_getevents(aio_ctx_, 0, kMaxEvents, events, &timeout);
    if (completed <= 0) return;

    for (int i = 0; i < completed; ++i) {
      IoToken token = reinterpret_cast<IoToken>(events[i].data);
      IoResult res;
      long result = static_cast<long>(events[i].res);
      if (result < 0) {
        res.bytes_transferred = -1;
        res.error_code = static_cast<int>(-result);
      } else {
        res.bytes_transferred = result;
        res.error_code = 0;
      }
      completed_[token] = res;
    }

    // Drain eventfd
    uint64_t val;
    ssize_t ret = read(event_fd_, &val, sizeof(val));
    (void)ret;
  }

  io_context_t aio_ctx_;
  int event_fd_;
  int direct_fd_;
  int regular_fd_;
  uint32_t io_depth_;
  std::atomic<IoToken> next_token_;
  std::mutex mutex_;
  std::unordered_set<IoToken> in_flight_;
  std::unordered_map<IoToken, IoResult> completed_;
};

}  // namespace hshm

#endif  // HSHM_ENABLE_LIBAIO

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_IO_LIBAIO_IO_H_
