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

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_IO_IOURING_IO_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_IO_IOURING_IO_H_

#if HSHM_ENABLE_IO_URING

#include "async_io.h"
#include <liburing.h>
#include <sys/eventfd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace hshm {

class IoUringAsyncIO : public AsyncIO {
 public:
  IoUringAsyncIO(uint32_t io_depth)
      : initialized_(false), event_fd_(-1), direct_fd_(-1), regular_fd_(-1),
        io_depth_(io_depth), next_token_(1) {
    memset(&ring_, 0, sizeof(ring_));
  }

  ~IoUringAsyncIO() override {
    Close();
  }

  bool Open(const std::string &path, int flags, mode_t mode) override {
    std::lock_guard<std::mutex> lock(mutex_);

    int base_flags = flags & ~O_DIRECT;

    regular_fd_ = open(path.c_str(), base_flags, mode);
    if (regular_fd_ < 0) return false;

    direct_fd_ = open(path.c_str(), base_flags | O_DIRECT, mode);

    int ret = io_uring_queue_init(io_depth_, &ring_, 0);
    if (ret < 0) {
      close(regular_fd_); regular_fd_ = -1;
      if (direct_fd_ >= 0) { close(direct_fd_); direct_fd_ = -1; }
      return false;
    }
    initialized_ = true;

    event_fd_ = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (event_fd_ >= 0) {
      ret = io_uring_register_eventfd(&ring_, event_fd_);
      if (ret < 0) {
        close(event_fd_);
        event_fd_ = -1;
      }
    }

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

    auto it = completed_.find(token);
    if (it != completed_.end()) {
      result = it->second;
      completed_.erase(it);
      in_flight_.erase(token);
      return true;
    }

    HarvestCompletions();

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
    if (initialized_) {
      io_uring_queue_exit(&ring_);
      initialized_ = false;
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

    struct io_uring_sqe *sqe = io_uring_get_sqe(&ring_);
    if (!sqe) return kInvalidIoToken;

    int fd = SelectFd(buffer, size);
    IoToken token = next_token_.fetch_add(1);

    if (is_write) {
      io_uring_prep_write(sqe, fd, buffer, size, offset);
    } else {
      io_uring_prep_read(sqe, fd, buffer, size, offset);
    }

    io_uring_sqe_set_data64(sqe, token);

    int ret = io_uring_submit(&ring_);
    if (ret <= 0) return kInvalidIoToken;

    in_flight_.insert(token);
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

  void HarvestCompletions() {
    const int kMaxEvents = 32;
    struct io_uring_cqe *cqes[kMaxEvents];
    unsigned count = io_uring_peek_batch_cqe(&ring_, cqes, kMaxEvents);

    for (unsigned i = 0; i < count; ++i) {
      IoToken token = io_uring_cqe_get_data64(cqes[i]);
      IoResult res;
      if (cqes[i]->res < 0) {
        res.bytes_transferred = -1;
        res.error_code = -cqes[i]->res;
      } else {
        res.bytes_transferred = cqes[i]->res;
        res.error_code = 0;
      }
      completed_[token] = res;
      io_uring_cqe_seen(&ring_, cqes[i]);
    }

    if (count > 0 && event_fd_ >= 0) {
      uint64_t val;
      ssize_t ret = read(event_fd_, &val, sizeof(val));
      (void)ret;
    }
  }

  struct io_uring ring_;
  bool initialized_;
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

#endif  // HSHM_ENABLE_IO_URING

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_IO_IOURING_IO_H_
