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

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_IO_POSIX_AIO_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_IO_POSIX_AIO_H_

#if !defined(_WIN32)

#include "async_io.h"
#include <aio.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <sys/types.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <memory>
#include <vector>

// O_DIRECT is a Linux extension; Darwin has no equivalent open flag and
// instead exposes uncached I/O through fcntl(F_NOCACHE) post-open. Define
// it as 0 so the `flags & ~O_DIRECT` / `flags | O_DIRECT` bitops still
// compile and the "direct" open path degrades gracefully into a second
// regular fd. Tests that exercise the aligned-buffer path still pass
// because aligned reads/writes work fine through the page cache.
#ifndef O_DIRECT
#define O_DIRECT 0
#endif

namespace ctp {

/*
 * Why a pool of descriptors: glibc's POSIX AIO keeps ONE request queue per
 * file descriptor and runs every request for the same aio_fildes strictly
 * sequentially on one helper thread. A bdev tier is a single backing file, so
 * with one fd all I/O to a tier was serialized. Measured on Aurora (IO500
 * through clio-fs on DAOS dfuse / Lustre, where libaio submits synchronously):
 * FUSE read callbacks averaged 45-115 ms with ~10 in flight, i.e. only
 * ~100-200 backing I/Os per second per node regardless of client count.
 * Duplicating the fd gives glibc several independent queues to run in
 * parallel. glibc also caps its helper pool at 20 threads by default, which
 * kPosixAioMinHelperThreads lifts (glibc only).
 */

/** Default descriptors per kind (buffered / O_DIRECT), original fd included. */
inline constexpr size_t kPosixAioDefaultFdsPerKind = 16;
/** Smallest accepted CTP_POSIX_AIO_FDS value (1 = original single-fd path). */
inline constexpr size_t kPosixAioMinFdsPerKind = 1;
/** Largest accepted CTP_POSIX_AIO_FDS value. */
inline constexpr size_t kPosixAioMaxFdsPerKind = 64;
/** Floor for glibc's aio_threads helper pool size. */
inline constexpr size_t kPosixAioMinHelperThreads = 64;
/** glibc aio_num: max outstanding requests per process. */
inline constexpr size_t kPosixAioMaxOutstanding = 1024;

class PosixAsyncIO : public AsyncIO {
 public:
  PosixAsyncIO(uint32_t io_depth)
      : direct_fd_(-1), regular_fd_(-1), next_token_(1), rr_counter_(0) {
    (void)io_depth;
  }

  ~PosixAsyncIO() override {
    Close();
  }

  /**
   * Open the file twice (buffered and O_DIRECT) and build the descriptor
   * pools. Pool width comes from CTP_POSIX_AIO_FDS (see ResolveFdsPerKind).
   * @param path file to open
   * @param flags open flags; O_DIRECT is stripped from the buffered fd
   * @param mode creation mode
   * @return false if the buffered open fails; a failed O_DIRECT open is
   *         tolerated and only disables the direct pool
   */
  bool Open(const std::string &path, int flags, mode_t mode) override {
    std::lock_guard<std::mutex> lock(mutex_);

    int base_flags = flags & ~O_DIRECT;

    regular_fd_ = open(path.c_str(), base_flags, mode);
    if (regular_fd_ < 0) return false;

    // CTP_POSIX_AIO_DIRECT=0 keeps every request on the buffered descriptors.
    // A tier file sees aligned requests (O_DIRECT) and unaligned ones
    // (buffered) for neighbouring ranges; once the pool runs them
    // concurrently, direct writes to a Lustre file failed with EIO
    // (IO500 ior-hard on Aurora), as mixing the two modes on one range can.
    const char *direct_env = std::getenv("CTP_POSIX_AIO_DIRECT");
    const bool use_direct =
        direct_env == nullptr || std::string(direct_env) != "0";
    direct_fd_ =
        use_direct ? open(path.c_str(), base_flags | O_DIRECT, mode) : -1;

    size_t fds_per_kind = ResolveFdsPerKind();
    InitGlibcAioOnce(fds_per_kind);
    regular_fds_ = BuildFdPool(regular_fd_, fds_per_kind);
    if (direct_fd_ >= 0) {
      direct_fds_ = BuildFdPool(direct_fd_, fds_per_kind);
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

  IoToken Write(void *buffer, size_t size, int64_t offset) override {
    return SubmitIO(buffer, size, offset, true);
  }

  IoToken Read(void *buffer, size_t size, int64_t offset) override {
    return SubmitIO(buffer, size, offset, false);
  }

  bool IsComplete(IoToken token, IoResult &result) override {
    std::lock_guard<std::mutex> lock(mutex_);

    auto done = done_.find(token);
    if (done != done_.end()) {
      result = done->second;
      done_.erase(done);
      return true;
    }

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

  /**
   * Cancel all pending requests on the descriptor each was submitted to, then
   * close every pooled descriptor (originals included) exactly once.
   */
  void Close() override {
    std::lock_guard<std::mutex> lock(mutex_);

    // Requests are spread over the pool, so cancel each on its own fd.
    for (auto &kv : pending_) {
      aio_cancel(kv.second->aio_fildes, kv.second.get());
    }
    pending_.clear();
    done_.clear();

    // Slot 0 of each pool is the original fd, so closing the pools covers it.
    CloseFdPool(direct_fds_);
    CloseFdPool(regular_fds_);
    direct_fd_ = -1;
    regular_fd_ = -1;
  }

  int GetEventFd() const override {
    return -1;  // POSIX AIO does not support eventfd
  }

 protected:
  /** The kernel's per-request control block (POSIX struct aiocb). */
  using ControlBlock = struct aiocb;

  /**
   * Hand one request to the kernel's POSIX AIO. Virtual so tests can make it
   * refuse with EAGAIN, which macOS does once a process has more than
   * kern.aio.process_max (16 by default) requests outstanding.
   * @param cb the control block to submit
   * @param is_write true for aio_write, false for aio_read
   * @return 0 on success, else -1 with errno set
   */
  virtual int StartAio(ControlBlock *cb, bool is_write) {
    return is_write ? aio_write(cb) : aio_read(cb);
  }

 private:
  IoToken SubmitIO(void *buffer, size_t size, int64_t offset, bool is_write) {
    std::lock_guard<std::mutex> lock(mutex_);

    int fd = SelectFd(buffer, size, offset);
    IoToken token = next_token_.fetch_add(1);

    auto cb = std::make_unique<struct aiocb>();
    memset(cb.get(), 0, sizeof(struct aiocb));

    cb->aio_fildes = fd;
    cb->aio_buf = buffer;
    cb->aio_nbytes = size;
    cb->aio_offset = offset;

    if (StartAio(cb.get(), is_write) == 0) {
      pending_[token] = std::move(cb);
      return token;
    }
    if (errno != EAGAIN) {
      return kInvalidIoToken;
    }
    // Issue #1156: EAGAIN means the AIO request limit is reached, not that
    // the I/O is bad. Failing it surfaced as a bdev write error with 0 bytes
    // (safe_bdev fans one put out to more members than macOS allows), so do
    // the I/O synchronously and report it through IsComplete as usual.
    done_[token] = SyncIO(fd, buffer, size, offset, is_write);
    return token;
  }

  /**
   * Perform one request synchronously, retrying short transfers.
   * @param fd file descriptor
   * @param buffer I/O buffer
   * @param size bytes to transfer
   * @param offset file offset
   * @param is_write true to write, false to read
   * @return bytes transferred and errno, in AsyncIO's result form
   */
  static IoResult SyncIO(int fd, void *buffer, size_t size, int64_t offset,
                         bool is_write) {
    IoResult result;
    char *p = static_cast<char *>(buffer);
    size_t done = 0;
    while (done < size) {
      ssize_t n = is_write ? pwrite(fd, p + done, size - done,
                                    static_cast<off_t>(offset + done))
                           : pread(fd, p + done, size - done,
                                   static_cast<off_t>(offset + done));
      if (n < 0 && errno == EINTR) continue;
      if (n < 0) {
        result.bytes_transferred = -1;
        result.error_code = errno;
        return result;
      }
      if (n == 0) break;  // read past end of file
      done += static_cast<size_t>(n);
    }
    result.bytes_transferred = static_cast<ssize_t>(done);
    result.error_code = 0;
    return result;
  }

  /**
   * Read the descriptor pool width from CTP_POSIX_AIO_FDS. Unset or
   * unparsable values use kPosixAioDefaultFdsPerKind; set values are clamped
   * to [kPosixAioMinFdsPerKind, kPosixAioMaxFdsPerKind].
   * @return descriptors per kind, original fd included
   */
  static size_t ResolveFdsPerKind() {
    const char *env = std::getenv("CTP_POSIX_AIO_FDS");
    if (env == nullptr || *env == '\0') return kPosixAioDefaultFdsPerKind;
    char *end = nullptr;
    long value = std::strtol(env, &end, 10);
    if (end == env || *end != '\0') return kPosixAioDefaultFdsPerKind;
    if (value < static_cast<long>(kPosixAioMinFdsPerKind)) {
      return kPosixAioMinFdsPerKind;
    }
    if (value > static_cast<long>(kPosixAioMaxFdsPerKind)) {
      return kPosixAioMaxFdsPerKind;
    }
    return static_cast<size_t>(value);
  }

  /**
   * Raise glibc's AIO helper-thread and outstanding-request limits. Runs once
   * per process, before the first aio submission, because aio_init only
   * takes effect for requests issued after it. Non-glibc platforms (macOS)
   * keep their own limits; their EAGAIN path falls back to synchronous I/O.
   * @param fds_per_kind pool width in use, sizes the helper thread count
   */
  static void InitGlibcAioOnce(size_t fds_per_kind) {
#if defined(__GLIBC__)
    static std::once_flag aio_init_once;
    std::call_once(aio_init_once, [fds_per_kind]() {
      struct aioinit init;
      std::memset(&init, 0, sizeof(init));
      init.aio_threads = static_cast<int>(
          std::max(kPosixAioMinHelperThreads, fds_per_kind * 2));
      init.aio_num = static_cast<int>(kPosixAioMaxOutstanding);
      aio_init(&init);
    });
#else
    (void)fds_per_kind;
#endif
  }

  /**
   * Build one descriptor pool by dup()-ing the original fd. Slot 0 is the
   * original; each dup is a distinct fd and so a distinct glibc per-fd queue.
   * A dup() failure stops growth and is not an error.
   * @param original_fd fd returned by open(), must be >= 0
   * @param width total slots wanted, original included
   * @return the pool; slot 0 is original_fd
   */
  static std::vector<int> BuildFdPool(int original_fd, size_t width) {
    std::vector<int> pool{original_fd};
    while (pool.size() < width) {
      int dup_fd = dup(original_fd);
      if (dup_fd < 0) break;
      pool.push_back(dup_fd);
    }
    return pool;
  }

  /**
   * Close every descriptor in a pool and empty the pool.
   * @param pool descriptors to close; cleared on return
   */
  static void CloseFdPool(std::vector<int> &pool) {
    for (int fd : pool) {
      close(fd);
    }
    pool.clear();
  }

  /**
   * Choose the descriptor for one request. The O_DIRECT pool is used only
   * when the buffer, the length AND the file offset are all
   * DirectIoAlign()-aligned (4 KiB, or the page size when that is larger --
   * Lustre needs it); anything else uses the buffered pool. The offset check
   * was missing: an aligned buffer written at an unaligned file offset (a
   * truncate zeroing a page tail) went to O_DIRECT and failed with EINVAL.
   * Within the chosen pool, descriptors are taken round-robin so concurrent
   * requests spread over independent glibc per-fd queues.
   * @param buffer I/O buffer
   * @param size I/O length
   * @param offset file offset
   * @return fd to submit on, or -1 if the file is not open
   */
  int SelectFd(void *buffer, size_t size, int64_t offset) const {
    bool aligned =
        (reinterpret_cast<uintptr_t>(buffer) % DirectIoAlign() == 0) &&
        (size % DirectIoAlign() == 0) &&
        (offset % static_cast<int64_t>(DirectIoAlign()) == 0);
    const std::vector<int> &pool =
        (aligned && !direct_fds_.empty()) ? direct_fds_ : regular_fds_;
    if (pool.empty()) return -1;
    size_t slot =
        rr_counter_.fetch_add(1, std::memory_order_relaxed) % pool.size();
    return pool[slot];
  }

  int direct_fd_;
  int regular_fd_;
  std::vector<int> direct_fds_;   /**< O_DIRECT pool; slot 0 is direct_fd_ */
  std::vector<int> regular_fds_;  /**< buffered pool; slot 0 is regular_fd_ */
  std::atomic<IoToken> next_token_;
  mutable std::atomic<size_t> rr_counter_; /**< round-robin pool cursor */
  std::mutex mutex_;
  std::unordered_map<IoToken, std::unique_ptr<struct aiocb>> pending_;
  std::unordered_map<IoToken, IoResult> done_;  /**< completed synchronously */
};

}  // namespace ctp

#endif  // !defined(_WIN32)

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_IO_POSIX_AIO_H_
