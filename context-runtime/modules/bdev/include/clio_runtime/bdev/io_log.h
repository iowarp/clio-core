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

#ifndef CLIO_RUNTIME_BDEV_IO_LOG_H_
#define CLIO_RUNTIME_BDEV_IO_LOG_H_

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

namespace clio::run::bdev {

/**
 * One line per DEVICE WRITE: when it started and how long it took.
 *
 * A benchmark cannot otherwise say how much of an arm's wall-clock was the
 * bytes actually moving. The compressor's per-chunk `io_ms` is the await
 * latency of the put coroutine -- it includes task hand-offs and scheduling,
 * sums to many times the elapsed loop, and measures a durable file write on an
 * untiered arm but a memcpy on a tiered one. This log instead brackets the
 * transfer itself inside the bdev transport, so a caller can take the UNION of
 * the intervals (writes overlap) and get elapsed I/O.
 *
 * The bracket matches upstream NeuroPress's VOL boundary: the device-to-host
 * copy of the compressed image counts as I/O, together with the write it feeds
 * (gpucompress_hdf5_vol: vol_d2h_copy_ms + vol_io_queue_wait_ms + drain).
 *
 * OFF unless CLIO_IO_LOG names a file: one fprintf per write is cheap next to
 * an 8 MiB transfer, but nothing pays for it in production.
 */
class IoLog {
 public:
  /** The process-wide log. */
  static IoLog *Get() {
    static IoLog log;
    return &log;
  }

  /** True when CLIO_IO_LOG asked for a log; callers can skip the clock. */
  bool enabled() const { return fp_ != nullptr; }

  /** Steady-clock nanoseconds, the stamp Record() expects. */
  static long long Now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }

  /**
   * Append one completed write.
   *
   * @param tier which device class moved the bytes ("file" or "ram").
   * @param bytes bytes this write transferred.
   * @param start_ns steady-clock nanoseconds when it began, from Now().
   * @param ms how long it took, milliseconds.
   */
  void Record(const char *tier, unsigned long long bytes, long long start_ns,
              double ms) {
    if (fp_ == nullptr) return;
    std::lock_guard<std::mutex> lock(mutex_);
    std::fprintf(fp_, "%s,%llu,%lld,%.6f\n", tier, bytes, start_ns, ms);
  }

 private:
  IoLog() {
    const char *path = std::getenv("CLIO_IO_LOG");
    if (path == nullptr || *path == '\0') return;
    fp_ = std::fopen(path, "w");
    if (fp_ != nullptr) std::fprintf(fp_, "tier,bytes,start_ns,ms\n");
  }

  ~IoLog() {
    if (fp_ != nullptr) std::fclose(fp_);
  }

  std::FILE *fp_ = nullptr;
  std::mutex mutex_;
};

}  // namespace clio::run::bdev

#endif  // CLIO_RUNTIME_BDEV_IO_LOG_H_
