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

#ifndef CLIO_CTE_COMPRESSOR_CODEC_POOL_H_
#define CLIO_CTE_COMPRESSOR_CODEC_POOL_H_

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace clio::cte::compressor {

/**
 * A small fixed pool of host threads for CPU codec work.
 *
 * A runtime worker that runs a codec inline blocks every task queued behind
 * it for the whole decode (a 1 MB bzip2 page is tens of ms). GPU page faults
 * reach the compressor on few workers, so inline decoding serializes them.
 * Handlers instead Submit() the codec call here and yield until it is done,
 * which lets one worker keep many decodes in flight.
 */
class CodecPool {
 public:
  /**
   * Start the pool.
   * @param threads worker threads (0 = half the hardware threads, >= 2)
   */
  explicit CodecPool(unsigned threads = 0) {
    if (threads == 0) {
      threads = std::max(2u, std::thread::hardware_concurrency() / 2);
    }
    for (unsigned i = 0; i < threads; ++i) {
      threads_.emplace_back([this] { Loop(); });
    }
  }

  ~CodecPool() {
    {
      std::lock_guard<std::mutex> g(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto &t : threads_) t.join();
  }

  CodecPool(const CodecPool &) = delete;
  CodecPool &operator=(const CodecPool &) = delete;

  /**
   * Queue `fn` to run on a pool thread.
   * @param fn the job; it must signal its own completion to the submitter
   */
  void Submit(std::function<void()> fn) {
    {
      std::lock_guard<std::mutex> g(mu_);
      jobs_.push_back(std::move(fn));
    }
    cv_.notify_one();
  }

 private:
  /** Pool thread body: run jobs until stopped and drained. */
  void Loop() {
    for (;;) {
      std::function<void()> fn;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this] { return stop_ || !jobs_.empty(); });
        if (jobs_.empty()) return;  // stop_ and drained
        fn = std::move(jobs_.front());
        jobs_.pop_front();
      }
      fn();
    }
  }

  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::function<void()>> jobs_;
  bool stop_ = false;
  std::vector<std::thread> threads_;
};

}  // namespace clio::cte::compressor

#endif  // CLIO_CTE_COMPRESSOR_CODEC_POOL_H_
