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

/**
 * @file async_log.cc
 * @brief The background log thread of async_log.h.
 */

#include "clio_cte/compressor/async_log.h"

#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <thread>
#include <utility>

namespace clio::cte::compressor {

namespace {

/** @brief The queue and its thread; leaked on purpose (runs until exit). */
struct LogQueue {
  std::mutex mu;
  std::condition_variable has_work;  ///< signalled when work is queued
  std::condition_variable idle;      ///< signalled when the queue empties
  std::deque<std::function<void()>> work;
  bool busy = false;  ///< the thread is running a closure
};

/** Run queued closures in order, forever. */
void LogThreadMain(LogQueue *q) {
  std::unique_lock<std::mutex> lock(q->mu);
  for (;;) {
    q->has_work.wait(lock, [q] { return !q->work.empty(); });
    auto job = std::move(q->work.front());
    q->work.pop_front();
    q->busy = true;
    lock.unlock();
    job();
    lock.lock();
    q->busy = false;
    if (q->work.empty()) q->idle.notify_all();
  }
}

/** @return the queue, starting its thread and the exit drain on first use. */
LogQueue *Queue() {
  static LogQueue *q = [] {
    auto *queue = new LogQueue();
    std::thread(LogThreadMain, queue).detach();
    std::atexit(DrainLogWork);
    return queue;
  }();
  return q;
}

}  // namespace

void PostLogWork(std::function<void()> work) {
  LogQueue *q = Queue();
  {
    std::lock_guard<std::mutex> lock(q->mu);
    q->work.push_back(std::move(work));
  }
  q->has_work.notify_one();
}

void DrainLogWork() {
  LogQueue *q = Queue();
  std::unique_lock<std::mutex> lock(q->mu);
  q->idle.wait(lock, [q] { return q->work.empty() && !q->busy; });
}

}  // namespace clio::cte::compressor
