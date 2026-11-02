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

#ifndef CTP_THREAD_THREAD_H_
#define CTP_THREAD_THREAD_H_

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "clio_ctp/types/argpack.h"
#include "clio_ctp/types/bitfield.h"
#include "clio_ctp/types/numbers.h"

#if CTP_ENABLE_PTHREADS
#include <pthread.h>
#endif
#if CTP_ENABLE_THALLIUM
#include <thallium.hpp>
#endif
#include <thread>

namespace ctp {

/** Available threads that are mapped */
enum class ThreadType { kNone, kPthread, kArgobots, kCuda, kRocm, kStdThread };

/** Thread-local key */
union ThreadLocalKey {
#if CTP_ENABLE_PTHREADS
  pthread_key_t pthread_key_;
#endif
#if CTP_ENABLE_THALLIUM
  ABT_key argobots_key_;
#endif
#if CTP_ENABLE_WINDOWS_THREADS
  DWORD windows_key_;
#endif
};

/** Thread Group Context */
struct ThreadGroupContext {
  // NOTE(llogan): Argobots supports various schedulers, etc.
  int nothing_;
};

/** Thread group */
struct ThreadGroup {
#if CTP_ENABLE_THALLIUM
  ABT_xstream abtxstream_ = nullptr;
#endif
};

template <typename FUN, typename... Args>
struct ThreadParams {
  FUN func_;
  ArgPack<Args...> args_;

  ThreadParams(FUN &&func, Args &&...args)
      : func_(std::forward<FUN>(func)), args_(std::forward<Args>(args)...) {}
};

/** Thread */
struct Thread {
  ThreadGroup group_;
#if CTP_ENABLE_THALLIUM
  ABT_thread abt_thread_ = nullptr;
#endif
#if CTP_ENABLE_PTHREADS
  pthread_t pthread_thread_;
#endif
  std::thread std_thread_;
};

}  // namespace ctp

namespace ctp::thread {

/** Thread-local key */
using ctp::ThreadLocalKey;

/** Thread group */
using ctp::ThreadGroup;

/** Thread */
using ctp::Thread;

/** Thread group context */
using ctp::ThreadGroupContext;

/** Thread-local storage */
class ThreadLocalData {
 public:
  // CTP_CROSS_FUN
  // void destroy() = 0;

  template <typename TLS>
  CTP_CROSS_FUN static void destroy_wrap(void *data) {
    if (data) {
      // TODO(llogan): Figure out why this segfaults on exit
      //   if constexpr (std::is_base_of_v<ThreadLocalData, TLS>) {
      //     static_cast<TLS *>(data)->destroy();
      //   }
    }
  }
};

/** Represents the generic operations of a thread */
class ThreadModel {
 public:
  ThreadType type_;

 public:
  /** Initializer */
  CTP_INLINE_CROSS_FUN
  explicit ThreadModel(ThreadType type) : type_(type) {}

  /** Get the thread model type */
  CTP_INLINE_CROSS_FUN
  ThreadType GetType() { return type_; }

#if CTP_IS_HOST
  /**
   * Wait until cond() returns true, escalating so the fast path stays
   * low-latency without pinning a core indefinitely on slow waits:
   *   1. Tight spin (no yield) for ~200us  — reacts within ~one cond() check
   *      of completion (~100ns), avoiding the tens-of-microseconds scheduling
   *      latency a bare Yield()/Sleep() costs on Windows.
   *   2. Yield-spin for ~5ms               — hands the time slice back each
   *      iteration but stays hot.
   *   3. Exponentially growing SleepForUs  — near-zero CPU for long waits.
   *
   * Defined once in thread_model.cc against CTP_THREAD_MODEL's uniform
   * Yield()/SleepForUs(), so a single implementation serves every thread
   * model. The condition is evaluated frequently; keep it cheap and
   * side-effect-free (e.g. an atomic flag load).
   */
  CTP_DLL void BusyWait(const std::function<bool()> &cond);
#endif
};

}  // namespace ctp::thread

#endif  // CTP_THREAD_THREAD_H_
