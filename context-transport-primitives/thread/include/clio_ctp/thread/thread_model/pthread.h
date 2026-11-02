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

#ifndef CTP_THREAD_PTHREAD_H_
#define CTP_THREAD_PTHREAD_H_

#if CTP_ENABLE_PTHREADS

#include <errno.h>
#ifdef __APPLE__
#include <signal.h>
#include <time.h>
#include <unistd.h>
#endif

#include "clio_ctp/introspect/system_info.h"
#include "clio_ctp/types/atomic.h"
#include "clio_ctp/util/errors.h"
#include "thread_model.h"

namespace ctp::thread {

class Pthread : public ThreadModel {
 public:
  ThreadLocalKey tid_key_;
  ctp::ipc::atomic<ctp::big_uint> tid_counter_;

 public:
  /** Default constructor */
  CTP_INLINE_CROSS_FUN
  Pthread() : ThreadModel(ThreadType::kPthread) {
    tid_counter_ = 1;
    CreateTls<void>(tid_key_, nullptr);
  }

  /** Destructor */
  ~Pthread() = default;

  /** Initialize pthread */
  CTP_CROSS_FUN
  void Init() {}

  /** Yield the thread for a period of time */
  CTP_CROSS_FUN
  void SleepForUs(size_t us) {
#if CTP_IS_HOST
    usleep(us);
#endif
  }

  /** Yield thread time slice.
   *
   * Used inside tight spin loops while waiting on an atomic flag (e.g. ring
   * buffer Recv, FUTURE_COMPLETE polling). It MUST be cheap — Linux rounds
   * any nanosleep below the hrtimer slack (~50µs by default) UP to that
   * slack, which destroys IPC latency. A previous implementation called
   * nanosleep({0, 100ns}); each call actually slept ~54µs and inflated
   * single-RTT SHM latency from ~1µs to ~58µs. Use a CPU pause hint
   * instead — it is <20ns, is a no-op for correctness, and lets the CPU
   * relax branch prediction / pipeline. Callers that want to back off to
   * sched_yield or to truly sleep should escalate explicitly. */
  CTP_CROSS_FUN
  void Yield() {
#if CTP_IS_HOST
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield" ::: "memory");
#else
    sched_yield();
#endif
#endif
  }

  /** Create thread-local storage */
  template <typename TLS>
  CTP_CROSS_FUN bool CreateTls(ThreadLocalKey &key, TLS *data) {
#if CTP_IS_HOST
    int ret = pthread_key_create(&key.pthread_key_,
                                 ThreadLocalData::destroy_wrap<TLS>);
    if (ret != 0) {
      return false;
    }
    return SetTls(key, data);
#else
    return false;
#endif
  }

  /** Create thread-local storage */
  template <typename TLS>
  CTP_CROSS_FUN bool SetTls(ThreadLocalKey &key, TLS *data) {
#if CTP_IS_HOST
    pthread_setspecific(key.pthread_key_, data);
    return true;
#else
    return false;
#endif
  }

  /** Get thread-local storage */
  template <typename TLS>
  CTP_CROSS_FUN TLS *GetTls(const ThreadLocalKey &key) {
#if CTP_IS_HOST
    TLS *data = (TLS *)pthread_getspecific(key.pthread_key_);
    return data;
#else
    return nullptr;
#endif
  }

  /** Get the TID of the current thread */
  CTP_CROSS_FUN
  ThreadId GetTid() {
#if CTP_IS_HOST
    size_t tid = (size_t)GetTls<void>(tid_key_);
    if (!tid) {
      tid = tid_counter_.fetch_add(1);
      SetTls<void>(tid_key_, (void *)tid);
    }
    tid -= 1;
    return ThreadId{(ctp::u64)tid};
#else
    return ThreadId{0};
#endif
  }

  /** Create a thread group */
  CTP_CROSS_FUN
  ThreadGroup CreateThreadGroup(const ThreadGroupContext &ctx) {
    return ThreadGroup{};
  }

  /** Spawn a thread */
  template <typename FUNC, typename... Args>
  CTP_CROSS_FUN Thread Spawn(ThreadGroup &group, FUNC &&func, Args &&...args) {
    Thread thread;
    thread.group_ = group;
    ThreadParams<FUNC, Args...> *params = new ThreadParams<FUNC, Args...>(
        std::forward<FUNC>(func), std::forward<Args>(args)...);
    pthread_create(&thread.pthread_thread_, nullptr,
                   SpawnWrapper<FUNC, Args...>, (void *)params);
    return thread;
  }

  /** Wrapper for spawning a thread */
  template <typename FUNC, typename... Args>
  static void *SpawnWrapper(void *arg) {
    ThreadParams<FUNC, Args...> *params =
        static_cast<ThreadParams<FUNC, Args...> *>(arg);
    PassArgPack::Call(std::forward<ArgPack<Args...>>(params->args_),
                      std::forward<FUNC>(params->func_));
    delete params;
    return nullptr;
  }

  /** Join a thread */
  CTP_CROSS_FUN
  void Join(Thread &thread) {
#if CTP_IS_HOST
    pthread_join(thread.pthread_thread_, nullptr);
#endif
  }

  /** Best-effort join with a timeout; detach if the timeout elapses. Returns
   *  true if the thread was joined cleanly, false if it had to be detached. */
  CTP_CROSS_FUN
  bool TimedJoinOrDetach(Thread &thread, uint64_t timeout_ms) {
#if CTP_IS_HOST
    if (thread.pthread_thread_ == 0) return true;
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t nsec = timeout_ms * 1000000ull;
    ts.tv_sec  += nsec / 1000000000ull;
    ts.tv_nsec += nsec % 1000000000ull;
    if (ts.tv_nsec >= 1000000000LL) { ts.tv_sec++; ts.tv_nsec -= 1000000000LL; }
#ifdef __APPLE__
    // pthread_timedjoin_np is a glibc extension and is not available on
    // Darwin. Poll with pthread_kill(thread, 0): returns 0 while alive,
    // ESRCH once the thread has exited, at which point we can join it
    // without blocking. Safe here because thread.pthread_thread_ is
    // cleared after join/detach, so we never poll a stale handle.
    int r = ETIMEDOUT;
    uint64_t deadline_ns =
        static_cast<uint64_t>(ts.tv_sec) * 1000000000ull +
        static_cast<uint64_t>(ts.tv_nsec);
    while (true) {
      if (pthread_kill(thread.pthread_thread_, 0) == ESRCH) {
        r = pthread_join(thread.pthread_thread_, nullptr);
        break;
      }
      struct timespec now;
      clock_gettime(CLOCK_REALTIME, &now);
      uint64_t now_ns =
          static_cast<uint64_t>(now.tv_sec) * 1000000000ull +
          static_cast<uint64_t>(now.tv_nsec);
      if (now_ns >= deadline_ns) { r = ETIMEDOUT; break; }
      struct timespec sleep_ts = {0, 1000000};  // 1ms poll interval
      nanosleep(&sleep_ts, nullptr);
    }
#else
    int r = pthread_timedjoin_np(thread.pthread_thread_, nullptr, &ts);
#endif
    if (r == 0) {
      thread.pthread_thread_ = 0;
      return true;
    }
    pthread_detach(thread.pthread_thread_);
    thread.pthread_thread_ = 0;
    return false;
#else
    (void)thread; (void)timeout_ms;
    return true;
#endif
  }
};

}  // namespace ctp::thread

#endif  // CTP_ENABLE_PTHREADS

#endif  // CTP_THREAD_PTHREAD_H_
