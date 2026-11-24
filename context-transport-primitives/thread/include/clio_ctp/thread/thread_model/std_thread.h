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

#ifndef CTP_THREAD_STD_THREAD_H_
#define CTP_THREAD_STD_THREAD_H_

// StdThread is always available as it uses standard C++ features
#include <thread>

#include "clio_ctp/introspect/system_info.h"
#include "thread_model.h"

namespace ctp::thread {

/** Represents the generic operations of a thread */
class StdThread : public ThreadModel {
 public:
  /** Initializer */
  CTP_INLINE_CROSS_FUN
  StdThread() : ThreadModel(ThreadType::kStdThread) {}

  /** Initialize std thread */
  CTP_CROSS_FUN
  void Init() {}

  /** Sleep thread for a period of time. Routes through SystemInfo so Windows
   *  gets a high-resolution waitable timer (sub-ms) instead of
   *  std::this_thread::sleep_for's coarse ~1-15.6ms tick. The Win32 timer code
   *  stays in system_info.cc, keeping <windows.h> out of this header. */
  CTP_CROSS_FUN
  void SleepForUs(size_t us) {
#if CTP_IS_HOST
    SystemInfo::SleepForUs(us);
#else
    (void)us;
#endif
  }

  /** Yield thread time slice */
  CTP_CROSS_FUN
  void Yield() { std::this_thread::yield(); }

  /** Create thread-local storage */
  template <typename TLS>
  CTP_CROSS_FUN bool CreateTls(ThreadLocalKey &key, TLS *data) {
#if CTP_IS_HOST
    return SystemInfo::CreateTls(key, (void *)data);
#else
    return false;
#endif
  }

  /** Create thread-local storage */
  template <typename TLS>
  CTP_CROSS_FUN bool SetTls(ThreadLocalKey &key, TLS *data) {
#if CTP_IS_HOST
    return SystemInfo::SetTls(key, (void *)data);
#else
    return false;
#endif
  }

  /** Get thread-local storage */
  template <typename TLS>
  CTP_CROSS_FUN TLS *GetTls(const ThreadLocalKey &key) {
#if CTP_IS_HOST
    return static_cast<TLS *>(SystemInfo::GetTls(key));
#else
    return nullptr;
#endif
  }

  /** Get the TID of the current thread */
  CTP_CROSS_FUN
  ThreadId GetTid() { return ThreadId(SystemInfo::GetTid()); }

  /** Get the thread model type */
  CTP_INLINE_CROSS_FUN
  ThreadType GetType() { return type_; }

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
    thread.std_thread_ =
        std::thread(std::forward<FUNC>(func), std::forward<Args>(args)...);
    return thread;
  }

  /** Join a thread */
  CTP_CROSS_FUN
  void Join(Thread &thread) { thread.std_thread_.join(); }

  /** Best-effort join with a timeout; detach if the timeout elapses. Returns
   *  true if the thread was joined cleanly, false if it had to be detached.
   *  std::thread has no portable timed-join, so on this model the timeout is
   *  ignored and a blocking join is performed. TODO: implement via native
   *  handle if a deadline is required on Windows. */
  CTP_CROSS_FUN
  bool TimedJoinOrDetach(Thread &thread, uint64_t /*timeout_ms*/) {
    if (!thread.std_thread_.joinable()) return true;
    thread.std_thread_.join();
    return true;
  }

  /** Set CPU affinity for thread */
  CTP_CROSS_FUN
  void SetAffinity(Thread &thread, int cpu_id) {}
};

}  // namespace ctp::thread

#endif  // CTP_THREAD_STD_THREAD_H_
