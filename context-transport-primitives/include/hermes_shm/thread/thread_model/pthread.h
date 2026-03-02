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

#ifndef HSHM_THREAD_PTHREAD_H_
#define HSHM_THREAD_PTHREAD_H_

#if HSHM_ENABLE_PTHREADS

#include <errno.h>
#ifdef __APPLE__
#include <unistd.h>
#endif

#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/types/atomic.h"
#include "hermes_shm/util/errors.h"
#include "thread_model.h"

namespace hshm::thread {

class Pthread : public ThreadModel {
 public:
  ThreadLocalKey tid_key_;
  hipc::atomic<hshm::big_uint> tid_counter_;

 public:
  /** Default constructor */
  HSHM_INLINE_CROSS_FUN
  Pthread() : ThreadModel(ThreadType::kPthread) {
    tid_counter_ = 1;
    CreateTls<void>(tid_key_, nullptr);
  }

  /** Destructor */
  ~Pthread() = default;

  /** Initialize pthread */
  HSHM_CROSS_FUN
  void Init() {}

  /** Yield the thread for a period of time */
  HSHM_CROSS_FUN
  void SleepForUs(size_t us) {
#if HSHM_IS_HOST
    usleep(us);
#endif
  }

  /** Yield thread time slice */
  HSHM_CROSS_FUN
  void Yield() {
#if HSHM_IS_HOST
    sched_yield();
#endif
  }

  /** Create thread-local storage */
  template <typename TLS>
  HSHM_CROSS_FUN bool CreateTls(ThreadLocalKey &key, TLS *data) {
#if HSHM_IS_HOST
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
  HSHM_CROSS_FUN bool SetTls(ThreadLocalKey &key, TLS *data) {
#if HSHM_IS_HOST
    pthread_setspecific(key.pthread_key_, data);
    return true;
#else
    return false;
#endif
  }

  /** Get thread-local storage */
  template <typename TLS>
  HSHM_CROSS_FUN TLS *GetTls(const ThreadLocalKey &key) {
#if HSHM_IS_HOST
    TLS *data = (TLS *)pthread_getspecific(key.pthread_key_);
    return data;
#else
    return nullptr;
#endif
  }

  /** Get the TID of the current thread */
  HSHM_CROSS_FUN
  ThreadId GetTid() {
#if HSHM_IS_HOST
    size_t tid = (size_t)GetTls<void>(tid_key_);
    if (!tid) {
      tid = tid_counter_.fetch_add(1);
      SetTls<void>(tid_key_, (void *)tid);
    }
    tid -= 1;
    return ThreadId{(hshm::u64)tid};
#else
    return ThreadId{0};
#endif
  }

  /** Create a thread group */
  HSHM_CROSS_FUN
  ThreadGroup CreateThreadGroup(const ThreadGroupContext &ctx) {
    return ThreadGroup{};
  }

  /** Spawn a thread */
  template <typename FUNC, typename... Args>
  HSHM_CROSS_FUN Thread Spawn(ThreadGroup &group, FUNC &&func, Args &&...args) {
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
  HSHM_CROSS_FUN
  void Join(Thread &thread) {
#if HSHM_IS_HOST
    pthread_join(thread.pthread_thread_, nullptr);
#endif
  }
};

}  // namespace hshm::thread

#endif  // HSHM_ENABLE_PTHREADS

#endif  // HSHM_THREAD_PTHREAD_H_
