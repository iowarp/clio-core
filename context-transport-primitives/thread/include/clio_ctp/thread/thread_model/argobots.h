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

#ifndef CTP_THREAD_ARGOBOTS_H_
#define CTP_THREAD_ARGOBOTS_H_

#if CTP_ENABLE_THALLIUM

#include <errno.h>

#include <thallium.hpp>

#include "clio_ctp/introspect/system_info.h"
#include "clio_ctp/util/errors.h"
#include "thread_model.h"

namespace ctp::thread {

class Argobots : public ThreadModel {
 public:
  /** Default constructor */
  CTP_INLINE_CROSS_FUN
  Argobots() : ThreadModel(ThreadType::kArgobots) {}

  /** Destructor */
  CTP_CROSS_FUN
  ~Argobots() = default;

  /** Initialize Argobots */
  CTP_CROSS_FUN
  void Init() { ABT_init(0, nullptr); }

  /** Yield the current thread for a period of time */
  CTP_CROSS_FUN
  void SleepForUs(size_t us) {
    /**
     * TODO(llogan): make this API flexible enough to support argobots fully
     * tl::thread::self().sleep(*CTP->rpc_.server_engine_,
                               CTP->server_config_.borg_.blob_reorg_period_);
     */
#if CTP_IS_HOST
    usleep(us);
#endif
  }

  /** Yield thread time slice */
  CTP_CROSS_FUN
  void Yield() {
#if CTP_IS_HOST
    ABT_thread_yield();
#endif
  }

  /** Create thread-local storage */
  template <typename TLS>
  CTP_CROSS_FUN bool CreateTls(ThreadLocalKey &key, TLS *data) {
#if CTP_IS_HOST
    int ret = ABT_key_create(ThreadLocalData::template destroy_wrap<TLS>,
                             &key.argobots_key_);
    if (ret != ABT_SUCCESS) {
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
    int ret = ABT_key_set(key.argobots_key_, data);
    return ret == ABT_SUCCESS;
#else
    return false;
#endif
  }

  /** Get thread-local storage */
  template <typename TLS>
  CTP_CROSS_FUN TLS *GetTls(const ThreadLocalKey &key) {
#if CTP_IS_HOST
    TLS *data;
    ABT_key_get(key.argobots_key_, (void **)&data);
    return (TLS *)data;
#else
    return nullptr;
#endif
  }

  /** Get the TID of the current thread */
  CTP_CROSS_FUN
  ThreadId GetTid() {
#if CTP_IS_HOST
    ABT_thread thread;
    ABT_thread_id tid;
    ABT_thread_self(&thread);
    ABT_thread_get_id(thread, &tid);
    return ThreadId{tid};
#else
    return ThreadId{0};
#endif
  }

  /** Create a thread group */
  CTP_CROSS_FUN
  ThreadGroup CreateThreadGroup(const ThreadGroupContext &ctx) {
#if CTP_IS_HOST
    ABT_xstream xstream;
    ABT_xstream_create(ABT_SCHED_NULL, &xstream);
    return ThreadGroup{xstream};
#else
    return ThreadGroup{nullptr};
#endif
  }

  /** Spawn a thread */
  template <typename FUNC, typename... Args>
  CTP_CROSS_FUN Thread Spawn(ThreadGroup &group, FUNC &&func, Args &&...args) {
#if CTP_IS_HOST
    Thread thread;
    ThreadParams<FUNC, Args...> *params = new ThreadParams<FUNC, Args...>(
        std::forward<FUNC>(func), std::forward<Args>(args)...);
    thread.group_ = group;
    ABT_thread_create_on_xstream(group.abtxstream_, SpawnWrapper<FUNC, Args...>,
                                 (void *)params, ABT_THREAD_ATTR_NULL,
                                 &thread.abt_thread_);
    return thread;
#else
    return Thread{};
#endif
  }

  /** Wrapper for spawning a thread */
  template <typename FUNC, typename... Args>
  static void SpawnWrapper(void *arg) {
    ThreadParams<FUNC, Args...> *params =
        static_cast<ThreadParams<FUNC, Args...> *>(arg);
    PassArgPack::Call(std::forward<ArgPack<Args...>>(params->args_),
                      std::forward<FUNC>(params->func_));
    delete params;
  }

  /** Join a thread */
  CTP_CROSS_FUN
  void Join(Thread &thread) {
#if CTP_IS_HOST
    ABT_thread_join(thread.abt_thread_);
#endif
  }

  /** Set CPU affinity for thread */
  CTP_CROSS_FUN
  void SetAffinity(Thread &thread, int cpu_id) {
#if CTP_IS_HOST
    ABT_xstream_set_affinity(thread.group_.abtxstream_, 1, &cpu_id);
#endif
  }
};

}  // namespace ctp::thread

#endif  // CTP_ENABLE_THALLIUM

#endif  // CTP_THREAD_ARGOBOTS_H_
