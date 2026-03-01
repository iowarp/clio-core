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

#ifndef HSHM_THREAD_MUTEX_H_
#define HSHM_THREAD_MUTEX_H_

#include "hermes_shm/thread/thread_model_manager.h"
#include "hermes_shm/types/atomic.h"
#include "hermes_shm/types/numbers.h"

namespace hshm {

struct Mutex {
  ipc::atomic<hshm::min_u64> lock_;
  ipc::atomic<hshm::min_u64> head_;
  ipc::atomic<hshm::min_u32> try_lock_;
  /** Default constructor */
  HSHM_INLINE_CROSS_FUN
  Mutex() : lock_(0), head_(0), try_lock_(0) {}

  /** Copy constructor */
  HSHM_INLINE_CROSS_FUN
  Mutex(const Mutex &other) {}

  /** Explicit initialization */
  HSHM_INLINE_CROSS_FUN
  void Init() {
    lock_ = 0;
    head_ = 0;
  }

  /** Acquire lock */
  HSHM_INLINE_CROSS_FUN
  void Lock(u32 owner) {
    min_u64 tkt = lock_.fetch_add(1);
    do {
      for (int i = 0; i < 1; ++i) {
        if (tkt == head_.load()) {
          return;
        }
      }
      HSHM_THREAD_MODEL->Yield();
    } while (true);
  }

  /** Try to acquire the lock */
  HSHM_INLINE_CROSS_FUN
  bool TryLock(u32 owner) {
    if (try_lock_.fetch_add(1) > 0 || lock_.load() > head_.load()) {
      try_lock_.fetch_sub(1);
      return false;
    }
    Lock(owner);
    return true;
  }

  /** Unlock */
  HSHM_INLINE_CROSS_FUN
  void Unlock() {
    head_.fetch_add(1);
  }
};

struct ScopedMutex {
  Mutex &lock_;
  bool is_locked_;

  /** Acquire the mutex */
  HSHM_INLINE_CROSS_FUN explicit ScopedMutex(Mutex &lock, u32 owner)
      : lock_(lock), is_locked_(false) {
    Lock(owner);
  }

  /** Release the mutex */
  HSHM_INLINE_CROSS_FUN
  ~ScopedMutex() { Unlock(); }

  /** Explicitly acquire the mutex */
  HSHM_INLINE_CROSS_FUN
  void Lock(u32 owner) {
    if (!is_locked_) {
      lock_.Lock(owner);
      is_locked_ = true;
    }
  }

  /** Explicitly try to lock the mutex */
  HSHM_INLINE_CROSS_FUN
  bool TryLock(u32 owner) {
    if (!is_locked_) {
      is_locked_ = lock_.TryLock(owner);
    }
    return is_locked_;
  }

  /** Explicitly unlock the mutex */
  HSHM_INLINE_CROSS_FUN
  void Unlock() {
    if (is_locked_) {
      lock_.Unlock();
      is_locked_ = false;
    }
  }
};

}  // namespace hshm

namespace hshm::ipc {

using hshm::Mutex;
using hshm::ScopedMutex;

}  // namespace hshm::ipc

#undef Mutex
#undef ScopedMutex

#endif  // HSHM_THREAD_MUTEX_H_
