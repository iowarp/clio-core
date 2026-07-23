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

#ifndef CLIO_RUNTIME_INCLUDE_COMUTEX_H_
#define CLIO_RUNTIME_INCLUDE_COMUTEX_H_

#include <clio_ctp/thread/lock/mutex.h>
#include "clio_runtime/types.h"

namespace clio::run {

/**
 * CoMutex - Reentrant cooperative mutex for coroutine-based task execution.
 * When a parent task holds the lock and spawns a subtask on the same worker,
 * the subtask can reacquire the lock without deadlocking.
 *
 * SOUNDNESS: the reentrancy fast path is gated on holder_tid_, an atomic
 * OS-thread id — every legitimate reentrant acquire runs on the OS thread
 * that already owns the lock (same-stack nesting, or a same-worker subtask
 * of the holding parent), and only that gate makes the unsynchronized
 * holder_/depth_ accesses sound. Comparing holder_ alone (the previous
 * version) is a data race: a thread could observe a STALE holder_ equal to
 * its own identity from an EARLIER hold and enter the critical section
 * while another thread held the mutex. See CoRwLock for the full analysis
 * (#807 stress-test corruption).
 */
class CoMutex {
 public:
  ctp::Mutex lock_;
  LockOwnerId holder_;
  ctp::ipc::atomic<u64> holder_tid_;
  u32 depth_;

  CoMutex() : holder_tid_(0), depth_(0) {}

  /** Copy constructor reinitializes to unlocked (needed for std::vector) */
  CoMutex(const CoMutex &other) : holder_tid_(0), depth_(0) { (void)other; }

  /** True iff the calling OS thread currently owns the mutex. Only then may
   *  holder_ / depth_ be touched without lock_. */
  bool HeldByThisThread() const {
    u64 self = GetCoLockThreadId();
    return self != 0 &&
           holder_tid_.load(std::memory_order_acquire) == self;
  }

  void Lock() {
    if (HeldByThisThread() && GetCurrentLockOwnerId() == holder_) {
      ++depth_;
      return;
    }
    lock_.Lock(0);
    holder_ = GetCurrentLockOwnerId();
    holder_tid_.store(GetCoLockThreadId(), std::memory_order_release);
    depth_ = 1;
  }

  bool TryLock() {
    if (HeldByThisThread() && GetCurrentLockOwnerId() == holder_) {
      ++depth_;
      return true;
    }
    if (!lock_.TryLock(0)) return false;
    holder_ = GetCurrentLockOwnerId();
    holder_tid_.store(GetCoLockThreadId(), std::memory_order_release);
    depth_ = 1;
    return true;
  }

  void Unlock() {
    --depth_;
    if (depth_ == 0) {
      holder_.Clear();
      holder_tid_.store(0, std::memory_order_release);
      lock_.Unlock();
    }
  }
};

/**
 * ScopedCoMutex - RAII mutex wrapper for CoMutex
 */
struct ScopedCoMutex {
  CoMutex &lock_;

  explicit ScopedCoMutex(CoMutex &lock) : lock_(lock) { lock_.Lock(); }

  ~ScopedCoMutex() { lock_.Unlock(); }
};

}  // namespace clio::run

#endif  // CLIO_RUNTIME_INCLUDE_COMUTEX_H_
