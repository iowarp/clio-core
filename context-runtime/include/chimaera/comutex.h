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

#ifndef CHIMAERA_INCLUDE_CHIMAERA_COMUTEX_H_
#define CHIMAERA_INCLUDE_CHIMAERA_COMUTEX_H_

#include <hermes_shm/thread/lock/mutex.h>
#include "chimaera/types.h"

namespace chi {

/**
 * CoMutex - Reentrant cooperative mutex for coroutine-based task execution.
 * When a parent task holds the lock and spawns a subtask on the same worker,
 * the subtask can reacquire the lock without deadlocking.
 */
class CoMutex {
 public:
  hshm::Mutex lock_;
  LockOwnerId holder_;
  u32 depth_;

  CoMutex() : depth_(0) {}

  /** Copy constructor reinitializes to unlocked (needed for std::vector) */
  CoMutex(const CoMutex &other) : depth_(0) { (void)other; }

  void Lock() {
    LockOwnerId cur = GetCurrentLockOwnerId();
    if (cur == holder_) {
      ++depth_;
      return;
    }
    lock_.Lock(0);
    holder_ = cur;
    depth_ = 1;
  }

  bool TryLock() {
    LockOwnerId cur = GetCurrentLockOwnerId();
    if (cur == holder_) {
      ++depth_;
      return true;
    }
    if (!lock_.TryLock(0)) return false;
    holder_ = cur;
    depth_ = 1;
    return true;
  }

  void Unlock() {
    --depth_;
    if (depth_ == 0) {
      holder_.Clear();
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

}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_COMUTEX_H_
