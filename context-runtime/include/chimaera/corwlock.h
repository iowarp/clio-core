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

#ifndef CHIMAERA_INCLUDE_CHIMAERA_CORWLOCK_H_
#define CHIMAERA_INCLUDE_CHIMAERA_CORWLOCK_H_

#include <hermes_shm/thread/lock/rwlock.h>
#include "chimaera/types.h"

namespace chi {

/**
 * CoRwLock - Reentrant cooperative read-write lock for coroutine-based
 * task execution. Allows write->write and write->read reentrancy
 * from the same logical execution context (parent/subtask chain).
 */
class CoRwLock {
 public:
  hshm::RwLock lock_;
  LockOwnerId holder_;
  u32 write_depth_;
  u32 read_depth_;

  CoRwLock() : write_depth_(0), read_depth_(0) {}

  /** Deleted copy constructor (matches hshm::RwLock) */
  CoRwLock(const CoRwLock &other) = delete;

  CoRwLock(CoRwLock &&other) noexcept
      : lock_(std::move(other.lock_)),
        holder_(other.holder_),
        write_depth_(other.write_depth_),
        read_depth_(other.read_depth_) {
    other.holder_.Clear();
    other.write_depth_ = 0;
    other.read_depth_ = 0;
  }

  CoRwLock &operator=(CoRwLock &&other) noexcept {
    if (this != &other) {
      lock_ = std::move(other.lock_);
      holder_ = other.holder_;
      write_depth_ = other.write_depth_;
      read_depth_ = other.read_depth_;
      other.holder_.Clear();
      other.write_depth_ = 0;
      other.read_depth_ = 0;
    }
    return *this;
  }

  void ReadLock() {
    LockOwnerId cur = GetCurrentLockOwnerId();
    if (cur == holder_) {
      ++read_depth_;
      return;
    }
    lock_.ReadLock(0);
  }

  void ReadUnlock() {
    LockOwnerId cur = GetCurrentLockOwnerId();
    if (cur == holder_ && read_depth_ > 0) {
      --read_depth_;
      return;
    }
    lock_.ReadUnlock();
  }

  void WriteLock() {
    LockOwnerId cur = GetCurrentLockOwnerId();
    if (cur == holder_) {
      ++write_depth_;
      return;
    }
    lock_.WriteLock(0);
    holder_ = cur;
    write_depth_ = 1;
  }

  void WriteUnlock() {
    --write_depth_;
    if (write_depth_ == 0 && read_depth_ == 0) {
      holder_.Clear();
      lock_.WriteUnlock();
    }
  }
};

/**
 * ScopedCoRwReadLock - RAII read lock wrapper for CoRwLock
 */
struct ScopedCoRwReadLock {
  CoRwLock &lock_;

  explicit ScopedCoRwReadLock(CoRwLock &lock) : lock_(lock) {
    lock_.ReadLock();
  }

  ~ScopedCoRwReadLock() { lock_.ReadUnlock(); }
};

/**
 * ScopedCoRwWriteLock - RAII write lock wrapper for CoRwLock
 */
struct ScopedCoRwWriteLock {
  CoRwLock &lock_;

  explicit ScopedCoRwWriteLock(CoRwLock &lock) : lock_(lock) {
    lock_.WriteLock();
  }

  ~ScopedCoRwWriteLock() { lock_.WriteUnlock(); }
};

}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_CORWLOCK_H_
