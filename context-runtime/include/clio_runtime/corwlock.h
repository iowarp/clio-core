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

#ifndef CLIO_RUNTIME_INCLUDE_CORWLOCK_H_
#define CLIO_RUNTIME_INCLUDE_CORWLOCK_H_

#include <clio_ctp/thread/lock/rwlock.h>
#include "clio_runtime/types.h"

namespace clio::run {

/**
 * CoRwLock - Reentrant cooperative read-write lock for coroutine-based
 * task execution. Allows write->write and write->read reentrancy
 * from the same logical execution context (parent/subtask chain).
 *
 * SOUNDNESS: holder_ and the depth counters are plain fields; they may be
 * touched without the underlying lock ONLY by the OS thread that currently
 * owns the write lock. Every legitimate reentrant acquire runs on that
 * thread (same-stack nesting, or a same-worker subtask of the holding
 * parent — LockOwnerId embeds the worker id), so the fast path is gated on
 * holder_tid_, an atomic OS-thread id: another thread can never store OUR
 * tid, and our own release of it is visible to us in program order, so
 * (holder_tid_ == this thread) <=> this thread genuinely holds the write
 * lock right now. The previous version compared holder_ alone — an
 * unsynchronized read that could observe a STALE value equal to the
 * caller's own identity from an EARLIER hold, granting lock-free access
 * while another thread held the write lock. Under the #807 stress test
 * that let ToFullPtr read alloc_map_ mid-rehash (spurious misses of
 * registered allocators -> NULL-source memcpy in the RAM bdev), and the
 * same false grant on the write side double-entered critical sections and
 * corrupted the underlying RwLock — wedging every worker.
 */
class CoRwLock {
 public:
  ctp::RwLock lock_;
  LockOwnerId holder_;
  ctp::ipc::atomic<u64> holder_tid_;
  u32 write_depth_;
  u32 read_depth_;

  CoRwLock() : holder_tid_(0), write_depth_(0), read_depth_(0) {}

  /** Deleted copy constructor (matches ctp::RwLock) */
  CoRwLock(const CoRwLock &other) = delete;

  CoRwLock(CoRwLock &&other) noexcept
      : lock_(std::move(other.lock_)),
        holder_(other.holder_),
        holder_tid_(other.holder_tid_.load()),
        write_depth_(other.write_depth_),
        read_depth_(other.read_depth_) {
    other.holder_.Clear();
    other.holder_tid_.store(0);
    other.write_depth_ = 0;
    other.read_depth_ = 0;
  }

  CoRwLock &operator=(CoRwLock &&other) noexcept {
    if (this != &other) {
      lock_ = std::move(other.lock_);
      holder_ = other.holder_;
      holder_tid_.store(other.holder_tid_.load());
      write_depth_ = other.write_depth_;
      read_depth_ = other.read_depth_;
      other.holder_.Clear();
      other.holder_tid_.store(0);
      other.write_depth_ = 0;
      other.read_depth_ = 0;
    }
    return *this;
  }

  /** True iff the calling OS thread currently owns the write lock. Only
   *  then may holder_ / the depth counters be touched without lock_. */
  bool HeldByThisThread() const {
    u64 self = GetCoLockThreadId();
    return self != 0 &&
           holder_tid_.load(std::memory_order_acquire) == self;
  }

  void ReadLock() {
    // Write->read reentrancy for the holding thread's logical context.
    if (HeldByThisThread() && GetCurrentLockOwnerId() == holder_) {
      ++read_depth_;
      return;
    }
    lock_.ReadLock(0);
  }

  void ReadUnlock() {
    if (HeldByThisThread() && read_depth_ > 0 &&
        GetCurrentLockOwnerId() == holder_) {
      --read_depth_;
      return;
    }
    lock_.ReadUnlock();
  }

  void WriteLock() {
    if (HeldByThisThread() && GetCurrentLockOwnerId() == holder_) {
      ++write_depth_;
      return;
    }
    lock_.WriteLock(0);
    holder_ = GetCurrentLockOwnerId();
    holder_tid_.store(GetCoLockThreadId(), std::memory_order_release);
    write_depth_ = 1;
  }

  void WriteUnlock() {
    --write_depth_;
    if (write_depth_ == 0 && read_depth_ == 0) {
      holder_.Clear();
      holder_tid_.store(0, std::memory_order_release);
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

}  // namespace clio::run

#endif  // CLIO_RUNTIME_INCLUDE_CORWLOCK_H_
