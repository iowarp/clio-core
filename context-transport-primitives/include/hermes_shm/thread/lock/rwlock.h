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

#ifndef HSHM_THREAD_RWLOCK_H_
#define HSHM_THREAD_RWLOCK_H_

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/thread/lock.h"
#include "hermes_shm/thread/thread_model_manager.h"
#include "hermes_shm/types/atomic.h"
#include "hermes_shm/types/numbers.h"

namespace hshm {

class RwLockMode {
 public:
  typedef int Type;
  CLS_CONST Type kNone = 0;
  CLS_CONST Type kWrite = 1;
  CLS_CONST Type kRead = 2;
};

/** A reader-writer lock implementation */
struct RwLock {
  ipc::atomic<RwLockMode::Type> mode_;
  ipc::atomic<hshm::reg_uint> readers_;
  ipc::atomic<hshm::reg_uint> writers_;
  ipc::atomic<hshm::reg_uint> cur_writer_;
  ipc::atomic<hshm::big_uint> ticket_;
  /** Default constructor */
  HSHM_CROSS_FUN
  RwLock()
      : readers_(0),
        writers_(0),
        ticket_(0),
        mode_(RwLockMode::kNone),
        cur_writer_(0) {}

  /** Explicit constructor */
  HSHM_CROSS_FUN
  void Init() {
    readers_ = 0;
    writers_ = 0;
    ticket_ = 0;
    mode_ = RwLockMode::kNone;
    cur_writer_ = 0;
  }

  /** Delete copy constructor */
  HSHM_CROSS_FUN
  RwLock(const RwLock &other) = delete;

  /** Move constructor */
  HSHM_CROSS_FUN
  RwLock(RwLock &&other) noexcept
      : readers_(other.readers_.load()),
        writers_(other.writers_.load()),
        ticket_(other.ticket_.load()),
        mode_(other.mode_.load()),
        cur_writer_(other.cur_writer_.load()) {}

  /** Move assignment operator */
  HSHM_CROSS_FUN
  RwLock &operator=(RwLock &&other) noexcept {
    if (this != &other) {
      readers_ = other.readers_.load();
      writers_ = other.writers_.load();
      ticket_ = other.ticket_.load();
      mode_ = other.mode_.load();
      cur_writer_ = other.cur_writer_.load();
    }
    return *this;
  }

  /** Acquire read lock */
  HSHM_CROSS_FUN
  void ReadLock(uint32_t owner) {
    RwLockMode::Type mode;

    // Increment # readers. Check if in read mode.
    readers_.fetch_add(1);

    // Wait until we are in read mode
    do {
      UpdateMode(mode);
      if (mode == RwLockMode::kRead) {
        return;
      }
      if (mode == RwLockMode::kNone) {
        bool ret = mode_.compare_exchange_weak(mode, RwLockMode::kRead);
        if (ret) {
          return;
        }
      }
      HSHM_THREAD_MODEL->Yield();
    } while (true);
  }

  /** Release read lock */
  HSHM_CROSS_FUN
  void ReadUnlock() { readers_.fetch_sub(1); }

  /** Acquire write lock */
  HSHM_CROSS_FUN
  void WriteLock(uint32_t owner) {
    RwLockMode::Type mode;
    uint32_t cur_writer;

    // Increment # writers & get ticket
    writers_.fetch_add(1);
    uint64_t tkt = ticket_.fetch_add(1);

    // Wait until we are in read mode
    do {
      UpdateMode(mode);
      if (mode == RwLockMode::kNone) {
        mode_.compare_exchange_weak(mode, RwLockMode::kWrite);
        mode = mode_.load();
      }
      if (mode == RwLockMode::kWrite) {
        cur_writer = cur_writer_.load();
        if (cur_writer == tkt) {
          return;
        }
      }
      HSHM_THREAD_MODEL->Yield();
    } while (true);
  }

  /** Release write lock */
  HSHM_CROSS_FUN
  void WriteUnlock() {
    writers_.fetch_sub(1);
    cur_writer_.fetch_add(1);
  }

 private:
  /** Update the mode of the lock */
  HSHM_INLINE_CROSS_FUN
  void UpdateMode(RwLockMode::Type &mode) {
    // When # readers is 0, there is a lag to when the mode is updated
    // When # writers is 0, there is a lag to when the mode is updated
    mode = mode_.load();
    if ((readers_.load() == 0 && mode == RwLockMode::kRead) ||
        (writers_.load() == 0 && mode == RwLockMode::kWrite)) {
      mode_.compare_exchange_weak(mode, RwLockMode::kNone);
    }
  }
};

/** Acquire the read lock in a scope */
struct ScopedRwReadLock {
  RwLock &lock_;
  bool is_locked_;

  /** Acquire the read lock */
  HSHM_CROSS_FUN
  explicit ScopedRwReadLock(RwLock &lock, uint32_t owner)
      : lock_(lock), is_locked_(false) {
    Lock(owner);
  }

  /** Release the read lock */
  HSHM_CROSS_FUN
  ~ScopedRwReadLock() { Unlock(); }

  /** Explicitly acquire read lock */
  HSHM_CROSS_FUN
  void Lock(uint32_t owner) {
    if (!is_locked_) {
      lock_.ReadLock(owner);
      is_locked_ = true;
    }
  }

  /** Explicitly release read lock */
  HSHM_CROSS_FUN
  void Unlock() {
    if (is_locked_) {
      lock_.ReadUnlock();
      is_locked_ = false;
    }
  }
};

/** Acquire scoped write lock */
struct ScopedRwWriteLock {
  RwLock &lock_;
  bool is_locked_;

  /** Acquire the write lock */
  HSHM_CROSS_FUN
  explicit ScopedRwWriteLock(RwLock &lock, uint32_t owner)
      : lock_(lock), is_locked_(false) {
    Lock(owner);
  }

  /** Release the write lock */
  HSHM_CROSS_FUN
  ~ScopedRwWriteLock() { Unlock(); }

  /** Explicity acquire the write lock */
  HSHM_CROSS_FUN
  void Lock(uint32_t owner) {
    if (!is_locked_) {
      lock_.WriteLock(owner);
      is_locked_ = true;
    }
  }

  /** Explicitly release the write lock */
  HSHM_CROSS_FUN
  void Unlock() {
    if (is_locked_) {
      lock_.WriteUnlock();
      is_locked_ = false;
    }
  }
};

}  // namespace hshm

namespace hshm::ipc {

using hshm::RwLock;
using hshm::ScopedRwReadLock;
using hshm::ScopedRwWriteLock;

}  // namespace hshm::ipc

#endif  // HSHM_THREAD_RWLOCK_H_
