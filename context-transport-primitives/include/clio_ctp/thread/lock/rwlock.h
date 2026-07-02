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

#ifndef CTP_THREAD_RWLOCK_H_
#define CTP_THREAD_RWLOCK_H_

#include "clio_ctp/constants/macros.h"
#include "clio_ctp/thread/lock.h"
#include "clio_ctp/thread/thread_model_manager.h"
#include "clio_ctp/types/atomic.h"
#include "clio_ctp/types/numbers.h"

namespace ctp {

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
  ipc::atomic<ctp::reg_uint> readers_;
  ipc::atomic<ctp::reg_uint> writers_;
  ipc::atomic<ctp::reg_uint> cur_writer_;
  ipc::atomic<ctp::big_uint> ticket_;
  /** Default constructor */
  CTP_CROSS_FUN
  RwLock()
      : readers_(0),
        writers_(0),
        ticket_(0),
        mode_(RwLockMode::kNone),
        cur_writer_(0) {}

  /** Explicit constructor */
  CTP_CROSS_FUN
  void Init() {
    readers_ = 0;
    writers_ = 0;
    ticket_ = 0;
    mode_ = RwLockMode::kNone;
    cur_writer_ = 0;
  }

  /** Copy constructor (no-op, mirrors ctp::Mutex): copying a live
   *  lock would clone its state, which is never what callers want, but
   *  containers (e.g. priv::vector<RwLock>::operator=) need *some*
   *  copy constructor to be callable. Construct a fresh, unheld lock. */
  CTP_CROSS_FUN
  RwLock(const RwLock & /*other*/)
      : readers_(0),
        writers_(0),
        ticket_(0),
        mode_(RwLockMode::kNone),
        cur_writer_(0) {}

  /** Copy assignment (no-op for state, same rationale as copy ctor). */
  CTP_CROSS_FUN
  RwLock &operator=(const RwLock & /*other*/) {
    readers_.store(0);
    writers_.store(0);
    ticket_.store(0);
    mode_.store(RwLockMode::kNone);
    cur_writer_.store(0);
    return *this;
  }

  /** Move constructor */
  CTP_CROSS_FUN
  RwLock(RwLock &&other) noexcept
      : readers_(other.readers_.load()),
        writers_(other.writers_.load()),
        ticket_(other.ticket_.load()),
        mode_(other.mode_.load()),
        cur_writer_(other.cur_writer_.load()) {}

  /** Move assignment operator */
  CTP_CROSS_FUN
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

  /** Acquire read lock.
   *
   *  @param owner           legacy/unused owner id.
   *  @param writer_priority when true, make this otherwise reader-preferring
   *         lock writer-preferring for THIS acquisition: defer entering while a
   *         writer is waiting on or holds the lock. This prevents a sustained
   *         reader stream from starving writers — the reader-preferring default
   *         lets a reader in whenever the lock is already in read mode,
   *         regardless of a waiting writer, so writers can livelock (observed as
   *         an icx/Windows ctest timeout in the unordered_map_ll growth stress
   *         test). The wait happens BEFORE registering as a reader: incrementing
   *         readers_ first and only then waiting for writers_==0 would deadlock
   *         the writer, which spins for readers_==0. Stragglers that slip past
   *         before a writer bumps writers_ are bounded — the writer's
   *         ticket-based WriteLock drains them — so no reader is blocked
   *         forever. Default false preserves the reader-preferring behavior that
   *         reentrant callers (e.g. CoRwLock's read->read nesting) depend on. */
  CTP_CROSS_FUN
  void ReadLock(uint32_t owner, bool writer_priority = false) {
    RwLockMode::Type mode;

    if (writer_priority) {
      while (writers_.load() > 0) {
#if !CTP_IS_DEVICE_PASS
        CTP_THREAD_MODEL->Yield();
#endif
      }
    }

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
#if !CTP_IS_DEVICE_PASS
      CTP_THREAD_MODEL->Yield();
#endif
    } while (true);
  }

  /** Release read lock */
  CTP_CROSS_FUN
  void ReadUnlock() { readers_.fetch_sub(1); }

  /** Acquire write lock */
  CTP_CROSS_FUN
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
        // Use load_device() for cross-SM L2 visibility on GPU.
        mode = mode_.load_device();
      }
      if (mode == RwLockMode::kWrite) {
        // Use load_device() for cross-SM L2 visibility on GPU.
        cur_writer = cur_writer_.load_device();
        if (cur_writer == tkt) {
          return;
        }
      }
#if !CTP_IS_DEVICE_PASS
      CTP_THREAD_MODEL->Yield();
#endif
    } while (true);
  }

  /** Release write lock */
  CTP_CROSS_FUN
  void WriteUnlock() {
    writers_.fetch_sub(1);
    cur_writer_.fetch_add(1);
  }

  /** @return true while a writer holds (or is waiting/draining for) this lock.
   *  Used by ContainerPtr::IsPlugged() so a reader can observe that a
   *  migration/upgrade writer is in progress on the container. */
  CTP_CROSS_FUN
  bool IsWriteLocked() const { return writers_.load() > 0; }

 private:
  /** Update the mode of the lock */
  CTP_INLINE_CROSS_FUN
  void UpdateMode(RwLockMode::Type &mode) {
    // When # readers is 0, there is a lag to when the mode is updated
    // When # writers is 0, there is a lag to when the mode is updated
    // Use load_device() for cross-SM L2 visibility on GPU.
    mode = mode_.load_device();
    if ((readers_.load_device() == 0 && mode == RwLockMode::kRead) ||
        (writers_.load_device() == 0 && mode == RwLockMode::kWrite)) {
      mode_.compare_exchange_weak(mode, RwLockMode::kNone);
    }
  }
};

/** Acquire the read lock in a scope */
struct ScopedRwReadLock {
  RwLock &lock_;
  bool is_locked_;

  /** Acquire the read lock */
  CTP_CROSS_FUN
  explicit ScopedRwReadLock(RwLock &lock, uint32_t owner)
      : lock_(lock), is_locked_(false) {
    Lock(owner);
  }

  /** Release the read lock */
  CTP_CROSS_FUN
  ~ScopedRwReadLock() { Unlock(); }

  /** Explicitly acquire read lock */
  CTP_CROSS_FUN
  void Lock(uint32_t owner) {
    if (!is_locked_) {
      lock_.ReadLock(owner);
      is_locked_ = true;
    }
  }

  /** Explicitly release read lock */
  CTP_CROSS_FUN
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
  CTP_CROSS_FUN
  explicit ScopedRwWriteLock(RwLock &lock, uint32_t owner)
      : lock_(lock), is_locked_(false) {
    Lock(owner);
  }

  /** Release the write lock */
  CTP_CROSS_FUN
  ~ScopedRwWriteLock() { Unlock(); }

  /** Explicity acquire the write lock */
  CTP_CROSS_FUN
  void Lock(uint32_t owner) {
    if (!is_locked_) {
      lock_.WriteLock(owner);
      is_locked_ = true;
    }
  }

  /** Explicitly release the write lock */
  CTP_CROSS_FUN
  void Unlock() {
    if (is_locked_) {
      lock_.WriteUnlock();
      is_locked_ = false;
    }
  }
};

}  // namespace ctp

namespace ctp::ipc {

using ctp::RwLock;
using ctp::ScopedRwReadLock;
using ctp::ScopedRwWriteLock;

}  // namespace ctp::ipc

#endif  // CTP_THREAD_RWLOCK_H_
