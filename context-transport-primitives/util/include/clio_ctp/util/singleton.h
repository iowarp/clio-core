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

#ifndef CTP_SHM_SINGLETON_H
#define CTP_SHM_SINGLETON_H

#include <atomic>
#include <memory>

#include "clio_ctp/constants/macros.h"
#include "clio_ctp/thread/lock/spin_lock.h"

namespace ctp {

/**
 * A class to represent singleton pattern
 * Does not require specific initialization of the static variable
 *
 * NOTE(llogan): Python does NOT play well with this singleton.
 * I find that it will duplicate the singleton when loading wrapper
 * functions. It is very strange, but this one should be avoided for
 * codes that plan to be called by python.
 * */
template <typename T, bool WithLock>
class SingletonBase {
 public:
  static T *GetInstance() {
    if (GetObject() == nullptr) {
      if constexpr (WithLock) {
        ctp::ScopedSpinLock lock(GetSpinLock(), 0);
        new ((T *)GetData()) T();
        GetObject() = (T *)GetData();
      } else {
        new ((T *)GetData()) T();
        GetObject() = (T *)GetData();
      }
    }
    return GetObject();
  }

  static ctp::SpinLock &GetSpinLock() {
    // alignas: SpinLock holds ipc::atomic<u64> members needing 8-byte
    // alignment. A bare char[] is only 1-byte aligned, so on aarch64 the
    // atomic load/fetch_add instructions (LDADD/LDAXR) fault with SIGBUS when
    // this storage lands off an 8-byte boundary (as it does inside the python
    // extension .so). x86 tolerates the misalignment; ARM does not.
    alignas(ctp::SpinLock) static char spinlock_data_[sizeof(ctp::SpinLock)] = {0};
    return *(ctp::SpinLock *)spinlock_data_;
  }

  static T *GetData() {
    // alignas(T): a bare char[] is 1-byte aligned; if T contains atomics
    // (e.g. SystemInfo) an unaligned atomic access SIGBUSes on aarch64.
    alignas(T) static char data_[sizeof(T)] = {0};
    return (T *)data_;
  }

  static T *&GetObject() {
    static T *obj_ = nullptr;
    return obj_;
  }
};

/** Singleton default case declaration */
template <typename T>
using Singleton = SingletonBase<T, true>;

/** Singleton without lock declaration */
template <typename T>
using LockfreeSingleton = SingletonBase<T, false>;

/**
 * A class to represent singleton pattern
 * Does not require specific initialization of the static variable
 * */
template <typename T, bool WithLock>
class CrossSingletonBase {
 public:
  CTP_INLINE_CROSS_FUN
  static T *GetInstance() {
    if (GetObject() == nullptr) {
      if constexpr (WithLock) {
        ctp::ScopedSpinLock lock(GetSpinLock(), 0);
        new ((T *)GetData()) T();
        GetObject() = (T *)GetData();
      } else {
        new ((T *)GetData()) T();
        GetObject() = (T *)GetData();
      }
    }
    return GetObject();
  }

  CTP_INLINE_CROSS_FUN
  static ctp::SpinLock &GetSpinLock() {
    // alignas: SpinLock holds ipc::atomic<u64> members needing 8-byte
    // alignment. A bare char[] is only 1-byte aligned, so on aarch64 the
    // atomic load/fetch_add instructions (LDADD/LDAXR) fault with SIGBUS when
    // this storage lands off an 8-byte boundary (as it does inside the python
    // extension .so). x86 tolerates the misalignment; ARM does not.
    alignas(ctp::SpinLock) static char spinlock_data_[sizeof(ctp::SpinLock)] = {0};
    return *(ctp::SpinLock *)spinlock_data_;
  }

  CTP_INLINE_CROSS_FUN
  static T *GetData() {
    // alignas(T): a bare char[] is 1-byte aligned; if T contains atomics
    // (e.g. SystemInfo) an unaligned atomic access SIGBUSes on aarch64.
    alignas(T) static char data_[sizeof(T)] = {0};
    return (T *)data_;
  }

  CTP_INLINE_CROSS_FUN
  static T *&GetObject() {
    static T *obj_ = nullptr;
    return obj_;
  }
};

/** Singleton default case declaration */
template <typename T>
using CrossSingleton = CrossSingletonBase<T, true>;

/** Singleton without lock declaration */
template <typename T>
using LockfreeCrossSingleton = CrossSingletonBase<T, false>;

/**
 * Makes a singleton. Constructs during initialization of program.
 * Does not require specific initialization of the static variable.
 * */
template <typename T>
class GlobalSingleton {
 private:
  static T obj_;

 public:
  GlobalSingleton() = default;

  static T *GetInstance() { return &obj_; }
};
template <typename T>
T GlobalSingleton<T>::obj_;

/**
 * Makes a singleton. Constructs during initialization of program.
 * Does not require specific initialization of the static variable.
 * */
#if CTP_IS_HOST
template <typename T>
using GlobalCrossSingleton = GlobalSingleton<T>;
#else
template <typename T>
using GlobalCrossSingleton = LockfreeCrossSingleton<T>;
#endif

/**
 * C-style singleton with global variables
 */
#define CTP_DEFINE_GLOBAL_VAR_H(T, NAME) extern __TU(T) NAME;
#define CTP_DEFINE_GLOBAL_VAR_CC(T, NAME) __TU(T) NAME = T{};
#define CTP_GET_GLOBAL_VAR(T, NAME) ctp::GetGlobalVar<__TU(T)>(NAME)
template <typename T>
static inline T *GetGlobalVar(T &instance) {
  return &instance;
}

/**
 * Cross-device C-style singleton with global variables
 */
#if CTP_IS_HOST
#define CTP_DEFINE_GLOBAL_CROSS_VAR_H(T, NAME) extern __TU(T) NAME;
#define CTP_DEFINE_GLOBAL_CROSS_VAR_CC(T, NAME) __TU(T) NAME = T{};
#define CTP_GET_GLOBAL_CROSS_VAR(T, NAME) \
  ctp::GetGlobalCrossVar<__TU(T)>(NAME)
template <typename T>
CTP_CROSS_FUN static inline T *GetGlobalCrossVar(T &instance) {
  return &instance;
}
#else
#define CTP_DEFINE_GLOBAL_CROSS_VAR_H(T, NAME)
#define CTP_DEFINE_GLOBAL_CROSS_VAR_CC(T, NAME)
#define CTP_GET_GLOBAL_CROSS_VAR(T, NAME) \
  ctp::CrossSingleton<__TU(T)>::GetInstance()
#endif

/**
 * C-style pointer singleton with global variables.
 *
 * No DLL decoration here: globals declared via this macro are typically
 * local to a single DLL, or — when accessed across DLL boundaries on
 * Windows — must be decorated with a per-DLL API macro at the use site
 * (Windows requires explicit __declspec(dllimport) on data symbols
 * imported from another DLL; CMake's WINDOWS_EXPORT_ALL_SYMBOLS handles
 * function symbols but not data).
 */
#define CTP_DEFINE_GLOBAL_PTR_VAR_H(T, NAME) \
  extern ::std::atomic<__TU(T) *> NAME;
#define CTP_DEFINE_GLOBAL_PTR_VAR_CC(T, NAME) \
  ::std::atomic<__TU(T) *> NAME{nullptr};
#define CTP_GET_GLOBAL_PTR_VAR(T, NAME) ctp::GetGlobalPtrVar<__TU(T)>(NAME)
/**
 * Publish-once lazy accessor for a process-global singleton pointer.
 *
 * This used to be an unsynchronized check-then-set:
 *
 *   if (instance == nullptr) { instance = new T(); }
 *
 * Two threads could both observe null, each construct a T, and the last
 * writer would win the variable -- silently orphaning the other instance.
 * For CLIO_POOL_MANAGER that was not merely wasteful but fatal: ServerInit()
 * would run on the orphan, every later lookup would return the blank
 * survivor, and since a blank manager can never become initialized the
 * route-retry loop would spin its whole 30s deadline and fail the task with
 * (u32)-1. That is the shared root cause of issues #923 (Linux leak-check),
 * #928 (macOS adapters) and the windows-11-arm safe_bdev failure in #929 --
 * the "two instance addresses" signature those issues were filed on.
 *
 * The CAS below makes construction publish-once: whoever installs the
 * pointer first wins, the loser destroys its speculative instance and
 * adopts the winner's. Correctness does not depend on a shared lock, which
 * matters because this function is inlined into every module -- a
 * function-local static mutex would be per-DLL on Windows and so would not
 * actually serialize the threads racing on this one shared variable.
 *
 * The variable is std::atomic<T*> rather than a raw T*: acquire/release on
 * the pointer is what orders the constructor's writes against another
 * thread's first dereference, and unlike std::atomic_ref it is available on
 * every toolchain this project builds with (atomic_ref needs libc++ 19,
 * newer than the AppleClang used on the macOS runners).
 */
template <typename T>
static inline T *GetGlobalPtrVar(::std::atomic<T *> &instance) {
  T *observed = instance.load(::std::memory_order_acquire);
  if (observed != nullptr) {
    return observed;
  }
  T *fresh = new T();
  if (instance.compare_exchange_strong(observed, fresh,
                                       ::std::memory_order_acq_rel,
                                       ::std::memory_order_acquire)) {
    return fresh;
  }
  // Lost the publish race: another thread installed its instance first.
  // `observed` now holds the winner, which is the one everyone must share.
  //
  // The delete is well-defined even for a polymorphic T with a non-virtual
  // destructor: `fresh` came from `new T()` two lines up, so the static and
  // dynamic types are identical and no base-pointer slicing is possible.
  // -Wdelete-non-virtual-dtor cannot see that and fires in every TU that
  // instantiates this template (ConfigManager, IpcManager, ... are all
  // polymorphic), so silence it narrowly here rather than leaving warnings
  // scattered across the build.
#if defined(__GNUC__) || defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wdelete-non-virtual-dtor"
#endif
  delete fresh;
#if defined(__GNUC__) || defined(__clang__)
#  pragma GCC diagnostic pop
#endif
  return observed;
}

/**
 * Cross-device C-style pointer singleton with global variables
 */
#if CTP_IS_HOST
#define CTP_DEFINE_GLOBAL_CROSS_PTR_VAR_H(T, NAME) extern __TU(T) * NAME;
#define CTP_DEFINE_GLOBAL_CROSS_PTR_VAR_CC(T, NAME) __TU(T) *NAME = nullptr;
#define CTP_GET_GLOBAL_CROSS_PTR_VAR(T, NAME) \
  ctp::GetGlobalCrossPtrVar<__TU(T)>(NAME)
template <typename T>
CTP_CROSS_FUN static inline T *GetGlobalCrossPtrVar(T *&instance) {
  if (instance == nullptr) {
    instance = new T();
  }
  return instance;
}
#else
#define CTP_DEFINE_GLOBAL_CROSS_PTR_VAR_H(T, NAME)
#define CTP_DEFINE_GLOBAL_CROSS_PTR_VAR_CC(T, NAME)
#define CTP_GET_GLOBAL_CROSS_PTR_VAR(T, NAME) \
  ctp::CrossSingleton<__TU(T)>::GetInstance()
#endif

}  // namespace ctp

#endif  // CTP_SHM_SINGLETON_H
