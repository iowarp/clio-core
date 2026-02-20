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

#ifndef HSHM_SHM_SINGLETON_H
#define HSHM_SHM_SINGLETON_H

#include <memory>

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/thread/lock/spin_lock.h"

namespace hshm {

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
        hshm::ScopedSpinLock lock(GetSpinLock(), 0);
        new ((T *)GetData()) T();
        GetObject() = (T *)GetData();
      } else {
        new ((T *)GetData()) T();
        GetObject() = (T *)GetData();
      }
    }
    return GetObject();
  }

  static hshm::SpinLock &GetSpinLock() {
    static char spinlock_data_[sizeof(hshm::SpinLock)] = {0};
    return *(hshm::SpinLock *)spinlock_data_;
  }

  static T *GetData() {
    static char data_[sizeof(T)] = {0};
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
  HSHM_INLINE_CROSS_FUN
  static T *GetInstance() {
    if (GetObject() == nullptr) {
      if constexpr (WithLock) {
        hshm::ScopedSpinLock lock(GetSpinLock(), 0);
        new ((T *)GetData()) T();
        GetObject() = (T *)GetData();
      } else {
        new ((T *)GetData()) T();
        GetObject() = (T *)GetData();
      }
    }
    return GetObject();
  }

  HSHM_INLINE_CROSS_FUN
  static hshm::SpinLock &GetSpinLock() {
    static char spinlock_data_[sizeof(hshm::SpinLock)] = {0};
    return *(hshm::SpinLock *)spinlock_data_;
  }

  HSHM_INLINE_CROSS_FUN
  static T *GetData() {
    static char data_[sizeof(T)] = {0};
    return (T *)data_;
  }

  HSHM_INLINE_CROSS_FUN
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
#if HSHM_IS_HOST
template <typename T>
using GlobalCrossSingleton = GlobalSingleton<T>;
#else
template <typename T>
using GlobalCrossSingleton = LockfreeCrossSingleton<T>;
#endif

/**
 * C-style singleton with global variables
 */
#define HSHM_DEFINE_GLOBAL_VAR_H(T, NAME) extern __TU(T) NAME;
#define HSHM_DEFINE_GLOBAL_VAR_CC(T, NAME) __TU(T) NAME = T{};
#define HSHM_GET_GLOBAL_VAR(T, NAME) hshm::GetGlobalVar<__TU(T)>(NAME)
template <typename T>
static inline T *GetGlobalVar(T &instance) {
  return &instance;
}

/**
 * Cross-device C-style singleton with global variables
 */
#if HSHM_IS_HOST
#define HSHM_DEFINE_GLOBAL_CROSS_VAR_H(T, NAME) extern __TU(T) NAME;
#define HSHM_DEFINE_GLOBAL_CROSS_VAR_CC(T, NAME) __TU(T) NAME = T{};
#define HSHM_GET_GLOBAL_CROSS_VAR(T, NAME) \
  hshm::GetGlobalCrossVar<__TU(T)>(NAME)
template <typename T>
HSHM_CROSS_FUN static inline T *GetGlobalCrossVar(T &instance) {
  return &instance;
}
#else
#define HSHM_DEFINE_GLOBAL_CROSS_VAR_H(T, NAME)
#define HSHM_DEFINE_GLOBAL_CROSS_VAR_CC(T, NAME)
#define HSHM_GET_GLOBAL_CROSS_VAR(T, NAME) \
  hshm::CrossSingleton<__TU(T)>::GetInstance()
#endif

/**
 * C-style pointer singleton with global variables
 */
#define HSHM_DEFINE_GLOBAL_PTR_VAR_H(T, NAME) extern __TU(T) * NAME;
#define HSHM_DEFINE_GLOBAL_PTR_VAR_H_DLL(DLL_ATTR, T, NAME) \
  extern DLL_ATTR __TU(T) * NAME;
#define HSHM_DEFINE_GLOBAL_PTR_VAR_CC(T, NAME) __TU(T) *NAME = nullptr;
#define HSHM_GET_GLOBAL_PTR_VAR(T, NAME) hshm::GetGlobalPtrVar<__TU(T)>(NAME)
template <typename T>
static inline T *GetGlobalPtrVar(T *&instance) {
  if (instance == nullptr) {
    instance = new T();
  }
  return instance;
}

/**
 * Cross-device C-style pointer singleton with global variables
 */
#if HSHM_IS_HOST
#define HSHM_DEFINE_GLOBAL_CROSS_PTR_VAR_H(T, NAME) extern __TU(T) * NAME;
#define HSHM_DEFINE_GLOBAL_CROSS_PTR_VAR_CC(T, NAME) __TU(T) *NAME = nullptr;
#define HSHM_GET_GLOBAL_CROSS_PTR_VAR(T, NAME) \
  hshm::GetGlobalCrossPtrVar<__TU(T)>(NAME)
template <typename T>
HSHM_CROSS_FUN static inline T *GetGlobalCrossPtrVar(T *&instance) {
  if (instance == nullptr) {
    instance = new T();
  }
  return instance;
}
#else
#define HSHM_DEFINE_GLOBAL_CROSS_PTR_VAR_H(T, NAME)
#define HSHM_DEFINE_GLOBAL_CROSS_PTR_VAR_CC(T, NAME)
#define HSHM_GET_GLOBAL_CROSS_PTR_VAR(T, NAME) \
  hshm::CrossSingleton<__TU(T)>::GetInstance()
#endif

}  // namespace hshm

#endif  // HSHM_SHM_SINGLETON_H
