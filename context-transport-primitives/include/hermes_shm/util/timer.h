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

#ifndef HSHM_TIMER_H
#define HSHM_TIMER_H

#include <time.h>

#include <chrono>
#include <functional>
#include <vector>

#include "hermes_shm/constants/macros.h"
// #include "hermes_shm/data_structures/internal/shm_archive.h"
#include "singleton.h"

namespace hshm {

template <typename T>
class TimepointBase {
 public:
  std::chrono::time_point<T> start_;

 public:
  HSHM_INLINE_CROSS_FUN void Now() {
#if HSHM_IS_HOST
    start_ = T::now();
#endif
  }
  HSHM_INLINE_CROSS_FUN double GetNsecFromStart(TimepointBase &now) const {
#if HSHM_IS_HOST
    double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         now.start_ - start_)
                         .count();
    return elapsed;
#else
    return 0;
#endif
  }
  HSHM_INLINE_CROSS_FUN double GetUsecFromStart(TimepointBase &now) const {
    return GetNsecFromStart(now) / 1000;
  }
  HSHM_INLINE_CROSS_FUN double GetMsecFromStart(TimepointBase &now) const {
    return GetNsecFromStart(now) / 1000000;
  }
  HSHM_INLINE_CROSS_FUN double GetSecFromStart(TimepointBase &now) const {
    return GetNsecFromStart(now) / 1000000000;
  }
  HSHM_INLINE_CROSS_FUN double GetNsecFromStart() const {
#if HSHM_IS_HOST
    std::chrono::time_point<T> end = T::now();
    double elapsed =
        (double)std::chrono::duration_cast<std::chrono::nanoseconds>(end -
                                                                     start_)
            .count();
    return elapsed;
#else
    return 0;
#endif
  }
  HSHM_INLINE_CROSS_FUN double GetUsecFromStart() const {
    return GetNsecFromStart() / 1000;
  }
  HSHM_INLINE_CROSS_FUN double GetMsecFromStart() const {
    return GetNsecFromStart() / 1000000;
  }
  HSHM_INLINE_CROSS_FUN double GetSecFromStart() const {
    return GetNsecFromStart() / 1000000000;
  }
};

class NsecTimer {
 public:
  double time_ns_;

 public:
  NsecTimer() : time_ns_(0) {}

  HSHM_INLINE_CROSS_FUN double GetNsec() const { return time_ns_; }
  HSHM_INLINE_CROSS_FUN double GetUsec() const { return time_ns_ / 1000; }
  HSHM_INLINE_CROSS_FUN double GetMsec() const { return time_ns_ / 1000000; }
  HSHM_INLINE_CROSS_FUN double GetSec() const { return time_ns_ / 1000000000; }
};

template <typename T>
class TimerBase : public TimepointBase<T>, public NsecTimer {
 public:
  /** Constructor */
  HSHM_INLINE_CROSS_FUN
  TimerBase() {}

  /** Resume timer */
  HSHM_INLINE_CROSS_FUN void Resume() { TimepointBase<T>::Now(); }

  /** Pause timer */
  HSHM_INLINE_CROSS_FUN double Pause() {
    time_ns_ += TimepointBase<T>::GetNsecFromStart();
    return time_ns_;
  }

  /** Reset timer */
  HSHM_INLINE_CROSS_FUN void Reset() {
    Resume();
    time_ns_ = 0;
  }

  /** Get microseconds since timer started */
  HSHM_INLINE_CROSS_FUN double GetUsFromEpoch() const {
#if HSHM_IS_HOST
    std::chrono::time_point<std::chrono::system_clock> point =
        std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(
               point.time_since_epoch())
        .count();
#else
    return 0;
#endif
  }
};

typedef TimerBase<std::chrono::high_resolution_clock> HighResCpuTimer;
typedef TimerBase<std::chrono::steady_clock> HighResMonotonicTimer;
typedef HighResMonotonicTimer Timer;
typedef TimepointBase<std::chrono::high_resolution_clock> HighResCpuTimepoint;
typedef TimepointBase<std::chrono::steady_clock> HighResMonotonicTimepoint;
typedef HighResMonotonicTimepoint Timepoint;

/** Timer that measures actual CPU time for the calling thread */
class CpuTimer {
 public:
  double time_ns_ = 0;
  struct timespec start_{0, 0};

  HSHM_INLINE_CROSS_FUN void Resume() {
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start_);
  }
  HSHM_INLINE_CROSS_FUN double Pause() {
    struct timespec end;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &end);
    time_ns_ += (end.tv_sec - start_.tv_sec) * 1e9
              + (end.tv_nsec - start_.tv_nsec);
    return time_ns_;
  }
  HSHM_INLINE_CROSS_FUN void Reset() { time_ns_ = 0; Resume(); }
  HSHM_INLINE_CROSS_FUN double GetNsec() const { return time_ns_; }
  HSHM_INLINE_CROSS_FUN double GetUsec() const { return time_ns_ / 1000; }
  HSHM_INLINE_CROSS_FUN double GetMsec() const { return time_ns_ / 1e6; }
  HSHM_INLINE_CROSS_FUN double GetSec() const { return time_ns_ / 1e9; }
};

template <int IDX>
class PeriodicRun {
 public:
  HighResMonotonicTimer timer_;

  PeriodicRun() { timer_.Resume(); }

  template <typename LAMBDA>
  void Run(size_t max_nsec, LAMBDA &&lambda) {
    size_t nsec = timer_.GetNsecFromStart();
    if (nsec >= max_nsec) {
      lambda();
      timer_.Reset();
    }
  }
};

#define HSHM_PERIODIC(IDX) \
  hshm::CrossSingleton<hshm::PeriodicRun<IDX>>::GetInstance()

}  // namespace hshm

#endif  // HSHM_TIMER_H
