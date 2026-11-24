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

#ifndef CTP_INCLUDE_HSHM_UTIL_AUTO_TRACE_H_
#define CTP_INCLUDE_HSHM_UTIL_AUTO_TRACE_H_

#include <iostream>

#include "formatter.h"
#include "logging.h"
#include "timer.h"

namespace ctp {

#define AUTO_TRACE(LOG_CODE) ctp::AutoTrace<LOG_CODE> _ctp_tracer_(__func__)
#define TIMER_START(NAME) _ctp_tracer_.StartTimer(NAME)
#define TIMER_END() _ctp_tracer_.EndTimer()

/** Trace function execution times */
template <int LOG_CODE>
class AutoTrace {
 private:
  HighResMonotonicTimer timer_;
  HighResMonotonicTimer timer2_;
  std::string fname_;
  std::string internal_name_;

 public:
  CTP_INLINE AutoTrace(const char *fname) {
    if constexpr (LOG_CODE >= 0) {
      fname_ = fname;
      _StartTimer(timer_);
    }
  }

  CTP_INLINE
  ~AutoTrace() {
    if constexpr (LOG_CODE >= 0) {
      _EndTimer(timer_);
    }
  }

  CTP_INLINE
  void StartTimer(const char *internal_name) {
    if constexpr (LOG_CODE >= 0) {
      internal_name_ = "/" + std::string(internal_name);
      _StartTimer(timer2_);
    }
  }

  CTP_INLINE
  void EndTimer() {
    if constexpr (LOG_CODE >= 0) {
      _EndTimer(timer2_);
    }
  }

 private:
  template <typename... Args>
  CTP_INLINE void _StartTimer(HighResMonotonicTimer &timer) {
    timer.Resume();
    HIPRINT("{}{}", fname_, internal_name_);
  }

  CTP_INLINE
  void _EndTimer(HighResMonotonicTimer &timer) {
    timer.Pause();
    HIPRINT("{}{} {}ns", fname_, internal_name_, timer.GetNsec());
    timer.Reset();
    internal_name_.clear();
  }
};

}  // namespace ctp

#endif  // CTP_INCLUDE_HSHM_UTIL_AUTO_TRACE_H_
