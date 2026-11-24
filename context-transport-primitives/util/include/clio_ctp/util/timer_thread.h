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

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_UTIL_TIMER_THREAD_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_UTIL_TIMER_THREAD_H_

#include "timer.h"

namespace ctp {

class ThreadTimer : public NsecTimer {
 public:
  int rank_;
  int nprocs_;
  std::vector<Timer> timers_;

 public:
  explicit ThreadTimer(int nthreads) {
    nprocs_ = nthreads;
    timers_.resize(nprocs_);
  }

  void SetRank(int rank) { rank_ = rank; }

  void Resume() { timers_[rank_].Resume(); }

  void Pause() { timers_[rank_].Pause(); }

  void Reset() { timers_[rank_].Reset(); }

  void Collect() {
    std::vector<double> rank_times;
    rank_times.reserve(nprocs_);
    for (Timer &t : timers_) {
      rank_times.push_back(t.GetNsec());
    }
    time_ns_ = *std::max_element(rank_times.begin(), rank_times.end());
  }
};

}  // namespace ctp

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_UTIL_TIMER_THREAD_H_
