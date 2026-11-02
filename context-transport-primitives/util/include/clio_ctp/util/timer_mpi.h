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

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_UTIL_TIMER_MPI_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_UTIL_TIMER_MPI_H_

#if CTP_ENABLE_MPI

#include <mpi.h>

#include "timer.h"

namespace ctp {

class MpiTimer : public Timer {
 public:
  MPI_Comm comm_;
  int rank_;
  int nprocs_;

 public:
  explicit MpiTimer(MPI_Comm comm) : comm_(comm) {
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &nprocs_);
  }

  MpiTimer& Collect() { return CollectAvg(); }

  MpiTimer& CollectMax() {
    MPI_Barrier(comm_);
    double my_nsec = GetNsec();
    MPI_Allreduce(&my_nsec, &time_ns_, 1, MPI_DOUBLE, MPI_MAX, comm_);
    return *this;
  }

  MpiTimer& CollectMin() {
    MPI_Barrier(comm_);
    double my_nsec = GetNsec();
    MPI_Allreduce(&my_nsec, &time_ns_, 1, MPI_DOUBLE, MPI_MIN, comm_);
    return *this;
  }

  MpiTimer& CollectAvg() {
    MPI_Barrier(comm_);
    double my_nsec = GetNsec();
    double total_time_ns;
    MPI_Allreduce(&my_nsec, &total_time_ns, 1, MPI_DOUBLE, MPI_SUM, comm_);
    time_ns_ = total_time_ns / nprocs_;
    return *this;
  }
};

}  // namespace ctp
#endif  // CTP_ENABLE_MPI
#endif  // CTP_SHM_INCLUDE_HSHM_SHM_UTIL_TIMER_MPI_H_
