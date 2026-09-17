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

#include "basic_test.h"
#include "clio_ctp/util/logging.h"
#include "clio_ctp/util/timer.h"
#include "clio_ctp/util/timer_mpi.h"
#include "clio_ctp/util/timer_thread.h"

#if CTP_ENABLE_MPI
#include <mpi.h>
#endif

#if CTP_ENABLE_OPENMP
#include <omp.h>
#endif

TEST_CASE("TestPeriodic") {
  HILOG_PERIODIC(0, 0, ctp::Unit<size_t>::Seconds(1), "Print periodic 1");
  sleep(1);
  HILOG_PERIODIC(0, 0, ctp::Unit<size_t>::Seconds(1), "Print periodic 2");
  HILOG_PERIODIC(0, 0, ctp::Unit<size_t>::Seconds(1), "Print periodic 3");
}

TEST_CASE("TestTimepoint") {
  ctp::Timepoint timer;
  timer.Now();
  sleep(2);
  HLOG(kInfo, "Print timer: {}", timer.GetSecFromStart());
}

TEST_CASE("TestTimer") {
  ctp::Timer timer;
  timer.Resume();
  sleep(3);
  timer.Pause();
  HLOG(kInfo, "Print timer: {}", timer.GetSec());
}

#if CTP_ENABLE_MPI
TEST_CASE("TestMpiTimer") {
  ctp::MpiTimer mpi_timer(MPI_COMM_WORLD);
  mpi_timer.Resume();
  sleep(3);
  mpi_timer.Pause();
  HLOG(kInfo, "Print timer (Collect): {}", mpi_timer.Collect().GetSec());
  HLOG(kInfo, "Print timer (Min): {}", mpi_timer.CollectMin().GetSec());
  HLOG(kInfo, "Print timer (Max): {}", mpi_timer.CollectMax().GetSec());
  HLOG(kInfo, "Print timer (Avg): {}", mpi_timer.CollectAvg().GetSec());
}
#endif

#if CTP_ENABLE_OPENMP
TEST_CASE("TestOmpTimer") {
  ctp::ThreadTimer omp_timer(4);
#pragma omp parallel shared(omp_timer) num_threads(4)
  {
    omp_timer.SetRank(omp_get_thread_num());
    omp_timer.Resume();
    sleep(3);
    omp_timer.Pause();
#pragma omp barrier
  }
  omp_timer.Collect();
  HLOG(kInfo, "Print timer: {}", omp_timer.GetSec());
}
#endif
