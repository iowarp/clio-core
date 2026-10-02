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
/**
 * One occupancy report, shared by every CUDA edition of the demo workload.
 *
 * The three editions -- synchronous source, clio-coroc state machine, and
 * C++20 device coroutine -- exist to be compared, and a comparison is only
 * worth the paper it is on if all three are measured by the same code. So the
 * measurement lives here and each edition passes in its kernel.
 *
 * Read `regs` together with `local`. A register count that fell because the
 * allocator spilled to local memory is not an improvement, and the delta
 * pipeline's finding that `__launch_bounds__` cuts the coroutine kernels from
 * 192 to 64 registers is only meaningful because it came with LOCAL:0.
 */
#ifndef CLIO_RUNTIME_TEST_CO_CORO_CUDA_REPORT_H_
#define CLIO_RUNTIME_TEST_CO_CORO_CUDA_REPORT_H_

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

namespace clio::co::report {

/** Abort with the failing call's source text if a CUDA call did not succeed.
 *  @param e     the status to check
 *  @param what  source text of the call, for the message */
inline void Check(cudaError_t e, const char *what) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "cuda: %s failed: %s\n", what, cudaGetErrorString(e));
    std::exit(2);
  }
}

#define CUDA_CHECK(expr) ::clio::co::report::Check((expr), #expr)

/**
 * Static resource cost of one kernel, straight from the driver.
 *
 * @param tag    edition name, prefixed to every line so runs can be diffed
 * @param kernel the __global__ function to interrogate
 */
template <class KernelT>
void ReportStatic(const char *tag, KernelT kernel) {
  cudaFuncAttributes a{};
  CUDA_CHECK(cudaFuncGetAttributes(&a, kernel));
  std::fprintf(stderr,
               "%s: regs/thread=%d  local=%zu B  shared=%zu B  const=%zu B  "
               "maxthreads=%d\n",
               tag, a.numRegs, a.localSizeBytes, a.sharedSizeBytes,
               a.constSizeBytes, a.maxThreadsPerBlock);
}

/**
 * Warps resident per SM against block size, from the driver's own occupancy
 * calculator -- the same query Nsight reports as "theoretical occupancy".
 *
 * A sweep rather than one number, because occupancy is a property of (kernel,
 * block size) and the correctness geometry these tests run uses eight threads
 * per block, which no real launch would.
 *
 * @param tag    edition name
 * @param kernel the __global__ function to interrogate
 */
template <class KernelT>
void ReportOccupancy(const char *tag, KernelT kernel) {
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  cudaFuncAttributes a{};
  CUDA_CHECK(cudaFuncGetAttributes(&a, kernel));
  const int max_warps = p.maxThreadsPerMultiProcessor / p.warpSize;

  std::fprintf(stderr, "%s: device=%s sm_%d%d  SMs=%d  maxwarps/SM=%d\n", tag,
               p.name, p.major, p.minor, p.multiProcessorCount, max_warps);

  static const int kBlockSizes[] = {32, 64, 128, 256, 512, 1024};
  for (int bs : kBlockSizes) {
    if (bs > a.maxThreadsPerBlock) continue;
    int blocks = 0;
    CUDA_CHECK(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, bs, 0));
    const int warps = blocks * (bs / p.warpSize);
    std::fprintf(
        stderr,
        "%s: block=%4d  blocks/SM=%2d  warps/SM=%2d  occupancy=%5.1f%%\n", tag,
        bs, blocks, warps, 100.0 * warps / max_warps);
  }
}

}  // namespace clio::co::report

#endif  // CLIO_RUNTIME_TEST_CO_CORO_CUDA_REPORT_H_
