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

// gpu_lz4 decode launch, CUDA / ROCm. The kernel body is the shared
// ctp::gpu_lz4::DecodeJobLane (clio_ctp/compress/gpu_lz4.h); the SYCL twin is
// gpu_lz4_sycl.cc.

#include <cstddef>
#include <cstdint>

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include "clio_ctp/compress/gpu_lz4.h"

namespace {

/** One block per page; each warp decodes whole chunks cooperatively. */
CTP_GPU_KERNEL void GpuLz4DecodeKernel(ctp::gpu_lz4::DecodeJob *jobs,
                                       uint32_t njobs) {
  if (blockIdx.x >= njobs) return;
  constexpr uint32_t kWarp = 32;
  ctp::gpu_lz4::DecodeJobWarp(&jobs[blockIdx.x], threadIdx.x % kWarp, kWarp,
                              threadIdx.x / kWarp, blockDim.x / kWarp,
                              []() { __syncwarp(); });
}

}  // namespace

#if defined(_WIN32) && !defined(__CUDA_ARCH__)
#pragma comment(linker, "/EXPORT:ctp_gpu_lz4_decode_launch")
#endif
extern "C" int ctp_gpu_lz4_decode_launch(ctp::gpu_lz4::DecodeJob *jobs,
                                         uint32_t njobs, void *stream) {
  if (njobs == 0) return 0;
#if CTP_ENABLE_CUDA
  GpuLz4DecodeKernel<<<njobs, ctp::gpu_lz4::kDecodeGroup, 0,
                       static_cast<cudaStream_t>(stream)>>>(jobs, njobs);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
#else
  GpuLz4DecodeKernel<<<njobs, ctp::gpu_lz4::kDecodeGroup, 0,
                       static_cast<hipStream_t>(stream)>>>(jobs, njobs);
  return hipGetLastError() == hipSuccess ? 0 : 1;
#endif
}

#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
