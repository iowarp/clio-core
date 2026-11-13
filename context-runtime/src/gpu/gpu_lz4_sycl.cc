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

// gpu_lz4 decode launch, SYCL (Intel GPUs on Aurora; any SYCL device). The
// kernel body is the shared ctp::gpu_lz4::DecodeJobLane
// (clio_ctp/compress/gpu_lz4.h); the CUDA/ROCm twin is gpu_lz4_gpu.cc.
//
// The queue is the caller's -- GpuApi's stream type under SYCL -- so the
// decode runs on the same device and context as the runtime's allocations
// (the CTE's hbm tier, the gpu_vector page cache), and reads them directly.

#include <cstddef>
#include <cstdint>

#if CTP_ENABLE_SYCL && !(CTP_ENABLE_CUDA || CTP_ENABLE_ROCM)

#include <sycl/sycl.hpp>

#include "clio_ctp/compress/gpu_lz4.h"

extern "C" int ctp_gpu_lz4_decode_launch(ctp::gpu_lz4::DecodeJob *jobs,
                                         uint32_t njobs, void *stream) {
  if (njobs == 0) return 0;
  if (stream == nullptr) return 1;
  auto *q = static_cast<sycl::queue *>(stream);
  constexpr uint32_t kGroup = ctp::gpu_lz4::kDecodeGroup;
  try {
    q->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(njobs) * kGroup),
                          sycl::range<1>(kGroup)),
        [=](sycl::nd_item<1> it) {
          // One work-group per page; each SUB-GROUP decodes whole chunks
          // cooperatively, at the device's native sub-group width (16 on
          // Intel Max GPUs) -- the decoder takes the width as a parameter.
          const uint32_t job = static_cast<uint32_t>(it.get_group(0));
          auto sg = it.get_sub_group();
          ctp::gpu_lz4::DecodeJobWarp(
              &jobs[job], static_cast<uint32_t>(sg.get_local_linear_id()),
              static_cast<uint32_t>(sg.get_local_linear_range()),
              static_cast<uint32_t>(sg.get_group_linear_id()),
              static_cast<uint32_t>(sg.get_group_linear_range()),
              [&]() { sycl::group_barrier(sg); });
        });
  } catch (const sycl::exception &) {
    return 1;
  }
  return 0;
}

#endif  // CTP_ENABLE_SYCL && !(CTP_ENABLE_CUDA || CTP_ENABLE_ROCM)
