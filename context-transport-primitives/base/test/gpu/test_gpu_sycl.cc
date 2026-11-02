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
 * Intel GPU unit test using SYCL/oneAPI
 *
 * Tests Intel GPU compute using SYCL USM (Unified Shared Memory) on
 * Aurora's Intel Data Center GPU Max (Ponte Vecchio / PVC).
 *
 * Uses #ifdef CTP_ENABLE_SYCL so this file compiles only when SYCL is
 * enabled. Existing CUDA/ROCm code is not affected since no CUDA/ROCm
 * headers or keywords are used here.
 */

#ifdef CTP_ENABLE_SYCL

#include <catch2/catch_all.hpp>
#include <sycl/sycl.hpp>

/**
 * Test: GpuSyclVectorFill
 *
 * Allocates SYCL shared arrays, dispatches kernels, and verifies results on
 * the CPU. Tests basic GPU dispatch and USM data access patterns.
 */
// Use a single TEST_CASE with one SECTION to avoid recreating the SYCL queue
// across Catch2 section re-runs, which can cause segfaults on Intel GPU.
TEST_CASE("GpuSyclVectorFill", "[gpu][sycl]") {
  // Static queue: constructed once for the lifetime of the process.
  static sycl::queue q{sycl::gpu_selector_v};

  SECTION("IntelGpuKernels") {
    INFO("Device: " << q.get_device().get_info<sycl::info::device::name>());

    // --- Test 1: parallel_for fill ---
    {
      constexpr size_t kSize = 256;
      int *buf = sycl::malloc_shared<int>(kSize, q);
      REQUIRE(buf != nullptr);
      for (size_t i = 0; i < kSize; ++i) buf[i] = -1;
      q.parallel_for(sycl::range<1>(kSize), [=](sycl::id<1> idx) {
         buf[idx] = static_cast<int>(idx.get(0));
       }).wait();
      for (size_t i = 0; i < kSize; ++i) {
        REQUIRE(buf[i] == static_cast<int>(i));
      }
      sycl::free(buf, q);
    }

    // --- Test 2: parallel_for scale ---
    {
      constexpr size_t kSize = 512;
      int *data = sycl::malloc_shared<int>(kSize, q);
      int *out = sycl::malloc_shared<int>(kSize, q);
      REQUIRE(data != nullptr);
      REQUIRE(out != nullptr);
      for (size_t i = 0; i < kSize; ++i) { data[i] = static_cast<int>(i); out[i] = 0; }
      q.parallel_for(sycl::range<1>(kSize), [=](sycl::id<1> idx) {
         out[idx] = data[idx] * 2;
       }).wait();
      for (size_t i = 0; i < kSize; ++i) {
        REQUIRE(out[i] == static_cast<int>(i * 2));
      }
      sycl::free(data, q);
      sycl::free(out, q);
    }
  }
}

#else  // CTP_ENABLE_SYCL not defined

#include <catch2/catch_all.hpp>

TEST_CASE("GpuSyclVectorFill_Disabled", "[gpu][sycl]") {
  SUCCEED("SYCL not enabled in this build");
}

#endif  // CTP_ENABLE_SYCL
