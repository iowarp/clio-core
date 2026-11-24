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

#include <cuda_runtime.h>
#include <new>
#include <cstdio>
#include "container.h"

class Sum : public Container {
 public:
  __device__ int Run() override {
    return 25 + 35;
  }
};

__global__ void AllocateSumKernel(Container *c) {
  new (c) Sum();
}

// RunKernel is defined here (same TU as Sum) so that the device-side vtable
// for Sum::Run is in the same linked device image as the call site. CUDA's
// separate-compilation device linker does not reliably resolve virtual dispatch
// when the vtable owning TU and the call-site TU are different.
__global__ void RunKernel(Container *c, int *ret) {
  *ret = c->Run();
}

extern "C" Container* Allocate() {
  Container *d_obj = nullptr;
  cudaError_t err = cudaMalloc(&d_obj, sizeof(Sum));
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
    return nullptr;
  }
  AllocateSumKernel<<<1, 1>>>(d_obj);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    cudaFree(d_obj);
    return nullptr;
  }
  return d_obj;
}

extern "C" int RunAndGetResult(Container *d_obj) {
  int *d_ret = nullptr;
  cudaError_t err = cudaMalloc(&d_ret, sizeof(int));
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc d_ret failed: %s\n", cudaGetErrorString(err));
    return -1;
  }
  cudaMemset(d_ret, 0, sizeof(int));
  RunKernel<<<1, 1>>>(d_obj, d_ret);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "RunKernel failed: %s\n", cudaGetErrorString(err));
    cudaFree(d_ret);
    return -1;
  }
  int result = 0;
  cudaMemcpy(&result, d_ret, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_ret);
  return result;
}
