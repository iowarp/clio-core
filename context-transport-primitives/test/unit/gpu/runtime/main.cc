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
#ifndef _WIN32
#include <dlfcn.h>
#endif
#include <cstdio>
#include <cstdlib>
#include "container.h"

__global__ void RunKernel(Container *c, int *ret) {
  *ret = c->Run();
}

int main() {
  // Load the shared library
  void *lib = dlopen(GPU_RUNTIME_LIB_PATH, RTLD_NOW);
  if (!lib) {
    fprintf(stderr, "FAIL: dlopen: %s\n", dlerror());
    return 1;
  }

  // Get the factory function
  using AllocateFn = Container* (*)();
  auto Allocate = reinterpret_cast<AllocateFn>(dlsym(lib, "Allocate"));
  if (!Allocate) {
    fprintf(stderr, "FAIL: dlsym: %s\n", dlerror());
    dlclose(lib);
    return 1;
  }

  // Allocate the object on the device
  Container *d_obj = Allocate();
  if (!d_obj) {
    fprintf(stderr, "FAIL: Allocate returned nullptr\n");
    dlclose(lib);
    return 1;
  }

  // Allocate device memory for the result
  int *d_ret = nullptr;
  cudaMalloc(&d_ret, sizeof(int));

  // Launch kernel that calls virtual method
  RunKernel<<<1, 1>>>(d_obj, d_ret);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "FAIL: RunKernel: %s\n", cudaGetErrorString(err));
    cudaFree(d_ret);
    cudaFree(d_obj);
    dlclose(lib);
    return 1;
  }

  // Copy result back and check
  int result = 0;
  cudaMemcpy(&result, d_ret, sizeof(int), cudaMemcpyDeviceToHost);

  if (result == 60) {
    printf("PASS: result = %d\n", result);
  } else {
    printf("FAIL: expected 60, got %d\n", result);
  }

  // Cleanup
  cudaFree(d_ret);
  cudaFree(d_obj);
  dlclose(lib);

  return (result == 60) ? 0 : 1;
}
