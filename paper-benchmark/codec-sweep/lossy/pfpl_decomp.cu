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
 * @file pfpl_decomp.cu
 * @brief The decompression half of pfpl_shim.cuh over the UNMODIFIED upstream
 * f32_abs_decomp_gpu.cu (burtscher/PFPL, BSD-3, IPDPS'25). Only its main() is
 * renamed out of the way; its kernels are launched as that main() does.
 */
#include "pfpl_shim.cuh"

#define main pfpl_decomp_upstream_main  // upstream CLI entry; unused here
#include "f32_abs_decomp_gpu.cu"
#undef main

namespace {

/** @brief Launch geometry and grow-only scratch, built once. */
struct State {
  bool ready = false;
  int blocks = 0;      // upstream: SMs * (maxThreadsPerSM / TPB)
  int *d_size = nullptr, *h_size = nullptr;
  int *d_fullcarry = nullptr;
  size_t carry_len = 0;
};

/** @return the state, initialized on first use (ready=false on error) */
State &GetState() {
  static State st;
  if (st.ready) return st;
  int dev = 0, sms = 0, tpsm = 0;
  if (cudaGetDevice(&dev) != cudaSuccess ||
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
      cudaDeviceGetAttribute(&tpsm, cudaDevAttrMaxThreadsPerMultiProcessor, dev) != cudaSuccess ||
      cudaMalloc(&st.d_size, sizeof(int)) != cudaSuccess ||
      cudaMallocHost(&st.h_size, sizeof(int)) != cudaSuccess) {
    return st;
  }
  st.blocks = sms * (tpsm / TPB);
  st.ready = true;
  return st;
}

/** @return the kernel-written size after the stream drains, -1 on error */
long long Finish(State &st, cudaStream_t s) {
  if (cudaMemcpyAsync(st.h_size, st.d_size, sizeof(int), cudaMemcpyDeviceToHost, s) != cudaSuccess ||
      cudaStreamSynchronize(s) != cudaSuccess || cudaGetLastError() != cudaSuccess) {
    return -1;
  }
  return *st.h_size;
}

}  // namespace

extern "C" int pfpl_abs_decompress(const void *d_cmp, void *d_out,
                                   size_t n_bytes, cudaStream_t s) {
  State &st = GetState();
  if (!st.ready) return -1;
  d_reset<<<1, 1, 0, s>>>();
  d_decode<<<st.blocks, TPB, 0, s>>>(static_cast<const byte *>(d_cmp),
                                     static_cast<byte *>(d_out), st.d_size);
  const long long got = Finish(st, s);
  if (got < 0) return -1;
  return static_cast<size_t>(got) == n_bytes ? 0 : -3;
}
