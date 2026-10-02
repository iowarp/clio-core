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
 * @file fpc_state.cuh
 * @brief Per-translation-unit scratch shared by fpc_encode.cuh and
 * fpc_decode.cuh. Included once per wrapper, after the upstream program, so
 * each algorithm/direction gets its own copy (everything here is static).
 */
#ifndef CLIO_EXTERNAL_FPC_STATE_CUH_
#define CLIO_EXTERNAL_FPC_STATE_CUH_

/** FPC_FN(compress) -> <FPC_NAME>_compress; FPC_NAME is set by each wrapper. */
#define FPC_CAT2(a, b) a##_##b
#define FPC_CAT(a, b) FPC_CAT2(a, b)
#define FPC_FN(x) FPC_CAT(FPC_NAME, x)

namespace {

/** Largest input accepted: upstream indexes bytes with int. */
constexpr size_t kFpcMaxInput = size_t{1} << 30;

/** Launch geometry and grow-only scratch for one wrapper. */
struct FpcState {
  bool ready = false;
  int blocks = 0;               /**< upstream: SMs * (maxThreadsPerSM / TPB) */
  int *d_size = nullptr;        /**< kernel-written output size */
  int *h_size = nullptr;        /**< pinned copy of d_size */
  int *d_fullcarry = nullptr;   /**< encoder's per-chunk carry array */
  size_t fullcarry_len = 0;     /**< ints d_fullcarry holds */
};

/** @return the wrapper's state, initialized on first use (ready=false on error). */
FpcState &FpcGetState() {
  static FpcState st;
  if (st.ready) return st;
  int dev = 0, sms = 0, tpsm = 0;
  if (cudaGetDevice(&dev) != cudaSuccess ||
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) !=
          cudaSuccess ||
      cudaDeviceGetAttribute(&tpsm, cudaDevAttrMaxThreadsPerMultiProcessor,
                             dev) != cudaSuccess ||
      cudaMalloc(&st.d_size, sizeof(int)) != cudaSuccess ||
      cudaMallocHost(&st.h_size, sizeof(int)) != cudaSuccess) {
    return st;
  }
  st.blocks = sms * (tpsm / TPB);
  st.ready = true;
  return st;
}

/**
 * Copy the kernel-written size to the host and wait for the stream.
 * @return the size, or -1 on a CUDA error anywhere on the stream.
 */
long long FpcFinish(FpcState &st, cudaStream_t stream) {
  if (cudaMemcpyAsync(st.h_size, st.d_size, sizeof(int),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess ||
      cudaGetLastError() != cudaSuccess) {
    return -1;
  }
  return *st.h_size;
}

}  // namespace

#endif  // CLIO_EXTERNAL_FPC_STATE_CUH_
