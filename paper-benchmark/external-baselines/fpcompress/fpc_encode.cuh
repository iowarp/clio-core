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
 * @file fpc_encode.cuh
 * @brief The compress half of fpc_api.h, included right after an upstream
 * *-compressor-single.cu with FPC_NAME set to the algorithm's prefix. Uses
 * that program's static d_reset / d_encode, CS and TPB, launched as its main()
 * does, minus the file I/O and minus its input copy (a D2D copy into a buffer
 * that no preprocessor touches in the single-precision programs).
 */
#include "fpc_state.cuh"

extern "C" size_t FPC_FN(compress_bound)(size_t n) {
  const size_t chunks = (n + CS - 1) / CS;
  return sizeof(int) + chunks * sizeof(short) + chunks * CS;
}

extern "C" int FPC_FN(compress)(const void *d_in, size_t n, void *d_out,
                                size_t out_cap, size_t *out_bytes,
                                cudaStream_t stream) {
  if (!d_in || !d_out || !out_bytes || n == 0 || n > kFpcMaxInput) return -3;
  if (out_cap < FPC_FN(compress_bound)(n)) return -2;
  FpcState &st = FpcGetState();
  if (!st.ready) return -1;
  const size_t chunks = (n + CS - 1) / CS;
  if (chunks > st.fullcarry_len) {
    if (st.d_fullcarry) cudaFree(st.d_fullcarry);
    st.fullcarry_len = 0;
    if (cudaMalloc(&st.d_fullcarry, chunks * sizeof(int)) != cudaSuccess) {
      st.d_fullcarry = nullptr;
      return -1;
    }
    st.fullcarry_len = chunks;
  }
  d_reset<<<1, 1, 0, stream>>>();
  if (cudaMemsetAsync(st.d_fullcarry, 0, chunks * sizeof(int), stream) !=
      cudaSuccess) {
    return -1;
  }
  d_encode<<<st.blocks, TPB, 0, stream>>>(
      static_cast<const byte *>(d_in), static_cast<int>(n),
      static_cast<byte *>(d_out), st.d_size, st.d_fullcarry);
  const long long got = FpcFinish(st, stream);
  if (got <= 0) return -1;
  *out_bytes = static_cast<size_t>(got);
  return 0;
}
