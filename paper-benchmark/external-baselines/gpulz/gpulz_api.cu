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
 * @file gpulz_api.cu
 * @brief gpulz_api.h over the UNMODIFIED upstream GPULZ kernels.
 *
 * Build with the upstream clone on the include path (install_codecs.sh pins
 * commit 314d6cf); its compressKernelI / compressKernelIII / decompressKernel
 * are used exactly as its main() launches them. Only main() itself is renamed
 * out of the way. The upstream repository carries a copyright notice but no
 * license, so its source is compiled from the clone, never copied here.
 */
#include "gpulz_api.h"

// Every header gpulz.cu pulls in is included FIRST, so the `main` rename below
// cannot reach into them (their include guards stop a second expansion).
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cub/cub.cuh>
#include <fstream>
#include <iostream>

#define main gpulz_upstream_main  // upstream's CLI entry point; unused here
#include "gpulz.cu"
#undef main

namespace {

using Sym = INPUT_TYPE;                          // upstream: uint32_t symbols
constexpr uint32_t kMagic = 0x5A4C5047u;         // "GPLZ"
constexpr size_t kBlock = BLOCK_SIZE;            // bytes per GPULZ block
constexpr size_t kFlagsPerBlock = BLOCK_SIZE / sizeof(Sym) / 8;  // flag bytes

/** The stream's fixed prefix; everything after it is located from it. */
struct Header {
  uint32_t magic;       /**< kMagic */
  uint32_t num_blocks;  /**< ceil(n / kBlock) */
  uint64_t n;           /**< uncompressed bytes */
  uint32_t flag_bytes;  /**< bytes of match flags */
  uint32_t data_bytes;  /**< bytes of literals and (length, offset) pairs */
};
static_assert(sizeof(Header) == 24, "the offsets after the header must stay 4-aligned");

/** One grow-only device buffer. */
struct Buf {
  void *p = nullptr;  /**< device allocation */
  size_t cap = 0;     /**< bytes at p */
};

/**
 * The calling thread's scratch. Only grows and is never released: a
 * thread_local destructor may run after the CUDA context is gone (cuszp.h).
 */
struct Ctx {
  Buf flag_size, data_size;  /**< per-block counts compressKernelI writes */
  Buf tmp_flags, tmp_data;   /**< per-block staging before compaction */
  Buf scan;                  /**< CUB's temporary storage */
  Buf pad;                   /**< zero-padded input / whole-block output */
  Buf in_copy;               /**< aligned copy of a misaligned stream */
};

Ctx &ThreadCtx() {
  thread_local Ctx c;
  return c;
}

/**
 * Buffer b with at least `need` bytes, reallocating only to grow.
 *
 * @param b the buffer.
 * @param need bytes required.
 * @return device pointer, or nullptr when cudaMalloc failed.
 */
void *Grow(Buf &b, size_t need) {
  if (need == 0) need = 1;
  if (b.cap >= need) return b.p;
  if (b.p != nullptr) cudaFree(b.p);
  b.p = nullptr;
  b.cap = 0;
  if (cudaMalloc(&b.p, need) != cudaSuccess) {
    cudaGetLastError();  // clear the sticky error
    b.p = nullptr;
    return nullptr;
  }
  b.cap = need;
  return b.p;
}

/** @return GPULZ blocks covering n bytes. */
size_t NumBlocks(size_t n) { return (n + kBlock - 1) / kBlock; }

/** @return bytes of the header plus both offset arrays for nb blocks. */
size_t FixedBytes(size_t nb) {
  return sizeof(Header) + 2 * (nb + 1) * sizeof(uint32_t);
}

/** @return true when p is aligned for the stream's u32 offsets and symbols. */
bool Aligned(const void *p) {
  return reinterpret_cast<uintptr_t>(p) % alignof(Sym) == 0;
}

/** @return 0 when the stream finished cleanly, -1 on any CUDA error. */
int Finish(cudaStream_t s) {
  if (cudaStreamSynchronize(s) != cudaSuccess) return -1;
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

/**
 * Input as whole, aligned GPULZ blocks: d_in itself when it already is,
 * else a zero-padded copy in the thread's scratch.
 *
 * @return device pointer to nb * kBlock bytes, or nullptr on failure.
 */
const Sym *BlockedInput(Ctx &c, const void *d_in, size_t n, size_t nb,
                        cudaStream_t s) {
  const size_t padded = nb * kBlock;
  if (padded == n && Aligned(d_in)) return static_cast<const Sym *>(d_in);
  void *p = Grow(c.pad, padded);
  if (p == nullptr ||
      cudaMemsetAsync(p, 0, padded, s) != cudaSuccess ||
      cudaMemcpyAsync(p, d_in, n, cudaMemcpyDeviceToDevice, s) != cudaSuccess) {
    return nullptr;
  }
  return static_cast<const Sym *>(p);
}

}  // namespace

extern "C" size_t gpulz_compress_bound(size_t n) {
  const size_t nb = NumBlocks(n);
  return FixedBytes(nb) + nb * (kFlagsPerBlock + kBlock);
}

extern "C" int gpulz_compress(const void *d_in, size_t n, void *d_out,
                              size_t out_cap, size_t *out_bytes,
                              cudaStream_t s) {
  if (d_in == nullptr || d_out == nullptr || out_bytes == nullptr || n == 0 ||
      !Aligned(d_out)) {
    return -3;
  }
  const size_t nb = NumBlocks(n);
  if (nb * kBlock > UINT32_MAX) return -2;  // upstream offsets are 32-bit
  if (out_cap < FixedBytes(nb)) return -2;
  Ctx &c = ThreadCtx();
  const Sym *in = BlockedInput(c, d_in, n, nb, s);
  auto *flag_size = static_cast<uint32_t *>(Grow(c.flag_size, (nb + 1) * sizeof(uint32_t)));
  auto *data_size = static_cast<uint32_t *>(Grow(c.data_size, (nb + 1) * sizeof(uint32_t)));
  auto *tmp_flags = static_cast<uint8_t *>(Grow(c.tmp_flags, nb * kFlagsPerBlock));
  auto *tmp_data = static_cast<uint8_t *>(Grow(c.tmp_data, nb * kBlock));
  if (in == nullptr || flag_size == nullptr || data_size == nullptr ||
      tmp_flags == nullptr || tmp_data == nullptr) {
    return -1;
  }
  auto *base = static_cast<uint8_t *>(d_out);
  auto *flag_off = reinterpret_cast<uint32_t *>(base + sizeof(Header));
  auto *data_off = flag_off + nb + 1;
  auto *flags = reinterpret_cast<uint8_t *>(data_off + nb + 1);
  const auto nb32 = static_cast<uint32_t>(nb);

  // Upstream main()'s pipeline: match, two exclusive scans, compaction.
  const int min_encode = sizeof(Sym) == 1 ? 2 : 1;
  compressKernelI<Sym><<<nb32, THREAD_SIZE, 0, s>>>(
      const_cast<Sym *>(in), nb32, flag_size, data_size, tmp_flags, tmp_data,
      min_encode);
  size_t t1 = 0, t2 = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, t1, flag_size, flag_off, nb + 1, s);
  cub::DeviceScan::ExclusiveSum(nullptr, t2, data_size, data_off, nb + 1, s);
  void *scan = Grow(c.scan, t1 > t2 ? t1 : t2);
  if (scan == nullptr) return -1;
  cub::DeviceScan::ExclusiveSum(scan, t1, flag_size, flag_off, nb + 1, s);
  cub::DeviceScan::ExclusiveSum(scan, t2, data_size, data_off, nb + 1, s);

  // The totals decide where the data section starts and whether it fits.
  uint32_t totals[2] = {0, 0};
  if (cudaMemcpyAsync(&totals[0], flag_off + nb, sizeof(uint32_t),
                      cudaMemcpyDeviceToHost, s) != cudaSuccess ||
      cudaMemcpyAsync(&totals[1], data_off + nb, sizeof(uint32_t),
                      cudaMemcpyDeviceToHost, s) != cudaSuccess ||
      Finish(s) != 0) {
    return -1;
  }
  const size_t total = FixedBytes(nb) + totals[0] + totals[1];
  if (total > out_cap) return -2;
  compressKernelIII<Sym><<<nb32, THREAD_SIZE, 0, s>>>(
      nb32, flag_off, data_off, tmp_flags, tmp_data, flags, flags + totals[0]);
  const Header h{kMagic, nb32, static_cast<uint64_t>(n), totals[0], totals[1]};
  if (cudaMemcpyAsync(base, &h, sizeof(h), cudaMemcpyHostToDevice, s) !=
          cudaSuccess ||
      Finish(s) != 0) {
    return -1;
  }
  *out_bytes = total;
  return 0;
}

extern "C" int gpulz_decompress(const void *d_in, size_t in_bytes, void *d_out,
                                size_t n, cudaStream_t s) {
  if (d_in == nullptr || d_out == nullptr || n == 0 ||
      in_bytes < sizeof(Header)) {
    return -3;
  }
  Ctx &c = ThreadCtx();
  if (!Aligned(d_in)) {  // the offsets are read as u32 in place
    void *p = Grow(c.in_copy, in_bytes);
    if (p == nullptr || cudaMemcpyAsync(p, d_in, in_bytes, cudaMemcpyDeviceToDevice,
                                        s) != cudaSuccess) {
      return -1;
    }
    d_in = p;
  }
  Header h{};
  if (cudaMemcpyAsync(&h, d_in, sizeof(h), cudaMemcpyDeviceToHost, s) !=
          cudaSuccess ||
      Finish(s) != 0) {
    return -1;
  }
  const size_t nb = NumBlocks(n);
  if (h.magic != kMagic || h.n != n || h.num_blocks != nb ||
      FixedBytes(nb) + h.flag_bytes + h.data_bytes > in_bytes) {
    return -3;
  }
  const auto *base = static_cast<const uint8_t *>(d_in);
  auto *flag_off = const_cast<uint32_t *>(
      reinterpret_cast<const uint32_t *>(base + sizeof(Header)));
  auto *data_off = flag_off + nb + 1;
  auto *flags = reinterpret_cast<uint8_t *>(data_off + nb + 1);

  // decompressKernel writes whole blocks: straight into d_out when it holds
  // them, else into scratch and then the first n bytes across.
  const size_t padded = nb * kBlock;
  const bool direct = padded == n && Aligned(d_out);
  Sym *out = direct ? static_cast<Sym *>(d_out)
                    : static_cast<Sym *>(Grow(c.pad, padded));
  if (out == nullptr) return -1;
  const auto nb32 = static_cast<uint32_t>(nb);
  decompressKernel<Sym><<<(nb32 + 31) / 32, 32, 0, s>>>(
      out, nb32, flag_off, data_off, flags, flags + h.flag_bytes);
  if (!direct && cudaMemcpyAsync(d_out, out, n, cudaMemcpyDeviceToDevice, s) !=
                     cudaSuccess) {
    return -1;
  }
  return Finish(s);
}
