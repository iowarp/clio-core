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

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_GPULZ_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_GPULZ_H_

#if CTP_ENABLE_COMPRESS && CTP_ENABLE_GPULZ

#include <cuda_runtime.h>
#include <gpulz_api.h>

#include <cstddef>
#include <cstdint>

#include "compress.h"

namespace ctp {

namespace gpulz_detail {

/**
 * The calling thread's stream and staging buffers for GPULZ, created on first
 * use and only grown -- the same fairness rule as cuszp_detail::Context, so no
 * codec pays a per-chunk stream or allocation another codec does not. Never
 * released: a thread_local destructor may run after the CUDA context is gone.
 */
struct Context {
  cudaStream_t stream = nullptr; /**< this thread's GPULZ stream */
  void *d_in = nullptr;          /**< device copy of a host input */
  size_t in_cap = 0;             /**< bytes at d_in */
  void *d_out = nullptr;         /**< device output when the caller's won't do */
  size_t out_cap = 0;            /**< bytes at d_out */
};

/**
 * Grow one of the context's buffers to at least `need` bytes.
 *
 * @param p   the buffer pointer.
 * @param cap its capacity, updated.
 * @param need bytes required.
 * @return true when the buffer holds `need` bytes.
 */
inline bool Grow(void **p, size_t *cap, size_t need) {
  if (*cap >= need) return true;
  if (*p != nullptr) cudaFree(*p);
  *p = nullptr;
  *cap = 0;
  if (cudaMalloc(p, need) != cudaSuccess) {
    cudaGetLastError();  // clear the sticky error
    *p = nullptr;
    return false;
  }
  *cap = need;
  return true;
}

/** @return the calling thread's context, or nullptr if its stream failed. */
inline Context *AcquireContext() {
  thread_local Context ctx;
  if (ctx.stream == nullptr && cudaStreamCreate(&ctx.stream) != cudaSuccess) {
    ctx.stream = nullptr;
    return nullptr;
  }
  return &ctx;
}

/** @return true for device or managed memory (see ctp::NvComp). */
inline bool IsDeviceAccessible(const void *ptr) {
  cudaPointerAttributes attr;
  if (cudaPointerGetAttributes(&attr, ptr) != cudaSuccess) {
    cudaGetLastError();  // reset the sticky error from the failed query
    return false;
  }
  return attr.type == cudaMemoryTypeDevice ||
         attr.type == cudaMemoryTypeManaged;
}

/** @return true when p is 4-byte aligned, as the stream's offsets need. */
inline bool Aligned4(const void *p) {
  return reinterpret_cast<uintptr_t>(p) % 4 == 0;
}

/**
 * The input on the device: `input` itself when it already is, else a copy in
 * the context's staging buffer.
 *
 * @return device pointer, or nullptr on failure.
 */
inline const void *DeviceInput(Context *c, const void *input, size_t size) {
  if (IsDeviceAccessible(input)) return input;
  if (!Grow(&c->d_in, &c->in_cap, size) ||
      cudaMemcpyAsync(c->d_in, input, size, cudaMemcpyHostToDevice,
                      c->stream) != cudaSuccess) {
    return nullptr;
  }
  return c->d_in;
}

}  // namespace gpulz_detail

/**
 * GPULZ (ICS'23) GPU LZSS LOSSLESS compressor for multi-byte data.
 *
 * https://github.com/hpdps-group/ICS23-GPULZ, through gpulz_api.h: a library
 * interface paper-benchmark/external-baselines/install_codecs.sh builds over
 * the unmodified upstream kernels (2048-byte blocks of 4-byte symbols, a
 * 32-symbol match window). Any byte length round-trips bit-exactly; the stream
 * is self-describing apart from the decompressed length, which the byte
 * interface carries out of band for every codec.
 *
 * Like NvComp/Ndzip this wrapper is adaptive: a GPU-accessible buffer is used
 * in place, anything else is staged through the thread's device buffers, so
 * host callers (and the unit tests) work too.
 */
class Gpulz : public Compressor {
 public:
  Gpulz() = default;

  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    using namespace gpulz_detail;
    if (output == nullptr || input == nullptr || input_size == 0) return false;
    Context *c = AcquireContext();
    if (c == nullptr) return false;
    const void *d_in = DeviceInput(c, input, input_size);
    if (d_in == nullptr) return false;

    // Straight into the caller's buffer only when it is device memory,
    // aligned, and large enough for GPULZ's worst case.
    const size_t bound = gpulz_compress_bound(input_size);
    const bool out_is_device = IsDeviceAccessible(output);
    void *d_out = output;
    if (!out_is_device || !Aligned4(output) || output_size < bound) {
      if (!Grow(&c->d_out, &c->out_cap, bound)) return false;
      d_out = c->d_out;
    }
    const size_t cap = d_out == output ? output_size : c->out_cap;

    size_t comp_bytes = 0;
    int rc;
    {
      CodecKernelTimer _kt(c->stream);  // same CUDA-event bracket as nvcomp
      rc = gpulz_compress(d_in, input_size, d_out, cap, &comp_bytes, c->stream);
    }
    if (rc != 0 || comp_bytes == 0 || comp_bytes > output_size) return false;
    if (d_out != output) {
      const cudaMemcpyKind kind =
          out_is_device ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
      if (cudaMemcpyAsync(output, d_out, comp_bytes, kind, c->stream) !=
              cudaSuccess ||
          cudaStreamSynchronize(c->stream) != cudaSuccess) {
        return false;
      }
    }
    output_size = comp_bytes;
    return true;
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    using namespace gpulz_detail;
    if (output == nullptr || input == nullptr || output_size == 0) return false;
    Context *c = AcquireContext();
    if (c == nullptr) return false;
    const void *d_in = DeviceInput(c, input, input_size);
    if (d_in == nullptr) return false;

    const size_t n = output_size;
    const bool out_is_device = IsDeviceAccessible(output);
    void *d_out = output;
    if (!out_is_device) {
      if (!Grow(&c->d_out, &c->out_cap, n)) return false;
      d_out = c->d_out;
    }
    if (gpulz_decompress(d_in, input_size, d_out, n, c->stream) != 0) {
      return false;
    }
    if (d_out != output &&
        (cudaMemcpyAsync(output, d_out, n, cudaMemcpyDeviceToHost, c->stream) !=
             cudaSuccess ||
         cudaStreamSynchronize(c->stream) != cudaSuccess)) {
      return false;
    }
    output_size = n;
    return true;
  }
};

}  // namespace ctp

#endif  // CTP_ENABLE_COMPRESS && CTP_ENABLE_GPULZ

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_GPULZ_H_
