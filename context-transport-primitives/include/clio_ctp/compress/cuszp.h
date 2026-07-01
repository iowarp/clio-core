/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_CUSZP_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_CUSZP_H_

#if CTP_ENABLE_COMPRESS && CTP_ENABLE_CUSZP

#include <cuda_runtime.h>
#include <cuSZp.h>

#include <cstdint>

#include "compress.h"

namespace ctp {

/**
 * GPU error-bounded LOSSY float compressor backed by cuSZp.
 *
 * Unlike the byte-oriented lossless codecs (lz4/zstd/nvcomp-*), cuSZp is an
 * error-bounded lossy compressor for 32-bit float fields: it trades a bounded
 * amount of precision (the absolute error bound `eb`) for large ratios on smooth
 * / scientific data, and — critically for the compressed GPU vector — it runs
 * ENTIRELY ON THE GPU (the host API launches its kernels on a stream), so data
 * never has to leave the device to be compressed.
 *
 * Implements the synchronous ctp::Compressor interface. cuSZp requires device
 * pointers, so buffers are handled adaptively like the nvcomp wrapper: a
 * GPU-resident pointer is used in place (zero-copy); a host pointer is staged
 * through a temporary device buffer (H2D in, D2H out). The compressed stream is
 * NOT self-describing about `eb` or element count, so Decompress reuses the same
 * `eb` (from the preset) and derives nbEle from the caller's output size — the
 * runtime already carries the original size in its CompressionHeader.
 *
 * The input byte length must be a whole number of floats.
 */
class Cuszp : public Compressor {
 public:
  explicit Cuszp(float error_bound = 1e-3f) : eb_(error_bound) {}

  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    if (input_size == 0 || (input_size % sizeof(float)) != 0) return false;
    const size_t nbEle = input_size / sizeof(float);
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) return false;
    uint8_t *d_in = nullptr, *d_cmp = nullptr;
    bool free_in = false, ok = false;
    do {
      d_in = ToDeviceInput(input, input_size, stream, &free_in);
      if (!d_in) break;
      // cuSZp worst case is ~the input size; add slack and always use a temp so
      // we can bounds-check the caller's buffer against the real cmp size.
      const size_t cap = input_size + input_size / 8 + 1024;
      if (cudaMalloc(&d_cmp, cap) != cudaSuccess) break;
      size_t cmp_size = 0;
      uint3 dims = {0, 0, 0};
      cuSZp_compress(d_in, d_cmp, nbEle, &cmp_size, eb_, CUSZP_DIM_1D, dims,
                     CUSZP_TYPE_FLOAT, CUSZP_MODE_OUTLIER, stream);
      if (cudaStreamSynchronize(stream) != cudaSuccess) break;
      if (cmp_size == 0 || cmp_size > output_size) break;  // caller buffer small
      cudaMemcpyKind kind = IsDeviceAccessible(output) ? cudaMemcpyDeviceToDevice
                                                       : cudaMemcpyDeviceToHost;
      if (cudaMemcpy(output, d_cmp, cmp_size, kind) != cudaSuccess) break;
      output_size = cmp_size;
      ok = true;
    } while (false);
    if (free_in) cudaFree(d_in);
    if (d_cmp) cudaFree(d_cmp);
    cudaStreamDestroy(stream);
    return ok;
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    if (output_size == 0 || (output_size % sizeof(float)) != 0) return false;
    const size_t nbEle = output_size / sizeof(float);
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) return false;
    uint8_t *d_cmp = nullptr, *d_dec = nullptr;
    bool free_in = false, ok = false;
    do {
      d_cmp = ToDeviceInput(input, input_size, stream, &free_in);
      if (!d_cmp) break;
      if (cudaMalloc(&d_dec, output_size) != cudaSuccess) break;
      uint3 dims = {0, 0, 0};
      cuSZp_decompress(d_dec, d_cmp, nbEle, input_size, eb_, CUSZP_DIM_1D, dims,
                       CUSZP_TYPE_FLOAT, CUSZP_MODE_OUTLIER, stream);
      if (cudaStreamSynchronize(stream) != cudaSuccess) break;
      cudaMemcpyKind kind = IsDeviceAccessible(output) ? cudaMemcpyDeviceToDevice
                                                       : cudaMemcpyDeviceToHost;
      if (cudaMemcpy(output, d_dec, output_size, kind) != cudaSuccess) break;
      ok = true;
    } while (false);
    if (free_in) cudaFree(d_cmp);
    if (d_dec) cudaFree(d_dec);
    cudaStreamDestroy(stream);
    return ok;
  }

 private:
  static bool IsDeviceAccessible(const void *ptr) {
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess) {
      cudaGetLastError();  // clear sticky error from the failed query
      return false;
    }
    return attr.type == cudaMemoryTypeDevice ||
           attr.type == cudaMemoryTypeManaged;
  }

  static uint8_t *ToDeviceInput(void *input, size_t size, cudaStream_t stream,
                                bool *owned) {
    *owned = false;
    if (IsDeviceAccessible(input)) return static_cast<uint8_t *>(input);
    uint8_t *d = nullptr;
    if (cudaMalloc(&d, size) != cudaSuccess) return nullptr;
    if (cudaMemcpyAsync(d, input, size, cudaMemcpyHostToDevice, stream) !=
        cudaSuccess) {
      cudaFree(d);
      return nullptr;
    }
    *owned = true;
    return d;
  }

  float eb_;  // absolute error bound
};

}  // namespace ctp

#endif  // CTP_ENABLE_COMPRESS && CTP_ENABLE_CUSZP

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_CUSZP_H_
