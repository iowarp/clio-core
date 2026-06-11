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

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_NDZIP_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_NDZIP_H_

#if CTP_ENABLE_COMPRESS && CTP_ENABLE_NDZIP

#include <cuda_runtime.h>

// These MUST precede the ndzip headers: ndzip/ndzip.hh uses assert() and
// unqualified size()/data() (std::) but relies on its includer for them.
#include <cassert>
#include <cstdint>
#include <iterator>

#include <ndzip/cuda.hh>
#include <ndzip/ndzip.hh>

#include "compress.h"

namespace ctp {

/**
 * ndzip GPU high-throughput, LOSSLESS compressor for floating-point data -- the
 * GPU lossless analog of the CPU codecs, complementing NvComp's general-purpose
 * byte codecs with a predictor designed specifically for scientific float data.
 *
 * https://github.com/celerity/ndzip
 *
 * ndzip's cuda_compressor reads and writes GPU device memory. Like NvComp, this
 * wrapper is adaptive: a GPU-accessible pointer is used in place (zero-copy);
 * otherwise data is staged through a temporary device buffer (H2D for inputs,
 * D2H for outputs), so host callers (and the unit test harness) work too.
 *
 * Constrained by the byte-stream Compressor interface (no type/shape), the input
 * is interpreted as a 1D array of 32-bit floats; Compress requires input_size to
 * be a multiple of sizeof(float). ndzip is LOSSLESS, so Decompress reconstructs
 * the input bit-exactly. The original element count is taken from the caller-
 * supplied output_size (decompressed bytes), since the ndzip stream does not
 * self-describe its extent -- mirroring how the byte interface already carries
 * the decompressed length out-of-band for the other codecs.
 *
 * Tested against the ndzip master CUDA C++ API (make_cuda_compressor /
 * make_cuda_decompressor, extent, compressor_requirements). The NVIDIA
 * devcontainer builds a matching ndzip revision (with NDZIP_WITH_CUDA=ON).
 */
class Ndzip : public Compressor {
 public:
  Ndzip() = default;

  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    if (input == nullptr || (input_size % sizeof(value_t)) != 0) {
      return false;  // float codec; need whole 32-bit floats
    }
    const size_t n = input_size / sizeof(value_t);

    ndzip::extent ext(1);  // 1D
    ext[0] = n;
    const size_t bound_bytes =
        ndzip::compressed_length_bound<value_t>(ext) * sizeof(compressed_t);

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
      return false;
    }
    value_t *d_in = nullptr;
    compressed_t *d_out = nullptr;
    ndzip::index_type *d_len = nullptr;
    bool free_in = false;
    bool free_out = false;
    bool ok = false;
    do {
      d_in = static_cast<value_t *>(
          ToDeviceInput(input, input_size, stream, &free_in));
      if (d_in == nullptr) break;

      // Output: write straight into the caller's buffer only if it is a GPU
      // buffer large enough for ndzip's worst case; else use a device temp.
      const bool out_is_device = IsDeviceAccessible(output);
      if (out_is_device && output_size >= bound_bytes) {
        d_out = static_cast<compressed_t *>(output);
      } else {
        if (cudaMalloc(&d_out, bound_bytes) != cudaSuccess) break;
        free_out = true;
      }

      if (cudaMalloc(&d_len, sizeof(ndzip::index_type)) != cudaSuccess) break;

      auto compressor = ndzip::make_cuda_compressor<value_t>(
          ndzip::compressor_requirements{ext}, stream);
      compressor->compress(d_in, ext, d_out, d_len);

      ndzip::index_type len_words = 0;
      if (cudaMemcpyAsync(&len_words, d_len, sizeof(ndzip::index_type),
                          cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        break;
      }
      if (cudaStreamSynchronize(stream) != cudaSuccess) break;
      const size_t comp_bytes =
          static_cast<size_t>(len_words) * sizeof(compressed_t);
      if (comp_bytes == 0 || comp_bytes > output_size) break;

      // Deliver to the caller's buffer if we compressed into a temp.
      if (static_cast<void *>(d_out) != output) {
        const cudaMemcpyKind kind =
            out_is_device ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
        if (cudaMemcpy(output, d_out, comp_bytes, kind) != cudaSuccess) break;
      }
      output_size = comp_bytes;
      ok = true;
    } while (false);

    if (d_len != nullptr) cudaFree(d_len);
    if (free_out) cudaFree(d_out);
    if (free_in) cudaFree(d_in);
    cudaStreamDestroy(stream);
    return ok;
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    if (output == nullptr || input == nullptr ||
        (output_size % sizeof(value_t)) != 0) {
      return false;
    }
    const size_t n = output_size / sizeof(value_t);

    ndzip::extent ext(1);  // 1D
    ext[0] = n;

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
      return false;
    }
    compressed_t *d_in = nullptr;
    value_t *d_out = nullptr;
    bool free_in = false;
    bool free_out = false;
    bool ok = false;
    do {
      d_in = static_cast<compressed_t *>(
          ToDeviceInput(input, input_size, stream, &free_in));
      if (d_in == nullptr) break;

      const bool out_is_device = IsDeviceAccessible(output);
      if (out_is_device) {
        d_out = static_cast<value_t *>(output);
      } else {
        if (cudaMalloc(&d_out, n * sizeof(value_t)) != cudaSuccess) break;
        free_out = true;
      }

      auto decompressor =
          ndzip::make_cuda_decompressor<value_t>(/*dims=*/1, stream);
      decompressor->decompress(d_in, d_out, ext);
      if (cudaStreamSynchronize(stream) != cudaSuccess) break;

      if (static_cast<void *>(d_out) != output) {
        if (cudaMemcpy(output, d_out, n * sizeof(value_t),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
          break;
        }
      }
      output_size = n * sizeof(value_t);
      ok = true;
    } while (false);

    if (free_out) cudaFree(d_out);
    if (free_in) cudaFree(d_in);
    cudaStreamDestroy(stream);
    return ok;
  }

 private:
  using value_t = float;
  using compressed_t = ndzip::compressed_type<value_t>;  // 4 bytes for float

  /** See ctp::NvComp::IsDeviceAccessible. */
  static bool IsDeviceAccessible(const void *ptr) {
    cudaPointerAttributes attr;
    if (cudaPointerGetAttributes(&attr, ptr) != cudaSuccess) {
      cudaGetLastError();  // reset the sticky error from the failed query
      return false;
    }
    return attr.type == cudaMemoryTypeDevice ||
           attr.type == cudaMemoryTypeManaged;
  }

  /** See ctp::NvComp::ToDeviceInput. */
  static void *ToDeviceInput(void *input, size_t size, cudaStream_t stream,
                             bool *owned) {
    *owned = false;
    if (IsDeviceAccessible(input)) {
      return input;
    }
    void *d = nullptr;
    if (cudaMalloc(&d, size) != cudaSuccess) {
      return nullptr;
    }
    if (cudaMemcpyAsync(d, input, size, cudaMemcpyHostToDevice, stream) !=
        cudaSuccess) {
      cudaFree(d);
      return nullptr;
    }
    *owned = true;
    return d;
  }
};

}  // namespace ctp

#endif  // CTP_ENABLE_COMPRESS && CTP_ENABLE_NDZIP

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_NDZIP_H_
