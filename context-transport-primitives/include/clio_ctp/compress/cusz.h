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

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_CUSZ_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_CUSZ_H_

#if CTP_ENABLE_COMPRESS && CTP_ENABLE_CUSZ

#include <cuda_runtime.h>
#include <cusz.h>

#include <cstdint>
#include <cstring>

#include "compress.h"

namespace ctp {

/**
 * cuSZ GPU error-bounded LOSSY compressor for floating-point scientific data --
 * the GPU lossy analog of the LibPressio sz3/zfp entries, the way NvComp is the
 * GPU lossless analog of the CPU codecs.
 *
 * https://github.com/szcompressor/cuSZ
 *
 * cuSZ operates directly on GPU device memory. Like NvComp, this wrapper is
 * adaptive: a pointer that already refers to GPU-accessible memory is used in
 * place (zero-copy); otherwise the data is staged through a temporary device
 * buffer (H2D for inputs, D2H for outputs), so host callers (and the unit test
 * harness) work too while device callers get the zero-copy path automatically.
 *
 * Like the existing zfp/sz3/fpzip/zfp-sycl entries, and constrained by the
 * byte-stream Compressor interface (which carries neither type nor shape), the
 * input buffer is interpreted as a 1D array of 32-bit floats. Compress requires
 * input_size to be a multiple of sizeof(float).
 *
 * Self-describing framing: the output blob is laid out as
 *   [ Prefix (magic + elem count + psz_header) ][ cuSZ compressed byte-stream ]
 * so Decompress rebuilds the cuSZ resource manager from the embedded header
 * without any out-of-band metadata (cuSZ's compressed stream alone does not
 * carry its decode metadata).
 *
 * IMPORTANT -- LOSSY. Each value is reconstructed within the configured error
 * bound (relative by default), not bit-exact. Presets map to error bounds:
 * FAST=1e-2 (loose), BALANCED=1e-3, BEST=1e-4 (tight).
 *
 * Tested against the cuSZ master / 0.14 C API (psz_create_resource_manager,
 * psz_compress_float, psz_decompress_float, psz_release_resource). The NVIDIA
 * devcontainer pins a matching cuSZ revision.
 */
class Cusz : public Compressor {
 public:
  /**
   * @param eb error bound (default 1e-3).
   * @param mode error-bound mode: Rel (relative) or Abs (absolute).
   */
  explicit Cusz(double eb = 1e-3, psz_mode mode = Rel)
      : eb_(eb), mode_(mode) {}

  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    if (input == nullptr || (input_size % sizeof(float)) != 0) {
      return false;  // float codec; need whole 32-bit floats
    }
    if (output_size < sizeof(Prefix)) {
      return false;
    }
    const size_t n = input_size / sizeof(float);

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
      return false;
    }
    float *d_in = nullptr;
    bool free_in = false;
    psz_resource *mgr = nullptr;
    bool ok = false;
    do {
      d_in = static_cast<float *>(
          ToDeviceInput(input, input_size, stream, &free_in));
      if (d_in == nullptr) break;

      psz_len len = {n, 1, 1};  // x, y, z (1D)
      psz_pipeline pipeline = {Lorenzo, HistGeneric, HF, CodecNull};
      mgr = psz_create_resource_manager(F4, len, pipeline, stream);
      if (mgr == nullptr) break;

      Prefix prefix;
      std::memset(&prefix, 0, sizeof(prefix));
      prefix.magic = kMagic;
      prefix.elems = static_cast<uint64_t>(n);

      uint8_t *d_comp = nullptr;  // device buffer owned by the manager
      size_t comp_bytes = 0;
      psz_rc2 rc = {mode_, eb_, kRadius};
      if (psz_compress_float(mgr, rc, d_in, &prefix.header, &d_comp,
                             &comp_bytes) != 0 ||
          d_comp == nullptr) {
        break;
      }

      const size_t total = sizeof(Prefix) + comp_bytes;
      if (total > output_size) break;  // caller buffer too small

      // Frame: [prefix][compressed stream]. d_comp lives in the manager's
      // internal device buffer, so it must be copied out before release.
      const bool out_is_device = IsDeviceAccessible(output);
      uint8_t *out = static_cast<uint8_t *>(output);
      const cudaMemcpyKind kind =
          out_is_device ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
      if (out_is_device) {
        if (cudaMemcpyAsync(out, &prefix, sizeof(Prefix),
                            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
          break;
        }
      } else {
        std::memcpy(out, &prefix, sizeof(Prefix));
      }
      if (cudaMemcpyAsync(out + sizeof(Prefix), d_comp, comp_bytes, kind,
                          stream) != cudaSuccess) {
        break;
      }
      if (cudaStreamSynchronize(stream) != cudaSuccess) break;
      output_size = total;
      ok = true;
    } while (false);

    if (mgr != nullptr) psz_release_resource(mgr);
    if (free_in) cudaFree(d_in);
    cudaStreamDestroy(stream);
    return ok;
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    if (output == nullptr || input == nullptr || input_size < sizeof(Prefix)) {
      return false;
    }

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
      return false;
    }
    float *d_out = nullptr;
    bool free_out = false;
    psz_resource *mgr = nullptr;
    bool ok = false;
    do {
      // Pull the prefix (magic + header) back to the host.
      Prefix prefix;
      if (IsDeviceAccessible(input)) {
        if (cudaMemcpy(&prefix, input, sizeof(Prefix),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
          break;
        }
      } else {
        std::memcpy(&prefix, input, sizeof(Prefix));
      }
      if (prefix.magic != kMagic) break;  // not one of our blobs

      const size_t n = static_cast<size_t>(prefix.elems);
      if (n * sizeof(float) > output_size) break;  // caller buffer too small

      // cuSZ writes into device memory; use the caller's buffer if it is a GPU
      // buffer, else decode into a temp and copy D2H.
      const bool out_is_device = IsDeviceAccessible(output);
      if (out_is_device) {
        d_out = static_cast<float *>(output);
      } else {
        if (cudaMalloc(&d_out, n * sizeof(float)) != cudaSuccess) break;
        free_out = true;
      }

      // The compressed stream must be device-resident for cuSZ.
      uint8_t *d_stream = nullptr;
      bool free_stream = false;
      const size_t comp_bytes = pszheader_compressed_bytes(&prefix.header);
      uint8_t *stream_src = static_cast<uint8_t *>(input) + sizeof(Prefix);
      if (IsDeviceAccessible(input)) {
        d_stream = stream_src;
      } else {
        if (cudaMalloc(&d_stream, comp_bytes) != cudaSuccess) break;
        free_stream = true;
        if (cudaMemcpyAsync(d_stream, stream_src, comp_bytes,
                            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
          cudaFree(d_stream);
          break;
        }
      }

      mgr = psz_create_resource_manager_from_header(&prefix.header, stream);
      bool decoded = mgr != nullptr &&
                     psz_decompress_float(mgr, d_stream, comp_bytes, d_out) == 0;
      if (decoded && cudaStreamSynchronize(stream) == cudaSuccess) {
        if (!out_is_device) {
          decoded = cudaMemcpy(output, d_out, n * sizeof(float),
                               cudaMemcpyDeviceToHost) == cudaSuccess;
        }
        if (decoded) {
          output_size = n * sizeof(float);
          ok = true;
        }
      }
      if (free_stream) cudaFree(d_stream);
    } while (false);

    if (mgr != nullptr) psz_release_resource(mgr);
    if (free_out) cudaFree(d_out);
    cudaStreamDestroy(stream);
    return ok;
  }

  /** Set the error bound. */
  void SetErrorBound(double eb) { eb_ = eb; }
  /** Get the error bound. */
  double GetErrorBound() const { return eb_; }

 private:
  /** cuSZ quantization radius (cuSZ's historical default). */
  static constexpr uint16_t kRadius = 512;
  static constexpr uint32_t kMagic = 0x5A535543u;  // "CUSZ"

  // Self-describing prefix carried ahead of the cuSZ codestream. Embeds the
  // psz_header so Decompress can rebuild the resource manager from the bytes.
  struct Prefix {
    uint32_t magic;
    uint32_t pad;       // keep `elems` 8-aligned
    uint64_t elems;     // float element count
    psz_header header;  // cuSZ metadata needed to decode
  };

  /**
   * True if `ptr` is dereferenceable by the GPU directly (device or UVM/managed
   * memory). Plain/pinned host pointers return false so the caller stages a
   * copy. A failed query is treated as "not device".
   */
  static bool IsDeviceAccessible(const void *ptr) {
    cudaPointerAttributes attr;
    if (cudaPointerGetAttributes(&attr, ptr) != cudaSuccess) {
      cudaGetLastError();  // reset the sticky error from the failed query
      return false;
    }
    return attr.type == cudaMemoryTypeDevice ||
           attr.type == cudaMemoryTypeManaged;
  }

  /**
   * Return a device pointer holding `size` bytes of `input`: used in place if
   * already GPU-accessible (*owned=false), else copied H2D on `stream`
   * (*owned=true). Returns nullptr on failure.
   */
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

  double eb_;       // error bound
  psz_mode mode_;   // error-bound mode (Rel / Abs)
};

}  // namespace ctp

#endif  // CTP_ENABLE_COMPRESS && CTP_ENABLE_CUSZ

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_CUSZ_H_
