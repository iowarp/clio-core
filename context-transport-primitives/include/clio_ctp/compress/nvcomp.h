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

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_NvComp_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_NvComp_H_

#if CTP_ENABLE_COMPRESS && CTP_ENABLE_NVCOMP

#include <cuda_runtime.h>
#include <nvcomp/ans.hpp>
#include <nvcomp/deflate.hpp>
#include <nvcomp/gdeflate.hpp>
#include <nvcomp/lz4.hpp>
#include <nvcomp/nvcompManagerFactory.hpp>
#include <nvcomp/snappy.hpp>
#include <nvcomp/zstd.hpp>

#include <cstdint>
#include <memory>

#include "compress.h"

namespace ctp {

/**
 * Supported nvcomp GPU compression algorithms. All are general-purpose,
 * byte-oriented lossless codecs that fit the byte-stream Compressor interface;
 * each is run with nvcomp's default options (no per-algorithm tuning exposed).
 * (nvcomp's GZIP manager is decompression-only, so it is intentionally absent;
 * use DEFLATE for GPU-side deflate-stream compression.)
 */
enum class NvCompAlgo {
  LZ4,
  SNAPPY,
  ZSTD,
  GDEFLATE,
  DEFLATE,
  ANS,
};

/**
 * GPU compressor backed by NVIDIA nvcomp's high-level Manager API.
 *
 * Implements the synchronous ctp::Compressor interface and runs the nvcomp
 * manager on a private CUDA stream, synchronizing before returning. Buffers are
 * handled adaptively: if a pointer already refers to GPU-accessible memory
 * (device or UVM/managed), it is used in place (zero-copy); otherwise the data
 * is staged through a temporary device buffer (H2D for inputs, D2H for outputs).
 * This keeps it correct for host callers while delivering the zero-copy path
 * automatically whenever the data is already resident on the GPU.
 *
 * The compressed bitstream uses NVCOMP_NATIVE (self-describing) format, so
 * Decompress reconstructs the manager directly from the compressed bytes.
 */
class NvComp : public Compressor {
 public:
  explicit NvComp(NvCompAlgo algo = NvCompAlgo::LZ4) : algo_(algo) {}

  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
      return false;
    }
    uint8_t *d_in = nullptr;
    uint8_t *d_out = nullptr;
    bool free_in = false;
    bool free_out = false;
    bool ok = false;
    do {
      // The nvcomp Manager API reports errors by throwing (and compress() is
      // void, so a throw is its only failure signal); make_shared can also
      // throw std::bad_alloc. Catch everything so no exception escapes (honoring
      // the bool contract) and the cleanup block below always runs.
      try {
        // Input: use directly if already on the GPU, else stage a H2D copy.
        d_in = ToDeviceInput(input, input_size, stream, &free_in);
        if (!d_in) break;

        std::shared_ptr<nvcomp::nvcompManagerBase> mgr = MakeManager(stream);
        if (!mgr) break;
        nvcomp::CompressionConfig cfg = mgr->configure_compression(input_size);

        // Output: write straight into the caller's buffer only if it is a GPU
        // buffer large enough for nvcomp's worst case; otherwise use a temp.
        // (Device-direct requires >= max_compressed_buffer_size, while the temp
        // path only needs to fit the actual compressed size -- this asymmetry
        // is intentional; an in-between device buffer falls back to temp + D2D.)
        const bool out_is_device = IsDeviceAccessible(output);
        if (out_is_device && output_size >= cfg.max_compressed_buffer_size) {
          d_out = static_cast<uint8_t *>(output);
        } else {
          if (cudaMalloc(&d_out, cfg.max_compressed_buffer_size) !=
              cudaSuccess) {
            break;
          }
          free_out = true;
        }

        mgr->compress(d_in, d_out, cfg);
        if (cudaStreamSynchronize(stream) != cudaSuccess) break;
        size_t comp_size = mgr->get_compressed_output_size(d_out);
        if (comp_size > output_size) break;  // caller buffer too small

        // Deliver to the caller's buffer if we compressed into a temp.
        if (d_out != static_cast<uint8_t *>(output)) {
          cudaMemcpyKind kind =
              out_is_device ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
          if (cudaMemcpy(output, d_out, comp_size, kind) != cudaSuccess) break;
        }
        output_size = comp_size;
        ok = true;
      } catch (...) {
        // ok stays false; free_in/free_out (set before any throwing call) make
        // the cleanup below release exactly what was allocated.
      }
    } while (false);
    if (free_in) cudaFree(d_in);
    if (free_out) cudaFree(d_out);
    cudaStreamDestroy(stream);
    return ok;
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
      return false;
    }
    uint8_t *d_in = nullptr;
    uint8_t *d_out = nullptr;
    bool free_in = false;
    bool free_out = false;
    bool ok = false;
    do {
      // The nvcomp Manager API reports errors by throwing (create_manager,
      // configure_decompression, and the void decompress() included); catch
      // everything so no exception escapes and the cleanup below always runs.
      try {
        // Input: use directly if already on the GPU, else stage a H2D copy.
        d_in = ToDeviceInput(input, input_size, stream, &free_in);
        if (!d_in) break;

        // create_manager parses the NVCOMP_NATIVE header and syncs the stream.
        std::shared_ptr<nvcomp::nvcompManagerBase> mgr =
            nvcomp::create_manager(d_in, stream);
        if (!mgr) break;
        nvcomp::DecompressionConfig cfg = mgr->configure_decompression(d_in);
        if (cfg.decomp_data_size > output_size) break;  // caller buffer too small

        const bool out_is_device = IsDeviceAccessible(output);
        if (out_is_device) {
          d_out = static_cast<uint8_t *>(output);
        } else {
          if (cudaMalloc(&d_out, cfg.decomp_data_size) != cudaSuccess) break;
          free_out = true;
        }

        mgr->decompress(d_out, d_in, cfg);
        if (cudaStreamSynchronize(stream) != cudaSuccess) break;

        if (d_out != static_cast<uint8_t *>(output)) {
          if (cudaMemcpy(output, d_out, cfg.decomp_data_size,
                         cudaMemcpyDeviceToHost) != cudaSuccess) {
            break;
          }
        }
        output_size = cfg.decomp_data_size;
        ok = true;
      } catch (...) {
        // ok stays false; free_in/free_out (set before any throwing call) make
        // the cleanup below release exactly what was allocated.
      }
    } while (false);
    if (free_in) cudaFree(d_in);
    if (free_out) cudaFree(d_out);
    cudaStreamDestroy(stream);
    return ok;
  }

 private:
  /** Default per-chunk uncompressed size for nvcomp managers (64 KiB). */
  static constexpr size_t kChunkSize = 1 << 16;

  /**
   * True if `ptr` can be dereferenced by the GPU directly (device or UVM/
   * managed memory). Pinned-host and plain-host pointers return false so the
   * caller stages a copy. A failed query is treated as "not device".
   */
  static bool IsDeviceAccessible(const void *ptr) {
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess) {
      cudaGetLastError();  // reset the sticky error from the failed query
      return false;
    }
    return attr.type == cudaMemoryTypeDevice ||
           attr.type == cudaMemoryTypeManaged;
  }

  /**
   * Return a device pointer holding `size` bytes of `input`. If `input` is
   * already GPU-accessible it is used in place (*owned = false); otherwise a
   * device buffer is allocated and the data copied H2D on `stream`
   * (*owned = true). Returns nullptr on failure (*owned = false).
   */
  static uint8_t *ToDeviceInput(void *input, size_t size, cudaStream_t stream,
                                bool *owned) {
    *owned = false;
    if (IsDeviceAccessible(input)) {
      return static_cast<uint8_t *>(input);
    }
    uint8_t *d = nullptr;
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

  /** Construct the nvcomp manager for the configured algorithm. */
  std::shared_ptr<nvcomp::nvcompManagerBase> MakeManager(cudaStream_t stream) {
    switch (algo_) {
      case NvCompAlgo::LZ4:
        return std::make_shared<nvcomp::LZ4Manager>(
            kChunkSize, nvcompBatchedLZ4DefaultOpts, stream);
      case NvCompAlgo::SNAPPY:
        return std::make_shared<nvcomp::SnappyManager>(
            kChunkSize, nvcompBatchedSnappyDefaultOpts, stream);
      case NvCompAlgo::ZSTD:
        return std::make_shared<nvcomp::ZstdManager>(
            kChunkSize, nvcompBatchedZstdDefaultOpts, stream);
      case NvCompAlgo::GDEFLATE:
        return std::make_shared<nvcomp::GdeflateManager>(
            kChunkSize, nvcompBatchedGdeflateDefaultOpts, stream);
      case NvCompAlgo::DEFLATE:
        return std::make_shared<nvcomp::DeflateManager>(
            kChunkSize, nvcompBatchedDeflateDefaultOpts, stream);
      case NvCompAlgo::ANS:
        return std::make_shared<nvcomp::ANSManager>(
            kChunkSize, nvcompBatchedANSDefaultOpts, stream);
    }
    return nullptr;
  }

  NvCompAlgo algo_;
};

}  // namespace ctp

#endif  // CTP_ENABLE_COMPRESS && CTP_ENABLE_NVCOMP

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_NvComp_H_
