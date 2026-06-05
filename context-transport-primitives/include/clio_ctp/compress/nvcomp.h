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
#include <nvcomp/lz4.hpp>
#include <nvcomp/nvcompManagerFactory.hpp>

#include <cstdint>
#include <memory>

#include "compress.h"

namespace ctp {

/** Supported nvcomp GPU compression algorithms. */
enum class NvCompAlgo {
  LZ4,
};

/**
 * GPU compressor backed by NVIDIA nvcomp's high-level Manager API.
 *
 * Implements the synchronous ctp::Compressor interface over host buffers: each
 * call copies the input to the device, runs the nvcomp manager on a private
 * CUDA stream, synchronizes, then copies the result back to the host. The
 * compressed bitstream uses NVCOMP_NATIVE (self-describing) format, so
 * Decompress can reconstruct the manager directly from the compressed bytes.
 *
 * Note: this performs its own H2D/D2H copies because the current Compressor
 * contract hands us host pointers. A future device-pointer-aware path could
 * skip the copies when the data is already resident on the GPU.
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
    bool ok = false;
    do {
      if (cudaMalloc(&d_in, input_size) != cudaSuccess) break;
      if (cudaMemcpyAsync(d_in, input, input_size, cudaMemcpyHostToDevice,
                          stream) != cudaSuccess) {
        break;
      }
      std::shared_ptr<nvcomp::nvcompManagerBase> mgr = MakeManager(stream);
      if (!mgr) break;
      nvcomp::CompressionConfig cfg = mgr->configure_compression(input_size);
      if (cudaMalloc(&d_out, cfg.max_compressed_buffer_size) != cudaSuccess) {
        break;
      }
      mgr->compress(d_in, d_out, cfg);
      if (cudaStreamSynchronize(stream) != cudaSuccess) break;
      size_t comp_size = mgr->get_compressed_output_size(d_out);
      if (comp_size > output_size) break;  // caller buffer too small
      if (cudaMemcpy(output, d_out, comp_size, cudaMemcpyDeviceToHost) !=
          cudaSuccess) {
        break;
      }
      output_size = comp_size;
      ok = true;
    } while (false);
    if (d_in) cudaFree(d_in);
    if (d_out) cudaFree(d_out);
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
    bool ok = false;
    do {
      if (cudaMalloc(&d_in, input_size) != cudaSuccess) break;
      if (cudaMemcpyAsync(d_in, input, input_size, cudaMemcpyHostToDevice,
                          stream) != cudaSuccess) {
        break;
      }
      // create_manager parses the NVCOMP_NATIVE header and synchronizes stream.
      std::shared_ptr<nvcomp::nvcompManagerBase> mgr =
          nvcomp::create_manager(d_in, stream);
      if (!mgr) break;
      nvcomp::DecompressionConfig cfg = mgr->configure_decompression(d_in);
      if (cfg.decomp_data_size > output_size) break;  // caller buffer too small
      if (cudaMalloc(&d_out, cfg.decomp_data_size) != cudaSuccess) break;
      mgr->decompress(d_out, d_in, cfg);
      if (cudaStreamSynchronize(stream) != cudaSuccess) break;
      if (cudaMemcpy(output, d_out, cfg.decomp_data_size,
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        break;
      }
      output_size = cfg.decomp_data_size;
      ok = true;
    } while (false);
    if (d_in) cudaFree(d_in);
    if (d_out) cudaFree(d_out);
    cudaStreamDestroy(stream);
    return ok;
  }

 private:
  /** Default per-chunk uncompressed size for nvcomp managers (64 KiB). */
  static constexpr size_t kChunkSize = 1 << 16;

  /** Construct the nvcomp manager for the configured algorithm. */
  std::shared_ptr<nvcomp::nvcompManagerBase> MakeManager(cudaStream_t stream) {
    switch (algo_) {
      case NvCompAlgo::LZ4:
        return std::make_shared<nvcomp::LZ4Manager>(
            kChunkSize, nvcompBatchedLZ4DefaultOpts, stream);
    }
    return nullptr;
  }

  NvCompAlgo algo_;
};

}  // namespace ctp

#endif  // CTP_ENABLE_COMPRESS && CTP_ENABLE_NVCOMP

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_NvComp_H_
