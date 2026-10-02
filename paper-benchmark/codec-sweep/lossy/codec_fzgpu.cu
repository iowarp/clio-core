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
 * @file codec_fzgpu.cu
 * @brief FZ-GPU (HPDC'23, szcompressor/FZ-GPU) for lossy_sweep, over the
 * UNMODIFIED upstream src/fz.cu: only its main() is renamed; its dual-quant
 * Lorenzo pre-quantization and fused bitshuffle kernels are launched exactly
 * as its runFzgpu() launches them, with the buffers allocated beforehand
 * (runFzgpu allocates them outside its timed region too).
 *
 * The compressed data is FZ-GPU's own three device arrays (bit flags, block
 * start positions, packed words), kept inside this wrapper rather than packed
 * into one buffer; its size is upstream's own formula. The per-element sign
 * array upstream also writes is not part of it: the decoder takes the sign
 * from bit 15 of each code (lorenzo_var.cuh), as upstream's size accounts.
 */
#include <cuda_runtime.h>
#include <dirent.h>
#include <stdint.h>
#include <sys/stat.h>
#include <thrust/copy.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cub/cub.cuh>
#include <fstream>
#include <iostream>

#define main fzgpu_upstream_main  // upstream CLI entry; unused here
#include "fz.cu"
#undef main

#include "lossy_codec.h"

namespace {

/** @brief FZ-GPU's kernels on wrapper-owned device buffers. */
class FzgpuCodec : public LossyCodec {
 public:
  ~FzgpuCodec() override { Release(); }
  std::string Name() const override { return "fzgpu"; }
  /** Nothing is written to the harness buffer; see the file comment. */
  size_t MaxCompressedBytes(const Field &) override { return 4096; }
  bool PrepareCompress(const float *, const Field &f, double,
                       cudaStream_t) override {
    if (f.n() != n_ && !Allocate(f)) return false;
    // Per-run state upstream zeroes before its timed region.
    return cudaMemset(d_bitflag_, 0, 4 * chunk_) == cudaSuccess &&
           cudaMemset(d_start_, 0, 4 * blocks4k_) == cudaSuccess &&
           cudaMemset(d_csize_, 0, 4 * blocks4k_) == cudaSuccess &&
           cudaMemset(d_offset_, 0, 4) == cudaSuccess;
  }
  size_t Compress(const float *d_in, const Field &f, double eb, uint8_t *,
                  size_t, cudaStream_t s) override {
    float t = 0;
    cusz::experimental::launch_construct_LorenzoI_var<float, uint16_t, float>(
        const_cast<float *>(d_in), d_quant_, d_sign_, Dims(f), eb, t, s);
    compressionFusedKernel<<<Grid(), dim3(32, 32), 0, s>>>(
        (uint32_t *)d_quant_, (uint32_t *)d_packed_, d_offset_, d_bitflag_,
        d_start_, d_csize_);
    uint32_t offset_sum = 0;
    if (cudaMemcpyAsync(&offset_sum, d_offset_, 4, cudaMemcpyDeviceToHost, s) !=
            cudaSuccess || cudaStreamSynchronize(s) != cudaSuccess) {
      return 0;
    }
    // upstream: 4*dataChunkSize + 4*offsetSum + 4*(quantCodeBytes/4096)
    return 4 * chunk_ + 4 * size_t(offset_sum) + 4 * blocks4k_;
  }
  bool PrepareDecompress(const uint8_t *, size_t, const Field &, double,
                         cudaStream_t) override {
    return cudaMemset(d_dquant_, 0, 2 * padded_) == cudaSuccess;
  }
  bool Decompress(const uint8_t *, size_t, const Field &f, double eb,
                  float *d_out, cudaStream_t s) override {
    float t = 0;
    decompressionFusedKernel<<<Grid(), dim3(32, 32), 0, s>>>(
        (uint32_t *)d_packed_, (uint32_t *)d_dquant_, d_bitflag_, d_start_);
    cusz::experimental::launch_reconstruct_LorenzoI_var<float, uint16_t, float>(
        d_sign_, d_dquant_, d_out, Dims(f), eb, t, s);
    return cudaGetLastError() == cudaSuccess;
  }

 private:
  static dim3 Dims(const Field &f) {
    return dim3(unsigned(f.x), unsigned(f.y), unsigned(f.z));
  }
  dim3 Grid() const { return dim3(unsigned(padded_ / 2048)); }

  /** Size every buffer for f as runFzgpu does. @return success */
  bool Allocate(const Field &f) {
    Release();
    const int block_size = 16;
    size_t q = f.n() * 2;  // bytes of uint16 codes, padded to 4096
    q = q % 4096 == 0 ? q : q - q % 4096 + 4096;
    padded_ = q / 2;
    chunk_ = q % (block_size * UINT32_BIT_LEN) == 0
                 ? q / (block_size * UINT32_BIT_LEN)
                 : q / (block_size * UINT32_BIT_LEN) + 1;
    blocks4k_ = q / 4096;
    bool ok = cudaMalloc(&d_quant_, 2 * padded_) == cudaSuccess &&
              cudaMalloc(&d_sign_, padded_) == cudaSuccess &&
              cudaMalloc(&d_packed_, 2 * padded_) == cudaSuccess &&
              cudaMalloc(&d_bitflag_, 4 * chunk_) == cudaSuccess &&
              cudaMalloc(&d_offset_, 4) == cudaSuccess &&
              cudaMalloc(&d_start_, 4 * blocks4k_) == cudaSuccess &&
              cudaMalloc(&d_csize_, 4 * blocks4k_) == cudaSuccess &&
              cudaMalloc(&d_dquant_, 2 * padded_) == cudaSuccess;
    // The padding past n codes is never written; it must read as zero.
    ok = ok && cudaMemset(d_quant_, 0, 2 * padded_) == cudaSuccess;
    if (ok) n_ = f.n();
    return ok;
  }
  void Release() {
    for (void *p : {(void *)d_quant_, (void *)d_sign_, (void *)d_packed_,
                    (void *)d_bitflag_, (void *)d_offset_, (void *)d_start_,
                    (void *)d_csize_, (void *)d_dquant_}) {
      if (p) cudaFree(p);
    }
    d_quant_ = d_packed_ = d_dquant_ = nullptr;
    d_sign_ = nullptr;
    d_bitflag_ = d_offset_ = d_start_ = d_csize_ = nullptr;
    n_ = 0;
  }

  size_t n_ = 0, padded_ = 0, chunk_ = 0, blocks4k_ = 0;
  uint16_t *d_quant_ = nullptr, *d_packed_ = nullptr, *d_dquant_ = nullptr;
  bool *d_sign_ = nullptr;
  uint32_t *d_bitflag_ = nullptr, *d_offset_ = nullptr, *d_start_ = nullptr,
           *d_csize_ = nullptr;
};

}  // namespace

std::unique_ptr<LossyCodec> MakeLossyCodec() {
  return std::make_unique<FzgpuCodec>();
}
