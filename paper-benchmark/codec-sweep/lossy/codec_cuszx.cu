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
 * @file codec_cuszx.cu
 * @brief cuSZx (the GPU path of szcompressor/SZx, HPDC'22) for lossy_sweep.
 *
 * cuSZx's entry points take and return HOST arrays: they allocate device
 * memory, copy in, run the kernel, copy back and finish packing on the CPU.
 * That is the codec as shipped, so the timed calls are those entry points
 * as-is; the wrapper only stages the harness's device field to the host
 * beforehand and the reconstruction back afterwards (both untimed). Its times
 * therefore include PCIe and CPU work the device-resident codecs do not have.
 * Block size 128, SZx's usual setting.
 */
#include <szx/cuszx_entry.h>

#include <cstdlib>
#include <vector>

#include "lossy_codec.h"

namespace {

/** @brief cuSZx through its host-pointer API. */
class CuszxCodec : public LossyCodec {
 public:
  ~CuszxCodec() override { std::free(cmp_); }
  std::string Name() const override { return "cuszx"; }
  size_t MaxCompressedBytes(const Field &) override { return 4096; }
  std::string Note() const override { return "host API: includes PCIe+CPU"; }
  bool PrepareCompress(const float *d_in, const Field &f, double,
                       cudaStream_t) override {
    host_.resize(f.n());
    return cudaMemcpy(host_.data(), d_in, f.n() * sizeof(float),
                      cudaMemcpyDeviceToHost) == cudaSuccess;
  }
  size_t Compress(const float *, const Field &f, double eb, uint8_t *, size_t,
                  cudaStream_t) override {
    std::free(cmp_);
    size_t bytes = 0;
    cmp_ = cuSZx_fast_compress_args_unpredictable_blocked_float(
        host_.data(), &bytes, static_cast<float>(eb), f.n(), kBlockSize);
    return cmp_ ? bytes : 0;
  }
  bool Decompress(const uint8_t *, size_t, const Field &f, double, float *,
                  cudaStream_t) override {
    out_ = nullptr;
    cuSZx_fast_decompress_args_unpredictable_blocked_float(&out_, f.n(), cmp_);
    return out_ != nullptr;
  }
  bool AfterDecompress(float *d_out, const Field &f, cudaStream_t) override {
    const bool ok = cudaMemcpy(d_out, out_, f.n() * sizeof(float),
                               cudaMemcpyHostToDevice) == cudaSuccess;
    std::free(out_);
    out_ = nullptr;
    return ok;
  }

 private:
  static constexpr int kBlockSize = 128;
  std::vector<float> host_;
  unsigned char *cmp_ = nullptr;
  float *out_ = nullptr;
};

}  // namespace

std::unique_ptr<LossyCodec> MakeLossyCodec() {
  return std::make_unique<CuszxCodec>();
}
