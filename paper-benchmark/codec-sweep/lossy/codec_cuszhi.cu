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
 * @file codec_cuszhi.cu
 * @brief cuSZ-Hi (SC'25, research snapshot shixun404/cuSZ-Hi) for
 * lossy_sweep. The snapshot carries no license, so it is used only as an
 * installed library from a pinned clone; nothing of it is copied here.
 *
 * Configured as its README prescribes for compression: the spline3
 * predictor, ABS mode (the harness supplies the absolute bound), and its
 * default ratio-first interpolation auto-tuning. Variants are its two
 * lossless schemes: "cr" (Huffman + LC, higher ratio) and "tp" (Huffman-free,
 * faster). The context is built from the same argv its CLI parses, so every
 * other setting keeps the snapshot's default.
 */
#include <cstdio>
#include <string>
#include <vector>

#include "context.h"
#include "cusz.h"
#include "tehm.hh"

#include "lossy_codec.h"

namespace {

/** @brief cuSZ-Hi through psz_compress / psz_decompress. */
class CuszhiCodec : public LossyCodec {
 public:
  ~CuszhiCodec() override {
    Release();
    delete ctx_;
    if (outlier_tmp_) cudaFree(outlier_tmp_);
  }
  std::string Name() const override { return "cuszhi"; }
  std::vector<std::string> Variants() const override { return {"cr", "tp"}; }
  bool SetVariant(const std::string &v) override {
    if (v != "cr" && v != "tp") return false;
    scheme_ = v;
    return true;
  }
  size_t MaxCompressedBytes(const Field &f) override {
    return f.n() * sizeof(float) * 2 + (1 << 20);
  }
  bool PrepareCompress(const float *, const Field &f, double eb,
                       cudaStream_t) override {
    char ebs[64], dims[96];
    std::snprintf(ebs, sizeof(ebs), "%.17g", eb);
    std::snprintf(dims, sizeof(dims), "%zux%zux%zu", f.x, f.y, f.z);
    std::vector<std::string> args = {"cuszhi", "-z", "-t", "f32", "-m", "abs",
                                     "-e", ebs, "--dim3", dims, "--predictor",
                                     "spline3", "-s", scheme_, "-i", "/dev/null"};
    std::vector<char *> argv;
    for (auto &a : args) argv.push_back(a.data());
    delete ctx_;
    ctx_ = new psz_context;
    pszctx_create_from_argv(ctx_, static_cast<int>(argv.size()), argv.data());
    // A compressor is initialized once per run, as the CLI does. Its buffers
    // belong to the C++ compressor behind comp_->compressor, which
    // psz_release() (a plain `delete comp`) never destroys; delete it here or
    // every run leaks a full set (out of memory on QMCPACK's 630 MB field).
    Release();
    comp_ = psz_create(pszdefault_framework(), F4);
    len_ = pszlen{f.x, f.y, f.z, 1};
    return comp_ && psz_compress_init(comp_, len_, ctx_) == CUSZ_SUCCESS;
  }
  size_t Compress(const float *d_in, const Field &, double, uint8_t *d_out,
                  size_t cap, cudaStream_t s) override {
    psz::TimeRecord rec;
    uint8_t *d_c = nullptr;
    size_t bytes = 0;
    if (psz_compress(comp_, const_cast<float *>(d_in), len_, &d_c, &bytes,
                     &header_, &rec, s) != CUSZ_SUCCESS ||
        bytes == 0 || bytes > cap) {
      return 0;
    }
    // The stream lives in the compressor; persisting it is part of the work.
    return cudaMemcpyAsync(d_out, d_c, bytes, cudaMemcpyDeviceToDevice, s) ==
                   cudaSuccess ? bytes : 0;
  }
  bool PrepareDecompress(const uint8_t *, size_t, const Field &f, double,
                         cudaStream_t) override {
    if (f.n() > outlier_cap_) {
      if (outlier_tmp_) cudaFree(outlier_tmp_);
      outlier_cap_ = 0;
      if (cudaMalloc(&outlier_tmp_, f.n() * sizeof(float)) != cudaSuccess) return false;
      outlier_cap_ = f.n();
    }
    return psz_decompress_init(comp_, &header_) == CUSZ_SUCCESS;
  }
  bool Decompress(const uint8_t *d_cmp, size_t bytes, const Field &, double,
                  float *d_out, cudaStream_t s) override {
    psz::TimeRecord rec;
    return psz_decompress(comp_, const_cast<uint8_t *>(d_cmp), bytes, d_out,
                          outlier_tmp_, len_, &rec, s) == CUSZ_SUCCESS;
  }

 private:
  /** Destroy the compressor and its device buffers, then the C handle. */
  void Release() {
    if (!comp_) return;
    delete static_cast<cusz::CompressorF4 *>(comp_->compressor);
    psz_release(comp_);
    comp_ = nullptr;
  }

  std::string scheme_ = "cr";
  psz_context *ctx_ = nullptr;
  pszcompressor *comp_ = nullptr;
  pszheader header_{};
  pszlen len_{};
  float *outlier_tmp_ = nullptr;
  size_t outlier_cap_ = 0;
};

}  // namespace

std::unique_ptr<LossyCodec> MakeLossyCodec() {
  return std::make_unique<CuszhiCodec>();
}
