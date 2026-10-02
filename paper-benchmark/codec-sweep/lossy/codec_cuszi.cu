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
 * @file codec_cuszi.cu
 * @brief cuSZ-I (SC'24) for lossy_sweep: cuSZ v0.19's spline-interpolation
 * predictor (SplineY24, the cuSZ-i one) with its histogram and Huffman stage,
 * through the psz C API on device memory.
 *
 * Variants pick the Huffman coder: "spline-pbk" uses cuSZ's default refined
 * coder (HFR_PBKC), "spline-hf" the classic one (HF). The context is built
 * and freed outside the timed calls, like every other codec's workspace.
 * SplineY24 is 3-D only (cuSZ's own CLI documentation); a 2-D field is
 * reported as not applicable rather than run on a predictor that does not
 * support it.
 */
#include <cusz.h>

#include "lossy_codec.h"

namespace {

/** @brief cuSZ v0.19 with the spline predictor. */
class CusziCodec : public LossyCodec {
 public:
  ~CusziCodec() override {
    if (cctx_) psz_free(cctx_);
    if (dctx_) psz_free(dctx_);
  }
  std::string Name() const override { return "cuszi"; }
  std::vector<std::string> Variants() const override {
    return {"spline-pbk", "spline-hf"};
  }
  bool SetVariant(const std::string &v) override {
    if (v == "spline-pbk") codec_ = DEFAULT_CODEC;
    else if (v == "spline-hf") codec_ = HF;
    else return false;
    return true;
  }
  /** cuSZ falls back to storing outliers; size the buffer generously. */
  size_t MaxCompressedBytes(const Field &f) override {
    return f.n() * sizeof(float) * 2 + (1 << 20);
  }
  std::string Note() const override { return note_; }
  bool PrepareCompress(const float *, const Field &f, double,
                       cudaStream_t s) override {
    note_.clear();
    if (f.dims() != 3) {
      note_ = "n/a: SplineY24 is 3-D only";
      return false;
    }
    if (cctx_) psz_free(cctx_);
    cctx_ = psz_compress_init(F4, psz_len3{f.x, f.y, f.z}, s);
    return cctx_ != nullptr;
  }
  size_t Compress(const float *d_in, const Field &, double eb, uint8_t *d_out,
                  size_t cap, cudaStream_t s) override {
    const psz_ppl ppl{SplineY24, DEFAULT_HISTOGRAM, codec_, CodecNull};
    if (psz_compress_process_float(cctx_, ppl, eb, const_cast<float *>(d_in)) != 0)
      return 0;
    uint8_t *d_arch = nullptr;
    size_t bytes = 0;
    if (psz_compress_archive(cctx_, &header_, &d_arch, &bytes) != 0 || bytes > cap)
      return 0;
    // The archive lives in the context; persisting it is part of compressing.
    if (cudaMemcpyAsync(d_out, d_arch, bytes, cudaMemcpyDeviceToDevice, s) != cudaSuccess)
      return 0;
    return bytes;
  }
  bool AfterCompress(uint8_t *, size_t, cudaStream_t) override {
    psz_free(cctx_);
    cctx_ = nullptr;
    return true;
  }
  bool PrepareDecompress(const uint8_t *, size_t, const Field &, double,
                         cudaStream_t s) override {
    if (dctx_) psz_free(dctx_);
    dctx_ = psz_decompress_init(&header_, s);
    return dctx_ != nullptr;
  }
  bool Decompress(const uint8_t *d_cmp, size_t bytes, const Field &, double,
                  float *d_out, cudaStream_t) override {
    return psz_decompress_process_float(dctx_, const_cast<uint8_t *>(d_cmp),
                                        bytes, d_out) == 0;
  }
  bool AfterDecompress(float *, const Field &, cudaStream_t) override {
    psz_free(dctx_);
    dctx_ = nullptr;
    return true;
  }

 private:
  psz_ctx *cctx_ = nullptr, *dctx_ = nullptr;
  psz_header header_{};
  psz_codec codec_ = DEFAULT_CODEC;
  std::string note_;
};

}  // namespace

std::unique_ptr<LossyCodec> MakeLossyCodec() {
  return std::make_unique<CusziCodec>();
}
