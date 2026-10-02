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
 * @file codec_pfpl.cu
 * @brief PFPL (IPDPS'25, burtscher/PFPL) for lossy_sweep, in its ABS mode:
 * the harness passes the absolute bound it derived from the value range.
 * (PFPL's own REL mode is POINT-WISE relative, a different bound.)
 */
#include "lossy_codec.h"
#include "pfpl_shim.cuh"

namespace {

/** @brief PFPL through pfpl_shim.cuh. */
class PfplCodec : public LossyCodec {
 public:
  std::string Name() const override { return "pfpl"; }
  size_t MaxCompressedBytes(const Field &f) override {
    return pfpl_abs_bound(f.n() * sizeof(float));
  }
  size_t Compress(const float *d_in, const Field &f, double eb, uint8_t *d_out,
                  size_t cap, cudaStream_t s) override {
    size_t bytes = 0;
    return pfpl_abs_compress(d_in, f.n() * sizeof(float),
                             static_cast<float>(eb), d_out, cap, &bytes, s) == 0
               ? bytes : 0;
  }
  bool Decompress(const uint8_t *d_cmp, size_t, const Field &f, double,
                  float *d_out, cudaStream_t s) override {
    return pfpl_abs_decompress(d_cmp, d_out, f.n() * sizeof(float), s) == 0;
  }
};

}  // namespace

std::unique_ptr<LossyCodec> MakeLossyCodec() {
  return std::make_unique<PfplCodec>();
}
