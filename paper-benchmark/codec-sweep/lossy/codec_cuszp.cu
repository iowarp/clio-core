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
 * @file codec_cuszp.cu
 * @brief cuSZp for lossy_sweep, built twice: against cuSZp V3.0.0 (VGC,
 * SC'25; -DCUSZP_V3, 1-D/2-D/3-D processing) and against V2.0.1 (cuSZp2,
 * SC'24; 1-D only). Both expose cuSZp_compress/decompress with clashing
 * symbols, hence two binaries.
 *
 * cuSZp skips writing blocks whose values all quantize to zero, so the
 * decompression target must start zeroed; that memset is part of the timed
 * decompression here (it is required for a correct result).
 *
 * Variants are the encoding modes the paper offers: plain (delta +
 * fixed-length) and outlier (adds outlier preservation, highest ratio on
 * smooth data). V3 processes the field in its native dimensionality.
 */
#include <cuSZp.h>

#include "lossy_codec.h"

namespace {

/** @brief One cuSZp version; the mode is the variant. */
class CuszpCodec : public LossyCodec {
 public:
  std::string Name() const override {
#ifdef CUSZP_V3
    return "cuszp3";
#else
    return "cuszp2";
#endif
  }
  std::vector<std::string> Variants() const override {
    return {"plain", "outlier"};
  }
  bool SetVariant(const std::string &v) override {
    if (v == "plain") mode_ = CUSZP_MODE_PLAIN;
    else if (v == "outlier") mode_ = CUSZP_MODE_OUTLIER;
    else return false;
    return true;
  }
  /** Worst case is a little over the input; cuSZp documents no bound. */
  size_t MaxCompressedBytes(const Field &f) override {
    return f.n() * sizeof(float) * 11 / 10 + (1 << 20);
  }
  size_t Compress(const float *d_in, const Field &f, double eb, uint8_t *d_out,
                  size_t, cudaStream_t s) override {
    size_t bytes = 0;
#ifdef CUSZP_V3
    cuSZp_compress(const_cast<float *>(d_in), d_out, f.n(), &bytes,
                   static_cast<float>(eb), Dim(f), Dims(f), CUSZP_TYPE_FLOAT,
                   mode_, s);
#else
    cuSZp_compress(const_cast<float *>(d_in), d_out, f.n(), &bytes,
                   static_cast<float>(eb), CUSZP_TYPE_FLOAT, mode_, s);
#endif
    return bytes;
  }
  bool Decompress(const uint8_t *d_cmp, size_t bytes, const Field &f,
                  double eb, float *d_out, cudaStream_t s) override {
    if (cudaMemsetAsync(d_out, 0, f.n() * sizeof(float), s) != cudaSuccess)
      return false;
#ifdef CUSZP_V3
    cuSZp_decompress(d_out, const_cast<uint8_t *>(d_cmp), f.n(), bytes,
                     static_cast<float>(eb), Dim(f), Dims(f), CUSZP_TYPE_FLOAT,
                     mode_, s);
#else
    cuSZp_decompress(d_out, const_cast<uint8_t *>(d_cmp), f.n(), bytes,
                     static_cast<float>(eb), CUSZP_TYPE_FLOAT, mode_, s);
#endif
    return cudaGetLastError() == cudaSuccess;
  }

 private:
#ifdef CUSZP_V3
  /** @return the field's own dimensionality */
  static cuszp_dim_t Dim(const Field &f) {
    return f.dims() == 3 ? CUSZP_DIM_3D
                         : (f.dims() == 2 ? CUSZP_DIM_2D : CUSZP_DIM_1D);
  }
  /** @return cuSZp's dims: x = fastest, matching the CLI's -d order reversed */
  static uint3 Dims(const Field &f) {
    return make_uint3(static_cast<unsigned>(f.x), static_cast<unsigned>(f.y),
                      static_cast<unsigned>(f.z));
  }
#endif
  cuszp_mode_t mode_ = CUSZP_MODE_PLAIN;
};

}  // namespace

std::unique_ptr<LossyCodec> MakeLossyCodec() {
  return std::make_unique<CuszpCodec>();
}
