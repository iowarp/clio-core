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
 * @file codec_fsz.cu
 * @brief FSZ (SC'26, szcompressor/FSZ) for lossy_sweep: its device API with a
 * reusable workspace; FSZ has no modes to tune, it adapts per tile.
 */
#include <fsz/fsz.h>

#include "lossy_codec.h"

namespace {

/** @brief FSZ through fsz_compress / fsz_decompress. */
class FszCodec : public LossyCodec {
 public:
  ~FszCodec() override { if (ws_) fsz_workspace_destroy(ws_); }
  std::string Name() const override { return "fsz"; }
  size_t MaxCompressedBytes(const Field &f) override {
    return fsz_max_compressed_bytes(f.n());
  }
  bool PrepareCompress(const float *, const Field &f, double,
                       cudaStream_t) override {
    if (ws_ && fsz_workspace_capacity(ws_) >= f.n()) return true;
    if (ws_) fsz_workspace_destroy(ws_);
    ws_ = nullptr;
    return fsz_workspace_create(&ws_, f.n()) == FSZ_STATUS_OK;
  }
  size_t Compress(const float *d_in, const Field &f, double eb, uint8_t *d_out,
                  size_t, cudaStream_t s) override {
    if (fsz_compress(d_in, d_out, f.n(), static_cast<float>(eb), ws_, s,
                     &res_) != FSZ_STATUS_OK) {
      return 0;
    }
    return res_.cmp_size;
  }
  bool Decompress(const uint8_t *d_cmp, size_t, const Field &f, double eb,
                  float *d_out, cudaStream_t s) override {
    return fsz_decompress(d_out, d_cmp, f.n(), static_cast<float>(eb), &res_,
                          ws_, s) == FSZ_STATUS_OK;
  }

 private:
  fsz_workspace_t *ws_ = nullptr;
  fsz_compress_result_t res_{};
};

}  // namespace

std::unique_ptr<LossyCodec> MakeLossyCodec() {
  return std::make_unique<FszCodec>();
}
