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

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_Brotli_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_Brotli_H_

#if HSHM_ENABLE_COMPRESS

#include <brotli/decode.h>
#include <brotli/encode.h>

#include "compress.h"

namespace hshm {

class Brotli : public Compressor {
 public:
  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    BrotliEncoderState *state =
        BrotliEncoderCreateInstance(nullptr, nullptr, nullptr);
    if (state == nullptr) {
      return false;
    }

    const ::size_t bufferSize = BrotliEncoderMaxCompressedSize(input_size);
    if (bufferSize > output_size) {
      HLOG(kError,
            "Output buffer is probably too small for Brotli compression.");
    }
    ::size_t out_sz = output_size;
    int ret = BrotliEncoderCompress(
        BROTLI_PARAM_QUALITY, BROTLI_OPERATION_FINISH, BROTLI_DEFAULT_MODE,
        input_size, reinterpret_cast<uint8_t *>(input), &out_sz,
        reinterpret_cast<uint8_t *>(output));
    output_size = out_sz;
    BrotliEncoderDestroyInstance(state);
    return ret != 0;
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    BrotliDecoderState *state =
        BrotliDecoderCreateInstance(nullptr, nullptr, nullptr);
    if (state == nullptr) {
      return false;
    }
    ::size_t out_sz = output_size;
    int ret = BrotliDecoderDecompress(
        input_size, reinterpret_cast<const uint8_t *>(input), &out_sz,
        reinterpret_cast<uint8_t *>(output));
    output_size = out_sz;
    BrotliDecoderDestroyInstance(state);
    return ret != 0;
  }
};

}  // namespace hshm

#endif  // HSHM_ENABLE_COMPRESS

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_Brotli_H_
