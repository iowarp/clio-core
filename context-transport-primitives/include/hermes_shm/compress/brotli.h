/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Distributed under BSD 3-Clause license.                                   *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Illinois Institute of Technology.                        *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of Hermes. The full Hermes copyright notice, including  *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the top directory. If you do not  *
 * have access to the file, you may request a copy from help@hdfgroup.org.   *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

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

    const size_t bufferSize = BrotliEncoderMaxCompressedSize(input_size);
    if (bufferSize > output_size) {
      HLOG(kError,
            "Output buffer is probably too small for Brotli compression.");
    }
    int ret = BrotliEncoderCompress(
        BROTLI_PARAM_QUALITY, BROTLI_OPERATION_FINISH, BROTLI_DEFAULT_MODE,
        input_size, reinterpret_cast<uint8_t *>(input), &output_size,
        reinterpret_cast<uint8_t *>(output));
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
    int ret = BrotliDecoderDecompress(
        input_size, reinterpret_cast<const uint8_t *>(input), &output_size,
        reinterpret_cast<uint8_t *>(output));
    BrotliDecoderDestroyInstance(state);
    return ret != 0;
  }
};

}  // namespace hshm

#endif  // HSHM_ENABLE_COMPRESS

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_Brotli_H_
