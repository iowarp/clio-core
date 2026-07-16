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

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_Blosc_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_Blosc_H_

#if CTP_ENABLE_COMPRESS && CTP_ENABLE_BLOSC2

#include <blosc2.h>

#include "compress.h"
#include "clio_ctp/util/singleton.h"

namespace ctp {

class BloscInit {
 public:
  BloscInit() { blosc2_init(); }
  ~BloscInit() { blosc2_destroy(); }
};
#define BLOSC_INIT ctp::Singleton<BloscInit>::GetInstance()

class Blosc : public Compressor {
 public:
  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    // Initialize Blosc2
    BLOSC_INIT;

    // Create a context for compression
    blosc2_context *cctx = blosc2_create_cctx(BLOSC2_CPARAMS_DEFAULTS);
    if (!cctx) {
      return false;
    }

    // Compress the data
    output_size =
        blosc2_compress_ctx(cctx, input, input_size, output, output_size);

    // Release the compression context
    blosc2_free_ctx(cctx);
    return true;
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    // Create a context for decompression
    blosc2_context *dctx = blosc2_create_dctx(BLOSC2_DPARAMS_DEFAULTS);
    if (!dctx) {
      return false;
    }

    // Decompress the data
    output_size =
        blosc2_decompress_ctx(dctx, input, input_size, output, output_size);

    // Release the decompression context
    blosc2_free_ctx(dctx);
    return true;
  }
};

}  // namespace ctp

#endif  // CTP_ENABLE_COMPRESS && CTP_ENABLE_BLOSC2

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_Blosc_H_
