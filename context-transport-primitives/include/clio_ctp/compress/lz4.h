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

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_Lz4_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_Lz4_H_

#if CTP_ENABLE_COMPRESS

#include <lz4.h>

#include "compress.h"

namespace ctp {

class Lz4 : public Compressor {
 public:
  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    if ((size_t)LZ4_compressBound(input_size) > output_size) {
      HLOG(kInfo, "Lz4: output buffer is potentially too small");
    }
    output_size = LZ4_compress_default((char *)input, (char *)output,
                                       (int)input_size, (int)output_size);
    return output_size != 0;
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    // LZ4_decompress_safe returns a NEGATIVE value on failure. Assigning it
    // straight into a size_t made it enormous, and `!= 0` then reported
    // SUCCESS -- so a failed decompression was handed back as valid data with
    // a garbage length. Check the signed result before converting.
    int rc = LZ4_decompress_safe((char *)input, (char *)output,
                                 (int)input_size, (int)output_size);
    if (rc <= 0) {
      output_size = 0;
      return false;
    }
    output_size = static_cast<size_t>(rc);
    return true;
  }
};

}  // namespace ctp

#endif  // CTP_ENABLE_COMPRESS

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_Lz4_H_
