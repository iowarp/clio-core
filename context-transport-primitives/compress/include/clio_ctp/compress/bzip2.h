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

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_BZIP2_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_BZIP2_H_

// Optional codec: see the note in snappy.h.
#if CTP_ENABLE_COMPRESS && CTP_ENABLE_BZIP2

#include <bzlib.h>

#include "compress.h"

namespace ctp {

class Bzip2 : public Compressor {
 public:
  int level_;
  int verbosity_ = 0;
  int work_factor_ = 30;

 public:
  Bzip2() : level_(9) {}
  explicit Bzip2(int level) : level_(level) {}

  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    unsigned int output_size_int = output_size;
    int ret = BZ2_bzBuffToBuffCompress((char *)output, &output_size_int,
                                       (char *)input, input_size, level_,
                                       verbosity_, work_factor_);
    output_size = output_size_int;
    return ret == BZ_OK;
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    unsigned int output_size_int = output_size;
    int small = 0;
    int ret = BZ2_bzBuffToBuffDecompress((char *)output, &output_size_int,
                                         (char *)input, input_size, small,
                                         verbosity_);
    output_size = output_size_int;
    return ret == BZ_OK;
  }
};

}  // namespace ctp

#endif  // CTP_ENABLE_COMPRESS && CTP_ENABLE_BZIP2

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_BZIP2_H_
