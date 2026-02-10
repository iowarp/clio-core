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

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_Snappy_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_Snappy_H_

#if HSHM_ENABLE_COMPRESS

#include <snappy-sinksource.h>
#include <snappy.h>

#include "compress.h"

namespace hshm {

class Snappy : public Compressor {
 public:
  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    ::size_t out_sz = output_size;
    snappy::RawCompress((char *)input, input_size, (char *)output,
                        &out_sz);
    output_size = out_sz;
    bool ret = snappy::IsValidCompressedBuffer((char *)output, out_sz);
    return ret;
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    snappy::ByteArraySource source(reinterpret_cast<const char *>(input),
                                   input_size);
    snappy::UncheckedByteArraySink sink(reinterpret_cast<char *>(output));
    output_size = snappy::UncompressAsMuchAsPossible(&source, &sink);
    return output_size != 0;
  }
};

}  // namespace hshm

#endif  // HSHM_ENABLE_COMPRESS

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_Snappy_H_
