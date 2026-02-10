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

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_Lzo_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_Lzo_H_

#if HSHM_ENABLE_COMPRESS

#include <lzo/lzo1x.h>

#include "compress.h"

namespace hshm {

class Lzo : public Compressor {
 private:
  // Work memory buffer required by LZO for compression
  // LZO1X_1_15_MEM_COMPRESS is typically ~64-128KB depending on platform
  alignas(16) unsigned char work_mem_[LZO1X_1_15_MEM_COMPRESS];

 public:
  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    lzo_uint out_sz = output_size;
    int ret = lzo1x_1_15_compress(
        reinterpret_cast<const lzo_bytep>(input), input_size,
        reinterpret_cast<lzo_bytep>(output), &out_sz, work_mem_);
    output_size = out_sz;
    return ret == 0;  // LZO returns 0 (LZO_E_OK) on success
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    lzo_uint out_sz = output_size;
    int ret = lzo1x_decompress(reinterpret_cast<const lzo_bytep>(input),
                               input_size, reinterpret_cast<lzo_bytep>(output),
                               &out_sz, nullptr);
    output_size = out_sz;
    return ret == 0;  // LZO returns 0 (LZO_E_OK) on success
  }
};

}  // namespace hshm

#endif  // HSHM_ENABLE_COMPRESS

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_Lzo_H_
