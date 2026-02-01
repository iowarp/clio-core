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

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_Zstd_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_Zstd_H_

#if HSHM_ENABLE_COMPRESS

#include <zstd.h>

#include "compress.h"

namespace hshm {

class Zstd : public Compressor {
 public:
  Zstd() = default;

  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    if (ZSTD_compressBound(input_size) > output_size) {
      HLOG(kInfo, "Output buffer is potentially too small for compression");
    }
    output_size =
        ZSTD_compress(output, output_size, input, input_size, ZSTD_maxCLevel());
    return output_size != 0;
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    output_size = ZSTD_decompress(output, output_size, input, input_size);
    return output_size != 0;
  }
};

}  // namespace hshm

#endif  // HSHM_ENABLE_COMPRESS

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_Zstd_H_
