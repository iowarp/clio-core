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

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_Lz4_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_Lz4_H_

#if HSHM_ENABLE_COMPRESS

#include <lz4.h>

#include "compress.h"

namespace hshm {

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
    output_size = LZ4_decompress_safe((char *)input, (char *)output,
                                      (int)input_size, (int)output_size);
    return output_size != 0;
  }
};

}  // namespace hshm

#endif  // HSHM_ENABLE_COMPRESS

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_Lz4_H_
