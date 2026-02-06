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
    int ret = lzo1x_1_15_compress(
        reinterpret_cast<const lzo_bytep>(input), input_size,
        reinterpret_cast<lzo_bytep>(output), &output_size, work_mem_);
    return ret == 0;  // LZO returns 0 (LZO_E_OK) on success
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    int ret = lzo1x_decompress(reinterpret_cast<const lzo_bytep>(input),
                               input_size, reinterpret_cast<lzo_bytep>(output),
                               &output_size, nullptr);
    return ret == 0;  // LZO returns 0 (LZO_E_OK) on success
  }
};

}  // namespace hshm

#endif  // HSHM_ENABLE_COMPRESS

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_Lzo_H_
