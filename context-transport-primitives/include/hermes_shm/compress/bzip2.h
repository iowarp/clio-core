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

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_BZIP2_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_BZIP2_H_

#if HSHM_ENABLE_COMPRESS

#include <bzlib.h>

#include "compress.h"

namespace hshm {

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

}  // namespace hshm

#endif  // HSHM_ENABLE_COMPRESS

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_BZIP2_H_
