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
    snappy::RawCompress((char *)input, input_size, (char *)output,
                        &output_size);
    bool ret = snappy::IsValidCompressedBuffer((char *)output, output_size);
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
