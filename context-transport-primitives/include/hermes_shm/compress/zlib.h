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

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_Zlib_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_Zlib_H_

#if HSHM_ENABLE_COMPRESS

#include <zlib.h>

#include "compress.h"

namespace hshm {

class Zlib : public Compressor {
 public:
  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    z_stream stream;
    stream.zalloc = Z_NULL;
    stream.zfree = Z_NULL;
    stream.opaque = Z_NULL;

    if (deflateInit(&stream, Z_DEFAULT_COMPRESSION) != Z_OK) {
      HLOG(kError, "Error initializing zlib compression.");
      return false;
    }

    stream.avail_in = input_size;
    stream.next_in = reinterpret_cast<Bytef *>(reinterpret_cast<char *>(input));

    stream.avail_out = output_size;
    stream.next_out = reinterpret_cast<Bytef *>(output);

    if (deflate(&stream, Z_FINISH) != Z_STREAM_END) {
      std::cerr << "Error compressing data with zlib." << std::endl;
      deflateEnd(&stream);
      return false;
    }

    output_size = stream.total_out;
    deflateEnd(&stream);
    return true;
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    z_stream stream;
    stream.zalloc = Z_NULL;
    stream.zfree = Z_NULL;
    stream.opaque = Z_NULL;

    if (inflateInit(&stream) != Z_OK) {
      HLOG(kError, "Error initializing zlib decompression.");
      return false;
    }

    stream.avail_in = input_size;
    stream.next_in = reinterpret_cast<Bytef *>(input);

    stream.avail_out = output_size;
    stream.next_out = reinterpret_cast<Bytef *>(output);

    if (inflate(&stream, Z_FINISH) != Z_STREAM_END) {
      HLOG(kError, "Error decompressing data with zlib.");
      inflateEnd(&stream);
      return false;
    }

    output_size = stream.total_out;
    inflateEnd(&stream);
    return true;
  }
};

}  // namespace hshm

#endif  // HSHM_ENABLE_COMPRESS

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_Zlib_H_
