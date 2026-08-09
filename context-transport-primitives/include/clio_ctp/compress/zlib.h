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

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_Zlib_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_Zlib_H_

// Optional codec: see the note in snappy.h.
#if CTP_ENABLE_COMPRESS && CTP_ENABLE_ZLIB

#include <zlib.h>

#include "compress.h"

namespace ctp {

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

}  // namespace ctp

#endif  // CTP_ENABLE_COMPRESS && CTP_ENABLE_ZLIB

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_Zlib_H_
