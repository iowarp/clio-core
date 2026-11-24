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

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_Snappy_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_Snappy_H_

// Optional codec: compiled out when the library was not found at
// configure time. CompressionFactory returns nullptr for it instead,
// which is the same contract the optional GPU codecs already use.
#if CTP_ENABLE_COMPRESS && CTP_ENABLE_SNAPPY

#include <snappy-sinksource.h>
#include <snappy.h>

#include "compress.h"

namespace ctp {

class Snappy : public Compressor {
 public:
  /**
   * Compress input into output.
   *
   * RawCompress writes up to MaxCompressedLength(input_size) bytes whatever
   * the buffer size, so a smaller output buffer is refused up front instead
   * of overrun. The old post-compress IsValidCompressedBuffer pass decoded
   * the whole output again (~60% extra compress time, #1214) and could not
   * fail: RawCompress always emits a valid stream.
   * @param output destination, output_size bytes
   * @param output_size in: capacity; out: compressed size
   * @param input source bytes
   * @param input_size number of source bytes
   * @return false when output is too small
   */
  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    if (output_size < snappy::MaxCompressedLength(input_size)) return false;
    ::size_t out_sz = 0;
    snappy::RawCompress(static_cast<const char *>(input), input_size,
                        static_cast<char *>(output), &out_sz);
    output_size = out_sz;
    return true;
  }

  /**
   * Decompress input into output.
   *
   * RawUncompress validates while decoding and is faster than the streaming
   * UncompressAsMuchAsPossible path used before (#1214). That path also
   * wrote through an unchecked sink, so a stream larger than output_size
   * overran the buffer; the decoded length is now checked first.
   * @param output destination, output_size bytes
   * @param output_size in: capacity; out: decompressed size
   * @param input compressed bytes
   * @param input_size number of compressed bytes
   * @return false for an invalid stream or one larger than output
   */
  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    const char *in = static_cast<const char *>(input);
    ::size_t raw_len = 0;
    if (!snappy::GetUncompressedLength(in, input_size, &raw_len) ||
        raw_len > output_size) {
      return false;
    }
    if (!snappy::RawUncompress(in, input_size, static_cast<char *>(output))) {
      return false;
    }
    output_size = raw_len;
    return raw_len != 0;
  }
};

}  // namespace ctp

#endif  // CTP_ENABLE_COMPRESS && CTP_ENABLE_SNAPPY

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_Snappy_H_
