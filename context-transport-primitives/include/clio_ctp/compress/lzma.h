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

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_Lzma_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_Lzma_H_

// Optional codec: see the note in snappy.h.
#if CTP_ENABLE_COMPRESS && CTP_ENABLE_LZMA

#include <lzma.h>

#include "compress.h"

namespace ctp {

class Lzma : public Compressor {
 public:
  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    lzma_stream strm = LZMA_STREAM_INIT;
    lzma_ret ret;

    // Initialize the LZMA encoder with preset LZMA_PRESET_DEFAULT
    ret = lzma_easy_encoder(&strm, LZMA_PRESET_DEFAULT, LZMA_CHECK_CRC64);
    if (ret != LZMA_OK) {
      HLOG(kError, "Error initializing LZMA compression.");
      return false;
    }

    // Set input buffer and size
    strm.next_in = reinterpret_cast<const uint8_t *>(input);
    strm.avail_in = input_size;

    // Set output buffer and size
    strm.next_out = reinterpret_cast<uint8_t *>(output);
    strm.avail_out = output_size;

    // Compress the data
    ret = lzma_code(&strm, LZMA_FINISH);
    if (ret != LZMA_OK && ret != LZMA_STREAM_END) {
      HLOG(kError, "Error compressing data with LZMA.");
      lzma_end(&strm);
      return false;
    }

    output_size -= strm.avail_out;

    // Finish compression
    lzma_end(&strm);
    return true;
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    lzma_stream strm = LZMA_STREAM_INIT;
    lzma_ret ret;

    // Initialize the LZMA decoder
    ret = lzma_stream_decoder(&strm, UINT64_MAX, LZMA_CONCATENATED);
    if (ret != LZMA_OK) {
      HLOG(kInfo, "Error initializing LZMA decompression.");
      return false;
    }

    // Set input buffer and size
    strm.next_in = reinterpret_cast<const uint8_t *>(input);
    strm.avail_in = input_size;

    // Set output buffer and size
    strm.next_out = reinterpret_cast<uint8_t *>(output);
    strm.avail_out = output_size;

    // Decompress the data
    ret = lzma_code(&strm, LZMA_FINISH);
    if (ret != LZMA_OK && ret != LZMA_STREAM_END) {
      HLOG(kError, "Error decompressing data with LZMA.");
      lzma_end(&strm);
      return false;
    }

    output_size -= strm.avail_out;

    // Finish decompression
    lzma_end(&strm);
    return true;
  }
};

}  // namespace ctp

#endif  // CTP_ENABLE_COMPRESS && CTP_ENABLE_LZMA

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_Lzma_H_
