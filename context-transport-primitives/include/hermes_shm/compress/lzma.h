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

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_Lzma_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_Lzma_H_

#if HSHM_ENABLE_COMPRESS

#include <lzma.h>

#include "compress.h"

namespace hshm {

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

}  // namespace hshm

#endif  // HSHM_ENABLE_COMPRESS

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_Lzma_H_
