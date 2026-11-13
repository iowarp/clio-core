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

/**
 * ctp::GpuLz4: the gpu_lz4 page format (gpu_lz4.h) as an ordinary host
 * Compressor, registered in CompressionFactory as "gpu-lz4" (wire id 25).
 *
 * This is the HOST half: the factory entry exists so the id is reserved,
 * named, and usable by any tool that round-trips a buffer through the
 * registry. Device decoding is the LZ4 ChiMod's (context-transfer-engine/lz4),
 * which launches the shared decoder straight into GPU memory. The registry's
 * "executes on device" flag is therefore false: the nvCOMP-centric compressor
 * module must never send this format down its batched nvCOMP path.
 */

#ifndef CLIO_CTP_COMPRESS_GPU_LZ4_COMPRESSOR_H_
#define CLIO_CTP_COMPRESS_GPU_LZ4_COMPRESSOR_H_

#include <cstring>
#include <vector>

#include "compress.h"
#include "gpu_ans.h"

namespace ctp {

/** Host compressor for the gpu_lz4 page format (no external dependency). */
class GpuLz4 : public Compressor {
 public:
  /**
   * Encode `input` as one gpu_lz4 page.
   * @param output       destination buffer
   * @param output_size  IN: capacity; OUT: encoded bytes
   * @return false when the encoded page does not fit in the capacity
   */
  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    if (input_size == 0) return false;
    std::vector<uint8_t> enc;
    gpu_lz4::EncodePage(input, input_size, enc);
    if (enc.size() > output_size) return false;
    std::memcpy(output, enc.data(), enc.size());
    output_size = enc.size();
    return true;
  }

  /**
   * Decode one gpu_lz4 page.
   * @param output       destination buffer
   * @param output_size  IN: capacity; OUT: decoded bytes
   * @return false when the input is not a valid page or does not fit
   */
  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    const uint64_t raw =
        gpu_lz4::RawSizeOf(static_cast<const uint8_t *>(input), input_size);
    if (raw == 0 || raw > output_size) return false;
    if (!gpu_lz4::DecodePageAny(input, input_size, output, output_size)) {
      return false;
    }
    output_size = static_cast<size_t>(raw);
    return true;
  }
};

/**
 * Host compressor for the gpu_ans page format (interleaved rANS, gpu_ans.h):
 * registered as "gpu-ans" (wire id 26). Same split as GpuLz4: this is the
 * host half; the LZ4 ChiMod decodes it on the GPU.
 */
class GpuAns : public GpuLz4 {
 public:
  /** Encode `input` as one gpu_ans page (see GpuLz4::Compress). */
  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    if (input_size == 0) return false;
    std::vector<uint8_t> enc;
    gpu_lz4::EncodePageAns(input, input_size, enc);
    if (enc.size() > output_size) return false;
    std::memcpy(output, enc.data(), enc.size());
    output_size = enc.size();
    return true;
  }
  // Decompress is GpuLz4's: DecodePageAny dispatches on the page's magic.
};

}  // namespace ctp

#endif  // CLIO_CTP_COMPRESS_GPU_LZ4_COMPRESSOR_H_
