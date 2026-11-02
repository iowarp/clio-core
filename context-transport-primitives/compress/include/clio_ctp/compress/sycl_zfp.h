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

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_SYCL_ZFP_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_SYCL_ZFP_H_

#if CTP_ENABLE_COMPRESS && CTP_ENABLE_ZFP_SYCL

#include <zfp.h>

#include <cstdint>
#include <cstring>

#include "compress.h"

namespace ctp {

/**
 * GPU (SYCL) zfp compressor -- the Intel-XPU analog of NvComp, wrapping zfp's
 * SYCL execution policy the way NvComp wraps nvcomp.
 *
 * The SYCL acceleration lives entirely inside libzfp (its kernels are compiled
 * by icpx); this wrapper only calls zfp's plain C API and selects
 * zfp_exec_sycl, so it compiles with the ordinary CTP toolchain and merely
 * *links* a SYCL-enabled libzfp. zfp's SYCL backend itself stages host<->device
 * transfers internally, so host buffers may be passed directly.
 *
 * IMPORTANT -- LOSSY, FIXED-RATE ONLY. zfp's GPU backends (CUDA and SYCL alike)
 * implement only the fixed-rate mode; reversible (lossless), fixed-precision and
 * fixed-accuracy all fail (zfp_compress returns 0) on the SYCL execution policy.
 * So this compressor is lossy: each value is encoded in a fixed number of bits
 * (the "rate"). Higher rate -> better fidelity, lower compression.
 *
 * Like the existing LibPressio-backed zfp/sz3/fpzip entries, and constrained by
 * the byte-stream Compressor interface (which carries no type or shape), the
 * input buffer is interpreted as a 1D array of 32-bit floats. Callers passing
 * other types get the same caveat as those entries. Compress requires
 * input_size to be a multiple of sizeof(float).
 *
 * Compress prepends a small self-describing prefix (magic + element count +
 * rate) ahead of the codestream, so Decompress reconstructs the field and rate
 * from the bytes -- correct regardless of the rate this object was built with.
 * (zfp fixed-rate codestreams carry no metadata, which would make a multi-rate
 * codec silently mis-decode if the rate didn't match. zfp's own
 * zfp_write_header/zfp_read_header are NOT usable here: the SYCL backend's
 * zfp_compress overwrites a pre-written header, so we carry our own prefix.)
 */
class SyclZfp : public Compressor {
 public:
  /** @param rate fixed-rate bits per value (per 32-bit float). */
  explicit SyclZfp(double rate = 16.0) : rate_(rate) {}

  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    if (input == nullptr || (input_size % sizeof(float)) != 0) {
      return false;  // fixed-rate float codec; need whole 32-bit floats
    }
    if (output_size < sizeof(Prefix)) {
      return false;
    }
    const size_t n = input_size / sizeof(float);
    uint8_t *out = static_cast<uint8_t *>(output);

    zfp_field *field = zfp_field_1d(input, zfp_type_float, n);
    zfp_stream *zfp = zfp_stream_open(nullptr);
    bool ok = false;
    if (field != nullptr && zfp != nullptr) {
      zfp_stream_set_rate(zfp, rate_, zfp_type_float, 1, zfp_false);
      // set_execution does not validate the mode; a false return means the
      // SYCL backend isn't compiled into libzfp at all.
      if (zfp_stream_set_execution(zfp, zfp_exec_sycl)) {
        // Codestream goes after our prefix (kPrefixBytes is 8-aligned, so the
        // zfp buffer stays aligned given an aligned output).
        const size_t cap = output_size - kPrefixBytes;
        if (zfp_stream_maximum_size(zfp, field) <= cap) {
          bitstream *stream = stream_open(out + kPrefixBytes, cap);
          if (stream != nullptr) {
            zfp_stream_set_bit_stream(zfp, stream);
            zfp_stream_rewind(zfp);
            const size_t zsize = zfp_compress(zfp, field);  // 0 => failure
            if (zsize != 0) {
              const Prefix p{kMagic, 0u, static_cast<uint64_t>(n), rate_};
              std::memcpy(out, &p, sizeof(Prefix));
              output_size = kPrefixBytes + zsize;
              ok = true;
            }
            stream_close(stream);
          }
        }
      }
    }
    if (zfp != nullptr) zfp_stream_close(zfp);
    if (field != nullptr) zfp_field_free(field);
    return ok;
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    if (output == nullptr || input == nullptr || input_size < sizeof(Prefix)) {
      return false;
    }
    Prefix p;
    std::memcpy(&p, input, sizeof(Prefix));
    if (p.magic != kMagic) {
      return false;  // not one of our blobs
    }
    const size_t n = static_cast<size_t>(p.elems);
    if (n * sizeof(float) > output_size) {
      return false;  // caller buffer too small
    }
    uint8_t *in = static_cast<uint8_t *>(input);

    zfp_field *field = zfp_field_1d(output, zfp_type_float, n);
    zfp_stream *zfp = zfp_stream_open(nullptr);
    bool ok = false;
    if (field != nullptr && zfp != nullptr) {
      // Rate comes from the prefix, so decode is correct even if this object
      // was constructed with a different preset than compression used.
      zfp_stream_set_rate(zfp, p.rate, zfp_type_float, 1, zfp_false);
      if (zfp_stream_set_execution(zfp, zfp_exec_sycl)) {
        bitstream *stream =
            stream_open(in + kPrefixBytes, input_size - kPrefixBytes);
        if (stream != nullptr) {
          zfp_stream_set_bit_stream(zfp, stream);
          zfp_stream_rewind(zfp);
          if (zfp_decompress(zfp, field) != 0) {
            output_size = n * sizeof(float);
            ok = true;
          }
          stream_close(stream);
        }
      }
    }
    if (zfp != nullptr) zfp_stream_close(zfp);
    if (field != nullptr) zfp_field_free(field);
    return ok;
  }

 private:
  // Self-describing prefix carried ahead of the zfp codestream. 24 bytes,
  // 8-aligned so the codestream that follows stays 8-aligned for zfp.
  static constexpr uint32_t kMagic = 0x5A465053u;  // "ZFPS"
  struct Prefix {
    uint32_t magic;
    uint32_t pad;     // keep `rate` 8-aligned within the struct
    uint64_t elems;   // float element count
    double rate;      // fixed-rate bits per value used at compression
  };
  static constexpr size_t kPrefixBytes = sizeof(Prefix);  // 24

  double rate_;  // fixed-rate bits per 32-bit float value
};

}  // namespace ctp

#endif  // CTP_ENABLE_COMPRESS && CTP_ENABLE_ZFP_SYCL

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_SYCL_ZFP_H_
